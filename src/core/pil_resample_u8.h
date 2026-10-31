// pil_resample_u8.h — Pillow-exact 8-bit separable resampling with a selectable
// filter (Pillow src/libImaging/Resample.c: precompute_coeffs,
// normalize_coeffs_8bpc, ImagingResampleHorizontal_8bpc/Vertical_8bpc).
//
// Adapted from image_preproc::resize_bicubic_pil_u8_hwc (image_preprocess.cpp),
// which established the three properties that make this bit-exact rather than
// off by one: clip8 to uint8 BETWEEN the passes, a per-output tap count, and
// tap-RANGE clamping with renormalisation at the borders. Coefficients are
// evaluated in double and quantised to 22-bit fixed point like Pillow. A pass
// is skipped when its axis keeps its size (Pillow's need_horizontal/vertical).
//
// Header-only so the standalone qwen35 OCR library can use it (Surya OCR 2.5's
// package scale_to_fit uses Image.Resampling.LANCZOS). Hermetic guard and Pillow
// parity: tests/test_pil_resample_u8.cpp + tests/test_surya25_tokenizer_parity.py.
#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

namespace core_pil {

enum class filter { bicubic, lanczos };

inline double bicubic_filter(double x) { // Pillow bicubic_filter, a = -0.5
    constexpr double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
    return 0.0;
}
inline double sinc_filter(double x) {
    if (x == 0.0) return 1.0;
    x = x * M_PI;
    return std::sin(x) / x;
}
inline double lanczos_filter(double x) { // Pillow: truncated sinc, support 3
    if (-3.0 <= x && x < 3.0) return sinc_filter(x) * sinc_filter(x / 3);
    return 0.0;
}

struct coeffs {
    int kmax = 0;
    std::vector<int> bmin, bcount;
    std::vector<int32_t> k;
};

inline void precompute(int in_size, int out_size, filter f, coeffs & c) {
    constexpr int precision_bits = 32 - 8 - 2;
    const double base = f == filter::lanczos ? 3.0 : 2.0;
    const double scale = double(in_size) / double(out_size);
    const double filterscale = scale < 1.0 ? 1.0 : scale;
    const double support = base * filterscale;
    const double ss = 1.0 / filterscale;
    c.kmax = int(std::ceil(support)) * 2 + 1;
    c.bmin.assign(size_t(out_size), 0);
    c.bcount.assign(size_t(out_size), 0);
    c.k.assign(size_t(out_size) * c.kmax, 0);
    std::vector<double> kd(size_t(c.kmax), 0.0);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = (xx + 0.5) * scale;
        int xmin = int(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = int(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        if (xmax < 0) xmax = 0;
        if (xmax > c.kmax) xmax = c.kmax;
        double ww = 0.0;
        for (int x = 0; x < xmax; ++x) {
            const double arg = (x + xmin - center + 0.5) * ss;
            const double w = f == filter::lanczos ? lanczos_filter(arg) : bicubic_filter(arg);
            kd[size_t(x)] = w;
            ww += w;
        }
        int32_t * k = c.k.data() + size_t(xx) * c.kmax;
        for (int x = 0; x < xmax; ++x) {
            const double v = ww != 0.0 ? kd[size_t(x)] / ww : kd[size_t(x)];
            k[x] = int32_t(v < 0.0 ? -0.5 + v * (1 << precision_bits) : 0.5 + v * (1 << precision_bits));
        }
        c.bmin[size_t(xx)] = xmin;
        c.bcount[size_t(xx)] = xmax;
    }
}

inline uint8_t clip8(int32_t acc) {
    const int32_t v = acc >> (32 - 8 - 2);
    return uint8_t(v <= 0 ? 0 : v >= 255 ? 255 : v);
}

// HWC uint8 -> HWC uint8, `channels` interleaved, Pillow pass order and skips.
inline bool resize_u8(const uint8_t * src, int sw, int sh, int channels, int dw, int dh, filter f,
                      std::vector<uint8_t> & out) {
    out.clear();
    if (!src || sw < 1 || sh < 1 || dw < 1 || dh < 1 || channels < 1 || channels > 4) return false;
    constexpr int32_t half = int32_t(1) << (32 - 8 - 2 - 1);
    std::vector<uint8_t> mid;
    const uint8_t * rows = src;
    int width = sw;
    if (dw != sw) {
        coeffs cx;
        precompute(sw, dw, f, cx);
        mid.assign(size_t(sh) * dw * channels, 0);
        for (int y = 0; y < sh; ++y)
            for (int xo = 0; xo < dw; ++xo) {
                const int32_t * k = cx.k.data() + size_t(xo) * cx.kmax;
                const int xmin = cx.bmin[size_t(xo)], n = cx.bcount[size_t(xo)];
                for (int c = 0; c < channels; ++c) {
                    int32_t acc = half;
                    const uint8_t * row = src + size_t(y) * sw * channels + c;
                    for (int x = 0; x < n; ++x) acc += k[x] * int32_t(row[size_t(xmin + x) * channels]);
                    mid[(size_t(y) * dw + xo) * channels + c] = clip8(acc);
                }
            }
        rows = mid.data();
        width = dw;
    }
    if (dh == sh) {
        out.assign(rows, rows + size_t(sh) * width * channels);
        return true;
    }
    coeffs cy;
    precompute(sh, dh, f, cy);
    out.assign(size_t(dh) * width * channels, 0);
    for (int yo = 0; yo < dh; ++yo) {
        const int32_t * k = cy.k.data() + size_t(yo) * cy.kmax;
        const int ymin = cy.bmin[size_t(yo)], n = cy.bcount[size_t(yo)];
        for (int x = 0; x < width; ++x)
            for (int c = 0; c < channels; ++c) {
                int32_t acc = half;
                for (int y = 0; y < n; ++y) acc += k[y] * int32_t(rows[(size_t(ymin + y) * width + x) * channels + c]);
                out[(size_t(yo) * width + x) * channels + c] = clip8(acc);
            }
    }
    return true;
}

} // namespace core_pil
