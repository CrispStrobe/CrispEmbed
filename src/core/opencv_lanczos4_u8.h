// opencv_lanczos4_u8.h — bit-exact port of OpenCV's generic cv::resize
// INTER_LANCZOS4 path for 8-bit images (modules/imgproc/src/resize.cpp,
// OpenCV 4.11: interpolateLanczos4 + resizeGeneric_ with HResizeLanczos4 /
// VResizeLanczos4<uchar,int,short,FixedPtCast<int,uchar,22>>).
//
// The generic kernel uses 11-bit fixed-point coefficients and exact integer
// accumulation, so results do not depend on SIMD or threading. They do differ
// from OpenCV builds that dispatch INTER_LANCZOS4 to Intel IPP (x86 wheels with
// cv::ipp::useIPP() enabled); see docs/oxr-dfine.md.
//
// Header-only, no dependencies; interleaved channels (1..4).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace core_cv {

inline void opencv_interpolate_lanczos4(float x, float * coeffs) {
    static const double s45 = 0.70710678118654752440084436210485;
    static const double cs[][2] = { { 1, 0 },  { -s45, -s45 }, { 0, 1 },  { s45, -s45 },
                                    { -1, 0 }, { s45, s45 },   { 0, -1 }, { -s45, s45 } };
    const double cv_pi = 3.1415926535897932384626433832795;
    float sum = 0;
    double y0 = -(x + 3) * cv_pi * 0.25, s0 = std::sin(y0), c0 = std::cos(y0);
    for (int i = 0; i < 8; i++) {
        float y0_ = (x + 3 - i);
        if (std::fabs(y0_) >= 1e-6f) {
            double y = -y0_ * cv_pi * 0.25;
            coeffs[i] = (float)((cs[i][0] * s0 + cs[i][1] * c0) / (y * y));
        } else {
            coeffs[i] = 1e30f;
        }
        sum += coeffs[i];
    }
    sum = 1.f / sum;
    for (int i = 0; i < 8; i++) coeffs[i] *= sum;
}

// saturate_cast<short>(float): cvRound (round half to even) then clamp.
inline short opencv_saturate_short(float v) {
    const long r = std::lrint(v);
    return (short)std::min<long>(32767, std::max<long>(-32768, r));
}

// Resize src (sw x sh, cn interleaved uint8 channels, tightly packed) to
// dst (dw x dh). Identical sizes are copied, as cv::resize does.
inline void opencv_resize_lanczos4_u8(const uint8_t * src, int sw, int sh, int cn, uint8_t * dst, int dw, int dh) {
    if (sw == dw && sh == dh) {
        std::copy(src, src + (size_t)sw * sh * cn, dst);
        return;
    }
    const int ksize = 8, ksize2 = 4, coef_scale = 1 << 11;
    const double inv_scale_x = (double)dw / sw, inv_scale_y = (double)dh / sh;
    const double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;
    const int width = dw * cn;
    std::vector<int> xofs(width), yofs(dh);
    std::vector<short> ialpha((size_t)width * ksize), ibeta((size_t)dh * ksize);
    float cbuf[8];
    for (int dx = 0; dx < dw; dx++) {
        float fx = (float)((dx + 0.5) * scale_x - 0.5);
        int sx = (int)std::floor(fx);
        fx -= sx;
        for (int k = 0; k < cn; k++) xofs[dx * cn + k] = sx * cn + k;
        opencv_interpolate_lanczos4(fx, cbuf);
        for (int k = 0; k < ksize; k++)
            ialpha[(size_t)dx * cn * ksize + k] = opencv_saturate_short(cbuf[k] * coef_scale);
        for (int k = ksize; k < cn * ksize; k++)
            ialpha[(size_t)dx * cn * ksize + k] = ialpha[(size_t)dx * cn * ksize + k - ksize];
    }
    for (int dy = 0; dy < dh; dy++) {
        float fy = (float)((dy + 0.5) * scale_y - 0.5);
        int sy = (int)std::floor(fy);
        fy -= sy;
        yofs[dy] = sy;
        opencv_interpolate_lanczos4(fy, cbuf);
        for (int k = 0; k < ksize; k++) ibeta[(size_t)dy * ksize + k] = opencv_saturate_short(cbuf[k] * coef_scale);
    }
    const int swidth = sw * cn;
    // Horizontal pass, per source row, cached (integer arithmetic: exact).
    std::vector<int> hrow((size_t)sh * width);
    std::vector<uint8_t> hdone(sh, 0);
    auto hresize = [&](int sy) {
        if (hdone[sy]) return;
        hdone[sy] = 1;
        const uint8_t * S = src + (size_t)sy * swidth;
        int * D = hrow.data() + (size_t)sy * width;
        for (int x = 0; x < width; x++) {
            const short * alpha = ialpha.data() + (size_t)x * ksize;
            const int sx = xofs[x] - cn * 3;
            int v = 0;
            for (int j = 0; j < 8; j++) {
                int sxj = sx + j * cn;
                if ((unsigned)sxj >= (unsigned)swidth) {
                    while (sxj < 0) sxj += cn;
                    while (sxj >= swidth) sxj -= cn;
                }
                v += S[sxj] * alpha[j];
            }
            D[x] = v;
        }
    };
    const int * rows[8];
    for (int dy = 0; dy < dh; dy++) {
        const int sy0 = yofs[dy];
        for (int k = 0; k < ksize; k++) {
            int sy = sy0 - ksize2 + 1 + k;
            sy = sy >= 0 ? (sy < sh ? sy : sh - 1) : 0;
            hresize(sy);
            rows[k] = hrow.data() + (size_t)sy * width;
        }
        const short * beta = ibeta.data() + (size_t)dy * ksize;
        uint8_t * out = dst + (size_t)dy * width;
        for (int x = 0; x < width; x++) {
            int s = rows[0][x] * beta[0] + rows[1][x] * beta[1] + rows[2][x] * beta[2] + rows[3][x] * beta[3] +
                    rows[4][x] * beta[4] + rows[5][x] * beta[5] + rows[6][x] * beta[6] + rows[7][x] * beta[7];
            int r = (s + (1 << 21)) >> 22;
            out[x] = (uint8_t)std::min(255, std::max(0, r));
        }
    }
}

} // namespace core_cv
