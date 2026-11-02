// cv_compat.cpp — see cv_compat.h. Portions translated from OpenCV
// (https://github.com/opencv/opencv, modules/imgproc/src/resize.cpp and
// color_rgb.simd.hpp), Apache License 2.0. Changes: single-threaded uint8
// only, std::vector buffers, explicit unsupported-path errors.
#include "cv_compat.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace cv_compat {
namespace {
struct decimate_alpha {
    int si, di;
    float alpha;
};

int cv_round(float v) {
    return (int)std::nearbyint(v); // cvRound: round half to even (lrintf)
}
uint8_t saturate_u8(float v) {
    const int i = cv_round(v);
    return (uint8_t)(i < 0 ? 0 : i > 255 ? 255 : i);
}

int compute_resize_area_tab(int ssize, int dsize, int cn, double scale, std::vector<decimate_alpha> & tab) {
    int k = 0;
    for (int dx = 0; dx < dsize; dx++) {
        double fsx1 = dx * scale;
        double fsx2 = fsx1 + scale;
        double cell_width = std::min(scale, ssize - fsx1);

        int sx1 = (int)std::ceil(fsx1), sx2 = (int)std::floor(fsx2);

        sx2 = std::min(sx2, ssize - 1);
        sx1 = std::min(sx1, sx2);

        if (sx1 - fsx1 > 1e-3) {
            tab[size_t(k)].di = dx * cn;
            tab[size_t(k)].si = (sx1 - 1) * cn;
            tab[size_t(k++)].alpha = (float)((sx1 - fsx1) / cell_width);
        }

        for (int sx = sx1; sx < sx2; sx++) {
            tab[size_t(k)].di = dx * cn;
            tab[size_t(k)].si = sx * cn;
            tab[size_t(k++)].alpha = float(1.0 / cell_width);
        }

        if (fsx2 - sx2 > 1e-3) {
            tab[size_t(k)].di = dx * cn;
            tab[size_t(k)].si = sx2 * cn;
            tab[size_t(k++)].alpha = (float)(std::min(std::min(fsx2 - sx2, 1.), cell_width) / cell_width);
        }
    }
    return k;
}
} // namespace

std::vector<uint8_t> resize_area_u8(const uint8_t * src, int width, int height, int cn, int dst_w, int dst_h) {
    if (!src || width <= 0 || height <= 0 || dst_w <= 0 || dst_h <= 0 || cn < 1 || cn > 4)
        throw std::invalid_argument("resize_area_u8: invalid geometry");
    if (dst_w == width && dst_h == height) return std::vector<uint8_t>(src, src + size_t(width) * height * cn);
    // cv::resize: inv_scale = dsize/ssize; hal::resize: scale = 1/inv_scale.
    const double inv_scale_x = (double)dst_w / width, inv_scale_y = (double)dst_h / height;
    const double scale_x = 1. / inv_scale_x, scale_y = 1. / inv_scale_y;
    if (!(scale_x >= 1 && scale_y >= 1))
        throw std::invalid_argument("resize_area_u8: INTER_AREA upscaling uses OpenCV's bilinear path (not ported)");
    const int iscale_x = (int)std::nearbyint(scale_x), iscale_y = (int)std::nearbyint(scale_y);
    if (std::abs(scale_x - iscale_x) < DBL_EPSILON && std::abs(scale_y - iscale_y) < DBL_EPSILON)
        throw std::invalid_argument("resize_area_u8: integer-ratio fast area path is not ported");

    std::vector<decimate_alpha> xtab(size_t(width) * 2 + 2), ytab(size_t(height) * 2 + 2);
    const int xtab_size = compute_resize_area_tab(width, dst_w, cn, scale_x, xtab);
    const int ytab_size = compute_resize_area_tab(height, dst_h, 1, scale_y, ytab);
    std::vector<int> tabofs(size_t(dst_h) + 1);
    int dy = 0;
    for (int k = 0; k < ytab_size; k++) {
        if (k == 0 || ytab[size_t(k)].di != ytab[size_t(k - 1)].di) tabofs[size_t(dy++)] = k;
    }
    tabofs[size_t(dy)] = ytab_size;

    std::vector<uint8_t> dst(size_t(dst_w) * dst_h * cn);
    const int dwidth = dst_w * cn;
    std::vector<float> buf(static_cast<size_t>(dwidth), 0.f);
    std::vector<float> sum(static_cast<size_t>(dwidth), 0.f);
    const int j_start = tabofs[0], j_end = tabofs[size_t(dst_h)];
    int prev_dy = ytab[size_t(j_start)].di;
    for (int j = j_start; j < j_end; j++) {
        const float beta = ytab[size_t(j)].alpha;
        const int cur_dy = ytab[size_t(j)].di;
        const int sy = ytab[size_t(j)].si;
        const uint8_t * S = src + size_t(sy) * width * cn;
        std::fill(buf.begin(), buf.end(), 0.f);
        if (cn == 1) {
            for (int k = 0; k < xtab_size; k++) {
                const int dxn = xtab[size_t(k)].di;
                const float alpha = xtab[size_t(k)].alpha;
                const float prod = S[xtab[size_t(k)].si] * alpha;
                buf[size_t(dxn)] += prod;
            }
        } else {
            for (int k = 0; k < xtab_size; k++) {
                const int sxn = xtab[size_t(k)].si;
                const int dxn = xtab[size_t(k)].di;
                const float alpha = xtab[size_t(k)].alpha;
                for (int c = 0; c < cn; ++c) {
                    const float prod = S[sxn + c] * alpha;
                    buf[size_t(dxn + c)] = buf[size_t(dxn + c)] + prod;
                }
            }
        }
        if (cur_dy != prev_dy) {
            uint8_t * D = dst.data() + size_t(prev_dy) * dwidth;
            for (int dx = 0; dx < dwidth; dx++) {
                D[dx] = saturate_u8(sum[size_t(dx)]);
                sum[size_t(dx)] = beta * buf[size_t(dx)];
            }
            prev_dy = cur_dy;
        } else {
            for (int dx = 0; dx < dwidth; dx++) {
                const float prod = beta * buf[size_t(dx)];
                sum[size_t(dx)] += prod;
            }
        }
    }
    uint8_t * D = dst.data() + size_t(prev_dy) * dwidth;
    for (int dx = 0; dx < dwidth; dx++) D[dx] = saturate_u8(sum[size_t(dx)]);
    return dst;
}

std::vector<uint8_t> rgb_to_gray_u8(const uint8_t * rgb, int width, int height) {
    std::vector<uint8_t> out(size_t(width) * height);
    for (size_t i = 0; i < out.size(); ++i) {
        const uint8_t * p = rgb + 3 * i;
        out[i] = uint8_t((p[0] * 9798 + p[1] * 19235 + p[2] * 3735 + (1 << 14)) >> 15);
    }
    return out;
}

} // namespace cv_compat
