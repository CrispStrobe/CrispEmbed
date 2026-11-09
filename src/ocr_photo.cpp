// Explicit, length-checked photo-editor operations. No inference or global gates.
#include "crispembed.h"
#include "ocr_crop.h"
#include "scan_cleanup.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace {

constexpr size_t max_pixels = 64u * 1024u * 1024u;
constexpr int max_axis = 32768;

bool valid_input(const uint8_t * pixels, size_t length, int width, int height, int channels) {
    return pixels && width >= 3 && height >= 3 && width <= max_axis && height <= max_axis &&
           (channels == 1 || channels == 3 || channels == 4) && size_t(width) * height <= max_pixels &&
           length == size_t(width) * height * channels;
}

bool clear_output(uint8_t ** pixels, int * width, int * height) {
    if (pixels) *pixels = nullptr;
    if (width) *width = 0;
    if (height) *height = 0;
    return pixels && width && height;
}

using image_buffer = std::unique_ptr<uint8_t, decltype(&std::free)>;

} // namespace

extern "C" int crispembed_scan_photo_angle(const uint8_t * pixels, size_t pixels_len, int width, int height,
                                           int channels, float max_angle_deg, float * out_correction_degrees) {
    if (out_correction_degrees) *out_correction_degrees = 0.0f;
    if (!out_correction_degrees || !valid_input(pixels, pixels_len, width, height, channels) ||
        !std::isfinite(max_angle_deg) || max_angle_deg <= 0 || max_angle_deg > 20)
        return -1;
    try {
        const float detected = scan_cleanup_detect_angle_rgb_bounded(pixels, width, height, channels, max_angle_deg);
        if (!std::isfinite(detected) || std::abs(detected) > max_angle_deg) return -1;
        *out_correction_degrees = detected == 0.0f ? 0.0f : -detected;
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" int crispembed_scan_photo_rotate(const uint8_t * pixels, size_t pixels_len, int width, int height,
                                            int channels, float clockwise_degrees, uint8_t ** out_pixels,
                                            int * out_width, int * out_height) {
    if (!clear_output(out_pixels, out_width, out_height) || !valid_input(pixels, pixels_len, width, height, channels) ||
        !std::isfinite(clockwise_degrees) || std::abs(clockwise_degrees) > 20)
        return -1;
    try {
        if (clockwise_degrees == 0) {
            image_buffer copy(static_cast<uint8_t *>(std::malloc(pixels_len)), &std::free);
            if (!copy) return -1;
            std::memcpy(copy.get(), pixels, pixels_len);
            *out_width = width;
            *out_height = height;
            *out_pixels = copy.release();
            return 0;
        }
        // The image's y axis points down: a positive standard rotation matrix
        // therefore rotates clockwise. Sample inverse-mapped pixel centres.
        constexpr double pi = 3.14159265358979323846;
        const double radians = clockwise_degrees * pi / 180.0;
        const double cosine = std::cos(radians), sine = std::sin(radians);
        const double wd = std::ceil((width - 1) * std::abs(cosine) + (height - 1) * std::abs(sine)) + 1;
        const double hd = std::ceil((height - 1) * std::abs(cosine) + (width - 1) * std::abs(sine)) + 1;
        if (wd > max_axis || hd > max_axis || wd * hd > max_pixels) return -1;
        const int ow = int(wd), oh = int(hd);
        const size_t bytes = size_t(ow) * oh * channels;
        image_buffer output(static_cast<uint8_t *>(std::malloc(bytes)), &std::free);
        if (!output) return -1;
        std::memset(output.get(), 255, bytes);
        const double cx = (width - 1) * 0.5, cy = (height - 1) * 0.5;
        const double ox = (ow - 1) * 0.5, oy = (oh - 1) * 0.5;
        for (int y = 0; y < oh; ++y) {
            for (int x = 0; x < ow; ++x) {
                const double sx = cosine * (x - ox) + sine * (y - oy) + cx;
                const double sy = -sine * (x - ox) + cosine * (y - oy) + cy;
                if (sx < 0 || sy < 0 || sx > width - 1 || sy > height - 1) continue;
                const int x0 = int(sx), y0 = int(sy), x1 = std::min(x0 + 1, width - 1),
                          y1 = std::min(y0 + 1, height - 1);
                const double fx = sx - x0, fy = sy - y0;
                for (int c = 0; c < channels; ++c) {
                    const double top = pixels[(size_t(y0) * width + x0) * channels + c] * (1 - fx) +
                                       pixels[(size_t(y0) * width + x1) * channels + c] * fx;
                    const double bottom = pixels[(size_t(y1) * width + x0) * channels + c] * (1 - fx) +
                                          pixels[(size_t(y1) * width + x1) * channels + c] * fx;
                    output.get()[(size_t(y) * ow + x) * channels + c] =
                        uint8_t(std::lround(top * (1 - fy) + bottom * fy));
                }
            }
        }
        *out_width = ow;
        *out_height = oh;
        *out_pixels = output.release();
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" int crispembed_scan_photo_perspective(const uint8_t * pixels, size_t pixels_len, int width, int height,
                                                 int channels, const float corners_xy[8], uint8_t ** out_pixels,
                                                 int * out_width, int * out_height) {
    if (!clear_output(out_pixels, out_width, out_height) || !corners_xy ||
        !valid_input(pixels, pixels_len, width, height, channels))
        return -1;
    try {
        float x[4], y[4];
        for (int i = 0; i < 4; ++i) {
            x[i] = corners_xy[2 * i];
            y[i] = corners_xy[2 * i + 1];
        }
        int ow = 0, oh = 0;
        const auto image = ocr_crop::extract_quad_projective_ordered(pixels, width, height, channels, x, y, &ow, &oh);
        if (image.empty()) return -1;
        image_buffer output(static_cast<uint8_t *>(std::malloc(image.size())), &std::free);
        if (!output) return -1;
        std::memcpy(output.get(), image.data(), image.size());
        *out_width = ow;
        *out_height = oh;
        *out_pixels = output.release();
        return 0;
    } catch (...) {
        return -1;
    }
}

extern "C" void crispembed_scan_photo_free_image(uint8_t * pixels) {
    std::free(pixels);
}
