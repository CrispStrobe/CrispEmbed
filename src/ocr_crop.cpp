#include "ocr_crop.h"
#include "classical_preproc.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace ocr_crop {

orientation_info orient_180_rgb_info(std::vector<uint8_t> & pixels, int width, int height) {
    orientation_info info;
    if (width < 8 || height < 8 || pixels.size() != (size_t)width * height * 3) return info;
    std::vector<uint8_t> gray((size_t)width * height);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            const uint8_t * p = pixels.data() + ((size_t)y * width + x) * 3;
            gray[(size_t)y * width + x] = (uint8_t)((77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8);
        }
    }
    float confidence = 0.0f;
    info.angle = detect_text_angle(gray.data(), width, height, &confidence);
    info.confidence = confidence;
    if (info.angle != 180 || confidence < 0.75f) return info;
    for (int y = 0; y < (height + 1) / 2; y++) {
        for (int x = 0; x < width; x++) {
            const size_t a = ((size_t)y * width + x) * 3;
            const size_t b = ((size_t)(height - 1 - y) * width + (width - 1 - x)) * 3;
            if (a >= b) continue;
            for (int c = 0; c < 3; c++) std::swap(pixels[a + c], pixels[b + c]);
        }
    }
    info.corrected = true;
    return info;
}

orientation_info orient_180_gray_info(std::vector<uint8_t> & pixels, int width, int height) {
    orientation_info info;
    if (width < 8 || height < 8 || pixels.size() != (size_t)width * height) return info;
    float confidence = 0.0f;
    info.angle = detect_text_angle(pixels.data(), width, height, &confidence);
    info.confidence = confidence;
    if (info.angle != 180 || confidence < 0.75f) return info;
    for (int y = 0; y < (height + 1) / 2; y++) {
        for (int x = 0; x < width; x++) {
            const size_t a = (size_t)y * width + x;
            const size_t b = (size_t)(height - 1 - y) * width + (width - 1 - x);
            if (a < b) std::swap(pixels[a], pixels[b]);
        }
    }
    info.corrected = true;
    return info;
}

bool orient_180_rgb(std::vector<uint8_t> & pixels, int width, int height) {
    return orient_180_rgb_info(pixels, width, height).corrected;
}

bool orient_180_gray(std::vector<uint8_t> & pixels, int width, int height) {
    return orient_180_gray_info(pixels, width, height).corrected;
}

std::vector<uint8_t> extract(const uint8_t * pixels, int width, int height, int channels, int x, int y, int crop_w,
                             int crop_h, int padding, int * out_width, int * out_height) {
    if (out_width) *out_width = 0;
    if (out_height) *out_height = 0;
    if (!pixels || width <= 0 || height <= 0 || channels <= 0 || crop_w <= 0 || crop_h <= 0) return {};

    const int x0 = std::max(0, x - std::max(0, padding));
    const int y0 = std::max(0, y - std::max(0, padding));
    const int x1 = std::min(width, x + crop_w + std::max(0, padding));
    const int y1 = std::min(height, y + crop_h + std::max(0, padding));
    const int w = x1 - x0;
    const int h = y1 - y0;
    if (w <= 0 || h <= 0) return {};

    std::vector<uint8_t> result((size_t)w * h * channels);
    for (int row = 0; row < h; row++) {
        const uint8_t * source = pixels + ((size_t)(y0 + row) * width + x0) * channels;
        std::memcpy(result.data() + (size_t)row * w * channels, source, (size_t)w * channels);
    }
    if (out_width) *out_width = w;
    if (out_height) *out_height = h;
    return result;
}

std::vector<uint8_t> extract_quad(const uint8_t * pixels, int width, int height, int channels, const float qx[4],
                                  const float qy[4], int padding, int * out_width, int * out_height) {
    if (out_width) *out_width = 0;
    if (out_height) *out_height = 0;
    if (!pixels || width <= 0 || height <= 0 || channels <= 0 || !qx || !qy) return {};
    struct point {
        float x, y;
    };
    std::array<point, 4> p{}, o{};
    for (int i = 0; i < 4; ++i) p[i] = { qx[i], qy[i] };
    float min_sum = INFINITY, max_sum = -INFINITY, min_diff = INFINITY, max_diff = -INFINITY;
    for (const auto & v : p) {
        const float sum = v.x + v.y, diff = v.x - v.y;
        if (sum < min_sum) {
            min_sum = sum;
            o[0] = v;
        }
        if (sum > max_sum) {
            max_sum = sum;
            o[2] = v;
        }
        if (diff < min_diff) {
            min_diff = diff;
            o[1] = v;
        }
        if (diff > max_diff) {
            max_diff = diff;
            o[3] = v;
        }
    }
    auto distance = [](point a, point b) { return std::hypot(a.x - b.x, a.y - b.y); };
    const int pad = std::max(0, padding);
    const int ow = std::max(1, (int)std::lround(std::max(distance(o[0], o[1]), distance(o[3], o[2]))) + 2 * pad);
    const int oh = std::max(1, (int)std::lround(std::max(distance(o[0], o[3]), distance(o[1], o[2]))) + 2 * pad);
    std::vector<uint8_t> result((size_t)ow * oh * channels);
    auto sample = [&](float x, float y, int c) -> uint8_t {
        x = std::clamp(x, 0.0f, (float)(width - 1));
        y = std::clamp(y, 0.0f, (float)(height - 1));
        const int x0 = (int)std::floor(x), y0 = (int)std::floor(y);
        const int x1 = std::min(width - 1, x0 + 1), y1 = std::min(height - 1, y0 + 1);
        const float fx = x - x0, fy = y - y0;
        auto at = [&](int xx, int yy) { return pixels[((size_t)yy * width + xx) * channels + c]; };
        const float a = at(x0, y0) * (1 - fx) + at(x1, y0) * fx;
        const float b = at(x0, y1) * (1 - fx) + at(x1, y1) * fx;
        return (uint8_t)std::lround(a * (1 - fy) + b * fy);
    };
    for (int y = 0; y < oh; ++y) {
        const float v = std::clamp((y - pad) / (float)std::max(1, oh - 1 - 2 * pad), 0.0f, 1.0f);
        for (int x = 0; x < ow; ++x) {
            const float u = std::clamp((x - pad) / (float)std::max(1, ow - 1 - 2 * pad), 0.0f, 1.0f);
            const float tx = o[0].x + u * (o[1].x - o[0].x), ty = o[0].y + u * (o[1].y - o[0].y);
            const float bx = o[3].x + u * (o[2].x - o[3].x), by = o[3].y + u * (o[2].y - o[3].y);
            for (int c = 0; c < channels; ++c)
                result[((size_t)y * ow + x) * channels + c] = sample(tx + v * (bx - tx), ty + v * (by - ty), c);
        }
    }
    if (out_width) *out_width = ow;
    if (out_height) *out_height = oh;
    return result;
}

std::vector<uint8_t> prepare(const uint8_t * pixels, int width, int height, int channels,
                             const prepare_options & options, int * out_width, int * out_height, int * out_channels) {
    if (out_width) *out_width = 0;
    if (out_height) *out_height = 0;
    if (out_channels) *out_channels = 0;
    if (!pixels || width <= 0 || height <= 0 || (channels != 1 && channels != 3)) return {};

    const int output_channels = options.grayscale ? 1 : channels;
    int resized_w = options.target_width > 0 ? options.target_width : width;
    int resized_h = options.target_height > 0 ? options.target_height : height;
    if (options.mode == resize_mode::preserve_aspect) {
        const double sx = options.target_width > 0 ? (double)options.target_width / width : 1.0;
        const double sy = options.target_height > 0 ? (double)options.target_height / height : 1.0;
        double scale = options.target_width > 0 && options.target_height > 0 ? std::min(sx, sy) : std::max(sx, sy);
        if (options.max_width > 0) scale = std::min(scale, (double)options.max_width / width);
        if (options.target_width == 0 && options.target_height == 0 && options.max_width > 0)
            scale = std::min(1.0, scale);
        resized_w = std::max(1, (int)std::lround(width * scale));
        resized_h = std::max(1, (int)std::lround(height * scale));
    } else if (options.max_width > 0) {
        resized_w = std::min(resized_w, options.max_width);
    }

    const int canvas_w = options.pad_to_target && options.target_width > 0 ? options.target_width : resized_w;
    const int canvas_h = options.pad_to_target && options.target_height > 0 ? options.target_height : resized_h;
    if (canvas_w <= 0 || canvas_h <= 0 || resized_w > canvas_w || resized_h > canvas_h) return {};
    std::vector<uint8_t> result((size_t)canvas_w * canvas_h * output_channels, options.pad_value);
    const int offset_x = (canvas_w - resized_w) / 2;
    const int offset_y = (canvas_h - resized_h) / 2;
    auto source_value = [&](int x, int y, int c) -> uint8_t {
        const uint8_t * p = pixels + ((size_t)y * width + x) * channels;
        if (options.grayscale) {
            if (channels == 1) return p[0];
            return (uint8_t)((77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8);
        }
        return p[c];
    };
    for (int y = 0; y < resized_h; y++) {
        const double source_y = ((y + 0.5) * height / resized_h) - 0.5;
        const int y0 = std::max(0, std::min(height - 1, (int)std::floor(source_y)));
        const int y1 = std::min(height - 1, y0 + 1);
        const double fy = std::max(0.0, source_y - std::floor(source_y));
        for (int x = 0; x < resized_w; x++) {
            const double source_x = ((x + 0.5) * width / resized_w) - 0.5;
            const int x0 = std::max(0, std::min(width - 1, (int)std::floor(source_x)));
            const int x1 = std::min(width - 1, x0 + 1);
            const double fx = std::max(0.0, source_x - std::floor(source_x));
            uint8_t * dst = result.data() + ((size_t)(offset_y + y) * canvas_w + offset_x + x) * output_channels;
            for (int c = 0; c < output_channels; c++) {
                const double top = source_value(x0, y0, c) * (1.0 - fx) + source_value(x1, y0, c) * fx;
                const double bottom = source_value(x0, y1, c) * (1.0 - fx) + source_value(x1, y1, c) * fx;
                dst[c] = (uint8_t)std::lround(top * (1.0 - fy) + bottom * fy);
            }
        }
    }
    if (out_width) *out_width = canvas_w;
    if (out_height) *out_height = canvas_h;
    if (out_channels) *out_channels = output_channels;
    return result;
}

} // namespace ocr_crop
