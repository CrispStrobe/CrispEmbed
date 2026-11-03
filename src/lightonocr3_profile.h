// LightOnOCR-3-1B input contract. The legacy LightOnOCR-2 path does not call this.
#pragma once
#include "core/torch_bicubic_u8.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace lightonocr3 {
template <size_t N> inline bool same_ids(const std::vector<int32_t> & actual, const std::array<int32_t, N> & expected) {
    return actual.size() == N && std::equal(actual.begin(), actual.end(), expected.begin());
}

inline bool token_contract_matches(std::string_view model, std::string_view revision,
                                   const std::vector<int32_t> & prefix, const std::vector<int32_t> & suffix,
                                   const std::vector<int32_t> & grounding_prefix,
                                   const std::vector<int32_t> & grounding_suffix, const std::vector<int32_t> & stops,
                                   const std::vector<int32_t> & special) {
    return model == "lightonai/LightOnOCR-3-1B" && revision == "23d544085c886bbdace647bcfab91203299417e2" &&
           same_ids(prefix, std::array<int32_t, 7>{ 151644, 8948, 151645, 198, 151644, 872, 198 }) &&
           same_ids(suffix, std::array<int32_t, 5>{ 151645, 198, 151644, 77091, 198 }) && grounding_prefix == prefix &&
           same_ids(grounding_suffix, std::array<int32_t, 8>{ 1951, 287, 198, 151645, 198, 151644, 77091, 198 }) &&
           same_ids(stops, std::array<int32_t, 2>{ 151645, 151643 }) &&
           same_ids(special, std::array<int32_t, 14>{ 151643, 151644, 151645, 151646, 151647, 151648, 151649, 151650,
                                                      151651, 151652, 151653, 151654, 151655, 151656 });
}

// Pinned PixtralImageProcessor: floor downscale, ceil to 14, no upscale except
// patch alignment. An odd patch grid is valid; unfold drops the unmatched edge.
inline std::pair<int, int> image_size(int height, int width) {
    if (height <= 0 || width <= 0) throw std::invalid_argument("invalid image dimensions");
    const double ratio = std::max(double(height) / 1540, double(width) / 1540);
    if (ratio > 1) {
        height = int(std::floor(height / ratio));
        width = int(std::floor(width / ratio));
    }
    const int h = ((height - 1) / 14 + 1) * 14;
    const int w = ((width - 1) / 14 + 1) * 14;
    if (h < 28 || w < 28) throw std::invalid_argument("image has no 2x2 merge on one axis");
    return { h, w };
}

inline std::vector<float> preprocess(const uint8_t * rgb, int height, int width, int & ph, int & pw) {
    const auto shape = image_size(height, width);
    const int h = shape.first, w = shape.second;
    ph = h / 14;
    pw = w / 14;
    std::vector<uint8_t> resized;
    core_torch_resize::resize_rgb(rgb, height, width, h, w, resized);
    const float mean[] = { 0.48145466f, 0.4578275f, 0.40821073f };
    const float stddev[] = { 0.26862954f, 0.26130258f, 0.27577711f };
    std::vector<float> out(size_t(3) * h * w);
    // TorchvisionBackend fuses rescale into mean/std in float32 before normalize.
    for (int c = 0; c < 3; ++c) {
        const float mean255 = mean[c] * 255.0f, std255 = stddev[c] * 255.0f;
        for (int i = 0; i < h * w; ++i)
            out[size_t(c) * h * w + i] = (float(resized[size_t(i) * 3 + c]) - mean255) / std255;
    }
    return out;
}
} // namespace lightonocr3
