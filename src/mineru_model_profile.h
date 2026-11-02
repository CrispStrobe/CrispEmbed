// MinerU2.5-Pro-2604 frontend contract. The inference gate is checked by the
// owning qwen2vl context; these pure helpers are also used by weight-free QA.
#pragma once

#include "core/torch_bicubic_u8.h"
#include "image_preprocess.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace mineru25_profile {

constexpr const char * family = "mineru2.5-pro-2604";
constexpr const char * gate = "CRISPEMBED_MINERU25_PRO";
constexpr int context_length = 8192;
constexpr int no_repeat_ngram = 100;

// Python round(n / 28), including ties-to-even, without depending on the
// process floating-point rounding mode. Do not clamp before the area branch:
// that changes HF's minimum-area behavior for very narrow inputs.
inline int round_factor_even(int value, int factor = 28) {
    const int quotient = value / factor, remainder = value % factor;
    return factor * (quotient + (2 * remainder > factor || (2 * remainder == factor && (quotient & 1))));
}

inline bool smart_resize(int height, int width, int min_pixels, int max_pixels, int & rh, int & rw) {
    rh = rw = 0;
    if (height < 1 || width < 1 || height > 65536 || width > 65536 || int64_t(height) * width > 67108864 ||
        double(std::max(height, width)) / std::min(height, width) > 200.0 || min_pixels != 50176 ||
        max_pixels != 1605632)
        return false;
    rh = round_factor_even(height);
    rw = round_factor_even(width);
    if (int64_t(rh) * rw > max_pixels) {
        const double beta = std::sqrt(double(height) * width / max_pixels);
        rh = std::max(28, int(std::floor(height / beta / 28)) * 28);
        rw = std::max(28, int(std::floor(width / beta / 28)) * 28);
    } else if (int64_t(rh) * rw < min_pixels) {
        const double beta = std::sqrt(double(min_pixels) / (double(height) * width));
        rh = int(std::ceil(height * beta / 28)) * 28;
        rw = int(std::ceil(width * beta / 28)) * 28;
    }
    return rh >= 28 && rw >= 28 && rh % 28 == 0 && rw % 28 == 0 && int64_t(rh) * rw <= 2 * int64_t(max_pixels);
}

// Model card processor: Qwen2VLImageProcessorFast, use_fast=True. Its CPU
// uint8 AA bicubic coefficients and fused rescale/normalize differ from PIL
// and from the generic float bicubic path. The outer page/crop transformations
// in MinerUClient still use PIL; those occur before this function.
inline bool preprocess_rgb(const uint8_t * pixels, int height, int width, int channels,
                           const image_preproc::config & cfg, image_preproc::result & output) {
    output = {};
    if (!pixels || (channels != 1 && channels != 3 && channels != 4) || cfg.patch_size != 14 ||
        cfg.temporal_patch_size != 2 || cfg.merge_size != 2 || cfg.target_longest || cfg.deskew)
        return false;
    int rh = 0, rw = 0;
    if (!smart_resize(height, width, cfg.min_pixels, cfg.max_pixels, rh, rw)) return false;
    for (int c = 0; c < 3; ++c)
        if (!std::isfinite(cfg.mean[c]) || !std::isfinite(cfg.std[c]) || cfg.std[c] <= 0) return false;
    std::vector<uint8_t> rgb;
    const uint8_t * input = pixels;
    if (channels != 3) {
        rgb.resize(size_t(height) * width * 3);
        for (size_t i = 0; i < size_t(height) * width; ++i)
            for (int c = 0; c < 3; ++c) {
                // MinerU get_rgb_image uses PIL.convert("RGB"): drop alpha,
                // preserving RGB bytes even for fully transparent pixels.
                const unsigned value = pixels[size_t(channels) * i + (channels == 1 ? 0 : c)];
                rgb[3 * i + c] = uint8_t(value);
            }
        input = rgb.data();
    }
    std::vector<uint8_t> resized;
    core_torch_resize::resize_rgb(input, height, width, rh, rw, resized);
    image_preproc::result result;
    result.resized_h = rh;
    result.resized_w = rw;
    result.grid_thw[0] = 1;
    result.grid_thw[1] = rh / 14;
    result.grid_thw[2] = rw / 14;
    result.n_patches = (rh / 14) * (rw / 14);
    result.row_dim = 3 * 2 * 14 * 14;
    result.patches.reserve(size_t(result.n_patches) * result.row_dim);
    float mean[3], stddev[3];
    for (int c = 0; c < 3; ++c) {
        mean[c] = cfg.mean[c] * 255.0f;
        stddev[c] = cfg.std[c] * 255.0f;
    }
    // Merge-block order, then C,T,P,P: exact HF view(...).permute(...).
    // Both temporal slices refer to the same RGB frame.
    for (int by = 0; by < rh / 28; ++by)
        for (int bx = 0; bx < rw / 28; ++bx)
            for (int iy = 0; iy < 2; ++iy)
                for (int ix = 0; ix < 2; ++ix)
                    for (int c = 0; c < 3; ++c)
                        for (int t = 0; t < 2; ++t)
                            for (int py = 0; py < 14; ++py)
                                for (int px = 0; px < 14; ++px) {
                                    const int y = by * 28 + iy * 14 + py, x = bx * 28 + ix * 14 + px;
                                    result.patches.push_back((float(resized[(size_t(y) * rw + x) * 3 + c]) - mean[c]) /
                                                             stddev[c]);
                                }
    output = std::move(result);
    return true;
}

inline bool is_eos(int32_t token) {
    return token == 151645 || token == 151643;
}

struct token_choice {
    int id = -1;
    float logit = -std::numeric_limits<float>::infinity();
    float probability = 0;
};

// NoRepeatNGramLogitsProcessor(100) sees input_ids (prompt + generated IDs)
// in HF. Unlike the generic OCR three-gram heuristic, this admits repeated
// table cells/formula fragments and keeps the Python scoring probabilities.
inline token_choice greedy_choice(const float * logits, int vocab, const std::vector<int32_t> & history) {
    token_choice result;
    if (!logits || vocab < 1) return result;
    std::unordered_set<int32_t> banned;
    const size_t k = no_repeat_ngram - 1;
    if (history.size() >= k) {
        for (size_t i = 0; i + k < history.size(); ++i) {
            if (std::equal(history.begin() + i, history.begin() + i + k, history.end() - k))
                banned.insert(history[i + k]);
        }
    }
    for (int v = 0; v < vocab; ++v) {
        if (!std::isfinite(logits[v])) return {};
        if (!banned.count(v) && logits[v] > result.logit) {
            result.id = v;
            result.logit = logits[v];
        }
    }
    if (result.id < 0) return result;
    double sum = 0;
    for (int v = 0; v < vocab; ++v)
        if (!banned.count(v)) sum += std::exp(double(logits[v]) - result.logit);
    result.probability = float(1.0 / sum);
    return result;
}

// Match Python bytes.decode(errors='replace') for incomplete UTF-8 at a
// generation cap. Well-formed text (including every structured special token)
// is unchanged. Each maximal valid prefix of an invalid sequence is replaced.
inline std::string replace_invalid_utf8(const std::string & input) {
    std::string output;
    output.reserve(input.size());
    for (size_t i = 0; i < input.size();) {
        const uint8_t lead = uint8_t(input[i]);
        int n = lead < 0x80 ? 1
                            : (lead >= 0xc2 && lead <= 0xdf
                                   ? 2
                                   : (lead >= 0xe0 && lead <= 0xef ? 3 : (lead >= 0xf0 && lead <= 0xf4 ? 4 : 0)));
        if (n == 1) {
            output += input[i++];
            continue;
        }
        size_t consumed = 1;
        bool valid = n > 0;
        for (int j = 1; valid && j < n; ++j) {
            if (i + j >= input.size()) {
                valid = false;
                break;
            }
            const uint8_t byte = uint8_t(input[i + j]);
            if (byte < 0x80 || byte > 0xbf ||
                (j == 1 && ((lead == 0xe0 && byte < 0xa0) || (lead == 0xed && byte > 0x9f) ||
                            (lead == 0xf0 && byte < 0x90) || (lead == 0xf4 && byte > 0x8f)))) {
                valid = false;
                break;
            }
            ++consumed;
        }
        if (valid) {
            output.append(input, i, n);
            i += n;
        } else {
            output += "\xef\xbf\xbd";
            i += consumed;
        }
    }
    return output;
}

} // namespace mineru25_profile
