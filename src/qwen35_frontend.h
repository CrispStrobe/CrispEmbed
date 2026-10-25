#pragma once
#include "core/gguf_loader.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
namespace qwen35_ocr {
struct image_config {
    int min_pixels = 65536, max_pixels = 16777216;
};
struct image_inputs {
    std::vector<uint8_t> resized_rgb;
    std::vector<float> patches;
    std::vector<int> grid;
    int height = 0, width = 0;
};
// Pinned Qwen2VL torchvision CPU uint8 bicubic, fused .5 normalization, P16/T2/M2.
// Still RGB only; maximum 1024 patches matches the current native vision bound.
bool preprocess_rgb(const uint8_t * rgb, int height, int width, const image_config & config, image_inputs & output);
class frontend_tokenizer {
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, int32_t> tokens_, merges_;
    std::vector<int> added_, special_;
    int eos_ = -1;
    bool ovis_ = false;
    bool ovis3_ = false;
    std::string image_template_;
    std::vector<int> stops_;
    bool enabled() const;

public:
    bool load(gguf_context * metadata);
    bool encode(const std::string & text, std::vector<int> & ids) const;
    bool decode(const std::vector<int> & ids, std::string & text) const;
    // LightOn custom text, or the pinned original Ovis prompt (empty argument).
    bool image_prompt(const std::string & prompt, int image_tokens, std::vector<int> & ids,
                      std::vector<int> & modalities, std::string & rendered) const;
    int eos() const { return eos_; }
    bool is_ovisocr2() const { return ovis_; }
    bool is_ovisocr3() const { return ovis3_; }
    bool is_stop(int id) const;
    image_config image_limits() const { return (ovis_ || ovis3_) ? image_config{ 448 * 448, 2880 * 2880 } : image_config{}; }
};
} // namespace qwen35_ocr
