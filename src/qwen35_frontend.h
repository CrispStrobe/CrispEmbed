#pragma once
#include "core/gguf_loader.h"
#include "core/wordlevel_added.h"
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
    // Surya OCR 2.5 format 4: character-level WordLevel vocabulary (88526 ids).
    // Requires CRISPEMBED_SURYA25_FRONTEND; BPE state above stays empty.
    bool surya_ = false;
    core_wordlevel::tokenizer wordlevel_;
    std::string image_template_;
    std::vector<int> stops_;
    bool enabled() const;
    bool load_surya25(gguf_context * metadata);

public:
    bool load(gguf_context * metadata);
    // MiniCPM-OCR-1.0 GGUF (arch minicpm_ocr): same Qwen3.5 byte-level BPE,
    // 248100-entry vocabulary, stops [248044, 248046]. Fills the same state as
    // load(), so encode()/decode()/is_stop() are shared; image_prompt() is not
    // used for this profile (the MiniCPM slice layout lives in minicpm_ocr.cpp).
    bool load_minicpm_ocr(gguf_context * metadata);
    bool encode(const std::string & text, std::vector<int> & ids) const;
    bool decode(const std::vector<int> & ids, std::string & text) const;
    // LightOn custom text, or the pinned original Ovis prompt (empty argument).
    // Surya 2.5: the package prompt text (empty means the block-OCR prompt),
    // rendered image-first through the checkpoint chat template.
    bool image_prompt(const std::string & prompt, int image_tokens, std::vector<int> & ids,
                      std::vector<int> & modalities, std::string & rendered) const;
    int eos() const { return eos_; }
    bool is_ovisocr2() const { return ovis_; }
    bool is_ovisocr3() const { return ovis3_; }
    bool is_surya25() const { return surya_; }
    int vocab_size() const { return surya_ ? int(wordlevel_.size()) : int(vocab_.size()); }
    int image_pad_id() const { return surya_ ? 11 : 248056; }
    bool is_stop(int id) const;
    image_config image_limits() const {
        // Surya: vLLM mm_processor_kwargs of surya-ocr 0.23.0 (backends/vllm.py).
        if (surya_) return image_config{ 3136, 6291456 };
        return (ovis_ || ovis3_) ? image_config{ 448 * 448, 2880 * 2880 } : image_config{};
    }
};
} // namespace qwen35_ocr
