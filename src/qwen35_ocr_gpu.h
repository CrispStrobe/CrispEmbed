// qwen35_ocr_gpu.h - default-off standard-ggml-op pipeline for the qwen35_ocr engines
// (OvisOCR2/3, LightOnOCR-3-0.8B "Qwen3.5-OCR"), gate CRISPEMBED_QWEN35_OCR_GPU.
//
// The CPU paths (vision_cpu, decoder_cpu, graph_kv_decoder) use plain C++ loops or nine
// ggml custom ops that only the CPU backend can execute. This pipeline builds the same
// two models from standard ggml ops only, so ggml_backend_sched can place every node on
// a GPU backend (CUDA, Metal, Vulkan) with the weights resident there; without a GPU it
// runs the same graphs on the CPU backend.
//
//   vision   one graph per image: patch linear, host position/rotary tables (identical to
//            vision_cpu via vision_host_inputs), 12 pre-LN blocks (block-diagonal frame
//            attention), patch merger
//   decoder  one graph per call with a persistent KV cache and delta-net conv/recurrent
//            state on the backend, multimodal positions via ggml_rope_multi(IMROPE)
//
// Derived from the pinned upstream Python, transformers 5.5.4
// models/qwen3_5/modeling_qwen3_5.py (sha256 aee59d55ee4e8ce0e50bf0e279796b85c2c66a28dfae55c3fcbb62fa9bcba048);
// line citations are in qwen35_ocr_gpu.cpp. Precision contract: token IDs, text and stop
// exact against the CPU ground truth; confidence drift is measured, not required to be 0.
#pragma once

#include "qwen35_attention.h"
#include "qwen35_decoder.h"
#include "qwen35_vision.h"

#include <memory>
#include <vector>

namespace qwen35_ocr {
class ggml_pipeline {
public:
    ggml_pipeline();
    ~ggml_pipeline();
    ggml_pipeline(const ggml_pipeline &) = delete;
    ggml_pipeline & operator=(const ggml_pipeline &) = delete;
    // core_env::on("CRISPEMBED_QWEN35_OCR_GPU").
    static bool enabled();
    // Weights are the CPU-mapped tensors of the context. On a GPU backend they are copied
    // once to a backend buffer; on the CPU backend they are used in place.
    bool init(const decoder_weights & decoder, const vision_weights & vision, const attention_config & attention,
              const vision_config & vision_config, int capacity, int threads);
    const char * backend_name() const;
    bool on_gpu() const;
    // merged: [tokens / merge^2, output] row-major (= vision_outputs::merged).
    bool vision(const float * patches, int tokens, const std::vector<int> & grid, std::vector<float> & merged);
    void reset();
    int tokens() const;
    // positions: [3][tokens] t/h/w axes (qwen35_ocr::image_positions / cached_positions).
    bool forward(const float * hidden, int tokens, const float * positions, std::vector<float> & logits);

private:
    struct impl;
    std::unique_ptr<impl> p_;
};
} // namespace qwen35_ocr
