// qwen35_graph_kv_decoder.h - default-off ggml graph path for the small Qwen3.5 text
// decoder (24 blocks, 3x gated delta-net : 1x gated full attention, width 1024).
//
// Same mathematics as qwen35_ocr::decoder_cpu (the qualified reference CPU path),
// built as one ggml graph per call with a persistent KV cache (full-attention
// layers) and persistent conv/recurrent state (delta-net layers):
//   * prefill: all prompt rows in one graph; decode: one row per graph;
//   * matrices are read in their stored type; no per-step whole-matrix
//     conversion. Matmul precision is selected by CRISPEMBED_QWEN35_GRAPH_MATMUL:
//       "exact"        F32 activations everywhere. Decode rows (<= 8 tokens)
//                      dequantize one weight row at a time and use the F32 dot;
//                      prefill casts each matrix to F32 inside the graph (once per
//                      prompt) and runs the F32 GEMM, as decoder_cpu does.
//       "native"       ggml_mul_mat on the stored type (F16 activations for F16
//                      weights). Fastest; separately A/B'd for token exactness.
//       unset/"f32"    F32 copies of every matrix made once at init (memory x2).
//   * delta-net softplus/sigmoid and the gated RMSNorm use the reference scalar
//     formulas (custom ops) and ggml_gated_delta_net, as linear_core_cpu does;
//   * RoPE uses host cos/sin tables computed with the reference formula.
// 1-D sequential positions by default (equal t/h/w axes, as MiniCPM-OCR); forward() also
// takes explicit t/h/w position axes for the multimodal qwen35_ocr engines.
//
// Gate: enabled() unless CRISPEMBED_QWEN35_GRAPH_DECODER=0 (core_env::explicitly_off). MiniCPM-OCR, the only
// caller, turns it on by default (=0 opts out); decoder_cpu and every other engine are unchanged.
#pragma once

#include "qwen35_decoder.h"

#include <cstdint>
#include <memory>
#include <vector>

typedef struct ggml_backend * ggml_backend_t;

namespace qwen35_ocr {
class graph_kv_decoder {
public:
    graph_kv_decoder();
    ~graph_kv_decoder();
    graph_kv_decoder(const graph_kv_decoder &) = delete;
    graph_kv_decoder & operator=(const graph_kv_decoder &) = delete;
    static bool enabled();
    // Binds mapped weights (map_small_decoder) living in CPU-accessible buffers.
    // `capacity` bounds prompt + generated tokens (KV cache rows).
    // `allow_f32_copies` = false turns the "f32" matmul mode into "exact" (F32 dot
    // products without the copies; MiniCPM-OCR's low-memory fallback).
    // `default_matmul` ("f32", "exact" or "native") applies when CRISPEMBED_QWEN35_GRAPH_MATMUL
    // is unset or empty; nullptr keeps "f32".
    bool init(const decoder_weights & weights, const attention_config & config, ggml_backend_t backend, int capacity,
              bool allow_f32_copies = true, const char * default_matmul = nullptr);
    // Bytes the F32 copies would allocate under the current CRISPEMBED_QWEN35_GRAPH_MATMUL
    // (0 unless the mode resolves to "f32").
    static uint64_t f32_copy_bytes(const decoder_weights & weights);
    // Starts a new sequence (clears KV/conv/recurrent state).
    void reset();
    int tokens() const;
    // Appends `tokens` rows of input embeddings ([tokens, width] row-major) at
    // positions tokens()..tokens()+tokens-1 and returns the last row's logits.
    // Optional captures: per-layer residual outputs and final-norm rows (prefill dumps).
    // `positions` (optional, [3][tokens] t/h/w axes as qwen35_ocr::image_positions /
    // cached_positions produce): multimodal RoPE with the reference interleaved sections
    // (OvisOCR2/3 via CRISPEMBED_QWEN35_OCR_GRAPH_DECODER). nullptr = 1-D past + row.
    bool forward(const float * hidden, int tokens, int threads, std::vector<float> & logits,
                 std::vector<std::vector<float>> * layers = nullptr, std::vector<float> * norm = nullptr,
                 const float * positions = nullptr);
    const char * mode() const;

private:
    struct impl;
    std::unique_ptr<impl> p_;
};
} // namespace qwen35_ocr
