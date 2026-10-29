#pragma once
#include "qwen35_decoder.h"
namespace josev2b_backbone {
// Explicit F32-only text component:320 tensors, no vision, lm_head or joint head.
// Both CRISPEMBED_QWEN35_OCR and CRISPEMBED_JOSEV_BACKBONE must be enabled.
bool map(const core_gguf::tensor_map & tensors, qwen35_ocr::decoder_weights & weights);
bool run(const qwen35_ocr::decoder_weights & weights, const std::vector<int> & ids, int threads,
         qwen35_ocr::decoder_outputs & output);
// Separate mixed-storage entrypoints. Vocabulary, all norms, linear gates and
// convolution remain F32; legacy map/run above never accept quantized matrices.
bool map_quantized(const core_gguf::tensor_map & tensors, qwen35_ocr::decoder_weights & weights);
bool run_quantized(const qwen35_ocr::decoder_weights & weights, const std::vector<int> & ids, int threads,
                   qwen35_ocr::decoder_outputs & output);
} // namespace josev2b_backbone
