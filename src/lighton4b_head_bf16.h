#pragma once
#include "lighton4b_bf16_linear.h"
#include <algorithm>
#include <cmath>
#include <vector>
namespace qwen35_ocr {
// Compatibility experiment: only final projection output dtype, not weights,
// hidden states, caches or projection accumulation. Off preserves F32 logits.
inline void lighton4b_head_bf16(std::vector<float> & logits) {
    if (core_env::on("CRISPEMBED_LIGHTON4B_HEAD_BF16")) lighton4b_bf16_round(logits);
}
// Torch argmax chooses the first (lowest token ID) among equal maxima.
// NaN is rejected explicitly; infinities retain their normal ordering.
inline int lighton4b_greedy_id(const std::vector<float> & logits) {
    if (logits.empty() || std::any_of(logits.begin(), logits.end(), [](float x) { return std::isnan(x); })) return -1;
    return int(std::max_element(logits.begin(), logits.end()) - logits.begin());
}
} // namespace qwen35_ocr
