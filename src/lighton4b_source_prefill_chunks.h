#pragma once
#include "core/env_gate.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>
namespace qwen35_ocr {
inline bool lighton4b_source_prefill_chunks_mode(bool physical) {
#if defined(__EMSCRIPTEN__)
    (void)physical;
    return false;
#else
    return physical && core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL");
#endif
}
inline bool lighton4b_source_prefill_aligned_mode(bool physical) {
#if defined(__EMSCRIPTEN__)
    (void)physical;
    return false;
#else
    return physical && core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL_ALIGNED");
#endif
}
inline bool lighton4b_source_prefill_aligned_valid(bool physical) {
    return !lighton4b_source_prefill_aligned_mode(physical) || lighton4b_source_prefill_chunks_mode(physical);
}
inline bool lighton4b_source_prefill_budget(int tokens, int past = 0) {
    return tokens >= 1 && past >= 0 && past <= 512 && tokens <= 512 - past;
}
struct lighton4b_prefill_chunk {
    int offset, tokens, logical_offset, logical_tokens;
};
// Pinned6c01 server completion policy: media boundaries and the recurrent
// checkpoint four tokens before prompt end. Bound<=512 keeps the next default
// checkpoint (4+n_ubatch=516) outside this experiment; no case-sized constants.
inline bool lighton4b_source_prefill_plan(const std::vector<int> & modalities,
                                          std::vector<lighton4b_prefill_chunk> & output, bool aligned = false) {
    output.clear();
    if (modalities.empty() || modalities.size() > 512 ||
        !std::all_of(modalities.begin(), modalities.end(), [](int v) { return v == 0 || v == 1; }))
        return false;
    std::vector<lighton4b_prefill_chunk> complete;
    auto segment = [&](int start, int length) {
        for (int offset = start; offset < start + length;) {
            const int left = start + length - offset;
            int count = std::min(16, left);
            // Preserve the source batch>=8 dispatch even for a short final
            // component tile: e.g. logical20 becomes12+8, not16+4.
            if (left > 16 && left - count < 8) {
                count = left - 8;
                // Source preferred Q4 uses groups of4 and one final scalar
                // tail per logical batch. Keep nonfinal tiles divisible4
                // while retaining the source Q6 batch>=8 route for both.
                if (aligned && count % 4 != 0) count = 8;
            }
            complete.push_back({ offset, count, start, length });
            offset += count;
        }
    };
    for (int start = 0; start < int(modalities.size());) {
        int end = start + 1;
        while (end < int(modalities.size()) && modalities[end] == modalities[start]) ++end;
        if (end == int(modalities.size()) && modalities[start] == 0 && end - start > 4) {
            segment(start, end - start - 4);
            segment(end - 4, 4);
        } else
            segment(start, end - start);
        start = end;
    }
    output = std::move(complete);
    return true;
}
// Copy the original section-major T/Y/X axes without rebasing token positions.
inline bool lighton4b_prefill_chunk_positions(const float * positions, int total, const lighton4b_prefill_chunk & chunk,
                                              std::vector<float> & output) {
    output.clear();
    if (!positions || total < 1 || total > 512 || chunk.offset < 0 || chunk.tokens < 1 || chunk.tokens > 16 ||
        chunk.offset > total - chunk.tokens)
        return false;
    std::vector<float> complete(size_t(3) * chunk.tokens);
    for (int axis = 0; axis < 3; ++axis) {
        const float * begin = positions + size_t(axis) * total + chunk.offset;
        if (!std::all_of(begin, begin + chunk.tokens, [](float v) { return std::isfinite(v); })) return false;
        std::copy_n(begin, chunk.tokens, complete.data() + size_t(axis) * chunk.tokens);
    }
    output = std::move(complete);
    return true;
}
} // namespace qwen35_ocr
