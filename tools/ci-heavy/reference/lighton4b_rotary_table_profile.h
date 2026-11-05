#pragma once
// Proposed model-arithmetic reader, test-only until complete-profile validation.
// SPDX-License-Identifier: MIT
#include "qwen35_vision.h"
#include "core/env_gate.h"
#include <array>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <limits>
namespace qwen35_ocr {
namespace lighton4b_rotary_profile {
#include "lighton4b_rotary_table_data.inc"
inline bool lookup(const vision_config & c, int tokens, const std::vector<int> & rows, const std::vector<int> & columns,
                   bool capability_supported, std::vector<float> & cosine, std::vector<float> & sine) {
    if (!core_env::on("CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_TABLE") || !capability_supported ||
        std::fegetround() != FE_TONEAREST || !std::numeric_limits<float>::is_iec559 || sizeof(float) != 4 ||
        c.width != 1024 || c.heads != 16 || c.merge != 2 || c.theta != 10000.0f || tokens < 4 || tokens > 4096 ||
        tokens % 4 || rows.size() != size_t(tokens) || columns.size() != size_t(tokens))
        return false;
    for (int i = 0; i < tokens; ++i)
        if (rows[i] < 0 || rows[i] >= 1024 || columns[i] < 0 || columns[i] >= 1024) return false;
    cosine.resize(size_t(tokens) * 64);
    sine.resize(cosine.size());
    for (int t = 0; t < tokens; ++t)
        for (int i = 0; i < 32; ++i) {
            const int coordinate = i < 16 ? rows[t] : columns[t], frequency = i % 16;
            float co, si;
            const uint32_t cb = rotary_bits[(coordinate * 16 + frequency) * 2],
                           sb = rotary_bits[(coordinate * 16 + frequency) * 2 + 1];
            std::memcpy(&co, &cb, 4);
            std::memcpy(&si, &sb, 4);
            for (int half = 0; half < 2; ++half) {
                const size_t at = size_t(t) * 64 + i + half * 32;
                cosine[at] = co;
                sine[at] = si;
            }
        }
    return true;
}
} // namespace lighton4b_rotary_profile
} // namespace qwen35_ocr
