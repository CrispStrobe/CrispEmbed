#pragma once
#include "qwen35_vision.h"
#include <cstdint>
#include <cstring>
namespace qwen35_ocr {
// Round finite F32 to the source BF16 value using round-to-nearest-even.
inline float lighton4b_bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        if (bits & 0x007fffffu) bits |= 0x00400000u;
    } else
        bits += 0x7fffu + ((bits >> 16) & 1u);
    bits &= 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
bool lighton4b_vision_bf16_cpu(const vision_weights &, const vision_config &, const float *, int,
                               const std::vector<int> &, int, vision_outputs &);
} // namespace qwen35_ocr
