#pragma once
#include "ggml.h"
#include <cstdint>

// GGUF general.file_type uses the standardized file encoding enum, not ggml's
// internal ggml_ftype or tensor ggml_type. Keep both backend enums unchanged.
// GGUF specification6af560d5 docs/gguf.md; extended file types llama.h63412d87.
namespace core_gguf_file_type {
constexpr uint32_t q4_k_s = 14;
constexpr uint32_t q5_k_s = 16;
constexpr uint32_t q6_k = 18;
constexpr uint32_t legacy_q4_k = 12;
constexpr uint32_t unknown = UINT32_MAX;
inline constexpr bool known_q4_k(uint32_t value) {
    // Caller must also authenticate the exact Q4 profile, actual tensor types
    // and existing quantization gates. A file-level label proves no payload.
    return value == q4_k_s || value == legacy_q4_k;
}
inline constexpr uint32_t standard(ggml_ftype value) {
    switch (value) {
    case GGML_FTYPE_ALL_F32:
        return 0;
    case GGML_FTYPE_MOSTLY_F16:
        return 1;
    case GGML_FTYPE_MOSTLY_Q4_0:
        return 2;
    case GGML_FTYPE_MOSTLY_Q4_1:
        return 3;
    case GGML_FTYPE_MOSTLY_Q8_0:
        return 7;
    case GGML_FTYPE_MOSTLY_Q5_0:
        return 8;
    case GGML_FTYPE_MOSTLY_Q5_1:
        return 9;
    case GGML_FTYPE_MOSTLY_Q2_K:
        return 10;
    case GGML_FTYPE_MOSTLY_Q3_K:
        return 11;
    case GGML_FTYPE_MOSTLY_Q4_K:
        return q4_k_s;
    case GGML_FTYPE_MOSTLY_Q5_K:
        return q5_k_s;
    case GGML_FTYPE_MOSTLY_Q6_K:
        return q6_k;
    case GGML_FTYPE_MOSTLY_IQ4_NL:
        return 25;
    case GGML_FTYPE_MOSTLY_IQ4_XS:
        return 30;
    default:
        return unknown;
    }
}
} // namespace core_gguf_file_type
