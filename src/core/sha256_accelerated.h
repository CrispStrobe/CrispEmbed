// Optional native SHA-256 compression for digest-verified inpainting loads.
// Reuses CrispEmbed's existing scalar state, padding and digest formatting.
// The ordinary core_sha256::file/bytes functions and model_mgr are unchanged.
//
// The SHA-NI implementation below is new code derived from the instruction
// semantics, not copied from a third-party implementation:
//   NIST FIPS 180-4, sections 5.1, 5.2 and 6.2:
//   https://doi.org/10.6028/NIST.FIPS.180-4
//   Intel instruction reference, SHA256RNDS2/SHA256MSG1/SHA256MSG2:
//   https://www.intel.com/content/dam/develop/external/us/en/documents/319433-024-697869.pdf
// No optional crypto dependency, shell command or global ISA build flag.
#pragma once

#include "core/sha256.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

// x86-64 guarantees operating-system XMM state support. Other targets and
// compilers retain the scalar path. A baseline x86 build can compile this
// isolated function without enabling SHA/SSE4.1 for the rest of the binary.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER) &&                          \
    !defined(CRISPEMBED_SHA256_DISABLE_INTRINSICS)
#define CRISPEMBED_SHA256_HAS_TARGET_INTRINSICS 1
#include <cpuid.h>
#include <immintrin.h>
#else
#define CRISPEMBED_SHA256_HAS_TARGET_INTRINSICS 0
#endif

namespace core_sha256_accel {

struct Stats {
    bool requested = false;
    bool hardware_available = false;
    bool sha_ni_selected = false;
    uint64_t input_bytes = 0;
    uint64_t complete_blocks = 0;
    uint64_t accelerated_blocks = 0;
};

// Separate pure predicate permits negative feature controls without ever
// executing an unsupported instruction. CPUID.1:ECX[9,19], CPUID.7.0:EBX[29].
inline bool required_features(uint32_t leaf1_ecx, uint32_t leaf7_ebx) {
    constexpr uint32_t ssse3 = UINT32_C(1) << 9;
    constexpr uint32_t sse41 = UINT32_C(1) << 19;
    constexpr uint32_t sha = UINT32_C(1) << 29;
    return (leaf1_ecx & (ssse3 | sse41)) == (ssse3 | sse41) && (leaf7_ebx & sha) != 0;
}

inline bool compiled_sha_ni() {
    return CRISPEMBED_SHA256_HAS_TARGET_INTRINSICS != 0;
}

inline bool hardware_available() {
#if CRISPEMBED_SHA256_HAS_TARGET_INTRINSICS
    static const bool available = [] {
        if (__get_cpuid_max(0, nullptr) < 7) return false;
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        __cpuid_count(1, 0, eax, ebx, ecx, edx);
        const uint32_t leaf1_ecx = ecx;
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return required_features(leaf1_ecx, ebx);
    }();
    return available;
#else
    return false;
#endif
}

namespace detail {

using Compress = void (*)(core_sha256::sha256_state &, const uint8_t *, size_t);

inline void scalar_blocks(core_sha256::sha256_state & state, const uint8_t * input, size_t blocks) {
    for (size_t i = 0; i < blocks; ++i) core_sha256::sha256_compress(state, input + 64 * i);
}

#if CRISPEMBED_SHA256_HAS_TARGET_INTRINSICS
__attribute__((target("sha,ssse3,sse4.1"), noinline)) inline void sha_ni_blocks(core_sha256::sha256_state & state,
                                                                                const uint8_t * input, size_t blocks) {
    // The same public SHA-256 round constants already used by core/sha256.h.
    static const uint32_t constants[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };
    const __m128i endian = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    // SHA256RNDS2 orders the working variables from high to low as A,B,E,F
    // and C,D,G,H. Explicit construction avoids an opaque state shuffle.
    const uint32_t initial_abef[4] = { state.h[5], state.h[4], state.h[1], state.h[0] };
    const uint32_t initial_cdgh[4] = { state.h[7], state.h[6], state.h[3], state.h[2] };
    __m128i abef = _mm_loadu_si128(reinterpret_cast<const __m128i *>(initial_abef));
    __m128i cdgh = _mm_loadu_si128(reinterpret_cast<const __m128i *>(initial_cdgh));
    for (size_t block = 0; block < blocks; ++block) {
        const __m128i saved_abef = abef, saved_cdgh = cdgh;
        __m128i words[4];
        for (int group = 0; group < 16; ++group) {
            const int slot = group & 3;
            __m128i message;
            if (group < 4) {
                message = _mm_shuffle_epi8(
                    _mm_loadu_si128(reinterpret_cast<const __m128i *>(input + 64 * block + 16 * group)), endian);
            } else {
                // Four new words from W[t-16..t-1]. The aligned middle term
                // is W[t-7..t-4]; MSG2 supplies the two dependent new words.
                const __m128i next = words[(slot + 1) & 3];
                const __m128i middle = words[(slot + 2) & 3];
                const __m128i last = words[(slot + 3) & 3];
                message = _mm_sha256msg1_epu32(words[slot], next);
                message = _mm_add_epi32(message, _mm_alignr_epi8(last, middle, 4));
                message = _mm_sha256msg2_epu32(message, last);
            }
            words[slot] = message;
            __m128i round =
                _mm_add_epi32(message, _mm_loadu_si128(reinterpret_cast<const __m128i *>(constants + 4 * group)));
            cdgh = _mm_sha256rnds2_epu32(cdgh, abef, round);
            round = _mm_shuffle_epi32(round, 0x0e);
            abef = _mm_sha256rnds2_epu32(abef, cdgh, round);
        }
        abef = _mm_add_epi32(abef, saved_abef);
        cdgh = _mm_add_epi32(cdgh, saved_cdgh);
    }
    uint32_t first[4], second[4];
    _mm_storeu_si128(reinterpret_cast<__m128i *>(first), abef);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(second), cdgh);
    state.h[0] = first[3];
    state.h[1] = first[2];
    state.h[2] = second[3];
    state.h[3] = second[2];
    state.h[4] = first[1];
    state.h[5] = first[0];
    state.h[6] = second[1];
    state.h[7] = second[0];
}
#endif

} // namespace detail

class Hasher {
public:
    explicit Hasher(bool request_acceleration) {
        stats_.requested = request_acceleration;
        stats_.hardware_available = hardware_available();
        stats_.sha_ni_selected = request_acceleration && stats_.hardware_available;
#if CRISPEMBED_SHA256_HAS_TARGET_INTRINSICS
        if (stats_.sha_ni_selected) compress_ = detail::sha_ni_blocks;
#endif
    }

    bool update(const void * source, size_t size) {
        if (finished_ || failed_) return false;
        if (!source && size) {
            failed_ = true;
            return false;
        }
        const auto * input = static_cast<const uint8_t *>(source);
        state_.total_bits += uint64_t(size) * 8;
        stats_.input_bytes += uint64_t(size);
        if (state_.buf_len && size) {
            const size_t take = std::min(size, sizeof(state_.buf) - state_.buf_len);
            std::memcpy(state_.buf + state_.buf_len, input, take);
            state_.buf_len += take;
            input += take;
            size -= take;
            if (state_.buf_len == sizeof(state_.buf)) {
                blocks(state_.buf, 1);
                state_.buf_len = 0;
            }
        }
        const size_t count = size / 64;
        if (count) {
            blocks(input, count);
            input += count * 64;
            size -= count * 64;
        }
        if (size) {
            std::memcpy(state_.buf, input, size);
            state_.buf_len = size;
        }
        return true;
    }

    std::string finish() {
        if (failed_) return {};
        if (!finished_) {
            // Existing padding and final length encoding are reused verbatim.
            // At most two final blocks use scalar compression; no new padding
            // implementation or precision/length convention is introduced.
            digest_ = core_sha256::sha256_finish(state_);
            finished_ = true;
        }
        return digest_;
    }

    const Stats & stats() const { return stats_; }

private:
    void blocks(const uint8_t * input, size_t count) {
        compress_(state_, input, count);
        stats_.complete_blocks += uint64_t(count);
        if (stats_.sha_ni_selected) stats_.accelerated_blocks += uint64_t(count);
    }
    core_sha256::sha256_state state_;
    detail::Compress compress_ = detail::scalar_blocks;
    Stats stats_;
    bool failed_ = false, finished_ = false;
    std::string digest_;
};

inline std::string bytes(const void * data, size_t size, bool request_acceleration, Stats * stats = nullptr) {
    Hasher state(request_acceleration);
    const bool ok = state.update(data, size);
    const std::string result = ok ? state.finish() : std::string();
    if (stats) *stats = state.stats();
    return result;
}

// Gate-off without instrumentation delegates to the original file function.
// When scalar statistics are requested, preserve that function's fread,
// sha256_update and final-padding operations while counting their actual
// bytes. Gate-on validates every input byte using the selected compressor.
// All paths return the same digest or empty string on open/read/close failure.
inline std::string file(const std::string & path, bool request_acceleration, Stats * stats = nullptr) {
    if (stats) *stats = Stats{};
    if (!request_acceleration && !stats) return core_sha256::file(path);
    if (!request_acceleration) {
        FILE * input = std::fopen(path.c_str(), "rb");
        if (!input) return {};
        core_sha256::sha256_state state;
        std::array<uint8_t, 64 * 1024> buffer{};
        uint64_t input_bytes = 0;
        for (;;) {
            const size_t count = std::fread(buffer.data(), 1, buffer.size(), input);
            if (count) {
                core_sha256::sha256_update(state, buffer.data(), count);
                input_bytes += uint64_t(count);
            }
            if (count < buffer.size()) break;
        }
        const bool read_ok = std::ferror(input) == 0;
        const int closed = std::fclose(input);
        stats->hardware_available = hardware_available();
        stats->input_bytes = input_bytes;
        stats->complete_blocks = input_bytes / 64;
        if (!read_ok || closed != 0) return {};
        return core_sha256::sha256_finish(state);
    }
    FILE * input = std::fopen(path.c_str(), "rb");
    if (!input) return {};
    Hasher state(true);
    std::array<uint8_t, 64 * 1024> buffer{};
    for (;;) {
        const size_t count = std::fread(buffer.data(), 1, buffer.size(), input);
        if (count) state.update(buffer.data(), count);
        if (count < buffer.size()) break;
    }
    const bool read_ok = std::ferror(input) == 0;
    const int closed = std::fclose(input);
    if (stats) *stats = state.stats();
    if (!read_ok || closed != 0) return {};
    return state.finish();
}

// Call only after matching the complete observed digest to its trusted pin,
// and only when the caller's existing profiling gate is enabled. Stable
// labels identify the model; no filesystem path is written to the log.
inline void profile_verified(const char * label, const Stats & stats, const std::string & digest) {
    std::fprintf(stderr,
                 "inpaint_hash: label=%s backend=%s requested=%d hardware_available=%d bytes=%llu "
                 "complete_blocks=%llu accelerated_blocks=%llu sha256=%s\n",
                 label, stats.sha_ni_selected ? "sha-ni" : "scalar", int(stats.requested),
                 int(stats.hardware_available), static_cast<unsigned long long>(stats.input_bytes),
                 static_cast<unsigned long long>(stats.complete_blocks),
                 static_cast<unsigned long long>(stats.accelerated_blocks), digest.c_str());
}

} // namespace core_sha256_accel

#undef CRISPEMBED_SHA256_HAS_TARGET_INTRINSICS
