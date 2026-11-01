// Streaming SHA-256 (FIPS180-4), adapted from this repository's existing
// examples/cli/model_mgr.cpp. No shell commands or optional crypto dependency.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace core_sha256 {

struct sha256_state {
    uint32_t h[8] = { 0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                      0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u };
    uint64_t total_bits = 0;
    uint8_t buf[64] = {};
    size_t buf_len = 0;
};

inline uint32_t sha_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

inline void sha256_compress(sha256_state & s, const uint8_t * block) {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
    };

    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) | ((uint32_t)block[i * 4 + 1] << 16) | ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 64; i++) {
        const uint32_t s0 = sha_rotr(w[i - 15], 7) ^ sha_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = sha_rotr(w[i - 2], 17) ^ sha_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = s.h[0], b = s.h[1], c = s.h[2], d = s.h[3];
    uint32_t e = s.h[4], f = s.h[5], g = s.h[6], hh = s.h[7];

    for (int i = 0; i < 64; i++) {
        const uint32_t S1 = sha_rotr(e, 6) ^ sha_rotr(e, 11) ^ sha_rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t temp1 = hh + S1 + ch + k[i] + w[i];
        const uint32_t S0 = sha_rotr(a, 2) ^ sha_rotr(a, 13) ^ sha_rotr(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = S0 + maj;

        hh = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    s.h[0] += a;
    s.h[1] += b;
    s.h[2] += c;
    s.h[3] += d;
    s.h[4] += e;
    s.h[5] += f;
    s.h[6] += g;
    s.h[7] += hh;
}

inline void sha256_update(sha256_state & s, const uint8_t * data, size_t len) {
    s.total_bits += (uint64_t)len * 8;
    while (len > 0) {
        const size_t take = std::min(len, sizeof(s.buf) - s.buf_len);
        memcpy(s.buf + s.buf_len, data, take);
        s.buf_len += take;
        data += take;
        len -= take;
        if (s.buf_len == sizeof(s.buf)) {
            sha256_compress(s, s.buf);
            s.buf_len = 0;
        }
    }
}

inline std::string sha256_finish(sha256_state & s) {
    // Pad in place rather than via sha256_update(), which would fold the
    // padding into total_bits and corrupt the length field.
    const uint64_t bits = s.total_bits;

    s.buf[s.buf_len++] = 0x80;
    if (s.buf_len > 56) {
        memset(s.buf + s.buf_len, 0, 64 - s.buf_len);
        sha256_compress(s, s.buf);
        s.buf_len = 0;
    }
    memset(s.buf + s.buf_len, 0, 56 - s.buf_len);
    for (int i = 0; i < 8; i++) s.buf[56 + i] = (uint8_t)(bits >> (56 - i * 8));
    sha256_compress(s, s.buf);

    char out[65];
    for (int i = 0; i < 8; i++) snprintf(out + i * 8, 9, "%08x", s.h[i]);
    return std::string(out, 64);
}

// Lowercase hex digest of a file, or "" when it cannot be read.
inline std::string file(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return "";

    sha256_state s;
    std::array<uint8_t, 64 * 1024> chunk{};
    for (;;) {
        const size_t n = fread(chunk.data(), 1, chunk.size(), f);
        if (n > 0) sha256_update(s, chunk.data(), n);
        if (n < chunk.size()) break;
    }
    const bool ok = ferror(f) == 0;
    const int closed = fclose(f);
    if (!ok || closed != 0) return "";
    return sha256_finish(s);
}

// Hash a bounded memory payload; a null pointer is valid only for zero bytes.
inline std::string bytes(const void * data, size_t size) {
    if (!data && size) return "";
    sha256_state state;
    sha256_update(state, static_cast<const uint8_t *>(data), size);
    return sha256_finish(state);
}

} // namespace core_sha256
