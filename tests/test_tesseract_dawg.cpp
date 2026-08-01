#include "tesseract_dawg.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

static void append_u16(std::vector<uint8_t> & bytes, uint16_t value) {
    bytes.push_back((uint8_t)value);
    bytes.push_back((uint8_t)(value >> 8));
}

static void append_u32(std::vector<uint8_t> & bytes, uint32_t value) {
    for (int i = 0; i < 4; ++i) bytes.push_back((uint8_t)(value >> (8 * i)));
}

static void append_u64(std::vector<uint8_t> & bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) bytes.push_back((uint8_t)(value >> (8 * i)));
}

int main() {
    // unicharset_size=7 => 3 letter bits, flags start at bit 3, next node at 6.
    // Root: 'a' -> node 2, 'b' -> terminal; node 2: 'c' -> word end.
    const uint64_t a = 0u | (2ull << 6);
    const uint64_t b = 1u | (1ull << 3) | (1ull << 5); // marker + word end, next=0
    const uint64_t c = 2u | (1ull << 3) | (1ull << 5);
    std::vector<uint8_t> bytes;
    append_u16(bytes, 42);
    append_u32(bytes, 7);
    append_u32(bytes, 3);
    append_u64(bytes, a);
    append_u64(bytes, b);
    append_u64(bytes, c);

    tesseract_dawg::Dawg dawg;
    std::string error;
    if (!tesseract_dawg::parse(bytes, dawg, &error) ||
        !tesseract_dawg::prefix_matches(dawg, { 0 }, false) ||
        !tesseract_dawg::prefix_matches(dawg, { 0, 2 }, true) ||
        !tesseract_dawg::prefix_matches(dawg, { 1 }, true) ||
        tesseract_dawg::prefix_matches(dawg, { 0 }, true) ||
        tesseract_dawg::prefix_matches(dawg, { 0, 3 }, false)) {
        std::fprintf(stderr, "valid dawg test failed: %s\n", error.c_str());
        return 1;
    }

    bytes[0] = 0;
    if (tesseract_dawg::parse(bytes, dawg, &error) || error != "invalid dawg magic") {
        std::fprintf(stderr, "invalid dawg rejection failed\n");
        return 1;
    }

    // A malformed unicharset header must be rejected before any edge parsing.
    bytes.clear();
    append_u16(bytes, 42);
    append_u32(bytes, 0);
    append_u32(bytes, 1);
    append_u64(bytes, 0);
    if (tesseract_dawg::parse(bytes, dawg, &error) || error != "invalid dawg unicharset size") {
        std::fprintf(stderr, "malformed dawg unicharset rejection failed: %s\n", error.c_str());
        return 1;
    }
    std::puts("tesseract dawg: PASS");
    return 0;
}
