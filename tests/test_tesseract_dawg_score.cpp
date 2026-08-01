#include "tesseract_dawg_score.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

static void u16(std::vector<uint8_t> & bytes, uint16_t value) {
    bytes.push_back((uint8_t)value);
    bytes.push_back((uint8_t)(value >> 8));
}

static void u32(std::vector<uint8_t> & bytes, uint32_t value) {
    for (int i = 0; i < 4; ++i) bytes.push_back((uint8_t)(value >> (8 * i)));
}

static void u64(std::vector<uint8_t> & bytes, uint64_t value) {
    for (int i = 0; i < 8; ++i) bytes.push_back((uint8_t)(value >> (8 * i)));
}

int main() {
    // DAWG word [1, 2]: root letter 1 -> node 2, then letter 2 at EOW.
    // Each forward edge run has its marker terminator.
    std::vector<uint8_t> bytes;
    u16(bytes, 42);
    u32(bytes, 7);
    u32(bytes, 3);
    u64(bytes, 1u | (2ull << 6));
    u64(bytes, (1ull << 3) | (1ull << 5));
    u64(bytes, 2u | (1ull << 3) | (1ull << 5));

    tesseract_dawg::Dawg dawg;
    std::string error;
    if (!tesseract_dawg::parse(bytes, dawg, &error)) {
        std::fprintf(stderr, "score fixture parse failed: %s\n", error.c_str());
        return 1;
    }
    const std::vector<std::vector<int>> codes = { { 0 }, { 4, 5 }, { 6 } };
    const std::vector<std::string> tokens = { " ", "a", "b" };
    std::map<std::string, tesseract_dawg::Dawg> dawgs;
    dawgs.emplace("lstm-system-dawg", dawg);

    // The first word is complete at the space; the second is only counted
    // when include_final is requested. A multi-code UID (4,5) is composed.
    const std::vector<int> two_words = { 4, 5, 6, 0, 4, 5, 6 };
    if (tesseract_dawg_score::word_bonus(two_words, codes, tokens, dawgs, false) != 0.25f ||
        tesseract_dawg_score::word_bonus(two_words, codes, tokens, dawgs, true) != 0.50f ||
        tesseract_dawg_score::word_bonus({ 4, 5 }, codes, tokens, dawgs, true) != 0.0f) {
        std::fprintf(stderr, "multi-code DAWG score mismatch\n");
        return 1;
    }
    std::puts("tesseract dawg score: PASS");
    return 0;
}
