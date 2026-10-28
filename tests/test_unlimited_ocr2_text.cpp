// Unlimited-OCR-2 output text vs the original decode (modeling_unlimitedocr.py:1201,
// tokenizer.decode(gen, skip_special_tokens=False)). Replays the Kaggle trial's original-model
// ids -> original text byte-exactly, plus ByteLevel/UTF-8 edge cases and the clean() rendering.
#include "core/clean_exit.h"
#include "unlimited_ocr2_text.h"
#include "unlimited_ocr2_text_fixture.h"

#include <cstdio>
#include <string>
#include <vector>

static int fails = 0;
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                               \
            fails++;                                                                                                   \
        }                                                                                                              \
    } while (0)

static int crispembed_test_main() {
    using namespace uocr2_text;
    using namespace uocr2_text_fixture;
    static constexpr int kEos = 1;

    std::vector<std::string> vocab(128831);
    for (const auto & p : kPieces) vocab[(size_t)p.id] = p.text;

    // 1. Trial replay: original ids -> original decoded text, and the clean rendering.
    for (const auto & c : kCases) {
        const std::string raw = decode_raw(vocab, c.ids, c.n, kEos);
        if (raw != c.text) fprintf(stderr, "case %s raw:\n%s\n--- want:\n%s\n", c.name, raw.c_str(), c.text);
        CHECK(raw == c.text);
        CHECK(clean(raw) == c.clean);
        CHECK(clean(raw).find('<') == std::string::npos || std::string(c.name) == "table");
    }
    // The table keeps its row/cell structure in both renderings.
    {
        const auto & t = kCases[1];
        CHECK(std::string(t.name) == "table");
        const std::string raw = decode_raw(vocab, t.ids, t.n, kEos);
        CHECK(raw.find("<tr><td>Product</td><td>Region</td>") != std::string::npos);
        CHECK(clean(raw).rfind("<table><tr><td>Product</td>", 0) == 0);
    }

    // 2. EOS is never part of the text; out-of-vocab ids are skipped.
    {
        const int32_t ids[] = { 30, kEos, 32, 999999, -5 };
        CHECK(decode_raw(vocab, ids, 5, kEos) == "<>");
    }

    // 3. ByteLevel whole-token fallback: a piece with any unmapped char keeps its raw bytes,
    //    including chars that WOULD map ("\xC4\xA0" = U+0120 -> space when alone).
    {
        std::vector<std::string> v = { "\xC4\xA0", "\xC4\xA0\xEF\xBD\x9C", "\xC3\x83", "\xC2\xA9", "a" };
        const int32_t a[] = { 0, 4 };
        CHECK(decode_raw(v, a, 2, -1) == " a");
        const int32_t b[] = { 1 };
        CHECK(decode_raw(v, b, 1, -1) == "\xC4\xA0\xEF\xBD\x9C"); // not " \xEF\xBD\x9C"
        // U+00C3 -> byte 0xC3, U+00A9 -> 0xA9: together "é"; 0xC3 alone is invalid -> U+FFFD.
        const int32_t c[] = { 2, 3 };
        CHECK(decode_raw(v, c, 2, -1) == "\xC3\xA9");
        const int32_t d[] = { 2, 4 };
        CHECK(decode_raw(v, d, 2, -1) == "\xEF\xBF\xBD"
                                         "a");
    }

    // 4. from_utf8_lossy maximal-subpart substitution.
    CHECK(utf8_lossy("ok") == "ok");
    CHECK(utf8_lossy("\xE2\x82"
                     "A") == "\xEF\xBF\xBD"
                             "A");                                               // truncated 3-byte: one U+FFFD
    CHECK(utf8_lossy("\xF0\x80") == "\xEF\xBF\xBD\xEF\xBF\xBD");                 // F0 needs 90..BF
    CHECK(utf8_lossy("\xED\xA0\x80") == "\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD"); // surrogate
    CHECK(utf8_lossy("\xF0\x9F\x98\x80") == "\xF0\x9F\x98\x80");
    CHECK(utf8_lossy("\xC0\xAF") == "\xEF\xBF\xBD\xEF\xBF\xBD");

    // 5. clean(): headers, levels, multi-page, unknown <BLOCK> payloads kept.
    CHECK(clean("<PAGE><BLOCK>title<LEVEL2>[1, 2, 3, 4]<FORK>A\n<BLOCK>text[5, 6, 7, 8]<FORK>B") == "A\nB");
    CHECK(clean("<PAGE><BLOCK>text[1, 2, 3, 4]<FORK>A\n<PAGE><BLOCK>text[1, 2, 3, 4]<FORK>B") == "A\n\nB");
    CHECK(clean("<BLOCK>not a header: x<FORK>y") == "not a header: xy");
    CHECK(clean("<BLOCK>image<FORK><CHILD>z") == "z");

    // 6. Output-mode knob: raw is the default; unknown values fall back to raw.
    CHECK(parse_output_mode(nullptr) == output_mode::raw);
    CHECK(parse_output_mode("") == output_mode::raw);
    CHECK(parse_output_mode("raw") == output_mode::raw);
    CHECK(parse_output_mode("clean") == output_mode::clean);
    CHECK(parse_output_mode("markdown") == output_mode::clean);
    CHECK(parse_output_mode("legacy") == output_mode::legacy);
    CHECK(parse_output_mode("0") == output_mode::raw);

    if (!fails) printf("test-unlimited-ocr2-text: OK\n");
    return fails ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
