// Complete-profile model arithmetic proof; no checkpoint or activation input.
#include "lighton4b_vision_bf16.cpp"
#include "../tools/ci-heavy/reference/lighton4b_rotary_table_profile.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace qwen35_ocr;
static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_TABLE", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_TABLE", value, 1);
    else
        unsetenv("CRISPEMBED_LIGHTON4B_VISION_BF16_ROTARY_TABLE");
#endif
}
int main(int argc, char ** argv) {
    if (argc != 4) return 2;
    if (!source_avx2_softmax_supported()) throw std::runtime_error("source-compatible CPU profile required");
    vision_config c;
    c.width = 1024;
    c.heads = 16;
    c.merge = 2;
    c.theta = 10000;
    std::vector<int> rows{ 0, 0, 1, 1 }, cols{ 0, 1, 0, 1 };
    int controls = 0;
    auto rejected = [&](const vision_config & cfg, int tokens, const std::vector<int> & r, const std::vector<int> & col,
                        bool cap) {
        std::vector<float> co{ 123 }, si{ 456 };
        if (lighton4b_rotary_profile::lookup(cfg, tokens, r, col, cap, co, si) || co != std::vector<float>{ 123 } ||
            si != std::vector<float>{ 456 })
            throw std::runtime_error("unsupported request modified output");
        ++controls;
    };
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        gate(value);
        rejected(c, 4, rows, cols, true);
    }
    gate("1");
    rejected(c, 4, rows, cols, false);
    auto bad = c;
    bad.theta = std::nextafter(10000.0f, INFINITY);
    rejected(bad, 4, rows, cols, true);
    bad = c;
    bad.theta = NAN;
    rejected(bad, 4, rows, cols, true);
    bad = c;
    bad.width = 2048;
    rejected(bad, 4, rows, cols, true);
    bad = c;
    bad.heads = 32;
    rejected(bad, 4, rows, cols, true);
    bad = c;
    bad.merge = 1;
    rejected(bad, 4, rows, cols, true);
    rejected(c, 0, rows, cols, true);
    rejected(c, 3, rows, cols, true);
    rejected(c, 4100, rows, cols, true);
    auto badrows = rows;
    badrows[0] = -1;
    rejected(c, 4, badrows, cols, true);
    badrows[0] = 1024;
    rejected(c, 4, badrows, cols, true);
    rejected(c, 4, {}, cols, true);
    rejected(c, 4, rows, {}, true);
    int rounding = std::fegetround();
    std::fesetround(FE_DOWNWARD);
    rejected(c, 4, rows, cols, true);
    std::fesetround(rounding);
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", "1");
    _putenv_s("CRISPEMBED_OVISOCR2_FRONTEND", "1");
    _putenv_s("CRISPEMBED_OVISOCR2_INFERENCE", "1");
    _putenv_s("CRISPEMBED_OVISOCR2_LARGE_IMAGES", "1");
#else
    for (const char * name : { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_OVISOCR2_FRONTEND", "CRISPEMBED_OVISOCR2_INFERENCE",
                               "CRISPEMBED_OVISOCR2_LARGE_IMAGES" })
        setenv(name, "1", 1);
#endif
    std::ifstream cases(argv[1]);
    std::ofstream output(argv[2], std::ios::binary);
    int count = 0, tokens, triples;
    while (cases >> tokens >> triples) {
        std::vector<int> grid(triples * 3);
        for (int & value : grid)
            if (!(cases >> value)) throw std::runtime_error("case truncated");
        std::vector<int> r, col, offsets, heights, widths;
        if (!geometry(c, tokens, grid, r, col, offsets, heights, widths))
            throw std::runtime_error("admitted geometry rejected");
        std::vector<float> co, si;
        if (!lighton4b_rotary_profile::lookup(c, tokens, r, col, true, co, si))
            throw std::runtime_error("supported lookup rejected");
        output.write(reinterpret_cast<const char *>(co.data()), co.size() * 4);
        output.write(reinterpret_cast<const char *>(si.data()), si.size() * 4);
        ++count;
    }
    if (!output || !cases.eof()) return 3;
    std::ofstream raw(argv[3], std::ios::binary);
    raw.write(reinterpret_cast<const char *>(lighton4b_rotary_profile::rotary_bits.data()),
              sizeof(lighton4b_rotary_profile::rotary_bits));
    if (!raw) return 4;
    std::cout << "{\"case_count\":" << count << ",\"fallback_controls\":" << controls
              << ",\"constant_bytes\":" << sizeof(lighton4b_rotary_profile::rotary_bits)
              << ",\"product_mkl_dependency\":false}\n";
    return 0;
}
