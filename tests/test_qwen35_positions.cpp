#include "qwen35_positions.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", value);
#else
    setenv("CRISPEMBED_QWEN35_OCR", value, 1);
#endif
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    std::ifstream input(argv[1]);
    int cases = 0;
    if (!(input >> cases) || cases < 1 || cases > 100) return 2;
    gate("1");
    bool ok = true;
    for (int i = 0; i < cases; ++i) {
        std::string name;
        int tokens, grids, expected_delta, cached_tokens;
        if (!(input >> name >> tokens >> grids >> expected_delta >> cached_tokens) || tokens < 1 || tokens > 4096 ||
            grids < 0 || grids > 16 || cached_tokens < 1 || cached_tokens > 16)
            return 2;
        std::vector<int> types(tokens), grid(3 * grids);
        std::vector<float> expected(4 * tokens), cached(3 * cached_tokens);
        for (auto & x : types)
            if (!(input >> x)) return 2;
        for (auto & x : grid)
            if (!(input >> x)) return 2;
        for (auto & x : expected)
            if (!(input >> x)) return 2;
        for (auto & x : cached)
            if (!(input >> x)) return 2;
        qwen35_ocr::position_outputs result;
        std::vector<float> next;
        const bool match = qwen35_ocr::image_positions(types, grid, 2, result) && result.generation == expected &&
                           result.rope_delta == expected_delta &&
                           qwen35_ocr::cached_positions(tokens, cached_tokens, result.rope_delta, next) &&
                           next == cached;
        std::printf("%s positions %s tokens=%d delta=%d cached=%d exact_integer_match=%d\n", match ? "PASS" : "FAIL",
                    name.c_str(), tokens, result.rope_delta, cached_tokens, match);
        ok &= match;
    }
    qwen35_ocr::position_outputs result;
    std::vector<float> next;
    const auto reject = [&](const std::vector<int> & types, const std::vector<int> & grid, int merge = 2) {
        result.axes = { 42 };
        return !qwen35_ocr::image_positions(types, grid, merge, result) && result.axes.empty() &&
               result.generation.empty() && result.rope_delta == 0;
    };
    bool controls = reject({}, {}) && reject({ 1 }, {}) && reject({ 1 }, { 1, 2, 2, 1 }) &&
                    reject({ 0 }, { 1, 2, 2 }) && reject({ 1, 1 }, { 1, 2, 2 }) && reject({ 1 }, { 1, 3, 2 }) &&
                    reject({ 1 }, { 2, 2, 2 }) && reject({ 2 }, { 1, 2, 2 }) && reject({ -1 }, {}) &&
                    reject({ 0 }, {}, 0) && reject({ 1 }, { 1, 2147483647, 2 }) &&
                    reject(std::vector<int>(4097, 0), {}) && !qwen35_ocr::cached_positions(0, 1, 0, next) &&
                    !qwen35_ocr::cached_positions(4096, 1, 0, next) && !qwen35_ocr::cached_positions(4, 1, -5, next) &&
                    !qwen35_ocr::cached_positions(4, 1, 2147483647, next);
    // Negative deltas are valid: image area can exceed max(H,W).
    controls &= qwen35_ocr::cached_positions(78, 1, -56, next) && next == std::vector<float>({ 22, 22, 22 });
    gate("0");
    controls &= reject({ 0 }, {}) && !qwen35_ocr::cached_positions(1, 1, 0, next) && next.empty();
    gate("1");
    std::printf("%s positions rejection, bounds, reset, negative delta and master=0 controls\n",
                controls ? "PASS" : "FAIL");
    return ok && controls ? 0 : 1;
}
