#include "tesseract_recoder.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <vector>

static bool equal(const std::vector<int> & a, const std::vector<int> & b) {
    return a == b;
}

static int test_main() {
    const std::vector<std::vector<int>> codes = { { 0 }, { 1, 2 }, { 3 } };
    std::vector<int> unichars, starts;
    if (!tesseract_recoder::compose_classes({ 0, 1, 2, 3 }, codes, unichars, starts) || !equal(unichars, { 0, 1, 2 }) ||
        !equal(starts, { 0, 1, 3 })) {
        std::fprintf(stderr, "exact recoder composition failed\n");
        return 1;
    }
    unichars = { 99 };
    starts = { 77 };
    if (tesseract_recoder::compose_classes({ 0, 2 }, codes, unichars, starts)) {
        std::fprintf(stderr, "invalid recoder composition unexpectedly succeeded\n");
        return 1;
    }
    if (!unichars.empty() || !starts.empty()) {
        std::fprintf(stderr, "failed composition retained stale output\n");
        return 1;
    }
    const std::vector<std::vector<int>> ambiguous = { { 4 }, { 4, 5 }, { 5 } };
    unichars = { 99 };
    starts = { 77 };
    if (!tesseract_recoder::compose_classes({ 4, 5 }, ambiguous, unichars, starts) || !equal(unichars, { 1 }) ||
        !equal(starts, { 0 })) {
        std::fprintf(stderr, "ambiguous recoder composition failed\n");
        return 1;
    }
    if (!tesseract_recoder::compose_classes_partial({ 0, 9, 1, 2, 3 }, codes, unichars, starts) ||
        !equal(unichars, { 0, -1, 1, 2 }) || !equal(starts, { 0, 1, 2, 4 })) {
        std::fprintf(stderr, "partial recoder composition failed\n");
        return 1;
    }
    const std::vector<std::vector<int>> beam_codes = { { 4, 5 }, { 6 } };
    if (!tesseract_recoder::prefix_legal({ 4 }, beam_codes, true) ||
        !tesseract_recoder::prefix_legal({ 4, 5 }, beam_codes, false) ||
        !tesseract_recoder::prefix_legal({ 6 }, beam_codes, false) ||
        tesseract_recoder::prefix_legal({ 4 }, beam_codes, false) ||
        tesseract_recoder::prefix_legal({ 4, 7 }, beam_codes, true)) {
        std::fprintf(stderr, "recoder beam prefix legality failed\n");
        return 1;
    }
    std::puts("tesseract recoder: PASS");
    return 0;
}

int main() {
    core_util::clean_exit(test_main());
}
