#include "tesseract_recoder.h"

#include <cstdio>
#include <vector>

static bool equal(const std::vector<int> & a, const std::vector<int> & b) { return a == b; }

int main() {
    const std::vector<std::vector<int>> codes = { { 0 }, { 1, 2 }, { 3 } };
    std::vector<int> unichars, starts;
    if (!tesseract_recoder::compose_classes({ 0, 1, 2, 3 }, codes, unichars, starts) ||
        !equal(unichars, { 0, 1, 2 }) || !equal(starts, { 0, 1, 3 })) {
        std::fprintf(stderr, "exact recoder composition failed\n");
        return 1;
    }
    unichars.clear();
    starts.clear();
    if (tesseract_recoder::compose_classes({ 0, 2 }, codes, unichars, starts)) {
        std::fprintf(stderr, "invalid recoder composition unexpectedly succeeded\n");
        return 1;
    }
    const std::vector<std::vector<int>> ambiguous = { { 4 }, { 4, 5 }, { 5 } };
    unichars.clear();
    starts.clear();
    if (!tesseract_recoder::compose_classes({ 4, 5 }, ambiguous, unichars, starts) ||
        !equal(unichars, { 1 }) || !equal(starts, { 0 })) {
        std::fprintf(stderr, "ambiguous recoder composition failed\n");
        return 1;
    }
    std::puts("tesseract recoder: PASS");
    return 0;
}
