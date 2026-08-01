#include "easyocr_postprocess.h"
#include "core/clean_exit.h"

#include <cmath>
#include <cstdio>

static int crispembed_test_main() {
    const std::vector<std::string> vocabulary = { "a", "b", "c" };
    std::string decoded;
    int invalid = -1;
    if (!easyocr_postprocess::ctc_greedy_decode({ 0, 1, 1, 0, 2, 2, 0, 3 }, vocabulary, &decoded, &invalid)) return 1;
    if (decoded != "abc" || invalid != -1) return 2;
    if (easyocr_postprocess::ctc_greedy_decode({ 1, 4 }, vocabulary, &decoded, &invalid) || invalid != 4) return 3;
    const float confidence = easyocr_postprocess::confidence_custom_mean({ 0.8f, 0.9f });
    const float expected = std::pow(0.8f * 0.9f, 2.0f / std::sqrt(2.0f));
    if (std::fabs(confidence - expected) > 1e-6f || easyocr_postprocess::confidence_custom_mean({}) != 0.0f) return 4;
    std::string error;
    if (!easyocr_postprocess::validate_vocabulary(vocabulary, &error) ||
        easyocr_postprocess::validate_vocabulary({ "a", "a" }, &error) ||
        easyocr_postprocess::validate_vocabulary({ "" }, &error) ||
        !easyocr_postprocess::vocabulary_contains(vocabulary, "b") ||
        easyocr_postprocess::vocabulary_contains(vocabulary, "z"))
        return 5;
    std::printf("easyocr-postprocess PASS ctc=abc confidence=%.6f\n", confidence);
    return 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
