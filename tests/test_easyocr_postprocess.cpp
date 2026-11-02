#include "easyocr_postprocess.h"
#include "ocr_reading_order.h"
#include "core/clean_exit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct test_box {
    float x, y, w, h;
};
struct test_region {
    test_box box;
};

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
    // EasyOCR compute_ratio_and_resize: int(64 * (w / h)); legacy rounded up.
    if (easyocr_postprocess::recognizer_resized_width(451, 51, true) != 565 ||
        easyocr_postprocess::recognizer_resized_width(451, 51, false) != 566 ||
        easyocr_postprocess::recognizer_resized_width(486, 48, true) != 648 ||
        easyocr_postprocess::recognizer_resized_width(10, 3, true) != 213)
        return 6;
    // Values from easyocr.recognition.adjust_contrast_grey(target=0.5).
    std::vector<uint8_t> low = { 100, 102, 104, 110, 120, 121, 122, 130, 131, 140, 150, 151 };
    const std::vector<uint8_t> low_expected = { 97, 105, 114, 140, 182, 187, 191, 225, 229, 255, 255, 255 };
    std::vector<uint8_t> high = { 0, 255, 0, 255, 10, 240 };
    const std::vector<uint8_t> high_expected = high;
    if (!easyocr_postprocess::adjust_contrast_grey(low, 0.5) || low != low_expected ||
        easyocr_postprocess::adjust_contrast_grey(high, 0.5) || high != high_expected)
        return 7;
    // cv2.imread(IMREAD_GRAYSCALE) of an RGB PNG holding these three pixels.
    if (easyocr_postprocess::grey_libpng(10, 200, 30) != 123 ||
        easyocr_postprocess::grey_libpng(255, 255, 255) != 255 || easyocr_postprocess::grey_libpng(123, 45, 67) != 70)
        return 8;
    // scan_page_pd lines 29-33 (CLI boxes): tightly spaced lines the legacy
    // comparator ordered differently on libc++ and libstdc++. Every input
    // permutation must give the same top-to-bottom order.
    const std::vector<test_region> lines = {
        { { 125.5f, 722.8f, 421.2f, 28.4f } }, { { 82.7f, 740.7f, 463.3f, 34.0f } },
        { { 84.0f, 762.0f, 465.7f, 31.4f } },  { { 81.1f, 785.1f, 468.1f, 32.7f } },
        { { 300.0f, 786.0f, 50.0f, 30.0f } },
    };
    const std::vector<int> expected_order = { 0, 1, 2, 3, 4 };
    std::vector<int> perm = { 0, 1, 2, 3, 4 };
    do {
        std::vector<test_region> shuffled;
        for (int i : perm) shuffled.push_back(lines[(size_t)i]);
        const auto order = ocr_reading_order::rows(shuffled);
        std::vector<int> original;
        for (int i : order) original.push_back(perm[(size_t)i]);
        if (original != expected_order) return 9;
    } while (std::next_permutation(perm.begin(), perm.end()));
    std::printf("easyocr-postprocess PASS ctc=abc confidence=%.6f\n", confidence);
    return 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
