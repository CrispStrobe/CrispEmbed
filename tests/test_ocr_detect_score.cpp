// Hermetic check of the DB postprocess score for thin text lines.
//
// A 5-row line at probability 0.62 on a 0.10 background loses its score to
// the one-pixel dilation ring when the ring is averaged in (issue 71 owner
// review: the first subtitle line of the DBNet street crop). With
// score_undilated the score is taken over thresholded pixels only, as in
// upstream PaddleOCR DBPostProcess, and the line is kept.
#include "ocr_detect.h"

#include <cstdio>
#include <vector>

int main() {
    const int w = 120, h = 40;
    std::vector<float> prob(size_t(w) * h, 0.10f);
    for (int y = 18; y < 23; ++y)
        for (int x = 20; x < 100; ++x) prob[size_t(y) * w + x] = 0.62f;

    int failures = 0;
    auto check = [&](bool ok, const char * what) {
        std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what);
        failures += ok ? 0 : 1;
    };
    using ocr_detect::postprocess_probability_map;
    using ocr_detect::score_mode;
    const auto legacy = postprocess_probability_map(prob.data(), h, w, 0.3f, 0.5f, 1.5f, 10, 1.0f, 1.0f, 1, 1000,
                                                    score_mode::fast, false);
    const auto fixed = postprocess_probability_map(prob.data(), h, w, 0.3f, 0.5f, 1.5f, 10, 1.0f, 1.0f, 1, 1000,
                                                   score_mode::fast, true);
    const auto undilated = postprocess_probability_map(prob.data(), h, w, 0.3f, 0.5f, 1.5f, 10, 1.0f, 1.0f, 0, 1000,
                                                       score_mode::fast, true);
    check(legacy.empty(), "dilated-ring score drops the thin line (legacy behaviour kept)");
    check(fixed.size() == 1 && fixed[0].score > 0.61f && fixed[0].score < 0.63f,
          "score_undilated keeps it with the thresholded mean");
    check(fixed.size() == 1 && fixed[0].y <= 18.0f && fixed[0].y + fixed[0].h >= 23.0f, "box covers the line");
    check(undilated.size() == 1 && undilated[0].score == fixed[0].score, "no dilation: flag is a no-op on the score");
    return failures ? 1 : 0;
}
