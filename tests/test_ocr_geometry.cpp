#include "ocr_detect.h"

#include <cassert>

int main() {
    const auto options = ocr_detect::rapid_defaults();
    assert(options.prob_threshold == 0.3f);
    assert(options.box_threshold == 0.5f);
    assert(options.target_short_side == 736);
    assert(options.max_side == 2000);
    assert(options.min_height == 30);
    assert(options.width_height_ratio == 8.0f);
    assert(options.max_candidates == 1000);
    assert(options.dilation == 1);
    assert(options.scoring == ocr_detect::score_mode::fast);
    return 0;
}
