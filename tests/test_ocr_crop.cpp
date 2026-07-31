#include "ocr_crop.h"
#include "core/clean_exit.h"

#include <cassert>

static int crispembed_test_main() {
    const uint8_t pixels[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    }; // 3x2, RGB
    int w = 0, h = 0;
    auto crop = ocr_crop::extract(pixels, 2, 3, 3, 1, 1, 1, 1, 1, &w, &h);
    assert(w == 2 && h == 3);
    assert(crop.size() == 18);
    assert(crop[0] == 0 && crop[1] == 1 && crop[2] == 2);

    auto clipped = ocr_crop::extract(pixels, 2, 3, 3, 1, 2, 4, 4, 2, &w, &h);
    assert(w == 2 && h == 3);
    assert(clipped.size() == 18);
    assert(!ocr_crop::orient_180_rgb(clipped, w, h));
    std::vector<uint8_t> gray((size_t)w * h, 255);
    assert(!ocr_crop::orient_180_gray(gray, w, h));
    return 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
