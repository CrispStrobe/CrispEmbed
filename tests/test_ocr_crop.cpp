#include "ocr_crop.h"
#include "core/clean_exit.h"

#include <cassert>

static int crispembed_test_main() {
    const uint8_t pixels[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
    }; // 3x2, RGB
    int w = 0, h = 0, channels = 0;
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

    ocr_crop::prepare_options prep;
    prep.target_height = 4;
    prep.max_width = 6;
    prep.grayscale = true;
    auto prepared = ocr_crop::prepare(pixels, 2, 3, 3, prep, &w, &h, &channels);
    assert(w == 3 && h == 4 && channels == 1);
    assert(prepared.size() == 12);

    const float qx[4] = { 5, 15, 15, 5 };
    const float qy[4] = { 6, 6, 10, 10 };
    std::vector<uint8_t> quad_pixels(20 * 20, 128);
    auto quad = ocr_crop::extract_quad(quad_pixels.data(), 20, 20, 1, qx, qy, 2, &w, &h);
    assert(w == 14 && h == 8);
    assert(quad.size() == (size_t)w * h);

    prep = {};
    prep.target_width = 6;
    prep.target_height = 6;
    prep.mode = ocr_crop::resize_mode::stretch;
    prep.pad_to_target = true;
    auto stretched = ocr_crop::prepare(pixels, 2, 3, 3, prep, &w, &h, &channels);
    assert(w == 6 && h == 6 && channels == 3);
    assert(stretched.size() == 108);
    return 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
