// Unlimited-OCR-2 tiling token arithmetic vs the reference (modeling_unlimitedocr.py).
#include "unlimited_ocr2_tiles.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                               \
            fails++;                                                                                                   \
        }                                                                                                              \
    } while (0)

int main() {
    using namespace uocr2_tiles;
    CHECK(queries_per_side(640) == 10);
    CHECK(queries_per_side(1024) == 16);
    CHECK(n_vis(1, 1, false) == 273);
    // 1024x768 -> 4x3 grid: reference projector [12,100,1280], IMG span 1503 (not 3393).
    CHECK(n_vis(4, 3, true) == 1503);
    CHECK(n_vis(4, 3, true) != 16 * 3 * (16 * 4 + 1) + 273);
    // 640 tile: SAM grid 40 (nP) vs base 64, 100 tokens -- never upsampled to 256.
    CHECK(!sam_upsample_to_base(100, 256, 40, 64, true));
    CHECK(sam_upsample_to_base(100, 256, 40, 64, false)); // legacy reduced-res path
    CHECK(!sam_upsample_to_base(256, 256, 64, 64, false));
    if (!fails) printf("test-unlimited-ocr2-tiles: OK\n");
    return fails ? 1 : 0;
}
