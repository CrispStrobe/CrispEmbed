// MiniCPM-OCR-1.0 slicing/grid/token arithmetic (src/minicpm_ocr_tiles.h).
// Expected values are the MiniCPM-V processor algorithm evaluated in Python
// (find_best_resize / get_sliced_grid / get_refine_size with 28-px snapping) on
// the five frozen reference-case sizes. They pin the C++ port to that algorithm;
// they are NOT yet confirmed by the hosted reference capture (spec "verify" list).
#include "core/clean_exit.h"
#include "minicpm_ocr_tiles.h"

#include <cstdio>

static int fails = 0;
#define CHECK(c)                                                                                                       \
    do {                                                                                                               \
        if (!(c)) {                                                                                                    \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);                                               \
            fails++;                                                                                                   \
        }                                                                                                              \
    } while (0)

static int crispembed_test_main() {
    using namespace minicpm_ocr_tiles;
    const config c;
    CHECK(py_round(2.5) == 2 && py_round(3.5) == 4 && py_round(-0.5) == 0);
    CHECK(ensure_divide(10, 28) == 28);
    CHECK(ensure_divide(1321, 28) == 1316);

    plan p;
    // line 560x64: unsliced, upscaled overview 1316x140 -> 47*5 tokens.
    CHECK(make_plan(560, 64, c, p));
    CHECK(p.grid.w == 0 && p.parts.size() == 1);
    CHECK(p.overview.w == 1316 && p.overview.h == 140 && p.total_tokens == 235);
    // tiny 60x30: unsliced, upscaled to 644x308.
    CHECK(make_plan(60, 30, c, p));
    CHECK(p.overview.w == 644 && p.overview.h == 308 && p.total_tokens == 23 * 11);
    // para/table 1024x768: 2x2 grid, overview 504x392, refine 1008x784.
    CHECK(make_plan(1024, 768, c, p));
    CHECK(p.grid.w == 2 && p.grid.h == 2 && p.parts.size() == 5);
    CHECK(p.overview.w == 504 && p.overview.h == 392);
    CHECK(p.refine.w == 1008 && p.refine.h == 784);
    CHECK(p.parts[4].row == 1 && p.parts[4].col == 1 && p.parts[4].x0 == 504 && p.parts[4].y0 == 392);
    CHECK(p.total_tokens == 5 * 252);
    // tall 700x2100: multiple 8 -> 2 cols x 4 rows, overview 252x784, refine 728x2240.
    CHECK(make_plan(700, 2100, c, p));
    CHECK(p.grid.w == 2 && p.grid.h == 4 && p.parts.size() == 9);
    CHECK(p.overview.w == 252 && p.overview.h == 784);
    CHECK(p.refine.w == 728 && p.refine.h == 2240);
    CHECK(p.parts[1].size.w == 364 && p.parts[1].size.h == 560 && p.parts[1].tokens == 260);
    CHECK(p.total_tokens == 9 * 28 + 8 * 260); // overview 18x56 patches -> 252
    // max_slice_nums caps the multiple; slice_mode off never slices.
    config off = c;
    off.slice_mode = false;
    CHECK(make_plan(4000, 4000, off, p) && p.parts.size() == 1);
    CHECK(sliced_grid(4000, 4000, c).w * sliced_grid(4000, 4000, c).h <= 9);
    CHECK(!make_plan(0, 10, c, p));

    // Bucketed 70x70 positions (torch double-arange semantics).
    const int b10[] = { 0, 7, 14, 21, 28, 35, 42, 49, 56, 63 };
    const auto k10 = bucket_coords(10, 70);
    for (int i = 0; i < 10; ++i) CHECK(k10[i] == b10[i]);
    const int b32[] = { 0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 24 };
    const auto k32 = bucket_coords(32, 70);
    for (int i = 0; i < 12; ++i) CHECK(k32[i] == b32[i]);
    const auto ids = position_ids(2, 3, 70);
    CHECK(ids.size() == 6 && ids[0] == 0 && ids[3] == 35 * 70);
    const auto full = bucket_coords(70, 70);
    for (int i = 0; i < 70; ++i) CHECK(full[i] == i);

    if (!fails) printf("test-minicpm-ocr-tiles: OK\n");
    return fails ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
