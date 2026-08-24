// test_lfm2_naflex.cpp — pins the LFM2.5-VL NaFlex tiling math.
//
// Golden values come from tools/lfm2_vl_tiling_oracle.py, which is HF's own
// image_processing_lfm2_vl.py functions extracted VERBATIM (pure math, no
// torch). That is what makes this a guard rather than a restatement of the
// thing under test.
//
// What it is actually guarding, and why each one is not visible to any float
// diff:
//   * the split trigger — get it wrong and a page is silently squashed into
//     one tile (or a small one is needlessly split); shapes stay valid either
//     way;
//   * banker's rounding in round_by_factor — Python's round() is half-to-EVEN
//     and std::round() is not, so a .5 flips one grid step;
//   * the equal-ratio tie-break — dropping it sends 2048x2048 to a 1x1 grid
//     instead of 3x3, i.e. 256 image tokens instead of 2304, with no error;
//   * the token-count formulas that the prompt markup is built from.
//
// Hermetic and weight-free: no model, no ggml, no image.

#include "../src/lfm2_naflex.h"

#include <cstdio>
#include <cstdlib>

static int g_failures = 0;

static void check_int(const char * what, int w, int h, int got, int want) {
    if (got != want) {
        std::printf("  FAIL %-22s %dx%d: got %d, want %d\n", what, w, h, got, want);
        g_failures++;
    }
}

int main() {
    const lfm2_naflex::params p;

    // { width, height, too_large, grid_w, grid_h, smart_w, smart_h }
    // Produced by tools/lfm2_vl_tiling_oracle.py. grid_* is only meaningful
    // when too_large is 1; the oracle reports 1x1 otherwise.
    struct golden {
        int w, h, big, gw, gh, sw, sh;
    };
    static const golden cases[] = {
        { 500, 650, 0, 1, 1, 448, 576 },   // the validated single-tile fixture
        { 150, 200, 0, 1, 1, 224, 320 },   // thumbnail: must NOT be upscaled to the full budget
        { 300, 1000, 0, 1, 1, 256, 928 },  // tall strip
        { 3000, 4000, 1, 2, 3, 416, 576 }, // A4 at 300 dpi
        { 1000, 300, 0, 1, 1, 928, 256 },  // wide banner
        { 2048, 2048, 1, 3, 3, 512, 512 }, // square: the tie-break case
        { 1000, 1000, 1, 2, 2, 512, 512 }, // square, one grid step smaller
        { 1024, 1024, 1, 2, 2, 512, 512 },
        { 1700, 2200, 1, 2, 3, 448, 576 },  // US letter at 200 dpi
        { 4000, 1000, 1, 4, 1, 1024, 256 }, // panorama
        { 2000, 1000, 1, 4, 2, 704, 352 },
        { 1920, 2485, 1, 2, 3, 448, 576 }, // tests/regression/.../commons_test_ocr_document.jpg
        { 1920, 2518, 1, 2, 3, 416, 576 }, // .../german_official_print.jpg
        { 768, 1552, 1, 2, 4, 352, 704 },  // .../receipt_historical.png
        { 452, 317, 0, 1, 1, 448, 320 },   // .../simple_form.png
        { 513, 513, 0, 1, 1, 512, 512 },   // just under the tolerance
        { 1, 1, 0, 1, 1, 256, 256 },       // degenerate: min band, no divide-by-zero
        { 31, 31, 0, 1, 1, 256, 256 },
        { 33, 17, 0, 1, 1, 384, 192 },
    };

    std::printf("lfm2 naflex tiling: %zu golden layouts\n", sizeof(cases) / sizeof(cases[0]));
    for (const auto & c : cases) {
        check_int("too_large", c.w, c.h, (int)lfm2_naflex::is_image_too_large(c.h, c.w, p), c.big);

        int sw = 0, sh = 0;
        lfm2_naflex::smart_resize(c.h, c.w, p, &sw, &sh);
        check_int("smart_resize.w", c.w, c.h, sw, c.sw);
        check_int("smart_resize.h", c.w, c.h, sh, c.sh);

        // Both sides must be divisible by patch_size * downsample_factor, or
        // the projector's pixel_unshuffle drops the last row/column of patches.
        check_int("smart_w % 32", c.w, c.h, sw % (p.encoder_patch_size * p.downsample_factor), 0);
        check_int("smart_h % 32", c.w, c.h, sh % (p.encoder_patch_size * p.downsample_factor), 0);

        if (c.big) {
            int gw = 0, gh = 0;
            lfm2_naflex::grid_layout(c.h, c.w, p, &gw, &gh);
            check_int("grid_w", c.w, c.h, gw, c.gw);
            check_int("grid_h", c.w, c.h, gh, c.gh);
            check_int("grid tiles <= max", c.w, c.h, gw * gh <= p.max_tiles ? 1 : 0, 1);
        }
    }

    // Token counts the prompt markup is built from.
    check_int("tokens_per_tile", 512, 512, lfm2_naflex::tokens_per_tile(p), 256);
    check_int("tokens_for_image", 448, 576, lfm2_naflex::tokens_for_image(576, 448, p), 252);
    check_int("tokens_for_image", 448, 320, lfm2_naflex::tokens_for_image(320, 448, p), 140);

    // The whole prompt budget for the multi-tile reference fixture: 2x3 tiles
    // plus a 448x576 thumbnail is exactly the 1788 image tokens the reference's
    // own metadata records.
    check_int("image tokens (1920x2485)", 1920, 2485,
              6 * lfm2_naflex::tokens_per_tile(p) + lfm2_naflex::tokens_for_image(576, 448, p), 1788);

    // round_by_factor must be half-to-EVEN, like Python's round().
    check_int("round_by_factor 48/32", 0, 0, lfm2_naflex::round_by_factor(48, 32), 64);
    check_int("round_by_factor 16/32", 0, 0, lfm2_naflex::round_by_factor(16, 32), 0);
    check_int("round_by_factor 80/32", 0, 0, lfm2_naflex::round_by_factor(80, 32), 64);
    check_int("round_by_factor 112/32", 0, 0, lfm2_naflex::round_by_factor(112, 32), 128);

    if (g_failures) {
        std::printf("FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
