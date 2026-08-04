// test_msac_tiling.cpp — H2OVL Multi-Scale Adaptive Cropping tile geometry.
//
// Model-free: MSAC is pure preprocessing, so the grids and tile counts can be
// checked against the upstream algorithm without loading any weights. That
// matters because a wrong tile stack does not crash — h2ovl-mississippi-2b
// simply answers with fluent nonsense, which is the failure mode least likely
// to be noticed by a smoke test that only asserts "produced some text".
//
// Expected values transcribed from H2OVL's image_process.py:
//
//   load_single_image(msac=True):
//     coarse = dynamic_preprocess(min_num=1, use_thumbnail=True)
//     fine   = dynamic_preprocess2(min_num=3, prior_aspect_ratio=coarse_grid)
//     result = fine[:-1] + coarse[:-1] + fine[-1:]
//
//   dynamic_preprocess2 keeps a candidate (cols, rows) only when
//   prior_cols % cols != 0 and prior_rows % rows != 0, and a thumbnail is
//   appended only when a pass produced more than one block — so a 1x1 pass
//   contributes nothing once its last element is dropped.
//
// The 800x800 case has no admissible fine grid at max_dynamic_patch=6 (it
// would need 3x3 = 9 blocks). Upstream would raise; we must decline rather
// than silently fall back to single-scale, which would feed the model the
// exact layout it cannot read.

#include "image_preprocess.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <vector>

namespace {

struct Case {
    int w, h;
    bool want_ok;
    int want_tiles;
    int want_rows, want_cols; // the coarse grid, reported as the page layout
    const char * note;
};

int run() {
    const Case cases[] = {
        { 606, 1000, true, 13, 3, 2, "portrait page: coarse 3x2, fine 2x3" },
        { 1000, 606, true, 13, 2, 3, "landscape: mirror of the above" },
        { 448, 448, true, 5, 1, 1, "single coarse block contributes nothing" },
        { 2000, 500, true, 11, 1, 4, "wide strip: coarse 1x4, fine 2x3" },
        { 800, 800, false, 0, 0, 0, "no admissible fine grid — must decline" },
    };

    int failures = 0;
    for (const Case & c : cases) {
        std::vector<unsigned char> img((size_t)c.w * c.h * 3, 128);

        image_preproc::internvl_config cfg;
        cfg.image_size = 448;
        cfg.min_dynamic_patch = 1;
        cfg.max_dynamic_patch = 6; // H2OVL's max_dynamic_patch
        image_preproc::internvl_result out;

        const bool ok = image_preproc::preprocess_internvl_msac_rgb(img.data(), c.h, c.w, 3, cfg, out);

        bool pass = (ok == c.want_ok);
        if (pass && ok) {
            pass = out.n_tiles == c.want_tiles && out.grid_rows == c.want_rows && out.grid_cols == c.want_cols &&
                   out.tiles.size() == (size_t)out.n_tiles * 3 * cfg.image_size * cfg.image_size;
        }
        failures += !pass;

        std::printf("%s %4dx%-4d ok=%d tiles=%2d grid=%dx%d   (want ok=%d tiles=%2d grid=%dx%d)  %s\n",
                    pass ? "PASS" : "FAIL", c.w, c.h, (int)ok, out.n_tiles, out.grid_rows, out.grid_cols,
                    (int)c.want_ok, c.want_tiles, c.want_rows, c.want_cols, c.note);
    }

    if (failures) {
        std::printf("\nFAIL: %d MSAC tiling case(s) diverge from the H2OVL reference.\n", failures);
        return 1;
    }
    std::printf("\nPASS: MSAC tile geometry matches the H2OVL reference.\n");
    return 0;
}

} // namespace

static int crispembed_test_main() {
    return run();
}

// The guard in tools/check_test_clean_exit.sh: a one-shot binary must not run
// ggml's static GPU-device destructor at exit (it aborts on Metal / faults on
// CUDA). These tests touch no GPU today, but they link crispembed-core, so the
// teardown is one added dependency away from firing.
int main() {
    core_util::clean_exit(crispembed_test_main());
}
