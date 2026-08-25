// test_lfm2_tiling.cpp — hermetic guard for LFM2.5-VL multi-tile NaFlex.
//
// Written BEFORE the multi-tile implementation and watched to fail, per
// dev-guide HARD RULE 2c. No weights, no model, no image decode, no network —
// pure integer layout math, so it runs in microseconds and belongs in the
// model-free CI tier.
//
// What it pins, and why each one is a defect that SHAPES ALONE WOULD ALLOW:
//
//  1. round_by_factor is BANKER'S rounding. Python's round() is half-to-EVEN;
//     C++ std::round is half-away-from-zero. They disagree whenever
//     value/factor lands on k + 0.5 for even k. This is not a rounding nicety:
//     at 144x4000 the two answers are "one tile, 252 image tokens" and "a 1x10
//     split, 2812 image tokens". Checked twice over — directly against every
//     value in [0, 4096] where the two rules differ (an exact algebraic
//     invariant, no tolerance), and end-to-end through four layout cases
//     chosen because the whole pipeline diverges on them.
//
//  2. The split trigger, the grid search and its tie-break, the whole-image
//     target size, and the thumbnail size, against golden vectors produced by
//     transformers' own functions (tools/lfm2_vl_tiling_oracle.py).
//
//  3. The image-token COUNT per tile and for the thumbnail. `ceil`, not floor:
//     the processor rounds the patch grid UP here while the projector's
//     pixel_unshuffle rounds DOWN. With the correct resize factor the grid is
//     always even and the two agree — this pins that they still agree.
//
//  4. The token-ID markup: the <|img_row_R_col_C|> / <|img_thumbnail|> /
//     <image> sequence. This is squarely in the diff harness's blind zone
//     (HARD RULE 3b) — a wrong id table produces byte-identical logits from a
//     wrong prompt, and reads as "the model is weak on multi-tile" rather than
//     as a bug. Pinned as an arithmetic law (contiguous ids) AND as an exact
//     emitted sequence.
//
//  5. Which side of the row/col SWAP we are on. `crop_image_to_patches`
//     returns (images, grid_width, grid_height) in every transformers version,
//     but `resize_and_split` unpacks it as (images, num_rows, num_cols) up to
//     4.57.x and (images, num_cols, num_rows) from 5.0 — so the old form
//     transposed the labels on any non-square grid. 5.x is geometric and is
//     what the shipped model is prompted with; that is the default and what
//     these vectors pin. The legacy variant stays behind a gate and is pinned
//     here too, so the two cannot silently converge.
//
//     This is not hypothetical: the port shipped the 4.57.x behaviour first,
//     because 4.57.6 was what happened to be installed, and prompt-token
//     parity against a transformers 5.x reference caught it — 4 of 1816 ids,
//     first at 519.

#include "core/clean_exit.h"
#include "lfm2_vl_tiling.h"

#include "lfm2_tiling_golden.h"
#include "lfm2_posembed_golden.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool ok, const char * what) {
    g_checks++;
    if (!ok) {
        printf("  FAIL %s\n", what);
        g_failures++;
    }
}

void check_eq(int got, int want, const char * what, const char * ctx) {
    g_checks++;
    if (got != want) {
        printf("  FAIL %-28s %-24s got %d, want %d\n", ctx, what, got, want);
        g_failures++;
    }
}

// The naive implementation this guard exists to reject.
int naive_round_by_factor(int value, int factor) {
    return (int)std::round((double)value / (double)factor) * factor;
}

} // namespace

static int crispembed_test_main() {
    using namespace lfm2_vl_tiling;

    const config cfg; // the shipped processor_config.json of LFM2.5-VL-3B

    printf("LFM2.5-VL multi-tile NaFlex layout guard\n");

    // ── 1. banker's rounding, as an exact algebraic invariant ──────────────
    //
    // Every value in [0, 4096] where half-to-even and half-away-from-zero
    // disagree at factor 32. No signal, no tolerance: the two rules either
    // agree with Python or they do not.
    {
        int n_diff = 0;
        for (int i = 0; i < lfm2_tiling_golden::kNumRoundCases; i++) {
            const auto & rc = lfm2_tiling_golden::kRoundCases[i];
            check_eq(round_by_factor(rc.value, rc.factor), rc.expected, "round_by_factor", "banker's");
            // Confirm the case is load-bearing: the naive rule must MISS it,
            // or the guard is wider than the defect and proves nothing.
            if (naive_round_by_factor(rc.value, rc.factor) != rc.expected) n_diff++;
        }
        check(n_diff == lfm2_tiling_golden::kNumRoundCases,
              "every pinned rounding case actually separates the two rules");
        printf("  %d rounding cases, all separating half-to-even from std::round\n",
               lfm2_tiling_golden::kNumRoundCases);

        // And the half-to-even rule itself, stated directly rather than by table.
        check_eq(round_half_to_even(0.5), 0, "round_half_to_even(0.5)", "law");
        check_eq(round_half_to_even(1.5), 2, "round_half_to_even(1.5)", "law");
        check_eq(round_half_to_even(2.5), 2, "round_half_to_even(2.5)", "law");
        check_eq(round_half_to_even(3.5), 4, "round_half_to_even(3.5)", "law");
        check_eq(round_half_to_even(-0.5), 0, "round_half_to_even(-0.5)", "law");
        check_eq(round_half_to_even(-2.5), -2, "round_half_to_even(-2.5)", "law");
        check_eq(round_half_to_even(2.4999), 2, "round_half_to_even(2.4999)", "law");
        check_eq(round_half_to_even(2.5001), 3, "round_half_to_even(2.5001)", "law");
    }

    // ── 1b. the target-ratio ORDER, which decides every tie ────────────────
    {
        int n = 0;
        const ratio * r = default_target_ratios(&n);
        check_eq(n, lfm2_tiling_golden::kNumTargetRatios, "target ratio count", "ratios");
        for (int i = 0; i < n && i < lfm2_tiling_golden::kNumTargetRatios; i++) {
            check_eq(r[i].w, lfm2_tiling_golden::kTargetRatios[i].w, "ratio w", "ratios");
            check_eq(r[i].h, lfm2_tiling_golden::kTargetRatios[i].h, "ratio h", "ratios");
        }
        // Every candidate is admissible under the tile budget.
        for (int i = 0; i < n; i++)
            check(r[i].w * r[i].h >= cfg.min_tiles && r[i].w * r[i].h <= cfg.max_tiles, "ratio within tile budget");
    }

    // ── 2-3. the golden layout table ───────────────────────────────────────
    for (int i = 0; i < lfm2_tiling_golden::kNumCases; i++) {
        const auto & g = lfm2_tiling_golden::kCases[i];
        char ctx[96];
        snprintf(ctx, sizeof(ctx), "%dx%d", g.width, g.height);

        const layout L = compute_layout(g.width, g.height, cfg);

        check_eq(L.split ? 1 : 0, g.split, "split", ctx);
        check_eq(L.grid_w, g.grid_w, "grid_w", ctx);
        check_eq(L.grid_h, g.grid_h, "grid_h", ctx);
        check_eq(L.rows, g.rows, "rows", ctx);
        check_eq(L.cols, g.cols, "cols", ctx);
        check_eq(L.n_tiles, g.n_tiles, "n_tiles", ctx);
        check_eq(L.n_images, g.n_images, "n_images", ctx);
        check_eq(L.has_thumb ? 1 : 0, g.has_thumb, "has_thumb", ctx);
        check_eq(L.target_w, g.target_w, "target_w", ctx);
        check_eq(L.target_h, g.target_h, "target_h", ctx);
        check_eq(L.resized_w, g.resized_w, "resized_w", ctx);
        check_eq(L.resized_h, g.resized_h, "resized_h", ctx);
        check_eq(L.tile_tokens, g.tile_tokens, "tile_tokens", ctx);
        check_eq(L.thumb_tokens, g.thumb_tokens, "thumb_tokens", ctx);
        check_eq(L.total_tokens, g.total_tokens, "total_tokens", ctx);

        // Structural invariants the golden table must not be able to violate.
        check(L.n_tiles == L.grid_w * L.grid_h, "n_tiles == grid_w * grid_h");
        check(L.n_images == L.n_tiles + (L.has_thumb ? 1 : 0), "n_images == n_tiles + thumb");
        check(!L.split || (L.target_w == cfg.tile_size * L.grid_w && L.target_h == cfg.tile_size * L.grid_h),
              "split target is an exact whole number of tiles");
        check(L.n_tiles >= cfg.min_tiles && L.n_tiles <= cfg.max_tiles, "tile count within [min_tiles, max_tiles]");
        // The resize factor exists precisely so the patch grid stays even; an
        // odd grid is silently truncated by the projector's pixel_unshuffle.
        check(L.resized_w % (cfg.patch_size * cfg.downsample_factor) == 0 &&
                  L.resized_h % (cfg.patch_size * cfg.downsample_factor) == 0,
              "smart_resize output is divisible by patch_size * downsample_factor");
    }
    printf("  %d layout cases pinned against the transformers oracle\n", lfm2_tiling_golden::kNumCases);

    // ── 4-5. token markup ──────────────────────────────────────────────────
    const token_ids tok; // the ids shipped in the GGUF vocab

    // The contiguity law, stated as arithmetic rather than as a 100-entry table.
    for (int r = 1; r <= 10; r++) {
        for (int c = 1; c <= 10; c++) {
            check_eq(tok.row_col(r, c), 124908 + (r - 1) * 10 + (c - 1), "row_col id", "law");
        }
    }
    check_eq(tok.thumbnail, tok.row_col(10, 10) + 1, "thumbnail follows the 100 row/col ids", "law");
    check_eq(tok.image, 124907, "<image>", "law");
    check_eq(tok.image_start, 125009, "<|image_start|>", "law");
    check_eq(tok.image_end, 125010, "<|image_end|>", "law");
    // Every id distinct and every one reachable — the property a typo breaks.
    {
        std::vector<int32_t> all;
        for (int r = 1; r <= 10; r++)
            for (int c = 1; c <= 10; c++) all.push_back(tok.row_col(r, c));
        all.push_back(tok.thumbnail);
        all.push_back(tok.image);
        all.push_back(tok.image_start);
        all.push_back(tok.image_end);
        bool distinct = true;
        for (size_t a = 0; a < all.size() && distinct; a++)
            for (size_t b = a + 1; b < all.size(); b++)
                if (all[a] == all[b]) {
                    distinct = false;
                    break;
                }
        check(distinct, "all image special-token ids are distinct");
    }

    for (int i = 0; i < lfm2_tiling_golden::kNumCases; i++) {
        const auto & g = lfm2_tiling_golden::kCases[i];
        char ctx[96];
        snprintf(ctx, sizeof(ctx), "%dx%d markup", g.width, g.height);

        const layout L = compute_layout(g.width, g.height, cfg);
        std::vector<int32_t> ids;
        build_image_markup(L, tok, ids);

        // The markup must contain exactly total_tokens <image> placeholders —
        // this is what the splice loop consumes one projector row per.
        int n_image = 0;
        for (int32_t id : ids)
            if (id == tok.image) n_image++;
        check_eq(n_image, g.total_tokens, "<image> count", ctx);
        check_eq((int)ids.size(), g.total_tokens + g.n_labels + (g.has_thumb ? 1 : 0) + 2, "markup length", ctx);
        check_eq(ids.front(), tok.image_start, "opens with <|image_start|>", ctx);
        check_eq(ids.back(), tok.image_end, "closes with <|image_end|>", ctx);

        // The exact label sequence, in order, with the right run length after each.
        size_t p = 1;
        for (int k = 0; k < g.n_labels; k++) {
            check_eq(ids[p], tok.row_col(g.labels[k].row, g.labels[k].col), "tile label", ctx);
            p++;
            for (int t = 0; t < g.tile_tokens; t++) {
                check_eq(ids[p], tok.image, "tile run", ctx);
                p++;
            }
        }
        if (g.has_thumb) {
            check_eq(ids[p], tok.thumbnail, "<|img_thumbnail|>", ctx);
            p++;
            for (int t = 0; t < g.thumb_tokens; t++) {
                check_eq(ids[p], tok.image, "thumbnail run", ctx);
                p++;
            }
        } else if (!g.split) {
            // Single-tile: a bare run of <image>, no row/col prefixes at all.
            for (int t = 0; t < g.total_tokens; t++) {
                check_eq(ids[p], tok.image, "single-tile run", ctx);
                p++;
            }
        }
        check_eq((int)p, (int)ids.size() - 1, "markup fully consumed", ctx);
    }
    printf("  %d markup sequences pinned\n", lfm2_tiling_golden::kNumCases);

    // ── 5b. the legacy-swap variant is a DIFFERENT sequence ────────────────
    //
    // If the two ever agree on a non-square grid, one of them has silently
    // adopted the other and the gate is measuring nothing.
    {
        config geo = cfg;
        geo.legacy_label_swap = true;
        int n_nonsquare = 0, n_differ = 0;
        for (int i = 0; i < lfm2_tiling_golden::kNumCases; i++) {
            const auto & g = lfm2_tiling_golden::kCases[i];
            if (!g.split || g.grid_w == g.grid_h) continue;
            n_nonsquare++;
            const layout Lu = compute_layout(g.width, g.height, cfg);
            const layout Lg = compute_layout(g.width, g.height, geo);
            // Same pixels, same tiles, same token count — only the labels move.
            check_eq(Lg.n_tiles, Lu.n_tiles, "legacy keeps n_tiles", "swap");
            check_eq(Lg.total_tokens, Lu.total_tokens, "legacy keeps token count", "swap");
            check_eq(Lg.rows, Lu.cols, "legacy rows == geometric cols", "swap");
            check_eq(Lg.cols, Lu.rows, "legacy cols == geometric rows", "swap");
            std::vector<int32_t> a, b;
            build_image_markup(Lu, tok, a);
            build_image_markup(Lg, tok, b);
            if (a != b) n_differ++;
        }
        check(n_nonsquare > 0, "the golden table contains a non-square grid to test with");
        check_eq(n_differ, n_nonsquare, "legacy labels differ on every non-square grid", "swap");
    }

    // ── 6. the regression canary, stated explicitly ────────────────────────
    //
    // commons_example_receipt.png is 500x650 and must NOT split, or the
    // measured 45-character A/B result stops being a comparison.
    {
        const layout L = compute_layout(500, 650, cfg);
        check(!L.split, "the 500x650 fixture stays single-tile");
        check_eq(L.total_tokens, 252, "the 500x650 fixture still emits 252 image tokens", "canary");
        check_eq(L.resized_w, 448, "fixture resized_w", "canary");
        check_eq(L.resized_h, 576, "fixture resized_h", "canary");
        std::vector<int32_t> ids;
        build_image_markup(L, tok, ids);
        // Byte-identical to what the pre-multi-tile build emitted.
        check_eq((int)ids.size(), 254, "fixture markup is <|image_start|> + 252 + <|image_end|>", "canary");
        for (size_t i = 1; i + 1 < ids.size(); i++)
            if (ids[i] != tok.image) {
                check(false, "fixture markup carries no tile labels");
                break;
            }
    }

    // ── 7. the acceptance fixture's markup, pinned against a real reference ──
    //
    // commons_test_ocr_document.jpg is 1920x2485 → a 2-wide, 3-tall grid. The
    // transformers 5.x reference dumped on Kaggle put <|img_row_2_col_1|>
    // (124918) third; the 4.57.x behaviour puts <|img_row_1_col_3|> (124910)
    // there. That one id is the whole difference, and it is invisible to every
    // cosine in the harness.
    {
        const layout L = compute_layout(1920, 2485, cfg);
        check_eq(L.grid_w, 2, "acceptance fixture grid_w", "fixture");
        check_eq(L.grid_h, 3, "acceptance fixture grid_h", "fixture");
        check_eq(L.rows, 3, "acceptance fixture rows (geometric)", "fixture");
        check_eq(L.cols, 2, "acceptance fixture cols (geometric)", "fixture");
        check_eq(L.total_tokens, 1788, "acceptance fixture image tokens", "fixture");

        std::vector<int32_t> ids;
        build_image_markup(L, tok, ids);
        // Third element = second tile label = index 1 + 1*(tile_tokens+1).
        const int third_label_at = 1 + 2 * (L.tile_tokens + 1);
        check_eq(ids[third_label_at], 124918, "third tile label is <|img_row_2_col_1|>", "fixture");

        config legacy = cfg;
        legacy.legacy_label_swap = true;
        std::vector<int32_t> lids;
        build_image_markup(compute_layout(1920, 2485, legacy), tok, lids);
        check_eq(lids[third_label_at], 124910, "legacy puts <|img_row_1_col_3|> there", "fixture");
    }

    // ── 8. position-embedding resample vs real F.interpolate ────────────────
    //
    // Siglip2 resizes its learned 16x16 position table with
    // F.interpolate(bilinear, align_corners=False, antialias=True). antialias
    // is a no-op on upscale, so a plain bilinear matched every shape the tiler
    // produces — and silently would not for a grid under 16 in a dimension
    // (a 150x200 image gives 14x20). No fixture we run exercises that, which
    // is precisely why it needs a hermetic pin rather than a decode check.
    {
        const int G = lfm2_posembed_golden::kGrid, D = lfm2_posembed_golden::kDim;
        std::vector<float> src((size_t)G * G * D);
        for (int y = 0; y < G; y++)
            for (int x = 0; x < G; x++)
                for (int d = 0; d < D; d++)
                    src[((size_t)y * G + x) * D + d] = (float)((y * 37 + x * 11 + d * 5) % 101) / 101.0f;

        double worst = 0.0;
        for (int i = 0; i < lfm2_posembed_golden::kNumCases; i++) {
            const auto & cse = lfm2_posembed_golden::kCases[i];
            std::vector<float> got((size_t)cse.out_h * cse.out_w * D, 0.0f);
            resize_bilinear_aa(src.data(), G, G, D, got.data(), cse.out_h, cse.out_w);
            double max_abs = 0.0;
            for (size_t k = 0; k < got.size(); k++)
                max_abs = std::max(max_abs, (double)std::fabs(got[k] - cse.expect[k]));
            worst = std::max(worst, max_abs);
            // f32 accumulation against an f64 reference; 1e-6 is far tighter
            // than the defect (dropping antialias moves 8x8 by ~0.18).
            check(max_abs < 1e-6, cse.name);
            if (max_abs >= 1e-6) printf("       %s: max_abs=%.3e\n", cse.name, max_abs);
        }
        printf("  %d position-embedding resamples vs torch F.interpolate, worst max_abs=%.2e\n",
               lfm2_posembed_golden::kNumCases, worst);
    }

    printf("%s — %d checks, %d failures\n", g_failures ? "FAIL" : "PASS", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

// Route through core_util::clean_exit per the tools/check_test_clean_exit.sh
// guard: a one-shot binary that returns from main() can crash in GPU-device
// teardown at exit. Host-only here, but the guard is blanket over tests/*.cpp.
int main() {
    core_util::clean_exit(crispembed_test_main());
}
