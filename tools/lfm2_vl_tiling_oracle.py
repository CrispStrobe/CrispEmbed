#!/usr/bin/env python3
"""LFM2.5-VL tiling oracle — HF's own functions, extracted verbatim.

This script produces the golden vectors that `tests/test_lfm2_tiling.cpp` pins,
per dev-guide HARD RULE 2c: the guard is written (and watched to fail) before
the C++ multi-tile NaFlex code exists.

These are VERBATIM extracts from transformers
`models/lfm2_vl/image_processing_lfm2_vl_fast.py` and `processing_lfm2_vl.py`
(transformers 4.57.6) -- pure math, no torch or torchvision needed, which is
what makes them a real oracle rather than a reimplementation of the thing under
test. Do NOT "clean them up": the value is that they are byte-for-byte what
upstream runs.

Two upstream behaviours are load-bearing and easy to "fix" by accident:

  * Python's round() is BANKER'S rounding (half-to-even). C++ std::round is
    half-away-from-zero. They differ whenever `value / factor` lands exactly on
    k + 0.5 for even k -- e.g. height 80 with factor 32 gives 64 in Python and
    96 in C++. This is the single most likely silent divergence.

  * `crop_image_to_patches` returns `(images, grid_width, grid_height)` in
    every version, but `resize_and_split` changed how it unpacks that:

        transformers <= 4.57.x:  images, num_rows, num_cols = ...
        transformers >= 5.0:     images, num_cols, num_rows = ...

    The old form made num_rows the grid WIDTH, transposing the
    <|img_row_R_col_C|> labels on any non-square grid. Upstream fixed it. THIS
    ORACLE EMITS THE 5.x (geometric) MAPPING, because that is what the shipped
    model is prompted with -- confirmed by prompt-token parity against a
    reference dumped with 5.x. Set LEGACY_LABEL_SWAP below to reproduce 4.57.x.
    The C++ mirrors this with `LFM2_VL_TILE_LABELS_LEGACY_SWAP`.

Config is the shipped `processor_config.json` of LiquidAI/LFM2.5-VL-3B, NOT the
Lfm2VlImageProcessorFast class defaults (which differ: min_tiles=2, BILINEAR).

Usage:
    python tools/lfm2_vl_tiling_oracle.py                 # human-readable table
    python tools/lfm2_vl_tiling_oracle.py --emit-header   # golden-vector header
"""

import argparse
import math

# ── processor_config.json, LiquidAI/LFM2.5-VL-3B ────────────────────────────
P = 16  # encoder_patch_size
DS = 2  # downsample_factor
TILE = 512  # tile_size
MIN_TILES = 1
MAX_TILES = 10
MIN_TOKENS = 64  # min_image_tokens
MAX_TOKENS = 256  # max_image_tokens
TOLERANCE = 2.0  # max_pixels_tolerance
USE_THUMBNAIL = True
DO_IMAGE_SPLITTING = True

# transformers <= 4.57.x transposed the tile row/col labels; 5.x does not.
# False = 5.x behaviour = what the shipped model sees = the parity target.
LEGACY_LABEL_SWAP = False


# ── verbatim from image_processing_lfm2_vl_fast.py ──────────────────────────
def round_by_factor(number, factor):
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height

    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)

        # update best ratio if we found a closer match
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        # if equally close, prefer the ratio that better matches the original image area
        elif ratio_diff == best_ratio_diff:
            target_area = image_size * image_size * ratio[0] * ratio[1]
            if area > 0.5 * target_area:
                best_ratio = ratio

    return best_ratio


def target_ratios(min_tiles, max_tiles):
    ratios = [
        (w, h)
        for n in range(min_tiles, max_tiles + 1)
        for w in range(1, n + 1)
        for h in range(1, n + 1)
        if min_tiles <= w * h <= max_tiles
    ]
    return sorted(set(ratios), key=lambda x: x[0] * x[1])


def get_grid_layout(height, width, min_tiles, max_tiles, tile_size):
    aspect_ratio = width / height
    ratios = target_ratios(min_tiles, max_tiles)
    grid_width, grid_height = find_closest_aspect_ratio(aspect_ratio, ratios, width, height, tile_size)
    target_width = tile_size * grid_width
    target_height = tile_size * grid_height
    total_patches = grid_width * grid_height
    return grid_width, grid_height, target_width, target_height, total_patches


def smart_resize(height, width, downsample_factor, min_image_tokens, max_image_tokens, encoder_patch_size):
    total_factor = encoder_patch_size * downsample_factor
    smart_resize_min_pixels = min_image_tokens * encoder_patch_size**2 * downsample_factor**2
    smart_resize_max_pixels = max_image_tokens * encoder_patch_size**2 * downsample_factor**2

    h_bar = max(total_factor, round_by_factor(height, total_factor))
    w_bar = max(total_factor, round_by_factor(width, total_factor))

    if h_bar * w_bar > smart_resize_max_pixels:
        beta = math.sqrt((height * width) / smart_resize_max_pixels)
        h_bar = max(total_factor, math.floor(height / beta / total_factor) * total_factor)
        w_bar = max(total_factor, math.floor(width / beta / total_factor) * total_factor)
    elif h_bar * w_bar < smart_resize_min_pixels:
        beta = math.sqrt(smart_resize_min_pixels / (height * width))
        h_bar = math.ceil(height * beta / total_factor) * total_factor
        w_bar = math.ceil(width * beta / total_factor) * total_factor

    return w_bar, h_bar


def is_image_too_large(height, width, max_image_tokens, encoder_patch_size, downsample_factor, max_pixels_tolerance):
    total_factor = encoder_patch_size * downsample_factor
    h_bar = max(encoder_patch_size, round_by_factor(height, total_factor))
    w_bar = max(encoder_patch_size, round_by_factor(width, total_factor))
    return h_bar * w_bar > max_image_tokens * encoder_patch_size**2 * downsample_factor**2 * max_pixels_tolerance


# ── verbatim from processing_lfm2_vl.py::Lfm2VlProcessor._get_image_num_tokens ──
def get_image_num_tokens(image_height, image_width, tile_size, downsample_factor, encoder_patch_size, use_thumbnail):
    thumbnail_tokens = 0
    if use_thumbnail:
        num_patches_height = image_height // encoder_patch_size
        num_patches_width = image_width // encoder_patch_size
        dwn_num_patches_height = math.ceil(num_patches_height / downsample_factor)
        dwn_num_patches_width = math.ceil(num_patches_width / downsample_factor)
        thumbnail_tokens = dwn_num_patches_height * dwn_num_patches_width

    num_patches_tile = tile_size // encoder_patch_size
    dwn_num_patches_tile = math.ceil(num_patches_tile / downsample_factor)
    tile_tokens = dwn_num_patches_tile * dwn_num_patches_tile

    return thumbnail_tokens, tile_tokens


# ── the composite the C++ has to reproduce ─────────────────────────────────
def layout(width, height):
    """Everything preprocess_image + build_token_ids must agree with.

    Mirrors `Lfm2VlImageProcessorFast.resize_and_split` followed by
    `Lfm2VlProcessor.expand_text_with_placeholders`.
    """
    do_split = not (MIN_TILES == MAX_TILES == 1)
    too_large = is_image_too_large(height, width, MAX_TOKENS, P, DS, TOLERANCE)
    # smart_resize runs unconditionally; its output is `image_sizes`, which is
    # the THUMBNAIL size when splitting and the only image when not.
    new_width, new_height = smart_resize(height, width, DS, MIN_TOKENS, MAX_TOKENS, P)

    split = bool(too_large and do_split)
    if split:
        gw, gh, target_w, target_h, n_tiles = get_grid_layout(height, width, MIN_TILES, MAX_TILES, TILE)
        # The unpack (see the module docstring). 5.x: num_cols = grid_width.
        rows, cols = (gw, gh) if LEGACY_LABEL_SWAP else (gh, gw)
    else:
        gw = gh = 1
        target_w, target_h = new_width, new_height
        n_tiles = 1
        rows = cols = 1

    thumb_tokens, tile_tokens = get_image_num_tokens(new_height, new_width, TILE, DS, P, USE_THUMBNAIL)

    labels = []
    if rows > 1 or cols > 1:
        for row in range(rows):
            for col in range(cols):
                labels.append((row + 1, col + 1))
        n_images = n_tiles + (1 if thumb_tokens > 0 else 0)
        total_tokens = n_tiles * tile_tokens + thumb_tokens
        has_thumb = thumb_tokens > 0
    else:
        n_images = 1
        total_tokens = thumb_tokens
        has_thumb = False

    return {
        "width": width,
        "height": height,
        "split": split,
        "grid_w": gw,
        "grid_h": gh,
        "rows": rows,
        "cols": cols,
        "n_tiles": n_tiles,
        "n_images": n_images,
        "has_thumb": has_thumb,
        "target_w": target_w,
        "target_h": target_h,
        "thumb_w": new_width if split else 0,
        "thumb_h": new_height if split else 0,
        "resized_w": new_width,
        "resized_h": new_height,
        "tile_tokens": tile_tokens,
        "thumb_tokens": thumb_tokens if (split or True) else 0,
        "total_tokens": total_tokens,
        "labels": labels,
    }


CASES = [
    (500, 650, "the fixture (does not split -- regression canary)"),
    (150, 200, "small thumbnail (upscaled into the token band)"),
    (300, 1000, "tall strip"),
    (1000, 300, "wide banner"),
    (3000, 4000, "A4 scan @ 300 dpi"),
    (1700, 2200, "US letter @ 200 dpi"),
    (2048, 2048, "square"),
    (4000, 1000, "panorama"),
    (80, 80, "banker's rounding: 80/32 = 2.5 -> Python 64, C++ std::round 96"),
    (1200, 80, "banker's rounding on one axis only"),
    # These four DIVERGE end-to-end between banker's and half-away-from-zero
    # rounding, so they are the cases that actually catch the trap rather than
    # merely exercising it. Found by sweeping every width against a few heights.
    (144, 4000, "banker's: no split / 252 tok; std::round: 1x10 split / 2812 tok"),
    (272, 272, "banker's: 256x256 / 64 tok; std::round: 288x288 / 81 tok"),
    (80, 4000, "banker's: 80x4000 -> 4000 tall; std::round: 3616 tall"),
    (144, 650, "banker's: 128 wide / 80 tok; std::round: 160 wide / 100 tok"),
    (513, 513, "just over one tile"),
    # Landscape/portrait pairs: the clearest demonstration of the row/col swap.
    # 1024x768 is cut into 2 geometric rows of 3 tiles and LABELLED 3 rows of 2.
    (1024, 768, "landscape 4:3 -- 3 tiles across, 2 down"),
    (768, 1024, "portrait 3:4 -- 2 tiles across, 3 down"),
    # The fixture the Kaggle acceptance run uses; its prompt ids are pinned
    # against a transformers 5.x reference.
    (1920, 2485, "commons_test_ocr_document.jpg -- the acceptance fixture"),
    (5000, 400, "extreme panorama -- 10x1, the max_tiles ceiling"),
    (2480, 3508, "A4 @ 300 dpi, exact ISO pixel dims"),
    (1024, 1024, "exactly 2x2 tiles of raw pixels"),
]


def print_table():
    print("target_ratios(%d,%d) = %s" % (MIN_TILES, MAX_TILES, target_ratios(MIN_TILES, MAX_TILES)))
    print()
    hdr = (
        "%11s %5s %7s %11s %5s %4s %13s %11s %8s %9s %6s"
        % ("W x H", "split", "gw x gh", "rows x cols", "tiles", "imgs",
           "resized WxH", "thumb WxH", "tile tok", "thumb tok", "total")
    )
    print(hdr)
    print("-" * len(hdr))
    for w, h, label in CASES:
        r = layout(w, h)
        grid = "%dx%d" % (r["grid_w"], r["grid_h"])
        rc = "%dx%d" % (r["rows"], r["cols"])
        res = "%dx%d" % (r["resized_w"], r["resized_h"])
        thumb = ("%dx%d" % (r["thumb_w"], r["thumb_h"])) if r["split"] else "-"
        print(
            "%4d x%5d %5s %7s %11s %5d %4d %13s %11s %8d %9d %6d   # %s"
            % (w, h, "YES" if r["split"] else "no", grid, rc, r["n_tiles"],
               r["n_images"], res, thumb,
               r["tile_tokens"] if r["split"] else 0,
               r["thumb_tokens"], r["total_tokens"], label)
        )
    print()
    for w, h, label in CASES:
        r = layout(w, h)
        if r["labels"]:
            print("%dx%d labels: %s" % (w, h, " ".join("r%dc%d" % (a, b) for a, b in r["labels"])))


def emit_header():
    lines = []
    lines.append("// GENERATED by tools/lfm2_vl_tiling_oracle.py -- DO NOT EDIT BY HAND.")
    lines.append("//")
    lines.append("// Golden vectors for LFM2.5-VL multi-tile NaFlex, produced by running")
    lines.append("// transformers' own (verbatim-extracted) tiling + token-markup functions")
    lines.append("// against the shipped processor_config.json of LiquidAI/LFM2.5-VL-3B.")
    lines.append("//")
    lines.append("// Regenerate with:  python tools/lfm2_vl_tiling_oracle.py --emit-header \\")
    lines.append("//                       > tests/lfm2_tiling_golden.h")
    lines.append("")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <stdint.h>")
    lines.append("")
    lines.append("namespace lfm2_tiling_golden {")
    lines.append("")
    lines.append("struct label { int row, col; };")
    lines.append("")
    lines.append("struct layout_case {")
    lines.append("    const char * name;")
    lines.append("    int width, height;")
    lines.append("    int split;                 // 1 = image is tiled")
    lines.append("    int grid_w, grid_h;        // tile grid, geometric")
    lines.append("    int rows, cols;            // what the processor LABELS them (see the swap)")
    lines.append("    int n_tiles;               // grid_w * grid_h")
    lines.append("    int n_images;              // tiles + thumbnail")
    lines.append("    int has_thumb;")
    lines.append("    int target_w, target_h;    // whole-image resize before cutting into tiles")
    lines.append("    int resized_w, resized_h;  // smart_resize output (= thumbnail size when split)")
    lines.append("    int tile_tokens;           // image tokens per 512x512 tile")
    lines.append("    int thumb_tokens;          // image tokens for the smart_resize'd image")
    lines.append("    int total_tokens;          // total <image> tokens in the prompt")
    lines.append("    int n_labels;")
    lines.append("    const label * labels;")
    lines.append("};")
    lines.append("")

    for i, (w, h, label) in enumerate(CASES):
        r = layout(w, h)
        if r["labels"]:
            body = ", ".join("{%d, %d}" % (a, b) for a, b in r["labels"])
            lines.append("static const label kLabels%d[] = { %s };" % (i, body))
        else:
            lines.append("static const label kLabels%d[1] = { {0, 0} };  // unused" % i)
    lines.append("")
    lines.append("static const layout_case kCases[] = {")
    for i, (w, h, label) in enumerate(CASES):
        r = layout(w, h)
        lines.append(
            "    { %s, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, kLabels%d },"
            % (
                '"%s"' % label.replace('"', "'"),
                r["width"],
                r["height"],
                int(r["split"]),
                r["grid_w"],
                r["grid_h"],
                r["rows"],
                r["cols"],
                r["n_tiles"],
                r["n_images"],
                int(r["has_thumb"]),
                r["target_w"],
                r["target_h"],
                r["resized_w"],
                r["resized_h"],
                r["tile_tokens"],
                r["thumb_tokens"],
                r["total_tokens"],
                len(r["labels"]),
                i,
            )
        )
    lines.append("};")
    lines.append("")
    lines.append("static const int kNumCases = %d;" % len(CASES))
    lines.append("")
    lines.append("// _target_ratios(min_tiles=1, max_tiles=10), in the exact order upstream")
    lines.append("// iterates it. Within one area that order comes from CPython's set")
    lines.append("// iteration, not from the comprehension -- and it decides ties.")
    lines.append("struct ratio_pair { int w, h; };")
    tr = target_ratios(MIN_TILES, MAX_TILES)
    lines.append("static const ratio_pair kTargetRatios[] = { %s };"
                 % ", ".join("{%d, %d}" % (w, h) for w, h in tr))
    lines.append("static const int kNumTargetRatios = %d;" % len(tr))
    lines.append("")
    lines.append("// round_by_factor(value, 32) as Python computes it (banker's rounding).")
    lines.append("struct round_case { int value, factor, expected; };")
    lines.append("static const round_case kRoundCases[] = {")
    seen = set()
    for factor in (32,):
        for v in range(0, 4097):
            e = round_by_factor(v, factor)
            naive = int(math.floor(v / factor + 0.5)) * factor  # C++ std::round
            if e != naive and (v, factor) not in seen:
                seen.add((v, factor))
                lines.append("    { %d, %d, %d }," % (v, factor, e))
    lines.append("};")
    lines.append("static const int kNumRoundCases = sizeof(kRoundCases) / sizeof(kRoundCases[0]);")
    lines.append("")
    lines.append("}  // namespace lfm2_tiling_golden")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-header", action="store_true", help="emit tests/lfm2_tiling_golden.h")
    args = ap.parse_args()
    if args.emit_header:
        emit_header()
    else:
        print_table()
