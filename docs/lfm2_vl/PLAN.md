# LFM2.5-VL-3B — vision-language OCR backend

Engine: `src/lfm2_vl_ocr.{h,cpp}` · ShortConv decode step: `src/lfm2_shortconv.h`
Guard: `tests/test_lfm2_shortconv.cpp` (hermetic, weight-free, ~10 ms)

## NOW — active work

Branch `feat/lfm2vl-multitile`, off `main` @ `ebc8bc95`. Multi-tile NaFlex is
IMPLEMENTED and gated OFF; the acceptance run has not happened yet.

- **DONE** — both decode bugs (§1, §2), KV decode as default, NaFlex
  single-tile resize (§3), registry + companion download, gated debug prints.
- **DONE** — §4 multi-tile NaFlex: layout, tiling, per-tile vision encode,
  per-tile token markup, hermetic guard, reference-dumper support, prompt-token
  parity check. Behind `LFM2_VL_MULTI_TILE=1`, default off.
- **IN FLIGHT** — the acceptance run. Kernel
  `tools/kaggle/lfm2-vl-multitile/` (chr1s4,
  `chr1s4/lfm2-5-vl-multitile-acceptance`). It builds with CUDA, dumps a
  blueprint reference for a SPLITTING fixture, checks prompt-token parity and
  per-stage cosines, then runs the decoded-output A/B (multi-tile off/on x
  Q4_K/F16) scored as CER/WER against `ground_truth.json`.
- **NEXT** — flip `LFM2_VL_MULTI_TILE` on by default ONLY with the canary
  identical across arms and a stated CER improvement (rule 3). Then bicubic
  resample (`LFM2_VL_BICUBIC`, already wired, needs its own A/B), then an
  uncontended timing run. The vision encoder (~250 s per 1024-patch image on
  this VPS) dominates and is where perf work belongs.

### A/B result — the acceptance gate

`commons_example_receipt.png`, Q4_K, 15 tokens, both arms with the §2 mask fix:

| arm | output | chars | decode |
|---|---|--:|--:|
| KV-cached | `Jackson-Washington⏎6640 Ortiz Cove, Markmouth` | 45 | ~19 s/tok |
| full recompute | `Jackson-Washington⏎6640 Ortiz Cove, Markmouth` | 45 | ~175 s/tok |
| **defaults, post-§3** | `Jackson-Washington⏎6640 Ortiz Cove, Markmouth` | 45 | ~19 s/tok |

Byte-identical, and an exact character-for-character prefix of this fixture's
entry in `tests/regression/images/cc0/ground_truth.json` — a manual
transcription made without consulting any OCR output. Equal quality and ~9×
faster is what earns the default flip (dev guide rule 3).

⚠ **The timings are a RATIO only.** Another agent ran a parallel Rust build
across both arms, so rule 5 (identical load, quiet box, median of ≥3) is not
satisfied and no absolute number here is quotable.

## §1 — ShortConv decode reduced over the wrong axis

`build_decode_step_graph` built the depthwise-conv product as `[conv_k, D]`
(flat index `d*K + k`) and then summed the taps with `ggml_view_1d` at byte
offsets `{0, D, 2D}` — which sums flat indices `{i, i+D, i+2D}` and straddles
taps and channels. The output has the correct shape `[D]` and a plausible
magnitude, so every shape check, every allocation, and every downstream op was
happy. 22 of the 30 LFM2 layers are ShortConv, so the decode was garbage from
layer 0.

Symptom: decode step 0 argmax `17177` ('差') instead of `1870` ('son').

This is why the whole KV-cache investigation (cache population, F32 vs F16
cache, view sizes, `ggml_cont` on K/V, sched backend assignment) came up empty
— the KV cache was correct the entire time.

**Fix**: the reduction moved to `lfm2_shortconv::step()`, which transposes the
*kernel* into the window's `[D, K]` layout so the tap axis is the slow axis and
each tap's `D` channel values are one contiguous run.

**Guard** (`tests/test_lfm2_shortconv.cpp`, written before the fix and watched
to fail — `max_abs` 1.07 at D=4, 2.68 at D=2048): checks the decode step against
(a) a scalar reference and (b) the *prefill* `ggml_conv_1d_dw` path's last
column. (b) is the invariant that actually matters, because it is what makes
KV-cached decode agree with full recompute.

| | before | after |
|---|---|---|
| decode step 0 vs scalar ref | max_abs 2.68 | **0.000** (exact) |
| decode step 0 vs prefill conv | max_abs 2.68 | **5.0e-4** (F16 kernel rounding) |
| decode step 0 argmax | 17177 ('差') | **1870 ('son')** ✓ |

## §2 — the full-recompute path never wrote its causal masks

`build_prefill_graph` declares one `causal_mask_<il>` input per attention layer.
The prefill call site wrote them; the no-KV full-recompute call site did not.

An unset ggml input is **not zero** — it is whatever the graph allocator last
left in that buffer. So in the path that had been made the default *because it
was believed correct*, every token after the first was decoded under an
allocator-leftover attention pattern.

That is exactly the observed `Jackpot!`: the first token comes from the real
prefill (correct — "Jack"), and everything after it came from garbage masks.
The dev guide's trap, verbatim: the wrong reference agreed with the wrong C++,
so parity proved nothing.

**Fix**: both callers now go through one `set_prefill_seq_inputs()`.

**Guard**: `declare_input()` records every input a graph declares, and
`audit_graph_inputs()` prints a `BUG:` line for any that nobody wrote. This is
structural — a new input added to `build_prefill_graph` and not to the setter
is reported at the next run rather than silently read as leftovers.

## Measured (Q4_K, `tests/regression/images/cc0/commons_example_receipt.png`, 15 tokens)

`LFM2_VL_KV_CACHE=1`, after both fixes:

```
Jackson-Washington
6640 Ortiz Cove, Markmouth
```

Real receipt text, correct English. Previously `Jackpot!` (recompute, garbage
masks) and `Jack差…` (KV, scrambled conv).

⚠ **Timings from this session are NOT usable.** The box was running another
agent's parallel Rust build throughout; the dev guide's rule 5 (both arms under
identical load, median of ≥3, never a contended box) is not satisfied. For the
record only, contended: vision encoder 299 s, projector 16 s, prefill 273 tok
172 s, KV decode ≈19 s/token. The vision encoder — not the decode — is the
dominant cost on this VPS and is the next real perf target.

## Env gates

| gate | default | meaning |
|---|---|---|
| `CRISPEMBED_ACCEPT_LFM_LICENSE` | off | required; LFM-1.0 is revenue-capped |
| `LFM2_VL_KV_CACHE` | **on** | KV-cached per-token decode; `=0` restores full recompute |
| `LFM2_VL_ZERO_CONV_STATE` | off | debug: zero the ShortConv state cache |
| `LFM2_VL_MULTI_TILE` | **off** | split a large page into a tile grid + thumbnail (§4) |
| `LFM2_VL_TILE_LABELS_GEOMETRIC` | off | label tiles by geometry instead of reproducing upstream's row/col swap; unvalidated |
| `LFM2_VL_BICUBIC` | off | PIL-matching Catmull-Rom resample (HF uses `resample: 3`); needs its own A/B |
| `LFM2_VL_LEGACY_RESIZE` | off | pre-blueprint NaFlex resize: factor=P, min=max=tile², and `std::round` instead of half-to-even |
| `LFM2_VL_NO_REPEAT_NGRAM` | 5 | greedy no-repeat n-gram size |
| `LFM2_VL_DBG` | off | diagnostics |
| `LFM2_VL_DIFF_REF` | unset | per-stage diff archive |

## §3 — NaFlex resize was matched on the wrong two parameters

Reading `Lfm2VlImageProcessor.smart_resize` (transformers `models/lfm2_vl/
image_processing_lfm2_vl.py`) against ours turned up two argument mismatches.
The algorithm was already right; the numbers fed to it were not.

**(a) Rounding factor.** HF rounds to `encoder_patch_size * downsample_factor`
= 32; we rounded to `patch_size` = 16. HF's own docstring says why: *"Both
dimensions are divisible by `encoder_patch_size` * `downsample_factor`. This
ensures no padding is needed in the downsampling step."* With 16 the patch grid
can come out **odd** in a dimension, and the projector's 2× `pixel_unshuffle`
integer-divides (`pW / f`, `pH / f`), so the last row or column of patches is
discarded — a strip of the page silently lost. Three of four common shapes hit
it:

| shape | old grid | new grid | old tokens | new tokens |
|---|---|---|--:|--:|
| 150×200 thumbnail | 37×28 **odd** | 20×14 | 259 | 70 |
| **500×650 (the fixture)** | 36×28 | 36×28 | **252** | **252** |
| 300×1000 strip | 58×17 **odd** | 58×16 | 246 | 232 |
| 3000×4000 scan | 36×27 **odd** | 36×26 | 243 | 234 |

**(b) Pixel bound.** HF uses a token *band*,
`min_image_tokens(64)..max_image_tokens(256)`, each × `P² · ds²`. We pinned
`min = max = tile_size²`, so a small image was upscaled to the full budget —
the 150×200 thumbnail above became 259 tokens instead of 70.

Fixed, with `LFM2_VL_LEGACY_RESIZE=1` restoring the old parameters. The two
agree **exactly** on the validated fixture (576×448, 36×28, 252 tokens), so this
is a provable no-op there and a fix everywhere else. A `WARNING:` now fires if a
grid ever comes out indivisible by the downsample factor.

Still unmatched: HF uses `resample: 3` = **bicubic**; `preprocess_image` does
bilinear. Untested — it needs a fixture where it changes the decode.

## §4 — Multi-tile NaFlex (implemented, gated off pending the acceptance run)

A 300 dpi A4 scan was squashed into one 448x576 tile — **252 image tokens for a
whole page**. With `LFM2_VL_MULTI_TILE=1` it becomes a 2x3 grid of 512x512
tiles plus a whole-page thumbnail: **1770 image tokens, 7x the visual detail.**
This was the largest remaining quality gap in the backend.

Layout lives in `src/lfm2_vl_tiling.h`, weight-free and header-only. That is
deliberate: everything it computes sits downstream of every tensor and is
therefore invisible to the diff harness (HARD RULE 3b). A wrong grid or a wrong
token id yields perfectly healthy activations from the wrong prompt, and reads
as "the model is weak on multi-tile" rather than as a bug. Extracting it makes
it hermetically testable.

### What reading the blueprint turned up that the handover did not have

**(a) Upstream transposes its own row/col labels.** `resize_and_split` does

```python
images, num_rows, num_cols = self.crop_image_to_patches(...)
```

and `crop_image_to_patches` returns `(processed_images, grid_width,
grid_height)`. So `num_rows` is the grid **width**. A portrait A4 is cut into
3 geometric rows of 2 tiles and labelled `<|img_row_1_col_1..3|>`,
`<|img_row_2_col_1..3|>` — 2 rows of 3. Confirmed against the real processor:
1024x768 reports `rows=3, cols=2` while its tiles are 2 rows of 3.

That is what the deployed model is prompted with, so it is the parity target
and the default. `LFM2_VL_TILE_LABELS_GEOMETRIC=1` restores the intuitive
mapping; it is opt-in and unvalidated, and the guard pins that the two produce
DIFFERENT markup on every non-square grid so they cannot silently converge.

**(b) The handover's golden table overcounted.** A4 is 6x256 + a 234-token
thumbnail = **1770**, not 1792; US letter is 1788. The thumbnail token count is
`ceil((h/P)/ds) * ceil((w/P)/ds)` on the smart_resize output and is only 256
when the thumbnail happens to be square. The oracle now emits the table so no
human transcribes it again.

**(c) The class defaults are not what runs.** `Lfm2VlImageProcessorFast` says
`min_tiles=2` and BILINEAR; the shipped `processor_config.json` says
`min_tiles=1` and `resample=3` (bicubic). The config wins.

### The banker's-rounding trap, and why the guard is narrower than it

Python's `round()` is half-to-**even**; C++ `std::round` is half-away-from-zero.
They disagree whenever `dimension / 32` lands on `k + 0.5` for even `k`. This is
not a rounding nicety:

| page | Python (correct) | `std::round` |
|---|---|---|
| 144x4000 | 1 tile, 252 image tokens | **1x10 split, 2812 tokens** |
| 272x272 | 256x256, 64 tokens | 288x288, 81 tokens |
| 80x4000 | 4000 tall | 3616 tall |

Guarded twice: directly against every value in `[0, 4096]` where the two rules
differ at factor 32 — exact arithmetic, no tolerance, no signal — and
end-to-end through four layout cases found by sweeping widths against heights
precisely because the whole pipeline diverges on them. Watched to fail:
swapping `round_by_factor`'s integer half-to-even for `std::round` reds 64
rounding cases plus the four end-to-end cases.

`image_preproc::smart_resize` keeps `std::round` and is left alone — other
engines' parity is measured against their own references, and changing a shared
helper to fix one engine is how you break three.

### The three orders that must agree

Tile pixel order, projector row order, and `<image>` markup order are all
"tiles in reading order, thumbnail last". `generate()`'s splice loop needed no
change — verified rather than assumed: it walks `image_embeds` row by row,
consuming one per `<image>` token. `encode_vision_tiles` now refuses outright
when the projector row count and the layout's token count disagree, instead of
letting the splice run off the end.

### Validation so far

| check | result |
|---|---|
| hermetic layout guard (`test-lfm2-tiling`) | **PASS**, 19474 checks, 20 layout cases, 20 markup sequences |
| oracle vs the REAL `Lfm2VlImageProcessorFast`, 44 image sizes | **PASS**, exact on grid, row/col info, tile and thumbnail patch grids, token counts |
| token ids vs the shipped GGUF vocab | **0/100 mismatches** on the `124908 + (R-1)*10 + (C-1)` formula |
| 500x650 canary, gate off | **byte-identical** — 1 image, 36x28 patches, 252 image tokens, 273 prompt tokens, `Jackson-Washington\n6640 Ortiz Cove, Markmouth`, 45 chars |
| multi-tile structural run, 1024x544 | 2x1 grid + thumbnail 672x352, 743 image tokens, 3 images of 32x32 patches — matches the oracle |
| **decoded output on a splitting page** | **NOT YET RUN** — this is why the gate is off |

`tools/lfm2_vl_tiling_hf_check.py` is the cross-check against the real
processor. It needs torch + torchvision, so it is a developer tool, not a build
step: run it after a transformers upgrade, then regenerate
`tests/lfm2_tiling_golden.h`. The golden header itself is hermetic.

### Cost, and why the acceptance run is on Kaggle

The vision encoder is ~250 s per 1024-patch image on this VPS, and a split A4
page is SEVEN images — ~30 minutes of encode for one arm, before the ~1800-token
prefill. The gate needs four arms (off/on x Q4_K/F16, rule 4.2). Kernel:
`tools/kaggle/lfm2-vl-multitile/`, account chr1s4.

## Still open (from the port handover)

- **Multi-tile NaFlex** — implemented and gated off (§4). What remains is the
  acceptance run, not the code.
- **Shared causal mask** — `build_prefill_graph` allocates one `n_tokens²` F16
  mask PER attention layer, and all 8 are identical. At 273 tokens that is 1.2
  MB; at a 2837-token multi-tile prompt it is ~129 MB, ~113 MB of it redundant.
  Output-neutral to share one tensor. Not done: measured need first, and it
  touches the single-tile path too.
- **Bicubic resample** — HF `resample: 3`, ours is bilinear.
- **README / `docs/ocr_backend_matrix.md`** — DONE (backend-table row + a matrix
  row carrying both decode bugs and the measured output).
- **Registry entry + auto-download** — DONE, as `lfm2-vl`, pointing straight at
  the official `LiquidAI/LFM2.5-VL-3B-GGUF` rather than a `cstr/*` mirror: our
  loader reads it unchanged, so a re-host would be a copy to keep in sync for
  nothing. This needed a new registry capability — `ModelEntry.companion_*`,
  because a VL model is two files (LLM + `mmproj-*` vision tower) and
  `lfm2_vl_ocr_init` finds the tower by scanning the model's own directory, so
  registering the LLM alone would have installed something that cannot load.
  Both files are SHA-256 pinned (`tools/fetch_model_hashes.py` scans every
  resolve-URL in the array, so the companion was covered with no generator
  change). `lfm1.0` was already classified restricted, so the download requires
  `--accept-license lfm1.0`, on top of the engine's `CRISPEMBED_ACCEPT_LFM_LICENSE=1`.
- **Debug prints** — DONE, moved behind `LFM2_VL_DBG=1`. The fixture-specific
  `expected 1870 ('son')` line is gone.
- **Projector parity** sits at cos 0.958 (F16 drift) while the vision encoder is
  0.999 and `llm_logits_last` is 0.9995. Worth a look, but it is not what was
  breaking the decode.
- **Debug prints** — several unconditional `[lfm2_vl]` `fprintf`s should move
  behind `LFM2_VL_DBG=1`.
