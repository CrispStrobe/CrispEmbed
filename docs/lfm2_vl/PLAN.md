# LFM2.5-VL-3B — vision-language OCR backend

Engine: `src/lfm2_vl_ocr.{h,cpp}` · ShortConv decode step: `src/lfm2_shortconv.h`
Guard: `tests/test_lfm2_shortconv.cpp` (hermetic, weight-free, ~10 ms)

## NOW — active work

Branch `feat/lfm2vl-kv-decode`, tip `49170ba5`. Nothing in flight.

- **DONE** — the "KV-cached decode is broken" bug. It was never the KV cache
  (§1); the KV cache was correct the whole time.
- **DONE** — a second bug, in the path that had been made the *default* because
  it was believed correct (§2).
- **DONE** — A/B settled, KV decode is now the default (`LFM2_VL_KV_CACHE=0`
  restores full recompute). Three independent runs on the fixture, all 45 chars,
  all identical.
- **DONE** — NaFlex resize parameters matched to the blueprint (§3), registry
  entry + companion-file support, README + backend matrix, debug prints gated,
  layer-types diagnostic ordering + fallback string.
- **NEXT** — multi-tile NaFlex (see below; needs Kaggle, the trigger rule is
  pinned); bicubic resample; an uncontended timing run. The vision encoder
  (~300 s) dominates, not the decode — that is where perf work belongs.

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
| `LFM2_VL_LEGACY_RESIZE` | off | pre-fix NaFlex resize params (factor=P, min=max=tile²) |
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

## Multi-tile NaFlex — NOT implemented, and the blueprint rule is now pinned

`processor_config.json`: `do_image_splitting=true`, `max_tiles=10`,
`min_tiles=1`, `use_thumbnail=true`, `tile_size=512`, `max_pixels_tolerance=2.0`.

`Lfm2VlImageProcessor._is_image_too_large` is the exact trigger:

```
h_bar = max(P, round_by_factor(height, P * ds))      # round-half-to-EVEN
w_bar = max(P, round_by_factor(width,  P * ds))
split  <=>  h_bar * w_bar > max_image_tokens * P**2 * ds**2 * max_pixels_tolerance
        i.e. > 256 * 256 * 4 * 2.0 = 524288 px
```

**The current fixture does NOT split** — 500×650 gives h_bar·w_bar = 640·512 =
327680 < 524288 — which is what makes the single-tile result above a legitimate
comparison rather than an accident. Any page above ~524k rounded pixels does
split, and we would silently squash it into one tile instead.

**Full handover with the complete blueprint, golden vectors, verified token IDs
and a cost model: `/mnt/volume1/naflex-todos.md`.** The oracle that produced the
golden layouts is `tools/lfm2_vl_tiling_oracle.py` (HF's own functions extracted
verbatim — pure math, no torch), so the guard test can be written before the
code per HARD RULE 2c.

All 100 `<|img_row_R_col_C|>` tokens plus `<|img_thumbnail|>` are ALREADY in the
shipped GGUF vocab at contiguous ids — `124908 + (R-1)*10 + (C-1)`, thumbnail
125008, verified 0 mismatches over all 100. No converter work is needed.

Implementing it needs `crop_image_to_patches` + `find_closest_aspect_ratio` +
the thumbnail append, and `Lfm2VlProcessor` (`return_row_col_info=true`) for the
per-tile token markup. It also needs a reference dump to validate against, and
the vision encoder costs ~300 s **per tile** on this VPS — a 10-tile page is
~50 min of encode alone. Per the dev guide's division of labour that belongs on
Kaggle, not here.

## Still open (from the port handover)

- **Multi-tile NaFlex** — see the section above; the trigger rule is pinned, the
  implementation is not started.
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
