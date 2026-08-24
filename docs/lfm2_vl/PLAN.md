# LFM2.5-VL-3B — vision-language OCR backend

Engine: `src/lfm2_vl_ocr.{h,cpp}` · ShortConv decode step: `src/lfm2_shortconv.h`
NaFlex tiling math: `src/lfm2_naflex.h`
Guards: `tests/test_lfm2_shortconv.cpp`, `tests/test_lfm2_naflex.cpp`
(both hermetic, weight-free, ~10 ms)

## NOW — active work

Branch `perf/lfm2vl-mac`. Nothing in flight.

- **DONE** — both decode bugs (§1, §2) and the NaFlex resize parameters (§3).
- **DONE** — §4 the resize FILTER, which turns out to be the whole of the
  "projector cos 0.958 F16 drift" that §3 left open. It was never F16 drift.
- **DONE** — §5 the projector moved off a scalar CPU loop onto the sched
  backend: 6581 ms → 8 ms, 31% of the pipeline gone.
- **DONE** — §6 multi-tile NaFlex, validated against the reference's own
  golden `prompt_token_ids` (1816/1816 exact).
- **NEXT** — decode is now the dominant cost by a wide margin (§7). Also open:
  a second reference fixture that actually exercises bicubic-vs-PIL rounding,
  and an uncontended timing run.

## Measured — the whole lane, M1 Metal, Q4_K, 512 max tokens

`tests/regression/images/cc0`, CER/WER against
`tests/regression/images/cc0/ground_truth.json` (manual transcription, no OCR
consulted). Greedy decode, so the text is deterministic and the CER column is
exact; the millisecond column is NOT (see the warning below).

| fixture | CER before | CER after | WER before | WER after | ms before | ms after |
|---|--:|--:|--:|--:|--:|--:|
| commons_example_receipt.png | 0.329 | 0.329 | 0.786 | 0.786 | 63708 | 26887 |
| simple_form.png | 0.364 | 0.413 | 0.444 | 0.556 | 21533 | 9212 |
| receipt_historical.png | 0.087 | 0.108 | 0.319 | 0.319 | 94884 | 44317 |
| german_official_print.jpg | 0.361 | **0.268** | 0.537 | **0.425** | 69098 | 57787 |
| commons_test_ocr_document.jpg | 0.701 | **0.289** | 0.903 | **0.272** | 89897 | 61140 |
| **mean** | 0.368 | **0.281** | 0.598 | **0.472** | | |

End to end, baseline → §4+§5+§6: mean CER **0.368 → 0.217**, mean WER
**0.598 → 0.368**.

"after" = §4 + §5, single-tile (the shipped default at the time of that run).
Adding §6 multi-tile on top, same box, same 512-token cap:

| fixture | tiles | CER single | CER multi | WER single | WER multi | ms single | ms multi |
|---|--:|--:|--:|--:|--:|--:|--:|
| commons_example_receipt.png | 1 | 0.329 | 0.329 | 0.786 | 0.786 | 26887 | 44883 |
| simple_form.png | 1 | 0.413 | 0.413 | 0.556 | 0.556 | 9212 | 15840 |
| receipt_historical.png | 2x4 + thumb | 0.108 | **0.014** | 0.319 | **0.062** | 44317 | 145741 |
| german_official_print.jpg | 2x3 + thumb | 0.268 | **0.052** | 0.425 | **0.194** | 57787 | 128828 |
| commons_test_ocr_document.jpg | 2x3 + thumb | 0.289 | 0.278 | 0.272 | 0.243 | 61140 | 180487 |
| **mean** | | 0.281 | **0.217** | 0.472 | **0.368** | | |

commons_test_ocr_document is TRUNCATED at the 512-token cap in both arms (2188
and 2198 chars against a 2981-char ground truth), so most of its CER is missing
text, not misread text — do not read that row as "multi-tile barely helped".
Rerun at 1024 tokens, where neither arm truncates:

| commons_test_ocr_document.jpg | chars | CER | WER | ms |
|---|--:|--:|--:|--:|
| single tile | 2962 / 2981 | 0.042 | 0.052 | 74431 |
| **2x3 + thumbnail** | **2981 / 2981** | **0.021** | **0.002** | 118794 |

WER 0.002 on a two-column book page — one word wrong in the whole page, at
1.6x the wall clock. That is the row the default rests on.

The two single-tile fixtures are unchanged to the character, which is the
correctness check that matters: the tiling path must be a no-op below the
tolerance. Their millisecond difference is box noise, not the feature.

⚠ **Timings are RATIOS on a contended box.** Two `node --test` processes at
100% CPU ran through the "before" column and were gone by the "after" column;
Firefox was busy throughout. Dev-guide rule 5 (identical load, median of ≥3,
quiet box) is NOT satisfied, so no absolute number here is quotable. The
stage-level shifts (projector 6581 → 8 ms) are far larger than the noise and
are safe to read as real; the ±20% differences are not.

## Env gates

| gate | default | meaning |
|---|---|---|
| `CRISPEMBED_ACCEPT_LFM_LICENSE` | off | required; LFM-1.0 is revenue-capped |
| `LFM2_VL_KV_CACHE` | **on** | KV-cached per-token decode; `=0` restores full recompute |
| `LFM2_VL_MULTI_TILE` | **on** | multi-tile NaFlex; `=0` forces the fast single-tile path |
| `LFM2_VL_BILINEAR_RESIZE` | off | pre-§4 bilinear point sampler |
| `LFM2_VL_PROJ_GGML` | **on** | projector on the sched backend; `=0` restores the scalar loop |
| `LFM2_VL_PROJ_GELU_TANH` | off | tanh GELU in the projector instead of erf |
| `LFM2_VL_LEGACY_RESIZE` | off | pre-§3 NaFlex resize params (factor=P, min=max=tile²) |
| `LFM2_VL_FLASH_ATTN` | off | `ggml_flash_attn_ext` in the vision encoder |
| `LFM2_VL_ZERO_CONV_STATE` | off | debug: zero the ShortConv state cache |
| `LFM2_VL_NO_REPEAT_NGRAM` | 5 | greedy no-repeat n-gram size |
| `LFM2_VL_DBG` | off | diagnostics |
| `LFM2_VL_DIFF_REF` | unset | per-stage diff archive |

## Reference archives

`cstr/crispembed-regression-fixtures`:

- `lfm2_vl/commons_example_receipt/ref.gguf` (13 MB) — single tile.
  `projector_out`, `llm_embed`, `llm_layer_0..3`, `llm_logits_last`.
- `lfm2_vl/commons_test_ocr_document/ref.gguf` (309 MB) — **multi-tile**, and
  the far more useful of the two. 41 tensors: `pixel_values_img0..6` (the
  golden PREPROCESSED patches — this is what identified §4),
  `vis_patch_embed_img*`, `vis_layer_0..3`, `vis_post_ln_img*`,
  `projector_out_img*`, `llm_embed`, `llm_layer_0..3`, `llm_logits_last`,
  `prompt_token_ids` (1816), `spatial_shapes`. Metadata records
  `n_prompt_tokens=1816`, `n_image_tokens=1788`, `n_encoded_images=7`.

Per-stage parity, Q4_K, multi-tile, `cos_global` (see the note under §5 on why
not `cos_min`):

| stage | cos_global |
|---|--:|
| `prompt_token_ids` | 1816/1816 ids **exact** |
| `vis_post_ln_img0` | 0.999994 |
| `projector_out_img0` | 0.999854 |
| `projector_out_img6` (thumbnail) | 0.999882 |
| `llm_embed` | 0.999746 |
| `llm_layer_0` | 0.999702 |
| `llm_layer_3` | 0.999551 |
| `llm_logits_last` | 0.993685 |

## §1 — ShortConv decode reduced over the wrong axis

`build_decode_step_graph` built the depthwise-conv product as `[conv_k, D]`
(flat index `d*K + k`) and then summed the taps with `ggml_view_1d` at byte
offsets `{0, D, 2D}` — which sums flat indices `{i, i+D, i+2D}` and straddles
taps and channels. The output has the correct shape `[D]` and a plausible
magnitude, so every shape check, every allocation, and every downstream op was
happy. 22 of the 30 LFM2 layers are ShortConv, so the decode was garbage from
layer 0. Symptom: decode step 0 argmax `17177` ('差') instead of `1870` ('son').

**Fix**: the reduction moved to `lfm2_shortconv::step()`, which transposes the
*kernel* into the window's `[D, K]` layout so the tap axis is the slow axis.

**Guard** (`tests/test_lfm2_shortconv.cpp`, written before the fix and watched
to fail — `max_abs` 1.07 at D=4, 2.68 at D=2048): the decode step against
(a) a scalar reference and (b) the *prefill* `ggml_conv_1d_dw` path's last
column. (b) is the invariant that matters: it is what makes KV-cached decode
agree with full recompute.

## §2 — the full-recompute path never wrote its causal masks

`build_prefill_graph` declares one `causal_mask_<il>` input per attention
layer. The prefill call site wrote them; the no-KV full-recompute call site did
not. An unset ggml input is **not zero** — it is whatever the graph allocator
last left in that buffer. That is the observed `Jackpot!`: the first token came
from the real prefill ("Jack"), everything after it from garbage masks.

**Fix**: both callers go through one `set_prefill_seq_inputs()`.
**Guard**: `declare_input()` records every input a graph declares and
`audit_graph_inputs()` prints `BUG:` for any that nobody wrote — structural, so
a new input added to the builder and not to the setter is reported at the next
run rather than silently read as leftovers.

## §3 — NaFlex resize was matched on the wrong two parameters

**(a) Rounding factor.** HF rounds to `encoder_patch_size * downsample_factor`
= 32; we rounded to `patch_size` = 16. With 16 the patch grid can come out
**odd** in a dimension, and the projector's 2× `pixel_unshuffle`
integer-divides, so the last row or column of patches is discarded — a strip of
the page lost without a word.

**(b) Pixel bound.** HF uses a token *band*, `min_image_tokens(64) ..
max_image_tokens(256)`, each × `P² · ds²`. We pinned `min = max = tile_size²`,
so a small image was upscaled to the full budget (a 150×200 thumbnail became
259 tokens instead of 70).

Both agree exactly on the 500×650 fixture (576×448, 36×28, 252 tokens), so this
was a provable no-op there and a fix everywhere else. `LFM2_VL_LEGACY_RESIZE=1`
restores the old parameters. Now pinned by `tests/test_lfm2_naflex.cpp`.

## §4 — the resize FILTER, and what "projector cos 0.958" actually was

§3 fixed the resize *dimensions* and left the projector at cos 0.9575, recorded
as "F16 drift". It was not drift. HF resizes with `resample: 3` = **BICUBIC**,
run antialiased; we point-sampled a bilinear with `align_corners`-style
mapping. Measured against the reference's own `pixel_values_img6`
(commons_test_ocr_document, 1920×2485 → 448×576):

| resampler | cos_min | cos_mean | max_abs | \|mine\| (ref 508.23) |
|---|--:|--:|--:|--:|
| ours, bilinear align_corners | 0.815898 | 0.955351 | 1.4171 | 542.22 |
| **PIL BICUBIC** | **0.999999** | **1.000000** | **0.0078** | **508.23** |
| PIL BILINEAR | 0.994601 | 0.998708 | 0.2353 | 501.94 |
| PIL LANCZOS | 0.997148 | 0.999407 | 0.2118 | 511.31 |
| torchvision BICUBIC aa=True | 0.999936 | 0.999991 | 0.1202 | 508.30 |
| torchvision BICUBIC aa=False | 0.811639 | 0.951463 | 1.6033 | 553.40 |
| torchvision BILINEAR aa=False | 0.852335 | 0.964056 | 1.2859 | 542.40 |

Read the magnitude column, not just the cosine (HARD RULE 2b): the
un-antialiased sampler put ~6.7% of extra energy into the patches. `max_abs
0.0078` for PIL BICUBIC is 2/255 — i.e. uint8 rounding, nothing else.

`image_preproc::resize_bicubic_u8_hwc` already existed in this repo (the
Qwen2VL lane) and is exactly PIL's separable Catmull-Rom with antialiasing;
this engine simply was not calling it. Swapping it in:

| stage | before | after |
|---|--:|--:|
| `projector_out` cos | 0.957512 | **0.998966** |
| `projector_out` \|mine\| (ref 34.8739) | 35.6392 | **34.8514** |
| `llm_embed` cos | 0.957512 | **0.998966** |
| `llm_layer_0` cos | 0.964127 | **0.995645** |
| `llm_layer_3` cos | 0.965679 | **0.996240** |
| `llm_logits_last` cos | 0.997605 | **0.998088** |

`LFM2_VL_BILINEAR_RESIZE=1` restores the old sampler.

**The transferable lesson**: the parity gap was in the INPUT, and no amount of
staring at the model would have found it. What found it was one reference
tensor of preprocessed pixels — diff the image before the model.

### Not the cause: the projector GELU

`config.json` says `"projector_hidden_act": "gelu"`, and transformers'
`ACT2FN["gelu"]` is `GELUActivation` → `F.gelu(approximate='none')`, the exact
erf form; the vision tower is the other one (`"gelu_pytorch_tanh"`), which is
what made the tanh approximation look right here. So the projector now uses
`ggml_gelu_erf`. Measured, it is worth ~5e-5 of cosine on this fixture
(0.957468 tanh vs 0.957512 erf, pre-§4) — a blueprint-correctness change, not
the fix for anything. `LFM2_VL_PROJ_GELU_TANH=1` restores tanh.

## §5 — the projector was a scalar CPU loop (6581 ms → 8 ms)

`encode_vision`'s MLP was a single-threaded triple loop over
`n_proj × (4608×2048 + 2048×2048)` plus a `to_f32()` of ~28 MB of F16 weights
**on every image**. On M1 it measured 6581 ms of a 63708 ms pipeline — the
single largest stage, larger than the vision encoder. As a ggml graph on the
sched backend it is 8 ms, with byte-identical decoded output on the fixture.
`LFM2_VL_PROJ_GGML=0` restores the scalar path.

The `pixel_unshuffle` output also changed layout, from `[i * n_proj + p]` to
ggml-native `[p * C_us + i]`. That is Python's own layout for the stage
(`[B, W//f, H//f, C*f²]`, channels fastest), so it feeds `ggml_mul_mat`
directly and a future reference dump of it compares element-wise.

**On reading these diffs**: `diff_stage` now prints `cos_global` beside
`cos_min` and judges on it. `crispembed_diff.h`'s own `Report` docs say why —
on a quantized artifact `cos_min` is a per-row minimum dominated by
numerically fragile near-blank rows (0.81 while `cos_global` is 0.9997), and a
gate that always cries wolf gets ignored. `cos_min` stays the right gate at
reference precision.

Separately: `crispembed_diff.h` read `GGML_TYPE_I32` as type **5**, which is a
quantized type in every ggml enum there has ever been. Any I32 reference tensor
was silently skipped ("skipping … not F32/I32") — so the `prompt_token_ids`
guard below did nothing at all until this was fixed to 26. A guard that cannot
fail is not a guard.

## §6 — multi-tile NaFlex

`Lfm2VlImageProcessor.resize_and_split`: a page over the tolerance is resized
to a `tile_size` grid chosen by aspect ratio, split row-major, and followed by
the whole page smart_resized as one extra "thumbnail" image. The trigger is

```
h_bar = max(P, round_by_factor(height, P * ds))      # round-half-to-EVEN
w_bar = max(P, round_by_factor(width,  P * ds))
split  <=>  h_bar * w_bar > max_image_tokens * P**2 * ds**2 * max_pixels_tolerance
        i.e. > 256 * 256 * 4 * 2.0 = 524288 px
```

Three of the five CC0 fixtures are over it. Each tile contributes
`(tile_size / P / ds)² = 256` tokens; the thumbnail contributes its own grid.

The prompt markup is `<|image_start|>`, then per tile
`<|img_row_R_col_C|>` + 256 `<image>`, then `<|img_thumbnail|>` + the thumbnail's
tokens, then `<|image_end|>`. All 100 row/col marker ids are already in the
shipped vocab at `124908 + (R-1)*10 + (C-1)`, thumbnail 125008 — no converter
work.

**The gate that matters**: the reference ships `prompt_token_ids`, and on
commons_test_ocr_document we emit **1816 prompt tokens / 1788 image tokens,
all 1816 ids exact**. Nothing else can see the tile order, the marker ids or
the per-tile counts — every float diff is blind to them, and a wrong marker
still yields a well-shaped embedding sequence. (`llm_embed` cos_global 0.999746
would have passed with the tiles in the wrong order.)

`tests/test_lfm2_naflex.cpp` pins the tiling math against
`tools/lfm2_vl_tiling_oracle.py` (HF's own functions, extracted verbatim) over
19 golden layouts. It was watched to fail on both defect classes before being
kept: `std::round` instead of banker's rounding → 2 failures; dropping the
equal-ratio tie-break → 10 failures, including 2048×2048 collapsing from a 3×3
grid to 1×1 (2304 image tokens → 256, silently).

### Cost, and the default

Multi-tile is 2–3× the wall clock on a page that splits: 7–9 vision encodes
instead of 1, and a 1816–2290-token prefill instead of 273.

It is nevertheless the **default**, and the reasoning is worth stating because
the dev guide's rule 3 ("flip only when the new path wins on speed AND
quality") reads the other way at first glance. Rule 3 governs *perf*
optimizations, where the old path is the correct one. Here it is the opposite:
`do_image_splitting` is true in the shipped config, and the reference prompt
for a 1920×2485 page carries 1788 image tokens, so **single-tile on a page over
the tolerance is the deviation**, not the new path. Rule 1 (match the
blueprint) wins. And the quality difference is not marginal — CER 0.108 →
0.014 and 0.268 → 0.052 on the two fixtures that are not truncation-limited,
with the sub-tolerance fixtures unchanged to the character.

`LFM2_VL_MULTI_TILE=0` keeps the fast single-tile path for anyone who wants the
2–3× back and can live with the CER.

## §7 — where the time goes now, and what is next

M1 Metal, Q4_K, commons_example_receipt (500×650, single tile, 16 tokens):

| stage | ms |
|---|--:|
| preprocess | 22 |
| vision encoder (1008 patches, 27 layers) | ~2700 |
| projector | 8 |
| prefill (273 tokens) | ~2500 |
| decode | ~110–270 ms/token |

Decode is now the whole cost of any real page: 512 tokens is 60–140 s. It is
also the least optimized path — the graph is rebuilt and re-allocated every
token, the 22 ShortConv state buffers round-trip through the CPU every token,
and the lm_head is a 2048×128000 Q6_K matvec. A 3B Q4_K model on M1 should be
bandwidth-bound at roughly 30 ms/token, so there is a ~4× gap to explain
before optimizing anything.

Also open:

- **`to_f32(token_embd)` was 1.05 GB per image.** The whole 128000×2048 Q6_K
  table was dequantized on every `generate()` to read the ~2 k rows a page uses.
  Replaced by a row-at-a-time `embed_lookup`.
- **Bicubic rounding.** Ours resamples in float and rounds once at the end;
  PIL rounds to uint8 between the horizontal and vertical passes. Worth ≤1/255
  and currently unmeasured — it needs a fixture where it changes the decode.
- **Vision encoder `LFM2_VL_FLASH_ATTN`.** The gate exists and has never been
  A/B'd.
- **A quiet-box timing run.** Nothing in this document is an absolute number.
