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
  golden `prompt_token_ids` (1816/1816 exact), and on by default.
- **DONE** — §7 the LLM attention was being routed to the CPU backend by a
  `GGML_PREC_F32` stamp, with the whole KV cache copied both ways every token;
  and the decode step materialised the KV cache again on top of that.
- **DONE** — §8 `LFM2_VL_FLASH_ATTN` in the vision encoder was broken (a
  spurious permute after `flash_attn_ext`); fixed, and now default on.
- **DONE** — §9 head-to-head vs `llama-mtmd-cli` on the same GGUF: speed is a
  wash, and the quality gap was our own no-repeat-ngram default, now off. The
  receipt transcript is byte-identical to llama.cpp's.
- **DONE** — §11 the resampler is now bit-exact with Pillow (0 differing
  pixels on four page resizes), which closes the open item §9 left.
- **DONE** — §12 merged the parallel `feat/lfm2vl-multitile` session: shared
  tiling header + its 21311-check guard, the antialiased position-embedding
  resample, one shared causal mask, orchestrator/CLI wiring, and the
  blueprint-vs-port results over 8 documents.
- **NEXT** — §13: decode graph reuse, the ShortConv state round-trip, prefill,
  and an uncontended timing run. Nothing here is an absolute number. Also the
  PIL between-pass rounding experiment from §9.

## Measured — the whole lane, M1 Metal, Q4_K

`tests/regression/images/cc0`, against
`tests/regression/images/cc0/ground_truth.json` (manual transcription, no OCR
consulted). Harness: `tools/bench_lfm2_vl.py`.

**Two error rates, and both matter.** `fmt` is CER/WER after stripping markdown
table scaffolding (pipes, rule rows, `**`) from both sides. The gap is not
noise — it is the model choosing a different output FORMAT. On
commons_example_receipt the raw CER is 0.337 and the normalised one 0.092: the
receipt is read almost correctly and then emitted as a pipe table, which the
plain-text ground truth does not have. Quoting only the raw number sends you
hunting a recognition bug that is not there; quoting only the normalised one
hides a real difference in what the engine hands a caller.

### Quality: baseline vs everything, same 512-token cap

| fixture | CER | | WER | | fmt CER | | fmt WER | |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| | **base** | **now** | **base** | **now** | **base** | **now** | **base** | **now** |
| commons_example_receipt.png | 0.337 | 0.337 | 0.786 | 0.786 | 0.092 | 0.092 | 0.329 | 0.329 |
| simple_form.png | 0.372 | 0.417 | 0.444 | 0.556 | 0.364 | 0.413 | 0.444 | 0.556 |
| receipt_historical.png | 0.098 | **0.040** | 0.319 | **0.062** | 0.087 | **0.014** | 0.319 | **0.062** |
| german_official_print.jpg | 0.371 | **0.063** | 0.537 | **0.194** | 0.361 | **0.052** | 0.537 | **0.194** |
| commons_test_ocr_document.jpg | 0.702 | **0.280** | 0.903 | **0.243** | 0.701 | **0.278** | 0.903 | **0.243** |
| **mean** | 0.376 | **0.228** | 0.598 | **0.368** | 0.321 | **0.170** | 0.507 | **0.277** |

### Final state, all defaults, 1024 tokens (nothing truncates)

⚠ This table predates the §9 no-repeat-ngram flip. With the current default the
means are **fmt CER 0.106 / fmt WER 0.204** (receipt 0.092 → 0.045, German
0.052 → 0.040, historical receipt 0.014 → 0.013); see §9 for the full table.

| fixture | images | CER | WER | fmt CER | fmt WER | ms |
|---|--:|--:|--:|--:|--:|--:|
| commons_example_receipt.png | 1 | 0.337 | 0.786 | 0.092 | 0.329 | 14229 |
| simple_form.png | 1 | 0.417 | 0.556 | 0.413 | 0.556 | 4011 |
| receipt_historical.png | 2x4+thumb | 0.040 | 0.062 | 0.014 | 0.062 | 55510 |
| german_official_print.jpg | 2x3+thumb | 0.063 | 0.194 | 0.052 | 0.194 | 57715 |
| commons_test_ocr_document.jpg | 2x3+thumb | 0.024 | **0.002** | 0.021 | **0.002** | 97482 |
| **mean** | | **0.176** | **0.320** | **0.119** | **0.228** | |

WER 0.002 on a two-column book page is one word wrong in 2981 characters.

### Honest notes on these numbers

- **simple_form.png got worse** (fmt CER 0.364 → 0.413), the one regression.
  It is a 452x317 UI screenshot that smart_resize leaves at ~1:1, where the
  bicubic resampler is nearly an identity and the difference is uint8 rounding.
  Both arms miss the bottom half of the form; the new one reads `Nombre` for
  `Number` and `Recharger` for `Rechercher` where the old one got them right,
  and adds `[ ]` checkbox markers. 247 characters of ground truth at "medium"
  confidence — this is a small, low-information fixture and the parity evidence
  (cos_min 0.8159 → 0.999999 against the golden pixels) points the other way.
  Recorded, not explained away.
- **The receipt fixtures' raw CER is mostly formatting**, per the note above.
- **The decode-side changes are not bit-identical over long decodes.** Metal's
  flash-attention is not the CPU's. Byte-identical text was confirmed at 32 and
  200 tokens on two fixtures; over ~800 tokens the doc page moves from CER
  0.021 to 0.024 (WER 0.002 in both) — about 9 characters in 2981. That is the
  price of §7's 2.6x, and it is stated rather than rounded away.
- **Q4_K only.** The vision half is effectively validated at reference
  precision (the mmproj is F16 and `vis_post_ln` sits at cos_global 0.999994),
  but no F16 LLM arm was run.

### Speed

Same fixtures, all defaults, before → after (baseline column at 512 tokens,
final at 1024; neither truncates except the doc page in the baseline):

| fixture | ms before | ms after |
|---|--:|--:|
| commons_example_receipt.png | 63708 | **14229** |
| simple_form.png | 21533 | **4011** |
| receipt_historical.png | 94884 | 55510 (and 9 images instead of 1) |
| german_official_print.jpg | 69098 | 57715 (7 images instead of 1) |
| commons_test_ocr_document.jpg | 89897 | 97482 (7 images instead of 1) |

The single-tile pages are 4.5x and 5.4x faster. The pages that now split do
7-9x the vision and prefill work and still come out level or better, which is
the whole point of §5 and §7.

⚠ **Timings are RATIOS on a contended box.** Another agent's Rust build and a
busy Firefox ran through most of these; dev-guide rule 5 (identical load,
median of >= 3, quiet box) is satisfied only for the explicitly interleaved
A/Bs in §7 and §8. No absolute millisecond here is quotable.

## Env gates

| gate | default | meaning |
|---|---|---|
| `CRISPEMBED_ACCEPT_LFM_LICENSE` | off | required; LFM-1.0 is revenue-capped |
| `LFM2_VL_KV_CACHE` | **on** | KV-cached per-token decode; `=0` restores full recompute |
| `LFM2_VL_MULTI_TILE` | **on** | multi-tile NaFlex; `=0` forces the fast single-tile path |
| `LFM2_VL_BILINEAR_RESIZE` | off | pre-§4 bilinear point sampler |
| `LFM2_VL_TV_RESIZE` | off | torchvision-shaped bicubic instead of the PIL-exact one (§11) |
| `LFM2_VL_PER_LAYER_MASK` | off | one causal mask per attention layer instead of one shared (§12) |
| `LFM2_VL_LEGACY_TILE_LABELS` | off | transformers <= 4.57.x transposed `<\|img_row_R_col_C\|>` labels (§12) |
| `LFM2_VL_CPU_PROJECTOR` | off | alias for `LFM2_VL_PROJ_GGML=0` |
| `LFM2_VL_DECODE_PROFILE` | off | per-phase decode timing (build / alloc / inputs / compute / readback) |
| `LFM2_VL_PROJ_GGML` | **on** | projector on the sched backend; `=0` restores the scalar loop |
| `LFM2_VL_PROJ_GELU_TANH` | off | tanh GELU in the projector instead of erf |
| `LFM2_VL_LEGACY_RESIZE` | off | pre-§3 NaFlex resize params (factor=P, min=max=tile²) |
| `LFM2_VL_FLASH_ATTN` | **on** | `ggml_flash_attn_ext` in the vision encoder; `=0` restores manual attention |
| `LFM2_VL_ZERO_CONV_STATE` | off | debug: zero the ShortConv state cache |
| `LFM2_VL_NO_REPEAT_NGRAM` | **0 (off)** | greedy no-repeat n-gram size; `=5` restores the old default |
| `LFM2_VL_DBG` | off | diagnostics |
| `LFM2_VL_ATTN_PREC_F32` | off | stamp `GGML_PREC_F32` on the LLM attention — forces it onto the CPU on Metal |
| `LFM2_VL_KV_VIEW` | **on** | read the KV cache as a strided view; `=0` materialises it per token |
| `LFM2_VL_FORCE_CPU` | off | pin the whole engine to the CPU backend |
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

### The A/B

Same build, same box, `LFM2_VL_MULTI_TILE` the only difference. 512-token cap
except the last row:

| fixture | tiles | fmt CER single | fmt CER multi | fmt WER single | fmt WER multi |
|---|--:|--:|--:|--:|--:|
| commons_example_receipt.png | 1 | 0.092 | 0.092 | 0.329 | 0.329 |
| simple_form.png | 1 | 0.413 | 0.413 | 0.556 | 0.556 |
| receipt_historical.png | 2x4 + thumb | 0.108 | **0.014** | 0.319 | **0.062** |
| german_official_print.jpg | 2x3 + thumb | 0.268 | **0.052** | 0.425 | **0.194** |
| commons_test_ocr_document.jpg @1024 | 2x3 + thumb | 0.042 | **0.021** | 0.052 | **0.002** |

The two sub-tolerance fixtures are unchanged **to the character**, which is the
correctness check that matters: the tiling path must be a no-op below the
trigger, and it is. At the 512-token cap commons_test_ocr_document is truncated
in both arms (2188 and 2198 chars against a 2981-char ground truth), so its CER
there measures the cap rather than the feature — hence the 1024-token row.

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

## §7 — the decode was running on the CPU

`build_prefill_graph` and `build_decode_step_graph` both stamped
`GGML_PREC_F32` on their `ggml_flash_attn_ext` output. In **this fork** that is
not a hint: `ggml-metal-device.m` (CrispASR patch #83) returns `false` from
`supports_op` for any flash-attention op carrying `PREC_F32`, because Apple's FA
kernel uses `simdgroup_half8x8` tiles regardless of K type and leaks ~1e-4
against CPU. `ggml_backend_sched` therefore routed **all eight attention layers
to the CPU backend**, copying Q, K and V out of the Metal buffers and the result
back — every token.

At a 1816-position KV cache that is 8 layers × 2 tensors × 1816 × 2048 × 4 B
≈ 240 MB of GPU↔CPU round trip per token, which is why decode cost grew with
context far faster than the arithmetic does.

| | PREC_F32 (old) | Metal FA (now) |
|---|--:|--:|
| receipt, n_kv ≈ 290 | 159 ms/token | **62 ms/token** |
| doc page, n_kv 1816 | 170 ms/token | **55 ms/token** |
| `llm_logits_last` cos_global | 0.998088 | 0.998087 |
| decoded text, 32 and 200 tokens | — | identical |

`LFM2_VL_ATTN_PREC_F32=1` restores the stamp. This is also the
"lfm2/Metal-only-sched CPU-fallback assert" on the ggml-8be60f infra list: with
the stamp gone the LLM graph needs no CPU split at all.

### §7b — and it rebuilt the whole KV cache every layer, every token

The decode step read the cache as `reshape_3d → permute → ggml_cont`, to hand
`flash_attn_ext` a contiguous `[head_dim, n_kv, n_kv_heads]`. But the cache
layout already addresses that shape with plain strides — element (i, s, h) sits
at `i*4 + s*nb[1] + h*head_dim*4` — so a `ggml_view_3d` does it for free. The
`cont` was copying 238 MB per token at n_kv=1816 for no information gain.

Interleaved ×3 on the doc page, 24 tokens each, decode per token:

| | run 1 | run 2 | run 3 |
|---|--:|--:|--:|
| copies (`LFM2_VL_KV_VIEW=0`) | 76 ms | 78 ms | 79 ms |
| strided view (default) | **55 ms** | **54 ms** | **55 ms** |

1.42×, decoded text identical over 200 tokens on a 9-image page.

## §8 — `LFM2_VL_FLASH_ATTN` in the vision encoder was broken

The gate existed and had never been A/B'd. `ggml_flash_attn_ext` already
returns `[head_dim, n_heads, n_seq]` — it permutes internally — and the code
applied the manual path's trailing permute to its output as well, scrambling
heads into positions. Every shape stayed valid.

With the gate on, before the fix: `vis_post_ln` cos_global **0.563**, |mine|
1212 against a reference norm of 991, and the model answered a supermarket
receipt with *"The image shows a room with a table and chairs. There is a sign
on the wall that reads 'Welcome to our office.'"*

Skipping the trailing permute on the flash path makes it match exactly —
`vis_post_ln` cos_global 0.999994 and |mine| 991.1918 in **both** arms,
byte-identical text over 200 tokens — and it is faster: vision encoder median
**2237 → 1747 ms** over 5 interleaved reps (1.28×). So the gate is now default
ON; `LFM2_VL_FLASH_ATTN=0` restores the manual masked attention.

Same defect class as the Jun-2026 flash wave in `layout` / `math` / `deepseek`:
a spurious trailing permute after `flash_attn_ext`.

## §9 — head to head with llama.cpp, and the decode config that was costing us

The GGUF LiquidAI ships is a **llama.cpp** export, so `llama-mtmd-cli` is the
reference implementation for it. Same model, same mmproj, same prompt
("OCR this image. Output the text content."), greedy, `-n 1024`, alternating
runs on the same box. `llama.cpp` b9700 (Homebrew), format-normalised rates:

| fixture | crispembed CER | llama.cpp CER | crispembed WER | llama.cpp WER | crispembed s | llama.cpp s |
|---|--:|--:|--:|--:|--:|--:|
| commons_example_receipt.png | 0.092 | **0.045** | 0.329 | **0.257** | 30.0 | 29.8 |
| simple_form.png | 0.413 | **0.360** | 0.556 | **0.444** | 11.8 | 11.6 |
| receipt_historical.png | 0.014 | 0.013 | 0.062 | 0.062 | 106.1 | **99.0** |
| german_official_print.jpg | 0.052 | **0.037** | 0.194 | **0.119** | **63.8** | 66.6 |
| commons_test_ocr_document.jpg | 0.021 | 0.021 | **0.002** | 0.004 | **101.7** | 120.9 |
| **mean** | 0.118 | **0.095** | 0.229 | **0.185** | | |

**Speed is a wash.** Per-stage, the vision encoders are the same engine's worth
of work: 3362 vs 3377 ms on the receipt, 29959 vs 30532 on the 9-image
historical receipt, 16772 vs 16506 on the German page, 19367 vs 23510 on the
book page. Wall clock lands within a few percent either way, ours ahead on the
two largest pages and behind on one. Nothing here says either implementation is
meaningfully faster than the other on M1 Metal.

**Quality was NOT a wash, and it was our decode config.** Our greedy applied a
no-repeat-ngram constraint of 5 by default. On documents that is not a
degeneration guard, it is damage: a receipt legitimately repeats 5-grams
(`| 1 | $4`, ` Accessory | `, a column of prices), and forbidding them forces
the decoder off the correct token — which is exactly what produced the missing
PST amount and the `$48 .04` on the fixture.

Setting `LFM2_VL_NO_REPEAT_NGRAM=0`:

| fixture | fmt CER n=5 | fmt CER n=0 | llama.cpp |
|---|--:|--:|--:|
| commons_example_receipt.png | 0.092 | **0.045** | 0.045 |
| simple_form.png | 0.413 | 0.409 | 0.360 |
| receipt_historical.png | 0.014 | **0.013** | 0.013 |
| german_official_print.jpg | 0.052 | **0.040** | 0.037 |
| commons_test_ocr_document.jpg | 0.021 | 0.021 | 0.021 |
| **mean fmt CER** | 0.118 | **0.106** | 0.095 |
| **mean fmt WER** | 0.229 | **0.204** | 0.185 |

Better on three, unchanged on two, worse on none, no degeneration anywhere, and
**the receipt transcript becomes byte-identical to llama.cpp's** — 493
characters, same GGUF, same prompt, both greedy. That is the strongest parity
statement in this document: our preprocessing, SigLIP2 tower, projector, LFM2
hybrid decode and tokenizer reproduce llama.cpp exactly on a real page. So the
default is now **off**; `LFM2_VL_NO_REPEAT_NGRAM=5` restores it for a model
that does loop.

### What is left, honestly

After the flip we sit at mean fmt CER 0.106 against llama.cpp's 0.095. The gap
is two fixtures:

- **simple_form.png 0.409 vs 0.360.** A 452x317 UI screenshot that smart_resize
  leaves at ~1:1, so the resampler barely runs and the difference is uint8
  rounding — the one place where "we resample in float and round once, PIL
  rounds between the horizontal and vertical passes" can actually bite. This is
  the fixture that also regressed in §4, and it is the same suspicion.
- **german_official_print.jpg 0.040 vs 0.037.** Three thousandths of CER over
  1008 characters, i.e. about three characters, after ~700 greedy steps. Long
  decodes diverge; see the note under the measured tables.

Neither is a structural difference. The obvious next experiment is emulating
PIL's between-pass uint8 rounding in `resize_bicubic_u8_hwc` and re-running
simple_form.

## §11 — PIL is not torchvision, and now we are bit-exact with it

§4 established that HF resizes bicubic-with-antialias and swapped in
`image_preproc::resize_bicubic_u8_hwc`. That function is **torchvision**'s
`interpolate(antialias=True)`. HF's slow image processors — the ones with
`resample: 3`, which is what produced the reference — go through **PIL**, and
the two are different resamplers. Measured against `Image.resize(BICUBIC)`:

| resize | torchvision-shaped | + between-pass rounding | PIL-exact |
|---|--:|--:|--:|
| 452x317 → 448x320 (simple_form) | max 12/255, 15681 px differ | max 12/255, 11004 | **0 of 430080** |
| 500x650 → 448x576 (receipt) | max 18/255, 21075 px | max 1/255, 9 px | **0 of 774144** |
| 1920x2485 → 448x576 (doc thumb) | max 6/255, 49660 px | max 1/255, 96 px | **0 of 774144** |
| 768x1552 → 352x704 (historical) | — | — | **0 of 743424** |

Three things account for it, and all three are needed for the zero:

1. **PIL rounds to uint8 between the horizontal and vertical passes.** Carrying
   float through both is most of the 18/255.
2. **The tap count varies per output pixel** (`xmax = (int)(center + support +
   0.5) - xmin`), where the torchvision form uses one fixed `ceil(2*support)`
   per axis. One extra near-zero tap shifts the whole kernel after
   renormalisation.
3. **At the borders PIL clamps the tap RANGE and renormalises over what
   survives**; the torchvision form clamps the INDEX, i.e. replicates the edge
   pixel. On a UI screenshot with content at the edge that is visible.

And a fourth that only shows up once the first three are right: PIL's 8-bit
path is **fixed point**, quantising each normalised kernel to 22-bit integers,
and it evaluates the filter in **double**. Routing the port through the
existing float32 `cubic_kernel` left ~25% of a small downscale off by 1/255.

`image_preproc::resize_bicubic_pil_u8_hwc` is the new function;
`resize_bicubic_u8_hwc` is untouched, so the Qwen2VL lane keeps its
torchvision resampler. `LFM2_VL_TV_RESIZE=1` switches this engine back.

Per-stage effect on the multi-tile reference:

| stage (cos_global) | torchvision | PIL-exact |
|---|--:|--:|
| `vis_post_ln_img0` | 0.999994 | **0.999996** |
| `projector_out_img0` | 0.999850 | **0.999890** |
| `vis_post_ln_img6` (thumbnail) | 0.999988 | **0.999991** |
| `llm_embed` | 0.999744 | **0.999818** |

and the thumbnail's `|mine|` becomes 990.5215, which is the reference's value
exactly. End to end it is worth simple_form fmt CER 0.409 → **0.397** and
nothing anywhere else — no regression on any fixture, and preprocess got
*faster* (integer math on a uint8 intermediate). Mean fmt CER 0.106 → **0.103**
against llama.cpp's 0.095.

`tests/test_pil_resize.cpp` pins it against four literal Pillow outputs on a
deterministic 12x12 pattern plus two algebraic invariants (a constant image
survives any resize exactly — partition of unity; a hard edge never leaves
[0,255] — ringing is clipped). Watched to fail: dropping the between-pass
rounding breaks three of the four goldens AND the constant-image invariant;
using the float32 kernel breaks the 5x5 downscale on 19 of 75 elements.

### simple_form is still the outlier, and the resize was not all of it

0.397 against llama.cpp's 0.360. Our preprocessing is now bit-identical to
Pillow, i.e. to what the HF reference feeds the model, and llama.cpp resamples
with its own `clip.cpp` code that is not PIL — so on this fixture llama.cpp is
*differently* wrong rather than more correct, and it lands better by luck. It
is 247 characters of "medium"-confidence ground truth on a 452x317 UI
screenshot where both engines miss the bottom half of the form. Recorded and
left alone rather than tuned toward.

## §12 — merged from the parallel session (`feat/lfm2vl-multitile`)

A second session worked this backend at the same time, from the same
`ebc8bc95` base, on a Linux VPS and a Kaggle P100. Neither branch contained the
other. Where we overlapped the conclusions agreed — multi-tile on by default,
the projector as a ggml graph, bicubic resample, the `crispembed_diff.h` I32
bug, `cos_global` beside `cos_min`, an oracle-pinned tiling guard, the
prompt-token-id gate, and `no_repeat_ngram = 0`. Two independent
investigations, different hardware, different oracles, same answers; that is
worth more than either alone.

What came across from that branch, all of it kept:

**The tiling math itself.** `src/lfm2_vl_tiling.h` replaces the header this
session wrote: it is a superset (`compute_layout`, `build_image_markup`,
`find_closest_aspect_ratio`, the antialiased resample) and its guard
`tests/test_lfm2_tiling.cpp` runs 21311 checks against the oracle, including 21
pinned markup sequences. Rewiring this engine onto it left
`prompt_token_ids` at 1816/1816 exact and every per-stage `cos_global`
unchanged or better, which is the proof the swap was safe.

**The row/col label swap — a defect neither cosine nor CER can see.**
`crop_image_to_patches` always returns `(images, grid_width, grid_height)`, but
`resize_and_split` unpacked it as `num_rows, num_cols` in transformers <= 4.57.x
and as `num_cols, num_rows` in >= 5.0. The old form transposed every
`<|img_row_R_col_C|>` label on a non-square grid — a portrait A4 cut into 3 rows
of 2 was labelled 2 rows of 3. That branch first reproduced 4.57.x, because
4.57.6 was what happened to be installed, and the prompt-token check caught it
against a 5.x-dumped reference: 4 of 1816 ids differed, first at index 519.
Geometric is the default; `LFM2_VL_LEGACY_TILE_LABELS=1` reproduces the old
form. (This session's independent implementation happened to be geometric
already, which is why its token gate passed from the start — luck, not care.)

**Antialiased position-embedding resample.**
`Siglip2VisionEmbeddings.resize_positional_embeddings` uses
`F.interpolate(bilinear, align_corners=False, antialias=True)`; we did plain
bilinear. antialias is a no-op on upscale, so it matched every shape the tiler
currently produces and silently would not for a patch grid below 16 in a
dimension (a 150x200 image gives 14x20). The replacement is a verbatim
transcription of ATen's `_compute_weights_aa`, verified against
`F.interpolate` to 4.4e-16 and pinned by six golden resamples in the tiling
guard (worst max_abs 1.19e-07). Watched to fail: forcing support=1 reds exactly
the three downscale cases and leaves the three upscale ones exact.

**One causal mask instead of eight.** `build_prefill_graph` allocated an
`n_tokens^2` F16 mask per attention layer, all eight holding the same
lower-triangular pattern: 1.2 MB at a 273-token prompt but **113 MB at a
2837-token multi-tile one**, and multi-tile is now the default. A/B'd here:
byte-identical output, `audit_graph_inputs()` clean in both arms.
`LFM2_VL_PER_LAYER_MASK=1` restores per-layer masks.

**Orchestrator and CLI wiring.** `docs/contributing.md` point 5 was missing
entirely — lfm2_vl was reachable via `--ocr` (arch auto-detect) but absent from
`ocr_orchestrator::engine`, so `--ocr-pipeline --ocr-engine lfm2-vl` could not
select it. Added as engine 19 across `map_engine`, `run_engine`, `engine_name`,
`is_vlm_engine`, the free path, and both CLI help strings. That work also found
the multi-surface trap the dev guide calls the #1 recurring bug: there are TWO
hand-maintained VLM lists over the same set (`is_vlm_engine()` and `is_vlm` in
`main.cpp`), and being in one but not the other resolved `model_a` down the
DETECTOR branch — the engine was handed a DBNet path and failed inside the
vision graph rather than at load. Both sites now name each other.

**Tooling**: the blueprint-vs-port Kaggle harness, the multi-tile acceptance
kernel, the expanded oracle, the HF cross-check, and a reference dumper that
emits per-image vision stages and pre-weight pixel values.

### Port vs the Python blueprint, across 8 real documents

From that branch, and the most valuable number in this document: every other
check compares us to a hand transcription, which conflates *is the port
faithful* with *is the model any good at this page*. Running
`Lfm2VlForConditionalGeneration.generate()` itself — greedy, same prompt, same
budget — isolates the first. Blueprint on CPU float32, ours Q4_K on CUDA:

| fixture | ours vs blueprint | ours vs GT | blueprint vs GT |
|---|--:|--:|--:|
| commons_test_ocr_document.jpg | **0.0000** | 0.0007 | 0.0007 |
| receipt_historical.png | 0.0065 | 0.0130 | 0.0117 |
| german_official_print.jpg (Fraktur) | 0.0201 | 0.0476 | 0.0387 |
| german_kurrent_handwriting.jpg | 0.0255 | — | — |
| german_official_document.jpg | 0.0457 | — | — |
| public_domain_formula_photo.jpg | 0.3320 | — | — |
| arabic_handwriting.jpg | 0.4679 | — | — |

**Prompt token counts match the blueprint on all 8**, up to a 3x3 grid plus
thumbnail at 2591 tokens. Where ground truth exists we track the blueprint
closely, i.e. most of the residual Fraktur error is the MODEL's, not the port's.

⚠ That branch's own correction, kept because it is the more useful half of the
result: `handwritten_letter.jpg` scored CER 0.0000, but BOTH sides emit the
single character `"A"`. That is agreement on a degenerate output, not a
transcription, and it was quoted twice as "byte-identical on a 10-image page"
before the text was read. What it legitimately proves is its 2591-token prompt.

⚠ **Its CER numbers are not directly comparable to the tables above.** That
harness normalises with `re.sub(r"\s+", " ", s.strip())`, which collapses
NEWLINES; the tables in this document keep line structure and only collapse
runs of spaces. On a two-column book page the model's line breaks differ from
the ground truth's while the words do not, which is the whole of the gap
between its 0.0007 and this document's 0.024 on the same fixture — and note
that this document's **WER** on it is 0.002, i.e. the two agree once the metric
does. Neither normalisation is wrong; quoting one against the other is.

### And what that branch does not have

Everything in §7, §8 and §11 is Metal-only or PIL-only and could not surface on
a VPS or a P100:

- it still stamps `GGML_PREC_F32` on the LLM attention, so on Metal all eight
  attention layers run on the CPU with the KV cache copied both ways per token
  (§7). CUDA and CPU ignore the flag; only this fork's Metal backend refuses
  the op, which is exactly why a CUDA/CPU investigation cannot see it;
- it still `ggml_cont`s the whole KV cache per layer per token (§7b);
- its `LFM2_VL_FLASH_ATTN` vision path still carries the spurious permute (§8) —
  default off, so latent, but it is a landmine;
- its resample comment calls `image_preproc::resize_bicubic_u8_hwc`
  "PIL-matching"; measured, it is not (§11);
- it still materialises the whole 128000x2048 Q6_K embedding table per image.

## §13 — where the time goes now, and what is next

**The decode step, profiled** (`LFM2_VL_DECODE_PROFILE=1`), M1 Metal, Q4_K,
63 steps, ms/token:

| | receipt, n_kv ~290 | doc page, n_kv 1830 |
|---|--:|--:|
| graph build | 0.46 | 0.11 |
| `sched_reset` + `sched_alloc_graph` | 3.11 | 2.76 |
| input upload | 0.02 | 0.03 |
| **GPU compute** | **100.13** | **105.49** |
| readback + ShortConv state | 0.65 | 0.54 |

That retires two items this document previously listed as the next targets.
Graph reuse and moving the ShortConv state into a backend buffer are together
worth **at most ~3.5 ms of 105**; `GGML_SCHED_DEBUG=2` shows the decode graph is
a single split entirely on MTL0, with no CPU fallback left after §7. And the
cost barely moves with context (100 vs 105 ms at n_kv 290 vs 1830), so it is
fixed per-token weight traffic and kernel-dispatch overhead, not attention.

For scale: the model is 1.67 GB, so one pass over the weights at M1's ~68 GB/s
is ~24 ms — we are ~4x off that, and `llama-mtmd-cli` on the same file measures
in the same range or slower (§9). Closing it means fusing ops beyond what
llama.cpp does, which is a project, not a patch.

Still open:

- **Prefill.** ~11–27 s for a 1800–2300-token multi-tile prompt is the other
  half of a split page, and it has not been profiled at all.
- **`--ocr-max-tokens` is ignored by `--ocr-pipeline`** — it affects all eight
  VLM engines, not just this one. Inherited from the parallel session's open
  list; not fixed here.
- **simple_form.png**, the one fixture where we sit behind llama.cpp (§9, §11).
- **F16 arm.** Everything here is Q4_K. The vision half is effectively at
  reference precision (F16 mmproj, `vis_post_ln` cos_global 0.999996); the LLM
  half has no F16 measurement on this box.
- **A quiet-box timing run.** No absolute millisecond in this document is
  quotable; the interleaved medians in §7, §8 and §11 are.
