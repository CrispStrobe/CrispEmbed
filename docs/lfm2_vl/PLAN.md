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
- **NEXT** — §9: decode graph reuse, the ShortConv state round-trip, prefill,
  and an uncontended timing run. Nothing here is an absolute number.

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
| `LFM2_VL_PROJ_GGML` | **on** | projector on the sched backend; `=0` restores the scalar loop |
| `LFM2_VL_PROJ_GELU_TANH` | off | tanh GELU in the projector instead of erf |
| `LFM2_VL_LEGACY_RESIZE` | off | pre-§3 NaFlex resize params (factor=P, min=max=tile²) |
| `LFM2_VL_FLASH_ATTN` | **on** | `ggml_flash_attn_ext` in the vision encoder; `=0` restores manual attention |
| `LFM2_VL_ZERO_CONV_STATE` | off | debug: zero the ShortConv state cache |
| `LFM2_VL_NO_REPEAT_NGRAM` | 5 | greedy no-repeat n-gram size |
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

## §9 — where the time goes now, and what is next

M1 Metal, Q4_K, after §4–§8:

| stage | receipt (1 tile, n_kv 290) | doc page (7 tiles, n_kv 1816) |
|---|--:|--:|
| preprocess | ~20 ms | ~230 ms |
| vision encoder | ~1750 ms | ~1750 ms × 7 |
| projector | ~8 ms | ~8 ms × 7 |
| prefill | ~1300 ms | ~8000 ms |
| decode | ~62 ms/token | ~55 ms/token |

Still open:

- **Graph reuse in decode.** The graph is rebuilt, `sched_reset` and
  `sched_alloc_graph` run, and ~600 Metal dispatches are re-encoded every
  token. Building once against the full `max_seq` KV with a mask input would
  remove all of it.
- **ShortConv state round-trip.** 22 layers × (one read + one write) of CPU↔GPU
  state per token. It could live in a backend buffer with the shift in-graph.
- **Prefill.** ~8 s for 1816 tokens is the other half of a multi-tile page.
- **Bicubic rounding.** Ours resamples in float and rounds once at the end; PIL
  rounds to uint8 between the horizontal and vertical passes. Worth ≤1/255 and
  currently unmeasured — it needs a fixture where it changes the decode.
- **`to_f32(token_embd)` was 1.05 GB per image** — the whole 128000×2048 Q6_K
  table dequantized on every `generate()` to read the ~2 k rows a page uses.
  Replaced by a row-at-a-time `embed_lookup`; `llm_embed` is bit-identical.
- **A quiet-box timing run.** Nothing in this document is an absolute number:
  every measurement here was taken beside another agent's Rust build and a busy
  Firefox. The interleaved medians are trustworthy as ratios; the milliseconds
  are not.
