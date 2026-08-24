# LFM2.5-VL-3B — vision-language OCR backend

Engine: `src/lfm2_vl_ocr.{h,cpp}` · ShortConv decode step: `src/lfm2_shortconv.h`
Guard: `tests/test_lfm2_shortconv.cpp` (hermetic, weight-free, ~10 ms)

## NOW — active work

Branch `feat/lfm2vl-multitile`, off `main` @ `ebc8bc95`. **Multi-tile NaFlex is
implemented, validated and ON by default.** Nothing in flight.

- **DONE** — both decode bugs (§1, §2), KV decode as default, NaFlex
  single-tile resize (§3).
- **DONE** — §4 multi-tile NaFlex, default ON. Per-stage parity >0.99 global at
  every stage (`llm_logits_last` 0.999981 at F16), prompt token ids
  byte-identical over 1816 tokens, CER 0.0191 → **0.0007** on Q4_K and
  0.0127 → **0.0007** on F16, and the non-splitting canary byte-identical.
- **DONE** — bicubic resample, default ON. It was the entire per-stage gap.
- **DONE** — antialiased position-embedding resample, matching Siglip2's
  `antialias=True`. A no-op for every grid the tiler currently produces (all
  upscale from the 16x16 table) and a real fix below that; pinned against
  torch's own `F.interpolate` to 1.19e-07.
- **NEXT** — an uncontended timing run; a second splitting fixture in another
  script; the shared causal mask (memory, below). The vision encoder still
  dominates on CPU and is where perf work belongs.

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
| `LFM2_VL_FORCE_CPU` | off | ignore `crispasr_init_gpu_backend()` and run on CPU |
| `LFM2_VL_CPU_PROJECTOR` | off | run the projector MLP as a scalar CPU loop instead of a ggml graph; the graph is 6.1x faster on CPU and far more on GPU, where the scalar path was 85% of the pipeline |
| `LFM2_VL_PER_LAYER_MASK` | off | one causal-mask tensor per attention layer instead of one shared; all 8 are identical, so sharing saves ~113 MB at a 2837-token prompt |
| `LFM2_VL_FLASH_ATTN` | off | use `ggml_flash_attn_ext` in the VISION tower instead of manual attention. Off because the tower is bidirectional and passes `mask=nullptr`; per HARD RULE 5 that is full attention, not "masking handled", so the two are only equivalent while every patch is real. They are today — we never pad, unlike HF, which pads to `max_num_patches` and masks — but the gate must stay off if padding is ever introduced. Unmeasured. |
| `LFM2_VL_MULTI_TILE` | **on** | split a large page into a tile grid + thumbnail (§4); `=0` restores the single squashed tile |
| `LFM2_VL_TILE_LABELS_LEGACY_SWAP` | off | reproduce transformers <= 4.57.x, which transposed the tile row/col labels; 5.x (the default) does not |
| `LFM2_VL_BICUBIC` | **on** | PIL-matching Catmull-Rom resample, as `processor_config.json` specifies; `=0` restores the align-corners bilinear this port shipped with |
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

**(a) The row/col labels depend on the transformers version, and the first
reading of this was wrong.** `crop_image_to_patches` returns `(images,
grid_width, grid_height)` in every version, but `resize_and_split` unpacks it

```python
images, num_rows, num_cols = self.crop_image_to_patches(...)   # <= 4.57.x
images, num_cols, num_rows = self.crop_image_to_patches(...)   # >= 5.0
```

The old form made `num_rows` the grid WIDTH, transposing the labels on any
non-square grid. **Upstream fixed it.** This port shipped the 4.57.x behaviour
first, purely because 4.57.6 was what happened to be installed on the dev box —
a textbook HARD RULE 13 miss: the blueprint I read was not the code the model
is deployed against.

**Prompt-token parity caught it**, on its first real run, exactly as intended:

```
DIFF prompt_token_ids  FAIL 4/1816 ids differ,
     first at 519: mine=124910 ref=124918
```

124910 is `<|img_row_1_col_3|>`, 124918 is `<|img_row_2_col_1|>`. Four token ids
out of 1816 — invisible to every cosine in the harness, and it would have read
as "the model is weak on multi-tile" (HARD RULE 3b's blind zone, precisely).

Geometric is now the default; `LFM2_VL_TILE_LABELS_LEGACY_SWAP=1` restores
4.57.x. `tools/lfm2_vl_tiling_hf_check.py` decides which mapping to expect by
INTROSPECTING `resize_and_split`'s source rather than parsing a version string,
so it passes on either transformers and says which one it saw.

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
| multi-tile structural run, 1024x544 | 2x1 grid + thumbnail 672x352, 3 images of 32x32/32x32/42x22 patches → 256 + 256 + 231 = **743 image tokens**, exactly the oracle's number; 767-token prompt, `img_pos used=743`, no warnings |
| **decoded output on a splitting page** | **CORRECT.** The 1024x544 top strip of `commons_test_ocr_document.jpg` decodes to `Lorem Ipsum\n\nAlice was` in 8 tokens — an exact match for that fixture's ground-truth opening. So the tiling, the per-tile encode, the markup and the splice produce a readable page, not fluent nonsense. |
| **decoded A/B on a full splitting page, CUDA** | **multi-tile wins decisively** on Q4_K and F16 — the single-tile arm hallucinates, multi-tile transcribes. See "The A/B" below. |
| **CER/WER numbers** | re-running: the first attempt's ground-truth lookup keyed on a nonexistent field and scored nothing |
| per-stage parity vs a CPU-fp32 reference | `cos_min` FAILs, but that is a per-ROW minimum on a Q4_K-vs-fp32 comparison — the harness's documented false-alarm regime. `cos_global` now printed; resample tested in pixel space and rejected as the cause |

`tools/lfm2_vl_tiling_hf_check.py` is the cross-check against the real
processor. It needs torch + torchvision, so it is a developer tool, not a build
step: run it after a transformers upgrade, then regenerate
`tests/lfm2_tiling_golden.h`. The golden header itself is hermetic.

The 743-token count is the load-bearing one: the layout predicted it, the
projector produced exactly that many rows, and the splice consumed exactly
that many (`img_pos used=743`). All three orders agree.

⚠ No timing from that run is quotable — the box was at load average 24 with 41
sessions. For the record only, contended: prefill 767 tokens 362 s, decode
~63 s/token. The vision encoder was ~230–245 s **per image**, which is the whole
reason the scored A/B is a GPU job.

### The A/B — measured on Kaggle, CUDA (P100), 2026-08-24

`commons_test_ocr_document.jpg`, 1920x2485 → a 2x3 grid + a 576x448 thumbnail,
7 encoded images, **1788 image tokens / 1816 prompt tokens** — a figure the HF
processor produced independently and which the layout code matched exactly.

| arm | transcript (first 80 chars) |
|---|---|
| Q4_K, multi-tile **off** | `Lorem Ipsum⏎⏎Aurice volgam et agus non veris tincti dictad sitigis beque nere si` |
| Q4_K, multi-tile **on** | `Lorem Ipsum⏎⏎Alice was beginning to get very tired of sitting by her sister in t` |
| F16, multi-tile **off** | `Lorem Ipsum⏎⏎Aurice voluptas ego aut non veris tincti dictad sitest leges ac era` |
| F16, multi-tile **on** | `Lorem Ipsum⏎⏎Alice was beginning to get very tired of sitting by her sister in t` |

Ground truth opens `Lorem Ipsum / Alice was beginning to get very tired of /
sitting by her sister in the café`. **The single-tile arm hallucinates fluent
Latin-looking nonsense; the multi-tile arm transcribes the page.** Both quants,
same result — this is not a quantization artifact.

That is the acceptance test (HARD RULE 3), and it is the reason to read the
text rather than a summary metric: a CER number alone would have said "worse"
without saying that the old path was *inventing* text. CER/WER are being
measured in the re-run — the first attempt keyed the ground-truth lookup on a
field name that does not exist and silently scored nothing.

The canary held: `commons_example_receipt.png` gave the same 45 characters with
the gate on and off, so multi-tile does not touch a page that should not split.

### Per-stage parity — PASSED, >99% at every stage

Against a **CPU float32** blueprint reference (torch could not use the P100 at
all and fell back, which makes it the best possible reference precision), on
the splitting fixture, with the defaults as they now ship:

| stage | Q4_K cos_global | F16 cos_global |
|---|--:|--:|
| `pixel_values_img0..6` | permutation artifact, see below | — |
| `vis_post_ln_img0..6` | 0.999964 – 0.999999 | same (tower is F16 in both) |
| `projector_out_img0..6` | 0.998979 – 0.999968 | same |
| `llm_embed` | 0.999753 | 0.999753 |
| `llm_layer_0..3` | 0.999556 – 0.999709 | 0.999759 – 0.999781 |
| **`llm_logits_last`** | 0.993555 | **0.999981** |
| **`prompt_token_ids`** | **PASS — 1816 ids byte-identical** | **PASS** |

Every stage clears 0.99 global. The only figure below 0.999 is the Q4_K logits
at 0.9936, and the F16 arm reading 0.999981 on the same stage is what says that
is quantization damage and not structure — which is exactly why the dev guide
asks for both arms.

`cos_min` stays lower (0.79–0.99) because it is a per-ROW minimum on a
quantized artifact: `crispembed_diff.h` documents that regime, and `cos_mean`
sitting at 0.998–1.000 is the tell that it is a handful of fragile rows rather
than a broken tensor.

**The resample was the whole gap.** The same run with `LFM2_VL_BICUBIC=0`:

| stage | bicubic | legacy bilinear |
|---|--:|--:|
| `vis_post_ln_img6` | 0.999989 | 0.978098 |
| `projector_out_img6` | 0.999880 | **0.676896** |
| `llm_embed` | 0.999753 | **0.851482** |
| `llm_layer_3` | 0.999556 | **0.806714** |

Nothing else needed fixing. The tiling, the tile order, the per-tile encode,
the projector unshuffle, the markup and the splice were all already right.

**Confirmed clean on the final run (v6, commit `c567f3f9`):** every stage
>0.99 `cos_global`, `pixel_values_img0..6` now 0.999983 – 0.999991 after the
permutation fix, `prompt_token_ids` PASS, F16 `llm_logits_last` 0.999982, and
**no stage below 0.99**.

### The projector is bit-exact — verified offline, no GPU

The mmproj GGUF ships only `mm.1` and `mm.2`; there is no projector LayerNorm
tensor, while `Lfm2VlMultiModalProjector.forward` does
`pixel_unshuffle -> layer_norm -> linear_1 -> act -> linear_2`. That looked like
a missing operation.

It is not. Replaying our projector in numpy from the reference's own
`vis_post_ln_imgN` (its input) against `projector_out_imgN` (its output), using
`mm.1`/`mm.2` straight out of the GGUF:

| variant | cos_global | cos_min | \|mine\| | \|ref\| |
|---|--:|--:|--:|--:|
| **no layer_norm — what we do** | **1.000000** | **1.0000** | 35.44 | 35.44 |
| LayerNorm(gamma=1, beta=0) first | 0.843391 | 0.3142 | 44.82 | 35.44 |

So the released conversion has folded the LayerNorm away, and adding one back
would BREAK parity. The same test simultaneously proves the `pixel_unshuffle`
ordering and the tanh-GELU are exactly right — cosine 1.000000 leaves no room
for either to be wrong. Cost: one 92 MB download and a few seconds of numpy,
because the archive happens to contain both sides of that one stage.

⚠ **`pixel_values` needs permuting before it means anything.** HF flattens a
patch row as `(py, px, c)`, channel fastest, because its `patch_embedding` is
an `nn.Linear` over that layout. We flatten `(px, py, c)` to match how the GGUF
stores the same weight (conv-style `[kW, kH, in_C, out_C]`), and the
converter's permutation is what reconciles them. Compared raw, the stage reads
cos 0.63 with `|mine|` 546.96 against `|ref|` 547.03 — norms equal to 0.01%,
which is the textbook signature of a permutation rather than a difference
(HARD RULE 2b earning its place again). The runtime now reorders into the
reference's layout before comparing. Left as it was, it is a permanent false
FAIL sitting at the earliest stage in the archive — the worst possible place
for one.

### Perf — the projector was 85% of the pipeline

Measured on a P100, per image, BEFORE the fix:

| stage | time | share |
|---|--:|--:|
| vision encoder, 27 ViT layers, 1024 patches | 0.244 s | 4.4% |
| **projector MLP** | **4.70 s** | **85.3%** |
| prefill 1816 tokens | 1.71 s | 4.4% |
| decode | 3.41 s | 8.8% |

The projector cost **19x the entire vision transformer it post-processes** —
7 GFLOP running at 1.48 GFLOP/s on hardware that does ~9000. Three causes, all
in the same block: a hand-rolled scalar single-threaded BLAS-less matmul; a
`to_f32()` that re-dequantized 55 MB of fc1/fc2 weights on EVERY image (7x per
page); and a per-token heap allocation.

It is now a ggml graph — `mul_mat -> GELU -> mul_mat`, the same arithmetic on
whatever backend the engine is already using. **VPS (CPU): 5596 ms -> 921 ms,
6.1x**, output byte-identical. The GPU win should be much larger since the
matmuls move off the CPU entirely; that number is pending.

⚠ **The first version of this was 15.6x faster and WRONG** — the receipt
decoded as `(   )`. `us_data` is channel-major (`chan * n_proj + tok`), which is
what the scalar loop indexes and what `projector_unshuffle` is diffed in, while
a ggml tensor with `ne0 = C_us` wants `chan + tok * C_us`. The matmul consumed
a transposed input at exactly the right shape and produced fluent nonsense.
Only the A/B caught it — a speed number alone would have shipped it.

### Cost, and why the acceptance run is on Kaggle

The vision encoder is ~250 s per 1024-patch image on this VPS, and a split A4
page is SEVEN images — ~30 minutes of encode for one arm, before the ~1800-token
prefill. The gate needs four arms (off/on x Q4_K/F16, rule 4.2). Kernel:
`tools/kaggle/lfm2-vl-multitile/`, account chr1s4.

## Still open (from the port handover)

- **Multi-tile NaFlex** — DONE and default ON (§4).
- **Position-embedding antialias** — `Siglip2VisionEmbeddings.resize_positional_embeddings`
  interpolates with `antialias=True`; we do not. An exact no-op while both patch
  grid dimensions are >= the 16x16 source table, which every shape this engine
  currently produces satisfies (a 512 tile gives 32x32). It bites only for a
  small image whose grid falls under 16 in a dimension — 150x200 gives 14x20 —
  and a runtime WARNING now fires in exactly that case rather than the port
  quietly feeding the model position embeddings it was not trained on. Deferred
  on purpose: it does not touch the multi-tile parity this branch is gated on,
  and a correct antialiased resample is its own change with its own A/B.
- **Shared causal mask** — DONE. `build_prefill_graph` allocated one
  `n_tokens²` F16 mask per attention layer and all 8 held the same
  lower-triangular pattern: 1.2 MB at the 273-token single-tile prompt, 113 MB
  of it redundant at a 2837-token multi-tile one. Now one shared tensor;
  `LFM2_VL_PER_LAYER_MASK=1` restores per-layer. A/B on the canary: both arms
  `Jackson-Washington / 6640 Ortiz Cove, Markmouth`, byte-identical to the
  pre-change baseline, `audit_graph_inputs()` clean in both.
- **Bicubic resample** — DONE, and now the default. It is the dominant term in
  the per-stage gap (thumbnail projector `cos_min` 0.3446 → 0.9847), improves
  F16 decoded CER 0.5129 → 0.4968, costs ~1% wall clock, and leaves the 500x650
  canary byte-identical (`Jackson-Washington / 6640 Ortiz Cove, Markmouth`, 45
  chars, both ways). `LFM2_VL_BICUBIC=0` restores bilinear.
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
