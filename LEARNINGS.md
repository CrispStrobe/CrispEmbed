# CrispEmbed — Technical Learnings

## DAT: Conv+BN fusion silently skipped on F32 models (to_f32 returns t->data) (2026-06)

`dat_sr.cpp` parity vs a *genuine* reference (the real PyTorch DAT-light run on
weights reconstructed from the f32 GGUF — `tools/dump_dat_reference_from_gguf.py`)
was only cos 0.9906 (vs the other SR engines' 0.999+). Root cause: the init-time
Conv+BN fusion dequantized weights with the file's `to_f32(t, buf)` helper, which
— like several `*_to_f32` helpers in this repo — **returns `t->data` directly for
GGML_TYPE_F32 tensors and leaves the passed `buf` empty** (only the quantized/F16
paths fill `buf`). The fusion code then read `cw = dequant(...)` and guarded with
`if (!cw.empty() && !bw.empty())`, so on an **F32 model** every conv's `cw` was
empty, fusion was skipped, and the BatchNorm in the AIM dwconv / channel- and
spatial-interaction branches was **dropped entirely** (there is no separate BN
application — fusion is the only BN path). On F16 models the bug is hidden because
`to_f32` fills `buf`. Fix: in the dequant lambda, when `to_f32` returns a pointer
≠ `buf.data()`, copy it in (`buf.assign(p, p + ggml_nelements(t))`). Output cos
0.9906 → 0.999995; all 20 captured stages ≥0.99998.

**Lesson:** when a `*_to_f32(t, buf)` helper has a fast F32 path that returns
`t->data` without touching `buf`, never use `buf.empty()`/`buf.size()` as a
proxy for "did I get the data" — use the returned pointer. This silently
miscompiles only on F32 (not F16/quant) models, so it survives F16-only testing.
Genuine ground truth (real model run) was needed to catch it: a self-consistent
ref reverse-engineered from the same engine would have hidden it.

## SwinIR shifted-window: shift sign must match the precomputed attn_mask (2026-06)

SwinIR's `swinir_sr.cpp` produced an output whose `test-swinir-diff` cosine read
−0.91 (looked anti-correlated). Two distinct issues were tangled together:

1. **The real bug — `cyclic_shift` sign convention.** The shifted (odd-index)
   Swin blocks roll the feature map by `ws/2`, partition into windows, and add a
   *precomputed* `attn_mask` (loaded from the GGUF) that blocks attention between
   the regions that wrap around at the bottom/right edges. That mask is built for
   `torch.roll(-ws/2)` (= numpy `np.roll(x, -ws/2)`, i.e. `out[y]=in[(y+ws/2)%H]`).
   The engine's `cyclic_shift(..., -ws/2)` computed `in[(y-ws/2)%H]` — the
   **opposite** roll. Forward and reverse shifts still cancelled (the round-trip
   was self-consistent), so the bug was invisible except where it interacts with
   the mask: the **edge / wrap-around windows** got the mask meant for the other
   convention, mixing token regions that should be blocked. Result: divergence
   localised at image edges in the shifted blocks, compounding through the four
   RSTBs (rstb_0 cos 0.9994 → rstb_3 max_abs 147, engine ≈ 2× ref at edges).
   **Fix:** forward shift `+ws/2`, reverse `−ws/2` so partition order and mask
   align. All stages then cos ≥ 0.99997, output (pre-clamp float) cos 0.999996.
   Lesson: when a precomputed mask encodes a geometric convention, the shift /
   partition / reverse must all match *that* convention end-to-end; a self-
   consistent-but-opposite round trip silently corrupts only the masked windows.

2. **A red-herring test metric.** `crispembed_diff::Ref::compare` reports
   `cos_min` = the worst per-row cosine, where row size = `shape.back()`. For the
   SR output stored CHW with gguf-reversed shape `[256,256,3]`, that "row" is **3
   horizontally-adjacent pixels in one channel**, and the C++ side is uint8-
   clamped to `[0,1]` while the ref is raw float (can be negative). A single
   near-zero edge triple where clamping disagrees in sign drives `cos_min` to
   −0.91 even when the image is essentially identical (global cos 0.999996). The
   previous session correctly flipped the shift, saw `cos_min` go −0.91 → −0.001,
   and wrongly concluded the flip "wasn't the fix." Always sanity-check a diff
   harness's reduction (worst-row vs global) before trusting a single scalar —
   especially on quantized/clamped outputs. `test_swinir_diff.cpp` now gates on
   the image-level (global + per-RGB-channel) cosine.

## Porting a scalar conv engine to ggml_conv_2d: reverse the kernel ne (2026-06)

When converting an SR/restoration engine from scalar conv nested loops to a
`ggml_conv_2d` graph (pan_sr was the latest), the recurring trap is the conv
kernel's `ne`. The Python converters write each PyTorch weight `[OC,IC,KH,KW]`
with a plain `astype` — no permute — so in the GGUF the *data* is KW-innermost
(row-major `[OC][IC][KH][KW]`, exactly what the scalar `weight[o*ic*kh*kw +
c*kh*kw + ky*kw + kx]` indexing assumes) but the stored `ne` is `[OC,IC,KH,KW]`.

`ggml_conv_2d(ctx, kernel, input, ...)` wants the kernel as `ne=[KW,KH,IC,OC]`
over that same byte layout. So at weight-prep time **reverse the four ne axes
and copy the raw dequantized buffer unchanged** — do not permute the data:

```cpp
// src ne = [OC,IC,KH,KW] (PyTorch order), data KW-innermost
int64_t ne[4] = { t->ne[3], t->ne[2], t->ne[1], t->ne[0] }; // -> [KW,KH,IC,OC]
ggml_tensor * w = ggml_new_tensor(gw_ctx, GGML_TYPE_F32, 4, ne);
ggml_backend_tensor_set(w, dequantized_src, 0, ggml_nbytes(w)); // bytes as-is
```

Feeding the native ne instead aborts in `ggml_im2col` with `GGML_ASSERT(OW>0)`
because it reads `OC` as the kernel width (e.g. 40 > input W=32 → negative OW).
Do **not** use `ggml_n_dims` to detect conv kernels — a 1×1 weight `[OC,IC,1,1]`
reports 2 dims and would be left un-reversed. Key off the `.weight` name suffix
and always treat those as 4D. Biases (`.bias`, 1-D) are copied as-is.

Verification corollary: the diff harness feeds the C++ engine a uint8-quantized
input (`round(x*255)/255`), so the torch reference generator must snap its input
to the same 1/255 grid. Skip this and a ±1/255 input perturbation amplifies
through a 4× SR network to ~0.4 max-abs, dropping one image row to cos ≈0.9959
even though the math is correct. Matched, pan's graph and scalar both reach
cos_min 0.999997. See `tools/dump_pan_reference_from_gguf.py`.

## ggml_flash_attn_ext accepts non-contiguous Q/K/V (2026-06)

`ggml_flash_attn_ext` on both Metal and CUDA handles non-contiguous inputs
natively — it does NOT require `ggml_cont` wrappers around `ggml_permute` on
Q, K, or V before the flash attention call. Adding `ggml_cont` is redundant
and forces an extra allocation + copy at the Metal side. The correct pattern:

```cpp
Q = ggml_permute(gctx, Q, 0, 2, 1, 3);   // [hd, T, H, 1] — non-contiguous OK
K = ggml_permute(gctx, K, 0, 2, 1, 3);
V = ggml_permute(gctx, V, 0, 2, 1, 3);
ggml_tensor * attn = ggml_flash_attn_ext(gctx, Q, K, V, mask, scale, 0.f, 0.f);
```

Previously the code had `ggml_cont(ggml_permute(...))` — removed in
`decoder_embed.cpp` (`29d8a08`) and `bidirlm_vision.cpp` (`fd8cd09`).

Note: for KV-cache *outputs* that you `ggml_set_output` and read back via
`ggml_backend_tensor_get`, you DO still need `ggml_cont` (see the cross-
backend KV-cache audit entry below) — the non-contiguous exception only
applies to flash_attn *inputs*.

## SIMD dot_product as the universal optimization lever (2026-06)

A single `dot_product()` function with AVX2+FMA/NEON in `core/cpu_ops.h`
accelerated 30+ runtimes because every CPU-side matmul, attention dot
product, and LSTM gate computation reduces to the same inner loop:
`sum += a[i] * b[i]`. By making `linear_cpu` and `mha_1q_cpu` delegate
to `dot_product()`, all existing callers got SIMD for free without any
per-engine changes. Key pattern: dual-accumulator unroll (16-wide) with
single-accumulator cleanup (8-wide) plus scalar tail — handles any
dimension without alignment requirements.

## DequantCache: init-time vs per-call weight dequantization (2026-06)

The biggest single performance win across the codebase was caching
dequantized weights. The anti-pattern: `to_f32(tensor)` returns a new
`std::vector<float>` each call, so a 30-layer decoder running 512 steps
re-dequantizes the same immutable weights `30 * 4 * 512 = 61K` times.
The fix: `DequantCache` struct with `unordered_map<void*, vector<float>>`
keyed on `tensor->data`. First call dequantizes; subsequent calls return
the cached pointer. Thread-safety: one cache per inference context (not
global static — that was the math_ocr bug).

## Thread-local buffers eliminate per-call allocations in shared primitives (2026-06)

Hot-path functions like `mha_1q_cpu`, `swiglu_ffn`, and `layernorm2d_cpu` were
allocating `std::vector<float>` on every call (per head, per position, per step).
`thread_local std::vector<float>` with lazy resize eliminates all heap allocation
after the first call while remaining thread-safe. Pattern: check `size() < N`,
resize if needed, use `.data()`. The buffer persists across calls on the same
thread. For callers with pre-existing scratch buffers, offer an optional
`float* scores_buf = nullptr` parameter so they can pass their own.

## BatchNorm fusion into conv weights at load time (2026-06)

For Conv→BN sequences in eval mode, BN can be algebraically folded into the
preceding conv's weights: `new_W[o] = bn_scale[o] * conv_W[o]`,
`new_b[o] = bn_scale[o] * conv_b[o] + bn_shift[o]`, where
`bn_scale = bn_weight / sqrt(bn_var + eps)` and
`bn_shift = bn_bias - bn_mean * bn_scale`. This eliminates the BN pass
entirely at runtime. Applied to TBSRN (11 conv+BN pairs) and DAT SR (54
conv+BN pairs across 3 patterns: dwconv, channel_interaction, spatial_interaction).
Key detail: the
fused weights go into a separate `fused` map checked first by `get()` — if
found, returns the fused version; otherwise falls through to DequantCache.
Output differs by ±1-2 pixel values due to float associativity
(`scale * (W*x)` ≠ `(scale*W) * x`), which is expected and acceptable.

## ggml_backend_sched vs ggml_gallocr for repeated inference (2026-06)

`ggml_gallocr` checks whether reallocation is needed on every `alloc_graph`
call (walks the graph to compare sizes). `ggml_backend_sched` with
`sched_reserve` pre-allocates for a bucket size, then `sched_alloc_graph`
just assigns pointers — much faster for repeated same-shape calls. The
T-bucketing pattern (8/16/32/64/128/256/512) reduces re-reserves when input
lengths vary slightly. For LFM2 (350M, 16 layers), graph+alloc overhead
dropped from ~2ms to ~0.7ms per call, but compute at ~700ms dominates —
the win is primarily architectural (aligns with the encoder path, enables
future GPU dispatch via the scheduler's multi-backend support).

## BPE merge: linked list + priority queue for O(N log N) (2026-06)

The standard BPE merge loop finds the lowest-rank adjacent pair, merges it,
and repeats. Naive implementation: O(N) scan per iteration × O(N) erase = O(N²).
Fix: doubly-linked list of symbol nodes + min-heap of `(rank, left_node_id)`.
Pop the best pair, merge left←right, requeue affected neighbors. Stale entries
(where the pair text no longer matches the rank) are filtered on pop by
re-checking the merge table. Total: O(N log N). Applied to both `bpe.h`
(used by lightonocr, granite_speech) and `tokenizer_bpe.cpp` (all BPE models).

## T5 attention uses scale=1.0 (no 1/sqrt(d)) (2026-06)

Unlike standard Transformer attention where `scores = QK^T / sqrt(d_k)`,
T5 uses raw dot products with scale=1.0. When using `ggml_flash_attn_ext`,
pass `scale=1.0f` explicitly. The relative position bias is added as a mask
tensor. In the pix2struct encoder, there's no relative bias (positions come
from row/col embeddings) — pass `nullptr` mask + `scale=1.0f`.

## Hann-window tiling for SR/restoration models (2026-06)

All SR/denoise models can process arbitrary image sizes via overlapping
tiles with raised-cosine blending. Key formula for the blend weight at
each pixel: `w = hann_y * hann_x` where `hann(p) = 0.5 - 0.5*cos(pi*p/overlap)`
in the overlap zone, 1.0 elsewhere. Accumulate weighted outputs + weight
map, divide at the end. Tile size must be aligned to the model's
downsample factor (e.g. 16 for a 4-stage U-Net with 2x per stage).

## Per-engine confidence: softmax without modifying logits (2026-06)

When adding per-token confidence to all 15 OCR engines, the key insight:
compute `exp(logits[best] - max_logit) / sum_exp` in a separate block
WITHOUT overwriting the logits array. Many engines have debug prints
referencing raw logit values after the argmax. Overwriting logits for
softmax breaks those prints. The one-pass approach
`conf = 1.0 / sum_exp` (since `exp(best - max) = exp(0) = 1` when best
IS the max) is both faster and non-destructive.

## llama.cpp GGUF ≠ CrispEmbed GGUF (2026-06)

llama.cpp splits VLMs into LLM + mmproj GGUFs with different tensor
names (`blk.N.attn_q` vs our `llm.layers.N.attn.q`, `v.blk.N` vs
`vis.blocks.N`). When a model is only available as llama.cpp GGUFs
(e.g. german-ocr-3.1), use `merge-llamacpp-qwen2vl-gguf.py` to:
1. Read both GGUFs (no gguf pip dependency — standalone reader)
2. Map tensor names via regex
3. Merge metadata from both sources
4. Write single combined GGUF with our naming convention

## Qwen2-VL engine handles 4+ architectures (2026-06)

The `qwen2vl_ocr` engine reads all hyperparams from GGUF metadata with
prefix probing for `qwen2vl.*` and `qwen3vl.*`. This
means it handles:
- Qwen2-VL (original)
- Qwen2.5-VL (updated)
- Qwen3-VL / FireRed-OCR (new attention patterns)
- Nanonets-OCR2 (pruned Qwen2-VL, 16L vs 28L)

Key: never hardcode layer counts or dimensions — always read from GGUF.

## Reference GGUF consistency trap (2026-06)

MixTex showed cos=-1.0 on decoder parity, causing 3 hours of debugging.
Root cause: the reference dumper captured encoder stages from a
preprocessed synthetic image, but decoder stages from `model.generate()`
which used ViTImageProcessor preprocessing — different encoder outputs
feeding the same decoder reference. Fix: always run encoder+decoder
from the SAME input in the reference dumper, or use the reference's
own `enc_layernorm` output to drive the Python decoder comparison.

## ggml_add doesn't support mixed types (2026-06)

`ggml_add(f32, f16)` crashes with `binary_op: unsupported types`.
`ggml_cast` in the graph SHOULD convert F16→F32, but some code paths
read model weights directly via `ggml_backend_tensor_get` assuming F32
layout — this reads wrong data sizes from F16 tensors (`tensor read out
of bounds`). Fix: use `tensor_to_f32()` helper that reads raw bytes via
`ggml_nbytes()` then dequantizes via `ggml_get_type_traits()->to_float`.

## DeepSeek-OCR-2: from a never-run port to character-perfect OCR (2026-06)

> **Status: WORKING.** Character-perfect OCR on Metal + q4_k. The history below
> is preserved because the failure modes (stub converter, scrambled/aliased KV
> cache, missing instruction prompt, wrong normalization) are instructive — jump
> to "RESOLVED (2026-06)" for the fixes that landed it.

`deepseek_ocr2.cpp` + `convert-deepseek-ocr2-to-gguf.py` were committed in a
single feat commit and **never ran end-to-end** — the published GGUF
(`cstr/deepseek-ocr2-crispembed-GGUF`) will not even load. Diagnosed via the
HF blueprints (`deepencoderv2.py`, `modeling_deepseekocr2.py`,
`modeling_deepseekv2.py`) + a metadata dump:

1. **Converter is a stub — no tensor renaming.** It does
   `for name in header: writer.add_tensor(name, ...)`, emitting raw HF names
   (`model.sam_model.*`, `model.qwen2_model.model.model.layers.N.*`,
   `model.layers.N.*`). The engine loads short names (`v.*`, `qe.*`, `l.*`), so
   it finds *zero* tensors. An audit of all converters shows this is the only
   complex-VLM converter with no rename map (lightonocr=20, pix2struct=19,
   firered=17, layout=14, decoder-embed=10, qwen2vl/internvl2/glm/got=6-8). The
   renames=0 outliers otherwise are simple SR/restoration nets (esrgan, swinir,
   scunet, …) whose engines match the raw names — those are fine.
2. **merges written as array-of-arrays.** tokenizer.json stores merges as
   `[a, b]` pairs; the converter wrote them as a GGUF nested array (elemtype 9),
   which ggml rejects ("invalid GGUF type 9"). Must flatten to `"a b"` strings
   (same fix as Qari). FIXED in the converter.
3. **Tensor names exceed GGML_MAX_NAME (64).** Because of (1), names like
   `model.qwen2_model.model.model.layers.N.post_attention_layernorm.weight` (70
   chars) blow the ggml limit — a free consequence of not renaming.

Engine bugs found by blueprint comparison (use_mla=False → standard
`LlamaAttention`, so an agent's MLA RoPE/scale findings were false positives):
- **MoE gate**: config `norm_topk_prob=False`, `routed_scaling_factor=1.0` →
  use the raw top-k softmax probs; the engine renormalized them. FIXED.
- **Qwen2 vision encoder** (`CustomQwen2`): blueprint concatenates
  `[visual, queries]`, applies a token-type mask (visual↔visual bidirectional;
  queries→all-visual + causal-among-queries), and returns `y[:, n_query:]`. Our
  engine has the order reversed, is fully bidirectional, and returns the first
  half. NOT fixed (needs a loadable GGUF to verify).

**Progress (2026-06):** the converter now has the full HF→engine rename map and
deepseek_ocr2 is wired into the `--ocr` arch dispatcher, so the model **loads
(2707 tensors) and runs through the SAM stack without crashing**. Two crashes
fixed along the way: (a) the SAM downsample `net_3` outputs 896 channels (the
Qwen2 dim), not the config's nominal `downsample_channels` 1024 — derive the
channel counts from the weight `ne[1]`, not a hardcoded 1024 (was an OOB read);
(b) the LLM rmsnorm multiplied an f32 activation by the f16 norm weight, which
ggml's elementwise ops reject — `ensure_f32` the weight.

The Qwen2 vision encoder forward had 5 bugs (all confirmed vs `deepencoderv2.py`
`CustomQwen2Decoder`, which subclasses `Qwen2Model`, `rope_theta=1e6`):
(1) concat `[visual, queries]`, not `[queries, visual]`; (2) a token-type
attention mask (visual↔visual bidirectional; queries→all-visual + causal-among-
queries) — the engine was fully bidirectional; (3) **RoPE is applied**
(positions 0..T-1) — the engine omitted it; (4) return `y[:, n_vis:]` (the query
half), not the first half; (5) apply the final `qe.output_norm`. All five FIXED.

**RESOLVED (2026-06): character-perfect OCR on Metal + q4_k.** The key
unlock was the user's insight — quantize to q4_k and run on the Metal GPU
(`cmake -B build -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON`) instead of
fighting CPU speed. With Metal active (~20 s prefill, MoE expert dispatch
parallelized across `n_threads` with `std::thread`), the pipeline became fast
enough to iterate. The remaining bugs, in the order they unblocked the output:

1. **KV-cache axis scramble.** K/V were `permute(0,2,1,3)`'d before flattening
   to `[nkv*hd, T]`, transposing the token and head axes vs the reload's
   `reshape_3d(hd, nkv, n_past)`. Flatten with a plain `cont()+reshape`, no
   permute — exactly the verified qwen2vl_ocr path. This alone fixed the first
   generated token.
2. **KV-cache buffer aliasing** (the "first token right, rest garbage"
   signature). `k_out`/`v_out` were views sharing the *same* `cont(K/V)` buffer
   the attention path consumes; under the no-alloc scheduler that buffer is
   recycled once attention reads it, so prefill computed the right first token
   but **cached garbage**. Give the cache outputs their own `cont` and
   `ggml_build_forward_expand` them (they are not ancestors of `layer_output`).
   Isolation that nailed this: `DS_NO_KV` (recompute the full sequence each
   step) produced " Paris." for "The capital of France is" while the cached
   path produced "Paris vro vro…".
3. **Prompt construction.** The decoder was fed 256 placeholder tokens with no
   bos, no view-separator and no instruction. The HF `infer` + plain template
   builds `[bos] + <image>*256 + <view_sep> + tokenize("Free OCR.")`; the 257
   image/sep slots are masked-scatter-replaced by `[global_features(256),
   view_seperator(1)]`. Assemble that as an embedding matrix directly (text
   slots from `embed_tokens`, image slots from the projector, separator from the
   learned `v.view_separator`). `image_token_id` is **128815**, not the Qwen
   `151643` the old heuristic assumed; eos is **1** (`<｜end▁of▁sentence｜>`).
4. **Byte-level BPE I/O.** Added a `core_bpe` encoder (merges loaded from the
   GGUF) to tokenize the instruction, and the **inverse** GPT-2 byte map on
   decode so pieces render as text (`Ġ`→space) instead of raw byte-unicode.
5. **Image preprocessing.** DeepSeek-OCR2 uses `mean=std=0.5` ([-1,1]), **not**
   CLIP normalization, and `ImageOps.pad` (aspect-preserving resize + gray
   border), not a stretch resize. With CLIP mean/std the model hallucinated a
   markdown table; with the correct preprocessing it reads the page verbatim.

The diagnostics added for this (`DS_DBG`, `DS_NO_KV`, `DS_TEXT_TEST`) are
env-gated and left in. The `crispembed-quantize` tool keeps the MoE router
(`*.mlp_gate.weight` / `ffn_gate_inp`) and the `qe.*` Qwen2 encoder at Q8_0,
which is what makes q4_k safe for this MoE.

### Speed (~15×) + portability + a numerics red herring (2026-06)

Three findings from making it fast and verifying it on macOS/Metal:

1. **The qwen2 encoder was the one stage NOT on the GPU** — `encode_qwen2` was
   pure CPU-scalar (naive O(T²) attention + per-token `linear_cpu`/`swiglu_cpu`),
   ~9 min of vision on an M1 while SAM and the decoder were already ggml graphs.
   Ported it to `build_qwen2_enc_layer_graph` (one graph over all `T=n_vis+n_query`
   tokens, no KV cache), reusing the in-file `build_llm_layer_attn` pattern: NEOX
   `ggml_rope_ext`, GQA interleave, `ggml_soft_max_ext` + a precomputed F16
   bidirectional/query mask. **Vision (SAM+enc+proj) ~9 min → ~37 s.** Verified
   bit-equivalent to the scalar path (per-layer `cos_min` matched to 5 digits);
   `DS_QWEN2_SCALAR=1` keeps the scalar reference for A/B.

2. **Decoder use-after-free, masked by the slow encoder.** `build_llm_layer_attn`
   built its graph in a **local** `std::vector meta` buffer and *returned the
   graph* — the buffer freed on return, leaving `gf` dangling. Latent UB: fine on
   Linux (freed heap intact), `EXC_BAD_ACCESS` in `ggml_backend_sched_alloc_graph`
   on macOS. It never surfaced before because every run died/timed-out in the
   9-min scalar encoder, *before* reaching decode-step-0. The fast encoder exposed
   it immediately. Fix: own the meta buffer **in** the returned struct (move
   preserves `data()`), matching the SAM pattern (caller-scoped buffer). With this
   the full OCR runs end-to-end in ~2 min and is character-perfect.

3. **The encoder's parity "failure" is a metric artifact, not a bug.** Against an
   fp32 PyTorch reference the encoder output looks broken (`cos_min`→−0.07,
   `cos_mean`~0.5 over 24 layers). But an **independent naive-fp32 NumPy
   reimplementation diverges identically** (cos_mean 0.57 vs the C++ 0.50) — so
   the gap is inherent fp32-vs-PyTorch-SDPA sensitivity on this model's
   **attention-sink massive activations** (token 0, channel 570, growing to ~410),
   not a code error. Confirmed not quantization: q8_0-roundtripping the weights in
   PyTorch keeps cos_mean 0.9995. `cos_min` is misleading here (dominated by the
   one massive channel) — judge the encoder by `cos_mean`, or just by the OCR
   text, which is correct. Lesson: before chasing a "bug" on a massive-activation
   model, reproduce the reference path naively in fp32 — if *it* also diverges,
   the divergence is numerical, not yours.

**Then the decoder MoE was the whole budget.** With the encoder on the GPU,
`moe_ffn_cpu` (scalar, per-token, re-dequantizing q4_k experts every step) was
~99% of LLM time: ~2000 ms/layer prefill, ~47 ms/layer/token decode; attention
was already negligible. Ported it into the layer graph via `ggml_mul_mat_id`,
reusing crispembed.cpp's BERT-MoE pattern (router→softmax→`ggml_top_k`→`get_rows`
weights→`mul_mat_id` gate/up/down→weighted sum + a combined shared expert).
The 64 per-expert tensors are stacked once at load into `[in,out,n_exp]`
(`stack_moe_experts`, a `memcpy` of each quantized expert into its slice — same
shape/type so blocks stay aligned; +~1.3 GB, gated so the CPU path doesn't pay
it). Per-layer prefill ~2015 ms → ~50 ms (~40×); **full OCR ~121 s → ~43 s**,
byte-identical output. `DS_MOE_CPU=1` keeps the scalar path.

**Then vision — and the surprise was the convs, not the attention.** Per-stage
timers showed SAM's 12 attention layers (already Metal) were only ~3 s; the rest
was scalar CPU: the **neck/downsample `conv2d_cpu`** (3.7-8 s, thread-variance-
prone) and patch embed (~2 s). Threading both (exact) roughly halved them, then
porting the neck/downsample to `ggml_conv_2d` (`build_sam_neck_graph`: 4 convs +
2 channel-axis LayerNorm-2d via permute→`ggml_norm`→affine→permute) dropped it to
**~150 ms** (~20-40×). Gotcha: the conv kernels are Q8_0 (vision floor), and you
**cannot reshape a quantized `[768,256]` to `[1,1,768,256]`** (`ne[0]=1` breaks
the 32-block) — dequant to F32 and feed as graph inputs. SAM ~12 s → ~4.7 s,
`sam_output` cos unchanged (0.999253). `DS_SAM_CONV_CPU=1` keeps the CPU chain.
Net: full OCR ~9 min (start of session, never completed) → **~23 s**, character-
perfect. Remaining costs: model load (~5-12 s, cold disk + Metal buffer copy) and
the SAM attention (~4 s, hard to flash-attn due to decomposed rel-pos bias).

Harness notes from the hunt: the per-stage diff was comparing the **wrong
tensors** (pre-neck 4096×768 vs the final 256×896 SAM output; pre-norm full-seq
vs the post-norm query-half encoder output) — the dead `diff_ref_path` is now
wired to a `DS_REF` env var with corrected comparison points + per-layer
bisection. `tools/dump_deepseek_ocr2_reference.py` was rewritten to instantiate
the vision modules standalone from `deepencoderv2.py` (the bundled MoE
`modeling_deepseekv2.py` won't import on transformers ≥4.48 — `LlamaFlashAttention2`
was removed), so the reference GGUF builds on CPU without the 3.4B decoder.

## Qwen2-VL (Qari-OCR) parity: four independent bugs, four layers

Qari-OCR (a Qwen2-VL-2B Arabic OCR fine-tune) produced garbage. The diff
harness (`test-qwen2vl-diff`) plus a PyTorch ground-truth comparison localized
**four** independent bugs, one per layer of the stack. Methodology: dump the
HF model's vision-merger output, token_ids and per-layer LLM hidden states for a
real image, inject them into the C++ engine via test hooks (`GEN_FROM_REF`,
`LLM_FROM_REF`), and bisect.

1. **Vision MLP activation.** Qwen2-VL `VisionMlp` uses `ACT2FN[hidden_act]`
   with the *vision* config default `hidden_act="quick_gelu"` (`x·σ(1.702x)`),
   NOT the merger's exact `nn.GELU()`. The engine used `ggml_gelu_erf` for both.
   Fix: `ggml_gelu_quick` for the ViT block, keep `ggml_gelu_erf` for the
   merger. (vis_layer_0 cos 0.995 → 0.99999.)

2. **Vision 2D-RoPE inv_freq.** `VisionRotaryEmbedding` is built with
   `dim = head_dim/2`, so `inv_freq[j] = theta^(-2j/(head_dim/2))`. Using
   `head_dim` in the denominator makes the frequencies decay half as fast —
   a subtle error (layer 0 still ~0.995) that compounds over 32 layers to
   destroy the merger (cos 0.06). Shared by Qwen2.5-VL (same
   `VisionRotaryEmbedding(head_dim//2)`), so the fix is correct for both.
   (vis_merger 0.37 → 0.99; last_logits 0.96 → 0.9999.)

3. **`last_logits` validates only the LAST position.** The prefill's final-token
   logit matched HF at 0.9999, yet generation was garbage. For the last token,
   causal == bidirectional attention, so a correct last-token logit does NOT
   prove the per-position KV is right. Always validate intermediate positions
   (per-layer hidden states across the WHOLE sequence) before trusting prefill —
   `LLM_FROM_REF` (inject HF embeds) gave cos 0.99997 across all 714 positions
   and proved the LLM forward correct, isolating the bug to generation.

4. **No-cache decode dropped the image; KV-cache outputs were pruned.** The
   single-token decode fell back to a full-recompute path that called
   `run_llm_forward` WITHOUT the image (image-blind → the model "describes a
   blank page" / answers conversationally). Pass the image through. The KV-cache
   path itself had two bugs that only surfaced once the recompute was fixed:
   - **(a) side outputs pruned.** `k_out_N`/`v_out_N` are not ancestors of the
     logits, so `ggml_build_forward_expand(gf, logits)` dropped them and
     `ggml_graph_get_tensor` returned null → silent no-cache. Call
     `ggml_build_forward_expand` on each side output explicitly.
   - **(b) cached V was a view into a reused buffer.** In the decode step
     `K_new` comes from `ggml_rope_multi` (materialized) but `V_new` was a
     `ggml_reshape_3d` *view*. Marking a view as a graph output and reading it
     back under a no-alloc scheduler returns GARBAGE — the scheduler reuses that
     buffer for later ops. Symptom: the cached K matched HF to ~1e-3 but the
     cached V was off by 6-9 (massive-activation magnitudes) at scattered
     elements; generation matched for 1-2 tokens then collapsed. Fix:
     `ggml_cont` K_new/V_new before `ggml_set_output`. (The prefill already did
     `reshape_2d(ggml_cont(V), ...)`; the decode forgot the cont.)
   General rule: **any tensor you `ggml_set_output` and read back via
   `ggml_backend_tensor_get` under a no-alloc scheduler must be materialized
   (`ggml_cont`), not a view** — otherwise its storage is fair game for reuse.

### Cross-backend audit of the view-as-output KV bug (2026-06)

Two KV-cache designs exist across the autoregressive decoders:
- **Read-back into a host vector** (qwen2vl, lightonocr, deepseek_ocr2): compute
  K/V in the graph, `set_output`, `ggml_backend_tensor_get` into `std::vector`,
  feed back as inputs next step. **Vulnerable** to both the prune bug (side
  outputs not expanded) and the view bug (V is a reshape view). This is the
  pattern that broke Qari.
- **`ggml_cpy` into a persistent KV buffer** (got_ocr, glm_ocr, internvl2_ocr):
  allocate persistent K/V tensors, `ggml_build_forward_expand(gf, ggml_cpy(K,
  k_view))`. No host read-back, cpy materializes. **Safe** — prefer this pattern
  for new backends.

Audit result: `lightonocr.cpp` had the cont bug AND three more (see next
section); now fixed and matching HF. `deepseek_ocr2.cpp` had the same read-back +
reshape-view pattern — the cont fix is applied (provably correct), but it is
NOT verified end-to-end (no local model) and, like lightonocr, may have further
architecture bugs needing a diff-vs-HF pass. got/glm/
internvl2 are safe (cpy pattern). All CPU-scalar decoders (mixtex, bttr, hmer,
posformer, ppformulanet*, granite_vision, math_ocr, decoder_embed) have no ggml
decode graph → immune.

## LightOnOCR-2-1B (Pixtral ViT + Qwen3): four bugs, fixed via HF diff

LightOnOCR looped ("ALIEN…", then digit garbage) despite a git note claiming the
KV cache was "confirmed working". A PyTorch diff (dump `vision_tower` /
`multi_modal_projector` / `language_model` layer outputs, compare per-row cos)
localized **four** bugs — the projection ones were the blockers (proj cos was
≈0, i.e. random):

1. **KV-cache V was a reshape view** marked `set_output` and read back → garbage
   (the cross-backend bug above). `ggml_cont` before `set_output` + expand the
   side outputs.
2. **Pixtral 2D-RoPE built interleaved but applied rotate-half.** The cos/sin
   table used interleaved layout (`dim[2i]`=h, `dim[2i+1]`=w) and `freqs[i]` for
   both axes, but `apply_rope` is rotate-half. Pixtral
   (`PixtralRotaryEmbedding`) is rotate-half: `freqs=1/theta^(2k/dim)`, height
   uses `freqs[::2]`, width `freqs[1::2]`, angle vector `[h(dim/4)|w(dim/4)]`
   **repeated** across the two halves (idx j and j+dim/2 share the angle).
   Verified numerically equal to HF (max diff 0.0).
3. **Projector RMSNorm in the wrong place.** `Mistral3MultiModalProjector` is
   `norm → patch_merger → linear_1 → gelu → linear_2`; the C++ applied the norm
   *last*. The norm is an RMSNorm over the per-patch D-dim vision features,
   **before** the 2×2 merge.
4. **Patch-merger merge order was patch-major, not channel-major.** Mistral3's
   `Mistral3PatchMerger` uses `F.unfold`, which lays out the `D·merge²` vector
   **channel-major**: `[c0·(k00,k01,k10,k11), c1·…]`. The C++ concatenated whole
   patches (`[patch00's D ch | patch01's D ch | …]`) — a permutation the merging
   weight can't undo (→ proj cos ≈ 0, random). Fix: `dst[c*msq + kpos] = src[c]`.

Result: first token and OCR text match HF exactly ("Qari OCR parity smoke test /
Invoice number: QA-2026-0616 / Total due: $42.75 / Please return plain text
only."). `vis_out` cos ≈0.99 / `proj_out` ≈0.88 vs fp32 HF is just q4_k
quantization — the greedy output is still exact.

**Diagnostic tells:** a VLM that says "I'm just a text-based assistant" or
paraphrases the OCR instruction (often in Chinese, the Qwen base language) is
not seeing the image — suspect splice/decode, not vision. Conversely, output
that's coherent-but-wrong-content with a correct first token points at
per-position KV, not the prefill logit.

**Out-of-distribution caveat:** Qari is an *Arabic full-page* OCR model. Test it
on rendered Arabic (PIL + raqm, `direction='rtl'`), not sparse English — with a
describe-style prompt and a tiny image it legitimately answers "blank page",
matching HF.

## GELU variant matters for token classification

HuggingFace/PyTorch uses erf-exact GELU (`torch.nn.functional.gelu`), not the
tanh approximation from the original BERT paper. For embedding retrieval the
difference is negligible (cos ~0.9999), but for token classification (NER) the
small per-value differences (~1e-4) compound through 12 layers and flip argmax
on borderline tokens. Fix: always use `ggml_gelu_erf` for BERT/XLM-R models.

Before fix: 2/4 entities detected (missing Apple ORG, Cupertino LOC).
After fix: 4/4 entities match Python exactly, all scores > 0.997.

## Cased vs uncased BERT tokenizer auto-detection

BERT-cased models (e.g. `dslim/bert-base-NER`) require case-preserving
tokenization. The GGUF doesn't store `do_lower_case`. Detect from vocab:
if single-letter uppercase tokens ("A", "B", ...) exist in the vocab, it's
a cased model. WordPiece tokenizer must skip `tolower()` in that case.

Wrong casing: "Barack" → ["bar", "##ack"] (6 subwords, all predicted O).
Correct casing: "Barack" → ["Barack"] (1 token, predicted B-PER with 0.999).

## BiACM: attention score fusion, not embedding fusion

LiLT's BiACM (Bidirectional Attention Complementation) adds text and layout
attention **scores** before softmax — NOT the embeddings themselves. Each stream
maintains separate Q/K/V projections and separate FFN layers. Only the attention
score matrices are shared (added element-wise at each layer). This means
`ggml_flash_attn_ext` cannot be used — scores must be computed manually
(`Q @ K^T`), summed, then passed to `ggml_soft_max` before applying to V.

## Layout embedding concatenation order

LiLT layout embeddings: 6 position lookups concatenated in this exact order:
`x(x0), y(y0), x(x1), y(y1), h(y1-y0), w(x1-x0)` → 6×128 = 768d.
Getting h/w swapped (w before h) causes cos=0.28 at the embedding level,
cascading to cos=-0.35 at layer 11. Getting it right: cos=1.000000 on all layers.

## RoBERTa position IDs start at 2, not 1

RoBERTa uses `padding_idx=1`, so positions start at `padding_idx + 1 = 2`.
Using offset 1 (like standard BERT) causes cos=0.97 at layer 0, degrading
to cos=0.80 by layer 11 — subtle enough to look like a precision issue
but actually a systematic bug. Fixed: `pos_ids[i] = i + 2` for RoBERTa/XLM-R.

## crispembed-diff test: always use matching inputs

Parity tests must use identical inputs between Python reference and C++.
The LiLT diff test initially used hardcoded bboxes in C++ that didn't match
the Python dumper's dynamically-generated bboxes — cos dropped to 0.20 at
the layout embedding level. Fix: store `input_ids` and `bbox` in the
reference GGUF and read them in the C++ test.

## Shared library pattern: conditional fallback

When extracting code into shared `crisp_*/` libraries from CrispASR:
- Guard CrispASR-specific code (auto-download, model registry) with
  `#ifdef CRISPASR_BUILD` — set via CMake `target_compile_definitions`
- Both repos check `EXISTS "${CRISP_*_DIR}/CMakeLists.txt"` and fall
  back to local copies when the sibling repo is absent
- Link order matters on MinGW: consumer before provider
- Always run full unit tests (439 in CrispASR) after the refactor

## Qwen2-VL vs Qwen2.5-VL config field names

The vision config schema differs between Qwen2-VL and Qwen2.5-VL:

| Field | Qwen2-VL | Qwen2.5-VL |
|-------|----------|------------|
| ViT block dim | `embed_dim` (1280) | `hidden_size` (1280) |
| Merger output | `hidden_size` (1536) | `out_hidden_size` (2048) |
| FFN size | `mlp_ratio` (4) → computed | `intermediate_size` (3420) |
| Input channels | `in_chans` (3) | `in_channels` (3) |

Critically, Qwen2-VL's `hidden_size` is the **merger output** (= LLM input dim),
not the ViT block dim. The GGUF `vision.hidden_size` must be set to `embed_dim`
(the ViT block dim), not `hidden_size`. Getting this wrong means every attention
head_dim computation in the engine is wrong → garbage output, no obvious crash.

Fix: use `getattr(vc, 'embed_dim', vc.hidden_size)` for the block dim,
`getattr(vc, 'intermediate_size', None) or embed_dim * mlp_ratio` for FFN.

## NAFNet dequant: always use ggml_get_type_traits for quantized tensors

When loading quantized GGUF weights for CPU-scalar inference, the `to_f32()`
helper must handle all quantized types — not just F32 and F16. The pattern:

```cpp
const auto * traits = ggml_get_type_traits(t->type);
if (traits && traits->to_float) {
    traits->to_float(t->data, buf.data(), n);
}
```

Failing to do this (e.g. `memset(buf, 0, ...)` for unknown types) silently
produces zero weights → the model runs but outputs garbage. The cosine drops
from 0.998 to 0.932 — high enough to look "plausible" but clearly wrong.
This was caught because Q8_0 and Q4_K produced *identical* cosines (both
were reading zeros), which shouldn't happen if real dequantization is working.

## NAFNet quantizer: per-channel scale factors are add operands

NAFNet's `beta` and `gamma` tensors `[1, C, 1, 1]` are used as element-wise
multiply operands in residual connections (`x = input + block_output * beta`).
Despite being 4D, they have only C elements and are extremely sensitive to
quantization. The quantizer's `is_add_operand` guard must include them:
`.beta` and `.gamma` patterns alongside `.ls1`/`.ls2` (LayerScale).

NAFNet conv weights (1x1 and 3x3 DW) all have ne[0] < 32 (the Q8_0 block
size), so the quantizer's existing `ncols % qk != 0` guard already skips
them. The beta/gamma guard is defense-in-depth.

## Hough deskew: run Sobel on grayscale, not binarized image

For document deskew via Hough transform, running Sobel edge detection on
the raw grayscale image works much better than binarizing first then running
Sobel. Binarization destroys gradient information at text boundaries,
especially for thin lines and anti-aliased edges at small angles. The top-5%
edge threshold (not top-10%) gives enough votes for reliable angle detection.

## ggml quantized reshape: dequant BEFORE reshape

ggml quantized types (Q8_0, Q4_K, etc.) store data in fixed-size blocks
(e.g. Q8_0 = 32 elements/block). `ggml_reshape_4d` changes shape metadata
without moving data. If the reshape creates `ne[0]` smaller than the block
size (e.g. ne[0]=3 for a 3×3 conv kernel), subsequent operations that
read the quantized data will access invalid block boundaries → crash.

**Rule**: Always dequant quantized tensors to F32 BEFORE reshaping to
arbitrary dimensions. Then reshape, then cast to F16 if needed.

Also: ggml only supports Q→F32 dequantization (`ggml_cast`), not Q→F16.
Attempting `ggml_cast(Q8_0, F16)` hits `GGML_ABORT("fatal error")` in
`ggml_compute_forward_dup`. Always go Q→F32→F16 (two casts).

## GPU backends: read weights via ggml_backend_tensor_get, not t->data

Models that mix a ggml graph (GPU) with CPU-scalar fallback code — like
`surya_det.cpp`, whose LiteMLA + decode head stay scalar — must not read
weight bytes through `tensor->data`. On a CPU backend `t->data` is the host
buffer, but on a GPU backend (Metal/CUDA) it is not a valid host pointer, so a
direct `memcpy(out, t->data, …)` or `traits->to_float(t->data, …)` reads
garbage (or crashes). Route every host-side weight read through
`ggml_backend_tensor_get(t, dst, 0, ggml_nbytes(t))`, which copies out of the
tensor's own buffer regardless of backend, then dequantise from that staging
buffer. (Apple Silicon's unified memory happens to make `t->data` work for
Metal in many cases, but relying on that is non-portable and breaks on CUDA.)

Companion lesson: switch the backend with `ggml_backend_init_best()` (not a
hardcoded `ggml_backend_cpu_init()`) and gate `ggml_backend_cpu_set_n_threads`
behind `ggml_backend_is_cpu()` — it is a CPU-only call. Provide an env override
(e.g. `SURYA_DET_FORCE_CPU=1`) so parity debugging can pin the CPU path.

## Per-layer parity comparison must happen inside the layer loop

When comparing C++ per-layer outputs against a reference GGUF, the
comparison must happen immediately after each layer completes — NOT
in a separate loop after all layers finish. The hidden state buffer
is overwritten by each subsequent layer, so a post-loop comparison
would compare every layer's reference against only the final layer's
output (producing spurious failures on early layers and a spurious
pass on the last layer).

## SAM ViT windowed attention: LN before partition

SAM ViT-B applies LayerNorm BEFORE window partitioning. For windowed
layers in a ggml per-layer graph:
1. Apply LN1 on CPU to the full (unpartitioned) hidden state
2. Window-partition BOTH the LN'd state (graph input) and the original
   state (residual input)
3. The graph uses `skip_ln1=true` to avoid double-normalization
4. The residual connection uses the original (pre-LN) partitioned data

Getting this wrong (LN after partition) introduces zeros from padding
tokens into the LayerNorm statistics, corrupting edge windows.

## DeBERTa rel_embd must be dequantized for CPU-side expansion

DeBERTa's relative position embeddings are expanded on CPU (log-bucket
indexing → [H, T*T] tensor) before the ggml graph runs. With quantized
models (Q8_0/Q4_K), the `rel_embd.weight` tensor is no longer F32 —
reading it via `ggml_backend_tensor_get` gives raw quantized bytes.
Must use `tensor_to_f32_backend()` which reads raw bytes then calls
`ggml_get_type_traits(type)->to_float()` to dequantize. Same applies
to `encoder_ln_w/b` used in the LayerNorm applied to rel_embd.

## Dual-backbone GLiNER: parameterize span mode and hidden dim

GLiNER models differ in span representation mode:
- markerV1 (LFM2): concat(proj_start, proj_end, proj_first) → 3*hidden
- markerV0 (DeBERTa): concat(proj_start, proj_end) → 2*hidden

The out_project MLP input dimension changes accordingly. Parameterize
`span_cat_dim` based on span_mode rather than hardcoding `3*hidden`.
Also parameterize `head_dim_gl` (GLiNER head dimension) separately from
`enc_hidden` (encoder output dimension) to handle the 768→512 projection.

## PARSeq two-stream decoder (XLNet-style attention)

PARSeq's decoder uses a two-stream design from XLNet where both position
queries and content tokens are maintained separately. Key details:

1. **Token ordering is non-standard**: `[EOS=0, chars=1..94, BOS=95, PAD=96]`.
   The head output excludes BOS and PAD, so it has 95 classes (0=EOS + 94 chars).
   This is because `BaseTokenizer` puts `specials_first=(EOS,)` before charset
   and `specials_last=(BOS,PAD)` after.

2. **Context construction**: The content stream at decode position k is NOT just
   the token embedding. It's `pos_queries[k-1] + embed(token_k)` for k>=1, and
   just `embed(BOS)` for k=0 (no position query for BOS — it's "null context").

3. **norm_c is essential**: Context K/V in self-attention are normalized by
   `norm_c` (LayerNorm), while queries are normalized by `norm_q`. Skipping
   norm_c produces garbage.

4. **Efficient AR decode**: At step i, only one query position is used
   (`pos_queries[i]`), with context tokens 0..i. No causal mask needed since
   T=i+1 and all positions are visible. The paper's `query_mask` only matters
   for the full N-step forward (training/refinement).

5. **Non-square patch kernel**: Patch embedding uses Conv2d with kernel [4,8]
   (height 4, width 8). ggml_conv_2d doesn't support non-square kernels, so
   patch embedding runs CPU-side as a manual extract+matmul.

## ggml GQA broadcasting (critical for decoder models)

`ggml_mul_mat` natively broadcasts ne[2] when `b->ne[2] % a->ne[2] == 0`.
For GQA (16 Q heads, 8 KV heads): **do NOT explicitly repeat K/V**.
`ggml_repeat` tiles `[h0..h7, h0..h7]` which is WRONG for GQA (should
be `[h0,h0,h1,h1,...]`). Just let mul_mat broadcast — it handles the
interleaved head mapping correctly internally.

Also: after attention, reshape to `q_dim = n_heads × head_dim` (NOT
`hidden_size`). For GQA models, q_dim ≠ hidden_size (e.g. 2048 vs 1024).

## BERT post-LN vs pre-LN

BERT uses post-LayerNorm: `attn → residual_add → LN → FFN → residual_add → LN`.
Many newer models (GPT, LLaMA) use pre-LN. Getting this wrong produces
output that looks plausible but has completely wrong magnitudes.

## RoPE application order

For Qwen3: RoPE is applied on `[head_dim, n_heads, T]` tensor (BEFORE
permute to `[head_dim, T, n_heads]`). `ggml_rope_ext` requires ne[2]=T
(the position dimension), which matches the unpermuted layout. Applying
RoPE after permute crashes with dimension mismatch.

At position 0, RoPE is identity (cos=1, sin=0), so position-0 values
match regardless of whether RoPE is applied. Debug with position > 0
to verify RoPE correctness.

## Tokenizer types for embedding models

| Model family | Tokenizer | Implementation |
|---|---|---|
| BERT/MiniLM/GTE | WordPiece | Greedy longest-match with ## prefix |
| XLM-RoBERTa/E5/Arctic/PIXIE | SentencePiece Unigram | Viterbi DP (NOT bigram merge) |
| Qwen3/Octen/F2LLM | GPT-2 BPE | core_bpe byte-level BPE with merges |
| Gemma3/Harrier-270M | SentencePiece BPE | BPE merges with ▁ space marker + BOS/EOS |

Auto-detected from GGUF metadata: `tokenizer.ggml.type` (0=WP, 1=BPE, 2=SP)
or heuristic (vocab > 100K → SentencePiece).

### Critical: SentencePiece Unigram needs Viterbi, not bigram merge

The llama.cpp-style bigram merge (priority queue, highest-score-first)
does NOT produce correct tokenization for Unigram models like XLM-R.
Example: "▁world" exists as token 8999, but bigram merge breaks it into
["▁w", "or", "ld"] because greedy pair merging can't find the global optimum.

**Viterbi DP**: For each position i, try all vocab tokens ending at i,
pick the segmentation with the highest total score. O(n × max_token_len).
This matches HuggingFace's `tokenizers` library exactly.

### SentencePiece BPE vs GPT-2 BPE

These are different tokenizer families with different pre-processing:
- GPT-2 BPE: byte-level encoding (spaces → Ġ), no BOS/EOS by default
- SentencePiece BPE (Gemma): spaces → ▁ (U+2581), BOS/EOS tokens

### Vocab scores for SentencePiece

SentencePiece Unigram models need per-token scores for Viterbi. These come from:
1. `tokenizer.sp_model.GetScore(i)` — but not available for all tokenizer classes
2. `tokenizer.json` → `model.vocab` → list of `[token, score]` pairs

If scores are missing (all zeros), the tokenizer degenerates to random merging.

## Per-op debugging methodology

Same as CrispASR: dump every intermediate tensor from BOTH HF reference
and our ggml graph, compare at each stage. The divergence point identifies
the exact broken operation. For Octen-Embedding-0.6B, this revealed:
- input_ln: MATCH
- q_proj/k_proj: MATCH
- q_norm/k_norm: MATCH
- o_proj: MISMATCH → GQA repeat was wrong
- Fix: remove ggml_repeat, let mul_mat broadcast → MATCH

## RoBERTa/XLM-R position embedding offset

RoBERTa-family models (XLM-R, PIXIE-Rune, arctic-embed-l-v2) offset position
IDs by `padding_idx + 1 = 2`. Position IDs for a 4-token sequence are
`[2, 3, 4, 5]`, not `[0, 1, 2, 3]`. Position embedding index 1 is all-zeros
(padding), index 0 is low-norm. Getting this wrong produces ~0.74 cosine sim
instead of 0.999.

Stored as `bert.position_offset` in GGUF metadata.

## Gemma3 architecture specifics

Gemma3 (Harrier-270M) differs from Qwen3/LLaMA in several critical ways:

1. **RMSNorm uses `(1 + weight)`**: Gemma3 RMSNorm computes
   `output * (1.0 + weight)` instead of `output * weight`. The stored weights
   do NOT include the +1 offset. Missing this makes all layer outputs wrong.

2. **Embedding scale**: Token embeddings are multiplied by `sqrt(hidden_size)`.
   The exact value is stored in `embed_tokens.embed_scale` (f16 precision:
   `sqrt(640) ≈ 25.25` not `25.298`).

3. **Extra norms**: 4 norms per layer (not 2):
   - `input_layernorm` → before attention
   - `post_attention_layernorm` → after attention, BEFORE residual add
   - `pre_feedforward_layernorm` → before FFN
   - `post_feedforward_layernorm` → after FFN, BEFORE residual add

4. **Attention scaling**: Uses `query_pre_attn_scalar` (= head_dim) instead
   of `sqrt(head_dim)`. Scale = `1/sqrt(qpas)`.

5. **gelu_pytorch_tanh**: Activation function; ggml_gelu uses tanh approx.

6. **head_dim != hidden_size/n_heads**: Gemma3 has head_dim=256, hidden=640,
   n_heads=4. Standard calculation gives 160, but explicit head_dim is 256.

7. **SentencePiece BPE tokenizer**: Uses ▁ space marker (not GPT-2 Ġ),
   needs BOS(2) at start and EOS(1) at end.

## Ollama integration learnings

### Architecture: Ollama uses ggml via CGO (same as CrispEmbed)

Both Ollama and CrispEmbed use ggml for tensor computation. Ollama wraps ggml
ops in Go structs via CGO (`C.ggml_mul_mat`, `C.ggml_rms_norm`). CrispEmbed
calls ggml directly from C++. The computation graphs are functionally identical.

### Phantom-space token vocabulary (critical for WordPiece)

Ollama's WordPiece tokenizer expects tokens in SentencePiece-style format:
- `"hello"` → `"▁hello"` (prepend ▁)
- `"##ing"` → `"ing"` (strip ##)
- `"[CLS]"` → `"[CLS]"` (keep special tokens)

Without this transformation, cos drops from 1.0 to ~0.19.

### GELU variant matters (exact erf vs tanh approximation)

BERT uses exact GELU (erf-based). Ollama's `.GELU()` uses tanh approximation
(`ggml_gelu_inplace`). Must use `.GELU_ERF()` for BERT/XLM-R encoder models.
Difference: cos 0.996 → 1.000.

### SentencePiece Unigram needs Viterbi DP, not pairwise merge

Ollama's existing `SentencePiece` tokenizer uses BPE-style greedy pairwise
merge (priority queue). This is WRONG for Unigram models (XLM-R, e5-small).
We added `SentencePieceUnigram` using Viterbi DP (same as CrispEmbed's
tokenizer_spm.cpp). Must also prepend space before tokenization.

### Gemma3 (1+weight) RMSNorm must be pre-baked for Ollama

Ollama's RMSNorm does `rms_norm(x) * weight`. Gemma3 needs `rms_norm(x) * (1 + weight)`.
CrispEmbed handles this at runtime with a `ones` tensor. For Ollama export,
pre-add +1 to all norm weights in the GGUF.

### Quantized token_types breaks Ollama binary ops

Ollama's ggml doesn't support `f32 + q8_0` in elementwise ops. The tiny
`token_types.weight` tensor (2 rows) must be kept as f32 during quantization.
Error: `binary_op: unsupported types: dst: f32, src0: f32, src1: q8_0`.

### Nil-guards needed for optional model components

Ollama's Qwen3 model.go unconditionally calls `QueryNorm.Forward()` — panics
for models without QK-norm (e.g. Jina v5). Gemma3 embed.go unconditionally
iterates `Dense` projection — panics for models without it (Harrier-270M).

### Jina v5 LoRA adapters need merge before export

Jina v5 models use task-specific LoRA adapters (retrieval, classification,
clustering, text-matching). Must call `model.set_adapter("retrieval")` then
`model.merge_and_unload()` before GGUF export. The `encode()` method does
more than standard forward+pool, so merged output won't exactly match HF.

### SentencePiece BERT models should use bert arch, not xlmr

Models like multilingual-e5-small report `model_type="bert"` with SentencePiece
tokenizer. These are BERT models (no position offset), not XLM-R. Only true
`roberta`/`xlm-roberta` types need the `xlmr` arch with position offset.

`paraphrase-multilingual-MiniLM-L12-v2` is another instance of this pattern —
BERT (post-LN) body + 250K-token XLM-R SentencePiece-Unigram vocab. The
converter detects this from `config.model_type == "bert"` and writes
`bert.position_offset = 0`. End-to-end cosine vs HF: **1.000000** on f16/f32,
**197/197 encoder tensors bit-exact** (max\|Δ\|=0) — see
`tests/parity_layers_bert.py`.

### SPLADE detection must look at checkpoint files, not state_dict

`AutoModelForMaskedLM.from_pretrained()` will *silently random-initialise*
missing `cls.predictions.*` keys instead of failing. Checking
`any("cls.predictions" in k for k in model.state_dict())` therefore returns
True for **every** plain encoder checkpoint, baking a random MLM head into
the GGUF (~600 KB of garbage tensors that load as "MLM/SPLADE head loaded"
at runtime).

The fix in `models/convert-bert-to-gguf.py` is to peek at the safetensors /
pytorch_model.bin header directly via `safe_open()` and only call
`AutoModelForMaskedLM` if `cls.predictions.` or `lm_head.` keys are
**actually present in the checkpoint**. `output_loading_info=True` looked
like an obvious alternative but returns inconsistent shapes (single model
vs 5-tuple) depending on `use_safetensors`, so the header-peek path is the
robust one.

This bug affected every plain `sentence-transformers/*` and `all-MiniLM-*`
conversion prior to 2026-05-11. Re-converting those models drops the file
size by ~1 MB each and removes the misleading "MLM head loaded" log line.

## PPFormulaNet-S / Texo-Distill OCR port

### MBart uses PRE-LN, not POST-LN

Despite MBart config saying `layer_norm_eps` and having `*_layer_norm` weights,
the HuggingFace MBart decoder applies **PRE-LN**: LayerNorm before attention/FFN,
with the residual connection skipping the LN. The TrOCR decoder (math_ocr.cpp)
uses POST-LN. Getting this wrong produces completely different logit distributions
— the first token diverges from logit 16.1 (correct) to 1.7 (wrong).

```
PRE-LN (MBart):                    POST-LN (TrOCR):
  residual = x                        Q = linear(x)
  x = LN(x)                          ...attn...
  Q = linear(x)                      x = x + attn_out
  ...attn...                          x = LN(x)
  x = residual + attn_out
```

The encoder diff test (cos=1.0) will NOT catch decoder LN ordering bugs —
you MUST also dump and compare decoder layer outputs from the Python reference.

### ODR violations from shared struct names

Multiple `.cpp` files defining `struct dec_layer` in the anonymous namespace
causes One Definition Rule violations. The linker may silently use the wrong
definition (144 bytes from decoder_embed_internal.h instead of 208 bytes from
ppformulanet_ocr.cpp), causing heap-buffer-overflow in `map_tensors`. ASAN
catches this immediately. Fix: use unique struct names (`ppfn_dec_layer`).

### UniMERNet preprocessing is NOT ImageNet

PPFormulaNet-S/Texo uses UniMERNet's image processor:
- Convert to grayscale, replicate to 3ch
- Resize preserving aspect ratio, pad with **black** (not white)
- Normalize: **mean=0.7931, std=0.1738** (NOT ImageNet 0.485/0.229)
- Input is always 384x384

Using ImageNet normalization produces garbage output even though the encoder
activations look reasonable — the model was trained with different pixel statistics.

### HGNetv2 StemBlock padding

StemBlock uses kernel_size=2 convolutions (stem2a, stem2b) with padding=0.
Before each, the input must be explicitly padded with `F.pad(x, (0,1,0,1))`.
Without this, the spatial dimensions mismatch at the concat step (pool output
vs stem2b output differ by 1 pixel).

### Conv-BN folding for CNN encoders

BatchNorm after Conv2d can be algebraically folded at conversion time:
```
fused_w = conv_w * (bn_weight / sqrt(bn_var + eps))
fused_b = bn_bias - bn_mean * (bn_weight / sqrt(bn_var + eps))
```
This eliminates all BN parameters from the GGUF, saving memory and compute.
The BTTR/HMER ports already did this; PPFormulaNet has ~150 BN layers to fold.

### 20M models are too small for Q4_K

The Texo-distill model (20M params, 384 d_model) produces identical output at
F32/F16/Q8_0, but Q4_K degrades noticeably — subscripts become wrong, tokens
repeat. The attention projections (384x384) and embedding table (1264x384) are
small enough that 4-bit quantization loses critical precision. Ship Q8_0 (22 MB)
as the smallest reliable variant.

### Debug prints: gate behind env vars, never remove

Decoder debug fprintf traces (`tok_emb+pos`, `after embed_ln`, `logits[91]`)
were essential for diagnosing the PRE-LN bug. Gate them behind
`getenv("PPFN_DEBUG")` rather than deleting. The crispembed-diff harness only
validates the encoder — decoder bugs require manual layer-by-layer tracing.

## Quantization notes

### Python gguf vs C++ quantizer

The Python `gguf` library (`pip install gguf`) only implements quantization
for basic types: Q4_0, Q5_0, Q5_1, Q8_0. K-quants (Q4_K, Q5_K, Q6_K) are
listed in the enum but `quantize_blocks` raises `NotImplementedError`.

Additionally, the Python library's string array handling in GGUFReader/GGUFWriter
can corrupt metadata when copying GGUF files — we observed Q8_0 models from the
Python quantizer producing cos=0.78 vs the same model's F32, while the C++ quantizer
produces cos=0.9997.

**Use the C++ quantizer for all quantization.** It calls ggml's native
`ggml_quantize_chunk` which supports all types including K-quants.

### Embedding tables and aggressive quantization

Token embedding tables (`token_embd.weight`) are very sensitive to quantization.
Quantizing them to Q4_K degrades output quality significantly (cos drops from
0.999 to 0.71 for some models). The CrispEmbed quantizer keeps embedding tables
at F32 for Q4_K/Q5_K; only Q8_0 and F16 are allowed to touch them.

### K-quant fallback chain

K-quants (Q4_K/Q5_K/Q6_K) require row widths divisible by 256. Many embedding
model tensors have rows of 384 or 768 which aren't 256-aligned. The quantizer
falls back: Q4_K→Q4_0, Q5_K→Q5_0, Q6_K→Q8_0. This means small-dim models
get Q4_0 instead of Q4_K for most tensors.

### ggml_get_rows for quantized embeddings

The BERT encoder must use `ggml_get_rows` (ggml graph op) for embedding table
lookup, not manual `ggml_backend_tensor_get` with float pointer arithmetic.
`ggml_get_rows` handles dequantization internally and works with any tensor type.
Manual CPU-side extraction assumes F32 layout and crashes on quantized models.

## Server performance: buffer reuse

The biggest server-mode optimization is reusing `graph_buf` and `work_buf` across
encode calls. Without this, every request allocates ~50-200MB (graph context +
compute workspace), causing 3x overhead from malloc/free.

With buffer reuse: gte-small goes from 8.8 to 27.8 texts/sec (3.2x improvement).

## BLAS/MKL for embedding models

BLAS (OpenBLAS/MKL) provides minimal benefit for embedding inference because:
- Quantized kernels (Q8_0/Q4_K) use ggml's SIMD paths, not BLAS
- BERT encoder matrices are moderate-sized (384x384 to 1024x4096)
- BLAS overhead dominates for small matrices

For CPU speed: use Q8_0 quantization. For GPU: build with `-DGGML_CUDA=ON` or
`-DGGML_VULKAN=ON` — the `ggml_backend_sched` dispatcher handles offloading.

## ggml_backend_sched with CPU-only

When using `ggml_backend_sched` in CPU-only mode, calling it repeatedly with
different graphs causes segfaults because the scheduler holds stale tensor
references from freed graph contexts. Solution: only create the scheduler when
a GPU backend is detected (`!ggml_backend_is_cpu(backend)`). For CPU-only,
direct `ggml_graph_compute` with a persistent work buffer is faster anyway.

## Matmul optimization — what we use, what's available

### Current state (as of April 2026)

Our embedding models have small matrices: 384×384 (MiniLM/GTE) to 1024×4096
(Qwen3 FFN). For these sizes, overhead per matmul call matters more than
raw FLOP throughput.

### CPU matmul options (ggml-cpu)

| Option | Default | Effect | Our impact |
|--------|---------|--------|-----------|
| `GGML_LLAMAFILE` | OFF | Custom SGEMM kernels optimized for small F32 matmul | **HIGH** for F32 models |
| `GGML_AVX512` | OFF | 512-bit SIMD (2x wider than AVX2) | **HIGH** if CPU supports |
| `GGML_AVX512_VNNI` | OFF | Hardware int8 dot products | Medium for Q8_0 models |
| `GGML_AMX_TILE` | OFF | Intel AMX for int8/BF16 (Sapphire Rapids+) | None (needs new CPU) |
| `GGML_OPENMP` | ON | Thread parallelism | Already enabled |

**Enable for best CPU performance:**
```bash
cmake -S . -B build -DGGML_LLAMAFILE=ON   # custom SGEMM
cmake -S . -B build -DGGML_AVX512=ON      # if CPU supports (check /proc/cpuinfo)
```

### CUDA matmul options

| Option | Default | Effect |
|--------|---------|--------|
| `GGML_CUDA_FA` | ON | Flash attention CUDA kernel |
| `GGML_CUDA_GRAPHS` | OFF | Multi-op fusion via CUDA graph capture |
| `GGML_CUDA_FORCE_MMQ` | OFF | Force quantized matmul kernels (vs cuBLAS) |
| `GGML_CUDA_FA_ALL_QUANTS` | OFF | Flash attn for all quant types |

CUDA auto-selects between MMQ (quantized matmul) and cuBLAS (F32) based
on matrix size and GPU compute capability. For our 384×384 Q8_0 matrices,
MMQ is usually selected (faster than cuBLAS for small quantized matmul).

### Why HF PyTorch is still competitive on CUDA

HF PyTorch uses cuBLAS with operator fusion via torch.compile/TorchScript.
For a 22M-param model (MiniLM), the GPU is underutilized — compute time
is dominated by kernel launch overhead and memory transfers, not FLOP
throughput. Both HF and CrispEmbed run at ~10ms, limited by the GPU's
minimum latency per kernel launch (~5μs × ~200 kernels = ~1ms overhead).

### Batched matmul on GPU

Single matmul `W[H,H] × X[H, T*B]` is much faster than B separate
`W[H,H] × X[H, T]` calls because:
1. One cuBLAS/MMQ launch vs B launches
2. Better GPU occupancy (more work per SM)
3. Memory access amortization

Our true batched graph concatenates all texts and uses 4D flash attention
with batch dimension. The matmuls naturally batch via the flattened T*B dim.

### QKV weight fusion

Pre-merging Q/K/V weight matrices into `[H, 3H]` reduces 3 matmul calls
to 1 per layer. The merged tensor must live in the same backend buffer as
the model weights (ggml_backend_alloc_ctx_tensors) so it works on GPU.

On CPU: ~0.5ms savings (15.3ms vs 16.8ms for MiniLM).
On GPU: minor savings (kernel launch overhead reduction).

## Optimization experiment results (April 2026)

| Optimization | CPU Impact | GPU Impact | Verdict |
|---|---|---|---|
| QKV weight fusion (1 matmul vs 3) | 15.3ms vs 17.0ms (**+11%**) | minor | **Keep** — matmul reduction wins |
| Flash attention (fused QKV attn) | 16.8→15.3ms | significant | **Keep** |
| Scheduler reservation (bucket T) | no change | may help | Keep (no cost) |
| GGML_LLAMAFILE | 15.3→14.7ms (**+4%**) | N/A | **Enable by default** |
| AVX512 (if CPU supports) | 15.3→14.4ms (**+6%**) | N/A | Enable if available |
| F16 model weights | 15.3→17.7ms (**-14%**) | may help (tensor cores) | **Skip on CPU** |
| Removing ggml_cont (no QKV fusion) | 15.3→17.0ms (**-10%**) | N/A | Don't remove |
| True batched graph (4D flash attn) | slower on CPU | should help | GPU only |

### Why we can't easily match HF PyTorch

1. **Graph rebuild cost**: ggml rebuilds the graph from scratch every call (~1ms).
   PyTorch JIT-compiles and caches the execution plan.
2. **No CPU operator fusion**: ggml CPU executes each op separately (separate memory pass
   for norm, mul, add). ORT/PyTorch fuse these into single kernels.
3. **No persistent CUDA graphs**: PyTorch can capture and replay GPU command streams.
   ggml has `GGML_CUDA_GRAPHS` but it's designed for llama.cpp's specific graph topology.
4. **Batch matmul**: PyTorch's cuBLAS wrapper handles batched matmul natively.
   Our 4D reshape + flash attention adds overhead vs native batch support.

### Practical CPU performance ceiling

For MiniLM (22M params, 6 layers, 384d) on 4-thread CPU:
- **15.3ms** with all optimizations (QKV fusion + flash attn + llamafile)
- **~14ms** theoretical minimum (pure matmul compute time)
- **~1ms** graph rebuild overhead we can't eliminate
- HF PyTorch on same CPU: **54ms** (CrispEmbed is **3.5x faster on CPU**)

### Practical GPU performance ceiling

For MiniLM on RTX A1000 (budget laptop GPU):
- **10.6ms** current (with all optimizations)
- **~5ms** theoretical minimum (kernel launch overhead + small matrix underutilization)
- HF PyTorch: **9.5ms** (they have better GPU batching)
- Gap is ~1ms — likely kernel launch overhead from ggml's per-op dispatch

## Windows build

Windows users often forget `--recursive` when cloning. The CMakeLists.txt now
checks for `ggml/CMakeLists.txt` existence and prints a helpful error message.
Build scripts (`build-windows.bat`, `build-vulkan.bat`, `build-cuda.bat`) auto-
detect VS2022 and Vulkan/CUDA SDKs.

## ggml operator fusion — what exists, what doesn't

### Existing fused ops (backend-specific)

**CUDA** (automatic when graph patterns match):
- RMSNorm + Mul (`ggml_cuda_op_rms_norm_fused`)
- RMSNorm + Mul + Add (`ggml_cuda_op_rms_norm_fused_add`)
- Multi-Add (up to 8 chained adds → 1 kernel)
- FFN gate: MUL_MAT + ADD + MUL_MAT + ADD + GLU → 1 kernel
- RoPE + SetRows fused
- Unary + Mul (SILU/Sigmoid/Softplus)

**Vulkan**: Add + RMSNorm (controlled by `GGML_VK_DISABLE_FUSION`)
**Metal**: Generic fusion framework with `use_fusion` flag
**CPU**: **No fusion at all** — every op executes individually

### What this means for performance

On **CPU**, there's a fundamental ~3x gap vs ONNX Runtime because:
1. ORT does Level3 graph JIT compilation: constant folding, op fusion, layout
   optimization, kernel selection — all at graph compile time
2. ggml has no graph optimization pass; fusion only happens in GPU backends
   during compute, not at graph construction time
3. Each ggml CPU op does a separate memory pass (read+write). Fusing
   LayerNorm (norm+mul+add = 3 passes) into 1 pass saves bandwidth

On **GPU (CUDA)**, the gap should be much smaller because:
1. CUDA backend automatically fuses RMSNorm+Mul, FFN gates, multi-add
2. `ggml_flash_attn_ext` runs as a single fused CUDA kernel
3. Matmul uses cuBLAS (same as PyTorch/ONNX)
4. Memory bandwidth is 10-20x higher on GPU, so fusion matters less

### What we optimized (practical CPU-side)

1. **Pre-merged QKV weights**: concatenate Q/K/V weight matrices into one
   [H, 3H] tensor at load time. One matmul instead of three per layer.
   Saves ~0.5ms for 6-layer 384d model.

2. **Flash attention**: `ggml_flash_attn_ext` replaces 8 separate ops
   (permute, cont, mul_mat, scale, softmax, mul_mat, permute, reshape)

3. **Graph caching**: build ggml graph once per sequence length, reuse
   across calls. Eliminates ~3ms of ggml_init + graph construction.

4. **Buffer reuse**: graph_buf and work_buf persist across calls.

### Why not modify ggml for CPU fusion?

Considered but impractical because:
- ggml's CPU backend is designed for portability (pure C + SIMD intrinsics)
- Adding a graph optimization pass would affect all ggml users
- The `ggml_map_custom` API allows custom kernels but doesn't help with
  matmul (the expensive op) — ggml's SIMD matmul is already well-optimized
- Fusing norm+mul+add saves < 0.1ms per text (memory-bound, not compute-bound)
- The 3x gap to ONNX is dominated by ORT's matmul scheduling and cache
  optimization, not by op fusion per se

### GPU prediction

On CUDA, CrispEmbed should match or beat ONNX because:
- cuBLAS matmul is the same engine ORT uses
- ggml's CUDA fusion handles the same patterns ORT fuses
- Flash attention is implemented as a single CUDA kernel
- No Python/ONNX overhead in our C++ server

Estimated GPU performance for MiniLM (RTX 3060):
- CrispEmbed CUDA: ~2-4ms (model fits entirely in GPU memory)
- fastembed ONNX+CUDA: ~2-4ms (cuBLAS + graph optimization)
- Likely on par, with CrispEmbed winning on server overhead

## Prompt prefix system for RAG models

Many embedding models require query/passage prefixes for optimal retrieval:
- BGE: `"Represent this sentence for searching relevant passages: "`
- E5: `"query: "` / `"passage: "`
- Nomic: `"search_query: "` / `"search_document: "`
- Jina v5: `"Query: "` / `"Document: "`

Implementation: prefix is stored in `crispembed_context::prefix` and prepended
to the raw text before tokenization in both `crispembed_encode()` and
`crispembed_encode_batch()`. This is correct because:
1. The prefix is part of the semantic input (not a tokenizer-level construct)
2. All tokenizer types (WordPiece/SentencePiece/BPE) handle it naturally
3. fastembed-rs does the same (injects prefix before tokenizer.encode)

**Not applied to sparse/colbert/reranker**: These have different input semantics.
Sparse retrieval operates on raw terms. Rerankers take (query, document) pairs
where the model handles the joint encoding.

## Bi-encoder vs cross-encoder reranking

Both approaches are valuable for RAG and complement each other:

**Bi-encoder** (embed query + docs independently, cosine similarity):
- Fast: encode once, compare N documents with dot products
- Same model used for initial retrieval AND reranking
- Quality limited by the embedding space
- CrispEmbed: `rerank_biencoder()` in Python/Rust, uses `encode_batch()` + dot product

**Cross-encoder** (encode query-document pairs jointly):
- Slow: each (query, doc) pair requires a full forward pass
- Much higher quality (joint attention between query and document tokens)
- Typically used as second-stage reranker after bi-encoder retrieval
- CrispEmbed: `rerank()` in Python/Rust, uses `crispembed_rerank()` C API

**RAG pipeline pattern**: bi-encoder retrieval (top-100) → cross-encoder reranking (top-10)

## Model registry for RAG feature parity

When adding new models to the registry (`model_mgr.cpp`), the key metadata is:
- **name**: short name for CLI/auto-download
- **filename**: GGUF filename (may include `-q8_0` suffix for default quant)
- **url**: HuggingFace direct download URL under `cstr/` namespace
- **desc**: architecture, dimension, language, parameter count

Models that are encoder-only (BERT/XLM-R) use the existing convert-bert-to-gguf.py.
Models that are decoder-based (Qwen3/Gemma3) use convert-decoder-embed-to-gguf.py.
Rerankers are encoder models with a classifier head — use `--crisp` flag to include
the classifier weights in the GGUF.

## MPNet relative position bias

MPNet uses T5-style relative position bias instead of absolute position embeddings.
The bias is a learned `Embedding(32, 12)` — 32 logarithmic distance buckets × 12
attention heads. For each (query_pos, key_pos) pair, a bucket index is computed
via logarithmic distance binning, then the bias is looked up and added to
attention scores before softmax.

**Our implementation** (CrispEmbed):
- Precompute the full `[T, T, n_heads]` bias matrix in C++ at encode time
- Pass it as the F16 mask parameter to `ggml_flash_attn_ext`
- Flash attention adds it to scores natively — no manual attention needed
- Result: cos=0.999997 vs HuggingFace

**llama.cpp approach** (PR #21880):
- Compute bucket indices in the ggml graph via `build_inp_pos_bucket_enc()`
- Look up bias weights with `build_pos_bias()` (ggml graph ops)
- Pass as `kq_b` to `build_attn()` which adds it to attention scores
- Tensor stored transposed `[n_heads, n_buckets]` on layer 0

**Key difference**: We precompute in C++ (simpler, works on CPU), they compute in
the ggml graph (GPU-accelerable, more modular). Both produce identical results.
Our approach is ~10 lines of C++ vs their ~50 lines of graph builder code.

**Bugs found during MPNet implementation**:
- Python `or` operator treats `cls_token_id=0` as falsy → falls through to
  default 101. Fix: use `is not None` check
- MPNet needs position offset = 2 (same as RoBERTa), but `model_type="mpnet"`
  was not included in the offset detection

## Reranker model conversion notes

Cross-encoder rerankers (bge-reranker, ms-marco-MiniLM, mxbai-rerank) have a
classifier head on top of the encoder:
- **1-layer**: `classifier.dense.weight [H,1]` + `classifier.dense.bias [1]`
  → CLS hidden → Linear → scalar score
- **2-layer** (RobertaClassificationHead): `classifier.dense.weight [H,H]` +
  `classifier.out_proj.weight [1,H]` + biases
  → CLS hidden → Linear → tanh → Linear → scalar score

The converter must include these weights. Detection: `crispembed.is_reranker`
is set based on presence of `classifier.dense.weight` in the GGUF.

Some rerankers (ms-marco-MiniLM) use `num_labels=1` with no activation,
while others (bge-reranker) use sigmoid/softmax. CrispEmbed returns the raw
logit — the caller decides on thresholding.

## ModernBERT architecture (pre-LN)

ModernBERT (gte-modernbert-base, modernbert-embed-large) uses **pre-LayerNorm**
ordering, which differs from standard BERT's post-LN:

**Post-LN (BERT/XLM-R/MPNet):**
```
attn(input) → residual_add(input) → LN → FFN → residual_add → LN
```

**Pre-LN (ModernBERT):**
```
LN(input) → attn → residual_add(input) → LN → FFN → residual_add
```

Pre-LN has the LayerNorm *before* each sub-layer, with the residual connection
bypassing the norm. This is the same as GPT-2/LLaMA-style normalization.

Detection: `bert.pre_ln` GGUF metadata flag. Combined with:
- GeGLU activation (GELU-gated FFN instead of SwiGLU)
- RoPE (no position embeddings)
- No biases on attention or FFN
- Fused QKV weights

ModernBERT is essentially a bidirectional LLaMA with GELU instead of SiLU.
CrispASR has a reference implementation in `examples/talk-llama/models/modern-bert.cpp`.

### ModernBERT debugging: cos 0.69 → 0.97

Two bugs caused cos=0.69 across 22 layers (1-layer was 0.999):

**Bug 1: Wrong SEP token.** The BPE merge re-loading after tensor init
was calling `load(vocab, merges, eos_id=sep_id, pad_id, suffix_id=unk_id=3, ...)`
instead of `suffix_id=-1`. This made the tokenizer append token 3 (unk)
instead of 50282 (SEP). The wrong token propagated through all 22 layers
of the transformer, compounding the error.

Lesson: when re-initializing a tokenizer after loading merges, preserve
ALL original parameters — don't substitute defaults for parameters that
were carefully set during the first init.

**Bug 2: Separate GELU+MUL vs fused ggml_geglu.** Our code used:
```cpp
up = matmul(fc1_w, cur);     // [inter, T]
gate = matmul(ffn_gate_w, cur); // [inter, T]
up = gelu(up);
ffn = mul(up, gate);
```

llama.cpp uses:
```cpp
up_gate = matmul(ffn_up_gate_w, cur); // [2*inter, T]
ffn = ggml_geglu(up_gate);           // fused: gelu(first_half) * second_half
```

The fused `ggml_geglu` is a single ggml operation that avoids intermediate
rounding between the GELU and multiply. With 22 layers × ~1000 intermediate
dimensions, the accumulated rounding difference is significant for pre-LN
models (where residual connections pass raw values without normalization reset).

Fix: store the original fused `Wi` / `up_gate_proj` weight in the GGUF
and use `ggml_geglu` instead of separate ops. Also use `ggml_swiglu` for
NomicBERT-style SwiGLU.

**Why post-LN models don't have this problem:** In post-LN models (BERT),
LayerNorm after each residual add normalizes the hidden state to unit
variance. This effectively "resets" any accumulated floating-point drift.
In pre-LN models, the raw residual passes directly to the next layer,
allowing small per-layer errors (~0.001) to compound nonlinearly.

**Per-layer theta:** ModernBERT alternates sliding (theta=10000) and global
(theta=160000) attention. For encoding (not generation), sliding window
masking is NOT applied — confirmed by llama.cpp's `build_attn_inp_no_cache()`.

## Head-to-head benchmark: CrispEmbed vs FastEmbed

**MiniLM-L6 (6 layers, 384d)**: CrispEmbed is **9.5x faster** on single text
and **10.8x faster** on batch. This is our best-optimized model: QKV fusion
reduces 3 matmuls to 1 per layer, flash attention replaces 8 separate ops,
and graph caching eliminates rebuild overhead.

**BGE-small (12 layers, 384d)**: FastEmbed is **1.7x faster**. ONNX Runtime's
Level3 graph JIT compilation (operator fusion, layout optimization, cache-aware
scheduling) gives it an edge on 12-layer models. Our per-op execution on CPU
has higher overhead per layer.

**Arctic-M (12 layers, 768d)**: Tied on batch (126 vs 127ms). As hidden size
grows, matmul compute dominates over per-op overhead, equalizing performance.

**Conclusion**: CrispEmbed wins decisively on small models (6 layers) where
per-op overhead matters most. On larger models, ONNX Runtime's graph optimization
closes the gap. GPU (CUDA/Metal) should favor CrispEmbed across all sizes due
to ggml's fused CUDA kernels and flash attention.

## DeBERTa-v2 disentangled attention (full parity)

DeBERTa-v2's attention computes three components, all now implemented:
1. **c2c** (content-to-content): standard Q×K^T
2. **c2p** (content-to-position): Q × K_proj(rel_embd)^T
3. **p2c** (position-to-content): K × Q_proj(rel_embd)^T

### Key implementation details

**Pre-expansion approach**: Rather than gather+matmul at runtime, we pre-expand
the position embeddings on CPU: `P[H, T*T]` where `P[:, i*T+j] = LN(rel_emb[bucket(i-j)+256])`.
Then project through K/Q weights and use batched matmul to compute all scores.

**Critical: HF uses bucket(query-key) for BOTH c2p AND p2c**. This is
counter-intuitive — you'd expect p2c to use bucket(key-query). But HF's
`disentangled_attention_bias` gathers p2c using the same relative position
index, then transposes the result. To achieve this with pre-expansion, we
transpose the T×T grid for p2c: `P_p2c = P.reshape(H,T,T).permute(0,2,1)`.

**Encoder LayerNorm on position embeddings**: HF applies `encoder.LayerNorm`
to `rel_embeddings.weight` BEFORE using them in attention (`get_rel_embedding()`).
This is separate from the post-encoder LayerNorm. Missing this causes ~15%
error in position scores.

**Position projection biases**: HF's `key_proj`/`query_proj` are `nn.Linear`
which include bias. Must add `k_bias` to Pk and `q_bias` to Pq.

**Log-bucket formula** (`make_log_bucket_position`): Uses signed bucket values
centered at `att_span` (= position_buckets = 256). The log denominator is
`log((max_relative_positions - 1) / mid)`, NOT `log((max_pos/2 - 1) / mid)`.

**Attention output reshape**: After V-weighted sum `[hd, T_q, nh]`, must permute
to `[hd, nh, T_q]` BEFORE reshaping to `[H, T]`. Without this permute, head
dimensions get incorrectly interleaved.

**Score scaling**: `1/sqrt(3 * head_dim)` when both c2p and p2c are present
(the 3 = 1 + num_position_attention_types).

### ggml_permute semantics (output-position convention)

`ggml_permute(a, ax0, ax1, ax2, ax3)`: `axes[k]` means "source dimension k
goes to result dimension `axes[k]`". So `permute(a, 0, 2, 1, 3)` on
`[hd, nh, T, B]` gives `[hd, T, nh, B]` (dims 1 and 2 swap).

This is the OPPOSITE of numpy's `transpose` where you specify source→result.

## Rust crate verification

The CrispEmbed Rust crate (`crispembed/`) wraps the C API via `crispembed-sys`
(cmake build.rs). Verified features:
- Dense encode (384d, correct values match Python)
- Batch encode (3 vectors, correct)
- Prefix set/get
- Matryoshka truncation (128d from 384d)
- Bi-encoder reranking (correct ordering)
- Capability queries (has_sparse, has_colbert, is_reranker)

The crate links dynamically (`dylib=crispembed`). Set `LD_LIBRARY_PATH` to the
build output directory. Static linking would avoid this but requires listing
all ggml dependencies in build.rs.

## BidirLM-Omni: 3D interleaved MRoPE via ggml IMROPE

HF `BidirLMOmniTextRotaryEmbedding.apply_interleaved_mrope` builds a per-token
`freqs_t` of length `head_dim/2` from three position channels `(t, h, w)` and
the configured `mrope_section = [s_t, s_h, s_w]` (default `[24, 20, 20]`):

- Start with `freqs_t = freqs[t]` (channel 0 across the entire vector).
- Replace indices `slice(1, 3*s_h, 3)` with `freqs[h]` at those indices.
- Replace indices `slice(2, 3*s_w, 3)` with `freqs[w]` at those indices.
- Anything beyond `3*s_h` (resp. `3*s_w`) stays in the t-channel.

For `[24, 20, 20]` and `head_dim=128` (so 64 cos/sin pairs), this produces:
T at positions 0, 3, …, 60, 63; H at 1, 4, …, 58; W at 2, 5, …, 59; T at 61–63
beyond the H/W slice ends.

ggml's `GGML_ROPE_TYPE_IMROPE` takes 4-channel positions `(t, h, w, e)` and
sections `[s_t, s_h, s_w, s_e]`. Its sector check is:

- `sector%3==0 && sector < 3*s_t` → `theta_t`
- `sector%3==1 && sector < 3*s_h` → `theta_h`
- `sector%3==2 && sector < 3*s_w` → `theta_w`
- otherwise → `theta_e`

For sections `[24, 20, 20, 0]` ggml routes sectors 61 and 62 to `theta_e`,
whereas HF leaves them on the T channel. The fix is to **pin `pos_e = pos_t`
per-token**: with that, `theta_e == theta_t` numerically at every sector and
the ggml IMROPE output matches HF byte-for-byte. The position tensor passed
to `ggml_rope_multi` therefore has shape `(4*T,)` laid out as
`[pos_t, pos_h, pos_w, pos_t]` (the tail mirrors the head).

For text-only inputs the three channels are all equal, so MRoPE collapses to
plain NEOX RoPE — `decoder_embed.cpp` keeps using `ggml_rope_ext` on the
text-only path to stay bit-identical with the pre-Phase-3 baseline tests.

## BidirLM-Omni: decoder scheduler init was missing

Before Phase 3 the decoder branch in `crispembed_init` never created a
`ggml_backend_sched` or sized `compute_meta` — those were only set up by
`load_model()` on the encoder branch. `decoder_encode_tokens` checks
`(sched != nullptr && compute_meta != nullptr)` and falls back to direct
`ggml_graph_compute` when either is null, so BidirLM-Omni text and audio
were silently running CPU-only on Metal builds.

Fix: in the decoder branch, after `load_decoder_model`, allocate

```cpp
const int graph_nodes = std::max(4096, ctx->dec->n_layer * 50 + 256);
ctx->sched = ggml_backend_sched_new(...);
ctx->compute_meta.resize(ggml_tensor_overhead() * graph_nodes
                       + ggml_graph_overhead_custom(graph_nodes, false));
```

The `4096` floor is important: with image-conditioned text the graph adds an
input mask + patch (2 ops), per-layer DeepStack adds (n_ds ops), and
`ggml_rope_multi` instead of `ggml_rope_ext` (no node-count delta but extra
per-tensor metadata). 28 layers × ~50 ops ≈ 1400 still fits, but the floor
keeps headroom for future architectural growth and avoids surprising
allocation failures. Verify with `--save-baseline` / `--compare-baseline` in
`tests/benchmark_bidirlm.py` — text-only cosine should remain ≥ 0.99999
against the baseline taken before this change.

## BidirLM-Omni: parity reference dtype matters

When validating a quantized GGUF against a HuggingFace reference, the
**reference dtype** is part of the comparison and silently shifts the
upper bound. BidirLM-Omni-2.5B-Embedding ships its `model.safetensors`
in bf16 — that's the dtype the model was trained in. Loading it into
torch and calling `.to(torch.float32)` doesn't reconstruct any
pre-bf16 information; it just zero-pads the mantissa. So a cosine of
~0.94 vs HF fp32 is two distinct quantization steps stacked (bf16
trained → q4_k storage, then bf16 → fp32 upcast for the reference),
not "the q4_k is broken."

The fix in `tests/test_bidirlm_image_text.py`: the reference dtype is
a `--ref-dtype` flag, defaulting to bf16. Match the trained dtype.

## BidirLM-Omni: q4_k quantization cosine ceiling

Empirically, **q4_k vs HF bf16 settles at ~0.94 cosine** for the 2.5B
embedding variant, on both text-only (`tests/test_bidirlm_text.py`)
and image+text (`tests/test_bidirlm_image_text.py` /
`tests/test_bidirlm_image_text_lite.py`). That's the q4_k *intrinsic*
cosine — not a Phase 3 multimodal-injection bug.

The README's "cosine ≥ 0.99999" gate is for **graph regressions**
(CrispEmbed-q4_k vs a saved CrispEmbed-q4_k baseline from before a
code change); it doesn't measure CrispEmbed-vs-HF. To get ≥ 0.99
cosine vs HF bf16 you need q8_0 or higher precision.

Concretely measured (April 2026, q4_k against HF bf16 on /tmp/cat.jpg):

| path | cosine |
|---|---|
| text-only (`encode("Hello world")`) | 0.93–0.95 |
| multimodal (`encode_with_image_ids`) | 0.94 |

When debugging Phase 3 parity, run *both* test paths against the same
quant — if the multimodal cosine matches the text-only cosine for the
same prompts (modulo image content), the multimodal graph is fine and
the gap is the quant's intrinsic precision floor. If multimodal is
lower than text-only, that's a Phase 3 bug.

## BidirLM-Omni: image preprocessor parity is governed by mean/std, not the JPEG decoder

When porting HF Qwen2VLImageProcessorFast to C++ for `image_preprocess.cpp`,
the initial cosine vs HF was 0.97 — well below the ≥0.99 target. The
intuition was "stb_image's JPEG decoder differs from PIL/libjpeg-turbo by
a few LSBs, that propagates through the bicubic resize." That was wrong:

- Adding a PIL-decoded-RGB pass-through (`crispembed_preprocess_image_rgb`,
  skipping stb's JPEG decode entirely) moved cosine from 0.987 to 0.987.
- Switching `bicubic_resize_u8_to_f32` to round-clamp to integer (mimicking
  torchvision's uint8 round-trip on `tvF.resize(uint8, antialias=True)`)
  also moved cosine from 0.987 to 0.987.

The actual cause was the `image_preproc::config` defaults using OpenAI CLIP
mean/std `[0.481, 0.458, 0.408]` / `[0.269, 0.261, 0.276]`, while
**BidirLM-Omni's `preprocessor_config.json` specifies `mean = std = [0.5,
0.5, 0.5]`** (the SimVL / Qwen2-VL convention that maps `[0,1]` → `[-1,1]`).
Every normalized pixel value was off by a roughly-constant linear transform,
which has *high* flat cosine (0.987) but huge max-abs-diff (1.19 in
normalized space). The numbers had a strong mean-shift, which cosine
similarity is largely insensitive to until rescaled by std.

After fixing the defaults: pixel_values cosine 0.987 → 0.999989,
encode_image embedding cosine 0.970 → 0.999984. Sub-1e-5 residual is
sub-pixel torchvision-uint8 bicubic kernel weight quantization (PyTorch
uses int16 weights for the uint8 AA path; we use float weights).

`min_pixels` and `max_pixels` were also wrong for BidirLM-Omni (the
defaults from Qwen2-VL: 56² and 14²·4·1280; BidirLM uses 256² and 1024²
per the preprocessor config). For our test image these happened to land
on the same `smart_resize` output, but a different aspect ratio could
have produced a different grid_thw.

Lesson: when matching a model's preprocessor, read the actual
`preprocessor_config.json` from the HF repo. Don't assume CLIP defaults.
The converter (`models/convert-decoder-embed-to-gguf.py`) now writes
`bidirlm.vision.image_mean / image_std / min_pixels / max_pixels` into
the GGUF so future model variants can be picked up without guessing.

## BidirLM-Omni: image-embed splice via mask + add

HF does `inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)`
to replace token-embed rows at every `image_token_id` placeholder with vision
tower output. ggml has no native `masked_scatter`, so `decoder_embed.cpp`
emulates it with two host-prepared inputs:

- `in_keep_mask` shape `(1, T)` — 1.0 at text positions, 0.0 at image positions.
- `in_patch` shape `(H, T)` — `image_embeds[k]` row at the k-th image position,
  zeros at text positions.

```
cur = ggml_get_rows(token_embd, ids_t)
cur = ggml_mul(cur, in_keep_mask)   // zero out image-position rows
cur = ggml_add(cur, in_patch)       // splice image_embeds in at those rows
```

The `(1, T) * (H, T)` mul broadcasts the leading dim over H — same trick the
vision tower uses for the 4-corner pos-embed gather. DeepStack adds use the
same `(H, T)` patch shape, one per layer for the first `n_deepstack` layers,
zero everywhere except at image positions; `cur = ggml_add(cur, ds_patches[il])`
after each layer's residual+ffn output mirrors HF's `_deepstack_process`.

## Distribution: install(EXPORT) + ggml SHARED don't compose

When `crispembed-shared` is a SHARED library that PRIVATEly links a SHARED
ggml backend (`ggml-cpu`, `ggml-base`, …), `install(TARGETS crispembed-shared
EXPORT crispembed-targets)` errors with:

> install(EXPORT "crispembed-targets" …) includes target "crispembed-shared"
> which requires target "ggml-cpu" that is not in any export set.

The reason: even for PRIVATE link deps of a SHARED lib, CMake records them
as `IMPORTED_LINK_DEPENDENT_LIBRARIES` so downstream consumers know what
runtime SO names the .so will dlopen. install(EXPORT) demands those deps
either be in some export set or be system-IMPORTED.

Two viable workarounds:

1. **Hand-rolled IMPORTED target** (what CrispEmbed does, mirroring
   CrispASR): skip `install(EXPORT)` entirely. The `crispembed-config.cmake.in`
   uses `find_library(crispembed_LIBRARY crispembed HINTS …)` plus
   `add_library(crispembed::crispembed UNKNOWN IMPORTED)` to manufacture
   the IMPORTED target at config time. Runtime resolution of `libggml*.so`
   siblings is handled entirely by the .so's RPATH (`$ORIGIN` /
   `@loader_path`), not by the consumer's link line.
2. **Add ggml to the same EXPORT** via `set_target_properties(ggml*
   PROPERTIES EXPORT_NAME …)` and put ggml's install in your export.
   More invasive and requires patching the ggml submodule.

(1) is the right choice when the .so is the only thing the user sees and
ggml is implementation detail; (2) is right when you want consumers to be
able to `find_package(ggml)` separately.

## Distribution: relocatable pkg-config via ${pcfiledir}

`@CMAKE_INSTALL_PREFIX@` in a `.pc.in` is bound at configure time, not
install time. A user who runs `cmake --install build --prefix /opt/foo`
gets a `.pc` file with `prefix=/usr/local` (the configure default), and
`pkg-config --libs crispembed` returns wrong paths.

The fix is the **relocatable** pattern — set `prefix` from the .pc file's
own location:

```pc
prefix=${pcfiledir}/../..
libdir=${prefix}/lib
```

Since the .pc lives at `<prefix>/lib/pkgconfig/crispembed.pc`, going
`../..` from there is the prefix dir, no matter where the user dropped it.
Verified across `cmake --install --prefix /tmp/...`, tarball extraction
into `/opt/foo`, and the standard `/usr/local`.

## Distribution: forward-declared structs need typedefs for C consumers

`crispembed.h` had `struct crispembed_context;` plus function signatures
like `crispembed_context * ctx`. In C++ the struct name lives in the type
namespace so this compiles; in **C** the caller has to write
`struct crispembed_context *` everywhere. Adding

```c
typedef struct crispembed_context crispembed_context;
typedef struct crispembed_hparams { … } crispembed_hparams;
```

(forward-decl style for opaque types, full definition for value types) was
caught by the install verification test — a plain-C consumer of the
freshly `cmake --install`-ed header. The build directory consumers
(crispembed-cli, crispembed-server) didn't catch it because they're
compiled as C++.

## CNN forward path for face models (Phase 8)

### Available ggml ops for CNN
- `ggml_conv_2d(a, b, s0, s1, p0, p1, d0, d1)` — standard 2D conv
- `ggml_conv_2d_dw(a, b, ...)` — depthwise 2D conv
- `ggml_pool_2d(a, op, k0, k1, s0, s1, p0, p1)` — average/max pool
- `ggml_relu`, `ggml_leaky_relu(a, slope, inplace)` — activations
- No `ggml_prelu` — implement as: `relu(x) + slope * (x - relu(x))`
  where slope is a learned [C, 1, 1] tensor per channel

### BatchNorm folding
At inference time, BN is folded into the preceding Conv:
```
w_new = w * gamma / sqrt(var + eps)
b_new = (b - mean) * gamma / sqrt(var + eps) + beta
```
This eliminates all BN tensors from the forward pass.

### Conv2d output layout in ggml
`ggml_conv_2d` output: `[OW, OH, OC]` — width-first (ne[0]=OW).
To match HF's `[OC, OH, OW]` (channel-first): `permute(2, 1, 0)`.
This matters for position embeddings in ViT but NOT for CNNs
(CNNs are translation-equivariant — spatial order preserved naturally).

### SFace architecture (MobileFaceNet)
27 Conv layers (14 depthwise separable blocks), PReLU activation,
final GDC pool → FC(50176→128). 128-D L2-normalized embedding.
Input: 112×112 aligned face crop.

### SCRFD architecture (ResNet-50 + FPN)
58 Conv layers, ReLU activation, FPN with 3 scales (stride 8/16/32).
9 output heads: 3 × (confidence [N,1], bbox [N,4], landmarks [N,10]).
Dynamic input size (typically 640×640).
Needs NMS post-processing.

### AuraFace architecture (ResNet-100)
103 Conv layers, PReLU, 49 residual Add connections.
512-D ArcFace-compatible embedding. Apache 2.0.

### CrispASR CNN reference
CrispASR has CNN forward paths for marblenet (depthwise 1D conv),
wav2vec2 (grouped conv), and others. Same ggml ops, similar patterns.
Patches at tools/upstream-prs/ may be needed for CUDA conv2d.

### YuNet ggml Transpose behavior (2D vs 3D tensors)
The `replay_graph()` Transpose op does a real 2D transpose for tensors
where `ggml_n_dims == 2` (i.e., last dimension is 1). YuNet's cls/obj
outputs have 1 channel and thus get physically transposed, while bbox/kps
with 4/10 channels remain in the original ggml layout. This requires
different spatial indexing for each:
- cls/obj (1 channel, transposed): `data[row + col * grid_h]`
- bbox/kps (multi-channel, passthrough): `data[col + row * grid_w + chan * plane]`

## ViT / CLIP parity: patch ordering bug (FIXED — cos 0.8 → 1.0)

**Previously**: CLIP and SigLIP vision achieved cos ≈ 0.8 vs HuggingFace.
This was incorrectly attributed to FP32 matmul accumulation order differences.

**Actual root cause (fixed 2026-06-06)**: The `ggml_permute(2,1,0)` used to
reshape `[OW, OH, D]` → `[D, OH, OW]` produced column-major spatial ordering
when flattened to `[D, T]`: `t = oh + ow*OH`. But HuggingFace's `flatten(2)`
gives row-major: `t = oh*OW + ow`. Every patch beyond (0,0) got the wrong
position embedding, causing systematic error at the very first layer that
compounded through all 12 layers.

**Fix**: `ggml_permute(1,2,0,3)` produces `[D, OW, OH]` with `ne[0]=D,
ne[1]=OW, ne[2]=OH`. When flattened to `[D, T]`, patches follow row-major
`t = ow + oh*OW = oh*OW + ow`, matching HF.

**Result**: Per-layer cos = 1.000000 across all 12 layers. Final embedding
cos = 0.9998 vs HuggingFace (SigLIP-base-384).

**Lesson**: Always verify data layout empirically, especially with
`ggml_permute` where the axis semantics ("old dim N goes to new position
axN") differ from numpy/PyTorch conventions. The first few values matched
(patch 0 is at position 0 in both orderings) which masked the bug.

### SigLIP attention pooling head: missing residual

HF's `SiglipMultiheadAttentionPoolingHead` computes:
```
residual = probe + attention(probe, x_cat, x_cat)
output = residual + MLP(LayerNorm(residual))
```

The final `residual +` was initially missing in our implementation,
producing cos=0.17 vs HF. After fix: cos=0.74 (same precision ceiling
as other ViT models).

## Handwritten Math OCR (HMER + BTTR)

### Image polarity auto-detection

Both HMER and BTTR expect white-on-black input (ink = 1.0, background = 0.0).
Real-world images are typically black-on-white. Both implementations auto-detect
by checking the mean pixel value: if mean > 0.5, the image is inverted
(`pixel = 1.0 - pixel`). This avoids requiring the user to preprocess images.

### BTTR architecture (DenseNet + Transformer decoder)

BTTR (Bidirectionally Trained Transformer, ICDAR 2021) uses:
- DenseNet encoder (growth=24, 16 layers × 3 blocks, 1-channel grayscale)
- Conv 1×1 projection to d=256
- 2D sinusoidal position encoding (added to encoder features)
- Standard nn.TransformerDecoder (3 layers, 8 heads, d=256, FFN=1024)
- Post-LayerNorm, fused QKV weights preserved from PyTorch
- 113 LaTeX tokens, 6.5M params

Key implementation details:
- BN is folded into Conv at convert time (same as face models)
- Fused QKV weights: kept as-is, split via ggml_view_2d in the decoder
- Decoder uses causal mask for autoregressive generation
- Cross-attention: Q from decoder, K/V from encoder features

### HMER architecture (DenseNet-121 + GRU attention)

HMER uses a coverage-based GRU attention decoder (not Transformer):
- DenseNet-121 encoder (growth=32, 3 blocks of [6, 12, 24] layers)
- 2-channel input: grayscale + mask (coverage mechanism)
- GRU decoder with attention (not self-attention — attends to encoder features)
- Coverage vector prevents the decoder from re-attending to the same regions
- 112 LaTeX tokens, 6.8M params

### Dequantization for CNN inference

When running quantized HMER/BTTR models (Q4_K/Q8_0), the DenseNet Conv2D
kernels need dequantization because `ggml_conv_2d` only supports F32/F16
weights. Both implementations call `ggml_backend_tensor_get` to read
quantized data into a CPU buffer, then use `ggml_quantize.h` functions
to dequantize to F32 before building the conv2d graph node.

**Important**: ggml only supports `quantized → F32` cast (in `ggml_compute_forward_dup`).
Direct `Q8_0 → F16` cast triggers a fatal error. Always dequant to F32 first,
then cast F32 → F16 as a separate step.

### Conv weight reshape for GGUF

PyTorch Conv2D stores weights as [out_ch, in_ch, kh, kw] (4D). GGUF
requires 2D tensors for quantization. The converter flattens to
[out_ch, in_ch * kh * kw] for storage. At load time, the C++ code
reshapes back to the 4D layout expected by `ggml_conv_2d`.

**Pitfalls in the 2D→4D reshape** (resolved 2026-06-06):

1. **`ggml_n_dims()` collapses trailing 1s**: A 4D weight `[3,3,1,1]`
   (OC=1, IC=1) reports `ndims=2`, same as a genuinely flattened 2D weight.
   Fix: validate `KW*KH*IC*OC == nelements` before applying reshape.

2. **Depthwise conv IC detection**: DW weights are `[OC, 1*KH*KW]` when
   flattened. Using input channels as IC gives `kernel_area = 9/16 = 0`.
   Fix: parse the group attr from the graph node `[s1p1g16]` BEFORE
   the reshape. When `group > 1`, set IC=1.

3. **OC=1 weights report ndims=1**: Flattened `[IC*KH*KW, 1]` has
   `ne[1]=1`, so `ggml_n_dims = 1`. Use `ndims <= 2` to catch these.

### YuNet raw tensor cos vs ONNX — layout difference, not a bug

Raw tensor cos between C++ replay_graph and ONNX reference is 0.35-0.85
for bbox/kps outputs. This is NOT a parity issue — the Transpose and
Reshape handlers in replay_graph don't rearrange memory for 3D+ tensors
(passthrough). The result is planar `[C, H, W]` layout in ggml vs
interleaved `[H*W, C]` in ONNX. The YuNet decode loop uses matching
indexing: `col + row*grid_w + chan*plane` for the planar layout.

Verified: decoded detection coordinates match OpenCV FaceDetectorYN to
sub-pixel accuracy (< 0.5px diff) on both single-face and multi-face
images. The cls tensors (1 channel) show cos=0.985-0.992 because layout
is irrelevant for single-channel data.

## PosFormer port — encoder/decoder debugging

### 2D sinusoidal positional encoding: sin/cos MUST share frequency

PyTorch `ImgPosEnc` computes inv_freq with `arange(0, half_d, 2)` → 64 values.
Each sin/cos pair uses the SAME frequency: `sin(x * f_i), cos(x * f_i)`.

The initial C++ used different freq indices for sin vs cos:
```cpp
enc[2*i]     += sinf(x_norm * inv_freq[2*i]);     // freq 2i
enc[2*i + 1] += cosf(x_norm * inv_freq[2*i + 1]); // freq 2i+1 ← WRONG
```
Fix: both must use `inv_freq[i]` (or `inv_freq[2*i]` from a 128-element array).
Symptom: encoder cosine dropped to 0.58.

### Operation ordering: pos_enc THEN LayerNorm (not reversed)

PyTorch encoder does: `feature_proj → rearrange → pos_enc_2d → LayerNorm`.
The C++ initially did: `feature_proj → rearrange → LayerNorm → pos_enc_2d`.
LayerNorm normalizes the combined feature+pos signal; applying it before
pos encoding means the positional encoding is un-normalized.

### No ReLU after feature projection

PyTorch's `self.feature_proj = nn.Conv2d(...)` has no activation. The C++
had a spurious `relu_ip()` that clipped half the signal.

### Missing decoder input LayerNorm (the biggest bug)

PyTorch decoder does:
```python
tgt = self.word_embed(tgt)  # nn.Sequential(Embedding, LayerNorm)
tgt = self.pos_enc(tgt)     # sinusoidal pos encoding
tgt = self.norm(tgt)        # ← SECOND LayerNorm, was missing in C++
```

This `decoder.norm` was not in the GGUF converter OR the C++ inference.
Symptom: layer 0 self-attention output had cos=0.868 at step 0 (should
be 1.0). After adding `dec.input_norm` to converter and C++ decoder:
cos=1.000000 at every step, max_diff < 0.00001.

**Lesson**: never attribute divergence to "FP accumulation." If cosine is
below 0.999 at step 0, there is a real bug. Trace layer-by-layer with
intermediate dumps (after SA, after CA, after FFN) to find it.

### ARM (Attention Refinement Module) incremental mode is correct

The incremental ARM with per-ARM-instance accumulators matches the PyTorch
batch cumsum exactly, IF the encoder and decoder embedding are correct.
The ARM was never the bug — the divergence came entirely from the four
encoder/decoder bugs listed above.

### Bi-directional beam search vs greedy

PosFormer's published 62.7% uses bi-directional beam search (L2R + R2L
decode, cross-rate, pick best). The C++ implements L2R greedy only. Direct
comparison must use the PyTorch decoder.forward() in a manual greedy loop,
NOT the model.beam_search() which includes the bi-directional scoring.

### Kaggle kernel patterns — MUST follow established conventions

1. **Always clone CrispASR and import kaggle_harness** — never reimplement
   token resolution, progress logging, or GPU detection. The harness has
   been debugged across 15+ kernels.
2. **kernel-metadata.json uses string "true"** not boolean true.
3. **P100 (sm_60) + PyTorch**: Kaggle's pre-installed PyTorch (CUDA 12.x)
   dropped sm_60 support. Fix: `pip install torch --index-url .../cu118`
   which still supports P100 GPU. Do NOT fall back to CPU.
4. **Dataset mount path**: Kaggle mounts `chr1str/crispasr-hf-token` at
   `/kaggle/input/datasets/chr1str/crispasr-hf-token/`, NOT at
   `/kaggle/input/crispasr-hf-token/`. The harness was patched to scan
   both paths.
5. **Kaggle Secrets API**: intermittently returns ConnectionError. The
   dataset file fallback is the reliable path.
6. **Validation speed**: PosFormer's `approximate_joint_search` uses
   bi-directional beam search (beam_size=10) on all 986 test images.
   This takes 30-60 min per validation step. Override with greedy
   beam_size=1 for ~10x faster validation during training.
7. **Heartbeat**: wrap `trainer.fit()` in `kh.build_heartbeat("train")`
   so Kaggle logs show the run is alive during long operations.
8. **W&B run resume**: using a fixed `id=` with `resume="allow"` lets
   multi-session training continue the same W&B run. But if you kill
   and restart, the charts mix old+new data. Change the run ID for
   a clean restart.
9. **Vocabulary ordering is critical**: PosFormer uses an alphabetical
   dictionary (!, (, ), +, ...). Building vocab from `Counter.most_common()`
   sorts by frequency ({, }, 1, 2, ...), scrambling 110/113 token indices.
   The model trains "successfully" (internal metrics look fine) but the
   checkpoint is completely unusable with the original dictionary, GGUF
   converter, or C++ inference. ALWAYS use the canonical dictionary.txt.
10. **OOV tokens**: 14 CROHME captions contain `'` (apostrophe) which is
    not in PosFormer's 110-token dictionary. Filter these before training
    or the DataLoader crashes with KeyError.
11. **Cosine warm restarts are dangerous**: CosineAnnealingWarmRestarts
    (T_0=30) reset LR from 0.008→0.08 at epoch 94, crashing val_ExpRate
    from 57% to 38%. The model briefly recovered to 60.1% then fell
    again. Plain CosineAnnealingLR (no restarts) is safer. The 60.1%
    peak was lost because the checkpoint was overwritten.
12. **Never delete HF checkpoints hastily**: HuggingFace has git history
    — deleted files can be recovered via `hf_hub_download(revision=SHA)`.
    But always back up to /mnt/storage first before deleting.
13. **Dataset license verification**: figshare uploads can have wrong
    licenses (user picks any license, no verification). CROHME+HME100K
    on figshare claims CC BY 4.0 but the original datasets are NC/
    proprietary. Always check the original source, not re-uploads.
14. **UniMER dataset (Apache 2.0)**: wanderkid/UniMER_Dataset on HF has
    978K printed math images (ArXiv+Pix2tex) under Apache 2.0. The
    CROHME and HME100K subsets are excluded from this license ("requires
    manual download for copyright"). Best commercial data source found.
15. **MathWriting augmentation works**: Adding 2000 MathWriting samples
    (filtered to v1 110-token vocab from deepcopy/MathWriting-human on HF)
    to CROHME training broke the 59.3% ceiling → 60.5% verified.
    47% of MathWriting is compatible with v1 vocab (~109K out of 230K).
16. **Beam=10 bi-directional doesn't help our model**: 60.3% beam=10 vs
    60.5% beam=1 — beam search actually hurts by 0.2%. The R2L path
    sometimes picks worse hypotheses that beat correct L2R in cross-scoring.
    This differs from SJTU's published model where beam=10 added ~6 points.
17. **ReduceLROnPlateau is the key to peaks**: The best val_ExpRate always
    came right after an LR drop (0.08→0.005 gave 57%, 0.005→0.00125 gave
    62%). Manual LR patching in checkpoint files works when callbacks fail.
18. **Use deepcopy/MathWriting-human for MathWriting data**: Pre-rasterized
    JPG images + LaTeX strings on HuggingFace. Much faster than downloading
    and parsing 230K InkML files from Google Storage.

## NomicBERT v2-moe: hidden biases and GPT2 config

NomicBERT extends `GPT2Config`, so standard attribute names are missing:
`intermediate_size` → `n_inner`, `hidden_act` → missing (default GELU).
Patch onto config before accessing.

**Critical**: NomicBERT v1.5 has NO Wqkv/out_proj biases, but v2-moe
DOES have them (`Wqkv.bias [2304]`, `out_proj.bias [768]`). The original
converter assumed "no bias" based on v1.5 — this caused cos ≈ 0.955 parity
(consistent across all texts, easily mistaken for a precision issue rather
than a missing-data bug). Always check `bias is not None` dynamically
rather than hardcoding assumptions from one model variant.

Diagnosis approach: tensor diff showed all 148 weights bit-exact (0.0),
proving the bug was runtime-only. Layer-by-layer dump (`CRISPEMBED_DUMP_LAYERS`)
showed divergence starting at the attention output (before residual/LN),
which pointed to QKV projection. Manual `x @ W.T` matched HF weights
but not `Wqkv(x)` — the missing bias term.

## MoE encoder: ggml_mul_mat_id layout

For `ggml_mul_mat_id(A, B, ids)`:
- A shape `[ne0, ne1, n_experts]`, B shape `[ne0, K, T]`, ids `[K, T]`
- Result: `[ne1, K, T]` — transposes A along ne0/ne1 (same as mul_mat)
- For expert fc2 (down projection): HF stores `w2 [n_exp*inter, hidden]`,
  used as `act_out @ w2` (NO transpose). For ggml we need ne0=inter,
  ne1=hidden → numpy `[n_exp, hidden, inter]` → converter does
  `.permute(0, 2, 1)` on the `[n_exp, inter, hidden]` reshape.

## GELU variants matter for NomicBERT

NomicBERT uses `nn.GELU(approximate='none')` (exact erf-based), not the
tanh approximation. ggml provides both: `ggml_gelu()` (tanh approx) and
`ggml_gelu_erf()` (exact). Per-element error is ~1e-4 but compounds over
12 layers. Use `ggml_gelu_erf` for NomicBERT (both MoE expert and dense
FFN layers). Standard BERT typically uses `gelu_new` (tanh approx).

## General OCR: DBNet + TrOCR

### ConvTranspose2d weight layout differs from Conv2d
PyTorch Conv2d: `(OC, IC, KH, KW)` → flattened `(OC, IC*KH*KW)`.
PyTorch ConvTranspose2d: `(IC, OC, KH, KW)` → flattened `(IC, OC*KH*KW)`.

ggml `conv_transpose_2d_p0` expects kernel `[KW, KH, OC, IC]` — note IC
and OC are swapped vs regular `conv_2d` kernel `[KW, KH, IC, OC]`.
Needed a separate `prep_deconv_weight()` that reshapes to `(KW, KH, OC, IC)`.

### ODR violations with common struct names
`struct dec_layer` was defined in both `math_ocr.cpp` (30 pointer fields,
240 bytes) and `decoder_embed_internal.h` (18 pointer fields, 144 bytes).
In the test binary (linking only math_ocr), the correct 240-byte version
was used. In the CLI binary (linking everything), the 144-byte version won,
causing heap-buffer-overflow when math_ocr tried to write 30 fields into
18-field-sized allocations.

Fix: namespace-prefix struct names (`math_ocr_dec_layer`). ASAN caught this
immediately — always test with the full binary, not just individual TU tests.

### XLM-R / SentencePiece fairseq vocab offset
TrOCR uses XLMRobertaTokenizer which adds a fairseq offset to SentencePiece
token IDs. Raw `SentencePiece.id_to_piece(43778)` returns the wrong string.
Must use HF `AutoTokenizer.convert_ids_to_tokens(43778)` to get correct
mapping. Also: use `convert_ids_to_tokens()` (not `decode()`) to preserve
the `▁` word boundary markers for proper space reconstruction.

### DBNet FPNC (FPN-Cat) architecture
MMOCR's FPNC is NOT standard FPN. Standard FPN: lateral (1×1) → top-down →
smooth (3×3), all at 256ch. FPNC: lateral (1×1, 256ch) → top-down → smooth
(3×3, **64ch**), then concatenate all 4 levels (4×64=256ch). No output conv.
The smooth conv reduces channels, not the lateral.

### ggml_interpolate replaces ggml_upscale_ext
`ggml_upscale_ext` is deprecated. Use `ggml_interpolate(ctx, a, ne0, ne1,
ne2, ne3, mode)` with `GGML_SCALE_MODE_BILINEAR` for FPN upsampling.
Nearest-neighbor vs bilinear makes a visible difference in detection parity
(cos_min drops from 1.0 to 0.0 with nearest on some rows).

## Quantizer skips 3D tensors

`tools/quantize.cpp` line 172 skips tensors with ndims > 2 ("skipping N-D
tensor (conv2d)"). This was added for face model conv kernels (4D) but
also catches MoE expert weights (3D: `[n_exp, dim1, dim2]`). For
nomic-v2-moe, this means expert weights stay F32 in all quants, limiting
Q8_0 compression to 1.6x instead of potential ~3x. Fix: quantize 3D
tensors by iterating over the outermost dimension.

## Qwen2.5-VL: KV cache for VLM generation

### Prefill K/V extraction pattern

The prefill forward pass computes all prompt tokens at once. To extract
per-layer K/V for caching, add output tensors **after mRoPE but before
GQA repeat**: the K/V at shape (head_dim, n_kv_heads, n_tokens) is what
goes into the cache. GQA repeat is reapplied in each decode step.

```cpp
// In prefill graph, after RoPE:
K_flat = ggml_reshape_2d(g, ggml_cont(g, K), kv_dim, n_tokens);
ggml_set_name(K_flat, "k_out_0");
ggml_set_output(K_flat);
```

### Decode step graph: single token + cache concat

The decode step takes one token embedding + cached K/V as inputs.
K/V cache tensors are 2D (kv_dim, n_kv) passed as graph inputs,
reshaped to 3D, concatenated with the new single-token K/V on dim 2,
then GQA-repeated for attention.

No causal mask needed — a single query token always attends to all
cached KV tokens (it's always the latest position).

### Token embedding lookup for quantized models

During decode, embed_tokens may be quantized (Q8_0/Q4_K). Can't just
index into the data directly. Solution: build a mini ggml graph with
`ggml_get_rows(embed_tokens, [token_id])` to handle dequantization.

### KV cache memory budget

36 layers × 2 (K+V) × kv_dim(256) × n_tokens × 4 bytes.
For 500 prompt tokens: 36 × 2 × 256 × 500 × 4 = 36 MB.
For 2000 tokens: 144 MB. Well within budget.

## Qwen2.5-VL: BPE tokenizer from GGUF

### Standard ggml tokenizer keys

Write to GGUF: `tokenizer.ggml.tokens` (string array), 
`tokenizer.ggml.merges` (string array), `tokenizer.ggml.model` = "gpt2",
`tokenizer.ggml.type` = 1 (BPE).

Load in C++: read arrays from GGUF metadata, pass to `BPETokenizer.load()`.

### GPT-2 byte-level decode

BPE tokens are unicode codepoints, not raw bytes. Decode: concatenate
token strings, then reverse the `bytes_to_unicode()` mapping. The table
maps printable ASCII + Latin-1 to themselves, and remaining bytes to
codepoints 256+. Build the inverse table once at init.

### Chat template in C++

Hardcode special token IDs (im_start=151644, system=8948, user=872,
assistant=77091, etc.) and use the BPE tokenizer for the user prompt
text only. This avoids needing a Jinja template engine in C++.

## Qwen2.5-VL: ggml_set_output memory impact

Marking N intermediate tensors as output prevents ggml's graph allocator
from reusing their memory. For 32 ViT + 36 LLM layers, this adds ~500 MB
of pinned memory — enough to OOM on 8 GB machines.

Fix: only set_output when diff comparison is active (`ctx.diff_ref_path`
is non-empty). Logits tensor always needs set_output.

## Kaggle: always use the full harness

Never simplify or inline the CrispASR kaggle_harness.py. It has:
- `kh.build_heartbeat()` — prevents Kaggle killing long ops (uploads)
- `kh.resolve_hf_token()` — 3-tier auth (env → Secret → dataset file)
- `kh.step()` — JSONL progress to /kaggle/working + HF mirror
- `kh.install_build_toolchain()` — ninja + ccache + mold

Bundle `kaggle_harness.py` in the push directory as fallback.
Use `chr1s4/crispasr-hf-token` dataset (chr1s4's own, not chr1str's).
Don't `pip install torch` (pre-installed, 2 GB download wastes time/OOMs).
## LFM2 backbone: causal → bidirectional (GLiNER NER port)

Porting the LFM2.5 backbone from CrispASR (causal audio model) to
CrispEmbed (bidirectional NER encoder) required exactly two changes:

1. **Attention mask**: causal `(j <= i) ? 0 : -inf` → pass `nullptr`
   to `ggml_flash_attn_ext` for full bidirectional attention.
2. **Conv padding**: left-pad `pad=K-1` → center-pad `pad=(K-1)/2`
   for symmetric (bidirectional) convolutions.

Everything else (RMSNorm, SwiGLU FFN, RoPE, GQA, ShortConv gating)
is identical. The layer_types string `"ccaccaccacacacac"` is the same
pattern for both the 1.5B audio and 350M NER models.

## GLiNER layer fusion: sigmoid not softmax

GLiNER's `LayersFuser` uses **sigmoid** gates (independent per-layer),
NOT softmax (competing across layers). The squeeze-and-excitation
pattern is: squeeze(hidden→1) per layer → mean over tokens → W1→ReLU→W2
→ **sigmoid** → element-wise multiply each layer → sum → output_projection.

Using softmax instead produced cos=0.65 vs reference. Sigmoid gives
cos=1.000000.

## GLiNER pipeline: word-level pooling before BiLSTM

GLiNER's `subtoken_pooling="first"` means: after the backbone + layer
fusion, take the first BPE subtoken of each word to get word-level
representations, THEN run the BiLSTM on word-level only. The entity
type reps (at `<<ENT>>` positions) are extracted from the fused
token-level output BEFORE the BiLSTM.

Running the BiLSTM on the full token sequence (including label prefix
tokens) produces cos=-0.96 vs reference. Word-level gives cos=1.000000.

## GLiNER tokenization: regex word splitter

GLiNER's `WhitespaceTokenSplitter` uses regex `r"\w+(?:[-_]\w+)*|\S"`,
NOT simple whitespace splitting. This separates punctuation from words:
"Cupertino," → ["Cupertino", ","]. Simple whitespace splitting glues
punctuation to the word, causing entity span mismatches.

## GLiNER input format

The input sequence is: `BOS <<ENT>> label1 <<ENT>> label2 ... <<SEP>> text EOS`.
Note: `<<ENT>>` before each label (not `<<SEP>>` between), single
`<<SEP>>` after all labels, BOS at start, EOS at end.

## ggml_conv_1d_dw requires F16 kernel weights

`ggml_conv_1d_dw` internally uses `ggml_im2col` which asserts
`src0->type == GGML_TYPE_F16`. When model weights are F32, cast
the conv kernel to F16 before passing to `ggml_conv_1d_dw`:
`ggml_cast(ctx, w.conv_conv_w, GGML_TYPE_F16)`.

## ggml_gallocr works with model weight tensors

Model weight tensors that already have a backend buffer are skipped
by `ggml_gallocr_alloc_graph` — it only allocates compute tensors.
So model weights can be used directly as operands in graphs allocated
with gallocr. No need for `ggml_backend_sched` for this use case.

However: `ggml_add` with a 1D bias tensor (ne[0]=D) broadcasts
correctly over a 2D tensor (D, N) — no `ggml_repeat` needed.
Using `ggml_repeat` with a reshaped view of a weight tensor can
cause subtle issues.

## Dequantizing backend tensors to CPU

Model weight tensors in Q8_0/Q4_K backend buffers can't be read
with `ggml_backend_tensor_get(t, dst, 0, nelements*sizeof(float))`
— that reads raw quantized bytes. Use:
```cpp
std::vector<uint8_t> raw(ggml_nbytes(t));
ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
ggml_get_type_traits(t->type)->to_float(raw.data(), out, nelements);
```

## Cache dequantized weights for CPU-side ops

When CPU-side operations (BiLSTM, layer fusion) read quantized model
weights every call, the dequant overhead adds ~50-100ms per call.
Cache the F32 versions in the context struct at init time — they're
small (~52 MB for BiLSTM + fuser weights) and eliminate per-call cost.

## Batched span MLP via ggml graph: 2-3x speedup

GLiNER's span scoring evaluates hundreds of spans, each through a
2-layer MLP (3072→4096→1024). The naive approach runs each span as a
separate CPU scalar matmul. Batching all spans into a single ggml
`mul_mat` (3072, n_spans) × (3072, 4096) leverages BLAS and gives
2-3.5x speedup on the GLiNER head.

Two-pass approach works well: pass 1 computes proj_start/end/first +
prompt_rep (independent of spans), then CPU assembles span
concatenations, pass 2 computes batched out_project + scoring.

## Swin shifted-window attention: cyclic_shift vs torch.roll

When implementing `torch.roll(x, shifts=-s, dims)` as a C++ cyclic shift
function, the sign convention is inverted:
- `torch.roll(shifts=s)`: `out[y] = in[(y - s) % H]`
- `cyclic_shift(shift_h=s)`: `out[y] = in[(y + s) % H]`

So `torch.roll(shifts=-3)` = `cyclic_shift(shift_h=+3)`. Getting this
wrong produces cos=0.0 on the shifted data — completely scrambled.

Also: HF Swin pads to window-size multiples BEFORE the cyclic shift.
The mask is built on the padded dimensions. If you shift first then pad,
the data layout differs in the boundary/padding zone.

## Swin GELU: tanh approx vs exact erf

HF Swin uses `nn.GELU()` which is exact erf-based GELU:
`0.5 * x * (1 + erf(x / sqrt(2)))`. The commonly-used tanh
approximation `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`
introduces small systematic errors that compound through multiple
layers. Always check which variant the upstream model uses.

## Per-step diff isolates bugs faster than E2E comparison

When block 1 had cos=0.995, comparing only the block output gave no
clue where the error originated. Adding per-step diff checkpoints
(LN1 → shift → windows → attention → merge → residual → FFN → output)
immediately showed that LN1 was perfect (cos=1.0) but shifted data
was cos=0.0 — pinpointing the cyclic shift function as the culprit.
Always instrument fine-grained checkpoints for shifted/masked operations.

## Swin PatchMerging: sub-patch concat order matters

HF Swin PatchMerging concatenates 2×2 sub-patches in the order
`[TL, BL, TR, BR]` (top-left, bottom-left, top-right, bottom-right):
```python
x0 = x[:, 0::2, 0::2, :]  # top-left
x1 = x[:, 1::2, 0::2, :]  # bottom-left
x2 = x[:, 0::2, 1::2, :]  # top-right
x3 = x[:, 1::2, 1::2, :]  # bottom-right
```
The natural C loop `(2y, 2x), (2y, 2x+1), (2y+1, 2x), (2y+1, 2x+1)`
gives `[TL, TR, BL, BR]` — positions 1 and 2 swapped. This feeds the
wrong channels into the LayerNorm+Linear reduction, corrupting ALL
subsequent stages (cos=-0.37 at encoder output).

## RoBERTa position embedding offset (+2)

RoBERTa position embeddings have `padding_idx=1`. Content positions
start at index `padding_idx + 1 = 2`. So decode step 0 uses
`pos_embed[2]`, step 1 uses `pos_embed[3]`, etc. Using index 0
(random values, norm=0.57) and index 1 (all zeros, padding) produces
wrong embeddings that cascade through the decoder.

## Tesseract `.traineddata` binary format

The `.traineddata` file is a custom archive: `int32 n_entries`, then
`n_entries × int64` offsets (-1 = absent). Component 17 = LSTM network.

The LSTM component has a recursive binary tree starting with
`Network::Serialize`: `int8 type_enum` (if 0 → read type string),
then `int8 training, int8 needs_bp, int32 flags, int32 ni, int32 no,
int32 num_weights, uint32 name_len + chars`.

Key gotcha: `kDoubleFlag = 128` (not 2!). Controls whether scales/arrays
use float64 vs float32. The kInt8Flag is 1, kAdamFlag is 4.

## Tesseract int8 weight dequantization

Stored scale = `runtime_scale * INT8_MAX`. At load:
`loaded_scale = stored / 127`. Runtime: `output = dot(int8_w, int8_input) * loaded_scale`.
For float dequant: `float_w = int8_w * stored_scale` (the RAW value,
NOT divided by 127). The factor of 127 cancels with the int8 input
quantization. Dividing by 127 gives weights 127x too small.

## Tesseract Convolve layer is NOT a learned conv

`Convolve` in Tesseract just stacks (im2col) the 3×3 neighborhood —
no trainable weights. The actual "convolution" is a `FullyConnected`
layer with tanh activation in a `Series` after the `Convolve`.

## XYTranspose wraps SummLSTM, not the other way

`Reversed(NT_XYTRANSPOSE)` extends `Plumbing` — it's a container that
reads child networks (including count + learning rates). The `Lfys`
VGSL layer is actually: `XYTranspose(SummLSTM)` where the XYTranspose
swaps axes before and after so the LSTM runs over the y-dimension.

## Vertical shear for deskew, not horizontal

Leptonica's `pixFindSkew` uses VERTICAL shear (shift columns up/down)
to score alignment, NOT horizontal (shift rows left/right). Horizontal
shear doesn't change row sums for horizontal text lines. Vertical
shear moves pixels between rows, which is what the differential
square-sum scoring measures.

## 1-bit DWA morphology: massive speedup from word-level ops

Packing 32 pixels per uint32 and using word-level OR (dilation) gives
21x speedup over float separable morphology and 32x less memory.
The cache efficiency from 1-bit packing dominates even over the
algorithmic improvement.

## PAN super-resolution for low-DPI OCR

Tested PAN 4x upscale on 75dpi text (150×9 px → 600×36 px):
- 75dpi raw → Tesseract: "C Melbe Wesld1" (garbage)
- 75dpi + PAN 4x → Tesseract: "Hello Werdd 123" (1 char error)
- 150dpi raw → Tesseract: "Hello World 123" (perfect)

Key findings:
1. PAN 4x rescues unreadable 75dpi text — garbage → mostly correct
2. Don't cleanup (binarize/deskew) before SR — destroys sub-10px text
3. Don't cleanup after SR either — upscaled text is clean enough
4. For 150dpi+, no SR needed — OCR works fine directly
5. Optimal pipeline: estimate DPI → if < 150, PAN 4x → OCR

The `estimate_dpi()` heuristic assumes longer edge ≈ 11 inches (A4/letter).
This is wrong for cropped regions but acceptable for full pages.

## SigLIP ViT ggml graph: tensor layout and permute pattern

SigLIP ViT (D=1152, n_heads=16, d_head=72, T=729 patches) ggml graph
matches the pattern established in `bidirlm_vision.cpp` (Qwen2VL ViT).
Key differences vs Qwen2VL:

1. **No RoPE**: SigLIP uses absolute position embeddings added before
   the transformer loop. No cos/sin tensors needed.

2. **Separate Q, K, V projections**: bidirlm uses fused QKV; SigLIP
   uses three independent `vis.layer.N.attn.{q,k,v}.weight` matrices.
   Three `ggml_mul_mat` calls per layer instead of one.

3. **GELU (tanh approx)** in FFN: `ggml_gelu`. Not `ggml_gelu_erf`.

**Tensor shapes through the attention block:**
- Input x: `[D, T]` ggml (ne[0]=D fast dim, ne[1]=T tokens)
- After QKV mul_mat + bias: `[D, T]`
- After `reshape_3d(Q, d_head, n_heads, T)`: `[d_head, n_heads, T]`
- After `permute(0, 2, 1, 3)` + `cont`: `[d_head, T, n_heads]` contiguous
- scores = `mul_mat(K, Q)`: K=[d_head,T,n_heads], Q=[d_head,T,n_heads]
  → `[T_k, T_q, n_heads]`. `ggml_mul_mat(a,b)` computes b@a^T so
  the inner dim (ne[0]) must match: both ne[0]=d_head ✓
- After `soft_max_ext(scores, null, 1/sqrt(d_head), 0)`: same shape,
  softmax over dim 0 (key axis) ✓
- V_perm = `permute(V, 1, 0, 2, 3)` + `cont`: `[T, d_head, n_heads]`
- attn = `mul_mat(V_perm, scores)`: [T,d_head,n_heads] × [T_k,T_q,n_heads]
  → `[d_head, T_q, n_heads]`
- After `permute(attn, 0, 2, 1, 3)` + `cont`: `[d_head, n_heads, T]`
- After `reshape_2d(attn, D, T)`: `[D, T]` — per-token D-vector with
  head-major interleaving: [h0_d0, h0_d1, ..., h1_d0, ...] ✓

**ggml_norm broadcasting**: `ggml_norm(g, x, eps)` normalizes along ne[0].
For x=[D,T], each of the T token vectors is independently normalized ✓.
`ggml_mul(g, normed, w)` where w=[D] broadcasts over T via `ggml_can_repeat`
(T % 1 == 0). No reshape needed for bias/scale vectors.

**Feature extraction**: `ggml_set_output(ggml_cont(g, x))` at each
`feature_layers[fi]` layer index. The ggml_cont ensures the feature tensor
is a separate node (not an alias of x); the scheduler keeps its buffer live
after the full graph runs. Four feature layers → four independent tensors
read back via `ggml_backend_tensor_get` after compute.

**compute_meta buffer**: Pre-allocated in `vis_compute_meta` (owned by
`granite_vision_context`). Passed to `ggml_init({size, data, no_alloc=true})`.
`ggml_free(g)` after compute frees only the `ggml_context` struct (small
malloc), NOT the buffer itself — so the graph nodes in compute_meta stay
valid while the scheduler holds the alloc. Buffer is overwritten on the
next inference call (graph rebuilt from scratch each call). This avoids
heap allocation per inference. Same pattern as `bidirlm_vision.cpp`.

**Scalar fallback**: `gv_run_vit_graph` sets `feat_outs.assign(n_feat, {})`
at entry. If the graph fails (alloc fails, compute fails, missing weights),
it returns early with all feat_outs empty. The caller checks
`n_feat > 0 && layer_outputs[0].empty()` to trigger the scalar path.

## granite_vision: converter writes weights transposed — reshape before ggml_mul_mat

`models/convert-granite-vision-to-gguf.py` writes 2D weights with their
PyTorch `[out, in]` shape **un-reversed**, so the GGUF `ne` is transposed
relative to ggml's convention (ne[0] = the fast/contraction = `in` dim). The
CPU-scalar path (`gv_linear` + DequantCache) is immune — it dequantizes the
raw bytes and indexes with explicit `id`/`od`. But **`ggml_mul_mat` asserts**
`GGML_ASSERT(ggml_can_mul_mat(a, b))` for every non-square weight (k/v:
512×2048, gate/up: 8192×2048, down: 2048×8192, projector linear_1:
2048×4608, vision fc1/fc2). Square weights (q/o, linear_2) happen to pass.

Fix: relabel the contiguous data with a reshape before the matmul —
`ggml_mul_mat(g, ggml_reshape_2d(g, w, w->ne[1], w->ne[0]), x)`. This is a
no-op for square weights and a pure view (no copy) otherwise. The projector and
LLM graphs (`gv_run_projector_graph`, `gv_run_llm_body`) were missing it and
crashed until `feat/granite-vision-ne-fix`.

**Caveat (this is a real footgun — see next entry):** the reshape is only valid
on a *quantized* tensor when the **target `ne[0]` is a multiple of the quant
block size**. It happens to hold for the Q4_K LLM weights (256-block; reshaped
`ne[0]` ∈ {2048, 8192}) and the Q8_0 projector (32-block; 4608/2048), but NOT
for the Q8_0 vision `ffn.down` (32-block; reshaped `ne[0]=4304`, 4304 % 32 ≠ 0).
There the reshaped view mis-strides the blocks and silently returns garbage.

Beware: a *systemic* load-time `ne` swap would double-correct the vision FFN
(which already reshapes per-site) — pick one approach, not both.

## granite_vision: the ggml-graph divergence was a Q8_0 reshape, not alloc reuse (2026-06-21)

`gv_run_vit_graph` on Metal produced cos ~0 from the first encoder layer with
`max_abs` exploding through the residual stream. A prior handover diagnosed an
"unpatched ggml-alloc in-place buffer-reuse defect" (claiming the input `x_t`
went NaN after compute). **That was wrong** — re-verifying with
`CRISPEMBED_GRANITE_VIS_DBG=1` showed `x_t` *unchanged* after compute, no NaN.

Real cause: `ggml_reshape_2d` on a **quantized** weight whose target `ne[0]`
isn't a multiple of the quant block size. The converter stores 2D weights with
transposed `ne` (ne[0]=out), so the FFN matmul relabels via reshape. For the
Q8_0 `vis.layer.*.ffn.down` weight, `ne[0]` goes 1152→4304 and 4304 % 32 ≠ 0,
so the reshaped view's 32-element Q8_0 blocks no longer align to rows — every
down-projection from layer 0 onward is garbage that compounds through the
residual adds (hence "explodes" and looks like an alloc smash). The square Q8_0
attention weights are used raw (ne[0]=in=1152 already carries the blocks) and
were always fine; the F16 `ffn.up` reshapes safely (per-element). This mirrors
the merged `restormer`/`nafnet` "transposed 2D conv weight" fix, which also
casts to F32 before reshaping a quantized weight.

Fix (`gv_run_vit_graph` FFN): dequantize quantized weights to F32 *before* the
reshape — `ggml_is_quantized(w->type) ? ggml_cast(g, w, F32) : w`, then
`ggml_reshape_2d`. `ggml_cast` reads the Q8_0 with its *original* (correct) `ne`
so the dequant is right; the post-cast F32 reshape is a plain element relabel.
Result: per-layer parity with the scalar ViT (cos 0.9996–0.99987 early; 0.96
late = the scalar's `F.layer_norm` eps 1e-5 vs model 1e-6 ref artifact, NOT a
bug), and ~3 s vs ~100 s scalar. Gated behind `CRISPEMBED_GRANITE_VIS_GRAPH=1`.
Two caveats keep it opt-in: the **ggml-CPU** backend still drifts at late layers
(cos ~0.84 — only Metal is validated), and the full Metal OCR path is still
blocked on the LLM Metal graph.

**Separately on the LLM Metal graph:** the same handover called it "broken on
Metal (emits EOS immediately)". The 7-token self-consistent diff
(`granite-llm-ref`) **passes on Metal at cos 0.9999 through logits**, so the LLM
math/weights/reshape are correct. But a full 784-token prefill with spliced image
features yielded **0 decode steps** (first token = EOS) on Metal while the same
prefill on ggml-CPU decoded correctly. Root cause was NOT attention/length — see
the dedicated "Metal mul_mm F16 activation overflow" entry below.

## Metal mul_mm F16 activation overflow on massive activations (2026-06-21)

The Granite LLM Metal graph (`gv_run_llm_body`) produced correct logits for text
tokens at any length, and correct results on the ggml-CPU backend, but cascaded
to **NaN from layer 9** during the real OCR prefill on Metal — yielding 0 decode
steps. Localized by dumping per-layer max-abs over all tokens: the residual
stream carries a **massive activation** (~1.1e4 — a few outlier dims, amplified
~×12 by the `embedding_multiplier` applied to the spliced image features), held
fine through layer 7, then exactly 2 elements overflow to `inf` at layer 8.

Root cause: Apple's batched matmul kernel `kernel_mul_mm_*` (selected when
`ne11 > ne11_mm_min`, hardcoded 8, i.e. the T=784 **prefill**; T=1 **decode** uses
`mul_mv`) casts activations to **F16** for the simdgroup matrix units. The SwiGLU
`silu(gate)*up` product for the massive dimension exceeds F16 range (65504) at
the first layer whose FFN weights are large enough → `inf` → NaN. This is the
same phenomenon `ggml_mul_mat_set_prec(GGML_PREC_F32)` targets "for phi-2" — but
on Metal that flag does NOT change `mul_mm` (the kernel name has no prec variant),
so it does not help here. F32 KV cache, disabling fusion/concurrency, and manual
attention all leave the overflow (it is in the FFN matmul, not attention/KV).

Fix (`gv_run_llm_body` FFN): scale the down-projection activation down before the
matmul and back up after — a **lossless exponent shift** (F16 is floating-point,
so scaling preserves relative precision) that keeps the F16 cast in range:
`down = scale(mul_mat(Wd, scale(silu(gate)*up, 1/256)), 256)`. 256 gives headroom
to ~1.6e7. Result: full Metal OCR (vision graph + LLM graph) returns the correct
text in ~22 s (vision ~3 s, 784-tok prefill ~12 s, decode ~5 s) vs the scalar
path's ~100 s vision + ~8 min prefill. Both graphs now DEFAULT ON for GPU
backends; scalar stays default on ggml-CPU (where the ViT still drifts).

Diagnostic lesson: the LLM diff originally exercised only the scalar
`gv_llm_decode_step`, NOT the ggml graph — so "LLM diff passes" said nothing
about `gv_run_llm_body`. Threading a dump_cb through `gv_run_llm_body` and running
it under `granite_vision_dump_llm` when `CRISPEMBED_GRANITE_LLM_GRAPH=1` is what
finally put the graph under the harness. Always confirm the diff drives the path
you think it does.

## ggml-CPU ViT precision: F16-table gelu + Q8_0-quantized activations (2026-06-21)

After the Metal Granite ViT graph reached scalar parity, the **ggml-CPU** backend
of the same graph still drifted to cos ~0.84 at late layers (vs Metal's 0.96),
accumulating over the 27 encoder layers. Two CPU-only precision losses, both
absent on Metal (which runs these in F32):

1. **`ggml_gelu` (tanh) on CPU is a precomputed F16 lookup table.**
   `ggml_vec_gelu_f32` (ggml-cpu/vec.h) does `ggml_table_gelu_f16[fp32_to_fp16(x)]`
   — the input is quantized to F16 and the result is an F16 table value. `silu`
   (the LLM FFN) is computed directly in F32, which is why the LLM CPU graph never
   showed this. `ggml_tanh`/`ggml_gelu_erf` on CPU are also direct F32. Fix:
   compute tanh-gelu from primitives —
   `0.5*x*(1 + tanh(√(2/π)*(x + 0.044715*x³)))` with `ggml_tanh`/`ggml_mul`/
   `ggml_scale` (all F32). Closed about half the gap (0.84 → 0.90).

2. **CPU `mul_mat` against a Q8_0 weight quantizes the F32 activation to Q8_0**
   for the dot product (its `vec_dot_type`), coarser than Metal's `mul_mm` F16
   activation cast. The square attention q/k/v/o weights were used raw (Q8_0), so
   on CPU every attention projection ran an int8 activation dot → drift. Fix:
   `ggml_cast` the (square) attention weights — and the F16 FFN up — to F32 **on
   the CPU backend only** (`ggml_backend_is_cpu(ctx->backend)`), forcing an F32
   activation dot. No-op on GPU, where raw Q8_0 + `mul_mm` is already accurate and
   faster, and avoids the extra F32 weight copy. Closed the rest (0.90 → 0.958 =
   scalar parity); CPU end-to-end OCR then returns the correct text.

Both Granite graphs are now correct on Metal AND ggml-CPU and default ON for all
backends. Takeaway: ggml's CPU backend silently trades precision for speed
(F16-table activations, activation requantization to the weight type); when a
graph drifts on CPU but not GPU, suspect those before the math.

## Self-consistent crispembed-diff reference from the GGUF — no original weights needed (2026-06-21)

To numerically validate a backend you don't have the original framework checkpoint
for, build the diff reference from the **GGUF the runtime already loads**: reverse
the converter's name map (every `convert-*-to-gguf.py` has an explicit
`add(gguf_name, torch_name)` table), dequantize each gguf tensor, build the torch
`state_dict`, `load_state_dict(strict=False)` into the original arch, run the
forward, and dump input+stages+output as the `-ref.gguf`. This proves
**C++ runtime == framework on identical weights** (catches algorithm bugs; not
weight-conversion bugs — for those use the real checkpoint). It's the granite-llm
-ref pattern generalized; used to verify hat_sr's OCAB at output cos 0.999968 with
no HAT `.pth`. See `tools/dump_hat_reference_from_gguf.py`. Gotchas: computed
buffers (`relative_position_index`, `attn_mask`, `mean`) aren't in the gguf —
`strict=False` keeps the arch's `__init__`-computed ones; assert that no *weight*
keys are missing. Pick a small input size (e.g. 32) so the scalar C++ forward in
the diff harness finishes quickly. Refs get uploaded to the model's HF repo.

Corollary methodology trap: **a diff test that exists in `tests/` but isn't wired
into CMake was never actually run.** `tests/test_hat_diff.cpp` was present for
months but had no `add_executable` — so "there's a test for HAT" was false. When a
backend is "validated," confirm the harness target is built and the reference
exists, not just that the `.cpp` is in the tree.

## VLM/OCR decoder perf: keep the LM head on-GPU; don't re-copy KV history (2026-06-21)

A cross-backend audit of the OCR decoders (granite_vision vs qwen2vl_ocr,
internvl2_ocr, smoldocling_ocr, deepseek_ocr2) surfaced two cheap decode wins
granite was missing — both worth checking on any ggml decoder:

1. **Run the LM head (vocab projection) IN-GRAPH, not on CPU.** Granite computed
   the ~`vocab×dim` logits with `core_cpu::linear_cpu` every token, which also
   forces a hidden-state GPU→CPU readback. qwen2vl/internvl2/deepseek append
   `ggml_mul_mat` for the last token to the graph and read back `[vocab]`. Moving
   it on-GPU (`gv_run_llm_body`'s optional `logits_out`) cut granite decode
   270→165 ms/tok (~1.6×). Gotchas: the last-token hidden is post-final-norm so
   it's O(1) → the T=1 `mul_mv` is F32-safe (no massive-activation overflow); and
   Granite's tied embedding weight has the transposed `ne` (`ne[0]=vocab`), so
   relabel it to `[in=dim, out=vocab]` (cast quantized first) before `mul_mat` —
   same pattern as the FFN weights.

2. **Don't `ggml_cont` the KV history every step.** `flash_attn_ext` reads K/V via
   `nb` strides, so pass the `[d_head, Lk, n_kv]` cache *views* straight in. The
   `cont` was copying the entire history each layer each token (~64 MB/step at
   len ~800). Writing new K/V into the cache still uses `ggml_cpy` into the
   persistent buffer (correct); only the read-side full-history copy was waste.

**Cross-backend reference** (who does what, as of 2026-06): LM head on-GPU —
qwen2vl, internvl2, deepseek (granite now too); LM head on CPU — smoldocling.
**Decode-graph reuse**: only `deepseek_ocr2` builds a *persistent* T=1 decode
graph once (`build_persistent_decode_graph`) and just sets inputs + recomputes
per step; everyone else (incl. granite) rebuilds the graph per token. qwen2vl's
cheaper half-measure: keep KV-read views + mask at constant `[0..max_seq]` shape
so `ggml_backend_sched_alloc_graph` takes the no-realloc fast path. KV cache is
F16/Metal-resident across all. No OCR decoder uses `sched_reserve` yet.

**Profile before chasing graph reuse.** Env-gated timers around granite's
`gv_run_llm_body` showed a T=1 decode token is **~95 % GPU `graph_compute`**
(~135 ms); `ggml_init`+graph build ≈ 0.8 ms and `sched_reset`+`sched_alloc_graph`
≈ 5 ms are noise. So a persistent decode graph / `sched_reserve` would save ~5 ms
of ~140 ms — **not worth the refactor here** (contrast the LEARNINGS note above
where alloc *was* the bottleneck for a 16-layer 350 M model; for a 2 B/40-layer
model the GPU compute dominates). Decode is **dispatch-bound on ~800 tiny T=1
kernels** (each `mul_mv` etc. underutilizes the GPU), so the lever is **fewer
kernels per token**, not graph management. Concrete win: the SwiGLU down-proj
F16-overflow guard (÷256/×256 exponent shift) only matters for the prefill
`mul_mm` F16 cast — skip it for T ≤ 8 (decode `mul_mv` is F32-safe), cutting 2
dispatches/layer → decode 165 → 139 ms/tok. Cumulative 270 → 139 (~1.9×).

The one-shot *total* is still dominated by the 784-token prefill (~12 s, ~100 %
GPU compute) + Metal **pipeline compilation** (compiled lazily on first kernel
use, so it lands inside the first vision/prefill `graph_compute` and inflates
one-shot timings by seconds; a persistent server pays it once). The decode wins
compound for long (multi-hundred-token) document OCR, which is the real workload.

## granite_vision OCR: the two image-path bugs (vision parity passed but OCR hallucinated)

The vision tower passed crispembed-diff at cos≈0.99999, yet end-to-end OCR
produced a fluent but wrong document (`<doc>…indd…</doc>`). Two bugs, both in
the *inference* path the parity test never exercised (it feeds an already-
ranged reference image and skips the LLM):

1. **SigLIP normalization**: feed `(pixel/255 - 0.5)/0.5` → `[-1, 1]`
   (preprocessor_config: mean=std=0.5), not `[0,1]`. Wrong range → garbage
   features → hallucination.
2. **Image features ×embedding_multiplier**: HF `LlavaNextForConditionalGeneration`
   scatters raw projector features into `inputs_embeds`, then the Granite LM
   multiplies the *whole* tensor (text + image) by `embedding_multiplier`
   (12.0). So spliced vision rows must be ×12 too — otherwise they are 12×
   weaker than text and the LM ignores the image. The one-space difference
   between two runs (with vs without the normalization change) was the tell:
   the image *was* reaching the LM but far too weakly to matter.

The Granite LLM decode itself (RoPE = HF `rotate_half` / ggml NEOX, attention
scaled by `attention_multiplier` = 1/64 not 1/√d, embedding/residual/logits
multipliers, tied lm_head) is validated layer-by-layer by
`tools/dump_granite_llm_reference.py` (builds the reference straight from the
dequantized GGUF — no 5 GB HF checkout needed) at cos=1.0.

## granite_vision: a self-consistent diff can hide a broken graph (2026-06)

CRITICAL methodology lesson. There are TWO references and they prove different
things:
- `tools/dump_granite_llm_reference.py` → `granite-llm-ref.gguf` is built from
  the **same dequantized GGUF** the C++ uses. Passing it (cos 1.0) only proves
  the C++ math matches a numpy reimplementation on identical weights. It CANNOT
  catch weight corruption or a wrong runtime backend.
- `tools/kaggle/granite-vision-parity/granite_vision_parity.py` →
  `granite-vision-ref.gguf` is built from the **real HF safetensors** (true
  parity). It catches weight AND code bugs. Download from
  `cstr/granite-vision-crispembed-GGUF` (37 MB) — no need to rerun on Kaggle.

A prior handover trusted the self-consistent LLM cos 1.0 + a stale "vision cos
0.99998" and concluded OCR garbage was a prompt/template bug. The HF-blueprint
diff instead showed the **Metal/ggml SigLIP ViT graph (`gv_run_vit_graph`)
outputs cos 0.05** while patch_embed (CPU scalar) is cos 1.0 and the scalar ViT
loop is cos 0.96. To localize, `granite_vision_dump_vision` was extended to emit
`vis_patch_embed` + per-feature-layer `vis_layer_N` stages, and
`CRISPEMBED_GRANITE_VIS_SCALAR/CPU` levers added: the break is at the first
encoder layer, identical across q4_k/q8_0 and flash/manual attn.

**CORRECTION (2026-06-21): the alloc-reuse diagnosis below was WRONG.** A later
session re-verified independently: on Metal the graph's input `x_t` is *intact*
after compute (not NaN), so there is no in-place buffer reuse. The "identical
across q4_k/q8_0 and flash/manual attn" evidence does *not* implicate alloc — it
equally fits a systematic weight-layout bug, since both test models carry **Q8_0
vision** weights and the bug is in the FFN, not attention. Real cause: see the
"Q8_0 reshape" entry directly below. The fix (cast the FFN weights to F32 before
the transposed-ne reshape) brings every `gv_run_vit_graph` layer to scalar
parity (cos 0.9996–0.99987 early; late layers track the scalar's 0.96 ref-eps
artifact) and runs ~3 s vs ~100 s scalar. Lesson preserved: a self-consistent
diff (`granite-llm-ref`) can't catch backend/weight bugs — only the HF-blueprint
`granite-vision-ref` can; AND a handover's stated root cause must be re-verified,
not trusted (here "NaN on CPU / alloc reuse" was never reproduced).

Also: the on-disk `*-q4_k.tok.gguf` had **Q4_0 vision weights** (quantized
before the `vis→Q8_0` fix in `tools/quantize.cpp`), collapsing scalar vision to
cos 0.32. Requantize from F16 with the current quantizer (now also keeps
`proj.*` at Q8_0) → vision parity matches q8_0. `projector_hidden_act="gelu"`
is exact **erf** (`ggml_gelu_erf`), NOT the vision tower's tanh.

## ggml_flash_attn_ext handles GQA natively (2026-06)

`ggml_flash_attn_ext` supports Group Query Attention without explicit
`ggml_repeat` to tile KV heads to match Q head count. The CPU kernel
(ops.cpp:8216) uses broadcast factors `rk2 = neq2/nek2` to map Q heads
to KV heads internally. This means you can pass K/V with `n_kv_heads`
directly — the kernel repeats automatically.

Before: 20+ tensor ops per layer (reshape_4d + new_tensor_4d + repeat +
reshape_3d, twice for K and V). After: zero ops, just pass nkv-head
tensors. Applied to internvl2, lightonocr, got_ocr, glm_ocr (-76 lines).

## F16 norm weights need ggml_cast in ggml graphs (2026-06)

Q4_K GGUF models store RMSNorm/LayerNorm weights as F16 (not F32).
When building ggml graphs, `ggml_mul(rms_norm_output, weight)` fails
with "binary_op: unsupported types: dst: f32, src0: f32, src1: f16"
because `ggml_rms_norm` always outputs F32 but the weight is F16.

Fix: `if (w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);`
before the multiply. This affected smoldocling's ggml LLM decoder and
got_ocr's patch embedding bias on Q4_K models.

## DenseNet ggml graph: ceil_mode pooling + concat dimension (2026-06)

Converting DenseNet encoders to ggml graphs requires careful handling of:

1. **ceil_mode pooling**: `ggml_pool_2d` uses floor mode. For ceil_mode
   equivalent, pad the spatial dims by 1 before pooling (`ggml_pad`).
   Getting this wrong causes spatial dimension mismatches in `ggml_concat`.

2. **DenseNet concat**: `ggml_concat(a, b, 2)` concatenates along ne[2]
   (channels in ggml's W,H,C layout). The assertion `a->ne[0]==b->ne[0]
   && a->ne[1]==b->ne[1]` means spatial dimensions must match exactly.
   Any pooling size mismatch cascades into a concat assertion failure.

3. **Conv weight layout**: ggml_conv_2d needs weights as F16 4D tensors
   [KW, KH, IC, OC]. Weights stored as 2D [IC*KH*KW, OC] need
   `ggml_reshape_4d` + `ggml_cast(F16)` before use.

hmer_ocr's ggml encoder is the working reference (`273969d`).
bttr/posformer share the same architecture but need separate
implementation due to different BN handling (folded vs separate).

## flash_attn_ext requires non-contiguous (permuted) inputs (2026-06)

`ggml_flash_attn_ext` reads Q/K/V strides from the tensor metadata to
handle multi-head layout. The standard pattern (from `vit_embed.cpp`):

```cpp
Q = ggml_permute(g, Q, 0, 2, 1, 3);  // [hd, T, nh] — non-contiguous view
// do NOT ggml_cont()!
attn = ggml_flash_attn_ext(g, Q, K, V, mask, scale, 0, 0);
attn = ggml_reshape_2d(g, attn, D, T);  // output [hd, nh, T] → [D, T]
```

Adding `ggml_cont()` before `flash_attn_ext` copies the data into a new
contiguous layout where the head/token axes are transposed from what
`flash_attn_ext` expects. This produces completely wrong attention output
(cos=0.56 vs reference). The fix: remove `ggml_cont()` and pass the
permuted views directly. Output is `[hd, nh, T]`, `reshape_2d(D, T)`
works directly without an extra permute.

## PIL ImageOps.pad: bicubic + clamp (2026-06)

PIL's `ImageOps.pad` calls `Image.resize(BICUBIC)` then pastes centered.
To match exactly in C++:

1. **Bicubic kernel**: Catmull-Rom (`a = -0.5`), NOT Keys (`a = -1`).
2. **Coordinate mapping**: `sx = (x + 0.5) / scale - 0.5` (center-to-center).
3. **Clamp output to [0, 1]**: Catmull-Rom overshoots at sharp edges (text
   boundaries produce values > 1.0 or < 0.0). PIL clips internally to
   [0, 255] before returning uint8. Without clamping, cos_min drops from
   0.9999 to 0.991 at the patch embedding stage.
4. **Padding value**: `int(mean * 255) / 255.0`, NOT `mean` directly.
   `int(0.5 * 255) = 127`, `127/255 = 0.498039 ≠ 0.5`.

## Never blame quantization for parity failures (2026-06)

Quantization (F16, Q8_0, even Q4_K) explains at most ~5% cosine
deviation from the F32 reference (cos_min ≥ 0.95). If parity is worse
than that, there IS a structural bug — a wrong resize method, a missing
ggml_cont, a wrong weight mapping, a wrong activation function, etc.

During the Unlimited-OCR port, hours were wasted blaming "F16 quantization
noise cascading through 24 CLIP layers" when the actual bugs were:
(1) `ggml_cont()` before `flash_attn_ext`, (2) bilinear vs bicubic resize,
(3) missing pixel clamp, (4) wrong BPE token ID. The BF16→F16 conversion
was LOSSLESS (max_abs_diff=0.000000 for all SAM weights checked).

Rule: when crispembed-diff shows cos < 0.95, always bisect with the diff
harness. Never accept "quantization noise" as an explanation.

## Unlimited-OCR decoder: prefill attention output was scrambled (2026-06)

The MoE decoder produced pure garbage ("by", or open-ended text collapsing
to multilingual noise) even though SAM/CLIP vision parity was fine. Root
cause was in the LLM self-attention output reshape:

```cpp
attn = ggml_flash_attn_ext(g, Q, Kfull, Vfull, mask, scale, 0, 0); // [hd, nh, T]
attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));            // BUG → [hd, T, nh]
attn = ggml_reshape_2d(g, attn, D, T);
```

`flash_attn_ext` returns `[hd, nh, T]`; the llama.cpp idiom is to
`reshape_2d` that **directly** to `[D, T]`. The extra `permute(0,2,1,3)`
turns it into `[hd, T, nh]`, so the subsequent `reshape_2d` interleaves the
head and token axes — *unless T==1*, where the permute is a no-op. That is
why the bug was invisible during single-token decode (the persistent-decode
path) but corrupted every multi-token **prefill**: a scrambled attention
output feeds the residual stream → next layer's input → the whole KV cache
is built from corrupted hidden states. Fix: delete the permute, reshape the
flash output straight to `[D, T]` (see `build_llm_layer_attn` and
`build_persistent_decode_graph`). After the fix the pure-text decoder is
coherent and the OCR path reads the first text region correctly
(`<|det|>text [x1,y1,x2,y2]<|/det|>Hello World`).

Two related gotchas found alongside it:

- **Persistent-decode KV cache must be zero-initialized.** The PD graph
  pre-allocates `max_kv` cache slots but fills only `n_past`; flash_attn
  reads the unused slots *before* the (-inf) mask is applied, and on the
  shared scheduler they hold leftover garbage from the vision graphs.
  `NaN + (-inf) = NaN`, which poisons every logit (→ endless `<bos>`). The
  short text-only test didn't trip it because its buffers were clean. Zero
  the whole `t_k_cache`/`t_v_cache` on the first PD step. (The PD path still
  drifts from the rebuild path after the first decode token on vision-heavy
  prefills — a residual Metal flash_attn numerics issue with the padded KV
  layout — so the verified-correct per-step rebuild path is the default;
  PD is opt-in via `UOCR_PD=1`.)

- **Instruction tokenization.** `"\nFree OCR."` byte-level-BPEs to
  `[Ċ=201, Free=21431, ĠOCR=126041, .=16]` (verified against the model's
  own tokenizer.json). Feeding the leading newline `Ċ=201` makes this port
  emit EOS immediately — a sensitivity that points at a small residual
  mismatch in the image-block embeddings (`image_newline`/`view_separator`)
  vs HF. The no-newline form reproduces correct first-region OCR. The grid
  is row-major and correct (a transpose wrongly maps the bottom image line
  to the top).

### Diff-harness findings vs `unlimited-ocr-ref.gguf` (vision stages)

With the reference GGUF (`cstr/unlimited-ocr-crispembed-GGUF/unlimited-ocr-ref.gguf`,
stages `sam_patch_embed` … `vision_features`) and `UOCR_REF=`:

- **Use `cos_mean`, not `cos_min`.** The letterboxed-gray padding tokens are
  near-zero vectors whose cosine is pure noise, dragging `cos_min` to ~0 even
  when the tensor is fine. `crispembed_diff::Report` already computes
  `cos_mean`; the engine now prints it.
- **The SAM encoder is the real vision bug.** `cos_mean` degrades
  monotonically: `sam_patch_embed` 0.980 → `sam_output` 0.734 →
  `clip_output` 0.61 → `vision_features` 0.71. Injecting the reference's
  `sam_output` (`UOCR_INJECT_REF=1`) snaps every downstream stage to **0.998**
  — so CLIP, the fusion concat, the projector and `assemble_vision_features`
  are all correct; the error is entirely inside SAM (preprocessing input is
  ~0.98, then the ViT layers accumulate error). Individual content patch
  embeddings match the reference to the digit; the divergence is broad-but-
  small, consistent with the windowed/global attention + relative-position
  path, not a single catastrophic op. Not yet root-caused.
- **q4_k decode is the hard ceiling for multi-line OCR.** Even with the
  reference vision injected (cos 0.998), `flash_attn` forced to F32
  (`UOCR_FA_F32=1`), and `no_repeat_ngram` enabled
  (`UOCR_NO_REPEAT_NGRAM=3`), the q4_k decoder reads the first region
  perfectly then garbles the rest (it *does* surface "is a test" / the date,
  but mangled). So multi-line correctness is gated on the q4_k expert/
  attention weights, not on vision or sampling. q8_0 reads further on the
  text-only probe ("…the city of Paris") but its OCR path OOMs the 16 GB
  machine (>6 GB). Full multi-line OCR realistically needs q8_0/f16 weights
  (more RAM) plus the SAM fix. `no_repeat_ngram` only helps the decoder
  *advance* past the stuck-box loop; it cannot fix wrong-token generation.
