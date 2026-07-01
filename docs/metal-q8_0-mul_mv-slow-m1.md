# Metal: Q8_0 `mul_mv` (single-token decode) is anomalously slow on M1

**Status:** investigation / upstream candidate. Reproduced on Apple M1 (base,
8-core GPU, ggml-metal, embedded shaders). Not yet filed upstream.

## Summary

For **single-token (T=1) autoregressive decode** — i.e. matrix–**vector**
products (`GGML_OP_MUL_MAT` with `ne1 == 1`, dispatched as `mul_mv`) — Q8_0
weights are **slower than both F16 and Q4_K** on this M1. That is backwards:
Q8_0 is half the bytes of F16 and simpler to dequantize than a k-quant, so it
should be the *fastest* correct option, not the slowest.

Measured on GOT-OCR2 (Qwen2-0.5B decoder, 24 layers, 1024d), identical graph,
only the decoder weight type differs, per-token decode with KV cache active:

| Decoder weights | Decode / token | Bytes/token (decoder) |
|-----------------|----------------|-----------------------|
| Q4_K | **~20 ms** | ~0.28 GB |
| F16  | ~38 ms | ~0.55 GB |
| Q8_0 | ~42 ms | ~0.29 GB |

If decode were purely weight-bandwidth-bound we'd expect Q8_0 ≈ Q4_K (they move
almost the same number of bytes) and both well under F16. Instead Q8_0 is ~2×
slower than Q4_K and even edges past F16 — so the Q8_0 `mul_mv` path is spending
time on something other than reading weights.

Reproduce:

```bash
CRISPEMBED_GOT_OCR_BENCH=1 GOT_OCR_STEP_PROFILE=1 \
  crispembed -m got-ocr2-q8_0.gguf --ocr page.png
# vs got-ocr2-q4_k.gguf / got-ocr2-f16.gguf
```

`GOT_OCR_STEP_PROFILE` confirms the entire delta is inside
`ggml_backend_sched_graph_compute` (build ≈ 0.3 ms, alloc ≈ 2 ms, readback ≈
0.1 ms are constant across quants) — it is the compute kernels, specifically the
per-layer Q8_0 × F32 mat-vecs (q/k/v/o + gate/up/down), that are slow.

## Likely cause

On Metal, the fast tiled `kernel_mul_mm` (matrix–matrix) path is used when the
activation side has enough columns; for T=1 decode the runtime falls back to the
mat-**vector** family (`kernel_mul_mv_*` / the `_ext_` variants). The k-quant
mat-vec kernels (Q4_K etc.) are heavily tuned; the Q8_0 mat-vec path on
pre-M5/A-series-without-tensor-API hardware appears to take a less-optimized
route (per-thread `float` accumulation over `dot(float4,float4)` rather than the
integer-dot block form the CPU uses), which on M1 is slower than the k-quant
kernel despite Q8_0 being "simpler".

This is adjacent to the known Metal Q8_0 issue captured in CrispASR's
`tools/upstream-prs/09-metal-q8_0-bit-match.md` ("Q8_0 × F32 bit-match mul_mat
under `GGML_PREC_F32`"). That PR is about *bit-exactness* (the Q8_0 mat-vec ext
path diverges from CPU by ~1e-3), but it documents the same underlying fact: the
Q8_0 × F32 Metal mat-vec path is a second-class citizen relative to Q4_K.

**Note:** this slowness is orthogonal to correctness. Q8_0 output is correct
(cos ≥ 0.99996 vs f32, see [`got-ocr2.md`](got-ocr2.md)); it is purely a speed
regression for the single-token-decode shape.

## Practical consequence

For small autoregressive decoders on M1 (and likely other pre-tensor-API Apple
GPUs), **prefer Q4_K over Q8_0** — it is smaller *and* faster per token with no
measurable accuracy cost. GOT-OCR2 therefore ships Q4_K as the default. This
guidance may not hold on M5/A19+ (tensor API / fused matmul kernels) or on
CUDA/Vulkan, where the Q8_0 mat-vec kernels are competitive — re-measure per
backend before generalizing.

## Follow-up (not yet done)

- Confirm with `test-backend-ops` that the isolated `mul_mv` op for Q8_0 × F32
  at `ne1==1` is slower than Q4_K × F32 on M1 (removes GOT-OCR2 as a variable).
- If confirmed, the fix is a tuned `kernel_mul_mv_q8_0_f32` (integer block-dot,
  simdgroup reduction) mirroring the k-quant mat-vec kernels; file against
  ggml-org/ggml (`src/ggml-metal/`), same path as the merged conv-transpose PR
  #1477.
