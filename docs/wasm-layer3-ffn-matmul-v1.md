# Layer3 FFN mul_mat same-input replay v1

The [v4 live-norm attribution](../tests/results/packaging/wasm-live-norm-v4-attribution.json)
found the first budget failure at layer3 `fc2_raw` (0.00331). It came after a
jump of about 300x at `fc1_raw` (5.73e-4), with all 208 same-input norm replays
bit-exact. This protocol separates kernel accumulation order from inherited
input error at those two `mul_mat` operations.

## Code paths the builds actually compile (a06, read from the sources and the production compile inventories)

| | native `libcrispembed` (x86-64, `-mavx2 -mfma`, `GGML_USE_LLAMAFILE`) | embed WASM (`-msimd128 -pthread`, no llamafile) |
|---|---|---|
| fc1 weight | IQ4_NL (384 is not a multiple of 256), activation Q8_0 | same |
| fc1 activation quantizer | x86 `quantize_row_q8_0` (AVX2, round-nearest-even) | `quantize_row_q8_0_wasm_nearest` (held `CRISPEMBED_WASM_Q8_0_ROUND=1`) |
| fc1 kernel | llamafile `tinyBLAS_Q0_AVX<block_iq4_nl, block_q8_0>`: per output one 8-lane F32 accumulator; lane L = exact int sum of elements 4L..4L+3 of each 32-block; `fma(d_w*d_a, lane, acc)` over the 12 blocks; `hsum` = ((x0+x4)+(x2+x6))+((x1+x5)+(x3+x7)) | `ggml_vec_dot_iq4_nl_q8_0_generic` (no WASM kernel; `arch-fallback.h`): per block `sumf += d*(int sum)` in scalar F32, no FMA |
| fc2 weight | IQ4_XS, activation Q8_K | same |
| fc2 activation quantizer | `quantize_row_q8_K_ref` (x86 has no SIMD version) | WASM SIMD `quantize_row_q8_K` (same formula; can differ only when the positive and negative extremes tie) |
| fc2 kernel | AVX2 `ggml_vec_dot_iq4_xs_q8_K` (llamafile has no IQ4_XS): per 256-superblock exact int32 lanes including integer sub-block scales, one FMA per superblock per lane, `hsum_float_8` | `ggml_vec_dot_iq4_xs_q8_K_generic`: 8 scalar F32 multiply-adds per superblock (48 per output), each rounded |
| intermediates | FP16 scales widened exactly to F32; int32 lanes stay below 2^24 so the conversion is exact | same |

Weights load into a plain CPU buffer, not CPU_REPACK. The probe checks the
buffer name.

## Protocol

[Workflow](../.github/workflows/wasm-layer3-ffn-matmul-v1.yml),
[caller](../tools/ci-heavy/wasm_layer3_ffn_matmul_v1.py),
[analysis](../tools/wasm-layer3-ffn-matmul/analysis.py),
[probe](../tests/wasm-layer3-ffn-matmul/probe.cpp).

1. Build the probe natively (llamafile ON) and for WASM (`-pthread`, run under
   node). Save the full compile inventories first. Then require every ggml
   kernel file's arithmetic profile (`-m*`, `-O*`, fp/fast-math, `-pthread`,
   `-std`, defines; linkage-only defines and `-fPIC` ignored) to equal the
   pinned production full inventories (run 40852835039). The probe must use
   `-ffp-contract=off`.
2. Weight-free controls before model access: authored random IQ4_NL [384x64]
   and IQ4_XS [1536x64] weights through the real ggml graph `mul_mat`. Native
   must be bit-exact with the probe's native-order emulation and WASM with its
   generic-order emulation. Each must differ from the other order
   (sensitivity), and weights must sit in a plain CPU buffer.
3. Only then fetch the pinned v4 captures (run 41432135298) and the frozen
   model. fc1 runs on native, wasm0 and wasm1 `attention_output`; fc2 on
   native and wasm `gelu`; each on both builds.
4. Controls: both sides reproduce their own captured `fc1_raw`/`fc2_raw`
   bit-exactly, and the emulations reproduce production on every model case.
5. Report per op: four-way metrics against `native_kernel(native_input)`; the
   kernel-versus-input decomposition; Q8 activation-byte equality; and the
   expected effect of the native-order candidate executed by the WASM build.
   Full 12-row metrics and 0.002/0.99999 budgets apply. Numerical negatives
   exit 1 after the complete report.

The candidate is not enabled anywhere; no gate or default changes.

## Attempts
