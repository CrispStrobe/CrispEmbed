# TeleOCR and MinerU2.5-Pro on ggml-Metal (Apple M1)

Branch `feat/metal-qwen2vl` (from `feat/teleocr`). Host: MacBook Apple M1, 8 GPU
cores, 16 GiB unified memory, shared with other agents; every model run held
the shared M1 GPU lock, so no two heavy jobs overlapped. Receipts:
[TeleOCR](../tests/results/teleocr/metal-m1/) and
[MinerU](../tests/results/mineru25/metal-m1/).

## Defects found and fixed

1. **Every flash attention ran on the CPU.** All four `qwen2vl_ocr` attention
   sites tag `flash_attn_ext` with `GGML_PREC_F32`. The bundled ggml-metal
   refuses PREC_F32 flash attention in `supports_op` (CrispASR patch #83), so
   the scheduler moved attention to the CPU backend in every vision block,
   every prefill layer and every decode step, copying the K/V views across
   (`GGML_SCHED_DEBUG`: one CPU split per layer per step). New gate
   `CRISPEMBED_QWEN2VL_METAL_FA` (default on for Metal; `=0` restores the old
   path) leaves the precision at default on Metal, which selects the native
   `kernel_flash_attn_ext[_vec]_f32` kernels: F32 K/V and F32 accumulators,
   Q rounded to F16. After the fix the TeleOCR graph has zero CPU splits.
2. **Metal detection never matched.** The first version compared the registry
   name with `"Metal"`; ggml-metal registers as `"MTL"` (`GGML_METAL_NAME`).
   `src/core/ggml_cpu_backend.h` (`is_metal`, DL build path) has the same
   `"Metal"` comparison and should be checked by its owners.
3. **PREC_F32 matmuls on Metal took slow, activation-rounding routes.** In the
   bundled ggml-metal, PREC_F32 means "mirror ggml-cpu's quantized vec_dot":
   Q8_0/Q4_K weights pre-quantize the F32 activations to Q8_0/Q8_K, and F16
   weights go through `mul_mv_ext` at every batch size. TeleOCR's reference
   arithmetic is F32 activations against dequantized weights (its CPU branch
   uses `quant_f32_matmul` / an F32 cast), so the Q8/Q8_K pre-quantization is
   also a numerical mismatch, and it is 10-30x slower for batches
   ([kernel timings](../tests/results/teleocr/metal-m1/kernel-bench.txt)).
   ggml patch (`CrispStrobe/ggml@crispembed/metal-mul-mat-prec-hp`): a MUL_MAT
   with PREC_F32 plus the `GGML_METAL_MUL_MAT_PREC_F32_HP` marker in
   `op_params[1]` keeps F32 activations: `mul_mv` for one row, `mul_mv_ext`
   for 2-8 rows, and the existing simdgroup_float8x8 `kernel_mul_mm_*_hp` for
   larger batches. Untagged and plain PREC_F32 ops are unchanged. Gates:
   `CRISPEMBED_TELEOCR_METAL_F32_MATMUL` (default on for TeleOCR on Metal;
   `=0` restores plain PREC_F32) and `CRISPEMBED_MINERU25_PRO_METAL_F32_MATMUL`
   (default off; MinerU keeps ggml-metal's default half-tile kernels).

Optional, default-off gates added for measurement:
`CRISPEMBED_TELEOCR_REUSE_DECODE_GRAPH` (persistent 256-token-bucket decode
graph, as MinerU's existing gate), `CRISPEMBED_QWEN2VL_METAL_SPLIT_K` (splits
non-32-aligned batch reductions such as TeleOCR's 3420-wide vision FFN, which
otherwise hit ggml-metal's bounds-checked mul_mm at about a quarter of the
aligned kernel's throughput), and `CRISPEMBED_QWEN2VL_STAGE_DUMP=<dir>` (raw
F32 vision-merger, prefill final-norm, prefill last logits and first 32
decode-step logits/final-norm of the first recognition, for cosine A/B).

## Metal precision traps checked

| Trap | Where | Status |
|---|---|---|
| `kernel_mul_mm` rounds F32 activations to F16 tiles | every batch matmul without PREC_F32 | TeleOCR uses the F32-tile `_hp` kernels; MinerU default keeps half tiles (CPU ggml F16/Q8 also rounds activations) |
| PREC_F32 Q8_0/Q4_K activation pre-quantization | TeleOCR Q8_0/Q4_K_M before the fix | avoided by the HP marker |
| Flash attention Q in F16 | all attention on Metal | measured by decoded IDs and final-state cosine |
| PREC_F32 flash attention refused, CPU fallback | all attention before the fix | fixed (gate above) |
| GELU table | MinerU vision | unchanged: exact F32 `quick_gelu_f32` is device-independent graph arithmetic |
| Softmax/RMSNorm | Metal kernels accumulate in F32 | no change needed |

## Reproduce

```sh
cmake -B build-metal -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-metal --target test-teleocr-run test-mineru25-run test-bench-qwen2vl-metal-ops
python tools/qwen2vl_device_ab.py --model teleocr --gguf teleocr-f16.gguf --precision f16 \
  --device metal --build build-metal --out out/teleocr --dump dumps/teleocr/f16-metal
python tools/qwen2vl_stage_cosine.py --ref dumps/teleocr/f16-cpu --cand dumps/teleocr/f16-metal \
  --ref-receipt out/teleocr/f16-cpu.json --cand-receipt out/teleocr/f16-metal.json --width 1024 \
  --out out/teleocr/f16-metal.cosine.json
python tools/qwen2vl_device_summary.py out/teleocr --cosine-dir out/teleocr --out out/teleocr/summary.json
```
