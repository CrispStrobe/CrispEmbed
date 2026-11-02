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

## Results (Apple M1, warm = best of repeats 2..3, one model run per lock)

Reference IDs are the committed original-Python IDs
(`tests/results/teleocr/hosted-parity.json`, 6 cases, cap 1024;
`tests/results/mineru25/native-parity.json`, 8 cases, cap 2048). Every run of
every arm is compared, not only the first. Cosine is against a native CPU F32
computation: for TeleOCR the CPU F16 path (it dequantizes the bf16-exact F16
weights and computes in F32); for MinerU the CPU F32 GGUF converted from the
pinned checkpoint (SHA-256 `ef6ea875…`, equal to the qualified F32 GGUF).
"Worst row" is the lowest per-token cosine over all prefill final-norm rows;
decode cosines cover the first 32 steps while IDs agree.

### TeleOCR

| Precision | Device | Exact | Warm s (6 cases) | Global hidden cos | Worst row | Prefill logits | Worst decode logits |
|---|---|---|---|---|---|---|---|
| F16 | Metal | 6/6 | 112.1 | 0.999996 | 0.998285 | 1.000000 | 0.999998 |
| Q8_0 | Metal | 6/6 | 97.1 | 0.999818 | 0.990959 | 0.999934 | 0.999649 |
| Q8_0 | CPU | 6/6 | 493.9* | 0.999822 | 0.990959 | 0.999934 | 0.999644 |
| Q4_K_M | Metal | 6/6 | 87.7 | 0.985714 | **0.901205** | 0.989039 | **0.936968** |
| Q4_K_M | CPU | 6/6 | 447.1* | 0.985715 | **0.901204** | 0.989022 | **0.936951** |
| F16 | CPU (reference) | 6/6 | 604.7* | 1 | 1 | 1 | 1 |

\* CPU timings were measured while the shared host ran other agents' CPU work
(load average up to 39); an earlier quieter CPU F16 run of `text` took 15.6 s
warm versus 4.5 s on Metal. On the three cases kept for the legacy Metal path
(flash attention on CPU, plain PREC_F32 matmuls) F16 took 39.9 s versus 16.9 s
with the fixes, Q8_0 49.3 s versus 15.8 s.

Q4_K_M misses the 0.94 bar on two figures — the system-prompt rows 0-8
(tiny-norm attention-sink tokens, worst 0.9012) and one formula decode step
(0.9370) — identically on CPU and Metal to five digits, so this is the
precision of the Q4_K/Q6_K decoder, not the runtime. Decoded IDs stay exact.

### MinerU2.5-Pro-2604

| Precision | Matmul kernels | Exact | Warm s (8 cases) | Global hidden cos | Worst row | Worst decode logits |
|---|---|---|---|---|---|---|
| F16 | half tiles (old default) | 8/8 | 99.2 | 0.999913 | **0.880011** | 0.999999 |
| F16 | F32 tiles (new default) | 8/8 | 95.2 | 0.999999 | 0.999578 | 1.000000 |
| Q8_0 (F16 vision) | half tiles | 8/8 | 77.0 | 0.999890 | **0.881305** | 0.999831 |
| Q8_0 (F16 vision) | F32 tiles | 8/8 | 85.1 | 0.999960 | 0.999342 | 0.999840 |
| "q4_k-dq8" (all Q8_0) | half tiles | 8/8 | 79.4 | 0.999099 | **0.725269** | 0.999594 |
| "q4_k-dq8" (all Q8_0) | F32 tiles | 8/8 | 78.3 | 0.999217 | **0.802009** | 0.999585 |
| F32 | CPU (reference) | 8/8 | 386.8* | 1 | 1 | 1 |

The half-tile drop (German prompt row 1961, 0.880) is runtime: the same
weights with F32 tiles give 0.9996, and the vision-merger worst row moves from
0.99096 to 0.99995. The Kaggle "Q4_K with Q8_0 decoder" recipe
(`q4_k --tensor-type 'l.*=q8_0'`) contains **no Q4_K tensor**: the
quantizer floors vision at Q8_0 and the override covers the whole decoder
(297 matrices, all Q8_0, embeddings included). Its weak rows come from the
Q8_0 vision tower (German merger worst row 0.978) and remain on both kernel
choices, so they are precision. Unlike CUDA, it was exact on layout-page on
Metal. Real Q4_K decoder candidates are listed below.

## Precision-policy re-score (docs/precision-acceptance.md)

Final output = the logits at every generated position. Each candidate is
teacher-forced on the committed original-Python IDs (diagnostic
`CRISPEMBED_QWEN2VL_FORCE_IDS`), so every position is conditioned on the same
prefix as the reference, and the minimum cosine over all positions of all
cases is compared with the bar (F32 > 0.99, F16 > 0.98, Q8_0 > 0.97,
Q4_K > 0.94). References: MinerU native CPU F32 (converted from the pinned
checkpoint); TeleOCR native CPU F16 (F32 compute on the bf16-exact weights).
Decoded agreement is a separate free-running run in the same process.
"Worst stage" is the lowest of the vision-merger rows, the prefill
final-norm rows and the first 32 decode final-norm states.

### TeleOCR (reference: CPU F16, F32 compute; 6 cases)

| Precision | Device | Final-output cosine (worst case) | Bar | Pass | Decoded exact | First divergence | Worst stage |
|---|---|---|---|---|---|---|---|
| f16 | gpu | 0.999883 (table) | > 0.98 | yes | 6/6 | - | code prefill_final_norm(row) 0.8952 |
| f16 | metal | 0.999997 (table) | > 0.98 | yes | 6/6 | - | code prefill_final_norm(row) 0.9983 |
| q4_k | cpu | 0.815384 (formula) | > 0.94 | no | 5/6 | code@364 | code prefill_final_norm(row) 0.5539 |
| q4_k | gpu | 0.873399 (formula) | > 0.94 | no | 5/6 | code@364 | synthetic prefill_final_norm(row) 0.5501 |
| q4_k | metal | 0.835997 (formula) | > 0.94 | no | 5/6 | code@364 | code prefill_final_norm(row) 0.5545 |
| q4_k-attn8 | cpu | 0.852627 (formula) | > 0.94 | no | 4/6 | table@35, code@364 | code prefill_final_norm(row) 0.5554 |
| q4_k-attn8 | gpu | 0.843844 (formula) | > 0.94 | no | 5/6 | code@364 | text prefill_final_norm(row) 0.5518 |
| q4_k-attn8 | metal | 0.866214 (formula) | > 0.94 | no | 5/6 | code@364 | code prefill_final_norm(row) 0.5561 |
| q4_k-m | cpu | 0.904747 (layout) | > 0.94 | no | 6/6 | - | code prefill_final_norm(row) 0.9012 |
| q4_k-m | gpu | 0.921535 (layout) | > 0.94 | no | 6/6 | - | code prefill_final_norm(row) 0.8765 |
| q4_k-m | metal | 0.904728 (layout) | > 0.94 | no | 6/6 | - | code prefill_final_norm(row) 0.9012 |
| q8_0 | cpu | 0.998452 (layout) | > 0.97 | yes | 6/6 | - | code prefill_final_norm(row) 0.9910 |
| q8_0 | gpu | 0.993890 (formula) | > 0.97 | yes | 6/6 | - | code prefill_final_norm(row) 0.8939 |
| q8_0 | metal | 0.998480 (layout) | > 0.97 | yes | 6/6 | - | code prefill_final_norm(row) 0.9910 |

TeleOCR F16 on CPU is the reference itself. Every Q4_K recipe (plain, ATTN8,
the published Q4_K_M) misses 0.94 on all three devices with near-identical
scores; the lowest positions are confident steps (top-2 margin 14-23 logits,
argmax unchanged) whose full-vocabulary logit tail moves. CPU/Metal/CUDA
numbers for one recipe agree within 0.02-0.04, so runtime is not the limit.

### MinerU2.5-Pro (reference: CPU F32; 8 cases)

| Precision | Device | Final-output cosine (worst case) | Bar | Pass | Decoded exact | First divergence | Worst stage |
|---|---|---|---|---|---|---|---|
| f16 | cpu | 1.000000 (japanese) | > 0.98 | yes | 8/8 | - | german prefill_final_norm(row) 1.0000 |
| f16 | gpu | 0.999946 (german) | > 0.98 | yes | 8/8 | - | german prefill_final_norm(row) 0.8647 |
| f16 | metal | 1.000000 (german) | > 0.98 | yes | 8/8 | - | german prefill_final_norm(row) 0.9996 |
| q4_k | cpu | 0.914481 (japanese) | > 0.94 | no | 6/8 | german@3, arabic@4 | german prefill_final_norm(row) 0.2888 |
| q4_k | gpu | 0.923131 (japanese) | > 0.94 | no | 6/8 | german@3, arabic@4 | german prefill_final_norm(row) 0.2818 |
| q4_k | metal | 0.924067 (japanese) | > 0.94 | no | 6/8 | german@3, arabic@4 | german prefill_final_norm(row) 0.7962 |
| q4_k-df16 | cpu | 0.999372 (japanese) | > 0.94 | yes | 7/8 | layout-page@180 | german prefill_final_norm(row) 0.2902 |
| q4_k-df16 | gpu | 0.998766 (japanese) | > 0.94 | yes | 7/8 | layout-page@180 | german prefill_final_norm(row) 0.2820 |
| q4_k-df16 | metal | 0.999787 (japanese) | > 0.94 | yes | 8/8 | - | german prefill_final_norm(row) 0.8028 |
| q4_k-dq8 | cpu | 0.998114 (german) | > 0.94 | yes | 8/8 | - | german prefill_final_norm(row) 0.2896 |
| q4_k-dq8 | gpu | 0.998809 (japanese) | > 0.94 | yes | 7/8 | layout-page@180 | german prefill_final_norm(row) 0.2803 |
| q4_k-dq8 | metal | 0.999585 (japanese) | > 0.94 | yes | 8/8 | - | german prefill_final_norm(row) 0.8020 |
| q4_k-vf16 | cpu | 0.925344 (japanese) | > 0.94 | no | 6/8 | german@3, arabic@4 | layout-rot90 prefill_final_norm(row) 0.9607 |
| q4_k-vf16 | gpu | 0.939804 (japanese) | > 0.94 | no | 5/8 | german@3, arabic@4, layout-page@38 | german prefill_final_norm(row) 0.8410 |
| q4_k-vf16 | metal | 0.930233 (japanese) | > 0.94 | no | 6/8 | german@3, arabic@4 | arabic prefill_final_norm(row) 0.9613 |
| q8_0 | cpu | 0.998803 (german) | > 0.97 | yes | 8/8 | - | layout-rot90 prefill_final_norm(row) 0.9984 |
| q8_0 | gpu | 0.999314 (japanese) | > 0.97 | yes | 8/8 | - | german prefill_final_norm(row) 0.8575 |
| q8_0 | metal | 0.999838 (german) | > 0.97 | yes | 8/8 | - | german prefill_final_norm(row) 0.9993 |

"q4_k-dq8" and "q4_k-df16" pass but contain no Q4_K tensor (the quantizer
floors vision at Q8_0 and the override keeps the whole decoder at Q8_0/F16).
Recipes with Q4_K decoder matrices ("q4_k", "q4_k-vf16") fail on every device
and flip German at token 3 and Arabic at token 4 everywhere. Layout-page
token 180 (CPU df16, CUDA dq8/df16) is a near-tie: the F32 reference's top-2
margin there is 0.024 logits (IDs 22 vs 23), with logit cosine 1.0. Worst
stage: the German prompt final-norm row falls to 0.28-0.29 for every Q8_0
vision recipe on CPU and CUDA (int8 / half activation handling) but stays at
0.80 on Metal with F32-activation kernels; CUDA F16/Q8_0 show 0.86 on the
same row (CUDA half-precision activations), Metal 0.9996.


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
