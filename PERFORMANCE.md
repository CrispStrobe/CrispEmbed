# CrispEmbed Performance

## sched alloc-once/compute-many FIXED in the fork — the O6 replay crash was a restore-on-success design assumption, not a Metal bug (2026-08-06)

Fable-task 5. The `UOCR_PD=1 UOCR_PD_REPLAY=1` gen=2 SIGSEGV was localized
with a new fork diagnostic (`CRISPASR_METAL_PROFILE=3`: per-op profile plus
a pre-encode node/src-buffer trace): the second compute of the stored splits
executed a Metal node whose src still pointed at the CPU split's output —
`ggml_metal_buffer_get_id` then dereferences a CPU buffer's context as a
Metal buffer (poisoned AGXBuffer). Root cause in the fork's sched: split
rewires user-graph srcs to its input copies and RESTORES them on every exit
of compute_splits — correct for the universal reset-per-step pattern,
fatal for alloc-once/compute-many (the replay ran with original
cross-backend srcs).

Fork fix `890278a8` (`sync/upstream-v0.17`): the mutation log records the
rewired src, compute_splits re-applies at entry, and disposal is
state-flagged. Two wrong intermediate cuts recorded honestly, each caught
by a regression the other case demanded: restore-at-reset writes into
recycled graph memory when engines rebuild graphs (vision CPY asserted with
a foreign src), while a pure clear leaves a build-then-run engine's graph
mutated across its alloc-to-first-compute reset (ppocr decoded 0 regions).
The flag distinguishes applied vs restored logs; only applied logs are
written back at disposal.

Verified (M1): the replay repro runs end-to-end (was SIGSEGV at gen=2,
4/4); ppocr medium page byte-identical (`a3a5f938`), layout_detect
byte-identical (`afde4fd4`), unlimited-ocr default unchanged, 4/4
model-free spot checks. Honest PD verdict UNCHANGED: replay ≈ re-init
(181 vs 166 ms/step, single noisy samples) and the PD decode still
diverges — PD stays opt-in research; what this fix buys is a SAFE sched
contract for every future persistent-graph engine (deepseek-style
reset-per-step no longer the only crash-free pattern) and it unblocks the
R8 ICB-replay lane, which needs exactly alloc-once/compute-many.

## layout O2b: six value projections share one input — batched GPU graph is 3.5x on the stage, ships OPT-IN on a 0.500-score threshold flip (Apple M1, 2026-08-06)

Fable-task 6. Re-profiled Phase 2 on current main FIRST (flat-im2col moved
every layout number): Phase 1 is now 561-701 ms (was ~1.4 s), Phase 2
545-588 ms with **value-proj 178-211 ms as the new #1 stage (~35%)** —
larger in share than the O2-era ~101 ms estimate. The CPU lane has no
headroom: `cpu_linear` is already AXPY-vectorized + threaded and runs at its
memory-bound ceiling at `-t 4`.

Structural fact the estimate missed: `memory` (the encoder output) is
CONSTANT across the six decoder layers — all six value projections consume
the same input, so one single-backend gallocr graph computes them together
(one 11-MB upload, one shared transpose-cont, six mul_mats + bias, one
readback per layer). Measured: value-proj **178-211 -> 54-55 ms (3.5x)**,
Phase 2 545-588 -> **366-385 ms (~35%)**, page call ~1.21 -> ~1.14 s.

**Ships opt-in (`CRISPEMBED_LAYOUT_VALPROJ_GPU=1`), default OFF:** the GPU
contraction order is not byte-identical. 6-fixture sweep: 3 pages
byte-identical; german_official / receipt_example show +-0.001 score /
+-0.1 px jitter; **commons_test flips a borderline region at score exactly
0.500 across the confidence threshold (13 -> 14 regions)** — the
LAYOUT_CONV_F16 drift class, so the default stays CPU per the A/B rule
(default arm verified byte-identical to baseline). Follow-ups that could
earn the flip: region-level quality gate across the full fixture set, score
hysteresis at the threshold, and the CUDA arm (different contraction there
anyway; det/rec already flipped on CUDA).

## CUDA-rec 0-results: quantized graph residents uploaded F32 bytes as q8_0 blocks — fixed, byte-exact on P100; det flips to GPU-by-default on CUDA (O11) (2026-08-06)

Fable-task 1. The conv-ab v2 capture's two suspects (logits readback layout,
CTC decode) were both wrong — and the bug was never CUDA-specific.
`pp_graph_resident`'s native-quant path (keep a quantized head weight in its
native type on GPU backends, `pp_graph_linear`) fell into the F32 upload
branch: it dequantized the source to floats and copied those raw bit
patterns into a q8_0-typed resident. The f16 scale field decoded from float
mantissa bytes lands on Inf/NaN, all 18,710 logits go NaN, and
`max_element`'s NaN-poisoned compare returns index 0 — the CTC blank — so
every crop decodes to the empty string: `boxes=38 results=0`, rc=0,
activations sane through stage4 (the exact capture signature).

**Reproduced on M1 Metal** with the q8-head artifact (`results=0`): Metal
had only ever been validated on the f16 artifact, which is why the bug wore
a CUDA costume. Fix: same-type residents copy the raw source bytes (staged
through the host — the source may sit in a backend buffer); cross-type keeps
the F16/F32 conversion paths, asserted exhaustive.

**Proof (kernel `chr1s4/crispembed-cuda-rec-fix` v1, Tesla P100):** q8-head
CUDA fused graph vs `NO_GRAPH` scalar CPU reference — strip 12=12 regions,
page 38=38, decoded text **byte-identical** (similarity 1.0000) on both
fixtures; f16 control healthy. M1 gates: q8 strip 0→12 results conf 0.94,
q8 graph == scalar reference byte-for-byte, f16 page byte-identical to
baseline (`a3a5f938`), 12/12 model-free CI checks.

**O11 (unblocked, same branch):** det residency is per-backend-kind — a
~29 ms device-NAME probe (no backend init) defaults the det graph to GPU
only when a CUDA device is present (18x measured: 596-614 ms vs 11.0 s
medium det, conv-ab v1+v2); Metal keeps the CPU default (9x slower there).
`CRISPEMBED_PPOCRV6_DET_GPU=0|1` overrides; `FORCE_CPU`/`DET_GPU_LOAD`
retain their meanings. M1: default still CPU, page output byte-identical;
`=1` runs the det graph on MTL0, output STILL byte-identical
(backend-portable graph). **CUDA auto-engagement proven (kernel v2, P100):**
page detect 9516 ms (v1, CPU det) → **595 ms (16x)**, strip 2315 → 385 ms,
boxes unchanged (38/12), decoded text still byte-equal to the scalar
reference in every arm; page total 12.7 → 3.8 s. Both rec-fix proofs PASS
identically with O11 active.

## OCR runtime residency survey — which engine computes where, and why (2026-08-05)

Code-verified sweep at `9f731fb5` of every OCR-lane engine's backend selection
and `ggml_backend_sched` composition. **The backend/residency column below was
read out of the source, not carried from the audit tables further down this
file** — those are stale in both directions (see "Corrections" at the end of
this section). Timing figures are cited from their own dated sections.

### The distinction that matters: loading backend != computing backend

Nearly every engine calls `crispasr_init_gpu_backend()`
(`src/core/gpu_backend_pref.h`) with a `*_FORCE_CPU` escape, so grepping for
that call tells you almost nothing about where the math runs. Three patterns
exist:

1. **GPU sched** — `ggml_backend_sched_new({gpu, cpu_fallback}, …, 2, …)`.
   Real GPU compute; the CPU entry is only the ggml-mandated last-backend
   fallback.
2. **GPU load, CPU compute** — weights pulled through a GPU backend, then a
   *separate* `enc_sched` built over a lone `ggml_backend_cpu_init()`. The GPU
   handle is frequently freed immediately after load. Reading the load site
   alone misreads these as GPU engines.
3. **CPU everywhere** — deliberate, because Metal lost the measured A/B, or
   because the engine has no ggml graph at all.

### Detector + recognizer lanes (the document baseline)

| Engine | Compute default | Why | Optimizations present |
|---|---|---|---|
| DBNet `ocr_detect` | **CPU** (GPU only under `OCR_DETECT_USE_GPU=1`) | Metal `conv_2d`/`conv_transpose_2d` measured slower than the SIMD CPU path at these resolutions; recorded in `gpu_backend_pref.h` | ggml graph (ResNet+FPN), persistent `gallocr`, scanline box scoring, convex hull + rotating calipers, degenerate-component fallback |
| TrOCR `math_ocr` (rec half of `dbnet_trocr`) | **GPU sched** | Transformer, matmul-bound | **Persistent single decode graph (~4x)** — one of only three engines that reuse a cgraph; beam search; dequant cache; embeddings pre-cached before the decode loop |
| PP-OCRv6 det | **CPU** (`CRISPEMBED_PPOCRV6_DET_GPU_LOAD=1` for GPU) | Same conv verdict as DBNet | ggml graph is the default (`…_DET_SCALAR=1` restores scalar). Medium tier 6.9 s -> 1.0 s and 41.4 s -> 8.7 s vs the scalar detector |
| PP-OCRv6 rec | **GPU sched** (shared backend) | matmul/CTC-bound | **Shape-keyed graph cache** (rebuild only on width/batch change) + **batch-fused multi-crop graph** (`CRISPEMBED_PPOCRV6_BATCH_GRAPH`, on by default; `…_BATCH_GRAPH_CPU_ONLY` restores the old CPU-only restriction) |
| EasyOCR CRAFT + CRNN | **GPU sched** both | — | **Graph built once in `_init` and reused** (`ggml_gallocr_alloc_graph` at init) — the static input shape makes this free |
| Surya det | **GPU sched** | — | Hybrid *by design*: stages 0-2 ggml graph, stage-3 LiteMLA CPU-scalar (runs at 38x38, cheap); BN pre-folded into conv; `nth_element` thresholds |
| PARSeq | **GPU sched** | — | 12-layer ViT graph with flash-attn, dequant cache, cross-attention K/V precompute |
| Tesseract-LSTM | **CPU only** — zero `ggml_build_forward_expand` in the file | Hand-written LSTM, no graph. `CRISPEMBED_TESSERACT_GPU_LOAD=1` only pulls weights and then frees the backend; T18 measured Metal init at ~85% of a one-shot CLI | All weights dequantized at load (zero runtime dequant), SIMD `core_cpu::dot_product` gates, gated int8 recurrent-kernel cache, ~30 opt-in gates (recode/DAWG/pageseg). Parallelism lives in the orchestrator: `CRISPEMBED_TESSERACT_WORKERS` 1 -> 690 ms, 4 -> 300 ms, 8 -> 292 ms |
| `layout_detect` (RT-DETR) | **GPU sched** | — | Backbone+FPN+AIFI in one graph with flash-attn, persistent gallocr, `partial_sort` top-K |

### VLM OCR engines

| Engine | Compute | KV cache | Notes |
|---|---|---|---|
| internvl2, glm, got | GPU sched | F16 ggml, zero-copy | flash-attn; internvl2 caches the *vision* graph (`vis_graph_cached`) |
| qwen2vl / qwen3vl / olmocr | GPU sched | F16 device-resident | fused QKV, precomputed 2D RoPE; decode graph **rebuilt per step**; manual Q@K+softmax+V, no flash |
| deepseek_ocr2 (MoE) | GPU sched | **F32** default; `DS2_KV_F16=1` opt-in (memory win, not quality-neutral on CPU — see the G6 row) | 12 per-layer graphs per token; `DS2_FAST_DECODE=1` persistent graph **measured no win** (build+alloc was 1-6% of decode) and stays opt-in. `DS_LLM_FLASH` opt-in, slower on CPU |
| unlimited_ocr (SAM+CLIP+MoE) | GPU sched | — | ~40 `UOCR_*` gates. `UOCR_PD=1` persistent-decode path **segfaults at gen=2** (pre-existing, opt-in, default unaffected, unowned) |
| smoldocling | **split**: SigLIP vision on GPU, connector + LLM + KV + LM-head on CPU | F16 | G1/F4: vision 2.9-4.6x, totals 2.1-2.25x |
| granite_vision | **split**, same shape | F16 | — |
| **lightonocr** | **CPU only** — `ggml_backend_cpu_init()` hardcoded, no env escape | F16 persistent | Has flash-attn and a monolithic vision graph, all of it on CPU. 31.6 s cold on M1 |
| **pix2struct** | **CPU only** | present | ggml encoder graph on a CPU-only `enc_sched` |

### Math / formula / music OCR

| Engine | Compute | Note |
|---|---|---|
| math_ocr, ppformulanet_l, smt_ocr, transcoda, tromr | GPU sched | ppformulanet_l has batched windows + precomputed RPE, but its 8-layer D=512 decoder is still scalar |
| **bttr, hmer, posformer, mixtex, flova, ppformulanet** | **CPU** | Pattern 2's worst case: the encoders *were* ported to ggml graphs, but `enc_sched` is built over a single `ggml_backend_cpu_init()`. The "prefer GPU backend" comments above those load sites are stale — the code below them calls `ggml_backend_cpu_init()`. mixtex's Swin window attention is additionally still scalar |

### SR / denoise (feeds the OCR chains)

The family computes on a CPU-only `enc_sched`. `dat/hat/swinir` use `init_best`
**only to load**, then copy dequantized weights into a CPU-resident context;
`esrgan/safmn/restormer/instructir` skip even that copy. This is already
recorded as reprioritized-down: there is no GPU sibling to match, so SR-on-GPU
is unsolved research (Metal `ggml_conv_2d` + a GPU-resident weight/graph path),
not a residency toggle. `nafnet`, `safmn` and `pplcnet_orientation` do build
two-backend scheds and will use the GPU.

### Corrections to the audit tables below

- `bttr`/`posformer`/`hmer`/`flova`/`mixtex`/`ppformulanet` are listed in the
  2026-07-11 re-verification as "DenseNet/HGNetv2 -> ggml graphs (default)".
  True, and incomplete: those graphs run on a **CPU-only sched**, so the port
  bought SIMD, not GPU dispatch.
- `lightonocr` appears in the VLM maturity table with "GPU: Yes". It is
  **CPU-only** at HEAD (`lightonocr.cpp:239`), and unlike its siblings has no
  `*_FORCE_CPU`-style gate, so it cannot even be A/B'd without a code change.
- The P3 "`--gpu-backend` ignored (`crispembed.cpp:81` calls `init_best()`
  directly)" gap is **closed** — `crispembed.cpp:101` routes through
  `crispasr_init_gpu_backend()`.
- "0 runtimes reuse the built cgraph" is no longer literally true: math_ocr
  (persistent decode graph), easyocr (static-shape init-time graph) and
  ppocrv6 rec (shape-keyed cache) all do. The claim holds for the VLM decode
  step.

### Where more optimization would pay, ranked by strength of evidence

1. **Tesseract recognizer batching + weight/graph reuse.** The only gap with
   hard numbers on both ends: recognition is 260-354 ms of a ~310 ms stage on
   `scan_strip`, and **38.34 s of 38.69 s** on the German Fraktur page against
   official Tesseract's 9.34 s — an explicit speed *and* quality blocker.
   Detector and crop are 3-4 ms and 102/250 ms. `…_REUSE_SCRATCH` exists but
   its variance (279 vs 282 ms in one pair, 329-338 vs ~300 in others) is too
   wide to claim a win; it needs a warm/paired protocol before anything else.
2. **`layout_detect` deformable cross-attention** — still a 6-nested-loop CPU
   bilinear grid-sample, instrumented as the dominant Phase-2 decoder cost
   (`_deform_ms`, `layout_detect.cpp:1430`). Last surviving June P0.
3. **The CPU-sched formula encoders** (bttr, hmer, posformer, mixtex, flova,
   ppformulanet, pix2struct) — the expensive half of the port is done; they are
   one `sched_new` argument from GPU dispatch. Best work-to-payoff ratio here,
   but the DBNet/PP-OCRv6-det verdict warns that conv-heavy graphs can lose on
   Metal, so each needs its own A/B.
4. **lightonocr's hardcoded CPU backend** — a flash-attn, monolithic-graph
   engine spending 31.6 s cold on CPU with no gate to A/B it.
5. **Decode-step graph caching — but measure first.** Still the nominal #1
   unrealized lever, yet the one time it was built (deepseek T14) it won
   nothing because build+alloc was 1-6% of decode. qwen2vl/granite/smoldocling
   are the remaining candidates; profile the overhead fraction before porting.
   Also blocked on WebGPU (traps `unreachable`), so it needs per-backend
   gating.
6. **`conv2d_cpu` per-patch gather -> true im2col+GEMM, and multithread it.**
   `core/cpu_ops.h:577`, patch buffer at `:609` — one patch at a time into a
   `thread_local` buffer, with a SIMD dot per output channel. This is
   the shared floor under every CPU-resident conv engine — the whole SR family,
   DBNet, PP-OCRv6 det, the DenseNet encoders. Single highest-leverage core
   change.
7. **`scunet_denoise`** — the only SR engine with no `DequantCache` (18 other
   `.cpp` files have one; `grep DequantCache src/scunet_denoise.cpp` is empty).
   Its Swin blocks are still scalar, though no longer serial: WMSA is now
   window-parallel across `n_threads` (`scunet_denoise.cpp:37,269,538`), which
   the stale audit row below predates.
8. **ggml-metal ICB (indirect command buffer) replay** — Metal decode is
   per-op-dispatch bound; CUDA already has graph capture. Highest ceiling,
   highest cost, upstream-shaped work.

## PP-OCR rec Metal batch profile: im2col was 70% of the graph — flat-dispatch kernel 2.3x recognize, 1.6x layout_detect, byte-identical (Apple M1, 2026-08-06)

Fable-task 2 of the OCR-optimization handover. Fixture: `scan_page_pd.png`
606x1000 → 38 boxes, medium det + medium rec f16, Metal, `-t 4`, branch
`perf/ppocr-rec-profile`. New permanent instrumentation this session: the
`[ppocrv6-graph-bench]` line now splits `graph_ms` / `stage_ms` (input
staging) / `alloc_ms` / `readback_ms` / `readback_mb`, and
`CRISPEMBED_PPOCRV6_GRAPH_STOP` gained `stem|stage1|stage2|stage3` stops
beside the existing `backbone|decoder` (profiling-only; decode is garbage on
stop arms).

**Every O13b follow-up candidate was falsified by measurement:**

- **Readback is ~1%**, not a lever: the 60-69 MB logits of a batch-8 width
  group read back in 25-30 ms of a 2.6-3.1 s graph. On-device top-k/argmax
  is pointless on M1 (unified memory).
- **The 18,710-class head is ~1-2%**: `GRAPH_STOP=decoder` (everything but
  the head linear) ≈ the full graph within noise (2574/2765/1526 vs
  2624/2814/1540 ms). "The CTC head dominates" was an inference from the
  output shape, not a measurement.
- **Batch-size tuning wins nothing**: a single width-960 crop's prefix graph
  is ~334 ms; the batch-8 group is ~3104 ms — perfectly linear, no batching
  economy to tune.
- **`CRISPEMBED_PPOCRV6_CONV_DIRECT=1`** (native Metal `ggml_conv_2d_direct`,
  previously unmeasured): recognize 12.5-16 s → **85-88 s, a 6-7x
  REGRESSION**, 3/3 interleaved pairs, output byte-identical. Stays gated.

**Stage attribution** (stop-arm deltas, width-896 batch-8 group): stem ~97 ms,
stage1 ~90, **stage2 ~625, stage3 ~1530, stage4 ~550**, SVTR decoder + head ~0
within noise — the PPLCNetV4 backbone convs own ~95%.

**Per-op attribution** (fork per-op profiler `CRISPASR_METAL_PROFILE=2`,
serialized, batch-8 width-896 graph, 737 nodes): **IM2COL 70-71%**
(1.8-2.0 s, 62 nodes, ~30 ms avg), MUL_MAT 17.5%, CONT 8%. The heavy nodes
are the `ggml_conv_2d_dw` lowerings (IC==1, N=C*batch=4096): the standard
Metal im2col kernel loops each thread N/ntg0 times with per-plane strides on
both src and dst — ~0.4 GB/s on a ~50 MB F16 materialization. The 1x1 convs'
im2cols (8-thread threadgroups) were individually cheap.

**Fix (ggml fork `89a2039d`, branch `sync/upstream-v0.17`):**
`kernel_im2col_flat` — one thread per dst element from a
`(ceil(OW*CHW/256), OH, N)` grid; contiguous dst writes, 2D-local dw reads,
32-bit row-local index math only. Two failed cuts recorded honestly: a fully
flat int64-divmod index was SLOWER than legacy (int64 division is emulated on
Apple GPUs and cost more than the copy); extending naively without the grid
change regressed 25%. Selection predicate (`N*KH*KW < 128 || IC == 1`) is
shared between pipeline getter and encoder; **`CRISPASR_METAL_IM2COL_FLAT=0`
restores the legacy kernel** (bisection lever). This re-derives the pre-v0.17
`CRISPASR_METAL_IM2COL_OCC` variant that the sync dropped, as its drop note
prescribed.

**Verdict (interleaved same-binary pairs, 3x, all outputs byte-identical):**

| arm | recognize (3 pairs) | page total |
|---|---|---|
| legacy | 14022 / 13431 / 13685 ms | 16.3-16.8 s |
| flat | **5843 / 5994 / 6201 ms (2.3x)** | **8.6-9.3 s (1.8x)** |

Fused batch-8 width-group graphs: 2.6-3.1 s → 0.7-1.2 s (~2.6x). Side win —
**layout_detect** (RT-DETR, N=1 convs also hit the tiny-threadgroup class):
1556/1624 ms → **951/991 ms (1.6x)**, regions byte-identical, which revises
O1's "Phase 1 is steady-state Metal compute" wall downward. Quality gates:
18-fixture ppocr-lane sweep (fox, scan_page, scan_strip + 15 CC0) byte-identical
flat-vs-legacy, 13/13 model-free CI checks, `test-backend-smoke` correct=1.
Timing caveat: box shared with parallel sessions (load 4-6 during pairs);
interleaving carries the verdict, and the effect (2.3x) dwarfs the noise.
Open on other backends: the flat predicate is Metal-only; CUDA/Vulkan im2col
untouched. The `crispstrobe-ops` branch (old v0.10 lineage, recently active)
does NOT carry this kernel — port it there if that lineage ships Metal.

## Kaggle conv A/B (O8+O9): the im2col interchange WINS on x86, PP-OCR det flips to GPU on CUDA (18x), LAYOUT_CONV_F16 stays gated (P100 + Xeon, 2026-08-06)

**v2 addendum (same kernel, second P100 run — v1 verdicts REPLICATE):** o8
gemm4 17.5-20.4 s across both runs, det-CUDA 550-614 ms vs det-CPU
9.5-11.0 s, f16-layout region drift again. Two v2-specific findings:

- **dbnet CPU-vs-CUDA arm: INCONCLUSIVE as designed.** Whole-pipeline wall
  (114.3 s CPU-det vs 111.7 s CUDA-det) is dominated by the TrOCR
  recognition of 38 regions, so the det fraction is invisible — and the two
  arms' decoded TEXT DIFFERS, meaning the det backend changes boxes or the
  probability map. A v3 needs a det-only dbnet harness (bench print + box
  compare), not a wall clock. No dbnet-CUDA claim is made.
- **CUDA-rec 0-results diagnosis capture**: with
  `CRISPEMBED_PPOCRV6_GRAPH_DEBUG=1` on CUDA0, the fused rec graph's tap
  activations are numerically sane through `stage4`, the 18,710-wide logits
  graph builds and computes — yet results=0 (rc=0, boxes=38/12). The fault
  is therefore DOWNSTREAM of graph compute: the logits readback layout or
  the CTC decode consuming it (a scrambled readback argmaxes onto the blank
  class and CTC-collapses every crop to empty, which reproduces the exact
  symptom). This is the starting point for the Fable-queue CUDA-rec item;
  the full stderr is archived in the kernel output.

Kernel `chr1s4/crispembed-conv-ab` v1 (Tesla P100-PCIE-16GB sm_60, Intel Xeon
2.00 GHz), full log `convab.log` in the kernel output; every arm's stdout was
captured and cross-compared per the proof-of-work rule.

**O8 — R6 conv2d arms on x86** (PP-OCRv6 medium SCALAR detector,
`scan_page_pd`, detector-only, 3 interleaved rounds, 38 regions in every arm;
bitwise unit gate 180/180 with the GEMM gate on AND off — byte-equality holds
on AVX2 lanes):

| arm | round times (ms) | vs legacy |
|---|---|--:|
| legacy | 35934 / 37226 / 34689 | — |
| gemm nt=1 | 30820 / 31508 / 31489 | **~0.87x (13% win)** |
| gemm nt=4 | 20438 / 20352 / 20336 | **0.57x (1.76x)** |

**The M1 verdict was L2-size-dependent, exactly as hypothesized**: on M1's
12 MB shared L2 the interchange alone lost 4-7%; on this small-private-L2
Xeon it wins ~13% before threading. Disposition unchanged for now (gate
stays opt-in) — but a per-arch default (interchange on x86, legacy on
Apple-Silicon) is now evidence-backed if an engine wants it.

**O9a — PP-OCRv6 det residency on CUDA**: the "CPU by measurement" default is
a **Metal-only verdict — it FLIPS on CUDA**:

| arm | det graph_ms | boxes |
|---|--:|--:|
| det CPU graph (Xeon) | 11008 | 38 |
| det CUDA graph (`CRISPEMBED_PPOCRV6_DET_GPU_LOAD=1`) | **596-614** | 38 |

18x, identical box count. This is the first hard datapoint for the
per-backend-kind residency defaults (O11): a CUDA/A1000 build should default
PP-OCR det to the GPU graph while M1 keeps CPU.

**O9b — new bug found: PP-OCR REC on CUDA emits 0 results** in BOTH arms
(boxes=38, recognize runs 3.6-3.7 s, stdout ~19 bytes — the "decoded text
identical" check between arms is therefore VACUOUS, and the det verdict above
rests on box counts + det-bench, not text). Unowned; smells like the known
CUDA contiguity/get_rows class — needs the cuda-diag treatment before any
CUDA rec default exists.

**O9c — `LAYOUT_CONV_F16` on P100**: no speed change (warm Phase 1 66.6 ms
f32 vs 65.9 ms f16 — sm_60 has no tensor cores) and a quality drift (20 -> 19
regions, first-line diff). Stays gated off everywhere measured so far; the
tensor-core question needs a T4 assignment (Kaggle randomizes) — v2. Also
notable: layout Phase 1 on CUDA is **60-110 ms vs M1 Metal's ~1.4 s** — the
M1 cost is Metal-specific, not inherent to the graph.

v2 TODO for the kernel: add the DBNet CUDA arm (the Fraktur-lane detect cost)
and retry until a T4 assignment for the f16 tensor-core arm.

## Tesseract Fraktur page re-measured (O4/R1): recognition is 0.4 s not 38.3 s, official is 1.8 s not 9.3 s, and detection owns the gap (Apple M1, 2026-08-06)

Measure-first pass on the R1 backlog item — the "highest evidence" item's
evidence was two-generations stale. Same fixture
(`german_official_print.jpg` 1920x2518), same comparator
(`tools/benchmark_tesseract_page.py`, `test-ocr-orchestrator`, workers 4,
`frk` q8-seeded), 3 repeats, all reported:

| | old record (2026-08-02) | current main |
|---|--:|--:|
| native recognize | 38,338 ms | **382-423 ms** |
| native detect (dbnet, short-side 736) | 102 ms | 3,804-3,822 ms |
| native stage total | 38,690 ms | 4,314-4,368 ms |
| official tesseract 5.5.2 (end-to-end) | 9,340 ms | 1,803-1,833 ms |
| CER vs official text | 0.5279 | **0.2351** (stable across repeats) |

Attribution, verified by disable-arm: the **int8 recurrent-weight cache**
(`379434b1` + `e49d390d`, landed 2026-08-01/02, **default-ON** — the survey's
"gated int8 recurrent-kernel cache" wording was wrong, the env is the opt-out
`CRISPEMBED_TESSERACT_DISABLE_INT_CACHE=1`) gives recognize 4,346 ms -> 581 ms
(7.5x) on its own; the Metal-init load skip (`25ceb9db`, 5.9 s -> 0.47 s per
recognizer load — the old "recognize" span also swallowed worker loads) and
LSTM scratch reuse (`31f71239`) cover the rest of the 38 s. Detector artifact
is not a factor (q8 vs f16 dbnet both ~3.85 s).

**The Fraktur-lane frontier now** (same runs):

| route | stage | detect | recognize | CER |
|---|--:|--:|--:|--:|
| dbnet + int8 rec | 4.31-4.37 s | 3.80-3.82 s | 0.38-0.42 s | **0.2351** |
| classical pageseg + int8 rec | **1.15-1.24 s** | 0.04 s | 1.07-1.16 s | 0.4123 |
| official 5.5.2 | 1.80-1.83 s | — | — | (reference) |

Neither native route dominates: the pageseg route is FASTER THAN OFFICIAL but
loses ~0.18 CER; the dbnet route wins quality but pays 3.8 s of CPU conv
graph (consistent with dbnet's documented ~10 s/1472x736 CPU cost; Metal
measured 139 s — no help there; CUDA untested, and dbnet is NOT yet in the
conv-ab kernel's O9 phase — add in v2). The H9 column-count router already
arbitrates the two routes. Remaining items: dbnet-on-CUDA, pageseg quality
(the existing crop-geometry/decoder-semantics lane). **Recognizer per-line
batching — R1's prescribed fix — is dead: ~0.4 s total at stake.**

## VLM decode-graph overhead (O5): R5 closed — 2.2% qwen2vl, 9.2% granite, and smoldocling never had a graph (Apple M1, 2026-08-06)

The R5 backlog item's own rule was "profile build+alloc as a fraction of
decode; single-digit percent closes the item." Branch
`perf/o5-decode-overhead`, `scan_strip.png`, `-t 4`:

| engine | decode | build+alloc | fraction | steps | source |
|---|--:|--:|--:|--:|---|
| qwen2.5-vl-3b q4_k | 20708 ms | 446 ms | **2.2%** | 125 | existing `QWEN_DBG=1` per-step build/upload/compute/read timers (measured under heavy interactive load — CPU-side build share is inflated if anything; correct OCR text) |
| granite-vision-2b q4_k | 11660 ms (64.8 ms/tok) | 1075 ms | **9.2%** | 180 | NEW permanent split on the `[granite_ocr-bench] decode:` line |
| smoldocling q8_0 | — | — | **0 by construction** | — | `sd_llm_decode_step` is a hand-written CPU loop (`core_cpu` + `sd_linear`) with a host `std::vector` KV cache — there is no decode graph to cache |

Verdict: **no persistent-decode-graph port is justified for any of the three**
— the deepseek T14 result (built it, won nothing, 1-6% overhead) generalizes.
granite's 9.2% is the borderline case: a port would buy at most ~1 s of a
21.7 s end-to-end run; the number is on its bench line for whoever revisits.

Survey corrections bagged along the way: smoldocling's KV is HOST memory (not
"F16 device-resident"), and granite's DEFAULT decode is the per-step ggml
graph (`gv_run_llm_body` T=1, Metal-resident F16 KV) with the scalar loop as
fallback — the R5 text had both backwards in parts.

## PP-OCRv6 medium-tier page profile (O13b): recognize is 71-74%, det ignored -t, and the CPU-only story is 4-6x off (Apple M1, 2026-08-06)

Measure-first profile on the production surface
(`crispembed -m rec --ocr page --ocr-engine ppocrv6 --ocr-det det`), medium
det + medium rec f16, `scan_page_pd.png` 606x1000 → 38 boxes, Metal build,
branch `perf/ppocr-profile`. Text output identical across every arm below.

Stage split (`CRISPEMBED_PPOCRV6_BENCH=1`), `-t 4`, three runs:

| stage | ms |
|---|--:|
| detect (CPU ggml graph) | 5992-6502 |
| crop | 2.7-3.4 |
| orientation | 1.2-1.4 |
| recognize (Metal fused batch graphs) | 16720-19131 |
| model loads (separate line) | ~1232 |

**Recognize is 71-74% of the page.** Per-graph attribution
(`CRISPEMBED_PPOCRV6_GRAPH_BENCH=1`): the fused width-group graphs cost
2-4 s each on MTL0 (e.g. 3931 ms for an `18710x120` logits output) — the
18,710-class CTC head dominates the output side. This, not anything on the
R1-R8 list, is the #1 PP-OCR lever; next step is a per-batch Metal profile
(conv vs SVTR decoder vs head vs readback).

**Detector bug found and fixed: `ppocrv6_det::init(const char*, int)`
declared its thread parameter anonymously and never applied it** — every
caller's `-t` silently ran at ggml's default. With the fix, `-t` is honored:
graph_ms 9.1-9.3 s at a TRUE `-t 1` vs 6.0-6.4 s at `-t 4` vs 6.4-7.9 s at
`-t 8` — saturating at 4 threads (memory-bound). OCR text byte-identical
across t=1 / t=4 / pre-fix binaries (threading partitions rows; no element's
reduction order changes).

**Rec backend truth** (same fixture, `-t 4`): Metal graph **17-19 s**;
CPU scalar (`CRISPEMBED_PPOCRV6_FORCE_CPU=1`) **107.5 s**; CPU graph
(`FORCE_CPU=1 BATCH_GRAPH=1` — FORCE_CPU alone deliberately means the scalar
reference) **69.4 s** (measured under load; det in that run inflated to
18 s, so treat 69 s as an upper bound). The one-shot Metal default is right
on this box. The portability finding: **a GPU-less platform runs medium-tier
rec 4-6x slower than Metal** — the concrete PP-OCR stake in the R6-x86 /
CUDA lane, and a reason to consider promoting the CPU-graph combo where no
GPU exists.

**Diagnostic fix:** the orchestrator's `[ppocrv6-width-bench]` line printed
PRE-bucket width estimates — "27 unique widths" on a page whose real graph
shapes number 8 (the recognizer buckets to 64 by default). It now mirrors
the bucketing; the old line materially misled this very profile.

Unowned follow-ups: `CRISPEMBED_PPOCRV6_ONESHOT_CPU_MAX_REGIONS` observed
not taking effect in-process (the backend stayed MTL0 at 999); pre-existing
`test-ppocrv6-det-diff` FAIL vs `PP-OCRv6_medium_det-fox-ref-final.gguf`
(prob-map cos 0.25, empty intermediate taps, byte-identical with and without
the threads fix — stale-ref suspicion).

## layout_detect Phase 1 (O1): steady-state Metal compute, no warmup story; F16 conv path measured SLOWER on M1 and stays gated (2026-08-05)

Follow-up to the Phase-2 fix below, branch `perf/layout-phase1`,
`layout-heron-f32.gguf` / `scan_page_pd.png`, Metal build.

- **Warm == cold.** With the new `CRISPEMBED_LAYOUT_REPEAT=N` CLI diagnostic,
  three back-to-back detects in one process timed 1603.7 / 1404.1 / 1524.1 ms
  of Phase 1 — no meaningful pipeline-warmup component. The new bench-line
  split shows graph compute is all of it (feat readback+misc: 5-8 ms).
- **The ~1.4 s is the post-9.8x state**: the in-code metal-prof note records
  direct convs at 11.4 s and the im2col+GEMM rewrite at ~1.2 s; the graph is
  ~99.6% GPU-execute. For a ~100-GFLOP-class detector this is a few percent
  of M1 f32 peak — the inefficiency is per-op (im2col traffic, f32 GEMM), not
  graph-structural.
- **`LAYOUT_CONV_F16=1` (new, opt-in, default OFF) LOSES on M1 Metal:**
  Phase 1 2241/2181 ms vs 1661/1408 ms F32 in the same interleaved session.
  The path composes F32-activation `ggml_im2col` with an F16 destination +
  F16xF16 `mul_mat` (the fork's `ggml_conv_2d` forces F32 im2col for F32
  activations, and Metal's im2col kernel rejects an F16 *source* — the naive
  cast-both-inputs version aborts with `unsupported op 'IM2COL'`). Quality is
  not the problem: same 20 regions, same order, scores within ±0.002, boxes
  within ±0.1 px. Kept gated per the plausible-path rule — on F16-tensor-core
  backends (CUDA/A1000) this is the first one-env A/B to run.
- **O2a rider:** the Phase-2 self-attn block re-read and re-transposed
  ~0.85 MB x 6 layers of immutable weights every call; now cached per layer.
  Regions byte-identical (verified pre- and post-format). The saving is below
  the loaded dev box's noise floor and is recorded as removed work, not a ms
  claim. Next Phase-2 candidate: value projection on GPU (est. 101 -> ~30 ms
  at `-t 4`), open as O2b.

## scunet_denoise R7 closed by measurement: weight dequant copies are ~0.1% of a tile pass — no DequantCache (Apple M1, 2026-08-05)

The R7 backlog item ("only SR engine without a DequantCache") argued from
presence, not cost. A permanent atomic accumulator in `to_f32` (printed on the
`CRISPEMBED_SCUNET_BENCH` total line, branch `perf/r7-scunet`) measured, on
`scunet-color-f32.gguf` / `scan_strip.png` 520x260 at `-t 4`: **all weight
`to_f32` copies sum to ~4-5 ms of a ~4.3 s tile pass** (14.1 ms cumulative
over three tile passes totalling ~12.7 s). For an f16 artifact (fp16→fp32
conversion of ~18 MB of weights per tile) the bound is a few times that —
still ~1%. No cache added; denoised PPM output byte-identical with the
instrumentation in place. scunet's actual cost is the Swin/conv compute
(~27 s for this small image), which belongs to the deprioritized SR-on-GPU
research item, not to caching.

## layout_detect Phase 2: the R2 "deform loop dominates" claim was stale — the level input projection was 64%, fixed 2.66x, byte-identical (Apple M1, 2026-08-05)

Measure-first pass on the R2 backlog item (branch `perf/r2-deform`,
`layout-heron-f32.gguf`, `scan_page_pd.png` 606x1000, Metal build, `-t 4`).
New per-stage timers are permanent behind `CRISPEMBED_LAYOUT_DETECT_BENCH`:

| Phase-2 stage | before (ms) | after (ms) |
|---|--:|--:|
| level input projection | **548.8** | 28.5-37.0 |
| value projection | 110.5 | 100.7-115.9 |
| enc-bbox-head | 37.6 | 37.2-38.0 |
| self-attn ggml block | 70.4 | 44.2-49.4 |
| ffn ggml block | 13.1 | 12.8-14.7 |
| deformable-sample loop | 16.1 | 14.7-18.3 |
| **Phase 2 total** | **846-856** | **311-337 (median 318)** |

The deformable-sample loop — the R2 backlog's named target, "instrumented as
the dominant Phase-2 decoder cost" — is **~2% of Phase 2 on current main**;
that claim predates `2a43e4f4` (2026-07-11), which threaded the decoder
cpu_linears and dethroned it. The survey carried it forward unverified.

The actual hotspot was the decoder **level input projection** (1x1 conv over
8400 tokens, `layout_detect.cpp` `input_proj`): a scalar `(n, o, i)` nest
whose inner reduction strode `feat_col` by N_lv, single-threaded — 549 ms.
Rewritten in the `2a43e4f4` AXPY form (contiguous inner axis, identical
per-element accumulation order over `i`, threaded over disjoint output rows;
`cpu_linear` itself could not serve because its square-weight heuristic picks
the MatMul convention and this weight is ONNX Conv `(out, in)`). Ungated, per
the `2a43e4f4` precedent, because the arithmetic is order-identical:
**CLI region output byte-identical to baseline at `-t 1` and `-t 4`**,
re-verified after the clang-format rebuild. Contiguity alone (before
threading) is 549 -> 77.6 ms at `-t 1` (7.1x).

Whole layout call: 2332 -> 1686-1817 ms. **Phase 1 (Metal backbone+encoder,
~1.4 s, thread-count-invariant) is now ~80% of the call** — the next lever is
GPU-graph-side (profile warmup vs steady-state and sched composition first),
not another scalar island. After it, Phase 2's remaining items are the
value projection (~101 ms) and the self-attn block's per-call weight
re-dequant/re-transpose/re-upload.

## R6 conv2d_cpu im2col-tile A/B: threading is the win (2.04x wall at nt=4), the interchange alone is not — on M1 (2026-08-05)

`core_cpu::conv2d_im2col_cpu` (branch `perf/conv2d-gemm`) gathers a tile of
output positions into an L2-sized column buffer and runs the output-channel
loop OUTSIDE the position loop, so each weight row is read once per tile
instead of once per output pixel; a fork-join pool threads over tiles. Every
output element is still `bias + dot_product(patch, w_row, K)` on a patch
gathered in the same order, so the result is **bitwise identical to the
generic path at any thread count** — enforced by an exact-equality unit guard
(`test_conv2d_im2col_equivalence`, 9 shapes covering tile tails, the
16-position tile floor, groups, boundary gathers, nt=1 and nt=4). Gates:
`CRISPEMBED_CONV2D_GEMM=1` (+ `CRISPEMBED_CONV2D_THREADS=N`, default 1),
default OFF.

**Protocol.** PP-OCRv6 medium detector, scalar path
(`CRISPEMBED_PPOCRV6_DET_SCALAR=1`, the canonical `conv2d_cpu` workload),
`scan_page_pd.png` 606x1000, detector-only (`PPOCRV6_DIRECT_MAX_REGIONS=0`),
same binary, arms selected by env gate, interleaved pairs, one process per
run, M1 16 GB with an interactive user (1-min loadavg 6.7-9.8). All 5 nt=4
pairs and the smoke pair reported, nothing trimmed. All arms returned
detector_regions=38.

| pair | legacy elapsed_ms | gemm nt=4 elapsed_ms | ratio |
|---|--:|--:|--:|
| 1 | 31871 | 15643 | 0.491 |
| 2 | 48401 (load spike) | 15307 | 0.316 |
| 3 | 31910 | 15598 | 0.489 |
| 4 | 31730 | 15530 | 0.489 |
| 5 | 31581 | 15567 | 0.493 |

Median ratio **0.489 = 2.04x wall**, nt=4 arm spread 2% (15307-15643 ms)
while the legacy arm ate the load spike. Total user CPU rises 31.6 -> ~45.2 s
(E-core spillover + interchange overhead), so this is a latency win, not an
energy win.

**nt=1 is a small REGRESSION on this box**: +6.6% user CPU in the smoke pair
(31.74 -> 33.84 s) and +4.4% in a back-to-back pair under heavy load (19.37
-> 20.22 s). Reading: the M1's 12 MB shared L2 already holds these weight
matrices, so the legacy path's per-pixel weight re-streaming was largely
cache-served and the column buffer's write+read traffic is pure overhead.
Caveat observed while measuring: absolute user CPU for the IDENTICAL legacy
arm swung 19.4 vs 31.6 s between quiet-ish and loaded batches (macOS
QoS/core-type placement), so only within-pair ratios are quoted.

**Verdict: opt-in, default unchanged** (per the standing A/B rule). The win
to bank now is threading for engines that want it; the interchange hypothesis
still needs the small-L2 x86 arm — **TODO: Kaggle AVX2 A/B of the same three
arms** (also the relevant CPU baseline for any CUDA-box residency decision).
A register-blocked GEMM micro-kernel (changes accumulation order, forfeits
byte-equality) stays open until this path's x86 verdict is in.

Rider (R4, same branch): `lightonocr` gained the missing backend gate —
`CRISPEMBED_LIGHTONOCR_GPU=1` (got_ocr sched pattern, CPU fallback,
`set_n_threads` guarded), `CRISPEMBED_LIGHTONOCR_FORCE_CPU=1` override.
Default arm verified 0 Metal markers + byte-identical output to the
pre-change binary; Metal arm proven live (`ggml_metal` init in stderr) with
**decoded text identical to CPU** on `scan_strip.png` q4_k. First probe:
Metal 7.2 s wall / 1.4 s user vs CPU(-t 4) 5.4 s wall / 20.4 s user — no
wall win on this small fixture, CPU stays the default; the question is now
askable per backend/fixture without a code change.

## Embedder one-shot CLI init: 4.8x, and it was a 683 MB shader-cache file (Apple M1 Metal, 2026-08-05)

A one-shot `crispembed -m model.gguf --json "text"` paid ~0.9 s of fixed init
before doing ~6-20 ms of work, which is why ONNX Runtime "felt faster" despite
CrispEmbed being ~1.4x ahead warm-vs-warm. T18 profiled it
(`CRISPEMBED_INIT_BENCH=1`) instead of guessing: 683 of the 820 ms of internal
init was ggml-metal opening its persistent `MTLBinaryArchive` pipeline cache at
`~/Library/Caches/ggml-metal/Apple_M1.archive`, which had grown to 683 MB
(~1 ms/MB to open). The archive is append-only across every engine and binary on
the machine, it measurably bought nothing (first encode 20.3 ms with it, 17.4 ms
without), and a one-shot CLI can never write an entry back to it — it is
serialised from a static destructor and these binaries leave via `_exit()`.
CrispEmbed now skips an archive above a 64 MB cap
(`CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB=0` restores the old behaviour).

**Protocol.** Interleaved alternating arms, one process per run, medians over
7 pairs (one-shot) / 5 pairs (batch); both arms are the SAME binary selected by
env gate; pairs gated to 1-min `vm.loadavg` <= 8 (observed 1.8-2.5, **0 pairs
discarded**); an M1 16 GB with an interactive user on it. Nothing trimmed.

| case | before | after | speedup |
|---|--:|--:|--:|
| multilingual-e5-small q8_0, one-shot `--json "ein test"` | 895 ms (892-901) | **186 ms** (184-187) | **4.81x** |
| arctic-embed-m-v2 q8_0, one-shot | 911 ms (908-916) | **202 ms** (200-264) | **4.51x** |
| e5-small, warm batch of 512 texts | 5977 ms | 5451 ms | 1.10x |
| arctic, warm batch of 64 texts | 1672 ms | 908 ms | 1.84x |

**Output is byte-identical** — verified on the vectors, 64 texts per model:
worst cosine 1.000000000, `|before|` = `|after|` = 1.000000 (ratio
1.000000000), and the emitted JSON compares equal byte-for-byte. Every change
is in init; no math path was touched. The batch rows are the no-regression
gate: warm throughput did not move except by the init saving.

Init breakdown, e5-small (before → after): Metal device+pipeline-cache
683.1 → 29.4 ms; duplicate GGUF metadata parse 29.7 → 0.0 ms (the loader used to
re-parse a file the caller had already parsed); GGUF parse 29.3 ms, weights
46.9 ms, vocab read 6.0 ms, **SentencePiece build 12.0 ms** — the 250k-vocab
XLM-R tokenizer build was the ticket's prime suspect and is 2% of the cost.

Backend/thread sweep after the fix, batch-64 (each arm includes its own init):

| model | Metal `-t 1` | CPU `-t 1` | CPU `-t 4` |
|---|--:|--:|--:|
| multilingual-e5-small q8_0 | 0.77 s | 0.76 s | **0.35 s** |
| arctic-embed-m-v2 q8_0 | **0.91 s** | 2.77 s | 0.91 s |

The backend default is no longer the interesting knob; the `-t 1` default is.
`--gpu-backend cpu` also now genuinely skips GPU init (0.86 s → 0.14 s one-shot)
— it previously matched no GPU device, warned, and fell back to Metal anyway.

## DeepSeek-OCR-2 decode: a persistent step graph is 1.40x — but not for the reason the task assumed (Apple M1 Metal, 2026-08-05)

`deepseek_ocr2` built, allocated, computed and freed **one graph per layer per
token** (12 per token) plus a separate LM-head graph, bouncing the hidden state
host<->device 24 times per token. T14 replaced that with a single decode-step
graph (embedding lookup -> 12 layers -> final norm -> logits), the
`qwen2vl_ocr.cpp::build_decode_step_graph` pattern.

**Protocol.** Interleaved, alternating arms, one process per run, 9 scored pairs
plus a discarded cold pair, `commons_example_receipt.png` (217 generated tokens,
both arms identical), pairs gated to 1-min loadavg <= 8 (observed 1.4-2.5) on an
otherwise idle M1 with an interactive user. Both arms are the SAME binary
selected by env gate. Numbers are the `[deepseek-ocr2-stage-bench]` line, which
is net-of-load.

| arm | decode med | min | max | spread | total med | prefill med | sam med | qwen2_enc med |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| legacy per-layer | 11473.7 ms | 11022.9 | 12117.2 | 0.095 | 15815.2 ms | 461.1 | 2754.4 | 378.6 |
| persistent (default) | **8191.5 ms** | 8035.6 | 13463.7 | 0.663 | **12784.8 ms** | 462.1 | 2821.5 | 375.5 |

**1.40x decode / 1.24x end-to-end, decoded text byte-identical on all 25 gold
fixtures.** Per-pair ratios 0.700 / 0.676 / 0.971 / 0.694 / 1.049 / 0.697 /
0.718 / 1.176 / 0.725, median **0.700**; 6 of 9 sit at 0.68-0.73. The persistent
arm's 0.663 spread is three upward excursions, not a wider distribution — its
floor is flat at 8035-8192 ms while legacy's own spread is 0.095. Median quoted,
spread quoted, nothing trimmed. Stages neither arm touches are unchanged, which
is the control: prefill 461 vs 462 ms, sam 2754 vs 2822 ms, qwen2_enc 379 vs
376 ms.

**CPU arm — 5/5 synth byte-identical; ONE cc0 fixture differs by ONE codepoint,
and it is explained.** On `commons_example_receipt.png` under `DS2_FORCE_CPU=1`
the legacy arm emits `**Jackson–Washington**` (U+2013 en dash) where the
persistent arm emits `**Jackson-Washington**` (ASCII hyphen), plus one extra
blank line. That is the entire diff. Mechanism: the legacy path runs the final
RMSNorm host-side in `rmsnorm_cpu` (sequential f32 accumulation) and dispatches
the LM head as its own graph, while the persistent path does both in-graph with
a different reduction order; the last-bit logit difference resolves a near-tie
between `-` and `–`. The comparison matrix shows this is a property of the
FIXTURE, not of the new path:

| config | bytes | sha12 | CER |
|---|--:|---|--:|
| CPU legacy | 574 | `ad1afaa6e857` | 0.22604 |
| CPU persistent | 573 | `06cc11c9184c` | **0.22359** |
| Metal legacy | 566 | `046b089ec4e7` | **0.22359** |
| Metal persistent | 566 | `046b089ec4e7` | **0.22359** |

Three of the four configurations agree exactly, and the outlier is **legacy on
CPU**, not the new path — the persistent arm on CPU converges to the same text
both Metal arms produce. CPU-vs-Metal disagreement (table-cell whitespace) is
strictly larger than arm-vs-arm disagreement on this page. CER moves 0.22604 ->
0.22359 on this one fixture, i.e. toward the cross-backend consensus; every other
scored fixture is bit-for-bit equal, so both corpora's mean CER is identical to
5 decimals between arms.

### The premise was wrong: build+alloc was 1%, not the bottleneck

The task was scoped as "amortise the per-layer graph rebuild". `DS_PROFILE=1`
says that rebuild is **26 ms out of 5223 ms of decode on CPU (1%)** and ~3-6% on
Metal. Amortising it could never have paid 40%. What actually paid is that one
graph per token replaces **13 backend dispatches and 24 host<->device
hidden-state transfers per token**. **Before porting this pattern to
qwen2vl/granite/smoldocling, measure the overhead fraction** — the lever is
dispatch and transfer count, not graph construction, and an engine whose decode
is already one dispatch per token has nothing to win here.

### Copying qwen2vl's KV read verbatim was a 2.42x REGRESSION

qwen2vl reads the FULL allocated `max_seq` every step and lets an F16 mask hide
the unwritten tail — correct, and cheap at its shapes. Here `max_seq` is
`n_prompt + max_new + 64` = **1408** while only ~478 slots are ever live, so every
layer of every token attended over ~3x too many slots and materialised three full
`cont(permute(...))` copies of a `[1280 x 1408]` K/V. Measured, same protocol,
5 pairs:

| arm | decode med | per-pair ratio vs legacy |
|---|--:|---|
| legacy per-layer | 13654.9 ms | — |
| persistent, full-`max_seq` read | 32419.5 ms | 2.669 / 2.098 / 2.424 / 3.439 / 2.362 (med **2.424**) |

Fixed by bucketing the read depth to a multiple of 256 (`DS2_KV_BUCKET`, default
256; `0` restores the qwen2vl behaviour), which keeps the shape CONSTANT across
consecutive steps — the property `sched_alloc`'s no-realloc fast path actually
needs — while reading only a little more than is live. Decode 32419 -> 8192 ms.
**A borrowed pattern can invert on the ratio of allocated `max_seq` to live
slots; that ratio is the thing to check before copying one.**

### Gates

`CRISPEMBED_DEEPSEEK_OCR2_BENCH=1` emits `[deepseek-ocr2-stage-bench]`
(net-of-load; prefill and decode accumulated separately). Decode path:
`DS2_LEGACY_DECODE=1` restores per-layer, `DS2_FAST_DECODE=1` names the default
explicitly, `DS2_KV_BUCKET=<n>` tunes the read depth, `DS2_KV_F16=1` switches the
cache to F16 (a PRECISION change, gated separately and NOT part of the
byte-identity gate — unquantified), `DS2_FORCE_CPU=1` pins the CPU backend.

## Tesseract lane: classical page segmentation is 4x faster — on the pages it suits

Measured 2026-08-02/03 on a genuinely quiet box (load 1.7-3.4, the first quiet
window in this whole round). `CRISPEMBED_TESSERACT_PAGESEG=1` replaces the DBNet
CNN detector with projection-based segmentation, which is what Tesseract itself
does — it has no neural detector at all, and that is the whole reason
`tesseract-cli` reads a page in 0.17 s while our lane took 0.70 s.

**On the 20-fixture synthetic corpus (exact ground truth), it wins on both axes:**

| detector path | CER | ms/page |
|---|--:|--:|
| DBNet (current default) | 0.02880 | 884 |
| `CRISPEMBED_TESSERACT_PAGESEG=1` | **0.01460** | **220** |

4.0x faster and CER halved, which also puts the lane ahead of `tesseract-cli`
(CER 0.026-0.036) on quality and level with it on speed (220 ms vs ~170 ms).
Single-page wall clock was 0.70 -> 0.18 s against tesseract-cli's 0.17 s,
reproducible to the centisecond over three rounds.

**CORRECTED 2026-08-03 — the real-scan failure was a bundled flag, not the
segmenter.** `CRISPEMBED_TESSERACT_PAGESEG` sets *two* things: use classical
segmentation, **and skip cleanup** (deskew/binarise/whiten), the latter at a
second site in `ocr_orchestrator.cpp` on the grounds that "page segmentation
measures row ink on the original page". On real scans that second half is what
destroys the result. Same fixture, same segmenter, cleanup the only difference:
`receipt_historical.png` yields **2 boxes without cleanup and 38 with**.

Corrected comparison, regions/characters:

| fixture | DBNet | classical, no cleanup | classical + cleanup |
|---|---|---|---|
| `receipt_historical.png` | 40 / 494 | 2 / 11 | **38 / 616** |
| `commons_example_receipt.png` | 17 / 195 | 8 / 144 | **18 / 280** |
| `german_official_print.jpg` | 21 / 848 | 23 / 944 | 21 / 835 |
| `commons_test_ocr_document.jpg` | **31 / 1754** | 9 / 149 | 18 / 677 |
| `german_official_document.jpg` | **25 / 836** | 7 / 19 | 8 / 34 |
| `simple_table.jpg` | 1 / 5 | **2 / 40** | 1 / 5 |
| `public_domain_formula_photo.jpg` | 0 / 0 | **11 / 84** | 0 / 0 |

With cleanup left on, classical segmentation **beats DBNet on receipts** (616 vs
494 characters on the historical one). It still loses on dense documents. And the
no-cleanup variant is the only configuration that finds anything at all on
`simple_table` and `formula_photo`. Three regimes, no dominant path — so the
earlier "projection segmentation loses 92-98% of the text" statement was
measuring the flag bundle, not the segmenter, and is withdrawn.

### The router: implemented, and both accept-tests falsified

`CRISPEMBED_TESSERACT_SEG_ROUTER=1` runs classical segmentation and falls back to
DBNet when the result looks implausible. The routing *decision* is the unsolved
part, and two candidate probes were measured and rejected:

**Ink coverage** (Otsu binarise, fraction of foreground inside the boxes;
`CRISPEMBED_TESSERACT_SEG_COVERAGE=1`). Rejected by a clean counterexample:
`commons_test_ocr_document.jpg` scores **1.0000 coverage while losing 91% of the
text**. The failure is not ink landing outside boxes — it is boxes at paragraph
granularity when the recogniser needs lines. `receipt_historical` 0.0043 and
`simple_form` 0.0460 are correctly flagged, but `formula_photo` 0.2615 and
`simple_table` 0.4601 are false alarms on pages where classical is the *only*
path that works.

**Median box height / page height.** Rejected: the ranges overlap outright.
Failures 0.037, 0.008, 0.102; successes 0.017, 0.045, 0.083, 0.035, 0.170.

The instrumentation stays because it is cheap and gated, and the negative results
are recorded so the next attempt starts past them. What the data says a working
probe must capture is *granularity relative to the page's text*, not placement
and not absolute size — something like expected line count from the ink row
profile against actual box count.

**A caveat on the labels themselves:** none of the 14 CC0 scans has ground truth,
so "better" above is a character-count proxy against DBNet. Tuning a threshold on
eight proxy-labelled pages would be fitting noise. This is blocked on O8 (corpus
provenance), and that dependency should be respected rather than worked around.

### Cleanup is a SECOND, independent axis — and the corpora disagree about it

`CRISPEMBED_TESSERACT_PAGESEG_CLEANUP=1` now keeps cleanup on while still using
classical segmentation, so the two decisions can be measured separately for the
first time. Default unchanged.

20-fixture synthetic corpus (exact ground truth):

| arm | CER | ms/page |
|---|--:|--:|
| DBNet (default) | 0.02880 | 893 |
| classical, no cleanup | **0.01460** | **224** |
| classical + cleanup | 0.03590 | 265 |

Real CC0 scans, regions/characters:

| fixture | DBNet | classical | classical + cleanup |
|---|---|---|---|
| `receipt_historical.png` | 40 / 494 | 2 / 11 | **38 / 616** |
| `commons_example_receipt.png` | 17 / 195 | 8 / 144 | **18 / 280** |
| `commons_test_ocr_document.jpg` | **31 / 1754** | 9 / 149 | 18 / 677 |
| `german_official_document.jpg` | **25 / 836** | 7 / 19 | 8 / 34 |
| `german_official_print.jpg` | 21 / 848 | **23 / 944** | 21 / 835 |
| `simple_table.jpg` | 1 / 5 | **2 / 40** | 2 / 31 |

**The two corpora want opposite settings.** Cleanup roughly *halves* quality on
clean rendered text (CER 0.0146 -> 0.0359) and is worth 56x the characters on a
historical receipt (11 -> 616). The original code comment — that cleanup moves
the coordinates the row-ink profile depends on — is correct for rendered text and
exactly wrong for scans, where cleanup is what makes the row profile legible.

So the routing problem is **two-dimensional**: {DBNet, classical} x {cleanup,
no cleanup}, four combinations, and each of them is the best choice on some page
in this small set. No single default is defensible, which is why all four remain
reachable and the default is unchanged. It also means a router that only picks
the segmenter would still leave most of the available quality on the table.

### H9 router: column detection works, and the segmentation axis is solved

`CRISPEMBED_TESSERACT_SEG_ROUTER=1` routes on detected column count. A candidate
gutter is a run of near-empty columns in the page interior **that has ink on both
sides on a majority of text-bearing rows** — the both-sides test is what makes it
work. Column totals alone called `german_official_print` two-column and
`receipt_historical` three-column, because a ragged right margin or a short
receipt line leaves a tall empty band indistinguishable from a gutter.

Detection, 9/9 correct:

| fixture | columns |
|---|--:|
| `commons_test_ocr_document.jpg` | **2** |
| `german_official_print`, `receipt_historical`, `commons_example_receipt`, `simple_form`, `simple_table` | 1 |
| synthetic clean / noise / skew | 1 |

End-to-end routing, CC0 (characters):

| fixture | routed path | router | DBNet |
|---|---|--:|--:|
| `commons_test_ocr_document.jpg` | dbnet (fallback) | 1754 | 1754 |
| `german_official_print.jpg` | classical | 835 | 848 |
| `receipt_historical.png` | classical | **600** | 494 |
| `commons_example_receipt.png` | classical | **280** | 195 |

The multi-column page keeps DBNet's full output; the single-column receipts gain
20-25% more text. On the synthetic corpus with cleanup left off, the router gives
**CER 0.01538 at 254 ms/page against DBNet's 0.02880 at 893 ms** — half the error
at 3.5x the speed.

**The cleanup axis is still unrouted, and it is the reason this is not yet a
default.** Cleanup wants to be OFF for clean rendered text (synthetic CER 0.0154
off vs 0.0316 on) and ON for real scans (`receipt_historical` 600 characters vs
11). The router currently inherits whatever the `PAGESEG` flag implies, so no
single invocation is best on both corpora:

| setting | synthetic CER | receipts |
|---|--:|---|
| router (cleanup on) | 0.03156 | good (600 / 280 chars) |
| router + `PAGESEG` (cleanup off) | **0.01538** | poor (cleanup disabled globally) |

Both axes need a per-page decision. The segmentation axis now has a validated
probe. The cleanup axis does not, and two candidate signals were measured and
**rejected** — recorded here so the next attempt does not repeat them:

| probe | synthetic | scans | verdict |
|---|---|---|---|
| paper-class noise (sd above Otsu) | 6.9-13.2 | 5.0-14.0 | **overlaps completely** |
| illumination spread (8x8 tile 90th-pct background) | 0.00 all six | 0.0-45.0 | **counterexample** |

The illumination probe looked promising — every synthetic fixture scores exactly
0.00 — until `commons_example_receipt.png` also scored 0.00 while *wanting*
cleanup (144 characters without it, 280 with). So the axis is not
render-versus-scan, which was the framing behind both probes. Whatever separates
"cleanup helps" from "cleanup hurts" is not global image statistics.

**Root cause, found by looking at the damage instead of computing another
scalar.** Dumping `--cleanup-only` on `synth_00_clean.png` and comparing shows
exactly what cleanup does to a page it hurts: background whitening
(morphological closing) **erodes the antialiased glyph edges** — strokes thin,
serifs break, sentence-final periods almost vanish. On clean rendered type whose
shape depends on mid-grey antialiasing that is destructive; on a scan with thick
saturated ink and background artefacts the same operation removes only the
artefacts. That explains the CER split (0.0146 without cleanup, 0.0316 with)
without reference to noise or illumination.

Knowing the mechanism did **not** yield a working probe. Two further candidates
were measured and rejected:

| probe | result |
|---|---|
| ink retained after cleanup | `commons_example_receipt` keeps only **38%** of its ink and reads *better* (280 vs 144 chars) — what was removed there was background, not glyphs |
| recognizer mean confidence | **2/5 correct**, and biased: picks cleanup-ON even where OFF yields more text (`german_official_print` 944 chars at 0.70 against 835 at 0.76) |

That confidence result matters beyond this item: **cleaned crops read as more
confident even when the text is worse.** Confidence is not a safe quality gate
anywhere in this pipeline.

The local version was then tried too — ink loss **inside** detected text boxes
against loss **outside** them, which should separate glyph erosion from artefact
removal. It does not, and it surfaced a structural blocker that rules out the
whole family:

- In-box loss does not separate: 15.7% on `synth_00_clean` (cleanup unwanted)
  sits between `receipt_historical` 5.5% and `commons_example_receipt` 65.2%
  (both wanted).
- **Cleanup changes the image geometry.** `german_official_print.jpg` comes back
  2532x1938 from 2518x1920, because deskew rotates and crop trims. Any
  before/after *pixel-aligned* comparison is therefore undefined on precisely the
  pages that need deskew — which is most real scans. Seven probes in, this is the
  finding with the longest reach: it invalidates the entire before/after-diff
  family, not just this instance.

Seven approaches are now falsified on this axis (ink coverage, box height, paper
noise, illumination spread, ink retention, confidence, in-box erosion). Two
independent obstacles are now identified rather than suspected: no page-level
statistic separates ink-that-is-glyph from ink-that-is-artefact, and no
pixel-aligned before/after measure survives cleanup's own geometry change.

**Recommendation: stop probing and get labels.** Every remaining idea is a
proxy for "which output is more correct", and that question is answerable
directly and cheaply for a handful of pages. Transcribing 5-10 CC0 scans turns
the cleanup axis into a two-arm scoring run, and simultaneously unblocks O8, the
WER column, and the H9 acceptance gate. Continuing to invent proxies against
character-count labels — one of which was already shown to be directionally wrong
on `german_official_document` — has a worse expected return than an hour of
transcription.

Until then the router stays opt-in.

### Looking at the fixtures corrected the labels and found the actual signal

The router probes were being tuned against character counts on scans nobody had
looked at. Opening them changed two things.

**Half the CC0 set cannot score the English lane at all.** `german_official_document.jpg`
is an 1848 handwritten Fraktur *Bürger-Brief* — ornate Kurrent script, fold
creases, photographed at an angle — being read by an **English** Tesseract-LSTM.
Its "836 characters" against classical's "34" is not a win; it is an English
model transliterating Fraktur (`Sveer- Dvie]` for *Bürger-Brief*,
`Dir Sberitggermfier` for *Wir Ober-Bürgermeister*). Both arms are wrong; the
character count merely rewards the one that hallucinates more fluently. **That
label is withdrawn**, and the same applies to `arabic_handwriting.jpg`,
`german_kurrent_handwriting.jpg`, `handwritten_letter.jpg` (handwriting),
`arabic_printed_line.png` (needs `tesseract-ara`) and
`public_domain_sheet_music.jpg` (not prose). A Fraktur model is in the cache
(`tesseract-frk-*`) and is what that fixture actually needs.

`simple_table.jpg` is 200x102: its title and 5x5 grid are legible but the cell
digits are unrecoverable at that resolution even upscaled 6x, so it can support a
*directional* judgement (DBNet returning 5 characters for ~35 text cells is a
clear miss) but never a CER gate.

**The real routing signal is column count.** `commons_test_ocr_document.jpg` —
the fixture where classical loses hardest, 1754 characters against 677 — is
clean printed English in **two columns**. A horizontal row-ink projection merges
the two columns into single rows, which is projection segmentation's textbook
failure mode. That explains every result in the table above without appeal to
density or noise: the pages classical wins on (receipts, `german_official_print`,
the synthetic corpus) are single-column; the one it loses hardest on is
two-column.

That is also a cheap probe, and a principled one rather than a fitted threshold:
take the **vertical** ink projection and look for a sustained low-ink valley
spanning the page height. One column, no valley. It costs the same pass the
segmenter already makes. Neither ink coverage nor box height could have found
this, because both are page-level scalars and the property is structural.

**Recommended next step for H9:** implement column detection as the accept test,
validate on the six fixtures that the English lane can legitimately score
(`commons_test_ocr_document` two-column, `german_official_print`,
`receipt_historical`, `commons_example_receipt`, `simple_form`, `simple_table`
single-column), and keep the Fraktur/Arabic/handwriting fixtures out of the
English-lane gate entirely.

**This is now blocked on labels, not on ideas.** Choosing among four
configurations per page needs ground truth on real scans; the CC0 set has none,
so every "better" above is a character-count proxy. See O8.

**Still the actionable item: a router, not a flip** — and separately, the
cleanup coupling should be unbundled from the segmentation choice, since they
are independent decisions that the single `PAGESEG` flag currently forces to
move together.


## ⚠ The 1x1 and depthwise conv fast paths are NOT wins — measured, both stay off

**Retracted: the "9.1% faster on M1" figure this file previously headlined does
not replicate.** It came from a single median-of-3 A/B. Repeating it as
interleaved off/on pairs — which cancels drift, unlike two separate medians —
gives this:

| host | pair deltas (gate off -> on) | mean | sd | 95% CI |
|---|---|--:|--:|---|
| M1 Mac (NEON) | +15.7, -1.5, -1.2, +1.9 | +3.7% | 8.1 | [-4.2, +11.7] |
| VPS Xeon (AVX2) | -7.6, -2.3, -3.3, -13.3, +1.9, -4.2 | -4.8% | 5.2 | [-8.9, -0.7] |

Read the Mac row carefully: three of the four pairs sit inside ±2%, and the mean
is carried entirely by pair 1, whose *baseline* was the outlier (10.23 CPU-s
against ~9.5 for the others) rather than its gated arm being fast. Drop that
pair and the mean is **-0.3%**. The confidence interval spans zero. The original
8.59 -> 7.81 reading was the same artifact: an unlucky high baseline, not a fast
kernel.

**Conclusions that survive:**

- On x86 the 1x1 tiled kernel is a **regression** — five of six pairs negative,
  CI excludes zero, roughly -5%.
- On ARM it is **neutral**, not a 9% win.
- There is therefore **no architectural flip**. The earlier "sign flips with the
  instruction set" story was a noisy Mac measurement set against a real x86
  regression. Both gates stay off, and this time the reason is that neither
  earns its place, not that the evidence is split.
- The depthwise gate's single-shot 3.6% on the Mac is unverified against this
  same noise floor and was neutral on x86. Treat it as unmeasured.

### The measurement protocol itself was the bug

This is the durable finding. PLAN §1 prescribes median-of-3 CPU-seconds with a
control before and after, and that protocol **cannot resolve an effect of this
size on either machine**:

| host | sd of paired delta | interleaved pairs needed to resolve a 5% effect at 95% |
|---|--:|--:|
| M1 Mac | 8.1% | **41** |
| VPS Xeon | 5.2% | **16** |

A median-of-3 is three samples, and comparing two separately-taken medians is
strictly worse than pairing because slow drift lands entirely in one arm. Any
past finding in this file at the 3-10% level that rests on a single median-of-3
should be treated as unverified until re-run as interleaved pairs with n
reported. The control-before/control-after bracket does not rescue it: both
controls can agree within 30% while the measured arms straddle a 15% swing, as
pair 1 above did.

**What to do instead:** interleave the arms in one loop, report every pair, and
publish the spread rather than a single number. If the CI includes zero, the
honest result is "no measurable effect", which is a perfectly good outcome for a
gated path.

## PP-OCRv6 scalar detector — where the convolution time goes (Apple M1, 2026-08-02)

Per-convolution profile of the CPU scalar detector, the dominant cost of the
PP-OCRv6 lane and the shared shape family for the other classical lanes.
Enable with `CRISPEMBED_PPOCRV6_DET_PROFILE=1`; fixture
`tests/regression/images/cc0/german_official_print.jpg` (1920x2518, detector
input 960x736).

| convolution class | share of detector conv time | rate |
|---|--:|--:|
| 1x1 pointwise | 51.6% | ~1.2 GF/s |
| depthwise (groups == channels) | 20.4% | 0.02-0.19 GF/s |
| deconv (2x2 stride 2) | 6.4% | ~0.6 GF/s |
| all other convolutions | 21.6% | ~1.0 GF/s |

**Read the shares, not the absolute totals.** The same profile measured
17,506 ms and 12,089 ms of total convolution time on two runs minutes apart,
because this box routinely sits at load 30-110 with several agent builds
running; the class proportions moved by under 3 points across the same pair.
Any absolute figure here is contended wall clock and is not a benchmark.

Heaviest single layer: a 7x7 **depthwise** convolution, 96 channels at 240x184,
2394 ms and 13.7% of all convolution time at 0.17 GF/s. Depthwise is the
generic `conv2d_cpu` path's worst case by construction — with one input and one
output channel per group there is nothing to amortise the patch gather against,
so it gathers a kh*kw window and consumes it in a single `dot_product`, once per
output pixel.

### 1x1 convolution kernel A/B

Same binary, gate off vs on, CPU-seconds (`user+sys`) median-of-3, bracketed by
the external `tesseract` load control. Wall clock is unusable on this box; CPU
time held across the window.

| arm | CPU-s |
|---|--:|
| control (`tesseract`, before) | 0.40 |
| `CRISPEMBED_CONV1X1_FAST` off (default) | 8.59 |
| `CRISPEMBED_CONV1X1_FAST=1` | **7.81** |
| control (`tesseract`, after) | 0.42 |

**This 9.1% did not replicate — see the retraction at the top of this file.**
Repeated as interleaved pairs the kernel is neutral on ARM and a ~5% regression
on x86; the 8.59 baseline here was an unlucky-high sample. The description of
the traversal change below is still accurate as a description; it is the
speed claim that failed: the old form streamed the whole output plane once per (oc, ic)
pair, the current one blocks the pixel axis into 8192-element tiles so a tile's
input slab stays L2-resident and computes four output channels at a time.
Decoded-text equivalence: **34 of 34 fixtures identical, 0 differing** — the 20
synthetic ground-truth fixtures plus all 14 CC0 scans, gate off vs on, same
binary. H1's acceptance criteria (CPU-seconds down, decoded output identical,
control in range) are therefore met **for this engine**.

The gate nevertheless stays opt-in, which is the disciplined reading of H1's own
constraint: `conv2d_cpu` is shared by 15 engines — `ppocrv6_det`, `ppocrv6_ocr`,
`pplcnet_orientation`, `surya_det`, `nafnet_denoise`, `text_sr`, `got_ocr`,
`deepseek_ocr2`, `unlimited_ocr`, `ppformulanet_ocr`, `ppformulanet_l_ocr`,
`bttr_ocr`, `hmer_ocr`, `posformer_ocr` — and exactly one of them has been
measured. A global flip needs an A/B per engine with a runnable fixture, not an
extrapolation from the one with the largest 1x1 share.

### Depthwise convolution kernel A/B — a partial win, and why it is only partial

`conv2d_depthwise_cpu` (`CRISPEMBED_CONVDW_FAST=1`) replaces the per-output-pixel
gather with a loop inversion: per channel and output row, walk the kh*kw taps and
accumulate a whole output row per tap, so each tap is a contiguous axpy, the
input row stays in L1 across all taps, and the boundary test becomes a per-tap
column range in closed form. Same binary, same fixture, CPU-seconds median-of-3,
box at load 23:

| arm | CPU-s |
|---|--:|
| control (`tesseract`, before) | 0.48 |
| both gates off | 8.69 |
| `CRISPEMBED_CONVDW_FAST=1` | 8.38 |
| `CRISPEMBED_CONV1X1_FAST=1` + `CRISPEMBED_CONVDW_FAST=1` | 7.95 |
| control (`tesseract`, after) | 0.45 |

**3.6% for depthwise alone — well short of what a 20.4% share should give**, so
the kernel is maybe 25-30% faster rather than the several-fold the 0.02-0.19
GF/s rate suggested was available. Recorded as a gated partial win, not a
success.

Two things ruled out for whoever picks this up:

- **It is not a vectorization failure from pointer aliasing.** The obvious
  suspicion is that `orow[ox] += wv * s[ox - lo]` cannot vectorize because `out`
  and `in` might alias. Checked with `clang++ -O3 -Rpass=loop-vectorize`: the
  aliasing and `__restrict` forms *both* vectorize, width 4 interleave 4, via
  runtime alias checks. Adding `__restrict` is not the fix.
- **Per-layer profiler numbers cannot settle this at n=1.** A single profiled run
  per arm reported total convolution time moving 8787 -> 9402 ms *between arms*,
  which is larger than the effect being measured. Only the median-of-3
  CPU-seconds A/B above is trustworthy here.

The remaining suspect is loop-invocation overhead rather than the loop body: the
7x7 layer runs 49 taps x 240 rows x 96 channels = **1.13M invocations** of a
~184-element inner loop, each paying a runtime alias check, prologue and a
remainder epilogue. The fix that follows from that is to make each invocation do
more work — unroll across taps so one pass over the output row accumulates
several kx taps at once, cutting both the output-row traffic and the invocation
count by the unroll factor, with the row borders kept on the current general
path. Not attempted yet.

### EasyOCR lane — where the time actually goes (2026-08-02)

`CRISPEMBED_EASYOCR_STAGE_BENCH=1` on `commons_test_ocr_document.jpg`
(1920x2518, 289 detector boxes grouped to 27 lines). Absolute figures are
contended wall clock on a box at load 30+; the shares are the result.

| stage | ms | share |
|---|--:|--:|
| detect (DBNet) | 24,778 | ~55% of lane |
| recognize loop | 20,130 | ~45% of lane |
| — crop extraction | 41 | 0.2% of loop |
| — `set_width` graph rebuilds | 4,825 | 24% of loop |
| — recognize | 15,263 | 76% of loop |

**Detection is the larger half of this lane.** The recognizer is 76% of 45%,
about a third of the lane, so the "EasyOCR CRNN is 2.2x the Tesseract LSTM"
comparison is measuring a whole-lane number against a component.

`EASYOCR_WIDTH_SORT=1` cuts graph rebuilds from 25 to 14 (27 regions, 14
distinct canvas widths) and `set_width` from 4,825 to 2,348 ms — roughly half,
~12% of the recognition loop but only ~5% of the lane. That reconciles with the
0-3% previously recorded for P6, which was measured against total lane time
where detection dominates. The ceiling is the distinct-width count, not the
region count.

Fixed while measuring this: the width-sort key hardcoded the 2-pixel detector
crop margin while the recognition loop applies it only when
`add_detector_crop_margin` is set, so the external-geometry path sorted by
widths it never requests — 19 rebuilds against 15 distinct widths. Verified
19 -> 15 after deriving the key from the same `pad`.

### Cost of a GPU backend an engine never computes on

`text_sr`, `tps_locnet` and `bert_ner` each built a GPU backend, used it only to
pull the GGUF through `core_gguf::load_weights`, copied every weight out to host
vectors and freed it, without ever running a graph on it — the same bug P2 fixed
in `tesseract_lstm`. No GGUF for any of the three is cached locally, so rather
than extrapolate from P2's 12.5x the cost was measured at its source with
`test-backend-smoke`, which builds a backend and runs one trivial graph on it.
Median-of-3, box at load 38:

| backend | CPU-s | wall |
|---|--:|--:|
| `metal` | 2.62 | 6.71 |
| `cpu` | 0.03 | 0.03 |

~2.6 CPU-seconds and ~6.7 s wall per invocation, dominated by Metal shader
library compilation. This is init plus a trivial graph rather than pure init,
and it is a per-process cost — it dominates a one-shot CLI invocation of a small
model and vanishes in a warm server. All three now load through
`ggml_backend_cpu_init()`, with the old path kept behind `TEXT_SR_GPU_LOAD`,
`TPS_LOCNET_GPU_LOAD` and `BERT_NER_GPU_LOAD`. An end-to-end before/after on a
real model for each engine is still outstanding.

## EasyOCR GGML parity benchmarks — Apple M1, 2026-08-01

All measurements below use the same `scan_strip.png` input and Miniconda
PyTorch reference where stated. Native timings are warm graph timings unless
noted; they are acceptance evidence, not claims that the current implementation
meets the speed target.

### Recognizers

| Path | Native Metal | Python CPU reference | Ratio | Output/parity |
|---|---:|---:|---:|---|
| Latin Gen2 formula, width 200 | 16.523 ms | 12.460 ms | 1.33x | `x=0442` both; all stages pass |
| Latin Gen2 scan, width 128 | 10.885 ms | 7.137 ms | 1.53x | `82` both; all stages pass |
| Latin Gen1 ResNet, width 128 | 154.082 ms | 78.648 ms | 1.96x | `==#` both; all stages pass |
| English Gen2 scan, width 200 | 16.536 ms | 10.035 ms | 1.65x | `032` both; all stages pass |
| English Gen2 scan, width 128 | 10.697 ms | 7.287 ms | 1.47x | `@32` both; strict timestep-11 row cosine remains open |

Native is slower in every recognizer measurement. These are cross-device
directional comparisons; graph/kernel and dynamic-width optimization remain
open. Repeated native outputs are stable after fixing persistent LSTM state
storage aliasing.

### CRAFT detector

| Backend/model | Native graph | Python CPU reference | Ratio | Output/parity |
|---|---:|---:|---:|---|
| Metal, runtime-BN F16 | 850.018 ms | 396.027 ms | 2.15x | 106 boxes both; taps pass |

The runtime-BN F32 graph matches captured Python tensors to floating-point
noise. Runtime-BN F16 also decodes 106 boxes. The older folded-F16 artifact
decoded 107 because accumulated CNN/BN error crossed a threshold; it is stale.
CPU-forced and Metal CRAFT outputs are byte-identical on this fixture.

### DBNet detector

| Backend/model | Graph | Postprocess | Total | Python CPU reference | Ratio | Output/parity |
|---|---:|---:|---:|---:|---:|---|
| CPU, F16, 1 thread | 4178.6 ms | 8.3 ms | 4186.9 ms | 1213.450 ms | 3.45x | all taps pass; 96 regions |
| CPU, F16, 4 threads, persistent graph | 5661.1 ms warm | ~10 ms | ~5661 ms | 1213.450 ms | 4.67x | 98 rapid regions; `Brighton` present |
| CPU, F16, 8 threads, persistent graph | 2907.2 ms warm | ~10 ms | ~2907 ms | 1213.450 ms | 2.40x | 98 rapid regions; `Brighton` present |
| Metal, F16, persistent graph | 3499.4 ms warm | ~10 ms | ~3499 ms | 577.342 ms MPS | 6.06x | 98 rapid regions; `Brighton` present |

The Python reference reports `torch.get_num_threads()=4` and
`torch.get_num_interop_threads()=8`. Thus the 8-thread native result is the
best available throughput measurement but is not a same-thread comparison;
native remains slower even with twice the reference compute threads. On the
same M1 Metal device, the Python MPS blueprint averages `577.342 ms`, making
native Metal `6.06x` slower; this isolates the remaining gap to CrispEmbed's
Metal convolution/deconvolution kernels. F16
matches the fresh official MMOCR reference at backbone, neck, head, and
probability-map boundaries. The detector now uses a shape-keyed persistent
GGML graph; diagnostic tap retention is opt-in via
`OCR_DETECT_CAPTURE_TAPS=1`. Native quality is on par on this fixture, but all
native backend timings miss the reference speed target. Increasing CPU threads
and graph persistence help operationally but do not close the compute gap. Q4_K decodes the same 96 regions but diverges
at `backbone_stage_0` (global cosine `0.9960006`, RMS `0.07697`) and ends at
final-map cosine `0.9311001`; Q4_K is a quantization-quality TODO, not an
accepted parity variant.

An opt-in `OCR_DETECT_DIRECT_CONV=1` experiment was not promoted. GGML's CPU
direct-convolution kernel requires F32 weights, and the F32 direct graph did
not complete a diff run within roughly two minutes on the shared M1; it is not
parity or performance evidence. The default persistent im2col path is
unchanged. A later optimized/vectorized direct kernel remains a performance
TODO. One subsequent baseline run was resource-contended (44.1 s cold /
66.7 s warm with 8 threads), so it is excluded from the stable ratios above.
An attempted cumulative per-tap profiler was also rejected: prefix graphs
shared the persistent tensor arena and changed the restored run to zero boxes.
It produced no valid stage timings; isolated-allocator profiling remains open.

Benchmark results on Intel Xeon Skylake (4 threads), CPU-only, no GPU.

## Server Mode Latency (model loaded once)

Single-text encoding latency via HTTP server (`/embed` endpoint).

| Model | Quant | Params | Dim | Avg (ms) | Texts/s |
|-------|-------|--------|-----|----------|---------|
| all-MiniLM-L6-v2 | F32 | 22M | 384 | 15.5 | 64 |
| arctic-embed-xs | F32 | 22M | 384 | 15.5 | 64 |
| gte-small | F32 | 33M | 384 | 30 | 33 |
| octen-0.6b | Q8_0 | 600M | 1024 | 308 | 3.2 |
| octen-0.6b | Q4_K | 600M | 1024 | 294 | 3.4 |

## macOS Metal (Apple M1)

Benchmarked with Metal backend + embedded shaders, `./benchmark.sh --multi -n 20`.

### all-MiniLM-L6-v2 (22M params, 384d)

| Engine | Single text | Batch (10 texts) |
|--------|------------|-------------------|
| fastembed-rs (Rust, ONNX) | 3.9 ms / 258 t/s | 19 ms / 533 t/s |
| **CrispEmbed Python** (Metal, ctypes) | 4.0 ms / 248 t/s | 62 ms / 161 t/s |
| HuggingFace sentence-transformers | 11.4 ms / 88 t/s | 23 ms / 431 t/s |
| CrispEmbed Server (Metal + HTTP) | 21.9 ms / 45 t/s | 31 ms / 318 t/s |
| FastEmbed Python (ONNX) | 33.5 ms / 30 t/s | -- |

### gte-small (33M params, 384d)

| Engine | Single text | Batch (10 texts) |
|--------|------------|-------------------|
| fastembed-rs (Rust, ONNX) | 4.1 ms / 243 t/s | 21 ms / 479 t/s |
| **CrispEmbed Python** (Metal, ctypes) | 6.4 ms / 155 t/s | 70 ms / 142 t/s |
| HuggingFace sentence-transformers | 22.6 ms / 44 t/s | 226 ms / 44 t/s |
| CrispEmbed Server (Metal + HTTP) | 24.9 ms / 40 t/s | 52 ms / 190 t/s |

### arctic-embed-xs (22M params, 384d)

| Engine | Single text | Batch (10 texts) |
|--------|------------|-------------------|
| **CrispEmbed Python** (Metal, ctypes) | 3.7 ms / 267 t/s | 46 ms / 220 t/s |
| fastembed-rs (Rust, ONNX) | 4.0 ms / 251 t/s | 29 ms / 342 t/s |
| FastEmbed Python (ONNX) | 4.1 ms / 244 t/s | -- |
| CrispEmbed Server (Metal + HTTP) | 22.2 ms / 44 t/s | 35 ms / 287 t/s |

CrispEmbed Python wrapper (ctypes, Metal) matches or beats fastembed-rs for
single-text latency. Batch throughput gap is due to per-text Python loop --
a C-level batch API would close it.

### VLM OCR decode — GOT-OCR2 (Qwen2-0.5B decoder), per token

| Decoder weights | Decode / token | Size |
|-----------------|----------------|------|
| **Q4_K** | **~20 ms** | 445 MB |
| F16  | ~38 ms | 1.44 GB |
| Q8_0 | ~42 ms | 599 MB |

Q4_K is fastest **and** smallest — prefer it over Q8_0 for autoregressive
decode on M1. Q8_0 being slower than F16 here is a Metal `mul_mv` (single-token
mat-vec) kernel issue, not a bandwidth effect; see
[`docs/metal-q8_0-mul_mv-slow-m1.md`](docs/metal-q8_0-mul_mv-slow-m1.md). Correctness
is unaffected (all three quants: cos ≥ 0.99996 vs f32, identical OCR — see
[`docs/got-ocr2.md`](docs/got-ocr2.md)).

### DBNet text detection — scanline box scoring (2026-07-13, M1 CPU)

`extract_boxes`' polygon scoring was O(bbox_area × contour_len) (ray-cast every
bbox pixel against the full traced contour) — pathological when a degenerate
component yields a very long contour. Rewritten as a scanline polygon fill
(even-odd-identical → **byte-identical boxes**, `OCR_DETECT_SCALAR_SCORE=1` for
the old path). dbnet-ic15-q4_k, forced CPU, a 10-line page:

| Stage | Before | After |
|-------|--------|-------|
| DBNet postprocess | 43 326 ms | **1 540 ms** (~28×) |
| Detection total (graph 3 s + postproc) | 46.4 s | **4.9 s** |
| Full DBNet+TrOCR pipeline (14 regions) | ~46 s | **7.2 s** (detect 4.4 · batch-enc 2.5 · decode 0.3) |

Note the decode is not the pipeline bottleneck here — the detection conv graph
and the ViT crop encoder are (both inherent compute). See LEARNINGS / HISTORY.

### DBNet degenerate-component fallback (2026-07-31, Apple M1 Metal)

The existing scanline scorer exposed a second postprocessing failure: valid
4-connected DBNet components could produce a one-point contour, making polygon
score zero and rejecting every box at the default `box_threshold=0.5`. The
postprocessor now falls back to the component bounding box and mean probability
for contours with fewer than three points.

On `tests/regression/images/fox.png` with
`dbnet-ic15-q4_k.gguf`, this changes detection from **0 to 10 boxes**. The
full DBNet+TrOCR pipeline recognizes 10 regions in about **5.0 s warm** on the
M1 Metal path (detection ~3.0 s, batched crop encoding ~1.8 s, decoding ~0.2 s).

### External document-parser comparison (2026-07-31)

The local CrispEmbed live check used the repeatable `fox.png` fixture and
GGUF models from `$CRISPEMBED_GGUF_DIR`:

| Engine / environment | Detection | Recognition | Timing | Quality check |
|---|---:|---:|---:|---|
| CrispEmbed DBNet + TrOCR, Apple M1 Metal | 10 regions | 10 regions | ~5.0–5.3 s/image warm | 8/10 words exact, CER 6.1% |

Expected text was `THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG 12345`; the two
word errors were `TAX` for `FOX` and `IAZY` for `LAZY`.

The comparison implementation could not be executed on this host: its CPU probe requires the
OpenCV development package, its live production path requires the documented
CUDA/TensorRT stack, and no usable Docker daemon or NVIDIA device is available.
Its repository reports **520–559 images/s** for forms/receipts and **200+
images/s** for dense documents on one RTX 5090, plus **92% FUNSD / 93% CORD
word-F1** and **0.90 OmniDocBench-125 overall at 20 pages/s**. Those are
The external NVIDIA benchmark claims are not measurements from this machine, and
are not directly comparable to the M1 single-image fixture above.

The actionable conclusion is to keep CrispEmbed's portable GGUF/Metal path,
but prioritize OCR quality and detector/recognizer batching before claiming
production parity. A fair head-to-head requires the same document corpus,
warmup policy, output metric, and an NVIDIA CUDA/TensorRT host.

#### TrOCR quantization A/B

This was not a TrOCR-vs-Python failure. The same detector, crops, decoder, and
ggml runtime were run with the recommended Q8 recognizer versus the locally
available Q4 model:

| Recognizer | Model size | Output on fox fixture | Warm total |
|---|---:|---|---:|
| TrOCR-small-printed Q4_K | 43 MB | `TAX`, `IAZY` errors | 4.75 s |
| TrOCR-small-printed Q8_0 | 64 MB | exact 10/10 words | 5.13 s |

The model card explicitly warns that Q4_K degrades this narrow 256-dimensional
decoder and recommends Q8_0. Q8_0 is therefore the immediate quality fix;
the ~8% end-to-end cost increase is small because detection and the encoder
dominate this fixture.

## Ollama Integration (Q8_0, Apple M1)

All CrispEmbed models verified in Ollama fork with Ollama-compatible GGUF export.

### Encoder Models (Q8_0 and Q4_K vs HuggingFace F32)

| Model | Dim | Q8_0 cos | Q4_K cos | Q8_0 Size | Q4_K Size |
|-------|-----|----------|----------|-----------|-----------|
| all-MiniLM-L6-v2 | 384 | 0.9998 | 0.970 | 24 MB | 18 MB |
| gte-small | 384 | 0.9999 | 0.991 | 34 MB | 24 MB |
| arctic-embed-xs | 384 | 0.9999 | 0.995 | 24 MB | 18 MB |
| multilingual-e5-small | 384 | 0.9999 | 0.990 | 126 MB | 115 MB |
| pixie-rune-v1 | 1024 | cross-lingual OK | cross-lingual OK | 581 MB | 437 MB |
| arctic-embed-l-v2 | 1024 | L2-norm=1.0 | L2-norm=1.0 | 581 MB | 437 MB |
| granite-embedding-97m-r2 | 384 | 0.99958 | not built | 106 MB | — |
| granite-embedding-311m-r2 | 768 | 0.99976 | not built | 331 MB | — |

### Decoder Models (Q8_0 and Q4_K in Ollama)

| Model | Arch | Dim | Q8_0 Size | Q4_K Size | L2-Norm | Diversity |
|-------|------|-----|-----------|-----------|---------|-----------|
| qwen3-embed-0.6b | Qwen3 | 1024 | 610 MB | 300 MB | 1.000 | 0.599 |
| octen-0.6b | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.649 |
| f2llm-v2-0.6b | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.711 |
| harrier-0.6b | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.504 |
| harrier-270m | Gemma3 | 640 | 287 MB | 239 MB | 1.000 | 0.922 |
| jina-v5-nano | Qwen3 | 768 | 222 MB | 168 MB | 1.000 | 0.237 |
| jina-v5-small | Qwen3 | 1024 | 610 MB | 400 MB | 1.000 | 0.746 |

All 13 Ollama-verified Q4_K models: L2-normalized, semantically correct embeddings.
Diversity = 1 - avg cosine similarity between 4 different test texts (higher = better discrimination).

## GPU Inference (CUDA)

Tested on NVIDIA RTX A1000 Laptop GPU (4GB VRAM), via HTTP server.

| Model | Quant | Avg (ms) | Texts/s | Batch (10) |
|-------|-------|----------|---------|------------|
| all-MiniLM-L6-v2 | F32 | 7.4 | 135 | 211/s |

GPU inference **matches HuggingFace PyTorch** (10.6ms vs 10.8ms) and
**beats fastembed ONNX** (10.6ms vs 13.4ms). Both HF and CrispEmbed use
CUDA on this hardware. The ggml_backend_sched dispatcher offloads
matmul, flash attention, and norm ops to CUDA.

True batched encoding: single graph with 4D flash attention for B texts.
Batch mode (10 texts): 190-211 texts/s on CUDA. HF gets 347/s via
PyTorch's native batch parallelism (more mature GPU batching).

## CPU Batch Mode

| Model | Batch Latency | Per-text | Texts/s |
|-------|--------------|----------|---------|
| all-MiniLM-L6-v2 | 114ms | 11.4ms | 88 |

Optimizations: graph caching, flash attention (fused QKV), buffer reuse,
sorted batch processing (group by token count for graph cache hits).

**True single-graph batching for bidirectional encoders (2026-07, opt-in).** The
default path encodes each text in its own graph. Two fused batch paths are available
for absolute-position encoders (BERT/XLM-R/MiniLM/BGE/E5), both bit-parity with
per-sequence encoding (cos ≥ 0.9999):

| Path | Env | Attention | Notes |
|------|-----|-----------|-------|
| Packed block-diagonal | `CRISPEMBED_ENCODER_PACKED=1` | O(T_total²) | one graph, block-diagonal mask; token-budget grouped (`CRISPEMBED_ENCODER_PACK_MAXTOK`, def 384). Size/backend dependent |
| Rectangular 4D per-item | `CRISPEMBED_ENCODER_4D=1` | O(B·T²) | separate 4D items + per-item pad mask; length-sorted chunks (`CRISPEMBED_ENCODER_4D_GROUP`, def 32) |

The 4D path is **consistently faster than both sequential and packed** (≈1.2×–1.5× at
batch 8/32/128 of short texts, on M1 CPU) and is the recommended path; it stays opt-in
pending a real-Metal A/B (measurements above are CPU-only). See PLAN.md § C3.

## Comparison with HuggingFace and fastembed (ONNX)

Single-text latency, same hardware (CPU, 4 threads).

| Model | CrispEmbed | HF PyTorch | fastembed ONNX | vs HF | vs ONNX |
|-------|-----------|------------|----------------|-------|---------|
| MiniLM-L6-v2 | **15.5ms** | 54ms | 29.5ms | **3.5x faster** | **1.9x faster** |
| gte-small | **30ms** | 79ms | -- | **2.6x faster** | -- |
| arctic-embed-xs | **15.5ms** | -- | 4.9ms | -- | 0.32x |

Optimizations: graph caching, flash attention, pre-merged QKV weights, buffer reuse.

CrispEmbed is **1.9-3.5x faster than HF PyTorch** and **1.9x faster than fastembed ONNX**
for MiniLM on pure CPU. Fastembed ONNX is 3x faster for arctic-embed-xs due to ORT's
Level3 graph JIT compilation (operator fusion, fused LayerNorm, layout optimization).
We apply QKV weight fusion and flash attention but cannot match ORT's runtime compilation.

Key advantages:
- No Python runtime overhead (direct C++ inference)
- No ONNX runtime dependency
- Graph + work buffer reuse across calls
- ~20MB binary vs ~500MB Python + ONNX environment

## Model Sizes

| Model | F32 | Q8_0 | Q4_K | Q8_0 ratio |
|-------|-----|------|------|------------|
| all-MiniLM-L6-v2 | 87 MB | 24 MB | 19 MB | 3.6x |
| gte-small | 128 MB | 35 MB | 25 MB | 3.7x |
| arctic-embed-xs | 87 MB | 24 MB | 19 MB | 3.6x |
| multilingual-e5-small | 453 MB | 123 MB | 113 MB | 3.7x |
| pixie-rune-v1 | 2.2 GB | 580 MB | 436 MB | 3.7x |
| arctic-embed-l-v2 | 2.2 GB | 580 MB | 436 MB | 3.7x |
| granite-embedding-97m-r2 | 362 MB (F16) | 106 MB | not built | 3.5x |
| granite-embedding-311m-r2 | 1.1 GB (F16) | 331 MB | not built | 3.5x |
| octen-0.6b | 1.6 GB | 607 MB | 397 MB | 2.7x |
| f2llm-v2-0.6b | 1.6 GB | 607 MB | 397 MB | 2.7x |
| jina-v5-nano | 585 MB | 219 MB | 164 MB | 2.7x |
| jina-v5-small | 1.6 GB | 607 MB | 397 MB | 2.7x |
| harrier-0.6b | 1.6 GB | 607 MB | 397 MB | 2.7x |
| harrier-270m | 741 MB | 279 MB | 231 MB | 2.7x |
| qwen3-embed-0.6b | 1.6 GB | 607 MB | 291 MB | 2.7x |

## Quantization Quality

Cosine similarity between F32 and quantized models (1.0 = identical).

| Model | Q8_0 | Q4_K |
|-------|------|------|
| all-MiniLM-L6-v2 | 0.9995 | 0.97 |
| gte-small | 0.9998 | 0.99 |
| arctic-embed-xs | 0.9999 | 0.99 |
| multilingual-e5-small | 0.9999 | 0.99 |
| pixie-rune-v1 | 0.9991 | 0.95 |
| arctic-embed-l-v2 | 0.9989 | 0.95 |
| granite-embedding-97m-r2 | 0.99958 | not built |
| granite-embedding-311m-r2 | 0.99976 | not built |
| octen-0.6b | 0.9995 | 0.97 |
| harrier-0.6b | 0.9999 | 0.99 |
| harrier-270m | 0.9998 | 0.99 |
| qwen3-embed-0.6b | 0.9996 | 0.97 |

| all-mpnet-base-v2 | 0.9998 | 0.99 |
| nomic-embed-text-v1.5 | 0.9994 | -- |
| gte-modernbert-base | 0.9999 | -- |
| bge-small-en-v1.5 | 0.9999 | 0.99 |
| bge-base-en-v1.5 | 0.9999 | 0.99 |
| bge-large-en-v1.5 | 0.9999 | 0.99 |
| all-MiniLM-L12-v2 | 0.9999 | 0.99 |
| mxbai-embed-large-v1 | 1.0000 | 0.99 |
| snowflake-arctic-embed-m | 0.9999 | 0.99 |
| snowflake-arctic-embed-l | 0.9999 | 0.99 |

Q8_0: all > 0.995. Q4_K: most > 0.95.

## BLAS Acceleration

OpenBLAS 0.3.26, Intel Xeon Skylake, 4 threads.

| Model | Quant | no-BLAS | BLAS | Speedup |
|-------|-------|---------|------|---------|
| gte-small | F32 | 114ms | 123ms | 0.9x |
| gte-small | Q8_0 | 116ms | 116ms | 1.0x |
| octen-0.6b | Q8_0 | 422ms | 410ms | 1.0x |

BLAS provides minimal benefit because quantized kernels use ggml's SIMD paths.
Use Q8_0 for CPU speed, GPU (CUDA/Vulkan) for maximum throughput.

## RAG Retrieval Quality

Retrieval quality on synthetic IR dataset (50 documents, 15 queries, graded relevance).
Model: all-MiniLM-L6-v2. Hardware: Intel Xeon Skylake, 4 threads, CPU-only.

| Engine | Model | MRR@10 | NDCG@10 | Recall@10 | Recall@100 | Time |
|--------|-------|--------|---------|-----------|------------|------|
| CrispEmbed F32 | all-MiniLM-L6-v2 | 1.0000 | 0.7846 | 0.7556 | 1.0000 | 0.63s |
| CrispEmbed F32 | bge-small-en-v1.5 | 1.0000 | 0.7470 | 0.6889 | 1.0000 | 3.19s |
| CrispEmbed Q8_0 | bge-small-en-v1.5 | 1.0000 | 0.7470 | 0.6889 | 1.0000 | 3.00s |

MRR@10 = 1.0: the most relevant document is always ranked first.
Recall@100 = 1.0: all relevant documents found within top-100.

**Key finding**: GGUF F32 embeddings produce identical retrieval quality to
HuggingFace (both are bit-identical, cos >= 0.999). Q8_0 quantization
(cos >= 0.995) should produce negligible retrieval quality degradation.

## Bi-Encoder Reranking

Bi-encoder reranking uses cosine similarity of L2-normalized embeddings.
CrispEmbed's `rerank_biencoder()` encodes query + all documents in a single
batch call, then computes dot products.

Example (all-MiniLM-L6-v2, query: "What is machine learning?"):

| Document | Score |
|----------|-------|
| Machine learning is a subset of artificial intelligence. | 0.7124 |
| Neural networks learn patterns from training data. | 0.5897 |
| The weather in Paris is mild in spring. | 0.0153 |

Correct ranking with clear separation between relevant and irrelevant docs.

## Feature Parity with fastembed-rs

| Feature | CrispEmbed | fastembed-rs | Winner |
|---------|-----------|-------------|--------|
| Single-text latency (MiniLM, M1 Metal) | 3.6 ms | 3.8 ms | CrispEmbed |
| Batch throughput (10 texts, M1 Metal) | 787 t/s | 528 t/s | CrispEmbed |
| Binary size | ~20 MB | ~500 MB (ONNX) | CrispEmbed |
| Quantization quality (Q8_0) | cos > 0.995 | INT8 varies | CrispEmbed |
| Model count (embedding) | 37 | 49 | fastembed-rs |
| Model count (reranker) | 7 | 20 | fastembed-rs |
| Sparse retrieval | BGE-M3 + SPLADE | SPLADE + BGE-M3 | Tie |
| ColBERT multi-vector | Yes | No | CrispEmbed |
| Image embedding | SigLIP + BidirLM-Omni | 5 models | Tie |
| Prompt prefix | Yes | Yes | Tie |
| Bi-encoder reranking | Yes | Yes | Tie |
| GPU backends | CUDA/Metal/Vulkan | ONNX EP | Tie |

## Notes

- CrispEmbed uses ggml inference with SIMD-optimized quantized matmul
- Graph and work buffers are reused across calls (3.2x throughput improvement)
- When built with CUDA/Vulkan/Metal, `ggml_backend_sched` auto-dispatches to GPU
- Decoder models (Qwen3/Gemma3) are 10-15x slower than encoders (28 layers vs 6)
- Server mode eliminates model loading overhead (~100-300ms per cold start)
- Prompt prefix adds negligible overhead (string concatenation before tokenization)
- Bi-encoder reranking cost = 1 batch encode + N dot products (O(N*dim) after encode)

## Latency Benchmark (Intel Xeon Skylake, CPU, 4 threads)

Single-text and batch (10 texts) encoding latency via Python ctypes wrapper.

| Model | Dim | Single (ms) | Batch 10 (ms) | Texts/s |
|-------|-----|------------|---------------|---------|
| all-MiniLM-L6-v2 | 384 | 12.7 | 48.8 | 205 |
| bge-small-en-v1.5 | 384 | 34.5 | 537.3 | 19 |
| all-MiniLM-L12-v2 | 384 | 443.0 | 239.0 | 42 |
| bge-base-en-v1.5 | 768 | 124.4 | 543.4 | 18 |
| all-mpnet-base-v2 | 768 | 66.4 | 292.9 | 34 |
| nomic-embed-text-v1.5 | 768 | 88.9 | 310.2 | 32 |

MiniLM-L6 is fastest (6.4ms single). NomicBERT is efficient for its size
(768d in 41.4ms). Batch throughput varies due to model size and graph complexity.

## Head-to-Head: CrispEmbed vs FastEmbed (ONNX)

Same models, same texts, same hardware (Intel Xeon, 4 threads, CPU-only).

| Model | Engine | Single (ms) | Batch 10 (ms) | Texts/s |
|-------|--------|------------|---------------|---------|
| all-MiniLM-L6-v2 | **CrispEmbed** | **6.4** | **23.6** | **424** |
| all-MiniLM-L6-v2 | FastEmbed | 60.8 | 255.9 | 39 |
| bge-small-en-v1.5 | CrispEmbed | 14.7 | 55.4 | 181 |
| bge-small-en-v1.5 | **FastEmbed** | **8.4** | **41.2** | **243** |
| snowflake-arctic-embed-m | CrispEmbed | 40.1 | **126.5** | **79** |
| snowflake-arctic-embed-m | FastEmbed | **33.3** | 127.5 | 78 |
| all-mpnet-base-v2 | CrispEmbed | 31.2 | 138.7 | 72 |
| nomic-embed-text-v1.5 | CrispEmbed | 41.4 | 150.6 | 66 |

**CrispEmbed vs FastEmbed**: CrispEmbed is **9.5x faster** on MiniLM-L6 (our most
optimized model: QKV fusion + flash attention + graph caching). On 12-layer models
(BGE-small, Arctic-M), FastEmbed's ONNX Runtime graph optimization (Level3 JIT,
operator fusion) gives it a slight edge. On Arctic-M batch, they're tied.

**Cosine similarity**: CrispEmbed vs FastEmbed cos=0.999999-1.000000 on all models.

## Per-Step Benchmark Instrumentation

Every runtime in CrispEmbed has opt-in per-step timing controlled by environment
variables. Set `CRISPEMBED_<MODULE>_BENCH=1` to get `[module-bench]` lines on
stderr showing millisecond timing for each processing phase (preprocess, encoder,
decoder, postprocess, per-tile, per-decode-step, total).

Zero overhead when unset — the flag is read once at init and stored as a bool.

| Category | Env vars |
|---|---|
| Embedding | `CRISPEMBED_CRISPEMBED_BENCH`, `VIT_EMBED`, `CNN_EMBED`, `CLIP_TEXT`, `LFM2_EMBED`, `DECODER_EMBED` |
| OCR detect | `CRISPEMBED_OCR_DETECT_BENCH`, `LAYOUT_DETECT`, `SURYA_DET`, `CC_DETECT` |
| OCR recognize | `CRISPEMBED_PARSEQ_BENCH`, `BTTR`, `HMER`, `POSFORMER`, `TESSERACT`, `PIX2STRUCT`, `MIXTEX`, `MATH_OCR`, `PPFN`, `PPFN_L` |
| VLM/LLM OCR | `CRISPEMBED_QWEN2VL_BENCH`, `GOT_OCR`, `GLM_OCR`, `GRANITE_OCR`, `INTERNVL2`, `DEEPSEEK_OCR2`, `LIGHTONOCR`, `SMOLDOCLING` |
| Super-resolution | `CRISPEMBED_ESRGAN_BENCH`, `DAT_SR`, `HAT_SR`, `PAN_SR`, `SAFMN_SR`, `SWINIR_SR`, `TBSRN_SR`, `TEXT_SR` |
| Denoise/restore | `CRISPEMBED_NAFNET_BENCH`, `SCUNET`, `RESTORMER`, `INSTRUCTIR`, `ADAIR` |
| NER/KIE | `CRISPEMBED_GLINER_BENCH`, `BERT_NER`, `LILT_KIE` |
| Pipeline | `CRISPEMBED_OCR_PIPELINE_BENCH`, `OCR_ORCH`, `KIE_PIPELINE`, `SCAN_CLEANUP`, `TABLE_PARSE` |
| Misc | `CRISPEMBED_PCS_BENCH`, `FIREREDPUNC`, `BIDIRLM_AUDIO`, `BIDIRLM_VISION`, `FACE_ALIGN`, `DEWARP`, `TPS_LOCNET` |

Example:
```
CRISPEMBED_PARSEQ_BENCH=1 ./crispembed-cli ocr image.png
# [parseq-bench] preprocess: 0.3 ms
# [parseq-bench] encoder graph (12 layers): 4.2 ms
# [parseq-bench] decoder CA K/V precompute: 0.1 ms
# [parseq-bench] decoder total (5 steps): 1.8 ms
# [parseq-bench] total: 6.4 ms
```

---

## Runtime Optimization Audit (June 2026)

Full line-by-line code review of all ~57K lines of C++ across 60+ runtime files.
Covers every runtime in the codebase: what optimizations are already in place,
and where the biggest opportunities remain.

### Methodology

Every `.cpp` and `.h` file in `src/` was read in full. Findings are grouped by
runtime category. "Existing" means the optimization is already implemented;
"Missing" means there is a concrete opportunity for improvement.

---

### 1. Core Shared Infrastructure (`src/core/`)

**Files**: `cpu_ops.h` (292L), `vlm_attention.h` (222L), `bpe.h` (218L),
`gguf_loader.cpp/.h` (487L), `mel.cpp/.h` (416L)

#### Already optimized

| Technique | Where | Notes |
|-----------|-------|-------|
| Memory-mapped model loading | `gguf_loader.cpp` | `mmap`/`MapViewOfFile`, zero-copy weight access |
| Double-precision accumulator | `cpu_ops.h` LayerNorm/RMSNorm | Prevents float cancellation on long vectors |
| GPU-safe dequantization | `cpu_ops.h` `to_f32()` | Uses `ggml_backend_tensor_get`, works for Metal/CUDA tensors |
| Quantized weight support | `cpu_ops.h` `to_f32()` | Handles F32/F16/Q4/Q8 via `ggml_get_type_traits()->to_float` |
| In-place activations | `cpu_ops.h` | `silu_inplace`, `hardswish_inplace`, `relu6_inplace` |
| Numerically-stable softmax | `cpu_ops.h` | Max-subtract before `expf` |
| GQA support | `vlm_attention.h` | `kv_repeat = n_heads / n_kv_heads` reduces KV memory |
| Lazy byte_encoder table | `bpe.h` | Built once, cached in static |
| Two-pass GGUF loading | `gguf_loader.cpp` | Metadata pass is no-alloc |
| Mel spectrogram parameterization | `mel.cpp` | Single code path for 9 audio models |

#### Opportunities

| Priority | Location | Issue | Impact |
|----------|----------|-------|--------|
| **P0** | `cpu_ops.h` `linear_cpu` | No SIMD — naive scalar matmul O(N*M) | 4-8x with AVX2/NEON |
| **P0** | `cpu_ops.h` `linear_cpu` (tensor overload) | Re-dequantizes full weight matrix every call — no caching | Eliminates thousands of redundant alloc+dequant per decode |
| **P1** | `vlm_attention.h` `apply_rope` | `powf`/`cosf`/`sinf` computed per-element; no frequency table precomputation | 3-5x on RoPE-heavy models |
| **P1** | `mel.cpp` mel projection | Naive triple-loop matmul (T*128*201 ≈ 38M scalar MACs) | 10-20x with SIMD/BLAS |
| **P1** | `cpu_ops.h` `conv2d_cpu` | 6-nested scalar loops, no im2col or tiling | 5-10x with im2col+GEMM |
| **P2** | `vlm_attention.h` `gqa_attn_step` | `std::vector<float> scores(n_kv)` allocated per-head inside loop | Remove per-head allocation churn |
| **P2** | `vlm_attention.h` `swiglu_ffn` | Allocates two intermediate_dim vectors every call | Pre-allocate once |
| **P2** | `mel.cpp` STFT loop | Each frame's FFT is independent — no OpenMP parallelism | Linear speedup with core count |
| **P2** | `gguf_loader.cpp` mmap | No `madvise(MADV_SEQUENTIAL)` hint | Better kernel readahead on cold loads |
| **P3** | `gguf_loader.h` tensor map | `std::map` instead of `std::unordered_map` | ~2-5x faster tensor lookups |
| **P3** | `bpe.h` BPE merge loop | O(N^2) in symbol count; `vector::erase` from middle | Priority queue → O(N log N) |
| **P3** | `cpu_ops.h` `layernorm2d_cpu` | Iterates `(y, x, c)` but accesses stride-H*W — cache-hostile | NHWC layout or transpose |

---

### 2. VLM OCR Runtimes (Vision-Language Models)

**Files**: `qwen2vl_ocr` (2432L), `deepseek_ocr2` (1719L), `internvl2_ocr` (1715L),
`granite_vision_ocr` (614L), `got_ocr` (1455L), `glm_ocr` (1216L),
`lightonocr` (1365L), `smoldocling_ocr` (1011L), `pix2struct` (690L)

#### Optimization maturity ranking

> **REFRESH 2026-07-20 (code-verified):** every VLM decoder below now DEFAULTS to a
> ggml F16-KV GPU decode path; the `core_vlm` CPU-scalar decode survives only as a
> gated fallback (`CRISPEMBED_*_SCALAR` / `use_ggml` guards). The pre-refresh columns
> claiming "F32 CPU vectors" / "CPU scalar (core_vlm)" / "no KV cache" for
> qwen2vl/smoldocling/granite/pix2struct were STALE. Corrected:

| Rank | Runtime | LLM decode (default) | KV cache | GPU |
|------|---------|----------------------|----------|-----|
| 1 | **internvl2_ocr** | ggml flash_attn | F16 ggml tensor (zero-copy) | Yes |
| 2 | **glm_ocr** | ggml flash_attn (monolithic) | F16 ggml tensor | Yes |
| 3 | **got_ocr** | ggml flash_attn | F16 ggml tensor | Yes |
| 4 | **qwen2vl_ocr** | ggml + `build_decode_step_graph` | **F16 ggml backend** (`alloc_kv_cache`) | Yes |
| 5 | **lightonocr** | ggml flash_attn | F16 ggml persistent (`ggml_cpy`) | Yes |
| 6 | **deepseek_ocr2** | ggml per-layer graphs (default); persistent single-graph decode implemented but **opt-in** `DS2_FAST_DECODE=1` — measured NO win 2026-08-05, see the T14 section | **F32** default (`alloc_ds_kv_cache`); F16 opt-in `DS2_KV_F16=1` | Yes |
| 7 | **smoldocling_ocr** | `sd_run_llm_body` ggml (default; `use_ggml`) | **F16 ggml backend**; core_vlm = fallback | Yes |
| 8 | **granite_vision_ocr** | `gv_run_llm_body` ggml (default; diff cos 0.9999) | **F16 ggml backend**; core_vlm = opt-out | Yes |
| 9 | **pix2struct** | CPU scalar + DequantCache | KV cache (Phase 2) — CPU, GPU port low-priority | No |

#### Already optimized (best practices found in at least one runtime)

| Technique | Where | Notes |
|-----------|-------|-------|
| Flash attention (`ggml_flash_attn_ext`) | internvl2, glm, got, lightonocr, smoldocling (vision) | Fused Q@K+softmax+V in single op |
| F16 KV cache in ggml tensors | internvl2, glm, got, lightonocr | Zero-copy view+cpy writes, halves memory |
| Prefill/decode separation | qwen2vl, internvl2, deepseek, got, glm, lightonocr | Full-sequence prefill, single-token decode |
| Fused QKV projection | qwen2vl | Single matmul for Q/K/V |
| `ggml_backend_sched` GPU dispatch | qwen2vl, internvl2, deepseek, got, glm | Automatic CPU/GPU placement |
| Precomputed RoPE tables | qwen2vl (2D), got, lightonocr (2D) | Host-side cos/sin computed once |
| Monolithic vision graph | glm, lightonocr | All layers in single graph (vs per-layer rebuild) |
| Skip logits during prefill | smoldocling | Skips V-sized LM head matmul for non-last tokens |
| Lazy expert dequant (MoE) | deepseek | Only dequantizes selected experts |
| Multi-threaded MoE dispatch | deepseek | Token-parallel expert evaluation |
| Periodic wbufs.clear() | smoldocling | Prevents unbounded dequant buffer growth |

#### Opportunities

| Priority | Issue | Affected runtimes | Impact |
|----------|-------|-------------------|--------|
| ~~**P0**~~ DONE | ~~Adopt F16 ggml KV cache (internvl2 pattern)~~ **— landed; all VLM decoders default to ggml F16-KV GPU decode (verified 2026-07-20, see maturity table)** | qwen2vl, deepseek, smoldocling, granite | Eliminates O(seq_len) per-step re-upload; halves memory |
| **P0** | Use `ggml_flash_attn_ext` for LLM decode | qwen2vl, deepseek | qwen2vl uses manual Q@K+softmax+V; deepseek uses per-layer graphs |
| **P0** | Move granite to ggml graphs | granite_vision_ocr | Entire engine is CPU-scalar — 10-50x potential speedup |
| **P0** | Implement batched prefill for smoldocling/granite | smoldocling, granite | Token-at-a-time through 30-40 LLM layers is catastrophic |
| **P0** | Move pix2struct to ggml graphs + add KV cache | pix2struct | Fully scalar, no KV cache, O(T^2) recompute per step |
| **P1** | Patch embedding conv → ggml matmul | ALL 9 runtimes | Every runtime uses scalar 6-deep nested loops |
| **P1** | Move deepseek Qwen2 encoder to ggml | deepseek_ocr2 | 24-layer bidirectional transformer entirely CPU-scalar |
| **P1** | Single multi-layer LLM graph (vs per-layer) | deepseek | 12 graph builds per decode token |
| **P1** | Cache dequantized weights | qwen2vl, deepseek, lightonocr, got, smoldocling, granite | `to_f32()` re-dequantizes same weights every call |
| **P1** | Scalar CPU downsample/merger → ggml | glm, got | Conv+matmul neck/projector still scalar |
| **P2** | InternVl2: native GQA in flash_attn (skip ggml_repeat) | internvl2 | Avoids duplicating KV heads before attention |
| **P2** | Vision tiles: batch multiple tiles in one graph | internvl2 | Currently sequential per-tile graph allocation |
| **P2** | Token embed via direct read (not mini-graph) | qwen2vl | Building a full ggml graph for one `ggml_get_rows` |
| **P2** | Decode graph reuse (not rebuild per step) | deepseek | Graph structure is identical across steps |
| **P2** | Windowed attention in qwen2vl vision | qwen2vl | window_size=112 declared but unused in graph |
| **P3** | LM head on CPU → ggml for deepseek final norm+head | deepseek | (D=1280, V=129280) scalar matmul for lm_head |
| **P3** | F32 causal mask → F16 | qwen2vl | internvl2 already uses F16 mask |

---

### 3. Math/Formula OCR Runtimes

**Files**: `math_ocr` (1241L), `mixtex_ocr` (1198L), `bttr_ocr` (1134L),
`hmer_ocr` (1013L), `posformer_ocr` (946L), `ppformulanet_ocr` (944L),
`ppformulanet_l_ocr` (1474L)

#### Encoder optimization ranking

| Rank | Runtime | Encoder type | Approach |
|------|---------|-------------|----------|
| 1 | **ppformulanet_l_ocr** | SAM-ViT | ggml graph, batched windows, precomputed RPE |
| 2 | **math_ocr** | DeiT | ggml graph |
| 3 | **ppformulanet_ocr** | HGNetv2 (CNN) | Scalar CPU with shared `core/cpu_ops.h` helpers |
| 4 | **mixtex_ocr** | Swin-Tiny | Scalar CPU with shared helpers |
| 5 | **bttr_ocr** | DenseNet | Scalar CPU with duplicated local helpers |
| 5 | **posformer_ocr** | DenseNet | Scalar CPU with duplicated local helpers |
| 5 | **hmer_ocr** | DenseNet-121 | Scalar CPU with duplicated local helpers |

#### Already optimized

| Technique | Where | Notes |
|-----------|-------|-------|
| ggml graph encoder (SIMD matmuls) | ppformulanet_l, math_ocr | Vision layers computed via ggml graphs |
| Batched windows in ggml graph | ppformulanet_l | All 16 windows processed in parallel |
| Precomputed RPE lookup tables at init | ppformulanet_l | `get_rel_pos()` done once, stored per-layer |
| Cross-attention K/V pre-computation | ALL 7 runtimes | Projected once from encoder output before decode loop |
| Self-attention KV cache | ALL except hmer (GRU) | Per-layer growing cache for autoregressive decoding |
| Dequant cache | math_ocr, bttr, hmer, posformer | Avoids redundant F16→F32 conversion |
| Pre-cached embeddings before decode loop | math_ocr | Token + position tables dequantized once |
| Folded BatchNorm | hmer | BN params pre-folded into conv weights |
| Beam search | bttr_ocr | Full beam search with length normalization |
| Bilinear image resize | bttr, hmer, posformer | Higher quality than nearest-neighbor |
| GELU as tanh approximation | ppformulanet | Avoids expensive `erf()` |

#### Opportunities

| Priority | Issue | Affected runtimes | Impact |
|----------|-------|-------------------|--------|
| **P0** | DenseNet encoder → ggml graphs or im2col+GEMM | bttr, posformer, hmer | All convolutions are 7-nested-loop scalar — dominates runtime |
| **P0** | Swin encoder → ggml graphs | mixtex | 12500-token window attention is scalar O(N^2*D) per window |
| **P0** | HGNetv2 CNN encoder → ggml | ppformulanet | 57M-param CNN at 384x384 via scalar `conv2d_cpu` |
| **P1** | Add beam search | mixtex, math_ocr, hmer, posformer, ppformulanet, ppformulanet_l | Only bttr has it; beam width=3 helps math OCR accuracy significantly |
| **P1** | Migrate duplicated helpers to `core/cpu_ops.h` | bttr, hmer, posformer | ~300 lines of duplicated conv2d/relu/layernorm/linear in each |
| **P1** | Cache dequantized weights at init | mixtex, ppformulanet, ppformulanet_l | `to_f32()` called per-block per-call, same weights every time |
| **P1** | ppformulanet_l: scalar decoder → ggml | ppformulanet_l | Encoder is ggml-optimized but 8-layer D=512 decoder is still scalar |
| **P2** | Pre-compute attention masks (shifted windows) | mixtex | Recomputed from scratch per block — deterministic for fixed dims |
| **P2** | Pre-compute 2D positional encoding | bttr, posformer | sinf/cosf/powf recomputed every inference call |
| **P2** | ggml context reuse across layers | ppformulanet_l | New 8MB context allocated and freed for each of 12 layers |
| **P2** | Global dequant cache → per-context | math_ocr | Global static `unordered_map` is thread-unsafe |
| **P2** | Nearest-neighbor → bilinear resize | math_ocr, mixtex, ppformulanet, ppformulanet_l | 4 of 7 runtimes use nearest-neighbor |
| **P3** | bttr beam search: top-K selection instead of full sort | bttr | O(V*beam_width) candidates created then sorted |
| **P3** | hmer coverage conv per step | hmer | conv2d(256,256,3x3) per decoder step — expensive attention mechanism |

---

### 4. Embedding & NER Runtimes

**Files**: `decoder_embed` (1638L), `vit_embed` (674L), `clip_text_embed` (433L),
`cnn_embed` (1323L), `lfm2_embed` (722L), `bert_ner` (321L), `gliner_ner` (1703L),
`lilt_kie` (676L), `fireredpunc` (802L), `bidirlm_vision` (692L), `bidirlm_audio` (129L)

#### Already optimized

| Technique | Where | Notes |
|-----------|-------|-------|
| Flash attention | vit_embed, clip_text, lfm2_embed, gliner_ner (GQA), fireredpunc, decoder_embed (batch path) | `ggml_flash_attn_ext` |
| Fused QKV weights | vit_embed, bidirlm_vision | Q/K/V concatenated at load → single matmul |
| Batched encoding with prefix sharing | decoder_embed | Detects shared prefix, deduplicates (B-1)*P tokens |
| F16 attention mask | decoder_embed, clip_text | Halves mask memory |
| Fused soft_max_ext | decoder_embed (batch), bidirlm_vision | Scale + mask + softmax in one ggml op |
| BN folding at load | cnn_embed | BatchNorm params pre-folded into affine scale+shift |
| LoRA hot-swap | decoder_embed | CPU-side merge/unmerge with lazy base weight snapshot |
| Pre-cached BiLSTM weights | gliner_ner | Dequantized to F32 once at init |
| DeBERTa disentangled attention | gliner_ner | Full c2c + c2p + p2c implementation |
| Pre-computed bilinear position interpolation | bidirlm_vision | Corner indices + weights baked once per encode |
| Pre-computed 2D RoPE cos/sin | bidirlm_vision | Full tables on CPU, passed as graph inputs |
| Generic ONNX graph replayer | cnn_embed | Can replay arbitrary CNN topologies from metadata |
| `ggml_gallocr` reuse | lfm2_embed | Allocator stored on context, reused across calls |
| Gemma3 numerical stability | decoder_embed | RMSNorm output clamped to [-1000, 1000] for F16 safety |
| Delegates to CrispASR encoder | bidirlm_audio | Reuses existing optimized audio encoder |

#### Opportunities

| Priority | Issue | Affected runtimes | Impact |
|----------|-------|-------------------|--------|
| **P0** | No flash attention in single-text path | decoder_embed | Uses manual Q@K+softmax+V; only batch path uses flash_attn |
| **P1** | BiLSTM is fully scalar | gliner_ner | 4*512*1024 + 4*512*512 ≈ 3M MACs per timestep, no SIMD/BLAS |
| **P1** | Layer fusion matmuls are scalar | gliner_ner | [1024, 1024] output projection per token via scalar loops |
| **P1** | Graph rebuilt every call | ALL 11 runtimes | Graph structure is identical for same seq_len; should cache |
| **P1** | No flash attention | bidirlm_vision, lilt_kie | Manual Q@K+softmax+V despite amenable structure |
| **P2** | Fuse QKV in clip_text | clip_text_embed | 3 separate matmuls where 1 would suffice |
| **P2** | Scalar L2 normalization | decoder_embed, vit_embed, lfm2_embed, bidirlm_audio | Could use SIMD or ggml ops |
| **P2** | Scalar dense projection matmul | decoder_embed | Triple-nested scalar loop for post-pooling projection |
| **P2** | DeBERTa relative position expansion O(T^2*H) | gliner_ner | Creates [H, T*T] F32 tensor on CPU every call; T=200 → 117MB |
| **P2** | `ggml_gallocr` rebuilt per call | vit_embed, clip_text, cnn_embed, fireredpunc, gliner_ner, decoder_embed | Only lfm2_embed reuses the allocator |
| **P3** | No batched encode API | vit_embed, clip_text, lfm2_embed, bert_ner, gliner_ner, lilt_kie, fireredpunc | Single-input only |
| **P3** | Conv1D kernel cast every call | lfm2_embed | `ggml_cast` adds a graph node per invocation; pre-cast at load |
| **P3** | F32 attention mask | bidirlm_vision | F16 would halve the 20MB mask for 2304 tokens |
| **P3** | WordPiece re-tokenization for word counting | fireredpunc | Re-tokenizes each word to count subtokens; track during initial pass |

---

### 5. Super-Resolution & Image Restoration Runtimes

**Files**: `dat_sr` (1396L), `hat_sr` (945L), `swinir_sr` (695L), `esrgan_sr` (252L),
`safmn_sr` (438L), `pan_sr` (383L), `tbsrn_sr` (533L), `text_sr` (670L),
`nafnet_denoise` (564L), `scunet_denoise` (792L), `restormer` (749L),
`instructir` (469L), `adair` (944L)

#### Already optimized

| Technique | Where | Notes |
|-----------|-------|-------|
| Tiling with Hann-window overlap blending | dat_sr, hat_sr, swinir_sr, pan_sr, text_sr, restormer | Raised-cosine window prevents seam artifacts |
| Dequant cache | dat_sr | `dequant_cache` avoids re-dequantizing the same tensor |
| Ping-pong buffer reuse | esrgan_sr, nafnet, text_sr | Swap buf_a/buf_b to avoid allocation per layer |
| BatchNorm fusion at inference | dat_sr, tbsrn_sr | Pre-computed `scale = weight / sqrt(var+eps)` |
| GPU-safe tensor reads | 12 of 13 runtimes | `ggml_backend_tensor_get()` instead of `tensor->data` |
| Transposed attention (C×C not HW×HW) | restormer, adair | Efficient for high-resolution images |
| Scratch buffer reuse | safmn_sr, swinir_sr, tbsrn_sr, nafnet, text_sr | Pre-allocated tmp buffers passed to blocks |
| Bicubic upscale with Keys kernel | text_sr | Proper reconstruction filter |
| Single-tile fast path | dat_sr | Skips tiling overhead for small images |
| FORCE_CPU env var | Most runtimes | Debug override for backend selection |

#### Opportunities

| Priority | Issue | Affected runtimes | Impact |
|----------|-------|-------------------|--------|
| **P0** | No SIMD anywhere — all conv/linear/attention is scalar | ALL 13 runtimes | conv2d accounts for ~80% of compute; 5-10x with SIMD |
| **P0** | No weight dequant caching | 12 of 13 (all except dat_sr) | Re-dequant same weights per-block per-image |
| **P0** | Per-pixel vector allocations in scunet | scunet_denoise | `std::vector<float>` allocated per spatial position in LN and MLP — 100K+ heap allocs per Swin block |
| **P1** | No tiling support | esrgan, safmn, nafnet, scunet, instructir, adair | OOM or poor cache behavior for images >512px |
| **P1** | Batch linear/GEMM instead of per-token calls | dat_sr, swinir_sr, hat_sr, scunet | QKV as N separate `linear_cpu` calls → one GEMM |
| **P1** | Redundant CHW↔HWC layout conversions | dat_sr, hat_sr | 30-50 full-image transposes per forward pass |
| **P2** | Pre-compute attention masks and position biases | hat_sr, swinir_sr, dat_sr | Rebuilt per tile despite being deterministic for fixed size |
| **P2** | `ctx->get()` unbounded wbufs growth | hat_sr, swinir_sr, pan_sr, text_sr, nafnet, restormer, instructir, adair | Appends new dequantized vector every call, never reuses |
| **P2** | Fuse BatchNorm into conv weights at model load | dat_sr, tbsrn_sr | Currently applied as separate pass after conv |
| **P2** | instructir SCA weight dequant inside per-channel loop | instructir | Re-dequantizes entire weight matrix C times |
| **P3** | scunet conv_transpose2d scatter-add | scunet | Writes to output with random access — cache-unfriendly |
| **P3** | PE2D recomputed every SRB iteration | tbsrn_sr | `tbsrn_pe2d(64, ...)` called 5 times with identical params |
| **P3** | restormer rst_layernorm_bf computes variance twice | restormer | First sum-of-squares pass is dead work |
| **P3** | adair FFT zero-pads to next power of 2 | adair | 129→256, wastes ~2x compute; mixed-radix would help |

---

### 6. Detection, Pipelines & Utilities

**Files**: `layout_detect` (1872L), `surya_det` (1341L), `ocr_detect` (947L),
`parseq_ocr` (810L), `tesseract_lstm` (663L), `ocr_pipeline` (169L),
`ocr_orchestrator` (940L), `ocr_render` (600L), `table_parse` (393L),
`kie_pipeline` (316L), `cc_detect` (280L), `image_preprocess` (520L),
`classical_preproc` (690L), `face_align` (193L), `dewarp` (309L),
`scan_cleanup` (572L), `morph_fast` (312L), `tps_warp` + `tps_locnet` (508L),
`pdf_info` (739L), `pcs` (817L), `tokenizer*` (764L)

#### Already optimized

| Technique | Where | Notes |
|-----------|-------|-------|
| ggml graph for full backbone+encoder | layout_detect, ocr_detect | ResNet + FPN + attention all in one graph |
| ggml graph for ViT encoder | parseq_ocr | 12-layer ViT with flash_attn |
| ggml graph for XLM-RoBERTa | pcs | 12-layer encoder with flash_attn |
| Hybrid graph + scalar forward | surya_det | Stages 0-2 ggml graph, LiteMLA scalar |
| Flash attention | layout_detect (AIFI), parseq_ocr, pcs | `ggml_flash_attn_ext` where applicable |
| Dequant cache | parseq_ocr | Maps tensor data pointers to F32 buffers |
| All weights dequanted at load | tesseract_lstm | Zero runtime dequant cost |
| BN pre-folded into conv | surya_det | Eliminates BN arithmetic at runtime |
| Union-find with path compression | cc_detect | O(α(N)) CC labeling |
| 32-pixel word-level morphology | morph_fast | 32x throughput vs per-pixel ops |
| Integral images for Sauvola binarization | scan_cleanup, classical_preproc | O(1) per-pixel mean/variance |
| Separable bicubic resize with anti-aliasing | image_preprocess | Matches torchvision quality |
| `__builtin_popcount` for row sums | classical_preproc | Hardware-accelerated bit counting |
| partial_sort for top-K queries | layout_detect | Avoids full sort |
| `std::nth_element` for thresholds | surya_det | O(N) partial sort |
| Viterbi DP for SentencePiece | tokenizer_spm | Optimal segmentation |
| Convex hull + rotating calipers | ocr_detect | Min-area rotated rectangles |
| Lazy engine loading | ocr_orchestrator | Unused engines have zero overhead |
| Early exit for flat pages | dewarp | Skip warp if max_disp < 2px |
| DPI estimation via PDF metadata | ocr_orchestrator | Auto-selects SR tier |
| Pre-computed resampling weights | image_preprocess | Index + weight arrays built once per dimension |
| Gaussian elimination with partial pivoting | tps_warp | Robust TPS solve |
| Cross-attention K/V pre-computation | parseq_ocr | Computed once from encoder output |

#### Opportunities

| Priority | Issue | Affected runtimes | Impact |
|----------|-------|-------------------|--------|
| **P0** | Deformable cross-attention is CPU-scalar | layout_detect | 6-nested-loop bilinear sampling — dominates decoder runtime |
| **P1** | LSTM gates have no SIMD | tesseract_lstm | Hot inner dot-product loop is unvectorized |
| **P1** | LiteMLA attention is CPU-scalar | surya_det | O(N^2 * head_dim) scalar matmuls (stubbed graph path) |
| **P1** | Sequential region recognition | ocr_pipeline, table_parse | Each crop recognized individually — batch into single encoder pass |
| **P1** | Image loaded from disk multiple times | ocr_orchestrator | stbi_load called N times for N engine attempts on same image |
| **P1** | Cleaned image written to temp PNG then re-loaded | ocr_orchestrator | PNG encode/decode round-trip; pass pixel buffer directly |
| **P1** | min_pool/max_pool are O(K^2) per pixel | scan_cleanup | K=51 means ~2500 comparisons/pixel; deque-based sliding window → O(1) amortized |
| **P2** | Otsu threshold duplicated 6 times | table_parse, cc_detect, classical_preproc, scan_cleanup, dewarp | Extract to `core/` shared utility |
| **P2** | Per-step allocations in parseq AR decode | parseq_ocr | ~15 vectors allocated/freed per decode step × 26 steps |
| **P2** | TPS warp evaluates all N control points per pixel | tps_warp | Coarse grid + bilinear interpolation of displacement field |
| **P2** | No multithreading in pixel-level ops | image_preprocess, dewarp, scan_cleanup, face_align | All pixel loops single-threaded despite accepting n_threads |
| **P2** | BPE merge is O(N^2 * V) | tokenizer_bpe | Priority queue → O(N log V) |
| **P2** | Locnet weights re-dequantized every predict call | tps_locnet | Cache F32 weights at load time |
| **P2** | Hough voting O(edge_pixels * n_angles) | scan_cleanup | Quadratic for dense images |
| **P3** | WordPiece uses linear scan for longest match | tokenizer | Trie would be O(len) |
| **P3** | PDF parser loads entire file into memory | pdf_info | mmap would handle large PDFs |
| **P3** | Debug fprintf unconditionally in production | layout_detect, surya_det, ocr_detect | Should be gated behind verbosity level |
| **P3** | `std::vector<float>` return-by-value in hot paths | surya_det, parseq_ocr, face_align | Allocates and copies large buffers; use pre-allocated workspaces |

---

### Cross-Cutting Summary

#### What the codebase does well

1. **ggml graph acceleration** — The best runtimes (internvl2, glm_ocr, decoder_embed batch)
   use ggml compute graphs for all heavy math, getting SIMD-optimized matmuls and
   automatic GPU dispatch for free.

2. **Flash attention** — Used in 10+ runtimes for fused Q@K+softmax+V with proper scaling.

3. **Quantized model support** — Universal GGUF loading handles F32/F16/Q8_0/Q4_K
   transparently. Cosine similarity vs F32 is >0.995 for Q8_0 across all models.

4. **Memory-mapped weights** — `mmap`/`MapViewOfFile` in `gguf_loader.cpp` avoids
   copying multi-GB model files into userspace.

5. **KV cache** — Most autoregressive decoders implement proper KV caching with
   incremental append (best: internvl2's F16 ggml-resident zero-copy cache).

6. **Tiling with overlap blending** — SR/restoration runtimes handle arbitrary image
   sizes via overlapping tiles with Hann-window blending.

#### Top 10 highest-impact optimization opportunities

| # | Opportunity | Scope | Status |
|---|------------|-------|--------|
| 1 | **SIMD in `core/cpu_ops.h` helpers** | 30+ runtimes | **DONE** — `dot_product()` AVX2+FMA/NEON, 710 FMA instructions |
| 2 | **Dequantized weight caching** | ~40 runtimes | **DONE** — `DequantCache` in core; migrated smoldocling + granite |
| 3 | **Adopt F16 ggml KV cache** (internvl2 pattern) | 6 VLM decoders | Partial — pix2struct (F32 vector), lightonocr, granite, smoldocling, qwen2vl done |
| 4 | **Flash attention everywhere** | 5 runtimes | **DONE** (3/5) — decoder_embed, bidirlm_vision, pix2struct. lilt_kie incompatible (BiACM). deepseek pending. |
| 5 | **Move remaining scalar encoders to ggml graphs** | 7 encoders | **DONE** (pix2struct). DenseNet (bttr/posformer/hmer) and Swin (mixtex) remain. |
| 6 | **Batched prefill for VLM decoders** | smoldocling, granite | **DONE** — smoldocling F16 KV + batched prefill, granite projector+LLM graphs |
| 7 | **Graph caching** | All 60+ runtimes | Pending (architectural) |
| 8 | **Pre-compute RoPE frequency tables** | core_vlm users | **DONE** — `RoPEFreqTable`; migrated smoldocling + granite |
| 9 | **Batch linear → GEMM** in SR attention | 5 SR runtimes | **DONE** — dat_sr, swinir_sr, hat_sr, scunet, mixtex via `linear_batch_cpu` |
| 10 | **Eliminate per-step heap allocations** | 12 runtimes | **DONE** — pix2struct, bttr, posformer, hmer, math, parseq, mha_1q_cpu, vlm_attention, layernorm2d |
| 11 | **BPE tokenizer O(N²) → O(N log N)** | bpe.h + tokenizer_bpe | **DONE** — linked list + priority queue |
| 12 | **`std::unordered_map` for tensor lookup** | gguf_loader + 14 files | **DONE** — O(1) avg lookups |

#### Architectural recommendations

1. ~~Centralize dequant caching in `core/cpu_ops.h`~~ — **DONE**: `DequantCache` struct
   added. Migrated in smoldocling_ocr and granite_vision_ocr.

2. ~~Add SIMD to `linear_cpu` and `conv2d_cpu`~~ — **DONE** for `linear_cpu` (AVX2+FMA
   and NEON via `dot_product()`). `conv2d_cpu` still scalar (needs im2col restructure).

3. **Standardize KV cache on internvl2 pattern**: F16 ggml backend tensors with
   `ggml_view` + `ggml_cpy` writes. Port this to all VLM decoders.

4. **Migrate remaining duplicated helpers**: bttr, hmer, and posformer each have ~300
   lines of duplicated conv2d/relu/layernorm/linear. Migrate to `core/cpu_ops.h`.

5. **`ggml_gallocr` reuse** — DONE. Persistent gallocr on context for 7 engines
   (vit_embed, clip_text_embed, parseq_ocr, cnn_embed, ocr_detect, surya_det,
   layout_detect). LFM2 migrated to `ggml_backend_sched` + T-bucketing.

---

## Runtime Optimization Audit — Re-verification (2026-07-11)

Full re-sweep of the codebase against the June audit above. **Nearly every P0/P1
the June audit flagged has since been executed.** The tables above are retained
for history but are now stale: read this section for the current state. Findings
here were verified against current code (`git` HEAD), not carried from the doc.

### June-audit claims that are now WRONG (code has moved on)

| June claim | Current reality | Evidence |
|---|---|---|
| `conv2d_cpu` "still scalar / needs im2col restructure" (arch-rec #2) | Per-patch gather into a `thread_local` buffer + SIMD `dot_product` per output channel, with a hoisted interior-fast-path boundary check. Effectively single-patch im2col+SIMD. | `core/cpu_ops.h:345-400` |
| `mel.cpp` projection "naive triple-loop matmul" | `core_cpu::dot_product` fast path for the contiguous layout; scalar retained only for transposed/accumulator cases | `core/mel.cpp:116-117` |
| VLM decoders "F32 CPU KV re-uploaded each step / CPU-scalar" (qwen2vl, deepseek, smoldocling, granite, pix2struct) | Device-resident KV everywhere, but the F16 half of this claim was **WRONG for deepseek_ocr2** and is corrected 2026-08-05: its cache is device-resident **F32** by default (`alloc_ds_kv_cache`), F16 only under `DS2_KV_F16=1`. Its flash path is also opt-in (`DS_LLM_FLASH`), not the default. | qwen2vl_ocr.cpp:1091-1092,2412; deepseek_ocr2.cpp `alloc_ds_kv_cache`/`ds_kv_type`; granite_vision_ocr.cpp:626-627; smoldocling_ocr.cpp:685-686; pix2struct.cpp:347 |
| SR "No SIMD anywhere / no dequant caching 12-of-13 / no tiling" | 11/13 SR runtimes on ggml graphs; DequantCache fleet-wide (12 files); Hann-window tiling universal | esrgan_sr.cpp:362; instructir.cpp:164; scunet_denoise.cpp:327 |
| decoder_embed "no flash in single-text path" | Single-text path (B≤1) now calls `ggml_flash_attn_ext` | decoder_embed.cpp:1196,1421 |
| gliner "BiLSTM fully scalar" | Gate matmuls use `core_cpu::dot_product` (SIMD); only the per-timestep sequencing is inherent | gliner_ner.cpp:915-916 |
| tesseract "LSTM gates no SIMD" | SIMD via `core_cpu::dot_product` | tesseract_lstm.cpp:256-257 |
| Math-OCR scalar encoders (DenseNet bttr/hmer/posformer, HGNetv2 ppformulanet, Swin mixtex) | DenseNet/HGNetv2 → ggml graphs (default); mixtex projections on ggml (window attention still scalar — see gaps) | bttr/posformer/hmer/ppformulanet; mixtex_ocr.cpp:126 |

### Verified DONE since June (net-new work)

- **Device-resident KV cache** across the VLM decoder set; **persistent
  single decode graph** in math_ocr (TrOCR ~4×). **Correction 2026-08-05:** the
  claim that deepseek_ocr2 had a persistent single decode graph was wrong — it
  rebuilt 12 graphs per token until T14 built one, and the result was measured
  as no win and left opt-in (`DS2_FAST_DECODE=1`). Its KV is F32, not F16.
  Other VLM decoders (qwen2vl/granite/smoldocling) have device-resident KV but
  still rebuild the decode graph per step.
- **WebGPU/WASM tier** (OCR build): ~950 lines of authored WGSL kernels
  (LayerNorm, IM2COL/CONV_2D/POOL_2D/CONV_TRANSPOSE_2D/UPSCALE/ARANGE) landed in
  the pinned `ggml` submodule. Detection ~60×, det+rec pipeline ~1.8×, ~2.8×
  total vs SIMD-CPU. Multithreaded via `--proxy-to-pthread`.
- **Beam search** added to math_ocr, bttr, ppformulanet, ppformulanet_l.
- **imatrix** quant rollout (20 models); confirmed **zero inference cost** —
  the eval-callback early-returns unless `CRISPEMBED_IMATRIX_OUT` is set
  (`imatrix.cpp:131`). It is a calibration/quant-time artifact only.

### True remaining gaps (2026-07-11)

| P | Area | Gap | Impact |
|---|---|---|---|
| **P1** | Graph caching (all runtimes) | **0 runtimes reuse the built cgraph.** Decoders rebuild + `sched_reset`+`alloc_graph` per token; device KV landed but the graph around it is rebuilt each step. Blocked on WebGPU (traps `unreachable`) — needs per-backend gating (safe on Metal/CPU). | #1 unrealized lever |
| **P1** | layout_detect | Deformable cross-attention still CPU-scalar 6-nested bilinear grid-sample | Dominates DETR decoder; the one surviving June P0 |
| **P1** | text_sr | ~~Only SR engine still fully scalar (`tsr_conv2d`)~~ — conv now delegates to SIMD `core_cpu::conv2d_cpu` (this branch). Remaining: a full ggml graph for GPU offload | Convs SIMD-accelerated; GPU path still open |
| **P1** | WebGPU embedding tier | `build-embed-wasm.sh` has no `--webgpu` path — text embeddings are CPU-only in browser | Whole embedding browser tier misses the proven GPU path |
| **P2** | scunet_denoise | Swin blocks still scalar; only SR engine without DequantCache (`scunet_denoise.cpp:32`) | Transformer half unaccelerated + repeated dequant |
| **P2** | SR-on-GPU (whole family) | **Correction:** the ENTIRE SR family runs conv on a CPU-only `enc_sched` (`swinir_sr.cpp:447` prints `ggml_conv_2d (CPU sched)`); dat/hat/swinir use `init_best` only to LOAD weights, then copy dequantized weights into a CPU-resident context. esrgan/safmn/restormer/instructir just skip that copy. There is no GPU sibling to match — SR-on-GPU is unsolved research (Metal `ggml_conv_2d` + GPU-resident weight/graph path), not a residency toggle | Reprioritized down |
| DONE | safmn honor `n_threads` | Was hardcoded to a 1-thread conv sched (`safmn_sr.cpp:255`); now honors `-t N` like siblings | **~2.3×** (16.2s→7.1s, 8-core Mac), bit-identical output |
| **P2** | mixtex_ocr | Swin window attention still scalar (`mixtex_ocr.cpp:126`) | Encoder O(N²·D)-bound |
| **P2** | qwen2vl/granite/smoldocling | Decode graph rebuilt per step (KV device-resident, but not the persistent-graph pattern math_ocr uses; deepseek_ocr2 was wrongly listed here — see 2026-08-05 correction). **Measure the overhead fraction before porting:** deepseek's per-step build+alloc was only 1-6% of decode, so its persistent graph won nothing | Per-step build/launch overhead — only where it is actually a large fraction |
| **P2** | deepseek_ocr2 | LLM/enc flash are opt-in default-off (measured slower on CPU); re-benchmark on Metal/CUDA | Backend-dependent |
| **P3** | Build/infra | No LTO/IPO; `GGML_BLAS=OFF` (Accelerate not guaranteed for CPU-fallback matmul); `--gpu-backend` ignored (`crispembed.cpp:81` calls `init_best()` directly); app-level OpenMP possibly unlinked; Metal F16 mul_mm guard in only 5/~40 GPU files | Broad low-effort |
| **P3** | Misc | ocr_orchestrator PNG round-trip + N reloads; gliner DeBERTa rel-pos [H,T²] ~117MB/call; ppformulanet_l decoder scalar; `conv2d_cpu` not GEMM-batched/multithreaded | Localized |

### Highest-ceiling paths forward

1. **Decode-step graph cache** (per-backend gated) — cache the *decode-step*
   graph, not the encoder graph (encoder caching is a measured dud + a GPU
   use-after-free landmine). Templates: the `sched_reserve`+T-bucket pattern in
   the text encoder and lfm2.
2. **ggml-metal ICB (indirect command buffer) replay** — Metal decode is
   per-op-dispatch bound; CUDA-graph capture already solves the CUDA side.
3. **Finish residual scalar kernels** (layout_detect deformable, text_sr, scunet
   Swin, mixtex window attn) and upgrade `conv2d_cpu` per-patch → true
   im2col+GEMM (batch all output channels) + multithread.

---

## VLM OCR Benchmarks (Intel Xeon Skylake, 4 threads, CPU-only)

### Qwen3-VL-2B-Instruct (q4_k, 1.5 GB)

End-to-end OCR on 800×300 invoice image. `QWEN_DBG=1` for per-stage timing.

| Setting | Patches | Vision | Prefill | Decode/step | Quality |
|---------|---------|--------|---------|-------------|---------|
| Default (max_pixels=16M) | 900 (18×50) | 24.5s | 35.3s | 5.0s | 5/5 lines |
| `CRISPEMBED_MAX_PIXELS=65536` | 208 (8×26) | 15.0s | 21.7s | — | 4/5 lines |

**Speedup**: 1.6× faster vision+prefill (60s → 37s) at minor quality loss.

`CRISPEMBED_MAX_PIXELS` reduces input resolution before patch extraction.
Useful for CPU-only deployment where speed matters more than pixel-perfect OCR.
Applies to all VLM OCR engines that use `image_preprocess.cpp`.

## Local M1 Metal OCR engine sweep (2026-07-31)

Command: `python3 tests/ocr_engine_benchmark.py --repeats 1 --timeout 45
--output /tmp/crispembed-ocr-benchmark.json`.  These are cold end-to-end
process times on the local Apple M1; they include model loading and should not
be read as steady-state service throughput.  Quality is scored only against
the manifest's known fixture text.

| Engine | Status | Cold ms | Quality |
|---|---:|---:|---|
| GOT-OCR2 | ok | 15,662 | exact |
| GLM-OCR | ok | 32,884 | exact |
| InternVL2-1B | ok | 24,908 | CER 0.540 (prompt text included) |
| Qwen2-VL-3B | timeout/error | 70,757 | no transcript before 45 s |
| LightOnOCR | ok | 31,561 | unscored; plausible transcript |
| MixTeX | ok | 7,523 | exact specialist formula |
| Flova | ok | 16,153 | exact specialist LilyPond |
| Pix2TeX | ok | 5,520 | exact specialist formula |
| Texteller-3 | ok | 11,403 | CER 7.293; unusable on this fixture |
| Tesseract-LSTM line-crop pipeline | ok | 7,552 | CER 0.040; 10 DBNet regions, all recognized |
| PARSeq-tiny | ok | 921 | unscored full-page smoke (`Gooducalicanos.com`); scene-line recognizer |

The proper DBNet+TrOCR Q8 pipeline remains the ordinary document baseline:
10/10 regions, 10/10 recognized regions, 8.05 s cold on the same M1 run. The
Tesseract result is now measured through the actual DBNet→line-crop→LSTM
pipeline: 10 regions, all recognized, 7.55 s cold / 8.09 s warm, CER 0.040
(punctuation-only drift). PARSeq remains recognizer-only and still needs a
line-crop orchestrator benchmark.

The manifest contained 51 entries: 8 completed, 1 timed out, and 42 explicit
skips because a sample or local model was unavailable.  This is a coverage
report, not a claim that the skipped engines are unsupported.  The reusable
driver stores all output and stderr tails in JSON for follow-up runs.

### Tesseract reference parity and gated page-segmentation cost (2026-08-01)

This is a same-fixture quality/cost cross-check on `scan_strip.png`, not a
claim that all full-page Tesseract behavior is matched. Official timings are
stock Tesseract CLI TSV wall time; native timings are the instrumented
detector→group→crop→recognizer stage total. The native subprocess elapsed time
also includes test-binary/model setup and is therefore not used as the pure
pipeline comparison.

| Path | Output quality vs official | Official wall ms | Native stage ms | Native result |
|---|---|---:|---:|---|
| Legacy/fallback | Best current native path, but CER/WER `0.0179/0.0841`; confidence `0.895` vs `0.9108` | 315.9–349.9 | 310.7 | 12 regions, 567 chars |
| Component | Worse: CER/WER `0.0322/0.1121` | 315.9–349.9 | 266.8 | 12 regions, 569 chars |
| Baseline | Same CER/WER as legacy, no quality gain; IoU lower | 315.9–349.9 | 282.2 | 12 regions |
| Projection | Worse: CER/WER `0.0250/0.1121`; IoU best but text worse | 315.9–349.9 | 360.1 | 12 regions |

Native recognition dominates the stage (`260.3–353.8 ms`); detector and crop
were approximately `3–4 ms` each. A worker sweep retained identical CER/WER
and measured native stage totals of `690.3 ms` at one worker, `300.7 ms` at four,
and `292.1 ms` at eight. The immediate performance TODO is recognizer batching,
graph/weight reuse, and fair warm-run measurement; the detector is not the
current bottleneck. The immediate quality TODO is full-page crop/spacing/text
parity: native is not yet output-equivalent even where region count matches.

An activation-scratch reuse prototype is gated by
`CRISPEMBED_TESSERACT_REUSE_SCRATCH`; it preserves CER/WER but measured about
`279.1 ms` versus `282.3 ms` in one paired run, while earlier repeated runs
were `329–338 ms` versus the prior `~300 ms` result. The variance is too large
to claim an improvement, so it is disabled by default and remains an
optimization TODO.

Use `tools/benchmark_tesseract_page.py` for repeated, policy-specific runs;
its summary separates official CLI, native subprocess, native stage, and
recognizer timings and retains every per-run quality record.

The German official-print page remains materially worse: native default is 21
regions vs official 25, CER `0.307`, WER `0.404`, and confidence `0.836` vs
`0.866`. Paired warm/cold timing and per-stage reference timing for this page
remain TODOs. The Fraktur line diagnostic is also not a speed claim because its
available input is a full page under PSM7 rather than an identical transcribed
line crop.

Latest normalized-artifact rerun (current Fraktur Q8 artifact) is worse still:
official Tesseract took `9.34 s` for 25 lines/881 chars at confidence `0.8658`,
while native took `38.69 s` of stage time for 23 regions/1,235 chars at
confidence `0.768`, CER `0.5279`, and WER `0.5390`. Native recognition consumed
`38.34 s`; detection was `102 ms` and crop `250 ms`. The earlier Q8/F16
measurement is retained as historical until artifact and control conditions
are pinned identically. This is an explicit speed and quality blocker.

### Fraktur recognizer precision matrix (same German page/control)

| Recognizer artifact | Native stage | Regions/chars | Confidence | CER/WER | Assessment |
|---|---:|---:|---:|---:|---|
| `frk-q8_0` | 38.69 s | 23 / 1,235 | 0.768 | 0.5279 / 0.5390 | Faster, but worse text |
| `frk-f32` | 102.41 s | 23 / 1,164 | 0.767 | 0.4672 / 0.5461 | Better CER, far too slow |
| `frk-int8-source-q8-candidate` | 64.44 s | 23 / 1,164 | 0.767 | 0.4672 / 0.5461 | F32-like output, still too slow |
| `frk-mixed-lstm0hh-f32` | 23.42 s | 23 / 1,146 | 0.765 | 0.4603 / 0.5390 | Best measured CER/speed frontier, still worse than official |

Official Tesseract remained 25 lines/881 chars. Precision therefore changes
output quality as well as speed; optimizing standard Q8 alone cannot achieve
reference quality. Same-artifact warm/cold benchmarks and recognizer
optimization remain required before selecting the production Fraktur tier.

The mixed-precision candidate is generated with
`models/mix-tesseract-gguf.py`: Q8 remains the default base, while explicitly
selected critical tensors are copied from F32. The selected
`lstm.0.weight_hh` profile is not a production default; it remains gated
until repeat benchmarks, page-region parity, and decoded-text quality gates
improve.

Fresh Miniconda regeneration from `/opt/homebrew/share/tessdata/frk.traineddata`
now gives exact input parity (`cosine=1.0`, both norms `122.453`). Against
now gives exact input parity (`cosine=1.0`, both norms `122.453`). The old Q8
artifact lacked `sample_iteration`, causing the earlier `0.983119` logits
result and seeded-padding mismatch. A freshly converted F32 model reaches
9/9 stages with logits cosine `0.993819`; a mixed Q8/F32 candidate carrying
the recovered seed reaches 9/9 and `0.994876`. Both still decode differently
from Python, so the mixed candidate is not production-accepted. References are
stored at `/Volumes/backups/ai/crispembed-gguf/tesseract-frk-ref-fresh.gguf`
and `tesseract-frk-ref-int8fc.gguf`.

GGUF metadata audit of `/Volumes/backups/ai/crispembed-gguf/` found 46
Tesseract model artifacts: 45 lack `tesseract_lstm.sample_iteration`; only
`tesseract-eng-homebrew-intmeta-f32-sample6352704.gguf` carries it. The missing
seed can change every out-of-bounds Convolve padding value, so those artifacts
require regeneration or metadata-preserving reconstruction before parity
acceptance.

After regenerating the Python reference with exact int8 LSTM arithmetic, the
fresh F32 Fraktur GGUF passes all 9 captured stages exactly (final logits max
error `2.09e-7`) and decoded text matches Python. The seed-preserving mixed
Q8/F32 candidate remains below parity (`logits cosine 0.989655`) and decodes
differently; quantization quality is the remaining blocker.

Quantization policy improvement: `models/quantize.py` now supports repeatable
`--keep-pattern` rules, allowing callers to retain critical recurrent or
output tensors at source precision without changing the established default
quantization behavior. The policy is unit-tested and remains opt-in.

The public-domain fixture smoke path (`tests/ocr_fixture_smoke.py`) exercised
seven CC0/public-domain images through Tesseract plus skew/content detection:
all PNG/JPEG paths passed.  The original TIFF receipt correctly exposed a
format gap (`cannot load`); a PNG derivative is now included for the OCR
pipeline while the source TIFF remains available for a future native TIFF
decoder test.

### Tesseract runtime regression and recovery

A remote-main merge temporarily replaced the int-mode/scratch Tesseract
runtime with an older F32-only implementation. On
`tests/regression/images/scan_strip.png` with the same Fraktur Q8 artifact,
recognition measured `50.15 s` in that regression. Restoring the known-good
runtime and adding LUTs for the existing Tesseract nonlinear interpolation
contract measured `34.32 s`, with unchanged output: 12 regions, 566 chars,
CER `0.03375`, and WER `0.15044`. The required int-mode, LUT, and gated
scratch symbols are now protected by a runtime-contract test. The remaining
speed gap to official Tesseract is still an active TODO.

### Full local matrix comparison (M1 Metal, 2026-07-31)

The expanded manifest sweep completed 11 engines, recorded 2 errors, and
reported 41 explicit non-sample/non-model skips. Representative outputs:

| Engine/lane | Cold ms | Result |
|---|---:|---|
| GOT-OCR2 | 22,073 | exact fox transcript |
| GLM-OCR | 38,086 | exact fox transcript |
| InternVL2-1B | 28,523 | transcript plus prompt wrapper; CER 0.54 |
| Qwen2-VL-3B | 90,113 timeout | no output within limit |
| LightOnOCR | 69,289 | plausible transcript; currently unscored |
| Tesseract via DBNet line crops | 32,041 | 10 regions; CER 0.040 |
| PARSeq | 6,252 | `Gooducalicanos.com`; recognizer-only smoke |
| SmolDocling | 16,334 | text present but duplicated DocTags regions; payload CER 0.86 |
| MixTeX | 13,286 | exact specialist LaTeX |
| Flova | 36,293 | exact LilyPond |
| Pix2TeX | 8,980 | exact LaTeX |
| TexTeller | 18,491 | CER 7.293; unusable on fixture |

SmolDocling is therefore supported and live-tested; its next fix is structural
deduplication/DocTags parsing, not model discovery. Unlimited-OCR's Q4_K stacked
artifact is complete at 2,252,419,328 bytes and now has a successful M1 Metal
run when loaded from the system volume: 45,967 ms total (SAM 15,663 ms, CLIP
2,260 ms, projection/assembly 5,835 ms, decoder 22,205 ms), with two correctly
decoded text regions. The external backup-volume no-copy path (`UOCR_MMAP=1`)
also completes: 40,391 ms cold benchmark time and CER 0.010, with the one
character difference being a harmless title-box coordinate drift. Qwen2-VL is
runnable but did not complete this M1 budget.

### Tesseract seeded model rebuild (2026-08-01)

The 45 unseeded model artifacts were not all independently valid: missing
`tesseract_lstm.sample_iteration` changes seeded out-of-bounds convolution
padding. The 12 installed canonical sources were hash-matched to the old
GGUFs, then freshly converted with Miniconda. The backup store now contains 42
readable `*-seeded.gguf` companions: F32/F16 are freshly converted, while Q8_0
and Q4_K retain the old quantized tensor bytes and receive only the verified
source metadata. All 42 carry a nonzero seed. No speed or OCR-quality claim is
made yet for quantized companions; per-language `crispembed-diff` and decoded
output checks remain TODO. One old Fraktur mixed candidate is truncated and
was excluded.

Chinese seeded F32 is the first decoded-output exception: all 9 stages pass
with aligned magnitudes, but the old native decoder returned an empty string
while the Python reference returned `<141>`. This is a harness-blind recoder
mapping defect, not a graph discrepancy. The native fallback now exposes the
unmapped class; no Chinese OCR-quality or quantized-speed claim is accepted
until recode-beam composition is implemented and tested.

German's apparent quality gap was traced to the Python reference, not the
native graph. Upstream Tesseract's `generate_lut.py` computes the 4096 tanh and
logistic table entries with double-precision `math.tanh/exp`, then stores them
as `TFloat`; the reference had evaluated NumPy float32 nonlinearities directly.
Regenerated German references now pass all 9 stages exactly through the LSTM
and finish at max logit error `3.58e-7`; native and Python both decode ` s.`.

The corrected seeded F32 sweep has exact native/Python decoded parity for all
12 languages on the controlled line. The stale Spanish reference was also
regenerated after the LUT correction; its former one-blank decoded mismatch
was a reference artifact. Korean's prior 6/200 argmax differences disappear
when the production native LUT uses the same generated-table values as
upstream Tesseract; the final Korean run has 0/200 mismatches and exits 0.

All 51 corrected canonical F32/F16/Q8_0/Q4_K files are now uploaded to the
intended `cstr/tesseract-lstm-GGUF` and `cstr/tesseract-frk-GGUF` repositories.
Remote metadata spot-checks confirm nonzero `sample_iteration`; no
`mlx-community` upload was made.

### Tesseract cached-int8 recurrent kernel gate

On the same scan-strip input and `tesseract-frk-q8_0.gguf`, cached and
uncached int-mode decoding both returned `SEEEES`. The cached path measured
`35.4 ms` LSTM time versus `1,035.6 ms` with
`CRISPEMBED_TESSERACT_DISABLE_INT_CACHE=1`, a `29.3x` speedup with identical
decoded output. Cached mode is therefore the default; the environment gate is
retained for parity diagnostics and alternate architectures.

Full-page validation on `scan_strip.png` confirmed the same result: cached and
uncached paths both produced 12 regions/566 chars with CER `0.03375` and WER
`0.15044`. Cached native stage time was `22.11 s` versus `157.59 s` uncached,
for a `7.1x` speedup; detect plus crop was only `46.1 ms` cached. The remaining
Fraktur page-quality gap is therefore in recognition/output parity, not DBNet
or crop throughput.

The comparator now stores both normalized decoded strings. The scan-strip
official/native pair is 451/566 chars with CER `0.03375`; representative
differences are `50`→`80`, `ay`→`8ay`, capitalization (`Such`/`such`,
`Scheme`/`scheme`), and punctuation/hyphen spacing. This confirms the next
quality work should inspect crop geometry and decode semantics, not detector
throughput.

### Tesseract crop-border A/B (2026-08-01)

The Fraktur line crop now has an opt-in `CRISPEMBED_TESSERACT_CROP_PAD` gate;
the default remains 2 pixels. On `scan_strip.png`, all candidates produced 12
regions, so this is not a segmentation-count issue:

| Border | Chars | CER | WER | Recognize ms |
|---:|---:|---:|---:|---:|
| 0 px | 570 | 0.07460 | 0.30088 | 7,237.5 |
| 1 px | 567 | 0.04796 | 0.20354 | 6,686.3 |
| 2 px (default) | 566 | 0.03375 | 0.15044 | 9,217.4 |
| 4 px | 571 | 0.03552 | 0.15044 | 10,666.3 |

The 2-pixel crop remains the best measured quality point. The gate is retained
for other scan resolutions and architectures; the next quality TODO is
Tesseract-compatible decode/recoder semantics for the residual substitutions
and punctuation differences.

### Tesseract page-segmentation and beam A/B (2026-08-01)

The existing page-segmentation policies were compared on the same fixture and
official reference. Every policy emitted 12 regions:

| Policy | Chars | CER | WER | Recognize ms | Output result |
|---|---:|---:|---:|---:|---|
| Legacy fallback | 566 | 0.03375 | 0.15044 | 9,217 | baseline native text |
| Projection | 567 | 0.03197 | 0.12389 | 9,661 | best measured WER/CER |
| Baseline matcher | 566 | 0.03375 | 0.15044 | 14,720 | identical quality, slower |
| Projection + beam 8 | 567 | 0.03197 | 0.12389 | 29,748 | text-identical to greedy |

Projection remains opt-in because its CER improvement is small and it does not
reach official output parity; beam width 8 is retained only for diagnostics
because it adds roughly 3x recognition cost without changing text. The next
quality work is line-image/crop geometry and Tesseract decoder semantics.

The line-confidence comparator now accepts `--tessdata-dir` so official TSV
results do not depend on a potentially stale `TESSDATA_PREFIX`. On the valid
German tiny-line fixture with `/opt/homebrew/share/tessdata`, official output
is `1` at word confidence `0.588557`; native greedy is `G` with word confidence
`0.883064`, while beam-8 is `GEIEE` with sequence confidence `0.535476` and
zero fabricated character confidences. The beam contract passes, but text and
greedy confidence calibration are worse than the official reference and remain
TODOs.

With `--require-official-words --require-greedy-text-match`, that same fixture
exits `1`: the official-word gate passes, while the text gate fails. This keeps
the confidence contract from being mistaken for OCR-quality parity.

### Tesseract page-box geometry A/B (2026-08-01)

`CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD` now controls the symmetric expansion of
legacy component rows; the default remains 3 px. On the scan-strip fixture,
both tighter candidates preserved 12 regions and the same decoded text:

| Box pad | Chars | CER | WER | Recognize ms |
|---:|---:|---:|---:|---:|
| 1 px | 566 | 0.03375 | 0.15044 | 17,453 |
| 2 px | 566 | 0.03375 | 0.15044 | 12,155 |
| 3 px (default) | 566 | 0.03375 | 0.15044 | 9,217 |

The box geometry is therefore not the dominant error on this fixture. Keep
the gate for other scan resolutions, but do not change the default or use
tighter boxes as a quality claim.

### Tesseract composed-recorder gate (2026-08-01)

`CRISPEMBED_TESSERACT_RECODE_COMPOSE` now enables exact segmentation of
collapsed CTC classes into serialized multi-code unichar entries. It is
opt-in: the existing single-code fallback remains the production default.
Fraktur default versus opt-in output/confidence is byte-identical on the
controlled line, and a Chinese smoke input passes both modes without a crash;
that smoke did not exercise a multi-code emission, so no quality promotion is
claimed yet.

### Tesseract confidence harness and line calibration (2026-08-01)

The confidence comparator was hardened against non-UTF-8 Tesseract stderr and
stale inherited `TESSDATA_PREFIX` values. On a cropped Fraktur line, official
and native F32 text differed only by one missing space (`1 hey` vs `1hey`).
Official mean word confidence was `0.7060`; native greedy word confidence was
`0.9726`, while beam reported sequence confidence `0.9924` and no character
confidences. This is a calibration/aggregation gap, not evidence for changing
the recognizer weights or beam default.

The page comparator now uses the same explicit tessdata/environment isolation.
The scan-strip baseline is unchanged with that correction: official 12 lines,
113 words, 451 chars versus native 12 regions and 566 chars, CER `0.03375`,
WER `0.15044`.

The page comparator now has an opt-in `--require-text-match` gate and retains
the normalized official/native page strings in its comparison output. The
scan-strip baseline therefore remains explicitly non-green for exact output
parity even though its CER/WER metrics are measurable.

The confidence harness was rerun after rebuilding `test-confidence`, using the
explicit Homebrew tessdata directory and the seeded Fraktur Q8 GGUF. Official
PSM 7 TSV returned `iE` at mean word confidence `0.043433` in `5,881 ms`;
native greedy returned `BEEES` at word confidence `0.884625` in `305 ms`, and
beam-8 returned the same `BEEES` with sequence confidence `0.644788` and zero
per-character confidences in `984 ms`. The official-word check passed, but
decoded text and greedy calibration did not. This is evidence for a remaining
Tesseract decoder/recoder and confidence-aggregation quality TODO, not a
performance acceptance result; the beam path remains diagnostic.

Converter smoke (2026-08-01): Miniconda converted the installed Homebrew
`eng.traineddata` to `/tmp/crispembed-eng-dawg-smoke.gguf` successfully. The
6.6 MiB GGUF contains the three available LSTM DAWG payloads
(`lstm-punc-dawg`, `lstm-system-dawg`, and `lstm-number-dawg`), each with a
base64 payload and SHA-256 metadata. This verifies preservation only; the
artifact is not a promoted backup model and native dictionary scoring is still
unimplemented.

The regenerated DAWG-bearing smoke GGUF loads successfully in the native
runtime and reports `dawg=3`; the live confidence target passed `35/35` checks
on `scan_strip.png`. The decoded smoke text was `Se`; this validates metadata
acceptance only and is not a page-quality or DAWG-parity result.

The native load path now performs the same structural checks in a standalone
DAWG validator. `test-tesseract-dawg` passes the minimal valid edge fixture and
rejects malformed input; this adds negligible load-time validation and no
runtime OCR scoring cost because DAWG traversal remains disabled.

The opt-in system-DAWG prefix filter was A/B tested on the regenerated English
smoke GGUF with recoder beam width 8. Both unfiltered and filtered runs passed
`37/37`, decoded `Se`, and reported sequence confidence `0.562293`; the filter
did not alter this fixture. This is a safety/observability result, not dictionary
quality parity, and the default remains unchanged.

A seeded-artifact page-gate rerun correction (2026-08-01): the earlier 2-box
report was stale binary evidence. After rebuilding `test-ocr-orchestrator`
following the remote pageseg changes, the canonical Q8 DBNet IC15 detector plus
corrected Fraktur seeded F32 and Q8_0 recognizers both emitted 12 boxes/lines
and passed the pipeline gate. Exact text still fails: both runs measured
CER/WER `0.03922/0.13274`; F32 took 12,373 ms total with confidence delta
`0.01647`, and Q8 took 14,560 ms with confidence delta `0.01447`. The remaining
quality gap is punctuation, spacing, and glyph output from line
recognition/decoding, not detector box count or a precision-only failure. The
stale 2-box result is rejected and should not be used as a performance or
compatibility baseline.

The native crop diagnostic now dumps the exact recognizer inputs on demand via
`CRISPEMBED_TESSERACT_CROP_DUMP_DIR`. The rebuilt Q8 scan-strip run produced
12 grayscale crops, with heights 22–32 px and the final crop 76×25 px. This
confirms valid line geometry, but does not yet establish equivalence with
Tesseract CLI's internal line normalization. A direct single-crop CLI A/B was
not accepted because the installed Homebrew Tesseract/Leptonica could not
reopen a valid dumped PNG; repeat after fixing that environment before drawing
quality conclusions.

The diagnostic also emits `crops.tsv`. A verified Q8 run produced 12 records
plus the header; source boxes map to crop sizes 438×22 through 462×32, with a
final 76×25 crop. The first line begins at page `y=0`, so edge clipping is now
an explicit geometry item for the official-Tesseract comparison.

The opt-in vertical ink-trim A/B is a rejected quality optimization: native
recognition improved from 11,351.6 ms to 10,407.1 ms, but CER/WER degraded
from `0.03922/0.13274` to `0.04278/0.14159` and the character delta grew from
116 to 121. Keep `CRISPEMBED_TESSERACT_CROP_TRIM_INK` diagnostic-only.

The component page-box pad A/B is also neutral for quality: with
`CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD=0`, native output remained byte-identical
to the default and CER/WER stayed `0.03922/0.13274` with 12 regions. Do not
count this as a speed win; the isolated run's timing was not stable enough for
an optimization claim.

The existing component-row segmentation A/B is a quality regression: it kept
12 regions but produced CER/WER `0.10873/0.20354` versus the legacy baseline
`0.03922/0.13274`, including a corrupted first line. Keep the component policy
diagnostic-only and do not use the malformed-path run as benchmark evidence.

`tools/compare_tesseract_crop_geometry.py` now provides a reproducible
geometry-only benchmark. The current 12-line run reports mean native-minus-
official deltas `dx=-2.08`, `dy=+1.83`, `dw=+4.33`, `dh=+1.50`; worst rows are
width `+80`, vertical offset `+14`, and height `+12`. These are row-boundary
quality findings, not a measured runtime regression.

The gated row-blob-bounds A/B fixes the largest local geometry error: CER/WER
improved to `0.03209/0.11504`, mean width delta fell to `+2.42`, and worst
width delta fell from `+80` to `+13`, with 12 regions preserved. This is a
quality improvement on scan-strip, but remains diagnostic-only until validated
on more page fixtures; exact output parity still fails.

The per-line page comparator was corrected to group official TSV words by
page/block/paragraph/line rather than by `word_num`. On the corrected
row-blob-bounds run, both paths emit 12 lines and only 3/12 lines match
exactly. The first differing line is line 0 (`<< 4 ...` official versus
`“< A ...` native); lines 4, 7, and 9 match exactly. Overall CER/WER remains
`0.03209/0.11504`, so this is a recognition/crop or decoder-quality TODO,
not a segmentation-count or ordering failure. Native benchmark was
`detect=89.5 ms`, `crop=258.5 ms`, `recognize=17216.3 ms`, `total=17564.4 ms`;
official Tesseract CLI elapsed `47761.8 ms` in this run, but these timings are
not yet a controlled backend-speed comparison.

The first divergent line was checked at the tensor boundary. Native crop 0
was dumped and a Python reference was regenerated from the installed Fraktur
traineddata. `test-tesseract-lstm-diff` passed every captured stage (input,
convolution, conv-FC, maxpool, four LSTM stages, and logits); the lowest
cosine was `0.997755`, with recurrent mine/ref norms `35.8611/35.8704`, and
the native/Python decoded strings were identical. The official Homebrew CLI
cannot reopen local PNG/PGM/TIFF files in this environment, so direct CLI
single-crop confirmation is blocked; the page-level mismatch is nevertheless
localized to official page segmentation/line normalization rather than GGUF
recognition math. Use the comparator's new `--crop-dump-dir` option for fresh
crop manifests. `tools/compare_tesseract_crop_diff.py` now automates the
per-crop Python-reference regeneration and native `test-tesseract-lstm-diff`
run while refusing to overwrite an existing reference.

On the CC0 German printed-document fixture, official Tesseract emitted 28
lines/153 words/897 characters while native DBNet emitted 23 lines/862
characters. CER/WER was `0.32984/0.67974`; native stages measured
`detect=982.4 ms`, `crop=670.0 ms`, `recognize=19594.7 ms`, and
`total=21247.2 ms`. Since five lines are missing or merged before recognition,
index-paired per-line errors are not a valid recognizer benchmark. The page
comparator now reports `alignment_valid=false` when line counts differ. This
fixture is a detector/line-geometry TODO, separate from the crop-level tensor
parity proven on scan-strip.

The comparator now exposes the native Tesseract-like route explicitly with
`--native-pageseg`. On `scan_strip.png`, this route produced 12/12 lines,
CER/WER `0.03209/0.11504`, and 3/12 exact lines. Its native stage timing was
`detect=12.6 ms`, `crop=644.8 ms`, `recognize=11856.4 ms`,
`total=12513.8 ms`. The route is not using DBNet for box generation; its
quality is identical to the established classical row path, so the remaining
gap is page segmentation/line normalization and decoder semantics.

The repeated benchmark wrapper now accepts `--native-pageseg` and records
`detector_route`, preserving the DBNet-versus-native distinction across
multi-repeat timing runs. Its route flag and comparator selection are covered
by the 10-test focused harness.

On the CC0 German page, the explicit native route emitted the same 23 lines
and 862 characters as the DBNet route, versus 28 official lines and 897
characters. CER/WER stayed `0.32984/0.67974`; native timing was
`detect=1014.9 ms`, `crop=605.7 ms`, `recognize=14263.4 ms`,
`total=15885.6 ms`. This is a shared five-line geometry/coverage gap, not
evidence that either recognizer is worse on aligned crops.

The German native crop manifest has 23 rows versus 28 official TSV rows. The
geometry comparator now marks this as `alignment_valid=false` and reports the
number of index-paired rows; its former mean `dy=257.7` was an alignment
artifact, not a measured crop offset. A merge-aware line matcher remains a
detector/geometry TODO before using per-row geometry deltas on this fixture.

The crop comparator now has `--match-by-geometry`: the German native run
matched 23 rows monotonically and exposed five unmatched official rows
(`0,2,3,4,26`). It still exits 1 for the count mismatch, and the resulting
matched deltas remain diagnostic until one-to-many merged-row matching is
implemented.

Source inspection indicates several official rows are nested decorative marks
inside larger text boxes, so the merge report now labels a primary official
box and nested rows. This prevents a speculative production split based only
on TSV row count. On German, native row 0 has primary official index 1 and
nested indices 2 and 4; native row 9 has primary index 13 and nested index 12;
native row 22 has primary index 26 with no fully-contained nested row.

The geometry report now exposes `merged_official_groups` when one native row
covers at least half the vertical extent of multiple official rows. This
separates merge candidates from genuinely missing rows without changing the
production native pageseg policy.

On the German fixture, the report finds merge candidates native `0` →
official `1..4`, native `9` → official `12..13`, and native `22` → official
`26..27`; official row `0` remains unmatched. These are concrete geometry
targets for row-splitting work, not recognizer timing or tensor-parity data.

## AdaIR F16 runtime audit (2026-08-02)

The AdaIR F32 reference path remains valid on the 64×64 diff fixture: ggml
convolution measured cosine `0.999382`, max absolute error `0.027892`, and
about `2.65 s` inference. The scalar-gated F32 path also passed at cosine
`0.999379`, but measured about `16.8 s`, so it remains a correctness fallback,
not a performance path.

The original `adair-5d-f16.gguf` reproduced the backend assertion in the
per-kernel CPU convolution cache. Allocation guards now prevent the process
abort and disable the ggml convolution route for that context, preserving the
existing scalar fallback. The completed F16 fallback runs returned cosine
`0.729509` and max absolute error `0.707725`; therefore crash-freedom must not
be confused with output parity. Timings ranged from roughly `7.3 s` to
`180.5 s` while other large builds were contending for the host, so they are
not suitable as a stable benchmark.

The F32→F16 rebuild made with the repository quantizer was also tested and
produced the same F16 cosine failure. Tensor inspection found representative
metadata changes from `[3,3,3,48]` to `[27,48]` and `[1,1,48,144]` to
`[48,144]`, with a mixture of F32 and F16 tensor types. Raw values for the
sampled weights were close to the F32 source, but the runtime still reported
an allocation size of zero for the F16 kernel descriptor. Explicit CPU buffer
selection and manual buffer allocation did not change the outcome and were
reverted.

Current status: F32 is the only AdaIR precision cleared for release. F16
requires a loader/converter descriptor audit, tensor-level dequant parity
checks, and an end-to-end cosine gate before it can be uploaded or registered.

## AdaIR F16 — root-caused and fixed (2026-08-02, `feat/ocr-followups`)

The audit above closed on the right observation — a zero-size F16 kernel
descriptor — but attributed it to the buffer allocator. It was neither an
allocator nor a converter defect. Two facts the audit had already collected
point at the real cause once they are read together: the header shapes change
(`[1,1,IC,OC]` → `[IC,OC]`) while the *values* do not.

`tools/quantize.cpp` (~line 167) flattens every 4-D F32 conv weight to 2-D
`[IC*KH*KW, OC]` so the output header is valid for a quantized tensor. That is
deliberate and other engines depend on it. But `src/adair.cpp` inferred three
hidden widths from `->ne[3]`, which is `1` on any flattened tensor:
`gdfn_forward`'s hidden width, and the FreModule `rate_conv` and `ChannelGate`
MLP widths. `hidden = 1` makes `half = hidden / 2` zero, so the next 1×1 conv is
built with `ic == 0` — a kernel with zero elements. That is the descriptor the
allocator refused, and with the guards in place it degraded to cos `0.729509`
instead of aborting.

Independent check that the artifacts were never at fault: across 60 randomly
sampled tensors, `adair-5d-f16.gguf` versus `adair-5d-f32.gguf` gives worst
cosine `0.999998` and worst max_abs `1.22e-4` — pure F16 rounding.

Fix: `conv1x1_out_channels()` derives OC from `ggml_nelements(t) / ic`, correct
under both layouts, with fail-loud guards at all three sites.
`ADAIR_LEGACY_NE3_DIMS=1` restores the old `ne[3]` read so both arms live in one
binary.

Measured on the 64×64 `adair-ref.gguf` fixture, same binary, ggml conv path:

| artifact | arm | cos | max_abs |
|---|---|--:|--:|
| `adair-5d-f32.gguf` | default | `0.999382` | `0.027892` |
| `adair-5d-f16.gguf` | default (fixed) | **`0.999383`** | `0.027871` |
| `adair-5d-f16.gguf` | `ADAIR_LEGACY_NE3_DIMS=1` | `0.729509` | `0.707725` |

The f32 number reproduces the audit's `0.999382` exactly, so it is the
regression control, not a re-measurement. **No timings are quoted here**: the
box carried load average 55–127 from parallel agent builds throughout, and the
same 64×64 fixture took `312 s` at f32 against a quiet-box reference of
`2.65 s`. Re-time on a quiet machine before any performance claim.

Remaining before the f16 can ship: upload it to `cstr/adair-GGUF`, add its
SHA-256 to `examples/cli/model_hashes.h`, and repoint the `adair-5d` registry
entry. The runtime no longer blocks it.

End-to-end through the real CLI, not just the diff harness (HARD RULE #3):
`crispembed --adair-model <f16|f32> --adair <96x96 crop of tests/regression/images/fox.png>`
returns rc=0 for both at the same 27,661-byte PPM, and the two restored images
agree at cosine `1.0`, max_abs `1/255`, mean_abs `7e-5`. Output mean/std is
`242.821 / 35.92` for both — real image content, not a blank or saturated frame.

### Exactly which artifacts the flatten touches (measured, 2026-08-02)

The flatten condition is `ggml_n_dims(t) == 4 && t->type == GGML_TYPE_F32` — 4-D
**F32** only. Verified by running `crispembed-quantize … q8_0` over
`surya-det-f16.gguf`, whose conv weights are 4-D **F16**: the output still reports
`stem.in_conv.weight` as `[3,3,3,32]`, all 79 4-D tensors intact. `adair-5d-f16.gguf`,
quantized from an F32 source, has none left.

So the exposure rule is: **a GGUF is affected iff its conv weights are stored 4-D
F32 and it is then run through `crispembed-quantize`.** Neither the precision
label nor the file name tells you — the *source dtype plus the producer* does.

### Follow-up resolved: one of the two suspect sites was a real bug

The first version of this entry listed `src/surya_det.cpp:700` and
`src/tps_locnet.cpp:219` together as "same bug class, latent". Checking them
properly split them apart:

- **`src/surya_det.cpp:700` is NOT a bug.** `g_conv` normalises the layout before
  that read — `if (ggml_n_dims(w) == 2) { OC = w->ne[1]; w = ggml_reshape_4d(...); }`
  runs first, so `w->ne[3]` is already 4-D by the time the grouped-pointwise
  branch reads it. No change made.
- **`src/tps_locnet.cpp:219` WAS a real, reachable bug — now fixed.** It reads
  `ne[3]` at load time with no such normalisation, and
  `models/convert-tps-loc-to-gguf.py` **defaults to F32** (`--fp16` is opt-in), so
  a quantized tps-loc GGUF is precisely the affected shape. Instrumented proof
  before the fix: all four layers loaded as `ne=[27,16,1,1] … [576,128,1,1]`,
  `ndims=2`, `channels=1` — against `16/32/64/128` from the 4-D build. `channels`
  is not cosmetic: it feeds the fc1 input width (line 243) and the per-layer
  output channels (line 300). Fixed with the `src/cnn_embed.cpp:148` convention,
  `ggml_n_dims(w) == 2 ? ne[1] : ne[3]`.

Guarded by a new hermetic case in `tests/test_tps_locnet.cpp` (no model file
needed): it builds the same synthetic network twice, once 4-D and once flattened
to 2-D from a fixed seed so the bytes are identical, and requires the predicted
control points to match exactly. Written before the fix and watched fail —
worst deviation `0.026871 px` before, `0.000000 px` after, suite 14/15 → 15/15.

### Sweeping the SR/denoise engines: two more quantized models were unusable

Having a cheap repro (quantize a local artifact, run the CLI, compare to the
source) made it worth checking the rest. **Nobody had ever run a quantized
esrgan or scunet** — both aborted on the first try. Neither is the `ne[3]`
bug; they are two distinct defects that only a quantized artifact reaches.

| engine | source | quantized | verdict |
|---|---|---|---|
| esrgan | `esrgan-x4-f32.gguf` | **aborted** | graph node budget — fixed |
| scunet | `scunet-color-f32.gguf` | **aborted** | flattened kernel in cache — fixed |
| pan | `pan-x4-f16.gguf` | ok | clean both ways |
| swinir | `swinir-light-x4-f16.gguf` | ok | clean both ways |
| tbsrn | `tbsrn-telescope-f16.gguf` | ok | clean both ways |

**esrgan — `GGML_ASSERT(cgraph->n_nodes < cgraph->size)`.** `esrgan_prep_conv`
already reshapes a flattened weight correctly, so the layout was never the
issue; the graph budget was. Measured on the 18-conv x4 model at 64×32: f32
builds **283** nodes against a `n_convs*12+100 = 316` budget (33 to spare),
but a quantized GGUF builds **335** — the dequant cast plus `ggml_cont` per conv
add ~3 nodes each — overflowing by 19. Budget is now `n_convs*16+128`
(~24 % headroom over the quantized measurement); it is graph metadata only, so
over-reserving costs nothing. q8_0 now matches f32 at cosine `0.999998`,
PSNR `51.89 dB`. **q4_k runs but degrades sharply — PSNR `29.55 dB`,
max_abs `91/255`** on a 2 MB network; treat q8_0 as the usable quant here.

**scunet — `GGML_ASSERT(a->ne[2] == b->ne[2])`.** This one *is* the flatten. The
persistent kernel cache in `scunet_init` copies the source `ne` verbatim
(`{t->ne[0], t->ne[1], t->ne[2], t->ne[3]}`, commented "ggml-native, as-is"), so
a flattened weight is cached as `[K*K*IC, OC, 1, 1]` and `ggml_conv_2d` sees
`ne[2]==1` instead of IC. The bytes are still in ggml kernel order, so
`scunet_run_conv` now restores the shape from the call-site dims. Conventions
were *measured* on the working f32 path rather than assumed — plain conv is
`[kw,kh,ic,oc]`, `conv_transpose_2d_p0` is `[kw,kh,oc,ic]` (ic=128 oc=64 →
`[2,2,64,128]`). q8_0 now matches f32 at cosine `0.999999`, PSNR `60.54 dB`,
max_abs `1/255`.

Regression control for both: the f32 output is **byte-identical** before and
after each patch, so neither touches the working path.

**The first version of that guard was worthless and it is worth recording why.**
It passed against the unfixed code. The synthetic model's `loc.fc2.weight` was
all zeros ("bias will provide the initial grid"), so the predicted points were
`fc2.bias` alone and did not depend on the conv stack at all — every assertion
downstream was blind to the channel count. The instrumented `channels=1` print is
what exposed it, not the green result. `fc2.weight` now carries small non-zero
random values so the conv stack actually reaches the output. A test that cannot
see the defect is not a test, and a passing new guard should be distrusted until
it has been observed failing.


## h2ovl-mississippi-2b — the degenerate output is NOT a graph bug (2026-08-02)

Two Kaggle runs under the full harness regime settled where this model actually
breaks. Recorded because the three standing suspects in PLAN were all
graph-level, and the evidence rules the graph out.

**`chr1str/h2ovl-2b-convert` failed on purpose.** Its decoded-output gate fired —
`rc=0` but **29 characters for a full page** — and it exited rather than publish.
No repo was created. A conversion that produces a loadable, structurally valid
GGUF (565 tensors, `use_msac` preserved, no missing LLM matrices) is not a
cleared model, and this is the gate doing its job.

**MSAC is eliminated as the cause.** The same run logs
`internvl2_ocr: MSAC two-scale tiling` and `606x1000 → 19 tiles (3x2 grid,
448px)`, so the two-scale tiling added in `45dfc704` demonstrably runs — and the
output is still degenerate. That was one of the three recorded suspects.

**`chr1str/h2ovl-2b-parity` then baked the reference and ran the diff:**

| stage set | result |
|---|--:|
| f16 vs `-ref.gguf`, **27 stages** | **cos_min `0.999972`** |

So vision encoder, projector, `llm_embed` and the first LLM decoder layers all
reproduce the Python blueprint to ~1e-5. **Scope, stated precisely so this is not
over-read:** the harness runs a synthetic gradient image and a 5-token synthetic
sequence with `--max-llm-layers 4`. It therefore does NOT cover the full 24-layer
stack, the real prompt tokens, sliding-window attention (never engages at T=5),
sampling, EOS handling, or detokenisation.

Within that scope the ported compute is correct, which relocates the bug to the
harness-blind zone (dev guide HARD RULE #3b) and matches the recorded rule
*"cos=1.0 but greedy diverges ⇒ the bug is the sampling loop, not the graph"*.
Remaining suspects, reordered by this evidence:

1. the Danube-1.8B chat/prompt template (most likely — it is the one thing the
   diff harness cannot see and the one thing H2OVL changes vs InternVL2)
2. EOS / stop-token id
3. the greedy/sampling loop and detokenisation

Dropped: Mistral sliding-window attention and the 32H/8KV GQA ratio — the
attention math is inside those 27 passing stages.

Artifacts: the reference is checkpointed unconditionally (it costs a full run to
produce) at **`cstr/crispembed-regression-fixtures`**, path
`internvl2/h2ovl-mississippi-2b/ref.gguf` (107 MB) — a new fixtures dataset
mirroring CrispASR's convention. Model artifacts stay unpublished: f16 4421 MB,
q8_0 2186 MB, q4_k 1391 MB all exist in-kernel but none may ship until the
decoded-output gate passes.


## h2ovl-2b — two independent defects, and a corrected metric (2026-08-02, local)

### Correction to the "27 stages" figure

`test-internvl2-diff` printed a row for **every** LLM layer even when the
reference held fewer. A reference dumped `--max-llm-layers 4` therefore produced
20 rows of `cos=1.000000 max_abs=0.000000 FAIL` — comparisons against an absent
tensor. That is why the parity kernel reported "27 stages": 7 real, 20 fabricated.
Fixed (`ref.has(name)` guard); the same diff now reports 7.

The f16 verdict survives the correction: the fakes report exactly `1.000000`, so
a `cos_min` of `0.999972` must have come from a real stage, i.e. all 7 real f16
stages passed. But the count was wrong and any future automated gate reading that
output would have been diluted by 20 free passes.

### q4_k destroys this LLM; f16 does not

Same reference, same binary, `h2ovl-mississippi-2b`:

| stage | f16 (Kaggle) | q4_k (local) |
|---|--:|--:|
| vis_patch_embed | — | `0.999999` PASS |
| vis_proj_output | — | `0.998630` FAIL |
| llm_embed | — | `0.999361` PASS |
| **llm_layer_0** | — | **`0.594995`** FAIL |
| llm_layer_1 | — | `0.217947` FAIL |
| llm_layer_2 | — | `-0.268615` FAIL |
| llm_layer_3 | — | `-0.279113` FAIL |
| **min over the 7 real stages** | **`0.999972`** | **`-0.279113`** |

Vision survives quantization and the token embedding is fine, then the decoder
collapses from the very first layer and goes anti-correlated by layer 2. That is
not quantization drift — f16 is clean at 1e-5 through the same stages — so the
q4_k recipe is wrong for this architecture (2560d, 32H/8KV ⇒ head_dim 80,
vocab 32010). The per-backend quant rules in `tools/quantize.cpp` are the place
to look; **do not ship a q4_k for this model.**

### The template fix is confirmed at the plumbing level, not yet at the output

With `5f617351`, the h2ogpt2 template is selected correctly on the published
q4_k (`no <|im_start|>/<|im_end|> in vocab but <|end|>=32009 present`) and MSAC
runs (`19 tiles (3x2 grid, 448px)`). Decoded output changed character: from a
single repeated token to **"The text is in English."** — a coherent, well-formed
reply. The model is now answering a prompt instead of emitting noise, which is
what the fix was supposed to achieve.

It is still not OCR, and there are two live explanations that this run cannot
separate, because the only artifact small enough to fit locally is the one with
the broken decoder:

1. the q4_k decoder collapse above, and
2. the instruction: we send `"OCR this image."`, while the model card drives it
   as `<image>\n{question}` with prose instructions ("Please describe the image
   in detail"). The image block placement already matches the card.

Next: run the gates at q8_0/f16, where the decoder is intact. Blocked locally on
disk (1.5 GB free, q8_0 is 2.3 GB); the parity kernel is computing both.


## h2ovl-2b — resolved, and the quant ladder (2026-08-03)

### Invocation: three defects, all required

Nothing here was a graph bug — `test-internvl2-diff` was clean at f16 the whole
time. The failures were all in the harness-blind zone:

1. **Wrong chat template** (`5f617351`, mine). The checkpoint declares
   `template: h2ogpt2` (`<|prompt|>…<|end|><|answer|>`); `build_prompt()` emitted
   InternVL2 ChatML. This vocab has no `<|im_start|>`/`<|im_end|>`, so
   `add_special(-1)` dropped every role marker silently.
2. **Spurious BOS + terse instruction** (`fcebf561`, parallel session).
   `add_bos_token: false` upstream; and `"OCR this image."` is too weak. Their
   2×2 A/B showed neither fix works alone.
3. **BOS default evaluated before vocab inference** (`98c2ae21`, mine). (2) only
   took effect for GGUFs carrying `internvl2.chat_template`. Without the key,
   `h2ogpt2` is unknown until `build_reverse_map()` scans the vocab — which runs
   *after* the default — so BOS stayed on. **Every published artifact is in that
   state.** Measured on the published q8_0: defaults → EMPTY output;
   `CRISPEMBED_INTERNVL2_ADD_BOS=0` → full transcription. After reordering,
   defaults transcribe.

Corroborating evidence that vision was never at fault: with only fix (1), the
same build reads `fox.png` as `The quick brown fox jumps over the lazy dog. 12345`
— exactly right. The full-page "serene landscape with a winding river" output was
confabulation on a hard page with a weak instruction, not a blind model.

### Quant ladder — Q8_0 is the floor for THIS checkpoint

Same reference, same binary, 7 real stages:

| precision | size | llm_layer_0 | llm_layer_2 | verdict |
|---|--:|--:|--:|---|
| f16 | 4421 MB | 0.999972 | 0.999972 | transcribes |
| **q8_0** | **2186 MB** | **0.998033** | **0.995498** | **transcribes** |
| q4_k + attn held Q8_0 | 1578 MB | 0.922039 | 0.543576 | still wrong |
| q4_k | 1391 MB | 0.594995 | −0.268615 | anti-correlated |
| q6_k | 1792 MB | — | — | **fails to load** |

Not a shape problem: every `ne[0]` involved (2560, 6912) is 256-divisible, so
Q4_K applies cleanly and still wrecks the decoder. Holding attention at Q8_0
recovers about half — kept, since it is cheap — but is not sufficient.

**It does not generalise.** A first version of this guard refused sub-Q8 for the
whole `internvl2` arch; that was wrong and would have broken two working models —
`internvl2-1b` ships q4_k in the registry and OCRs correctly, and `h2ovl-800m` is
recorded verified at q4_k. One measured checkpoint does not license blocking
others, so `tools/quantize.cpp` now **warns** and names the decoded-output gate
instead of refusing.

`vis_proj_output` sits at 0.998630 in every quant and 0.999972 at f16 — the
vision tower is Q8_0 in all of them, which is why the quants are bit-identical
there. Left alone deliberately: q8_0 transcribes correctly, and tuning a
synthetic-gradient cosine when the decoded output already passes is chasing the
wrong gate.


### Corroboration and the sibling regression (2026-08-03)

**Pre-fix baseline, from the parity kernel** (it cloned before the template fix,
so it captured the old behaviour at all three precisions on the same page):

| precision | pre-fix decoded output |
|---|---|
| f16 | `within.` (7 chars) |
| q8_0 | `withinself.` (11) |
| q4_k | `.assistant.assist.ass.assass.` (29) |

Two things fall out. f16 was degenerate too, so this was never a precision
problem. And the q4_k output is leaking the literal token **`assistant`** — the
ChatML role word being fed to a model that never saw it, which is the root cause
printing itself.

⚠ One number does not reconcile: that run reports q8_0 `cos_min 0.905481` where
the same comparison locally gives `0.988750`. Different backend (Kaggle CPU vs
Metal) and its count still included the 20 fabricated rows. The ladder's shape
agrees across both; neither figure should be quoted as exact until re-run on one
backend.

**Sibling regression check.** The BOS-ordering change alters the default for
every h2ogpt2 checkpoint, so `h2ovl-800m` (q4_k, in the registry) was re-run:

- `fox.png` → `The quick brown fox jumps over the lazy dog. 12345` — exact
- full page → **1748 chars**, opening `These two girls had been above an hour in
  the place…`, hyphenation preserved

So the 800m is unaffected and still correct **at q4_k** — which is the concrete
evidence that refusing sub-Q8 arch-wide would have been wrong, and that the
2b's collapse is checkpoint-specific rather than a property of the family.


### The q8_0 cosine gap: reconciled, and it is the harness input, not a CPU defect (2026-08-03)

The unreconciled figure above is explained. Same binary, same files, only the
backend varies:

| stage | Metal | CPU (`INTERNVL2_OCR_FORCE_CPU=1`) |
|---|--:|--:|
| vis_patch_embed | 0.999999 | 1.000000 |
| **vis_proj_output** | **0.998630** | **0.912992** |
| llm_layer_0 | 0.998033 | 0.982747 |
| llm_layer_3 | 0.988750 | 0.962142 |

CPU min is `0.912992` against the parity kernel's `0.905481` — also CPU, on a
different box. Hypothesis confirmed: the gap was backend, and the residual is
CPU-implementation/threading variation between machines. Neither number was
wrong; they measured different things.

**It does not reach the output, and that is the part that matters.** The same
CPU path on `h2ovl-800m` transcribes the full page at **1749 chars** against
Metal's **1748**, same text. So the divergence is confined to the diff harness,
whose vision input is a *synthetic gradient* — an out-of-distribution image that
amplifies numerical differences far more than a real page does. That also
explains why `vis_proj_output` reads FAIL against the 0.999 threshold on every
quant while every one of them reads real pages correctly.

Two consequences worth carrying forward:

- **Quote a vision-stage cosine with its backend, or not at all.** A single
  number here is meaningless — the same artifact spans 0.913 to 0.999 depending
  only on where it ran.
- **The 0.999 gate on `vis_proj_output` is mis-calibrated for a synthetic
  gradient.** It fails artifacts that decode correctly, which is a gate that
  cries wolf. Not changed here (a threshold tuned against one model is how
  gates rot); recorded so the FAIL is read as expected rather than chased.

### Registry health after the day's churn

All green, verified rather than assumed:

| check | result |
|---|---|
| SHA-256 pins | 242 pinned, 0 unpinned, `model_hashes.h` current |
| URL reachability | 242 distinct download URLs, **0 non-200** |
| Licence chains | `tests/check_registry_licenses.py` rc=0 |


## Vision-stage parity raised to the f16 ceiling (2026-08-03)

### The bisection that settles it: there is no code defect

The reference had carried `vis_layer_0..23` and `vis_pixel_unshuffle` since it
was baked, and nothing ever compared them — so a divergence between
`vis_patch_embed` (1.000000) and `vis_proj_output` (0.912992 on CPU) had 24
unbisected layers to hide in. The graph already names and `set_output`s every
one; the harness now reads them.

**At f16 every stage passes**: `vis_layer_*` 1.000000 → 0.999902,
`vis_pixel_unshuffle` 0.999691, `vis_proj_output` 0.999974, `llm_layer_*`
1.000000. The port is exact. Two things previously read as bugs are not:

- the "discontinuity" at `vis_layer_12` (0.90 → 0.64, max_abs 5.85 → 26.65) is
  quantization, not a code path — f16 is smooth through it;
- `vis_pixel_unshuffle` at 0.380 was **not** a layout/reshape mismatch, which
  was the obvious reading. It is 0.999691 at f16.

Everything measured was Q8_0 error compounding through 24 residual blocks.

### So the lever is vision precision, and it works

| stage | vision Q8_0 | vision F16 |
|---|--:|--:|
| vis_layer_11 | 0.900823 | **0.999994** |
| vis_layer_23 | 0.924021 | **0.999982** |
| vis_pixel_unshuffle | 0.380373 | **0.999691** |
| vis_proj_output | 0.912992 | **0.999974** |

Every vision stage now PASSes on CPU, the backend that was worst. 2186 → 2471 MiB
(+13%), and the page still transcribes. The decoder stays Q8_0: its stages are
0.98/0.96 and the output is right, while F16 there means the 4.4 GB file.

### Scoped to the Q8_0 target, because the first version was wrong

Applied to the whole arch, this rule took **`internvl2-1b` from 758 MB to
1135 MB** — the quantize step *inflating* the edge/WASM model 1.5×, defeating
the only reason that artifact exists. Now gated on `ftype == Q8_0`: the quality
tier buys parity, Q4_K stays the size tier. Verified both ways — edge q4_k gets
0 vision→F16 conversions, h2ovl q8_0 gets all 98. `CRISPEMBED_QUANTIZE_NO_VISION_F16=1`
bisects it without a rebuild.

That is the second rule this session that had to be narrowed after measuring a
sibling; the pattern to watch is a rule derived from one checkpoint being
applied to a family whose members have different goals.

### Shipped

`cstr/h2ovl-mississippi-2b-crispembed-GGUF` q8_0 replaced in place
(2591566112 bytes, sha `497cd047…`, verified byte-identical to the local file),
pin regenerated, 242 pinned / 0 unpinned.


## h2ovl-800m brought up to the same standard (2026-08-03)

The 800m shipped as q4_k only, with **no reference at all** — so its parity had
never been measured, only its decoded output eyeballed. Ran the full regime
locally (it is small enough: f16 is 1853 MB): convert → bake reference → quantize
→ per-stage diff → decoded output.

**f16 — the port is exact here too.** `llm_embed` and `llm_layer_0..3` are all
1.000000; `vis_proj_output` 0.999701. Two stages sit just under the gate —
`vis_layer_23` 0.998665 and `vis_pixel_unshuffle` 0.998199 — which is f16-vs-f32
rounding against a float32 numpy reference, the same class as the 2b, not a
defect.

**q8_0 with the vision tower at F16** reproduces the f16 vision numbers exactly
(0.998665 / 0.998199 / 0.999701), confirming the new rule does what it claims on
a second checkpoint.

### The finding worth carrying: the synthetic probe does not track decoded quality

| model / precision | llm_layer_2 | decoded page |
|---|--:|---|
| h2ovl-2b q4_k | **−0.268615** | fluent but wrong |
| h2ovl-800m q8_0 | **+0.494781** | **transcribes, 1764 chars** |

Two comparable-looking cosines, opposite verdicts. The 800m q8_0 shows
`llm_layer_1..3` at 0.70 / 0.49 / 0.66 with max_abs ~22 and still transcribes the
page correctly; the 2b q4_k at a numerically similar magnitude produces
confident nonsense.

The distinction that survives is **sign**: the 2b q4_k is *anti-correlated*
(−0.27), meaning the signal inverted, while the 800m is *degraded but aligned*
(+0.49). A per-stage threshold alone would have rejected a good artifact and,
at a slightly different cut, accepted a bad one. This is the concrete case for
why the decoded-output roundtrip is the gate and the cosine is the bisection
tool — exactly what HARD RULE #3 says, now with a counterexample in-repo.

**Registry unchanged for the 800m: it stays on q4_k.** That artifact transcribes
(1749 chars, fox exact) at 676 MB against the q8_0's 1175 MB, and this is the
edge/WASM model. The q8_0 is published as a quality tier, not promoted.


## CORRECTION: I was reading a per-row worst case as tensor parity (2026-08-03)

`Report::cos_min` is the **minimum cosine over rows**. On the diff harness's
5-token synthetic probe, one fragile token position sinks it while the tensor as
a whole matches. I quoted it as if it were the stage's parity for most of this
session, and built a conclusion on it that was wrong.

`Report` has always carried `cos_global`, `mine_norm` and `ref_norm`; **no print
site ever showed them** — so HARD RULE #2b ("cosine is scale-blind, always read
`|mine|` vs `|ref|`") was being violated by the tool itself. Every internvl2
print site now emits `cos_min`, `cos_glob`, `max_abs`, `|mine|`, `|ref|`.

### What the missing columns were hiding

| artifact / stage | cos_min | cos_glob | \|mine\| vs \|ref\| |
|---|--:|--:|---|
| 800m q8_0 llm_layer_2 | 0.494781 | **0.999975** | 2762.7 / 2779.2 |
| 2b q8_0 llm_layer_3 | 0.962142 | **0.999300** | 20.62 / 20.69 |
| 2b q4_k llm_layer_3 | −0.270306 | **0.968178** | 21.24 / 20.69 |

The magnitudes also explain the "max_abs 22" that looked catastrophic: the
activation norm legitimately jumps **18.85 → 2510** between LLM layers 0 and 1 —
**in the reference too**. This model has massive activations; an absolute error
of 22 on a norm-2510 tensor is 0.9%, entirely proportionate. Without the `|ref|`
column that was unreadable.

### Two verdicts corrected

**h2ovl-800m q8_0 is NOT degraded.** I recorded its LLM as "cratering" at 0.49.
Globally it is 0.999975 — as clean as anything here. The artifact was always
fine; the metric was wrong.

**The "sign is what survives" finding is WITHDRAWN.** I wrote that a positive
0.49 transcribes while a negative −0.27 fails, and that sign was the durable
discriminator. That was pattern-matching on two per-row worst cases. The honest
discriminator is the **global** cosine: 0.999975 (800m q8_0, correct output) vs
0.968178 (2b q4_k, wrong output). Sign had nothing to do with it.

**The q4_k withdrawal still stands, for a properly stated reason.** Not
"anti-correlated by layer 2" — that was `cos_min`. It is `cos_glob` decaying
0.994215 → 0.968178 across just 4 of 24 layers, ~45x the shipped q8_0's error at
the same depth, compounding over the remaining 20, with a decoded output that
was wrong. The decision was right; my justification for it was not.

### Gate note

`is_pass()` is keyed on `cos_min` at 0.999, so on this probe nearly every stage
prints FAIL while being globally excellent — a gate that always cries wolf
teaches people to ignore it. Not changed here: `crispembed_diff.h` is shared by
every engine and re-keying it on one model's evidence is the exact mistake this
session already made twice. Recorded so the FAILs are read correctly, with the
columns now present to do that.


## Full 54-stage trace to the logits (2026-08-03)

The reference had only ever been dumped with `--max-llm-layers 4`, which also
silently skips `llm_output_norm` and `llm_logits` — two stages
`test_internvl2_diff.cpp` already knew how to compare and had never been given
data for. So the harness stopped 20 layers short of the decision boundary. Re-dumped
without the cap: **54 stages** (27 vision + 24 LLM + embed + output_norm + logits),
108 MB, at `internvl2/h2ovl-mississippi-2b/ref-full.gguf`.

Shipped q8_0 (vision F16), CPU, against all 54:

| stage | cos_min | cos_glob | \|mine\| / \|ref\| |
|---|--:|--:|---|
| vis_patch_embed | 1.000000 | 1.000000 | 124.51 / 124.51 |
| vis_pixel_unshuffle | 0.999695 | 1.000000 | 1739.23 / 1739.25 |
| vis_proj_output | 0.999974 | 1.000000 | 183.15 / 183.15 |
| llm_layer_0 | 0.982747 | 0.999863 | 7.77 / 7.79 |
| llm_layer_11 | 0.967229 | 0.999066 | 86.62 / 86.52 |
| llm_layer_23 | 0.525327 | 0.998276 | 980.20 / 961.45 |
| llm_output_norm | 0.629808 | 0.997297 | 323.44 / 331.49 |
| **llm_logits** | 0.613839 | **0.998919** | **1602.46 / 1604.54** |

**The logits — the actual decision boundary — reproduce the blueprint at
cos_glob 0.998919 with magnitudes 0.13 % apart.** That is the first time this
model has been measured there at all, and it is the number that matters: token
selection happens on this tensor.

The `cos_glob` curve declines smoothly and monotonically from 0.999863 to
0.998276 across 24 layers with **no discontinuity anywhere** — the signature of
accumulating quantization error, not a defective op. Every earlier "jump" I
chased (`vis_layer_12`, `llm_layer_1`) was an artifact of reading `cos_min`.

Two things the trace does flag, neither load-bearing today:

- `llm_output_norm` is the weakest global stage (0.997297) and the only one whose
  magnitude is off by more than ~1 % (323.44 vs 331.49, −2.4 % low). The logits
  recover to 0.998919 immediately after, so it is not propagating, but it is the
  one stage worth a second look if this model ever misbehaves again.
- `|mine|` runs 1.9 % high at `llm_layer_23` (980.20 vs 961.45) — the massive-
  activation dimensions amplifying q8_0 rounding. Bounded and self-correcting
  through the final norm.


### f16 ceiling: the port is exact to the logits

Same 54-stage reference, f16, CPU:

| stage | cos_min | cos_glob | max_abs | \|mine\| / \|ref\| |
|---|--:|--:|--:|---|
| vis_proj_output | 0.999974 | 1.000000 | 0.003354 | 183.1485 / 183.1475 |
| llm_layer_0 | 1.000000 | 1.000000 | 0.000004 | 7.7938 / 7.7938 |
| llm_layer_11 | 1.000000 | 1.000000 | 0.000033 | 86.5238 / 86.5239 |
| llm_layer_23 | 1.000000 | 1.000000 | 0.001282 | 961.4495 / 961.4499 |
| llm_output_norm | 1.000000 | 1.000000 | 0.000217 | 331.4937 / 331.4938 |
| **llm_logits** | **1.000000** | **1.000000** | **0.000069** | **1604.5433 / 1604.5439** |

**Every stage passes, `cos_min` included.** The h2ovl-2b port reproduces the
Python blueprint exactly, all the way to the tensor token selection reads. There
is no code defect anywhere in this engine for this model.

That also settles the `cos_min` question directly rather than by argument: at f16
the per-row minimum is 1.000000, so the low `cos_min` values at q8_0 are
quantization landing on numerically fragile rows — not a structural mismatch and
not a harness artifact. And `llm_output_norm`, flagged above as the weakest
global stage at q8_0 (0.997297, −2.4 % magnitude), is exact at f16
(331.4937 vs 331.4938), so that too is quantization, not the norm implementation.

Complete verdict for h2ovl-mississippi-2b:

| precision | logits cos_glob | decoded page |
|---|--:|---|
| f16 | 1.000000 | transcribes |
| q8_0 (shipped, vision F16) | 0.998919 | transcribes |
| q4_k | — (withdrawn) | fluent but wrong |


## Diff-gate regime, and full references for both checkpoints (2026-08-03)

### The gate was right; the regime was unstated

`is_pass()` keys on `cos_min`, and the f16 trace proves that is the **correct**
gate for port correctness: h2ovl-2b at f16 scores `cos_min` 1.000000 on all 54
stages including the logits, so a single mishandled token position would still
fail loudly — which is what this harness is for.

It is the wrong gate for judging a *quantized* artifact, where the question is
"is the damage acceptable", not "is the port correct". The same model at q8_0
reads `cos_min` 0.61 on the logits while `cos_global` is 0.998919 and it
transcribes a page correctly.

So rather than weaken a header shared by every engine on one model's evidence:

- **default unchanged** — 0.999 on `cos_min`; verified f16 still scores 55 PASS
  / 0 FAIL after the change;
- `CRISPEMBED_DIFF_COS_THRESHOLD` overrides the threshold per run, for quantized
  sweeps;
- `is_pass_global()` added for callers whose question is the aggregate.

Additive: no existing engine's verdict moves.

### Both references now reach the logits

Both had been dumped `--max-llm-layers 4`, which also silently drops
`llm_output_norm` and `llm_logits`. Re-dumped and republished to
`cstr/crispembed-regression-fixtures`:

| checkpoint | stages | covers |
|---|--:|---|
| h2ovl-mississippi-2b | **54** | 24 vision + 24 LLM + embed + output_norm + logits |
| h2ovl-mississippi-800m | **46** | 24 vision + 16 LLM + embed + output_norm + logits |

The 2b model repo's copy was replaced too, so neither location hands out a
reference that stops short of the decision boundary.

## CPU baseline of shipped binaries (2026-08-04)

`-DGGML_NATIVE=OFF` is now pinned on every release/wheel leg (#41 — a runner
with AVX-512 shipped a Windows cpu artifact that took SIGILL on Raptor Lake).
This is a **portability** change, not a perf change, and the distinction
matters for every number in this file:

- **ggml-cpu** on x86_64 goes from the runner's `/arch:AVX2` or `/arch:AVX512`
  lottery to a fixed SSE4.2+AVX+AVX2+FMA+F16C+BMI2. Where the lottery had
  landed on AVX2 — which is what the AMD runners give, and what all the
  measurements below were taken on — the emitted code is unchanged.
- **`core/cpu_ops.h`** keeps its SIMD paths. `CRISPEMBED_NATIVE` now follows
  `GGML_NATIVE`, and with native off the crispembed targets are compiled
  `-mavx2 -mfma -mf16c` instead of dropping to scalar, so the `dot_product()` /
  `linear_cpu` AVX2+FMA work recorded above is still in the shipped binaries.
- **linux-arm64** pins `armv8.2-a+fp16+dotprod`, which is what `-mcpu=native`
  was already resolving to on the Neoverse-N1 runners — the dotprod quant
  kernels are retained.

So release timings remain comparable to the AVX2 numbers in this file. What is
gone is the possibility of a release built with AVX-512 kernels: nobody should
benchmark a release artifact and attribute a delta to an ISA the artifact is
no longer allowed to use. Benchmark *local* builds (`-march=native`, the
default) when the question is peak CPU throughput, and say which of the two
you measured.

## SmolDocling DocTags contract fix — stage split after the tiling port (2026-08-04, M1, CPU-only build)

The T15 output-contract fix (added-token vocab restore, reference tiling
preprocessing, max-tokens wiring — PLAN T15) changes the engine's cost
shape: vision now runs N tiles + 1 global view per page instead of one
squashed 512x512, so vision dominates end-to-end time. Measured with
`CRISPEMBED_SMOLDOCLING_BENCH=1`, q8_0, warm, CPU backend (build had
GGML_METAL=OFF; the engine is CPU-hardcoded regardless — see the backend
matrix row):

| fixture | grid (tiles+global) | vision+connector | prefill | decode | total |
|---|---|--:|--:|--:|--:|
| fox.png 800x200 | 1x4 + 1 = 5 | 31,744 ms | 3,760 ms (347 tok) | 1,332 ms (44 tok) | 37,296 ms |

Per-512px-tile SigLIP forward is ~6.3 s CPU — the single dominant cost.
Full pages (12 tiles + global, e.g. 606x1000 scan_page_pd) measured
71-103 s wall end-to-end in the same configuration. Quality on the same
run: fox payload CER 0.86 -> 0.0 (payload exact vs ground truth), prompt
ids byte-identical to the reference processor (347/347). The obvious next
perf lever is the backend port of the SigLIP graph (matrix row TODO):
per-tile vision is compute-bound, exactly the shape that wins on GPU,
while the 135M-LLM per-token decode is the shape recorded as CPU-favored
(see the persistent-decode LEARNINGS entry) — split residency, don't move
the decode blindly.

## SmolDocling vision split residency (G1/F4, 2026-08-05, M1, Metal)

The split-residency port landed: `vis.*` weights go to the GPU backend via
`core_gguf::load_weights_split`, the SigLIP graphs run there, and the
connector + 135M decode + LM head + KV stay CPU. `SMOLDOCLING_FORCE_CPU=1`
/ `--gpu-backend cpu` restore all-CPU. Same-binary interleaved (metal,cpu)
pairs, loadavg-gated, pair 0 discarded, median of 3
(`tests/results/g1/SUMMARY.md` has the full gates):

| fixture | stage | metal | cpu | speedup |
|---|---|--:|--:|--:|
| fox.png (5 tiles) | vision+connector | 1,241 ms | 3,562 ms | 2.9x |
| fox.png | total | 1,876 ms | 4,216 ms | 2.25x |
| scan_page_pd (13 tiles) | vision+connector | 3,163 ms | 14,500 ms | 4.6x |
| scan_page_pd | total | 11,021 ms | 22,681 ms | 2.06x |

Decode step time is unchanged (~12-19 ms both arms — decode never moved).
NOTE the 31.7 s fox vision row above came from a DIFFERENT (CPU-only,
2026-08-04) build; the same-binary CPU baseline today is 3.6 s, so the only
apples-to-apples claim is this table's interleaved pairs. Decoded output:
CPU arm byte-identical to the T15 recorded outputs 5/5; Metal arm
matches/beats the reference implementation on every GT-labelled page, with
one documented CPU-vs-Metal trajectory divergence (receipt_historical,
stripped CER 0.238 CPU vs 0.372 Metal, both better than the reference's
0.493 — Metal F16 activation rounding, same class as deepseek T14/G2).
