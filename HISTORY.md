# CrispEmbed — History

Completed milestones and work log. See PLAN.md for current roadmap.

---

## August 6, 2026 — all-queue session (active-work board archive)

Shipped rows moved off the PLAN.md active-work board at the round-N+1
handover. Evidence: PERFORMANCE.md dated sections + commits named inline.

| Since | Branch / worktree | Task | Status |
|-------|-------------------|------|--------|
| 2026-08-06 | *(scoping landed via `perf/glm-grid-sizing`, no code change)* | **GLM round 2 CLOSED as falsified — all three cheap ViT levers lose to measurement.** (1) Threads: plumbing correct, ViT scales 3.0x (-t1 9.49 → -t4 3.15 s). (2) Grid: the 'padding on wide-short images' theory DIED on the fixtures' actual dims — fox is 800x200 (160K px) vs strip 520x260 (135K px); the −16% time is exactly area-proportional and the Qwen2VL-style smart-resize (min 12,544 / max 9.6M px, aspect-preserving) is behaving correctly. (3) Quant: the q8 artifact's vision tower is already 460 MB Q8_0 mat-muls + ~4 MB F32/F16 norms — no miss. ⇒ GLM ViT-on-CPU is genuinely compute-bound (~3.1 s/160K px at -t4 M1; the field user's ~24 s/image is consistent with a 2-core i3). Remaining levers are DEEP work (CPU flash-attn arm A/B, kernel-level q8 matmul, earlier spatial merge) — dedicated session, uncertain payoff; for subtitle users the practical guidance is ppocr (6.4 s/img CPU, 'surprisingly good') over VLM engines. Rider recorded: glm hallucinates continuation text on thin strips | **DONE (falsified)** |
| 2026-08-06 | *(landed via `fix/vlm-ram-guard`)* | **VLM RAM preflight DONE + GLM CPU perf SCOPED.** (a) New `core/ram_guard.h` wired into deepseek_ocr2/glm_ocr/unlimited_ocr init: refuses (default) or warns (`CRISPEMBED_RAM_GUARD=warn`; `=0` disables; `…_AVAILABLE_MB=N` test hook) when weights×1.3+1 GiB exceeds available RAM — the 4 GB-host freeze (SubtitleEdit-13238) now fails in milliseconds with the numbers instead of thrashing the machine. Gates: simulated 3200 MiB box refuses the 2148 MiB uocr model rc=1; warn/disable/normal arms byte-identical output. (b) **GLM CPU perf scoped, CORRECTED on a quiet box**: thread plumbing is FINE (ViT scales 3.0x, -t1 9.49 s → -t4 3.15 s on fox; the first 7.9 s@-t4 reading was load-skewed) — the REAL lever is the input grid: subtitle-shaped `scan_strip` (a fraction of fox's area) still costs 2.65 s vs 3.15 (−16% only) ⇒ preprocessing pads wide-short images to a near-full grid; round-2 = grid sizing for subtitle crops + `GLM_OCR_VISION_FLASH` CPU arm. Rider observed: glm HALLUCINATES continuation sentences on the strip (invented text beyond the image) — VLM-on-thin-strip failure mode, record when the lane opens. (c) crispstrobe-ops `00285218` NOT needed here — it was backported FROM this lineage (`1f849774`). deepseek CPU perf: mooted on <8 GB hosts by (a); adequate-RAM profile is its own session | **DONE** |
| 2026-08-06 | *(landed via `fix/glm-no-repeat-ngram`)* | **GLM-OCR repetition guard DONE** (SubtitleEdit PR-13238 user feedback): `glm_ocr` was the ONE VLM engine decoding with a bare argmax — wired the shared F1 `argmax_no_repeat_ngram` (ngram 3, `GLM_OCR_NO_REPEAT_NGRAM=N` override). Red-green on `german_kurrent_handwriting`: guard-off reproduces the reported pathology ('-ich' ~30x runaway, 2051 B), guard-on breaks it (205 B); clean fox byte-identical both arms. **Two more items from the same feedback, recorded unowned:** (1) PP-OCRv6 on a 2-core/no-GPU box = ~6.4 s/subtitle-image ('surprisingly good quality', 4m16s vs Paddle-standalone 47s) — the O13b portability gap with a real user attached; the R6/MK opt-in kernels + O7 per-engine adoption are the levers. (2) **deepseek-ocr2 froze a 4 GB-RAM desktop 3/3 before the first output** — swap-thrash class; needs a model-size-vs-RAM preflight guard (refuse or warn instead of thrashing the host) | **DONE** |
| 2026-08-06 | *(landed via `probe/vulkan-bringup`)* | **O10 DONE — Vulkan on Kaggle: NO.** Kernel `chr1s4/crispembed-vulkan-probe` v2 (P100): the container mounts NO NVIDIA Vulkan userspace (no ICD manifests, no `libGLX_nvidia.so.0` — the NVIDIA container runtime exposes compute+utility only, never graphics), so no device can exist regardless of loader/tools; writing `nvidia_icd.json` cannot help. Secondary: `glslc` not apt-installable on jammy (shaderc prebuilt needed) — moot behind the capability wall. ⚠ v1 reported fake stage successes via `\| tail` rc-laundering (HARD RULE 8, in the probe itself) — only v2 is evidence. Consequence: CPU lavapipe stays the only Kaggle-viable Vulkan validation; real-GPU Vulkan verdicts need non-Kaggle hardware. Evidence `PERFORMANCE.md` top entry | **DONE** |
| 2026-08-06 | *(investigation landed via `fix/pageseg-round2`, no code change)* | **Pageseg round 2 SCOPED, not fixed — the "truncated row" is a symptom of fragment-level row grouping.** `CRISPEMBED_TESSERACT_LEGACY_ROW_DEBUG` row map on the Fraktur page: the legacy segmenter's blob **median height is 6 px** (ink fragments, not letters, at threshold 82) so `max_row_gap=4` — the greedy y-chain assigns lines on fragment adjacency, which is why an entire Inhalt continuation line ("Auditoriats auf das Reichsmilitärgericht…") vanishes and its predecessor is clipped mid-word. NOT the `column_ink>=4` x-tightening suspected in round 1 (that code is in the projection path, which doesn't run here). A robust fix = baseline-clustered line assignment (assign fragments to lines by baseline estimate, not y-adjacency chains) — a dedicated-session project, wrongly sized as an increment. Round-1's 0.271 stands; decoder levers (`--recode-beam`/`--dawg-score`) remain the other unmeasured residual axis. Also caught: the ROW_DEBUG `out=yes/filtered` label is index-based after an unrelated sort — unreliable, fix when the lane reopens | **DONE (scoping)** |
| 2026-08-06 | *(landed via `fix/pageseg-quality`)* | **Fable task 4 ROUND 1 DONE — classical Fraktur CER 0.412 → 0.271, route faster (1307→1115 ms): the garbage was RULES, not recognition.** Per-line/crop-dump diagnosis: the masthead's two horizontal rules fed the LSTM ~350 junk chars (native 1110 vs official 881); the 3 gated segmentation policies are all far worse (4.14/0.98/1.59 — over-split), legacy geometry was fine. Fix: `reject_separator_rows` (both segmenters; `…KEEP_RULES=1` opt-out) with a MEASURED discriminator — first cut (max-scanline ≥80%) missed the dashed 1899 rules (33-35%); the real separation is **column coverage** (rules 0.99-1.00 dashed-or-not, text ≤0.76) + height ≤0.6×median + aspect ≥8. Gates: synthetic 20-corpus arms identical (0.02210), 4 CC0 fixtures byte-identical, orchestrator+spot checks green, dbnet route untouched. **Round-2 residuals to 0.235 documented in PERFORMANCE.md**: ornament fragments (~60 chars, needs confidence/adjacency signal), one right-truncated row (split-band `column_ink>=4` x-tightening suspect), recoder/decoder semantics (`--recode-beam`/`--dawg-score` unmeasured post-int8-cache) | **DONE** |
| 2026-08-06 | *(landed via `perf/r6-gemm-microkernel`)* | **Fable task 3 DONE — R6 register-blocked GEMM micro-kernel, wins on BOTH architectures, ships opt-in.** Outer-product consume on packed [k][r] weight / [k][c] column blocks (NEON 4x4 `vfmaq_laneq`, AVX2 8x4 broadcast-fmadd, scalar fallback; dot-consume remainders), `CRISPEMBED_CONV2D_MK=1` implies the GEMM path; bitwise tile path untouched as the exact-equality reference arm, mk arm tolerance-gated ≤1e-4 (196/196, 9 shapes × nt=1,4). **M1: process CPU 25.8-27.0 → 17.7-18.0 s (−34%), text byte-identical** — wins where the plain interchange LOST (the mk win is FLOP-side, survives a big L2). **x86 (kernel `chr1s4/crispembed-mk-ab` v1, Xeon AVX2): unit gate 196/196 on the AVX2 path's first run; CPU legacy 252.8-254.5 → mk-nt1 238.7-241.7 s (−14 s ≈ −40% of the det-stage share), output byte-identical across ALL 5 arms** — stacks on the interchange as the small-L2 story predicted. Per-engine default adoption stays the O7 item. Evidence `PERFORMANCE.md` top entry | **DONE** |
| 2026-08-06 | *(landed via `fix/sched-replay-crash`)* | **Fable task 5 DONE — the O6 replay crash ROOT-CAUSED + FIXED in the fork (`890278a8`): sched restore-on-success broke alloc-once/compute-many.** Split rewires user srcs to input copies and restored them on EVERY compute exit ⇒ a second compute of the same allocation ran with original cross-backend srcs ⇒ Metal dereferenced a CPU buffer as a Metal buffer (poisoned AGXBuffer, gen=2 SIGSEGV). Fix: log records the rewired src, compute re-applies at entry, disposal is state-flagged (applied ⇒ write originals back — the ppocr build-then-run flow needs it; restored ⇒ pure clear — engines that rebuild graphs over old memory need THAT; both wrong cuts caught by regressions and recorded). New fork diagnostic `CRISPASR_METAL_PROFILE=3` (pre-encode node/src-buffer trace) did the localization. Gates: replay repro runs clean; ppocr `a3a5f938` + layout `afde4fd4` byte-identical; uocr default unchanged; 4/4 spot checks. PD verdict unchanged (still divergent, replay ≈ re-init) — the win is the sched CONTRACT + unblocking R8 ICB replay. Evidence `PERFORMANCE.md` top entry | **DONE** |
| 2026-08-06 | *(landed via `perf/layout-o2b-valueproj`)* | **Fable task 6 DONE — layout O2b value-proj batched onto the GPU, 3.5x on the stage, ships OPT-IN.** Fresh Phase-2 profile first (post flat-im2col): value-proj is the new #1 stage at 178-211 ms of 545-588 ms (`cpu_linear` already AXPY+threaded = memory-bound ceiling, no CPU headroom). Structural find: `memory` is constant across all 6 decoder layers ⇒ ONE gallocr graph (one upload, shared transpose, 6 mul_mats, one readback) ⇒ **value-proj 55 ms, Phase 2 −35% (→366-385 ms)**. NOT byte-identical (GPU contraction): 3/6 fixtures byte-identical, 2 sub-pixel/±0.001 jitter, but **commons_test flips a region at score exactly 0.500 across the threshold (13→14)** — LAYOUT_CONV_F16 drift class ⇒ **default stays CPU** per the A/B rule (`CRISPEMBED_LAYOUT_VALPROJ_GPU=1` opts in; default verified byte-identical to baseline). Follow-ups recorded: full-fixture region gate / score hysteresis / CUDA arm. Evidence `PERFORMANCE.md` top entry | **DONE** |
| 2026-08-06 | *(landed via `fix/cuda-rec-zero-results`)* | **Fable task 1 DONE — CUDA-rec 0-results ROOT-CAUSED + FIXED + O11 SHIPPED.** Not CUDA-specific and not readback/CTC (both v2-capture suspects wrong): `pp_graph_resident`'s native-quant path uploaded raw F32 bytes into a q8_0-typed resident — misread f16 scale bytes ⇒ NaN logits ⇒ NaN-poisoned `max_element` returns index 0 = CTC blank ⇒ every crop decodes empty. **Reproduced on M1 Metal with the q8-head artifact** (Metal was only ever validated on f16); fix = same-type residents copy raw source bytes (host-staged). **P100 proof (kernel `chr1s4/crispembed-cuda-rec-fix` v1+v2): q8 CUDA fused graph BYTE-IDENTICAL to the scalar reference on both fixtures** (strip 12=12, page 38=38, sim 1.0000), f16 control clean. **O11 shipped same branch:** det residency per-backend-kind (CUDA→GPU-graph via a ~29 ms device-name probe, Metal keeps CPU); CUDA auto-engagement proven in v2 — **page detect 9516→595 ms (16x)**, boxes unchanged, page total 12.7→3.8 s; M1 default untouched (byte-identical). Gates: 12/12 model-free CI, M1 q8 0→12 results, f16 baseline byte-identical. Evidence `PERFORMANCE.md` top entry | **DONE** |
| 2026-08-06 | *(landed via `perf/ppocr-rec-profile`)* | **Fable task 2 DONE — rec Metal profile: im2col was 70% of the graph; flat-dispatch kernel = recognize 2.3x (13.4-14.0 → 5.8-6.2 s), layout_detect 1.6x, byte-identical.** All three brief candidates FALSIFIED by measurement: readback ~1%, the 18,710-class head ~1-2% (STOP=decoder ≈ full), batching perfectly linear; bonus negative: `CONV_DIRECT=1` is a **6-7x regression** (3/3 pairs). Real hotspot: the `ggml_conv_2d_dw` im2col lowering (IC==1, N=C*batch) at ~0.4 GB/s on the standard Metal kernel. Fix = ggml fork `89a2039d` (`sync/upstream-v0.17`): `kernel_im2col_flat`, predicate `N*KH*KW<128 \|\| IC==1`, `CRISPASR_METAL_IM2COL_FLAT=0` restores legacy; two failed cuts recorded (int64 divmods, no-grid) — the win needed 32-bit grid-borne indexing. Gates: 18-fixture ppocr sweep byte-identical, 13/13 model-free CI, backend-smoke. New permanent diagnostics: GRAPH_BENCH stage/alloc/readback split, GRAPH_STOP stem/stage1-3. O1's layout "Phase 1 wall" revised down (1.56→0.95 s). Evidence `PERFORMANCE.md` "PP-OCR rec Metal batch profile" | **DONE** |

## August 5-6, 2026 — OCR runtime optimization round (active-work board archive)

Shipped rows moved off the PLAN.md active-work board at the round handover.
Evidence for every row: PERFORMANCE.md dated sections + the commits named inline.

| Since | Branch / worktree | Task | Status |
|-------|-------------------|------|--------|
| 2026-08-06 | *(landed via `fix/nthreads-plumbing`)* | **Ignored-n_threads audit (O13b bug class) — 3 more engines fixed**: `ppocrv6_ocr` (the RECOGNIZER — all its CPU modes ran at ggml default regardless of `-t`), `pplcnet_orientation`, `pix2struct` (CPU-ONLY engine with `(void)n_threads`!), `easyocr_ocr`. All now apply the caller thread count to CPU backend + sched fallback. Live check: pix2struct `-t 4` 1.46x vs `-t 1`, output byte-identical. With ppocrv6_det (O13b) that is **5 engines** that silently dropped `-t` | **DONE** |
| 2026-08-06 | *(kernel `chr1s4/crispembed-conv-ab` v2 COMPLETE — P100 again, no T4 draw)* | **v2 results**: v1's O8/O9a/O9c verdicts REPLICATE (gemm4 17.5-20.4 s, det-CUDA 550-614 ms vs CPU 9.5-11.0 s, f16-layout region drift again). **O9d dbnet arm INCONCLUSIVE as designed** — whole-pipeline wall (114.3 s CPU-det vs 111.7 s CUDA-det) is TrOCR-rec-dominated so det isn't isolated, and the two arms' decoded text DIFFERS (det backend changes boxes/probmap — needs a det-only harness + box-level compare in v3, not a wall clock). **O9b rec-debug capture is the diagnosis**: on CUDA the fused batch graph computes NUMERICALLY SANE activations through stage4 and builds the 18710-wide logits graph, yet results=0 — the fault is AFTER the graph: suspect the logits READBACK layout or the CTC decode consuming it (a scrambled readback argmaxes onto blank → CTC collapses every crop to empty, matching 0 results with rc=0). Full stderr archived in the kernel output. This is Fable-queue item 1's starting point | **DONE** |
| 2026-08-06 | *(landed via `fix/uocr-pd-segfault`)* | **O6 DONE — UOCR_PD=1 no longer crashes.** Trigger isolated (4/4 reproducible, Metal build): **replaying the sched-allocated PD graph without re-alloc** dies at gen=2 inside ggml-metal op encode; forcing full reset+alloc+KV-reupload per step runs 1024/1024 clean. That safe re-init is now the PD default; the crashing replay is `UOCR_PD_REPLAY=1` (sched/Metal debugging only). Honest verdict: PD-with-reinit ~183 ms/step vs rebuild ~90 ms/step AND still divergent — **PD has no winning mode**, stays opt-in research. Rider: `[decode]` summary printed use_pd=1 on rebuild runs (re-derived, not reported) — fixed. Default output byte-identical. NOTE for the ggml lane: the fork's sched does not support alloc-once/compute-many on this MoE graph — one earlier "concurrency" theory was RETRACTED (built on wrong-build-dir CPU runs) | **DONE** |
| 2026-08-06 | *(measurement-only, docs on `main`)* | **O4/R1 PREMISE DEAD — recognition is 0.4 s, not 38.3 s** (Fraktur page, 3 repeats): already fixed by the default-ON int8 recurrent cache (`379434b1`/`e49d390d`, verified 7.5x by disable-arm) + Metal-init skip + scratch reuse; survey's "gated" label was wrong (opt-out). Official also re-measured: **1.8 s not 9.3 s**; CER improved 0.528→**0.235**. The gap is now ALL dbnet detection (3.8 s = 88% of the 4.3 s stage); the classical-pageseg route is **faster than official (1.2 s vs 1.8 s)** at CER 0.412. Successors: dbnet CUDA arm (add to conv-ab v2), pageseg quality lane; recognizer batching is dead. Full table in the rewritten R1 section + `PERFORMANCE.md` | **DONE** |
| 2026-08-06 | *(kernel `chr1s4/crispembed-conv-ab` v1 COMPLETE — P100 + 2 GHz Xeon)* | **O8+O9 verdicts in.** O8 (x86): unit gate **180/180 on AVX2 both arms** (bitwise holds); 3 interleaved rounds, 38 regions all arms: legacy 34.7-37.2 s, **gemm nt=1 30.8-31.5 s (interchange alone WINS ~13% on small-L2 x86 — the M1 −5% verdict was L2-size-dependent, hypothesis confirmed)**, gemm nt=4 20.3-20.4 s (**1.76x**). O9: **PP-OCR det verdict FLIPS on CUDA — graph 596-614 ms GPU vs 11.0 s CPU (18x), boxes 38=38**; `LAYOUT_CONV_F16` on P100 (no tensor cores): no speed change (66→66 ms warm) and **regions drift 20→19** — stays gated everywhere measured; layout Phase 1 on CUDA is 60-110 ms (M1 Metal's 1.4 s is Metal-specific). **New bug found: PP-OCR REC on CUDA emits 0 results in both arms** (boxes=38, rec runs 3.6 s, empty output — det verdict unaffected; needs the cuda-diag treatment, unowned). v2 TODO: dbnet CUDA arm + a T4 rerun for the tensor-core f16 question | **DONE** |
| 2026-08-06 | *(landed via `perf/o5-decode-overhead`)* | **O5 DONE — R5 decode-graph caching CLOSED for all 3 candidates**: qwen2vl build+alloc **2.2%** of decode (446/20708 ms, 125 steps, existing QWEN_DBG timers); granite **9.2%** (1075/11660 ms, 180 steps, NEW permanent build/compute split on its decode bench line) — single-digit both, persistent-graph port not worth it (deepseek T14 repeats); **smoldocling: premise structurally wrong** — its decode is a hand-written CPU loop with a HOST vector KV, no graph exists to cache (and granite's DEFAULT is the per-step ggml graph, scalar is fallback — survey wording corrected) | **DONE** |
| 2026-08-06 | *(landed via `perf/ppocr-profile`)* | **O13b DONE — PP-OCR medium-tier profile named the hotspots + 2 fixes landed**: recognize = **71-74%** of the page (Metal fused batch graphs, 2-4 s per width group, 18,710-class CTC head — the #1 PP-OCR lever), detect = 25%. **Bug fixed: `ppocrv6_det::init` declared but NEVER APPLIED its n_threads param** (every `-t` silently ran ggml default; now honored, text byte-identical, det graph saturates at 4 threads = memory-bound). Width-bench diagnostic fixed (printed pre-bucket widths: "27 unique" vs the real 8 graph shapes). Backend truth: rec Metal 17-19 s vs CPU-graph 69 s vs CPU-scalar 107 s — Metal default correct; **CPU-only platforms are 4-6x off on medium rec** (the portability gap). Unowned: ONESHOT_CPU_MAX_REGIONS env doesn't take effect in-process; pre-existing det-diff fox-ref FAIL (cos 0.25, identical pre/post). Evidence in `PERFORMANCE.md` | **DONE** |
| 2026-08-05 | *(landed via `perf/layout-phase1`)* | **O1 answered + O2a landed** — Phase 1 (~1.4 s) is **steady-state Metal compute, not warmup** (new `CRISPEMBED_LAYOUT_REPEAT=N` CLI diagnostic: warm==cold; new compute/readback bench split: readback ~5-8 ms). New opt-in `LAYOUT_CONV_F16=1` (F16-dst im2col + F16 mul_mm) **measured SLOWER on M1 Metal (2.2 s vs 1.4 s)**, quality fine (±0.002 score) — kept gated for CUDA/A1000 where it should win. O2a: per-call self-attn weight re-read/re-transpose (~5 MB/call) cached in the layer, **regions byte-identical**; saving below the loaded box's noise floor, claimed as removed work only. Next Phase-2 candidate recorded: value-proj on GPU (est. 101→~30 ms) | **DONE** |
| 2026-08-05 | *(landed via `perf/r7-scunet`)* | **R7 measured and CLOSED — no DequantCache warranted**: new permanent `to_f32` timing on the `CRISPEMBED_SCUNET_BENCH` line shows ALL weight dequant copies are **~4-5 ms of a ~4.3 s tile pass (~0.1%)** on the f32 artifact (f16 bound: ~1%). The item argued from grep-presence, not cost — third stale backlog premise caught by measure-first this session. Output byte-identical with the instrumentation; scunet's real cost is Swin/conv compute (deprioritized SR-on-GPU research) | **DONE** |
| 2026-08-05 | *(landed via `perf/r2-deform`)* | **R2 premise STALE — real hotspot found + fixed, 2.66x Phase 2** — measured FIRST (new permanent per-stage timers behind `CRISPEMBED_LAYOUT_DETECT_BENCH`): the deform loop is **16 ms of 856 ms Phase 2 (~2%)** on current main — the survey's "dominant cost" claim predates `2a43e4f4`. Actual hotspot: decoder **level input projection** (scalar strided nest, single-threaded) at **549 ms = 64% of Phase 2**; rewritten AXPY+threaded (byte-identical accumulation), level-proj ~15x at `-t 4`, **Phase 2 846→318 ms**, layout call 2332→~1700 ms, **CLI regions byte-identical** both thread counts. Deform loop deliberately untouched. New top costs recorded in the R2 backlog note: Phase 1 Metal backbone ~1.4 s (~80% of call), then value-proj + self-attn weight re-upload | **DONE** |
| 2026-08-05 | *(landed via `perf/conv2d-gemm`)* | **R6 built + M1-measured, R4 DONE** — `core_cpu::conv2d_im2col_cpu` (im2col tiles + oc-outer interchange + fork-join threads), **bitwise-identical by construction** (exact-equality unit guard, 9 shapes, nt=1+4; 180/180). Gated `CRISPEMBED_CONV2D_GEMM=1`/`CRISPEMBED_CONV2D_THREADS=N`, default OFF. M1 A/B (PP-OCRv6 medium scalar det, 5 interleaved pairs): **nt=4 wall 2.04x, won every pair; nt=1 4-7% SLOWER** (12 MB shared L2 already holds the weights — threading is the M1 win, interchange needs the small-L2 x86/Kaggle arm, TODO). R4: `CRISPEMBED_LIGHTONOCR_GPU=1`/`_FORCE_CPU=1` gate landed (got_ocr sched pattern); default verified byte-identical + 0 Metal markers; Metal arm proven live, decoded text identical, no wall win on the small fixture → CPU stays default. Evidence: `PERFORMANCE.md` "R6 conv2d_cpu im2col-tile A/B" | **DONE** |
| 2026-08-05 | *(landed on `main`, docs only — no code touched)* | **OCR runtime residency survey DONE** — code-verified sweep at `9f731fb5` of every OCR-lane engine's backend selection + `ggml_backend_sched` composition. Full tables in `PERFORMANCE.md` ("OCR runtime residency survey", top of file); ranked backlog as **R1-R8** in "OPEN TASKS — OCR runtime residency and optimization backlog" below. **Key correction: the loading backend is not the computing backend** — bttr/hmer/posformer/mixtex/flova/ppformulanet build ggml encoder graphs but run them on a CPU-only `enc_sched` (their "prefer GPU backend" comments are stale), and `lightonocr` is hardcoded CPU with no `*_FORCE_CPU` gate despite the VLM maturity table claiming "GPU: Yes". Also closed: the P3 "`--gpu-backend` ignored" gap (`crispembed.cpp:101` routes through the helper). **No engine defaults changed** | **DONE** |
| 2026-08-05 | *(landed, round-7 coordinator)* | **v0.17.6 RELEASED** (`23a5d5e0` bump + tag on green-CI tip `902a6e1b`; release run `31016913759` SUCCESS, **16/16 assets verified** — same complete set as v0.17.5): /rerank server abort fix, mxbai/ms-marco -g7c re-ships, erf pooler default, DS_/BENCH/UOCR_* `=0` gate audits (incl. the `UOCR_PD=0` segfault), reranker -f7 imatrix re-pins + new bge-v2-m3-q4k alias, `CRISPEMBED_QUANT_IMATRIX_QKV` selector, Windows `test_env_gate` MSVC fix (**Windows CI had been red since `d04f3572`** — now green). Published notes dropped from the tree (`5f756ab5`, tag retains its copy) | **DONE** |
| 2026-08-05 | *(landed `f34bf0b5`, round-7 coordinator's own work)* | **Reranker sub-Q8 re-pin DONE**: local Metal+CPU cross-check reproduced the Kaggle rerank-f7 A/B (f16 raw scores to ~3dp, f7 dscore to 4dp; tau band ±0.009-0.013 = 2-3 near-tie flips, q8_0 itself swings 0.009 across backends). jina `-q4k` alias re-pinned to `-f7` (dscore −25% both backends, tau in-band); **bge-reranker-v2-m3 got its FIRST sub-Q8 alias** `bge-reranker-v2-m3-q4k` → `-q4_k-imatrix-f7` (tau .920→.942 CPU / .947 Metal, dscore −29/−33%; beats iq4_xs-f7 on tau both backends). q8_0 stays default both families; jina iq4_xs-f7 best-tau finding recorded, no alias added. Both aliases fresh-download SHA-verified on the rebuilt binary (MTL0 proven, HF-scale scores, correct ordering). Evidence `tests/results/repin-f7/SUMMARY.md` | **DONE** |
| 2026-08-05 | *(landed `f7d34896`+`cb2489bb`; delegated + coordinator-verified: diff read line-by-line — default path structurally identical to shipped behavior; `test-imatrix-alias` 59/59 re-run; the key artifact claim re-verified independently with my own gguf read — 24 direct q/k keys bit-identical, cos vs merged L0 = 0.215)* | **mxbai q/k imatrix provenance A/B MERGED** — finding REAL: DeBERTa-v2 applies q/k a second time to rel-position embeddings (`crispembed.cpp:1166/:1195`), so the collector files direct `blk.N.attn_{q,k}` entries carrying rel-pos statistics (bit-identical across all layers, zero q-vs-k info) that shadow the correct merged-alias vector. **But NOT a quality defect**: 6-cell A/B (q4_k/iq4_xs/q3_k × both models, 192 scores/cell vs official ONNX) — `direct` pooled best (.2677 vs merged .2927), and wins BOTH models at q3_k where importance matters most. **Coordinator decision: default stays `direct`**; new opt-in selector `CRISPEMBED_QUANT_IMATRIX_QKV=direct\|merged\|sum` (default reproduces the shipped xsmall q4_k-imatrix-g7c BIT-FOR-BIT). **Premise correction (imatrix-row claim from `87e11a4e`): the Kaggle "mxbai regressing tail arm" was measured on the pre-g7c ContextPooler-less base (`quant_src` header) — on the corrected -g7c base imatrix HELPS (xsmall q4_k τ .9067→.9333); no re-pin/re-ship warranted.** Evidence `tests/results/mxbai-qk-imatrix/SUMMARY.md` | **DONE** |
| 2026-08-05 | *(landed `d45f3889`+`ee576e23`; delegated + coordinator-verified: test-env-gate re-run, base/PD=0/DBG=1/DBG=0 arms re-run myself on the agent binary — stdout byte-identical to the recorded arms and to manifest gold, MTL0 confirmed in all 44 recorded stderr files, pre-fix segfault artifact + code mechanism read directly)* | **UOCR_* gate sweep MERGED** — 40 call sites / 17 boolean vars → `core_env::on()`; 7 value-carrying vars + BENCH left; new `UOCR_DBG=1` gate-resolution line. **Headline: on pre-fix main, `UOCR_PD=0` turned the persistent-decode path ON and SEGFAULTED with empty stdout** (crash-severity `=0` inversion — strongest case yet for the remaining sweep). No-op on defaults: parent-commit default run byte-identical, CER 0.0000 vs manifest gold. Evidence `tests/results/uocr-gates/SUMMARY.md` (44 serialized model runs, 83 checks). **Follow-up, unowned: the `UOCR_PD=1` persistent-decode path segfaults at gen=2 on main (pre-existing, 7/44 runs, all PD-path)** — opt-in path, default unaffected; needs its own session. Parse-level-only verification recorded honestly for `UOCR_FA_F32`/`UOCR_OPT_PD_F32`/`UOCR_INJECT_*` (no observable marker / crash-shadowed / needs a ref dump) | **DONE** |
| 2026-08-05 | *(landed in CrispASR, round-6 coordinator's own work)* | **G8=F10 DONE** (CrispASR `fd3c0e5e`): (1) T18 `--gpu-backend cpu` short-circuit synced into their `gpu_backend_pref.h` (+ cli.cpp now propagates the cpu pref; miotts's direct init_best bypass routed through the helper; `CRISPASR_GPU_PREF_CPU_LEGACY=1` value-parsed restore). (2) **Found+fixed a pre-existing crash on their main**: the vendored whisper wrapper's #214 pref filter was desynced from `make_buft_list` (weights on a device the sched doesn't carry → `sched_backend_id_from_cur` abort) AND lacked the metal→mtl alias — BOTH `--gpu-backend cpu` and `--gpu-backend metal` crashed any `-l auto` LID run; shared `whisper_dev_matches_gpu_pref()` now filters both sites. (3) **PLAN #88 write-path DECIDED** (their HISTORY §60o addendum): keep flush-at-device-free, reject flush-per-run + per-engine scoping, adopt the T18/G4 open-time cap as their `core/metal_pipeline_cache_policy.h` (`CRISPASR_METAL_PIPELINE_CACHE_MAX_MB` default 64; capping skips the open AND no-ops the flush, so oversized archives stop growing — their normal-exiting CLIs were the growth source; CrispEmbed one-shots deliberately never write). Verified: 1441/1441 units (6 new incl. red-gate legacy arm + both cap spellings), 4 live paraformer+LID arms RC=0 byte-identical transcripts with per-arm backend stderr proof, format+check scripts green. Shared-ggml-fork patch itself unchanged | **DONE** |
| 2026-08-05 | *(landed `0a72e267`; delegated + coordinator-verified, flip decision + summary coordinator's own)* | **mxbai GELU A/B MERGED** — 3 findings. (1) erf-vs-tanh REAL but tiny: erf collapses the f16 residual vs the official-repo ONNX reference 12-96× (max Δ 2e-4 → 3e-6), tau 1.000 both arms, q8 quant error dominates 200-1000×; **default FLIPPED to erf-exact** (coordinator decision — no shipped artifact carries a pooler, so the flip perturbs nothing shipped; `CRISPEMBED_RERANK_POOLER_GELU_ERF=0` restores tanh, three spellings verified on the final binary). (2) **UNASSIGNED FIND: shipped mxbai-rerank q8_0 artifacts have NO pooler stage at all** (G7c one architecture over; coordinator re-read the tensor lists + SHA-matched the pins): calibration ±0.3 vs ±6, xsmall ranking near-INVERTED (tau −0.2/−0.733) with wrong top-1 both queries; fresh main-converter f16 matches ONNX to 1e-4 → stale artifacts, re-ship claimed below. (3) **Rider CONFIRMED+FIXED: server `POST /rerank` aborted the whole process** on any quantized 2-layer/pooler reranker (raw H×H tensor_get of Q8_0 weights — the known overrun class; live for jina-reranker-v2!); duplicate cache block deleted, apply_classifier's dequant-safe path serves both surfaces; coordinator re-ran server spot (HF-scale scores, server stays up). Evidence `tests/results/mxbai-gelu/SUMMARY.md` | **DONE** |
| 2026-08-05 | *(landed `da0272e8`, round-6 coordinator's own work)* | **mxbai artifact re-ship DONE** (G7c playbook): both models regenerated from fresh mixedbread-ai checkpoints with the UNCHANGED main converter (ollama mode, `pooler: ok (act=gelu)` both), f16 + 4 quants each (imatrix quants on the fresh `-f7` imatrices, "72 with imatrix" both), decoded-score-gated vs the committed ONNX refs BEFORE upload (f16 max Δ ≤9e-6 orderings identical; q8_0 0.04-0.10; 4-bit 0.14-0.71 with two documented near-tie swaps — q8_0 stays the pinned tier), 10 `*-g7c.gguf` uploaded (old files kept), READMEs note the defect, 2 registry aliases re-pointed + 2 new `model_hashes.h` pins, fresh-download SHA-verified spot-runs both aliases (HF-scale, correct top-1). Re-ship addendum in `tests/results/mxbai-gelu/SUMMARY.md` | **DONE** |
| 2026-08-05 | *(landed; delegated + coordinator-verified: env-gate/imatrix-alias/no-repeat-ngram re-run, own three-spelling e5 re-run — =0 silent, stdout md5-stable, MTL0 proof; conversion sites spot-read)* | **`CRISPEMBED_*_BENCH` presence-gate audit MERGED** — not "8+ engines" but **68 presence-based sites across 60 files**, all now routed through one hoisted helper `src/core/env_gate.h` / `core_env::on()` (set, non-empty, not `"0"` => on; deepseek's `ds_env_on` semantics verbatim, its internals untouched). `grep getenv src/ \| grep BENCH` is now empty: the two already-value-parsed sites (`CRISPEMBED_INIT_BENCH`, `FIREREDPUNC_BENCH`) were folded in too, and deepseek's BENCH gate — which the DS_ audit deliberately left presence-based pending this sweep — now uses the shared helper. **Every one of the 69 is diagnostic-only** (read per site: each sets a `ctx->bench` consumed solely by `if (bench) fprintf(stderr,…)`, or guards an fprintf; the 5 that look load-bearing are documented in the evidence). Evidence `tests/results/bench-gates/SUMMARY.md`: 24/24 checks, 4 gates × 3 spellings serialized with `--gpu-backend metal`, stdout byte-identical in every arm; **pre-fix control on the parent-commit binary shows `=0` printing the bench lines** and pre/post stdout identical (no-op vs main). New hermetic `test-env-gate` (10 checks, wired into the CI model-free tier — which also picked up the missing `./build/test-no-repeat-ngram` run); red-gate proven by temporarily reverting the helper to `return e != nullptr`. **Recorded, NOT touched: 267 presence-based sites over 156 distinct non-BENCH vars**, and unlike BENCH many are output-affecting — biggest cluster is `unlimited_ocr.cpp`'s ~40 `UOCR_*` gates, the exact mirror of the fixed `DS_*` set; that conversion needs a per-gate output A/B | **DONE** |
| 2026-08-05 | *(landed `87e11a4e`; delegated + coordinator-verified: driver diff reviewed, HF uploads listed, coverage lines read)* | **Reranker imatrix re-collection MERGED** (F7b leftover): 6/7 published reranker imatrices carried the `leaf_N` defect (control: bge-reranker-base was clean — F16 q/k/v skips the pre-merge, `crispembed.cpp:861`). Kernel `chr1s4/crispembed-imatrix-rerank-f7` v1: 7/7 re-collected on the CORRECT bases (`base_file` override pins ms-marco to the `-g7c` artifacts — `pick_base_gguf` preferred the superseded pre-pooler file), all new imatrices `leaf_N=0`, coverage e.g. L-6 18→36 / jina 36→72 / bge-m3 72→144 "with imatrix". **Pipeline hardened**: an `-imatrix` arm reading `0 with imatrix` now RAISES instead of silently shipping no-importance quants; per-run coverage digest + raw rerank logits in every A/B. 29 files uploaded (`*-f7.imatrix`, `*-{q4_k-imatrix,iq4_xs}-f7.gguf`, ms-marco composed as `-g7c-f7`), no published file replaced, no pin touched. A/B (Kaggle x86): clear wins bge-v2-m3 (tau .9244→.9556, dscore −26%) + jina (−23% dscore); ms-marco tau up on both; **mxbai soft (new finding, recorded not fixed: DeBERTa q/k imatrix provenance wrong — collected over rel-position inputs because quantize.cpp prefers direct name match over the merged alias; needs own A/B)**. **OPEN coordinator decisions (next round): re-pin jina `-q4_k-imatrix` (SHA-pinned, `model_hashes.h:251`) and bge-v2-m3 sub-Q8 aliases to the `-f7` artifacts — local-Metal cross-check first per G3 precedent** | **DONE** |
| 2026-08-05 | *(landed, round-5 coordinator's own work)* | **G7c MERGED** (`63997e2c`, expanded far beyond the archived brief): shipped ms-marco rerankers were converted WITHOUT the BertPooler stage — native scored `dot(CLS,w)+b` where HF scores `classifier(tanh(pooler(CLS)))`; calibration destroyed (±0.2 vs ±11), tail ranking reordered (τ 0.733). Converter-only fix: fold BERT pooler+classifier into the verified 2-layer tanh head; suppress the stray pooler emit for Roberta-head rerankers; truthful `bert.pooler_act`. f16 matches the ONNX reference to ≤0.0009 (local miniconda torch DISQUALIFIED as parity reference — NaN/bus-error/garbage on BERT forwards); 10 `*-g7c.gguf` artifacts uploaded (old files kept so released binaries' pins keep working), 4 `model_hashes.h` pins + registry aliases re-pointed, fresh-download SHA-verified spot-runs on both pinned alias types. Evidence `tests/results/g7c/SUMMARY.md`. Recorded, not fixed: mxbai gelu is tanh-approx vs HF erf (own A/B needed); reranker imatrix files still pre-F7. **G7b DECIDED (closes it): no ST-pooler-tanh parity path** — pre-pooler CLS stays the embedding output (feature-extraction convention; the fixed LaBSE converter already matches HF CLS at cos 1.000000); no demand signal two rounds running (G7a precedent); the converter now records `bert.pooler_act=tanh` truthfully, so a future opt-in ST-parity path has the metadata it needs | **DONE** |
| 2026-08-05 | *(landed, round-5 coordinator's own work)* | **DS_ value-parse audit MERGED** (`91ebb55d`, post-v0.17.5-tag): all presence-based boolean gates in `src/deepseek_ocr2.cpp` now value-parsed via one `ds_env_on()` helper — the 9 from the G2b row PLUS two same-class finds (`DS2_FORCE_CPU`, `=0` used to force CPU; `DS_PROFILE`). New `DS_DBG=1` gate-resolution stderr line (markerless gates now carry parse proof). Evidence `tests/results/ds-gates/`: 26 serialized runs, 42/42 checks — every `=0` arm byte-identical to the absent baseline; every `=1` arm proven engaged (decode-path blockers, `mtl0=0` under FORCE_CPU, `kv=f16, flash` line, 300 s/576 s CPU-path wall signatures); default receipt run byte-identical to the recorded g2b arm (no-op vs main); hermetic mmap + no-repeat-ngram re-run on the final formatted binary. `CRISPEMBED_DEEPSEEK_OCR2_BENCH` left presence-based deliberately (codebase-wide `*_BENCH` convention, 8+ engines — separate audit if wanted) | **DONE** |
| 2026-08-05 | *(landed, round-4 coordinator's own work)* | **G2b MERGED** (`8c210291`): `DS2_CROP_MODE` now defaults ON (`=0` restores single-view). Diagnosis from the recorded g2 arms: BOTH regressions are formatting-only trajectory drift — receipt_historical is one bold-vs-plain near-tie at char 82 then markdown-list self-conditioning (alnum-content CER flat 0.125→0.129 Metal, IMPROVES 0.152→0.110 CPU; Metal-crop reads OPR correctly where CPU reads QPR, GT=OPR); synth_01_noise is 4 inserted colons, content byte-equal. Gates (`tests/results/g2b/`): gold gate cer=0.000 BOTH manifest entries with crops engaging on fox, run on the final binary; byte-identity 12/12 default-ON == g2 crop arms (Metal+CPU cc0) + `=0` == g2 base arms + synth spot 2/2 (all modulo the g2 runner's trailing-newline strip). Also value-parsed `DS2_LEGACY_DECODE` (was presence-based, `=0` forced legacy; verified all three spellings). **Follow-up (small, unowned): value-parse audit of the remaining presence-based `DS_*` gates** (`DS_MMAP`, `DS_MOE_CPU`, `DS_DBG`, `DS_SAM_CONV_CPU`, `DS_QWEN2_ENC_FLASH`, `DS_QWEN2_SCALAR`, `DS_LLM_FLASH`, `DS_NO_KV`, `DS_LMHEAD_CPU`) — one A/B each, not batched | **DONE** |
| 2026-08-05 | *(landed, round-4 coordinator's own work)* | **G1/F4 MERGED** (`703161b1`): SmolDocling vision split residency — `vis.*` weights on the GPU backend via new `core_gguf::load_weights_split` (ported from CrispASR #69a), SigLIP graphs on Metal, connector+decode+LM-head+KV stay CPU. GPU is the DEFAULT (matches every other VLM lane; deepseek precedent); `SMOLDOCLING_FORCE_CPU=1` (value-parsed, both spellings verified) + `--gpu-backend cpu` restore all-CPU. Gates in `tests/results/g1/SUMMARY.md`: CPU arm byte-identical to T15 recorded outputs 5/5; Metal matches/beats the REFERENCE on every GT page; one documented divergence (receipt_historical 0.238→0.372 stripped CER vs GT, reference 0.493 — Metal F16 rounding, T14/G2 class); interleaved timing fox vision 3562→1241 ms (2.9×), scan 14500→3163 ms (4.6×), totals 2.1-2.25×; MTL0 per-run. NOTE the old "31.7 s fox vision" was a different CPU-only build — same-binary CPU is 3.6 s. G8 NOT claimed (CrispASR active today; recon: their gpu_backend_pref.h still lacks the T18 cpu short-circuit, their box disk-full — deferred) | **DONE** |
| 2026-08-05 | *(landed, coordinator's own work)* | **G2/F5 MERGED** (`d5788a88` port + `e81c827e` acceptance): DeepSeek-OCR2 dynamic-crop, opt-in `DS2_CROP_MODE=1`, blueprint line-by-line at pinned `aaa02f38`. Full gates in `tests/results/g2/SUMMARY.md`: crop-off byte-identical to f1 baselines 10/10 both backends; cc0 raw CER mean Metal 0.657→**0.236**, CPU 0.279→**0.185 (beats the A4 reference 0.187)**; commons_test 0.0074 both backends; **F1's Metal german 1024-cap FIXED** (366 tok, CER 2.14→0.195); gold gate cer=0.000 both manifest entries. Stays opt-in per A/B rule: Metal `receipt_historical` regresses 0.138→0.305 with crops (CPU improves — Metal-specific) + synth_01_noise 0.015→0.045. **Follow-up G2b:** diagnose that Metal crop regression, then decide the default flip (reference contract runs crop_mode=True). `~/.cache/hf-regression` restored by the gold-gate re-download | **DONE** |
| 2026-08-05 | *(landed round-3 wave 2)* | **G6 MERGED** (`510f35d0` results + `73beea9f` gate fix, coordinator-verified: byte-identity claims re-checked with my own cmp both directions, diff scope confirmed results-only, gate fix spot-run in both spellings). Verdict: `DS2_KV_F16` **stays opt-in** — memory feature (KV 165→82.5 MB, −84 MB footprint), timing parity (~1% within noise, 4 interleaved pairs), NOT quality-neutral on CPU (all 5 cc0 pages perturb, aggregate +0.008 CER; Metal 24/25 identical, its one "win" is a cap-adjacent greedy lottery). Agent found the gate was presence-based (`DS2_KV_F16=0` ENABLED f16) — now value-parsed. Full tables `tests/results/g6/SUMMARY.md`. **G5 DONE + MERGED** (`5fcd7006`, coordinator's own A/B + decision): embed one-shot defaults to `min(4, cores)` threads (−t1 lost 2-3× on every model tested: e5 0.84→0.40 s, arctic 3.05→1.01, f2llm-330m 2.95→1.01; embeddings md5-identical across thread counts); Metal stays default backend (model-dependent vs CPU-t4), `CRISPEMBED_ONESHOT_CPU` stays OFF | **DONE** |
| 2026-08-05 | *(landed round-3 wave 1)* | **G4 MERGED** (`c1ccb1f4`, coordinator-verified: diff inspected — single-site hoist in `crispasr_init_gpu_backend()` covers all ~40 lanes with the EMBED path's exact guard, + two direct-`init_best` bypass sites (nafnet enc, safmn opt-in); my own pix2tex spot-run: 3 arms byte-identical, cap diagnostic fires on new binary, `MAX_MB=0` bypass restores archive load, MTL0 in every stderr; `test-backend-smoke` re-run green). Agent's per-lane table in its branch commit message; clean verdict pix2tex init 985→313 ms (~1 ms/MB of the 652 MiB archive). Brief corrections found: SmolDocling is CPU-only (no Metal lane), `CRISPEMBED_INIT_BENCH` exists only in the EMBED path. **683 MB `~/Library/Caches/ggml-metal/Apple_M1.archive` DELETED** (the scheduled coordinator step). **G3 DONE + MERGED** (`464f812f`): local Metal+CPU cross-check reproduced the Kaggle F7b numbers (CPU to 4dp; Metal delta ≤0.002, same ordering) → BOTH arctic sub-Q8 aliases re-pinned to `-f7` (q4_k+imx .9614→.9937 mean, iq4_xs .9757→.9867), fresh-download SHA-verified spot-run OK, `test-imatrix-alias` 59/59. q8_0 stays default. **Granite-r2 decision: NO new sub-Q8 aliases** (311m gain .99745→.99823 negligible, 97m artifacts stay HF-only; ab numbers in `cstr/granite-embedding-*-r2-GGUF/*-imatrix-ab.txt`) | **DONE** |
| 2026-08-05 | *(landed round-3 wave 1)* | **G7d MERGED** (`10d160ba`, coordinator-verified: diff inspected, `require=` kwarg confirmed in the harness the drivers bootstrap, py_compile + vendored-harness guard re-run myself — 106 checks, 0 failures, 15 copies): the three upload-bearing drivers (`unlimited-ocr-convert`, `crispembed-splade-fix`, `deepseek-ocr2-convert`) now call `resolve_hf_token(require=True)` before any compute | **DONE** |
| 2026-08-04/05 | *(engine-portfolio round — ALL LANDED)* | T13 olmocr, T14 deepseek decode (`82ce1024`), T15 smoldocling (`7de85cb7`), T18 one-shot init (`c178308f`), granite-r2 (`110dd082`), tokenize_simple audit (`357dee53`), imatrix quants (`38c708b2`+`926df0ae` — 330m follow-up MERGED), metallib CMake pin (`9288d3b5`). Detail lives in the dated status blocks below and in `tests/results/*_2026-08-04.json`; do not re-derive | **DONE** |
| 2026-08-05 | *(landed, coordinator's own work)* | **F1 MERGED** (`e9f84f16`): deepseek-ocr2 no-repeat-ngram guard, default 20 per the contract, `DS2_NO_REPEAT_NGRAM=0` restores; helper hoisted to `src/core/no_repeat_ngram.h` shared by qwen2vl/internvl2/deepseek + hermetic `test-no-repeat-ngram` (13 checks, model-free CI). Full acceptance evidence in the commit message and `tests/results/f1/`; F1 status block below | **DONE** |
| 2026-08-05 | *(landed same day, wave 2)* | **F8 MERGED** (`f31c6531`, coordinator-verified: hermetic 24+15 checks re-run 0 failures, LaBSE battery 20/20 vs an independently REGENERATED HF golden, 0/20 on the unfixed binary, 4 shipped models token-id-IDENTICAL old-vs-new under my own runs). Verdict: nothing LaBSE-class was shipped; the CONVERSION PATH was broken 0/20 (converter >100k heuristic + runtime routing + per-byte pre-tokenizer) — all three layers fixed, absent-key = historical behavior. See §F8 outcome below. **F9b MERGED** (`3ade993a`, coordinator-verified: 15/15 copies hash-identical to CrispASR canonical `342c5f7f`, 13-test gate re-run per copy = 15×13/13, 8 upload-bearing drivers flipped to `resolve_hf_token(require=True)`) | **DONE** |
| 2026-08-05 | *(landed same day, wave 3)* | **F7b DONE** (kernel `chr1s4/crispembed-imatrix-t19` v3 complete; driver config merged `045102a0`; coordinator verified the `-f7` uploads exist on HF and the 4 pinned artifact SHAs still match `model_hashes.h` exactly). Headline: with real q/k/v importance, arctic q4_k+imatrix goes **.9480/.9614 → .9910/.9937 min/mean** (plain q4_k reproduced to 4dp; f2llm-80m control unchanged — comparability proven), and **q4_k+imatrix now BEATS iq4_xs+imatrix on the BERT side** (inverts T19-E3's IQ4_XS headline there; decoder side keeps the old ordering). Granite-r2 pair got first-time imatrix artifacts under canonical names. Decision items in §F7b below. **Test guards merged** (`fcc60afd`): `test-imatrix-alias` (59 checks, fails 44 on name drift; both naming sites now share `src/core/imatrix_alias.h`) + `tests/test_vendored_kaggle_harness.py` (106 checks × 15 copies, fails 5 on the pre-F9b copy), both in model-free CI | **DONE** |
| 2026-08-05 | *(landed same day)* | **F7 MERGED** (`68033e8d`, coordinator-verified: coverage 36→72-with-imatrix re-run independently, fresh collector imatrix has 12 per-layer `qkv_merged` entries and 0 `leaf_N`, hermetic battery re-run green, q4_k+imatrix now separates from plain q4_k — e5-small cos_min 0.9847→0.9889). Kaggle t19 re-collection/re-quant of every published BERT-family imatrix artifact is the follow-up (F7b below). **F9 MERGED in CrispASR** (`342c5f7f`, 13 hermetic tests re-run green). ⚠ F9 correction: canonical CrispASR harness already globbed both mount depths; the resolver that lost the t19 uploads is **CrispEmbed's stale VENDORED copy** — see F9b below | **DONE** |

## August 5, 2026 — v0.17.3 shipped with no Linux CUDA archive

The ggml 0.10.2 → 0.17.0 pin move brought one consequence nobody checked for:
ggml 0.17 calls the CUDA **driver** API. `libggml-cuda` therefore has to link
`libcuda`, which 0.10.2 never needed. The `linux-x86_64-cuda` leg died at the
link step:

```
libggml-cuda.so.0.17.0: undefined reference to `cuGetErrorString'
```

A build machine has no NVIDIA driver — that is what the toolkit's
`lib64/stubs/libcuda.so` is for — but CMake only finds it when the stubs
directory is on the library path. On this runner the stub lives at
`/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs/libcuda.so`, which is not
a path CMake searches by default.

Cost: **v0.17.3 published with neither Linux CUDA archive.** Only the Windows
CUDA zip made it. Windows was unaffected because `cuda.lib` sits in the
toolkit's `lib/x64`, found without help.

Fixed by locating the stub and passing `-DCMAKE_LIBRARY_PATH`. Two details
worth keeping:

- The lookup runs **before** the compile and fails the job immediately when the
  stub is absent. Previously the failure arrived an hour in, at link time,
  which is precisely why it landed as a silently-missing artifact rather than
  as a red build somebody reacted to. *Check a build prerequisite at the point
  you can still cheaply say what is wrong.*
- Linking the stub gives `libggml-cuda` a `DT_NEEDED` on `libcuda.so.1` — the
  same driver contract both CUDA archives already declared, and already on
  `check-bundled-deps.py`'s allow list. The user-facing requirement did not
  change.

Also a process note: a published tag cannot be repaired by fixing `main`, since
the release workflow runs from the tagged commit. v0.17.3 keeps its incomplete
asset set and v0.17.4 supersedes it. A `workflow_dispatch` dry run before
tagging would have caught this — the mechanism existed by then and was not
used, because the tag was cut by a different session in parallel.

---

## August 4, 2026 — Linux glibc floor: 2.38 → 2.27 (#42, second issue)

The other half of #42, and the one that had been sitting in the tree as a
known limitation rather than a bug. Built on the `ubuntu-latest` runner, the
Linux archives required `GLIBC_2.38` / `GLIBCXX_3.4.32`, so they refused to
start on Ubuntu 22.04, Debian 12 and anything else on glibc ≤ 2.37 — an
independent second reason a user saw a startup failure, unfixed by v0.17.1's
OpenBLAS work.

Both Linux release legs now build inside `quay.io/pypa/manylinux_2_28`
(AlmaLinux 8). Measured on the resulting artifact:

| | before | after |
|---|---|---|
| glibc | 2.38 | **2.27** |
| GLIBCXX | 3.4.32 | **3.4.22** |
| runs on | Ubuntu 24.04+, Debian 13+, EL 9+ | **Ubuntu 18.04+, Debian 10+, EL 8+** |

Container toolchain: gcc/g++ 14.2.1, cmake 4.4.2, patchelf 0.17.2, glibc 2.28.
C++17 is unaffected — an old *glibc* does not mean an old compiler, which is
the whole reason manylinux images pair a modern gcc-toolset with an ancient
libc.

Mechanics worth keeping:

- `container:` is a **matrix key**, empty on the macOS/Windows entries, which
  GitHub treats as "no container". One job still covers all four platforms.
- `git config --global --add safe.directory '*'` **before** `actions/checkout`:
  in a container the workspace uid differs from git's, and the submodule
  checkout otherwise dies on "dubious ownership".
- The apt/sudo deps step is skipped in-container — AlmaLinux has neither, and
  the image already ships patchelf. A toolchain step prints
  gcc/cmake/patchelf/glibc versions so a future base-image change is visible
  in the log rather than inferred from a mystery failure.
- `python` → `python3` for the check scripts: the manylinux image has no bare
  `python`.
- `check-bundled-deps.py --max-glibc 2.28` runs on those legs, so the floor is
  **enforced rather than assumed**. Confirmed the gate bites by running it
  against the published v0.17.2 tarball: exit 1, *"requires GLIBC_2.38 … but
  the declared floor is 2.28"*.

Still outside the container: the CUDA legs (they need the CUDA toolkit
installed on the host), so those archives keep the higher floor. Recorded in
the README rather than left to be rediscovered.

### The wheels, and the RUNPATH bug hiding behind the floor

`python/pyproject.toml` recorded this as blocked on "reshuffling pyproject.toml
to repo root + CMake in `CIBW_BEFORE_ALL_LINUX`". It was not: cibuildwheel
needs docker itself, so it cannot run inside a job-level container — but
nothing stops calling `docker run` directly with the same image it packages
into. The `.so` is now compiled in `manylinux_2_28_{x86_64,aarch64}` and staged
as before.

Removing the cause let `repair-wheel-command` go back to a real
`auditwheel repair` instead of the `cp {wheel} {dest_dir}/` workaround. Result:
wheels now carry
`manylinux_2_27_aarch64.manylinux_2_28_aarch64` tags — PyPI-acceptable, where
the previous bare `linux_aarch64` is rejected outright.

Two things surfaced only by doing it, each hidden behind the previous one:

1. **My own check staged its input wrong.** The first verification step built a
   `_wheelcheck/` directory with `find -type f`, which drops the SONAME
   symlinks, so `libggml-base.so.0` read as unbundled and the step failed on
   its own staging while the artifact was fine. Fixed by checking
   `python/crispembed` — the directory cibuildwheel actually packages — which
   is both simpler and a stronger assertion. *Verify the real payload, not a
   copy you assembled to verify with.*
2. **The wheels never patched RUNPATH.** `auditwheel repair` then failed with
   *"Cannot repair wheel, because required library libggml-base.so.0 could not
   be located"* — the staged libs still carried the build-tree RUNPATH
   (absolute paths inside the build container), so nothing could resolve a
   sibling. The release packaging has patched `$ORIGIN` since forever; the
   wheels only ever copied. It went unnoticed because the repair command was a
   `cp` no-op and the Python binding happens to import the libs in dependency
   order at load time — so the wheels worked by accident rather than by
   construction.

Confirmed end to end: RUNPATH `$ORIGIN`, floor 2.27, and cibuildwheel's own
`test-command` (which imports the package and calls `_find_lib()`) passing on
all four platforms, so auditwheel's relocation demonstrably did not break lib
discovery.

---

## August 4, 2026 — Linux CUDA archive: a second, self-inflicted exit 127 (#42)

Filed upstream by `niksedk` after SubtitleEdit#13205, and correct on every
point — including one this repo had written down as a deliberate choice and
never re-examined.

`crispembed-linux-x86_64-cuda.tar.gz` bundles no CUDA runtime. The workflow
comment stated the contract as *"End users must have a matching CUDA driver
(12.x) on the host"*, which is **wrong**: a driver provides `libcuda.so.1`,
while `libcudart.so.12` / `libcublas.so.12` come from the CUDA *toolkit*.
Verified against the published v0.17.1 archive:

```
libggml.so.0      -> libggml-cuda.so.0  (bundled, hard DT_NEEDED)
libggml-cuda.so.0 -> libcudart.so.12    <-- toolkit, not bundled
                  -> libcublas.so.12    <-- toolkit, not bundled
                  -> libcuda.so.1       <-- driver, correctly external
```

Because `libggml.so` hard-links `libggml-cuda.so`, a driver-only machine dies
in the loader at startup — exit 127, no output, and **no degrading to the CPU
backend**, since the process never starts. Identical failure mode to the
OpenBLAS one. The Windows CUDA zip has bundled `cudart64_12.dll` +
`cublas64_12.dll` all along, which is the only reason this never bit Windows;
the Linux/Windows asymmetry was the bug.

Fixed by shipping **both**, rather than changing what an existing pin resolves
to (Subtitle Edit and others pin these exact filenames):

- `crispembed-linux-x86_64-cuda.tar.gz` — unchanged name, unchanged contents,
  now honestly documented as requiring the toolkit runtime.
- `crispembed-linux-x86_64-cuda-bundled.tar.gz` — same build plus the CUDA
  runtime closure, needing only the driver.

Implementation notes worth keeping:

- The closure is taken from `ldd` output rather than a hardcoded list, so
  `libcublasLt` and `libnvJitLink` (which `libcublasLt` pulls in on 12.x) come
  along on their own and the set cannot rot on a CUDA bump.
- `libcuda.so.1` is explicitly excluded. Bundling a driver library would pin
  users to whatever driver the runner had.
- **`DT_RUNPATH` is not transitive**: `libcublas.so.12` resolving
  `libcublasLt.so.12` does not inherit `libggml-cuda.so`'s `RUNPATH`, so every
  bundled CUDA lib needs its own `$ORIGIN` (what manylinux/PyTorch wheels do
  to their vendored CUDA libs).
- NVIDIA's EULA permits redistributing these runtime components; the bundled
  archive carries `NVIDIA-NOTICE.txt` saying what is NVIDIA's, under which
  terms, and that the driver library is not included.
  **Correction (found by inspecting the published v0.17.2 archive):** it was
  meant to carry `NVIDIA-EULA.txt` as well, and does not. The
  `cp "$CUDA_PATH/EULA.txt" … || true` silently did nothing, because the
  sub-package install (nvcc/cudart/cublas) lays down no `EULA.txt` at the
  toolkit root. The copy now searches the plausible locations and *logs*
  which way it went instead of swallowing the failure. `NVIDIA-NOTICE.txt`
  carries the terms by reference either way, so v0.17.2 is not
  mis-licensed — but the documentation overclaimed, which is its own defect.
- `scripts/check-bundled-deps.py` gained `--allow PATTERN`, and each archive is
  now verified against its **own** contract: slim may want the toolkit runtime,
  bundled may want only the driver. The assumption that "CUDA users normally
  have CUDA installed" is now a checked fact instead of a comment.

Also confirmed from #42 and already fixed in v0.17.1: the CI-drift observation
(`build.yml` said `-DGGML_BLAS=OFF` while the release tarball shipped
`libggml-blas.so`) — `release.yml` had diverged from `build.yml`.

---

## August 4, 2026 — crispembed-sys required sources even to link a prebuilt

`build.rs` called `resolve_src_root()` unconditionally as the first statement
of `main()`, before it looked for a prebuilt library. That function panics when
the C/C++ sources are absent, so both escape hatches — `CRISPEMBED_SYS_LIB_DIR`
and the `build/` / `build-cuda/` / `build-vulkan/` probe — were unreachable
without a full source tree including the ggml submodule. A consumer holding a
perfectly good `libcrispembed.so` still needed ~1 GB of sources checked out,
which defeats the point of shipping prebuilt libraries and contradicts the
README's "no cmake, no source build".

A regression, introduced by `a3156a2a` (crates.io publishability): v0.16.1 used
`manifest_dir.parent().expect(...)`, which never fires in practice.

- `try_prebuilt` now takes `manifest_dir` instead of a resolved source root and
  probes `build*/` relative to the crate's parent and `vendor/` directly. No
  source validation — `build/libcrispembed.so` links the same whether or not
  ggml was ever initialised.
- `resolve_src_root()` is called lazily, only on the path that actually runs
  cmake. Its doc comment now says so.
- A `CRISPEMBED_SYS_LIB_DIR` that is set but holds no library emits a
  `cargo:warning` naming what was looked for, instead of silently falling
  through to a source build the user thought they had opted out of.
- `rust-crates.yml` gains a regression test: compile `build.rs` standalone with
  a source-less `CARGO_MANIFEST_DIR` and assert it links the prebuilt. Verified
  both ways — it passes on the fix and exits 101 on the pre-fix file. Nothing
  else in CI could have caught this, because every other job runs in-tree where
  the sources trivially exist.

Behaviour checked across four cases: prebuilt via env var with no sources
(was: panic, now links it); env var pointing somewhere empty (warns, then falls
through); sibling `build/` with no sources (was: panic, now links it); and a
real in-tree checkout with no env var (unchanged — still resolves
`build-cuda`).

---

## August 4, 2026 — every Linux tarball was unlaunchable (SubtitleEdit#13205)

**Reported** (via Subtitle Edit, which downloads our prebuilt binaries):
`crispembed-server` exits with code 127 and prints nothing, on EndeavourOS and
on Linux Lite. PaddleOCR standalone works on the same machine. The
`Cannot load libcuda.so.1` line in the terminal is unrelated — that is the
host app probing hardware decoders at startup.

**Cause.** Exit 127 from a dynamically-linked binary is the loader failing to
resolve a library; the process dies before `main()`, so nothing of ours can
print. Verified by parsing the dynamic section of the shipped v0.17.0
artifacts:

```
crispembed-server -> libggml.so.0      (bundled, RUNPATH $ORIGIN)
libggml.so.0      -> libggml-blas.so.0 (bundled)
libggml-blas.so.0 -> libopenblas.so.0  <-- not in the archive
```

`-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS` on the Linux release legs made
OpenBLAS a hard link-time dependency that the tarball never carried. Both
`crispembed-linux-x86_64.tar.gz` and `crispembed-linux-arm64.tar.gz` were
affected, in every release — anyone without OpenBLAS already installed could
never start it. The workflow apt-installed `libopenblas-dev` so the CMake BLAS
probe would succeed, so the runner always had the library and the artifact
never did: the failure could not be reached from CI by construction. Same
shape as #41.

It was also buying nothing: this repo's own `PERFORMANCE.md` "BLAS
Acceleration" table measures OpenBLAS at 0.9–1.0x on these models, and
`LEARNINGS.md` already said BLAS is minimal-benefit here because quantized
kernels use ggml's SIMD paths. The workflow comment justifying it ("big
matmuls of large encoders") was never measured.

**Fixed:**

- Linux release legs (x86_64 and arm64) now build `-DGGML_BLAS=OFF`.
  LLAMAFILE's tinyBLAS stays on for x86_64 and needs no external library.
  macOS keeps `GGML_BLAS_VENDOR=Apple` — Accelerate is a system framework.
- `libopenblas-dev` removed from the Linux build deps. A re-enabled
  `GGML_BLAS` now fails loudly at configure time on CI instead of quietly
  producing a tarball that exits 127 on every user's machine.
- New `scripts/check-bundled-deps.py`, run on the staged `pkg/` of every Linux
  leg. It parses each ELF's dynamic section and fails if a `DT_NEEDED` entry is
  neither bundled nor part of a base glibc system, and reports the glibc floor.
  `libgomp.so.1` is deliberately not treated as base-system.
- `release.yml` gains `workflow_dispatch`, with all seven publish steps guarded
  on `startsWith(github.ref, 'refs/tags/')`. Previously the only way to see a
  packaged artifact was to publish a release, which is why nobody looked.

Verified by running the new guard against the actual published v0.17.0
tarballs: it flags `libggml-blas.so.0.10.2 needs libopenblas.so.0` on both
architectures and passes everything else.

**Still open:** the tarballs are built on Ubuntu 24.04 and require glibc ≥ 2.38
/ GLIBCXX ≥ 3.4.32, so they will not start on Ubuntu 22.04 or Debian 12 even
with OpenBLAS installed. That is a second, independent reason a user can see a
startup failure. Already recorded for the wheels in `python/pyproject.toml`;
the fix for both is a manylinux-container build.

**Retracted observation:** an earlier draft of this entry claimed v0.17.0 had
published no `crispembed-linux-x86_64-cuda.tar.gz` or
`crispembed-windows-x86_64-cuda.zip`. That was wrong. The asset list was read
shortly after the release was created, and the CUDA legs take ~1 h, so they
simply had not attached yet. Both assets are present in v0.17.0, and a
`workflow_dispatch` dry run of the whole release later came back green on all
eleven legs including both CUDA ones. Nothing to fix — read a release's assets
only once every leg has finished.

---

## August 4, 2026 — a HuggingFace rate limit was being reported as pin drift

`main-health`'s "Verify model SHA-256 pins are current" step went red with
`model_hashes.h is stale`. It was not stale. Every one of the ~240 HF tree-API
probes came back `HTTP 429`, so every pin resolved to nothing, the regenerated
header naturally differed from the committed one, and the job announced drift
that did not exist. The licence step next to it already degrades transient
failures to "?" rather than "✗"; the pin step had no such distinction.

`tools/fetch_model_hashes.py` now separates "upstream did not answer" from
"upstream answered differently":

- New `TransientFetchError` for 408/425/429/5xx and socket timeouts, with four
  attempts and exponential backoff that honours `Retry-After` (capped at 30 s).
  A 404 still propagates — a renamed or deleted repo IS drift and must fail.
- `resolve()` returns the unreachable repo keys. `--check` reports
  **INCONCLUSIVE** (exit 0) when the header differs *and* something was
  unreachable, and still fails when everything was reachable and the header
  differs.
- Generating the header now REFUSES to write when any repo was unreachable
  (`--force` overrides). Previously a run during a rate limit would have
  silently rewritten 117 good pins as unpinned — the check that exists to stop
  a swapped re-host, disarmed by a transient network condition.
- Optional `HF_TOKEN` (wired in `main-health.yml`, empty when no secret is
  configured) raises the anonymous per-IP ceiling.

Verified against mocked responses: persistent 429 → `TransientFetchError`;
404 → propagates; 503-then-200 → recovers on the third attempt; header-differs
+ unreachable → exit 0, header-differs + all-reachable → exit 1.

---

## August 4, 2026 — Windows CI unbroken (`build.yml` windows-x86_64)

`build.yml`'s windows leg had been red on every push for the day. Three
independent Windows-portability breaks, all from work that only ever built on
Linux/macOS:

- **`<windows.h>`'s `min`/`max` macros.** Any TU that pulls windows.h in
  transitively turns `std::min(a, b)` into `std::((a) < (b) ? ...)` — MSVC
  C2589 — and then cascades into unrelated-looking errors many lines away
  (`'ocr_crop::extract_quad': function does not take 7 arguments`,
  `'b': references must be initialized` on a range-for, `'hypot': no
  overloaded function takes 1 arguments`). All of those were one macro in
  `ocr_orchestrator.cpp`'s crop loop. Fixed globally with `-DNOMINMAX`
  alongside the existing `_USE_MATH_DEFINES` in the MSVC block; nothing in the
  tree relies on the macros (checked — every bare `min(`/`max(` is in a
  comment).
- **`setenv` in `src/ocr_orchestrator.cpp`.** POSIX-only, added by the
  PP-OCRv6 one-shot CPU-routing work. Now `set_env_if_unset()`: `_putenv_s`
  on Windows behind a presence check, which is what `overwrite=0` meant.
- **`setenv`/`unsetenv`/`mkstemp` in `test_image_provenance` and
  `test_provenance_marking`** — the two model-free tests `build.yml` runs on
  every push. Env access now uses the `#ifdef _WIN32 _putenv_s` helper the
  other tests already carry; the temp file uses `core_tmp::make_private()`,
  the project's one portable "created by us, unpredictable name" helper.
  Caveat recorded in the test: Windows cannot hold an empty-valued env var, so
  the "empty is off" check degenerates to the unset case there.

Fixing those let the build get far enough to expose two more, plus a static
guard that had been failing alongside it the whole time:

- **`<windows.h>` drags in the legacy `<winsock.h>`**, which then collides with
  `<winsock2.h>` — 102 `'sockaddr': 'struct' type redefinition` errors inside
  the Windows SDK, all from `crispembed-server`. It bites on include *order*
  and so was invisible until someone added a header: `server.cpp` includes
  `core/temp_file.h` (windows.h) at line 38 and `httplib.h` (winsock2.h) at
  line 47. Fixed with `-DWIN32_LEAN_AND_MEAN` globally, repeated defensively in
  `core/temp_file.h` for out-of-tree consumers. Nothing in the tree uses an API
  that flag excludes (checked).
- **A second POSIX `setenv`** in `examples/cli/main.cpp`, from the same
  one-shot routing work. `--cache-dir` two lines away already had the
  `#ifdef _WIN32 _putenv_s` guard, which is what the new call should have
  copied.
- **`tools/check_test_clean_exit.sh`** was red on five tests
  (`test_image_provenance`, `test_msac_tiling`, `test_provenance_marking`,
  `test_render_provenance`, `test_temp_file`). Each now follows the documented
  pattern: body renamed to `crispembed_test_main()`, thin
  `main() { core_util::clean_exit(crispembed_test_main()); }`. None touches a
  GPU today, but they link `crispembed-core`, so ggml's static device teardown
  is one dependency away.

---

## August 4, 2026 — release artifacts pinned to a fixed CPU baseline (#41)

**Reported:** v0.16.1's `crispembed-windows-x86_64.zip` (cpu) died with
`Illegal instruction` on an i9-14900KF (Raptor Lake: AVX2, no AVX-512)
immediately after tokenizer load. The previous release ran the same model on
the same machine, and the cuda/vulkan zips' CPU fallback worked.

**Cause:** the cpu leg left `GGML_NATIVE` at its default (ON). On MSVC that
pulls in `ggml-cpu/cmake/FindSIMD.cmake`, which *runs* an AVX-512 probe binary
on the build machine (`check_c_source_runs`) and compiles the entire CPU
backend `/arch:AVX512` when it succeeds. GitHub's `windows-latest` pool mixes
AVX-512-capable Intel hosts with AVX2-only AMD hosts, so the ISA of a release
was decided by which runner happened to pick up the job. The cuda leg was the
only one already pinning `-DGGML_NATIVE=OFF` — which is exactly why the
reporter found its CPU fallback healthy.

The same hazard applied to every other release leg: `-march=native` on
linux-x86_64, and `-mcpu=native` + `dotprod`/`i8mm`/`sve`/`sme` run-probes on
both arm64 legs. Python wheels had it on all four platforms.

**Fixed:**

- Every redistributable leg in `release.yml` and `python-wheels.yml` now passes
  `-DGGML_NATIVE=OFF`. x86_64 lands on SSE4.2+AVX+AVX2+FMA+F16C+BMI2;
  linux-arm64 pins `-DGGML_CPU_ARM_ARCH=armv8.2-a+fp16+dotprod` (what
  `-mcpu=native` was already resolving to on the Neoverse-N1 runners, so
  determinism rather than a floor change); macos-arm64 takes Apple clang's
  arm64 default.
- `CRISPEMBED_NATIVE` now defaults to `GGML_NATIVE` (and to OFF when
  cross-compiling), so one flag makes the whole tree portable. Previously
  `-DGGML_NATIVE=OFF` alone still left CrispEmbed's own translation units on
  `-march=native`. With native off, those targets now mirror ggml's configured
  baseline (`-mavx2 -mfma -mf16c`) instead of dropping to scalar, so
  `cpu_ops.h`'s intrinsics stay compiled in.
- New `scripts/check-cpu-baseline.py`, run after configure on every release and
  wheel leg. It checks the cache options *and* scans the generated compile
  lines (`build.ninja` / `*.vcxproj` / `flags.make`) for banned tokens —
  necessary because `FindSIMD.cmake` sets `GGML_AVX512` as a normal variable
  that shadows the cache entry, so with NATIVE on the cache can read `OFF`
  while the compile line says `/arch:AVX512`.

Verified locally on MSVC 19.44 and clang: `-DGGML_NATIVE=OFF` yields
`/arch:AVX2` (MSVC) / `-msse4.2 -mf16c -mfma -mbmi2 -mavx -mavx2` (clang) with
`-mavx2 -mfma -mf16c` on the crispembed targets; the default local build still
gets `-march=native`; and a deliberate `-DGGML_AVX512=ON` configure —
reproducing the shipped artifact — is caught by both halves of the checker.
Shipped baselines are now documented in the README. Details in `LEARNINGS.md`
§"GGML_NATIVE probes the BUILD machine".

---

## August 3, 2026 — PLAN.md active-work board cleared; OCR perf round (H1-H9)

52 completed rows archived from the PLAN.md in-flight table, plus the OCR
performance round summarised below. Only genuinely in-flight rows remain in
PLAN.md.

### OCR performance round — what shipped, and what was retracted

**Shipped (all gated, no default changed):**

- `CRISPEMBED_PPOCRV6_DET_PROFILE=1` — per-convolution cost table for the scalar
  detector. Result: 1x1 pointwise convolutions are **51.6%** of detector
  convolution time, depthwise **20.4%**, deconv 6.4%. This reframed H2: H1 *is*
  H2's lever, not a sibling item.
- `CRISPEMBED_EASYOCR_STAGE_BENCH=1` — splits the EasyOCR loop into
  detect/crop/set_width/recognize with width-rebuild counts. Result: **detection
  is 55% of the lane**, so the "CRNN is 2.2x the Tesseract LSTM" comparison was
  setting a whole-lane number against a component.
- `CRISPEMBED_TESSERACT_SEG_ROUTER=1` — H9 segmentation router, routing on
  detected column count. **9/9 correct** on the labelled fixtures.
- `CRISPEMBED_TESSERACT_PAGESEG_CLEANUP=1` — unbundles cleanup from the
  segmentation choice (they were forced to move together by one flag).
- `CRISPEMBED_CONV1X1_FAST`, `CRISPEMBED_CONVDW_FAST`, `CRISPEMBED_DOT_WIDE` —
  three CPU kernels, all measured **not** to be wins, all kept gated off.
- Three wasted Metal inits removed (`text_sr`, `tps_locnet`, `bert_ner`), each
  worth ~2.6 CPU-s / 6.7 s wall per invocation; old paths gated behind
  `TEXT_SR_GPU_LOAD` / `TPS_LOCNET_GPU_LOAD` / `BERT_NER_GPU_LOAD`.
- Width-sort key bug: the pre-pass hardcoded the 2-pixel detector margin while
  the loop applied it conditionally, so the external-geometry path sorted by
  widths it never requested (19 rebuilds against 15 distinct widths).
- `test-core-cpu-ops` grew equivalence guards for all three kernels, 172/172 on
  **both** NEON and AVX2. Each guard was verified to fail on an injected bug
  before being trusted.

**Retracted during the round — recorded because the retraction is the finding:**

- *"1x1 conv fast path is 9.1% faster on M1"* — did not replicate. As
  interleaved pairs: Mac +15.7/-1.5/-1.2/+1.9 (mean carried entirely by one
  outlier **baseline**; -0.3% without it), x86 -4.8% (5/6 negative). The kernel
  is **neutral on ARM and a ~5% regression on x86**.
- *"The sign flips with the instruction set"* — built on the above; one noisy
  Mac number against one real x86 regression.
- *"The same C++ runs 5-8x slower on M1 than x86"* — compared a contended Mac
  (load 30-110) against a quiet Xeon (0.38). The quiet Xeon totals 2,652 ms of
  convolution against ~2,350 ms for the Mac when quiet; the machines are
  comparable.
- *"Projection segmentation loses 92-98% of the text on real scans"* — was
  measuring a **bundled flag**. `CRISPEMBED_TESSERACT_PAGESEG` also disables
  cleanup; with cleanup left on, classical segmentation **beats** DBNet on
  receipts (616 characters vs 494).
- *"H5: model load is 0.37 s of the tesseract lane"* — that is **cold page
  cache**, not copy overhead, and it is the detector rather than the recognizer:
  1415 ms cold, **7.0 ms warm**.

**Methodology finding (the durable one).** PLAN §1's median-of-3 CPU-seconds
protocol cannot resolve effects of this size. Measured sd of the paired delta is
**8.1%** on the Mac and **5.2%** on the VPS, so resolving a 5% effect at 95%
needs **41** and **16** interleaved pairs. Comparing two separately-taken medians
is strictly worse than pairing, because drift lands entirely in one arm — and the
control bracket does not catch it: both tesseract controls agreed within 30%
across the very pair that read +15.7%.

### Archived in-flight rows

| 2026-08-03 | `chore/ai-act-imageroot` / `.codex/worktrees/chore-ai-act-imageroot` | **Picked: `--image-root` confined 13 of 33 endpoints, not all of them — the row below (round-3) states the opposite in good faith and was wrong.** A local `extract_image_path` lambda at `server.cpp:2093`, left over from before the confinement work, **shadowed** the file-scope `extract_image_path()` for every handler registered after it. Both names resolve, neither warns, and the local one calls `json_extract_strings` without `path_within()` — so confinement covered the first 13 endpoints (`/face` and `/detect` among them, which is why the biometric surface was never exposed) and silently missed the next 20: all 8 SR engines, `/restormer`, `/scunet/denoise`, `/instructir/restore`, `/adair/restore`, every `/preprocess/*`, `/scan/split`, `/scan/content`, `/ocr/document`. Confirmed by brace-depth analysis that the lambda stays in scope for all 20, and present on `origin/main` plus every `chore/ai-act-*` branch. Impact is an unauthenticated arbitrary-image READ that a deployer believed they had closed, with exfiltration via the SR/restore response (which returns the image base64-encoded); `output`/`file`/`model` were never affected because they call `extract_path_field` by its own name. Sharpest illustration: inside `/preprocess/dewarp` the write destination (line 2249) was confined while the read source (2246) was not. **Fix:** deleted the lambda, left a comment naming the hazard, and rerouted `extract_path_field` through core_json's depth-1 finder so confinement and parsing agree on which field is "the image" — the file-scope version previously used a bare `body.find("\"image\"")`, so a nested decoy `{"meta":{"image":"/a"},"image":"/b"}` made the server read a different path than a validating proxy in front would. **Why it survived three audit rounds:** `tests/test_image_root.py` only ever probed `/detect`, which sits above the shadow, so the test could not have caught it — it now probes `/scan/split` and `/scan/content` too (no model needed), i.e. both sides of the boundary. **Carry-forward:** the previous four findings came from re-checking claims already written down; this one came from re-checking a *fix* already verified by a *test*. A passing test proves the endpoint it names, not the sentence in POLICY it was written to support. Also in this pass: POLICY §1 now states we are the **provider** and not merely the deployer of the Space and WASM demo (Art. 3(3); Art. 50(1)/(2) are provider duties, (3)/(4) deployer ones, and Art. 2(12) does not reach Art. 50), and records that the Art. 2(12) FOSS reading covers our MIT code but not a system assembled with a `cc-by-nc*` or vendor-restricted checkpoint. Coordination: `examples/server/server.cpp` + `tests/test_image_root.py` + POLICY/PLAN only; no engine, graph or orchestrator code. **DONE, and the bug was MEASURED rather than inferred** — I rebuilt the unfixed `server.cpp` specifically to avoid shipping a fix for a defect I had only read. Same build, same flags, `--image-root` set: `POST /scan/split {"image":"<outside-root>/secret.png"}` returns `{"pages": 1, "width": 640, "height": 640}` with **0** rejections logged before, and `{"error": "missing 'image' field"}` with the rejection logged after; `/scan/content` identical. The nested decoy `{"meta":{"image":"<in-root>"},"image":"/etc/hosts"}` is refused after the depth-1 change, i.e. the server and a validating proxy now agree on which field is "the image". In-root reads still serve, and behaviour is unchanged when `--image-root` is unset (`path_within` returns true on an empty root). Landed as `54aeaecb` on `chore/ai-act-imageroot`. | **COMPLETED** |
| 2026-08-03 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Followed the crispembed-diff protocol properly and it found a fault in my measurements, not the model.** Two rule-2b violations by the tool itself: `Report` always carried `cos_global`/`mine_norm`/`ref_norm` and **no print site showed them**, and I quoted `cos_min` — a per-row worst case — as tensor parity all session. All internvl2/test print sites now emit `cos_min`, `cos_glob`, `max_abs`, `|mine|`, `|ref|`. **More intermediates:** the reference had only ever been dumped `--max-llm-layers 4`, which also skips `llm_output_norm` and `llm_logits` — stages the test could already compare but had never been given data for, leaving the harness 20 layers short of the decision boundary. Re-dumped to **54 stages** (`ref-full.gguf`, 108 MB). **Result — f16 is EXACT to the logits:** every one of 54 stages passes, `cos_min` included, `llm_logits` cos 1.000000 / max_abs 0.000069 / |mine| 1604.5433 vs |ref| 1604.5439. No code defect in this engine for this model. Shipped q8_0 reaches the logits at cos_glob **0.998919**, magnitudes 0.13% apart, with a smooth monotonic decline and **no discontinuity** — every 'jump' I chased (`vis_layer_12`, `llm_layer_1`) was a `cos_min` artifact. **Corrections pushed:** h2ovl-800m q8_0 is NOT degraded (cos_glob 0.999975, I called it 'cratering'); the 'sign is what survives' finding is **withdrawn** (built on two per-row worst cases); the q4_k withdrawal stands but for cos_glob 0.994→0.968 over 4 of 24 layers plus wrong decoded output, not 'anti-correlated'. ⚠ **Left for a decision, not taken unilaterally:** `is_pass()` keys on `cos_min`, so nearly every stage prints FAIL while globally excellent — a gate that always cries wolf. `crispembed_diff.h` is shared by every engine; re-keying it on one model's evidence is the mistake already made twice today. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked: `--image-root` does not confine what POLICY §4 implies it does.** The sentence is literally true — it confines every `{"image": ...}` READ — but it reads as a filesystem sandbox for the server, and three request fields bypass it entirely: **`/preprocess/dewarp` `"output"` is an arbitrary file WRITE** (fopen "wb" on a client-supplied path, so any file the process can write is creatable/truncatable — strictly worse than the read I originally fixed); **`/preprocess/tps-dewarp` `"model"` loads a client-supplied path as a GGUF and executes it as a ggml graph**, which is the exact hazard the SHA-256 pinning work is premised on; and `/pdf/dpi` `"file"` is an unconfined read. Fix: generalise the confinement helper to any path-valued field and apply it to `image`/`output`/`file`; add a separate `--model-root` for model paths, since a model legitimately lives outside an image directory and folding it into `--image-root` would be the wrong shape. Then state in POLICY §4 exactly what is confined rather than leaving the impression of a sandbox. Coordination: `examples/server/server.cpp` + POLICY/docs only; no engine, graph or orchestrator code. **DONE.** Confinement helper is field-agnostic now and applied to `image`/`output`/`file`; `model` gets its own `--model-root` rather than being folded in — a model legitimately lives outside an image dir, and conflating a code-execution surface with a data one is the wrong shape even where the directories coincide. `tests/test_image_root.py` grew the two cases that matter and asserts on the FILESYSTEM, not the response: after posting an output path outside the root, no file exists there — a 200 with an error body would have proven nothing. 8/8 pass. POLICY §4 now enumerates what each root covers instead of implying a sandbox. **Pattern worth noting across the last four findings:** each came from re-checking a claim I had already written down (marking coverage twice, harness pinning, this one), and each time the code was right where I had looked and wrong where I had not — an argument for the coverage tests now in CI over more careful greps. | **COMPLETED** |
| 2026-08-03 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Artifact audit + h2ovl-800m brought up to standard.** Audit found three gaps, all closed: (1) the 2b model repo carried a **4-vision-layer partial reference** (28 MB) alongside the full 24-layer one in fixtures — the partial could not have found today's vision drift, so it was replaced rather than left as a trap; (2) the **800m had no reference at all**, so its parity had never been measured, only its output eyeballed; (3) the 800m had no quality-tier quant. Ran the full regime for the 800m locally (f16 1853 MB, small enough): convert → bake ref (32 stages) → quantize → per-stage diff → decoded output. **f16 exact** (`llm_embed`/`llm_layer_0..3` all 1.000000, `vis_proj_output` 0.999701; `vis_layer_23` 0.998665 and unshuffle 0.998199 are f16-vs-f32 rounding, same class as the 2b). **q8_0 + vision-F16 reproduces the f16 vision numbers exactly** — the new quant rule confirmed on a second checkpoint. **Finding worth carrying:** the synthetic probe does **not** track decoded quality. 800m q8_0 reads `llm_layer_2 = +0.494781` and **transcribes at 1764 chars**; 2b q4_k reads `−0.268615` and emits confident nonsense. The distinction that survives is **sign** — inverted vs degraded-but-aligned. A per-stage threshold alone would have rejected a good artifact here. Recorded as the in-repo counterexample for HARD RULE #3. **Registry deliberately unchanged for the 800m** — stays on q4_k (676 MB, transcribes, fox exact) against q8_0's 1175 MB, because this is the edge/WASM model; q8_0 published as a tier, not promoted. Artifacts: 2b repo = f16 + q8_0(vision-F16) + full ref, q4_k withdrawn; fixtures = refs for **both** checkpoints. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked: a real gap in my OWN provenance work, found by re-verifying the claim rather than trusting it.** **12 server endpoints return unmarked AI-processed images** — `/text/sr`, `/pan/sr`, `/hat/sr`, `/dat/sr`, `/safmn/sr`, `/esrgan/sr`, `/swinir/sr`, `/tbsrn/sr`, `/restormer`, `/scunet/denoise`, `/instructir/restore`, `/adair/restore`. They base64 RAW RGB bytes into a JSON field, so they never touch `core_imgout::emit` and my earlier coverage grep (`stbi_write`/`P6`/`P5`) missed all of them. These are precisely the engines POLICY §5 is about — the ones that SYNTHESISE detail — so "every image CrispEmbed returns to you is marked" was false for the highest-risk surface in the document. Fix: route through `emit_to_string` so the base64 payload is a marked PNG, add an explicit `"format"` field so clients can tell what they got, keep raw under `CRISPEMBED_IMAGE_FORMAT=ppm` as the back-compat escape, and factor the 12 near-identical base64 blocks into ONE helper — duplication is exactly what caused the temp-file defect I fixed earlier this branch. Coordination: `examples/server/server.cpp` only; no engine, graph or orchestrator code; no overlap with the five other active branches. **DONE.** All 12 now encode through `emit_to_string`, so the base64 payload is a marked PNG (plus a C2PA manifest when an identity is configured); each response gained an explicit `"format"` field (`png`, or `raw` under `CRISPEMBED_IMAGE_FORMAT=ppm`) so clients can tell rather than infer. The 12 near-identical base64 loops are one helper now — duplication is exactly what produced the temp-file defect earlier on this branch. Verified against a LIVE server: `/adair/restore` returns `format=png`, the payload has the PNG signature, PIL decodes 96x96 RGB, and the chunk records `engine=adair` with `digitalSourceType=algorithmicallyEnhanced`. Documented in `docs/provenance.md` including the response-shape change. | **COMPLETED** |
| 2026-08-03 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Vision-stage parity raised to the f16 ceiling — and the answer was precision, not a bug.** Bisected the encoder: the reference carried `vis_layer_0..23` + `vis_pixel_unshuffle` since it was baked and nothing ever compared them, so the gap between `vis_patch_embed` 1.000000 and `vis_proj_output` 0.912992 had 24 layers to hide in. **At f16 every stage passes** (`vis_layer_*` 1.000000→0.999902, unshuffle 0.999691, proj 0.999974, `llm_layer_*` 1.000000) ⇒ the port is exact. Two apparent bugs disproved: the `vis_layer_12` discontinuity is quantization (f16 smooth through it), and unshuffle 0.380 is **not** a layout mismatch (0.999691 at f16). **Fix:** hold the internvl2 vision tower at F16 for the Q8_0 target — proj `0.912992 → 0.999974`, unshuffle `0.380373 → 0.999691`, every vision stage PASS on CPU, +13% size, page still transcribes. Decoder stays Q8_0 (output correct; F16 = the 4.4 GB file). **Scoped after measuring the sibling:** arch-wide it took `internvl2-1b` 758 → 1135 MB, inflating the edge/WASM model 1.5x — now gated on `ftype == Q8_0`, verified 0 conversions on edge q4_k and 98 on h2ovl q8_0, with `CRISPEMBED_QUANTIZE_NO_VISION_F16=1` to bisect. **Second rule this session narrowed after a sibling check** — the pattern is a rule from one checkpoint applied to a family with different goals. **Shipped:** q8_0 replaced in place (2591566112 B, sha `497cd047…`, verified byte-identical local↔remote), card carries the per-stage table, registry size 2592 MB, pins regenerated **242/0**, `--list-models` correct. ⚠ **Mishap worth recording:** a 10-min tool timeout killed a chained `cp && upload` mid-copy, leaving a truncated `h2ovl-mississippi-2b-q8_0.gguf` locally. Caught by checking size+digest before trusting it; source intact, nothing published half-written. Upload direct from the source with `path_in_repo` instead of copying first. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** provenance on TEXT output, the analogue of the image work. hOCR/ALTO/PDF already name `CrispEmbed` as producer but carry **no version**, so an archived document cannot be traced to a build. Adding it: `ocr-system` (idiomatic — Tesseract writes `tesseract 4.1.1`), ALTO `<softwareVersion>`, PDF XMP. Contained: the version is a compile-time constant, no plumbing. **Deliberately NOT doing the bigger piece, and recording why:** POLICY §6 distinguishes CTC/attention recognisers (transcribe) from VLM engines (which "confabulate through a smudge rather than leave it blank"), and an archived hOCR cannot tell them apart. Recording the ENGINE would be the true analogue of naming the engine in the image marking. But neither render call site has the engine in scope — `crispembed_ocr_render` (crispembed.cpp:6127) is a pure formatter over already-computed results, and the server's /document path (server.cpp:3328) builds the renderer before the pipeline runs — so it needs threading engine identity through the orchestrator plus a public C-ABI change. That is invasive across exactly the files `feat/easyocr-ggml`, `feat/ppocr-next-20260731` and `feat/ocr-engine-parity` are actively changing; starting it now would generate conflicts and step on their half. **Proposed as a follow-up for whoever owns the orchestrator next** — design would be an optional `ocr_render_set_engine()` plus one field on the pipeline result, not a signature change. Also noting: the 12 `/tmp/cpp_*.bin` writes in `src/layout_detect.cpp` are NOT the temp-file defect I just fixed — all gated behind `LAYOUT_DEBUG`, and the fixed names are a deliberate contract with a Python reference dumper (the C++ side READS `/tmp/py_cross_out.bin`); randomising them would break the parity workflow. Documented as a caveat instead (in contributing.md, with the residual risk of enabling it on a shared host). **DONE:** version now in hOCR `ocr-system`, ALTO `<softwareVersion>` (a standard element that was simply absent) and PDF XMP CreatorTool/ProducerTool, from ONE `producer_name()` so three string literals cannot drift. Plain text deliberately stays clean — callers pipe it and a header would land inside the transcription. `tests/test_render_provenance.cpp` (11 checks) reads the version from the same macro the code does, requires the formats to agree, and fails if the version is `unknown` — which would otherwise satisfy every check while recording nothing. Wired into `build.yml`. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** verify the claim I put in POLICY §5 / contributing.md that every emitted image is marked, by auditing EVERY image-writing call site. Result: CLI (15) and server (3) all go through `core_imgout::emit`; three writers in `src/ocr_orchestrator.cpp` do not, and on inspection they are internal — two `temp_png_path()` handoffs and one debug crop dump behind `CRISPEMBED_TESSERACT_CROP_DUMP_DIR`. So the marking claim holds for output, but the wording is looser than the truth and will be tightened. **The audit found a real defect on the way: `temp_png_path()` (ocr_orchestrator.cpp:257) builds a PREDICTABLE `/tmp/crispembed_ocr_<pid>_<counter>.png` and `stbi_write_png` opens it with `fopen("wb")` — follows symlinks, world-readable under default umask.** That is the same defect I fixed in `server.cpp` earlier this branch (`/tmp/crispembed_doc_<pid>_<n>.img` -> `mkstemp` 0600); I fixed one instance and missed this one. Same sensitivity: the content is the user's scanned page. Coordination: `ocr_orchestrator.cpp` is touched by `feat/easyocr-ggml` and `feat/ppocr-next-20260731`, so this change is confined to the temp-path helper and touches no engine dispatch, graph or crop geometry. **DONE.** Root cause was TWO hand-rolled copies of the same logic, so there is now one: `src/core/temp_file.h`, used by both. mkstemp creates the file itself (unpredictable, O_EXCL, 0600) and the path is returned, which stays safe for callers that must write by name because the file already exists and is ours. `tests/test_temp_file.cpp` pins what regresses silently: file already exists, no group/other mode bits, regular file, suffix survives (callers dispatch on it), and 32 calls give 32 paths differing in MORE THAN ONE position — plain distinctness would have passed the old `<pid>_<counter>` scheme, which differed by a single digit. Wired into `build.yml`. Marking claim tightened in POLICY §5 and contributing.md to "every image CrispEmbed *returns to you*", with internal temporaries called out as unmarked and deleted — the old wording overstated it. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** my own follow-up — the three benchmark harnesses were pinned to `CRISPEMBED_IMAGE_FORMAT=ppm` when PNG became the default, so they validate the LEGACY path and not what users get; a defect in the PNG path would pass them. On inspection the pinning was over-cautious: all three already use PIL, which reads both formats, and the only hard dependency is one magic-byte check in `tests/ocr_preprocessor_benchmark.py`. Doing it in two parts: (a) prove the invariant that makes unpinning safe — PNG and PPM must decode to IDENTICAL pixels, i.e. the format change is lossless and does not perturb any downstream metric; (b) make the harnesses format-agnostic and drop the pins. Coordination: touches only my own provenance work plus three benchmark scripts; no engine/graph/model code; all 12 other open rows belong to `perf/ocr-h-items`, `feat/easyocr-ggml`, `feat/ocr-engine-parity` and `feat/ppocr-next-20260731` and are untouched. **DONE.** (a) The invariant is now a test, not an assumption: emit the same pixels as PNG and as Netpbm, decode the PNG, require byte-identical pixels — gray and RGB, at 23x17 (deliberately not a multiple of anything). These harnesses measure PSNR/SSIM/CER on those bytes, so had the formats disagreed anywhere every restoration metric would have shifted when the default changed and read as a model regression. (b) Pins dropped from all three: the two PIL-based ones needed nothing, and `ocr_preprocessor_benchmark.py`'s magic-byte check now accepts both and names the temp file for what it actually is (a `.pnm` holding PNG bytes would mislead anyone inspecting the dir). Smoke-tested the real path: `--cleanup-only` emits a 606x1000 PNG the harness accepts, PIL decodes, carrying the provenance chunk. **Note for whoever runs the restoration benchmarks next:** they now exercise the default PNG path, so a first run after this may differ from older numbers only by file size, never by pixels. | **COMPLETED** |
| 2026-08-03 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Both loose ends closed.** (1) **q8_0 cosine reconciled — it was the backend, and benign.** Same binary/files: `vis_proj_output` is `0.998630` on Metal and `0.912992` on CPU; CPU min `0.912992` vs the parity kernel's `0.905481` (also CPU, different box). Crucially it does **not** reach the output — the same CPU path transcribes the full page at 1749 chars vs Metal's 1748, same text. The divergence lives in the diff harness, whose vision input is a synthetic gradient that amplifies numerical differences a real page does not. Two carry-forwards recorded in PERFORMANCE.md: never quote a vision-stage cosine without its backend (same artifact spans 0.913–0.999), and the 0.999 gate on `vis_proj_output` is mis-calibrated for that synthetic input — it fails artifacts that decode correctly. Left unchanged deliberately: a threshold retuned against one model is how gates rot. (2) **Registry health all green, verified not assumed:** 242 pins / 0 unpinned and `model_hashes.h` current; **242 distinct URLs, 0 non-200**; licence check rc=0. Also re-verified `h2ovl-800m` on CPU as well as Metal after the BOS-ordering change. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** real C2PA Content Credentials for image outputs, default-on, mirroring CrispASR (user decision). **Probe done first, and it moved the plan.** (1) CrispASR's C2PA does NOT port: its `format_for_ext` returns audio MIME types only, and its "native" path is a vendored `third_party/c2pa-audio` submodule that only does `sign_wav`/`sign_mp3`/`sign_m4a`/`sign_flac`. Reusable part is the CMake plumbing (`Findc2pa.cmake`, `CrispasrC2pa.cmake`, prebuilt fetch of c2pa-rs 0.89.3). (2) **c2pa-rs REJECTS self-signed certs** — "the certificate was self-signed" — so CrispASR's baked self-signed default cert would not work through this path at all; a leaf+CA chain is required. Verified working: 77-byte PNG -> 42,631-byte signed PNG, manifest reads back via `c2pa_reader_from_stream`. (3) **Container blocker**: we emit raw PPM/PGM and C2PA has no PPM binding, so PNG output is a prerequisite (`stb_image_write.h` already vendored). (4) Manifest overhead is ~42 KB, dominated by an auto-generated JPEG thumbnail — worth disabling for small crops. Also correcting a copy-paste hazard: CrispASR asserts `c2pa.created` + `trainedAlgorithmicMedia`, right for TTS which is wholly synthetic; our inputs are real captures we enhance, so the truthful assertion is `c2pa.edited` + `algorithmicallyEnhanced` — copying CrispASR's verbatim would have made every restored scan claim to be wholly AI-generated. Coordination: image-output paths + CMake only; no OCR/model/graph code; no overlap with the other five active branches. **DONE, per the recommendation the user accepted: default-on marking, NO key in the repo.** Images are PNG by default (Netpbm has no metadata container — the format was the blocker, not the policy) with an `iTXt` chunk naming the engine; `CRISPEMBED_IMAGE_FORMAT=ppm` restores raw output and is how the three benchmark harnesses are pinned rather than teaching each a PNG decoder. C2PA layered on when `CRISPEMBED_C2PA_CERT/_KEY` are set, `-DCRISPEMBED_C2PA_FETCH=ON` pulls c2pa-rs; absence of lib or cert is a supported state that still yields a marked PNG. `scripts/make-c2pa-cert.sh` builds a per-installation leaf+CA chain (self-signed is REJECTED by c2pa-rs; key must be PKCS#8 or you get an opaque ASN.1 error). Assertion is `c2pa.edited` + IPTC `algorithmicallyEnhanced`, NOT CrispASR's `c2pa.created` + `trainedAlgorithmicMedia` — right for TTS, false for us, and the test asserts the wrong term is absent. **Single stb_image_write definition moved to `core/image_out.cpp`** (was in ocr_orchestrator.cpp; duplicate symbols at link) — kept stdio in, since ocr_orchestrator writes crops by path and a test externs `stbi_write_png`. Verified: end-to-end via adair, PIL reads the iTXt, and the PNG is SMALLER than the PPM it replaces (19,505 vs 27,661 bytes). **Hardened to 35 checks + CI + docs:** the test now validates every chunk with an INDEPENDENT bit-by-bit CRC-32 (the table-driven one in image_out.h cannot validate itself, and a wrong chunk CRC is ignored by stb/PIL but rejects the file in strict decoders — also cross-checked against zlib.crc32); plus chunk order, the iTXt five-field layout, buffer==file byte-identity, MIME-matches-bytes, input rejection, and empty-engine handling. **Server sites now use the same path** — they still wrote raw Netpbm, so an image over HTTP was marked differently from the same image from the CLI; `/preprocess/dewarp` returns via a new `emit_to_string()` that hands back the Content-Type with the bytes so the two cannot disagree. **The tests now RUN**: all three are model-free and network-free, wired into `build.yml` on every push — they previously existed and gated nothing. `docs/provenance.md` added (what each level proves, why `algorithmicallyEnhanced` not `trainedAlgorithmicMedia`, the two c2pa-rs constraints, c2patool verification, env vars); five engine docs stated `> out.ppm`, which would now produce a mislabelled file, and are corrected. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** the last open finding from my own AI Act audit — **Art. 50(2) machine-readable marking**. POLICY §5 states plainly that CrispEmbed adds no watermark or C2PA provenance marking to any output and that an integrator "must add it yourself"; the Digital Omnibus grace period for systems already on the market ends **2 December 2026**. Nothing in `src/`/`examples/` mentions c2pa/watermark/provenance today (verified). Scope is the CAPABILITY, opt-in and OFF by default — POLICY's reasoned position is that document restoration is standard editing under Recital 134 and needs no marking, and I am not overturning that unilaterally; this closes the "you must build it yourself" gap for integrators whose use sits away from the document case. Constraint discovered first: outputs are raw PPM/PGM to stdout or file, formats with no metadata container, so the only in-band channel is the PPM/PGM header comment. Coordination: touches the image-output paths only, no OCR/model/graph code, no overlap with `perf/ocr-h-items`, `feat/easyocr-ggml`, `feat/ppocr-next-20260731`, `feat/ocr-engine-parity` or `feat/ocr-followups`. **DONE:** `CRISPEMBED_MARK_GENERATED=1` emits a Netpbm header comment from all 18 emission points (15 CLI, 3 server), naming the ENGINE so a reader can tell synthesised detail (ESRGAN/NAFNet/SCUNet) from resampling (deskew/dewarp) — not recoverable from the pixels. Off by default, so §5's document-case position is unchanged. Safe only because stb_image's PNM loader skips `#` runs (`stbi__pnm_skip_whitespace`, verified) — `tests/test_provenance_marking.cpp` pins that round-trip, off-by-default, and that every emitted line is a comment (a stray non-`#` line would be parsed as the image dimensions). Verified end-to-end on adair; output still decodes in PIL. Documented in POLICY §5 + README as what it is: a strippable comment with no cryptographic binding — tamper-evident provenance still needs C2PA and a signing identity. | **COMPLETED** |
| 2026-08-02 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** third AI Act audit pass. Verified the round-1/2 controls hold in code (gate at `crispembed_face_init` keyed on declared model type; no 1:N primitive; no prohibited-category model in the registry; both CI-enforced), then closed six gaps the earlier passes left. (1) POLICY.md never reached PyPI/pub.dev users — `setup.py` now stages it into the wheel (verified present in the built wheel) and both package READMEs carry the Art. 5 prohibitions + GDPR Art. 9 note; (2) `/face`+`/detect` read arbitrary server-side paths — new `--image-root` confines all 30 `{"image":…}` endpoints via `weakly_canonical` + component-wise prefix, new `tests/test_image_root.py` covers absolute/traversal/symlink/sibling-prefix escapes; (3) Art. 4 AI-literacy duty was absent → new POLICY §8, and §1 now admits the project *deploys* two systems (Space, WASM demo), not only ships a component; (4) Art. 50 restated as in force (2 Aug 2026 has passed) with the no-watermark absence stated as unresolved; (5) model downloads were unverified → SHA-256 pins for all 232 pinnable registry URLs (`examples/cli/model_hashes.h`, generated by `tools/fetch_model_hashes.py` from HF LFS oids), fail-closed on mismatch/unpinned/non-HTTPS, wired into `main-health.yml`; (6) uploaded pages moved off predictable `/tmp` names to `mkstemp` 0600. **Side finding: 8 registry URLs are 404** — `pix2struct-base-q8_0.gguf` and `lid-glotlid-f16.gguf` are filename typos (repos hold `-f32`/`glotlid-f16`), and `InstructIR`/`AdaIR`/4 `*-crispembed-GGUF` repos have no GGUF uploaded. **Follow-up in the same branch: 4 of the 8 now fixed.** `instructir-f16.gguf` (quantized from the published f32; output cos 1.0, max 1 LSB vs f32) and `pix2struct-base-q8_0.gguf` (byte-identical greedy decode vs f32) were built and uploaded to `cstr/instructir-GGUF` / `cstr/pix2struct-GGUF`; pix2struct's registry size said 300 MB but the real q8_0 is 467 MB. `glotlid` had two bugs — the `lid-` prefix is CrispASR's naming convention and never existed in this repo, and "3.3 MB" was wrong by ~250x (GlotLID-V3 f16 is 848 MB) — now points at `glotlid-f16.gguf`, with `glotlid-q8`/`glotlid-q4k` added; FUNCTIONALLY UNVERIFIED because this repo has no LID engine (`text_lid_dispatch.h` is an optional CrispASR header behind `__has_include`). Repo ids normalized to canonical lowercase `cstr/instructir-GGUF` / `cstr/adair-GGUF`, which previously only resolved via HF's case redirect. Unpinned URLs 8 -> 4. The remaining 4 are the VLM repos that do not exist, with no artifact anywhere on the backup volume, so they need real reconversion (`convert-qwen2vl-to-gguf.py` for german-ocr-3.1 and nanonets-ocr2-1.5b, `convert-internvl2-to-gguf.py` for both H2OVL). **~~TODO~~ RESOLVED 2026-08-02 (`feat/ocr-followups`) — AdaIR F16 was an engine bug, not a bad artifact, and this row called it correctly:** `adair-5d-f16.gguf` quantizes cleanly but aborts at run time on the default ggml_conv path with `GGML_ASSERT(buf != NULL && "tensor buffer not set")` in ggml_backend_tensor_set, from `adair_kernel` (src/adair.cpp:692) via `adair_conv`. f32 runs fine and InstructIR f16 from the same quantizer runs fine, so it is specific to src/adair.cpp's F16 path. Root cause NOT established: the scalar fallback (ADAIR_SCALAR=1) was too slow to finish, so whether the bug is confined to the ggml path is unknown. No f16 was uploaded; the registry points at the published f32 (115 MB) instead. **VLM reconversion follow-up: 3 of the 4 missing VLMs now built, verified and uploaded; unpinned URLs 8 -> 1.** Sources found: `keyvan-ai/german-ocr-3.1` (llama.cpp split GGUFs; byte-identical mirror of `Keyven/german-ocr-3.1`), `nanonets/Nanonets-OCR2-1.5B-exp` (the `-exp` suffix is why a plain `Nanonets-OCR2-1.5B` probe 404s), `h2oai/h2ovl-mississippi-{2b,800m}`. (1) **german-ocr-3.1** — merged upstream F16 LLM + F16 mmproj via `merge-llamacpp-qwen2vl-gguf.py` (4.12 GB, matching the original b58d7805 commit note), quantized to q4_k = 1684 MB (registry said 1301; corrected). Verified: near-perfect transcription of `scan_page_pd.png`. NB merging the *pre-quantized* Q4_K_M LLM instead is wrong — it yields a 2.3 GB hybrid, not the recorded recipe. (2) **h2ovl-800m** — 644 MB q4_k (registry said 398; corrected), verified legible full-page OCR with edge-model artifacts. (3) **nanonets-ocr2-1.5b** — 1346 MB q4_k, *exactly* the size the registry already claimed, verified near-perfect full-page transcription (421 tokens). **Two converter bugs found and fixed, both silent-failure classes:** (a) `convert-internvl2-to-gguf.py` routed the LLM attention layout off `config.model_type`, so H2OVL's Danube LLMs (model_type `llama` for 800m, `mistral` for 2b) went down the InternLM2 fused-`wqkv` branch, every lookup missed, `lw()` returned False silently, and the writer emitted **only the per-layer norms** — a GGUF with no LLM weight matrices that loads fine then segfaults in `ggml_mul_mat` on a null tensor (381 tensors instead of 493). Now routed on tensor-name presence, plus a post-export guard that aborts naming the missing tensors. This means PLAN's earlier "H2OVL-Mississippi-2B **Ported**" claim was stale for BOTH H2OVL models. (b) `convert-qwen2vl-to-gguf.py` wrote `qwen2vl.tie_word_embeddings` twice when a checkpoint ships no `lm_head` (gguf raises on duplicate keys), killing conversion after the vision tower was written; now derived once from whether `lm_head.weight` is actually in the checkpoint. **TODO — h2ovl-mississippi-2b is the one still unshipped.** It converts (565 tensors, 4.42 GB) and loads, but emits degenerate output at BOTH f16 and q4_k — f16 repeats one token then EOS, q4_k emits "." then EOS — so it is not a quantization artifact. Suspects not yet separated: chat/prompt template for Danube-1.8B, Mistral sliding-window attention, or the 32H/8KV GQA ratio (the working 800m is 16H/8KV). Its registry URL still 404s and is commented as such. **Measurement warning:** a parallel session drove load average to 75-316 during this work; the first full-page nanonets run produced zero tokens in 900 s purely from CPU starvation and would have been misread as a hang. Re-run VLM timings on a quiet machine. **h2ovl-mississippi-2b root cause: `use_msac`, now IMPLEMENTED.** The 2b sets `use_msac: true`, the 800m false — the only material difference between the working and broken model (same `template: h2ogpt2`, vocab, downsample/ps_version; rope_theta differs 10000 vs 100000 but is read correctly and the engine is arch-agnostic). H2OVL's Multi-Scale Adaptive Cropping tiles the page twice: coarse grid, then a fine grid keeping only ratios where `prior_cols%c!=0 && prior_rows%r!=0` (so it is not a sub-grid), concatenated `fine[:-1] + coarse[:-1] + fine[-1:]`, thumbnail last. Single-scale tiles give a model trained on that layout fluent nonsense. Implemented in `image_preprocess::preprocess_internvl_msac_rgb` + `internvl2.use_msac` dispatch. **Two parity bugs in the existing tiler had to be fixed for the fine grid to come out right:** (a) aspect ties were broken toward FEWER tiles; upstream breaks them toward more when `area > 0.5*size*size*blocks` — for 800x800 that is 1x1 vs 2x2; (b) a pass producing one block gets no thumbnail upstream, so `[:-1]` drops the tile itself and it contributes nothing — we gave 6 tiles where upstream gives 5. `tests/test_msac_tiling.cpp` pins all five cases against values transcribed from H2OVL `image_process.py`, including 800x800 where no admissible fine grid exists and we must DECLINE rather than fall back to single-scale. Model-free. The tie-break touches every InternVL model, so h2ovl-800m was re-run end to end: same 7 tiles (3x2), same 1122-byte transcription. **The 2b GGUF is being built by `tools/kaggle/h2ovl-convert`, not locally** — 4.3 GB safetensors + 4.4 GB f16 + 1.4 GB q4_k exceeds this machine's free space and a local attempt ran the disk out mid-write. The kernel asserts `use_msac` survived into the GGUF and the LLM attention/FFN tensors exist, OCR-smoke-tests the page fixture, and refuses to upload unless MSAC tiling ran and >=200 chars came back. **Also this pass:** adair-5d back on f16 (115 -> 59 MB) now that 67ec560c fixed the runtime shape bug — f16 vs f32 cos 0.99999994, max 1 LSB; and `--adair-model`/`--instructir-model` now resolve registry names, which they never did, so those two entries were unreachable from the CLI. Touches `examples/cli/model_mgr.cpp`, `examples/server/server.cpp`, packaging, POLICY/README, CI — **no OCR/model/graph code**. Verified: SHA-256 against 4 NIST vectors incl. the 56-byte padding edge; pinned download verifies, tampered pin rejected + cache left clean, unpinned refused, override works; biometric gate 11/11 still passes; image-root 5/5 + rejection logging; `tools/format.sh` clean.  **COORDINATION (h2ovl-2b, 3 sessions):** this branch owns the CONVERT half only — `tools/kaggle/h2ovl-convert` + the MSAC runtime. `feat/ocr-followups` owns the PARITY half (`tools/kaggle/h2ovl-parity`) and `tools/kaggle/h2ovl-publish`. I nearly broke that twice and both are worth recording: (1) my kernel called `create_repo` WITHOUT `private=True` on the same repo their publish kernel creates private — `exist_ok=True` does not flip visibility, so whichever ran first decided it, and mine would have made a model that emits 29 chars of `.assistant.assist` PUBLIC. Pre-created the repo private out of band, then removed my upload path. (2) Their row says they were *waiting on convert to publish the f16*; h2ovl-publish has since landed f16+q8_0+q4_k in that private repo, so that dependency is satisfied. **MSAC is implemented and the 2b is still broken, but the suspect list is now small.** Their parity run measured 27 stages at cos_min 0.999972 vs the Python blueprint, so the ported compute is right — it is NOT the graph and NOT the MSAC tile math. Combined with my fix that the call site hardcoded min/max_dynamic_patch to 1/12 and threw away the GGUF's declared 6 (19 tiles where the reference gives 13; the 800m is unaffected, same 3x2 grid either way), the remaining suspects are prompt construction, sampling, and detokenisation. **Next, locally, not on Kaggle:** `cstr/crispembed-regression-fixtures` already has `internvl2/h2ovl-mississippi-2b/ref.gguf` (112 MB) and the private model repo has the q4_k (1.46 GB) — that pair is enough for `tests/test_internvl2_diff.cpp` on this Mac, so my kernel's own 4-layer ref-gen is redundant and should be dropped. **CAUTION on the parity claim everyone is now reasoning from.** "27 stages at cos_min 0.999972" is almost certainly VISION-ONLY: `tools/dump_internvl2_reference.py` emits exactly 27 vision-side stages (`vis_patch_embed` + `vis_layer_0..23` + `vis_pixel_unshuffle` + `vis_proj_output`), and the parity kernel runs the dumper with `--max-llm-layers 4`, which would add `llm_embed` + 4 layers = 32 if the LLM side were counted. So what is established is that the InternViT tower and the projector are right. The 24-layer Danube-2 decoder, the LM head and the logits are NOT covered — and that is exactly where a mistral-vs-llama porting bug would live, the 2b being the `mistral` one while the working 800m is `llama`. Confirm the stage list in the parity log before concluding "the compute is right, the fault is downstream of the logits"; on this reading the decoder is the prime suspect, not detokenisation. **Decoder coverage now exists** (shared tooling, not their kernel): the diff harness ran `run_llm_forward`, printed the output SHAPE, freed the buffers and compared NOTHING — and returned 0 unconditionally, so even the vision stages were advisory. Added `llm_output_norm` + `llm_logits` to `dump_internvl2_reference.py` (emitted only on a full-stack dump; tied head read from the weights, not a config flag), and the harness now compares both, counts failures, returns non-zero, and checks **argmax of the last position separately** — cosine stays high while the argmax moves, and the argmax is what generation acts on, which is exactly this failure's shape. To get decoder coverage, regenerate the fixture WITHOUT `--max-llm-layers 4`. **h2ovl-mississippi-2b WORKS — unpinned URLs 8 -> 0.** The remaining bug was the hardcoded patch limit, not MSAC and not the decoder: with the model's declared `max_dynamic_patch=6` honoured, the run tiles to **13** (the reference geometry, was 19) and returns rc=0 with 1109 chars instead of 29 chars of `.assistant.assist`. Published, validated, repo flipped private->public now that it earns it (it was private by `h2ovl-publish`'s correct call while broken), registry entry activated at the measured 1459 MB, ref.gguf alongside it. **RESOLVED — it was the invocation, and it took BOTH halves.** Compared against upstream `conversation.py`/`modeling_h2ovl_chat.py`/`tokenizer_config.json` rather than guessing. (a) `add_bos_token: false` for BOTH h2ovl checkpoints, and upstream `chat()` just calls `tokenizer(query)` — the blueprint prompt has no `<s>`; we prepended one unconditionally. (b) `"OCR this image."` is too terse for a general-purpose VLM; upstream's own examples are explicit imperatives, and `qwen2vl_ocr` had already been forced into the same change. **Neither alone works** — the 2x2 matrix is: BOS+terse -> describes; BOS+explicit -> describes; noBOS+terse -> describes; noBOS+explicit -> TRANSCRIBES. That is why one-at-a-time attempts kept failing. Full page, defaults only: 2b 1806 chars verbatim (keeps curly quotes and `dis- played` hyphenation); 800m 1749 chars, also improved — it used to emit `NEW PAGE / <- / 36 / PRIDE AND PREJUDICE.` layout artifacts. Converter emits `internvl2.tokenizer.add_bos_token`; for older GGUFs the h2ogpt2 template defaults it off. `CRISPEMBED_INTERNVL2_PROMPT` / `CRISPEMBED_INTERNVL2_ADD_BOS` added for bisecting. **Trap worth remembering:** reading `tok.h2ogpt2` before the GGUF template key was parsed made the BOS default a silent no-op; only the A/B caught it. Superseded TODO was:  it *describes* the page ("The image presents a page from a book, specifically page 36...") rather than transcribing it, while quoting the content accurately. The internvl2 engine is handing H2OVL a captioning-style prompt; it needs an OCR/transcribe instruction, as the qwen2vl engine already does. **Also landed this pass:** `--check-sizes` in `tools/fetch_model_hashes.py` (found 4 more wrong registry sizes on first run: bidirlm-omni 2.6 GB->1834 MB, ppformulanet-l 180->252, transcoda 120->69, bttr-hw 5->11; all fixed, 241 clean, wired into main-health); adair-5d back on f16 (115->59 MB) after 67ec560c; and `--adair-model`/`--instructir-model` now resolve registry names, which they never did.| **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** validate the opt-in Tesseract DAWG scorer, model-owned runtime lookup, and diagnostic beam-confidence contract after the remote recoder merge; fix prefix ranking/token boundaries, wire runtime tests, and keep production dictionary scoring/calibration disabled | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** close the remaining beam-confidence comparator gap so `--require-beam-sequence-only` rejects fabricated word certainty as well as character certainty; add model-free coverage | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** make the documented Tesseract row-blob-bounds geometry A/B reproducible through the page comparator and repeated benchmark manifests, while keeping it diagnostic-only | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** align the standalone Tesseract geometry comparator with the row-blob-bounds benchmark switch and record the policy in its JSON output | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** reconcile the stale EasyOCR-plan int-mode status with the detailed parity evidence, keeping recoder/DAWG and full-page decoded parity explicitly open | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** preserve unmapped Tesseract recoder classes as explicit `<class>` diagnostics instead of silently dropping or exposing numeric class labels; keep full composed-script parity open | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** preserve valid composed recoder segments around unmapped classes with a diagnostic partial composer; leave the default decoder and full composed-script parity gate unchanged | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** consolidate repeated CRAFT/DBNet warm-graph probes into a versioned JSON manifest with explicit reference/native timing ratios and box-count quality status; keep device mismatch and page-text parity visible. Live scan-strip manifest: CRAFT native/reference `29,511.835/11,480.765 ms` (`2.57x`) with `106=106` boxes; DBNet `44,647.873/16,153.006 ms` (`2.76x`) with native `98` boxes, reference count unavailable in the timing-only probe. | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** produce an independent EasyOCR Python page manifest for `lines` mode and compare ordering, line grouping, crop geometry, decoded text, and confidence against the native DBNet→EasyOCR handoff; keep page parity separate from detector-only timing. Live `scan_strip.png`: Python CRAFT produced 11 lines; native DBNet produced 12. The first mismatch is line 0 (`"They are going to be , encamped near   Brighton"` vs `& They are going to be, encamped near   Brighton`), with geometry `[62,0,412,25]` vs `[46.97,0,423.54,21.76]`; all subsequent records shift, so page quality parity is **not** passed. | **COMPLETED — parity failed; quality TODO** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** replay independent Python EasyOCR line boxes through the native CRNN to separate recognizer/crop parity from DBNet detector geometry; preserve the failed page gate if identical boxes still diverge. External replay now uses exact caller-supplied boxes (no native 2-pixel margin), returns 11/11 regions, and still diverges in native text/confidence (line 0 Python `"They are going to be , encamped near   Brighton"` vs native `They are going to be, encamped near   Brighton`; confidence `0.8541` vs `0.5483`; line 4 is a severe recognition failure). | **COMPLETED — recognizer/crop quality TODO** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** run fresh `crispembed-diff` on exact Python line crops before changing recognizer math. English Gen2 line 0 and the worst line pass input, features, sequence input, both BiLSTM outputs, and logits; line 0 decodes identically, while the worst line reproduces Python's own poor decode. Input cosine is `0.99981`; recurrent/logit cosines are at least `0.99972`; feature global cosine is `0.99993` (sparse per-row feature cosine is not a valid promotion gate). The remaining page discrepancy is therefore detector geometry/crop selection, recognizer asset/preprocessing identity, and Python/native postprocess confidence—not an unexplained GGML LSTM divergence. | **COMPLETED — page quality still open; no recognizer math change justified** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** make the EasyOCR manifest boundary distinguish the padded postprocess `crop` from the actual recognizer input. Python and native manifests now emit `recognizer_crop`; `compare_easyocr_manifests.py --recognizer-crop-only` checks exact model-input geometry while preserving legacy crop comparisons by default. Contract tests pass, and the rebuilt `test-easyocr-pipeline` links at `[88/88]`. A real external replay confirmed 11/11 caller regions and showed the remaining text/confidence mismatch is genuine output quality, not a mislabeled crop field. | **COMPLETED — page quality TODO remains** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** add a dependency-free EasyOCR interoperability contract test covering Python `lines`/`words` ordering, crop/normalized geometry, and LayoutLM `apply_ocr=False` serialization; keep real-page reference parity as the separate live gate. `tests/test_easyocr_interop_contract.py` passes with 3 words, 2 grouped lines, and ordered LayoutLM sidecar metadata | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** retain PP-OCRv6 detector/crop/orientation/recognizer per-stage timings in the reproducible benchmark JSON; parser and stderr-capture slice. `tests/ppocrv6_pipeline_benchmark.py` now sets the bench switch, parses native stderr, preserves partial timeout telemetry, and labels unavailable stage rows. A live tiny German fixture produced detector/crop/orientation/recognizer timings and 34 detector boxes → 30 recognized results; full 10-fixture/medium quality sweep remains pending | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** add dependency-free PP-OCRv6 benchmark-parser, backend-capability, and OCR interoperability contract tests to the mandatory OCR regression smoke job; leave model/gold execution artifact-gated. Workflow YAML and all four smoke/contract checks pass locally; the gold step skips unless an artifact-equipped runner supplies `CRISPEMBED_GGUF_DIR` | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** generalize the PP-OCRv6 graph-gold harness from hard-coded small-only artifacts to explicit tiny/small/medium tier selections with tier-specific reference fixtures. The harness now supports all three tiers; tiny remains explicitly blocked until its legacy 16-tensor Arabic reference is regenerated as a full graph gold archive, while small remains the default accepted lane | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** add opt-in PP-OCRv6 detector graph-vs-CPU box geometry diagnostics without changing the production CPU accept-gate; report count, greedy matches, mean IoU, and minimum IoU for each diagnostic run. Implemented and compiled; the available tiny fox fixture reports graph=0 vs CPU=2, so detector graph geometry remains a quality/performance TODO and is not accepted by default | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** repair the manifest-driven O9 engine benchmark so structured detector specs (`{repo,file,revision}`) are normalized like recognizer specs; add a contract test and rerun Tesseract/PARSeq/PP-OCRv6 rows. Fixed structured detector normalization; Tesseract-LSTM `175.7 s` CER `0.040`, German Tesseract `101.8 s` unscored, PARSeq `1.206 s` unscored. The PP-OCRv6 artifacts load-fail (`missing stem conv`) and are now correctly marked errors instead of false `ok` rows | **COMPLETED** |
| 2026-08-01 | `chore/ai-act-hardening` / `.codex/worktrees/chore-ai-act-hardening` | **Picked:** third AI Act audit, run against `origin/main` after `f7f89032` landed. Re-verified the code-backed claims and found them all true: no 1:N/gallery primitive in the C ABI, no emotion/age/gender/ethnicity code anywhere, no scraping tooling, server gate fails closed before `crispembed_face_init()`, gate keyed on declared model type so a renamed `.gguf` is still caught, `/doc` temp uploads tracked and unlinked. Two gaps remained, both closed here: (1) the deployed GitHub Pages demo (`examples/wasm-ocr/index.html`) carried no notice at all while the HF Space had one — added an AI-output/data-locality/no-biometrics footer linking POLICY.md; (2) POLICY §3 and README asserted the absence of biometric-categorisation *models* in terms a reader could mistake for a guarantee about capability — CLIP/SigLIP zero-shot means the caller supplies the classifier, now stated. Also strengthened §7: Art. 53 does not engage for task-specific models at all (the quantization argument is the fallback), and Art. 53(2) does not waive the copyright policy or training-data summary. Docs/HTML only — no C/C++, no rebuild needed | **COMPLETED** |
| 2026-08-01 | `chore/ai-act-audit-followups` / `.codex/worktrees/chore-ai-act-audit-followups` | **Picked:** close the five gaps a second AI Act audit found in the `chore/ai-act-policy` work. (1) biometric gate moved into `crispembed_face_init()` so the Python/Rust/Dart bindings are covered, not just CLI+server — new ABI `crispembed_accept_biometric_use()`; (2) `check_registry_licenses.py` read only HF's `license` tag and missed `license_name`, so the 4 correct lfm2 rows failed — fixed, now exit 0, and wired into `main-health.yml`; (3)(4)(5) POLICY.md: Art. 50(2) reframed as reasoned-position-not-settled-exemption, OCR-VLM text addressed, and the regulatory dates corrected — the Omnibus is **Reg (EU) 2026/1744, OJ 24 Jul 2026**, not "adopted June 2026". Touches `src/crispembed.{h,cpp}`, `examples/cli/model_mgr.*`, bindings, POLICY/README/PLAN, `tests/check_registry_licenses.py` — **no OCR/model/graph code**. Verified: CLI + Python both refuse a recognition model without acknowledgement and load it with one, byte-identical embeddings either way; licence check exits 0; `tools/format.sh` clean. | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** fix O9 pipeline benchmark routing to use each manifest entry’s engine family (`ppocrv6`) instead of the tiered display name (`ppocrv6-tiny`), which had sent PP-OCRv6 fixtures through generic DB postprocessing and produced false `missing stem conv` load failures. Rebuilt CLI and verified tiny `4.98 s`/2 regions, small `20.82 s`/2 regions; medium exceeded the `120 s` guard and is recorded as a timeout, not a quality pass | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** resolve the remaining official PP-OCRv6 quality discrepancy by testing the HF/PaddleX preprocessing contract (RGB/BGR, resize, normalization) and CTC decode on known-text crops; promote a runtime change only if native output diverges from the official source under the same input. Result: preprocessing is aligned; remaining issue is checkpoint/vocabulary/line-crop quality | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** validate PP-OCRv6 checkpoint provenance, CTC vocabulary selection, and line-crop suitability against known-text crops before claiming quality parity; preserve native/reference decoded strings and timing evidence. Root cause was confirmed upstream: 320 is a minimum width, not a cap; the native/reference path now preserves dynamic CTC width and the 18,710-class space vocabulary | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** make the static PP-OCRv6 recognizer graph safe with dynamic-width line crops; bypass the fixed 320-wide graph for wider crops and retain the CPU reference path until a dynamic-shape graph is implemented and benchmarked. Live graph-debug test on an 800×100 fox line reports `input elements=55296` and cleanly bypasses the 320-wide graph; CPU decodes `The quick brown fox jumps` | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** implement and benchmark a truly dynamic-width PP-OCRv6 GGML graph, with width-keyed graph/cache ownership and CPU/Metal parity; keep the current explicit CPU fallback as the acceptance baseline. Implemented width-keyed graph rebuilds that retain the loaded GGUF source weights; a single process now runs 320-wide and 384-wide crops with graph outputs `80x3x384` and `96x3x384`, and graph-accepted text matches CPU (`De t 4 dg 14` / `The quick brown fox jumps`) | **COMPLETED — CPU graph validated; Metal parity pending** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** run the width-keyed recognizer graph on Metal for 320/384-wide crops, compare graph-accepted output and stage logits to CPU, and retain CPU fallback for any backend-specific divergence. Metal `MTL0` builds both widths and graph-accepted text matches CPU (`De t 4 dg 14` / `The quick brown fox jumps`); no dynamic gold archive exists yet for stage-logit comparison. Two-crop cold process timing was `28.32 s` Metal versus `2.94 s` CPU, so Metal is currently slower due pipeline compilation and remains diagnostic-only | **COMPLETED — parity passes; performance TODO** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** add reusable Metal pipeline/cache timing and dynamic-width gold logits, then benchmark warm 320/384-width recognizer graphs; do not promote Metal acceptance until cold/warm cost and numerical parity are recorded. Fixed repeated Metal scheduler reuse: re-plan Metal buffers per invocation while CPU retains allocation reuse. Same-width repeated Metal now exits 0 with identical text (`19.78 s`/2 crops); alternating 320/384/320/384 also exits 0 with identical text (`21.57 s`/4 crops). Dynamic stage-logit gold remains pending | **COMPLETED — stability fixed; performance/logit TODO** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** reduce Metal dynamic-width overhead by caching compiled graph plans per width or batching same-width crops; add width-specific gold logits before any Metal acceptance promotion. Added `tests/ppocrv6_width_benchmark.py`, which preserves decoded strings and graph shapes for grouped/alternating runs. Current `n=1` Metal timings: short `2582.1 ms`, wide `2250.0 ms`, alternating pair `2780.6 ms`; all return 0 with exact CPU-matching text and MTL0 shapes `80x3x384`/`96x3x384`. CPU controls are `468.3`/`403.8`/`463.8 ms`. | **COMPLETED — benchmark harness; cache/logit TODO** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** generate width-specific official-source activation golds and add CPU/Metal stage-logit comparison to the width benchmark; quantify whether current Metal numerical drift is acceptable before enabling any production graph gate. `tests/ppocrv6_width_benchmark.py` now accepts separate short/wide references and reports per-stage cosines. Fresh golds pass CPU logits cosine `0.999892` at 320 width and `0.999993` at 384; Metal passes `0.999861` and `0.999993`; decoded text is identical in every case. The older 320 archive was stale and was regenerated from the corrected official mirror | **COMPLETED — numerical parity passes; Metal remains opt-in for cost** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** benchmark Metal graph acceptance on the full PP-OCRv6 line/page route with dynamic-width gold coverage, then decide whether the recognizer graph can leave diagnostic-only mode. Added `--recognizer-graph` to `tests/ppocrv6_pipeline_benchmark.py`. The isolated 384-wide gold lane passes (`logits cos=0.999993`), but the German CC0 full route with 33 regions exceeded the 120 s guard under Metal graph acceptance; the prior CPU-accepted route completes in about 20.8 s. Keep recognizer graph acceptance diagnostic-only; full-route batching/residency is required before promotion | **COMPLETED — promotion rejected on measured cost** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** reduce full-page Metal graph cost by batching same-width line crops or reusing width-keyed graph residency across the detector→crop→recognizer loop; compare full-route text and stage timings against the CPU-accepted baseline. Before batching, added a safe per-page graph budget: with 33 detected regions the explicit graph request now selects CPU fallback and completes instead of timing out; measured German CC0 route `38.55 s`, 33/33 results, 1,146 chars, with recognize `32.65 s`. | **COMPLETED — safe fallback; batching still required for speed** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** batch same-width PP-OCRv6 line crops in the full route, preserving original order and per-line dynamic widths; compare batched CPU/Metal logits and decoded text against the current scalar fallback. Added native width-distribution telemetry and JSON capture. The current orchestrator still calls the recognizer one crop at a time; a German 33-region live run remained over the 180 s graph-debug guard, so no batching claim is made and graph acceptance stays budgeted/diagnostic-only | **COMPLETED — instrumentation and safety decision; no batch API yet** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** add a real PP-OCRv6 recognizer batch API grouped by identical dynamic model width, retain original result ordering, and require CPU-vs-Metal per-stage logits plus decoded-text parity before enabling it in the full route. Added the C ABI batch contract and wired the detector→crop→orientation→recognizer route through it. Live small-rec two-crop contract (fox + receipt) completed both items with byte-identical scalar/batch text; CPU sample was scalar `9.564 s`, grouped batch `6.070 s` (`1.58x`, warm-cache/small-sample evidence only) | **COMPLETED — safe grouped API; fused graph still required** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** implement fused GGML batch dimensions for same-width PP-OCRv6 crops, with bounded batch size, per-item error isolation, CPU/Metal logits cosine gates, and full-route German CC0 benchmark comparison. Added a bounded batch dimension to the tiny logits graph and kept it behind `CRISPEMBED_PPOCRV6_BATCH_GRAPH`. Important correction: the first `52.5 ms`/`35.1 ms` CPU smoke had `CRISPEMBED_PPOCRV6_FORCE_CPU`, which intentionally disables graphs, so it proved grouped scalar parity, not fused graph execution. A real Metal fused probe exposed a GGML pooling shape assertion; Metal is explicitly forced back to grouped scalar execution and no GPU promotion is claimed | **COMPLETED — safe gate; fused CPU proof still pending** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** make fused batching Metal-safe by preserving per-item spatial dimensions through pooling/reshape and adding CPU-vs-Metal logits cosine checks; then extend the fused path to large-stem SVTR only after the tiny lane is stable. Added an explicit Metal capability gate and fallback telemetry. German CC0 full route remains complete and text-bearing (`33/33`, `1,146` chars); current run was `68.75 s` total (`46.45 s` recognition), with `22` unique dynamic widths, so no same-width page batch gain is claimed | **COMPLETED — safe gate and baseline; shape rework remains** |
| 2026-08-02 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Picked:** the orphaned AdaIR F16 TODO from `feat/tesseract-kernel-opt` (that branch merged into `main` at `ee099eb0` and is gone; the item was left `IN PROGRESS`). Root cause identified before any edit: `tools/quantize.cpp` (~line 167) flattens every 4-D F32 conv weight to 2-D `[IC*KH*KW, OC]` in the output header, and `src/adair.cpp` infers three hidden dims from `->ne[3]`, which is `1` on a flattened tensor. Confirmed against the artifacts — `net.decoder_level1.0.ffn.project_in.weight` is `[1,1,96,510]` in `adair-5d-f32.gguf` and `[96,510]` in both `adair-5d-f16.gguf` and the rebuild. `hidden=1` ⇒ `half = hidden/2 = 0` ⇒ a conv with `ic=0` ⇒ the zero-size kernel descriptor the earlier audit saw. **No OCR/perf overlap — does not touch H1–H8 or any paused codex branch.** **Fixed:** `conv1x1_out_channels()` derives OC from `ggml_nelements(t)/ic`, correct under both layouts, with fail-loud guards at all three sites; `ADAIR_LEGACY_NE3_DIMS=1` keeps the old read so both arms are in one binary. Measured on `adair-ref.gguf` (64×64), same binary: f32 `cos 0.999382 / max_abs 0.027892` (reproduces the audit exactly ⇒ regression control), `adair-5d-f16.gguf` **`0.999383 / 0.027871`**, and the `adair-5d-f16-rebuilt.gguf` quantizer rebuild the audit also blamed gives the **identical** `0.999383 / 0.027871` — so neither artifact was ever bad. Independent artifact check: 60 sampled tensors f16-vs-f32 worst cosine `0.999998`, worst max_abs `1.22e-4`, i.e. pure F16 rounding. End-to-end through the real CLI (not just the diff harness): a 96×96 restore returns rc=0 on both models with the outputs agreeing at cosine `1.0`, max_abs `1/255`. **No timings claimed** — load average was 55–127 from parallel agents all session and the 64×64 fixture took `312 s` at f32 against a `2.65 s` quiet-box reference. Registry still ships f32 on purpose: repointing needs the f16 uploaded to `cstr/adair-GGUF` + a SHA-256 pin in `model_hashes.h`, which is an outward-facing step left for the owner. Exposure is narrower than "f16": only `tools/quantize.cpp` output flattens — a converter-emitted f16 keeps 4-D shapes (`surya-det-f16.gguf` has 79 genuinely 4-D F16 tensors), so the *producer* predicts the layout, not the precision. Follow-up recorded, not blind-fixed: `src/surya_det.cpp:700` and `src/tps_locnet.cpp:219` read conv OC off `ne[3]` the same way and would misread a quantizer-produced artifact (`src/cnn_embed.cpp:148` is the both-layouts precedent); neither ships one today and neither is verifiable on this box. | **COMPLETED — runtime fixed; f16 upload/registry DONE by `chore/ai-act-audit-round3` (adair-5d-f16 uploaded to cstr/adair-GGUF, registry repointed 115->59 MB, pinned)** |
| 2026-08-02 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Picked:** settle the `ne[3]` conv-output-channel follow-up left by the AdaIR F16 fix instead of leaving it as speculation. `surya-det-f16.gguf` is only 77 MB, so the claim IS testable here: run it through `crispembed-quantize` to produce the flattened layout and A/B surya detection against the converter-made 4-D artifact. Fix `src/surya_det.cpp:700` only if the test actually breaks. **Not** h2ovl-mississippi-2b — that needs ~9–10 GB for source + conversion and both volumes are full (internal 6.7 GB free, backups 7.8 GB), so it is disk-blocked, not skipped. **Result — the two suspects split apart.** (1) The flatten fires only on 4-D **F32**: quantizing `surya-det-f16.gguf` (4-D F16) to q8_0 leaves all 79 4-D tensors intact, so precision alone does not predict exposure — source dtype + producer does. (2) **`src/surya_det.cpp:700` is NOT a bug** — `g_conv` reshapes a 2-D weight to 4-D *before* that read, so my earlier 'same bug class' note was wrong. No change. (3) **`src/tps_locnet.cpp:219` WAS real and is fixed** — it reads `ne[3]` at load with no normalisation and `convert-tps-loc-to-gguf.py` **defaults to F32**, so a quantized tps-loc GGUF hits it; instrumented pre-fix the four layers loaded `ndims=2, channels=1` instead of `16/32/64/128`, and `channels` feeds the fc1 input width and per-layer output channels. Fixed with the `cnn_embed.cpp:148` convention. New hermetic guard in `tests/test_tps_locnet.cpp` (no model file) compares 4-D vs flattened builds of the same fixed-seed weights: worst control-point deviation `0.026871 px` → `0.000000 px`, suite 14/15 → 15/15. **The guard's first version passed against the broken code** — the synthetic `fc2.weight` was all zeros so the output was `fc2.bias` alone and never touched the conv stack; `fc2.weight` now carries small non-zero values. Recorded because a green new guard means nothing until it has been seen to fail. **Sweep extension — two quantized SR/denoise models turned out to have never been run at all, and both aborted.** `esrgan` (`GGML_ASSERT(cgraph->n_nodes < cgraph->size)`): not a layout bug — `esrgan_prep_conv` reshapes correctly — but the graph budget. Measured 18-conv x4 at 64x32: f32 builds `283` nodes vs the `n_convs*12+100 = 316` budget, quantized builds `335` (dequant cast + `ggml_cont` add ~3 nodes/conv) and overflows by 19; budget now `n_convs*16+128`. q8_0 vs f32 cosine `0.999998`/PSNR `51.89 dB`; **q4_k runs but degrades hard (`29.55 dB`, max_abs `91/255`) — q8_0 is the usable quant.** `scunet` (`GGML_ASSERT(a->ne[2] == b->ne[2])`): this one IS the flatten — the persistent kernel cache copies source `ne` verbatim, so a flattened weight caches as `[K*K*IC, OC, 1, 1]`; `scunet_run_conv` now restores the shape from call-site dims, with the conventions **measured** on the working f32 path (plain `[kw,kh,ic,oc]`, transpose `[kw,kh,oc,ic]`) rather than assumed. q8_0 vs f32 cosine `0.999999`/PSNR `60.54 dB`. `pan`/`swinir`/`tbsrn` clean both ways. Regression control: f32 output **byte-identical** before/after both patches. Suites after: tps-locnet 15/15, tps-warp 19/19, core-cpu-ops 118/118. | **COMPLETED** |
| 2026-08-03 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **h2ovl-mississippi-2b SHIPPED at q8_0 — the last 404 registry URL is closed.** Three invocation defects, all required, none in the graph (`test-internvl2-diff` was clean at f16 throughout): (1) wrong chat template — `h2ogpt2`, not ChatML, and this vocab has no `<|im_start|>` so every role marker was silently dropped (`5f617351`, mine); (2) spurious BOS + terse instruction, neither sufficient alone (`fcebf561`, parallel session — I verified `add_bos_token: false` directly in the checkpoint); (3) **the BOS default was evaluated before the vocab inference that sets `h2ogpt2`, so (2) was a no-op on every GGUF lacking the template key — i.e. all published artifacts.** Measured on the published q8_0: defaults → EMPTY, `ADD_BOS=0` → full transcription; after reordering, defaults transcribe (`98c2ae21`). Vision was never at fault — with only (1), `fox.png` read back exactly. **Quant ladder** (7 real stages vs the blueprint ref): f16 `0.999972`, q8_0 `0.998033`/`0.995498` transcribes, q4_k+attn-held-Q8_0 `0.922`/`0.544`, q4_k `0.594995`/`-0.268615` anti-correlated, q6_k **fails to load**. Q8_0 is the floor for this checkpoint; not a shape issue (2560/6912 are 256-divisible). **q4_k withdrawn from HF** — it emits confident wrong text, the worst OCR failure mode. `tools/quantize.cpp` **warns** rather than refuses: my first version refused for the whole arch and broke `internvl2-1b` (ships q4_k, works) and `h2ovl-800m` (recorded verified at q4_k) — caught by the regression check. **Shipped:** repo public + carded per-precision, registry → q8_0 (2292 MB), SHA-256 pinned via the generator, **242 pinned / 0 unpinned**, `--list-models` shows it. Reference at `cstr/crispembed-regression-fixtures` → `internvl2/h2ovl-mississippi-2b/ref.gguf`. Harness fix `e5db01f1` (diff no longer fabricates FAIL rows for absent stages). | **COMPLETED** |
| 2026-08-02 | `chore/ai-act-audit-fixes` / `.codex/worktrees/chore-ai-act-audit-fixes` | **Picked:** fourth AI Act audit, run against `main` at `52172c10`. Re-verified the code-backed claims and this time **executed** the gate against real GGUFs (yunet + sface pulled from `cstr/*`) rather than trusting that the test exists: all 8 prior cases pass, including the renamed-model case. Also verified the registry is clean of emotion/age/gender/ethnicity models (555 entries), that no training code exists (so §7's "quantization only" argument holds), and — since Reg (EU) 2026/1744 postdates the assistant's knowledge cutoff — checked POLICY's whole date table against the EUR-Lex text: **accurate**, including 2 Dec 2026 for the new Art. 5(1)(ba)/(bb) NCII/CSAM prohibitions. Three gaps found and closed here: (1) `--dim` returned *before* the gate on both CLI paths that reach it, making the CLI laxer than `crispembed_face_init()` — all three CLI sites now share one `cnn_biometric_ok()` helper keyed on declared type (which defaults to `recognition`, so it fails closed), and the gate test grew 3 cases; (2) POLICY §4 claimed "both gates key off declared type" while the `--face-pipeline` gate was unconditional-before-load — that path is now type-keyed too, so the sentence is true as written; (3) neither POLICY nor README told deployers that the server acknowledgement is **once per process** with **no authentication** and server-side-path input — documented in both, plus a startup warning when a recognition model is loaded on a non-loopback bind. Touches `examples/cli/main.cpp`, `examples/server/server.cpp`, `tests/test_biometric_gate.py`, POLICY/README/PLAN — **no OCR/model/graph code**. Gap (1) was mis-placed from the start, not a regression: `git log -L` shows the gate was added *below* the pre-existing `--dim` early-return in `6d87d6bd`. Verified before/after with the same toolchain — a CLI built from `HEAD:examples/cli/main.cpp` prints `128` (sface's template width) unacknowledged on **both** paths; the patched CLI refuses both. Gate test now 11/11 with real yunet+sface GGUFs, server warning fires on `--host 0.0.0.0` and stays silent on loopback, `--face-pipeline` still refuses without ack and runs with it, text embed + `--dim` unaffected, `format.sh --check` and `check_registry_licenses.py` clean | **COMPLETED** |


## July 20, 2026 — PLAN.md active-work board cleared (all in-flight items landed)

The `PLAN.md` "🚧 Active work in flight" table had accumulated 18 rows, **every
one marked DONE/landed**, plus a "Pending work" section whose A1–A4 items and the
`modern-bert` "FOUND" diagnosis had all shipped but were still written as open
(the same staleness that sent a fresh session chasing an already-shipped
modern-bert task). Verified each against the live code, then cleared the board and
the shipped "pending" blocks. No code changed. This entry indexes what was
removed and preserves the specifics for items lacking their own dated section.

**Already covered by a dedicated HISTORY section (pointers only):**
- **`gemma-embedding` EmbeddingGemma GGUFs** — July 17 entry (`138ee0c`).
- **JSON I/O hardening + `core_json` + community-GGUF compat (#34/#33, A1–A4,
  B1/B2, scalar migration, parity + precision-control automation, CI drift
  guards)** — July 16 entry. Re-verified in code: A1 all four endpoints
  (`/embed`, `/rerank`, `/ner/extract`, `/kie/extract`) route through
  `json_extract_strings`; A2 generic `general.architecture` hparam read +
  `CRISPEMBED_ARCH_HPARAMS`/`CRISPEMBED_STRICT_HPARAMS` gates; A3 matrix has 10
  entries with automated `prove_quant_control.py`; A4 emsdk pinned 6.0.2 across
  workflows + `main-health.yml` red-`main` self-report. All shipped.
- **Community `modern-bert` BPE-tokenizer GGUFs** — `77b829b`+`d3f447b` (recorded
  in the July 16 entry + LEARNINGS). Model-string-authoritative tokenizer dispatch
  (gpt2→BPE over the vocab-size heuristic), BPE merges from the
  `tokenizer.ggml.merges` KV array, GPT-2 ByteLevel regex pre-tokenizer, loader
  aliases + inverted dual-RoPE-theta/SWA metadata + exact-erf GeGLU (arch-gated so
  the GTE-v1.5 tanh path is untouched). q8_0 vs HF `emb_ln_out` cos=0.999928, 22
  layers 0.9999+, f16 control 1.000000 every stage, final CLS q8_0=0.999602;
  tokens match HF `[50281,25521,1533,50282]`; matrix margin −0.089→0.51.
- **Transcoda-59M OMR** — July 13 entry. **DeepSeek-OCR-2 #4 stacked MoE experts
  (−1.3 GB)** — July 13 entry. **DBNet scanline box scoring (28×)** — July 13
  entry. **QKV-fusion probe (negative) + detector-postprocess audit** — July 13
  entry. **Kaggle CUDA Class-A/Gap-5 confirmation (portfolio 14→0)** — July 13
  entry.

**Encoder ground-truth parity harness (A3 follow-on, shipped 2026-07-16).**
Extended the A3 matrix from rc/shape/garbage-guard to per-stage ground truth vs
the original HF model. Tools: `tests/hf_parity_community.py`,
`tools/dump_encoder_reference.py`, `tests/test_encoder_diff.py`,
`CRISPEMBED_DUMP_LAYERS_GGUF`. Measured q4_k vs HF fp32: bge-small 0.9962, MiniLM
0.9919, nomic-v2-moe 0.9797, nomic-v1.5 0.9515 — all confirmed **quant floor, not
bug** by the f16/f32 control (both encoder paths cos=1.000000 at every layer;
nomic-v1.5's drop is a sharp last-block step, layer_10 0.9977→layer_11 0.9499, a
real quality fact → prefer f16/q8). Precision control automated: matrix entries
carry `control_file`+`control_min_cos`, `prove_quant_control.py --all` proves
quant-not-bug in one command. Found three code-invisible harness bugs: last block
renamed to `encoder_out` (feeder-to-pooling silently absent; missing stages now
FAIL), `NomicBertModel.forward()` rejects `output_hidden_states` (forward-hook
fallback), and a pre-LN-vs-post-LN structural-gate mismatch (fixed by capturing
block-0 input via `forward_pre_hook`; gate now prints `|ours|`/`|ref|`).

**e5-small / granite community-matrix closure (2026-07-16).**
`granite-embedding-107m-multilingual` ADDED (first SPM matrix entry, `bert` +
`t5`/unigram → SPM via model-string dispatch, CLS): q4_k `emb_ln_out` cos=0.999951
+ 6 layers 0.9928–0.9969, final CLS 0.996145, f16 control 1.000000, margin 0.31.
`multilingual-e5-small` **CLOSED as a won't-fix**: the `rodion-m` fp32 GGUF omits
`bert.position_offset` so crispembed uses 0, but intfloat's XLM-RoBERTa e5 needs
offset 2 → structural gate cos=0.467 (pure position shift, norms match). Not
auto-detectable — granite shares the same RoBERTa bos=0/eos=2 SPM tokenizer yet
needs offset 0, and a `position_embd` row-count heuristic was ruled out (e5,
granite, bge GGUFs all `[384,512]` with ctx=512). The offset must be carried in
the GGUF; no speculative heuristic shipped.

**Official `lfm2` LFM2.5-Embedding GGUF now loads (2026-07-16).** Same class as the
modern-bert fix, bigger: `src/lfm2_embed.cpp` was written for our converter's
`lfm.*` tensor names + `lfm2.<our>` hparam keys + a `lfm2.layer_types` string, so
`LiquidAI/LFM2.5-Embedding-350M-GGUF` (canonical `blk.N.*`/`lfm2.*`, no layer-types
string) aborted on a missing tensor. Fixed with tensor/hparam aliases, conv/attn
layer-types derived from tensor presence, `head_count_kv` read as a per-layer array
→ max, and a memory-preserving reshape of the depthwise-conv weight to `[K,1,C]`
(the export ships it 2D `[K,C]`, which crashed `ggml_conv_1d_dw`). Validated via
`test-lfm2-diff` vs the raw HF `Lfm2BidirectionalModel`: q8_0 = 0.9999 at every
stage (post_embed gate + 16 layers + `cls_norm` pooled) on short and long text;
f16 control 1.000000 every stage; garbage-guard margin 0.76. Matrix entry
`LFM2.5-Embedding-350M` added.

**OMR engines + fixtures (shipped 2026-07-13, on `main`).**
- **Polyphonic-TrOMR** (`feat/tromr-engine`) — `src/tromr_ocr.cpp` (cos 1.0 / 100%
  argmax / byte-exact); HF `cstr/tromr-GGUF` (f32 + q8_0 31 MB, F16 backbone,
  Apache-2.0 card); registry + regression fixture (cer 0.000).
- **Flova / omr_transformer** (`feat/flova-omr`) — handwritten/whiteboard OMR
  (donut-swin + mBART VED → LilyPond). `src/flova_ocr.cpp` (cos 1.0 / 40-40
  argmax / byte-exact incl. native preproc), `test_flova_diff.cpp`, CMake,
  dispatcher + registry. HF `cstr/flova-omr-GGUF` (f32 573 MB + q8_0 162 MB,
  Apache-2.0). Fixture (`feat/flova-regression-fixture`, `67ddc99`): `staff_flova.png`
  + golden LilyPond `c'2 a''8 c''8 r4 c'1 e'8 c'8 c'8 a''8 f'4 a'8 c'8`, cer 0.000.
- **SMT regression fixture** (`feat/smt-regression-fixture`) — `staff_smt.png`;
  `run_one.py --name smt` PASS (garbage-guard + cer 0.000 vs `smt-grandstaff-q8_0`
  from `cstr/smt-grandstaff-GGUF`, CPU==Metal, deterministic bekern decode).
  Completes the OMR guardrail trio (SMT/TrOMR/Flova).
- **SMT++ full-page pianoform OMR** (`feat/smt-fp-fullpage`, `PRAIG/smt-fp-grandstaff`)
  — fp checkpoint = `antoniorv6/SMT` main rewrite (scaled attn `d_head^-0.5`, no
  pre-head ReLU, decoder tensor rename, head Linear, `reduce_ratio=1.0`).
  **Correctness fix: NO invert** — the checkpoint's repo is plain Grayscale+ToTensor
  (I'd wrongly copied SMT-plusplus `RandomInvert`); WITH invert even the real HF
  model degenerates to `8 . r` repetition, WITHOUT it reads correctly and
  terminates (per-stage cos was 1.0 either way — only the decoded roundtrip vs a
  no-invert reference caught it). **Perf 485→~26 ms/step (~18×)** via persistent
  device KV + reserved gallocr sched-free + cross-K/V stored once mul_mat-ready;
  whole page ~2 min (was not finishing). Byte-identical CPU==Metal, f32==q8_0
  (2312 tok). q5_k shipped (13 MB, 0.04% token-CER); q4_k not (degenerates).
  Quantizer guards for `decoder.out_layer` + ConvNext encoder → Q8_0. HF
  `cstr/smt-fp-grandstaff-GGUF` (invert=false), registry `smt-fp`. CI fixture
  skipped (full-page decode too slow).
- **CROHME handwritten-formula fixtures** (`feat/handwritten-fixtures`) — closed
  the bttr/hmer/posformer `expected_text: null` gap. **No bug**: the 3 CROHME
  models were guarded on a *printed* (out-of-domain) image; on rendered CROHME 2014
  all three read simple formulas correctly + deterministically (CPU==Metal). Added
  a `sample_hf` harness mechanism that fetches one CROHME image from
  `Kitajiang/test2_CROHME2014` (pinned rev, row 23 `C_t=C+C=2C`) at test time so
  the CC-BY-NC-SA data stays OUT of the MIT repo; pinned `expected_text` for all 3
  (`run_one` cer 0.000).

**Unlimited-OCR stacked MoE experts** (`feat/uocr-stacked-experts`, 2026-07-14).
Verbatim port of the ds-ocr2 #4 stacked-experts win (same DeepSeek-V2 MoE).
Kaggle-reconverted `baidu/Unlimited-OCR` (byte-validated vs source; the v1 3 h hang
was fixed by the single-thread OMP/BLAS converter prefix), uploaded f16+q4_k
`-stacked` to `cstr/unlimited-ocr-crispembed-GGUF` (rev `b11fef884fee`, non-clobber).
M1 Metal q4_k A/B: output byte-identical on all 3 loader paths; peak footprint
4.32→3.11 GB (−1.21 GB, −28%). Registry promoted to stacked-default; regression
entry `unlimited-ocr-stacked`.

**layout-heron `dec_0_cross_out` — the last portfolio FAIL** (`debug/layout-cross`,
`d7f0480` fix + `e9bba14` docs). NOT an inference bug: the 300 decoder queries are
picked by `partial_sort` over ~8400 near-tie encoder proposals
(`layout_detect.cpp:1318`), so a tiny backend FP delta in `enc_output` (cos 0.99999)
reorders near-tie ranks and the index-aligned `cross_out` cos craters (mean 0.79 /
min −0.08 Metal) even though the VALUES are correct (final boxes unaffected —
score-sort + NMS). Fixed by comparing `dec_0_cross_out` **permutation-tolerantly**
(best-cosine match: PASS Metal 0.947/0.999, CPU 0.967/0.999; simulated scrambles
still collapse to ≤0.08 vs the 0.85 gate). Portfolio **14→0 FAIL**.

**Kaggle reranker τ-eval** (`crispembed-imatrix-quant`, 2026-07-13) — full
7-reranker roster on the n=30 corpus. **imatrix always cuts q4_k score-drift (7/7)
but its effect on ranking τ is model-dependent**: big win on ms-marco-L-12
(0.853→0.929) + jina (0.929→0.942), neutral on bge, but **degrades** both mxbai
rerankers −0.076 (iq4_xs beats q4_k+imatrix there). So `q4_k+imatrix` is **not** a
universal reranker recommendation — validate per-model (the old n=5 corpus missed
both the mxbai regression and the ms-marco-L-12 win). All imatrix quants
re-uploaded to `cstr/*-GGUF`; jina q4_k-imatrix also validated locally on Metal
(EN+DE rerank correct).

**crispembed Dart pub.dev quality** (`chore/pub-crispembed-dart`, 2026-07-15) —
crispembed **0.15.1** to 160/160 pana points (added example/README, enabled
`lints/core`, brace/dangling-doc fixes). Docs/lint only, no behaviour change.

---

## July 17, 2026 — Community/official `gemma-embedding` GGUFs load (routing + SPM-BPE tokenizer + Dense)

crispembed could not load the official llama.cpp EmbeddingGemma export
(`ggml-org/embeddinggemma-300m-*-GGUF`, `general.architecture=gemma-embedding`):
it crashed, and the naive "make it load" fix produced silently-weak embeddings.
The handover blamed missing Dense modules or a gemma-norm convention; both were
wrong — the dominant bug was the **tokenizer**, same class as the modern-bert
fix. All on `main`.

- **Crash → routing (arch-gated, 3 edits).** The hyphenated `gemma-embedding`
  missed the decoder allow-list and fell through to the generic MHA encoder
  graph, whose QKV reshape overran the GQA K/V (3 heads / 1 kv, head_dim 256) →
  `GGML_ASSERT`. Routed to `decoder_embed.cpp` (already a full Gemma3 block);
  forced `is_bidirectional` (export sets `attention.causal=false`, no
  `is_bidirectional` key).
- **The real bug: SentencePiece loaded as char-level BPE.** The decoder
  tokenizer loader hardcoded BPE, but the GGUF is a llama.cpp SPM export
  (`tokenizer.ggml.model=llama`, `scores`, **no** `merges`). Loaded as
  BPE-with-0-merges it char-tokenized every input ("hello world" → 11 single-char
  tokens) → garbage (garbage-guard margin 0.038). Fix: detect merge-less+scored
  vocabs → route to `SentencePieceTokenizer`, and add an **SPM-BPE bigram-merge**
  mode (llama.cpp SPM algorithm). Gemma's `scores` are merge RANKS, not unigram
  log-probs, so the existing Viterbi over-segments (picks `▁w+or+ld` over the
  single token `▁world`). Viterbi kept as default (XLM-R Unigram untouched). Also
  honor `add_space_prefix=false`. Tokens now match HF token-for-token; margin
  0.038 → 0.39.
- **Dense baked for HF-compatibility.** The `gemma-embedding` GGUF omits the
  SentenceTransformers Dense head (llama.cpp applies it from an external file), so
  raw output is orthogonal to real EmbeddingGemma (cos −0.02 vs HF). New tool
  `models/add-st-dense-to-gguf.py` copies the GGUF verbatim (raw quant bytes, all
  metadata) and appends `dense.0/1.weight` (F32) from `2_Dense`/`3_Dense` — which
  `decoder_embed.cpp` already applies post-pool. Result: cos vs the full HF
  `SentenceTransformer` pipeline = **0.985** (min 0.9852 / mean 0.9891 over the
  parity triplet). Backbone control isolates it: cos(pre-Dense mean-pool) = 0.9835,
  so no norm bug — the residual is the QAT-vs-vanilla checkpoint difference + the
  known gemma3-backbone/Dense-bottleneck discrepancy + q8_0.
- **Shipped:** HF `cstr/embeddinggemma-300m-GGUF/embeddinggemma-300m-qat-q8_0-dense.gguf`
  (gemma-license card + provenance); registry `embeddinggemma-300m-qat`; matrix
  entry `embeddinggemma-300m-qat` (arch `gemma-embedding`) with the HF-parity gate
  — all **10** community-matrix entries PASS. The general routing+tokenizer fix
  now loads any llama.cpp SPM decoder-embed GGUF, not just this one.
- **Two self-corrections worth recording** (see LEARNINGS): I nearly shipped the
  matrix entry without its HF-parity gate on a false premise (sentence-transformers
  *is* installed — its bare import fails on the `USE_TF=0` TF-integration gotcha);
  and the round-trip audit of `add-st-dense-to-gguf.py` caught it re-emitting
  `GGUFReader`'s synthetic `GGUF.*` header pseudo-keys as literal metadata
  (kv_count 35→38, "Duplicate key" warnings) — fixed, GGUF re-uploaded.

## July 16, 2026 — JSON I/O hardening + `core_json` + community-GGUF ecosystem compat

Landed a cluster of correctness fixes around HTTP/CLI JSON handling and
community-GGUF loading, plus a ground-truth parity methodology. All on `main`.

- **#34 — server JSON input parser mis-split escaped payloads.** The embedding
  endpoints hand-scanned request bodies: `body.find(']')` took the first bracket
  even inside a string value, and the `"`-pair loop ignored `\"`/`\\`. A payload
  whose inputs contained `]`, `\"` or `\\` produced the wrong input cardinality
  ("returned 7 embeddings for 6 inputs"). Fixed with an escaping-aware parser.
- **A1 — completed the migration.** `/embed`, `/rerank`, `/ner/extract`, `/kie`
  carried the identical bug (worse than assumed: a `]` in the *first* array
  element dropped *every* element). Zero delimiter-scan parses remain.
- **B1/B2 + centralization → `src/core/json.h` (`core_json`).** Completed the
  escaper (was `"`,`\`,`\n` only; now every control char per RFC 8259, the exact
  inverse of the decoder — round-trip property tested over all 256 bytes) and made
  key location structural (a decoy `"key"` *value* no longer matches, reachable via
  `/ner` labels). The server AND CLI each had a diverged `json_escape` (3 vs 5
  chars, both echoing OCR text) — unified into `core_json`; the CLI's latent
  control-char bug fixed for free (proven live: CLI `--json` on tab text emits
  `"a\tb"`, strict-JSON valid). Also routed the server's ~14 scalar/image field
  reads (`conf`/`threshold`/`max_tokens`/`extract_image_path`) through `core_json`,
  closing the same decoy bug for scalars. One env gate
  (`CRISPEMBED_SERVER_LEGACY_JSON=1`) reverts the whole surface for A/B.
- **#33 — nomic-embed-text-v2-moe wouldn't load** (`missing required tensor
  attn.q.weight`). Community/llama.cpp GGUFs use the `nomic-bert-moe.*` metadata
  keys + fused `attn_qkv`/stacked `ffn_*_exps` tensors. Fixed; HF cosine parity
  mean 0.9839.
- **A2 — arch-driven hparams + strict mode.** Generalized #33: read
  `general.architecture` and derive `<arch>.<field>` keys, so any community GGUF
  resolves with no new code (the per-model alias list stops growing). Missing
  *required* hparams previously fell back to silent defaults (384-dim/6-layer) →
  silent-garbage embedding; `CRISPEMBED_STRICT_HPARAMS=1` makes it hard-fail.
  A/B: existing models byte-identical on/off; nomic byte-identical to #33's build.
- **A3 + ground-truth parity.** A community-GGUF import matrix
  (`tests/community_gguf_matrix.json` + `run_community_gguf.py`) — because we tested
  our own `cstr/*` conversions, not the ecosystem's (which is what #33 was). Added
  HF/PyTorch per-stage parity (`tools/dump_encoder_reference.py` +
  `tests/test_encoder_diff.py` + `CRISPEMBED_DUMP_LAYERS_GGUF`) and automated the
  precision control (`prove_quant_control.py`): re-run at f16/f32 to prove a low
  q4_k cosine is quantization, not a bug. Proven for bge-small (f32=1.000000/stage),
  nomic-v1.5 (f16=1.000000), nomic-v2-moe (f16≥0.9998) — all three encoder paths
  (bert, nomic-bert, nomic-bert-moe) are exact; all quant gaps are quantization.
- **A4 — CI drift guards.** WASM CI had sat red for 2 days from an unpinned
  `setup-emsdk` drifting `latest` 6.0.2→6.0.3 (a clang that SIGSEGVs on
  `layout_detect.cpp`). Pinned 6.0.2; added `tools/check_workflow_pins.sh` (fails
  on any unpinned toolchain step, self-tested both arms) + a daily `main-health`
  cron that self-reports a red `main`.
- **Fixed + shipped: community `modern-bert` GGUFs** (`feat/modernbert-community-gguf`,
  `77b829b`). Wider matrix coverage found gte-modernbert-base won't load; a
  loader-alias-only attempt produced GARBAGE (structural gate `emb_ln_out` cos
  0.58), tracing the true first divergence to the TOKENIZER (dispatch read only
  crispembed's own `tokenizer.ggml.type`, ignoring the standard
  `tokenizer.ggml.model="gpt2"` → WordPiece instead of BPE). Fixed the tokenizer
  FIRST, then the loader: (1) model-string-authoritative dispatch when the numeric
  type is absent; (2) BPE merges from the `tokenizer.ggml.merges` KV array; (3) a
  GPT-2 ByteLevel regex pre-tokenizer (arch-gated); (4) loader aliases
  (attn_norm/ffn_norm/output_norm + GeGLU-by-shape reroute of the fused `ffn_up`
  [H,2·inter]) + metadata (pre_ln, inverted dual RoPE theta, sliding-window→
  local/global) + exact-erf GeGLU. **Per-stage q8_0 vs HF: emb_ln_out cos=0.999928
  (gate PASS) + all 22 layers 0.9999+; f16 control cos=1.000000 at EVERY stage
  (graph exact, gap is quant); final CLS-pool cos q8_0=0.999602 / f16=0.999999.**
  Tokens match HF `[50281,25521,1533,50282]`. `tests/community_gguf_matrix.json`
  entry `gte-modernbert-base` (garbage-guard margin 0.51, was −0.089); full
  5-model matrix still PASS. Deep-dive in LEARNINGS.md.

## July 13, 2026 — DeepSeek-OCR-2 #4: converter-emitted stacked MoE experts (−1.3 GB resident)

Closed the last open DeepSeek-OCR-2 memory lever (`feat/ds-ocr2-stacked-experts`).
The MoE decoder shipped per-expert 2D weights that `stack_moe_experts()` rebuilt
into 3D `[in,out,n_exp]` tensors at load — so both copies (~1.3 GB) sat resident.

- **Converter** now emits `l.blk.{i}.ffn_{gate,up,down}_exps.weight` directly:
  `np.stack(experts, axis=0)` → ggml `ne=[in,out,n_exp]`, byte-identical to the
  runtime stack (expert `e` at `e*nb[2]`). **Loader** loads them straight into
  `gate_exps` (no copy, no stacking pass), builds per-expert views for the
  `DS_MOE_CPU` fallback (with `view->buffer` set to dodge the Metal device-pointer
  deref), and keeps a backward-compat path for legacy per-expert GGUFs. Quantizer
  already handled 3D experts (per-row; `down` `ne[0]=896` falls to Q4_0 exactly as
  the per-expert down did — not a regression).
- **Kaggle reconvert** (`chr1s4/crispembed-deepseek-ocr2-stacked-convert`)
  byte-validated the stacked slices vs the source safetensors (all checks) and
  uploaded f16 + q4_k to `cstr/deepseek-ocr2-crispembed-GGUF` as NEW `-stacked`
  files (non-clobbering — the rev-pinned regression GGUF untouched).
- **Local M1 Metal A/B (q4_k, back-to-back):** decoded output IDENTICAL ("The
  quick brown fox jumps over the lazy dog. 12345", cer 0.0) on all three loader
  paths (prestacked / DS_MOE_CPU views / legacy); **peak footprint 5.27 → 3.97 GB
  (−1.30 GB, −25%)**. (RSS is a misleading metric here — mmap page cache; footprint
  is the real number. See LEARNINGS.) Regression entry `deepseek-ocr2-stacked`
  added; registry default promoted to the stacked q4_k.
- **Ported the same optimization to `unlimited-ocr`** (`feat/uocr-stacked-experts`,
  `baidu/Unlimited-OCR` — the same DeepSeek-V2 MoE): verbatim converter+loader port,
  Kaggle-reconverted + byte-validated, HF `-stacked` files (rev `b11fef884fee`).
  M1 Metal q4_k A/B: **output byte-identical on all 3 loader paths; peak footprint
  4.32 → 3.11 GB (−1.21 GB, −28%)**. Registry promoted; first-ever regression entry
  `unlimited-ocr-stacked` added. (The v1 Kaggle run hung ~3h with no progress — the
  numpy expert accumulate/stack thrashed under multithreaded OpenBLAS; fixed by the
  dev-guide-mandated single-thread OMP/BLAS + unbuffered converter prefix.)
  `crispembed.cpp` BERT/NLLB MoE embedders already load pre-stacked 3D experts — no
  change needed. Those are the only three `ggml_mul_mat_id` paths.

## July 13, 2026 — Transcoda-59M zero-shot OMR engine (clean-room, byte-exact, persistent-KV decode)

Ported **Transcoda-59M** (`btrkeks/transcoda-59M-zeroshot-v1`) — full-page score
image → Humdrum `**kern`, OMR-NED SOTA on real historical scans — as the fourth
OMR engine (`src/transcoda_ocr.{h,cpp}`, arch `transcoda_ocr`). Architecture:
ConvNeXt-V2-Tiny encoder (GRN blocks, no LayerScale) + 2-layer projector + 2D
sinusoidal PE dual-memory bridge + 8-layer pre-LN RoPE cross-attention decoder,
untied LM head. 58.8 M params.

**Clean-room** (weights CC-BY-4.0, reference code AGPL): written from the paper
(arXiv 2605.10835) + HF config/data files + an activation oracle
(`tools/dump_transcoda_reference.py`, gitignored — running the model is fact-
gathering). First build hit all stages cos = 1.000000 (encoder + all 8 decoder
blocks + logits, CPU & Metal), argmax 191/191, native preprocessing bit-exact vs
the oracle. Greedy `**kern` decode is **byte-identical to the HF reference** (460
chars / 203 tokens) at both f32 and q8_0 after fixing four decode-side bugs (KV
view-stale, `/`-separator, per-occurrence repetition penalty, oracle 192-token
cap — see LEARNINGS). q8_0 (65 MB, 3.4×) needed a quantizer keep-guard for the
ConvNeXt conv2d kernels.

**Perf:** replaced the naive host-shuttled KV path with a persistent device-
resident KV cache (cross K/V computed once, self K/V written in-graph — the
got_ocr pattern), **2.4–4× faster decode, byte-identical** on Metal and CPU; the
host path stays behind `TRANSCODA_OCR_HOST_KV=1`.

Shipped: HF `cstr/transcoda-omr-GGUF` (f32 + q8_0 + CC-BY-4.0 card w/ attribution,
license verified), model registry, regression fixture (`page_transcoda.png` from
CC-BY-4.0 verovio-synth-omr, cer 0.000), README/omr.dart wiring. Deferred: beam-3
and grammar-constrained (`**kern` GBNF) decode.

---

## July 13, 2026 — Kaggle CUDA regression: Class-A/Gap-5 fixes CONFIRMED (clean run)

Ran the OCR-portfolio regression on Kaggle CUDA (T4/P100,
`tools/kaggle/ocr-portfolio-regression`) against current `main`. First run
ERRORed on **`No space left on device`** — `REGRESSION_WORK` pointed at
`/kaggle/working` (~20 GB), so a multi-GB multi-model pull ENOSPC'd. **Fixed the
kernel** to stage downloads under `/tmp` (~70 GB, `8f175cb`) and re-ran clean
(v9): 44 models, 14 FAIL, and the previously-ENOSPC'd models now PASS
(modernbert, mixtex, bidirlm-vision, clip-text, bert_ner, lfm2_colbert, tromr).

**Confirmation goal MET — the Class-A device-pointer + Gap-5 free-after-load
fixes flip FAIL→PASS on CUDA:** `deepseek-ocr2`, `dat`, `swinir`, `qwen2vl-3b`
(was a Gap-6 TIMEOUT), and **`lfm2_colbert`** (the CUDA multivec-corruption fix)
all PASS.

**The 14 FAILs are NOT regressions in the fixed engines** — triaged from the log:
- `glm-ocr` (cer 4.3), `internvl2-1b` (cer 5.4): **known Class-B** older-arch
  vision divergence (still needs Turing/Pascal to localize — as documented).
- `pcs`, `fireredpunc`, `fullstop-punc`: `FileNotFoundError:
  build/bin/test-punct-diff` — the crisp_punc test binary isn't built in the
  CrispEmbed CUDA config. **Test-harness gap, not an engine failure.**
- `layout-heron`: `diff harness died from signal 6` (SIGABRT **teardown** after
  results) — the Gap-5 harness-tolerance item (parse stages before the
  returncode<0 check).
- `granite-vision`: text OCR **PASSES** (cer 0.163 < 0.180); only 3 diff-harness
  stages read cos 0.95–0.97 on CUDA (projector) — output correct, threshold strict.
- `hat` (+ `pan`/`tbsrn`/`lilt`/`lfm2`): diff harness "no parseable stage lines"
  / long runtime — harness/format issues on CUDA, per-engine detail TBD.

**Follow-ups — harness fixes DONE (`be6ec54`):** `run_diff` now parses stages
before reacting to the exit code, so a teardown-crash-after-valid-stages is a
WARN not a FAIL (Gap-5); `run_check` SKIPs when its binary isn't built (fixes the
false `test-punct-diff` FAIL — crisp_punc isn't in the CrispEmbed CUDA kernel).
**v10: 14→9 FAILs** (pcs/fireredpunc/fullstop/gliner cleared).

**Diff-parser fix DONE (`2af57b1`) — 6 of the remaining 9 were FALSE FAILs.**
`lfm2`/`lilt`/`layout-heron`/`hat`/`pan`/`tbsrn` reported "no parseable stage
lines" because the parser missed their output formats: `lfm2`/`lilt` wrap
PASS/FAIL in **ANSI colour codes** (defeating the anchored regex); `hat`/`pan`/
`tbsrn` print `name   cos_min=…` (spaces, **no colon**); `lilt`/`layout` use
aligned **column tables**. Now strip ANSI first + cover the colon-less and 2
table formats — verified against the real `test-lfm2-diff` output (20 stages
parse, worst cos_min=0.999848, genuinely passing). **v11 confirmed: 46 models,
4 FAIL** (was 14) — hat/pan/tbsrn/lilt/lfm2 all PASS. Every false FAIL is gone;
the harness follow-ups are complete.

**RESOLVED — all FAILs closed (portfolio 14 → 0).** A diagnostic kernel
(`tools/kaggle/crispembed-cuda-diag`, Tesla P100 / Pascal sm_60) exercised each
under its env gates; none of the "Class-B" ones were real CUDA vision
divergences:
- **`glm-ocr` + `internvl2-1b` — a stdout banner, not vision garbage
  (`7998f3c`).** Both printed their load banner (`loading… Vision:… LLM:… KV
  cache… Ready`) via `printf` → **stdout**, and `run_one`'s `--ocr` text-match
  captures stdout, so `actual` = the banner (cer 4.3/5.4). Both OCR the fox
  **correctly** on CUDA *and* CPU; only the harness saw the banner. Routed all
  banners to stderr (matching `qwen2vl_ocr`).
- **`granite-vision` — text OCR PASSES;** the projector diff drift is
  cross-toolchain FP strictness (identical CUDA=CPU=scalar on P100), threshold
  already 0.95.
- **`layout-heron` — one genuine CUDA bug + one comparison artifact.** The
  SIGABRT was `fattn.cu:602` — Pascal (sm_60) has **no flash-attention kernel**;
  fixed by a manual attention fallback (`49cb38a`, `LAYOUT_DETECT_FLASH=1`
  restores flash). The subsequent `dec_0_cross_out` FAIL was **not** an inference
  bug: the 300 decoder queries are picked by a `partial_sort` over ~8400 near-tie
  encoder proposals, so a tiny backend FP delta in enc_output (cos 0.99999)
  reorders near-tie ranks and the index-aligned per-query cosine craters even
  though the cross_out *values* are correct (final boxes unaffected — score-sort
  + NMS). Fixed by comparing that stage **permutation-tolerantly** (best-cosine
  match; `d7f0480`). See LEARNINGS.md → "A parity stage downstream of a
  topk/argsort selection craters by query PERMUTATION."

**Bottom line:** the diagnostic-first approach (test on the box via env gates)
was essential — a blind "fix the Class-B vision divergence" would have chased a
non-existent bug. One real CUDA bug (Pascal flash-abort) + one stdout-banner
harness bug + one topk-permutation comparison artifact + cross-toolchain FP
strictness. Portfolio now **46 models, 0 FAIL**.

---

## July 13, 2026 — QKV-fusion probe (measured negative) + detector-postprocess audit

Three follow-on investigations run while the CUDA regression kernel built:

- **QKV fusion for got_ocr LLM decode — measured negative, reverted.** Probed a
  gated `GOT_OCR_QKV_FUSE` (graph-time `ggml_concat` of the q/k/v projection
  weights + one matmul + split). Result: **`ggml_concat` mishandles q4_k weights
  → garbage output** (1023-step runaway decode, "recognition failed"), and
  re-concatenating per step is **3× slower** (42.8 vs ~12.9 ms/step). A correct
  fusion needs manual **load-time q4_k row-block byte-stacking**; and by the
  memory-bound-decode analysis (T=1 mul_mv reads the weight, so 3 q4_k matmuls
  move the same bytes as one fused q4_k matmul) it saves only ~2 matmul launches
  per layer — ~4 % of the ~11 % host slice on a compute-bound decode. High
  effort, sub-5 % ceiling → deferred. Probe reverted (no code change). Note:
  got_ocr's **vision** tower already ships a fused `attn_qkv`; only the LLM
  decoder keeps separate q/k/v (that's where the GGUF stores them).
- **surya_det (the recommended doc detector) — clean by inspection.** Box
  extraction is O(Σ bbox_area) (bounded; no DBNet-style bbox×contour blowup) and
  the encoder is a bench-covered ggml graph. No hidden postprocess bug. (`--ocr-det`
  is DBNet-specific — `ocr_detect::load` rejects surya; surya runs via the
  orchestrator.)
- **cc_detect — clean.** Proper two-pass union-find connected components, O(w·h).
- **DBNet 3 s graph** (now the detector's dominant cost after the 28× postprocess
  fix) is the ResNet-18 + FPN + DB-head conv stack; the two head ConvTranspose2d
  deconvs (×4 upsample to full res) are the suspected cost, but detection is not
  the OCR-pipeline bottleneck (per-region TrOCR recognition dominates) and a
  deconv→sub-pixel-conv rewrite is a model-level change that alters output —
  deferred.

Net: item 1 (DBNet postprocess, 28×) was the real win; the detector-postprocess
family is otherwise clean, and decoder op-fusion is now confirmed marginal with
hard evidence (not just analysis).

---

## July 13, 2026 — DBNet detection: scanline box scoring (28× faster postprocess)

Investigated the "DBNet detector on Metal" item and found it was **misframed**.
The CPY abort was already fixed (`dequant_rows_f32` via get_rows), and detection
graph-compute is only **~3 s on CPU** (Metal `conv_transpose_2d` is still ~13×
slower, so CPU stays the correct default). Measuring the full detector exposed
the real bottleneck: **`extract_boxes` postprocess was ~43 s** — 15× the graph.

Root cause: `score_polygon` tested every bbox pixel against the **full** traced
contour (O(bbox_area × contour_len)), and `trace_contour` can emit a very long
contour (up to `w*h*2`) on a degenerate component, so the product exploded.
Rewrote `score_polygon` as a **scanline polygon fill**: each row's edge crossings
are computed once, then a pixel's inside/outside is an `upper_bound` over the
sorted crossings — even-odd-identical to the per-pixel ray-cast (inside iff an
odd number of crossings lie strictly right of x). **Byte-identical box output.**

- Same-binary A/B (dbnet-ic15-q4_k, forced CPU, 10-line page): postprocess
  **43326 → 1540 ms (~28×)**; total detection **46.4 → 4.9 s**. Boxes cmp-identical
  on the page (14) and fox (1). `OCR_DETECT_SCALAR_SCORE=1` restores the old path.
- Lesson (again): measure the dominant cost first — the "GPU-accelerate detection"
  premise chased a 3 s graph while a 43 s CPU postprocess dominated. (`74b8ac5`)
- **Full-pipeline impact (verified end-to-end):** on the same 14-region page the
  whole DBNet+TrOCR pipeline went **~46 s → 7.2 s** (detect 4.4 s [graph 3.0 +
  postproc 1.3] · batch-encode 2.5 s [14 ViT passes] · decode **0.3 s**). Note the
  decode is NOT the bottleneck here — detection-conv-graph + the ViT encoder are,
  both inherent compute. Confirmed no more algorithmic/O(n²) fruit in the OCR
  detect→recognize path; the remaining levers are all model/kernel-level.

---

## July 13, 2026 — SR/restoration engines → fused ggml graphs (complete)

Ported the super-resolution / restoration engines from per-conv mini-graphs (a
fresh graph init/alloc/compute/read-back for every conv) to fused ggml graphs.
All verified against the PyTorch reference via `test-<engine>-diff` and A/B'd
against the legacy path (identical output), env-gated per engine.

- **SAFMN** (`8594cee`): whole forward = ONE fused graph. **2.2× faster
  (6.1s→2.8s) AND more accurate (cos 1.000000 vs 0.994)** — F32 convs + exact
  `ggml_gelu_erf` (the tanh approx alone dropped cos to 0.947). Tiny/overhead-
  bound; Metal is a net loss (default CPU, `SAFMN_SR_METAL`/`SAFMN_SR_LEGACY`).
- **NAFNet** (`14a8393`) + **InstructIR** (`e1eb1dc`): fused per-block graph,
  cos ≥ 0.999998, output identical to legacy. Both are NAFNet-family =
  **compute-bound**, so fusion is perf-NEUTRAL (cleaner code, not faster).
  NAFNet defaults to Metal (~15%; `NAFNET_CPU`); InstructIR is CPU-only (GPU
  conv_2d hits a Metal f32×f16 mul_mv pipeline issue). Gates `*_LEGACY`.
- **Restormer** (`663f661`): was ALREADY fused — `rst_transformer_block_ggml`
  (MDTA transposed-attention + GDFN in one graph) is the default, `RESTORMER_
  SCALAR` the fallback. Only the stale "CPU-scalar" header was corrected.
- **scunet, swinir, tbsrn, hat, adair, dat**: already single-graph
  (`forward_expand=1`) — verified sensible (swinir 0.9984, dat 0.99999). No work.

**Finding (see LEARNINGS / memory):** the fusion win is entirely about
overhead-bound (tiny SAFMN → 2.2×) vs compute-bound (larger engines →
perf-neutral). Two recurring gotchas: erf-vs-tanh GELU, and conv weight-layout
scrambling (GGUF `[OC,IC,KH,KW]` bytes vs ggml's `[KW,KH,IC,OC]` — a plain
reshape scrambles them; copy bytes into the right layout).

## July 13, 2026 — got_ocr decode: redundant Q cont dropped (byte-identical; cont-removal doesn't generalize)

Tested whether math_ocr's ~30% decode cont-removal generalizes to the VLM
decoders (PLAN flagged qwen2vl/got/glm/internvl2/lightonocr). It does **not**,
for decoder-only engines whose KV comes from a cache. In got_ocr's cached decode
path, `Kfull`/`Vfull` are already fed to `ggml_flash_attn_ext` as non-cont cache
views, so only **Q** carried a removable `ggml_cont` (permute(0,2,1,3) already
gives flash-attn the row-contiguous input it needs). Dropped it, gated
`GOT_OCR_ATTN_CONT=1` for bisection (mirrors `MATH_OCR_ATTN_CONT`).

- **Byte-identical on Metal AND CPU** (`GOT_OCR_FORCE_CPU=1`), cont-off vs cont-on,
  on a one-line fox image and a 10-line / 117-step page (409-byte transcript,
  `cmp`-identical). Strict node-count cleanup (one fewer copy kernel per layer per
  step).
- **Latency within noise** (loaded box, loadavg ~19; decode_total medians
  ~identical). got_ocr decode is compute-bound (~89% GPU-execute), so removing
  Q's cont is a micro-gap — kept default cont-off because it's never-worse and
  matches the math_ocr convention, but it is not a perf headline. PLAN's op-count
  lever updated with this caveat so the generalization isn't re-chased. (`5011848`)

---

## July 13, 2026 — PLAN.md sorted: completed backlog archived here

PLAN.md had grown to ~3,400 lines, most of it DONE narrative interleaved with a
thin layer of still-open work. Sorted it down to ~700 lines (current architecture
+ genuinely-open/in-progress items + the llama.cpp support-matrix reference); the
completed material below moved out of PLAN. Most of it is already recorded in the
dated entries further down and in `LEARNINGS.md` — this entry is the index of what
was removed and preserves the PLAN-unique specifics. No code changed; full prior
PLAN.md text remains in git history.

**llama.cpp convergence backlog (C1–C6) — all shipped.**
- **C1 imatrix quantization** — `src/imatrix.{h,cpp}` (eval-callback collector,
  `CRISPEMBED_IMATRIX_OUT`), `crispembed-quantize --imatrix`, `tools/imatrix_ab.py`.
  IQ4_XS/IQ4_NL wired. Kaggle rollout re-quantized **all 38 dense embedders + 7
  rerankers + NER/GLiNER/ColBERT/sparse**; registry defaults repointed to each
  model's max-cosine flavor (decoder embedders → q4_k+imatrix, BERT/XLM-R encoders
  → iq4_xs+imatrix; f2llm-v2-0.6b + nomic-v1.5 kept q8_0; rerankers ship q8_0). Full
  A/B tables were in PLAN; the closure is HISTORY "July 2–3 2026" imatrix entries.
  Sub-closures: C1b rerankers (Kendall-τ; bge-reranker-base was shipped HEADLESS →
  reconverted; `tests/audit_gguf_heads.py` release gate), C1c fixed-label NER
  (span-F1 1.0), C1d GLiNER (opt-in sched for the collector), C1e ColBERT+sparse
  (splade-pp converter bug fixed). Bilingual EN+DE eval corpora (CC0) added;
  bilingual re-calibration measured not-worth-it (imatrix is language-agnostic).
- **C2 data-driven GGUF behavior flags** — pooling / causal-attention /
  add_bos_token / add_eos_token now read from GGUF metadata; verified byte-identical
  across WordPiece/SPM/BPE/LFM2 families.
- **C3 batched-encoder throughput** — `encode_tokens_packed` (block-diagonal seg
  mask) + `encode_tokens_4d` (rectangular per-item mask). Metal verdict: **PACKED is
  the batching mode (5–7× vs sequential, parity cos 1.0)**; 4D is the CPU tool
  (1.18–1.48×). Backend-conditional default (packed ON for GPU, OFF for CPU).
- **C4 cross-call prefix KV cache** — `decoder_encode_tokens_cached`
  (`dec_prefix_cache`), Qwen3 + Gemma3; CPU bit-equal, Metal cos ≥ 0.9999995,
  ≈2.07× compute-only. Default ON, `CRISPEMBED_DECODER_PREFIX_CACHE=0` opts out.
  Landmine: `ggml_cont` K and V before `set_output` (view-snapshot staleness).
- **C5 mtmd preprocessing** — `src/image_preprocess.{h,cpp}` (smart_resize +
  PIL-`a=-0.5` bicubic), wired into qwen2vl/bidirlm/mixtex. Bicubic-`a` A/B resolved
  by local measurement (HF uses PIL a=-0.5; a=-0.75 is strictly worse).
- **C6 flash-attn epilogue audit** — swept all 39 `ggml_flash_attn_ext` sites across
  22 engines; no surviving double-permute; codified as a reusable graph guard.
- **mmproj interop, both directions, 3 families** — export
  (`export-mmproj-llamacpp.py`) + import via a family-dispatch on
  `models/gguf_merge_core.py` (unified `merge-llamacpp-gguf.py`): Qwen2-VL,
  SmolVLM/Idefics3, InternVL2.5/3, each validated end-to-end. Rule: un-permute q/k
  is arch-dependent (llama yes, qwen2/NEOX no); map ViT FFN fc1/fc2 by output dim.
  Tests: `test_mmproj_interop.py` + `test_mmproj_smolvlm.py`.

**June-2026 optimization-TODO audit — fully closed.** The line-by-line review of
~57K lines / 60+ runtimes completed: P0 (SIMD `core/cpu_ops.h`, DequantCache, F16 KV
across all decoder engines, granite full-Metal graph path, pix2struct rewrite,
scunet heap hoist), P1 (flash-attn everywhere, scalar encoders → ggml graphs,
patch-embed → im2col+matmul, RoPE freq tables, batched-linear GEMM in SR attention,
batched region recognition), P2 (LFM2 sched+T-bucket, graph caching, gallocr reuse,
native GQA in flash-attn, BatchNorm fusion, mel OpenMP/SIMD), P3 (BPE min-heap,
WordPiece trie, alloc hoists, bilinear resize, beam search, morph_fast, SIMD
norms/softmax). Only open remnant: SR fused-single-graph (SAFMN pattern) — now in
PLAN.

**Per-backend performance passes — DONE:** lightonocr (2.09×), qwen2vl (+OCR
correctness 4-bug fix), deepseek_ocr2 (OCR correct; perf-sweep regression reverted;
MoE-compute is the only remaining lever, now in PLAN), got_ocr, glm_ocr (+5-bug OCR
fix), granite_vision (full Metal graph, 270→139 ms/tok), smoldocling, internvl2,
SR/denoise SIMD, embedding flash-attn. unlimited_ocr remains IN PROGRESS (open items
moved to PLAN).

**Implementation blueprints — DONE:** prefix-shared decoder-batch KV cache
(`decoder_encode_tokens_batch`), batched-decoder F16 mask + Gemma3 NaN clamp, and the
WASM build target (`build-wasm.sh` / `build-embed-wasm.sh`, 3 tiers incl. WebGPU;
GitHub Pages demo). Detail in the July 4–5 2026 WASM entries below.

**Runtime speedup roadmap (2026-07-11 sweep) — Tier-2 wins closed:** scunet Swin
MLP → SIMD GEMM (1.69×) + WMSA window-loop threading; gliner DeBERTa encoder rel-pos
dedup (1.28–1.71×, byte-identical); layout_detect Phase-2 `cpu_linear` → SIMD AXPY
(~1.26×) + backbone `conv_2d_direct` → im2col GEMM (~9.8× Phase-1, default flipped);
surya_det grouped-pointwise-conv graph-path crash fixed; safmn honor `n_threads`
(~2.3×); tps_locnet dequant hoist; debug-`fprintf` gating (layout/surya/ocr_detect).
Decode-step graph cache shipped for got_ocr/internvl2/glm_ocr/lightonocr/math_ocr
(remaining decoders + the ICB/op-count lever moved to PLAN). Negatives recorded (do
not re-chase): esrgan intra-op threading (slower), restormer double-variance (audit
was wrong), conv2d_cpu → im2col (marginal), got_ocr/glm_ocr conv swap (~4%).

**Regression-guardrail closure (2026-07):** SR/restoration (11) + esrgan/safmn + lilt
+ lfm2 + decoder_embed/vit_embed/clip_text/cnn_embed-face/tps_locnet/fireredpunc/pcs/
bidirlm-vision/bidirlm-text auto-guarded in `tests/regression/manifest.json`. Wave
regressions found by tracing: **layout** (double-permute after flash_attn, `6027b56`)
and **nafnet** (scrambled conv-kernel layout + residency). Disambiguated non-bugs:
gliner (dead reference, engine fine), lfm2/lfm2_colbert/bert_ner (dumper bugs).
lfm2_colbert CUDA multivec corruption fixed (rebuild graph after `sched_reserve`,
P100 cos 0.57→0.996). **pcs reached full ONNX parity** (Unigram Viterbi tokenizer +
5 more root causes); fullstop-punc got the same treatment. Open residuals (bert_ner
download-blocked ref, face-recognition unguarded) moved to PLAN.

**CUDA-backend gaps (Kaggle + local Ampere sm_86):** **Class-A device-pointer
weight-read SIGSEGVs fixed across 8 engines** (deepseek-ocr2/dat/tbsrn/unlimited/
math_ocr/smoldocling/parseq/tesseract — host-guard the zero-copy path, else
`ggml_backend_tensor_get`; commits 42ef0ea/28fb9b1); full `->data` census clean.
**Gap-5 free-after-load teardown** hardened (keep `wl_backend`, free after
`free_weights`). **Class-B** (glm/internvl2/qwen2vl-3b garbage on Turing/Pascal only)
remains open → PLAN.

**OCR correctness/stability (issue #25, 2026-06-30):** VLM repetition
(`argmax_no_repeat_ngram` n=3 in internvl2/qwen2vl/got_ocr/math_ocr), got-ocr2 graph
crashes, DBNet Metal CPY worked around (get_rows dequant + CPU-default), self-contained
CI artifacts, ggml v0.10.0 Metal residency + lfm2 sched teardown aborts fixed
(`GGML_METAL_NO_RESIDENCY` default + `core_util::clean_exit`). Open: DBNet full Metal
CPY path → PLAN. **GPU + quantization audit (2026-06-16):** ~28 engines full-GPU,
~10 GPU-safe, 0 CPU-only; all have `<ENGINE>_FORCE_CPU=1`.

**TrOCR recognizer investigation (2026-07-07):** WASM ≡ native token-for-token; GGUF
≈ HF; the trailing-repeat bug fixed (`6791af5`). Low quality is trocr-small's ceiling
on scene-text crops, not the port. Remaining accuracy/speed levers → PLAN.

**Next-gen + handwritten-math OCR ports — DONE:** PaddleOCR-VL 0.9B/1.6,
SmolDocling, Qari-OCR, TexTeller 3.0, Uni-MuMER-Qwen3-VL-2B, Uni-MuMER-Qwen2.5-VL-3B.
License rejections retained in PLAN's next-gen table (dots.ocr, MinerU2.5, Hunyuan).
**SMT (printed OMR) — DONE, shipped `cstr/smt-grandstaff-GGUF` at 96.3%** (per-stage
cos 1.0; the invert was the only bug — SMT-main preprocessing has no RandomInvert);
TrOMR + handwritten phase-2 remain in PLAN's OMR section.

**scan_cleanup / unpaper feature ports (2026-07) — all 6 evaluated:** despeckle
(heavy-speckle CER 0.580→0.032), blackfilter (8-CC labelling + 40%-page guard +
sharpness gate), 2-up page splitting, content-mask detection — all clean-room, MIT.
grayfilter/blurfilter deliberately skipped (subsumed by morphological-closing
whitening); deskew corner-fill already correct. Consensus deskew (Hough × DSS) +
per-params deskew across all image paths (`ce7f1c4`). Harness:
`tools/scan_cleanup_bench.py`.

**core/ refactoring:** `core/cpu_ops.h` + `core/vlm_attention.h` extracted (728+134
lines deduped, 185 unit tests). `core/vlm_decoder.h` deferred → PLAN.

---

## July 12, 2026 — P3 backlog sweep (every item triaged)

Worked the whole low-priority backlog to a clean end state — each item is now
DONE, WON'T-DO with a verified reason, or externally blocked. Real changes:

- **internvl2 diff-harness input guard** (`fix/internvl-diff-input-guard`). An
  earlier InternVL import looked broken (`vis_patch_embed cos=-0.936`) — the
  cause was **mine**: I dumped the HF reference on a real image while the harness
  feeds a synthetic gradient. Re-validated correctly (dump without `--image`):
  `vis_patch_embed cos=0.999999`, import **identical to the native converter** at
  every stage. Added a guard so it can't recur: `dump_internvl2_reference.py`
  stamps `diff.input_mode`, and `test_internvl2_diff` refuses a non-gradient
  reference. (A residual `vis_proj_output cos=-0.098` is a pre-existing
  InternViT-vs-HF projector gap present in the native path too, not the interop.)
  Corrected the LEARNINGS entry that had mis-labeled the −0.936 a "convention
  artifact," and pruned a stale "STRONG LEAD" red herring from the PLAN.
- **Reranker eval corpus** expanded 16→30 self-authored CC0 EN+DE graded groups
  (`RERANK_EVAL`); the Kendall-τ run stays Kaggle-only.
- **Bicubic `a` A/B resolved by local measurement** (no 4 GB model): HF vision
  processors resize via PIL (`a=−0.5`), which CrispEmbed already uses; `a=−0.75`
  is cos<0.00002 worse. Fixed the inaccurate kernel comment.

Resolved by analysis (no code, correct outcome): `<__media__>` marker
(mtmd-internal, no CrispEmbed entry point); LFM2 ShortConv→`ggml_ssm_conv`
(already Metal-covered via im2col+mul_mat, and ssm_conv is causal vs the
bidirectional embed conv); reverse export for SmolVLM/InternVL (no use case —
both already ship as llama.cpp GGUFs); CrispASR `gpu_backend_pref.h` sync
(already committed `9f2e68f7`, logically identical); bidirlm re-quant (cosmetic
+ Kaggle-only); esrgan tiles (measured slower). See PLAN.md status block.

---

## July 12, 2026 — mmproj interop: 3rd family (InternVL) + diff-harness validation

### Unified import CLI + README (`feat/mmproj-unified-cli`)
- `models/merge-llamacpp-gguf.py`: one entry point that auto-detects the family
  from the mmproj's `clip.projector_type` (qwen2vl_merger / idefics3 / internvl)
  and dispatches to the matching per-family merge — clean errors for unsupported
  or missing projectors. `tests/test_mmproj_dispatch.py` (routing + full
  end-to-end) in the smoke tier. Documented under README "Converting models →
  Importing a stock llama.cpp VL model" (the capability was previously
  undiscoverable — README didn't mention it).

### InternVL2.5/3 import (`feat/mmproj-internvl`)
- `models/merge-llamacpp-internvl-gguf.py`: import a stock llama.cpp InternVL2.5/3
  pair (arch=qwen2 LLM + `internvl` mmproj) into CrispEmbed's `internvl2` engine.
  **Validated end-to-end**: ggml-org/InternVL2_5-1B merges, loads, OCRs correctly
  on Metal, and the diff-harness intermediates match the native converter to 6
  decimals. Third distinct arch on the shared `gguf_merge_core.py` dispatch
  (after Qwen2-VL + SmolVLM).
- New transforms vs SmolVLM: **vision QKV re-fusion** (mmproj splits attn_q/k/v →
  loader wants fused `attn_qkv`; byte-concat, no permute — vision has no RoPE);
  **arch-conditional q/k un-permute** — arch=qwen2 uses NEOX RoPE so q/k copy
  VERBATIM (un-permuting gave garbage; this was THE bug). MLP connector
  (`mm.model.mlp.{0,1,3}`→`v.proj.{norm,fc1,fc2}`), layer-scale ls1/ls2, class
  token, dynamic-tiling metadata injected per InternVL2.5 defaults. ViT FFN
  fc1/fc2 mapped by output dim — here `ffn_up`=fc1, the INVERSE of SmolVLM.
- `tests/test_mmproj_internvl.py` (no download) + wired into the regression smoke
  tier. Folded the shared `llama_unpermute_qk_rows` into `gguf_merge_core.py`.

### Import-validation discipline (both SmolVLM + InternVL)
- Per the standing rule "test intermediates AND outputs, not just outputs":
  validated each import THREE ways — (1) ground-truth output vs `llama-mtmd-cli`
  on the same GGUF; (2) HF per-stage reference dump vs `build/test-*-diff`; (3)
  isolation — the native converter on the same HF model gives IDENTICAL cosines,
  proving import ≡ native. Caught that InternVL OCRs correctly while
  `vis_patch_embed cos=-0.936` (a pre-existing internvl2-harness convention
  artifact, present in the native path too — not the interop). See LEARNINGS.md.

---

## July 12, 2026 — C4 prefix cache, math_ocr decode fusion, two-way mmproj interop

### C4 — cross-call prefix KV cache for decoder embeddings (`feat/c4-cross-call-prefix-kv`)
- When consecutive `encode()` calls share an instruction prefix (Jina-v5 /
  Qwen3-Embedding "Instruct:…\nQuery:" prompts), compute the prefix once and
  reuse it. The decoder-embed path is a single-shot prefill (flash-attn over the
  whole sequence), so with causal attention the prefix tokens' per-layer
  post-rope K/V + final hidden are independent of any suffix.
- `dec_prefix_cache` (per-context): build the cache via a prefix-only graph, then
  a suffix-only graph whose queries attend to `[cached prefix K/V | fresh suffix
  K/V]` (rectangular flash-attn). Full/cold/miss path is the untouched
  `decoder_encode_tokens` (byte-identical). Bidirectional models ineligible;
  invalidated on LoRA swap. Default ON, `CRISPEMBED_DECODER_PREFIX_CACHE=0` opts out.
- Both graphs compute on a single-backend **gallocr** (not the sched): the sched
  aliases the 2·n_layer interior `set_output` K/V snapshots to one buffer. The
  injected per-layer inputs are marked `set_output` so gallocr keeps them distinct.
- **Landmine (cost most of the debug):** V was a `ggml_reshape_3d` VIEW —
  `set_output` on a view does NOT protect the source `v_proj` buffer, so the
  readback was stale garbage (K, a fresh rope output, was fine; `prefix_hidden`
  looked correct because flash read V in time). Fix: `ggml_cont` K and V before
  marking them output. (See LEARNINGS.)
- **Verified:** CPU bit-equal (cos 1.0, max_abs 0.0) cached-vs-full on octen-0.6b
  q8 (Qwen3) + harrier-270m q8 (Gemma3); Metal cos ≥ 0.9999995; no-prefix
  byte-identical to the pre-C4 binary. Speed: 40 long-prefix prompts 2.16→1.30s
  end-to-end, **≈2.07× compute-only** (octen q8 Metal). Test:
  `tests/test_prefix_cache.py`.

### math_ocr decode — drop redundant conts (~30% faster decode)
- Step-0 measurement first: decode-step graph = 355 nodes; encoder 200 ms vs
  decoder 44 ms (decode ~18% of compute). The step already uses flash_attn_ext,
  and the brief-flagged QKV concat is only ~1.3% of compute.
- The real overhead was `ggml_cont` after every permute in `g_mha_1q` —
  flash_attn only needs row-contiguous src (`nb0==type_size`), which
  `permute(0,2,1,3)` preserves. Removed 36 redundant copy-kernels/step →
  **355→319 nodes, decode 45.5→31.5 ms (~30%)**, transcript byte-identical on
  Metal AND CPU. `MATH_OCR_ATTN_CONT=1` restores.
- Negative result (gate-caught): the same conversion on the 578×578 *encoder*
  attention is byte-identical on Metal but DIVERGES on the CPU kernel — kept
  manual F32, documented inline.

### mmproj interop, BOTH directions (Qwen2-VL ↔ llama.cpp)
- **Export** (`models/export-mmproj-llamacpp.py`): CrispEmbed combined Qwen2-VL
  GGUF → a llama.cpp `mmproj-*.gguf`. Complete `clip.*` schema extracted
  empirically from a real reference (27 KV / 520 tensors, no guessing).
  Validated end-to-end: the exported mmproj + LLM run in `llama-mtmd-cli` and
  OCR fox.png correctly.
- **Import** (fixed `merge-llamacpp-qwen2vl-gguf.py`): a stock llama.cpp
  Qwen2-VL-2B now loads + OCRs correctly in `crispembed --ocr` on Metal AND CPU
  ("The quick brown fox jumps over the lazy dog. 12345", identical to
  `llama-mtmd-cli`). Four bugs fixed:
  1. Merge renamed tensors to names the loader can't read (`vis.blocks.*`) →
     SIGSEGV. Keep native `v.blk.*`/`blk.*` + concat the split temporal patch embed.
  2. ViT-FFN **fc1/fc2 role inversion** — llama.cpp's mmproj inverts
     `ffn_up`/`ffn_down` vs the projection direction (biases prove it); map fc1 by
     output dim, not name.
  3. Loader `v.post_ln` merger-norm + tied-`lm_head` fallbacks for native GGUFs.
  4. **The image was silently dropped** (real cause of "text not visible"):
     `qwen2vl.image_token_id` is absent from llama.cpp GGUFs, so the splice used
     default `0` while the prompt emitted `<|image_pad|>=151655` → never spliced.
     Fixed the default to 151655 + the merge now writes the token IDs.
  - Localized via an HF diff-harness + the **inject-embeds discriminator**
    (zeros/random/HF embeds → identical output = image ignored), which flipped a
    phantom vision hunt to the real LLM-splice bug in one test.
- Regression: the shipped Qwen2.5-VL-3B still detects correctly + OCRs fox.png;
  all mmproj changes are gated to the Qwen2-VL (non-SwiGLU / missing-metadata)
  path only.

### mmproj interop hardening + regression test (`feat/mmproj-interop-tests`)
- Added `tests/test_mmproj_interop.py`: a pure-Python, zero-download round-trip
  test that synthesizes tiny llama.cpp-shaped LLM + mmproj fixtures and drives
  the **real** merge + export scripts via subprocess, for both F16 and F32 patch
  dtypes. Guards all four silent 2026-07-12 reverse-interop bug classes
  (identity naming, vision special-token injection, temporal-patch concat,
  merge⇆export inverse) + a full 40-tensor byte-identical round-trip. Wired into
  the `regression.yml` smoke tier (no binary, no network).
- Writing the test immediately caught **two latent bugs** that had shipped:
  1. `export-mmproj-llamacpp.py` still read legacy `vis.*`/`proj.*` tensor names,
     which the merge script stopped producing when it switched to identity
     naming — so `export --in <real merged gguf>` found zero vision tensors. Its
     own `--self-test` never caught it (it round-tripped synthetic legacy names).
     Rewrote export to read native `v.blk.*`/`mm.*` names + invert the temporal
     patch concat (split back into two slices, dtype-preserving).
  2. The merge's patch concatenation hardcoded `np.float16`, silently corrupting
     F32 patch embeddings. Now views by the tensor's real element width
     (byte-exact for any unquantized dtype).
  See LEARNINGS.md "Two 'inverse' interop scripts drift silently…".

### mmproj interop generalized to a 2nd VL family — SmolVLM (`feat/mmproj-multiarch`)
- Extracted `models/gguf_merge_core.py`: the shared hand-rolled GGUF read/write
  core (byte-exact quantized copy). Ported the Qwen2-VL merge onto it (−300 dup
  lines, round-trip test proves byte-identical) — the "family-dispatch" base.
- Added `models/merge-llamacpp-smolvlm-gguf.py`: import a stock llama.cpp
  **SmolVLM (Idefics3)** pair (arch=llama LLM + idefics3 mmproj) into CrispEmbed's
  `smoldocling` engine. **Validated end-to-end**: ggml-org/SmolVLM-256M-Instruct
  merged + loaded + OCR'd `The quick brown fox…` correctly on Metal. Every map is
  grounded in the real inspected files + the native converter's target format,
  not guessed.
- Three transforms nailed (all in `tests/test_mmproj_smolvlm.py`, no download):
  1. **q/k un-permute** — llama.cpp permutes q/k for its interleaved RoPE; the
     CrispEmbed loader wants HF rotate_half layout. Without this the LLM produces
     fluent garbage. Byte-exact row-shuffle (works on Q8_0). *This was the bug.*
  2. SigLIP FFN fc1/fc2 name-inversion (map by output dim, as with Qwen2-VL ViT).
  3. 4-D Conv2d patch → 2-D flatten (pure C-order shape relabel, byte-identical).
  Tokenizer (gpt2 BPE) passes through as `tokenizer.ggml.*` (loader reads it as a
  fallback); `<image>`=49190 injected. Wired into the `regression.yml` smoke tier.
  See LEARNINGS.md "Importing a llama.cpp LLM: un-permute q/k…".

---

## July 10, 2026 — TrOCR decoder: persistent KV cache + no_repeat_ngram

### Persistent device-side KV cache (`perf/trocr-persistent-kv`)
- Replaced CPU-side `std::vector<float>` KV cache with persistent ggml tensors
  on the compute device (adopted from lightonocr.cpp pattern)
- Self K/V: `[D, max_seq, n_layers]` — written via `ggml_cpy` at `n_past` (O(1)/step)
- Cross K/V: `[D, n_enc, n_layers]` — uploaded once, read via `ggml_view`
- Eliminates O(n²) growing cache re-uploads + 1200 cross-attn re-uploads per region
- **Result: ~4.4s/region on CPU (down from ~19s/region) — 4x speedup**
- Verified: 61/61 regions on scan_page_pd.png (P&P scan), 3/3 on pp_clean.png

### WASM full pipeline end-to-end verified
- Rebuilt WASM with persistent KV cache fix
- **First successful full-pipeline WASM run on a real scanned page**
- 61 regions detected + recognized on scan_page_pd.png (606×1000) in 1186s
- Previously crashed with ggml hash table overflow / memory OOB

### no_repeat_ngram trigram blocking
- Ported `argmax_no_repeat_ngram` from qwen2vl_ocr.cpp/got_ocr.cpp
- Bans tokens that would complete an already-seen 3-gram
- Fixes TOOO→TOO, SUMMERER→SUMMER divergence vs HF

---

## July 6, 2026 — 12 OCR engines verified in the browser; model picker

Extended the WebGPU sweep to 12 engines — all produce correct text.
Standouts: TexTeller-3 (177 MB) 29.2 s -> 5.5 s on GPU (5.4x, best LaTeX
quality of the math engines); PP-FormulaNet-L 113 -> 43 s (2.6x); trocr
small handwritten verified against the NATIVE engine (the wasm-CPU leg,
not GPU, was the drifting one on an out-of-distribution input). The demo's
single-model tab gained a grouped preset picker (13 entries) that fills the
still-editable URL field — manual override preserved, harnesses untouched.

---

## July 5, 2026 (night) — decoder-on-CPU split for the WebGPU tier

MATH_OCR_DEC_CPU=1: decoder weights duplicated into a CPU buffer so the
sched runs autoregressive decode on CPU while the encoder stays on GPU.
Demo worker enables it for both webgpu tiers. TrOCR decode 216 -> 48 ms;
pipeline essentially a wash (164 -> 160 s) but region-text parity with CPU
improved. e2e 15/15 webgpu + 13/13 default.

---

## July 5, 2026 (evening) — WebGPU compat tier (Asyncify), WebKit verified, SW fix

`--webgpu-compat` Asyncify variant for JSPI-less browsers, auto-picked via
WebAssembly.Suspending detection, deployed under webgpu-compat/ (15/15 e2e
in Chromium via ?gpuCompat=1). WebKit engine verified end-to-end on the CPU
tier (ground-truth match) after scoping coi-sw to document/script/wasm
responses only — WebKit kills service workers mid-stream on large proxied
downloads. Playwright WebKit 26.5 ships JSPI, so real Safari may use the
JSPI GPU build directly. TrOCR phase profiling: GPU encoder 5.5x, GPU
decoder 5x SLOWER (48->231 ms) — decoder-split/batched-decode deferred
with data in PLAN.

---

## July 5, 2026 (later) — engine sweep: six OCR engines correct on WebGPU; OPFS cache

Per-engine browser sweep (engine-sweep.js): pix2tex 2.6x, trocr 4.0x on
WebGPU; parseq/hmer/bttr/tesseract correct (tiny models stay faster on
CPU). Fixed parseq-on-WebGPU garbage: raw-gallocr engine + flash_attn_ext,
which ggml-webgpu silently compiles out under Emscripten — manual attention
under __EMSCRIPTEN__ + metadata-pool bump. OPFS model cache added to the JS
wrapper (awaited write, persist(), clear link in the demo) — revisits load
models with zero network. README/PLAN wasm sections rewritten to match
reality. e2e 13/13 + 15/15 (webgpu).

---

## July 5, 2026 — WebGPU conv stack: full OCR graph on GPU, browser test-backend-ops

Five more WGSL kernels (IM2COL, POOL_2D, CONV_TRANSPOSE_2D, UPSCALE
nearest+bilinear, ARANGE) + the earlier LayerNorm, carried as
patches/ggml-webgpu-ops.patch and drafted for upstream
(CrispASR tools/upstream-prs/22). All validated by ggml's own
test-backend-ops compiled to wasm and EXECUTED in headless Chromium
(IM2COL 77/77, POOL_2D 128/128, UPSCALE 11/11, NORM 20/20, CT2D 3/3,
ARANGE 2/2) — a browser-CI capability upstream doesn't have. Demo's WebGPU
tier now also runs DBNet detection on GPU (OCR_DETECT_USE_GPU=1 in the
worker): detection 90 s -> 1.5 s (~60x), det+rec pipeline 291 s -> 164 s
(1.78x) with box parity. Root-caused the "0 detections" mystery to
ggml-webgpu silently no-op'ing unhandled ops (UPSCALE) on the sched-less
path — the patch adds a warning. Ecosystem survey (wllama/whisper.cpp/
transformers.js et al.) archived in LEARNINGS/memory; top follow-up:
OPFS model cache (wllama pattern, MIT).

---

## July 4, 2026 (night) — WebGPU LayerNorm kernel: ~2.8× total vs CPU

Local WGSL LayerNorm (GGML_OP_NORM) for ggml-webgpu, applied as
patches/ggml-webgpu-layernorm.patch by build-wasm.sh --webgpu — the ViT
encoder's 24+ per-pass LayerNorms no longer round-trip to CPU.
Same-conditions A/B: webgpu 2.46-3.11 s → 1.67-1.79 s (~1.4×), ~2.8× vs the
SIMD CPU build; output byte-identical to native GT in every run. IM2COL &
friends (DBNet conv stack on GPU) deferred — needs 4 kernels, upstream-scale
work.

---

## July 4, 2026 (evening) — WebGPU tier for the WASM demo (~2.2×, experimental)

ggml's WebGPU backend (emdawnwebgpu/Dawn, JSPI) now builds to WASM via
`./build-wasm.sh --webgpu`, deploys under `webgpu/` on Pages, and is offered
as an opt-in checkbox when the browser has `navigator.gpu`. pix2tex
recognition: ~3.0-3.5 s vs 6.4-7.7 s on the SIMD CPU build (M1, warm),
output byte-identical to native across repeat runs; unsupported ops
(LayerNorm, IM2COL) fall back to CPU inside the engine's scheduler — adding
those WGSL shaders upstream is the next perf step. Porting details (shader
embedder fix, JSPI_EXPORTS + async ccall wrapper, resizable-heap vs
writeBuffer, non-re-entrant encoder graph cache removed) in LEARNINGS.md.
Verified: browser e2e green for all three tiers (plain 13/13, threaded,
webgpu incl. GT byte-equality); native output unchanged.

---

## July 4, 2026 (later) — WASM demo: Web Worker + threads + the missing SIMD kernels

Follow-up to the #31 fix after user feedback ("2nd tab seems to hang"): the
pipeline WORKED but computed on the main thread — a frozen tab for minutes is
indistinguishable from a hang.

**Worker offload.** All inference moved to a Web Worker (`ocr-worker.js`);
the page stays responsive (e2e asserts a <1.5 s main-thread round-trip during
compute), with live engine progress (new per-region prints in ocr_pipeline)
and an elapsed-seconds ticker. Explicit Process button; image/model in any
order.

**SIMD was silently off.** Under emcmake, CMAKE_SYSTEM_PROCESSOR=x86 → ggml
"Unknown CPU architecture → generic implementations" → arch/wasm/quants.c
never compiled; every quantized matmul was scalar. Fix:
`-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm` in both wasm build scripts (~1.5-2×).

**Threads.** `build-wasm.sh --threads` → `build-wasm-threads/`, deployed
under `threaded/` on Pages; `coi-sw.js` (COOP/COEP service worker +
controllerchange one-shot reload) makes GH Pages crossOriginIsolated; the
page auto-picks the threaded build when isolated (default min(4, cores-1)
threads). Pipeline A/B on the scan strip: 4 threads ≈ 1.5–1.8× vs single.
Two emscripten-6 pthread-in-worker gotchas (see LEARNINGS): the factory
deadlocks if first called inside an active onmessage handler (instantiate at
worker top level), and pthread workers spawn from self.location.href
(mainScriptUrlOrBlob is gone) — ocr-worker.js doubles as a pthread shim
(`self.name === 'em-pthread'` → importScripts the module and yield).

---

## July 4, 2026 — WASM OCR actually works in browsers (#31): UAF fix, verified e2e, GH Pages demo

Issue #31's reporter said the WASM OCR "still doesn't seem to work" — and every
piece of it was indeed broken end-to-end, previously "verified" only by node
tests that special-cased the crashes (`passed++` around a known ViT abort).

**Root cause, engine:** `math_ocr` cached the encoder graph in ctx but built it
in a ggml context whose `mem_buffer` was a stack-local `std::vector` → freed
before compute; the CPU backend's mul_mat work buffer reused the block and the
quantize-activations write clobbered the cached tensor structs. Hard
`memory access out of bounds` in every browser (this was the whole "ViT models
exceed WASM limits" myth — pix2tex AND TrOCR), reproducible native segfault on
some inputs (dbnet+trocr on a 520×260 crop). Fix: ggml-owned metadata pools
(`mem_buffer = nullptr`) for both cached graphs (single + batch). Details in
LEARNINGS.md.

**Root cause, integration:** all three default model URLs in the demo pointed
at HF repos that never existed (cstr/pix2tex-GGUF etc. → 401); the JS wrapper
used `module.HEAPU8` without exporting heap views (`EXPORTED_RUNTIME_METHODS`),
so `recognize()` threw in every modern-emscripten browser build; serve.py
forced COEP unconditionally.

**Verification (new, in CI):** `tests/wasm-browser/e2e.test.js` — Playwright
headless-Chromium test that drives the real demo page: fetch model → UI load →
canvas RGBA → recognize. pix2tex output must EQUAL the native CLI ground truth
(`x = \frac { - b \pm \sqrt { b ^ { 2 } - 4 a c } } { 2 a }`) — it does,
byte-identical. Gated `WASM_E2E_PIPELINE=1` also runs DBNet+TrOCR on a real
scan crop (tests/regression/images/scan_strip.png): 8 regions, words match
native GT (MAMMAA/LIKE/SUMMER…/HEAVEN), ~142 s single-threaded. build-wasm.yml
now runs node smoke + wrapper + browser e2e on every push.

**Release/deploy:** release-wasm.yml never ran once — `on: release` never
fires for releases created by release.yml with GITHUB_TOKEN; now triggers on
the `v*` tag push (with a wait-for-release loop) and its stale 2 MiB size gate
(wasm is 2.2 MB) is aligned to 4 MiB. New deploy-pages.yml publishes the demo
to https://crispstrobe.github.io/CrispEmbed/ on every main push.

---

## July 3, 2026 — imatrix everywhere: GLiNER (sched), ColBERT, Sparse; splade converter bug fixed

Closed out the non-embedding imatrix classes. All in a worktree, cherry-picked to main.

**GLiNER (C1d).** GLiNER used `ggml_gallocr` + `ggml_backend_graph_compute` (no eval-callback), so the
collector couldn't attach. Added an opt-in `ggml_backend_sched` (built only when calibrating) + a
`gliner_cc` alloc/compute helper applied to all 4 compute sites, and flush in `gliner_ner_free` (this
context isn't freed via `crispembed_free`, and clean_exit skips atexit). gliner-deberta → iq4_xs
(span-F1 1.0); gliner-lfm → q8_0. Investigated gliner-lfm's q4_k span-F1 0.941: **not a bug** — a
uniform 2% score shift tips 3 detections scoring 0.50–0.51 under the 0.5 threshold (same LFM2 backbone
hits 0.9975 on lfm2-colbert). A cautionary tale in coarse binary metrics at n=6.

**ColBERT + Sparse (C1e).** Added `colbert` (per-token cosine) and `sparse` (sparse-vector cosine)
harness modes. lfm2-colbert → q4_k+im 0.9975; splade-pp → iq4_xs 0.996.

**splade-pp was broken — a general converter bug.** Its GGUFs shipped with only the encoder (no MLM
head), so `--sparse` failed. Root cause: `convert-bert-to-gguf.py` tried `AutoModelForTokenClassification`
before the MLM check, and HF **random-inits** a `classifier.weight` for SPLADE (config num_labels=2), so
it was mis-detected as a 2-label NER model and the real `cls.predictions.*` head was dropped. Fixed by
deciding the head from the **checkpoint files** (authoritative) rather than the random-init-prone loaded
model — a real classifier wins (reranker/NER), else a real MLM head means SPLADE, else embedder.
Reconverted with `--sparse` verified before upload; sparse restored. This would have silently broken any
SPLADE/MLM conversion.

Also identified the SOTA permissive EN+DE eval-corpora path (MMTEB / MIRACL / Tatoeba) for scaling the
small A/B sets — see PLAN.

---

## July 3, 2026 — device-pointer weight-read crash class fixed across 8 engines (local Ampere CUDA)

A local NVIDIA CUDA GPU (RTX A1000 Laptop, Ampere **sm_86**, 4 GB, CUDA 13.0)
became available, so Gap-5/Gap-6 CUDA faults could be reproduced locally instead of
via ~50-min Kaggle round-trips. This exposed a **backend-agnostic crash class**
distinct from the arch-specific vision garbage.

**The bug (Class A):** engines that dequantize/read a MODEL WEIGHT on the host by
dereferencing `t->data` directly (`memcpy(t->data)`, `(fp16*)t->data`,
`traits->to_float(t->data)`, `return (const float*)t->data`). On a weight resident
on a device-local backend (CUDA/Vulkan/SYCL/HIP) `t->data` is a DEVICE pointer, so
the host read **SIGSEGVs**. Safe on CPU and Metal (Apple unified memory is
host-visible) — which is exactly why these "worked on Metal/CPU, crashed only on
CUDA." Fix everywhere: keep the zero-copy fast path only for host-visible buffers
(`!t->buffer || ggml_backend_buffer_is_host(t->buffer)`), else read via
`ggml_backend_tensor_get`.

- **deepseek-ocr2** — the Gap-6 "FAIL". SIGSEGV in `precompute_rpe_tables` reading
  SAM `rel_pos`. Reproduced, root-caused, fixed, **runtime-verified on local CUDA**
  (character-perfect fox OCR). (42ef0ea)
- **dat / tbsrn** — SIGSEGV'd 3/3 on Ampere during load-time BatchNorm fusion (dat
  `to_f32` returned `t->data`; tbsrn BN lambda `memcpy`'d `t->data`). The earlier
  DAT F32-fusion `buf.assign(p,…)` correctness fix is what began dereferencing the
  device pointer. Fixed → **dat cos 0.999995, tbsrn 0.999362, exit 0** on CUDA. Also
  gave all three SR engines a free-after-load backend-lifetime fix (keep
  `ctx->wl_backend`, free after `free_weights`). (28fb9b1)
- **unlimited_ocr, math_ocr, smoldocling_ocr, parseq_ocr, tesseract_lstm** — same
  antipattern, fixed by inspection (compile-verified). (42ef0ea)

**Codebase audit complete:** full `->data` census (52 refs / 14 files) + ggml
host-accessor check — no Class-A instance remains. `granite_vision`, `instructir`/
nafnet/safmn, `decoder_embed` (CPU-fallback branch only), `imatrix` (host gguf ctx)
are all safe.

**Class B (arch-specific vision garbage) is NOT this bug:** glm-ocr, internvl2-1b,
qwen2vl-3b all produce CORRECT OCR on local Ampere sm_86. Their Kaggle `cer>4` /
TIMEOUT is an older-arch (Turing sm_75 / Pascal sm_60) vision-encoder numerical
divergence — still open, needs Kaggle hardware. (qwen2vl-3b's "TIMEOUT" = garbage
→ runaway generation to `max_tokens=2048`, not a hang/OOM.) glm's per-stage diff
"FAIL" is a stale-reference artifact (identical on CPU).

---

## July 3, 2026 — imatrix rerankers + 3 dense backfills; DeBERTa quant-read bug fixed

Extended imatrix coverage past the dense embedders. All in a worktree, cherry-picked to main.

**3 dense embedders backfilled.** granite-embedding-107m/278m-multilingual and gte-modernbert-base
now carry imatrix quants (all 38 dense embedders done); registry defaults repointed to their
max-cosine flavor (granite-278m q4_k+im 0.9960, granite-107m iq4_xs 0.9935, gte-modernbert iq4_xs
0.9892). Confirmed the ModernBERT encoder flows through the collector locally first.

**Rerankers (C1b).** Added a `rerank` MODE to the Kaggle harness: calibration runs the `--rerank`
cross-encoder path (the imatrix collector fires on it with zero code change), and the A/B metric
becomes mean **Kendall-tau** on the doc ranking vs full-precision gold, with mean|dscore| as
tiebreaker — cosine is meaningless for a (query,doc) scorer. Validated locally (ms-marco-L6: q4_k+im
halves dscore, iq4_xs+im τ=1.0) by exec'ing the harness's own functions before the run. All 7
rerankers quantized; defaults wired by the rule "smallest flavor with τ=1.0, else q8_0": jina-v2 →
q4_k+im, ms-marco-L6/L12 → iq4_xs; bge + mxbai → q8_0 (τ<1.0 at 4-bit). imatrix reliably improves
dscore across all 7. The eval set is small (n=5) so τ is coarse — flagged as future work.

**DeBERTa `rel_embd` quant-read bug (commit 73a016e).** The 2 mxbai rerankers failed — not an imatrix
issue: they're DeBERTa, and `rel_embd` (a 2-D disentangled-attention weight) ships Q8_0/Q4_K, but both
position-expansion paths read it as raw F32 → `offset+size > ggml_nbytes` abort. So mxbai-rerank-* and
gliner-deberta could not run on ANY quantized GGUF. My earlier "it runs" was a `head`-pipe rc artifact.
Fix: dequant-safe `core_cpu::to_f32`; verified mxbai reranks + collects imatrix. Same class as the
granite / pcs-q4k / MLM quant-read bugs.

**Fixed-label NER (C1c).** Added a `ner` harness MODE (micro span-F1 A/B vs full-precision gold);
bert-base-NER and xlmr-ner-hrl quantized, both → iq4_xs (span-F1 1.0). Their BERT-NER encoder is a
shared crispembed_context, so the collector fired on `--ner` unchanged. First needed a `bert_ner`
classifier-dequant fix (commit 85feaeb) — the Q8_0/Q4_K `ner.classifier.weight` was read as raw F32
(`unsupported type 8`), so both **failed to load on any quant** (a third quant-read-crash instance this
session, after DeBERTa rel_embd and the rerankers). GLiNER (gliner-deberta/lfm) stays uncovered: its
`ggml_gallocr` compute path has no eval-callback hook for the collector.

## July 3, 2026 — lfm2_colbert ColBERT multivec: CUDA-only graph-reuse corruption fixed + P100-verified

`crispembed_encode_multivec` produced garbage on CUDA only: `colbert_output` cos **0.571643**
(backbone `hidden` cos −0.702160 on a Tesla P100) while the identical q8_0 backbone passed 20/20 in
the dense-encode graph on the same device and scored 0.998 on CPU/Metal. Two earlier hypotheses were
disproven first (the `set_output`-on-live-intermediate theory, and a cont-copy of `cur` — both gave
byte-identical CUDA numbers; `ggml_set_output` can't change computed values). **Root cause:** the
ColBERT graph re-allocated the *same* `ggml_cgraph` it had just handed to
`ggml_backend_sched_reserve`; `ggml_backend_sched_reset` doesn't null `tensor->buffer`, so the
reserve pass's stale buffer/residency assignment was reused at `sched_alloc_graph` → mis-computed
backbone on CUDA (Metal tolerates it). The dense path (`lfm2_embed_encode_to`) already rebuilds a
fresh graph after reserve; the multivec path didn't. **Fix:** factor graph construction into a
lambda and rebuild after the bucket-change reserve, mirroring the dense path (`src/lfm2_embed.cpp`).

**Verified by an on-GPU A/B on the exact handover hardware** (Tesla P100, compute 6.0) — a Kaggle
kernel built github `main` (baseline) and the fix branch side by side against the same q8_0 GGUF +
HF-float32 reference: `main` cos **0.571643** FAIL (hidden −0.702160, reproducing the handover to 6
decimals) → fix cos **0.995885** PASS (hidden +0.922054). The wired 0.99 regression guardrail now
passes on CUDA. A codebase sweep confirmed the bug was **not systemic** — the other five
`sched_reserve` sites (dense lfm2 + three `crispembed.cpp` encoder paths) all rebuild a fresh graph
between reserve and alloc. See `LEARNINGS.md` for the transferable rule and the InternVL2 sibling
case (same reuse anti-pattern, opposite backend: Metal-crash vs CUDA-silent-corruption).

**Verification-harness gotcha:** the ref-gen A/B kernel had to force CrispEmbed's local `crisp_*`
fallback copies (`-DCRISP_PUNC_DIR=/nonexistent` …) — an adjacent CrispASR clone otherwise pulls in
a version-skewed `crisp_punc` (missing `core/gpu_backend_pref.h`, since vendored on main in
`8846a84`), which broke the CUDA build for reasons unrelated to the engine under test.

## July 3, 2026 — imatrix C1 closed: `clean_exit` vs `atexit` bug fixed, all embedders complete

Finished the imatrix rollout by fixing a subtle correctness bug in the collector and
completing the last big decoder embedders. All work in a separate worktree, cherry-picked
to `origin/main`.

**The bug.** The last three models re-quantized (qwen3-embed-4b, octen-8b, qwen3-embed-8b)
produced `-q4_k-imatrix.gguf` files whose A/B cosine was **bit-identical to the plain
baseline** — the imatrix wasn't being applied. Root cause was not the quantizer or the
model: the collector flushed its GGUF only from an `atexit` handler, but every one-shot
CrispEmbed binary exits via `core_util::clean_exit()` → `_exit()` (which skips ggml's Metal
static-dtor teardown *and* all atexit handlers). Calibration collected the stats correctly,
exited rc=0 with valid embeddings, and discarded them at exit → empty `.imatrix` → quantizer
fell back to unweighted. It looked model-specific only because `clean_exit` landed
mid-rollout: the first 27 embedders were calibrated before it (fine), the last 3 after
(empty). Reproduced locally on jina-v5-nano (small Qwen3 decoder) and instrumented the
eval-callback to confirm it fired and matched every weight — the flush was the only failure.

**The fix (commit 07439db).** Flush explicitly from `crispembed_free()` (runs before
`clean_exit`), guarded by `g_flushed` so atexit + explicit paths write at most once per
process. Also vendored `src/core/gpu_backend_pref.h` (a new CrispASR shared-core header)
to unblock the build. See `LEARNINGS.md → "The collector wrote nothing"`.

**Completion.** Re-ran the Kaggle harness (chr1s4, `FORCE=1`, branch with the fix) for the
three models; all now show strong imatrix deltas vs q8_0 gold: qwen3-embed-4b 0.9683→0.9881,
octen-8b 0.9746→0.9902, qwen3-embed-8b 0.9742→0.9934. octen-8b/qwen3-embed-4b's previously
mislabeled files are now genuine imatrix quants. Registry defaults verified **optimal for
all 30** — each resolves to its max-cosine A/B flavor (decoder embedders → q4_k+imatrix,
BERT/XLM-R encoders → iq4_xs+imatrix; f2llm-v2-0.6b + nomic-v1.5 kept at q8_0, both <0.91 at
4-bit). qwen3-embed-8b registry entry repointed + `-q4k`/`-iq4xs`/`-q8` aliases added.

## July 2, 2026 — Gap-4 embedding/face/tail-engine regression guardrails + 3 real bugs

Closed the June-wave audit's "Gap 4" (engines with no standing diff test) and, in the
process, surfaced three genuine shipped bugs. All parity work done in a separate worktree,
`tools/format.sh`'d, merged to main. See `LEARNINGS.md` for the transferable lessons.

**Guardrails added (5 engines, each vs an INDEPENDENT reference, wired in
`tests/regression/manifest.json`, verified end-to-end via `run_one.py`):**
- **tps_locnet** — `test-tps-parity` vs an independent numpy forward over the shipped
  BN-folded GGUF (`dump_tps_reference_from_gguf.py`; PaddleOCR `.pdparams` is geo-blocked on
  bcebos). cos **1.000000**. Aligned the harness to `cos_min=` for the run_one parser.
- **vit_embed** — new `test-vit-embed-diff` vs HF SigLIP `get_image_features`. cos **0.9915**
  (fixed the dumper's SigLIP path for transformers 4.57). Ref → `cstr/siglip-base-GGUF`.
- **face_detect (cnn_embed / SCRFD)** — new `test-face-diff` vs an independent
  **insightface-SCRFD** run over `det_10g.onnx` (`dump_face_reference.py`). Matches within
  **2.45 px** on a FLUX-generated synthetic face fixture. Ref → `cstr/scrfd-det-10g-GGUF`.
- **fireredpunc** — new generic `test-punct-diff` golden text-match (the punct C API exposes
  only restored text). Wired `run_check`.
- **decoder_embed** — new `test-decoder-embed-diff` vs HF Qwen3-Embedding-0.6B (last-token
  pool). cos **0.9993**; also confirmed on Kaggle CUDA. Ref → `cstr/qwen3-embed-0.6b-GGUF`.

**Bugs found (handovers written; not wired — a guard would be red until fixed):**
- **pcs** — the shipped `pcs-xlmr-base-q4_k.gguf` **crashes on every inference**: it reads
  Q4_K/Q4_0 FC-head weights via raw `ggml_backend_tensor_get` into F32 buffers. In sibling
  repo `CrispASR/crisp_punc`. Wave commit `4a498d1`.
- **clip_text** — embeddings only cos 0.79 vs HF: the CLIP BPE tokenizer never applies the
  `</w>` word-boundary convention (emits GPT-2-style tokens). Pre-existing, not a wave regression.
- **lfm2_colbert** — ColBERT multivec diverges on **CUDA** (cos 0.57 vs 0.998 CPU/Metal); the
  backbone `cur` is computed wrong under the ColBERT graph while the dense graph's identical
  backbone is fine. First fix hypothesis (set_output-on-live-intermediate) was **empirically
  disproven** and reverted — real cause is graph-structural, needs per-layer CUDA localization.

**Kaggle ref-gen kernel (`tools/kaggle/crispembed-ref-gen`) hardened:** added decoder_embed +
bidirlm entries; pinned `transformers==4.57.6` (the Kaggle image's build crashed BidirLM's Qwen2
tokenizer + drifted LiLT/DiT); fixed the verify heuristic that false-flagged lfm2's
`PASS: 20 FAIL: 0`. text_sr stays blocked (no checkpoint anywhere).

**Update (July 3): bidirlm-text CLOSED — it was a SHIPPED-GGUF converter bug.** The
`bidirlm-omni-2.5b*` GGUFs cratered the text tower to cos 0.047 (vision fine at 0.997). Not
pooling/mrope: a fresh re-export with the current `convert-decoder-embed-to-gguf.py` gives text
cos 1.000000 (f16) / 0.9992 (q8_0) and still passes vision (0.9966). Re-quantized + uploaded the
corrected `bidirlm-omni-2.5b-q8_0.gguf` + `bidirlm-text-ref.gguf` to `cstr/bidirlm-omni-2.5b-GGUF`;
wired `bidirlm-text`. Follow-up done: added a `--text-only` flag to `convert-decoder-embed-to-gguf.py`
(gates the audio/vision Phase-2/3 blocks; the current converter otherwise always writes the combined
GGUF — the old `-textonly` repo was a stale Phase-1 conversion). Regenerated + uploaded ALL variants
of BOTH repos from the fresh conversion: full-omni `{f16,q4_k,q5_k,q6_k}` and textonly
`{f16,q8_0,q4_k,q5_k,q6_k}` (textonly q8_0 text cos 0.9992). bidirlm is NOT an imatrix decoder (the
registry ships q8_0), so plain k-quants are correct. Also confirmed CLOSED this pass: clip_text
(`fa66a02` clip_style=true → cos 1.0, wired) and pcs (dequant FC-head weights → q4_k no longer
crashes, wired) — both fixed by other agents from my handovers.

## July 2, 2026 — ggml v0.10.0 GPU-teardown regressions fixed + bidirlm-vision parity harness

A ggml submodule bump to **v0.10.0 (`8be60f83`)** silently changed two runtime
contracts, crashing GPU runs that had worked days earlier. All fixed; neither was
CrispEmbed logic. See `LEARNINGS.md → "ggml v0.10.0 … GPU-teardown regressions"`.

**The regressions.**
- **Metal residency-set teardown abort** (`ggml-metal-device.m:612`). v0.10.0 added
  a Metal GPU keep-alive cache (180 s, background heartbeat) with a hard teardown
  assert `[rsets->data count]==0` in `ggml_metal_device_free`. Any leaked GPU buffer
  at process exit aborts (SIGABRT / exit 134) **after** results print — corrupting
  exit codes and making passing one-shot CLI / `test-*-diff` runs report false
  "signal 6" failures (and breaking `run_one.py` pass/fail).
- **Scheduler CPU-fallback assert** (`ggml-backend.cpp:1736`). `ggml_backend_sched_new`
  now requires the last backend to be CPU; `lfm2_embed` built a Metal-only sched and
  aborted at load, masking lfm2 entirely.
- **CUDA hits the same leak** as SIGSEGV/SIGABRT (swinir/dat/tbsrn, gliner/
  lfm2_colbert/layout-heron/lfm2_embed) — no `NO_RESIDENCY` switch there.

**Fixes.**
- Library constructor in `core/gguf_loader.cpp` sets `GGML_METAL_NO_RESIDENCY` by
  default (opt back in with `CRISPEMBED_METAL_RESIDENCY=1` for long-lived hosts,
  which free via `crispembed_free` and are leak-clean). Restores pre-bump behavior.
- `lfm2_embed` appends a CPU fallback backend (fireredpunc issue-#68 pattern) — lfm2
  now runs on Metal, **test-lfm2-diff per-layer cos ≥ 0.9999**.
- **`core_util::clean_exit(rc)`** (`src/core/clean_exit.h`): flush + `std::_Exit`,
  skipping the static-dtor GPU-device teardown backend-agnostically. Applied to the
  CLI + all 90 `tests/*.cpp` mains; long-lived hosts keep `crispembed_free`. Fixes
  **both Metal and CUDA** teardown crashes, preserves pass/fail exit codes.
- **CI guard** (`tools/check_test_clean_exit.sh` + a `build.yml` job) fails if any
  test/CLI main bypasses `clean_exit`.
- Server + all bindings (Python/Rust/Dart load `libcrispembed` as a shared lib, so
  the constructor runs at load) get the safe default automatically — no changes.

**bidirlm-vision parity + regression harness.** The cached bidirlm q4_k GGUF was a
stale **text-only** export (0 vision tensors); re-downloaded the full HF q4_k (315
`visual.*` tensors, +427 MB). Confirmed the tower is correct — live HF parity
(`test_bidirlm_vision.py`) gives **image_embeds cos 0.997 (q8_0)**, deepstack
0.9998/0.9938; q4_k's 0.97 is the quant floor on massive-activation deepstack dims.
Added the standard `-ref.gguf` guard (`tools/dump_bidirlm_vision_reference.py` +
`test-bidirlm-vision-diff` + manifest entry; ref on `cstr/bidirlm-omni-2.5b-GGUF`)
so bidirlm vision now has the same CI-wired parity guard as the other engines —
`run_one --name bidirlm-vision` PASSes. Also fixed `crispembed.image.preprocess_image`
(`return_tensors="pt"` fallback for pt-only custom processors).

## July 2, 2026 — imatrix quantization (C1): 31 embedders re-quantized, registry defaults switched to best flavor

Started from a **llama.cpp parity audit** (which of our architectures upstream now
supports, and what to borrow — recorded in `PLAN.md` + `LEARNINGS.md → "llama.cpp
implementation reference"`). Top convergence item was **C1: importance-matrix
(imatrix) quantization** — the highest-leverage fix for our q4_k accuracy floor,
and offline-only (no graph risk).

**Implementation.**
- `src/imatrix.{h,cpp}` — an eval-callback collector gated by `CRISPEMBED_IMATRIX_OUT`.
  On every `ggml_mul_mat` whose src0 is a named model weight it accumulates the
  per-column sum-of-squares of the activation (src1), keyed by the GGUF weight name,
  merges with any prior file, and flushes a GGUF imatrix at exit. Wired into the
  encoder + decoder + lfm2 embedding schedulers; zero overhead when unset.
- `crispembed-quantize --imatrix <file>` feeds per-tensor importance to
  `ggml_quantize_chunk`. Added **IQ4_XS + IQ4_NL** types (IQ4_XS→IQ4_NL→Q4_0 fallback
  for non-256-aligned rows).
- Local A/B harness `tools/imatrix_ab.py`; Kaggle batch harness
  `tools/kaggle/crispembed-imatrix-quant/` (per-model → batch → idempotent-skip →
  big-base path), full kh regime (heartbeat, dataset token, ccache), CPU build.

**Rollout — 31 embedders** now carry imatrix quants (q4_k+imatrix, iq4_xs) with
`-imatrix-ab.txt` A/B summaries, uploaded under DISTINCT names (baselines never
overwritten). imatrix always lifts 4-bit; **IQ4_XS+imatrix wins on the XLM-R/BERT
encoders** (smaller AND higher cos), **q4_k+imatrix on the Qwen3/LFM2 decoder
embedders**. Examples (cos vs full-precision gold): jina-v5-small q4_k 0.979→0.990;
bge-m3 iq4_xs 0.981; nomic-v1.5 iq4_xs 0.837→0.905. GTE `NewModel` and nomic-v2-MoE
both worked. The 4B/8B decoder embedders (octen/qwen3-embed) use a **big-base path**:
calibrate + A/B-gold on the q8_0 (fits Kaggle's ~13 GB RAM), quantize from the f32
base (streaming), stage in `/tmp`.

**Model registry** (`model_mgr.cpp`) — every covered model's auto-download default
now resolves to its A/B-winning imatrix flavor, with `-q4k`(imatrix)/`-iq4xs`/`-q8`
aliases; several bad old defaults fixed (encoders were serving 1–2 GB full-precision
`.gguf`; e5-large was 2.2 GB F32). f2llm-v2-0.6b + nomic-embed-text-v1.5 keep q8_0
(4-bit too lossy).

**Quantizer bug found + fixed.** embeddinggemma-300m produced an unloadable GGUF —
PROVEN (by diffing vs the working reference q8_0) to be `crispembed-quantize`
quantizing the SentenceTransformer Dense/Matryoshka heads (`dense.0`/`dense.1`) to
q8_0 where the loader needs F32. Fix: a `dense.*` keep-F32 guard; re-quantized output
loads + embeds cleanly. Benefits any ST-Dense model. No models were ever actually
broken — one quantizer bug (fixed), two poor-at-4bit models (defaulted to q8_0), and
an early harness auto-detect bug that picked jina-v5's same-size LoRA task variant
(fixed: prefer exact base name + exclude task variants).

## July 2, 2026 — C3 batched-encoder throughput (packed + 4D), ModernBERT validated E2E, EmbeddingGemma verified

**C3 — batched embedding throughput (llama.cpp-parity item).** The encoder batch path
was disabled (looped single-encodes; the previous fused path padded but never masked
padding). Shipped two opt-in paths for absolute-position encoders (BERT/XLM-R/MiniLM/
BGE/E5 — no MPNet rel-bias / DeBERTa rel-embd / RoPE):
- **Packed block-diagonal** (`CRISPEMBED_ENCODER_PACKED=1`): B sequences packed end-to-end
  into one graph with an F16 block-diagonal `seg_mask` → `flash_attn_ext` (the
  `bidirlm_vision` pattern), positions restart per segment. Bit-parity (cos ≥ 0.9999) but
  attention is **O(T_total²)** (the mask still computes masked cells) → backend/size
  dependent (uncapped packing was a 3.7× loss); greedy token-budget grouping caps it;
  kept opt-in.
- **Rectangular 4D per-item mask** (`CRISPEMBED_ENCODER_4D=1`): sequences kept as separate
  4D items `[hd,T,nh,B]` + per-item `pad_mask [T,T,1,B]` (−inf on padded keys) →
  attention **O(B·T²)**. Length-sort + chunk (`CRISPEMBED_ENCODER_4D_GROUP`, default 32).
  Parity cos **1.0 / 0.9999697**, **consistently faster than sequential AND packed**
  (1.18×–1.48×). The real throughput fix; opt-in pending a real-Metal A/B (this box is
  CPU-only, `GGML_METAL=OFF`). `tests/test_encoder_batch.py`.

**ModernBERT (gte-modernbert-base) validated end-to-end** — structurally supported but never
parity-checked, and broke three ways; now cos **0.999999** (short) / **0.999998** (113-tok
doc) vs HF, **0.99976** q8_0. (1) *Local-path converter bug*: BPE-tokenizer, CLS-pooling and
Unigram-score detection all called `hf_hub_download(repo_id=args.model)`, which throws on a
local path and was silently caught → fell back to WordPiece + mean pooling (cos 0.46). Fixed
with `_resolve_file()` at all three sites (convert with `--crisp`; ollama mode never runs BPE
detection). (2) *Missing sliding-window local attention*: only the RoPE θ alternated
global/local — the local layers' ±`local_attention`/2 window mask was absent, so they attended
globally and long docs diverged (113-tok 0.9826 → 0.999998). Added a per-layer `swa_mask`;
converter emits `bert.local_attention`; A/B lever `CRISPEMBED_ENCODER_NO_SWA=1`. Guards:
`test_modernbert_parity.py` + a compiled `test-modernbert-diff` wired into the regression
manifest (q8_0-vs-f32 0.9919, floor 0.99; SWA-off craters cos to −0.87). GGUFs + ref →
`cstr/gte-modernbert-base-GGUF`; registry entry added.

**Bug fixed en route:** `crispembed_encode_tokens_raw` (+ a sibling raw path) branched only
SPM/WordPiece — **missing the BPE case** → BPE encoders (ModernBERT) were mis-tokenized via
WordPiece in the raw API (113 → 103 tokens). Added the `use_bpe` branch.

**EmbeddingGemma-300m** — the two-Dense (768→3072→768) + mean-pool + Matryoshka pipeline
verified correct (~0.997 vs HF). The residual is **not precision** (identical at f16 and f32):
it's a small Gemma3-backbone discrepancy amplified by the non-orthogonal Dense bottleneck
(the Dense/pooling code and weights match HF exactly). Registry pooling label corrected to
mean-pool (was "last-token").

**CI hygiene:** fixed the two durably-red gates from concurrent work — **Lint** (clang-format
`model_mgr.cpp` + `test_clip_tokenizer_parity.cpp`; whole tree now clean) and **OCR-regression**
(`test_driver_smoke.py` wrongly required `sample`/`expected_text` on `diff_only` and `run_check`
entries that legitimately have neither).

## July 2, 2026 — Metal residency-abort swept across all conv-front-end engines (9 fixed)

Generalized the nafnet/restormer residency finding into a full audit and found the
same crash in **7 more** engines: esrgan, safmn (SR) and bttr, hmer, posformer,
mixtex, ppformulanet (math-OCR). All load weights on `init_best` (Metal/CUDA) but run
every graph on a CPU `enc_sched`, so referencing the GPU-buffer weight leaves aborts
graph alloc on Metal (`pre-allocated tensor … buffer (MTL0) that cannot run`) and
segfaults on CUDA. Since these engines do no GPU compute, loading weights on CPU is
behavior-preserving — fixed all 9 (nafnet, restormer, esrgan, safmn, bttr, hmer,
posformer, mixtex, ppformulanet). Verified on the default Metal build: SR diffs pass
(esrgan/safmn cos 0.987); math-OCR reads a rendered quadratic formula correctly
(bttr/ppformulanet exact `\frac{-b\pm\sqrt{b^2-4ac}}{2a}`, posformer near-exact, hmer
structurally right). The rest of the family is Metal-safe (swinir/dat/hat/pan/tbsrn/
adair/scunet preload conv weights; instructir CPU weights; text_sr fully scalar).
Key lesson: **audit conv→ggml engines on the default GPU backend** — a FORCE_CPU diff
is blind to this whole class, which is why it shipped unseen (the math-OCR engines
have no regression coverage at all). Separately noted: mixtex_ocr now runs but has a
pre-existing decode-degeneration bug (unrelated to residency) — tracked for follow-up.

## July 2, 2026 — nafnet denoise fixed (conv→ggml scramble + residency); restormer's Metal abort closed

Closed the last two conv→ggml-wave regressions. **nafnet** was the coverage gap
(no diff harness, only reachable via `--denoise`): added `test-nafnet-diff` + a
`diff_only` regression entry (ref uploaded to `cstr/nafnet-sidd-GGUF/nafnet-ref.gguf`)
+ a `NAFNET_SCALAR=1` A/B gate. The A/B proved the engine, not the dumper: scalar
conv path cos **0.999998** vs ref, ggml path **0.538**. Fixed three sub-bugs in
`conv2d_ggml` — (1) the kernel-layout scramble (hand-rolled converter writes numpy
`[OC,IC,KH,KW]`; the old `permute(3,2,1,0)` physically reordered the bytes instead
of reinterpreting them as ggml `[KW,KH,IC,OC]`; 1×1 convs also hit a second wrong
branch via `ggml_n_dims()==2` collapse), (2) depthwise kernels must be F16 to match
`ggml_conv_2d_dw`'s hardcoded-F16 im2col, (3) a Metal/CUDA residency abort (weights
on init_best referenced from the CPU conv sched). Result: ggml==scalar==ref, cos
**0.999998 on Metal AND CPU**; `--ocr --denoise` reads the fox line end-to-end.

While auditing whether sibling runtimes shared the bug, found **restormer still
aborted on Metal** — its earlier "CPU==Metal fixed" was CPU-only. The layout fix was
real, but a separate residency bug (weights on the freed init_best backend,
referenced from the CPU conv sched) aborted at `patch_embed` on Metal and segfaulted
on CUDA; it only ever passed under `RESTORMER_FORCE_CPU=1`. Fixed by loading
restormer's weights on CPU (all its compute is on the CPU enc_sched anyway) —
`test-restormer-diff` now passes on Metal (cos 0.999997, was an abort). All other SR
engines (swinir/hat/pan/tbsrn/dat/adair/scunet preload kernels onto enc_backend;
instructir/safmn/esrgan via official GGUFWriter or CPU weights) verified Metal-safe.
See `LEARNINGS.md → "nafnet_denoise — RESOLVED"` + the restormer Fix-notes correction.

## July 2, 2026 — DeepSeek-OCR-2: perf-sweep regression fixed (restore c58913c), + mandatory A/B rule

deepseek_ocr2 OCR produced garbled output (`章的 flix Bailly …` / `&# &#`
repetition) on **both** Metal and CPU (byte-identical, deterministic) — even on
the recovered character-perfect Jun-19 q4_k. Ruled out: my edits, Metal (CPU
matches), the q4_k data, ggml (SHA unchanged), the converter (unchanged). A
git-bisect over `src/deepseek_ocr2.cpp` (`38e3801..e803e9f`) pinned it to the
**Jun-20 "perf sweep"**, which introduced MULTIPLE regressions with **no env gate
and no A/B test**: `c75b95d` swapped the Qwen2 vision-encoder's manual masked GQA
attention for `ggml_flash_attn_ext` (mishandles the custom bidirectional mask →
garbled vision); the flash_attn-LLM / persistent-decode commits added a decode
repetition-degeneration (HF `infer()` uses `no_repeat_ngram_size=20`; the greedy
decoder had none). The last-fully-good commit is **`c58913c`** (Jun-19) — *after*
the ~15× Metal vision-graph speedups yet *before* the regressions.

**Fix (Option A): restore `deepseek_ocr2.cpp` to `c58913c`** — reverts the
regressing perf commits while keeping the Metal vision speedups. Verified on M4
Metal at q4_k (Jun-19 model, rev `a465ab6cf4b5`): fox.png → "The quick brown fox
jumps over the lazy dog. 12345"; a 6-line document page → verbatim. Recovered the
character-perfect q4_k from HF commit history (`resolve/<rev>/…`) after the HF
f16/q4_k were clobbered by a bad 04:00 reconvert. Added a `deepseek-ocr2`
regression-manifest entry (rev-pinned) — the model had zero regression coverage,
which is why the broken default shipped unseen. The reverted perf paths must be
re-added one-at-a-time behind env gates, each A/B-tested vs decoded output before
flipping the default — codified as a new mandatory rule in
`crispasr-crispembed-dev.md` (dev guide) and `LEARNINGS.md`. Meta-lesson: a "perf"
change isn't done when it's fast — only when its decoded output equals a trusted
reference; `expected_text: null` == never validated.

## July 2, 2026 — Granite-Vision OCR: missing-tokenizer packaging bug fixed + GGUFs re-uploaded; release infra ported

Granite-Vision 3.3-2B OCR emitted raw token IDs (`<322><322>…`) on both backends
because the uploaded GGUFs carried **no tokenizer** (`tokenizer=MISSING (0 tokens)`)
— a converter/packaging bug, not a ggml regression (per-stage diff vs the ref was
healthy). Fix folds the BPE tokenizer + late-added scalars into
`models/convert-granite-vision-to-gguf.py` (new `array<string>` KV writer +
`load_tokenizer()` writing `tokenizer.tokens`/`tokenizer.merges` +
`attention_multiplier`/`rms_eps`/`bos`/`eos`) so a fresh convert is complete, and
makes `patch-granite-gguf-tokenizer.py` idempotent. All three published GGUFs
(q4_k/q8_0/f16) in `cstr/granite-vision-crispembed-GGUF` were re-patched and
re-uploaded via Xet (~84 MB new data each — the rest deduped). Verified end-to-end:
banner now `tokenizer=embedded (49156 tokens)`; `--ocr fox.png` returns readable
text on **CPU and Metal** (q8_0 exact match `The quick brown fox jumps over the
lazy dog. 12345`). Baked that `expected_text` into the regression manifest.
See `LEARNINGS.md → "granite-vision — … packaging bug"` (RESOLVED note).

Also ported CrispASR's release tooling: `scripts/bump-version.sh` +
`scripts/sync-version.py` (retargeted to CrispEmbed's crates/packages) so a single
`scripts/bump-version.sh X.Y.Z` writes VERSION, propagates it to
Cargo.toml/pyproject.toml/pubspec.yaml, commits, and tags — fixing the long-standing
drift (VERSION 0.7.0 vs Cargo 0.4.0 vs pyproject 0.3.2 vs tags at v0.12.0).

## July 2, 2026 — PaddleOCR-VL: SIGSEGV fixed, OCR working end-to-end (CPU + Metal)

`paddleocr-vl-0.9b` crashed with `EXC_BAD_ACCESS` in `_platform_memmove` during
the forward pass on **both** backends (exit 139, zero output) — it had never
been validated end-to-end. The handover
(`handover-prompts/paddleocr-vl-sigsegv-fix.md`) and the audit's LEARNINGS entry
both blamed the "8:1 GQA broadcast" hazard from `fbae7ba`; **both were wrong**.
A debug/`-O0` build turned the opaque `memmove` SIGSEGV into an exact
`ggml_reshape` assert, and reading the real tensor dims out of the GGUF gave the
answer immediately.

**Result**: fox.png → `The quick brown fox jumps over the lazy dog.` on **q8_0,
CPU and Metal**, stopping cleanly on `</s>`. qwen2.5-vl-3b (the primary user of
the shared `qwen2vl_ocr` engine) is unaffected — no regression.

**Two independent, unrelated bugs (+ tokenizer fallout):**
1. **The crash — ERNIE-4.5 uses `head_dim=128` while `hidden/heads = 1024/16 =
   64`.** The engine assumed `head_dim = D/n_heads` everywhere, so the Q/K/V
   reshapes (`attn_q.weight` is `[1024, 2048]`) and the post-attention
   reshape-to-`D` overran the tensors → SIGSEGV in Release, reshape assert in
   debug. Corroborated by the mRoPE sections `[16,24,24]` summing to 64 =
   head_dim/2. Fixed by adding `llm_hparams.head_dim` (from a config key or
   derived from `q_w->ne[1]/n_heads`) and reshaping attention output to
   `q_dim = head_dim*n_heads`, not `D`. No-op for Qwen (head_dim == D/n_heads).
2. **Empty/garbage output — SentencePiece vocab loaded as GPT-2 BPE.** ERNIE's
   vocab uses `▁` for spaces + `<0xXX>` byte tokens, but the OCR tokenizer loaded
   every GGUF as byte-level BPE, silently dropping all prompt whitespace → the
   model saw `OCR:Assistant:` and emitted `</s>` first. The chat tokens were also
   hardcoded to Qwen's `<|im_*|>` (151644/5), out of range for the 103424-row
   ERNIE embed table (a second `get_rows` assert). Fixed by detecting
   PaddleOCR-VL, emitting the real ERNIE template
   `<|begin_of_sentence|>User: <image>OCR:\nAssistant: ` (trailing space is
   load-bearing), stopping on `</s>`=2 (per `generation_config.json`, **not**
   `<|end_of_sentence|>`=100272), auto-detecting the `▁` vocab → SPM +
   add_dummy_prefix, and decoding with a `▁`→space / `<0xXX>`→byte SPM decoder.

`expected_text` for paddleocr-vl is now baked into the regression manifest.
Remaining: generate/upload `paddleocr-vl-ref.gguf` to enable
`test-paddleocr-vl-diff` (needs the HF model + upload).

## July 2, 2026 — Qwen2.5-VL OCR: hallucinated description fixed (4 bugs), both backends

`qwen2.5-vl-3b` (`qwen2vl_ocr` engine) fabricated a description ("mathematical
symbols, Greek letters α/β/γ, summations, integrals…") instead of reading the
image, identically on Metal and CPU. The `expected_text: null` in the regression
manifest was the tell: this was a **never-validated path**, not the
scalar→ggml-wave regression the handover/LEARNINGS suspected (and `fbae7ba` never
touched qwen2vl — handover suspect #1 was a red herring). Four independent
Qwen2.5-VL-specific bugs, all in `src/qwen2vl_ocr.cpp`; after the fix both
backends read `The quick brown fox jumps over the lazy dog. 12345` (cer≈0) at q4_k:

1. **Vision 2D RoPE built in raster order.** The preprocessor always emits patches
   in `(h//m,w//m,m,m)` merge-block order and HF's `rot_pos_emb` permutes the
   position ids the same way, but `compute_vision_rope`'s `merge_order` arg keyed
   off `is_qwen2_vl` → false for the RMSNorm 2.5 variant → every patch rotated with
   a neighbour's position, scrambling spatial structure. Dominant bug (fixing it
   alone flips pure hallucination into reading real words). Gate on
   `deepstack_indexes.empty()`.
2. **Merger grouped the wrong patches.** The CPU spatial merge chose consecutive-vs
   -raster grouping off `is_qwen2_vl`, sending Qwen2.5-VL through a raster gather
   that mis-groups merge-block-ordered data (the deepstack extract already assumed
   consecutive — the tell). Same gate.
3. **Windowed attention was unimplemented.** `window_size` (112) and
   `fullatt_block_indexes` ([7,15,23,31]) were loaded but never used — every ViT
   block did full attention. Implemented as an equivalent in-place additive mask
   (0 within a window, -inf across) via `soft_max_ext` on non-fullatt blocks; full
   blocks keep `flash_attn_ext`. No physical reorder/reverse-permute needed —
   window attention only restricts the *set* a patch attends to, which is
   storage-order independent. Opt-out `QWEN2VL_OCR_NO_WINDOW=1`.
4. **No OCR prompt for arch `qwen2vl`.** Only `qwen3vl` got the transcription
   prompt; `qwen2vl` fell back to "Describe this image." → verbose prose (fails a
   bare-text CER match). Applied the OCR prompt to both archs.

Baked `expected_text` into the regression manifest (was null). Per-stage HF ref
not regenerated (needs the ~7 GB model + torch dumper) — verified end-to-end
transcript on both backends. Commit `86d0830`. Deep-dive: `LEARNINGS.md` →
"qwen2vl-3b hallucinated OCR — RESOLVED". Meta-lesson (again): `expected_text:
null` == never validated; verify handover root-cause claims independently.

**Follow-up (same day) — the `deepstack_indexes.empty()` gate regressed Qwen3-VL;
fixed by making it unconditional.** `86d0830` gated rope order + merger grouping
on `deepstack_indexes.empty()`, assuming Qwen3-VL was `is_qwen2_vl=false`. It's
actually **`is_qwen2_vl=true`** (LayerNorm ViT) *with* deepstack, so it had been
using the correct merge-block/consecutive path via the old `is_qwen2_vl` gate —
the new gate flipped it to raster → garbage OCR (`qwen3-vl-2b` → `T11123456789…`).
`patchify_qwen_layout` emits merge-block order for *every* family member, so both
rope order and merger grouping are now **unconditional** (no gate). Verified
`qwen3-vl-2b` and `qwen2.5-vl-3b` both read the fox line on CPU and Metal.
Lesson: `is_qwen2_vl` is a ViT-norm flag (LayerNorm vs RMSNorm), not a family
selector — don't repurpose it (or a proxy) for preprocessing-order decisions.
## July 2, 2026 — restormer: denoise working (ggml conv-weight layout + real MDTA)

`restormer-denoise-f16` emitted blocky rainbow garbage (mean 147 / std 120 vs a
clean ~242) on **both** Metal and CPU. The prior handover's "CORRECTED root
cause" (convs are fine, bug only in the block graph) was itself wrong. Two
independent bugs, both fixed and validated against a PyTorch ground-truth value
and an end-to-end denoise test.

1. **Conv-weight layout scrambled for EVERY conv — the primary garbage source.**
   The GGUF converter writes conv weights raw as numpy `(OC,IC,KH,KW)` C-order
   and the loader keeps `ne` unreversed, so the correct `ggml_conv_2d` kernel is a
   **plain reshape of the contiguous bytes to ggml `[KW,KH,IC,OC]`** — no permute,
   no transpose, no shuffle. The load-time pre-permute (oc-fastest shuffle) and the
   `rst_prep_w` / `rst_conv2d_ggml` 2D-reshape heuristics all mis-laid-out the
   kernels. Proof: PyTorch `patch_embed[0,0,0]` = **0.645721**; old ggml gave
   0.161163, fixed ggml gives 0.645721. Deleted the pre-permute; both conv sites
   now reshape the raw buffer directly. Note: `RESTORMER_SCALAR=1` was **not** a
   clean reference — `rst_forward_tile` runs the U-Net convs through
   `rst_conv2d_ggml` in both modes, so the scalar path was garbage too (168.9).

2. **ggml MDTA block graph was a fake single-head attention.** It used `ggml_norm`
   as a stand-in for L2-normalize, ran one full `C×C` attention (no per-head split
   — wrong for the 2/4/8-head levels), and dropped the learned per-head
   `temperature`. Rewrote to match the scalar reference: reshape
   `[HW, d_k, n_heads]`, `rms_norm` over spatial (= L2normalize·√HW, folded into a
   `temperature/HW` scale), per-head batched `mul_mat`, softmax over the key axis.
   Also fixed `rst_ln2d_ggml`: the denoise model is BiasFree (`has_bias=0`), so
   `ggml_norm`'s mean-subtraction was wrong — now `x/sqrt(var+eps)·w`, no centering.

**Result**: gray σ=25 noise mean|err| 19.84 → **2.15** (~90% removed); CPU==Metal
to 0; ggml path == scalar path (identical image); full 800×200 fox now clean
(243.1/51.3, was 147/120). Commit `d54b304` (merged `67bbbb6`). Handover:
`handover-prompts/restormer-ggml-conv-weight-permute-fix.md`.

## July 1, 2026 — GLM-OCR: garbage OCR fixed (5 bugs + q8_0), verified vs real model

`glm-ocr` (zai-org/GLM-OCR, 0.9B) produced garbage OCR on every backend. The
prior handover (`handover-prompts/glm-ocr-vision-rope-fix.md`) blamed a single
missing vision RoPE — **wrong on two counts**: it read `glm4v` modeling code
(GLM-OCR is the distinct `glm_ocr` / `glm_ocr_vision` arch, only in transformers
`main`), and its "confirmed" reference dumps were the **stale no-rope** ones, so
"matches ref without rope" validated nothing. RoPE was needed, but was 1 of **5**
independent bugs. Ground truth came from running the **real model** (transformers
`main` + the ~1.8 GB checkpoint) and diffing every stage.

**Result**: fox.png → `The quick brown fox jumps over the lazy dog. 12345` on
**f16, q8_0, and q4_k, CPU and Metal**, matching the real model token-for-token.

**5 bugs fixed** (all verified against the real transformers-`main` model):
1. **Missing vision 2D RoPE** — Qwen2-VL-style, `dim=head_dim/2`, θ=10000, per-patch
   `[h·f, w·f]` freqs, `emb=cat(rot,rot)`, NEOX split-half. Raster patch order +
   per-patch `(row,col)` is equivalent to HF merge-window order under full
   attention. On by default; `GLM_OCR_VISION_ROPE=0` disables. Q/K match at 0.99999.
2. **Wrong merger structure** — `GlmOcrVisionPatchMerger` is
   `proj → LayerNorm(1e-5) → GELU(erf) → down(silu(gate)·up)` (no trailing norm);
   code had `proj → SwiGLU → LayerNorm`. This made image embeds uncorrelated
   (cos ≈ 0); after the fix they match the real `vis.merger` at mean cos 0.99.
3. **Fixed 336² instead of dynamic resolution** — processor is
   `Glm46VImageProcessor` (Qwen2-VL smart-resize, min/max pixels, dims ×28), not a
   fixed square. Squashing fox.png (800×200, 4:1) destroyed it. Added smart-resize
   + a variable grid flowing through patchify → rope → merger → prompt image-token
   count → LLM image mRoPE. fox.png → 812×196, grid 14×58 = ref `image_grid_thw`.
4. **LLM image mRoPE positions** — matched `get_rope_index`/`get_vision_position_ids`:
   image patch `(row,col)` → `temporal=start, h=start+row, w=start+col`; text resumes
   at `start+max(gh,gw)`; decode continues from the *compressed* position
   (`ctx.mrope_next_pos`), since image tokens compress positions.
5. **Prompt / EOS / decode** — correct template
   `[gMASK]<sop><|user|>\n<image>Text Recognition:<|assistant|>\n` (old prompt
   dropped `[gMASK]` + instruction, added a spurious `<|system|>`); stop on both
   eos ids `[59246, 59253]`; GPT-2 byte-level decode (was emitting `Ġ`/`Ċ`).

**q8_0 and q4_k** (both backends): dequantize weights **before** reshaping. The downsample
weight was reshaped to leading dim 2 before `ggml_cast` to F32, splitting q8_0's
32-element blocks → CPU garbage + the Metal `GGML_ASSERT(ne00 % blck)` abort. Same
class as got-ocr `11c2bc7`. q8_0 weights themselves were fine (not corrupt).

**Sink-token / diff-gate finding**: this ViT has massive outlier activations
(`max_abs`→~1900). On the synthetic-gradient diff image a few "sink" tokens' cos
collapses in C++ (ggml) but not in numpy — verified it's **not** weight precision
(numpy f16-weights stays 0.9999) nor compute precision (f32-vs-f64 stays 0.9999),
but ggml-vs-BLAS **reduction order** on catastrophic-cancellation tokens. It's a
test-image artifact: on real images C++ vs the real model is median 1.000 (OCR
exact). So the per-token `cos_min` diff gate can't hold for glm-ocr — the diff
block was removed from the regression (opt-in per model; affects only glm-ocr) and
`expected_text` guards correctness.

**Files**: `src/glm_ocr.{cpp,h}`, `tools/dump_glm_ocr_reference.py` (added rope +
corrected merger), `tests/regression/manifest.json` (expected_text; diff removed),
`tests/test_glm_ocr_{diff,image}.cpp` (new `encode_vision(H,W)` signature). Reference
`glm-ocr-ref-full.gguf` regenerated (rope + merger) and re-uploaded to
`cstr/glm-ocr-crispembed-GGUF`. Commits `cb681e0`, `908c667`, `4f3a392`, `0e914a3`.
Deep-dive: `LEARNINGS.md` → "glm-ocr: five real bugs". Meta-lesson: independently
reproduce a handover's root-cause claim before building on it — and know your
reference (`get_image_features` ≠ the merger hook; stale gguf refs ≠ the model).

---

## June 23, 2026 — Unlimited-OCR port (Baidu, SAM + CLIP + DeepSeek-V2 MoE)

Ported `baidu/Unlimited-OCR` (MIT, 3.3B params) as a new OCR engine.

**Architecture**: SAM ViT-B (12L, 768d) → CLIP-L/14 (24L, 1024d, receives SAM
features as patch embeddings — dual-encoder "DeepLIP") → fusion concat(CLIP[:,1:],
SAM.flatten) → Linear(2048,1280) → DeepSeek-V2 MoE decoder (12L, 1280d, 64
routed experts top-6, 2 shared, layer 0 dense).

**Files added**:
- `src/unlimited_ocr.{h,cpp}` — 2358-line C++ engine
- `models/convert-unlimited-ocr-to-gguf.py` — GGUF converter (2710 tensors)
- `tools/dump_unlimited_ocr_reference.py` — reference dumper (45 stages)
- `tools/kaggle/unlimited-ocr-parity/` — Kaggle parity kernel
- `tools/kaggle/unlimited-ocr-gpu-test/` — Kaggle GPU kernel

**Integration**: CMakeLists, `ocr_orchestrator.{h,cpp}` (enum + dispatch + free),
`crispembed.cpp` (`map_engine` case 13), `main.cpp` (CLI `--ocr-engine unlimited_ocr`),
`model_mgr.cpp` (model registry), `quantize.cpp` (`c.*` vision guard for CLIP→Q8_0).

**4 bugs found and fixed**:
1. **`ggml_cont()` before `flash_attn_ext`** — `flash_attn_ext` handles
   non-contiguous (permuted) tensors via strides. Adding `ggml_cont()` creates a
   contiguous copy with a different data layout than `flash_attn_ext` expects.
   Fix: pass permuted tensors directly (matches `vit_embed.cpp` pattern).
2. **Bilinear resize** — PIL `ImageOps.pad` uses BICUBIC (Catmull-Rom, a=-0.5).
   The C++ used bilinear interpolation. Every image pixel differed.
3. **Missing clamp [0,1]** — Catmull-Rom produces overshoots at sharp edges (text
   boundaries). PIL clips to [0,255] internally. Without clamping, cos_min=0.991
   at patch_embed. With: cos_min=0.9999.
4. **Wrong BPE token** — `core_bpe::tokenize_simple` produces `ĠOCR`=126041
   (with space prefix) instead of `OCR`=119316 for the instruction "\nFree OCR.".
   Wrong token caused LLM to hallucinate "Freeware" instead of performing OCR.
   Hardcoded correct IDs pending core_bpe fix.

**Parity (F16, Kaggle 30GB RAM)**:
- SAM all 12 layers: cos ≥ 0.999 PASS
- CLIP layer 0-1: cos ≥ 0.999 PASS
- clip_output: cos=0.997, vision_features: cos=0.998
- Model outputs structured OCR with bounding boxes (text quality needs more work)

**GGUF models**: `cstr/unlimited-ocr-crispembed-GGUF` (F16 6.4GB, Q8_0 3.4GB, Q4_K 2.1GB)

---

## June 21, 2026 — SR roster: full verification + conv→ggml sweep (scunet/tbsrn/dat/hat/adair)

Verified every non-blocked SR/restoration engine against an independent reference
and ported the conv path to ggml where it pays off. Each port is benchmarked and
gated by the result — default-on where it wins, opt-in where it's a wash/slowdown.

- **scunet** (`1b66701`) — conv + ConvTranspose2d → `ggml_conv_2d` / `_p0` on a CPU
  sched, **~6.7×/tile**, all stages cos=1.0. Gotcha: scunet stores conv kernels in
  ggml-native order (no ne-reversal, unlike pan/swinir).
- **tbsrn** (`0e30df2`) — 6 conv sites → ggml; verified vs a new self-consistent
  ref (`dump_tbsrn_reference_from_gguf.py`, reverses the converter rename +
  un-transposes Linear weights), output cos 0.999362. Attention-bound → modest.
- **dat** (`c70af4c` fix, `be79546` perf) — built a *genuine* ref (real PyTorch
  DAT-light on gguf-reconstructed weights) and found+fixed a real bug: **Conv+BN
  fusion silently skipped on F32 models** (`to_f32` returns `t->data`, leaves buf
  empty → fusion guard never fired → BN dropped). Output cos 0.9906 → **0.999995**.
  conv→ggml done but **gated opt-in** (`DAT_SR_GGML_CONV=1`) — net slowdown on this
  attention-bound engine (per-conv graph overhead > conv speedup).
- **hat** (`4d5cdc4`) — 6 top-level convs → ggml, **~1.3×/tile** (upsample/conv_last
  run at 4× resolution so convs matter); window/OCAB attention + CAB convs stay
  scalar. Output cos 0.999965 vs the validated hat-ref.
- **adair** (`a7bd61f` verify, perf follow-up) — verified correct via a genuine
  real-AdaIR ref (upstream `c-yn/AdaIR`, weights reconstructed from
  `adair-5d-f32.gguf`, all 587 params load), output cos **0.999379**. conv→ggml
  done, **~5.2×/tile** (15441 → 2951 ms on 64²): all conv sites (U-Net
  down/up/reduce/output + the MDTA/GDFN/cross-attn/FreModule convs threaded
  through the block helpers) → `ggml_conv_2d` / `_dw` on a CPU sched, kernel
  cache keyed by the dequantized weight POINTER (drop-in for the pointer-passed
  `conv2d`), F16-cast in-graph; the 2D FFT (AFLB) + attention softmax stay
  scalar. Default ON, opt out `ADAIR_SCALAR=1`. cos 0.999385 ggml — no regression.

Refs uploaded to HF (`cstr/text-super-resolution-gguf`): swinir, tbsrn, dat, adair.
Methodology note in LEARNINGS: genuine ground truth (real model run), not a
self-consistent ref derived from the engine, was required to catch the dat bug.
`text_sr` remains permanently blocked (no public model).

## June 21, 2026 — SwinIR-light: shifted-window mask sign bug (output −0.91 → ~1.0)

`swinir_sr.cpp`'s shifted (odd-index) Swin blocks rolled the feature map with the
wrong sign in `cyclic_shift`, so the forward shift was `roll(+ws/2)` while the
precomputed `attn_mask` (and the numpy reference) assume `roll(-ws/2)`. Forward
and reverse shifts cancelled, so the round trip looked fine — but the wrap-around
(edge) windows got the mask for the opposite convention, mixing token regions
that should be blocked. The error localised at image edges in the shifted blocks
and compounded through the four RSTBs (rstb_3 max_abs 147, engine ≈ 2× ref at
edges). Fix: forward `+ws/2`, reverse `−ws/2`. All stages now cos ≥ 0.99997,
output (float) cos 0.999996.

The reported "−0.91 anti-correlated output" was a separate red herring:
`test_swinir_diff.cpp` used `crispembed_diff`'s worst-per-row cosine with row
size = `shape.back()` = 3, i.e. 3 horizontally-adjacent pixels of a uint8-clamped
CHW image vs the raw-float ref — one near-zero edge triple tanks it. The test now
gates on the image-level (global + per-channel) cosine, and the reference dumper
(`tools/dump_swinir_reference.py`) now uses exact erf-GELU to match `nn.GELU()`.
Self-consistent gguf-fed ref generator saved as
`tools/dump_swinir_reference_from_gguf.py`. Conv→ggml port still TODO.

## June 21, 2026 — PAN super-resolution: scalar conv loops → ggml_conv_2d graph

`pan_sr.cpp`'s per-tile forward, previously hand-rolled scalar convolution
nested loops, now runs as a single `ggml_conv_2d` graph on the `enc_sched`
backend — the same pattern restormer/esrgan/safmn/nafnet already use. The
16× SCPA trunk (conv1_a/k1/conv1_b/PAConv/conv3 + residual), trunk_conv skip,
two nearest-2× upsample stages with pixel attention, and the bilinear input
skip all map directly to ggml ops: `ggml_conv_2d`, `ggml_leaky_relu(0.2)`,
`ggml_sigmoid`, `ggml_mul`, `ggml_concat` (channel dim), `ggml_upscale(NEAREST)`,
and `ggml_interpolate(BILINEAR)` (default half-pixel == torch `align_corners=False`).

Two gotchas worth recording:

- **Transposed conv ne.** The PAN GGUF stores conv weights in PyTorch axis
  order `[OC,IC,KH,KW]` (the converter does a plain `astype`, no permute), but
  the *data* is KW-innermost. `ggml_conv_2d` wants `ne=[KW,KH,IC,OC]` over those
  exact bytes, so the graph-weight prep **reverses the four ne axes** while
  copying the raw dequantized buffer unchanged. Feeding the native ne tripped
  `ggml_im2col: OW>0` (it read OC=40 as the kernel width). `ggml_n_dims` can't
  identify conv kernels (1×1 weights report 2 dims), so the prep keys off the
  `.weight` name suffix and always treats them as 4D.
- **Reference input quantization.** The diff harness feeds the engine a
  uint8-quantized input (`round(x*255)/255`); the torch reference must snap its
  input to the same 1/255 grid or a ±1/255 perturbation amplifies through the
  4× network to ~0.4 max-abs and one image row drops to cos 0.9959. With the
  matched input, graph and scalar both hit **cos_min=0.999997** vs the
  self-consistent torch reference (`tools/dump_pan_reference_from_gguf.py`),
  the residual being pure uint8 output rounding (max-abs 1.96e-3).

`test-pan-diff` is the gate; `PAN_SR_SCALAR=1` keeps the scalar path for A/B.
Reference uploaded to `cstr/text-super-resolution-gguf/pan-ref.gguf`. (`913b4f5`)

---

## June 21, 2026 — Granite Vision OCR: full Metal graph path (vision + LLM) now works

The whole Granite-Vision 3.3-2B OCR pipeline now runs on the Metal GPU **by
default** and returns the correct text in **~22 s** (vision ~3 s, 784-tok prefill
~12 s, decode ~5 s) vs the scalar path's ~100 s vision + ~8 min prefill. The
June-20 handover (`handover-prompts/granite-vision-graph-fix.md`) attributed both
the broken ViT and the broken LLM Metal graph to one "ggml-alloc in-place
buffer-reuse defect" — that was wrong on both counts (the input tensor was intact
after compute; no NaN from alloc). Re-verifying each claim independently found two
distinct, real bugs:

- **ViT graph (`gv_run_vit_graph`)** — `ggml_reshape_2d` applied to the **Q8_0
  `vis.ffn.down` weight** whose reshaped `ne[0]=4304` is not a multiple of the
  32-element Q8_0 block → mis-strided dequant → garbage from layer 0 (the
  "explosion" that looked like an alloc smash). Fix: dequantize quantized FFN
  weights to F32 before the reshape. The square Q8_0 attention weights (used raw)
  and the F16 `up` weight were always fine. Per-layer parity with the scalar ViT
  (cos 0.9996–0.99987; late layers track the scalar's 0.96 eps ref-artifact).
  (`a5b527f`)
- **LLM graph (`gv_run_llm_body`)** — correct for text at any length and on the
  ggml-CPU backend, but cascaded to NaN from layer 8 in the real OCR prefill on
  Metal. Localized via per-layer max-abs dumps: the residual carries a **massive
  activation** (~1.1e4 — outlier dims amplified ×12 by `embedding_multiplier` on
  the spliced image features). Apple's batched matmul `kernel_mul_mm_*` (T>8
  prefill; T=1 decode uses `mul_mv`) casts activations to **F16**, so the SwiGLU
  `silu(gate)*up` product overflows F16 (65504) in the down projection.
  `ggml_mul_mat_set_prec(F32)`, F32 KV cache, disabling fusion/concurrency, and
  manual attention all leave it. Fix: scale the down activation ÷256 before the
  matmul and ×256 after — a lossless exponent shift. (`52400a6`)

- **ggml-CPU ViT drift** — with the Metal path working, the ggml-CPU ViT graph
  still drifted to cos ~0.84 at late layers (vs Metal/scalar 0.96). Two CPU-only
  precision losses accumulate over 27 layers: ggml's CPU tanh-`gelu` routes
  through an **F16 lookup table** (input quantized to F16), and CPU `mul_mat`
  against a **Q8_0 weight quantizes the F32 activation to Q8_0** for the dot
  product (coarser than Metal's F16 `mul_mm`). Fix: explicit F32 tanh-gelu (via
  `ggml_tanh` = direct `tanhf`) + dequantize the square attention weights and the
  F16 FFN up to F32 on the CPU backend only (no-op on GPU). CPU ViT now matches
  Metal/scalar (layer 26 cos 0.844 → 0.958) and CPU end-to-end OCR is correct.
  (`2dc3b79`)

Also: threaded a `dump_cb` through `gv_run_llm_body` so the LLM diff actually
exercises the ggml graph (it previously only ran the scalar decode — which is why
the bug stayed hidden); LLM-graph diff now 7/7 cos 0.9999. Both graphs are now
DEFAULT ON for **all** backends (Metal + ggml-CPU); `CRISPEMBED_GRANITE_VIS_SCALAR`
/ `_LLM_SCALAR` opt out. See LEARNINGS "Q8_0 reshape", "Metal mul_mm F16 activation
overflow", and "ggml-CPU ViT precision".

**Decode perf** — compared the decoder hot paths against the sibling OCR backends
(qwen2vl/internvl2/deepseek) and adopted two wins they already had: (1) run the
tied-embedding LM head **in-graph on Metal** for the last token (gv_run_llm_body's
new `logits_out`) instead of a per-token `core_cpu::linear_cpu` matmul + hidden
readback; (2) drop the per-layer `ggml_cont` of the full KV history — pass the
cache views straight to `flash_attn_ext`. Decode **270 → 165 ms/tok (~1.6×)**, OCR still
correct on both backends. Then profiled the decode call (env-gated timers): it is
**~95 % GPU `graph_compute`** (~135 ms) — graph build (~0.8 ms) + `sched_alloc`
(~5 ms) are negligible, so a persistent decode graph (deepseek's
`build_persistent_decode_graph`) would NOT help. Decode is **dispatch-bound on
~800 tiny T=1 kernels**, so the win is cutting kernel count: skip the SwiGLU
down-proj ÷256/×256 F16-overflow guard for T=1 (it only matters for the prefill
`mul_mm` F16 cast; T=1 `mul_mv` is F32-safe). That took decode **165 → 139 ms/tok**
— **270 → 139 ms/tok (~1.9×) cumulative**, parity intact (LLM diff 7/7). Prefill
(~12 s) is ~100 % GPU compute + one-time Metal pipeline compilation (a persistent
server amortizes the latter); not fixable by graph management.

## June 21, 2026 — Backend audit: no other broken engines; esrgan/restormer/hat_sr

Swept all `src/` engines for broken / wrong-output / gated-off-because-broken
paths (distinct from "correct but not yet on ggml graphs"). **Granite was the only
backend the docs called broken, and it's now fixed** — every other `*_SCALAR` /
`*_FORCE_CPU` gate is "graph is the validated default, scalar is the opt-out." Two
real *accuracy* defects and one verification gap found and closed:

- **esrgan_sr** (`70afc70`): the default ggml graph approximated the body
  per-channel **PReLU with plain `ggml_relu`**, dropping the slope — so the GPU
  path was *less accurate than its own scalar fallback*. Implemented true PReLU
  from primitives (`relu(x) + slope·min(0,x)`, slope broadcast `[1,1,oc]`, F32
  cast for Metal). ggml has no PReLU op.
- **restormer** (`89a1955`): removed a 167-line `#if 0` block with a
  `_DISABLED` graph builder + a stale duplicate function — misleading dead scaffold.
- **hat_sr** (`a8d8676`): the OCAB had a "simplified, may not match" comment and
  `test_hat_diff.cpp` existed but was **never registered in CMake** → HAT had never
  actually been diffed. Wired up `test-hat-diff`, built a **self-consistent
  reference from the gguf weights** (new `tools/dump_hat_reference_from_gguf.py`:
  reverse the converter name map → load into the torch HAT arch → forward → ref;
  no original `.pth` needed), and verified: C++ vs torch **output cos 0.999968**.
  The OCAB + full pipeline are correct; the hedge was wrong. `hat-ref.gguf`
  uploaded to HF `cstr/text-super-resolution-gguf`.

Also scrubbed stale PLAN.md text (granite "BROKEN on Metal", and the GPU roster
that still listed restormer/nafnet/esrgan/safmn/mixtex as scalar though they have
ggml graphs).

## June 20, 2026 — Granite Vision OCR: root-caused via HF-blueprint diff, scalar restored

End-to-end Granite-Vision 3.3-2B OCR was producing garbage. A prior handover
(`handover-prompts/granite-vision-ocr-generation.md`) blamed the chat template;
that was wrong on every count. Methodical per-layer diffing against the **true
HF reference** (`granite-vision-ref.gguf`, built from real safetensors by
`tools/kaggle/granite-vision-parity/granite_vision_parity.py`) located the
actual faults:

- **Template is correct** as-is — the real HF `chat_template` uses
  `<|system|>/<|user|>/<|assistant|>` as plain text (LLaVA-Next style).
- **LLM math is correct** — `granite-llm-ref.gguf` self-consistency cos 1.0.
- **The ggml SigLIP ViT graph (`gv_run_vit_graph`) miscomputes** — HF-blueprint
  cos **0.05** (NaN/residual blow-up on CPU; broken from the first encoder
  layer), independent of quantization (q4_k≡q8_0) and attention form
  (flash≡manual). Same ggml-alloc buffer-reuse family as the Metal LLM graph.
- **The on-disk `q4_k.tok` model had Q4_0 vision weights** (quantized before the
  `vis→Q8_0` fix) → vision cos 0.32 even on the scalar path.

Fixes (branch `fix/granite-vision-real`):
- Default vision to the diff-validated **scalar ViT**; gate the broken graph
  behind `CRISPEMBED_GRANITE_VIS_GRAPH=1`.
- Projector GELU tanh→`ggml_gelu_erf` (`projector_hidden_act="gelu"`=erf).
- Quantizer keeps `proj.*` at Q8_0 (alongside `vis.*`); requantized a proper
  `granite-vision-3.3-2b-q4_k-visq8.gguf` (Q4_K LLM + Q8_0 vision/projector) —
  vision parity now matches q8_0 exactly.
- `core_gguf::tensor_map` alias in `gguf_loader.{h,cpp}` ending the cross-repo
  std::map↔unordered_map flip-flop with CrispASR.
- Diagnostic levers (`CRISPEMBED_GRANITE_VIS_SCALAR/GRAPH/CPU/DBG`) + per-layer
  harness dump stages (`vis_patch_embed`, `vis_layer_N`).

**Verified end-to-end OCR**: scalar vision + CPU LLM graph
(`CRISPEMBED_GRANITE_CPU=1 CRISPEMBED_GRANITE_LLM_GRAPH=1`) →
`<doc> The quick brown fox jumps over 1234. </doc>`. The graph backends remain
broken (Metal vision + Metal LLM) — the perf follow-up is
`handover-prompts/granite-vision-graph-fix.md`.

---

## June 20, 2026 — Performance optimization sweep

Full line-by-line audit of all ~57K lines across 60+ runtimes, followed by
systematic implementation of the highest-impact items.

### Core infrastructure
- **SIMD `dot_product()`** in `cpu_ops.h` — AVX2+FMA (x86-64) + NEON (ARM),
  used by `linear_cpu`, `mha_1q_cpu`, and all callers. 710+ FMA instructions
  in libcrispembed.so. `-march=native` via `CRISPEMBED_NATIVE` cmake option.
- **`DequantCache`** — per-context init-time weight caching, eliminates
  thousands of redundant dequant+alloc per decode session. Deployed to 15+
  runtimes (smoldocling, granite, bttr, hmer, posformer, 7 SR runtimes, etc.).
- **`RoPEFreqTable`** — precomputed frequency table eliminates `powf` per
  element per step. Deployed to smoldocling, granite.
- **`otsu_threshold()`** — extracted from 4 duplicated implementations to
  shared `cpu_ops.h`.
- **`std::unordered_map`** for tensor lookup in `gguf_loader.h` (was `std::map`).

### Runtime migrations
- **bttr/hmer/posformer** — replaced ~900 lines of duplicated conv2d/relu/
  layernorm/linear with `core_cpu` shared versions (SIMD-accelerated).
- **tesseract_lstm + gliner_ner** — LSTM gate inner loops use `dot_product()`.
- **smoldocling/granite** — DequantCache, RoPEFreqTable, SIMD linear_cpu,
  SIMD LM head matmul. Removed unused local helpers.
- **scunet_denoise** — hoisted per-pixel heap allocations outside spatial loops
  (was 100K+ allocs per swin block).
- **math_ocr** — global dequant cache → per-context DequantCache.
- **pcs** — FC head weights cached at init (no per-call GPU→CPU transfer).
- **mel.cpp** — SIMD mel projection via `dot_product()` (~38M MACs accelerated).
- **Orchestrator** — pre-load image once, pass pixels to 9 VLM engines.

### SR/restoration tiling
Added Hann-window overlap tiling to all 6 runtimes that lacked it:
esrgan_sr, safmn_sr, nafnet_denoise, scunet_denoise, instructir, adair.
Configurable via env vars. Small images bypass tiling.

### Other
- Sliding-window min/max pool in scan_cleanup (O(1) amortized via monotonic
  deque, was O(K) per pixel — ~50x for K=51).
- pdf_info: mmap instead of fread for large PDFs.
- layout_detect: ~30 debug printfs gated behind LAYOUT_DEBUG env var.
- ppformulanet_l: removed 370 lines of dead scalar encoder code.
- restormer: removed dead rst_gdfn() stub, fixed double variance computation.
- BPE merge: priority queue O(N log N) in bpe.h + tokenizer_bpe.cpp.
- WordPiece trie: O(len) longest-match via trie traversal (was O(len²) suffix scan).
- DAT SR BatchNorm fusion: 54 conv+BN pairs fused at load time (3 per AIM block × 18).
- Bilinear resize: replaced nearest-neighbor in 6 math/OCR runtimes for better quality.
- morph_fast: power-of-2 horizontal dilation for large kernels.
- tps_warp: coarse grid + bilinear interpolation (was O(W*H*N) with sqrt+log per pixel).
- gliner_ner: DeBERTa relative position tensor cached (was 117MB per call at T=200).
- OpenMP: parallelized pixel-level loops in image_preprocess, dewarp, scan_cleanup.
  Also mel.cpp STFT loop (parallel across frames, `if(T > 16)` guard).
- lightonocr: decode graph reuse — build once, update input data only across steps.
- parseq_ocr: encoder graph caching — built once, reused across recognize calls.
- internvl2: vision encoder graph cached across tile invocations.
- hmer_ocr: DenseNet encoder converted from scalar to ggml graph (3x speedup).

### Benchmark instrumentation (56 runtimes)
Added opt-in per-step timing to all 56 runtime files (61 files, ~1800 lines).
Each runtime has a `CRISPEMBED_<MODULE>_BENCH` env var that gates
`[module-bench]` stderr output. Zero overhead when unset — flag read once
at init, stored as bool. Covers: preprocess, encoder, decoder, per-tile,
per-decode-step, postprocess, total.

### Per-backend VLM optimizations
- **lightonocr** (2.09x total): flash attn default, direct embed lookup,
  F16 ggml KV cache (internvl2 pattern), patch embed → ggml matmul.
- **got_ocr**: patch embed → ggml matmul, neck+downsample+projector → ggml
  graph (conv2d_direct + LN2d via permute+norm + mul_mat).
- **glm_ocr**: downsample+merger → ggml graph (conv2d_direct + batched SwiGLU).
- **smoldocling**: patch embed → ggml matmul, F16 norm weight cast fix
  (unblocked ggml LLM on Q4_K models).
- **Native GQA**: removed ggml_repeat KV head expansion before flash_attn_ext
  in internvl2, lightonocr, got_ocr, glm_ocr (-76 lines total).
  flash_attn handles GQA via broadcast factors (rk2 = neq2/nek2).

### Pix2Struct full rewrite
- **ggml graph encoder**: 12-layer T5 encoder as single ggml graph with
  `ggml_flash_attn_ext` (scale=1.0 for T5), GeGLU FFN, `ggml_rms_norm`.
  Encoder time: ~930ms (was ~2-5s scalar).
- **KV-cached decoder**: incremental self-attn cache (O(T) not O(T²)),
  cross-attn K/V pre-computed once via ggml graph. Decoder step0 cos=1.0000.
- **Batched patch projection**: 128 sequential `linear_cpu` → single
  `ggml_mul_mat` in encoder graph.

### Decoder allocation hoisting (6 runtimes)
Pre-allocated `dec_scratch` struct on each context, reused across all steps:
- bttr_ocr (~30 allocs/step), posformer_ocr (~36), hmer_ocr (~15),
  math_ocr (scalar path), parseq_ocr (~18), pix2struct (72).

### Flash attention adoption
- **decoder_embed**: both single-text and batch paths → `ggml_flash_attn_ext`.
  F16 causal mask for non-bidirectional models.
- **bidirlm_vision**: F16 block-diagonal mask (halves mask memory).

### Batched linear for SR attention
- `linear_batch_cpu` primitive added to `core/cpu_ops.h`.
- dat_sr, swinir_sr, hat_sr, scunet_denoise, mixtex_ocr: per-token QKV/proj/FFN
  loops converted to single batched calls + SIMD dot_product in attention.

### cpu_ops.h SIMD acceleration (shared primitives)
- **layernorm_cpu**: AVX2+FMA SIMD for mean (parallel sum), variance
  (sub+fmadd accumulation), and scale+shift (fused v*w+b). Used by 12 engines.
- **rmsnorm_cpu**: AVX2+FMA SIMD for sum-of-squares (fmadd) and scale
  (8-wide multiply). Used by 12 engines.
- **softmax**: AVX2 SIMD for max-reduction (_mm256_max_ps) and
  normalization (8-wide multiply by 1/sum). Exp loop stays scalar.
- **mha_1q_cpu**: Swapped V accumulation loop from d-outer/ki-inner
  (cache-unfriendly) to ki-outer/d-inner (sequential V row access) +
  AVX2+FMA 8-wide vectorized inner loop. Used by ppformulanet, math_ocr.
- **layernorm2d_cpu**: Replaced 3 strided-access loops (stride H*W,
  cache-hostile) with gather→layernorm_cpu→scatter pattern. Gathered
  buffer is contiguous so norm benefits from layernorm SIMD.

### Allocation infrastructure
- **LFM2 ggml_backend_sched + T-bucketing**: migrated from raw
  `ggml_gallocr` to `ggml_backend_sched` with sequence-length bucketing
  (8/16/32/64/128/256/512). Same pattern as BERT encoder. Graph+alloc
  overhead: ~2ms → ~0.7ms for same-bucket inputs.
- **Persistent gallocr reuse**: 7 engines (vit_embed, clip_text_embed,
  parseq_ocr, cnn_embed, ocr_detect, surya_det, layout_detect) moved from
  per-call gallocr new/free to per-context persistent allocator.
- **TBSRN BatchNorm fusion**: fused 11 conv+BN pairs (2 per SRB × 5 + 1
  final) into conv weights at load time. Eliminates all runtime BN calls.
- **2D PE caching**: TBSRN (fixed 64×16×64, reused across 5 SRB blocks),
  BTTR/PosFormer (cached for last-used h×w dims). Eliminates ~327K
  sinf/cosf evaluations per inference on repeated same-size calls.

### Quantitative results
- 70+ optimization items completed (from 53 originally identified + extras).
- ~1500 lines of duplicated code removed across the codebase.
- SIMD active in 30+ runtimes via shared `dot_product()` / `linear_cpu` /
  `layernorm_cpu` / `rmsnorm_cpu` / `softmax` / `mha_1q_cpu`.
- All changes verified: 99/99 cpu_ops tests, 97/97 vlm_attention tests,
  live MiniLM-L6 embedding inference bit-identical to baseline.

---

## June 19, 2026 — core/cpu_ops.h refactoring (Phase 1)

Extracted ~100 lines of CPU-scalar helper functions duplicated across 6+ engine
files into a shared header-only `src/core/cpu_ops.h` (namespace `core_cpu`).

**Functions extracted:** to_f32 (GPU-safe dequant), layernorm_cpu (raw + tensor
overloads), layernorm2d_cpu, rmsnorm_cpu, linear_cpu (raw + tensor overloads),
conv2d_cpu (with groups), gelu (tanh approx), gelu_erf (exact), silu/silu_inplace,
softmax, hardswish_inplace, relu6_inplace, relu_inplace, mha_1q_cpu.

**Engines refactored:** surya_det, got_ocr, ppformulanet_l_ocr, ppformulanet_ocr,
deepseek_ocr2, mixtex_ocr. Net: −728 lines deleted, +74 lines added (using decls).

**Key design decisions:**
- No default `eps` parameter — every call site must be explicit to prevent silent
  behavior changes across engines that historically used 1e-5, 1e-6, or 1e-12.
- `gelu` vs `gelu_erf`: two variants because engines use different approximations
  (mixtex uses erf-exact matching `nn.GELU()`; ppformulanet_l uses tanh approx).
- `conv2d_cpu` has `groups=1` default so engines without grouped convolution
  don't need to change call sites.
- All `to_f32` upgraded to GPU-safe `ggml_backend_tensor_get` path (some engines
  previously used direct `t->data` access which fails on GPU backends).

**Testing:** 88 unit tests (test_core_cpu_ops.cpp), verified parity on surya-det
(q8_0, f16), glm-ocr-diff (cos=1.000000 all checkpoints), mixtex-diff (identical
to main branch output), ppformulanet (q4_0).

---

## June 19, 2026 — OCR Confidence, HF Uploads, LFM2.5

### Per-character/token confidence for all OCR engines

Added softmax-based confidence tracking to every OCR engine's greedy
decode loop. 15 engines now expose `<engine>_confidences()` +
`<engine>_mean_confidence()`. Wired through: C API
(`crispembed_math_ocr_confidences`), Rust FFI, Python
(`CrispMathOcr.confidences()`, `CrispOcrOrchestrator.region_rec_confidence()`),
and Server (JSON `"confidence"` + `"token_confidences"` fields).

Engines: parseq, tesseract_lstm, math_ocr, hmer, bttr, posformer, mixtex,
ppformulanet, ppformulanet_l, glm_ocr, got_ocr, qwen2vl_ocr, internvl2_ocr,
granite_vision, lightonocr. Test suite: 44/44 pass (26 unit + 18 live).

### dots.ocr — REMOVED from main (license issue)

dots.ocr (rednote-hilab) claims MIT on HuggingFace but has a supplemental
"dots.ocr LICENSE AGREEMENT" with PRC governing law (Hangzhou Arbitration),
unilateral license amendment (90-day forced migration), prohibited uses,
mandatory "Built with dots.mocr" attribution, and trademark restrictions.
Code moved to feat/dots-ocr branch only, with license warnings. HF repo
set to private.

### New model registry entries

- **FireRed-OCR** (Qwen3-VL 2B) — `cstr/firered-ocr-crispembed-GGUF`
- **H2OVL-Mississippi-0.8B** — smallest VLM OCR (OCRBench 751, 398MB Q4_K)
- **Nanonets-OCR2-1.5B** — Qwen2-VL pruned (16L), runs on qwen2vl_ocr
- **german-ocr-3.1** — Qwen2.5-VL fine-tune for German business docs
  (new `merge-llamacpp-qwen2vl-gguf.py` tool for split llama.cpp GGUFs)

### LFM2.5-Embedding + ColBERT (LiquidAI)

- LFM2.5-Embedding-350M: 1024d CLS hybrid embeddings, 11 languages
- LFM2.5-ColBERT-350M: per-token 128d multi-vector output
- Both: converter, parity test, registry, HF upload

### HuggingFace uploads

All OCR model repos now have F16 + Q8_0 + Q4_K GGUFs with READMEs:
granite-vision, lightonocr, dots-ocr, firered-ocr. DeepSeek-OCR-2
quantization running on Kaggle (6.4GB F16 too large for VPS).

### Bug fixes

- Layout Q8_0/F16 crash: `tensor_to_f32()` for all decoder weight reads
- MixTex decoder: parity VERIFIED (cos=1.0 — was reference GGUF inconsistency)
- LightOnOCR prompt: correct chat template token IDs for OCR output
- Qwen2-VL KV cache: cont V view fix for correct token-for-token decode

---

## June 16, 2026 — LightOnOCR-2-1B (OCR Arena #2)

End-to-end port of [lightonai/LightOnOCR-2-1B](https://huggingface.co/lightonai/LightOnOCR-2-1B)
(Apache-2.0, 1B params, OCR Arena #2 with ELO 1697).

- **Architecture**: Pixtral ViT (24L, 1024d, 2D RoPE, SiLU FFN) + spatial merge 2×2
  projection + Qwen3 decoder (28L, 1024d, GQA 16/8, QK norm, SwiGLU)
- **Converter**: `models/convert-lightonocr-to-gguf.py` — lazy safetensors loading
- **Engine**: `src/lightonocr.{h,cpp}` — vision encoder + projection + decoder
- **Key challenge**: Pixtral 2D RoPE (interleaved h/w frequencies, not mRoPE)
- **QK norm fix**: model produced EOS without chat template prompt framing;
  fixed by embedding prefix/suffix text tokens around image features
- **GGUF**: F16 (2.2GB), Q8_0 (1.0GB), Q4_K (622MB) — `cstr/lightonocr-GGUF`
- **Dispatch**: `--ocr` auto-detects from GGUF arch, `--ocr-engine lightonocr`
- **Orchestrator**: wired as single-shot VLM engine
- **Decode**: O(n²) full recompute per token (KV cache TODO)

---

## June 15-16, 2026 — KIE, LiLT, BERT NER, LID, Truecasing, Shared Libraries

### Key Information Extraction (KIE)

Two-phase pipeline for extracting structured fields from document images.

**Phase 1 — OCR + NER**: Chains OCR orchestrator (text detection + recognition)
with GLiNER zero-shot NER. Character offset tracking maps NER entities back to
source OCR regions with bounding boxes.
- Files: `src/kie_pipeline.{h,cpp}`, C API `crispembed_kie_*`
- CLI: `--kie FILE --kie-labels "total,date,vendor"`
- Server: `POST /kie/extract`
- Bindings: Python `CrispKIE`, Dart `CrispKIE`

**Phase 2 — LiLT Layout Transformer**: Dual-stream encoder (RoBERTa 768d +
layout transformer 192d) with BiACM (bidirectional attention complementation).
Token classification for form understanding (FUNSD: question/answer/header).
- Architecture: 130.7M params, 12 layers, 12 heads, MIT license
- Parity: 25/25 layers cos=1.000000 vs HuggingFace
- Files: `src/lilt_kie.{h,cpp}`, converter, ref dumper, diff test
- HF models: `cstr/lilt-funsd-GGUF`, `cstr/lilt-base-GGUF` (F32/Q8_0/Q4_K)

### BERT / XLM-R Fixed-Label NER

Fixed-label token classification NER using existing BERT/XLM-R encoders with
a Linear(hidden, num_labels) head. Auto-detected from GGUF (`ner.classifier.weight`).
Same `crispembed_ner_*` API — backend auto-dispatched (GLiNER vs BERT NER).

- `dslim/bert-base-NER`: 110M, CoNLL-03, 9 labels (PER/LOC/ORG/MISC), MIT
- `Davlan/xlm-roberta-base-ner-hrl`: 278M, 10 languages, 9 labels, MIT
- GELU fix: switched all BERT FFN to erf-exact (matching HF/PyTorch)
- Cased tokenizer fix: auto-detect `do_lower_case` from vocab content
- `crispembed_encode_tokens_raw()`: unnormalized hidden states for classification
- HF models: `cstr/bert-base-NER-GGUF`, `cstr/xlmr-ner-hrl-GGUF`

### Language Identification (LID)

Text-based LID integrated into OCR orchestrator for automatic Tesseract model
selection. ISO 639-1 → Tesseract 639-3 mapping (12 languages).

- Shared library: `CrispASR/crisp_lid/` (fastText + CLD3 + dispatch)
- Orchestrator: `config.lid_model`, runs LID after OCR, populates `result.detected_lang`
- Tesseract auto-select: `model_b = "auto"` → resolves `tesseract-{lang}-q8_0.gguf`
- Server: `POST /lid/detect`, `--lid MODEL` flag
- Bindings: Python `CrispTextLID`, Dart `CrispTextLID`
- C API: `crispembed_ocr_pipeline_detected_lang()`

### Truecasing

Post-OCR truecasing (German noun capitalization) via BiLSTM character-level model.

- Shared library: `CrispASR/crisp_truecase/` (stat + CRF + BiLSTM)
- Orchestrator: `config.truecase_model`, applied to `full_text` after OCR
- CLI: `--truecase-model MODEL`
- Bindings: Python `CrispTruecaser`, Dart `CrispTruecaser`

### Shared Libraries (cross-repo with CrispASR)

Extracted 3 new shared libraries to eliminate code drift between CrispASR and CrispEmbed:

| Library | Purpose | LOC |
|---------|---------|-----|
| `crisp_punc/` | Punctuation restoration (FireRedPunc + PCS) | 1666 |
| `crisp_lid/` | Text LID (fastText + CLD3 + dispatch) | 2098 |
| `crisp_truecase/` | Truecasing (stat + CRF + BiLSTM) | 1002 |

All follow the `crisp_audio/` pattern: self-contained CMakeLists, auto-detect
core target (`crispasr-core` or `crispembed-core`), conditional fallback to
local copies when sibling repo is absent.

### Table Structure Recognition

Rule-based table parser: morphological line detection → grid intersection →
per-cell OCR → HTML `<table>` output. No model needed.
- Files: `src/table_parse.{h,cpp}`, C API, CLI `--table`, server `POST /table/parse`
- Test: 14/14 pass (ruled + borderless grids)

### Orchestrator Tests

Comprehensive test suite: 56/56 PASS across 10 sections (classifier, accept-gate,
multi-stage escalation, chain selection, C API, edge cases, punctuation).

### Handover Prompts

All 18 handover prompts completed.

---

## June 2026 — Text Super-Resolution (PAN, TBSRN, NAFNet-SR)

Three engines for upscaling low-resolution text images before OCR, integrated
into the document preprocessing pipeline.

### PAN 4× whole-image super-resolution

Pixel Attention Network (PAN) for 4× upscaling of full document pages.

- **Architecture**: shallow feature extraction (Conv3×3) → 6 SC-PA blocks
  (depthwise-separable conv + pixel attention gates) → PixelShuffle(4) upsampler.
  272K parameters, C++ forward pass.
- **GGUF**: `pan-x4-f16.gguf` — 0.5 MB F16.
- **Converter**: `models/convert-pan-to-gguf.py`.
- **Parity**: cos=0.999654 vs PyTorch reference (F16, full-page input).
- **License**: Apache-2.0.

### TBSRN 2× per-line super-resolution

Text Before Super-Resolution Network (TBSRN) for 2× upscaling of individual
OCR text-line crops (telescope training scheme).

- **Architecture**: shallow feature extraction → 3 residual groups (6 TSA blocks
  each, transformer-style self-attention on spatial tokens) → PixelShuffle(2)
  upsampler. 1.1M parameters, C++ forward pass.
- **GGUF**: `tbsrn-telescope-f16.gguf` — 2 MB F16.
- **Converter**: `models/convert-tbsrn-to-gguf.py`.
- **Parity**: cos=0.999985 vs PyTorch reference (F16, 32×128 text-line crop).
- **License**: Apache-2.0.

### NAFNet-SR engine (no model yet)

Engine scaffolding for NAFNet-SR custom super-resolution models. Reuses the
existing `nafnet_denoise.cpp` architecture with a configurable upsampling tail.
No pre-trained GGUF included — supply a custom trained checkpoint via `--sr-model`.

### Integration matrix

| Surface | PAN (`--pan-sr`) | TBSRN (`--tbsrn-sr`) | NAFNet-SR (`--sr-model`) |
|---------|-----------------|----------------------|--------------------------|
| C API | `crispembed_pan_sr_*` | `crispembed_tbsrn_sr_*` | `crispembed_nafnet_sr_*` |
| CLI | `--pan-sr` | `--tbsrn-sr` | `--sr-model` |
| Server | `POST /pan/sr` | `POST /tbsrn/sr` | — |
| Python | `CrispPanSr` | `CrispTbsrnSr` | — |
| Rust | `CrispPanSr` | `CrispTbsrnSr` | — |

New files: `src/pan_sr.{h,cpp}`, `src/tbsrn_sr.{h,cpp}`,
`models/convert-pan-to-gguf.py`, `models/convert-tbsrn-to-gguf.py`,
`tools/dump_pan_reference.py`, `tools/dump_tbsrn_reference.py`,
`tests/test_pan_sr.cpp`, `tests/test_tbsrn_sr.cpp`.

### Auto-SR in orchestrator

The orchestrator's `--sr-model` now auto-detects PAN vs NAFNet-SR from
the GGUF architecture metadata. Tested on 75 DPI single-line text:
- 75 DPI raw → OCR: `C Melbe Wesld1` (garbage)
- 75 DPI + PAN 4x → OCR: `Hello Werdd 123` (1 char error, readable)
- 150 DPI raw → OCR: `Hello World 123` (perfect, no SR needed)

Finding: do NOT apply classical cleanup (binarize/deskew) to low-DPI
images — it destroys sub-10px text. PAN alone is sufficient.

---

## June 2026 — Tesseract LSTM OCR + classical preprocessing + renderers

### Tesseract LSTM line-recognition engine

Ported Tesseract's LSTM line-recognition engine to CrispEmbed via GGML.
126 languages from `.traineddata` files (435 KB–1.7 MB Q8_0 per language).

- Converter (`convert-tesseract-to-gguf.py`): recursive binary tree parser,
  int8 dequant, gate reorder, GGUF emit. Supports tessdata_best + tessdata_fast.
- C++ engine (`tesseract_lstm.{h,cpp}`): Conv stacking → FC+tanh → MaxPool →
  SummLSTM → LSTMs → Softmax → CTC decode. Pure CPU, no ggml graph.
- Python reference (`dump_tesseract_reference.py`): pure-numpy forward pass.
- Parity: 8/8 stages cos_min=1.000000. Spaces + punctuation emitted natively.
- 12 language GGUFs on HuggingFace (`cstr/tesseract-lstm-GGUF`).

### Classical preprocessing (from Leptonica, BSD-2)

CPU-only, model-free, fast tier. Self-contained C++, no Leptonica dependency.

- 1-bit DWA morphology (`morph_fast`): 21x speedup, 32x less memory.
- CC text line detection (`cc_detect`): model-free, 4.3ms/page, zero downloads.
- Adaptive Otsu (`classical_preproc`): per-tile + bilinear interpolation.
- Differential-square-sum deskew: 3ms/page, binary search on 4x-reduced image.
- CC despeckle: flood-fill + size filter.
- Background normalization: tile-based 90th-percentile + smoothing.
- Page dewarping (`dewarp`): cubic baseline fitting + disparity warp. 10ms.

### OCR result renderers (`ocr_render`)

Plain text (configurable separator), hOCR (XHTML), ALTO 3.1 (XML),
searchable PDF (invisible text layer, rendering mode 3). 36/36 tests.
Wired into CLI (`--output-format`), C API, Rust, Python.

### Punctuation restoration

FireRedPunc + PCS copied from CrispASR. Auto-detect from GGUF arch.
CLI `--punct-model`, C API, orchestrator integration. Registered in model_mgr.

### OCR pipeline orchestrator

Wired into HTTP server, Python, Dart, Rust. Full params in all layers.
CORS headers. VLM escalation in Rust. Verbose logging (`CRISPEMBED_VERBOSE_OCR`).
GOT-OCR2 GPU scheduler fix. CC detect as model-free detector option.

### Wiring

All new capabilities: C API + Rust FFI + safe Rust + Python bindings.
docs/contributing.md updated with utility library checklist + integration matrix.
### Additional improvements (June 15)

- **Searchable PDF with image**: JPEG XObject embedding + glyph-width-aware
  text positioning (Tm matrix, font scaled to match bbox width).
- **PDF/A-2b metadata**: XMP metadata stream + sRGB OutputIntent.
- **Refined DBNet postprocessing**: Moore contour tracing + convex hull +
  min-area rotated rectangle (rotating calipers) + polygon-interior scoring.
- **Text angle classification**: heuristic 0°/180° detection via
  ascender/descender asymmetry.
- **Image downsampling calculator**: DPI + max_pixels aware.
- **OCR quality scoring**: dictionary-based word matching.

63 new tests total, all passing.

---

## June 2026 — Qari-OCR (Arabic with diacritics, 2B, Apache-2.0)

Port of NAMAA-Space/Qari-OCR-0.2.2.1-VL-2B-Instruct — Arabic OCR with
full tashkeel (diacritics) support. Fine-tuned from Qwen2-VL-2B-Instruct
via LoRA (r=16, α=16, 324 adapter pairs) on 50K Arabic OCR samples.

**Architecture**: Same Qwen2-VL family as existing `qwen2vl_ocr.cpp`:
- Vision: 32L ViT (embed_dim=1280, hidden_size=1536, 16 heads)
- Spatial merger: 2×2, mlp 5120→1536
- LLM: 28L Qwen2 (1536d, GQA 12Q/2KV, FFN=8960)
- Total: ~2B params

**No new C++ code** — the existing qwen2vl_ocr engine reads all dimensions
from GGUF metadata and handles both Qwen2-VL-2B and Qwen2.5-VL-3B.

**Converter fix**: Qwen2-VL config uses `embed_dim`/`mlp_ratio`/`in_chans`
instead of Qwen2.5-VL's `intermediate_size`/`in_channels`/`out_hidden_size`.
Added `getattr` fallbacks in `convert-qwen2vl-to-gguf.py`. Key insight:
vision `hidden_size` (1536) ≠ ViT block dim (`embed_dim`=1280) — must
write `embed_dim` as the GGUF vision.hidden_size for correct block computation.

**Conversion**: Kaggle kernel (16 GB RAM needed) merges 324 LoRA pairs
tensor-by-tensor into fp16 base, then converts to GGUF + quantizes.
Took 4 kernel iterations to get right (config field name mismatches).

**GGUFs**: `cstr/qari-ocr-crispembed-GGUF` — F16 (4.7 GB), Q8_0 (2.3 GB),
Q4_K (1.6 GB). Registry entry: `qari-ocr`.

**Parity**: Not yet verified per-layer (needs Kaggle). The qwen2vl engine
has cos=1.000 parity on Qwen2.5-VL-3B; the 2B variant uses the same code
path with different dimensions. Test kernel prepared but not yet run.

**Performance** (published): WER=0.221, CER=0.059, BLEU=0.597.

---

## June 2026 — Scan cleanup (document preprocessing pipeline)

Two-tier document scan preprocessing module — pure C++, no external
tool dependencies.

### Tier 1 — Classical (no model needed)

Four operations, ~500 LOC in `src/scan_cleanup.{h,cpp}`:

1. **Deskew**: Sobel edge detection → Hough line accumulator → median angle
   → bilinear affine rotation. Detects 3-degree skew exactly on synthetic tests.
2. **Binarization**: Otsu global (histogram between-class variance) and
   Sauvola adaptive (integral image for O(1) per-pixel local mean/stddev).
3. **Border crop**: row/column intensity projection → content rectangle detection.
4. **Background whitening**: morphological open (min-pool → max-pool) estimates
   background surface, then divide to normalize. Reduces background variance
   to near zero.

### Tier 2 — Learned denoising (NAFNet, MIT license)

Port of megvii-research/NAFNet (ECCV 2022) for image restoration.
Non-linear Activation Free Network — uses SimpleGate (channel split ×
element-wise multiply) instead of ReLU/GELU.

**Architecture**: U-Net with NAFBlocks.
- Intro: Conv3x3 (3→32)
- Encoder: [2,2,4,8] NAFBlocks at [32,64,128,256] channels
- Downsampling: Conv2x2 stride 2
- Middle: 12 NAFBlocks at 512 channels
- Decoder: [2,2,2,2] NAFBlocks with PixelShuffle(2) upsampling + skip connections
- Ending: Conv3x3 (32→3) + input residual
- 29.2M params, pre-trained on SIDD (smartphone denoising)

**NAFBlock**: LayerNorm2d → Conv1x1(c→2c) → DepthwiseConv3x3(2c) →
SimpleGate(2c→c) → SCA(AvgPool→Conv1x1) → Conv1x1(c→c) → residual×beta
→ LayerNorm2d → Conv1x1(c→2c) → SimpleGate → Conv1x1(c→c) → residual×gamma

**Implementation**: CPU-scalar forward pass in `src/nafnet_denoise.{h,cpp}`.
All standard ops: conv2d (1x1, 3x3, depthwise), LayerNorm2d, element-wise
multiply, global average pool, PixelShuffle.

**Parity** (64x64, all vs PyTorch reference):
- F32:  cos=0.9980, max_diff=48 px
- F16:  cos=0.9980, max_diff=48 px
- Q8_0: cos=0.9980, max_diff=47 px
- Q4_K: cos=0.9977, max_diff=48 px

Residual gap from 1.0 is uint8 quantization at input/output boundaries
(PyTorch processes float32 end-to-end; C++ goes u8→f32→model→f32→u8).

**Bug found**: `to_f32()` dequant function returned zeros for Q8_0/Q4_K
types instead of using `ggml_get_type_traits()->to_float`. Fixed.

**Quantizer fix**: added `.beta`/`.gamma` to the `is_add_operand` guard
in `tools/quantize.cpp` so NAFNet's per-channel residual scale factors
are never quantized (they're tiny [1,C,1,1] tensors used in element-wise
multiply — quantizing them corrupts the residual connections).

**GGUFs**: `cstr/nafnet-sidd-GGUF` — F16 (56 MB), Q8_0 (30 MB), Q4_K (16 MB).
Registry entry: `nafnet-denoise`.

### Wiring

All surfaces wired:
- **C API**: `crispembed_scan_cleanup_{init,process,free,defaults}` +
  `crispembed_scan_cleanup_process_simple` (for FFI without struct-by-value)
- **CLI**: `--cleanup` (preprocess before OCR), `--cleanup-only FILE` (standalone)
- **Server**: `POST /scan/cleanup` (always available, no model needed)
- **Python**: `CrispScanCleanup` class with `.process()` (file/PIL/numpy)
- **Rust**: `CrispScanCleanup` safe wrapper
- **Dart/Flutter**: `CrispScanCleanup` via `process_simple` FFI

**New files**: `src/scan_cleanup.{h,cpp}`, `src/nafnet_denoise.{h,cpp}`,
`models/convert-nafnet-to-gguf.py`, `tools/dump_nafnet_reference.py`,
`tests/test_scan_cleanup.cpp`.

---

## June 2026 — Surya detector GPU backend (Metal on M1)

`surya_det.cpp` hardcoded `ggml_backend_cpu_init()`, so even after the CUDA
build was fixed on Kaggle (GGML_CUDA_NO_VMM=ON) the detector still ran CPU-only.
Switched to `ggml_backend_init_best()` so the stage 0-2 and stage-3-block0
graphs run on the best available backend — Metal on Apple Silicon, CUDA
elsewhere — with `SURYA_DET_FORCE_CPU=1` to pin CPU for parity debugging and a
CPU fallback if no GPU backend initialises.

One gotcha: the scalar LiteMLA and decode-head paths dequantised weights via
`to_f32()`, which read `t->data` directly. That is fine for a CPU buffer but
`t->data` is not a valid host pointer on a GPU buffer, so the reads were routed
through `ggml_backend_tensor_get()` instead.

Verified on an M1 (Apple7, MTL0): F16 and Q8_0 both run on Metal, heatmap
parity vs CPU to ~3 decimals (sub-pixel bounding-box drift from F16 matmul
accumulation order). Stage 0-2 graph ~4.4 s GPU vs ~5.9 s CPU, stage-3 block0
~0.75 s vs ~0.94 s; the speedup is modest because LiteMLA + decode head stay
CPU-scalar. CUDA build separately confirmed on Kaggle P100 (Q8_0+F16 → 17
regions). Surya GPU is now marked done in PLAN.md.

---

## June 2026 — Surya detector Q8_0/Q4_K crash fix

The surya text detector (`surya_det.cpp`) crashed on quantized models (Q8_0, Q4_K)
with a segfault in `ggml_compute_forward_dup`. Root cause: two issues in `g_conv()`:

1. **Reshape before dequant**: `ggml_reshape_4d` on Q8_0 tensors created `ne[0]=3`
   (for 3×3 conv kernels), violating Q8_0's block alignment requirement (32 elements
   per block). The subsequent cast operation read invalid block data.

2. **Q→F16 cast unsupported**: ggml only implements quantized→F32 dequantization,
   not quantized→F16. The direct `ggml_cast(Q8_0, F16)` hit `GGML_ABORT`.

**Fix**: Dequant Q→F32 first, then reshape to 4D, then cast F32→F16 for `ggml_conv_2d`.
All four variants (F32, F16, Q8_0, Q4_K) now detect identically on synthetic test images.

Kaggle P100 testing confirmed F16 works (195s, 17 regions detected). CUDA cmake
still fails due to upstream ggml `CUDA::cuda_driver` target issue on Kaggle.

---

## June 2026 — GOT-OCR2 engine (0.7B, SAM ViT-B + Qwen2-0.5B, Apache-2.0)

Port of stepfun-ai/GOT-OCR2_0 — end-to-end document OCR handling plain text,
LaTeX math, tables, and formatted output. Fourth VLM in CrispEmbed.

**Architecture**: SAM ViT-B (12L, 768d, 12 heads, LayerNorm+GELU, windowed
attention ws=14 with global at [2,5,8,11], decomposed relative position encoding)
→ Neck (Conv 768→256, 1×1 → LN2d → Conv 256→256, 3×3 → LN2d) → Downsample
(Conv 256→512→1024, stride 2, 4096→256 tokens) → Linear(1024,1024) projector
→ Qwen2-0.5B (24L, 1024d, MHA 16/16, SiLU SwiGLU, standard RoPE θ=1M)
→ autoregressive generation with KV cache.

**Key differences from GLM-OCR**: Vision uses LayerNorm+GELU (not RMSNorm+SiLU),
no Q/K norm, SAM-style windowed+global attention with decomposed RPE (not CogViT).
LLM is standard pre-norm Qwen2 (2 norms/layer, not post-norm 4 norms/layer),
MHA (not GQA), standard RoPE (not mRoPE), tied word embeddings.

**Parity**: All checkpoints cos ≥ 0.999 (vision layers, neck, downsample,
projector, LLM layers).

**GGUFs**: `cstr/got-ocr2-crispembed-GGUF` — F16 (1.34 GB), Q8_0 (569 MB),
Q4_K (422 MB).

**New files**: `src/got_ocr.{h,cpp}`, `models/convert-got-ocr-to-gguf.py`,
`tools/dump_got_ocr_reference.py`, `tests/test_got_ocr_diff.cpp`.

---

## June 2026 — GLM-OCR engine (0.9B, CogViT + GLM-0.5B, MIT)

Port of zai-org/GLM-OCR — #1 on OmniDocBench V1.5, 8 languages, MIT license.
Third VLM in CrispEmbed, with three architectural firsts:

**Architecture**: CogViT (24L, 1024d, RMSNorm+SwiGLU, Q/K RMSNorm, Conv3D
patches) → RMSNorm → Conv2D downsample (stride 2, 576→144 tokens) → Merger
(proj + SwiGLU + LayerNorm) → GLM-0.5B (16L, 1536d, GQA 16/8).

**Unique features**: post-norm (4 norms/layer), Q upscale (1536→2048),
learned Conv2D downsample, mRoPE sections [16,24,24].

**Full pipeline**: KV cache (F16, prefill+decode), vision-text splice
(144 image tokens), tokenizer decode, E2E image→text verified.

**Parity**: 11/11 cos=1.000000 (8 vision + 3 LLM).

**GGUFs**: `cstr/glm-ocr-crispembed-GGUF` — F16 (2.5 GB), Q8_0 (1.1 GB),
Q4_K (849 MB).

**New files**: `src/glm_ocr.{h,cpp}`, `models/convert-glm-ocr-to-gguf.py`,
`tools/dump_glm_ocr_reference.py`, `tests/test_glm_ocr_{diff,e2e,image}.cpp`.

---

## June 2026 — Layout detection fixes + BGE-M3 crash fix

**Layout detection (RT-DETRv2):** Three bugs fixed, score 0.047 → 0.114:
1. AIFI self-attention head interleaving — permute `[hd, N, nh] → [hd, nh, N]`
   before reshape. Encoder features now exact-match Python.
2. Initial reference points — RT-DETRv2 uses `sigmoid(gather(enc_bbox_head(ALL) +
   logit_anchors, top_k))`, not `enc_bbox_head(gathered_queries)`.
3. Identified decoder `cpu_linear` weight convention mismatch (remaining gap).

**BGE-M3 crash:** `clip_text::load()` accepted any model with a tokenizer, loading
BGE-M3 (250K vocab XLM-R) as a 49K-vocab CLIP model → crash. Fixed by checking for
`clip_text.hidden_size` metadata key. BGE-M3 now loads correctly with sparse + ColBERT heads.

**AuraFace Q4_K:** 124 MB → 35 MB (3.5x compression), cos=0.961 vs F16.

---

## June 2026 — GLiNER DeBERTa-v3 NER (Apache-2.0)

Added DeBERTa-v3-base backbone to GLiNER NER — `urchade/gliner_medium-v2.1`,
the most popular GLiNER model (25k+ downloads), fully Apache-2.0 licensed.

**Architecture:** DeBERTa-v3-base (12L, 768h, disentangled c2c+c2p+p2c attention
with log-bucketed relative positions) + 768→512 projection + BiLSTM (hidden=256)
+ GLiNER markerV0 head (start+end only, no first-token projection).

**Implementation:** Unified `src/gliner_ner.cpp` with dual-backbone support.
Backbone auto-detected from `gliner.backbone` GGUF metadata. SentencePiece
tokenizer (128K vocab) via existing `tokenizer_spm.cpp`.

**Quantization:** F32 (747 MB), Q8_0 (198 MB, identical output), Q4_K (152 MB,
minor span merging at edges).

**New files:** `models/convert-gliner-deberta-to-gguf.py`, HF repo at
`cstr/gliner-deberta-GGUF`.

---

## June 2026 — PARSeq scene text recognition (Apache-2.0)

Scene text recognition port: PARSeq (ECCV 2022, baudm/parseq, Apache-2.0).
First dedicated scene text (non-math, non-document) OCR model in CrispEmbed.
Two variants: base (24M params) and tiny (6M params).

**Architecture**: 12-layer pre-LN ViT encoder (patch [4,8], img 32×128,
128 tokens, GELU FFN, fused QKV) → 1-layer two-stream Transformer decoder
(XLNet-style: position queries attend to context via norm_q/norm_c, then
cross-attend to encoder memory) → Linear head (95 classes: 94 printable
ASCII chars + EOS).

**Key design**: Two-stream attention where context tokens combine position
queries + character embeddings. Token ordering: EOS=0, chars=1..94, BOS=95,
PAD=96 (not the typical BOS-first). Single query per AR step for efficiency.

**Variants**:
- Base: embed_dim=384, 6 enc heads, 12 dec heads (head_dim=32)
  F32=91MB, Q8_0=24MB, Q4_K=13MB
- Tiny: embed_dim=192, 3 enc heads, 6 dec heads
  F16=12MB, Q8_0=6MB

**Encoder**: runs as ggml graph (flash_attn_ext, BLAS-backed matmuls).
Patch embedding done CPU-side (non-square kernel [4,8] not supported by
ggml_conv_2d). **Decoder**: CPU-scalar (1 layer, graph overhead not worth it).

**Parity**: Verified identical output to PyTorch on multiple test images.
All quantization levels (F32/Q8_0/Q4_K) produce identical decoded text.

**New files**: `src/parseq_ocr.{h,cpp}`, `models/convert-parseq-to-gguf.py`,
`tools/dump_parseq_reference.py`, `tests/test_parseq.cpp`.

**Bugs found during port**:
1. Token ordering: PARSeq uses `[EOS, chars, BOS, PAD]` not `[BOS, chars, EOS, PAD]`
   — BOS=95, EOS=0 in both head output and embedding space.
2. Context construction: `ctx[0] = embed(BOS)`, `ctx[k] = pos_queries[k-1] + embed(pred)`
   — position queries are added to character embeddings, except BOS which has none.
3. norm_c: context K/V in self-attention must be LayerNorm'd via norm_c (not raw).
4. Head excludes BOS and PAD: 95 output classes = EOS(0) + 94 chars(1..94).

**License**: Apache-2.0 (baudm/parseq). Fully commercial.

---

## June 2026 — InternVL2.5-2B OCR engine (VLM, MIT)

Full vision-language model port: InternVL2.5-2B (2.1B params, MIT license)
for multilingual document OCR. Second VLM in CrispEmbed after Qwen2.5-VL,
with KV cache for efficient autoregressive generation.

**Architecture**: InternViT-300M (24L, 1024d, 16 heads, LayerNorm + GELU +
LayerScale, 448×448 per tile) → pixel unshuffle (4:1, 1024→4096 dim) →
MLP projector (LN-Linear-GELU-Linear, 4096→2048) → InternLM2.5-1.8B
decoder (24L, 2048d, GQA 16/8, SwiGLU, RMSNorm, RoPE θ=1M).

**Key features**:
- Dynamic tiling: 1-12 tiles of 448×448 + optional thumbnail
- KV cache: F16 persistent cache, prefill+decode verified identical
- Vision-text splice: mask-based embedding replacement at `<IMG_CONTEXT>`
- C++ tokenizer decode: SentencePiece BPE from GGUF vocab (▁→space, byte fallback)
- OCRBench ~830 (top tier for models under 3B)

**Parity (F32, all vs Python reference via diff harness):**
- Vision encoder: 4/4 layers cos=1.000000
- Pixel unshuffle + MLP projector: cos=1.000000
- LLM decoder: 2/2 layers cos=1.000000

**E2E verification**: German invoice (600×400, 7 tiles) correctly extracts
invoice number, date, recipient, address, all line items with prices, and
net total.

**New files**: `src/internvl2_ocr.{h,cpp}`, `models/convert-internvl2-to-gguf.py`,
`tools/dump_internvl2_reference.py`, `tests/test_internvl2_{diff,e2e,image}.cpp`,
`tests/test_internvl2_ocr.py`, `hf_readmes/internvl2.5-2b-crispembed-GGUF.md`.

**GGUFs**: `cstr/internvl2.5-2b-crispembed-GGUF` — F16 (4.9 GB), Q8_0 (2.2 GB),
Q4_K (1.4 GB). Vision weights kept at Q8_0 floor in quantizer.

**License**: MIT (OpenGVLab/InternVL2_5-2B).

**Sibling variants on the same engine** (no new code — just GGUF conversion +
registry entries, the InternViT vision tower and projector are shared):
- **InternVL2-1B** (0.9B, MIT) — InternViT-300M + Qwen2-0.5B decoder. Edge/WASM
  target, OCRBench 779. GGUFs: F16 (~1.8 GB), Q8_0 (~1.0 GB), Q4_K (~0.5 GB).
- **H2OVL-Mississippi-2B** (2.1B, Apache-2.0) — InternViT + H2O-Danube2-1.8B
  (Mistral arch). OCRBench 782. GGUFs: F16 (1.2 GB), Q4_K (457 MB).

---

## June 2026 — GLiNER zero-shot NER (LFM2.5 backbone)

Added zero-shot Named Entity Recognition via SauerkrautLM-LFM2.5-GLiNER.
First non-embedding, non-OCR NLP task in CrispEmbed.

**Architecture:** LFM2.5-350M bidirectional backbone (ported from CrispASR's
LFM2-Audio implementation) with:
- 16 layers (10 ShortConv + 6 GQA attention), SwiGLU FFN
- Bidirectional attention (no causal mask) + symmetric conv padding
- Layer fusion (squeeze-and-excitation with sigmoid gates)
- BiLSTM (1-layer bidirectional, word-level)
- GLiNER head: SpanMarkerV1 span representation + dot-product scorer

**Parity (all vs Python reference via diff harness):**
- All 16 backbone layers: cos=1.000000
- Layer fusion: cos=1.000000
- BiLSTM: cos=1.000000
- End-to-end: 17/17 entities match across 5 test texts, mean score Δ=0.030

**New files:** `src/gliner_ner.{h,cpp}` (C++ runtime), `models/convert-gliner-lfm-to-gguf.py`
(converter), `tools/dump_gliner_reference.py` (reference dumper), C API
(`crispembed_ner_*`), server `POST /ner/extract`, Python `CrispNER`, Rust `CrispNER`,
Dart `CrispNER`.

**License:** LFM Open License v1.0 (free under $10M revenue).

---

## June 2026 — Qwen2.5-VL OCR engine (German document OCR)

### Qwen2.5-VL-3B port (feat/keyven-german-ocr branch → merged to main)

Full vision-language model port: Qwen2.5-VL-3B-Instruct as the base
for Keyven/german-ocr-3 (German business document OCR fine-tune).
First VLM in CrispEmbed — all prior OCR models were encoder-decoder
without a language model backbone.

**Architecture**: 32-layer ViT (1280d, 16 heads, 14×14 patches, 2D RoPE,
windowed attention) → spatial merger (2×2 merge, RMSNorm, FC-GELU_erf-FC,
5120→2048d) → 36-layer Qwen2.5 LLM decoder (2048d, GQA 16Q/2KV heads,
SwiGLU FFN 11008d, mRoPE sections=[16,24,24], rope_theta=1M).

**Parity**: cos=1.000000 across all checkpoints:
- Vision encoder: 32/32 ViT layers + spatial merger
- LLM decoder: 2/2 tested layers with mRoPE
- Patch embedding, token embedding: exact match

**End-to-end generation**: Q4_K (2.6 GB) produces coherent German text
from test invoice image. Prompt: "Extrahiere die Rechnung im Bild als JSON"
→ Output: "Um die Rechnung im Bild als" (8 tokens, greedy).

**GGUFs uploaded** to `cstr/qwen2.5-vl-3b-crispembed-GGUF`:
- F16: 7.57 GiB (converted on Kaggle, 73s)
- Q8_0: 3.93 GiB (2x compression)
- Q4_K: 2.61 GiB (3x, vision weights kept at Q8_0 floor)

**Key technical challenges solved**:
1. **Memory-efficient reference dumper** — numpy-based layer-by-layer
   forward pass via safetensors (600 MB peak vs 7.5 GB for PyTorch load).
2. **ggml_set_output()** — without it, graph allocator reuses intermediate
   tensor memory; reading post-compute gives garbage. Gate behind diff mode.
3. **GQA interleave** — `ggml_repeat` tiles [0,1,0,1,...] but GQA needs
   [0,0,...,1,1,...]. Fix: reshape to 4D, repeat on inner dim, reshape back.
4. **mRoPE neghalf** — `GGML_ROPE_TYPE_MROPE` uses neghalf rotation with
   dim pairs (j, j+half), not adjacent (j, j+1). Position tensor layout:
   [t0..tn, h0..hn, w0..wn, 0..0] (4 × n_tokens).
5. **Vision-text splicing** — `x = embed * keep_mask + image_patches`
   (keep_mask=0 at image_pad positions).
6. **Quantizer vision floor** — Q4_K degrades OCR; vision encoder weights
   forced to Q8_0 minimum in `tools/quantize.cpp`.
7. **AutoConfig version hell** — Kaggle's older transformers nests LLM
   params in text_config differently. Fixed: read raw config.json directly.
8. **WASM build fix** — `-sENVIRONMENT=web,worker` required when `-pthread`
   is enabled (pre-existing CI failure, fixed as part of this work).

**Standalone CLI pipeline** (completed 2026-06-12):
- C++ image preprocessor wired into `recognize_raw()` — smart_resize,
  bicubic, normalize, patchify via `image_preprocess.cpp`
- BPE tokenizer loaded from GGUF at init — `set_prompt()` tokenizes
  any text, chat template built in C++ with proper token IDs
- GPT-2 byte decoder for UTF-8 output text
- KV cache: prefill extracts per-layer K/V, decode steps reuse cache
  (O(1) per token instead of O(n) full recompute)
- GGUFs v2 on HuggingFace: all three (F16, Q8_0, Q4_K) include BPE
  tokenizer data (vocab + merges)

**Files added**:
- `src/qwen2vl_ocr.{h,cpp}` — C++ engine + C ABI (~1500 lines)
- `models/convert-qwen2vl-to-gguf.py` — GGUF converter (lazy tensor, with tokenizer)
- `tools/dump_qwen2vl_reference.py` — parity reference dumper
- `tools/qwen2vl_tokenize.py` — chat template tokenizer helper
- `tools/kaggle/qwen2vl-convert/` — Kaggle conversion + quantization kernel
- `tests/test_qwen2vl.cpp` — unit + smoke tests (14/14 pass)
- `tests/test_qwen2vl_diff.cpp` — per-layer parity diff test
- `tests/test_qwen2vl_e2e.cpp` — end-to-end generation test

**Remaining** (see PLAN.md blueprint):
- Load Keyven/german-ocr-3 fine-tuned weights (same arch, different weights)
- Windowed ViT attention (correct but slower without it)
- Python bindings, CrispCalc Dart catalog

---

## June 2026 (late) — surya text detector + MixTex LaTeX OCR

### surya-ocr-2 text detector port

EfficientViT-Large segformer (38M params, 91 languages incl. German).
Segmentation-based text line detection. OpenRail-M license (free <$5M).

**Architecture**: Stem + 4 CNN stages (FusedMBConv, MBConv) + 6
EfficientVitBlock (LiteMLA linear attention) + SegFormer FPN decode head.
Input 1200×1200, output 300×300 heatmap → polygon bounding boxes.

**Parity**: Verified exact match vs Python reference (heatmap max=0.9649,
mean=0.0113, both exact). Per-stage activation means match to 4dp through
all 10 encoder stages + decode head.

**Performance**: ggml graph acceleration for stages 0-2 + block0
(17s graph vs ~10min scalar = 35x). Total: ~1 min (was ~13 min).

**Quantized**: F32=147MB, F16=74MB, Q8_0=41MB (3.6x), Q4_K=23MB (6.5x).
All uploaded to https://huggingface.co/cstr/surya-det-GGUF

**Fully wired**: C ABI (`crispembed_text_det_*`), HTTP server
(`POST /text/detect`), Python bindings (`CrispTextDetect`), model
registry with auto-download, test binaries.

**Bugs found and fixed**:
1. `H /= 2` gives wrong result for odd H (75→37 instead of 38)
2. Stage 2+3 MBConv used ReLU6 instead of Hardswish
3. F16 GGUF: bias tensors need F32 cast before ggml_add

### MixTex Chinese+English LaTeX OCR port

Swin-Tiny encoder + 4-layer RoBERTa decoder (86M params, Apache-2.0).
First Swin (shifted-window attention) encoder in CrispEmbed.

**Architecture**: Patch embed (Conv2d 4×4) → 4 Swin stages
(depths=[2,2,6,2], window_size=7, shifted windows, relative position
bias) → patch merging → final LayerNorm → 4-layer RoBERTa decoder
with cross-attention → BPE tokenizer (25681 tokens, LaTeX + CJK).

**Parity**: cos=1.000000 on all encoder blocks (non-shifted and shifted).
Per-block diff harness verified: enc_embed, s0_b0_out, s0_b1_ln1,
s0_b1_attn_out_windowed, s0_b1_attn_merged, s0_b1_attn_res, s0_b1_out
all pass with max_abs < 2e-5. Quantized (Q8_0) produces identical output.

**Bugs found and fixed** (6 total):
1. Swin PatchMerging must pad odd dims before halving (125→126→63 not 125→62)
2. Cyclic shift sign convention — `cyclic_shift(shift_h=s)` computes
   `out[y]=in[(y+s)%H]` but `torch.roll(shifts=s)` computes
   `out[y]=in[(y-s)%H]`. Signs were inverted for both forward and reverse.
3. Pad-then-shift order — HF Swin pads to window-size multiples FIRST,
   then applies torch.roll. C++ was shifting on the unpadded grid then
   padding. This changes where boundary tokens end up in windows.
4. GELU variant — C++ used tanh approximation, HF Swin uses `nn.GELU()`
   (exact erf). Changed to `0.5 * x * (1 + erff(x / sqrt(2)))`.
5. PatchMerging 2×2 concat order — HF concatenates [TL, BL, TR, BR] but
   C++ had [TL, TR, BL, BR]. All 4 encoder stages diverged.
6. RoBERTa position embedding offset — positions start at index 2
   (padding_idx=1), not 0. Using index 0 reads wrong embeddings.

**Decoder parity** (step 0, all vs HF reference):
All checkpoints cos=1.000000 — embedding, self-attention, cross-attention
Q/K/V, all 4 decoder layers, and step-0 logits. Real math formula
`x^2 + y^2 = r^2` produces correct LaTeX matching HF for ~15 tokens.

**Debugging methodology**: Systematic per-step diff comparison with
named Python reference tensors. The per-step approach was critical:
encoder blocks all passed but stage output failed → PatchMerging bug.
Decoder embedding + self-attention passed but cross-attention failed
→ pre-computed K/V from wrong encoder output → traced back to PatchMerging.

**GGUFs**: F32=329MB, F16=165MB, Q8_0, Q4_K.
Wired into unified math OCR dispatch (auto-detect from GGUF arch).

---

## June 2026 (late) — PosFormer handwritten math OCR

### PosFormer port (feat/posformer-port branch)

PosFormer = BTTR + Attention Refinement Module (ARM) for coverage-aware
decoding. Source: SJTU-DeepVisionLab/PosFormer (BSD-2, academic-only).
6.4M params, 113 LaTeX tokens, 24.9 MB F32 GGUF.

**Architecture**: DenseNet encoder (same as BTTR) + 3-layer Transformer
decoder (d=256, 8 heads, FFN=1024) + shared ARM module. ARM applies
coverage-based attention refinement between decoder layers 0→1 and 1→2.

**CROHME 2014 eval (986 images, greedy L2R)**:
- Raw match:    552/986 = **56.0%** (vs BTTR 49.2%, HMER 36.1%)
- Parsed match: 605/986 = **61.4%** (vs BTTR 49.8%, HMER 36.3%)
- Published 62.7% uses bi-directional beam search; ~6pp gap is expected.

**Quantized**: Q8_0 (12 MB), Q4_K (10 MB) — both lossless on test images.
Uploaded to HuggingFace: https://huggingface.co/cstr/posformer-hw-GGUF

**Port verified**: per-step logit cosine similarity = 1.000000 vs PyTorch
reference (max diff < 0.00001). Four encoder/decoder bugs found and fixed:
1. Spurious ReLU after feature projection Conv1x1
2. LayerNorm and 2D positional encoding order swapped
3. Sin/cos frequency indexing error (cos used wrong frequency in each pair)
4. Missing decoder input LayerNorm (decoder.norm after embed + pos_enc)

**Debugging methodology**: PyTorch-side layer dump scripts
(tests/parity/posformer_*.py) + C++ POSFORMER_DUMP env-gated intermediate
dumps. Compare cosine similarity per-layer, per-step. First divergence
at layer 0 self-attention output led to finding the missing LayerNorm.

**Files**: `posformer_ocr.{h,cpp}`, `convert-posformer-to-gguf.py`,
`test_posformer.cpp`, `test_posformer_batch.cpp`,
`tests/parity/posformer_*.py`.

**Training pipeline** (v25, 25 iterations to get right):
Kaggle kernel at https://www.kaggle.com/code/chr1str/posformer-train-on-mathwriting
W&B at https://wandb.ai/cze-github/posformer-hmer

Key issues solved during Kaggle kernel development:
- P100 GPU (sm_60): install torch+cu118 (supports sm_60), not CPU fallback
- Auth: clone CrispASR at runtime, import kaggle_harness (3-tier auth).
  Dataset mounts at `/kaggle/input/datasets/chr1str/crispasr-hf-token/`,
  NOT `/kaggle/input/crispasr-hf-token/`. Harness patched to scan both.
- **Vocab bug**: `build_vocab_from_zip` sorted by frequency, scrambling
  110/113 token indices. Model trained 25 epochs was unusable. Fixed:
  use canonical PosFormer dictionary.txt (alphabetical order).
- OOV: 14 CROHME captions have `'` not in dictionary. Filtered.
- Validation speed: override beam_size=10 bidir → beam_size=1 greedy.
  Full bidir takes 30-60 min per val epoch.
- Heartbeat: `kh.build_heartbeat("train")` for Kaggle keepalive.

**Training progress** (correct vocab, label smoothing 0.1):
- Epoch 8: 22.4% beam=1
- Epoch 64: 43.4% beam=1 (pre-LR-fix)
- Epoch 93: 57.0% beam=1 (LR=0.005, surpasses SJTU published 56.0%)
- Epoch 108: 59.3% beam=1 (CROHME-only ceiling)
- Epoch 125: 61.9% val_ExpRate (CROHME + 1000 MathWriting, LR=0.005)
- **Epoch 182: 60.5% beam=1 / 60.3% beam=10** (CROHME + 2000 MathWriting,
  LR=0.00125 after ReduceLROnPlateau drop). Best verified full eval.
- W&B peak: 62.03% val_ExpRate at step 304,204

Key findings:
- MathWriting augmentation (2000 samples) broke the 59.3% CROHME-only ceiling
- ReduceLROnPlateau drop (0.005→0.00125) triggered the 62% peak
- Beam=10 bi-directional does NOT help (60.3% < 60.5% beam=1)
- Model is better at greedy than bi-directional decoding
- deepcopy/MathWriting-human on HF has pre-rasterized images (no InkML parsing)

See PLAN.md for v2 expanded vocab design (183 tokens, 206K samples).

**License**: SJTU weights = academic-only. Retrained weights on CROHME
= CC BY-NC-SA 3.0 (NC). Fine for "buy me a coffee" app: app code is
commercial, weights downloaded separately with NC terms acceptance.
All handwritten math datasets are NC. The C++ inference is clean-room.

---

## June 2026 — OCR feature parity across all surfaces

### PosFormer port merged to main
- `posformer_ocr.cpp` (961 LOC): DenseNet encoder + Transformer decoder
  with Attention Refinement Module (ARM), ported from `feat/posformer-port`
- Wired into unified dispatcher (`MATH_OCR_POSFORMER` enum + all switch blocks)
- Converter: `models/convert-posformer-to-gguf.py`
- Registry: `posformer-crohme` at `cstr/posformer-crohme-GGUF` (CC BY-NC-SA 3.0)
- 57% exact match on CROHME 2014 (best handwritten model)

### General OCR pipeline (detect + recognize) wired everywhere
- **CLI**: `--ocr-det MODEL --ocr-rec MODEL --ocr IMAGE` (new flags)
- **Server**: `POST /ocr` endpoint (detect text regions → recognize each crop)
- **Python**: `CrispOcrPipeline(det_model, rec_model)` — `run()` + `recognize()`
- **Rust**: `OcrPipeline::new()` / `run()` + `MathOcr::recognize_gray()`
- **Flutter/Dart**: `CrispOcrPipeline` class + `OcrResult` + FFI typedefs

### Registry expanded
- Added: pix2tex-mfr, texo-distill, posformer-crohme, dbnet-det,
  trocr-printed, layout-heron (6 new entries, 8 OCR total)

### Stale worktrees cleaned
- Merged and removed: feat/posformer-port, feat/layout-detect-fix,
  feat/layout-parity, feat/ocr-detect
- CrispASR: removed worktree-feat+tts-watermark-metadata,
  worktree-fix-piper-roundtrip

---

## June 2026 — RT-DETRv2 Layout Detection

### Document layout analysis: ResNet-50 + HybridEncoder + deformable decoder
- Architecture: ResNet-50-D backbone + HybridEncoder (AIFI self-attention +
  FPN/PAN with CSP-RepVGG blocks) + 6-layer transformer decoder with
  deformable multi-scale cross-attention (300 queries, 17 classes)
- 14 parity bugs found and fixed via systematic layer-by-layer diff:
  AIFI pos/LN/residual, PAN lateral features, cpu_linear weight convention,
  converter weight transposition (Gemm/Split/Transpose patterns),
  decoder_input_proj Conv convention, valid_mask, query_pos_head architecture,
  bilinear resize, grid_sample alignment
- All encoder stages cos=1.0 with exact input (verified via crispembed-diff)
- Detection score 0.934 on test images (HF reference: 0.955)
- Performance: 21s with BLAS (was 178s without — 8.5x speedup)
- Quantized: F32 161 MB, Q8_0 43 MB (3.7x compression)
- Published: huggingface.co/cstr/layout-heron-gguf (F32 + Q8_0)
- Fully wired: C ABI, CLI (`--layout`), server (`POST /layout/detect`),
  Python (`CrispLayout`), Rust (`CrispLayout`), Dart/Flutter
- Source: docling-project/docling-layout-heron (Apache-2.0, 42M params)

---

## June 2026 — WASM build (math OCR in browser)

### CrispEmbed compiled to WebAssembly via Emscripten
- `build-wasm.sh`: emcmake cmake, CPU-only, SIMD128, MODULARIZE=1
- Output: `crispembed_ocr.js` (62K) + `crispembed_ocr.wasm` (999K)
- `wasm/ocr_wrapper.c`: thin C entry point exposing `wasm_ocr_init`,
  `wasm_ocr_recognize_gray`, `wasm_ocr_recognize`, `wasm_ocr_free`
- Emscripten guards: `model_mgr.cpp` (disable curl/wget),
  `gguf_loader.cpp` (skip mmap, use fread fallback)
- `cmake/FindThreads.cmake`: stub override creates no-op Threads::Threads
  target, avoiding -pthread and SharedArrayBuffer/COOP/COEP requirement
- Integrated into CrispCalc web/PWA: `dart:js_interop` bridge, IndexedDB
  model caching, conditional import selects WASM provider on web
- All existing OCR models work: pix2tex, HMER, BTTR, PosFormer, Texo,
  PP-FormulaNet-L (auto-detected from GGUF architecture tag)
- Tested end-to-end: model load (16.8 MB, 1.5s) + encoder (578 tokens)
  + decoder (201 tokens) → LaTeX output in Node.js

### HuggingFace Space
- `hf-space/`: Docker build (two-stage) + Gradio UI (3 tabs: text
  embeddings, math OCR, health)
- Pattern: C++ `crispembed-server` on :8090 + Gradio on :7860
- Default models: all-MiniLM-L6-v2 (text) + hmer-hw (OCR)
- Tested: cos=0.785 for similar texts, `x² + 1 = 0` → `x ^ { 2 } + 1 = 0`
- Live at https://huggingface.co/spaces/cstr/CrispEmbed

### CI
- `build-wasm.yml`: builds WASM on push/PR, uploads artifacts
- `deploy-hf-space.yml`: auto-deploys `hf-space/` to HuggingFace on push

---

## June 2026 — PP-FormulaNet-L OCR (181M params)

### Full in-graph ViT encoder with decomposed RPE
- **Full ggml graph encoder**: all 12 ViT layers run as single ggml graphs
  with attention + decomposed relative position bias entirely in-graph
- Window batching: all 16 windows × 12 heads processed as one batch dimension
  via reshape + permute, enabling efficient batched matmuls
- Decomposed RPE in-graph: two matmuls (rp_h@Q, rp_w@Q_permuted) with
  broadcast-add to attention scores (Granite NLE pattern)
- LN ordering fix: for windowed layers, LayerNorm1 applied on CPU before
  window partition to match HF's LN→pad→QKV ordering. Prevents LN(0)=bias
  corruption of padding tokens (cos jumped from 0.973 to 0.9999)
- Per-layer parity: cos ≥ 0.99997 on all 12 layers
- Performance: 53s encoder with BLAS+Q8_0 (60s F32, was 77s hybrid, ~500s scalar)

### Printed math OCR: SAM-ViT encoder + MBart decoder
- New architecture: SAM-style ViT encoder (12 layers, 768d, 12 heads)
  with windowed attention (ws=14) + global attention (layers 2,5,8,11)
  and decomposed relative position bias
- Neck: Conv1x1 + LayerNorm2d + Conv3x3 + LayerNorm2d (768 → 256)
- Multi-modal projector: 2× Conv3x3(stride=2) + 2× Linear (256 → 512)
  Output: (144, 512) sequence for decoder
- MBart PRE-LN decoder: 8 layers, 16 heads, d_model=512, FFN=2048
- 768x768 RGB input, UniMERNet preprocessing pipeline
- Encoder parity: cos=0.999962 vs HuggingFace reference (F32)
- Quantization: F32 (692 MB), F16 (347 MB), Q8_0 (241 MB, cos=0.999940),
  Q4_K (122 MB, cos=0.997595) — all produce identical decoded LaTeX
- Smart Q8_0: critical tensors (embeddings, LN, rel_pos, lm_head) in F16
- Auto-detected from GGUF metadata (`general.architecture = ppformulanet_l`)
- Wired into unified `--ocr` CLI, C ABI, model registry, CrispCalc Dart catalog
- Source: PaddlePaddle/PP-FormulaNet-L_safetensors (Apache-2.0)
- New GGUF loader helper: `kv_i32_array()` for int32 metadata arrays

### Full-stack wiring
- HTTP server: `POST /math/ocr` endpoint (`--ocr` flag, stb_image load, JSON response)
- Python bindings: `CrispMathOcr` class with `recognize()` and `recognize_gray()`
- Updated contributing.md with server + Python binding steps
- Updated public C header comments to list all supported architectures

## June 2026 — PPFormulaNet-S / Texo-Distill OCR

### Printed math OCR: HGNetv2 + MBart decoder (20M params)
- New architecture: HGNetv2 CNN encoder (StemBlock, 4 HG_Stages, LightConvBNAct)
  + MBart Transformer decoder (2 layers, 16 heads, 384 d_model)
- Conv-BN folding in GGUF converter: all BatchNorm absorbed into preceding Conv2d
- CPU-side CNN forward pass for encoder (all standard ops: conv2d, relu, maxpool, concat)
- MBart PRE-LN decoder: LayerNorm before attention/FFN, residual skips LN
- UniMERNet preprocessing: aspect-ratio-preserving resize + black pad + grayscale
  normalize (mean=0.7931, std=0.1738)
- ODR fix: renamed internal dec_layer → ppfn_dec_layer to avoid linker collision
  with decoder_embed_internal.h
- Added `--ocr` CLI flag for unified auto-detection (pix2tex/hmer/bttr/ppformulanet)
- Quantized: F16 (39 MB), Q8_0 (22 MB, identical quality), Q4_K (13 MB, degraded)
- GGUF models published: huggingface.co/cstr/texo-distill-gguf
- Diff regime: encoder cos=1.000000, decoder verified via layer-by-layer debug traces
- Source: Texo (AGPL-3.0) distilled from PP-FormulaNet-S (Apache-2.0)
  trained on UniMER-1M (CC-BY-4.0)

## June 2026 — Nomic v2 MoE Encoder

### Mixture-of-Experts encoder support
- First MoE embedding model: nomic-embed-text-v2-moe (8 experts, top-2, GELU)
- Fully in-graph MoE routing: ggml_top_k + ggml_get_rows + ggml_mul_mat_id
- Mixed architecture: odd layers = MoE FFN, even layers = dense GELU FFN
- Converter handles GPT2-style config (NomicBERT extends GPT2Config),
  per-layer MoE/dense auto-detection, expert weight stacking [n_exp, dim, dim]
- Fixed missing Wqkv + out_proj biases (present in v2-moe but not v1.5)
- Exact erf-GELU activation (NomicBERT uses nn.GELU(approximate='none'))
- Parity: cos=1.000000 vs HuggingFace on all test texts
- Quantized variants: F16 (1344 MB), Q8_0 (1122 MB, cos=0.9996), Q4_K (1095 MB, cos=0.966)
- GGUFs published to cstr/nomic-embed-text-v2-moe-GGUF on HuggingFace
- Extended parity_layers_bert.py harness with --arch nomic (QKV split, MoE expert tensor diff)
- Added CRISPEMBED_DUMP_LAYERS env var for per-layer intermediate tensor dumps

---

## June 2026 — LoRA Hot-Swap, Batched Decoder, Face Pipeline

### LoRA adapter hot-swap
- Runtime switching between Jina v5 per-task LoRA adapters (retrieval,
  classification, clustering, text-matching) without re-loading the model
- Pre-compute approach: `W' = W + (α/r)·B@A` on CPU at switch time (~10-50ms)
- Converter `--lora-mode=separate` stores base weights + per-adapter A/B
  tensors (F16) in a single GGUF with metadata
- Lazy base weight snapshot with dequant→merge→requant for quantized models
- C API: `crispembed_set_lora/get_lora/list_lora`
- CLI: `--lora NAME`, `--list-lora`
- Python: `set_lora()`, `lora` property, `list_lora()` on CrispEmbed

### Batched decoder graph
- Single ggml graph compute for N decoder texts (was: N sequential passes)
- Block-diagonal causal mask (text i cannot attend to text j), padding
  positions get self-attention to prevent softmax NaN
- Independent RoPE positions per text, pad to T_max
- Per-text last-token / mean pooling after graph compute
- **3.3x speedup** on batch of 4 (Jina v5 nano, CPU)
- Parity: cos >= 0.999 vs sequential encoding on all test texts

### Face pipeline Python completion
- `CrispFacePipeline` exported in `__init__.py`
- `from_registry()` class methods on `CrispFace` and `CrispFacePipeline`
  for auto-download by registry name
- Unit tests (`tests/test_face_python.py`): 12 tests covering detection,
  recognition, pipeline, match, edge cases
- Example script (`examples/face_search.py`): index faces from directory,
  query by image, top-K cosine matches

### BTTR beam search decoder
- Beam search with configurable width (default 5) for BTTR handwritten
  math OCR — improves exact-match accuracy over greedy decoding

### Windows CI fix
- `M_PI` undefined on MSVC: added `#ifndef M_PI` portable fallback in
  `bttr_ocr.cpp`

---

## June 2026 — CLIP/SigLIP Vision + Text, YuNet, HMER/BTTR OCR

### YuNet lightweight face detection
- 228 KB GGUF (vs SCRFD 16 MB), ShuffleNetV2 backbone, 640×640 input
- IoU 0.99 vs OpenCV FaceDetectorYN, score diff < 0.01, landmark diff < 2px
- Converter unchanged (existing `convert-face-to-gguf.py` handles YuNet's ops)
- Key gotcha: ggml Transpose op does real 2D transpose for n_dims==2 tensors,
  requiring different spatial indexing for 1-channel (cls/obj) vs multi-channel
  (bbox/kps) outputs
- Uploaded to `cstr/yunet-GGUF`, in auto-download registry

### CLIP text encoder (new module)
- `clip_text_embed.{h,cpp}`: pre-LN transformer with causal mask, EOS pooling,
  text_projection, BPE tokenizer embedded in GGUF
- `convert-clip-text-to-gguf.py`: extracts text tower + tokenizer from HF CLIP
- C API (`crispembed_clip_text_*`), Python wrapper (`CrispClipText`), server
  `/clip/text` endpoint
- Cross-modal text↔image search works end-to-end
- Uploaded: `cstr/clip-text-base-GGUF` (244 MB), `cstr/clip-text-large-GGUF` (474 MB)

### CLIP / SigLIP vision models
- Fixed `vit_embed.cpp`: CLS token prepend, CLS pooling for CLIP, quick_gelu
  via FP32 ggml primitives, attention pooling residual skip connection
- Converted and uploaded 6 vision GGUFs:
  - `cstr/clip-vit-base-patch16-GGUF` (329 MB, MIT)
  - `cstr/clip-vit-large-patch14-GGUF` (1.2 GB)
  - `cstr/clip-vit-large-patch14-336-GGUF` (1.2 GB)
  - `cstr/siglip-large-256-GGUF` (1.2 GB, Apache 2.0)
  - `cstr/siglip-so400m-patch14-384-GGUF` (1.6 GB)

### Handwritten math OCR (HMER + BTTR)
- HMER: DenseNet-121 encoder + GRU attention decoder (with coverage).
  Source: whywhs/Pytorch-HMER (code: MIT), trained on CROHME 2016
  (CC BY-NC-SA 3.0). Weights inherit NC.
  112 LaTeX tokens, ~6.8M params, ~4-5 MB Q4_K.
  `hmer_ocr.{h,cpp}`, `convert-hmer-to-gguf.py`. CLI: `--hmer FILE`.
  Auto-detect image polarity and invert if needed. Dequant support.

- BTTR: DenseNet encoder (growth=24, 3 blocks) + Transformer decoder
  (3 layers, 8 heads, d=256). Source: Green-Wood/BTTR (code: MIT),
  trained on CROHME 2014 (CC BY-NC-SA 3.0). Weights inherit NC.
  113 LaTeX tokens, 6.5M params. 49.2% raw / 49.8% parsed on CROHME.
  `bttr_ocr.{h,cpp}`, `convert-bttr-to-gguf.py`.
  BN folded into conv, fused QKV preserved.

### SFace quantization (conv2d quant support)
- Converter flattens 4D conv weights to 2D [OC, IC*KH*KW] for quantization
- Runtime: dequant Q8/Q4→F32, reshape back to 4D, cast to F16 for ggml_conv_2d
- SFace results: F32=37MB, Q8_0=10MB (cos=0.9999), Q4_K=6MB (cos=0.974)
- Uploaded to `cstr/sface-GGUF` (F32 + Q8_0 + Q4_K)
- Same pattern applies to AuraFace and SCRFD (reconverted with flat conv)
- AuraFace: 249 MB (Q8_0 only compresses FC → 212 MB; conv rows too small for Q8_0)
- SCRFD: 17 MB (minimal Q8_0 gain — detection heads are small)
- AuraFace F16: 249→125 MB (2.0x, lossless — conv2d casts to F16 anyway)
- SCRFD F16: 17→8 MB (2.0x, lossless)
- Added F16 support to quantizer (Q8_0/Q4_K need row÷32; F16 has no alignment limit)

### Face model quantized graph replay fixed
- YuNet F16/Q8_0 inference via graph replayer now works (was crashing)
- Three fixes: (1) parse Conv group attrs before 2D→4D reshape for
  correct depthwise IC detection, (2) handle ggml_n_dims returning 2
  for 4D weights with trailing 1s via element count validation,
  (3) dequant Q→F32 before F16 cast (ggml only supports Q→F32)
- Q8_0 detection matches F32 with minor quantization drift (conf 0.731 vs 0.749)
- Old-style 4D-weight GGUFs and new-style 2D-flattened GGUFs both work
- YuNet parity verified: sub-pixel match vs OpenCV FaceDetectorYN on both
  single-face and multi-face images (x/y/w/h diff < 0.5px, conf diff < 0.01)
- Raw tensor cos vs ONNX (0.35-0.85) is a false alarm — planar (ggml) vs
  interleaved (ONNX) layout of the same correct data; decoded coords match

### SigLIP text encoder verified
- cos=1.000000 vs HuggingFace on all test texts
- SentencePiece BPE vocab decoded correctly by Viterbi unigram algorithm
- Key finding: SP BPE training doesn't change inference algorithm — Viterbi works

### Model registry expansion
- 13 new auto-download entries: 2 face detection (yunet, scrfd-det-10g),
  2 face recognition (auraface-v1, sface), 8 vision/text (CLIP + SigLIP),
  1 SigLIP-base
- Total registry: ~58 models

### Vision parity fixed: cos 0.8 → 1.0
- Root cause: patch embedding `ggml_permute(2,1,0)` produced column-major
  spatial ordering (t = ow*OH + oh), but HuggingFace uses row-major
  (t = oh*OW + ow via flatten(2)). Every patch beyond (0,0) got the
  wrong position embedding.
- Fix: `ggml_permute(1,2,0,3)` produces [D, OW, OH] which flattens to
  row-major matching HF. Per-layer cos goes from ~0.3 to 1.000000.
- Final embedding cos = 0.9998 vs HuggingFace (SigLIP-base-384)
- CLIP ViT also verified: cos=1.000000, max_diff=0.000001 (clip-vit-base-patch16)
- Was NOT "FP32 non-associativity" as previously hypothesized — it was
  a simple permutation index bug that scrambled patch positions

---

## v0.7.0 — May 2026

### Registry status

45 models in registry, 151 GGUF variants published on HF:
25 encoder models + 11 decoder models + 12 rerankers + 1 SPLADE + 2 multimodal.
Typical per-model: F32 + Q8_0 + Q4_K; about a dozen also have Q5_K / Q6_K / F16.

Key parity results (cos vs HuggingFace reference):

| Model | Type | Dim | CosSim |
|-------|------|-----|--------|
| all-MiniLM-L6-v2 | BERT | 384 | 1.000000 |
| bge-small/base/large-en-v1.5 | BERT | 384/768/1024 | 1.000000 |
| gte-base/large-en-v1.5 | GTE | 768/1024 | 1.000000 |
| nomic-embed-text-v1.5 | NomicBERT | 768 | 1.000000 |
| nomic-embed-text-v2-moe | NomicBERT MoE | 768 | 1.000000 |
| mxbai-embed-large-v1 | BERT | 1024 | 1.000000 |
| all-mpnet-base-v2 | MPNet | 768 | 1.000000 |
| multilingual-e5-small/base/large | XLM-R | 384/768/1024 | 1.000000 |
| snowflake-arctic-embed-m/l | BERT/XLM-R | 768/1024 | 1.000000 |
| bge-m3 (dense+sparse+ColBERT) | XLM-R | 1024 | 1.000000 |
| splade-pp-en-v1 | BERT SPLADE | 768 | 1.000000 |
| granite-embedding-278m/107m | XLM-R | 768/384 | 1.000000 |
| gte-modernbert-base | ModernBERT | 768 | 0.9999 |
| pixie-rune-v1 | XLM-R | 1024 | 0.999993 |
| octen-0.6b | Qwen3 | 1024 | 0.999891 |
| octen-8b | Qwen3 | 4096 | 0.965 (Q4_K vs bf16 HF) |
| qwen3-embed-4b | Qwen3 | 2560 | 0.974 (Q4_K vs bf16 HF) |
| harrier-0.6b / harrier-270m | Qwen3/Gemma3 | 1024/640 | 0.999959/948 |
| jina-v5-nano/small | Qwen3 | 1024 | 0.999941 |
| bge-reranker-v2-m3 | XLM-R reranker | - | verified |
| ms-marco-MiniLM-L-6/12-v2 | BERT reranker | - | verified |

### Optimizations completed

- ggml_backend_sched GPU dispatch (encoder + decoder full-graph)
- All 45 models quantized (Q8_0 + Q4_K) and uploaded to HuggingFace
- Graph/work buffer reuse: 27.8 texts/s server throughput (gte-small)
- Matryoshka dimension truncation via -d N flag
- BLAS/MKL/CUDA/Vulkan/Metal build support
- Windows build scripts
- C++ quantizer with K-quant fallback chain
- QKV weight fusion (1 matmul vs 3 per layer)
- Flash attention with optional position bias mask
- ggml graph decoder for math OCR (27x speedup over scalar)

### Bindings and platforms

| Binding | CrispEmbed | CrispASR |
|---|---|---|
| C API | Complete | Complete (whisper.h) |
| Python (ctypes) | Complete + tested | Complete + tested |
| Rust (crate) | Complete + tested | Complete + compiled |
| Dart/Flutter (FFI) | Complete | Created |
| iOS (Metal) | CI green | CI green |
| Android (NDK) | CI green (arm64/armv7/x86_64) | CI green |
| Windows | CI green | CI green |
| macOS (Metal) | CI green | CI green |
| Linux | CI green | CI green |

### CrispEmbed advantages over fastembed-rs

- **ColBERT multi-vector** retrieval (fastembed-rs doesn't have it)
- **Matryoshka dimension truncation** (fastembed-rs doesn't have it)
- **GGUF quantization** (Q8_0, Q4_K — smaller than ONNX INT8/INT4)
- **9.5x faster on MiniLM-L6** (most popular embedding model)
- **GPU dispatch** via ggml_backend_sched (CUDA/Metal/Vulkan)
- **Ollama-compatible** server with 4 API dialects
- **Flutter/Dart** wrapper for mobile apps
- **iOS/Android** build scripts with full CI
- **20MB binary** vs ~500MB Python+ONNX environment

### Commercially permissive stack (no NC restrictions)

The full pipeline uses only Apache 2.0 / MIT models:
- Text: any CrispEmbed encoder model (BERT/XLM-R/etc.)
- Image: SigLIP (Apache 2.0) or CLIP (MIT)
- Face detection: SCRFD (Apache 2.0) or YuNet (Apache 2.0)
- Face recognition: AuraFace-v1 512-D (Apache 2.0) or SFace 128-D (Apache 2.0)
- Face landmarks: MediaPipe FaceLandmarker (Apache 2.0)
- Audio: CrispASR (our own, Apache 2.0)

### Resolved known issues

1. **NomicBERT** — Root cause: gate/up weights (fc11/fc12) were swapped in old GGUF;
   also needed Ollama tensor name fallback. F32 cos=1.0, Q8_0 cos=0.998.

2. **EmbeddingGemma-300m** — cos=1.0000 F32, 0.9998 Q8_0, 0.9954 Q5_K.
   Root causes: missing `is_bidirectional=1`, wrong pooling, BPE merges not loading,
   Dense layers being quantized. All fixed.

3. **Jina v5 nano/small** — Models use task-specific LoRA adapters; converter now
   merges `retrieval` adapter. Nano F32 cos=1.0, Small F32 cos=0.9999.

4. **all-mpnet-base-v2** — Old GGUF was missing `relative_attention_bias.weight`.
   Reconverted with bias tensor. cos=0.987-0.999.

5. **gte-modernbert-base** — Validation wrongly required `ln1` for pre-LN models.
   Fixed validation. cos=0.9999.

6. **DeBERTa-v2 disentangled attention** — c2p/p2c relative position bias with
   log-bucket encoding now fully implemented. mxbai-rerank-xsmall-v1 and
   mxbai-rerank-base-v1 both working.

7. **Full regression sweep (2026-05-17)**: 34 models tested, all pass. 5 models
   fixed and re-uploaded to HF.

---

## May 2026 — Multimodal & Vision

### BidirLM-Omni (text + audio + image)

- [x] Text path through `decoder_embed.cpp` (cos >= 0.999 vs HF bf16)
- [x] Audio path through `bidirlm_audio.cpp` + crisp_audio (cos = 0.995 vs HF)
- [x] Vision tower in `bidirlm_vision.cpp` (cos >= 0.999 vs HF bf16)
- [x] DeepStack injection + 3D interleaved-MRoPE (cos = 0.998903 vs HF bf16)
- [x] `crispembed_encode_text_with_image` C ABI + Python wrapper
- [x] `crispembed_encode_with_image_ids` (pre-tokenized variant for parity tests)
- [x] CLI `--image FILE` + `--image-raw patches.f32 --grid-thw T,H,W`
- [x] Decoder `ggml_backend_sched` initialization
- [x] Memory-efficient lite parity test
- [x] In-process C++ image preprocessor (smart_resize + Catmull-Rom bicubic)
- [x] BPE special-token handling for Qwen-style tokens
- [x] Stale-GGUF fallbacks for missing metadata
- [x] Image batching in `encode_text_with_image`

### Phase 8: Vision — Image Embeddings, Face Detection & Recognition

#### 8A. SigLIP Image Embedding (DONE)

cos=0.996 vs HF. Uploaded to cstr/siglip-base-GGUF.
- GGUF converter, ViT forward path, image preprocessing
- CLI: `crispembed -m siglip-base.gguf --image photo.jpg`

#### 8B. Face Detection — SCRFD (DONE)

Scores match ONNX Runtime. Uploaded to cstr/scrfd-det-10g-GGUF.
- Generic ONNX graph replayer (Conv, ReLU, Add, Pool, Resize, Concat, Sigmoid)
- FPN + multi-scale detection heads + NMS
- Letterbox preprocessing + coordinate scaling
- C API, Python, Rust, Dart wrappers

#### 8C. Face Recognition — AuraFace + SFace (DONE)

cos=0.9999 vs ONNX for both models.
- BN folding/precomputation, 512-D/128-D embeddings
- Full detect-align-encode pipeline
- C API, Python, Rust, Dart wrappers
- Server API: `/detect`, `/face` endpoints

---

## April 2026 — RAG Feature Parity

- [x] Full Python/Rust/Dart wrapper: sparse, ColBERT, reranker, set_dim, set_prefix
- [x] Bi-encoder reranking API (Python + Rust + Dart): cosine similarity ranking
- [x] Prompt prefix system (C/Rust/Python/Dart): auto-prepend query/passage prefixes
- [x] 21 verified embedding models (cos >= 0.999 vs HuggingFace)
- [x] 5 reranker models (bge-reranker-base, ms-marco L6/L12, mxbai-rerank xsmall/base)
- [x] 27 HuggingFace repos with GGUF models + README cards
- [x] RAG retrieval quality benchmark (tests/bench_rag.py): MRR@10, NDCG@10, Recall@k
- [x] Reranking benchmark (tests/bench_rerank.py): cross-encoder vs bi-encoder
- [x] Head-to-head benchmark vs FastEmbed:
  - MiniLM-L6: CrispEmbed **9.5x faster** single, **10.8x faster** batch
  - BGE-small: FastEmbed 1.7x faster (ONNX graph JIT optimization)
  - Arctic-M: tied on batch (126 vs 127ms)
  - cos = 0.999999-1.000000 cross-engine on all models
- [x] Demo apps (Python + Rust) for both CrispEmbed and CrispASR

---

## May 12, 2026 — Face Pipeline Complete

Full detect -> align -> encode pipeline for face recognition.

### RAG parity: prompt prefixes + new models

- Added auto-prefix system: BGE, E5, Nomic, Jina models get query/passage
  prefixes auto-applied.
- Converted 3 new models: SPLADE-PP-en-v1, granite-embedding-278m/107m.
- Model registry: 47 models total.

### SCRFD preprocessing + anchor decode fixes (3 bugs)

1. RGB-BGR channel swap
2. Anchor center offset (integer grid, no 0.5 offset)
3. Top-left placement (not centered letterbox)

### SCRFD anchor decode fix (data layout mismatch)

Channel-last vs interleaved indexing. After fix: detection counts match
InsightFace exactly on all test images.

### Face alignment fix (4 sign errors in normal equations)

After fix: alignment matches InsightFace `norm_crop` with MAE=0.00.
Per-face embedding cos=0.994-0.999 vs InsightFace ArcFace.

### Pipeline implementation

- `cnn_embed::detect_file()` — letterbox resize, coordinate scaling
- `cnn_embed::encode_aligned()` — 5-point landmark similarity transform + encode
- `cnn_embed::face_pipeline()` — detect -> align -> encode in one call
- CLI, C API, Server API, Python/Rust/Dart wrappers all complete

### Models converted

- SCRFD-10GF (16.1 MB)
- w600k_r50 ArcFace (166 MB)
- AuraFace-v1 (248.6 MB)
- SFace (36.8 MB)

---

## May 11-12, 2026 — Vision Models & Parity Fixes

### SigLIP image embedding
- Converter: `models/convert-siglip-to-gguf.py`
- Forward path: `src/vit_embed.cpp` — cos=0.996 vs HF mean-pool
- Native `--image` flag with stb_image preprocessing
- Uploaded: cstr/siglip-base-GGUF

### Face detection (SCRFD)
- Generic ONNX graph replayer in `src/cnn_embed.cpp`
- FPN backbone + multi-scale detection heads
- Anchor decode + NMS at strides 8/16/32
- Semicolon delimiter for ONNX tensor names with commas

### Face recognition (SFace + AuraFace)
- SFace MobileFaceNet: cos=0.9999 vs ONNX, 128-D
- AuraFace ResNet-100: cos=0.9999 vs ONNX, 512-D
- BN folding/precomputation at converter time
- PReLU: relu(x) + slope * (x - relu(x))
- Conv F32->F16 auto-cast for ggml_conv_2d

### Text model parity fixes (35 models)
- GTE v1.5: post-LN + GeGLU half swap + NTK RoPE
- Jina reranker v2: post-LN + position offset
- NomicBERT: SwiGLU fc11/fc12 swap
- Ollama format: auto-strip prefix, dual metadata keys, pooling type mapping

---

## Apr 12, 2026 — v0.1.0 Release

30-commit session: FastConformer extraction, granite 3.x support,
NeMo FC-CTC, omniASR, Silero LID, CI, Windows, Vulkan, benchmarks.
Tagged v0.1.0 release with multi-platform binaries.

## August 3, 2026 — active-work board archived (36+ completed rows moved out of PLAN.md)

PLAN.md's "Active work in flight" table had grown to 56 rows, 51 of them
completed. The table exists so parallel sessions can see what is *claimed*;
once a row lands it is noise that hides the five things actually in flight.
Archived verbatim below so nothing is lost, with the open follow-ups they
contained promoted into PLAN.md's "Next actions" instead of being buried in a
COMPLETED row's prose.

| Since | Branch / worktree | Task | Status |
|-------|-------------------|------|--------|
| 2026-08-03 | `chore/ai-act-imageroot` / `.codex/worktrees/chore-ai-act-imageroot` | **Picked: `--image-root` confined 13 of 33 endpoints, not all of them — the row below (round-3) states the opposite in good faith and was wrong.** A local `extract_image_path` lambda at `server.cpp:2093`, left over from before the confinement work, **shadowed** the file-scope `extract_image_path()` for every handler registered after it. Both names resolve, neither warns, and the local one calls `json_extract_strings` without `path_within()` — so confinement covered the first 13 endpoints (`/face` and `/detect` among them, which is why the biometric surface was never exposed) and silently missed the next 20: all 8 SR engines, `/restormer`, `/scunet/denoise`, `/instructir/restore`, `/adair/restore`, every `/preprocess/*`, `/scan/split`, `/scan/content`, `/ocr/document`. Confirmed by brace-depth analysis that the lambda stays in scope for all 20, and present on `origin/main` plus every `chore/ai-act-*` branch. Impact is an unauthenticated arbitrary-image READ that a deployer believed they had closed, with exfiltration via the SR/restore response (which returns the image base64-encoded); `output`/`file`/`model` were never affected because they call `extract_path_field` by its own name. Sharpest illustration: inside `/preprocess/dewarp` the write destination (line 2249) was confined while the read source (2246) was not. **Fix:** deleted the lambda, left a comment naming the hazard, and rerouted `extract_path_field` through core_json's depth-1 finder so confinement and parsing agree on which field is "the image" — the file-scope version previously used a bare `body.find("\"image\"")`, so a nested decoy `{"meta":{"image":"/a"},"image":"/b"}` made the server read a different path than a validating proxy in front would. **Why it survived three audit rounds:** `tests/test_image_root.py` only ever probed `/detect`, which sits above the shadow, so the test could not have caught it — it now probes `/scan/split` and `/scan/content` too (no model needed), i.e. both sides of the boundary. **Carry-forward:** the previous four findings came from re-checking claims already written down; this one came from re-checking a *fix* already verified by a *test*. A passing test proves the endpoint it names, not the sentence in POLICY it was written to support. Also in this pass: POLICY §1 now states we are the **provider** and not merely the deployer of the Space and WASM demo (Art. 3(3); Art. 50(1)/(2) are provider duties, (3)/(4) deployer ones, and Art. 2(12) does not reach Art. 50), and records that the Art. 2(12) FOSS reading covers our MIT code but not a system assembled with a `cc-by-nc*` or vendor-restricted checkpoint. Coordination: `examples/server/server.cpp` + `tests/test_image_root.py` + POLICY/PLAN only; no engine, graph or orchestrator code. **DONE, and the bug was MEASURED rather than inferred** — I rebuilt the unfixed `server.cpp` specifically to avoid shipping a fix for a defect I had only read. Same build, same flags, `--image-root` set: `POST /scan/split {"image":"<outside-root>/secret.png"}` returns `{"pages": 1, "width": 640, "height": 640}` with **0** rejections logged before, and `{"error": "missing 'image' field"}` with the rejection logged after; `/scan/content` identical. The nested decoy `{"meta":{"image":"<in-root>"},"image":"/etc/hosts"}` is refused after the depth-1 change, i.e. the server and a validating proxy now agree on which field is "the image". In-root reads still serve, and behaviour is unchanged when `--image-root` is unset (`path_within` returns true on an empty root). Landed as `54aeaecb` on `chore/ai-act-imageroot`. | **COMPLETED** |
| 2026-08-03 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Followed the crispembed-diff protocol properly and it found a fault in my measurements, not the model.** Two rule-2b violations by the tool itself: `Report` always carried `cos_global`/`mine_norm`/`ref_norm` and **no print site showed them**, and I quoted `cos_min` — a per-row worst case — as tensor parity all session. All internvl2/test print sites now emit `cos_min`, `cos_glob`, `max_abs`, `|mine|`, `|ref|`. **More intermediates:** the reference had only ever been dumped `--max-llm-layers 4`, which also skips `llm_output_norm` and `llm_logits` — stages the test could already compare but had never been given data for, leaving the harness 20 layers short of the decision boundary. Re-dumped to **54 stages** (`ref-full.gguf`, 108 MB). **Result — f16 is EXACT to the logits:** every one of 54 stages passes, `cos_min` included, `llm_logits` cos 1.000000 / max_abs 0.000069 / |mine| 1604.5433 vs |ref| 1604.5439. No code defect in this engine for this model. Shipped q8_0 reaches the logits at cos_glob **0.998919**, magnitudes 0.13% apart, with a smooth monotonic decline and **no discontinuity** — every 'jump' I chased (`vis_layer_12`, `llm_layer_1`) was a `cos_min` artifact. **Corrections pushed:** h2ovl-800m q8_0 is NOT degraded (cos_glob 0.999975, I called it 'cratering'); the 'sign is what survives' finding is **withdrawn** (built on two per-row worst cases); the q4_k withdrawal stands but for cos_glob 0.994→0.968 over 4 of 24 layers plus wrong decoded output, not 'anti-correlated'. ⚠ **Left for a decision, not taken unilaterally:** `is_pass()` keys on `cos_min`, so nearly every stage prints FAIL while globally excellent — a gate that always cries wolf. `crispembed_diff.h` is shared by every engine; re-keying it on one model's evidence is the mistake already made twice today. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked: `--image-root` does not confine what POLICY §4 implies it does.** The sentence is literally true — it confines every `{"image": ...}` READ — but it reads as a filesystem sandbox for the server, and three request fields bypass it entirely: **`/preprocess/dewarp` `"output"` is an arbitrary file WRITE** (fopen "wb" on a client-supplied path, so any file the process can write is creatable/truncatable — strictly worse than the read I originally fixed); **`/preprocess/tps-dewarp` `"model"` loads a client-supplied path as a GGUF and executes it as a ggml graph**, which is the exact hazard the SHA-256 pinning work is premised on; and `/pdf/dpi` `"file"` is an unconfined read. Fix: generalise the confinement helper to any path-valued field and apply it to `image`/`output`/`file`; add a separate `--model-root` for model paths, since a model legitimately lives outside an image directory and folding it into `--image-root` would be the wrong shape. Then state in POLICY §4 exactly what is confined rather than leaving the impression of a sandbox. Coordination: `examples/server/server.cpp` + POLICY/docs only; no engine, graph or orchestrator code. **DONE.** Confinement helper is field-agnostic now and applied to `image`/`output`/`file`; `model` gets its own `--model-root` rather than being folded in — a model legitimately lives outside an image dir, and conflating a code-execution surface with a data one is the wrong shape even where the directories coincide. `tests/test_image_root.py` grew the two cases that matter and asserts on the FILESYSTEM, not the response: after posting an output path outside the root, no file exists there — a 200 with an error body would have proven nothing. 8/8 pass. POLICY §4 now enumerates what each root covers instead of implying a sandbox. **Pattern worth noting across the last four findings:** each came from re-checking a claim I had already written down (marking coverage twice, harness pinning, this one), and each time the code was right where I had looked and wrong where I had not — an argument for the coverage tests now in CI over more careful greps. | **COMPLETED** |
| 2026-08-03 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Artifact audit + h2ovl-800m brought up to standard.** Audit found three gaps, all closed: (1) the 2b model repo carried a **4-vision-layer partial reference** (28 MB) alongside the full 24-layer one in fixtures — the partial could not have found today's vision drift, so it was replaced rather than left as a trap; (2) the **800m had no reference at all**, so its parity had never been measured, only its output eyeballed; (3) the 800m had no quality-tier quant. Ran the full regime for the 800m locally (f16 1853 MB, small enough): convert → bake ref (32 stages) → quantize → per-stage diff → decoded output. **f16 exact** (`llm_embed`/`llm_layer_0..3` all 1.000000, `vis_proj_output` 0.999701; `vis_layer_23` 0.998665 and unshuffle 0.998199 are f16-vs-f32 rounding, same class as the 2b). **q8_0 + vision-F16 reproduces the f16 vision numbers exactly** — the new quant rule confirmed on a second checkpoint. **Finding worth carrying:** the synthetic probe does **not** track decoded quality. 800m q8_0 reads `llm_layer_2 = +0.494781` and **transcribes at 1764 chars**; 2b q4_k reads `−0.268615` and emits confident nonsense. The distinction that survives is **sign** — inverted vs degraded-but-aligned. A per-stage threshold alone would have rejected a good artifact here. Recorded as the in-repo counterexample for HARD RULE #3. **Registry deliberately unchanged for the 800m** — stays on q4_k (676 MB, transcribes, fox exact) against q8_0's 1175 MB, because this is the edge/WASM model; q8_0 published as a tier, not promoted. Artifacts: 2b repo = f16 + q8_0(vision-F16) + full ref, q4_k withdrawn; fixtures = refs for **both** checkpoints. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked: a real gap in my OWN provenance work, found by re-verifying the claim rather than trusting it.** **12 server endpoints return unmarked AI-processed images** — `/text/sr`, `/pan/sr`, `/hat/sr`, `/dat/sr`, `/safmn/sr`, `/esrgan/sr`, `/swinir/sr`, `/tbsrn/sr`, `/restormer`, `/scunet/denoise`, `/instructir/restore`, `/adair/restore`. They base64 RAW RGB bytes into a JSON field, so they never touch `core_imgout::emit` and my earlier coverage grep (`stbi_write`/`P6`/`P5`) missed all of them. These are precisely the engines POLICY §5 is about — the ones that SYNTHESISE detail — so "every image CrispEmbed returns to you is marked" was false for the highest-risk surface in the document. Fix: route through `emit_to_string` so the base64 payload is a marked PNG, add an explicit `"format"` field so clients can tell what they got, keep raw under `CRISPEMBED_IMAGE_FORMAT=ppm` as the back-compat escape, and factor the 12 near-identical base64 blocks into ONE helper — duplication is exactly what caused the temp-file defect I fixed earlier this branch. Coordination: `examples/server/server.cpp` only; no engine, graph or orchestrator code; no overlap with the five other active branches. **DONE.** All 12 now encode through `emit_to_string`, so the base64 payload is a marked PNG (plus a C2PA manifest when an identity is configured); each response gained an explicit `"format"` field (`png`, or `raw` under `CRISPEMBED_IMAGE_FORMAT=ppm`) so clients can tell rather than infer. The 12 near-identical base64 loops are one helper now — duplication is exactly what produced the temp-file defect earlier on this branch. Verified against a LIVE server: `/adair/restore` returns `format=png`, the payload has the PNG signature, PIL decodes 96x96 RGB, and the chunk records `engine=adair` with `digitalSourceType=algorithmicallyEnhanced`. Documented in `docs/provenance.md` including the response-shape change. | **COMPLETED** |
| 2026-08-03 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Vision-stage parity raised to the f16 ceiling — and the answer was precision, not a bug.** Bisected the encoder: the reference carried `vis_layer_0..23` + `vis_pixel_unshuffle` since it was baked and nothing ever compared them, so the gap between `vis_patch_embed` 1.000000 and `vis_proj_output` 0.912992 had 24 layers to hide in. **At f16 every stage passes** (`vis_layer_*` 1.000000→0.999902, unshuffle 0.999691, proj 0.999974, `llm_layer_*` 1.000000) ⇒ the port is exact. Two apparent bugs disproved: the `vis_layer_12` discontinuity is quantization (f16 smooth through it), and unshuffle 0.380 is **not** a layout mismatch (0.999691 at f16). **Fix:** hold the internvl2 vision tower at F16 for the Q8_0 target — proj `0.912992 → 0.999974`, unshuffle `0.380373 → 0.999691`, every vision stage PASS on CPU, +13% size, page still transcribes. Decoder stays Q8_0 (output correct; F16 = the 4.4 GB file). **Scoped after measuring the sibling:** arch-wide it took `internvl2-1b` 758 → 1135 MB, inflating the edge/WASM model 1.5x — now gated on `ftype == Q8_0`, verified 0 conversions on edge q4_k and 98 on h2ovl q8_0, with `CRISPEMBED_QUANTIZE_NO_VISION_F16=1` to bisect. **Second rule this session narrowed after a sibling check** — the pattern is a rule from one checkpoint applied to a family with different goals. **Shipped:** q8_0 replaced in place (2591566112 B, sha `497cd047…`, verified byte-identical local↔remote), card carries the per-stage table, registry size 2592 MB, pins regenerated **242/0**, `--list-models` correct. ⚠ **Mishap worth recording:** a 10-min tool timeout killed a chained `cp && upload` mid-copy, leaving a truncated `h2ovl-mississippi-2b-q8_0.gguf` locally. Caught by checking size+digest before trusting it; source intact, nothing published half-written. Upload direct from the source with `path_in_repo` instead of copying first. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** provenance on TEXT output, the analogue of the image work. hOCR/ALTO/PDF already name `CrispEmbed` as producer but carry **no version**, so an archived document cannot be traced to a build. Adding it: `ocr-system` (idiomatic — Tesseract writes `tesseract 4.1.1`), ALTO `<softwareVersion>`, PDF XMP. Contained: the version is a compile-time constant, no plumbing. **Deliberately NOT doing the bigger piece, and recording why:** POLICY §6 distinguishes CTC/attention recognisers (transcribe) from VLM engines (which "confabulate through a smudge rather than leave it blank"), and an archived hOCR cannot tell them apart. Recording the ENGINE would be the true analogue of naming the engine in the image marking. But neither render call site has the engine in scope — `crispembed_ocr_render` (crispembed.cpp:6127) is a pure formatter over already-computed results, and the server's /document path (server.cpp:3328) builds the renderer before the pipeline runs — so it needs threading engine identity through the orchestrator plus a public C-ABI change. That is invasive across exactly the files `feat/easyocr-ggml`, `feat/ppocr-next-20260731` and `feat/ocr-engine-parity` are actively changing; starting it now would generate conflicts and step on their half. **Proposed as a follow-up for whoever owns the orchestrator next** — design would be an optional `ocr_render_set_engine()` plus one field on the pipeline result, not a signature change. Also noting: the 12 `/tmp/cpp_*.bin` writes in `src/layout_detect.cpp` are NOT the temp-file defect I just fixed — all gated behind `LAYOUT_DEBUG`, and the fixed names are a deliberate contract with a Python reference dumper (the C++ side READS `/tmp/py_cross_out.bin`); randomising them would break the parity workflow. Documented as a caveat instead (in contributing.md, with the residual risk of enabling it on a shared host). **DONE:** version now in hOCR `ocr-system`, ALTO `<softwareVersion>` (a standard element that was simply absent) and PDF XMP CreatorTool/ProducerTool, from ONE `producer_name()` so three string literals cannot drift. Plain text deliberately stays clean — callers pipe it and a header would land inside the transcription. `tests/test_render_provenance.cpp` (11 checks) reads the version from the same macro the code does, requires the formats to agree, and fails if the version is `unknown` — which would otherwise satisfy every check while recording nothing. Wired into `build.yml`. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** verify the claim I put in POLICY §5 / contributing.md that every emitted image is marked, by auditing EVERY image-writing call site. Result: CLI (15) and server (3) all go through `core_imgout::emit`; three writers in `src/ocr_orchestrator.cpp` do not, and on inspection they are internal — two `temp_png_path()` handoffs and one debug crop dump behind `CRISPEMBED_TESSERACT_CROP_DUMP_DIR`. So the marking claim holds for output, but the wording is looser than the truth and will be tightened. **The audit found a real defect on the way: `temp_png_path()` (ocr_orchestrator.cpp:257) builds a PREDICTABLE `/tmp/crispembed_ocr_<pid>_<counter>.png` and `stbi_write_png` opens it with `fopen("wb")` — follows symlinks, world-readable under default umask.** That is the same defect I fixed in `server.cpp` earlier this branch (`/tmp/crispembed_doc_<pid>_<n>.img` -> `mkstemp` 0600); I fixed one instance and missed this one. Same sensitivity: the content is the user's scanned page. Coordination: `ocr_orchestrator.cpp` is touched by `feat/easyocr-ggml` and `feat/ppocr-next-20260731`, so this change is confined to the temp-path helper and touches no engine dispatch, graph or crop geometry. **DONE.** Root cause was TWO hand-rolled copies of the same logic, so there is now one: `src/core/temp_file.h`, used by both. mkstemp creates the file itself (unpredictable, O_EXCL, 0600) and the path is returned, which stays safe for callers that must write by name because the file already exists and is ours. `tests/test_temp_file.cpp` pins what regresses silently: file already exists, no group/other mode bits, regular file, suffix survives (callers dispatch on it), and 32 calls give 32 paths differing in MORE THAN ONE position — plain distinctness would have passed the old `<pid>_<counter>` scheme, which differed by a single digit. Wired into `build.yml`. Marking claim tightened in POLICY §5 and contributing.md to "every image CrispEmbed *returns to you*", with internal temporaries called out as unmarked and deleted — the old wording overstated it. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** my own follow-up — the three benchmark harnesses were pinned to `CRISPEMBED_IMAGE_FORMAT=ppm` when PNG became the default, so they validate the LEGACY path and not what users get; a defect in the PNG path would pass them. On inspection the pinning was over-cautious: all three already use PIL, which reads both formats, and the only hard dependency is one magic-byte check in `tests/ocr_preprocessor_benchmark.py`. Doing it in two parts: (a) prove the invariant that makes unpinning safe — PNG and PPM must decode to IDENTICAL pixels, i.e. the format change is lossless and does not perturb any downstream metric; (b) make the harnesses format-agnostic and drop the pins. Coordination: touches only my own provenance work plus three benchmark scripts; no engine/graph/model code; all 12 other open rows belong to `perf/ocr-h-items`, `feat/easyocr-ggml`, `feat/ocr-engine-parity` and `feat/ppocr-next-20260731` and are untouched. **DONE.** (a) The invariant is now a test, not an assumption: emit the same pixels as PNG and as Netpbm, decode the PNG, require byte-identical pixels — gray and RGB, at 23x17 (deliberately not a multiple of anything). These harnesses measure PSNR/SSIM/CER on those bytes, so had the formats disagreed anywhere every restoration metric would have shifted when the default changed and read as a model regression. (b) Pins dropped from all three: the two PIL-based ones needed nothing, and `ocr_preprocessor_benchmark.py`'s magic-byte check now accepts both and names the temp file for what it actually is (a `.pnm` holding PNG bytes would mislead anyone inspecting the dir). Smoke-tested the real path: `--cleanup-only` emits a 606x1000 PNG the harness accepts, PIL decodes, carrying the provenance chunk. **Note for whoever runs the restoration benchmarks next:** they now exercise the default PNG path, so a first run after this may differ from older numbers only by file size, never by pixels. | **COMPLETED** |
| 2026-08-03 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Both loose ends closed.** (1) **q8_0 cosine reconciled — it was the backend, and benign.** Same binary/files: `vis_proj_output` is `0.998630` on Metal and `0.912992` on CPU; CPU min `0.912992` vs the parity kernel's `0.905481` (also CPU, different box). Crucially it does **not** reach the output — the same CPU path transcribes the full page at 1749 chars vs Metal's 1748, same text. The divergence lives in the diff harness, whose vision input is a synthetic gradient that amplifies numerical differences a real page does not. Two carry-forwards recorded in PERFORMANCE.md: never quote a vision-stage cosine without its backend (same artifact spans 0.913–0.999), and the 0.999 gate on `vis_proj_output` is mis-calibrated for that synthetic input — it fails artifacts that decode correctly. Left unchanged deliberately: a threshold retuned against one model is how gates rot. (2) **Registry health all green, verified not assumed:** 242 pins / 0 unpinned and `model_hashes.h` current; **242 distinct URLs, 0 non-200**; licence check rc=0. Also re-verified `h2ovl-800m` on CPU as well as Metal after the BOS-ordering change. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** real C2PA Content Credentials for image outputs, default-on, mirroring CrispASR (user decision). **Probe done first, and it moved the plan.** (1) CrispASR's C2PA does NOT port: its `format_for_ext` returns audio MIME types only, and its "native" path is a vendored `third_party/c2pa-audio` submodule that only does `sign_wav`/`sign_mp3`/`sign_m4a`/`sign_flac`. Reusable part is the CMake plumbing (`Findc2pa.cmake`, `CrispasrC2pa.cmake`, prebuilt fetch of c2pa-rs 0.89.3). (2) **c2pa-rs REJECTS self-signed certs** — "the certificate was self-signed" — so CrispASR's baked self-signed default cert would not work through this path at all; a leaf+CA chain is required. Verified working: 77-byte PNG -> 42,631-byte signed PNG, manifest reads back via `c2pa_reader_from_stream`. (3) **Container blocker**: we emit raw PPM/PGM and C2PA has no PPM binding, so PNG output is a prerequisite (`stb_image_write.h` already vendored). (4) Manifest overhead is ~42 KB, dominated by an auto-generated JPEG thumbnail — worth disabling for small crops. Also correcting a copy-paste hazard: CrispASR asserts `c2pa.created` + `trainedAlgorithmicMedia`, right for TTS which is wholly synthetic; our inputs are real captures we enhance, so the truthful assertion is `c2pa.edited` + `algorithmicallyEnhanced` — copying CrispASR's verbatim would have made every restored scan claim to be wholly AI-generated. Coordination: image-output paths + CMake only; no OCR/model/graph code; no overlap with the other five active branches. **DONE, per the recommendation the user accepted: default-on marking, NO key in the repo.** Images are PNG by default (Netpbm has no metadata container — the format was the blocker, not the policy) with an `iTXt` chunk naming the engine; `CRISPEMBED_IMAGE_FORMAT=ppm` restores raw output and is how the three benchmark harnesses are pinned rather than teaching each a PNG decoder. C2PA layered on when `CRISPEMBED_C2PA_CERT/_KEY` are set, `-DCRISPEMBED_C2PA_FETCH=ON` pulls c2pa-rs; absence of lib or cert is a supported state that still yields a marked PNG. `scripts/make-c2pa-cert.sh` builds a per-installation leaf+CA chain (self-signed is REJECTED by c2pa-rs; key must be PKCS#8 or you get an opaque ASN.1 error). Assertion is `c2pa.edited` + IPTC `algorithmicallyEnhanced`, NOT CrispASR's `c2pa.created` + `trainedAlgorithmicMedia` — right for TTS, false for us, and the test asserts the wrong term is absent. **Single stb_image_write definition moved to `core/image_out.cpp`** (was in ocr_orchestrator.cpp; duplicate symbols at link) — kept stdio in, since ocr_orchestrator writes crops by path and a test externs `stbi_write_png`. Verified: end-to-end via adair, PIL reads the iTXt, and the PNG is SMALLER than the PPM it replaces (19,505 vs 27,661 bytes). **Hardened to 35 checks + CI + docs:** the test now validates every chunk with an INDEPENDENT bit-by-bit CRC-32 (the table-driven one in image_out.h cannot validate itself, and a wrong chunk CRC is ignored by stb/PIL but rejects the file in strict decoders — also cross-checked against zlib.crc32); plus chunk order, the iTXt five-field layout, buffer==file byte-identity, MIME-matches-bytes, input rejection, and empty-engine handling. **Server sites now use the same path** — they still wrote raw Netpbm, so an image over HTTP was marked differently from the same image from the CLI; `/preprocess/dewarp` returns via a new `emit_to_string()` that hands back the Content-Type with the bytes so the two cannot disagree. **The tests now RUN**: all three are model-free and network-free, wired into `build.yml` on every push — they previously existed and gated nothing. `docs/provenance.md` added (what each level proves, why `algorithmicallyEnhanced` not `trainedAlgorithmicMedia`, the two c2pa-rs constraints, c2patool verification, env vars); five engine docs stated `> out.ppm`, which would now produce a mislabelled file, and are corrected. | **COMPLETED** |
| 2026-08-03 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** the last open finding from my own AI Act audit — **Art. 50(2) machine-readable marking**. POLICY §5 states plainly that CrispEmbed adds no watermark or C2PA provenance marking to any output and that an integrator "must add it yourself"; the Digital Omnibus grace period for systems already on the market ends **2 December 2026**. Nothing in `src/`/`examples/` mentions c2pa/watermark/provenance today (verified). Scope is the CAPABILITY, opt-in and OFF by default — POLICY's reasoned position is that document restoration is standard editing under Recital 134 and needs no marking, and I am not overturning that unilaterally; this closes the "you must build it yourself" gap for integrators whose use sits away from the document case. Constraint discovered first: outputs are raw PPM/PGM to stdout or file, formats with no metadata container, so the only in-band channel is the PPM/PGM header comment. Coordination: touches the image-output paths only, no OCR/model/graph code, no overlap with `perf/ocr-h-items`, `feat/easyocr-ggml`, `feat/ppocr-next-20260731`, `feat/ocr-engine-parity` or `feat/ocr-followups`. **DONE:** `CRISPEMBED_MARK_GENERATED=1` emits a Netpbm header comment from all 18 emission points (15 CLI, 3 server), naming the ENGINE so a reader can tell synthesised detail (ESRGAN/NAFNet/SCUNet) from resampling (deskew/dewarp) — not recoverable from the pixels. Off by default, so §5's document-case position is unchanged. Safe only because stb_image's PNM loader skips `#` runs (`stbi__pnm_skip_whitespace`, verified) — `tests/test_provenance_marking.cpp` pins that round-trip, off-by-default, and that every emitted line is a comment (a stray non-`#` line would be parsed as the image dimensions). Verified end-to-end on adair; output still decodes in PIL. Documented in POLICY §5 + README as what it is: a strippable comment with no cryptographic binding — tamper-evident provenance still needs C2PA and a signing identity. | **COMPLETED** |
| 2026-08-02 | `chore/ai-act-audit-round3` / `.codex/worktrees/chore-ai-act-audit-round3` | **Picked:** third AI Act audit pass. Verified the round-1/2 controls hold in code (gate at `crispembed_face_init` keyed on declared model type; no 1:N primitive; no prohibited-category model in the registry; both CI-enforced), then closed six gaps the earlier passes left. (1) POLICY.md never reached PyPI/pub.dev users — `setup.py` now stages it into the wheel (verified present in the built wheel) and both package READMEs carry the Art. 5 prohibitions + GDPR Art. 9 note; (2) `/face`+`/detect` read arbitrary server-side paths — new `--image-root` confines all 30 `{"image":…}` endpoints via `weakly_canonical` + component-wise prefix, new `tests/test_image_root.py` covers absolute/traversal/symlink/sibling-prefix escapes; (3) Art. 4 AI-literacy duty was absent → new POLICY §8, and §1 now admits the project *deploys* two systems (Space, WASM demo), not only ships a component; (4) Art. 50 restated as in force (2 Aug 2026 has passed) with the no-watermark absence stated as unresolved; (5) model downloads were unverified → SHA-256 pins for all 232 pinnable registry URLs (`examples/cli/model_hashes.h`, generated by `tools/fetch_model_hashes.py` from HF LFS oids), fail-closed on mismatch/unpinned/non-HTTPS, wired into `main-health.yml`; (6) uploaded pages moved off predictable `/tmp` names to `mkstemp` 0600. **Side finding: 8 registry URLs are 404** — `pix2struct-base-q8_0.gguf` and `lid-glotlid-f16.gguf` are filename typos (repos hold `-f32`/`glotlid-f16`), and `InstructIR`/`AdaIR`/4 `*-crispembed-GGUF` repos have no GGUF uploaded. **Follow-up in the same branch: 4 of the 8 now fixed.** `instructir-f16.gguf` (quantized from the published f32; output cos 1.0, max 1 LSB vs f32) and `pix2struct-base-q8_0.gguf` (byte-identical greedy decode vs f32) were built and uploaded to `cstr/instructir-GGUF` / `cstr/pix2struct-GGUF`; pix2struct's registry size said 300 MB but the real q8_0 is 467 MB. `glotlid` had two bugs — the `lid-` prefix is CrispASR's naming convention and never existed in this repo, and "3.3 MB" was wrong by ~250x (GlotLID-V3 f16 is 848 MB) — now points at `glotlid-f16.gguf`, with `glotlid-q8`/`glotlid-q4k` added; FUNCTIONALLY UNVERIFIED because this repo has no LID engine (`text_lid_dispatch.h` is an optional CrispASR header behind `__has_include`). Repo ids normalized to canonical lowercase `cstr/instructir-GGUF` / `cstr/adair-GGUF`, which previously only resolved via HF's case redirect. Unpinned URLs 8 -> 4. The remaining 4 are the VLM repos that do not exist, with no artifact anywhere on the backup volume, so they need real reconversion (`convert-qwen2vl-to-gguf.py` for german-ocr-3.1 and nanonets-ocr2-1.5b, `convert-internvl2-to-gguf.py` for both H2OVL). **~~TODO~~ RESOLVED 2026-08-02 (`feat/ocr-followups`) — AdaIR F16 was an engine bug, not a bad artifact, and this row called it correctly:** `adair-5d-f16.gguf` quantizes cleanly but aborts at run time on the default ggml_conv path with `GGML_ASSERT(buf != NULL && "tensor buffer not set")` in ggml_backend_tensor_set, from `adair_kernel` (src/adair.cpp:692) via `adair_conv`. f32 runs fine and InstructIR f16 from the same quantizer runs fine, so it is specific to src/adair.cpp's F16 path. Root cause NOT established: the scalar fallback (ADAIR_SCALAR=1) was too slow to finish, so whether the bug is confined to the ggml path is unknown. No f16 was uploaded; the registry points at the published f32 (115 MB) instead. **VLM reconversion follow-up: 3 of the 4 missing VLMs now built, verified and uploaded; unpinned URLs 8 -> 1.** Sources found: `keyvan-ai/german-ocr-3.1` (llama.cpp split GGUFs; byte-identical mirror of `Keyven/german-ocr-3.1`), `nanonets/Nanonets-OCR2-1.5B-exp` (the `-exp` suffix is why a plain `Nanonets-OCR2-1.5B` probe 404s), `h2oai/h2ovl-mississippi-{2b,800m}`. (1) **german-ocr-3.1** — merged upstream F16 LLM + F16 mmproj via `merge-llamacpp-qwen2vl-gguf.py` (4.12 GB, matching the original b58d7805 commit note), quantized to q4_k = 1684 MB (registry said 1301; corrected). Verified: near-perfect transcription of `scan_page_pd.png`. NB merging the *pre-quantized* Q4_K_M LLM instead is wrong — it yields a 2.3 GB hybrid, not the recorded recipe. (2) **h2ovl-800m** — 644 MB q4_k (registry said 398; corrected), verified legible full-page OCR with edge-model artifacts. (3) **nanonets-ocr2-1.5b** — 1346 MB q4_k, *exactly* the size the registry already claimed, verified near-perfect full-page transcription (421 tokens). **Two converter bugs found and fixed, both silent-failure classes:** (a) `convert-internvl2-to-gguf.py` routed the LLM attention layout off `config.model_type`, so H2OVL's Danube LLMs (model_type `llama` for 800m, `mistral` for 2b) went down the InternLM2 fused-`wqkv` branch, every lookup missed, `lw()` returned False silently, and the writer emitted **only the per-layer norms** — a GGUF with no LLM weight matrices that loads fine then segfaults in `ggml_mul_mat` on a null tensor (381 tensors instead of 493). Now routed on tensor-name presence, plus a post-export guard that aborts naming the missing tensors. This means PLAN's earlier "H2OVL-Mississippi-2B **Ported**" claim was stale for BOTH H2OVL models. (b) `convert-qwen2vl-to-gguf.py` wrote `qwen2vl.tie_word_embeddings` twice when a checkpoint ships no `lm_head` (gguf raises on duplicate keys), killing conversion after the vision tower was written; now derived once from whether `lm_head.weight` is actually in the checkpoint. **TODO — h2ovl-mississippi-2b is the one still unshipped.** It converts (565 tensors, 4.42 GB) and loads, but emits degenerate output at BOTH f16 and q4_k — f16 repeats one token then EOS, q4_k emits "." then EOS — so it is not a quantization artifact. Suspects not yet separated: chat/prompt template for Danube-1.8B, Mistral sliding-window attention, or the 32H/8KV GQA ratio (the working 800m is 16H/8KV). Its registry URL still 404s and is commented as such. **Measurement warning:** a parallel session drove load average to 75-316 during this work; the first full-page nanonets run produced zero tokens in 900 s purely from CPU starvation and would have been misread as a hang. Re-run VLM timings on a quiet machine. **h2ovl-mississippi-2b root cause: `use_msac`, now IMPLEMENTED.** The 2b sets `use_msac: true`, the 800m false — the only material difference between the working and broken model (same `template: h2ogpt2`, vocab, downsample/ps_version; rope_theta differs 10000 vs 100000 but is read correctly and the engine is arch-agnostic). H2OVL's Multi-Scale Adaptive Cropping tiles the page twice: coarse grid, then a fine grid keeping only ratios where `prior_cols%c!=0 && prior_rows%r!=0` (so it is not a sub-grid), concatenated `fine[:-1] + coarse[:-1] + fine[-1:]`, thumbnail last. Single-scale tiles give a model trained on that layout fluent nonsense. Implemented in `image_preprocess::preprocess_internvl_msac_rgb` + `internvl2.use_msac` dispatch. **Two parity bugs in the existing tiler had to be fixed for the fine grid to come out right:** (a) aspect ties were broken toward FEWER tiles; upstream breaks them toward more when `area > 0.5*size*size*blocks` — for 800x800 that is 1x1 vs 2x2; (b) a pass producing one block gets no thumbnail upstream, so `[:-1]` drops the tile itself and it contributes nothing — we gave 6 tiles where upstream gives 5. `tests/test_msac_tiling.cpp` pins all five cases against values transcribed from H2OVL `image_process.py`, including 800x800 where no admissible fine grid exists and we must DECLINE rather than fall back to single-scale. Model-free. The tie-break touches every InternVL model, so h2ovl-800m was re-run end to end: same 7 tiles (3x2), same 1122-byte transcription. **The 2b GGUF is being built by `tools/kaggle/h2ovl-convert`, not locally** — 4.3 GB safetensors + 4.4 GB f16 + 1.4 GB q4_k exceeds this machine's free space and a local attempt ran the disk out mid-write. The kernel asserts `use_msac` survived into the GGUF and the LLM attention/FFN tensors exist, OCR-smoke-tests the page fixture, and refuses to upload unless MSAC tiling ran and >=200 chars came back. **Also this pass:** adair-5d back on f16 (115 -> 59 MB) now that 67ec560c fixed the runtime shape bug — f16 vs f32 cos 0.99999994, max 1 LSB; and `--adair-model`/`--instructir-model` now resolve registry names, which they never did, so those two entries were unreachable from the CLI. Touches `examples/cli/model_mgr.cpp`, `examples/server/server.cpp`, packaging, POLICY/README, CI — **no OCR/model/graph code**. Verified: SHA-256 against 4 NIST vectors incl. the 56-byte padding edge; pinned download verifies, tampered pin rejected + cache left clean, unpinned refused, override works; biometric gate 11/11 still passes; image-root 5/5 + rejection logging; `tools/format.sh` clean.  **COORDINATION (h2ovl-2b, 3 sessions):** this branch owns the CONVERT half only — `tools/kaggle/h2ovl-convert` + the MSAC runtime. `feat/ocr-followups` owns the PARITY half (`tools/kaggle/h2ovl-parity`) and `tools/kaggle/h2ovl-publish`. I nearly broke that twice and both are worth recording: (1) my kernel called `create_repo` WITHOUT `private=True` on the same repo their publish kernel creates private — `exist_ok=True` does not flip visibility, so whichever ran first decided it, and mine would have made a model that emits 29 chars of `.assistant.assist` PUBLIC. Pre-created the repo private out of band, then removed my upload path. (2) Their row says they were *waiting on convert to publish the f16*; h2ovl-publish has since landed f16+q8_0+q4_k in that private repo, so that dependency is satisfied. **MSAC is implemented and the 2b is still broken, but the suspect list is now small.** Their parity run measured 27 stages at cos_min 0.999972 vs the Python blueprint, so the ported compute is right — it is NOT the graph and NOT the MSAC tile math. Combined with my fix that the call site hardcoded min/max_dynamic_patch to 1/12 and threw away the GGUF's declared 6 (19 tiles where the reference gives 13; the 800m is unaffected, same 3x2 grid either way), the remaining suspects are prompt construction, sampling, and detokenisation. **Next, locally, not on Kaggle:** `cstr/crispembed-regression-fixtures` already has `internvl2/h2ovl-mississippi-2b/ref.gguf` (112 MB) and the private model repo has the q4_k (1.46 GB) — that pair is enough for `tests/test_internvl2_diff.cpp` on this Mac, so my kernel's own 4-layer ref-gen is redundant and should be dropped. **CAUTION on the parity claim everyone is now reasoning from.** "27 stages at cos_min 0.999972" is almost certainly VISION-ONLY: `tools/dump_internvl2_reference.py` emits exactly 27 vision-side stages (`vis_patch_embed` + `vis_layer_0..23` + `vis_pixel_unshuffle` + `vis_proj_output`), and the parity kernel runs the dumper with `--max-llm-layers 4`, which would add `llm_embed` + 4 layers = 32 if the LLM side were counted. So what is established is that the InternViT tower and the projector are right. The 24-layer Danube-2 decoder, the LM head and the logits are NOT covered — and that is exactly where a mistral-vs-llama porting bug would live, the 2b being the `mistral` one while the working 800m is `llama`. Confirm the stage list in the parity log before concluding "the compute is right, the fault is downstream of the logits"; on this reading the decoder is the prime suspect, not detokenisation. **Decoder coverage now exists** (shared tooling, not their kernel): the diff harness ran `run_llm_forward`, printed the output SHAPE, freed the buffers and compared NOTHING — and returned 0 unconditionally, so even the vision stages were advisory. Added `llm_output_norm` + `llm_logits` to `dump_internvl2_reference.py` (emitted only on a full-stack dump; tied head read from the weights, not a config flag), and the harness now compares both, counts failures, returns non-zero, and checks **argmax of the last position separately** — cosine stays high while the argmax moves, and the argmax is what generation acts on, which is exactly this failure's shape. To get decoder coverage, regenerate the fixture WITHOUT `--max-llm-layers 4`. **h2ovl-mississippi-2b WORKS — unpinned URLs 8 -> 0.** The remaining bug was the hardcoded patch limit, not MSAC and not the decoder: with the model's declared `max_dynamic_patch=6` honoured, the run tiles to **13** (the reference geometry, was 19) and returns rc=0 with 1109 chars instead of 29 chars of `.assistant.assist`. Published, validated, repo flipped private->public now that it earns it (it was private by `h2ovl-publish`'s correct call while broken), registry entry activated at the measured 1459 MB, ref.gguf alongside it. **RESOLVED — it was the invocation, and it took BOTH halves.** Compared against upstream `conversation.py`/`modeling_h2ovl_chat.py`/`tokenizer_config.json` rather than guessing. (a) `add_bos_token: false` for BOTH h2ovl checkpoints, and upstream `chat()` just calls `tokenizer(query)` — the blueprint prompt has no `<s>`; we prepended one unconditionally. (b) `"OCR this image."` is too terse for a general-purpose VLM; upstream's own examples are explicit imperatives, and `qwen2vl_ocr` had already been forced into the same change. **Neither alone works** — the 2x2 matrix is: BOS+terse -> describes; BOS+explicit -> describes; noBOS+terse -> describes; noBOS+explicit -> TRANSCRIBES. That is why one-at-a-time attempts kept failing. Full page, defaults only: 2b 1806 chars verbatim (keeps curly quotes and `dis- played` hyphenation); 800m 1749 chars, also improved — it used to emit `NEW PAGE / <- / 36 / PRIDE AND PREJUDICE.` layout artifacts. Converter emits `internvl2.tokenizer.add_bos_token`; for older GGUFs the h2ogpt2 template defaults it off. `CRISPEMBED_INTERNVL2_PROMPT` / `CRISPEMBED_INTERNVL2_ADD_BOS` added for bisecting. **Trap worth remembering:** reading `tok.h2ogpt2` before the GGUF template key was parsed made the BOS default a silent no-op; only the A/B caught it. Superseded TODO was:  it *describes* the page ("The image presents a page from a book, specifically page 36...") rather than transcribing it, while quoting the content accurately. The internvl2 engine is handing H2OVL a captioning-style prompt; it needs an OCR/transcribe instruction, as the qwen2vl engine already does. **Also landed this pass:** `--check-sizes` in `tools/fetch_model_hashes.py` (found 4 more wrong registry sizes on first run: bidirlm-omni 2.6 GB->1834 MB, ppformulanet-l 180->252, transcoda 120->69, bttr-hw 5->11; all fixed, 241 clean, wired into main-health); adair-5d back on f16 (115->59 MB) after 67ec560c; and `--adair-model`/`--instructir-model` now resolve registry names, which they never did.| **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** validate the opt-in Tesseract DAWG scorer, model-owned runtime lookup, and diagnostic beam-confidence contract after the remote recoder merge; fix prefix ranking/token boundaries, wire runtime tests, and keep production dictionary scoring/calibration disabled | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** close the remaining beam-confidence comparator gap so `--require-beam-sequence-only` rejects fabricated word certainty as well as character certainty; add model-free coverage | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** make the documented Tesseract row-blob-bounds geometry A/B reproducible through the page comparator and repeated benchmark manifests, while keeping it diagnostic-only | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** align the standalone Tesseract geometry comparator with the row-blob-bounds benchmark switch and record the policy in its JSON output | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** reconcile the stale EasyOCR-plan int-mode status with the detailed parity evidence, keeping recoder/DAWG and full-page decoded parity explicitly open | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** preserve unmapped Tesseract recoder classes as explicit `<class>` diagnostics instead of silently dropping or exposing numeric class labels; keep full composed-script parity open | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** preserve valid composed recoder segments around unmapped classes with a diagnostic partial composer; leave the default decoder and full composed-script parity gate unchanged | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** consolidate repeated CRAFT/DBNet warm-graph probes into a versioned JSON manifest with explicit reference/native timing ratios and box-count quality status; keep device mismatch and page-text parity visible. Live scan-strip manifest: CRAFT native/reference `29,511.835/11,480.765 ms` (`2.57x`) with `106=106` boxes; DBNet `44,647.873/16,153.006 ms` (`2.76x`) with native `98` boxes, reference count unavailable in the timing-only probe. | **COMPLETED** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** produce an independent EasyOCR Python page manifest for `lines` mode and compare ordering, line grouping, crop geometry, decoded text, and confidence against the native DBNet→EasyOCR handoff; keep page parity separate from detector-only timing. Live `scan_strip.png`: Python CRAFT produced 11 lines; native DBNet produced 12. The first mismatch is line 0 (`"They are going to be , encamped near   Brighton"` vs `& They are going to be, encamped near   Brighton`), with geometry `[62,0,412,25]` vs `[46.97,0,423.54,21.76]`; all subsequent records shift, so page quality parity is **not** passed. | **COMPLETED — parity failed; quality TODO** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** replay independent Python EasyOCR line boxes through the native CRNN to separate recognizer/crop parity from DBNet detector geometry; preserve the failed page gate if identical boxes still diverge. External replay now uses exact caller-supplied boxes (no native 2-pixel margin), returns 11/11 regions, and still diverges in native text/confidence (line 0 Python `"They are going to be , encamped near   Brighton"` vs native `They are going to be, encamped near   Brighton`; confidence `0.8541` vs `0.5483`; line 4 is a severe recognition failure). | **COMPLETED — recognizer/crop quality TODO** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** run fresh `crispembed-diff` on exact Python line crops before changing recognizer math. English Gen2 line 0 and the worst line pass input, features, sequence input, both BiLSTM outputs, and logits; line 0 decodes identically, while the worst line reproduces Python's own poor decode. Input cosine is `0.99981`; recurrent/logit cosines are at least `0.99972`; feature global cosine is `0.99993` (sparse per-row feature cosine is not a valid promotion gate). The remaining page discrepancy is therefore detector geometry/crop selection, recognizer asset/preprocessing identity, and Python/native postprocess confidence—not an unexplained GGML LSTM divergence. | **COMPLETED — page quality still open; no recognizer math change justified** |
| 2026-08-02 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** make the EasyOCR manifest boundary distinguish the padded postprocess `crop` from the actual recognizer input. Python and native manifests now emit `recognizer_crop`; `compare_easyocr_manifests.py --recognizer-crop-only` checks exact model-input geometry while preserving legacy crop comparisons by default. Contract tests pass, and the rebuilt `test-easyocr-pipeline` links at `[88/88]`. A real external replay confirmed 11/11 caller regions and showed the remaining text/confidence mismatch is genuine output quality, not a mislabeled crop field. | **COMPLETED — page quality TODO remains** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** add a dependency-free EasyOCR interoperability contract test covering Python `lines`/`words` ordering, crop/normalized geometry, and LayoutLM `apply_ocr=False` serialization; keep real-page reference parity as the separate live gate. `tests/test_easyocr_interop_contract.py` passes with 3 words, 2 grouped lines, and ordered LayoutLM sidecar metadata | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** retain PP-OCRv6 detector/crop/orientation/recognizer per-stage timings in the reproducible benchmark JSON; parser and stderr-capture slice. `tests/ppocrv6_pipeline_benchmark.py` now sets the bench switch, parses native stderr, preserves partial timeout telemetry, and labels unavailable stage rows. A live tiny German fixture produced detector/crop/orientation/recognizer timings and 34 detector boxes → 30 recognized results; full 10-fixture/medium quality sweep remains pending | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** add dependency-free PP-OCRv6 benchmark-parser, backend-capability, and OCR interoperability contract tests to the mandatory OCR regression smoke job; leave model/gold execution artifact-gated. Workflow YAML and all four smoke/contract checks pass locally; the gold step skips unless an artifact-equipped runner supplies `CRISPEMBED_GGUF_DIR` | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** generalize the PP-OCRv6 graph-gold harness from hard-coded small-only artifacts to explicit tiny/small/medium tier selections with tier-specific reference fixtures. The harness now supports all three tiers; tiny remains explicitly blocked until its legacy 16-tensor Arabic reference is regenerated as a full graph gold archive, while small remains the default accepted lane | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** add opt-in PP-OCRv6 detector graph-vs-CPU box geometry diagnostics without changing the production CPU accept-gate; report count, greedy matches, mean IoU, and minimum IoU for each diagnostic run. Implemented and compiled; the available tiny fox fixture reports graph=0 vs CPU=2, so detector graph geometry remains a quality/performance TODO and is not accepted by default | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** repair the manifest-driven O9 engine benchmark so structured detector specs (`{repo,file,revision}`) are normalized like recognizer specs; add a contract test and rerun Tesseract/PARSeq/PP-OCRv6 rows. Fixed structured detector normalization; Tesseract-LSTM `175.7 s` CER `0.040`, German Tesseract `101.8 s` unscored, PARSeq `1.206 s` unscored. The PP-OCRv6 artifacts load-fail (`missing stem conv`) and are now correctly marked errors instead of false `ok` rows | **COMPLETED** |
| 2026-08-01 | `chore/ai-act-hardening` / `.codex/worktrees/chore-ai-act-hardening` | **Picked:** third AI Act audit, run against `origin/main` after `f7f89032` landed. Re-verified the code-backed claims and found them all true: no 1:N/gallery primitive in the C ABI, no emotion/age/gender/ethnicity code anywhere, no scraping tooling, server gate fails closed before `crispembed_face_init()`, gate keyed on declared model type so a renamed `.gguf` is still caught, `/doc` temp uploads tracked and unlinked. Two gaps remained, both closed here: (1) the deployed GitHub Pages demo (`examples/wasm-ocr/index.html`) carried no notice at all while the HF Space had one — added an AI-output/data-locality/no-biometrics footer linking POLICY.md; (2) POLICY §3 and README asserted the absence of biometric-categorisation *models* in terms a reader could mistake for a guarantee about capability — CLIP/SigLIP zero-shot means the caller supplies the classifier, now stated. Also strengthened §7: Art. 53 does not engage for task-specific models at all (the quantization argument is the fallback), and Art. 53(2) does not waive the copyright policy or training-data summary. Docs/HTML only — no C/C++, no rebuild needed | **COMPLETED** |
| 2026-08-01 | `chore/ai-act-audit-followups` / `.codex/worktrees/chore-ai-act-audit-followups` | **Picked:** close the five gaps a second AI Act audit found in the `chore/ai-act-policy` work. (1) biometric gate moved into `crispembed_face_init()` so the Python/Rust/Dart bindings are covered, not just CLI+server — new ABI `crispembed_accept_biometric_use()`; (2) `check_registry_licenses.py` read only HF's `license` tag and missed `license_name`, so the 4 correct lfm2 rows failed — fixed, now exit 0, and wired into `main-health.yml`; (3)(4)(5) POLICY.md: Art. 50(2) reframed as reasoned-position-not-settled-exemption, OCR-VLM text addressed, and the regulatory dates corrected — the Omnibus is **Reg (EU) 2026/1744, OJ 24 Jul 2026**, not "adopted June 2026". Touches `src/crispembed.{h,cpp}`, `examples/cli/model_mgr.*`, bindings, POLICY/README/PLAN, `tests/check_registry_licenses.py` — **no OCR/model/graph code**. Verified: CLI + Python both refuse a recognition model without acknowledgement and load it with one, byte-identical embeddings either way; licence check exits 0; `tools/format.sh` clean. | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** fix O9 pipeline benchmark routing to use each manifest entry’s engine family (`ppocrv6`) instead of the tiered display name (`ppocrv6-tiny`), which had sent PP-OCRv6 fixtures through generic DB postprocessing and produced false `missing stem conv` load failures. Rebuilt CLI and verified tiny `4.98 s`/2 regions, small `20.82 s`/2 regions; medium exceeded the `120 s` guard and is recorded as a timeout, not a quality pass | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** resolve the remaining official PP-OCRv6 quality discrepancy by testing the HF/PaddleX preprocessing contract (RGB/BGR, resize, normalization) and CTC decode on known-text crops; promote a runtime change only if native output diverges from the official source under the same input. Result: preprocessing is aligned; remaining issue is checkpoint/vocabulary/line-crop quality | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** validate PP-OCRv6 checkpoint provenance, CTC vocabulary selection, and line-crop suitability against known-text crops before claiming quality parity; preserve native/reference decoded strings and timing evidence. Root cause was confirmed upstream: 320 is a minimum width, not a cap; the native/reference path now preserves dynamic CTC width and the 18,710-class space vocabulary | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** make the static PP-OCRv6 recognizer graph safe with dynamic-width line crops; bypass the fixed 320-wide graph for wider crops and retain the CPU reference path until a dynamic-shape graph is implemented and benchmarked. Live graph-debug test on an 800×100 fox line reports `input elements=55296` and cleanly bypasses the 320-wide graph; CPU decodes `The quick brown fox jumps` | **COMPLETED** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** implement and benchmark a truly dynamic-width PP-OCRv6 GGML graph, with width-keyed graph/cache ownership and CPU/Metal parity; keep the current explicit CPU fallback as the acceptance baseline. Implemented width-keyed graph rebuilds that retain the loaded GGUF source weights; a single process now runs 320-wide and 384-wide crops with graph outputs `80x3x384` and `96x3x384`, and graph-accepted text matches CPU (`De t 4 dg 14` / `The quick brown fox jumps`) | **COMPLETED — CPU graph validated; Metal parity pending** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** run the width-keyed recognizer graph on Metal for 320/384-wide crops, compare graph-accepted output and stage logits to CPU, and retain CPU fallback for any backend-specific divergence. Metal `MTL0` builds both widths and graph-accepted text matches CPU (`De t 4 dg 14` / `The quick brown fox jumps`); no dynamic gold archive exists yet for stage-logit comparison. Two-crop cold process timing was `28.32 s` Metal versus `2.94 s` CPU, so Metal is currently slower due pipeline compilation and remains diagnostic-only | **COMPLETED — parity passes; performance TODO** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** add reusable Metal pipeline/cache timing and dynamic-width gold logits, then benchmark warm 320/384-width recognizer graphs; do not promote Metal acceptance until cold/warm cost and numerical parity are recorded. Fixed repeated Metal scheduler reuse: re-plan Metal buffers per invocation while CPU retains allocation reuse. Same-width repeated Metal now exits 0 with identical text (`19.78 s`/2 crops); alternating 320/384/320/384 also exits 0 with identical text (`21.57 s`/4 crops). Dynamic stage-logit gold remains pending | **COMPLETED — stability fixed; performance/logit TODO** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** reduce Metal dynamic-width overhead by caching compiled graph plans per width or batching same-width crops; add width-specific gold logits before any Metal acceptance promotion. Added `tests/ppocrv6_width_benchmark.py`, which preserves decoded strings and graph shapes for grouped/alternating runs. Current `n=1` Metal timings: short `2582.1 ms`, wide `2250.0 ms`, alternating pair `2780.6 ms`; all return 0 with exact CPU-matching text and MTL0 shapes `80x3x384`/`96x3x384`. CPU controls are `468.3`/`403.8`/`463.8 ms`. | **COMPLETED — benchmark harness; cache/logit TODO** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** generate width-specific official-source activation golds and add CPU/Metal stage-logit comparison to the width benchmark; quantify whether current Metal numerical drift is acceptable before enabling any production graph gate. `tests/ppocrv6_width_benchmark.py` now accepts separate short/wide references and reports per-stage cosines. Fresh golds pass CPU logits cosine `0.999892` at 320 width and `0.999993` at 384; Metal passes `0.999861` and `0.999993`; decoded text is identical in every case. The older 320 archive was stale and was regenerated from the corrected official mirror | **COMPLETED — numerical parity passes; Metal remains opt-in for cost** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** benchmark Metal graph acceptance on the full PP-OCRv6 line/page route with dynamic-width gold coverage, then decide whether the recognizer graph can leave diagnostic-only mode. Added `--recognizer-graph` to `tests/ppocrv6_pipeline_benchmark.py`. The isolated 384-wide gold lane passes (`logits cos=0.999993`), but the German CC0 full route with 33 regions exceeded the 120 s guard under Metal graph acceptance; the prior CPU-accepted route completes in about 20.8 s. Keep recognizer graph acceptance diagnostic-only; full-route batching/residency is required before promotion | **COMPLETED — promotion rejected on measured cost** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** reduce full-page Metal graph cost by batching same-width line crops or reusing width-keyed graph residency across the detector→crop→recognizer loop; compare full-route text and stage timings against the CPU-accepted baseline. Before batching, added a safe per-page graph budget: with 33 detected regions the explicit graph request now selects CPU fallback and completes instead of timing out; measured German CC0 route `38.55 s`, 33/33 results, 1,146 chars, with recognize `32.65 s`. | **COMPLETED — safe fallback; batching still required for speed** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** batch same-width PP-OCRv6 line crops in the full route, preserving original order and per-line dynamic widths; compare batched CPU/Metal logits and decoded text against the current scalar fallback. Added native width-distribution telemetry and JSON capture. The current orchestrator still calls the recognizer one crop at a time; a German 33-region live run remained over the 180 s graph-debug guard, so no batching claim is made and graph acceptance stays budgeted/diagnostic-only | **COMPLETED — instrumentation and safety decision; no batch API yet** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** add a real PP-OCRv6 recognizer batch API grouped by identical dynamic model width, retain original result ordering, and require CPU-vs-Metal per-stage logits plus decoded-text parity before enabling it in the full route. Added the C ABI batch contract and wired the detector→crop→orientation→recognizer route through it. Live small-rec two-crop contract (fox + receipt) completed both items with byte-identical scalar/batch text; CPU sample was scalar `9.564 s`, grouped batch `6.070 s` (`1.58x`, warm-cache/small-sample evidence only) | **COMPLETED — safe grouped API; fused graph still required** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** implement fused GGML batch dimensions for same-width PP-OCRv6 crops, with bounded batch size, per-item error isolation, CPU/Metal logits cosine gates, and full-route German CC0 benchmark comparison. Added a bounded batch dimension to the tiny logits graph and kept it behind `CRISPEMBED_PPOCRV6_BATCH_GRAPH`. Important correction: the first `52.5 ms`/`35.1 ms` CPU smoke had `CRISPEMBED_PPOCRV6_FORCE_CPU`, which intentionally disables graphs, so it proved grouped scalar parity, not fused graph execution. A real Metal fused probe exposed a GGML pooling shape assertion; Metal is explicitly forced back to grouped scalar execution and no GPU promotion is claimed | **COMPLETED — safe gate; fused CPU proof still pending** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** make fused batching Metal-safe by preserving per-item spatial dimensions through pooling/reshape and adding CPU-vs-Metal logits cosine checks; then extend the fused path to large-stem SVTR only after the tiny lane is stable. Added an explicit Metal capability gate and fallback telemetry. German CC0 full route remains complete and text-bearing (`33/33`, `1,146` chars); current run was `68.75 s` total (`46.45 s` recognition), with `22` unique dynamic widths, so no same-width page batch gain is claimed | **COMPLETED — safe gate and baseline; shape rework remains** |
| 2026-08-02 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Picked:** settle the `ne[3]` conv-output-channel follow-up left by the AdaIR F16 fix instead of leaving it as speculation. `surya-det-f16.gguf` is only 77 MB, so the claim IS testable here: run it through `crispembed-quantize` to produce the flattened layout and A/B surya detection against the converter-made 4-D artifact. Fix `src/surya_det.cpp:700` only if the test actually breaks. **Not** h2ovl-mississippi-2b — that needs ~9–10 GB for source + conversion and both volumes are full (internal 6.7 GB free, backups 7.8 GB), so it is disk-blocked, not skipped. **Result — the two suspects split apart.** (1) The flatten fires only on 4-D **F32**: quantizing `surya-det-f16.gguf` (4-D F16) to q8_0 leaves all 79 4-D tensors intact, so precision alone does not predict exposure — source dtype + producer does. (2) **`src/surya_det.cpp:700` is NOT a bug** — `g_conv` reshapes a 2-D weight to 4-D *before* that read, so my earlier 'same bug class' note was wrong. No change. (3) **`src/tps_locnet.cpp:219` WAS real and is fixed** — it reads `ne[3]` at load with no normalisation and `convert-tps-loc-to-gguf.py` **defaults to F32**, so a quantized tps-loc GGUF hits it; instrumented pre-fix the four layers loaded `ndims=2, channels=1` instead of `16/32/64/128`, and `channels` feeds the fc1 input width and per-layer output channels. Fixed with the `cnn_embed.cpp:148` convention. New hermetic guard in `tests/test_tps_locnet.cpp` (no model file) compares 4-D vs flattened builds of the same fixed-seed weights: worst control-point deviation `0.026871 px` → `0.000000 px`, suite 14/15 → 15/15. **The guard's first version passed against the broken code** — the synthetic `fc2.weight` was all zeros so the output was `fc2.bias` alone and never touched the conv stack; `fc2.weight` now carries small non-zero values. Recorded because a green new guard means nothing until it has been seen to fail. **Sweep extension — two quantized SR/denoise models turned out to have never been run at all, and both aborted.** `esrgan` (`GGML_ASSERT(cgraph->n_nodes < cgraph->size)`): not a layout bug — `esrgan_prep_conv` reshapes correctly — but the graph budget. Measured 18-conv x4 at 64x32: f32 builds `283` nodes vs the `n_convs*12+100 = 316` budget, quantized builds `335` (dequant cast + `ggml_cont` add ~3 nodes/conv) and overflows by 19; budget now `n_convs*16+128`. q8_0 vs f32 cosine `0.999998`/PSNR `51.89 dB`; **q4_k runs but degrades hard (`29.55 dB`, max_abs `91/255`) — q8_0 is the usable quant.** `scunet` (`GGML_ASSERT(a->ne[2] == b->ne[2])`): this one IS the flatten — the persistent kernel cache copies source `ne` verbatim, so a flattened weight caches as `[K*K*IC, OC, 1, 1]`; `scunet_run_conv` now restores the shape from call-site dims, with the conventions **measured** on the working f32 path (plain `[kw,kh,ic,oc]`, transpose `[kw,kh,oc,ic]`) rather than assumed. q8_0 vs f32 cosine `0.999999`/PSNR `60.54 dB`. `pan`/`swinir`/`tbsrn` clean both ways. Regression control: f32 output **byte-identical** before/after both patches. Suites after: tps-locnet 15/15, tps-warp 19/19, core-cpu-ops 118/118. | **COMPLETED** |
| 2026-08-03 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **h2ovl-mississippi-2b SHIPPED at q8_0 — the last 404 registry URL is closed.** Three invocation defects, all required, none in the graph (`test-internvl2-diff` was clean at f16 throughout): (1) wrong chat template — `h2ogpt2`, not ChatML, and this vocab has no `<|im_start|>` so every role marker was silently dropped (`5f617351`, mine); (2) spurious BOS + terse instruction, neither sufficient alone (`fcebf561`, parallel session — I verified `add_bos_token: false` directly in the checkpoint); (3) **the BOS default was evaluated before the vocab inference that sets `h2ogpt2`, so (2) was a no-op on every GGUF lacking the template key — i.e. all published artifacts.** Measured on the published q8_0: defaults → EMPTY, `ADD_BOS=0` → full transcription; after reordering, defaults transcribe (`98c2ae21`). Vision was never at fault — with only (1), `fox.png` read back exactly. **Quant ladder** (7 real stages vs the blueprint ref): f16 `0.999972`, q8_0 `0.998033`/`0.995498` transcribes, q4_k+attn-held-Q8_0 `0.922`/`0.544`, q4_k `0.594995`/`-0.268615` anti-correlated, q6_k **fails to load**. Q8_0 is the floor for this checkpoint; not a shape issue (2560/6912 are 256-divisible). **q4_k withdrawn from HF** — it emits confident wrong text, the worst OCR failure mode. `tools/quantize.cpp` **warns** rather than refuses: my first version refused for the whole arch and broke `internvl2-1b` (ships q4_k, works) and `h2ovl-800m` (recorded verified at q4_k) — caught by the regression check. **Shipped:** repo public + carded per-precision, registry → q8_0 (2292 MB), SHA-256 pinned via the generator, **242 pinned / 0 unpinned**, `--list-models` shows it. Reference at `cstr/crispembed-regression-fixtures` → `internvl2/h2ovl-mississippi-2b/ref.gguf`. Harness fix `e5db01f1` (diff no longer fabricates FAIL rows for absent stages). | **COMPLETED** |
| 2026-08-02 | `chore/ai-act-audit-fixes` / `.codex/worktrees/chore-ai-act-audit-fixes` | **Picked:** fourth AI Act audit, run against `main` at `52172c10`. Re-verified the code-backed claims and this time **executed** the gate against real GGUFs (yunet + sface pulled from `cstr/*`) rather than trusting that the test exists: all 8 prior cases pass, including the renamed-model case. Also verified the registry is clean of emotion/age/gender/ethnicity models (555 entries), that no training code exists (so §7's "quantization only" argument holds), and — since Reg (EU) 2026/1744 postdates the assistant's knowledge cutoff — checked POLICY's whole date table against the EUR-Lex text: **accurate**, including 2 Dec 2026 for the new Art. 5(1)(ba)/(bb) NCII/CSAM prohibitions. Three gaps found and closed here: (1) `--dim` returned *before* the gate on both CLI paths that reach it, making the CLI laxer than `crispembed_face_init()` — all three CLI sites now share one `cnn_biometric_ok()` helper keyed on declared type (which defaults to `recognition`, so it fails closed), and the gate test grew 3 cases; (2) POLICY §4 claimed "both gates key off declared type" while the `--face-pipeline` gate was unconditional-before-load — that path is now type-keyed too, so the sentence is true as written; (3) neither POLICY nor README told deployers that the server acknowledgement is **once per process** with **no authentication** and server-side-path input — documented in both, plus a startup warning when a recognition model is loaded on a non-loopback bind. Touches `examples/cli/main.cpp`, `examples/server/server.cpp`, `tests/test_biometric_gate.py`, POLICY/README/PLAN — **no OCR/model/graph code**. Gap (1) was mis-placed from the start, not a regression: `git log -L` shows the gate was added *below* the pre-existing `--dim` early-return in `6d87d6bd`. Verified before/after with the same toolchain — a CLI built from `HEAD:examples/cli/main.cpp` prints `128` (sface's template width) unacknowledged on **both** paths; the patched CLI refuses both. Gate test now 11/11 with real yunet+sface GGUFs, server warning fires on `--host 0.0.0.0` and stays silent on loopback, `--face-pipeline` still refuses without ack and runs with it, text embed + `--dim` unaffected, `format.sh --check` and `check_registry_licenses.py` clean | **COMPLETED** |
| 2026-08-02 | `feat/ocr-followups` / `.claude/worktrees/feat-ocr-followups` | **Picked:** the orphaned AdaIR F16 TODO from `feat/tesseract-kernel-opt` (that branch merged into `main` at `ee099eb0` and is gone; the item was left `IN PROGRESS`). Root cause identified before any edit: `tools/quantize.cpp` (~line 167) flattens every 4-D F32 conv weight to 2-D `[IC*KH*KW, OC]` in the output header, and `src/adair.cpp` infers three hidden dims from `->ne[3]`, which is `1` on a flattened tensor. Confirmed against the artifacts — `net.decoder_level1.0.ffn.project_in.weight` is `[1,1,96,510]` in `adair-5d-f32.gguf` and `[96,510]` in both `adair-5d-f16.gguf` and the rebuild. `hidden=1` ⇒ `half = hidden/2 = 0` ⇒ a conv with `ic=0` ⇒ the zero-size kernel descriptor the earlier audit saw. **No OCR/perf overlap — does not touch H1–H8 or any paused codex branch.** **Fixed:** `conv1x1_out_channels()` derives OC from `ggml_nelements(t)/ic`, correct under both layouts, with fail-loud guards at all three sites; `ADAIR_LEGACY_NE3_DIMS=1` keeps the old read so both arms are in one binary. Measured on `adair-ref.gguf` (64×64), same binary: f32 `cos 0.999382 / max_abs 0.027892` (reproduces the audit exactly ⇒ regression control), `adair-5d-f16.gguf` **`0.999383 / 0.027871`**, and the `adair-5d-f16-rebuilt.gguf` quantizer rebuild the audit also blamed gives the **identical** `0.999383 / 0.027871` — so neither artifact was ever bad. Independent artifact check: 60 sampled tensors f16-vs-f32 worst cosine `0.999998`, worst max_abs `1.22e-4`, i.e. pure F16 rounding. End-to-end through the real CLI (not just the diff harness): a 96×96 restore returns rc=0 on both models with the outputs agreeing at cosine `1.0`, max_abs `1/255`. **No timings claimed** — load average was 55–127 from parallel agents all session and the 64×64 fixture took `312 s` at f32 against a `2.65 s` quiet-box reference. Registry still ships f32 on purpose: repointing needs the f16 uploaded to `cstr/adair-GGUF` + a SHA-256 pin in `model_hashes.h`, which is an outward-facing step left for the owner. Exposure is narrower than "f16": only `tools/quantize.cpp` output flattens — a converter-emitted f16 keeps 4-D shapes (`surya-det-f16.gguf` has 79 genuinely 4-D F16 tensors), so the *producer* predicts the layout, not the precision. Follow-up recorded, not blind-fixed: `src/surya_det.cpp:700` and `src/tps_locnet.cpp:219` read conv OC off `ne[3]` the same way and would misread a quantizer-produced artifact (`src/cnn_embed.cpp:148` is the both-layouts precedent); neither ships one today and neither is verifiable on this box. | **COMPLETED — runtime fixed; f16 upload/registry DONE by `chore/ai-act-audit-round3` (adair-5d-f16 uploaded to cstr/adair-GGUF, registry repointed 115->59 MB, pinned)** |


### What landed in the 2026-08-02/03 OCR wave (the short version)

- **h2ovl-mississippi-2b now works and is shipped.** Three invocation defects,
  none in the graph: wrong chat template (`h2ogpt2`, not ChatML), a spurious BOS
  plus too-terse instruction, and the BOS default being evaluated before the
  vocab inference that selects the template. Verified to the logits — f16 is
  **exact on all 54 stages**, shipped q8_0 reaches `cos_glob` 0.998919.
  q4_k withdrawn (confident wrong text). Registry → q8_0, pinned.
- **h2ovl-800m** brought to the same standard: reference baked, q8_0 published,
  registry deliberately left on q4_k (edge model).
- **AdaIR F16** root-caused (quantizer flattens 4-D conv weights; three widths
  read off `ne[3]`) and shipped by `chore/ai-act-audit-round3`.
- **tps_locnet** channel inference fixed for quantized GGUFs; `surya_det` cleared
  of the same charge after checking.
- **esrgan / scunet** made runnable when quantized at all (graph-node budget;
  flattened kernel in the persistent cache) — publishing their q8_0 is N4 in
  PLAN.md.
- **Diff harness**: stopped fabricating FAIL rows for absent stages, now prints
  `cos_glob` + `|mine|`/`|ref|` (HARD RULE #2b), gained
  `CRISPEMBED_DIFF_COS_THRESHOLD` and `is_pass_global()`, and both h2ovl
  references were re-dumped to reach the logits.
