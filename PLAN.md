# CrispEmbed — Architecture & Roadmap

Lightweight, dependency-free text/image/audio embedding inference via ggml.
Same philosophy as CrispASR: pure C/C++, GGUF models, quantisation,
GPU-ready via ggml backends (CUDA/Metal/Vulkan), no Python at runtime.

## 🚧 Active work in flight (update + push to `main` at EVERY checkpoint)

Multiple sessions/worktrees run in parallel and push to `main` concurrently.
Before starting a task, add a row; at every checkpoint update it and push this
file to `main` so others see what's claimed (avoids duplicate work + CI-cancel
races). Remove the row when the branch lands.

| Since | Branch / worktree | Task | Status |
|-------|-------------------|------|--------|
| 2026-08-05 | `perf/ppocr-profile` (`.claude/worktrees/perf-ppocr-profile`) | **O13b — PP-OCRv6 end-to-end measure-first profile** (det/crop/rec/orientation split on the medium tier, Metal build, existing bench gates; goal: name the CURRENT hotspot before prescribing anything) | IN PROGRESS |
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
| 2026-08-05 | *(queued — launches after G4's model-verify finishes; one heavy model consumer at a time on this box)* | **Claimed (G6=F6):** quantify `DS2_KV_F16` vs F32 KV — decoded CER, memory, decode time, both backends, guard-on (default), both decode arms, against the `tests/results/f1/` baseline (T14-era numbers no longer reproduce post-tokenfix) | **QUEUED** |
| 2026-08-01 | `feat/ocr-engine-parity` / `.claude/worktrees/feat-ocr-engine-parity` | **Picked:** end-to-end head-to-head parity (CER/WER **and** latency) of the CrispEmbed OCR lanes against system Tesseract 5.5.2, Python EasyOCR 1.7.2, and Python PaddleOCR 2.10.0. See "OCR external head-to-head" below for the harness, the reachability fixes, and the first measured gaps. Touches `examples/cli/main.cpp`, `examples/cli/model_mgr.cpp`, `src/crispembed.{h,cpp}` engine-id mapping, `src/ocr_orchestrator.{h,cpp}` (new `engine::easyocr` case only), and new `tests/` scripts — **no OCR graph/runtime math** | **IN PROGRESS** |
| 2026-07-31 | `feat/easyocr-ggml` / `.codex/worktrees/feat-easyocr-ggml` | **Picked:** unify CRAFT/DBNet/Tesseract-style segmentation with EasyOCR lines and LayoutLM/Tesseract words; then validate downstream OCR handoffs. Latest checkpoint: fresh Latin Gen1/Gen2 and English fixed-width references pass; only English’s actual width-128 scan retains the documented dynamic-width row-wise logits residual | **IN PROGRESS** |
| 2026-08-02 | `feat/ppocr-next-20260731` | **Picked:** rework the tiny fused graph around an explicit per-item branch/sequence dimension that survives pooling, permutation, and CTC flattening on Metal; add a two-crop gold-logit cosine contract before considering any Metal batch execution. Keep `CRISPEMBED_PPOCRV6_BATCH_GRAPH` CPU-only until that contract passes | **IN PROGRESS** |
| 2026-08-04/05 | *(engine-portfolio round — ALL LANDED)* | T13 olmocr, T14 deepseek decode (`82ce1024`), T15 smoldocling (`7de85cb7`), T18 one-shot init (`c178308f`), granite-r2 (`110dd082`), tokenize_simple audit (`357dee53`), imatrix quants (`38c708b2`+`926df0ae` — 330m follow-up MERGED), metallib CMake pin (`9288d3b5`). Detail lives in the dated status blocks below and in `tests/results/*_2026-08-04.json`; do not re-derive | **DONE** |
| 2026-08-05 | *(landed, coordinator's own work)* | **F1 MERGED** (`e9f84f16`): deepseek-ocr2 no-repeat-ngram guard, default 20 per the contract, `DS2_NO_REPEAT_NGRAM=0` restores; helper hoisted to `src/core/no_repeat_ngram.h` shared by qwen2vl/internvl2/deepseek + hermetic `test-no-repeat-ngram` (13 checks, model-free CI). Full acceptance evidence in the commit message and `tests/results/f1/`; F1 status block below | **DONE** |
| 2026-08-05 | *(landed same day, wave 2)* | **F8 MERGED** (`f31c6531`, coordinator-verified: hermetic 24+15 checks re-run 0 failures, LaBSE battery 20/20 vs an independently REGENERATED HF golden, 0/20 on the unfixed binary, 4 shipped models token-id-IDENTICAL old-vs-new under my own runs). Verdict: nothing LaBSE-class was shipped; the CONVERSION PATH was broken 0/20 (converter >100k heuristic + runtime routing + per-byte pre-tokenizer) — all three layers fixed, absent-key = historical behavior. See §F8 outcome below. **F9b MERGED** (`3ade993a`, coordinator-verified: 15/15 copies hash-identical to CrispASR canonical `342c5f7f`, 13-test gate re-run per copy = 15×13/13, 8 upload-bearing drivers flipped to `resolve_hf_token(require=True)`) | **DONE** |
| 2026-08-05 | *(landed same day, wave 3)* | **F7b DONE** (kernel `chr1s4/crispembed-imatrix-t19` v3 complete; driver config merged `045102a0`; coordinator verified the `-f7` uploads exist on HF and the 4 pinned artifact SHAs still match `model_hashes.h` exactly). Headline: with real q/k/v importance, arctic q4_k+imatrix goes **.9480/.9614 → .9910/.9937 min/mean** (plain q4_k reproduced to 4dp; f2llm-80m control unchanged — comparability proven), and **q4_k+imatrix now BEATS iq4_xs+imatrix on the BERT side** (inverts T19-E3's IQ4_XS headline there; decoder side keeps the old ordering). Granite-r2 pair got first-time imatrix artifacts under canonical names. Decision items in §F7b below. **Test guards merged** (`fcc60afd`): `test-imatrix-alias` (59 checks, fails 44 on name drift; both naming sites now share `src/core/imatrix_alias.h`) + `tests/test_vendored_kaggle_harness.py` (106 checks × 15 copies, fails 5 on the pre-F9b copy), both in model-free CI | **DONE** |
| 2026-08-05 | *(landed same day)* | **F7 MERGED** (`68033e8d`, coordinator-verified: coverage 36→72-with-imatrix re-run independently, fresh collector imatrix has 12 per-layer `qkv_merged` entries and 0 `leaf_N`, hermetic battery re-run green, q4_k+imatrix now separates from plain q4_k — e5-small cos_min 0.9847→0.9889). Kaggle t19 re-collection/re-quant of every published BERT-family imatrix artifact is the follow-up (F7b below). **F9 MERGED in CrispASR** (`342c5f7f`, 13 hermetic tests re-run green). ⚠ F9 correction: canonical CrispASR harness already globbed both mount depths; the resolver that lost the t19 uploads is **CrispEmbed's stale VENDORED copy** — see F9b below | **DONE** |


## HANDOVER — round 8 (written 2026-08-05, after the round-7 session)

Round 7 is COMPLETE (evidence in the board rows above — do not re-derive):
**reranker -f7 re-pins** (`f34bf0b5`: jina q4k re-pinned, FIRST bge-v2-m3
sub-Q8 alias added, local Metal+CPU cross-check reproduced Kaggle to 4dp on
dscore), **mxbai q/k imatrix provenance A/B** (`f7d34896`: defect REAL —
bit-identical rel-pos direct entries shadow the merged vector — but NOT a
quality defect; default stays `direct`, opt-in
`CRISPEMBED_QUANT_IMATRIX_QKV=direct|merged|sum`; the Kaggle "mxbai
regressing tail" premise was a pre-g7c-base artifact, no re-ship needed),
**UOCR_* gate sweep** (`d45f3889`: 40 sites/17 vars value-parsed;
**`UOCR_PD=0` used to SEGFAULT with empty output** — crash-severity `=0`
inversion), and **v0.17.6 RELEASED** (16/16 assets; Windows CI fixed — it
had been red since `d04f3572` under cancelled/superseded runs).

### Remaining work, in value order (orchestrator Fable; per-task tiers noted)

- **OCR runtime residency backlog R1-R8** (new, unowned; see "OPEN TASKS — OCR
  runtime residency and optimization backlog" below and the `PERFORMANCE.md`
  survey it derives from). Highest-evidence item is **R1, Tesseract recognizer
  batching** — recognition is `38.34 s` of the `38.69 s` Fraktur page stage vs
  official's `9.34 s`, and the existing `…_REUSE_SCRATCH` prototype is
  unreadable because its run-to-run variance exceeds its effect (fix the
  measurement protocol first). **R3** (promote the six CPU-sched formula
  encoders to a GPU sched) is the cheapest work-to-payoff but must be one
  engine per A/B — conv-heavy graphs demonstrably lose on Metal, which is why
  DBNet and PP-OCRv6-det are CPU by default. **R5** (decode-step graph cache)
  is the one to be skeptical of: profile build+alloc as a fraction of decode
  before porting, because T14 did the work and won nothing.
- **Non-BENCH presence-gate sweep, remainder** (successor; Opus-capable with
  the ds/uocr-gates methodology, conversion-set + merge decisions
  coordinator's). UOCR_* is done; ~227 presence-based sites over ~139 vars
  remain, several output-affecting: `CRISPEMBED_PPOCRV6_FORCE_CPU`,
  `EASYOCR_FORCE_CPU`, `NAFNET_CPU`, `CRISPEMBED_TESSERACT_FORCE_CPU`,
  `SAFMN_SR_METAL`, `ESRGAN_SCALAR`, `CRISPEMBED_PPOCRV6_NO_GRAPH`,
  `GLM_OCR_DECODE_CACHE`, `CRISPEMBED_NO_KV_CACHE`, `LAYOUT_DETECT_FLASH`
  (full list `tests/results/bench-gates/SUMMARY.md` §out-of-scope). The UOCR
  precedent shows this class can be crash-severity, not cosmetic. Per-gate
  decoded-output A/B, three spellings; engine-sized batches, one engine per
  branch.
- **UOCR_PD segfault bisect** (new, unowned; Fable-tier — decode-graph work,
  never delegate the math). The opt-in persistent-decode path
  (`UOCR_PD=1`) faults at gen=2 in the compute, 7/7 reproductions,
  pre-existing on main, default path unaffected. Evidence pointers in
  `tests/results/uocr-gates/SUMMARY.md`; deepseek's working PD path
  (`DS_*` twin engine) is the reference implementation to diff against.
- **T16 (TableFormer port), T17 (Fraktur bisect)** — dedicated sessions,
  briefs in OPEN TASKS; Fable-tier. T16 still needs the A5
  document-structure gold.
- **N3 (OCR perf H2/H4/H5/H6), N4 (esrgan/scunet q8 publish — NEVER ship
  esrgan q4_k), N7 (OCR/VL quantize-and-run sweep)** — unowned, briefs in
  the round-2 archive.
- No release next round unless something lands: v0.17.6 is fresh.

### Discipline deltas learned THIS round (additive)

- **The cwd trap bit a merge this round**: an ff-merge run inside the
  feature worktree prints "Already up to date" + "Everything up-to-date" and
  main NEVER MOVES — it looks exactly like success. Merge from the main
  tree, confirm `git branch --show-current` = main, and judge by the push
  line showing `<old>..<new> main -> main`.
- **Check `gh run list` conclusions on main before trusting "CI is fine"** —
  Windows had been red for a full day of merges, masked by rapid pushes
  cancelling runs (cancelled ≠ green). Rapid-push cancellation makes red
  invisible unless you look at the last COMPLETED run per workflow. Always
  do this before a release bump.
- **MSVC portability**: no `setenv`/`unsetenv` (use `_putenv_s`); the
  Windows CRT cannot represent a set-but-empty env var (`set FOO=` deletes
  it) — compile such checks out on `_WIN32` rather than faking them.
- **Read a Kaggle ab-record's `quant_src=` header before trusting its
  verdict** — the "mxbai imatrix regression" was measured against the
  broken pre-g7c base; on the corrected base the sign flips. A/B records
  inherit the defects of their baseline artifact.
- **tau on the 30-pair RERANK_EVAL fixture has a ±0.009–0.013
  cross-build/backend near-tie band** (q8_0 itself swings 0.009 between
  CPU and Metal). Judge imatrix quality by mean|dscore| (continuous);
  use tau only for orderings, never for sub-0.01 verdicts.
- **Marker-file gating works for cross-agent heavy-run serialization**: the
  coordinator touches `<scratch>/MATRIX_DONE` when its heavy runs end; the
  agent's brief says poll-then-serialize. Round 7 ran three lanes of model
  runs on the 16 GB box with zero overlap incidents.

### Environment as left (2026-08-05, post-round-7)

- Main volume ~16 GB free. Session scratchpad model caches deleted (repin
  evidence copied into `tests/results/repin-f7/`). `/tmp/crispembed-regression`
  untouched. `~/.cache/crispembed-local/` unchanged.
- **v0.17.6 is the latest tag** (16 assets verified). main = `5f756ab5`+.
  Round-7 worktrees/branches all removed; the three pre-existing IN PROGRESS
  rows (feat/ocr-engine-parity, feat/easyocr-ggml, feat/ppocr-next-20260731)
  + older .codex worktrees remain — check the board before touching.
- No new HF artifacts this round (re-pins/aliases only point at the
  round-6 `-f7` uploads; nothing replaced).
- HF account cstr, token `../.env`; always `HF_HOME=~/.cache/hf-<task>` or
  the session scratchpad. Kaggle chr1s4, one kernel at a time. Python
  `/Users/christianstrobele/miniconda3/bin/python` — NOT for torch parity
  on BERT-class forwards (ONNX Runtime instead). CrispASR unchanged this
  round (their main `057ce9f3`+; push a ## CLAIMED block before touching).

## HANDOVER — round 7 (ARCHIVED 2026-08-05; all four lanes + release consumed — see round 8 above)

Round 6 is COMPLETE (evidence in the board rows above — do not re-derive):
**G8=F10** (CrispASR `fd3c0e5e`: T18 cpu short-circuit synced, a pre-existing
`--gpu-backend cpu`/`metal` LID crash found+fixed in their vendored whisper
wrapper, PLAN #88 write-path DECIDED with the cache cap adopted there),
**mxbai GELU A/B** (`0a72e267`: erf-exact now the pooler default, server
`/rerank` process-abort on quantized 2-layer rerankers fixed — was live for
jina-reranker-v2), **mxbai artifact re-ship** (`da0272e8`: the shipped pair
had NO ContextPooler — near-inverted xsmall rankings; 10 `*-g7c.gguf`
uploaded, 2 pins re-pointed, fresh-download verified), **`*_BENCH`
presence-gate audit** (`d04f3572`: 68 sites → `core_env::on()` in
`src/core/env_gate.h`, hermetic `test-env-gate` in model-free CI), and
**reranker imatrix re-collection** (`87e11a4e`: 6/7 published reranker
imatrices had the `leaf_N` defect; 7/7 re-collected on correct bases, 29
files uploaded, pipeline now RAISES on `0 with imatrix`).

### Remaining work, in value order (orchestrator Fable; per-task tiers noted)

- **Reranker sub-Q8 re-pin decisions** (coordinator, small; measurement
  delegable). The imatrix row records the numbers: jina
  `-q4_k-imatrix` (SHA-pinned, `examples/cli/model_hashes.h:251`) and
  bge-reranker-v2-m3 sub-Q8 aliases are the clear `-f7` candidates
  (tau +.031 / dscore −26% for bge-m3). Local-Metal cross-check FIRST
  (G3 precedent; Kaggle A/B was x86-CPU-only).
- **mxbai DeBERTa q/k imatrix provenance A/B** (small, Opus-tier with
  gates). Recorded in the imatrix row: `quantize.cpp` prefers the direct
  `blk.N.attn_{q,k}` match over the merged alias, and DeBERTa-v2 applies
  q/k a second time to rel-position embeddings — so mxbai q/k importance
  was collected over the WRONG inputs. Options: prefer the merged alias for
  DeBERTa, or accumulate both. Judge by decoded rerank scores.
- **Output-affecting presence-gate sweep** (successor to the BENCH audit;
  session-sized, NOT mechanical). 267 presence-based sites over 156
  non-BENCH vars remain; many select compute paths (`=0` changes OUTPUT).
  Priority cluster: `src/unlimited_ocr.cpp`'s ~40 `UOCR_*` gates — the
  exact mirror of the fixed `DS_*` set. Needs a per-gate output A/B, the
  ds-gates methodology (`tests/results/ds-gates/run_gates.sh` is the
  template).
- **T16 (TableFormer port), T17 (Fraktur bisect)** — dedicated sessions,
  briefs in OPEN TASKS; Fable-tier, never delegate the math. T16 still
  needs the A5 document-structure gold.
- **N3 (OCR perf H2/H4/H5/H6), N4 (esrgan/scunet q8 publish — NEVER ship
  esrgan q4_k), N7 (OCR/VL quantize-and-run sweep)** — unowned, briefs in
  the round-2 archive.
- **A release is now credible**: post-v0.17.5 main carries the DS_ audit,
  G7c, the mxbai rerank fixes (crash + calibration), the BENCH audit, and
  the imatrix tooling — a reasonable v0.17.6 for a session that wants one
  (CrispASR-style process: RELEASE_NOTES + scripts/bump-version.sh).

### Discipline deltas learned THIS round (additive)

- **A device-pref filter must be applied at BOTH backend-init AND
  weight-buffer planning.** CrispASR's whisper wrapper filtered devices in
  `whisper_backend_init_gpu` but not `make_buft_list` → weights on a device
  the sched doesn't carry → `sched_backend_id_from_cur` abort. Any future
  per-device filter: grep for every place that enumerates devices.
- **Don't assume the cwd reset — verify with `pwd`.** The reset-to-main-tree
  behavior is real but not universal; this round a format+build "in the main
  tree" had actually persisted in a worktree (harmless here, but the inverse
  mistake runs stale main-tree binaries). Absolute paths remain the rule.
- **`cmake --build build --target <cli>` can report "Built target" without
  recompiling a changed source** when the object belongs to a sibling
  library target (make-based dirs). Judge by the `Building CXX object` line
  for the file you changed, and re-run the behavioral check after ANY
  rebuild that shows no compile line.
- **New boolean env gates use `core_env::on()`** (`src/core/env_gate.h`,
  guarded by `test-env-gate` in model-free CI). Never write
  `getenv(X) != nullptr` again.
- **Duplicated lazy-init blocks are a crash surface** — the `/rerank` abort
  came from a second copy of the classifier-cache population that had not
  received the single-doc path's dequant fix. Populate caches in exactly
  one place.

### Environment as left (2026-08-05, post-round-6)

- Main volume ~25 GB free. Session scratchpad GGUF/HF caches deleted.
  `/tmp/crispembed-regression` untouched. `~/.cache/crispembed-local/`
  unchanged, all registry-pinned.
- v0.17.5 latest tag; main = `da0272e8`+ (this handover lands after it).
  Round-6 worktrees/branches all removed; the three pre-existing IN
  PROGRESS rows (feat/ocr-engine-parity, feat/easyocr-ggml,
  feat/ppocr-next-20260731) + older .codex worktrees remain — check the
  board before touching.
- New HF artifacts this round: `cstr/mxbai-rerank-{xsmall,base}-v1-GGUF`
  `*-g7c.gguf` (f16 + 4 quants each, READMEs note the defect, old files
  kept for old pins) and 29 reranker `-f7` imatrix/quant files across 7
  repos (ms-marco composed as `-g7c-f7`). All new mxbai q8_0 pins
  fresh-download SHA-verified.
- CrispASR main `057ce9f3`+ (G8 landed there as `fd3c0e5e`; their box
  stays hazardous — check their PLAN before claiming anything).
- HF account cstr, token `../.env`; always `HF_HOME=~/.cache/hf-<task>` or
  the session scratchpad. Kaggle chr1s4 (new kernel
  `chr1s4/crispembed-imatrix-rerank-f7` v1 good), one kernel at a time.
  Python `/Users/christianstrobele/miniconda3/bin/python` — NOT for torch
  parity on BERT-class forwards (ONNX Runtime instead).

## HANDOVER — round 6 (ARCHIVED 2026-08-05; all five lanes consumed — see round 7 above)

Round 5 is COMPLETE (evidence in the board rows above — do not re-derive):
**v0.17.5 was cut by a parallel session** (`51e7d729` bump + tag; my merges
landed post-tag), **DS_ value-parse audit** (`91ebb55d`: every presence-based
boolean gate in deepseek_ocr2.cpp value-parsed via `ds_env_on()`, incl. two
finds beyond the brief — `DS2_FORCE_CPU`, `DS_PROFILE`; new `DS_DBG=1`
gate-resolution stderr line; 42/42 three-spelling checks,
`tests/results/ds-gates/`), and **G7c expanded** (`63997e2c`: shipped
ms-marco rerankers had NO BertPooler stage — scores ±0.2 instead of ±11,
tail reordered; converter-only fold to the 2-layer tanh head, f16 ≤0.0009 vs
the ONNX reference, 10 `*-g7c.gguf` artifacts uploaded + 4 pins re-pointed,
`tests/results/g7c/SUMMARY.md`). **G7b DECIDED closed** (no ST-pooler parity
path; G7a precedent). LEARNINGS' 2026-07-03 ms-marco RANK-head claim
corrected in place.

### Remaining work, in value order (model-tier notes: the orchestrator
should be Fable; per-task tiers noted)

- **G8 = F10, CrispASR twins** (other repo, coordinate; Opus-capable with a
  strict brief, Fable preferred for the decisions). Recon 2026-08-05: their
  `gpu_backend_pref.h` still lacks the T18 `--gpu-backend cpu` short-circuit;
  PLAN #88 (pipeline-cache write path) unclaimed. HAZARDS unchanged: backups
  disk ~1.8 GB free, several concurrent agents, load spikes, CI perpetually
  cancelled (not a signal). Push a CLAIMED block to their main BEFORE
  starting; sync logic not bytes (pcs.cpp rule).
- **mxbai erf-vs-tanh GELU A/B** (new rider from G7c; small, Opus-level with
  strict gates). The DeBERTa ContextPooler in `crispembed_apply_classifier`
  uses tanh-approx GELU where HF `gelu` is erf-exact (same class as the
  granite projector finding). One variable; judge by decoded rerank scores vs
  an ONNX reference (NOT local torch — see discipline below); mxbai pair only.
- **Reranker imatrix re-collection** (F7b leftover; Opus-level — established
  t19 Kaggle pipeline). All published reranker `.imatrix` files are pre-F7
  (no attn q/k/v coverage). Note the ms-marco ones must be re-collected on
  the `-g7c` artifacts.
- ~~**`CRISPEMBED_*_BENCH` presence-gate audit**~~ — DONE on
  `feat/bench-gates` (68 sites / 60 files through `core/env_gate.h`); see the
  board row. **Successor, unowned:** the same `=0`-inverts sweep for the 267
  presence-based NON-BENCH gates (156 distinct vars). That one is NOT
  mechanical — many select a backend or compute path, so each needs a decoded
  output A/B, not a compile+smoke. Start with `unlimited_ocr.cpp`'s ~40
  `UOCR_*` gates: they mirror the already-fixed `DS_*` set one-for-one.
- **T16 (TableFormer port), T17 (Fraktur bisect)** — dedicated sessions,
  briefs in OPEN TASKS. Both are graph/decoder-semantics work: **Fable-level,
  never delegate the math** (dev-guide rule). T16 still needs the A5
  document-structure gold.
- **N3 (OCR perf H2/H4/H5/H6)** — Opus-level with the standing timing
  discipline; brief in the round-2 archive. **N4 (esrgan/scunet q8 publish —
  NEVER ship esrgan q4_k)** and **N7 (OCR/VL quantize-and-run sweep)** —
  mechanical with clear gates, Opus-level; briefs in the round-2 archive.
- No release this round: v0.17.5 is fresh; accumulated post-tag main (DS_
  audit + G7c) is not yet a v0.17.6.

### Discipline deltas learned THIS round (additive)

- **The shell cwd resets to the MAIN tree after any command that `cd`s
  elsewhere** — a bare `./build/crispembed` then runs the main tree's stale
  binary (this round rebuilt and "verified" the wrong build before catching
  it). Run worktree binaries by ABSOLUTE path, or re-`cd` in every command.
- **Local miniconda torch mis-executes BERT-class forwards** (all-NaN padded
  batches, bus errors in tiny Linears, garbage orderings; fresh re-download
  did not help). Parity references on this box come from ONNX Runtime
  (`Xenova/<model>` exports are faithful; onnxruntime 1.25.1 in miniconda) or
  a remote box. A broken reference nearly mis-attributed G7c.
- **Conversion mode must match the published `.imatrix` names**: a `--crisp`
  re-conversion of an ollama-mode artifact gets `0 with imatrix` (silently
  no-importance quants). Always read the quantizer's `N with imatrix` line.
- **Replacing HF artifacts in place breaks released binaries' SHA pins** —
  ship fixes under task-suffixed names (`-g7c`, G3's `-f7` precedent), keep
  the old files, re-point registry + `model_hashes.h`.
- **macOS ships bash 3.2**: no `declare -A` in test runner scripts (a
  comparison block died on it; the runs survived, the comparisons re-ran
  standalone).
- **"Verified vs upstream" claims can be code-level only** — G7c's defect
  hid behind a LEARNINGS claim that never inspected the shipped GGUF's
  tensor list. Verify artifact-level: read the tensor names.

### Environment as left (2026-08-05, post-round-5)

- Main volume ~26 GB free. `/tmp/crispembed-regression` intact (~8.4 GB,
  ephemeral). `~/.cache/crispembed-local/` unchanged. Session scratchpad
  cleaned (GGUFs/ONNX deleted).
- v0.17.5 is the latest tag. Round-5 worktrees/branches removed; the three
  pre-existing IN PROGRESS board rows (feat/ocr-engine-parity,
  feat/easyocr-ggml, feat/ppocr-next-20260731) + older .codex worktrees
  remain — check the board before touching.
- New HF artifacts: `cstr/ms-marco-MiniLM-L-{6,12}-v2-GGUF` `*-g7c.gguf`
  (f16 + 4 quants each), READMEs note the fix; old files retained for old
  releases' pins. All 4 new pins fresh-download SHA-verified.
- HF: account cstr, token `../.env`, always `HF_HOME=~/.cache/hf-<task>` (or
  scratchpad). Kaggle chr1s4, one kernel at a time. Python
  `/Users/christianstrobele/miniconda3/bin/python` (but NOT for torch parity
  references — see discipline).

## HANDOVER — round 5 (ARCHIVED 2026-08-05; DS_ audit + G7b/c consumed — see round 6 above)

Round 4 is COMPLETE (both items coordinator's own work, evidence in the board
rows above — do not re-derive): **G1** (SmolDocling vision split residency
`703161b1`+`ad28b77e`, GPU default, vision 2.9-4.6× on Metal,
`tests/results/g1/SUMMARY.md`) and **G2b** (`8c210291`, `DS2_CROP_MODE`
default ON after the regressions were proven formatting-only,
`tests/results/g2b/SUMMARY.md`). New shared infra: `core_gguf::
load_weights_split` (CrispASR #69a logic) is now available to every engine.

### Remaining work, in value order

- **G7b/c** (LaBSE ST-pooler parity product decision; `bert.pooler_act`
  gelu-vs-tanh A/B) — unchanged, briefs in the round-3/round-2 archives.
  **G7a decided this round: NOT publishing a LaBSE GGUF** (no demand signal
  two rounds running; the fixed converter on main regenerates everything, so
  the `hf-f8` leftovers were deleted per the regenerate-don't-trust rule).
- **G8 = F10, CrispASR twins** — recon done 2026-08-05: their
  `gpu_backend_pref.h` still lacks the T18 cpu short-circuit and PLAN #88 is
  unclaimed on their board, BUT their box is hazardous (backups disk ~1.8 GB
  free, several concurrent agents, load spikes 100+). Claim with a CLAIMED
  block pushed to their main first; verify locally per their conventions
  (their CI is perpetually cancelled — not a signal).
- **DS_* value-parse audit** (new, small, unowned) — see the G2b board row.
- **T16 (TableFormer), T17 (Fraktur bisect)** — dedicated sessions, briefs in
  OPEN TASKS. **N3 (OCR perf H2/H4/H5/H6), N4 (esrgan/scunet q8 publish —
  NEVER ship esrgan q4_k), N7 (OCR/VL quantize-and-run sweep)** — unowned,
  briefs in the round-2 archive sections.
- A release: accumulated main (G1-G7d, crop default, cache cap, arctic
  re-pin, thread default) is a strong v0.17.5 — CrispASR-style process
  (RELEASE_NOTES + scripts/bump-version.sh), still uncut.

### Environment as left (2026-08-05, post-round-4)

- Main volume ~28 GB free. `~/.cache/hf-f7` and `~/.cache/hf-f8` DELETED
  (G3 done / G7a decided). `/tmp/crispembed-regression` (~8.4 GB, both
  deepseek gold GGUFs) intact — reboot-ephemeral. `~/.cache/crispembed-local/`
  unchanged, all registry-pinned.
- Round-4 worktrees/branches removed. The three pre-existing IN PROGRESS
  board rows + older .codex worktrees remain — check the board before
  touching. No tag cut this round.
- Discipline deltas THIS round (additive): (1) result-dir `.txt` framing —
  the g2 corpus runner strips the CLI's trailing newline, so raw-CLI captures
  cmp as DIFF against recorded arms; normalize trailing newlines before
  byte-comparing. (2) run_one.py needs miniconda python (system python3
  lacks huggingface_hub). (3) The T15 "31.7 s fox vision" number was a
  different CPU-only build — same-binary baselines only (G1 re-learned it).

## HANDOVER — round 4 (ARCHIVED 2026-08-05; G1/G2b consumed — see round 5 above)

Round 3 is COMPLETE — every G-item except G1/G8 consumed, all
coordinator-verified before merge: **G2** (deepseek dynamic-crop port
`d5788a88`+`e81c827e` — CPU cc0 CER now BEATS the A4 reference, Metal german
1024-cap fixed; opt-in `DS2_CROP_MODE=1`), **G3** (arctic sub-Q8 aliases
re-pinned to `-f7` `464f812f`, granite-r2 alias decision: none), **G4**
(Metal cache cap in every GPU lane `c1ccb1f4`; 683 MB archive DELETED),
**G5** (embed one-shot `-t` default → min(4,cores) `5fcd7006`), **G6**
(`DS2_KV_F16` quantified, stays opt-in; gate value-parsed `73beea9f`),
**G7d** (driver fail-fast `10d160ba`). Evidence: board rows above,
`tests/results/g2/SUMMARY.md`, `tests/results/g6/SUMMARY.md`. Do not
re-derive.

### Remaining work, in value order

**G1 = F4 — SmolDocling vision backend split-residency (OWN WORK, quiet
box).** Brief unchanged in the round-2 archive below. G4 confirmed at run
time the engine is CPU-only today (`ggml_backend_cpu_init`,
src/smoldocling_ocr.cpp:297) — exactly what this item changes.

**G2b — deepseek crop-mode follow-ups (new, from G2's acceptance).**
(a) The Metal `receipt_historical` CER regression under crops (0.138→0.305)
is FORMATTING drift, not content garbage — Metal's decode wraps items in
heavier markdown (`- **item**: price`) than the plain-text GT; CPU reads the
same content at 0.135. Diagnose why the Metal trajectory goes markdown-heavy
(same class as the T14 near-tie divergence), then (b) decide the
`DS2_CROP_MODE` default flip — the reference contract runs crop_mode=True,
so default-ON is the contract-faithful end state; the flip is a coordinator
decision and also needs the synth_01_noise 0.015→0.045 delta re-examined.

**G7a/b/c — LaBSE/WordPiece leftovers (small, unowned).** (a) publishing a
LaBSE GGUF stays OPTIONAL (no demand signal this round — convert with
`--crisp`, battery, upload, pin; REGENERATE the f16, don't trust leftovers);
(b) ST `2_Dense`==BertPooler parity (cos ≈ −0.05 vs full ST stack) — wants a
product decision; (c) `bert.pooler_act` gelu-vs-tanh default (rerank-only
today) — changing it perturbs rerank outputs, needs its own A/B.

**G8 = F10 — CrispASR twins (other repo, coordinate before touching).**
Brief in the round-2 archive. CrispASR main was active again today.

**T16 (TableFormer), T17 (Fraktur bisect)** — dedicated sessions, briefs in
OPEN TASKS. **N3 OCR perf H-items, N4 esrgan/scunet q8 publish, N7 OCR/VL
quantize-and-run sweep** — unowned, briefs in the round-2 archive sections.

### Discipline deltas learned THIS round (additive)

- **Value-parse env gates; presence-based gates invert `=0`.** `DS2_KV_F16=0`
  ENABLED f16 until `73beea9f`. When touching any engine, check its gates for
  the `getenv(X) ?` pattern before A/B-ing with `X=0`.
- **Hoisting an Apple-specific header into a shared header breaks non-Apple
  builds** — G4's hoist needed a platform guard, caught and fixed by a
  parallel session (`bbc2a516`). Guard before pushing, not after CI reds.
- **Read the transcripts before classifying a CER delta.** The "Metal crop
  regression" is markdown-formatting drift with correct content; a CER
  number alone would have mis-filed it as a vision bug.
- **Serialize heavy work even when only correctness is claimed.** Running
  the G2 matrix + G3 downloads + the G4 agent concurrently produced a
  69-minute page decode (results valid, wall-clock wrecked). One heavy
  consumer at a time is also a throughput rule.
- **`tools/format.sh --fix` prints "rewrote N files" even when bytes are
  unchanged** (idempotent output) — don't panic-rebuild on the message, but
  the cheap rebuild habit stays correct.
- **Gold-gate artifacts cache under `/tmp/crispembed-regression/`**
  (`run_one.py --work-dir` default, `REGRESSION_WORK` env) — NOT
  `~/.cache/hf-regression` (a round-3 note said hf-regression was the cache;
  it never was for run_one; /tmp is reboot-ephemeral, so gold gates after a
  reboot re-download ~4.5 GB).
- **Main moves under you mid-round** (two pushes from parallel sessions
  today) — always `git fetch` + rebase before the ff-merge push; the board
  table prevented all duplicate work.

### Environment as left (2026-08-05 late)

- Main volume ~23 GB free. `~/.cache/hf-f7` grew to 2.9 GB (arctic f32 gold
  + 5 quants — served G3's cross-check, now DELETABLE). `~/.cache/hf-f8`
  (3.5 GB) still deletable once G7a is decided. `/tmp/crispembed-regression`
  holds ~4.5 GB of gold-gate deepseek artifacts (ephemeral, safe to leave).
  `~/.cache/crispembed/arctic-embed-m-v2-q4_k-imatrix-f7.gguf` is the newly
  pinned registry artifact (keep).
- **The 683 MB Metal shader archive is DELETED** (G4's scheduled step). It
  can only regrow from long-running processes (one-shot CLIs `_exit()` before
  the write); the cap keeps any regrowth bounded at open time.
- All round-3 worktrees/branches removed. Remaining worktrees belong to the
  three pre-existing IN PROGRESS sessions (board table) + older .codex ones.
- Kaggle unchanged (`chr1s4/crispembed-imatrix-t19` v3 latest good run).
  v0.17.4 remains the latest tag; this round shipped no tag — the accumulated
  main (crop port, cache cap, re-pin, thread default) is a reasonable v0.17.5
  candidate for a session that wants a release.

## HANDOVER — round 3 (ARCHIVED 2026-08-05 late; G2-G7d consumed — see round 4 above; G1/G8 briefs still live below)

Read this section, the "Active work in flight" table above, and the status
blocks it references BEFORE doing anything. The 2026-08-05 follow-up round is
COMPLETE: **F1** (deepseek no-repeat-ngram guard, `e9f84f16`, full status
block below), **F7+F7b** (imatrix QKV coverage fix `68033e8d` + Kaggle
re-run — arctic q4_k+imatrix .9614→.9937 mean, `-f7` artifacts on HF, pins
untouched), **F8** (LaBSE-class WordPiece conversion path was broken 0/20 —
three-layer fix `f31c6531`), **F9+F9b** (CrispASR harness fail-fast
`342c5f7f` + all 15 stale vendored copies re-synced `3ade993a`), and
hermetic CI guards around every fix (`fcc60afd` + `test-no-repeat-ngram`,
each verified to FAIL on the defect it guards). All coordinator-verified
before merge; evidence in the status blocks and `tests/results/f1/`. Do not
re-derive any of it.

**Session shape that worked twice now, recommended again:** one heavy item
as the orchestrator's own work, the rest delegated with acceptance-gated
briefs the coordinator re-verifies BEFORE merging (re-run hermetic tests
yourself, regenerate goldens independently, spot-run artifacts). Agent
output is plausible-until-verified — this round two agent briefs were
CORRECTED by verification (F9: the stale resolver was CrispEmbed's vendored
copy, not CrispASR canonical). Default flips, promotions, pin changes, and
ground-truth edits are never delegated.

### Remaining work, in value order (briefs live in the archived handover below unless restated)

**G1 = F4 — SmolDocling vision backend port (OWN WORK, do not delegate).**
Brief unchanged below. Needs a QUIET box (it is graph/residency A/B work) —
do not run it alongside delegated model-running agents. Remember the
worktree Metal trap in the discipline deltas below.

**G2 = F5 — DeepSeek-OCR2 dynamic-crop port (session-sized; value ROSE
with F1's data).** Brief unchanged below, plus new evidence from the F1
matrix (`tests/results/f1/`): the remaining cc0 gap is now clearly
crop-mode + a METAL-SPECIFIC trajectory problem — CPU reads the cc0 set at
mean CER 0.25-0.28 vs Metal 0.66, and `german_official_print` loops-with-
varying-tokenization ONLY on Metal (still caps at 1024 even guarded; exact
ngram bans cannot break a loop that re-tokenizes itself). Port the
reference's crop logic (blueprint line-by-line), gate it separately from
the F1 guard, re-run the F1 matrix arms + gold gate. If the Metal german
cap survives crop mode, it becomes its own Metal-numerics item.

**G3 — arctic imatrix re-pin decision (coordinator, small; measurement
delegable).** §F7b outcome above: the shipped pinned
`arctic-embed-m-v2-q4_k-imatrix.gguf` measures far below the `-f7` re-quant
(.9614 vs .9937 mean, Kaggle x86 CPU). Do the local-Metal cross-check
(e5-f32 + imatrix artifacts cached in `~/.cache/hf-f7`; T19-E3 saw ~0.002
backend delta), then re-pin the registry alias to the `-f7` artifact and
update `model_hashes.h`. q8_0 stays default regardless. Also decide whether
granite-r2's new canonical-name imatrix artifacts get registry aliases.

**G4 = F2 — Metal pipeline-cache cap adoption across the other Metal lanes
(delegable).** Brief unchanged below. The 683 MB archive at
`~/Library/Caches/ggml-metal/` is STILL on disk; delete it once the cap is
adopted everywhere.

**G5 = F3 — embed-CLI `-t 1` default (coordinator decision, small).**
Brief unchanged below (T18 data).

**G6 = F6 — quantify DS2_KV_F16 (delegable, small).** Brief unchanged
below. Note it now composes with F1: run it guard-on (the default), both
arms, and use `tests/results/f1/` as the comparison baseline — the T14-era
numbers no longer reproduce post-tokenfix (see the F1 status block).

**G7 = F8b — LaBSE/WordPiece follow-ups (delegable, small).** §F8 outcome
above: (a) optionally publish a LaBSE GGUF (convert with `--crisp`, battery,
upload, pin — the fixed converter is on main; agent's fixed f16 lives in
the session scratchpad but REGENERATE, don't trust a leftover); (b) ST
`2_Dense`==BertPooler parity gap (cos ≈ −0.05 vs full ST stack) — decide
whether CLS+pooler-tanh parity is wanted; (c) `bert.pooler_act` gelu-vs-tanh
default (rerank-only today); (d) flip `unlimited-ocr-convert` /
`crispembed-splade-fix` / `deepseek-ocr2-convert` drivers to
`resolve_hf_token(require=True)` (they bootstrap kh from the CrispASR clone
so they already have the resolver, not the fail-fast).

**G8 = F10 — CrispASR twins (other repo, coordinate before touching).**
Brief unchanged below. Note CrispASR main is active (another session pushed
`f0f9f242` today) — fetch + check its PLAN before claiming.

**T16 (TableFormer) and T17 (Fraktur bisect)** — unchanged, dedicated
sessions; briefs in OPEN TASKS below. **N3 OCR perf H-items, N4 esrgan/scunet
q8 publish, N7 OCR/VL quantize-and-run sweep** — still unowned, briefs in
the archived handover's board sections below.

### Discipline deltas learned THIS round (additive to the archived ones)

- **A fresh worktree's cmake configures GGML_METAL=OFF on this box** (bit
  T19-E4 and now F1). Always `-DGGML_METAL=ON` explicitly, then verify
  `GGML_METAL:BOOL=ON` in CMakeCache AND MTL0 in the run's stderr. The
  metallib EMBED pin (`9288d3b5`) works once Metal is actually ON.
- **The backend device name prints ONLY with an explicit `--gpu-backend`
  flag** — the default `ggml_backend_init_best()` path is silent, so "no
  MTL0 in stderr" on a default run proves nothing in either direction. Pass
  `--gpu-backend metal` / `cpu` explicitly on EVERY A/B arm so each run's
  own stderr carries backend proof. (This is how F1 caught its own
  CPU-mislabelled-as-Metal smoke runs — timings nearly identical across
  "backends" is the tell.)
- **When a per-arm identity gate fails, run the baseline (feature-OFF) arms
  before concluding.** F1's CPU cc0 "failure" was fully pre-existing —
  guard-off arms diverged at the SAME first byte. Attribution turned a
  blocked gate into an accepted, explained one in ~20 min of compute.
- **Exact-ngram repetition bans cannot break loops that vary their
  tokenization** ("Aufraktvert ren"/"Aufraktvertre ten"). Record such pages
  as decode-trajectory problems, not guard failures.
- **Two-dot `git diff origin/main` on a pre-rebase branch shows phantom
  reversions** of everything main gained since the branch point (misread
  twice this round, F8 and F7b). Use three-dot `origin/main...HEAD` (or
  `git show --stat` per commit) to see a branch's real change set.
- **`git worktree remove` fails on worktrees containing submodules** —
  use `--force`, or `rm -rf` + `git worktree prune`.
- **Agent briefs must forbid box-wide process kills.** One agent ran
  `pkill -f ninja` to retarget its own build and could have killed a
  parallel session's build. Put "never pkill/killall anything you did not
  start" in every brief on this shared box.
- **`format.sh` runs as a pre-commit hook here** — if you formatted after
  testing, the committed bytes are the formatted ones; rebuild+rerun the
  cheap hermetic targets post-format (non-semantic, but proves the
  committed state is the tested state).

### Environment as left (2026-08-05 evening)

- Main volume ~24 GB free (was 42 — session caches below account for it).
  `~/.cache/hf-f8` (3.5 GB, LaBSE f16s + HF snapshot) is DELETABLE once G7a
  is decided; `~/.cache/hf-f7` (455 MB, e5 f32 + shipped imatrix) KEEP for
  G3's cross-check; `~/.cache/hf-regression` (~4.5 GB, both pinned deepseek
  q4_k GGUFs) KEEP — it makes future gold-gate runs download-free.
- `~/.cache/crispembed-local/` unchanged from the last handover (all
  registry-pinned). New HF artifacts: `cstr/{arctic-embed-m-v2,f2llm-v2-80m}-GGUF`
  `-f7` imatrix quants + ab files; `cstr/granite-embedding-{97m,311m}-multilingual-r2-GGUF`
  first-time imatrix artifacts (canonical names). All pinned SHAs verified
  untouched.
- The 683 MB Metal shader archive is still at `~/Library/Caches/ggml-metal/`
  (G4 deletes it). v0.17.4 remains the latest tag; the round shipped no tag.
- All this session's worktrees and branches are removed. Remaining
  worktrees belong to other sessions — check the board table before
  touching. CrispASR main = `f0f9f242` (active today; F9 landed there as
  `342c5f7f`).
- Kaggle: `chr1s4/crispembed-imatrix-t19` v3 is the latest good run; one
  kernel at a time; the t19 driver now hard-fails without an HF token.

## HANDOVER — follow-up round (ARCHIVED 2026-08-05 evening; F1/F7/F8/F9 + b-items consumed — see round 3 above; the F2/F3/F4/F5/F6/F10 briefs below are still the canonical briefs)

Read this section, the "Active work in flight" table above, and the dated
status blocks it points to BEFORE doing anything. The 2026-08-04/05
engine-portfolio round is COMPLETE: all six lanes (T13/T14/T15/T18,
granite-r2, tokenize_simple audit, imatrix quants) are merged to main,
coordinator-verified, with results artifacts under `tests/results/` and
shipped models re-pinned. v0.17.4 is tagged. What remains is the follow-up
backlog below, in value order, each self-contained.

**Session shape that worked and is recommended again:** one heavy item as the
orchestrator's own work, the rest delegated to agents with acceptance-gated
briefs the coordinator verifies BEFORE merging (re-run their hermetic tests
yourself, regenerate any goldens independently, spot-run one shipped artifact,
fresh-download registry entries). Agent output is plausible-until-verified.
Default flips, promotions, and ground-truth edits are never delegated.

### F1 — DeepSeek-OCR2 repetition guard (HIGHEST VALUE, delegable with strict gates)

The lane implements NO repetition guard while the reference contract
(`tests/regression/gold/deepseek-ocr2/contract.json`) specifies
`no_repeat_ngram_size=20`. 2 of 5 cc0 pages spiral into the 1024-token cap
(commons_test_ocr_document loops "and that they were filled with rubbish",
simple_form loops a box list) — that alone drives cc0 CER to ~1.06 while
`receipt_historical` already BEATS the reference when decode terminates
(0.1198 vs 0.3633). **Do:** port `argmax_no_repeat_ngram` from
`qwen2vl_ocr.cpp` / `internvl2_ocr.cpp` into the deepseek decode (BOTH the
persistent default and `DS2_LEGACY_DECODE` paths — they must stay comparable),
env-gated with the old behavior restorable. **Acceptance:** decoded text
judged, not cosine — synth 20/20 CER unchanged (no spiral there = guard must
be a no-op), cc0 CER moves materially toward the reference's 0.187 raw /
0.111 stripped, spiral pages terminate before the cap, CPU and Metal, both
decode paths byte-identical to each other per arm. This CHANGES OUTPUT — the
coordinator re-runs the gold gate before merge.

#### F1 status [DONE 2026-08-05, merged `e9f84f16`, coordinator-verified]

**Shipped:** `argmax_no_repeat_ngram` at the single argmax site both decode
arms share, default ngram=20 (the contract's `no_repeat_ngram_size`);
`DS2_NO_REPEAT_NGRAM=0` restores the plain argmax. Confidence is now
stabilised on the global max (bit-identical to the old `1/sum_e` when the
guard does not fire). Helper hoisted to `src/core/no_repeat_ngram.h` and
shared by all three carriers (qwen2vl/internvl2 swap is verbatim code,
compile-checked + unit-tested; no local fixture exists for those two —
their guard is the hermetic test).

**Acceptance (13-sweep matrix, `tests/results/f1/`, decoded text only; box
carried load, no timing claims):**
- **Arm identity (guard on):** Metal 25/25 + CPU synth 5/5 byte-identical.
  CPU cc0 4/5 differ between arms — **pre-existing, proven**: guard-OFF
  baseline arms diverge on the same pages at the SAME first byte (german
  char 67, simple_form char 165); T14's legacy host-side reduction-order
  near-tie mechanism. The guard introduces no arm divergence.
- **No-op where nothing spirals:** synth 25/25 byte-identical guard-vs-base;
  synth CER unchanged (0.00228 raw).
- **Termination:** CPU all 5 cc0 pages terminate (german 1024-cap→228 tok,
  simple_form→90). Metal commons_test_ocr_document 1024→720 tok
  (CER 0.83→0.33). ⚠ **Metal `german_official_print` still caps**: its loop
  varies tokenization ("Aufraktvert ren"/"Aufraktvertre ten") so no exact
  20-gram ever repeats — exact-ngram bans cannot break it. Baseline also
  caps (CER 2.08 vs 2.14 guarded); it is the Metal-vs-CPU trajectory gap
  (CPU reads the same page at 0.59), F5's lane, not a guard regression.
- **cc0 CER vs the A4 reference (0.187 raw / 0.111 stripped):** Metal mean
  0.744→0.657 raw; CPU 0.254 (legacy) / 0.279 (persistent). Post-tokenfix
  note: the T14-era numbers no longer reproduce — `simple_form` no longer
  spirals on Metal even unguarded (52 tok, CER 0.45 vs T14's 2.69), so the
  tokenize_simple fix already moved this lane; the owed post-merge re-gate
  is hereby recorded in these tables.
- **Gold gate:** fox.png `cer=0.000` + garbage-guard PASS on BOTH manifest
  entries (per-expert and stacked), run with the final merged binary.

**Found, not fixed:** (1) the Metal german cap above (F5/crop-mode is the
likely fix — more image tokens, better-conditioned decode); (2) CPU cc0
Metal-vs-CPU quality gap is large on loop-prone pages (CPU 0.25 vs Metal
0.66 mean) — worth a look when F5 lands; (3) the CPU arm near-tie
divergence is inherent to the legacy arm's host-side norm/LM-head and was
accepted with attribution (T14 precedent).

### F2 — Metal pipeline-cache cap adoption across the other Metal lanes (delegable)

T18 capped the MTLBinaryArchive read for the EMBED path only
(`src/core/metal_pipeline_cache_policy.h`, default 64 MB cap via
`CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB`). Every other Metal lane
(`crispasr_init_gpu_backend()` callers: all OCR/VLM/SR engines) still pays
~1 ms/MB of archive size at init. Adoption is one line per lane
(`core_metal_cache::apply()`), but verification is per-lane: init-time
before/after (the `CRISPEMBED_INIT_BENCH=1` instrument exists) + decoded
output unchanged on one fixture per lane. The 683 MB archive at
`~/Library/Caches/ggml-metal/` is still on disk; deleting it is safe and
worth doing once the cap is everywhere. NOTE the cache's deeper problem
(no `_exit()`ing binary can ever WRITE it — the clean_exit class) belongs to
CrispASR PLAN #88, not this repo.

### F3 — Embed-CLI `-t 1` default (coordinator decision, small)

T18's data (its PLAN status block): post-fix one-shot batch-64 with init
included — e5-small: Metal 0.77 s / CPU -t1 0.76 s / CPU -t4 0.35 s;
arctic-m-v2: Metal 0.91 s / CPU -t1 2.77 s / CPU -t4 0.91 s. The backend is
no longer the knob; the shipped `-t 1` is. A blanket small-embedder→CPU rule
is right on one model and wrong on the other. Decide a size/thread rule
(e.g. default -t to min(4, cores) for embed CLI, keep Metal default), A/B it
on both models + one more size class, and only then touch
`CRISPEMBED_ONESHOT_CPU` (ships OFF today).

### F4 — SmolDocling vision backend port (OWN WORK — graph/residency, do not delegate)

Post-T15 stage split (PERFORMANCE.md): vision+connector 31.7 s of 37.3 s on
fox (5 sub-images; ~6.3 s per 512² SigLIP forward, CPU); full pages 72-103 s
with 13 sub-images. The engine hardcodes `ggml_backend_cpu_init()`. Per-tile
SigLIP is compute-bound = GPU-shaped; the 135M per-token decode is the
CPU-favored shape (persistent-decode LEARNINGS) — SPLIT residency, do not
move decode blindly. Also batch the N+1 tiles through one graph if memory
allows. Gates: payload byte-identical CPU vs Metal on the 5-page T15 set
(artifact `tests/results/ocr_parity_smoldocling_2026-08-04.json` has the raw
paired outputs), same-window interleaved timing, `SMOLDOCLING_FORCE_CPU`-style
gate, MTL0-in-stderr verification (the metallib trap is FIXED by the CMake
pin `9288d3b5`, but verify the stderr anyway — that habit caught it).

### F5 — DeepSeek-OCR2 dynamic-crop contract gap (session-sized)

Native feeds a single 1024² view (257 image tokens); the reference uses
dynamic cropping up to 1121 tokens (crop threshold 768 for v2 — A4 status
block). This is the main native-vs-reference cc0 quality gap once F1 lands,
shared identically by both decode arms. Port the reference's crop logic
(read its processor code — blueprint line-by-line), gate it, and re-run the
gold CER gate. Combines naturally with F1 in one session but keep the two
changes SEPARATELY gated (never two variables in one A/B).

### F6 — Quantify DS2_KV_F16 (delegable, small)

Implemented in T14, deliberately unquantified (precision change kept out of
the byte-identity gate). Measure: decoded text vs F32 KV on the 25-fixture
gold (CER delta), memory, and decode time, both backends. Keep opt-in unless
it wins quality-neutral.

### F7 — imatrix QKV coverage fix (delegable, well-scoped)

`src/crispembed.cpp:799-832` pre-merges q/k/v into one F32 tensor at load and
never `ggml_set_name`s it → the imatrix collector files its statistics under
ggml's auto `leaf_N` and the quantizer matches nothing — every BERT-family
`attn.{q,k,v}.weight` quantizes with NO importance (arctic: only 36/73
tensors covered). The collected leaf_N vector (width 768 = QKV input) is
already the correct importance for all three — fix is naming + a quantizer
alias, not new infrastructure. Then re-run the arctic imatrix pipeline
(kernel `tools/kaggle/crispembed-imatrix-t19/`, corpus committed) and expect
q4_k+imatrix to finally separate from plain q4_k (today 0.948/0.961 vs
0.947/0.958 — barely). Continuous metrics, never thresholded-only.

### F8 — LaBSE-class WordPiece audit (delegable, small)

WordPiece vocabs >100k still take the old detection heuristic (deliberate
blast-radius decision in granite-r2). Audit the shipped LaBSE-class GGUF:
token-id parity vs HF on the standard battery; fix via the tokenizer.json
`model.type` path if wrong, with the absent-key=historical-behavior rule.

#### F8 outcome (2026-08-05, `f31c6531`) — audit found the conversion path broken, fixed 3 layers

Nothing LaBSE-class was shipped (no registry entry, no cstr GGUF). Converting
`sentence-transformers/LaBSE` (501k WordPiece) exposed three stacked defects:
converter `is_sentencepiece` >100k heuristic, runtime `n>100000 → SPM`
routing, and the historical per-byte ASCII pre-tokenizer (can never match HF
on CJK/unicode-punct/NBSP). Fixed: converter honours tokenizer.json
`model.type == "WordPiece"` + writes `tokenizer.ggml.pre = "bert"` when
declared; routing hoisted to pure `resolve_tokenizer_family()`
(src/tokenizer.h, explicit numeric type is FINAL; community `model="bert"`
+ >100k corner deliberately frozen); HF-faithful BertPreTokenizer in
`src/core/bert_pretok.h` gated on `pre="bert"` (absent key = historical
byte path, shipped GGUFs byte-identical — verified on 4 models). Hermetic
`tests/test_bert_pretokenize.cpp` in model-free CI. E2E: fixed LaBSE f16 vs
HF f32 CLS = cos 1.000000 (10 texts).

**F8b (open, small):** (a) publishing a LaBSE GGUF is now possible if wanted
(convert + battery + upload + pin). (b) LaBSE's ST `2_Dense` is bit-equal to
the BertModel pooler (tanh); CrispEmbed matches pre-pooler CLS, not the full
ST stack (cos vs pooled ≈ −0.05) — full ST parity needs pooler-tanh at CLS
pooling. (c) `bert.pooler_act` defaults to `"gelu"` where BERT's pooler is
tanh (currently rerank-only, harmless). (d) Three upload-bearing kernels
without a vendored harness (`unlimited-ocr-convert`, `crispembed-splade-fix`,
`deepseek-ocr2-convert`) bootstrap `kh` from the CrispASR clone so they get
F9's resolver but not fail-fast — flipping them to `require=True` is cheap.

### F9 — Kaggle harness token-glob fix (CrispASR repo, delegable)

`resolve_hf_token()` misses the LONG dataset mount path
(`/kaggle/input/datasets/<acct>/<slug>/`) — a kernel on such a worker
completes and then loses every upload to 401 (cost one full 21-min imatrix
run). The t19 kernel carries the local fix; hoist it into CrispASR's
`kaggle_harness.py` so every future kernel gets it. Also carried there:
kaggle_usage.md gotcha #26 (script kernels ship only code_file — vendor
data in the repo clone).

### F7b — re-collect + re-quantize the published BERT-family imatrix artifacts (post-F7, Kaggle)

F7 (`68033e8d`) fixed the coverage defect, so **every published BERT-family
`.imatrix` on HF (e5, arctic, bge, …) still carries the `leaf_N` defect and
every published `*-q4_k-imatrix.gguf` was built with no q/k/v importance.**
Re-run the t19 pipeline (`tools/kaggle/crispembed-imatrix-t19/`, corpus
committed) with an F7-fixed binary; expect q4_k+imatrix to separate (local
e5-small evidence: cos_min 0.9847→0.9889, mean 0.9889→0.9913). The e5-small
f32 + shipped imatrix are cached under `~/.cache/hf-f7` for this. One kernel
at a time; promotion decisions stay with the coordinator (IQ4_XS note in
T19-E3 applies).

#### F7b outcome (2026-08-05, kernel v3) — coordinator decision items

Numbers in the wave-3 row above; full A/B in
`cstr/*-GGUF/*-f7-imatrix-ab.txt` (and granite's canonical-name ab files).
Left OPEN deliberately:
1. **Re-point the registry's arctic q4_k-imatrix alias at the `-f7` artifact?**
   The shipped pinned `arctic-embed-m-v2-q4_k-imatrix.gguf` (301cae98…) was
   built with NO q/k/v importance and measures far below the `-f7` re-quant
   (.9614 vs .9937 mean). The A/B ran on Kaggle x86 CPU only — do the
   local-Metal cross-check first (T19-E3 saw ~0.002 backend FP delta), then
   re-pin. q8_0 stays the default regardless (q4_k+imat .9937 < q8 .9996).
2. **IQ4_XS guidance narrowed:** T19-E3's "IQ4_XS+imatrix is the best sub-Q8"
   held only under the coverage defect for BERT-family models; post-F7 arctic
   q4_k+imatrix wins all three tails. Decoder-family (f2llm) keeps the IQ4_XS
   ordering. Re-measure per family; never generalise across the pre-merge
   boundary.
3. granite-311m iq4_xs quantizes some `ffn.fc2` tensors as iq4_nl fallback
   (dimension constraint) — benign, note when reading its size numbers.

### F9b — CrispEmbed's vendored kaggle_harness.py copies are stale (the ACTUAL t19 culprit)

F9's verification corrected the brief: CrispASR's canonical harness has
globbed both mount depths since `81826457` (2026-06-20); what lost the t19
uploads is the stale vendored copy in
`tools/kaggle/crispembed-imatrix-quant/` (hard-coded owner + name-filtered
scan; several other `tools/kaggle/*/kaggle_harness.py` copies exist with
~300-line drift vs canonical). CrispEmbed kernels clone CrispEmbed, so the
canonical fix never reaches them. **Do:** re-sync each vendored copy from
CrispASR canonical (now also carrying F9's `resolve_hf_token(require=True)`
fail-fast — uploading kernels should call it first), checking each kernel
dir for deliberate local drift before overwriting. Sync logic, not bytes,
where a copy has real local changes (pcs.cpp rule).

### F10 — CrispASR twins (other repo, coordinate before touching)

(a) CrispASR's copy of `gpu_backend_pref.h` has the same `--gpu-backend cpu`
falls-through-to-Metal bug T18 fixed here (the copies had already diverged;
sync LOGIC not bytes, per the pcs.cpp rule). (b) The Metal pipeline-cache
write path is broken for every `_exit()`ing binary (PLAN #88's call:
flush-per-run, scope per engine, or retire).

### T16 (TableFormer port) and T17 (Fraktur bisect) — unchanged, dedicated sessions

Their briefs stand in the OPEN TASKS section below. T16 needs the A5
document-structure gold (also still open). T17's reference-first bisect notes
are in the Tesseract sections.

### Discipline deltas learned this round (additive to the standing rules)

- **Worktrees: use a real ggml submodule checkout** (`git submodule update
  --init ggml`), NOT the symlink dance — it builds AND rebases; a leftover
  symlink makes every rebase fail with "expected submodule path not to be a
  symbolic link" (fix: rm + mkdir + submodule update).
- **NEVER `git add -A` in a worktree** — it stages the ggml gitlink and
  reverts main's pin (this exact mistake shipped and had to be repaired at
  `78b7137e`). Stage files by name.
- **Metal claims need MTL0 in the run's own stderr** — the CMakeCache proves
  nothing in either direction. The stale-EMBED_LIBRARY trap is fixed by the
  CMake pin, but the cheap tripwire that caught it (watch a stage neither A/B
  arm touches) stays mandatory for timing work.
- **No window-encoded regexes in ugrep** (`[^x]{0,120}(alt)[^x]{0,160}`
  builds a diverging DFA, 5.5 GB RSS, machine-killing); use `-C 2` for
  context.
- **Timing on this box:** ambient user load is real; interleaved pairs with a
  loadavg gate (discard >8), median + spread reported un-trimmed, and the
  floor-vs-excursion argument (a load excursion can only inflate) is how a
  noisy verdict gets decided.
- **Weights storage:** big GGUFs live on the backup SSD under
  `ai/crispembed-ggufs/` with symlinks back into `~/.cache/crispembed-local/`
  (the CLI cache dir); keep the main volume ≥10 GB free; delete f16s once
  quants are verified; one heavy (>1 GB) process at a time.

### Environment as left (2026-08-05)

- Main volume ~42 GB free; backup SSD ~3 GB free (olmocr q4_k moved there).
- `~/.cache/crispembed-local/`: e5-small q8, arctic-m-v2 q8 + iq4_xs,
  f2llm f16s+q8s, deepseek-ocr2 q4_k (2.2 GB), smoldocling q8 (FIXED vocab),
  granite-r2 pair, olmocr q4_k (symlink to SSD). All registry-pinned.
- HF: account cstr, token in `../.env` (HF_TOKEN); ALWAYS `HF_HOME=~/.cache/hf-<task>`
  (default cache symlinks to the full backup volume; uploads also need it or
  they die read-only). Kaggle: chr1s4 works, one kernel at a time,
  machine_shape "NvidiaTeslaT4", stage under /tmp, delete before re-push.
- All 2026-08-04/05 worktrees and branches removed; only the three
  pre-existing IN PROGRESS rows above remain claimed by other sessions.

### Next actions — scoped for a fresh session

Read `../crispasr-crispembed-dev.md` first. The three HARD RULES that actually
bit this codebase most recently: **#2b** (cosine is scale-blind — always read
`|mine|`/`|ref|`), **#3** (decoded output is the only acceptance test), **#8**
(never report green off a pipeline's exit code).

**Board discipline.** The table above is for work claimed *right now*. Add a row
before starting, update it at each checkpoint, move it to `HISTORY.md` when it
lands. It reached 56 rows — 52 of them finished — before this cleanup, which is
exactly how a fresh session re-derives something already shipped.

#### Owned and in flight — coordinate before touching
| # | Item | Owner |
|---|---|---|
| N1 | ~~PP-OCRv6 fused batch graph on Metal~~ **DONE 2026-08-04** (`perf/ppocrv6-speed`, merged, DEFAULT): the "Metal fourth-dimension pooling" abort was a zeroed in-out width parameter — every batch graph was built at width 0 and asserted on ANY backend. Width seeded + large-stem small/medium extended (per-item 3D/4D through tokenization, SVTR attention, and an in-graph skip+head finish to per-item logits). Promotion evidence: 26/26 fixtures byte-identical to scalar-per-crop, parity harness PASS tiny+small on CPU and Metal, receipt recognize **3743→2563 ms** / engine **4067→2885 ms** (6 fused Metal groups) — ahead of official paddle-3.7's 5.9 s, 1.3x from the onnxruntime 2.2 s ceiling. `BATCH_MAX` default 8; `CRISPEMBED_PPOCRV6_BATCH_GRAPH=0`/`NO_BATCH_GRAPH` disables | **CLOSED** |
| N2 | EasyOCR full-page quality. Per-stage diff on identical crops already **passes** (input 0.99981, recurrent/logits ≥0.99972) ⇒ the gap is detector geometry / crop selection / postprocess confidence. **Do not re-open the LSTM.** | `feat/easyocr-ggml` |
| N3 | OCR perf H-items — **now UNOWNED**: `perf/ocr-h-items` landed and archived its row while this cleanup was in progress. Remaining: H2 detector scalar path, H4 batched crops, H5 tesseract load, H6 resize-by-text-height. H1/H3/H7 done. Brief + measurement rules in §"OCR performance — self-contained handover prompts" | *(free to pick up)* |
| — | OCR external head-to-head (CER/WER + latency vs system Tesseract / EasyOCR / PaddleOCR) | `feat/ocr-engine-parity` |

#### Unowned and ready to pick up

**N4 — publish q8_0 for esrgan and scunet.** Both were *unrunnable when
quantized* until `06bc5d7a` (esrgan graph-node budget; scunet flattened kernel in
the persistent cache). Fixes verified — esrgan q8_0 cos 0.999998 / 51.9 dB,
scunet q8_0 0.999999 / 60.5 dB vs f32, and f32 output byte-identical before and
after — but **no quantized artifact was ever published**, so the registry still
ships f32 only. Quantize → verify decoded output → upload → pin.
⚠ Do **not** ship an esrgan q4_k: 29.55 dB / max_abs 91 against q8_0's 51.89 dB.

**N7 — extend the quantize-and-run sweep to the OCR/VL engines.**
`crispembed-quantize` output is far less exercised than f32/f16 and the failures
cluster there: of five SR/denoise engines swept, **two had never been run
quantized at all and both aborted**. Not yet done for OCR/VL. Method: quantize a
local artifact, run its CLI, diff the decoded output against the f32/f16 run.

#### Deliberately deferred — recorded so they are not re-derived as questions

**N5 — re-convert published h2ovl artifacts to carry `internvl2.chat_template`.**
The runtime infers the `h2ogpt2` template from the vocab (no `<|im_start|>`,
`<|end|>` present) because the published f16/q8_0 predate the key. Verified on
both checkpoints, so this is hygiene, not a defect, against a ~6.7 GB re-upload
plus re-pin.

**N6 — `llm_output_norm` is h2ovl-2b's weakest q8_0 stage** (`cos_glob` 0.997297,
the only stage >1 % off in magnitude: 323.44 vs 331.49). **Exact at f16**, so it
is quantization not the norm implementation, and the logits recover to 0.998919
right after. First place to look if this model misbehaves again.

#### Standing traps, all re-confirmed this session
- **`cos_min` is a per-row minimum.** Right gate for port correctness (h2ovl-2b
  f16 = 1.000000 on all 54 stages incl. logits); wrong gate for a quantized
  artifact (same model q8_0 = 0.61 on the logits while `cos_glob` is 0.998919 and
  it transcribes a page). Use `CRISPEMBED_DIFF_COS_THRESHOLD` / `is_pass_global()`
  for quant sweeps, and always read `|mine|`/`|ref|`.
- **Dump references WITHOUT `--max-llm-layers`.** The cap also silently drops
  `llm_output_norm` and `llm_logits`, leaving the harness short of the decision
  boundary. Both h2ovl references are now full (54 / 46 stages) in
  `cstr/crispembed-regression-fixtures`.
- **A rule measured on one checkpoint is not a rule for the family.** Narrowed
  twice this session: refusing sub-Q8 for all `internvl2` broke `internvl2-1b`
  and `h2ovl-800m` (both fine at q4_k); vision→F16 for all `internvl2` inflated
  the edge model 758 → 1135 MB. Re-measure a sibling before generalising.
- **Timeouts truncate files.** A 10-min tool timeout killed a chained
  `cp && upload` mid-copy, leaving a truncated GGUF that still looked plausible.
  Check size *and* digest before trusting a copy; upload straight from the
  source via `path_in_repo`.
- **`.env` breaks `git push`.** Sourcing it injects a token that overrides the
  credential helper — `env -u GH_TOKEN -u GITHUB_TOKEN git push`.
- **A shipped binary must be checked as a PACKAGE, not as a build.**
  SubtitleEdit#13205: every Linux tarball through v0.17.0 was unlaunchable —
  `libggml-blas.so.0` hard-needed `libopenblas.so.0`, which the archive never
  carried, so `crispembed-server` died in the loader with exit 127 and no
  output. The workflow apt-installed `libopenblas-dev` so the BLAS probe would
  pass, which means the runner always had the library and the artifact never
  did: **the failure was structurally unreachable from CI**. And BLAS was
  measured at 0.9–1.0x here (`PERFORMANCE.md`), so it was costing a platform
  for nothing. Linux release legs are now `-DGGML_BLAS=OFF`, the apt install is
  gone (a re-enabled BLAS now fails loudly at configure), and
  `scripts/check-bundled-deps.py` fails packaging on any `DT_NEEDED` that is
  neither bundled nor base-system. `release.yml` also takes
  `workflow_dispatch` now (publish steps guarded on `refs/tags/`) so an
  artifact can be produced and inspected without cutting a tag. Still open:
  the glibc 2.38 / GLIBCXX 3.4.32 floor from building on Ubuntu 24.04 — no
  Ubuntu 22.04 or Debian 12 — which needs a manylinux-container build, same as
  the wheels (`python/pyproject.toml`).
- **`GGML_NATIVE` probes the BUILD machine — never ship a native build.** It
  defaults ON and *runs* code on the builder: an AVX-512 `check_c_source_runs`
  probe on MSVC, `-march=native` on GCC/Clang, `-mcpu=native` plus
  dotprod/i8mm/sve/sme run-probes on ARM. So the artifact's ISA is a property of
  whichever runner took the job, and GitHub's pools are heterogeneous. #41:
  v0.16.1's Windows cpu zip shipped `/arch:AVX512` and died with `Illegal
  instruction` on a Raptor Lake i9 — CI cannot reproduce it, because no runner
  lacks the extension the runner had. Every redistributable leg now passes
  `-DGGML_NATIVE=OFF` and runs `scripts/check-cpu-baseline.py build`.
  Second trap inside the first: with NATIVE on, `FindSIMD.cmake` sets
  `GGML_AVX512` as a *normal* variable that shadows the cache entry, so
  `CMakeCache.txt` can read `OFF` while the compile line says `/arch:AVX512` —
  check the generated build files, not the cache. `CRISPEMBED_NATIVE` follows
  `GGML_NATIVE` so one flag covers the tree. Full write-up in `LEARNINGS.md`.


EasyOCR cross-check benchmark checkpoint (10 repeated recognitions, identical image/
width; native Metal versus Miniconda PyTorch CPU reference): Latin Gen2 formula
200 `16.523/12.460 ms`, scan 128 `10.885/7.137 ms`, Latin Gen1 scan 128
`154.082/78.648 ms`, English Gen2 scan 200 `16.536/10.035 ms`, and scan 128
`10.697/7.287 ms` (native/reference totals). Outputs match in every case:
`x=0442`, `82`, `==#`, `032`, and `@32`; the English width-128 strict
row-wise logits gate is still open despite decoded parity. Native is slower in
all measured graph/total paths, so graph/kernel and width optimization are
performance TODOs. CRAFT, DBNet page modes, and Tesseract still need equivalent
timing/output manifests; their existing parity checks are not performance
acceptance evidence.

Tesseract confidence checkpoint (2026-08-01): after rebuilding
`test-confidence`, the explicit `/opt/homebrew/share/tessdata` Fraktur PSM 7
comparison on `scan_strip.png` produced official `iE` (mean word confidence
`0.043433`), native greedy `BEEES` (word confidence `0.884625`), and native
beam-8 `BEEES` (sequence confidence `0.644788`, no fabricated character
confidences). Official-word validity passed, but decoded text and confidence
calibration failed; recoder/DAWG and page/line confidence aggregation remain
quality TODOs. Timings were official `5881 ms`, native greedy `305 ms`, and
beam `984 ms`; these whole-process values are diagnostic only.

**PP-OCRv6 checkpoint/line-crop follow-up (2026-08-02).** The official-source
reference and native f16 agree on the decoded garbage for the CC0 line/page
fixtures: Arabic printed line `¿づE₆¿づLyi` and receipt `上批业/|` (the German
document is likewise not a valid line-crop quality gate). The 18,710-entry CTC
vocabulary is present and ends with the expected space token, so this is not a
missing-vocabulary load failure. Keep quality blocked until the exact official
checkpoint provenance and a known-good PP-OCRv6 line-crop sample are verified;
do not improve CER by silently switching to a different model family.

**PP-OCRv6 official preprocessing A/B (2026-08-02).** Audited the official
PaddleX/HF `preprocessor_config.json` and PaddleOCR `inference.yml` against the
native path. Both use 48-pixel height, pixel-center bilinear resize, right
padding to width 320, and `(pixel/255 - 0.5) / 0.5`; PaddleOCR's shipped
inference contract is BGR while the HF processor advertises RGB conversion.
Fresh official-source runs on `fox.png` decoded `上ai` in both channel modes;
the native f16 recognizer also decoded `上ai`. On the German uneven-illumination
fixture, the official source decoded `澳臻肉企NM` (BGR) and `澳臻肉門企NM`
(RGB), while native BGR decoded `澳臻肉企NM`; this confirms the runtime is
aligned to the BGR official inference path, not suffering a hidden channel or
normalization mismatch. Added `CRISPEMBED_PPOCRV6_RGB=1` as a diagnostic-only
native A/B switch; production remains BGR. The actual text is still nonsense
in both native and official outputs, so this is a checkpoint/model-quality
problem, not a runtime-quality fix. **IN PROGRESS:** validate checkpoint
provenance/vocabulary and line-crop suitability before any model promotion.

Tesseract decoder groundwork (2026-08-01): the converter now preserves every
present DAWG component from `.traineddata` as base64 GGUF metadata with a
component SHA-256 digest. This supplies the missing serialized source for a
future native dictionary scorer; it does not claim DAWG traversal, recoder
beam, or output parity, and existing GGUFs must be regenerated before they can
use the metadata.

The native loader now reads the DAWG manifest, validates that every listed
component has a nonempty payload, and reports the loaded count in its model
diagnostic. Older GGUFs remain compatible with zero DAWG entries. This closes
metadata integrity only; DAWG traversal/scoring and decoded-output parity are
still open.

The loader now also structurally validates each base64 SquishedDawg payload:
magic/header, dimensions, edge-array bounds, next-node bounds, and forward-edge
run termination. `test-tesseract-dawg` passes and the regenerated English
smoke model loads with `dawg=3`; no dictionary score is applied and no
production beam behavior changed.

The DAWG diagnostic layer now also supports exact-word membership lookup over
unichar-ID sequences. The minimal fixture covers a positive and negative
lookup; this is traversal infrastructure only, with no score or production
decoder behavior change.

DAWG metadata decoding is now strict about quartet length, padding placement,
and unused padding bits, with malformed-input coverage in
`test-tesseract-dawg`; corrupt payloads are rejected before traversal.

The DAWG layer now exposes a reusable parsed context that caches the decoded
edge array and bit masks for repeated exact/prefix queries. This is preparation
for beam integration only; production OCR still does not consult dictionary
state.

The native Tesseract API now exposes exact-word and legal-prefix queries against
the model-owned cached DAWG contexts by component name. Missing components and
null contexts fail closed; production decoding still does not invoke them.

The API also provides a tri-state result: invalid prefix, legal non-terminal
prefix, or complete word. This is a scorer-facing contract only; no dictionary
state changes OCR output yet.

An opt-in `CRISPEMBED_TESSERACT_DAWG_PREFIX` filter now applies cached system
DAWG prefix legality inside the recoder beam after a prefix fully composes to
unichar IDs. Incomplete recoder codes remain open; default decoding is
unchanged and official parity is still pending.

The same filter A/B on the English smoke model passed 37/37 in both modes,
decoded `Se` in both, and produced sequence confidence `0.562293` in both.
This confirms safe operation only; dictionary quality parity remains open.

Native Tesseract model loading now constructs one cached parsed DAWG context per
manifest component and frees them with the recognizer. The cached dictionaries
remain diagnostic-only; production OCR does not apply their state.

The diagnostic DAWG layer now distinguishes exact-word membership from legal
prefix membership. Its fixture covers `1` as a legal non-terminal prefix and
`1,2` as the complete word; neither path is used by production OCR yet.

CRAFT's old folded-F16 diff printed error statistics: the earliest divergent
stage was `basenet_0` (`max_abs=1.52823`, RMS `0.195515`, global cosine
`0.995623`), which propagated to score-map `max_abs=0.06910`, RMS `0.008026`,
global cosine `0.999716` and changed the threshold-sensitive decoded box count
from Python's 106 to native's 107. Re-converting with runtime BN (raw
convolution weights plus explicit BN scale/shift) makes F32 match to
floating-point noise; runtime-BN F16 reaches score-map global cosine
`0.9999999` and 106 boxes. The CPU-forced and Metal outputs are byte-identical.
The folded artifact is stale; threshold tuning is not the fix.

Detector benchmark audit: the fresh CRAFT reference for `scan_strip.png` uses a
288x544 canvas and decodes 106 boxes; runtime-BN F32/F16 native runs decode 106,
so CRAFT box parity passes. Native diff runtime was ~2.34 s;
the Python dump was ~9.13 s including model load and serializing 84 tensors, not
an inference-only comparison. DBNet native page smoke measured 6.63 s in line
mode (12 units, 1.34 s summed recognizer work) and 6.67 s in word mode (98
units, 2.50 s summed recognizer work). The restored official DBNet checkpoint
now has tensor parity evidence below; native's 98-word segmentation remains
not comparable to Tesseract's 106-word segmentation. These are explicit
quality/performance TODOs.

Fresh DBNet reference checkpoint parity is now available: the official MMOCR
IC15 ResNet-18 checkpoint was restored and dumped with Miniconda Python on the
same 736x1472 preprocessed `scan_strip.png`. Native F16 passes the final
probability-map boundary (`max_abs=0.00154233`, RMS `0.00008044`, cosine
`0.9999974`, global `1.0000000`) and decodes 96 regions. Q4_K decodes the same
96 regions but fails tensor parity (`cosine=0.9311001`, global `0.9986384`),
so its prior README parity claim is stale for this reference. The DBNet diff
harness now retains every backbone, lateral, smooth, fused, head, and final
probability-map tap; F16 passes all of them. Q4_K's earliest divergence is
already at
`backbone_stage_0` (global cosine `0.9960006`, RMS `0.07697`), and it worsens
through the neck to final-map cosine `0.9311001`; this is a quantization
quality TODO, not a postprocessing issue. The detector now uses a shape-keyed
persistent GGML graph, with diagnostic tap retention enabled only by
`OCR_DETECT_CAPTURE_TAPS=1`. The fresh native CPU-forced page benchmark reports
detector graph `4178.6 ms`, postprocess `8.3 ms`, total `4186.9 ms`, and 12 line
units. The Miniconda reference uses 4 compute threads and 8 interop threads.
Corrected rapid-mode repeated benchmarking gives native CPU `5661.1 ms` warm
with 4 threads (`4.67x` slower), `2907.2 ms` warm with 8 threads (`2.40x`),
versus `1213.450 ms` reference. The same Python blueprint on MPS averages
`577.342 ms`; native Metal `3499.4 ms` is therefore `6.06x` slower on the
same device. Both native backends pass all taps in diff mode and retain
readable output; native CPU/Metal convolution and deconvolution kernels remain
mandatory optimization TODOs.

An opt-in `OCR_DETECT_DIRECT_CONV=1` experiment was investigated against the
GGML direct-convolution op. CPU requires F32 kernels, and the resulting direct
graph did not finish a diff run within roughly two minutes; it remains
disabled and is not parity/performance evidence. The default persistent
im2col path is unchanged. A vectorized direct kernel is a future performance
item. A later 8-thread baseline run was resource-contended (44.1 s cold,
66.7 s warm) and is excluded from the stable benchmark ratios.
An initial per-tap prefix-graph profiler was discarded because shared arena
allocation changed the restored output to zero boxes. No stage timing from that
experiment is evidence; profiling needs an isolated allocator before use.

CRAFT repeated inference benchmark: after one warm-up, 10 runs on the captured
288x544 `scan_strip.png` input averaged `396.027 ms` for Miniconda PyTorch CPU
and `850.018 ms` for native runtime-BN F16 Metal graph compute, with 106 boxes
from both (`2.15x` native/reference directional slowdown). CRAFT quality is
on par on this fixture; its graph/kernel path remains a performance TODO.

The old folded-F16 CRAFT taps showed error accumulation through the VGG. The
runtime-BN conversion removes that divergence: F32 captured taps match to
floating-point noise, and F16 remains within the accepted global gate. The
CPU-forced and Metal runtime-BN runs are byte-identical, including taps, score
map, and 106-box decoded result. Remaining CRAFT work is repeated inference
benchmarking, not postprocessing threshold tuning.
| 2026-07-31 | `main` | External document-parser-informed OCR pipeline: structured routing, in-memory handoffs, service contracts, batching, and benchmark gates | **IN PROGRESS** |
| 2026-07-31 | `main` | Real-world public-domain OCR corpus and manifest-driven multi-engine live benchmarks | **IN PROGRESS** |
| 2026-08-01 | `feat/tesseract-fraktur` / `CrispEmbed-tesseract-fraktur` worktree | **Picked:** validate Tesseract beam/sequence confidence against official line/page outputs; improve gated blob→row segmentation while preserving DBNet as default; optimize the recognizer precision frontier with reproducible mixed-precision GGUF candidates | **IN PROGRESS** |
| 2026-08-02 | `feat/tesseract-kernel-opt` / `.codex/worktrees/feat-tesseract-kernel-opt` | **Picked:** optimize the cached Tesseract int-mode LSTM kernel and immutable-weight reuse; preserve the exact seeded-output contract, benchmark warm recognition against official/native baselines, and keep the precision fallback gated until parity holds | **COMPLETED** |
| 2026-08-02 | `feat/tesseract-kernel-opt` / `.codex/worktrees/feat-tesseract-kernel-opt` | **Picked:** reuse per-LSTM temporary vectors across sequential line recognitions; retain isolated per-context ownership, exact cached/uncached output parity, and the existing diagnostic gates | **COMPLETED** |
| 2026-08-02 | `feat/tesseract-kernel-opt` / `.codex/worktrees/feat-tesseract-kernel-opt` | **Picked:** reproduce the AdaIR F16 ggml buffer assertion from the registry audit, repair only the F16 backend path if the root cause is local, and retain F32/scalar fallback coverage. Added allocation guards (no backend assert) and fixed `DequantCache` identity for backend-resident tensors. F32 remains cos 0.999382 / 2.65 s; scalar F32 remains cos 0.999379. F16 and a fresh F32→F16 rebuild now exit cleanly but remain cos 0.729509 / max_abs 0.707725, so F16 is not shippable. Diagnostics showed the CPU buffer allocator reports an allocation size of zero for the F16 kernel descriptor; explicitly selecting or manually allocating the CPU buffer did not change the result and was reverted. F16 metadata also flattens representative 4-D weights (`[3,3,3,48]` → `[27,48]`, `[1,1,48,144]` → `[48,144]`) while mixing F32/F16 tensors. ~~**TODO:** fix or reject this degenerate F16 descriptor/loader path, then rerun tensor-level and end-to-end parity before any upload.~~ **Closed 2026-08-02 on `feat/ocr-followups`** — the "allocation size of zero" was a downstream symptom, not an allocator fault. The flattened metadata this row already noted *is* the cause: `src/adair.cpp` read three hidden widths off `ne[3]`, which is `1` once flattened, collapsing the GDFN width to 1 and building the next conv with `ic=0`. f16 is now cos `0.999383` and the rebuild is identical, so neither artifact was degenerate. | **COMPLETED (superseded by `feat/ocr-followups`)** |

Mixed-precision checkpoint: the old Q8 artifact lacked `sample_iteration`.
Fresh F32 conversion reaches 9/9 stages with logits cosine `0.993819`; a
seed-preserving mixed Q8/F32 candidate reaches 9/9 and `0.994876`. Both decoded
outputs still differ from Python, so mixed precision is not promoted. The
blueprint now explicitly models native row-wise int8 FC arithmetic.

Backup audit found 46 Tesseract model GGUFs, but 45 lack
`tesseract_lstm.sample_iteration`; only the explicitly named Homebrew English
artifact has the seed. Missing-seed language and quantized variants require
regeneration or verified metadata reconstruction before parity acceptance.
Fresh F32 Fraktur conversion now passes all 9 stages exactly (logits max error
`2.09e-7`) and decoded text matches Python after the blueprint adopted int8
LSTM row arithmetic. The seed-preserving mixed Q8/F32 candidate still fails at
logits cosine `0.989655` and decoded output, so quantization—not preprocessing
or graph topology—is the remaining quality blocker.
| 2026-07-31 | `feat/ppocr-next-20260731` | O10.1 live preprocessor benchmark harness: raw/cleanup/binarize outcome rows on CC0/German fixtures | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O9/O10 reproducible PP-OCRv6 tiny/small/medium benchmark JSON wrapper for the 10-fixture detector/orientation/recognizer sweep; tiny/small live sweeps validated, medium first fixture passes in 125.34 s (full sweep still exceeds the 900 s guard and remains pending). Fresh routed fox benchmark after engine-name fix: CPU tiny `4.98 s`, small `20.82 s`; Metal tiny `24.52 s`, small `34.28 s`; medium timed out at `120.6 s` on both routes because the medium detector remains CPU-only. | **IN PROGRESS** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** measure repeated warm p50/p95 timings for runnable PP-OCRv6 tiny/small lanes and preserve cold-vs-warm evidence in the O9 survey; medium remains excluded from the warm lane until its CPU detector path is optimized. Metal repeated fox runs: tiny cold `49.14 s`, warm p50 `42.91 s`; small cold `70.66 s`, warm p50 `84.87 s` (the harness now persists per-run timings and inclusive p95; the small run’s p95 is not claimed from the pre-field artifact) | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** audit the PP-OCRv6 recognizer quality gate after the external head-to-head invalidated cosine-only parity. Local PaddleX exposes PP-OCRv6 detector code but no recognizer blueprint; the current torch mirror still guesses head count/activation/pooling and reproduces unreadable text. Keep the lane unvalidated and block default promotion until the official recognizer implementation/config is recovered and text-gold parity passes | **COMPLETED — quality gate remains blocked** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** align PP-OCRv6 large-recognizer activation semantics with the recovered PaddleX source: StemBlock Conv-BN uses ReLU; LightSVTR conv branches use configured SiLU. Updated native CPU/graph and `dump_ppocrv6_reference.py`; small fox taps pass through stage4 (`0.999958`), head input (`0.999992`), logits (`0.999996`), and full graph logits (`0.999995`). Regenerated official-source-backed Arabic/receipt/German gold archives; the required small graph lane passes `0.999992–0.999997`, while native and reference still decode nonsensical `¿づE₆¿づLyi`, `上批业/|`, and `澳臻肉し企M`. | **COMPLETED — parity fixed, quality blocked** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O11 backend/graph capability audit: record CPU-only, partial-graph, and full-GGML-backend paths per OCR engine and prevent unsupported GPU claims. The capability matrix covers all 12 required OCR families, concrete CPU seams, PP-OCRv6/PP-LCNet partial claims, and explicit Metal/CUDA build boundaries; `tests/test_ocr_backend_matrix.py` is a mandatory smoke guard. CUDA execution and per-engine performance remain separate follow-ups | **COMPLETED** |
| 2026-07-31 | `main` | O11.1 PP-OCRv6 detector/recognizer graph port: replace CPU conv/linear forward with persistent ggml graphs on CPU/Metal/CUDA; preserve Q8 head policy and parity taps | **PENDING** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** O11.1 full-graph implementation contract: persistent static-shape graphs, scheduler-selected backend, backend-resident/dequantized weights, reusable input/output staging, batched line crops, and CPU cosine/logit parity fallback; PP-OCRv6 tiny recognizer now runs one persistent graph through logits with CPU/Metal accepted-output parity on two fixtures, while the detector constructs an opt-in full stem/backbone/neck/head graph via `CRISPEMBED_PPOCRV6_DET_GRAPH=1`; corrected detector neck channel order brings graph-vs-CPU probability cosine to 0.99113 and head pre-sigmoid to 0.99898, but graph box count is still 31 vs CPU 30, so detector CPU accept-gate fallback remains. The regenerated small/medium fox references are valid (the prior small archive had `large_stem2a` length `87,768`); CPU logits parity is `0.999998` small and `0.999992` medium. The small and medium recognizers now build a persistent GGML stem+backbone graph on CPU: all six large-stem taps are `1.000000`, stage taps are `1.000000`/`0.999994` small and `1.000000`/`0.999983` medium, and the CPU SVTR decoder receives the graph backbone with end-to-end logits unchanged (`0.999998`/`0.999992`). The asymmetric stride transition is now supported. Metal reaches stage4 cosine `0.999907` small / `0.999969` medium and logits cosine `0.999982` / `0.999986`, decoding `涨RiI` in both cases. The opt-in `CRISPEMBED_PPOCRV6_SVTR_GRAPH=1` seam graphs SVTR tokenization, and `CRISPEMBED_PPOCRV6_SVTR_DECODER_GRAPH=1` now graphs both SVTR attention/MLP blocks plus final norm; CPU and Metal small/medium runs preserve output parity, with Metal logits cosine `0.999982`/`0.999986` and full graph timing `214`/`512 ms` on the fox crop. Multi-fixture direct recognizer smoke now shows identical CPU fallback/CPU full-graph text on Arabic line, receipt, and German document fixtures, and Metal full-graph text matches CPU on all three. Regenerated gold archives for those fixtures are backed up under `/Volumes/backups/ai/crispembed-gguf/`; the new `tests/test_ppocrv6_graph_gold.py --require` lane passes CPU at `0.999995–0.999996` and Metal at `0.999956–0.999982` (lowest on German), with unchanged decoded text. Remaining work is detector geometry parity and wiring the artifact-backed lane into model-equipped CI; keep both graph gates opt-in until that lane is reproducible there | **IN PROGRESS** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O11.2 PP-LCNet line/page orientation graph port: backend-scheduled depthwise/pointwise/SE blocks with CPU fallback and orientation gates; canonical weight layout and `[1280,2]` linear view are implemented. Metal passes 9/10 German/Arabic/derived parity fixtures; the one uneven-illumination Arabic outlier localizes to Metal accumulation around SE block 4 (`1.0679/3.2166` final logit deltas), while CPU graph passes (`0.0046/0.0139`). Repeated standalone Metal execution passes 31/31 with required per-crop scheduler reallocation, and the explicit pipeline graph smoke remains `141/141` with 30/30 regions. The safe production behavior is complete: graph stays opt-in and automatically falls back to CPU unless explicit debug acceptance is requested. Metal SE/depthwise numerical parity remains a separate optimization TODO | **COMPLETED — safe fallback shipped** |
| 2026-07-31 | `main` | O11.3 GPU preprocessing handoff: benchmark and, where beneficial, graph-accelerate detector resize/normalize, quad warp, crop batching, and postprocessing without changing geometry | **COMPLETED — retain CPU geometry path** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** O11.3 preprocessing/geometry cost split: `CRISPEMBED_PPOCRV6_DET_BENCH=1` reports detector normalize/graph/total timings and `CRISPEMBED_PPOCRV6_BENCH=1` reports routed detector, quad crop, orientation, and recognizer timings. German CC0 Metal measured detector 6.9 s, crop 3.4 ms, CPU orientation 358.6 ms, recognition 455.2 ms; explicit Metal orientation graph is safe but 1.15 s for 30 crops. Crop/warp is not the bottleneck and GPU orientation is slower, so no geometry graph promotion is justified; retain CPU preprocessing while preserving the opt-in graph for diagnostics | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O11.4 OCR portfolio graph audit: capability matrix schema now classifies PP-OCRv6 and PP-LCNet as partial rather than overstating GPU support; tiny/small/medium tier details and acceptance gates are recorded. Source audit confirms FormulaNet, MixTeX, SmolDocling, HMER/BTTR/PosFormer are CPU-scheduled today; VLM audit records concrete CPU seams (window partition, spatial merge, host-side unshuffle/merge, scalar merger, and opt-in neck/MoE fallbacks). PP-OCRv6’s gated full recognizer graphs are now reflected accurately; detector geometry and per-engine residency/performance remain separate follow-ups | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O11.5 backend build/device matrix: Metal macOS, CUDA Linux, CPU reference; graph smoke accepts an explicit backend, reports requested/selected device, and passes on Apple M1 with `GGML_METAL=ON`; CUDA remains pending. Metal PP-OCRv6 orchestrator smoke now passes 141/141; recognizer graph and detector diagnostic graph execute on MTL0 | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O11.6 graph performance/parity gates: backend smoke records warm compute latency after a successful graph run; per-line recognizer timing and non-CPU F16 resident detector weights are measured. Metal detector diagnostic improved from ~3.6 s to ~3.3 s on German CC0 with unchanged probability cosine `0.99114`, so it remains CPU-accepted. Small/medium stem+backbone timing is CPU `10,377`/`10,791 ms` versus Metal `222`/`650 ms` on the fox crop; the full SVTR decoder graph now measures `214`/`512 ms` on Metal and multi-fixture gold logits remain `0.999956–0.999986` with decoded parity. Recognizer performance/parity gates are therefore complete; detector geometry acceptance and CUDA remain separate TODOs | **COMPLETED** |
| 2026-08-01 | `feat/ppocr-next-20260731` | **Picked:** O11.7 PP-OCRv6 recognizer weight-cache optimization: immutable convolution dequantization, static-shape scheduler allocation, backend-resident weights, and explicit incompatible-tensor diagnostics are implemented. Corrected asymmetric graph convolution dimensions, tiny activation topology, and the 10-token head shape. CPU/Metal debug taps and small/medium multi-fixture graph gold pass with unchanged decoded output; the tiered gold harness now rejects stale 16-tensor references. Recognizer cache/graph work is complete; detector geometry acceptance and the remaining Metal orientation outlier stay explicitly gated | **COMPLETED** |
| 2026-07-31 | `main` | O11.7 persistent graph and weight-cache optimization: reuse static shapes, scheduler buffers, dequantized critical weights, and batched line crops | **COMPLETED — recognizer path; detector geometry remains gated** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.4 live PP-OCRv6 detector → quad crop → PP-LCNet line orientation → recognizer regression across 10 CC0/derived fixtures using cached Q8/F16 artifacts | **COMPLETED** |
| 2026-07-31 | `diagnose/pp-ocrv6-quality` / `.codex/worktrees/diagnose-pp-ocrv6-quality` | **Picked:** PP-OCRv6 Python/C++ detector geometry and crop parity on 10 CC0 fixtures; DBNet→PP-OCRv6 line/word path comparison added; quad handoff and PP-LCNet PIR→GGUF decoder landed; native classifier wired as optional `model_c` stage with NumPy/native cosine parity and CC0 sweep harness | **IN PROGRESS** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.2 deterministic problematic-input corpus: skew, border, illumination, haze, speckle, low-DPI, JPEG, rotation, perspective, and mixed-orientation variants with parent hashes/recipes | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.6 shared crop preparation contract: aspect/stretch geometry, fixed height/width, max width, padding, and RGB/grayscale output | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.4 structured line-orientation telemetry: detected angle/confidence and applied-rotation metadata through native/C APIs | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.8 benchmark accept-gate classification and preprocessor output provenance | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.5 model-free four-way page-orientation fallback and explicit C API; learned classifier remains separate | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.8 VLM cleanup safeguard: explicit opt-in barrier for destructive classical/learned cleanup on full-page VLM stages | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.8 structured stage metrics through the native/C pipeline API | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.5 orientation API surface: explicit `/preprocess/orientation` advisory endpoint with angle/confidence | **COMPLETED** |
| 2026-07-31 | `feat/ppocr-next-20260731` | **Picked:** O10.7 explicit multi-page autorotate option for `/ocr/document`, with confidence gate and temporary rotated page handoff | **COMPLETED** |

### OCR external head-to-head — CrispEmbed vs Tesseract / EasyOCR / PaddleOCR

Every OCR parity artifact in this repo so far compares CrispEmbed against a
*per-stage tensor reference*. That proves a graph is faithful to its blueprint;
it does not answer whether the shipped pipeline reads a page as well or as fast
as the engine it ports. Two new dependency-light scripts close that gap:

- `tests/ocr_synth_corpus.py` — deterministic rendered corpus (20 fixtures: 4
  paragraphs x clean/blur/noise/lowdpi/skew) carrying its own exact ground
  truth, so CER/WER becomes an absolute number. On clean rendered text every
  mature engine scores near zero, which makes a port bug separable from a hard
  input.
- `tests/ocr_external_parity.py` — runs system Tesseract, Python EasyOCR, Python
  PaddleOCR and the native lanes over the same images. Reports absolute CER/WER
  vs ground truth, *port-fidelity* CER vs the upstream engine's own reading of
  the same image, and latency in two deliberately separate columns: `proc_ms`
  (whole invocation, includes model load) and `engine_ms` (load-excluded, from
  the native stage-bench stderr or from the in-process Python engines). The
  columns are not comparable to each other; a non-zero exit is never timed as a
  win.

**Reachability fixes (the lanes were not runnable at all).** `engine::ppocrv6`
is fully implemented in `ocr_orchestrator.cpp` — detector, quad crop, PP-LCNet
orientation, recognizer, stage bench — but had **no `map_engine` C-ABI id and no
`--ocr-engine` name**, so the PaddleOCR-parity lane could not be invoked from
the CLI, the C ABI, or any binding. EasyOCR had a validated
`easyocr_pipeline::{load,run_file}` but no orchestrator engine at all. Both are
now wired: `--ocr-engine ppocrv6` (C-ABI id 16) and `--ocr-engine easyocr`
(id 17), plus `--ocr-cls` for the optional PP-LCNet 0/180 classifier and
registry entries naming the locally-converted artifacts.

**Where the lanes stand (20 synthetic fixtures with exact ground truth).**
Quality is settled; latency is only partly settled — see the caveat below.

| engine | kind | CER | WER | CER vs tesseract-cli |
|---|---|--:|--:|--:|
| `crispembed-ppocrv6` | native | **0.0031** | **0.0178** | 0.0253 |
| `paddleocr-py` | external | 0.0185 | 0.1153 | 0.0368 |
| `tesseract-cli:eng` | external | 0.0256 | 0.0890 | — |
| `crispembed-tesseract` | native | 0.0290 | 0.1623 | 0.0490 |
| `easyocr-py` | external | 0.0769 | 0.2363 | 0.0928 |
| `crispembed-easyocr` | native | 0.0808 | 0.3190 | 0.0974 |

All three native lanes reached or beat their upstreams on character error.
`crispembed-ppocrv6` — which produced pure noise at the start of this work — is
now the most accurate arm in the comparison, 8x below `tesseract-cli` and 6x
below PaddleOCR's own Python pipeline. `crispembed-tesseract` (0.0290 vs
0.0256) and `crispembed-easyocr` (0.0808 vs 0.0769) sit within noise of theirs.
WER remains the weaker column for the native lanes, which is a spacing/grouping
question rather than a recognition one and is the natural next target.

Starting point for comparison: `crispembed-tesseract` 0.0814, `crispembed-easyocr`
0.1412, `crispembed-ppocrv6` unusable.

**Latency: the dominant costs are found and two are fixed.** Three separate
things were eating the time, none of them the recognizer math:

1. **The detector enlarged pages that already resolved their text** — 84% of
   the tesseract lane (`detect=4938 ms` against `recognize=119 ms`). Capping the
   upscale is 4.7x and *also* cut CER 2.8x. Fixed, on by default (below).
2. **A CPU-only recognizer was initialising Metal to load itself.**
   `tesseract_lstm` is entirely host-side, but its loader asked for a GPU
   backend purely to pull the GGUF through `core_gguf::load_weights` — spinning
   up Metal, shader library and all, for a sub-2 MB model, then freeing it.
   Measured 4971 ms cold / 1069 ms warm against **4.8 ms** on the CPU backend.
   The whole invocation went **5.9 s -> 0.47 s (12.5x)**, output byte-identical.
   Fixed; `CRISPEMBED_TESSERACT_GPU_LOAD` restores the old path.
3. **PP-OCRv6's recognizer ran a CPU scalar SVTR.** Its detector already
   follows the correct never-upscale convention (`min(1, 960/max(w,h))`), so the
   remaining cost was compute. The now-correct graph is **~1.9x faster
   end-to-end with byte-identical decoded text on 26/26 fixtures** (20 synthetic
   + 6 CC0 scans, largest 71 regions; `synth_00_clean` 1214 -> 651 ms, the
   1920x2518 `german_official_print` scan 9369 -> 4964 ms). **Promoted to
   default**; `CRISPEMBED_PPOCRV6_NO_GRAPH` reverts. Scope is the recognizer
   only — the detector graph stays diagnostic-only on geometry parity, and the
   tiny variant keeps its own accept gate since the evidence is for small.

Where that leaves a one-shot CLI invocation on `synth_00_clean.png`. Measured
by a self-gating harness that waits for load average < 6, then brackets the run
with the `tesseract-cli` control before *and* after and discards the window if
the two readings differ by more than 30%. This window: control 0.12/0.15 s,
which is the true quiet baseline.

| engine | quiet wall | vs `tesseract-cli` | earlier this round |
|---|--:|--:|--:|
| `tesseract-cli:eng` | 0.135 s | — | — |
| `crispembed-tesseract` | **0.48 s** | 3.6x | 5.9 s at the start |
| `crispembed-ppocrv6` | **1.39 s** | 10.3x | 3.70 s |
| `crispembed-easyocr` | **1.47 s** | 10.9x | 2.06 s |

`tesseract-cli` is the only like-for-like comparison: one subprocess per image,
model load included on both sides. The Python arms (`easyocr-py` 0.75 s,
`paddleocr-py` 1.07 s) are measured in-process with the model already resident,
so quoting our per-invocation CLI against them would flatter or damn us
arbitrarily — that is exactly the `proc_ms`/`engine_ms` asymmetry the harness
refuses to collapse into one column.

**Remaining speed work.**

**The "shared GPU backend" item was wrong and is withdrawn.** It assumed a
pipeline initialises Metal once per engine. Counting the actual inits after the
detector fixes: **tesseract lane 0, EasyOCR 1, PP-OCRv6 1**. DBNet already
defaults to CPU (Metal conv is slower for it — `OCR_DETECT_USE_GPU` is opt-in),
and the two detector loaders now use the CPU backend, so no lane duplicates the
init. A refcounted shared backend would be speculative machinery for a problem
that no longer occurs; the plumbing was written, measured against reality, and
reverted. The cost is a *single* Metal init, not duplication.

**And that single init earns its keep** — checked rather than assumed, since the
tesseract lane's whole 12.5x came from deleting a Metal init. Median-of-3
CPU-seconds, control 0.35-0.39 s:

| lane | with Metal | forced CPU |
|---|--:|--:|
| EasyOCR recognizer | **3.65** | 6.43 |
| PP-OCRv6 recognizer | **3.25** | 3.75 |

So both defaults stay. The same run re-confirms the graph promotion from the
CPU-scalar reference: PP-OCRv6 **3.25 with the graph versus 5.50 with
`NO_GRAPH`**, a 1.7x that matches the 1.9x wall-clock figure measured earlier.

**EasyOCR profiled, and an earlier claim in this file corrected.** On a real
1920x2485 document (`commons_test_ocr_document.jpg`, 31 units) on a quiet box:
`load=2645 ms detect+recognize=12362 ms`. Compute dominates by 5x. The earlier
"load is 94% of the stage" reading came from the tiny synthetic page under heavy
contention, where a single Metal init blocked for tens of seconds — it was a
contention artifact, not a property of the lane, and should not be planned
against.

Same page, same DBNet detector, the two lanes differ almost entirely in their
recognizer: **tesseract lane 8.13 s wall / 7.86 s user, EasyOCR lane 17.98 s /
9.57 s user**. Per-line Tesseract LSTM runs 46-69 ms.

**Negative result — recognizer graph rebuilds are NOT the cost (do not retry).**
`easyocr_ocr_set_width()` tears down and rebuilds the graph whenever the canvas
width changes, and width is per crop (bucketed to a multiple of 64), so reading
order rebuilds every time consecutive lines land in different buckets. Sorting
regions by canvas width makes that O(distinct widths) instead of O(regions).
Implemented and A/B'd: **3.00 vs 3.02 CPU-s on a 47-region receipt and 7.43 vs
7.68 on the 31-unit document** — 0-3%, at the edge of noise. The rebuild is
cheap because it is graph construction plus a gallocr pass, with no weight
reload. Reverted rather than kept gated: ~25 lines of reordering in a
result-ordering-sensitive path is the wrong trade for 3%.

**What is actually left.** (a) PP-OCRv6's detector is CPU scalar convolution and
is that lane's dominant compute; its graph stays diagnostic-only on box-geometry
parity, so closing that parity is the unlock and it is real work, not a knob.
(b) DBNet costs ~400 ms on a capped 572x188 page against `tesseract-cli`'s
0.135 s for the whole job — CPU by deliberate choice (Metal conv measured
slower), so the win is kernel work. (c) The EasyOCR CRNN is ~2.2x the Tesseract
LSTM on the same detections and has no per-stage split below
`detect+recognize` yet.

**Measure with CPU time on this box.** `user+sys` stayed within 12% across runs
where wall clock swung 10x, and an early cross-run wall comparison of the
detector loader reported the *opposite* of what a same-binary A/B later showed.
Contention-robust ratios at the time of writing (median-of-3 CPU-seconds,
`tesseract-cli` control 0.46-0.49 s): tesseract lane 3.4x, PP-OCRv6 6.6x,
EasyOCR 8.9x control.

#### 2026-08-03 — "at least match PP-OCR original" verdict: warm A/B settled, first real-scan labels

The two holes named above are now measured (binary rebuilt at `main` HEAD,
Metal ON, `tests/ocr_external_parity.py`, repeats=3; raw JSON archived as
`parity-synth.json` / `parity-cc0.json` under `/Volumes/backups/ai/crispembed-gguf/`).

**Warm speed on the 20-fixture synth corpus is NOT matched.** Load-excluded
`engine_ms`, both arms interleaved in the same window (`tesseract-cli` control
476 ms vs the 135 ms quiet baseline, so read ratios, not absolutes): native
median **2085 ms** vs `paddleocr-py` **1484 ms** — per fixture 1.25-1.83x,
sum-over-corpus **1.48x** slower. Quality re-confirmed unchanged (CER 0.0031
vs 0.0185; native wins or ties 19/20 fixtures).

**Where the warm gap lives (stage bench, quiet box).** `synth_00_clean`:
`detect=1415.6 crop=0.3 orientation=0.2 recognize=594.2` — the CPU-scalar
detector is **70%** of a small page, so T3/T6/T7 are the levers there.
`commons_example_receipt` (40 crops, 11 distinct widths): `detect=2034.6
recognize=4238.9` — many-crop pages are recognizer-bound, which is T4's
batching case. One-shot load adds `recognizer+orientation=1.1 s` (T5).

**Real scans, scored for the first time (T1 labels landed,
`tests/regression/images/cc0/ground_truth.json`, 5 fixtures transcribed by eye
with zoom verification; `simple_table.jpg` stays directional-only — its cell
digits are unrecoverable, and both paddle and native detect 0 regions on it).**

| fixture | paddle CER | native CER | verdict |
|---|--:|--:|---|
| `german_official_print.jpg` (Fraktur letterpress) | 0.2250 | **0.0486** | native 4.6x better |
| `receipt_historical.png` (dot-matrix) | 0.2409 | **0.0260** | native 9x better |
| `commons_test_ocr_document.jpg` (two-column) | 0.7638 | 0.7612 | tie — CER for *every* arm (tesseract 0.7551 too) is dominated by column reading order vs the column-ordered ground truth, not recognition |
| `commons_example_receipt.png` (clean monospace) | **0.0074** | 0.0885 | native 12x worse |
| `simple_form.png` (452x317 UI, ~7 px labels) | **0.6275** | 0.7368 | native worse |

Warm speed on these pages: native median `engine_ms` 9414 vs paddle 7964
(**1.18x**), and native is *faster* than paddle on 2 of 5 (two-column doc
15.6 s vs 20.7 s, clean receipt 6.2 s vs 7.9 s).

**The clean-receipt loss is a systematic symbol-class failure, not noise:**
`$`→`S` throughout, `I`/`f`/`1`→`:`, `H`→`ll` (`Have`→`llave`), while paddle
reads the same lines correctly with identical region count (40). Probed:
`CRISPEMBED_PPOCRV6_RGB=1` changes nothing; the medium recognizer fixes word
shapes (`Qty`, `Price`, `Product`) but keeps the `$`→`S` class. The GGUF
embeds the full 18,710-class label list, so it is not a truncated charset.
Two candidate causes, deliberately not guessed at: (a) the known blocked gate
— official PP-OCRv6 *recognizer* inference is still unrecoverable locally
(PaddleX exposes only the detector blueprint), so text-gold parity on these
crops cannot be checked on this box; (b) model asymmetry — `paddleocr-py`'s
English lane runs a 96-class `en` recognizer while our lane runs the
18,710-class multilingual PP-OCRv6, and thin-glyph symbol confusion is exactly
where that asymmetry would bite. See T10.

**Verdict against the "at least match PP-OCR original" requirement.**
*Quality:* ahead of the original on rendered text (6x) and on hard real scans
(Fraktur 4.6x, dot-matrix 9x); behind on clean monospace symbols and tiny UI
text. *Speed (warm, load-excluded):* behind 1.4-1.5x on small pages
(detector-bound), 1.18x median on real pages, ahead on some large pages. Not
yet a uniform match; the remaining work is exactly T3/T4/T5/T6 (speed) and T10
(the symbol-class quality gap).

#### 2026-08-03 (later) — the "official v6 is unrecoverable locally" blocker is DEAD; T10 root-caused to a port bug; the true baseline moved

Everything below supersedes the 2026-08-01 "recognizer blueprint unrecoverable /
quality gate blocked" record and the paddleocr-2.10 arm as the parity target
(2.10 runs a 96-class `en` PP-OCRv4 recognizer; it remains a valid engine-level
reference but is NOT the original of what we ported).

**Official PP-OCRv6 now runs on this Mac, three independent ways:**
1. `~/venvs/paddleocr3` — paddleocr 3.7.0 + paddlepaddle 3.3.1; v6 is the
   default pipeline; `PaddleOCR(text_detection_model_name="PP-OCRv6_small_det",
   text_recognition_model_name="PP-OCRv6_small_rec")`; models cached under
   `~/.paddlex/official_models/`. (HPI/ONNX plugin is Linux-x86-only — plain
   Paddle backend is the Mac path.)
2. `~/venvs/rapidocr` — a community onnxruntime pipeline (3.9.2) with ONNX exports
   of PP-OCRv6 tiny/small/medium det+rec, Paddle-free, models cached in the
   venv's `rapidocr/models/`. `TextRecognition`-style single-crop runs and
   full-pipeline both work; also the ONNX graphs double as architecture ground
   truth (op histogram small rec: 13xErf-GELU, 5xSiLU, 10xReLU, 5xHardSigmoid,
   softmax in-graph).
3. Source blueprints, freshly cloned (shallow) under `/Volumes/backups/code/`:
   `PaddleOCR` (main @2661c7c — full v6 rec modeling source
   `ppocr/modeling/backbones/rec_lcnetv4.py`, neck `necks/rnn.py:242-345`
   `EncoderWithLightSVTR`, head `heads/rec_ctc_head.py`, C++ reference
   `deploy/cpp_infer`, ONNX tar URLs in
   `paddleocr-js/packages/core/src/resources/model-asset.ts:19-29`),
   community ONNX-runtime ports of v6 (one MIT C++ pipeline with an exact
   pre/post contract and CPU perf tricks).

**T10 verdict: PORT BUG, not model asymmetry.** There is no `en`-specific v6
recognizer (all 50 languages share the multilingual model — the community ONNX port confirms:
`model_resolver.py:104-108`), and BOTH official runners read
`commons_example_receipt.png` essentially perfectly with the same
tier/generation we run: paddleocr-3.7 v6_small got all 12 `$`, every `1`, and
`Transaction ID` (47 regions); the v6_small ONNX export likewise (one `S`, one
`_Card` artifact). Our lane's `$`->`S`, `I`/`f`/`1`->`:`, `H`->`ll` is ours.

**The true speed baseline (same v6_small models, warm, this M1):** official
paddle `synth_00_clean` **0.83 s** vs our engine 2.0 s (**2.4x slower** — worse
than the 1.4x measured against the 2.10/v4 arm); receipt official **5.9 s** vs
our 6.2 s (near par). The community onnxruntime pipeline does the receipt in **2.2 s** —
2.7x faster than both Paddle and us; that is the realistic CPU ceiling to aim
at, and ORT gets it with width-bucketed batching (see T4 notes).

#### 2026-08-04 — speed round (`perf/ppocrv6-speed`, merged): the "1.48x warm gap" was mostly an accounting artifact; one-shot is now workload-adaptive

**The stage-bench was lying.** `[ppocrv6-stage-bench] detect/total` spanned
STAGE ENTRY, folding the ~1.1 s one-shot recognizer Metal init into "detect"
— so every "load-excluded" `engine_ms` built on that line (including the
2026-08-03 1.48x verdict) was inflated by ~1.1 s. Fixed to net-of-load (the
tesseract lane was already correct; the easyocr harness regex captured the
load-inclusive `total=` and now captures `detect+recognize=`).

**Corrected numbers (same binary, quiet windows, tesseract-cli control
184-480 ms).** synth_00_clean true warm compute: detect 320 ms (conv 285 of
it) + recognize 553 ms Metal = **874 ms vs official paddle-3.7 v6_small's
830 ms warm in-process — 1.05x, near-parity**, not 1.48x. Corpus medians,
net-of-load: synth native-Metal 1018 ms / native-CPU 2001 ms; labelled-CC0
native-Metal **6712 ms** — faster than paddleocr-2.10's 7933 ms and in the
same band as paddle-3.7 (receipt: official 5.9 s vs our 6.5 s window-noisy).

**T5 landed, then corrected to workload-adaptive.** Always-CPU one-shot
(first cut) was right for a 3-box page (1.86 s vs 2.04-2.17 s Metal) and
wrong for a 47-box page (14.0 s vs 8.1 s). Detection (a ~3 ms CPU load) now
runs BEFORE the recognizer initialises; in CLI one-shot mode
(`CRISPEMBED_PPOCRV6_ONESHOT`) the orchestrator forces CPU only when boxes <=
`CRISPEMBED_PPOCRV6_ONESHOT_CPU_MAX_REGIONS` (default 8, from the ~175 ms/crop
CPU-minus-Metal delta vs the ~1.1 s init). Measured one-shot walls:
synth_00_clean **1.83-1.99 s** (CPU picked), receipt **7.37-8.08 s** (Metal
picked), text byte-identical both. Server/library defaults unchanged.

**T4 narrow-crop experiment: real but not free — kept opt-in.**
`CRISPEMBED_PPOCRV6_WIDTH_FLOOR=<n>` replaces the pad-everything-to-320
contract with natural width (ceil to 32). Receipt recognize 12425→8097 ms at
floor 128 (1.53x) with byte-identical text — but the full-corpus gate found
floor<=192 flips 1-2 tiny single-glyph crops (`@`→`0` on receipt_historical
at 128; same fixtures still differ at 192), so the official 320 floor stays
default. All 20 synth fixtures are byte-identical at any floor (their crops
are wider than 320 natural).

#### 2026-08-04 (evening) — competitive verdict vs the community onnxruntime pipeline; width bucketing default; two negative results

Same-window interleaved shootout on this M1, warm, same v6_small models:

| page | ours | community ORT pipeline | official paddle-3.7 |
|---|--:|--:|--:|
| synth_00_clean (3 crops) | **555 ms** | 710-923 ms | 830 ms |
| commons_example_receipt (47 crops) | ~2.9-3.3 s | **1.84 s** | 5.9 s |

Quality on the receipt: ours 0 errors, the ORT pipeline 2 (`S`, `_Card`),
paddle-3.7 0. So: **ahead of everything on small pages, ahead of official
paddle everywhere, behind the ORT pipeline ~1.5x only on many-crop
throughput.** The residual is precisely attributed: with fused batching the
receipt's recognize is ~2.2 s of genuine Metal graph compute (~33 ms/crop in
batch-8 groups; shape rebuilds only ~200 ms) at ~7 GF/s — ggml's Metal conv
efficiency at 48-pixel heights, not our graph structure. That is
kernel-level work (upstream ggml territory), recorded as the open item.

**2026-08-04 (night) — the kernel arc ran its course; the residual is now
exactly attributed and every structural alternative is measured.** Metal
profiler (host-encode vs GPU-execute) + stage-stop diagnostics on the
receipt's batch-8 groups: encode <1 ms, ~300 ms/group is GPU execution of 686
nodes in ONE split (no CPU fallbacks). Backbone (~550 conv nodes) = **58 ms**;
neck+SVTR decoder (~130 tiny-tensor nodes on [120,320]) = **~250 ms**; head
free. Group cost scales linearly with items → real per-item work in the
decoder op mix, not dispatch latency. Four alternatives measured negative,
kept as env gates (`perf/ppocrv6-conv`, merged): direct-conv kernel
(recognizer 3.6x slower, detector unchanged — the Metal det penalty is NOT
im2col), hybrid CPU decoder (scalar 18,710-class head too expensive),
mega-batch single dispatch (padding waste dominates), plus the earlier F16
residents and native-q8 negatives. **Remaining speed item: ggml Metal
kernels for the small-tensor decoder op mix — upstream work with the op-mix
profile on record.**

Landed: **width bucketing default (step 64)** — model width rounds UP to a
multiple of 64 so nearby widths share a graph shape and fuse (receipt: 12
widths → 5 fused groups, recognize −11%; the synth page's 3 crops now fuse
into one group, 688→555 ms; 25-fixture CER mean 0.06408 vs 0.06410, jitter
both ways; `CRISPEMBED_PPOCRV6_WIDTH_BUCKET=0` disables).

Negative results, measured interleaved, kept as env gates: **F16 conv
residents on Metal are 33% SLOWER** for these shapes (3/3 pairs,
`CRISPEMBED_PPOCRV6_GRAPH_F16=1` re-enables); **native-q8 linear residents
are flat** because the q8 policy artifact ships its head high-precision
(passthrough kept for genuinely-quantized heads). CPU-backend fused batch is
also slower than Metal fused (4.5 s vs 2.6 s receipt) — the CPU graph is not
the escape route either.

**What still separates us from the ceiling.** (a) ~~detector~~ **T7 closed
same day** — the detector graph (CPU backend) is now the default at 1.5-1.7x
over scalar (see T7 [CLOSED]); with it, synth_00_clean warm compute is
~**730 ms vs official paddle-3.7's 830 ms — we are AHEAD of the original on
small pages**; (b) many-crop pages want the Metal fused batch graph (N1,
adopted): per-crop dispatch is the remaining distance to the
onnxruntime ceiling on the receipt (2.2 s); (c) T3/H2 scalar-kernel work now
only matters for the scalar escape path and medium tier; (d) Metal conv perf
is the blocker for detector-on-GPU (9x slower than CPU graph today) — on
this M1 the evidence says CPU graph is the right default, GPU via
CUDA/Vulkan is where the graph's portability pays.

**2026-08-04 postscript — resolved.** The stage-diff never had to run: the
bisection (recognizer correct on raw crops, corruption reproduced by running
the direct harness on the cleanup stage's output image) landed on the
scan-cleanup preprocessing, not the recognizer. See T10 [RESOLVED] for the
fix, the post-fix numbers (receipt CER 0.0025, CC0 lane now faster AND more
accurate than paddleocr-py), and the synth `_noise` trade-off now owned by T2.
The suspect list below is retained because it documents the official inference
contract, which remains the reference for any future recognizer work.

**Activation audit came back CLEAN** — our stem ReLU / channel-mixer GELU /
neck SiLU / tiny-guide Hardswish placement matches the recovered official
source, so the bug is in a subtler contract point. Ranked suspects with the
official file:line, all now diffable against the local ONNX: (1) backbone exit
pooling at inference is `avg_pool2d(kernel=(3,2))` with implicit **stride
(3,2) — width halved**, training uses adaptive `[1,40]`
(`rec_lcnetv4.py:637-642`); (2) residual wraps the **channel mixer only**, the
token-mixer skip is folded into the fused RepDWConv (`rec_lcnetv4.py:501-505`
— the paper formula is misleading); (3) SVTR `Block` `prenorm=False` actually
means PRE-norm arithmetic (flag name inverted, `rec_svtrnet.py:272-278`); (4)
preprocessing: BGR, per-image `max_wh_ratio=max(320/48, w/h)`, INTER_LINEAR,
right-pad with 0.0 **in normalized space** (= gray 127.5), softmax already
in-graph (`deploy/cpp_infer/.../processors.cc:48-128`, `rec_ctc_head.py:117`).

**Measurement discipline, learned the hard way here.** This box runs 3–6
concurrent agent builds; load average hit 103 mid-sweep and `tesseract-cli`
itself drifted from 150 ms to 10.3 s inside the same harness. Every number
above is quoted with the control that bracketed it, and several runs were
discarded outright. `tests/ocr_external_parity.py` prints the `tesseract-cli`
arm for exactly this reason — if it is far above ~150 ms the run is measuring
contention, not engines.

**The detector, not the recognizers, is the whole latency gap — and its resize
rule is an upstream deviation.** Stage bench on `synth_00_clean.png` via the
tesseract lane: `detect=4938 ms group=1.8 ms crop=0.8 ms recognize=119 ms`. So
84% of the time is DBNet, and the recognizers are already cheap. The cause is
`ocr_detect::detect_rgb_ex`'s `scale = target_short_side / min(w,h)` with no cap
at 1.0: a 572x188 fixture is enlarged ~3.5x to ~2016x672, i.e. **12.6x the
source pixels**. Upstream DBNet/PaddleOCR (`limit_type="max"`) only ever
shrinks.

`detect_options::max_upscale` (env `CRISPEMBED_OCR_DET_MAX_UPSCALE`, 0 = today's
uncapped behaviour) makes this A/B-able without a rebuild. Same binary,
back-to-back, 20 synthetic fixtures, `tesseract-cli` held as a load control at
120–157 ms across all arms:

| lane | uncapped | cap 1.5 | cap 1.0 |
|---|--:|--:|--:|
| `crispembed-tesseract` CER | 0.0814 | 0.0309 | **0.0290** |
| `crispembed-tesseract` proc ms | 6462 | 2145 | **1369** |
| `crispembed-easyocr` CER | 0.1412 | 0.0955 | **0.0808** |
| `crispembed-easyocr` engine ms | 6295 | 2396 | **1565** |

That is 2.8x lower character error at 4.7x the speed for the tesseract lane, and
it lands both lanes at CER parity with their upstreams (`tesseract-cli` 0.0256,
`easyocr-py` 0.0769). Enlarging a page the scan never resolved was costing both
time *and* accuracy.

**It now ships on by default, gated on image size rather than switched off.**
The first cut was left off because `tests/regression/images/cc0/simple_table.jpg`
(200x102 — a thumbnail, not a page) went from one detected region to zero when
capped. The fix is `upscale_floor`: the cap applies only once the short side is
at least 120 px, which is the difference between "this page already resolves
its text" and "this is a thumbnail". 120 is measured, not picked — the 616x149
low-DPI fixtures are *better* capped on both axes (CER 0.025 at 1.25 s versus
0.066 at 58 s), so the exemption must not reach them, while the 102 px
thumbnail needs it. Verified by the resize decisions: 616x149 and 572x188 now
pass through unscaled while 200x102 still goes to 1443x736 and keeps its
region. Both knobs stay overridable
(`CRISPEMBED_OCR_DET_MAX_UPSCALE`, `CRISPEMBED_OCR_DET_UPSCALE_FLOOR`).
A cap=2.0 arm was also run but is discarded: `tesseract-cli` went 157 -> 1936 ms
on the same fixture inside it, so that run was contended, not slow.

**PP-OCRv6 — root-caused against the real PaddleOCR source and FIXED.**

The accepted PP-OCRv6 parity evidence was per-stage cosine against
`tools/dump_ppocrv6_reference.py`, a hand-written torch mirror with guessed
details — and the mirror could not read text either, so the native port that
matched it at cosine 0.9999 could not read text either. `git clone`ing
PaddleOCR and reading `ppocr/modeling/backbones/rec_lcnetv4.py`,
`ppocr/modeling/necks/rnn.py` and `tools/infer/predict_rec.py` settled every
guess. Four were wrong:

1. **Stem activation.** `StemBlock` is built from `ConvBNAct`, whose activation
   is `ReLU()`; the mirror used SiLU. (Landed on `main` in parallel by the
   Tesseract session.)
2. **The neck dropped the local-conv residual.** Upstream is
   `z = z + local_conv(z)`; the mirror had `z = local_conv(z)`.
3. **The neck skip landed in the wrong place.** `skip = skip_conv(x)` is
   computed first but added **after** the SVTR blocks and the final norm; the
   mirror added it before the blocks and never after.
4. **Recognizer input width.** `max_wh_ratio` is seeded with `imgW/imgH` and
   grows to the widest crop, so 320 is a **floor**: a 520x35 line is 713 px.
   The mirror and the runtime capped width at 320, crushing 44 characters into
   40 CTC timesteps — undecodable by any model, correct or not.

A fifth lives in the vocabulary: `use_space_char: true` makes the label list
`blank + 18708 dict entries + ' '`, which is where the head's 18710 outputs
come from. The GGUF carries only the 18708, so class 18709 decoded to nothing
and every space was dropped.

Result on `synth_00_clean.png`, native CPU lane, all three lines exact
including punctuation:

| before | after |
|---|---|
| `iiiiii` | `The quick brown fox jumps over the lazy dog.` |
| `laúieyotiieieioieioni.` | `Pack my box with five dozen liquor jugs.` |
| `íotuinióniiaieiasaieró` | `How vexingly quick daft zebras jump!` |

Setting `CRISPEMBED_PPOCRV6_FIXED_WIDTH=1` reproduces the old
`Te qu c   vr  .` exactly, which is what pins the width cap as the cause rather
than a correlate.

Open follow-ups for this lane: (a) **latency is unmeasured** — the box sat at
load average 101 from concurrent sessions during the A/B, and wider input plus
the O(tokens^2) CPU scalar SVTR attention means a real cost that needs a quiet
back-to-back run; (b) the opt-in `CRISPEMBED_PPOCRV6_{SVTR_,DET_}GRAPH` paths
still fold the skip the old way and were not touched; (c) the converter should
emit the space class itself rather than the runtime appending it; (d) the
recognizer still confuses `e`/`c` on one Courier fixture, which is an ordinary
error class, not a structural one. Re-run
`tests/ocr_external_parity.py` with `crispembed-ppocrv6` enabled once the box is
quiet to put this lane in the head-to-head table.

**Original stage-bench observation (PP-OCRv6 small, Metal, `synth_00_clean.png`, 3 lines).**
Decoded output is not text: `iiiiii` / `laúieyotiieieioieioni.` /
`íotuinióniiaieiasaieró` against a ground truth of "The quick brown fox jumps
over the lazy dog." — with `mean_conf=0.94`, so the confidence signal does not
detect it. Detector geometry is plausible (3 boxes for 3 lines); the recognizer
output is wrong. Stage bench on the same run: `detect=62177.3 ms
recognize=35630.8 ms total=97844.2 ms`, i.e. ~98 s for a 600x200 image that
PaddleOCR reads in a fraction of a second. That run was CPU-contended; uncontended,
`test-ppocrv6-direct` reports 1774 ms total for the same three lines, so the
98 s figure is contention and the standing latency question for this lane is
moot until it produces text at all.

### OCR performance backlog — every idea, with status and handover

Status vocabulary: **DONE** shipped on by default; **GATED** implemented, works,
output-verified, default off with the measurement that kept it off; **OPEN** not
implemented. Every gated path stays in the tree — a path that does not win today
can win under a different engine mix, and re-deriving it costs more than the
gate does.

#### Shipped (default on)

| # | Change | Win | Gate to revert |
|---|---|--:|---|
| P1 | Detector stops enlarging pages that already resolve their text (`max_upscale`, `upscale_floor=120`) | 4.7x **and** CER 0.0814→0.0290 | `CRISPEMBED_OCR_DET_MAX_UPSCALE=0` |
| P2 | Tesseract-LSTM loads via CPU backend instead of spinning up Metal for a host-side engine | lane 5.9 s → 0.47 s (12.5x) | `CRISPEMBED_TESSERACT_GPU_LOAD=1` |
| P3 | PP-OCRv6 small/medium recognizer graph promoted | 1.9x wall / 1.7x CPU, 26/26 identical text | `CRISPEMBED_PPOCRV6_NO_GRAPH=1` |
| P4 | PP-OCRv6 detector loads via CPU backend | 7-14% CPU | `CRISPEMBED_PPOCRV6_DET_GPU_LOAD=1` |

#### Implemented, gated off (working — reuse these before rewriting them)

| # | Path | Env | Why it is off | When it becomes worth turning on |
|---|---|---|---|---|
| P5 | Process-shared GPU backend (refcount-free singleton + `crispasr_free_gpu_backend`) | `CRISPEMBED_SHARED_GPU_BACKEND=1` | After P2/P4 no lane inits Metal more than once (tesseract 0, EasyOCR 1, PP-OCRv6 1), so it saves nothing today | The moment two GPU-resident engines share a process: a VLM stage beside a recognizer, the detector graph being promoted, or server/batch use. **Hazard:** one `ggml_backend_t` driven from several threads (`CRISPEMBED_TESSERACT_WORKERS`) is not promised safe |
| P6 | EasyOCR width-sorted recognition (O(distinct widths) graph rebuilds instead of O(regions)) | `EASYOCR_WIDTH_SORT=1` | 0-3%, edge of noise — the rebuild is graph construction + gallocr, no weight reload | If the recognizer graph gains an expensive build step (weight residency, kernel specialisation, a shape-keyed resident cache). Also the natural companion to P9 |
| P7 | PP-OCRv6 detector full graph | `CRISPEMBED_PPOCRV6_DET_GRAPH=1` | Box geometry not at parity (31 boxes vs CPU 30) | See O1 — this is the single biggest remaining win |
| P9 | Tiled + 4-wide-unrolled 1x1 convolution (`conv2d_1x1_cpu`) | `CRISPEMBED_CONV1X1_FAST=1` | **Not a win.** Interleaved pairs: neutral on M1/NEON (-0.3% excluding one outlier baseline, CI spans zero), **-4.8% regression** on x86/AVX2 (5/6 pairs negative). Decoded text identical on both | Only if a future engine or shape shows a real gain under interleaved-pair measurement with n reported |
| P10 | Loop-inverted depthwise convolution (`conv2d_depthwise_cpu`) | `CRISPEMBED_CONVDW_FAST=1` | Single-shot 3.6% on M1 is **unverified** against a measured 8.1% noise floor; neutral on x86. Decoded text identical on both | After the tap-unrolling rework, re-measured as interleaved pairs |
| P11 | Four-accumulator `dot_product_wide` | `CRISPEMBED_DOT_WIDE=1` | Slower on M1 (8.52 -> 8.90, single shot). Built to test the hypothesis that 2 accumulator chains starve the M1's FMA pipes; the prediction failed, which is why that explanation was withdrawn | If a future profile shows a genuinely FMA-latency-bound dot rather than a memory-bound one |

#### Open, in descending measured value

**O1 — PP-OCRv6 detector graph box-geometry parity is NOT a speed item.**
Corrected 2026-08-02 by measuring it. The premise in earlier notes — that the
CPU scalar detector is slow and promoting its graph would fix that — is wrong.
Quiet box (load 4.2), 1920x2518 page, same fixture and binary:

| detector path | total |
|---|--:|
| **CPU scalar (shipped default)** | **2350 ms** |
| graph on the CPU backend | 6132 ms (graph alone 2829 ms) |
| graph on Metal (`CRISPEMBED_PPOCRV6_DET_GRAPH=1`) | 15933 ms (graph alone 12663 ms) |

The graph is 2.6x slower on CPU and 6.8x slower on Metal than the hand-written
scalar path it falls back to, which matches the DBNet finding already recorded
in `ocr_detect.cpp` (Metal conv2d/conv-transpose measured ~139 s GPU vs ~10 s
CPU on an M1). So closing box-geometry parity is a **correctness and
portability** goal — it is what a CUDA or Vulkan deployment would need, and it
removes a diagnostic-only caveat from the matrix — but nobody should expect a
speedup from it on this hardware, and it should not be prioritised as one.
Anyone picking it up: the geometry comparator already exists
(`report_graph_box_geometry`, `CRISPEMBED_PPOCRV6_DET_GRAPH_COMPARE=1`), though
it did not emit on the run above and needs its call site checked first.

**O2 — the detector's CPU scalar path is the real target, at 2350 ms.** It is
the dominant cost of the PP-OCRv6 lane and DBNet is the shared detector for the
other two, and ggml graphs are demonstrably the wrong tool for it here (O1). So
this is kernel work on the scalar code: per-node-trace one detect call, and
check specifically whether 1x1 convolutions are being routed through an im2col
path, since a 1x1 conv is a pure channel matmul and its im2col is a copy that
materialises a large intermediate for nothing. The `QWEN3_TTS_CODEC_FASTCONV`
precedent in the dev guide is exactly this shape and was worth 3x.

**O2 answered 2026-08-02 — the trace exists now, and it says the cost is
concentrated in two shapes, not spread out.** `CRISPEMBED_PPOCRV6_DET_PROFILE=1`
prints a per-convolution table keyed on shape signature, sorted heaviest first,
with GF/s so a slow layer is distinguishable from a merely large one. On
`german_official_print.jpg`:

| class | share of detector conv time | rate |
|---|--:|--:|
| 1x1 pointwise | **51.6%** | ~1.2 GF/s |
| depthwise | **20.4%** | **0.02-0.19 GF/s** |
| deconv | 6.4% | ~0.6 GF/s |
| other conv | 21.6% | ~1.0 GF/s |

Read the shares, not the totals: the absolute figures move by 1.5x between runs
on this box, the proportions do not. Two consequences. First, **H1 is the lever
for O2/H2** — the pointwise layers are not "one thing to check", they are half
of everything. Second, **depthwise is a lever nobody had listed**: the single
most expensive layer in the whole network is one 7x7 depthwise at 240x184, 13.7%
of all convolution time at 0.17 GF/s. That is the generic path's worst case by
construction — with one input and one output channel per group there is nothing
to amortise the patch gather against, so it gathers a kh*kw window and consumes
it in a single dot_product, per output pixel. `conv2d_depthwise_cpu`
(`CRISPEMBED_CONVDW_FAST=1`) inverts the nest instead: per channel and output
row, walk the taps and accumulate a whole row per tap, so each tap is a
contiguous axpy, the input row stays in L1 across all taps, the gather
disappears, and the boundary test becomes a per-tap column range in closed form.
Implemented and equivalence-guarded; **not yet A/B'd for speed.**

**O3 — EasyOCR CRNN is ~2.2x the Tesseract LSTM on identical detections**
(17.98 s vs 8.13 s wall on the same 31-unit page, same DBNet boxes). No
per-stage split exists below `detect+recognize` yet; add one mirroring the
tesseract/ppocrv6 load-vs-compute benches before targeting anything.

**O4 — Batched crop recognition.** Every lane recognizes line crops one at a
time. Crops sharing a canvas width could go through one graph dispatch as a
batch dimension, which is where P6's width grouping stops being cosmetic. Needs
a batched graph in each recognizer; largest expected win on many-region pages
(71 regions on one CC0 scan).

**O5 — Model load is still ~0.37 s of the tesseract lane's 0.47 s.** Now that
compute is small, load dominates a one-shot CLI invocation. Options: mmap the
GGUF instead of copying weights into host vectors, or keep a warm process /
server for repeated pages. `tesseract-cli` pays load per invocation too and
still totals 0.135 s, so there is headroom here.

**O6 — Detector resize rule keyed on estimated text height rather than image
size.** P1's `upscale_floor=120` is a proxy: what actually matters is whether
glyphs are tall enough for the detector, not whether the page is. A cheap
stroke-width or connected-component estimate would let the cap apply to a
low-DPI thumbnail with big text and stay off for a high-DPI page of tiny text.

**O7 — Per-engine profiling of the remaining VLM/OMR lanes.** This whole round
covered only the three classical lanes. The load-vs-compute split found a wasted
Metal init in two of three engines it was applied to; it has not been applied to
GOT/GLM/Qwen/InternVL/SmolDocling/the OMR engines, and the same class of bug
(GPU backend created for a host-side path) is plausible in any of them. Cheap to
check: `grep -n crispasr_init_gpu_backend src/*.cpp` and ask, per engine,
whether its compute actually runs on that backend.

### OCR performance — self-contained handover prompts

Everything a fresh agent needs is in this section. Read **§0 Setup**,
**§1 How to measure** and **§3 Current state** once, then take a task from
**OPEN TASKS** below. Nothing here assumes prior context beyond this file.

> **Start here.** Tasks are **T1-T7**, ordered by expected value. **T1
> (transcribe 5-10 CC0 scans) blocks T2 and is the highest-leverage item** — the
> real-scan half of this backlog cannot be scored without it, and seven
> successive probes were already falsified against unlabelled proxies.
>
> Two things will waste your time if you skip §1 and §3: the old median-of-3
> timing recipe **cannot resolve** effects below ~15% on the Mac (measured sd
> 8.1% per paired delta — use interleaved pairs), and three CPU kernels plus
> seven routing probes have already been written and measured **not** to work.
> The Aug 2026 round is written up in `HISTORY.md`; read that before re-attempting
> anything in this area.

#### §0 Setup (do this first, every time)

```bash
# 1. NEVER edit the main checkout. Make a worktree.
cd /Users/christianstrobele/code/CrispEmbed
git worktree add .claude/worktrees/<your-task> -b <your-branch> main
cd .claude/worktrees/<your-task>

# 2. ggml is a gitlink placeholder in a fresh worktree; cmake needs a real tree.
#    Symlink it TO BUILD, restore the gitlink BEFORE any git command.
rm -rf ggml && ln -s /Users/christianstrobele/code/CrispEmbed/ggml ggml

# 3. Configure + build (Metal + embedded shaders; ~15 min cold, ~1 min warm)
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BLAS=ON
cmake --build build -j6
```

**The ggml trap, which will bite you.** `git stash`, `git checkout`, `git add`
and `git commit` all refuse to run (or silently reset the link) while `ggml` is
a symlink. The cycle is: symlink → build/measure → `rm -f ggml && git checkout
HEAD -- ggml` → `git add`/`commit`/`push` → symlink again. If you skip the
re-symlink, the next `cmake --build` fails with `ninja: error: rebuilding
'build.ninja'` **and you will unknowingly measure the stale binary**.

**Models** live in `~/crispembed-live-cache/` (also mirrored at
`/Volumes/backups/ai/crispembed-gguf/`, which is often unmounted — prefer the
home path). The five used below are all present:

| lane | detector | recognizer |
|---|---|---|
| tesseract | `dbnet-ic15-q8_0.gguf` | `tesseract-eng-q8_0-seeded.gguf` |
| easyocr | `dbnet-ic15-q8_0.gguf` | `easyocr-english-g2-f16.gguf` |
| ppocrv6 | `PP-OCRv6_small_det-f16.gguf` | `PP-OCRv6_small_rec-q8-head.gguf` |

**Python** is `~/miniconda3/bin/python` — never the system `python3`. Set
`USE_TF=0` for anything importing transformers.

**Test corpus** — 20 fixtures that carry their own exact ground truth, so CER is
absolute rather than cross-engine agreement. It is generated, not checked in:

```bash
~/miniconda3/bin/python tests/ocr_synth_corpus.py --output ~/crispembed-ocr-synth
```

**⚠ Do not put the corpus under `/tmp`.** Verified 2026-08-02: a corpus written
to `/tmp/ocr-synth` is readable by `build/crispembed` but **not** by the Homebrew
`tesseract`, which fails with `Leptonica Error in findFileFormat: image file not
found` — the session's `/tmp` is a private mapping the external binary cannot
see. Since `tesseract` is the load control for every measurement below, a `/tmp`
corpus silently breaks the control while the native lanes appear to work. A home
path works for both. (`/tmp` also gets wiped between sessions.)

Real scans live at `tests/regression/images/cc0/` (no ground truth; use them for
cross-engine agreement and for many-region stress — `commons_test_ocr_document.jpg`
is 1920x2518 and yields 31 units, `receipt_example.png` yields 47).

**Run one lane:**

```bash
C=~/crispembed-live-cache
./build/crispembed --ocr-pipeline ~/crispembed-ocr-synth/synth_00_clean.png \
  --ocr-engine ppocrv6 --ocr-det $C/PP-OCRv6_small_det-f16.gguf \
                       --ocr-rec $C/PP-OCRv6_small_rec-q8-head.gguf
# expected: the three-line pangram, exactly, punctuation included
```

**Run the full head-to-head** (native lanes vs system Tesseract / Python EasyOCR
/ Python PaddleOCR, reporting CER, WER and latency):

```bash
USE_TF=0 ~/miniconda3/bin/python tests/ocr_external_parity.py \
  --images ~/crispembed-ocr-synth --model-dir ~/crispembed-live-cache --repeats 3
```

#### §1 How to measure on this box — read this or your numbers will be wrong

**Use interleaved off/on PAIRS, not two separate medians.** This supersedes the
old median-of-3 recipe, which was measured in Aug 2026 to be incapable of
resolving the effects this backlog deals in. Measured standard deviation of a
paired delta:

| host | sd of paired delta | interleaved pairs to resolve a 5% effect at 95% |
|---|--:|--:|
| M1 Mac (shared, load 15-110) | **8.1%** | **41** |
| VPS Xeon (idle) | **5.2%** | **16** |

A median-of-3 is three samples. Comparing two separately-taken medians is
strictly worse than pairing, because slow drift lands entirely in one arm — that
is exactly how a single A/B produced "+15.7%" for a kernel that is really
neutral, with the outlier in its *baseline* rather than its gated arm. The
tesseract control bracket does **not** rescue it: both controls agreed within 30%
across that very pair.

```bash
# Interleaved paired A/B. Report EVERY pair and the spread, never one number.
one() { /usr/bin/time -p "$@" 2>&1 >/dev/null \
        | awk '/real|user|sys/{a[$1]=$2}END{printf "%.2f", a["user"]+a["sys"]}'; }
for p in 1 2 3 4 5 6; do
  o=$(one ./build/...)                 # arm A
  n=$(one env MYGATE=1 ./build/...)    # arm B
  awk -v o=$o -v n=$n 'BEGIN{printf "%s %s %+.1f%%\n", o, n, 100*(o-n)/o}'
done
```

If the 95% interval includes zero, the honest result is **"no measurable
effect"** — a perfectly good outcome for a gated path. Do not publish a single
median as a win.

**Where to run.** Do not saturate the Mac; it hosts 3-6 agent sessions and sits
at load 15-110. Anything needing more than a handful of runs goes to the **VPS**
(CPU-only, 4 cores, usually idle — the right target for CPU kernel work) or
**Kaggle** (see `../kaggle_usage.md`). Caveat: the VPS/Kaggle are x86/AVX2 and
the Mac is ARM/NEON, and `core_cpu::dot_product` has separate arms for each, so
report both rather than substituting one for the other.

**Absolute timings from the Mac are not comparable to anything.** The same
detector profile measured 2,652 ms (quiet VPS), ~2,350 ms (quiet Mac) and
12,089-21,009 ms (contended Mac). Use within-run *ratios*, which are stable.

**(a) A crash mints a fake win.** A non-zero exit or empty output must never be
timed. zsh does **not** word-split unquoted variables, so `cpu $CMD` runs nothing
and reports `0.00`, and `env "$VARS" cmd` with two assignments in one string sets
only a mangled first variable — that trap silently produced an identical-arms
"null result" during the Aug 2026 round. Always check the decoded text alongside
the timing.

**(b) Never claim a win without output equivalence.** The corpus check is 34
fixtures (20 synthetic + 14 CC0):

```bash
C=~/crispembed-live-cache; same=0; diff=0
for f in ~/crispembed-ocr-synth/*.png tests/regression/images/cc0/*.png tests/regression/images/cc0/*.jpg; do
  [ -f "$f" ] || continue
  a=$(./build/test-ppocrv6-direct $C/PP-OCRv6_small_det-f16.gguf $C/PP-OCRv6_small_rec-q8-head.gguf "$f" 2>/dev/null | grep -o 'text=.*' | tr '\n' '|')
  b=$(MYGATE=1 ./build/test-ppocrv6-direct $C/PP-OCRv6_small_det-f16.gguf $C/PP-OCRv6_small_rec-q8-head.gguf "$f" 2>/dev/null | grep -o 'text=.*' | tr '\n' '|')
  [ "$a" = "$b" ] && same=$((same+1)) || { diff=$((diff+1)); echo "DIFF $(basename $f)"; }
done; echo "identical=$same differing=$diff"
```

**(c) Recognizer confidence is NOT a quality signal.** Measured Aug 2026:
cleaned crops read as *more* confident while producing worse text (one fixture
gave 944 characters at 0.70 confidence against 835 at 0.76). Do not use
`mean_conf` as an accept gate.


#### §2 Existing instrumentation (use it before adding more)

Splitting **model load** from **compute** has already found a wasted GPU-backend
init worth 12.5x in one engine and 7–14% in another. It is the highest-yield
first move on any lane.

| env | prints |
|---|---|
| `CRISPEMBED_OCR_ORCH_BENCH=1` | per-stage orchestrator totals + accept gate |
| `CRISPEMBED_TESSERACT_BENCH=1` | `[tesseract-load-bench]` detector/recognizer load, `[tesseract-stage-bench]` detect/group/crop/recognize |
| `CRISPEMBED_PPOCRV6_BENCH=1` | `[ppocrv6-load-bench]`, `[ppocrv6-stage-bench]` |
| `CRISPEMBED_PPOCRV6_DET_BENCH=1` | `[ppocrv6-det-bench]` preprocess/graph/total, `accepted=` |
| `CRISPEMBED_EASYOCR_BENCH=1` | `[easyocr-stage-bench]` load / detect+recognize |
| `CRISPEMBED_EASYOCR_STAGE_BENCH=1` | `[easyocr-recognize-bench]` detect, and inside the region loop crop / set_width / recognize, plus `width_calls` and `width_changes` (H3) |
| `CRISPEMBED_PPOCRV6_DET_PROFILE=1` | `[ppocrv6-det-profile]` per-convolution table for the scalar detector: ms, share, calls, GF/s, shape signature, heaviest first (H2) |
| `CRISPEMBED_OCR_DETECT_BENCH=1` | DBNet resize decision (`raw WxH -> WxH`) |

#### §3 Current state — do not re-derive these

**Quality is done and we are at or ahead of every upstream.** 20-fixture
synthetic corpus, exact ground truth:

| engine | kind | CER | WER |
|---|---|--:|--:|
| `crispembed-ppocrv6` | native | **0.0031** | 0.0178 |
| `paddleocr-py` | external | 0.0185 | 0.1153 |
| `tesseract-cli` | external | 0.0256 | 0.0890 |
| `crispembed-tesseract` | native | 0.0290 | 0.1623 |
| `easyocr-py` | external | 0.0769 | 0.2363 |
| `crispembed-easyocr` | native | 0.0808 | 0.3190 |

**Speed is the open work, and the gap is the DETECTOR, not the recognizer.**
Quiet box (load 1.7-3.4), single page:

| engine | wall | parallelism | vs tesseract-cli |
|---|--:|--:|--:|
| `tesseract-cli` | 0.17 s | 0.85x | — |
| `crispembed-tesseract` | 0.70 s | 0.99x | 4.1x |
| `crispembed-ppocrv6` | ~1.9 s | 0.62x | 11x |
| `crispembed-easyocr` | ~1.5 s | — | 9x |

Four facts that should stop you re-deriving them:

1. **Both we and tesseract-cli are single-threaded** (0.99x vs 0.85x). We are not
   losing to threading; we do ~4x more work per page. Tesseract has **no neural
   detector at all** — it segments with classical projection/component analysis.
   That is the whole architectural gap.
2. **Model load is cold page cache, not code.** Detector load is 1415 ms cold and
   **7.0 ms warm**. It only dominates a one-shot CLI on a single image; on any
   multi-page workload it is paid once. H5's premise was wrong (see HISTORY).
3. **Promoting the detector ggml graph is a dead end for speed** — 2.6x slower on
   CPU, 6.8x on Metal than the scalar path (O1). Detector work is kernel work.
4. **Three CPU kernels were written and all measured NOT to be wins**
   (`CRISPEMBED_CONV1X1_FAST`, `CRISPEMBED_CONVDW_FAST`, `CRISPEMBED_DOT_WIDE`).
   They stay gated off. Do not re-attempt them without reading HISTORY first.

**Within the detector**, per-convolution profile
(`CRISPEMBED_PPOCRV6_DET_PROFILE=1`): 1x1 pointwise **51.6%**, depthwise
**20.4%**, deconv 6.4%, other 21.6%.

---

## OPEN TASKS — engine-portfolio round (2026-08-04): match/beat the reference implementation of every open-licensed lane

Scope decision: the portfolio targets are **Tesseract, DeepSeek-OCR, olmOCR,
Qwen2.5-VL, and Docling's open backends**, each held to the PP-OCRv6 standard
(reference implementation locally runnable; CER/decoded-text gates; net-of-load
timing; interleaved A/Bs; negatives recorded).

**Licensing gates (checked 2026-08-04):** Tesseract Apache-2.0; DeepSeek-OCR
MIT; olmOCR Apache-2.0 (Qwen2.5-VL-7B fine-tune); Docling components
MIT/Apache; **Qwen2.5-VL: 7B Apache-2.0 is shippable, 3B is Qwen Research
License — dev-reference only, never ship**. dots.ocr: NOT non-commercial (its
MIT-based agreement allows commercial use), but the 2026 rejection STANDS on
the supplemental terms (PRC governing law/arbitration, unilateral 90-day
amendment, use-based prohibitions, mandatory attribution) — see HISTORY;
re-admitting it is a policy decision, not a technical one. The only surviving
dots work is an uncommitted qwen2vl-variant diff in a stale worktree.

Method note for every item below: the PP-OCRv6 campaign closed five
"hard blockers" that were all small mislabeled defects. Assume the same here.
Reproduce recorded claims before building on them; diff the INPUT each lane
saw before diffing model stages; audit what every bench line actually spans.

### T11 — Reachability: every engine invocable, the document pipeline reachable from the CLI

Six enum engines have no CLI name (`deepseek_ocr2`, `tesseract_fraktur`,
`parseq`, `pix2struct`, `granite_vision`, `unified` — `examples/cli/main.cpp`
`eng_id` map vs `src/ocr_orchestrator.h:41-59`), and `--ocr-pipeline` can
never set `layout_model`/`table_model`/`formula_model`/`route_*`, so the
existing layout→table→formula→markdown assembly is C-ABI/server-only. This
exact bug class hid ppocrv6 for months (no `map_engine` id, no CLI name).
**Do:** name every engine; add `--ocr-layout/--ocr-table/--ocr-formula` (or
one `--ocr-document` preset) to the pipeline path; extend
`tests/test_ocr_backend_matrix.py` to assert enum↔CLI-name coverage so the
class cannot recur. **Acceptance:** each engine runs by name on a fixture;
the receipt produces markdown with a table via CLI alone; the matrix smoke
fails if a future engine ships nameless.

### T11 status [DONE 2026-08-04]: all 18 engines CLI-reachable; document pipeline + markdown via CLI; `tests/test_cli_engine_names.py` guards enum↔name coverage. Found while validating (pre-existing, unowned): feeding a Tesseract GGUF into the FLAT pipeline's rec slot mis-dispatches as `math_ocr` (vocab=1200) and SEGFAULTS on region 1 — the flat rec loader needs an arch check that fails loudly instead.

### T12 — External-parity arms + bench-span audit for all six families

**Span-audit half DONE 2026-08-04:** all VLM/OCR lanes
(qwen2vl/got/glm/internvl2/lightonocr/unlimited/granite/smoldocling/parseq/
pix2struct/layout) time per-stage from their own stage start — net-of-load by
construction; the two known bug patterns (ppocrv6 stage-entry span, easyocr
regex on load-inclusive total) do not recur. One real gap:
**`deepseek_ocr2` emits NO bench line at all** — its recorded timings are
PLAN prose. T14 must add a `[deepseek-ocr2-stage-bench]` following the
ppocrv6 net-of-load convention before any perf work.

`tests/ocr_external_parity.py` covers tesseract/easyocr/paddle only. **Do:**
add reference arms — pip Docling (full document parse), HF-transformers
Qwen2.5-VL-7B (dev-only 3B allowed for debugging), olmOCR's own toolkit,
HF DeepSeek-OCR — plus document-level ground truth (olmOCR's bench data can
seed it; extend the T1 conventions). Audit every lane's stage-bench line for
the load-inclusion bug found twice already (ppocrv6 detect spanned stage
entry; easyocr harness regex captured load-inclusive total). **Acceptance:**
one table, per family: native CER/WER + net-of-load engine_ms vs its
reference on shared fixtures; every stage-bench span documented as
net-of-load or split like `[easyocr-stage-bench]`.

### Engine-portfolio round — self-contained agent briefs (delegable as written)

Each brief is executable by a fresh agent without this session's context.
**Shared rules for every brief:** read `../crispasr-crispembed-dev.md` HARD
RULES first. One heavy process at a time (16 GB shared Mac); new Python envs
under `~/venvs/<name>` — NEVER touch miniconda's pinned paddleocr 2.10.0 (it
is a recorded baseline arm). Timing claims need a same-window control
(`tesseract-cli` on the same fixture) or CPU-seconds; `proc_ms` and
`engine_ms` are never comparable to each other. Record negative results in
PLAN with the measured numbers. Claim a row in the active-work table before
starting; push PLAN to main at checkpoints. Describe third-party pipelines by
concept, not project name, in code comments. Harness conventions live in
`tests/ocr_external_parity.py` (adapter = an `Engine` subclass; see
`PaddleOCRPy` for the in-process pattern) and ground truth in
`tests/regression/images/cc0/ground_truth.json` (records[]={file,text};
as-printed hyphenation, column reading order, provenance + per-fixture
confidence).

**A1 — parity arm: pip document-parser reference (Docling).** `python -m venv
~/venvs/docling && pip install docling`. Add an `Engine` subclass that runs
the full document parse in-process (warm ⇒ `proc_ms == engine_ms`), extracts
plain text for CER and keeps the markdown for later structure gates. Run on
`~/crispembed-ocr-synth` and the labelled CC0 dir; **acceptance:** a harness
row (CER/WER/engine_ms) for both corpora in a JSON artifact + PLAN table.
Trap: its OCR backend choice matters — record which backend it selected.

**A2 — parity arm: transformers Qwen2.5-VL.** venv `~/venvs/qwenvl`
(torch + transformers, MPS). Reference = **7B** (Apache); the 3B may be used
for local smoke ONLY (Qwen Research License — never publish 3B numbers as
the reference). Prompt must match our lane's transcription prompt (see
`src/qwen2vl_ocr.cpp` prompt constants). Pages may take minutes; if >10
min/page, run a documented fixture subset. **Acceptance:** harness row +
saved per-fixture transcripts (they become the gold for our lane's CER gate).

**A3 — parity arm + gold: olmOCR toolkit.** venv `~/venvs/olmocr`
(`pip install olmocr`). It consumes PDFs: wrap the image fixtures into
single-page PDFs (img2pdf) and record that recipe. Save (a) its transcripts
as document-level gold, (b) the exact anchored prompts it builds per page —
those are the T13 prompt contract. **Acceptance:** harness row + a
`tests/regression/gold/olmocr/` gold set + the prompt-contract notes in PLAN.

**A4 — parity arm + gold: HF DeepSeek-OCR reference.** venv
`~/venvs/deepseekocr` (transformers per the model card — read the card's
exact `infer()` call, do not guess prompts). 16 GB caution: if the reference
OOMs locally, run it on Kaggle per `../kaggle_usage.md` and bring back
transcripts only. **Acceptance:** harness row + saved reference transcripts
for the fixtures — these are the CER gate T14 requires before its perf work.

**A5 — document-level ground truth.** Extend the T1 conventions to document
STRUCTURE gold (reading order, tables-as-HTML, headers) for 3-5 pages,
seeded from A1/A3 outputs and human-verified. **Acceptance:** a gold file
with per-page provenance + a scoring note (order-sensitive CER stays; add a
structure comparison usable by T16).

**A6 — docs hygiene sweep.** Re-verify every row of
`docs/ocr_backend_matrix.md` against the code's actual gates (known stale:
PP-FormulaNet-L is GPU-capable; batch-graph and width-bucketing are now
defaults; detector graph rows). Re-date or delete impossibility claims per
the LEARNINGS rule ("re-date your impossibility claims"). **Acceptance:**
`tests/test_ocr_backend_matrix.py` and `tests/test_cli_engine_names.py` pass
and every matrix claim cites a gate or a dated measurement.

**A1 status [DONE 2026-08-04, merged `7b4d8fa7`]:** `DoclingPy` arm +
order-blind `wer_unordered` metric in `tests/ocr_external_parity.py`; artifact
`tests/results/ocr_parity_docling_2026-08-04.json` (versions pinned, repro
lines). Numbers (repeats=3, warm ⇒ proc_ms==engine_ms; `tesseract-cli:eng`
same-window control, its proc_ms is load-INCLUSIVE):

| corpus | n | arm | CER | WER | WER-unord | med engine_ms |
|---|--:|---|--:|--:|--:|--:|
| synth | 20 | docling-py | 0.11269 | 0.14367 | **0.02483** | 1580 |
| synth | 20 | tesseract-cli | 0.02561 | 0.08900 | 0.10027 | (proc 133) |
| cc0-labelled | 5 | docling-py | 0.44934 | 0.51996 | 0.50708 | 6581 |
| cc0-labelled | 5 | tesseract-cli | 0.54721 | 0.73225 | 0.58255 | (proc 272) |

Findings that gate how this arm may be quoted: (a) **backend trap confirmed**
— auto-selection picked **RapidOCR/torch with the default `lang=["chinese"]`
PP-OCRv6-small models** (ocrmac/onnxruntime/easyocr absent in the venv), so
this arm is near-same-model as our `ppocrv6` lane, NOT an independent
recognizer; differences measure pipeline/layout, and installing ocrmac or
onnxruntime silently changes the arm. (b) ~83% of its synth word error is
READING ORDER (4/20 fixtures glyph-perfect with swapped lines); unordered its
recognition beats the tesseract control — both metrics now reported. (c) The
layout stage can silently discard OCR'd text: `simple_form.png` exports 0
chars (CER 1.0) as two PICTURE clusters while recognized items score 0.1862;
`force_full_page_ocr` does not help; adapter records both views. (d)
`receipt_historical.png` total failure: CER 0.9974 at 19.7 s (slowest).
(e) 12-24x slower than the load-inclusive tesseract control. (f) Environment:
`~/.cache/huggingface` symlinks to the backups volume which is 100% FULL —
every HF download needs `HF_HOME=$HOME/.cache/hf-docling` until cleared.

**A4 status [DONE 2026-08-04, merged]:** BOTH checkpoints run (the brief
named v1 but T14's engine is `deepseek_ocr2` — a v1-only gate would not have
been a gate): gold for each under `tests/regression/gold/deepseek-ocr{,2}/`
(free_ocr + grounding raw/.mmd views, contract.json captured off the
checkpoints' OWN infer()/generate). Determinism verified byte-identical on
re-run; scoring reproduced from the committed tree at merge. Numbers
(free_ocr, mean [markup-stripped]): v2 synth **0.00199**, cc0 0.18743
[**0.11063**]; v1 synth 0.00193, cc0 0.54041 [0.11132] — v1 answers plain-OCR
with HTML tables on receipts (2.04 raw CER on receipt_historical). Document
mode after the checkpoint's own post-processing: v2 synth **0.00044** (beats
plain mode). Contract deltas that matter for T14: crop threshold differs
v1=640 vs v2=768 (different image-token counts on 13/25 fixtures); eager
attention is FORCED on sm_75 (checkpoint offers only eager/FA2, no sdpa);
**bf16 WORKS on T4** (probed: matmul/conv2d/SDPA — corrects the assumption
A2/A3 ran under); transformers must be ≤4.46.x (remote code loses .generate()
on 4.50+); the transformers path lacks the vLLM recipe's n-gram
logits-processor guard → grounding mode can spiral on form pages (v1: 2073
tokens of repeating box list on simple_form; v2 degrades 4x less). **T14 is
now UNBLOCKED**: its CER gate is the deepseek-ocr2 gold set.

**A3 status [DONE 2026-08-04, merged]:** 25-page raw+parsed gold in
`tests/regression/gold/olmocr/` + `contract.json` (captured by IMPORTING the
toolkit's own `build_page_query`, olmocr 0.4.27). Contract vs our T13 lane:
prompt **byte-identical** (sha c12f21ac, 557 B — re-verified independently at
merge), text-before-image CONFIRMED, default system message CONFIRMED, 1288
render CONFIRMED (small pages UPSCALED; pdftoppm rounds 2 fixtures to 1289;
grids match native on 24/25 — synth_02_lowdpi lands across a round-to-28
boundary). **Divergences that shape the T13 gate:** (1) the toolkit's FIRST
attempt samples at temperature **0.1**, not 0.0 (differs from greedy on 3/5
cc0 pages, CER 0.001-0.006); (2) the reference disagrees with ITSELF across
serving stacks (vLLM emitted an early stop on simple_form: CER 0.372 vs
transformers 0.049 — gold ships from transformers; both kept); (3) reference
front matter must start at byte 0 and validates strictly (invalid → retry);
our lane is deliberately more permissive. ⇒ T13 acceptance is a CER
threshold vs gold, NEVER a byte diff. Harness: synth 20/20 CER 0.0; cc0 mean
CER 0.17735 raw / **0.04057 markup-stripped** (the 0.70 receipt outlier is
the prompt returning an HTML table, as instructed; stripping HURTS
receipt_historical, so both views are quoted). No retries on any of 25
pages. vLLM-on-T4 works after two fixes (libcuda.so symlink; leave
VLLM_ATTENTION_BACKEND unset — forcing it breaks the vision tower), ~2x
faster than transformers serving.

**A2 status [DONE 2026-08-04, merged `add33c26`]:** `Qwen25VLPy` adapter +
Kaggle kernel (`chr1s4/crispembed-qwen-vl-ocr-parity`, 2x T4); **25/25 gold**
transcripts in `tests/regression/gold/qwen2.5-vl-7b/` with full manifest
(model rev `cc59489`, our lane's exact prompt/template, greedy,
dtype=float16 — T4 cast, bf16 spot-check on Ampere+ is an open follow-up;
`sdpa` verified output-identical to eager on 3 fixtures). Numbers: synth
20/20 CER 0.0 (corpus does not discriminate); CC0 mean CER **0.02902** vs
tesseract 0.46114 — receipt_historical 0.0195 (vs 0.453/0.997 for
tesseract/docling), simple_form 0.109 (weakest gold page — eyeballed, real
form-widget text). Local 7B was abandoned by arithmetic (15.45 GiB weights vs
16 GiB unified). OOM ledger: ten measured device-split attempts recorded in
the artifact; the two ~4.8 Mpix pages only pass with heavy host offload
(`0=1GiB,1=3GiB,cpu=60GiB`, 2512/1414 s) because sm_75 has no
memory-efficient SDPA for that mask shape — so **no timing column in this
artifact is latency** (a resident `latency_reference` config is recorded:
2389.7 ms/page, cannot read the two largest pages). 3B used for smoke only,
no numbers published (license gate).

**A6 status [DONE 2026-08-04, merged]:** matrix re-verified row by row; every
claim now cites a gate/code line or a dated measurement, `UNVERIFIED as of
2026-08-04` labels where neither exists (PARSeq, SMT-family, Unlimited-OCR,
PP-FormulaNet-L residency; Surya timing; CUDA never smoked here).
PP-FormulaNet-L split out as GPU-capable; batch-graph/width-bucketing/detector
graph defaults corrected; `docs/ppocrv6.md` rewritten (it documented gates that
no longer exist). Surprises: DBNet detection is deliberately CPU-default (GPU
measured 14x WORSE, issue #25); `GRAPH_ACCEPT` gates only the single-crop lane.
Known leftover for the next ppocrv6 toucher: `src/ppocrv6_det.cpp:935-940`
comment still describes the retired `DET_GRAPH` gate.

*(T13-T17 below are NOT agent-delegable as-is: they are port/bisect work —
run them as dedicated sessions with the full board context.)*

### T18 — Embedder one-shot fixed init (~1.2-1.4 s) dominates CLI latency; warm compute already beats onnxruntime

Measured 2026-08-04, M1 16GB, same-window A/B, 64 German sentences (~12 words,
padded len 30), multilingual-e5-small q8_0 vs the official fp32 ONNX export on
onnxruntime 1.25.1 CPU EP (tokenizers batch, mean-pool+L2, warm):

| config | load/init | warm per-text (batch 64) | single-text warm |
|---|--:|--:|--:|
| crispembed q8_0 (Metal, one-shot CLI) | **~1.2-1.4 s** | 5.7-11.7 ms (marginal) | n/a (one-shot pays init) |
| onnxruntime fp32 CPU | 0.44 s session | 12.1-14.4 ms | 13.6 ms |

Output parity q8 vs ONNX fp32: cosine min 0.99993 / mean 0.99995 (n=64).
So the "ONNX is much faster" experience is NOT compute — warm-vs-warm we are
~1.4x ahead — it is the **fixed one-shot init**: ~1.2-1.4 s regardless of
model size (132 MB e5 and 23 MB MiniLM both pay it → not weight I/O), and
`--gpu-backend cpu` still initializes the Metal device (stderr shows the
pipeline-cache load either way), so the flag does not skip the cost.
**Do:** (a) make `--gpu-backend cpu` actually skip GPU device init for the
embed path; (b) profile the remaining fixed cost (SPM tokenizer build for the
250k XLM-R vocab is a suspect) and lazy-init what one-shot embedding does not
need; (c) consider a CPU-default for small embedders in one-shot CLI mode
(T5 precedent: workload-dependent backend). Server mode already amortizes —
this is a CLI/scripting-latency item. **Acceptance:** one-shot
`crispembed -m multilingual-e5-small --json "text"` total time down ≥3x with
embeddings byte-identical (or cosine ≥0.9999) to today's, and no regression
in warm batch throughput.

### T18 status [DONE 2026-08-05, `feat/t18-embed-oneshot-init`, NOT merged]: 4.8x one-shot, byte-identical output, and the cost was NOT what the ticket assumed

**Headline: 895 ms → 186 ms (4.81x) one-shot on multilingual-e5-small q8_0,
STILL ON METAL, output byte-identical.** The ~1.3 s the ticket recorded
reproduced as 0.89-0.91 s on a quiet box (same shape, lower absolute — the
earlier figure was presumably measured under load); the *structure* of the
claim was right and the *suspect* was wrong.

**Per-component init profile** (`CRISPEMBED_INIT_BENCH=1`, the instrument this
branch adds — M1 16 GB, multilingual-e5-small q8_0, medians):

| component | before | after | note |
|---|--:|--:|---|
| `crispembed_init/arch_detect_gguf_open` | 29.3 ms | 29.3 ms | GGUF metadata parse (250k-token vocab KV) |
| `load_model/gguf_init_from_file` | 29.7 ms | **0.0 ms** | was a SECOND parse of the same file — now reuses the first |
| `load_model/vocab_read` | 6.0 ms | 6.0 ms | 250k strings out of the KV array |
| `load_model/tokenizer_build` | 12.0 ms | 12.0 ms | **the recorded SPM suspect — 12 ms, not the problem** |
| `load_model/backend_init` | **683.1 ms** | **29.4 ms** | Metal device + pipeline cache |
| `load_model/sched+meta` | 0.8 ms | 0.5 ms | |
| `load_model/weights_load` | 46.9 ms | 46.4 ms | |
| first `crispembed_encode` | 21.0 ms | ~17-20 ms | includes Metal PSO JIT |
| **process wall** | **895 ms** | **186 ms** | |

**The real cause: ggml-metal's persistent `MTLBinaryArchive` pipeline cache.**
ggml carries a CrispASR patch (PLAN #88) that opens
`~/Library/Caches/ggml-metal/<device>.archive` before any PSO is created. That
archive is append-only across every engine and every Crisp binary that ever ran
on the box; on this machine it had reached **683 MB**, and opening it costs
~1 ms/MB — 683 of the 820 ms of internal init. Two things make it strictly a
loss for a one-shot CLI:

1. **It buys nothing measurable.** First encode was 20.3 ms with the archive
   open and 17.4 ms with it skipped — marginally *worse* with it. macOS keeps
   its own system-level shader cache underneath, which is what actually makes
   the second run fast.
2. **A one-shot CrispEmbed binary can never repay it.** The archive is
   serialised back only from `ggml_metal_device_free()`, which runs at
   static-destructor time — and the one-shot CLIs leave via
   `core_util::clean_exit` → `_exit()`, which skips it (the known
   clean_exit-bypasses-atexit hazard, striking somewhere new). Proven directly:
   pointed at an empty `GGML_METAL_PIPELINE_CACHE` dir the run logs "pipeline
   cache created" and exits leaving the directory **empty**. So the CLI pays the
   open and never writes an entry — read-only cost, forever.

**Levers applied, in measured order** (each independently gated; the gates ARE
the A/B mechanism — one binary, both arms):

| # | Lever | Gate to restore old behaviour | Measured delta (e5-small one-shot) |
|---|---|---|--:|
| 1 | Skip a Metal pipeline-cache archive larger than a cap (default 64 MB), decided by `stat` before the device exists (`src/core/metal_pipeline_cache_policy.h`) | `CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB=0` | **−654 ms** |
| 2 | `--gpu-backend cpu` genuinely returns the CPU backend instead of falling through to `ggml_backend_init_best()` (`src/core/gpu_backend_pref.h`) | `CRISPEMBED_GPU_PREF_CPU_LEGACY=1` | 0.86 s → 0.14 s **on that flag** (6.1x); no effect on the default path |
| 3 | Reuse `crispembed_init()`'s GGUF parse in `load_model()` / the decoder tokenizer load instead of parsing the file a second time | `CRISPEMBED_GGUF_REPARSE=1` | **−29 ms** |
| 4 | `CRISPEMBED_ONESHOT_CPU=1` picks CPU when no `--gpu-backend` was given (CLI only) | off by default | −40 ms, opt-in — see recommendation |

Lever 2 has a sharp edge worth remembering: the obvious implementation,
`ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)`, **still initialises
Metal** because enumerating the registry constructs every device. It measured
29 ms of Metal init on a "cpu" request. `ggml_backend_cpu_init()` touches no
registry and is the correct call.

**Acceptance** (interleaved same-binary A/B, medians, `sysctl vm.loadavg` first
value gate >8 — 0 pairs discarded, load stayed 1.8-2.5 throughout):

| case | before | after | speedup | output |
|---|--:|--:|--:|---|
| e5-small one-shot `--json "ein test"` (n=7 pairs) | 895 ms (892-901) | **186 ms** (184-187) | **4.81x** | byte-identical |
| arctic-embed-m-v2 q8_0 one-shot (n=5) | 911 ms (908-916) | **202 ms** (200-264) | **4.51x** | byte-identical |
| e5-small warm batch-512 (n=5) | 5977 ms | 5451 ms | 1.10x (no regression) | byte-identical, 64/64 |
| arctic warm batch-64 (n=5) | 1672 ms | 908 ms | 1.84x | byte-identical, 64/64 |

Output identity was checked on the actual vectors, not a summary: 64 texts per
model, `worst cos = 1.000000000`, `|before| = |after| = 1.000000`, ratio
1.000000000, and the JSON is byte-for-byte equal. The math path is untouched —
every change is in init.

**Negative / refuted results, on the record:**
- **The SPM tokenizer suspect is refuted.** Building the 250k-entry XLM-R
  SentencePiece tokenizer is 12.0 ms and reading the vocab out of the GGUF is
  6.0 ms — together 2% of the old fixed cost. No quadratic construction, no
  disk cache needed. Do not spend time here.
- **Weight I/O was never the story either**, confirming the ticket: e5-small
  (132 MB) and arctic (330 MB) both paid the same ~683 ms Metal init.
- **CPU-default for small embedders is now a much weaker lever than it looked.**
  Before the fix it would have saved ~700 ms; after it, 40 ms.
- `ggml_backend_dev_by_type(...CPU)` as a "cheap CPU" path: measured worse than
  useless (see above), kept out.

**CPU-default recommendation — data for the coordinator, decision NOT taken
here.** Post-fix sweep, batch-64, times include that arm's own init:

| model | Metal `-t 1` | CPU `-t 1` | CPU `-t 4` |
|---|--:|--:|--:|
| multilingual-e5-small q8_0 | 0.77 s | 0.76 s | **0.35 s** |
| arctic-embed-m-v2 q8_0 | **0.91 s** | 2.77 s | 0.91 s |

One-shot single text post-fix: e5-small 0.18 s Metal vs 0.14 s CPU; arctic
0.20 s vs 0.17 s. So: **the backend default is no longer the interesting knob —
the `-t 1` default is.** CPU with 4 threads beats Metal by 2.2x on the small
embedder and ties on the large one, while CPU at the shipped `-t 1` is 3x
*worse* than Metal on the large one. A blanket "small embedders default to CPU"
switch would be defensible on e5-small and wrong on arctic-at-`-t 1`;
`CRISPEMBED_ONESHOT_CPU=1` ships gated off so the flip can be made with a
size/thread rule rather than a guess. Suggested follow-up before any flip:
measure a thread-count default for the embed CLI, which looks like the larger
untaken win.

**Found, not fixed:**
1. **Every Metal lane in the repo pays this same archive-open cost**, not just
   the embedder — the OCR/VLM/SR engines all call `crispasr_init_gpu_backend()`.
   The policy header is deliberately standalone; adopting it elsewhere is a
   one-line `core_metal_cache::apply()` before the backend init. Only the embed
   path is measured and changed on this branch.
2. **The pipeline cache is arguably broken repo-wide**, not merely oversized: no
   one-shot Crisp binary that exits via `clean_exit`/`_exit()` can write to it,
   so it can only be filled by long-running or normally-unwinding processes
   while every short process pays to read it. Whether the patch should flush at
   the end of a run, scope the archive per engine, or be retired is CrispASR
   PLAN #88's call, not this branch's.
3. The 683 MB archive is still on disk (this branch only stops *reading* it);
   deleting it is safe and reclaims the space.
4. `-t` defaults to 1 for the embed CLI (see recommendation above).

**Env gates added:** `CRISPEMBED_INIT_BENCH`,
`CRISPEMBED_METAL_PIPELINE_CACHE_MAX_MB`, `CRISPEMBED_GGUF_REPARSE`,
`CRISPEMBED_GPU_PREF_CPU_LEGACY`, `CRISPEMBED_ONESHOT_CPU` — all documented in
README "One-shot CLI startup" and in the headers themselves. Model-free CI
battery re-run green (backend-smoke auto/metal/cpu, provenance x3, msac,
temp-file, qwen 39, o200k 85, bpe 246 checks). `test-backend-smoke cpu` now
reports `name=CPU type=0` where it used to report `MTL0` — the same
fall-through, visible in a test that had been passing over it.

**T19-E1 status [DONE 2026-08-04, merged]:** F2LLM-v2 **80m/160m/330m
shipped** (cstr, f16+q8_0, registry+pins; 0.6B was already shipped and needed
nothing). Converter docstring claim was REAL — worked as-is. Contract: last-token
pool, L2, query prompt "Instruct: Given a question, retrieve passages that can
help answer the question.\nQuery: ", no doc prefix, EOS <|im_end|> (NOT
Qwen3-Embedding's <|endoftext|>). Parity: f16 cos 1.000000 all sizes; q8_0
≥0.9989 except 0.6B 0.9909 (+3.8% norms — known-soft, consistent with
LEARNINGS). German retrieval 5/5 everywhere; independently re-verified at
merge (registry download, 0.611>0.147>-0.030). **The real find — a shipped
tokenizer bug:** `core_bpe::tokenize_simple` collapsed all whitespace runs to
single spaces (newlines deleted) — cos 0.9803 on code, 0.9907 on this
family's OWN query prompt, hidden on newline-free text; proven by reproducing
our magnitudes in HF with collapsed input. Degraded the already-shipped 0.6B
and by construction every Qwen-family embedder (qwen3-embed, octen, jina-v5,
harrier). Fixed via a real declared-regex `qwen_pretokenize`
(`CRISPEMBED_BPE_LEGACY_WHITESPACE=1` restores), guarded by a hermetic
39-check test in model-free CI (verified fail-on-broken). **Follow-up filed:**
other `tokenize_simple` callers (lfm2, OCR engines) likely share the defect —
audit them. Phantom-bug note preserved in the commit: a 0.845/0.756 "port
bug" on 330m/80m was the agent's own harness double-applying the new auto
query prefix — weights were always correct.

**T19-E1-FOLLOWUP status [DONE 2026-08-04, `feat/tokenize-simple-audit`, NOT
merged]:** audit of the remaining `core_bpe::tokenize_simple` callers. Complete
inventory (`grep -rn tokenize_simple src/ examples/ tests/`) is FOUR sites in
three files; all four are converted, each keeping
`CRISPEMBED_BPE_LEGACY_WHITESPACE=1` as the restore gate.

| Caller | Checkpoint | Declared `pre_tokenizer` | Battery BEFORE | Real-input exposure | Fixed | AFTER |
|---|---|---|---|---|---|---|
| `src/lfm2_embed.cpp:362` | `LiquidAI/LFM2.5-Embedding-350M` | Split + ByteLevel, `…\|\p{N}{1,3}\|…` (Qwen regex, 3-digit runs) | 14/40 cases wrong | **LIVE** — arbitrary user text; wrong ids on 951/1508 random strings (63.1%) | yes → `tokenize_lfm2` | 0/40; ids 0/1508 wrong; embedding cos vs HF 0.9857 → **0.9997** |
| `src/deepseek_ocr2.cpp:2417` | `deepseek-ai/DeepSeek-OCR-2` | Split SEQUENCE: `\p{N}{1,3}`, CJK/kana runs, then a `[\p{P}\p{S}]`-based regex | 15/40 cases wrong | **LIVE** — the fixed `"\nFree OCR."` prompt, every page | yes → `tokenize_deepseek` | 0/40; prompt ids now byte-exact vs HF |
| `src/deepseek_ocr2.cpp:2304` | same | same | same | latent — inside `getenv("DS_TEXT_TEST")` | yes | same |
| `src/unlimited_ocr.cpp:2877` | `baidu/Unlimited-OCR` | byte-identical to DeepSeek-OCR-2's | same | latent — inside `getenv("UOCR_TEXT_TEST")`; the production prompt is hardcoded ids | yes | same |

**Headline (deepseek-ocr2, the requested deliverable): the prompt ids did NOT
match the reference contract, and now do.** `tests/regression/gold/deepseek-ocr2/contract.json`
gives `free_ocr = "<image>\nFree OCR. "`, which `format_messages()` strips to
`"<image>\nFree OCR."`; the reference tokenizer encodes that as
`[128815, 201, 21431, 126041, 16]` — `<image>` (matching the contract's
`image_token_id`) followed by `Ċ Free ĠOCR .`. `tokenize_simple` DELETED the
leading newline and emitted **3 ids where the reference emits 4**:
`[21431, 126041, 16]`, i.e. every page ran with token `201` missing from the
instruction. `tokenize_deepseek("\nFree OCR.")` now returns
`[201, 21431, 126041, 16]`, byte-identical to HF. The GGUF's vocab/merges come
verbatim from the same `tokenizer.json`
(`models/convert-deepseek-ocr2-to-gguf.py:170`), so the check transfers.
Per the split of work the full decode gate is the coordinator's after both
branches land — the 5.3 GB model was NOT run here.

**Two further real bugs the audit turned up, both pre-existing:**

1. **The merged E1 fix was itself still wrong on non-ASCII punctuation.**
   `qwen_is_letter` answered true for every byte >= 0x80, so `\p{L}` swallowed
   quotes, dashes, currency and emoji into the neighbouring word. HF splits
   `sagte „Hallo“ heute` into 5 pre-tokens; we produced 3. `«quote»`, `€£abc`,
   `→→x`, `a©®b`, `中文，测试。` all wrong — i.e. ordinary German typographic
   text on EVERY Qwen-family embedder, which is the German-retrieval workload
   T19 exists for. Fixed by classifying codepoints against real Unicode general
   categories (`src/core/unicode_class.h`, generated, 774 ranges): 9 of 40
   qwen battery cases were failing, now 0.
2. **`bpe_one`'s merge heap had no tie-break.** HuggingFace orders its BPE heap
   by `(rank, pos)` both ascending; `std::priority_queue` with a rank-only
   comparator leaves equal ranks in an unspecified order, so a run of
   equal-rank pairs could merge from the middle: `"qqqqqc"` gave `qq q qq c`
   instead of `qq qq q c`. Cost 4 of 1508 random strings on EVERY vocab tried
   (qwen, lfm2, deepseek). One-line comparator fix; affects all byte-level BPE
   callers, not just the audited ones.

**Verification.** Guard written before the fix and watched fail (HARD RULE 2c):
`tests/test_bpe_pretokenize.cpp`, 246 hermetic checks, model-free CI alongside
`test-qwen-pretokenize`. Pre-fix it reported **38 pre-tokenizer failures**
(qwen 9 / lfm2 14 / deepseek 15) plus **2 tie-break failures**; post-fix 0.
Golden splits are HuggingFace's own `pre_tokenize_str()` output, regenerated by
`tools/gen_bpe_pretokenize_test.py`. Beyond the fixture: 4000 random
mixed-script strings per family pre-tokenize identically to HF (0 mismatches),
and with the real vocab+merges loaded, 1508 strings tokenize to **identical ids**
for all three (0 mismatches, against 63% wrong under `tokenize_simple`).
`test-qwen-pretokenize` (the E1 guard) stays at 39/39.

**Acceptance.** lfm2: newline-heavy German text, q8_0, CLS pooling per the
model's `1_Pooling/config.json`, reference = the repo's own
`Lfm2BidirectionalModel` via `trust_remote_code` (the plain causal `AutoModel`
is the WRONG reference and scores 0.09 — worth knowing before anyone re-runs
this). cos vs HF 0.985685 → **0.999686**; the residue is q8_0. Caveat per HARD
RULE 2b: the CLI L2-normalizes, so `|mine|` is 1.0 by construction and this
number is scale-blind — magnitude parity is the `test-lfm2-diff` harness's job
and was not re-run. Control: newline-free ASCII text is **byte-identical**
before/after, confirming why the defect stayed invisible. unlimited_ocr:
decoded output provably unchanged — the converted call is inside the
`UOCR_TEXT_TEST` debug block and the production prompt is the hardcoded
`{34030, 76466, 16}`, which this audit independently re-verified equals HF's
`document parsing.`; the 3.3 GB model was not run.

**Not fixed / known approximations.** `\p{N}` for the CJK-adjacent scripts is
exact via the table, but codepoints absent from it default to letter (correct
for every script tried); `\s` is Unicode White_Space; the deepseek stage-3
alternative 1 is ASCII-only as declared. Everything measurable is covered by
the fuzz above. Also unaudited: `tokenize_simple` itself is left in place and
still exported — it is now only reachable through the legacy gate.

**T19-E2 status [DONE 2026-08-04, merged]:** `arctic-embed-m-v2` shipped
(f32/q8_0/q4_k on cstr, registry+pins, q8_0 default — q4_k without imatrix is
weak here: cos_min 0.954, imatrix TODO). Per-stage parity cos 1.000000 after
TWO REAL pre-existing bugs the port exposed: (1) **the fused gated-FFN branch
never applied `ffn.fc2.bias`** — invisible on ModernBERT (no bias) but live in
every shipped GTE v1.5 GGUF; the tensors are IN the published files, so
`gte-base/large-en-v1.5` are repaired in place (shipped q8_0 cos vs HF
0.985→0.9996), no re-upload needed; (2) gated-FFN activation was guessed
per-arch (tanh) where HF uses exact-erf — now self-describing via
`bert.ffn_act` (absent key = historical behavior, published GGUFs
byte-identical). Also: `query_prefix()` had NO arctic rule — the shipped
`arctic-embed-l-v2`/v1 models were running UNPREFIXED; wired for both
generations. German retrieval sanity 5/5 top-1 at f16/q8/q4; independently
re-verified end-to-end at merge (registry download + auto-prefix + erf
kernel; ECB 0.652 > Rhein 0.307 > Kartoffelsalat 0.055).
**granite-r2 gap report (backbone PROVEN via token-id bypass, per-stage cos
0.99994+):** blocked ONLY on tokenizers — (a) `is_sentencepiece` misfires on
BPE vocabs >100k in BOTH converter (`convert-bert-to-gguf.py:418`) and
runtime (`crispembed.cpp:546/576`) → fix = read tokenizer.json `model.type`;
(b) 97m needs an o200k-style regex pre-tokenizer (~1 function beside
`gpt2_pretokenize`); (c) 311m needs the existing SPM-BPE mode wired
(embedder path hardcodes `spm_style=false`). Small, well-scoped follow-up.
**New open item (repo-wide, pre-existing):** `--biencoder` applies the QUERY
prefix to documents too (`examples/cli/main.cpp:2461`, context-level prefix)
— affects bge/e5/nomic/lfm2/arctic; cost ~0.03-0.07 cosine, no rank flips
measured, but needs its own A/B before changing (silently alters output).

**T19-E3 status [branch `feat/imatrix-quants`, 2026-08-04]:** imatrix quants for
`arctic-embed-m-v2` + the F2LLM-v2 family. Support already existed end to end
(`src/imatrix.{h,cpp}` collector gated on `CRISPEMBED_IMATRIX_OUT`, installed on
the sched in `crispembed.cpp:627/2341` and flushed from `crispembed_free`;
`tools/quantize.cpp --imatrix`; `tools/kaggle/crispembed-imatrix-quant/`;
`tools/imatrix_ab.py`) — nothing new was built. Three defects were found in it.

**(1) The calibration corpus never shipped.** A Kaggle *script* kernel carries
only its `code_file` (usage #26), so `read_corpus`'s `Path(__file__).parent`
lookup always missed and every imatrix quant to date silently calibrated on the
10-sentence English `_CALIB_FB` fallback — recorded as `calib=10` in the
uploaded `*-imatrix-ab.txt`. Corpora now load from the CLONE and a miss raises.

**(2) imatrix covers only 36 of arctic's 73 quantized tensors.**
`src/crispembed.cpp:799-832` pre-merges q/k/v into one F32 `L.qkv_w` at load
time and never `ggml_set_name`s it, so the collector files that matmul's
statistics under ggml's auto name `leaf_N`, which matches nothing at quantize
time — every `enc.N.attn.{q,k,v}.weight` is quantized with NO importance
(quantizer prints `36 with imatrix`, vs f2llm-80m's `56` of 57 and 0.6b's `196`
of 197; the decoder path does not pre-merge). The collected `leaf_N` vector is
width 768 = the QKV input, i.e. already the correct importance vector for all
three — so the fix is naming + a quantizer alias, not new infrastructure.
**Affects every BERT-family imatrix quant shipped** (bge / e5 / MiniLM / mpnet /
gte / arctic). TODO, not done here (runtime graph code).

**(3) imatrix is a NO-OP for q8_0.** Every f2llm q8_0-vs-q8_0+imatrix pair came
back identical to 6 dp (0.999684 / 0.999555 / 0.999161 / 0.992944) —
`ggml_quantize_chunk` ignores the importance vector for Q8_0. So the "soft
0.6B q8_0" cannot be improved this way; that lane is closed.

A/B: cosine vs the full-precision GGUF over 65 held-out texts (43 doc + 22
through the model's own query prompt; German + English + code + newline-heavy),
calibrated on 134 disjoint texts. Kaggle CPU arms, cross-checked locally on
Metal (arctic mean 0.9584 vs 0.9607 local — backend FP delta only):

| model | q8_0 min/mean | q4_k min/mean | q4_k+imat min/mean | verdict |
|---|---|---|---|---|
| arctic-embed-m-v2 | .9994/.9996 | .9466/.9584 | .9480/.9614 | better, still weak |
| f2llm-v2-80m | .9992/.9997 | .9499/.9727 | .9455/.9767 | mean better, **min worse** |
| f2llm-v2-160m | .9993/.9996 | .9331/.9652 | .9495/.9719 | better |
| f2llm-v2-330m | .9986/.9992 | .8840/.9230 | .9179/.9501 | clearly better |
| f2llm-v2-0.6b (local) | .9964/.9975 | .6044/.6911 | .7821/.8238 | far better, still unusable |

Norm ratio is 1.0000 for every arm on every text: the pooled-embedding API
L2-normalizes, so the "+3.8 % norm inflation" noted in E1 is not observable (or
consequential) through it — that metric only guards against a quant that breaks
normalization. German retrieval stayed 5/5 top-1 for EVERY arm including
0.6b q4_k at cos 0.69, which is exactly why a thresholded check cannot gate an
imatrix decision; its distractor scores tell the real story (gold
0.628/0.167/0.039 vs q4_k 0.786/0.519/**0.480**).

**The strongest result is one the q4_k-only brief would have missed: IQ4_XS
+imatrix beats Q4_K+imatrix on BOTH tails and is smaller, on all four models
that survive 4 bits at all.** min/mean: arctic .9667/.9757 vs .9480/.9614 (270
vs 274 MB); 80m .9601/.9812 vs .9455/.9767 (74.4 vs 74.7); 160m .9645/.9766 vs
.9495/.9719 (142.9 vs 143.5); 330m .9443/.9619 vs .9179/.9501 (259.5 vs 261.6).
It also repairs the one place imatrix made Q4_K *worse* (80m's min, 0.9499 ->
0.9455). If anything below Q8_0 is ever promoted, it should be IQ4_XS.

**The rule is not universal — the 0.6b inverts it**: iq4_xs+imatrix .6936/.7889
vs q4_k+imatrix .7654/.8115. Both are unusable, so the 0.6b keeps no sub-Q8
alias, but do not generalize IQ4_XS to a model without measuring it.

**Kaggle:** `chr1s4/crispembed-imatrix-t19`. Run 1 completed the full pipeline
for all five models in 21 min and then lost every artifact to `401` on each
upload — `resolve_hf_token()` does not glob the LONG dataset mount path
`/kaggle/input/datasets/<acct>/<slug>/`, which is the only layout that worker
had (`HF auth: /kaggle/input contains 1 entries: ['datasets']` →
`hf_token_ok: False`), while the ccache warm globs it and succeeded on the same
run. The driver now globs both and aborts up front when no token is found. **A
CrispASR harness fix is the proper home for this** — every kernel on such a
worker silently loses its uploads.

**Conclusion: imatrix helps q4_k everywhere but promotes nothing.** q4_k stays
far below q8_0 on all five, so q8_0 remains the right default and no registry
default was flipped. The one shipped file this touches — f2llm-v2-0.6b's
existing `-q4_k-imatrix.gguf` (calibrated on the 10-text fallback) — measures
min .7891 / mean .8345 locally, slightly ABOVE the new corpus's .7821/.8238, so
the re-calibration is published under `-c2` names and is NOT a promotion
candidate; its SHA is pinned in `model_hashes.h` and was not overwritten.

**T19-E4 status [DONE 2026-08-04, branch `feat/granite-r2-tokenizers`]:**
`granite-embedding-{97m,311m}-multilingual-r2` **shipped** (cstr, f16+q8_0,
registry + SHA pins, Q8_0 default — no imatrix calibrated yet). E2's three
gap items were all real and all fixed; a fourth defect fell out of the
token-id diff. Contract from each model card's own snippet: **CLS pooling,
L2-normalize, NO query or document prefix** (both `config_sentence_transformers.json`
prompts are empty strings), ModernBERT backbone, 8192 ctx.

- **(a) BPE-vs-SPM detection.** `is_sentencepiece` was
  `hasattr(sp_model) or vocab_size > 100000`, so any BPE vocab over 100k
  converted as SentencePiece. tokenizer.json `model.type` now overrides it.
  WordPiece vocabs over 100k (**LaBSE**, 501k) are deliberately LEFT on the
  historical path so nothing else changes — **still open**, and its shipped
  GGUF is worth an audit.
- **(b) o200k pre-tokenizer (97m).** The first pre-tokenizer here that
  **branches on letter case** — two of its seven alternatives are
  `[Lu Lt Lm Lo M]* [Ll Lm Lo M]+` and its mirror. The repo's historical
  "any byte >= 0x80 is a letter" shortcut puts every non-ASCII letter in BOTH
  classes, which splits an all-caps German word after its umlaut
  (`ÄRGER` -> `Ä` + `RGER`). Needed a real general-category table:
  `src/core/unicode_categ.h` (generated, 2779 ranges, `tools/gen_unicode_categ.py`).
- **(c) SPM-BPE mode (311m).** Wired via `tokenizer.ggml.is_spm_bpe` (the key
  the decoder path already read), including across the post-weight-load merges
  reload. Its post-processor prepends `<bos>` and appends NOTHING and the
  tokenizer exposes no cls/sep at all, so cls/sep/add_bos/add_eos now come
  from the TemplateProcessing template, not the BERT 101/102 defaults.
- **(d) NEW, found by the id diff — the merges blob cannot hold a newline.**
  `tokenizer.merges` is a NEWLINE-separated tensor, so a merge that CONTAINS a
  newline is unrepresentable. The Gemma vocab has **465** of them
  (`"\n\n" -> "\n\n\n"`), so `a\n\nb` tokenized as two separate newline
  tokens. Fixed with a NUL-separated `tokenizer.merges_nul` tensor emitted
  ALONGSIDE the legacy one only when needed, preferred when present — old
  binaries keep reading the legacy tensor unchanged. **Any other SPM-BPE GGUF
  converted by this script has the same defect baked in; re-conversion is the
  only fix.**

Numbers (CPU; the HF f32 reference was first validated against each model
card's own published cos_sim matrix):

| model | token ids vs HF | f16 cos_min | f16 pre-norm ratio | q8_0 cos_min | q8_0 pre-norm ratio | German 10-doc |
|---|---|---|---|---|---|---|
| granite-97m-r2 | **20/20 exact** | 1.000000 | 1.000000 | 0.999580 | 0.998388..1.001975 | 5/5 f16 + q8 |
| granite-311m-r2 | **20/20 exact** | 1.000000 | 1.000000 | 0.999758 | 0.999349..1.000586 | 5/5 f16 + q8 |

The id battery is German umlauts + all-caps, multi-space runs, newlines/tabs/
CRLF, a code snippet, unicode punctuation/quotes/currency, NBSP + soft hyphen,
CJK, Cyrillic, emoji, long compounds, contractions and digit groups. Retrieval
scores match the HF reference to 3 decimals (ECB 0.909/0.939, Rhein
0.945/0.949, Kartoffelsalat 0.937/0.941).

**Regression (the detection change must not touch any shipped model).** New
binary vs one built from the SAME tree with only the three changed sources
reverted — identical compiler flags, identical ggml, so the comparison is not
confounded: `multilingual-e5-small-q8`, `arctic-embed-m-v2`,
`gte-modernbert-base`, `f2llm-v2-80m`, `nomic-embed-text-v1.5-q8` are
**BIT-IDENTICAL, 0 token diffs**, covering the XLM-R/SPM, WordPiece,
ModernBERT-BPE (incl. the merges-tensor read) and decoder-BPE paths. ⚠ The
first attempt A/B'd against the main checkout's binary and showed cos ~0.9994
"changes" — that build has `GGML_METAL=ON` and the worktree's does not
([[build-dir-can-be-cpu-only]]); token ids were identical throughout, which is
what said the tokenizer was innocent.

**Guard:** `tests/test_o200k_pretokenize.cpp`, hermetic (no vocab, weights or
network), goldens from HuggingFace's own `pre_tokenize_str`, in the model-free
CI job. 85 checks. Written before the implementation and verified to FAIL on
three independent mutations: the naive non-ASCII-is-a-letter table (27
failures), `\p{N}{1,3}` narrowed to `{1,1}` (6), and the dropped contraction
suffix (2).

**Second NEW pre-existing defect, in the pinning tool itself:**
`tools/fetch_model_hashes.py` matched resolve-URLs with a regex over the raw
C++, so a URL written as ADJACENT string literals (what clang-format produces
past 120 columns) never matched and the entry was silently **left unpinned** —
`unpinned: 0` cannot see it, because such URLs never enter the list. That is
how **`granite-embedding-278m-multilingual` and `-107m-multilingual` shipped
with no SHA pin at all.** The tool now splices adjacent literals first; both
are pinned in this branch's regeneration.

**⚠ MERGE NOTE:** branch `feat/tokenize-simple-audit` adds
`src/core/unicode_class.h` for the same job. That table carries no case
information and so cannot serve the o200k split; `unicode_categ.h` here is a
strict SUPERSET and maps 1:1 onto its enum (mapping documented in the header).
Keep `unicode_categ.h`, express `core_uc_class` as that mapping, drop the
other generator.

**Not done / TODO:** no imatrix q4_k for either model (Q8_0 is the registry
default, mirroring arctic-embed-m-v2 — add them to the imatrix lane); the
LaBSE >100k-WordPiece detection question above; and neither model was measured
on Metal (all numbers here are CPU, worktree built `GGML_METAL=OFF`).

### T19 — German embedder quality snapshot (official MTEB data, 2026-08-04) + port candidates

Sources: MTEB(deu, v1) leaderboard (user-provided, open-weights/license filter)
+ our own extraction from the official `mteb/results` parquet (8.6M rows; 4
canonical German retrieval/rerank tasks, full-coverage models only). The two
agree where they overlap.

**German mean (MTEB deu v1) by size class, open models:**
`F2LLM-v2` (codefuse-ai, **Qwen3Model arch, Apache-2.0**) dominates every
class: 0.6B=63.08, 330M=61.61, **160M=57.35** (beats e5-large-instruct at
3.5x smaller), 80M=55.56. Caveat: zero-shot 63% (vs e5's 94%) — part of the
edge may be benchmark-adjacent training. Established models: e5-large-instr
57.14, arctic-l-v2 55.72 (best Retr. column of the field: **57.09**),
e5-base 53.03, e5-small 51.49, granite-278m 51.79/107m 50.16.
**German retrieval-only (our parquet extraction, ret4):** embeddinggemma-300m
**0.8947** (best <600M, gemma license), jina-v5-small 0.8761 (CC-BY-NC — dev
only), arctic-l-v2 0.8725, bge-m3 0.8714, arctic-m-v2 0.8569,
qwen3-embed-0.6b 0.8562, e5-base 0.8513, harrier-0.6b 0.8382,
granite-311m-r2 0.8287, harrier-270m 0.8240, granite-97m-r2 0.8005,
e5-small 0.7488 (n=2), MiniLM-multi 0.7327.

**In-registry verdict for German today:** `arctic-embed-l-v2` is the
evidence-backed quality pick (NOT qwen3-embed/harrier — qwen3-embed has no
official deu coverage and trails on ret4; harrier's "SOTA" desc oversells
German). `bge-m3` when hybrid dense+sparse+colbert wanted.
`embeddinggemma-300m` is in the registry (gemma license gate) and is the top
small retrieval scorer — but carries the known ~0.002 backbone discrepancy
(C8); spot-check German retrieval quality of OUR artifact before recommending.
`multilingual-e5-small` demoted to "cheapest acceptable" (51.49; fine at
118 MB, auto-prefixed).

**Port candidates, by value:** (1) **F2LLM-v2 family** — Qwen3Model + Apache;
likely loads via the existing qwen3-embed converter path with minor config
deltas; the 160M (57.35 deu, 640d, 40k ctx) would be the best small German
embedder we ship, 0.6B the best overall. (2) arctic-embed-m-v2.0 (0.8569
ret4, 305M — sibling of the l-v2 we already ship). (3) granite-r2 97m/311m
(8k ctx, Apache). Verify each with a German retrieval smoke vs the HF
reference before registry entry (decoded-embedding cosine + a 10-doc German
retrieval sanity, per HARD RULE #3 analog).

### T13 — olmOCR lane (the one absent family; cheapest add)

Zero trace in the repo. It is an Apache-2.0 Qwen2.5-VL-7B fine-tune, so the
`qwen2vl_ocr` engine and converter path should carry it. **Do:** convert the
olmOCR-2 checkpoint; implement its document-anchoring prompt contract;
registry + CLI name; gold fixtures from its toolkit. q4_k first (16 GB box;
DeepSeek at 5.3 GB peak ran). **Acceptance:** decoded output parity vs the
olmOCR toolkit on ≥5 anchored pages (their own eval format), HARD RULE #3
decoded-text gate, and a T12 harness row.

### T14 — DeepSeek-OCR: persistent decode graph + F16 KV (open lever #2) + CER gate

The decode graph is rebuilt and freed per layer per token, KV is F32 —
explicitly the one engine the GPU-decode "done" note does not cover
(§DeepSeek-OCR-2 levers). The qwen2vl engine next door already has the
persistent `build_decode_step_graph` + F16-KV pattern. Warm profile today:
~12 s total, decode 3.8 s. **Do:** copy the pattern; keep `DS_*` fallbacks;
CER gate via T12 BEFORE the perf work (no recorded reference parity exists).
**Acceptance:** decoded text unchanged on the existing fixtures, warm decode
time down with interleaved A/B, a reference CER row, and the CLI name from
T11.

### T14 status [DONE 2026-08-05, `feat/t14-deepseek2-decode-graph`]

**Shipped:** `[deepseek-ocr2-stage-bench]` (the T12 gap — `CRISPEMBED_DEEPSEEK_OCR2_BENCH=1`,
net-of-load, prefill/decode split so a prefill change cannot masquerade as a
decode win) + a persistent single-graph decode step, now the DEFAULT, with the
per-layer path kept selectable (`DS2_LEGACY_DECODE=1`).

**Acceptance (a) decoded text:** byte-identical, legacy vs persistent, on all
**25** gold fixtures on Metal (20 synth + 5 labelled CC0; identical SHA-256 over
the concatenated transcripts and identical `gen_tokens` per page) and on the CPU
subset. Never diffed against the gold itself — gold is a threshold reference.

**Acceptance (b) interleaved same-window A/B** (Metal, M1, `commons_example_receipt.png`,
217 generated tokens, 9 scored pairs + a discarded cold pair, alternating arms,
one process per run, pairs load-gated at 1-min loadavg ≤ 8; observed 1.4-2.5):

| arm | decode med | min | max | spread | total med | prefill med | sam med |
|---|--:|--:|--:|--:|--:|--:|--:|
| legacy per-layer | 11473.7 ms | 11022.9 | 12117.2 | 0.095 | 15815.2 ms | 461.1 ms | 2754.4 ms |
| persistent graph | **8191.5 ms** | 8035.6 | 13463.7 | 0.663 | **12784.8 ms** | 462.1 ms | 2821.5 ms |

**1.40x decode, 1.24x end-to-end.** Per-pair ratios 0.700 / 0.676 / 0.971 /
0.694 / 1.049 / 0.697 / 0.718 / 1.176 / 0.725 — median **0.700**, with 6 of 9
clustered at 0.68-0.73 and three upward excursions. Legacy's own spread is only
0.095, so the persistent arm's 0.663 is excursion-driven, not a wider
distribution; the median is the honest headline and the spread is quoted rather
than trimmed. **(c) No regression in the untouched stages:** prefill 461 vs 462
ms, sam 2754 vs 2822 ms, qwen2_enc 379 vs 376 ms — prefill deliberately still
runs the per-layer path.

**The task's stated premise was wrong, and that is the reusable finding.** T14
was scoped as "the decode graph is rebuilt and freed per layer per token" ⇒
amortise the rebuild. Measured with `DS_PROFILE=1`, the legacy path's graph
build+alloc is **1% of decode on CPU (26 ms of 5223 ms) and ~3-6% on Metal** —
there was never enough build overhead to be worth amortising. The win is
elsewhere: one graph per token replaces **13 backend dispatches and 24
host<->device hidden-state transfers per token**. Before porting this pattern to
qwen2vl/granite/smoldocling (PERFORMANCE.md P2), measure the overhead fraction
first — the lever is dispatch/transfer count, not graph construction.

**Copying qwen2vl verbatim was a 2.42x REGRESSION, and this is the trap to
record.** qwen2vl reads the full allocated `max_seq` every step and lets the
mask hide the tail. Here `max_seq` is `n_prompt+max_new+64` = 1408 while only
~478 slots are ever live, so every layer of every token attended over ~3x too
many slots and materialised three full `cont(permute(...))` copies of a
`[1280 x 1408]` K/V. Interleaved on Metal: decode 13654.9 ms legacy vs 32419.5
ms persistent, per-pair ratios 2.669 / 2.098 / 2.424 / 3.439 / 2.362 (median
**2.424**), no overlap. Fixed by bucketing the read depth to a multiple of 256
(`DS2_KV_BUCKET`, 0 restores the qwen2vl behaviour), which keeps the constant
shape that lets `sched_alloc` skip reallocation while reading only a little more
than is live: decode 32419 ms -> 8192 ms. **A pattern that is right for one
engine can be inverted by that engine's `max_seq`-to-live-slots ratio.**

**F16 KV (`DS2_KV_F16=1`) is implemented but deliberately NOT measured as part of
the acceptance gate and NOT default.** It is a precision change, so bundling it
with the graph refactor would have made any text diff unattributable; the
byte-identity gate ran with the cache dtype held fixed at F32. Quantifying it is
open work.

**Blocking infra bug found and worked around: Metal was silently OFF.** A build
dir configured once without Metal caches `GGML_METAL_EMBED_LIBRARY=OFF`, and
`option()` never revisits a cached value, so a later `-DGGML_METAL=ON` leaves the
library un-embedded; ggml then writes `default.metallib` to `build/bin/` while
`ggml_metal_library_init` looks beside `argv[0]` in `build/`, fails, and falls
back to CPU **while `CMakeCache.txt` still reads `GGML_METAL:BOOL=ON`**. Every
measurement taken before this was found was CPU mislabelled as Metal (`sam`
17.7 s vs 3.2 s). Worked around with `ln -sf bin/default.metallib
build/default.metallib`; the real fix (`GGML_METAL_EMBED_LIBRARY=ON`, or CMake
copying the metallib beside the executable) is **unowned follow-up work** —
it affects every Metal claim this repo makes from a `build/` binary. Full
mechanism in LEARNINGS.md.

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

**CER vs the A4 gold's own ground truth (threshold reference, never a byte
diff).** Both arms score IDENTICALLY to 5 decimals, as byte-identity requires —
so this is a threshold observation about the lane, not a T14 result. Numbers are
**pre-`feat/tokenize-simple-audit`** (that branch restores the dropped `\n` in
`"\nFree OCR."` and will move every page; a post-merge re-gate is owed):

| corpus | n | arm | CER raw | CER stripped | reference (A4) |
|---|--:|---|--:|--:|--:|
| synth | 20 | legacy | 0.00567 | 0.00348 | 0.00199 |
| synth | 20 | persistent | 0.00567 | 0.00348 | 0.00199 |
| cc0 | 5 | legacy | 1.06321 | 1.01739 | 0.18743 / 0.11063 |
| cc0 | 5 | persistent | 1.06321 | 1.01739 | 0.18743 / 0.11063 |

The cc0 mean is **not** a broad recognition gap — it is two pages spiralling into
the `max_new`=1024 cap:

| fixture | native CER | stripped | ref CER | chars | ref chars | gen tokens |
|---|--:|--:|--:|--:|--:|--:|
| `commons_example_receipt.png` | 0.2236 | 0.0270 | 0.2113 | 566 | 559 | 217 |
| `commons_test_ocr_document.jpg` | 0.0406 | 0.0339 | 0.0074 | 2958 | 2991 | 690 |
| `receipt_historical.png` | **0.1198** | 0.1263 | 0.3633 | 753 | 1070 | 434 |
| `german_official_print.jpg` | 2.2438 | 2.2398 | 0.1933 | 2962 | 1078 | **1024** |
| `simple_form.png` | 2.6883 | 2.6599 | 0.1619 | 762 | 287 | **1024** |

On the three pages that terminate normally the lane is competitive and
`receipt_historical` **beats** the reference (0.1198 vs 0.3633). The two capped
pages end in literal `FinlandFinlandFinland...`.

**Root cause of that, found not fixed — the lane implements no repetition guard
at all.** The captured contract (`tests/regression/gold/deepseek-ocr2/`) records
the reference generating with **`no_repeat_ngram_size=20`**; `src/deepseek_ocr2.cpp`
takes a plain `std::max_element` argmax with no equivalent, while
`qwen2vl_ocr.cpp` and `internvl2_ocr.cpp` both already carry
`argmax_no_repeat_ngram`. Porting that helper is a self-contained, high-value
follow-up that should recover both capped pages; it is deliberately out of this
branch because it changes decoded output and needs its own quality gate.

**Found, not fixed (each is someone else's lane):** (1) `--gpu-backend cpu`
silently falls through to Metal because `crispasr_init_gpu_backend()` scans only
GPU/iGPU devices — T18 owns it; this branch added the engine-local
`DS2_FORCE_CPU=1` it needed instead. (2) The lane feeds a single 1024x1024 view
(257 image tokens) while the A4 reference uses dynamic cropping (up to 1121
tokens), so native CC0 CER cannot approach the reference's until crop mode is
ported — a contract gap, not a T14 regression, and both arms share it
identically. (3) `--ocr-engine`'s help string omits `deepseek-ocr2` (and other
ids) though `eng_id` accepts it. (4) Two CC0 pages hit the 1024 `max_new` cap in
both arms.

**Artifacts:** `tests/results/t14/` (per-fixture transcripts + `runs.json`
stage-bench rows for every arm, both A/B windows) and
`tests/run_deepseek_ocr2_bench.py` (sweep + load-gated interleave modes).

### T15 — SmolDocling: fix the DocTags output before touching speed

Tensor parity 0.9999 but LIVE payload CER 0.86 from duplicated DocTags —
the harness-blind zone (LEARNINGS: diff the input/output contract, not more
tensors). Backend is hardcoded `ggml_backend_cpu_init`. **Do:** first
deduplicate/parse DocTags against reference output on gold pages; only then
un-hardcode the backend and A/B GPU. **Acceptance:** payload CER on gold
pages comparable to the reference implementation's own output; then
backend A/B with text gates. Related Docling-quality debt to carry: layout
detection score 0.934 vs HF reference 0.955.

### T15 status [DONE 2026-08-04, `feat/t15-smoldocling-doctags`]: contract fixed, native ≥ reference on 4/5 pages; backend port deferred with data

The "duplicated DocTags" was NOT a dedup/parsing problem — it was THREE
stacked contract defects, all invisible to tensor parity (the recorded 0.9999
was measured against a dumper that hand-squashed to 512², i.e. a
matched-WRONG-preprocessing reference):

1. **Converter dropped all 145 added tokens** (`model.vocab` only → vocab
   49152, not 49280), so detok silently deleted every generated
   `<loc_N>`/`<doctag>`/`<row_r_col_c>` id (out-of-range → `continue`) —
   the "mangled markup". Fixed in `convert-smoldocling-to-gguf.py`;
   **GGUFs converted before 2026-08-04 are defective** — all three quants
   re-converted and re-uploaded to `cstr/smoldocling-GGUF`, q8_0 SHA
   re-pinned, fresh-download re-verified.
2. **Preprocessing fed one squashed nearest-neighbor 512² image**; the
   reference does Lanczos longest-edge-2048 → round-up-to-512-multiples →
   512² tiles + squashed global view + `<fake>`/`<row_r_col_c>`/`<global-img>`
   prompt layout. The squashed input is what made the decoder hallucinate
   the duplicate regions. Ported exactly (incl. the "\n"+"\n"→single-token
   1116 BPE subtlety); prompt ids byte-identical to the reference processor
   on fox.png (347/347, pixel_values [1,5,3,512,512]).
   `SMOLDOCLING_LEGACY_PREPROC=1` restores the old path.
3. **max_tokens hardcoded 128** (a parity-era TODO) and
   `crispembed_ocr_model_set_max_tokens` never dispatched here — every page
   silently truncated. Default 1024, `--ocr-max-tokens` wired.

Also: registry name `smoldocling` added (engine was `-m <path>`-only — the
T11 reachability class), DocTags-aware payload scoring in
`ocr_engine_benchmark.py`.

**Acceptance (artifact `tests/results/ocr_parity_smoldocling_2026-08-04.json`,
raw paired outputs included):** fox payload CER **0.86 → 0.0000** (exact);
vs cc0 ground truth, native q8_0 **beats the transformers-f32 reference on
its own model**: commons 0.0077 vs 0.0956, receipt_historical 0.2344 vs
0.4935; scan_page_pd native visibly more correct (ref truncated at the
1024 cap with misreads). simple_form: shared failure — native emits a clean
`<picture>` classification (CER 1.0), the reference DEGENERATES into a
"Véhévé…" repetition loop (raw CER 3.23) — same receipt/form-class chaos
recorded for the A1/A3 references. q4_k and f16 fox-gated too (q4_k locs
shift more; payload exact).

**Deferred with data (the "then" half):** backend un-hardcode. Stage split
now: vision+connector 31.7 s of 37.3 s total on fox (5 sub-images, CPU) —
the port target is the per-tile SigLIP graph (compute-bound, GPU-shaped);
the 135M per-token decode is the CPU-favored shape per the persistent-decode
LEARNINGS. Split residency; do NOT move decode blindly. PERFORMANCE.md has
the table. Full pages are 72–103 s CPU — slower than pre-fix (N+1 vision
forwards) and worth the backend session.

### T16 — TableFormer port (MIT) + reading order — the missing Docling half

Tables today are rule-based morphology + Tesseract cells; no learned
structure model exists. TableFormer (docling-models, MIT) is the port
target; OTSL output feeds the existing `build_markdown` HTML path. Reading
order is nobody's job: the two-column fixture scores CER ~0.76 for EVERY
arm including system Tesseract — a column-aware ordering pass would improve
all lanes at once. **Acceptance:** table-structure accuracy vs pip Docling
on shared fixtures; two-column CER drops materially for at least the
tesseract and ppocrv6 lanes with no single-column regression.

### T17 — Tesseract quality round: Fraktur decode, WER, crop batching

Corpus CER is near-par (0.0290 vs 0.0256) but: Fraktur line decode is WRONG
vs official (`BEEES` vs `iE` on scan_strip — official binary locally
runnable, same reference-first bisect as the receipt); WER 0.1623 vs 0.0890
is spacing/grouping; Fraktur only survives at F32 (Q8 logits 0.9897); DAWGs
load but production decode ignores them; crops are recognized one at a time
(46-69 ms/line — the fused-batching/width-bucketing pattern from ppocrv6 is
the template). **Acceptance:** Fraktur line decode matches official on the
known fixture; corpus WER moves toward the reference's 0.089; batching
A/B'd text-identical.

---

## OPEN TASKS — ordered by expected value (prior round)

### T1 — Transcribe 5-10 CC0 scans [DONE 2026-08-03 for the 5 scoreable English-lane fixtures]

**Landed:** `tests/regression/images/cc0/ground_truth.json` (branch
`feat/cc0-ground-truth`, merged) — manual transcription with per-fixture
confidence and conventions (as-printed hyphenation, column reading order,
bleed-through excluded). `simple_table.jpg` is excluded as directional-only
per the trap note below; the out-of-scope fixtures (Fraktur manuscript,
Arabic, handwriting, sheet music) remain unlabelled. T2 and the real-scan CER
column are now unblocked; first scored results are in the 2026-08-03
head-to-head subsection above. Original brief kept below for the remaining
out-of-scope fixtures.

### T1 (original brief) — Transcribe 5-10 CC0 scans [BLOCKS T2, O8, and the WER column]

**This is the highest-leverage task available and it gates the others.** Every
remaining routing decision is a proxy for "which output is more correct", and
that question is answerable directly for a handful of pages.

**Why it is blocking.** The 14 CC0 scans have **no ground truth**, so every
"better" judgement in this area is a character-count proxy against DBNet — and
that proxy was proven *directionally wrong* on `german_official_document.jpg`,
where an English model transliterating 1848 Fraktur scored "better" purely for
hallucinating more fluently. Seven candidate probes have now been falsified
against labels of that quality (see T2). Tuning an eighth has a worse expected
return than an hour of transcription.

**Do** Transcribe the six fixtures the English lane can legitimately be scored
on: `commons_test_ocr_document.jpg` (two-column English print),
`german_official_print.jpg`, `receipt_historical.png`,
`commons_example_receipt.png`, `simple_form.png`, `simple_table.jpg`. Store them
in the same schema as `~/crispembed-ocr-synth/ground_truth.json`
(`records[] = {file, text}`) under `tests/regression/images/cc0/`, and record
provenance explicitly — who transcribed, and confidence per fixture.

**Known traps.** `simple_table.jpg` is 200x102: its title and 5x5 grid are
legible but the cell **digits are unrecoverable** even upscaled 6x, so mark it
directional-only, never a CER gate. These fixtures are **out of scope** for the
English lane entirely and must not be scored with it: `german_official_document`
(1848 Fraktur), `arabic_handwriting`, `german_kurrent_handwriting`,
`handwritten_letter` (handwriting), `arabic_printed_line` (needs
`tesseract-ara`), `public_domain_sheet_music` (not prose). A Fraktur model is
already cached (`tesseract-frk-*`) if someone wants that lane scored properly.

**Acceptance** A ground-truth file that lets `tests/ocr_external_parity.py`
report absolute CER on real scans, with per-fixture provenance and confidence.

---

### T2 — H9 cleanup axis: pick cleanup per page [BLOCKED ON T1]

**State.** H9's *segmentation* axis is **solved**:
`CRISPEMBED_TESSERACT_SEG_ROUTER=1` routes on detected column count (a gutter
with ink on **both sides** across a majority of text rows), **9/9 correct**.
Two-column pages fall back to DBNet and keep its full output; single-column
receipts gain 20-25% more text. On the synthetic corpus with cleanup off:
**CER 0.01538 at 254 ms/page against DBNet's 0.02880 at 893 ms** — half the error
at 3.5x the speed, which is level with `tesseract-cli` on speed and ahead on
quality.

**What blocks the default.** The *cleanup* axis pulls the other way and has no
probe. Cleanup wants OFF for clean rendered text (CER 0.0154 vs 0.0316) and ON
for real scans (`receipt_historical` 600 characters vs 11). So the router is
`{DBNet, classical} x {cleanup, no cleanup}` and each of the four is best on some
page.

**Root cause (established, do not re-derive).** Background whitening
(morphological closing) **erodes antialiased glyph edges** — strokes thin, serifs
break, periods vanish. Destructive on clean rendered type whose shape depends on
mid-grey antialiasing; harmless on scans with thick saturated ink, where it
removes only artefacts. Confirmed visually via `--cleanup-only`.

**Seven probes already falsified — do not repeat them:**

| probe | why it failed |
|---|---|
| ink coverage (Otsu, fraction inside boxes) | `commons_test_ocr_document` scores **1.0000** while losing 91% of text |
| median box height / page height | ranges overlap outright |
| paper-class noise | synthetic 6.9-13.2 vs scans 5.0-14.0 |
| illumination spread (tiled background) | `commons_example_receipt` scores 0.00 like the synthetics, yet wants cleanup |
| ink retained after cleanup | that fixture keeps only 38% of ink and reads *better* |
| recognizer mean confidence | 2/5, and biased toward cleanup-ON |
| in-box vs out-of-box ink loss | does not separate, **and** cleanup changes image geometry (2518x1920 -> 2532x1938), which invalidates every pixel-aligned before/after measure |

**Two structural obstacles are now identified rather than suspected:** no
page-level statistic separates ink-that-is-glyph from ink-that-is-artefact, and
no pixel-aligned before/after measure survives cleanup's own geometry change.

**Do** After T1, stop probing: run both cleanup arms on the labelled pages and
score them. If a cheap signal correlates with the scored winner, use it; if not,
the honest answer may be to run both arms and keep the better — recognition is
~200-400 ms while the detector is the expensive stage, so a second pass is
affordable on the classical path.

**Acceptance** No CER regression on the 20 synthetic fixtures **and** no CER
regression on the newly labelled CC0 scans, with per-page time down where the
router accepts classical. Report both corpora — either alone gives the opposite
answer.

---

### T3 — Depthwise convolution: unroll across taps [detector, ~20% of conv time]

Depthwise runs at **0.02-0.19 GF/s** against ~1.2 GF/s for pointwise, and is
20.4% of detector convolution time. `conv2d_depthwise_cpu`
(`CRISPEMBED_CONVDW_FAST=1`) already inverts the loop nest so each tap is a
contiguous axpy, but measured only ~3.6% and is **unverified against an 8.1%
noise floor** — treat it as unmeasured.

**Ruled out:** it is *not* an aliasing-blocked vectorization problem
(`clang -Rpass=loop-vectorize` shows the aliasing and `__restrict` forms both
vectorize, width 4 interleave 4). The remaining suspect is invocation overhead:
the 7x7 layer runs **1.13M invocations** of a ~184-element inner loop, each
paying a runtime alias check, prologue and remainder epilogue.

**Do** Unroll across taps so one pass over an output row accumulates several kx
taps at once, cutting both output-row traffic and invocation count by the unroll
factor; keep row borders on the current general path. Measure as interleaved
pairs on the VPS **and** the Mac.

**Acceptance** CPU-seconds down on both hosts (or explicitly neutral on one),
decoded text identical on the 34-fixture check, and the equivalence guard in
`test-core-cpu-ops` still passing on NEON and AVX2.

---

### T4 — Batched crop recognition (was H4)

Every lane recognizes line crops one at a time. Crops sharing a canvas width
could go through one graph dispatch as a batch dimension. Largest win on
many-region pages — one CC0 scan yields 71 regions.

`EASYOCR_WIDTH_SORT=1` already groups equal-width crops adjacently, and its sort
key was fixed in Aug 2026 (it had hardcoded the 2-pixel detector margin, so the
external-geometry path sorted by widths it never requested). Measured ceiling:
27 regions over **14 distinct widths**, so grouping can remove at most 13 of 25
graph rebuilds; `set_width` is 24% of the recognition loop.

Start with PP-OCRv6, whose graph is already default and shape-keyed via
`pp_graph_build(c, width)`. A bounded fused batch exists behind
`CRISPEMBED_PPOCRV6_BATCH_GRAPH` (CPU-only; Metal hit a pooling shape assertion).

**Reference design (an MIT-licensed C++ onnxruntime pipeline of v6, cloned under `/Volumes/backups/code/`) — its
onnxruntime CPU path does the 47-region receipt in 2.2 s where we take 6.2 s,
and its rec pipeline (`src/recognition/cpu_paddle_rec.cpp`) is the design to
steal from:** width bucketing with a fine step (`ceil(w/16)*16`, batch only
within a bucket — a coarse fixed table over-pads, a consecutive-N grouping
after sorting over-pads at bucket boundaries); natural content width with
floor 32 instead of pad-to-320 (fewer wasted columns); single
warpPerspective from the original image straight to `(w,48)` (the
crop-then-resize path resamples twice and destroys small glyphs); a 2-row
duplicate-crop batch self-test asserting both rows decode identically
(catches row-stride bugs); zero-copy decode out of runtime-owned output; and
a NEON argmax over the 18,710-wide class rows (their SIMD is AVX2-only —
unclaimed headroom on this M1).

**Acceptance** Decode each row independently and compare against the unbatched
result for the same crop, plus the 34-fixture text check.

---

### T5 — Unblock the ppocrv6 lane's per-invocation Metal cost

On a single page the ppocrv6 recognizer's Metal init costs more than it saves:
forcing CPU is **~9% faster** (2.04 -> 1.87 s, 4/4 interleaved pairs, decoded
text identical). It amortises over a multi-page process, so this is a
workload-dependent default, not a flip: a one-shot CLI wants CPU, a warm server
wants Metal. Wire it to the invocation mode rather than hardcoding either.

---

### T6 — Detector resize rule keyed on text height (was H6)

Unchanged. `upscale_floor=120` in `src/ocr_detect.h` is a proxy for "are the
glyphs big enough"; a stroke-width or connected-component estimate measures the
real thing. **Acceptance, both required:** no CER regression on the 20-fixture
corpus, **and** `tests/regression/images/cc0/simple_table.jpg` (200x102) still
detects its region — capping it unconditionally takes that fixture from 1 region
to 0.

---

### T7 — PP-OCRv6 detector graph geometry parity [CLOSED 2026-08-04 — one-line bug; graph promoted to default and it IS a performance item after all]

**The divergence was an arithmetic bug, not a postprocessor disagreement:**
the graph's fused-stage insert-SE applied `ggml_scale(gate, 0.2f)` *and*
`ggml_scale_bias(gate, 0.2f, 0.5f)` — hard-sigmoid squashed to `0.04x+0.5`
where the scalar path (and Paddle's SELayer) use `0.2x+0.5`; the proc-stage
SE never had the extra scale, which is why divergence started exactly at
`fused0` (cosine 0.988). After the fix: probability cosine ~1e-8 with equal
norms on synth/german/receipt.

**And the "2.6-6.8x slower" claim was a backend artifact:** `DET_GRAPH`
*implied* GPU load, so the graph had only ever been timed on Metal (1693 ms)
— on the CPU backend the same graph runs **175 ms vs 316 ms scalar** on
synth_00_clean and **1363 ms vs 2056 ms** on the 1920x2518 Fraktur page.
Promoted to default for tiny/small on CPU (25-fixture labelled CER net-better,
0.06394 vs 0.06410; receipt hits 0.00000; box-level diffs are threshold
jitter). `CRISPEMBED_PPOCRV6_DET_SCALAR=1` restores scalar;
`CRISPEMBED_PPOCRV6_DET_GPU_LOAD` is the explicit GPU opt-in.

**Same day, the medium tier followed:** `run_medium_neck` (RepLKFPN: adjust /
top-down / project / bottom-up / lateral / med_ic refinement) is now in the
persistent graph, every `med_*` tap at cosine 0.99999998-1.0, probability
0.99999999 with equal norms and same box counts. Detector time
**6911→1024 ms** (synth page) and **41438→8711 ms** (`german_official_print`),
German CER graph 0.04856 vs scalar 0.04955 — the CPU-scalar medium detector
was why the medium tier blew the 120 s benchmark guard, so the highest-quality
tier is now actually usable. Medium graph default like tiny/small.

Remaining: Metal conv perf, and the comparator's own graph-box extraction
(emits `graph=0` — the accept path is exercised instead).

---

### T8 — Server: 8 JSON field reads still scan textually [DONE 2026-08-04, merged via `chore/ocr-followups-0804`]

All listed sites (3× `text`, 2× `format`, `results`, `autorotate`, `images`)
plus the per-result-object `text` moved onto `core_json` depth-1 helpers;
9 nested-decoy checks added to `tests/test_server_json_input.cpp` (all pass).
Two extras found while in there: (a) the `images` array was the one
path-valued input that BYPASSED `--image-root` confinement — each entry now
goes through `path_within` like the single-image field; (b) the T11
segfault (Tesseract GGUF in the flat rec slot) is fixed — `math_ocr_init`
now refuses foreign GGUFs loudly naming their `general.architecture`
(positive-tested: pix2tex-mfr q4_k still maps 12/12+6/6). Residual, NOT
fixed: an engine load failure inside the flat pipeline still yields
`regions=0` with exit code 0 — indistinguishable from a blank page for a
benchmarking caller (HARD RULE #8 class); needs a status channel through
the orchestrator before the CLI can exit nonzero. Original brief below.

### T8 (original brief) — small, self-contained

**State.** `extract_path_field` was moved onto `core_json`'s depth-1 finder on
2026-08-03 (commit `54aeaecb`), so every *path* field — `image`, `output`,
`file`, `model` — is now read the same way the confinement checks it. Eight
non-path reads were not converted and still do a bare `body.find("\"key\"")`.

**Why it matters.** A textual scan matches the key anywhere, including inside a
nested object. `{"meta":{"format":"a"},"format":"b"}` makes the server take the
nested value while a validating proxy in front reads the top-level one, so the
two disagree about what was requested. That is exactly the disagreement the
`image` field had. These are non-path fields, so nothing is reachable through
them the way an arbitrary path was — this is consistency work, not an open hole,
and should not be written up as a vulnerability.

**Do** Convert `examples/server/server.cpp` lines **1218, 1872, 1935** (`"text"`),
**2378, 3243** (`"format"`), **2394** (`"results"`), **3249** (`"autorotate"`),
**3255** (`"images"`) to `core_json::json_extract_strings` /
`json_extract_number`. Line numbers are as of `b95f4f93`; re-grep
`body\.find("\\"` before trusting them.

**Acceptance** `tests/test_server_json_input.cpp` gains a nested-decoy case per
converted field, each failing before the change and passing after. No behaviour
change for well-formed requests.

---

### T10 — PP-OCRv6 symbol-class gap [RESOLVED 2026-08-04 — it was the cleanup stage, not the recognizer]

**Root cause, proven by bisection; fix merged as `fix/ppocrv6-cleanup-default`.**
The recognizer port is CORRECT: on the pipeline's own dumped crops it decodes
the receipt perfectly, scalar and batch identically (`test-ppocrv6-rec`
parity=PASS 5/5), and the activation audit (stem ReLU / channel-mixer GELU /
neck SiLU ×5 / tiny-guide Hardswish) matches both `rec_lcnetv4.py` and the
official ONNX op pattern (13 Erf, 5 `x·σ(x)` SiLU, 10 ReLU, 5 HardSigmoid).
The corruption came from `--ocr-pipeline`'s scan-cleanup stage:
`scan_cleanup_process` converts to grayscale and runs **despeckle +
blackfilter unconditionally** (defaults on, and
`--no-deskew/--no-crop-borders/--no-whiten` do not touch them — they have no
CLI switches at all), eroding thin strokes on clean rendered type before the
detector ever sees the page. `test-ppocrv6-direct` on the cleaned image
reproduces `$`→`S`, `:tem`, `QLY`, `Frice` byte-for-byte; on the raw image it
reads everything.

**Fix (merged):** the ppocrv6 stage skips destructive cleanup by default,
mirroring the VLM carve-out and the official pipeline (which detects on the
raw page); `CRISPEMBED_PPOCRV6_CLEANUP=1` restores the old behaviour.
Same-binary validation: labelled CC0 mean CER 0.332→**0.293** / WER
0.557→**0.473** (receipt 0.0885→**0.0025**, beating official paddle's 0.0074;
form 0.737→0.615, also ahead of paddle; Fraktur 0.0486→0.0535 and dot-matrix
0.0260→0.0273, noise-level), and the lane's median engine_ms fell
9414→**7682**, now below paddleocr-py's 7933 on these pages. Cost, reported
rather than hidden: synth mean CER 0.0031→0.0070 (still 2.7x ahead of
paddle's 0.0185), concentrated in the `_noise` variants where despeckle acted
as a denoiser (`synth_00/01/02/03_noise` +0.008/+0.008/+0.007/+0.026-ish,
plus a ±0.02-0.04 wobble on two `clean` fixtures in opposite directions).

**Follow-ups spawned:** (a) the T2 cleanup router now owns the noise-page
axis, with fresh per-arm ppocrv6 evidence on both corpora; (b) despeckle and
blackfilter deserve CLI flags — today they are unreachable knobs; (c) the
official-ONNX `-ref.gguf` stage-diff remains the right tool if a
recognizer-level anomaly ever resurfaces (reference:
`~/venvs/rapidocr/.../models/PP-OCRv6_rec_small.onnx`; do NOT use the repo's
gold archives as the reference — they are dumps of our own torch mirror and
prove only self-consistency).

---

### T9 — Re-read the control tests for coverage, not correctness (method, ~1h)

**Why this exists.** The `--image-root` shadow survived **three** AI Act audit
rounds. `tests/test_image_root.py` existed, passed, and was written precisely to
prove that control — but it probed `/detect`, which sat on the confined side of
the bug, while POLICY.md claimed the control covered *every* endpoint. A passing
test about one endpoint was read as evidence for a sentence about thirty-three.
Greps did not catch it either, because both the correct and the broken call
sites read `extract_image_path(body)` verbatim.

**Do** For each remaining stated control, ask one question: *does the test cover
the whole set the prose claims, or one member of it?*
- **Biometric gate** (`crispembed_face_init`) — POLICY §4 says every binding
  funnels through it. Does the test exercise Python/Rust/Dart/CLI/server, or one?
- **SHA-256 model pinning** — POLICY §7 says every auto-download URL is pinned.
  `fetch_model_hashes.py --check` runs in CI; does it assert *coverage* of the
  registry, or only that listed pins match?
- **Provenance marking** — POLICY §5 says every image returned is marked. The 12
  base64 SR/restore endpoints were missed once already for exactly this reason.

**Acceptance** Either a test extended to the full set, or a written note in
POLICY.md narrowing the claim to what is actually verified. Narrowing the prose
is a valid outcome and is often the honest one.

---

### Standing constraint — EU AI Act (not a task; do not "close" it)

The 2026-08-03 audit closed every actionable finding. What remains is **not
implementable**: POLICY.md §5/§6 argue that document restoration and
transcription fall outside Art. 50(2) via Recital 134. That is a reasoned
position, untested by any regulator or court, and it is the ceiling on any
"CrispEmbed is compliant" claim — do not let a green test suite be read as
having settled it. Two dates still run: **2 December 2026** ends both the
Art. 50(2) machine-readable-marking grace period for systems already on the
market and the NCII/CSAM prohibition transitional period. If the §5 argument is
ever abandoned, the marking path already exists (PNG `iTXt` by default, signed
C2PA when `CRISPEMBED_C2PA_CERT`/`_KEY` are set) — that decision is policy, not
implementation. Regulatory dates were last verified against the OJ text on
2026-08-03 (Reg. (EU) 2026/1744); re-verify before relying on them.

---


### PP-OCRv6 detector-to-recognizer contract (selected follow-up)

The PP-OCRv6 port must follow the official PaddleOCR handoff rather
than treating a segmentation result as an axis-aligned crop. The canonical
path is:

`PP-OCRv6 DB segmentation → DBPostProcess quadrilateral → ordered perspective
warp → line-orientation classifier → PP-OCRv6 recognizer`

Requirements:

1. Keep the detector's four-point polygon through postprocessing, including
   Paddle's contour score, `thresh=0.2`, `box_thresh=0.45`, `unclip_ratio=1.4`,
   candidate cap, clipping, and reading-order sort. The external
   `$CRISPEMBED_GGUF_DIR/dbnet-ic15-f16.gguf` remains a fallback
   detector, not a replacement for the PP-OCRv6 detector.
2. Add a shared quadrilateral crop helper equivalent to the upstream PP-OCR
   `get_rotate_crop_image`: canonical TL/TR/BR/BL ordering, perspective
   transform, width/height derivation, clipping, and deterministic minimum
   height padding. Do not collapse rotated or skewed regions to an
   axis-aligned rectangle.
3. Port the optional PP-LCNet text-line orientation classifier (0°/180°),
   exposing predicted angle, confidence, and whether a rotation was applied
   through native and C API results. The existing heuristic 180° detector is
   only a fallback when the classifier is unavailable.
4. Apply the official recognizer input contract after warping: RGB, height 48,
   aspect-preserving width, cap/pad to 320, declared normalization, and the
   model dictionary/CTC decode. Preserve sensitive head weights at higher
   precision and keep `-ref.gguf` fixtures for each tier.
5. Provide one comparison harness for four combinations: PP-det→PP-rec,
   DBNet→PP-rec with grouped line boxes, DBNet→PP-rec word boxes, and the
   direct legacy DBNet pipeline. Report detector polygons, crop dimensions,
   orientation decisions, region/text counts, latency, CER/WER where an oracle
   exists, and `crispembed-diff` cosine metrics.
6. Gate tiny/small/medium on the 10 CC0 fixtures plus derived rotation,
   skew, perspective, low-DPI, and mixed-orientation fixtures. A candidate is
   not publishable until Python/PaddleX and native agree on crop geometry and
   the complete pipeline no longer emits full-page recognizer garbage on the
   German fixtures.

Parity note (2026-07-31): `inference.yml` declares `DecodeImage(img_mode=BGR)`;
the public crop handoff remains RGB, with the recognizer swapping to BGR at
its model boundary. Exact native detector crops now reproduce the visible
German title geometry. On the Fraktur crop, Python and native logits agree at
cosine 0.99998--0.99999, but both decode `Rieilhs–刊臂懒s²ł1&tt.`; this is a
source-model recognition-quality limitation, not a native crop/inference
parity failure. Do not publish a quality claim for this fixture until a
Paddle/Python official run confirms whether the model itself has the same
limitation.

Implementation order: shared quad warp and telemetry; PP-LCNet classifier;
PP-det→PP-rec integration; DBNet fallback adapter; per-stage diff fixtures;
benchmark and regression manifest; then publish only to `cstr/` from the
external model volume. This work must be merged from remote `main` before
each landing checkpoint and pushed back to remote `main` after the checkpoint.

### Historical German Fraktur OCR inventory and port plan (2026-07-31)

The directly portable German-Fraktur model is the official Tesseract `frk`
traineddata shipped by `tesseract-lang`. Local inspection with
`combine_tessdata -u` confirms an LSTM network, `lstm-unicharset`, and
`lstm-recoder`; the existing converter successfully produced
`$CRISPEMBED_GGUF_DIR/tesseract-frk-f32.gguf` (3.6 MiB, 933,763
parameters). This is the first Fraktur model to use in the native
`tesseract_lstm` GGUF path. Preserve the output alphabet, including long-s
and historical characters, and keep the sensitive output layer at F32 for
the first quantization gate. The upstream Tesseract language data and
runtime are Apache-2.0; retain the exact source checksum and attribution in
the eventual model README.

| Source | What it contains | License/provenance | GGML/GGUF decision |
|---|---|---|---|
| [Tesseract `frk`](https://github.com/tesseract-ocr/tessdata) | Current LSTM Fraktur recognizer; local `frk.traineddata` is the 4.00-alpha synthetic-trained network with 100 output codes | Apache-2.0 Tesseract language data; verify exact tessdata revision/checksum per artifact | **Port now:** existing `convert-tesseract-to-gguf.py` and `tesseract_lstm.cpp`; test on German Fraktur crops and full pages |
| [`paalberti/tesseract-dan-fraktur`](https://github.com/paalberti/tesseract-dan-fraktur) `deu_frak` | German Fraktur Tesseract 3.02 package containing the compiled `deu_frak.traineddata`, `deu_frak.config`, `unicharambigs`, dictionaries/lists, build script, and many paired `.tif`/`.box` training samples (including font and scanned-page examples) | Repository `COPYING` says Apache-2.0; retain attribution and separately verify provenance of the historical source scans and annotations before redistributing or using them as a new corpus | **Compatibility and training-data candidate:** the legacy package is not an LSTM GGUF input and must not be passed to the current converter as if it were; use it with a legacy Tesseract 3.02 path, or use the paired image/box material for clean-room retraining into a supported LSTM format after provenance review |
| [`jze/ocropus-model_fraktur`](https://github.com/jze/ocropus-model_fraktur) | OCRopus `pyrnn.gz` and CLSTM Fraktur character models; reports 1.089% held-out error on its own test set | No explicit license found in repository; source books/datasets also need provenance review | **Do not redistribute/port yet:** license clarification required; format is not Tesseract LSTM |
| [`chreul/19th-century-fraktur-OCR`](https://github.com/chreul/19th-century-fraktur-OCR) | MIT Calamari five-model voting ensemble, single Calamari model, and OCRopus model for 19th-century German Fraktur; models expect binarized line images; repository includes at most 50 lines per book so users can adapt transcription guidelines, while the complete corpus is obtained from the linked GT publication | Repository MIT; the supplied model/adaptation files are distinct from the complete training corpus, whose source and ground-truth licensing must be checked independently | **Benchmark/retraining candidate:** no current GGUF importer; use the supplied samples for reproducible adaptation tests, then retrain into a supported LSTM graph if the full corpus permissions and target transcription policy are clear |
| [`yingyangle/fraktur`](https://github.com/yingyangle/fraktur) | Research project using GT4HistOCR line data, character segmentation, zoning/black-pixel features, and a k-NN classifier; reports about 92% character accuracy | **No repository license or explicit reuse grant found.** GT4HistOCR itself is CC-BY-4.0, but that does not license this repository's code, pickles, feature data, or derived artifacts | **Reference only:** do not copy, train from its artifacts, or redistribute until the authors clarify licensing; independently reproduce the simple feature baseline from the separately licensed GT4HistOCR data if useful |
| [`UB-Mannheim/AustrianNewspapers`](https://github.com/UB-Mannheim/AustrianNewspapers) | NewsEye/READ Austrian newspaper ground truth, 1864–1911; revised PAGE-XML with Fraktur/Antiqua, baselines, regions, and long-s transcription | Original dataset explicitly CC BY 4.0; preserve attribution to Mühlberger/Hackl, Austrian National Library, and the repository revision | **Eligible training source:** strong Fraktur/Antiqua classifier and Tesseract fine-tuning corpus after source/page split and attribution manifest |
| [`UB-Mannheim/reichsanzeiger-gt`](https://github.com/UB-Mannheim/reichsanzeiger-gt) | 119,429 lines from 197 German newspaper pages, 1820–1939; Fraktur and Latin, with long-s and historical characters | Current GitHub repository advertises CC0-1.0; verify the exact revision and downloaded scan URLs before redistribution | **Highest-priority training/evaluation source:** large, directly relevant Fraktur corpus; use for classifier and Tesseract fine-tuning, preserving its transcription conventions |
| [`ulb-sachsen-anhalt/ulb-groundtruth-eval-odem-ger`](https://github.com/ulb-sachsen-anhalt/ulb-groundtruth-eval-odem-ger) | OCR-D Phase III ULB VD18 German-Fraktur Page-XML ground truth: 39,823 text lines across 1,026 pages and 6,298 text regions, dated 1700–1799, with non-text region annotations | Repository metadata says CC-BY-4.0, while the GitHub license badge says CC-BY-SA-4.0; treat it conservatively as CC-BY-SA-4.0 until ULB resolves the discrepancy, and preserve the repository citation | **Highest-priority training/evaluation source:** convert Page-XML lines/regions into explicit line crops and manifests; useful for 18th-century Fraktur, layout/region evaluation, and clean-room LSTM fine-tuning once the license is settled |
| [`UB-Mannheim/dach-gt`](https://github.com/UB-Mannheim/dach-gt) | Ground truth and full text for selected prints from German libraries; repository ships data plus image-download tooling and supports PAGE/ALTO/escriptorium workflows | CC0-1.0 repository license; retain institution/source URLs and verify any remote scan terms | **Eligible broad training source:** mine Fraktur lines and full-text alignment for detector/recognizer evaluation and clean-room training; keep each institution as a separate provenance split |
| [`jbaiter/archiscribe-corpus`](https://github.com/jbaiter/archiscribe-corpus) | 4,255 German-Fraktur lines from 112 works across 73 years (1800s–1890s), with transcription directories and Archive.org/IIIF source links | CC-BY-4.0; attribution and source-work provenance are required | **Eligible evaluation/adaptation source:** strong chronological/style diversity; import the line images and transcriptions only with an attribution manifest and use work-level splits to avoid leakage |
| [`UB-Mannheim/charlottenburger-amtsschrifttum`](https://github.com/UB-Mannheim/charlottenburger-amtsschrifttum) | 26 pages of German Fraktur ground truth from 1879–1919, including long-s (ſ), German Mark (ℳ), double oblique hyphen (⸗), fractions, and downloadable image URLs | CC0-1.0 | **Small but valuable regression/adaptation fixture:** use for historical-character coverage, line-recognition tests, and domain adaptation; do not treat its 26 pages as a standalone general corpus |
| [`UB-Mannheim/Reichsanzeiger`](https://github.com/UB-Mannheim/Reichsanzeiger) | Apache-licensed software/data support for the digital edition of *Deutscher Reichsanzeiger und Preußischer Staatsanzeiger*, including the SQL image/issue mapping and project metadata | Apache-2.0 repository; the underlying scans, full text, and linked digital-edition assets must retain their own terms | **Pipeline/provenance reference, not a recognizer:** use the issue/image mapping to locate and reproduce evaluation pages; keep this Apache repository separate from the `reichsanzeiger-gt` CC0 ground-truth corpus |
| [`SimoneRebora/OCRFraktur`](https://github.com/SimoneRebora/OCRFraktur) | OCRopus Fraktur training project: 4,287 real lines from *Tiroler Soldaten-Zeitung*, 3,000 synthetically generated lines from three Fraktur fonts, a 410-line test set, and a `TSZ_Fraktur_model.pyrnn.gz` model reporting 0.479% CER on its own test set | **No repository license found.** The README cites the Musil edition and source texts/fonts; those inputs have independent provenance and must not be assumed redistributable | **Reference only until licensing is clarified:** valuable for synthetic-data design and an OCRopus baseline, but no redistribution, training, or GGUF conversion from its artifacts; independently recreate the experiment from cleared sources if needed |
| [`phildiderichsen/MeMo-Fraktur-OCR-code`](https://github.com/phildiderichsen/MeMo-Fraktur-OCR-code) | Rule-based correction/evaluation workflow: Tesseract re-OCR with `frk`, `fraktur`, and `dan`, alternative OCR comparison, regex/context rules, SymSpell, VRT alignment, and staged intermediate outputs | **No repository license found.** Its PDFs, OCR outputs, dictionaries, and external corpora require separate permission review | **Concepts only:** reproduce the error-analysis, multi-model voting, historical spelling, hyphenation, and correction stages in our own code; do not copy source or derived artifacts into a distributable model/data package |
| [`muratyanasoglu/AI-Powered-OCR-Project-for-Ancient-Languages-Ancient-Greek-Fraktur-Old-German-Classical-Latin`](https://github.com/muratyanasoglu/AI-Powered-OCR-Project-for-Ancient-Languages-Ancient-Greek-Fraktur-Old-German-Classical-Latin) | Django/PyTesseract application using Tesseract 5, OpenCV/Pillow preprocessing, configurable PSM/OEM, `deu_latf`/`frk`/other language packs, and optional Gemini translation/analysis; no custom OCR weights | No explicit repository license found in the inspected tree; Tesseract language data and Gemini service have separate terms | **Workflow reference only:** useful ideas for script-specific preprocessing and output export, but it contributes no native recognizer, detector, or GGUF-compatible weights; do not bundle its external language packs or API integration without review |
| [`Nargizi/oppy`](https://github.com/Nargizi/oppy) | Small German-Fraktur PDF OCR wrapper; dependencies show PyMuPDF, OpenCV, `pytesseract`, and `fsspec`, with package modules for PDF/image/text handling | MIT repository license; Tesseract and any downloaded language data remain separately licensed | **Reference only:** it uses the system Tesseract executable through PyTesseract rather than a custom model; compare its PDF rasterization/preprocessing ideas with our native pipeline, but there is nothing to convert to GGUF |
| [`UB-Mannheim/digitue-gt`](https://github.com/UB-Mannheim/digitue-gt) | CC0 transcriptions for digitized Tübingen books/journals, including Fraktur-tagged material; images fetched from UB Tübingen URLs | Repository advertises CC0-1.0; confirm that downloaded image endpoints carry compatible reuse terms and keep source URLs | **Eligible source:** mine Fraktur/Antiqua line crops for clean-room classifier training and evaluation; retain provenance even under CC0 |
| [`samprietoserrano/archival-ocr-transcription`](https://github.com/samprietoserrano/archival-ocr-transcription) | MIT-licensed Python workflow and outputs for a 1719 German Fraktur book by Peter Kolb; extraction used Google Document AI and Transkribus, followed by local cleanup, historical spell-checking, and reading-order assembly | Repository MIT; Internet Archive scans, DTA corpora, CLARIN GeMiCorpus, and Transkribus/Google outputs retain their own terms | **Reference/evaluation source, not a model port:** useful for a historical Fraktur page fixture, reading-order and post-correction tests, but it supplies no portable LSTM/CNN weights. Audit source-image and corpus permissions before using its text/images for training |
| SchriftLotse `party-v4` | Swin-base vision encoder plus 40M-parameter Llama decoder; page-wise recognition with line prompts | Apache-2.0 model release | **Not a near-term port:** custom Kraken multimodal architecture; potentially reusable after a dedicated Swin/Llama graph and tokenizer audit |
| SchriftLotse Kraken BLLA | Trainable neural baseline/line segmenter | Apache-2.0 via Kraken/model release | **Not a near-term port:** PyTorch Kraken BLLA format and baseline geometry are separate from DBNet; evaluate as an external segmentation oracle first |
| SchriftLotse `orli` | ConvNeXtV2-tiny encoder, multi-scale adapter, autoregressive transformer decoder for baselines and reading order | Apache-2.0 model release | **Not a near-term port:** new autoregressive layout graph; consider only after native line/ordering gates |
| SchriftLotse `trocr-kurrent-19`, `trocr-kurrent-early`, `trocr-medieval` | Historical handwriting TrOCR checkpoints | MIT according to the inventory | **Potential later port:** assess against existing TrOCR encoder/decoder implementation; verify exact HF architecture and tokenizer before conversion |
| SchriftLotse `trocr-modern` | German handwritten TrOCR checkpoint | AFL-3.0 according to the inventory | **Potential later port with license review:** same TrOCR compatibility check, but AFL-3.0 obligations must be documented |
| SchriftLotse Microsoft TrOCR processor | Processor/tokenizer only, not a recognizer model | MIT according to the inventory | **Reuse only as preprocessing/tokenizer reference;** no standalone GGUF model |
| SchriftLotse `qwen-embed` | Qwen3 Embedding 0.6B semantic search model | Apache-2.0 | **Already covered by decoder embedding infrastructure;** not an OCR recognizer |
| [Xilinx `LSTM-PYNQ`](https://github.com/Xilinx/LSTM-PYNQ) `Fraktur_OCR.ipynb` | Quantized BiLSTM Fraktur OCR overlay for PYNQ; notebook constructs `PynqFrakturOCR`, downloads FPGA bitstream and Fraktur weights, and recognizes lines from *Wanderungen durch die Mark Brandenburg* | Repository BSD-3-Clause; the notebook cites the FINN-L paper and an Insiders Technologies text dataset, whose data/model provenance must be audited separately | **Reference only for now:** the model is coupled to FINN/PYNQ fixed-point hardware and does not expose a standard Tesseract/PyTorch checkpoint. Porting would require recovering the quantized layer weights, codec, preprocessing, and CTC decode from `lstm/src/network/fraktur`; do not assume it is interchangeable with `tesseract_lstm` GGUF |

OCR-D’s historical-print catalogue narrows the practical Tesseract targets:

| OCR-D/Tesseract resource | Meaning | Port status |
|---|---|---|
| `deu_latf` (formerly `frk`) | Current German Fraktur language model with some Antiqua coverage | **Directly portable:** use the local `frk.traineddata` LSTM conversion already validated; preserve the upstream name/revision mapping in metadata |
| `Fraktur` | Broader Fraktur script model, including non-German characters and some Antiqua | **Candidate:** obtain the exact `.traineddata`, verify its `lstm` component, then run the same converter/parity/Fraktur regression gates |
| GT4HistOCR-derived models (`GT4HistOCR_*`, `frak2021`, UB Mannheim models) | Historical-print models trained from GT4HistOCR and related German/Fraktur corpora; OCR-D recommends these for broad historical coverage | **Highest next priority:** locate the exact `.traineddata` artifact and license/attribution terms, convert if it contains an LSTM component, and compare against `deu_latf` on our corpus |
| `deu_frak` | Older Tesseract 3 German Fraktur model | **Legacy benchmark only:** OCR-D explicitly says it is no longer recommended; it is not a current LSTM conversion input |

OCR-D documents that Tesseract 4.1+ `.traineddata` files contain an
`unicharset` and neural `lstm` weights, and that models can be combined (for
example `deu+deu_latf` or `Fraktur+Latin`) at an accuracy/runtime cost. The
official Tesseract language data is Apache-2.0, but GT4HistOCR training data is
CC-BY-4.0 and individual UB Mannheim derivatives may have additional
attribution or release conditions. Record the exact model URL, checksum,
training corpus, and license before placing any derivative in the shared
model volume or publishing a GGUF.

Required Fraktur implementation sequence:

1. Add `tesseract-frk-f32.gguf` and a sensitive-head `q8_0` derivative only on
   `$CRISPEMBED_GGUF_DIR`; never commit large weights.
   Apply the same policy to every `tesseract_lstm` language variant: all
   quantized Q8/Q4 artifacts must retain `output.weight` and `output.bias` at
   the source precision (F16 or F32), while only recurrent matrices may be
   quantized. Existing multilingual artifacts must be regenerated if their
   output projection was previously quantized.
2. Add an explicit `tesseract-fraktur` stage/profile using DBNet line crops →
   grayscale crop → `tesseract_lstm` `frk` model, with the normal Tesseract
   path remaining available for modern German.
3. Add a Fraktur regression fixture containing the German title crop and
   full-page `german_official_print.jpg`; compare native GGUF, Python/
   Tesseract, and system Tesseract `-l frk` outputs with CER/WER where an
   oracle exists. Preserve `ſ`, `ß`, ligatures, and Unicode normalization in
   the comparison.
4. Run crispasr-diff-style intermediate parity for the converted `frk` model,
   then test F32, head-only Q8, and debug Q4. Do not publish a quant until
   the Fraktur crop remains readable and the output-layer parity gate passes.
5. Keep `deu_frak` and the Calamari/OCRopus models as separately licensed
   external benchmarks; do not silently convert or redistribute them as
   Apache artifacts.
6. Add OCR-D/GT4HistOCR model comparisons (`deu_latf`, `Fraktur`, and the
   best available `frak2021`/GT4HistOCR derivative) before deciding whether
   `tesseract-frk-f32.gguf` is the production default. Include the Xilinx
   LSTM-PYNQ result only as a hardware-reference baseline unless its weights
   and codec can be legally and technically recovered.

### Fraktur line-classifier retraining (clean-room alternative to AGPL model)

`impresso-project/frakturline-classification-cnn` is useful as a routing
component, not as a recognizer: it classifies a grayscale line crop as
`fraktur` or `other`. Its published architecture is small and reproducible:
approximately 2.1M parameters, input `1x60x800`, three convolutional stages
with ReLU/max-pooling and LayerNorm, adaptive max-pooling to `1x8`, then
`1024 -> 128 -> 1` fully connected layers. The model card reports a 99.75%
accuracy result on its held-out test set, but the model and the linked
Impresso datasets are AGPL-3.0. Do not copy its weights, code, or dataset
into a permissively licensed CrispEmbed artifact.

We can train an independent equivalent classifier from the published
architecture and independently licensed data:

1. Use CC-BY-4.0 GT4HistOCR Fraktur lines and the MIT-licensed
   [`chreul/19th-century-fraktur-OCR`](https://github.com/chreul/19th-century-fraktur-OCR)
   repository's included sample/model-training material for the positive
   class, with attribution and source revision/checksum. The chreul
   repository is MIT-licensed and explicitly supplies small training samples,
   Calamari/OCRopus models, and source-book provenance; that MIT grant does
   not automatically relicense the original historical scans or every
   upstream source corpus, so audit each selected file before redistribution.
   Build the `other` class from separately licensed Antiqua/Latin line corpora,
   or annotate our own public-domain CC0 fixtures. Do not derive a replacement
   dataset by relabeling or mirroring the AGPL Impresso dataset.
2. Split by source publication/page, never by individual crop, to prevent
   font/page leakage. Add hard negatives: ornate Antiqua, mixed lines,
   ornaments, headers, low-DPI scans, skew, bleed-through, and short lines.
3. Reimplement the architecture independently in the existing Python
   training/reference tooling, export an ONNX/PyTorch state dict, and write a
   small GGUF converter with F32 classifier head and F16 convolution weights.
   The native runtime should use the existing crop helper and expose
   `fraktur_probability`, `script_class`, and an abstain/uncertain state.
4. Use the classifier only to route line crops: `fraktur` → `frk`/`deu_latf`
   or a GT4HistOCR-derived Tesseract model; `other` → modern German/Latin
   recognizer. It must not replace recognition or silently alter text.
5. Validate on an independently held-out Fraktur/Antiqua corpus and our
   German page fixtures. Report balanced accuracy, precision/recall/F1,
   calibration, abstention rate, and downstream OCR CER/WER against always
   `frk`, always `deu`, and the classifier route. A 99% classifier score is
   not sufficient if routing worsens recognition.
6. Publish only the independently trained weights and a complete data/
   attribution manifest. Keep the AGPL Impresso model as an external
   benchmark/reference, not as a CrispEmbed dependency.

EasyOCR checkpoint: the CRAFT detector graph now passes Python input, VGG taps,
U-Net feature map, NHWC score-map, and decoded box-count parity on CPU and
Metal; DBNet→EasyOCR crop smoke now runs with the existing cstr/dbnet-ic15
artifact. The next slice explicitly separates detector geometry from ordering:
EasyOCR line grouping, Tesseract/LayoutLM word ordering, and one structured
handoff contract. DBNet Python box/text parity and production orchestration
remain pending.

The reusable `easyocr_pipeline` now exposes the selected `lines` and `words`
policies, runs native DBNet detection plus GPU-resident EasyOCR recognition,
and returns text, detector/recognizer confidence, pixel boxes, reading-order
metadata, and normalized LayoutLM boxes. The model-backed regression produces
12 line records and 98 word records on `scan_strip.png`; Python box/text
reference parity remains a separate pending gate.

The harness-blind postprocessing reference is now explicit: Python EasyOCR
`readtext(detail=1)` JSON is converted by
`tools/easyocr_postprocess_reference.py` into a versioned manifest covering
ordering, crop geometry, text, confidence, and LayoutLM normalization. Its
model-free test passes both line and word policies.

The native `test-easyocr-pipeline` can emit the same manifest schema, and
`tools/compare_easyocr_manifests.py` reports the first record-level mismatch.
The Miniconda runner `tools/run_easyocr_reference_page.py` now produces an
independent Python CRAFT+English Gen-2 `readtext(detail=1)` manifest. On
`scan_strip.png`, Python produced 11 lines while native DBNet→CRNN produced
12. The first mismatch is line 0 (`"They are going to be , encamped near
Brighton"` versus `& They are going to be, encamped near   Brighton`), with
geometry `[62,0,412,25]` versus `[46.97,0,423.54,21.76]`; the rest therefore
cannot be zipped as equivalent lines. Python recognition confidence was
`0.8541` versus native `0.4472` on that first record. Detector confidence is
explicitly unavailable from EasyOCR's public tuple and is not fabricated;
the comparator has `--ignore-detector-confidence` for this case. The manifests
are backed up under `/Volumes/backups/ai/crispembed-gguf/`. Page text/geometry
parity is failed and remains a quality TODO; this is evidence, not a claim
that either detector is universally better.

The detector-independent production handoff is now explicit: `run_regions`
accepts caller-supplied detector boxes and applies the configured lines/words
ordering, crop, recognizer, and LayoutLM normalization path. The model-backed
pipeline test replays the DBNet boxes through this API and matches the normal
98-record run; this validates the boundary, not external Tesseract TSV parity.
The public signature now accepts only `easyocr_layout::region`, keeping
DBNet-specific types inside the implementation. The compile/link proof passes;
the post-merge model replay is currently blocked by the unrelated shared
`ggml` submodule checkout difference and is not claimed green.

The first real page comparison confirms why TSV parity cannot be asserted by
zipping records: native DBNet/CRNN `words` mode emits 98 records, while
Tesseract 5.5.2 `--psm 6` emits 106 TSV words. The first geometry already
differs before recognition (`[46.97,0,62.56,20.88]` native versus
`[50,0,58,19]` TSV), and later indices shift with segmentation. Full-page
Tesseract TSV also reads `Drighton;` on this fixture, whereas the instrumented
internal PSM7 crop reads `Brighton`; page segmentation and crop selection remain
the active parity gate.

The harness-blind CTC/vocabulary/confidence gate is now covered by the native
`easyocr_postprocess` module and `test-easyocr-postprocess`. CTC uses blank 0
with repeated-token collapse, vocabulary entries are 1-based and validated,
and confidence follows EasyOCR's nonblank `custom_mean` formula.

The page-pipeline audit found that EasyOCR's `get_image_list` uses OpenCV
`resize(..., interpolation=1)` (bilinear), whereas the standalone recognizer
fixture was generated with PIL bicubic. An experimental native bilinear
substitution failed the existing diff at `sequence_input`, `bilstm_0`, and
`logits` (`Ea` versus `5a`), so it is not retained in production until a
matching bilinear `-ref.gguf` is regenerated.

An experimental strict port of EasyOCR's horizontal gap thresholds was also
rejected: applied directly to the DBNet artifact it split the 98 fragmented
regions into 26 recognition units instead of the existing 12 line units.
DBNet therefore needs a detector-specific line adapter before those thresholds
can replace the current y-band grouping.

The first DBNet adapter is now explicit in `easyocr_layout`: it preserves the
fragment-tolerant y-band aggregation for DBNet line crops, while word mode
continues to use left-to-right y-band ordering. The adapter is covered by the
layout regression and the model-backed page smoke; horizontal-gap splitting is
still a later detector-specific refinement.

### Tesseract parity status — controlled line proven; page/CLI parity pending

The repository contains a `.traineddata` → GGUF converter, a pure-Python
Tesseract LSTM reference dumper, and `test-tesseract-lstm-diff`. That is an
available validation path, not evidence that the shipped models match the
original Tesseract engine. The controlled exact `eng.traineddata` line run is
now complete; backup artifacts must still be regenerated consistently when
model metadata changes, and page/CLI parity remains separate.

The native implementation is also a line recognizer. Tesseract's page
segmentation, word boundaries, spacing, and reading order are separate
postprocessing behavior. A Python forward pass can prove GGUF/runtime math
against parsed weights, but does not by itself prove full Tesseract CLI parity.
Do not mark the full lane green until page segmentation, spacing, reading order,
and decoded page output are compared with the original engine.

> **Board cleared 2026-07-20** — all 18 previously-listed in-flight items had
> landed; the index + preserved specifics are in `HISTORY.md` "July 20, 2026 —
> PLAN.md active-work board cleared". Add a row here when you START a task; remove
> it when the branch lands.
>
> Completed milestones live in `HISTORY.md`; technical deep-dives in
> `LEARNINGS.md`. This file tracks the current architecture and what is
> still **pending**.

## OPEN TASKS — OCR runtime residency and optimization backlog (2026-08-05)

Derived from the code-verified residency sweep at `9f731fb5`. Full per-engine
tables (which engine computes on CPU vs GPU, why, and what it already has) are
in `PERFORMANCE.md`, section "OCR runtime residency survey" at the top of that
file. Read that section before picking any item here.

The governing finding: **the backend an engine loads weights on is not the
backend it computes on.** Three patterns coexist — a real GPU sched, a
GPU-load/CPU-compute split where the GPU handle is freed after load, and
deliberate all-CPU. Grepping for `crispasr_init_gpu_backend()` misclassifies
the second group entirely.

Every item below keeps the standing A/B rule: interleaved paired runs, report
every pair and the spread, and a new path ships opt-in behind an env gate until
it demonstrably beats the default on quality *and* time.

### R1 — Tesseract recognizer batching + weight/graph reuse [highest evidence]

The only gap with hard numbers on both ends. Recognition is `260-354 ms` of a
~310 ms stage on `scan_strip`, and **`38.34 s` of `38.69 s`** on the German
Fraktur page against official Tesseract's `9.34 s` — already recorded as an
explicit speed *and* quality blocker. Detection is `102 ms` and crop `250 ms`
on that page; the detector is not the bottleneck.

`CRISPEMBED_TESSERACT_REUSE_SCRATCH` exists but its measured variance
(`279.1` vs `282.3 ms` in one pair, `329-338` vs ~`300 ms` in others) is too
wide to claim anything. **Fix the measurement protocol first** (warm runs,
interleaved pairs, per-stage timing) — otherwise R1's own result will be
unreadable too. Then: batch line crops into one recognizer pass, and reuse
weights/scratch across lines. `CRISPEMBED_TESSERACT_WORKERS` already gives
1 -> 690 ms, 4 -> 300 ms, 8 -> 292 ms, i.e. thread-level parallelism has
saturated; the remaining win is per-line work, not more workers.

### R2 — `layout_detect` deformable cross-attention — **PREMISE STALE; real hotspot found + FIXED 2026-08-05**

**The "dominant Phase-2 cost" claim was measured before `2a43e4f4`** (the
2026-07-11 cpu_linear threading) and survived into the survey uncorrected: on
current main the deform loop is **16 ms of an 856 ms Phase 2 (~2%)** on
`scan_page_pd` at `-t 4`. New permanent per-stage timers behind
`CRISPEMBED_LAYOUT_DETECT_BENCH` found the ACTUAL dominant cost: the decoder
**level input projection** (1x1 conv over 8400 tokens; scalar `(n,o,i)` nest,
inner reduction striding `feat_col` by N_lv, single-threaded) — **549 ms,
64% of Phase 2**. Landed on `perf/r2-deform`: rewritten in the `2a43e4f4`
AXPY form + threaded over output rows, byte-identical accumulation order.
Result: level-proj 549 -> 77.6 ms at `-t 1` (contiguity alone, 7.1x) and
~30 ms at `-t 4` (~15x); **Phase 2 846 -> 318 ms median (2.66x)**; whole
layout call 2332 -> ~1700 ms. **CLI region output byte-identical** at both
thread counts. The deform loop itself stays as-is deliberately (16 ms does
not justify restructuring risk). Remaining, re-scoped honestly: **Phase 1
(Metal backbone+encoder, ~1.4 s) is now ~80% of the layout call** — that is a
GPU-graph question (profile split composition/warmup vs steady-state before
touching), not a scalar-island one; and Phase 2's next items are value-proj
(101 ms) + the per-call weight re-dequant/re-transpose/re-upload in the
self-attn ggml block.

### R3 — Promote the CPU-sched formula encoders to a GPU sched

`bttr`, `hmer`, `posformer`, `mixtex`, `flova`, `ppformulanet`, `pix2struct`
all build ggml encoder graphs and then run them on `enc_sched` over a single
`ggml_backend_cpu_init()`. The expensive half of the port (scalar -> graph) is
already done; these are one `sched_new` argument from GPU dispatch.

**Do not batch this.** The DBNet and PP-OCRv6-det verdicts are that conv-heavy
graphs can *lose* on Metal, which is exactly why those two are CPU by default.
One engine, one A/B, one gate at a time. Also fix the stale "prefer GPU
backend" comments above their load sites while touching them.

### R4 — `lightonocr` has no backend gate at all — **DONE 2026-08-05**

**Landed on `perf/conv2d-gemm`** (rider on the R6 branch). The gate exists:
`CRISPEMBED_LIGHTONOCR_GPU=1` opts into `crispasr_init_gpu_backend()` (got_ocr
sched pattern: CPU fallback appended, `ggml_backend_cpu_set_n_threads` sites
guarded behind `ggml_backend_is_cpu`), `CRISPEMBED_LIGHTONOCR_FORCE_CPU=1`
overrides. **Default unchanged and verified**: 0 Metal markers in the default
arm, output byte-identical to the pre-change binary. Metal arm proven live
(`ggml_metal` init in stderr), decoded text IDENTICAL to CPU on
`scan_strip.png` q4_k. First probe (loaded M1, single pair): Metal 7.2 s wall
/ 1.4 s user vs CPU `-t 4` 5.4 s wall / 20.4 s user — **no wall win on the
small fixture, CPU stays the default**; the flip decision now just needs
per-fixture/per-backend pairs. Full numbers in `PERFORMANCE.md` ("R6
conv2d_cpu im2col-tile A/B", rider paragraph). Note: the survey's "31.6 s
cold" did not reproduce on this fixture (~5.4 s CPU warm) — re-measure before
citing it.

### R5 — Decode-step graph caching — measure the overhead fraction FIRST

Still the nominal #1 unrealized lever, and still unproven. When it was actually
built (T14, deepseek) it **won nothing**, because per-step build+alloc was only
1-6% of decode; the result shipped opt-in as `DS2_FAST_DECODE=1`. Remaining
candidates are qwen2vl/granite/smoldocling, which have device-resident KV but
rebuild the decode graph each step.

Required first step: profile build+alloc as a fraction of decode on each
candidate. If it is single-digit percent, close the item for that engine
instead of porting. Also blocked on WebGPU (traps `unreachable`), so any landed
path needs per-backend gating. Note the premise "0 runtimes reuse the built
cgraph" is now false — math_ocr (persistent decode graph), easyocr
(static-shape init-time graph) and ppocrv6 rec (shape-keyed cache) all do.

### R6 — `conv2d_cpu`: per-patch gather -> true im2col+GEMM, and multithread — **BUILT + M1-MEASURED 2026-08-05, x86 arm open**

**Landed on `perf/conv2d-gemm`**: `core_cpu::conv2d_im2col_cpu` — im2col
position tiles + oc-outer loop interchange + fork-join threading, **bitwise
identical to the generic path by construction** (same patch order, same
`dot_product` per element; exact-equality unit guard over 9 shapes at nt=1
and nt=4). Gates `CRISPEMBED_CONV2D_GEMM=1` / `CRISPEMBED_CONV2D_THREADS=N`,
**default OFF per the A/B rule**. M1 verdict (PP-OCRv6 medium scalar det,
interleaved pairs, full table in `PERFORMANCE.md`): **nt=4 wall 2.04x, won
all 5 pairs**; **nt=1 is 4-7% SLOWER** — the M1's 12 MB shared L2 already
holds these weight matrices, so the interchange alone doesn't pay here; the
win available today is threading. Remaining, in order:

1. **Kaggle AVX2 A/B of the same three arms** (small private L2 is where the
   interchange hypothesis should win; also the honest CPU baseline for any
   CUDA/discrete-GPU residency decision). Per the offload directive, not on
   this Mac.
2. Per-engine opt-ins where latency matters (the SR family, DBNet, scalar
   det fallback) — the gate is process-wide today, engines can pass their
   own `n_threads` via `conv2d_im2col_cpu` directly.
3. Register-blocked GEMM micro-kernel — changes accumulation order, so it
   forfeits byte-equality and needs decoded-output A/Bs per engine; only
   worth opening if the x86 arm shows the memory-side win is real.
4. Fold the two private threaded copies (`deepseek_ocr2.cpp:287`,
   `unlimited_ocr.cpp:267`) onto the shared kernel once its default story
   settles.

### R7 — `scunet_denoise` — the missing `DequantCache` — **CLOSED 2026-08-05: measured, not worth it**

The item argued from presence (18 other files have one), not from cost.
Measured on `perf/r7-scunet` (permanent atomic accumulator in `to_f32`,
printed on the `CRISPEMBED_SCUNET_BENCH` total line): on
`scunet-color-f32.gguf` / `scan_strip.png` at `-t 4`, ALL weight `to_f32`
copies sum to **~4-5 ms of a ~4.3 s tile pass (~0.1%)**. A cache would be
dead code; for an f16 artifact the bound is a few times that — still ~1%.
**No DequantCache added; the instrumentation stays** so the number is
re-checkable per artifact. scunet's real cost is the Swin/conv compute itself
(~27 s for a 520x260 image), which belongs to the explicitly-deprioritized
SR-on-GPU research item. Third stale premise found by measure-first this
session (after R2's deform loop and R4's "31.6 s cold").

Prior correction retained: WMSA is window-parallel across `n_threads`
(default follows `-t`).

### R8 — ggml-metal ICB (indirect command buffer) replay

Metal decode is per-op-dispatch bound; CUDA-graph capture already solves the
CUDA side. Highest ceiling of anything on this list and the highest cost —
upstream-shaped work in the pinned `ggml` submodule, not an engine change.
Listed so it is not rediscovered as a cheap idea.

### Explicitly NOT on this list

- **SR-on-GPU as a family.** Already reprioritized down and it stays down:
  `dat/hat/swinir` use `init_best` only to load, then copy dequantized weights
  into a CPU context. There is no GPU sibling to match — this is unsolved
  research (Metal `ggml_conv_2d` + a GPU-resident weight/graph path), not a
  residency toggle.
- **Flipping DBNet or PP-OCRv6 det to GPU.** Both are CPU *by measurement*.
  Re-run the A/B if the conv kernels change (R6), not before.
- **`UOCR_PD=1` gen=2 segfault.** Real and pre-existing, but it is a
  correctness bug on an opt-in path, tracked in the round-7 board row — not a
  performance item.

## OPEN TASKS — OCR optimization roadmap after the R2/R4/R6/R7 session (2026-08-05)

State after that session: R6 built + M1-measured (opt-in), R4 done, R2's real
hotspot fixed 2.66x, R7 closed by measurement. **Meta-finding that governs
everything below: three of the four backlog premises tested did not survive
measurement** (R2's "deform loop dominates" was ~2%, R4's "31.6 s cold" did
not reproduce, R7's cache-by-analogy was 0.1%). Every item here therefore
starts with a measure step, and its "expected outcome" is a forecast to be
checked, not a promise.

**Priority (user-directed, 2026-08-05): work that helps the best runtimes
first** — the document pipeline (PP-OCRv6, layout_detect), Tesseract, and the
production VLM lanes. Agreed order: **O1+O2 (layout) → O13b (PP-OCR profile)
→ O5 (VLM decode measure-first) → O4 (Tesseract R1, own session)**, with the
Kaggle batch (O8+O9) interleaved as the discrete-GPU unlock.

### Lane A — local (M1), immediately actionable

- **O1. Layout Phase 1 profile — DONE 2026-08-05** (`perf/layout-phase1`).
  Answer: the ~1.4 s is **steady-state Metal graph compute** — warm calls
  equal cold (`CRISPEMBED_LAYOUT_REPEAT=N` diagnostic landed), feat readback
  is 5-8 ms (bench line now splits compute/readback), and the earlier
  direct->im2col 9.8x is the state of the art for this graph on M1. The
  "structural >=1.5x" hope did NOT materialize on Metal: the new opt-in
  `LAYOUT_CONV_F16=1` path (F16-dst im2col + F16 mul_mm, composed manually
  because the fork forces F32 im2col for F32 activations and Metal im2col
  rejects F16 sources) measured **SLOWER — 2.2 s vs 1.4 s** — with quality
  fine (same 20 regions, ±0.002 score / ±0.1 px). Kept gated: on CUDA/A1000
  tensor cores this path is the one-env A/B to run first (see O9/O11).
  Remaining M1 lever is per-op Metal kernel work (R8 territory).
- **O2. Layout Phase 2 residue — O2a DONE, O2b open.** O2a (landed, same
  branch): the self-attn block re-read + re-transposed ~5 MB of immutable
  weights per call; now cached per layer, regions byte-identical; saving is
  real but below the loaded dev box's noise floor (claimed as removed work,
  not a ms figure). **O2b (open): value projection on the GPU** — six
  256x256 x 8400-token matmuls, est. 101 -> ~30 ms at `-t 4` (upload 8.6 MB
  once, read back 51 MB, ~10 ms GPU compute); needs a quiet box or
  Kaggle-style pairs to prove.
- **O3. R3 one engine at a time** — flip the formula-encoder `enc_sched`s to
  `{gpu, cpu}` behind per-engine gates (bttr, hmer, posformer, mixtex, flova,
  ppformulanet, pix2struct; several models cached locally). Expected outcome:
  split verdict — conv-heavy DenseNet encoders likely LOSE on Metal (DBNet
  precedent), attention-shaped ones (pix2struct) may win; guaranteed
  deliverable is measurability + dead stale comments, and the gates matter
  because the verdicts likely FLIP on discrete GPUs (see O9/O11).
- **O4. R1 Tesseract recognizer** — the biggest honest gap (38.7 s native vs
  9.3 s official on the Fraktur page; recognition is ~99% of it). Fix the
  measurement protocol first, then per-line batching + weight/scratch reuse
  (thread-level parallelism is already saturated). Expected outcome: least
  predictable; 1.5-3x plausible from batching, full parity probably also
  needs the decoder-semantics quality lane. Multi-session, Fable-tier, never
  delegate the math.
- **O5. R5 measure-first** — profile decode-graph build+alloc as a fraction
  of decode on qwen2vl / granite / smoldocling before porting anything.
  Expected outcome: most likely CLOSES the item for 2-3 engines (deepseek
  precedent: 1-6%); small chance one engine shows a deepseek-like 1.4x where
  the real cost is host<->device bounces rather than build time. Cheap
  (an afternoon), information-dense.
- **O6. `UOCR_PD=1` gen=2 segfault** — correctness, opt-in path, already
  reproduced 7/44. Expected outcome: findable memory-lifetime bug (KV-view /
  `ggml_cont` class is the usual suspect); one session.
- **O7. R6 engine adoption** — pass engine `n_threads` into
  `conv2d_im2col_cpu` on latency-sensitive CPU conv paths (SR family,
  pplcnet, det scalar fallback). Expected outcome (high confidence, already
  measured): ~2x wall on those paths; needs per-engine wiring + pairs.

### Lane B — offloaded (Kaggle), unblocks the discrete-GPU story

- **O8. R6 x86/AVX2 three-arm A/B** (legacy / im2col nt=1 / im2col nt=4).
  Decides the interchange's fate where L2 is small; also the honest CPU
  baseline for any CUDA residency decision. Expected outcome: threading wins
  again; interchange-alone somewhere between neutral and +30% (genuinely
  uncertain — that is the point of the run).
- **O9. CUDA residency re-A/Bs** — DBNet + PP-OCRv6-det conv graphs on a real
  CUDA box, decoded-output roundtrips included (CUDA's stricter contiguity
  asserts make Metal/CPU passes non-transferable). Expected outcome:
  moderately confident the Metal "CPU by measurement" verdicts flip on CUDA,
  which would make them Metal-only defaults and justify O11. **This is the
  single most PP-OCR-relevant open item for discrete-GPU users.**
- **O10. Vulkan bring-up decision** — try Vulkan compute on a Kaggle GPU
  image, or wait for a local discrete-GPU box. Expected outcome: uncertain
  (headless ICDs may not work); the deliverable is knowing where Vulkan can
  be tested at all. Until then "fast on Vulkan" is unverifiable by policy.

### Lane C — architecture, once Lane B lands

- **O11. Per-backend-kind residency defaults** in `gpu_backend_pref.h` —
  default = f(engine, backend kind), so an A1000/CUDA user gets GPU conv
  engines while M1 keeps its measured CPU defaults. Mechanical once O8/O9
  supply verdicts.
- **O12. R8 ggml-metal ICB replay** — upstream-shaped, highest Metal-decode
  ceiling, weeks not days; CUDA already has graph capture so this is
  Metal-only leverage.
- **O13. Backlog hygiene** — (a) re-measure the remaining R-items' premises
  before investing (R1's stage split, R5's per-engine numbers); (b)
  **PP-OCR-specific measure-first profile (O13b)**: det is already ggml-graph
  (6.9 -> 1.0 s medium tier) and rec has the batch-fused shape-keyed graph,
  so the next PP-OCR win is NOT on the old list — profile the current
  det+rec+orientation pipeline end-to-end on the medium tier to find the
  current hotspot (crop/resize? rec width buckets? orientation? det
  postprocess?) before prescribing anything. Expected outcome per this
  session's base rate: the profile will name something the backlog does not.

## OCR pipeline workstream — actionable items

### EasyOCR / LayoutLM compatibility — IN PROGRESS

- Port all EasyOCR detector assets (CRAFT, DBNet18, DBNet50) and all shipped
  Gen-1/Gen-2 CRNN recognizers through the GGUF → `-ref.gguf` →
  `crispembed-diff` → decoded-output protocol.
- Preserve per-artifact source/license metadata; do not infer a fine-tuned
  checkpoint's license solely from its backbone.
- [x] Add a weight-free LayoutLMv2/v3 handoff contract for externally produced
  words and normalized boxes. `tools/validate_layoutlm_handoff.py` emits the
  exact `apply_ocr=False` processor payload and retains confidence/pixel boxes
  as sidecar metadata; Transformers' `apply_ocr=True` path uses PyTesseract.
  A live model invocation remains unnecessary for this contract gate.
- Acceptance requires reference parity, decoded text, and real pipeline output
  checks before quantization or registry integration.

#### OCR interoperability blueprint — selected item

The engines do not share one detector. Tesseract supplies page segmentation and
LSTM line recognition; EasyOCR uses CRAFT by default and optionally DBNet, then
groups boxes into line crops for its recognizer; LayoutLMv2/v3 processors normally
call PyTesseract and consume words plus normalized boxes, while
`apply_ocr=False` accepts an external OCR result. Transformers as a library has
no universal OCR detector: TrOCR is recognizer-only and OCR-free image-language
models use their own visual encoder.

The native boundary is therefore:

`detector (CRAFT | DBNet | external/Tesseract geometry) → boxes/scores →
ordering/grouping policy → word or line crops → recognizer → structured words`

Every production path must emit the same weight-free record: text, pixel box,
confidence, block, line, reading-order index, and optional `[0,1000]`
normalized box. The selected implementation keeps two explicit policies:

- `lines`: EasyOCR-compatible y-grouping and x-order, followed by dynamic-width
  EasyOCR CRNN recognition.
- `words`: Tesseract/LayoutLM-compatible word ordering, preserving individual
  boxes and confidence for downstream processors.

Do not port Tesseract as a DBNet checkpoint. Port its segmentation/order
semantics where useful, retain native CRAFT/DBNet inference, and compare the
policies on the same page fixtures. Acceptance is decoded page text and
downstream handoff parity, not detector-box similarity alone.

#### Interoperability gates

- [x] Make detector output and ordering policy first-class production adapters;
      `easyocr_pipeline::run_regions` now accepts detector-independent boxes and
      applies the selected `lines`/`words` policy through the production crop /
      recognizer path; the pipeline test exercises the injected-geometry handoff.
- [x] Validate `lines` against EasyOCR grouping and decoded line text on a
      page fixture with a Python reference manifest. The live comparison is
      intentionally failed: Python CRAFT yields 11 lines, native DBNet yields
      12, and the first text/geometry/confidence mismatch is recorded above;
      detector/order/crop quality parity remains open.
- [ ] Validate `words` against Tesseract TSV-style geometry/order and preserve
      confidence, pixel boxes, and normalized LayoutLM boxes.
- [x] Add native handoff invariants for word-mode line/x ordering and
      normalized-box bounds; external Tesseract TSV geometry/text parity is
      still pending.
- [x] Add a standard-library Tesseract TSV geometry/order comparator and
      self-test; a real page comparison remains an evidence gate, not a claim
      of Tesseract text parity.
- [ ] Run the same structured words through LayoutLMv2/v3 with
      `apply_ocr=False`; verify serialization and ordering independently of
      logits.
- [x] Keep Tesseract LSTM as a separately measured recognizer lane; the
      `test-ocr-identical-crops` harness feeds the exact same RGB crop to the
      dynamic-width EasyOCR CRNN and grayscale Tesseract LSTM, then reports both
      confidence conventions. On three official TSV boxes, both lanes preserve
      the text structure but differ in ambiguous characters/punctuation (for
      example `I`/`[` and the final quote), confirming these are recognizer
      outputs rather than detector/order differences.
- [x] Prove the controlled line-recognizer boundary separately: the exact
      Homebrew `eng.traineddata` hash, Python `-ref.gguf`, native captures,
      decoded text, and official instrumented PSM7 internal crop all match;
      logits differ by at most `6.6e-7` with cosine `1.000000`.
      New reference dumps record the source image dimensions, and the native
      diff harness rejects a mismatched fixture before reporting stage cosine
      scores; this prevents stale `-ref.gguf` archives from masquerading as
      model or runtime parity failures.
- [ ] Compare page segmentation, spacing, and CLI crop geometry independently;
      this remains open because direct line fixtures are not the same internal
      crops selected by official PSM7.
      The native classical page-segmentation adapter is now wired behind the
      explicit `--tesseract-pageseg`/stage option and has a model-free synthetic
      geometry regression. It now also bypasses generic scan cleanup so page
      segmentation sees original-image coordinates, and its row threshold
      rejects sparse antialiasing bridges on gray paper; this does not close
      real CLI parity.
      On `scan_strip.png`, the tuned native CLI path improved from 3 to 7
      decoded regions, while official Tesseract `--psm 3/6` emits 12 lines;
      exact RGB-to-gray conversion is now shared with the proven reference.
      Height-based splitting now recovers 12 candidates and 12 decoded regions
      on `scan_strip.png`; crop widths are tightened per split band. Text still
      differs on `Meryton` and punctuation/quotes, so decoded page parity and
      official crop equivalence remain open.
      The env-gated page-segmentation path now also bypasses cleanup (the
      structured `page_segmentation` parameter already did), so it preserves
      original-page coordinates. On `scan_strip.png` the component fallback
      recovers 12 regions, 567 native characters, and CER/WER `0.0179/0.0841`
      against official 12-line output; native mean confidence is `0.895` versus
      official mean word confidence `0.9108`. This improves geometry and text
      parity but does not make the experimental path production-default.
      The `CRISPEMBED_TESSERACT_COMPONENT_PAGESEG` control now reaches the
      documented component prototype through the orchestrator; its current
      `scan_strip.png` result is 12 regions, 569 chars, and confidence `0.878`,
      so the legacy/fallback adapter remains the default classical choice.
      Review of Tesseract `textord/makerow.cpp` confirms its authoritative
      boundary is connected blobs assigned by vertical overlap, line size,
      spacing, and fitted baselines; our projection splitter is only an
      interim adapter. An opt-in component prototype is available behind
      `CRISPEMBED_TESSERACT_COMPONENT_PAGESEG`. After a Tesseract-style
      reassociation pass for short/detached blobs it produces the expected 12
      rows on `scan_strip.png`, but its enlarged first-line crop currently
      worsens recognizer output, so it remains experimental and is not enabled
      in production.
      `tools/compare_tesseract_page_geometry.py` now measures the independent
      geometry boundary from official TSV level-4 rows. On `scan_strip.png`
      The legacy component path remains the default because the German
      official-print gate regressed under the newer baseline matcher. With
      `CRISPEMBED_TESSERACT_COMPONENT_BASELINE=1`, the baseline experiment now
      has 12/12 indexed rows with mean IoU `0.813562` after vertical crop
      tightening; the projection fallback has 12/12 with mean IoU `0.865993`.
      Its first-line crop is `[48,0,434,20]` and short final-row crop
      `[27,237,72,22]`; both page ends decode coherently, although the
      baseline variant drops the final exclamation mark. Character choices and
      quote/spacing differences remain a
      decoded-text parity gate. A page-level beam A/B at widths 1, 5, 10, and
      25 keeps the same first-line choices; generic CTC beam search remains
      opt-in and is not the cause of the remaining CLI discrepancy.
      The geometry comparator now passes detector and recognizer models
      separately through the CLI (`--ocr-det`/`--ocr-rec`); its component
      prototype measurement is 12/12 lines with indexed mean IoU `0.826222` on
      `scan_strip.png`, rather than the previous false zero-row result.
      The comparator also accepts optional `--min-native-lines` and `--min-iou`
      gates; the component fixture passes at 12 lines and IoU `0.82`, while
      diagnostic runs without thresholds remain non-gating.
      Geometry reports now include detector/recognizer SHA-256 hashes and the
      indexed reading-order policy, matching the aggregate page-metrics
      provenance record without serializing local model paths.
      The geometry comparator now also reports mean/max absolute crop deltas
      and mean absolute inter-line gap deltas, with optional max-delta gates;
      this keeps spacing/crop drift visible even when row count and IoU pass.
      Its projection/component/baseline policy flags are now mutually
      exclusive and clear inherited policy variables before each run, so an
      A/B result cannot silently use a stale segmentation mode.
      It now reports monotonic reading-order checks for both TSV and native
      boxes and can gate them with `--require-reading-order`; indexed IoU is
      therefore no longer the only signal for an ordering regression.
      `tests/test_tesseract_page_geometry.py` covers the ordering, crop-delta,
      spacing-delta, and equal-count ordering-regression cases without model
      files.
      The regression workflow now runs this test in the Tier 0 smoke job and
      triggers on changes to the comparator test, keeping these gates active
      in PR and `main` CI.
      The aggregate page-metrics harness now selects `legacy-fallback`,
      `component`, `baseline`, or `projection` explicitly and clears stale
      policy environment variables between runs. On `scan_strip.png`, the
      measured CER/WER are `0.0179/0.0841`, `0.0322/0.1121`,
      `0.0179/0.0841`, and `0.0250/0.1121` respectively; legacy-fallback
      remains the best default.
      Each aggregate JSON record now includes detector and recognizer
      SHA-256 hashes plus the ordering-comparison policy, without writing
      local model paths into reports.
      The aggregate harness now supports opt-in minimum-region and maximum
      CER/WER acceptance gates and reports their individual results, allowing
      page-quality regressions to fail explicitly instead of being hidden in
      diagnostic JSON.
      The model-free geometry regression also covers aggregate gate pass/fail
      semantics and verifies that all quality gates remain opt-in.
      It also exercises all four page-segmentation policy labels, keeping the
      legacy default and each experimental mode explicit in reports.
- [x] Record the exact `.traineddata` SHA-256 in both converted Tesseract
      GGUF metadata and dumped reference GGUF metadata; the actual reference
      run and controlled-line stage/output parity are complete; page parity
      remains open.
- [x] Align the diagnostic Tesseract reference dumper and native recognizer
      with the actual Leptonica `pixScaleGrayLI` fixed-16 bilinear contract
      (top-left sampling, integer weights, replicated edges). The previous
      half-pixel resize was wrong and caused the first real divergence on a
      thin receipt crop.
- [x] Trace the receipt discrepancy against upstream Tesseract/Leptonica:
      `commons-receipt-line-3` moved from input cosine `0.990245` and
      `after_conv_fc` `0.983611` to all 9 diff stages at cosine `1.000000`,
      with matching mine/ref norms. Two additional CC0 Commons fixtures are
      vendored with URLs, licenses, and SHA-256 metadata; full CLI page and
      decoder-choice parity remain open.
- [x] Trace the remaining `Brighton`/`Drighton` discrepancy to the exact
      installed model's `training_flags=65`: Tesseract's `TF_INT_MODE` is set,
      so the CLI quantizes inputs and per-output weight rows to int8 before
      its recode beam. GGUF/reference metadata now records `training_flags`
      and `int_mode`; the current F32 graph remains measurable against the
      F32 Python reference but is not yet CLI-logit parity.
- [x] Add the first int8-equivalent Tesseract activation path and an int-mode
      Python `-ref.gguf`; input/intermediate/output boundaries pass the 0.99
      diff gate (int-mode logits cosine `0.997227`). This shifts native output
      to `Lhey ... Drighton`, but does not yet reproduce the CLI's
      `ihey ... Brighton`.
- [x] Add an opt-in CTC prefix beam (`CRISPEMBED_TESSERACT_BEAM_WIDTH`) and
      test widths 2, 3, 5, 10, 16, 25, and 50. It leaves the int-mode result
      unchanged, proving generic CTC beam search alone is not the CLI choice
      mechanism.
- [x] Close the int-mode network logit gap with Tesseract's lookup-table
      nonlinearities and exact quantized matrix arithmetic. Recode/dictionary
      scoring remains a separate pending gate below.
- [x] Added Tesseract's 1/256 LUT nonlinearities and reconstructed per-row
      int8 matrix accumulation. Native/Python int-mode parity improved to
      logits cosine `0.998405` with identical decoded output; CTC and
      Viterbi/recode-style diagnostic beams at widths 2-50 still do not select
      the CLI's `Brighton`.
- [x] Compare native against an instrumented official Tesseract PSM7 run at
      the internal activation boundary. The earliest divergence was the
      `Convolve` layer's out-of-image cells: Tesseract fills them with its
      seeded `TRand`, not zeros. GGUF now preserves `sample_iteration`, and
      both converter/reference/native paths reproduce the exact seeded padding.
      On the 601x36 official PSM7 crop, every captured stage passes at cosine
      1.000000 (logits max error 6.6e-7, cosine 1.000000), and native decodes
      the same `Brighton` path as official Tesseract. This closes arithmetic
      and preprocessing parity for this controlled line; recode/dictionary
      scoring and full-page segmentation remain separate gates.
- [x] Preserve the serialized Tesseract recoder map/offsets in the runtime and
      add an opt-in recoder-prefix legality layer to the diagnostic beam.
      Width-25 constrained decoding reproduces `Brighton` on the official
      crop with all 9 network diff stages still passing. Full RecodeBeamSearch
      certainty aggregation and DAWG dictionary scoring remain pending and
      are not enabled in production.
- [x] Harden `test-tesseract-lstm-diff` so decoded metadata mismatches fail the
      test. The Python reference now matches native `stb_image` RGB-to-gray
      conversion (`(77R+150G+29B)>>8`); all six direct line fixtures pass
      decoded parity with exact input tensors and stage cosines at or above
      `0.998821`.
- [x] Sweep six existing CC0/public-domain English line fixtures through the
      exact int-mode native/Python harness: all 6 references pass the
      captured-stage 0.99 parity gate, and the constrained width-25 beam remains
      stable on the official `Brighton` crop. Official CLI PSM7 output was
      separately measured with `language_model_ngram_on=0` and `=1`; both
      settings produced identical text on all six lines. The remaining
      differences (crop width, spacing, and case) are page-segmentation/CLI
      geometry, not DAWG scoring evidence.
- [x] Run exact hashed Homebrew English references after the Leptonica fix:
      the controlled line fixture decodes identically in native/Python as
      `_ “ ihey are going to be encamped near Drighton ;`, with all 9 stages
      passing the 0.99 gate and logits cosine `0.999863`. The full-page image
      is not a valid single-line recognizer fixture; page segmentation remains
      a separate acceptance gate.
- [ ] Record detector/ordering/recognizer provenance and checkpoint licenses;
      never relabel the cstr DBNet artifact or publish it under another account.

- [x] Fix native Tesseract confidence propagation. The converter/reference
      already expose CTC softmax probabilities, but `src/tesseract_lstm.cpp`
      discarded them and appended `0.0` for every decoded character. Greedy
      decoding now returns the selected timestep probability, so page-level
      confidence is no longer spuriously zero. This was a runtime bug, not a
      GGUF conversion bug.
- [x] Define the native beam confidence contract. A prefix/recode beam output
      is a sequence-level hypothesis assembled across timesteps, so a
      per-character confidence cannot be copied from one timestep. Native now
      exposes a length-normalized CTC sequence probability separately and
      leaves beam `char_conf` empty. Greedy decoding continues to expose
      selected timestep probabilities. This follows Tesseract's distinction
      between character certainty and word-level aggregation; it is not a
      claim of exact `WERD_RES::certainty` parity.
- [ ] Validate beam confidence against the official engine's certainty
      aggregation and implement per-character posterior/marginal scores only
      if that comparison establishes a stable mapping. Keep beam decoding
      opt-in until recoder and DAWG scoring are also matched.
      `test-confidence --tesseract-image MODEL LINE.png` now exercises the
      direct recognizer contract for greedy/beam comparisons without page
      segmentation overhead; it is diagnostic until a non-empty, transcribed
      line fixture is wired into the acceptance gate.
      `tools/compare_tesseract_line_confidence.py` now measures the same
      contract against official TSV on three English line fixtures: greedy text
      matches on two of three, while the first line remains `Lhey`/`Drighton`;
      greedy sequence-vs-official-mean-word confidence deltas were `+0.0053`,
      `-0.0847`, and `-0.0643`. Beam confidence is intentionally reported as a
      separate sequence probability (not a fabricated per-word certainty), so
      this does not close the official certainty gate. Tesseract source review
      now confirms the native greedy `word_confidence` mapping: minimum
      selected-path `log(probability)`, followed by `clamp(100 + 5*certainty,
      0, 100)`. On the direct second-line fixture this is `0.965889` versus
      official `0.959698`; page-level aggregation and beam certainty remain
      open.
      On the available German Fraktur line fixture, official Tesseract
      produces `1` with mean word confidence `0.5886`; native greedy produces
      `GI` with sequence confidence `0.5985` and two timestep confidences;
      native beam-8 produces `GIIEE` with sequence confidence `0.5808` and no
      per-character confidences. The confidence scale is close, but decoded
      text is not yet parity, so beam remains diagnostic and opt-in.
      The diagnostic now also reports native character min/mean probabilities;
      on the F32 artifact these are `0.0902`/`0.1978` for greedy, while beam
      correctly reports zero per-character values and sequence confidence
      `0.5592`. These are diagnostics only, not a production calibration.
      After correcting the aggregation to exclude blank/repeated CTC
      timesteps, native Fraktur greedy `word_confidence` is `0.8797` versus
      official TSV `0.5886`; beam remains `0`. Semantics now match the
      min-emitted-character rule, but calibration and decoded-text parity are
      still open.
      The line-confidence comparator now provides opt-in gates for greedy
      word-confidence calibration and the beam sequence-only contract; its
      model-free tests cover pass, calibration failure, fabricated beam
      character confidence, and missing-beam cases.
      It now records recognizer SHA-256 provenance and official word-confidence
      min/median/max alongside the mean, making calibration spread visible
      without treating those distributions as beam-character parity.
      It now accepts an explicit `--tessdata-dir` for the official subprocess;
      without that path, a stale `TESSDATA_PREFIX` can silently yield zero TSV
      words and invalidate the confidence comparison. With the explicit
      Homebrew tessdata directory on `german-line-tiny.png`, official output is
      `1` at confidence `0.588557`, native greedy is `G` with word confidence
      `0.883064`, and native beam-8 is `GEIEE` with sequence confidence
      `0.535476` and no character confidences. The beam contract passes, but
      text and greedy confidence calibration remain open quality TODOs.
      The comparator also provides `--require-official-words`, and its model-
      free contract test rejects empty official TSV references instead of
      allowing a misconfigured data path to appear to pass.
      It also provides `--require-greedy-text-match`; confidence/beam checks
      must not be reported as OCR-quality acceptance when native text differs.
      On the explicit German tiny-line fixture, enabling both
      `--require-official-words` and `--require-greedy-text-match` correctly
      exits 1 (`official_words_present=true`, `greedy_text_matches=false`),
      preserving the known native `G` versus official `1` quality gap.
      Both page-metrics and line-confidence comparators now emit elapsed
      milliseconds for each official subprocess and native subprocess/line
      run, so quality claims can be paired with measured cost.
      Page metrics can additionally request native detect/group/crop/recognize
      timings with `--benchmark`; official Tesseract remains an external CLI
      timing baseline unless built with matching internal instrumentation.
      The comparator uses the orchestrator-level benchmark switch, keeping
      recognizer-only timing separate from the full detector-to-recognizer
      pipeline timing.

The gated page-segmentation experiment currently gives 21 regions, 1,128
characters, and 0.836 mean confidence on the German official-print fixture.
The projection fallback gives 24 regions, 1,606 characters, and 0.702 mean
confidence. The reproducible TSV comparison reports 25 official lines, 141
words, 881 non-whitespace word characters, and 0.866 mean word confidence;
the native default blob path reports 21 regions, 1,128 characters, 0.836
confidence, CER 0.307, and WER 0.404 against the official text. Neither
experimental path is yet a quality match, so DBNet remains the production
default. The comparison is reproducible with
`tools/compare_tesseract_page_metrics.py`.

The component row-gap sweep (0, 2, 4, 6, and 8 pixels) did not improve the
quality gate: the best alternate produced 23 regions but lower confidence
(0.805). The default row fitter remains unchanged and the tuning variable is
not enabled in production.

The subsequent baseline-row matcher regression produced 49 regions and failed
the native gate. It is now preserved behind
`CRISPEMBED_TESSERACT_COMPONENT_BASELINE`; the legacy component grouping is
the default experimental component path again and restores 21 regions, 1,128
characters, 0.836 confidence, and 78/78 model-gated orchestrator checks.
The model-free orchestrator suite now also guards the two-row component
geometry and reading order, preventing this default/gated-path regression from
returning silently.
The component adapter now falls through to the baseline matcher only when
legacy grouping returns no boxes; this preserves the measured legacy
German-page behavior while avoiding an empty classical result on harder
layouts. The fallback remains behind the classical page-segmentation gate and
DBNet is unchanged.

Latest reproducible German-page rerun (2026-08-02, current `frk` Q8 artifact)
measured official Tesseract at 25 lines/881 chars, confidence `0.8658`, and
`9.34 s`; native measured 23 regions/1,235 chars, confidence `0.768`, CER
`0.5279`, WER `0.5390`, and `38.69 s` native stage. Native per-stage timing was
detect `102.0 ms`, crop `249.7 ms`, and recognize `38,338.3 ms`. This is worse
in both output quality and speed. The older 21-region/`0.307` CER result is
retained as a historical measurement because artifact/control conditions must
be normalized before calling it a regression. TODO: pin the exact recognizer
artifact and benchmark controls in the matrix, then fix the Fraktur page path.

The same fixture/control was then run against the available recognizer
variants. Standard `frk-q8_0` produced 23 regions/1,235 chars, confidence
`0.768`, CER `0.5279`, WER `0.5390`, and native stage `38.69 s`; `frk-f32`
produced 23/1,164, confidence `0.767`, CER `0.4672`, WER `0.5461`, and
`102.41 s`; `frk-int8-source-q8-candidate` matched the F32 output metrics and
produced `64.44 s`. Official output remained 25 lines/881 chars in all runs.
This isolates a precision/artifact quality tradeoff: standard Q8 is faster but
not output-equivalent, while F32/source-Q8 is more accurate on CER but much
slower and still not page-parity. TODO: choose and optimize a high-precision
Fraktur artifact only after same-artifact warm/cold benchmarks and decoded
output gates are stable.

The first reproducible mixed-precision experiment keeps Q8 as the base and
restores only `lstm.0.weight_hh` from F32 with
`models/mix-tesseract-gguf.py`. On the same page it produced 23 regions/1,146
chars, confidence `0.765`, CER `0.4603`, WER `0.5390`, and native recognition
`23.42 s` (detect `58.2 ms`, crop `55.4 ms`); official Tesseract produced
25/881, confidence `0.8658`, and `5.55 s` in that run. This improves CER over
Q8 (`0.5279`) and is much faster than full F32, but remains worse than the
reference in regions, text, confidence, WER, and speed. Keep it gated and
test additional small recurrent critical sets before promotion.

During the next sync, remote `main` had accidentally replaced the 1,025-line
int-mode/scratch implementation with an older 659-line F32-only runtime. The
regression was caught by the scan benchmark: recognition rose to `50.15 s`
for 12 Fraktur lines. The known-good runtime was restored and protected by
`tests/test_tesseract_runtime_contract.py`; the current LUT-enabled int-mode
run is `34.32 s` recognition with unchanged output (12 regions, CER `0.03375`,
WER `0.15044`). This remains slower than official Tesseract and is an active
optimization TODO, but the critical fast path is now guarded against silent
branch overwrites.

The generic quantizer now accepts repeatable `--keep-pattern` fnmatch rules.
This makes critical-weight retention reproducible for every model family while
leaving the default policy unchanged; the Tesseract mixed-precision packer
remains the safer byte-preserving path for existing artifacts.

#### Cross-check survey — quality and cost status

Every checked path must be classified by both decoded output quality and
measured cost; a cosine or successful exit code alone is not an OCR-quality
claim.

- **Controlled Tesseract line / Python reference / native GGUF:** on par for
  the validated exact line contract. The captured stage tensors and logits
  pass the existing `crispembed-diff` gates (cosines at or above `0.99`, with
  the proven controlled run at `1.000000`), and decoded output matches. The
  remaining gap is per-step timing: the diff harness records parity but does
  not yet emit reference and native elapsed time for every graph stage. TODO:
  add stage timing without weakening the cosine gate.
- **English Tesseract line-confidence path:** mixed, not full parity. Greedy
  decoded text matches two of three checked line fixtures; the remaining
  outputs contain `Lhey`/`Drighton` substitutions. Previously measured greedy
  sequence-confidence deltas versus official TSV word confidence were
  `+0.0053`, `-0.0847`, and `-0.0643`. This is worse on the affected lines,
  and the confidence calibration TODO remains open. The comparator now emits
  official/native elapsed milliseconds, but a controlled same-crop timing
  table for all three fixtures is still TODO.
- **German Fraktur line-confidence path:** worse, not on par. Official output
  is `1`; native greedy is `GI`, and native beam-8 is `GIIEE`. Native greedy
  word confidence is `0.8797` versus official TSV `0.5886`; beam correctly
  exposes sequence confidence only and zero character confidences, but that
  semantic contract is not certainty parity. TODO: obtain a transcribed,
  same-crop Fraktur line fixture and validate beam/greedy output and timing
  against the official engine.
- **`scan_strip.png` full-page legacy/fallback path:** close but worse. Both
  paths produce 12 regions; native produces 567 characters versus official
  453, CER is `0.0179`, WER is `0.0841`, and confidence is `0.895` versus
  `0.9108`. The new timed run measured official TSV at `342.6 ms` and native
  DBNet→Tesseract at `1105.2 ms` (`3.2x` slower). TODO: profile detector,
  crop/warp, and recognizer separately; the native path must close this cost
  gap before any default-path promotion.
- **`scan_strip.png` gated alternatives:** projection and baseline preserve
  12/12 rows but remain worse or non-superior in decoded output; measured
  geometry means are projection IoU `0.865993`, baseline IoU `0.813562`, and
  component IoU `0.826222`. Aggregate CER/WER are respectively
  `0.0250/0.1121`, `0.0179/0.0841`, and `0.0322/0.1121` for projection,
  baseline, and component. Per-policy native stage timings are now recorded
  below; retain all modes gated because no alternate is a quality and speed
  improvement.
- **German official-print page:** worse, not on par. The native default
  classical path reports 21 regions, 1,128 characters, confidence `0.836`,
  CER `0.307`, and WER `0.404`; official Tesseract reports 25 lines, 881
  non-whitespace word characters, and confidence `0.866`. Projection reports
  24 regions, 1,606 characters, and confidence `0.702`. TODO: add paired
  reference/native cold-load and warm per-stage timings for this fixture and
  improve segmentation/text before considering promotion.
- **Portfolio engine sweep:** this is a separate cross-engine benchmark, not
  Tesseract parity. The local M1 Metal sweep completed 11 engines, with 2
  timeout/error entries and explicit missing-model/sample statuses. TODO:
  attach the same quality/cost classification to every available engine and
  keep specialist/VLM outputs separate from plain-text OCR.

The first per-stage `scan_strip.png` benchmark now exists for all four native
policies. Official Tesseract TSV elapsed times were approximately
`315.9–349.9 ms`; native pipeline stage totals were legacy `310.7 ms`,
component `266.8 ms`, baseline `282.2 ms`, and projection `360.1 ms`.
Recognizer time dominated each native stage (`260.3–353.8 ms`), while native
detect and crop were each about `3–4 ms`. The comparator subprocess elapsed
times are higher than those stage totals because the model-gated test binary
also performs setup/fixture work; they must not be presented as pure OCR
latency. Official internal detect/recognize split timings are still unavailable
from the stock CLI, so a per-step apples-to-apples speed claim remains TODO.
The quality/cost table is currently: legacy best quality and close stage cost;
baseline same measured CER/WER but not a geometry/decoded-output improvement;
component worse CER/WER; projection slower and worse CER/WER. None is ready for
default promotion.

The current line diagnostic on the available Fraktur full-page/PSM7 input
measured official `278.6 ms`, native greedy `244.3 ms`, and native beam `110.3
ms`; these are not a same-crop benchmark and must not be used as a speed claim.
TODO: repeat timing on the cleared transcribed line fixtures and report cold
load, warm greedy, warm beam, and per-stage reference/native costs together.

The worker sweep on `scan_strip.png` preserves CER/WER while reducing native
stage total from `690.3 ms` at one worker to `300.7 ms` at four and `292.1 ms`
at eight. The next performance implementation task is therefore recognizer
batching/graph and immutable-weight reuse; detector and crop are only a few
milliseconds in the instrumented path. Do not trade away the recorded output
quality gates while optimizing this hotspot.

An activation-scratch reuse prototype was measured against the existing path:
it preserves CER/WER exactly. A paired run measured `279.1 ms` scratch versus
`282.3 ms` default, but earlier repeated runs were roughly `329–338 ms` versus
the prior `~300 ms` four-worker measurement; this is not a reliable gain. It
is retained behind `CRISPEMBED_TESSERACT_REUSE_SCRATCH` and is disabled by
default until a repeated controlled benchmark demonstrates improvement.
`tools/benchmark_tesseract_page.py` now automates repeated policy/worker runs
and summarizes median/p90 official CLI, native subprocess, native stage, and
recognizer timings alongside CER/WER/confidence deltas.

Beam width 8 is not a performance candidate on this workload: the live
full-page run reached several seconds to tens of seconds per line, versus the
normal greedy recognizer's sub-second-to-low-single-digit-second line timings.
The beam remains an opt-in diagnostic path until recoder/DAWG parity and a
usable latency budget are both demonstrated.

English Gen-2 now has a committed `test-easyocr-diff` harness, passes the agreed
0.99 per-stage cosine gate in F32 and folded-F16 forms, and decodes `5a`.
The VGG graph now derives feature channels and sequence width dynamically for
the remaining VGG recognizer variants.
Gen1 ResNet naming/folding and its residual graph path now pass a real Latin
Gen-1 checkpoint through conversion, Python reference dumping, per-stage
`crispembed-diff` (including magnitudes), and decoded-output parity (`=#4#4#`).

This workstream is informed by the external document-parser comparison, but keeps CrispEmbed's
ggml portability (CPU, CUDA, Metal, Vulkan, and WASM). Items are scoped so each
can land and be measured independently.

### O1 — Restore a trustworthy OCR baseline [COMPLETED]

- Fix duplicate region emission in the batched DBNet + TrOCR path.
- Add a regression test for one output region per detected region and no
  duplicated reading-order text.
- Record baseline latency and region/text counts in `PERFORMANCE.md`.

**Started:** DBNet postprocessing now handles degenerate one-point contours;
the local fox fixture improves from 0 to 10 detected regions. The remaining
baseline work is an automated model-backed assertion and sequential/batched
comparison.

**Done when:** batch and sequential recognition produce equivalent region counts
and no duplicate text on the OCR fixture set. The benchmark harness now accepts
`--expect-regions` and repeated `--expect-text` assertions for CI.

### O2 — Define a structured document result contract [COMPLETED]

- Add a C++ `ocr_document` result containing page dimensions, text regions,
  layout regions, tables, formulas, confidence, and engine provenance.
- Keep the existing orchestrator result and C API source-compatible; provide an
  adapter first, then migrate callers.
- Add serialization tests for empty, text-only, and mixed structured results.

**Started:** `ocr_orchestrator::result` now carries page dimensions and optional
layout regions. Layout inference is lazy and remains disabled unless
`config.layout_model` is set; existing callers and default latency are unchanged.

**Done when:** callers can consume one structured result without depending on a
specific OCR engine.

### O3 — Add CPU-only region routing after layout detection [COMPLETED]

- Introduce a pure routing module with `text`, `table`, `formula`, and
  `fallback` destinations.
- Route by layout label, confidence tier, containment/overlap, and explicit
  per-request feature policy; suppress duplicate text when a specialized
  recognizer owns a region.
- Unit-test every decision seam without model weights.

**Started:** `ocr_orchestrator::result` now carries the model-free routing plan;
table/formula/image policy is explicit in `config` and text-only by default.

**Done when:** a synthetic page produces a deterministic routing plan and the
existing specialized engines can be dispatched from it.

### O4 — Remove temporary image files from stage handoffs [COMPLETED]

- Add an in-memory RGB image/crop view shared by cleanup, detection, and
  recognizers; retain file APIs as load-and-forward wrappers.
- Make cleanup output ownership explicit and avoid unnecessary copies.

**Started:** `ocr_detect::detect_rgb` and `ocr_pipeline::run_raw` now accept
borrowed interleaved pixels; file APIs forward through them. The orchestrator
cleanup handoff still uses a temporary PNG and is the next O4 slice.

**Done when:** cleanup → detection/recognition runs without creating
`/tmp/crispembed_ocr_*.png`, with CPU/Metal output parity.

### O5 — Make capabilities and failures explicit [COMPLETED]

- Add an OCR capability query for loaded engines, languages, output types, and
  structure stages.
- Validate incompatible requests before inference; use stable errors instead of
  silent empty structure results.
- Add image dimension/pixel guards and per-item batch error isolation.

**Started:** enabling table/formula routing now fails at initialization unless
the required layout and specialized GGUF backends are configured.

**Done when:** every advertised feature is executable or rejected with a stable,
test-covered reason.

### O6 — Add reusable pipeline pooling and batch execution [COMPLETED]

- Define a bounded OCR pipeline pool for server use; retain the current path for
  single-threaded and WASM builds.
- Batch compatible crop recognition, cap batch size, and isolate bad inputs.
- Add queue/deadline metrics before changing defaults.

**Started:** DBNet+TrOCR inference contexts now serialize mutable decoder state
with an internal mutex, preventing concurrent callers from corrupting KV/cache
state. `ocr_pipeline_pool` now provides bounded isolated contexts with blocking
slot acquisition. The basic C OCR API selects the pool size from
`CRISPEMBED_OCR_POOL_SIZE` (default `1`); server-level queue/deadline metrics
remain a follow-up operational enhancement.

**Done when:** concurrent requests do not share mutable decoder state and batch
  throughput improves without changing decoded text.

### O7 — Establish unified accuracy/performance gates [COMPLETED]

- Add fixtures for receipt, form, dense page, screenshot, photo, table, and
  formula workloads.
- Measure CER/WER or exact-match, region recall, structure accuracy, p50/p95
  latency, memory, and batch throughput.
- Add regression thresholds and decoded-output checks for optimizations.

**Started:** `tests/ocr_benchmark.py` runs the real detector and pipeline test
binaries and reports region counts, decoded regions, and stage timings as text
or JSON. It uses local GGUFs and does not download models implicitly.

**Done when:** one reproducible command reports OCR quality and cost, suitable
for CI. **Complete:** `tests/ocr_benchmark.py` provides this command and JSON
output.

### O8 — Make corpus provenance and real-world coverage explicit [IN PROGRESS]

- Keep deterministic/reference fixtures for unit and per-stage parity checks,
  but do not use them as claims about real-world OCR quality.
- Add at least one public-domain/CC0 input for every production stage: text
  detection/recognition, layout, tables, cleanup, orientation, handwriting,
  multilingual routing, super-resolution, PDF routing, formulas, and OMR.
- Record source page, license, URL, SHA-256, and annotation status for every
  vendored asset. Add derived rotation/skew variants only from public-domain
  inputs.
- Acquire larger CC0 receipt and Arabic document sets separately, with a
  documented acceptance/download step instead of silently bundling them.

**Started:** `tests/regression/cc0_sources.json`, the fetched
`tests/regression/images/cc0/` seed set, and `corpus_manifest.json` now cover
receipts, forms, tables, Arabic printed/handwritten text, handwriting, cleanup,
orientation, layout, specialist lanes, and a dedicated German lane (modern
photo document, historical German print, and Kurrent handwriting). Gold
transcription review remains open for these robustness fixtures.

### O9 — Benchmark every available engine on shared inputs [IN PROGRESS]

- Use the checked-in regression manifest to enumerate every engine with a
  sample and local GGUF; report missing samples/models as explicit skips.
- Record cold load and warm inference time, return status, output excerpt, and
  CER/exact match when a gold transcription exists.
- Keep full-page VLM, ordinary OCR, math, and OMR scores separate; do not
  compare specialist outputs as plain-text OCR.
- Run the same engine sweep on the public-domain corpus after human gold
  annotations land, then add per-engine quality/latency thresholds.
- Maintain a complete matrix for model-backed engines even when a GGUF is not
  cached; the benchmark can fetch the manifest-pinned artifact with
  `--download-missing`.

**Started:** `tests/ocr_engine_benchmark.py` completed the local M1 Metal
sweep: 11 engines completed, 2 timed out/errored, and the remaining entries
were explicitly reported as missing samples, missing models, or model-needed
ports. SmolDocling is live-tested; Tesseract-LSTM is measured through DBNet
line crops; Unlimited-OCR is being fetched for its live run. Tesseract-LSTM
and PARSeq are present as recognizer-only rows; the DBNet+TrOCR document
baseline is measured separately by `tests/ocr_benchmark.py`. Results are
written as JSON with no silent omission. Unlimited-OCR subsequently completed
on M1 Metal from the system volume in 45,967 ms with correct two-region text
output; its GGUF was restored to the backup volume afterward.

The checked-in matrix now has a model-free CI coverage guard at
`tests/regression/test_engine_matrix.py`: all 23 portfolio engines must remain
present with a lane, runtime, fixture, and explicit availability status.

#### O9 measured survey — native vs reference status (2026-08-01)

This is the current evidence boundary. Timings below are cold end-to-end M1
Metal process times unless marked otherwise; they include model load and are
not directly comparable to warm service throughput. “Exact” means exact output
against the pinned reference/fixture, not a claim of SOTA quality. Missing
reference gold or a missing model is recorded as unmeasured, not as a pass.

| Path | Native timing / reference comparison | Output quality | Explicit follow-up |
|---|---|---|---|
| DBNet + TrOCR | 8.05 s cold; ~5.0 s warm on fox; DBNet postprocess optimization reduced 43.3 s → 1.54 s | 10/10 regions and 10/10 recognized; ordinary document baseline | Add shared German CC0 CER/WER and warm p50/p95 against the Python/onnxruntime references |
| Tesseract-LSTM via DBNet line crops | 7.55 s cold / 8.09 s warm in the first sweep; expanded run 32.0 s cold due process/model conditions | Line crop CER 0.040; controlled line arithmetic/reference parity is exact, but page path is worse: 21 native regions vs 25 official lines, CER 0.307/WER 0.404 | Normalize benchmark conditions; match page segmentation, crop geometry, spacing, and CLI decode |
| PARSeq | 0.921 s first sweep / 6.25 s expanded cold | Recognizer-only smoke (`Gooducalicanos.com`), no gold quality score | Add line-crop harness and CER/exact-match against scene-text gold |
| GOT-OCR2 | 15.662 s / 22.073 s cold | Exact fox transcript | Warm/p95 benchmark and full German CC0 quality lane |
| GLM-OCR | 32.884 s / 38.086 s cold | Exact fox transcript | Warm/p95 benchmark and German CC0 quality lane |
| InternVL2-1B | 24.908 s / 28.523 s cold | Worse than reference presentation: CER 0.540 and prompt text included | Strip prompt wrapper, compare normalized text, then optimize tile/vision residency |
| Qwen2-VL-3B | 70.757 s / 90.113 s before timeout | No transcript within the 45 s budget; quality unscored | Model-size/quantization tier benchmark and timeout budget decision |
| LightOnOCR | 31.561 s / 69.289 s cold | Plausible but unscored | Add gold transcription and separate prompt-wrapper normalization |
| SmolDocling | 16.334 s expanded | Worse: duplicated DocTags regions, payload CER 0.86 | Deduplicate/parse DocTags before any speed work |
| Unlimited-OCR | 45.967 s system-volume run; 40.391 s `UOCR_MMAP=1` backup-volume run | Correct two-region output; CER 0.010, one harmless title-box coordinate drift | Keep mmap path; split SAM/CLIP/projection/decoder timings and compare warm runs |
| MixTeX | 7.523 s / 13.286 s cold | Exact specialist LaTeX | Add warm/p95 and German/CN math fixtures; window attention remains CPU-scheduled |
| Flova | 16.153 s / 36.293 s cold | Exact specialist LilyPond | Add warm/p95 and handwritten German/music fixture coverage |
| Pix2TeX / Texteller | Pix2TeX 5.520 s / 8.980 s; Texteller 11.403 s / 18.491 s | Pix2TeX exact; Texteller worse/unusable (CER 7.293) | Keep Texteller out of quality tier; add formula-domain error analysis |
| SMT | 0.37 s incremental vs 1.98 s full recompute for ~100 tokens (5.4× faster) | 96.3% GrandStaff; native/HF token agreement 100% on validated samples; q8_0 exact, q4_k ~32% and rejected | Keep F32/Q8; do not expose Q4_K as quality tier |
| Polyphonic-TrOMR / Flova OMR / Transcoda | No common live timing in the O9 sweep | Per-stage/decoded parity documented; TrOMR and Flova byte-exact on validated references; Transcoda model lane still missing in the matrix run | Add common real-photo/historical-score warm/cold benchmark |
| PP-FormulaNet, HMER, BTTR, PosFormer, Texo | No common live timing in the O9 sweep | Stage parity exists for selected fixtures; no shared gold quality/throughput claim | Run model-backed math benchmark with exact LaTeX and CER/exact-match |
| PP-OCRv6 pipeline | German CC0 Metal: detector 6.9 s, quad crop 3.4 ms, CPU orientation 358.6 ms, recognition 455.2 ms; tiny graph diagnostic was previously 3.46 s vs 2.44 s CPU on `HI` crop; small/medium stem+backbone graph warm time is CPU 10.377/10.791 s versus Metal 0.222/0.650 s on the fox crop; full gated SVTR graph is 0.214/0.512 s on Metal | CPU accept-gate pipeline: 30 regions/139 chars, confidence 0.93, 141/141 smoke; detector graph remains worse (cos 0.99113, 31 vs 30 boxes); tiny recognizer graph reaches accepted-output parity on CPU and Metal for `HI`/fox; small/medium full SVTR graph reaches logits cos 0.999982/0.999986 on Metal with unchanged decoded output; three additional Arabic/receipt/German direct fixture runs have CPU fallback=CPU graph=Metal graph text | Fix detector box parity; add gold-backed cropped-line references and automate multi-fixture graph acceptance before changing the default gate |
| PP-LCNet orientation | CPU 0.36 s vs explicit Metal graph 1.15 s for 30 crops | 9/10 Metal parity fixtures; one uneven-illumination Arabic outlier (1.0679/3.2166 logits); CPU graph passes | Resolve Metal SE/depthwise numerical drift or keep CPU default; measure warm reuse only after reliability |
| EasyOCR/DBNet/LayoutLM handoff | No external Python page timing yet | Native model-backed run: 12 lines/98 words; geometry IoU 0.916 against TSV baseline, but native emits 98 vs Tesseract 106 words; external EasyOCR text parity pending | Install/reference Python environment and compare full manifests, CER, geometry, and warm timings |

Portfolio-wide TODOs created by this survey: (1) normalize cold-load versus
warm-inference measurement and report p50/p95; (2) attach a gold transcription
or exact structured reference to every runnable lane; (3) run the explicit
`model-needed` lanes with `--download-missing`; (4) do not optimize a lane
whose output is currently worse until its structural postprocessing/parity
failure is fixed; and (5) keep specialist metrics separate from plain-text OCR.

### O10 — Preprocessing inventory, parity, and live outcome gates [IN PROGRESS]

The OCR front-end needs its own measured regression track. Our restoration
inventory is broader than the lightweight OCR reference pipelines, but we are
missing several inexpensive geometry/orientation safeguards that often matter
more than another restoration model.

#### Existing CrispEmbed preprocessing

- Classical scan cleanup: dual-detector deskew consensus, border/content crop,
  background whitening, Otsu/Sauvola binarization, and fast binary morphology.
- Page analysis: PDF effective-DPI profiling, page split detection, content
  bounding box detection, source-type classification, and classical dewarp.
- Orientation: heuristic 0°/180° text-crop correction and rotated detection
  boxes; no learned page-orientation model yet.
- Learned restoration: NAFNet denoise, SCUNet, Restormer, InstructIR, and
  AdaIR.
- Super-resolution: PAN, TBSRN, HAT, DAT, ESRGAN, SwinIR, and SAFMN.
- Learned/classical dewarp: TPS dewarp and the classical baseline.
- VLM policy: full-page VLMs skip destructive scan cleanup and perform their
  own letterboxing/resizing; variable-resolution VLMs honor the max-pixels
  budget.

#### Reference capabilities to reproduce or explicitly reject

- Detector geometry: configurable minimum/maximum side limits, short-side
  target sizing, minimum-height padding, wide/short-image padding, and
  aspect-ratio-preserving letterbox policy.
- DB postprocessing: configurable segmentation threshold, box threshold,
  unclip ratio, optional dilation, candidate cap, and fast/accurate score mode.
- Line orientation: a dedicated 0°/180° classifier, confidence threshold, and
  an explicit all-lines mode for mixed-orientation documents.
- Page orientation: a learned 0°/90°/180°/270° classifier for PDF pages and
  photographed pages.
- Crop preparation: one shared policy for classifier geometry, recognizer
  geometry, aspect-preserving padding, and full-resolution recognition crops.
- PDF ingestion: native page rendering, page-image rotation, worker-pool
  accumulation, and the same preprocessing/OCR path as image inputs.
- Operational controls: per-stage enable/disable flags, hard errors for
  unavailable optional stages, request deadlines, and stage-level metrics.

#### Implementation slices

1. **O10.1 — Live preprocessor benchmark harness.** **Implemented in
   `feat/ppocr-next-20260731`; merged to `main`.** Add
   `tests/ocr_preprocessor_benchmark.py`. For every real CC0/German fixture,
   run raw input, classical cleanup variants, deskew, binarization, dewarp,
   denoise, and every locally available SR/restoration model. The harness
   accepts `--include-derived` to sweep every traced O10.2 robustness variant.
   Record stage
   latency, output dimensions, pixel statistics, detector regions, OCR text,
   confidence, and CER/exact match when gold text exists. Also report text
   delta versus the raw-image baseline when no verified gold transcription is
   available. Synthetic degradations remain unit stress tests, not quality
   claims.

2. **O10.2 — Problematic-input corpus.** **Implemented in
   `feat/ppocr-next-20260731`;** extend the public-domain corpus with
   verified derived variants: ±4°/±8° skew, dark border, uneven illumination,
   haze, speckle, low-DPI downsample, JPEG damage, 90°/180°/270° rotation,
   perspective/curved-page distortion, and mixed upright/upside-down lines.
   Every derived file must retain its parent SHA-256 and transformation recipe.

3. **O10.3 — Detector geometry policy.** **Implemented in `cf5f79b` and the
   follow-up routed-stage wiring.** Add a shared configuration object and
   C API fields for `min_side_len`, `max_side_len`, `min_height`,
   `width_height_ratio`, padding mode, `unclip_ratio`, dilation, score mode,
   and candidate cap. Default to safe current behavior; expose compatibility
   presets for short text strips, wide receipts, dense scans, and photos.
   The detector now provides `detect_options`/`rapid_defaults`, and the C API
   stage struct forwards these controls through DBNet, Tesseract, and PARSeq
   detector paths. Fast mode avoids pathological contour tracing; accurate
   polygon scoring remains available explicitly.

4. **O10.4 — Learned line orientation.** **Classical telemetry slice implemented
   in `feat/ppocr-next-20260731`;** port a small permissively licensed
   0°/180° line-angle classifier to GGUF/ggml. Integrate it after detection
   and before every line recognizer, including Tesseract-LSTM crops. Retain
   the current heuristic as a no-model fallback. Add per-line angle,
   confidence, and whether a rotation was applied to structured results.
   The existing classical 0°/180° safeguard is now shared by TrOCR,
   Tesseract-LSTM, and PARSeq line crops, and structured angle/confidence
   metadata is exposed per region. The learned classifier remains outstanding.

5. **O10.5 — Learned page orientation.** **Model-free fallback implemented in
   `feat/ppocr-next-20260731`;** port a small four-way page-orientation
   model. Apply it before PDF/image routing only when confidence clears a
   configurable threshold. Never rotate VLM inputs implicitly unless the
   caller enables the option, because VLM letterboxing is model-specific. The
   fallback exposes `crispembed_detect_page_orientation` and
   `/preprocess/orientation`, and never rotates
   input implicitly; a learned PP-LCNet classifier remains outstanding.

6. **O10.6 — Shared crop preprocessing.** **Implemented in
   `feat/ppocr-next-20260731`;** consolidate classifier and
   recognizer crop resizing/padding into one tested module. Support
   aspect-preserving and stretch modes, fixed height, maximum width, and
   grayscale/RGB contracts. Add parity fixtures for short, tall, wide,
   upside-down, and tightly clipped lines.

7. **O10.7 — PDF render/autorotate path.** **Image-page autorotation and
   multi-page DPI API slices
   implemented in `feat/ppocr-next-20260731`;** add native page rendering and
   page-level accumulation where the platform supports it. Reuse PDF DPI
   profiling to select render DPI, then apply page orientation and the normal
   document pipeline. `/ocr/document` now accepts explicit
   `"autorotate": true`, applies the confidence-gated fallback, and hands
   rotated temporary pages through the normal renderer. Keep the existing
   parser-only path for minimal builds. Bindings can now request all-page DPI
   metadata with ownership-safe `crispembed_pdf_all_pages_dpi` APIs.

8. **O10.8 — Stage routing, safeguards, and metrics.** **Benchmark accept-gate,
   VLM cleanup safeguard, and structured stage-metrics slices implemented in
   `feat/ppocr-next-20260731`;**
   **benchmark accept-gate slice
   implemented in `feat/ppocr-next-20260731`;** make preprocessing selection
   evidence-based: classical cleanup for scans, no destructive cleanup for
   VLMs/photos, denoise for noisy captures, SR only for low-DPI inputs, and
   orientation only above confidence thresholds. Add accept-gate comparisons
   so a preprocessor is rejected when it lowers confidence or worsens CER
   beyond the configured tolerance. The benchmark now records input/output
   checksums and dimensions plus conservative helped/neutral/harmed/
   unavailable/error outcome labels; changed text without verified gold is
   reported as unavailable rather than claimed as an improvement. Full-page
   VLM stages now skip destructive cleanup unless a future explicit VLM
   preprocessing override is added. The native/C pipeline API now exposes
   per-stage elapsed time, cleanup-applied flag, gate result, text yield, and
   confidence.

#### Required benchmark output

Each fixture/stage row must include:

- input and output dimensions, channels, and file checksum;
- cold load time, warm stage time, and peak/working-set estimate where
  available;
- detector box count, recognized region count, mean confidence, and text;

The live engine benchmark now includes both English and German
Tesseract-LSTM detector+line-crop rows, with the German model downloaded from
the pinned permissive registry entry when requested. Tesseract benchmark rows
use the F16 DB detector rather than a policy-disallowed Q4 detector.
- gold CER/exact match when verified, otherwise raw-baseline text delta;
- `helped`, `neutral`, `harmed`, `unavailable`, or `error` classification;
- stderr tail and stable failure reason for model/backend failures.

#### Acceptance gates

- Every production preprocessor has at least one real CC0/German live fixture.
- Every problematic-input variant runs through raw plus all applicable stages.
- No default preprocessor may worsen verified CER beyond its configured gate.
- A stage that cannot run is reported explicitly; it is never silently skipped.
- Orientation, geometry, cleanup, and restoration effects are reported
  separately, so a strong recognizer cannot hide a harmful preprocessor.
- Results are reproducible from one command and committed as benchmark JSON;
  large GGUFs support the external-volume no-copy path via `UOCR_MMAP=1`.

#### O10.9 — Named model candidates, licensing, and quality tiers

License policy: MIT, Apache-2.0, BSD-2/3-Clause, ISC, and similarly
permissive licenses are acceptable candidates for the core distribution. Do
not add NC/ND, research-only, or unclear checkpoint artifacts to the default
model registry. Repository-code licensing and pretrained-weight licensing must
be recorded separately before publishing a GGUF.

| Stage | Candidate model | License/reuse status | Quality position | Decision |
|---|---|---|---|---|
| Page orientation | `PP-LCNet_x1_0_doc_ori` | PaddleOCR is Apache-2.0; verify the exact exported checkpoint terms | Strong practical choice: four-way 0°/90°/180°/270° classifier, official docs report 99.06% on its test set | First port candidate |
| Line orientation | `PP-LCNet_x1_0_textline_ori` | Same Apache-2.0 project/weight-provenance audit required | Strong practical choice for per-line 0°/180° correction | First port candidate |
| Text detection | `PP-OCRv6` det | Apache-2.0 PaddleOCR code; model artifact provenance must be pinned and audited | Current practical high-quality/throughput baseline; supports multilingual deployment | Port/benchmark when PP-OCRv6 branch lands |
| Text recognition | `PP-OCRv6` rec | Same code/weight distinction as detector | Current practical high-quality/throughput baseline; one unified family is preferable to many language-specific recognizers | Port/benchmark with detector |
| Text detection fallback | `cstr/dbnet-ic15-GGUF` (`DBNet` ResNet-18) | Apache-2.0 declared for the converted artifact; source and dataset provenance documented; Challenge 4 is distinct from ICDAR2015-TextSR ODbL | Mature, reliable fallback; generally below current PP-OCR quality on difficult documents | Cleared; Q8/F16 default, Q4_K debug-only |
| Denoising | `NAFNet` | Upstream repository/checkpoint terms require explicit audit before redistribution | Strong efficient restoration baseline; upstream describes it as state-of-the-art for its restoration tasks | Keep only with artifact audit |
| Denoising/deblurring | `Restormer` | MIT repository license | Strong high-resolution denoising/deblurring/deraining model; official repo calls it SOTA for those tasks | Safe preferred learned restorer |
| Denoising | `SCUNet` | Verify upstream repository and checkpoint terms before registry inclusion | Lightweight practical denoiser, attractive for CPU/Metal | Keep as optional pending audit |
| Super-resolution | `HAT` / `HAT-S` | Apache-2.0 repository | Stronger quality-oriented SR candidate; HAT-S is a useful smaller tier | Safe preferred quality SR candidate |
| Super-resolution | `SwinIR` | Apache-2.0 repository; verify model-data/checkpoint terms | Strong broad baseline for classical/real-world SR, denoising, and JPEG artifact reduction | Safe preferred general SR candidate |
| Super-resolution | `Real-ESRGAN` | BSD-3-Clause repository; each released checkpoint still needs provenance audit | Strong practical real-world SR, but can hallucinate texture and harm OCR | Optional photo-only fallback, never unconditional |
| Super-resolution | `DAT` | Use only an explicitly permissive checkpoint/export; otherwise audit-required | High-quality transformer SR candidate, heavier than HAT-S/SwinIR | Optional quality tier after license audit |
| Super-resolution | `PAN` | Apache-2.0 model card/export candidate available | Very small and fast; useful low-resolution baseline, not current SOTA | Keep as fast CPU tier |
| Super-resolution | `SAFMN` | Apache-2.0 source/export; current GGUF card records Apache-2.0 | Excellent efficiency/size tradeoff, not absolute SOTA | Keep as default lightweight SR candidate |
| Text SR | `TBSRN` | Checkpoint license/provenance must be audited | Text-focused SR is more relevant to OCR than generic photorealistic SR | Keep only behind OCR CER gate |
| Learned dewarp | `UVDoc` / document-unwarping models | Candidate only after exact checkpoint license audit | Better fit than generic image restoration for curved pages | Prefer classical dewarp first; port if real fixtures show need |
| PDF orientation/render | PDFium + `PP-LCNet_x1_0_doc_ori` | PDFium and classifier terms must be retained in notices; classifier checkpoint audit required | Strong operational solution rather than an image-restoration model | Implement native render/autorotate path |

#### O10.9a — MMOCR model/checkpoint license audit

MMOCR's repository is Apache-2.0, but that is the license of the toolbox
code, not automatically the license of every downloaded checkpoint. The
official model-zoo page lists 48 checkpoints and links the weights separately,
without providing a per-checkpoint SPDX grant. Therefore every MMOCR weight
below is **audit-required**, unless a separately verified checkpoint license
is recorded in `tests/regression/manifest.json` and the model registry.

| MMOCR model family | Checkpoints in the official zoo | Current reuse status | CrispEmbed decision |
|---|---|---|---|
| DBNet | ResNet-18/50, DCNv2, oCLIP; ICDAR2015/SynthText/TotalText | The specific `cstr/dbnet-ic15-GGUF` ResNet-18 artifact declares Apache-2.0 and documents MMOCR + ICDAR2015 Incidental Scene Text provenance; other zoo checkpoints remain separate audits | Existing port remains; Q8/F16 default | Cleared for the specific cstr artifact; do not generalize to every zoo checkpoint |
| DBNet++ | ResNet-50, DCNv2, oCLIP; ICDAR2015 | Same checkpoint uncertainty; stronger detector but larger | Optional quality tier only after audit |
| Mask R-CNN | CTW1500/ICDAR2015, ResNet-50/oCLIP | Code/framework may be Apache-2.0; checkpoint and backbone provenance unresolved | Do not add yet |
| DRRG | CTW1500 | Checkpoint license not stated in zoo | Do not add yet |
| FCENet | CTW1500/ICDAR2015/TotalText, ResNet-50/DCNv2/oCLIP | Checkpoint license not stated in zoo | Candidate for curved text only after audit |
| PANet | CTW1500/ICDAR2015, ResNet-18 | Checkpoint license not stated in zoo | Low priority; audit before use |
| PSENet | CTW1500/ICDAR2015, ResNet-50/oCLIP | Checkpoint license not stated in zoo | Low priority; audit before use |
| TextSnake | CTW1500, ResNet-50/oCLIP | Checkpoint license not stated in zoo | Candidate for arbitrary shapes only after audit |
| ABINet | Vision-only and iterative; ST/MJ | Checkpoint license not stated in zoo; language-model/data provenance especially important | Do not bundle pending full audit |
| ASTER | ResNet-45, ST/MJ | Checkpoint license not stated in zoo | Rectification idea is reusable; checkpoint excluded pending audit |
| CRNN | Mini-VGG, MJ | Checkpoint license not stated in zoo; dataset/training terms unresolved | Do not add; Tesseract remains the tiny permissive lane |
| MASTER | ResNet-31, ST/MJ/SA | Checkpoint license not stated in zoo | Do not add pending audit |
| NRTR | Modality-transform and ResNet-31 variants, ST/MJ | Checkpoint license not stated in zoo | Do not add pending audit |
| RobustScanner | ResNet-31, ST-sub/MJ-sub/SA-real | Checkpoint license not stated in zoo | Do not add pending audit |
| SAR | ResNet-31 parallel/sequential, ST-sub/MJ-sub/SA-real | Checkpoint license not stated in zoo | Do not add pending audit |
| SATRN | Shallow and small, ST/MJ | Checkpoint license not stated in zoo | Do not add pending audit |
| SVTR | Small/base, ST/MJ | Checkpoint license not stated in zoo; promising scene-text quality/size tradeoff | First new MMOCR recognizer to investigate, but no auto-download until weights are cleared |
| SDMGR | Visual/novisual/open-set, WildReceipt | Checkpoint, dataset, and KIE-label provenance unresolved | Do not add; existing KIE pipeline is preferred |

The official zoo reports the strongest listed detection result for DBNet++
ResNet-50-oCLIP at 0.8882 ICDAR2015 hmean-IoU, and lists SVTR-small/base as
scene-text recognizers; these are quality signals only, not license grants.
The architecture/code may be studied or clean-room reimplemented, but the
specific weights remain blocked until their provenance is documented.

#### Explicitly excluded or non-default candidates

- `Texo-Distill` is AGPL-3.0 and remains outside the permissive core model
  registry; use `PP-FormulaNet-L`, `PP-FormulaNet-S`, or another audited
  Apache/MIT/BSD formula model instead.
- Any `CC-BY-NC`, `CC-BY-NC-SA`, research-only, or unclear checkpoint is not a
  default alternative even when its architecture is attractive. It may be
  supported in a user-supplied/private model path if the caller accepts the
  license, but it must not be bundled or auto-downloaded by the default
  registry.
- Generic GAN/diffusion SR models must not be called automatically on OCR
  inputs. They can create plausible but incorrect glyph detail; acceptance is
  downstream OCR CER/confidence, not visual sharpness.

#### Quality/SOTA policy

“SOTA” is task-specific and must not be treated as a blanket OCR claim. The
selection policy is:

1. `PP-LCNet_x1_0_doc_ori` and `PP-LCNet_x1_0_textline_ori` for cheap learned
   orientation;
2. `PP-OCRv6` det/rec for the primary practical OCR baseline;
3. `Restormer` or `NAFNet` for restoration when live CER proves it helps;
4. `HAT`/`SwinIR` for quality SR and `SAFMN`/`PAN` for low-resource SR;
5. `Real-ESRGAN` only for photo inputs and only behind a no-harm gate;
6. classical cleanup and no preprocessing remain valid winners when the raw
   or VLM path scores better.

Every named model must receive a matrix row with: exact source URL, revision,
license, weight license, parameter count, GGUF quantization, live latency,
CER delta on problematic fixtures, and a human-reviewed accept/reject result.

#### O10.10 — Existing-model reconciliation

The following are not new port candidates: the repository already contains
runtime implementations, converters, tests, and local GGUF artifacts for them:

| Already available | Runtime / artifact status |
|---|---|
| `NAFNet` | `src/nafnet_denoise.cpp`; `models/convert-nafnet-to-gguf.py`; local `nafnet-sidd-w32-q8_0.gguf` |
| `SCUNet` | `src/scunet_denoise.cpp`; `models/convert-scunet-to-gguf.py`; local `scunet-color-f32.gguf` |
| `Restormer` | `src/restormer.cpp`; `models/convert-restormer-to-gguf.py`; local `restormer-denoise-f16.gguf` |
| `HAT` | `src/hat_sr.cpp`; `models/convert-hat-to-gguf.py`; local `hat-sr-x4-f16.gguf` |
| `SwinIR` | `src/swinir_sr.cpp`; `models/convert-swinir-to-gguf.py`; local `swinir-light-x4-f16.gguf` |
| `Real-ESRGAN` | `src/esrgan_sr.cpp`; `models/convert-esrgan-to-gguf.py`; local `esrgan-x4-f32.gguf` |
| `DAT` | `src/dat_sr.cpp`; `models/convert-dat-to-gguf.py`; local `dat-light-x2-f16.gguf` |
| `PAN` | `src/pan_sr.cpp`; `models/convert-pan-to-gguf.py`; local `pan-x4-f16.gguf` |
| `SAFMN` | `src/safmn_sr.cpp`; `models/convert-safmn-to-gguf.py`; local `safmn-x4-f32.gguf` |
| `TBSRN` | `src/tbsrn_sr.cpp`; `models/convert-tbsrn-to-gguf.py`; local `tbsrn-telescope-f16.gguf` |

PP-OCRv6 detector/recognizer GGUF files also exist in the shared model volume
(`PP-OCRv6_{tiny,small,medium}_{det,rec}-*.gguf`), including policy Q4_K and
Q8_0 variants. They are **not yet integrated into the current main-branch
runtime**; the PP-OCRv6 task remains an integration/port task, not a model
acquisition task. The port must include detector postprocessing, recognizer
dictionary handling, line orientation, and live German/Arabic/receipt tests.

The genuinely new preprocessing ports are therefore:

- `PP-LCNet_x1_0_doc_ori` — four-way page orientation;
- `PP-LCNet_x1_0_textline_ori` — learned 0°/180° line orientation;
- `UVDoc` or an equivalently permissive document-unwarping model — only if
  classical/TPS dewarp fails the curved-page fixtures;
- native PDFium rendering/autorotation integration, which is pipeline plumbing
  rather than a new image-restoration model.

Before adding another restoration model, O10 must benchmark the existing ten
models on the same degraded fixtures and promote only models that improve
downstream OCR CER/confidence. The current default candidates are therefore
`SAFMN`/`PAN` for cheap SR, `NAFNet`/`Restormer` for denoise, and `SwinIR` or
`HAT` for quality SR, subject to the license and no-harm gates above.

### Validation follow-up — external document parser [COMPLETED]

- Unit gates passed: region router, pipeline pool, orchestrator (62/62), and
  render tests.
- Live M1 Metal gate passed: DBNet detected 10/10 fox fixture regions and
  TrOCR recognized 10/10; measured warm total was 5.0–5.3 s/image, with 8/10
  exact words and 6.1% CER.
- The comparison implementation's live execution is environment-blocked, not silently skipped: the
  CPU configure probe lacks OpenCV development files, while the production
  path requires CUDA/TensorRT and this host has no NVIDIA device/usable Docker
  daemon. The documented NVIDIA numbers are recorded in
  `PERFORMANCE.md` as reference claims only.
- Next actionable benchmark item: run both engines on a shared corpus on an
  NVIDIA host, then add detector/recognizer quality and throughput thresholds
  to `tests/ocr_benchmark.py`.
- Quantization A/B resolved the current fox errors: TrOCR-small-printed Q4_K
  produced 8/10 exact words, while the same ggml pipeline with the recommended
  Q8_0 model produced 10/10. Keep Q8_0 as the default quality model; do not
  treat Q4_K as a quality-preserving OCR quantization.
- Q8 is now the benchmark/WASM/example default. The pipeline rejects filenames
  identifying TrOCR Q4_K unless `CRISPEMBED_DEBUG_ALLOW_OCR_Q4=1` is set.
  Text crops also receive a classical 0°/180° orientation check, and results
  now expose TrOCR mean/per-character confidence values.
- Added parity-facing structured output: deterministic reading-order indices
  and lightweight Markdown export are available from the orchestrator result
  and C API after each page run.
- Added modular server/API discovery: `/capabilities`, `/health/live`, and
  `/health/ready`; structured pipeline responses now include reading order and
  Markdown. Pipeline params and native server flags can independently enable
  layout, Tesseract-backed table cells, and PP-FormulaNet formulas.
- Added a `unified` pipeline stage backed by `crispembed_ocr_model_*`: any
  metadata-dispatched GGUF engine can now be selected as an escalation or
  specialist stage without adding another orchestrator-specific enum. This
  preserves the existing modular engine matrix, including Tesseract-LSTM,
  PARSeq, VLMs, math, and music engines where full-page/crop routing makes
  sense.

### Sequencing and boundaries

Land O1 first, then O2/O3 as the structured result and router foundation. O4 is
the first performance refactor; O5/O6 apply mainly to server builds. O7 starts
with CPU fixtures and expands to Metal/CUDA where hardware is available. Do not
replace ggml with TensorRT or make the core runtime NVIDIA-only.

## Goal

Replace ONNX-runtime-based embedding pipelines (fastembed, sentence-transformers)
with a single `crispembed` binary + C library that:

1. Loads any supported model from a GGUF file (auto-detect architecture)
2. Tokenizes input text (WordPiece / SentencePiece / BPE from GGUF metadata)
3. Runs the transformer encoder or decoder via ggml graph
4. Pools + normalizes → output embedding vector
5. Supports Q4_K / Q5_K / Q6_K / Q8_0 / F16 / F32 quantisation
6. Exposes a C API, CLI, HTTP server, Python, Rust, and Dart wrappers

## Architecture (v0.11)

```
Input text / image / audio
    │
    ├─► Text ──► Tokenizer (WordPiece / SentencePiece / BPE)
    │              │
    │              ├─► Encoder path (BERT, XLM-R, MPNet, NomicBERT,
    │              │     ModernBERT, GTE v1.5, DeBERTa-v2, SPLADE)
    │              │     Token + Pos [+ Type] embeddings
    │              │     N × Transformer layer (LN → MHA → FFN → residual)
    │              │     Pooling (mean / CLS) + optional heads
    │              │     → dense / sparse / ColBERT / reranker output
    │              │
    │              ├─► Decoder path (Qwen3, Gemma3, BidirLM-Omni text)
    │              │     Token embeddings + RoPE
    │              │     N × (RMSNorm → GQA → SwiGLU/GeGLU → residual)
    │              │     Last-token / mean pooling + L2 normalize
    │              │
    │              └─► LFM2 path (LFM2.5, lfm2_embed.cpp)
    │                    RMSNorm + GQA, 350M, BOS-only tokenization
    │                    → dense / ColBERT multi-vector output
    │
    ├─► Image ──► ViT path (SigLIP/CLIP: vit_embed.cpp)
    │               Conv2D patch embed → transformer → mean pool → L2
    │
    ├─► Image ──► BidirLM-Omni vision (bidirlm_vision.cpp)
    │               Qwen2VL ViT + patch merger + DeepStack
    │               → image_embeds spliced into decoder
    │
    ├─► Image ──► CNN path (cnn_embed.cpp)
    │               SCRFD/YuNet face detection (FPN + anchor decode + NMS)
    │               ArcFace/SFace/AuraFace face recognition
    │
    ├─► Audio ──► BidirLM-Omni audio (bidirlm_audio.cpp)
    │               crisp_audio Whisper-shape encoder → mean pool → 2048-d
    │
    ├─► Math  ──► DeiT encoder + TrOCR decoder (math_ocr.cpp)
    │               Printed math → LaTeX via ggml graph compute
    │
    ├─► Math  ──► HMER: DenseNet-121 + GRU attention (hmer_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME 2016)
    │
    ├─► Math  ──► BTTR: DenseNet + Transformer decoder (bttr_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME 2014, 53% exact match)
    │
    ├─► Math  ──► PosFormer: BTTR + ARM coverage (posformer_ocr.cpp)
    │               Handwritten math → LaTeX (CROHME, improved over BTTR)
    │
    ├─► Math  ──► MixTex: Swin-Tiny + RoBERTa (mixtex_ocr.cpp)
    │               Chinese+English LaTeX OCR (25681 BPE vocab)
    │
    ├─► Math  ──► PP-FormulaNet-S: HGNetv2 + MBart (ppformulanet_ocr.cpp)
    │               57M params, 384×384 input
    │
    ├─► Math  ──► PP-FormulaNet-L: SAM-ViT + MBart (ppformulanet_l_ocr.cpp)
    │               181M params, 768×768 input
    │
    ├─► OCR   ──► DBNet + TrOCR pipeline (ocr_pipeline.cpp)
    │               Text detection → recognition → reading-order sort
    │
    ├─► OCR   ──► Surya-OCR-2 detector (surya_det.cpp)
    │               EfficientViT + SegFormer, 38M, 91 languages
    │
    ├─► OCR   ──► Qwen2.5-VL / Qwen2-VL (qwen2vl_ocr.cpp)
    │               VLM doc OCR; german-ocr-3 (3B), FireRed-OCR, Qari-OCR, Nanonets
    │
    ├─► Layout ─► RT-DETRv2 docling-heron (layout_detect.cpp)
    │               ResNet-50 + deformable xattn, 17 document classes
    │
    ├─► OCR   ──► PARSeq scene text recognition (parseq_ocr.cpp)
    │               ViT + Transformer, 24M, 94-char ASCII, Apache-2.0
    │
    ├─► OCR   ──► InternVL2 (internvl2_ocr.cpp)
    │               InternViT + InternLM2.5 VLM, 1B/2B, MIT (+ H2OVL)
    │
    ├─► OCR   ──► GLM-OCR (glm_ocr.cpp)
    │               CogVLM2 + GLM-4, 0.9B, 8 languages, MIT
    │
    ├─► OCR   ──► GOT-OCR2 (got_ocr.cpp)
    │               SAM ViT-B + Qwen2-0.5B, document+math+table, Apache-2.0
    │
    ├─► OCR   ──► LightOnOCR-2-1B (lightonocr.cpp)
    │               Pixtral ViT + Qwen3, 1B, OCR Arena #2, Apache-2.0
    │
    ├─► OCR   ──► DeepSeek-OCR-2 (deepseek_ocr2.cpp)
    │               SAM ViT + Qwen2 + MoE decoder, 3.4B, multilingual
    │
    ├─► OCR   ──► Granite Vision 3.3-2B (granite_vision_ocr.cpp)
    │               SigLIP2 + Granite-3.1-2B, OCRBench 852, Apache-2.0
    │
    ├─► OCR   ──► Tesseract LSTM (tesseract_lstm.cpp)
    │               DBNet detection + per-line LSTM, 126 languages
    │
    ├─► NER   ──► BERT/XLM-R token classification (bert_ner.cpp)
    │               Fixed-label NER: PER/LOC/ORG/MISC, auto-detected
    │
    ├─► NER   ──► GLiNER zero-shot (gliner_ner.cpp)
    │               LFM2.5/DeBERTa-v3 + BiLSTM + span matching
    │
    ├─► KIE   ──► OCR + NER pipeline (kie_pipeline.cpp)
    │               Phase 1: OCR→NER. Phase 2: LiLT layout-aware
    │
    ├─► KIE   ──► LiLT layout transformer (lilt_kie.cpp)
    │               Dual-stream RoBERTa + BiACM, 130M, FUNSD, MIT
    │
    ├─► LID   ──► Text language identification (crisp_lid)
    │               CLD3 / GlotLID, Tesseract auto-select
    │
    ├─► Table ──► Rule-based table structure (table_parse.cpp)
    │               Line detection + grid + cell OCR → HTML
    │
    ├─► OCR   ──► PaddleOCR-VL (qwen2vl_ocr.cpp) — DONE
    │               NaViT ViT + ERNIE-4.5-0.3B, 109 langs, Apache-2.0
    │               OmniDocBench SOTA 96.3% (1.6) / 0.9B variant
    │
    ├─► Math  ──► Uni-MuMER-Qwen3-VL-2B (via qwen2vl_ocr.cpp)
    │               Handwritten math → LaTeX, 2.1B, Apache-2.0, 82% CROHME
    │
    ├─► Math  ──► Uni-MuMER-Qwen2.5-VL-3B (via qwen2vl_ocr.cpp)
    │               Handwritten math → LaTeX, 3.4B, Apache-2.0, 82.25% CROHME
    │
    │   ── PLANNED ──
    │
    └─► OCR   ──► SmolDocling (256M, Apache-2.0) — DONE: SigLIP + SmolLM2, DocTags
                    Idefics3/SmolVLM, IBM Research, DocTags output (tiny, EN-only)
```

(Evaluated and **rejected** for licensing: dots.ocr — supplemental PRC
agreement (rednote/Xiaohongshu), not pure MIT; MinerU2.5-Pro — commercial
thresholds + gated HF; Hunyuan-OCR — custom Tencent license, excludes
EU/UK/South Korea. See the next-gen table below.)

## Supported architectures (v0.11)

| Architecture | Tokenizer | Key features | Example models |
|---|---|---|---|
| BERT encoder | WordPiece | Post-LN, GELU FFN | MiniLM, BGE, SPLADE |
| XLM-R encoder | SentencePiece Unigram | Post-LN, GELU, pos_offset=2 | E5, PIXIE, arctic-l-v2, granite |
| MPNet encoder | WordPiece | Post-LN, T5-style rel attn bias | all-mpnet-base-v2 |
| NomicBERT encoder | WordPiece | Post-LN, SwiGLU, RoPE | nomic-embed-text-v1.5 |
| NomicBERT MoE encoder | SentencePiece | Post-LN, MoE 8-expert top-2, GELU, RoPE | nomic-embed-text-v2-moe |
| ModernBERT encoder | BPE | Pre-LN, GeGLU, RoPE, per-layer theta | gte-modernbert-base |
| GTE v1.5 encoder | WordPiece | Post-LN, GeGLU, NTK RoPE | gte-base/large-en-v1.5 |
| DeBERTa-v2 encoder | WordPiece | Post-LN, c2p/p2c disentangled attn | mxbai-rerank-xsmall/base-v1 |
| Qwen3 decoder | GPT-2 BPE | RMSNorm, SwiGLU, RoPE, GQA | Octen, F2LLM, Jina v5, Harrier-0.6B |
| Gemma3 decoder | SentencePiece BPE | Gemma RMSNorm(1+w), GeGLU | Harrier-270M, EmbeddingGemma-300m |
| LFM2 (bidirectional) | GPT-2 BPE | Pre-norm RMSNorm, GQA, RoPE, BOS-only | LFM2.5-Embedding-350M, LFM2.5-ColBERT |
| BidirLM-Omni | GPT-2 BPE | Bidirectional Qwen3, MRoPE, DeepStack | BidirLM-Omni-2.5B |
| ViT (SigLIP/CLIP) | — | Conv2D patch embed, CLS/mean/attn pool | siglip-base, clip-vit-base |
| CLIP text | CLIP BPE | Pre-LN, causal mask, EOS pool | clip-text-base/large |
| CNN (SCRFD/YuNet) | — | FPN, anchor decode, NMS | scrfd-det-10g, yunet |
| CNN (ArcFace) | — | ResNet-100, 512-D L2 | w600k_r50, auraface-v1, sface |
| DeiT+TrOCR | — | ggml graph encoder + decoder | pix2tex-mfr |
| HMER | — | DenseNet-121 + GRU attention | hmer (handwritten math) |
| BTTR | — | DenseNet + Transformer decoder | bttr (handwritten math) |
| PosFormer | — | DenseNet + Transformer + ARM | posformer (handwritten math) |
| MixTex | BPE (25681) | Swin-Tiny + RoBERTa 4L decoder | mixtex (CN+EN LaTeX) |
| PP-FormulaNet-S | BPE (50000) | HGNetv2 CNN + MBart 2L decoder | ppformulanet (57M) |
| PP-FormulaNet-L | BPE (50000) | SAM-ViT + MBart 8L decoder | ppformulanet-l (181M) |
| DBNet | — | ResNet-18 + FPN + DB head | text detection (12M) |
| Surya-Det | — | EfficientViT + SegFormer | surya-ocr-2 detector (38M, 91 langs) |
| RT-DETRv2 | — | ResNet-50 + deformable xattn | layout-heron (17 classes) |
| Qwen2.5-VL / Qwen2-VL / Qwen3-VL | tiktoken | ViT-32L + spatial merger + Qwen LLM; runtime ne-fix for transposed-weight GGUFs | german-ocr-3 (3B), FireRed-OCR, Qari-OCR, Nanonets, PaddleOCR-VL |
| InternVL2 | tiktoken | InternViT + InternLM2.5 LLM | internvl2-1b/2b, H2OVL |
| GLM-OCR | BPE | CogVLM2 + GLM-4 decoder | glm-edge-ocr (0.9B) |
| GOT-OCR2 | BPE | SAM ViT-B + Qwen2-0.5B | got-ocr2 (0.7B) |
| LightOnOCR | tiktoken | Pixtral ViT + Qwen3 decoder | lightonocr-2-1b (1B) |
| DeepSeek-OCR-2 | tiktoken | SAM ViT + Qwen2 + MoE decoder | deepseek-ocr2 (3.4B) |
| Granite Vision | tiktoken/BPE | SigLIP2 ViT + Granite-3.1 LLM | granite-vision-3.3-2b |
| PARSeq | — | ViT + AR/NAR Transformer | parseq (24M, 94-char) |
| Tesseract LSTM | — | DBNet det + LSTM line rec | 126 languages |
| LiLT | RoBERTa BPE | RoBERTa + layout transformer + BiACM | lilt-funsd (130M) |
| BERT NER | WordPiece/SP | BERT/XLM-R + Linear classifier | bert-ner, xlmr-ner-hrl |
| Table parser | — | Rule-based morphology + grid detection | table_parse (no model) |

## Shared code with CrispASR

| Component | Source | Reuse method |
|-----------|--------|-------------|
| ggml | submodule | identical |
| GGUF loader | src/core/gguf_loader.{h,cpp} | copy |
| Attention helper | src/core/attention.h | copy (header-only) |
| FFN helper | src/core/ffn.h | copy (header-only) |
| httplib.h | examples/server/ | copy |
| crisp_audio | CrispASR build | shared library |
| crisp_punc | CrispASR/crisp_punc/ | shared library (FireRedPunc + PCS) |
| crisp_lid | CrispASR/crisp_lid/ | shared library (CLD3 + GlotLID) |
| crisp_truecase | CrispASR/crisp_truecase/ | shared library (stat + CRF + BiLSTM) |

## File layout (current)

```
CrispEmbed/
├── CMakeLists.txt
├── README.md
├── PLAN.md                     architecture + roadmap (this file)
├── HISTORY.md                  completed milestones
├── LEARNINGS.md                technical notes
├── PERFORMANCE.md              benchmarks
├── ggml/                       (submodule)
├── src/
│   ├── crispembed.{h,cpp}      C API + encoder graph + OCR-model dispatch
│   ├── decoder_embed.{h,cpp}   decoder graph (Qwen3/Gemma3/BidirLM)
│   ├── lfm2_embed.cpp          LFM2.5 dense + ColBERT multi-vector
│   ├── bidirlm_vision.cpp      BidirLM-Omni vision tower
│   ├── bidirlm_audio.cpp       BidirLM-Omni audio tower
│   ├── vit_embed.{h,cpp}       SigLIP/CLIP ViT vision encoder
│   ├── clip_text_embed.{h,cpp} CLIP/SigLIP text encoder
│   ├── cnn_embed.{h,cpp}       SCRFD/YuNet/ArcFace/SFace
│   ├── image_preprocess.{h,cpp} C++ image preprocessor
│   ├── math_ocr.{h,cpp}        DeiT+TrOCR printed math OCR
│   ├── hmer_ocr / bttr_ocr / posformer_ocr / mixtex_ocr / ppformulanet*  math OCR
│   ├── qwen2vl_ocr / internvl2_ocr / glm_ocr / got_ocr / lightonocr      VLM OCR
│   ├── deepseek_ocr2 / granite_vision_ocr / parseq_ocr / tesseract_lstm  OCR engines
│   ├── tokenizer*.{h,cpp}      WordPiece + SentencePiece + BPE
│   └── core/                   shared helpers (gguf_loader, bpe, mel, cpu_ops)
├── examples/
│   ├── cli/main.cpp            CLI binary
│   └── server/server.cpp       HTTP server (4 API dialects)
├── models/                     GGUF conversion scripts
├── python/crispembed/          ctypes wrapper
├── crispembed-sys/             Rust FFI bindings
├── crispembed/                 Rust safe wrapper
├── flutter/crispembed/         Dart/Flutter FFI plugin
├── tools/quantize.cpp          C++ quantizer
└── tests/                      parity + benchmark scripts
```

## Pending work

Only genuinely-open, in-progress, or reference material lives below. **Completed
milestones — the imatrix quant rollout (C1), batched-encoder throughput (C3),
prefix KV cache (C4), mtmd-preprocessing port (C5), flash-attn epilogue audit
(C6), mmproj interop, the June-2026 optimization-TODO sweep, per-backend perf
passes, the SR conv→ggml sweep, the regression-guardrail closure, the CUDA
device-pointer fixes, and the scan_cleanup / unpaper feature ports — have moved
to `HISTORY.md`** (deep technical notes in `LEARNINGS.md`). Before starting any
item: read LEARNINGS "measure the DOMINANT cost before fixing a flagged
micro-gap" and "the build dir was silently CPU-only"; verify
`GGML_METAL:BOOL=ON` in `build/CMakeCache.txt`; check `git worktree list` +
`git log main..<branch>` for a concurrent session's finished work; all edits in
a worktree (ggml symlink dance, see CLAUDE.md).

### Shipped ecosystem-compat work (A1–A4, modern-bert, A3 parity harness) — see HISTORY.md

The A1–A4 JSON/hparam/CI hardening, the community `modern-bert` BPE-tokenizer
fix, the encoder ground-truth parity harness, and the e5-small/granite matrix
closure all shipped 2026-07-16 (verified in code 2026-07-20). Details + preserved
specifics: HISTORY.md "July 20, 2026" + "July 16, 2026"; deep-dives in LEARNINGS.md.

### Community-matrix coverage roadmap — candidate archs to add

Matrix entries (10): bge-small + all-MiniLM (`bert`, split-QKV/abs-pos/WordPiece),
nomic-v1.5 (`nomic-bert`), nomic-v2-moe (`nomic-bert-moe`), gte-modernbert
(`modern-bert`, gpt2-BPE), granite-107m (`bert` + `t5`/SPM, CLS), gte-base-en-v1.5
(`NewModel` tanh-GeGLU), Qwen3-Embedding-0.6B (`qwen3` decoder), LFM2.5-Embedding
(`lfm2`), embeddinggemma-300m-qat (`gemma-embedding`). Each remaining family below
exercises a DISTINCT loader/graph path not yet guarded against a
third-party GGUF. Ordered by coverage value; every one is a load + shape +
garbage-guard + HF per-stage entry (the granite recipe), each MUST be gated on the
per-stage structural cosine (a garbage-guard-only pass hides an e5-style shift).
Availability probed 2026-07-16 (repos listed are candidates, not yet validated):

| Candidate | arch / path it covers | Fits dense driver? | Candidate community GGUF | Watch-out |
|---|---|---|---|---|
| **Qwen3-Embedding-0.6B** | `qwen3` DECODER embed — last-token pool, **causal**, gpt2-BPE decoder path (distinct from modern-bert's ENCODER BPE) | ✅ (last-token) | `Qwen/Qwen3-Embedding-0.6B-GGUF` (official) + many | **ADDED + validated (2026-07-16), CLEAN — no loader change.** decoder_embed.cpp already takes blk.N.* + the gpt2-BPE KV-merges path is handled. Final cosine vs HF: q8 mean 0.999727, **f16 mean 1.000000** (graph exact); garbage margin 0.58 |
| **EmbeddingGemma-300m** | `gemma-embedding` — mean pool, **Dense bottleneck + Matryoshka** projection | ✅ (mean) | `ggml-org/embeddinggemma-300m-qat-q8_0-GGUF`, `unsloth/…`, `lmstudio-community/…` | **ADDED + SHIPPED (2026-07-17, `138ee0c`).** Real bug was the tokenizer (SPM loaded as char-level BPE), not Dense/norm; arch-gated routing to `decoder_embed.cpp` + SPM-BPE bigram-merge mode + Dense baked via `models/add-st-dense-to-gguf.py`. HF-full parity **0.985**; registry `embeddinggemma-300m-qat`; matrix entry (HF gate), 10/10 PASS; HF `cstr/embeddinggemma-300m-GGUF`. Full write-up: HISTORY.md 2026-07-17; deep-dive: LEARNINGS.md "Community `gemma-embedding`". |
| **LFM2.5-Embedding-350M** | `lfm2` bidirectional hybrid — ShortConv + attention, **BOS-only wrap** | ✅ (CLS, pooling_type=2) | `LiquidAI/LFM2.5-Embedding-350M-GGUF` (official) | **FIXED + SHIPPED (2026-07-16), added to matrix.** Was a loader gap — `lfm2_embed` requires our `lfm.*` tensor names + `lfm2.<our>` hparam keys + a `lfm2.layer_types` c/a string; the official llama.cpp export uses `blk.N.*` + canonical `lfm2.*` keys + no layer-types string. Same class as modern-bert (alias gap), bigger. **Complete fix recipe (exact tensor + hparam maps, layer-type-from-tensor-presence, per-stage gate) in the "FOUND (2026-07-16): official `lfm2`…" subsection just below.** GGUFs already downloaded. Needs a quiet box for the build + `test-lfm2-diff` per-stage validation |
| **GTE-v1.5 (gte-base-en-v1.5)** | `NewModel` NTK-RoPE + GeGLU **tanh** (the path the modern-bert `geglu_erf` gate was explicitly kept OFF for) | ✅ | `cstr/gte-base-en-v1.5-GGUF` (our own; llama.cpp ❌ so third-party rare) | **ADDED + validated per-stage (2026-07-16).** q8 vs HF fp32: emb_ln_out gate 0.999927, all layers PASS (encoder_out 0.9926). Guards the tanh-GeGLU branch stays correct next to modern-bert's erf branch. Arch coverage (own GGUF), not ecosystem-compat |
| **MPNet (all-mpnet-base-v2)** | MPNet two-stream / T5-style rel-attn bias — **we are unique** | ✅ | `cstr/all-mpnet-base-v2-GGUF` (our own; no third-party — llama.cpp ❌) | **ADDED (2026-07-20)** — matrix guard `all-mpnet-base-v2`; HF final-cos 0.997 realistic / 0.987 short (f32==q8_0 → structural residual, not quant); guards the unique rel-attn-bias graph. Arch coverage, not ecosystem-compat |
| **XLM-R-large / multilingual-e5-large** | `bert`+SPM XLM-R at 1024-dim | ✅ | `soichisumi/…-Q8_0-GGUF`, `phate334/…`, `walsons/…` | **EXPECT the e5-small position-offset FAILURE** (XLM-R needs offset 2; community `bert`-arch GGUFs omit `position_offset`). Add ONLY if a community GGUF declares the offset — else it documents the same known gap |
| **SPLADE-v3 (sparse)** | MLM/sparse head — `has_sparse` path, NOT dense | sparse metric (sparse-cos), not the garbage guard | `mradermacher/Splade-V3-GGUF` — **HEADLESS, unusable (2026-07-20)** | **Driver already does SPLADE** (CLI `--sparse`, `crispembed_encode_sparse`, `splade-pp-en-v1` ships at sparse-cos 0.996, `audit_gguf_heads` guards the head through quant). The COMMUNITY GGUF can't be supported: `mradermacher/Splade-V3-GGUF` (arch `bert`, 197 tensors, inspected) has **NO `cls.predictions.*`/MLM head** — llama.cpp drops it at convert, so it loads as a plain dense encoder (same class as e5-small/EmbeddingGemma "community export drops the head"; no loader alias recovers an absent tensor). Only OUR converter (`convert-bert --crisp`, reads checkpoint files) keeps the head. **`naver/splade-v3` ADDED + SHIPPED (2026-07-20):** `convert-bert --crisp` (MLM head detected+kept) → f16/q8_0/iq4_xs+imatrix, sparse-cos vs HF **1.0000 / 1.0000 / 0.9971**; HF `cstr/splade-v3-GGUF` (CC-BY-NC-SA-4.0 card + attribution); registry `splade-v3`/`splade-v3-q8` (NC → `--accept-license`), `upload_to_hf.py` + `audit_gguf_heads` entries. |
| **DeBERTa-v2** | disentangled c2p/p2c rel-attn (`rel_embd`, `position_buckets`) — **we are unique**, highest-complexity encoder path | ✅ | **none found** (llama.cpp ❌, no community GGUF exists) | Blocked on the absence of any third-party GGUF; only our own conversion exists |

Status: **Qwen3-Embedding, LFM2.5-Embedding, granite-107m, GTE-v1.5, EmbeddingGemma,
and MPNet all ADDED**; **e5-small CLOSED** (under-specified export). **Genuinely
remaining candidates:** **XLM-R-large** expected to reproduce the e5 offset gap
(add only as a documented negative, or if a community GGUF declares
`position_offset`); **DeBERTa-v2** blocked on GGUF availability (no third-party
export exists). Do each
on a quiet box (250K-vocab SPM reads + HF forwards are slow under contention) and
gate on the per-stage structural cosine. **SPLADE-v3 is NOT a remaining driver
gap** — sparse retrieval ships (`splade-pp-en-v1`, sparse-cos 0.996); the community
`Splade-V3-GGUF` is headless (documented above), so only an optional converter-add
of our own `naver/splade-v3` GGUF remains.

### Transcoda OMR decode enhancements (deferred, 2026-07-13)

The shipped `transcoda_ocr` engine uses greedy decode (byte-identical to the HF
reference; persistent device-KV, 2.4–4×). The paper's two higher-accuracy decode
modes are **deferred** — both are large, and neither is byte-exactly validatable,
so they were intentionally NOT shipped (byte-exact-or-bust discipline). Concrete
plans for a follow-up session:

- **Beam search (width 3)** — the paper's headline (OMR-NED 18.46% vs greedy
  ~higher on Verovio-synth). HF config: `num_beams=3, length_penalty=1.0,
  repetition_penalty=1.1, early_stopping=True`.
  - *Where*: a `decode_beam(ctx, n_beams)` in `src/transcoda_ocr.cpp`, gated
    `TRANSCODA_OCR_NUM_BEAMS=N` (opt-in; greedy stays the default). Per-beam
    next-token logits via either B independent persistent KV caches (extend
    `pk_*` to a `[..., B]` beam dim) or the full-recompute `run_decoder` per beam
    (simplest, O(B·L²) — fine for opt-in).
  - *Algorithm* (mirror HF `BeamSearchScorer`): keep B live beams (init scores
    `[0,-inf,-inf]`), each step apply per-unique-token rep-penalty + `log_softmax`,
    add to beam score, take top-`2B` over the flattened `B×vocab`, route eos
    candidates to a finished pool with score `/(len**length_penalty)`, keep the
    top-B non-eos as the next beams; early-stop when B finished hypotheses exist;
    return the best finished (or best live) hypothesis.
  - *Validation*: (1) on the confident synth page `sample_page.png`, HF beam-3 ==
    greedy, so mine must be **byte-exact == greedy** there (a real regression
    gate); (2) on a real Polish scan (`btrkeks/polish-scores`, license "other" —
    LOCAL validation only, do NOT commit the image), HF beam-3 diverges from
    greedy at accent/ornament tokens (`16b#JJ`→`16bJJ`) and spine markers
    (`*^`/`*v`) — target **CER-close** to the HF beam-3 dump (byte-exact over a
    512-token uncapped scan is not realistically achievable; cascading). HF
    references already captured: `scratch-transcoda/oracle_beam3.kern.txt`,
    `polish_beam3.kern.txt`.

- **Grammar-constrained decode** — guarantees structurally-valid `**kern`
  (paper's `grammars/kern.gbnf` via xgrammar logits processors). Large: needs a
  GBNF parser + a per-step token-mask constraint engine (llama.cpp's
  `llama-grammar` is the reference, ~1k LOC). *Where*: a `kern_grammar.{h,cpp}`
  constraint module + a mask hook in the decode loop, gated
  `TRANSCODA_OCR_GRAMMAR=1`. *Validation*: structural only (every output parses as
  valid kern); no byte-exact HF target (xgrammar's tie-breaking differs). Lowest
  priority — greedy already emits valid kern on clean inputs.

### Optical Music Recognition (OMR) — models to port (2026-07-12)

OMR is "OCR for staff notation": the winning modern approach is exactly the
TexTeller shape — vision encoder + autoregressive transformer decoder emitting
a linearized notation token sequence. This reuses the existing
VisionEncoderDecoder machinery (`math_ocr.cpp` path). Output format is
irrelevant to us (bekern / **kern / MusicXML / LilyPond are all parseable
downstream), so we optimize for arch fit + license, not output dialect.

**Two distinct problems:** printed staff notation (tractable, MIT weights
exist) and handwritten (hard; the real license risk is on the *training
data*, not the code — see landmine below).

**Licensing methodology — AGPL *code* is NOT a blocker (verified 2026-07-13).**
The gate is the **weights** license (we redistribute GGUFs) and the **engine
authorship**, which are independent of an upstream repo's *code* license:
- If the **weights** are permissive (MIT / Apache / **CC BY**), the GGUF is
  redistributable regardless of the training-code license. AGPL/GPL on the
  *code* does not attach to CC-BY *weights*. (Training-data license only matters
  if we redistribute the data or retrain — not for shipping pretrained weights.)
- The **engine** is written **clean-room**: run the upstream Python as an
  *oracle* (reference-activation dumps — no derivative) and implement from a
  **facts-only spec** (architecture, tensor shapes, op order, hparams, eps/scale
  — all uncopyrightable) + the paper + configs. Never transcribe AGPL source
  line-by-line. Two-team wall: the brief-writer may read the AGPL `.py`; the
  implementer sees only the facts brief. (Permissive blueprints don't need this.)
- Hard rejects shrink to: **gated / unlicensed / non-permissive weights**, or an
  **11B+ base under a restrictive model license**, or a **non-single-model
  pipeline** (poor ggml fit).

| # | Model | Params | License (code / weights) | Architecture | Output | Handles | Effort | Status |
|---|-------|--------|--------------------------|-------------|--------|---------|--------|--------|
| 1 | **Sheet Music Transformer (SMT)** | 21.4M | **MIT / MIT** | ConvNext encoder + Transformer decoder | bekern | Printed polyphonic | Low | **DONE** — `src/smt_ocr.cpp` shipped (per-stage cos 1.0, 96.3% GrandStaff) |
| 1b | **SMT++ full-page** | ~10.9M | **MIT / MIT** — `PRAIG/smt-fp-grandstaff` (public, **not gated**, verified HF card) | full-page extension of SMT (curriculum-trained) | bekern | **Full-page pianoform** (no separate layout stage) | **Low–Med** | **DOABLE — top permissive target.** Verify arch delta vs base SMT first: deep-research *refuted* (2-1) the "same-arch, curriculum-only" claim, so confirm the graph before assuming free reuse. If same graph → near-free extension of shipped SMT |
| 2 | **Transcoda-59M-zeroshot** | 58.8M | **AGPL code / CC BY 4.0 weights** (`btrkeks/transcoda-59M-zeroshot-v1`, verified HF card) | ConvNeXt-V2-Tiny enc + 8L Transformer dec (d512/8h, **RoPE**) | **kern | **Full-page + historical scans** (zero-shot); **current OMR-NED SOTA** (Polish 63.97%, Verovio 18.46% — beats SMT++ & Legato) | **Med** | **DOABLE — accuracy leader.** Weights CC BY 4.0 → GGUF redistribution clean (attribute). Engine **clean-room** (code is AGPL). Arch fully in-tree: ConvNeXt-V2 ≈ SMT's ConvNext, RoPE decoder ≈ Qwen3; add 3000-token kern BPE tok; optional GBNF grammar-constrained decode. Training data `polish-scores` = `license: other` (irrelevant to CC-BY weight redistribution) |
| 3 | Polyphonic-TrOMR (NetEase) | ~22M | **Apache-2.0 / Apache-2.0** | ViT + multi-head Transformer decoder (rhythm/pitch/lift/note) | symbolic text | Printed polyphonic photos | Medium | **DONE** — `src/tromr_ocr.cpp` (cos 1.0 / 100% argmax / byte-exact); `cstr/tromr-GGUF` |
| 4 | **Flova/omr_transformer** | 143M | Apache-2.0 / Apache-2.0 | Donut VED (DonutSwin + 4L mBART) | LilyPond | artificial + **handwritten** + whiteboard (monophonic) | Medium | **DONE** — `src/flova_ocr.cpp` (cos 1.0 / 40-40 argmax / byte-exact); `cstr/flova-omr-GGUF` (f32 + q8_0); CLI + registry wired |
| 5 | oemer | 2× U-Net | MIT / MIT | 2 segmentation U-Nets + numpy reconstruction | MusicXML | Printed, photos, skewed | High | Reference-only — multi-model + rule-based reconstruction, poor ggml fit |
| ~~6~~ | ~~Legato~~ | ~11B | MIT (trained delta) / **Llama-3.2 license + GATED** | frozen Llama-3.2-11B-Vision + trained decoder | ABC | full-page | — | **REJECTED** — 11B base under Meta's Llama license + contact-gated weights; MIT covers only the delta. Too big + non-permissive base |
| ~~7~~ | ~~starry / FindLab~~ | — | **no code license / gated, unlicensed weights** | 7-microservice pipeline (PyTorch+TF+ONNX) | LilyPond/kern | complex polyphonic | — | **REJECTED** — not a single model (poor ggml fit) *and* weights token-gated with no stated license |
| ~~8~~ | ~~Clarity-OMR~~ | — | (unverified) | PDF→MusicXML **pipeline** | MusicXML | printed | High | Reference-only — multi-stage pipeline, not a single VED model |
| ~~9~~ | ~~homr (liebharc)~~ | — | **AGPL-3.0** (code) | pipeline + TrOMR | MusicXML | printed/camera | — | **REJECTED** — pipeline (poor ggml fit); the underlying TrOMR is already shipped separately (Apache-2.0) |

**Recommended priority (updated 2026-07-20 — SMT/SMT++/TrOMR/Flova/Transcoda ALL shipped):**

1. ~~**SMT++ full-page**~~ **SHIPPED** — `smt_ocr.cpp` (arch reuse confirmed) +
   registry `smt-fp`; HF `cstr/smt-fp-grandstaff-GGUF`. (Was the "best next step";
   done.)
2. ~~**Transcoda-59M**~~ **SHIPPED** — clean-room `src/transcoda_ocr.cpp` +
   registry `transcoda`; HF `cstr/transcoda-omr-GGUF` (CC-BY-4.0). The only genuine
   remaining Transcoda work is the *optional* beam-3 + GBNF `**kern`
   grammar-constrained decode (still greedy-only; see "Transcoda OMR decode
   enhancements" — the sole open OMR-engine lever).

3. **Handwritten *polyphonic* — the real remaining gap.** No permissive model
   fills it: Flova (shipped) is monophonic-toy; the strong performers are all
   rejected (Legato = Llama-11B/gated, starry = gated/unlicensed pipeline, homr =
   AGPL pipeline). Reach it by *fine-tuning* a shipped graph (SMT or Transcoda)
   on synthetic + license-clean handwritten-style data — same engine, new weights.

3. **Polyphonic-TrOMR — DONE (2026-07-13).** Genuinely accurate model (reads
   clefs/keys/rhythms/pitches correctly on real photos). The ggml engine
   `src/tromr_ocr.cpp` (ResNetV2 SAME-pad backbone + hybrid ViT encoder →
   x-transformers 12-sublayer decoder with SIGLU attn-on-attn + GEGLU FF → 4
   parallel heads, autoregressive over rhythm/pitch/lift streams) is written,
   wired (dispatcher + CMake + `test-tromr-diff` + CLI `--ocr` auto-detect), and
   **validated CPU-only vs the reference model**: every diff-harness stage cos
   **1.0** (backbone, ViT context, all 12 decoder blocks, all 4 logit heads),
   **100% per-position argmax agreement** teacher-forced (66/66, 85/85), greedy
   decode **byte-exact** vs the authors' `examples/{1,2,3}.txt`, Metal == CPU.
   q8_0 also decodes byte-exact. ~~**Remaining:** HF upload `cstr/tromr-GGUF`
   (f32 + q8_0) + `model_mgr.cpp` registry entry.~~ **DONE** — `tromr` registry
   entry (model_mgr.cpp) points at `cstr/tromr-GGUF`.
   Corrections vs the (now-removed) handover brief found in validation: ViT scale
   is **32^-0.5** not 64^-0.5; the converter emitted tensor names >64 chars that
   the ggml loader rejects (`GGML_MAX_NAME`) → shortened the backbone prefix to
   `enc.bb`; the quantizer must keep `enc.bb`/`enc.proj` convs unquantized
   (flatten+quantize → reshape-to-4D abort). See LEARNINGS.md.
   Weights: `tromr/workspace/checkpoints/img2score_epoch47.pth` (86.3 MB)
   committed directly into the Apache-2.0 repo (not LFS → covered by the repo
   license), with a 4-file tokenizer set (`tokenizer_{lift,pitch,rhythm,note}.json`).
   Architectural wrinkle vs SMT: TrOMR is **not** a single autoregressive stream
   — it has *parallel classification heads* (rhythm / pitch / lift / note) per
   decoder timestep, so the port needs 4 output projections + a merge step, not
   one LM head. `homr` wraps this same model but is AGPL — weights taken from
   the NetEase repo, not homr.

**Reuse map (assessed 2026-07-12, feat/smt-omr worktree):** ~70% of the SMT
port reuses existing infra —
- **Decoder + decode loop + C ABI:** `src/math_ocr.cpp` is already SMT's exact
  shape ("Hybrid CNN + ViT encoder → cross-attention Transformer decoder → token
  sequence"): KV-cached decoder, greedy + beam decode, batched encode, per-token
  confidences. SMT's "classic Transformer decoder" == TrOCR == this; port by
  config, not new graph code.
- **Converter:** `models/convert-trocr-safetensors-to-gguf.py` already handles
  the decoder + top-level `decoder_start_token_id`. New `convert-smt-to-gguf.py`
  = that file + a ConvNext encoder tensor mapping.
- **ConvNext encoder (the one new piece):** CrispASR has ConvNeXt blocks in
  `f5_tts / vibevoice / qwen3_tts / kugelaudio / outetts_wavtok` (1-D/audio, but
  identical block: dwconv → LN → pwconv → GELU → pwconv → layer-scale → residual)
  + `core/activation.h`; CrispEmbed has mature `ggml_conv_2d` engines (`swinir`,
  `nafnet`, `cnn_embed`, `adair`, `tbsrn`) for the 2-D image side. Adapt, not
  invent.
- **Shared load/preproc/vocab:** math_ocr grayscale-resize-normalize;
  `core/{gguf_loader,cpu_ops,bpe}.h`; bekern = fixed lookup vocab (simpler than
  any in-tree BPE).
- New work = 2-D ConvNext encoder + bekern vocab + encoder-side converter.

**Confirmed SMT architecture (2026-07-12, from SMT++ source + safetensors header):**
Total **21.4M params, F32, 360 tensors, 85.5 MB** `model.safetensors`. Greedy
manual decode (no HF `.generate()`), seed `<bos>=4426`, stop `<eos>=8822`,
`pad=0`, up to `maxlen=1281` steps.
- **⚠ Convert against SMT++ tensor names, NOT SMT-main.** The shipped
  grandstaff/camera-grandstaff weights only match `SMT-plusplus/smt_model/
  modeling_smt.py` (`input_attention`/`cross_attention`/`ffNet`/`out_layer`); the
  SMT-main repo has a rewritten module whose names match no checkpoint.
  `smt-string-quartets` ships **no weights** (README only).
- **Encoder** = stock HF `ConvNextModel(num_channels=1, num_stages=3,
  hidden_sizes=[64,128,256], depths=[3,3,9])`. Plain ConvNeXt, no attention. Stem
  Conv2d(1→64,k4,s4)+LN; stage-1/2 downsample Conv2d(k2,s2); **16× H/W reduction**.
  Last stage already outputs 256 = `d_model`, so **no encoder→decoder projection**.
  `encoder.layernorm` (pooler LN) is in the ckpt but **dead** on the inference path
  (`last_hidden_state` is pre-pooler). Tensors:
  `encoder.embeddings.patch_embeddings.{weight[64,1,4,4],bias}`,
  `encoder.encoder.stages.{0,1,2}.layers.{i}.{dwconv,layernorm,pwconv1,pwconv2,layer_scale_parameter}`,
  `encoder.encoder.stages.{1,2}.downsampling_layer.{0=LN,1=Conv2d}`.
- **Decoder** = 8 layers, d_model=256, **4 heads** (hd=64), **FFN dim=256 (1×, not
  4×)**, activation **ReLU** (+ `end_relu` before the head). Post-norm:
  self-attn→norm1→cross-attn→norm2→FFN→norm3. Token emb `nn.Embedding[20578,256]`;
  **embeddings NOT tied** to head. LM head = `Conv1d(256→20578,k1)` →
  `decoder.out_layer.weight[20578,256,1]` (squeeze trailing 1 → Linear) + bias.
  Tensors: `decoder.embedding.weight`, `decoder.decoder.layers.{0..7}.
  {input_attention,cross_attention}.{lq,lk,lv,out_proj}.{weight,bias}`,
  `.ffNet.{0,3}.{weight,bias}`, `.{norm1,norm2,norm3}.{weight,bias}`.
- **Positional encodings are NOT in the checkpoint — bake as constants.** (a) 1-D
  sinusoidal added to decoder token embeddings; (b) 2-D sinusoidal
  (`dim=256`, first 128ch=row H, last 128ch=col W, `div=exp(-arange(0,dim//2,2)/dim·ln1e4)`).
- **⚠ Cross-attention key≠value:** encoder output flattened over H×W;
  the 2-D PE is added to the **KEYS only**; **VALUES are the raw** flattened
  features. Query = decoder states. Cross-attn has no mask; self-attn is causal.
- **Preprocessing:** grayscale, **always color-invert** (`RandomInvert(p=1.0)` —
  mandatory, not augmentation), `ToTensor` → **[0,1], NO mean/std normalize**.
  `cv2.resize` bilinear at `reduce_ratio=0.5`, height floored/capped ~256px
  (`maxh=256`, `maxw=3056`).
- **bekern vocab** = fixed word-level lookup (NOT BPE), `out_categories=20578`,
  identical across grandstaff/camera. `w2i`/`i2w` embedded in `config.json`
  (875 kB) and as `vocab/*.npy`. Split GT on whitespace/`·` delimiter; layout
  tokens `<b>` break / `<s>` space / `<t>` tab.
- **SMT vs SMT++:** identical neural graph; SMT++ gains are training-side
  (curriculum + synthetic full pages). Full-page = same graph, bigger images +
  longer decode + layout tokens, no extra module. **Target single-system
  grandstaff first** (the only checkpoints with published weights).

**Port progress (2026-07-12, feat/smt-omr worktree):**
- ✅ `models/convert-smt-to-gguf.py` — torch-free, verbatim SMT++ names, squeezes
  `out_layer` 1×1 conv→Linear, bakes 1-D decoder PE, records `smt.scale_attention=
  False`. Verified GGUF: arch `smt_ocr`, 361 tensors, 20578-tok vocab, 83 MB.
- ✅ `tools/dump_smt_reference.py` — loads REAL SMT++ model (hooks, not a
  re-forward), dumps 18 per-stage F32 tensors → `smt_ref.gguf`. Validated on a
  real GrandStaff test image: enc 336×128→`(256,8,21)` (16× reduction, 168 mem
  tokens), decode emits correct bekern (`**ekern_1.0 <t> … *clefG2 <b> …`).
  Test assets in scratchpad: `smt-grandstaff/`, `SMT-plusplus/` clone, `gs_test0.png`
  (+ `.gt.txt`), `smt_ref.gguf`. Note: cloned `SMTConfig` needs a
  `super().__init__(**kwargs)` patch to load under transformers 4.57.
- ✅ `src/smt_ocr.{h,cpp}` ggml engine (ConvNext encoder + cross-attn decoder +
  greedy decode) + `tests/test_smt_diff.cpp` + CMake wiring. **Full per-stage
  parity vs `smt_ref.gguf` (CPU):** enc_stage0/1/2 + enc_output + mem_key
  cos_min ≥ 0.999996; dec_tok_emb + dec_layer0–7 + logits cos_min = **1.000000**.
  Native greedy decode emits correct bekern (header/clefs/meter/barlines match
  GT exactly; `*k[]` vs GT `*k[b-]` is the model's own prediction — the Python
  ref emits `*k[]` too). Bugs found & fixed during bring-up: (a) off-trunk
  `enc_stageN` snapshots weren't in the graph (`to_tokens` forks off the trunk)
  → `ggml_build_forward_expand` each; (b) `crispembed_diff.h` GGUF reader only
  decodes F32 (its I32 branch checks a stale type id 5, but this ggml tags I32
  as 26) → dumper now stores `token_ids` as F32.
- ✅ Preprocessing parity: `recognize_raw` now does cv2-bilinear resize +
  RandomInvert + BGR-as-RGB grayscale → native decode is **token-identical to
  HF** on real GrandStaff scores (100% on 3/4; 4th matched to the ref cap), CPU
  and Metal.
- ✅ Wiring: `src/crispembed.cpp` dispatcher (`arch=="smt_ocr"` → all 4 switches),
  so `crispembed -m smt.gguf --ocr score.png` works end-to-end (verified 69/69 vs
  HF); `smt_ocr_recognize_raw` added; `examples/cli/model_mgr.cpp` registry entry
  (`smt-grandstaff`). Server/bindings inherit via the generic `crispembed_ocr_model_*`.
- ✅ Quantize: `tools/quantize.cpp` keeps SMT conv kernels (`dwconv`/`downsampling`)
  and the baked PE (`positional`) F32; engine reshapes the quantizer's flattened
  2-D conv headers back to 4-D. **q8_0 (24 MB) decodes identically to HF (100%);
  q4_k (17 MB) is too lossy for the AR decode (~32%) — ship f32 + q8_0 only.**
- ✅ KV-cache: incremental decode (cross K/V precomputed once, self K/V grown per
  step via concat). Token-identical to the full-recompute path (kept behind
  `SMT_OCR_FULL_DECODE=1` for A/B) and to HF, CPU + Metal. **5.4× faster** (0.37 s
  vs 1.98 s for ~100 tokens); the gain grows with sequence length.
- ✅ GGUF upload: `cstr/smt-grandstaff-GGUF` (f32 83 MB + q8_0 24 MB + MIT model
  card; card license verified `mit`). Registry auto-download works end-to-end.
- ✅ **Preprocessing fixed → SMT WORKS at 96.3%.** The engine had been *inverting*
  the image (SMT-plusplus's `convert_img_to_tensor` has `RandomInvert(p=1.0)`), but
  `smt-grandstaff` is an **SMT-main** model whose preprocessing is `Grayscale→
  ToTensor` with **NO invert**. Inverting → ~30%; correct (non-inverted) → **96.3%**
  on the clean `antoniorv6/grandstaff` test split (per-image 91.8/96.2/96.7/99.6%).
  Full pipeline: RGB (no cv2-BGR swap), `reduce_ratio=1.0`, `width=min(w,3056)`,
  `height=max(h,256)`, grayscale, no invert. Fixed in `recognize_raw` + the dumper.
- ✅ **Fully validated:** per-stage diff cos=1.0; C++ decode == Python blueprint
  (100% token agreement, 10 fresh images); **C++ engine vs ground truth = 96.3%.**
  SMT-plusplus's unscaled forward confirmed correct (SMT-main's forward → 0% garbage
  on this checkpoint). The port was exact all along — the invert was the only bug.
  Lesson: [[validate-intermediates-and-outputs]] — a "reads-structure-not-detail"
  pattern across models was a preprocessing/input bug, not model quality; derive
  preprocessing from the model's OWN repo (SMT-main, not the SMT-plusplus fork).

**Landmines:**
- **⚠ SMT attention is UNSCALED.** `MHA.forward` computes `bmm(q,k)` then softmax
  with **no** `1/sqrt(head_dim)` — `self.scale_factor` is defined but never
  applied (verified in source, not the abstract). The C++ must NOT scale QK^T
  (converter records `smt.scale_attention=False`). Also: token embeddings are
  **not** scaled by `sqrt(d_model)` (no `scale_embedding`).
- **Cross-attn key≠value:** memory_key = flattened encoder features **+ 2-D PE**;
  memory_value = **raw** flattened features. Easy to wire both to the same tensor.
- **Encoder `last_hidden_state` is pre-pooler-LN** → `encoder.layernorm` in the
  ckpt is dead weight; don't apply it. Feature map is `(256, H/16, W/16)`.
- **Handwritten training-data license trap:** the canonical handwritten OMR
  datasets — **MUSCIMA++ / CVC-MUSCIMA — are CC BY-NC-SA (non-commercial)**.
  Training weights on them contaminates the *weights* for commercial use (same
  pattern as the old PosFormer/BTTR/HMER math models). PrIMuS / Camera-PrIMuS /
  GrandStaff are printed/synthetic and license-clean. Keep handwritten training
  data NC-free from day one if shipped weights must be commercially usable.
- **VisionEncoderDecoder `decoder_start_token_id`** comes from the *top-level*
  config, not the nested decoder config (the TexTeller start-token bug that
  poisoned position-0 KV — see the TexTeller 3.0 entry above). SMT's converter
  must resolve the start token the same way.
- Watch F16 Metal matmul overflow on large activations (see
  [[metal-mul-mm-f16-overflow]]) as with all VED ports.

**Sources:** SMT [github.com/antoniorv6/SMT](https://github.com/antoniorv6/SMT) ·
[SMT++](https://github.com/antoniorv6/SMT-plusplus) ·
[HF smt-grandstaff (MIT)](https://huggingface.co/antoniorv6/smt-grandstaff) ·
[PRAIG collection](https://huggingface.co/collections/PRAIG/sheet-music-transformer-6853c4ca1bd7980a91677dfd).
oemer [github.com/BreezeWhite/oemer (MIT)](https://github.com/BreezeWhite/oemer).
TrOMR [github.com/NetEase/Polyphonic-TrOMR (Apache-2.0, weights `img2score_epoch47.pth` 86 MB in-repo)](https://github.com/NetEase/Polyphonic-TrOMR).
[Flova/omr_transformer (Apache-2.0)](https://huggingface.co/Flova/omr_transformer).
homr [github.com/liebharc/homr (AGPL-3.0)](https://github.com/liebharc/homr).

### OCR — next-gen models to port

| # | Model | Params | OmniDocBench | License | Architecture | Status |
|---|-------|--------|-------------|---------|-------------|--------|
| ~~1~~ | ~~dots.ocr~~ | ~~3B~~ | ~~88.4%~~ | ~~NOT pure MIT~~ | — | REJECTED: supplemental PRC license (rednote/Xiaohongshu) |
| 2 | **PaddleOCR-VL-0.9B** | 0.9B | — | Apache-2.0 | NaViT + ERNIE-4.5-0.3B | **DONE + verified E2E** (2026-07-02): reuses qwen2vl_ocr engine; fox.png → "The quick brown fox…" on CPU+Metal. Was SIGSEGV-ing (ERNIE head_dim=128≠D/heads) + empty output (SPM vocab loaded as GPT-2 BPE); both fixed. Q8_0/Q4_K on HF |
| 3 | **PaddleOCR-VL-1.6** | 0.9B | 96.3% SOTA | Apache-2.0 | NaViT + ERNIE-4.5-0.3B (same arch, improved training) | **DONE**: same engine/fixes as 0.9B; Q8_0/Q4_K on HF |
| ~~4~~ | ~~MinerU2.5-Pro~~ | ~~1.2B~~ | ~~90.7%~~ | ~~NOT pure Apache~~ | — | REJECTED: commercial thresholds, mandatory attribution, gated HF |
| 5 | **SmolDocling** | 256M | — | Apache-2.0 | Idefics3/SmolVLM, IBM Research | DONE: engine + parity cos=0.9999, HF `cstr/smoldocling-GGUF` |
| ~~6~~ | ~~Hunyuan-OCR~~ | ~~1B~~ | — | ~~Custom Tencent~~ | — | REJECTED: excludes EU/UK/South Korea |
| 7 | **Qari-OCR** | 4B | Apache-2.0 | Qwen2-VL fine-tune (Arabic only) | **DONE (shipped)** — registry `qari-ocr` → `qari-ocr-2b-q4_k.gguf`. Vision parity fixed; direct "output only text" prompt; filename-independent `general.name` detection. |

~~**Remaining**~~ **DONE (both shipped)**: FireRed-OCR (registry `firered-ocr` / `firered-ocr-q4k`, Qwen3-VL 2B) and german-ocr-3 (registry `german-ocr-3.1`, Qwen2.5-VL) both reuse the qwen2vl_ocr engine; runtime ne-fix handles GGUF converters that store weights in PyTorch (out, in) order. (NB: `src/fireredpunc.cpp` is a different model — BERT punctuation, not FireRed-OCR.)

#### OCRBench leaderboard reference (small VLMs, ≤3B)

| Rank | Model | LLM | Params | OCRBench | License | Status |
|------|-------|-----|--------|----------|---------|--------|
| 1 | Granite Vision 3.3-2B | Granite-3.1-2B | 3B | 852 | Apache-2.0 | **Ported** |
| 2 | InternVL2.5-2B* | InternLM2.5-1.8B | 2.1B | ~830 | MIT | **Ported** |
| 3 | MiniMonkey | InternLM2-1.8B | ~2B | 806 | — | Low priority |
| 4 | H2OVL-Mississippi-2B | H2O-Danube-1.8B | 2.1B | 782 | Apache-2.0 | **Ported** |
| 5 | InternVL2-1B | Qwen2-0.5B | 0.9B | 779 | MIT | **Ported** (edge) |
| 6 | InternVL2-4B | Phi-3-mini | ~4B | 776 | MIT | Low (too big) |
| 7 | H2OVL-Mississippi-0.8B | H2O-Danube3-0.5B | 0.8B | 751 | Apache-2.0 | Low (tiny) |

*InternVL2.5-2B not on the original leaderboard slice but scores higher than
InternVL2-2B (768).

### llama.cpp parity — support matrix (reference)

A living audit of which CrispEmbed architectures llama.cpp supports (upstream
`ggml-org/llama.cpp` @ ~`4fc4ec5`, July 2026), how it implements them, and where
we remain unique. Deep technical notes live in `LEARNINGS.md → "llama.cpp
implementation reference"`. **The convergence backlog derived from this audit
(C1 imatrix quant, C3 batched throughput, C4 prefix KV, C5 mtmd preprocessing,
C6 flash-attn epilogue, mmproj interop) all shipped — see HISTORY.md.** This
section is kept only as the capability reference. Any future borrow must still
land behind an A/B on BOTH speed and quality, on CPU and Metal.

#### Support matrix (CrispEmbed arch → llama.cpp)

Text-embedding encoders:

| CrispEmbed | in llama.cpp | llama.cpp arch id | note |
|---|---|---|---|
| BERT | ✅ | `bert` | one shared `bert.cpp` graph, config from GGUF |
| XLM-RoBERTa | ✅ | `bert` | RoBERTa/XLM-R fold into `bert`; pos-offset + SPM handled |
| NomicBERT | ✅ | `nomic-bert` | SwiGLU + RoPE |
| NomicBERT-MoE | ✅ | `nomic-bert-moe` | PR #12466; 8-expert top-2 |
| ModernBERT | ✅ | `modern-bert` | SWA global/local + per-layer RoPE θ |
| MPNet | ❌ | — | T5-style rel-attn bias unimplemented — **we are unique** |
| GTE-v1.5 (`NewModel`) | ❌ | — | NTK-RoPE `NewModel` unsupported (#6821) — **we are unique** |
| DeBERTa-v2 | ❌ | — | disentangled c2p/p2c has no ggml graph — **we are unique** |
| SPLADE (sparse) | ❌ | — | MLM head dropped at convert — **we are unique** |
| bge-m3 sparse+ColBERT | ❌ (dense only) | — | tri-head only in fork `iz0eyj/llama.cpp-mv` — **we are unique** |

Decoder / hybrid embedders:

| CrispEmbed | in llama.cpp | arch id | note |
|---|---|---|---|
| Qwen3-Embedding | ✅ | `qwen3` (embed mode) | last-token, **causal** (Qwen3-Emb is trained causal — correct); Instruct/Query prefix is caller-side |
| EmbeddingGemma | ✅ | `gemma-embedding` | Dense/Matryoshka projection supported via `--sentence-transformers-dense-modules`; mean, non-causal |
| LFM2 / LFM2.5 | ✅ | `lfm2` (+`lfm2moe`) | PR #14620; ShortConv via `ggml_ssm_conv`, conv tensors F32 |
| LFM2.5-Embedding | ✅ | `lfm2` embed | official LiquidAI GGUFs, bidirectional |
| LFM2.5-ColBERT | ⚠️ partial | `lfm2` + `--pooling none` | per-token out; MaxSim client-side |
| BidirLM-Omni | ❌ | — | not present — **we are unique** |

Reranking: `--pooling rank` (RANK=4), `/v1/rerank` (PR #9510). bge-reranker-v2-m3
/ base, jina-v2, ms-marco-MiniLM ✅. Qwen3-Reranker ✅ (needs `cls.output.weight`
+ template). mxbai-rerank (DeBERTa-v2) ❌.

Vision / VLM-OCR (via `libmtmd`, projector-id keyed):

| CrispEmbed | in llama.cpp | projector id | note |
|---|---|---|---|
| Qwen2/2.5-VL | ✅ | `qwen2vl_merger` / `qwen2.5vl_merger` | 2D RoPE `build_rope_2d()`, window-attn |
| Qwen3-VL (+MoE) | ✅ | `qwen3vl_merger` | **DeepStack + IMROPE** — same family as our BidirLM-Omni |
| InternVL2/2.5/3 | ✅* | `internvl` | OpenGVLab (non-HF) checkpoints only |
| GLM-4V / GLM-OCR | ✅ | `glm4v` | AIMv2 tower, **dynamic** resize — ours matches now (Glm46VImageProcessor Qwen2VL smart-resize, shipped `dfd5653`; verified OCR 2026-07-13) |
| Granite Vision 3.x | ✅ | `mlp` (LLaVA-Next) | multi-level feature concat + anyres |
| SmolVLM/SmolDocling/Idefics3 | ✅ | `idefics3` | SigLIP + pixel-shuffle |
| Pixtral / LightOnOCR-1B | ✅ | `pixtral` / `lightonocr` | LightOnOCR-2 declined (#18943) |
| DeepSeek-OCR / Unlimited-OCR | ✅ | `deepseekocr` / `deepseekocr2` | hybrid SAM+CLIP DeepEncoder |
| PaddleOCR-VL | ✅ | `paddleocr` | NaViT + M-RoPE (`ggml_rope_multi`) |
| GOT-OCR2 | ❌ | — | SAM path exists only inside DeepSeek-OCR — **we are unique** |
| CLIP/SigLIP standalone image **or** text embed | ❌ | — | mtmd is tower-only (per-patch, LLM-sized); no text tower — **we are unique** |
| Math OCR (pix2tex/TrOCR/HMER/BTTR/PosFormer/MixTex/PP-FormulaNet/PARSeq/Tesseract/Pix2Struct) | ❌ | — | enc-dec/CTC out of llama.cpp's class — **we are unique** |

**Reverse interop (import a stock llama.cpp mmproj INTO CrispEmbed):** shipped +
validated for the three rows where both a working CrispEmbed loader and a
downloadable mmproj exist — `qwen2vl_merger`, `idefics3`, `internvl` — via the
auto-detecting `models/merge-llamacpp-gguf.py` (see the status block below and
README "Importing a stock llama.cpp VL model"). Qwen2-VL is bidirectional
(export too). The rest need either a non-crashing dynamic-preproc loader
(`glm4v`) or an mmproj llama.cpp doesn't ship (`GOT-OCR2`).

Entirely outside the ggml ecosystem (CrispEmbed-only): **face** (YuNet/SCRFD/
AuraFace/SFace), **detection/layout** (DBNet/RT-DETRv2/Surya-Det), **NER/KIE**
(GLiNER/LiLT; BERT-NER only an *unmerged* PR #19725), **LID** (CLD3/GlotLID),
**punctuation** (FireRedPunc/Fullstop/PCS), and **image restoration/SR** (NAFNet/
SwinIR/HAT/Restormer/SCUNet/SAFMN/DAT/InstructIR/AdaIR — only ESRGAN/RRDBNet
exists, and in `stable-diffusion.cpp`, not llama.cpp).

### Feature gaps vs fastembed-rs

| Gap | Impact | Effort | Notes |
|---|---|---|---|
| ~~Qwen3-VL multimodal~~ **DONE** | — | — | Qwen3-VL OCR/VLM shipped: engine (`qwen2vl_ocr.cpp` DeepStack + interleaved-mRoPE + qk-norm) + registry `qwen3vl-2b`. (Only a Qwen3-VL *embedding* model — not the OCR path — would still be open, if ever wanted.) |

### DeepSeek-OCR-2 performance (remaining levers)

The pipeline is now mostly on Metal (encoder, MoE decode, SAM convs + patch
embed, LM head) — full OCR ~9 min (never completed) → ~12 s warm. Profiled
warm breakdown: load ~9 s cold / 0.8 s warm · SAM ~4.7 s · decode ~3.8 s ·
enc+proj ~1.1 s. Remaining levers, ranked by leverage:

- [x] **#1 Load-path prefetch — DONE, but not the bottleneck.** Added
  `madvise(MADV_SEQUENTIAL/WILLNEED)` to `core_gguf::load_weights` (correct
  practice, helps genuinely disk-bound cold loads on other systems). On *this*
  machine it didn't move the needle, and the diagnostic explains why: the disk
  reads 2.1 GB in **1.17 s** and a warm load is **0.8 s** — so the ~9–18 s cold
  loads are **memory-pressure / swap**, not readahead. During a run the process
  holds ~5 GB (2.1 model + 1.3 stacked experts + 0.65 embed-f32 + Metal) on a
  16 GB box, so file pages and new allocations contend and swap. → the real load
  lever is **reducing the footprint** (#3, #4), not prefetch.
- [~] **#2 Decode graph reuse — PARTIAL (KV persistent; graph NOT).** Corrected
  2026-07-20 (code-verified): the **KV cache is persistent** device tensors
  (`alloc_ds_kv_cache`, written in-graph via `ggml_cpy`/views) — but the decode
  **graph is still rebuilt + freed every layer, every token** (`build_llm_layer_attn`
  → `ggml_free(lag.gctx)` in the per-layer loop). No persistent/single multi-layer
  graph yet; the `g_ds_build_us`/`g_ds_compute_us` profiler exists precisely to
  decide if the persistent-single-graph port is worth it. So **the persistent
  single-step decode graph + F16 KV (cache is F32 today) are genuinely OPEN for
  deepseek specifically** — the one engine the GPU-decode "done" note above does
  NOT fully cover.
- [x] **#3 Per-row embedding dequant — DONE (core win).** The decode hot path
  `get_embedding` lambda is per-row (`ggml_backend_tensor_get` + `to_float`),
  replacing the 655 MB full-table copy held across decode. (Sub-detail corrected:
  the prompt-assembly `put_tok` still full-table-dequants once via `to_f32`, freed
  after — not per-row; line refs drifted to ~2424/~1897.)
- [x] **#4 Converter-emitted stacked experts (memory) — DONE
  (`feat/ds-ocr2-stacked-experts`).** Converter emits `ffn_{gate,up,down}_exps
  [in,out,n_exp]` (byte-identical to `stack_moe_experts`); loader loads them
  directly + per-expert views for the `DS_MOE_CPU` fallback + backward-compat.
  Kaggle-reconverted + byte-validated vs source; f16/q4_k on HF as `-stacked`
  (non-clobber). **M1 Metal q4_k A/B: peak footprint 5.27→3.97 GB (−1.30 GB),
  decoded output identical on all 3 loader paths.** Registry auto-download default
  **promoted to `deepseek-ocr2-q4_k-stacked.gguf`** (loader backward-compatible).
  Deep-dive in LEARNINGS.
- [ ] **#5 SAM flash-attention (marginal, skip unless needed).** The SAM
  attention uses a decomposed rel-pos bias (rel_h/rel_w added to scores), which
  blocks `ggml_flash_attn_ext` unless the bias is materialized as a [T,T] mask —
  fiddly, and the win is small (~3–4 s SAM is mostly the genuine 4096-token
  global attention compute).

All deepseek perf paths are env-gated with validated CPU fallbacks
(`DS_QWEN2_SCALAR`, `DS_MOE_CPU`, `DS_SAM_CONV_CPU`, `DS_LMHEAD_CPU`, `DS_MMAP`,
`DS_REF` parity harness, `DS_DBG` timers).

### Open performance levers

Each needs a target GGUF (q8_0 preferred, to isolate from q4_k noise) and a
before/after parity + latency measurement — never land a "perf" change on a
compile-only check. A/B every change against ground truth and gate behind an env
var (see `../crispasr-crispembed-dev.md` "A/B every perf optimization").

- **ENCODER (embedding) path — the domain the 2026-07-16 community-GGUF work
  landed in, and NOT otherwise in this backlog (encoders are fast: 6–22 layers,
  batched).** One concrete micro-lever spotted:
  - **MoE FFN redundant `ggml_repeat` (nomic-bert-moe / nomic-embed-text-v2-moe).**
    The MoE FFN in `src/crispembed.cpp` explicitly expands the input
    `cur [H,TB] → [H,K,TB]` with `ggml_repeat` before `ggml_mul_mat_id`. llama.cpp's
    canonical MoE reshapes to `[H,1,TB]` and lets `mul_mat_id` BROADCAST the
    singleton expert-slot dim, so the repeat materializes K copies of the
    activations per MoE layer for nothing (6 MoE layers × K=2 on nomic-v2-moe).
    Gate landed on `main` (`5abc4de`), broadcast path behind
    `CRISPEMBED_MOE_NO_REPEAT=1` (default keeps the repeat).
    **Correctness VALIDATED (2026-07-16):** default vs `CRISPEMBED_MOE_NO_REPEAT=1`
    on `nomic-embed-text-v2-moe` is **BYTE-IDENTICAL (max_abs_diff=0.0, cos=1.0)**
    at BOTH f16 and q4_k (50-token input) — the broadcast is exactly the repeat, so
    HF cosine is unchanged by construction. **Latency INCONCLUSIVE / neutral:** a
    7-run bench A/B (graph-compute, T=50, Metal) gave repeat median ~188 ms vs
    norepeat ~195 ms but with ±100 ms run-to-run swings at load ~9 — the
    distributions fully overlap, so no reliable delta (matches the "may be
    perf-neutral" expectation; the repeat materializes only ~1.8 MB total). **Flip
    decision deferred:** per the dev-guide rule (flip only when it wins on speed AND
    quality), a clean flip needs a genuine quiet box (load <3) for a back-to-back
    median; until then keep opt-in — correctness is no longer the blocker, only a
    trustworthy latency number is.
#### HEADLINE remaining lever — GPU recognizer AR decode (scoped 2026-07-20)

PERFORMANCE.md calls the per-region CPU-bound token loop "the real speed path".
**This is NOT greenfield:** `internvl2_ocr` (maturity rank 1) already ships the
target pattern — `ggml_flash_attn_ext` LLM decode + **F16 KV in ggml tensors
(zero-copy view + `ggml_cpy` writes)** + prefill/decode separation + sched GPU
dispatch; `glm_ocr` (rank 2) and `got_ocr` (rank 3) confirm it. The project =
**propagate that proven pattern to the laggard engines**, ranked by leverage
(PERFORMANCE.md "Optimization maturity ranking" + "Opportunities"). Beyond the
KV swap, the top layer is a **persistent single-step decode graph** (build once,
`gallocr` once at max KV, dispatch sched-free per step, **re-set ALL inputs each
compute** — the moonshine/OMR pattern; already proved here: smt-fp 18×, transcoda
2.4–4×, byte-identical).

**✅ CODE-VERIFIED 2026-07-20 — ALREADY DONE; PERFORMANCE.md's maturity table is
STALE.** Auditing the actual LLM-decode path of every engine (not the table): they
**all default to a ggml F16-KV GPU decode**, with the `core_vlm` CPU-scalar path
kept only as a gated fallback. So this "headline project" is closed. Evidence:
- **`qwen2vl_ocr`** — F16 GPU KV (`GGML_TYPE_F16` on `ctx.backend`) + `ggml_flash_attn_ext`
  + `build_decode_step_graph` (0 `core_vlm`).
- **`smoldocling_ocr`** — `sd_run_llm_body` ggml graph handles decode T=1 with F16
  backend KV; `use_ggml = (llm_sched && sd_alloc_kv_cache())` is the DEFAULT,
  `sd_llm_decode_step` (`core_vlm`) is the fallback.
- **`granite_vision_ocr`** — `gv_run_llm_body` ggml + F16 backend KV is DEFAULT
  (`if (!getenv("CRISPEMBED_GRANITE_LLM_SCALAR")) use_graph = gv_alloc_kv_cache()`),
  diff-validated cos 0.9999 vs granite-llm-ref; `core_vlm` is the opt-out. The old
  "10–50× / entire LLM CPU-scalar" is stale.
- **`pix2struct`** — KV cache + DequantCache (Phase 2/3), not "no KV, O(T²)".
- **`deepseek_ocr2`** — ggml per-layer graphs + flash + `alloc_kv` (not `core_vlm`).
- **`internvl2`/`glm`/`got`/`lightonocr`** — the reference implementations.

**Only genuine sliver left (micro, not the headline):** `deepseek_ocr2` builds
per-layer graphs (≈12 builds/token) rather than one multi-layer graph — a graph-shape
tidy, F16 KV already present. And the *persistent single-step graph* (build once,
reuse) is only in qwen2vl/lightonocr/deepseek; the others rebuild the step graph each
token but already on-GPU. Both are marginal vs the closed headline. **PERFORMANCE.md's
"Optimization maturity ranking" + "Opportunities" tables need a refresh to match.**

**Tier 2 — polish:** `lightonocr` GPU dispatch (has persistent F16 KV, GPU=No);
`internvl2` native GQA in flash (skip `ggml_repeat`).

**Landmines (non-negotiable):**
- **CUDA contiguity (LEARNING 35):** `ggml_get_rows` needs a contiguous index
  (`ggml_cont` before it). "Correct on CPU AND Metal" is NOT sufficient — CUDA has
  stricter per-op asserts; the decoded-roundtrip MUST run on a real CUDA box
  (Kaggle P100) before flipping any GPU default. [[flashattn-ext-already-permutes]]
- **Metal `set_output` snapshots LIE** on the sched — bisect on the genuine
  truncated output (`..._MAX_LAYERS=N`), not per-intermediate snapshots.
  [[set-output-on-view-stale]]
- **Metal `mul_mm` F16 overflow** (large ×N activations) → scale 1/256 pre-matmul,
  ×256 post. [[metal-mul-mm-f16-overflow]]
- **CPU-pinned decode re-copies GPU weights every token** — `load_weights_split`
  (encoder→GPU, decoder→CPU) to kill cross-backend traffic; **per-step GPU dispatch
  is launch-bound for tiny models** — the persistent graph is the win, not
  sched-free per-step. Measure the CPU baseline on the right BLAS first (parakeet
  lesson: a "GPU idle" gap was half a CPU-BLAS artifact).
- ggml scheduler: run side graphs before alloc; never reset between alloc and
  compute on the same graph.

**Validation gates (per change; env-gate every path, NEVER delete the scalar one):**
1. Per-stage `crispembed_diff` structural parity (cos ≥ 0.999).
2. **Decoded-output roundtrip is the ONLY acceptance test** — OCR a real doc, read
   the text. Test BOTH f16 AND q4_k.
3. A/B back-to-back under IDENTICAL load on a quiet box (loaded timing lies ±20%);
   final GPU-default flip gated on a Kaggle CUDA decoded-roundtrip.
4. Add a regression entry with `expected_text`; keep `<ENGINE>_CPU_DECODE=1` fallback.

**Sequencing:** ~3–5 focused sessions, each needs a quiet box + one Kaggle run.
S1 `smoldocling_ocr` (core_vlm→ggml LLM decode) → S2 `granite_vision_ocr` (10–50×, instrument-first)
→ S3 `deepseek_ocr2` (single graph + F16 KV). qwen2vl/pix2struct already done; the
the pattern first.

- **SR/restoration — fused ggml graphs: COMPLETE (2026-07-13).** Every engine
  now runs a fused ggml graph, not per-conv mini-graphs. Ported this session:
  - **SAFMN** (`8594cee`): whole forward = ONE fused graph (erf-GELU) — **2.2×
    faster AND more accurate (cos 1.000000 vs 0.994)**. Tiny/overhead-bound, so
    fusion is a big win; Metal is a net loss here (default CPU, `SAFMN_SR_METAL`).
  - **NAFNet** (`14a8393`) + **InstructIR** (`e1eb1dc`): fused per-block graph,
    cos ≥ 0.999998, output identical to legacy. NAFNet-family = **compute-bound**,
    so fusion is perf-NEUTRAL (cleaner, not faster). NAFNet defaults to Metal
    (modest ~15%; `NAFNET_CPU`); InstructIR is CPU-only (GPU conv_2d hits a Metal
    f32×f16 mul_mv pipeline issue). Gates: `NAFNET_LEGACY` / `INSTRUCTIR_LEGACY`.
  - **Restormer**: was ALREADY fused — `rst_transformer_block_ggml` (MDTA + GDFN
    in one graph) is the default; `RESTORMER_SCALAR` is the fallback (cos 0.999997
    both). Only the stale "CPU-scalar" header was corrected.
  - **scunet, swinir, tbsrn, hat, adair, dat**: already build a single graph
    (`forward_expand=1`, no per-conv helpers) — verified sensible (swinir 0.9984,
    dat 0.99999, hat 0.89 q8_0). No work needed; the "CPU-scalar" labels were loose.
  **Key finding:** the fusion win depends on overhead-bound (tiny SAFMN → 2.2×)
  vs compute-bound (NAFNet/InstructIR → perf-neutral). Metal helps only where
  per-dispatch overhead is small relative to compute. Env gates per engine.
- **SR-on-GPU — conv weight residency (research, deferred).** The entire SR
  family computes convs on a CPU-only `enc_sched` with CPU-resident F32 kernels;
  there is no GPU sibling to match. Real SR-on-GPU needs Metal `ggml_conv_2d` for
  these shapes + a GPU-resident weight/graph path the family currently avoids —
  research, not a residency toggle. Reprioritized down.
- **Decode-step graph cache — remaining decoders.** Shipped (sched-free gallocr,
  reserved once at max KV, byte-identical, per-engine env gate) for got_ocr,
  internvl2, glm_ocr, lightonocr, math_ocr. **Still open, each needs the
  single-backend decode check first:** `smoldocling` (CPU LM head outside the
  graph), `granite` (shares the vision sched), `deepseek_ocr2` (per-layer-per-step
  → needs the persistent-graph variant). Modest win (~3% light decoders, ~0% heavy;
  real value is load-insensitivity). `qwen2vl` does NOT fit (multi-backend decode).
- **ggml-metal ICB replay / op-count reduction (the real Metal decode lever).**
  Warm Metal decode is ~82% GPU-execute (per-kernel launch across ~355 sequential
  ops), so ICB (which only collapses the ~18% host-encode) caps at ~18% and is
  NOT justified for CrispEmbed's light decoders. The tractable in-tree lever is
  **fewer, bigger ops per step** — fuse per-layer norm/scale/bias chains, QKV,
  the GLU elementwise chain, prefer `ggml_soft_max_ext`. Per-decoder graph surgery
  in each `build_decoder_step_graph`; verify output cos ≈ 1.0 + node-count +
  latency per model. Re-measure heavy decoders with `CRISPASR_METAL_PROFILE=1`
  before any ICB work. **Caveat (measured 2026-07-13):** the math_ocr ~30%
  cont-removal does NOT generalize to decoder-only VLM engines — got_ocr's cached
  decode already feeds K/V as cache views, so only Q's cont was removable
  (byte-identical, but latency within noise; `5011848`, `GOT_OCR_ATTN_CONT=1`).
  **Op-fusion measured marginal too (2026-07-13):** (a) Metal already auto-fuses
  (`use fusion=true`; `kernel_norm_mul_add`, `kernel_bin_fuse` kernels handle the
  norm/scale/bias + GLU elementwise chains at dispatch), so graph-level elementwise
  fusion is redundant there; (b) attention is already flash-fused; (c) these
  decode steps are compute-bound (got_ocr ~89% GPU-execute), capping any dispatch
  reduction at the ~11% host slice; (d) the trocr decoder is already lean (319
  nodes, 55 ms/16 tok — the ViT *encoder* at 212 ms is trocr's real cost, not the
  decoder). The only non-auto-fusable win is **QKV concat-matmul** (3→1), but a
  probe (`GOT_OCR_QKV_FUSE`, 2026-07-13) confirmed it's not worth it: `ggml_concat`
  **mishandles q4_k** (garbage output) and re-concatenating per step is 3× slower,
  so a correct fusion needs manual load-time q4_k row-block byte-stacking — and on
  a memory-bound T=1 decode that only saves ~2 matmul launches/layer (~4%).
  Deferred; see HISTORY.
  (DeepSeek-OCR-2's MoE-compute lever is detailed in its own subsection above.)
- **unlimited_ocr — remaining deferred items.** `UOCR_PD=1` persistent T=1 decode
  graph (blocked on a small flash-attn padded-vs-exact-KV numerical drift that
  changes argmax by ~step 3; ~14% decode win if solved); `UOCR_OPT_GGML_WINDOW=1`
  (SAM window partition in-graph, ~2–5%, deferred); SAM flash-attn (won't — the
  decomposed RPE bias defeats the O(T) benefit).
- **text_sr — blocked on a public checkpoint** (NAFNet text-SR; registry URL
  empty, no shipped GGUF). Conv paths are guarded transitively by the `nafnet`
  entry; PixelShuffle/bicubic tail unguarded. To train one on clean (Apache/MIT)
  data see `docs/text_sr_training_data.md`.
- **esrgan tile-loop parallelism (concurrency project, deferred).** Intra-op
  threading measured SLOWER (tiled convs don't thread-scale). The real lever is
  running whole 128px tiles concurrently → needs per-thread backend+sched
  replication (the tile loop shares one `ctx->enc_sched`). Verify on a quiet box.
- **TrOCR recognizer accuracy/speed.** eos/length-penalty parity is still TODO
  (the trigram-repeat bug is fixed). The bigger levers: swap DBNet-ic15
  (scene-text) for a document-text detector on dense pages; steer document OCR to
  the doc-VLMs (PaddleOCR-VL / SmolDocling); GPU (WebGPU/Metal) recognizer decode
  is the real speed path (the per-region AR token loop is CPU-bound).

### Open correctness / infrastructure

- **CUDA regression — the 4 FAILs are RESOLVED / explained (P100-verified 2026-07-13).**
  A diagnostic kernel (`tools/kaggle/crispembed-cuda-diag`, Tesla P100 / Pascal
  sm_60) diagnosed each under its env gates, then a 2nd run verified the fix:
  - **`layout-heron` — FIXED (`49cb38a`).** The flash→manual attention fallback
    removed the `fattn.cu:602` abort; P100 CUDA now runs `test-layout-diff` to
    **8/8 stages PASS, DIFF PASSED** (dec_0_cross_out 0.977). ✅
  - **`glm-ocr` + `internvl2` — FIXED (`7998f3c`): it was a stdout banner, NOT
    vision garbage.** Both engines printed their load banner (`glm_ocr: loading…
    Vision:… LLM:… KV cache… Ready`) via `printf` → **stdout**, and `run_one`'s
    `--ocr` text-match captures stdout — so `actual` = the banner (cer 4.3/5.4,
    mis-read as "Class-B CUDA vision garbage"). The P100 diagnostic proved both
    OCR the fox **correctly** on CUDA *and* CPU; only the harness saw the banner.
    Routed all banners to stderr to match the passing engines (qwen2vl_ocr, …). ✅
  - **`granite-vision` — text OCR PASSES**; the projector diff drift is
    cross-toolchain FP strictness (identical CUDA=CPU=scalar on P100), threshold
    already 0.95. ✅
  - **Bottom line: NONE of the 4 were real CUDA vision divergences.** It was one
    genuine CUDA bug (layout flash-abort on Pascal) + a stdout-banner harness bug
    (glm/internvl2) + cross-toolchain FP threshold strictness (granite). The
    diagnostic-first approach (test on the box via env gates) was essential — a
    blind "fix the Class-B vision divergence" would have chased a non-existent bug.
  - **RESULT: portfolio 14 → 0 FAIL** across the fix waves (harness `be6ec54`;
    parser `2af57b1`; layout flash→manual `49cb38a`; banner→stderr `7998f3c`;
    parser value-dump/nameless `c26abc4`; layout perm-tolerant `debug/layout-cross`).
    glm-ocr, internvl2, granite all PASS on P100 now. **All original FAILs fixed** —
    every "Class-B" one was a harness/output bug, not CUDA vision divergence.
  - **The last FAIL (`layout-heron` `dec_0_cross_out`) — ROOT-CAUSED + FIXED
    (`debug/layout-cross`).** NOT flaky and NOT an inference bug. The apparent
    "non-determinism" (0.977 v2 vs −0.034 v14 on P100; −0.08/−0.19 on Metal
    manual/flash) is a **query-permutation comparison artifact**. The 300 decoder
    queries are chosen by `partial_sort` over ~8400 near-tie encoder proposals
    (`layout_detect.cpp:1318`); a tiny backend FP delta in enc_output (Metal/CUDA
    vs the CPU/Python reference — max_abs 0.02, cos 0.99999) reshuffles near-tie
    ranks, so "query i" in our output is a *different physical proposal* than the
    reference's "query i". Instrumented proof: the initial queries themselves show
    per-query cos mean 0.78 / 111 below 0.9 (matching cross_out's mean 0.79), the
    top-5 ranks agree, and the cross_out **values are correct** (best-cosine
    matching each ref query → cos_mean 0.999, 299/300 unique = clean bijection).
    Final boxes are unaffected (score-sort + NMS). **Fix:** `test_layout_diff.cpp`
    compares this stage permutation-tolerantly (`perm_tolerant_cos`); now PASS on
    Metal (0.947/0.999), Metal+flash (0.947), CPU (0.967/0.999). Guardrail keeps
    full power — simulated scrambles (feature-shuffle/sign-flip/roll) collapse to
    ≤0.08 vs the 0.85 gate, and s3..enc_output still guard the encoder-scramble
    class strictly at 0.99. Manifest threshold 0.97→0.85 + comment corrected (the
    old "backend-independent" note was wrong).

  Original diagnostic detail (the run that overturned 3 of the 4 assumptions):
  - **`layout-heron` — REAL CUDA bug (fixable).** `test-layout-diff` aborts:
    `ggml/src/ggml-cuda/fattn.cu:602 fatal error` in `ggml_cuda_flash_attn_ext`
    → `GGML_ABORT` because Pascal (sm_60) has **no flash-attention kernel**
    (`get_best_fattn_kernel == BEST_FATTN_KERNEL_NONE`). With
    `LAYOUT_DETECT_FORCE_CPU=1` **all 8 stages PASS (cos 1.0)** — so the graph is
    correct; the engine just runs `flash_attn_ext` on a single CUDA backend that
    bypasses the scheduler's `supports_op` CPU-fallback. **Fix:** don't use the
    CUDA flash kernel where it's unsupported — either (a) route layout attention
    through a scheduler that honours `ggml_cuda_flash_attn_ext_supported` (returns
    false on sm_60 → runs on CPU), or (b) give `layout_detect` a manual masked
    attention fallback (`mul_mat`+`soft_max_ext`+`mul_mat`, mask=nullptr = full
    attn) selected when flash is unsupported. Verify: `test-layout-diff` PASSES on
    P100. NOTE T4 (Turing sm_75) HAS flash — this only bites Pascal.
  - **`granite-vision` — NOT a CUDA bug.** The projector stages fail **identically
    on CUDA, `GRANITE_VIS_SCALAR`, AND full-CPU (`GRANITE_CPU`)** on the P100 box
    (cos 0.952 / 0.958 / 0.955 — same to 2 dp across all three), while they PASS on
    the Mac. So it is a **cross-toolchain FP-strictness gap** (Kaggle gcc vs Mac
    clang on high-magnitude projector activations, max_abs ~2.7–4.3), NOT a CUDA
    divergence, and the **OCR text passes** (cer 0.163). **Fix:** relax the
    projector-stage diff thresholds (≈0.95, they gate a real crater by going
    negative) — a parity-harness strictness fix, not a model change.
  - **`glm-ocr` — NOT a CUDA bug.** `test-glm-ocr-diff` vis_layers 14–23 fail at
    cos 0.96–0.98 **identically on CUDA and CPU** on P100 (vis_layer_23: CUDA
    0.9630 vs CPU 0.9632; max_abs up to 217) — same cross-toolchain strictness as
    granite. And on a clean generated fox image glm reads it **correctly on CUDA**
    (`"The quick brown fox jumps over the lazy dog 12345"`). So glm's vision is not
    CUDA-garbage. Its portfolio FAIL is the **text-match on the repo `fox.png`
    (800×200)** specifically — untested CPU-vs-CUDA yet (see below).
  - **`internvl2` — reads a generated fox CORRECTLY on P100 CUDA** (identical to
    CPU). No ref uploaded, so no per-stage diff. Its portfolio FAIL is likewise the
    text-match on the repo `fox.png` (800×200), not universal vision garbage.
  - **Open sub-question (glm + internvl2 portfolio garbage):** the repo `fox.png`
    is 800×200 (the diagnostic used a 640×96 render). Next diagnostic run must OCR
    the **repo** `tests/regression/images/fox.png` under default vs `*_FORCE_CPU`
    for both engines — if CPU is also garbage there, it's a Kaggle-BUILD issue
    (like granite/glm diff), not CUDA; if only CUDA is garbage, it's a genuine
    larger-image CUDA vision divergence to localize. The vis-diff being CPU=CUDA
    identical strongly suggests the former.
  - **Full data:** the diagnostic log is on Kaggle
    (`chr1s4/crispembed-cuda-diagnostic-4-remaining-fails`, transcript in
    `/kaggle/working/diag.log`); see HISTORY.md.
- **DBNet detector — mostly resolved (2026-07-13).** The CPY abort was already
  fixed (`dequant_rows_f32` via get_rows); the real cost was the CPU postprocess
  (43 s → 1.5 s, scanline box scoring `74b8ac5`, see HISTORY). Detection graph
  compute is only ~3 s on CPU and Metal `conv_transpose_2d` is still ~13× slower,
  so **CPU stays the correct default** — a faster Metal `conv_transpose_2d` (or a
  1/4-res prob-map + cheap upscale) is the only remaining, low-value, upstream
  lever for GPU-default detection.
- **bidirlm-omni GGUF re-quant follow-up.** The text-tower converter bug is fixed
  and `bidirlm-omni-2.5b-q8_0.gguf` re-uploaded (text cos 1.0 f16 / 0.9992 q8_0),
  but the repo's f16 + imatrix q4_k/q5_k/q6_k and the whole `-textonly` repo are
  still the OLD (text-broken) conversion — regenerate them from the fresh f16
  (imatrix variants via the imatrix pipeline). Kaggle-only (16 GB Mac OOMs).
- **Regression-guardrail residuals.** `bert_ner` dumper written but its ref is
  download-blocked; face *recognition* (arcface/sface) unguarded (no local rec
  GGUF; detection is guarded). All SR/restoration (11) + esrgan/safmn + lilt +
  lfm2 + the closed engines are auto-guarded in `tests/regression/manifest.json`.
- **`core/vlm_decoder.h` — deferred.** A unified scalar decode loop; only 2 scalar
  engines remain, so abstracting is premature. Revisit if a 3rd appears.

- **Tesseract seeded-artifact regeneration (2026-08-01).** The 12 installed
  canonical `.traineddata` sources were SHA-256 matched to the existing GGUFs.
  Fresh Miniconda F32/F16 conversions and metadata-repaired Q8_0/Q4_K
  companions are now in `/Volumes/backups/ai/crispembed-gguf/` as `*-seeded.gguf`.
  Forty-two companions are readable and carry nonzero `sample_iteration`; old
  files remain available for rollback. The old Fraktur `mixed-lstm0ih` candidate
  is truncated and was excluded. Per-language diff and decoded-output gates are
  still required before promotion to canonical names.

- **Tesseract int-mode kernel optimization — COMPLETED.** Cache each LSTM
  gate's int8 weights, bias quantization, and scale at model load, then
  quantize each input/hidden vector once per timestep. The prior implementation
  recomputed scales and rounded every weight inside every gate dot product.
  Benchmark and decoded-output parity must pass before promotion.

- **Tesseract int-mode kernel optimization (2026-08-02).** Packed each cached
  gate row as `[W_ih | W_hh]` and each timestep activation as `[input | hidden]`,
  reducing the hot path to one contiguous int8 accumulation per gate while
  preserving the existing quantization, LUT, and fallback semantics. On the
  seeded English Q8 model and `scan_strip.png`, three CLI runs decoded `S` in
  both modes: cached LSTM `24.2/16.3/18.9 ms`, uncached `896.3/443.5/233.7
  ms`. The output contract is unchanged; the packed cache remains the default
  and `CRISPEMBED_TESSERACT_DISABLE_INT_CACHE=1` remains the diagnostic
  fallback.

- **Tesseract LSTM scratch reuse (2026-08-02).** Added a per-context,
  environment-gated `lstm_scratch` arena for the hidden/cell/gate and int8
  activation vectors used by SummLSTM and the recurrent layers. The default
  allocation path is unchanged; `CRISPEMBED_TESSERACT_REUSE_SCRATCH=1` reuses
  buffers only within one recognizer context. On a dimension-matched seeded
  English fixture, reuse and fresh allocation both decoded `Etaansen `, so the
  output contract remains exact. The path is covered by the runtime contract
  test and remains opt-in until a repeated page benchmark demonstrates a
  material allocation reduction.

- **Tesseract composed-recoder output — IN PROGRESS.** The Chinese seeded F32
  reference passes all tensor stages but exposed native dropping of unmapped
  multi-code recoder classes (`native=''`, Python=`<141>`). The native
  diagnostic fallback now preserves `<class>`; full recode-beam composition
  and dictionary scoring remain a production-quality TODO.

- **Tesseract HF artifact publication (2026-08-01).** All 51 corrected
  canonical F32/F16/Q8_0/Q4_K files are uploaded to the intended
  `cstr/tesseract-lstm-GGUF` and `cstr/tesseract-frk-GGUF` repositories. Remote
  metadata spot-checks confirm nonzero `sample_iteration`; no `mlx-community`
  repository was used.

- **Tesseract seeded F32 sweep (2026-08-01).** Corrected references and native
  diffs now pass decoded parity for all 12 canonical languages on the
  controlled line. German's former 3/150 mismatch and Spanish's one-blank
  mismatch were stale-reference effects from NumPy float32 LUT generation;
  Korean's former 6/200 mismatch was fixed by aligning the production native
  LUT with upstream generated double-precision entries cast to `TFloat`. All
  12 runs report 9/9 stages and exit 0.

  The cache now has an explicit `CRISPEMBED_TESSERACT_DISABLE_INT_CACHE`
  fallback for controlled parity comparisons; cached mode remains the default.
  Controlled scan-strip validation passed: cached and uncached decoded text
  were both `SEEEES`; LSTM time was `35.4 ms` cached versus `1,035.6 ms`
  uncached (`29.3x` faster). Promote the cache as the default and retain the
  disable gate for future architecture-specific diagnostics.
  Full-page scan-strip validation also passed: both modes produced 12 regions,
  566 chars, CER `0.03375`, and WER `0.15044`; cached stage time was `22.11 s`
  versus `157.59 s` uncached (`7.1x` faster). Detection and crop together
  remained below `50 ms`, so recurrent recognition is the active bottleneck.

  The page comparator now preserves normalized official and native decoded text
  in its JSON output. On `scan_strip.png`, the remaining errors are concrete
  crop/decode differences (`50`→`80`, `ay`→`8ay`, `Such`/`such`, `Scheme`/
  `scheme`, and punctuation/hyphen spacing), not a region-count mismatch.
  Use these actual strings to guide crop and decoder fixes rather than treating
  CER/WER alone as a sufficient quality signal.

- **Tesseract crop-border A/B (2026-08-01).** `CRISPEMBED_TESSERACT_CROP_PAD`
  is now an opt-in gate around the Fraktur line-crop border; the production
  default remains 2 pixels. On `scan_strip.png`, 0/1/2/4 pixels produced
  12/12/12/12 regions and CER/WER `0.07460/0.30088`, `0.04796/0.20354`,
  `0.03375/0.15044`, and `0.03552/0.15044`, respectively. Keep 2 pixels as
  the default. Recognition output remains worse than official Tesseract in
  substitutions and punctuation despite matching region count; decoder,
  recoder, and line-image parity remain the active quality TODOs.

- **Tesseract page comparator hardening (2026-08-01).** The page CER/WER
  harness now supports explicit `--tessdata-dir`, clears stale inherited
  `TESSDATA_PREFIX`, and decodes subprocess diagnostics with replacement for
  invalid bytes. An explicit-tessdata scan-strip rerun remains stable at
  official 12 lines/113 words/451 chars versus native 12 regions/566 chars,
  CER `0.03375`, WER `0.15044`; no tessdata warning remains. It now also
  supports `--require-text-match`, and stores normalized official/native text
  in the comparison object so approximate CER/WER gates cannot be mistaken
  for exact page-output parity.

- **Tesseract confidence harness hardening (2026-08-01).** The line-confidence
  comparator now tolerates non-UTF-8 Tesseract diagnostics and removes stale
  inherited `TESSDATA_PREFIX` when an explicit tessdata directory is supplied.
  A valid cropped Fraktur line run produced official/native text differing only
  by spacing (`1 hey` vs `1hey`), but official mean word confidence was
  `0.7060` versus native greedy word confidence `0.9726`; beam sequence
  confidence was `0.9924` with zero fabricated character confidences. Keep the
  confidence calibration gate open until line-vs-word aggregation is compared
  against the official certainty contract.

- **Tesseract page-segmentation and beam A/B (2026-08-01).** Projection
  improved the scan-strip comparison to CER/WER `0.03197/0.12389` from the
  legacy `0.03375/0.15044`, with the same 12 regions; baseline matching was
  unchanged and slower. Keep projection behind
  `CRISPEMBED_TESSERACT_PAGESEG_PROJECTION` because the improvement is not yet
  official-output parity. Beam width 8 on projection was text-identical to
  greedy (`0.03197/0.12389`) but increased recognition from `9.66 s` to
  `29.75 s`; keep it diagnostic-only. Next TODO: line-image/crop geometry and
  Tesseract-compatible decoder/recoder semantics.

- **Tesseract page-box geometry A/B (2026-08-01).** Added gated
  `CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD` with the existing 3 px expansion as
  the default. Legacy-page tests at 1/2/3 px all emitted 12 regions, 566 chars,
  and identical CER/WER `0.03375/0.15044`; 1 px and 2 px were also slower than
  the default in the measured runs. Keep the control for alternate scan
  resolutions, but the current fixture points away from box expansion and
  toward line-image preprocessing/decoder semantics.

- **Seeded page-gate rerun correction (2026-08-01).** The earlier report of
  only 2 boxes/lines was invalid evidence: `test-ocr-orchestrator` was stale
  after the remote pageseg changes. After rebuilding the actual target, with
  proof `[76/76] Linking CXX executable test-ocr-orchestrator`, the canonical
  Q8 DBNet IC15 detector plus both corrected Fraktur seeded recognizers emitted
  the established 12 boxes/lines. The pipeline gate passed in both runs, but
  the exact-text gate remains non-green. F32 measured CER/WER
  `0.03922/0.13274`, 12,373 ms total, and confidence delta `0.01647`; Q8
  measured the same CER/WER, 14,560 ms total, and confidence delta `0.01447`.
  Native text still differs in punctuation, spacing, and several glyphs, so
  the remaining TODO is recognizer line-image/decoder parity, not detector
  geometry or a precision-only failure. The rejected stale-binary result must
  not be used to diagnose detector compatibility.

- **Tesseract line-input diagnostic (2026-08-01).** Added the opt-in
  `CRISPEMBED_TESSERACT_CROP_DUMP_DIR` hook to dump the exact grayscale crops
  passed to the native LSTM recognizer. On `scan_strip.png`, the rebuilt Q8
  run produced 12 crops (heights 22–32 px; final crop 76×25 px), matching the
  valid 12-line geometry. This rules out the stale 2-box observation and makes
  Tesseract's internal page-segmentation/line normalization the next parity
  comparison target; do not attribute the text gap to GGUF precision yet.
  The hook now also writes `crops.tsv` with source boxes and pixel ranges;
  the verified Q8 run wrote 12 crop records plus a header and exposed the
  first line as an edge-clipped box at `y=0`. Compare this manifest against
  official TSV line geometry before changing recognizer preprocessing.

- **Tesseract crop ink-trim A/B (2026-08-01).** The opt-in
  `CRISPEMBED_TESSERACT_CROP_TRIM_INK` experiment trims vertical paper around
  dark pixels while preserving native box geometry. On the canonical Q8
  scan-strip run it reduced native recognition from `11,351.6 ms` to
  `10,407.1 ms`, but worsened CER/WER from `0.03922/0.13274` to
  `0.04278/0.14159` and increased the character delta from 116 to 121.
  Reject this preprocessing for production; the remaining mismatch is not
  explained by excess vertical paper alone.

- **Tesseract page box-pad A/B (2026-08-01).** Setting the legacy component
  `CRISPEMBED_TESSERACT_PAGESEG_BOX_PAD=0` retained 12/12 regions and produced
  byte-identical native text and unchanged CER/WER `0.03922/0.13274` versus
  the default pad. Reject box expansion as the explanation for the token
  mismatch; the next target is row-boundary construction and baseline
  assignment, not another crop-padding variant.

- **Tesseract component-row A/B (2026-08-01).** The existing opt-in
  `--component`/`CRISPEMBED_TESSERACT_COMPONENT_PAGESEG` policy retained 12
  regions but degraded scan-strip quality to CER/WER `0.10873/0.20354` and
  corrupted the first line (`40eNArEBOg 10DE EES EEN`). Reject it; the legacy
  row-clustering path remains the active baseline. A separate alternate run
  with a malformed model path emitted no metrics and was discarded as invalid
  evidence.

- **Tesseract crop-geometry comparator (2026-08-01).** Added
  `tools/compare_tesseract_crop_geometry.py`, which compares `crops.tsv`
  against official Tesseract TSV level-4 rows. On the current 12-line
  scan-strip fixture, counts match, but native boxes average `dx=-2.08`,
  `dy=+1.83`, `dw=+4.33`, `dh=+1.50`; the largest deltas are row 5 width
  `+80`, row 10 vertical offset `+14`, and row 5 height `+12`. This identifies
  local row construction/splitting errors, not a global crop-padding issue.

- **Tesseract row-blob-bounds A/B (2026-08-01).** Direct debug showed row 5
  assigned blobs at `x=29..343` but the vertical ink scan widened its crop to
  `x=27..428` using neighboring-row pixels. The gated
  `CRISPEMBED_TESSERACT_PAGESEG_ROW_BLOB_BOUNDS` mode prevents that expansion.
  On scan-strip it improved CER/WER from `0.03922/0.13274` to
  `0.03209/0.11504`; geometry mean `dw` improved from `+4.33` to `+2.42`,
  worst width delta from `+80` to `+13`, while counts stayed 12/12. Keep it
  gated pending additional page fixtures; exact text parity is still open.

- **Tesseract composed-recorder (2026-08-01).** Added opt-in
  `CRISPEMBED_TESSERACT_RECODE_COMPOSE`, which segments collapsed CTC output
  classes against the serialized multi-code recoder and emits complete
  unichar tokens. The existing single-code/fallback decoder remains the
  default. Fraktur default and opt-in outputs are byte-identical on the
  controlled line; a Chinese smoke run passes both modes without crashes, but
  did not emit a multi-code class, so full composed-recoded quality parity and
  dictionary/DAWG scoring remain open.

- **Per-line page comparator correction (2026-08-01).** Official TSV words
  are now grouped by page/block/paragraph/line; the previous diagnostic
  accidentally included `word_num` and compared 113 words as 113 lines. The
  corrected row-bounds run still emits 12 native and 12 official lines, but
  only 3/12 are exact. The first divergence is line 0 (`<< 4 ...` official
  versus `“< A ...` native); lines 4, 7, and 9 match exactly. Page CER/WER
  remains `0.03209/0.11504`. The active TODO is line crop/normalization or
  decoder semantics; no recognizer math change is justified before a
  tensor-level diff of the corresponding crop.

- **Line-0 crop tensor diff (2026-08-01).** Dumped native crop 0 and created
  a fresh Python reference from `/opt/homebrew/share/tessdata/frk.traineddata`.
  `test-tesseract-lstm-diff` passed input, convolution, conv-FC, maxpool, all
  four LSTM stages, and logits; the minimum cosine was `0.997755`, mine/ref
  norms were `35.8611/35.8704` at the lowest recurrent stage, and both native
  and Python decoded `“< A hey are gomg to be encamped near Brighton ;`.
  The official Homebrew Tesseract CLI cannot reopen local image files in this
  environment (PNG, PGM, and TIFF all fail in Leptonica), but its page TSV
  line differs. This proves the line-0 quality discrepancy is in official
  page segmentation/line normalization, not the native GGUF recognizer.
  The page comparator now has `--crop-dump-dir` for fresh reproducible dumps,
  and `tools/compare_tesseract_crop_diff.py` automates regeneration of a
  per-crop Python reference plus the native tensor diff without invoking the
  broken local-image CLI path.

- **CC0 German page cross-check (2026-08-01).** The same canonical Q8 DBNet
  plus seeded German recognizer was run on
  `tests/regression/images/cc0/german_official_print.jpg`. Official Tesseract
  produced 28 lines/153 words/897 characters; native produced 23 lines/862
  characters, with CER/WER `0.32984/0.67974`. The native benchmark was
  `detect=982.4 ms`, `crop=670.0 ms`, `recognize=19594.7 ms`,
  `total=21247.2 ms`. Because the line counts differ, index-paired line CER
  is not a recognizer-quality measure here: this fixture opens a separate
  detector/line-geometry coverage gate. The comparator now marks such
  per-line alignment as invalid instead of implying a valid line-by-line
  pairing.

- **Explicit native Tesseract-like route (2026-08-01).** Added
  `--native-pageseg` to the page comparator; it sets the classical route and
  reports `detector_route=native-tesseract-pageseg`, rather than silently
  treating every run as DBNet. On `scan_strip.png`, the native route emitted
  12/12 lines with CER/WER `0.03209/0.11504`, 3/12 exact lines, and stage
  timing `detect=12.6 ms`, `crop=644.8 ms`, `recognize=11856.4 ms`,
  `total=12513.8 ms`. This matches the prior classical result and confirms
  that DBNet is not involved in this route; official-output parity remains
  open at page segmentation/line normalization and decoder semantics.

- **Native-route benchmark propagation (2026-08-01).** Extended
  `tools/benchmark_tesseract_page.py` with `--native-pageseg` and explicit
  `detector_route` output, so repeated A/B runs cannot silently fall back to
  DBNet. The wrapper and route-selection behavior are covered by the focused
  Miniconda test suite.

- **Native route on CC0 German page (2026-08-01).** The explicit
  `--native-pageseg` route on `german_official_print.jpg` also emitted 23
  lines/862 characters, versus official Tesseract's 28 lines/897 characters;
  CER/WER remained `0.32984/0.67974`. Native stage timing was
  `detect=1014.9 ms`, `crop=605.7 ms`, `recognize=14263.4 ms`,
  `total=15885.6 ms`. Thus both our DBNet and native row routes currently
  share the same five-line coverage gap on this fixture; neither result is a
  valid recognizer-quality comparison until line geometry is aligned.

- **German crop geometry guard (2026-08-01).** The native-route crop manifest
  has 23 rows while official TSV has 28. The old geometry tool's index-paired
  summary therefore produced meaningless deltas (for example mean `dy=257.7`)
  and exited nonzero only because of the count mismatch. It now reports
  `alignment_valid=false` and `paired_rows`, so those deltas cannot be treated
  as geometry measurements until a line-matching strategy handles merges and
  missing rows.

- **Merge-aware German geometry diagnostic (2026-08-01).** Added
  `--match-by-geometry` to the crop comparator. On the native German manifest
  it matched 23 native rows monotonically and identified five unmatched
  official rows (`0,2,3,4,26`) instead of treating merged/missing lines as
  same-index pairs. The tool remains strict (exit 1 while counts differ), and
  its matched deltas are diagnostic only until the row matcher accounts for
  true one-to-many merges.

- **Nested-row classification (2026-08-01).** Source-pixel review shows the
  German groups include official TSV decoration rows nested inside larger
  boxes, not necessarily missing text lines. `merged_official_groups` now
  identifies the largest primary official box and nested official indices;
  do not split a native crop merely because TSV emitted nested marks.

  On the German page, native row 0 has primary official index 1 with nested
  indices 2 and 4; native row 9 has primary index 13 with nested index 12;
  native row 22 has primary index 26 and no fully-contained nested row. The
  remaining unmatched official row 0 is not explained by a nested decoration.

- **One-to-many merge reporting (2026-08-01).** The geometry diagnostic now
  reports native rows whose vertical span covers multiple official rows as
  `merged_official_groups`, separating true row merges from simply missing
  official rows. This is diagnostic only; no production row-splitting default
  is changed until the merged groups are reviewed against the source pixels.

- **German merge candidates (2026-08-01).** On the native CC0 German
  manifest, the diagnostic identifies native row 0 covering official rows
  `1..4`, native row 9 covering `12..13`, and native row 22 covering `26..27`.
  Official row 0 remains unmatched. These are the first concrete source-pixel
  targets for a future row-splitting adapter; no production default changes
  are justified from geometry alone.
