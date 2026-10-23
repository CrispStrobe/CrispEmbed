# Executable handoff — updated 23 October 2026

Read this file, [PLAN](../PLAN.md), [the model contract](ultrabert.md),
[contributing](contributing.md) and [the shared consumer handoff](current-state-and-next-steps.md).
Completed evidence belongs in [HISTORY](../HISTORY.md); measured limits are in
[PERFORMANCE](../PERFORMANCE.md). The default-off CPU/F32 UltraBERT API is merged; general release/model quality
remains conditional. The app still consumes immutable CrispEmbed0.17.12.

## Completed and integrated

- Wheel packaging: macOS12.0 target consistency, actual installed bundled-library
  loading/C-ABI calls, and diagnostic Threads linkage. All4 platform legs,
  23 installed wheels and aggregation pass [39003273845](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39003273845)
  at `e75a32f1882a00389bac1304797f44c08ce372d4`.
  [Report](../tests/results/packaging/wheel-matrix.json). These are preview0.17.11
  wheels, not a new release/PyPI upload; producer hashes are recorded and large
  artifact bytes stay remote.
- Social preview: reviewed PNG/metadata, all native/Flutter/WASM/secret checks
  pass at `ebcadb1ffa87750972836dd858fe03e355b0fde0`; PR68 merged.
  Deployment39009694992 succeeded at f6f84aba; live HTML/PNG checked23 October: correct absolute metadata, HTTP200/image/png,1200×630 and committed PNG bytes.
- UltraBERT source/schema, both218-tensor F32/F16 conversions, original7-case/
  2,730-stage captures, test-only full-layer native graph and scoped operator
  diagnostics. All initial failures are preserved. PR74 now adds the bounded
  default-off CPU/F32 production dispatch; full-stage failures remain.
- Native BPE plus test-only UltraBERT templates match all8 original text rows and
  explicit masks; wrong terminal-SEP and ID-zero-padding alternatives each fail
  all8. Default/zero gates and4 log controls pass
  [39007901207](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39007901207)
  at `d7b84d1d4714b0d906518bbbbda21aa558d54a74`.
  [Report](../tests/results/ultrabert/tokenizer-audit.json). This does not exercise
  public model loading, bindings or task quality.
- OCR quantization panel, public evidence auditor, LFM2.5 Encoder and typed
  EmbeddingGemma milestones remain archived; their open quality/GPU/performance
  tasks and permissive OCR queue remain in PLAN. PR67 platform/Flutter/WASM/
  contract/secret checks pass at `bc1d990504a7816bfec93277ba0096fc4efa7cd4`.

## UltraBERT attribution — M1 evidence complete; F32 output qualification next

### U1 — Completed local audit; preservation and attribution next

Corrected [39009703362](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39009703362)
completed all14 native arms at `f6f84aba6ba22b27b9f15cfc08f1d182524062d2`.
[Complete report](../tests/results/ultrabert/local-operator-audit.json) preserves
all producer fields. All28 log hashes were independently checked/reparsed;
ordinary/exact/84 OPERATOR records exactly match the earlier reviewed audit.
All700 exact controls pass;183 F32/1,294 F16 ordinary failures remain.

The additional42 LINEAR records pass all21 F32 probes; F16 retains six down
projection failures against original F32 targets, max1.12353516. SiLU/product
probes pass both formats. Newly executed Python local down differs from the
frozen full-model output by up to0.0029296875; original anchors are retained.
This does not identify a broken activation or establish full-model parity.
The red job is the expected strict numerical conclusion, not structural failure.
Do not repeat it without a justified new experiment.

The bounded M1 audit atf0dbe58d is complete and independently reviewed; see the
[returned report](../tests/results/ultrabert/m1-validation.json). Do not repeat its
threads/MKLDNN arms. Next isolate original projection/reduction reproducibility using fixed input
arrays, checked loaded weights and backend/CPU/thread fingerprints. Compare
original module outputs, F64 diagnostics and frozen anchors separately; backend
alternatives must not silently become new full-model references. All strict
replay modes and original0.99999/0.002 gates remain unchanged. Use existing
`tools/` original-module capture and `crispembed-diff` stage definitions.

### U2 — Attribute remaining numerical differences

[Initial native report](../tests/results/ultrabert/initial-native-parity.json):
F32 has183 intermediate failures; F16 has1,294. F32 raw CLS cosine rounds1.0,
max error6.72340393e-5; this does not waive layer errors. Fixed gates remain
cosine0.99999 and max absolute0.002. F32 first threshold exceedance is layer14
FFN output, after smaller layer0 projection/norm differences. Compare magnitudes,
RMS, exact stage order and final raw/normalized CLS, not cosine alone.

[Original-input operator report](../tests/results/ultrabert/operator-audit.json)
passes41/42 F32 probes: Unicode whole MLP error0.0048828125 drops to0.000671386719
with frozen gate/up. F16 retains24 local failures. U1 shows all F32 local
SiLU/product/down probes pass while composed errors remain; replay
projection/reduction alternatives only where its results justify them. Preserve
working gated alternatives and rerun the unchanged complete panel for a correction.

Strict original replay is separately unresolved. Isolated replay rejects an
empty gate difference2.0265579223632812e-6; full replay rejects287 empty-case old
payloads, first layer0 Q5.960464477539062e-7 despite hook neutrality.
[Isolated failure](../tests/results/ultrabert/initial-linear-capture-failure.json),
[full failure/all differences](../tests/results/ultrabert/full-forward-capture-failure.json).
Both used the same pinned Python/Torch/Transformers versions; hardware/reduction
cause is not established. Record actual backend/CPU/library fingerprints and
check loaded tensor bits and earliest operator inputs before attributing cause.
Both strict replay modes must remain available; never silently regenerate their
anchors or call U1 local targets full-model parity.

### U3 — F32 API and Rust/Dart bindings merged; complete remaining input surfaces

[PR74](https://github.com/CrispStrobe/CrispEmbed/pull/74) merged3f3d023a after
all exact-source4be9f864 checks, including ten Build/Flutter39943915528 jobs.
[API39943940607](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39943940607)
passes all twelve pooled/seven actual text-row and C API/Python/CLI/HTTP checks.
[Public API evidence](../tests/results/ultrabert/public-f32-api.json).
The default-off CPU/F32 path covers ordinary text and sequential batches,512
attended tokens including BOS, explicit prefixes and raw/normalized CLS. Literal0
rejects. Public pairs, reserved-token inputs, F16/quantized/GPU/Matryoshka remain
excluded. Historical183/1294 intermediate failures and strict replay failures
are unchanged; F16 raw CLS0.0185 still fails all seven original cases.

Merged [PR75](https://github.com/CrispStrobe/CrispEmbed/pull/75) at2f078406
integrates actual wrapper consumers and a `bindings_only` workflow arm.
Initial39972325056 built Rust/library but stopped before inference on a
versioned-SONAME checker bug. The exact-path guard is corrected; failure is
archived in initial-bindings-loader-failure.json. Corrected39979622930 passes all42 pooled metrics/14 binding ID rows and
consumer/gate/failure contracts, independently checked with all artifact bytes.
[Evidence](../tests/results/ultrabert/public-bindings.json). Raw/dense/normalized-token
CLS maxima are6.723403930664062e-5/5.941838026046753e-7/
6.07222318649292e-7, with minimum cosine0.999999999994188 in both languages.
The initial loader failure, unchanged frozen references and all183/1294 full-stage
failures remain. Do not repeat the successful model run; preserve remote raw
vectors/logs before their21 January2027 expiry.
Pair API is a later explicit contract: BOS0+A+SEP15+B, no terminal separator,
attended BOS despite pad ID0; do not change the generic BERT template globally.
The optional [M1 native API handover](handover-m1-ultrabert-f32-api.md) supplies
Apple execution coverage without repeating the completed Python audit.

### U4 — Independent quality, quantization and publication

The [owned retrieval smoke](ultrabert-retrieval-smoke.md) is complete in
[39950831201](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39950831201)
at812e56e5/native4be9f864. All32 actual IDs/64 vector comparisons pass; original
SentenceTransformer/native Recall@1/MRR@10/NDCG@10 all1.0 and all8 top1 match.
Raw/normalized maxima7.343292236328125e-5/5.513429641723633e-7, minimum cosine
0.999999999999019. [Full scores/provenance/review](../tests/results/ultrabert/retrieval-smoke.json).
Both structural initial failures are retained. The original wrapper ran first;
raw vectors/logs remain remote in ultrabert-retrieval-39950831201 through
21 January2027. Preserve the same bytes before expiry; use verified original_run
reuse for justified native retries, never regenerate these original anchors.

Next choose a separately licensed representative retrieval dataset and freeze
revision, split, queries/corpus/gold, prompt/normalization and quality/ranking
acceptance before new candidate outputs. Execute original SentenceTransformer
first, then the gated native API on hosted resources, reporting misses, score
matrices and numerical magnitudes separately. The owned smoke is not MTEB/BEIR
reproduction. Longer contexts and GPU/latency/RSS need distinct resource/parity
plans;512 remains the public bound. Further attribution requires a distinct
hypothesis and existing original probes; do not repeat the four backend arms.

Quantization needs declared development/calibration inputs plus independent
quality evaluation; keep norms and other critical precision choices measured.
Use `crispembed-quantize` for Q4_K, retain failed recipes and their errors. Complete
Apache license/notice/base provenance, HF model card, remote byte/hash verification,
pinned aliases and empty-cache downloads only for a validated candidate. Do not
publish current experimental UltraBERT artifacts. Base150M/380M masked-LM heads
are separate later ports; do not discard the untied head or infer task-head training.

## Original arrays preserved; use the verified copy after original expiry

Original producer38997562520 at `0573f1847db68914052a98341a22fceeb1358518`
was copied without regeneration by [39913800444](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39913800444).
All seven archive hashes/2730 stage payloads and original conversion records pass.
[Preservation report](../tests/results/ultrabert/reference-preservation.json)
records original identity and inventory. Remote artifact
`ultrabert-original-38997562520-preserved-39913800444` expires
**21 January2027 06:48:52 UTC**. Original artifact expires31 October2026.
The native workflow accepts `reference_run=38997562520` plus
`preserved_run=39913800444`; original report/source/payload checks remain intact.
Before the new expiry, make another verified remote copy or durable archive,
without rerunning original inference. Do not use the preservation run as a new
reference producer or silently substitute recomputed arrays.

Bounded reproducibility [39915714794](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39915714794)
completed at8031ae1c. [Full report](../tests/results/ultrabert/reproducibility.json)
retains four arms ×seven cases ×386 captured stages and112 same-input projection
records. Every case first differs at layer0 Q;287 stages differ in every arm.
All selected loaded weight bits match the pinned checkpoint. MKLDNN on/off is
identical at fixed thread count; threads1/2 change document projections/full
outputs, but neither reproduces frozen anchors. F64 projections do not restore
all frozen values either. This narrows controls without establishing a cause.

The [M1 handover](handover-m1-validation.md) records the completed independent
CPU/BLAS comparison. Full returned diagnostics preserve earliest RoPE differences,
original projection inputs and raw/normalized CLS. Worst raw CLS6.52e-5 and
normalized6.41e-7 do not imply every hidden state passes: long/padded hidden
error0.005791664 remains. Retain original0.99999/0.002 gates and native183/1294
failures. Next use the scoped F32 output lane U3 rather than rerunning unchanged
backend controls; any further attribution experiment needs a distinct hypothesis.

## Other executable fronts and boundaries

| Lane | Next bounded action | Completion evidence / limits |
| --- | --- | --- |
| Integration CI | Completed39009694933 at f6f84aba: all10 Build jobs including Flutter; WASM/lint/contracts pass | Scheduled health/OCR regression green through22 October; rerun affected checks for new code |
| PR68 deployment | Completed39009694992; live metadata/PNG verified23 October | Deployment revision f6f84aba; no new app/runtime release |
| PosFormer PR73 | Merged75e87ea6 after source790f9a6b all checks, Build/Flutter39933251542 and dedicated39933245147 pass | Source/default lane complete; actual successor assets/consumer qualification and original-checkpoint diagnosis remain;7/50 unchanged |
| Version/release | Development main0.17.13; published/app0.17.12; actual Linux compatibility measured | PR73 restores the published trained default with legacy0 retained. Actual successor artifact/hash/consumer qualification remains required before any tag |
| Qwen OCR numerical debt | Use original-embedding/vision isolation and controlled linear-reduction probes in PLAN | Keep924 historical records/56 failures, held-out F16 six/13 failures and all decoded/quality gold/caps; no numerical waiver |
| OCR quantization/publication | Independent development/quality work after the completed nine-run Q8/mixed/Q4 panel | No tested recipe currently qualifies; Q4 receipt truncates/unclosed HTML. CER/WER, cells, completion, confidence and format matter, not cosine alone |
| Permissive OCR queue | OvisOCR2 source/license/header, original metadata39983101540 and [full F32/F16 export39992662147](../tests/results/qwen35-ocr/ovisocr2-export.json) pass; PR77 merged2f8b5354 | [PR78](https://github.com/CrispStrobe/CrispEmbed/pull/78) adds separately gated frontend atccb6059a; hosted40017014043 passes; PR78 merged918c1c8d. Inspect original/native packed-pixel, grid, ID/mask/modality/template and stop/profile/gate results before merge. No native model alias/decoded parity. Xiaomi follows; Indra public-code-only |
| Existing encoders | LFM2.5 official F16 CPU/CUDA T4 audit passes594 direct checks/both15-case API panels; [report](../tests/results/lfm2-encoder/cuda-f16.json). One-session paired T4 timing passes independent420-sample review | Other GPU/precision/MMQ, throughput, longer contexts and independent quality remain open; typed Gemma GPU/task/performance is separate |
| OCR portfolio/app | Select one bounded portfolio task from PLAN or app lane in shared handoff | Retain original fixture/metric/decoded acceptance and owner; SAM work belongs to its existing owner; no unrelated dependency upgrade |

All builds, suites, model/batch/browser work run on hosted CI or authorized remote
GPU resources. On shared infrastructure, first inspect load/RAM/swap/free space.
Use isolated worktrees, preserve unrelated files/owners, and keep private machine
setup/access notes ignored. Optional ignored environment notes may supplement
this public handoff; the task definitions above do not depend on them.

## Completed bounded paired GPU performance; follow-ups remain

`tools/kaggle/lfm2-encoder-cuda-performance/` initial recipe9fd70ce5/versionv1 hit its30-minute build timeout at97%, before
inference. Versionv2/7a17864f extends only the build limit to60 minutes, journals
timeouts and terminates its process group. Original worker bytes are unchanged. The
v2 run completed all four blocks and15 cases. Independent review matches the
pinned recipe/worker and all18 source declarations, selected build/configure/
placement logs, four raw-block SHA/content pairs and all60 rows/420 timing
samples. CPU0/CUDA180 scheduler `MUL_MAT` assignments confirm placement. In
this one T4 session all15 cases are faster on CUDA (3.48–120.11x); pooled14-
sample median9/20/649-token latencies are136.5423/4.88886 ms,
304.819/11.0264 ms and9667.336/80.4885 ms (CPU/CUDA). The
public script pins the audited runtime898af2c2, F16/modelbb26e78c and complete
original fixturee3599a94 hashes. CPU/CUDA/CUDA/CPU processes retain all per-case
seven-repeat latencies after two warmups, initialization separately, exact IDs
and original CLS gates. Scheduler probes require actual CUDA use and CPU
control isolation. Peak RSS includes initialization/quality probes; no VRAM
peak, throughput, full GPU residency or general speedup claim is justified.
Reported CPU/CUDA CLS gates pass, but raw CLS arrays were not retained in the
performance artifact, so those numerical metrics cannot be independently
recomputed there. The separate594-stage audit remains the stronger numerical
evidence. Retain raw reports and the v1 timeout; other hardware, precision,
throughput, task quality and defaults remain open.
[Full timing producer and independent review](../tests/results/lfm2-encoder/cuda-f16-performance.json).

Development desktop CPU portability merged via
[PR76](https://github.com/CrispStrobe/CrispEmbed/pull/76) at1f8ff4d5.
Actual [39979940916](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39979940916)
passes Linux/macOS/Windows desktop configure/build/smoke/install jobs; all three
pin both native-off options and pass the release CPU-baseline checker before
compilation. [Reviewed flags and log hashes](../tests/results/packaging/development-cpu-baseline.json).
Cross-host installed bundle reuse and successor release assets remain separate.

## Ovis — original GPU reference complete; native and quality execution pending

Frontend PR78 merged918c1c8d after exact pixels/IDs/grids/masks/template,
Unicode and negative controls40017014043. Controlled original GPU recipe5fe3fc90
passes actual FP16/TRITON_ATTN decoder/Torch-SDPA vision/Triton-GDN execution.
Both image references reach EOS248046; the unchanged auto-FlashInfer T4 path
fails and is retained. [Full GPU proof](../tests/results/qwen35-ocr/ovisocr2-vllm-reference.json)
and [immutable fixture preservation](../tests/results/qwen35-ocr/ovisocr2-vllm-reference-preservation.json)
keep the exact original bytes; do not regenerate them or modify the positive
GPU receipt now consumed by downstream source/hash gates.

Continue the five pending steps in [PLAN](../PLAN.md). Native model run
[40034685690](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40034685690)
is queued at468f5504; do not duplicate it. Download only
`ovisocr2-model-reports-RUN`. Keep the large model/reference and raw actual
native vector artifacts remote, retrieving specific diagnostics only when needed.
Trace the earliest differing original module with absolute errors/norms, caches,
all scores and actual decoded IDs/text/EOS; retain all failures and raw strict
flags. Existing vision/final-norm original backend controls are captured before
candidate outputs, with unchanged hidden/cache/logit thresholds.

Actual public C/Python/CLI/HTTP/Rust/Dart, independent quality, protected Q8/Q4
and development-only calibration workflows are prepared in draftPR79. They
require reviewed predecessor evidence; none has a model-backed result yet.
Publication is conditional on all planned outcomes, license-chain provenance,
qualified candidate hashes and actual immutable empty-cache HF readback.
The optional [M1 build handoff](https://github.com/CrispStrobe/CrispEmbed/blob/8b13e839/docs/handover-m1-ovisocr2.md)
is a build-only fallback; the current original backend-control protocol is Linux.
The app remains pinned0.17.12. No Ovis model weights or alias are published.
