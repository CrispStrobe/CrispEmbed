# Executable handoff — updated 26 October 2026

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

- MI-GAN's bounded fixed256 CPU/F32 prototype is integrated through
  [PR85](https://github.com/CrispStrobe/CrispEmbed/pull/85) at`1b74b12a`.
  [Corrected40344396533](../tests/results/migan/native-parity.json) at
  `2526fcac5ee76e8a476157291d8277781e6348e4` passes494 comparisons and all8
  image gates against original40330674649/`9bd33154`. The initial256 stale
  observer fields remain in [the rejected receipt](../tests/results/migan/initial-native-parity.json);
  the repair retains test-only allocation owners and preserves all8 ordinary
  final hashes. This completes the prototype, not issue71's product/API,
  representative subtitle/perceptual/video or model-publication work.
- [RoPE40336758177](../tests/results/ultrabert/rope-attribution.json) at
  `b689f871fffe33b741061269561fc2ebb276ff97` completes the bounded operator
  lane:1,344 comparisons/336 bit controls pass. No production RoPE gate follows.
- [OvisOCR3 preservation40287253703](../tests/results/qwen35-ocr/ovisocr3-vector-preservation.json)
  is integrated throughPR83: all1,662 original payloads survive verified remote
  copy/readback. Copy12635202371 expires24 January2027; source-quality and
  publication failures remain unchanged. MI-GAN's corrected artifacts and the
  RoPE raw artifact also expire24 January2027; renew exact bytes before expiry.

## UltraBERT attribution — completed operator evidence; F32 quality pending

### U1 — Completed local and RoPE audits; remaining numerical debt

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
threads/MKLDNN arms. The distinct [RoPE audit40336758177](../tests/results/ultrabert/rope-attribution.json)
also completes: all336 current original table fields match the frozen tables,
and all336 source-table-fed native Q/K rotations are bit-exact. Ordinary Q/K
error stays below0.000161648. Empty position0 rotation is exact and cannot
explain earliest empty pre-RoPE Q or strict empty-replay drift; composed
nonempty causality is unmeasured. No production RoPE gate is justified.
Further attribution needs a distinct hypothesis using preserved inputs/weights
and actual backend fingerprints. Keep all strict replay modes and original
0.99999/0.002 gates; never replace frozen full-model anchors with local targets.

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

The full canonical SciFact panel is executing as
[40375626336](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40375626336)
at`e35cd1a0ba3c2dcac1fdfbded739f5663d0b8902`, after independently reviewed
original-only [prepare40372423343](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40372423343).
[The preparation receipt](https://github.com/CrispStrobe/CrispEmbed/blob/025b3424002bad099e9827e7b5b52c3f033d461e/tests/results/ultrabert/scifact-canonical-preparation.json)
freezes all5,483 actual wrapper ID/mask rows before forward:300 queries/5,183
documents,460 documents right-truncated to512 tokens, no truncated queries or
dropped rows. Thirteen original preflight rows are hook-neutral; conservative
worst-original-shard projection is2,840.06 seconds against13,200 seconds.
The eight original shards and frozen original receipt must precede all eight
native shards. Numerical gates remain0.002/0.99999; NDCG@10/MRR@10/Recall@100
degradation must stay within0.001 and top1 agreement at least99%. Full vector,
ranking and all-miss/margin review remain pending; preflight is no quality proof.

Strict author-column v1 rejections remain preserved. Distinct canonical v2 was
chosen before model outputs after exact legacy-JSONL/Parquet equality and a
complete author-difference journal; literal stored columns/query prefix are
retained without normalization. Its data/model notices remain distinct, including
CC BY-SA4.0 benchmark terms and upstream claim/abstract attribution. Public
pairs remain excluded. The candidate's UltraBERT-specific right-truncation gate
is default-off; unset/literal0 retain overflow rejection. Longer contexts and
GPU/latency/RSS need separate plans. Do not repeat the four old backend arms.

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
| Permissive OCR queue | OvisOCR2 reviewed [native40049244654](../tests/results/qwen35-ocr/ovisocr2-model-audit.json) passes14/14 frozen arms; [public40053283383](../tests/results/qwen35-ocr/ovisocr2-public-surfaces.json) passes27 arms; [first F16 quality40056641800](../tests/results/qwen35-ocr/ovisocr2-f16-quality.json), publication40074370335 and actual published consumer40080458492 pass. Profiling, issue71 feasibility and [fourth selective quant review](../tests/results/qwen35-ocr/ovisocr2-q4-attention-f16-quality.json) are complete | All four quant attempts are rejected; published F16 remains selected. Expanded-runtime F16 replay40178330437 matches the frozen first run, but final-source14/11 are positive;2048 is a completed numerical rejection. OvisOCR3 original reference is captured; its original eleven-case plain OCR quality is negative at the fixed1024 cap. Native14/F16 fidelity are reviewed; public diagnostic is independently positive with model/publication flags false; no OvisOCR3 alias or publication. Preserve initial failures and raw strict flags |
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

## OvisOCR2 completed qualification and follow-ups

Frontend PR78 merged918c1c8d after exact pixels/IDs/grids/masks/template,
Unicode and negative controls40017014043. Controlled original GPU recipe5fe3fc90
passes actual FP16/TRITON_ATTN decoder/Torch-SDPA vision/Triton-GDN execution.
Both image references reach EOS248046; the unchanged auto-FlashInfer T4 path
fails and is retained. [Full GPU proof](../tests/results/qwen35-ocr/ovisocr2-vllm-reference.json)
and [immutable fixture preservation](../tests/results/qwen35-ocr/ovisocr2-vllm-reference-preservation.json)
keep the exact original bytes; do not regenerate them or modify the positive
GPU receipt now consumed by downstream source/hash gates.

The initial native run
[40041958521](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40041958521)
at cc038b00 retains its943 frozen fields,871 earlier byte-identical members
and12/14 result; both vision arms fail unchanged budgets in the
[initial numerical evidence](../tests/results/qwen35-ocr/initial-ovisocr2-model-vision-failure.json).
Existing precision A/B40047163838 atdf65e911 passes the `both` and
`cascade_precise_attn` presets for F32/F16. The reviewed
[full frozen replay40049244654](../tests/results/qwen35-ocr/ovisocr2-model-audit.json)
at a2a17735 uses explicit `both`, passes all14 native arms, retains all943
original fields and matches nine IDs/EOS248046/`Hello OCR\n\n12345`.
Its four raw strict0.002 vision exceedances remain recorded and pass the
unchanged backend-calibrated budgets. Do not recapture the original reference.
Initial40034685690 stopped before
forward on the tuple/list processor guard. The correction compares exact JSON
values and records every field/type; second40039159696 confirms it before
stopping on the generation-config assumption. The corrected source checks metadata
AutoConfig and actual raw-config fallback separately, with exact equality.
Third40040392649 confirms both, completes original forward, and stops on72
uncaptured dictionary-backed cache fields. The completed cc038b00 capture observes integer0
states and required all871 retained NPY members unchanged plus exactly72 new
stages before native work. Artifact12539012965 stays remote, expires21 January
2027; preserve it remotely before expiry. Raw decoded output and all source
contracts remain unchanged.
[Initial report and replay evidence](../tests/results/qwen35-ocr/initial-ovisocr2-model-processor-contract-failure.json). Download only
`ovisocr2-model-reports-RUN`. Keep the large model/reference and raw actual
native vector artifacts remote, retrieving specific diagnostics only when needed.
Trace the earliest differing original module with absolute errors/norms, caches,
all scores and actual decoded IDs/text/EOS; retain all failures and raw strict
flags. Existing vision/final-norm original backend controls are captured before
candidate outputs, with unchanged hidden/cache/logit thresholds.

The eleven-case original GPU quality reference is complete and
[preserved](../tests/results/qwen35-ocr/ovisocr2-vllm-quality-reference.json). All
cases reach EOS; all eight authored texts have WER0. Strict-v1 table border
attribute failures remain retained. Original HF and native quality are separate.

Actual [public40053283383](../tests/results/qwen35-ocr/ovisocr2-public-surfaces.json)
at b33c5bad passes27 C/Python/CLI/HTTP/Rust/Dart model-backed arms with
exact text and bounded confidence. The frozen
[original HF eleven-case quality40054043629](../tests/results/qwen35-ocr/ovisocr2-original-quality.json)
at a0784b19 has all11 EOS248046/no cap, eight synthetic WER0 cases and15/15
inventory cells under the predeclared Ovis presentation policy. Its raw
newline CER, high realistic document text errors and strict-v1 border parse
failures remain visible. The independently reviewed
[first native F16 quality40056641800](../tests/results/qwen35-ocr/ovisocr2-f16-quality.json)
at638380f6 completes all eleven cases. Its1,841 generated IDs, raw text and
processed text match frozen original HF case by case; every case stops at
EOS248046 without a cap hit. The eight authored cases have WER0, the inventory
has15/15 cells, and bounded synthetic/document paired screens pass. Its
F16 GGUF remains the qualified `d3a733a5...` checkpoint, now published
byte-identically as documented below.
The first F16 wrapper was frozen at8b421ba5 with file SHA-256
`ff68a1cd4c4cc74b5092b748a69bd61c29b80c823b1d4219985ee3595c508bd6`.
Its small report artifact is12547361841 (`ovisocr2-quality-reports-40056641800`),
ZIP SHA-256 `e99344ae5b0c54bc8d3d6db680a8431e6829c9b75e755049d20ab1632534c90f`;
the producer JSON SHA-256 is `648b2c252b08e1e56008fe81b2ce5aa9c1390eaf9cf93b7b8e5699303cf0925e`.
Reuse the reviewed wrapper and small report for quant source/score gates.
The first native run used an AMD EPYC7763 CPU with two threads; original HF ran
on a different CPU. Per-case wall times and cumulative child RSS are diagnostic;
do not claim a paired speed result or eleven-case identity across hosts.

Quant phases use the immutable first F16 eleven-case scores, original inputs
and independent review. The reusable
[quality workflow](../.github/workflows/ovisocr2-quality.yml) defaults its
quant host control to one actual `dev-hello` F16 case after exact source-tree,
ggml gitlink, runner/environment/CMake AST, package and thread checks against
the first run: ten Git tree/blob paths, six AST contracts and25 recorded
native source hashes must agree. Its actual PNG/RGB, nine IDs, text, EOS,
cap, confidence, wall time and cumulative RSS upper bound are recorded.
Direct driver calls default
to the original `full` eleven-case host control; the workflow also accepts
`full`. The short control supports only a one-case host check. The full
eleven-case F16 quality baseline remains the first reviewed run, while each
quantized candidate still runs all eleven cases, the complete original
numerical comparison and a retained before/after 473-tensor audit. Calibrated
Q4 additionally retains the small development importance files separately
from the large candidate GGUF.

Protected [Q8 40061433357](../tests/results/qwen35-ocr/ovisocr2-q8-quality.json)
is complete and independently rejected. The first captured
`prefill.decoder.layer.0` error is0.00376427 (cosine0.999920726); all eleven
cases stop at EOS without a cap, but the two-column page changes `Lorem Ipsum`
to `Lorem Ispum` and grows697→700 generated IDs with worse raw/content
CER/WER. Ten cases retain first-F16 IDs/text. The complete negative Q8
diagnostic authorizes separate Q4 experiments and blocks Q8 publication.
Actual [plain Q4](../tests/results/qwen35-ocr/ovisocr2-q4-quality.json) and
[calibrated Q4](../tests/results/qwen35-ocr/ovisocr2-q4-calibrated-quality.json)
are complete and rejected for numerical and document nonregression failures.
All11 cases in each reach EOS without caps; F16 local controls pass18/18.
The72-vector [development-only calibration proof](../tests/results/qwen35-ocr/ovisocr2-q4-calibrated-calibration.json)
is positive separately from candidate qualification. All473 tensor records and
261 raw manifest entries per Q4 arm are retained; large payloads stay remote.
[Q8 candidate-only replay](../tests/results/qwen35-ocr/ovisocr2-q8-raw-diff.json)
retains243 vectors and first QKV max0.0564889908, without replacing the rejected
Q8 quality wrapper. Earlier pre-model failures and raw strict flags stay in
HISTORY and their immutable receipts.

[Completed selection](../tests/results/qwen35-ocr/ovisocr2-quant-attempts.json)
chooses qualified first F16. [Actual publication40074370335](../tests/results/qwen35-ocr/ovisocr2-publication.json)
at121dc943 passes upload and immutable empty-cache full readback at
[Hub b7229e46](https://huggingface.co/cstr/OvisOCR2-GGUF/tree/b7229e46f31bcf342d0999966bae8a03bf02bb07).
Anonymous tree/small-file verification passes; no probes/test weights are
included. The [initial403](../tests/results/qwen35-ocr/initial-ovisocr2-publication-authorization-failure.json)
stays separate; the temporary publisher binding is removed after success.
The real `ovisocr2-f16` URL/SHA alias and both loader fixes are wired. All25
recorded native source files and qualified `src`/ggml/CMake objects remain
unchanged; ten current consumer controls pass, including native-source drift
rejection and Darwin RSS units. Actual
[consumer40080458492](../tests/results/qwen35-ocr/ovisocr2-published-consumer.json)
at5296e955 passes two fresh-cache full model hash checks, offline CLI cache
reuse and actual CLI/HTTP text, region and existing confidence budgets. Both
independent reviews verify all seven report members/six logs, seven helper and
25 native source hashes. Public APIs expose no IDs/stops; the separately frozen
native nine-ID/EOS proof remains unchanged. PR79 is merged after final-source
and merged-source checks. The original qualification loop is closed. Bounded
latency/RSS attribution is now complete in
[the profiling receipt](../tests/results/qwen35-ocr/ovisocr2-profiling.json) and
[PERFORMANCE](../PERFORMANCE.md).
Model publication does not upgrade the app or create a runtime release.
The optional [M1 consumer handoff](handover-m1-ovisocr2.md)
pins the same published model and native code with corrected Darwin RSS units.
It is an optional Apple alias/API audit; full original backend qualification remains Linux.
The app remains pinned0.17.12; published F16 and wired source alias do not upgrade it.

### Next work after the completed bounded Ovis audit loop

The bounded decoder weight-cache follow-up is complete. Actual inference
[40215901483](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40215901483)
at`e6ee5afd93b2c49d3107e0b518e0944eaa6ada80` passes all28 native programs and
the eleven-case F16 quality job, with all1,841 first-F16 outputs/confidences
exact. Its overall workflow is red because a new manifest adapter incorrectly
required a`file` field absent from decoder records. Preserve that rejected
producer and [initial failure](../tests/results/qwen35-ocr/initial-weight-cache-manifest-failure.json).
[Analysis-only40222430001](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40222430001)
at`02cdfa7bf7ceb7d0ae46abd6c0a7de59a0cbbf3b` rehashes the actual3,324 retained
vectors and qualifies all14 arms per gate with1,662 bit-identical pairs.
It performs no inference/conversion and creates no new original reference.
[Full reviewed evidence](../tests/results/qwen35-ocr/ovisocr2-weight-cache.json),
[quality](../tests/results/qwen35-ocr/ovisocr2-weight-cache-quality.json).

The cache is per context, decoder-only,256MiB default/1GiB hard retained-buffer
limit, stable first-fit/no eviction; quantized matrices and the970MiB full head
retain the original fallback. Serial/parallel cold admission, oversized fallback,
literal0, reset, independent and survivor controls pass on all three native
platforms in [40216852682](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40216852682).
The runtime/CMake/numerical checker remain identical to the actual inference
source above; subsequent changes strengthen the component fixture and fix
postprocessing. Three timing pairs give only2.734% cold/0.797% warm median
reductions, one warm regression and increased RSS; keep the default **off**.

For justified future cache audits use the`weight-cache` phase of
`.github/workflows/ovisocr2-metadata.yml`, reviewing its exact source pins first.
The`weight-cache-reanalysis` phase accepts only source run40215901483 and its
pinned preserved artifacts; do not rerun it without a new analysis concern.
Older`current-replay`/larger-input helpers intentionally reject the new cache
runtime source scope. Use their retained exact producer source/arrays for2048
attribution, or declare a separately versioned source contract; never widen a
historical allowlist silently. Raw artifact12609314203 remains remote through
23 January2027; preserve its original bytes before expiry. The first-FFN
quantized-weight attribution below is complete. The subsequent norm/position
operator work is complete; its corrected composed panel remains pending below.
Adoption needs representative paired resource evidence; app/runtime release and other model qualifications
remain separate.

The [issue 71 source feasibility audit](inpainting-feasibility.md) is complete.
MI-GAN has separately licensed MIT code and weights. Its integrated fixed256
original/native prototype now passes the bounded unchanged panel above. Public
API/registry, representative still-image/subtitle/perceptual/video quality and
publication remain pending under [issue71](https://github.com/CrispStrobe/CrispEmbed/issues/71).
LaMa still needs authoritative checkpoint terms and an FFT-capable implementation.

The fourth OvisOCR2 precision attempt,
[attention-F16/FFN-Q4 40172527229](../tests/results/qwen35-ocr/ovisocr2-q4-attention-f16-quality.json),
is complete and independently rejected. All473 tensor records satisfy the
declared72 FFN-Q4/401 byte-exact policy, including78 unchanged F16 attention
matrices. Both first-linear probes pass. The full decoder first fails at
`prefill.decoder.layer.0` (maximum absolute error0.0328894854, minimum cosine
0.997915924); historical-receipt content CER/WER and two-column raw/content
CER/WER regress despite11 EOS completions. The
[four-attempt selection](../tests/results/qwen35-ocr/ovisocr2-quant-attempts-v2.json)
keeps the published F16. Do not tune the frozen held-out cases or replace the
first F16 receipt.

The expanded-runtime OvisOCR2 F16 replay
[40178330437](../tests/results/qwen35-ocr/ovisocr2-expanded-runtime-f16-quality.json)
independently matches all1,841 frozen first-F16 IDs, raw/processed text,
stops, caps and input/confidence records on11 pages; both paired quality screens
pass. This preserves the first baseline and does not qualify subsequent core
edits. The [final-source frozen14-arm run40183491593](../tests/results/qwen35-ocr/ovisocr2-final-runtime-frozen-replay.json)
is independently positive with unchanged comparison metrics. The
[final-source F16 eleven-case run40183493328](../tests/results/qwen35-ocr/ovisocr2-final-runtime-f16-quality.json)
is also independently positive, including all1,841 IDs/text/stops/confidences
and both recomputed quality screens. The bounded larger-input run
[40177774574](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40177774574)
is complete and independently reviewed in [the combined receipt](../tests/results/qwen35-ocr/ovisocr2-larger-inputs.json).
Frozen/1024/1200 pass14/14;2048 passes8/14 with six composed prefill/full-prefix
norm failures (F32/F16 max0.003945589/0.004386902). Original prepared embeddings
pass; all nine IDs/raw text/EOS still match. No numerical waiver follows.
Default1024 remains;2048 is a four-gate unqualified experiment. Full producer JSON
is retained in Git, large raw arrays remotely. Follow the scoped2048 attribution
lane in PLAN before broader-page promotion.

OvisOCR3 [metadata](../tests/results/qwen35-ocr/ovisocr3-metadata.json),
[original943-stage reference](../tests/results/qwen35-ocr/ovisocr3-original-reference.json)
and controlled original GPU proof are reviewed. Its
[original eleven-case plain OCR capture40178328864](../tests/results/qwen35-ocr/ovisocr3-original-quality.json)
preserves all outputs and is **quality negative**: the two-column page reaches
the frozen1024-token cap without EOS (CER0.52365, WER0.83993). Keep the prompt,
gold, cap and scorer fixed. The earlier native14 run40179894600 exposed two
test-harness classification/prompt defects; the final-source native14 run
[40183489780](../tests/results/qwen35-ocr/ovisocr3-native-model.json)
is independently positive:14/14 arms,29 native source files, both473-tensor
exports and1,662 raw declarations. Four strict vision flags per precision remain
under unchanged original-backend calibration. The distinct Ovis3 context rejects
1200 patches even with old2 large-image flags enabled, then recovers.
[Paired F16 eleven-case40187405735](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40187405735)
is independently reviewed at ee719aef: all11 source cases/1,679 IDs/raw text/
stops/caps match, with identical raw scores and negative source completion.
Selected-probability max absolute error versus original F32 is0.0002462863;
confidence arrays are finite/aligned, not bit-exact. Native source and all
reviewed gates are bound before weights. [Interface-only diagnostic40196755934](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40196755934)
is independently positive on all29 recorded checks across C/Python/CLI/HTTP/
Rust/Dart, ABI/cache and14 expected negative gate/profile/prompt probes; see
[the reviewed diagnostic](../tests/results/qwen35-ocr/ovisocr3-public-diagnostic.json).
Its dispatch must explicitly name
`ovisocr3-native14-f16-40183489780`; initial40196033239 omitted that argument
and failed safe before model transfer/build, with its original report retained. Even faithful diagnostics cannot mark the model qualified or publish
an alias while the original completion screen fails. Keep OvisOCR2 and OvisOCR3
proofs, prompts and stop scopes distinct.

The active order and acceptance gates remain in [PLAN](../PLAN.md). Preserve
large model/reference/raw-vector artifacts remotely and inspect only small
reports here. Do not promote queued work, a component cosine, or a source audit
to decoded-quality or publication proof.


### Completed bounded first-FFN attribution — PR82 integrated

`audit/ovis-ffn-20261025` implements the [frozen five-arm protocol](ovisocr2-first-ffn-attribution.md).
Use metadata workflow `phase=first-ffn` on that branch. The test-only executable
includes the actual decoder source; production `src` remains byte-identical to
native tree `e5cbd629818f0424e272bc84469e9375eee804d2`, ggml `890278a8`.
It consumes rejected run40172527229's exact candidate/raw artifacts and original
reference40049244654, with service/ZIP/member hashes checked remotely. Original
SF-derived F16 matrix hashes must reproduce the frozen before-quantization
records. No original array is regenerated, and no new candidate is quantized.

[Actual40231922668](../tests/results/qwen35-ocr/ovisocr2-first-ffn-attribution.json)
passes all five arms at`a5883feb`: all11,010,048 independent dequantized values,
original reconstructed layer and retained rejected native layer match exactly.
Same-weight FFN errors stay below1.669e-6; original Python with Q4 reproduces
layer error0.03288919. No local implementation mismatch is identified. Small
ZIP/source/log review covers all20 metrics; large raw arrays remain remote and
were not separately downloaded for review. Raw artifact12613722786 expires
23 January2027. All document-quality rejections/published F16 remain unchanged.
Two superseded dispatches were cancelled while queued before model execution.
PR82 merged at`160520e8086cccb87a4dc577f684a5f5b0aa7c10` after all22 final PR
checks; all11 fresh-main workflows pass, including
[Build40234266896](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40234266896)
and [metadata controls40234266978](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40234266978).
All five actual audited file hashes and the native source tree match main.
Production source did not change, so this is no runtime release or app upgrade.
Use the separately scoped2048 attribution lane; any new precision
selection needs a fresh development/evaluation protocol rather than fitting the
observed held-out documents. Review the separately executing corrected2048
composed panel; MI-GAN product work remains distinct. Leave SAM with its owner.


### November Ovis3 vector expiry — preserved copy available

[Preservation40287253703](../tests/results/qwen35-ocr/ovisocr3-vector-preservation.json)
copies source40183489780/26dcb07f without inference. Source artifact12595929739
expires24 November2026; copy12635202371
`ovisocr3-native14-raw-40183489780-preserved-40287253703` expires24 January2027.
It contains `original-native-vectors.zip` and `member-manifest.json`. The inner
ZIP retains source SHA006d73e811f18bad66dabf9c1cf7f8476b0c845957d733e4a2fcd3a599a5cd3b;
the wrapper ZIP is SHA cb82d736244ed1af5ac912e7f32f06a85b9f0198f543f212e0be7dc9ca836695.
All1,662 manifest payloads are rehashed twice on hosted CI, including actual
remote readback; the small manifest reconstruction/review is recorded separately.
Use this as a preserved source archive, never a new reference producer. Retrieve
and inspect large archives only on hosted resources. Original numerical flags,
source-quality failure and publication-disabled status stay unchanged. Preserve
other January archives before their recorded deadlines; future renewal must
verify these original ZIP/member bytes rather than regenerate model outputs.


### Current executing loop,26 October — completion remains pending

The integrated MI-GAN prototype, RoPE audit and Ovis3 preservation above are
completed evidence. The remaining loop requires actual review of these lanes:

- Ovis corrected composed [40376882009](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40376882009)
  at`8bb9f074d1ca6fa2cb2be6a892fc2fd116202912`: both eleven-case quality jobs
  pass all1,841 IDs/text/stops against the first F16 baseline; gate1's measured
  maximum confidence difference is2.4796e-5. Full frozen/grid numerical review
  remains pending. Initial40356657327 stopped on raw/public report path
  redaction before the affected fourteen-arm audits; its
  [negative receipt](https://github.com/CrispStrobe/CrispEmbed/blob/8bb9f074d1ca6fa2cb2be6a892fc2fd116202912/tests/results/qwen35-ocr/ovisocr2-position-report-redaction-failure.json)
  remains immutable. The correction verifies raw JSON identity and permits only
  the declared comparison-copy path redaction. No budget/default change follows.
- SciFact full40375626336 at`e35cd1a0` remains pending after the reviewed
  preparation above. Preserve source/freeze/receipt identities through the
  original-first barrier, native shards and aggregate; v1 remains rejected.
- Initial packaged consumer [40354251257](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40354251257)
  at`49161f608af811154bfeb816ac38caea34224cae`, producer
  `40a82158146faf0cd44de2a4ee7056e7b0c6a58c`, is **negative**.
  [Windows](https://github.com/CrispStrobe/CrispEmbed/blob/0b0fded32b1dcbd75abd1451d63b08123520be9b/tests/results/packaging/successor-windows-reader-initial.json)
  stopped on a UTF8 header decoded with the Windows locale; the source-only
  repair reads UTF8 explicitly and uses MSVC`/utf-8`.
  [Apple](https://github.com/CrispStrobe/CrispEmbed/blob/0b0fded32b1dcbd75abd1451d63b08123520be9b/tests/results/packaging/successor-apple-initial.json)
  stopped on PNG identity before library/model initialization or inference.
  [Browser](https://github.com/CrispStrobe/CrispEmbed/blob/0b0fded32b1dcbd75abd1451d63b08123520be9b/tests/results/packaging/successor-wasm-initial.json)
  executed but fails unchanged parity: max0.00837269/cosine0.99864984.
  The [completed source trace](../tests/results/packaging/successor-wasm-source-trace.json)
  preserves all eight original output anchors, exact embedding norm and first
  captured attention0 divergence. Same-input Q8 rounding differs; full-network
  causality and correction remain under actual A/B, without budget widening.
- Initial Linux image proof40376735641 failed member coverage before comparing
  PNGs: the pinned renderer also emits a216-byte NOTICE. Corrected
  [40386968045](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40386968045)
  at`f284b1f67c1ae34082afb81a54301c6ab392c02a` preserves that notice and
  Python3.11.17 renderer ABI. The [reviewed input proof](../tests/results/packaging/posformer-corpus-proof.json)
  now passes all50 original encoded PNG hashes/manifest/dimensions/notice.
  [Mac-only40399358753](../tests/results/packaging/successor-mac-only.json)
  at7d561149 is complete: actual artifact/RGB checks and released17.12/default/1
  all50 IDs/text/EOS/confidences exact. All200 logs and legacy0 outputs verify;
  the7/50 recognition score is unchanged. Do not repeat this old40 diagnostic.
  Final integrated runtime still requires fresh archives and full Mac consumers.
  A Darwin RSS reporting-unit bug is repaired in the probe with raw units retained;
  old producer JSON is preserved and no speed/memory improvement is claimed.
- Conditional WASM rounding
  [40391544739](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40391544739)
  at`580e3ae3be3671ddce5e945b0725d0583ec34110` passes actual operators but its
  [complete model A/B](../tests/results/packaging/wasm-rounding-ab-v1.json)
  rejects8/10 final cases at unchanged0.002/0.99999. Legacy stages/finals,
  observers and interleaved contexts preserve anchors. Attention0 improves to
  max1.19e-7, layers0–2 below9.54e-7, then layer3 reaches0.00291893 and encoder_out
  reaches0.02850568. Retain raw artifact12692387342 remotely; trace the first
  layer3 operator with actual retained inputs and weights before changing scale,
  precision or defaults. Keep both converters and the per-backend gate off.
  Initial common-control failure40385214740 and queued cancellations stay retained.
- Review-only integration branch`integration/bounded-closure-20261026` combines
  the frozen lane sources. Producer branches stay unchanged, and historical
  whole-source guards intentionally reject changed runtime trees. Final integrated
  source has a separately reviewed [final-integrated-v1 validator](ovisocr2-final-integrated-v1.md)
  at5a843b06, with13 small controls and retained real-report reanalysis passing.
  Its exact f3f1b9e7/a06 source closure keeps both defaults off; actual reviewed
  Ovis/WASM completion and artifact/report pins are still pending and block model
  access. Any changed source/default requires an explicit new closure. Fresh
  seven native/two WASM producers and all actual consumers remain required;
  initial producer success is not that proof.
  The [aggregation repair](../tests/results/packaging/successor-desktop-vector-review-repair.json)
  now independently compares all four desktop vectors with unchanged gates.
  Its14 controls and retained Linux ARM/x86 cross-check pass; actual final
  four-platform aggregation remains required.

Fetch exact feature commits and source guards rather than reconstructing old
commands. This handoff is preparatory; it does not close the loop, qualify a
release/model portfolio or authorize a tag, model publication or app-pin update.
The app remains immutable0.17.12, and issue71's product scope remains pending.

For SciFact, the registered metadata phase`ultrabert-scifact-prepare` runs only
preparation. After independent review,`ultrabert-scifact-panel` forwards
`native_run_id` as preparation run and`native_artifact_name` as its receipt SHA.
Do not dispatch the full panel without the actual dataset/length/timing receipt.
For assets, use both completed producer IDs on the same candidate source;
`successor-consumer` forwards`native_run_id` and`original_run` to native/WASM
producer inputs, plus`candidate_run_id` as the reviewed PNG proof's`corpus_run`.
Registered`successor-corpus` calls the Linux image-only proof workflow. No
latest-run lookup, source substitution, tag or app-pin change.

Preserve every initial cancellation, HTTP service error and negative receipt.
Queued jobs are not evidence. Builds/inference/corpus/framework work stay hosted;
only small reports are needed for independent review. GitHub push/dispatch
HTTP500 errors were observed during SciFact registration; retry exact commits,
check remote refs and actual service run identities, and avoid duplicate panels.

## Newly requested permissive models

See [MoEdernColBERT](moedern-colbert.md) and [Josev](josev2b.md), their pinned
contracts and PLAN for exact staged acceptance. Smallest MoEdern original
40412164041 uses source790f7b1d; no native result yet. The corrected neutral
observer handles ST's direct backbone.forward. Josev first GPU preflight atb6e74739
rejected Python3.11 versus actual3.13 before payload access; preserve that attempt
and freeze a corrected runtime before original capture. Keep public Python versus
calibrated author-GGUF profiles distinct. Native/API/alias/retrieval/decision
quality and quantization remain explicit follow-ups, with no local heavy work.
