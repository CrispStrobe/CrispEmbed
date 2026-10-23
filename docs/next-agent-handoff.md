# Executable handoff — updated 23 October 2026

Read this file, [PLAN](../PLAN.md), [the model contract](ultrabert.md),
[contributing](contributing.md) and [the shared consumer handoff](current-state-and-next-steps.md).
Completed evidence belongs in [HISTORY](../HISTORY.md); measured limits are in
[PERFORMANCE](../PERFORMANCE.md). This is a handoff, not a claim of production
UltraBERT support. The app still consumes immutable CrispEmbed0.17.12.

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
  diagnostics. All initial failures are preserved. No production dispatch exists.
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

## UltraBERT attribution — M1 experiment owned externally

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

The user is executing the bounded M1 audit atf0dbe58d; await its raw reports and
check every original-stage/projection and raw/normalized CLS record. Do not
duplicate that experiment. Next isolate original projection/reduction reproducibility using fixed input
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

### U3 — Wire a production UltraBERT profile after numerical qualification

Use [the pinned architecture contract](ultrabert.md) and all contributing surfaces:
new default-off `core_env::on` backend, architecture detection, C ABI, CLI/server,
Python/Rust/Dart and model registry. Preserve literal0 behavior and existing
encoders. Reuse primitives, not unrelated model defaults.

Start from the tested [tokenizer profile](../tests/test_ultrabert_tokenizer.cpp).
Actual integration needs GPT-2 splitting, one BOS0, separator15 only between
pairs, no terminal EOS, explicit query-prefix choice and attended BOS despite
sharing pad ID0. Generic `BPETokenizer::encode_pair` is a BERT template and appends
a final separator. The encoder loader does not select GPT-2 splitting for
`pre=gpt2` as the decoder loader does; route explicitly for this profile. Do not
make an untested global tokenizer change. CLS pooling is neither last-token nor
mean pooling; raw and optional normalized output remain separate.

Acceptance: normal public text APIs must reproduce the frozen ID/mask and native
intermediate/final-vector panel, including query/document, Unicode, empty, pairs
and batching. A metadata-only tokenizer fixture or prepared-ID graph alone is
insufficient. Keep the initial512-token bound explicit and reject overflow;
32K/GPU support needs separate memory/geometry/parity measurements. Native lacks
48 original RoPE-table captures and post-RoPE Q/K; add original-source probes
only as needed and check old anchors before relying on added arrays.

### U4 — Independent quality, quantization and publication

After U3, execute the SentenceTransformer wrapper and compare raw CLS, optional
normalization, named/default prompts and ordering. Freeze independent retrieval
inputs/gold before candidate outputs; report task metrics separately from numerical
parity. Then evaluate longer contexts and GPU/latency/RSS on appropriate resources.

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

Next use the [bounded M1 handover](handover-m1-validation.md) for an independent
CPU/BLAS comparison, or a separately justified process-level linear-dispatch
experiment. Record actual CPU/library dispatch before attribution; do not repeat
thread/MKLDNN settings or choose new reference arrays from the alternatives.
The original .99999/.002 gates and native183/1294 failures remain unchanged.

## Other executable fronts and boundaries

| Lane | Next bounded action | Completion evidence / limits |
| --- | --- | --- |
| Integration CI | Completed39009694933 at f6f84aba: all10 Build jobs including Flutter; WASM/lint/contracts pass | Scheduled health/OCR regression green through22 October; rerun affected checks for new code |
| PR68 deployment | Completed39009694992; live metadata/PNG verified23 October | Deployment revision f6f84aba; no new app/runtime release |
| PosFormer PR73 | PR72 merged7f1ecd33; actual released/candidate50-case IDs/text/EOS match. Default restoration790f9a6b is queued in [PR73](https://github.com/CrispStrobe/CrispEmbed/pull/73) | Require full exact-source checks, Linux actual-default/release oracle, gate and bounded reference reports before merge;7/50 quality unchanged |
| Version/release | Development main0.17.13; published/app0.17.12; actual Linux compatibility measured | PR73 proposes restoring the published trained default with legacy0 retained. Actual successor artifact/hash/consumer qualification remains required before any tag |
| Qwen OCR numerical debt | Use original-embedding/vision isolation and controlled linear-reduction probes in PLAN | Keep924 historical records/56 failures, held-out F16 six/13 failures and all decoded/quality gold/caps; no numerical waiver |
| OCR quantization/publication | Independent development/quality work after the completed nine-run Q8/mixed/Q4 panel | No tested recipe currently qualifies; Q4 receipt truncates/unclosed HTML. CER/WER, cells, completion, confidence and format matter, not cosine alone |
| Permissive OCR queue | Audit and reuse the smallest applicable shared backend, starting OvisOCR2/Xiaomi-OCR-0 | PLAN gives licenses/storage/dependencies; recheck exact checkpoint/base terms and original blueprint before weights; Indra stays public-code-only |
| Existing encoders | LFM2.5 official F16 CPU/CUDA T4 audit passes594 direct checks/both15-case API panels; [report](../tests/results/lfm2-encoder/cuda-f16.json) | Other GPU/precision/MMQ, paired latency, longer contexts and independent quality remain open; typed Gemma GPU/task/performance is separate |
| OCR portfolio/app | Select one bounded portfolio task from PLAN or app lane in shared handoff | Retain original fixture/metric/decoded acceptance and owner; SAM work belongs to its existing owner; no unrelated dependency upgrade |

All builds, suites, model/batch/browser work run on hosted CI or authorized remote
GPU resources. On shared infrastructure, first inspect load/RAM/swap/free space.
Use isolated worktrees, preserve unrelated files/owners, and keep private machine
setup/access notes ignored. Optional ignored environment notes may supplement
this public handoff; the task definitions above do not depend on them.
