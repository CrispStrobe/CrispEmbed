# Executable handoff — 17 October 2026

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
  Deployment must be checked at its actual deployed revision.
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

## Start here: verify the corrected audit, then one bounded correction

### U1 — Finish the corrected same-input operator audit

Initial [39008070299](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39008070299)
failed at `a349539e01701f53144b100623b67bcd62489436` after compilation,
25 model-free controls, five Torch controls and all seven local captures passed.
The first empty F32 ordinary/probe metrics agree, but merged stderr split a
buffered OPERATOR record before parsing. No completed 14-arm result exists.
[Raw producer report and failure provenance](../tests/results/ultrabert/initial-local-audit-failure.json)
retain this failure. The correction flushes stdout before the auxiliary reference
loader writes stderr; strict parsing and all numerical gates remain unchanged.

Corrected [run39009703362](https://github.com/CrispStrobe/CrispEmbed/actions/runs/39009703362)
is queued at integrated source `f6f84aba6ba22b27b9f15cfc08f1d182524062d2`.
Inspect this run and exact source before dispatching anything:

```sh
gh run list --workflow ultrabert-native.yml --limit 5
```

If none exists for the corrected source, dispatch once (never duplicate an active
run), recording its immutable source and run URL:

```sh
gh workflow run ultrabert-native.yml --ref main -f reference_run=38997562520 -f operator_audit=true -f linear_audit=true -f linear_local_audit=true -f linear_forward_audit=false
```

Workflow: [ultrabert-native.yml](../.github/workflows/ultrabert-native.yml).
Driver: [ultrabert_native.py](../tools/ci-heavy/ultrabert_native.py).
Original operator observer: [dump_ultrabert_linear_reference.py](../tools/dump_ultrabert_linear_reference.py).
Native comparisons: [test_ultrabert_diff.cpp](../tests/test_ultrabert_diff.cpp),
using `crispembed_diff::Ref`, not a parallel parity framework.

Fetch only `ultrabert-native.json`, `linear-reference/linear-reference.json` and
28 small ordinary/operator logs from artifact `ultrabert-native-<corrected-run-id>`.
Inspect archive members with ZIP ranges, or extract the small members on a
hosted runner. Do not download model/reference arrays onto a shared host.

Acceptance: verify run/source, all file hashes, 7 local-reference records, original
archive/conversion replay, 14 native arms, 700 exact controls, 84 old OPERATOR
records and 42 additional LINEAR records. Reparse logs using `parse_log`,
`parse_operator_log` and `parse_linear_log(..., local_operators=True)`; require
ordinary probe-on/off metrics/exits identical. Local targets use the original
SiLU/down modules with fixed frozen gate/up values and identical replayed products.
`layer_14.local_down` is an additional operator target; it never replaces frozen
`layer_14.ffn_out`. Retain their discrepancy and all ordinary failures.
The corrected source still needs its actual 14-arm hosted conclusion. A failed
numerical job can contain completed measurements; inspect its report rather than
equating workflow failure with a build error.

If the run fails structurally, retain raw reports/log hashes, fix that specific
failure, and rerun one immutable source. If complete, archive every producer
field and its original report SHA, summarize local versus composed errors/norms,
and select only the next earliest justified operator experiment. No weight,
prompt, case, cap or tolerance fitting. Do not repeat an already queued run.

### U2 — Attribute remaining numerical differences

[Initial native report](../tests/results/ultrabert/initial-native-parity.json):
F32 has183 intermediate failures; F16 has1,294. F32 raw CLS cosine rounds1.0,
max error6.72340393e-5; this does not waive layer errors. Fixed gates remain
cosine0.99999 and max absolute0.002. F32 first threshold exceedance is layer14
FFN output, after smaller layer0 projection/norm differences. Compare magnitudes,
RMS, exact stage order and final raw/normalized CLS, not cosine alone.

[Original-input operator report](../tests/results/ultrabert/operator-audit.json)
passes41/42 F32 probes: Unicode whole MLP error0.0048828125 drops to0.000671386719
with frozen gate/up. F16 retains24 local failures. U1 can distinguish local
SiLU/product/down differences from projection/upstream amplification; replay
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

## Preserve original arrays before artifact expiry

Approved producer [38997562520](https://github.com/CrispStrobe/CrispEmbed/actions/runs/38997562520),
source `0573f1847db68914052a98341a22fceeb1358518`, artifact
`heavy-38997562520`, expires **31 October2026 03:09:55 UTC**. Before expiry, copy
it to durable authorized remote storage/longer-retained hosted artifacts. Verify
the original raw manifest SHA, all7 archive hashes, stage payloads and conversion
identities against [reference-audit.json](../tests/results/ultrabert/reference-audit.json).
Keep the original producer provenance when adding a new storage locator. Never
replace expired tensors with a newly computed reference under the same identity.

## Other executable fronts and boundaries

| Lane | Next bounded action | Completion evidence / limits |
| --- | --- | --- |
| Integration CI | Inspect new main's exact Build/Flutter, WASM, lint, contracts and secret runs | Component source checks above are not proof of new merge-head conclusions; fix actual integration failures, no local build batches |
| PR68 deployment | Inspect the Pages run from the merged source and actual production HTML/PNG | Absolute OG/Twitter URLs, PNG200/image/png/1200×630 and served revision; PR CI alone is not deployment |
| PosFormer PR60 | Port only trained input LayerNorm and ceil-pool correction from78493a31 onto current main | Fresh original/reference/decoded native tests; preserve unchanged7/50 result, existing public API and stable tag; do not merge its old release branch wholesale |
| Version/release | Reconcile main0.17.11 with published0.17.12 before a future release | Repository bump procedure, native/WASM hashes and actual consumer validation atomically; no app pin or PyPI change as cleanup |
| Qwen OCR numerical debt | Use original-embedding/vision isolation and controlled linear-reduction probes in PLAN | Keep924 historical records/56 failures, held-out F16 six/13 failures and all decoded/quality gold/caps; no numerical waiver |
| OCR quantization/publication | Independent development/quality work after the completed nine-run Q8/mixed/Q4 panel | No tested recipe currently qualifies; Q4 receipt truncates/unclosed HTML. CER/WER, cells, completion, confidence and format matter, not cosine alone |
| Permissive OCR queue | Audit and reuse the smallest applicable shared backend, starting OvisOCR2/Xiaomi-OCR-0 | PLAN gives licenses/storage/dependencies; recheck exact checkpoint/base terms and original blueprint before weights; Indra stays public-code-only |
| Existing encoders | LFM2.5/typed Gemma independent task accuracy, GPU and long-context/performance lanes | Their dedicated model docs and frozen references; publication already done does not establish independent task quality |
| OCR portfolio/app | Select one bounded portfolio task from PLAN or app lane in shared handoff | Retain original fixture/metric/decoded acceptance and owner; SAM work belongs to its existing owner; no unrelated dependency upgrade |

All builds, suites, model/batch/browser work run on hosted CI or authorized remote
GPU resources. On shared infrastructure, first inspect load/RAM/swap/free space.
Use isolated worktrees, preserve unrelated files/owners, and keep private machine
setup/access notes ignored. Optional ignored environment notes may supplement
this public handoff; the task definitions above do not depend on them.
