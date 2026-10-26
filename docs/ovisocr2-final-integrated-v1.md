# Prepared final integrated Ovis validation v1

This lane is source preparation only. It has no dispatched build, model result,
default decision, release qualification or app upgrade. The
[protocol](ovisocr2-final-integrated-v1.json) deliberately contains pending
review dependencies and rejects before service/model access until those exact
reviewed identities are supplied in a new audited commit.

The runtime/build/binding closure is the immutable integration
`f3f1b9e7d5385d900b68bdf4bd14677425a15eee`, including all source files and
ggml`a06fc0bb46b73cc10011fc90ae4054aee8438aeb`. It explicitly pins vision
SHA256`bc479ff5…`, encoder`99bb8a23…` and UltraBERT profile`fc5af20f…`.
The protocol records complete Git object IDs, including build recipes, native
comparison programs, regression gold, relevant frozen receipts and shared
Python parsers. Added/deleted/dirty production source and altered dependency
checkout reject. Later auditor documentation may differ; production or default
changes require an explicit new closure, never a latest-source substitution.

Both interpolation and WASM rounding remain default-off at this declaration.
Before dispatch/model access, root must fill two reviewed-completion records:

- Actual Ovis composed result: immutable run/source/workflow/conclusion plus all
  required small artifact ZIP IDs, names, bytes, SHA256s and selected report-member
  SHA256s.
- Actual WASM rounding A/B outcome with the same complete service/report pins.

Each record must explicitly confirm the experimental panel completed (a
structural pre-model failure is not a completed panel), and distinguish `candidate-supported` from
`candidate-rejected-default-off`, and explicitly retain the final default off.
A reviewed negative experiment is retained evidence, not qualification or
permission to enable it. The validator verifies completed service state and
every declared ZIP/member identity before any model transfer. The human review
disposition is an explicit immutable protocol input; it is not inferred from a
workflow's green status. This lane always reports release qualification false.
Any later source/default choice needs a separately reviewed source closure.

The registered metadata phase is`final-integrated-v1`; it calls
[the dedicated workflow](../.github/workflows/ovisocr2-final-integrated-v1.yml).
One portable two-worker native build precedes two interpolation-gate frozen
jobs. Both frozen jobs finish before the two quality jobs start, keeping total
model parallelism at two. Quality observations still run after a completed
numerical rejection, while explicit workflow cancellation stops subsequent
work. All four jobs must pass their own declared checks before the bounded
final-source scope can be reported positive.

Each frozen job executes the unchanged fourteen-arm F32/F16 matrix: delta,
vision, decoder prepared/native-vision/native-positions/native-frontend and
owning context, plus bounds/recovery and owning OCR controls. It reuses original
run40041958521/`cc038b00`'s943-stage reference and immutable exports from
39992662147/`5e197bae`; no original forward or reference recapture occurs.
Reference GGUF SHA256 remains
`e8e20c654645092ef11e8632db6d84393761df3f8b1d34bc48ae1ddac668e588`.
F16 model SHA256 remains
`d3a733a5d5ba5d63f5275500fad4cff585b82e2fb90d1def2fe767d04edf57db`.
Gate0 also requires the retained ordinary vector hashes exactly. Existing
calibrated numerical gates and all raw strict vision flags remain unchanged;
the wrapper does not reinterpret a strict flag as a new tolerance. All command
logs, exits, raw vector manifests and arrays survive failures. This frozen-only
lane makes no new1024/1200/2048 grid qualification claim.

Each quality job calls the unchanged full F16 phase in
`tools/ci-heavy/ovisocr2_quality_run.py`, reusing original quality40054043629
and its preserved document pixels. It does not choose a short smoke control.
All eleven cases,1,841 IDs, raw/processed text, stops/EOS/caps, input PNG/RGB
hashes, first-F16 scores and paired screens must equal
[the first native F16 receipt](../tests/results/qwen35-ocr/ovisocr2-f16-quality.json).
Gate0 additionally requires every selected confidence and mean exactly.
Gate1 requires finite selected probabilities with exact lengths and a valid
recomputed mean; all confidence/mean drift is reported against that same first
receipt, without inventing an exactness claim or a new confidence tolerance.
The actual C ABI smoke must also provide matching text and valid selected
probabilities/mean. Numerical, quality, source and default conclusions remain
separate.

The new driver reuses low-level `source_and_exports`, `source_reference`,
`native_arms`, `raw_manifest` and `timed_command` helpers. It does not call or
monkeypatch the historical position `run`/`preflight` or widen any old allowlist.
`current-replay`, historical native replay and position workflows retain their
original source contracts. The new guard explicitly accepts only the declared
assembled source rather than pretending it is unchanged from an older model
qualification.

Ten lightweight stdlib controls exercise pending dependencies without network
access, exact source closure, new/dirty source rejection, reviewed-negative
default handling, full case coverage, confidence policy and numerical/score
negative cases. No local compiler, image rendering, model/framework or raw-array
download is needed. Large model/reference/diff artifacts remain hosted; reports
and manifests are sufficient for independent review. Published F16, historical
precision/quality failures, UltraBERT numerical debt and the app's immutable
0.17.12 pin remain unchanged.
