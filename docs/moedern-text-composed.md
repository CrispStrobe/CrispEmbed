# Prepared MoEdern1.5M raw-text composition

This separate source lane is based on Unigram preparation
`3da7cf8e9a4fcd6342fc9a6ac00d07de42b7c324`, including the exact strict-dictionary
recorder and historical normalized-role-matcher corrections. No old frontend,79/v2 kernel, producer, guard, golden array,
root CMake, registry, C API or binding file changes. Unigram execution/review
identities are deliberately pending in the [new contract](moedern-text-composed.json).
The actual original-first tokenizer producer must be reviewed and explicitly
pinned before this lane can access any dependency API, archive or model payload.
No dispatch or composed parity is claimed by source preparation.

The internal `moedern_text` context accepts a separate exact tokenizer component
and old strict32-tensor F32 model. All three value-parsed gates are required:
`CRISPEMBED_MOEDERN_TEXT_COMPOSED`, `CRISPEMBED_MOEDERN_TEXT_FRONTEND` at frontend
load, and `CRISPEMBED_MOEDERN_COLBERT`. They remain default off. The public-internal
header offers `encode(context, raw_text, query/document, result)` and
`score(context, raw_query, raw_document, value)`; neither accepts IDs, hidden
states, embeddings or reference callbacks. Output contains exact input IDs,
supplied/effective/scoring masks, query expansion positions and all/selected
128-dimensional normalized rows. Scoring includes the original CLS/SEP/role
policy; no generic pooling or token dropping is applied. Failures clear outputs.

A new internal namespace/file copies the bounded v2 engine. Its deterministic
adaptation is source-checked against the immutable original: namespace and
interface changes, test-only callbacks/types/input snapshots/labels, and generic
ordinary finite-error text. Algebra, operation order, F32 weights, CPU operations,
routing, norms, selection and MaxSim stay unchanged. Ordinary preprocessing
removes every callback and stage-label argument; existing finite checks remain.
The attention probability buffer remains actual math workspace, not a retained
reference artifact. Test hook declarations live in a separate macro-guarded
header, and the ordinary API has no observer argument or testing types.

The standalone build creates ordinary and test archives without registering any
root library or public entry point. Actual `nm` and string controls must exclude
reference callback symbols, `std::function` and selected stage labels from the
ordinary archive, with a positive observed-export control. Raw-text ordinary,
observed and repeated all/selected outputs must be byte-identical; callback
failure/recovery, gate0/recovery, role/bounds rejection, score reset/recovery and
two-context destruction/ownership are checked on actual compiled components.
All55 actual engine observations per case are retained only in test artifacts.

The new hosted workflow first rejects pending Unigram identities before any
network request, optional package or build. Once populated after independent
review, it authenticates exact service/source/whole-ZIP/member identities for
the complete tokenizer result, original report and component. Explicit pinned
member names handle artifact namespace roots; no latest lookup, path guessing or
historical-guard waiver is allowed. The measured frontend source/header must
also equal this source before its component is admitted. All ten tokenizer
streams must match unchanged original IDs and three masks before old model
payload transfer. Exact successful native-v2 lineage remains separate from
original and supplemental-observer continuity evidence.

The only model is the immutable F32 original GGUF from successful original
[40412164041](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40412164041),
SHA256 `9f9792f0ae76430496ac93951cc09a0804468ba3f22b4314b303aea85dfbe8fa`,
with its exact conversion receipt and old primary receipt `b50ca1369f45...`.
No original framework, weights conversion or original forward is repeated.
Only raw text and explicit roles reach the composed executable. Expected IDs
and old arrays remain independent comparison inputs, never model inputs.

All ten complete and selected final-vector targets (20 F32 arrays), three old
MaxSim scores and all exact input/mask contracts stay frozen. Every vector uses
actual `crispembed_diff::Ref` and independent NumPy metrics under unchanged
max-absolute0.002/minimum-row-cosine0.99999. Ref equal/orthogonal-row controls
precede the panel; numerical rc1 is retained. Raw-text scores compare separately
against old scores and independent query-row-max summation at0.002. Every case
and pair is attempted, and structural and numerical failures remain distinct.
No batch axis is removed or original array reshaped. New callbacks do not make
additional intermediate original-target claims; native-v2's explicit112 mapped/
138 unmapped coverage and rejected old byte continuity remain unchanged.

Execution is hosted-only, CPU/F32, one/two threads, two build workers,60-minute
bound with2GiB available RAM/4GiB free disk. Only array tooling is installed;
no Torch, Transformers, Sentence Transformers or tokenizer is reexecuted here.
Small reports and actual arrays/adapters/binaries/archives are preserved on every
outcome. This does not qualify public APIs,27M/115M, quantization, representative
retrieval, runtime defaults, release or application adoption. Later explicit
public integration must retain these role/selection semantics and requires
actual reviewed composition first.

The initial tokenizer comparison
[40449174879](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40449174879)
remains rejected: two of34 owned overlap cases omitted retained normalized role
matchers. All ten original streams and1040 normalization compositions passed,
but that result cannot admit a composed model run. The corrected producer must
independently pass all34 cases, the unchanged ten Unigram units, four input
negatives and all six metadata negatives before its explicit identities can
replace the pending sentinel here. Original gold is never replaced by native
outputs or by an earlier partially passing tokenizer producer.
