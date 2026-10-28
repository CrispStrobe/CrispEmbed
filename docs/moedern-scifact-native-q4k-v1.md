# MoEdern full native F32 and padded Q4_K SciFact replay

This is hosted source preparation, with one explicit `1.5M`, `27M` or `115M`
variant per CPU session. No native full-corpus result or quantized retrieval
qualification is asserted by preparing this lane. All model gates remain off by
default. Publication remains false until the actual evidence is reviewed.

The worker authenticates the three separately reviewed original sources in
[`moedern-scifact-native-q4k-v1.json`](moedern-scifact-native-q4k-v1.json), including
the selected variant's exact immutable report, six complete raw-file hashes,
5,483 canonical identities, token IDs, masks, vector offsets and per-row hashes,
all 300 original rankings, and original source files. It also authenticates the
older ten authored raw texts, 20 vector arrays and three scores. Four original
query-expansion tensors prove the padding rule. The full SciFact producer did
not capture a separate expansion tensor; those positions are explicitly derived
from its observed supplied mask and the already frozen fixed-32 query rule.
The originals are read-only inputs; this lane never runs an original model.

The root library, CLI and server use LLAMAFILE ON, native/BLAS/CUDA/Metal OFF,
two build workers and two inference threads. The standalone test targets link
that root profile. Model-free Q4 controls, actual Ref positive/bad-row controls,
actual arithmetic compilation flags and ordinary-library observer exclusion
precede model access. A tokenizer-only metadata probe must reproduce every ID,
supplied/effective/scoring mask and expansion position for all 5,483 raw texts
before **any** embeddings for that model. Original IDs or vectors are never
supplied to the native text interface. The auxiliary family F32 control uses
IDs emitted by the actual native frontend.

Strict original F32 controls run first: the older authored panel and family
graph, then every full-corpus all-token and selected-token vector and every
300×5,183 MaxSim score. Eight lossless test-only adapters preserve contiguous
128-wide rows and end on original case boundaries. Both actual
`crispembed_diff::Ref` and independent NumPy metrics retain the unchanged maximum
absolute 0.002 and minimum token-row cosine 0.99999 gates, plus magnitudes, L2,
RMS and signed/absolute mean errors. Ref numeric exit 1 is collected across all
eight shards; structural rejection is distinct. Every per-row metric stays in a
remote raw journal. Adapter files can be reconstructed from the retained streams,
exact case spans and SHA256 identities; redundant adapter copies are removed
only after a successful structural comparison.

Only strict full F32 success permits the separately declared real padded-Q4_K
conversion. Actual types, all logical/storage shapes, 20/20/52 Q4_K tensor counts,
critical F32 bytes, tokenizer metadata and file type are independently checked.
The candidate repeats the authored and full-corpus tests. Twelve gate toggles
and recovery controls, actual invalid-input/output-clearing controls, repeat
identity, and actual C/Python/Rust/Dart/CLI/HTTP consumers use the existing
versioned helpers. The CLI supplies selected vectors; its all-vector shape-only
borrowed C data is excluded from cross-client bit-identity claims.

MaxSim remains F32 dot products and F64 query-token sums, with all 300 test
queries, all 5,183 documents and ascending document-ID tie breaks. Q4 numerical
drift stays explicitly separate from retrieval acceptance: every NDCG@10, MRR@10
and Recall@100 drop must be at most 0.001, and top-one agreement at least 0.99.
Six actual raw-text public score calls must also agree with independently
computed native-vector scores within 0.002. Passing a component cosine does not
qualify retrieval or publication.

The first retained original corpus-vector stage is `normalized_all`, followed
by `normalized_selected`. This evidence cannot attribute an internal attention
or FFN cause. A failure requires a separately reviewed targeted attribution
capture; no missing encoder states are invented or old gates relaxed.

The coordinator renders `@COMMIT@` and `@VARIANT@` in
[`kernel.py`](../tools/kaggle/moedern-scifact-native-q4k/kernel.py), attaches exact
frozen original outputs and the existing authorized artifact-read input for the
older 1.5M original if needed, checks all account sessions and capacity, then
submits one CPU task. No credentials or private input identifiers belong in Git.
The bootstrap rejects GPU allocation and unsupported Python/host resources,
uses five SHA-pinned Linux CPython 3.13 wheels in an isolated venv, checks their
dependency closure and owners, and caps the model task at nine hours. Measured
children use Linux `wait4` per-child RSS (KiB and bytes), monotonic wall time,
stable exit/signal/timeout records, and group kill plus reap. This is not a
concurrent process-tree RSS sum. The setup wrapper also retains every log and
partial failure. Large streams, models, token journals and binaries stay remote.

Expected small result: `moedern-scifact-native-q4k/result/review/native.json`.
Detailed Ref/per-row journals, client outputs, ranking matrices and candidate
GGUF are under the same result's `raw/`; original payloads remain in their
separately frozen producer outputs. No original recapture is authorized here.

## Initial 27M setup rejection

The first submitted source `8d9beb29317cb946202b4e9044e0cdb3fb27d298`
reached its actual isolated entry point after source, submodule, locked-package,
pip-check and 21 small controls passed. It then rejected during a stdlib import:
`authored_v1.py` loaded the frozen replay helper by filename without declaring
its sibling-import directory, so `contract_v2` could not be resolved. The worker
never entered `main`, created no native report, read no original arrays and ran
no compilation or model inference. This is a setup negative, not numerical
parity evidence. The unrelated `sitecustomize` warning about `wrapt` is retained
and no unlocked package was added.

[The exact bootstrap receipt](../tests/results/moedern-colbert/scifact-native-27m-initial-bootstrap.json)
and [version-bound small-file hashes and limits](../tests/results/moedern-colbert/scifact-native-27m-initial-setup-rejection.json)
preserve that outcome. The bounded correction declares only the already pinned
helper directory and exercises the actual `python -I worker_v1.py --help` path
before any wheel installation. A fresh-process stdlib regression explicitly
rejects framework/array/model package imports during that smoke check. Original
bytes, runtime, math, panels and gates remain unchanged. No replacement is
authorized until source review and a fresh capacity check.

## First 1.5M native build: archive discovery correction

The corrected-import worker completed source/original admission and the root
CPU library build, then failed while configuring the replay executables.
`find_library(GGML_CPU)` searched `ggml/src/ggml-cpu`; the pinned backend
`a06fc0bb` creates its default CPU target from `ggml/src/CMakeLists.txt`,
so the static archive is `ggml/src/libggml-cpu.a` in this build profile.
The correction searches that exact root-build directory with `NO_DEFAULT_PATH`
and `REQUIRED` retained. No alternate system library or inference change is
permitted. The initial configure failure remains an execution negative; no
model forward or numerical/retrieval result was produced. Retry this corrected
source only after review and fresh hosted capacity admission.
