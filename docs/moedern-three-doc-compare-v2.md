# MoEdern retained three-document compare-only v2

Initial [40825166813](https://github.com/CrispStrobe/CrispEmbed/actions/runs/40825166813)
at `b86e8130947b59e09dfe4dde0e2dae01957c443d` saves all six ordinary fresh-original
final vector matrices, then fails on the adapter name `doc-24726600.all`. The
unchanged test writer permits only letters, digits, underscores and dots. It
rejects the hyphen before any model-vector actual Ref comparison, original
observer/repeat or native forward. Authored Ref positive/changed-row controls
passed separately. Preserve the [original producer](../tests/results/moedern-colbert/three-doc-initial-40825166813-original.json),
[hosted producer](../tests/results/moedern-colbert/three-doc-initial-40825166813-hosted.json),
[initial log](../tests/results/moedern-colbert/three-doc-initial-40825166813-log.log)
and [independent initial review](../tests/results/moedern-colbert/three-doc-initial-40825166813-review.json).

The adapter defect is distinct from these reported fresh-original errors against
the frozen full-original final vectors. All six fresh matrices are byte-different.
The exact checkpoint, runtime dictionary and32 loaded weight hashes match the
frozen original; the fresh host reports AMD EPYC7763, while the old full-original
receipt does not record its CPU model. Hardware/operator causality is unproven.

| Document | All/selected max absolute | Minimum row cosine | Fixed0.002/0.99999 |
| --- | --- | --- | --- |
| `24726600` | 4.24683094e-7 | 0.9999999999989746 | both pass |
| `24928817` | 0.0355934128 | 0.9907455389 | both fail |
| `5849439` | 0.0557457618 | 0.9740250689 | both fail |

[The new protocol](moedern-three-doc-compare-v2.json) and
[caller](../tools/ci-heavy/moedern_three_doc_compare_v2.py) recover measurements
from the retained artifact only. Old b86 helpers, guards, adapter name rules,
inputs, golds and0.002/0.99999 budgets remain unchanged. Explicit safe tensor IDs
`doc_24726600_all` etc. have a reversible mapping to original keys, filenames,
shapes and byte hashes; no float, shape or source case name is rewritten.

The [standalone workflow](../.github/workflows/moedern-three-doc-compare-v2.yml)
first authenticates exact failed run/source/attempt/job and both service artifacts:
small12875424745 (23,632 bytes) and raw12876890015 (2,701,071 bytes). It rehashes
only the already-retained eighteen matrix copies, packet/report identities,
text/IDs/three masks, document expansion-position lists, source-model identities
and span provenance. It inherits the old preparation's authenticated whole-blob
lineage; it does not download or rehash the old full corpus again. The original
ZIP copies retain the original dataset provenance/notices remotely.

Then it builds only the frozen header-only C++ comparator, records compiler/
command/binary hashes and installs only NumPy2.2.6. It executes three comparator
calls, six matrices each, with actual crispembed-diff::Ref and independent metrics:

1. Frozen original versus retained fresh original.
2. Frozen original versus retained native93.
3. Retained fresh original versus retained native93.

Every numerical rejection remains a record; collect all eighteen before the
final nonzero numerical exit. Structural completion and numerical qualification
are separate. Report absolute/RMS/mean errors, L2 and min/max magnitudes, first
bit/budget difference and worst component with original input token position/ID.
Verify selected vectors are exact scoring-mask slices of their own all-vectors.
Old-native/fresh-original proximity is diagnostic and cannot replace frozen gold.

No model, tokenizer, inference framework or original/native forward is accessed.
There is no GGML runtime build. The separately declared zero-forward analysis
admission is2GiB available RAM/2GiB disk; the old14GiB/24GiB model guards are
untouched. Keep all raw copies/adapters and structured failure evidence remote
for90 days. Source preparation authorizes no dispatch, new stage protocol,
precision recipe, model publication, default change, runtime release or app pin.
Any precision protection decision still requires Python intermediates, actual
Ref comparisons and final task-quality evidence under unchanged gates.
