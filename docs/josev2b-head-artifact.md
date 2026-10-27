# Josev original preservation and head artifact

This hosted CPU lane preserves the reviewed original output and converts only
its separate joint head. The original receipt is
`51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090`,
worker`b2edf8c83c9aca1f8a9210198d47798ac13828b8`, model revision
`6f0a31c8ed683aba79097eb3f48df2c9ab560afd`. The checked-in small receipts contain
four neutral/repeat cases,251 ordered array declarations and122 exact
checkpoint-promoted/loaded F32 tensor identities. Raw arrays stay remote.

The [workflow](../.github/workflows/josev2b-head-artifact.yml) uses a configured
private read capability for the original output, without embedding operator or
access references in Git. The helper first verifies the public original receipt,
then requests the exact version through every page of the underlying SDK.
Kaggle2.2.2 parses an explicit version in `kernels_output` but drops it from the
request; [the upstream reproduction](https://github.com/Kaggle/kaggle-cli/issues/1200)
identifies `version_label="v<N>"` as the actual API contract. Independent metadata
probes accept`v5` and reject bare`5`/`latest`. This helper sends`v5` explicitly and
requires the exact original receipt before fetching array payloads. There is no
latest-version fallback. Signed download URLs remain in memory.

All251 NPY files must match exact sizes, hashes, NPY1 headers, shapes, F32/I64
dtypes and finite/mask policy. Ordered `(case,stage,occurrence)` descriptors
preserve repeated per-field module calls. The native input export retains raw
C-order float bytes: hidden input uses the declared singleton batch1 rows, and
each field keeps its own lexical means and exact question/option spans. The
source `[1,T,2048]` observer shape remains unchanged; the typed input interface
receives its contiguous `T*2048` storage. Inputs use
`crispembed-josev-head-input-v1`, original integer kinds and ordered fields.

The component converter fetches the61MB published head under its exact
size/SHA guard, promotes BF16 exactly to F32 and independently checks every
written GGUF dimension and byte hash against the actual original loaded head.
It preserves all eight original source/configuration/license notices. The
artifact architecture remains `josev2b_head_reference`: this is not a full
backbone, tokenizer or public model alias, and no extra temperature is applied.
See [the conversion contract](josev2b-head-conversion.md).

Small review artifacts contain completion/error phase, full file hashes,
conversion proof and native-input adapter provenance. A separate remote artifact
retains all original arrays, the derived inputs and head GGUF/source notices.
Seven standard-library controls cover exact reference guards, versioned pagination,
malformed NPY layouts and independent per-field byte copies; eight existing head
controls remain unchanged. None of those local controls executes payload fetching
or conversion. Actual hosted completion and native numerical parity are pending.
