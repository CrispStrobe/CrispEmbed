# Standard GGUF file-type metadata repair

This source-only successor fixes a file-header label. It changes no GGML enum,
tensor encoding, tensor selection, numeric conversion, model math or original
reference. Actual tensor Q4_K type remains12. The writer previously copied its
internal `ggml_ftype` into `general.file_type`; the standardized metadata enum
uses different numbers. Frozen producers, conversion receipts and numerical
rejections remain immutable. A corrected metadata label does not qualify a
numerically rejected model.

The authoritative [GGUF specification](https://raw.githubusercontent.com/ggml-org/ggml/6af560d55df03ad92116e3c0a697779584477e85/docs/gguf.md)
(43,175 bytes, SHA256 `1dead27b6a522709f0127d194e58c21dbbbf00ba1c64fe37c54d1a9048b05020`)
defines Q4_K_S14, Q4_K_M15, Q5_K_S16 and Q6_K18 for the file header. Extended IQ
values are pinned to [official llama.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/63412d8737b3efa30748dd79ceb2c93487e19f2b/include/llama.h)
(90,962 bytes, SHA256 `c0854771d3760a41642c4e59429c5b3b9fc774887c59d37eeadc4527681732cd`).
The shared constants map every exposed CLI method explicitly: Q4_K12→14,
Q5_K13→16, Q6_K14→18, IQ4_NL19→25, IQ4_XS22→30; existing F16, Q4_0/1,
Q5_0/1, Q8_0, Q2_K and Q3_K labels stay unchanged. The default writer now emits
standard headers. `--legacy-file-type-header` is an explicit reproduction option,
with a warning; it is not a publication mode. No mixed-M recipe is claimed.
Custom recipes must disclose actual precision counts/roles. In particular,
Josev recipe C has316 F32,118 F16 and only8 Q4_K tensors; do not advertise it as
a generally four-bit model. `general.file_type` is optional summary metadata;
actual tensor inventory and the immutable recipe identify precision.

Josev's bounded scanner and loaded metadata validator, UltraBERT's profile, and
MoEdern's padded-Q4 profile accept standard14 or known legacy12. All existing
quantization gates, architecture/source metadata, critical precision, real-Q4
presence and shape checks remain. A generic Q3/Q6 file is not admitted merely
because its header integer collides. MiniCPM and Unlimited use actual tensor
types rather than this file header; their new publication contracts still need
canonical metadata. Valid F32 behavior is unchanged. No gate default, C ABI,
release version, app0.17.12 pin or model alias changes here.

## Frozen source boundaries

The whole baseline is633f435bdee220a7f424a9cfcf54847f5bf20878, with source tree
`0c4341808c7136b84f0085da85e8c22b951c27a1` and exact a06 GGML. It already
contains six MoEdern padded-Q4 source differences from9c475d0d; those objects
are preserved and declared separately. All other baseline source, including the
Josev/Qwen arithmetic, matches9c. This successor changes only three profile
metadata headers and the writer, and adds the shared standard constants header.
The source closure pins all runtime/build/old-scorer/frozen-reference inputs,
all six existing MoE object pairs, and every new helper. It never widens a
historical whole-tree guard. Current mixed recipe producer633/06 stays frozen.

## Canonicalization input and proof

The new stdlib helper consumes a separately reviewed descriptor with schema
`crispembed-gguf-file-type-input-v1`, family, recipe, exact producer source,
producer report SHA/bytes, exact source-model SHA/bytes, explicit report JSON
paths linking that model/recipe, architecture/profile markers, and a complete
inventory. Each inventory row has name, semantic_shape, stored_shape, numeric
tensor type, bytes and payload SHA. All identities and payload hashes are
verified **before copying**. This authenticates a selected producer; it does not
replace the native model's strict profile validator or establish task quality.

Only a known legacy-Q4 UINT32 value12 may become14. The helper creates a fresh
file, never mutates or aliases its source, and patches only the existing
four-byte value. It verifies unchanged file size, data offsets, tensor ranks,
shapes/types/payload hashes, other raw metadata, and whole prefix/suffix bytes
around that span, then atomically retains the canonical file. It never infers a
migration from integer14 alone: that can mean internal Q6 or standard Q4.

The immutable output schema is `crispembed-gguf-file-type-canonicalization-v1`:

- `completed`, `family`, `recipe`, `producer_source`, `producer_report`;
- `source_model` and `canonical_model`, each `{bytes, sha256}`;
- `allowed_changes`: exactly one `{key: general.file_type, kind: UINT32,
  offset, old: 12, new: 14}`;
- `unchanged_prefix_sha256`, `unchanged_suffix_sha256`,
  `unchanged_metadata_sha256` (sorted JSON map of other-KV raw hashes);
- `tensors`: complete rows with `name`, `semantic_shape`, `stored_shape`,
  `storage_rank`, `type`, `offset`, `bytes`, `source_sha256`, `canonical_sha256`;
- `payloads_byte_exact`, `semantic_shapes_unchanged`, `source_preserved`,
  `metadata_only_bridge`, and `original_rerun: false`;
- `input_descriptor_sha256`: SHA of sorted compact canonical JSON of the input
  descriptor; CLI/bridge also add `input_descriptor_file_sha256`;
- `runtime_source`, `source_profile_sha256`, `numerical_qualified: false` and
  `publication_qualified: false`.

Physical rank is not confused with semantic shape: old Josev residual scorer
[512,1] may already be stored as rank1. The canonicalizer preserves the actual
stored rank exactly; it does not restore/change that earlier canonicalization.

## Hosted controls and old-four bridge

The standalone workflow builds only a tiny actual C quantizer/fixture and
compares all13 exposed methods' legacy/default outputs. Header values, actual
tensor types and all non-header bytes are checked. These builds are hosted,
not on the shared development host. Tiny stdlib authored fixtures cover
identities, duplicate keys, shapes/extents, bad storage, ambiguous header14,
source/destination aliases and the exact authorized repair.

`tools/ci-heavy/gguf_file_type_bridge.py` is a separate hosted, pre-staged-input
entry point. It requires the exact declared source, locked CPython3.13/NumPy
profile,14GiB available RAM/64GiB disk, retained original251 arrays, immutable
published F32, reviewed legacy model/descriptor/producer report and a freshly
built public-library SHA/source proof. It repeats **native** old-four strict
F32 control before the migration/candidate controls; it never repeats original
forwards. Both legacy and canonical models run plain/observed/repeat full227
actual Ref plus independent NumPy comparisons. Every captured stage and full
response must be bit-identical between the two headers. The actual C ABI runs
all four raw requests, gate-off/NULL negatives and recovery on both models;
all response bytes must agree. Ordinary archives/public library must exclude
Josev observer symbols. Public C ABI calls are exercised through ctypes; this
does not claim a new compiled C-client/Rust/Dart or release audit.

The report schema is `crispembed-gguf-file-type-same-payload-v1`, with
`runtime_source`, `source_profile_sha256`, `canonicalization_report_sha256`,
`source_model`, `canonical_model`, `four_cases` per-mode stage/response equality,
`all_mapped_stage_bytes_exact`, `all_typed_response_bytes_exact`,
`all_public_capi_response_bytes_exact`, actual `public_capi` library/gate/recovery
results, and explicit false numerical/publication qualification. Both strict
original numerical panels and all preexisting Q4 negatives stay visible.

No selected models or actual bridge results are declared by this preparation.
NEW20 owner must separately pin and review exact successor runtime objects,
canonicalization proof, legacy/canonical whole-model hashes and same-payload
report before any new holdout output. Ultra/MoE actual model continuity and their
unchanged vector/ranking/task gates remain necessary before publication. Future
converter wrappers need explicit new standard metadata contracts; old converters
that assert12 are intentionally not edited or silently bypassed. Source controls
alone do not qualify any model. No workflow is dispatched here.
