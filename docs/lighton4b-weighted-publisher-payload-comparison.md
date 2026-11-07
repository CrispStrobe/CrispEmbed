# Weighted candidate versus published Q4_K_M payloads

After the verified whole-row mappings, **225 of 249 quantized tensor payloads
are byte-identical** between our original-F16 weighted candidate and the published
Q4_K_M import. This includes every Q6 tensor and the tied embedding/head. The
only 24 raw hash differences are recurrent output projections whose columns use
different coordinates. This narrows the conversion comparison; it does not
attribute decoded quality to those differences.

The [complete named comparison](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4km-candidate/quantized-payload-comparison.json)
records all 249 names, types, shapes, sizes, native/mapped/raw-upstream SHA256s,
source offsets, whole-row map hashes and source-file identities. It joins the
candidate's completed 723-payload audit with the completed physical-import audit.
The latter import finished before its later PyYAML setup failure; no model forward
or failed reference assertion is reclassified.

| Quantized payload group | Exact mapped SHA matches | Different coordinate SHA |
|---|---:|---:|
| Q6_K, including tied embedding/head |33|0|
| Q4_K |192|24|
| Total |225 (2,552,432,640 bytes)|24 (141,557,760 bytes)|

All 96 non-identity whole-row permutations on quantized tensors are accounted
for. The 225 matches include those mapped rows and the identity-layout weights.
The source of the native file is the authenticated unquantized F16 carrier;
it was not requantized from the publisher's Q4/Q6 file. Matching these emitted
quantized bytes is stronger than a type-schedule comparison, but lossy quantized
bytes cannot prove equality of the original floating checkpoint tensors.

## Exact sources and coordinate map

Native candidate SHA is
`5d133ae02c12611572c02e2d43bd06f4fecc5e059f94709a13bdb36458fadb29`;
its producer is `a4f2c908ffaf3f17eb770973017e3de8a08bb121`, with all 248 body
importance vectors actually applied. The original F16 source SHA is
`757ad5d0ea7699b3d0ce4cbb70978de442e5b95e660da0a13cf048980e255606`.

The publisher file at revision `8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01` is
2,708,804,576 bytes, SHA
`b1dd753e6f7d7d7ee3e86134337dd1ae7b08e2ed181989ae8264aa670d8ccc26`.
The completed import is
`843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43`,
with its exact [packer and mapping source](https://github.com/CrispStrobe/CrispEmbed/tree/a6644247c48cb25e1b7e4b8da9e8ff91644eaca7/tools).
Its mapping-profile SHA is
`cf3af1c7f7893a06cf1660358e3ec7084dd6352730e8bcfd384e749e4f11edb1`.
The comparison includes source-blob and audit hashes rather than inferring a
publisher conversion commit from its model card.

Each recurrent value head has 128 channels. For native HF coordinate `h*128+c`,
the physical coordinate is `(16*(h%2)+h//2)*128+c`. The activation bridge writes
`physical[perm[i]] = native_HF[i]`. Its 4096-entry uint32-le map has SHA
`d85c5067c02735fb9f5701d64a1f04ff43d45d20d57612f0a015a57b7ce72b5c`.
Every native 256-element block maps across two distinct physical 256-element
blocks; the 128-channel tiles themselves remain intact. Q4_K/Q6_K shared block
parameters prevent a generally lossless packed-block reorder at that boundary.

Crucially, the imported output matrices intentionally retain physical columns,
while the custom weighted candidate retains HF columns. Their different raw
hashes therefore do **not** measure canonical represented-weight error. A bounded
inverse-coordinate dequantized comparison, and ideally a direct original-F16
physical-order quantization control, would test that remaining question without
requantizing existing quantized weights.

## What remains separate

The floating text tensors, direct gamma/A semantics, vision features and compute
kernels are outside this 249-payload comparison. The
[reconstructed-prefix operator evidence](lighton4b-physical-prefix-primitives.md)
already separates scalar RMS and direct-quantized versus dequantized-F32
arithmetic; it is not an actual full source graph trace. The physical same-feature
decoder test and the original-HF candidate's full outputs remain separate causal
experiments. A legacy environment flag alone does not establish kernel selection:
the current native `project_cpu` materializes F32 and `project_rows_cpu` uses F32
tiles.

The [candidate's completed table/Arabic outputs](lighton4b-imatrix-q4km-full-calibration-results.md)
meet those observed calibration floors, with a remaining BF16 table gap. Form is
still pending and post-selection. This read-only metadata comparison downloaded
no model, ran no inference or quantization, and changes no quality claim, default,
public model fence or release artifact.
