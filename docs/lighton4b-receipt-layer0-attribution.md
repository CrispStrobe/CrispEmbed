# Receipt layer0: measured preferred-Q4 tail mismatch

The [same-input operator receipt](../tests/results/lighton4b/receipt-layer0-operators/same-input-operators.json)
binds unchanged runtime `a8a22749ab0b757ae1f317b6fa5b84ffd148b939`, tools
`b148dd84f`, original producer receipt
`eace978c3675c91002ea1ed68fe8d763b7da4a44021c6b5390c77501568cee70` and
outer receipt `31bef6429668151f3b1f108aec934ac1dfb78941a58c9f48c19956818d555910`.
The bounded hosted CPU worker loads only one layer's authenticated parameter
ranges, repeats no source/model forward, and retains102 artifacts/101,461,199bytes
remotely, including the actual component executables and libraries.

Before attribution, the unchanged production layer0 component reproduces every
one of256 retained native hidden rows byte for byte. Duplicate production GDN
execution reproduces attention, recurrent state and convolution FIFO bits.
Actual source/native image incoming state and FIFO are also exact after the
recorded head/state layout conversion. All242 image input-normalization and
Q6 QKV rows match the source exactly.

The first differing captured operator boundaries are the sibling preferred-Q4
Z, alpha and beta projections, on image rows232,233,240,241. Source/native
physical head ordering is canonicalized before comparison.

| Image stage | RMS error | Maximum absolute error |
|---|---:|---:|
| Z | 4.6618477831e-8 | 4.2915344238e-6 |
| Alpha | 1.1570377195e-7 | 2.8610229492e-6 |
| Beta | 5.1129103929e-8 | 1.4305114746e-6 |
| GDN gated output | 7.4318699848e-9 | 6.2379986048e-6 |
| Layer0 hidden | 9.7245843916e-7 | 7.8201293945e-5 |

Z magnitudes are1.1075648161 for both source and native; the receipt records
source/native RMS and maximum magnitudes for every captured stage. The unchanged
native image plan ends with10+8 rows at image offsets224 and234. Source evaluates
one242-row image graph.

Separate preferred-Q4 interventions keep the exact source input and original
weight bytes fixed. All five matrices—Z, alpha, beta, FFN gate and FFN up—match
all source242 outputs exactly. A16-row intervention at image224 is also exact.
The10-row intervention at224 differs only in local rows8,9; the8-row intervention
at234 differs only in local rows6,7. For Z/alpha/beta, these inputs equal actual
native inputs and both interventions reproduce actual native projection bytes.
For the FFN probes, source postnorm differs from actual native postnorm; those
results establish a source-input route effect, without claiming actual native
projection closure.

This measured pattern agrees with the pinned source's
[preferred-buffer quantization loop](https://github.com/ggml-org/llama.cpp/blob/6c01da6278ab6c69ed67d9124806b50d2b299ba7/ggml/src/ggml-cpu/repack.cpp#L4697):
complete groups of four input rows use matrix quantization and remaining rows
use the single-row converter. Source242 leaves its two-row tail at240; native10+8
moves one tail to232 and processes source tail240 as complete groups instead.
The intervention establishes this local batch-route discrepancy. It does not
establish that a planner correction alone repairs the complete decoded receipt.

The [first report-size admission failure](../tests/results/lighton4b/receipt-layer0-operators/initial-admission-negative.json)
and [second working-directory setup failure](../tests/results/lighton4b/receipt-layer0-operators/source-check-negative.json)
remain separate. Both stop before native build/parameter acquisition/operator
execution. Five hosted small controls pass on the corrected attempt. The
[artifact-only window context](../tests/results/lighton4b/receipt-layer0-operators/artifact-window-context.json)
is separately preserved. Small-report review does not claim independent rehash
of the102 remote raw/binary artifacts.

Next is a separately gated generic planner candidate that keeps nonfinal tiles
aligned with source row groups while retaining the legacy planner exactly.
Source-stage controls and frozen decoded panels remain acceptance gates.
[The failed fixed receipt floor](lighton4b-physical-rope-fixed-audit.md) and
[observed BF16 regressions](lighton4b-physical-rope-observed-audit.md) remain
binding. All experimental gates stay default-off; no release, alias, application
pin or qualification changes.

The [separate aligned candidate controls](../tests/results/lighton4b/receipt-layer0-operators/aligned-source-controls.json)
now pass on runtime `f8bffd1217fa41a958047d88dafb782a4b1cd989`, tools
`c8eec8b4cd117ddfdf4853f52c58022599e24844`, producer receipt
`2d0d4131b12c6392bb92e494644e9f7d4900b17ba6c7389beaa96baae2a91305` and
outer receipt `5e11a778b52cb1b58b600f847c2ac19b41c86a5fb3bfc67e507588f2f013b8ad`.
All22 original native OFF stage/cache arrays and all56 actual source ON stage
comparisons are byte-exact, including all256 hidden0 rows; image-end recurrent
state and FIFO are also exact. The five supported C++ units pass, including
8295 source-planner/cache checks, and all four actual512/1536 CLI missing-model
admissions return4. The exact retained native-document hash is
`046a34d094dff2334e474da7d4a8db9ba98bc8bc437fdc570e1d6d4ca8233d0a`.
The69 producer-owned artifacts total144,904,408bytes and remain remote.
Independent review is small-report/source scoped, with no fresh full raw/binary
rehash claimed. The155.37-second worker time is not model-performance evidence.

The candidate preserves the legacy planner by default. Its separately gated
rule only moves a misaligned nonfinal balancing tile after an eight-row tile;
all4096 frozen legacy plans remain unchanged. Hosted controls verify4096
aligned classifications, actual linear/attention caches and a32-layer decoder
using shared synthetic quantized weights, including mode-toggle recovery and
active/inactive nested cache contamination. These are model-free controls.
Fresh frozen observed/fixed decoded A/B remains pending; component equality
does not waive either earlier decoded negative or provisional-label limits.
