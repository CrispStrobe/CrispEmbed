# Q8 restoration calibration sensitivity

[Hosted42185018279](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42185018279)
at`67f47d4353a41b6c8d9deca054c24aac06b783ea` completed all five predeclared
arms on table and Arabic calibration images. FullQ8 greatly reduces first-score
error; mixer restoration fixes the table's two-token prefix, while restoring
FFN-down or allFFN alone does not. Every original first-score comparison still
fails the unchanged strict numerical gates. No arm is quality-qualified.
Heldout receipt and letter were untouched; no full generation/EOS was attempted.

| Arm | Q8 restored tensors | Payload increase overQ4 | Table first-score RMS / max | Table two-token output | Arabic first-score RMS / max |
| --- | --- | --- | --- | --- | --- |
| Q4 baseline | 0 | 0 | 0.706260 / 4.22656 | `<table>` | 0.364631 / 2.95313 |
| all mixers | 152 | 652,083,200 bytes | 0.654575 / 2.89844 | `<figure` | 0.293539 / 2.19531 |
| FFN-down | 32 | 377,487,360 bytes | 0.650900 / 4.35938 | `<table>` | 0.325577 / 1.83594 |
| allFFN | 96 | 1,132,462,080 bytes | 0.632844 / 2.74414 | `<table>` | 0.291611 / 2.12500 |
| fullQ8 decoder | 248 | 1,784,545,280 bytes | 0.0712989 / 0.328125 | `<figure` | 0.0394947 / 0.260742 |

Original table prefix is IDs`[27,17251]`, decoded`<figure`; Q4/FFN-only outputs
are`[20353,29]`, decoded`<table>`. All Arabic prefixes match original
`[67321,20704]`, decoded`أب`. Equality covers only these two tokens. The table
FFN-down maximum error actually worsens despite lower RMS. Lower error or a
correct prefix is not complete OCR quality proof. The fullQ8 improvement is
about90% in RMS on both calibration cases; partial restorations do not isolate
one sufficient low-bit fix. Mixer and FFN effects interact through the decoder.

Each arm uses identical native arithmetic, verified vision gates and final-head
BF16 rounding. Rejected decoder BF16,rotary AVX2,direct quantized dot,weight cache,
imatrix and parallel conversion alternatives remain off. Fresh caches are
created independently per arm. Original BF16 references are retained frozen
first-score arrays, independently rehashed; no Torch/source forward runs here.
Consequently there is no per-arm independent same-weight original oracle claim.

## Actual stage localization

All five prepared-input hashes per case match original. Vision/embedding/head
parameters are protected and shared. Decoder input is constructed once per case
and reused unchanged. All nonbaseline arms first differ at the earliest recorded
block output,`layer0`; these measurements localize the response to the decoder
rather than input processing. They do not locate a first differing operator
inside block0 and do not compare these hidden states to original source stages.

| MixerQ8 versus nativeQ4 | Block0 max / RMS | Block0 Q4 / candidate max magnitude | Block31 max / RMS | Block31 Q4 / candidate max magnitude |
| --- | --- | --- | --- | --- |
| table,220160 values/block | 0.155319 / 0.00558722 | 11.5042 / 11.5422 | 3.69657 / 0.172595 | 76.8053 / 80.5019 |
| Arabic,209920 values/block | 0.110185 / 0.00433838 | 7.33496 / 7.41468 | 5.69520 / 0.231235 | 76.8053 / 80.5019 |

Every value differs in these mixer block0/block31 comparisons. Full32-block
magnitudes,counts,hashes and errors for all arms are in the report. Since only
mixer weights change in this arm, the earliest changed weight operators lie in
block0 GDN projections; that is a dependency inference, not a measured
operator-level mismatch. Subsequent unchanged FFN weights receive changed input.
No claim about an original arithmetic bug follows from restoration-versus-Q4
hidden-state differences.

## Frozen policy and actual donor controls

The[tracked protocol](lighton4b-restoration-sensitivity-v1.json) commits exact
selected names and table/Arabic-only scope before new outputs. Full immutable
GGUF hashes are Q4`f873eb3d15229ec29c59124fc6340c195a19f697a233d3164489a54d26eccd91`
andQ8`5888532207f0d98590cc028cbf92366f8a41ec925568f0c308bd69d9af6342f1`,
revision`a7ec2e54103165a96b8bb6dc3478a2887edd7459` of the retained experimental
source. Per-tensor base/donor hashes are frozen before the first native forward
and checked on actual loaded bytes before inference. Independent review of the
loaded TSV confirms all248 paired hashes. All475 remaining tensors match across
both immutable files, including protected vision,norms and sharedQ8 tied head.
Actual overlays preserve tied-head identity and every unselected pointer.

Model-free controls verify248 unique slots, selected counts0/152/32/96/248 and
invalid layer-count rejection before model acquisition. The binary loads the
ordinary Q4 owner and only the248 Q8 decoder donors, never duplicate protected
vision/head payloads. Test-only borrowed pointer overlays neither change
production GGUF/profile admission nor silently replace the frozen mixed recipe.
Current LightOn owner admitsQ4_K/Q8_0, notQ6_K; upstream selectiveQ6 ideas need
separate owner/protocol work before execution.

Guards require12GiB available RAM,20GiB scratch and2GiB output. Native execution
is bounded35 minutes; the one hosted job is capped50 minutes. Actual combined
native elapsed time is774.826 seconds, including shared setup/hash/vision work.
Per-arm table times are67.097/67.525/68.046/69.108/69.760 seconds; Arabic times
68.077/69.586/68.035/69.664/70.172 seconds in the table's arm order. These include
prefill,head,two-token work and stage writes, not pure kernel benchmarks.

[Full small report](../tests/results/qwen35-ocr/lightonocr3-4b-restoration-sensitivity/report.json),
[frozen payload policy](../tests/results/qwen35-ocr/lightonocr3-4b-restoration-sensitivity/frozen-policy.json),
[actual loaded hashes](../tests/results/qwen35-ocr/lightonocr3-4b-restoration-sensitivity/loaded-payloads.tsv)
and[review](../tests/results/qwen35-ocr/lightonocr3-4b-restoration-sensitivity/review.json)
preserve exact first-score hashes,all stages,source/input controls and limitations.
Large raw first-score/block arrays remain in artifact13487264438,261,400,811 bytes,
SHA256`a2770bf08bf669c3373b52378015c131f0adc56edaab8c53ae534e269ad70780`,
90-day retention. No new PR,runtime release,default change or additional job.
