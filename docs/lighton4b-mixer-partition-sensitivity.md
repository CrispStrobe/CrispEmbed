# Q8 mixer architecture partition sensitivity

[Run42187953019](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42187953019)
at`d874f0c85672340c09cee6b8adf37e0a104c73f9` completes the six predeclared
calibration arms. On table, linear/GDN restoration fixes the observed two-token
prefix; full-attention restoration alone does not. Arabic benefits from both
groups in first-score RMS. This narrows a table-prefix contributor, not all Q4
loss: effects are distributed and nonmonotonic. All12 strict original first-score
gates still fail. No full generation/EOS or heldout test was run.

| Arm | Q8 tensors | Payload increase | Table RMS / max | Table prefix | Arabic RMS / max |
| --- | --- | --- | --- | --- | --- |
| Q4 | 0 | 0 | 0.706260 / 4.22656 | `<table>` | 0.364631 / 2.95313 |
| linear/GDN | 120 | 505,282,560 bytes | 0.642260 / 4.03125 | `<figure` | 0.336366 / 2.97656 |
| full attention | 32 | 146,800,640 bytes | 0.664900 / 3.52734 | `<table>` | 0.327429 / 3.53125 |
| all mixers | 152 | 652,083,200 bytes | 0.654575 / 2.89844 | `<figure` | 0.293539 / 2.19531 |
| mixers+FFN-down | 184 | 1,029,570,560 bytes | 0.558216 / 2.87500 | `<figure` | 0.256652 / 2.10938 |
| fullQ8 decoder | 248 | 1,784,545,280 bytes | 0.0712989 / 0.328125 | `<figure` | 0.0394947 / 0.260742 |

Original table prefix is`[27,17251]`, decoded`<figure`; wrong table prefix is
`[20353,29]`, decoded`<table>`. All Arabic arms match`[67321,20704]`, decoded`أب`.
FullQ8 remains substantially closer than every partial restoration. Adding full
attention to GDN slightly worsens table RMS while lowering maximum error; maximum
Arabic error worsens for either individual subgroup despite lower RMS. The
higher-precision mixers+down control improves both RMS values, but is not qualified
quality. Two tokens do not establish complete OCR agreement.

## Actual localization and repeated controls

Full-attention-only preserves every nativeQ4 block0–2 output byte and first
changes block3. GDN first changes block0. Table GDN block0 max/RMS difference
versus nativeQ4 is0.155319/0.00558722, with Q4/candidate maximum magnitudes
11.5042/11.5422; block31 reaches4.75927/0.133194. Table full-attention block3
max/RMS0.102844/0.00527823 has magnitudes12.9944/12.9651; block31 reaches
2.20835/0.111534. Arabic GDN block0max/RMS0.110185/0.00433838; attention
block3max/RMS0.0545132/0.00394290. These are recorded block-output changes versus
nativeQ4, not original internal error or first-operator attribution.

Across this and the[first panel](lighton4b-restoration-sensitivity.md), repeated
Q4/all-mixers/fullQ8 controls match all6 full first-score hashes and192 layer
hashes exactly:198/198. Both cases retain all5 official prepared-input hashes.
Model-free controls verify disjoint120/32 architecture groups, their exact152
union and the separate32 FFN-down addition before acquisition. All248 loaded
base/donor hash pairs match the committed previous donor freeze; all475 protected
tensors match, including vision,norms and shared tiedQ8 embedding/head. Fresh
caches and fixed precision/gates isolate restoration groups within the process.

## The120 GDN matrices and next bounded question

There are24 recurrent layers and five rank2 projections per layer:

| Projection class | Count | HF matrix shape (rows,columns) | Q8 payload increase overQ4 |
| --- | --- | --- | --- |
| in_proj_qkv | 24 | 8192,2560 | 251,658,240 bytes |
| in_proj_z | 24 | 4096,2560 | 125,829,120 bytes |
| in_proj_a and in_proj_b | 48 | 32,2560 each | 1,966,080 bytes |
| out_proj | 24 | 2560,4096 | 125,829,120 bytes |

QKV is the original compact Q2048/K2048/V4096 layout; no within-tensor row slicing
or head reordering occurs. Protected F32 convolution,decay,norm parameters remain
unchanged. The next proposed screen partitions these four disjoint projection
classes, alongside Q4,GDN120,mixers+down184 and fullQ8248 controls. Exact names
must be frozen before outputs; calibration table/Arabic only, heldout untouched.
This document does not approve a new production recipe or infer a winning
projection before that experiment.

## Reproduction and preserved evidence

The[predeclared protocol](lighton4b-mixer-partition-sensitivity-v1.json) and
[report](../tests/results/qwen35-ocr/lightonocr3-4b-mixer-partition/report.json)
retain exact group names,first-score/prefix/block metrics and timings. The
[frozen payload policy](../tests/results/qwen35-ocr/lightonocr3-4b-mixer-partition/frozen-policy.json),
[actual loaded hashes](../tests/results/qwen35-ocr/lightonocr3-4b-mixer-partition/loaded-payloads.tsv)
and[review](../tests/results/qwen35-ocr/lightonocr3-4b-mixer-partition/review.json)
bind independent digest/partition checks and the198 repeated control hashes.
Actual combined native elapsed time is1004.084 seconds; individual arms run
73.658–79.945 seconds including prefill,head,two tokens and stage writes, not
kernel-only benchmarks. Original first-score/capture identities remain in the
[first panel review](../tests/results/qwen35-ocr/lightonocr3-4b-restoration-sensitivity/review.json).
Large arrays remain remote in90-day artifact13490015744,313,344,622 bytes,
SHA256`31bc48d3bbecc09871d58ae7a6a2adc46ec1f12763b2a87eef89ca62c888d963`.

No per-arm same-weight source oracle ran. Current owner supports Q4_K/Q8_0,notQ6_K.
Test-only borrowed pointer overlays do not change production profile admission,
frozen model recipes,defaults or release. Receipt/letter heldout cases and the
original receipt's capped/noEOS negative remain untouched.
