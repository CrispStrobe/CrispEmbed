# Q8 GDN projection-class calibration screen

[Run42191354208](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42191354208)
at`7650200667371963b084e05d4353d4b687b0656a` completes all eight predeclared
arms on table/Arabic calibration only. GDN output-projection restoration is the
best single projection class for table first-score RMS. QKV is best single class
for Arabic. Restoring a+b alone slightly worsens RMS on both. These are screening
results: every original first-score comparison still fails the unchanged strict
gates, and no complete decoded output/EOS or heldout case was tested.

| Arm | Q8 tensors | Payload increase | Table RMS / max | Table prefix | Arabic RMS / max |
| --- | --- | --- | --- | --- | --- |
| Q4 | 0 | 0 | 0.706260 / 4.22656 | `<table>` | 0.364631 / 2.95313 |
| GDN QKV | 24 | 251,658,240 bytes | 0.668973 / 4.00195 | `<figure` | 0.348032 / 2.57813 |
| GDN Z | 24 | 125,829,120 bytes | 0.698002 / 3.98438 | `<figure` | 0.357053 / 2.59375 |
| GDN a+b | 48 | 1,966,080 bytes | 0.707848 / 4.45703 | `<table>` | 0.364840 / 3.01563 |
| GDN output | 24 | 125,829,120 bytes | 0.616934 / 3.34375 | `<figure` | 0.352079 / 2.35938 |
| allGDN | 120 | 505,282,560 bytes | 0.642260 / 4.03125 | `<figure` | 0.336366 / 2.97656 |
| mixers+FFN-down | 184 | 1,029,570,560 bytes | 0.558216 / 2.87500 | `<figure` | 0.256652 / 2.10938 |
| fullQ8 decoder | 248 | 1,784,545,280 bytes | 0.0712989 / 0.328125 | `<figure` | 0.0394947 / 0.260742 |

Original table two-token prefix is`[27,17251]`, decoded`<figure`; the wrong
prefix is`[20353,29]`, decoded`<table>`. All Arabic prefixes match original
`[67321,20704]`, decoded`أب`. QKV,Z and output each flip the table prefix despite
quite different RMS changes. Thus prefix equality alone cannot rank whole-model
quality. Output-only lowers table RMS about12.65% with125,829,120 extra payload
bytes; allGDN adds four times that payload yet has worse table RMS. Arabic still
benefits more from the combinedGDN group. FullQ8 remains much closer than every
selective arm. No single projection is established as the sole source ofQ4 loss.

## Controls and actual block response

Four restored classes are disjoint and exhaust120 GDN matrices across24 recurrent
layers: QKV24,Z24,a+b48,output24. Each tensor is copied whole; compact packed
QKV rows are neither split nor reordered. Model-free counts/partition controls
run before acquisition. Full-file hashes,all248 loaded Q4/Q8 donor pairs and
all475 protected tensors match frozen identities. Protected vision,norms and
tiedQ8 head/embedding stay unchanged; every unselected pointer is checked. All5
actual prepared-input hashes match original on both cases. Every arm starts
fresh caches under the same fixed native arithmetic and corrected vision/head
gates; rejected alternatives remain disabled.

RepeatedQ4,GDN,mixers+down,fullQ8 controls match the architecture panel in all8
first-score hashes plus256 block hashes:264/264. That is reproducible native
control evidence, not source-model decoded equivalence.

All nonbaseline restorations first change the earliest recorded block output,
block0. Table output-only block0 max/RMS versus nativeQ4 is0.0656118/0.00536461;
Q4/candidate maximum magnitudes are11.5042/11.5456,with all220160 values differing.
At block31 max/RMS reaches3.06524/0.101190. Arabic output-only block0 max/RMS is
0.0677717/0.00417176,maximum magnitudes7.33496/7.36153,all209920 values differing;
block31 reaches5.27399/0.159448. The a+b arm has much smaller block0 RMS
0.000329730(table),0.000309924(Arabic),yet slightly worsens final-score RMS.
All32 block magnitudes/counts/hashes remain in the report. These are comparisons
against nativeQ4 block outputs,not original internal error or measured first
primitive-operator differences.

## Evidence and next bounded candidate

The[exact protocol](lighton4b-gdn-projection-sensitivity-v1.json) was committed
before new outputs. [Full small report](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-projection/report.json),
[frozen payloads](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-projection/frozen-policy.json),
[actual loaded hashes](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-projection/loaded-payloads.tsv)
and[review](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-projection/review.json)
preserve all first-score hashes,prefixes,timings,264 repeated controls and limits.
Original first-score/capture identities are retained by the
[first panel review](../tests/results/qwen35-ocr/lightonocr3-4b-restoration-sensitivity/review.json).
Combined native elapsed time is1292.313 seconds; per-arm time70.972–77.053 seconds
includes prefill,head,two tokens and stage writes,not a pure kernel benchmark.
Large arrays stay in90-day artifact13491248353,417,199,881 bytes,
SHA256`159e0ed86bc1fd9ec799f40e65f43ad2116294f015e15d16349224d47931ef99`.

A separately named exploratory Q4_K+24GDN-output-Q8 policy is a justified next
calibration candidate because it is the smallest single-class payload delta tied
for a corrected table prefix and has the best table RMS. Selection must stay
fixed before full output: all24 `linear_attn.out_proj.weight` tensors,and no
others. It is notQ4_K_M. Full native and genuine original BF16 table/Arabic output
still need measurement; no per-arm same-weight source oracle ran in this screen.
Heldout receipt/letter remain untouched; the original receipt cap/noEOS negative
remains. Current owner supportsQ4_K/Q8_0,notQ6_K. No production admission,default,
model alias,release or quality qualification changes in this evidence commit.
