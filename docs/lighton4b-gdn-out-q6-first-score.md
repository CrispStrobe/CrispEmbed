# LightOn4B: fixed24 GDN-output Q6 calibration screen

The fixed24-output Q6 candidate restores the original first token on the table
calibration case. This is a one-token screen; it does not show that the complete
table, EOS or heldout output is fixed. Arabic keeps its first token but its
probability under the candidate decreases. No runtime default, registry or release
changes follow from this result.

[Worker42219710049](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42219710049)
ran source[`10d3fad88`](https://github.com/CrispStrobe/CrispEmbed/tree/10d3fad88f59c787dee855a564bfd8fed57962d0),
after [model-free42218056668](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42218056668)
passed700 new Q6-output checks,713 preserved32-Q6 checks,112 frozenQ8-output
checks and existing native/mixed/BF16 controls. The separate32-Q6 policy and its
[negative table screen](lighton4b-q6-candidate-screen.md) remain preserved.
No initial failure or retry occurred in this24-output lane.

The [exact recipe](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-q6-first-score/recipe.json)
selects only `l.layers.N.linear_attn.out_proj.weight` for `N%4!=3`:
24 matrices quantized directly from original BF16 checkpoint6b0d/b710, never Q8
or Q4 donors. All699 remaining payloads are byte-exact copies from plainQ4f873,
including474 protected F32 tensors and the Q8 tied embedding/output. All723
source/payload mappings and all24 independent native/Python Q6-to-BF16 hashes
pass in the [conversion receipt](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-q6-first-score/conversion.json).
The custom marker is `lighton4b-q4k-gdn-out-q6-v1`; native admission requires
both default-off Q6 capability and GDN-output-Q6 gates with exact typed metadata.
This is a custom24-output policy, separate from the upstream-style32-Q6 recipe.

The same worker/binary ran plainQ4, authenticated retainedQ8-output and fresh
Q6-output, in that order for table and Arabic only. Each retained Q4/Q8 first-score
hash had to match before accepting the Q6 measurement. HEAD_BF16 and parallel
dequant were1; BF16 quant-weight and BF16 linear gates were0, with verified vision
gates fixed. The [protocol](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-q6-first-score/protocol.json)
pins the original captures, image/input manifests and all model hashes. No
original batch JSON or heldout member was decompressed/read; the retained Q8 ZIP
was authenticated as opaque transport and only its model member was extracted.
No reference model forward or full F32 model was instantiated.

All errors below use the frozen genuine original-BF16 first-score row as the
denominator. Probability is assigned to its original winning ID; it is not a
calibrated OCR confidence or human accuracy measure.

| Case / policy | First decoded token | Score RMS | Maximum error | Original-ID probability |
|---|---|---:|---:|---:|
| Table / Q4 | `<table` (20353) | .706260330 | 4.2265625 | .405577880 |
| Table / Q6-output24 | `<` (27) | .604549443 | 3.453125 | .544191861 |
| Table / Q8-output24 | `<` (27) | .616933547 | 3.34375 | .514095096 |
| Arabic / Q4 | أ (67321) | .364631240 | 2.953125 | .871767024 |
| Arabic / Q6-output24 | أ (67321) | .359154916 | 2.5625 | .840038611 |
| Arabic / Q8-output24 | أ (67321) | .352079051 | 2.359375 | .847671905 |

All six actual `crispembed-diff` strict score comparisons fail; equal/changed
controls pass. All six independent generations stop at the configured one-token
cap without EOS. Actual teacher0, prefill and greedy0 score hashes agree within
each arm. The [complete worker report](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-q6-first-score/report.json)
and [evidence index](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-q6-first-score/evidence.json)
preserve every score/stage hash, gate, negative and independent decoded token.

Processor arrays, all24 retained vision-layer outputs, text embedding, prefill
input and positions are byte-exact across all three arms. The first different
retained native stage is decoder layer0, consistent with the changed layer0
output weights. Q4 versus Q6-output layer0 RMS/max error is .005484310/.082976580
for table (absolute magnitudes11.504187/11.538329) and .004257535/.082976580 for
Arabic (magnitudes7.334964/7.357350). These compare actual native policy arms;
no same-weight original-BF16 decoder-stage capture was executed for this candidate,
and the retained layer output does not isolate the first differing inner operator.

The candidate is4,095,839,136 bytes, SHA256
`83e065c472aa12d1c75be997c22d5038350540798f6ff7dbf6ca0240765c1c7a`.
It adds64,881,184 bytes including metadata versus plainQ4, and is60,948,480 bytes
smaller than the retained Q8-output candidate. Candidate and complete raw native
arrays remain in [artifact13503268909](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42219710049/artifacts/13503268909),
ZIP3,529,294,747 bytes, service SHA256
`ede38c5efde5a874daf8f5157d86b2bed033764f8518275d735cbc674266e719`,
expiry2027-02-04T12:42:23Z. Worker/conversion receipt SHA256 values are respectively
`48ae6e7a5f58f401141a69312b0a3c210096a7400d4738b516393518241723ee`
and `01391eb13e0c82017b4773f3c6b457512bcdc85e9db57e1b9f8eda8e64d00d3e`.
Only small receipts, decoded tokens and selected layer0 arrays were downloaded
for review; the large model remains remote. Preflight measured15,772,647,424 bytes
available RAM and114,893,475,840 bytes shared free disk. Peak RSS was not measured.

Next is separately bounded full independent generation on these two calibration
cases with this exact candidate, compared with frozen original/Q4/Q8-output text.
No heldout selection, full decoded-quality, EOS or release claim is admitted here.
