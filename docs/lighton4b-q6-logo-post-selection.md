# Fixed Q6 candidate: observed-logo diagnostic remains negative

The frozen24-output Q6 candidate still returns `A`, IDs32/248046, on this logo.
It matches the earlier Q4/Q8-output IDs, text and primary EOS. Against the frozen
65-ID genuine-BF16 caption it differs from the first ID: CER215/216=.995370,
WER37/38=.973684. These are raw model-agreement errors, not human OCR accuracy.
The fixture is historically namedhandwritten_letter but contains a stylized logo;
it is not representative handwritten-letter OCR.

[Run42230729575](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42230729575)
uses source[`c7e6720ab`](https://github.com/CrispStrobe/CrispEmbed/tree/c7e6720ab8fa5b1deb8db4ff81ca534a95a17a6c).
This is explicitly a post-selection diagnostic on an already observed case,
not clean heldout evaluation. Candidate83e065c4… and the24-output policy remain
unchanged from the completed [calibration result](lighton4b-gdn-out-q6-full-calibration.md).
No weight reselection, default, registry or release promotion follows.

The [closed protocol](../tests/results/qwen35-ocr/lightonocr3-4b-q6-logo-post-selection/protocol.json)
authenticates the original e87983eb… capture and prior e00b5c66… Q4/Q8 receipt,
plus the complete prior logo archive.700 model-free controls and the70-hash table
anchor pass before the new case. Actual154-token/560-patch prepared inputs match
original BF16. All36 prior protected predecoder hashes match, including processor,
vision, embedding and prefill inputs/positions. Every returned greedy score row
is finite/full-vocabulary and selects its emitted ID by argmax; independent
primary EOS248046 and cap256 are checked. The run reaches EOS after2 IDs, without
hitting a cap or timeout. No reference model forward or full F32 model is run.

| Policy | First-score RMS against BF16 | Maximum error | Original-ID27 probability | Full decoded output |
|---|---:|---:|---:|---|
| Retained Q4 | .643224032 | 4.1015625 | .053843877 | `A` |
| Fixed Q6-output24 | .593816195 | 3.51953125 | .091131878 | `A` |
| Retained Q8-output24 | .597031043 | 3.46875 | .089890432 | `A` |

The small numerical change does not repair the decoded disagreement. The first
retained native layer after the exact predecoder boundary differs between Q6
and Q8: layer0 RMS.001250457, maximum.018257380, absolute magnitudes9.805857 and
9.808157. This compares actual source-authenticated native policy arms; it does
not isolate all remaining runtime versus quantization error. There is no genuine
original-BF16 oracle with this candidate's Q6 weights in this diagnostic.

Native logo execution takes163.1102s, peak child RSS4,584,783,872 bytes. Older Q4/Q8
workers took about106s each; these are different workers/builds, so no controlled
speed conclusion is drawn. The fixed60-minute global/45-minute child limit was
not reached. Scope remains one table anchor and one observed-logo diagnostic.

The [complete report](../tests/results/qwen35-ocr/lightonocr3-4b-q6-logo-post-selection/gdn-out-q6-logo-post-selection.json),
[verified evidence index](../tests/results/qwen35-ocr/lightonocr3-4b-q6-logo-post-selection/evidence.json),
byte-identical prior receipts, source review, controls, raw decoded text and
GNUtime records are preserved. CER/WER were independently recomputed from the
retained source IDs/text; decoded bytes and hashes match the report. Report SHA256:
`973fe43f678d315cd6617ab70ef18fc2f04681f5c4e68361d95368d644790f33`.
Raw native stages and greedy scores remain90days in
[artifact13510375701](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42230729575/artifacts/13510375701),
ZIP118,857,060 bytes, service SHA256
`f662983cc9e4b6090b605dfc8d7d23bbc7f054c24cb9d8e9c784480ef21d4035`,
expiry2027-02-04T14:04:49Z. The4.1GB candidate stays in its existing retained
artifact; only small reports and decoded text were downloaded for this review.

The positive two-case calibration remains intact, and this known-case negative
remains explicit. Neither result establishes general OCR quality or a release.
