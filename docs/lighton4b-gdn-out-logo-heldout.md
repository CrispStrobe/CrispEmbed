# Unchanged GDN-out recipe: heldout A/logo diagnostic

[Run42209944112](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42209944112)
at `0effa01fd5faf0916b3143111d087bff4687afeb` completes both independent native
greedy arms. Both output `A` followed by EOS, with IDs `[32,248046]`; the
unchanged candidate gives **no heldout decoded agreement gain**. Genuine original
BF16 emits a65-token French figure description, starting `<figure>`, ending EOS.
The legacy case name `handwritten_letter` denotes a red capital A/logo glyph;
it is neither a handwritten letter document nor representative paragraph OCR.

| Same heldout input | Plain Q4_K | Q4_K +24 GDN-out Q8 |
|---|---:|---:|
| Output IDs / EOS / cap |2 / yes / no|2 / yes / no|
| Matching initial original IDs |0|0|
| Character edits /216 reference codepoints |215|215|
| Word edits /38 reference words |37|37|
| First-score RMS error |0.6432240321|0.5970310431|
| First-score maximum absolute error |4.1015625|3.46875|
| First-score cosine |0.9308952736|0.9408545798|
| Native elapsed seconds |105.98|105.96|
| Peak RSS bytes |4,523,798,528|4,647,583,744|

CER0.9953703704/WER0.9736842105 measure agreement with the original model's
figure description, not human OCR accuracy: both native outputs correctly contain
the visible glyph but omit that description. Numerical improvement does not
qualify decoded quality. Both first-score strict comparisons remain negative.
Only the first original score row is replayed; all native greedy rows are retained.
The legacy `teacher_forced_ids` field lists all65 reference IDs, but `greedy-full`
consumes one prompt teacher row and no reference continuation IDs.

The [predeclared protocol](lighton4b-gdn-out-logo-heldout-v1.json) freezes the
candidate before output. Source runtime is exactly the calibration engine
`26e6ae6635648a09bdbb9ede19705c3e31271260`, checked through408 file hashes and
ggml gitlink `3a27b6c665b191cb230be367db5c52d81dd3e2ac`. It deliberately excludes
later Q6 and weight-precision changes. Both arms use four threads, the verified
vision gates, HEAD_BF16=1, PARALLEL_DEQUANT=1, BF16_QUANT_WEIGHTS=0 and
BF16_LINEAR=0; the new recipe gate alone differs. Rejected rotary/direct-dot
alternatives remain off. No new source-framework forward runs.

Genuine source `42e5ce6f9bc6bd01865ac4f21d2157c98d3475f1` uses the original
BF16 checkpoint `lightonai/LightOnOCR-3-4B@b71010f095dc3735de7ee6b9969bff8f8f50b39a`.
Its authenticated capture SHA256 is
`e87983eb3c289b5a8b8218e875b863a52d3aa0d1c5d10eeafe10418d83989d48`.
The original65 score metadata entries/EOS/no-cap are admitted before native output;
all five prepared inputs match their frozen hashes. The source first score is
`ca5e79c514b1306d4db3c7b51454f99a5ad7ffcc79e3b52f992ae48eb71103e9`.
Historical original CPU dispatch was not separately captured; this is a frozen
reference diagnostic, not a new cross-platform oracle or same-weight candidate
source replay.

Before heldout inference the hosted worker independently rehashes the entire
previous calibration archive,3,858,198,504 bytes/SHA256
`8072676afb687ee4ed39e17d34f585690c3cbbce567ca8d4caa18f0d112181a1`,
and complete candidate4,156,787,616 bytes/SHA256
`40e54560ed349c077100c4e39276d29c02b98d6bec3611ac7b3a88d5a771c710`.
All723 actual tensor spans match the original frozen donor selection:24 selected
Q8 tensors and699 protected payloads. Census is474 F32/224 Q4_K/25 Q8_0,
including the unchanged tied head. It remains exact named
`lighton4b-q4k-gdn-out-q8-v1`, file_type14, not Q4_K_M. The full recipe/source
metadata and selected manifest are checked, with no new selection or repacking.

All1004 archived F32 arrays are independently streamed and hashed, including452
native greedy-score entries and276 common full/smoke array pairs. Archived raw
text/ID pairs and calibration edit counts are independently verified. Four fresh
calibration smoke controls reproduce132 first-score/block hash pairs and four
prefixes before the heldout arms. Hosted model-free recipe/old-path controls and
48 parallel conversion bit-equivalence cases pass. Resource preflight enforces
cgroup-aware memory and disk bounds; no local build/model execution occurs.

The [report](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-logo/report.json),
[independent raw text/source audit](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-logo/independent-review.json),
[723 payload hashes](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-logo/independent-payload-hashes.json)
and [1004 array hashes](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-logo/independent-array-hashes.json)
preserve the result. A separate reviewer checks the complete original capture
SHA, all65 IDs and decoded text, independently fetches both raw native ID/text
pairs and recomputes edit distances; all agree. Local review checks all723
payload receipt hashes against the predeclared names. Whole model/archive bytes
are audited remotely; no bulk artifact download occurs locally.

New artifact13500455158 retains raw heldout arrays and reports,320,511,641 bytes,
GitHub-declared SHA256
`a80192c1f9a75e2bb4e8e644fa23318193857e40f4cc83276595d92708768371`.
The complete candidate remains in90-day artifact13497191104, without duplication.
No receipt heldout case is executed; its original capped/no-EOS limit remains.
This diagnostic does not select a new recipe, promote a default/model alias,
change an app pin or constitute a release. Calibration table gain remains real
but calibration-selected; this first distinct heldout case has no decoded gain.
