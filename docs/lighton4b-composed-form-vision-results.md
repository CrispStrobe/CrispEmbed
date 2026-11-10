# Observed composed form vision correction

A complete **pixels-only native** form diagnostic improves the observed output.
Source `b98567cfd46aaf594f33076aaab4ff8e4610edbb` changes only
`CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE` and
`CRISPEMBED_LIGHTON4B_PHYSICAL_F16_VISION` between fresh child processes.
Q4 repacking and BF16 head rounding stay0; the same physical Q4_K_M weights,
decoder, F16 attention, RGB/prompt/positions, context2048 and cap512 are fixed.
Neither arm injects captured vision features.

[Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-composed-form-vision/v2/composed-form-vision.json)
SHA256 is `aec0d0b10befb5b3498b55a87f9cda785998242c0d9f49a4a1b3f72a92ed160b`,
worker778.920s. Component receipt039b and all73 owned files were rehashed
before model acquisition; all28 actual source stage arrays were authenticated.
Candidate SHA256 remains `843e66c31e5516cd6597cf20de544906f75ba024bbcbbab3f2c9085a61e79d43`.

| Measure | Old composed | Corrected composed |
|---|---:|---:|
| IDs / completion |194 / EOS|167 / EOS|
| Character edits vs BF16 output reference (656 codepoints) |259|175|
| Word edits vs BF16 output reference (90 words) |47|45|
| Character / word edits vs actual upstream output |261 /45|80 /18|
| First-score RMS vs upstream |0.192548|0.105889|
| First-score maximum absolute error |1.001385|0.558670|
| Child wall seconds |285.420|139.885|
| Observed peak RSS bytes |4,326,182,912|4,326,293,504|
| Q4 repack / converted F32 weight-cache bytes |0 /0|0 /0|

Old mode reproduces all194 frozen raw greedy score rows,38 text stages and29
vision/processor anchors before the corrected arm starts. Corrected mode matches
all167 retained source-feature score rows,38 text stages, complete IDs/raw text
and EOS bytewise; all28 actual source vision stages and processor patches match.
Every greedy row is finite and argmax-checked, with actual prefill/teacher0/greedy0
row identity. The corrected composed output therefore recovers the existing
source-feature result without supplying those features to native inference.

The independently pinned upstream Q4_K_M output has189/656 character edits and
46/90 word edits against the same177-ID BF16 output reference. Corrected mode
meets that observed form floor, but does not reproduce the actual upstream
transcription or strict first-score numerics. BF16 output is a reference, not
human ground truth. This worker uses one teacher row plus full independent
greedy output, not a177-position original-prefix score panel.

[Independent review](../tests/results/qwen35-ocr/lightonocr3-4b-composed-form-vision/v2/independent-review.json)
rehashes all six downloaded text/ID/summary files and recomputes raw character
and word edits against both authenticated references. Full arrays remain
remote under source-bound hash/finite/argmax/anchor receipts. RSS is sampled
VmHWM at up-to-1s intervals and can miss short peaks. Wall time includes model
loading, prefill, generation and writes with different output lengths; no
isolated whole-model speedup is claimed.

The initial submission used a stale requested dependency alias. The API omitted
that mount; mandatory receipt admission stopped in24.180s before worker/model
acquisition or forward. [Raw setup negative](../tests/results/qwen35-ocr/lightonocr3-4b-composed-form-vision/v1-dependency-negative/bootstrap-receipt.json)
is preserved separately. V2 corrects the canonical binding only, leaving source
and bootstrap arithmetic unchanged.

**No shipping or default promotion.** The corrected table/Arabic complete
composed comparisons are next. The existing table floor and representative
held-out gates remain unresolved; Q4 repacking remains a documented regression
and must stay off. Global vendor traits and application release pins are unchanged.
