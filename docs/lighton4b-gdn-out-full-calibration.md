# Full calibration of the24 GDN output Q8 candidate

[Run42199032449](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42199032449)
at`26e6ae6635648a09bdbb9ede19705c3e31271260` completes four actual independent
greedy runs: originalQ4 and fixed`lighton4b-q4k-gdn-out-q8-v1`,each on table and
Arabic calibration. The candidate materially improves agreement on table while
preserving exact Arabic output. Table remains substantially different from genuine
BF16 output; these calibration results do not qualify the model or a default.

| Case/arm | Generated tokens / original | EOS / cap | Matching initial IDs | Character edits / reference | Word edits / reference |
| --- | --- | --- | --- | --- | --- |
| table Q4 | 212 / 222 | yes / no | 0 | 441/607 =72.65% | 99/106 =93.40% |
| table candidate | 228 / 222 | yes / no | 36 | 224/607 =36.90% | 61/106 =57.55% |
| Arabic Q4 | 6 / 6 | yes / no | all6 | 0/12 | 0/2 |
| Arabic candidate | 6 / 6 | yes / no | all6 | 0/12 | 0/2 |

Character edits decrease217(49.2%) on table; word edits decrease38. Q4 starts
`<table>` and the candidate restores the source's`<figure>` prefix and caption
style. Both Arabic outputs exactly read`أبجدية عربية`,including all IDs and EOS.
This measures agreement with the retained model output,not human OCR accuracy.
Case,punctuation and whitespace are retained; CER uses Unicode codepoints and
WER uses Unicode whitespace words. No normalization or budget relaxation occurs.
All first-score strict numerical gates still fail (table RMS0.706260→0.616934,
Arabic0.364631→0.352079). A restored prefix alone was insufficient; complete
output now establishes this limited improvement and its remaining error.

## Verified file and actual pipeline

The candidate is4,156,787,616 bytes,SHA256
`40e54560ed349c077100c4e39276d29c02b98d6bec3611ac7b3a88d5a771c710`.
It copies exactly24 complete GDN output matrices from the pinned Q8 file,retains
699 other Q4-source payloads,with125,829,120 additional payload bytes and544
additional header bytes. All723 persisted tensor payloads are rehashed in the
producer,and an independent review of all723 audit records matches frozen source
hashes. Tensor census474F32/224Q4_K/25Q8_0 includes the protected tiedQ8 head.
File type remains14 (Q4_K_S base plus exact custom recipe tag),neverQ4_K_M15.
The[predeclared recipe](lighton4b-q4k-gdn-out-q8-v1.json) binds its names/source
hashes/manifest and calibration selection before complete output.

Before model acquisition,112 recipe/type/gate/metadata controls and48 bit-exact
parallel conversion cases pass. Actual candidate metadata,default-off context,
conflicting recipe gates and wrong base tag controls pass. Every serialized-file
smoke reproduces the earlier projection screen's32 block hashes,first-score hash
and two-token prefix:132 SHA pairs and4 prefix controls before complete outputs.
The[fixed smoke controls](lighton4b-gdn-out-smoke-controls-v1.json) are test-only.
Native RGB/pixels,prompt,grid and modalities match official frozen inputs; the
un-padded all-ones attention-mask contract is checked before payload execution.
No reference activation or teacher continuation ID enters greedy generation.

Both arms fix verified vision/headBF16 corrections and parallel dequantization1;
BF16_QUANT_WEIGHTS0,BF16_LINEAR0,directdot0,cache0 and rejected rotary0 avoid
confounding the separately tested precision diagnostic. The new recipe gate is1
only for the candidate and0 for Q4. New`greedy-full` runs up to256 independent
IDs and stores every greedy score. Its legacy`teacher_forced_ids` report field
lists the reference sequence,not consumed continuation input; actual teacher
positions are1. The old full teacher replay mode is preserved.

## Runtime and independent review

| Case/arm | Native wall seconds | Peak RSS bytes |
| --- | --- | --- |
| table Q4 | 1348.85 | 4,470,517,760 |
| table candidate | 1585.16 | 4,515,020,800 |
| Arabic Q4 | 81.81 | 4,469,534,720 |
| Arabic candidate | 87.66 | 4,514,562,048 |

These are same-job total model/vision/decoder/output timings. Table output lengths
differ,so they do not establish a kernel speed comparison. Cgroup/RAM/disk and
process-group/time bounds all pass; no arm times out or hits its generation cap.

Independent range retrieval verifies the actual four raw ID/text pairs against
reported values and authenticated original capture hashes. A separate full-matrix
Levenshtein computation reproduces every character/word edit count. The
[review](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-full/review.json),
[report](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-full/report.json),
[723-record candidate audit](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-full/candidate-audit.json)
and[raw text audit](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-full/independent-text-audit.json)
retain sources,binaries,timings,metrics and exact raw outputs. Native/reference
text is preserved beside these receipts. Whole archive and candidate rehash will
also run remotely as a prerequisite to the heldout diagnostic; no bulk model was
downloaded onto the shared workspace for this review.

Candidate GGUF and all arrays remain in90-day artifact13497191104,
3,858,198,504 bytes,GitHub-declared SHA256
`8072676afb687ee4ed39e17d34f585690c3cbbce567ca8d4caa18f0d112181a1`.
No heldout input runs here. The source receipt remains capped/noEOS; historical
four-case CPU dispatch was not separately recorded. Next is the predeclared
heldout legacy`handwritten_letter` case,actually a red capital A/logo with a
complete65-token BF16 reference,not handwriting/paragraph OCR. Candidate bytes
and selection must stay unchanged. No quality/default/release/alias promotion.
