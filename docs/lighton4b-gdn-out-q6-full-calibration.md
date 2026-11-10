# LightOn4B24-output Q6: full calibration result

The fixed24-GDN-output Q6 candidate produces exactly the retained Q8-output IDs,
text and primary EOS on both calibration cases. It uses60,948,480 fewer bytes.
Arabic matches genuine original BF16 exactly; the table still differs from that
reference. This is calibration evidence, with no heldout, default or release
promotion.

[Worker42223984643](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42223984643)
ran source[`30c2332da`](https://github.com/CrispStrobe/CrispEmbed/tree/30c2332da09827723a79f8f55d51c550096edb51)
and completed its worker step in26m39s. All compiled native source hashes are
pinned to first-score source10d3, equivalent to the selectively integrated
PR155 native sources at90d2aeb0fd63adc4b9f41ce88c094b68a3bcedbf. That PR head was
reverified after this run. Both exact named Q6/Q8 policies remain default-off;
old profiles and gates remain preserved. No converter/reference model worker
was added to production artifacts.

The candidate is the exact4,095,839,136-byte file from the
[first-score screen](lighton4b-gdn-out-q6-first-score.md), SHA256
`83e065c472aa12d1c75be997c22d5038350540798f6ff7dbf6ca0240765c1c7a`.
It was extracted from an authenticated retained archive, without reconversion.
The original conversion proves24 matrices quantized directly from original BF16,
699 byte-exact copied payloads, Q8 tied-head identity and independent Q6 decode
hashes. The complete model remains in the original90-day artifact13503268909.

The [full protocol](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-q6-full/protocol.json)
requires700 model-free owner/metadata/grouped-projection controls and two native
cap1 smokes before either full case. Each smoke and full prefill matches all70
retained processor, embedding, vision, input/position, decoder-layer, final-norm
and score hashes from the admitted Q6 first-score run. Both smokes pass. The full
runner uses one teacher row and separate independent greedy generation at cap256,
four threads, HEAD_BF16=1, BF16_QUANT_WEIGHTS=0 and the fixed prior precision gates.
Every returned raw greedy row is finite, has the full248,320 vocabulary and selects
its emitted ID by argmax. Actual last-ID248046, summary, token count and cap are
checked separately. Both cases reach primary EOS without hitting the cap.

| Case / policy | IDs including EOS | CER against genuine BF16 | WER against genuine BF16 | Exact genuine BF16 |
|---|---:|---:|---:|---|
| Table / retained Q4 | 212 | 441/607 =72.6524% | 99/106 =93.3962% | No; first ID differs |
| Table / Q6-output24 | 228 | 224/607 =36.9028% | 61/106 =57.5472% | No; first difference at ID36 |
| Table / retained Q8-output24 | 228 | 224/607 =36.9028% | 61/106 =57.5472% | No; first difference at ID36 |
| Arabic / Q4, Q6-output24, Q8-output24 | 6 each | 0/12 | 0/2 | Yes |

CER uses raw Unicode codepoints; WER uses whitespace-separated words. There is
no normalization or human ground-truth claim. The primary denominator is genuine
original BF16. Separate candidate-against-Q4/Q8 directions remain in the report.
The authenticated retained baseline is source26e6ae663, report SHA256ee94f2a3…;
its original capture/input/model hashes are checked, and its Q4/Q8 first-score
hashes match the admitted same-worker three-arm screen. Full Q4/Q8 outputs were
reused, not rerun or used as generation inputs.

Q6 and Q8 decoded equality is byte-exact for685 table UTF8 bytes and23 Arabic
UTF8 bytes, including equal228/6 token IDs and EOS. Their raw numerical score
hashes differ at all234 shared causal rows. The earlier six strict first-score
failures remain preserved. Thus decoded agreement does not establish numerical
parity. Earlier retained native layer0 errors/magnitudes remain applicable to
these prefills because all70 full-stage hashes exactly match that screen.
No original-BF16 model with this candidate's Q6 weights was forwarded, and no
inner-operator source capture or full F32 model was constructed.

Actual Q6 table generation takes1085.1835s, peak child RSS4,525,862,912 bytes;
Arabic76.6707s, RSS4,534,755,328 bytes. The older Q8 run reports1585.16s/87.66s,
but workers/builds differ, so these are observations, not a controlled speed win.
Preflight records15,798,980,608 bytes effective RAM,114,893,320,192 bytes shared
free disk and3,145,724kB free/total swap. The strict90-minute global and45-minute
child limits were not reached. No worker retry or generation cap waiver occurred.

The [complete report](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-q6-full/gdn-out-q6-full-calibration.json),
[verified index](../tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-q6-full/evidence.json),
raw decoded text, GNUtime records, controls and byte-identical first-score/baseline
receipts are retained in this evidence change. Report SHA256:
`954e9aacd91498ce3ebf5764689e8489d4a189f86e619aee4f0bcdeb722d33fa`.
Complete raw greedy scores and stage arrays remain90days in
[artifact13506428243](https://github.com/CrispStrobe/CrispEmbed/actions/runs/42223984643/artifacts/13506428243),
ZIP266,196,191 bytes, service SHA256
`c0ca2607a067df0856ab366f55a9633998fcec03d74c159416c74d99cfb3261e`,
expiry2027-02-04T13:15:07Z. Only small receipts and decoded text were downloaded.

The table's remaining36.9% character disagreement and57.5% word disagreement
remain material negatives. Arabic's earlier lower original-token probability is
also preserved despite the exact full decode. These two calibration cases do not
qualify heldout images, general OCR quality or a release. No further model job is
part of this result.
