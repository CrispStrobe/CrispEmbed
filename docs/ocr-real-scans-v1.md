# Twenty real document scans: initial evidence and remaining parity work

The [frozen manifest](ocr-real-scans-v1.json) contains fifteen public-domain
scan pages and five CORD-v2 receipts, with source/revision links, attributions,
encoded-image hashes, ground truth and fixed output caps. This is an owned
comparison set, not OCRBench or OmniDocBench. Wikisource transcriptions retain
their CC BY-SA 4.0 declaration and contributor-history links under
[Wikimedia's content terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content).
CORD images/annotations retain CC BY 4.0, as declared in the
[pinned author dataset card](https://huggingface.co/datasets/naver-clova-ix/cord-v2/blob/7f0115a4b758a71d6473b8d085751692da2fef98/README.md)
and [author repository](https://github.com/clovaai/cord). These data are not
relicensed under the code or model license. The manifest records the original
public-domain scan sources and transcription modifications individually.

The unchanged producer reports and exact source/version/report identities are
under [tests/results/ocr-real-scans](../tests/results/ocr-real-scans).
Original MiniCPM completed twenty outputs at source `2acee0b1`; original
Unlimited completed twenty at `4daea203`. Their manifest bytes match; the source
difference adds the scorer's pure-Python fallback. Both exact model revisions
and framework versions are retained in the reports. The initial native Unlimited
run uses `2acee0b1`, before the separately verified raw-decoder repair.

**One page is ineligible:** `en-form-economic-journal-67` was served with encoded
SHA256 `b09f8c1c…`, instead of frozen `4d6e0d71…`, in MiniCPM's original and
Unlimited's native run. The initial harness recorded the rejection but still ran
inference. Its results remain preserved and visible; they do not enter valid
pair denominators. A matching token sequence cannot repair a failed input proof.
Future workers now reject changed encoded image bytes or Parquet hashes before
admission, and retain admitted encoded images remotely for exact replay.

For the nineteen valid initial Unlimited pairs, eleven token-ID sequences match
exactly and zero raw-text strings match. The other eight cases retain the first
differing token, both lengths, caps and timings in the
[paired review](../tests/results/ocr-real-scans/unlimited-initial-review.json).
No earliest operator defect is established by those token differences. The raw
text failures precede the four-case decoder repair and require replay through
that actual repaired decoder; a synthetic-panel pass does not establish this
real-scan result. Normalized text/CER/WER remain distinct from raw IDs/text.

Both original and native Unlimited hit the fixed 1,024-token caps on receipt
rows 29 and 4. CORD ground truth is partial; word recall is its primary diagnostic,
and large CER alone cannot establish a transcription failure. Do not change
caps, prompts, gold or the scorer to improve these historical results. Timing
records include different process/load scopes and hosts; no speed ratio is
qualified from them.

Next: finish collecting MiniCPM's current native report; authenticate all
remaining input pairs; replay the repaired Unlimited decoder on retained IDs;
then localize meaningful remaining ID differences with original/native
preprocessing, vision, projector, prefill and token/logit intermediates and
crispembed-diff. Keep coordinates, source-model completion limits and body-text
regressions separate. Reconstruct the missing image input as a separately
versioned capture if exact frozen bytes cannot be recovered. These results do
not qualify the full real-scan panel, any Q4 model, a release or a default change.
