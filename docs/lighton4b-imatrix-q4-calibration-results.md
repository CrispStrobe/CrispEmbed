# External-text weighted plain-Q4 calibration

The experiment is bounded to the previously observed `simple_table` and
`arabic_printed_line` calibration cases. It is not heldout or human OCR accuracy
qualification. No model hub upload, runtime default or release is involved.

The exact reviewed worker is
[`3ee2b69dd24f24eae4a183b5f9821c8239b77350`](https://github.com/CrispStrobe/CrispEmbed/tree/3ee2b69dd24f24eae4a183b5f9821c8239b77350).
Its [protocol](https://github.com/CrispStrobe/CrispEmbed/blob/3ee2b69dd24f24eae4a183b5f9821c8239b77350/docs/lighton4b-imatrix-q4-calibration-protocol-v1.json)
requires fresh original-derived F16 weights, the exact regenerated unweighted
`f873eb3d…` container, both frozen native first-score hashes, all 248 actual
importance-vector applications and byte-identical 474 F32 tensors plus the tied
Q8 embedding/output tensor. There is no Q8/Q4-to-Q4 requantization or F32 model.

The initial CPU worker, source `2058b11b51fa45ed733df9f5f6c0f2159671b4bf`,
stopped during `venv` creation because Kaggle Python 3.13 could not run
`ensurepip`. It did not build, download a checkpoint, quantize weights or run
inference. Its preflight reported 20,940,541,952 free disk bytes and 31,859,840 KiB
available host RAM, with no swap. The
[raw failure receipt/logs](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4/bootstrap-v1/)
are retained byte-for-byte with SHA256 hashes. A scoped Git attribute preserves
Git's trailing-space hint lines without modifying those raw bytes.

The authorized retry changes only virtual-environment setup: `--without-pip`,
visibility of the installed pip module through system-site-packages, and an
explicit isolated-prefix guard before installing pinned packages. Four offline
admission controls, all eight adapter controls on the real published 3.63 MB file,
syntax checks and the isolated-prefix/pip-import check pass. No GitHub job was
dispatched. Source code, resource caps and candidate controls are fixed before
execution; no fallback or baseline substitution is allowed.

The published statistics are external text calibration. Their repository
declares Apache-2.0, but underlying corpus content/license and publisher
source-weight revision are not independently authenticated. No pretrained image
importance matrix is assumed. The pinned ggml weighted Q4 path also uses a
different optimizer than its null-reference path; a measured change cannot be
attributed solely to corpus choice.

The retry completed conversion and all four first-score measurements, then hit
the predeclared **45-minute table full-generation child limit**. Controller time
was 5,144.935 seconds (85.75 minutes), below its 105-minute global cap. The service
was observed in ERROR state about 93.5 minutes after submission; startup and output
publication time are not separately measured. No limit was extended and no new
worker or model inference was launched afterward.

The candidate is 4,030,958,240bytes, SHA256
`65fcdbd4dfd331e9a48ace0cde086b980f4fb0a61d78555f8b7ef8157948f42f`.
It remains 474 F32 + 248 Q4_K + 1 tied Q8_0 tensors. All 248 Q4 payloads changed;
all 475 protected payloads remain byte-identical to the regenerated unweighted
control. The extra 288 bytes are metadata/header overhead, not higher precision.
The production quantizer's completed counter and individual application lines
both show 248 applications. All 248 actual C++ normalized-mean hashes also match
the adapter's complete source/target/width/count map, including 24 inverse V-head
column permutations. The
[raw receipts and full maps](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4/bootstrap-v2/)
are preserved without editing their bytes.

| Calibration first score | BF16 source | Regenerated plain Q4 | Weighted plain Q4 |
|---|---:|---:|---:|
| Table selected ID |27|20353|27|
| Table probability of source ID |0.693934|0.405578|0.590433|
| Table 27-over-20353 logit margin |0.875|−0.25|0.875|
| Table raw logit RMS error |0|0.706260|0.865078|
| Table maximum absolute error |0|4.226563|4.351563|
| Arabic selected ID |67321|67321|67321|
| Arabic probability of source ID |0.931880|0.871767|0.892794|
| Arabic raw logit RMS error |0|0.364631|0.387335|
| Arabic maximum absolute error |0|2.953125|2.218750|

Raw RMS worsens, but a global logit shift affects that metric without affecting
softmax. Descriptive **posthoc** shift-invariant checks improve: source-to-native
next-token KLD changes from 0.181511 to 0.125953 on the table and from 0.030166
to 0.015504 on Arabic; centered RMS changes from 0.685013 to 0.623312 and
from 0.364349 to 0.360166. These are one-position observations on previously seen
calibration cases, not a selection threshold, clean heldout result or the
publisher's 40-chunk English KLD benchmark. The source/null/weighted score RMS
magnitudes are 1.962688/2.048131/1.748706 for the table and
2.112402/2.089276/2.018764 for Arabic. Component metrics do not establish decoded
quality.

Arabic full generation completes with all six IDs, EOS and text exactly matching
genuine BF16 and the retained plain-Q4/Q8-output results: raw CER 0/12 and WER 0/2.
The table leaves 225 contiguous score rows, positions 0–224. Its last retained row
has argmax 24, not EOS 248046; `generated.ids`, `generated.txt` and a completed native
summary are absent. Therefore table CER/WER, completed EOS and a fixed-table
claim are unavailable. The first token is repaired; the full table remains
unqualified. No decoded improvement over the retained Q4/Q8-output table outputs
is asserted.

The independent read-only consumer authenticates the original v1 bootstrap
`5be2b05e…`, source `42e5ce6f…`, both capture hashes and both source-row hashes.
Pure Python `math.fsum` reproduces all four first-row error/probability results
within 1e−12; all 248,320 native logits in each row are BF16 values expanded to F32.
No reference model runs, F32 model, checkpoint download or additional inference
is involved. Its
[complete diagnostic receipt](../tests/results/qwen35-ocr/lightonocr3-4b-imatrix-q4/bootstrap-v2/independent-consumer.json)
records row hashes, magnitudes, posthoc metrics and the partial-table limit.

Private Kaggle CPU output version 2 contains the
actual candidate and all native outputs. A bounded Range GET independently checks
its `GGUF` magic and total byte count; the full model SHA is producer-computed,
not a second full-model download/rehash. All 734 output file names are retained
in the public inventory without signed URLs or private access references. The
provider exposes no retention expiry, so no 90-day guarantee is asserted.

Two raw status flags need interpretation: bootstrap `quantization_executed` and
`model_forward_executed` remain their initial false values when the overall child
fails, despite successful conversion and completed score/Arabic runs. Likewise
`full_generation_executed=false` denotes the incomplete two-case full phase;
Arabic itself is complete. The raw flags are preserved, not rewritten, and the
per-case receipts/logs establish the actual work. The null-file gate passed, so
its scratch-cleanup diagnostic limitation was not exercised.

The adapter coordinates are proven against audited upstream `6c01da627…`;
the publisher's actual producing converter/quantizer commit is not disclosed.
The Q4_K_M header's `general.name` declares
`B71010F095Dc3735De7Ee6B9969Bff8F8F50B39A`, matching the official b710 reference
revision. That declaration does not independently verify source tensor-content
identity or the producing conversion/calibration environment. This experiment tests the fixed hash-pinned file
under that explicit coordinate assumption. It does not authenticate the original
calibration environment, establish image calibration or qualify a release.
