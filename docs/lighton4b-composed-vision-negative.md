# Composed vision is worse on the observed form

The paired observed-form run completed at source `8be08d1ff` in Kaggle CPU service `lighton4b-composed-vision-observed-form-v1`, version1. [Raw receipt](../tests/results/qwen35-ocr/lightonocr3-4b-composed-form/composed-vision.json) SHA256 `1458d07e6098b4b1b8fa078c18fe0b636f832a0d1219f789b035e1a033ef4ba0`; [independent review](../tests/results/qwen35-ocr/lightonocr3-4b-composed-form/independent-review.json). This is a negative development diagnostic, not held-out or shipping qualification.

| Arm | IDs / completion | Character edits vs original reference | Word edits vs original reference | First-score RMS / max vs actual upstream |
| --- | --- | --- | --- | --- |
| Actual upstream Q4_K_M | Retained complete output | 189/656 | 46/90 | Reference |
| Native, upstream merged features | 167 / EOS | 175/656 | 45/90 | .105889 / .558670 |
| Native, composed vision | 194 / EOS | 259/656 | 47/90 | .192548 / 1.001385 |

Original BF16 text is a reference output, not human ground truth. Direct composed-vs-upstream output distance is261/647 characters and45/86 words; the corresponding source-fed distances are80/647 and18/86. Both native arms completed independently within fixed512 caps. Composed vision fails both predeclared reference-distance floors against upstream and remains no-ship.

Both native arms use the same physical text weights, executable/libraries and frozen decoder gates. The source-fed arm exactly reproduced every retained greedy row and all38 prefill-stage anchors before composed execution. Paired native preprocessing patches, text embeddings and positions are byte-identical. The earliest available comparable difference is merged vision: RMS .0112370, max1.681473 over358,400 values (upstream maximum magnitude5.431318). The actual source feature bytes are authenticated. The subsequent prompt-splice RMS is .0107141/max1.681473; all32 text-layer difference metrics are preserved in the receipt.

Actual upstream preprocessing and visual block intermediates are unavailable here. Canonical publisher F16 projector versus protected native vision weight/precision equivalence is unproven. This result therefore compares deployed projectors on identical RGB; it does not isolate a kernel or prove a weight-conversion defect. Next work must authenticate canonical weights/precision transforms and compare the earliest source-observable vision stage before any gated correction. Source-feature success cannot rescue composed failure.

Native arm times were129.57s source-fed and282.11s composed; whole worker743.04s includes transfers, build and lossless packing. Different generated lengths make this a diagnostic timing comparison. All raw vectors and full greedy rows remain remote. Independent local review downloaded only needed vectors (18.05MB plus1.34MB source-score/log subset), verified12 vector hashes, six paired stage metrics and raw text/IDs/EOS, and independently recalculated character/word edits. First-score FP64 RMS reductions differed by2 ULP/source-fed and1 ULP/composed; maxima and hashes are exact, and no runtime tolerance was changed. Experimental gates and public admission remain unchanged.
