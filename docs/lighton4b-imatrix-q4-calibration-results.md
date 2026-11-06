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

Execution receipts and numerical/decoded results will be recorded after the
single bounded retry completes. A null-file mismatch would stop the experiment;
this frozen worker would remove that scratch file during cleanup and does not
save its mismatching whole-file digest. That failure-preservation limitation must
remain explicit if the gate fails. The service has no exposed retention expiry,
so no 90-day retention guarantee is claimed for private kernel output.
