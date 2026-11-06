# External-text importance-weighted Q4 calibration protocol

This is a prepared Kaggle CPU experiment, not a model qualification. It uses the
[verified adapter](lighton4b-imatrix-verification.md) and the exact
[machine protocol](lighton4b-imatrix-q4-calibration-protocol-v1.json). The separate
32-Q6, 24-Q6 and 24-Q8 recipes remain unchanged.

The input is the authenticated original-derived F16 text/F32 vision container,
not an existing quantized model. The worker first regenerates the unweighted Q4
container and requires its complete `f873eb3d…` hash, then requires the frozen
table and Arabic first-score hashes with the current native binary. Either
failure stops the experiment before weighted quantization. This distinguishes
platform/build drift from the candidate effect.

The weighted command uses the ordinary production `crispembed-quantize` path:

```text
crispembed-quantize SOURCE CANDIDATE q4_k --tensor-type 'v.*=f32' --standard-file-type --imatrix ADAPTED
```

The strict adapted profile requires all 248 dense text projection vectors and
rejects skipped or mismatched vectors. The worker independently requires 248
actual `(imatrix)` applications and the completed counter. It audits all 723
tensor names, shapes and types, and hashes all payloads. All 474 F32 tensors and
the tied Q8 embedding/output tensor must remain byte-identical to the null
control. The candidate remains plain Q4 in every other projection; this is not
the publisher's Q4_K_M tensor policy.

Only the two previously observed calibration captures are opened from the
immutable original-output mount. Capture and array hashes, image bytes and
native preprocessing are checked. First scores are measured against genuine
BF16 source scores. If at least 40 minutes remain, independent greedy generation
runs with a 256-token cap and one teacher row. Every generated score's argmax,
EOS/cap and summary are checked. Raw CER/WER use the genuine BF16 decoded string
as denominator; retained plain-Q4/Q8-output strings provide secondary agreement
comparisons. This measures model agreement rather than human OCR accuracy.

The source template is
[tools/kaggle/lighton4b-imatrix-q4/kernel.py](../tools/kaggle/lighton4b-imatrix-q4/kernel.py),
calling the
[bounded controller](../tools/ci-heavy/lighton4b_imatrix_q4_candidate.py).
Before submission, replace its source token with a reviewed commit and configure
the original kernel-output mount in private service metadata. No account names
or credentials belong in source. No GitHub Actions dispatch is used.

The worker requires 18 GiB free disk, 8 GiB available RAM and 2 GiB cgroup
headroom. It builds with two threads and infers with four. Source plus one Q4
output consume about 13.8 GB; the null output is deleted only after its file,
payload and two score controls pass. Quantization expands at most one tensor,
not a complete F32 model. The controller stops at 105 minutes, each child at
45 minutes and bootstrap at 110 minutes, inside a 120-minute service budget.
Known task-owned transient source/build directories are cleaned; candidate,
partial output, raw scores and receipts stay in private kernel output. The
provider does not expose a retention guarantee here, so no 90-day claim is made.

The statistics are published text-calibration numbers, not image/OCR activation
statistics. The repository declares Apache-2.0, but underlying calibration text
content/license and publisher source-weight revision are not authenticated.
Weighted Q4 also chooses a different optimizer branch than unweighted Q4. No
result may be attributed solely to corpus choice, called heldout, promoted to a
default, or published as a release through this experiment.
