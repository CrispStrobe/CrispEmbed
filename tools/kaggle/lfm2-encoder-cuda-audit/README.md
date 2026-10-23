# LFM2 Encoder CUDA reference audit

One private Kaggle GPU job validates the existing supported F16 encoder against
the published frozen FP32 suite. It is independent of the M1 UltraBERT audit.
The checkpoint retains its existing LFM1.0 terms; this adds no new model alias.

The uploaded script pins:

- CrispEmbed898af2c2 (explicit CUDA/no-fallback test selection), with its exact
  ggml submodule pointer and unmodified runtime arithmetic.
- Original15-case reference dataset at e3599a94, including649-token input,
  token IDs, all14 layers, final features/CLS and masked logits.
- Official F16 file at bb26e78c with SHA256
  `f097dbdb27c565b6f08cdbbfeb5d674ad7c205cfd1d8362f5415e243b2ec2e93`.
- Standard authentication harness13956033. Only optional download authentication
  is resolved; this job does not publish model files or use cache datasets.

It builds ggml CUDA for the detected device and runs explicit CPU/CUDA probes,
rejects an unknown backend, compares every layer on all15 inputs, then executes
the normal C ABI/Python suite. Each backend must complete297 direct checks,
exact vocabulary/IDs, all15 mask top-1 matches and unchanged minimum cosine0.999/
maximum relative error0.02. API measurements are collected first and strictly
qualified afterward, so numerical failures remain visible and fail the job.
Native GPU defaults are not promoted by this experiment.

Explicit device selection cannot silently fall back. The direct dump invokes
that backend's graph compute, while ordinary API logs must contain successful
scheduler CUDA matmul assignments. This is not a claim that every scheduler
operation stays GPU-resident. No quantized MMQ or latency claim is made.

No Torch import is required. A P100 can validate this F16 ggml path when the
installed toolkit supports sm60. Unsupported toolkit/device combinations exit
inconclusive before model downloads/builds; never repeatedly draw hardware.

Use the account-neutral metadata and standard `tools/kaggle/kpush.py` wrapper
from CrispASR, following the private environment instructions for credentials.
Check active GPU sessions before submission; use one slot. The initial submission
uses `SCRIPT_VERSION=lfm2-encoder-cuda-v1`. Never repush an active job; a justified
rerun needs a new version and recorded uploaded-source hash. Source pushes alone
do not update a Kaggle kernel. Verify pulled script bytes after submission.

Keep models, references and CUDA builds remote. Fetch only
`lfm2-cuda-audit.json`, the small API JSONs and selected failure logs needed for
review. The report preserves all failures, command exits/log hashes and exact
reference inventories. Actual kernel/account/access locators stay in ignored
operator notes. Public evidence must name source/script versions and retain
producer fields, raw report SHA and stated limitations before qualification.
