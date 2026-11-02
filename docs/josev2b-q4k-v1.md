# Protected Josev packed Q4_K experiment

This is source preparation for one hosted CPU experiment. It is default-off,
with model inference unexecuted and unqualified. The original v5 reference, all F32 loaders, public
bindings and existing source guards stay unchanged. No release, app pin, model
publication or task-quality result follows from a converted file.

The immutable input is
[`cstr/Josev-2B-SystemOne-GGUF@bffe6252`](https://huggingface.co/cstr/Josev-2B-SystemOne-GGUF/tree/bffe62522f826b4c33f2e731d5b870ced5150a17),
`josev2b-systemone-f32.gguf`, 7,659,180,448 bytes,
SHA256 `72684e39d0eed526b40c5150c78b2d881d324c39e16af48060b3d741f526ccf2`.
Its 442 tensors have 1,912,297,796 semantic parameters. The recipe selects 150
backbone attention/MLP/linear-attention projection matrices for actual Q4_K.
Every selected column width is already 2048 or 6144 and divisible by 256;
padding, row fusion and legacy Q4_0 fallback are forbidden. The expected tensor
payload is 2,934,531,344 bytes before GGUF metadata/alignment.

All 122 `head.*` tensors, the tied token/lexical embedding, norms, convolutions,
recurrent `A_log`/`dt_bias`, and linear-attention `in_proj_a`/`in_proj_b` remain
F32. These comprise the 292 excluded tensors. The independent
[packer](../tools/quantize_josev2b_q4k.py) authenticates the input and schema,
then invokes the unchanged C++ quantizer with explicit F32 exclusions. Readback
checks every name, semantic shape and actual type; it hashes all tensor bodies,
requires every critical F32 byte to match the source and validates finite Q4_K
scales. Tokenizer/provenance metadata must remain byte-identical except for the
quantizer's declared file type/version. Only after those checks does it change
the same-length format value `crispembed-josev-packed-f32-v1` to
`crispembed-josev-packed-q4k-v1` in place. The final header/extents and untouched
metadata are rechecked. Unit axes may have a shorter serialized GGUF rank from
the upstream quantizer; the four semantic GGML axes remain exact and both ranks
are reported.

The existing runtime mixed-profile guards require both
`CRISPEMBED_JOSEV_QUANT=1` and `CRISPEMBED_QWEN35_QUANT=1`, in addition to the
seven existing packed/SystemOne component gates. Unset, empty and zero values
reject. Old F32 mapping and old metadata-only tokenizer loading remain strict.
This preparation changes no inference C++ source or public API.

The new [standalone CMake harness](../tests/josev-q4k/CMakeLists.txt) uses the
production `GGML_LLAMAFILE=ON` profile, CPU two threads, native instruction
autodetection off and compilation with two workers. The ordinary archive has
no observer macros; a distinct tested archive retains original named stages.
Actual symbols/strings and positive observer symbols are checked. Existing
model-free shape/gate/metadata controls and real `crispembed-diff::Ref`
positive/numeric/structural controls run first.

The [Kaggle CPU worker](../tools/kaggle/josev-q4k/worker.py) accepts only the
frozen v5 report SHA256
`51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090`,
worker source `b2edf8c83c9aca1f8a9210198d47798ac13828b8`, the frozen four
requests, and all 251 original NPY files (279,087,860 bytes). The coordinator
attaches this immutable input privately; the public template carries no account
or access reference. All payload hashes and original source hashes are checked
before the published F32 input is requested. Originals are copied byte-for-byte
to the remote result for preservation; no original model is executed and no
original IDs or hidden arrays are injected into native inference.

Actual F32 plain/observed/repeat execution must first pass all 227 ordered
original stage comparisons and four complete typed answers. IDs, attention
masks, spans, lexical IDs and field order are exact. Only then may conversion
or quantized inference begin. Q4_K repeats the same full capture and fixed
`Ref`/independent NumPy checks, preserving absolute/RMS/mean errors, norms,
magnitudes, minimum row cosine and the earliest differing/failing stage. The
original gates remain 0.002 absolute / 0.99999 row cosine; final logits 0.002,
probabilities 0.0001 and ordinal expectation 0.001. Numerical rejections are
collected across all four cases and are never relabeled F32 parity.

Separately declared owned-task checks retain every raw JSON response, exact
typed JSON equality, ordered discrete-option argmax identity and the unchanged
0.001 ordinal-expectation bound. No new score tolerance is selected after an
outcome. Six loaded-context unset/empty/zero negatives must clear outputs;
recovery and ordinary/observed/repeat raw final bits must agree. Even a passing
owned task result does not qualify independent task quality or publication.
Representative labeled evaluation remains explicitly pending.

The bootstrap needs a reviewed exact commit substituted for `@COMMIT@` and a
private v5 input attachment. It rejects a GPU reservation, enforces 14 GiB
available RAM/24 GiB disk before payloads and stops after a bounded four-hour
model worker. It uses an isolated CPython 3.13 environment without ensurepip,
with only SHA-pinned pip and NumPy; no Torch, tokenizer or model framework is
installed. Source/stdlib tests are suitable for a constrained development host;
all compilation, model downloads and inference belong on hosted infrastructure.
The source template performs no submission. Check all account jobs and the
five-session limit before any future approved dispatch.

## Initial hosted setup rejection and corrected accounting

The first actual v1 at source `5234e69c75dc000b9ae1eb0f5f47c79abd50af6a`
[stopped at resource preflight](../tests/results/josev2b/q4k-initial-setup-rejection.json):
the CPU image has no `/usr/bin/time`. Source/submodule checks, locked pip/NumPy,
pip check and all eleven stdlib tests passed. The earlier `wrapt` sitecustomize
warning was nonterminal. The worker had not read original arrays, compiled the
probe, requested the published model or run inference. The exact small
[worker receipt](../tests/results/josev2b/q4k-initial-native.json) and
[bootstrap receipt](../tests/results/josev2b/q4k-initial-bootstrap.json) remain
unchanged, with log identities retained separately. This supplies no numerical
or model-quality conclusion.

The correction uses [Linux per-child wait4](../tools/ci_process.py), with wall
time measured by a monotonic clock and `ru_maxrss` converted from KiB to bytes.
The direct child is reaped once by wait4; no Popen poll/wait or cumulative
RUSAGE_CHILDREN peak consumes its resource record. Timeouts kill the process
group and reap the direct child; worker termination allows cleanup before the
outer deadline kills the wrapper. Per-child peak RSS is the largest-process
metric, not the simultaneous memory sum of a compiler process tree. Five tiny
stdlib controls cover successful/high-and-low independent peaks, exit7, signal
termination, process-group timeout, interruption cleanup and split stdout. No
external timing package, model math, original output or acceptance gate changes.
Actual corrected compilation/inference remains pending coordinator review.

### Second setup rejection

The [second hosted bootstrap](../tests/results/josev2b/q4k-child-controls-initial-review.json)
passed source, isolated tooling and all eleven packer controls, then stopped
before model access. A child-accounting test parsed the last merged log line as
a PID; provider startup diagnostics could follow the actual PID. The corrected
test uses one explicit PID marker and exercises noisy child stderr. It still
checks that timeout kills the process group and reaps the measured child. No
package is added and no model, recipe, original output or budget changes.

### Original array extent accounting

The [third hosted preflight](../tests/results/josev2b/q4k-array-extent-initial-review.json)
verified every one of the 251 original file hashes, then rejected an accounting
error before model access or compilation. The frozen tensor payload is
279,087,860 bytes; the NPY files total 279,119,988 bytes, including 251 unchanged
128-byte headers. The worker now requires both independently derived extents.
All original file hashes, shapes, dtypes, inputs and outputs remain unchanged.
A new control exercises this exact frozen metadata and rejects altered file,
shape and payload counts without loading any arrays.
