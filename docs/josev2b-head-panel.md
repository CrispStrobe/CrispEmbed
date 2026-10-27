# Frozen Josev head component panel

This is a standalone hosted build and replay harness for the
[head component](josev2b-head-native.md). It preserves head source
`d8b99a9c8e3a5cf0f6699fc026d3396d26e03763` and GGML
`890278a8342c620197c90e702e1188bcab94f510`. The original receipt remains
`51a2e16b2ee1163b4cf08215263135c0ceb760a4525b1bb89c66f9bd8436a090`,
worker `b2edf8c83c9aca1f8a9210198d47798ac13828b8`. No new original run,
backbone, public API, calibration, model alias or default change is included.
There is no local compile, payload read or native result in this source preparation.

The [standalone CMake project](../tests/josev-head/CMakeLists.txt) builds an
ordinary static head archive, a separate observer archive, the original probe,
and an actual `crispembed_diff::Ref` comparator. It does not include root CMake.
CUDA is optional and default off; when enabled, the probe still requires an exact
requested device and prohibits CPU fallback. Build with at most two workers on
hosted resources. The parent owns source registration, GPU bootstrap/private
inputs and actual dispatch; this harness performs no network/install/build action.

Hosted build example after exact source/submodule checkout:

```sh
cmake -S tests/josev-head -B head-build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
cmake --build head-build -j2
```

For the genuine GPU task, enable `GGML_CUDA` and select the reviewed hardware's
CUDA architecture during configuration. Full history or explicit immutable source
fetches must make both original worker`b2edf8c8` and head source`d8b99a9c`
available. Keep the actual initialized GGML checkout at890. GHA accepts only the
CPU panel; the genuine Kaggle GPU task requires the ordered`cpu CUDA0` panel and
fresh capability, driver and memory proof. A CPU-only Kaggle task is rejected.

The generic callable is
`run_head_panel(reference_dir, headGGUF, input_dir, build, out, backends)` in
[the driver](../tools/josev2b_head_panel.py). `build` is an already built
standalone directory. The callable always returns a report dictionary, including
recorded rejections. Callers must require `component_parity is True`; `completed`
alone means the case/backend attempts were recorded. The CLI takes the same paths and explicit backend list:

```sh
python tools/josev2b_head_panel.py \
  --reference-dir artifact/original \
  --head-gguf artifact/head/joint-head-f32.gguf \
  --input-dir artifact/inputs --build head-build --output panel --backends cpu
```

The frozen reference directory may be a separate read-only mount. Every one of
its251 NPYs, receipt and input freeze is revalidated. The head's parent artifact
root must provide`review/artifact.json`, `ordered-events.json`,
`native-input-adapters.json`, `head-conversion.json`, plus the sibling conversion
receipt and pinned`head/joint_schema_model.py`. The artifact completion declaration
must bind exact51+b2,251 arrays/279,087,860 payload bytes and the actual head hash,
size and122-tensor count. Every conversion tensor is checked against the captured
loaded-head F32 bytes, dtype transformation, shape and scalar adapter. Each exported
hidden/lexical raw file must have exact original bytes and provenance; field IDs,
types, spans and order must match the frozen serializer. There is no input rewrite.

Build preflight inspects actual compile commands for separate ordinary/observer
translations, then checks symbols and strings to exclude test observers and stage
labels from the ordinary archive. Hosted probe controls exercise metadata/input,
gates, observer failure and recovery. Tiny actual Ref controls require identical
F32 success, a small absolute-error single-row cosine rejection despite a loose
environment override, and structural shape rejection. These controls execute the
compiled comparator, rather than a Python cosine stand-in.

All123 original head/final events per backend map by ordered
`(stage,occurrence,logical_shape)`; repeated per-field labels are retained. Lossless
F32 GGUF adapters reverse C-order axes only in their GGUF dimension declarations
and preserve every float bit. Equal element counts do not permit original reshapes.
The actual C++ Ref comparator applies explicit fixed`max_abs<=0.002` and
`min_row_cosine>=0.99999` gates, ignoring environment threshold overrides. Independent
NumPy float64 reductions measure maximum/mean absolute error, mean signed error,
RMS, L2 norms, signed/absolute magnitudes and minimum contiguous-row cosine. Metric
agreement is separately checked; its reporting allowance never changes either
model gate. Ordinary/observed/repeat final raw bytes and probe JSON values must
also agree independently.

Final decoded responses are checked separately. Only the unmodified AST of the
pure publisher`systemone_answer` function is executed, after verifying the complete
pinned module SHA. No Torch import or whole-module execution occurs. It receives
actual native F32 probabilities and the frozen request/option order, preserving
original type behavior, ties, score weighting and rounding. Full response/usage
and answer maps are compared with the original official response.

A comparator numeric exit1 is collected, and all four cases on every requested
backend continue. Structural failures retain their own phase/error category and
partial files. The driver has a30-minute replay bound, with per-probe/comparator
limits, and records wall time and maximum child RSS. Exit0 means this head-only
owned smoke passed; exit1 is numerical/decoded rejection and exit2 is structural
rejection. Neither success nor component cosine establishes backbone,
representative quality, media, API or release qualification. The parent must retain
small review reports and all large evidence remotely, including failures. No
runtime version or release publication is created by this source-only harness.
