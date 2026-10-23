# Bounded M1 OvisOCR2 build and model audit

The Linux hosted queue currently has no running job. This optional Apple lane
can check the same candidate while the original GPU reference is being qualified.
It supplements Linux coverage and does not replace its numerical result.
Do not regenerate existing original processor captures or loosen any threshold.

Use an isolated worktree of `test/ovisocr2-native-parity`, reviewed starting
source `949a0a45`. Record the exact chosen full source SHA and verify the diff
before running a newer branch head. Keep all paths, access information and full
machine output in ignored environment notes, not public Markdown.

Before compiling, inspect load, available memory, swap changes and free space.
Use at most two build jobs. Stop costly work if the machine is under pressure;
never delete unrelated caches or projects. A 16GB machine still needs headroom
for the model and per-projection buffers. No GPU/MPS inference is requested.

## First stage: build without models

After fetching the branch, create an isolated worktree and configure:

```sh
cmake -S . -B build-ovis-audit -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF -DGGML_METAL=OFF \
  -DGGML_BLAS=OFF -DCRISPEMBED_BUILD_SHARED=ON
python3 scripts/check-cpu-baseline.py build-ovis-audit
cmake --build build-ovis-audit --target test-qwen35-decoder \
  test-qwen35-vision test-qwen35-delta test-ovisocr2-context \
  test-ovisocr2-quality-run -j2
```

Preserve configure/build logs and binary hashes. If resources still permit,
build the shared library, CLI and HTTP server targets to cover the changed
public dispatch/orchestrator units. Follow the exact target names in CMakeLists.
Do not launch the full suite or corpus. Report initial compiler/linker errors
with source and line, then fix only the demonstrated defect in the worktree.

## Second stage: original CPU F32 and native F32/F16

Wait for `tests/results/qwen35-ocr/ovisocr2-vllm-reference.json` with reviewed
positive original GPU execution. The hosted model driver deliberately fails
before dependencies or weights without that receipt. Do not bypass its gate.
Use an isolated Python3.12 environment, Apple-compatible Torch2.11.0 and
Torchvision0.26.0 wheels, Transformers5.16.1, Tokenizers0.23.3, NumPy2.2.6,
Pillow12.1.0, HF Hub1.33.0, GGUF0.19.0 and Safetensors0.8.0. The Linux CPU-wheel
index is not the source of Apple wheels; record actual wheel hashes/versions.

The current model controller requires a real MKLDNN-enabled Linux CPU and an
original MKLDNN-off control. It is deliberately not an executable Apple model
protocol yet. Finish the build-only first stage; an Apple reference extension
needs its own explicitly reviewed backend control and separate report. Do not
fabricate MKLDNN availability or bypass the Linux reference gate on an M1.
The instructions below describe the remaining contract for that extension.

The executable source is `tools/ci-heavy/ovisocr2_model_audit.py` and the
original adapter is `tools/dump_ovisocr2_reference.py`. Its `--preflight` checks
public service/source receipts before framework imports. If adapting the
controller for macOS, preserve the original script and all changes in Git,
including dylib/build-path differences. Keep its three processor-anchor SHA
comparisons, readable RGB SHA, full473-tensor export hashes, all72 hidden-layer
comparisons, complete scores/cache/logits/IDs/raw text, owning-context and
negative controls. CPU Torch on Apple is a distinct backend reference; state
that explicitly. Keep the original0.002 cache/raw budgets and all strict gates.

Run F32 with one owning context; F16 can use the two-context arm only while
memory permits. Preserve every failed arm and earliest stage/magnitude/error.
No selected final output or cosine can waive a failed hidden/logit comparison.

Store large model/reference artifacts remotely or in task-owned local storage;
return only small JSON/log reports, source/binary/input hashes and needed images.
Include peak RSS, wall time and new swapouts, with private paths scrubbed from
public evidence. Do not publish models or flip defaults from this build lane.

The primary agent continues GPU, Linux, public API, independent quality and
calibration/publication work. This handoff does not mark any of them complete.
