# Prepared German GECToR CPU target

This standalone experimental target executes the exact fine-tuned Proofbird
German BERT encoder and both GECToR classifiers. It accepts prepared token IDs;
tokenization, first-subword alignment, iterative editing, locale policy and
UTF-16 source offsets remain responsibilities of the shared Quds provider.

The accepted source is `proofbird/gector-bert-de` revision
`8e3b50beef7d775cedb7b876009e88bcf2e8d6af`, original ONNX SHA-256
`6d7b43b4bd95c7a00c02c8babc7569d28558df894afe4874d96c202e6ab9a464`.
The converter is [`models/convert-gector-onnx-to-gguf.py`](../../models/convert-gector-onnx-to-gguf.py).
Its manifest binds all tensor layouts, exact source values, tokenizer/label
assets and actual precisions. GGUF source metadata identifies a conversion;
consumers should verify a trusted distribution hash before loading a model.

## Build and dependency boundary

```sh
git submodule update --init ggml
cmake -S tools/gector -B build-gector -DCMAKE_BUILD_TYPE=Release
cmake --build build-gector --parallel 2
ctest --test-dir build-gector --output-on-failure
```

The shared library compiles exactly `src/gector.cpp`, the existing
`src/core/gguf_loader.cpp`, and the pinned ggml CPU dependency. It does not
configure the parent project, model catalog/downloader, OCR/document stack,
C2PA, PDFium or sibling CrispASR components. GPU, BLAS, OpenMP, runtime backend
loading, KleidiAI and HBM dependencies are disabled. ggml's MIT llamafile SGEMM
is enabled. `GGML_NATIVE` and `CRISPEMBED_NATIVE` are off; x86 follows ggml's
fixed AVX2/FMA/F16C baseline, not the build machine's native ISA.

`compile_commands.json` records the actual compiled source set. Qualification
must also record linked system libraries, toolchain, model/notice files and
their applicable licenses for the chosen platform. This source boundary does
not by itself qualify every distributed binary or platform.

The install target includes the CrispEmbed license and the ggml/llamafile
[third-party notices](THIRD_PARTY_NOTICES). The initial GNU/Linux build resolves
system libstdc++ and libgcc with the GCC Runtime Library Exception, and GNU
libc/libm/loader with their own LGPL/component notices. They are not bundled by
this target and are not MIT dependencies. A policy excluding those system
runtimes as well needs a separately qualified toolchain/platform build.

Targets:

- `crispembed-gector`: the small shared C API library.
- `crispembed-gector-run`: bounded prepared-ID driver with raw logits, hidden
  states, optional per-layer traces and forward timing.
- `crispembed-gector-quantize`: the existing ggml quantizer as a development
  tool; it is not linked into the inference library.
- `test-gector`: model-free argument, gate and model-admission controls.

## Execution contract

Set `CRISPEMBED_GECTOR=1` to enable model creation and inference. Unset, empty,
`0`, `false`, `off` and `no` disable the path (the words are case-insensitive).
The general CrispEmbed embedding and OCR libraries are unchanged.

The C declarations are in [`src/gector.h`](../../src/gector.h). Inputs are
explicit int64 arrays, batch size one, 1 through 512 tokens and a matching
all-ones mask. IDs must lie in `[0,31102)`. Token type IDs are zero; absolute
positions are `0..T-1`. There is no implicit truncation, padding, special-token
insertion or int64 narrowing. Unsupported metadata, shape, precision, missing
tensors and truncated payloads fail before weight execution.

Both output arrays are required and caller-owned: token-major F32 `[T,5000]`
label logits and `[T,2]` detection logits. Optional hidden output is `[T,768]`.
Capacities count float elements, not bytes. Output ranges must not overlap,
and errors leave them untouched. Calls on one context serialize; destroy a
context only after its users stop. Reusing a context retains loaded weights and
a graph workspace for the most recent sequence length and diagnostic mode.
The shared GGUF loader can map model pages directly. Keep the model file
immutable for the entire context lifetime; replacing it requires closing the
old context and loading a verified new artifact. Never truncate or modify a
file still used by a live context.

`crispembed_gector_forward_trace` can deliver the normalized embedding, all
12 encoder-layer outputs, and both heads to a callback. These temporary views
are valid only within the callback. Do not re-enter or destroy that context
from the callback. Trace outputs are retained only when explicitly requested;
ordinary inference does not retain every intermediate layer.

The graph reproduces word-plus-type-plus-position ordering, source LayerNorm
epsilon, exact Erf GELU and separate source Q/K scaling operations. Attention
uses explicit F32 logits/softmax and the source post-softmax NaN-to-zero step.
It does not inherit the general encoder's flash-attention rounding. Both
trained heads are ggml matrix operations over the final hidden states.

## Bounded driver

`ids.txt` contains whitespace-separated decimal token IDs with source special
positions already present:

```sh
CRISPEMBED_GECTOR=1 build-gector/crispembed-gector-run \
  --model gector-de-f32.gguf --ids ids.txt --output result \
  --threads 2 --warmup 1 --repeats 5 --trace-prefix trace
```

This writes `result.labels.f32`, `result.detect.f32`, `result.encoder.f32` and
`result.json`. Binary arrays are little-endian F32, token-major, without a
header; the driver rejects big-endian hosts. The JSON records shape, explicit
threads, load time, individual forward measurements, median and nearest-rank
p95. When requested, `trace.embedding.f32`, `trace.layer.0.f32` through
`trace.layer.11.f32`, `trace.labels.f32` and `trace.detect.f32` contain one
additional diagnostic pass. It is excluded from the timing samples.

Forward timing includes graph setup when needed, input/output transfers and
both heads. It excludes model loading, tokenization, edit decoding, locale and
editor guards. A warmup of zero includes the first graph build in the samples.
These timings are not complete-checker latency.

F32 parity, mixed Q4_K quality, full corrections and intended-device limits
are separately recorded in the [feature roadmap](../../docs/roadmaps/german-gector-proofing.md).
The existence of this target does not enable an application default or qualify
a model recipe.
