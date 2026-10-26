# MI-GAN fixed-256 CPU/F32 prototype

This is a separately built experimental graph, following the unchanged original
module and [frozen reference protocol](migan-reference.md). Source preparation
and native execution are separate: no native result qualifies before the first
actual original checkpoint SHA, tensor inventory and eight-case capture have
been reviewed and frozen. Preparation alone establishes no numerical, image,
backend or representative inpainting quality result.

[`src/migan.cpp`](../src/migan.cpp) implements the original256→4 encoder and
4→256 synthesis graph on the CPU backend, retaining every learned filter,
upsample insertion mask, noise constant and noise strength. The loader requires
exactly154 original-name finite F32 tensors, their fixed shapes,
`migan.image_size=256`, original source commit and a separately supplied reviewed
checkpoint SHA matching GGUF metadata. The owning hosted driver must also check
the actual GGUF SHA against the reviewed conversion receipt.

Both model initialization and execution require the shared value-parsed
`core_env::on("CRISPEMBED_MIGAN")` gate. Unset, empty and literal0 reject before
loading/execution. The context uses the CPU backend directly and makes no GPU
or fallback claim. A context is used serially; it is not a concurrent inference
API. The first comparison uses two native CPU threads, while the original uses
four: those timings cannot establish a paired speedup.

Pointwise/fromRGB/toRGB convolution explicitly constructs F32 im2col followed by
F32 matmul and requests F32 precision. Direct depthwise convolution uses the
pinned ggml's contiguous WHCN F32 source/kernel/F32 accumulation path. Every
constructed graph node must be F32 and supported by the selected CPU backend.
This makes activation precision an executable check. The graph uses the loaded
4×4 filters, nearest2× upsample, loaded insertion masks and exact
`pad_ext(left=2,right=1,top=2,bottom=1)`. Noise is multiplied and added from the
loaded state at its original position. Leaky ReLU, sqrt2 gain and clamp256 occur
after depthwise and after pointwise/upsample/noise in each separable layer.
Additions and observed convolution outputs remain distinct; the original raw
convolution hooks snapshot values before later in-place Python operations.

The C++ header exposes an internal prepared-F32 graph plus fixed-size RGB8/
binary-mask preprocessing and original uint8 truncation/composition helpers.
Dimensions and byte counts are strict; nonbinary masks and nonfinite input/
output reject. White255 means fill. Known RGB bytes are copied from the original
image. There is no arbitrary-size resize/crop, ONNX smoothing wrapper, public
C ABI, orchestrator/CLI/server/binding/registry integration or installed artifact.

## Hosted validation interface

Build only the explicit `crispembed-migan` and `test-migan-diff` targets in
hosted CI. The targets are `EXCLUDE_FROM_ALL`, and neither is added to the
production shared library or install lists. `test-migan-diff` links a separate
`crispembed-migan-test` archive compiled with a test-only observer; the ordinary
prototype archive contains no observer callback or original reference payload.

The test command is:

```sh
test-migan-diff MODEL.gguf REFERENCE.gguf CHECKPOINT_SHA256 \
  RGB.raw FILL_MASK.raw ORIGINAL_COMPOSITED.raw DUMP_DIR
```

All three `.raw` inputs are exact256×256 uint8 payloads (RGB HWC, mask HW, RGB
HWC). The hosted driver derives their bytes from the frozen original case and
checks their identities before execution. `REFERENCE.gguf` is that case's
separate original diagnostic artifact. The harness performs actual RGB8
preprocessing, compares its tensor to the original and executes that native
input. It never substitutes golden outputs or reconstructed operator outputs.

Subtitle-bar and edge cases execute a second observed graph. Its retained nodes
are marked `ggml_set_output` **before allocation**, preventing allocator reuse
from destroying intermediates. Ordinary and observed final output bytes must
match exactly. Each original stage must be compared exactly once, including
block inputs and tuple outputs; missing or extra stage coverage rejects.
The other six cases execute only the ordinary graph and final comparison.
References and native arrays stay remote; only small reports/images need local
review. Allocator bytes and timings are diagnostics, not backend or speed proof.

`crispembed-diff` supplies stage maximum absolute error, MAE, RMS, per-row/global
cosine and both L2 magnitudes. The test enforces the frozen0.002/0.99999 gates
directly, without the harness's optional cosine environment override. It records
the first failing stage in execution order, checks finite values and dimensions,
and writes actual native `.f32` arrays for attribution. Decoded fill RGB8 must
pass max1/MAE0.1/PSNR50dB; known pixels must be bit exact. Empty fills have null
image metrics, and exact filled images have null PSNR plus zero MSE and
`exact_fill=true` (infinite PSNR). Both native generated/composited raw RGB8
images are preserved for the owning hosted driver to encode and review.

Machine-readable stdout records use `MIGAN_CONTRACT`, `MIGAN_GATES`,
`MIGAN_DIFF`, `MIGAN_IMAGE` and `MIGAN_SUMMARY` prefixes. Exit0 requires complete
coverage, every float/image gate and observer neutrality; exit1 retains numerical
or decoded-image failures, and exit2 indicates setup/contract/coverage failure.
`--contracts` exercises pure RGB/mask helpers; `--reject MODEL SHA` checks a
hosted malformed-model control with the gate enabled. These are validation
interfaces, not product CLI options.

No local builds, framework imports, model access or inference are part of this
preparation. Formatting and existing dependency-free contracts can be checked
locally; actual compilation, eight-case comparison, resource measurements and
negative model variants require hosted execution after the original freeze.
Any failed alternative, strict flag or initial run must remain retained.

The graph decomposition also studies/adapts
[vision.cpp at26a752912d49f6c4ff4545b35a1bdf7400d349ed](https://github.com/Acly/vision.cpp/tree/26a752912d49f6c4ff4545b35a1bdf7400d349ed).
Its full [MIT notice](third-party/vision-cpp-LICENSE.txt), Copyright2025 The
vision.cpp authors, is retained. The original MI-GAN source/weight MIT notices
remain with original artifacts. No SH-GAN or ONNX smoothing source is copied.
This prototype changes no runtime release, application pin, model alias or
default. F16/quantization, other backends, real subtitle/scene quality and video
remain separate protocols.
