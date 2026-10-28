# Frozen real-scan RGB input audit

This is source preparation for a **model-free, decode-only** comparison of all
20 images in [the unchanged real-scan manifest](ocr-real-scans-v1.json). It tests
the hypothesis that identical encoded image bytes can produce different RGB
bytes in Pillow and the production STB decoder. No execution result, numerical
outcome, cause of an OCR token mismatch, quality qualification or production
change is claimed here.

## Exact decoder boundaries

The production implementation is `src/image_preprocess.cpp:7–9`: nonstatic
`STB_IMAGE_IMPLEMENTATION` from
[`examples/stb_image.h` at a06](https://github.com/CrispStrobe/ggml/blob/a06fc0bb46b73cc10011fc90ae4054aee8438aeb/examples/stb_image.h).
The header is STB2.28, 284,733 bytes, SHA256
`38e08c1c5ab8869ae8d605ddaefa85ad3fea24a2964fd63a099c0c0f79c70bcc`.
Its original license text remains in the header. The common native CLI path
[`examples/cli/main.cpp:2203`](https://github.com/CrispStrobe/CrispEmbed/blob/492c99c253879338f6f87b1074125767e00e44f4/examples/cli/main.cpp#L2203)
calls `stbi_load(...,3)` before passing RGB to either MiniCPM or Unlimited.

The [Mini original worker](../tools/kaggle/ocr-real-scans/ref_worker.py) opens
the admitted path with Pillow and converts to RGB. The actual Unlimited
[`infer` path at its pinned revision](https://huggingface.co/baidu/Unlimited-OCR-2/blob/04647c7a6098d8d58915f003536a741c82eafb40/modeling_unlimitedocr.py#L1070)
does the same direct conversion. That 55,724-byte source was authenticated by
Git blob `2a717b55c0509995978d0a1e78802a997c26660e`; its observed SHA256 is
`f6df813cef23cc28b1daca40fd8cb20740d05a238e7b9dfa2334531fdcf39f1f`.
The separate Unlimited `load_image` helper applies EXIF transpose, but `infer`
does not call it. This audit applies no EXIF transpose, ICC color transform,
resize, model normalization, tiling or cleanup.

## Tiny hosted implementation

[`ocr-rgb-input-v1.json`](ocr-rgb-input-v1.json) freezes the source and protocol.
The build extracts the **exact production prefix through the STB include** from
the authenticated translation unit. It keeps the same production headers and
macros and compiles that prefix plus a small caller. There is no root build,
GGML backend, scan-cleanup implementation, model loader or inference weight.

The declared Linux x64 portable Release profile comes from the production
CMake native-off path and pinned ggml AVX2/FMA/F16C defaults: GNU C++17, O3,
NDEBUG, PIC, AVX2/FMA/F16C. Actual generated compiler records must match; extra
STBI overrides, native ISA selection, fast-math and unchecked response files
reject. Under those exact flags, the relevant preprocessor macros of the
generated prefix must equal those of the unchanged full production translation
unit, including the SSE2 decoder path. This is a checked source-derived profile,
not an assertion about unretained historical compiler binaries.

Pillow11.3.0 matches the original real-worker version pin. Three official
SHA-pinned CPython3.12 wheels provide Pillow, NumPy2.2.6 metrics and PyArrow17
for the already frozen CORD image bytes. The isolated worker records actual
package owners, the Pillow extension hash and JPEG/libjpeg-turbo/zlib versions.
Historical reference reports did not pin their wheel/libjpeg binaries; matching
version strings do not assert historical binary identity.

## Input admission and results

The 75,543-byte manifest has SHA256
`0857bffc6f4a3616292db255d7f3f34c0b58a52a7286a2791cb00b77499c7700`.
Every encoded image must match its exact declared length and SHA **before
either decoder runs**. Retained encoded files can be supplied explicitly;
otherwise hosted execution uses the frozen public URLs. CORD's exact parquet
SHA is checked before parsing its declared `image.bytes` rows, then each encoded
image passes its own admission gate. Failed transfers or identities remain
visible as exclusions with neither decoder attempted. A failure of one admitted
decoder does not hide the independent other decoder's result.

Each page reports manifest dimensions separately from actual Pillow and STB
dimensions, RGB channel/byte counts and hashes. Equal-geometry results include
maximum absolute, mean absolute and RMS byte differences, changed values and
pixels, first differing pixel/channel with both RGB triplets, and per-channel
metrics. Signed differences mean STB minus Pillow. There is no similarity
tolerance: exactness is reported as observed. Any RGB difference establishes
only this input boundary; attributing late OCR ID changes requires a separate
model-stage experiment with the frozen originals and unchanged gates.

All 20 identities stay in the report, including errors. Ground truth, prompts,
caps and scoring normalization are not read as audit targets or changed. Raw
encoded files and both RGB streams remain in the remote raw artifact, along
with unmodified manifest credits and ground-truth provenance/licenses. CORD
CC BY4.0 and Wikisource/public-domain scan credits remain separate. Credits
describe the admitted frozen sources, not a changed input rejected by its hash.

## Execution contract

The coordinator must register the new standalone
[`ocr-rgb-input-audit-v1.yml`](../.github/workflows/ocr-rgb-input-audit-v1.yml)
before a reviewed dispatch; it does not expand the metadata call graph. The
workflow checks RAM/disk before packages or images, uses two build workers and
an isolated Python environment, and uploads small review and raw artifacts
separately for90 days even after failure. No dispatch or main registration is
performed by this source-preparation lane.

Expected small result: `review/rgb-input.json`; raw RGB and encoded images belong
in the separate raw artifact and must not be downloaded for the first review.
Success requires both decoders to complete on all20 admitted pages, while RGB
inequality remains a diagnostic result rather than a quality gate. A future
genuine Kaggle CPU caller may use the same driver and exact locks only after an
actual free-slot/capacity check; no such caller or job is submitted here. No
model recapture, token or text generation, alias/default change or release
qualification follows from this audit.
