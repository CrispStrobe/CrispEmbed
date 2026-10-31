# TeleOCR: native GGUF / GGML integration

TeleOCR uses CrispEmbed's native C++ Qwen vision/decoder engine. The converter,
single-image OCR, explicit pipeline stage, Python/Rust/Dart bindings and HTTP
route all use `general.architecture = teleocr`. The optional document parser
performs layout, crops and formatting in Python while keeping every model
forward in the same native GGML engine.

See the [retained validation record](teleocr-validation.md) for the actual CPU
results and reproducibility details. **Strict numerical qualification remains
false**, so this implementation remains experimental and disabled by default.

**License: non-commercial.** TeleOCR's vision tower derives from
Qwen2.5-VL-3B-Instruct, released under the Qwen RESEARCH LICENSE AGREEMENT
(research/evaluation only), although the TeleOCR card declares `apache-2.0`.
See [the weight-license finding](teleocr-license.md#weight-license-lineage).
Loading therefore also requires `CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1`, and
the registry download requires `--accept-license qwen-research`.

The new path is opt-in with `CRISPEMBED_TELEOCR=1`, following the repository's
new-backend policy. An unset, empty, or literal `0` variable rejects TeleOCR
loading. The shared repository gate treats any other nonempty value (including
`false`) as enabled; use literal `0` to disable it.
Published GGUFs: [cstr/TeleOCR-GGUF](https://huggingface.co/cstr/TeleOCR-GGUF) at immutable commit `36f626c1`, with registry aliases `teleocr-f16`, `teleocr-q8_0` and `teleocr-q4_k-m` pinned by SHA-256. Each was token-exact against the pinned original model on all six frozen cases (hosted receipts in `tests/results/teleocr/hosted-*.json`); strict numerical qualification remains false.

TeleOCR uses the CPU backend by default. `CRISPEMBED_TELEOCR_GPU=1` opts into
experimental GPU placement, while `QWEN2VL_OCR_FORCE_CPU=1` still forces CPU.
The retained execution evidence covers CPU; GPU numerical/decoded parity and
platform qualification remain separate and must not be inferred from a GPU
allocation or a successful build.

## Source and license boundaries

The checkpoint is [XingChen-AGI/TeleOCR](https://huggingface.co/XingChen-AGI/TeleOCR),
pinned to revision `b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d` for conversion and
validation. The Apache-2.0 Rust reference is
[teleocr-rs at 957cdbabe80e86d652c5fcc938ed7fbcf1eb2bbe](https://github.com/rzafiamy/teleocr-rs/tree/957cdbabe80e86d652c5fcc938ed7fbcf1eb2bbe).
See [the detailed license/provenance record](teleocr-license.md) and
`NOTICE.teleocr`. Apache-derived files retain their notices; this does not
relicense those files to MIT. The surrounding CrispEmbed project retains its
MIT license.

Conversion reads config, tokenizer and safetensors data. It does not import
checkpoint Python modules or execute `trust_remote_code`. The optional Python
reference dumper is a separate validation tool and is not part of inference.

## Architecture contract

TeleOCR's `qwen2_5_vl` config name alone is insufficient to select the math.
Its decoder has Qwen3-style per-head Q/K RMSNorm and an explicit head width
that differs from `hidden_size / num_attention_heads`.

| Part | TeleOCR configuration |
| --- | --- |
| Vision | 32 blocks; width 1280; 16 heads; SwiGLU intermediate width 3420 |
| Patch embedding | RGB, 14 × 14 pixels, temporal size 2 (the still frame is repeated) |
| Vision attention | 112-pixel windows; full attention at blocks 7, 15, 23, 31 |
| Image merger | 2 × 2 patches per output token; output width 1024 |
| Decoder | 28 blocks; hidden width 1024; intermediate width 3072 |
| Decoder attention | 16 query heads, 8 KV heads, **128 values per head** |
| Rotary positions | Non-interleaved multimodal sections `[16, 24, 24]`; theta 1,000,000 |
| Vocabulary | 151,936 rows; output head tied to token embeddings |
| Container | `teleocr.gguf_version = 1`; strict profile and tensor-shape checks |

The tensor naming convention reuses CrispEmbed's `v.*` and `l.*` keys. The
converter writes the `qwen2vl.*` dimensions and the QK-normalization flag
explicitly. A normal Qwen2/2.5-VL or Qwen3-VL file does not satisfy the TeleOCR
stage's model-profile check.

Image preprocessing follows the pinned **slow `Qwen2VLImageProcessor` with
Pillow 13.0.0** (`use_fast=False`). Its bicubic resize selects the vertical
pass first when the positive height reduction exceeds twice the width
reduction; this detail changes pixel rounding on some mixed
downscale/upscale inputs. The native
TeleOCR path implements the same pass-order selection. The validation
dependencies in `models/requirements-teleocr-reference.txt` freeze this
processor contract; another Pillow version or a fast processor constitutes
a different reference and must be checked separately.

## Convert the checkpoint

Use Python 3.11 or newer:

```sh
python -m pip install -r models/requirements-teleocr.txt
python models/convert-teleocr-to-gguf.py \
  --model XingChen-AGI/TeleOCR \
  --revision b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d \
  --output teleocr-f16.gguf --dtype f16
```

For an existing checkpoint directory, conversion works offline:

```sh
python models/convert-teleocr-to-gguf.py \
  --model ./TeleOCR --output teleocr-f16.gguf --dtype f16
```

`--dtype` accepts `f32`, `f16`, or `q8_0`. The converter produces a JSON
manifest next to the GGUF, checks all 701 source tensors, verifies that the
two tied embedding/head tensors are identical, then writes 700 tensors.
Unknown, missing, incorrectly shaped or incompatible tensors are errors.
An existing output is not overwritten unless `--force` is supplied.

The Q8 recipe quantizes 196 decoder matrices. It preserves vision, merger
and the tied embedding in F16 and the 340 normalization/bias vectors in F32.
Quantization support is distinct from quality qualification: use the retained
per-format results to decide which precision fits a particular workload.

An explicitly experimental Q4_K recipe is available through the existing
quantizer after building its `crispembed-quantize` target:

```sh
./build/crispembed-quantize teleocr-f16.gguf teleocr-q4_k.gguf q4_k \
  --tensor-type 'v.*=f16' --tensor-type 'l.embed_tokens.weight=f16'
```

This recipe quantizes the same 196 decoder matrices and retains 164 F16 and
340 F32 tensors. The vision and tied embedding overrides are intentional.
The quantizer writes the TeleOCR recipe and resulting tensor-type census to
its provenance metadata. This command does not establish decoded or quality
parity for Q4_K; evaluate its retained differential results separately.

## Build and run raw OCR

For a Git checkout, initialize the pinned submodule with
`git submodule update --init --recursive`. Source ZIP users skip this step:
the archive already contains the pinned ggml sources and has no Git metadata.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF
cmake --build build --target crispembed-cli crispembed-shared crispembed-server --parallel 2

export CRISPEMBED_TELEOCR=1 CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1
./build/crispembed -m teleocr-f16.gguf --ocr page.png \
  --ocr-max-tokens 4096 -t 2 --json
```

The default raw task is `Please output the text content from the image.`
Its model-card generation defaults are repetition penalty 1.05, presence and
frequency penalties 0, no n-gram ban, greedy decoding and a 4096-token cap.

Raw task selection uses the existing prompt interface:

| Task | Exact prompt |
| --- | --- |
| Text | `Please output the text content from the image.` |
| Table | `This is the image of a table. Please output the table in OTSL format.` |
| Formula | `Please write out the expression of the formula in the image using LaTeX format.` |
| Code | `The image contains a code snippet, please output the parsing result.` |
| Layout | `Analyze the image layout.` |
| Scientific figure | `This is a scientific figure. Please extract the table implied by this figure.` |

```sh
./build/crispembed -m teleocr-f16.gguf --ocr table.png \
  --ocr-prompt 'This is the image of a table. Please output the table in OTSL format.' \
  --ocr-max-tokens 4096 -t 2
```

Raw table output is OTSL and raw layout output contains normalized boxes/type
labels. Use the document parser below for crops, table HTML and Markdown.

## Explicit orchestrator stage

Public stage ID **24** is TeleOCR. Existing stage IDs are unchanged. This is
the full-stage builder's ID space, not the older `--vlm-engine` escalation
numbering.

```sh
./build/crispembed --ocr-pipeline page.png --ocr-engine teleocr \
  --ocr-rec teleocr-f16.gguf --ocr-max-tokens 4096 -t 2 --json
```

The stage owns its model path independently of other Qwen stages. Switching
models frees and reloads that context. Each call resets the stage prompt and
generation cap to its configured values, so an earlier stage cannot leak a
custom prompt or smaller cap into a later stage. VLM cleanup behavior follows
the existing orchestrator policy.

### Python

```python
from crispembed import CrispOcrModel, CrispOcrOrchestrator, OCR_ENGINE_TELEOCR

ocr = CrispOcrModel("teleocr-f16.gguf", n_threads=2,
                    lib_path="build/libcrispembed.so")
ocr.set_max_tokens(4096)
text = ocr.recognize("page.png")
scores = ocr.confidences()

# Optional explicit generation controls; these return False on other models.
assert ocr.set_generation_params(repetition_penalty=1.0,
                                 presence_penalty=1.0,
                                 frequency_penalty=0.005,
                                 no_repeat_ngram=100)
print(ocr.get_generation_params())

orch = CrispOcrOrchestrator(
    router=False, n_threads=2, lib_path="build/libcrispembed.so",
    stages=[{"engine": OCR_ENGINE_TELEOCR, "model_a": "teleocr-f16.gguf",
             "vlm_max_tokens": 4096, "min_chars": 1, "min_confidence": 0}],
)
result = orch.run("page.png")
```

Set the environment gate before creating either context. The library filename
is `libcrispembed.dylib` on macOS and `crispembed.dll` on Windows.

### C, Rust and Dart

The existing `crispembed_ocr_model_*` C API automatically dispatches a TeleOCR
GGUF. `CRISPEMBED_OCR_ENGINE_TELEOCR` selects the explicit stage builder. The
new `crispembed_ocr_model_set_generation_params` and corresponding getter
return 1 only for valid TeleOCR settings; they leave unsupported engines and
invalid requests unchanged. Each getter output pointer may be null to skip
that value.

Rust uses `OcrModel`, its `set_generation_params`/`generation_params` methods,
and `OCR_ENGINE_TELEOCR` with `OcrStageSpec`. Dart uses `CrispOcrModel`, its
`setGenerationParams`/`generationParams` methods, and
`crispOcrEngineTeleocr` with `CrispOcrStageSpec`. These wrappers reuse the
existing native image and context ownership APIs.

### HTTP

```sh
CRISPEMBED_TELEOCR=1 CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1 ./build/crispembed-server \
  --ocr teleocr-f16.gguf --ocr-max-tokens 4096 -t 2 \
  --host 127.0.0.1 --port 8080 --image-root ./pages

curl http://127.0.0.1:8080/ocr/model \
  -H 'Content-Type: application/json' \
  -d '{"image":"./pages/table.png","max_tokens":1024,
       "prompt":"This is the image of a table. Please output the table in OTSL format.",
       "repetition_penalty":1.0,"presence_penalty":1.0,
       "frequency_penalty":0.005,"no_repeat_ngram":100}'
```

The response retains the existing `latex` text field, token confidence array
and timing fields. Prompt, cap and penalty overrides are request scoped and
are restored on success and on image-loading errors. Invalid or unsupported
penalty requests return HTTP 400. The existing `/math/ocr` alias also works.
Use `--ocr-pipeline --ocr-engine teleocr --ocr-rec FILE` for the explicit
orchestrator endpoint `/ocr/pipeline`.

## Parse a complete document

`tools/teleocr_parse.py` implements a sequential two-pass document pipeline:

1. Resize a copy of the page to 1036 × 1036 and infer layout.
2. Validate boxes/types and retain the model's reading order.
3. Crop the source page, mask polygon exteriors for segmentation mode, rotate
   blocks upright and pad/upscale narrow crops.
4. Infer the appropriate text, formula, table, code, seal or figure task.
5. Convert OTSL spans to escaped HTML, normalize math/code, and write Markdown
   plus optional structured JSON containing raw outputs and warnings.

```sh
python -m pip install numpy Pillow
CRISPEMBED_TELEOCR=1 CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1 python tools/teleocr_parse.py page.png \
  --model teleocr-f16.gguf --lib build/libcrispembed.so \
  --output page.md --json page.json --threads 2

# Photograph/curved-page layout uses polygons:
CRISPEMBED_TELEOCR=1 CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1 python tools/teleocr_parse.py photo.jpg \
  --model teleocr-f16.gguf --lib build/libcrispembed.so \
  --mode segmentation --output photo.md --json photo.json

# Optional PDF input renders pages through PDFium:
python -m pip install pypdfium2
CRISPEMBED_TELEOCR=1 CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE=1 python tools/teleocr_parse.py document.pdf \
  --model teleocr-f16.gguf --lib build/libcrispembed.so \
  --pdf-dpi 200 --output document.md --json document.json
```

The parser's prompts deliberately include the leading newline used by the
document-pipeline reference. Its task penalties differ from raw model-card
defaults: repetition 1.0; presence 1.0 and frequency 0.05 for ordinary blocks,
frequency 0.005 for tables/figures/seals; layout presence/frequency 0; n-gram
ban 100 throughout. The raw OCR API does not silently select these settings
from prompt text.

Headers, footers, page numbers and related page furniture are omitted from
Markdown unless `--paratext` is supplied; they remain in JSON. Image/list/
equation-block container regions are retained but not transcribed. The
default crop source limit is 4,500,000 pixels and the layout limit is 512
blocks. `--max-tokens` applies separately to layout and each block.

Malformed layout lines produce warnings, and a page with no valid layout
blocks fails. A malformed OTSL table retains its raw output in a code fence
with a warning instead of silently becoming an empty table. JSON records when
the token count reaches its cap; this is conservative and does not claim an
EOS/completion distinction that the generic token-confidence API cannot make.
The parser does not batch neural forwards, reconstruct figure images or
promise benchmark-level layout quality.

## Tests and interpretation

Weight-free contract and parser tests:

```sh
python tests/test_teleocr_converter.py
python tests/test_teleocr_parse.py
python tests/test_teleocr_surface_contracts.py
python tests/test_cli_engine_names.py
python tests/test_ocr_max_tokens_surfaces.py
python tests/test_teleocr_loader.py --lib build/libcrispembed.so \
  --report teleocr-loader.json
python tests/test_teleocr_quantizer.py --quantizer build/crispembed-quantize \
  --report quantizer-provenance-tests.json
```

The independent reference and native differential tests use the pinned slow
processor described above. In a separate Python 3.12 validation environment,
install the CPU PyTorch wheels before the remaining pinned dependencies (the
pinned NumPy requires Python 3.12 or newer):

```sh
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r models/requirements-teleocr-reference.txt
```

Reference testing requires a **full pinned checkpoint snapshot**. The
production converter deliberately downloads only the configuration,
preprocessor, tokenizer and safetensors data it needs; that minimal cache does
not include all executable model files, chat-template/generation settings and
sample assets used by the independent reference dumper. Download the complete
snapshot separately:

```sh
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="XingChen-AGI/TeleOCR",
    revision="b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d",
    local_dir="./TeleOCR",
)
PY
```

Then generate frozen references and run the native differential harness:

```sh
python tools/dump_teleocr_reference.py --model-dir ./TeleOCR \
  --out teleocr-reference --threads 2 --max-new-tokens 1024
cmake --build build --target test-teleocr-diff --parallel 2
python tools/run_teleocr_validation.py --binary build/test-teleocr-diff \
  --model teleocr-f32.gguf --reference-dir teleocr-reference \
  --out teleocr-validation-f32 --threads 2
```

The runner checks reference/image hashes, takes each generation cap from its
frozen reference metadata, and retains every case log and JSON report even
when another case fails. It refuses a nonempty output directory so an earlier
failure is not overwritten. Numerical activation checks and decoded-token/EOS
checks are separate report fields; matching text alone does not make a
numerical failure pass. Use a separate output directory for each precision
and each revision under investigation.

Actual public-interface consistency on a bounded sample:

```sh
python tests/test_teleocr_public.py \
  --lib build/libcrispembed.so --cli build/crispembed \
  --server build/crispembed-server --model teleocr-f16.gguf \
  --image text.png --max-tokens 8 --threads 2 \
  --report teleocr-public.json
```

This harness checks the disabled gate, native default/cap/prompt/penalty
controls, decoded text and confidence consistency across C, Python,
orchestrator, CLI and HTTP, including request-state recovery after an error.
It is an **interface consistency** test: its C output is an anchor from the
same native engine, not independent reference-model evidence. The separate
reference dumper and native differential harness must establish numerical
and decoded agreement with the pinned original model. Passing weight-free
tests or a compilation alone does not establish OCR quality, quantized
quality, GPU parity, platform qualification or publication readiness.

`tests/teleocr_public_bindings.rs` and `tests/teleocr_public_bindings.dart`
exercise the actual safe wrappers, including stage 24, generation setters,
reset behavior, UTF-8 output and context ownership. Their expected-text file
must contain the public harness's exact `anchor_text` bytes without an added
newline. These are additional interface checks against that same native
anchor, rather than independent model-reference tests. Retain the source
snapshots and hashes used to build/run these consumers with their results.

For a bounded native document-parser check:

```sh
python tests/test_teleocr_parser_native.py --lib build/libcrispembed.so \
  --model teleocr-f16.gguf --image tests/fixtures/teleocr/synthetic-text.png \
  --mode detection --out teleocr-parser-detection --max-tokens 1024 --threads 2
```

Use `--mode segmentation` and a suitable document image to exercise polygon
layout/cropping. This harness requires nonempty typed blocks and Markdown,
finite bounded confidences, and no generation reaching its cap. It retains
every raw layout/crop generation before interpretation, together with the
Markdown, JSON and checks, including when a later check fails. These are
document-pipeline smoke checks, not layout/transcription accuracy benchmarks.
