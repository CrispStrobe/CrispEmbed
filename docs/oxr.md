# OXR-1.0: experimental native GGML integration

This integration implements the OXR recognition and reading-order model in
native GGML, adds a pinned GGUF converter, and connects the native model to
OXR's original Python document helpers. The document route preserves the
upstream layout parsing, crops, reading-order recovery, content repairs, JSON,
and Markdown assembly. Layout detection remains a separate model: supply
regions explicitly, use the native D-FINE layout detector
([docs/oxr-dfine.md](oxr-dfine.md), `layout_model=`), or configure the original
D-FINE layout service.

**Quality parity and speed parity have not been established.** The code and
component controls are implemented; full checkpoint conversion, model-backed
stage comparisons, decoded-output comparisons, and benchmarks remain to be
run. `CRISPEMBED_OXR=1` is required to activate the model. Keep the experimental
status when describing or redistributing this implementation.

## Scope and interfaces

| Interface or component | Implemented behavior |
| --- | --- |
| Native model, `src/oxr_ocr.cpp` | SigLIP2-NaViT vision encoder, soft visual vocabulary, Qwen3 prefill and cached greedy decoding, original image/tokenizer contracts, explicit EOS and cap diagnostics. |
| `crispembed_oxr_*` public C API | One resident model context; arbitrary task prompt with image or text-only input; owned JSON result and error reporting. |
| Python `CrispOXR.predict` | One image request with upstream task prompts or an explicit prompt; native result and trace. |
| Python `CrispOXR.process_pages` / `process_document` | Original document helpers with native recognition and reading order, supplied regions or explicit external layout service, complete upstream JSON and Markdown outputs. |
| Generic OCR, CLI, orchestrator engine 26 | Recognition on the supplied image. Engine names are `oxr` and `oxr-1.0`; an explicit local GGUF is required. The existing HTTP orchestrator uses the same stage. |
| Rust `Oxr` and Dart `CrispOXR` | Owning wrappers for image and text-only C API requests; compilation and runtime validation remain pending. |
| D-FINE layout detector | Native GGML in `layout_detect` (`layout.arch=dfine`), converter `models/convert-oxr-dfine-to-gguf.py`, `crispembed_oxr_layout_*` API returning the service's `/inference` JSON; see [oxr-dfine.md](oxr-dfine.md). External service and supplied regions remain supported. |

The generic OCR/orchestrator stage does not run the Python document helper.
For page segmentation, block ordering, tables, and assembled document output,
use the `CrispOXR` document methods with an appropriate layout source.

## Immutable sources and license

The implementation is tied to these exact sources:

| Source | Revision / contract |
| --- | --- |
| [SoMarkAI/OXR-1.0 checkpoint and custom model code](https://huggingface.co/SoMarkAI/OXR-1.0/tree/c708407f9d2b4df658f2543d9132b62bb7221df4) | `c708407f9d2b4df658f2543d9132b62bb7221df4` |
| [SoMarkAI/OXR document pipeline](https://github.com/SoMarkAI/OXR/tree/c1a1dcb312ee9f691404f127785510453c0d9350) | `c1a1dcb312ee9f691404f127785510453c0d9350` |
| GGML submodule | `a06fc0bb46b73cc10011fc90ae4054aee8438aeb` |
| Original Transformers reference | Transformers `5.13.1`, Torch `2.11.0`, torchvision `0.26.0`, tokenizers `0.22.2`, safetensors `0.8.0`, Pillow `12.3.0` |
| Published serving profile | OXR's out-of-tree vLLM `0.25.1` plugin at the pinned model revision |

[`models/oxr-source.json`](../models/oxr-source.json) records file sizes and
SHA-256 hashes, the optional layout checkpoint, and the separate processor
profiles. The converter authenticates all required source files, including
the complete 3,068,857,520-byte `model.safetensors`, before accepting the model.
It checks all 752 tensor names and shapes, BF16 source storage, tied embedding
and output weights, the 151,669-entry tokenizer, and BPE/added-token metadata.

OXR is distributed under **Apache License 2.0 with additional OXR terms**.
Read the [actual pinned OXR LICENSE](https://github.com/SoMarkAI/OXR/blob/c1a1dcb312ee9f691404f127785510453c0d9350/LICENSE),
[`licenses/oxr-LICENSE`](../licenses/oxr-LICENSE),
[`licenses/oxr-APACHE-2.0`](../licenses/oxr-APACHE-2.0), and
[`licenses/oxr-NOTICE`](../licenses/oxr-NOTICE). The additional terms include a
separate commercial-license requirement above either USD 2 million in total
funding or annual revenue, consolidated across affiliates; SoMark OXR
attribution for online services; and a competing hosted OCR/document API
restriction that applies independently of those thresholds. CrispEmbed's
general license does not replace these OXR-specific terms.

## Model and preprocessing contract

The arithmetic follows the pinned `modeling_oxr.py`, checkpoint configuration,
and serving plugin. OXR has a custom visual-token pathway; loading it as an
ordinary Qwen vision-language checkpoint would omit essential operations.
The implementation is in [`src/oxr_ocr.cpp`](../src/oxr_ocr.cpp), with pure
image/request contracts in
[`src/oxr_model_profile.h`](../src/oxr_model_profile.h).

| Component | Native implementation |
| --- | --- |
| Image frontend | RGB uint8; alpha discarded; smart resize to a multiple of 32 with Python ties-to-even rounding; Pillow bilinear integer resampling; mean/std 0.5; merge-major `3 x 16 x 16` patches. |
| Vision encoder | 27 layers, width 1152, 16 heads, head dimension 72, MLP width 4304, LayerNorm epsilon `1e-6`. |
| Vision positions and attention | Learned 32 x 32 positions interpolated bicubically with `align_corners=False`, 2-D rotary positions, upstream window permutation and inverse permutation. This checkpoint uses full attention in every vision layer. |
| Vision activation/output | Explicit F32 tanh GELU graph matching `gelu_pytorch_tanh`; final hidden state is taken before the stored vision post-LayerNorm, as in OXR. |
| Visual vocabulary | 2 x 2 spatial merge to width 4608; projection to 65,532 logits; LayerNorm epsilon `1e-5`; softmax; four zero columns; weighted sum over the 65,536 x 1024 visual embedding table. |
| Image indicators | Start/end embeddings from visual-table rows 65,532 and 65,533, with the original image-placeholder sequence. |
| Language decoder | 28 Qwen3 layers, hidden width 1024, MLP width 3072, 16 query heads, 8 KV heads, head dimension 128, Q/K RMSNorm epsilon `1e-6`, half-split RoPE with theta `1e6`, SwiGLU. |
| Tokenizer and output | NFC plus the checkpoint BPE and all 26 added tokens; raw and special-token-filtered text; EOS IDs 151643 and 151645. |

The query projection width is 2048 even though the decoder hidden width is
1024. The soft visual vocabulary is evaluated in chunks of 256 merged tokens
to bound temporary matrices. A contiguous transposed visual embedding table
is prepared once at model initialization and reused; this costs an additional
256 MiB of F32 storage. Activations and recurrent K/V state remain F32 in both
supported storage recipes. These choices describe the implemented graph;
they are not a measured accuracy or performance result.

### Processor and generation profiles are explicit

| Profile | Minimum pixels | Maximum pixels | Repetition-penalty history |
| --- | ---: | ---: | --- |
| Public native default / published vLLM processor | 200704 | 921600 | Prompt and completion |
| Original HF `preprocess_inputs` defaults | 200704 | 2408448 | Completion only |
| Primary reference protocol in this repository | 200704 | 921600 | Completion only, explicitly selected to replay original HF generation |

The public C and Python APIs use the first row. The reference harness calls
the original HF model with the explicitly chosen pixel limits and uses
`--hf-generation 1` in the native runner. The low-level API in
[`src/oxr_ocr.h`](../src/oxr_ocr.h) exposes pixel-limit and HF-generation setters
for these replays. It also preserves the corresponding HF pad-mask/position
behavior. There is no implicit claim that the public native profile is
numerically identical to a vLLM BF16 deployment.

The history distinction matters when repetition penalty differs from 1.0:
the original HF model generates through `inputs_embeds` with `inputs=None`,
so its penalty sees generated IDs only. The vLLM service also includes prompt
IDs. Ordinary requests use penalty 1.0, but the original document helper may
retry capped text with penalty 1.1. Qualification must therefore include
recovery cases under the intended profile.

The chat template is the original user/image/assistant sequence. No system
message or thinking prefix is inserted. Supply the task prompt itself;
literal `<image>` and `<video>` placeholders are rejected because the runtime
constructs the image sequence. The processed `text` omits the 14 tokens marked
special by the checkpoint; `raw_text` preserves their decoded representation.

### Context and token budgets

The native total context is **32,768 tokens**, from the nested Qwen3
configuration. The outer OXR configuration's larger limit is not used to
extend the decoder. Prompt tokens, image tokens, and requested completion
must fit this total; an excessive explicit cap fails without silent clamping.

The low-level native default and public C API `max_new_tokens=0` mean
**2048 completion tokens**. Python `CrispOXR` instead explicitly defaults to
**8192**, matching the original document configuration. Passing 0 to an
individual Python `predict` request resets that request to the native 2048
default; the Python constructor's document budget must be positive. Reading
order uses the upstream smaller budget `min(document_budget, max(16, 4 * n))`
for `n` regions.

## Convert and build

Run model acquisition, full native builds, and model qualification on a
suitable build/validation machine. Use an isolated Python environment. The
converter needs no Torch installation and does not execute remote model code:

```bash
python -m pip install -r tools/oxr-conversion-requirements.txt
python models/convert-oxr-to-gguf.py \
  --model SoMarkAI/OXR-1.0 --dtype f32 --output oxr-f32.gguf
```

The model ID always resolves to the immutable revision above. To keep a local
checkpoint for both conversion and the original reference, download the
manifest's exact files and pass that directory:

```bash
python - <<'PY'
import json
from huggingface_hub import snapshot_download

manifest = json.load(open('models/oxr-source.json'))
snapshot_download(
    manifest['model_id'], revision=manifest['revision'],
    allow_patterns=list(manifest['files']), local_dir='checkpoint',
)
PY
python models/convert-oxr-to-gguf.py \
  --model checkpoint --dtype f32 --output oxr-f32.gguf
```

Choose one of these conversion routes; the converter refuses to overwrite an
existing GGUF or receipt. `--check-only` authenticates a checkpoint without
writing the requested output. Successful conversion writes a companion
`oxr-f32.gguf.json` with source/output hashes, exact tensor storage, and explicit
unverified quality/speed status.

`--dtype f32` is the primary numerical baseline. `--dtype f16` writes the
`f16-f32-sensitive` recipe: eligible matrices use F16 storage while embeddings,
visual-table weights, visual-head weights, output weights, norms, and vectors
remain F32. It is a separate candidate requiring its own quality and speed
evidence. No Q4 or Q8 OXR recipe is qualified or supplied.

For a Git checkout, initialize the pinned submodules first:

```bash
git submodule update --init --recursive
```

The accompanying source archive includes the pinned GGML source and does not
need that Git operation. Build the production interfaces and validation tools:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF
cmake --build build --target crispembed-cli crispembed-shared \
  test-oxr-profile test-oxr-api test-oxr-graph test-oxr-run test-oxr-diff -j2
```

Use CrispEmbed's existing CUDA, Vulkan, or Metal CMake options for a separate
backend build. GPU behavior and performance require independent validation.
The portable CPU build above is the primary qualification configuration.

## Recognition APIs

### Python image requests

Install the pinned helper environment for the complete examples below, then
activate native OXR:

```bash
python -m pip install -r tools/oxr-blueprint-requirements.txt
export PYTHONPATH=python
export CRISPEMBED_OXR=1
export CRISPEMBED_ACCEPT_OXR_LICENSE=1  # after reviewing licenses/oxr-LICENSE
export CRISPEMBED_OXR_FORCE_CPU=1
```

```python
from crispembed import CrispOXR

with CrispOXR('oxr-f32.gguf', n_threads=4,
              lib_path='build/libcrispembed.so') as model:
    result = model.predict('formula.png', task='formula', max_new_tokens=512)
    print(result['text'])
    print(result['eos'], result['capped'], result['finish_reason'])
```

Use the actual `.dylib` or `.dll` path on other platforms. Inputs may be local
image paths, Pillow images, or RGB uint8 NumPy arrays; grayscale and RGBA are
converted to RGB. OpenCV arrays normally require BGR-to-RGB conversion before
this API. A context manager frees the resident model. Calls on one Python
context are serialized.

| Python task | Exact upstream prompt |
| --- | --- |
| `text` | `Recognize the content of the image.` |
| `formula` | `Recognize the formula of the image into KaTex format.` |
| `table` | `Recognize the table of the image into html format.` |
| `code` | `Recognize the code of the image into markdown code format.` |
| `page` | `Recognize the whole page content of the image.` |
| `chemical_structure` | `Recognize the chemical structure of the image into SMILES.` |

An explicit `prompt=` takes precedence over `task` and is passed through
unchanged. `task='layout'` exposes the upstream diagnostic model prompt; it
does not replace the D-FINE detector in document processing. `predict` returns
cap information directly, so callers must check completion before treating
its output as complete.

### C API and existing CLI

The public declarations are in [`src/crispembed.h`](../src/crispembed.h).
This function illustrates ownership and error handling for a caller-provided,
tightly packed RGB image:

```c
#include "crispembed.h"
#include <stdint.h>
#include <stdio.h>

int recognize_oxr(const uint8_t *rgb, int width, int height) {
    crispembed_oxr_context *ctx = crispembed_oxr_init("oxr-f32.gguf", 4);
    if (!ctx) return 1;
    const char *json = crispembed_oxr_predict(
        ctx, rgb, width, height, 3,
        "Recognize the content of the image.", 2048, 1.0f);
    int status = json ? 0 : 1;
    if (json) puts(json);
    else fprintf(stderr, "%s\n", crispembed_oxr_last_error(ctx));
    crispembed_oxr_free(ctx);
    return status;
}
```

Set the environment gate before initialization. C prompts must be non-NULL
and at most 65,536 bytes. The C API accepts 1-, 3-, or 4-channel uint8 buffers;
text-only requests use NULL pixels and zero width, height, and channels.
Returned JSON and error strings belong to the context. Copy them before the
next call or destruction, and serialize all calls on a context. JSON includes
`text`, `raw_text`, input/prompt IDs, completion IDs, `grid_thw`, EOS/cap and
finish reason, requested/effective budgets, processor/history profile,
dimensions, confidence, and measured request time.

Recognition through the generic OCR interface:

```bash
CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1 CRISPEMBED_OXR_FORCE_CPU=1 ./build/crispembed \
  -m oxr-f32.gguf --ocr crop.png --ocr-max-tokens 512
```

Recognition through explicit orchestrator engine 26:

```bash
CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1 CRISPEMBED_OXR_FORCE_CPU=1 ./build/crispembed \
  --ocr-pipeline page.png --ocr-engine oxr \
  --ocr-rec oxr-f32.gguf --ocr-max-tokens 2048
```

That orchestrator stage rejects output that has not reached EOS. No published
model alias, automatic GGUF download, or standalone native D-FINE endpoint is
introduced. Rust exposes `Oxr::predict_rgb` and `predict_text`; Dart exposes
`CrispOXR.predictRgb` and `predictText`. Those wrappers have not yet been
compiled or exercised against the native library.

## Complete document processing

### Supplied regions

`process_pages` takes predecoded pages and one region list per page. Region
boxes use `[x1, y1, x2, y2]` in that page's actual pixel coordinates. Each
region has `type`, `bbox`, and optional `score` (default 1.0). The accepted
class names are `Caption`, `Footnote`, `Formula`, `Text`, `Page-footer`,
`Page-header`, `Picture`, `Title`, `Table`, `Code Block`, `Stamp`, and
`Chemical Structure`.

For a simple page known to contain only text, an explicitly supplied full-page
text region is sufficient to exercise the helper. Use actual detector output
for pages with multiple regions, tables, figures, or columns:

```python
import json
from pathlib import Path
from PIL import Image
from crispembed import CrispOXR

with Image.open('page.png') as opened:
    page = opened.convert('RGB')

regions = [[{
    'type': 'Text',
    'bbox': [0, 0, page.width, page.height],
    'score': 1.0,
}]]

with CrispOXR('oxr-f32.gguf', n_threads=4,
              lib_path='build/libcrispembed.so',
              max_new_tokens=8192) as model:
    document = model.process_pages(
        [page], layouts=regions, file_name='page.png',
        output_formats=('json', 'markdown'),
    )
    Path('page.md').write_text(document['outputs']['markdown'], encoding='utf-8')
    Path('page.json').write_text(
        json.dumps(document['outputs']['json'], ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(document['metadata']['crispembed'])
    print(model.last_trace)
```

`layout_response=` alternatively accepts the original structured D-FINE batch
response. `CrispOXR.layout_response_from_regions` constructs that response from
the convenient region lists. Both routes still run the original validation,
score filtering, clipping, and NMS. Supply either `layouts` or
`layout_response`, not both. With neither supplied, either `layout_model=`
(native D-FINE GGUF, see [oxr-dfine.md](oxr-dfine.md)) or an explicit
`layout_url` is required.

### Original external layout service and PDFs

The detector is an HGNetv2-B2/HybridEncoder/D-FINE model with 12 classes,
960-pixel input, and four decoder layers. It is now available natively as the
`dfine` architecture of `layout_detect` ([oxr-dfine.md](oxr-dfine.md)); the
heron RT-DETRv2 weights and class contract remain separate. The external
service route below is unchanged.

Deploy the original detector according to the
[pinned upstream instructions](https://github.com/SoMarkAI/OXR/blob/c1a1dcb312ee9f691404f127785510453c0d9350/README.md).
The optional `layout_analysis/last.pth` is recorded separately in the source
manifest and is not included by the recognition-only download command.
The supplied helper requirements do not install the detector's neural runtime.
Once its `/inference` service is available, configure it explicitly:

```python
from crispembed import CrispOXR

with CrispOXR('oxr-f32.gguf', n_threads=4,
              lib_path='build/libcrispembed.so',
              layout_url='http://127.0.0.1:8081',
              max_new_tokens=8192) as model:
    document = model.process_document('report.pdf')
    print(document['outputs']['markdown'])
```

The configured layout service receives page images. OXR recognition and
reading-order requests remain in the native context. `process_document`
accepts local PDF/image paths or bytes; bytes require `file_name=` so the
original decoder can select the format. Supplied PDF regions must refer to
the original helper's rendered page geometry. `aprocess_pages` and
`aprocess_document` are the corresponding methods for an active asyncio
event loop. They use a worker thread and preserve one-context serialization;
they do not provide vLLM continuous batching.

### Preserved helper behavior and deliberate completion policy

The adapter verifies the exact bytes of all 50 upstream Python files before
loading them in a private per-context namespace. Its source snapshot digest
is `232ac3f2cc0e195b07622cd770d406524f964de6bf6a7e1d5a5db92eb558332f`.
The installed package must match the pinned revision; `blueprint_path=` may
instead identify an exact checkout. Source drift is rejected. The original
package in the application's normal import namespace is not patched.

The private adapter replaces the recognition client and layout boundary while
preserving the original crop rounding, BGR helper boundary, class-to-prompt
routing, normalized reading-order prompt, integer-permutation parsing,
incomplete-order repair and XY-Cut++ fallback, formula numbering, chemical
equation labeling, code/formula/table postprocessing, table-image restoration,
deduplication, and final document serialization. Header/footer retention,
`element_formats`, `asset_dir`, and `asset_url_prefix` are forwarded to the
original helpers. `table_image_placeholders` and `full_overlap_dedupe` retain
their upstream default of false. The original table-image fallback can retain
a raster image when restoration fails.

Native generation is greedy (`temperature=0`, `top_p=1`); these are the
supported document model options along with `repetition_penalty`. The original
recovery remains: capped text-like requests retry with penalty at least 1.1,
and capped table requests retry after a 0.75 crop resize. After that recovery,
the adapter rejects logical requests still capped or lacking EOS by default.
Set `allow_truncated_content=True` only when explicitly accepting that
upstream incomplete-content behavior. Every native attempt remains in
`last_trace`, including prompt, RGB hash and geometry, budget, penalty, and
the complete native result. Document metadata records the layout source,
native request count, unresolved requests, and blueprint revision.

## Graph optimization gates

| Environment variable | Behavior |
| --- | --- |
| `CRISPEMBED_OXR=1` | Required activation gate for this experimental model. |
| `CRISPEMBED_ACCEPT_OXR_LICENSE=1` | Required acknowledgement of the OXR License's additional terms (USD 2M commercial thresholds, SoMark attribution for online services, no competing hosted API). Without it model load fails on every surface; `crispembed-server` also warns about sections 2 and 3 when an OXR stage is configured. |
| `CRISPEMBED_OXR_FORCE_CPU=1` | Force CPU for the primary reference/qualification profile. |
| `CRISPEMBED_OXR_GPU` | `1` forces the compiled GPU backend, `0` the CPU. Unset = auto: Metal for the qualified F16 recipe on Apple builds (see GPU section), CPU for every other recipe and platform. `CRISPEMBED_OXR_FORCE_CPU=1` always wins. |
| `CRISPEMBED_OXR_GPU_HP_MM=0` | GPU contexts set ggml's `GGML_METAL_PREC_F32_MM=1` (F32-tile `_hp` `mul_mm` for PREC_F32 products, process-wide, an explicit value wins); `0` keeps ggml's mat-vec PREC_F32 route. |
| `CRISPEMBED_OXR_GPU_FLASH=1` | GPU: backend-native flash attention (Metal stages Q as half) instead of the default F32 dense GPU attention. |
| `CRISPEMBED_OXR_GPU_VISION_FLASH` | GPU: `1`/`0` choose flash/dense for the vision tower only. |
| `CRISPEMBED_OXR_GPU_FLASH_CPU=1` | GPU: the original PREC_F32 flash node, which ggml-metal refuses and the scheduler runs on the CPU in every layer. |
| `CRISPEMBED_OXR_GPU_F16_HALF_ACT=1` | GPU: F16-weight products without PREC_F32 (half-staged activations). |
| `CRISPEMBED_OXR_GPU_SPLIT_F16=1` | GPU: F32/F16 products as hi/lo half-kernel terms (`core/split_f16_matmul.h`). |
| `CRISPEMBED_OXR_VTE_Q8=1` | Store the transposed visual table as Q8_0 (68 MiB instead of 256 MiB). |
| `CRISPEMBED_OXR_TIMING=1` | Per-stage wall times on stderr (vision+vocabulary, prefill, decode). |
| `CRISPEMBED_OXR_FLASH_ATTN=0` | Select dense attention instead of the default flash-attention path. |
| `CRISPEMBED_OXR_REUSE_DECODE_GRAPH=1` | Opt into decode graph reuse; default is off. |

The ordinary native path keeps weights resident, builds a composed vision and
prefill graph, uses an F32 KV cache, and computes single-token decoding against
that cache. The optional reusable decoder graph writes dynamic K/V rows and
updates positions/masks while grouping attention extents into 256-token
buckets. It rebuilds at bucket boundaries. Unused cache rows remain masked,
including on shorter subsequent requests.

Graph reuse is an optimization candidate until ordinary/reused output and
same-host speed comparisons pass. The weight-free GGML operator test covers
the contiguous visual projection, explicit F32 GELU, head-dimension-72
attention, dynamic cache writes, the 256 boundary, and a shorter subsequent
request. That test is prepared but has not yet been executed. Memory reuse or
fewer graph constructions alone do not establish throughput improvement.

## GPU: Metal (Apple M1), 1 November 2026

Receipts: [`tests/results/oxr/metal-m1/`](../tests/results/oxr/metal-m1/). Reference token IDs and
text are the original Transformers F32 outputs of the hosted lane (`hosted-41277219829/<case>/python.json`;
200704..921600 pixels, completion-only history, 1024-token cap). Model: the published
`cstr/OXR-1.0-GGUF` F16 recipe. Stage comparisons are against the native CPU run of the same file.

Two Metal defects were found and fixed (both behind gates, old paths kept):

1. **Attention ran on the CPU.** OXR tags every flash-attention node `GGML_PREC_F32`; ggml-metal
   refuses such a node (its tiles stage Q as half), so the scheduler copied every layer's
   attention to the CPU and back. GPU contexts now lower attention to F32 dense products on the
   GPU (query-blocked for 3600-patch pages, grouped KV by broadcast, no repeat copies).
   `CRISPEMBED_OXR_GPU_FLASH=1` selects Metal's native flash kernel instead (token-exact on 5/5 too,
   but up to 1.1 absolute difference in `vision.merge` vs CPU); `CRISPEMBED_OXR_GPU_FLASH_CPU=1`
   restores the original CPU-routed node.
2. **F32-precise products ran on mat-vec kernels.** `GGML_PREC_F32` products in this ggml fork go
   to `mul_mv_ext` or bespoke integer mat-vec kernels at any batch size: ~90 GFLOP/s on the M1 vs
   1100-1600 GFLOP/s for the existing F32-tile `_hp` `mul_mm` (`tools/metal/mm_bench.cpp`). The
   ggml patch `GGML_METAL_PREC_F32_MM` (submodule branch `crispembed/metal-prec-f32-mm`,
   `patches/ggml-metal-prec-f32-mm.patch`) routes PREC_F32 products with more than 8 rows to
   `_hp`. OXR GPU contexts enable it (`CRISPEMBED_OXR_GPU_HP_MM=0` opts out).

| F16, Metal arm | token IDs vs original (5 cases) | max abs vs CPU F16: `vision.merge` / `llm.norm` / logits | warm s: formula / japanese / table / receipt / german |
| --- | --- | --- | --- |
| CPU (4 threads) | exact 5/5 | — | 28.2 / 27.9 / 34.0 / 55.4 / 149.7 |
| dense F32 attention + `_hp` products (**default**) | exact 5/5 | <= 1.1e-2 / 6.5e-3 / 1.1e-4 (cosine 1.0000000) | 2.8 / 3.4 / 13.9 / 12.6 / 44.8 |
| + native flash (`CRISPEMBED_OXR_GPU_FLASH=1`) | exact 5/5 | <= 1.1 / 0.70 / 5.1e-3 | 2.3 / 3.1 / 11.3 / 11.4 / 24.2 |
| dense F32, mat-vec PREC_F32 route (`CRISPEMBED_OXR_GPU_HP_MM=0`) | exact 5/5 | <= 8e-3 / 4.5e-3 / 8.4e-5 | 12.7 / 12.4 / 31.0 / 34.1 / 115.8 |
| native flash + half-activation F16 products | exact 5/5 | <= 1.0 / 1.2 / 1.5e-2 | 5.1 / 5.7 / 17.2 / 14.8 / 41.3 |

The host was shared (load average 8-50, swap in use, model loads of 3-4 minutes from a contended
external disk), so timings are only indicative; CPU and the default Metal arm ran in the same
batch session. Per-stage traces (`CRISPEMBED_OXR_TIMING=1`) for formula on the default arm:
vision + visual vocabulary 1.7-1.8 s, prefill 0.23 s, decode ~42 ms/token.

**Default changed:** with `CRISPEMBED_OXR_GPU` unset, an Apple build loads the F16 recipe
(`f16-f32-sensitive`, no quantized tensors) on Metal with the default arm above. Every other recipe
and platform stays on the CPU unless `CRISPEMBED_OXR_GPU=1`. `CRISPEMBED_OXR_GPU=0` or
`CRISPEMBED_OXR_FORCE_CPU=1` keep the CPU. The F32 recipe was not run on Metal here (no local F32
conversion: the shared volume was below its free-space floor). CUDA (Kaggle T4, CPU-routed or CUDA
flash attention of the previous code) diverged on German at token 87; the new dense GPU attention
is the default for every GPU backend under `CRISPEMBED_OXR_GPU=1` but has not been re-run on CUDA.

## Validation evidence and remaining gates

The implementation session completed the following component controls:

The [implementation validation receipt](../tests/results/oxr/implementation-validation.json)
records the checks, source hashes, and unexecuted qualification gates.

| Control | Result and scope |
| --- | --- |
| Python binding and original helper integration | 16 tests passed with mocked native inference and the real pinned document helper. Includes RGB/BGR, source isolation, exact prompts, reading order, recovery, formula/table processing, JSON, and Markdown. |
| Converter and reference/comparator contracts | 28 tests passed. Includes source/schema/tensor checks, precision recipes, tokenizer fixtures, comparison rejection paths, and protocol controls. |
| Public C API | 30 checks passed using a mocked model backend: ownership, input/error controls, JSON diagnostics, and recovery from failed calls. |
| Native profile/preprocessing | 31 checks passed, covering pure image/request contracts and fixed fixtures without model weights. |
| C++ integration syntax | 11 translation units passed syntax compilation; this does not prove complete native linking. |
| Native GGML operator graph | Prepared as `test-oxr-graph`; not executed. |
| All 66 real model stages, output parity, and speed | Harness and hosted workflow prepared; not executed. |

These results are not neural-model quality evidence. A complete native
library/CLI build and link, checkpoint conversion/load, model inference,
representative document evaluation, GPU evaluation, Rust compilation, Dart
analysis, and hosted workflow dispatch have not been completed. No GitHub
push, pull request, or release is part of this source delivery.

After building in the pinned helper environment, repeat the component tests:

```bash
./build/test-oxr-profile
./build/test-oxr-api
./build/test-oxr-graph
PYTHONPATH=python python -m unittest discover -s tests -p 'test_oxr*.py' -v
```

### Freeze the original reference

Use a separate pinned reference environment, the authenticated `checkpoint`
directory, and the same machine/thread count for native comparisons. For CPU:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r tools/oxr-reference-requirements.txt
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
mkdir -p evidence

python tools/oxr_validate.py reference --model checkpoint \
  --image tests/regression/images/formula_quadratic.png --task formula \
  --min-pixels 200704 --max-pixels 921600 --repetition-penalty 1.0 \
  --dtype f32 --device cpu --attention sdpa \
  --threads 4 --repeat 3 --max-tokens 256 \
  --output evidence/python.json --dump-reference evidence/python.gguf
```

This runs the authenticated original custom model, including original
preprocessing and generation. It refuses to replace frozen reference files.
It records package/source hashes, request IDs, output IDs/text, image geometry,
completion state, cold/warm timing, and all 66 stage tensors. To investigate
the original HF pixel default, capture a separate reference with
`--max-pixels 2408448`; use that same limit in the native stage runner.
The CLI's SMILES task is named `chemical`; Python's binding calls it
`chemical_structure`.

Timed calls are ordinary unobserved inference. Source-default generation and
explicit cached generation must produce the same result. Transformers 5.13.1
forces cache use for the decoder-only `inputs_embeds` path even when the
stored generation configuration says `use_cache=False`; the report retains
that distinction. Confidence/stage capture occurs separately and must leave
the original output unchanged. Loading and image-file decode are excluded
from resident request timing; image preprocessing, vision, prefill, cached
greedy decode, transfers, and detokenization are included. Model-load time is
reported separately.

### Compare every native stage

```bash
CRISPEMBED_OXR=1 CRISPEMBED_ACCEPT_OXR_LICENSE=1 CRISPEMBED_OXR_FORCE_CPU=1 ./build/test-oxr-diff \
  --model oxr-f32.gguf --reference evidence/python.gguf \
  --image tests/regression/images/formula_quadratic.png \
  --prompt 'Recognize the formula of the image into KaTex format.' \
  --min-pixels 200704 --max-pixels 921600 --hf-generation 1 \
  --repetition-penalty 1.0 --threads 4 \
  --max-abs 0.002 --min-cos 0.99999 \
  > evidence/stages.json 2> evidence/stages.stderr.log
```

The mandatory inventory is `input.patches`, vision patch/position/embedding,
all 27 vision layers, merged features, visual logits/probabilities/embeddings,
LLM input, all 28 decoder layers, final norm, and final logits. Every stage
must have the exact shape, finite values, one expected observation, maximum
absolute error at most 0.002, and minimum row cosine at least 0.99999.
`input.patches` has the stricter absolute limit `2e-6`. Input IDs and image grid
must match exactly, as must final-logit argmax and the first generated token.

The comparator additionally runs ordinary and observed one-token inference
on the same native context and requires identical text, token IDs, confidence,
EOS, and cap state. Missing/extra stages, invalid shapes, nonfinite values,
observer effects, or a failed comparison produce a failing exit status. This
is an F32-to-F32 composed-prefill protocol; complete autoregressive output is
checked separately below. Repeat with `CRISPEMBED_OXR_FLASH_ATTN=0` and preserve
distinct stage reports for the dense path.

### Compare complete output and resident speed

```bash
python tools/oxr_validate.py compare \
  --reference evidence/python.json --native build/test-oxr-run \
  --gguf oxr-f32.gguf --image tests/regression/images/formula_quadratic.png \
  --max-slowdown 1.0 --output evidence/native.json

CRISPEMBED_OXR_REUSE_DECODE_GRAPH=1 python tools/oxr_validate.py compare \
  --reference evidence/python.json --native build/test-oxr-run \
  --gguf oxr-f32.gguf --image tests/regression/images/formula_quadratic.png \
  --max-slowdown 1.0 --output evidence/native-reuse.json
```

The comparator replays the frozen prompt, pixels, history semantics, penalty,
token budget, repeats, and thread count. Every repeat must preserve exact
request IDs, generated IDs, raw/processed text, grid, EOS/cap state, and
effective budget. An unfinished request fails parity. It retains native
stdout/stderr and full reports on failure. Add `--gold path/to/reference.txt`
for CER/WER against independent ground truth; without it, those text metrics
only measure agreement with Python.

The primary speed gate requires **CPU, the same host fingerprint, the same
thread count, an F32 Python reference, and the F32 native conversion recipe**.
It compares median warm request times after the first request. With
`--max-slowdown 1.0`, native time must be no greater than Python time, and the
complete request-parity gate must pass. A mixed F16 candidate or a BF16 Python
run cannot satisfy this primary same-precision speed gate. Observer-capture
times are not used as baseline timings.

A passing single image establishes only that case and configuration. Before
claiming parity, repeat frozen comparisons over text, formulas, multilingual
pages, code, chemical structures, tables, varied aspect ratios and pixel
budgets, long generation, recovery, and multi-page documents with the same
layout results. Include independent ground truth and compare final blocks,
reading order, table structure, and Markdown, as well as request output.
Public-profile recovery also needs a pinned vLLM comparison because its
penalty history differs from HF. GPU speed and vLLM batching/throughput are
separate, unmeasured claims.

### Hosted workflow

[`.github/workflows/oxr.yml`](../.github/workflows/oxr.yml) builds the component
and qualification tools. Its optional `workflow_dispatch` input
`model_check=true` acquires the pinned checkpoint, converts F32, freezes the
original reference, compares all stages, checks decoded output/resident
speed, and repeats dense attention and decode graph reuse controls. It retains
evidence even when a comparison fails. The model job has a preliminary
14 GiB RAM / 24 GiB free-disk guard; successful execution and actual peak
memory remain to be measured. The workflow has not been dispatched as part
of this implementation session.

The accompanying full source archive preserves the preceding MinerU
integration and includes this OXR implementation, pinned GGML source,
requirements, licenses, tests, reference tools, and this guide. Model weights
and compiled binaries are acquired/built separately using the commands above.
