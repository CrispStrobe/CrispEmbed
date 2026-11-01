# MinerU2.5-Pro-2604-1.2B in CrispEmbed

## Status and scope

This is an **experimental source implementation**, based on CrispEmbed
`15adb672f71f8470a23db1284dafa102515adae9`. Enable it explicitly with
`CRISPEMBED_MINERU25_PRO=1`; unset, empty, and literal `0` keep it disabled.
No model-backed quality or speed qualification is claimed by this delivery.
The model checkpoint was not downloaded or executed during implementation.
Weight-free tests and syntax checks are recorded separately in the delivery
[validation receipt](../tests/results/mineru25/implementation-validation.json).
The supplied hosted workflow has not been dispatched.

The integration contains native GGML model inference, an authenticated GGUF
converter, document interfaces, original-Python reference capture, a native
preprocessing comparator, a 64-stage graph comparator and a paired timing/output
driver. It does not publish a model alias, change an existing default, create a
release, or update an application runtime pin. Q8/Q4 recipes and quantized quality
are not qualified. CUDA, Metal, Vulkan, Windows and macOS model execution require
their own validation.

| Interface | Execution and document contract |
|---|---|
| `crispembed_mineru_predict`, Python `CrispMinerU.predict`, Rust `MinerU`, Dart `CrispMinerU` | Native GGML vision and language model; one image/region or a text-only prompt. Returns text, structural tokens, IDs, stop flags and timing. |
| Python `CrispMinerU.two_step_extract` / `batch_two_step_extract` | Native GGML inference plus the **unchanged pinned Python document helper** for layout preparation, crop geometry, table-image masking/restoration, formula repairs, image/chart processing and document postprocessing. No Python neural model is loaded. Pages/regions execute serially in one resident native context. |
| `crispembed_mineru_extract`, Python `native_extract`, orchestrator engine 25 / `mineru25-pro` | Pure C++ **basic** two-step page extraction. Includes layout resize/parsing, crops/rotation, recognition, basic OTSL-to-HTML and Markdown. It does not reproduce every Python document repair. It rejects tables containing images and directs callers to the full helper path. |
| `crispembed_mineru_extract_document` / `_last_document`, Python `native_two_step_extract` / `native_render`, Rust `MinerU::extract_document`, Dart `extractDocument`, engine 25 with `CRISPEMBED_MINERU25_PRO_FULL_DOC=1` | **Complete native document route** (no Python): every mineru-vl-utils branch translated to C++, built from the shared layout/router/document-model components. Default off; see [Full native document route](#full-native-document-route). |
| Generic `CrispOcrModel` / CLI `-m FILE --ocr IMAGE` | Existing Qwen2-VL dispatch recognizes the new GGUF profile; default task is **Text Recognition on that image**, not automatic page layout extraction. |

The native basic route must not be described as the complete Python document
pipeline. The full Python adapter intentionally reuses the original helper rather
than replacing its detailed table/formula rules with approximations. Optional
cross-page table merging additionally needs the original MinerU table-merge
package; absent helpers or native inference errors are reported, not ignored.

## Full native document route

`CRISPEMBED_MINERU25_PRO_FULL_DOC=1` (default off; unset, empty and `0` keep it
off) enables a pure C++ route whose output is designed to be byte-identical to
`CrispMinerU.two_step_extract` + `to_markdown` (pinned helper
`c9797cde8f54cc48306196fc6aebe73ea31dd4a3`) given the same recognition outputs.
The basic route (`crispembed_mineru_extract`) is unchanged and remains the
default for engine 25.

**Composition from shared components.** Step 1's VLM layout output (upstream
regex, exact `int()`/dedup/fsum/table-internal filtering) is a layout provider
emitting `layout_detect::region`s (pixel bbox, mapped heron label, score 1)
plus MinerU extras (raw type, angle, continuation, exact normalized bbox).
`ocr_region_router` plans destinations (table, formula, image-analysis
fallback, text); recognition runs per region with MinerU's prompts through an
injected predictor (native GGML in production, recorded outputs in tests).
The MinerU-specific steps are translated faithfully (`src/mineru_document.cpp`,
`src/mineru_postprocess.cpp`, `src/mineru_table_image.cpp`) and the result is
written into the `crisp_marker` document model. The **upstream profile**
(`mineru-json`, `mineru-markdown`) renders exact mineru-vl-utils JSON
(`json.dumps(..., ensure_ascii=False, separators=(",", ":"))`, including key
order, Python float repr and `null` content) and `json2md` Markdown from that
model; the same document also renders through the Marker `markdown`/`html`/
`json`/`chunks` renderers. Additive model change: `crisp_marker::block`
gained `std::map<std::string,std::string> attributes` (`mineru.*` keys).

**Helper branches ported.** Layout: parse/dedup/table-internal text filter,
internal image captions, image_block-covered visuals, image-analysis size
eligibility, skip lists and `not_extract_list`. Crops: Pillow ties-to-even
crop, 90/180/270 transpose, `resize_by_need` (aspect>50 white padding, min edge
28 bicubic). Tables: `build_table_image_map`, absorbed images,
`mask_and_encode_table_image` (average colour, inclusive rectangle, rotated
boxes, CPython `random` MT19937 tokens, Pillow-exact baseline JPEG + base64
data URIs), OTSL→HTML (docling), token restoration, `<eq>` wrapping. Image/
chart: tagged fields, chemical, flowchart (repetition detection, node-id
legalisation, invalid edge/subgraph repair, strict Mermaid), chart repetition
repair and malformed-table rejection, pure_table (Markdown/OTSL/HTML),
pure_formula, natural_image captions, `sub_type`. Text: display→inline,
macro spacing, underscores. Equations: delimiters, `\left/\right` repair
(array tagging, upstream quirks such as `\right.` matching any character),
double subscripts, eqqcolon, 313 `\big` substitutions (generated from the
pinned source), `<` spacing, unbalanced braces, brackets; equation_block
merging with `\tag` handling. `[Non-Text]` cleanup, list_item→text,
abandon_list/paratext, simple_post_process, and the helper's exception
fallback (e.g. an equation_block over unextracted equations returns the
mutated original list). Truncation follows `allow_truncated_content`.

**Exactness infrastructure.** `src/pyre.h` implements CPython `str`/`re`
semantics over code points (backtracking VM, explicit stack; Unicode
`\s\w\d\b`, `str.lower` incl. Final_Sigma, `splitlines`, `strip`), with
predicate tables generated from CPython (`tools/gen/gen_pyunicode_tables.py`;
the hosted lane regenerates them for its reference interpreter and records
whether they changed). Sources are compiled with `-ffp-contract=off`.

**Not identical by construction.** (1) Token *glyphs* drawn into table crops
that contain images: upstream uses Pillow/FreeType with the first host font
(Arial on macOS, DejaVu on Linux). The native route draws an embedded DejaVu
Sans atlas with the same size search and placement; masks, colours, tokens and
restored data URIs are exact, but those crops' pixels (and therefore possibly
the model's reading of the token) can differ. (2) Unseeded token RNG is OS
entropy on both sides (non-deterministic upstream too); pass
`table_token_seed` for reproducibility. (3) A JPEG `comment` carried in a
source image's PIL `info` is not propagated (raw pixels carry none).
(4) **Cross-page table merge is not ported**: upstream delegates to the
optional MinerU `table_merge` helpers (MinerU 4.1.3 → DocVortex, separate
additional-terms licence) that are not part of mineru-vl-utils; the native
route rejects `enable_cross_page_table_merge`. Use the Python
`batch_two_step_extract` for it.

**Evidence.** Weight-free, local (CPython 3.11.8, Pillow 12.3.0 /
libjpeg-turbo 3.1.4.1): 56,700 regex checks on every helper pattern; 66,000
function cases (22 functions) in `tools/mineru_doc/fuzz_postprocess.py`;
120 JPEG encodings byte-identical to Pillow; 617 OTSL rows
(`tests/data/otsl_profiles_golden.tsv`); 300 synthetic pages covering all
branches identical in blocks JSON, Markdown, document-model renders, every
region request and every non-glyph crop hash
([receipt](../tests/results/mineru25/native-doc-synthetic-parity.json)).
Model-backed (hosted run 41251475795, published F16 GGUF, CPython 3.12.15,
x86 CPU, `max_new_tokens=2048`, image analysis on): all **8/8** protocol cases
are byte-identical between `two_step_extract`+`to_markdown` and
`native_two_step_extract` in blocks JSON and Markdown, with identical
per-request prompts, crop sizes, token IDs, text and EOS (50 requests: 8 layout + 42 regions,
none capped); the recorded recognition also replays identically (8/8). The
weight-free gates passed on the same host with Unicode tables regenerated for
CPython 3.12. Route wall times are equal within noise (inference dominates).
No protocol case contains a table with embedded images, so the token-glyph
exception was not exercised by the model run
([receipt](../tests/results/mineru25/native-doc-model-parity.json)).
The gate stays default-off pending review; proposing default-on is now
supported by this evidence, keeping the basic and Python routes.

```bash
CRISPEMBED_MINERU25_PRO=1 CRISPEMBED_MINERU25_PRO_FULL_DOC=1 ./build/crispembed \
  --ocr-pipeline page.png --ocr-engine mineru25-pro --ocr-rec mineru25-pro-f16.gguf
```

```python
with CrispMinerU('mineru25-pro-f16.gguf', lib_path='build/libcrispembed.so') as m:
    blocks = m.native_two_step_extract('page.png', table_token_seed=0)
    print(m.native_render('mineru-markdown'))   # == to_markdown(two_step_extract(...))
    print(m.native_render('html'))              # Marker renderer, same document
```

### Shared OTSL→HTML

`src/otsl_table.h` is the single OTSL converter for MinerU, Marker and TeleOCR,
with explicit profiles: `upstream_docling` (exact mineru-vl-utils/docling,
used by the full route), `mineru_basic_v1` (moved verbatim from
`mineru_layout.h`; basic route output unchanged) and `marker_strict` (moved
verbatim from `marker_document.cpp`; validating, with warnings; TeleOCR and
Marker callers unchanged). `test-otsl-table` checks all three against outputs
captured from the pre-extraction implementations on 617 inputs; the profiles
disagree with upstream on 56 (basic) and 527 (marker) of them, which is why
no caller was switched.

## QuickGELU operator helper (from the OXR delivery)

The OXR delivery independently found the same F16-table QuickGELU rounding that
the default-on `CRISPEMBED_MINERU25_PRO_EXACT_GELU` gate removes. Its
`src/mineru_graph_ops.h` helper now owns the identical F32 expression
`x * sigmoid(1.702 * x)` behind that unchanged gate (`=0` still selects the table
path), and `test-mineru25-decode-graph` adds its 268-value operator control
against an independent double reference. The expression is unchanged from the
graph qualified on eight cases at `5c6005b9`; the control runs in hosted CI.

## Frozen sources

| Source | Immutable revision |
|---|---|
| [Model checkpoint](https://huggingface.co/opendatalab/MinerU2.5-Pro-2604-1.2B/tree/d3f5e08d073c21466bbabe21c71bb1e9c2e595da) | `d3f5e08d073c21466bbabe21c71bb1e9c2e595da` |
| [MinerU Python utilities](https://github.com/opendatalab/mineru-vl-utils/tree/c9797cde8f54cc48306196fc6aebe73ea31dd4a3) | `c9797cde8f54cc48306196fc6aebe73ea31dd4a3` |
| [Transformers model reference](https://github.com/huggingface/transformers/tree/v4.57.3/src/transformers/models/qwen2_vl) | `4.57.3`, `Qwen2VLImageProcessorFast`, model-card `use_fast=True` |
| GGML submodule | `a06fc0bb46b73cc10011fc90ae4054aee8438aeb` |

`models/mineru25-pro-source.json` authenticates the exact checkpoint/config/
tokenizer files with byte sizes and SHA-256 digests. The actual safetensors
header and tokenizer were inspected without downloading the 2,312,126,640-byte
weight payload. The converter rejects source drift, unexpected tensors, missing
weights, incorrect shapes or incompatible tokenizer/configuration contracts.
The Python document adapter verifies the pinned utility package's Python source
tree before using it. A different upstream version requires a new explicit audit.

The model and `mineru-vl-utils` declare Apache-2.0; the translated utility
algorithms retain the license and notice in `licenses/mineru-vl-utils-*`.
The optional full MinerU package is a separate dependency with its own
[additional-terms license](https://github.com/opendatalab/MinerU/blob/0ef2c20c4a56e68f81a66647f819d924ef2d89ce/LICENSE.md).
That optional package is not bundled or installed by the base requirements.

## Model and preprocessing fidelity

This checkpoint is a Qwen2-VL model. The native profile enforces its dimensions:

| Component | Contract |
|---|---|
| Vision | 32 layers, width 1280, 16 heads, 14×14 spatial patches, 2 temporal patches, 2×2 spatial merge |
| Vision activations | QuickGELU in vision blocks; exact erf GELU in the patch merger; LayerNorm epsilon `1e-6` |
| Language | 24 layers, width 896, intermediate width 4864, 14 query heads / 2 KV heads, head dimension 64 |
| Positions | Vision 2-D rotary positions; language MRoPE sections `[8,12,12]`, theta `1e6`; image-aware continuation positions |
| Image processor | Fast TorchVision uint8 antialiased bicubic resize, Python ties-to-even size rounding, CLIP normalization, merge-major patch order, temporal duplication |
| Pixel limits | 50,176–1,605,632; processor factor 28 |
| Context | Nested text configuration's 8192-token context; `max_new_tokens=0` uses the remaining budget |
| Decode | Greedy Transformers profile; no-repeat-100-gram over the prompt plus generated history; EOS IDs 151645 and 151643 |
| Output | BOS/EOS/PAD are removed for `text`; layout, box, rotation and OTSL tokens are preserved; `raw_text` and terminal EOS IDs remain available separately |

The supplied Python target is the model-card **fast** processor. The slow PIL
Qwen processor is a different numerical profile and is not silently substituted.
At the document-helper level, layout pages still use Pillow bicubic at
1036×1036, and crops use Pillow's ties-to-even coordinates, 90/180/270-degree
rotation, white padding beyond aspect ratio 50, and minimum edge 28. These are
distinct operations from the model's fast processor. RGBA conversion discards
alpha, matching the original helper's `convert('RGB')`.

The Transformers client does not apply the vLLM-specific presence/frequency
penalty pipeline. Consequently this implementation cannot claim vLLM throughput
or decoding parity merely because the same checkpoint is used. Quantized
weights, alternative processor versions and alternative generation backends
need separate named profiles and evaluation.

**Image decoding.** The native resize reproduces the pinned fast processor
pixel-exactly on x86 (hosted probe: up- and down-scaled real and random inputs,
`tools/ci-heavy/mineru25_resize_probe.py`). JPEG decoding is a separate input
step: the CLI/C helpers decode with `stb_image`, while the Python blueprint uses
PIL (libjpeg-turbo), and the two produce slightly different pixels (the table
fixture differed in 1,186 of 338,688 patch values, max 0.029 normalized). Pass
already-decoded RGB, or PNG, when exact agreement with a PIL-based pipeline
matters. Qualification feeds both sides one PIL-decoded PNG and records it.

**Vision activation precision.** ggml's CPU `gelu_quick` rounds its input
and output through an fp16 table, while the original evaluates
`x * sigmoid(1.702 x)` in F32. MinerU uses the exact F32 form by default
(`CRISPEMBED_MINERU25_PRO_EXACT_GELU=0` restores the table). Four hosted cases
decoded token-exactly either way; with the exact form the 64-stage errors fell
about 100x (vision layer 0 max error 1.8e-3 to 2.1e-5, logits 2.0e-3 to 8.1e-6).

**Stage budgets.** `test-mineru25-diff` reports two verdicts. The strict v1
budget (max abs <= 0.002, per-row cosine >= 0.99999) is unchanged and still
decides its exit code; it is below one float32 ulp for the decoder's massive
activations (~5e4), so no non-bit-identical F32 implementation passes it. The
reported v2 budget, chosen after inspecting those v1 failures, scales by each
stage's reference RMS: cosine >= 0.99999, relative RMS <= 1e-5 and max abs <=
max(0.002, 0.005 x reference RMS). Replayed on all retained reports it passes
every stage on the exact-GELU path and rejects the fp16-table path from vision
layer 0. Both verdicts are retained in every receipt.

## Conversion and build

Conversion uses NumPy, safetensors, Hugging Face Hub and gguf; it does not need
PyTorch. The pinned converter downloads the checkpoint if `--model` is omitted.
For a local checkpoint, all required files must match the source manifest.

```bash
python -m pip install -r tools/mineru25-conversion-requirements.txt
python models/convert-mineru25-pro-to-gguf.py \
  --output mineru25-pro-f32.gguf --dtype f32
python models/convert-mineru25-pro-to-gguf.py \
  --model checkpoint --output mineru25-pro-f16.gguf --dtype f16
```

F32 is the primary numerical baseline. F16 converts supported matrix weights
while retaining the recipe's sensitive vectors/embeddings in F32; inspect the
emitted `.gguf.json` conversion receipt for exact tensor storage. MinerU keeps
the recurrent K/V state in F32, including the F16 storage recipe, to avoid an
unrecorded recurrent-precision change. A smaller GGUF is not quality evidence.

On an appropriate build/validation machine:

```bash
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=OFF -DCRISPEMBED_NATIVE=OFF
cmake --build build --target crispembed-cli crispembed-shared \
  test-mineru-layout test-mineru25-profile test-mineru25-decode-graph \
  test-mineru25-run test-mineru25-preprocess test-mineru25-diff -j2
```

The source archive includes the pinned GGML source, so an unpacked archive
does not require a Git submodule operation. Use the existing CrispEmbed backend
build options for a separately validated GPU build. The implementation session
did not perform a complete native library build or GPU run.

## Use the complete document helper with native inference

```bash
python -m pip install -r tools/mineru25-blueprint-requirements.txt
export CRISPEMBED_MINERU25_PRO=1
export QWEN2VL_OCR_FORCE_CPU=1
export PYTHONPATH=python
```

```python
from crispembed import CrispMinerU

with CrispMinerU('mineru25-pro-f32.gguf', n_threads=4,
                lib_path='build/libcrispembed.so', image_analysis=True) as model:
    blocks = model.two_step_extract('page.png')
    print(model.to_markdown(blocks))

    # Includes raw text, IDs, EOS/cap, input geometry and measured request time.
    formula = model.predict('formula.png', '\nFormula Recognition:', 512)
    print(formula['text'], formula['eos'], formula['capped'])
```

Use the platform's actual shared-library path (`.dylib`/`.dll` where applicable).
One context owns one resident model and serializes requests. Context-manager
exit releases it. `batch_two_step_extract` preserves page order and reuses
weights; it is not continuous batching. Page methods reject incomplete generation
by default. `allow_truncated_content=True` explicitly permits the original
helper's truncated-content behavior while retaining cap information in
`last_trace`. No output truncation is treated as a quality pass. Explicit token requests that
would exceed the supported 8192-token total context are rejected rather than
silently shortened. The upstream Python implementation can accept larger explicit
requests; that extended-context behavior is outside this bounded native profile.

Cross-page cell merging is optional and requires
`enable_cross_page_table_merge=True` plus the official MinerU table-merge
helpers. The inspected optional source was MinerU 4.1.3 at
`0ef2c20c4a56e68f81a66647f819d924ef2d89ce`, which delegates to DocVortex.
This optional integration has only dependency/error controls, not a successful
model-backed cross-page demonstration.

For the native basic page route:

```bash
CRISPEMBED_MINERU25_PRO=1 ./build/crispembed \
  --ocr-pipeline page.png --ocr-engine mineru25-pro \
  --ocr-rec mineru25-pro-f32.gguf --ocr-max-tokens 2048
```

For a single text crop through the existing generic OCR interface:

```bash
CRISPEMBED_MINERU25_PRO=1 ./build/crispembed \
  -m mineru25-pro-f32.gguf --ocr crop.png --ocr-max-tokens 512
```

The same explicit engine is wired into the existing HTTP orchestrator. Engine
24 is appended to the C stage map; existing IDs are unchanged. An explicit
local GGUF is required. No download URL or checksum is invented for an
unpublished model alias. Rust exposes `MinerU::predict_rgb`, `predict_text` and
`extract_native_basic`; Dart exposes the corresponding `CrispMinerU` methods.

## Graph optimization and its acceptance gate

The shared graph uses GGML matrix operations, attention, rotary positions and
backend-resident K/V storage. MinerU adds correct profile-specific arithmetic
without replacing other Qwen model defaults.

`CRISPEMBED_MINERU25_PRO_REUSE_DECODE_GRAPH=1` selects an additional experimental
decode graph that uses dynamic `ggml_set_rows` cache updates and reuses its
allocation between tokens. It buckets the attention extent in 256-slot increments
instead of scanning the full 8192-token allocation for every short request.
The ordinary graph remains available for A/B comparison. This optimization is
default-off because its full-model quality and latency have not been measured.
The synthetic two-layer test covers repeat calls and the 256-slot boundary;
its source and syntax were checked locally, but executing the linked test is a
hosted build gate.

Do not turn the optimization on by default based on implementation plausibility,
component results or a single short page. Require decoded tokens/text, stop
conditions, stage errors and same-host timing to pass together.

## Reproduce qualification

`.github/workflows/mineru25-pro.yml` provides weight-free build controls on
matching pull requests and an explicit `workflow_dispatch` full-model F32 CPU
smoke. The model job requires at least 14 GiB RAM and 20 GiB free disk and retains
raw reports, source/package provenance and original reference tensors even when
a comparison fails. It is a smoke gate, not a representative document benchmark.

On a suitable authorized machine, first freeze the original reference:

```bash
python -m pip install -r tools/mineru25-reference-requirements.txt
git clone https://github.com/opendatalab/mineru-vl-utils.git blueprint
git -C blueprint checkout c9797cde8f54cc48306196fc6aebe73ea31dd4a3
python tools/mineru25_validate.py reference \
  --model checkpoint --blueprint blueprint --image formula.png --task formula \
  --dtype f32 --device cpu --threads 4 --repeat 3 --max-tokens 512 \
  --output evidence/python.json --dump-reference evidence/python.gguf

export CRISPEMBED_MINERU25_PRO=1
export QWEN2VL_OCR_FORCE_CPU=1
./build/test-mineru25-preprocess --image formula.png \
  --reference evidence/python.gguf --task formula
./build/test-mineru25-diff mineru25-pro-f32.gguf evidence/python.gguf 4 0.002 0.99999
python tools/mineru25_validate.py compare --reference evidence/python.json \
  --native build/test-mineru25-run --gguf mineru25-pro-f32.gguf \
  --image formula.png --gold formula.txt --max-slowdown 1.0 \
  --output evidence/native.json
```

Repeat the native comparison with the reuse-graph gate set to `1`; preserve both
receipts. `--gold` is optional but is needed for ground-truth CER/WER rather than
only distance from the Python output. Numerical failures exit nonzero. The
preprocessing test checks actual decoded image bytes through the native fast
processor against the pinned F32 reference; the stage test deliberately feeds
the original patches to isolate graph errors. It composes native vision output
into the language model, requires all 64 declared stages, rejects missing or
nonfinite observations, and checks last-logit argmax. Default stage gates require
both maximum absolute error ≤0.002 and per-row cosine ≥0.99999. These are proposed
acceptance budgets, not measured achievements; do not relax them after seeing a
failure without a separately reviewed protocol.

The timing driver runs ordinary original inference without observers, records
cold and resident warm runs separately, and performs untimed observer-neutrality
checks for IDs and intermediate captures. Native timings include helper resize,
model preprocessing, inference and result production on every repetition.
The comparator requires the same source image, task, prompt, expanded input IDs,
completion IDs, text, EOS/cap, hardware identity and thread count before it can
pass the CPU speed gate. It preserves every resident output to catch stale state
or drift. A ratio ≤1.0 means the native warm median was no slower for that exact
request; it does not establish aggregate throughput or another backend's speed.

Before claiming general parity, freeze an independent corpus covering layouts,
tables (including spans/images), dense formulas, multilingual text, rotated
crops, continuation blocks, long outputs, repeated requests and optional image
analysis. Compare full-helper document JSON/Markdown as well as raw model text,
measure CER/WER against actual gold, use suitable table/formula task metrics,
and retain every failure. Evaluate production dtype and GPU/CPU hardware
separately. The full-model corpus, numerical, speed, cross-platform and
quantization gates remain open in this source delivery.
