# Xiaomi-OCR-0 experimental native profile

This lane adds the exact `xiaomi-research/Xiaomi-OCR-0` checkpoint at
`bbd8cf8d21f256b3644b44cb860f3a7cecb93c30` to the existing small Qwen3.5
CPU implementation. It is opt-in and has no registry/download alias, published
GGUF, benchmark result, precision qualification, or change to existing models.
The [source contract](../tests/results/qwen35-ocr/xiaomi0-source-contract.json)
records the original metadata, tensor header, framework sources, task prompts,
and layout dependency. No original/native model inference was performed while
preparing this source-only change.

## Initial hosted metadata failure — 4 November 2026

[Run 41810171225](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41810171225)
at `2f1bbdb11c3612acf7399e31b33ac715027f3184` installed the exact dependencies
and passed `pip check`, then stopped at the image-processor metadata guard.
It ran without `--inference`; no checkpoint, native build, or model forward was
reached. Its original worker log and result remain negative evidence.

The initial artifact does not retain the processor's individual attributes.
Review and bounded stdlib execution of the pinned Transformers 5.17.0
[`BaseImageProcessor._standardize_kwargs`](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/image_processing_utils.py)
establish a deterministic guard defect: it converts the checkpoint's channel
mean/std lists into tuples, while the guard required Python list equality.
Both numeric sequences remain exactly `(0.5, 0.5, 0.5)`; the original check
rejects them solely because tuple and list equality is false. The upstream
`SizeDict` explicitly supports dictionary equality and is not that defect.

The correction compares the complete mean/std sequences by value, retains their
runtime container types, and preserves exact geometry, patch sizes, resize,
normalization and RGB settings. It additionally checks the original `1/255`
rescaling factor, requires real boolean processing flags, and pins the loaded
base-processor source bytes. A failure reports every differing field with its
expected and observed values; successful metadata records those settings and
the verified framework hashes. Three new model-free controls cover tuple/list
equivalence, 18 distinct incorrect settings, and retained failure details;
all 29 Xiaomi controls pass locally.

This is a source correction, not a successful hosted metadata or frontend
receipt. Rerun metadata on the reviewed correction source first. The frozen
two-image inference protocol and the independent document experiment remain
subsequent gates.

## Exact source and reuse boundary

The [official checkpoint](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0/tree/bbd8cf8d21f256b3644b44cb860f3a7cecb93c30)
contains 488 BF16 tensors: 473 main-model tensors and 15 MTP tensors. The main
model has the existing 12-layer vision tower and 24-layer hybrid Qwen3.5 decoder:
1024 text width, 768 vision width, 248320 vocabulary entries, three linear
attention layers followed by full attention, and tied input/output embeddings.
The pinned Transformers 5.17.0
[original implementation](https://github.com/huggingface/transformers/blob/v5.17.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
explicitly ignores `^mtp.*` while loading ordinary generation. The converter
checks all 15 exact names and shapes, verifies the complete checkpoint hash,
records every omitted name, then checks the remaining 473 tensors against the
original model instantiated on the meta device. Unknown names remain errors.
This does not implement speculative decoding.

The source model card declares Apache-2.0 and names
[Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68)
as its base. That exact base revision includes an Apache-2.0 license. The Xiaomi
model repository has no separate LICENSE file; the audited
[upstream code repository](https://github.com/SeerRay-Lab/Xiaomi-OCR-0/tree/b1a9bb4c07060c453785e9151715ccda6b96f0b1)
contains an Apache-2.0 license. Source fixtures retain both licenses.

The source has neither `generation_config.json` nor
`preprocessor_config.json`. Their absence is preserved. The nested image
processor is in `processor_config.json`. Source analysis of
`GenerationConfig.from_model_config` selects the non-default outer EOS 248046,
not the nested text EOS 248044. Hosted metadata validation must reproduce that
resolution before export. It also checks the effective tokenizer: the serialized
tokenizer and Qwen2Tokenizer class use the plain-letter split regex, while
`tokenizer_config.json` declares a combining-mark variant. A mismatch rejects;
the converter does not silently rewrite either source file.

## Native contract

The export uses architecture `qwen35_ocr`, format 4, profile `xiaomi-ocr-0`.
Admission requires all three gates:

- `CRISPEMBED_QWEN35_OCR=1`
- `CRISPEMBED_XIAOMI_OCR0_FRONTEND=1`
- `CRISPEMBED_XIAOMI_OCR0_INFERENCE=1`

`0` disables each gate. F32 and F16 tensors are supported in this experimental
lane; quantized Xiaomi tensors are rejected. The four exact original task prompts
are accepted by `qwen35_ocr_recognize_file` / `qwen35_ocr_recognize_raw`:

| Task | Prompt | Raw result |
|---|---|---|
| Text | `Extract the text in the image.` | Text |
| Table | `Parse the table in the image into OTSL.` | OTSL |
| Formula | `Identify the formula in the image and represent it using LATEX format.` | LaTeX |
| Document | The exact long document prompt in the source contract; an empty caller prompt selects it | Markdown with OTSL tables and LaTeX formulas |

The native message is image first, then task text, with the original non-thinking
assistant prefix. Only EOS 248046 stops this HF profile. The context uses the
existing 4096 total-token limit and at most 1024 vision patches; the default
output cap is 4096, but total context can stop generation sooner. The original
processor permits much larger images. Native inputs beyond the implemented
patch bound reject rather than receiving an implicit smaller resolution. These
limits are material for whole documents and require separate measured expansion.

Existing LightOn and Ovis profile admission, numerical operations, stop policies,
and qualification records are preserved. The two diagnostic callbacks accept
read-only actual Xiaomi logits and prefill stages. They default to null and
cannot be attached to another profile; no reference tensor is passed into the
production inference path.

The generic C API already dispatches this architecture to the exact-profile
loader, so a locally converted file is accessible through
`crispembed_ocr_model_init`, Python `CrispOcrModel` / `CrispMathOcr`, and
`crispembed -m FILE --ocr IMAGE` with the three gates. Resetting the generic token
cap selects Xiaomi's 4096 default; its prompt setter accepts only the four exact
tasks. Named orchestrator `qwen35-ocr` remains the format-1 selector and rejects
format 4. There is no Xiaomi model alias. Generic text/confidence access does not
expose native token IDs or stop reasons; the diagnostic runner supplies those
fields for the separately declared evidence protocols.

## The two-stage document dependency

The original document pipeline uses **PP-DocLayoutV3**, region crops, task-specific
recognition, and page assembly. It is not implemented by registering Xiaomi as a
whole-page OCR engine. The source audit pins the actual Transformers detector:
`PaddlePaddle/PP-DocLayoutV3_safetensors` at
`e4f489dc536556fc1be02b973ba06756e0d4a2ac`, including its config and
133270468-byte checkpoint SHA-256. No native PP-DocLayoutV3 graph is added here.
An existing generic layout detector is not accepted as this dependency.

`tools/xiaomi0_layout_contract.py` is the bounded, hosted **original layout**
producer. It verifies the actual PP checkpoint, imports the pinned upstream
layout postprocessor, executes `PPDocLayoutV3ForObjectDetection`, and preserves
region order, labels, normalized boxes, skipped visual regions, and exact region
prompts. More than the requested region budget rejects instead of truncating.
It returns normalized `[0,1]` boxes for evaluation and preserves original
`[0,1000]` boxes for cropping. It requires a local checkout of the exact upstream
source revision and the exact layout checkpoint directory.

The upstream vLLM region request places text before the image and defaults to
16384 output tokens. The HF profile here places the image first and has the
native limits above. The contracts record that difference. Neither a successful
region run nor this external layout producer reproduces the upstream end-to-end
benchmark. Native layout, polygon/PaddleX variants, upstream page assembly, full-document
scoring, and vLLM parity remain separate work.

## Hosted verification recipe

The original README pins Transformers 5.17.0, torch 2.14.0, and torchvision
0.29.0. The hosted recipe additionally requires `huggingface_hub`, `safetensors`,
`gguf`, `numpy`, `pillow`, and the repository's build dependencies. Layout reference
production additionally needs the pinned upstream Python dependencies. The
runner creates its own isolated environment and installs exact direct dependency
pins before metadata work. Resolved transitive packages are retained in its
package inventory. The initial recipe requires Linux x86_64 / Python 3.12.

```bash
gh workflow run heavy-cpu.yml --ref feat/ocr-benchmark-integration \
  -f script=tools/ci-heavy/xiaomi0_reference.py -f build_mode=portable \
  -f isolated_lane=xiaomi0-metadata
gh workflow run heavy-cpu.yml --ref feat/ocr-benchmark-integration \
  -f script=tools/ci-heavy/xiaomi0_reference.py \
  -f args=--inference -f build_mode=portable -f timeout_minutes=180 \
  -f isolated_lane=xiaomi0-inference
```

No extra workflow pip input is needed. Results use `HEAVY_OUT/xiaomi0`;
models, all GGUF files, caches, and builds use the disjoint
`HEAVY_SCRATCH/xiaomi0`. Explicit `--output` / `--scratch` overrides require fresh
directories. RAM, swap, load, and free disk are recorded before dependency/model
work; metadata requires at least 3 GiB available RAM / 8 GiB scratch, and inference
requires 9 GiB / 12 GiB, with 2 GiB output space. A shared scratch/output volume
must have their combined reserve (10 GiB for metadata or 14 GiB for inference).
Commands have time limits and child process cleanup. Model files are not uploaded.

Both commands require hosted CI. They first verify source files, loaded framework
sources, actual generation defaults, prompts, tokenizer, meta tensor schema, and
original/native image preprocessing. The second then executes the frozen
[two-image protocol](../tests/fixtures/xiaomi0-bounded/protocol.json): exactly two
original F32 cases and the same two native cases in F32 and F16, each capped at
16 generated tokens. It uses portable CPU builds with both native compiler flags
disabled. No quantization is attempted.

Every one of the original model's 474 parameter shapes, including the tied head,
is compared with the audited checkpoint header. The recipe also emits a native
build receipt with actual portable CMake flags and executable hashes. Its
generic public-admission probe checks gates, task setters, and cap resets without
adding model forwards to the frozen generation budget.

The result retains every generated logit row, 40 prefill stage captures, IDs,
text, confidence, stop reason, cap, source and image hashes. It reports the
owner's F32 `>0.99` / F16 `>0.98` final-logit bars beside strict `0.99999` / `0.002`
stage diagnostics and decoded agreement. Different generation prefixes are
recorded explicitly. The small protocol remains an integration experiment even
if every measured comparison passes. It does not change old failure evidence or
establish benchmark, full-document, or publication qualification.

## Strict tables and bounded region assembly

`tools/xiaomi0_document.py` implements the six original OTSL tags against the
pinned upstream `postprocess/otsl.py` blueprint. Twelve valid-table fixtures were
produced by executing that original pure-Python converter. The strict adapter
rejects missing origins, conflicting/nonrectangular spans, ragged rows, text on
continuation tags, and over-budget input. It escapes literal cell content and
emits real HTML tables with row/column spans. It does not infer table headers,
repair malformed spans, or convert Markdown pipes into tables.

Its model-free `assemble` mode consumes the original layout receipt, a frozen
`xiaomi0-region-protocol/v1`, and actual native region captures. The protocol
records the canonical layout digest, rendered input hash/dimensions/recipe, three
task caps, total budget, precision, native runtime revision, GGUF hash, and runner
hash. Initial bounds are 16 regions, 256 tokens per call, and 2048 tokens in total;
larger cases need a separately declared experiment.

The hosted `recognize` mode additionally takes the page image, local GGUF,
`test-xiaomi0-run` executable, and its `native-build.json` receipt. It validates
artifact hashes and the exact export profile, preserves original crops, and
records every actual ID/cap/EOS/confidence. Failed or malformed tables leave the
raw captures intact and fail assembly. It uses the diagnostic runner because the
generic OCR C API has no token/stop introspection.

Output follows `crispembed-ocr-document-v1`: ordered normalized blocks, HTML
tables, literal formulas, and every expected generation call. All detected
regions retain their order, including explicitly skipped visuals. This bounded
assembly does not implement the upstream page-assembly heuristics, its vLLM
request order, or a native PP-DocLayoutV3 graph. It remains unqualified until a
separate frozen original/native document experiment is actually executed.

Local checks use no framework, model weights, native build, or inference:

```bash
python -m unittest discover -s tests -p 'test_xiaomi0_*.py' -v
python tools/qwen35_source_pins.py --check
```
