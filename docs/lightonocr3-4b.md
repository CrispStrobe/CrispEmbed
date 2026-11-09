# LightOnOCR-3-4B — pinned source and explicit export path

Updated 4 November 2026. Implementation lane:
[`feat/ocr-next-models-20261104` / PR143](https://github.com/CrispStrobe/CrispEmbed/pull/143).

**Status: explicit conversion and grouped-head component APIs are implemented; full 4B inference and export remain unqualified.**
The source preflight verifies the original metadata, actual safetensors header
and bounded tokenizer-prefix evidence without loading weights. A separate,
explicit converter now requires the full pinned tokenizer and checkpoint bytes
plus the original meta-model schema before writing format 5. Its execution,
GGUF readback, native inference and document quality remain open. The additive
default-off grouped-head component and original/meta hosted protocols are now
registered, with actual numerical results still pending. PR141's
LightOn1B/0.8B evidence is merged and preserved.

This is the next new checkpoint in the [OCR expansion roadmap](ocr-benchmark-expansion.md).
Its upstream difficult-text results motivate evaluation; no benchmark number is a
CrispEmbed measurement or a guarantee that the larger model improves every task.

## Immutable sources and acquired evidence

| Material | Pin and scope |
|---|---|
| Model | [`lightonai/LightOnOCR-3-4B`](https://huggingface.co/lightonai/LightOnOCR-3-4B/tree/b71010f095dc3735de7ee6b9969bff8f8f50b39a), revision `b71010f095dc3735de7ee6b9969bff8f8f50b39a` |
| Original files | Exact config, generation config, processor/preprocessor config, tokenizer config, chat template and two model cards are preserved in the [source fixture directory](../tests/fixtures/lightonocr3-4b-source/) |
| Safetensors header | Actual range-acquired header, 89,456 JSON bytes including padding; SHA-256 `a1e36638b195f2d5943e000fcb069b275cb22ccc28b1b6184ddee7bdea7ef081` |
| Checkpoint declaration | One `model.safetensors`, 9,078,620,536 bytes; HF-declared SHA-256 `6b0dab74a2dab02d5b10ca56322255024ed8dca35c02746b42e9f296aaeb4e67`. The payload was not acquired or independently hashed |
| Original capture | [audit_manifest.json](../tests/fixtures/lightonocr3-4b-source/audit_manifest.json), SHA-256 `d7a6fc3398a35004e81271bd08ea36f0b7dce6b1285f92c1bb237cd799005a09`; it retains original small-file hashes, range provenance and the captured HF API response |
| Tokenizer comparison | [Separate prefix audit](../tests/fixtures/lightonocr3-4b-source/tokenizer_prefix_audit.json), using exact 65,536-byte HTTP 206 prefixes from 4B and 0.8B; neither prefix is a complete tokenizer JSON document |
| Original implementation | [`lightonai/LightOnOCR` at `36755d4`](https://github.com/lightonai/LightOnOCR/tree/36755d461be079737860a5f03ae0c803501269e9); the model config declares Transformers `5.5.4`, which is not an observed loaded runtime |
| Native source review | CrispEmbed main [`2f748825`](https://github.com/CrispStrobe/CrispEmbed/tree/2f748825ff3b56b8f2b89d156986b94218ae28c3) |

The pinned model card declares Apache-2.0 for the model. Its checkpoint tree has
no standalone LICENSE file. The fixture directory therefore preserves the exact
Apache license from the separately pinned official implementation, with that
distinction recorded in [NOTICE.md](../tests/fixtures/lightonocr3-4b-source/NOTICE.md).
The API response is a captured observation; dynamic repository statistics need
not be byte-identical on a later query even at the same checkpoint revision.

The full tokenizer is declared as 19,989,343 bytes with SHA-256
`87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4`.
That declaration is not full-file verification. The default preflight requires
no full tokenizer or checkpoint download.

## Actual checkpoint contract

The schema is derived from the pinned config independently of the captured
header, then compared against every tensor name, shape, BF16 dtype and byte
interval. Missing or extra tensors, changed shapes, duplicate JSON keys,
overlapping/gapped offsets and changed original bytes fail closed.

| Component | Actual source contract |
|---|---|
| Architecture | `Qwen3_5ForConditionalGeneration`, dense text model |
| Text | 32 layers, hidden width 2,560, FFN width 9,216 |
| Layer schedule | 24 linear-attention layers and eight full-attention layers; every fourth layer is full attention |
| Full attention | 16 query heads, four KV heads, head width 256; gated query projection `[8192, 2560]` |
| Linear attention | 16 key heads, 32 value heads, key/value width 128, convolution width four; QKV `[8192, 2560]`, Z `[4096, 2560]` |
| Vision | 24 layers, width 1,024, 16 heads, FFN 4,096; learned positions `[2304, 1024]`; no DeepStack |
| Patches and merger | Spatial patch 16, temporal patch two, spatial merge two; merger FC1 `[4096, 4096]`, FC2 `[2560, 4096]` |
| Vocabulary | 248,320 entries; embedding `[248320, 2560]` is tied to the output head and stored once |
| RoPE | Partial factor 0.25, interleaved sections `[11, 11, 10]`, theta 10,000,000 |
| MTP | Zero layers and no MTP tensors |
| Census | 723 BF16 tensors; 4,539,265,536 parameters including the tied embedding once |

The census decomposes as `24*14 + 8*11 + 2 + 24*12 + 9 = 723`.
Tensor intervals cover 9,078,531,072 payload bytes without gaps. Adding the
89,456-byte header and its eight-byte length prefix matches the declared file
size exactly. This proves metadata/header consistency, not the payload contents
or execution fidelity.

## Frontend and generation distinctions

The processor uses RGB, rescaling by 1/255, mean/std 0.5, bicubic resampling,
minimum 65,536 and maximum 16,777,216 pixels. The source is a
`Qwen3VLProcessor` with the recorded Qwen2-VL image-processor configuration.
The 4B/0.8B processor files are byte-identical in the audited sources; that alone
does not establish tokenization or native image parity.

The captured tokenizer prefixes contain NFC normalization and the same 33
added-token definitions. Their pretokenizer and decoder differ:

| Field | 4B | 0.8B at `36d8636925327c671bd8e479c517a004f7566bd9` |
|---|---|---|
| Pretokenizer regex | Includes combining marks through `\p{M}` | Serialized tokenizer lacks these additions |
| Pretokenizer ByteLevel `trim_offsets` | `false` | `true` |
| Decoder ByteLevel flags | `add_prefix_space`, `trim_offsets`, `use_regex` all `false` | All three `true` |

The 0.8B tokenizer config's regex declaration is not a substitute for its actual
serialized tokenizer. Preserve the exact 4B identity and compare real token IDs
on combining marks, multilingual text, whitespace and special-token boundaries.
The bounded prefix evidence establishes these metadata differences; tokenizer
execution parity and the complete vocabulary/merge payload remain separate gates.

The template supports plain OCR with one image and empty text, or the literal
`grounding` task. Preserve the exact template and assistant prefix
`<think>\n\n</think>\n\n`; do not introduce an extra newline before grounding
content. Grounding coordinates use the declared normalized 0–1000 convention.

Top-level config and `generation_config.json` use EOS **248046**, while the
nested text config declares **248044**. The preflight exposes both and does not
silently substitute one for the other. Generation config declares cache use and
pad **248044**; it does not declare temperature, `top_p`, `do_sample` or a
generation token cap. A future original-run protocol must state those choices.

The pinned [model card](https://huggingface.co/lightonai/LightOnOCR-3-4B/blob/b71010f095dc3735de7ee6b9969bff8f8f50b39a/README.md)
and [official client](https://github.com/lightonai/LightOnOCR/blob/36755d461be079737860a5f03ae0c803501269e9/lightonocr/client.py)
have different example sampling/rendering choices. Record the selected recipe
explicitly; do not blend card examples, client defaults and benchmark adapters
into an allegedly identical reference run.

## Executable preflight

From the repository root, using only the Python standard library:

```bash
python models/lightonocr3_4b_profile.py
```

For a separately prepared directory containing the exact six small metadata
files and the extracted original `model.safetensors.header.json`:

```bash
python models/lightonocr3_4b_profile.py --metadata-dir /path/to/pinned-metadata
```

The original fixture manifests and bounded tokenizer evidence remain the trusted
capture in the repository. `--model` and `--revision` must match the immutable
source; `--revision main` is rejected before model-directory access.
`--verify-tokenizer` optionally hashes an already-present exact `tokenizer.json`;
it never downloads one and does not qualify tokenizer execution.

Successful JSON output keeps `payload_verified`, `native_supported`,
`native_qualified`, `export_qualified`, `native_alias_qualified`,
`image_max_qualified`, `full_document_qualified` and
`framework_runtime_observed` false for the default source-only run. Optional
tokenizer hashing changes only tokenizer payload verification, not any runtime
qualification. The model checkpoint payload is never opened.

The existing [controls workflow](../.github/workflows/controls.yml) runs:

```bash
python tests/test_lightonocr3_4b_profile.py -v
```

At implementation commit `c41469b24cb7835b4fe2706835b59db583392c76`,
[hosted controls run 41908139315](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41908139315)
passes all 23 new cases and all 236 tests across 18 unittest executions.
[Static guards](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41908139236/job/126848192178)
also pass all 51 CI-selection cases, including the five new routing/workflow
checks. The jobs checked out PR merge `34fcb32e`, whose tree exactly equals the
implementation tree. See the [machine-readable receipt](../tests/results/lightonocr3-4b/source-ci-c41469b2.json).
This is source/control evidence; the overall platform Build was still in flight
when the receipt was recorded.

Routing lists reviewed source fixtures explicitly. New unknown fixture files
still select ordinary validation, and adding a source fixture cannot suppress
runtime dependencies. Changing the selector itself still selects the full
ordinary CI lanes. Hosted results are recorded separately from local syntax and
bounded CLI checks; neither establishes model quality.

All seven selected integration workflows pass at `91b24764`, after main
`17c96639` was merged. The combined controls job passes 317 tests in 27 suites.
The [integration receipt](../tests/results/lightonocr3-4b/integration-ci-91b24764.json)
records each run and the actual PR checkout. This is ordinary CI evidence,
not execution of the new converter or native model.

## Explicit 4B export contract

The shared converter accepts this checkpoint only with
`--profile lightonocr3-4b`, its exact model ID and immutable revision. An implicit
generic conversion of the 4B model is rejected before filesystem or network
access. The command below is a future hosted conversion entry point; it has not
been executed as part of the source/control milestone:

```bash
python models/convert-qwen35-ocr-to-gguf.py \
  --model lightonai/LightOnOCR-3-4B \
  --revision b71010f095dc3735de7ee6b9969bff8f8f50b39a \
  --profile lightonocr3-4b \
  --dtype f16 \
  --precision-recipe lightonocr3-4b-vision-f32 \
  --output lightonocr3-4b-vision-f32.gguf
```

Admission verifies six exact metadata files and the complete tokenizer before
the checkpoint download. Conversion then checks the original header/schema,
streams the whole checkpoint SHA-256, verifies the loaded Transformers 5.5.4
model/config source hashes, and compares the original meta-device model's
724 state-dict entries against all 723 stored tensors plus the tied output-head
alias. There is no MTP exclusion or additional stored head. Each exported tensor
records its source shape/BF16 payload hash and converted shape/type/payload hash;
the Conv3D patch tensor alone is flattened to `[1024, 1536]` for GGUF.

| Export | F16 tensors | F32 tensors | Meaning |
|---|---:|---:|---|
| `--dtype f32` | 0 | 723 | Full F32 baseline |
| F16, `original-f16` | 349 | 374 | Matrices in F16; norms, convolution and recurrent coefficients remain F32 |
| F16, `lightonocr3-4b-vision-f32` | 249 | 474 | Separately named 24-layer recipe, retaining 100 vision matrices in F32 |

The historical `vision-f32` recipe still means exactly 52 matrices across
12 vision layers. It is rejected for 4B; the new recipe is rejected for other
profiles. The 4B writer emits **format 5**, which current native format 1–4
admission rejects. It includes the distinct tokenizer settings, head geometry,
vision dimensions and both EOS declarations. Every native/export/document/
alias qualification flag remains false, and `gguf_roundtrip_verified` remains
false until a separate actual serialization/readback run proves it.

Sixteen stdlib export controls pass in
[hosted run 41919572206](https://github.com/CrispStrobe/CrispEmbed/actions/runs/41919572206)
at `aa661568`, with 333 controls across 28 suites. The separate Ovis export audit
also passes. Ordinary Build/WASM/Rust/image jobs at that checkpoint stopped on an
existing workflow-contract assertion requiring the source-preflight command on
its own inline `run:` line. The original step is restored; the new controls run
in a separate block without changing that historical assertion.

The export controls check source identity, metadata/schema rejection, precision membership,
dispatch ordering and explicit unqualified flags. Full-tokenizer admission is
mocked only where a test needs to reach later metadata logic; those controls do
not claim payload or framework execution. The new
[`lighton4b_export_meta.py`](../tools/ci-heavy/lighton4b_export_meta.py) hosted probe
instantiates the pinned original model on the meta device without downloading
checkpoint weights, then validates the actual 724-key schema, 723 unique tensors
and tied parameter identity. Fourteen stdlib probe controls accompany it; actual
meta construction is a separate hosted result.

## Default-off grouped-head component and hosted reference

`linear_grouped_core_cpu` and `linear_grouped_cpu` are separate entry points in
the Qwen3.5 component library. Both require `CRISPEMBED_QWEN35_OCR=1` and
`CRISPEMBED_QWEN35_GDN_GROUPED_HEADS=1`; unset, empty and literal `0` disable the
new path. Existing `linear_core_cpu`, `linear_cpu`, `project_cpu`, `delta_cpu`
and their default callers retain their old implementation. Equal-head input to
the new entry delegates to the old API. The old source-pin receipts are unchanged;
the reviewed additive change is recorded in the separate amendment chain.

The grouped path keeps compact `(2*key_heads + value_heads)*width` convolution
channels and cache history. It repeats complete Q/K heads **after** causal
convolution and SiLU, then reuses the existing recurrence and gated norm. Key
and value widths must match; the projected entry accepts only F32/F16 tensors.
This component has no owning model context, tokenizer, image frontend or public
model admission. Its F16 storage acceptance is not F16 numerical qualification.

The independent [`lighton4b_gdn_reference.py`](../tools/ci-heavy/lighton4b_gdn_reference.py)
protocol executes the unmodified Transformers **5.5.4** constructor, forward,
chunk/recurrent functions and `DynamicCache`, pinned to release commit
[`75d3bdcd`](https://github.com/huggingface/transformers/tree/75d3bdcd4b3ba70cee21287218d9764f33da41f0).
It verifies three whole source modules and six function hashes before using the
original Torch fallback. Package versions are fixed in the dedicated requirements
file; execution uses CPU/F32, two threads and deterministic operations.

| Synthetic case | Key/value heads | Batch | Prefill tokens | Covered boundary |
|---|---|---:|---:|---|
| `equal16-t65` | 16/16 | 1 | 65 | Unchanged equal-head behavior and chunk boundary |
| `grouped16-32-t65` | 16/32 | 1 | 65 | Actual 4B head geometry and chunk boundary |
| `grouped16-32-t1` | 16/32 | 1 | 1 | Short-prefill convolution padding |
| `grouped2-4-t3-b2` | 2/4 | 2 | 3 | Independent batch state and short history |

All four cases use head width 128, convolution width four, **synthetic hidden
width 16 and synthetic weights**. Each runs prefill, two cached tokens and a
fresh-cache full-prefix pass. Original observation-neutrality and incremental/
full-prefix controls run before native compilation. Original arrays and hashes
are frozen first; primary native cache state is never replaced with reference
state. Separate original-input raw-core and delta lanes attribute errors while
maintaining their own native caches.

The native runner produces 52 named comparisons per case (208 total) and
31 controls for the equal-head case or 25 for each grouped case. Comparisons
report full element coverage, finite values, magnitudes, max/mean absolute error,
RMS error and minimum row cosine. Acceptance remains max absolute error
`0.002` and minimum row cosine `0.99999`; corruption, missing stages, repeated
controls or changed frozen sources fail. The 17 stdlib protocol controls verify
these contracts without invoking a framework.

The existing controls workflow selects this bounded reference job from the full
PR merge-base or entire push diff, including renames. Unknown events or incomplete
diff evidence select it; a proven unrelated diff may skip it. Twenty selector
controls cover those decisions. The job uses a clean Python 3.12 environment,
recursive submodules, readonly repository permissions and a 25-minute limit.
Checkpoint acquisition is disabled. Reports, original/native raw arrays and
negative results are uploaded together for 90 days, within a 768 MiB protocol
bound. `gdn/component-report.json` and `meta/export-meta.json` are the small review
entry points; raw arrays stay in the hosted artifact.

Even a passing component result would establish only this synthetic CPU/F32
scope. Full-width checkpoint F32/F16 parity, actual GGUF serialization/readback,
decoder/vision ownership, preprocessing/tokenizer behavior, complete document
outputs, quantization and public aliases remain independent gates.

## Native implementation sequence and acceptance gates

| Order | Concrete missing work | Required evidence before advancement |
|---|---|---|
| 1 | Freeze the actual loaded original framework, full tokenizer, image/rendering and generation recipe. Capture the original first | Immutable runtime/package/input/weight hashes; exact prompts, preprocessing arrays, token IDs, complete outputs and EOS/cap status |
| 2 | Additive default-off 16-key/32-value-head component and hosted reference are implemented | Run the registered original/native comparison and retain earliest-stage absolute error/magnitude plus all cache controls; then extend evidence to full-width checkpoint F32/F16 |
| 3 | Replace small-model ownership/mapping assumptions with an explicit pinned 4B path | All 723 tensors validated; 32-layer text and 24-layer vision parameters actually passed through recognition and caches; no generic family-wide admission |
| 4 | Implement bounded full-vocabulary projection; the distinct 24-layer precision recipe is implemented and unqualified | Complete logits, bounded peak memory, actual F32/F16 serialization/readback and unchanged historical 12-layer `vision-f32` behavior |
| 5 | Add a default-off native path using `core_env` helpers and documented gates | Same-input original/native F32/F16 stage and full-logit comparisons, then actual free-generation text/blocks, EOS and long-page evidence |
| 6 | Evaluate useful document tasks and deployment tiers, then quantization/publication separately | Licensed text/table/formula/layout/handwriting/degraded/multilingual inputs; quality, latency/RSS, F16 and per-precision gates; immutable upload/readback and public-consumer checks before an alias |

The current blockers are visible in
[`qwen35_ocr.cpp`](https://github.com/CrispStrobe/CrispEmbed/blob/2f748825ff3b56b8f2b89d156986b94218ae28c3/src/qwen35_ocr.cpp),
[`qwen35_linear.cpp`](https://github.com/CrispStrobe/CrispEmbed/blob/2f748825ff3b56b8f2b89d156986b94218ae28c3/src/qwen35_linear.cpp),
[`qwen35_delta.cpp`](https://github.com/CrispStrobe/CrispEmbed/blob/2f748825ff3b56b8f2b89d156986b94218ae28c3/src/qwen35_delta.cpp),
[`qwen35_decoder.cpp`](https://github.com/CrispStrobe/CrispEmbed/blob/2f748825ff3b56b8f2b89d156986b94218ae28c3/src/qwen35_decoder.cpp),
and the [converter](https://github.com/CrispStrobe/CrispEmbed/blob/2f748825ff3b56b8f2b89d156986b94218ae28c3/models/convert-qwen35-ocr-to-gguf.py).

The owning context currently expects 473 tensors and the small model's fixed
dimensions. If the existing equal-head QKV formula were used with 32 value
heads, it would expect `3*32*128 = 12288` channels; the actual grouped checkpoint
has `2*16*128 + 32*128 = 8192`. Its embedding/output matrix has 635,699,200
elements, exceeding `project_cpu()`'s 536,870,912-element limit. Materializing
that matrix in F32 alone would occupy 2,542,796,800 bytes, before other tensors
or caches; this is arithmetic, not a measured peak-memory result. Raising the
limit or adding an alias does not resolve these architectural requirements.

Keep all historical LightOn0.8B strict failures and PR141's LightOn1B results.
This separate source profile does not amend their source pins or acceptance bars.
