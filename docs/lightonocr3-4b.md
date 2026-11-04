# LightOnOCR-3-4B — pinned source preflight

Updated 4 November 2026. Implementation lane:
[`feat/ocr-next-models-20261104` / PR143](https://github.com/CrispStrobe/CrispEmbed/pull/143).

**Status: source preflight implemented; native inference and export remain unqualified.**
The first slice verifies the original metadata, the actual safetensors header and
bounded tokenizer-prefix evidence. It does not load model weights, run a framework,
produce GGUF, change native admission or publish a registry alias. The existing
LightOn1B/0.8B qualification and cache work remain in PR141.

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

Routing lists reviewed source fixtures explicitly. New unknown fixture files
still select ordinary validation, and adding a source fixture cannot suppress
runtime dependencies. Changing the selector itself still selects the full
ordinary CI lanes. Hosted results are recorded separately from local syntax and
bounded CLI checks; neither establishes model quality.

## Native implementation sequence and acceptance gates

| Order | Concrete missing work | Required evidence before advancement |
|---|---|---|
| 1 | Freeze the actual loaded original framework, full tokenizer, image/rendering and generation recipe. Capture the original first | Immutable runtime/package/input/weight hashes; exact prompts, preprocessing arrays, token IDs, complete outputs and EOS/cap status |
| 2 | Generalize linear attention to 16 key / 32 value heads, including Q/K expansion, convolution cache and recurrent state | Independent original comparison for the earliest affected stages, with absolute error and magnitude; keep the existing equal-head path intact |
| 3 | Replace small-model ownership/mapping assumptions with an explicit pinned 4B path | All 723 tensors validated; 32-layer text and 24-layer vision parameters actually passed through recognition and caches; no generic family-wide admission |
| 4 | Implement bounded full-vocabulary projection and a separate 24-layer precision recipe | Complete logits, bounded peak memory and unchanged old recipes; the historical 12-layer `vision-f32` recipe must retain its exact meaning |
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
