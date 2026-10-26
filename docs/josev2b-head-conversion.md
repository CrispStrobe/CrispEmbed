# Josev original joint-head F32 conversion preparation

This prepares a component artifact, with no conversion execution or native parity
claim. The [head schema](../tools/josev2b_head_schema.py) follows the pinned
[original class](https://huggingface.co/mayflowergmbh/Josev-2B/blob/6f0a31c8ed683aba79097eb3f48df2c9ab560afd/joint_schema_model.py):
122 named tensors and 30,472,708 parameters. Hidden width is 2048; head width 512,
two evidence-routing layers and four cross-field TransformerDecoderLayers.
The actual original loaded-head inventory must independently confirm these
counts and every shape before conversion. No checkpoint header or payload was
downloaded to derive the schema.

The separate [converter](../tools/convert_josev2b_head_f32.py) requires an explicit
reviewed original receipt SHA and full immutable original worker commit. It
checks completed original-only status, exact source/panel/model and neutral cases,
then actual checkpoint/loaded-head F32 identities for every tensor. Missing
reference evidence rejects before optional NumPy/GGUF imports or model access.
All original source identities remain fixed; a future environment profile does
not silently retarget a frozen receipt.

The original head can contain BF16 or F32 parameters. The converter reads a bounded
safetensors header, rejects wrong names/shapes/dtypes, gaps, overlaps, aliasing and
trailing bytes, then promotes BF16 by exact bit placement into F32. Every promoted
F32 tensor must equal the hash of the original loader's actual F32 tensor. No
quantization or temperature fitting occurs. Rank-zero scales use an explicitly
recorded one-element GGUF vector; all other shapes preserve PyTorch row order.
The written GGUF is independently read back; exact declared dimensions and F32
payload hashes must match, including the sign bit of zero. Eight small standard
library controls cover source receipts, malformed headers and byte preservation;
they do not execute checkpoint conversion.

The artifact architecture is `josev2b_head_reference`, deliberately separate from
a full inference model. It excludes the language/vision backbones, tokenizer and
semantic-span renderer. It records Apache-2.0 provenance and the official
PyTorch profile with no extra temperature. Preserve the publisher's license and
applicable notices beside any derived artifact. It does not qualify the separate
author-GGUF route's per-type calibration.

Execute only on hosted CPU CI or as postprocessing inside the approved original
GPU session, after the actual reference is reviewed. The required array/GGUF
packages belong in that execution environment; this preparation installs none.
Pass immutable receipts and task-owned fresh output paths:

```sh
python tools/convert_josev2b_head_f32.py \
  --checkpoint "$JOSEV_CHECKPOINT/joint_head.safetensors" \
  --original-report "$JOSEV_REFERENCE/reference.json" \
  --original-sha256 REVIEWED_RECEIPT_SHA256 \
  --original-source REVIEWED_FULL_WORKER_COMMIT \
  --output "$JOSEV_OUTPUT/joint-head-f32.gguf"
```

There is no caller workflow or dispatcher registration yet: the original GPU
reference remains pending. Preserve failed artifacts/logs and conversion receipts
remotely; fetch only small reports. Next is a gated native head consuming the
actual original final hidden rows, lexical option vectors and semantic spans,
with every projection/routing/field/logit/probability boundary compared through
the original dumper and crispembed-diff. Then compose it with an explicit
Qwen3.5-2B hidden-only prefill profile. This component preparation does not add
public APIs, model aliases, representative decision quality, quantized artifacts,
releases or an application update.
