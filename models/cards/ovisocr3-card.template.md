---
language:
  - en
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - ocr
  - gguf
  - crispembed
---

# OvisOCR3 F16 for CrispEmbed

This file is an independently reviewed CPU F16 conversion of
[`ATH-MaaS/OvisOCR3`](https://huggingface.co/ATH-MaaS/OvisOCR3/tree/${source_revision})
at commit `${source_revision}`. It is a fixed plain OCR task. The model file is
`ovisocr3-f16.gguf`, ${model_bytes} bytes, SHA-256 `${model_sha256}`.

The upstream model and declared Qwen3.5 base chain use Apache-2.0 terms. This
repository carries the upstream `LICENSE`, the pinned upstream
`UPSTREAM_README.md`, this card, and a publication `manifest.json`. The original
source audit is in CrispEmbed at
[`ovisocr3-source-contract.json`](https://github.com/CrispStrobe/CrispEmbed/blob/${evidence_source_commit}/tests/results/qwen35-ocr/ovisocr3-source-contract.json).
The model's declared base is `Qwen/Qwen3.5-0.8B`; that model declares
`Qwen/Qwen3.5-0.8B-Base`. Their audited revisions and license file hashes are
recorded in the source contract. Upstream does not identify the exact training
start commit. No separate speculative MTP checkpoint was found in the audited
main checkpoint; this conversion does not claim MTP support.

## Qualification and scope

The reviewed CrispEmbed source at `${public_source_commit}` compares the
original CPU F32 module with the converted F16 engine, eleven frozen plain OCR
cases, and model-backed C, Python, CLI, HTTP, Rust, and Dart surfaces. The
publication decision and exact report hashes are in `manifest.json`. The
qualified native path is CPU only and limited to 1024 image patches after
preprocessing. The upstream processor permits larger input resizes; those
sizes are outside this file's qualification. The original T4 FP16 vLLM run
uses controlled Triton decoder attention and is distinct from the CPU F32
reference. This card makes no speed, large-image, other-task, BF16, GPU-native,
or quantized-quality claim.

Three runtime gates are required for the experimental OvisOCR3 path:
`CRISPEMBED_QWEN35_OCR=1`, `CRISPEMBED_OVISOCR3_FRONTEND=1`, and
`CRISPEMBED_OVISOCR3_INFERENCE=1`. The reviewed CPU vision selection also
enables `CRISPEMBED_QWEN35_VISION_PRECISE_NORM=1` and
`CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=1`. The default CrispEmbed OCR model
is not changed by this publication.

## Run the qualified path

Build CrispEmbed from the qualified source commit above, download the GGUF from
this repository, and pass its local path explicitly:

```bash
CRISPEMBED_QWEN35_OCR=1 \
CRISPEMBED_OVISOCR3_FRONTEND=1 \
CRISPEMBED_OVISOCR3_INFERENCE=1 \
CRISPEMBED_QWEN35_VISION_PRECISE_NORM=1 \
CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=1 \
CRISPEMBED_QWEN35_VISION_CASCADE_NORM=0 \
  crispembed -t 2 --offline --ocr-pipeline page.png --ocr-engine ovisocr3 \
    --ocr-rec ovisocr3-f16.gguf --ocr-max-tokens 1024 --json
```

The command uses the fixed plain OCR prompt. The cap bounds runtime and can
yield partial text. Public pipeline results expose text/confidence; generated
IDs and stop IDs are checked by the separate low-level qualification harness.
