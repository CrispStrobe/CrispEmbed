---
license: apache-2.0
base_model: ATH-MaaS/OvisOCR2
base_model_relation: quantized
pipeline_tag: image-to-text
tags: [gguf, crispembed, ocr, experimental]
---

# OvisOCR2 for CrispEmbed (${precision})

This GGUF contains the complete main-model checkpoint of
[ATH-MaaS/OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) at
`cbdede82e048659b83310476ff807da825f8fe78`, converted for the experimental
CrispEmbed format-2 OvisOCR2 reader. The pinned upstream card and LICENSE declare
Apache-2.0. The source includes a separate MTP checkpoint that is outside this
non-speculative main-model export. Stock llama.cpp compatibility is not claimed.

| File | Bytes | SHA-256 | Recipe |
|---|---:|---|---|
| `${filename}` | ${model_bytes} | `${model_sha256}` | `${recipe}` |

The pinned upstream card is also preserved as `UPSTREAM_README.md`. The audited
CrispEmbed source commit is `${audited_source_commit}`. Until a
qualified release includes it, build that source and enable the three
experimental gates to load the GGUF by its explicit local path:

```sh
CRISPEMBED_QWEN35_OCR=1 CRISPEMBED_OVISOCR2_FRONTEND=1 CRISPEMBED_OVISOCR2_INFERENCE=1 ${quant_gate}\
  crispembed --ocr-pipeline page.png --ocr-engine ovisocr2 --ocr-rec ${filename}
```

The reader uses the pinned original task prompt and rejects a custom prompt on
the explicit Ovis stage. The bounded original/native, public API, independent
quality and quantization-selection receipts were reviewed before this file was
uploaded. ${calibration_note}
Their exact SHA-256 values and source commits are in `manifest.json`. The quality
panels cover four short self-authored heldout pages and three reviewed documents;
they do not establish broad OCR accuracy. The publication workflow verifies the
uploaded GGUF again from an empty cache at its immutable Hugging Face commit.

No default model alias, CrispEmbed runtime release, or app selection is activated
by this model repository. Original model probes and test-only weights are held
separately from this distribution.
