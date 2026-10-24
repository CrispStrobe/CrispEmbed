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
Apache-2.0. The audited 473-tensor checkpoint is the non-speculative main-model
schema. The source config declares MTP1, but this export makes no MTP support
claim. Stock llama.cpp compatibility is not claimed.

The Ovis card declares [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17)
as its base; that card declares [Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base/tree/dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68).
All three pinned metadata records declare Apache-2.0. Their LICENSE text is
identical after newline normalization; the root Base file has a different raw
SHA-256 due to line endings. The exact chain, raw file hashes and inspection
scope are in `manifest.json`. No separate NOTICE file was listed at those pins.

| File | Bytes | SHA-256 | Recipe |
|---|---:|---|---|
| `${filename}` | ${model_bytes} | `${model_sha256}` | `${recipe}` |

The pinned upstream card is also preserved as `UPSTREAM_README.md`. The audited
CrispEmbed source commit is `${audited_source_commit}`. This is a CPU-only
experimental reader. After the original image resize it accepts at most 1024
visual patches; the original 2880²-pixel maximum has not been qualified.
Until a qualified release includes it, build that source and enable the three
experimental entry gates and the audited vision math settings to load the GGUF
by its explicit local path:

```sh
CRISPEMBED_QWEN35_OCR=1 CRISPEMBED_OVISOCR2_FRONTEND=1 CRISPEMBED_OVISOCR2_INFERENCE=1 ${vision_gates}${quant_gate}\
  crispembed -t 2 --ocr-pipeline page.png --ocr-engine ovisocr2 \
    --ocr-rec ${filename} --ocr-max-tokens 1024
```

Recognition returns raw generated Markdown/HTML. The quality reports separately
record the upstream parser cleanup/filter view. The reader uses the pinned
original task prompt and rejects a custom prompt on the explicit Ovis stage. The bounded original/native, public API, independent
quality and quantization-selection receipts were reviewed before this file was
uploaded. ${calibration_note}
Their exact SHA-256 values and source commits are in `manifest.json`. The quality
panels cover four short self-authored heldout pages and three reviewed documents;
they do not establish broad OCR accuracy. The publication workflow verifies the
uploaded GGUF again from an empty cache at its immutable Hugging Face commit.

No default model alias, CrispEmbed runtime release, or app selection is activated
by this model repository. Original model probes and test-only weights are held
separately from this distribution.
