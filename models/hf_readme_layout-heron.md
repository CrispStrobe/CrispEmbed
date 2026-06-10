---
license: apache-2.0
language:
- en
tags:
- document-layout-analysis
- object-detection
- rt-detr
- gguf
- crispembed
base_model: docling-project/docling-layout-heron
pipeline_tag: object-detection
---

# Document Layout Analysis (RT-DETRv2 Heron) — GGUF

**Preview release** — document layout detection model for
[CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).
Detects 17 types of document regions: text, title, table, figure,
formula, caption, section header, list item, footnote, code, and more.

**Architecture**: RT-DETRv2 with ResNet-50 backbone (23M params) +
hybrid FPN/PAN encoder + transformer decoder with deformable
cross-attention (300 learned queries). 42M total parameters.

**Source**: [docling-project/docling-layout-heron](https://huggingface.co/docling-project/docling-layout-heron-onnx) (Apache 2.0).

## Status: Preview

The C++ inference implements the full RT-DETRv2 architecture:
- ResNet-50 backbone with decomposed BN folding ✓
- Hybrid encoder (FPN + PAN + AIFI self-attention) ✓
- 6-layer transformer decoder with deformable cross-attention ✓
- Detection heads with iterative bbox refinement ✓

Detection scores are currently lower than the ONNX reference (~0.12 vs ~0.65)
due to incomplete CSP block implementation and deformable attention refinement.
The model produces structurally correct detections but with reduced confidence.
Use with a lower score threshold (0.05-0.1) for best results.

## Model Variants

| Variant | Size | Notes |
|---------|------|-------|
| F32 | 161 MB | reference |
| **F16** | **81 MB** | recommended |

## Classes (17)

caption, footnote, formula, list_item, page_footer, page_header,
picture, section_header, table, text, title, document_index, code,
checkbox_selected, checkbox_unselected, form, key_value_region

## Usage

```bash
# Requires LAYOUT_DEBUG=1 env var for verbose output
./test-layout-detect layout-heron-f16.gguf document.png 0.1
```

## License

Apache 2.0 (same as [docling-project/docling-layout-heron](https://huggingface.co/docling-project/docling-layout-heron)).
