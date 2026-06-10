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

**Preview release** — document layout detection for
[CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).
Detects 17 types of document regions in a single pass.

**Architecture**: RT-DETRv2 — ResNet-50 backbone + hybrid FPN/PAN encoder
with AIFI self-attention + 6-layer transformer decoder with deformable
cross-attention (300 learned queries). 42M parameters.

**Source**: [docling-project/docling-layout-heron](https://huggingface.co/docling-project/docling-layout-heron-onnx) (Apache 2.0).

## Status: Preview

Full architecture implemented in C++ via ggml:
- ResNet-50 backbone with folded decomposed BN (Conv→Mul→Add) ✓
- Hybrid encoder: FPN + PAN with CSP blocks (SiLU activation) + AIFI ✓
- 6-layer transformer decoder with deformable cross-attention ✓
- CPU-side bilinear grid sampling for deformable attention ✓
- Detection heads with iterative bbox refinement ✓

Detection confidence is lower than ONNX reference (~0.07 vs ~0.65).
The encoder feature magnitudes are ~2-5× attenuated compared to the
ONNX reference, causing reduced decoder confidence. Use `score_threshold=0.05`
for best results. The model produces structurally correct detections.

Set `LAYOUT_DEBUG=1` for verbose per-layer output.

## Model Variants

| Variant | Size |
|---------|------|
| **F16** | **81 MB** |

## Classes (17)

caption, footnote, formula, list_item, page_footer, page_header,
picture, section_header, table, text, title, document_index, code,
checkbox_selected, checkbox_unselected, form, key_value_region

## License

Apache 2.0.
