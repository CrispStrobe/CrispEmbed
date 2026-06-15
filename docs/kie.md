# Key Information Extraction (KIE)

CrispEmbed provides two KIE backends for extracting structured fields from
document images (receipts, invoices, forms, business cards).

## Phase 1: OCR + NER Pipeline

Chains existing models — no new model needed:
1. **OCR**: DBNet text detection + TrOCR/VLM recognition → text + bounding boxes
2. **GLiNER NER**: zero-shot named entity recognition on the OCR text
3. **Mapping**: NER entities mapped back to source OCR regions via character offsets

Strengths: zero-shot (any field name at inference time), uses existing models.
Weakness: text-only NER ignores spatial layout.

```bash
./build/crispembed -m gliner-lfm-f32.gguf \
    --ocr-det dbnet-det-f16.gguf --ocr-rec trocr-printed-q8_0.gguf \
    --kie receipt.png --kie-labels "total,date,vendor" --json
```

### Pipeline flow

```
image → OCR detect → OCR recognize → concat text → GLiNER NER → map to bboxes → JSON
```

Files: `src/kie_pipeline.{h,cpp}`, `src/gliner_ner.{h,cpp}`, `src/ocr_orchestrator.{h,cpp}`

## Phase 2: LiLT (Layout-Aware Token Classification)

LiLT (Language-independent Layout Transformer) is a dual-stream encoder
that processes both text and spatial layout simultaneously.

### Architecture

```
Text stream:   RoBERTa (768d, 12 layers, 12 heads, vocab 50265)
Layout stream: Layout transformer (192d = 768/4, 12 layers, 12 heads)
Coupling:      BiACM at every layer
Output:        Token classification head (Linear → num_labels)
```

**Total parameters**: 130.7M (MIT license, SCUT-DLVCLab).

### BiACM (Bidirectional Attention Complementation)

At each layer, text and layout attention scores are combined before softmax:

```
text_Q, text_K, text_V = text_proj(text_hidden)
layout_Q, layout_K, layout_V = layout_proj(layout_hidden)

text_scores    = text_Q @ text_K^T / sqrt(head_dim)
layout_scores  = layout_Q @ layout_K^T / sqrt(layout_head_dim)

combined = text_scores + layout_scores  # BiACM fusion

text_attn_out    = softmax(combined) @ text_V
layout_attn_out  = softmax(combined) @ layout_V
```

This allows the layout stream to influence which tokens the text stream
attends to, and vice versa — enabling layout-aware reasoning without
pixel-level features.

### Layout Embeddings

Each token's bounding box `[x0, y0, x1, y1]` is encoded as:

```
6 embeddings (each 128d):
  x_embd(x0), y_embd(y0), x_embd(x1), y_embd(y1), h_embd(y1-y0), w_embd(x1-x0)
  → concatenate to 768d → Linear(768 → 192) → + position_embd → LayerNorm
```

Coordinates are in [0, 1000] range (normalized page coordinates).

### FUNSD Labels

The fine-tuned model (`lilt-funsd`) classifies tokens into 7 IOB labels:

| Label | Meaning |
|-------|---------|
| `O` | Outside any entity |
| `B-HEADER` | Beginning of a header |
| `I-HEADER` | Inside a header |
| `B-QUESTION` | Beginning of a form field label |
| `I-QUESTION` | Inside a form field label |
| `B-ANSWER` | Beginning of a form field value |
| `I-ANSWER` | Inside a form field value |

### Usage

```bash
# CLI — from JSON input (pre-tokenized)
echo '{"input_ids":[0,10566,35,5480,35,68,3818,4,2466,2],
  "bbox":[[0,0,0,0],[10,50,90,80],[90,50,110,80],[250,50,330,80],
          [330,50,350,80],[360,50,390,80],[390,50,430,80],
          [430,50,440,80],[440,50,470,80],[0,0,0,0]]}' > /tmp/input.json
./build/crispembed -m lilt-funsd-q8_0.gguf --lilt /tmp/input.json --json

# Python
from crispembed import CrispLiLT
lilt = CrispLiLT("lilt-funsd-q8_0.gguf")
tokens = lilt.classify(input_ids=[0, 10566, 35, 2],
                       bbox=[[0,0,0,0], [10,50,90,80], [90,50,110,80], [0,0,0,0]])
```

### Model Variants

| Model | Description | F32 | Q8_0 | Q4_K |
|-------|-------------|-----|------|------|
| `lilt-funsd` | FUNSD fine-tuned (7 labels) | 498 MB | 134 MB | 90 MB |
| `lilt-base` | Base encoder (no classifier) | 498 MB | 134 MB | 90 MB |

### Parity

crispembed-diff harness results (C++ vs HF transformers, F32):

```
text_embed      cos=1.000000  max_abs=9.54e-07  PASS
layout_embed    cos=1.000000  max_abs=2.04e-06  PASS
layer_0_text    cos=1.000000  max_abs=3.81e-06  PASS
layer_0_layout  cos=1.000000  max_abs=7.37e-04  PASS
...
layer_11_text   cos=1.000000  max_abs=7.77e-04  PASS
25/25 stages PASS, 16/16 token labels match (100%)
```

### Files

| File | Purpose |
|------|---------|
| `src/lilt_kie.{h,cpp}` | C++ inference engine (ggml graph with BiACM) |
| `src/kie_pipeline.{h,cpp}` | Composite KIE pipeline (Phase 1 + Phase 2) |
| `models/convert-lilt-to-gguf.py` | HF → GGUF converter |
| `tools/dump_lilt_reference.py` | Per-layer activation dumper |
| `tests/test_lilt_diff.cpp` | crispembed-diff parity test |
| `tests/test_lilt_kie.cpp` | Integration test |
| `tests/test_kie_pipeline.cpp` | E2E pipeline test |
