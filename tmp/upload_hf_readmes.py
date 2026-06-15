#!/usr/bin/env python3
"""Upload README model cards to HuggingFace repos."""
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from huggingface_hub import HfApi
api = HfApi()

FUNSD_README = r"""---
license: mit
tags:
- gguf
- lilt
- document-understanding
- key-information-extraction
- token-classification
- form-understanding
- layout-aware
base_model: philschmid/lilt-en-funsd
pipeline_tag: token-classification
---

# LiLT FUNSD — GGUF

GGUF conversion of [philschmid/lilt-en-funsd](https://huggingface.co/philschmid/lilt-en-funsd) for use with [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).

**LiLT** (Language-independent Layout Transformer) is a dual-stream encoder that combines RoBERTa (768d text) with a parallel layout transformer (192d) via BiACM (bidirectional attention complementation). It takes OCR text + bounding boxes and performs token classification for document understanding.

This variant is fine-tuned on **FUNSD** (Form Understanding in Noisy Scanned Documents) with 7 IOB labels: O, B-HEADER, I-HEADER, B-QUESTION, I-QUESTION, B-ANSWER, I-ANSWER.

## Model Details

| Property | Value |
|----------|-------|
| Architecture | LiLT (RoBERTa + Layout Transformer + BiACM) |
| Parameters | 130.7M |
| Hidden size | 768 (text) / 192 (layout) |
| Layers | 12 |
| Heads | 12 |
| Vocab | 50,265 (RoBERTa BPE) |
| Labels | 7 (FUNSD IOB) |
| License | MIT |
| Base model | [SCUT-DLVCLab/lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base) |

## Available Formats

| File | Format | Size |
|------|--------|------|
| `lilt-funsd-f32.gguf` | Float32 | 498 MB |
| `lilt-funsd-q8_0.gguf` | Q8_0 | 134 MB |
| `lilt-funsd-q4_k.gguf` | Q4_K | 90 MB |

## Usage

### Python
```python
from crispembed import CrispLiLT

lilt = CrispLiLT("lilt-funsd-q8_0.gguf")
tokens = lilt.classify(
    input_ids=[0, 10566, 35, 5480, 35, 68, 3818, 4, 2466, 2],
    bbox=[[0,0,0,0], [10,50,90,80], [90,50,110,80], [250,50,330,80],
          [330,50,350,80], [360,50,390,80], [390,50,430,80],
          [430,50,440,80], [440,50,470,80], [0,0,0,0]],
)
for t in tokens:
    print(f"{t['label']:15s} score={t['score']:.2f}")
```

### CLI
```bash
echo '{"input_ids":[0,10566,35,2],"bbox":[[0,0,0,0],[10,50,90,80],[90,50,110,80],[0,0,0,0]]}' > input.json
crispembed -m lilt-funsd-q8_0.gguf --lilt input.json --json
```

## Parity

Verified against HuggingFace transformers using the crispembed-diff harness:

- 25/25 encoder stages: cos_min = 1.000000
- 16/16 token labels match (100%)
- max_abs < 1.6e-03 across all layers

## Citation

```bibtex
@inproceedings{wang2022lilt,
  title={LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding},
  author={Wang, Jiapeng and Jin, Lianwen and Ding, Kai},
  booktitle={ACL},
  year={2022}
}
```
"""

BASE_README = r"""---
license: mit
tags:
- gguf
- lilt
- document-understanding
- layout-aware
- feature-extraction
base_model: SCUT-DLVCLab/lilt-roberta-en-base
pipeline_tag: feature-extraction
---

# LiLT Base — GGUF

GGUF conversion of [SCUT-DLVCLab/lilt-roberta-en-base](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base) for use with [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed).

**LiLT** (Language-independent Layout Transformer) is a dual-stream encoder that combines RoBERTa (768d text) with a parallel layout transformer (192d) via BiACM (bidirectional attention complementation). This is the **base model** (pre-trained, no task-specific head) — use it as a starting point for fine-tuning on your own document understanding tasks.

For a ready-to-use model fine-tuned on form understanding, see [cstr/lilt-funsd-GGUF](https://huggingface.co/cstr/lilt-funsd-GGUF).

## Model Details

| Property | Value |
|----------|-------|
| Architecture | LiLT (RoBERTa + Layout Transformer + BiACM) |
| Parameters | 130.7M |
| Hidden size | 768 (text) / 192 (layout) |
| Layers | 12 |
| Heads | 12 |
| Vocab | 50,265 (RoBERTa BPE) |
| License | MIT |

## Available Formats

| File | Format | Size |
|------|--------|------|
| `lilt-base-f32.gguf` | Float32 | 498 MB |
| `lilt-base-q8_0.gguf` | Q8_0 | 134 MB |
| `lilt-base-q4_k.gguf` | Q4_K | 90 MB |

## Architecture

LiLT's key innovation is **BiACM** (Bidirectional Attention Complementation):

1. Text and layout streams each compute separate Q/K/V projections
2. Attention scores from both streams are summed before softmax
3. Each stream applies the combined attention to its own values
4. Separate FFN layers process each stream independently

This allows layout information to guide text attention patterns (and vice versa) without requiring pixel-level image features.

### Layout Embeddings

Each token's bounding box [x0, y0, x1, y1] is encoded via 6 learned position embeddings (x, y, h, w) concatenated to 768d, projected to 192d, and combined with sequential position embeddings.

## Parity

Verified against HuggingFace transformers:
- 25/25 encoder stages: cos_min = 1.000000
- max_abs < 1.6e-03 across all layers

## Citation

```bibtex
@inproceedings{wang2022lilt,
  title={LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding},
  author={Wang, Jiapeng and Jin, Lianwen and Ding, Kai},
  booktitle={ACL},
  year={2022}
}
```
"""

api.upload_file(
    path_or_fileobj=FUNSD_README.encode('utf-8'),
    path_in_repo='README.md',
    repo_id='cstr/lilt-funsd-GGUF',
    commit_message='Add model card README',
)
print('lilt-funsd-GGUF README uploaded')

api.upload_file(
    path_or_fileobj=BASE_README.encode('utf-8'),
    path_in_repo='README.md',
    repo_id='cstr/lilt-base-GGUF',
    commit_message='Add model card README',
)
print('lilt-base-GGUF README uploaded')
