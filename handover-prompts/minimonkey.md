# Handover: MiniMonkey (mx262/MiniMonkey)

## Goal

Add MiniMonkey VLM OCR to CrispEmbed. Zero code changes needed — runs
on the existing `internvl2_ocr` engine. Just needs GGUF conversion +
model registry + test.

## Model details

- **HuggingFace**: `mx262/MiniMonkey`
- **Architecture**: `internvl_chat` (InternVL-based, same as InternVL2)
- **License**: Check repo (likely MIT via InternVL base)
- **OCRBench**: 794 (strong for 2.2B)
- **Params**: ~2.2B total

## Architecture (identical to InternVL2)

| Component | Config |
|-----------|--------|
| Vision | InternViT-300M: 24L, 1024d, 16 heads, 448px, patch=14 |
| Pixel shuffle | downsample_ratio=0.5, 1024→256 tokens |
| Projector | 2-layer MLP (4096→2048) |
| LLM | InternLM2: 24L, 2048d, 16/8 GQA heads, inter=8192 |
| Vocab | 92553 tokens |
| RoPE | theta=1000000, dynamic scaling factor=2.0 |

## What to do

### 1. Download safetensors (~4.4 GB)

The model is a single `model.safetensors` file. Needs a machine with
>8GB RAM to download (the 8GB VPS OOMs during download). Use:

```bash
huggingface-cli download mx262/MiniMonkey --local-dir tmp/minimonkey
```

Or on a machine with more RAM:
```python
from huggingface_hub import snapshot_download
snapshot_download("mx262/MiniMonkey", local_dir="tmp/minimonkey")
```

### 2. Convert to GGUF

Use the existing InternVL2 converter — it already handles InternLM2
backbones, pixel shuffle, and the MLP projector:

```bash
python models/convert-internvl2-to-gguf.py \
    --model tmp/minimonkey \
    --output tmp/minimonkey-f16.gguf --dtype f16
```

Expected: ~4.4 GB F16 GGUF, ~850 tensors.

Note: The converter was recently fixed to handle `rope_scaling: null`
(commit in convert-internvl2-to-gguf.py). MiniMonkey has
`rope_scaling: {"factor": 2.0, "type": "dynamic"}` which is already
supported.

### 3. Quantize

```bash
./crispembed-quantize tmp/minimonkey-f16.gguf \
    /mnt/storage/gguf-models/minimonkey-q4_k.gguf q4_k
```

Expected: ~1.2-1.5 GB Q4_K.

### 4. Test

```bash
# Should load and show correct config
./test-internvl2-e2e /mnt/storage/gguf-models/minimonkey-q4_k.gguf

# Expected output:
#   vision: 24 layers, 1024d, 16 heads, patch=14, merge=2
#   llm: 24 layers, 2048d, 16/8 heads, inter=8192
```

### 5. Add to model registry

In `examples/cli/model_mgr.cpp`, add after the H2OVL entries:

```cpp
{"minimonkey",
 "minimonkey-q4_k.gguf",
 "https://huggingface.co/cstr/minimonkey-crispembed-GGUF/resolve/main/minimonkey-q4_k.gguf",
 "MiniMonkey 2.2B VLM OCR (InternViT-300M + InternLM2, OCRBench 794)", "~1400 MB", "mit",
 "https://huggingface.co/cstr/minimonkey-crispembed-GGUF"},
```

### 6. Upload GGUF to HuggingFace

Upload both F16 and Q4_K to `cstr/minimonkey-crispembed-GGUF`.

## Why it works without code changes

The `internvl2_ocr` engine reads all hyperparameters from GGUF metadata:
- `num_hidden_layers`, `hidden_size`, `num_attention_heads`, etc.
- Vision encoder is identical InternViT-300M
- LLM is InternLM2 (same as InternVL2's LLM backbone)
- Projector is the same 2-layer MLP with pixel shuffle
- Tokenizer is InternLM2's SentencePiece (92553 tokens, embedded in GGUF)

The only difference from stock InternVL2-2B is the vocab size (92553 vs
32000) and RoPE scaling (dynamic factor=2.0 vs 1.0). Both are handled
by the existing converter and engine.

## Key files

- `models/convert-internvl2-to-gguf.py` — converter (works as-is)
- `src/internvl2_ocr.{h,cpp}` — engine (works as-is)
- `examples/cli/model_mgr.cpp` — add registry entry
- `tests/test_internvl2_e2e.cpp` — test binary (works as-is)

## Blocker on 8GB VPS

The 4.4GB safetensors download OOMs `huggingface_hub` on the 8GB VPS.
Solutions:
1. Use `wget` with streaming (avoids loading into RAM)
2. Use `huggingface-cli download` (CLI tool, streams to disk)
3. Do it on a Kaggle notebook or larger machine

The conversion itself also needs ~8GB RAM (loading safetensors + writing
GGUF). May need the same workaround.
