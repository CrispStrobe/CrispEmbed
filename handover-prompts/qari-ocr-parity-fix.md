# Handover: Fix Qari-OCR Parity Bug

## Target machine

16 GB M1 Mac. The Qari-OCR model is ~4.7 GB F16 GGUF — fits comfortably.
Even the Q4_K (1.7 GB) reproduces the bug.
Build with: `cmake -B build -G Ninja && ninja -C build test-qwen2vl-diff`
(Apple Silicon native, no ccache needed, builds in ~2 min)

## NEW FINDING (2026-06-16): Bug is in the VISION ENCODER

Running the diff test on the VPS with Q4_K confirmed:
```
vis_merger: cos_min=0.056324 max_abs=5.36e+01 FAIL
```
cos=0.056 means the vision encoder output is essentially **garbage** — the
bug is NOT in the LLM decoder, NOT in preprocessing. The 32-layer ViT
produces wrong features which propagate through the merger to the LLM.

Two fixes were already committed:
1. `gguf_get_val_u32` crash on BOOL-typed `tie_word_embeddings` — FIXED
2. Diff test tensor name compatibility (pixel_values vs input_patches) — FIXED

## The bug

Qari-OCR (a Qwen2-VL-2B fine-tune for Arabic OCR) produces hallucinated
prompt text instead of OCR output.

- **PyTorch output**: `'This image contains the text "Hello World 2024".'`
- **CrispEmbed output**: `'Below is the plain text representation...'` (prompt echo)
- **PyTorch top-1 token**: `This` (logit 18.13)
- **CrispEmbed top-1 token**: `Below` (logit 14.19)

The divergence is in **prefill logits** — vision embeddings reach the LLM
but produce wrong attention patterns, causing the decoder to ignore the
image and echo its system prompt.

## Root cause hypothesis

Qari-OCR is based on **Qwen2-VL-2B** (not Qwen2.5-VL). The C++ engine
(`src/qwen2vl_ocr.cpp`) already has variant-aware code paths, but something
is still wrong. The Kaggle diff harness confirmed:
- Token IDs: 58 tokens, 33 image_pad, grid [1,6,22] — all match PyTorch
- Vision shapes: correct (132×1280, merger 33×1536)
- **Divergence starts at prefill logits** — the first generated token differs

Since the bug is CONFIRMED in the vision encoder (cos=0.056 at merger),
focus investigation here. Ranked causes:

1. **Vision embed_dim confusion**: Qwen2-VL config has BOTH `embed_dim=1280`
   (ViT block dim) AND `hidden_size=1536` (merger output / LLM input).
   The engine reads `vhp.hidden_size` at line 378. If the GGUF stores
   `qwen2vl.vision.hidden_size=1536` but the ViT blocks are 1280-wide,
   the QKV projections and FFN will read wrong-sized weight matrices.
   **CHECK**: print `vhp.hidden_size` vs actual weight shapes of
   `v.blk.0.attn_qkv.weight`. If H=1536 but weight is [3840,1280],
   that's the bug. Fix: use `embed_dim` for ViT block dim.

2. **Fullatt block indexes missing**: Qwen2-VL uses windowed attention
   on most layers + full attention on specific layers. If
   `qwen2vl.vision.fullatt_block_indexes` is missing from GGUF, ALL
   layers use full attention (wrong). Check converter line ~370.

3. **Vision RoPE theta**: Hardcoded `theta=10000.0` at line 326. Confirm
   this matches PyTorch's `VisionRotaryEmbedding.inv_freq`.

4. **Window size**: Qwen2-VL uses `window_size=112` pixels = 8 patches
   (112/14). Check `qwen2vl.vision.window_size` in the GGUF.

5. **Quantization artifact** (unlikely at cos=0.056): Q4_K quantization
   could cause some error, but cos=0.056 is WAY beyond quantization noise.
   Try with F16 GGUF to rule out.

## How to debug

### Step 1: Download models and reference

```bash
# Download from HuggingFace
huggingface-cli download cstr/qari-ocr-crispembed-GGUF qari-ocr-2b-f16.gguf
huggingface-cli download cstr/qari-ocr-crispembed-GGUF qari-ocr-ref.gguf
```

### Step 2: Run the diff test

```bash
# Build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build test-qwen2vl-diff

# Run — this compares per-layer vision + LLM output against PyTorch reference
./build/test-qwen2vl-diff qari-ocr-2b-f16.gguf qari-ocr-ref.gguf
```

The diff test loads the model, runs the vision encoder using the SAME input
patches as PyTorch (from ref.gguf), and compares every intermediate tensor.
The first layer where `cos_min < 0.999` is where the bug lives.

### Step 3: If vision parity passes, test LLM

The ref.gguf also contains `llm_embed`, `llm_layer_{i}`, and `llm_logits_last`.
The diff test checks these too. If vision passes but LLM fails, the bug is
in the LLM forward or the vision→LLM splicing.

### Step 4: Generate a fresh PyTorch reference if needed

```bash
# Dump fresh reference with full vision + LLM intermediates
python tools/dump_qwen2vl_reference.py \
    --model oddadmix/Qari-OCR-0.2.2.1-VL-2B-Instruct-merged \
    --image test_image.png \
    --output qari-ocr-ref-fresh.gguf \
    --max-vis-layers 32 \
    --max-llm-layers 28
```

This uses safetensors lazy loading (~4 GB peak RAM). Needs `pip install
safetensors gguf numpy Pillow huggingface_hub`.

## Key files

| File | Purpose |
|------|---------|
| `src/qwen2vl_ocr.cpp` | The engine. Vision: lines 400-600 (ViT blocks), 550-620 (merger). LLM: lines 1050-1250 (attention+FFN), 1400-1600 (generate). |
| `src/qwen2vl_ocr.h` | C++ API + internal structs (`vision_result`, `llm_result`) |
| `tests/test_qwen2vl_diff.cpp` | Per-layer diff test binary |
| `src/crispembed_diff.h` | Diff harness (loads ref GGUF, cos/max_abs comparison) |
| `src/image_preprocess.h` | Image preprocessing (smart_resize, patchify, normalize) |
| `models/convert-qwen2vl-to-gguf.py` | GGUF converter (handles both Qwen2-VL and Qwen2.5-VL) |
| `tools/dump_qwen2vl_reference.py` | PyTorch reference dumper |

## Architecture: Qwen2-VL-2B (Qari-OCR base)

```
Vision encoder (32 ViT blocks, embed_dim=1280, 16 heads):
  Conv3D patchify (3→1280, patch=14, temporal=2)
  + 2D rotary position embedding (cos/sin per h,w)
  32× pre-LayerNorm ViT block:
    LayerNorm → QKV (fused 3840=3×1280) + bias → 2D-RoPE
    → attention (windowed 112×112 or full at specific blocks)
    → residual
    LayerNorm → GELU MLP (fc1: 1280→5120, fc2: 5120→1280) + bias
    → residual
  Spatial merge (2×2 → 1, dim 1280→5120)
  → LayerNorm → FC1 (5120→1536) → GELU → FC2 (1536→1536)

LLM decoder (28 layers, hidden=1536, GQA 12Q/2KV):
  embed_tokens → splice vision embeds at image_pad positions
  28× pre-RMSNorm Qwen2 block:
    RMSNorm → Q/K/V (with bias!) + mRoPE → GQA attention → residual
    RMSNorm → SwiGLU FFN (gate+up+down, intermediate=8960) → residual
  RMSNorm → lm_head → logits → greedy decode
```

## Qwen2-VL vs Qwen2.5-VL differences (engine must handle)

| Feature | Qwen2-VL (Qari-OCR) | Qwen2.5-VL (german-ocr) |
|---------|---------------------|------------------------|
| Vision FFN | GELU fc1/fc2 | SwiGLU gate/up/down |
| Vision norm | LayerNorm (with bias) | RMSNorm (no bias) |
| Vision embed_dim | 1280 | 1280 |
| Vision hidden_size | 1536 (merger output) | 2048 (merger output) |
| LLM attention bias | Yes (Q/K/V/O) | No |
| LLM hidden | 1536, 28 layers | 2048, 36 layers |
| rope_theta | 1000000 | 1000000 |

The engine auto-detects the variant by checking `norm1_b != nullptr` (line 251).

## What NOT to change

- Don't break Qwen2.5-VL (german-ocr, nanonets) — they work perfectly.
- The variant detection (`is_qwen2_vl`) is correct.
- The converter handles both variants correctly.
- Token IDs and grid dimensions already match — don't debug preprocessing.

## Success criteria

1. `test-qwen2vl-diff qari-ocr-2b-f16.gguf qari-ocr-ref.gguf` passes
   all stages with `cos_min >= 0.999`.
2. Running `crispembed --ocr qari-ocr-2b-f16.gguf test_arabic.png`
   produces correct Arabic OCR text instead of prompt echo.
3. Qwen2.5-VL models (german-ocr, nanonets) continue to work correctly
   (no regression).
