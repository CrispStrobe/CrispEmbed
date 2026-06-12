# GLM-OCR Architecture Notes (for PLAN.md blueprint)

## Source
- Model: zai-org/GLM-OCR (0.9B, MIT)
- GGUF: ggml-org/GLM-OCR-GGUF (Q8_0: 950 MB, F16: 1.79 GB) — TEXT-ONLY, separate mmproj
- OmniDocBench #1: 94.62 score

## Architecture

### Vision (CogViT, 24L/1024d/16H)
- Patch embed: Conv3D [1024, 3, 2, 14, 14] (temporal_patch_size=2, 336px)
- Per-layer: RMSNorm(1024) → fused QKV(3072,1024) + Q/K RMSNorm(64) → attention → proj + bias
- MLP: SwiGLU (gate+up+down) all with biases
- post_layernorm(1024) at end
- Spatial downsample: Conv2D [1536, 1024, 2, 2] — learned (NOT pixel unshuffle!)
- Merger: proj(1536→1536) + SwiGLU(1536→4608→1536) + LayerNorm(1536)

### LLM (GLM-0.5B, 16L/1536d/16H/8KV)
- Separate Q/K/V, no biases
- Q dim = 2048 (16×128), K/V dim = 1024 (8×128) — Q upscales from hidden=1536!
- mRoPE sections=[16,24,24], theta=10000
- SwiGLU FFN (4608)
- RMSNorm eps=1e-5
- vocab=59392, head_dim=128
- nextn_predict_layers=1 (multi-token prediction head)

### Key differences from InternVL2
- Vision: RMSNorm+SwiGLU (not LayerNorm+GELU), Q/K norm, Conv2D downsample
- LLM: mRoPE (not standard RoPE), Q upscale, MTP head
- More like Qwen2VL than InternVL2 in architecture
- Separate mmproj pattern in llama.cpp

### Tensor names (525 total in model.safetensors)
- Vision: model.visual.blocks.{i}.{attn.qkv,attn.proj,mlp.*,norm1,norm2,attn.q_norm,attn.k_norm}
- Merger: model.visual.{downsample,merger.{proj,gate_proj,up_proj,down_proj,post_projection_norm}}
- LLM: model.language_model.layers.{i}.{self_attn.{q,k,v,o}_proj,mlp.*,input_layernorm,post_attention_layernorm}
- Global: model.language_model.embed_tokens, lm_head

### Effort estimate: 3-4 days
- New C++ engine (vision different from InternVL2, LLM uses mRoPE like Qwen2VL)
- Can reuse mRoPE code from qwen2vl_ocr.cpp
- Converter + reference dumper + parity test
