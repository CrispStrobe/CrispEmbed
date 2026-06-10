# PP-FormulaNet-L Integration Handover

## Status
Texo-distill (20M, HGNetv2+MBart) is shipped. Next: PP-FormulaNet-L (181M, SAM-ViT+MBart).

## Model Details
- HF: `PaddlePaddle/PP-FormulaNet-L_safetensors` (Apache-2.0, safetensors, 181M params)
- Loads natively: `PPFormulaNetForConditionalGeneration.from_pretrained(...)`
- Config: `pp_formulanet` model type in HF transformers
- 398 tensors in model.safetensors

## Architecture

### Encoder: SAM-style ViT (from PaddleOCR's `Vary_VIT_B_Formula`)
- Patch embed: Conv2D(3→768, k=16, s=16), pos_embed (1, 48, 48, 768)
- 12 transformer blocks, each: PRE-LN → Attention → residual → PRE-LN → MLP → residual
- Attention: fused QKV (2304, 768), proj (768, 768)
- **Windowed attention** (layers 0,1,3,4,6,7,9,10): window_size=14, rel_pos_h/w (27, 64)
- **Global attention** (layers 2,5,8,11): full attention, rel_pos_h/w (95, 64)
- Window partitioning: pad to window_size multiple, reshape to windows, unpartition after attn
- Decomposed relative position: `attn += rel_h[:,:,:,:,None] + rel_w[:,:,:,None,:]`
- Neck: Conv1×1(768→256) + LayerNorm2d + Conv3×3(256→256) + LayerNorm2d
- Multi-modal projector: Conv3×3(256→512, s=2) + Conv3×3(512→1024, s=2)
  + Linear(1024→1024) + Linear(1024→512)
- Image: 768×768, output: (B, 12×12, 512) = (B, 144, 512) after projector

### Decoder: MBart PRE-LN (8 layers)
- embed_tokens (50000, 512), embed_positions (1026, 512), scale=sqrt(512)
- 8 layers: self_attn → cross_attn → FFN, all PRE-LN (LayerNorm before, residual skips)
- 16 attention heads, d_model=512, FFN=2048, GELU activation
- lm_head (50000, 512)

## Tensor naming (safetensors)
```
model.encoder.patch_embed.projection.{weight,bias}
model.encoder.pos_embed
model.encoder.layers.{i}.attn.qkv.{weight,bias}       # fused QKV
model.encoder.layers.{i}.attn.proj.{weight,bias}
model.encoder.layers.{i}.attn.rel_pos_h                # (27 or 95, 64)
model.encoder.layers.{i}.attn.rel_pos_w
model.encoder.layers.{i}.layer_norm1.{weight,bias}
model.encoder.layers.{i}.layer_norm2.{weight,bias}
model.encoder.layers.{i}.mlp.lin1.{weight,bias}
model.encoder.layers.{i}.mlp.lin2.{weight,bias}
model.encoder.neck.conv1.weight                         # (256, 768, 1, 1)
model.encoder.neck.conv2.weight                         # (256, 256, 3, 3)
model.encoder.neck.layer_norm1/2.{weight,bias}
model.encoder.multi_modal_projector.conv1.weight        # (512, 256, 3, 3)
model.encoder.multi_modal_projector.conv2.weight        # (1024, 512, 3, 3)
model.encoder.multi_modal_projector.linear_1.{weight,bias}
model.encoder.multi_modal_projector.linear_2.{weight,bias}
model.decoder.embed_tokens.weight                       # (50000, 512)
model.decoder.embed_positions.weight                    # (1026, 512)
model.decoder.layers.{i}.self_attn.{q,k,v}_proj.{weight,bias}
model.decoder.layers.{i}.self_attn.out_proj.{weight,bias}
model.decoder.layers.{i}.encoder_attn.{q,k,v}_proj.{weight,bias}
model.decoder.layers.{i}.encoder_attn.out_proj.{weight,bias}
model.decoder.layers.{i}.{self_attn,encoder_attn,final}_layer_norm.{w,b}
model.decoder.layers.{i}.fc1.{weight,bias}              # (2048, 512)
model.decoder.layers.{i}.fc2.{weight,bias}              # (512, 2048)
lm_head.weight                                          # (50000, 512)
```

## New ggml work required
1. **Window partitioning**: reshape (B, H, W, C) → (B*nW, ws, ws, C)
   - Pad H,W to multiples of window_size (14)
   - Reshape to (B, H/ws, ws, W/ws, ws, C) → transpose → (B*nW, ws, ws, C)
   - After attention: reverse reshape + crop padding
2. **Decomposed relative position bias**: 
   - rel_pos_h (2*ws-1, head_dim) indexed by relative row distance
   - rel_pos_w (2*ws-1, head_dim) indexed by relative col distance  
   - `attn[b,i,j] += q[b,i] @ rel_pos_h[row_i - row_j + ws-1]`
   - Added as: einsum("bhwc,hkc->bhwk") for rows, "bhwc,wkc->bhwk" for cols
3. **LayerNorm2d**: norm over channel dim (not last dim)
4. **Conv neck + projector**: standard Conv2D ops (already have in ppformulanet_ocr.cpp)

## What can be reused
- MBart PRE-LN decoder: identical to ppformulanet_ocr.cpp (wider: 512 vs 384, 8L vs 2L)
- Conv2D CPU: `conv2d_cpu` from ppformulanet_ocr.cpp
- Greedy decode loop: identical
- Image preprocessing: similar UniMERNet pipeline (768×768 instead of 384×384)
- GGUF converter pattern: follow convert-ppformulanet-to-gguf.py
- Reference dumper pattern: follow dump_ppformulanet_reference.py
- crispembed-diff: encoder + decoder layer comparison

## Source code references
- PaddleOCR encoder: `/mnt/volume1/PaddleOCR/ppocr/modeling/backbones/rec_vary_vit.py`
- PaddleOCR decoder: `/mnt/volume1/PaddleOCR/ppocr/modeling/heads/rec_ppformulanet_head.py`
- HF transformers: `transformers/models/pp_formulanet/modeling_pp_formulanet.py`
- PaddleOCR config: `/mnt/volume1/PaddleOCR/configs/rec/PP-FormuaNet/PP-FormulaNet-L.yaml`

## Estimated sizes
- F32: ~690 MB (too large)
- F16: ~345 MB
- Q8_0: ~180 MB
- Q4_K: ~100 MB (target for desktop)
