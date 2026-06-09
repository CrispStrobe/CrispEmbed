# PP-FormulaNet-L: Full ggml Graph Encoder Handover

## Status
Hybrid encoder working (80s, cos=0.990). Need full-graph encoder (~30s target).

## Current architecture
Per-layer hybrid: ggml graph for QKV+MLP, CPU for attention with decomposed rel_pos.
Bottleneck: 4 global layers × 2304² attention on CPU (~55s of the 80s total).

## What needs to be built: `build_full_layer_graph`

A single ggml graph per layer that does everything:
- Pre-LN → fused QKV → split Q/K/V → attention → proj → residual → LN → MLP → residual

### Window handling (outside graph)
- `window_partition(hidden, win_out, nP=48, ws=14, C=768)`:
  Rearrange [N, C] from raster order to window order [wN*nWindows, C]
  where nWindows=16 (48 padded to 56, 56/14=4, 4×4=16), wN=196
- `window_unpartition(win_in, hidden, nP, ws, C)`: reverse
- For global layers: no rearrangement (nWindows=1, T=N=2304)

### Attention with windows as batch dim
- Q, K, V: [hd, wN, nHeads*nWindows] — windows folded into batch dim with heads
- `ggml_mul_mat(K, Q)` → scores [wN, wN, nHeads*nWindows]
- This processes all 16 windows × 12 heads = 192 attention operations in one batched matmul

### Decomposed RPE in-graph (Granite pattern)
The key insight from CrispASR/src/granite_nle.cpp:964-971:
```cpp
rpe_blk_4d = ggml_reshape_4d(ctx, rpe, hd, blk_len, blk_len, 1);
Q_blk_4d = ggml_reshape_4d(ctx, Q, hd, 1, blk_len, n_heads);
pos_bias = ggml_mul_mat(rpe_blk_4d, Q_blk_4d); // [blk_len, blk_len, n_heads]
```

For SAM's decomposed RPE (two tables, H and W):
1. RPE tables: precomputed [hd, aH, aH] and [hd, aW, aW] in ggml col-major
2. Compute H-bias: `ggml_mul_mat(rp_h_4d, Q_h_4d)` → [aH, ?, ?, batch]
3. Compute W-bias: `ggml_mul_mat(rp_w_4d, Q_w_4d)` → [aW, ?, ?, batch]
4. Combine via reshape + broadcast add:
   - Reshape scores to [aW_k, aH_k, wN_q, batch]
   - H-bias broadcasts over aW_k: [1, aH_k, wN_q, batch]
   - W-bias broadcasts over aH_k: [aW_k, 1, wN_q, batch]
   - `ggml_add` broadcasts the size-1 dims

### RPE table format
At init, `reformat_rp_table` transposes from row-major [(q*aH+k)*hd+d] to
ggml col-major [hd, k, q] where element (d, k, q) is at d + k*hd + q*aH*hd.

### Graph inputs
- `layer_input`: [C, T] where T = wN*nWindows for windowed, N for global
- `rp_h`: [hd, aH, aH] static precomputed
- `rp_w`: [hd, aW, aW] static precomputed

### Key ggml ops needed
- `ggml_mul_mat(w, x)`: linear projections and attention matmuls
- `ggml_norm(x, eps)`: layer normalization
- `ggml_gelu(x)`: GELU activation
- `ggml_soft_max(scores)` or `ggml_soft_max_ext(scores, mask, scale, 0)`: softmax
- `ggml_reshape_3d/4d`, `ggml_permute`, `ggml_cont`: tensor reshaping
- `ggml_view_2d`: QKV split
- `ggml_add` with broadcasting: bias addition
- `ggml_scale`: attention scaling

### Verification approach
1. Run on synthetic test image with F32 model
2. Compare per-layer output against /tmp/ppfnl-ref.gguf reference
3. Target: cos >= 0.990 (matching current hybrid)
4. If layer diverges: use diff harness to find first diverging layer

### Previous attempt issues
- Full [N,N,nh] bias tensor with -1e9f masking for windowed layers → cos=0.898
  Bug: masking doesn't match HF's window partition (padding behavior differs)
  Fix: actual window partition (not masking)
- QKV recomputed in separate graphs → potential float accumulation mismatch
  Fix: single graph per layer, no QKV recomputation

### Files to edit
- `/mnt/volume1/CrispEmbed-ppfn/src/ppformulanet_l_ocr.cpp`
  - Add `window_partition`, `window_unpartition`, `reformat_rp_table` helpers
  - Add `build_full_layer_graph` function
  - Modify `run_encoder_graph` to use new function
  - Keep `run_encoder` (CPU scalar) as fallback

### Reference code
- Granite RPE pattern: `/mnt/volume1/CrispASR/src/granite_nle.cpp:946-986`
- MPNet static bias: `/mnt/volume1/CrispEmbed-ppfn/src/crispembed.cpp:728-894`
- Math OCR DeiT encoder graph: `/mnt/volume1/CrispEmbed-ppfn/src/math_ocr.cpp:281-338`
