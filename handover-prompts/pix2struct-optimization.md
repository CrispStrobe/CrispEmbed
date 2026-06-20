# Handover: Pix2Struct Performance Optimization

## Goal

Optimize `src/pix2struct.cpp` (709 lines) from fully CPU-scalar to ggml graph
compute with proper KV caching. Currently the slowest architecture per-param
in the codebase — no SIMD, no GPU, O(T²) recompute per decode step.

## Current State

Pix2Struct is a 282M encoder-decoder model (T5-style) for document/chart
understanding. 17 fine-tuned variants available. Architecture:

- **Encoder**: patch_projection + row/col position embeddings → 12 T5 layers
  (RMSNorm → self-attn with T5 relative bias → RMSNorm → SwiGLU FFN)
- **Decoder**: token embed → 12 T5 layers
  (RMSNorm → causal self-attn + T5 rel bias → RMSNorm → cross-attn → RMSNorm → SwiGLU FFN)
  → final norm → LM head → greedy decode

### What's wrong

1. **All compute is CPU-scalar** — `linear()`, `self_attn()`, `cross_attn()`,
   `swiglu_ffn()` are all hand-written scalar loops. No ggml graphs at all.
   Linear is O(N*M) with no SIMD.

2. **No KV cache** — `decoder_step()` takes `past_tokens` (all previous token
   embeddings) and reprocesses the **entire sequence** through all 12 layers
   at every decode step. Step N processes N tokens × 12 layers × 3 attention
   blocks = O(N²) total work.

3. **Re-dequantizes weights every call** — `to_f32()` re-dequantizes the
   same weight tensor from Q4_K/Q8_0 to F32 on every layer, every step.
   No caching.

4. **Cross-attention K/V not pre-computed** — encoder output is fixed, but
   cross-attn K/V are recomputed from encoder output at every decode step
   for every layer.

### Performance impact

On a 4-thread Xeon CPU:
- Encoder: ~2-5s for a typical chart image (variable patch count)
- Decoder: ~0.5s per step × 50 steps × growing T = several minutes total
- The O(T²) recompute makes it catastrophic for long outputs

## Optimization Plan

### Phase 1: ggml graph for encoder (~3x speedup)

Build the 12-layer encoder as a single ggml graph:
- `ggml_mul_mat` for all linear projections
- `ggml_flash_attn_ext` for self-attention (T5 rel bias as mask)
- `ggml_silu` + `ggml_mul` for SwiGLU
- `ggml_rms_norm` for norms
- Use `ggml_backend_sched` for GPU dispatch

Pattern to follow: `ppformulanet_l_ocr.cpp` lines 431-620 (SAM-ViT encoder
in ggml graph with windowed attention).

Key difference from standard ViT: T5 relative attention bias. The bias is a
function of relative position (buckets), not absolute position. Pre-compute
the bias matrix `(T, T, n_heads)` on CPU, pass as `ggml_flash_attn_ext` mask.

### Phase 2: KV cache for decoder (~10x speedup)

Add per-layer self-attention and cross-attention KV caches:

**Self-attention KV cache** (grows each step):
- After step 0: cache K/V for token 0 in all 12 layers
- Step N: compute K/V for token N only, append to cache, attend to full cache
- Pattern: `internvl2_ocr.cpp` KV cache (F16 ggml tensors, `ggml_view` +
  `ggml_cpy` writes)

**Cross-attention K/V pre-computation** (fixed, computed once):
- After encoder runs, project encoder output through cross-attn K/V weights
  for all 12 layers, store as F32/F16 tensors
- Each decode step only computes cross-attn Q from the decoder hidden state
- Pattern: `math_ocr.cpp` lines 461-480 (cross-attn K/V pre-compute)

### Phase 3: Weight dequant caching

Replace the `to_f32()` calls with `core_cpu::DequantCache`:
```cpp
#include "core/cpu_ops.h"
core_cpu::DequantCache dc;
const float *w = dc.get(tensor);  // cached after first call
```

Pattern: `smoldocling_ocr.cpp` lines 40-50.

## Files to modify

1. `src/pix2struct.cpp` — main implementation (709 lines, rewrite ~500)
2. `src/pix2struct.h` — add KV cache fields to context struct
3. `tests/test_pix2struct_diff.cpp` — parity test (if not exists, create)
4. `tools/dump_pix2struct_reference.py` — reference dumper (if not exists, check)

## Parity testing

The dev guide requires crispembed-diff parity testing. Pattern:

1. **Python reference**: `tools/dump_pix2struct_reference.py` (may already exist)
   dumps per-layer encoder + decoder intermediates to GGUF
2. **C++ diff**: Load model + reference GGUF, compare each intermediate
3. **Key stages**: `enc_layer_{i}`, `enc_final`, `cross_kv`, `dec_layer_{i}`,
   `logits_step_0`

Existing models for testing:
- `/mnt/storage/gguf-models/pix2struct-*` (check with `ls`)
- Use chart/table images from `tests/` or synthetic

## Constraints

- **8GB RAM VPS** — cannot load F16 for models >1B. Use q4_k/q8_0.
- **Always work in a worktree**: `git worktree add .claude/worktrees/feat-pix2struct-perf -b feat/pix2struct-perf`
- **Build with ccache**: `CCACHE_DIR=/mnt/volume1/.ccache ninja -j1`
- **-j1 for builds** — RAM is tight, -j2 can OOM with other processes
- **Never edit files on main** — always in the feature worktree

## Success criteria

1. Encoder fully in ggml graph (GPU-ready)
2. KV cache eliminates O(T²) recompute — each decode step is O(T) not O(step²)
3. Cross-attention K/V computed once, reused across all decode steps
4. Per-layer parity cos≥0.999 against Python reference
5. End-to-end output matches original (same generated text)
6. A/B benchmark showing speedup (expect 5-10x for 50-token generation)
