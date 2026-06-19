# Handover: Extract shared VLM attention + decoder to core/ headers

## Goal

Extract the duplicated KV-cached attention and Llama-family autoregressive
decode loop from 7+ VLM OCR engines into two shared headers:

1. `src/core/vlm_attention.h` — MHA with KV cache, RoPE, GQA repeat
2. `src/core/vlm_decoder.h` — Llama-family autoregressive decode loop

This is Phase 2-3 of the refactoring started by `core/cpu_ops.h` (Phase 1),
which already extracted `to_f32`, `layernorm`, `rmsnorm`, `linear`, `gelu`,
`silu`, etc. into a shared header with 88 unit tests. Follow the same pattern.

## Context

Read `/home/claudeuser/crispasr-crispembed-dev.md` first for build/test conventions.

### What Phase 1 did (reference for the pattern)

- Created `src/core/cpu_ops.h` with shared scalar helpers
- Replaced duplicated functions in 6 engines (surya_det, got_ocr, ppformulanet_l,
  ppformulanet, deepseek_ocr2, mixtex_ocr) — deleted 728 lines
- Added `tests/test_core_cpu_ops.cpp` with 88 unit tests
- Commit: `0566445 refactor: extract shared CPU helpers to core/cpu_ops.h`

### Engines that have duplicate attention/decode code

| Engine | File | RoPE | GQA | Heads | Norm | Activation | Special |
|--------|------|------|-----|-------|------|------------|---------|
| granite_vision | `src/granite_vision_ocr.cpp` | interleaved | 8/2 | 16q/2kv | RMSNorm | SiLU | Granite multipliers (embed=12, residual=0.22, logits=8) |
| internvl2 | `src/internvl2_ocr.cpp` | interleaved | yes | varies | RMSNorm | SiLU | InternLM2 style |
| qwen2vl | `src/qwen2vl_ocr.cpp` | mRoPE (multi-dim) | yes | varies | RMSNorm | SiLU | mRoPE 3D positions |
| got_ocr | `src/got_ocr.cpp` | interleaved | yes | varies | RMSNorm | SiLU | Standard Qwen2 |
| smoldocling | `src/smoldocling_ocr.cpp` | neghalf | 9/3 | 9q/3kv | RMSNorm | SiLU | SmolLM2, separate lm_head |
| lightonocr | `src/lightonocr.cpp` | ? | yes | varies | RMSNorm | SiLU | Pixtral ViT + Qwen3 |
| deepseek_ocr2 | `src/deepseek_ocr2.cpp` | ? | yes | varies | RMSNorm | SiLU | MoE decoder |

### What to extract

**`core/vlm_attention.h`** should provide:

```cpp
namespace core_vlm {

enum class RoPEStyle { NEGHALF, INTERLEAVED, MROPE };

// Apply RoPE to Q/K vectors in-place
void apply_rope(float * q, int n_heads, int head_dim,
                int position, float theta, RoPEStyle style);

// Single-token GQA self-attention with KV cache
// - Stores new K,V into cache at position n_past
// - Computes attention over all past K,V
// - Returns [n_heads * head_dim] output
void gqa_attn_step(
    const float * q, const float * k_new, const float * v_new,
    int n_heads, int n_kv_heads, int head_dim,
    float * kv_cache, int max_seq, int n_past,
    int layer_idx, int n_layers,
    float * output);

}  // namespace core_vlm
```

**`core/vlm_decoder.h`** should provide:

```cpp
namespace core_vlm {

struct DecoderConfig {
    int hidden_size;
    int n_heads, n_kv_heads, head_dim;
    int intermediate_size;
    int n_layers;
    int vocab_size;
    float rms_eps;
    float rope_theta;
    RoPEStyle rope_style;
    // Granite-specific multipliers (1.0 = no effect)
    float embed_multiplier = 1.0f;
    float residual_multiplier = 1.0f;
    float logits_divisor = 1.0f;
    bool tie_word_embeddings = false;
};

// Run one decode step: embed → N layers (RMSNorm→GQA→RMSNorm→SwiGLU) → logits
// Handles KV cache, RoPE, GQA internally.
// Returns argmax token ID, writes logits to `logits_out` if non-null.
int decode_step(
    const float * token_embed,  // [hidden_size]
    const DecoderConfig & cfg,
    // Weight accessor: returns f32 pointer for named tensor
    std::function<const float*(const std::string&)> get_weight,
    float * kv_cache, int max_seq, int n_past,
    float * logits_out,  // [vocab_size] or nullptr
    bool skip_logits = false);

}  // namespace core_vlm
```

### Key variations to handle

1. **RoPE style**: neghalf (`[-x_hi, x_lo]` rotation, SmolDocling/LFM2) vs interleaved (pairs, Granite/InternVL2) vs mRoPE (3D positions, Qwen2VL)
2. **GQA repeat factor**: varies by model (n_heads / n_kv_heads)
3. **Granite multipliers**: embed×12, residual×0.22, logits÷8 — other models use 1.0
4. **Tied vs separate lm_head**: SmolDocling has separate lm_head, some models tie embed
5. **MoE decoder** (DeepSeek-OCR-2): has expert routing instead of simple FFN — may need to stay engine-specific, or the decode_step takes an optional FFN callback
6. **Weight naming**: each engine has different GGUF tensor names — the `get_weight` callback abstracts this

### How to proceed

1. **Read all 7 engine files** — focus on the `*_decode_step` / `*_llm_decode_step` functions and KV cache management
2. **Diff the implementations** — note every variation in a table
3. **Design the shared API** — must handle all variations without performance overhead
4. **Write headers** with inline implementations (header-only, like cpu_ops.h)
5. **Write unit tests** (`tests/test_core_vlm_attention.cpp`) — synthetic weights, verify KV cache indexing, RoPE correctness for all styles, GQA repeat
6. **Replace in ONE engine first** (smoldocling is cleanest/simplest) — verify output is bit-identical
7. **Replace in remaining engines** one at a time, verifying each

### Build & test

```bash
cd /mnt/volume1/CrispEmbed
git worktree add .claude/worktrees/feat-core-vlm -b feat/core-vlm
cd .claude/worktrees/feat-core-vlm
git submodule update --init --recursive
mkdir -p build && cd build
CCACHE_DIR=/mnt/volume1/.ccache cmake .. -G Ninja \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DBUILD_SHARED_LIBS=ON
CCACHE_DIR=/mnt/volume1/.ccache ninja -j1  # -j1 for 8GB RAM
```

After replacing in each engine, verify:
- `ninja -j1` compiles clean (no warnings in changed files)
- Existing parity tests still pass (if model GGUFs are available)
- The shared header has comprehensive unit tests

### Files to read

- `src/core/cpu_ops.h` — Phase 1 reference (follow same pattern)
- `tests/test_core_cpu_ops.cpp` — Phase 1 test reference
- `src/smoldocling_ocr.cpp` — simplest VLM decoder (start here)
- `src/granite_vision_ocr.cpp` — has Granite multipliers
- `src/got_ocr.cpp` — standard Qwen2 decoder
- `src/internvl2_ocr.cpp` — InternLM2 decoder
- `src/qwen2vl_ocr.cpp` — mRoPE decoder (most complex)
- `src/lightonocr.cpp` — Pixtral + Qwen3
- `src/deepseek_ocr2.cpp` — MoE decoder (may be excluded)

### CrispASR reference

CrispASR's `src/core/bpe.h` and `src/core/attention.h` show the shared-header pattern used there. The `attention.h` has `kv_self_attn` which handles KV cache + GQA — read it for inspiration but don't copy directly (CrispEmbed's VLM decoders have different calling conventions).
