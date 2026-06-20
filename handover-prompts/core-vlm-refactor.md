# Core VLM Refactor — COMPLETED

Phase 2 of the core refactoring (Phase 1: `cpu_ops.h`, commit `0566445`).

**Commit:** `c730539 refactor: extract shared VLM scalar helpers to core/vlm_attention.h`

## What was done

Extracted duplicated scalar RoPE, GQA attention, KV cache management, and
SwiGLU FFN from the 2 pure-scalar VLM engines into `src/core/vlm_attention.h`.

### Files created

- `src/core/vlm_attention.h` — header-only, namespace `core_vlm`, `static inline`
  - `apply_rope()` — NEGHALF (SmolLM2/GPT-NeoX) and INTERLEAVED (Llama/Granite)
  - `alloc_kv_cache()`, `kv_k_offset()`, `kv_v_offset()` — flat F32 KV cache
  - `gqa_attn_step()` — single-token GQA self-attention with KV cache write
  - `swiglu_ffn()` — SwiGLU feed-forward step
- `tests/test_core_vlm_attention.cpp` — 23 test functions, 97 assertions

### Files modified

- `src/smoldocling_ocr.cpp` — replaced inline RoPE/attention/SwiGLU (~86 lines deleted)
- `src/granite_vision_ocr.cpp` — same (~82 lines deleted), Granite multipliers preserved in caller
- `CMakeLists.txt` — added `test-core-vlm-attention` target

### Net: 134 lines deleted, 37 added in engines (-97 lines)

## Verification

- Unit tests: 97/97 pass (both RoPE styles, GQA with repeat, KV cache, SwiGLU)
- Phase 1 regression: 88/88 pass (core_cpu tests)
- SmolDocling live smoke: correct OCR output end-to-end
- Granite Vision live: OOM on 8GB VPS (2B model), not a code issue
- Clean build: all 234 targets

## Why only 2 engines, not 7

The other 5 VLM engines use ggml graph ops (`ggml_rope_ext`, `ggml_flash_attn_ext`,
`ggml_mul_mat`) — one-line calls with nothing to extract. Only smoldocling and
granite_vision had genuinely duplicated ~200-line scalar decode loops.

## What remains (not planned, for reference)

- **Replace local helpers** — `sd_rmsnorm`/`sd_linear`/`sd_silu` and `gv_rmsnorm`/
  `gv_linear`/`gv_silu` could use `core_cpu::` equivalents (low priority, they're
  used by the vision encoder too so the local copies are still convenient)
- **Unified `decode_step()` abstraction** — deferred as premature with only 2 scalar
  engines; revisit if a 3rd lands
- **Granite live parity test** — needs 16GB+ RAM machine
