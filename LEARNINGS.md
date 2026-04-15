# CrispEmbed — Technical Learnings

## ggml GQA broadcasting (critical for decoder models)

`ggml_mul_mat` natively broadcasts ne[2] when `b->ne[2] % a->ne[2] == 0`.
For GQA (16 Q heads, 8 KV heads): **do NOT explicitly repeat K/V**.
`ggml_repeat` tiles `[h0..h7, h0..h7]` which is WRONG for GQA (should
be `[h0,h0,h1,h1,...]`). Just let mul_mat broadcast — it handles the
interleaved head mapping correctly internally.

Also: after attention, reshape to `q_dim = n_heads × head_dim` (NOT
`hidden_size`). For GQA models, q_dim ≠ hidden_size (e.g. 2048 vs 1024).

## BERT post-LN vs pre-LN

BERT uses post-LayerNorm: `attn → residual_add → LN → FFN → residual_add → LN`.
Many newer models (GPT, LLaMA) use pre-LN. Getting this wrong produces
output that looks plausible but has completely wrong magnitudes.

## RoPE application order

For Qwen3: RoPE is applied on `[head_dim, n_heads, T]` tensor (BEFORE
permute to `[head_dim, T, n_heads]`). `ggml_rope_ext` requires ne[2]=T
(the position dimension), which matches the unpermuted layout. Applying
RoPE after permute crashes with dimension mismatch.

At position 0, RoPE is identity (cos=1, sin=0), so position-0 values
match regardless of whether RoPE is applied. Debug with position > 0
to verify RoPE correctness.

## Tokenizer types for embedding models

| Model family | Tokenizer | Implementation |
|---|---|---|
| BERT/MiniLM/GTE | WordPiece | Greedy longest-match with ## prefix |
| XLM-RoBERTa/E5/Arctic/PIXIE | SentencePiece Unigram | Viterbi DP (NOT bigram merge) |
| Qwen3/Octen/F2LLM | GPT-2 BPE | core_bpe byte-level BPE with merges |
| Gemma3/Harrier-270M | SentencePiece BPE | BPE merges with ▁ space marker + BOS/EOS |

Auto-detected from GGUF metadata: `tokenizer.ggml.type` (0=WP, 1=BPE, 2=SP)
or heuristic (vocab > 100K → SentencePiece).

### Critical: SentencePiece Unigram needs Viterbi, not bigram merge

The llama.cpp-style bigram merge (priority queue, highest-score-first)
does NOT produce correct tokenization for Unigram models like XLM-R.
Example: "▁world" exists as token 8999, but bigram merge breaks it into
["▁w", "or", "ld"] because greedy pair merging can't find the global optimum.

**Viterbi DP**: For each position i, try all vocab tokens ending at i,
pick the segmentation with the highest total score. O(n × max_token_len).
This matches HuggingFace's `tokenizers` library exactly.

### SentencePiece BPE vs GPT-2 BPE

These are different tokenizer families with different pre-processing:
- GPT-2 BPE: byte-level encoding (spaces → Ġ), no BOS/EOS by default
- SentencePiece BPE (Gemma): spaces → ▁ (U+2581), BOS/EOS tokens

### Vocab scores for SentencePiece

SentencePiece Unigram models need per-token scores for Viterbi. These come from:
1. `tokenizer.sp_model.GetScore(i)` — but not available for all tokenizer classes
2. `tokenizer.json` → `model.vocab` → list of `[token, score]` pairs

If scores are missing (all zeros), the tokenizer degenerates to random merging.

## Per-op debugging methodology

Same as CrispASR: dump every intermediate tensor from BOTH HF reference
and our ggml graph, compare at each stage. The divergence point identifies
the exact broken operation. For Octen-Embedding-0.6B, this revealed:
- input_ln: MATCH
- q_proj/k_proj: MATCH
- q_norm/k_norm: MATCH
- o_proj: MISMATCH → GQA repeat was wrong
- Fix: remove ggml_repeat, let mul_mat broadcast → MATCH

## RoBERTa/XLM-R position embedding offset

RoBERTa-family models (XLM-R, PIXIE-Rune, arctic-embed-l-v2) offset position
IDs by `padding_idx + 1 = 2`. Position IDs for a 4-token sequence are
`[2, 3, 4, 5]`, not `[0, 1, 2, 3]`. Position embedding index 1 is all-zeros
(padding), index 0 is low-norm. Getting this wrong produces ~0.74 cosine sim
instead of 0.999.

Stored as `bert.position_offset` in GGUF metadata.

## Gemma3 architecture specifics

Gemma3 (Harrier-270M) differs from Qwen3/LLaMA in several critical ways:

1. **RMSNorm uses `(1 + weight)`**: Gemma3 RMSNorm computes
   `output * (1.0 + weight)` instead of `output * weight`. The stored weights
   do NOT include the +1 offset. Missing this makes all layer outputs wrong.

2. **Embedding scale**: Token embeddings are multiplied by `sqrt(hidden_size)`.
   The exact value is stored in `embed_tokens.embed_scale` (f16 precision:
   `sqrt(640) ≈ 25.25` not `25.298`).

3. **Extra norms**: 4 norms per layer (not 2):
   - `input_layernorm` → before attention
   - `post_attention_layernorm` → after attention, BEFORE residual add
   - `pre_feedforward_layernorm` → before FFN
   - `post_feedforward_layernorm` → after FFN, BEFORE residual add

4. **Attention scaling**: Uses `query_pre_attn_scalar` (= head_dim) instead
   of `sqrt(head_dim)`. Scale = `1/sqrt(qpas)`.

5. **gelu_pytorch_tanh**: Activation function; ggml_gelu uses tanh approx.

6. **head_dim != hidden_size/n_heads**: Gemma3 has head_dim=256, hidden=640,
   n_heads=4. Standard calculation gives 160, but explicit head_dim is 256.

7. **SentencePiece BPE tokenizer**: Uses ▁ space marker (not GPT-2 Ġ),
   needs BOS(2) at start and EOS(1) at end.

## Quantization notes

### Python gguf vs C++ quantizer

The Python `gguf` library (`pip install gguf`) only implements quantization
for basic types: Q4_0, Q5_0, Q5_1, Q8_0. K-quants (Q4_K, Q5_K, Q6_K) are
listed in the enum but `quantize_blocks` raises `NotImplementedError`.

Additionally, the Python library's string array handling in GGUFReader/GGUFWriter
can corrupt metadata when copying GGUF files — we observed Q8_0 models from the
Python quantizer producing cos=0.78 vs the same model's F32, while the C++ quantizer
produces cos=0.9997.

**Use the C++ quantizer for all quantization.** It calls ggml's native
`ggml_quantize_chunk` which supports all types including K-quants.

### Embedding tables and aggressive quantization

Token embedding tables (`token_embd.weight`) are very sensitive to quantization.
Quantizing them to Q4_K degrades output quality significantly (cos drops from
0.999 to 0.71 for some models). The CrispEmbed quantizer keeps embedding tables
at F32 for Q4_K/Q5_K; only Q8_0 and F16 are allowed to touch them.

### K-quant fallback chain

K-quants (Q4_K/Q5_K/Q6_K) require row widths divisible by 256. Many embedding
model tensors have rows of 384 or 768 which aren't 256-aligned. The quantizer
falls back: Q4_K→Q4_0, Q5_K→Q5_0, Q6_K→Q8_0. This means small-dim models
get Q4_0 instead of Q4_K for most tensors.

### ggml_get_rows for quantized embeddings

The BERT encoder must use `ggml_get_rows` (ggml graph op) for embedding table
lookup, not manual `ggml_backend_tensor_get` with float pointer arithmetic.
`ggml_get_rows` handles dequantization internally and works with any tensor type.
Manual CPU-side extraction assumes F32 layout and crashes on quantized models.

## Server performance: buffer reuse

The biggest server-mode optimization is reusing `graph_buf` and `work_buf` across
encode calls. Without this, every request allocates ~50-200MB (graph context +
compute workspace), causing 3x overhead from malloc/free.

With buffer reuse: gte-small goes from 8.8 to 27.8 texts/sec (3.2x improvement).

## BLAS/MKL for embedding models

BLAS (OpenBLAS/MKL) provides minimal benefit for embedding inference because:
- Quantized kernels (Q8_0/Q4_K) use ggml's SIMD paths, not BLAS
- BERT encoder matrices are moderate-sized (384x384 to 1024x4096)
- BLAS overhead dominates for small matrices

For CPU speed: use Q8_0 quantization. For GPU: build with `-DGGML_CUDA=ON` or
`-DGGML_VULKAN=ON` — the `ggml_backend_sched` dispatcher handles offloading.

## ggml_backend_sched with CPU-only

When using `ggml_backend_sched` in CPU-only mode, calling it repeatedly with
different graphs causes segfaults because the scheduler holds stale tensor
references from freed graph contexts. Solution: only create the scheduler when
a GPU backend is detected (`!ggml_backend_is_cpu(backend)`). For CPU-only,
direct `ggml_graph_compute` with a persistent work buffer is faster anyway.

## Windows build

Windows users often forget `--recursive` when cloning. The CMakeLists.txt now
checks for `ggml/CMakeLists.txt` existence and prints a helpful error message.
Build scripts (`build-windows.bat`, `build-vulkan.bat`, `build-cuda.bat`) auto-
detect VS2022 and Vulkan/CUDA SDKs.

## ggml operator fusion — what exists, what doesn't

### Existing fused ops (backend-specific)

**CUDA** (automatic when graph patterns match):
- RMSNorm + Mul (`ggml_cuda_op_rms_norm_fused`)
- RMSNorm + Mul + Add (`ggml_cuda_op_rms_norm_fused_add`)
- Multi-Add (up to 8 chained adds → 1 kernel)
- FFN gate: MUL_MAT + ADD + MUL_MAT + ADD + GLU → 1 kernel
- RoPE + SetRows fused
- Unary + Mul (SILU/Sigmoid/Softplus)

**Vulkan**: Add + RMSNorm (controlled by `GGML_VK_DISABLE_FUSION`)
**Metal**: Generic fusion framework with `use_fusion` flag
**CPU**: **No fusion at all** — every op executes individually

### What this means for performance

On **CPU**, there's a fundamental ~3x gap vs ONNX Runtime because:
1. ORT does Level3 graph JIT compilation: constant folding, op fusion, layout
   optimization, kernel selection — all at graph compile time
2. ggml has no graph optimization pass; fusion only happens in GPU backends
   during compute, not at graph construction time
3. Each ggml CPU op does a separate memory pass (read+write). Fusing
   LayerNorm (norm+mul+add = 3 passes) into 1 pass saves bandwidth

On **GPU (CUDA)**, the gap should be much smaller because:
1. CUDA backend automatically fuses RMSNorm+Mul, FFN gates, multi-add
2. `ggml_flash_attn_ext` runs as a single fused CUDA kernel
3. Matmul uses cuBLAS (same as PyTorch/ONNX)
4. Memory bandwidth is 10-20x higher on GPU, so fusion matters less

### What we optimized (practical CPU-side)

1. **Pre-merged QKV weights**: concatenate Q/K/V weight matrices into one
   [H, 3H] tensor at load time. One matmul instead of three per layer.
   Saves ~0.5ms for 6-layer 384d model.

2. **Flash attention**: `ggml_flash_attn_ext` replaces 8 separate ops
   (permute, cont, mul_mat, scale, softmax, mul_mat, permute, reshape)

3. **Graph caching**: build ggml graph once per sequence length, reuse
   across calls. Eliminates ~3ms of ggml_init + graph construction.

4. **Buffer reuse**: graph_buf and work_buf persist across calls.

### Why not modify ggml for CPU fusion?

Considered but impractical because:
- ggml's CPU backend is designed for portability (pure C + SIMD intrinsics)
- Adding a graph optimization pass would affect all ggml users
- The `ggml_map_custom` API allows custom kernels but doesn't help with
  matmul (the expensive op) — ggml's SIMD matmul is already well-optimized
- Fusing norm+mul+add saves < 0.1ms per text (memory-bound, not compute-bound)
- The 3x gap to ONNX is dominated by ORT's matmul scheduling and cache
  optimization, not by op fusion per se

### GPU prediction

On CUDA, CrispEmbed should match or beat ONNX because:
- cuBLAS matmul is the same engine ORT uses
- ggml's CUDA fusion handles the same patterns ORT fuses
- Flash attention is implemented as a single CUDA kernel
- No Python/ONNX overhead in our C++ server

Estimated GPU performance for MiniLM (RTX 3060):
- CrispEmbed CUDA: ~2-4ms (model fits entirely in GPU memory)
- fastembed ONNX+CUDA: ~2-4ms (cuBLAS + graph optimization)
- Likely on par, with CrispEmbed winning on server overhead
