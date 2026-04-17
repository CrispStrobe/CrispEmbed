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

## Ollama integration learnings

### Architecture: Ollama uses ggml via CGO (same as CrispEmbed)

Both Ollama and CrispEmbed use ggml for tensor computation. Ollama wraps ggml
ops in Go structs via CGO (`C.ggml_mul_mat`, `C.ggml_rms_norm`). CrispEmbed
calls ggml directly from C++. The computation graphs are functionally identical.

### Phantom-space token vocabulary (critical for WordPiece)

Ollama's WordPiece tokenizer expects tokens in SentencePiece-style format:
- `"hello"` → `"▁hello"` (prepend ▁)
- `"##ing"` → `"ing"` (strip ##)
- `"[CLS]"` → `"[CLS]"` (keep special tokens)

Without this transformation, cos drops from 1.0 to ~0.19.

### GELU variant matters (exact erf vs tanh approximation)

BERT uses exact GELU (erf-based). Ollama's `.GELU()` uses tanh approximation
(`ggml_gelu_inplace`). Must use `.GELU_ERF()` for BERT/XLM-R encoder models.
Difference: cos 0.996 → 1.000.

### SentencePiece Unigram needs Viterbi DP, not pairwise merge

Ollama's existing `SentencePiece` tokenizer uses BPE-style greedy pairwise
merge (priority queue). This is WRONG for Unigram models (XLM-R, e5-small).
We added `SentencePieceUnigram` using Viterbi DP (same as CrispEmbed's
tokenizer_spm.cpp). Must also prepend space before tokenization.

### Gemma3 (1+weight) RMSNorm must be pre-baked for Ollama

Ollama's RMSNorm does `rms_norm(x) * weight`. Gemma3 needs `rms_norm(x) * (1 + weight)`.
CrispEmbed handles this at runtime with a `ones` tensor. For Ollama export,
pre-add +1 to all norm weights in the GGUF.

### Quantized token_types breaks Ollama binary ops

Ollama's ggml doesn't support `f32 + q8_0` in elementwise ops. The tiny
`token_types.weight` tensor (2 rows) must be kept as f32 during quantization.
Error: `binary_op: unsupported types: dst: f32, src0: f32, src1: q8_0`.

### Nil-guards needed for optional model components

Ollama's Qwen3 model.go unconditionally calls `QueryNorm.Forward()` — panics
for models without QK-norm (e.g. Jina v5). Gemma3 embed.go unconditionally
iterates `Dense` projection — panics for models without it (Harrier-270M).

### Jina v5 LoRA adapters need merge before export

Jina v5 models use task-specific LoRA adapters (retrieval, classification,
clustering, text-matching). Must call `model.set_adapter("retrieval")` then
`model.merge_and_unload()` before GGUF export. The `encode()` method does
more than standard forward+pool, so merged output won't exactly match HF.

### SentencePiece BERT models should use bert arch, not xlmr

Models like multilingual-e5-small report `model_type="bert"` with SentencePiece
tokenizer. These are BERT models (no position offset), not XLM-R. Only true
`roberta`/`xlm-roberta` types need the `xlmr` arch with position offset.

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

## Matmul optimization — what we use, what's available

### Current state (as of April 2026)

Our embedding models have small matrices: 384×384 (MiniLM/GTE) to 1024×4096
(Qwen3 FFN). For these sizes, overhead per matmul call matters more than
raw FLOP throughput.

### CPU matmul options (ggml-cpu)

| Option | Default | Effect | Our impact |
|--------|---------|--------|-----------|
| `GGML_LLAMAFILE` | OFF | Custom SGEMM kernels optimized for small F32 matmul | **HIGH** for F32 models |
| `GGML_AVX512` | OFF | 512-bit SIMD (2x wider than AVX2) | **HIGH** if CPU supports |
| `GGML_AVX512_VNNI` | OFF | Hardware int8 dot products | Medium for Q8_0 models |
| `GGML_AMX_TILE` | OFF | Intel AMX for int8/BF16 (Sapphire Rapids+) | None (needs new CPU) |
| `GGML_OPENMP` | ON | Thread parallelism | Already enabled |

**Enable for best CPU performance:**
```bash
cmake -S . -B build -DGGML_LLAMAFILE=ON   # custom SGEMM
cmake -S . -B build -DGGML_AVX512=ON      # if CPU supports (check /proc/cpuinfo)
```

### CUDA matmul options

| Option | Default | Effect |
|--------|---------|--------|
| `GGML_CUDA_FA` | ON | Flash attention CUDA kernel |
| `GGML_CUDA_GRAPHS` | OFF | Multi-op fusion via CUDA graph capture |
| `GGML_CUDA_FORCE_MMQ` | OFF | Force quantized matmul kernels (vs cuBLAS) |
| `GGML_CUDA_FA_ALL_QUANTS` | OFF | Flash attn for all quant types |

CUDA auto-selects between MMQ (quantized matmul) and cuBLAS (F32) based
on matrix size and GPU compute capability. For our 384×384 Q8_0 matrices,
MMQ is usually selected (faster than cuBLAS for small quantized matmul).

### Why HF PyTorch is still competitive on CUDA

HF PyTorch uses cuBLAS with operator fusion via torch.compile/TorchScript.
For a 22M-param model (MiniLM), the GPU is underutilized — compute time
is dominated by kernel launch overhead and memory transfers, not FLOP
throughput. Both HF and CrispEmbed run at ~10ms, limited by the GPU's
minimum latency per kernel launch (~5μs × ~200 kernels = ~1ms overhead).

### Batched matmul on GPU

Single matmul `W[H,H] × X[H, T*B]` is much faster than B separate
`W[H,H] × X[H, T]` calls because:
1. One cuBLAS/MMQ launch vs B launches
2. Better GPU occupancy (more work per SM)
3. Memory access amortization

Our true batched graph concatenates all texts and uses 4D flash attention
with batch dimension. The matmuls naturally batch via the flattened T*B dim.

### QKV weight fusion

Pre-merging Q/K/V weight matrices into `[H, 3H]` reduces 3 matmul calls
to 1 per layer. The merged tensor must live in the same backend buffer as
the model weights (ggml_backend_alloc_ctx_tensors) so it works on GPU.

On CPU: ~0.5ms savings (15.3ms vs 16.8ms for MiniLM).
On GPU: minor savings (kernel launch overhead reduction).

## Optimization experiment results (April 2026)

| Optimization | CPU Impact | GPU Impact | Verdict |
|---|---|---|---|
| QKV weight fusion (1 matmul vs 3) | 15.3ms vs 17.0ms (**+11%**) | minor | **Keep** — matmul reduction wins |
| Flash attention (fused QKV attn) | 16.8→15.3ms | significant | **Keep** |
| Scheduler reservation (bucket T) | no change | may help | Keep (no cost) |
| GGML_LLAMAFILE | 15.3→14.7ms (**+4%**) | N/A | **Enable by default** |
| AVX512 (if CPU supports) | 15.3→14.4ms (**+6%**) | N/A | Enable if available |
| F16 model weights | 15.3→17.7ms (**-14%**) | may help (tensor cores) | **Skip on CPU** |
| Removing ggml_cont (no QKV fusion) | 15.3→17.0ms (**-10%**) | N/A | Don't remove |
| True batched graph (4D flash attn) | slower on CPU | should help | GPU only |

### Why we can't easily match HF PyTorch

1. **Graph rebuild cost**: ggml rebuilds the graph from scratch every call (~1ms).
   PyTorch JIT-compiles and caches the execution plan.
2. **No CPU operator fusion**: ggml CPU executes each op separately (separate memory pass
   for norm, mul, add). ORT/PyTorch fuse these into single kernels.
3. **No persistent CUDA graphs**: PyTorch can capture and replay GPU command streams.
   ggml has `GGML_CUDA_GRAPHS` but it's designed for llama.cpp's specific graph topology.
4. **Batch matmul**: PyTorch's cuBLAS wrapper handles batched matmul natively.
   Our 4D reshape + flash attention adds overhead vs native batch support.

### Practical CPU performance ceiling

For MiniLM (22M params, 6 layers, 384d) on 4-thread CPU:
- **15.3ms** with all optimizations (QKV fusion + flash attn + llamafile)
- **~14ms** theoretical minimum (pure matmul compute time)
- **~1ms** graph rebuild overhead we can't eliminate
- HF PyTorch on same CPU: **54ms** (CrispEmbed is **3.5x faster on CPU**)

### Practical GPU performance ceiling

For MiniLM on RTX A1000 (budget laptop GPU):
- **10.6ms** current (with all optimizations)
- **~5ms** theoretical minimum (kernel launch overhead + small matrix underutilization)
- HF PyTorch: **9.5ms** (they have better GPU batching)
- Gap is ~1ms — likely kernel launch overhead from ggml's per-op dispatch

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

## Prompt prefix system for RAG models

Many embedding models require query/passage prefixes for optimal retrieval:
- BGE: `"Represent this sentence for searching relevant passages: "`
- E5: `"query: "` / `"passage: "`
- Nomic: `"search_query: "` / `"search_document: "`
- Jina v5: `"Query: "` / `"Document: "`

Implementation: prefix is stored in `crispembed_context::prefix` and prepended
to the raw text before tokenization in both `crispembed_encode()` and
`crispembed_encode_batch()`. This is correct because:
1. The prefix is part of the semantic input (not a tokenizer-level construct)
2. All tokenizer types (WordPiece/SentencePiece/BPE) handle it naturally
3. fastembed-rs does the same (injects prefix before tokenizer.encode)

**Not applied to sparse/colbert/reranker**: These have different input semantics.
Sparse retrieval operates on raw terms. Rerankers take (query, document) pairs
where the model handles the joint encoding.

## Bi-encoder vs cross-encoder reranking

Both approaches are valuable for RAG and complement each other:

**Bi-encoder** (embed query + docs independently, cosine similarity):
- Fast: encode once, compare N documents with dot products
- Same model used for initial retrieval AND reranking
- Quality limited by the embedding space
- CrispEmbed: `rerank_biencoder()` in Python/Rust, uses `encode_batch()` + dot product

**Cross-encoder** (encode query-document pairs jointly):
- Slow: each (query, doc) pair requires a full forward pass
- Much higher quality (joint attention between query and document tokens)
- Typically used as second-stage reranker after bi-encoder retrieval
- CrispEmbed: `rerank()` in Python/Rust, uses `crispembed_rerank()` C API

**RAG pipeline pattern**: bi-encoder retrieval (top-100) → cross-encoder reranking (top-10)

## Model registry for RAG feature parity

When adding new models to the registry (`model_mgr.cpp`), the key metadata is:
- **name**: short name for CLI/auto-download
- **filename**: GGUF filename (may include `-q8_0` suffix for default quant)
- **url**: HuggingFace direct download URL under `cstr/` namespace
- **desc**: architecture, dimension, language, parameter count

Models that are encoder-only (BERT/XLM-R) use the existing convert-bert-to-gguf.py.
Models that are decoder-based (Qwen3/Gemma3) use convert-decoder-embed-to-gguf.py.
Rerankers are encoder models with a classifier head — use `--crisp` flag to include
the classifier weights in the GGUF.

## MPNet relative position bias

MPNet uses T5-style relative position bias instead of absolute position embeddings.
The bias is a learned `Embedding(32, 12)` — 32 logarithmic distance buckets × 12
attention heads. For each (query_pos, key_pos) pair, a bucket index is computed
via logarithmic distance binning, then the bias is looked up and added to
attention scores before softmax.

**Our implementation** (CrispEmbed):
- Precompute the full `[T, T, n_heads]` bias matrix in C++ at encode time
- Pass it as the F16 mask parameter to `ggml_flash_attn_ext`
- Flash attention adds it to scores natively — no manual attention needed
- Result: cos=0.999997 vs HuggingFace

**llama.cpp approach** (PR #21880):
- Compute bucket indices in the ggml graph via `build_inp_pos_bucket_enc()`
- Look up bias weights with `build_pos_bias()` (ggml graph ops)
- Pass as `kq_b` to `build_attn()` which adds it to attention scores
- Tensor stored transposed `[n_heads, n_buckets]` on layer 0

**Key difference**: We precompute in C++ (simpler, works on CPU), they compute in
the ggml graph (GPU-accelerable, more modular). Both produce identical results.
Our approach is ~10 lines of C++ vs their ~50 lines of graph builder code.

**Bugs found during MPNet implementation**:
- Python `or` operator treats `cls_token_id=0` as falsy → falls through to
  default 101. Fix: use `is not None` check
- MPNet needs position offset = 2 (same as RoBERTa), but `model_type="mpnet"`
  was not included in the offset detection

## Reranker model conversion notes

Cross-encoder rerankers (bge-reranker, ms-marco-MiniLM, mxbai-rerank) have a
classifier head on top of the encoder:
- **1-layer**: `classifier.dense.weight [H,1]` + `classifier.dense.bias [1]`
  → CLS hidden → Linear → scalar score
- **2-layer** (RobertaClassificationHead): `classifier.dense.weight [H,H]` +
  `classifier.out_proj.weight [1,H]` + biases
  → CLS hidden → Linear → tanh → Linear → scalar score

The converter must include these weights. Detection: `crispembed.is_reranker`
is set based on presence of `classifier.dense.weight` in the GGUF.

Some rerankers (ms-marco-MiniLM) use `num_labels=1` with no activation,
while others (bge-reranker) use sigmoid/softmax. CrispEmbed returns the raw
logit — the caller decides on thresholding.
