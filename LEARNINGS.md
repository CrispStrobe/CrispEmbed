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
