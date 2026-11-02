# Below-bar artifact repair (2 November 2026)

Harness: `tools/ci-heavy/below_bar_recipes.py` (heavy-cpu.yml, ubuntu-latest, 4 vCPU). The panel is the 13
CUDA-audit texts plus the 65-text held-out imatrix eval corpus. F2LLM queries go through its instruct prompt.
F2LLM and CLIP have 78 texts and BGE-M3 has 77. The quoted value is the minimum cosine over all texts (ColBERT:
over every content token, aligned to FlagEmbedding, which drops the CLS position). It is measured against the
original Python model, and also against the native F32 GGUF; the score is the lower of the two. Bars:
F32 0.99, F16 0.98, Q8_0 0.97, Q4_K 0.94. The precision is the dominant 2-D weight type by element count.

| artifact | before (min / mean vs original) | after (min / mean vs original) | file, MB |
|---|---|---|---|
| f2llm-v2-0.6b Q4_K | `-q4_k-imatrix` 0.7694 / 0.8220 (audit vs F32: 0.782 CUDA, 0.789 CPU) | `-q4_k-mixed` **0.9644 / 0.9762** | 454 (Q8_0 610, old 400) |
| clip-text-base Q4_K | `-q4_k` 0.4994 / 0.9615 before the runtime fixes (audit vs F32: 0.933 CUDA, 0.925 CPU); 0.9494 after them | `-q4_k-mixed` **0.9878 / 0.9953** | 51 |
| bge-m3 heads F16 | no published heads file; native F32 vs FlagEmbedding was dense 0.969, sparse 0.675, ColBERT 0.179 | `-heads-f16`: dense, sparse and ColBERT all **1.0000** | 1599 |
| bge-m3 heads Q8_0 | audit `crisp-q8_0` ColBERT 0.593 vs F16 | `-heads-q8_0-mixed`: dense **0.9995**, sparse **0.9991**, ColBERT **0.9726** | 718 |

Publications (HF commits, readback verified from an empty cache by SHA-256 and re-evaluated):
`cstr/f2llm-v2-0.6b-GGUF@c29299b7`, `cstr/clip-text-base-GGUF@3d52b127`, `cstr/bge-m3-GGUF@6016c355`.
No earlier file was overwritten or deleted. README cards append a dated "superseded" note.

## Findings

- **F2LLM-0.6B is a weight-sensitivity problem, not a runtime one.** The same Q4_K+imatrix weights,
  dequantized to F16 and run as F16 matmuls, still score 0.782 (Q4_K file: 0.767). Even uniform Q6_K only
  reaches 0.937. Per-group sensitivity: block 27 (+0.120), `ffn_down` (+0.094), `attn_v` (+0.038),
  block 26 (+0.030). The chosen recipe is Q4_K+imatrix, with `ffn_down` at Q6_K, `attn_v` and blocks 25-27 at
  Q8_0, and token embeddings at Q8_0. It is Q4_K-dominant: 288M Q4_K elements against 229M Q8_0 and 79M Q6_K.
  Seven candidates are in `f2llm-recipes/`.
- **CLIP: two reference/runtime defects masked the quantization question.**
  1. Native truncation to 77 tokens dropped `<|endoftext|>`, so pooling read a content token. F32 vs the
     original was 0.568 on the long audit text. Fixed in `src/clip_text_embed.cpp`:
     `CRISPEMBED_CLIP_TEXT_KEEP_EOS`, default on, `=0` restores the old path.
  2. The first reference used the slow `CLIPTokenizer`, which splits CJK characters without ftfy. The fast
     tokenizer matches OpenAI.
  After both, F32 scores 0.9989 and plain Q4_K+imatrix 0.978. The chosen recipe adds `attn.v` and block 0
  at Q8_0.
- **BGE-M3: the native F32 itself differed from FlagEmbedding. Both causes were runtime defects, not
  precision.**
  1. The XLM-R tokenizer.json normalizer `Replace(" {2,}", " ")` was missing, and trailing whitespace became
     a lone `▁`. Ten of 77 texts were tokenized differently. Fixed in `src/tokenizer_spm.cpp`:
     `CRISPEMBED_SPM_COLLAPSE_SPACES`, default on, `=0` is the old path. After the fix, 0 of 77 differ.
  2. The sparse head kept CLS/EOS/PAD/UNK weights, which FlagEmbedding drops. Fixed in `src/crispembed.cpp`:
     `CRISPEMBED_SPARSE_KEEP_SPECIAL=1` restores the old output. Before the fix, 0 of 77 term sets matched.
  After both, F32 and F16 equal FlagEmbedding (1.00000) on all three heads.
- **Why Q8_0 ColBERT was 0.59.** Not colbert_linear: at F16 or F32 it gives 0.6125. Not the CLS row: 0.999.
  Not activation quantization: the dequantized Q8_0 weights at F16 still give 0.474. Not near-zero vectors:
  raw ColBERT norms are at least 10.97. With the whitespace fix, 21 of ~2,970 token vectors stay below 0.97,
  mostly in the 20x-repeated Einstein text. Keeping blocks 0-11 at F16 fixes all of them (0.9726). Blocks 12-23,
  blocks 0-5, blocks 6-11, block 0, all attention or all FFN at F16 do not. The audit's own 0.59 also included
  the whitespace tokenizer defect: the invoice text had lone `▁` tokens.
- **Negative results (not published):**
  - BGE-M3 uniform Q8_0: ColBERT 0.612.
  - BGE-M3 Q4_K+imatrix: ColBERT 0.43; with attention and FC2 at Q8_0, 0.59; with blocks 0-11 at F16, 0.37.
    Its dense (0.982) and sparse (0.993) clear the bar, so a dense-only Q4_K stays possible.
  - F2LLM: uniform Q5_K 0.849, Q6_K 0.937, IQ4_XS+imatrix 0.690, Q4_K_M-like 0.890.

Run receipts are in the subdirectories, each with a `RUN.txt` link.
