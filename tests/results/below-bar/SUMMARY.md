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
| bge-m3 heads Q8_0, CUDA margin (3 Nov) | `-heads-q8_0-mixed` ColBERT 0.9667 on CUDA T4 | `-heads-q8_0-enc0-15-f16`: dense **0.9996**, sparse **0.9998**, ColBERT **0.9911** (CUDA **0.9922**) | 763 |

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

## BGE-M3 Q8_0 mix for CUDA (3 November 2026)

The blocks 0-11 mix scored ColBERT 0.9667 on CUDA (T4, cuda-audit run 7, against the CPU F16 base),
below the 0.97 bar; on CPU it is 0.9726. Candidates with more early blocks in F16 were measured in
the portable build on hosted CPU against FlagEmbedding (run 41588222873, `bgem3-cuda-margin/`) and
on a Kaggle T4 (cuda-audit batch h, `../gpu-audit/cuda-kaggle/run8-66a5eba0/`). The CUDA arm uses
the run-7 method: 13 audit texts, minimum over ColBERT tokens against the CPU arm of the F16 file.

| recipe (Q8_0, F16 blocks) | MiB | CPU vs original dense / sparse / ColBERT | CUDA vs CPU F16 ColBERT | CUDA dense / sparse |
|---|---:|---|---:|---|
| 0-11 (published 2 Nov) | 718 | 0.99945 / 0.99913 / 0.97255 | **0.96671** | 0.99949 / 0.99988 |
| 0-13 | 740 | 0.99949 / 0.99982 / **0.94881** | 0.98428 | 0.99959 / 0.99993 |
| **0-15 (published)** | **763** | 0.99955 / 0.99984 / **0.99110** | **0.99215** | 0.99963 / 0.99993 |
| 0-17 | 785 | 0.99961 / 0.99985 / 0.97249 | **0.95147** | 0.99969 / 0.99997 |
| 0-19 | 808 | 0.99970 / 0.99993 / 0.99224 | 0.99476 | 0.99975 / 0.99998 |
| 0-23 (all blocks; embeddings Q8_0) | 853 | 0.99999 / 0.99999 / 0.99270 | 0.99547 | 0.99999 / 1.00000 |

- The ColBERT minimum is not monotonic in the number of F16 blocks: 0-13 and 0-17 each lose
  one token in the repeated text 11 (CPU tokens 578 and 512), and each is below the bar on one backend. In 0-15, 0-19 and 0-23 the worst token is the
  same one (text 11, token 207), at the floor that the Q8_0 token embedding sets. 0-15 is the
  smallest recipe at that floor on both backends.
- **Reproducibility:** each candidate has the same SHA-256 on the hosted EPYC runner and on the Kaggle
  Xeon worker, and 0-11 rebuilt from a fresh `--crisp` F32 conversion is byte-identical to the
  published `bge-m3-heads-q8_0-mixed.gguf` (`b8ecf56f…`). The recipe uses only `--tensor-type`
  overrides; the quantizer's default output is unchanged.
- Published by run 41596007262 as `bge-m3-heads-q8_0-enc0-15-f16.gguf`, `cstr/bge-m3-GGUF@283645d8`,
  sha256 `3379a216…` (local = LFS = empty-cache readback, re-scored 0.99109 after readback). The
  registry alias `bge-m3-heads` now points to it. The 0-11 file stays as `bge-m3-heads-q8-enc0-11`,
  marked superseded and CPU-only.
- Timing (Kaggle T4 worker, 13 texts, ColBERT): CUDA 1.9 s for both 0-11 and 0-15, peak 879 vs 923 MiB.
  The CPU arm takes 15.4 s vs 17.0 s (the F16 file takes 21.1 s).

