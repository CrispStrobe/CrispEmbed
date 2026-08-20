# CrispEmbed v0.17.9

A model-compatibility and reliability release — eight commits since v0.17.8.
The headline is a native ggml port of MOST Embed DE, a 1.14B-parameter German
retrieval model based on NVIDIA's bidirectional Ministral3 embedding backbone.
This release also fixes ownership of mmap-backed GGUF weights and expands the
measured multilingual retrieval matrix to Arabic and Korean.

---

## Added: MOST Embed DE

- Native optimized graph for the bidirectional Ministral3 architecture: grouped
  query attention, NEOX YaRN rotary embeddings, the model's position-dependent
  query scale, RMSNorm/SwiGLU blocks, mean pooling and L2 normalization.
- Tekken ByteLevel BPE pre-tokenization matched to the upstream tokenizer,
  including its case-aware split and one-digit token boundaries.
- Self-describing `query: ` / `passage: ` retrieval prompts in GGUF metadata,
  with CLI registry fallbacks for older readers.
- Python reference dumping at every transformer boundary and native
  `crispembed-diff` layer dumps. F16 matches the original Transformers model at
  every captured boundary and on the final embedding (cosine 1.000000).
- Two hosted, SHA-256-pinned variants:
  - `most-embed-de` / `most-embed-de-q4k`: 918 MB, Q4_K feed-forward matrices
    with Q8_0 token embeddings and q/k/v/o attention projections. Across eight
    German query/document texts, minimum cosine is 0.987191 and all top-1
    retrieval results are preserved.
  - `most-embed-de-q8`: 1.22 GB, final cosine 0.999818.

The fine-tune is CC-BY-NC-4.0, so every registry alias is protected by
`--accept-license cc-by-nc-4.0`. The GGUF repository and model metadata also
retain the base model's OpenMDW-1.1 agreement and origin notice.

## Fixed: mmap-backed GGUF weight ownership

Weight mappings now stay alive through the backend buffer that references
them, rather than through incidental loader scope. A dedicated mapping-release
test covers the lifetime boundary. This prevents dangling mapped weights while
still releasing the mapping as soon as its owning buffer is destroyed.

## Fixed: batch tokenizer override precedence

The decoder batch-encode path now gives an explicitly disabled
`CRISPEMBED_BPE_HF_UNK` override the same precedence as the single-input path.
Batch and scalar encoding therefore agree when compatibility behavior is
intentionally switched off.

## Multilingual evaluation

The embedding and reranker evaluation harnesses now cover Arabic and Korean in
addition to Japanese. All cached multilingual models in the recorded matrix
pass the new checks. The accompanying audit also documents why vocabulary
script counts are not a reliable capability signal for byte-level embedding
tokenizers; those counts remain useful for OCR recognizers but are not exposed
as embedding-language claims.

