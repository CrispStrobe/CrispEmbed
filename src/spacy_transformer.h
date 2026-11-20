#ifndef CRISPEMBED_SPACY_TRANSFORMER_H
#define CRISPEMBED_SPACY_TRANSFORMER_H

#include "spacy_lexical.h"
#include "spacy_numeric.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;
typedef struct ggml_backend * ggml_backend_t;

// spaCy transformer pipelines (spacy-curated-transformers 0.3, curated-
// transformers 0.1; MIT): CamemBERT piece encoding (SentencePiece, one BOS/EOS
// per document, whitespace tokens get no pieces, ids shifted to the fairseq
// vocabulary), WithStridedSpans (windows of `window` pieces every `stride`,
// overlaps averaged pairwise), a RoBERTa-style post-LayerNorm encoder (exact
// GELU), BOS/EOS removal and the LastTransformerLayerListener's reduce_mean
// pooling per spaCy token. Heads (tagger/morphologizer/parser lower layer) are
// computed on the pooled vectors and returned as NumericOutput.
// Built only with CRISPEMBED_SPACY_TRF (SentencePiece C++ v0.2.1, Apache-2.0).
namespace crispembed {
namespace spacy {

struct TransformerConfig {
    int layers = 12, width = 768, heads = 12, intermediate = 3072;
    float layer_norm_eps = 1e-5f;
    int padding_idx = 1, max_positions = 514, model_max_length = 512;
    int window = 128, stride = 96;
    // "camembert_sentencepiece" or "roberta_byte_bpe".
    std::string piece_encoder = "camembert_sentencepiece";
    std::string sentencepiece_model; // serialized ModelProto bytes (CamemBERT)
    // GPT-2 byte-level BPE (RoBERTa): pieces in id order, ranked merges, the
    // pre-split pattern and special ids.
    std::vector<std::string> bpe_pieces;
    std::vector<std::pair<std::string, std::string>> bpe_merges;
    std::string bpe_split_pattern;
    int bos_id = 0, eos_id = 2, unk_id = 3;
    // BERT WordPiece ("bert_wordpiece"): pieces in id order, "##" marks
    // continuation pieces (curated-tokenizers WordPieceProcessor).
    std::vector<std::string> wordpiece_pieces;
    // First position id: RoBERTa-family padding_idx + 1; BERT 0. -1 = derive.
    int position_offset = -1;
};

class TransformerPipeline {
public:
    TransformerPipeline(TransformerConfig config, std::map<std::string, ggml_tensor *> weights, ggml_backend_t backend);
    ~TransformerPipeline();
    TransformerPipeline(const TransformerPipeline &) = delete;
    TransformerPipeline & operator=(const TransformerPipeline &) = delete;

    // tok2vec = pooled token vectors (n x width); tagger/morphologizer scores
    // when those heads exist; parser_cache = pad row + lower-layer outputs.
    NumericOutput run(const std::vector<LexicalToken> & tokens, const TraceSink & trace = {},
                      const CancelCheck & cancelled = {}) const;

    // Piece ids per token (fairseq ids), BOS/EOS excluded; for diagnostics.
    std::vector<std::vector<int>> pieces(const std::vector<LexicalToken> & tokens) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace spacy
} // namespace crispembed

#endif // CRISPEMBED_SPACY_TRANSFORMER_H
