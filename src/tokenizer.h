// tokenizer.h — WordPiece tokenizer for BERT-family models.
//
// Loaded from GGUF metadata (vocab stored as string array).
// Produces token IDs + attention mask for a single text input.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct embed_tokens {
    std::vector<int32_t> ids;
    std::vector<int32_t> type_ids;    // 0 for single-sentence
    std::vector<int32_t> attn_mask;   // 1 for real tokens, 0 for padding
};

class WordPieceTokenizer {
public:
    // Load vocab from a list of tokens (index = token id).
    // Special tokens: [CLS]=cls_id, [SEP]=sep_id, [UNK]=unk_id, [PAD]=pad_id.
    bool load(const std::vector<std::string> & vocab,
              int cls_id, int sep_id, int unk_id, int pad_id,
              int max_length = 512);

    // Tokenize a single text: [CLS] + tokens + [SEP], padded to max_length.
    embed_tokens encode(const std::string & text) const;

    int vocab_size() const { return (int)id_to_token_.size(); }
    int max_length() const { return max_length_; }

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    int cls_id_ = 101;
    int sep_id_ = 102;
    int unk_id_ = 100;
    int pad_id_ = 0;
    int max_length_ = 512;

    // WordPiece: split a single word into subword tokens.
    std::vector<int> wordpiece(const std::string & word) const;
};
