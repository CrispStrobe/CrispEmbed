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
    std::vector<int32_t> type_ids;  // 0 for single-sentence
    std::vector<int32_t> attn_mask; // 1 for real tokens, 0 for padding
};

class WordPieceTokenizer {
public:
    // Load vocab from a list of tokens (index = token id).
    // Special tokens: [CLS]=cls_id, [SEP]=sep_id, [UNK]=unk_id, [PAD]=pad_id.
    bool load(const std::vector<std::string> & vocab, int cls_id, int sep_id, int unk_id, int pad_id,
              int max_length = 512, bool do_lower_case = true);

    // Tokenize a single text: [CLS] + tokens + [SEP], padded to max_length.
    embed_tokens encode(const std::string & text) const;

    // Tokenize a sentence pair for cross-encoders/rerankers:
    // [CLS] text_a [SEP] text_b [SEP], type_ids 0/1, padded to max_length.
    embed_tokens encode_pair(const std::string & text_a, const std::string & text_b) const;

    int vocab_size() const { return (int)id_to_token_.size(); }
    int max_length() const { return max_length_; }
    // Look up the surface form of a token by id. Returns an empty string
    // (with a stable address valid for the tokenizer's lifetime) for
    // out-of-range ids. WordPiece subword continuations are prefixed with
    // "##" — callers can use that to group subwords back into words.
    const std::string & token_str(int id) const {
        static const std::string empty;
        if (id < 0 || (size_t)id >= id_to_token_.size()) return empty;
        return id_to_token_[(size_t)id];
    }

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    int cls_id_ = 101;
    int sep_id_ = 102;
    int unk_id_ = 100;
    int pad_id_ = 0;
    int max_length_ = 512;
    bool do_lower_case_ = true;

    // Trie for O(len) longest-match WordPiece lookup.
    // Two roots: trie_root_ for first pieces, trie_cont_ for ## continuations.
    struct TrieNode {
        int token_id = -1;                      // -1 = no token ends here
        std::unordered_map<char, int> children; // char → index in trie_nodes_
    };
    std::vector<TrieNode> trie_nodes_;
    int trie_root_ = -1; // index of root for first-piece tokens
    int trie_cont_ = -1; // index of root for continuation (##) tokens
    bool trie_built_ = false;

    void build_trie();

    // WordPiece: split a single word into subword tokens.
    std::vector<int> wordpiece(const std::string & word) const;
};

// SentencePiece-style tokenizer for XLM-RoBERTa models.
// Uses unigram (greedy longest-match) from vocab + optional scores.
class SentencePieceTokenizer {
public:
    bool load(const std::vector<std::string> & vocab, const std::vector<float> & scores, int bos_id, int eos_id,
              int unk_id, int pad_id, int max_length = 512);

    embed_tokens encode(const std::string & text) const;

    // Tokenize a sentence pair: <s> text_a </s> text_b </s>, type_ids all 0.
    embed_tokens encode_pair(const std::string & text_a, const std::string & text_b) const;

    // C2 behavior flags (tokenizer.ggml.add_bos_token / add_eos_token):
    // gate the <s>/</s> wrap in encode(). encode_pair() keeps the canonical
    // cross-encoder layout regardless — separators are structural there.
    void set_add_flags(bool add_bos, bool add_eos) {
        add_bos_ = add_bos;
        add_eos_ = add_eos;
    }

    int vocab_size() const { return (int)id_to_token_.size(); }
    int max_length() const { return max_length_; }
    // Look up the surface form of a token by id. Returns an empty string
    // (with a stable address valid for the tokenizer's lifetime) for
    // out-of-range ids. SentencePiece word-start tokens are prefixed with
    // the U+2581 marker ("▁") — callers can use that to group subwords
    // back into words.
    const std::string & token_str(int id) const {
        static const std::string empty;
        if (id < 0 || (size_t)id >= id_to_token_.size()) return empty;
        return id_to_token_[(size_t)id];
    }

private:
    std::unordered_map<std::string, int> token_to_id_;
    std::vector<std::string> id_to_token_;
    std::vector<float> scores_;
    int bos_id_ = 0;
    int eos_id_ = 2;
    int unk_id_ = 3;
    int pad_id_ = 1;
    bool add_bos_ = true; // wrap encode() with <s> (default = historical behavior)
    bool add_eos_ = true; // wrap encode() with </s>
    int max_length_ = 512;
    int max_token_len_ = 64; // max byte length of any vocab token

    std::vector<int> tokenize_text(const std::string & text) const;
};

// BPE tokenizer for decoder embedding models.
// Supports three modes:
//   GPT-2 style (Qwen3): byte-level encoding with Ġ space marker
//   SentencePiece style (Gemma): ▁ space marker, BOS/EOS tokens
//   CLIP style (OpenAI CLIP text): lowercase + whitespace-clean + regex
//     pre-tokenize + byte-level encoding with </w> end-of-word suffix
class BPETokenizer {
public:
    bool load(const std::vector<std::string> & vocab, const std::vector<std::string> & merges, int eos_id, int pad_id,
              int suffix_id,
              int bos_id = -1,        // -1 = no BOS
              bool spm_style = false, // true for SentencePiece BPE (Gemma)
              int max_length = 8192,
              bool spm_dummy_prefix = false, // add_dummy_prefix (ERNIE/SPM)
              bool clip_style = false);      // true for OpenAI CLIP text BPE (</w> end-of-word)

    embed_tokens encode(const std::string & text) const;

    int vocab_size() const { return (int)id_to_token_.size(); }
    int bos_id() const { return bos_id_; }
    int eos_id() const { return eos_id_; }
    int pad_id() const { return pad_id_; }
    const std::vector<std::string> & get_vocab() const { return id_to_token_; }

private:
    std::unordered_map<std::string, int32_t> token_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    std::vector<std::string> id_to_token_;
    int eos_id_ = 151645;
    int pad_id_ = 151643;
    int suffix_id_ = 151643;        // token appended after text (model-specific)
    int bos_id_ = -1;               // BOS token (-1 = none)
    bool spm_style_ = false;        // SentencePiece BPE mode
    bool spm_dummy_prefix_ = false; // SentencePiece add_dummy_prefix
    bool clip_style_ = false;       // OpenAI CLIP text BPE (</w> end-of-word suffix)
    int max_length_ = 8192;

    // SentencePiece BPE: merge-based tokenization on ▁-prefixed text
    std::vector<int32_t> bpe_merge(const std::string & text) const;

    // Rank-merge a list of initial symbols in place (shared by bpe_merge and
    // the CLIP path). Uses merge_rank_ with an O(N log N) priority queue.
    void merge_symbols(std::vector<std::string> & symbols) const;

    // OpenAI CLIP text pre-tokenizer: lowercase + whitespace-clean +
    // regex split (contractions / letter runs / single digits / punctuation
    // runs). Returns raw-byte pre-tokens (before byte-level encoding).
    std::vector<std::string> clip_pretokenize(const std::string & text) const;

    // Byte-encode one CLIP pre-token, append the </w> end-of-word marker to
    // the final symbol, rank-merge, and append the resulting vocab IDs.
    void clip_bpe_word(const std::string & pretoken, std::vector<int32_t> & out) const;
};
