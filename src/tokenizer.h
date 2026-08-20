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

// Resolve the tokenizer FAMILY (0=WordPiece, 1=BPE, 2=SentencePiece) an
// encoder GGUF selects. Pure function so the decision table is hermetically
// testable (tests/test_bert_pretokenize.cpp).
//
// Sources, in order of authority:
//   * `tokenizer.ggml.type` (CrispEmbed's own numeric key). When PRESENT it
//     is a declaration and is final — in particular an explicit WordPiece
//     (0) is honoured for vocabs over 100k (LaBSE, 501k tokens). No
//     historical GGUF carries type=0 with such a vocab (the old converter
//     could not produce one), so honouring it changes nothing shipped.
//   * `tokenizer.ggml.model` (community/llama.cpp string key: gpt2/bpe ->
//     BPE; t5/unigram/spm/llama -> SentencePiece; bert/wordpiece ->
//     WordPiece). Mapped exactly as before — including that a "bert" GGUF
//     with a >100k vocab still falls to the legacy heuristic below, wrong as
//     that is for a community LaBSE file: absent CrispEmbed metadata keeps
//     absolutely the historical behavior.
//   * Legacy fallback for GGUFs with neither key: vocabs over 100k are
//     assumed SentencePiece (the pre-tokenizer.json heuristic).
inline int resolve_tokenizer_family(bool type_key_present, int declared_type, const std::string & model_str,
                                    int64_t vocab_n) {
    int family = type_key_present ? declared_type : 0;
    if (!type_key_present) {
        if (model_str == "gpt2" || model_str == "bpe")
            family = 1;
        else if (model_str == "t5" || model_str == "unigram" || model_str == "spm" || model_str == "llama")
            family = 2;
        else if (model_str == "bert" || model_str == "wordpiece")
            family = 0;
        // else: leave 0 and let the legacy `n > 100000 -> SPM` heuristic
        // below decide (covers old GGUFs with neither key).
    }
    if (family == 0 && !type_key_present && vocab_n > 100000) family = 2;
    return family;
}

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

    // Enable the HF BertNormalizer + BertPreTokenizer pre-tokenization
    // (core/bert_pretok.h): Unicode whitespace, \p{P} isolation, per-ideograph
    // CJK split, control/format removal. Selected by `tokenizer.ggml.pre =
    // "bert"` — an ABSENT key keeps the historical per-byte
    // isspace/ispunct path, so every shipped GGUF tokenizes byte-identically.
    // Needed by LaBSE-class GGUFs (cased multilingual WordPiece).
    void set_bert_pretok(bool on) { bert_pretok_ = on; }

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
    bool bert_pretok_ = false; // HF Bert pre-tokenization; off = historical byte path

    // Pre-tokenize `text` into words: core_bert::pretokenize when
    // bert_pretok_, else the historical per-byte isspace/ispunct split.
    std::vector<std::string> split_words(const std::string & text) const;

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

    // Select the segmentation algorithm and space handling.
    //   bpe_merge=false (default): Unigram / Viterbi max-score path — correct
    //     for XLM-R and other Unigram SentencePiece models.
    //   bpe_merge=true: SentencePiece-BPE bigram greedy-merge (llama.cpp SPM) —
    //     correct for Gemma/Llama, whose `scores` are merge ranks, not unigram
    //     log-probs (Viterbi over ranks over-segments).
    //   add_space_prefix: prepend a leading ▁ dummy prefix (XLM-R convention).
    //     Gemma sets tokenizer.ggml.add_space_prefix=false → no dummy prefix.
    void set_spm_mode(bool bpe_merge, bool add_space_prefix) {
        bpe_merge_ = bpe_merge;
        add_space_prefix_ = add_space_prefix;
    }

    // Apply the SentencePiece `Precompiled` (nmt_nfkc charsmap) normalizer
    // before segmentation — see core/spm_norm.h. Every XLM-R-family Unigram
    // model declares it and we implemented none of it, so `…` tokenized to
    // three <unk> instead of the single `...`, and every fullwidth form and
    // U+3000 went through unnormalized.
    //
    // OFF by default at this layer and enabled per consumer, because only the
    // embedding path has been measured against HF. The clip_text (SigLIP)
    // path carries the SAME charsmap but wraps it in Lowercase/Strip steps we
    // do not implement, so flipping it there needs its own A/B first.
    void set_hf_normalize(bool on) { hf_normalize_ = on; }

    // SigLIP wraps the same charsmap in Lowercase + punctuation-strip +
    // whitespace-collapse + Strip (see core/spm_norm.h). Selecting it here
    // rather than inferring it keeps the embedder path on the plain charsmap.
    void set_siglip_normalize(bool on) {
        hf_normalize_ = on;
        siglip_normalize_ = on;
    }

private:
    // Applies the charsmap when enabled; identity otherwise.
    std::string hf_normalize_text(const std::string & text) const;

public:
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
    int max_token_len_ = 64;        // max byte length of any vocab token
    bool bpe_merge_ = false;        // SentencePiece-BPE bigram merge (Gemma/Llama)
    bool add_space_prefix_ = true;  // prepend leading ▁ dummy prefix (XLM-R)
    bool hf_normalize_ = false;     // HF Precompiled/nmt_nfkc charsmap (opt-in per consumer)
    bool siglip_normalize_ = false; // ...wrapped in SigLIP's Lowercase/Strip sequence

    std::vector<int> tokenize_text(const std::string & text) const; // Unigram / Viterbi
    std::vector<int> tokenize_bpe(const std::string & text) const;  // SentencePiece-BPE merge
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

    // Enable the GPT-2 ByteLevel regex pre-tokenizer for the GPT-2 byte-level
    // path (ModernBERT, tokenizer.ggml.pre = "modern-bert"). Off by default the
    // GPT-2 path uses a simpler whitespace-split pre-tokenizer (Qwen3 decoder).
    // Set separately from load() so a merges reload does not clear it.
    void set_gpt2_regex_pretok(bool v) { gpt2_regex_pretok_ = v; }

    // Enable the o200k_base regex pre-tokenizer (tokenizer.ggml.pre = "o200k";
    // granite-embedding-97m-multilingual-r2). Takes precedence over the GPT-2
    // regex flag. `ignore_merges` mirrors the tokenizer.json BPE flag: a
    // pre-token that is itself a vocab entry skips the merge table. Set
    // separately from load() so a merges reload does not clear it.
    void set_o200k_regex_pretok(bool v, bool ignore_merges = true) {
        o200k_regex_pretok_ = v;
        ignore_merges_ = ignore_merges;
    }

    // Enable the case-aware Ministral3/Tekken ByteLevel split declared in
    // tokenizer.json (single digits, no contraction suffix attachment).
    void set_ministral_regex_pretok(bool v) { ministral_regex_pretok_ = v; }

    // SentencePiece-BPE mode (▁ space marker). load() takes it as a parameter
    // and therefore resets it; the encoder path reads it back here across its
    // post-weight-load merges reload.
    bool spm_style() const { return spm_style_; }

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
    int suffix_id_ = 151643;              // token appended after text (model-specific)
    int bos_id_ = -1;                     // BOS token (-1 = none)
    bool spm_style_ = false;              // SentencePiece BPE mode
    bool spm_dummy_prefix_ = false;       // SentencePiece add_dummy_prefix
    bool clip_style_ = false;             // OpenAI CLIP text BPE (</w> end-of-word suffix)
    bool gpt2_regex_pretok_ = false;      // GPT-2 ByteLevel regex pre-tokenizer (ModernBERT)
    bool o200k_regex_pretok_ = false;     // o200k_base regex pre-tokenizer (granite-r2 97m)
    bool ministral_regex_pretok_ = false; // Ministral3/Tekken case-aware ByteLevel split
    bool ignore_merges_ = true;           // o200k: a whole-pre-token vocab hit skips the merges
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

    // GPT-2 ByteLevel regex pre-tokenizer (HF `ByteLevel` with use_regex=true):
    //   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    // Returns raw-byte pre-tokens (before byte-level encoding). Used by the
    // GPT-2 path when gpt2_regex_pretok_ is set (ModernBERT).
    std::vector<std::string> gpt2_pretokenize(const std::string & text) const;
};
