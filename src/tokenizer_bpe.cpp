// tokenizer_bpe.cpp — GPT-2 BPE tokenizer for decoder models.
// Uses core_bpe from CrispASR for the actual BPE merge algorithm.

#include "tokenizer.h"
#include "core/bpe.h"

#include <algorithm>
#include <cstring>

bool BPETokenizer::load(const std::vector<std::string> & vocab,
                         const std::vector<std::string> & merges,
                         int eos_id, int pad_id, int suffix_id,
                         int max_length) {
    id_to_token_ = vocab;
    token_to_id_.clear();
    token_to_id_.reserve(vocab.size());
    for (int i = 0; i < (int)vocab.size(); i++) {
        token_to_id_[vocab[i]] = i;
    }

    merge_rank_.clear();
    merge_rank_.reserve(merges.size());
    for (int i = 0; i < (int)merges.size(); i++) {
        merge_rank_[merges[i]] = i;
    }

    eos_id_ = eos_id;
    pad_id_ = pad_id;
    suffix_id_ = suffix_id;
    max_length_ = max_length;
    return !vocab.empty();
}

embed_tokens BPETokenizer::encode(const std::string & text) const {
    // Use core_bpe's tokenize_simple (GPT-2 byte-level BPE)
    auto ids = core_bpe::tokenize_simple(token_to_id_, merge_rank_, text);

    // Append suffix token if model uses one (detected during conversion)
    // Octen: 151643, F2LLM: 151645, Jina v5: none (-1)
    if (suffix_id_ >= 0) {
        ids.push_back(suffix_id_);
    }

    // Build result (no padding for decoder models — variable length)
    int seq_len = std::min((int)ids.size(), max_length_);

    embed_tokens result;
    result.ids.resize(seq_len);
    result.type_ids.resize(seq_len, 0);
    result.attn_mask.resize(seq_len, 1);

    for (int i = 0; i < seq_len; i++) {
        result.ids[i] = ids[i];
    }

    return result;
}
