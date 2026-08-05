// no_repeat_ngram.h — greedy argmax with HF-style no-repeat-ngram banning.
//
// Semantics match transformers' NoRepeatNGramLogitsProcessor over the given
// history: a candidate token v is banned when the last (ngram-1) history
// tokens followed by v would repeat an ngram already present in the history.
// ngram <= 1 or a history shorter than (ngram-1) bans nothing; if every
// token ends up banned (possible only for tiny vocabularies), fall back to
// the unconstrained argmax rather than emit nothing.
//
// One shared implementation for every greedy OCR decode loop
// (qwen2vl_ocr.cpp and internvl2_ocr.cpp carry it with ngram=3;
// deepseek_ocr2.cpp with the reference contract's ngram=20, env-gated).
// Guarded hermetically by tests/test_no_repeat_ngram.cpp.
#pragma once

#include <cmath>
#include <unordered_set>
#include <vector>

namespace core_decode {

// TokenT is the history's element type (int or int32_t depending on engine).
template <typename TokenT>
inline int argmax_no_repeat_ngram(const float * logits, int V, const std::vector<TokenT> & hist, int ngram) {
    std::unordered_set<int> banned;
    const int k = ngram - 1;
    const int n = (int)hist.size();
    if (ngram > 1 && n >= k && k > 0) {
        for (int i = 0; i + k < n; i++) {
            bool match = true;
            for (int j = 0; j < k; j++) {
                if (hist[i + j] != hist[n - k + j]) {
                    match = false;
                    break;
                }
            }
            if (match) banned.insert((int)hist[i + k]);
        }
    }
    int best_id = -1;
    float best = -INFINITY;
    for (int v = 0; v < V; v++) {
        if (!banned.empty() && banned.count(v)) continue;
        if (logits[v] > best) {
            best = logits[v];
            best_id = v;
        }
    }
    if (best_id < 0) {
        for (int v = 0; v < V; v++)
            if (logits[v] > best) {
                best = logits[v];
                best_id = v;
            }
    }
    return best_id;
}

} // namespace core_decode
