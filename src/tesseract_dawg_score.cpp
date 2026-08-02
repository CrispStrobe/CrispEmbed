#include "tesseract_dawg_score.h"

#include "tesseract_recoder.h"

namespace tesseract_dawg_score {

static bool complete_word_match(const std::map<std::string, tesseract_dawg::Dawg> & dawgs,
                                const std::vector<int> & word) {
    // Tesseract keeps ordinary words and number words in separate graphs.
    // Prefer the ordinary system graph when both contain a word, but allow
    // the number graph to score a numeric candidate when system is absent.
    for (const char * name : { "lstm-system-dawg", "lstm-number-dawg" }) {
        const auto it = dawgs.find(name);
        if (it != dawgs.end() && tesseract_dawg::prefix_matches(it->second, word, true)) return true;
    }
    return false;
}

static bool prefix_word_match(const std::map<std::string, tesseract_dawg::Dawg> & dawgs,
                              const std::vector<int> & word) {
    for (const char * name : { "lstm-system-dawg", "lstm-number-dawg" }) {
        const auto it = dawgs.find(name);
        if (it != dawgs.end() && tesseract_dawg::prefix_matches(it->second, word, false)) return true;
    }
    return false;
}

float word_bonus(const std::vector<int> & prefix, const std::vector<std::vector<int>> & codes,
                 const std::vector<std::string> & tokens, const std::map<std::string, tesseract_dawg::Dawg> & dawgs,
                 bool include_final, bool include_prefix) {
    if (dawgs.find("lstm-system-dawg") == dawgs.end() && dawgs.find("lstm-number-dawg") == dawgs.end()) return 0.0f;
    std::vector<int> unichars, starts;
    if (!tesseract_recoder::compose_classes(prefix, codes, unichars, starts)) return 0.0f;
    float bonus = 0.0f;
    std::vector<int> word;
    for (int uid : unichars) {
        if (uid == 0) {
            if (!word.empty() && complete_word_match(dawgs, word)) bonus += 0.25f;
            word.clear();
        } else if (uid >= 0 && uid < (int)tokens.size()) {
            word.push_back(uid);
        }
    }
    if (include_final && !word.empty()) {
        if (complete_word_match(dawgs, word))
            bonus += 0.25f;
        else if (include_prefix && prefix_word_match(dawgs, word))
            bonus += 0.02f;
    }
    return bonus;
}

} // namespace tesseract_dawg_score
