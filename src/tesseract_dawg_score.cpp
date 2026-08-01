#include "tesseract_dawg_score.h"

#include "tesseract_recoder.h"

namespace tesseract_dawg_score {

float word_bonus(const std::vector<int> & prefix, const std::vector<std::vector<int>> & codes,
                 const std::vector<std::string> & tokens,
                 const std::map<std::string, tesseract_dawg::Dawg> & dawgs, bool include_final) {
    const auto it = dawgs.find("lstm-system-dawg");
    if (it == dawgs.end()) return 0.0f;
    std::vector<int> unichars, starts;
    if (!tesseract_recoder::compose_classes(prefix, codes, unichars, starts)) return 0.0f;
    float bonus = 0.0f;
    std::vector<int> word;
    for (int uid : unichars) {
        if (uid == 0) {
            if (!word.empty() && tesseract_dawg::prefix_matches(it->second, word, true)) bonus += 0.25f;
            word.clear();
        } else if (uid >= 0 && uid < (int)tokens.size()) {
            word.push_back(uid);
        }
    }
    if (include_final && !word.empty() && tesseract_dawg::prefix_matches(it->second, word, true)) bonus += 0.25f;
    return bonus;
}

} // namespace tesseract_dawg_score
