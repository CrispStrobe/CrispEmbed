#include "easyocr_postprocess.h"

#include <cmath>
#include <unordered_set>

namespace easyocr_postprocess {

bool ctc_greedy_decode(const std::vector<int> & tokens, const std::vector<std::string> & vocabulary,
                       std::string * output, int * invalid_token) {
    if (!output) return false;
    output->clear();
    if (invalid_token) *invalid_token = -1;
    int previous = 0;
    for (const int token : tokens) {
        if (token < 0 || token > (int)vocabulary.size()) {
            if (invalid_token) *invalid_token = token;
            return false;
        }
        if (token != 0 && token != previous) *output += vocabulary[(size_t)token - 1];
        previous = token;
    }
    return true;
}

float confidence_custom_mean(const std::vector<float> & probabilities) {
    if (probabilities.empty()) return 0.0f;
    double log_product = 0.0;
    for (const float probability : probabilities) {
        if (!(probability > 0.0f) || probability > 1.0f) return 0.0f;
        log_product += std::log((double)probability);
    }
    return (float)std::exp(log_product * (2.0 / std::sqrt((double)probabilities.size())));
}

bool validate_vocabulary(const std::vector<std::string> & vocabulary, std::string * error) {
    std::unordered_set<std::string> seen;
    for (size_t i = 0; i < vocabulary.size(); ++i) {
        if (vocabulary[i].empty()) {
            if (error) *error = "vocabulary entry " + std::to_string(i) + " is empty";
            return false;
        }
        if (!seen.insert(vocabulary[i]).second) {
            if (error) *error = "duplicate vocabulary entry: " + vocabulary[i];
            return false;
        }
    }
    return true;
}

bool vocabulary_contains(const std::vector<std::string> & vocabulary, const std::string & token) {
    for (const auto & item : vocabulary)
        if (item == token) return true;
    return false;
}

} // namespace easyocr_postprocess
