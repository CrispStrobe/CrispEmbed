#pragma once

#include <string>
#include <vector>

namespace easyocr_postprocess {

// EasyOCR's CTC convention: token 0 is blank and vocabulary entries are 1-based.
bool ctc_greedy_decode(const std::vector<int> & tokens, const std::vector<std::string> & vocabulary,
                       std::string * output, int * invalid_token = nullptr);

// Exact EasyOCR custom_mean confidence: prod(p) ** (2 / sqrt(n)), excluding blanks.
float confidence_custom_mean(const std::vector<float> & nonblank_probabilities);

// Validate a dictionary/vocabulary before exposing it through GGUF metadata.
bool validate_vocabulary(const std::vector<std::string> & vocabulary, std::string * error = nullptr);
bool vocabulary_contains(const std::vector<std::string> & vocabulary, const std::string & token);

} // namespace easyocr_postprocess
