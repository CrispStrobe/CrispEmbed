#pragma once
#include "josev2b_text_frontend.h"
#include <string>
#include <vector>

namespace josev2b {
// Internal SystemOne JSON only. Probabilities are native F32, ordered exactly as
// encoded.questions/option_ids. No normalization, calibration, or model access.
class answer_serializer {
public:
    answer_serializer();
    bool enabled() const { return enabled_; }
    bool serialize(const std::string & request_json, const encoded_text & encoded,
                   const std::vector<std::vector<float>> & probabilities, std::string & output_json,
                   std::string & error) const;

private:
    bool enabled_;
};
#ifdef CRISPEMBED_JOSEV_ANSWER_TESTING
namespace answer_test {
std::string rounded4(double value);
}
#endif
} // namespace josev2b
