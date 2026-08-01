#pragma once

#include "tesseract_dawg.h"

#include <map>
#include <string>
#include <vector>

namespace tesseract_dawg_score {

// Return the diagnostic dictionary bonus for complete words in a recoder
// prefix. The CTC probability is never modified by this helper.
float word_bonus(const std::vector<int> & prefix, const std::vector<std::vector<int>> & codes,
                 const std::vector<std::string> & tokens,
                 const std::map<std::string, tesseract_dawg::Dawg> & dawgs, bool include_final);

} // namespace tesseract_dawg_score
