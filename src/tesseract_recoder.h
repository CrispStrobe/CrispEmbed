#pragma once

#include <vector>

namespace tesseract_recoder {

// Segment collapsed CTC output classes into serialized Tesseract unichar
// codes. Returns false when the class stream cannot be composed exactly.
bool compose_classes(const std::vector<int> & labels, const std::vector<std::vector<int>> & codes,
                     std::vector<int> & unichars, std::vector<int> & starts);

// Returns whether a collapsed CTC class prefix can be segmented into
// serialized recoder entries. With allow_partial, the final entry may be a
// strict prefix of one serialized code during beam expansion.
bool prefix_legal(const std::vector<int> & prefix, const std::vector<std::vector<int>> & codes, bool allow_partial);

} // namespace tesseract_recoder
