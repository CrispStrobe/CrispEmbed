#pragma once

#include <vector>

namespace tesseract_recoder {

// Segment collapsed CTC output classes into serialized Tesseract unichar
// codes. Returns false when the class stream cannot be composed exactly.
bool compose_classes(const std::vector<int> & labels, const std::vector<std::vector<int>> & codes,
                     std::vector<int> & unichars, std::vector<int> & starts);

} // namespace tesseract_recoder
