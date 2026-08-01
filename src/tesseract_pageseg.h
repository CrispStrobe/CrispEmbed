#pragma once

#include "ocr_detect.h"

#include <cstdint>
#include <vector>

namespace tesseract_pageseg {

// Fast classical text-line proposal path. It intentionally returns the same
// geometry contract as DBNet so it can be compared and selected by env gate.
std::vector<ocr_detect::text_box> segment_gray(const uint8_t * gray, int width, int height);

}
