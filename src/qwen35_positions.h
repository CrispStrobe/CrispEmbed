#pragma once
#include <vector>
namespace qwen35_ocr {
struct position_outputs {
    // Unpadded batch=1: chronological text plane followed by temporal/height/width.
    std::vector<float> generation;
    std::vector<float> axes;
    int rope_delta = 0;
};
// Still images only. Each contiguous modality=1 span consumes one [1,H,W] grid.
// Text is modality=0, including vision delimiters. Reject videos/padding.
bool image_positions(const std::vector<int> & modalities, const std::vector<int> & grid_thw, int spatial_merge,
                     position_outputs & output);
// Derives broadcast temporal/height/width coordinates from native cache length.
bool cached_positions(int cache_tokens, int tokens, int rope_delta, std::vector<float> & axes);
} // namespace qwen35_ocr
