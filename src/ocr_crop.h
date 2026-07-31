// Shared line-region extraction for detector-backed OCR engines.
#pragma once

#include <cstdint>
#include <vector>

namespace ocr_crop {

enum class resize_mode { stretch, preserve_aspect };

struct orientation_info {
    int angle = 0;
    float confidence = 0.0f;
    bool corrected = false;
};

// Shared line-crop preparation contract. A zero target dimension preserves
// the corresponding input dimension; max_width=0 means no width limit.
struct prepare_options {
    int target_width = 0;
    int target_height = 0;
    int max_width = 0;
    resize_mode mode = resize_mode::preserve_aspect;
    bool grayscale = false;
    bool pad_to_target = false;
    uint8_t pad_value = 255;
};

// Extract a clamped interleaved crop with symmetric padding where possible.
// The returned buffer is tightly packed and keeps the requested channel count.
std::vector<uint8_t> extract(const uint8_t * pixels, int width, int height, int channels, int x, int y, int crop_w,
                             int crop_h, int padding, int * out_width, int * out_height);

// Perspective-rectify an ordered quadrilateral (TL, TR, BR, BL).  This is
// the crop contract used by PaddleOCR/RapidOCR after DBPostProcess; retaining
// the polygon is essential for skewed and rotated text lines.
std::vector<uint8_t> extract_quad(const uint8_t * pixels, int width, int height, int channels, const float qx[4],
                                  const float qy[4], int padding, int * out_width, int * out_height);

// Convert an extracted crop to one shared recognizer geometry. The function
// is deterministic, does not mutate the input, and supports RGB or grayscale
// output contracts. It is deliberately separate from model-specific tensor
// normalization, which remains inside each recognizer implementation.
std::vector<uint8_t> prepare(const uint8_t * pixels, int width, int height, int channels,
                             const prepare_options & options, int * out_width, int * out_height, int * out_channels);

// Apply the existing classical 0°/180° text-line check in place. Returns true
// when a 180° rotation was applied.
bool orient_180_rgb(std::vector<uint8_t> & pixels, int width, int height);
bool orient_180_gray(std::vector<uint8_t> & pixels, int width, int height);
void rotate_180_rgb(std::vector<uint8_t> & pixels, int width, int height);
orientation_info orient_180_rgb_info(std::vector<uint8_t> & pixels, int width, int height);
orientation_info orient_180_gray_info(std::vector<uint8_t> & pixels, int width, int height);

} // namespace ocr_crop
