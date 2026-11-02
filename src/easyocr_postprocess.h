#pragma once

#include <string>
#include <vector>

namespace easyocr_postprocess {

// EasyOCR get_image_list uses height 64 and rounds the recognizer canvas up
// to a multiple of 64 while preserving the aspect-sized crop inside it.
int recognizer_canvas_width(int crop_width, int crop_height);

// Width the crop is resized to before right-padding to the canvas. EasyOCR's
// compute_ratio_and_resize uses int(model_height * (width / height)), i.e. it
// truncates (python_floor=true). The historical native rule rounded up, which
// is one pixel wider for every non-integral ratio and changes the resampling
// grid of the whole line (fox line 2: 566 vs 565 px, confidence 0.60 vs 0.95).
int recognizer_resized_width(int crop_width, int crop_height, bool python_floor);

// EasyOCR recognition.adjust_contrast_grey (numpy percentile with linear
// interpolation, float64 arithmetic, uint8 truncation). Applied in place to the
// height-64 resized grey crop; returns true when the image was changed.
bool adjust_contrast_grey(std::vector<uint8_t> & image, double target);

// Grey value cv2.imread(path, IMREAD_GRAYSCALE) produces for an 8-bit RGB PNG
// (libpng rgb_to_gray fixed point, fitted exactly on 74,538 distinct colours).
// cv2.cvtColor(RGB2GRAY), which Python EasyOCR uses for array inputs, is the
// BT.601 rounding the native path uses by default.
inline uint8_t grey_libpng(uint8_t r, uint8_t g, uint8_t b) {
    return (uint8_t)((9795u * r + 19234u * g + 3739u * b) >> 15);
}

// EasyOCR's CTC convention: token 0 is blank and vocabulary entries are 1-based.
bool ctc_greedy_decode(const std::vector<int> & tokens, const std::vector<std::string> & vocabulary,
                       std::string * output, int * invalid_token = nullptr);

// Exact EasyOCR custom_mean confidence: prod(p) ** (2 / sqrt(n)), excluding blanks.
float confidence_custom_mean(const std::vector<float> & nonblank_probabilities);

// Validate a dictionary/vocabulary before exposing it through GGUF metadata.
bool validate_vocabulary(const std::vector<std::string> & vocabulary, std::string * error = nullptr);
bool vocabulary_contains(const std::vector<std::string> & vocabulary, const std::string & token);

} // namespace easyocr_postprocess
