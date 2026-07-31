// ocr_detect.h — DBNet text detection via ggml.
//
// Loads GGUF models (from convert-dbnet-to-gguf.py), runs ResNet-18 +
// FPNC + DBHead forward pass, and returns text bounding boxes.
//
// Usage:
//   ocr_detect::context *ctx;
//   ocr_detect::load(&ctx, "dbnet-ic15-q4_k.gguf");
//   auto boxes = ocr_detect::detect_file(ctx, "document.png");
//   for (auto& b : boxes) {
//       printf("text at (%.0f,%.0f)-(%.0f,%.0f) conf=%.2f\n",
//              b.x, b.y, b.x + b.w, b.y + b.h, b.score);
//   }
//   ocr_detect::free(ctx);

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ocr_detect {

struct text_box {
    float x, y, w, h; // axis-aligned bounding box in original image coords
    float score;      // mean probability inside the detected region
    float angle;      // rotation angle (degrees), 0 for axis-aligned
    // Oriented quad (4 corners in original image coords, clockwise from top-left)
    float qx[4], qy[4];
};

struct context;

enum class score_mode {
    fast,
    accurate,
};

// Geometry and DB post-processing controls shared by routed detector callers.
struct detect_options {
    float prob_threshold = 0.3f;
    float box_threshold = 0.5f;
    float unclip_ratio = 1.5f;
    int target_short_side = 736;
    int max_side = 2000;
    int min_height = 30;
    float width_height_ratio = 8.0f;
    int max_candidates = 1000;
    int dilation = 1;
    score_mode scoring = score_mode::fast;
};

detect_options rapid_defaults();

// Load DBNet GGUF. Returns true on success.
bool load(context ** ctx, const char * path, int n_threads = 1);

// Detect text regions from preprocessed pixels [3, H, W] CHW float32
// (already normalized with ImageNet mean/std and padded to multiple of 32).
// Coordinates are in the pixel space of the input.
std::vector<text_box> detect(context * ctx, const float * pixels, int H, int W, float prob_threshold = 0.3f,
                             float box_threshold = 0.5f, float unclip_ratio = 1.5f);

// Detect from image file. Handles resize, normalize, pad, and coordinate
// rescaling back to original image space.
std::vector<text_box> detect_file(context * ctx, const char * path, float prob_threshold = 0.3f,
                                  float box_threshold = 0.5f, float unclip_ratio = 1.5f, int target_short_side = 736);

// Detect from interleaved uint8 pixels. RGB and grayscale input are accepted;
// coordinates in the returned boxes refer to the original image dimensions.
std::vector<text_box> detect_rgb(context * ctx, const uint8_t * pixels, int width, int height, int channels,
                                 float prob_threshold = 0.3f, float box_threshold = 0.5f, float unclip_ratio = 1.5f,
                                 int target_short_side = 736);

std::vector<text_box> detect_rgb_ex(context * ctx, const uint8_t * pixels, int width, int height, int channels,
                                    const detect_options & options);

std::vector<text_box> detect_file_ex(context * ctx, const char * path, const detect_options & options);

// Apply the shared DB postprocessor to an already-computed probability map.
// This is used by detector ports whose neural graph is not the legacy DBNet
// graph, so Python/C++ geometry stays identical.
std::vector<text_box> postprocess_probability_map(const float * prob_map, int map_h, int map_w,
                                                  float prob_threshold = 0.3f, float box_threshold = 0.5f,
                                                  float unclip_ratio = 1.5f, int min_area = 1, float scale_x = 1.0f,
                                                  float scale_y = 1.0f, int dilation = 0, int max_candidates = 1000,
                                                  score_mode scoring = score_mode::fast);

// Get probability map from last detection (for debugging/visualization).
// Returns nullptr if no detection has been run yet.
// Shape: [H_padded, W_padded], row-major, values in [0, 1].
const float * get_prob_map(const context * ctx, int * out_h, int * out_w);

// Free resources.
void free(context * ctx);

} // namespace ocr_detect
