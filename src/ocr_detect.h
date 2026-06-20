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
    float x, y, w, h;    // axis-aligned bounding box in original image coords
    float score;          // mean probability inside the detected region
    float angle;          // rotation angle (degrees), 0 for axis-aligned
    // Oriented quad (4 corners in original image coords, clockwise from top-left)
    float qx[4], qy[4];
};

struct context;

// Load DBNet GGUF. Returns true on success.
bool load(context** ctx, const char* path, int n_threads = 1);

// Detect text regions from preprocessed pixels [3, H, W] CHW float32
// (already normalized with ImageNet mean/std and padded to multiple of 32).
// Coordinates are in the pixel space of the input.
std::vector<text_box> detect(context* ctx, const float* pixels, int H, int W,
                              float prob_threshold = 0.3f,
                              float box_threshold = 0.5f,
                              float unclip_ratio = 1.5f);

// Detect from image file. Handles resize, normalize, pad, and coordinate
// rescaling back to original image space.
std::vector<text_box> detect_file(context* ctx, const char* path,
                                   float prob_threshold = 0.3f,
                                   float box_threshold = 0.5f,
                                   float unclip_ratio = 1.5f,
                                   int target_short_side = 736);

// Get probability map from last detection (for debugging/visualization).
// Returns nullptr if no detection has been run yet.
// Shape: [H_padded, W_padded], row-major, values in [0, 1].
const float* get_prob_map(const context* ctx, int* out_h, int* out_w);

// Free resources.
void free(context* ctx);

} // namespace ocr_detect
