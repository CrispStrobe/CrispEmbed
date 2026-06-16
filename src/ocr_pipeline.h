// ocr_pipeline.h — Full OCR pipeline: text detection + recognition.
//
// Combines DBNet text detection (ocr_detect) with TrOCR text recognition
// (math_ocr) into a single pipeline: image → detected text regions with
// recognized text.
//
// Usage:
//   ocr_pipeline::context *ctx;
//   ocr_pipeline::load(&ctx, "dbnet.gguf", "trocr.gguf");
//   auto results = ocr_pipeline::run_file(ctx, "document.png");
//   for (auto& r : results) {
//       printf("(%.0f,%.0f)-(%.0f,%.0f): %s\n",
//              r.box.x, r.box.y, r.box.x+r.box.w, r.box.y+r.box.h, r.text.c_str());
//   }
//   ocr_pipeline::free(ctx);

#pragma once

#include "ocr_detect.h"
#include <string>
#include <vector>

namespace ocr_pipeline {

struct ocr_result {
    ocr_detect::text_box box;    // bounding box in original image coords
    std::string text;             // recognized text
    float confidence;             // detection confidence (from DBNet score)
    float rec_confidence;         // recognition confidence (mean per-char softmax)
    std::vector<float> char_conf; // per-character confidence (empty if unavailable)
};

struct context;

// Load both detection and recognition models.
// det_path: DBNet GGUF, rec_path: TrOCR GGUF.
bool load(context** ctx, const char* det_path, const char* rec_path,
          int n_threads = 4);

// Run full pipeline on an image file.
// Returns detected text regions sorted in reading order (top→bottom, left→right).
std::vector<ocr_result> run_file(context* ctx, const char* image_path,
                                  float prob_threshold = 0.3f,
                                  float box_threshold = 0.5f,
                                  int target_short_side = 736);

// Run recognition only on a single crop (no detection).
// Useful when you have pre-cropped text regions.
std::string recognize_file(context* ctx, const char* image_path);

// Free resources.
void free(context* ctx);

} // namespace ocr_pipeline
