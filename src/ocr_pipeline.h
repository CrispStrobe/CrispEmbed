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

// TrOCR's narrow decoder loses recognition quality under Q4_K. Q4 is accepted
// only when the caller explicitly sets CRISPEMBED_DEBUG_ALLOW_OCR_Q4=1.
bool is_dangerous_q4_recognizer_path(const char * rec_path);
bool dangerous_q4_override_enabled();

struct ocr_result {
    ocr_detect::text_box box;           // bounding box in original image coords
    std::string text;                   // recognized text
    float confidence;                   // detection confidence (from DBNet score)
    float rec_confidence;               // recognition confidence (mean per-char softmax)
    std::vector<float> char_conf;       // per-character confidence (empty if unavailable)
    bool orientation_corrected = false; // classical 180° line correction applied
    int orientation_angle = 0;          // detected line angle in degrees, when available
    float orientation_confidence = 0.0f;
};

struct context;

// Load both detection and recognition models.
// det_path: DBNet GGUF, rec_path: TrOCR GGUF. The recommended recognizer is
// trocr-small-printed-q8_0.gguf; TrOCR Q4_K is rejected by default.
bool load(context ** ctx, const char * det_path, const char * rec_path, int n_threads = 1);

// Run full pipeline on an image file.
// Returns detected text regions sorted in reading order (top→bottom, left→right).
std::vector<ocr_result> run_file(context * ctx, const char * image_path, float prob_threshold = 0.3f,
                                 float box_threshold = 0.5f, int target_short_side = 736,
                                 const ocr_detect::detect_options * geometry = nullptr);

// Run detection and recognition on interleaved uint8 pixels. The input is
// borrowed for the duration of the call and may be RGB or grayscale.
std::vector<ocr_result> run_raw(context * ctx, const uint8_t * pixels, int width, int height, int channels,
                                float prob_threshold = 0.3f, float box_threshold = 0.5f, int target_short_side = 736,
                                const ocr_detect::detect_options * geometry = nullptr);

// Run recognition only on a single crop (no detection).
// Useful when you have pre-cropped text regions.
std::string recognize_file(context * ctx, const char * image_path);

// Free resources.
void free(context * ctx);

} // namespace ocr_pipeline
