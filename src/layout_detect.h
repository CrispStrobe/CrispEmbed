// layout_detect.h — Document layout analysis via RT-DETRv2 (ggml).
//
// Detects document regions: text, title, table, figure, formula, caption,
// section_header, list_item, footnote, page_header, page_footer, code,
// document_index, checkbox, form, key_value_region (17 classes).
//
// Architecture: ResNet-50 backbone + hybrid FPN/PAN encoder +
// transformer decoder with deformable cross-attention (300 queries).
//
// Source: docling-layout-heron (Apache 2.0).
//
// Usage:
//   layout_detect::context *ctx;
//   layout_detect::load(&ctx, "layout-heron.gguf");
//   auto regions = layout_detect::detect_file(ctx, "page.png");
//   for (auto& r : regions) {
//       printf("%s at (%.0f,%.0f)-(%.0f,%.0f) score=%.2f\n",
//              r.label, r.x1, r.y1, r.x2, r.y2, r.score);
//   }
//   layout_detect::free(ctx);

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace layout_detect {

// 17 document layout classes
enum class label_id : int {
    caption = 0, footnote, formula, list_item, page_footer,
    page_header, picture, section_header, table, text,
    title, document_index, code, checkbox_selected,
    checkbox_unselected, form, key_value_region,
    NUM_CLASSES
};

const char* label_name(label_id id);

struct region {
    float x1, y1, x2, y2;    // bbox in original image coordinates
    float score;
    label_id label;
    const char* label_name;   // pointer to static string
};

struct context;

// Load RT-DETRv2 GGUF model. Returns true on success.
bool load(context** ctx, const char* path, int n_threads = 1);

// Detect layout regions from image file. Handles resize to 640×640,
// normalize, and coordinate rescaling.
std::vector<region> detect_file(context* ctx, const char* path,
                                 float score_threshold = 0.3f);

// Detect from preprocessed pixels [3, 640, 640] CHW float32.
std::vector<region> detect(context* ctx, const float* pixels,
                            int orig_h, int orig_w,
                            float score_threshold = 0.3f);

// Free resources.
void free(context* ctx);

} // namespace layout_detect
