#pragma once

#include "easyocr_layout.h"
#include "ocr_detect.h"

#include <cstdint>
#include <string>
#include <vector>

namespace easyocr_pipeline {

struct result {
    easyocr_layout::word word;
    float detector_confidence = 0.0f;
    easyocr_layout::normalized_box normalized;
    int crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;
};

struct context;

bool load(context ** out, const char * detector_path, const char * recognizer_path, int n_threads = 1);
void free(context * ctx);

void set_ordering_mode(context * ctx, easyocr_layout::ordering_mode mode);
easyocr_layout::ordering_mode ordering_mode(const context * ctx);

std::vector<result> run_raw(context * ctx, const uint8_t * pixels, int width, int height, int channels);
std::vector<result> run_file(context * ctx, const char * image_path);

} // namespace easyocr_pipeline
