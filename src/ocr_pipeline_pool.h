// ocr_pipeline_pool.h — bounded pool of isolated DBNet+TrOCR contexts.

#pragma once

#include "ocr_pipeline.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ocr_pipeline_pool {

struct context;

bool load(context ** out, const char * det_path, const char * rec_path, int pool_size = 1, int n_threads = 1);

std::vector<ocr_pipeline::ocr_result> run_file(context * ctx, const char * image_path, float prob_threshold = 0.3f,
                                               float box_threshold = 0.5f, int target_short_side = 736);

std::vector<ocr_pipeline::ocr_result> run_raw(context * ctx, const uint8_t * pixels, int width, int height,
                                              int channels, float prob_threshold = 0.3f, float box_threshold = 0.5f,
                                              int target_short_side = 736);

std::string recognize_file(context * ctx, const char * image_path);

void free(context * ctx);

} // namespace ocr_pipeline_pool
