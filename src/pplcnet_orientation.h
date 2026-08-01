#pragma once

#include <cstdint>
#include <string>

namespace pplcnet_orientation {

struct result {
    int angle = 0;
    float confidence = 0.0f;
    float probabilities[2] = {};
    float logits[2] = {};
};

struct context;

context * init(const char * path, int n_threads = 1);
void free(context * ctx);
result classify_raw(context * ctx, const uint8_t * pixels, int width, int height, int channels);
result classify_file(context * ctx, const char * path);

} // namespace pplcnet_orientation
