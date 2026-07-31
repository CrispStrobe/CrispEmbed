#pragma once

#include <cstdint>
#include <vector>
#include <string>

namespace ppocrv6_det {

struct box {
    float x, y, w, h, score;
    float qx[4] = {}, qy[4] = {};
};
struct context;

context * init(const char * path, int n_threads = 1);
void free(context * ctx);
std::vector<box> detect_raw(context * ctx, const uint8_t * pixels, int width, int height, int channels,
                            float threshold = 0.3f);
std::vector<box> detect_file(context * ctx, const char * path, float threshold = 0.3f);
const float * last_probability(const context * ctx, int * height, int * width);
const float * last_stage(const context * ctx, const char * name, size_t * n);

} // namespace ppocrv6_det
