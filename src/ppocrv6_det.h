#pragma once

#include <cstdint>
#include <vector>

namespace ppocrv6_det {

struct box {
    float x, y, w, h, score;
};
struct context;

context * init(const char * path, int n_threads = 1);
void free(context * ctx);
std::vector<box> detect_raw(context * ctx, const uint8_t * pixels, int width, int height, int channels,
                            float threshold = 0.3f);
std::vector<box> detect_file(context * ctx, const char * path, float threshold = 0.3f);

} // namespace ppocrv6_det
