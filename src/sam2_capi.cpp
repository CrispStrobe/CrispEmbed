// sam2_capi.cpp — C API of the SAM 2.1 engine (crispembed_sam2_*), declared in crispembed.h.
//
// Kept apart from crispembed.cpp so that the engine also builds as the small standalone library
// crispembed-sam2 (CMake option CRISPEMBED_SAM2_LIBRARY), which applications that only segment link.

#include "crispembed.h"
#include "sam2.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {

void * crispembed_sam2_init(const char * model_path, int n_threads) {
    return sam2_init(model_path, n_threads);
}

void crispembed_sam2_free(void * ctx) {
    sam2_free(static_cast<sam2_context *>(ctx));
}

int crispembed_sam2_image_size(const void * ctx) {
    return sam2_image_size(static_cast<const sam2_context *>(ctx));
}

int crispembed_sam2_mask_size(const void * ctx) {
    return sam2_mask_size(static_cast<const sam2_context *>(ctx));
}

const char * crispembed_sam2_backend(const void * ctx) {
    return sam2_backend_name(static_cast<const sam2_context *>(ctx));
}

int crispembed_sam2_set_image(void * ctx, const uint8_t * rgb, int width, int height) {
    return sam2_set_image_rgb(static_cast<sam2_context *>(ctx), rgb, width, height);
}

int crispembed_sam2_set_image_f32(void * ctx, const float * chw) {
    return sam2_set_image_f32(static_cast<sam2_context *>(ctx), chw);
}

int crispembed_sam2_predict(void * ctx, const float * points_xy, const int * labels, int n_points, float * mask_logits,
                            float * scores) {
    return sam2_predict(static_cast<sam2_context *>(ctx), points_xy, labels, n_points, mask_logits, scores);
}

int crispembed_sam2_process(void * handle, const uint8_t * rgb, int width, int height, const float * points_xy,
                            const int * labels, int n_points, const float * box_xyxy, int multimask,
                            uint8_t ** out_masks, float * out_scores, int * out_count) {
    auto * ctx = static_cast<sam2_context *>(handle);
    if (!ctx || !rgb || !out_masks || !out_scores || !out_count || (n_points > 0 && (!points_xy || !labels))) return -1;
    if (n_points <= 0 && !box_xyxy) return -1;
    *out_masks = nullptr;
    *out_count = 0;
    if (int status = sam2_set_image_rgb(ctx, rgb, width, height)) return status;
    const int S = sam2_image_size(ctx), M = sam2_mask_size(ctx);
    // Prompts in the network frame: box corners (labels 2, 3) first, then the points.
    std::vector<float> xy;
    std::vector<int> lab;
    const float sx = (float)S / (float)width, sy = (float)S / (float)height;
    if (box_xyxy) {
        xy.insert(xy.end(), { box_xyxy[0] * sx, box_xyxy[1] * sy, box_xyxy[2] * sx, box_xyxy[3] * sy });
        lab.insert(lab.end(), { 2, 3 });
    }
    for (int i = 0; i < n_points; i++) {
        xy.push_back(points_xy[2 * i] * sx);
        xy.push_back(points_xy[2 * i + 1] * sy);
        lab.push_back(labels[i]);
    }
    std::vector<float> logits((size_t)4 * M * M);
    float scores[4];
    if (int status = sam2_predict(ctx, xy.data(), lab.data(), (int)lab.size(), logits.data(), scores)) return status;

    // Which tokens: 1..3 for several masks; else token 0 unless it is unstable (SAM 2's dynamic fallback).
    std::vector<int> tokens;
    if (multimask) {
        tokens = { 1, 2, 3 };
    } else {
        int inner = 0, outer = 0;
        for (int i = 0; i < M * M; i++) {
            inner += logits[i] > 0.05f;
            outer += logits[i] > -0.05f;
        }
        const float stability = outer > 0 ? (float)inner / (float)outer : 1.0f;
        int best = 1;
        for (int t = 2; t < 4; t++)
            if (scores[t] > scores[best]) best = t;
        tokens = { stability >= 0.98f ? 0 : best };
    }

    // Bilinear enlargement to the photo (align_corners=False), threshold at 0.
    auto out = static_cast<uint8_t *>(std::malloc(tokens.size() * (size_t)width * height));
    if (!out) return -4;
    const float fx = (float)M / (float)width, fy = (float)M / (float)height;
    for (size_t n = 0; n < tokens.size(); n++) {
        const float * map = logits.data() + (size_t)tokens[n] * M * M;
        uint8_t * target = out + n * (size_t)width * height;
        for (int y = 0; y < height; y++) {
            const float sy0 = std::max(fy * ((float)y + 0.5f) - 0.5f, 0.0f);
            const int y0 = std::min((int)sy0, M - 1), y1 = std::min(y0 + 1, M - 1);
            const float wy = sy0 - (float)y0;
            for (int x = 0; x < width; x++) {
                const float sx0 = std::max(fx * ((float)x + 0.5f) - 0.5f, 0.0f);
                const int x0 = std::min((int)sx0, M - 1), x1 = std::min(x0 + 1, M - 1);
                const float wx = sx0 - (float)x0;
                const float a = map[y0 * M + x0] + wx * (map[y0 * M + x1] - map[y0 * M + x0]);
                const float b = map[y1 * M + x0] + wx * (map[y1 * M + x1] - map[y1 * M + x0]);
                target[(size_t)y * width + x] = (a + wy * (b - a)) > 0.0f ? 255 : 0;
            }
        }
        out_scores[n] = scores[tokens[n]];
    }
    *out_masks = out;
    *out_count = (int)tokens.size();
    return 0;
}

void crispembed_sam2_free_masks(uint8_t * masks) {
    std::free(masks);
}

} // extern "C"
