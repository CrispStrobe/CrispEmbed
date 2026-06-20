// ocr_pipeline.cpp — Full OCR pipeline: DBNet detection + TrOCR recognition.
//
// Pipeline steps:
//   1. Load image
//   2. Run DBNet text detection → list of text_box
//   3. For each box: crop from original image → run TrOCR → text string
//   4. Return results sorted in reading order

#include "ocr_pipeline.h"
#include "ocr_detect.h"
#include "math_ocr.h"

// stb_image declarations (implementation lives in image_preprocess.cpp)
extern "C" {
    typedef unsigned char stbi_uc;
    stbi_uc *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
    void stbi_image_free(void *retval_from_stbi_load);
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ocr_pipeline {

struct context {
    ocr_detect::context* det = nullptr;
    math_ocr_context* rec = nullptr;
    int n_threads = 4;
    bool bench = false;
};

bool load(context** out, const char* det_path, const char* rec_path,
          int n_threads) {
    auto* ctx = new context();
    *out = ctx;
    ctx->n_threads = n_threads;
    ctx->bench = (std::getenv("CRISPEMBED_OCR_PIPELINE_BENCH") != nullptr);

    // Load detection model
    if (!ocr_detect::load(&ctx->det, det_path, n_threads)) {
        fprintf(stderr, "ocr_pipeline: failed to load detection model: %s\n", det_path);
        delete ctx;
        *out = nullptr;
        return false;
    }

    // Load recognition model
    ctx->rec = math_ocr_init(rec_path, n_threads);
    if (!ctx->rec) {
        fprintf(stderr, "ocr_pipeline: failed to load recognition model: %s\n", rec_path);
        ocr_detect::free(ctx->det);
        delete ctx;
        *out = nullptr;
        return false;
    }

    const math_ocr_hparams* hp = math_ocr_get_hparams(ctx->rec);
    fprintf(stderr, "ocr_pipeline: loaded det + rec (vocab=%d)\n", hp->vocab_size);
    return true;
}

// Crop a region from an RGB image. Returns RGB uint8 buffer.
static std::vector<uint8_t> crop_image(const unsigned char* img,
                                        int img_w, int img_h,
                                        int crop_x, int crop_y,
                                        int crop_w, int crop_h) {
    // Clamp to image bounds
    crop_x = std::max(0, crop_x);
    crop_y = std::max(0, crop_y);
    if (crop_x + crop_w > img_w) crop_w = img_w - crop_x;
    if (crop_y + crop_h > img_h) crop_h = img_h - crop_y;
    if (crop_w <= 0 || crop_h <= 0) return {};

    std::vector<uint8_t> crop(crop_w * crop_h * 3);
    for (int y = 0; y < crop_h; y++) {
        const uint8_t* src = img + ((crop_y + y) * img_w + crop_x) * 3;
        uint8_t* dst = crop.data() + y * crop_w * 3;
        memcpy(dst, src, crop_w * 3);
    }
    return crop;
}

std::vector<ocr_result> run_file(context* ctx, const char* image_path,
                                  float prob_threshold, float box_threshold,
                                  int target_short_side) {
    if (!ctx || !ctx->det || !ctx->rec || !image_path) return {};

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Step 1: Detect text regions
    auto t_detect = std::chrono::steady_clock::now();
    auto boxes = ocr_detect::detect_file(ctx->det, image_path,
                                          prob_threshold, box_threshold,
                                          1.5f, target_short_side);
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_detect).count();
        fprintf(stderr, "[ocr_pipeline-bench] detect: %.1f ms (%zu boxes)\n", ms, boxes.size());
    }

    if (boxes.empty()) {
        fprintf(stderr, "ocr_pipeline: no text detected in %s\n", image_path);
        return {};
    }
    fprintf(stderr, "ocr_pipeline: detected %zu text regions\n", boxes.size());

    // Step 2: Load original image for cropping
    int img_w, img_h, img_c;
    unsigned char* img = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
    if (!img) {
        fprintf(stderr, "ocr_pipeline: cannot load image for cropping: %s\n", image_path);
        return {};
    }

    // Step 3: For each box, crop and recognize
    std::vector<ocr_result> results;
    results.reserve(boxes.size());

    double rec_total_ms = 0.0;
    for (size_t i = 0; i < boxes.size(); i++) {
        auto& b = boxes[i];

        // Crop with some padding
        int pad = 2;
        int cx = std::max(0, (int)b.x - pad);
        int cy = std::max(0, (int)b.y - pad);
        int cw = (int)b.w + 2 * pad;
        int ch = (int)b.h + 2 * pad;

        auto crop = crop_image(img, img_w, img_h, cx, cy, cw, ch);
        if (crop.empty()) continue;

        // Actual crop dimensions after clamping
        int actual_w = std::min(cw, img_w - cx);
        int actual_h = std::min(ch, img_h - cy);
        if (actual_w <= 0 || actual_h <= 0) continue;

        // Run recognition on the crop
        auto t_rec = std::chrono::steady_clock::now();
        int out_len = 0;
        const char* text = math_ocr_recognize_raw(ctx->rec,
                                                    crop.data(),
                                                    actual_w, actual_h, 3,
                                                    &out_len);
        if (bench) {
            rec_total_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_rec).count();
        }

        ocr_result r;
        r.box = b;
        r.confidence = b.score;
        r.text = text ? std::string(text, out_len) : "";

        if (!r.text.empty()) {
            results.push_back(std::move(r));
        }
    }

    stbi_image_free(img);

    if (bench) {
        double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_total).count();
        fprintf(stderr, "[ocr_pipeline-bench] recognize (all boxes): %.1f ms\n", rec_total_ms);
        fprintf(stderr, "[ocr_pipeline-bench] total: %.1f ms\n", total_ms);
    }

    fprintf(stderr, "ocr_pipeline: recognized %zu/%zu regions\n",
            results.size(), boxes.size());
    return results;
}

std::string recognize_file(context* ctx, const char* image_path) {
    if (!ctx || !ctx->rec || !image_path) return "";
    int out_len = 0;
    const char* text = math_ocr_recognize_file(ctx->rec, image_path, &out_len);
    return text ? std::string(text, out_len) : "";
}

void free(context* ctx) {
    if (!ctx) return;
    if (ctx->det) ocr_detect::free(ctx->det);
    if (ctx->rec) math_ocr_free(ctx->rec);
    delete ctx;
}

} // namespace ocr_pipeline
