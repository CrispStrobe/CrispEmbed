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
#include "classical_preproc.h"

// stb_image declarations (implementation lives in image_preprocess.cpp)
extern "C" {
typedef unsigned char stbi_uc;
stbi_uc * stbi_load(char const * filename, int * x, int * y, int * channels_in_file, int desired_channels);
void stbi_image_free(void * retval_from_stbi_load);
}

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace ocr_pipeline {

bool is_dangerous_q4_recognizer_path(const char * rec_path) {
    if (!rec_path) return false;
    std::string path(rec_path);
    std::transform(path.begin(), path.end(), path.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return path.find("trocr-") != std::string::npos && path.find("q4_k") != std::string::npos;
}

bool dangerous_q4_override_enabled() {
    const char * value = std::getenv("CRISPEMBED_DEBUG_ALLOW_OCR_Q4");
    return value && (!std::strcmp(value, "1") || !std::strcmp(value, "true") || !std::strcmp(value, "yes"));
}

struct context {
    ocr_detect::context * det = nullptr;
    math_ocr_context * rec = nullptr;
    int n_threads = 1;
    bool bench = false;
    std::mutex infer_mutex; // decoder/context state is mutable across calls
};

bool load(context ** out, const char * det_path, const char * rec_path, int n_threads) {
    if (is_dangerous_q4_recognizer_path(rec_path) && !dangerous_q4_override_enabled()) {
        fprintf(stderr,
                "ocr_pipeline: refusing TrOCR Q4_K recognizer '%s'; use trocr-small-printed-q8_0.gguf "
                "or explicitly set CRISPEMBED_DEBUG_ALLOW_OCR_Q4=1\n",
                rec_path ? rec_path : "(null)");
        if (out) *out = nullptr;
        return false;
    }
    auto * ctx = new context();
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

    const math_ocr_hparams * hp = math_ocr_get_hparams(ctx->rec);
    fprintf(stderr, "ocr_pipeline: loaded det + rec (vocab=%d)\n", hp->vocab_size);
    return true;
}

// Crop a region from an RGB image. Returns RGB uint8 buffer.
static std::vector<uint8_t> crop_image(const unsigned char * img, int img_w, int img_h, int crop_x, int crop_y,
                                       int crop_w, int crop_h) {
    // Clamp to image bounds
    crop_x = std::max(0, crop_x);
    crop_y = std::max(0, crop_y);
    if (crop_x + crop_w > img_w) crop_w = img_w - crop_x;
    if (crop_y + crop_h > img_h) crop_h = img_h - crop_y;
    if (crop_w <= 0 || crop_h <= 0) return {};

    std::vector<uint8_t> crop(crop_w * crop_h * 3);
    for (int y = 0; y < crop_h; y++) {
        const uint8_t * src = img + ((crop_y + y) * img_w + crop_x) * 3;
        uint8_t * dst = crop.data() + y * crop_w * 3;
        memcpy(dst, src, crop_w * 3);
    }
    return crop;
}

static void orient_text_crop(std::vector<uint8_t> & crop, int w, int h) {
    if (w < 8 || h < 8) return;
    std::vector<uint8_t> gray((size_t)w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const uint8_t * p = crop.data() + ((size_t)y * w + x) * 3;
            gray[(size_t)y * w + x] = (uint8_t)((77 * p[0] + 150 * p[1] + 29 * p[2] + 128) >> 8);
        }
    }
    float confidence = 0.0f;
    if (detect_text_angle(gray.data(), w, h, &confidence) != 180 || confidence < 0.75f) return;
    for (int y = 0; y < h / 2 + (h & 1); ++y) {
        for (int x = 0; x < w; ++x) {
            const size_t a = ((size_t)y * w + x) * 3;
            const size_t b = ((size_t)(h - 1 - y) * w + (w - 1 - x)) * 3;
            if (a >= b) continue;
            for (int c = 0; c < 3; ++c) std::swap(crop[a + c], crop[b + c]);
        }
    }
}

std::vector<ocr_result> run_raw(context * ctx, const uint8_t * raw, int img_w, int img_h, int img_c,
                                float prob_threshold, float box_threshold, int target_short_side) {
    if (!ctx || !ctx->det || !ctx->rec || !raw || img_w <= 0 || img_h <= 0 || img_c <= 0) return {};
    std::lock_guard<std::mutex> lock(ctx->infer_mutex);

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Step 1: Detect text regions
    auto t_detect = std::chrono::steady_clock::now();
    auto boxes = ocr_detect::detect_rgb(ctx->det, raw, img_w, img_h, img_c, prob_threshold, box_threshold, 1.5f,
                                        target_short_side);
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_detect).count();
        fprintf(stderr, "[ocr_pipeline-bench] detect: %.1f ms (%zu boxes)\n", ms, boxes.size());
    }

    if (boxes.empty()) {
        fprintf(stderr, "ocr_pipeline: no text detected\n");
        return {};
    }
    fprintf(stderr, "ocr_pipeline: detected %zu text regions\n", boxes.size());

    // Step 2: Collect valid crops, batch-encode, then decode sequentially.
    struct crop_entry {
        std::vector<uint8_t> data;
        int w, h;
        size_t box_idx;
    };
    std::vector<crop_entry> crop_entries;
    crop_entries.reserve(boxes.size());

    for (size_t i = 0; i < boxes.size(); i++) {
        auto & b = boxes[i];
        int pad = 2;
        int cx = std::max(0, (int)b.x - pad);
        int cy = std::max(0, (int)b.y - pad);
        int cw = (int)b.w + 2 * pad;
        int ch = (int)b.h + 2 * pad;

        auto crop = crop_image(raw, img_w, img_h, cx, cy, cw, ch);
        if (crop.empty()) continue;

        int actual_w = std::min(cw, img_w - cx);
        int actual_h = std::min(ch, img_h - cy);
        if (actual_w <= 0 || actual_h <= 0) continue;

        orient_text_crop(crop, actual_w, actual_h);
        crop_entries.push_back({ std::move(crop), actual_w, actual_h, i });
    }

    std::vector<ocr_result> results;
    results.reserve(crop_entries.size());

    if (!crop_entries.empty()) {
        // Batch-encode all crops in a single encoder pass
        std::vector<const uint8_t *> ptrs(crop_entries.size());
        std::vector<int> cws(crop_entries.size()), chs(crop_entries.size());
        for (size_t i = 0; i < crop_entries.size(); i++) {
            ptrs[i] = crop_entries[i].data.data();
            cws[i] = crop_entries[i].w;
            chs[i] = crop_entries[i].h;
        }

        auto t_enc = std::chrono::steady_clock::now();
        bool enc_ok =
            math_ocr_encode_batch_raw(ctx->rec, ptrs.data(), cws.data(), chs.data(), (int)crop_entries.size());
        if (bench) {
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_enc).count();
            fprintf(stderr, "[ocr_pipeline-bench] batch encode (%zu crops): %.1f ms\n", crop_entries.size(), ms);
        }

        double rec_total_ms = 0.0;
        if (enc_ok) {
            for (size_t i = 0; i < crop_entries.size(); i++) {
                fprintf(stderr, "ocr_pipeline: recognizing region %zu/%zu\n", i + 1, crop_entries.size());
                auto t_rec = std::chrono::steady_clock::now();
                int out_len = 0;
                const char * text = math_ocr_decode_batch_crop(ctx->rec, (int)i, &out_len);
                if (bench)
                    rec_total_ms +=
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_rec).count();

                if (text && out_len > 0) {
                    ocr_result r;
                    r.box = boxes[crop_entries[i].box_idx];
                    r.confidence = r.box.score;
                    r.text = std::string(text, out_len);
                    r.rec_confidence = math_ocr_mean_confidence(ctx->rec);
                    int n_conf = 0;
                    const float * conf = math_ocr_confidences(ctx->rec, &n_conf);
                    if (conf && n_conf > 0) r.char_conf.assign(conf, conf + n_conf);
                    results.push_back(std::move(r));
                }
            }
        } else {
            // Fallback: sequential single-crop path
            for (size_t i = 0; i < crop_entries.size(); i++) {
                fprintf(stderr, "ocr_pipeline: recognizing region %zu/%zu\n", i + 1, crop_entries.size());
                int out_len = 0;
                const char * text = math_ocr_recognize_raw(ctx->rec, crop_entries[i].data.data(), crop_entries[i].w,
                                                           crop_entries[i].h, 3, &out_len);
                if (text && out_len > 0) {
                    ocr_result r;
                    r.box = boxes[crop_entries[i].box_idx];
                    r.confidence = r.box.score;
                    r.text = std::string(text, out_len);
                    r.rec_confidence = math_ocr_mean_confidence(ctx->rec);
                    int n_conf = 0;
                    const float * conf = math_ocr_confidences(ctx->rec, &n_conf);
                    if (conf && n_conf > 0) r.char_conf.assign(conf, conf + n_conf);
                    results.push_back(std::move(r));
                }
            }
        }

        if (bench) {
            double total_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_total).count();
            fprintf(stderr, "[ocr_pipeline-bench] recognize (all boxes): %.1f ms\n", rec_total_ms);
            fprintf(stderr, "[ocr_pipeline-bench] total: %.1f ms\n", total_ms);
        }
    }

    fprintf(stderr, "ocr_pipeline: recognized %zu/%zu regions\n", results.size(), boxes.size());
    return results;
}

std::vector<ocr_result> run_file(context * ctx, const char * image_path, float prob_threshold, float box_threshold,
                                 int target_short_side) {
    if (!ctx || !image_path) return {};
    int img_w = 0, img_h = 0, img_c = 0;
    unsigned char * raw = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
    if (!raw) {
        fprintf(stderr, "ocr_pipeline: cannot load image for cropping: %s\n", image_path);
        return {};
    }
    auto results = run_raw(ctx, raw, img_w, img_h, 3, prob_threshold, box_threshold, target_short_side);
    stbi_image_free(raw);
    return results;
}

std::string recognize_file(context * ctx, const char * image_path) {
    if (!ctx || !ctx->rec || !image_path) return "";
    std::lock_guard<std::mutex> lock(ctx->infer_mutex);
    int out_len = 0;
    const char * text = math_ocr_recognize_file(ctx->rec, image_path, &out_len);
    return text ? std::string(text, out_len) : "";
}

void free(context * ctx) {
    if (!ctx) return;
    if (ctx->det) ocr_detect::free(ctx->det);
    if (ctx->rec) math_ocr_free(ctx->rec);
    delete ctx;
}

} // namespace ocr_pipeline
