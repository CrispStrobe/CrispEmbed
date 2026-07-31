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
#include "ppocrv6_det.h"
#include "ppocrv6_ocr.h"
#include "core/gguf_loader.h"

// stb_image declarations (implementation lives in image_preprocess.cpp)
extern "C" {
typedef unsigned char stbi_uc;
stbi_uc * stbi_load(char const * filename, int * x, int * y, int * channels_in_file, int desired_channels);
void stbi_image_free(void * retval_from_stbi_load);
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
    ocr_detect::context * det = nullptr;
    math_ocr_context * rec = nullptr;
    int n_threads = 1;
    bool bench = false;
    ppocrv6_det::context * ppdet = nullptr;
    ppocrv6_ocr_context * pprec = nullptr;
    bool ppocrv6 = false;
};

static bool is_ppocrv6(const char * path) {
    gguf_context * g = core_gguf::open_metadata(path);
    if (!g) return false;
    bool yes = core_gguf::kv_str(g, "general.architecture", "") == "ppocrv6";
    core_gguf::free_metadata(g);
    return yes;
}

bool load(context ** out, const char * det_path, const char * rec_path, int n_threads) {
    auto * ctx = new context();
    *out = ctx;
    ctx->n_threads = n_threads;
    ctx->bench = (std::getenv("CRISPEMBED_OCR_PIPELINE_BENCH") != nullptr);

    if (is_ppocrv6(det_path) && is_ppocrv6(rec_path)) {
        ctx->ppdet = ppocrv6_det::init(det_path, n_threads);
        ctx->pprec = ppocrv6_ocr_init(rec_path, n_threads);
        if (!ctx->ppdet || !ctx->pprec) {
            if (ctx->ppdet) ppocrv6_det::free(ctx->ppdet);
            if (ctx->pprec) ppocrv6_ocr_free(ctx->pprec);
            delete ctx;
            *out = nullptr;
            return false;
        }
        ctx->ppocrv6 = true;
        return true;
    }

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

std::vector<ocr_result> run_file(context * ctx, const char * image_path, float prob_threshold, float box_threshold,
                                 int target_short_side) {
    if (!ctx || !image_path) return {};
    if (ctx->ppocrv6) {
        auto boxes = ppocrv6_det::detect_file(ctx->ppdet, image_path, std::min(prob_threshold, 0.2f));
        int iw, ih, ic;
        unsigned char * img = stbi_load(image_path, &iw, &ih, &ic, 3);
        if (!img) return {};
        std::vector<ocr_result> results;
        for (const auto & b : boxes) {
            int x = std::max(0, (int)b.x - 2), y = std::max(0, (int)b.y - 2);
            int cw = std::min(iw - x, (int)b.w + 4), ch = std::min(ih - y, (int)b.h + 4);
            auto crop = crop_image(img, iw, ih, x, y, cw, ch);
            if (crop.empty()) continue;
            int out_len = 0;
            const char * text = ppocrv6_ocr_recognize_raw(ctx->pprec, crop.data(), cw, ch, 3, &out_len);
            if (!text || out_len <= 0) continue;
            ocr_result r;
            r.box = { b.x, b.y, b.w, b.h, b.score, 0.0f, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
            r.confidence = b.score;
            r.rec_confidence = 0.0f;
            r.text.assign(text, out_len);
            results.push_back(std::move(r));
        }
        stbi_image_free(img);
        std::sort(results.begin(), results.end(), [](const ocr_result & a, const ocr_result & b) {
            return a.box.y == b.box.y ? a.box.x < b.box.x : a.box.y < b.box.y;
        });
        return results;
    }
    if (!ctx->det || !ctx->rec) return {};

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Step 1: Detect text regions
    auto t_detect = std::chrono::steady_clock::now();
    auto boxes = ocr_detect::detect_file(ctx->det, image_path, prob_threshold, box_threshold, 1.5f, target_short_side);
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_detect).count();
        fprintf(stderr, "[ocr_pipeline-bench] detect: %.1f ms (%zu boxes)\n", ms, boxes.size());
    }

    if (boxes.empty()) {
        fprintf(stderr, "ocr_pipeline: no text detected in %s\n", image_path);
        return {};
    }
    fprintf(stderr, "ocr_pipeline: detected %zu text regions\n", boxes.size());

    // Step 2: Load original image for cropping
    int img_w, img_h, img_c;
    unsigned char * img = stbi_load(image_path, &img_w, &img_h, &img_c, 3);
    if (!img) {
        fprintf(stderr, "ocr_pipeline: cannot load image for cropping: %s\n", image_path);
        return {};
    }

    // Step 3: Collect valid crops, batch-encode, then decode sequentially.
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

        auto crop = crop_image(img, img_w, img_h, cx, cy, cw, ch);
        if (crop.empty()) continue;

        int actual_w = std::min(cw, img_w - cx);
        int actual_h = std::min(ch, img_h - cy);
        if (actual_w <= 0 || actual_h <= 0) continue;

        crop_entries.push_back({ std::move(crop), actual_w, actual_h, i });
    }

    stbi_image_free(img);

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

std::string recognize_file(context * ctx, const char * image_path) {
    if (!ctx || !ctx->rec || !image_path) return "";
    int out_len = 0;
    const char * text = math_ocr_recognize_file(ctx->rec, image_path, &out_len);
    return text ? std::string(text, out_len) : "";
}

void free(context * ctx) {
    if (!ctx) return;
    if (ctx->ppocrv6) {
        ppocrv6_det::free(ctx->ppdet);
        ppocrv6_ocr_free(ctx->pprec);
        delete ctx;
        return;
    }
    if (ctx->det) ocr_detect::free(ctx->det);
    if (ctx->rec) math_ocr_free(ctx->rec);
    delete ctx;
}

} // namespace ocr_pipeline
