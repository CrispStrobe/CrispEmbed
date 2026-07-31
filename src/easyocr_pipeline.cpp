#include "easyocr_pipeline.h"

#include "easyocr_ocr.h"
#include "ocr_crop.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <mutex>

extern "C" {
typedef unsigned char stbi_uc;
stbi_uc * stbi_load(char const * filename, int * x, int * y, int * channels_in_file, int desired_channels);
void stbi_image_free(void * retval_from_stbi_load);
}

namespace easyocr_pipeline {

struct context {
    ocr_detect::context * detector = nullptr;
    easyocr_ocr_context * recognizer = nullptr;
    easyocr_layout::ordering_mode mode = easyocr_layout::ordering_mode::lines;
    std::mutex mutex;
};

bool load(context ** out, const char * detector_path, const char * recognizer_path, int n_threads) {
    if (!out || !detector_path || !recognizer_path) return false;
    *out = nullptr;
    auto * ctx = new context();
    if (!ocr_detect::load(&ctx->detector, detector_path, n_threads)) {
        delete ctx;
        return false;
    }
    ctx->recognizer = easyocr_ocr_init(recognizer_path, n_threads);
    if (!ctx->recognizer) {
        ocr_detect::free(ctx->detector);
        delete ctx;
        return false;
    }
    *out = ctx;
    return true;
}

void free(context * ctx) {
    if (!ctx) return;
    easyocr_ocr_free(ctx->recognizer);
    ocr_detect::free(ctx->detector);
    delete ctx;
}

void set_ordering_mode(context * ctx, easyocr_layout::ordering_mode mode) {
    if (!ctx) return;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->mode = mode;
}

easyocr_layout::ordering_mode ordering_mode(const context * ctx) {
    return ctx ? ctx->mode : easyocr_layout::ordering_mode::lines;
}

std::vector<result> run_raw(context * ctx, const uint8_t * pixels, int width, int height, int channels) {
    if (!ctx || !ctx->detector || !ctx->recognizer || !pixels || width <= 0 || height <= 0 || channels <= 0) return {};
    std::lock_guard<std::mutex> lock(ctx->mutex);

    const auto detected =
        ocr_detect::detect_rgb_ex(ctx->detector, pixels, width, height, channels, ocr_detect::rapid_defaults());
    std::vector<easyocr_layout::region> regions;
    regions.reserve(detected.size());
    for (const auto & box : detected) regions.push_back({ box.x, box.y, box.w, box.h, box.score });
    const auto ordered = easyocr_layout::order_regions(regions, ctx->mode);

    std::vector<result> results;
    results.reserve(ordered.size());
    for (size_t i = 0; i < ordered.size(); ++i) {
        const auto & region = ordered[i];
        const int x = std::max(0, (int)region.x - 2);
        const int y = std::max(0, (int)region.y - 2);
        const int crop_w = std::min(width - x, (int)region.w + 4);
        const int crop_h = std::min(height - y, (int)region.h + 4);
        int crop_width = 0, crop_height = 0;
        auto crop =
            ocr_crop::extract(pixels, width, height, channels, x, y, crop_w, crop_h, 0, &crop_width, &crop_height);
        if (crop.empty() || crop_width <= 0 || crop_height <= 0) continue;

        const int recognizer_width = ctx->mode == easyocr_layout::ordering_mode::words
                                         ? 200
                                         : std::max(200, (int)std::ceil(64.0 * crop_width / crop_height));
        if (!easyocr_ocr_set_width(ctx->recognizer, recognizer_width)) continue;
        int text_length = 0;
        const char * text =
            easyocr_ocr_recognize(ctx->recognizer, crop.data(), crop_width, crop_height, channels, &text_length);
        const float rec_confidence = easyocr_ocr_last_confidence(ctx->recognizer);
        result item;
        item.detector_confidence = region.score;
        item.word.text = text ? std::string(text, text_length) : std::string();
        item.word.x = region.x;
        item.word.y = region.y;
        item.word.w = region.w;
        item.word.h = region.h;
        item.word.confidence = rec_confidence;
        item.word.block = 0;
        item.word.line = region.line;
        item.word.index = (int)i;
        results.push_back(std::move(item));
    }
    const auto normalized = easyocr_layout::normalize_boxes(
        [&results]() {
            std::vector<easyocr_layout::word> words;
            words.reserve(results.size());
            for (const auto & item : results) words.push_back(item.word);
            return words;
        }(),
        width, height);
    for (size_t i = 0; i < results.size() && i < normalized.size(); ++i) results[i].normalized = normalized[i];
    return results;
}

std::vector<result> run_file(context * ctx, const char * image_path) {
    if (!ctx || !image_path) return {};
    int width = 0, height = 0, channels = 0;
    stbi_uc * pixels = stbi_load(image_path, &width, &height, &channels, 3);
    if (!pixels) return {};
    auto results = run_raw(ctx, pixels, width, height, 3);
    stbi_image_free(pixels);
    return results;
}

} // namespace easyocr_pipeline
