#include "easyocr_pipeline.h"

#include "easyocr_ocr.h"
#include "easyocr_postprocess.h"
#include "ocr_crop.h"
#include "ocr_detect.h"
#include "core/env_gate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

static std::vector<result> recognize_regions_locked(context * ctx, const std::vector<easyocr_layout::region> & regions,
                                                    const uint8_t * pixels, int width, int height, int channels,
                                                    bool add_detector_crop_margin) {
    const auto ordered = ctx->mode == easyocr_layout::ordering_mode::lines ? easyocr_layout::group_dbnet_lines(regions)
                                                                           : easyocr_layout::order_words(regions);

    // Opt-in: EASYOCR_WIDTH_SORT=1.
    //
    // easyocr_ocr_set_width() tears down and rebuilds the recognizer graph
    // whenever the canvas width changes, and width is derived per crop
    // (bucketed to a multiple of 64), so reading order rebuilds every time
    // consecutive lines land in different buckets -- O(regions) rebuilds on a
    // page with varied line widths. Visiting in width-sorted order makes that
    // O(distinct widths): each bucket is built once and every region sharing it
    // runs back to back. Recognition is independent per crop, so this reorders
    // work only; results are written back into reading-order slots below.
    //
    // MEASURED 2026-08-02: worth 0-3%, at the edge of noise (3.00 vs 3.02 CPU-s
    // on a 47-region receipt, 7.43 vs 7.68 on a 31-unit document), because the
    // rebuild is graph construction plus a gallocr pass with no weight reload.
    // Output is byte-identical. Kept off by default and gated rather than
    // deleted: it becomes worth more if the recognizer graph ever grows an
    // expensive build step (weight residency, kernel specialisation, a
    // shape-keyed resident cache), and it is the natural companion to batching
    // crops of equal width into one graph dispatch.
    std::vector<size_t> visit(ordered.size());
    for (size_t i = 0; i < visit.size(); ++i) visit[i] = i;
    if (std::getenv("EASYOCR_WIDTH_SORT") != nullptr) {
        // The sort key must use the SAME crop geometry the loop below will use,
        // or it orders by widths that are never requested and the grouping
        // under-delivers. This previously hardcoded the 2-pixel detector margin
        // while the loop applies it only when add_detector_crop_margin is set,
        // so on the external-geometry path (Python EasyOCR / Tesseract /
        // LayoutLM boxes, pad 0) the ordering was computed from the wrong
        // widths. Measured on commons_test_ocr_document.jpg via
        // CRISPEMBED_EASYOCR_STAGE_BENCH=1: 27 regions over 14 distinct canvas
        // widths, and the mismatched key left 19 graph rebuilds instead of 14.
        const int pad = add_detector_crop_margin ? 2 : 0;
        std::vector<int> canvas(ordered.size(), 0);
        for (size_t i = 0; i < ordered.size(); ++i) {
            const int x = std::max(0, (int)ordered[i].x - pad);
            const int y = std::max(0, (int)ordered[i].y - pad);
            const int cw = std::min(width - x, (int)ordered[i].w + 2 * pad);
            const int ch = std::min(height - y, (int)ordered[i].h + 2 * pad);
            canvas[i] = (cw > 0 && ch > 0) ? easyocr_postprocess::recognizer_canvas_width(cw, ch) : 0;
        }
        std::stable_sort(visit.begin(), visit.end(), [&canvas](size_t a, size_t b) { return canvas[a] < canvas[b]; });
    }

    // Per-stage recognition bench (CRISPEMBED_EASYOCR_STAGE_BENCH=1).
    //
    // PLAN.md H3: `[easyocr-stage-bench]` only reports load vs
    // detect+recognize, so nothing below the 12.4 s compute half of a real page
    // was ever attributed. This splits the per-region loop into crop extraction,
    // recognizer width changes (which tear down and rebuild the graph) and the
    // recognize call itself, and counts how many width rebuilds the page
    // actually triggered -- the number EASYOCR_WIDTH_SORT exists to reduce.
    const bool stage_bench = core_env::on("CRISPEMBED_EASYOCR_STAGE_BENCH");
    double crop_ms = 0.0, width_ms = 0.0, recognize_ms = 0.0;
    long width_calls = 0, width_changes = 0;
    int last_width = -1;
    const auto stage_clock = []() { return std::chrono::steady_clock::now(); };
    const auto stage_started = stage_clock();

    std::vector<result> slots(ordered.size());
    std::vector<char> filled(ordered.size(), 0);
    for (size_t vi = 0; vi < visit.size(); ++vi) {
        const size_t i = visit[vi];
        const auto & region = ordered[i];
        // EasyOCR crops caller-supplied horizontal boxes exactly. The native
        // DBNet route retains its historical two-pixel diagnostic margin, but
        // external geometry (Python EasyOCR/Tesseract/LayoutLM) must not be
        // enlarged before recognizer comparison.
        const int pad = add_detector_crop_margin ? 2 : 0;
        const int x = std::max(0, (int)region.x - pad);
        const int y = std::max(0, (int)region.y - pad);
        const int crop_w = std::min(width - x, (int)region.w + 2 * pad);
        const int crop_h = std::min(height - y, (int)region.h + 2 * pad);
        int crop_width = 0, crop_height = 0;
        const auto crop_t0 = stage_bench ? stage_clock() : std::chrono::steady_clock::time_point{};
        auto crop =
            ocr_crop::extract(pixels, width, height, channels, x, y, crop_w, crop_h, 0, &crop_width, &crop_height);
        if (stage_bench) crop_ms += std::chrono::duration<double, std::milli>(stage_clock() - crop_t0).count();
        if (crop.empty() || crop_width <= 0 || crop_height <= 0) continue;

        const int recognizer_width = easyocr_postprocess::recognizer_canvas_width(crop_width, crop_height);
        const auto width_t0 = stage_bench ? stage_clock() : std::chrono::steady_clock::time_point{};
        const bool width_ok = easyocr_ocr_set_width(ctx->recognizer, recognizer_width);
        if (stage_bench) {
            width_ms += std::chrono::duration<double, std::milli>(stage_clock() - width_t0).count();
            ++width_calls;
            if (recognizer_width != last_width) ++width_changes;
            last_width = recognizer_width;
        }
        if (!width_ok) continue;
        int text_length = 0;
        const auto rec_t0 = stage_bench ? stage_clock() : std::chrono::steady_clock::time_point{};
        const char * text =
            easyocr_ocr_recognize(ctx->recognizer, crop.data(), crop_width, crop_height, channels, &text_length);
        if (stage_bench) recognize_ms += std::chrono::duration<double, std::milli>(stage_clock() - rec_t0).count();
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
        item.crop_x = x;
        item.crop_y = y;
        item.crop_w = crop_width;
        item.crop_h = crop_height;
        slots[i] = std::move(item);
        filled[i] = 1;
    }
    if (stage_bench) {
        const double loop_ms = std::chrono::duration<double, std::milli>(stage_clock() - stage_started).count();
        fprintf(stderr,
                "[easyocr-recognize-bench] regions=%zu crop_ms=%.1f set_width_ms=%.1f recognize_ms=%.1f "
                "loop_ms=%.1f width_calls=%ld width_changes=%ld width_sort=%d\n",
                ordered.size(), crop_ms, width_ms, recognize_ms, loop_ms, width_calls, width_changes,
                std::getenv("EASYOCR_WIDTH_SORT") != nullptr ? 1 : 0);
    }

    // Back to reading order, dropping regions that produced no crop.
    std::vector<result> results;
    results.reserve(ordered.size());
    for (size_t i = 0; i < slots.size(); ++i)
        if (filled[i]) results.push_back(std::move(slots[i]));
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

std::vector<result> run_regions(context * ctx, const std::vector<easyocr_layout::region> & regions,
                                const uint8_t * pixels, int width, int height, int channels) {
    if (!ctx || !ctx->recognizer || !pixels || width <= 0 || height <= 0 || channels <= 0) return {};
    std::lock_guard<std::mutex> lock(ctx->mutex);
    return recognize_regions_locked(ctx, regions, pixels, width, height, channels, false);
}

std::vector<result> run_raw(context * ctx, const uint8_t * pixels, int width, int height, int channels) {
    if (!ctx || !ctx->detector || !ctx->recognizer || !pixels || width <= 0 || height <= 0 || channels <= 0) return {};
    std::lock_guard<std::mutex> lock(ctx->mutex);
    const bool stage_bench = core_env::on("CRISPEMBED_EASYOCR_STAGE_BENCH");
    const auto detect_t0 = std::chrono::steady_clock::now();
    const auto detected =
        ocr_detect::detect_rgb_ex(ctx->detector, pixels, width, height, channels, ocr_detect::rapid_defaults());
    if (stage_bench)
        fprintf(stderr, "[easyocr-recognize-bench] detect_ms=%.1f boxes=%zu\n",
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - detect_t0).count(),
                detected.size());
    std::vector<easyocr_layout::region> regions;
    regions.reserve(detected.size());
    for (const auto & box : detected) regions.push_back({ box.x, box.y, box.w, box.h, box.score });
    return recognize_regions_locked(ctx, regions, pixels, width, height, channels, true);
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
