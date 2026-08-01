#include "ocr_pipeline_pool.h"

#include <condition_variable>
#include <mutex>

namespace ocr_pipeline_pool {

struct context {
    std::vector<ocr_pipeline::context *> slots;
    std::vector<bool> busy;
    std::mutex mutex;
    std::condition_variable available;
};

static int acquire(context * ctx) {
    std::unique_lock<std::mutex> lock(ctx->mutex);
    ctx->available.wait(lock, [&] {
        for (bool busy : ctx->busy)
            if (!busy) return true;
        return false;
    });
    for (int i = 0; i < (int)ctx->busy.size(); ++i) {
        if (!ctx->busy[i]) {
            ctx->busy[i] = true;
            return i;
        }
    }
    return -1;
}

static void release(context * ctx, int slot) {
    {
        std::lock_guard<std::mutex> lock(ctx->mutex);
        ctx->busy[slot] = false;
    }
    ctx->available.notify_one();
}

bool load(context ** out, const char * det_path, const char * rec_path, int pool_size, int n_threads) {
    if (!out || !det_path || !rec_path || pool_size <= 0) return false;
    *out = nullptr;
    auto * ctx = new context();
    ctx->slots.reserve(pool_size);
    for (int i = 0; i < pool_size; ++i) {
        ocr_pipeline::context * slot = nullptr;
        if (!ocr_pipeline::load(&slot, det_path, rec_path, n_threads)) {
            for (auto * loaded : ctx->slots) ocr_pipeline::free(loaded);
            delete ctx;
            return false;
        }
        ctx->slots.push_back(slot);
    }
    ctx->busy.assign(ctx->slots.size(), false);
    *out = ctx;
    return true;
}

std::vector<ocr_pipeline::ocr_result> run_file(context * ctx, const char * image_path, float prob_threshold,
                                               float box_threshold, int target_short_side) {
    if (!ctx || !image_path || ctx->slots.empty()) return {};
    const int slot = acquire(ctx);
    auto result =
        ocr_pipeline::run_file(ctx->slots[slot], image_path, prob_threshold, box_threshold, target_short_side);
    release(ctx, slot);
    return result;
}

std::vector<ocr_pipeline::ocr_result> run_raw(context * ctx, const uint8_t * pixels, int width, int height,
                                              int channels, float prob_threshold, float box_threshold,
                                              int target_short_side) {
    if (!ctx || !pixels || ctx->slots.empty()) return {};
    const int slot = acquire(ctx);
    auto result = ocr_pipeline::run_raw(ctx->slots[slot], pixels, width, height, channels, prob_threshold,
                                        box_threshold, target_short_side);
    release(ctx, slot);
    return result;
}

std::string recognize_file(context * ctx, const char * image_path) {
    if (!ctx || !image_path || ctx->slots.empty()) return {};
    const int slot = acquire(ctx);
    const std::string result = ocr_pipeline::recognize_file(ctx->slots[slot], image_path);
    release(ctx, slot);
    return result;
}

void free(context * ctx) {
    if (!ctx) return;
    for (auto * slot : ctx->slots) ocr_pipeline::free(slot);
    delete ctx;
}

} // namespace ocr_pipeline_pool
