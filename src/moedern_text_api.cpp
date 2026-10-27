#include "moedern_text_api.h"
#include "moedern_text.h"
#include "core/env_gate.h"
#include <memory>
#include <mutex>
#include <stdexcept>
struct crispembed_moedern_context {
    std::unique_ptr<moedern_text::context, decltype(&moedern_text::free)> model{ nullptr, moedern_text::free };
    moedern_text::result storage;
    crispembed_moedern_result view{};
    std::string error;
    std::mutex mutex;
    void clear() {
        view = {};
        storage = {};
        error.clear();
    }
};
namespace {
thread_local std::string initialization_error;
void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
void gate() {
    require(core_env::on("CRISPEMBED_MOEDERN_PUBLIC_TEXT"), "MoEdern public text gate is OFF");
}
std::string text(const char * value) {
    require(value, "MoEdern text pointer is null");
    size_t n = 0;
    while (n <= 65536 && value[n]) ++n;
    require(n <= 65536, "MoEdern text exceeds64KiB");
    return std::string(value, n);
}
} // namespace
extern "C" crispembed_moedern_context * crispembed_moedern_init(const char * path, int threads) {
    initialization_error.clear();
    try {
        gate();
        auto ctx = std::make_unique<crispembed_moedern_context>();
        ctx->model.reset(moedern_text::init_packed(path, threads == 0 ? 2 : threads, initialization_error));
        return ctx->model ? ctx.release() : nullptr;
    } catch (const std::exception & e) {
        initialization_error = e.what();
        return nullptr;
    }
}
extern "C" void crispembed_moedern_free(crispembed_moedern_context * ctx) {
    delete ctx;
}
extern "C" const crispembed_moedern_result * crispembed_moedern_encode(crispembed_moedern_context * ctx,
                                                                       const char * input, int role) {
    if (!ctx) {
        initialization_error = "MoEdern context is null";
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->clear();
    try {
        gate();
        require(role == CRISPEMBED_MOEDERN_QUERY || role == CRISPEMBED_MOEDERN_DOCUMENT,
                "explicit MoEdern role required");
        const auto value = text(input);
        const bool encoded =
            moedern_text::encode(ctx->model.get(), value, static_cast<moedern_text::role>(role), ctx->storage);
        require(encoded, moedern_text::error(ctx->model.get()));
        const auto & s = ctx->storage;
        ctx->view = { 1,
                      sizeof(crispembed_moedern_result),
                      role,
                      int32_t(s.ids.size()),
                      int32_t(s.selected_vectors.size() / 128),
                      128,
                      int32_t(s.expansion_positions.size()),
                      s.ids.data(),
                      s.supplied_mask.data(),
                      s.effective_mask.data(),
                      s.scoring_mask.data(),
                      s.expansion_positions.empty() ? nullptr : s.expansion_positions.data(),
                      s.all_vectors.data(),
                      s.selected_vectors.data() };
        return &ctx->view;
    } catch (const std::exception & e) {
        ctx->clear();
        ctx->error = e.what();
        return nullptr;
    }
}
extern "C" int crispembed_moedern_score(crispembed_moedern_context * ctx, const char * query, const char * document,
                                        float * value) {
    if (value) *value = 0;
    if (!ctx) {
        initialization_error = "MoEdern context is null";
        return 0;
    }
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->clear();
    try {
        gate();
        require(value, "MoEdern score output is null");
        const auto q = text(query), d = text(document);
        const bool computed = moedern_text::score(ctx->model.get(), q, d, *value);
        require(computed, moedern_text::error(ctx->model.get()));
        return 1;
    } catch (const std::exception & e) {
        if (value) *value = 0;
        ctx->error = e.what();
        return 0;
    }
}
extern "C" const char * crispembed_moedern_error(const crispembed_moedern_context * ctx) {
    return ctx ? ctx->error.c_str() : initialization_error.c_str();
}
