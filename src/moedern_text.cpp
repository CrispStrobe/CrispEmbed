#include "moedern_text.h"
// The family engine is linked only where the build defines
// CRISPEMBED_MOEDERN_FAMILY_ENGINE (root library); historical 1.5M harnesses that
// compile this file alone keep linking and reject family packed models.
#ifdef CRISPEMBED_MOEDERN_FAMILY_ENGINE
#include "moedern_family_engine.h"
#endif
#include "moedern_text_engine_internal.h"
#include "moedern_text_frontend.h"
#include "core/env_gate.h"
#include <memory>
#include <mutex>
#include <stdexcept>
namespace moedern_text {
namespace detail = moedern_text_detail;
namespace {
void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
void gates() {
    require(core_env::on("CRISPEMBED_MOEDERN_TEXT_COMPOSED"), "MoEdern text composition gate is OFF");
    require(core_env::on("CRISPEMBED_MOEDERN_COLBERT"), "MoEdern native engine gate is OFF");
}
} // namespace
struct context {
    moedern_colbert::text_frontend tokenizer;
    std::unique_ptr<detail::context, decltype(&detail::free)> engine{ nullptr, detail::free };
    // 27M/115M packed models only, behind CRISPEMBED_MOEDERN_FAMILY; 1.5M keeps `engine`.
#ifdef CRISPEMBED_MOEDERN_FAMILY_ENGINE
    std::unique_ptr<moedern_family::context, decltype(&moedern_family::free)> family{ nullptr, moedern_family::free };
#endif
    std::string failure;
    bool packed = false;
    std::mutex mutex;
};
context * init(const char * model, const char * tokenizer, int threads, std::string & error) {
    error.clear();
    try {
        gates();
        require(threads >= 1 && threads <= 2, "bounded CPU one/two threads required");
        auto pending = std::make_unique<context>();
        const bool tokenizer_loaded = pending->tokenizer.load(tokenizer, error);
        require(tokenizer_loaded, error.c_str());
        pending->engine.reset(detail::init(model, threads, error));
        require(bool(pending->engine), error.c_str());
        return pending.release();
    } catch (const std::exception & e) {
        error = e.what();
        return nullptr;
    }
}
context * init_packed(const char * path, int threads, std::string & error) {
    error.clear();
    try {
        gates();
        require(core_env::on("CRISPEMBED_MOEDERN_PACKED"), "MoEdern packed gate is OFF");
        // CRISPEMBED_MOEDERN_FAST lifts the one/two-thread bound for the family fast path only.
        const bool fast = core_env::on("CRISPEMBED_MOEDERN_FAST");
        require(threads >= 1 && threads <= (fast ? 64 : 2), "bounded CPU one/two threads required");
        auto pending = std::make_unique<context>();
        const bool padded_q4k = core_env::on("CRISPEMBED_MOEDERN_Q4_K_PADDED");
        const bool tokenizer_loaded =
            padded_q4k ? pending->tokenizer.load_packed_q4k(path, error) : pending->tokenizer.load_packed(path, error);
        require(tokenizer_loaded, error.c_str());
        const std::string & revision = pending->tokenizer.family_revision();
        if (!revision.empty()) {
#ifndef CRISPEMBED_MOEDERN_FAMILY_ENGINE
            require(false, "MoEdern family engine not compiled in");
#else
            require(core_env::on("CRISPEMBED_MOEDERN_FAMILY") &&
                        ((padded_q4k && revision == "9d360b8b88ae11bea5784bf9012ff509fdd5c2da") ||
                         revision == "d2b7b1feaaac1bf87510ca9525208c3daeabcc5a" ||
                         revision == "66380d2ade3b5027bba83bdcdba052d0bb86dcdb"),
                    "MoEdern family gate is OFF or revision unqualified");
            require(threads <= 2 || core_env::on("CRISPEMBED_MOEDERN_FAST"), "bounded CPU one/two threads required");
            pending->family.reset(padded_q4k ? moedern_family::init_q4k(path, threads, error)
                                             : moedern_family::init(path, threads, error));
            require(bool(pending->family), error.c_str());
#endif
        } else {
            require(threads <= 2, "bounded CPU one/two threads required");
            pending->engine.reset(detail::init(path, threads, error));
            require(bool(pending->engine), error.c_str());
        }
        pending->packed = true;
        return pending.release();
    } catch (const std::exception & e) {
        error = e.what();
        return nullptr;
    }
}
void free(context * ctx) {
    delete ctx;
}
const char * error(const context * ctx) {
    return ctx ? ctx->failure.c_str() : "null MoEdern text context";
}
static bool encode_impl(context * ctx, const std::string & text, role task, result & output
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
                        ,
                        const moedern_text_test::observer * observer = nullptr
#endif
) {
    output = {};
    if (!ctx) return false;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->failure.clear();
    try {
        gates();
        require(!ctx->packed || core_env::on("CRISPEMBED_MOEDERN_PACKED"), "MoEdern packed gate is OFF");
        require(task == role::query || task == role::document, "explicit text query/document role required");
        moedern_colbert::text_tokens input;
        const bool tokenized = ctx->tokenizer.encode(
            text, task == role::query ? moedern_colbert::role::query : moedern_colbert::role::document, input,
            ctx->failure);
        require(tokenized, ctx->failure.c_str());
        detail::result native;
#ifdef CRISPEMBED_MOEDERN_FAMILY_ENGINE
        if (ctx->family) {
            require(core_env::on("CRISPEMBED_MOEDERN_FAMILY"), "MoEdern family gate is OFF");
            const bool family_computed =
                moedern_family::encode_tokens(ctx->family.get(), input.ids.data(), input.supplied_mask.data(),
                                              input.ids.size(), task == role::query, native.vectors);
            require(family_computed, moedern_family::error(ctx->family.get()));
            // The family engine consumes the supplied mask; its backbone attends to
            // every query position (expansion) and to the supplied mask otherwise.
            native.supplied_mask = input.supplied_mask;
            native.effective_mask =
                task == role::query ? std::vector<uint8_t>(input.ids.size(), 1) : input.supplied_mask;
        } else
#endif
        {
            const bool computed = detail::encode_tokens(ctx->engine.get(), input.ids.data(), input.supplied_mask.data(),
                                                        input.ids.size(), task, native
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
                                                        ,
                                                        observer ? *observer : moedern_text_test::observer{}
#endif
            );
            require(computed, detail::error(ctx->engine.get()));
        }
        require(native.supplied_mask == input.supplied_mask && native.effective_mask == input.effective_mask,
                "frontend/backbone attention contract differs");
        result pending;
        pending.task = task;
        pending.ids = std::move(input.ids);
        pending.supplied_mask = std::move(input.supplied_mask);
        pending.effective_mask = std::move(input.effective_mask);
        pending.scoring_mask = std::move(input.scoring_mask);
        pending.expansion_positions = std::move(input.expansion_positions);
        pending.all_vectors = std::move(native.vectors);
        require(pending.all_vectors.size() == pending.ids.size() * 128 &&
                    detail::select_rows(pending.all_vectors, pending.scoring_mask, pending.selected_vectors) &&
                    !pending.selected_vectors.empty(),
                "native scoring selection differs");
        output = std::move(pending);
        return true;
    } catch (const std::exception & e) {
        ctx->failure = e.what();
        return false;
    } catch (...) {
        ctx->failure = "unknown MoEdern text failure";
        return false;
    }
}
bool encode(context * ctx, const std::string & text, role task, result & output) {
    return encode_impl(ctx, text, task, output);
}
bool score(context * ctx, const std::string & query, const std::string & document, float & value) {
    value = 0;
    result q, d;
    if (!encode(ctx, query, role::query, q) || !encode(ctx, document, role::document, d)) return false;
    float pending = 0;
    if (!detail::maxsim(q.selected_vectors, d.selected_vectors, pending)) {
        ctx->failure = "invalid native raw-text MaxSim";
        return false;
    }
    value = pending;
    return true;
}
} // namespace moedern_text
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
namespace moedern_text_test {
bool encode_observed(moedern_text::context * ctx, const std::string & text, moedern_text::role task,
                     moedern_text::result & out, const observer & callback) {
    return moedern_text::encode_impl(ctx, text, task, out, &callback);
}
} // namespace moedern_text_test
#endif
