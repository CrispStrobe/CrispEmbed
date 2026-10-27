#include "crispembed.h"
#include "josev2b_composed.h"
#include <memory>
#include <string>
#include <exception>
struct crispembed_systemone_context {
    std::unique_ptr<josev2b::composed_context> model;
    std::string result, error;
};
namespace {
thread_local std::string initialization_error;
}
extern "C" crispembed_systemone_context * crispembed_systemone_init(const char * path, int threads) {
    initialization_error.clear();
    try {
        auto ctx = std::unique_ptr<crispembed_systemone_context>(new crispembed_systemone_context);
        ctx->model = josev2b::composed_context::load_packed(path, threads == 0 ? 2 : threads, initialization_error);
        return ctx->model ? ctx.release() : nullptr;
    } catch (const std::exception & e) {
        initialization_error = e.what();
        return nullptr;
    }
}
extern "C" void crispembed_systemone_free(crispembed_systemone_context * ctx) {
    delete ctx;
}
extern "C" const char * crispembed_systemone_run_json(crispembed_systemone_context * ctx, const char * request) {
    if (!ctx) {
        initialization_error = "SystemOne context is null";
        return nullptr;
    }
    ctx->result.clear();
    ctx->error.clear();
    if (!request) {
        ctx->error = "SystemOne JSON request is null";
        return nullptr;
    }
    try {
        size_t n = 0;
        while (n <= 65536 && request[n]) ++n;
        if (n > 65536) {
            ctx->error = "SystemOne request exceeds64KiB";
            return nullptr;
        }
        if (!ctx->model || !ctx->model->decide(std::string(request, n), ctx->result, ctx->error)) return nullptr;
        return ctx->result.c_str();
    } catch (const std::exception & e) {
        ctx->result.clear();
        ctx->error = e.what();
        return nullptr;
    }
}
extern "C" const char * crispembed_systemone_error(const crispembed_systemone_context * ctx) {
    return ctx ? ctx->error.c_str() : initialization_error.c_str();
}
extern "C" const char * crispembed_systemone_profile(const crispembed_systemone_context * ctx) {
    return ctx && ctx->model ? "official-pytorch-text-batch1-512-no-extra-temperature" : "";
}
