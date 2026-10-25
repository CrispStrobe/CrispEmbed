#include "core/clean_exit.h"
#include "qwen35_math.h"
#include "qwen35_weight_cache.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

using namespace qwen35_ocr;
static void env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
static bool exact(const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i].size() != b[i].size() || std::memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float)))
            return false;
    return true;
}
static int run() {
    const char * budget_name = "CRISPEMBED_QWEN35_WEIGHT_CACHE_BYTES";
    env(budget_name, nullptr);
#ifndef _WIN32
    if (converted_weight_cache::environment_budget() != converted_weight_cache::default_budget) return 1;
#endif
    for (const char * value : { "", "-1", "+1", " 1", "1x", "1073741825", "99999999999999999999999999" }) {
#ifdef _WIN32
        if (!*value) continue; // Windows empty assignment removes the variable.
#endif
        env(budget_name, value);
        if (converted_weight_cache::environment_budget()) return 1;
    }
    env(budget_name, "1073741824");
    if (converted_weight_cache::environment_budget() != converted_weight_cache::maximum_budget) return 1;
    env(budget_name, "0");
    if (converted_weight_cache().budget()) return 1;
    if (converted_weight_cache(converted_weight_cache::maximum_budget + 1).budget()) return 1;
    env("CRISPEMBED_QWEN35_OCR", "1");
    using owner = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    owner ctx(ggml_init({ 16 * 1024, nullptr, true }), ggml_free);
    if (!ctx) return 1;
    auto * half = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 32, 8);
    auto * full = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 9);
    auto * other = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 32, 8);
    auto * invalid = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 8);
    auto * quant = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, 32, 8);
    using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using buffer_owner = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    backend_owner backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!backend) return 1;
    buffer_owner buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!buffer) return 1;
    for (size_t i = 0; i < 256; ++i) {
        static_cast<ggml_fp16_t *>(half->data)[i] = ggml_fp32_to_fp16(float(int(i % 29) - 14) / 16);
        static_cast<ggml_fp16_t *>(other->data)[i] = ggml_fp32_to_fp16(float(int(i % 17) - 8) / 16);
        static_cast<float *>(invalid->data)[i] = 1;
    }
    static_cast<float *>(invalid->data)[255] = INFINITY;
    for (size_t i = 0; i < 288; ++i) static_cast<float *>(full->data)[i] = float(int(i % 19) - 9) / 8;
    std::vector<float> input(32 * 3);
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 13) - 6) / 16;
    converted_weight_cache cache(1024), independent(1024), zero(0);
    std::vector<std::vector<float>> old, cold, warm, disabled, foreign, expected_foreign;
    env("CRISPEMBED_QWEN35_WEIGHT_CACHE", "0");
    if (!project_cpu({ half, full }, input.data(), 32, 3, 2, old, &cache) || cache.entries() || cache.bytes()) return 1;
    env("CRISPEMBED_QWEN35_WEIGHT_CACHE", "1");
    env("CRISPEMBED_QWEN35_QUANT", "1");
    if (cache.get(quant, 2)) return 1; // This experiment deliberately retains quantized fallback.
    env("CRISPEMBED_QWEN35_QUANT", "0");
    if (zero.get(half, 2) || cache.get(nullptr, 2) || cache.get(invalid, 2) || cache.entries()) return 1;
    {
        converted_weight_cache both(4096);
        if (!project_cpu({ half, full }, input.data(), 32, 3, 2, cold, &both) || !exact(old, cold) ||
            !project_cpu({ half, full }, input.data(), 32, 3, 2, warm, &both) || !exact(old, warm) ||
            both.entries() != 2 || both.hits() != 2 || both.bytes() > both.budget())
            return 1;
    }
    for (const char * parallel : { "0", "1" }) {
        env("CRISPEMBED_QWEN35_PARALLEL_F16", parallel);
        if (!project_cpu({ half, full }, input.data(), 32, 3, 2, cold, &cache) || !exact(old, cold) ||
            cache.entries() != 1 || cache.bytes() != 1024 || cache.get(full, 2))
            return 1;
        const auto hits = cache.hits();
        if (!project_cpu({ half, full }, input.data(), 32, 3, 2, warm, &cache) || !exact(old, warm) ||
            cache.hits() != hits + 1 || cache.bytes() > cache.budget())
            return 1;
    }
    env("CRISPEMBED_QWEN35_WEIGHT_CACHE", "0");
    const auto hits = cache.hits();
    if (!project_cpu({ half, full }, input.data(), 32, 3, 2, disabled, &cache) || !exact(old, disabled) ||
        cache.hits() != hits)
        return 1;
    env("CRISPEMBED_QWEN35_WEIGHT_CACHE", "");
    if (cache.get(half, 2)) return 1;
    env("CRISPEMBED_QWEN35_WEIGHT_CACHE", "1");
    if (!project_cpu({ other }, input.data(), 32, 3, 1, expected_foreign) ||
        !project_cpu({ other }, input.data(), 32, 3, 1, foreign, &independent) || !exact(expected_foreign, foreign) ||
        exact({ old[0] }, foreign))
        return 1;
    cache.clear();
    if (cache.bytes() || cache.entries() || cache.hits()) return 1;
    if (!project_cpu({ other }, input.data(), 32, 3, 1, foreign, &independent) || !exact(expected_foreign, foreign) ||
        independent.hits() != 1)
        return 1;
    if (!project_cpu({ half }, input.data(), 32, 3, 2, cold, &cache) || !exact({ old[0] }, cold) ||
        cache.entries() != 1)
        return 1;
    cache.clear();
    independent.clear();
    std::puts("WEIGHT_CACHE_CONTROLS pass=1 literal0=1 budget=1 invalid_budget=1 cold_warm_exact=1 fallback=1 "
              "nonfinite=1 reset=1 independent=1 survivor=1");
    return 0;
}
int main() {
    core_util::clean_exit(run());
}
