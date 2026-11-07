#include "lighton4b_native_context.h"
#include "lighton4b_physical_native_ops.h"
#include "qwen35_math.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <string>
using namespace qwen35_ocr;
static void gate(const char * n, const char * v) {
#ifdef _WIN32
    _putenv_s(n, v ? v : "");
#else
    if (v)
        setenv(n, v, 1);
    else
        unsetenv(n);
#endif
}
static int run() {
    int checks = 0;
    auto check = [&](bool ok) {
        ++checks;
        if (!ok) {
            std::fprintf(stderr, "physical native-ops check=%d failed\n", checks);
            std::exit(1);
        }
    };
    for (const char * n : { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_QWEN35_QUANT", "CRISPEMBED_QWEN35_STREAM_PROJECTION",
                            "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", "CRISPEMBED_LIGHTON4B_QUANT",
                            "CRISPEMBED_LIGHTON4B_Q6_K", "CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL" })
        gate(n, "1");
    for (const char * n : { "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                            "CRISPEMBED_LIGHTON4B_HEAD_BF16" })
        gate(n, "0");
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 1024 * 1024, nullptr, true }), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    check(bool(ctx) && bool(backend));
    constexpr int width = 256, rows = 257, tokens = 4;
    auto * q4 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, width, rows);
    auto * q6 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q6_K, width, rows);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buf(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    check(bool(buf));
    std::vector<float> weights(width * rows), input(width * tokens), gamma(width);
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 71) - 35) / 32;
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 19) - 9) / 13;
    for (int i = 0; i < width; ++i) gamma[i] = float(i % 7 + 1) / 8;
    auto bits = [](const std::vector<std::vector<float>> & a, const std::vector<std::vector<float>> & b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i].size() != b[i].size() || std::memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float)))
                return false;
        return true;
    };
    std::vector<std::vector<float>> baseline, got;
    std::vector<float> flat, rms;
    for (auto * m : { q4, q6 }) {
        check(ggml_quantize_chunk(m->type, weights.data(), m->data, 0, rows, width, nullptr) == ggml_nbytes(m));
        gate("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS", "0");
        check(project_cpu({ m }, input.data(), width, tokens, 2, baseline, nullptr, true));
        for (const char * value : { static_cast<const char *>(nullptr), "", "0", "false", "off", "no", "1" }) {
            gate("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS", value);
            // Gate alone cannot change non-physical owners, including explicit ON.
            check(project_cpu({ m }, input.data(), width, tokens, 2, got, nullptr, false) && bits(got, baseline));
            check(project_cpu({ m }, input.data(), width, tokens, 2, got, nullptr, true));
            if (!value || std::string(value) != "1")
                check(bits(got, baseline));
            else {
                std::vector<std::vector<float>> direct;
                check(lighton4b_physical_matmul({ m }, input.data(), width, tokens, 2, direct) && bits(got, direct));
                check(project_rows_cpu(m, input.data(), width, tokens, 2, 31, flat, true) && flat == got[0]);
            }
        }
    }
    // Real bounded physical full-attention cache: both same-mode continuations
    // must succeed, while either arithmetic-mode toggle rejects atomically.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> cache_ctx(ggml_init({ 1024 * 1024, nullptr, true }), ggml_free);
    check(bool(cache_ctx));
    attention_weights aw{};
    aw.upstream_physical = true;
    aw.q = ggml_new_tensor_2d(cache_ctx.get(), GGML_TYPE_Q4_K, 2560, 8192);
    aw.k = ggml_new_tensor_2d(cache_ctx.get(), GGML_TYPE_Q4_K, 2560, 1024);
    aw.v = ggml_new_tensor_2d(cache_ctx.get(), GGML_TYPE_Q4_K, 2560, 1024);
    aw.out = ggml_new_tensor_2d(cache_ctx.get(), GGML_TYPE_Q4_K, 4096, 2560);
    aw.q_norm = ggml_new_tensor_1d(cache_ctx.get(), GGML_TYPE_F32, 256);
    aw.k_norm = ggml_new_tensor_1d(cache_ctx.get(), GGML_TYPE_F32, 256);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> cache_buf(
        ggml_backend_alloc_ctx_tensors(cache_ctx.get(), backend.get()), ggml_backend_buffer_free);
    check(bool(cache_buf));
    for (auto * tensor : { aw.q, aw.k, aw.v, aw.out }) std::memset(tensor->data, 0, ggml_nbytes(tensor));
    for (auto * tensor : { aw.q_norm, aw.k_norm }) std::fill_n(static_cast<float *>(tensor->data), 256, 1.0f);
    attention_config config;
    config.query_heads = 16;
    config.kv_heads = 4;
    std::vector<float> cache_input(2560, 0.125f);
    float positions[3] = { 0, 0, 0 };
    for (const char * mode : { "0", "1" }) {
        gate("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS", mode);
        attention_outputs first, next;
        check(attention_cpu(aw, config, cache_input.data(), positions, 1, 1, 2, nullptr, first));
        check(attention_cpu(aw, config, cache_input.data(), positions, 1, 1, 2, &first.cache, next));
        check(next.cache.tokens == 2 && next.cache.physical_native_ops == lighton4b_physical_native_ops(true));
        gate("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS", mode[0] == '0' ? "1" : "0");
        next.attention = { 42 };
        check(!attention_cpu(aw, config, cache_input.data(), positions, 1, 1, 2, &first.cache, next) &&
              next.attention.empty());
    }
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS", "1");
    check(lighton4b_physical_rms(input.data(), gamma.data(), width, tokens, 2, rms));
    check(rms.size() == input.size());
    // Failure must clear stale caller outputs, including nonfinite input.
    auto bad = input;
    bad[0] = std::numeric_limits<float>::quiet_NaN();
    got = { { 42 } };
    check(!lighton4b_physical_matmul({ q4 }, bad.data(), width, tokens, 2, got) && got.empty());
    rms = { 42 };
    check(!lighton4b_physical_rms(bad.data(), gamma.data(), width, tokens, 2, rms) && rms.empty());
    for (const char * conflict : { "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                                   "CRISPEMBED_LIGHTON4B_HEAD_BF16" }) {
        gate(conflict, "1");
        check(!lighton4b_physical_native_ops(true));
        gate(conflict, "0");
    }
    check(!lighton4b_physical_native_ops(false));
    std::printf("PHYSICAL_NATIVE_OPS pass=1 checks=%d gate_off_legacy_exact=1 nonphysical_unchanged=1 "
                "streamed_head_exact=1 physical_cache_toggle=1 learned_model_forward=0\n",
                checks);
    return 0;
}
int main() {
    core_util::clean_exit(run());
}
