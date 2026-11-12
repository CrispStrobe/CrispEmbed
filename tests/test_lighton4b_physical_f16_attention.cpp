#include "lighton4b_physical_f16_attention.h"
#include "lighton4b_native_context.h"
#include "ggml.h"
#include "core/clean_exit.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
    auto bits = [](const std::vector<float> & a, const std::vector<float> & b) {
        return a.size() == b.size() && !std::memcmp(a.data(), b.data(), a.size() * sizeof(float));
    };
    int checks = 0;
    auto check = [&](bool ok) {
        ++checks;
        if (!ok) {
            std::fprintf(stderr, "F16 attention check %d failed\n", checks);
            std::exit(1);
        }
    };
    for (const char * n : { "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_QWEN35_QUANT", "CRISPEMBED_LIGHTON4B_QUANT",
                            "CRISPEMBED_LIGHTON4B_Q6_K", "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT",
                            "CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL" })
        gate(n, "1");
    for (const char * n : { "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                            "CRISPEMBED_LIGHTON4B_HEAD_BF16", "CRISPEMBED_LIGHTON4B_PHYSICAL_NATIVE_OPS" })
        gate(n, "0");
    for (const char * v : { static_cast<const char *>(nullptr), "", "0", "false", "OFF", "no" }) {
        gate("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", v);
        check(!lighton4b_physical_f16_attention_mode(true));
        check(lighton4b_physical_f16_attention_valid(true));
    }
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", "1");
    check(!lighton4b_physical_f16_attention_mode(false));
    check(lighton4b_physical_f16_attention_valid(false));
    if (!lighton4b_physical_f16_attention_available()) {
        check(!lighton4b_physical_f16_attention_valid(true));
        std::printf("PHYSICAL_F16_ATTENTION_UNSUPPORTED_GUARD PASS %d\n", checks);
        return 0;
    }
    check(lighton4b_physical_f16_attention_valid(true));
    for (const char * bad : { "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                              "CRISPEMBED_LIGHTON4B_HEAD_BF16" }) {
        gate(bad, "1");
        check(!lighton4b_physical_f16_attention_valid(true));
        gate(bad, "0");
    }
    for (auto spec : { std::pair<int, int>{ 0, 1 },
                       { 0, 4 },
                       { 0, 16 },
                       { 1, 1 },
                       { 15, 1 },
                       { 255, 1 },
                       { 255, 2 },
                       { 256, 16 },
                       { 511, 1 },
                       { 2032, 16 } }) {
        const int past = spec.first, n = spec.second, total = past + n;
        attention_cache cache;
        cache.batch = 1;
        cache.heads = 4;
        cache.width = 256;
        cache.tokens = total;
        cache.upstream_physical = true;
        cache.physical_f16_attention = true;
        cache.keys.resize(size_t(total) * 1024);
        cache.values.resize(cache.keys.size());
        for (int h = 0; h < 4; ++h)
            for (int t = 0; t < total; ++t)
                for (int i = 0; i < 256; ++i) {
                    const size_t ix = (size_t(h) * total + t) * 256 + i;
                    cache.keys[ix] = t < past ? .25f : .1234567f;
                    cache.values[ix] = float(h + 1) + (t < past ? 0.f : .0001f);
                }
        const auto original = cache;
        std::vector<float> q(size_t(n) * 4096, 0), fused(size_t(n) * 8192, 0), pooled;
        attention_cache canonical;
        check(lighton4b_physical_f16_attention_cpu(q, fused, cache, past, n, 4, pooled, canonical));
        check(pooled.size() == size_t(n) * 4096 && canonical.tokens == total && canonical.physical_f16_attention);
        check(cache.keys == original.keys && cache.values == original.values);
        for (int t = 0; t < n; ++t) {
            const int visible = past + t + 1;
            const float p = ggml_fp16_to_fp32(ggml_fp32_to_fp16(1.f / visible));
            for (int h = 0; h < 16; ++h)
                for (int i = 0; i < 256; ++i)
                    check(pooled[(size_t(t) * 16 + h) * 256 + i] == (visible * p * float(h / 4 + 1)) * .5f);
        }
        for (int h = 0; h < 4; ++h)
            for (int t = 0; t < total; ++t) check(canonical.values[(size_t(h) * total + t) * 256] == float(h + 1));
        auto bad = cache;
        bad.keys[0] = 65536;
        check(!lighton4b_physical_f16_attention_cpu(q, fused, bad, past, n, 4, pooled, canonical) && pooled.empty() &&
              canonical.keys.empty());
        check(!lighton4b_physical_f16_attention_cpu(q, fused, cache, past, n, 2, pooled, canonical));
        if (past) {
            bad = cache;
            bad.values[0] = 1.0001f;
            check(!lighton4b_physical_f16_attention_cpu(q, fused, bad, past, n, 4, pooled, canonical));
        }
    }
    // Exercise actual attention cache fingerprint rejection before projection,
    // and off/nonphysical byte identity with small all-zero quantized weights.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 1 << 20, nullptr, true }), ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
    attention_weights w{};
    w.upstream_physical = true;
    w.q = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, 2560, 8192);
    w.k = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, 2560, 1024);
    w.v = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, 2560, 1024);
    w.out = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_K, 4096, 2560);
    w.q_norm = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 256);
    w.k_norm = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 256);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buf(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    check(bool(buf));
    for (auto * t : { w.q, w.k, w.v, w.out }) std::memset(t->data, 0, ggml_nbytes(t));
    for (auto * t : { w.q_norm, w.k_norm }) std::fill_n(static_cast<float *>(t->data), 256, 1.f);
    attention_config cfg;
    cfg.query_heads = 16;
    cfg.kv_heads = 4;
    std::vector<float> input(2560, .125f);
    float positions[3] = { 0, 0, 0 };
    attention_outputs off, on, next;
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", "0");
    check(attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, nullptr, off));
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", "1");
    check(!attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, &off.cache, next) && next.pooled.empty());
    check(attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, nullptr, on));
    check(on.cache.physical_f16_attention);
    check(attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, &on.cache, next));
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", "off");
    check(!attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, &on.cache, next));
    for (const char * value : { static_cast<const char *>(nullptr), "", "0", "false", "off", "no" }) {
        gate("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", value);
        check(attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, nullptr, next));
        check(bits(next.pooled, off.pooled) && bits(next.cache.keys, off.cache.keys) &&
              !next.cache.physical_f16_attention);
    }
    // RoPE mode is part of the real KV identity in both toggle directions.
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE", "0");
    check(attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, nullptr, off));
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE", "1");
    check(!attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, &off.cache, next));
    check(attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, nullptr, on) && on.cache.physical_rope);
    check(attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, &on.cache, next) && next.cache.physical_rope);
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_ROPE", "0");
    check(!attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, &on.cache, next));
    check(attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, &off.cache, next) && !next.cache.physical_rope);
    w.upstream_physical = false;
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", "0");
    check(attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, nullptr, off));
    gate("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", "1");
    check(attention_cpu(w, cfg, input.data(), positions, 1, 1, 4, nullptr, next));
    check(bits(next.pooled, off.pooled) && bits(next.cache.keys, off.cache.keys) && !next.cache.physical_f16_attention);
    std::printf("PHYSICAL_F16_ATTENTION PASS %d\n", checks);
    return 0;
}

int main() {
    core_util::clean_exit(run());
}
