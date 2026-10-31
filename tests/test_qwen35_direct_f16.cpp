#include "core/clean_exit.h"
#include "qwen35_math.h"
#include "qwen35_profile.h"
#include "qwen35_weight_cache.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

using namespace qwen35_ocr;
using outputs = std::vector<std::vector<float>>;

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

// Do not use assert(): these checks must remain active in release builds.
static bool check(bool condition, const char * label) {
    if (!condition) std::fprintf(stderr, "FAIL %s\n", label);
    return condition;
}

static uint64_t calls(const profile_record & record, profile_event event) {
    uint64_t result = 0;
    for (size_t phase = 0; phase < size_t(profile_phase::count); ++phase)
        result += record.events[phase][size_t(event)].calls;
    return result;
}

struct result {
    bool ok = false;
    bool captured = false;
    outputs values;
    profile_record profile;
};

static result project(const std::vector<ggml_tensor *> & matrices, const float * input, int width, int tokens,
                      int threads = 2, converted_weight_cache * cache = nullptr) {
    result r;
    {
        profile_capture capture(r.profile);
        r.captured = capture.enabled();
        r.ok = project_cpu(matrices, input, width, tokens, threads, r.values, cache);
    } // Timers and capture must finish before inspecting counters or moving r.
    return r;
}

static bool profile_is(const char * label, const result & r, uint64_t read, uint64_t convert) {
    const auto actual_read = calls(r.profile, profile_event::source_read);
    const auto actual_convert = calls(r.profile, profile_event::source_convert);
    const auto total = calls(r.profile, profile_event::project_total);
    const bool ok = r.captured && !r.profile.dropped && total == 1 && actual_read == read && actual_convert == convert;
    std::printf("PROFILE case=%s enabled=%d dropped=%d project_total=%llu source_read=%llu source_convert=%llu "
                "expected_read=%llu expected_convert=%llu\n",
                label, int(r.captured), int(r.profile.dropped), static_cast<unsigned long long>(total),
                static_cast<unsigned long long>(actual_read), static_cast<unsigned long long>(actual_convert),
                static_cast<unsigned long long>(read), static_cast<unsigned long long>(convert));
    return check(ok, label);
}

static bool exact(const outputs & a, const outputs & b, const char * label) {
    if (!check(a.size() == b.size(), label)) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!check(a[i].size() == b[i].size(), label)) return false;
        if (!a[i].empty() && std::memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float))) {
            for (size_t j = 0; j < a[i].size(); ++j) {
                if (std::memcmp(&a[i][j], &b[i][j], sizeof(float))) {
                    std::fprintf(stderr, "FAIL %s matrix=%zu index=%zu reference=%.9g actual=%.9g\n", label, i, j,
                                 double(a[i][j]), double(b[i][j]));
                    return false;
                }
            }
        }
    }
    return true;
}

// This is a synthetic operator sanity budget, not a decoded-quality gate.
// Direct F16 x F32 may round differently from materialized F32 x F32.
static bool close(const outputs & reference, const outputs & actual, const char * label, size_t tokens = 3) {
    if (!check(reference.size() == actual.size(), label)) return false;
    constexpr double abs_tolerance = 0.002;
    constexpr double cosine_minimum = 0.99999;
    bool ok = true;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!check(reference[i].size() == actual[i].size(), label)) return false;
        if (!check(tokens && reference[i].size() % tokens == 0, "numeric row shape")) return false;
        double max_abs = 0, max_ref = 0, sum_abs = 0, sum_sq = 0, cos_min = 1.0;
        double dot = 0, ref_sq = 0, actual_sq = 0;
        const size_t row_size = reference[i].size() / tokens;
        size_t worst = 0, violations = 0;
        for (size_t j = 0; j < reference[i].size(); ++j) {
            const double a = reference[i][j], b = actual[i][j];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                std::fprintf(stderr, "FAIL %s matrix=%zu index=%zu nonfinite reference=%.9g actual=%.9g\n", label, i, j,
                             a, b);
                ok = false;
                continue;
            }
            const double diff = std::fabs(a - b);
            if (diff > max_abs) {
                max_abs = diff;
                worst = j;
            }
            max_ref = std::max(max_ref, std::fabs(a));
            sum_abs += diff;
            sum_sq += diff * diff;
            dot += a * b;
            ref_sq += a * a;
            actual_sq += b * b;
            if (diff > abs_tolerance) ++violations;
            if ((j + 1) % row_size == 0) {
                const double cosine =
                    ref_sq && actual_sq ? dot / std::sqrt(ref_sq * actual_sq) : (ref_sq == actual_sq ? 1.0 : 0.0);
                cos_min = std::min(cos_min, cosine);
                dot = ref_sq = actual_sq = 0;
            }
        }
        std::printf("NUMERIC case=%s matrix=%zu count=%zu max_abs=%.9g mean_abs=%.9g rms=%.9g "
                    "cos_min=%.12g max_ref=%.9g worst_index=%zu violations=%zu abs_tol=%.9g cosine_min=%.9g\n",
                    label, i, reference[i].size(), max_abs, reference[i].empty() ? 0.0 : sum_abs / reference[i].size(),
                    reference[i].empty() ? 0.0 : std::sqrt(sum_sq / reference[i].size()), cos_min, max_ref, worst,
                    violations, abs_tolerance, cosine_minimum);
        if (violations || cos_min < cosine_minimum) ok = false;
    }
    return check(ok, label);
}

static bool rejected(const char * label, const std::vector<ggml_tensor *> & matrices, const float * input, int width,
                     int tokens, int threads) {
    outputs out{ { 123.0f } };
    const bool ok = project_cpu(matrices, input, width, tokens, threads, out);
    if (ok || !out.empty()) {
        std::fprintf(stderr, "FAIL %s returned=%d output_matrices=%zu\n", label, int(ok), out.size());
        return false;
    }
    return true;
}

static int run() {
    env("CRISPEMBED_QWEN35_OCR", "1");
    env("CRISPEMBED_QWEN35_PROFILE", "1");
    env("CRISPEMBED_QWEN35_WEIGHT_CACHE", "0");
    env("CRISPEMBED_QWEN35_PARALLEL_F16", "0");
    env("CRISPEMBED_QWEN35_QUANT", "0");
    env("CRISPEMBED_QWEN35_IMATRIX", "0");
    using context_owner = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using backend_owner = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using buffer_owner = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    context_owner ctx(ggml_init({ 16 * 1024, nullptr, true }), ggml_free);
    if (!check(bool(ctx), "context allocation")) return 1;
    constexpr int width = 64, tokens = 3;
    auto * half = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, width, 7);
    auto * half2 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, width, 5);
    auto * full = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 6);
    auto * bad_half = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, width, 7);
    auto * large_half = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 1024, 3584);
    auto * half_view = ggml_view_2d(ctx.get(), half, width, 7, half->nb[1], 0);
    ggml_set_name(half, "direct-half-1");
    ggml_set_name(half2, "direct-half-2");
    ggml_set_name(full, "fallback-f32");
    ggml_set_name(bad_half, "nonfinite-half");
    ggml_set_name(large_half, "large-direct-half");
    backend_owner backend(ggml_backend_cpu_init(), ggml_backend_free);
    if (!check(bool(backend), "backend allocation")) return 1;
    buffer_owner buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    if (!check(bool(buffer), "tensor buffer allocation")) return 1;
    for (auto * t : { half, half2, bad_half, large_half }) {
        for (size_t i = 0; i < size_t(ggml_nelements(t)); ++i)
            static_cast<ggml_fp16_t *>(t->data)[i] = ggml_fp32_to_fp16(float(int((i * 17 + 3) % 71) - 35) / 53.0f);
    }
    for (size_t i = 0; i < size_t(ggml_nelements(full)); ++i)
        static_cast<float *>(full->data)[i] = float(int((i * 11 + 5) % 61) - 30) / 47.0f;
    std::vector<float> input(width * tokens);
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i * 13 + 7) % 67) - 33) / 59.0f;

    // Unset, empty and literal zero must all select the old materialized path.
    env("CRISPEMBED_QWEN35_DIRECT_F16", "0");
    const auto baseline = project({ half, half2 }, input.data(), width, tokens);
    if (!check(baseline.ok && baseline.values.size() == 2 && baseline.values[0].size() == size_t(7 * tokens) &&
                   baseline.values[1].size() == size_t(5 * tokens),
               "baseline projection shape") ||
        !profile_is("gate-zero", baseline, 2, 2))
        return 1;
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        env("CRISPEMBED_QWEN35_DIRECT_F16", value);
        const auto r = project({ half, half2 }, input.data(), width, tokens);
        const char * label = !value ? "gate-unset" : (*value ? "gate-zero-repeat" : "gate-empty");
        if (!check(r.ok, label) || !profile_is(label, r, 2, 2) || !exact(baseline.values, r.values, label)) return 1;
    }

    env("CRISPEMBED_QWEN35_DIRECT_F16", "1");
    const auto direct = project({ half, half2 }, input.data(), width, tokens);
    if (!check(direct.ok, "direct multi-token") || !profile_is("direct multi-token", direct, 0, 0) ||
        !close(baseline.values, direct.values, "direct multi-token"))
        return 1;
    const auto view_projection = project({ half_view }, input.data(), width, tokens);
    if (!check(view_projection.ok, "view projection") || !profile_is("view fallback", view_projection, 1, 1) ||
        !exact({ baseline.values[0] }, view_projection.values, "view fallback values"))
        return 1;
    env("CRISPEMBED_QWEN35_DIRECT_F16", "0");
    const auto single_reference = project({ half }, input.data(), width, 1, 1);
    env("CRISPEMBED_QWEN35_DIRECT_F16", "1");
    const auto single_direct = project({ half }, input.data(), width, 1, 1);
    if (!check(single_reference.ok && single_direct.ok && single_reference.values.size() == 1 &&
                   single_reference.values[0].size() == 7,
               "single-token projections") ||
        !profile_is("direct single-token", single_direct, 0, 0) ||
        !close(single_reference.values, single_direct.values, "direct single-token", 1))
        return 1;

    std::vector<float> large_input(1024);
    for (size_t i = 0; i < large_input.size(); ++i) large_input[i] = float(int((i * 13 + 7) % 67) - 33) / 59.0f;
    env("CRISPEMBED_QWEN35_DIRECT_F16", "0");
    const auto large_reference = project({ large_half }, large_input.data(), 1024, 1);
    env("CRISPEMBED_QWEN35_DIRECT_F16", "1");
    const auto large_direct = project({ large_half }, large_input.data(), 1024, 1);
    if (!check(large_reference.ok && large_direct.ok, "large projections") ||
        !profile_is("direct large", large_direct, 0, 0) ||
        !close(large_reference.values, large_direct.values, "direct large", 1))
        return 1;

    // F32 must still materialize when adjacent to a direct F16 matrix.
    env("CRISPEMBED_QWEN35_DIRECT_F16", "0");
    const auto mixed_reference = project({ half, full }, input.data(), width, tokens);
    env("CRISPEMBED_QWEN35_DIRECT_F16", "1");
    const auto mixed = project({ half, full }, input.data(), width, tokens);
    if (!check(mixed_reference.ok && mixed.ok && mixed.values.size() == 2 && mixed_reference.values.size() == 2,
               "mixed projections") ||
        !profile_is("mixed F16/F32", mixed, 1, 1) || !close(mixed_reference.values, mixed.values, "mixed F16/F32") ||
        !exact({ mixed_reference.values[1] }, { mixed.values[1] }, "mixed F32 exact fallback"))
        return 1;

    // A pre-existing converted entry must neither be read nor added to for F16.
    env("CRISPEMBED_QWEN35_WEIGHT_CACHE", "1");
    env("CRISPEMBED_QWEN35_DIRECT_F16", "0");
    converted_weight_cache cache(64 * 1024);
    const auto warmup = project({ half }, input.data(), width, tokens, 2, &cache);
    if (!check(warmup.ok && cache.entries() == 1 && cache.bytes() > 0 && cache.hits() == 0, "cache warmup")) return 1;
    const auto entries = cache.entries(), bytes = cache.bytes(), hits = cache.hits();
    env("CRISPEMBED_QWEN35_DIRECT_F16", "1");
    const auto bypass = project({ half, half2 }, input.data(), width, tokens, 2, &cache);
    if (!check(bypass.ok, "cache bypass projection") || !profile_is("cache bypass", bypass, 0, 0) ||
        !check(cache.entries() == entries && cache.bytes() == bytes && cache.hits() == hits, "cache bypass counters") ||
        !close(baseline.values, bypass.values, "cache bypass")) {
        std::fprintf(stderr, "CACHE entries=%zu/%zu bytes=%zu/%zu hits=%zu/%zu\n", cache.entries(), entries,
                     cache.bytes(), bytes, cache.hits(), hits);
        return 1;
    }
    // The F32 member of a mixed projection retains ordinary cache admission/hits.
    const auto mixed_cold = project({ half, full }, input.data(), width, tokens, 2, &cache);
    if (!check(mixed_cold.ok && cache.entries() == entries + 1 && cache.bytes() > bytes && cache.hits() == hits,
               "mixed cache cold admission") ||
        !profile_is("mixed cache cold", mixed_cold, 1, 1) ||
        !close(mixed_reference.values, mixed_cold.values, "mixed cache cold"))
        return 1;
    const auto mixed_warm = project({ half, full }, input.data(), width, tokens, 2, &cache);
    if (!check(mixed_warm.ok && cache.entries() == entries + 1 && cache.hits() == hits + 1,
               "mixed cache warm F32 hit") ||
        !profile_is("mixed cache warm", mixed_warm, 0, 0) ||
        !exact(mixed_cold.values, mixed_warm.values, "mixed cache cold/warm"))
        return 1;
    env("CRISPEMBED_QWEN35_WEIGHT_CACHE", "0");

    if (!rejected("null matrix", { nullptr }, input.data(), width, tokens, 2) ||
        !rejected("empty matrices", {}, input.data(), width, tokens, 2) ||
        !rejected("too many matrices", { half, half, half, half, half }, input.data(), width, tokens, 2) ||
        !rejected("wrong width", { half }, input.data(), width - 1, tokens, 2) ||
        !rejected("null input", { half }, nullptr, width, tokens, 2) ||
        !rejected("zero tokens", { half }, input.data(), width, 0, 2) ||
        !rejected("zero threads", { half }, input.data(), width, tokens, 0))
        return 1;
    std::vector<float> bad_input = input;
    bad_input[13] = std::numeric_limits<float>::infinity();
    if (!rejected("infinite input", { half }, bad_input.data(), width, tokens, 2)) return 1;
    // Use nonzero inputs so an unvalidated Inf cannot be hidden by an unused row.
    std::vector<float> ones(width, 1.0f);
    static_cast<ggml_fp16_t *>(bad_half->data)[width * 6 + 17] =
        ggml_fp32_to_fp16(std::numeric_limits<float>::infinity());
    for (const char * gate : { "0", "1" }) {
        env("CRISPEMBED_QWEN35_DIRECT_F16", gate);
        if (!rejected(gate[0] == '0' ? "infinite F16 weight gate=0" : "infinite F16 weight gate=1", { bad_half },
                      ones.data(), width, 1, 2))
            return 1;
        if (!rejected(gate[0] == '0' ? "infinite second F16 weight gate=0" : "infinite second F16 weight gate=1",
                      { half, bad_half }, ones.data(), width, 1, 2))
            return 1;
    }
    env("CRISPEMBED_QWEN35_DIRECT_F16", "1");
    auto * full_data = static_cast<float *>(full->data);
    const float saved = full_data[width * 5 + 17];
    full_data[width * 5 + 17] = std::numeric_limits<float>::infinity();
    if (!rejected("infinite F32 fallback weight", { full }, ones.data(), width, 1, 2)) return 1;
    full_data[width * 5 + 17] = saved;
    env("CRISPEMBED_QWEN35_OCR", "0");
    if (!rejected("OCR gate off", { half }, input.data(), width, tokens, 2)) return 1;
    std::puts("DIRECT_F16_CONTROLS pass=1 gate_off=1 direct_numeric=1 profile=1 mixed_f32=1 cache_bypass=1 "
              "invalid=1 nonfinite=1");
    return 0;
}

int main() {
    core_util::clean_exit(run());
}
