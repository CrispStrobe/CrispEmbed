#include "lighton4b_bf16_linear.h"
#include "core/clean_exit.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
static int run() {
    bool ok = true;
    int checks = 0;
    auto check = [&](bool value) {
        ++checks;
        ok &= value;
        if (!value) std::printf("FAIL decoder-bf16-control=%d\n", checks);
    };
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        gate("CRISPEMBED_LIGHTON4B_BF16_LINEAR", value);
        check(!qwen35_ocr::lighton4b_bf16_linear_enabled());
    }
    check(qwen35_ocr::lighton4b_bf16_round(1.00390625f) == 1.0f);
    check(qwen35_ocr::lighton4b_bf16_round(1.01171875f) == 1.015625f);
    check(qwen35_ocr::lighton4b_bf16_round(-1.00390625f) == -1.0f);
    check(std::signbit(qwen35_ocr::lighton4b_bf16_round(-0.0f)));
    check(std::isinf(qwen35_ocr::lighton4b_bf16_round(std::numeric_limits<float>::infinity())));
    check(std::isnan(qwen35_ocr::lighton4b_bf16_round(std::numeric_limits<float>::quiet_NaN())));
    for (int width : { 128, 2560 }) {
        std::vector<float> ones(size_t(width), 1.0f), zero(size_t(width), 0.0f);
        const float scale = qwen35_ocr::lighton4b_bf16_rms_scale(ones.data(), width, 1e-6f);
        check(std::isfinite(scale) && qwen35_ocr::lighton4b_bf16_round(scale) == 1.0f);
        const float empty_scale = qwen35_ocr::lighton4b_bf16_rms_scale(zero.data(), width, 1e-6f);
        check(std::isfinite(empty_scale) && std::abs(empty_scale - 1000.0f) < 0.001f);
    }
    check(std::isnan(qwen35_ocr::lighton4b_bf16_rms_scale(nullptr, 128, 1e-6f)));
    std::vector<float> q(128), k(128), v(128), g(1, -0.1f), beta(1, 0.5f), legacy, state, output, final_state;
    q[0] = k[0] = 1.0f;
    v[0] = 0.25f;
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_LIGHTON4B_BF16_LINEAR", "0");
    auto candidate = [&]() {
        return qwen35_ocr::delta_pre_normalized_cpu(q.data(), k.data(), v.data(), g.data(), beta.data(), nullptr, 1, 1,
                                                    1, 128, 1, output, final_state);
    };
    output = { 9 };
    final_state = { 9 };
    check(!candidate() && output.empty() && final_state.empty());
    check(qwen35_ocr::delta_cpu(q.data(), k.data(), v.data(), g.data(), beta.data(), nullptr, 1, 1, 1, 128, 1, legacy,
                                state));
    gate("CRISPEMBED_LIGHTON4B_BF16_LINEAR", "1");
    check(qwen35_ocr::lighton4b_bf16_linear_enabled());
    check(candidate() && output.size() == 128 && final_state.size() == 16384 && output[0] > 0 &&
          std::all_of(output.begin(), output.end(), [](float x) { return std::isfinite(x); }));
    std::vector<float> repeated, repeated_state;
    check(qwen35_ocr::delta_cpu(q.data(), k.data(), v.data(), g.data(), beta.data(), nullptr, 1, 1, 1, 128, 1, repeated,
                                repeated_state) &&
          repeated == legacy && repeated_state == state);
    gate("CRISPEMBED_LIGHTON4B_BF16_LINEAR", "0");
    check(!candidate() && output.empty() && final_state.empty());
    std::printf("LIGHTON4B_BF16_LINEAR_CONTROLS checks=%d passed=%d model_payload=0\n", checks, ok);
    return ok ? 0 : 1;
}
int main() {
    core_util::clean_exit(run());
}
