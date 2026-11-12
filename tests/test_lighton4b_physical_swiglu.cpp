// Model-free gate and independent GGML graph controls. No model weights.
#include "lighton4b_physical_swiglu.h"
#include "lighton4b_linear_block.h"
#include "lighton4b_attention_block.h"
#include "core/clean_exit.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
using namespace qwen35_ocr;
static void need(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
static int test_main() {
    try {
        const char * flag = "CRISPEMBED_LIGHTON4B_PHYSICAL_SWIGLU";
        unsetenv(flag);
        need(!lighton4b_physical_swiglu_mode(true), "default off");
        std::vector<float> g(9216), u(9216), result{ 123 };
        for (size_t i = 0; i < g.size(); ++i) {
            g[i] = std::sin(float(i) * 0.013f) * 7;
            u[i] = std::cos(float(i) * 0.017f);
        }
        for (const char * value : { "", "0", "false", "FaLsE", "off", "OFF", "no" }) {
            setenv(flag, value, 1);
            need(!lighton4b_physical_swiglu_mode(true), "false alias");
            result = { 123 };
            need(!lighton4b_physical_swiglu_cpu(g, u, 1, 1, result) && result.empty(), "disabled clears");
        }
        setenv("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "1", 1);
        for (const char * key : { "CRISPEMBED_LIGHTON4B_BF16_LINEAR", "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS",
                                  "CRISPEMBED_LIGHTON4B_HEAD_BF16" })
            setenv(key, "0", 1);
        for (const char * value : { "1", "true", "on", "yes" }) {
            setenv(flag, value, 1);
            need(lighton4b_physical_swiglu_mode(true) && !lighton4b_physical_swiglu_mode(false), "physical only");
        }
        for (int n : { 1, 4, 16 }) {
            g.resize(size_t(n) * 9216);
            u.resize(g.size());
            for (size_t i = 0; i < g.size(); ++i) {
                g[i] = std::sin(float(i) * 0.013f) * 7;
                u[i] = std::cos(float(i) * 0.017f);
            }
            for (int threads : { 1, 4 }) {
                need(lighton4b_physical_swiglu_cpu(g, u, n, threads, result), "fused compute");
                auto ctx = std::unique_ptr<ggml_context, decltype(&ggml_free)>(
                    ggml_init({ 4ull * 1024 * 1024, nullptr, false }), ggml_free);
                auto backend = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>(ggml_backend_cpu_init(),
                                                                                           ggml_backend_free);
                need(ctx && backend, "reference backend");
                ggml_backend_cpu_set_n_threads(backend.get(), threads);
                auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 9216, n);
                auto * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 9216, n);
                std::memcpy(a->data, g.data(), g.size() * sizeof(float));
                std::memcpy(b->data, u.data(), u.size() * sizeof(float));
                auto * ref = ggml_swiglu_split(ctx.get(), a, b);
                auto * graph = ggml_new_graph(ctx.get());
                ggml_build_forward_expand(graph, ref);
                need(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "reference compute");
                need(result.size() == g.size() && !std::memcmp(ref->data, result.data(), g.size() * sizeof(float)),
                     "same-input GGML exact");
            }
        }
        auto rejected = [&](int n, int threads) {
            result = { 123 };
            need(!lighton4b_physical_swiglu_cpu(g, u, n, threads, result) && result.empty(), "reject clears");
        };
        rejected(0, 1);
        rejected(17, 1);
        rejected(16, 0);
        rejected(16, 257);
        u.pop_back();
        rejected(16, 1);
        u.push_back(1);
        g[0] = std::numeric_limits<float>::quiet_NaN();
        rejected(16, 1);
        g[0] = 0;
        u[0] = std::numeric_limits<float>::infinity();
        rejected(16, 1);
        u[0] = 0;
        setenv("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "0", 1);
        rejected(16, 1);
        setenv("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL", "1", 1);
        setenv("CRISPEMBED_LIGHTON4B_BF16_LINEAR", "1", 1);
        rejected(16, 1);
        std::puts(
            "physical SwiGLU: gate aliases, physical/BF16 scope, six GGML exact arms, finite/geometry controls pass");
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "physical SwiGLU FAIL: %s\n", e.what());
        return 1;
    }
}
int main() {
    core_util::clean_exit(test_main());
}
