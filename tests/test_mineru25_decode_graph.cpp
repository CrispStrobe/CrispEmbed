#include "core/clean_exit.h"
// Weight-free execution of the actual GGML decoder, using two tiny layers.
// Verifies dynamic cache writes, 256-token bucket transitions, and request
// isolation against the preserved graph path. This is not model quality proof.
#include "qwen2vl_ocr.h"
#include "mineru_graph_ops.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

#define REQUIRE(cond)                                                                                                  \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond);                                               \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

static bool quick_gelu_control(ggml_backend_t backend) {
    std::vector<float> values{ -100.0f,  -10.12345f, -3.12345f, -1.0003f,  -0.10007f, 0.0f,
                               0.10007f, 1.0003f,    3.12345f,  10.12345f, 100.0f };
    for (int i = 0; i <= 256; ++i) values.push_back(-8.0f + float(i) * 0.0625f + 0.000317f);
    auto * gc = ggml_init({ 1024 * 1024, nullptr, false });
    if (!gc) return false;
    auto * input = ggml_new_tensor_1d(gc, GGML_TYPE_F32, int64_t(values.size()));
    std::copy(values.begin(), values.end(), static_cast<float *>(input->data));
    auto * output = mineru_graph_ops::quick_gelu_f32(gc, input);
    auto * gf = ggml_new_graph_custom(gc, 128, false);
    ggml_build_forward_expand(gf, output);
    bool passed = ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
    double error = 0.0, rounded_error = 0.0;
    if (passed) {
        const auto * actual = static_cast<const float *>(output->data);
        for (size_t i = 0; i < values.size(); ++i) {
            // Independent double scalar reference, without calling the helper
            // or any GGML activation. Small absolute tolerance permits F32 math.
            const double x = values[i];
            const double expected = x / (1.0 + std::exp(-1.702 * x));
            error = std::max(error, std::fabs(double(actual[i]) - expected));
            // Verify the chosen inputs would detect an input/output F16 LUT.
            // This is independent of whether a future backend changes its LUT.
            const double half_x = ggml_fp16_to_fp32(ggml_fp32_to_fp16(values[i]));
            const float half_result =
                ggml_fp16_to_fp32(ggml_fp32_to_fp16(float(half_x / (1.0 + std::exp(-1.702 * half_x)))));
            rounded_error = std::max(rounded_error, std::fabs(double(half_result) - expected));
            passed = passed && std::isfinite(actual[i]);
        }
    }
    passed = passed && error <= 2e-6 && rounded_error >= 1e-4;
    std::printf("MinerU QuickGELU values=%zu max_abs=%.9g F16_rounding_control=%.9g %s\n", values.size(), error,
                rounded_error, passed ? "PASS" : "FAIL");
    ggml_free(gc);
    return passed;
}

static bool run() {
    env("CRISPEMBED_MINERU25_PRO", "1");
    env("CRISPEMBED_NO_KV_CACHE", "0");
    qwen2vl_ocr::context ctx;
    ctx.is_mineru25_pro = true;
    ctx.verbosity = 0;
    ctx.n_threads = 1;
    auto & h = ctx.m.lhp;
    h.hidden_size = 32;
    h.intermediate_size = 48;
    h.num_hidden_layers = 2;
    h.num_attention_heads = 2;
    h.num_key_value_heads = 1;
    h.head_dim = 16;
    h.vocab_size = 64;
    h.max_position_embeddings = 8192;
    h.rope_sections[0] = 2;
    h.rope_sections[1] = 3;
    h.rope_sections[2] = 3;
    h.rope_sections[3] = 0;

    ctx.backend = ggml_backend_cpu_init();
    REQUIRE(ctx.backend);
    ggml_backend_cpu_set_n_threads(ctx.backend, 1);
    if (!quick_gelu_control(ctx.backend)) {
        qwen2vl_ocr::free_(ctx);
        return false;
    }
    ctx.sched = ggml_backend_sched_new(&ctx.backend, nullptr, 1, 32768, false, false);
    ctx.compute_meta.resize(32768 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32768, false));
    ctx.model_ctx = ggml_init({ 128 * ggml_tensor_overhead() + 4096, nullptr, true });
    REQUIRE(ctx.sched && ctx.model_ctx);

    std::vector<std::pair<ggml_tensor *, bool>> weights;
    auto weight = [&](int in, int out = 1, bool norm = false) {
        auto * t = ggml_new_tensor_2d(ctx.model_ctx, GGML_TYPE_F32, in, out);
        weights.push_back({ t, norm });
        return t;
    };
    ctx.m.embed_tokens = weight(32, 64);
    ctx.m.output_norm_w = weight(32, 1, true);
    ctx.m.lm_head_w = ctx.m.embed_tokens;
    ctx.m.llm_layers.resize(2);
    for (auto & layer : ctx.m.llm_layers) {
        layer.attn_norm_w = weight(32, 1, true);
        layer.ffn_norm_w = weight(32, 1, true);
        layer.q_w = weight(32, 32);
        layer.q_b = weight(32);
        layer.k_w = weight(32, 16);
        layer.k_b = weight(16);
        layer.v_w = weight(32, 16);
        layer.v_b = weight(16);
        layer.o_w = weight(32, 32);
        layer.ffn_gate_w = weight(32, 48);
        layer.ffn_up_w = weight(32, 48);
        layer.ffn_down_w = weight(48, 32);
    }
    ctx.model_buf = ggml_backend_alloc_ctx_tensors(ctx.model_ctx, ctx.backend);
    REQUIRE(ctx.model_buf);
    ggml_backend_buffer_set_usage(ctx.model_buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    for (size_t wi = 0; wi < weights.size(); ++wi) {
        auto * tensor = weights[wi].first;
        std::vector<float> values(size_t(ggml_nelements(tensor)));
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = (weights[wi].second ? 1.0f : 0.0f) + 0.037f * std::sin(float((i + 1) * 13 + wi * 17));
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }

    bool passed = true;
    // The second sequence crosses the optimized graph's 256-slot boundary.
    // The third checks that a shorter subsequent request sees no stale cache.
    for (int prompt_size : { 7, 250, 3 }) {
        std::vector<int32_t> prompt(prompt_size);
        for (int i = 0; i < prompt_size; ++i) prompt[i] = (i * 7 + 3) % 64;
        qwen2vl_ocr::generate_result baseline, cached, repeat;
        env("CRISPEMBED_MINERU25_PRO_REUSE_DECODE_GRAPH", "0");
        REQUIRE(qwen2vl_ocr::generate(ctx, nullptr, 0, 0, nullptr, prompt.data(), prompt_size, 17, baseline));
        env("CRISPEMBED_MINERU25_PRO_REUSE_DECODE_GRAPH", "1");
        REQUIRE(qwen2vl_ocr::generate(ctx, nullptr, 0, 0, nullptr, prompt.data(), prompt_size, 17, cached));
        REQUIRE(qwen2vl_ocr::generate(ctx, nullptr, 0, 0, nullptr, prompt.data(), prompt_size, 17, repeat));
        passed = baseline.token_ids.size() == 17 && baseline.token_ids == cached.token_ids &&
                 cached.token_ids == repeat.token_ids && baseline.token_confidences.size() == 17 &&
                 cached.token_confidences.size() == 17 && repeat.token_confidences.size() == 17 &&
                 ctx.kvc.k->type == GGML_TYPE_F32 && ctx.kvc.v->type == GGML_TYPE_F32;
        float error = 0;
        for (size_t i = 0; passed && i < baseline.token_confidences.size(); ++i) {
            error = std::max(error, std::fabs(baseline.token_confidences[i] - cached.token_confidences[i]));
            error = std::max(error, std::fabs(cached.token_confidences[i] - repeat.token_confidences[i]));
        }
        passed = passed && error <= 1e-6f;
        std::printf("MinerU synthetic decode prompt=%d ids=%zu/%zu/%zu probability_max_abs=%.9g %s\n", prompt_size,
                    baseline.token_ids.size(), cached.token_ids.size(), repeat.token_ids.size(), error,
                    passed ? "PASS" : "FAIL");
        if (!passed) break;
    }
    qwen2vl_ocr::generate_result invalid;
    const int32_t token = 1;
    env("CRISPEMBED_MINERU25_PRO", "0");
    const bool gate_refused =
        !qwen2vl_ocr::generate(ctx, nullptr, 0, 0, nullptr, &token, 1, 1, invalid) && invalid.token_ids.empty();
    if (!gate_refused) std::fprintf(stderr, "FAIL: generate ran with CRISPEMBED_MINERU25_PRO=0\n");
    passed = passed && gate_refused;
    qwen2vl_ocr::free_(ctx);
    return passed;
}

int main() {
    core_util::clean_exit(run() ? 0 : 1);
}
