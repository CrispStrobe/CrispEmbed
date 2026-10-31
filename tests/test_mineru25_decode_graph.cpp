#include "core/clean_exit.h"
// Weight-free execution of the actual GGML decoder, using two tiny layers.
// Verifies dynamic cache writes, 256-token bucket transitions, and request
// isolation against the preserved graph path. This is not model quality proof.
#include "qwen2vl_ocr.h"
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
    if (!ctx.backend) return false;
    ggml_backend_cpu_set_n_threads(ctx.backend, 1);
    ctx.sched = ggml_backend_sched_new(&ctx.backend, nullptr, 1, 32768, false, false);
    ctx.compute_meta.resize(32768 * ggml_tensor_overhead() + ggml_graph_overhead_custom(32768, false));
    ctx.model_ctx = ggml_init({ 128 * ggml_tensor_overhead() + 4096, nullptr, true });
    if (!ctx.sched || !ctx.model_ctx) {
        qwen2vl_ocr::free_(ctx);
        return false;
    }

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
    if (!ctx.model_buf) {
        qwen2vl_ocr::free_(ctx);
        return false;
    }
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
        passed = passed && qwen2vl_ocr::generate(ctx, nullptr, 0, 0, nullptr, prompt.data(), prompt_size, 17, baseline);
        env("CRISPEMBED_MINERU25_PRO_REUSE_DECODE_GRAPH", "1");
        passed = passed && qwen2vl_ocr::generate(ctx, nullptr, 0, 0, nullptr, prompt.data(), prompt_size, 17, cached);
        passed = passed && qwen2vl_ocr::generate(ctx, nullptr, 0, 0, nullptr, prompt.data(), prompt_size, 17, repeat);
        if (!passed) break;
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
        std::printf("MinerU synthetic decode prompt=%d ids=%zu probability_max_abs=%.9g %s\n", prompt_size,
                    cached.token_ids.size(), error, passed ? "PASS" : "FAIL");
        if (!passed) break;
    }
    qwen2vl_ocr::generate_result invalid;
    const int32_t token = 1;
    env("CRISPEMBED_MINERU25_PRO", "0");
    passed = passed && !qwen2vl_ocr::generate(ctx, nullptr, 0, 0, nullptr, &token, 1, 1, invalid) &&
             invalid.token_ids.empty();
    qwen2vl_ocr::free_(ctx);
    return passed;
}

int main() {
    core_util::clean_exit(run() ? 0 : 1);
}
