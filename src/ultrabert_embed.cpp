// Pinned UltraBERT eager graph: weighted RMSNorm, split-half RoPE, inclusive
// local128/global every third block, SwiGLU, final RMSNorm and CLS. CPU/F32 only.
#include "ultrabert_embed.h"
#include "ultrabert_profile.h"
#include "core/env_gate.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>

struct ultrabert_context {
    core_gguf::WeightLoad weights;
    ggml_backend_t backend = nullptr;
    BPETokenizer tokenizer;
    std::vector<std::string> vocab;
    ~ultrabert_context() {
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
};
namespace {
using ultrabert_profile::require;
struct Work {
    ggml_context * graph = nullptr;
    ggml_gallocr_t allocator = nullptr;
    ~Work() {
        if (allocator) ggml_gallocr_free(allocator);
        if (graph) ggml_free(graph);
    }
};
std::vector<float> forward(ultrabert_context & ctx, const std::vector<int32_t> & ids, std::vector<float> * hidden) {
    Work work;
    const int T = int(ids.size()), H = 768, D = 128, heads = 6;
    auto weight = [&](const std::string & name, int64_t, int64_t = 1) { return ctx.weights.tensors.at(name); };
    const size_t graph_size = 8192;
    work.graph = ggml_init({ 32 * 1024 * 1024, nullptr, true });
    require(work.graph != nullptr, "Graph metadata allocation failed");
    auto * g = work.graph;
    auto * gf = ggml_new_graph_custom(g, graph_size, false);
    auto * input = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    auto * positions = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    auto * local_mask = ggml_new_tensor_3d(g, GGML_TYPE_F32, T, T, 1);
    auto * global_mask = ggml_new_tensor_3d(g, GGML_TYPE_F32, T, T, 1);
    for (auto * t : { input, positions, local_mask, global_mask }) ggml_set_input(t);
    std::vector<float> local((size_t)T * T), global(local.size());
    std::vector<int32_t> pos(T);
    for (int q = 0; q < T; ++q) {
        pos[q] = q;
        for (int k = 0; k < T; ++k) {
            const size_t index = (size_t)q * T + k;
            global[index] = 0;
            local[index] = std::abs(q - k) <= 128 ? 0 : std::numeric_limits<float>::lowest();
        }
    }
    auto norm = [&](ggml_tensor * t, ggml_tensor * w) { return ggml_mul(g, ggml_rms_norm(g, t, 1e-6f), w); };
    auto * cur = ggml_get_rows(g, weight("token_embd.weight", H, 49152), input);
    for (int i = 0; i < 24; ++i) {
        const std::string wp = "blk." + std::to_string(i) + ".";
        auto * residual = cur;
        auto * normalized = norm(cur, weight(wp + "attn_norm.weight", H));
        auto * q = ggml_mul_mat(g, weight(wp + "attn_q.weight", H, H), normalized);
        auto * k = ggml_mul_mat(g, weight(wp + "attn_k.weight", H, H), normalized);
        auto * v = ggml_mul_mat(g, weight(wp + "attn_v.weight", H, H), normalized);
        const bool is_global = i % 3 == 2;
        const float theta = is_global ? 2560000.f : 10000.f;
        q = ggml_rope_ext(g, ggml_reshape_3d(g, q, D, heads, T), positions, nullptr, D, GGML_ROPE_TYPE_NEOX, 0, theta,
                          1.f, 0.f, 1.f, 0.f, 0.f);
        k = ggml_rope_ext(g, ggml_reshape_3d(g, k, D, heads, T), positions, nullptr, D, GGML_ROPE_TYPE_NEOX, 0, theta,
                          1.f, 0.f, 1.f, 0.f, 0.f);
        q = ggml_permute(g, q, 0, 2, 1, 3);
        k = ggml_permute(g, k, 0, 2, 1, 3);
        v = ggml_permute(g, ggml_reshape_3d(g, v, D, heads, T), 0, 2, 1, 3);
        auto * scores = ggml_scale(g, ggml_mul_mat(g, k, q), 1.f / std::sqrt((float)D));
        scores = ggml_soft_max(g, ggml_add(g, scores, is_global ? global_mask : local_mask));
        auto * vt = ggml_cont(g, ggml_permute(g, v, 1, 0, 2, 3));
        auto * attention = ggml_mul_mat(g, vt, scores);
        attention = ggml_reshape_2d(g, ggml_cont(g, ggml_permute(g, attention, 0, 2, 1, 3)), H, T);
        attention = ggml_mul_mat(g, weight(wp + "attn_output.weight", H, H), attention);
        cur = ggml_add(g, residual, attention);
        residual = cur;
        normalized = norm(cur, weight(wp + "ffn_norm.weight", H));
        auto * gate = ggml_mul_mat(g, weight(wp + "ffn_gate.weight", H, 1024), normalized);
        auto * up = ggml_mul_mat(g, weight(wp + "ffn_up.weight", H, 1024), normalized);
        auto * ffn = ggml_mul_mat(g, weight(wp + "ffn_down.weight", 1024, H), ggml_mul(g, ggml_silu(g, gate), up));
        cur = ggml_add(g, residual, ffn);
    }
    cur = norm(cur, weight("output_norm.weight", H));
    ggml_build_forward_expand(gf, cur);
    work.allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx.backend));
    require(work.allocator && ggml_gallocr_alloc_graph(work.allocator, gf), "Graph allocation failed");
    ggml_backend_tensor_set(input, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
    ggml_backend_tensor_set(local_mask, local.data(), 0, local.size() * sizeof(float));
    ggml_backend_tensor_set(global_mask, global.data(), 0, global.size() * sizeof(float));
    require(ggml_backend_graph_compute(ctx.backend, gf) == GGML_STATUS_SUCCESS, "Graph compute failed");
    std::vector<float> cls(H);
    ggml_backend_tensor_get(cur, cls.data(), 0, H * sizeof(float));
    if (hidden) {
        hidden->resize(size_t(T) * H);
        ggml_backend_tensor_get(cur, hidden->data(), 0, hidden->size() * sizeof(float));
    }
    for (float value : cls) require(std::isfinite(value), "nonfinite CLS");
    if (hidden)
        for (float value : *hidden) require(std::isfinite(value), "nonfinite hidden output");
    return cls;
}
} // namespace

ultrabert_context * ultrabert_init(const char * path, int threads) {
    if (!path || threads <= 0 || !core_env::on("CRISPEMBED_ULTRABERT")) return nullptr;
    try {
        auto ctx = std::make_unique<ultrabert_context>();
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(core_gguf::open_metadata(path), gguf_free);
        ultrabert_profile::validate(meta.get());
        require(ultrabert_profile::load_tokenizer(meta.get(), ctx->tokenizer, ctx->vocab), "unsupported vocabulary");
        ctx->backend = ggml_backend_cpu_init();
        require(ctx->backend != nullptr, "CPU backend unavailable");
        ggml_backend_cpu_set_n_threads(ctx->backend, threads);
        require(core_gguf::load_weights(path, ctx->backend, "ultrabert", ctx->weights), "weight load failed");
        return ctx.release();
    } catch (const std::exception & e) {
        fprintf(stderr, "ultrabert: %s\n", e.what());
        return nullptr;
    }
}
void ultrabert_free(ultrabert_context * ctx) {
    delete ctx;
}
bool ultrabert_encode(ultrabert_context * ctx, const std::string & text, std::vector<float> & pooled,
                      std::vector<int32_t> * ids, std::vector<float> * hidden) {
    pooled.clear();
    if (ids) ids->clear();
    if (hidden) hidden->clear();
    if (!ctx) return false;
    try {
        std::vector<int32_t> tokens;
        require(ultrabert_profile::tokenize(ctx->tokenizer, text, tokens), "unsupported text or token overflow");
        pooled = forward(*ctx, tokens, hidden);
        if (ids) *ids = std::move(tokens);
        return true;
    } catch (const std::exception & e) {
        pooled.clear();
        if (ids) ids->clear();
        if (hidden) hidden->clear();
        fprintf(stderr, "ultrabert: %s\n", e.what());
        return false;
    }
}
void ultrabert_hparams(crispembed_hparams * out) {
    *out = {};
    out->n_vocab = 49152;
    out->n_max_tokens = ultrabert_profile::max_tokens;
    out->n_embd = out->n_output = 768;
    out->n_head = 6;
    out->n_layer = 24;
    out->n_intermediate = 1024;
    out->layer_norm_eps = 1e-6f;
}
const char * ultrabert_token(const ultrabert_context * ctx, int id) {
    if (!ctx || id < 0 || size_t(id) >= ctx->vocab.size()) return nullptr;
    return ctx->vocab[id].c_str();
}
const char * ultrabert_query_prefix() {
    return ultrabert_profile::query_prefix();
}
