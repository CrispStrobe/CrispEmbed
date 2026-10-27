// Independent implementation of the pinned Apache-2.0 model's mathematical
// contract, not a copied author engine. Source: VAGOsolutions/
// SauerkrautLM-MoEdernColBERT-1.5M@9d360b8b88ae11bea5784bf9012ff509fdd5c2da.
// See docs/moedern-colbert-native.md for source/license and unexecuted scope.
#include "moedern_text_engine_internal.h"
#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

namespace moedern_text_detail {
namespace {
constexpr int H = 128, P = 16, D = 64, HEADS = 2, EXPERTS = 8, TOP = 2;
constexpr float EPS = 1e-5f, ROUTER_EPS = 1e-9f;
constexpr const char * ARCH = "modernbert_hash_moe_v10";
constexpr const char * REV = "9d360b8b88ae11bea5784bf9012ff509fdd5c2da";
using floats = std::vector<float>;
using dims = std::array<int64_t, 4>;
void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
bool finite(const floats & values) {
    return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
}
std::map<std::string, dims> schema() {
    std::map<std::string, dims> s;
    auto add = [&](const std::string & n, int a, int b = 1, int c = 1) { s[n] = { a, b, c, 1 }; };
    add("token_embd.hash.weight", P, 30002);
    add("token_embd.proj.weight", P, H);
    add("token_embd.norm.weight", H);
    add("token_embd.norm.bias", H);
    for (int i = 0; i < 2; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        add(b + "attn_norm.weight", H);
        add(b + "attn_q.weight", H, H);
        add(b + "attn_k.weight", H, D);
        add(b + "attn_v.weight", H, D);
        add(b + "attn_output.weight", H, H);
        add(b + "ffn_norm.weight", H);
        add(b + "ffn_gate_inp.weight", H, EXPERTS);
        for (const auto * name : { "up", "gate", "down" }) {
            add(b + "ffn_" + name + "_exps.weight", H, H, EXPERTS);
            add(b + "ffn_" + name + "_shexp.weight", H, H);
        }
    }
    add("output_norm.weight", H);
    add("dense_head.weight", H, H);
    return s;
}
int key(gguf_context * g, const std::string & name, gguf_type type) {
    const int at = gguf_find_key(g, name.c_str());
    require(at >= 0 && gguf_get_kv_type(g, at) == type, "Missing/wrong metadata type: " + name);
    return at;
}
void metadata(gguf_context * g) {
    require(std::string(gguf_get_val_str(g, key(g, "general.architecture", GGUF_TYPE_STRING))) == ARCH,
            "Only explicit MoEdern architecture accepted");
    const std::map<std::string, uint32_t> ints = { { "hidden_size", H },
                                                   { "num_hidden_layers", 2 },
                                                   { "num_attention_heads", 2 },
                                                   { "num_key_value_heads", 1 },
                                                   { "vocab_size", 30002 },
                                                   { "hash_projections", P },
                                                   { "max_position_embeddings", 512 },
                                                   { "num_routed_experts", 8 },
                                                   { "num_shared_experts", 1 },
                                                   { "expert_dim", H },
                                                   { "top_k", 2 },
                                                   { "head_dim", D },
                                                   { "out_dim", H },
                                                   { "query_marker_token_id", 30000 },
                                                   { "document_marker_token_id", 30001 },
                                                   { "mask_token_id", 4 } };
    for (const auto & item : ints) {
        const auto name = std::string(ARCH) + "." + item.first;
        require(gguf_get_val_u32(g, key(g, name, GGUF_TYPE_UINT32)) == item.second,
                "Unsupported 1.5M metadata: " + name);
    }
    for (const auto & item : std::map<std::string, float>{ { "rope_theta", 160000.f },
                                                           { "norm_eps", EPS },
                                                           { "hash_norm_eps", EPS },
                                                           { "router_renorm_eps", ROUTER_EPS } }) {
        const auto name = std::string(ARCH) + "." + item.first;
        require(gguf_get_val_f32(g, key(g, name, GGUF_TYPE_FLOAT32)) == item.second,
                "Arithmetic metadata differs: " + name);
    }
    for (const auto & item : std::map<std::string, std::string>{
             { "hidden_activation", "gelu_erf" },
             { "quant_recipe", "f32" },
             { "source_revision", REV },
             { "tokenizer_sidecar_git_blob_sha1", "e1fc54fd0e0456a6c76106be4a338bfc08f29861" } }) {
        const auto name = std::string(ARCH) + "." + item.first;
        require(gguf_get_val_str(g, key(g, name, GGUF_TYPE_STRING)) == item.second,
                "Source/profile metadata differs: " + name);
    }
    const auto tokenizer =
        std::string(gguf_get_val_str(g, key(g, std::string(ARCH) + ".tokenizer_sidecar_sha256", GGUF_TYPE_STRING)));
    require(tokenizer.size() == 64 && tokenizer.find_first_not_of("0123456789abcdef") == std::string::npos,
            "Tokenizer sidecar digest required");
    const auto expected = schema();
    require(gguf_get_n_tensors(g) == int64_t(expected.size()), "Exact 32 tensors required");
    std::map<std::string, bool> seen;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const std::string name = gguf_get_tensor_name(g, i);
        auto it = expected.find(name);
        require(it != expected.end() && !seen[name], "Unknown/duplicate tensor: " + name);
        seen[name] = true;
        require(gguf_get_tensor_type(g, i) == GGML_TYPE_F32, "Only lossless F32 tensors accepted");
        const auto * ne = gguf_get_tensor_ne(g, i);
        require(std::equal(it->second.begin(), it->second.end(), ne), "Tensor shape differs: " + name);
    }
}
struct arena {
    ggml_context * ctx = nullptr;
    arena() {
        ctx = ggml_init({ 16 * 1024 * 1024, nullptr, false });
        require(ctx, "Graph allocation failed");
    }
    ~arena() {
        if (ctx) ggml_free(ctx);
    }
    arena(const arena &) = delete;
};
floats compute(arena & a, ggml_tensor * output, int threads) {
    auto * graph = ggml_new_graph_custom(a.ctx, 128, false);
    ggml_build_forward_expand(graph, output);
    require(ggml_graph_compute_with_ctx(a.ctx, graph, threads) == GGML_STATUS_SUCCESS, "CPU graph compute failed");
    require(output->type == GGML_TYPE_F32 && ggml_is_contiguous(output), "Unexpected graph output layout");
    const auto * begin = ggml_get_data_f32(output);
    floats result(begin, begin + ggml_nelements(output));
    require(finite(result), "Nonfinite graph output");
    return result;
}
ggml_tensor * input(arena & a, const floats & x, int width) {
    require(width > 0 && x.size() % width == 0, "Invalid projection input shape");
    require(finite(x), "Nonfinite projection input");
    auto * t = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, width, int64_t(x.size() / width));
    std::memcpy(t->data, x.data(), x.size() * sizeof(float));
    return t;
}
floats linear(ggml_tensor * weights, const floats & x, int threads, int expert = -1) {
    arena a;
    auto * w = weights;
    if (expert >= 0) {
        require(expert < EXPERTS && weights->ne[2] == EXPERTS, "Invalid expert");
        w = ggml_view_2d(a.ctx, weights, weights->ne[0], weights->ne[1], weights->nb[1],
                         size_t(expert) * weights->nb[2]);
    }
    return compute(a, ggml_mul_mat(a.ctx, w, input(a, x, int(w->ne[0]))), threads);
}
floats normalized(const floats & x, ggml_tensor * weight, ggml_tensor * bias, bool rms, int threads) {
    arena a;
    auto * y = rms ? ggml_rms_norm(a.ctx, input(a, x, H), EPS) : ggml_norm(a.ctx, input(a, x, H), EPS);
    y = ggml_mul(a.ctx, y, weight);
    if (bias) y = ggml_add(a.ctx, y, bias);
    return compute(a, y, threads);
}
void add(floats & x, const floats & y) {
    require(x.size() == y.size(), "Residual shape differs");
    for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
}
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
void emit(const observer & dump, const std::string & name, const floats & x, std::vector<int64_t> shape) {
    require(finite(x), "Nonfinite stage: " + name);
    if (dump)
        require(dump(name, { x.data(), scalar::f32, std::move(shape), x.size() * sizeof(float) }),
                "Observer rejected: " + name);
}
void emit_ids(const observer & dump, const std::string & name, const std::vector<int64_t> & x,
              std::vector<int64_t> shape) {
    if (dump)
        require(dump(name, { x.data(), scalar::i64, std::move(shape), x.size() * sizeof(int64_t) }),
                "Observer rejected: " + name);
}
#else
// Production retains the old finiteness checks, with no callback or labels.
#define emit(callback, label, values, ...) require(finite(values), "Nonfinite MoEdern stage")
#define emit_ids(...) ((void)0)
#endif
void softmax(float * x, int width) {
    const float maximum = *std::max_element(x, x + width);
    float sum = 0;
    for (int i = 0; i < width; ++i) {
        x[i] = std::exp(x[i] - maximum);
        sum += x[i];
    }
    require(std::isfinite(sum) && sum > 0, "Invalid softmax");
    for (int i = 0; i < width; ++i) x[i] /= sum;
}
floats gelu_product(const floats & gate, const floats & up) {
    require(gate.size() == up.size(), "Expert intermediate shape differs");
    floats x(gate.size());
    constexpr float inv_sqrt2 = .7071067811865475244f;
    for (size_t i = 0; i < x.size(); ++i) x[i] = (.5f * gate[i] * (1.f + std::erf(gate[i] * inv_sqrt2))) * up[i];
    return x;
}
bool valid_input(const int32_t * ids, const uint8_t * mask, size_t count, role task) {
    if (!ids || !mask || count < 3 || count > 512 || (task != role::query && task != role::document)) return false;
    if (ids[0] != 2 || ids[1] != (task == role::query ? 30000 : 30001) || !mask[0] || !mask[1]) return false;
    if (task == role::query && count != 32) return false;
    bool pad = false;
    size_t last = 0;
    for (size_t i = 0; i < count; ++i) {
        if (ids[i] < 0 || ids[i] >= 30002 || mask[i] > 1) return false;
        if (!mask[i]) {
            pad = true;
            if (task == role::query && ids[i] != 4) return false;
        } else {
            if (pad) return false;
            last = i;
        }
    }
    return ids[last] == 3;
}
} // namespace

struct context {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;
    int threads = 1;
    std::string failure;
    std::mutex mutex;
    ~context() {
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
    ggml_tensor * get(const std::string & name) const { return weights.tensors.at(name); }
};

context * init(const char * path, int threads, std::string & failure) {
    failure.clear();
    try {
        require(core_env::on("CRISPEMBED_MOEDERN_COLBERT"), "MoEdern experimental gate is OFF");
        require(path && *path && threads >= 1 && threads <= 64, "Invalid path/thread count");
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(file && file.tellg() > 0 && file.tellg() <= 16 * 1024 * 1024, "Bounded 1.5M model file required");
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(core_gguf::open_metadata(path), gguf_free);
        require(bool(meta), "GGUF metadata failed");
        metadata(meta.get());
        std::unique_ptr<context> c(new context);
        c->threads = threads;
        c->backend = ggml_backend_cpu_init();
        require(c->backend, "CPU backend allocation failed");
        require(core_gguf::load_weights(path, c->backend, "moedern1p5m", c->weights), "F32 weight loading failed");
        require(c->weights.tensors.size() == 32, "Loaded tensor count differs");
        for (const auto & row : schema()) {
            auto * tensor = c->get(row.first);
            require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                        std::equal(row.second.begin(), row.second.end(), tensor->ne),
                    "Loaded tensor layout differs");
            floats data(static_cast<size_t>(ggml_nelements(tensor)));
            ggml_backend_tensor_get(tensor, data.data(), 0, data.size() * sizeof(float));
            require(finite(data), "Nonfinite weight: " + row.first);
        }
        return c.release();
    } catch (const std::exception & e) {
        failure = e.what();
        return nullptr;
    }
}
void free(context * ctx) {
    delete ctx;
}
const char * error(const context * ctx) {
    return ctx ? ctx->failure.c_str() : "Null MoEdern context";
}

bool encode_tokens(context * ctx, const int32_t * ids, const uint8_t * supplied, size_t count, role task,
                   result & output
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
                   ,
                   const observer & dump
#endif
) {
    output = {};
    if (!ctx) return false;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->failure.clear();
    try {
        require(core_env::on("CRISPEMBED_MOEDERN_COLBERT"), "MoEdern experimental gate is OFF");
        require(valid_input(ids, supplied, count, task), "Invalid explicit role/ID/supplied-mask contract");
        const int S = int(count);
        result out;
        out.supplied_mask.assign(supplied, supplied + count);
        out.effective_mask = out.supplied_mask;
        if (task == role::query) std::fill(out.effective_mask.begin(), out.effective_mask.end(), 1);
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
        std::vector<int64_t> integers(ids, ids + count);
        emit_ids(dump, "input_ids", integers, { 1, S });
        integers.assign(supplied, supplied + count);
        emit_ids(dump, "supplied_mask", integers, { 1, S });
        integers.assign(out.effective_mask.begin(), out.effective_mask.end());
        emit_ids(dump, "effective_mask", integers, { 1, S });
#endif
        floats hidden;
        {
            arena a;
            auto * indices = ggml_new_tensor_1d(a.ctx, GGML_TYPE_I32, S);
            std::memcpy(indices->data, ids, count * sizeof(int32_t));
            hidden = compute(a, ggml_get_rows(a.ctx, ctx->get("token_embd.hash.weight"), indices), ctx->threads);
        }
        emit(dump, "embedding.lookup", hidden, { 1, S, P });
        hidden = linear(ctx->get("token_embd.proj.weight"), hidden, ctx->threads);
        emit(dump, "embedding.projection", hidden, { 1, S, H });
        hidden = normalized(hidden, ctx->get("token_embd.norm.weight"), ctx->get("token_embd.norm.bias"), false,
                            ctx->threads);
        emit(dump, "embedding.norm", hidden, { 1, S, H });
        for (int layer = 0; layer < 2; ++layer) {
            const std::string b = "blk." + std::to_string(layer) + ".";
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
            const std::string n = "layer." + std::to_string(layer) + ".";
#endif
            floats norm = normalized(hidden, ctx->get(b + "attn_norm.weight"), nullptr, true, ctx->threads);
            emit(dump, n + "attn.norm", norm, { 1, S, H });
            auto q = linear(ctx->get(b + "attn_q.weight"), norm, ctx->threads);
            auto k = linear(ctx->get(b + "attn_k.weight"), norm, ctx->threads);
            auto v = linear(ctx->get(b + "attn_v.weight"), norm, ctx->threads);
            emit(dump, n + "attn.q", q, { 1, S, H });
            emit(dump, n + "attn.k", k, { 1, S, D });
            emit(dump, n + "attn.v", v, { 1, S, D });
            floats qr(size_t(HEADS) * S * D), kr(qr.size()), vr(qr.size());
            for (int head = 0; head < HEADS; ++head)
                for (int t = 0; t < S; ++t)
                    for (int i = 0; i < D / 2; ++i) {
                        const float inv = 1.f / std::pow(160000.f, float(2 * i) / D), angle = float(t) * inv;
                        const float co = std::cos(angle), si = std::sin(angle);
                        const size_t a = (size_t(t) * HEADS + head) * D + i, z = (size_t(head) * S + t) * D + i;
                        qr[z] = q[a] * co - q[a + D / 2] * si;
                        qr[z + D / 2] = q[a + D / 2] * co + q[a] * si;
                        const size_t ka = size_t(t) * D + i;
                        kr[z] = k[ka] * co - k[ka + D / 2] * si;
                        kr[z + D / 2] = k[ka + D / 2] * co + k[ka] * si;
                        vr[z] = v[ka];
                        vr[z + D / 2] = v[ka + D / 2];
                    }
            emit(dump, n + "attn.q_rope", qr, { 1, HEADS, S, D });
            emit(dump, n + "attn.k_rope_repeat", kr, { 1, HEADS, S, D });
            emit(dump, n + "attn.v_repeat", vr, { 1, HEADS, S, D });
            floats heads(qr.size()), probabilities(size_t(S) * S * HEADS);
            for (int head = 0; head < HEADS; ++head)
                for (int t = 0; t < S; ++t) {
                    float * p = probabilities.data() + (size_t(head) * S + t) * S;
                    for (int j = 0; j < S; ++j) {
                        float dot = 0;
                        for (int i = 0; i < D; ++i)
                            dot += qr[(size_t(head) * S + t) * D + i] * kr[(size_t(head) * S + j) * D + i];
                        p[j] =
                            out.effective_mask[j] ? dot / std::sqrt(float(D)) : -std::numeric_limits<float>::infinity();
                    }
                    softmax(p, S);
                    for (int i = 0; i < D; ++i) {
                        float sum = 0;
                        for (int j = 0; j < S; ++j) sum += p[j] * vr[(size_t(head) * S + j) * D + i];
                        heads[(size_t(head) * S + t) * D + i] = sum;
                    }
                }
            emit(dump, n + "attn.probabilities", probabilities, { 1, HEADS, S, S });
            emit(dump, n + "attn.sdpa", heads, { 1, HEADS, S, D });
            floats merged(size_t(S) * H);
            for (int t = 0; t < S; ++t)
                for (int head = 0; head < HEADS; ++head)
                    std::copy_n(heads.data() + (size_t(head) * S + t) * D, D,
                                merged.data() + (size_t(t) * HEADS + head) * D);
            auto attention = linear(ctx->get(b + "attn_output.weight"), merged, ctx->threads);
            emit(dump, n + "attn.output_projection", attention, { 1, S, H });
            add(hidden, attention);
            emit(dump, n + "attn.residual", hidden, { 1, S, H });
            norm = normalized(hidden, ctx->get(b + "ffn_norm.weight"), nullptr, true, ctx->threads);
            emit(dump, n + "ffn.norm", norm, { 1, S, H });
            auto router = linear(ctx->get(b + "ffn_gate_inp.weight"), norm, ctx->threads);
            emit(dump, n + "ffn.router", router, { S, EXPERTS });
            auto probs = router;
            floats top_weights(size_t(S) * TOP);
            std::vector<int64_t> chosen(size_t(S) * TOP);
            floats margins(S);
            for (int t = 0; t < S; ++t) {
                auto * p = probs.data() + size_t(t) * EXPERTS;
                softmax(p, EXPERTS);
                std::array<int, EXPERTS> order{};
                for (int i = 0; i < EXPERTS; ++i) order[i] = i;
                std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
                margins[t] = p[order[TOP - 1]] - p[order[TOP]];
                require(margins[t] > 0 && p[order[0]] > p[order[1]], "Exact router tie is unqualified");
                const float denom = p[order[0]] + p[order[1]] + ROUTER_EPS;
                for (int i = 0; i < TOP; ++i) {
                    chosen[size_t(t) * TOP + i] = order[i];
                    top_weights[size_t(t) * TOP + i] = p[order[i]] / denom;
                }
            }
            emit(dump, n + "ffn.router_probabilities", probs, { S, EXPERTS });
            emit_ids(dump, n + "ffn.topk_indices", chosen, { S, TOP });
            emit(dump, n + "ffn.topk_weights", top_weights, { S, TOP });
            emit(dump, n + "ffn.selection_margin", margins, { S });
            floats routed(size_t(S) * H, 0);
            // Dense all-expert evaluation with routing weight before down, as
            // in the original. Reduction order still requires actual parity.
            for (int expert = 0; expert < EXPERTS; ++expert) {
                auto up = linear(ctx->get(b + "ffn_up_exps.weight"), norm, ctx->threads, expert);
                auto gate = linear(ctx->get(b + "ffn_gate_exps.weight"), norm, ctx->threads, expert);
                auto activated = gelu_product(gate, up);
                for (int t = 0; t < S; ++t) {
                    float weight = 0;
                    for (int j = 0; j < TOP; ++j)
                        if (chosen[size_t(t) * TOP + j] == expert) weight = top_weights[size_t(t) * TOP + j];
                    for (int j = 0; j < H; ++j) activated[size_t(t) * H + j] *= weight;
                }
                auto down = linear(ctx->get(b + "ffn_down_exps.weight"), activated, ctx->threads, expert);
                add(routed, down);
            }
            emit(dump, n + "ffn.routed", routed, { S, H });
            auto shared_up = linear(ctx->get(b + "ffn_up_shexp.weight"), norm, ctx->threads);
            auto shared_gate = linear(ctx->get(b + "ffn_gate_shexp.weight"), norm, ctx->threads);
            emit(dump, n + "ffn.shared_up", shared_up, { S, H });
            emit(dump, n + "ffn.shared_gate", shared_gate, { S, H });
            auto shared =
                linear(ctx->get(b + "ffn_down_shexp.weight"), gelu_product(shared_gate, shared_up), ctx->threads);
            emit(dump, n + "ffn.shared_down", shared, { S, H });
            add(routed, shared);
            emit(dump, n + "ffn.output", routed, { 1, S, H });
            add(hidden, routed);
            emit(dump, n + "output", hidden, { 1, S, H });
        }
        hidden = normalized(hidden, ctx->get("output_norm.weight"), nullptr, true, ctx->threads);
        emit(dump, "final_norm", hidden, { 1, S, H });
        hidden = linear(ctx->get("dense_head.weight"), hidden, ctx->threads);
        emit(dump, "dense", hidden, { 1, S, H });
        for (int t = 0; t < int(count); ++t) {
            float sum = 0;
            for (int i = 0; i < H; ++i) sum += hidden[size_t(t) * H + i] * hidden[size_t(t) * H + i];
            const float denom = std::max(std::sqrt(sum), 1e-12f);
            for (int i = 0; i < H; ++i) hidden[size_t(t) * H + i] /= denom;
        }
        emit(dump, "normalized", hidden, { 1, S, H });
        out.vectors = std::move(hidden);
        output = std::move(out);
        return true;
    } catch (const std::exception & e) {
        ctx->failure = e.what();
        output = {};
        return false;
    } catch (...) {
#ifdef CRISPEMBED_MOEDERN_COMPOSED_TESTING
        ctx->failure = "Unknown observer/encode exception";
#else
        ctx->failure = "Unknown MoEdern encode exception";
#endif
        output = {};
        return false;
    }
}

bool scoring_mask(const std::vector<int32_t> & ids, const std::vector<uint8_t> & supplied, role task,
                  const std::vector<int32_t> & skip_ids, std::vector<uint8_t> & mask, std::string & failure) {
    mask.clear();
    failure.clear();
    if (ids.size() != supplied.size() || !valid_input(ids.data(), supplied.data(), ids.size(), task)) {
        failure = "Invalid scoring inputs";
        return false;
    }
    if (std::any_of(skip_ids.begin(), skip_ids.end(), [](int32_t id) { return id < 0 || id >= 30002; })) {
        failure = "Invalid skiplist ID";
        return false;
    }
    mask = supplied;
    if (task == role::query)
        std::fill(mask.begin(), mask.end(), 1);
    else
        for (size_t i = 0; i < ids.size(); ++i)
            if (std::find(skip_ids.begin(), skip_ids.end(), ids[i]) != skip_ids.end()) mask[i] = 0;
    return true;
}
bool select_rows(const std::vector<float> & all, const std::vector<uint8_t> & mask, std::vector<float> & selected) {
    selected.clear();
    if (all.size() != mask.size() * H || !finite(all) ||
        std::any_of(mask.begin(), mask.end(), [](uint8_t x) { return x > 1; }))
        return false;
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i]) selected.insert(selected.end(), all.begin() + i * H, all.begin() + (i + 1) * H);
    return true;
}
bool maxsim(const std::vector<float> & q, const std::vector<float> & d, float & score) {
    score = 0;
    if (q.empty() || d.empty() || q.size() % H || d.size() % H || !finite(q) || !finite(d)) return false;
    for (size_t a = 0; a < q.size(); a += H) {
        float best = -std::numeric_limits<float>::infinity();
        for (size_t b = 0; b < d.size(); b += H) {
            float dot = 0;
            for (int i = 0; i < H; ++i) dot += q[a + i] * d[b + i];
            best = std::max(best, dot);
        }
        score += best;
    }
    return std::isfinite(score);
}
} // namespace moedern_text_detail

#ifndef CRISPEMBED_MOEDERN_COMPOSED_TESTING
#undef emit
#undef emit_ids
#endif
