// Size-parameterised MoEdern ColBERT engine (SauerkrautLM-MoEdernColBERT 1.5M,
// 27M and 115M, Apache-2.0); 115M adds gated depthwise dilated-conv layers. Same mathematics as
// moedern_text_engine.cpp, which stays byte-identical and keeps its 1.5M-only
// contract; this path is separately gated by CRISPEMBED_MOEDERN_FAMILY.
#include "moedern_family_engine.h"
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
#include <numeric>
#include <stdexcept>

namespace moedern_family {
namespace {
using floats = std::vector<float>;
using dims = std::array<int64_t, 4>;
constexpr float EPS = 1e-5f, ROUTER_EPS = 1e-9f, THETA = 160000.f;
constexpr const char * ARCH = "modernbert_hash_moe_v10";
constexpr int VOCAB = 30002, OUT = 128, KERNEL = 3;

// Only pinned revisions whose original panels are frozen are accepted.
const std::map<std::string, profile> & profiles() {
    static const std::map<std::string, profile> p = {
        { "9d360b8b88ae11bea5784bf9012ff509fdd5c2da", { "1.5M", 128, 2, 1, 16, 8, 2, 128, 2, { 0, 0 } } },
        { "d2b7b1feaaac1bf87510ca9525208c3daeabcc5a", { "27M", 640, 10, 5, 80, 8, 2, 640, 2, { 0, 0 } } },
        { "66380d2ade3b5027bba83bdcdba052d0bb86dcdb",
          { "115M", 1024, 16, 8, 192, 8, 2, 640, 10, { 1, 1, 0, 2, 1, 0, 1, 0, 2, 0 } } },
    };
    return p;
}
void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
bool finite(const floats & v) {
    return std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); });
}
std::map<std::string, dims> schema(const profile & p) {
    std::map<std::string, dims> s;
    auto add = [&](const std::string & n, int a, int b = 1, int c = 1) { s[n] = { a, b, c, 1 }; };
    const int H = p.hidden, KV = p.kv_heads * (p.hidden / p.heads), X = p.expert_dim, E = p.experts;
    add("token_embd.hash.weight", p.hash, VOCAB);
    add("token_embd.proj.weight", p.hash, H);
    add("token_embd.norm.weight", H);
    add("token_embd.norm.bias", H);
    for (int i = 0; i < p.layers; ++i) {
        const std::string b = "blk." + std::to_string(i) + ".";
        if (p.conv_dilation[i] > 0) {
            add(b + "conv_norm.weight", H);
            add(b + "conv_in.weight", H, 3 * H);
            add(b + "conv_gate_b_norm.weight", H);
            add(b + "conv_gate_c_norm.weight", H);
            add(b + "conv_dw.weight", H, KERNEL);
            add(b + "conv_out.weight", H, H);
            continue;
        }
        add(b + "attn_norm.weight", H);
        add(b + "attn_q.weight", H, H);
        add(b + "attn_k.weight", H, KV);
        add(b + "attn_v.weight", H, KV);
        add(b + "attn_output.weight", H, H);
        add(b + "ffn_norm.weight", H);
        add(b + "ffn_gate_inp.weight", H, E);
        add(b + "ffn_up_exps.weight", H, X, E);
        add(b + "ffn_gate_exps.weight", H, X, E);
        add(b + "ffn_down_exps.weight", X, H, E);
        add(b + "ffn_up_shexp.weight", H, X);
        add(b + "ffn_gate_shexp.weight", H, X);
        add(b + "ffn_down_shexp.weight", X, H);
    }
    add("output_norm.weight", H);
    add("dense_head.weight", H, OUT);
    return s;
}
uint32_t u32(gguf_context * g, const std::string & key) {
    const int at = gguf_find_key(g, (std::string(ARCH) + "." + key).c_str());
    require(at >= 0 && gguf_get_kv_type(g, at) == GGUF_TYPE_UINT32, "Missing metadata: " + key);
    return gguf_get_val_u32(g, at);
}
profile read_profile(gguf_context * g) {
    const int arch = gguf_find_key(g, "general.architecture");
    require(arch >= 0 && std::string(gguf_get_val_str(g, arch)) == ARCH, "Only explicit MoEdern architecture accepted");
    const int rev = gguf_find_key(g, (std::string(ARCH) + ".source_revision").c_str());
    require(rev >= 0, "Source revision required");
    const auto it = profiles().find(gguf_get_val_str(g, rev));
    require(it != profiles().end(), "Unqualified MoEdern revision (115M conv variant not supported)");
    const profile & p = it->second;
    const std::map<std::string, uint32_t> expect = { { "hidden_size", uint32_t(p.hidden) },
                                                     { "num_hidden_layers", uint32_t(p.layers) },
                                                     { "num_attention_heads", uint32_t(p.heads) },
                                                     { "num_key_value_heads", uint32_t(p.kv_heads) },
                                                     { "vocab_size", uint32_t(VOCAB) },
                                                     { "hash_projections", uint32_t(p.hash) },
                                                     { "num_routed_experts", uint32_t(p.experts) },
                                                     { "num_shared_experts", 1 },
                                                     { "expert_dim", uint32_t(p.expert_dim) },
                                                     { "top_k", uint32_t(p.top_k) },
                                                     { "head_dim", uint32_t(p.hidden / p.heads) },
                                                     { "out_dim", uint32_t(OUT) } };
    for (const auto & e : expect) require(u32(g, e.first) == e.second, "Profile metadata differs: " + e.first);
    if (std::any_of(p.conv_dilation.begin(), p.conv_dilation.end(), [](int d) { return d > 0; })) {
        require(u32(g, "conv_kernel_size") == KERNEL && u32(g, "conv_use_gate_norm") == 1, "Conv profile differs");
        const int at = gguf_find_key(g, (std::string(ARCH) + ".conv_dilations_per_layer").c_str());
        require(at >= 0 && gguf_get_kv_type(g, at) == GGUF_TYPE_ARRAY && int(gguf_get_arr_n(g, at)) == p.layers,
                "Per-layer dilations required");
        const auto type = gguf_get_arr_type(g, at);
        require(type == GGUF_TYPE_INT32 || type == GGUF_TYPE_UINT32, "Dilation array type differs");
        const auto * d = static_cast<const int32_t *>(gguf_get_arr_data(g, at));
        for (int i = 0; i < p.layers; ++i) require(d[i] == p.conv_dilation[i], "Layer kind/dilation differs");
    }
    const auto expected = schema(p);
    require(gguf_get_n_tensors(g) == int64_t(expected.size()), "Exact tensor inventory required");
    for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
        const auto row = expected.find(gguf_get_tensor_name(g, i));
        require(row != expected.end() && gguf_get_tensor_type(g, i) == GGML_TYPE_F32, "Unknown or non-F32 tensor");
        require(std::equal(row->second.begin(), row->second.end(), gguf_get_tensor_ne(g, i)), "Tensor shape differs");
    }
    return p;
}
struct arena {
    ggml_context * ctx;
    arena() : ctx(ggml_init({ 64 * 1024 * 1024, nullptr, false })) { require(ctx, "Graph allocation failed"); }
    ~arena() { ggml_free(ctx); }
    arena(const arena &) = delete;
};
floats compute(arena & a, ggml_tensor * out, int threads) {
    auto * graph = ggml_new_graph_custom(a.ctx, 128, false);
    ggml_build_forward_expand(graph, out);
    require(ggml_graph_compute_with_ctx(a.ctx, graph, threads) == GGML_STATUS_SUCCESS, "CPU graph compute failed");
    const auto * b = ggml_get_data_f32(out);
    floats r(b, b + ggml_nelements(out));
    require(finite(r), "Nonfinite graph output");
    return r;
}
ggml_tensor * input(arena & a, const floats & x, int width) {
    require(width > 0 && x.size() % width == 0 && finite(x), "Invalid projection input");
    auto * t = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, width, int64_t(x.size() / width));
    std::memcpy(t->data, x.data(), x.size() * sizeof(float));
    return t;
}
floats linear(ggml_tensor * w, const floats & x, int threads, int expert = -1) {
    arena a;
    if (expert >= 0) w = ggml_view_2d(a.ctx, w, w->ne[0], w->ne[1], w->nb[1], size_t(expert) * w->nb[2]);
    return compute(a, ggml_mul_mat(a.ctx, w, input(a, x, int(w->ne[0]))), threads);
}
floats normalized(const floats & x, int width, ggml_tensor * weight, ggml_tensor * bias, bool rms, int threads) {
    arena a;
    auto * y = rms ? ggml_rms_norm(a.ctx, input(a, x, width), EPS) : ggml_norm(a.ctx, input(a, x, width), EPS);
    y = ggml_mul(a.ctx, y, weight);
    if (bias) y = ggml_add(a.ctx, y, bias);
    return compute(a, y, threads);
}
void add(floats & x, const floats & y) {
    require(x.size() == y.size(), "Residual shape differs");
    for (size_t i = 0; i < x.size(); ++i) x[i] += y[i];
}
void softmax(float * x, int n) {
    const float m = *std::max_element(x, x + n);
    float s = 0;
    for (int i = 0; i < n; ++i) s += (x[i] = std::exp(x[i] - m));
    require(std::isfinite(s) && s > 0, "Invalid softmax");
    for (int i = 0; i < n; ++i) x[i] /= s;
}
floats gelu_product(const floats & gate, const floats & up) {
    floats x(gate.size());
    for (size_t i = 0; i < x.size(); ++i)
        x[i] = (.5f * gate[i] * (1.f + std::erf(gate[i] * .7071067811865475244f))) * up[i];
    return x;
}
} // namespace

struct context {
    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad weights;
    profile p;
    int threads = 1;
    std::string failure;
    std::mutex mutex;
    ~context() {
        core_gguf::free_weights(weights);
        if (backend) ggml_backend_free(backend);
    }
    ggml_tensor * get(const std::string & n) const { return weights.tensors.at(n); }
};

context * init(const char * path, int threads, std::string & failure) {
    failure.clear();
    try {
        require(core_env::on("CRISPEMBED_MOEDERN_FAMILY") && core_env::on("CRISPEMBED_MOEDERN_COLBERT"),
                "MoEdern family gates are OFF");
        require(path && *path && threads >= 1 && threads <= 64, "Invalid path/thread count");
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(file && file.tellg() > 0 && file.tellg() <= 512LL * 1024 * 1024, "Bounded model file required");
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(core_gguf::open_metadata(path), gguf_free);
        require(bool(meta), "GGUF metadata failed");
        std::unique_ptr<context> c(new context);
        c->p = read_profile(meta.get());
        c->threads = threads;
        c->backend = ggml_backend_cpu_init();
        require(c->backend && core_gguf::load_weights(path, c->backend, "moedernfamily", c->weights),
                "F32 weight loading failed");
        for (const auto & row : schema(c->p)) {
            auto * t = c->get(row.first);
            require(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) &&
                        std::equal(row.second.begin(), row.second.end(), t->ne),
                    "Loaded tensor layout differs");
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
const profile * get_profile(const context * ctx) {
    return ctx ? &ctx->p : nullptr;
}

bool encode_tokens(context * ctx, const int32_t * ids, const uint8_t * supplied, size_t count, bool query,
                   std::vector<float> & vectors) {
    vectors.clear();
    if (!ctx) return false;
    std::lock_guard<std::mutex> lock(ctx->mutex);
    ctx->failure.clear();
    try {
        require(core_env::on("CRISPEMBED_MOEDERN_FAMILY") && core_env::on("CRISPEMBED_MOEDERN_COLBERT"),
                "MoEdern family gates are OFF");
        require(ids && supplied && count >= 3 && count <= 512 && (!query || count == 32), "Invalid input shape");
        for (size_t i = 0; i < count; ++i)
            require(ids[i] >= 0 && ids[i] < VOCAB && supplied[i] <= 1, "Invalid input id/mask");
        const profile & p = ctx->p;
        const int S = int(count), H = p.hidden, HEADS = p.heads, D = H / HEADS, KVH = p.kv_heads, GROUP = HEADS / KVH,
                  E = p.experts, TOP = p.top_k;
        std::vector<uint8_t> mask(supplied, supplied + count);
        if (query) std::fill(mask.begin(), mask.end(), 1); // query expansion attends to every position
        floats hidden;
        {
            arena a;
            auto * idx = ggml_new_tensor_1d(a.ctx, GGML_TYPE_I32, S);
            std::memcpy(idx->data, ids, count * sizeof(int32_t));
            hidden = compute(a, ggml_get_rows(a.ctx, ctx->get("token_embd.hash.weight"), idx), ctx->threads);
        }
        hidden = linear(ctx->get("token_embd.proj.weight"), hidden, ctx->threads);
        hidden = normalized(hidden, H, ctx->get("token_embd.norm.weight"), ctx->get("token_embd.norm.bias"), false,
                            ctx->threads);
        for (int layer = 0; layer < p.layers; ++layer) {
            const std::string b = "blk." + std::to_string(layer) + ".";
            if (const int dil = p.conv_dilation[layer]; dil > 0) {
                // GatedShortConv: in_norm -> in_proj [B|C|x] -> RMS(B), RMS(C) -> B*x (masked)
                // -> depthwise Conv1d(k=3, dilation, same padding) -> C*conv -> out_proj -> +residual.
                const floats xin = normalized(hidden, H, ctx->get(b + "conv_norm.weight"), nullptr, true, ctx->threads);
                const floats bcx = linear(ctx->get(b + "conv_in.weight"), xin, ctx->threads);
                floats bg(size_t(S) * H), cg(size_t(S) * H), bx(size_t(S) * H);
                for (int t = 0; t < S; ++t) {
                    std::copy_n(bcx.data() + size_t(t) * 3 * H, H, bg.data() + size_t(t) * H);
                    std::copy_n(bcx.data() + size_t(t) * 3 * H + H, H, cg.data() + size_t(t) * H);
                }
                bg = normalized(bg, H, ctx->get(b + "conv_gate_b_norm.weight"), nullptr, true, ctx->threads);
                cg = normalized(cg, H, ctx->get(b + "conv_gate_c_norm.weight"), nullptr, true, ctx->threads);
                for (int t = 0; t < S; ++t)
                    for (int c = 0; c < H; ++c)
                        bx[size_t(t) * H + c] =
                            mask[t] ? bg[size_t(t) * H + c] * bcx[size_t(t) * 3 * H + 2 * H + c] : 0.f;
                auto * dw = ctx->get(b + "conv_dw.weight");
                floats w(size_t(H) * KERNEL);
                ggml_backend_tensor_get(dw, w.data(), 0, w.size() * sizeof(float));
                floats y(size_t(S) * H);
                for (int t = 0; t < S; ++t)
                    for (int c = 0; c < H; ++c) {
                        float acc = 0;
                        for (int tap = 0; tap < KERNEL; ++tap) {
                            const int src = t + (tap - (KERNEL - 1) / 2) * dil;
                            if (src >= 0 && src < S) acc += w[size_t(tap) * H + c] * bx[size_t(src) * H + c];
                        }
                        y[size_t(t) * H + c] = cg[size_t(t) * H + c] * acc;
                    }
                add(hidden, linear(ctx->get(b + "conv_out.weight"), y, ctx->threads));
                continue;
            }
            floats norm = normalized(hidden, H, ctx->get(b + "attn_norm.weight"), nullptr, true, ctx->threads);
            auto q = linear(ctx->get(b + "attn_q.weight"), norm, ctx->threads);
            auto k = linear(ctx->get(b + "attn_k.weight"), norm, ctx->threads);
            auto v = linear(ctx->get(b + "attn_v.weight"), norm, ctx->threads);
            floats qr(size_t(HEADS) * S * D), kr(qr.size()), vr(qr.size());
            for (int head = 0; head < HEADS; ++head) {
                const int kvh = head / GROUP; // repeat_interleave of KV heads
                for (int t = 0; t < S; ++t)
                    for (int i = 0; i < D / 2; ++i) {
                        const float angle = float(t) / std::pow(THETA, float(2 * i) / D);
                        const float co = std::cos(angle), si = std::sin(angle);
                        const size_t a = (size_t(t) * HEADS + head) * D + i, z = (size_t(head) * S + t) * D + i;
                        const size_t ka = (size_t(t) * KVH + kvh) * D + i;
                        qr[z] = q[a] * co - q[a + D / 2] * si;
                        qr[z + D / 2] = q[a + D / 2] * co + q[a] * si;
                        kr[z] = k[ka] * co - k[ka + D / 2] * si;
                        kr[z + D / 2] = k[ka + D / 2] * co + k[ka] * si;
                        vr[z] = v[ka];
                        vr[z + D / 2] = v[ka + D / 2];
                    }
            }
            floats heads(qr.size()), prob(S);
            for (int head = 0; head < HEADS; ++head)
                for (int t = 0; t < S; ++t) {
                    for (int j = 0; j < S; ++j) {
                        float dot = 0;
                        for (int i = 0; i < D; ++i)
                            dot += qr[(size_t(head) * S + t) * D + i] * kr[(size_t(head) * S + j) * D + i];
                        prob[j] = mask[j] ? dot / std::sqrt(float(D)) : -std::numeric_limits<float>::infinity();
                    }
                    softmax(prob.data(), S);
                    for (int i = 0; i < D; ++i) {
                        float sum = 0;
                        for (int j = 0; j < S; ++j) sum += prob[j] * vr[(size_t(head) * S + j) * D + i];
                        heads[(size_t(head) * S + t) * D + i] = sum;
                    }
                }
            floats merged(size_t(S) * H);
            for (int t = 0; t < S; ++t)
                for (int head = 0; head < HEADS; ++head)
                    std::copy_n(heads.data() + (size_t(head) * S + t) * D, D,
                                merged.data() + (size_t(t) * HEADS + head) * D);
            add(hidden, linear(ctx->get(b + "attn_output.weight"), merged, ctx->threads));
            norm = normalized(hidden, H, ctx->get(b + "ffn_norm.weight"), nullptr, true, ctx->threads);
            auto probs = linear(ctx->get(b + "ffn_gate_inp.weight"), norm, ctx->threads);
            std::vector<int> chosen(size_t(S) * TOP);
            floats weight(size_t(S) * TOP);
            for (int t = 0; t < S; ++t) {
                float * pr = probs.data() + size_t(t) * E;
                softmax(pr, E);
                std::vector<int> order(E);
                std::iota(order.begin(), order.end(), 0);
                std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return pr[x] > pr[y]; });
                require(pr[order[TOP - 1]] > pr[order[TOP]], "Exact router tie is unqualified");
                float denom = ROUTER_EPS;
                for (int i = 0; i < TOP; ++i) denom += pr[order[i]];
                for (int i = 0; i < TOP; ++i) {
                    chosen[size_t(t) * TOP + i] = order[i];
                    weight[size_t(t) * TOP + i] = pr[order[i]] / denom;
                }
            }
            floats routed(size_t(S) * H, 0);
            const int X = p.expert_dim;
            for (int expert = 0; expert < E; ++expert) {
                auto up = linear(ctx->get(b + "ffn_up_exps.weight"), norm, ctx->threads, expert);
                auto gate = linear(ctx->get(b + "ffn_gate_exps.weight"), norm, ctx->threads, expert);
                auto act = gelu_product(gate, up);
                for (int t = 0; t < S; ++t) {
                    float w = 0;
                    for (int j = 0; j < TOP; ++j)
                        if (chosen[size_t(t) * TOP + j] == expert) w = weight[size_t(t) * TOP + j];
                    for (int j = 0; j < X; ++j) act[size_t(t) * X + j] *= w;
                }
                add(routed, linear(ctx->get(b + "ffn_down_exps.weight"), act, ctx->threads, expert));
            }
            auto su = linear(ctx->get(b + "ffn_up_shexp.weight"), norm, ctx->threads);
            auto sg = linear(ctx->get(b + "ffn_gate_shexp.weight"), norm, ctx->threads);
            add(routed, linear(ctx->get(b + "ffn_down_shexp.weight"), gelu_product(sg, su), ctx->threads));
            add(hidden, routed);
        }
        hidden = normalized(hidden, H, ctx->get("output_norm.weight"), nullptr, true, ctx->threads);
        hidden = linear(ctx->get("dense_head.weight"), hidden, ctx->threads);
        for (int t = 0; t < S; ++t) {
            float sum = 0;
            for (int i = 0; i < OUT; ++i) sum += hidden[size_t(t) * OUT + i] * hidden[size_t(t) * OUT + i];
            const float d = std::max(std::sqrt(sum), 1e-12f);
            for (int i = 0; i < OUT; ++i) hidden[size_t(t) * OUT + i] /= d;
        }
        vectors = std::move(hidden);
        return true;
    } catch (const std::exception & e) {
        ctx->failure = e.what();
        vectors.clear();
        return false;
    }
}
} // namespace moedern_family
