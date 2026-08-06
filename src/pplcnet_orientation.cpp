#include "pplcnet_orientation.h"

#include "core/cpu_ops.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>

extern "C" {
unsigned char * stbi_load(const char *, int *, int *, int *, int);
void stbi_image_free(void *);
}

namespace pplcnet_orientation {

using core_cpu::conv2d_cpu;
using core_cpu::linear_cpu;
using core_cpu::to_f32;

struct conv {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
    int in = 0, out = 0, k = 1, stride = 1, groups = 1;
};

struct bn {
    ggml_tensor * mean = nullptr;
    ggml_tensor * var = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * bias = nullptr;
};

struct block {
    conv dw, pw;
    bn dw_bn, pw_bn;
    bool se = false;
    conv se1, se2;
    bool residual = false;
};

struct context {
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    int n_threads = 1;
    ggml_backend_sched_t graph_sched = nullptr;
    conv stem, head;
    bn stem_bn;
    std::vector<block> blocks;
    ggml_tensor * fc_w = nullptr;
    ggml_tensor * fc_b = nullptr;
    bool use_graph = false;
    std::vector<uint8_t> graph_meta;
    ggml_context * graph_ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * graph_input = nullptr;
    ggml_tensor * graph_stem_conv = nullptr;
    ggml_tensor * graph_stem = nullptr;
    ggml_tensor * graph_logits = nullptr;
    ggml_tensor * graph_pooled = nullptr;
    ggml_tensor * graph_head = nullptr;
    std::vector<ggml_tensor *> graph_taps;
};

static ggml_tensor * prefix(const core_gguf::tensor_map & m, const std::string & p) {
    for (const auto & it : m)
        if (it.first.rfind(p, 0) == 0) return it.second;
    return nullptr;
}

static conv make_conv(const core_gguf::tensor_map & m, int id, int in, int out, int k, int stride, int groups) {
    const std::string p = "ori.conv2d_" + std::to_string(id) + ".";
    conv c;
    c.w = prefix(m, p + "w_0");
    c.b = prefix(m, p + "b_0");
    c.in = in;
    c.out = out;
    c.k = k;
    c.stride = stride;
    c.groups = groups;
    return c;
}

static bn make_bn(const core_gguf::tensor_map & m, int id) {
    const std::string p = "ori.batch_norm2d_" + std::to_string(id) + ".";
    // Paddle's BatchNorm inputs are mean, variance, scale, bias.  The PIR
    // variable names are w_1, w_2, w_0, b_0 respectively.
    return { prefix(m, p + "w_1"), prefix(m, p + "w_2"), prefix(m, p + "w_0"), prefix(m, p + "b_0") };
}

static int bn_id_for_conv(int id) {
    if (id <= 23) return id;
    if (id == 26) return 24;
    if (id == 27) return 25;
    if (id == 30) return 26;
    return -1;
}

static bool apply_conv(const conv & c, const std::vector<float> & x, int h, int w, std::vector<float> & y, int & oh,
                       int & ow) {
    if (!c.w) return false;
    oh = (h + c.k - 1 - c.k) / c.stride + 1 + (c.k - 1) / c.stride; // replaced below for clarity
    const int pad = c.k / 2;
    oh = (h + 2 * pad - c.k) / c.stride + 1;
    ow = (w + 2 * pad - c.k) / c.stride + 1;
    y.assign((size_t)c.out * oh * ow, 0.0f);
    auto weights = to_f32(c.w);
    auto bias = to_f32(c.b);
    conv2d_cpu(x.data(), y.data(), weights.data(), bias.empty() ? nullptr : bias.data(), c.in, c.out, h, w, c.k, c.k,
               c.stride, pad, c.groups);
    return true;
}

static void apply_bn(std::vector<float> & x, const bn & b, int channels) {
    if (!b.mean || !b.var || !b.scale || !b.bias) return;
    const auto mean = to_f32(b.mean), var = to_f32(b.var), scale = to_f32(b.scale), bias = to_f32(b.bias);
    const size_t spatial = x.size() / channels;
    for (int c = 0; c < channels; ++c) {
        const float inv = scale[c] / std::sqrt(var[c] + 1e-5f);
        for (size_t i = 0; i < spatial; ++i)
            x[(size_t)c * spatial + i] = (x[(size_t)c * spatial + i] - mean[c]) * inv + bias[c];
    }
}

static void hardswish(std::vector<float> & x) {
    for (float & v : x) v *= std::clamp(v + 3.0f, 0.0f, 6.0f) / 6.0f;
}

static void relu(std::vector<float> & x) {
    for (float & v : x) v = std::max(0.0f, v);
}

static bool run_conv_bn(const conv & c, const bn & b, std::vector<float> & x, int & h, int & w) {
    std::vector<float> y;
    int oh = 0, ow = 0;
    if (!apply_conv(c, x, h, w, y, oh, ow)) return false;
    static int debug_first_conv = 0;
    if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG") && debug_first_conv++ == 0)
        fprintf(stderr, "pplcnet cpu conv %.7g %.7g %.7g %.7g\n", y[0], y[1], y[2], y[3]);
    apply_bn(y, b, c.out);
    h = oh;
    w = ow;
    hardswish(y);
    x.swap(y);
    return true;
}

static bool run_block(const block & b, std::vector<float> & x, int & h, int & w) {
    std::vector<float> input = x;
    if (!run_conv_bn(b.dw, b.dw_bn, x, h, w)) return false;
    if (b.se) {
        const int channels = b.dw.out;
        std::vector<float> pooled(channels, 0.0f);
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < h * w; ++i) pooled[c] += x[(size_t)c * h * w + i] / float(h * w);
        std::vector<float> gate;
        int gh = 0, gw = 0;
        if (!apply_conv(b.se1, pooled, 1, 1, gate, gh, gw)) return false;
        relu(gate);
        if (!apply_conv(b.se2, gate, 1, 1, pooled, gh, gw)) return false;
        for (float & v : pooled) v = std::clamp((v + 3.0f) / 6.0f, 0.0f, 1.0f);
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < h * w; ++i) x[(size_t)c * h * w + i] *= pooled[c];
    }
    if (!run_conv_bn(b.pw, b.pw_bn, x, h, w)) return false;
    if (b.residual && input.size() == x.size())
        for (size_t i = 0; i < x.size(); ++i) x[i] += input[i];
    return true;
}

static std::vector<float> preprocess(const uint8_t * px, int width, int height, int channels) {
    constexpr int H = 80, W = 160;
    std::vector<float> out((size_t)3 * H * W);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int sx = std::min(width - 1, std::max(0, (int)std::floor((x + 0.5f) * width / float(W))));
            const int sy = std::min(height - 1, std::max(0, (int)std::floor((y + 0.5f) * height / float(H))));
            const uint8_t * p = px + ((size_t)sy * width + sx) * channels;
            constexpr float mean[3] = { 0.485f, 0.456f, 0.406f };
            constexpr float stdev[3] = { 0.229f, 0.224f, 0.225f };
            for (int c = 0; c < 3; ++c)
                out[(size_t)c * H * W + y * W + x] = (p[std::min(c, channels - 1)] / 255.0f - mean[c]) / stdev[c];
        }
    return out;
}

static ggml_tensor * graph_conv(ggml_context * g, ggml_tensor * x, const conv & c) {
    if (!c.w) return nullptr;
    if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG"))
        fprintf(stderr, "pplcnet graph weight ne=%lld,%lld,%lld,%lld type=%d conv=%d->%d k=%d groups=%d\n",
                (long long)c.w->ne[0], (long long)c.w->ne[1], (long long)c.w->ne[2], (long long)c.w->ne[3],
                (int)c.w->type, c.in, c.out, c.k, c.groups);
    // The converter already stores convolution weights in ggml's canonical
    // [KW, KH, IC/G, OC] layout. Do not transpose them again here.
    ggml_tensor * w = c.w;
    if (w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);
    x = c.groups == c.in ? ggml_conv_2d_dw(g, w, x, c.stride, c.stride, c.k / 2, c.k / 2, 1, 1)
                         : ggml_conv_2d(g, w, x, c.stride, c.stride, c.k / 2, c.k / 2, 1, 1);
    if (c.b) x = ggml_add(g, x, ggml_reshape_3d(g, c.b, 1, 1, c.out));
    return x;
}

static ggml_tensor * graph_bn(ggml_context * g, ggml_tensor * x, const bn & b, int channels) {
    if (!b.mean || !b.var || !b.scale || !b.bias) return x;
    auto vec = [&](ggml_tensor * t) { return ggml_reshape_3d(g, t, 1, 1, channels); };
    if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG"))
        fprintf(stderr, "pplcnet bn x=%lld,%lld,%lld,%lld c=%d\n", (long long)x->ne[0], (long long)x->ne[1],
                (long long)x->ne[2], (long long)x->ne[3], channels);
    ggml_tensor * variance = ggml_scale_bias(g, vec(b.var), 1.0f, 1e-5f);
    ggml_tensor * inv = ggml_div(g, vec(b.scale), ggml_sqrt(g, variance));
    return ggml_add(g, ggml_mul(g, ggml_sub(g, x, vec(b.mean)), inv), vec(b.bias));
}

static ggml_tensor * graph_act(ggml_context * g, ggml_tensor * x) {
    return ggml_hardswish(g, x);
}

static ggml_tensor * graph_conv_bn(ggml_context * g, ggml_tensor * x, const conv & c, const bn & b) {
    x = graph_conv(g, x, c);
    return x ? graph_act(g, graph_bn(g, x, b, c.out)) : nullptr;
}

static ggml_tensor * graph_block(ggml_context * g, ggml_tensor * x, const block & b) {
    ggml_tensor * identity = x;
    x = graph_conv_bn(g, x, b.dw, b.dw_bn);
    if (!x) return nullptr;
    if (b.se) {
        const int spatial = (int)(x->ne[0] * x->ne[1]);
        ggml_tensor * pooled =
            ggml_scale(g, ggml_sum_rows(g, ggml_reshape_2d(g, x, spatial, b.dw.out)), 1.0f / float(spatial));
        pooled = ggml_reshape_3d(g, pooled, 1, 1, b.dw.out);
        pooled = graph_conv(g, pooled, b.se1);
        if (!pooled) return nullptr;
        pooled = ggml_relu(g, pooled);
        pooled = graph_conv(g, pooled, b.se2);
        if (!pooled) return nullptr;
        pooled = ggml_clamp(g, ggml_scale_bias(g, pooled, 1.0f / 6.0f, 0.5f), 0.0f, 1.0f);
        x = ggml_mul(g, x, pooled);
    }
    x = graph_conv_bn(g, x, b.pw, b.pw_bn);
    if (!x) return nullptr;
    return b.residual ? ggml_add(g, x, identity) : x;
}

static std::vector<float> preprocess_graph(const uint8_t * px, int width, int height, int channels) {
    constexpr int H = 80, W = 160;
    // ggml's [W,H,C] tensor stores each channel plane contiguously (CHW).
    std::vector<float> out((size_t)W * H * 3);
    constexpr float mean[3] = { 0.485f, 0.456f, 0.406f };
    constexpr float stdev[3] = { 0.229f, 0.224f, 0.225f };
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int sx = std::min(width - 1, std::max(0, (int)std::floor((x + 0.5f) * width / float(W))));
            const int sy = std::min(height - 1, std::max(0, (int)std::floor((y + 0.5f) * height / float(H))));
            const uint8_t * p = px + ((size_t)sy * width + sx) * channels;
            for (int c = 0; c < 3; ++c)
                out[(size_t)c * W * H + y * W + x] = (p[std::min(c, channels - 1)] / 255.0f - mean[c]) / stdev[c];
        }
    return out;
}

static bool build_graph(context * c) {
    c->graph_meta.resize(8 * 1024 * 1024);
    ggml_init_params params{ c->graph_meta.size(), c->graph_meta.data(), true };
    c->graph_ctx = ggml_init(params);
    if (!c->graph_ctx) return false;
    ggml_context * g = c->graph_ctx;
    // Convolution graphs use the canonical ggml image layout [W,H,C,N].
    c->graph_input = ggml_new_tensor_4d(g, GGML_TYPE_F32, 160, 80, 3, 1);
    ggml_set_name(c->graph_input, "pplcnet_input");
    ggml_set_input(c->graph_input);
    ggml_tensor * stem_conv = graph_conv(g, c->graph_input, c->stem);
    c->graph_stem_conv = stem_conv;
    ggml_tensor * x = stem_conv ? graph_act(g, graph_bn(g, stem_conv, c->stem_bn, c->stem.out)) : nullptr;
    if (!x) return false;
    c->graph_stem = x;
    if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG")) {
        ggml_set_output(c->graph_stem_conv);
        ggml_set_output(c->graph_stem);
    }
    for (const auto & b : c->blocks) {
        x = graph_block(g, x, b);
        if (!x) return false;
        c->graph_taps.push_back(x);
        if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG")) ggml_set_output(x);
    }
    const int spatial = (int)(x->ne[0] * x->ne[1]);
    x = ggml_scale(g, ggml_sum_rows(g, ggml_reshape_2d(g, x, spatial, 512)), 1.0f / float(spatial));
    c->graph_pooled = x;
    x = ggml_reshape_3d(g, x, 1, 1, 512);
    x = graph_conv(g, x, c->head);
    if (!x) return false;
    x = graph_act(g, x);
    c->graph_head = x;
    if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG")) {
        ggml_set_output(c->graph_pooled);
        ggml_set_output(c->graph_head);
    }
    const int head_spatial = (int)(x->ne[0] * x->ne[1]);
    x = ggml_scale(g, ggml_sum_rows(g, ggml_reshape_2d(g, x, head_spatial, 1280)), 1.0f / float(head_spatial));
    x = ggml_reshape_1d(g, x, 1280);
    // The converted GGUF tensor is already laid out contiguously for the
    // [in,out] matmul view; reshape the metadata without transposing values.
    c->graph_logits = ggml_add(g, ggml_mul_mat(g, ggml_reshape_2d(g, c->fc_w, 1280, 2), x), c->fc_b);
    if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG"))
        fprintf(stderr, "pplcnet graph fc ne=%lld,%lld type=%d\n", (long long)c->fc_w->ne[0], (long long)c->fc_w->ne[1],
                (int)c->fc_w->type);
    ggml_set_name(c->graph_logits, "pplcnet_logits");
    ggml_set_output(c->graph_logits);
    c->graph = ggml_new_graph_custom(g, 4096, false);
    ggml_build_forward_expand(c->graph, c->graph_logits);
    if (ggml_backend_is_cpu(c->backend)) {
        ggml_backend_t backends[] = { c->backend };
        c->graph_sched = ggml_backend_sched_new(backends, nullptr, 1, 4096, false, false);
    } else {
        c->cpu_backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(c->cpu_backend, std::max(1, c->n_threads));
        ggml_backend_t backends[] = { c->backend, c->cpu_backend };
        c->graph_sched = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, false);
    }
    if (!c->graph_sched) return false;
    ggml_backend_sched_reset(c->graph_sched);
    return ggml_backend_sched_alloc_graph(c->graph_sched, c->graph);
}

context * init(const char * path, int n_threads) {
    auto * c = new context();
    c->n_threads = std::max(1, n_threads);
    c->use_graph = std::getenv("PPLCNET_ORIENTATION_GRAPH") != nullptr &&
                   std::getenv("PPLCNET_ORIENTATION_GRAPH_PIPELINE") != nullptr;
    const bool graph_cpu = std::getenv("PPLCNET_ORIENTATION_GRAPH_CPU") != nullptr;
    c->backend = c->use_graph && !graph_cpu ? crispasr_init_gpu_backend() : ggml_backend_cpu_init();
    // Same ignored-n_threads bug class as ppocrv6_det/_ocr (O13b): the thread
    // parameter was declared anonymously and never applied.
    if (c->backend && ggml_backend_is_cpu(c->backend)) ggml_backend_cpu_set_n_threads(c->backend, c->n_threads);
    if (!core_gguf::load_weights(path, c->backend, "pplcnet_orientation", c->wl)) {
        free(c);
        return nullptr;
    }
    const auto & m = c->wl.tensors;
    c->stem = make_conv(m, 0, 3, 16, 3, 2, 1);
    c->stem_bn = make_bn(m, 0);
    struct spec {
        int dw, pw, in, out, k, stride;
        bool se;
        int se1, se2;
    };
    const spec specs[] = {
        { 1, 2, 16, 32, 3, 1, false, 0, 0 },      { 3, 4, 32, 64, 3, 2, false, 0, 0 },
        { 5, 6, 64, 64, 3, 1, false, 0, 0 },      { 7, 8, 64, 128, 3, 2, false, 0, 0 },
        { 9, 10, 128, 128, 3, 1, false, 0, 0 },   { 11, 12, 128, 256, 3, 2, false, 0, 0 },
        { 13, 14, 256, 256, 5, 1, false, 0, 0 },  { 15, 16, 256, 256, 5, 1, false, 0, 0 },
        { 17, 18, 256, 256, 5, 1, false, 0, 0 },  { 19, 20, 256, 256, 5, 1, false, 0, 0 },
        { 21, 22, 256, 256, 5, 1, false, 0, 0 },  { 23, 26, 256, 512, 5, 2, true, 24, 25 },
        { 27, 30, 512, 512, 5, 1, true, 28, 29 },
    };
    for (const auto & s : specs) {
        block b;
        b.dw = make_conv(m, s.dw, s.in, s.in, s.k, s.stride, s.in);
        b.dw_bn = make_bn(m, bn_id_for_conv(s.dw));
        b.pw = make_conv(m, s.pw, s.in, s.out, 1, 1, 1);
        b.pw_bn = make_bn(m, bn_id_for_conv(s.pw));
        b.se = s.se;
        b.residual = s.in == s.out && s.stride == 1;
        if (b.se) {
            b.se1 = make_conv(m, s.se1, s.in, s.in / 4, 1, 1, 1);
            b.se2 = make_conv(m, s.se2, s.in / 4, s.in, 1, 1, 1);
        }
        c->blocks.push_back(b);
    }
    c->head = make_conv(m, 31, 512, 1280, 1, 1, 1);
    c->fc_w = prefix(m, "ori.linear_0.w_0");
    c->fc_b = prefix(m, "ori.linear_0.b_0");
    if (c->use_graph && !build_graph(c)) {
        if (c->graph_sched) ggml_backend_sched_free(c->graph_sched);
        c->graph_sched = nullptr;
        if (c->cpu_backend) ggml_backend_free(c->cpu_backend);
        c->cpu_backend = nullptr;
        if (c->graph_ctx) ggml_free(c->graph_ctx);
        c->graph_ctx = nullptr;
        c->use_graph = false;
        fprintf(stderr, "pplcnet-orientation: graph build failed; using CPU reference\n");
    }
    return c;
}

void free(context * c) {
    if (!c) return;
    if (c->graph_sched) ggml_backend_sched_free(c->graph_sched);
    if (c->cpu_backend) ggml_backend_free(c->cpu_backend);
    if (c->graph_ctx) ggml_free(c->graph_ctx);
    core_gguf::free_weights(c->wl);
    if (c->backend) ggml_backend_free(c->backend);
    delete c;
}

result classify_raw(context * c, const uint8_t * px, int width, int height, int channels) {
    result r;
    if (!c || !px || width <= 0 || height <= 0 || channels <= 0) return r;
    if (c->use_graph) {
        auto input = preprocess_graph(px, width, height, channels);
        if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG"))
            fprintf(stderr, "pplcnet graph input %.7g %.7g %.7g %.7g\n", input[0], input[1], input[2], input[3]);
        ggml_backend_tensor_set(c->graph_input, input.data(), 0, input.size() * sizeof(float));
        // Metal's scheduler on pre-tensor Apple GPUs does not reliably reuse
        // a mixed CPU/Metal allocation across repeated depthwise executions.
        // Reallocate the static graph per crop until backend reuse is proven.
        ggml_backend_sched_reset(c->graph_sched);
        if (!ggml_backend_sched_alloc_graph(c->graph_sched, c->graph)) return r;
        if (!c->graph_sched || ggml_backend_sched_graph_compute(c->graph_sched, c->graph) != GGML_STATUS_SUCCESS)
            return r;
        if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG")) {
            float tap[4] = {};
            float conv_tap[4] = {};
            ggml_backend_tensor_get(c->graph_stem_conv, conv_tap, 0, sizeof(conv_tap));
            ggml_backend_tensor_get(c->graph_stem, tap, 0, sizeof(tap));
            fprintf(stderr, "pplcnet graph conv %.7g %.7g %.7g %.7g\n", conv_tap[0], conv_tap[1], conv_tap[2],
                    conv_tap[3]);
            fprintf(stderr, "pplcnet graph stem %.7g %.7g %.7g %.7g\n", tap[0], tap[1], tap[2], tap[3]);
            for (size_t i = 0; i < c->graph_taps.size(); ++i) {
                float block_tap[4] = {};
                ggml_backend_tensor_get(c->graph_taps[i], block_tap, 0, sizeof(block_tap));
                fprintf(stderr, "pplcnet graph block%zu %.7g %.7g %.7g %.7g\n", i, block_tap[0], block_tap[1],
                        block_tap[2], block_tap[3]);
            }
            float pooled_tap[4] = {}, head_tap[4] = {};
            ggml_backend_tensor_get(c->graph_pooled, pooled_tap, 0, sizeof(pooled_tap));
            ggml_backend_tensor_get(c->graph_head, head_tap, 0, sizeof(head_tap));
            fprintf(stderr, "pplcnet graph pooled %.7g %.7g %.7g %.7g head %.7g %.7g %.7g %.7g\n", pooled_tap[0],
                    pooled_tap[1], pooled_tap[2], pooled_tap[3], head_tap[0], head_tap[1], head_tap[2], head_tap[3]);
        }
        float logits[2] = {};
        ggml_backend_tensor_get(c->graph_logits, logits, 0, sizeof(logits));
        r.logits[0] = logits[0];
        r.logits[1] = logits[1];
        const float mx = std::max(logits[0], logits[1]);
        const float e0 = std::exp(logits[0] - mx), e1 = std::exp(logits[1] - mx);
        const float z = e0 + e1;
        r.probabilities[0] = e0 / z;
        r.probabilities[1] = e1 / z;
        r.angle = r.probabilities[1] > r.probabilities[0] ? 180 : 0;
        r.confidence = std::max(r.probabilities[0], r.probabilities[1]);
        const result graph_result = r;
        if (std::getenv("PPLCNET_ORIENTATION_GRAPH_ACCEPT")) return graph_result;
        // The graph is diagnostic-only until its logits agree with the CPU
        // reference. This also makes unsupported/approximate backend kernels
        // harmless to the production pipeline.
        int h = 80, w = 160;
        auto x = preprocess(px, width, height, channels);
        std::vector<float> y;
        if (!run_conv_bn(c->stem, c->stem_bn, x, h, w)) return {};
        for (size_t bi = 0; bi < c->blocks.size(); ++bi) {
            if (!run_block(c->blocks[bi], x, h, w)) return {};
            if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG"))
                fprintf(stderr, "pplcnet cpu block%zu %.7g %.7g %.7g %.7g\n", bi, x[0], x[1], x[2], x[3]);
        }
        std::vector<float> pooled(512, 0.0f);
        for (int ch = 0; ch < 512; ++ch)
            for (int i = 0; i < h * w; ++i) pooled[ch] += x[(size_t)ch * h * w + i] / float(h * w);
        if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG"))
            fprintf(stderr, "pplcnet cpu pooled %.7g %.7g %.7g %.7g\n", pooled[0], pooled[1], pooled[2], pooled[3]);
        int oh = 0, ow = 0;
        if (!apply_conv(c->head, pooled, 1, 1, y, oh, ow)) return {};
        hardswish(y);
        if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG"))
            fprintf(stderr, "pplcnet cpu head %.7g %.7g %.7g %.7g\n", y[0], y[1], y[2], y[3]);
        auto fw = to_f32(c->fc_w), fb = to_f32(c->fc_b);
        float cpu_logits[2] = {};
        linear_cpu(y.data(), cpu_logits, 1280, 2, fw.data(), fb.data());
        if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG"))
            fprintf(stderr, "pplcnet graph diagnostic logits graph=(%.9g,%.9g) cpu=(%.9g,%.9g)\n",
                    graph_result.logits[0], graph_result.logits[1], cpu_logits[0], cpu_logits[1]);
        const float cpu_mx = std::max(cpu_logits[0], cpu_logits[1]);
        const float ce0 = std::exp(cpu_logits[0] - cpu_mx), ce1 = std::exp(cpu_logits[1] - cpu_mx), cz = ce0 + ce1;
        r.logits[0] = cpu_logits[0];
        r.logits[1] = cpu_logits[1];
        r.probabilities[0] = ce0 / cz;
        r.probabilities[1] = ce1 / cz;
        r.angle = r.probabilities[1] > r.probabilities[0] ? 180 : 0;
        r.confidence = std::max(r.probabilities[0], r.probabilities[1]);
        return r;
    }
    auto x = preprocess(px, width, height, channels);
    if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG"))
        fprintf(stderr, "pplcnet cpu input %.7g %.7g %.7g %.7g\n", x[0], x[1], x[2], x[3]);
    int h = 80, w = 160;
    if (!run_conv_bn(c->stem, c->stem_bn, x, h, w)) return r;
    if (std::getenv("PPLCNET_ORIENTATION_GRAPH_DEBUG"))
        fprintf(stderr, "pplcnet cpu stem %.7g %.7g %.7g %.7g\n", x[0], x[1], x[2], x[3]);
    for (const auto & b : c->blocks)
        if (!run_block(b, x, h, w)) return r;
    std::vector<float> pooled(512, 0.0f);
    for (int ch = 0; ch < 512; ++ch)
        for (int i = 0; i < h * w; ++i) pooled[ch] += x[(size_t)ch * h * w + i] / float(h * w);
    std::vector<float> y;
    int oh = 0, ow = 0;
    if (!apply_conv(c->head, pooled, 1, 1, y, oh, ow)) return r;
    hardswish(y);
    auto fw = to_f32(c->fc_w), fb = to_f32(c->fc_b);
    float logits[2] = {};
    linear_cpu(y.data(), logits, 1280, 2, fw.data(), fb.data());
    r.logits[0] = logits[0];
    r.logits[1] = logits[1];
    const float mx = std::max(logits[0], logits[1]);
    const float e0 = std::exp(logits[0] - mx), e1 = std::exp(logits[1] - mx);
    const float z = e0 + e1;
    r.probabilities[0] = e0 / z;
    r.probabilities[1] = e1 / z;
    r.angle = r.probabilities[1] > r.probabilities[0] ? 180 : 0;
    r.confidence = std::max(r.probabilities[0], r.probabilities[1]);
    return r;
}

result classify_file(context * c, const char * path) {
    int w = 0, h = 0, ch = 0;
    auto * px = stbi_load(path, &w, &h, &ch, 3);
    if (!px) return {};
    auto r = classify_raw(c, px, w, h, 3);
    stbi_image_free(px);
    return r;
}

} // namespace pplcnet_orientation
