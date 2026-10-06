// sam2.cpp — SAM 2.1 image segmentation in ggml.
//
// Blueprint: facebookresearch/sam2 (sam2/modeling/backbones/hieradet.py, image_encoder.py,
// sam/prompt_encoder.py, sam/mask_decoder.py, sam/transformer.py) in image mode, as
// SAM2ImagePredictor runs it. All feature maps are kept channel-last ([C, W, H] in ggml ne
// order), which is torch's (H, W, C) memory order, so that 1x1 convolutions are matrix products
// and window partitioning is a reshape and a permute.
//
// Encoder (one graph per image): 7x7/4 patch embedding plus a precomputed position embedding;
// Hiera blocks (pre-norm, windowed or global multi-head attention, query max-pooling at stage
// changes, GELU MLP); FPN neck (1x1 lateral convolutions, nearest top-down on the two coarsest
// used levels); conv_s0 / conv_s1 for the decoder's high-resolution features; no_mem_embed.
// Decoder (one graph per prompt set): prompt embedding on the host (random Fourier features),
// two-way transformer, transposed-convolution upscaling, hypernetwork MLPs, IoU head.

#include "sam2.h"

#include "core/cpu_ops.h"
#include "core/env_gate.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct hparams {
    int image_size = 1024;
    int embed_dim = 96;
    int num_heads = 1;
    std::vector<int> stages{ 1, 2, 7, 2 };
    std::vector<int> global_att_blocks{ 5, 7, 9 };
    std::vector<int> window_spec{ 8, 4, 14, 7 };
    int q_pool = 3;
    int neck_dim = 256;
    std::vector<int> fpn_top_down_levels{ 2, 3 };
    int decoder_depth = 2;
    int decoder_heads = 8;
    int num_mask_tokens = 4;
};

// Per-block layout derived as hieradet.Hiera.__init__ does it.
struct block_spec {
    int dim, dim_out, heads, window;
    bool q_pool;
};

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

} // namespace

struct sam2_context {
    hparams hp;
    std::vector<block_spec> blocks;
    std::vector<int> stage_ends;
    int n_threads = 4;
    bool bench = false;
    bool probing = false;

    ggml_backend_t backend = nullptr;
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    std::vector<uint8_t> meta;
    core_gguf::WeightLoad wl;

    // Constants computed at load: the trunk's position embedding [C, 256, 256] and the decoder's
    // dense position encoding [256, 64 * 64], both on the backend.
    ggml_context * const_ctx = nullptr;
    ggml_backend_buffer_t const_buf = nullptr;
    ggml_tensor * pos_full = nullptr;
    ggml_tensor * dense_pe = nullptr;

    // Host copies of small prompt-encoder weights.
    std::vector<float> gauss;                // [2][128]
    std::vector<float> point_embed[4];       // [256] each
    std::vector<float> not_a_point, no_mask; // [256]
    std::vector<float> out_tokens;           // obj, iou, mask0..3: [6][256]

    // Encoder outputs of the current image (channel-last).
    bool has_image = false;
    std::vector<float> image_embed, high_res_0, high_res_1;
    std::map<std::string, std::vector<float>> probes;
    std::string backend_name;

    ggml_tensor * w(const std::string & name) const {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) fprintf(stderr, "sam2: missing tensor %s\n", name.c_str());
        return t;
    }
};

// ── graph helpers ─────────────────────────────────────────────────────────

static ggml_tensor * linear(ggml_context * g, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias) {
    ggml_tensor * y = ggml_mul_mat(g, weight, x);
    return bias ? ggml_add(g, y, bias) : y;
}

static ggml_tensor * layer_norm(ggml_context * g, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias,
                                float eps) {
    x = ggml_norm(g, x, eps);
    return ggml_add(g, ggml_mul(g, x, weight), bias);
}

// [C, W, H] -> windows [C, ws*ws, nW*nH] (zero padding at the end, as window_partition).
static ggml_tensor * window_partition(ggml_context * g, ggml_tensor * x, int ws, int & nw, int & nh) {
    const int64_t C = x->ne[0], W = x->ne[1], H = x->ne[2];
    const int pw = (int)((ws - W % ws) % ws), ph = (int)((ws - H % ws) % ws);
    if (pw || ph) x = ggml_pad(g, x, 0, pw, ph, 0);
    nw = (int)((W + pw) / ws);
    nh = (int)((H + ph) / ws);
    x = ggml_reshape_4d(g, x, C * ws, nw, ws, nh);
    x = ggml_cont(g, ggml_permute(g, x, 0, 2, 1, 3));
    return ggml_reshape_3d(g, x, C, ws * ws, nw * nh);
}

// Inverse of window_partition, cropped to [C, W, H].
static ggml_tensor * window_unpartition(ggml_context * g, ggml_tensor * x, int ws, int nw, int nh, int64_t W,
                                        int64_t H) {
    const int64_t C = x->ne[0];
    x = ggml_reshape_4d(g, x, C * ws, ws, nw, nh);
    x = ggml_cont(g, ggml_permute(g, x, 0, 2, 1, 3));
    x = ggml_reshape_3d(g, x, C, (int64_t)nw * ws, (int64_t)nh * ws);
    if (nw * ws != W || nh * ws != H) x = ggml_cont(g, ggml_view_3d(g, x, C, W, H, x->nb[1], x->nb[2], 0));
    return x;
}

// 2x2 max pooling, stride 2, of channel-last maps [C, w, h, B].
static ggml_tensor * max_pool_2x2(ggml_context * g, ggml_tensor * x) {
    ggml_tensor * t = ggml_cont(g, ggml_permute(g, x, 2, 0, 1, 3)); // [w, h, C, B]
    t = ggml_pool_2d(g, t, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
    return ggml_cont(g, ggml_permute(g, t, 1, 2, 0, 3)); // [C, w/2, h/2, B]
}

// Scaled dot-product attention of q [hd, Tq, heads, B] against k [hd, Tk, heads, B] and
// v given as [Tk, hd, heads, B]; returns [hd*heads, Tq, B].
static ggml_tensor * attention_core(ggml_context * g, ggml_tensor * q, ggml_tensor * k, ggml_tensor * vt, int hd) {
    ggml_tensor * kq = ggml_mul_mat(g, k, q); // [Tk, Tq, heads, B]
    kq = ggml_soft_max_ext(g, kq, nullptr, 1.0f / std::sqrt((float)hd), 0.0f);
    ggml_tensor * o = ggml_mul_mat(g, vt, kq);        // [hd, Tq, heads, B]
    o = ggml_cont(g, ggml_permute(g, o, 0, 2, 1, 3)); // [hd, heads, Tq, B]
    return ggml_reshape_3d(g, o, o->ne[0] * o->ne[1], o->ne[2], o->ne[3]);
}

// ── encoder ───────────────────────────────────────────────────────────────

static ggml_tensor * hiera_block(sam2_context * ctx, ggml_context * g, ggml_tensor * x, int index) {
    const block_spec & b = ctx->blocks[index];
    const std::string p = "hiera.blocks." + std::to_string(index) + ".";
    const int64_t W = x->ne[1], H = x->ne[2];
    ggml_tensor * xn = layer_norm(g, x, ctx->w(p + "norm1.weight"), ctx->w(p + "norm1.bias"), 1e-6f);

    ggml_tensor * shortcut = x;
    if (b.dim != b.dim_out) {
        shortcut = linear(g, xn, ctx->w(p + "proj.weight"), ctx->w(p + "proj.bias"));
        if (b.q_pool) shortcut = max_pool_2x2(g, shortcut);
    }
    const int64_t Wo = shortcut->ne[1], Ho = shortcut->ne[2];

    int nw = 1, nh = 1, side = 0;
    ggml_tensor * tokens;
    if (b.window > 0) {
        tokens = window_partition(g, xn, b.window, nw, nh);
        side = b.window;
    } else {
        tokens = ggml_reshape_3d(g, xn, b.dim, W * H, 1);
    }
    const int64_t T = tokens->ne[1], B = tokens->ne[2];
    const int hd = b.dim_out / b.heads;

    ggml_tensor * qkv = linear(g, tokens, ctx->w(p + "attn.qkv.weight"), ctx->w(p + "attn.qkv.bias")); // [3*dout, T, B]
    qkv = ggml_reshape_4d(g, qkv, hd, b.heads, 3, T * B);
    qkv = ggml_cont(g, ggml_permute(g, qkv, 0, 1, 3, 2)); // [hd, heads, T*B, 3]
    const size_t chunk = (size_t)hd * b.heads * T * B * sizeof(float);
    auto slice = [&](int n) { return ggml_view_3d(g, qkv, hd, b.heads, T * B, qkv->nb[1], qkv->nb[2], n * chunk); };
    ggml_tensor * q = slice(0);
    ggml_tensor * k = slice(1);
    ggml_tensor * v = slice(2);

    int64_t Tq = T;
    int side_out = side;
    if (b.q_pool) {
        // Max-pool the queries on their grid (per window, or the whole map for global attention).
        const int64_t gw = b.window > 0 ? side : W, gh = b.window > 0 ? side : H;
        q = ggml_reshape_4d(g, ggml_cont(g, q), (int64_t)hd * b.heads, gw, gh, B);
        q = max_pool_2x2(g, q);
        Tq = q->ne[1] * q->ne[2];
        side_out = side / 2;
    }
    q = ggml_cont(
        g, ggml_permute(g, ggml_reshape_4d(g, ggml_cont(g, q), hd, b.heads, Tq, B), 0, 2, 1, 3)); // [hd, Tq, heads, B]
    k = ggml_cont(
        g, ggml_permute(g, ggml_reshape_4d(g, ggml_cont(g, k), hd, b.heads, T, B), 0, 2, 1, 3)); // [hd, T, heads, B]
    v = ggml_cont(
        g, ggml_permute(g, ggml_reshape_4d(g, ggml_cont(g, v), hd, b.heads, T, B), 1, 2, 0, 3)); // [T, hd, heads, B]
    ggml_tensor * o = attention_core(g, q, k, v, hd);                                            // [dout, Tq, B]
    o = linear(g, o, ctx->w(p + "attn.proj.weight"), ctx->w(p + "attn.proj.bias"));

    if (b.window > 0) {
        o = window_unpartition(g, o, side_out, nw, nh, Wo, Ho);
    } else {
        o = ggml_reshape_3d(g, o, b.dim_out, Wo, Ho);
    }
    x = ggml_add(g, shortcut, o);
    ggml_tensor * m = layer_norm(g, x, ctx->w(p + "norm2.weight"), ctx->w(p + "norm2.bias"), 1e-6f);
    m = linear(g, m, ctx->w(p + "mlp.layers.0.weight"), ctx->w(p + "mlp.layers.0.bias"));
    m = ggml_gelu_erf(g, m);
    m = linear(g, m, ctx->w(p + "mlp.layers.1.weight"), ctx->w(p + "mlp.layers.1.bias"));
    return ggml_add(g, x, m);
}

static ggml_tensor * upsample_nearest_2x(ggml_context * g, ggml_tensor * x) {
    ggml_tensor * t = ggml_cont(g, ggml_permute(g, x, 2, 0, 1, 3)); // [W, H, C]
    t = ggml_upscale(g, t, 2, GGML_SCALE_MODE_NEAREST);
    return ggml_cont(g, ggml_permute(g, t, 1, 2, 0, 3));
}

static ggml_context * new_graph_context(sam2_context * ctx, int graph_size) {
    const size_t bytes =
        ggml_tensor_overhead() * (size_t)(graph_size + 64) + ggml_graph_overhead_custom(graph_size, false);
    ctx->meta.resize(bytes);
    ggml_init_params ip = { bytes, ctx->meta.data(), true };
    return ggml_init(ip);
}

static void read_tensor(ggml_tensor * t, std::vector<float> & out) {
    out.resize((size_t)ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
}

int sam2_set_image_f32(sam2_context * ctx, const float * chw) {
    if (!ctx || !chw) return -1;
    const double t0 = now_ms();
    const int S = ctx->hp.image_size;
    const int graph_size = 16384;
    ggml_context * g = new_graph_context(ctx, graph_size);
    ggml_cgraph * gf = ggml_new_graph_custom(g, graph_size, false);
    std::vector<std::pair<std::string, ggml_tensor *>> probe_list;
    auto probe = [&](const std::string & name, ggml_tensor * t) {
        if (ctx->probing) {
            ggml_set_output(t);
            probe_list.push_back({ name, t });
        }
    };

    ggml_tensor * image = ggml_new_tensor_4d(g, GGML_TYPE_F32, S, S, 3, 1);
    ggml_set_name(image, "image");
    ggml_set_input(image);

    ggml_tensor * x =
        ggml_conv_2d(g, ctx->w("hiera.patch_embed.proj.weight"), image, 4, 4, 3, 3, 1, 1); // [256,256,96,1]
    x = ggml_add(g, x, ggml_reshape_4d(g, ctx->w("hiera.patch_embed.proj.bias"), 1, 1, ctx->hp.embed_dim, 1));
    x = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3)); // [96, 256, 256, 1]
    x = ggml_reshape_3d(g, x, x->ne[0], x->ne[1], x->ne[2]);
    x = ggml_add(g, x, ctx->pos_full);
    probe("hiera.patch", x);

    std::vector<ggml_tensor *> stage_out;
    for (size_t i = 0; i < ctx->blocks.size(); i++) {
        x = hiera_block(ctx, g, x, (int)i);
        char name[32];
        snprintf(name, sizeof(name), "hiera.block.%02d", (int)i);
        probe(name, x);
        for (int end : ctx->stage_ends)
            if (end == (int)i) stage_out.push_back(x);
    }

    // FPN neck: convs[n - i] belongs to level i (finest first); top-down sum on the listed levels.
    const int n = (int)stage_out.size() - 1;
    std::vector<ggml_tensor *> level(stage_out.size(), nullptr);
    ggml_tensor * prev = nullptr;
    for (int i = n; i >= 0; i--) {
        const std::string c = "neck." + std::to_string(n - i) + ".";
        ggml_tensor * lateral = linear(g, stage_out[i], ctx->w(c + "weight"), ctx->w(c + "bias"));
        bool top_down = false;
        for (int l : ctx->hp.fpn_top_down_levels) top_down |= (l == i);
        prev = (top_down && prev) ? ggml_add(g, lateral, upsample_nearest_2x(g, prev)) : lateral;
        level[i] = prev;
    }
    for (int i = 0; i < n; i++) probe("neck." + std::to_string(i), level[i]); // scalp 1: the coarsest level is dropped

    ggml_tensor * high0 = linear(g, level[0], ctx->w("dec.conv_s0.weight"), ctx->w("dec.conv_s0.bias"));
    ggml_tensor * high1 = linear(g, level[1], ctx->w("dec.conv_s1.weight"), ctx->w("dec.conv_s1.bias"));
    ggml_tensor * embed = ggml_add(g, level[2], ctx->w("no_mem_embed"));
    for (auto * t : { high0, high1, embed }) ggml_set_output(t);
    ggml_build_forward_expand(gf, high0);
    ggml_build_forward_expand(gf, high1);
    ggml_build_forward_expand(gf, embed);
    for (auto & pr : probe_list) ggml_build_forward_expand(gf, pr.second);

    int status = 0;
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "sam2: encoder graph allocation failed\n");
        status = -2;
    } else {
        ggml_backend_tensor_set(image, chw, 0, (size_t)3 * S * S * sizeof(float));
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "sam2: encoder graph failed\n");
            status = -3;
        } else {
            read_tensor(embed, ctx->image_embed);
            read_tensor(high0, ctx->high_res_0);
            read_tensor(high1, ctx->high_res_1);
            ctx->probes.clear();
            for (auto & pr : probe_list) read_tensor(pr.second, ctx->probes[pr.first]);
            if (ctx->probing) {
                ctx->probes["image_embed"] = ctx->image_embed;
                ctx->probes["high_res_0"] = ctx->high_res_0;
                ctx->probes["high_res_1"] = ctx->high_res_1;
            }
            ctx->has_image = true;
        }
    }
    ggml_backend_sched_reset(ctx->sched);
    ggml_free(g);
    if (ctx->bench) fprintf(stderr, "sam2: encoder %.1f ms on %s\n", now_ms() - t0, ctx->backend_name.c_str());
    return status;
}

// Antialiased bilinear resize (torchvision Resize on a float tensor, antialias=True), then normalisation.
int sam2_set_image_rgb(sam2_context * ctx, const uint8_t * rgb, int width, int height) {
    if (!ctx || !rgb || width <= 0 || height <= 0) return -1;
    const int S = ctx->hp.image_size;
    struct taps {
        int low;
        std::vector<float> w;
    };
    auto weights = [](int input, int output) {
        std::vector<taps> out(output);
        const float scale = (float)input / (float)output;
        const float support = scale >= 1.0f ? scale : 1.0f;
        const float inverse = scale >= 1.0f ? 1.0f / scale : 1.0f;
        for (int i = 0; i < output; i++) {
            const float center = scale * ((float)i + 0.5f);
            const int low = std::max((int)(center - support + 0.5f), 0);
            const int high = std::min((int)(center + support + 0.5f), input);
            float total = 0.0f;
            for (int j = low; j < high; j++) {
                const float x = std::fabs(((float)j - center + 0.5f) * inverse);
                const float v = x < 1.0f ? 1.0f - x : 0.0f;
                out[i].w.push_back(v);
                total += v;
            }
            if (total != 0.0f)
                for (float & v : out[i].w) v /= total;
            out[i].low = low;
        }
        return out;
    };
    const auto across = weights(width, S), down = weights(height, S);
    static const float mean[3] = { 0.485f, 0.456f, 0.406f }, stdv[3] = { 0.229f, 0.224f, 0.225f };
    std::vector<float> chw((size_t)3 * S * S, 0.0f), rows((size_t)height * S);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < height; y++) {
            const uint8_t * line = rgb + (size_t)y * width * 3;
            for (int x = 0; x < S; x++) {
                float sum = 0.0f;
                for (size_t j = 0; j < across[x].w.size(); j++)
                    sum += (float)line[(across[x].low + j) * 3 + c] / 255.0f * across[x].w[j];
                rows[(size_t)y * S + x] = sum;
            }
        }
        float * plane = chw.data() + (size_t)c * S * S;
        for (int y = 0; y < S; y++) {
            float * target = plane + (size_t)y * S;
            for (size_t j = 0; j < down[y].w.size(); j++) {
                const float * source = rows.data() + (size_t)(down[y].low + j) * S;
                for (int x = 0; x < S; x++) target[x] += source[x] * down[y].w[j];
            }
            for (int x = 0; x < S; x++) target[x] = (target[x] - mean[c]) / stdv[c];
        }
    }
    return sam2_set_image_f32(ctx, chw.data());
}

// ── decoder ───────────────────────────────────────────────────────────────

// SAM's Attention: projections to an internal width, heads, out projection. q [256, Nq], k/v [256, Nk].
static ggml_tensor * dec_attention(sam2_context * ctx, ggml_context * g, const std::string & p, ggml_tensor * q,
                                   ggml_tensor * k, ggml_tensor * v) {
    const int heads = ctx->hp.decoder_heads;
    q = linear(g, q, ctx->w(p + "q_proj.weight"), ctx->w(p + "q_proj.bias"));
    k = linear(g, k, ctx->w(p + "k_proj.weight"), ctx->w(p + "k_proj.bias"));
    v = linear(g, v, ctx->w(p + "v_proj.weight"), ctx->w(p + "v_proj.bias"));
    const int di = (int)q->ne[0], hd = di / heads;
    const int64_t Nq = q->ne[1], Nk = k->ne[1];
    q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, q, hd, heads, Nq), 0, 2, 1, 3)); // [hd, Nq, heads]
    k = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, k, hd, heads, Nk), 0, 2, 1, 3)); // [hd, Nk, heads]
    v = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, v, hd, heads, Nk), 1, 2, 0, 3)); // [Nk, hd, heads]
    ggml_tensor * o =
        attention_core(g, ggml_reshape_4d(g, q, hd, Nq, heads, 1), ggml_reshape_4d(g, k, hd, Nk, heads, 1),
                       ggml_reshape_4d(g, v, Nk, hd, heads, 1), hd);
    o = ggml_reshape_2d(g, o, di, Nq);
    return linear(g, o, ctx->w(p + "out_proj.weight"), ctx->w(p + "out_proj.bias"));
}

static ggml_tensor * dec_norm(sam2_context * ctx, ggml_context * g, const std::string & p, ggml_tensor * x) {
    return layer_norm(g, x, ctx->w(p + ".weight"), ctx->w(p + ".bias"), 1e-5f);
}

static ggml_tensor * mlp(sam2_context * ctx, ggml_context * g, const std::string & p, ggml_tensor * x, int layers) {
    for (int i = 0; i < layers; i++) {
        x = linear(g, x, ctx->w(p + "layers." + std::to_string(i) + ".weight"),
                   ctx->w(p + "layers." + std::to_string(i) + ".bias"));
        if (i < layers - 1) x = ggml_relu(g, x);
    }
    return x;
}

// ConvTranspose2d(kernel 2, stride 2) of channel-last [Cin, W, H] with weight ne [Cin, Cout, 2, 2] -> [Cout, 2W, 2H].
static ggml_tensor * conv_transpose_2x2(ggml_context * g, ggml_tensor * x, ggml_tensor * weight, ggml_tensor * bias) {
    const int64_t Cin = x->ne[0], W = x->ne[1], H = x->ne[2], Cout = weight->ne[1];
    ggml_tensor * r = ggml_mul_mat(g, ggml_reshape_2d(g, weight, Cin, Cout * 4),
                                   ggml_reshape_2d(g, x, Cin, W * H)); // [Cout*kw*kh, W*H]
    r = ggml_reshape_4d(g, r, Cout * 2, 2, W, H);                      // [Cout*kw, kh, W, H]
    r = ggml_cont(g, ggml_permute(g, r, 0, 2, 1, 3));                  // [Cout*kw, W, kh, H]
    r = ggml_reshape_3d(g, r, Cout, 2 * W, 2 * H);
    return ggml_add(g, r, bias);
}

static void point_encoding(const sam2_context * ctx, float x, float y, float * out) {
    // _pe_encoding of coordinates normalised to [0, 1]: [sin(2*pi*(2c-1) @ G), cos(...)]
    const float cx = 2.0f * x - 1.0f, cy = 2.0f * y - 1.0f;
    const int half = (int)ctx->gauss.size() / 2;
    for (int j = 0; j < half; j++) {
        const float v = 2.0f * (float)M_PI * (cx * ctx->gauss[j] + cy * ctx->gauss[half + j]);
        out[j] = std::sin(v);
        out[half + j] = std::cos(v);
    }
}

int sam2_predict(sam2_context * ctx, const float * points_xy, const int * labels, int n_points, float * mask_logits,
                 float * scores) {
    if (!ctx || !ctx->has_image || n_points <= 0 || !points_xy || !labels || !mask_logits || !scores) return -1;
    const double t0 = now_ms();
    const int D = ctx->hp.neck_dim, S = ctx->hp.image_size, E = S / 16, M = S / 4;
    const int n_out = 2 + ctx->hp.num_mask_tokens; // obj score, iou, mask tokens
    const int n_tokens = n_out + n_points + 1;     // + padding point (boxes go in as points)

    // Tokens on the host: output tokens, then the prompt points (PromptEncoder._embed_points with pad=True).
    std::vector<float> tokens((size_t)n_tokens * D, 0.0f);
    std::memcpy(tokens.data(), ctx->out_tokens.data(), (size_t)n_out * D * sizeof(float));
    for (int i = 0; i <= n_points; i++) {
        float * row = tokens.data() + (size_t)(n_out + i) * D;
        const int label = i < n_points ? labels[i] : -1;
        if (label == -1) {
            std::memcpy(row, ctx->not_a_point.data(), D * sizeof(float));
            continue;
        }
        if (label < 0 || label > 3) return -1;
        point_encoding(ctx, (points_xy[2 * i] + 0.5f) / (float)S, (points_xy[2 * i + 1] + 0.5f) / (float)S, row);
        for (int c = 0; c < D; c++) row[c] += ctx->point_embed[label][c];
    }

    const int graph_size = 4096;
    ggml_context * g = new_graph_context(ctx, graph_size);
    ggml_cgraph * gf = ggml_new_graph_custom(g, graph_size, false);
    ggml_tensor * tok = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, n_tokens);
    ggml_tensor * img = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, (int64_t)E * E);
    ggml_tensor * hr0 = ggml_new_tensor_3d(g, GGML_TYPE_F32, D / 8, M, M);
    ggml_tensor * hr1 = ggml_new_tensor_3d(g, GGML_TYPE_F32, D / 4, M / 2, M / 2);
    ggml_tensor * no_mask = ggml_new_tensor_1d(g, GGML_TYPE_F32, D);
    for (auto * t : { tok, img, hr0, hr1, no_mask }) ggml_set_input(t);

    ggml_tensor * keys = ggml_add(g, img, no_mask); // dense prompt: no_mask_embed everywhere
    ggml_tensor * key_pe = ctx->dense_pe;
    ggml_tensor * queries = tok;
    for (int l = 0; l < ctx->hp.decoder_depth; l++) {
        const std::string p = "dec.tf.layers." + std::to_string(l) + ".";
        if (l == 0) {
            queries = dec_attention(ctx, g, p + "self_attn.", queries, queries, queries);
        } else {
            ggml_tensor * q = ggml_add(g, queries, tok);
            queries = ggml_add(g, queries, dec_attention(ctx, g, p + "self_attn.", q, q, queries));
        }
        queries = dec_norm(ctx, g, p + "norm1", queries);
        ggml_tensor * q = ggml_add(g, queries, tok);
        ggml_tensor * k = ggml_add(g, keys, key_pe);
        queries = ggml_add(g, queries, dec_attention(ctx, g, p + "t2i.", q, k, keys));
        queries = dec_norm(ctx, g, p + "norm2", queries);
        queries = ggml_add(g, queries, mlp(ctx, g, p + "mlp.", queries, 2));
        queries = dec_norm(ctx, g, p + "norm3", queries);
        q = ggml_add(g, queries, tok);
        k = ggml_add(g, keys, key_pe);
        keys = ggml_add(g, keys, dec_attention(ctx, g, p + "i2t.", k, q, queries));
        keys = dec_norm(ctx, g, p + "norm4", keys);
    }
    {
        ggml_tensor * q = ggml_add(g, queries, tok);
        ggml_tensor * k = ggml_add(g, keys, key_pe);
        queries = ggml_add(g, queries, dec_attention(ctx, g, "dec.tf.final_t2i.", q, k, keys));
        queries = dec_norm(ctx, g, "dec.tf.norm_final_attn", queries);
    }

    // Upscaling with the high-resolution features.
    ggml_tensor * src = ggml_reshape_3d(g, keys, D, E, E);
    ggml_tensor * up =
        conv_transpose_2x2(g, src, ctx->w("dec.output_upscaling.0.weight"), ctx->w("dec.output_upscaling.0.bias"));
    up = ggml_add(g, up, hr1);
    up = layer_norm(g, up, ctx->w("dec.output_upscaling.1.weight"), ctx->w("dec.output_upscaling.1.bias"), 1e-6f);
    up = ggml_gelu_erf(g, up);
    up = conv_transpose_2x2(g, up, ctx->w("dec.output_upscaling.3.weight"), ctx->w("dec.output_upscaling.3.bias"));
    up = ggml_add(g, up, hr0);
    up = ggml_gelu_erf(g, up); // [32, 256, 256]

    std::vector<ggml_tensor *> hyper;
    for (int i = 0; i < ctx->hp.num_mask_tokens; i++) {
        ggml_tensor * t = ggml_view_2d(g, queries, D, 1, queries->nb[1], (size_t)(2 + i) * queries->nb[1]);
        hyper.push_back(mlp(ctx, g, "dec.hyper." + std::to_string(i) + ".", ggml_cont(g, t), 3));
    }
    ggml_tensor * hyper_in = hyper[0];
    for (size_t i = 1; i < hyper.size(); i++) hyper_in = ggml_concat(g, hyper_in, hyper[i], 1);         // [32, 4]
    ggml_tensor * masks = ggml_mul_mat(g, ggml_reshape_2d(g, up, up->ne[0], (int64_t)M * M), hyper_in); // [M*M, 4]
    ggml_tensor * iou_tok = ggml_cont(g, ggml_view_2d(g, queries, D, 1, queries->nb[1], queries->nb[1]));
    ggml_tensor * iou = ggml_sigmoid(g, mlp(ctx, g, "dec.iou_head.", iou_tok, 3));
    ggml_set_output(masks);
    ggml_set_output(iou);
    ggml_build_forward_expand(gf, masks);
    ggml_build_forward_expand(gf, iou);

    int status = 0;
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        status = -2;
    } else {
        ggml_backend_tensor_set(tok, tokens.data(), 0, tokens.size() * sizeof(float));
        ggml_backend_tensor_set(img, ctx->image_embed.data(), 0, ctx->image_embed.size() * sizeof(float));
        ggml_backend_tensor_set(hr0, ctx->high_res_0.data(), 0, ctx->high_res_0.size() * sizeof(float));
        ggml_backend_tensor_set(hr1, ctx->high_res_1.data(), 0, ctx->high_res_1.size() * sizeof(float));
        ggml_backend_tensor_set(no_mask, ctx->no_mask.data(), 0, D * sizeof(float));
        if (ggml_backend_sched_graph_compute(ctx->sched, gf) != GGML_STATUS_SUCCESS) {
            status = -3;
        } else {
            ggml_backend_tensor_get(masks, mask_logits, 0, (size_t)ctx->hp.num_mask_tokens * M * M * sizeof(float));
            ggml_backend_tensor_get(iou, scores, 0, (size_t)ctx->hp.num_mask_tokens * sizeof(float));
        }
    }
    ggml_backend_sched_reset(ctx->sched);
    ggml_free(g);
    if (ctx->bench) fprintf(stderr, "sam2: decoder %.1f ms (%d points)\n", now_ms() - t0, n_points);
    return status;
}

// ── load ──────────────────────────────────────────────────────────────────

static std::vector<int> int_array(gguf_context * meta, const char * key, const std::vector<int> & fallback) {
    std::vector<int> v = core_gguf::kv_i32_array(meta, key);
    return v.empty() ? fallback : v;
}

// The trunk's position embedding at the token grid: interp @ small @ interp^T + tiled window, channel-last.
static std::vector<float> position_embedding(sam2_context * ctx, int grid) {
    const int C = ctx->hp.embed_dim;
    const std::vector<float> small = core_cpu::to_f32(ctx->w("hiera.pos_small"));   // [7(y)][7(x)][C]
    const std::vector<float> window = core_cpu::to_f32(ctx->w("hiera.pos_window")); // [8(y)][8(x)][C]
    const std::vector<float> interp = core_cpu::to_f32(ctx->w("hiera.pos_interp")); // [grid][7]
    const int s = (int)ctx->w("hiera.pos_small")->ne[1], ws = (int)ctx->w("hiera.pos_window")->ne[1];
    std::vector<double> rows((size_t)s * grid * C, 0.0); // [i(y source)][x][C]
    for (int i = 0; i < s; i++)
        for (int x = 0; x < grid; x++)
            for (int j = 0; j < s; j++) {
                const double a = interp[(size_t)x * s + j];
                const float * p = small.data() + ((size_t)i * s + j) * C;
                double * r = rows.data() + ((size_t)i * grid + x) * C;
                for (int c = 0; c < C; c++) r[c] += a * p[c];
            }
    std::vector<float> out((size_t)grid * grid * C);
    for (int y = 0; y < grid; y++)
        for (int x = 0; x < grid; x++) {
            float * o = out.data() + ((size_t)y * grid + x) * C;
            const float * wn = window.data() + ((size_t)(y % ws) * ws + (x % ws)) * C;
            for (int c = 0; c < C; c++) {
                double v = 0.0;
                for (int i = 0; i < s; i++) v += interp[(size_t)y * s + i] * rows[((size_t)i * grid + x) * C + c];
                o[c] = (float)v + wn[c];
            }
        }
    return out;
}

sam2_context * sam2_init(const char * model_path, int n_threads) {
    auto * ctx = new sam2_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "sam2: cannot open %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    hparams & hp = ctx->hp;
    hp.image_size = (int)core_gguf::kv_u32(meta, "sam2.image_size", 1024);
    hp.embed_dim = (int)core_gguf::kv_u32(meta, "sam2.embed_dim", 96);
    hp.num_heads = (int)core_gguf::kv_u32(meta, "sam2.num_heads", 1);
    hp.stages = int_array(meta, "sam2.stages", hp.stages);
    hp.global_att_blocks = int_array(meta, "sam2.global_att_blocks", hp.global_att_blocks);
    hp.window_spec = int_array(meta, "sam2.window_spec", hp.window_spec);
    hp.q_pool = (int)core_gguf::kv_u32(meta, "sam2.q_pool", 3);
    hp.neck_dim = (int)core_gguf::kv_u32(meta, "sam2.neck_dim", 256);
    hp.fpn_top_down_levels = int_array(meta, "sam2.fpn_top_down_levels", hp.fpn_top_down_levels);
    hp.decoder_depth = (int)core_gguf::kv_u32(meta, "sam2.decoder_depth", 2);
    hp.decoder_heads = (int)core_gguf::kv_u32(meta, "sam2.decoder_heads", 8);
    hp.num_mask_tokens = (int)core_gguf::kv_u32(meta, "sam2.num_mask_tokens", 4);
    core_gguf::free_metadata(meta);

    // Block layout as hieradet.Hiera.__init__ derives it.
    int sum = 0;
    for (int s : hp.stages) ctx->stage_ends.push_back((sum += s) - 1);
    std::vector<int> pool_blocks;
    for (size_t i = 0; i + 1 < ctx->stage_ends.size() && (int)pool_blocks.size() < hp.q_pool; i++)
        pool_blocks.push_back(ctx->stage_ends[i] + 1);
    int dim = hp.embed_dim, heads = hp.num_heads, stage = 1;
    for (int i = 0; i < sum; i++) {
        int dim_out = dim;
        int window = hp.window_spec[stage - 1];
        for (int gb : hp.global_att_blocks)
            if (gb == i) window = 0;
        for (int end : ctx->stage_ends)
            if (end == i - 1) {
                dim_out = dim * 2;
                heads *= 2;
                stage++;
            }
        bool pooled = false;
        for (int pb : pool_blocks) pooled |= (pb == i);
        ctx->blocks.push_back({ dim, dim_out, heads, window, pooled });
        dim = dim_out;
    }

    const bool force_cpu = core_env::on("SAM2_FORCE_CPU");
    ctx->backend = force_cpu ? ggml_backend_cpu_init() : crispasr_init_gpu_backend();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx->backend)) ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    ctx->backend_name = ggml_backend_name(ctx->backend);
    if (!core_gguf::load_weights(model_path, ctx->backend, "sam2", ctx->wl)) {
        fprintf(stderr, "sam2: failed to load weights from %s\n", model_path);
        sam2_free(ctx);
        return nullptr;
    }
    ctx->backend_cpu = ggml_backend_is_cpu(ctx->backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);
    std::vector<ggml_backend_t> backends{ ctx->backend };
    if (ctx->backend_cpu) backends.push_back(ctx->backend_cpu);
    ctx->sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), 32768, false, false);

    // Prompt-encoder weights on the host.
    auto host = [&](const char * name) { return core_cpu::to_f32(ctx->w(name)); };
    for (const char * name :
         { "hiera.pos_small", "hiera.pos_window", "hiera.pos_interp", "prompt.gauss", "prompt.not_a_point_embed.weight",
           "prompt.no_mask_embed.weight", "dec.obj_score_token.weight", "dec.iou_token.weight",
           "dec.mask_tokens.weight", "no_mem_embed" }) {
        if (!ctx->w(name)) {
            sam2_free(ctx);
            return nullptr;
        }
    }
    ctx->gauss = host("prompt.gauss");
    for (int i = 0; i < 4; i++)
        ctx->point_embed[i] = host(("prompt.point_embeddings." + std::to_string(i) + ".weight").c_str());
    ctx->not_a_point = host("prompt.not_a_point_embed.weight");
    ctx->no_mask = host("prompt.no_mask_embed.weight");
    for (const char * name : { "dec.obj_score_token.weight", "dec.iou_token.weight", "dec.mask_tokens.weight" }) {
        std::vector<float> v = host(name);
        ctx->out_tokens.insert(ctx->out_tokens.end(), v.begin(), v.end());
    }

    // Constants on the backend: position embedding of the trunk, dense position encoding of the decoder.
    const int grid = hp.image_size / 4, E = hp.image_size / 16, D = hp.neck_dim;
    ggml_init_params ip = { ggml_tensor_overhead() * 4, nullptr, true };
    ctx->const_ctx = ggml_init(ip);
    ctx->pos_full = ggml_new_tensor_3d(ctx->const_ctx, GGML_TYPE_F32, hp.embed_dim, grid, grid);
    ctx->dense_pe = ggml_new_tensor_2d(ctx->const_ctx, GGML_TYPE_F32, D, (int64_t)E * E);
    ctx->const_buf = ggml_backend_alloc_ctx_tensors(ctx->const_ctx, ctx->backend);
    const std::vector<float> pos = position_embedding(ctx, grid);
    ggml_backend_tensor_set(ctx->pos_full, pos.data(), 0, pos.size() * sizeof(float));
    std::vector<float> pe((size_t)D * E * E);
    for (int y = 0; y < E; y++)
        for (int x = 0; x < E; x++)
            point_encoding(ctx, ((float)x + 0.5f) / (float)E, ((float)y + 0.5f) / (float)E,
                           pe.data() + ((size_t)y * E + x) * D);
    ggml_backend_tensor_set(ctx->dense_pe, pe.data(), 0, pe.size() * sizeof(float));

    ctx->bench = core_env::on("CRISPEMBED_SAM2_BENCH");
    fprintf(stderr, "sam2: %d blocks, %d tensors, backend %s\n", (int)ctx->blocks.size(), (int)ctx->wl.tensors.size(),
            ctx->backend_name.c_str());
    return ctx;
}

void sam2_free(sam2_context * ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->const_buf) ggml_backend_buffer_free(ctx->const_buf);
    if (ctx->const_ctx) ggml_free(ctx->const_ctx);
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

int sam2_image_size(const sam2_context * ctx) {
    return ctx ? ctx->hp.image_size : 0;
}

int sam2_mask_size(const sam2_context * ctx) {
    return ctx ? ctx->hp.image_size / 4 : 0;
}

const char * sam2_backend_name(const sam2_context * ctx) {
    return ctx ? ctx->backend_name.c_str() : "";
}

void sam2_set_probing(sam2_context * ctx, bool on) {
    if (ctx) ctx->probing = on;
}

const std::map<std::string, std::vector<float>> & sam2_probes(const sam2_context * ctx) {
    return ctx->probes;
}
