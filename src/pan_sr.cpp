// pan_sr.cpp — PAN whole-image super-resolution (CPU-scalar).
//
// Forward:
//   conv_first(3→nf) → 16× SCPA → trunk_conv → skip
//   → nearest 2× → upconv1 → PA → LReLU → HRconv1 → LReLU
//   → nearest 2× → upconv2 → PA → LReLU → HRconv2 → LReLU  (if 4×)
//   → conv_last(unf→3) → + bilinear(input)
//
// SCPA:
//   conv1_a(nf→nf/2) → LReLU → k1(3×3) → LReLU
//   conv1_b(nf→nf/2) → LReLU → PAConv(k2→sigmoid→mask, k3*mask, k4) → LReLU
//   concat → conv3(nf→nf) → + residual

#include "pan_sr.h"
#include "core/cpu_ops.h"
#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>

// MSVC's <cmath> doesn't define M_PI without _USE_MATH_DEFINES.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ── Helpers ────────────────────────────────────────────────────────────

static void pan_conv2d(const float * input, int ic, int ih, int iw,
                       const float * weight, const float * bias,
                       int oc, int kh, int kw, int pad,
                       float * output) {
    int oh = ih + 2 * pad - kh + 1;
    int ow = iw + 2 * pad - kw + 1;
    for (int o = 0; o < oc; o++) {
        float b = bias ? bias[o] : 0.0f;
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                float sum = b;
                for (int c = 0; c < ic; c++)
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy + ky - pad, ix = ox + kx - pad;
                            if (iy >= 0 && iy < ih && ix >= 0 && ix < iw)
                                sum += input[c * ih * iw + iy * iw + ix]
                                     * weight[o * ic * kh * kw + c * kh * kw + ky * kw + kx];
                        }
                output[o * oh * ow + oy * ow + ox] = sum;
            }
        }
    }
}

static void pan_leaky_relu(float * data, int n, float slope = 0.2f) {
    for (int i = 0; i < n; i++)
        data[i] = data[i] > 0 ? data[i] : data[i] * slope;
}

static void pan_sigmoid(float * data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = 1.0f / (1.0f + expf(-data[i]));
}

// Nearest-neighbor 2× upsample: [C, H, W] → [C, 2H, 2W]
static void pan_nearest_2x(const float * src, int c, int h, int w, float * dst) {
    int oh = h * 2, ow = w * 2;
    for (int ch = 0; ch < c; ch++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                dst[ch * oh * ow + y * ow + x] = src[ch * h * w + (y / 2) * w + (x / 2)];
}

// Bilinear upsample for global residual
static void pan_bilinear(const float * src, int c, int h, int w, int scale, float * dst) {
    int oh = h * scale, ow = w * scale;
    for (int ch = 0; ch < c; ch++)
        for (int oy = 0; oy < oh; oy++) {
            float sy = ((float)oy + 0.5f) * h / oh - 0.5f;
            int iy = (int)floorf(sy); float fy = sy - iy;
            int iy0 = std::max(0, iy), iy1 = std::min(h - 1, iy + 1);
            for (int ox = 0; ox < ow; ox++) {
                float sx = ((float)ox + 0.5f) * w / ow - 0.5f;
                int ix = (int)floorf(sx); float fx = sx - ix;
                int ix0 = std::max(0, ix), ix1 = std::min(w - 1, ix + 1);
                dst[ch * oh * ow + oy * ow + ox] =
                    (1-fy)*((1-fx)*src[ch*h*w + iy0*w + ix0] + fx*src[ch*h*w + iy0*w + ix1])
                    + fy*((1-fx)*src[ch*h*w + iy1*w + ix0] + fx*src[ch*h*w + iy1*w + ix1]);
            }
        }
}

// ── Context ────────────────────────────────────────────────────────────

struct pan_sr_context {
    int nf, unf, nb, scale;
    int n_threads;
    bool bench;
    core_gguf::WeightLoad wl;
    core_cpu::DequantCache dcache;

    // ggml conv infrastructure
    ggml_backend_t       enc_backend  = nullptr;
    ggml_backend_sched_t enc_sched    = nullptr;

    // Persistent F32 graph weights (conv kernels + biases), keyed by tensor name.
    ggml_context *        gw_ctx = nullptr;
    ggml_backend_buffer_t gw_buf = nullptr;
    std::map<std::string, ggml_tensor *> gw;
    std::vector<uint8_t>  graph_meta;   // reused metadata buffer for the forward graph

    const float * get(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) { fprintf(stderr, "pan_sr: missing %s\n", name.c_str()); return nullptr; }
        return dcache.get(t);
    }

    // Fetch a persistent F32 graph-weight tensor by name (must exist).
    ggml_tensor * gwt(const std::string & name) {
        auto it = gw.find(name);
        if (it == gw.end()) { fprintf(stderr, "pan_sr: missing graph weight %s\n", name.c_str()); return nullptr; }
        return it->second;
    }
};

pan_sr_context * pan_sr_init(const char * model_path, int n_threads) {
    auto * ctx = new pan_sr_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) { fprintf(stderr, "pan_sr: failed to open %s\n", model_path); delete ctx; return nullptr; }
    ctx->nf    = core_gguf::kv_u32(meta, "pan.nf", 40);
    ctx->unf   = core_gguf::kv_u32(meta, "pan.unf", 24);
    ctx->nb    = core_gguf::kv_u32(meta, "pan.nb", 16);
    ctx->scale = core_gguf::kv_u32(meta, "pan.scale", 4);
    core_gguf::free_metadata(meta);

    bool force_cpu = (getenv("PAN_SR_FORCE_CPU") && atoi(getenv("PAN_SR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, backend, "pan", ctx->wl)) {
        fprintf(stderr, "pan_sr: failed to load weights\n");
        ggml_backend_free(backend); delete ctx; return nullptr;
    }
    ggml_backend_free(backend);

    fprintf(stderr, "pan_sr: nf=%d unf=%d nb=%d scale=%dx, %d tensors\n",
            ctx->nf, ctx->unf, ctx->nb, ctx->scale, (int)ctx->wl.tensors.size());
    ctx->bench = (std::getenv("CRISPEMBED_PAN_SR_BENCH") != nullptr);

    ctx->enc_backend = ggml_backend_cpu_init();
    if (ctx->enc_backend) {
        ggml_backend_cpu_set_n_threads(ctx->enc_backend, ctx->n_threads > 0 ? ctx->n_threads : 1);
        ggml_backend_t backends[] = { ctx->enc_backend };
        ctx->enc_sched = ggml_backend_sched_new(backends, nullptr, 1, 4096, false, false);
    }

    // Build persistent F32 graph weights for the ggml_conv_2d forward path.
    // Each conv kernel keeps its native ggml layout ne=[KW,KH,IC,OC] (the GGUF
    // already stores it that way — see pan_conv2d's [o*ic*kh*kw + c*kh*kw + ky*kw + kx]
    // indexing, which is row-major [OC][IC][KH][KW] == that ne). No permute needed.
    {
        size_t n_t = ctx->wl.tensors.size();
        ggml_init_params ip = { ggml_tensor_overhead() * (n_t + 4), nullptr, true };
        ctx->gw_ctx = ggml_init(ip);
        auto ends_with = [](const std::string & s, const char * suf) {
            size_t n = strlen(suf);
            return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
        };
        std::vector<std::pair<std::string, ggml_tensor *>> to_fill;
        for (auto & [name, t] : ctx->wl.tensors) {
            ggml_tensor * w;
            if (ends_with(name, ".weight")) {
                // Conv kernel: GGUF stores ne in PyTorch order [OC,IC,KH,KW] but the
                // data is KW-innermost. ggml_conv_2d wants ne=[KW,KH,IC,OC] over those
                // same bytes — reverse the 4 axes, copy the raw buffer unchanged.
                int64_t ne[4] = { t->ne[3], t->ne[2], t->ne[1], t->ne[0] };
                w = ggml_new_tensor(ctx->gw_ctx, GGML_TYPE_F32, 4, ne);
            } else {
                w = ggml_new_tensor(ctx->gw_ctx, GGML_TYPE_F32, ggml_n_dims(t), t->ne);
            }
            ggml_set_name(w, name.c_str());
            ctx->gw[name] = w;
            to_fill.push_back({name, w});
        }
        ctx->gw_buf = ggml_backend_alloc_ctx_tensors(ctx->gw_ctx, ctx->enc_backend);
        for (auto & [name, w] : to_fill) {
            const float * src = ctx->get(name);   // dequantized F32, native memory order
            if (src) ggml_backend_tensor_set(w, src, 0, ggml_nbytes(w));
        }
    }
    return ctx;
}

void pan_sr_free(pan_sr_context * ctx) {
    if (ctx) {
    if (ctx->gw_buf) ggml_backend_buffer_free(ctx->gw_buf);
    if (ctx->gw_ctx) ggml_free(ctx->gw_ctx);
    if (ctx->enc_sched) ggml_backend_sched_free(ctx->enc_sched);
    if (ctx->enc_backend) ggml_backend_free(ctx->enc_backend);
 core_gguf::free_weights(ctx->wl); delete ctx; }
}

int pan_sr_scale(const pan_sr_context * ctx) { return ctx ? ctx->scale : 0; }

// ── Single-tile forward (ggml graph) ──────────────────────────────────
//
// Full-image PAN forward as one ggml_conv_2d graph, run on enc_sched. Mirrors
// the scalar pan_forward_tile op-for-op so the parity test exercises identical
// math. Conv kernels keep native ggml layout (see gw prep in pan_sr_init).

static void pan_forward_tile_ggml(pan_sr_context * ctx,
                                  const float * input, int W, int H,
                                  float * output) {
    const int scale = ctx->scale;
    const int graph_size = 2048;
    size_t buf_size = ggml_tensor_overhead() * (graph_size + 64)
                    + ggml_graph_overhead_custom(graph_size, false);
    ctx->graph_meta.resize(buf_size);
    ggml_init_params ip = { buf_size, ctx->graph_meta.data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(g, graph_size, false);

    ggml_tensor * x = ggml_new_tensor_3d(g, GGML_TYPE_F32, W, H, 3);
    ggml_set_name(x, "x"); ggml_set_input(x);

    // conv with optional per-output-channel bias; pad applied symmetrically.
    auto conv = [&](ggml_tensor * in, const char * wn, const char * bn, int pad) -> ggml_tensor * {
        ggml_tensor * w = ctx->gwt(wn);
        ggml_tensor * y = ggml_conv_2d(g, w, in, 1, 1, pad, pad, 1, 1);
        if (bn) {
            ggml_tensor * b = ggml_reshape_4d(g, ctx->gwt(bn), 1, 1, w->ne[3], 1);
            y = ggml_add(g, y, b);
        }
        return y;
    };
    auto lrelu = [&](ggml_tensor * t) { return ggml_leaky_relu(g, t, 0.2f, false); };

    ggml_tensor * fea = conv(x, "conv_first.weight", "conv_first.bias", 1);
    ggml_tensor * fea_skip = fea;

    for (int i = 0; i < ctx->nb; i++) {
        char wn[80]; std::string p = "scpa." + std::to_string(i);
        ggml_tensor * residual = fea;

        auto nm = [&](const char * s) { snprintf(wn, sizeof(wn), "%s.%s", p.c_str(), s); return wn; };

        // Branch A: conv1_a(1×1) → LReLU → k1(3×3) → LReLU
        std::string a_w = nm("conv1_a.weight");
        ggml_tensor * a = lrelu(conv(fea, a_w.c_str(), nullptr, 0));
        std::string k1_w = nm("k1.weight");
        a = lrelu(conv(a, k1_w.c_str(), nullptr, 1));

        // Branch B: conv1_b(1×1) → LReLU → PAConv → LReLU
        std::string b_w = nm("conv1_b.weight");
        ggml_tensor * b = lrelu(conv(fea, b_w.c_str(), nullptr, 0));
        std::string k2_w = nm("paconv.k2.weight"), k2_b = nm("paconv.k2.bias");
        ggml_tensor * attn = ggml_sigmoid(g, conv(b, k2_w.c_str(), k2_b.c_str(), 0));
        std::string k3_w = nm("paconv.k3.weight");
        ggml_tensor * k3 = ggml_mul(g, conv(b, k3_w.c_str(), nullptr, 1), attn);
        std::string k4_w = nm("paconv.k4.weight");
        ggml_tensor * b2 = lrelu(conv(k3, k4_w.c_str(), nullptr, 1));

        ggml_tensor * cat = ggml_concat(g, a, b2, 2);   // channel dim
        std::string c3_w = nm("conv3.weight");
        fea = ggml_add(g, conv(cat, c3_w.c_str(), nullptr, 0), residual);
    }

    ggml_tensor * trunk = conv(fea, "trunk_conv.weight", "trunk_conv.bias", 1);
    fea = ggml_add(g, fea_skip, trunk);

    // Upsample stage 1
    fea = ggml_upscale(g, fea, 2, GGML_SCALE_MODE_NEAREST);
    fea = conv(fea, "upconv1.weight", "upconv1.bias", 1);
    {
        ggml_tensor * pa = ggml_sigmoid(g, conv(fea, "att1.weight", "att1.bias", 0));
        fea = lrelu(ggml_mul(g, fea, pa));
    }
    fea = lrelu(conv(fea, "hrconv1.weight", "hrconv1.bias", 1));

    // Upsample stage 2 (4×)
    if (scale == 4) {
        fea = ggml_upscale(g, fea, 2, GGML_SCALE_MODE_NEAREST);
        fea = conv(fea, "upconv2.weight", "upconv2.bias", 1);
        ggml_tensor * pa = ggml_sigmoid(g, conv(fea, "att2.weight", "att2.bias", 0));
        fea = lrelu(ggml_mul(g, fea, pa));
        fea = lrelu(conv(fea, "hrconv2.weight", "hrconv2.bias", 1));
    }

    ggml_tensor * out = conv(fea, "conv_last.weight", "conv_last.bias", 1);
    int oh = H * scale, ow = W * scale;
    ggml_tensor * ilr = ggml_interpolate(g, x, ow, oh, 3, 1, GGML_SCALE_MODE_BILINEAR);
    out = ggml_add(g, out, ilr);
    ggml_set_name(out, "out"); ggml_set_output(out);

    ggml_build_forward_expand(gf, out);
    if (!ggml_backend_sched_alloc_graph(ctx->enc_sched, gf)) {
        fprintf(stderr, "pan_sr: sched_alloc_graph failed\n"); ggml_free(g); return;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), input, 0, (size_t)3 * H * W * sizeof(float));
    if (ggml_backend_sched_graph_compute(ctx->enc_sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "pan_sr: graph_compute failed\n"); ggml_free(g); return;
    }
    ggml_backend_tensor_get(out, output, 0, (size_t)3 * oh * ow * sizeof(float));
    ggml_backend_sched_reset(ctx->enc_sched);
    ggml_free(g);
}

// ── Single-tile forward (scalar reference) ────────────────────────────

static void pan_forward_tile_scalar(pan_sr_context * ctx,
                             const float * input, int W, int H,
                             float * output) {
    int nf = ctx->nf, unf = ctx->unf, scale = ctx->scale;

    // conv_first
    std::vector<float> fea(nf * H * W);
    pan_conv2d(input, 3, H, W, ctx->get("conv_first.weight"), ctx->get("conv_first.bias"),
               nf, 3, 3, 1, fea.data());
    std::vector<float> fea_skip = fea;

    int gw = nf / 2;  // group_width = nf/reduction = nf/2

    // SCPA trunk
    for (int i = 0; i < ctx->nb; i++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "scpa.%d", i);
        std::string p(buf);

        std::vector<float> residual = fea;

        // Branch A: conv1_a(nf→gw, 1×1) → LReLU → k1(3×3) → LReLU
        std::vector<float> a(gw * H * W);
        pan_conv2d(fea.data(), nf, H, W, ctx->get(p + ".conv1_a.weight"), nullptr, gw, 1, 1, 0, a.data());
        pan_leaky_relu(a.data(), gw * H * W);
        std::vector<float> a2(gw * H * W);
        pan_conv2d(a.data(), gw, H, W, ctx->get(p + ".k1.weight"), nullptr, gw, 3, 3, 1, a2.data());
        pan_leaky_relu(a2.data(), gw * H * W);

        // Branch B: conv1_b(nf→gw, 1×1) → LReLU → PAConv → LReLU
        std::vector<float> b(gw * H * W);
        pan_conv2d(fea.data(), nf, H, W, ctx->get(p + ".conv1_b.weight"), nullptr, gw, 1, 1, 0, b.data());
        pan_leaky_relu(b.data(), gw * H * W);

        // PAConv: k2→sigmoid, k3*sigmoid, k4
        std::vector<float> attn(gw * H * W);
        pan_conv2d(b.data(), gw, H, W, ctx->get(p + ".paconv.k2.weight"), ctx->get(p + ".paconv.k2.bias"),
                   gw, 1, 1, 0, attn.data());
        pan_sigmoid(attn.data(), gw * H * W);

        std::vector<float> k3out(gw * H * W);
        pan_conv2d(b.data(), gw, H, W, ctx->get(p + ".paconv.k3.weight"), nullptr, gw, 3, 3, 1, k3out.data());
        for (int j = 0; j < gw * H * W; j++) k3out[j] *= attn[j];

        std::vector<float> b2(gw * H * W);
        pan_conv2d(k3out.data(), gw, H, W, ctx->get(p + ".paconv.k4.weight"), nullptr, gw, 3, 3, 1, b2.data());
        pan_leaky_relu(b2.data(), gw * H * W);

        // Concat [a2, b2] → conv3 → + residual
        std::vector<float> cat(nf * H * W);
        memcpy(cat.data(), a2.data(), gw * H * W * sizeof(float));
        memcpy(cat.data() + gw * H * W, b2.data(), gw * H * W * sizeof(float));

        pan_conv2d(cat.data(), nf, H, W, ctx->get(p + ".conv3.weight"), nullptr, nf, 1, 1, 0, fea.data());
        for (int j = 0; j < nf * H * W; j++) fea[j] += residual[j];
    }

    // trunk_conv + skip
    std::vector<float> trunk(nf * H * W);
    pan_conv2d(fea.data(), nf, H, W, ctx->get("trunk_conv.weight"), ctx->get("trunk_conv.bias"),
               nf, 3, 3, 1, trunk.data());
    for (int j = 0; j < nf * H * W; j++) fea[j] = fea_skip[j] + trunk[j];

    // Upsample stage 1
    int h1 = H * 2, w1 = W * 2;
    std::vector<float> up1(nf * h1 * w1);
    pan_nearest_2x(fea.data(), nf, H, W, up1.data());
    std::vector<float> uc1(unf * h1 * w1);
    pan_conv2d(up1.data(), nf, h1, w1, ctx->get("upconv1.weight"), ctx->get("upconv1.bias"),
               unf, 3, 3, 1, uc1.data());
    // PA1
    std::vector<float> pa1(unf * h1 * w1);
    pan_conv2d(uc1.data(), unf, h1, w1, ctx->get("att1.weight"), ctx->get("att1.bias"),
               unf, 1, 1, 0, pa1.data());
    pan_sigmoid(pa1.data(), unf * h1 * w1);
    for (int j = 0; j < unf * h1 * w1; j++) uc1[j] *= pa1[j];
    pan_leaky_relu(uc1.data(), unf * h1 * w1);
    // HRconv1
    std::vector<float> hr1(unf * h1 * w1);
    pan_conv2d(uc1.data(), unf, h1, w1, ctx->get("hrconv1.weight"), ctx->get("hrconv1.bias"),
               unf, 3, 3, 1, hr1.data());
    pan_leaky_relu(hr1.data(), unf * h1 * w1);

    float * cur = hr1.data();
    int cur_h = h1, cur_w = w1, cur_c = unf;

    // Upsample stage 2 (4×)
    std::vector<float> hr2;
    if (scale == 4) {
        int h2 = cur_h * 2, w2 = cur_w * 2;
        std::vector<float> up2(unf * h2 * w2);
        pan_nearest_2x(cur, unf, cur_h, cur_w, up2.data());
        std::vector<float> uc2(unf * h2 * w2);
        pan_conv2d(up2.data(), unf, h2, w2, ctx->get("upconv2.weight"), ctx->get("upconv2.bias"),
                   unf, 3, 3, 1, uc2.data());
        std::vector<float> pa2(unf * h2 * w2);
        pan_conv2d(uc2.data(), unf, h2, w2, ctx->get("att2.weight"), ctx->get("att2.bias"),
                   unf, 1, 1, 0, pa2.data());
        pan_sigmoid(pa2.data(), unf * h2 * w2);
        for (int j = 0; j < unf * h2 * w2; j++) uc2[j] *= pa2[j];
        pan_leaky_relu(uc2.data(), unf * h2 * w2);
        hr2.resize(unf * h2 * w2);
        pan_conv2d(uc2.data(), unf, h2, w2, ctx->get("hrconv2.weight"), ctx->get("hrconv2.bias"),
                   unf, 3, 3, 1, hr2.data());
        pan_leaky_relu(hr2.data(), unf * h2 * w2);
        cur = hr2.data(); cur_h = h2; cur_w = w2;
    }

    // conv_last + bilinear residual
    int oh = H * scale, ow = W * scale;
    std::vector<float> out(3 * oh * ow);
    pan_conv2d(cur, unf, cur_h, cur_w, ctx->get("conv_last.weight"), ctx->get("conv_last.bias"),
               3, 3, 3, 1, out.data());
    std::vector<float> ilr(3 * oh * ow);
    pan_bilinear(input, 3, H, W, scale, ilr.data());
    for (int j = 0; j < 3 * oh * ow; j++) output[j] = out[j] + ilr[j];
}

// Dispatch: ggml graph by default; PAN_SR_SCALAR=1 forces the scalar reference.
static void pan_forward_tile(pan_sr_context * ctx,
                             const float * input, int W, int H,
                             float * output) {
    static const bool use_scalar = (getenv("PAN_SR_SCALAR") && atoi(getenv("PAN_SR_SCALAR")));
    if (use_scalar || !ctx->enc_sched)
        pan_forward_tile_scalar(ctx, input, W, H, output);
    else
        pan_forward_tile_ggml(ctx, input, W, H, output);
}

// ── Tiled processing ──────────────────────────────────────────────────

int pan_sr_process(pan_sr_context * ctx,
                   const uint8_t * input, int width, int height,
                   int tile_size, int tile_overlap,
                   uint8_t ** output, int * out_width, int * out_height) {
    if (!ctx || !input || !output || width <= 0 || height <= 0) return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    int scale = ctx->scale;
    if (tile_size <= 0) tile_size = 128;
    if (tile_overlap <= 0) tile_overlap = 16;
    tile_overlap = std::min(tile_overlap, tile_size / 4);

    int ow = width * scale, oh = height * scale;
    int out_tile = tile_size * scale;
    int out_overlap = tile_overlap * scale;

    std::vector<float> accum(3 * oh * ow, 0.0f);
    std::vector<float> weight_map(oh * ow, 0.0f);

    // Full input [3, H, W] float [0, 1]
    auto t_pre = std::chrono::steady_clock::now();
    std::vector<float> full(3 * height * width);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                full[c * height * width + y * width + x] =
                    input[(y * width + x) * 3 + c] / 255.0f;
    if (bench) {
        auto t_pre_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[pan_sr-bench] preprocess: %.1f ms\n",
                ms_f(t_pre_end - t_pre).count());
    }

    int step = tile_size - tile_overlap;
    int ntx = std::max(1, (width + step - 1) / step);
    int nty = std::max(1, (height + step - 1) / step);

    fprintf(stderr, "pan_sr: %dx%d → %dx%d (%dx), tiles=%dx%d\n",
            width, height, ow, oh, scale, ntx, nty);

    for (int ty = 0; ty < nty; ty++) {
        for (int tx = 0; tx < ntx; tx++) {
            int x0 = std::min(tx * step, std::max(0, width - tile_size));
            int y0 = std::min(ty * step, std::max(0, height - tile_size));
            int tw = std::min(tile_size, width - x0);
            int th = std::min(tile_size, height - y0);

            std::vector<float> tile_in(3 * th * tw);
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < th; y++)
                    for (int x = 0; x < tw; x++)
                        tile_in[c * th * tw + y * tw + x] =
                            full[c * height * width + (y0 + y) * width + (x0 + x)];

            int otw = tw * scale, oth = th * scale;
            std::vector<float> tile_out(3 * oth * otw);
            auto t_fwd = std::chrono::steady_clock::now();
            pan_forward_tile(ctx, tile_in.data(), tw, th, tile_out.data());
            if (bench) {
                auto t_fwd_end = std::chrono::steady_clock::now();
                fprintf(stderr, "[pan_sr-bench] tile %d,%d forward: %.1f ms\n",
                        ty, tx, ms_f(t_fwd_end - t_fwd).count());
            }

            // Blend with Hann ramp at overlapping edges
            int ox0 = x0 * scale, oy0 = y0 * scale;
            for (int y = 0; y < oth; y++) {
                float wy = 1.0f;
                if (y0 > 0 && y < out_overlap)
                    wy = 0.5f - 0.5f * cosf((float)M_PI * y / out_overlap);
                if (y0 + th < height && y >= oth - out_overlap)
                    wy = 0.5f - 0.5f * cosf((float)M_PI * (oth - 1 - y) / out_overlap);
                for (int x = 0; x < otw; x++) {
                    float wx = 1.0f;
                    if (x0 > 0 && x < out_overlap)
                        wx = 0.5f - 0.5f * cosf((float)M_PI * x / out_overlap);
                    if (x0 + tw < width && x >= otw - out_overlap)
                        wx = 0.5f - 0.5f * cosf((float)M_PI * (otw - 1 - x) / out_overlap);
                    float w = wy * wx;
                    int dy = oy0 + y, dx = ox0 + x;
                    if (dy >= oh || dx >= ow) continue;
                    for (int c = 0; c < 3; c++)
                        accum[c * oh * ow + dy * ow + dx] += tile_out[c * oth * otw + y * otw + x] * w;
                    weight_map[dy * ow + dx] += w;
                }
            }
        }
    }

    auto t_post = std::chrono::steady_clock::now();
    uint8_t * out_buf = (uint8_t *)malloc(3 * oh * ow);
    if (!out_buf) return -1;
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++) {
            float w = weight_map[y * ow + x];
            if (w <= 0) w = 1.0f;
            for (int c = 0; c < 3; c++) {
                float v = accum[c * oh * ow + y * ow + x] / w * 255.0f;
                out_buf[(y * ow + x) * 3 + c] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
        }

    *output = out_buf;
    *out_width = ow;
    *out_height = oh;
    fprintf(stderr, "pan_sr: done %dx%d\n", ow, oh);
    if (bench) {
        auto t_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[pan_sr-bench] postprocess: %.1f ms\n",
                ms_f(t_end - t_post).count());
        fprintf(stderr, "[pan_sr-bench] total: %.1f ms\n",
                ms_f(t_end - t_total).count());
    }
    return 0;
}

void pan_sr_free_image(uint8_t * pixels) { free(pixels); }
