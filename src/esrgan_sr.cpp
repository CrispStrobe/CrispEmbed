// esrgan_sr.cpp — Real-ESRGAN SRVGGNetCompact (CPU-scalar).
//
// Forward pass: x → [Conv3x3 + PReLU] × 17 → Conv3x3 → PixelShuffle(4)
//               + nearest-upsample(input) → output
//
// Body layout: body.0=Conv(3→64), body.1=PReLU, body.2=Conv(64→64),
// body.3=PReLU, ..., body.32=Conv(64→64), body.33=PReLU, body.34=Conv(64→48)

#include "esrgan_sr.h"
#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const float * to_f32(const ggml_tensor * t, std::vector<float> & buf) {
    int64_t n = ggml_nelements(t);
    buf.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        size_t raw_sz = ggml_nbytes(t);
        std::vector<uint8_t> raw(raw_sz);
        ggml_backend_tensor_get(t, raw.data(), 0, raw_sz);
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) traits->to_float(raw.data(), buf.data(), n);
        else memset(buf.data(), 0, n * sizeof(float));
    }
    return buf.data();
}

static void conv2d(const float * input, int ic, int ih, int iw,
                   const float * weight, const float * bias,
                   int oc, float * output) {
    // 3×3, pad=1, stride=1, groups=1
    for (int o = 0; o < oc; o++) {
        float b = bias[o];
        for (int oy = 0; oy < ih; oy++) {
            for (int ox = 0; ox < iw; ox++) {
                float sum = b;
                for (int c = 0; c < ic; c++) {
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = oy + ky - 1;
                        if (iy < 0 || iy >= ih) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = ox + kx - 1;
                            if (ix < 0 || ix >= iw) continue;
                            sum += input[c * ih * iw + iy * iw + ix]
                                 * weight[o * ic * 9 + c * 9 + ky * 3 + kx];
                        }
                    }
                }
                output[o * ih * iw + oy * iw + ox] = sum;
            }
        }
    }
}

// PReLU: y = max(0,x) + slope * min(0,x). Per-channel slopes.
static void prelu(float * data, int c, int hw, const float * slopes) {
    for (int ch = 0; ch < c; ch++) {
        float s = slopes[ch];
        for (int i = 0; i < hw; i++) {
            float & v = data[ch * hw + i];
            if (v < 0) v *= s;
        }
    }
}

static void pixel_shuffle(const float * input, int c_in, int h, int w,
                           int r, float * output) {
    int c_out = c_in / (r * r), oh = h * r, ow = w * r;
    for (int c = 0; c < c_out; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                int ic = c * r * r + (y % r) * r + (x % r);
                output[c * oh * ow + y * ow + x] =
                    input[ic * h * w + (y / r) * w + (x / r)];
            }
}

static void nearest_upsample(const float * input, int c, int h, int w,
                              int scale, float * output) {
    int oh = h * scale, ow = w * scale;
    for (int ch = 0; ch < c; ch++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                output[ch * oh * ow + y * ow + x] =
                    input[ch * h * w + (y / scale) * w + (x / scale)];
}

// ── Context ────────────────────────────────────────────────────────

struct conv_layer { ggml_tensor * w; ggml_tensor * b; };
struct prelu_layer { ggml_tensor * slope; };

struct esrgan_context {
    ggml_context * gguf_ctx;
    ggml_backend_buffer_t gguf_buf;
    int scale, num_feat, num_conv;
    // body.0=conv, body.1=prelu, body.2=conv, ..., body.34=conv
    std::vector<conv_layer> convs;   // 18 convolutions
    std::vector<prelu_layer> prelus; // 17 PReLU layers
};

esrgan_context * esrgan_init(const char * model_path, int n_threads) {
    (void)n_threads;
    if (!model_path) return nullptr;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) return nullptr;
    int scale    = (int)core_gguf::kv_u32(meta, "esrgan.scale", 4);
    int num_feat = (int)core_gguf::kv_u32(meta, "esrgan.num_feat", 64);
    int num_conv = (int)core_gguf::kv_u32(meta, "esrgan.num_conv", 16);
    core_gguf::free_metadata(meta);

    bool force_cpu = (getenv("ESRGAN_SR_FORCE_CPU") && atoi(getenv("ESRGAN_SR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, backend, "esrgan", wl)) {
        ggml_backend_free(backend); return nullptr;
    }
    ggml_backend_free(backend);

    auto * ctx = new esrgan_context;
    ctx->gguf_ctx = wl.ctx;
    ctx->gguf_buf = wl.buf;
    ctx->scale = scale;
    ctx->num_feat = num_feat;
    ctx->num_conv = num_conv;

    // 18 convs (entry + 16 body + output), 17 PReLUs (entry + 16 body)
    int n_total = 2 * num_conv + 2 + 1; // body indices 0..34
    for (int i = 0; i < n_total; i++) {
        char wn[64], bn[64];
        snprintf(wn, sizeof(wn), "body.%d.weight", i);
        snprintf(bn, sizeof(bn), "body.%d.bias", i);
        ggml_tensor * w = core_gguf::try_get(wl.tensors, wn);
        ggml_tensor * b = core_gguf::try_get(wl.tensors, bn);
        if (b) {
            // Conv layer (has bias)
            ctx->convs.push_back({w, b});
        } else if (w) {
            // PReLU layer (weight = slopes, no bias)
            ctx->prelus.push_back({w});
        }
    }

    return ctx;
}

void esrgan_free(esrgan_context * ctx) {
    if (!ctx) return;
    core_gguf::WeightLoad wl;
    wl.ctx = ctx->gguf_ctx;
    wl.buf = ctx->gguf_buf;
    core_gguf::free_weights(wl);
    delete ctx;
}

int esrgan_get_scale(const esrgan_context * ctx) {
    return ctx ? ctx->scale : 0;
}

int esrgan_process_float(esrgan_context * ctx,
                         const float * input_chw, int width, int height,
                         float * output_chw) {
    if (!ctx || !input_chw || !output_chw) return -1;

    const int H = height, W = width, hw = H * W;
    std::vector<float> dq1, dq2;

    // Two ping-pong buffers
    std::vector<float> buf_a(input_chw, input_chw + 3 * hw);
    std::vector<float> buf_b;

    int ic = 3;
    int conv_idx = 0, prelu_idx = 0;

    // Process: conv → prelu pairs, then final conv (no prelu)
    int n_convs = (int)ctx->convs.size();
    for (int ci = 0; ci < n_convs; ci++) {
        int oc = (ci == n_convs - 1) ? 3 * ctx->scale * ctx->scale : ctx->num_feat;
        buf_b.resize(oc * hw);
        conv2d(buf_a.data(), ic, H, W,
               to_f32(ctx->convs[ci].w, dq1),
               to_f32(ctx->convs[ci].b, dq2),
               oc, buf_b.data());

        // PReLU after every conv except the last
        if (ci < n_convs - 1 && prelu_idx < (int)ctx->prelus.size()) {
            prelu(buf_b.data(), oc, hw,
                  to_f32(ctx->prelus[prelu_idx].slope, dq1));
            prelu_idx++;
        }

        std::swap(buf_a, buf_b);
        ic = oc;
    }

    // PixelShuffle
    int out_h = H * ctx->scale, out_w = W * ctx->scale;
    int out_hw = out_h * out_w;
    std::vector<float> sr(3 * out_hw);
    pixel_shuffle(buf_a.data(), ic, H, W, ctx->scale, sr.data());

    // Global residual: nearest-upsample input + sr
    nearest_upsample(input_chw, 3, H, W, ctx->scale, output_chw);
    for (int i = 0; i < 3 * out_hw; i++)
        output_chw[i] += sr[i];

    return 0;
}

int esrgan_process(esrgan_context * ctx,
                   const uint8_t * input, int width, int height,
                   uint8_t * output) {
    if (!ctx || !input || !output) return -1;
    int hw = width * height;
    std::vector<float> in_chw(3 * hw);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                in_chw[c * hw + y * width + x] =
                    (float)input[(y * width + x) * 3 + c] / 255.0f;

    int oh = height * ctx->scale, ow = width * ctx->scale;
    std::vector<float> out_chw(3 * oh * ow);
    int ret = esrgan_process_float(ctx, in_chw.data(), width, height, out_chw.data());
    if (ret != 0) return ret;

    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
            for (int c = 0; c < 3; c++) {
                float v = out_chw[c * oh * ow + y * ow + x] * 255.0f;
                output[(y * ow + x) * 3 + c] =
                    (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
    return 0;
}
