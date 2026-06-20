// tps_locnet.cpp — TPS localization network: small CNN that predicts
// control point displacements for thin-plate spline dewarping.
//
// Loads weights from GGUF (see models/convert-tps-loc-to-gguf.py).
// CPU-scalar forward pass — fast enough for document images (~5ms).
//
// Architecture (PaddleOCR "small"):
//   Conv0(3→16, 3x3) + ReLU + MaxPool2x2
//   Conv1(16→32, 3x3) + ReLU + MaxPool2x2
//   Conv2(32→64, 3x3) + ReLU + MaxPool2x2
//   Conv3(64→128, 3x3) + ReLU + AdaptiveAvgPool(1x1)
//   FC1(128→64) + ReLU
//   FC2(64→N*2)
//
// BN is folded into conv weights at conversion time (no BN at runtime).

#include "tps_warp.h"
#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers: CPU-scalar conv2d, pooling, FC
// ---------------------------------------------------------------------------

// Dequant any ggml tensor to float (GPU-safe: uses ggml_backend_tensor_get)
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

// Conv2d: [OC, IC, KH, KW] weights, [OC] bias
// Input/output are [C, H, W] planar float
static void conv2d(const float * input, int ic, int ih, int iw,
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
                for (int c = 0; c < ic; c++) {
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy + ky - pad;
                            int ix = ox + kx - pad;
                            if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                            sum += input[c * ih * iw + iy * iw + ix]
                                 * weight[o * ic * kh * kw + c * kh * kw + ky * kw + kx];
                        }
                    }
                }
                output[o * oh * ow + oy * ow + ox] = sum;
            }
        }
    }
}

// ReLU in-place
static void relu_inplace(float * data, int n) {
    for (int i = 0; i < n; i++)
        if (data[i] < 0) data[i] = 0;
}

// MaxPool2x2 stride 2: [C, H, W] → [C, H/2, W/2]
static void maxpool2x2(const float * input, int c, int h, int w,
                       float * output) {
    int oh = h / 2, ow = w / 2;
    for (int ch = 0; ch < c; ch++) {
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                float m = -1e30f;
                for (int dy = 0; dy < 2; dy++) {
                    for (int dx = 0; dx < 2; dx++) {
                        int iy = oy * 2 + dy, ix = ox * 2 + dx;
                        if (iy < h && ix < w) {
                            float v = input[ch * h * w + iy * w + ix];
                            if (v > m) m = v;
                        }
                    }
                }
                output[ch * oh * ow + oy * ow + ox] = m;
            }
        }
    }
}

// Adaptive average pooling to (1, 1): [C, H, W] → [C]
static void adaptive_avg_pool_1x1(const float * input, int c, int h, int w,
                                   float * output) {
    int hw = h * w;
    for (int ch = 0; ch < c; ch++) {
        float sum = 0;
        for (int i = 0; i < hw; i++) sum += input[ch * hw + i];
        output[ch] = sum / hw;
    }
}

// FC: output[o] = bias[o] + sum_i(weight[o*ic + i] * input[i])
static void fc_forward(const float * input, int ic,
                       const float * weight, const float * bias, int oc,
                       float * output) {
    for (int o = 0; o < oc; o++) {
        float sum = bias ? bias[o] : 0.0f;
        for (int i = 0; i < ic; i++)
            sum += weight[o * ic + i] * input[i];
        output[o] = sum;
    }
}

// ---------------------------------------------------------------------------
// Localization network context
// ---------------------------------------------------------------------------

struct tps_locnet {
    // GGUF state
    ggml_context * gguf_ctx;
    ggml_backend_buffer_t gguf_buf;

    // Hyperparams
    int num_fiducial;
    int fc_dim;
    int channels[4]; // per-conv output channels

    // Weights (pointers into gguf_ctx tensors, dequantized lazily)
    struct conv_layer {
        ggml_tensor * w; // [oc, ic, 3, 3]
        ggml_tensor * b; // [oc]
    } conv[4];

    struct fc_layer {
        ggml_tensor * w; // [oc, ic]
        ggml_tensor * b; // [oc]
    } fc1, fc2;

    bool bench = false;
};

tps_locnet * tps_locnet_load(const char * gguf_path) {
    if (!gguf_path) return nullptr;

    // Pass 1: metadata
    gguf_context * meta = core_gguf::open_metadata(gguf_path);
    if (!meta) return nullptr;

    int num_fiducial = (int)core_gguf::kv_u32(meta, "tps.num_fiducial", 10);
    int fc_dim = (int)core_gguf::kv_u32(meta, "tps.fc_dim", 64);

    core_gguf::free_metadata(meta);

    // Pass 2: load weights
    bool force_cpu = (getenv("TPS_LOCNET_FORCE_CPU") && atoi(getenv("TPS_LOCNET_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(gguf_path, backend, "tps_locnet", wl)) {
        ggml_backend_free(backend);
        return nullptr;
    }
    ggml_backend_free(backend);

    tps_locnet * net = new tps_locnet;
    net->gguf_ctx = wl.ctx;
    net->gguf_buf = wl.buf;
    net->num_fiducial = num_fiducial;
    net->fc_dim = fc_dim;
    net->bench = (std::getenv("CRISPEMBED_TPS_LOCNET_BENCH") != nullptr);

    // Bind tensors
    for (int i = 0; i < 4; i++) {
        std::string wn = "loc.conv" + std::to_string(i) + ".weight";
        std::string bn = "loc.conv" + std::to_string(i) + ".bias";
        net->conv[i].w = core_gguf::require(wl.tensors, wn.c_str(), "tps_locnet");
        net->conv[i].b = core_gguf::require(wl.tensors, bn.c_str(), "tps_locnet");
        if (!net->conv[i].w || !net->conv[i].b) {
            delete net;
            core_gguf::free_weights(wl);
            return nullptr;
        }
        net->channels[i] = (int)net->conv[i].w->ne[3]; // [kw, kh, ic, oc] in ggml
    }

    net->fc1.w = core_gguf::require(wl.tensors, "loc.fc1.weight", "tps_locnet");
    net->fc1.b = core_gguf::require(wl.tensors, "loc.fc1.bias", "tps_locnet");
    net->fc2.w = core_gguf::require(wl.tensors, "loc.fc2.weight", "tps_locnet");
    net->fc2.b = core_gguf::require(wl.tensors, "loc.fc2.bias", "tps_locnet");

    if (!net->fc1.w || !net->fc1.b || !net->fc2.w || !net->fc2.b) {
        delete net;
        core_gguf::free_weights(wl);
        return nullptr;
    }

    return net;
}

int tps_locnet_num_fiducial(const tps_locnet * net) {
    return net ? net->num_fiducial : 0;
}

int tps_locnet_predict(tps_locnet * net,
                       const uint8_t * gray, int w, int h,
                       float * out_x, float * out_y) {
    if (!net || !gray || !out_x || !out_y || w <= 0 || h <= 0) return 0;

    const bool bench = net->bench;
    auto t_total = std::chrono::steady_clock::now();

    const int F = net->num_fiducial;

    // Preprocess: grayscale → 3-channel planar float in [0,1]
    // (The localization net expects 3-channel input; replicate gray → RGB)
    auto t_pre0 = std::chrono::steady_clock::now();
    int cur_h = h, cur_w = w;
    std::vector<float> buf_a(3 * cur_h * cur_w);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < cur_h; y++) {
            for (int x = 0; x < cur_w; x++) {
                buf_a[c * cur_h * cur_w + y * cur_w + x] =
                    (float)gray[y * w + x] / 255.0f;
            }
        }
    }
    if (bench) {
        auto t_pre1 = std::chrono::steady_clock::now();
        fprintf(stderr, "[tps_locnet-bench] preprocess: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t_pre1 - t_pre0).count());
    }

    std::vector<float> buf_b;
    std::vector<float> dq_w, dq_b;

    // Forward pass: 4 conv blocks
    auto t_conv0 = std::chrono::steady_clock::now();
    int ic = 3;
    for (int i = 0; i < 4; i++) {
        int oc = net->channels[i];
        int oh = cur_h, ow = cur_w; // same-padding conv preserves size

        // Conv2d (3x3, pad=1)
        const float * wptr = to_f32(net->conv[i].w, dq_w);
        const float * bptr = to_f32(net->conv[i].b, dq_b);

        // GGUF data is in numpy row-major order: [OC, IC, KH, KW].
        // Our conv2d() expects the same layout — use directly.
        buf_b.resize(oc * oh * ow);
        conv2d(buf_a.data(), ic, cur_h, cur_w,
               wptr, bptr, oc, 3, 3, 1,
               buf_b.data());

        // ReLU
        relu_inplace(buf_b.data(), oc * oh * ow);

        // Pooling
        if (i < 3) {
            // MaxPool 2x2
            int ph = oh / 2, pw = ow / 2;
            buf_a.resize(oc * ph * pw);
            maxpool2x2(buf_b.data(), oc, oh, ow, buf_a.data());
            cur_h = ph;
            cur_w = pw;
        } else {
            // AdaptiveAvgPool(1,1)
            buf_a.resize(oc);
            adaptive_avg_pool_1x1(buf_b.data(), oc, oh, ow, buf_a.data());
            cur_h = 1;
            cur_w = 1;
        }

        ic = oc;
    }

    if (bench) {
        auto t_conv1 = std::chrono::steady_clock::now();
        fprintf(stderr, "[tps_locnet-bench] conv layers: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t_conv1 - t_conv0).count());
    }

    // FC1 + ReLU
    auto t_fc0 = std::chrono::steady_clock::now();
    const float * fc1_w = to_f32(net->fc1.w, dq_w);
    const float * fc1_b = to_f32(net->fc1.b, dq_b);

    // ggml stores FC as [ic, oc] — transpose to [oc, ic]
    std::vector<float> fc1_wt(net->fc_dim * ic);
    for (int o = 0; o < net->fc_dim; o++)
        for (int i = 0; i < ic; i++)
            fc1_wt[o * ic + i] = fc1_w[i * net->fc_dim + o];

    buf_b.resize(net->fc_dim);
    fc_forward(buf_a.data(), ic, fc1_wt.data(), fc1_b, net->fc_dim, buf_b.data());
    relu_inplace(buf_b.data(), net->fc_dim);

    // FC2
    const float * fc2_w = to_f32(net->fc2.w, dq_w);
    const float * fc2_b = to_f32(net->fc2.b, dq_b);

    std::vector<float> fc2_wt(F * 2 * net->fc_dim);
    for (int o = 0; o < F * 2; o++)
        for (int i = 0; i < net->fc_dim; i++)
            fc2_wt[o * net->fc_dim + i] = fc2_w[i * F * 2 + o];

    std::vector<float> raw_pts(F * 2);
    fc_forward(buf_b.data(), net->fc_dim, fc2_wt.data(), fc2_b, F * 2, raw_pts.data());

    // Convert from [-1, 1] normalized coords to pixel coords
    for (int i = 0; i < F; i++) {
        out_x[i] = (raw_pts[i * 2 + 0] + 1.0f) * 0.5f * (w - 1);
        out_y[i] = (raw_pts[i * 2 + 1] + 1.0f) * 0.5f * (h - 1);
    }

    if (bench) {
        auto t_fc1 = std::chrono::steady_clock::now();
        auto t_total1 = std::chrono::steady_clock::now();
        fprintf(stderr, "[tps_locnet-bench] FC: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t_fc1 - t_fc0).count());
        fprintf(stderr, "[tps_locnet-bench] total: %.3f ms\n",
                std::chrono::duration<double, std::milli>(t_total1 - t_total).count());
    }

    return F;
}

void tps_locnet_free(tps_locnet * net) {
    if (!net) return;
    core_gguf::WeightLoad wl;
    wl.ctx = net->gguf_ctx;
    wl.buf = net->gguf_buf;
    core_gguf::free_weights(wl);
    delete net;
}

// ---------------------------------------------------------------------------
// Full auto-dewarp pipeline
// ---------------------------------------------------------------------------

int tps_auto_dewarp(const uint8_t * gray, int w, int h,
                    const char * gguf_path,
                    uint8_t * out) {
    if (!gray || !out || !gguf_path || w <= 0 || h <= 0) return 1;

    tps_locnet * net = tps_locnet_load(gguf_path);
    if (!net) return 1;

    int F = tps_locnet_num_fiducial(net);
    std::vector<float> src_x(F), src_y(F);

    int n = tps_locnet_predict(net, gray, w, h, src_x.data(), src_y.data());
    tps_locnet_free(net);
    if (n <= 0) return 1;

    // Generate target grid: regular grid matching the fiducial layout.
    // PaddleOCR convention: F/2 points on top edge, F/2 on bottom edge,
    // evenly spaced horizontally.
    std::vector<float> dst_x(F), dst_y(F);
    int half = F / 2;
    for (int i = 0; i < half; i++) {
        float t = (float)i / (float)(half - 1);
        // Top row
        dst_x[i] = t * (w - 1);
        dst_y[i] = 0;
        // Bottom row
        dst_x[half + i] = t * (w - 1);
        dst_y[half + i] = (float)(h - 1);
    }

    return tps_warp_points(gray, w, h,
                           dst_x.data(), dst_y.data(),
                           src_x.data(), src_y.data(),
                           F, out, w, h, 255);
}
