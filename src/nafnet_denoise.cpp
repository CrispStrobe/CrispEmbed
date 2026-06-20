// nafnet_denoise.cpp — NAFNet image denoising (CPU-scalar implementation)
//
// NAFBlock forward:
//   x = LN1(x)
//   x = Conv1(1x1, c→2c) → DWConv(3x3, 2c) → SimpleGate(2c→c)
//   x = x * SCA(AvgPool→Conv1x1) → Conv3(1x1, c→c)
//   x = input + x * beta
//   x = LN2(x)
//   x = Conv4(1x1, c→2c) → SimpleGate(2c→c) → Conv5(1x1, c→c)
//   x = prev + x * gamma

#include "nafnet_denoise.h"
#include "core/cpu_ops.h"
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

// ── Helpers ─────────────────────────────────────────────────────────

// Dequant any ggml tensor to float — GPU-safe (uses ggml_backend_tensor_get)
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
        // Q8_0, Q4_K, etc. — read raw bytes then dequantize
        size_t raw_sz = ggml_nbytes(t);
        std::vector<uint8_t> raw(raw_sz);
        ggml_backend_tensor_get(t, raw.data(), 0, raw_sz);
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(raw.data(), buf.data(), n);
        } else {
            memset(buf.data(), 0, n * sizeof(float));
        }
    }
    return buf.data();
}

// Conv2d: [OC, IC, KH, KW] weights, [OC] bias (optional)
// Input/output are [C, H, W] planar float
static void conv2d_cpu(const float * input, int ic, int ih, int iw,
                       const float * weight, const float * bias,
                       int oc, int kh, int kw, int stride, int pad,
                       int groups, float * output) {
    int oh = (ih + 2 * pad - kh) / stride + 1;
    int ow = (iw + 2 * pad - kw) / stride + 1;
    int ic_per_group = ic / groups;
    int oc_per_group = oc / groups;

    for (int g = 0; g < groups; g++) {
        for (int oc_i = 0; oc_i < oc_per_group; oc_i++) {
            int oc_abs = g * oc_per_group + oc_i;
            float b = bias ? bias[oc_abs] : 0.0f;
            for (int oy = 0; oy < oh; oy++) {
                for (int ox = 0; ox < ow; ox++) {
                    float sum = b;
                    for (int ic_i = 0; ic_i < ic_per_group; ic_i++) {
                        int ic_abs = g * ic_per_group + ic_i;
                        for (int ky = 0; ky < kh; ky++) {
                            for (int kx = 0; kx < kw; kx++) {
                                int iy = oy * stride + ky - pad;
                                int ix = ox * stride + kx - pad;
                                if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                                float v = input[ic_abs * ih * iw + iy * iw + ix];
                                float w = weight[oc_abs * ic_per_group * kh * kw +
                                                  ic_i * kh * kw + ky * kw + kx];
                                sum += v * w;
                            }
                        }
                    }
                    output[oc_abs * oh * ow + oy * ow + ox] = sum;
                }
            }
        }
    }
}

// LayerNorm2d: normalize over C dimension for each spatial position
// Input: [C, H, W], output: [C, H, W]
static void layernorm2d(const float * input, int c, int h, int w,
                        const float * weight, const float * bias,
                        float * output) {
    float eps = 1e-6f;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // Compute mean and variance over channels
            float mean = 0;
            for (int ch = 0; ch < c; ch++) {
                mean += input[ch * h * w + y * w + x];
            }
            mean /= c;

            float var = 0;
            for (int ch = 0; ch < c; ch++) {
                float d = input[ch * h * w + y * w + x] - mean;
                var += d * d;
            }
            var /= c;

            float inv_std = 1.0f / sqrtf(var + eps);
            for (int ch = 0; ch < c; ch++) {
                float v = (input[ch * h * w + y * w + x] - mean) * inv_std;
                output[ch * h * w + y * w + x] = v * weight[ch] + bias[ch];
            }
        }
    }
}

// SimpleGate: split channels in half, multiply
// Input: [2C, H, W], output: [C, H, W]
static void simple_gate(const float * input, int c2, int h, int w, float * output) {
    int c = c2 / 2;
    int hw = h * w;
    for (int ch = 0; ch < c; ch++) {
        for (int i = 0; i < hw; i++) {
            output[ch * hw + i] = input[ch * hw + i] * input[(ch + c) * hw + i];
        }
    }
}

// Simplified Channel Attention: global avg pool → 1x1 conv → multiply
static void sca(const float * input, int c, int h, int w,
                const float * sca_weight, const float * sca_bias,
                float * output) {
    int hw = h * w;
    // Global average pooling → [C, 1, 1]
    std::vector<float> pooled(c, 0.0f);
    for (int ch = 0; ch < c; ch++) {
        float sum = 0;
        for (int i = 0; i < hw; i++) sum += input[ch * hw + i];
        pooled[ch] = sum / hw;
    }
    // 1x1 conv (channel mixing)
    std::vector<float> attn(c);
    for (int oc = 0; oc < c; oc++) {
        float sum = sca_bias[oc];
        for (int ic = 0; ic < c; ic++) {
            sum += sca_weight[oc * c + ic] * pooled[ic];
        }
        attn[oc] = sum;
    }
    // Multiply
    for (int ch = 0; ch < c; ch++) {
        for (int i = 0; i < hw; i++) {
            output[ch * hw + i] = input[ch * hw + i] * attn[ch];
        }
    }
}

// PixelShuffle: [C*r*r, H, W] → [C, H*r, W*r]
static void pixel_shuffle(const float * input, int c_in, int h, int w,
                          int r, float * output) {
    int c_out = c_in / (r * r);
    int oh = h * r, ow = w * r;
    for (int c = 0; c < c_out; c++) {
        for (int y = 0; y < oh; y++) {
            for (int x = 0; x < ow; x++) {
                int iy = y / r, ix = x / r;
                int ry = y % r, rx = x % r;
                int ic = c * r * r + ry * r + rx;
                output[c * oh * ow + y * ow + x] = input[ic * h * w + iy * w + ix];
            }
        }
    }
}

// conv2d_ggml: defined after nafnet_context (needs struct members)
static void conv2d_ggml(nafnet_context * ctx,
                         const float * input, int ic, int ih, int iw,
                         ggml_tensor * weight_t, ggml_tensor * bias_t,
                         int oc, int kh, int kw, int stride, int pad,
                         int groups, float * output);

// ── NAFBlock ────────────────────────────────────────────────────────

struct nafblock_weights {
    const float * beta;       // [1, C, 1, 1] → flatten to [C]
    const float * gamma;
    const float * norm1_w;    // [C]
    const float * norm1_b;
    const float * norm2_w;
    const float * norm2_b;
    const float * sca_w_f;    // dequantized SCA weight for scalar path
    const float * sca_b_f;
    // ggml tensor pointers for conv2d_ggml
    ggml_tensor * conv1_wt;   // [2C, C, 1, 1]
    ggml_tensor * conv1_bt;
    ggml_tensor * conv2_wt;   // [2C, 1, 3, 3] (depthwise)
    ggml_tensor * conv2_bt;
    ggml_tensor * conv3_wt;   // [C, C, 1, 1]
    ggml_tensor * conv3_bt;
    ggml_tensor * conv4_wt;   // [2C, C, 1, 1]
    ggml_tensor * conv4_bt;
    ggml_tensor * conv5_wt;   // [C, C, 1, 1]
    ggml_tensor * conv5_bt;
};

static void nafblock_forward(nafnet_context * ctx,
                             const float * input, int c, int h, int w,
                             const nafblock_weights & wt,
                             float * output,
                             std::vector<float> & tmp1,
                             std::vector<float> & tmp2,
                             std::vector<float> & tmp3) {
    int hw = h * w;
    int c2 = c * 2;

    // Part 1: spatial mixing
    tmp1.resize(c * hw);
    layernorm2d(input, c, h, w, wt.norm1_w, wt.norm1_b, tmp1.data());

    // Conv1: 1x1, c → 2c (ggml)
    tmp2.resize(c2 * hw);
    conv2d_ggml(ctx, tmp1.data(), c, h, w, wt.conv1_wt, wt.conv1_bt, c2, 1, 1, 1, 0, 1, tmp2.data());

    // Conv2: 3x3 depthwise (ggml)
    tmp3.resize(c2 * hw);
    conv2d_ggml(ctx, tmp2.data(), c2, h, w, wt.conv2_wt, wt.conv2_bt, c2, 3, 3, 1, 1, c2, tmp3.data());

    // SimpleGate: 2c → c (scalar — just element multiply)
    tmp1.resize(c * hw);
    simple_gate(tmp3.data(), c2, h, w, tmp1.data());

    // SCA (scalar — global pool + small matmul + multiply)
    tmp2.resize(c * hw);
    sca(tmp1.data(), c, h, w, wt.sca_w_f, wt.sca_b_f, tmp2.data());

    // Conv3: 1x1 (ggml)
    tmp1.resize(c * hw);
    conv2d_ggml(ctx, tmp2.data(), c, h, w, wt.conv3_wt, wt.conv3_bt, c, 1, 1, 1, 0, 1, tmp1.data());

    // Residual: input + tmp1 * beta
    tmp2.resize(c * hw);
    for (int ch = 0; ch < c; ch++) {
        float b = wt.beta[ch];
        for (int i = 0; i < hw; i++)
            tmp2[ch * hw + i] = input[ch * hw + i] + tmp1[ch * hw + i] * b;
    }

    // Part 2: channel mixing
    tmp1.resize(c * hw);
    layernorm2d(tmp2.data(), c, h, w, wt.norm2_w, wt.norm2_b, tmp1.data());

    // Conv4: 1x1, c → 2c (ggml)
    tmp3.resize(c2 * hw);
    conv2d_ggml(ctx, tmp1.data(), c, h, w, wt.conv4_wt, wt.conv4_bt, c2, 1, 1, 1, 0, 1, tmp3.data());

    // SimpleGate: 2c → c
    tmp1.resize(c * hw);
    simple_gate(tmp3.data(), c2, h, w, tmp1.data());

    // Conv5: 1x1 (ggml)
    tmp3.resize(c * hw);
    conv2d_ggml(ctx, tmp1.data(), c, h, w, wt.conv5_wt, wt.conv5_bt, c, 1, 1, 1, 0, 1, tmp3.data());

    // Residual: tmp2 + tmp3 * gamma
    for (int ch = 0; ch < c; ch++) {
        float g = wt.gamma[ch];
        for (int i = 0; i < hw; i++)
            output[ch * hw + i] = tmp2[ch * hw + i] + tmp3[ch * hw + i] * g;
    }
}

// ── Context ─────────────────────────────────────────────────────────

struct nafnet_context {
    int width;
    int n_stages;
    std::vector<int> enc_blk_nums;
    std::vector<int> dec_blk_nums;
    int middle_blk_num;
    int n_threads;
    bool bench;

    // Backend (kept alive for GPU-resident weight access)
    ggml_backend_t backend = nullptr;

    // ggml graph infrastructure for batched matmuls
    ggml_backend_t       enc_backend  = nullptr;
    ggml_backend_sched_t enc_sched    = nullptr;
    std::vector<uint8_t> enc_meta;

    // Weight data
    core_gguf::WeightLoad wl;
    std::string model_path;

    // Dequantized weight cache
    core_cpu::DequantCache dcache;

    const float * get_tensor(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) {
            fprintf(stderr, "nafnet: missing tensor %s\n", name.c_str());
            return nullptr;
        }
        return dcache.get(t);
    }
};

nafnet_context * nafnet_init(const char * model_path, int n_threads) {
    auto * ctx = new nafnet_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    ctx->model_path = model_path;

    // Pass 1: metadata
    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "nafnet: failed to open %s\n", model_path);
        delete ctx;
        return nullptr;
    }

    ctx->width = core_gguf::kv_u32(meta, "nafnet.width", 32);
    ctx->n_stages = core_gguf::kv_u32(meta, "nafnet.n_stages", 4);
    ctx->middle_blk_num = core_gguf::kv_u32(meta, "nafnet.middle_blk_num", 12);
    ctx->enc_blk_nums = core_gguf::kv_i32_array(meta, "nafnet.enc_blk_nums");
    ctx->dec_blk_nums = core_gguf::kv_i32_array(meta, "nafnet.dec_blk_nums");
    core_gguf::free_metadata(meta);

    // Pass 2: load weights — prefer GPU backend when available
    // Forward pass is scalar CPU but weights can reside on GPU (read via
    // ggml_backend_tensor_get). Backend kept alive for tensor access.
    bool force_cpu = (getenv("NAFNET_FORCE_CPU") && atoi(getenv("NAFNET_FORCE_CPU")));
    ctx->backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx->backend))
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    if (!core_gguf::load_weights(model_path, ctx->backend, "nafnet", ctx->wl)) {
        fprintf(stderr, "nafnet: failed to load weights\n");
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    fprintf(stderr, "nafnet: width=%d, enc=[", ctx->width);
    for (int i = 0; i < (int)ctx->enc_blk_nums.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", ctx->enc_blk_nums[i]);
    fprintf(stderr, "], mid=%d, dec=[", ctx->middle_blk_num);
    for (int i = 0; i < (int)ctx->dec_blk_nums.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", ctx->dec_blk_nums[i]);
    fprintf(stderr, "], %d tensors\n", (int)ctx->wl.tensors.size());

    ctx->bench = (std::getenv("CRISPEMBED_NAFNET_BENCH") != nullptr);

    // ggml conv infrastructure
    ctx->enc_backend = ggml_backend_cpu_init();
    if (ctx->enc_backend) {
        ggml_backend_cpu_set_n_threads(ctx->enc_backend, ctx->n_threads);
        ggml_backend_t backends[] = { ctx->enc_backend };
        ctx->enc_sched = ggml_backend_sched_new(backends, nullptr, 1, 4096, false, false);
    }

    return ctx;
}

void nafnet_free(nafnet_context * ctx) {
    if (ctx) {
        if (ctx->enc_sched) ggml_backend_sched_free(ctx->enc_sched);
        if (ctx->enc_backend) ggml_backend_free(ctx->enc_backend);
        core_gguf::free_weights(ctx->wl);
        if (ctx->backend) ggml_backend_free(ctx->backend);
        delete ctx;
    }
}

// ── ggml conv2d implementation ──────────────────────────────────────

static void conv2d_ggml(nafnet_context * ctx,
                         const float * input, int ic, int ih, int iw,
                         ggml_tensor * weight_t, ggml_tensor * bias_t,
                         int oc, int kh, int kw, int stride, int pad,
                         int groups, float * output) {
    if (!ctx->enc_sched || !weight_t) {
        // fallback to scalar
        std::vector<float> wf_buf, bf_buf;
        const float * wf = weight_t ? to_f32(weight_t, wf_buf) : nullptr;
        const float * bf = bias_t ? to_f32(bias_t, bf_buf) : nullptr;
        if (wf) conv2d_cpu(input, ic, ih, iw, wf, bf, oc, kh, kw, stride, pad, groups, output);
        return;
    }

    int max_nodes = 32;
    size_t buf_size = ggml_tensor_overhead() * max_nodes
                    + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> meta(buf_size);
    ggml_init_params ip = { buf_size, meta.data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(g, max_nodes, false);

    ggml_tensor * x = ggml_new_tensor_3d(g, GGML_TYPE_F32, iw, ih, ic);
    ggml_set_name(x, "x");
    ggml_set_input(x);

    ggml_tensor * w = weight_t;
    if (groups > 1) {
        if (ggml_n_dims(w) == 2) {
            if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16)
                w = ggml_cont(g, ggml_cast(g, w, GGML_TYPE_F32));
            w = ggml_reshape_4d(g, w, kw, kh, 1, w->ne[1]);
        }
        if (w->type != GGML_TYPE_F16) w = ggml_cast(g, w, GGML_TYPE_F16);
        x = ggml_conv_2d_dw(g, w, x, stride, stride, pad, pad, 1, 1);
    } else {
        if (ggml_n_dims(w) == 2) {
            if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16)
                w = ggml_cont(g, ggml_cast(g, w, GGML_TYPE_F32));
            w = ggml_reshape_4d(g, w, kw, kh, ic, w->ne[1]);
        }
        if (w->type != GGML_TYPE_F16) w = ggml_cast(g, w, GGML_TYPE_F16);
        x = ggml_conv_2d(g, w, x, stride, stride, pad, pad, 1, 1);
    }

    if (bias_t) {
        ggml_tensor * b = ggml_reshape_3d(g, bias_t, 1, 1, oc);
        x = ggml_add(g, x, b);
    }

    ggml_set_name(x, "out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_reset(ctx->enc_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->enc_sched, gf)) {
        fprintf(stderr, "nafnet: conv2d_ggml alloc failed\n");
        return;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), input, 0,
                             ic * ih * iw * sizeof(float));

    for (int i = 0; i < ggml_backend_sched_get_n_backends(ctx->enc_sched); i++) {
        ggml_backend_t be = ggml_backend_sched_get_backend(ctx->enc_sched, i);
        ggml_backend_dev_t dev = ggml_backend_get_device(be);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto * fn = (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (fn) fn(be, ctx->n_threads);
        }
    }
    ggml_backend_sched_graph_compute(ctx->enc_sched, gf);

    int oh = (ih + 2 * pad - kh) / stride + 1;
    int ow = (iw + 2 * pad - kw) / stride + 1;
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out"), output, 0,
                             oc * oh * ow * sizeof(float));
}

// ── Forward pass (single tile) ──────────────────────────────────────

static int nafnet_process_tile(nafnet_context * ctx,
                               const uint8_t * input, int width, int height,
                               uint8_t * output) {
    if (!ctx || !input || !output || width <= 0 || height <= 0) return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    int W = ctx->width;
    int ns = ctx->n_stages;

    // Pad to multiple of 2^n_stages
    int pad_mult = 1 << ns;  // 16 for 4 stages
    int ph = ((height + pad_mult - 1) / pad_mult) * pad_mult;
    int pw = ((width + pad_mult - 1) / pad_mult) * pad_mult;

    // Convert input to [3, ph, pw] float [0, 1]
    auto t_pre = std::chrono::steady_clock::now();
    std::vector<float> img(3 * ph * pw, 0.0f);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                img[c * ph * pw + y * pw + x] = input[(y * width + x) * 3 + c] / 255.0f;
            }
        }
    }
    // Reflect-pad if needed (simple: replicate edge)
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < ph; y++) {
            for (int x = width; x < pw; x++) {
                img[c * ph * pw + y * pw + x] = img[c * ph * pw + y * pw + (width - 1)];
            }
        }
        for (int y = height; y < ph; y++) {
            for (int x = 0; x < pw; x++) {
                img[c * ph * pw + y * pw + x] = img[c * ph * pw + (height - 1) * pw + x];
            }
        }
    }

    // Save input for final residual
    std::vector<float> input_save = img;

    // Scratch buffers for NAFBlock
    std::vector<float> tmp1, tmp2, tmp3;

    // Helper: get raw ggml tensor pointer (not dequantized)
    auto get_raw = [&](const std::string & name) -> ggml_tensor * {
        return core_gguf::try_get(ctx->wl.tensors, name.c_str());
    };

    // Helper to load a NAFBlock's weights
    auto load_block = [&](const std::string & prefix, int c) -> nafblock_weights {
        nafblock_weights wt;
        wt.beta    = ctx->get_tensor(prefix + ".beta");
        wt.gamma   = ctx->get_tensor(prefix + ".gamma");
        wt.norm1_w = ctx->get_tensor(prefix + ".norm1.weight");
        wt.norm1_b = ctx->get_tensor(prefix + ".norm1.bias");
        wt.norm2_w = ctx->get_tensor(prefix + ".norm2.weight");
        wt.norm2_b = ctx->get_tensor(prefix + ".norm2.bias");
        wt.sca_w_f = ctx->get_tensor(prefix + ".sca.weight");
        wt.sca_b_f = ctx->get_tensor(prefix + ".sca.bias");
        // ggml tensors for conv2d_ggml
        wt.conv1_wt = get_raw(prefix + ".conv1.weight");
        wt.conv1_bt = get_raw(prefix + ".conv1.bias");
        wt.conv2_wt = get_raw(prefix + ".conv2.weight");
        wt.conv2_bt = get_raw(prefix + ".conv2.bias");
        wt.conv3_wt = get_raw(prefix + ".conv3.weight");
        wt.conv3_bt = get_raw(prefix + ".conv3.bias");
        wt.conv4_wt = get_raw(prefix + ".conv4.weight");
        wt.conv4_bt = get_raw(prefix + ".conv4.bias");
        wt.conv5_wt = get_raw(prefix + ".conv5.weight");
        wt.conv5_bt = get_raw(prefix + ".conv5.bias");
        return wt;
    };

    int cur_h = ph, cur_w = pw;
    int cur_c = 3;

    // Intro conv: 3 → W (ggml)
    {
        std::vector<float> after_intro(W * cur_h * cur_w);
        conv2d_ggml(ctx, img.data(), 3, cur_h, cur_w,
                     get_raw("intro.weight"), get_raw("intro.bias"),
                     W, 3, 3, 1, 1, 1, after_intro.data());
        img = std::move(after_intro);
        cur_c = W;
    }

    if (bench) {
        auto t_pre_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[nafnet-bench] preprocess+intro: %.1f ms\n",
                ms_f(t_pre_end - t_pre).count());
    }
    fprintf(stderr, "nafnet: intro done (%dx%d, %d ch)\n", cur_w, cur_h, cur_c);

    // Encoder
    std::vector<std::vector<float>> skip_connections;
    for (int s = 0; s < ns; s++) {
        auto t_enc = std::chrono::steady_clock::now();
        int c = W * (1 << s);  // 32, 64, 128, 256

        // Run NAFBlocks
        for (int b = 0; b < ctx->enc_blk_nums[s]; b++) {
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "enc.%d.%d", s, b);
            auto wt = load_block(prefix, c);
            std::vector<float> block_out(c * cur_h * cur_w);
            nafblock_forward(ctx, img.data(), c, cur_h, cur_w, wt, block_out.data(), tmp1, tmp2, tmp3);
            img = std::move(block_out);
        }

        // Save skip connection
        skip_connections.push_back(img);

        // Downsample: Conv2d stride 2, kernel 2 (ggml)
        int next_c = c * 2;
        int next_h = cur_h / 2, next_w = cur_w / 2;
        {
            char name_w[64], name_b[64];
            snprintf(name_w, sizeof(name_w), "downs.%d.weight", s);
            snprintf(name_b, sizeof(name_b), "downs.%d.bias", s);
            std::vector<float> down_out(next_c * next_h * next_w);
            conv2d_ggml(ctx, img.data(), c, cur_h, cur_w,
                         get_raw(name_w), get_raw(name_b),
                         next_c, 2, 2, 2, 0, 1, down_out.data());
            img = std::move(down_out);
        }
        cur_c = next_c;
        cur_h = next_h;
        cur_w = next_w;

        if (bench) {
            auto t_enc_end = std::chrono::steady_clock::now();
            fprintf(stderr, "[nafnet-bench] enc stage %d: %.1f ms\n",
                    s, ms_f(t_enc_end - t_enc).count());
        }
        fprintf(stderr, "nafnet: enc stage %d done (%dx%d, %d ch)\n", s, cur_w, cur_h, cur_c);
    }

    // Middle blocks
    auto t_mid = std::chrono::steady_clock::now();
    for (int b = 0; b < ctx->middle_blk_num; b++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "mid.%d", b);
        auto wt = load_block(prefix, cur_c);
        std::vector<float> block_out(cur_c * cur_h * cur_w);
        nafblock_forward(ctx, img.data(), cur_c, cur_h, cur_w, wt, block_out.data(), tmp1, tmp2, tmp3);
        img = std::move(block_out);
    }
    if (bench) {
        auto t_mid_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[nafnet-bench] middle: %.1f ms\n",
                ms_f(t_mid_end - t_mid).count());
    }
    fprintf(stderr, "nafnet: middle done (%d blocks)\n", ctx->middle_blk_num);

    // Decoder
    for (int s = 0; s < ns; s++) {
        auto t_dec = std::chrono::steady_clock::now();
        // Upsample: 1x1 conv (c → 4c/4 * 4 = c for PixelShuffle) then PixelShuffle(2)
        int next_c = cur_c / 2;
        int next_h = cur_h * 2, next_w = cur_w * 2;
        {
            char name_w[64];
            snprintf(name_w, sizeof(name_w), "ups.%d.weight", s);
            int up_oc = cur_c * 2;
            std::vector<float> up_tmp(up_oc * cur_h * cur_w);
            conv2d_ggml(ctx, img.data(), cur_c, cur_h, cur_w,
                         get_raw(name_w), nullptr,
                         up_oc, 1, 1, 1, 0, 1, up_tmp.data());

            std::vector<float> ps_out(next_c * next_h * next_w);
            pixel_shuffle(up_tmp.data(), up_oc, cur_h, cur_w, 2, ps_out.data());
            img = std::move(ps_out);
        }

        // Add skip connection
        auto & skip = skip_connections[ns - 1 - s];
        for (int i = 0; i < next_c * next_h * next_w; i++) {
            img[i] += skip[i];
        }

        cur_c = next_c;
        cur_h = next_h;
        cur_w = next_w;

        // Run NAFBlocks
        for (int b = 0; b < ctx->dec_blk_nums[s]; b++) {
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "dec.%d.%d", s, b);
            auto wt = load_block(prefix, cur_c);
            std::vector<float> block_out(cur_c * cur_h * cur_w);
            nafblock_forward(ctx, img.data(), cur_c, cur_h, cur_w, wt, block_out.data(), tmp1, tmp2, tmp3);
            img = std::move(block_out);
        }

        if (bench) {
            auto t_dec_end = std::chrono::steady_clock::now();
            fprintf(stderr, "[nafnet-bench] dec stage %d: %.1f ms\n",
                    s, ms_f(t_dec_end - t_dec).count());
        }
        fprintf(stderr, "nafnet: dec stage %d done (%dx%d, %d ch)\n", s, cur_w, cur_h, cur_c);
    }

    // Ending conv: W → 3 (ggml)
    auto t_end_phase = std::chrono::steady_clock::now();
    {
        std::vector<float> ending(3 * cur_h * cur_w);
        conv2d_ggml(ctx, img.data(), W, cur_h, cur_w,
                     get_raw("ending.weight"), get_raw("ending.bias"),
                     3, 3, 3, 1, 1, 1, ending.data());
        img = std::move(ending);
    }

    // Add input residual
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < ph; y++) {
            for (int x = 0; x < pw; x++) {
                img[c * ph * pw + y * pw + x] += input_save[c * ph * pw + y * pw + x];
            }
        }
    }

    // Convert back to uint8 RGB, crop to original size
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                float v = img[c * ph * pw + y * pw + x] * 255.0f;
                output[(y * width + x) * 3 + c] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
        }
    }

    if (bench) {
        auto t_fin = std::chrono::steady_clock::now();
        fprintf(stderr, "[nafnet-bench] ending: %.1f ms\n",
                ms_f(t_fin - t_end_phase).count());
        fprintf(stderr, "[nafnet-bench] total: %.1f ms\n",
                ms_f(t_fin - t_total).count());
    }
    fprintf(stderr, "nafnet: done (%dx%d)\n", width, height);
    return 0;
}

// ── Tiled forward pass with Hann blending ──────────────────────────

static void build_blend_window_1x(int tile_size, int overlap, std::vector<float> & win) {
    win.resize(tile_size * tile_size);
    for (int y = 0; y < tile_size; y++) {
        float wy = 1.0f;
        if (y < overlap)
            wy = 0.5f - 0.5f * cosf((float)M_PI * y / overlap);
        else if (y >= tile_size - overlap)
            wy = 0.5f - 0.5f * cosf((float)M_PI * (tile_size - 1 - y) / overlap);
        for (int x = 0; x < tile_size; x++) {
            float wx = 1.0f;
            if (x < overlap)
                wx = 0.5f - 0.5f * cosf((float)M_PI * x / overlap);
            else if (x >= tile_size - overlap)
                wx = 0.5f - 0.5f * cosf((float)M_PI * (tile_size - 1 - x) / overlap);
            win[y * tile_size + x] = wy * wx;
        }
    }
}

int nafnet_process(nafnet_context * ctx,
                   const uint8_t * input, int width, int height,
                   uint8_t * output) {
    if (!ctx || !input || !output || width <= 0 || height <= 0) return -1;

    // Tile size must be multiple of pad_mult (16 for 4-stage U-Net)
    int pad_mult = 1 << ctx->n_stages;
    int tile_size = 256;
    int tile_overlap = 32;
    const char * ts_env = std::getenv("CRISPEMBED_NAFNET_TILE");
    if (ts_env) tile_size = std::max(pad_mult * 2, atoi(ts_env));
    // Round tile_size up to pad_mult
    tile_size = ((tile_size + pad_mult - 1) / pad_mult) * pad_mult;
    tile_overlap = std::min(tile_overlap, tile_size / 4);

    // Small image: single-shot
    if (width <= tile_size && height <= tile_size)
        return nafnet_process_tile(ctx, input, width, height, output);

    // Tiled processing (1:1 denoising, no upscale)
    std::vector<float> accum(3 * height * width, 0.0f);
    std::vector<float> weight_map(height * width, 0.0f);
    std::vector<float> blend_win;
    build_blend_window_1x(tile_size, tile_overlap, blend_win);

    int step = tile_size - tile_overlap;
    int n_tiles_x = std::max(1, (width + step - 1) / step);
    int n_tiles_y = std::max(1, (height + step - 1) / step);

    fprintf(stderr, "nafnet: %dx%d, tiles=%dx%d (size=%d, overlap=%d)\n",
            width, height, n_tiles_x, n_tiles_y, tile_size, tile_overlap);

    for (int ty = 0; ty < n_tiles_y; ty++) {
        for (int tx = 0; tx < n_tiles_x; tx++) {
            int x0 = std::min(tx * step, std::max(0, width - tile_size));
            int y0 = std::min(ty * step, std::max(0, height - tile_size));
            int tw = std::min(tile_size, width - x0);
            int th = std::min(tile_size, height - y0);

            // Extract tile (HWC uint8)
            std::vector<uint8_t> tile_in(tw * th * 3);
            for (int y = 0; y < th; y++)
                memcpy(tile_in.data() + y * tw * 3,
                       input + ((y0 + y) * width + x0) * 3, tw * 3);

            std::vector<uint8_t> tile_out(tw * th * 3);
            int ret = nafnet_process_tile(ctx, tile_in.data(), tw, th, tile_out.data());
            if (ret != 0) return ret;

            // Blend into accumulator
            for (int y = 0; y < th; y++) {
                for (int x = 0; x < tw; x++) {
                    float w = 1.0f;
                    if (tw == tile_size && th == tile_size)
                        w = blend_win[y * tile_size + x];
                    else {
                        if (x0 > 0 && x < tile_overlap)
                            w *= 0.5f - 0.5f * cosf((float)M_PI * x / tile_overlap);
                        if (y0 > 0 && y < tile_overlap)
                            w *= 0.5f - 0.5f * cosf((float)M_PI * y / tile_overlap);
                    }
                    int dy = y0 + y, dx = x0 + x;
                    for (int c = 0; c < 3; c++)
                        accum[c * height * width + dy * width + dx] +=
                            tile_out[(y * tw + x) * 3 + c] * w;
                    weight_map[dy * width + dx] += w;
                }
            }
        }
    }

    // Normalize and convert to uint8
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            float wt = weight_map[y * width + x];
            if (wt <= 0.0f) wt = 1.0f;
            for (int c = 0; c < 3; c++) {
                float v = accum[c * height * width + y * width + x] / wt;
                output[(y * width + x) * 3 + c] =
                    (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
        }
    return 0;
}
