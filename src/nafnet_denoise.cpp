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
#include "core/gguf_loader.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

// ── Helpers ─────────────────────────────────────────────────────────

// Dequant any ggml tensor to float
static const float * to_f32(const ggml_tensor * t, std::vector<float> & buf) {
    if (t->type == GGML_TYPE_F32) {
        return (const float *)t->data;
    }
    int64_t n = ggml_nelements(t);
    buf.resize(n);
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        // Q8_0, Q4_K, etc. — use ggml dequantization
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(t->data, buf.data(), n);
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

// ── NAFBlock ────────────────────────────────────────────────────────

struct nafblock_weights {
    const float * beta;       // [1, C, 1, 1] → flatten to [C]
    const float * gamma;
    const float * conv1_w;    // [2C, C, 1, 1]
    const float * conv1_b;
    const float * conv2_w;    // [2C, 1, 3, 3] (depthwise)
    const float * conv2_b;
    const float * conv3_w;    // [C, C, 1, 1]
    const float * conv3_b;
    const float * sca_w;      // [C, C, 1, 1]
    const float * sca_b;
    const float * conv4_w;    // [2C, C, 1, 1]
    const float * conv4_b;
    const float * conv5_w;    // [C, C, 1, 1]
    const float * conv5_b;
    const float * norm1_w;    // [C]
    const float * norm1_b;
    const float * norm2_w;
    const float * norm2_b;
};

static void nafblock_forward(const float * input, int c, int h, int w,
                             const nafblock_weights & wt,
                             float * output,
                             std::vector<float> & tmp1,
                             std::vector<float> & tmp2,
                             std::vector<float> & tmp3) {
    int hw = h * w;
    int c2 = c * 2;

    // Part 1: spatial mixing
    // x = LN1(input)
    tmp1.resize(c * hw);
    layernorm2d(input, c, h, w, wt.norm1_w, wt.norm1_b, tmp1.data());

    // Conv1: 1x1, c → 2c
    tmp2.resize(c2 * hw);
    conv2d_cpu(tmp1.data(), c, h, w, wt.conv1_w, wt.conv1_b, c2, 1, 1, 1, 0, 1, tmp2.data());

    // Conv2: 3x3 depthwise
    tmp3.resize(c2 * hw);
    conv2d_cpu(tmp2.data(), c2, h, w, wt.conv2_w, wt.conv2_b, c2, 3, 3, 1, 1, c2, tmp3.data());

    // SimpleGate: 2c → c
    tmp1.resize(c * hw);
    simple_gate(tmp3.data(), c2, h, w, tmp1.data());

    // SCA
    tmp2.resize(c * hw);
    sca(tmp1.data(), c, h, w, wt.sca_w, wt.sca_b, tmp2.data());

    // Conv3: 1x1
    tmp1.resize(c * hw);
    conv2d_cpu(tmp2.data(), c, h, w, wt.conv3_w, wt.conv3_b, c, 1, 1, 1, 0, 1, tmp1.data());

    // Residual: input + tmp1 * beta
    tmp2.resize(c * hw);
    for (int ch = 0; ch < c; ch++) {
        float b = wt.beta[ch];
        for (int i = 0; i < hw; i++) {
            tmp2[ch * hw + i] = input[ch * hw + i] + tmp1[ch * hw + i] * b;
        }
    }

    // Part 2: channel mixing
    // x = LN2(tmp2)
    tmp1.resize(c * hw);
    layernorm2d(tmp2.data(), c, h, w, wt.norm2_w, wt.norm2_b, tmp1.data());

    // Conv4: 1x1, c → 2c
    tmp3.resize(c2 * hw);
    conv2d_cpu(tmp1.data(), c, h, w, wt.conv4_w, wt.conv4_b, c2, 1, 1, 1, 0, 1, tmp3.data());

    // SimpleGate: 2c → c
    tmp1.resize(c * hw);
    simple_gate(tmp3.data(), c2, h, w, tmp1.data());

    // Conv5: 1x1
    tmp3.resize(c * hw);
    conv2d_cpu(tmp1.data(), c, h, w, wt.conv5_w, wt.conv5_b, c, 1, 1, 1, 0, 1, tmp3.data());

    // Residual: tmp2 + tmp3 * gamma
    for (int ch = 0; ch < c; ch++) {
        float g = wt.gamma[ch];
        for (int i = 0; i < hw; i++) {
            output[ch * hw + i] = tmp2[ch * hw + i] + tmp3[ch * hw + i] * g;
        }
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

    // Weight data
    core_gguf::WeightLoad wl;
    std::string model_path;

    // Dequantized weight cache
    std::vector<std::vector<float>> weight_bufs;

    const float * get_tensor(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) {
            fprintf(stderr, "nafnet: missing tensor %s\n", name.c_str());
            return nullptr;
        }
        weight_bufs.emplace_back();
        return to_f32(t, weight_bufs.back());
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

    // Pass 2: load weights (CPU backend)
    // NOTE: CPU-only — forward pass is scalar C++ (NAFNet U-Net with direct
    // tensor->data access, backend freed after load). GPU requires ggml graph rewrite.
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, backend, "nafnet", ctx->wl)) {
        fprintf(stderr, "nafnet: failed to load weights\n");
        ggml_backend_free(backend);
        delete ctx;
        return nullptr;
    }
    ggml_backend_free(backend);

    fprintf(stderr, "nafnet: width=%d, enc=[", ctx->width);
    for (int i = 0; i < (int)ctx->enc_blk_nums.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", ctx->enc_blk_nums[i]);
    fprintf(stderr, "], mid=%d, dec=[", ctx->middle_blk_num);
    for (int i = 0; i < (int)ctx->dec_blk_nums.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", ctx->dec_blk_nums[i]);
    fprintf(stderr, "], %d tensors\n", (int)ctx->wl.tensors.size());

    return ctx;
}

void nafnet_free(nafnet_context * ctx) {
    if (ctx) {
        core_gguf::free_weights(ctx->wl);
        delete ctx;
    }
}

// ── Forward pass ────────────────────────────────────────────────────

int nafnet_process(nafnet_context * ctx,
                   const uint8_t * input, int width, int height,
                   uint8_t * output) {
    if (!ctx || !input || !output || width <= 0 || height <= 0) return -1;

    int W = ctx->width;
    int ns = ctx->n_stages;

    // Pad to multiple of 2^n_stages
    int pad_mult = 1 << ns;  // 16 for 4 stages
    int ph = ((height + pad_mult - 1) / pad_mult) * pad_mult;
    int pw = ((width + pad_mult - 1) / pad_mult) * pad_mult;

    // Convert input to [3, ph, pw] float [0, 1]
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

    // Helper to load a NAFBlock's weights
    auto load_block = [&](const std::string & prefix, int c) -> nafblock_weights {
        nafblock_weights wt;
        wt.beta    = ctx->get_tensor(prefix + ".beta");
        wt.gamma   = ctx->get_tensor(prefix + ".gamma");
        wt.conv1_w = ctx->get_tensor(prefix + ".conv1.weight");
        wt.conv1_b = ctx->get_tensor(prefix + ".conv1.bias");
        wt.conv2_w = ctx->get_tensor(prefix + ".conv2.weight");
        wt.conv2_b = ctx->get_tensor(prefix + ".conv2.bias");
        wt.conv3_w = ctx->get_tensor(prefix + ".conv3.weight");
        wt.conv3_b = ctx->get_tensor(prefix + ".conv3.bias");
        wt.sca_w   = ctx->get_tensor(prefix + ".sca.weight");
        wt.sca_b   = ctx->get_tensor(prefix + ".sca.bias");
        wt.conv4_w = ctx->get_tensor(prefix + ".conv4.weight");
        wt.conv4_b = ctx->get_tensor(prefix + ".conv4.bias");
        wt.conv5_w = ctx->get_tensor(prefix + ".conv5.weight");
        wt.conv5_b = ctx->get_tensor(prefix + ".conv5.bias");
        wt.norm1_w = ctx->get_tensor(prefix + ".norm1.weight");
        wt.norm1_b = ctx->get_tensor(prefix + ".norm1.bias");
        wt.norm2_w = ctx->get_tensor(prefix + ".norm2.weight");
        wt.norm2_b = ctx->get_tensor(prefix + ".norm2.bias");
        return wt;
    };

    int cur_h = ph, cur_w = pw;
    int cur_c = 3;

    // Intro conv: 3 → W
    {
        auto * intro_w = ctx->get_tensor("intro.weight");
        auto * intro_b = ctx->get_tensor("intro.bias");
        std::vector<float> after_intro(W * cur_h * cur_w);
        conv2d_cpu(img.data(), 3, cur_h, cur_w, intro_w, intro_b, W, 3, 3, 1, 1, 1, after_intro.data());
        img = std::move(after_intro);
        cur_c = W;
    }

    fprintf(stderr, "nafnet: intro done (%dx%d, %d ch)\n", cur_w, cur_h, cur_c);

    // Encoder
    std::vector<std::vector<float>> skip_connections;
    for (int s = 0; s < ns; s++) {
        int c = W * (1 << s);  // 32, 64, 128, 256

        // Run NAFBlocks
        for (int b = 0; b < ctx->enc_blk_nums[s]; b++) {
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "enc.%d.%d", s, b);
            auto wt = load_block(prefix, c);
            std::vector<float> block_out(c * cur_h * cur_w);
            nafblock_forward(img.data(), c, cur_h, cur_w, wt, block_out.data(), tmp1, tmp2, tmp3);
            img = std::move(block_out);
        }

        // Save skip connection
        skip_connections.push_back(img);

        // Downsample: Conv2d stride 2, kernel 2
        int next_c = c * 2;
        int next_h = cur_h / 2, next_w = cur_w / 2;
        {
            char name_w[64], name_b[64];
            snprintf(name_w, sizeof(name_w), "downs.%d.weight", s);
            snprintf(name_b, sizeof(name_b), "downs.%d.bias", s);
            auto * dw = ctx->get_tensor(name_w);
            auto * db = ctx->get_tensor(name_b);
            std::vector<float> down_out(next_c * next_h * next_w);
            conv2d_cpu(img.data(), c, cur_h, cur_w, dw, db, next_c, 2, 2, 2, 0, 1, down_out.data());
            img = std::move(down_out);
        }
        cur_c = next_c;
        cur_h = next_h;
        cur_w = next_w;

        fprintf(stderr, "nafnet: enc stage %d done (%dx%d, %d ch)\n", s, cur_w, cur_h, cur_c);
    }

    // Middle blocks
    for (int b = 0; b < ctx->middle_blk_num; b++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "mid.%d", b);
        auto wt = load_block(prefix, cur_c);
        std::vector<float> block_out(cur_c * cur_h * cur_w);
        nafblock_forward(img.data(), cur_c, cur_h, cur_w, wt, block_out.data(), tmp1, tmp2, tmp3);
        img = std::move(block_out);
    }
    fprintf(stderr, "nafnet: middle done (%d blocks)\n", ctx->middle_blk_num);

    // Decoder
    for (int s = 0; s < ns; s++) {
        // Upsample: 1x1 conv (c → 4c/4 * 4 = c for PixelShuffle) then PixelShuffle(2)
        int next_c = cur_c / 2;
        int next_h = cur_h * 2, next_w = cur_w * 2;
        {
            char name_w[64];
            snprintf(name_w, sizeof(name_w), "ups.%d.weight", s);
            auto * uw = ctx->get_tensor(name_w);
            // Conv 1x1: cur_c → cur_c*2 (then PixelShuffle reduces by 4)
            // Actually ups weight is [cur_c*2, cur_c, 1, 1] → output [cur_c*2, h, w]
            // Then PixelShuffle(2): [cur_c*2, h, w] → [cur_c/2, 2h, 2w]
            int up_oc = cur_c * 2;
            std::vector<float> up_tmp(up_oc * cur_h * cur_w);
            conv2d_cpu(img.data(), cur_c, cur_h, cur_w, uw, nullptr, up_oc, 1, 1, 1, 0, 1, up_tmp.data());

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
            nafblock_forward(img.data(), cur_c, cur_h, cur_w, wt, block_out.data(), tmp1, tmp2, tmp3);
            img = std::move(block_out);
        }

        fprintf(stderr, "nafnet: dec stage %d done (%dx%d, %d ch)\n", s, cur_w, cur_h, cur_c);
    }

    // Ending conv: W → 3
    {
        auto * end_w = ctx->get_tensor("ending.weight");
        auto * end_b = ctx->get_tensor("ending.bias");
        std::vector<float> ending(3 * cur_h * cur_w);
        conv2d_cpu(img.data(), W, cur_h, cur_w, end_w, end_b, 3, 3, 3, 1, 1, 1, ending.data());
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

    fprintf(stderr, "nafnet: done (%dx%d)\n", width, height);
    return 0;
}
