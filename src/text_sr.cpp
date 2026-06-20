// text_sr.cpp — Text super-resolution via NAFNet U-Net + PixelShuffle.
//
// Same U-Net backbone as nafnet_denoise (NAFBlocks with SimpleGate + SCA),
// but the ending conv outputs 3*r*r channels → PixelShuffle(r) → upscaled.
// The global residual is a bicubic-upscaled input (not same-size identity).
//
// Large images are processed in overlapping tiles and blended with a raised
// cosine (Hann) window to avoid seam artifacts.

#include "text_sr.h"
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

// MSVC's <cmath> doesn't define M_PI without _USE_MATH_DEFINES (matches
// scan_cleanup.cpp / bttr_ocr.cpp). Define a portable fallback.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <string>
#include <vector>

// ── Helpers ────────────────────────────────────────────────────────────

static void tsr_conv2d(const float * input, int ic, int ih, int iw,
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
                                sum += input[ic_abs * ih * iw + iy * iw + ix]
                                     * weight[oc_abs * ic_per_group * kh * kw
                                              + ic_i * kh * kw + ky * kw + kx];
                            }
                        }
                    }
                    output[oc_abs * oh * ow + oy * ow + ox] = sum;
                }
            }
        }
    }
}

static void tsr_layernorm2d(const float * input, int c, int h, int w,
                            const float * weight, const float * bias,
                            float * output) {
    float eps = 1e-6f;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float mean = 0;
            for (int ch = 0; ch < c; ch++)
                mean += input[ch * h * w + y * w + x];
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

static void tsr_simple_gate(const float * input, int c2, int h, int w, float * output) {
    int c = c2 / 2;
    int hw = h * w;
    for (int ch = 0; ch < c; ch++)
        for (int i = 0; i < hw; i++)
            output[ch * hw + i] = input[ch * hw + i] * input[(ch + c) * hw + i];
}

static void tsr_sca(const float * input, int c, int h, int w,
                    const float * sca_weight, const float * sca_bias,
                    float * output) {
    int hw = h * w;
    std::vector<float> pooled(c, 0.0f);
    for (int ch = 0; ch < c; ch++) {
        float sum = 0;
        for (int i = 0; i < hw; i++) sum += input[ch * hw + i];
        pooled[ch] = sum / hw;
    }
    std::vector<float> attn(c);
    for (int oc = 0; oc < c; oc++) {
        float sum = sca_bias[oc];
        for (int ic = 0; ic < c; ic++)
            sum += sca_weight[oc * c + ic] * pooled[ic];
        attn[oc] = sum;
    }
    for (int ch = 0; ch < c; ch++)
        for (int i = 0; i < hw; i++)
            output[ch * hw + i] = input[ch * hw + i] * attn[ch];
}

static void tsr_pixel_shuffle(const float * input, int c_in, int h, int w,
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

// ── NAFBlock (identical to nafnet_denoise) ─────────────────────────────────

struct tsr_nafblock_weights {
    const float * beta;
    const float * gamma;
    const float * conv1_w, * conv1_b;
    const float * conv2_w, * conv2_b;
    const float * conv3_w, * conv3_b;
    const float * sca_w, * sca_b;
    const float * conv4_w, * conv4_b;
    const float * conv5_w, * conv5_b;
    const float * norm1_w, * norm1_b;
    const float * norm2_w, * norm2_b;
};

static void tsr_nafblock_forward(const float * input, int c, int h, int w,
                                 const tsr_nafblock_weights & wt,
                                 float * output,
                                 std::vector<float> & tmp1,
                                 std::vector<float> & tmp2,
                                 std::vector<float> & tmp3) {
    int hw = h * w;
    int c2 = c * 2;

    // Part 1: spatial mixing
    tmp1.resize(c * hw);
    tsr_layernorm2d(input, c, h, w, wt.norm1_w, wt.norm1_b, tmp1.data());

    tmp2.resize(c2 * hw);
    tsr_conv2d(tmp1.data(), c, h, w, wt.conv1_w, wt.conv1_b, c2, 1, 1, 1, 0, 1, tmp2.data());

    tmp3.resize(c2 * hw);
    tsr_conv2d(tmp2.data(), c2, h, w, wt.conv2_w, wt.conv2_b, c2, 3, 3, 1, 1, c2, tmp3.data());

    tmp1.resize(c * hw);
    tsr_simple_gate(tmp3.data(), c2, h, w, tmp1.data());

    tmp2.resize(c * hw);
    tsr_sca(tmp1.data(), c, h, w, wt.sca_w, wt.sca_b, tmp2.data());

    tmp1.resize(c * hw);
    tsr_conv2d(tmp2.data(), c, h, w, wt.conv3_w, wt.conv3_b, c, 1, 1, 1, 0, 1, tmp1.data());

    // Residual + beta
    tmp2.resize(c * hw);
    for (int ch = 0; ch < c; ch++) {
        float b = wt.beta[ch];
        for (int i = 0; i < hw; i++)
            tmp2[ch * hw + i] = input[ch * hw + i] + tmp1[ch * hw + i] * b;
    }

    // Part 2: channel mixing
    tmp1.resize(c * hw);
    tsr_layernorm2d(tmp2.data(), c, h, w, wt.norm2_w, wt.norm2_b, tmp1.data());

    tmp3.resize(c2 * hw);
    tsr_conv2d(tmp1.data(), c, h, w, wt.conv4_w, wt.conv4_b, c2, 1, 1, 1, 0, 1, tmp3.data());

    tmp1.resize(c * hw);
    tsr_simple_gate(tmp3.data(), c2, h, w, tmp1.data());

    tmp3.resize(c * hw);
    tsr_conv2d(tmp1.data(), c, h, w, wt.conv5_w, wt.conv5_b, c, 1, 1, 1, 0, 1, tmp3.data());

    // Residual + gamma
    for (int ch = 0; ch < c; ch++) {
        float g = wt.gamma[ch];
        for (int i = 0; i < hw; i++)
            output[ch * hw + i] = tmp2[ch * hw + i] + tmp3[ch * hw + i] * g;
    }
}

// ── Bicubic upscale (for the global residual) ──────────────────────────────

// Cubic interpolation kernel (Keys' convolution, a=-0.5)
static float cubic_weight(float x) {
    x = fabsf(x);
    if (x < 1.0f) return (1.5f * x - 2.5f) * x * x + 1.0f;
    if (x < 2.0f) return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    return 0.0f;
}

// Bicubic upscale: [C, H, W] → [C, H*r, W*r]  (planar float)
static void bicubic_upscale(const float * src, int c, int h, int w,
                            int r, float * dst) {
    int oh = h * r, ow = w * r;
    for (int ch = 0; ch < c; ch++) {
        const float * s = src + ch * h * w;
        float * d = dst + ch * oh * ow;
        for (int oy = 0; oy < oh; oy++) {
            float sy = ((float)oy + 0.5f) / r - 0.5f;
            int iy = (int)floorf(sy);
            float fy = sy - iy;
            for (int ox = 0; ox < ow; ox++) {
                float sx = ((float)ox + 0.5f) / r - 0.5f;
                int ix = (int)floorf(sx);
                float fx = sx - ix;
                float val = 0.0f;
                for (int jy = -1; jy <= 2; jy++) {
                    float wy = cubic_weight(fy - jy);
                    int cy = std::max(0, std::min(h - 1, iy + jy));
                    for (int jx = -1; jx <= 2; jx++) {
                        float wx = cubic_weight(fx - jx);
                        int cx = std::max(0, std::min(w - 1, ix + jx));
                        val += wy * wx * s[cy * w + cx];
                    }
                }
                d[oy * ow + ox] = val;
            }
        }
    }
}

// ── Context ────────────────────────────────────────────────────────────────

struct text_sr_context {
    int width;
    int n_stages;
    int upscale_factor;
    std::vector<int> enc_blk_nums;
    std::vector<int> dec_blk_nums;
    int middle_blk_num;
    int n_threads;
    bool bench;

    core_gguf::WeightLoad wl;
    std::string model_path;
    core_cpu::DequantCache dcache;

    const float * get_tensor(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) {
            fprintf(stderr, "text_sr: missing tensor %s\n", name.c_str());
            return nullptr;
        }
        return dcache.get(t);
    }
};

text_sr_context * text_sr_init(const char * model_path, int n_threads) {
    auto * ctx = new text_sr_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    ctx->model_path = model_path;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "text_sr: failed to open %s\n", model_path);
        delete ctx;
        return nullptr;
    }

    ctx->width          = core_gguf::kv_u32(meta, "text_sr.width", 32);
    ctx->n_stages       = core_gguf::kv_u32(meta, "text_sr.n_stages", 4);
    ctx->middle_blk_num = core_gguf::kv_u32(meta, "text_sr.middle_blk_num", 12);
    ctx->upscale_factor = core_gguf::kv_u32(meta, "text_sr.upscale_factor", 2);
    ctx->enc_blk_nums   = core_gguf::kv_i32_array(meta, "text_sr.enc_blk_nums");
    ctx->dec_blk_nums   = core_gguf::kv_i32_array(meta, "text_sr.dec_blk_nums");
    core_gguf::free_metadata(meta);

    if (ctx->upscale_factor != 2 && ctx->upscale_factor != 4) {
        fprintf(stderr, "text_sr: unsupported upscale_factor=%d (must be 2 or 4)\n",
                ctx->upscale_factor);
        delete ctx;
        return nullptr;
    }

    bool force_cpu = (getenv("TEXT_SR_FORCE_CPU") && atoi(getenv("TEXT_SR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, backend, "text_sr", ctx->wl)) {
        fprintf(stderr, "text_sr: failed to load weights\n");
        ggml_backend_free(backend);
        delete ctx;
        return nullptr;
    }
    ggml_backend_free(backend);

    fprintf(stderr, "text_sr: width=%d, upscale=%dx, enc=[", ctx->width, ctx->upscale_factor);
    for (int i = 0; i < (int)ctx->enc_blk_nums.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", ctx->enc_blk_nums[i]);
    fprintf(stderr, "], mid=%d, dec=[", ctx->middle_blk_num);
    for (int i = 0; i < (int)ctx->dec_blk_nums.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", ctx->dec_blk_nums[i]);
    fprintf(stderr, "], %d tensors\n", (int)ctx->wl.tensors.size());

    ctx->bench = (std::getenv("CRISPEMBED_TEXT_SR_BENCH") != nullptr);
    return ctx;
}

void text_sr_free(text_sr_context * ctx) {
    if (ctx) {
        core_gguf::free_weights(ctx->wl);
        delete ctx;
    }
}

int text_sr_upscale_factor(const text_sr_context * ctx) {
    return ctx ? ctx->upscale_factor : 0;
}

// ── Single-tile SR forward pass ────────────────────────────────────────────

// Process one tile through the U-Net + PixelShuffle ending.
// Input:  [3, th, tw] float [0,1]
// Output: [3, th*r, tw*r] float [0,1]
static void sr_forward_tile(text_sr_context * ctx,
                            const float * tile_in, int tw, int th,
                            float * tile_out) {
    int W  = ctx->width;
    int ns = ctx->n_stages;
    int r  = ctx->upscale_factor;

    // Pad to multiple of 2^n_stages
    int pad_mult = 1 << ns;
    int ph = ((th + pad_mult - 1) / pad_mult) * pad_mult;
    int pw = ((tw + pad_mult - 1) / pad_mult) * pad_mult;

    // Copy input to padded buffer (reflect-pad edges)
    std::vector<float> img(3 * ph * pw, 0.0f);
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < ph; y++) {
            int sy = std::min(y, th - 1);
            for (int x = 0; x < pw; x++) {
                int sx = std::min(x, tw - 1);
                img[c * ph * pw + y * pw + x] = tile_in[c * th * tw + sy * tw + sx];
            }
        }
    }

    // Save input for bicubic residual
    std::vector<float> input_save = img;

    std::vector<float> tmp1, tmp2, tmp3;

    auto load_block = [&](const std::string & prefix) -> tsr_nafblock_weights {
        tsr_nafblock_weights wt;
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

    // Intro conv: 3 -> W
    {
        auto * intro_w = ctx->get_tensor("intro.weight");
        auto * intro_b = ctx->get_tensor("intro.bias");
        std::vector<float> after_intro(W * cur_h * cur_w);
        tsr_conv2d(img.data(), 3, cur_h, cur_w, intro_w, intro_b, W, 3, 3, 1, 1, 1, after_intro.data());
        img = std::move(after_intro);
    }

    // Encoder
    std::vector<std::vector<float>> skip_connections;
    for (int s = 0; s < ns; s++) {
        int c = W * (1 << s);
        for (int b = 0; b < ctx->enc_blk_nums[s]; b++) {
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "enc.%d.%d", s, b);
            auto wt = load_block(prefix);
            std::vector<float> block_out(c * cur_h * cur_w);
            tsr_nafblock_forward(img.data(), c, cur_h, cur_w, wt, block_out.data(), tmp1, tmp2, tmp3);
            img = std::move(block_out);
        }
        skip_connections.push_back(img);

        int next_c = c * 2;
        int next_h = cur_h / 2, next_w = cur_w / 2;
        {
            char name_w[64], name_b[64];
            snprintf(name_w, sizeof(name_w), "downs.%d.weight", s);
            snprintf(name_b, sizeof(name_b), "downs.%d.bias", s);
            auto * dw = ctx->get_tensor(name_w);
            auto * db = ctx->get_tensor(name_b);
            std::vector<float> down_out(next_c * next_h * next_w);
            tsr_conv2d(img.data(), c, cur_h, cur_w, dw, db, next_c, 2, 2, 2, 0, 1, down_out.data());
            img = std::move(down_out);
        }
        cur_h = next_h;
        cur_w = next_w;
    }

    // Middle
    int cur_c = W * (1 << ns);
    for (int b = 0; b < ctx->middle_blk_num; b++) {
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "mid.%d", b);
        auto wt = load_block(prefix);
        std::vector<float> block_out(cur_c * cur_h * cur_w);
        tsr_nafblock_forward(img.data(), cur_c, cur_h, cur_w, wt, block_out.data(), tmp1, tmp2, tmp3);
        img = std::move(block_out);
    }

    // Decoder
    for (int s = 0; s < ns; s++) {
        int next_c = cur_c / 2;
        int next_h = cur_h * 2, next_w = cur_w * 2;
        {
            char name_w[64];
            snprintf(name_w, sizeof(name_w), "ups.%d.weight", s);
            auto * uw = ctx->get_tensor(name_w);
            int up_oc = cur_c * 2;
            std::vector<float> up_tmp(up_oc * cur_h * cur_w);
            tsr_conv2d(img.data(), cur_c, cur_h, cur_w, uw, nullptr, up_oc, 1, 1, 1, 0, 1, up_tmp.data());
            std::vector<float> ps_out(next_c * next_h * next_w);
            tsr_pixel_shuffle(up_tmp.data(), up_oc, cur_h, cur_w, 2, ps_out.data());
            img = std::move(ps_out);
        }

        auto & skip = skip_connections[ns - 1 - s];
        for (int i = 0; i < next_c * next_h * next_w; i++)
            img[i] += skip[i];

        cur_c = next_c;
        cur_h = next_h;
        cur_w = next_w;

        for (int b = 0; b < ctx->dec_blk_nums[s]; b++) {
            char prefix[64];
            snprintf(prefix, sizeof(prefix), "dec.%d.%d", s, b);
            auto wt = load_block(prefix);
            std::vector<float> block_out(cur_c * cur_h * cur_w);
            tsr_nafblock_forward(img.data(), cur_c, cur_h, cur_w, wt, block_out.data(), tmp1, tmp2, tmp3);
            img = std::move(block_out);
        }
    }

    // Ending conv: W -> 3*r*r, then PixelShuffle(r)
    {
        int ending_oc = 3 * r * r;
        auto * end_w = ctx->get_tensor("ending.weight");
        auto * end_b = ctx->get_tensor("ending.bias");
        std::vector<float> ending(ending_oc * cur_h * cur_w);
        tsr_conv2d(img.data(), W, cur_h, cur_w, end_w, end_b, ending_oc, 3, 3, 1, 1, 1, ending.data());

        // PixelShuffle: [3*r*r, ph, pw] -> [3, ph*r, pw*r]
        int out_h = ph * r, out_w = pw * r;
        std::vector<float> ps_out(3 * out_h * out_w);
        tsr_pixel_shuffle(ending.data(), ending_oc, cur_h, cur_w, r, ps_out.data());
        img = std::move(ps_out);
    }

    // Add bicubic-upscaled input as global residual
    {
        int out_h = ph * r, out_w = pw * r;
        std::vector<float> bicubic(3 * out_h * out_w);
        bicubic_upscale(input_save.data(), 3, ph, pw, r, bicubic.data());
        for (int i = 0; i < 3 * out_h * out_w; i++)
            img[i] += bicubic[i];
    }

    // Crop to [3, th*r, tw*r] and write output
    int out_h = th * r, out_w = tw * r;
    int full_h = ph * r, full_w = pw * r;
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                tile_out[c * out_h * out_w + y * out_w + x] =
                    img[c * full_h * full_w + y * full_w + x];
            }
        }
    }
}

// ── Tiled processing with Hann-window blending ────────────────────────────

// Build a 2D raised-cosine (Hann) blending window for overlap regions.
static void build_blend_window(int tile_size, int overlap, std::vector<float> & win) {
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

int text_sr_process(text_sr_context * ctx,
                    const uint8_t * input, int width, int height,
                    int tile_size, int tile_overlap,
                    uint8_t ** output, int * out_width, int * out_height) {
    if (!ctx || !input || !output || width <= 0 || height <= 0) return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    int r = ctx->upscale_factor;
    if (tile_size <= 0) tile_size = 256;
    if (tile_overlap <= 0) tile_overlap = 32;
    tile_overlap = std::min(tile_overlap, tile_size / 4);

    int ow = width * r;
    int oh = height * r;
    int out_tile = tile_size * r;
    int out_overlap = tile_overlap * r;

    // Accumulator + weight map for blending
    std::vector<float> accum(3 * oh * ow, 0.0f);
    std::vector<float> weight_map(oh * ow, 0.0f);

    // Build blend window at output resolution
    std::vector<float> blend_win;
    build_blend_window(out_tile, out_overlap, blend_win);

    // Convert full input to [3, H, W] float [0,1]
    std::vector<float> full_input(3 * height * width);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                full_input[c * height * width + y * width + x] =
                    input[(y * width + x) * 3 + c] / 255.0f;

    int step = tile_size - tile_overlap;
    int n_tiles_x = std::max(1, (width + step - 1) / step);
    int n_tiles_y = std::max(1, (height + step - 1) / step);

    fprintf(stderr, "text_sr: %dx%d -> %dx%d (%dx), tiles=%dx%d (size=%d, overlap=%d)\n",
            width, height, ow, oh, r, n_tiles_x, n_tiles_y, tile_size, tile_overlap);

    for (int ty = 0; ty < n_tiles_y; ty++) {
        for (int tx = 0; tx < n_tiles_x; tx++) {
            int x0 = std::min(tx * step, std::max(0, width - tile_size));
            int y0 = std::min(ty * step, std::max(0, height - tile_size));
            int tw = std::min(tile_size, width - x0);
            int th = std::min(tile_size, height - y0);

            // Extract tile
            std::vector<float> tile_in(3 * th * tw);
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < th; y++)
                    for (int x = 0; x < tw; x++)
                        tile_in[c * th * tw + y * tw + x] =
                            full_input[c * height * width + (y0 + y) * width + (x0 + x)];

            // SR forward pass
            int otw = tw * r, oth = th * r;
            std::vector<float> tile_out(3 * oth * otw);
            auto t_tile = std::chrono::steady_clock::now();
            sr_forward_tile(ctx, tile_in.data(), tw, th, tile_out.data());
            if (bench) {
                auto t_tile_end = std::chrono::steady_clock::now();
                fprintf(stderr, "[text_sr-bench] tile %d,%d: %.1f ms\n",
                        ty, tx, ms_f(t_tile_end - t_tile).count());
            }

            // Blend into accumulator
            int ox0 = x0 * r, oy0 = y0 * r;
            for (int y = 0; y < oth; y++) {
                for (int x = 0; x < otw; x++) {
                    // Use blend window if tile is full size, else weight=1
                    float w = 1.0f;
                    if (tw == tile_size && th == tile_size)
                        w = blend_win[y * out_tile + x];
                    else {
                        // Partial tile at image edge — apply Hann ramp at overlapping edges
                        if (x0 > 0 && x < out_overlap)
                            w *= 0.5f - 0.5f * cosf((float)M_PI * x / out_overlap);
                        if (y0 > 0 && y < out_overlap)
                            w *= 0.5f - 0.5f * cosf((float)M_PI * y / out_overlap);
                    }

                    int dy = oy0 + y, dx = ox0 + x;
                    if (dy >= oh || dx >= ow) continue;
                    for (int c = 0; c < 3; c++)
                        accum[c * oh * ow + dy * ow + dx] += tile_out[c * oth * otw + y * otw + x] * w;
                    weight_map[dy * ow + dx] += w;
                }
            }

            fprintf(stderr, "text_sr: tile [%d,%d] done\n", tx, ty);
        }
    }

    // Normalize by weights and convert to uint8
    uint8_t * out_buf = (uint8_t *)malloc(3 * oh * ow);
    if (!out_buf) return -1;

    for (int y = 0; y < oh; y++) {
        for (int x = 0; x < ow; x++) {
            float w = weight_map[y * ow + x];
            if (w <= 0.0f) w = 1.0f;
            for (int c = 0; c < 3; c++) {
                float v = accum[c * oh * ow + y * ow + x] / w * 255.0f;
                out_buf[(y * ow + x) * 3 + c] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
        }
    }

    *output = out_buf;
    *out_width = ow;
    *out_height = oh;
    fprintf(stderr, "text_sr: done (%dx%d)\n", ow, oh);
    if (bench) {
        auto t_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[text_sr-bench] total: %.1f ms\n",
                ms_f(t_end - t_total).count());
    }
    return 0;
}

void text_sr_free_image(uint8_t * pixels) {
    free(pixels);
}
