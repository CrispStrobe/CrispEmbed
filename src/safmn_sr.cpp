// safmn_sr.cpp — SAFMN super-resolution (CPU-scalar implementation).
//
// Architecture (lightweight x4, 228K params):
//   to_feat: Conv3x3(3→36, pad=1)
//   8× AttBlock:
//     LayerNorm → SAFM (multi-scale DW-Conv + 1x1 aggr + GELU + gate) → residual
//     LayerNorm → CCM  (Conv3x3 + GELU + Conv1x1) → residual
//   Global skip
//   to_img: Conv3x3(36→48, pad=1) + PixelShuffle(4)
//
// SAFM: splits channels into 4 chunks, processes each at different scales
// (full, 1/2, 1/4, 1/8) via DW-Conv3x3, upsamples back, concatenates,
// mixes with 1x1 conv, applies GELU, then element-wise multiplies by input.

#include "safmn_sr.h"
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

// ── Helpers ────────────────────────────────────────────────────────

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

// Conv2d: [OC, IC/groups, KH, KW] weights. Input/output [C, H, W] planar.
static void conv2d(const float * input, int ic, int ih, int iw,
                   const float * weight, const float * bias,
                   int oc, int kh, int kw, int pad, int groups,
                   float * output) {
    int oh = ih + 2 * pad - kh + 1;
    int ow = iw + 2 * pad - kw + 1;
    int ic_pg = ic / groups, oc_pg = oc / groups;
    for (int g = 0; g < groups; g++) {
        for (int o = 0; o < oc_pg; o++) {
            int oc_abs = g * oc_pg + o;
            float b = bias ? bias[oc_abs] : 0.0f;
            for (int oy = 0; oy < oh; oy++) {
                for (int ox = 0; ox < ow; ox++) {
                    float sum = b;
                    for (int c = 0; c < ic_pg; c++) {
                        int ic_abs = g * ic_pg + c;
                        for (int ky = 0; ky < kh; ky++) {
                            for (int kx = 0; kx < kw; kx++) {
                                int iy = oy + ky - pad, ix = ox + kx - pad;
                                if (iy < 0 || iy >= ih || ix < 0 || ix >= iw) continue;
                                sum += input[ic_abs * ih * iw + iy * iw + ix]
                                     * weight[oc_abs * ic_pg * kh * kw + c * kh * kw + ky * kw + kx];
                            }
                        }
                    }
                    output[oc_abs * oh * ow + oy * ow + ox] = sum;
                }
            }
        }
    }
}

// Channel-first LayerNorm: normalize over C for each spatial position.
static void layernorm_chw(const float * input, int c, int h, int w,
                          const float * weight, const float * bias,
                          float * output) {
    int hw = h * w;
    float eps = 1e-6f;
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float mean = 0;
            for (int ch = 0; ch < c; ch++)
                mean += input[ch * hw + y * w + x];
            mean /= c;
            float var = 0;
            for (int ch = 0; ch < c; ch++) {
                float d = input[ch * hw + y * w + x] - mean;
                var += d * d;
            }
            var /= c;
            float inv_std = 1.0f / sqrtf(var + eps);
            for (int ch = 0; ch < c; ch++) {
                float v = (input[ch * hw + y * w + x] - mean) * inv_std;
                output[ch * hw + y * w + x] = v * weight[ch] + bias[ch];
            }
        }
    }
}

static void gelu_inplace(float * data, int n) {
    for (int i = 0; i < n; i++) {
        float x = data[i];
        // Exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        data[i] = x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
    }
}

// Adaptive max pool 2D: [C, H, W] → [C, oh, ow]
static void adaptive_max_pool2d(const float * input, int c, int h, int w,
                                 int oh, int ow, float * output) {
    for (int ch = 0; ch < c; ch++) {
        for (int oy = 0; oy < oh; oy++) {
            int y0 = oy * h / oh;
            int y1 = (oy + 1) * h / oh;
            if (y1 <= y0) y1 = y0 + 1;
            for (int ox = 0; ox < ow; ox++) {
                int x0 = ox * w / ow;
                int x1 = (ox + 1) * w / ow;
                if (x1 <= x0) x1 = x0 + 1;
                float m = -1e30f;
                for (int y = y0; y < y1 && y < h; y++)
                    for (int x = x0; x < x1 && x < w; x++)
                        m = std::max(m, input[ch * h * w + y * w + x]);
                output[ch * oh * ow + oy * ow + ox] = m;
            }
        }
    }
}

// Nearest-neighbor upsample: [C, sh, sw] → [C, th, tw]
static void nearest_upsample(const float * input, int c, int sh, int sw,
                              int th, int tw, float * output) {
    for (int ch = 0; ch < c; ch++)
        for (int y = 0; y < th; y++)
            for (int x = 0; x < tw; x++)
                output[ch * th * tw + y * tw + x] =
                    input[ch * sh * sw + (y * sh / th) * sw + (x * sw / tw)];
}

// PixelShuffle: [C*r*r, H, W] → [C, H*r, W*r]
static void pixel_shuffle(const float * input, int c_in, int h, int w,
                           int r, float * output) {
    int c_out = c_in / (r * r);
    int oh = h * r, ow = w * r;
    for (int c = 0; c < c_out; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                int iy = y / r, ix = x / r;
                int ry = y % r, rx = x % r;
                int ic = c * r * r + ry * r + rx;
                output[c * oh * ow + y * ow + x] = input[ic * h * w + iy * w + ix];
            }
}

// ── Model context ──────────────────────────────────────────────────

struct safm_weights {
    ggml_tensor * mfr_w[4]; // DW-Conv3x3 weights [chunk_dim, 1, 3, 3]
    ggml_tensor * mfr_b[4]; // [chunk_dim]
    ggml_tensor * aggr_w;   // Conv1x1 [dim, dim, 1, 1]
    ggml_tensor * aggr_b;   // [dim]
};

struct ccm_weights {
    ggml_tensor * conv1_w; // Conv3x3 [hidden, dim, 3, 3]
    ggml_tensor * conv1_b;
    ggml_tensor * conv2_w; // Conv1x1 [dim, hidden, 1, 1]
    ggml_tensor * conv2_b;
};

struct attblock_weights {
    ggml_tensor * norm1_w, * norm1_b;
    safm_weights safm;
    ggml_tensor * norm2_w, * norm2_b;
    ccm_weights ccm;
};

struct safmn_context {
    ggml_context * gguf_ctx;
    ggml_backend_buffer_t gguf_buf;

    int scale, dim, n_blocks, n_levels;
    bool bench;

    ggml_tensor * to_feat_w, * to_feat_b;
    std::vector<attblock_weights> blocks;
    ggml_tensor * to_img_w, * to_img_b;
};

safmn_context * safmn_init(const char * model_path, int n_threads) {
    (void)n_threads;
    if (!model_path) return nullptr;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) return nullptr;

    int scale    = (int)core_gguf::kv_u32(meta, "safmn.scale", 4);
    int dim      = (int)core_gguf::kv_u32(meta, "safmn.dim", 36);
    int n_blocks = (int)core_gguf::kv_u32(meta, "safmn.n_blocks", 8);
    int n_levels = (int)core_gguf::kv_u32(meta, "safmn.n_levels", 4);
    core_gguf::free_metadata(meta);

    bool force_cpu = (getenv("SAFMN_SR_FORCE_CPU") && atoi(getenv("SAFMN_SR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();

    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, backend, "safmn", wl)) {
        ggml_backend_free(backend);
        return nullptr;
    }
    ggml_backend_free(backend);

    auto * ctx = new safmn_context;
    ctx->gguf_ctx = wl.ctx;
    ctx->gguf_buf = wl.buf;
    ctx->scale = scale;
    ctx->dim = dim;
    ctx->n_blocks = n_blocks;
    ctx->n_levels = n_levels;

    auto get = [&](const char * name) -> ggml_tensor * {
        return core_gguf::require(wl.tensors, name, "safmn");
    };

    ctx->to_feat_w = get("to_feat.weight");
    ctx->to_feat_b = get("to_feat.bias");
    ctx->to_img_w  = get("to_img.0.weight");
    ctx->to_img_b  = get("to_img.0.bias");

    ctx->blocks.resize(n_blocks);
    for (int i = 0; i < n_blocks; i++) {
        char buf[128];
        auto k = [&](const char * suffix) {
            snprintf(buf, sizeof(buf), "feats.%d.%s", i, suffix);
            return get(buf);
        };
        auto & b = ctx->blocks[i];
        b.norm1_w = k("norm1.weight"); b.norm1_b = k("norm1.bias");
        b.norm2_w = k("norm2.weight"); b.norm2_b = k("norm2.bias");
        for (int j = 0; j < n_levels; j++) {
            snprintf(buf, sizeof(buf), "feats.%d.safm.mfr.%d.weight", i, j);
            b.safm.mfr_w[j] = get(buf);
            snprintf(buf, sizeof(buf), "feats.%d.safm.mfr.%d.bias", i, j);
            b.safm.mfr_b[j] = get(buf);
        }
        b.safm.aggr_w = k("safm.aggr.weight");
        b.safm.aggr_b = k("safm.aggr.bias");
        b.ccm.conv1_w = k("ccm.ccm.0.weight");
        b.ccm.conv1_b = k("ccm.ccm.0.bias");
        b.ccm.conv2_w = k("ccm.ccm.2.weight");
        b.ccm.conv2_b = k("ccm.ccm.2.bias");
    }

    ctx->bench = (std::getenv("CRISPEMBED_SAFMN_SR_BENCH") != nullptr);
    return ctx;
}

void safmn_free(safmn_context * ctx) {
    if (!ctx) return;
    core_gguf::WeightLoad wl;
    wl.ctx = ctx->gguf_ctx;
    wl.buf = ctx->gguf_buf;
    core_gguf::free_weights(wl);
    delete ctx;
}

int safmn_get_scale(const safmn_context * ctx) {
    return ctx ? ctx->scale : 0;
}

// ── Forward pass ───────────────────────────────────────────────────

// Stage dump callback for parity testing (set via env var or null)
struct safmn_dump {
    const char * ref_path;
    std::vector<std::pair<std::string, std::vector<float>>> stages;
};

int safmn_process_float(safmn_context * ctx,
                        const float * input_chw, int width, int height,
                        float * output_chw) {
    if (!ctx || !input_chw || !output_chw || width <= 0 || height <= 0)
        return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    const int C = ctx->dim;
    const int H = height, W = width;
    const int hw = H * W;
    std::vector<float> dq1, dq2;

    // to_feat: Conv3x3(3→C, pad=1)
    std::vector<float> x(C * hw);
    conv2d(input_chw, 3, H, W,
           to_f32(ctx->to_feat_w, dq1), to_f32(ctx->to_feat_b, dq2),
           C, 3, 3, 1, 1, x.data());

    // Save global residual
    std::vector<float> residual(x.begin(), x.end());

    // Scratch buffers
    std::vector<float> norm_buf(C * hw);
    std::vector<float> safm_buf, safm_cat(C * hw);
    std::vector<float> pool_buf, conv_tmp, up_buf;
    std::vector<float> ccm_buf;
    int chunk_dim = C / ctx->n_levels;

    // 8 AttBlocks
    for (int bi = 0; bi < ctx->n_blocks; bi++) {
        auto t_blk = std::chrono::steady_clock::now();
        const auto & blk = ctx->blocks[bi];

        // ── SAFM branch ──
        // LayerNorm
        layernorm_chw(x.data(), C, H, W,
                      to_f32(blk.norm1_w, dq1), to_f32(blk.norm1_b, dq2),
                      norm_buf.data());

        // Multi-scale feature modulation
        for (int lv = 0; lv < ctx->n_levels; lv++) {
            const float * chunk_in = norm_buf.data() + lv * chunk_dim * hw;
            float * chunk_out = safm_cat.data() + lv * chunk_dim * hw;
            const float * mfr_w = to_f32(blk.safm.mfr_w[lv], dq1);
            const float * mfr_b = to_f32(blk.safm.mfr_b[lv], dq2);

            if (lv == 0) {
                // Full resolution DW-Conv3x3
                conv2d(chunk_in, chunk_dim, H, W,
                       mfr_w, mfr_b,
                       chunk_dim, 3, 3, 1, chunk_dim, chunk_out);
            } else {
                // Pool → DW-Conv → Upsample
                int s = 1 << lv; // 2, 4, 8
                int ph = std::max(1, H / s), pw = std::max(1, W / s);
                pool_buf.resize(chunk_dim * ph * pw);
                adaptive_max_pool2d(chunk_in, chunk_dim, H, W, ph, pw, pool_buf.data());

                conv_tmp.resize(chunk_dim * ph * pw);
                conv2d(pool_buf.data(), chunk_dim, ph, pw,
                       mfr_w, mfr_b,
                       chunk_dim, 3, 3, 1, chunk_dim, conv_tmp.data());

                nearest_upsample(conv_tmp.data(), chunk_dim, ph, pw, H, W, chunk_out);
            }
        }

        // 1x1 conv (aggr) + GELU + element-wise gate
        safm_buf.resize(C * hw);
        conv2d(safm_cat.data(), C, H, W,
               to_f32(blk.safm.aggr_w, dq1), to_f32(blk.safm.aggr_b, dq2),
               C, 1, 1, 0, 1, safm_buf.data());
        gelu_inplace(safm_buf.data(), C * hw);

        // Gate: safm_buf * norm_buf (element-wise)
        for (int i = 0; i < C * hw; i++)
            safm_buf[i] *= norm_buf[i];

        // Residual add
        for (int i = 0; i < C * hw; i++)
            x[i] += safm_buf[i];

        // ── CCM branch ──
        layernorm_chw(x.data(), C, H, W,
                      to_f32(blk.norm2_w, dq1), to_f32(blk.norm2_b, dq2),
                      norm_buf.data());

        int hidden = C * 2; // ffn_scale=2
        ccm_buf.resize(hidden * hw);
        conv2d(norm_buf.data(), C, H, W,
               to_f32(blk.ccm.conv1_w, dq1), to_f32(blk.ccm.conv1_b, dq2),
               hidden, 3, 3, 1, 1, ccm_buf.data());
        gelu_inplace(ccm_buf.data(), hidden * hw);

        conv2d(ccm_buf.data(), hidden, H, W,
               to_f32(blk.ccm.conv2_w, dq1), to_f32(blk.ccm.conv2_b, dq2),
               C, 1, 1, 0, 1, norm_buf.data());

        for (int i = 0; i < C * hw; i++)
            x[i] += norm_buf[i];
        if (bench) {
            auto t_blk_end = std::chrono::steady_clock::now();
            fprintf(stderr, "[safmn_sr-bench] block %d: %.1f ms\n",
                    bi, ms_f(t_blk_end - t_blk).count());
        }
    }

    // Global skip
    for (int i = 0; i < C * hw; i++)
        x[i] += residual[i];

    // to_img: Conv3x3 → PixelShuffle
    int out_ch = 3 * ctx->scale * ctx->scale;
    std::vector<float> pre_shuffle(out_ch * hw);
    conv2d(x.data(), C, H, W,
           to_f32(ctx->to_img_w, dq1), to_f32(ctx->to_img_b, dq2),
           out_ch, 3, 3, 1, 1, pre_shuffle.data());

    int out_h = H * ctx->scale, out_w = W * ctx->scale;
    pixel_shuffle(pre_shuffle.data(), out_ch, H, W, ctx->scale, output_chw);

    if (bench) {
        auto t_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[safmn_sr-bench] total: %.1f ms\n",
                ms_f(t_end - t_total).count());
    }
    return 0;
}

int safmn_process(safmn_context * ctx,
                  const uint8_t * input, int width, int height,
                  uint8_t * output) {
    if (!ctx || !input || !output) return -1;

    int hw = width * height;

    // Convert uint8 HWC → float CHW [0,1]
    std::vector<float> in_chw(3 * hw);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                in_chw[c * hw + y * width + x] =
                    (float)input[(y * width + x) * 3 + c] / 255.0f;

    int out_h = height * ctx->scale, out_w = width * ctx->scale;
    int out_hw = out_h * out_w;
    std::vector<float> out_chw(3 * out_hw);

    int ret = safmn_process_float(ctx, in_chw.data(), width, height, out_chw.data());
    if (ret != 0) return ret;

    // Convert float CHW → uint8 HWC, clamped [0, 255]
    for (int y = 0; y < out_h; y++)
        for (int x = 0; x < out_w; x++)
            for (int c = 0; c < 3; c++) {
                float v = out_chw[c * out_hw + y * out_w + x] * 255.0f;
                output[(y * out_w + x) * 3 + c] =
                    (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }

    return 0;
}
