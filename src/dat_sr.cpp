// dat_sr.cpp — DAT (Dual Aggregation Transformer) super-resolution.
//
// CPU-scalar forward pass for DAT-light x2 (Apache-2.0, ICCV 2023).
// Architecture: Conv3→LN→18×DATB→LN→3conv+skip→PixelShuffleDirect.
//
// DATB blocks alternate:
//   Even: Adaptive Spatial Attention (split-channel windowed, with AIM)
//   Odd:  Adaptive Channel Attention (transposed, L2-normalized, with AIM)
// Both followed by SGFN (Spatial-Gate Feed-Forward Network).
//
// Reference: https://github.com/zhengchen1999/DAT
// Tiling with overlap for large images (same pattern as pan_sr.cpp).

#include "dat_sr.h"
#include "core/gguf_loader.h"
#include "core/cpu_ops.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Dequantization helper
// ---------------------------------------------------------------------------

static const float * to_f32(const ggml_tensor * t, std::vector<float> & buf) {
    if (t->type == GGML_TYPE_F32) return (const float *)t->data;
    int64_t n = ggml_nelements(t);
    buf.resize(n);
    if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)t->data;
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) traits->to_float(t->data, buf.data(), n);
        else std::fill(buf.begin(), buf.end(), 0.0f);
    }
    return buf.data();
}

// ---------------------------------------------------------------------------
// CPU-scalar ops
// ---------------------------------------------------------------------------

static void linear_cpu(const float * x, int n_in, const float * w, const float * b,
                        int n_out, float * y) {
    core_cpu::linear_cpu(x, y, n_in, n_out, w, b);
}

static void layernorm_cpu(const float * x, int d, const float * g, const float * b,
                           float * y, float eps = 1e-5f) {
    float mean = 0;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= d;
    float var = 0;
    for (int i = 0; i < d; i++) { float dx = x[i] - mean; var += dx * dx; }
    float inv = 1.0f / sqrtf(var / d + eps);
    for (int i = 0; i < d; i++) y[i] = (x[i] - mean) * inv * g[i] + b[i];
}

static float gelu_f(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

static void leaky_relu(float * d, int n, float slope = 0.2f) {
    for (int i = 0; i < n; i++) if (d[i] < 0) d[i] *= slope;
}

// Conv2d(C_in, C_out, 3, 1, 1) — w is (C_out, C_in*9) row-major
static void conv2d_3x3(const float * in, int C_in, int H, int W,
                        const float * w, const float * b, int C_out, float * out) {
    for (int co = 0; co < C_out; co++) {
        float bias = b ? b[co] : 0.0f;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float sum = bias;
                for (int ci = 0; ci < C_in; ci++) {
                    for (int ky = -1; ky <= 1; ky++) {
                        int sy = y + ky;
                        if (sy < 0 || sy >= H) continue;
                        for (int kx = -1; kx <= 1; kx++) {
                            int sx = x + kx;
                            if (sx < 0 || sx >= W) continue;
                            sum += in[(ci*H + sy)*W + sx] * w[co*C_in*9 + ci*9 + (ky+1)*3 + (kx+1)];
                        }
                    }
                }
                out[(co*H + y)*W + x] = sum;
            }
        }
    }
}

// Conv2d(C, C, 3, 1, 1, groups=C) — depthwise, w is (C, 9)
static void dwconv2d_3x3(const float * in, int C, int H, int W,
                          const float * w, const float * b, float * out) {
    for (int c = 0; c < C; c++) {
        float bias = b ? b[c] : 0.0f;
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                float sum = bias;
                for (int ky = -1; ky <= 1; ky++) {
                    int sy = y + ky;
                    if (sy < 0 || sy >= H) continue;
                    for (int kx = -1; kx <= 1; kx++) {
                        int sx = x + kx;
                        if (sx < 0 || sx >= W) continue;
                        sum += in[(c*H + sy)*W + sx] * w[c*9 + (ky+1)*3 + (kx+1)];
                    }
                }
                out[(c*H + y)*W + x] = sum;
            }
        }
    }
}

// Conv2d(C_in, C_out, 1, 1, 0) — w is (C_out, C_in), same as linear per pixel
static void conv1x1(const float * in, int C_in, int H, int W,
                     const float * w, const float * b, int C_out, float * out) {
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            // Gather input pixel
            for (int co = 0; co < C_out; co++) {
                float sum = b ? b[co] : 0.0f;
                for (int ci = 0; ci < C_in; ci++)
                    sum += in[(ci*H + y)*W + x] * w[co*C_in + ci];
                out[(co*H + y)*W + x] = sum;
            }
        }
    }
}

// BatchNorm2d in eval mode: y = (x - mean) / sqrt(var + eps) * weight + bias
// Applied in-place, CHW layout
static void batchnorm_eval(float * data, int C, int H, int W,
                            const float * weight, const float * bias,
                            const float * running_mean, const float * running_var,
                            float eps = 1e-5f) {
    for (int c = 0; c < C; c++) {
        float scale = weight[c] / sqrtf(running_var[c] + eps);
        float shift = bias[c] - running_mean[c] * scale;
        for (int i = 0; i < H * W; i++)
            data[c * H * W + i] = data[c * H * W + i] * scale + shift;
    }
}

// Adaptive average pool to 1x1 — global average per channel
static void adaptive_avg_pool_1x1(const float * in, int C, int HW, float * out) {
    for (int c = 0; c < C; c++) {
        float sum = 0;
        for (int i = 0; i < HW; i++) sum += in[c * HW + i];
        out[c] = sum / HW;
    }
}

// Sigmoid
static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

static void pixel_shuffle(const float * in, int C, int H, int W, int r, float * out) {
    int C_out = C / (r * r);
    for (int c = 0; c < C_out; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                for (int ry = 0; ry < r; ry++)
                    for (int rx = 0; rx < r; rx++) {
                        int ci = c * r * r + ry * r + rx;
                        out[(c * H*r + y*r + ry) * W*r + x*r + rx] = in[(ci * H + y) * W + x];
                    }
}

// L2-normalize each row of (rows, cols) in-place
static void l2_normalize_rows(float * data, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float * row = data + r * cols;
        float norm = 0;
        for (int c = 0; c < cols; c++) norm += row[c] * row[c];
        norm = 1.0f / (sqrtf(norm) + 1e-12f);
        for (int c = 0; c < cols; c++) row[c] *= norm;
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct dat_sr_context {
    int embed_dim;
    int upscale;
    int num_heads;
    int split_size[2]; // [H_sp, W_sp] for branch 0
    std::vector<int> depth;
    std::string resi_connection;
    std::string upsampler;

    core_gguf::WeightLoad wl;
    std::unordered_map<std::string, ggml_tensor*> tensors;
    std::map<const void *, std::vector<float>> dequant_cache;

    float mean[3];
    float img_range;
    bool bench;
};

static const float * get_w(dat_sr_context * ctx, const std::string & name) {
    auto it = ctx->tensors.find(name);
    if (it == ctx->tensors.end()) return nullptr;
    auto cit = ctx->dequant_cache.find(it->second->data);
    if (cit != ctx->dequant_cache.end()) return cit->second.data();
    std::vector<float> buf;
    const float * p = to_f32(it->second, buf);
    if (p != (const float *)it->second->data) {
        ctx->dequant_cache[it->second->data] = std::move(buf);
        return ctx->dequant_cache[it->second->data].data();
    }
    return p;
}

static int64_t get_dim(dat_sr_context * ctx, const std::string & name, int axis) {
    auto it = ctx->tensors.find(name);
    if (it == ctx->tensors.end()) return 0;
    return it->second->ne[axis];
}

// Helper to build prefixed weight name
static std::string pfx(int rg, int blk, const char * suffix) {
    char buf[256];
    snprintf(buf, sizeof(buf), "layers.%d.blocks.%d.%s", rg, blk, suffix);
    return buf;
}

// ---------------------------------------------------------------------------
// Window partition / unpartition
// ---------------------------------------------------------------------------

// (N_total, C) → (nW, win_N, C) where win_N = H_sp * W_sp
// Input is in (H, W, C) order (N_total = H * W)
static void window_partition(const float * x, int H, int W, int C,
                              int H_sp, int W_sp, float * out) {
    int nH = H / H_sp, nW = W / W_sp;
    // out layout: (nH * nW, H_sp * W_sp, C)
    for (int wh = 0; wh < nH; wh++) {
        for (int ww = 0; ww < nW; ww++) {
            int win_idx = wh * nW + ww;
            for (int yy = 0; yy < H_sp; yy++) {
                for (int xx = 0; xx < W_sp; xx++) {
                    int src_y = wh * H_sp + yy;
                    int src_x = ww * W_sp + xx;
                    int token = yy * W_sp + xx;
                    const float * src = x + (src_y * W + src_x) * C;
                    float * dst = out + (win_idx * H_sp * W_sp + token) * C;
                    memcpy(dst, src, C * sizeof(float));
                }
            }
        }
    }
}

// (nW, win_N, C) → (H, W, C)
static void window_unpartition(const float * x, int H, int W, int C,
                                int H_sp, int W_sp, float * out) {
    int nH = H / H_sp, nW_dim = W / W_sp;
    for (int wh = 0; wh < nH; wh++) {
        for (int ww = 0; ww < nW_dim; ww++) {
            int win_idx = wh * nW_dim + ww;
            for (int yy = 0; yy < H_sp; yy++) {
                for (int xx = 0; xx < W_sp; xx++) {
                    int dst_y = wh * H_sp + yy;
                    int dst_x = ww * W_sp + xx;
                    int token = yy * W_sp + xx;
                    const float * src = x + (win_idx * H_sp * W_sp + token) * C;
                    float * dst = out + (dst_y * W + dst_x) * C;
                    memcpy(dst, src, C * sizeof(float));
                }
            }
        }
    }
}

// Circular roll of a (H, W, C) tensor by (shift_h, shift_w)
static void roll_hwc(const float * in, int H, int W, int C,
                      int shift_h, int shift_w, float * out) {
    for (int y = 0; y < H; y++) {
        int sy = ((y - shift_h) % H + H) % H;
        for (int x = 0; x < W; x++) {
            int sx = ((x - shift_w) % W + W) % W;
            memcpy(out + (y * W + x) * C, in + (sy * W + sx) * C, C * sizeof(float));
        }
    }
}

// Compute shift-window attention mask for one branch
// Returns mask of shape (nW, win_N, win_N) where masked positions are -100
static void compute_shift_mask(int H, int W, int H_sp, int W_sp,
                                int shift_h, int shift_w,
                                std::vector<float> & mask) {
    int nH = H / H_sp, nW = W / W_sp;
    int win_N = H_sp * W_sp;
    int n_windows = nH * nW;
    mask.resize(n_windows * win_N * win_N);

    // Create region IDs
    std::vector<int> region(H * W, 0);
    // h_slices: [0, H-H_sp), [H-H_sp, H-shift_h), [H-shift_h, H)
    // w_slices: [0, W-W_sp), [W-W_sp, W-shift_w), [W-shift_w, W)
    int h_bounds[4] = {0, H - H_sp, H - shift_h, H};
    int w_bounds[4] = {0, W - W_sp, W - shift_w, W};
    int cnt = 0;
    for (int hi = 0; hi < 3; hi++) {
        for (int wi = 0; wi < 3; wi++) {
            for (int y = h_bounds[hi]; y < h_bounds[hi+1]; y++)
                for (int x = w_bounds[wi]; x < w_bounds[wi+1]; x++)
                    region[y * W + x] = cnt;
            cnt++;
        }
    }

    // Partition region IDs into windows
    std::vector<int> win_regions(n_windows * win_N);
    for (int wh = 0; wh < nH; wh++) {
        for (int ww = 0; ww < nW; ww++) {
            int win_idx = wh * nW + ww;
            for (int yy = 0; yy < H_sp; yy++) {
                for (int xx = 0; xx < W_sp; xx++) {
                    int gy = wh * H_sp + yy, gx = ww * W_sp + xx;
                    win_regions[win_idx * win_N + yy * W_sp + xx] = region[gy * W + gx];
                }
            }
        }
    }

    // Build mask: same region → 0, different → -100
    for (int w = 0; w < n_windows; w++) {
        for (int i = 0; i < win_N; i++) {
            for (int j = 0; j < win_N; j++) {
                float val = (win_regions[w * win_N + i] == win_regions[w * win_N + j]) ? 0.0f : -100.0f;
                mask[w * win_N * win_N + i * win_N + j] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dynamic Position Bias (residual=False for inference)
// ---------------------------------------------------------------------------

static void dynamic_pos_bias(dat_sr_context * ctx, int rg, int blk, int branch,
                              int H_sp, int W_sp, int num_heads_branch,
                              std::vector<float> & pos_table) {
    // pos_table output: (win_N * win_N) values for each head → (num_heads_branch, win_N, win_N)
    // via relative_position_index lookup
    auto p = [&](const char * s) { return pfx(rg, blk, s); };
    char b[128];

    // Spatial_Attention.dim = embed_dim/2. DynamicPosBias gets dim = sa_dim/4.
    // pos_dim = dpb_dim / 4 = (embed_dim/2 / 4) / 4
    int pos_dim = (ctx->embed_dim / 2) / 4 / 4;
    int win_N = H_sp * W_sp;

    // Get rpe_biases: ((2*H_sp-1)*(2*W_sp-1), 2) and relative_position_index: (win_N, win_N)
    snprintf(b, sizeof(b), "layers.%d.blocks.%d.attn.attns.%d.rpe_biases", rg, blk, branch);
    const float * rpe = get_w(ctx, b);
    snprintf(b, sizeof(b), "layers.%d.blocks.%d.attn.attns.%d.relative_position_index", rg, blk, branch);
    const float * rpi_f = get_w(ctx, b);
    if (!rpe || !rpi_f) {
        pos_table.assign(num_heads_branch * win_N * win_N, 0.0f);
        return;
    }

    int rpe_len = (2*H_sp - 1) * (2*W_sp - 1);

    // pos_proj: Linear(2, pos_dim)
    snprintf(b, sizeof(b), "layers.%d.blocks.%d.attn.attns.%d.pos.pos_proj.weight", rg, blk, branch);
    const float * pp_w = get_w(ctx, b);
    snprintf(b, sizeof(b), "layers.%d.blocks.%d.attn.attns.%d.pos.pos_proj.bias", rg, blk, branch);
    const float * pp_b = get_w(ctx, b);

    // Compute pos = pos_proj(rpe_biases): (rpe_len, pos_dim)
    std::vector<float> pos(rpe_len * pos_dim);
    core_cpu::linear_batch_cpu(rpe, pos.data(), rpe_len, 2, pos_dim, pp_w, pp_b);

    // pos1: LN → ReLU → Linear (residual=false, so sequential not residual)
    auto apply_pos_block = [&](const char * block_name, float * data, int len, int in_dim, int out_dim) {
        char n1[192], n2[192], n3[192], n4[192];
        snprintf(n1, sizeof(n1), "layers.%d.blocks.%d.attn.attns.%d.pos.%s.0.weight", rg, blk, branch, block_name);
        snprintf(n2, sizeof(n2), "layers.%d.blocks.%d.attn.attns.%d.pos.%s.0.bias", rg, blk, branch, block_name);
        snprintf(n3, sizeof(n3), "layers.%d.blocks.%d.attn.attns.%d.pos.%s.2.weight", rg, blk, branch, block_name);
        snprintf(n4, sizeof(n4), "layers.%d.blocks.%d.attn.attns.%d.pos.%s.2.bias", rg, blk, branch, block_name);
        const float * ln_w = get_w(ctx, n1);
        const float * ln_b = get_w(ctx, n2);
        const float * fc_w = get_w(ctx, n3);
        const float * fc_b = get_w(ctx, n4);
        if (!ln_w || !fc_w) return;
        std::vector<float> tmp(len * out_dim);
        std::vector<float> normed(in_dim);
        for (int i = 0; i < len; i++) {
            layernorm_cpu(data + i*in_dim, in_dim, ln_w, ln_b, normed.data());
            for (int j = 0; j < in_dim; j++) normed[j] = std::max(0.0f, normed[j]);
            linear_cpu(normed.data(), in_dim, fc_w, fc_b, out_dim, tmp.data() + i*out_dim);
        }
        memcpy(data, tmp.data(), len * out_dim * sizeof(float));
    };

    // pos1: (rpe_len, pos_dim) → (rpe_len, pos_dim) [residual=false: sequential]
    apply_pos_block("pos1", pos.data(), rpe_len, pos_dim, pos_dim);
    // pos2: same
    apply_pos_block("pos2", pos.data(), rpe_len, pos_dim, pos_dim);
    // pos3: (rpe_len, pos_dim) → (rpe_len, num_heads_branch)
    // Need separate buffer since output dim differs
    std::vector<float> pos_final(rpe_len * num_heads_branch);
    {
        char n1[192], n2[192], n3[192], n4[192];
        snprintf(n1, sizeof(n1), "layers.%d.blocks.%d.attn.attns.%d.pos.pos3.0.weight", rg, blk, branch);
        snprintf(n2, sizeof(n2), "layers.%d.blocks.%d.attn.attns.%d.pos.pos3.0.bias", rg, blk, branch);
        snprintf(n3, sizeof(n3), "layers.%d.blocks.%d.attn.attns.%d.pos.pos3.2.weight", rg, blk, branch);
        snprintf(n4, sizeof(n4), "layers.%d.blocks.%d.attn.attns.%d.pos.pos3.2.bias", rg, blk, branch);
        const float * ln_w = get_w(ctx, n1);
        const float * ln_b = get_w(ctx, n2);
        const float * fc_w = get_w(ctx, n3);
        const float * fc_b = get_w(ctx, n4);
        if (ln_w && fc_w) {
            std::vector<float> normed(pos_dim);
            for (int i = 0; i < rpe_len; i++) {
                layernorm_cpu(pos.data() + i*pos_dim, pos_dim, ln_w, ln_b, normed.data());
                for (int j = 0; j < pos_dim; j++) normed[j] = std::max(0.0f, normed[j]);
                linear_cpu(normed.data(), pos_dim, fc_w, fc_b, num_heads_branch, pos_final.data() + i*num_heads_branch);
            }
        }
    }

    // pos_final: (rpe_len, num_heads_branch)
    // Build pos_table: (num_heads_branch, win_N, win_N)
    pos_table.resize(num_heads_branch * win_N * win_N);
    for (int i = 0; i < win_N; i++) {
        for (int j = 0; j < win_N; j++) {
            int idx = (int)rpi_f[i * win_N + j];
            for (int h = 0; h < num_heads_branch; h++) {
                pos_table[h * win_N * win_N + i * win_N + j] = pos_final[idx * num_heads_branch + h];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AIM (Adaptive Interaction Module) — shared by spatial and channel attention
// ---------------------------------------------------------------------------

// Apply DWConv + BN + GELU branch
static void aim_dwconv_branch(dat_sr_context * ctx, const float * v_chw, int C, int H, int W,
                               const std::string & prefix, float * conv_out) {
    // dwconv: DWConv3x3 → BN → GELU
    const float * dw_w = get_w(ctx, prefix + ".dwconv.0.weight");
    const float * dw_b = get_w(ctx, prefix + ".dwconv.0.bias");
    if (!dw_w) { memcpy(conv_out, v_chw, C*H*W*sizeof(float)); return; }
    dwconv2d_3x3(v_chw, C, H, W, dw_w, dw_b, conv_out);
    // BN
    const float * bn_w = get_w(ctx, prefix + ".dwconv.1.weight");
    const float * bn_b = get_w(ctx, prefix + ".dwconv.1.bias");
    const float * bn_m = get_w(ctx, prefix + ".dwconv.1.running_mean");
    const float * bn_v = get_w(ctx, prefix + ".dwconv.1.running_var");
    if (bn_w) batchnorm_eval(conv_out, C, H, W, bn_w, bn_b, bn_m, bn_v);
    // GELU
    for (int i = 0; i < C*H*W; i++) conv_out[i] = gelu_f(conv_out[i]);
}

// Compute channel interaction (AdaptiveAvgPool → Conv1x1 → BN → GELU → Conv1x1)
// Input: CHW, output: (C,) — one value per channel
static void aim_channel_map(dat_sr_context * ctx, const float * in_chw, int C, int H, int W,
                             const std::string & prefix, float * out_c) {
    int mid = C / 8;
    // AdaptiveAvgPool2d(1)
    std::vector<float> pooled(C);
    adaptive_avg_pool_1x1(in_chw, C, H*W, pooled.data());
    // Conv1x1 (C → mid) — applied to (C, 1, 1) which is just linear
    const float * w1 = get_w(ctx, prefix + ".channel_interaction.1.weight");
    const float * b1 = get_w(ctx, prefix + ".channel_interaction.1.bias");
    std::vector<float> t1(mid);
    if (w1) linear_cpu(pooled.data(), C, w1, b1, mid, t1.data());
    // BN on (mid, 1, 1) — simplified since H=W=1
    const float * bn_w = get_w(ctx, prefix + ".channel_interaction.2.weight");
    const float * bn_b = get_w(ctx, prefix + ".channel_interaction.2.bias");
    const float * bn_m = get_w(ctx, prefix + ".channel_interaction.2.running_mean");
    const float * bn_v = get_w(ctx, prefix + ".channel_interaction.2.running_var");
    if (bn_w) batchnorm_eval(t1.data(), mid, 1, 1, bn_w, bn_b, bn_m, bn_v);
    // GELU
    for (int i = 0; i < mid; i++) t1[i] = gelu_f(t1[i]);
    // Conv1x1 (mid → C)
    const float * w2 = get_w(ctx, prefix + ".channel_interaction.4.weight");
    const float * b2 = get_w(ctx, prefix + ".channel_interaction.4.bias");
    if (w2) linear_cpu(t1.data(), mid, w2, b2, C, out_c);
}

// Compute spatial interaction (Conv1x1 → BN → GELU → Conv1x1)
// Input: CHW, output: (H*W,) — one value per spatial position
static void aim_spatial_map(dat_sr_context * ctx, const float * in_chw, int C, int H, int W,
                             const std::string & prefix, float * out_hw) {
    int mid = C / 16;
    // Conv1x1 (C → mid)
    std::vector<float> t1(mid * H * W);
    const float * w1 = get_w(ctx, prefix + ".spatial_interaction.0.weight");
    const float * b1 = get_w(ctx, prefix + ".spatial_interaction.0.bias");
    if (w1) conv1x1(in_chw, C, H, W, w1, b1, mid, t1.data());
    // BN
    const float * bn_w = get_w(ctx, prefix + ".spatial_interaction.1.weight");
    const float * bn_b = get_w(ctx, prefix + ".spatial_interaction.1.bias");
    const float * bn_m = get_w(ctx, prefix + ".spatial_interaction.1.running_mean");
    const float * bn_v = get_w(ctx, prefix + ".spatial_interaction.1.running_var");
    if (bn_w) batchnorm_eval(t1.data(), mid, H, W, bn_w, bn_b, bn_m, bn_v);
    // GELU
    for (int i = 0; i < mid*H*W; i++) t1[i] = gelu_f(t1[i]);
    // Conv1x1 (mid → 1)
    std::vector<float> t2(1 * H * W);
    const float * w2 = get_w(ctx, prefix + ".spatial_interaction.3.weight");
    const float * b2 = get_w(ctx, prefix + ".spatial_interaction.3.bias");
    if (w2) conv1x1(t1.data(), mid, H, W, w2, b2, 1, t2.data());
    memcpy(out_hw, t2.data(), H * W * sizeof(float));
}

// ---------------------------------------------------------------------------
// Adaptive Spatial Attention (even blocks)
// ---------------------------------------------------------------------------

static void adaptive_spatial_attention(dat_sr_context * ctx, float * x, int H, int W,
                                        int rg, int blk) {
    int C = ctx->embed_dim;
    int N = H * W;
    int nh = ctx->num_heads;
    int nh_branch = nh / 2; // heads per branch
    int C_half = C / 2;
    int hd = C_half / nh_branch;
    std::string attn_pfx = pfx(rg, blk, "attn");

    // QKV projection: (N, C) → (N, 3C)
    const float * qkv_w = get_w(ctx, attn_pfx + ".qkv.weight");
    const float * qkv_b = get_w(ctx, attn_pfx + ".qkv.bias");
    if (!qkv_w) return;

    std::vector<float> qkv(N * 3 * C);
    core_cpu::linear_batch_cpu(x, qkv.data(), N, C, 3*C, qkv_w, qkv_b);

    // V (full, unpartitioned) for DWConv branch — reshape to CHW
    std::vector<float> v_chw(C * H * W);
    for (int i = 0; i < N; i++)
        for (int c = 0; c < C; c++)
            v_chw[c*H*W + i] = qkv[i*3*C + 2*C + c];

    // Pad to multiples of max split size
    int max_sp = std::max(ctx->split_size[0], ctx->split_size[1]);
    int pad_b = (max_sp - H % max_sp) % max_sp;
    int pad_r = (max_sp - W % max_sp) % max_sp;
    int pH = H + pad_b, pW = W + pad_r;
    int pN = pH * pW;

    // Rearrange QKV to (pH, pW, 3*C) with padding
    std::vector<float> qkv_hw(pN * 3 * C, 0.0f);
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++)
            memcpy(qkv_hw.data() + (y*pW + xi)*3*C, qkv.data() + (y*W + xi)*3*C, 3*C*sizeof(float));

    // Determine if this block uses shifted windows
    bool use_shift = false;
    if (rg % 2 == 0) {
        use_shift = (blk > 0 && (blk - 2) % 4 == 0);
    } else {
        use_shift = (blk % 4 == 0);
    }
    int shift_h0 = ctx->split_size[0] / 2; // shift for branch 0
    int shift_w0 = ctx->split_size[1] / 2;

    // Process each branch
    std::vector<float> attn_out(N * C, 0.0f);

    for (int br = 0; br < 2; br++) {
        int H_sp = (br == 0) ? ctx->split_size[0] : ctx->split_size[1];
        int W_sp = (br == 0) ? ctx->split_size[1] : ctx->split_size[0];
        int win_N = H_sp * W_sp;
        int nW = (pH / H_sp) * (pW / W_sp);

        // Extract this branch's QKV channels: first half or second half of C
        int c_off = br * C_half;

        // Build branch QKV: (pH, pW, C_half) for each of Q, K, V
        std::vector<float> br_qkv(pN * 3 * C_half);
        for (int i = 0; i < pN; i++) {
            for (int c = 0; c < C_half; c++) {
                br_qkv[i*3*C_half + c]             = qkv_hw[i*3*C + c_off + c];          // Q
                br_qkv[i*3*C_half + C_half + c]     = qkv_hw[i*3*C + C + c_off + c];      // K
                br_qkv[i*3*C_half + 2*C_half + c]   = qkv_hw[i*3*C + 2*C + c_off + c];    // V
            }
        }

        std::vector<float> shifted;
        const float * src_qkv = br_qkv.data();
        std::vector<float> mask;

        if (use_shift) {
            int sh = (br == 0) ? -shift_h0 : -shift_w0;
            int sw = (br == 0) ? -shift_w0 : -shift_h0;
            // Roll each of Q, K, V
            shifted.resize(pN * 3 * C_half);
            for (int part = 0; part < 3; part++) {
                // Extract (pH, pW, C_half) for this part
                std::vector<float> slice(pN * C_half), rolled(pN * C_half);
                for (int i = 0; i < pN; i++)
                    memcpy(slice.data() + i*C_half, br_qkv.data() + i*3*C_half + part*C_half, C_half*sizeof(float));
                roll_hwc(slice.data(), pH, pW, C_half, sh, sw, rolled.data());
                for (int i = 0; i < pN; i++)
                    memcpy(shifted.data() + i*3*C_half + part*C_half, rolled.data() + i*C_half, C_half*sizeof(float));
            }
            src_qkv = shifted.data();

            int mask_sh = (br == 0) ? shift_h0 : shift_w0;
            int mask_sw = (br == 0) ? shift_w0 : shift_h0;
            compute_shift_mask(pH, pW, H_sp, W_sp, mask_sh, mask_sw, mask);
        }

        // Window-partition Q, K, V: (pN, C_half) → (nW, win_N, C_half) each
        std::vector<float> wq(nW * win_N * C_half), wk(nW * win_N * C_half), wv(nW * win_N * C_half);
        {
            std::vector<float> tmp(pN * C_half);
            // Q
            for (int i = 0; i < pN; i++)
                memcpy(tmp.data() + i*C_half, src_qkv + i*3*C_half, C_half*sizeof(float));
            window_partition(tmp.data(), pH, pW, C_half, H_sp, W_sp, wq.data());
            // K
            for (int i = 0; i < pN; i++)
                memcpy(tmp.data() + i*C_half, src_qkv + i*3*C_half + C_half, C_half*sizeof(float));
            window_partition(tmp.data(), pH, pW, C_half, H_sp, W_sp, wk.data());
            // V
            for (int i = 0; i < pN; i++)
                memcpy(tmp.data() + i*C_half, src_qkv + i*3*C_half + 2*C_half, C_half*sizeof(float));
            window_partition(tmp.data(), pH, pW, C_half, H_sp, W_sp, wv.data());
        }

        // Compute dynamic position bias for this branch
        std::vector<float> pos_bias;
        dynamic_pos_bias(ctx, rg, blk, br, H_sp, W_sp, nh_branch, pos_bias);

        // Multi-head attention per window
        float scale = 1.0f / sqrtf((float)hd);
        std::vector<float> w_out(nW * win_N * C_half);

        for (int w = 0; w < nW; w++) {
            float * q_ptr = wq.data() + w * win_N * C_half;
            float * k_ptr = wk.data() + w * win_N * C_half;
            float * v_ptr = wv.data() + w * win_N * C_half;
            float * o_ptr = w_out.data() + w * win_N * C_half;

            // Per head: Q(win_N, hd) @ K^T(hd, win_N) → (win_N, win_N)
            for (int h = 0; h < nh_branch; h++) {
                std::vector<float> attn_scores(win_N * win_N);
                for (int i = 0; i < win_N; i++) {
                    for (int j = 0; j < win_N; j++) {
                        float dot = 0;
                        for (int d = 0; d < hd; d++)
                            dot += q_ptr[i*C_half + h*hd + d] * k_ptr[j*C_half + h*hd + d];
                        attn_scores[i*win_N + j] = dot * scale;
                    }
                }

                // Add position bias
                for (int i = 0; i < win_N * win_N; i++)
                    attn_scores[i] += pos_bias[h * win_N * win_N + i];

                // Add shift mask
                if (use_shift) {
                    for (int i = 0; i < win_N * win_N; i++)
                        attn_scores[i] += mask[w * win_N * win_N + i];
                }

                // Softmax per row
                for (int i = 0; i < win_N; i++) {
                    float max_s = -1e30f;
                    for (int j = 0; j < win_N; j++)
                        max_s = std::max(max_s, attn_scores[i*win_N + j]);
                    float sum = 0;
                    for (int j = 0; j < win_N; j++) {
                        attn_scores[i*win_N + j] = expf(attn_scores[i*win_N + j] - max_s);
                        sum += attn_scores[i*win_N + j];
                    }
                    for (int j = 0; j < win_N; j++)
                        attn_scores[i*win_N + j] /= sum;
                }

                // Attn @ V
                for (int i = 0; i < win_N; i++) {
                    for (int d = 0; d < hd; d++) {
                        float sum = 0;
                        for (int j = 0; j < win_N; j++)
                            sum += attn_scores[i*win_N + j] * v_ptr[j*C_half + h*hd + d];
                        o_ptr[i*C_half + h*hd + d] = sum;
                    }
                }
            }
        }

        // Window unpartition → (pH, pW, C_half)
        std::vector<float> branch_out(pN * C_half);
        window_unpartition(w_out.data(), pH, pW, C_half, H_sp, W_sp, branch_out.data());

        // Reverse shift
        if (use_shift) {
            int sh = (br == 0) ? shift_h0 : shift_w0;
            int sw = (br == 0) ? shift_w0 : shift_h0;
            std::vector<float> unrolled(pN * C_half);
            roll_hwc(branch_out.data(), pH, pW, C_half, sh, sw, unrolled.data());
            branch_out = std::move(unrolled);
        }

        // Crop to (H, W) and place in attn_out
        for (int y = 0; y < H; y++)
            for (int xi = 0; xi < W; xi++)
                for (int c = 0; c < C_half; c++)
                    attn_out[(y*W + xi)*C + c_off + c] = branch_out[(y*pW + xi)*C_half + c];
    }

    // --- AIM: merge attention and conv branches ---
    std::string aim_pfx = attn_pfx;

    // DWConv branch on V
    std::vector<float> conv_out(C * H * W);
    aim_dwconv_branch(ctx, v_chw.data(), C, H, W, aim_pfx, conv_out.data());

    // Channel map (from conv_x): (C,)
    std::vector<float> c_map(C);
    aim_channel_map(ctx, conv_out.data(), C, H, W, aim_pfx, c_map.data());

    // Spatial map (from attened_x reshaped to CHW): (H*W,)
    std::vector<float> attn_chw(C * H * W);
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++)
            for (int c = 0; c < C; c++)
                attn_chw[c*H*W + y*W + xi] = attn_out[(y*W + xi)*C + c];
    std::vector<float> s_map(H * W);
    aim_spatial_map(ctx, attn_chw.data(), C, H, W, aim_pfx, s_map.data());

    // C-I: attened_x *= sigmoid(channel_map) — channel_map is (1, C), broadcast to (N, C)
    for (int i = 0; i < N; i++)
        for (int c = 0; c < C; c++)
            attn_out[i*C + c] *= sigmoid_f(c_map[c]);

    // S-I: conv_x *= sigmoid(spatial_map) — spatial_map is (H*W,), broadcast to (C, H*W)
    for (int c = 0; c < C; c++)
        for (int i = 0; i < H*W; i++)
            conv_out[c*H*W + i] *= sigmoid_f(s_map[i]);

    // Sum and convert conv_out (CHW) to NHW-C layout
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++)
            for (int c = 0; c < C; c++)
                attn_out[(y*W + xi)*C + c] += conv_out[c*H*W + y*W + xi];

    // Output projection
    const float * proj_w = get_w(ctx, attn_pfx + ".proj.weight");
    const float * proj_b = get_w(ctx, attn_pfx + ".proj.bias");
    if (proj_w) {
        std::vector<float> projected(N * C);
        core_cpu::linear_batch_cpu(attn_out.data(), projected.data(), N, C, C, proj_w, proj_b);
        for (int i = 0; i < N*C; i++) x[i] += projected[i];
    }
}

// ---------------------------------------------------------------------------
// Adaptive Channel Attention (odd blocks)
// ---------------------------------------------------------------------------

static void adaptive_channel_attention(dat_sr_context * ctx, float * x, int H, int W,
                                        int rg, int blk) {
    int C = ctx->embed_dim;
    int N = H * W;
    int nh = ctx->num_heads;
    int hd = C / nh;
    std::string attn_pfx = pfx(rg, blk, "attn");

    // QKV projection
    const float * qkv_w = get_w(ctx, attn_pfx + ".qkv.weight");
    const float * qkv_b = get_w(ctx, attn_pfx + ".qkv.bias");
    if (!qkv_w) return;

    std::vector<float> qkv(N * 3 * C);
    core_cpu::linear_batch_cpu(x, qkv.data(), N, C, 3*C, qkv_w, qkv_b);

    // Reshape to per-head: Q, K, V each (nh, N, hd) → transpose to (nh, hd, N)
    // Then L2-normalize Q, K over last dim (N)
    std::vector<float> qt(nh * hd * N), kt(nh * hd * N), vt(nh * hd * N);
    for (int i = 0; i < N; i++) {
        for (int h = 0; h < nh; h++) {
            for (int d = 0; d < hd; d++) {
                qt[h*hd*N + d*N + i] = qkv[i*3*C + h*hd + d];
                kt[h*hd*N + d*N + i] = qkv[i*3*C + C + h*hd + d];
                vt[h*hd*N + d*N + i] = qkv[i*3*C + 2*C + h*hd + d];
            }
        }
    }

    // L2-normalize Q, K rows — each row is (N,) values
    for (int h = 0; h < nh; h++) {
        l2_normalize_rows(qt.data() + h*hd*N, hd, N);
        l2_normalize_rows(kt.data() + h*hd*N, hd, N);
    }

    // V for DWConv branch: reshape V to CHW
    std::vector<float> v_chw(C * H * W);
    for (int i = 0; i < N; i++)
        for (int c = 0; c < C; c++)
            v_chw[c*H*W + i] = qkv[i*3*C + 2*C + c];

    // Get temperature: (nh, 1, 1)
    const float * temp_w = get_w(ctx, attn_pfx + ".temperature");

    // Per-head channel attention: Q^T @ K → (hd, hd) per head
    // attn = softmax((Q @ K^T) * temperature)
    // output = attn @ V → (hd, N) per head
    std::vector<float> attn_out_nc(N * C); // (N, C)
    for (int h = 0; h < nh; h++) {
        float temp = (temp_w) ? temp_w[h] : 1.0f;
        float * qh = qt.data() + h*hd*N;
        float * kh = kt.data() + h*hd*N;
        float * vh = vt.data() + h*hd*N;

        // attn: (hd, hd) = Q(hd, N) @ K^T(N, hd) * temperature
        std::vector<float> attn(hd * hd);
        for (int i = 0; i < hd; i++) {
            for (int j = 0; j < hd; j++) {
                float dot = 0;
                for (int n = 0; n < N; n++)
                    dot += qh[i*N + n] * kh[j*N + n];
                attn[i*hd + j] = dot * temp;
            }
        }

        // Softmax per row
        for (int i = 0; i < hd; i++) {
            float max_s = -1e30f;
            for (int j = 0; j < hd; j++) max_s = std::max(max_s, attn[i*hd + j]);
            float sum = 0;
            for (int j = 0; j < hd; j++) {
                attn[i*hd + j] = expf(attn[i*hd + j] - max_s);
                sum += attn[i*hd + j];
            }
            for (int j = 0; j < hd; j++) attn[i*hd + j] /= sum;
        }

        // output = attn(hd, hd) @ V(hd, N) → (hd, N)
        // Then permute to (N, nh, hd) layout → (N, C)
        for (int d = 0; d < hd; d++) {
            for (int n = 0; n < N; n++) {
                float sum = 0;
                for (int j = 0; j < hd; j++)
                    sum += attn[d*hd + j] * vh[j*N + n];
                attn_out_nc[n*C + h*hd + d] = sum;
            }
        }
    }

    // --- AIM ---
    // DWConv branch on V
    std::vector<float> conv_out(C * H * W);
    aim_dwconv_branch(ctx, v_chw.data(), C, H, W, attn_pfx, conv_out.data());

    // Channel map (from attened_x — NOTE: reversed from spatial)
    std::vector<float> attn_chw(C * H * W);
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++)
            for (int c = 0; c < C; c++)
                attn_chw[c*H*W + y*W + xi] = attn_out_nc[(y*W + xi)*C + c];
    std::vector<float> c_map(C);
    aim_channel_map(ctx, attn_chw.data(), C, H, W, attn_pfx, c_map.data());

    // Spatial map (from conv_x — NOTE: reversed from spatial)
    std::vector<float> s_map(H * W);
    aim_spatial_map(ctx, conv_out.data(), C, H, W, attn_pfx, s_map.data());

    // S-I: attened_x *= sigmoid(spatial_map)
    for (int i = 0; i < N; i++)
        for (int c = 0; c < C; c++)
            attn_out_nc[i*C + c] *= sigmoid_f(s_map[i]);

    // C-I: conv_x *= sigmoid(channel_map)
    for (int c = 0; c < C; c++)
        for (int i = 0; i < H*W; i++)
            conv_out[c*H*W + i] *= sigmoid_f(c_map[c]);

    // Sum
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++)
            for (int c = 0; c < C; c++)
                attn_out_nc[(y*W + xi)*C + c] += conv_out[c*H*W + y*W + xi];

    // Output projection
    const float * proj_w = get_w(ctx, attn_pfx + ".proj.weight");
    const float * proj_b = get_w(ctx, attn_pfx + ".proj.bias");
    if (proj_w) {
        std::vector<float> projected(N * C);
        core_cpu::linear_batch_cpu(attn_out_nc.data(), projected.data(), N, C, C, proj_w, proj_b);
        for (int i = 0; i < N*C; i++) x[i] += projected[i];
    }
}

// ---------------------------------------------------------------------------
// SGFN (Spatial-Gate Feed-Forward Network)
// ---------------------------------------------------------------------------

static void sgfn_forward(dat_sr_context * ctx, float * x, int H, int W,
                          int rg, int blk) {
    int C = ctx->embed_dim;
    int N = H * W;
    int hidden = C * 2; // expansion_factor = 2
    std::string ffn = pfx(rg, blk, "ffn");

    const float * fc1_w = get_w(ctx, ffn + ".fc1.weight");
    const float * fc1_b = get_w(ctx, ffn + ".fc1.bias");
    const float * fc2_w = get_w(ctx, ffn + ".fc2.weight");
    const float * fc2_b = get_w(ctx, ffn + ".fc2.bias");
    if (!fc1_w || !fc2_w) return;

    int half_hidden = hidden / 2;

    // fc1: (N, C) → (N, hidden) + GELU
    std::vector<float> h(N * hidden);
    core_cpu::linear_batch_cpu(x, h.data(), N, C, hidden, fc1_w, fc1_b);
    for (int i = 0; i < N * hidden; i++)
        h[i] = gelu_f(h[i]);

    // SpatialGate: split into x1(half_hidden), x2(half_hidden)
    // x2 → LN → reshape to CHW → DWConv → reshape back → multiply with x1
    const float * sg_ln_w = get_w(ctx, ffn + ".sg.norm.weight");
    const float * sg_ln_b = get_w(ctx, ffn + ".sg.norm.bias");
    const float * sg_dw_w = get_w(ctx, ffn + ".sg.conv.weight");
    const float * sg_dw_b = get_w(ctx, ffn + ".sg.conv.bias");

    std::vector<float> gated(N * half_hidden);
    if (sg_ln_w && sg_dw_w) {
        // LN on x2 part
        std::vector<float> x2_normed(N * half_hidden);
        for (int i = 0; i < N; i++)
            layernorm_cpu(h.data() + i*hidden + half_hidden, half_hidden,
                          sg_ln_w, sg_ln_b, x2_normed.data() + i*half_hidden);

        // Reshape to CHW for DW conv
        std::vector<float> x2_chw(half_hidden * H * W);
        for (int y = 0; y < H; y++)
            for (int xi = 0; xi < W; xi++)
                for (int c = 0; c < half_hidden; c++)
                    x2_chw[c*H*W + y*W + xi] = x2_normed[(y*W + xi)*half_hidden + c];

        // DWConv
        std::vector<float> x2_conv(half_hidden * H * W);
        dwconv2d_3x3(x2_chw.data(), half_hidden, H, W, sg_dw_w, sg_dw_b, x2_conv.data());

        // Reshape back to NHW and multiply with x1
        for (int y = 0; y < H; y++)
            for (int xi = 0; xi < W; xi++)
                for (int c = 0; c < half_hidden; c++) {
                    int idx = (y*W + xi);
                    gated[idx*half_hidden + c] =
                        h[idx*hidden + c] * x2_conv[c*H*W + y*W + xi];
                }
    } else {
        // Fallback: just use first half
        for (int i = 0; i < N; i++)
            memcpy(gated.data() + i*half_hidden, h.data() + i*hidden, half_hidden*sizeof(float));
    }

    // fc2: (N, half_hidden) → (N, C)
    std::vector<float> out(N * C);
    core_cpu::linear_batch_cpu(gated.data(), out.data(), N, half_hidden, C, fc2_w, fc2_b);

    // Residual
    for (int i = 0; i < N*C; i++) x[i] += out[i];
}

// ---------------------------------------------------------------------------
// DATB block
// ---------------------------------------------------------------------------

static void datb_forward(dat_sr_context * ctx, float * x, int H, int W,
                          int rg, int blk) {
    int C = ctx->embed_dim;
    int N = H * W;

    // norm1
    const float * n1w = get_w(ctx, pfx(rg, blk, "norm1.weight"));
    const float * n1b = get_w(ctx, pfx(rg, blk, "norm1.bias"));
    std::vector<float> normed(N * C);
    for (int i = 0; i < N; i++)
        layernorm_cpu(x + i*C, C, n1w, n1b, normed.data() + i*C);

    // Attention (operates on normed, adds residual to x)
    // Copy normed into a working buffer that attention modifies
    std::vector<float> attn_in(normed);
    if (blk % 2 == 0) {
        adaptive_spatial_attention(ctx, attn_in.data(), H, W, rg, blk);
    } else {
        adaptive_channel_attention(ctx, attn_in.data(), H, W, rg, blk);
    }
    // attn_in now contains normed + attention output (residual added inside)
    // But attention adds residual to its input (normed), we need residual to x
    // Fix: attention functions do x[i] += projected[i], but they operate on a copy
    // We need to add the delta to the original x
    for (int i = 0; i < N*C; i++) x[i] += (attn_in[i] - normed[i]);

    // norm2 → SGFN (with residual)
    const float * n2w = get_w(ctx, pfx(rg, blk, "norm2.weight"));
    const float * n2b = get_w(ctx, pfx(rg, blk, "norm2.bias"));
    std::vector<float> normed2(N * C);
    for (int i = 0; i < N; i++)
        layernorm_cpu(x + i*C, C, n2w, n2b, normed2.data() + i*C);

    // SGFN operates on normed2, adds residual to x
    std::vector<float> ffn_in(normed2);
    sgfn_forward(ctx, ffn_in.data(), H, W, rg, blk);
    for (int i = 0; i < N*C; i++) x[i] += (ffn_in[i] - normed2[i]);
}

// ---------------------------------------------------------------------------
// Full DAT forward pass (single tile)
// ---------------------------------------------------------------------------

static void dat_forward(dat_sr_context * ctx, const float * rgb_in, int H, int W,
                         float * rgb_out) {
    int C = ctx->embed_dim;
    int N = H * W;

    // 1. Subtract mean, scale by img_range
    std::vector<float> input(3 * H * W);
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < H * W; i++)
            input[c*H*W + i] = (rgb_in[c*H*W + i] - ctx->mean[c]) * ctx->img_range;

    // 2. Shallow feature extraction: conv_first (3 → embed_dim)
    std::vector<float> shallow(C * H * W);
    conv2d_3x3(input.data(), 3, H, W,
               get_w(ctx, "conv_first.weight"), get_w(ctx, "conv_first.bias"),
               C, shallow.data());

    // 3. before_RG: rearrange (B,C,H,W) → (B,HW,C), then LayerNorm
    std::vector<float> x(N * C);
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++)
            for (int c = 0; c < C; c++)
                x[(y*W + xi)*C + c] = shallow[c*H*W + y*W + xi];

    const float * brg_w = get_w(ctx, "before_RG.1.weight");
    const float * brg_b = get_w(ctx, "before_RG.1.bias");
    if (brg_w) {
        std::vector<float> tmp(N * C);
        for (int i = 0; i < N; i++)
            layernorm_cpu(x.data() + i*C, C, brg_w, brg_b, tmp.data() + i*C);
        x = std::move(tmp);
    }

    // 4. ResidualGroups
    for (int rg = 0; rg < (int)ctx->depth.size(); rg++) {
        std::vector<float> rg_input(x); // save for residual

        // Run DATB blocks
        for (int blk = 0; blk < ctx->depth[rg]; blk++) {
            datb_forward(ctx, x.data(), H, W, rg, blk);
        }

        // Rearrange to CHW for conv
        std::vector<float> x_chw(C * H * W);
        for (int y = 0; y < H; y++)
            for (int xi = 0; xi < W; xi++)
                for (int c = 0; c < C; c++)
                    x_chw[c*H*W + y*W + xi] = x[(y*W + xi)*C + c];

        // ResidualGroup conv
        char buf[128];
        if (ctx->resi_connection == "3conv") {
            int mid = C / 4;
            std::vector<float> t1(mid * H * W), t2(mid * H * W), t3(C * H * W);
            snprintf(buf, sizeof(buf), "layers.%d.conv.0.weight", rg);
            conv2d_3x3(x_chw.data(), C, H, W, get_w(ctx, buf),
                       get_w(ctx, std::string(buf).replace(strlen(buf)-6, 6, "bias")), mid, t1.data());
            leaky_relu(t1.data(), mid * H * W);
            snprintf(buf, sizeof(buf), "layers.%d.conv.2.weight", rg);
            conv1x1(t1.data(), mid, H, W, get_w(ctx, buf),
                     get_w(ctx, std::string(buf).replace(strlen(buf)-6, 6, "bias")), mid, t2.data());
            leaky_relu(t2.data(), mid * H * W);
            snprintf(buf, sizeof(buf), "layers.%d.conv.4.weight", rg);
            conv2d_3x3(t2.data(), mid, H, W, get_w(ctx, buf),
                       get_w(ctx, std::string(buf).replace(strlen(buf)-6, 6, "bias")), C, t3.data());
            x_chw = std::move(t3);
        } else {
            std::vector<float> conv_out(C * H * W);
            snprintf(buf, sizeof(buf), "layers.%d.conv.weight", rg);
            conv2d_3x3(x_chw.data(), C, H, W, get_w(ctx, buf),
                       get_w(ctx, std::string(buf).replace(strlen(buf)-6, 6, "bias")), C, conv_out.data());
            x_chw = std::move(conv_out);
        }

        // Rearrange back to NHW-C and add RG residual
        for (int y = 0; y < H; y++)
            for (int xi = 0; xi < W; xi++)
                for (int c = 0; c < C; c++)
                    x[(y*W + xi)*C + c] = x_chw[c*H*W + y*W + xi] + rg_input[(y*W + xi)*C + c];
    }

    // 5. Final norm
    const float * fn_w = get_w(ctx, "norm.weight");
    const float * fn_b = get_w(ctx, "norm.bias");
    if (fn_w) {
        std::vector<float> tmp(N * C);
        for (int i = 0; i < N; i++)
            layernorm_cpu(x.data() + i*C, C, fn_w, fn_b, tmp.data() + i*C);
        x = std::move(tmp);
    }

    // 6. Rearrange back to CHW
    std::vector<float> deep(C * H * W);
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++)
            for (int c = 0; c < C; c++)
                deep[c*H*W + y*W + xi] = x[(y*W + xi)*C + c];

    // 7. conv_after_body
    std::vector<float> cab(C * H * W);
    if (ctx->resi_connection == "3conv") {
        int mid = C / 4;
        std::vector<float> t1(mid * H * W), t2(mid * H * W);
        conv2d_3x3(deep.data(), C, H, W,
                   get_w(ctx, "conv_after_body.0.weight"),
                   get_w(ctx, "conv_after_body.0.bias"), mid, t1.data());
        leaky_relu(t1.data(), mid * H * W);
        conv1x1(t1.data(), mid, H, W,
                get_w(ctx, "conv_after_body.2.weight"),
                get_w(ctx, "conv_after_body.2.bias"), mid, t2.data());
        leaky_relu(t2.data(), mid * H * W);
        conv2d_3x3(t2.data(), mid, H, W,
                   get_w(ctx, "conv_after_body.4.weight"),
                   get_w(ctx, "conv_after_body.4.bias"), C, cab.data());
    } else {
        conv2d_3x3(deep.data(), C, H, W,
                   get_w(ctx, "conv_after_body.weight"),
                   get_w(ctx, "conv_after_body.bias"), C, cab.data());
    }

    // Global skip: cab + shallow
    for (int i = 0; i < C*H*W; i++) deep[i] = cab[i] + shallow[i];

    // 8. Upsample
    int r = ctx->upscale;
    int oH = H * r, oW = W * r;
    if (ctx->upsampler == "pixelshuffledirect") {
        int up_ch = 3 * r * r;
        std::vector<float> up(up_ch * H * W);
        conv2d_3x3(deep.data(), C, H, W,
                   get_w(ctx, "upsample.0.weight"),
                   get_w(ctx, "upsample.0.bias"), up_ch, up.data());
        pixel_shuffle(up.data(), up_ch, H, W, r, rgb_out);
    } else {
        // pixelshuffle: conv_before_upsample → upsample → conv_last
        int num_feat = 64;
        std::vector<float> t1(num_feat * H * W);
        conv2d_3x3(deep.data(), C, H, W,
                   get_w(ctx, "conv_before_upsample.0.weight"),
                   get_w(ctx, "conv_before_upsample.0.bias"), num_feat, t1.data());
        leaky_relu(t1.data(), num_feat * H * W);
        // Upsample blocks (2x each for power-of-2 scale)
        int cur_h = H, cur_w = W;
        int ups = (int)(log2f(r) + 0.5f);
        for (int u = 0; u < ups; u++) {
            int up_ch2 = 4 * num_feat;
            std::vector<float> t2(up_ch2 * cur_h * cur_w);
            char uname[64];
            snprintf(uname, sizeof(uname), "upsample.%d.weight", u*2);
            conv2d_3x3(t1.data(), num_feat, cur_h, cur_w,
                       get_w(ctx, uname),
                       get_w(ctx, std::string(uname).replace(strlen(uname)-6, 6, "bias")),
                       up_ch2, t2.data());
            std::vector<float> t3(num_feat * cur_h*2 * cur_w*2);
            pixel_shuffle(t2.data(), up_ch2, cur_h, cur_w, 2, t3.data());
            cur_h *= 2; cur_w *= 2;
            t1.resize(num_feat * cur_h * cur_w);
            t1 = std::move(t3);
        }
        conv2d_3x3(t1.data(), num_feat, oH, oW,
                   get_w(ctx, "conv_last.weight"),
                   get_w(ctx, "conv_last.bias"), 3, rgb_out);
    }

    // 9. Denormalize
    for (int c = 0; c < 3; c++)
        for (int i = 0; i < oH * oW; i++)
            rgb_out[c*oH*oW + i] = rgb_out[c*oH*oW + i] / ctx->img_range + ctx->mean[c];
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

dat_sr_context * dat_sr_init(const char * model_path, int n_threads) {
    (void)n_threads;
    auto * ctx = new dat_sr_context();
    ctx->mean[0] = 0.4488f; ctx->mean[1] = 0.4371f; ctx->mean[2] = 0.4040f;
    ctx->img_range = 1.0f;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) { delete ctx; return nullptr; }

    ctx->embed_dim = (int)core_gguf::kv_u32(meta, "dat.embed_dim", 60);
    ctx->upscale = (int)core_gguf::kv_u32(meta, "dat.upscale", 2);
    ctx->num_heads = (int)core_gguf::kv_u32(meta, "dat.num_heads", 6);
    ctx->depth = core_gguf::kv_i32_array(meta, "dat.depth");
    if (ctx->depth.empty()) ctx->depth = {18};
    ctx->resi_connection = core_gguf::kv_str(meta, "dat.resi_connection", "3conv");
    ctx->upsampler = core_gguf::kv_str(meta, "dat.upsampler", "pixelshuffledirect");

    // Read split_size from GGUF if available, default [8, 32] for DAT-light
    ctx->split_size[0] = 8;
    ctx->split_size[1] = 32;

    core_gguf::free_metadata(meta);

    ggml_backend_t backend = ggml_backend_init_best();
    if (!core_gguf::load_weights(model_path, backend, "dat", ctx->wl)) {
        ggml_backend_free(backend);
        delete ctx;
        return nullptr;
    }
    ggml_backend_free(backend);
    ctx->tensors = std::move(ctx->wl.tensors);

    int total_blocks = 0;
    for (int d : ctx->depth) total_blocks += d;
    fprintf(stderr, "dat_sr: loaded embed=%d, heads=%d, depth=%d blocks, "
            "split=[%d,%d], upscale=%dx, %s+%s\n",
            ctx->embed_dim, ctx->num_heads, total_blocks,
            ctx->split_size[0], ctx->split_size[1],
            ctx->upscale, ctx->resi_connection.c_str(), ctx->upsampler.c_str());
    ctx->bench = (std::getenv("CRISPEMBED_DAT_SR_BENCH") != nullptr);
    return ctx;
}

int dat_sr_process(dat_sr_context * ctx,
                    const uint8_t * pixels, int w, int h,
                    int tile_w, int tile_h,
                    uint8_t ** out, int * out_w, int * out_h) {
    if (!ctx || !pixels || w <= 0 || h <= 0) return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    int r = ctx->upscale;
    int oW = w * r, oH = h * r;
    int max_sp = std::max(ctx->split_size[0], ctx->split_size[1]);

    // Default tile size: must be multiple of max_split_size
    if (tile_w <= 0) tile_w = max_sp * 2; // 64 for [8,32]
    if (tile_h <= 0) tile_h = max_sp * 2;
    // Round tile size up to multiple of max_sp
    tile_w = ((tile_w + max_sp - 1) / max_sp) * max_sp;
    tile_h = ((tile_h + max_sp - 1) / max_sp) * max_sp;

    int overlap = 8;
    int step_w = tile_w - overlap;
    int step_h = tile_h - overlap;

    // Pad image to multiple of max_sp
    int pad_w = ((w + max_sp - 1) / max_sp) * max_sp;
    int pad_h = ((h + max_sp - 1) / max_sp) * max_sp;

    // Convert uint8 RGB to float CHW [0,1] with padding
    auto t_pre = std::chrono::steady_clock::now();
    std::vector<float> input(3 * pad_h * pad_w, 0.0f);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int c = 0; c < 3; c++)
                input[c*pad_h*pad_w + y*pad_w + x] = pixels[(y*w + x)*3 + c] / 255.0f;
    // Mirror-pad the extra pixels
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < pad_h; y++) {
            int sy = y < h ? y : 2*h - y - 2;
            if (sy < 0) sy = 0;
            for (int x = w; x < pad_w; x++) {
                int sx = 2*w - x - 2;
                if (sx < 0) sx = 0;
                input[c*pad_h*pad_w + y*pad_w + x] = input[c*pad_h*pad_w + sy*pad_w + sx];
            }
        }
        for (int y = h; y < pad_h; y++) {
            int sy = 2*h - y - 2;
            if (sy < 0) sy = 0;
            for (int x = 0; x < pad_w; x++) {
                input[c*pad_h*pad_w + y*pad_w + x] = input[c*pad_h*pad_w + sy*pad_w + x];
            }
        }
    }
    if (bench) {
        auto t_pre_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[dat_sr-bench] preprocess: %.1f ms\n",
                ms_f(t_pre_end - t_pre).count());
    }

    int pad_oW = pad_w * r, pad_oH = pad_h * r;
    std::vector<float> accum(3 * pad_oH * pad_oW, 0.0f);
    std::vector<float> weight(pad_oH * pad_oW, 0.0f);

    // If the padded image fits in one tile, skip tiling entirely
    bool single_tile = (pad_w <= tile_w && pad_h <= tile_h);

    if (single_tile) {
        fprintf(stderr, "dat_sr: %dx%d → %dx%d (%dx), single pass\n", w, h, oW, oH, r);
        auto t_tile = std::chrono::steady_clock::now();
        std::vector<float> full_out(3 * pad_oH * pad_oW);
        dat_forward(ctx, input.data(), pad_h, pad_w, full_out.data());
        // Copy to accum, weight=1
        accum = std::move(full_out);
        std::fill(weight.begin(), weight.end(), 1.0f);
        if (bench) {
            auto t_tile_end = std::chrono::steady_clock::now();
            fprintf(stderr, "[dat_sr-bench] tile 0,0: %.1f ms\n",
                    ms_f(t_tile_end - t_tile).count());
        }
    } else {
        int n_tiles_x = (pad_w + step_w - 1) / step_w;
        int n_tiles_y = (pad_h + step_h - 1) / step_h;
        fprintf(stderr, "dat_sr: %dx%d → %dx%d (%dx), tiles=%dx%d (%dx%d)\n",
                w, h, oW, oH, r, n_tiles_x, n_tiles_y, tile_w, tile_h);

        for (int ty = 0; ty < n_tiles_y; ty++) {
            for (int tx = 0; tx < n_tiles_x; tx++) {
                int x0 = std::min(tx * step_w, std::max(0, pad_w - tile_w));
                int y0 = std::min(ty * step_h, std::max(0, pad_h - tile_h));
                int tw = std::min(tile_w, pad_w - x0);
                int th = std::min(tile_h, pad_h - y0);
                tw = (tw / max_sp) * max_sp;
                th = (th / max_sp) * max_sp;
                if (tw <= 0 || th <= 0) continue;

                std::vector<float> tile_in(3 * th * tw);
                for (int c = 0; c < 3; c++)
                    for (int y = 0; y < th; y++)
                        for (int x = 0; x < tw; x++)
                            tile_in[c*th*tw + y*tw + x] = input[c*pad_h*pad_w + (y0+y)*pad_w + (x0+x)];

                int otw = tw * r, oth = th * r;
                std::vector<float> tile_out(3 * oth * otw);
                auto t_tile = std::chrono::steady_clock::now();
                dat_forward(ctx, tile_in.data(), th, tw, tile_out.data());
                if (bench) {
                    auto t_tile_end = std::chrono::steady_clock::now();
                    fprintf(stderr, "[dat_sr-bench] tile %d,%d: %.1f ms\n",
                            ty, tx, ms_f(t_tile_end - t_tile).count());
                }

                int ox0 = x0 * r, oy0 = y0 * r;
                for (int y = 0; y < oth; y++) {
                    for (int x = 0; x < otw; x++) {
                        float wy = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * y / (oth - 1)));
                        float wx = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * x / (otw - 1)));
                        float wt = wy * wx;
                        int dy = oy0 + y, dx = ox0 + x;
                        for (int c = 0; c < 3; c++)
                            accum[c*pad_oH*pad_oW + dy*pad_oW + dx] += tile_out[c*oth*otw + y*otw + x] * wt;
                        weight[dy*pad_oW + dx] += wt;
                    }
                }
            }
        }
    }

    // Normalize and convert to uint8
    auto t_post = std::chrono::steady_clock::now();
    *out = (uint8_t *)malloc(oW * oH * 3);
    if (!*out) return -1;
    for (int y = 0; y < oH; y++)
        for (int x = 0; x < oW; x++) {
            float w_val = weight[y*pad_oW + x];
            if (w_val < 1e-6f) w_val = 1.0f;
            for (int c = 0; c < 3; c++) {
                float v = accum[c*pad_oH*pad_oW + y*pad_oW + x] / w_val * 255.0f;
                (*out)[(y*oW + x)*3 + c] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
        }

    *out_w = oW;
    *out_h = oH;
    if (bench) {
        auto t_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[dat_sr-bench] postprocess: %.1f ms\n",
                ms_f(t_end - t_post).count());
        fprintf(stderr, "[dat_sr-bench] total: %.1f ms\n",
                ms_f(t_end - t_total).count());
    }
    return 0;
}

void dat_sr_free_image(uint8_t * pixels) { free(pixels); }

void dat_sr_free(dat_sr_context * ctx) {
    if (ctx) {
        core_gguf::free_weights(ctx->wl);
        delete ctx;
    }
}
