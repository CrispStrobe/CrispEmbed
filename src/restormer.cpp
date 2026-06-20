// restormer.cpp — Restormer image restoration (CPU-scalar).
//
// U-Net forward:
//   patch_embed(3→48) → enc1(4×TB,48) → down → enc2(6×TB,96) → down
//   → enc3(6×TB,192) → down → latent(8×TB,384) → up+cat+reduce
//   → dec3(6×TB,192) → up+cat+reduce → dec2(6×TB,96) → up+cat
//   → dec1(4×TB,96) → refine(4×TB,96) → output(96→3) + input residual
//
// TransformerBlock:
//   LN → MDTA → residual → LN → GDFN → residual
//
// MDTA (transposed attention):
//   QKV = DWConv3(Conv1x1(x)) → split → reshape [B,h,C/h,HW]
//   → L2-normalize Q,K → attn = Q@K^T * temperature → softmax → attn@V
//   → reshape → Conv1x1
//
// GDFN: Conv1x1(dim→hidden*2) → DWConv3 → chunk → GELU(x1)*x2 → Conv1x1

#include "restormer.h"
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
#include <string>
#include <vector>

// ── Helpers ────────────────────────────────────────────────────────────

static const float * rst_to_f32(const ggml_tensor * t, std::vector<float> & buf) {
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

// Conv2D: weight [OC, IC, KH, KW] (or [OC, 1, KH, KW] for depthwise)
struct restormer_context;  // forward decl
static void rst_conv2d_ggml(restormer_context * ctx,
                             const float * input, int ic, int h, int w,
                             ggml_tensor * wt, ggml_tensor * bt,
                             int oc, int kh, int kw, int pad, int groups,
                             float * output);

static void rst_conv2d(const float * input, int ic, int h, int w,
                       const float * weight, const float * bias,
                       int oc, int kh, int kw, int pad, int groups,
                       float * output) {
    int oh = h + 2 * pad - kh + 1, ow = w + 2 * pad - kw + 1;
    int ic_g = ic / groups, oc_g = oc / groups;
    for (int g = 0; g < groups; g++) {
        for (int o = 0; o < oc_g; o++) {
            int oc_abs = g * oc_g + o;
            float b = bias ? bias[oc_abs] : 0.0f;
            for (int oy = 0; oy < oh; oy++) {
                for (int ox = 0; ox < ow; ox++) {
                    float sum = b;
                    for (int c = 0; c < ic_g; c++) {
                        int ic_abs = g * ic_g + c;
                        for (int ky = 0; ky < kh; ky++) {
                            int iy = oy + ky - pad;
                            if (iy < 0 || iy >= h) continue;
                            for (int kx = 0; kx < kw; kx++) {
                                int ix = ox + kx - pad;
                                if (ix < 0 || ix >= w) continue;
                                sum += input[ic_abs * h * w + iy * w + ix]
                                     * weight[oc_abs * ic_g * kh * kw + c * kh * kw + ky * kw + kx];
                            }
                        }
                    }
                    output[oc_abs * oh * ow + oy * ow + ox] = sum;
                }
            }
        }
    }
}

// BiasFree LayerNorm: x=[C,H,W], normalize over C at each spatial position
// Reshape to [HW, C], var over C, scale by weight
static void rst_layernorm_bf(float * data, int c, int h, int w,
                              const float * weight) {
    int hw = h * w;
    for (int i = 0; i < hw; i++) {
        // Compute variance over channels: var(-1, unbiased=False) = mean((x - mean)²)
        float mean = 0;
        for (int ch = 0; ch < c; ch++) mean += data[ch * hw + i];
        mean /= c;
        float var = 0;
        for (int ch = 0; ch < c; ch++) {
            float d = data[ch * hw + i] - mean;
            var += d * d;
        }
        var /= c;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int ch = 0; ch < c; ch++)
            data[ch * hw + i] = data[ch * hw + i] * inv * weight[ch];
    }
}

// WithBias LayerNorm: x=[C,H,W], normalize and shift
static void rst_layernorm_wb(float * data, int c, int h, int w,
                              const float * weight, const float * bias) {
    int hw = h * w;
    for (int i = 0; i < hw; i++) {
        float mean = 0;
        for (int ch = 0; ch < c; ch++) mean += data[ch * hw + i];
        mean /= c;
        float var = 0;
        for (int ch = 0; ch < c; ch++) {
            float d = data[ch * hw + i] - mean;
            var += d * d;
        }
        var /= c;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int ch = 0; ch < c; ch++)
            data[ch * hw + i] = (data[ch * hw + i] - mean) * inv * weight[ch] + bias[ch];
    }
}

static float rst_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// PixelShuffle: [C*r*r, H, W] → [C, H*r, W*r]
static void rst_pixel_shuffle(const float * input, int c_in, int h, int w,
                               int r, float * output) {
    int c_out = c_in / (r * r), oh = h * r, ow = w * r;
    for (int c = 0; c < c_out; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                output[c * oh * ow + y * ow + x] =
                    input[(c * r * r + (y % r) * r + (x % r)) * h * w + (y / r) * w + (x / r)];
}

// PixelUnshuffle: [C, H, W] → [C*r*r, H/r, W/r]
static void rst_pixel_unshuffle(const float * input, int c, int h, int w,
                                 int r, float * output) {
    int oh = h / r, ow = w / r, c_out = c * r * r;
    for (int oc = 0; oc < c_out; oc++) {
        int ic = oc / (r * r);
        int ry = (oc % (r * r)) / r;
        int rx = oc % r;
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                output[oc * oh * ow + y * ow + x] =
                    input[ic * h * w + (y * r + ry) * w + (x * r + rx)];
    }
}

// ── MDTA (Multi-DConv Head Transposed Attention) ───────────────────────

static void rst_mdta(const float * x, int C, int H, int W, int n_heads,
                     const float * qkv_w, const float * qkv_dw_w,
                     const float * proj_w, const float * temperature,
                     float * output,
                     std::vector<float> & scratch) {
    int HW = H * W;
    int C3 = C * 3;
    int d_k = C / n_heads;

    // QKV = DWConv3(Conv1x1(x))
    std::vector<float> qkv_1x1(C3 * H * W);
    rst_conv2d(x, C, H, W, qkv_w, nullptr, C3, 1, 1, 0, 1, qkv_1x1.data());
    std::vector<float> qkv(C3 * H * W);
    rst_conv2d(qkv_1x1.data(), C3, H, W, qkv_dw_w, nullptr, C3, 3, 3, 1, C3, qkv.data());

    // Split into Q, K, V — each [C, HW]
    float * Q = qkv.data();
    float * K = qkv.data() + C * HW;
    float * V = qkv.data() + 2 * C * HW;

    // L2-normalize Q and K over spatial dim (last dim = HW)
    // Q is [C, HW], normalize each row
    for (int c = 0; c < C; c++) {
        float norm_q = 0, norm_k = 0;
        for (int i = 0; i < HW; i++) {
            norm_q += Q[c * HW + i] * Q[c * HW + i];
            norm_k += K[c * HW + i] * K[c * HW + i];
        }
        norm_q = 1.0f / (sqrtf(norm_q) + 1e-12f);
        norm_k = 1.0f / (sqrtf(norm_k) + 1e-12f);
        for (int i = 0; i < HW; i++) {
            Q[c * HW + i] *= norm_q;
            K[c * HW + i] *= norm_k;
        }
    }

    // Transposed attention per head: attn[h] = Q_h @ K_h^T * temp → softmax → @ V_h
    // Q_h, K_h, V_h are [d_k, HW]
    // attn = Q_h @ K_h^T = [d_k, d_k] (channel attention, NOT spatial!)
    std::vector<float> attn_out(C * HW);

    for (int h = 0; h < n_heads; h++) {
        int off = h * d_k;
        float temp = temperature[h];

        // attn[i,j] = sum_hw Q[off+i, hw] * K[off+j, hw] * temp
        std::vector<float> attn(d_k * d_k);
        for (int i = 0; i < d_k; i++) {
            for (int j = 0; j < d_k; j++) {
                float s = 0;
                for (int hw = 0; hw < HW; hw++)
                    s += Q[(off + i) * HW + hw] * K[(off + j) * HW + hw];
                attn[i * d_k + j] = s * temp;
            }
            // Softmax over j
            float maxv = -1e9f;
            for (int j = 0; j < d_k; j++)
                if (attn[i * d_k + j] > maxv) maxv = attn[i * d_k + j];
            float sumexp = 0;
            for (int j = 0; j < d_k; j++) {
                attn[i * d_k + j] = expf(attn[i * d_k + j] - maxv);
                sumexp += attn[i * d_k + j];
            }
            for (int j = 0; j < d_k; j++)
                attn[i * d_k + j] /= sumexp;
        }

        // out[i, hw] = sum_j attn[i,j] * V[off+j, hw]
        for (int i = 0; i < d_k; i++) {
            for (int hw = 0; hw < HW; hw++) {
                float s = 0;
                for (int j = 0; j < d_k; j++)
                    s += attn[i * d_k + j] * V[(off + j) * HW + hw];
                attn_out[(off + i) * HW + hw] = s;
            }
        }
    }

    // Output projection
    rst_conv2d(attn_out.data(), C, H, W, proj_w, nullptr, C, 1, 1, 0, 1, output);
}

// (rst_gdfn stub removed — dead code, unused)

// GDFN with explicit hidden size
static void rst_gdfn_ex(const float * x, int C, int H, int W, int hidden2,
                        const float * in_w, const float * dw_w, const float * out_w,
                        float * output,
                        std::vector<float> & tmp) {
    int HW = H * W;
    int hidden = hidden2 / 2;

    // project_in: Conv1x1(C → hidden*2)
    tmp.resize(hidden2 * HW);
    rst_conv2d(x, C, H, W, in_w, nullptr, hidden2, 1, 1, 0, 1, tmp.data());

    // DWConv3x3(hidden*2, groups=hidden*2)
    std::vector<float> dw_out(hidden2 * HW);
    rst_conv2d(tmp.data(), hidden2, H, W, dw_w, nullptr, hidden2, 3, 3, 1, hidden2, dw_out.data());

    // chunk into x1, x2 → GELU(x1) * x2
    std::vector<float> gated(hidden * HW);
    for (int c = 0; c < hidden; c++)
        for (int i = 0; i < HW; i++)
            gated[c * HW + i] = rst_gelu(dw_out[c * HW + i]) * dw_out[(c + hidden) * HW + i];

    // project_out: Conv1x1(hidden → C)
    rst_conv2d(gated.data(), hidden, H, W, out_w, nullptr, C, 1, 1, 0, 1, output);
}

// ── TransformerBlock ───────────────────────────────────────────────────

struct rst_block_weights {
    const float * norm1_w;
    const float * norm1_b;  // null for BiasFree
    const float * attn_qkv_w;
    const float * attn_qkv_dw_w;
    const float * attn_proj_w;
    const float * attn_temp;
    const float * norm2_w;
    const float * norm2_b;  // null for BiasFree
    const float * ffn_in_w;
    const float * ffn_dw_w;
    const float * ffn_out_w;
    int n_heads;
    int hidden2;  // ffn hidden*2
};

static void rst_transformer_block(float * x, int C, int H, int W,
                                   const rst_block_weights & wt,
                                   std::vector<float> & scratch) {
    int n = C * H * W;

    // LN1 → MDTA → residual
    std::vector<float> normed(n);
    memcpy(normed.data(), x, n * sizeof(float));
    if (wt.norm1_b)
        rst_layernorm_wb(normed.data(), C, H, W, wt.norm1_w, wt.norm1_b);
    else
        rst_layernorm_bf(normed.data(), C, H, W, wt.norm1_w);

    std::vector<float> attn_out(n);
    rst_mdta(normed.data(), C, H, W, wt.n_heads,
             wt.attn_qkv_w, wt.attn_qkv_dw_w, wt.attn_proj_w, wt.attn_temp,
             attn_out.data(), scratch);
    for (int i = 0; i < n; i++) x[i] += attn_out[i];

    // LN2 → GDFN → residual
    memcpy(normed.data(), x, n * sizeof(float));
    if (wt.norm2_b)
        rst_layernorm_wb(normed.data(), C, H, W, wt.norm2_w, wt.norm2_b);
    else
        rst_layernorm_bf(normed.data(), C, H, W, wt.norm2_w);

    std::vector<float> ffn_out(n);
    rst_gdfn_ex(normed.data(), C, H, W, wt.hidden2,
                wt.ffn_in_w, wt.ffn_dw_w, wt.ffn_out_w,
                ffn_out.data(), scratch);
    for (int i = 0; i < n; i++) x[i] += ffn_out[i];
}

// ── ggml full-graph Transformer block ─────────────────────────────────

// Raw tensor version of block weights for ggml graph
struct rst_block_raw {
    ggml_tensor * norm1_w, * norm1_b;
    ggml_tensor * qkv_w, * qkv_dw_w, * proj_w, * temperature;
    ggml_tensor * norm2_w, * norm2_b;
    ggml_tensor * ffn_in_w, * ffn_dw_w, * ffn_out_w;
    int n_heads, hidden2;
};

struct restormer_context;

// LN2d in ggml: permute [W,H,C]→[C,H,W], norm over C, scale+bias, permute back
static ggml_tensor * rst_ln2d_ggml(ggml_context * g, ggml_tensor * x,
                                    ggml_tensor * w, ggml_tensor * b) {
    // x: [W, H, C]. Permute so C is ne[0]
    ggml_tensor * t = ggml_cont(g, ggml_permute(g, x, 2, 1, 0, 3)); // [C, H, W]
    t = ggml_norm(g, t, 1e-5f); // normalize over ne[0]=C
    if (w) {
        if (w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);
        t = ggml_mul(g, t, w); // broadcast [C] over H,W
    }
    if (b) {
        if (b->type != GGML_TYPE_F32) b = ggml_cast(g, b, GGML_TYPE_F32);
        t = ggml_add(g, t, b);
    }
    t = ggml_cont(g, ggml_permute(g, t, 2, 1, 0, 3)); // back to [W, H, C]
    return t;
}

// Prep weight: ensure 2D [IC*KH*KW, OC] F32 for conv, or pre-permuted.
static ggml_tensor * rst_prep_w(ggml_context * g, ggml_tensor * w,
                                 int IC, int KH, int KW) {
    if (w->type != GGML_TYPE_F32) w = ggml_cast(g, w, GGML_TYPE_F32);
    if (ggml_n_dims(w) == 2) {
        int64_t ik = (int64_t)IC * KH * KW;
        if (w->ne[0] == ik)
            w = ggml_reshape_4d(g, w, KW, KH, IC, w->ne[1]);
        else
            w = ggml_reshape_4d(g, ggml_cont(g, ggml_transpose(g, w)), KW, KH, IC, w->ne[0]);
    } else if (ggml_n_dims(w) >= 4) {
        w = ggml_cont(g, ggml_permute(g, w, 3, 2, 1, 0));
    }
    return ggml_cast(g, w, GGML_TYPE_F16);
}

// Forward declarations — implementations after restormer_free
static ggml_cgraph * rst_build_block_graph(restormer_context * ctx,
                                            const rst_block_raw & bw,
                                            int C, int H, int W);
static void rst_transformer_block_ggml(restormer_context * ctx,
                                        float * x, int C, int H, int W,
                                        const rst_block_raw & bw);

// ---- begin deferred implementations (need restormer_context) ----
// They are placed after restormer_free. Everything between here and
// the Context section is commented out.
#if 0
static ggml_cgraph * rst_build_block_graph_DISABLED(restormer_context * ctx,
                                            const rst_block_raw & bw,
                                            int C, int H, int W) {
    int HW = H * W, C3 = C * 3;
    int d_k = C / bw.n_heads;
    int hidden2 = bw.hidden2, hidden = hidden2 / 2;

    int graph_size = 256;  // generous
    size_t buf_size = ggml_tensor_overhead() * (graph_size + 100)
                    + ggml_graph_overhead_custom(graph_size, false);
    std::vector<uint8_t> & meta = *reinterpret_cast<std::vector<uint8_t>*>(
        &ctx->enc_meta_block);  // will add this field
    meta.resize(buf_size);
    ggml_init_params ip = { buf_size, meta.data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(g, graph_size, false);

    // Input: x [W, H, C]
    ggml_tensor * x = ggml_new_tensor_3d(g, GGML_TYPE_F32, W, H, C);
    ggml_set_name(x, "x"); ggml_set_input(x);

    // ---- LN1 + MDTA + residual ----
    ggml_tensor * normed = rst_ln2d_ggml(g, x, bw.norm1_w, bw.norm1_b);

    // QKV: Conv1×1(C→3C) + DWConv3×3(3C)
    ggml_tensor * qkv_w = rst_prep_w(g, bw.qkv_w, C, 1, 1);
    ggml_tensor * qkv = ggml_conv_2d(g, qkv_w, normed, 1, 1, 0, 0, 1, 1);

    ggml_tensor * dw_w = rst_prep_w(g, bw.qkv_dw_w, 1, 3, 3);
    qkv = ggml_conv_2d_dw(g, dw_w, qkv, 1, 1, 1, 1, 1, 1);

    // L2-normalize Q, K over spatial dim
    // qkv: [W, H, 3C]. Split into Q[W,H,C], K[W,H,C], V[W,H,C]
    // In ggml WHC layout: ne[2]=3C. Views at channel offsets.
    size_t plane = W * H * sizeof(float);
    ggml_tensor * Q = ggml_view_3d(g, qkv, W, H, C, qkv->nb[1], qkv->nb[2], 0);
    ggml_tensor * K = ggml_view_3d(g, qkv, W, H, C, qkv->nb[1], qkv->nb[2], C * plane);
    ggml_tensor * V = ggml_view_3d(g, qkv, W, H, C, qkv->nb[1], qkv->nb[2], 2 * C * plane);

    // L2 norm Q, K over spatial: for each channel, normalize the HW-length vector
    // Reshape to [HW, C], normalize over ne[0]=HW, reshape back
    // Actually ggml_norm normalizes over ne[0]. We need norm over spatial (HW).
    // Q is [W, H, C]. If we reshape to [W*H, C] = [HW, C], then ne[0]=HW.
    // ggml_norm over ne[0]=HW → normalizes each channel's spatial vector. ✓
    Q = ggml_cont(g, Q);
    K = ggml_cont(g, K);
    V = ggml_cont(g, V);
    ggml_tensor * Q_flat = ggml_reshape_2d(g, Q, HW, C);
    ggml_tensor * K_flat = ggml_reshape_2d(g, K, HW, C);
    Q_flat = ggml_norm(g, Q_flat, 1e-12f);  // L2-ish via layernorm (mean-centered)
    K_flat = ggml_norm(g, K_flat, 1e-12f);

    // Transposed attention: per head, attn = Q_h^T @ K_h → [d_k, d_k]
    // Q_flat [HW, C] → reshape to [HW, d_k, n_heads]
    // ggml_mul_mat on each head slice
    // For simplicity, do full C attention: attn = Q_flat^T @ K_flat → [C, C]
    // Then apply softmax per row, multiply with V
    // attn = ggml_mul_mat(K_flat[HW, C], Q_flat[HW, C]) → [C, C]
    ggml_tensor * V_flat = ggml_reshape_2d(g, V, HW, C);

    // Scale by temperature (per-head learnable)
    if (bw.temperature) {
        // temperature is [n_heads], expand to [C] by repeating d_k times
        // For now, apply as scalar scale (approximate)
        // TODO: proper per-head temperature
    }

    ggml_tensor * attn = ggml_mul_mat(g, K_flat, Q_flat); // [C, C]
    attn = ggml_soft_max(g, attn);

    // out = attn @ V_flat^T: [C, C] @ [C, HW] → [C, HW]? No...
    // attn is [C, C], V_flat is [HW, C]
    // We want out[c, hw] = sum_j attn[c, j] * V[j, hw]
    // = ggml_mul_mat(V_flat[HW, C], attn[C, C]) → [C, C]... wrong dims
    // Actually: out = attn @ V^T where V^T is [C, HW]
    // attn[C, C] @ V^T[C, HW]... ggml_mul_mat(a[K,M], b[K,N]) → [M,N]
    // So: ggml_mul_mat(V_flat_T[C, HW], attn_T) ... this is getting complex.

    // Simpler: out_flat = ggml_mul_mat(attn, V_flat) where attn=[C,C], V_flat=[HW,C]
    // ggml_mul_mat(attn[C,C], V_flat[HW,C]): K=C, M=C, N=HW → wrong, K must match ne[0]
    // attn ne[0]=C, V_flat ne[0]=HW. K=ne[0] must match: C != HW.

    // I need: result[c_out, hw] = sum_j attn[c_out, j] * V_flat[hw, j]
    // = result[c_out, hw] = sum_j attn[c_out, j] * V_flat[hw, j]
    // In ggml: ggml_mul_mat(a, b) where a=[K, M], b=[K, N] → result=[M, N]
    //   result[m, n] = sum_k a[k, m] * b[k, n]
    // I want: result[c_out, hw] = sum_j V[hw, j] * attn[c_out, j]
    // So: a = attn^T [C, C], b = V_flat [HW, C], K=C → result [C, HW] ✓
    // attn^T = ggml_transpose(attn)
    ggml_tensor * attn_t = ggml_transpose(g, attn);
    ggml_tensor * out_flat = ggml_mul_mat(g, attn_t, V_flat); // [C, HW]

    // Reshape back to [W, H, C]
    ggml_tensor * attn_out = ggml_reshape_3d(g, out_flat, W, H, C);

    // Output projection: Conv1×1(C→C)
    ggml_tensor * proj_w = rst_prep_w(g, bw.proj_w, C, 1, 1);
    attn_out = ggml_conv_2d(g, proj_w, attn_out, 1, 1, 0, 0, 1, 1);

    // Residual
    ggml_tensor * res1 = ggml_add(g, x, attn_out);

    // ---- LN2 + GDFN + residual ----
    normed = rst_ln2d_ggml(g, res1, bw.norm2_w, bw.norm2_b);

    // GDFN: Conv1×1(C→hidden*2) → DWConv3×3 → GELU gate → Conv1×1(hidden→C)
    ggml_tensor * ffn_in_w = rst_prep_w(g, bw.ffn_in_w, C, 1, 1);
    ggml_tensor * up = ggml_conv_2d(g, ffn_in_w, normed, 1, 1, 0, 0, 1, 1);

    ggml_tensor * ffn_dw_w = rst_prep_w(g, bw.ffn_dw_w, 1, 3, 3);
    up = ggml_conv_2d_dw(g, ffn_dw_w, up, 1, 1, 1, 1, 1, 1);

    // GELU gate: split channels, GELU(first) * second
    ggml_tensor * up1 = ggml_view_3d(g, up, W, H, hidden, up->nb[1], up->nb[2], 0);
    ggml_tensor * up2 = ggml_view_3d(g, up, W, H, hidden, up->nb[1], up->nb[2], hidden * plane);
    up1 = ggml_cont(g, up1);
    up2 = ggml_cont(g, up2);
    ggml_tensor * gated = ggml_mul(g, ggml_gelu(g, up1), up2);

    // project_out: Conv1×1(hidden→C)
    ggml_tensor * ffn_out_w = rst_prep_w(g, bw.ffn_out_w, hidden, 1, 1);
    ggml_tensor * ffn_out = ggml_conv_2d(g, ffn_out_w, gated, 1, 1, 0, 0, 1, 1);

    // Residual
    ggml_tensor * result = ggml_add(g, res1, ffn_out);

    ggml_set_name(result, "out");
    ggml_set_output(result);
    ggml_build_forward_expand(gf, result);
    return gf;
}

// Run one Transformer block via ggml graph
static void rst_transformer_block_ggml(restormer_context * ctx,
                                        float * x, int C, int H, int W,
                                        const rst_block_raw & bw) {
    ggml_cgraph * gf = rst_build_block_graph(ctx, bw, C, H, W);
    ggml_backend_sched_reset(ctx->enc_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->enc_sched, gf)) {
        fprintf(stderr, "restormer: block graph alloc failed\n");
        return;
    }

    int n = C * H * W;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), x, 0, n * sizeof(float));

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
    if (ggml_backend_sched_graph_compute(ctx->enc_sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "restormer: block graph compute failed\n");
        return;
    }

    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out"), x, 0, n * sizeof(float));
}
#endif  // deferred implementations

// ── Context ────────────────────────────────────────────────────────────

struct restormer_context {
    int dim;
    std::vector<int> num_blocks;
    std::vector<int> heads;
    float ffn_factor;
    int n_refine;
    bool has_bias;
    int n_threads;
    bool bench;

    core_gguf::WeightLoad wl;
    core_cpu::DequantCache dcache;

    // ggml conv infrastructure
    ggml_backend_t       enc_backend  = nullptr;
    ggml_backend_sched_t enc_sched    = nullptr;
    std::vector<uint8_t> enc_meta_block;  // graph metadata for per-block graphs

    // Pre-permuted conv weights: 4D PyTorch→2D ggml [IC*KH*KW, OC] F32
    // Stored per tensor name to avoid per-call permute overhead.
    ggml_context * prep_ctx = nullptr;
    ggml_backend_buffer_t prep_buf = nullptr;
    std::unordered_map<std::string, ggml_tensor *> prep_weights;

    const float * get(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) return nullptr;
        return dcache.get(t);
    }

    ggml_tensor * get_raw(const std::string & name) {
        // Return pre-permuted 2D weight if available (avoids per-call permute)
        auto it = prep_weights.find(name);
        if (it != prep_weights.end()) return it->second;
        return core_gguf::try_get(wl.tensors, name.c_str());
    }

    int64_t dim0(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        return t ? t->ne[0] : 0;
    }
};

restormer_context * restormer_init(const char * model_path, int n_threads) {
    auto * ctx = new restormer_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) { fprintf(stderr, "restormer: failed to open %s\n", model_path); delete ctx; return nullptr; }

    ctx->dim        = core_gguf::kv_u32(meta, "restormer.dim", 48);
    ctx->n_refine   = core_gguf::kv_u32(meta, "restormer.num_refinement_blocks", 4);
    ctx->has_bias   = core_gguf::kv_u32(meta, "restormer.bias", 0) != 0;
    ctx->num_blocks = core_gguf::kv_i32_array(meta, "restormer.num_blocks");
    ctx->heads      = core_gguf::kv_i32_array(meta, "restormer.heads");

    // Read ffn_factor — stored as f32 KV
    int idx = gguf_find_key(meta, "restormer.ffn_expansion_factor");
    ctx->ffn_factor = idx >= 0 ? gguf_get_val_f32(meta, idx) : 2.66f;

    core_gguf::free_metadata(meta);

    bool force_cpu = (getenv("RESTORMER_FORCE_CPU") && atoi(getenv("RESTORMER_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, backend, "restormer", ctx->wl)) {
        fprintf(stderr, "restormer: failed to load weights\n");
        ggml_backend_free(backend); delete ctx; return nullptr;
    }
    ggml_backend_free(backend);

    fprintf(stderr, "restormer: dim=%d, blocks=[%d,%d,%d,%d], heads=[%d,%d,%d,%d], "
            "ffn=%.2f, refine=%d, %d tensors\n",
            ctx->dim,
            ctx->num_blocks[0], ctx->num_blocks[1], ctx->num_blocks[2], ctx->num_blocks[3],
            ctx->heads[0], ctx->heads[1], ctx->heads[2], ctx->heads[3],
            ctx->ffn_factor, ctx->n_refine, (int)ctx->wl.tensors.size());
    ctx->bench = (std::getenv("CRISPEMBED_RESTORMER_BENCH") != nullptr);

    ctx->enc_backend = ggml_backend_cpu_init();
    if (ctx->enc_backend) {
        ggml_backend_cpu_set_n_threads(ctx->enc_backend, ctx->n_threads);
        ggml_backend_t backends[] = { ctx->enc_backend };
        ctx->enc_sched = ggml_backend_sched_new(backends, nullptr, 1, 4096, false, false);
    }

    // Pre-permute 4D conv weights → 2D [IC*KH*KW, OC] F32 for ggml_conv_2d.
    // This avoids per-call permute overhead (4D→F32→permute→cont→F16 on every conv).
    {
        // Count 4D conv weights to size the context
        int n4d = 0;
        for (auto & [name, t] : ctx->wl.tensors)
            if (ggml_n_dims(t) == 4 && name.find("weight") != std::string::npos) n4d++;

        if (n4d > 0) {
            // Allocate ggml context for pre-permuted weights (generous)
            size_t ctx_size = ggml_tensor_overhead() * (n4d + 1) + 4 * 1024 * 1024;
            ggml_init_params ip = { ctx_size, nullptr, true };  // no_alloc: backend allocates
            ctx->prep_ctx = ggml_init(ip);

            // Create F32 2D tensors and fill with permuted data
            size_t total_bytes = 0;
            std::vector<std::pair<std::string, ggml_tensor *>> to_fill;
            for (auto & [name, t] : ctx->wl.tensors) {
                if (ggml_n_dims(t) != 4 || name.find("weight") == std::string::npos) continue;
                // t: ne[0]=OC, ne[1]=IC, ne[2]=KH, ne[3]=KW (PyTorch order)
                int64_t OC = t->ne[0], IC = t->ne[1], KH = t->ne[2], KW = t->ne[3];
                // Create 2D: [IC*KH*KW, OC] — ggml conv_2d reshape format
                ggml_tensor * w2d = ggml_new_tensor_2d(ctx->prep_ctx, GGML_TYPE_F32, IC*KH*KW, OC);
                ggml_set_name(w2d, name.c_str());
                total_bytes += ggml_nbytes(w2d);
                to_fill.push_back({name, w2d});
            }

            ctx->prep_buf = ggml_backend_alloc_ctx_tensors(ctx->prep_ctx, ctx->enc_backend);

            // Fill: dequant source → permute → write to new tensor
            for (auto & [name, w2d] : to_fill) {
                ggml_tensor * src = ctx->wl.tensors[name];
                int64_t OC = src->ne[0], IC = src->ne[1], KH = src->ne[2], KW = src->ne[3];
                int64_t n = ggml_nelements(src);

                // Dequant source to F32
                std::vector<float> f32_src(n);
                if (src->type == GGML_TYPE_F32) {
                    ggml_backend_tensor_get(src, f32_src.data(), 0, n * sizeof(float));
                } else if (src->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> tmp(n);
                    ggml_backend_tensor_get(src, tmp.data(), 0, n * sizeof(ggml_fp16_t));
                    for (int64_t i = 0; i < n; i++) f32_src[i] = ggml_fp16_to_fp32(tmp[i]);
                } else {
                    size_t raw_sz = ggml_nbytes(src);
                    std::vector<uint8_t> raw(raw_sz);
                    ggml_backend_tensor_get(src, raw.data(), 0, raw_sz);
                    auto * traits = ggml_get_type_traits(src->type);
                    if (traits && traits->to_float) traits->to_float(raw.data(), f32_src.data(), n);
                }

                // Permute: src[oc][ic][kh][kw] → dst[ic*kh*kw][oc]
                // src memory: for kw, for kh, for ic, for oc → src[oc*IC*KH*KW + ic*KH*KW + kh*KW + kw]
                // dst memory: for oc, for (ic*kh*kw) → dst[oc * IC*KH*KW + ic*KH*KW + kh*KW + kw]
                // Wait — ggml ne[0] is innermost. src ne[0]=OC means OC is fastest.
                // src layout: src[kw * OC*IC*KH + kh * OC*IC + ic * OC + oc]
                // Actually in ggml: element at (i0,i1,i2,i3) = data[i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0]
                // For src: nb0=sizeof(type), nb1=ne[0]*nb0, nb2=ne[0]*ne[1]*nb0, nb3=ne[0]*ne[1]*ne[2]*nb0
                // So src[oc, ic, kh, kw] = data[kw*OC*IC*KH + kh*OC*IC + ic*OC + oc]
                // (oc=i0 fastest, kw=i3 slowest)
                //
                // For dst 2D [IC*KH*KW, OC]: element at (ikw, oc) = data[oc * IC*KH*KW + ikw]
                // where ikw = ic*KH*KW + kh*KW + kw

                std::vector<float> f32_dst(n);
                for (int64_t oc = 0; oc < OC; oc++)
                    for (int64_t ic = 0; ic < IC; ic++)
                        for (int64_t kh = 0; kh < KH; kh++)
                            for (int64_t kw = 0; kw < KW; kw++) {
                                int64_t src_idx = kw*OC*IC*KH + kh*OC*IC + ic*OC + oc;
                                int64_t ikw = ic*KH*KW + kh*KW + kw;
                                int64_t dst_idx = oc * IC*KH*KW + ikw;
                                f32_dst[dst_idx] = f32_src[src_idx];
                            }

                ggml_backend_tensor_set(w2d, f32_dst.data(), 0, n * sizeof(float));
                ctx->prep_weights[name] = w2d;
            }
            fprintf(stderr, "restormer: pre-permuted %d conv weights (%.1f MB)\n",
                    n4d, total_bytes / (1024.0 * 1024.0));
        }
    }

    return ctx;
}

void restormer_free(restormer_context * ctx) {
    if (ctx) {
        if (ctx->prep_buf) ggml_backend_buffer_free(ctx->prep_buf);
        if (ctx->prep_ctx) ggml_free(ctx->prep_ctx);
        if (ctx->enc_sched) ggml_backend_sched_free(ctx->enc_sched);
        if (ctx->enc_backend) ggml_backend_free(ctx->enc_backend);
        core_gguf::free_weights(ctx->wl);
        delete ctx;
    }
}

// ── Deferred implementations of full-graph block ──────────────────────

static ggml_cgraph * rst_build_block_graph(restormer_context * ctx,
                                            const rst_block_raw & bw,
                                            int C, int H, int W) {
    int HW = H * W, C3 = C * 3;
    int hidden2 = bw.hidden2, hidden = hidden2 / 2;

    int graph_size = 1024;
    size_t buf_size = ggml_tensor_overhead() * (graph_size + 500)
                    + ggml_graph_overhead_custom(graph_size, false);
    ctx->enc_meta_block.resize(buf_size);
    ggml_init_params ip = { buf_size, ctx->enc_meta_block.data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(g, graph_size, false);

    ggml_tensor * x = ggml_new_tensor_3d(g, GGML_TYPE_F32, W, H, C);
    ggml_set_name(x, "x"); ggml_set_input(x);

    // ---- LN1 + MDTA + residual ----
    ggml_tensor * normed = rst_ln2d_ggml(g, x, bw.norm1_w, bw.norm1_b);

    // QKV: Conv1×1 + DWConv3×3
    ggml_tensor * qkv_w = rst_prep_w(g, bw.qkv_w, C, 1, 1);
    ggml_tensor * qkv = ggml_conv_2d(g, qkv_w, normed, 1, 1, 0, 0, 1, 1);
    ggml_tensor * dw_w = rst_prep_w(g, bw.qkv_dw_w, 1, 3, 3);
    qkv = ggml_conv_2d_dw(g, dw_w, qkv, 1, 1, 1, 1, 1, 1);

    // Split Q, K, V and normalize
    size_t plane = W * H * sizeof(float);
    ggml_tensor * Q = ggml_cont(g, ggml_view_3d(g, qkv, W, H, C, qkv->nb[1], qkv->nb[2], 0));
    ggml_tensor * K = ggml_cont(g, ggml_view_3d(g, qkv, W, H, C, qkv->nb[1], qkv->nb[2], C * plane));
    ggml_tensor * V = ggml_cont(g, ggml_view_3d(g, qkv, W, H, C, qkv->nb[1], qkv->nb[2], 2 * C * plane));

    // L2 normalize Q, K over spatial (approximated as LayerNorm over HW)
    ggml_tensor * Q_flat = ggml_norm(g, ggml_reshape_2d(g, Q, HW, C), 1e-12f);
    ggml_tensor * K_flat = ggml_norm(g, ggml_reshape_2d(g, K, HW, C), 1e-12f);
    ggml_tensor * V_flat = ggml_reshape_2d(g, V, HW, C);

    // Transposed attention: attn = Q^T @ K → [C, C], softmax, @ V^T → [C, HW]
    ggml_tensor * attn = ggml_mul_mat(g, K_flat, Q_flat); // K=HW, M=C, N=C → [C, C]
    attn = ggml_soft_max(g, attn);
    // out = attn @ V^T: ggml_mul_mat(attn[C,C], V_T[C,HW]) → [C, HW]
    ggml_tensor * V_T = ggml_cont(g, ggml_transpose(g, V_flat)); // [C, HW]
    ggml_tensor * out_flat = ggml_mul_mat(g, attn, V_T); // K=C → [C, HW]
    // Transpose to [HW, C] and reshape to [W, H, C]
    out_flat = ggml_cont(g, ggml_transpose(g, out_flat)); // [HW, C]
    ggml_tensor * attn_out = ggml_reshape_3d(g, out_flat, W, H, C);

    // Output projection
    ggml_tensor * proj_w = rst_prep_w(g, bw.proj_w, C, 1, 1);
    attn_out = ggml_conv_2d(g, proj_w, attn_out, 1, 1, 0, 0, 1, 1);

    ggml_tensor * res1 = ggml_add(g, x, attn_out);

    // ---- LN2 + GDFN + residual ----
    normed = rst_ln2d_ggml(g, res1, bw.norm2_w, bw.norm2_b);

    ggml_tensor * ffn_in_w = rst_prep_w(g, bw.ffn_in_w, C, 1, 1);
    ggml_tensor * up = ggml_conv_2d(g, ffn_in_w, normed, 1, 1, 0, 0, 1, 1);
    ggml_tensor * ffn_dw_w = rst_prep_w(g, bw.ffn_dw_w, 1, 3, 3);
    up = ggml_conv_2d_dw(g, ffn_dw_w, up, 1, 1, 1, 1, 1, 1);

    // GELU gate: split channels, GELU(first) * second
    ggml_tensor * up1 = ggml_cont(g, ggml_view_3d(g, up, W, H, hidden, up->nb[1], up->nb[2], 0));
    ggml_tensor * up2 = ggml_cont(g, ggml_view_3d(g, up, W, H, hidden, up->nb[1], up->nb[2], hidden * plane));
    ggml_tensor * gated = ggml_mul(g, ggml_gelu(g, up1), up2);

    ggml_tensor * ffn_out_w = rst_prep_w(g, bw.ffn_out_w, hidden, 1, 1);
    ggml_tensor * ffn_out = ggml_conv_2d(g, ffn_out_w, gated, 1, 1, 0, 0, 1, 1);

    ggml_tensor * result = ggml_add(g, res1, ffn_out);
    ggml_set_name(result, "out"); ggml_set_output(result);
    ggml_build_forward_expand(gf, result);
    return gf;
}

static void rst_transformer_block_ggml(restormer_context * ctx,
                                        float * x, int C, int H, int W,
                                        const rst_block_raw & bw) {
    ggml_cgraph * gf = rst_build_block_graph(ctx, bw, C, H, W);
    ggml_backend_sched_reset(ctx->enc_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->enc_sched, gf)) {
        fprintf(stderr, "restormer: block graph alloc failed\n"); return;
    }
    int n = C * H * W;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), x, 0, n * sizeof(float));
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
    if (ggml_backend_sched_graph_compute(ctx->enc_sched, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "restormer: block graph compute failed\n"); return;
    }
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out"), x, 0, n * sizeof(float));
}

// ── ggml conv2d (per-call, for U-Net level convs) ────────────────────

static void rst_conv2d_ggml(restormer_context * ctx,
                             const float * input, int ic, int h, int w,
                             ggml_tensor * wt, ggml_tensor * bt,
                             int oc, int kh, int kw, int pad, int groups,
                             float * output) {
    if (!ctx->enc_sched || !wt) {
        // scalar fallback via dequant (per-call ggml overhead too high for U-Net)
        rst_conv2d(input, ic, h, w,
                   ctx->dcache.get(wt), bt ? ctx->dcache.get(bt) : nullptr,
                   oc, kh, kw, pad, groups, output);
        return;
    }
    int max_nodes = 32;
    size_t buf_size = ggml_tensor_overhead() * max_nodes
                    + ggml_graph_overhead_custom(max_nodes, false);
    std::vector<uint8_t> meta(buf_size);
    ggml_init_params ip = { buf_size, meta.data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(g, max_nodes, false);

    ggml_tensor * x = ggml_new_tensor_3d(g, GGML_TYPE_F32, w, h, ic);
    ggml_set_name(x, "x"); ggml_set_input(x);

    ggml_tensor * ww = wt;
    if (ww->type != GGML_TYPE_F32)
        ww = ggml_cast(g, ww, GGML_TYPE_F32);
    int ic_g = (groups > 1 && groups == ic) ? 1 : ic;
    if (ggml_n_dims(ww) == 2) {
        int64_t ik = (int64_t)ic_g * kh * kw;
        if (ww->ne[0] == ik) {
            ww = ggml_reshape_4d(g, ww, kw, kh, ic_g, ww->ne[1]);
        } else if (ww->ne[1] == ik) {
            ww = ggml_cont(g, ggml_transpose(g, ww));
            ww = ggml_reshape_4d(g, ww, kw, kh, ic_g, ww->ne[1]);
        } else {
            ww = ggml_reshape_4d(g, ww, kw, kh, ic_g, oc);
        }
    } else if (ggml_n_dims(ww) >= 4) {
        ww = ggml_cont(g, ggml_permute(g, ww, 3, 2, 1, 0));
    }
    ww = ggml_cast(g, ww, GGML_TYPE_F16);
    if (groups > 1 && groups == ic) {
        x = ggml_conv_2d_dw(g, ww, x, 1, 1, pad, pad, 1, 1);
    } else {
        x = ggml_conv_2d(g, ww, x, 1, 1, pad, pad, 1, 1);
    }
    if (bt) {
        ggml_tensor * b = ggml_reshape_3d(g, bt, 1, 1, oc);
        x = ggml_add(g, x, b);
    }
    ggml_set_name(x, "out"); ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    ggml_backend_sched_reset(ctx->enc_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->enc_sched, gf)) return;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), input, 0, ic*h*w*sizeof(float));
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
    int oh = h + 2*pad - kh + 1, ow = w + 2*pad - kw + 1;
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out"), output, 0, oc*oh*ow*sizeof(float));
}

// ── Forward pass (single tile) ─────────────────────────────────────────

static void rst_run_blocks(restormer_context * ctx, float * x, int C, int H, int W,
                           const std::string & prefix, int n_blocks, int n_heads,
                           std::vector<float> & scratch) {
    // Read hidden*2 from the actual weight shape (avoids ffn_factor rounding issues)
    char ffn_key[128];
    snprintf(ffn_key, sizeof(ffn_key), "%s.0.ffn.in.weight", prefix.c_str());
    int hidden2 = (int)ctx->dim0(ffn_key);
    if (hidden2 <= 0) hidden2 = (int)(C * ctx->ffn_factor) * 2;  // fallback
    for (int b = 0; b < n_blocks; b++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s.%d", prefix.c_str(), b);
        std::string p(buf);

        if (ctx->enc_sched && !std::getenv("RESTORMER_SCALAR")) {
            rst_block_raw bw = {};
            bw.norm1_w  = ctx->get_raw(p + ".norm1.weight");
            bw.norm1_b  = ctx->has_bias ? ctx->get_raw(p + ".norm1.bias") : nullptr;
            bw.qkv_w    = ctx->get_raw(p + ".attn.qkv.weight");
            bw.qkv_dw_w = ctx->get_raw(p + ".attn.qkv_dw.weight");
            bw.proj_w   = ctx->get_raw(p + ".attn.proj.weight");
            bw.temperature = ctx->get_raw(p + ".attn.temperature");
            bw.norm2_w  = ctx->get_raw(p + ".norm2.weight");
            bw.norm2_b  = ctx->has_bias ? ctx->get_raw(p + ".norm2.bias") : nullptr;
            bw.ffn_in_w = ctx->get_raw(p + ".ffn.in.weight");
            bw.ffn_dw_w = ctx->get_raw(p + ".ffn.dw.weight");
            bw.ffn_out_w = ctx->get_raw(p + ".ffn.out.weight");
            bw.n_heads  = n_heads;
            bw.hidden2  = hidden2;
            rst_transformer_block_ggml(ctx, x, C, H, W, bw);
        } else {
            rst_block_weights wt = {};
            wt.norm1_w      = ctx->get(p + ".norm1.weight");
            wt.norm1_b      = ctx->has_bias ? ctx->get(p + ".norm1.bias") : nullptr;
            wt.attn_qkv_w   = ctx->get(p + ".attn.qkv.weight");
            wt.attn_qkv_dw_w = ctx->get(p + ".attn.qkv_dw.weight");
            wt.attn_proj_w  = ctx->get(p + ".attn.proj.weight");
            wt.attn_temp    = ctx->get(p + ".attn.temperature");
            wt.norm2_w      = ctx->get(p + ".norm2.weight");
            wt.norm2_b      = ctx->has_bias ? ctx->get(p + ".norm2.bias") : nullptr;
            wt.ffn_in_w     = ctx->get(p + ".ffn.in.weight");
            wt.ffn_dw_w     = ctx->get(p + ".ffn.dw.weight");
            wt.ffn_out_w    = ctx->get(p + ".ffn.out.weight");
            wt.n_heads      = n_heads;
            wt.hidden2      = hidden2;
            rst_transformer_block(x, C, H, W, wt, scratch);
        }
    }
}

// Debug dump callback (set by test harness via env var)
static void rst_debug_dump(const char * name, const float * data, int n) {
    if (!std::getenv("CRISPEMBED_RESTORMER_DEBUG")) return;
    float mean = 0;
    for (int i = 0; i < n; i++) mean += data[i];
    mean /= n;
    fprintf(stderr, "  [DBG] %-20s  n=%d  first8=[%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f]  mean=%.6f\n",
            name, n,
            n > 0 ? data[0] : 0, n > 1 ? data[1] : 0,
            n > 2 ? data[2] : 0, n > 3 ? data[3] : 0,
            n > 4 ? data[4] : 0, n > 5 ? data[5] : 0,
            n > 6 ? data[6] : 0, n > 7 ? data[7] : 0, mean);
}

static void rst_forward_tile(restormer_context * ctx,
                             const float * input, int W, int H,
                             float * output) {
    int dim = ctx->dim;
    std::vector<float> scratch;

    // Pad to multiple of 8 (3 downsamples by 2)
    int pad_h = ((H + 7) / 8) * 8;
    int pad_w = ((W + 7) / 8) * 8;
    std::vector<float> img(3 * pad_h * pad_w, 0.0f);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < pad_h; y++)
            for (int x = 0; x < pad_w; x++)
                img[c * pad_h * pad_w + y * pad_w + x] =
                    input[c * H * W + std::min(y, H - 1) * W + std::min(x, W - 1)];

    int cH = pad_h, cW = pad_w;

    // patch_embed: Conv3(3→dim)
    int C1 = dim;
    std::vector<float> x(C1 * cH * cW);
    rst_debug_dump("img_input", img.data(), 8);
    const float * pe_w = ctx->get("patch_embed.weight");
    if (!pe_w) fprintf(stderr, "restormer: ERROR: patch_embed.weight not found!\n");
    else {
        // Check first few weight values
        rst_debug_dump("pe_weight", pe_w, 8);
    }
    rst_conv2d_ggml(ctx, img.data(), 3, cH, cW,
               ctx->get_raw("patch_embed.weight"),
               ctx->has_bias ? ctx->get_raw("patch_embed.bias") : nullptr,
               C1, 3, 3, 1, 1, x.data());
    rst_debug_dump("patch_embed", x.data(), C1 * cH * cW);
    // Manual check: output[0,0,0] should be sum of 3x3 weighted patches
    if (std::getenv("CRISPEMBED_RESTORMER_DEBUG")) {
        float manual = 0;
        for (int ic = 0; ic < 3; ic++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    int iy = 0 + ky - 1, ix = 0 + kx - 1;
                    if (iy < 0 || iy >= cH || ix < 0 || ix >= cW) continue;
                    manual += img[ic * cH * cW + iy * cW + ix]
                            * pe_w[0 * 3 * 3 * 3 + ic * 3 * 3 + ky * 3 + kx];
                }
        fprintf(stderr, "  [DBG] manual_pe[0,0,0] = %.6f (conv output[0] = %.6f)\n",
                manual, x[0]);
        // Also check oc=1
        float manual1 = 0;
        for (int ic = 0; ic < 3; ic++)
            for (int ky = 0; ky < 3; ky++)
                for (int kx = 0; kx < 3; kx++) {
                    int iy = 0 + ky - 1, ix = 0 + kx - 1;
                    if (iy < 0 || iy >= cH || ix < 0 || ix >= cW) continue;
                    manual1 += img[ic * cH * cW + iy * cW + ix]
                             * pe_w[1 * 3 * 3 * 3 + ic * 3 * 3 + ky * 3 + kx];
                }
        fprintf(stderr, "  [DBG] manual_pe[1,0,0] = %.6f (conv x[1*HW] = %.6f)\n",
                manual1, x[1 * cH * cW]);
        // Print pe_w[27..30] (first 3 values of oc=1)
        fprintf(stderr, "  [DBG] pe_w[27..29] = %.6f, %.6f, %.6f\n", pe_w[27], pe_w[28], pe_w[29]);
    }

    // Encoder level 1
    rst_run_blocks(ctx, x.data(), C1, cH, cW, "enc.0", ctx->num_blocks[0], ctx->heads[0], scratch);
    rst_debug_dump("enc1", x.data(), C1 * cH * cW);
    std::vector<float> enc1 = x;

    // Down 1→2: Conv3(dim→dim/2) → PixelUnshuffle(2) → dim*2 channels
    int C2 = dim * 2;
    {
        int half = dim / 2;
        std::vector<float> down_conv(half * cH * cW);
        rst_conv2d_ggml(ctx, x.data(), C1, cH, cW, ctx->get_raw("down.0.weight"), nullptr, half, 3, 3, 1, 1, down_conv.data());
        x.resize(C2 * (cH / 2) * (cW / 2));
        rst_pixel_unshuffle(down_conv.data(), half, cH, cW, 2, x.data());
        cH /= 2; cW /= 2;
    }

    // Encoder level 2
    rst_run_blocks(ctx, x.data(), C2, cH, cW, "enc.1", ctx->num_blocks[1], ctx->heads[1], scratch);
    std::vector<float> enc2 = x;

    // Down 2→3
    int C3 = dim * 4;
    {
        int half = dim;
        std::vector<float> down_conv(half * cH * cW);
        rst_conv2d_ggml(ctx, x.data(), C2, cH, cW, ctx->get_raw("down.1.weight"), nullptr, half, 3, 3, 1, 1, down_conv.data());
        x.resize(C3 * (cH / 2) * (cW / 2));
        rst_pixel_unshuffle(down_conv.data(), half, cH, cW, 2, x.data());
        cH /= 2; cW /= 2;
    }

    // Encoder level 3
    rst_run_blocks(ctx, x.data(), C3, cH, cW, "enc.2", ctx->num_blocks[2], ctx->heads[2], scratch);
    std::vector<float> enc3 = x;

    // Down 3→4
    int C4 = dim * 8;
    {
        int half = dim * 2;
        std::vector<float> down_conv(half * cH * cW);
        rst_conv2d_ggml(ctx, x.data(), C3, cH, cW, ctx->get_raw("down.2.weight"), nullptr, half, 3, 3, 1, 1, down_conv.data());
        x.resize(C4 * (cH / 2) * (cW / 2));
        rst_pixel_unshuffle(down_conv.data(), half, cH, cW, 2, x.data());
        cH /= 2; cW /= 2;
    }

    // Latent
    rst_run_blocks(ctx, x.data(), C4, cH, cW, "latent", ctx->num_blocks[3], ctx->heads[3], scratch);

    // Up 4→3: Conv3(C4→C4*2) → PixelShuffle(2) → C3
    {
        int up_oc = C4 * 2;
        std::vector<float> up_conv(up_oc * cH * cW);
        rst_conv2d_ggml(ctx, x.data(), C4, cH, cW, ctx->get_raw("up.0.weight"), nullptr, up_oc, 3, 3, 1, 1, up_conv.data());
        cH *= 2; cW *= 2;
        x.resize(C3 * cH * cW);
        rst_pixel_shuffle(up_conv.data(), up_oc, cH / 2, cW / 2, 2, x.data());
    }

    // Cat + reduce + decoder 3
    {
        int cat_c = C3 * 2;  // C3 + C3
        std::vector<float> cat(cat_c * cH * cW);
        memcpy(cat.data(), x.data(), C3 * cH * cW * sizeof(float));
        memcpy(cat.data() + C3 * cH * cW, enc3.data(), C3 * cH * cW * sizeof(float));
        rst_conv2d_ggml(ctx, cat.data(), cat_c, cH, cW,
                   ctx->get_raw("reduce.2.weight"),
                   ctx->has_bias ? ctx->get_raw("reduce.2.bias") : nullptr,
                   C3, 1, 1, 0, 1, x.data());
    }
    rst_run_blocks(ctx, x.data(), C3, cH, cW, "dec.2", ctx->num_blocks[2], ctx->heads[2], scratch);

    // Up 3→2
    {
        int up_oc = C3 * 2;
        std::vector<float> up_conv(up_oc * cH * cW);
        rst_conv2d_ggml(ctx, x.data(), C3, cH, cW, ctx->get_raw("up.1.weight"), nullptr, up_oc, 3, 3, 1, 1, up_conv.data());
        cH *= 2; cW *= 2;
        x.resize(C2 * cH * cW);
        rst_pixel_shuffle(up_conv.data(), up_oc, cH / 2, cW / 2, 2, x.data());
    }

    // Cat + reduce + decoder 2
    {
        int cat_c = C2 * 2;
        std::vector<float> cat(cat_c * cH * cW);
        memcpy(cat.data(), x.data(), C2 * cH * cW * sizeof(float));
        memcpy(cat.data() + C2 * cH * cW, enc2.data(), C2 * cH * cW * sizeof(float));
        rst_conv2d_ggml(ctx, cat.data(), cat_c, cH, cW,
                   ctx->get_raw("reduce.1.weight"),
                   ctx->has_bias ? ctx->get_raw("reduce.1.bias") : nullptr,
                   C2, 1, 1, 0, 1, x.data());
    }
    rst_run_blocks(ctx, x.data(), C2, cH, cW, "dec.1", ctx->num_blocks[1], ctx->heads[1], scratch);

    // Up 2→1 (NO reduce — decoder uses dim*2 = 96)
    {
        int up_oc = C2 * 2;
        std::vector<float> up_conv(up_oc * cH * cW);
        rst_conv2d_ggml(ctx, x.data(), C2, cH, cW, ctx->get_raw("up.2.weight"), nullptr, up_oc, 3, 3, 1, 1, up_conv.data());
        cH *= 2; cW *= 2;
        x.resize(C1 * cH * cW);
        rst_pixel_shuffle(up_conv.data(), up_oc, cH / 2, cW / 2, 2, x.data());
    }

    // Cat (enc1) — no reduce for level 1, decoder uses dim*2
    {
        int dec1_c = C1 * 2;  // 96
        std::vector<float> cat(dec1_c * cH * cW);
        memcpy(cat.data(), x.data(), C1 * cH * cW * sizeof(float));
        memcpy(cat.data() + C1 * cH * cW, enc1.data(), C1 * cH * cW * sizeof(float));
        x = std::move(cat);
    }
    int dec1_c = C1 * 2;  // 96
    rst_run_blocks(ctx, x.data(), dec1_c, cH, cW, "dec.0", ctx->num_blocks[0], ctx->heads[0], scratch);

    // Refinement
    rst_run_blocks(ctx, x.data(), dec1_c, cH, cW, "refine", ctx->n_refine, ctx->heads[0], scratch);

    // Output: Conv3(96→3) + input residual
    std::vector<float> out(3 * cH * cW);
    rst_conv2d_ggml(ctx, x.data(), dec1_c, cH, cW,
               ctx->get_raw("output.weight"),
               ctx->has_bias ? ctx->get_raw("output.bias") : nullptr,
               3, 3, 3, 1, 1, out.data());
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < H; y++)
            for (int xi = 0; xi < W; xi++)
                output[c * H * W + y * W + xi] = out[c * cH * cW + y * cW + xi] + input[c * H * W + y * W + xi];
}

// ── Tiled processing ──────────────────────────────────────────────────

int restormer_process(restormer_context * ctx,
                      const uint8_t * input, int width, int height,
                      int tile_size, int tile_overlap,
                      uint8_t ** output) {
    if (!ctx || !input || !output || width <= 0 || height <= 0) return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    if (tile_size <= 0) tile_size = 128;
    if (tile_overlap <= 0) tile_overlap = 16;
    tile_overlap = std::min(tile_overlap, tile_size / 4);

    std::vector<float> full(3 * height * width);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                full[c * height * width + y * width + x] =
                    input[(y * width + x) * 3 + c] / 255.0f;

    std::vector<float> accum(3 * height * width, 0.0f);
    std::vector<float> wmap(height * width, 0.0f);

    int step = tile_size - tile_overlap;
    int ntx = std::max(1, (width + step - 1) / step);
    int nty = std::max(1, (height + step - 1) / step);

    fprintf(stderr, "restormer: %dx%d, tiles=%dx%d (size=%d, overlap=%d)\n",
            width, height, ntx, nty, tile_size, tile_overlap);

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

            std::vector<float> tile_out(3 * th * tw);
            auto t_tile = std::chrono::steady_clock::now();
            rst_forward_tile(ctx, tile_in.data(), tw, th, tile_out.data());
            if (bench) {
                auto t_tile_end = std::chrono::steady_clock::now();
                fprintf(stderr, "[restormer-bench] tile %d,%d: %.1f ms\n",
                        ty, tx, ms_f(t_tile_end - t_tile).count());
            }

            // Blend with Hann ramp
            int ot = tile_overlap;
            for (int y = 0; y < th; y++) {
                float wy = 1.0f;
                if (y0 > 0 && y < ot) wy = 0.5f - 0.5f * cosf((float)M_PI * y / ot);
                if (y0 + th < height && y >= th - ot) wy = 0.5f - 0.5f * cosf((float)M_PI * (th - 1 - y) / ot);
                for (int x = 0; x < tw; x++) {
                    float wx = 1.0f;
                    if (x0 > 0 && x < ot) wx = 0.5f - 0.5f * cosf((float)M_PI * x / ot);
                    if (x0 + tw < width && x >= tw - ot) wx = 0.5f - 0.5f * cosf((float)M_PI * (tw - 1 - x) / ot);
                    float w = wy * wx;
                    int dy = y0 + y, dx = x0 + x;
                    for (int c = 0; c < 3; c++)
                        accum[c * height * width + dy * width + dx] += tile_out[c * th * tw + y * tw + x] * w;
                    wmap[dy * width + dx] += w;
                }
            }
        }
    }

    uint8_t * out_buf = (uint8_t *)malloc(3 * height * width);
    if (!out_buf) return -1;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++) {
            float w = wmap[y * width + x];
            if (w <= 0) w = 1.0f;
            for (int c = 0; c < 3; c++) {
                float v = accum[c * height * width + y * width + x] / w * 255.0f;
                out_buf[(y * width + x) * 3 + c] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
        }

    *output = out_buf;
    fprintf(stderr, "restormer: done %dx%d\n", width, height);
    if (bench) {
        auto t_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[restormer-bench] total: %.1f ms\n",
                ms_f(t_end - t_total).count());
    }
    return 0;
}

void restormer_free_image(uint8_t * pixels) { free(pixels); }
