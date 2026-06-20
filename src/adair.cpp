// adair.cpp — AdaIR all-in-one image restoration (CPU-scalar).
//
// Architecture: Restormer U-Net (MDTA + GDFN transformer blocks) with
// Adaptive Frequency Learning Blocks (AFLB) at each decoder skip.
//
// U-Net: patch_embed(3→48) → enc1(4×TB) → down → enc2(6×TB) → down
//   → enc3(6×TB) → down → latent(8×TB,384) → up+skip+fre3+reduce
//   → dec3(6×TB) → up+skip+fre2+reduce → dec2(6×TB) → up+skip+fre1
//   → dec1(4×TB) → refine(4×TB) → output(96→3) + residual
//
// AFLB (FreModule): FFT → split low/high by learnable mask →
//   cross-attention guidance → FreRefine (SpatialGate + ChannelGate)
//
// TransformerBlock = LN → MDTA (transposed channel attention) → LN → GDFN

#include "adair.h"
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

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ── Helpers ──

// GPU-safe: uses ggml_backend_tensor_get instead of direct tensor->data
static const float * to_f32(const ggml_tensor * t, std::vector<float> & buf) {
    if (!t) return nullptr;
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

static void conv2d(const float * in, int ic, int h, int w,
                   const float * wt, const float * bi,
                   int oc, int kh, int kw, int pad, int stride, int groups,
                   float * out) {
    int oh = (h + 2 * pad - kh) / stride + 1;
    int ow = (w + 2 * pad - kw) / stride + 1;
    int ic_pg = ic / groups, oc_pg = oc / groups;
    for (int g = 0; g < groups; g++)
        for (int o = 0; o < oc_pg; o++) {
            int oa = g * oc_pg + o;
            float b = bi ? bi[oa] : 0.0f;
            for (int oy = 0; oy < oh; oy++)
                for (int ox = 0; ox < ow; ox++) {
                    float sum = b;
                    for (int c = 0; c < ic_pg; c++) {
                        int ia = g * ic_pg + c;
                        for (int ky = 0; ky < kh; ky++)
                            for (int kx = 0; kx < kw; kx++) {
                                int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                                if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                    sum += in[ia * h * w + iy * w + ix]
                                         * wt[oa * ic_pg * kh * kw + c * kh * kw + ky * kw + kx];
                            }
                    }
                    out[oa * oh * ow + oy * ow + ox] = sum;
                }
        }
}

static void layernorm2d(const float * in, int c, int h, int w,
                        const float * wt, const float * bi, float * out) {
    int hw = h * w;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float mean = 0;
            for (int ch = 0; ch < c; ch++) mean += in[ch * hw + y * w + x];
            mean /= c;
            float var = 0;
            for (int ch = 0; ch < c; ch++) {
                float d = in[ch * hw + y * w + x] - mean; var += d * d;
            }
            var /= c;
            float inv = 1.0f / sqrtf(var + 1e-5f);
            for (int ch = 0; ch < c; ch++)
                out[ch * hw + y * w + x] =
                    (in[ch * hw + y * w + x] - mean) * inv * wt[ch] + bi[ch];
        }
}

// ── MDTA: Multi-DConv Head Transposed Attention ──

struct mdta_wt {
    ggml_tensor * qkv_w, * qkv_dw; // fused QKV: Conv1x1(C→3C) + DWConv3x3(3C)
    ggml_tensor * proj_w;
    ggml_tensor * temp; // [n_heads, 1, 1]
};

static void mdta_forward(const float * x, int C, int H, int W,
                          const mdta_wt & wt, int n_heads,
                          float * out,
                          std::vector<float> & dq1, std::vector<float> & dq2) {
    int hw = H * W, hd = C / n_heads;

    // Fused QKV: Conv1x1(C→3C) + DWConv3x3(3C), then split
    std::vector<float> qkv(3 * C * hw), qkv_dw(3 * C * hw);
    conv2d(x, C, H, W, to_f32(wt.qkv_w, dq1), nullptr, 3 * C, 1, 1, 0, 1, 1, qkv.data());
    conv2d(qkv.data(), 3 * C, H, W, to_f32(wt.qkv_dw, dq1), nullptr, 3 * C, 3, 3, 1, 1, 3 * C, qkv_dw.data());

    std::vector<float> q(C * hw), q_tmp(C * hw), k(C * hw), v(C * hw);
    memcpy(q.data(), qkv_dw.data(), C * hw * sizeof(float));
    memcpy(k.data(), qkv_dw.data() + C * hw, C * hw * sizeof(float));
    memcpy(v.data(), qkv_dw.data() + 2 * C * hw, C * hw * sizeof(float));

    // L2-normalize Q and K (per-channel)
    auto l2norm = [&](float * t) {
        for (int ch = 0; ch < C; ch++) {
            float norm = 0;
            for (int i = 0; i < hw; i++) norm += t[ch * hw + i] * t[ch * hw + i];
            norm = 1.0f / (sqrtf(norm) + 1e-6f);
            for (int i = 0; i < hw; i++) t[ch * hw + i] *= norm;
        }
    };
    l2norm(q.data());
    l2norm(k.data());

    const float * temp = to_f32(wt.temp, dq1);

    // Transposed attention: [B, h, C/h, HW] @ [B, h, HW, C/h] → [h, C/h, C/h]
    // Then [h, C/h, C/h] @ [h, C/h, HW] → [h, C/h, HW]
    std::vector<float> attn_out(C * hw, 0.0f);
    for (int head = 0; head < n_heads; head++) {
        float t_val = temp[head];
        int co = head * hd;
        // attn = softmax(Q^T @ K * temperature)
        // Q^T: [hd, HW]^T → [HW, hd]. K: [hd, HW].
        // Q^T @ K = [hd, hd]
        std::vector<float> attn(hd * hd);
        for (int i = 0; i < hd; i++)
            for (int j = 0; j < hd; j++) {
                float dot = 0;
                for (int p = 0; p < hw; p++)
                    dot += q[(co + i) * hw + p] * k[(co + j) * hw + p];
                attn[i * hd + j] = dot * t_val;
            }
        // Softmax per row
        for (int i = 0; i < hd; i++) {
            float mx = -1e30f;
            for (int j = 0; j < hd; j++) mx = std::max(mx, attn[i * hd + j]);
            float sum = 0;
            for (int j = 0; j < hd; j++) {
                attn[i * hd + j] = expf(attn[i * hd + j] - mx);
                sum += attn[i * hd + j];
            }
            for (int j = 0; j < hd; j++) attn[i * hd + j] /= sum;
        }
        // out = attn @ V → [hd, hd] @ [hd, HW] → [hd, HW]
        for (int i = 0; i < hd; i++)
            for (int p = 0; p < hw; p++) {
                float sum = 0;
                for (int j = 0; j < hd; j++)
                    sum += attn[i * hd + j] * v[(co + j) * hw + p];
                attn_out[(co + i) * hw + p] = sum;
            }
    }

    // Project out
    conv2d(attn_out.data(), C, H, W, to_f32(wt.proj_w, dq1), nullptr,
           C, 1, 1, 0, 1, 1, out);
}

// ── GDFN: Gated-DConv Feed-Forward ──

struct gdfn_wt {
    ggml_tensor * proj1_w; // Conv1x1(C→hidden) — no bias
    ggml_tensor * dw_w;    // DWConv3x3(hidden) — no bias
    ggml_tensor * proj2_w; // Conv1x1(hidden/2→C) — no bias
};

static void gdfn_forward(const float * x, int C, int H, int W,
                          const gdfn_wt & wt,
                          float * out,
                          std::vector<float> & dq1, std::vector<float> & dq2) {
    int hw = H * W;
    // Infer hidden dim from proj1 weight: ne[3] = hidden (output channels)
    int hidden = (int)wt.proj1_w->ne[3];

    // Conv1x1 expand (no bias)
    std::vector<float> h1(hidden * hw);
    conv2d(x, C, H, W, to_f32(wt.proj1_w, dq1), nullptr,
           hidden, 1, 1, 0, 1, 1, h1.data());

    // DWConv3x3 (no bias)
    std::vector<float> h2(hidden * hw);
    conv2d(h1.data(), hidden, H, W, to_f32(wt.dw_w, dq1), nullptr,
           hidden, 3, 3, 1, 1, hidden, h2.data());

    // Gated: split in half, GELU(x1) * x2
    int half = hidden / 2;
    std::vector<float> gated(half * hw);
    for (int ch = 0; ch < half; ch++)
        for (int i = 0; i < hw; i++) {
            float g = h2[ch * hw + i];
            g = g * 0.5f * (1.0f + erff(g * 0.7071067811865476f)); // GELU
            gated[ch * hw + i] = g * h2[(ch + half) * hw + i];
        }

    // Conv1x1 project (no bias)
    conv2d(gated.data(), half, H, W, to_f32(wt.proj2_w, dq1), nullptr,
           C, 1, 1, 0, 1, 1, out);
}

// ── TransformerBlock ──

struct tb_wt {
    ggml_tensor * norm1_w, * norm1_b;
    mdta_wt attn;
    ggml_tensor * norm2_w, * norm2_b;
    gdfn_wt ffn;
};

static void tb_forward(float * x, int C, int H, int W, int n_heads,
                        const tb_wt & wt,
                        std::vector<float> & dq1, std::vector<float> & dq2) {
    int hw = H * W;
    std::vector<float> normed(C * hw), out(C * hw);

    // LN → MDTA → residual
    layernorm2d(x, C, H, W, to_f32(wt.norm1_w, dq1), to_f32(wt.norm1_b, dq2), normed.data());
    mdta_forward(normed.data(), C, H, W, wt.attn, n_heads, out.data(), dq1, dq2);
    for (int i = 0; i < C * hw; i++) x[i] += out[i];

    // LN → GDFN → residual
    layernorm2d(x, C, H, W, to_f32(wt.norm2_w, dq1), to_f32(wt.norm2_b, dq2), normed.data());
    gdfn_forward(normed.data(), C, H, W, wt.ffn, out.data(), dq1, dq2);
    for (int i = 0; i < C * hw; i++) x[i] += out[i];
}

// ── FreModule (AFLB): FFT decomposition + cross-attention ──

// 2D FFT via row-wise + column-wise 1D FFT (radix-2)
static int next_pow2(int n) { int p = 1; while (p < n) p <<= 1; return p; }

static void fft1d(float * re, float * im, int n, bool inverse) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { std::swap(re[i], re[j]); std::swap(im[i], im[j]); }
    }
    for (int len = 2; len <= n; len <<= 1) {
        float ang = 2.0f * (float)M_PI / len * (inverse ? -1 : 1);
        float wR = cosf(ang), wI = sinf(ang);
        for (int i = 0; i < n; i += len) {
            float curR = 1, curI = 0;
            for (int j = 0; j < len / 2; j++) {
                float uR = re[i + j], uI = im[i + j];
                float vR = re[i + j + len / 2] * curR - im[i + j + len / 2] * curI;
                float vI = re[i + j + len / 2] * curI + im[i + j + len / 2] * curR;
                re[i + j] = uR + vR; im[i + j] = uI + vI;
                re[i + j + len / 2] = uR - vR; im[i + j + len / 2] = uI - vI;
                float nR = curR * wR - curI * wI;
                curI = curR * wI + curI * wR;
                curR = nR;
            }
        }
    }
    // Note: normalization is applied by the caller based on FFT convention.
    // PyTorch norm='forward' means: FFT divides by N, IFFT does not.
}

static void fft2d(const float * input, int C, int H, int W,
                   float * out_re, float * out_im) {
    int pH = next_pow2(H), pW = next_pow2(W);
    int phw = pH * pW;
    // Row-wise FFT
    std::vector<float> re(C * phw, 0.0f), im(C * phw, 0.0f);
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) re[c * phw + y * pW + x] = input[c * H * W + y * W + x];
            fft1d(&re[c * phw + y * pW], &im[c * phw + y * pW], pW, false);
        }
    // Column-wise FFT
    std::vector<float> col_re(pH), col_im(pH);
    for (int c = 0; c < C; c++)
        for (int x = 0; x < pW; x++) {
            for (int y = 0; y < pH; y++) {
                col_re[y] = re[c * phw + y * pW + x];
                col_im[y] = im[c * phw + y * pW + x];
            }
            fft1d(col_re.data(), col_im.data(), pH, false);
            for (int y = 0; y < pH; y++) {
                out_re[c * phw + y * pW + x] = col_re[y];
                out_im[c * phw + y * pW + x] = col_im[y];
            }
        }
    // norm='forward': divide by N = pH * pW
    float inv_n = 1.0f / (float)(pH * pW);
    for (int i = 0; i < C * phw; i++) {
        out_re[i] *= inv_n;
        out_im[i] *= inv_n;
    }
}

static void ifft2d(float * re, float * im, int C, int pH, int pW) {
    // Column-wise IFFT
    std::vector<float> col_re(pH), col_im(pH);
    for (int c = 0; c < C; c++)
        for (int x = 0; x < pW; x++) {
            for (int y = 0; y < pH; y++) {
                col_re[y] = re[c * pH * pW + y * pW + x];
                col_im[y] = im[c * pH * pW + y * pW + x];
            }
            fft1d(col_re.data(), col_im.data(), pH, true);
            for (int y = 0; y < pH; y++) {
                re[c * pH * pW + y * pW + x] = col_re[y];
                im[c * pH * pW + y * pW + x] = col_im[y];
            }
        }
    // Row-wise IFFT
    for (int c = 0; c < C; c++)
        for (int y = 0; y < pH; y++)
            fft1d(&re[c * pH * pW + y * pW], &im[c * pH * pW + y * pW], pW, true);
}

// Channel cross-attention (MDTA variant for cross-attention between two inputs)
struct cross_attn_wt {
    ggml_tensor * q_w, * q_dw;
    ggml_tensor * kv_w, * kv_dw;
    ggml_tensor * proj_w;
    ggml_tensor * temp;
};

static void cross_attn_forward(const float * q_in, const float * kv_in,
                                int C, int H, int W, int n_heads,
                                const cross_attn_wt & wt,
                                float * out,
                                std::vector<float> & dq1, std::vector<float> & dq2) {
    int hw = H * W, hd = C / n_heads;

    std::vector<float> q(C * hw), q_tmp(C * hw), k(C * hw), v(C * hw);
    // Q from q_in
    conv2d(q_in, C, H, W, to_f32(wt.q_w, dq1), nullptr, C, 1, 1, 0, 1, 1, q_tmp.data());
    conv2d(q_tmp.data(), C, H, W, to_f32(wt.q_dw, dq1), nullptr, C, 3, 3, 1, 1, C, q.data());
    // KV from kv_in
    std::vector<float> kv(2 * C * hw), kv_dw_buf(2 * C * hw);
    conv2d(kv_in, C, H, W, to_f32(wt.kv_w, dq1), nullptr, 2 * C, 1, 1, 0, 1, 1, kv.data());
    conv2d(kv.data(), 2 * C, H, W, to_f32(wt.kv_dw, dq1), nullptr, 2 * C, 3, 3, 1, 1, 2 * C, kv_dw_buf.data());
    memcpy(k.data(), kv_dw_buf.data(), C * hw * sizeof(float));
    memcpy(v.data(), kv_dw_buf.data() + C * hw, C * hw * sizeof(float));

    // L2-normalize Q, K
    auto l2norm = [&](float * t) {
        for (int ch = 0; ch < C; ch++) {
            float n = 0;
            for (int i = 0; i < hw; i++) n += t[ch * hw + i] * t[ch * hw + i];
            n = 1.0f / (sqrtf(n) + 1e-6f);
            for (int i = 0; i < hw; i++) t[ch * hw + i] *= n;
        }
    };
    l2norm(q.data()); l2norm(k.data());

    const float * temp = to_f32(wt.temp, dq1);
    std::vector<float> attn_out(C * hw, 0.0f);
    for (int head = 0; head < n_heads; head++) {
        float t_val = temp[head];
        int co = head * hd;
        std::vector<float> attn(hd * hd);
        for (int i = 0; i < hd; i++)
            for (int j = 0; j < hd; j++) {
                float dot = 0;
                for (int p = 0; p < hw; p++)
                    dot += q[(co + i) * hw + p] * k[(co + j) * hw + p];
                attn[i * hd + j] = dot * t_val;
            }
        for (int i = 0; i < hd; i++) {
            float mx = -1e30f;
            for (int j = 0; j < hd; j++) mx = std::max(mx, attn[i * hd + j]);
            float sum = 0;
            for (int j = 0; j < hd; j++) { attn[i * hd + j] = expf(attn[i * hd + j] - mx); sum += attn[i * hd + j]; }
            for (int j = 0; j < hd; j++) attn[i * hd + j] /= sum;
        }
        for (int i = 0; i < hd; i++)
            for (int p = 0; p < hw; p++) {
                float s = 0;
                for (int j = 0; j < hd; j++) s += attn[i * hd + j] * v[(co + j) * hw + p];
                attn_out[(co + i) * hw + p] = s;
            }
    }
    conv2d(attn_out.data(), C, H, W, to_f32(wt.proj_w, dq1), nullptr,
           C, 1, 1, 0, 1, 1, out);
}

// FreModule weights
struct fre_wt {
    ggml_tensor * para1, * para2;                 // [C, 1, 1]
    ggml_tensor * conv_w, * conv1_w;              // Conv3x3 for reconstructing from FFT
    cross_attn_wt cross_l, cross_h, cross_agg;    // 3 cross-attentions
    // FreRefine
    ggml_tensor * sg_w;                           // SpatialGate: [1, 2, 7, 7]
    ggml_tensor * cg_w1, * cg_w2;                // ChannelGate MLP: [C/16, C] + [C, C/16]
    ggml_tensor * proj_w, * proj_b;               // final projection
    // Rate/score
    ggml_tensor * rate_w1, * rate_w2;             // [C/8, C] + [2, C/8]
    ggml_tensor * score_w, * score_b;             // Conv7x7: [2, 2, 7, 7]
};

// fre_forward: inp_img (3ch) + decoder feature → refined feature
// The FFT operates on inp_img projected to C channels, not on intermediate features.
static void fre_forward(const float * inp_img_3ch, int img_h, int img_w,
                         const float * dec_feat, int C, int H, int W,
                         int n_heads,
                         const fre_wt & wt,
                         float * out, // [C, H, W]
                         std::vector<float> & dq1, std::vector<float> & dq2) {
    int hw = H * W;

    // Save dec_feat in case out aliases it (caller passes x.data() for both)
    std::vector<float> dec_copy(dec_feat, dec_feat + C * hw);
    dec_feat = dec_copy.data();

    // Bilinear interpolate inp_img to (H, W) if sizes differ
    std::vector<float> img_resized(3 * hw);
    if (img_h != H || img_w != W) {
        for (int c = 0; c < 3; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    float sy = ((float)y + 0.5f) * img_h / H - 0.5f;
                    float sx = ((float)x + 0.5f) * img_w / W - 0.5f;
                    sy = std::max(0.0f, std::min(sy, (float)(img_h - 1)));
                    sx = std::max(0.0f, std::min(sx, (float)(img_w - 1)));
                    int y0 = (int)sy, x0 = (int)sx;
                    int y1 = std::min(y0 + 1, img_h - 1), x1 = std::min(x0 + 1, img_w - 1);
                    float fy = sy - y0, fx = sx - x0;
                    img_resized[c * hw + y * W + x] =
                        (1-fy)*((1-fx)*inp_img_3ch[c*img_h*img_w + y0*img_w + x0] +
                                fx*inp_img_3ch[c*img_h*img_w + y0*img_w + x1]) +
                        fy*((1-fx)*inp_img_3ch[c*img_h*img_w + y1*img_w + x0] +
                            fx*inp_img_3ch[c*img_h*img_w + y1*img_w + x1]);
                }
    } else {
        memcpy(img_resized.data(), inp_img_3ch, 3 * hw * sizeof(float));
    }

    // Project img to C channels: conv1(3→C)
    std::vector<float> projected(C * hw);
    conv2d(img_resized.data(), 3, H, W, to_f32(wt.conv1_w, dq1), nullptr,
           C, 3, 3, 1, 1, 1, projected.data());

    // FFT on projected features
    int pH = next_pow2(H), pW = next_pow2(W);
    int phw = pH * pW;
    std::vector<float> fft_re(C * phw), fft_im(C * phw);
    fft2d(projected.data(), C, H, W, fft_re.data(), fft_im.data());

    // Adaptive mask: threshold from avg-pooled features via rate_conv
    // rate_conv: Conv1x1(C→C/8) + GELU + Conv1x1(C/8→2) → sigmoid → [B,2,1,1]
    std::vector<float> pool(C, 0.0f);
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < hw; i++) pool[c] += projected[c * hw + i];
        pool[c] /= hw;
    }
    int rc_hidden = (int)wt.rate_w1->ne[3]; // C/8
    const float * rw1 = to_f32(wt.rate_w1, dq1);
    std::vector<float> rc_h(rc_hidden);
    for (int o = 0; o < rc_hidden; o++) {
        float s = 0;
        for (int i = 0; i < C; i++) s += rw1[o * C + i] * pool[i];
        s = s * 0.5f * (1.0f + erff(s * 0.7071067811865476f)); // GELU
        rc_h[o] = s;
    }
    const float * rw2 = to_f32(wt.rate_w2, dq1);
    float thresh[2];
    for (int o = 0; o < 2; o++) {
        float s = 0;
        for (int i = 0; i < rc_hidden; i++) s += rw2[o * rc_hidden + i] * rc_h[i];
        thresh[o] = 1.0f / (1.0f + expf(-s)); // sigmoid
    }

    // Build spatial mask in FFT domain (center rectangle)
    // After fftshift: center is at (pH/2, pW/2)
    // Mask size: h_ = pH/128 * thresh[0], w_ = pW/128 * thresh[1]
    int h_half = (int)((float)(pH / 128) * thresh[0]);
    int w_half = (int)((float)(pW / 128) * thresh[1]);

    // fftshift the FFT
    std::vector<float> shifted_re(C * phw), shifted_im(C * phw);
    int sh_h = pH / 2, sh_w = pW / 2;
    for (int c = 0; c < C; c++)
        for (int y = 0; y < pH; y++)
            for (int x = 0; x < pW; x++) {
                int sy = (y + sh_h) % pH, sx = (x + sh_w) % pW;
                shifted_re[c * phw + y * pW + x] = fft_re[c * phw + sy * pW + sx];
                shifted_im[c * phw + y * pW + x] = fft_im[c * phw + sy * pW + sx];
            }

    // Mask: 1 inside center rect, 0 outside
    std::vector<float> mask(phw, 0.0f);
    for (int y = sh_h - h_half; y < sh_h + h_half && y < pH; y++)
        for (int x = sh_w - w_half; x < sh_w + w_half && x < pW; x++)
            if (y >= 0 && x >= 0) mask[y * pW + x] = 1.0f;

    // Split: low = mask * fft, high = (1-mask) * fft
    std::vector<float> lo_re(C * phw), lo_im(C * phw), hi_re(C * phw), hi_im(C * phw);
    for (int c = 0; c < C; c++)
        for (int i = 0; i < phw; i++) {
            float m = mask[i];
            lo_re[c * phw + i] = m * shifted_re[c * phw + i];
            lo_im[c * phw + i] = m * shifted_im[c * phw + i];
            hi_re[c * phw + i] = (1 - m) * shifted_re[c * phw + i];
            hi_im[c * phw + i] = (1 - m) * shifted_im[c * phw + i];
        }

    // Unshift + IFFT
    auto unshift = [&](std::vector<float> & re, std::vector<float> & im) {
        std::vector<float> tmp_re(C * phw), tmp_im(C * phw);
        for (int c = 0; c < C; c++)
            for (int y = 0; y < pH; y++)
                for (int x = 0; x < pW; x++) {
                    int sy = (y + pH - sh_h) % pH, sx = (x + pW - sh_w) % pW;
                    tmp_re[c * phw + y * pW + x] = re[c * phw + sy * pW + sx];
                    tmp_im[c * phw + y * pW + x] = im[c * phw + sy * pW + sx];
                }
        re = std::move(tmp_re); im = std::move(tmp_im);
    };
    unshift(lo_re, lo_im); unshift(hi_re, hi_im);
    ifft2d(lo_re.data(), lo_im.data(), C, pH, pW);
    ifft2d(hi_re.data(), hi_im.data(), C, pH, pW);

    // Take abs and crop to H×W
    std::vector<float> lo_feat(C * hw), hi_feat(C * hw);
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                lo_feat[c * hw + y * W + x] = fabsf(lo_re[c * phw + y * pW + x]);
                hi_feat[c * hw + y * W + x] = fabsf(hi_re[c * phw + y * pW + x]);
            }

    // Cross-attention: high guided by dec_feat, low guided by dec_feat
    std::vector<float> ca_h(C * hw), ca_l(C * hw);
    cross_attn_forward(hi_feat.data(), dec_feat, C, H, W, n_heads, wt.cross_l, ca_h.data(), dq1, dq2);
    cross_attn_forward(lo_feat.data(), dec_feat, C, H, W, n_heads, wt.cross_h, ca_l.data(), dq1, dq2);

    // FreRefine: SpatialGate(H→L) + ChannelGate(L→H)
    // SpatialGate: max+mean pool → conv7x7 → sigmoid → scale low by spatial gate
    std::vector<float> sg_in(2 * hw);
    for (int i = 0; i < hw; i++) {
        float mx = -1e30f, mn = 0;
        for (int c = 0; c < C; c++) {
            float v = ca_h[c * hw + i]; mx = std::max(mx, v); mn += v;
        }
        sg_in[i] = mx; sg_in[hw + i] = mn / C;
    }
    std::vector<float> sg_out(hw);
    conv2d(sg_in.data(), 2, H, W, to_f32(wt.sg_w, dq1), nullptr, 1, 7, 7, 3, 1, 1, sg_out.data());
    for (int i = 0; i < hw; i++) sg_out[i] = 1.0f / (1.0f + expf(-sg_out[i]));
    std::vector<float> lo_gated(C * hw);
    for (int c = 0; c < C; c++)
        for (int i = 0; i < hw; i++) lo_gated[c * hw + i] = ca_l[c * hw + i] * sg_out[i];

    // ChannelGate: avg pool → MLP → sigmoid → scale high
    std::vector<float> cg_pool(C, 0.0f);
    for (int c = 0; c < C; c++) {
        for (int i = 0; i < hw; i++) cg_pool[c] += ca_l[c * hw + i];
        cg_pool[c] /= hw;
    }
    int cg_hidden = (int)wt.cg_w1->ne[3];
    const float * cg1 = to_f32(wt.cg_w1, dq1);
    const float * cg2 = to_f32(wt.cg_w2, dq1);
    std::vector<float> cg_h(cg_hidden), cg_out_v(C);
    for (int o = 0; o < cg_hidden; o++) {
        float s = 0;
        for (int i = 0; i < C; i++) s += cg1[o * C + i] * cg_pool[i];
        cg_h[o] = std::max(0.0f, s);
    }
    for (int o = 0; o < C; o++) {
        float s = 0;
        for (int i = 0; i < cg_hidden; i++) s += cg2[o * cg_hidden + i] * cg_h[i];
        cg_out_v[o] = 1.0f / (1.0f + expf(-s));
    }
    std::vector<float> hi_gated(C * hw);
    for (int c = 0; c < C; c++)
        for (int i = 0; i < hw; i++) hi_gated[c * hw + i] = ca_h[c * hw + i] * cg_out_v[c];

    // Sum and project
    std::vector<float> refined(C * hw);
    for (int i = 0; i < C * hw; i++) refined[i] = lo_gated[i] + hi_gated[i];
    conv2d(refined.data(), C, H, W, to_f32(wt.proj_w, dq1), to_f32(wt.proj_b, dq2),
           C, 1, 1, 0, 1, 1, out);

    // Aggregate cross-attention of dec_feat with refined
    std::vector<float> ca_agg(C * hw);
    cross_attn_forward(dec_feat, out, C, H, W, n_heads, wt.cross_agg, ca_agg.data(), dq1, dq2);

    // Final: out = ca_agg * para1 + dec_feat * para2
    const float * p1 = to_f32(wt.para1, dq1);
    const float * p2 = to_f32(wt.para2, dq2);
    for (int c = 0; c < C; c++)
        for (int i = 0; i < hw; i++)
            out[c * hw + i] = ca_agg[c * hw + i] * p1[c] + dec_feat[c * hw + i] * p2[c];
}

// ── Downsample / Upsample ──

static void pixel_unshuffle(const float * in, int C, int H, int W, int r, float * out) {
    int oH = H / r, oW = W / r, oC = C * r * r;
    for (int c = 0; c < C; c++)
        for (int y = 0; y < oH; y++)
            for (int x = 0; x < oW; x++)
                for (int ry = 0; ry < r; ry++)
                    for (int rx = 0; rx < r; rx++) {
                        int oc = c * r * r + ry * r + rx;
                        out[oc * oH * oW + y * oW + x] =
                            in[c * H * W + (y * r + ry) * W + (x * r + rx)];
                    }
}

static void pixel_shuffle(const float * in, int C, int H, int W, int r, float * out) {
    int oC = C / (r * r), oH = H * r, oW = W * r;
    for (int c = 0; c < oC; c++)
        for (int y = 0; y < oH; y++)
            for (int x = 0; x < oW; x++) {
                int ic = c * r * r + (y % r) * r + (x % r);
                out[c * oH * oW + y * oW + x] = in[ic * H * W + (y / r) * W + (x / r)];
            }
}

// ── Model context ──

struct adair_context {
    ggml_backend_t backend = nullptr;
    ggml_context * gguf_ctx;
    ggml_backend_buffer_t gguf_buf;

    bool bench;
    ggml_tensor * patch_embed_w, * patch_embed_b;
    ggml_tensor * output_w, * output_b;

    std::vector<tb_wt> enc1, enc2, enc3;       // 4, 6, 6 blocks
    std::vector<tb_wt> latent;                  // 8 blocks
    std::vector<tb_wt> dec3, dec2, dec1;        // 6, 6, 4 blocks
    std::vector<tb_wt> refinement;              // 4 blocks

    // Downsample: Conv3x3 + PixelUnshuffle
    struct down_wt { ggml_tensor * w, * b; } down12, down23, down34;
    // Upsample: Conv3x3 + PixelShuffle
    struct up_wt { ggml_tensor * w, * b; } up43, up32, up21;
    // Channel reduce
    ggml_tensor * reduce3_w, * reduce3_b;
    ggml_tensor * reduce2_w, * reduce2_b;

    // FreModules
    fre_wt fre1, fre2, fre3;
};

static void load_cross_attn(core_gguf::WeightLoad & wl, const char * pfx, cross_attn_wt & ca) {
    auto g = [&](const char * s) { char b[256]; snprintf(b, sizeof(b), "%s.%s", pfx, s); return core_gguf::try_get(wl.tensors, b); };
    ca.q_w = g("q.weight"); ca.q_dw = g("q_dwconv.weight");
    ca.kv_w = g("kv.weight"); ca.kv_dw = g("kv_dwconv.weight");
    ca.proj_w = g("project_out.weight");
    ca.temp = g("temperature");
}

static void load_tb(core_gguf::WeightLoad & wl, const char * pfx, tb_wt & t) {
    auto g = [&](const char * s) { char b[256]; snprintf(b, sizeof(b), "%s.%s", pfx, s); return core_gguf::try_get(wl.tensors, b); };
    // AdaIR uses WithBias_LayerNorm which wraps nn.LayerNorm in .body
    t.norm1_w = g("norm1.body.weight"); t.norm1_b = g("norm1.body.bias");
    t.attn.qkv_w = g("attn.qkv.weight");
    t.attn.qkv_dw = g("attn.qkv_dwconv.weight");
    t.attn.proj_w = g("attn.project_out.weight");
    t.attn.temp = g("attn.temperature");
    t.norm2_w = g("norm2.body.weight"); t.norm2_b = g("norm2.body.bias");
    t.ffn.proj1_w = g("ffn.project_in.weight");
    t.ffn.dw_w = g("ffn.dwconv.weight");
    t.ffn.proj2_w = g("ffn.project_out.weight");
}

static void load_fre(core_gguf::WeightLoad & wl, const char * pfx, fre_wt & f) {
    auto g = [&](const char * s) { char b[256]; snprintf(b, sizeof(b), "%s.%s", pfx, s); return core_gguf::try_get(wl.tensors, b); };
    f.para1 = g("para1"); f.para2 = g("para2");
    f.conv_w = g("conv.weight"); f.conv1_w = g("conv1.weight");
    char ca[256];
    snprintf(ca, sizeof(ca), "%s.channel_cross_l", pfx); load_cross_attn(wl, ca, f.cross_l);
    snprintf(ca, sizeof(ca), "%s.channel_cross_h", pfx); load_cross_attn(wl, ca, f.cross_h);
    snprintf(ca, sizeof(ca), "%s.channel_cross_agg", pfx); load_cross_attn(wl, ca, f.cross_agg);
    f.sg_w = g("frequency_refine.SpatialGate.spatial.weight");
    f.cg_w1 = g("frequency_refine.ChannelGate.mlp.0.weight");
    f.cg_w2 = g("frequency_refine.ChannelGate.mlp.2.weight");
    f.proj_w = g("frequency_refine.proj.weight");
    f.proj_b = g("frequency_refine.proj.bias");
    f.rate_w1 = g("rate_conv.0.weight"); f.rate_w2 = g("rate_conv.2.weight");
    f.score_w = g("score_gen.weight"); f.score_b = g("score_gen.bias");
}

adair_context * adair_init(const char * model_path, int n_threads) {
    (void)n_threads;
    if (!model_path) return nullptr;

    bool force_cpu = (getenv("ADAIR_FORCE_CPU") && atoi(getenv("ADAIR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    if (!backend) return nullptr;
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, backend, "adair", wl)) {
        ggml_backend_free(backend); return nullptr;
    }

    auto * ctx = new adair_context;
    ctx->backend = backend;
    ctx->gguf_ctx = wl.ctx; ctx->gguf_buf = wl.buf;

    auto g = [&](const char * s) { return core_gguf::try_get(wl.tensors, s); };

    ctx->patch_embed_w = g("net.patch_embed.proj.weight");
    ctx->patch_embed_b = g("net.patch_embed.proj.bias");
    ctx->output_w = g("net.output.weight"); ctx->output_b = g("net.output.bias");

    auto load_blocks = [&](const char * stage, int n, std::vector<tb_wt> & vec) {
        vec.resize(n);
        for (int i = 0; i < n; i++) {
            char pfx[128]; snprintf(pfx, sizeof(pfx), "net.%s.%d", stage, i);
            load_tb(wl, pfx, vec[i]);
        }
    };
    load_blocks("encoder_level1", 4, ctx->enc1);
    load_blocks("encoder_level2", 6, ctx->enc2);
    load_blocks("encoder_level3", 6, ctx->enc3);
    load_blocks("latent", 8, ctx->latent);
    load_blocks("decoder_level3", 6, ctx->dec3);
    load_blocks("decoder_level2", 6, ctx->dec2);
    load_blocks("decoder_level1", 4, ctx->dec1);
    load_blocks("refinement", 4, ctx->refinement);

    auto load_down = [&](const char * name, adair_context::down_wt & d) {
        char w[128], b[128];
        snprintf(w, sizeof(w), "net.%s.body.0.weight", name);
        snprintf(b, sizeof(b), "net.%s.body.0.bias", name);
        d.w = g(w); d.b = g(b);
    };
    load_down("down1_2", ctx->down12);
    load_down("down2_3", ctx->down23);
    load_down("down3_4", ctx->down34);

    auto load_up = [&](const char * name, adair_context::up_wt & u) {
        char w[128], b[128];
        snprintf(w, sizeof(w), "net.%s.body.0.weight", name);
        snprintf(b, sizeof(b), "net.%s.body.0.bias", name);
        u.w = g(w); u.b = g(b);
    };
    load_up("up4_3", ctx->up43);
    load_up("up3_2", ctx->up32);
    load_up("up2_1", ctx->up21);

    ctx->reduce3_w = g("net.reduce_chan_level3.weight");
    ctx->reduce3_b = g("net.reduce_chan_level3.bias");
    ctx->reduce2_w = g("net.reduce_chan_level2.weight");
    ctx->reduce2_b = g("net.reduce_chan_level2.bias");

    load_fre(wl, "net.fre1", ctx->fre1);
    load_fre(wl, "net.fre2", ctx->fre2);
    load_fre(wl, "net.fre3", ctx->fre3);

    ctx->bench = (std::getenv("CRISPEMBED_ADAIR_BENCH") != nullptr);
    return ctx;
}

void adair_free(adair_context * ctx) {
    if (!ctx) return;
    core_gguf::WeightLoad wl;
    wl.ctx = ctx->gguf_ctx; wl.buf = ctx->gguf_buf;
    core_gguf::free_weights(wl);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

int adair_process_float(adair_context * ctx,
                        const float * input_chw, int width, int height,
                        float * output_chw) {
    if (!ctx || !input_chw || !output_chw) return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    int H = height, W = width;
    std::vector<float> dq1, dq2;

    // Patch embed: Conv3x3(3→48)
    int ch = 48;
    std::vector<float> x(ch * H * W);
    conv2d(input_chw, 3, H, W, to_f32(ctx->patch_embed_w, dq1),
           to_f32(ctx->patch_embed_b, dq2), ch, 3, 3, 1, 1, 1, x.data());

    // Encoder level 1: 4 TB at 48ch
    int heads[] = {1, 2, 4, 8};
    int cur_h = H, cur_w = W;
    auto t_enc1 = std::chrono::steady_clock::now();
    for (auto & tb : ctx->enc1) tb_forward(x.data(), ch, cur_h, cur_w, heads[0], tb, dq1, dq2);
    std::vector<float> skip1(x.begin(), x.end());
    if (bench) {
        auto t_enc1_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[adair-bench] enc1: %.1f ms\n",
                ms_f(t_enc1_end - t_enc1).count());
    }

    // Downsample 1→2: Conv3x3 + PixelUnshuffle(2)
    std::vector<float> ds(ch / 2 * cur_h * cur_w);
    conv2d(x.data(), ch, cur_h, cur_w, to_f32(ctx->down12.w, dq1), to_f32(ctx->down12.b, dq2),
           ch / 2, 3, 3, 1, 1, 1, ds.data());
    ch = ch * 2; cur_h /= 2; cur_w /= 2; // after pixel_unshuffle: ch/2 * 4 = ch*2
    x.resize(ch * cur_h * cur_w);
    pixel_unshuffle(ds.data(), ch / 4, cur_h * 2, cur_w * 2, 2, x.data());

    // Encoder level 2: 6 TB at 96ch
    auto t_enc2 = std::chrono::steady_clock::now();
    for (auto & tb : ctx->enc2) tb_forward(x.data(), ch, cur_h, cur_w, heads[1], tb, dq1, dq2);
    std::vector<float> skip2(x.begin(), x.end());
    if (bench) {
        auto t_enc2_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[adair-bench] enc2: %.1f ms\n",
                ms_f(t_enc2_end - t_enc2).count());
    }

    // Downsample 2→3
    ds.resize(ch / 2 * cur_h * cur_w);
    conv2d(x.data(), ch, cur_h, cur_w, to_f32(ctx->down23.w, dq1), to_f32(ctx->down23.b, dq2),
           ch / 2, 3, 3, 1, 1, 1, ds.data());
    ch *= 2; cur_h /= 2; cur_w /= 2;
    x.resize(ch * cur_h * cur_w);
    pixel_unshuffle(ds.data(), ch / 4, cur_h * 2, cur_w * 2, 2, x.data());

    // Encoder level 3: 6 TB at 192ch
    auto t_enc3 = std::chrono::steady_clock::now();
    for (auto & tb : ctx->enc3) tb_forward(x.data(), ch, cur_h, cur_w, heads[2], tb, dq1, dq2);
    std::vector<float> skip3(x.begin(), x.end());
    if (bench) {
        auto t_enc3_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[adair-bench] enc3: %.1f ms\n",
                ms_f(t_enc3_end - t_enc3).count());
    }

    // Downsample 3→4
    ds.resize(ch / 2 * cur_h * cur_w);
    conv2d(x.data(), ch, cur_h, cur_w, to_f32(ctx->down34.w, dq1), to_f32(ctx->down34.b, dq2),
           ch / 2, 3, 3, 1, 1, 1, ds.data());
    ch *= 2; cur_h /= 2; cur_w /= 2;
    x.resize(ch * cur_h * cur_w);
    pixel_unshuffle(ds.data(), ch / 4, cur_h * 2, cur_w * 2, 2, x.data());

    // Latent: 8 TB at 384ch
    auto t_latent = std::chrono::steady_clock::now();
    for (int bi = 0; bi < (int)ctx->latent.size(); bi++) {
        tb_forward(x.data(), ch, cur_h, cur_w, heads[3], ctx->latent[bi], dq1, dq2);
    }
    if (bench) {
        auto t_latent_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[adair-bench] latent: %.1f ms\n",
                ms_f(t_latent_end - t_latent).count());
    }

    // fre1(inp_img, latent): refine latent BEFORE upsample
    auto t_fre1 = std::chrono::steady_clock::now();
    fre_forward(input_chw, H, W, x.data(), ch, cur_h, cur_w, heads[2],
                ctx->fre1, x.data(), dq1, dq2);
    if (bench) {
        auto t_fre1_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[adair-bench] fre1: %.1f ms\n",
                ms_f(t_fre1_end - t_fre1).count());
    }

    // Upsample 4→3 + cat(skip3) + reduce
    int up_ch = ch * 2;
    std::vector<float> us(up_ch * cur_h * cur_w);
    conv2d(x.data(), ch, cur_h, cur_w, to_f32(ctx->up43.w, dq1), to_f32(ctx->up43.b, dq2),
           up_ch, 3, 3, 1, 1, 1, us.data());
    ch /= 2; cur_h *= 2; cur_w *= 2;
    x.resize(ch * cur_h * cur_w);
    pixel_shuffle(us.data(), up_ch, cur_h / 2, cur_w / 2, 2, x.data());
    // Cat with skip3 → 2*ch, reduce to ch
    std::vector<float> cat(2 * ch * cur_h * cur_w);
    memcpy(cat.data(), x.data(), ch * cur_h * cur_w * sizeof(float));
    memcpy(cat.data() + ch * cur_h * cur_w, skip3.data(), ch * cur_h * cur_w * sizeof(float));
    conv2d(cat.data(), 2 * ch, cur_h, cur_w, to_f32(ctx->reduce3_w, dq1), to_f32(ctx->reduce3_b, dq2),
           ch, 1, 1, 0, 1, 1, x.data());

    // Decoder level 3: 6 TB
    auto t_dec3 = std::chrono::steady_clock::now();
    for (auto & tb : ctx->dec3) tb_forward(x.data(), ch, cur_h, cur_w, heads[2], tb, dq1, dq2);

    // fre2(inp_img, out_dec3): replace dec3 output
    fre_forward(input_chw, H, W, x.data(), ch, cur_h, cur_w, heads[2],
                ctx->fre2, x.data(), dq1, dq2);
    if (bench) {
        auto t_dec3_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[adair-bench] dec3+fre2: %.1f ms\n",
                ms_f(t_dec3_end - t_dec3).count());
    }

    // Upsample 3→2 + cat(skip2) + reduce
    up_ch = ch * 2;
    us.resize(up_ch * cur_h * cur_w);
    conv2d(x.data(), ch, cur_h, cur_w, to_f32(ctx->up32.w, dq1), to_f32(ctx->up32.b, dq2),
           up_ch, 3, 3, 1, 1, 1, us.data());
    ch /= 2; cur_h *= 2; cur_w *= 2;
    x.resize(ch * cur_h * cur_w);
    pixel_shuffle(us.data(), up_ch, cur_h / 2, cur_w / 2, 2, x.data());
    cat.resize(2 * ch * cur_h * cur_w);
    memcpy(cat.data(), x.data(), ch * cur_h * cur_w * sizeof(float));
    memcpy(cat.data() + ch * cur_h * cur_w, skip2.data(), ch * cur_h * cur_w * sizeof(float));
    conv2d(cat.data(), 2 * ch, cur_h, cur_w, to_f32(ctx->reduce2_w, dq1), to_f32(ctx->reduce2_b, dq2),
           ch, 1, 1, 0, 1, 1, x.data());

    // Decoder level 2: 6 TB
    auto t_dec2 = std::chrono::steady_clock::now();
    for (auto & tb : ctx->dec2) tb_forward(x.data(), ch, cur_h, cur_w, heads[1], tb, dq1, dq2);

    // fre3(inp_img, out_dec2): replace dec2 output
    fre_forward(input_chw, H, W, x.data(), ch, cur_h, cur_w, heads[2],
                ctx->fre3, x.data(), dq1, dq2);
    if (bench) {
        auto t_dec2_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[adair-bench] dec2+fre3: %.1f ms\n",
                ms_f(t_dec2_end - t_dec2).count());
    }

    // Upsample 2→1 + cat(skip1) → 96ch (no reduce, dec1 is 96ch)
    up_ch = ch * 2;
    us.resize(up_ch * cur_h * cur_w);
    conv2d(x.data(), ch, cur_h, cur_w, to_f32(ctx->up21.w, dq1), to_f32(ctx->up21.b, dq2),
           up_ch, 3, 3, 1, 1, 1, us.data());
    ch /= 2; cur_h *= 2; cur_w *= 2;
    x.resize(ch * cur_h * cur_w);
    pixel_shuffle(us.data(), up_ch, cur_h / 2, cur_w / 2, 2, x.data());
    cat.resize(2 * ch * cur_h * cur_w);
    memcpy(cat.data(), x.data(), ch * cur_h * cur_w * sizeof(float));
    memcpy(cat.data() + ch * cur_h * cur_w, skip1.data(), ch * cur_h * cur_w * sizeof(float));
    ch *= 2; // now 96
    x.assign(cat.begin(), cat.end());

    // Decoder level 1: 4 TB at 96ch
    auto t_dec1 = std::chrono::steady_clock::now();
    for (auto & tb : ctx->dec1) tb_forward(x.data(), ch, cur_h, cur_w, heads[0], tb, dq1, dq2);

    // Refinement: 4 TB at 96ch
    for (auto & tb : ctx->refinement) tb_forward(x.data(), ch, cur_h, cur_w, heads[0], tb, dq1, dq2);

    // Output: Conv3x3(96→3) + residual
    conv2d(x.data(), ch, cur_h, cur_w, to_f32(ctx->output_w, dq1), to_f32(ctx->output_b, dq2),
           3, 3, 3, 1, 1, 1, output_chw);
    for (int i = 0; i < 3 * H * W; i++) output_chw[i] += input_chw[i];

    if (bench) {
        auto t_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[adair-bench] dec1+refine+output: %.1f ms\n",
                ms_f(t_end - t_dec1).count());
        fprintf(stderr, "[adair-bench] total: %.1f ms\n",
                ms_f(t_end - t_total).count());
    }
    return 0;
}

static int adair_process_tile(adair_context * ctx,
                              const uint8_t * input, int width, int height,
                              uint8_t * output) {
    int hw = width * height;
    std::vector<float> in_chw(3 * hw);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                in_chw[c * hw + y * width + x] = (float)input[(y * width + x) * 3 + c] / 255.0f;
    std::vector<float> out_chw(3 * hw);
    int ret = adair_process_float(ctx, in_chw.data(), width, height, out_chw.data());
    if (ret != 0) return ret;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++) {
                float v = out_chw[c * hw + y * width + x] * 255.0f;
                output[(y * width + x) * 3 + c] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
    return 0;
}

static void adair_blend_win(int ts, int ov, std::vector<float> & w) {
    w.resize(ts * ts);
    for (int y = 0; y < ts; y++) {
        float wy = 1.0f;
        if (y < ov) wy = 0.5f - 0.5f * cosf((float)M_PI * y / ov);
        else if (y >= ts - ov) wy = 0.5f - 0.5f * cosf((float)M_PI * (ts - 1 - y) / ov);
        for (int x = 0; x < ts; x++) {
            float wx = 1.0f;
            if (x < ov) wx = 0.5f - 0.5f * cosf((float)M_PI * x / ov);
            else if (x >= ts - ov) wx = 0.5f - 0.5f * cosf((float)M_PI * (ts - 1 - x) / ov);
            w[y * ts + x] = wy * wx;
        }
    }
}

int adair_process(adair_context * ctx,
                  const uint8_t * input, int width, int height,
                  uint8_t * output) {
    if (!ctx || !input || !output) return -1;
    // AdaIR U-Net needs tile_size multiple of 8 (3 downsample levels)
    int tile_size = 256;
    int tile_overlap = 32;
    const char * ts_env = std::getenv("CRISPEMBED_ADAIR_TILE");
    if (ts_env) tile_size = std::max(64, (atoi(ts_env) / 8) * 8);
    tile_overlap = std::min(tile_overlap, tile_size / 4);

    if (width <= tile_size && height <= tile_size)
        return adair_process_tile(ctx, input, width, height, output);

    std::vector<float> accum(3 * height * width, 0.0f);
    std::vector<float> wmap(height * width, 0.0f);
    std::vector<float> bwin;
    adair_blend_win(tile_size, tile_overlap, bwin);
    int step = tile_size - tile_overlap;
    int ntx = std::max(1, (width + step - 1) / step);
    int nty = std::max(1, (height + step - 1) / step);
    fprintf(stderr, "adair: %dx%d, tiles=%dx%d (size=%d)\n", width, height, ntx, nty, tile_size);
    for (int ty = 0; ty < nty; ty++)
        for (int tx = 0; tx < ntx; tx++) {
            int x0 = std::min(tx*step, std::max(0, width-tile_size));
            int y0 = std::min(ty*step, std::max(0, height-tile_size));
            int tw = std::min(tile_size, width-x0), th = std::min(tile_size, height-y0);
            std::vector<uint8_t> ti(tw*th*3), to(tw*th*3);
            for (int y=0;y<th;y++) memcpy(ti.data()+y*tw*3, input+((y0+y)*width+x0)*3, tw*3);
            if (adair_process_tile(ctx, ti.data(), tw, th, to.data()) != 0) return -1;
            for (int y=0;y<th;y++)
                for (int x=0;x<tw;x++) {
                    float w = (tw==tile_size&&th==tile_size) ? bwin[y*tile_size+x] : 1.0f;
                    if (tw!=tile_size||th!=tile_size) {
                        if (x0>0&&x<tile_overlap) w*=0.5f-0.5f*cosf((float)M_PI*x/tile_overlap);
                        if (y0>0&&y<tile_overlap) w*=0.5f-0.5f*cosf((float)M_PI*y/tile_overlap);
                    }
                    int dy=y0+y, dx=x0+x;
                    for (int c=0;c<3;c++) accum[c*height*width+dy*width+dx]+=to[(y*tw+x)*3+c]*w;
                    wmap[dy*width+dx]+=w;
                }
        }
    for (int y=0;y<height;y++)
        for (int x=0;x<width;x++) {
            float wt=wmap[y*width+x]; if(wt<=0)wt=1;
            for (int c=0;c<3;c++) {
                float v=accum[c*height*width+y*width+x]/wt;
                output[(y*width+x)*3+c]=(uint8_t)std::max(0.f,std::min(255.f,v+0.5f));
            }
        }
    return 0;
}
