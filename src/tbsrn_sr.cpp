// tbsrn_sr.cpp — TBSRN text-line super-resolution (CPU-scalar).
//
// Forward pass (inference only, no STN/GRU):
//   block1: Conv9(3→64, p=4) + PReLU
//   5× RecurrentResidualBlock:
//     Conv3(64→64, p=1) + BN + mish + Conv3(64→64, p=1) + BN
//     → reshape [B,64,H*W] → FeatureEnhancer → reshape back → residual
//   block7: Conv3(64→64, p=1) + BN   (runs on block6 output)
//   block8: (block1 + block7) → Conv3(64→256, p=1) + PixelShuffle(2) + mish
//           → Conv9(64→3, p=4) → tanh
//
// FeatureEnhancer:
//   concat(input[64], PE2D[64]) → [128, T] → transpose → [T, 128]
//   → MHA(h=4, d=128) + LN + FFN(128→128) + LN → Linear(128→64)
//   → transpose → [64, T]

#include "tbsrn_sr.h"
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

// ── Helpers ────────────────────────────────────────────────────────────

static const float * tbsrn_to_f32(const ggml_tensor * t, std::vector<float> & buf) {
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

// Conv2D: [OC, IC, KH, KW] weights, [OC] bias, planar [C, H, W]
static void tbsrn_conv2d(const float * input, int ic, int ih, int iw,
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

// BatchNorm2D eval: y = weight * (x - mean) / sqrt(var + eps) + bias
static void tbsrn_batchnorm2d(float * data, int c, int h, int w,
                               const float * weight, const float * bias,
                               const float * running_mean, const float * running_var) {
    int hw = h * w;
    for (int ch = 0; ch < c; ch++) {
        float scale = weight[ch] / sqrtf(running_var[ch] + 1e-5f);
        float shift = bias[ch] - running_mean[ch] * scale;
        for (int i = 0; i < hw; i++)
            data[ch * hw + i] = data[ch * hw + i] * scale + shift;
    }
}

static void tbsrn_prelu(float * data, int c, int hw, const float * slope) {
    // slope shape (1,) → shared across channels
    float a = slope[0];
    for (int i = 0; i < c * hw; i++)
        data[i] = data[i] > 0 ? data[i] : data[i] * a;
}

static void tbsrn_mish(float * data, int n) {
    for (int i = 0; i < n; i++) {
        float x = data[i];
        float sp = x > 20.0f ? x : logf(1.0f + expf(std::min(x, 20.0f)));
        data[i] = x * tanhf(sp);
    }
}

static void tbsrn_pixel_shuffle(const float * input, int c_in, int h, int w,
                                int r, float * output) {
    int c_out = c_in / (r * r);
    int oh = h * r, ow = w * r;
    for (int c = 0; c < c_out; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                int iy = y / r, ix = x / r;
                int ic = c * r * r + (y % r) * r + (x % r);
                output[c * oh * ow + y * ow + x] = input[ic * h * w + iy * w + ix];
            }
}

// LayerNorm on last dimension: data [T, D] in-place
static void tbsrn_layernorm(float * data, int T, int D,
                             const float * weight, const float * bias) {
    for (int t = 0; t < T; t++) {
        float * row = data + t * D;
        float mean = 0;
        for (int d = 0; d < D; d++) mean += row[d];
        mean /= D;
        float var = 0;
        for (int d = 0; d < D; d++) { float x = row[d] - mean; var += x * x; }
        var /= D;
        float inv = 1.0f / sqrtf(var + 1e-6f);
        for (int d = 0; d < D; d++)
            row[d] = (row[d] - mean) * inv * weight[d] + bias[d];
    }
}

// Linear: out[..., OD] = in[..., ID] @ W^T + b, W=[OD, ID]
static void tbsrn_linear(const float * input, int n, int id, int od,
                         const float * weight, const float * bias,
                         float * output) {
    for (int i = 0; i < n; i++) {
        const float * in_row = input + i * id;
        float * out_row = output + i * od;
        for (int o = 0; o < od; o++) {
            float sum = bias[o];
            for (int j = 0; j < id; j++)
                sum += in_row[j] * weight[o * id + j];
            out_row[o] = sum;
        }
    }
}

// Multi-head self-attention: Q=K=V=input [T, D], output [T, D]
static void tbsrn_mha(const float * input, int T, int D, int n_heads,
                       const float * Wq, const float * Bq,
                       const float * Wk, const float * Bk,
                       const float * Wv, const float * Bv,
                       const float * Wo, const float * Bo,
                       float * output,
                       std::vector<float> & scratch) {
    int d_k = D / n_heads;
    int batch_T_D = T * D;

    scratch.resize(4 * batch_T_D + T * T * n_heads);
    float * Q = scratch.data();
    float * K = Q + batch_T_D;
    float * V = K + batch_T_D;
    float * tmp = V + batch_T_D;
    float * scores = tmp; // reuse

    // Project Q, K, V
    tbsrn_linear(input, T, D, D, Wq, Bq, Q);
    tbsrn_linear(input, T, D, D, Wk, Bk, K);
    tbsrn_linear(input, T, D, D, Wv, Bv, V);

    // Scaled dot-product attention per head
    float scale = 1.0f / sqrtf((float)d_k);

    // Reorganize to [h, T, d_k] layout for easier matmul
    // Q/K/V are [T, D] = [T, h*d_k], we process head-by-head
    // output is accumulated in tmp (reused)
    std::vector<float> attn_out(batch_T_D, 0.0f);

    for (int h = 0; h < n_heads; h++) {
        // Extract head h: Q_h[t] = Q[t, h*d_k : (h+1)*d_k]
        // Compute scores[t1, t2] = sum_d Q_h[t1,d] * K_h[t2,d] * scale
        for (int t1 = 0; t1 < T; t1++) {
            float max_score = -1e9f;
            for (int t2 = 0; t2 < T; t2++) {
                float s = 0;
                for (int d = 0; d < d_k; d++)
                    s += Q[t1 * D + h * d_k + d] * K[t2 * D + h * d_k + d];
                s *= scale;
                scores[t1 * T + t2] = s;
                if (s > max_score) max_score = s;
            }
            // Softmax
            float sum_exp = 0;
            for (int t2 = 0; t2 < T; t2++) {
                scores[t1 * T + t2] = expf(scores[t1 * T + t2] - max_score);
                sum_exp += scores[t1 * T + t2];
            }
            for (int t2 = 0; t2 < T; t2++)
                scores[t1 * T + t2] /= sum_exp;

            // Weighted sum of V
            for (int d = 0; d < d_k; d++) {
                float val = 0;
                for (int t2 = 0; t2 < T; t2++)
                    val += scores[t1 * T + t2] * V[t2 * D + h * d_k + d];
                attn_out[t1 * D + h * d_k + d] = val;
            }
        }
    }

    // Output projection
    tbsrn_linear(attn_out.data(), T, D, D, Wo, Bo, output);
}

// 2D positional encoding: [d_model, H, W]
static void tbsrn_pe2d(int d_model, int H, int W, float * pe) {
    memset(pe, 0, d_model * H * W * sizeof(float));
    int d_half = d_model / 2;
    std::vector<float> div_term(d_half / 2);
    for (int i = 0; i < d_half / 2; i++)
        div_term[i] = expf(i * 2.0f * -(logf(10000.0f) / d_half));

    // Width encoding: first d_half channels
    for (int x = 0; x < W; x++) {
        for (int k = 0; k < d_half / 2; k++) {
            float val_sin = sinf(x * div_term[k]);
            float val_cos = cosf(x * div_term[k]);
            for (int y = 0; y < H; y++) {
                pe[(2 * k) * H * W + y * W + x] = val_sin;
                pe[(2 * k + 1) * H * W + y * W + x] = val_cos;
            }
        }
    }
    // Height encoding: second d_half channels
    for (int y = 0; y < H; y++) {
        for (int k = 0; k < d_half / 2; k++) {
            float val_sin = sinf(y * div_term[k]);
            float val_cos = cosf(y * div_term[k]);
            for (int x = 0; x < W; x++) {
                pe[(d_half + 2 * k) * H * W + y * W + x] = val_sin;
                pe[(d_half + 2 * k + 1) * H * W + y * W + x] = val_cos;
            }
        }
    }
}

// Bilinear resize: [C, H_in, W_in] → [C, H_out, W_out]
static void tbsrn_resize(const float * src, int c, int h_in, int w_in,
                         int h_out, int w_out, float * dst) {
    for (int ch = 0; ch < c; ch++) {
        for (int oy = 0; oy < h_out; oy++) {
            float sy = (oy + 0.5f) * h_in / h_out - 0.5f;
            int iy = (int)floorf(sy);
            float fy = sy - iy;
            int iy0 = std::max(0, iy), iy1 = std::min(h_in - 1, iy + 1);
            for (int ox = 0; ox < w_out; ox++) {
                float sx = (ox + 0.5f) * w_in / w_out - 0.5f;
                int ix = (int)floorf(sx);
                float fx = sx - ix;
                int ix0 = std::max(0, ix), ix1 = std::min(w_in - 1, ix + 1);
                float v = (1 - fy) * ((1 - fx) * src[ch * h_in * w_in + iy0 * w_in + ix0]
                                    + fx * src[ch * h_in * w_in + iy0 * w_in + ix1])
                        + fy * ((1 - fx) * src[ch * h_in * w_in + iy1 * w_in + ix0]
                              + fx * src[ch * h_in * w_in + iy1 * w_in + ix1]);
                dst[ch * h_out * w_out + oy * w_out + ox] = v;
            }
        }
    }
}

// ── Context ────────────────────────────────────────────────────────────

struct tbsrn_sr_context {
    int srb_nums;
    int hidden_units;  // 32 → 2*hidden_units = 64 channels
    int upscale_factor;
    int n_threads;
    bool bench;

    core_gguf::WeightLoad wl;
    std::vector<std::vector<float>> weight_bufs;

    const float * get(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) {
            fprintf(stderr, "tbsrn_sr: missing tensor %s\n", name.c_str());
            return nullptr;
        }
        weight_bufs.emplace_back();
        return tbsrn_to_f32(t, weight_bufs.back());
    }
};

tbsrn_sr_context * tbsrn_sr_init(const char * model_path, int n_threads) {
    auto * ctx = new tbsrn_sr_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "tbsrn_sr: failed to open %s\n", model_path);
        delete ctx;
        return nullptr;
    }

    ctx->srb_nums       = core_gguf::kv_u32(meta, "tbsrn.srb_nums", 5);
    ctx->hidden_units    = core_gguf::kv_u32(meta, "tbsrn.hidden_units", 32);
    ctx->upscale_factor  = core_gguf::kv_u32(meta, "tbsrn.upscale_factor", 2);
    core_gguf::free_metadata(meta);

    bool force_cpu = (getenv("TBSRN_SR_FORCE_CPU") && atoi(getenv("TBSRN_SR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, backend, "tbsrn", ctx->wl)) {
        fprintf(stderr, "tbsrn_sr: failed to load weights\n");
        ggml_backend_free(backend);
        delete ctx;
        return nullptr;
    }
    ggml_backend_free(backend);

    int C = 2 * ctx->hidden_units;
    fprintf(stderr, "tbsrn_sr: srb_nums=%d, channels=%d, upscale=%dx, %d tensors\n",
            ctx->srb_nums, C, ctx->upscale_factor, (int)ctx->wl.tensors.size());
    ctx->bench = (std::getenv("CRISPEMBED_TBSRN_SR_BENCH") != nullptr);
    return ctx;
}

void tbsrn_sr_free(tbsrn_sr_context * ctx) {
    if (ctx) {
        core_gguf::free_weights(ctx->wl);
        delete ctx;
    }
}

// ── Forward pass ───────────────────────────────────────────────────────

int tbsrn_sr_process(tbsrn_sr_context * ctx,
                     const uint8_t * input, int width, int height,
                     uint8_t ** output, int * out_width, int * out_height) {
    if (!ctx || !input || !output || width <= 0 || height <= 0) return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    int C = 2 * ctx->hidden_units;  // 64
    int LR_H = 16, LR_W = 64;
    int HR_H = LR_H * ctx->upscale_factor;
    int HR_W = LR_W * ctx->upscale_factor;

    // Convert input to [3, H, W] float [0, 1]
    std::vector<float> input_f(3 * height * width);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                input_f[c * height * width + y * width + x] =
                    input[(y * width + x) * 3 + c] / 255.0f;

    // Resize to LR size (16×64) if needed
    std::vector<float> lr(3 * LR_H * LR_W);
    if (width == LR_W && height == LR_H) {
        lr = std::vector<float>(input_f.begin(), input_f.begin() + 3 * LR_H * LR_W);
    } else {
        tbsrn_resize(input_f.data(), 3, height, width, LR_H, LR_W, lr.data());
    }

    // Normalize to [-1, 1] (TBSRN convention: tanh output range)
    // Actually looking at PaddleOCR code, input is [0,1] — the model outputs tanh [-1,1]
    // and postproc converts back. Let's keep input [0,1].

    // block1: Conv9(3→64, p=4) + PReLU
    std::vector<float> x(C * LR_H * LR_W);
    tbsrn_conv2d(lr.data(), 3, LR_H, LR_W,
                 ctx->get("block1.conv.weight"), ctx->get("block1.conv.bias"),
                 C, 9, 9, 4, x.data());
    tbsrn_prelu(x.data(), C, LR_H * LR_W, ctx->get("block1.prelu.weight"));
    std::vector<float> block1_out = x;

    // Scratch for FeatureEnhancer
    std::vector<float> mha_scratch;
    int T = LR_H * LR_W;  // 1024
    int D_fe = 128;

    // 5× RecurrentResidualBlock
    for (int i = 0; i < ctx->srb_nums; i++) {
        auto t_blk = std::chrono::steady_clock::now();
        char prefix[32];
        snprintf(prefix, sizeof(prefix), "srb.%d", i);
        std::string p(prefix);

        // Conv1 + BN1 + mish
        std::vector<float> residual(C * LR_H * LR_W);
        tbsrn_conv2d(x.data(), C, LR_H, LR_W,
                     ctx->get(p + ".conv1.weight"), ctx->get(p + ".conv1.bias"),
                     C, 3, 3, 1, residual.data());
        tbsrn_batchnorm2d(residual.data(), C, LR_H, LR_W,
                          ctx->get(p + ".bn1.weight"), ctx->get(p + ".bn1.bias"),
                          ctx->get(p + ".bn1.running_mean"), ctx->get(p + ".bn1.running_var"));
        tbsrn_mish(residual.data(), C * LR_H * LR_W);

        // Conv2 + BN2
        std::vector<float> res2(C * LR_H * LR_W);
        tbsrn_conv2d(residual.data(), C, LR_H, LR_W,
                     ctx->get(p + ".conv2.weight"), ctx->get(p + ".conv2.bias"),
                     C, 3, 3, 1, res2.data());
        tbsrn_batchnorm2d(res2.data(), C, LR_H, LR_W,
                          ctx->get(p + ".bn2.weight"), ctx->get(p + ".bn2.bias"),
                          ctx->get(p + ".bn2.running_mean"), ctx->get(p + ".bn2.running_var"));

        // FeatureEnhancer
        // res2 is [C, H, W] → reshape to [C, T] where T = H*W = 1024
        // Concat with PE2D → [128, T]
        std::vector<float> pe(64 * LR_H * LR_W);
        tbsrn_pe2d(64, LR_H, LR_W, pe.data());

        std::vector<float> feat(D_fe * T);  // [128, T]
        memcpy(feat.data(), res2.data(), C * T * sizeof(float));
        memcpy(feat.data() + C * T, pe.data(), 64 * T * sizeof(float));

        // Transpose [128, T] → [T, 128]
        std::vector<float> feat_t(T * D_fe);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < D_fe; d++)
                feat_t[t * D_fe + d] = feat[d * T + t];

        // Save for residual
        std::vector<float> origin = feat_t;

        // MHA self-attention
        std::string fe = p + ".fe";
        std::vector<float> mha_out(T * D_fe);
        tbsrn_mha(feat_t.data(), T, D_fe, 4,
                  ctx->get(fe + ".mha.linear0.weight"), ctx->get(fe + ".mha.linear0.bias"),
                  ctx->get(fe + ".mha.linear1.weight"), ctx->get(fe + ".mha.linear1.bias"),
                  ctx->get(fe + ".mha.linear2.weight"), ctx->get(fe + ".mha.linear2.bias"),
                  ctx->get(fe + ".mha.linear3.weight"), ctx->get(fe + ".mha.linear3.bias"),
                  mha_out.data(), mha_scratch);

        // LN1(origin + mha_out)
        for (int j = 0; j < T * D_fe; j++) feat_t[j] = origin[j] + mha_out[j];
        tbsrn_layernorm(feat_t.data(), T, D_fe,
                        ctx->get(fe + ".ln1.weight"), ctx->get(fe + ".ln1.bias"));

        std::vector<float> origin2 = feat_t;

        // FFN: Linear(128→128) + ReLU + Linear(128→128)
        std::vector<float> ffn_tmp(T * D_fe);
        tbsrn_linear(feat_t.data(), T, D_fe, D_fe,
                     ctx->get(fe + ".ffn.w1.weight"), ctx->get(fe + ".ffn.w1.bias"),
                     ffn_tmp.data());
        for (int j = 0; j < T * D_fe; j++)
            ffn_tmp[j] = ffn_tmp[j] > 0 ? ffn_tmp[j] : 0;  // ReLU
        std::vector<float> ffn_out(T * D_fe);
        tbsrn_linear(ffn_tmp.data(), T, D_fe, D_fe,
                     ctx->get(fe + ".ffn.w2.weight"), ctx->get(fe + ".ffn.w2.bias"),
                     ffn_out.data());

        // LN3(origin2 + ffn_out)
        for (int j = 0; j < T * D_fe; j++) feat_t[j] = origin2[j] + ffn_out[j];
        tbsrn_layernorm(feat_t.data(), T, D_fe,
                        ctx->get(fe + ".ln3.weight"), ctx->get(fe + ".ln3.bias"));

        // Output linear (128→64)
        std::vector<float> fe_out(T * C);
        tbsrn_linear(feat_t.data(), T, D_fe, C,
                     ctx->get(fe + ".linear.weight"), ctx->get(fe + ".linear.bias"),
                     fe_out.data());

        // Transpose [T, 64] → [64, T] → reshape [64, H, W]
        std::vector<float> fe_spatial(C * T);
        for (int t = 0; t < T; t++)
            for (int d = 0; d < C; d++)
                fe_spatial[d * T + t] = fe_out[t * C + d];

        // Residual: x = x + fe_spatial
        for (int j = 0; j < C * T; j++)
            x[j] += fe_spatial[j];
        if (bench) {
            auto t_blk_end = std::chrono::steady_clock::now();
            fprintf(stderr, "[tbsrn_sr-bench] block %d: %.1f ms\n",
                    i, ms_f(t_blk_end - t_blk).count());
        }
    }

    // block7: Conv3(64→64) + BN on block6 output (= x)
    std::vector<float> final_out(C * LR_H * LR_W);
    tbsrn_conv2d(x.data(), C, LR_H, LR_W,
                 ctx->get("final_conv.weight"), ctx->get("final_conv.bias"),
                 C, 3, 3, 1, final_out.data());
    tbsrn_batchnorm2d(final_out.data(), C, LR_H, LR_W,
                      ctx->get("final_bn.weight"), ctx->get("final_bn.bias"),
                      ctx->get("final_bn.running_mean"), ctx->get("final_bn.running_var"));

    // block8 input: block1 + block7
    std::vector<float> up_input(C * LR_H * LR_W);
    for (int j = 0; j < C * LR_H * LR_W; j++)
        up_input[j] = block1_out[j] + final_out[j];

    // UpsampleBlock: Conv(64→256, k=3, p=1) + PixelShuffle(2) + mish
    int up_oc = C * ctx->upscale_factor * ctx->upscale_factor;  // 256
    std::vector<float> up_conv(up_oc * LR_H * LR_W);
    tbsrn_conv2d(up_input.data(), C, LR_H, LR_W,
                 ctx->get("upsample.conv.weight"), ctx->get("upsample.conv.bias"),
                 up_oc, 3, 3, 1, up_conv.data());

    std::vector<float> up_ps(C * HR_H * HR_W);
    tbsrn_pixel_shuffle(up_conv.data(), up_oc, LR_H, LR_W, ctx->upscale_factor, up_ps.data());
    tbsrn_mish(up_ps.data(), C * HR_H * HR_W);

    // Final conv: Conv(64→3, k=9, p=4) + tanh
    std::vector<float> sr_out(3 * HR_H * HR_W);
    tbsrn_conv2d(up_ps.data(), C, HR_H, HR_W,
                 ctx->get("output_conv.weight"), ctx->get("output_conv.bias"),
                 3, 9, 9, 4, sr_out.data());
    for (int j = 0; j < 3 * HR_H * HR_W; j++)
        sr_out[j] = tanhf(sr_out[j]);

    // Convert tanh output [-1, 1] → uint8 [0, 255]
    uint8_t * out_buf = (uint8_t *)malloc(3 * HR_H * HR_W);
    if (!out_buf) return -1;
    for (int y = 0; y < HR_H; y++)
        for (int x = 0; x < HR_W; x++)
            for (int c = 0; c < 3; c++) {
                float v = (sr_out[c * HR_H * HR_W + y * HR_W + x] + 1.0f) * 0.5f * 255.0f;
                out_buf[(y * HR_W + x) * 3 + c] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }

    *output = out_buf;
    *out_width = HR_W;
    *out_height = HR_H;
    fprintf(stderr, "tbsrn_sr: done %dx%d → %dx%d\n", width, height, HR_W, HR_H);
    if (bench) {
        auto t_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[tbsrn_sr-bench] total: %.1f ms\n",
                ms_f(t_end - t_total).count());
    }
    return 0;
}

void tbsrn_sr_free_image(uint8_t * pixels) {
    free(pixels);
}
