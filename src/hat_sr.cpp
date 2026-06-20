// hat_sr.cpp — HAT super-resolution (CPU-scalar).
//
// Forward:
//   x = (input - mean) * img_range
//   shallow = conv_first(x)
//   deep = patch_embed → 6× RHAG → norm → patch_unembed → conv_after_body + shallow
//   output = conv_last(upsample(conv_before_upsample(deep))) / img_range + mean
//
// RHAG (Residual Hybrid Attention Group):
//   attn_blocks(x) → OCAB(x) → reshape → conv → reshape → + residual
//
// HAB (Hybrid Attention Block):
//   LN → [window_partition → W-MSA/SW-MSA → window_reverse] + conv_scale*CAB → + shortcut
//   → LN → MLP → + shortcut
//
// OCAB (Overlapping Cross-Attention Block):
//   LN → QKV → Q windows + KV unfold → cross-attention → window_reverse → proj → + shortcut
//   → MLP → + shortcut
//
// CAB (Channel Attention Block):
//   Conv3 → GELU → Conv3 → ChannelAttention(AvgPool → Conv1 → ReLU → Conv1 → Sigmoid → mul)

#include "hat_sr.h"
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

// ── Helpers ────────────────────────────────────────────────────────────

// Conv2D: weight [OC, IC, KH, KW], groups=1
static void hat_conv2d(const float * input, int ic, int h, int w,
                       const float * weight, const float * bias,
                       int oc, int kh, int kw, int pad,
                       float * output) {
    int oh = h + 2 * pad - kh + 1, ow = w + 2 * pad - kw + 1;
    for (int o = 0; o < oc; o++) {
        float b = bias ? bias[o] : 0.0f;
        for (int oy = 0; oy < oh; oy++)
            for (int ox = 0; ox < ow; ox++) {
                float sum = b;
                for (int c = 0; c < ic; c++)
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy + ky - pad, ix = ox + kx - pad;
                            if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                sum += input[c * h * w + iy * w + ix]
                                     * weight[o * ic * kh * kw + c * kh * kw + ky * kw + kx];
                        }
                output[o * oh * ow + oy * ow + ox] = sum;
            }
    }
}

// Linear: out[..., OD] = in[..., ID] @ W^T + b, W=[OD, ID]
static void hat_linear(const float * input, int n, int id, int od,
                       const float * weight, const float * bias,
                       float * output) {
    core_cpu::linear_batch_cpu(input, output, n, id, od, weight, bias);
}

// LayerNorm on last dim: data[N, D] in-place
static void hat_layernorm(float * data, int N, int D,
                          const float * weight, const float * bias) {
    for (int n = 0; n < N; n++) {
        float * row = data + n * D;
        float mean = 0;
        for (int d = 0; d < D; d++) mean += row[d];
        mean /= D;
        float var = 0;
        for (int d = 0; d < D; d++) { float x = row[d] - mean; var += x * x; }
        var /= D;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int d = 0; d < D; d++)
            row[d] = (row[d] - mean) * inv * weight[d] + bias[d];
    }
}

static float hat_gelu(float x) {
    // Exact GELU: x * Phi(x) = x * 0.5 * (1 + erf(x/sqrt(2)))
    return x * 0.5f * (1.0f + erff(x * 0.7071067811865476f));
}

static void hat_pixel_shuffle(const float * input, int c_in, int h, int w,
                              int r, float * output) {
    int c_out = c_in / (r * r), oh = h * r, ow = w * r;
    for (int c = 0; c < c_out; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                output[c * oh * ow + y * ow + x] =
                    input[(c * r * r + (y % r) * r + (x % r)) * h * w + (y / r) * w + (x / r)];
}

// ── Window operations ──────────────────────────────────────────────────

// window_partition: [B, H, W, C] → [(B*nH*nW), ws, ws, C]
// Data is in [HW, C] layout (sequence format). H and W are spatial dims.
static void hat_window_partition(const float * input, int H, int W, int C, int ws,
                                 float * output) {
    int nH = H / ws, nW = W / ws;
    // input layout: [H*W, C] row-major
    // output layout: [nH*nW, ws*ws, C]
    for (int wh = 0; wh < nH; wh++)
        for (int ww = 0; ww < nW; ww++)
            for (int y = 0; y < ws; y++)
                for (int x = 0; x < ws; x++) {
                    int src_idx = (wh * ws + y) * W + (ww * ws + x);
                    int dst_idx = (wh * nW + ww) * ws * ws + y * ws + x;
                    for (int c = 0; c < C; c++)
                        output[dst_idx * C + c] = input[src_idx * C + c];
                }
}

// window_reverse: [(nH*nW), ws*ws, C] → [H*W, C]
static void hat_window_reverse(const float * input, int H, int W, int C, int ws,
                               float * output) {
    int nH = H / ws, nW = W / ws;
    for (int wh = 0; wh < nH; wh++)
        for (int ww = 0; ww < nW; ww++)
            for (int y = 0; y < ws; y++)
                for (int x = 0; x < ws; x++) {
                    int src_idx = (wh * nW + ww) * ws * ws + y * ws + x;
                    int dst_idx = (wh * ws + y) * W + (ww * ws + x);
                    for (int c = 0; c < C; c++)
                        output[dst_idx * C + c] = input[src_idx * C + c];
                }
}

// Cyclic shift: [H, W, C] → shifted by (-sh, -sh) with wrap-around
static void hat_cyclic_shift(const float * input, int H, int W, int C, int sh,
                             float * output) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            int sy = ((y + sh) % H + H) % H;
            int sx = ((x + sh) % W + W) % W;
            for (int c = 0; c < C; c++)
                output[(sy * W + sx) * C + c] = input[(y * W + x) * C + c];
        }
}

// ── Window Attention ───────────────────────────────────────────────────

static void hat_window_attn(const float * x, int nw, int ws2, int C, int n_heads,
                            const float * qkv_w, const float * qkv_b,
                            const float * proj_w, const float * proj_b,
                            const float * rpb_table, const int * rpi,
                            const float * attn_mask, int n_mask_windows,
                            float scale,
                            float * output) {
    int d = C / n_heads;
    // x: [nw, ws2, C] → QKV: [nw, ws2, 3*C]
    std::vector<float> qkv(nw * ws2 * 3 * C);
    hat_linear(x, nw * ws2, C, 3 * C, qkv_w, qkv_b, qkv.data());

    // Split into Q, K, V and reshape to [nw, n_heads, ws2, d]
    // Then compute attention per window per head
    std::vector<float> out(nw * ws2 * C, 0.0f);

    // Pre-extract Q, K, V into contiguous [nw, n_heads, ws2, d] layout for speed
    std::vector<float> Q(nw * n_heads * ws2 * d);
    std::vector<float> K(nw * n_heads * ws2 * d);
    std::vector<float> V(nw * n_heads * ws2 * d);
    for (int w = 0; w < nw; w++) {
        for (int t = 0; t < ws2; t++) {
            const float * src = &qkv[(w * ws2 + t) * 3 * C];
            for (int h = 0; h < n_heads; h++) {
                int dst_idx = ((w * n_heads + h) * ws2 + t) * d;
                for (int di = 0; di < d; di++) {
                    Q[dst_idx + di] = src[0 * C + h * d + di] * scale;
                    K[dst_idx + di] = src[1 * C + h * d + di];
                    V[dst_idx + di] = src[2 * C + h * d + di];
                }
            }
        }
    }

    // Attention: per window, per head
    std::vector<float> scores(ws2);
    for (int w = 0; w < nw; w++) {
        for (int h = 0; h < n_heads; h++) {
            int qk_base = (w * n_heads + h) * ws2 * d;
            const float * Q_wh = &Q[qk_base];
            const float * K_wh = &K[qk_base];
            const float * V_wh = &V[qk_base];

            for (int q_pos = 0; q_pos < ws2; q_pos++) {
                const float * q_row = &Q_wh[q_pos * d];
                float max_score = -1e9f;

                for (int k_pos = 0; k_pos < ws2; k_pos++) {
                    const float * k_row = &K_wh[k_pos * d];
                    float s = core_cpu::dot_product(q_row, k_row, d);
                    s += rpb_table[rpi[q_pos * ws2 + k_pos] * n_heads + h];
                    if (attn_mask && n_mask_windows > 0)
                        s += attn_mask[(w % n_mask_windows) * ws2 * ws2 + q_pos * ws2 + k_pos];
                    scores[k_pos] = s;
                    if (s > max_score) max_score = s;
                }

                float sum_exp = 0;
                for (int k = 0; k < ws2; k++) {
                    scores[k] = expf(scores[k] - max_score);
                    sum_exp += scores[k];
                }
                float inv_sum = 1.0f / sum_exp;
                for (int k = 0; k < ws2; k++) scores[k] *= inv_sum;

                for (int di = 0; di < d; di++) {
                    float val = 0;
                    for (int k = 0; k < ws2; k++) val += scores[k] * V_wh[k * d + di];
                    out[(w * ws2 + q_pos) * C + h * d + di] = val;
                }
            }
        }
    }

    // Output projection
    std::vector<float> proj(nw * ws2 * C);
    hat_linear(out.data(), nw * ws2, C, C, proj_w, proj_b, proj.data());
    memcpy(output, proj.data(), nw * ws2 * C * sizeof(float));
}

// ── CAB (Channel Attention Block) ──────────────────────────────────────

static void hat_cab(const float * input, int C, int H, int W,
                    const float * conv1_w, const float * conv1_b,
                    const float * conv2_w, const float * conv2_b,
                    const float * ca_down_w, const float * ca_down_b,
                    const float * ca_up_w, const float * ca_up_b,
                    int compress_ratio, int squeeze_factor,
                    float * output) {
    int HW = H * W;
    int mid_c = C / compress_ratio;
    int sq_c = C / squeeze_factor;

    // Conv3(C→mid_c) + GELU
    std::vector<float> tmp1(mid_c * HW);
    hat_conv2d(input, C, H, W, conv1_w, conv1_b, mid_c, 3, 3, 1, tmp1.data());
    for (int i = 0; i < mid_c * HW; i++) tmp1[i] = hat_gelu(tmp1[i]);

    // Conv3(mid_c→C)
    std::vector<float> tmp2(C * HW);
    hat_conv2d(tmp1.data(), mid_c, H, W, conv2_w, conv2_b, C, 3, 3, 1, tmp2.data());

    // ChannelAttention: global avg pool → Conv1(C→sq_c) → ReLU → Conv1(sq_c→C) → Sigmoid
    std::vector<float> pooled(C);
    for (int c = 0; c < C; c++) {
        float sum = 0;
        for (int i = 0; i < HW; i++) sum += tmp2[c * HW + i];
        pooled[c] = sum / HW;
    }
    std::vector<float> ca_mid(sq_c);
    core_cpu::linear_cpu(pooled.data(), ca_mid.data(), C, sq_c, ca_down_w, ca_down_b);
    for (int o = 0; o < sq_c; o++) ca_mid[o] = ca_mid[o] > 0 ? ca_mid[o] : 0; // ReLU
    std::vector<float> ca_out(C);
    core_cpu::linear_cpu(ca_mid.data(), ca_out.data(), sq_c, C, ca_up_w, ca_up_b);
    for (int o = 0; o < C; o++) ca_out[o] = 1.0f / (1.0f + expf(-ca_out[o])); // Sigmoid
    // Multiply
    for (int c = 0; c < C; c++)
        for (int i = 0; i < HW; i++)
            output[c * HW + i] = tmp2[c * HW + i] * ca_out[c];
}

// ── OCAB (Overlapping Cross-Attention Block) ───────────────────────────

static void hat_ocab(float * x, int HW, int C, int H, int W, int ws, int n_heads,
                     int overlap_ws,
                     const float * qkv_w, const float * qkv_b,
                     const float * proj_w, const float * proj_b,
                     const float * rpb_table, const int * rpi,
                     const float * norm1_w, const float * norm1_b,
                     const float * norm2_w, const float * norm2_b,
                     const float * mlp_fc1_w, const float * mlp_fc1_b,
                     const float * mlp_fc2_w, const float * mlp_fc2_b,
                     int mlp_hidden) {
    int ws2 = ws * ws;
    int ows2 = overlap_ws * overlap_ws;
    int nH = H / ws, nW = W / ws;
    int nw = nH * nW;
    int d = C / n_heads;
    float scale = 1.0f / sqrtf((float)d);

    // Shortcut
    std::vector<float> shortcut(x, x + HW * C);

    // LN1
    std::vector<float> normed(HW * C);
    memcpy(normed.data(), x, HW * C * sizeof(float));
    hat_layernorm(normed.data(), HW, C, norm1_w, norm1_b);

    // QKV: [HW, 3C]
    std::vector<float> qkv(HW * 3 * C);
    hat_linear(normed.data(), HW, C, 3 * C, qkv_w, qkv_b, qkv.data());

    // Q from windows, KV from overlapping unfold
    // Q: partition normed into windows → [nw, ws2, C]
    // For simplicity, we'll use Q from window partition of the normed input
    // and KV from the full-image QKV with overlapping windows (unfold).
    // This is a simplified implementation — exact unfold matching requires
    // careful index computation.

    // Q windows: partition the Q portion
    // Reshape QKV to [H, W, 3, C] first
    std::vector<float> q_spatial(H * W * C);
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++)
            for (int c = 0; c < C; c++)
                q_spatial[(y * W + xi) * C + c] = qkv[(y * W + xi) * 3 * C + c];

    std::vector<float> q_windows(nw * ws2 * C);
    hat_window_partition(q_spatial.data(), H, W, C, ws, q_windows.data());

    // KV: unfold with overlap_ws kernel, stride ws, padding (overlap_ws-ws)/2
    // For each window, KV covers an overlap_ws × overlap_ws region centered on the window
    int pad = (overlap_ws - ws) / 2;
    std::vector<float> kv_spatial(H * W * 2 * C);
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++)
            for (int c = 0; c < 2 * C; c++)
                kv_spatial[(y * W + xi) * 2 * C + c] = qkv[(y * W + xi) * 3 * C + C + c];

    // For each window, extract the overlapping KV region
    std::vector<float> attn_out(nw * ws2 * C, 0.0f);

    // Pre-extract KV for each window's overlapping region (zero-pad out of bounds)
    // Layout: kv_windows[nw][ows2][2*C]
    std::vector<float> kv_win(nw * ows2 * 2 * C, 0.0f);
    for (int wh = 0; wh < nH; wh++) {
        for (int ww = 0; ww < nW; ww++) {
            int w_idx = wh * nW + ww;
            int ky0 = wh * ws - pad;
            int kx0 = ww * ws - pad;
            for (int k = 0; k < ows2; k++) {
                int ky = ky0 + k / overlap_ws;
                int kx = kx0 + k % overlap_ws;
                if (ky >= 0 && ky < H && kx >= 0 && kx < W) {
                    float * dst = &kv_win[(w_idx * ows2 + k) * 2 * C];
                    const float * src = &kv_spatial[(ky * W + kx) * 2 * C];
                    memcpy(dst, src, 2 * C * sizeof(float));
                }
                // else: remains zero (zero-padding)
            }
        }
    }

    // Attention per window per head
    std::vector<float> scores(ows2);
    for (int w = 0; w < nw; w++) {
        for (int h_idx = 0; h_idx < n_heads; h_idx++) {
            for (int q = 0; q < ws2; q++) {
                const float * q_row = &q_windows[(w * ws2 + q) * C + h_idx * d];
                float max_s = -1e9f;

                for (int k = 0; k < ows2; k++) {
                    const float * k_row = &kv_win[(w * ows2 + k) * 2 * C + h_idx * d];
                    float s = core_cpu::dot_product(q_row, k_row, d) * scale;
                    // RPI may contain negative indices (PyTorch wraps them)
                    int rpi_val = rpi[q * ows2 + k];
                    int rpb_len = (ws + overlap_ws - 1) * (ws + overlap_ws - 1);
                    if (rpi_val < 0) rpi_val += rpb_len;
                    s += rpb_table[rpi_val * n_heads + h_idx];
                    scores[k] = s;
                    if (s > max_s) max_s = s;
                }
                float sum_e = 0;
                for (int k = 0; k < ows2; k++) {
                    scores[k] = expf(scores[k] - max_s);
                    sum_e += scores[k];
                }
                float inv = 1.0f / sum_e;
                for (int k = 0; k < ows2; k++) scores[k] *= inv;

                for (int di = 0; di < d; di++) {
                    float val = 0;
                    for (int k = 0; k < ows2; k++)
                        val += scores[k] * kv_win[(w * ows2 + k) * 2 * C + C + h_idx * d + di];
                    attn_out[(w * ws2 + q) * C + h_idx * d + di] = val;
                }
            }
        }
    }

    // Window reverse
    std::vector<float> attn_full(HW * C);
    hat_window_reverse(attn_out.data(), H, W, C, ws, attn_full.data());

    // Proj + shortcut
    std::vector<float> proj(HW * C);
    hat_linear(attn_full.data(), HW, C, C, proj_w, proj_b, proj.data());
    for (int i = 0; i < HW * C; i++) x[i] = proj[i] + shortcut[i];

    // LN2 + MLP + shortcut
    std::vector<float> shortcut2(x, x + HW * C);
    hat_layernorm(x, HW, C, norm2_w, norm2_b);
    std::vector<float> mlp_mid(HW * mlp_hidden);
    hat_linear(x, HW, C, mlp_hidden, mlp_fc1_w, mlp_fc1_b, mlp_mid.data());
    for (int i = 0; i < HW * mlp_hidden; i++) mlp_mid[i] = hat_gelu(mlp_mid[i]);
    hat_linear(mlp_mid.data(), HW, mlp_hidden, C, mlp_fc2_w, mlp_fc2_b, x);
    for (int i = 0; i < HW * C; i++) x[i] += shortcut2[i];
}

// ── Context ────────────────────────────────────────────────────────────

struct hat_sr_context {
    int embed_dim, window_size, upscale, num_feat;
    int compress_ratio, squeeze_factor;
    float overlap_ratio, conv_scale;
    std::vector<int> depths, heads;
    int n_layers;
    int n_threads;
    bool bench;

    ggml_backend_t backend = nullptr;
    core_gguf::WeightLoad wl;
    core_cpu::DequantCache dcache;
    std::vector<std::vector<int>> i32_bufs;

    // Cached SW-MSA attention mask: reused when pH/pW repeat (common in tiling).
    int mask_cache_pH = 0, mask_cache_pW = 0;
    std::vector<float> mask_cache;

    const float * get(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) return nullptr;
        return dcache.get(t);
    }

    const int * get_i32(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t || t->type != GGML_TYPE_F32) return nullptr;
        // RPI stored as float but contains int indices — read via backend API
        int n = (int)ggml_nelements(t);
        i32_bufs.emplace_back(n);
        ggml_backend_tensor_get(t, i32_bufs.back().data(), 0, n * sizeof(int));
        return i32_bufs.back().data();
    }
};

hat_sr_context * hat_sr_init(const char * model_path, int n_threads) {
    auto * ctx = new hat_sr_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 1;
    ctx->conv_scale = 0.01f;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) { fprintf(stderr, "hat_sr: failed to open %s\n", model_path); delete ctx; return nullptr; }

    ctx->embed_dim      = core_gguf::kv_u32(meta, "hat.embed_dim", 180);
    ctx->window_size     = core_gguf::kv_u32(meta, "hat.window_size", 16);
    ctx->upscale         = core_gguf::kv_u32(meta, "hat.upscale", 4);
    ctx->num_feat        = core_gguf::kv_u32(meta, "hat.num_feat", 64);
    ctx->compress_ratio  = core_gguf::kv_u32(meta, "hat.compress_ratio", 3);
    ctx->squeeze_factor  = core_gguf::kv_u32(meta, "hat.squeeze_factor", 30);
    ctx->depths = core_gguf::kv_i32_array(meta, "hat.depths");
    ctx->heads  = core_gguf::kv_i32_array(meta, "hat.heads");
    ctx->n_layers = (int)ctx->depths.size();

    int idx = gguf_find_key(meta, "hat.overlap_ratio");
    ctx->overlap_ratio = idx >= 0 ? gguf_get_val_f32(meta, idx) : 0.5f;

    core_gguf::free_metadata(meta);

    bool force_cpu = (getenv("HAT_SR_FORCE_CPU") && atoi(getenv("HAT_SR_FORCE_CPU")));
    ctx->backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, ctx->backend, "hat", ctx->wl)) {
        fprintf(stderr, "hat_sr: failed to load weights\n");
        ggml_backend_free(ctx->backend); delete ctx; return nullptr;
    }

    fprintf(stderr, "hat_sr: embed=%d, ws=%d, upscale=%dx, layers=%d, %d tensors\n",
            ctx->embed_dim, ctx->window_size, ctx->upscale, ctx->n_layers,
            (int)ctx->wl.tensors.size());
    ctx->bench = (std::getenv("CRISPEMBED_HAT_SR_BENCH") != nullptr);
    return ctx;
}

void hat_sr_free(hat_sr_context * ctx) {
    if (!ctx) return;
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

int hat_sr_scale(const hat_sr_context * ctx) { return ctx ? ctx->upscale : 0; }

// ── HAB forward ────────────────────────────────────────────────────────

static void hat_hab_forward(hat_sr_context * ctx, float * x, int HW, int C,
                            int H, int W, const std::string & prefix,
                            int n_heads, int shift_size,
                            const int * rpi_sa, const float * attn_mask, int n_mask_windows) {
    int ws = ctx->window_size;
    int ws2 = ws * ws;
    int nw = (H / ws) * (W / ws);
    int d = C / n_heads;
    float scale = 1.0f / sqrtf((float)d);

    // Shortcut
    std::vector<float> shortcut(x, x + HW * C);

    // LN1
    hat_layernorm(x, HW, C, ctx->get(prefix + ".norm1.weight"),
                  ctx->get(prefix + ".norm1.bias"));

    // Conv_X (CAB) — operates on [C, H, W] layout
    std::vector<float> x_chw(C * HW);
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++)
            for (int c = 0; c < C; c++)
                x_chw[c * HW + y * W + xi] = x[(y * W + xi) * C + c];

    std::vector<float> conv_x_chw(C * HW);
    hat_cab(x_chw.data(), C, H, W,
            ctx->get(prefix + ".cab.conv1.weight"), ctx->get(prefix + ".cab.conv1.bias"),
            ctx->get(prefix + ".cab.conv2.weight"), ctx->get(prefix + ".cab.conv2.bias"),
            ctx->get(prefix + ".cab.ca_down.weight"), ctx->get(prefix + ".cab.ca_down.bias"),
            ctx->get(prefix + ".cab.ca_up.weight"), ctx->get(prefix + ".cab.ca_up.bias"),
            ctx->compress_ratio, ctx->squeeze_factor, conv_x_chw.data());

    // Convert CAB output back to [HW, C]
    std::vector<float> conv_x(HW * C);
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++)
            for (int c = 0; c < C; c++)
                conv_x[(y * W + xi) * C + c] = conv_x_chw[c * HW + y * W + xi];

    // Reshape x from [HW, C] to [H, W, C] for window ops
    // Cyclic shift if needed
    std::vector<float> shifted(HW * C);
    if (shift_size > 0)
        hat_cyclic_shift(x, H, W, C, -shift_size, shifted.data());
    else
        memcpy(shifted.data(), x, HW * C * sizeof(float));

    // Window partition
    std::vector<float> x_windows(nw * ws2 * C);
    hat_window_partition(shifted.data(), H, W, C, ws, x_windows.data());

    // Window attention
    std::vector<float> attn_windows(nw * ws2 * C);
    hat_window_attn(x_windows.data(), nw, ws2, C, n_heads,
                    ctx->get(prefix + ".attn.qkv.weight"),
                    ctx->get(prefix + ".attn.qkv.bias"),
                    ctx->get(prefix + ".attn.proj.weight"),
                    ctx->get(prefix + ".attn.proj.bias"),
                    ctx->get(prefix + ".attn.rpb"), rpi_sa,
                    shift_size > 0 ? attn_mask : nullptr,
                    shift_size > 0 ? n_mask_windows : 0,
                    scale, attn_windows.data());

    // Window reverse
    std::vector<float> attn_x(HW * C);
    hat_window_reverse(attn_windows.data(), H, W, C, ws, attn_x.data());

    // Reverse cyclic shift
    if (shift_size > 0) {
        std::vector<float> tmp(HW * C);
        hat_cyclic_shift(attn_x.data(), H, W, C, shift_size, tmp.data());
        attn_x = std::move(tmp);
    }

    // x = shortcut + attn_x + conv_scale * conv_x
    float cs = ctx->conv_scale;
    for (int i = 0; i < HW * C; i++)
        x[i] = shortcut[i] + attn_x[i] + cs * conv_x[i];

    // LN2 + MLP
    std::vector<float> shortcut2(x, x + HW * C);
    hat_layernorm(x, HW, C, ctx->get(prefix + ".norm2.weight"),
                  ctx->get(prefix + ".norm2.bias"));
    int mlp_hidden = (int)(C * 4);  // mlp_ratio = 4 by default...
    // Actually check from weight shape
    auto * fc1_t = core_gguf::try_get(ctx->wl.tensors, (prefix + ".mlp.fc1.weight").c_str());
    if (fc1_t) mlp_hidden = fc1_t->ne[0];

    std::vector<float> mlp_mid(HW * mlp_hidden);
    hat_linear(x, HW, C, mlp_hidden, ctx->get(prefix + ".mlp.fc1.weight"),
               ctx->get(prefix + ".mlp.fc1.bias"), mlp_mid.data());
    for (int i = 0; i < HW * mlp_hidden; i++) mlp_mid[i] = hat_gelu(mlp_mid[i]);
    hat_linear(mlp_mid.data(), HW, mlp_hidden, C, ctx->get(prefix + ".mlp.fc2.weight"),
               ctx->get(prefix + ".mlp.fc2.bias"), x);
    for (int i = 0; i < HW * C; i++) x[i] += shortcut2[i];
}

// ── Forward pass (single tile) ─────────────────────────────────────────

static void hat_forward_tile(hat_sr_context * ctx,
                             const float * input, int W, int H,
                             float * output) {
    int C = ctx->embed_dim;
    int ws = ctx->window_size;
    int scale = ctx->upscale;

    // Pad to multiple of window_size
    int pH = ((H + ws - 1) / ws) * ws;
    int pW = ((W + ws - 1) / ws) * ws;

    // Mean subtraction
    float mean_r = 0.4488f, mean_g = 0.4371f, mean_b = 0.4040f;
    std::vector<float> img(3 * pH * pW, 0.0f);
    for (int y = 0; y < pH; y++)
        for (int x = 0; x < pW; x++) {
            int sy = std::min(y, H - 1), sx = std::min(x, W - 1);
            img[0 * pH * pW + y * pW + x] = (input[0 * H * W + sy * W + sx] - mean_r);
            img[1 * pH * pW + y * pW + x] = (input[1 * H * W + sy * W + sx] - mean_g);
            img[2 * pH * pW + y * pW + x] = (input[2 * H * W + sy * W + sx] - mean_b);
        }

    // conv_first
    std::vector<float> shallow(C * pH * pW);
    hat_conv2d(img.data(), 3, pH, pW, ctx->get("conv_first.weight"),
               ctx->get("conv_first.bias"), C, 3, 3, 1, shallow.data());

    // Convert to [HW, C] sequence format for transformer (patch_embed)
    int HW = pH * pW;
    std::vector<float> x(HW * C);
    for (int y = 0; y < pH; y++)
        for (int xi = 0; xi < pW; xi++)
            for (int c = 0; c < C; c++)
                x[(y * pW + xi) * C + c] = shallow[c * pH * pW + y * pW + xi];

    // Patch embed norm (when patch_norm=True)
    const float * pe_norm_w = ctx->get("patch_embed.norm.weight");
    const float * pe_norm_b = ctx->get("patch_embed.norm.bias");
    if (pe_norm_w && pe_norm_b)
        hat_layernorm(x.data(), HW, C, pe_norm_w, pe_norm_b);

    // Build SW-MSA attention mask (cache by pH×pW — reused across tiles of same size)
    int nH = pH / ws, nW = pW / ws;
    int n_mask_windows = nH * nW;
    int shift_size = ws / 2;
    int ws2 = ws * ws;

    if (ctx->mask_cache_pH != pH || ctx->mask_cache_pW != pW) {
        ctx->mask_cache.assign(n_mask_windows * ws2 * ws2, 0.0f);
        std::vector<int> img_mask(pH * pW, 0);
        int cnt = 0;
        int h_slices[] = {0, pH - ws, pH - shift_size, pH};
        int w_slices[] = {0, pW - ws, pW - shift_size, pW};
        for (int hi = 0; hi < 3; hi++)
            for (int wi = 0; wi < 3; wi++) {
                for (int y = h_slices[hi]; y < h_slices[hi + 1]; y++)
                    for (int xi = w_slices[wi]; xi < w_slices[wi + 1]; xi++)
                        img_mask[y * pW + xi] = cnt;
                cnt++;
            }
        std::vector<int> mask_windows(n_mask_windows * ws2);
        for (int wh = 0; wh < nH; wh++)
            for (int ww = 0; ww < nW; ww++)
                for (int y = 0; y < ws; y++)
                    for (int xi = 0; xi < ws; xi++)
                        mask_windows[(wh * nW + ww) * ws2 + y * ws + xi] =
                            img_mask[(wh * ws + y) * pW + (ww * ws + xi)];
        for (int w = 0; w < n_mask_windows; w++)
            for (int i = 0; i < ws2; i++)
                for (int j = 0; j < ws2; j++) {
                    int diff = mask_windows[w * ws2 + i] - mask_windows[w * ws2 + j];
                    ctx->mask_cache[w * ws2 * ws2 + i * ws2 + j] = (diff != 0) ? -100.0f : 0.0f;
                }
        ctx->mask_cache_pH = pH;
        ctx->mask_cache_pW = pW;
    }
    const std::vector<float> & attn_mask = ctx->mask_cache;

    // Get relative position indices
    const float * rpi_sa_f = ctx->get("rpi_sa");
    const float * rpi_oca_f = ctx->get("rpi_oca");
    // These are stored as float but are int indices — reinterpret
    // Actually they were stored from PyTorch int64 tensors as float — need to convert
    int rpi_sa_n = ws2 * ws2;
    std::vector<int> rpi_sa(rpi_sa_n);
    if (rpi_sa_f) for (int i = 0; i < rpi_sa_n; i++) rpi_sa[i] = (int)rpi_sa_f[i];

    int overlap_ws = (int)(ws * ctx->overlap_ratio) + ws;
    int ows2 = overlap_ws * overlap_ws;
    int rpi_oca_n = ws2 * ows2;
    std::vector<int> rpi_oca(rpi_oca_n);
    if (rpi_oca_f) for (int i = 0; i < rpi_oca_n; i++) rpi_oca[i] = (int)rpi_oca_f[i];

    fprintf(stderr, "hat_sr: conv_first done, starting RHAG layers\n");

    // RHAG layers
    for (int li = 0; li < ctx->n_layers; li++) {
        std::vector<float> residual = x;
        char buf[64];
        fprintf(stderr, "hat_sr: RHAG layer %d/%d\n", li, ctx->n_layers);

        // HAB blocks
        for (int bi = 0; bi < ctx->depths[li]; bi++) {
            snprintf(buf, sizeof(buf), "layer.%d.hab.%d", li, bi);
            fprintf(stderr, "  HAB %d/%d %s\n", bi, ctx->depths[li], buf);
            int shift = (bi % 2 == 0) ? 0 : shift_size;
            hat_hab_forward(ctx, x.data(), HW, C, pH, pW, buf,
                           ctx->heads[li], shift,
                           rpi_sa.data(), attn_mask.data(), n_mask_windows);
        }

        // OCAB
        snprintf(buf, sizeof(buf), "layer.%d.ocab", li);
        int mlp_hidden_ocab = C * 2;  // mlp_ratio=2 for OCAB
        auto * fc1_t = core_gguf::try_get(ctx->wl.tensors,
            (std::string(buf) + ".mlp.fc1.weight").c_str());
        if (fc1_t) mlp_hidden_ocab = fc1_t->ne[0];

        hat_ocab(x.data(), HW, C, pH, pW, ws, ctx->heads[li], overlap_ws,
                 ctx->get(std::string(buf) + ".qkv.weight"),
                 ctx->get(std::string(buf) + ".qkv.bias"),
                 ctx->get(std::string(buf) + ".proj.weight"),
                 ctx->get(std::string(buf) + ".proj.bias"),
                 ctx->get(std::string(buf) + ".rpb"),
                 rpi_oca.data(),
                 ctx->get(std::string(buf) + ".norm1.weight"),
                 ctx->get(std::string(buf) + ".norm1.bias"),
                 ctx->get(std::string(buf) + ".norm2.weight"),
                 ctx->get(std::string(buf) + ".norm2.bias"),
                 ctx->get(std::string(buf) + ".mlp.fc1.weight"),
                 ctx->get(std::string(buf) + ".mlp.fc1.bias"),
                 ctx->get(std::string(buf) + ".mlp.fc2.weight"),
                 ctx->get(std::string(buf) + ".mlp.fc2.bias"),
                 mlp_hidden_ocab);

        // RHAG: patch_unembed → conv → patch_embed → + residual
        // patch_unembed: [HW, C] → [C, H, W]
        std::vector<float> x_chw(C * pH * pW);
        for (int y = 0; y < pH; y++)
            for (int xi = 0; xi < pW; xi++)
                for (int c = 0; c < C; c++)
                    x_chw[c * pH * pW + y * pW + xi] = x[(y * pW + xi) * C + c];

        // Conv3(C→C)
        snprintf(buf, sizeof(buf), "layer.%d", li);
        std::vector<float> conv_out(C * pH * pW);
        hat_conv2d(x_chw.data(), C, pH, pW,
                   ctx->get(std::string(buf) + ".conv.weight"),
                   ctx->get(std::string(buf) + ".conv.bias"),
                   C, 3, 3, 1, conv_out.data());

        // patch_embed: [C, H, W] → [HW, C]
        for (int y = 0; y < pH; y++)
            for (int xi = 0; xi < pW; xi++)
                for (int c = 0; c < C; c++)
                    x[(y * pW + xi) * C + c] = conv_out[c * pH * pW + y * pW + xi] + residual[(y * pW + xi) * C + c];
    }

    // Final norm
    hat_layernorm(x.data(), HW, C, ctx->get("norm.weight"), ctx->get("norm.bias"));

    // patch_unembed → conv_after_body → + shallow
    std::vector<float> deep_chw(C * pH * pW);
    for (int y = 0; y < pH; y++)
        for (int xi = 0; xi < pW; xi++)
            for (int c = 0; c < C; c++)
                deep_chw[c * pH * pW + y * pW + xi] = x[(y * pW + xi) * C + c];

    std::vector<float> after_body(C * pH * pW);
    hat_conv2d(deep_chw.data(), C, pH, pW,
               ctx->get("conv_after_body.weight"), ctx->get("conv_after_body.bias"),
               C, 3, 3, 1, after_body.data());
    for (int i = 0; i < C * pH * pW; i++) after_body[i] += shallow[i];

    // Reconstruction: conv_before_upsample (C→num_feat) + LeakyReLU + upsample + conv_last
    int nf = ctx->num_feat;
    std::vector<float> pre_up(nf * pH * pW);
    hat_conv2d(after_body.data(), C, pH, pW,
               ctx->get("conv_before_upsample.weight"),
               ctx->get("conv_before_upsample.bias"),
               nf, 3, 3, 1, pre_up.data());
    for (int i = 0; i < nf * pH * pW; i++)
        pre_up[i] = pre_up[i] > 0 ? pre_up[i] : pre_up[i] * 0.01f;  // LeakyReLU (negative_slope=0.01)

    // PixelShuffle upsample (2x per step)
    int cur_h = pH, cur_w = pW, cur_c = nf;
    std::vector<float> cur = pre_up;
    int n_ups = (int)(log2f((float)scale) + 0.5f);
    for (int i = 0; i < n_ups; i++) {
        int up_oc = 4 * cur_c;
        std::vector<float> up_conv(up_oc * cur_h * cur_w);
        char buf2[32]; snprintf(buf2, sizeof(buf2), "upsample.%d", i);
        hat_conv2d(cur.data(), cur_c, cur_h, cur_w,
                   ctx->get(std::string(buf2) + ".weight"),
                   ctx->get(std::string(buf2) + ".bias"),
                   up_oc, 3, 3, 1, up_conv.data());
        int nh = cur_h * 2, nw2 = cur_w * 2;
        std::vector<float> ps_out(cur_c * nh * nw2);
        hat_pixel_shuffle(up_conv.data(), up_oc, cur_h, cur_w, 2, ps_out.data());
        cur = std::move(ps_out);
        cur_h = nh; cur_w = nw2;
    }

    // conv_last
    int oH = H * scale, oW = W * scale;
    std::vector<float> final_out(3 * cur_h * cur_w);
    hat_conv2d(cur.data(), nf, cur_h, cur_w,
               ctx->get("conv_last.weight"), ctx->get("conv_last.bias"),
               3, 3, 3, 1, final_out.data());

    // Undo mean subtraction, crop to original size * scale
    for (int c = 0; c < 3; c++) {
        float m = (c == 0) ? mean_r : (c == 1) ? mean_g : mean_b;
        for (int y = 0; y < oH; y++)
            for (int xi = 0; xi < oW; xi++)
                output[c * oH * oW + y * oW + xi] =
                    final_out[c * cur_h * cur_w + y * cur_w + xi] + m;
    }
}

// ── Tiled processing ──────────────────────────────────────────────────

int hat_sr_process(hat_sr_context * ctx,
                   const uint8_t * input, int width, int height,
                   int tile_size, int tile_overlap,
                   uint8_t ** output, int * out_width, int * out_height) {
    if (!ctx || !input || !output || width <= 0 || height <= 0) return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    int scale = ctx->upscale;
    if (tile_size <= 0) tile_size = 64;
    if (tile_overlap <= 0) tile_overlap = 8;
    tile_overlap = std::min(tile_overlap, tile_size / 4);

    int ow = width * scale, oh = height * scale;
    int out_tile = tile_size * scale;
    int out_overlap = tile_overlap * scale;

    auto t_pre = std::chrono::steady_clock::now();
    std::vector<float> full(3 * height * width);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                full[c * height * width + y * width + x] =
                    input[(y * width + x) * 3 + c] / 255.0f;
    if (bench) {
        auto t_pre_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[hat_sr-bench] preprocess: %.1f ms\n",
                ms_f(t_pre_end - t_pre).count());
    }

    std::vector<float> accum(3 * oh * ow, 0.0f);
    std::vector<float> wmap(oh * ow, 0.0f);

    int step = tile_size - tile_overlap;
    int ntx = std::max(1, (width + step - 1) / step);
    int nty = std::max(1, (height + step - 1) / step);

    fprintf(stderr, "hat_sr: %dx%d → %dx%d (%dx), tiles=%dx%d\n",
            width, height, ow, oh, scale, ntx, nty);

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

            int otw = tw * scale, oth = th * scale;
            std::vector<float> tile_out(3 * oth * otw);
            auto t_tile = std::chrono::steady_clock::now();
            hat_forward_tile(ctx, tile_in.data(), tw, th, tile_out.data());
            if (bench) {
                auto t_tile_end = std::chrono::steady_clock::now();
                fprintf(stderr, "[hat_sr-bench] tile %d,%d: %.1f ms\n",
                        ty, tx, ms_f(t_tile_end - t_tile).count());
            }

            // Blend
            int ox0 = x0 * scale, oy0 = y0 * scale;
            for (int y = 0; y < oth; y++) {
                float wy = 1.0f;
                if (y0 > 0 && y < out_overlap)
                    wy = 0.5f - 0.5f * cosf((float)M_PI * y / out_overlap);
                if (y0 + th < height && y >= oth - out_overlap)
                    wy = 0.5f - 0.5f * cosf((float)M_PI * (oth - 1 - y) / out_overlap);
                for (int x = 0; x < otw; x++) {
                    float wx = 1.0f;
                    if (x0 > 0 && x < out_overlap)
                        wx = 0.5f - 0.5f * cosf((float)M_PI * x / out_overlap);
                    if (x0 + tw < width && x >= otw - out_overlap)
                        wx = 0.5f - 0.5f * cosf((float)M_PI * (otw - 1 - x) / out_overlap);
                    float w = wy * wx;
                    int dy = oy0 + y, dx = ox0 + x;
                    if (dy >= oh || dx >= ow) continue;
                    for (int c = 0; c < 3; c++)
                        accum[c * oh * ow + dy * ow + dx] += tile_out[c * oth * otw + y * otw + x] * w;
                    wmap[dy * ow + dx] += w;
                }
            }
        }
    }

    // Convert to uint8
    auto t_post = std::chrono::steady_clock::now();
    uint8_t * out_buf = (uint8_t *)malloc(3 * oh * ow);
    if (!out_buf) return -1;
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++) {
            float w = wmap[y * ow + x];
            if (w <= 0) w = 1.0f;
            for (int c = 0; c < 3; c++) {
                float v = accum[c * oh * ow + y * ow + x] / w * 255.0f;
                out_buf[(y * ow + x) * 3 + c] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
        }

    *output = out_buf;
    *out_width = ow;
    *out_height = oh;
    fprintf(stderr, "hat_sr: done %dx%d\n", ow, oh);
    if (bench) {
        auto t_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[hat_sr-bench] postprocess: %.1f ms\n",
                ms_f(t_end - t_post).count());
        fprintf(stderr, "[hat_sr-bench] total: %.1f ms\n",
                ms_f(t_end - t_total).count());
    }
    return 0;
}

void hat_sr_free_image(uint8_t * pixels) { free(pixels); }
