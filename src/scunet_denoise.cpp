// scunet_denoise.cpp — SCUNet Swin-Conv-UNet denoising (CPU-scalar).
//
// U-Net structure:
//   Head: Conv3x3(3→64)
//   Encoder: 3 stages × [4 ConvTransBlocks + stride-2 downsample]
//   Body: 4 ConvTransBlocks at 512ch
//   Decoder: 3 stages × [ConvTranspose2d upsample + 4 ConvTransBlocks]
//   Tail: Conv3x3(64→3)
//   Skip connections: element-wise add before each decoder upsample
//
// ConvTransBlock: split channels → conv branch (3x3 residual) +
//   trans branch (Swin window attention + MLP) → fuse via 1x1 conv.

#include "scunet_denoise.h"
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
                   int oc, int kh, int kw, int pad, int stride,
                   float * out) {
    int oh = (h + 2 * pad - kh) / stride + 1;
    int ow = (w + 2 * pad - kw) / stride + 1;
    for (int o = 0; o < oc; o++) {
        float b = bi ? bi[o] : 0.0f;
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                float sum = b;
                for (int c = 0; c < ic; c++)
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy * stride + ky - pad;
                            int ix = ox * stride + kx - pad;
                            if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                sum += in[c * h * w + iy * w + ix]
                                     * wt[o * ic * kh * kw + c * kh * kw + ky * kw + kx];
                        }
                out[o * oh * ow + oy * ow + ox] = sum;
            }
        }
    }
}

static void conv_transpose2d(const float * in, int ic, int h, int w,
                              const float * wt, const float * bi,
                              int oc, int kh, int kw, int stride,
                              float * out) {
    int oh = (h - 1) * stride + kh;
    int ow = (w - 1) * stride + kw;
    memset(out, 0, oc * oh * ow * sizeof(float));
    // Scatter-add
    for (int c = 0; c < ic; c++)
        for (int iy = 0; iy < h; iy++)
            for (int ix = 0; ix < w; ix++) {
                float v = in[c * h * w + iy * w + ix];
                for (int o = 0; o < oc; o++)
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++) {
                            int oy = iy * stride + ky;
                            int ox = ix * stride + kx;
                            out[o * oh * ow + oy * ow + ox] +=
                                v * wt[c * oc * kh * kw + o * kh * kw + ky * kw + kx];
                        }
            }
    if (bi)
        for (int o = 0; o < oc; o++)
            for (int i = 0; i < oh * ow; i++)
                out[o * oh * ow + i] += bi[o];
}

static void layer_norm(const float * in, int n, const float * w, const float * b,
                       float * out) {
    float mean = 0;
    for (int i = 0; i < n; i++) mean += in[i];
    mean /= n;
    float var = 0;
    for (int i = 0; i < n; i++) { float d = in[i] - mean; var += d * d; }
    var /= n;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++)
        out[i] = (in[i] - mean) * inv * w[i] + b[i];
}

static void gelu_inplace(float * d, int n) {
    for (int i = 0; i < n; i++)
        d[i] = d[i] * 0.5f * (1.0f + erff(d[i] * 0.7071067811865476f));
}

// ── Swin Window Multi-head Self-Attention ──

static void wmsa_forward(
    const float * x_chw, int C, int H, int W,
    const float * qkv_w, const float * qkv_b,
    const float * proj_w, const float * proj_b,
    const float * rpb, // [1, 2*win-1, 2*win-1]
    int n_heads, int win_size, bool shift,
    float * out_chw,
    std::vector<float> & scratch)
{
    int head_dim = C / n_heads;
    int N = win_size * win_size;
    int rpb_side = 2 * win_size - 1;

    // Pad H, W to multiples of win_size
    int Hp = ((H + win_size - 1) / win_size) * win_size;
    int Wp = ((W + win_size - 1) / win_size) * win_size;
    int nWh = Hp / win_size, nWw = Wp / win_size;
    int nW = nWh * nWw;

    // Pad input (zero-pad, matching F.pad in Python)
    std::vector<float> padded(C * Hp * Wp, 0.0f);
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                padded[c * Hp * Wp + y * Wp + x] = x_chw[c * H * W + y * W + x];

    // Cyclic shift: torch.roll(shifts=(-sh, -sh), dims=(2,3))
    // shifted[y,x] = original[(y+sh)%H, (x+sh)%W]
    std::vector<float> shifted;
    if (shift) {
        shifted.resize(C * Hp * Wp);
        int sh = win_size / 2;
        for (int c = 0; c < C; c++)
            for (int y = 0; y < Hp; y++)
                for (int x = 0; x < Wp; x++) {
                    int sy = (y + sh) % Hp, sx = (x + sh) % Wp;
                    shifted[c * Hp * Wp + y * Wp + x] = padded[c * Hp * Wp + sy * Wp + sx];
                }
        std::swap(padded, shifted);
    }

    // Window partition → (nW, N, C)  in HWC order
    std::vector<float> windows(nW * N * C);
    for (int wh = 0; wh < nWh; wh++)
        for (int ww = 0; ww < nWw; ww++) {
            int wi = wh * nWw + ww;
            for (int py = 0; py < win_size; py++)
                for (int px = 0; px < win_size; px++) {
                    int gy = wh * win_size + py, gx = ww * win_size + px;
                    int ni = py * win_size + px;
                    for (int c = 0; c < C; c++)
                        windows[wi * N * C + ni * C + c] =
                            padded[c * Hp * Wp + gy * Wp + gx];
                }
        }

    // QKV projection: (nW, N, C) → (nW, N, 3C)
    std::vector<float> qkv(nW * N * 3 * C);
    for (int wi = 0; wi < nW; wi++)
        for (int ni = 0; ni < N; ni++) {
            const float * tok = &windows[wi * N * C + ni * C];
            float * dst = &qkv[wi * N * 3 * C + ni * 3 * C];
            for (int o = 0; o < 3 * C; o++) {
                float sum = qkv_b[o];
                for (int i = 0; i < C; i++)
                    sum += qkv_w[o * C + i] * tok[i];
                dst[o] = sum;
            }
        }

    // Build relative position bias index
    std::vector<int> rpb_idx(N * N);
    for (int i = 0; i < win_size; i++)
        for (int j = 0; j < win_size; j++)
            for (int k = 0; k < win_size; k++)
                for (int l = 0; l < win_size; l++) {
                    int n1 = i * win_size + j, n2 = k * win_size + l;
                    int dy = i - k + win_size - 1, dx = j - l + win_size - 1;
                    rpb_idx[n1 * N + n2] = dy * rpb_side + dx;
                }

    // Build shift mask
    std::vector<float> attn_mask;
    if (shift) {
        attn_mask.resize(nW * N * N, 0.0f);
        std::vector<int> mask_area(Hp * Wp, 0);
        int sh = win_size / 2;
        int cnt = 0;
        int h_cuts[] = {0, Hp - win_size, Hp - sh, Hp};
        int w_cuts[] = {0, Wp - win_size, Wp - sh, Wp};
        for (int hi = 0; hi < 3; hi++)
            for (int wi_idx = 0; wi_idx < 3; wi_idx++) {
                for (int y = h_cuts[hi]; y < h_cuts[hi + 1]; y++)
                    for (int x = w_cuts[wi_idx]; x < w_cuts[wi_idx + 1]; x++)
                        mask_area[y * Wp + x] = cnt;
                cnt++;
            }
        for (int wh = 0; wh < nWh; wh++)
            for (int ww = 0; ww < nWw; ww++) {
                int wi = wh * nWw + ww;
                for (int n1 = 0; n1 < N; n1++)
                    for (int n2 = 0; n2 < N; n2++) {
                        int gy1 = wh * win_size + n1 / win_size;
                        int gx1 = ww * win_size + n1 % win_size;
                        int gy2 = wh * win_size + n2 / win_size;
                        int gx2 = ww * win_size + n2 % win_size;
                        if (mask_area[gy1 * Wp + gx1] != mask_area[gy2 * Wp + gx2])
                            attn_mask[wi * N * N + n1 * N + n2] = -100.0f;
                    }
            }
    }

    // Attention per window per head
    float scale = 1.0f / sqrtf((float)head_dim);
    std::vector<float> attn_out(nW * N * C);

    for (int wi = 0; wi < nW; wi++) {
        for (int hd = 0; hd < n_heads; hd++) {
            // Compute attention scores
            scratch.resize(N * N);
            for (int i = 0; i < N; i++) {
                const float * qi = &qkv[(wi * N + i) * 3 * C + hd * head_dim];
                for (int j = 0; j < N; j++) {
                    const float * kj = &qkv[(wi * N + j) * 3 * C + C + hd * head_dim];
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                    float bias = rpb[rpb_idx[i * N + j]];
                    float mask_val = shift ? attn_mask[wi * N * N + i * N + j] : 0.0f;
                    scratch[i * N + j] = dot * scale + bias + mask_val;
                }
            }
            // Softmax per row
            for (int i = 0; i < N; i++) {
                float mx = -1e30f;
                for (int j = 0; j < N; j++) mx = std::max(mx, scratch[i * N + j]);
                float sum = 0;
                for (int j = 0; j < N; j++) {
                    scratch[i * N + j] = expf(scratch[i * N + j] - mx);
                    sum += scratch[i * N + j];
                }
                for (int j = 0; j < N; j++) scratch[i * N + j] /= sum;
            }
            // Attn × V
            for (int i = 0; i < N; i++) {
                float * dst = &attn_out[(wi * N + i) * C + hd * head_dim];
                for (int d = 0; d < head_dim; d++) dst[d] = 0;
                for (int j = 0; j < N; j++) {
                    float a = scratch[i * N + j];
                    const float * vj = &qkv[(wi * N + j) * 3 * C + 2 * C + hd * head_dim];
                    for (int d = 0; d < head_dim; d++) dst[d] += a * vj[d];
                }
            }
        }
    }

    // Output projection: (nW, N, C) → (nW, N, C)
    std::vector<float> proj_out(nW * N * C);
    for (int wi = 0; wi < nW; wi++)
        for (int ni = 0; ni < N; ni++) {
            const float * tok = &attn_out[wi * N * C + ni * C];
            float * dst = &proj_out[wi * N * C + ni * C];
            for (int o = 0; o < C; o++) {
                float sum = proj_b[o];
                for (int i = 0; i < C; i++)
                    sum += proj_w[o * C + i] * tok[i];
                dst[o] = sum;
            }
        }

    // Window reverse → CHW
    std::vector<float> rev(C * Hp * Wp, 0.0f);
    for (int wh = 0; wh < nWh; wh++)
        for (int ww = 0; ww < nWw; ww++) {
            int wi = wh * nWw + ww;
            for (int py = 0; py < win_size; py++)
                for (int px = 0; px < win_size; px++) {
                    int ni = py * win_size + px;
                    int gy = wh * win_size + py, gx = ww * win_size + px;
                    for (int c = 0; c < C; c++)
                        rev[c * Hp * Wp + gy * Wp + gx] =
                            proj_out[wi * N * C + ni * C + c];
                }
        }

    // Reverse shift: torch.roll(shifts=(sh, sh)) undoes roll(shifts=(-sh, -sh))
    // unshifted[y,x] = shifted[(y-sh+H)%H, (x-sh+W)%W]
    if (shift) {
        shifted.resize(C * Hp * Wp);
        int sh = win_size / 2;
        for (int c = 0; c < C; c++)
            for (int y = 0; y < Hp; y++)
                for (int x = 0; x < Wp; x++) {
                    int sy = (y + Hp - sh) % Hp, sx = (x + Wp - sh) % Wp;
                    shifted[c * Hp * Wp + y * Wp + x] = rev[c * Hp * Wp + sy * Wp + sx];
                }
        std::swap(rev, shifted);
    }

    // Crop to original size
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                out_chw[c * H * W + y * W + x] = rev[c * Hp * Wp + y * Wp + x];
}

// ── Swin Block: LN → WMSA → residual → LN → MLP → residual ──

struct swin_block_wt {
    ggml_tensor * ln1_w, * ln1_b;
    ggml_tensor * qkv_w, * qkv_b;
    ggml_tensor * proj_w, * proj_b;
    ggml_tensor * rpb;
    ggml_tensor * ln2_w, * ln2_b;
    ggml_tensor * mlp0_w, * mlp0_b;
    ggml_tensor * mlp2_w, * mlp2_b;
};

// Global debug callback for swin block internals (set by debug forward pass)
static scunet_stage_cb g_swin_debug_cb;
static int g_swin_debug_block_id = -1;
static int g_swin_block_counter = 0;

static void swin_block_forward(
    float * x, int C, int H, int W,
    const swin_block_wt & wt,
    int n_heads, int win_size, bool shift,
    std::vector<float> & dq1, std::vector<float> & dq2,
    std::vector<float> & scratch)
{
    int hw = H * W;
    int cur_block = g_swin_block_counter++;

    // LN1 + WMSA
    std::vector<float> normed(C * hw);
    // Cache the weight pointers before the per-pixel loop
    const float * ln1_w_ptr = to_f32(wt.ln1_w, dq1);
    const float * ln1_b_ptr = to_f32(wt.ln1_b, dq2);
    // Pre-allocate pixel buffers outside the loop (was per-pixel before)
    std::vector<float> pix(C), pix_out(C);
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++) {
            // Gather pixel's channels
            for (int c = 0; c < C; c++) pix[c] = x[c * hw + y * W + xi];
            layer_norm(pix.data(), C, ln1_w_ptr, ln1_b_ptr, pix_out.data());
            for (int c = 0; c < C; c++) normed[c * hw + y * W + xi] = pix_out[c];
        }

    if (g_swin_debug_cb && cur_block == g_swin_debug_block_id)
        g_swin_debug_cb("ln1_chw", normed.data(), C * hw);

    // Cache all weight pointers before passing to wmsa
    const float * qkv_w_ptr = to_f32(wt.qkv_w, dq1);
    const float * qkv_b_ptr = to_f32(wt.qkv_b, dq2);
    const float * proj_w_ptr = to_f32(wt.proj_w, dq1);
    const float * proj_b_ptr = to_f32(wt.proj_b, dq2);
    const float * rpb_ptr = to_f32(wt.rpb, dq1);

    std::vector<float> attn_out(C * hw);
    wmsa_forward(normed.data(), C, H, W,
                 qkv_w_ptr, qkv_b_ptr,
                 proj_w_ptr, proj_b_ptr,
                 rpb_ptr,
                 n_heads, win_size, shift,
                 attn_out.data(), scratch);

    if (g_swin_debug_cb && cur_block == g_swin_debug_block_id)
        g_swin_debug_cb("wmsa_out", attn_out.data(), C * hw);

    for (int i = 0; i < C * hw; i++) x[i] += attn_out[i];

    if (g_swin_debug_cb && cur_block == g_swin_debug_block_id)
        g_swin_debug_cb("after_wmsa", x, C * hw);

    // LN2 + MLP
    const float * m0w = to_f32(wt.mlp0_w, dq1);
    const float * m0b = to_f32(wt.mlp0_b, dq2);
    int mlp_hidden = (int)wt.mlp0_b->ne[0];
    const float * m2w = to_f32(wt.mlp2_w, dq1);
    const float * m2b = to_f32(wt.mlp2_b, dq2);
    // Cache LN2 weights outside the per-pixel loop (was re-dequantized per pixel!)
    const float * ln2_w_ptr = to_f32(wt.ln2_w, dq1);
    const float * ln2_b_ptr = to_f32(wt.ln2_b, dq2);

    // Pre-allocate buffers outside the pixel loop (was per-pixel before —
    // 65536+ heap allocs per swin block for a 256x256 image)
    std::vector<float> pix_norm(C);
    std::vector<float> h(mlp_hidden);
    for (int y = 0; y < H; y++)
        for (int xi = 0; xi < W; xi++) {
            for (int c = 0; c < C; c++) pix[c] = x[c * hw + y * W + xi];
            layer_norm(pix.data(), C, ln2_w_ptr, ln2_b_ptr, pix_norm.data());

            // MLP: Linear(C→hidden) + GELU + Linear(hidden→C)
            for (int o = 0; o < mlp_hidden; o++) {
                float sum = m0b[o];
                for (int i = 0; i < C; i++) sum += m0w[o * C + i] * pix_norm[i];
                h[o] = sum;
            }
            gelu_inplace(h.data(), mlp_hidden);
            for (int o = 0; o < C; o++) {
                float sum = m2b[o];
                for (int i = 0; i < mlp_hidden; i++) sum += m2w[o * mlp_hidden + i] * h[i];
                x[o * hw + y * W + xi] += sum;
            }
        }

    if (g_swin_debug_cb && cur_block == g_swin_debug_block_id)
        g_swin_debug_cb("full_swin", x, C * hw);
}

// ── ConvTransBlock ──

struct ctb_weights {
    ggml_tensor * conv1_1_w, * conv1_1_b; // 1x1 split
    ggml_tensor * conv_blk_0, * conv_blk_2; // 3x3 residual conv (no bias)
    swin_block_wt trans;
    ggml_tensor * conv1_2_w, * conv1_2_b; // 1x1 fuse
};

static void ctb_forward(float * x, int C, int H, int W,
                         const ctb_weights & wt,
                         int n_heads, int win_size, bool shift,
                         std::vector<float> & dq1, std::vector<float> & dq2,
                         std::vector<float> & scratch) {
    int hw = H * W;
    int half = C / 2;

    // 1x1 conv split
    std::vector<float> split(C * hw);
    conv2d(x, C, H, W, to_f32(wt.conv1_1_w, dq1), to_f32(wt.conv1_1_b, dq2),
           C, 1, 1, 0, 1, split.data());

    // Conv branch (first half)
    std::vector<float> conv_in(half * hw);
    memcpy(conv_in.data(), split.data(), half * hw * sizeof(float));

    std::vector<float> conv_tmp(half * hw);
    conv2d(conv_in.data(), half, H, W, to_f32(wt.conv_blk_0, dq1), nullptr,
           half, 3, 3, 1, 1, conv_tmp.data());
    for (int i = 0; i < half * hw; i++) conv_tmp[i] = std::max(0.0f, conv_tmp[i]); // ReLU
    std::vector<float> conv_out(half * hw);
    conv2d(conv_tmp.data(), half, H, W, to_f32(wt.conv_blk_2, dq1), nullptr,
           half, 3, 3, 1, 1, conv_out.data());
    for (int i = 0; i < half * hw; i++) conv_out[i] += conv_in[i]; // residual

    // Trans branch (second half)
    std::vector<float> trans_buf(half * hw);
    memcpy(trans_buf.data(), split.data() + half * hw, half * hw * sizeof(float));
    swin_block_forward(trans_buf.data(), half, H, W, wt.trans,
                       n_heads, win_size, shift, dq1, dq2, scratch);

    // Fuse: cat → 1x1 conv
    std::vector<float> cat_buf(C * hw);
    memcpy(cat_buf.data(), conv_out.data(), half * hw * sizeof(float));
    memcpy(cat_buf.data() + half * hw, trans_buf.data(), half * hw * sizeof(float));

    std::vector<float> fused(C * hw);
    const float * f_w = to_f32(wt.conv1_2_w, dq1);
    const float * f_b = to_f32(wt.conv1_2_b, dq2);
    conv2d(cat_buf.data(), C, H, W, f_w, f_b, C, 1, 1, 0, 1, fused.data());

    // Residual
    for (int i = 0; i < C * hw; i++) x[i] += fused[i];
}

// ── Model context ──────────────────────────────────────────────────

struct scunet_stage {
    std::vector<ctb_weights> blocks;
    ggml_tensor * ds_w = nullptr, * ds_b = nullptr; // downsample (encoder)
    ggml_tensor * us_w = nullptr, * us_b = nullptr; // upsample (decoder)
};

struct scunet_context {
    ggml_context * gguf_ctx;
    ggml_backend_buffer_t gguf_buf;
    ggml_backend_t backend;

    int dim, win_size, head_dim;
    bool bench;
    ggml_tensor * head_w, * head_b;
    scunet_stage enc[3]; // down1, down2, down3
    scunet_stage body;
    scunet_stage dec[3]; // up3, up2, up1
    ggml_tensor * tail_w, * tail_b;
};

static void load_ctb(core_gguf::WeightLoad & wl, const char * prefix, ctb_weights & ctb) {
    auto g = [&](const char * suffix) -> ggml_tensor * {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s.%s", prefix, suffix);
        return core_gguf::try_get(wl.tensors, buf);
    };
    ctb.conv1_1_w = g("conv1_1.weight"); ctb.conv1_1_b = g("conv1_1.bias");
    ctb.conv1_2_w = g("conv1_2.weight"); ctb.conv1_2_b = g("conv1_2.bias");
    ctb.conv_blk_0 = g("conv_block.0.weight"); ctb.conv_blk_2 = g("conv_block.2.weight");
    ctb.trans.ln1_w = g("trans_block.ln1.weight"); ctb.trans.ln1_b = g("trans_block.ln1.bias");
    ctb.trans.qkv_w = g("trans_block.msa.embedding_layer.weight");
    ctb.trans.qkv_b = g("trans_block.msa.embedding_layer.bias");
    ctb.trans.proj_w = g("trans_block.msa.linear.weight");
    ctb.trans.proj_b = g("trans_block.msa.linear.bias");
    ctb.trans.rpb = g("trans_block.msa.relative_position_params");
    ctb.trans.ln2_w = g("trans_block.ln2.weight"); ctb.trans.ln2_b = g("trans_block.ln2.bias");
    ctb.trans.mlp0_w = g("trans_block.mlp.0.weight"); ctb.trans.mlp0_b = g("trans_block.mlp.0.bias");
    ctb.trans.mlp2_w = g("trans_block.mlp.2.weight"); ctb.trans.mlp2_b = g("trans_block.mlp.2.bias");
}

scunet_context * scunet_init(const char * model_path, int n_threads) {
    (void)n_threads;
    if (!model_path) return nullptr;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) return nullptr;
    int dim = (int)core_gguf::kv_u32(meta, "scunet.dim", 64);
    int win = (int)core_gguf::kv_u32(meta, "scunet.win_size", 8);
    int hd  = (int)core_gguf::kv_u32(meta, "scunet.head_dim", 32);
    core_gguf::free_metadata(meta);

    bool force_cpu = (getenv("SCUNET_FORCE_CPU") && atoi(getenv("SCUNET_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(backend))
        ggml_backend_cpu_set_n_threads(backend, n_threads > 0 ? n_threads : 2);
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, backend, "scunet", wl)) {
        ggml_backend_free(backend); return nullptr;
    }

    auto * ctx = new scunet_context;
    ctx->backend = backend;
    ctx->gguf_ctx = wl.ctx; ctx->gguf_buf = wl.buf;
    ctx->dim = dim; ctx->win_size = win; ctx->head_dim = hd;

    ctx->head_w = core_gguf::try_get(wl.tensors, "m_head.0.weight");
    ctx->head_b = core_gguf::try_get(wl.tensors, "m_head.0.bias");
    ctx->tail_w = core_gguf::try_get(wl.tensors, "m_tail.0.weight");
    ctx->tail_b = core_gguf::try_get(wl.tensors, "m_tail.0.bias");

    const char * enc_names[] = {"m_down1", "m_down2", "m_down3"};
    for (int s = 0; s < 3; s++) {
        ctx->enc[s].blocks.resize(4);
        for (int i = 0; i < 4; i++) {
            char pfx[64];
            snprintf(pfx, sizeof(pfx), "%s.%d", enc_names[s], i);
            load_ctb(wl, pfx, ctx->enc[s].blocks[i]);
        }
        char dsw[64], dsb[64];
        snprintf(dsw, sizeof(dsw), "%s.4.weight", enc_names[s]);
        snprintf(dsb, sizeof(dsb), "%s.4.bias", enc_names[s]);
        ctx->enc[s].ds_w = core_gguf::try_get(wl.tensors, dsw);
        ctx->enc[s].ds_b = core_gguf::try_get(wl.tensors, dsb);
    }

    ctx->body.blocks.resize(4);
    for (int i = 0; i < 4; i++) {
        char pfx[64]; snprintf(pfx, sizeof(pfx), "m_body.%d", i);
        load_ctb(wl, pfx, ctx->body.blocks[i]);
    }

    const char * dec_names[] = {"m_up3", "m_up2", "m_up1"};
    for (int s = 0; s < 3; s++) {
        char usw[64], usb[64];
        snprintf(usw, sizeof(usw), "%s.0.weight", dec_names[s]);
        snprintf(usb, sizeof(usb), "%s.0.bias", dec_names[s]);
        ctx->dec[s].us_w = core_gguf::try_get(wl.tensors, usw);
        ctx->dec[s].us_b = core_gguf::try_get(wl.tensors, usb);
        ctx->dec[s].blocks.resize(4);
        for (int i = 0; i < 4; i++) {
            char pfx[64]; snprintf(pfx, sizeof(pfx), "%s.%d", dec_names[s], i + 1);
            load_ctb(wl, pfx, ctx->dec[s].blocks[i]);
        }
    }

    ctx->bench = (std::getenv("CRISPEMBED_SCUNET_BENCH") != nullptr);
    return ctx;
}

void scunet_free(scunet_context * ctx) {
    if (!ctx) return;
    core_gguf::WeightLoad wl;
    wl.ctx = ctx->gguf_ctx; wl.buf = ctx->gguf_buf;
    core_gguf::free_weights(wl);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

int scunet_process_float(scunet_context * ctx,
                         const float * input_chw, int width, int height,
                         float * output_chw) {
    if (!ctx || !input_chw || !output_chw) return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    int H = height, W = width;
    std::vector<float> dq1, dq2, scratch;

    // Head
    std::vector<float> x(ctx->dim * H * W);
    conv2d(input_chw, 3, H, W, to_f32(ctx->head_w, dq1), to_f32(ctx->head_b, dq2),
           ctx->dim, 3, 3, 1, 1, x.data());

    // Save skips
    std::vector<float> skip_head(x.begin(), x.end());
    std::vector<std::vector<float>> skips;

    // Encoder
    int ch = ctx->dim;
    int cur_h = H, cur_w = W;
    int enc_channels[] = {64, 128, 256};
    for (int s = 0; s < 3; s++) {
        auto t_enc = std::chrono::steady_clock::now();
        int n_heads = std::max(1, (ch / 2) / ctx->head_dim);
        for (int i = 0; i < 4; i++) {
            // Check for null tensors
            const auto & blk = ctx->enc[s].blocks[i];
            if (!blk.conv1_1_w) fprintf(stderr, "  WARNING: conv1_1_w is null!\n");
            if (!blk.conv_blk_0) fprintf(stderr, "  WARNING: conv_blk_0 is null!\n");
            if (!blk.trans.qkv_w) fprintf(stderr, "  WARNING: qkv_w is null!\n");
            if (!blk.trans.rpb) fprintf(stderr, "  WARNING: rpb is null!\n");
            bool shift = (i % 2 == 1);
            ctb_forward(x.data(), ch, cur_h, cur_w, ctx->enc[s].blocks[i],
                        n_heads, ctx->win_size, shift, dq1, dq2, scratch);
        }
        // Downsample
        int next_ch = ch * 2;
        int nh = cur_h / 2, nw = cur_w / 2;
        std::vector<float> ds(next_ch * nh * nw);
        conv2d(x.data(), ch, cur_h, cur_w,
               to_f32(ctx->enc[s].ds_w, dq1), to_f32(ctx->enc[s].ds_b, dq2),
               next_ch, 2, 2, 0, 2, ds.data());
        skips.push_back(std::move(ds));
        x = skips.back(); // copy
        ch = next_ch; cur_h = nh; cur_w = nw;
        if (bench) {
            auto t_enc_end = std::chrono::steady_clock::now();
            fprintf(stderr, "[scunet-bench] enc stage %d: %.1f ms\n",
                    s, ms_f(t_enc_end - t_enc).count());
        }
    }

    // Body
    auto t_body = std::chrono::steady_clock::now();
    int body_heads = std::max(1, (ch / 2) / ctx->head_dim);
    for (int i = 0; i < 4; i++) {
        bool shift = (i % 2 == 1);
        ctb_forward(x.data(), ch, cur_h, cur_w, ctx->body.blocks[i],
                    body_heads, ctx->win_size, shift, dq1, dq2, scratch);
    }
    if (bench) {
        auto t_body_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[scunet-bench] body: %.1f ms\n",
                ms_f(t_body_end - t_body).count());
    }

    // Decoder
    int dec_channels[] = {256, 128, 64};
    for (int s = 0; s < 3; s++) {
        auto t_dec = std::chrono::steady_clock::now();
        // Skip connection (add before upsample)
        auto & sk = skips[2 - s];
        for (int i = 0; i < ch * cur_h * cur_w; i++) x[i] += sk[i];

        // ConvTranspose2d upsample
        int next_ch = ch / 2;
        int nh = cur_h * 2, nw = cur_w * 2;
        std::vector<float> us(next_ch * nh * nw);
        conv_transpose2d(x.data(), ch, cur_h, cur_w,
                         to_f32(ctx->dec[s].us_w, dq1), to_f32(ctx->dec[s].us_b, dq2),
                         next_ch, 2, 2, 2, us.data());
        x = std::move(us);
        ch = next_ch; cur_h = nh; cur_w = nw;

        int n_heads = std::max(1, (ch / 2) / ctx->head_dim);
        for (int i = 0; i < 4; i++) {
            bool shift = (i % 2 == 1);
            ctb_forward(x.data(), ch, cur_h, cur_w, ctx->dec[s].blocks[i],
                        n_heads, ctx->win_size, shift, dq1, dq2, scratch);
        }
        if (bench) {
            auto t_dec_end = std::chrono::steady_clock::now();
            fprintf(stderr, "[scunet-bench] dec stage %d: %.1f ms\n",
                    s, ms_f(t_dec_end - t_dec).count());
        }
    }

    // Final skip + tail
    for (int i = 0; i < ch * cur_h * cur_w; i++) x[i] += skip_head[i];
    conv2d(x.data(), ch, cur_h, cur_w,
           to_f32(ctx->tail_w, dq1), to_f32(ctx->tail_b, dq2),
           3, 3, 3, 1, 1, output_chw);

    if (bench) {
        auto t_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[scunet-bench] total: %.1f ms\n",
                ms_f(t_end - t_total).count());
    }
    return 0;
}

int scunet_process_float_debug(scunet_context * ctx,
                                const float * input_chw, int width, int height,
                                float * output_chw,
                                scunet_stage_cb cb) {
    if (!ctx || !input_chw || !output_chw) return -1;
    int H = height, W = width;
    std::vector<float> dq1, dq2, scratch;

    // Enable swin block debug for block 0 (first trans block in first CTB)
    g_swin_debug_cb = cb;
    g_swin_debug_block_id = 0; // block 1 = first shifted window block
    g_swin_block_counter = 0;

    std::vector<float> x(ctx->dim * H * W);
    conv2d(input_chw, 3, H, W, to_f32(ctx->head_w, dq1), to_f32(ctx->head_b, dq2),
           ctx->dim, 3, 3, 1, 1, x.data());
    if (cb) cb("head", x.data(), ctx->dim * H * W);

    std::vector<float> skip_head(x.begin(), x.end());
    std::vector<std::vector<float>> skips;

    const char * enc_names[] = {"m_down1", "m_down2", "m_down3"};
    int ch = ctx->dim, cur_h = H, cur_w = W;
    for (int s = 0; s < 3; s++) {
        int n_heads = std::max(1, (ch / 2) / ctx->head_dim);
        for (int i = 0; i < 4; i++) {
            bool shift = (i % 2 == 1);
            ctb_forward(x.data(), ch, cur_h, cur_w, ctx->enc[s].blocks[i],
                        n_heads, ctx->win_size, shift, dq1, dq2, scratch);
        }
        int next_ch = ch * 2, nh = cur_h / 2, nw = cur_w / 2;
        std::vector<float> ds(next_ch * nh * nw);
        conv2d(x.data(), ch, cur_h, cur_w,
               to_f32(ctx->enc[s].ds_w, dq1), to_f32(ctx->enc[s].ds_b, dq2),
               next_ch, 2, 2, 0, 2, ds.data());
        skips.push_back(std::move(ds));
        x = skips.back();
        ch = next_ch; cur_h = nh; cur_w = nw;
        if (cb) cb(enc_names[s], x.data(), ch * cur_h * cur_w);
    }

    int body_heads = std::max(1, (ch / 2) / ctx->head_dim);
    for (int i = 0; i < 4; i++) {
        bool shift = (i % 2 == 1);
        ctb_forward(x.data(), ch, cur_h, cur_w, ctx->body.blocks[i],
                    body_heads, ctx->win_size, shift, dq1, dq2, scratch);
    }
    if (cb) cb("body", x.data(), ch * cur_h * cur_w);

    const char * dec_names[] = {"m_up3", "m_up2", "m_up1"};
    for (int s = 0; s < 3; s++) {
        auto & sk = skips[2 - s];
        for (int i = 0; i < ch * cur_h * cur_w; i++) x[i] += sk[i];
        int next_ch = ch / 2, nh = cur_h * 2, nw = cur_w * 2;
        std::vector<float> us(next_ch * nh * nw);
        conv_transpose2d(x.data(), ch, cur_h, cur_w,
                         to_f32(ctx->dec[s].us_w, dq1), to_f32(ctx->dec[s].us_b, dq2),
                         next_ch, 2, 2, 2, us.data());
        x = std::move(us);
        ch = next_ch; cur_h = nh; cur_w = nw;
        int n_heads = std::max(1, (ch / 2) / ctx->head_dim);
        for (int i = 0; i < 4; i++) {
            bool shift = (i % 2 == 1);
            ctb_forward(x.data(), ch, cur_h, cur_w, ctx->dec[s].blocks[i],
                        n_heads, ctx->win_size, shift, dq1, dq2, scratch);
        }
        if (cb) cb(dec_names[s], x.data(), ch * cur_h * cur_w);
    }

    for (int i = 0; i < ch * cur_h * cur_w; i++) x[i] += skip_head[i];
    conv2d(x.data(), ch, cur_h, cur_w,
           to_f32(ctx->tail_w, dq1), to_f32(ctx->tail_b, dq2),
           3, 3, 3, 1, 1, output_chw);
    return 0;
}

int scunet_process(scunet_context * ctx,
                   const uint8_t * input, int width, int height,
                   uint8_t * output) {
    if (!ctx || !input || !output) return -1;
    int hw = width * height;
    std::vector<float> in_chw(3 * hw);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                in_chw[c * hw + y * width + x] =
                    (float)input[(y * width + x) * 3 + c] / 255.0f;

    std::vector<float> out_chw(3 * hw);
    int ret = scunet_process_float(ctx, in_chw.data(), width, height, out_chw.data());
    if (ret != 0) return ret;

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++) {
                float v = out_chw[c * hw + y * width + x] * 255.0f;
                output[(y * width + x) * 3 + c] =
                    (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
    return 0;
}
