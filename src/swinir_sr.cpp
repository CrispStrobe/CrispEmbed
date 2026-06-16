// swinir_sr.cpp — SwinIR-light image super-resolution (CPU-scalar).
//
// Forward:
//   conv_first(3→D) → patch_embed_norm
//   → 4× RSTB { 6× SwinBlock(LN→W-MSA/SW-MSA→res, LN→MLP→res) + conv + residual }
//   → norm → conv_after_body + skip → upsample(Conv+PixelShuffle)
//
// SwinBlock (even index): window-MSA (no shift)
// SwinBlock (odd index):  shifted-window-MSA (cyclic shift + attn_mask)

#include "swinir_sr.h"
#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
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

static const float * sir_to_f32(const ggml_tensor * t, std::vector<float> & buf) {
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

static void sir_conv2d(const float * in, int ic, int ih, int iw,
                       const float * w, const float * b,
                       int oc, int kh, int kw, int pad, float * out) {
    int oh = ih + 2 * pad - kh + 1;
    int ow = iw + 2 * pad - kw + 1;
    for (int o = 0; o < oc; o++) {
        float bias = b ? b[o] : 0.0f;
        for (int oy = 0; oy < oh; oy++)
            for (int ox = 0; ox < ow; ox++) {
                float sum = bias;
                for (int c = 0; c < ic; c++)
                    for (int ky = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy + ky - pad, ix = ox + kx - pad;
                            if (iy >= 0 && iy < ih && ix >= 0 && ix < iw)
                                sum += in[c * ih * iw + iy * iw + ix]
                                     * w[o * ic * kh * kw + c * kh * kw + ky * kw + kx];
                        }
                out[o * oh * ow + oy * ow + ox] = sum;
            }
    }
}

static void sir_layernorm(const float * in, float * out, int n, int d,
                          const float * w, const float * b) {
    for (int i = 0; i < n; i++) {
        float mean = 0;
        for (int j = 0; j < d; j++) mean += in[i * d + j];
        mean /= d;
        float var = 0;
        for (int j = 0; j < d; j++) { float x = in[i * d + j] - mean; var += x * x; }
        var /= d;
        float inv = 1.0f / sqrtf(var + 1e-5f);
        for (int j = 0; j < d; j++)
            out[i * d + j] = (in[i * d + j] - mean) * inv * w[j] + b[j];
    }
}

static float sir_gelu(float x) {
    return 0.5f * x * (1.0f + erff(x / sqrtf(2.0f)));
}

static void sir_softmax(float * data, int n) {
    float mx = data[0];
    for (int i = 1; i < n; i++) if (data[i] > mx) mx = data[i];
    float sum = 0;
    for (int i = 0; i < n; i++) { data[i] = expf(data[i] - mx); sum += data[i]; }
    for (int i = 0; i < n; i++) data[i] /= sum;
}

static void sir_linear(const float * in, float * out, int in_d, int out_d,
                       const float * w, const float * b) {
    for (int o = 0; o < out_d; o++) {
        float s = b ? b[o] : 0.0f;
        for (int i = 0; i < in_d; i++) s += in[i] * w[o * in_d + i];
        out[o] = s;
    }
}

// ── Swin window attention ──────────────────────────────────────────────

// [H, W, C] → [nWin, ws², C]
static void window_partition(const float * x, float * out,
                             int H, int W, int C, int ws) {
    int nH = H / ws, nW = W / ws;
    for (int wh = 0; wh < nH; wh++)
        for (int ww = 0; ww < nW; ww++) {
            int wi = wh * nW + ww;
            for (int y = 0; y < ws; y++)
                for (int xp = 0; xp < ws; xp++)
                    memcpy(out + (wi * ws * ws + y * ws + xp) * C,
                           x + ((wh * ws + y) * W + (ww * ws + xp)) * C,
                           C * sizeof(float));
        }
}

// [nWin, ws², C] → [H, W, C]
static void window_reverse(const float * win, float * out,
                           int H, int W, int C, int ws) {
    int nH = H / ws, nW = W / ws;
    for (int wh = 0; wh < nH; wh++)
        for (int ww = 0; ww < nW; ww++) {
            int wi = wh * nW + ww;
            for (int y = 0; y < ws; y++)
                for (int xp = 0; xp < ws; xp++)
                    memcpy(out + ((wh * ws + y) * W + (ww * ws + xp)) * C,
                           win + (wi * ws * ws + y * ws + xp) * C,
                           C * sizeof(float));
        }
}

// Cyclic shift
static void cyclic_shift(const float * in, float * out,
                         int H, int W, int C, int sh, int sw) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            memcpy(out + (y * W + x) * C,
                   in + (((y + sh + H) % H) * W + ((x + sw + W) % W)) * C,
                   C * sizeof(float));
}

static void sir_pixel_shuffle(const float * in, int c_in, int h, int w,
                              int r, float * out) {
    int c_out = c_in / (r * r);
    int oh = h * r, ow = w * r;
    for (int c = 0; c < c_out; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                int ic = c * r * r + (y % r) * r + (x % r);
                out[c * oh * ow + y * ow + x] = in[ic * h * w + (y / r) * w + (x / r)];
            }
}

// ── Swin block forward ────────────────────────────────────────────────

struct swin_blk_w {
    const float * ln1_w, * ln1_b;
    const float * qkv_w, * qkv_b;      // fused [3D, D]
    const float * proj_w, * proj_b;
    const float * rpb_table;            // [rpb_size, n_heads]
    const int32_t * rpb_index;          // [ws², ws²] int32
    const float * attn_mask;            // [nWin, ws², ws²] or null
    int n_attn_mask_wins;
    const float * ln2_w, * ln2_b;
    const float * mlp_up_w, * mlp_up_b;
    const float * mlp_dn_w, * mlp_dn_b;
};

// Process one Swin block on a feature map [H*W, D] (already in token layout)
static void swin_block_forward(float * x, int H, int W, int D,
                               int n_heads, int ws, bool do_shift,
                               const swin_blk_w & wt,
                               std::vector<float> & tmp) {
    int N = H * W;
    int hd = D / n_heads;
    float scale = 1.0f / sqrtf((float)hd);
    int ws2 = ws * ws;
    int nH = H / ws, nW = W / ws;
    int nWin = nH * nW;

    // LN1
    tmp.resize(N * D);
    sir_layernorm(x, tmp.data(), N, D, wt.ln1_w, wt.ln1_b);

    // Reshape to [H, W, D] for windowing
    std::vector<float> spatial(N * D);
    memcpy(spatial.data(), tmp.data(), N * D * sizeof(float));

    // Cyclic shift if needed
    std::vector<float> shifted;
    if (do_shift) {
        shifted.resize(N * D);
        cyclic_shift(spatial.data(), shifted.data(), H, W, D, -ws / 2, -ws / 2);
    }
    float * to_partition = do_shift ? shifted.data() : spatial.data();

    // Window partition: [H, W, D] → [nWin, ws², D]
    std::vector<float> windows(nWin * ws2 * D);
    window_partition(to_partition, windows.data(), H, W, D, ws);

    // Window attention per window
    std::vector<float> win_out(nWin * ws2 * D);
    for (int wi = 0; wi < nWin; wi++) {
        float * win_tokens = windows.data() + wi * ws2 * D;
        float * win_result = win_out.data() + wi * ws2 * D;

        // QKV projection: [ws², D] → [ws², 3D]
        std::vector<float> qkv(ws2 * 3 * D);
        for (int t = 0; t < ws2; t++)
            sir_linear(win_tokens + t * D, qkv.data() + t * 3 * D,
                       D, 3 * D, wt.qkv_w, wt.qkv_b);

        // Attention per head
        std::vector<float> attn_out(ws2 * D, 0.0f);
        for (int h = 0; h < n_heads; h++) {
            int off = h * hd;

            // Scores [ws², ws²]
            std::vector<float> scores(ws2 * ws2);
            for (int i = 0; i < ws2; i++) {
                const float * qi = qkv.data() + i * 3 * D + off;         // Q
                for (int j = 0; j < ws2; j++) {
                    const float * kj = qkv.data() + j * 3 * D + D + off; // K
                    float dot = 0;
                    for (int d = 0; d < hd; d++) dot += qi[d] * kj[d];
                    float s = dot * scale;

                    // RPB
                    if (wt.rpb_table && wt.rpb_index) {
                        int idx = wt.rpb_index[i * ws2 + j];
                        s += wt.rpb_table[idx * n_heads + h];
                    }

                    // Attention mask (shifted windows)
                    if (wt.attn_mask && do_shift) {
                        int mask_wi = wi % wt.n_attn_mask_wins;
                        s += wt.attn_mask[mask_wi * ws2 * ws2 + i * ws2 + j];
                    }

                    scores[i * ws2 + j] = s;
                }
            }

            // Softmax per row
            for (int i = 0; i < ws2; i++)
                sir_softmax(scores.data() + i * ws2, ws2);

            // Weighted sum of V
            for (int i = 0; i < ws2; i++) {
                for (int d = 0; d < hd; d++) {
                    float sum = 0;
                    for (int j = 0; j < ws2; j++) {
                        const float * vj = qkv.data() + j * 3 * D + 2 * D + off; // V
                        sum += scores[i * ws2 + j] * vj[d];
                    }
                    attn_out[i * D + off + d] = sum;
                }
            }
        }

        // Output projection
        for (int t = 0; t < ws2; t++)
            sir_linear(attn_out.data() + t * D, win_result + t * D,
                       D, D, wt.proj_w, wt.proj_b);
    }

    // Window reverse
    std::vector<float> merged(N * D);
    window_reverse(win_out.data(), merged.data(), H, W, D, ws);

    // Reverse cyclic shift
    if (do_shift) {
        std::vector<float> unshifted(N * D);
        cyclic_shift(merged.data(), unshifted.data(), H, W, D, ws / 2, ws / 2);
        merged = std::move(unshifted);
    }

    // Residual: x = x + attn_out
    for (int i = 0; i < N * D; i++) x[i] += merged[i];

    // LN2 + MLP + residual
    tmp.resize(N * D);
    sir_layernorm(x, tmp.data(), N, D, wt.ln2_w, wt.ln2_b);

    int mlp_dim = D * 2; // mlp_ratio=2
    std::vector<float> mlp_up(N * mlp_dim);
    std::vector<float> mlp_dn(N * D);
    for (int i = 0; i < N; i++) {
        sir_linear(tmp.data() + i * D, mlp_up.data() + i * mlp_dim,
                   D, mlp_dim, wt.mlp_up_w, wt.mlp_up_b);
        for (int j = 0; j < mlp_dim; j++)
            mlp_up[i * mlp_dim + j] = sir_gelu(mlp_up[i * mlp_dim + j]);
        sir_linear(mlp_up.data() + i * mlp_dim, mlp_dn.data() + i * D,
                   mlp_dim, D, wt.mlp_dn_w, wt.mlp_dn_b);
    }

    for (int i = 0; i < N * D; i++) x[i] += mlp_dn[i];
}

// ── Context ───────────────────────────────────────────────────────────

struct swinir_sr_context {
    int embed_dim, n_rstb, n_blocks, n_heads, window_size, mlp_ratio, upscale;
    int n_threads;
    core_gguf::WeightLoad wl;
    std::vector<std::vector<float>> wbufs;
    std::vector<std::vector<int32_t>> ibufs;

    const float * get(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) { fprintf(stderr, "swinir_sr: missing %s\n", name.c_str()); return nullptr; }
        wbufs.emplace_back();
        return sir_to_f32(t, wbufs.back());
    }
    const int32_t * get_i32(const std::string & name) {
        auto * t = core_gguf::try_get(wl.tensors, name.c_str());
        if (!t) { fprintf(stderr, "swinir_sr: missing %s\n", name.c_str()); return nullptr; }
        int64_t n = ggml_nelements(t);
        ibufs.emplace_back(n);
        if (t->type == GGML_TYPE_I32) {
            ggml_backend_tensor_get(t, ibufs.back().data(), 0, n * sizeof(int32_t));
        } else {
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(float));
            for (int64_t i = 0; i < n; i++) ibufs.back()[i] = (int32_t)tmp[i];
        }
        return ibufs.back().data();
    }
};

swinir_sr_context * swinir_sr_init(const char * model_path, int n_threads) {
    auto * ctx = new swinir_sr_context;
    ctx->n_threads = n_threads > 0 ? n_threads : 2;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) {
        fprintf(stderr, "swinir_sr: failed to open %s\n", model_path);
        delete ctx; return nullptr;
    }

    ctx->embed_dim   = core_gguf::kv_u32(meta, "swinir.embed_dim", 60);
    ctx->n_rstb      = core_gguf::kv_u32(meta, "swinir.n_rstb", 4);
    ctx->n_blocks    = core_gguf::kv_u32(meta, "swinir.n_blocks", 6);
    ctx->n_heads     = core_gguf::kv_u32(meta, "swinir.n_heads", 6);
    ctx->window_size = core_gguf::kv_u32(meta, "swinir.window_size", 8);
    ctx->mlp_ratio   = core_gguf::kv_u32(meta, "swinir.mlp_ratio", 2);
    ctx->upscale     = core_gguf::kv_u32(meta, "swinir.upscale", 4);
    core_gguf::free_metadata(meta);

    bool force_cpu = (getenv("SWINIR_SR_FORCE_CPU") && atoi(getenv("SWINIR_SR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(backend))
        ggml_backend_cpu_set_n_threads(backend, ctx->n_threads);
    if (!core_gguf::load_weights(model_path, backend, "swinir", ctx->wl)) {
        fprintf(stderr, "swinir_sr: failed to load weights\n");
        ggml_backend_free(backend); delete ctx; return nullptr;
    }
    ggml_backend_free(backend);

    fprintf(stderr, "swinir_sr: dim=%d, rstb=%d, blocks=%d, heads=%d, ws=%d, scale=%dx, %d tensors\n",
            ctx->embed_dim, ctx->n_rstb, ctx->n_blocks, ctx->n_heads,
            ctx->window_size, ctx->upscale, (int)ctx->wl.tensors.size());
    return ctx;
}

void swinir_sr_free(swinir_sr_context * ctx) {
    if (ctx) { core_gguf::free_weights(ctx->wl); delete ctx; }
}

int swinir_sr_scale(const swinir_sr_context * ctx) {
    return ctx ? ctx->upscale : 0;
}

// ── Single-tile forward ───────────────────────────────────────────────

static void swinir_forward_tile(swinir_sr_context * ctx,
                                const float * tile_in, int tw, int th,
                                float * tile_out) {
    int D = ctx->embed_dim;
    int ws = ctx->window_size;
    int scale = ctx->upscale;

    // Pad to multiple of window_size
    int pH = ((th + ws - 1) / ws) * ws;
    int pW = ((tw + ws - 1) / ws) * ws;

    // Copy input + reflect-pad
    std::vector<float> img(3 * pH * pW, 0.0f);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < pH; y++)
            for (int x = 0; x < pW; x++)
                img[c * pH * pW + y * pW + x] =
                    tile_in[c * th * tw + std::min(y, th - 1) * tw + std::min(x, tw - 1)];

    // conv_first: 3 → D
    std::vector<float> shallow(D * pH * pW);
    sir_conv2d(img.data(), 3, pH, pW,
               ctx->get("conv_first.weight"), ctx->get("conv_first.bias"),
               D, 3, 3, 1, shallow.data());

    // Save for global residual
    std::vector<float> shallow_save = shallow;

    // Reshape to token layout [H*W, D] + patch_embed norm
    int N = pH * pW;
    std::vector<float> tokens(N * D);
    // CHW → HWC
    for (int y = 0; y < pH; y++)
        for (int x = 0; x < pW; x++)
            for (int c = 0; c < D; c++)
                tokens[(y * pW + x) * D + c] = shallow[c * pH * pW + y * pW + x];

    sir_layernorm(tokens.data(), tokens.data(), N, D,
                  ctx->get("patch_norm.weight"), ctx->get("patch_norm.bias"));

    std::vector<float> tmp;

    // RSTB blocks
    for (int r = 0; r < ctx->n_rstb; r++) {
        std::vector<float> rstb_input = tokens;

        for (int b = 0; b < ctx->n_blocks; b++) {
            char prefix[80];
            snprintf(prefix, sizeof(prefix), "rstb.%d.block.%d", r, b);

            bool do_shift = (b % 2 == 1);

            swin_blk_w wt;
            wt.ln1_w    = ctx->get(std::string(prefix) + ".norm1.weight");
            wt.ln1_b    = ctx->get(std::string(prefix) + ".norm1.bias");
            wt.qkv_w    = ctx->get(std::string(prefix) + ".attn.qkv.weight");
            wt.qkv_b    = ctx->get(std::string(prefix) + ".attn.qkv.bias");
            wt.proj_w   = ctx->get(std::string(prefix) + ".attn.proj.weight");
            wt.proj_b   = ctx->get(std::string(prefix) + ".attn.proj.bias");
            wt.rpb_table = ctx->get(std::string(prefix) + ".attn.rpb_table");
            wt.rpb_index = ctx->get_i32(std::string(prefix) + ".attn.rpb_index");
            wt.ln2_w    = ctx->get(std::string(prefix) + ".norm2.weight");
            wt.ln2_b    = ctx->get(std::string(prefix) + ".norm2.bias");
            wt.mlp_up_w = ctx->get(std::string(prefix) + ".mlp.up.weight");
            wt.mlp_up_b = ctx->get(std::string(prefix) + ".mlp.up.bias");
            wt.mlp_dn_w = ctx->get(std::string(prefix) + ".mlp.down.weight");
            wt.mlp_dn_b = ctx->get(std::string(prefix) + ".mlp.down.bias");

            std::string mask_name = std::string(prefix) + ".attn_mask";
            auto * mask_t = core_gguf::try_get(ctx->wl.tensors, mask_name.c_str());
            if (mask_t && do_shift) {
                ctx->wbufs.emplace_back();
                wt.attn_mask = sir_to_f32(mask_t, ctx->wbufs.back());
                // attn_mask shape: [nWin, ws², ws²]
                wt.n_attn_mask_wins = (int)(ggml_nelements(mask_t) / (ws * ws * ws * ws));
            } else {
                wt.attn_mask = nullptr;
                wt.n_attn_mask_wins = 0;
            }

            swin_block_forward(tokens.data(), pH, pW, D,
                               ctx->n_heads, ws, do_shift, wt, tmp);
        }

        // Patch unembed: [H*W, D] → [D, H, W]  (HWC → CHW)
        std::vector<float> spatial(D * pH * pW);
        for (int y = 0; y < pH; y++)
            for (int x = 0; x < pW; x++)
                for (int c = 0; c < D; c++)
                    spatial[c * pH * pW + y * pW + x] = tokens[(y * pW + x) * D + c];

        // RSTB conv + residual
        char rconv_w[64], rconv_b[64];
        snprintf(rconv_w, sizeof(rconv_w), "rstb.%d.conv.weight", r);
        snprintf(rconv_b, sizeof(rconv_b), "rstb.%d.conv.bias", r);
        std::vector<float> conv_out(D * pH * pW);
        sir_conv2d(spatial.data(), D, pH, pW,
                   ctx->get(rconv_w), ctx->get(rconv_b),
                   D, 3, 3, 1, conv_out.data());

        // Add RSTB residual (in token layout)
        for (int i = 0; i < N * D; i++) {
            // rstb_input is in HWC, conv_out is in CHW — convert conv_out to HWC
        }
        // Actually: add in CHW then convert back to HWC for next RSTB
        // Simpler: work in CHW for the residual
        std::vector<float> rstb_chw(D * pH * pW);
        for (int y = 0; y < pH; y++)
            for (int x = 0; x < pW; x++)
                for (int c = 0; c < D; c++)
                    rstb_chw[c * pH * pW + y * pW + x] = rstb_input[(y * pW + x) * D + c];

        for (int i = 0; i < D * pH * pW; i++)
            conv_out[i] += rstb_chw[i];

        // Convert back to HWC for next RSTB's patch_embed
        for (int y = 0; y < pH; y++)
            for (int x = 0; x < pW; x++)
                for (int c = 0; c < D; c++)
                    tokens[(y * pW + x) * D + c] = conv_out[c * pH * pW + y * pW + x];

        // Re-apply patch_embed norm for next RSTB
        sir_layernorm(tokens.data(), tokens.data(), N, D,
                      ctx->get("patch_norm.weight"), ctx->get("patch_norm.bias"));

        if (getenv("CRISPEMBED_SWINIR_DEBUG")) {
            float mn = 1e9, mx = -1e9;
            double sum = 0;
            for (int i = 0; i < N * D; i++) {
                if (tokens[i] < mn) mn = tokens[i];
                if (tokens[i] > mx) mx = tokens[i];
                sum += tokens[i];
            }
            fprintf(stderr, "swinir_sr: rstb_%d tokens min=%.4f max=%.4f mean=%.4f\n",
                    r, mn, mx, sum / (N * D));
        }
    }

    // Final norm
    sir_layernorm(tokens.data(), tokens.data(), N, D,
                  ctx->get("norm.weight"), ctx->get("norm.bias"));

    // Patch unembed → CHW
    std::vector<float> features(D * pH * pW);
    for (int y = 0; y < pH; y++)
        for (int x = 0; x < pW; x++)
            for (int c = 0; c < D; c++)
                features[c * pH * pW + y * pW + x] = tokens[(y * pW + x) * D + c];

    // conv_after_body + global residual
    std::vector<float> body_out(D * pH * pW);
    sir_conv2d(features.data(), D, pH, pW,
               ctx->get("conv_after_body.weight"), ctx->get("conv_after_body.bias"),
               D, 3, 3, 1, body_out.data());
    for (int i = 0; i < D * pH * pW; i++)
        body_out[i] += shallow_save[i];

    // Upsample: Conv(D → 3*scale², 3×3) + PixelShuffle
    int up_oc = 3 * scale * scale;
    std::vector<float> up_conv(up_oc * pH * pW);
    sir_conv2d(body_out.data(), D, pH, pW,
               ctx->get("upsample.weight"), ctx->get("upsample.bias"),
               up_oc, 3, 3, 1, up_conv.data());

    int out_h = pH * scale, out_w = pW * scale;
    std::vector<float> ps_out(3 * out_h * out_w);
    sir_pixel_shuffle(up_conv.data(), up_oc, pH, pW, scale, ps_out.data());

    // Crop to actual output size and write
    int crop_h = th * scale, crop_w = tw * scale;
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < crop_h; y++)
            for (int x = 0; x < crop_w; x++)
                tile_out[c * crop_h * crop_w + y * crop_w + x] =
                    ps_out[c * out_h * out_w + y * out_w + x];
}

// ── Tiled processing (same Hann blending as text_sr) ──────────────────

static void build_blend_window(int size, int overlap, std::vector<float> & win) {
    win.resize(size * size);
    for (int y = 0; y < size; y++) {
        float wy = 1.0f;
        if (y < overlap) wy = 0.5f - 0.5f * cosf((float)M_PI * y / overlap);
        else if (y >= size - overlap) wy = 0.5f - 0.5f * cosf((float)M_PI * (size - 1 - y) / overlap);
        for (int x = 0; x < size; x++) {
            float wx = 1.0f;
            if (x < overlap) wx = 0.5f - 0.5f * cosf((float)M_PI * x / overlap);
            else if (x >= size - overlap) wx = 0.5f - 0.5f * cosf((float)M_PI * (size - 1 - x) / overlap);
            win[y * size + x] = wy * wx;
        }
    }
}

int swinir_sr_process(swinir_sr_context * ctx,
                      const uint8_t * input, int width, int height,
                      int tile_size, int tile_overlap,
                      uint8_t ** output, int * out_width, int * out_height) {
    if (!ctx || !input || !output || width <= 0 || height <= 0) return -1;

    int r = ctx->upscale;
    if (tile_size <= 0) tile_size = 64;
    if (tile_overlap <= 0) tile_overlap = 8;
    tile_overlap = std::min(tile_overlap, tile_size / 4);

    int ow = width * r, oh = height * r;
    int out_tile = tile_size * r;
    int out_overlap = tile_overlap * r;

    std::vector<float> accum(3 * oh * ow, 0.0f);
    std::vector<float> weight_map(oh * ow, 0.0f);

    std::vector<float> blend_win;
    build_blend_window(out_tile, out_overlap, blend_win);

    // Convert input to [3, H, W] float [0, 1]
    std::vector<float> full_input(3 * height * width);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                full_input[c * height * width + y * width + x] =
                    input[(y * width + x) * 3 + c] / 255.0f;

    int step = tile_size - tile_overlap;
    int ntx = std::max(1, (width + step - 1) / step);
    int nty = std::max(1, (height + step - 1) / step);

    fprintf(stderr, "swinir_sr: %dx%d → %dx%d (%dx), tiles=%dx%d\n",
            width, height, ow, oh, r, ntx, nty);

    for (int ty = 0; ty < nty; ty++) {
        for (int tx = 0; tx < ntx; tx++) {
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

            int otw = tw * r, oth = th * r;
            std::vector<float> tile_out(3 * oth * otw);
            swinir_forward_tile(ctx, tile_in.data(), tw, th, tile_out.data());

            // Blend
            int ox0 = x0 * r, oy0 = y0 * r;
            for (int y = 0; y < oth; y++)
                for (int x = 0; x < otw; x++) {
                    float w = 1.0f;
                    if (tw == tile_size && th == tile_size)
                        w = blend_win[y * out_tile + x];
                    else {
                        if (x0 > 0 && x < out_overlap)
                            w *= 0.5f - 0.5f * cosf((float)M_PI * x / out_overlap);
                        if (y0 > 0 && y < out_overlap)
                            w *= 0.5f - 0.5f * cosf((float)M_PI * y / out_overlap);
                    }
                    int dy = oy0 + y, dx = ox0 + x;
                    if (dy >= oh || dx >= ow) continue;
                    for (int c = 0; c < 3; c++)
                        accum[c * oh * ow + dy * ow + dx] +=
                            tile_out[c * oth * otw + y * otw + x] * w;
                    weight_map[dy * ow + dx] += w;
                }
        }
    }

    // Normalize + convert to uint8
    uint8_t * out_buf = (uint8_t *)malloc(3 * oh * ow);
    if (!out_buf) return -1;
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++) {
            float w = weight_map[y * ow + x];
            if (w <= 0.0f) w = 1.0f;
            for (int c = 0; c < 3; c++) {
                float v = accum[c * oh * ow + y * ow + x] / w * 255.0f;
                out_buf[(y * ow + x) * 3 + c] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
        }

    *output = out_buf;
    *out_width = ow;
    *out_height = oh;

    if (getenv("CRISPEMBED_SWINIR_DEBUG")) {
        // Print output pixel stats
        float mn = 256, mx = -1;
        double sum = 0;
        for (int i = 0; i < ow * oh * 3; i++) {
            float v = out_buf[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
            sum += v;
        }
        fprintf(stderr, "swinir_sr: output u8 min=%.0f max=%.0f mean=%.1f\n",
                mn, mx, sum / (ow * oh * 3));
    }

    fprintf(stderr, "swinir_sr: done (%dx%d)\n", ow, oh);
    return 0;
}

void swinir_sr_free_image(uint8_t * pixels) { free(pixels); }
