// ppformulanet_ocr.cpp — PPFormulaNet-S (HGNetv2 + MBart) via ggml.
//
// Follows the same patterns as math_ocr.cpp:
// - CPU-only ggml graph compute for encoder
// - Scalar decoder with KV caching for autoregressive generation
// - Greedy decoding loop

#include "ppformulanet_ocr.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// FP16/quantized → F32 dequantization
// ---------------------------------------------------------------------------

static std::vector<float> to_f32(const ggml_tensor* t) {
    if (!t) return {};
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t* src = (const ggml_fp16_t*)t->data;
        for (int i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        const auto* traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) {
            traits->to_float(t->data, out.data(), n);
        } else {
            memset(out.data(), 0, n * sizeof(float));
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// CPU-side math helpers
// ---------------------------------------------------------------------------

static void layernorm_cpu(const float* in, float* out, int D,
                          const ggml_tensor* w, const ggml_tensor* b) {
    auto wv = to_f32(w);
    auto bv = to_f32(b);
    double mean = 0;
    for (int i = 0; i < D; i++) mean += in[i];
    mean /= D;
    double var = 0;
    for (int i = 0; i < D; i++) { double d = in[i] - mean; var += d * d; }
    var /= D;
    float s = 1.0f / sqrtf((float)var + 1e-5f);
    for (int i = 0; i < D; i++)
        out[i] = ((in[i] - (float)mean) * s) * (wv.empty() ? 1.0f : wv[i])
                 + (bv.empty() ? 0.0f : bv[i]);
}

static void linear_cpu(const float* in, float* out, int in_dim, int out_dim,
                        const ggml_tensor* w, const ggml_tensor* b) {
    auto wv = to_f32(w);
    auto bv = to_f32(b);
    for (int o = 0; o < out_dim; o++) {
        float s = bv.empty() ? 0.0f : bv[o];
        for (int i = 0; i < in_dim; i++)
            s += in[i] * wv[o * in_dim + i];
        out[o] = s;
    }
}

static void mha_1q_cpu(const float* q, const float* k, const float* v,
                        float* out, int n_kv, int D, int n_heads) {
    int hd = D / n_heads;
    std::vector<float> result(D, 0.0f);
    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;
        // Compute attention scores for this head
        std::vector<float> scores(n_kv);
        for (int ki = 0; ki < n_kv; ki++) {
            float s = 0;
            for (int d = 0; d < hd; d++)
                s += q[off + d] * k[ki * D + off + d];
            scores[ki] = s / sqrtf((float)hd);
        }
        // Softmax
        float maxs = *std::max_element(scores.begin(), scores.end());
        float sum = 0;
        for (int ki = 0; ki < n_kv; ki++) {
            scores[ki] = expf(scores[ki] - maxs);
            sum += scores[ki];
        }
        for (int ki = 0; ki < n_kv; ki++) scores[ki] /= sum;
        // Weighted sum
        for (int d = 0; d < hd; d++) {
            float s = 0;
            for (int ki = 0; ki < n_kv; ki++)
                s += scores[ki] * v[ki * D + off + d];
            result[off + d] = s;
        }
    }
    memcpy(out, result.data(), D * sizeof(float));
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

// Conv layer: weight (2D flattened) + bias (from folded BN)
struct conv_layer {
    ggml_tensor* w = nullptr; // [out_ch, in_ch*kh*kw] flattened
    ggml_tensor* b = nullptr; // [out_ch]
};

// HG_Block conv layers + aggregation
struct hg_block {
    // For regular blocks: layers[i] has one conv
    // For light blocks: layers[i] has conv1 (pointwise) + conv2 (depthwise)
    struct layer_t {
        conv_layer conv;        // regular ConvBNAct
        conv_layer conv1;       // light: pointwise 1×1
        conv_layer conv2;       // light: depthwise
        bool is_light = false;
    };
    std::vector<layer_t> layers;
    conv_layer agg_squeeze;  // 1×1 squeeze
    conv_layer agg_excite;   // 1×1 excitation
};

// Stage: optional downsample + blocks
struct hg_stage {
    conv_layer downsample;      // depthwise conv (may be null)
    bool has_downsample = false;
    std::vector<hg_block> blocks;
    // Config
    int in_ch, mid_ch, out_ch, n_blocks, n_layers, kernel_size;
    bool light_block;
};

// Decoder layer (MBart)
struct ppfn_dec_layer {
    ggml_tensor *self_ln_w, *self_ln_b;
    ggml_tensor *self_q_w, *self_q_b, *self_k_w, *self_k_b, *self_v_w, *self_v_b;
    ggml_tensor *self_out_w, *self_out_b;
    ggml_tensor *cross_ln_w, *cross_ln_b;
    ggml_tensor *cross_q_w, *cross_q_b, *cross_k_w, *cross_k_b, *cross_v_w, *cross_v_b;
    ggml_tensor *cross_out_w, *cross_out_b;
    ggml_tensor *ff_ln_w, *ff_ln_b, *ff_up_w, *ff_up_b, *ff_down_w, *ff_down_b;
};

struct ppformulanet_ocr_context {
    ppformulanet_ocr_hparams hparams;

    // Encoder: Stem
    conv_layer stem[5]; // stem1..stem4 (index 0..4, skip index 0 for stem1 at [0])

    // Encoder: Stages
    hg_stage stages[4];

    // Projection
    ggml_tensor *proj_w, *proj_b;

    // Decoder
    ggml_tensor *tok_embed, *pos_embed_dec;
    ggml_tensor *embed_ln_w, *embed_ln_b;
    ggml_tensor *final_ln_w, *final_ln_b;
    ggml_tensor *lm_head_w, *lm_head_b;
    std::vector<ppfn_dec_layer> dec_layers;

    // Infrastructure
    std::vector<std::string> vocab;
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    int n_threads;
    std::string result_buf;

    // Cached encoder output
    std::vector<float> enc_out;      // [n_enc_tokens * enc_hidden]
    int n_enc_tokens = 0;
    std::vector<float> proj_out;     // [n_enc_tokens * dec_d_model]

    // Cross-attention K/V cache (precomputed from projected encoder output)
    std::vector<std::vector<float>> cross_k_cache;
    std::vector<std::vector<float>> cross_v_cache;
};

// ---------------------------------------------------------------------------
// Tensor lookup
// ---------------------------------------------------------------------------

static ggml_tensor* F(const std::map<std::string, ggml_tensor*>& m, const char* n) {
    auto it = m.find(n);
    if (it != m.end()) return it->second;
    std::string alt(n);
    for (auto& c : alt) if (c == '.') c = '_';
    it = m.find(alt);
    return it != m.end() ? it->second : nullptr;
}

static conv_layer map_conv(const std::map<std::string, ggml_tensor*>& m,
                            const char* prefix) {
    char buf[256];
    conv_layer c;
    snprintf(buf, sizeof(buf), "%s.weight", prefix);
    c.w = F(m, buf);
    snprintf(buf, sizeof(buf), "%s.bias", prefix);
    c.b = F(m, buf);
    return c;
}

static void map_tensors(ppformulanet_ocr_context* ctx) {
    const auto& m = ctx->wl.tensors;
    const auto& hp = ctx->hparams;
    char buf[256];

    // Stem
    const char* stem_names[] = {"enc.stem.stem1", "enc.stem.stem2a", "enc.stem.stem2b",
                                 "enc.stem.stem3", "enc.stem.stem4"};
    for (int i = 0; i < 5; i++)
        ctx->stem[i] = map_conv(m, stem_names[i]);

    // Stages
    for (int si = 0; si < 4; si++) {
        auto& st = ctx->stages[si];

        // Downsample
        snprintf(buf, sizeof(buf), "enc.stage%d.downsample", si);
        st.downsample = map_conv(m, buf);
        st.has_downsample = (st.downsample.w != nullptr);

        // Blocks
        st.blocks.resize(st.n_blocks);
        for (int bi = 0; bi < st.n_blocks; bi++) {
            auto& blk = st.blocks[bi];
            blk.layers.resize(st.n_layers);

            for (int li = 0; li < st.n_layers; li++) {
                auto& lay = blk.layers[li];
                if (st.light_block) {
                    lay.is_light = true;
                    snprintf(buf, sizeof(buf), "enc.stage%d.block%d.layer%d.conv1", si, bi, li);
                    lay.conv1 = map_conv(m, buf);
                    snprintf(buf, sizeof(buf), "enc.stage%d.block%d.layer%d.conv2", si, bi, li);
                    lay.conv2 = map_conv(m, buf);
                } else {
                    lay.is_light = false;
                    snprintf(buf, sizeof(buf), "enc.stage%d.block%d.layer%d", si, bi, li);
                    lay.conv = map_conv(m, buf);
                }
            }

            snprintf(buf, sizeof(buf), "enc.stage%d.block%d.agg_squeeze", si, bi);
            blk.agg_squeeze = map_conv(m, buf);
            snprintf(buf, sizeof(buf), "enc.stage%d.block%d.agg_excite", si, bi);
            blk.agg_excite = map_conv(m, buf);
        }
    }

    // Projection
    ctx->proj_w = F(m, "enc.proj.weight");
    ctx->proj_b = F(m, "enc.proj.bias");

    // Decoder
    ctx->tok_embed   = F(m, "dec.embed_tokens.weight");
    ctx->pos_embed_dec = F(m, "dec.embed_positions.weight");
    ctx->embed_ln_w  = F(m, "dec.embed_ln.weight");
    ctx->embed_ln_b  = F(m, "dec.embed_ln.bias");
    ctx->final_ln_w  = F(m, "dec.final_ln.weight");
    ctx->final_ln_b  = F(m, "dec.final_ln.bias");
    ctx->lm_head_w   = F(m, "dec.lm_head.weight");
    ctx->lm_head_b   = F(m, "dec.lm_head.bias");

    ctx->dec_layers.resize(hp.dec_layers);
    for (int i = 0; i < hp.dec_layers; i++) {
        auto& l = ctx->dec_layers[i];
        auto DL = [&](const char* s) {
            snprintf(buf, sizeof(buf), "dec.layers.%d.%s", i, s);
            return F(m, buf);
        };
        l.self_ln_w = DL("self_attn_ln.weight"); l.self_ln_b = DL("self_attn_ln.bias");
        l.self_q_w = DL("self_attn.q.weight"); l.self_q_b = DL("self_attn.q.bias");
        l.self_k_w = DL("self_attn.k.weight"); l.self_k_b = DL("self_attn.k.bias");
        l.self_v_w = DL("self_attn.v.weight"); l.self_v_b = DL("self_attn.v.bias");
        l.self_out_w = DL("self_attn.out.weight"); l.self_out_b = DL("self_attn.out.bias");
        l.cross_ln_w = DL("cross_attn_ln.weight"); l.cross_ln_b = DL("cross_attn_ln.bias");
        l.cross_q_w = DL("cross_attn.q.weight"); l.cross_q_b = DL("cross_attn.q.bias");
        l.cross_k_w = DL("cross_attn.k.weight"); l.cross_k_b = DL("cross_attn.k.bias");
        l.cross_v_w = DL("cross_attn.v.weight"); l.cross_v_b = DL("cross_attn.v.bias");
        l.cross_out_w = DL("cross_attn.out.weight"); l.cross_out_b = DL("cross_attn.out.bias");
        l.ff_ln_w = DL("ffn_ln.weight"); l.ff_ln_b = DL("ffn_ln.bias");
        l.ff_up_w = DL("ffn.up.weight"); l.ff_up_b = DL("ffn.up.bias");
        l.ff_down_w = DL("ffn.down.weight"); l.ff_down_b = DL("ffn.down.bias");
    }
}

// ---------------------------------------------------------------------------
// CNN Encoder: CPU-side forward pass
// ---------------------------------------------------------------------------
// All conv weights are pre-folded (BN absorbed). Weights stored as 2D:
//   [out_ch, in_ch * kh * kw] for regular convolutions
//   [out_ch, 1 * kh * kw]     for depthwise (groups=in_ch)
// We perform convolution manually on CPU. For a 57M model at 384×384 this
// is ~0.5-1 second on a modern CPU. Good enough for initial port; can move
// to ggml graph later for SIMD acceleration.

// Helper: 2D convolution, handles groups
static void conv2d_cpu(const float* in, float* out,
                        const float* weight, const float* bias,
                        int in_ch, int out_ch, int H, int W,
                        int kh, int kw, int stride, int pad,
                        int groups) {
    int out_H = (H + 2 * pad - kh) / stride + 1;
    int out_W = (W + 2 * pad - kw) / stride + 1;
    int ch_per_group_in = in_ch / groups;
    int ch_per_group_out = out_ch / groups;

    for (int oc = 0; oc < out_ch; oc++) {
        int g = oc / ch_per_group_out;
        int oc_in_group = oc % ch_per_group_out;
        float b = bias ? bias[oc] : 0.0f;

        for (int oy = 0; oy < out_H; oy++) {
            for (int ox = 0; ox < out_W; ox++) {
                float sum = b;
                for (int ic = 0; ic < ch_per_group_in; ic++) {
                    int actual_ic = g * ch_per_group_in + ic;
                    for (int ky = 0; ky < kh; ky++) {
                        for (int kx = 0; kx < kw; kx++) {
                            int iy = oy * stride - pad + ky;
                            int ix = ox * stride - pad + kx;
                            if (iy >= 0 && iy < H && ix >= 0 && ix < W) {
                                float pixel = in[actual_ic * H * W + iy * W + ix];
                                int w_idx = oc * (ch_per_group_in * kh * kw)
                                            + ic * kh * kw + ky * kw + kx;
                                sum += pixel * weight[w_idx];
                            }
                        }
                    }
                }
                out[oc * out_H * out_W + oy * out_W + ox] = sum;
            }
        }
    }
}

// ReLU in-place
static void relu_inplace(float* data, int n) {
    for (int i = 0; i < n; i++)
        if (data[i] < 0) data[i] = 0;
}

// MaxPool2d(kernel=2, stride=1, ceil_mode=True) on padded input
static void maxpool2d_k2s1_ceil(const float* in, float* out,
                                 int ch, int H, int W) {
    // With ceil_mode=True, output size = ceil((H - 2)/1) + 1 = H - 1
    // But since input is already padded (0,1,0,1), the effective H/W is H+1
    // Actually: MaxPool on the already-padded tensor.
    // Output: ceil((H - 2) / 1) + 1 = H - 1
    int out_H = H - 1;
    int out_W = W - 1;
    // With ceil_mode, output = ceil((H + 2*pad - kh) / stride) + 1
    // pad=0, kh=2, stride=1, ceil_mode: out = ceil((H-2)/1) + 1 = H-1
    // But Texo uses ceil_mode=True which can add an extra row:
    // PyTorch: out = floor((H + 2*pad - kh) / stride) + 1, but with ceil_mode
    // out = ceil((H + 2*pad - kh) / stride) + 1
    // For H=193, kh=2, stride=1, pad=0: out = ceil(191/1) + 1 = 192
    out_H = (H - 2 + 1 - 1) / 1 + 1;  // ceil div
    out_W = (W - 2 + 1 - 1) / 1 + 1;

    for (int c = 0; c < ch; c++) {
        for (int oy = 0; oy < out_H; oy++) {
            for (int ox = 0; ox < out_W; ox++) {
                float maxv = -1e30f;
                for (int ky = 0; ky < 2; ky++) {
                    for (int kx = 0; kx < 2; kx++) {
                        int iy = oy + ky;
                        int ix = ox + kx;
                        if (iy < H && ix < W) {
                            float v = in[c * H * W + iy * W + ix];
                            if (v > maxv) maxv = v;
                        }
                    }
                }
                out[c * out_H * out_W + oy * out_W + ox] = maxv;
            }
        }
    }
}

// Pad tensor: add right_pad columns and bottom_pad rows (zero-pad)
static std::vector<float> pad_tensor(const float* in, int ch, int H, int W,
                                      int top, int bottom, int left, int right) {
    int new_H = H + top + bottom;
    int new_W = W + left + right;
    std::vector<float> out(ch * new_H * new_W, 0.0f);
    for (int c = 0; c < ch; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                out[c * new_H * new_W + (y + top) * new_W + (x + left)] =
                    in[c * H * W + y * W + x];
    return out;
}

// Apply ConvBNAct: conv (folded weights) + ReLU
// Returns output dimensions via out_H, out_W
static std::vector<float> apply_conv(const conv_layer& cl,
                                      const float* in, int in_ch, int H, int W,
                                      int out_ch, int kh, int kw, int stride,
                                      int groups, bool use_relu,
                                      int& out_H, int& out_W) {
    auto wv = to_f32(cl.w);
    auto bv = to_f32(cl.b);
    int pad = (kh - 1) / 2;
    out_H = (H + 2 * pad - kh) / stride + 1;
    out_W = (W + 2 * pad - kw) / stride + 1;
    std::vector<float> out(out_ch * out_H * out_W);
    conv2d_cpu(in, out.data(), wv.data(), bv.data(),
               in_ch, out_ch, H, W, kh, kw, stride, pad, groups);
    if (use_relu) relu_inplace(out.data(), (int)out.size());
    return out;
}

// Run the HGNetv2 encoder on CHW float input
static void run_encoder(ppformulanet_ocr_context* ctx,
                         const float* rgb, int H, int W) {
    // rgb is [3, H, W] in CHW format, normalized

    fprintf(stderr, "ppfn: encoder start (%dx%d)\n", W, H);

    // ---- StemBlock ----
    // stem1: Conv(3→32, k=3, s=2) + ReLU
    int oH, oW;
    auto x = apply_conv(ctx->stem[0], rgb, 3, H, W, 32, 3, 3, 2, 1, true, oH, oW);
    int cur_ch = 32, cur_H = oH, cur_W = oW;
    fprintf(stderr, "ppfn: stem1 -> (%d, %d, %d)\n", cur_ch, cur_H, cur_W);

    // F.pad(x, (0,1,0,1)) before stem2a
    auto padded = pad_tensor(x.data(), cur_ch, cur_H, cur_W, 0, 1, 0, 1);
    int pH = cur_H + 1, pW = cur_W + 1;

    // stem2a: Conv(32→16, k=2, s=1) + ReLU (padding=0 since k=2, pad=(2-1)/2=0)
    auto x2 = apply_conv(ctx->stem[1], padded.data(), 32, pH, pW, 16, 2, 2, 1, 1, true, oH, oW);
    fprintf(stderr, "ppfn: stem2a -> (%d, %d, %d)\n", 16, oH, oW);

    // F.pad(x2, (0,1,0,1)) before stem2b
    auto x2_padded = pad_tensor(x2.data(), 16, oH, oW, 0, 1, 0, 1);
    int x2pH = oH + 1, x2pW = oW + 1;

    // stem2b: Conv(16→32, k=2, s=1) + ReLU
    x2 = apply_conv(ctx->stem[2], x2_padded.data(), 16, x2pH, x2pW, 32, 2, 2, 1, 1, true, oH, oW);
    fprintf(stderr, "ppfn: stem2b -> (%d, %d, %d)\n", 32, oH, oW);

    // MaxPool2d(k=2, s=1, ceil_mode=True) on the padded stem1 output
    int pool_H = pH - 1, pool_W = pW - 1;
    std::vector<float> x1(32 * pool_H * pool_W);
    maxpool2d_k2s1_ceil(padded.data(), x1.data(), 32, pH, pW);
    // After pool with ceil_mode on [32, pH, pW]: output is [32, pH-1, pW-1]
    // which should equal stem2b output size
    fprintf(stderr, "ppfn: pool -> (%d, %d, %d)\n", 32, pool_H, pool_W);

    // Verify sizes match
    if (pool_H != oH || pool_W != oW) {
        fprintf(stderr, "ppfn: WARNING: pool (%d,%d) != stem2b (%d,%d), adjusting\n",
                pool_H, pool_W, oH, oW);
    }

    // Concatenate [pool, stem2b] on channel dim → [64, H, W]
    int cat_ch = 64;
    int cat_H = std::min(pool_H, oH), cat_W = std::min(pool_W, oW);
    std::vector<float> cat(cat_ch * cat_H * cat_W);
    for (int c = 0; c < 32; c++)
        for (int y = 0; y < cat_H; y++)
            for (int xi = 0; xi < cat_W; xi++)
                cat[c * cat_H * cat_W + y * cat_W + xi] =
                    x1[c * pool_H * pool_W + y * pool_W + xi];
    for (int c = 0; c < 32; c++)
        for (int y = 0; y < cat_H; y++)
            for (int xi = 0; xi < cat_W; xi++)
                cat[(c + 32) * cat_H * cat_W + y * cat_W + xi] =
                    x2[c * oH * oW + y * oW + xi];

    cur_ch = 64; cur_H = cat_H; cur_W = cat_W;
    fprintf(stderr, "ppfn: concat -> (%d, %d, %d)\n", cur_ch, cur_H, cur_W);

    // stem3: Conv(64→32, k=3, s=2) + ReLU
    x = apply_conv(ctx->stem[3], cat.data(), 64, cur_H, cur_W, 32, 3, 3, 2, 1, true, oH, oW);
    cur_ch = 32; cur_H = oH; cur_W = oW;
    fprintf(stderr, "ppfn: stem3 -> (%d, %d, %d)\n", cur_ch, cur_H, cur_W);

    // stem4: Conv(32→48, k=1, s=1) + ReLU
    x = apply_conv(ctx->stem[4], x.data(), 32, cur_H, cur_W, 48, 1, 1, 1, 1, true, oH, oW);
    cur_ch = 48; cur_H = oH; cur_W = oW;
    fprintf(stderr, "ppfn: stem4 -> (%d, %d, %d)\n", cur_ch, cur_H, cur_W);

    // ---- Stages ----
    for (int si = 0; si < 4; si++) {
        auto& st = ctx->stages[si];

        // Downsample (depthwise conv, no ReLU)
        if (st.has_downsample) {
            auto dw = to_f32(st.downsample.w);
            auto db = to_f32(st.downsample.b);
            int ds_kh = 3, ds_kw = 3;
            int ds_pad = 1;
            int ds_oH = (cur_H + 2 * ds_pad - ds_kh) / 2 + 1;
            int ds_oW = (cur_W + 2 * ds_pad - ds_kw) / 2 + 1;
            std::vector<float> ds_out(cur_ch * ds_oH * ds_oW);
            conv2d_cpu(x.data(), ds_out.data(), dw.data(), db.data(),
                       cur_ch, cur_ch, cur_H, cur_W, ds_kh, ds_kw, 2, ds_pad, cur_ch);
            // No ReLU for downsample (use_act=False)
            x = std::move(ds_out);
            cur_H = ds_oH; cur_W = ds_oW;
            fprintf(stderr, "ppfn: stage%d downsample -> (%d, %d, %d)\n", si, cur_ch, cur_H, cur_W);
        }

        // HG_Blocks
        for (int bi = 0; bi < st.n_blocks; bi++) {
            auto& blk = st.blocks[bi];
            bool residual = (bi > 0);
            std::vector<float> identity;
            if (residual) identity = x;

            // Collect outputs for concatenation
            std::vector<std::vector<float>> outputs;
            outputs.push_back(x);  // input to block

            int layer_in_ch = cur_ch;
            int layer_out_ch = st.mid_ch;

            for (int li = 0; li < st.n_layers; li++) {
                auto& lay = blk.layers[li];
                int in_c = (li == 0) ? layer_in_ch : layer_out_ch;

                if (lay.is_light) {
                    // LightConvBNAct: conv1 (1×1 pointwise, no ReLU) + conv2 (depthwise + ReLU)
                    auto c1 = apply_conv(lay.conv1, x.data(), in_c, cur_H, cur_W,
                                          layer_out_ch, 1, 1, 1, 1, false, oH, oW);
                    x = apply_conv(lay.conv2, c1.data(), layer_out_ch, oH, oW,
                                    layer_out_ch, st.kernel_size, st.kernel_size, 1,
                                    layer_out_ch, true, oH, oW);
                } else {
                    // ConvBNAct: single conv + ReLU
                    x = apply_conv(lay.conv, x.data(), in_c, cur_H, cur_W,
                                    layer_out_ch, st.kernel_size, st.kernel_size, 1, 1, true, oH, oW);
                }
                cur_H = oH; cur_W = oW;
                outputs.push_back(x);
            }

            // Concatenate all outputs on channel dim
            int total_ch = 0;
            for (auto& o : outputs) total_ch += (int)(o.size() / (cur_H * cur_W));
            std::vector<float> concat(total_ch * cur_H * cur_W);
            int ch_offset = 0;
            for (auto& o : outputs) {
                int och = (int)(o.size() / (cur_H * cur_W));
                for (int c = 0; c < och; c++)
                    memcpy(&concat[(ch_offset + c) * cur_H * cur_W],
                           &o[c * cur_H * cur_W],
                           cur_H * cur_W * sizeof(float));
                ch_offset += och;
            }

            // Aggregation: squeeze (1×1) + ReLU → excite (1×1) + ReLU
            int squeeze_ch = st.out_ch / 2;
            x = apply_conv(blk.agg_squeeze, concat.data(), total_ch, cur_H, cur_W,
                            squeeze_ch, 1, 1, 1, 1, true, oH, oW);
            x = apply_conv(blk.agg_excite, x.data(), squeeze_ch, cur_H, cur_W,
                            st.out_ch, 1, 1, 1, 1, true, oH, oW);
            cur_ch = st.out_ch;

            // Residual
            if (residual && !identity.empty()) {
                for (int i = 0; i < (int)x.size() && i < (int)identity.size(); i++)
                    x[i] += identity[i];
            }

            fprintf(stderr, "ppfn: stage%d block%d -> (%d, %d, %d)\n", si, bi, cur_ch, cur_H, cur_W);
        }
    }

    // Flatten: (2048, H, W) → (H*W, 2048) sequence
    int N = cur_H * cur_W;
    ctx->n_enc_tokens = N;
    ctx->enc_out.resize(N * cur_ch);

    // Transpose CHW → NHW (spatial positions as sequence)
    for (int n = 0; n < N; n++) {
        int y = n / cur_W;
        int xi = n % cur_W;
        for (int c = 0; c < cur_ch; c++)
            ctx->enc_out[n * cur_ch + c] = x[c * cur_H * cur_W + y * cur_W + xi];
    }

    fprintf(stderr, "ppfn: encoder output (%d, %d)\n", N, cur_ch);
}

// ---------------------------------------------------------------------------
// Projection + decoder
// ---------------------------------------------------------------------------

static void project_encoder(ppformulanet_ocr_context* ctx) {
    // Linear projection: enc_out [N, 2048] → proj_out [N, 384]
    const int N = ctx->n_enc_tokens;
    const int E = ctx->hparams.enc_hidden;
    const int D = ctx->hparams.dec_d_model;

    auto wv = to_f32(ctx->proj_w);
    auto bv = to_f32(ctx->proj_b);

    ctx->proj_out.resize(N * D);
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            float s = bv[d];
            for (int e = 0; e < E; e++)
                s += ctx->enc_out[n * E + e] * wv[d * E + e];
            ctx->proj_out[n * D + d] = s;
        }
    }

    fprintf(stderr, "ppfn: projected (%d, %d)\n", N, D);
}

static void precompute_cross_kv(ppformulanet_ocr_context* ctx) {
    const int N = ctx->n_enc_tokens;
    const int D = ctx->hparams.dec_d_model;
    const int n_dec = ctx->hparams.dec_layers;

    ctx->cross_k_cache.resize(n_dec);
    ctx->cross_v_cache.resize(n_dec);

    for (int li = 0; li < n_dec; li++) {
        ctx->cross_k_cache[li].resize(N * D);
        ctx->cross_v_cache[li].resize(N * D);

        for (int n = 0; n < N; n++) {
            linear_cpu(&ctx->proj_out[n * D], &ctx->cross_k_cache[li][n * D],
                       D, D, ctx->dec_layers[li].cross_k_w, ctx->dec_layers[li].cross_k_b);
            linear_cpu(&ctx->proj_out[n * D], &ctx->cross_v_cache[li][n * D],
                       D, D, ctx->dec_layers[li].cross_v_w, ctx->dec_layers[li].cross_v_b);
        }
    }

    fprintf(stderr, "ppfn: cross K/V cached (%d layers, %d tokens)\n", n_dec, N);
}

static std::vector<float> decoder_step(ppformulanet_ocr_context* ctx,
                                        int tok, int step,
                                        std::vector<std::vector<float>>& kv_k,
                                        std::vector<std::vector<float>>& kv_v) {
    const auto& hp = ctx->hparams;
    const int D = hp.dec_d_model;
    const int V = hp.vocab_size;
    const int n_enc = ctx->n_enc_tokens;

    std::vector<float> x(D);

    // Token embedding * sqrt(d_model) + position embedding
    auto emb = to_f32(ctx->tok_embed);
    auto pe  = to_f32(ctx->pos_embed_dec);
    float scale = sqrtf((float)D);

    if (tok >= 0 && tok < V) {
        for (int i = 0; i < D; i++)
            x[i] = emb[tok * D + i] * scale;
    }

    // MBart position offset: position_ids = step + 2
    int pos = step + 2;
    if (pos < (int)(pe.size() / D)) {
        for (int i = 0; i < D; i++)
            x[i] += pe[pos * D + i];
    }

    // Embedding LayerNorm
    static const bool dbg = (std::getenv("PPFN_DEBUG") != nullptr);
    if (dbg && step == 0) {
        fprintf(stderr, "ppfn: [dbg] tok_emb+pos first 5: %.5f %.5f %.5f %.5f %.5f\n",
                x[0], x[1], x[2], x[3], x[4]);
    }
    layernorm_cpu(x.data(), x.data(), D, ctx->embed_ln_w, ctx->embed_ln_b);
    if (dbg && step == 0) {
        fprintf(stderr, "ppfn: [dbg] after embed_ln first 5: %.5f %.5f %.5f %.5f %.5f\n",
                x[0], x[1], x[2], x[3], x[4]);
    }

    // Decoder layers (MBart PRE-LN: LN before attention, residual skips LN)
    for (int li = 0; li < hp.dec_layers; li++) {
        const auto& l = ctx->dec_layers[li];

        // --- Self-attention (PRE-LN) ---
        std::vector<float> residual(x.begin(), x.end());
        layernorm_cpu(x.data(), x.data(), D, l.self_ln_w, l.self_ln_b);

        std::vector<float> q(D), k(D), v(D);
        linear_cpu(x.data(), q.data(), D, D, l.self_q_w, l.self_q_b);
        linear_cpu(x.data(), k.data(), D, D, l.self_k_w, l.self_k_b);
        linear_cpu(x.data(), v.data(), D, D, l.self_v_w, l.self_v_b);

        // Append to KV cache
        kv_k[li].insert(kv_k[li].end(), k.begin(), k.end());
        kv_v[li].insert(kv_v[li].end(), v.begin(), v.end());
        int n_kv = (int)(kv_k[li].size() / D);

        std::vector<float> sa(D);
        mha_1q_cpu(q.data(), kv_k[li].data(), kv_v[li].data(),
                   sa.data(), n_kv, D, hp.dec_heads);

        std::vector<float> sa_proj(D);
        linear_cpu(sa.data(), sa_proj.data(), D, D, l.self_out_w, l.self_out_b);
        for (int i = 0; i < D; i++) x[i] = residual[i] + sa_proj[i];  // residual skips LN

        // --- Cross-attention (PRE-LN) ---
        residual.assign(x.begin(), x.end());
        layernorm_cpu(x.data(), x.data(), D, l.cross_ln_w, l.cross_ln_b);

        std::vector<float> cq(D);
        linear_cpu(x.data(), cq.data(), D, D, l.cross_q_w, l.cross_q_b);

        std::vector<float> ca(D);
        mha_1q_cpu(cq.data(), ctx->cross_k_cache[li].data(),
                   ctx->cross_v_cache[li].data(),
                   ca.data(), n_enc, D, hp.dec_heads);

        std::vector<float> ca_proj(D);
        linear_cpu(ca.data(), ca_proj.data(), D, D, l.cross_out_w, l.cross_out_b);
        for (int i = 0; i < D; i++) x[i] = residual[i] + ca_proj[i];  // residual skips LN

        // --- FFN (PRE-LN) ---
        residual.assign(x.begin(), x.end());
        layernorm_cpu(x.data(), x.data(), D, l.ff_ln_w, l.ff_ln_b);

        std::vector<float> ff_up(hp.dec_ffn_dim);
        linear_cpu(x.data(), ff_up.data(), D, hp.dec_ffn_dim, l.ff_up_w, l.ff_up_b);
        // GELU activation
        for (int i = 0; i < hp.dec_ffn_dim; i++) {
            float vi = ff_up[i];
            ff_up[i] = 0.5f * vi * (1.0f + tanhf(0.7978845608f * (vi + 0.044715f * vi * vi * vi)));
        }
        std::vector<float> ff_down(D);
        linear_cpu(ff_up.data(), ff_down.data(), hp.dec_ffn_dim, D, l.ff_down_w, l.ff_down_b);
        for (int i = 0; i < D; i++) x[i] = residual[i] + ff_down[i];  // residual skips LN
    }

    // Final LayerNorm
    if (dbg && step == 0) {
        fprintf(stderr, "ppfn: [dbg] before final_ln first 5: %.5f %.5f %.5f %.5f %.5f\n",
                x[0], x[1], x[2], x[3], x[4]);
    }
    layernorm_cpu(x.data(), x.data(), D, ctx->final_ln_w, ctx->final_ln_b);
    if (dbg && step == 0) {
        fprintf(stderr, "ppfn: [dbg] after final_ln first 5: %.5f %.5f %.5f %.5f %.5f\n",
                x[0], x[1], x[2], x[3], x[4]);
    }

    // LM head
    std::vector<float> logits(V);
    linear_cpu(x.data(), logits.data(), D, V, ctx->lm_head_w, ctx->lm_head_b);
    if (dbg && step == 0) {
        fprintf(stderr, "ppfn: [dbg] logits[0:5]: %.4f %.4f %.4f %.4f %.4f\n",
                logits[0], logits[1], logits[2], logits[3], logits[4]);
        if (V > 224) fprintf(stderr, "ppfn: [dbg] logits[91]=%f logits[224]=%f\n", logits[91], logits[224]);
    }

    return logits;
}

static std::vector<int> greedy_decode(ppformulanet_ocr_context* ctx) {
    const auto& hp = ctx->hparams;
    const int max_steps = std::min(hp.max_seq_len, 512);

    std::vector<int> tokens;
    int tok = hp.decoder_start_token;

    // Self-attention KV caches (per decoder layer)
    std::vector<std::vector<float>> kv_k(hp.dec_layers);
    std::vector<std::vector<float>> kv_v(hp.dec_layers);

    for (int step = 0; step < max_steps; step++) {
        auto logits = decoder_step(ctx, tok, step, kv_k, kv_v);

        // Greedy argmax
        int best = 0;
        float best_s = logits[0];
        for (int v = 1; v < hp.vocab_size; v++)
            if (logits[v] > best_s) { best_s = logits[v]; best = v; }

        if (step < 5) {
            fprintf(stderr, "ppfn: dec step %d: tok=%d best=%d (%.3f)\n",
                    step, tok, best, best_s);
        }

        if (best == hp.eos_token || best == hp.pad_token) break;
        tokens.push_back(best);
        tok = best;
    }

    return tokens;
}

// ---------------------------------------------------------------------------
// Init / Free / API
// ---------------------------------------------------------------------------

ppformulanet_ocr_context* ppformulanet_ocr_init(const char* model_path, int n_threads) {
    auto ctx = std::make_unique<ppformulanet_ocr_context>();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        fprintf(stderr, "ppfn: can't open %s\n", model_path);
        return nullptr;
    }

    auto& hp = ctx->hparams;
    hp.image_size  = core_gguf::kv_u32(gctx, "ppfn.encoder.image_size", 384);
    hp.enc_hidden  = core_gguf::kv_u32(gctx, "ppfn.encoder.hidden_size", 2048);
    hp.dec_layers  = core_gguf::kv_u32(gctx, "ppfn.decoder.decoder_layers", 2);
    hp.dec_heads   = core_gguf::kv_u32(gctx, "ppfn.decoder.decoder_attention_heads", 16);
    hp.dec_d_model = core_gguf::kv_u32(gctx, "ppfn.decoder.d_model", 384);
    hp.dec_ffn_dim = core_gguf::kv_u32(gctx, "ppfn.decoder.decoder_ffn_dim", 1536);
    hp.vocab_size  = core_gguf::kv_u32(gctx, "ppfn.decoder.vocab_size", 50000);
    hp.max_seq_len = core_gguf::kv_u32(gctx, "ppfn.decoder.max_position_embeddings", 1027);
    hp.cross_attn_dim = hp.dec_d_model;
    hp.bos_token   = core_gguf::kv_u32(gctx, "ppfn.decoder.bos_token_id", 0);
    hp.eos_token   = core_gguf::kv_u32(gctx, "ppfn.decoder.eos_token_id", 2);
    hp.pad_token   = core_gguf::kv_u32(gctx, "ppfn.decoder.pad_token_id", 1);
    hp.decoder_start_token = core_gguf::kv_u32(gctx, "ppfn.decoder.decoder_start_token_id", 0);

    // Read stage configs
    for (int si = 0; si < 4; si++) {
        char buf[128];
        auto& st = ctx->stages[si];
        snprintf(buf, sizeof(buf), "ppfn.encoder.stage%d.in_channels", si);
        st.in_ch = core_gguf::kv_u32(gctx, buf, 48);
        snprintf(buf, sizeof(buf), "ppfn.encoder.stage%d.mid_channels", si);
        st.mid_ch = core_gguf::kv_u32(gctx, buf, 48);
        snprintf(buf, sizeof(buf), "ppfn.encoder.stage%d.out_channels", si);
        st.out_ch = core_gguf::kv_u32(gctx, buf, 128);
        snprintf(buf, sizeof(buf), "ppfn.encoder.stage%d.n_blocks", si);
        st.n_blocks = core_gguf::kv_u32(gctx, buf, 1);
        snprintf(buf, sizeof(buf), "ppfn.encoder.stage%d.n_layers", si);
        st.n_layers = core_gguf::kv_u32(gctx, buf, 6);
        snprintf(buf, sizeof(buf), "ppfn.encoder.stage%d.kernel_size", si);
        st.kernel_size = core_gguf::kv_u32(gctx, buf, 3);
        snprintf(buf, sizeof(buf), "ppfn.encoder.stage%d.light_block", si);
        st.light_block = core_gguf::kv_u32(gctx, buf, 0) != 0;
    }

    ctx->vocab = core_gguf::kv_str_array(gctx, "tokenizer.tokens");
    core_gguf::free_metadata(gctx);

    fprintf(stderr, "ppfn: enc_hidden=%d dec=%dL/%dH/%d vocab=%d(%zu)\n",
            hp.enc_hidden, hp.dec_layers, hp.dec_heads, hp.dec_d_model,
            hp.vocab_size, ctx->vocab.size());

    ctx->backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);

    if (!core_gguf::load_weights(model_path, ctx->backend, "ppfn", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        return nullptr;
    }
    fprintf(stderr, "ppfn: %zu tensors loaded\n", ctx->wl.tensors.size());

    map_tensors(ctx.get());

    // Verify key tensors
    int n_mapped = 0;
    for (int si = 0; si < 4; si++)
        for (int bi = 0; bi < ctx->stages[si].n_blocks; bi++)
            if (ctx->stages[si].blocks[bi].agg_squeeze.w) n_mapped++;
    fprintf(stderr, "ppfn: %d blocks mapped, %d dec layers\n",
            n_mapped, hp.dec_layers);

    return ctx.release();
}

void ppformulanet_ocr_free(ppformulanet_ocr_context* ctx) {
    if (!ctx) return;
    if (ctx->backend) ggml_backend_free(ctx->backend);
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

const ppformulanet_ocr_hparams* ppformulanet_ocr_get_hparams(
    const ppformulanet_ocr_context* ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

const char* ppformulanet_ocr_recognize(ppformulanet_ocr_context* ctx,
                                        const float* pixels,
                                        int width, int height, int* out_len) {
    if (!ctx || !pixels) return nullptr;
    const int S = ctx->hparams.image_size;

    // UniMERNet preprocessing:
    // 1. Resize maintaining aspect ratio to fit in S×S
    // 2. Pad with black (0) to fill S×S
    // 3. Convert grayscale → 3ch (replicate)
    // 4. Normalize: mean=0.7931, std=0.1738
    const float MEAN = 0.7931f;
    const float STD  = 0.1738f;

    // Compute scale to fit within S×S preserving aspect ratio
    float scale = std::min((float)S / width, (float)S / height);
    int new_w = std::min((int)(width * scale), S);
    int new_h = std::min((int)(height * scale), S);
    int pad_left = (S - new_w) / 2;
    int pad_top  = (S - new_h) / 2;

    // Fill with normalized black: (0.0 - 0.7931) / 0.1738
    float black_norm = (0.0f - MEAN) / STD;
    std::vector<float> rgb(3 * S * S, black_norm);

    // Bilinear resize + normalize into the center of the padded image
    for (int y = 0; y < new_h; y++) {
        for (int x = 0; x < new_w; x++) {
            // Map back to source coordinates
            float src_x = (float)x / scale;
            float src_y = (float)y / scale;
            int ox = std::min((int)src_x, width - 1);
            int oy = std::min((int)src_y, height - 1);
            float v = pixels[oy * width + ox];  // grayscale [0, 1]
            float normed = (v - MEAN) / STD;
            int dy = pad_top + y;
            int dx = pad_left + x;
            for (int c = 0; c < 3; c++)
                rgb[c * S * S + dy * S + dx] = normed;
        }
    }

    // Run encoder
    run_encoder(ctx, rgb.data(), S, S);

    // Project to decoder dimension
    project_encoder(ctx);

    // Precompute cross-attention K/V
    precompute_cross_kv(ctx);

    // Decode
    auto tokens = greedy_decode(ctx);

    // Detokenize: replace GPT-2 BPE Ġ (U+0120, bytes C4 A0) with space
    ctx->result_buf.clear();
    for (int tok : tokens) {
        if (tok < 0 || tok >= (int)ctx->vocab.size()) continue;
        const auto& s = ctx->vocab[tok];
        for (size_t i = 0; i < s.size(); ) {
            if (i + 1 < s.size() &&
                (unsigned char)s[i] == 0xC4 && (unsigned char)s[i+1] == 0xA0) {
                ctx->result_buf += ' ';
                i += 2;
            } else {
                ctx->result_buf += s[i];
                i++;
            }
        }
    }

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const char* ppformulanet_ocr_recognize_raw(ppformulanet_ocr_context* ctx,
                                            const uint8_t* pixel_bytes,
                                            int width, int height, int channels,
                                            int* out_len) {
    if (!ctx || !pixel_bytes) return nullptr;

    // Convert to grayscale float [0, 1]
    std::vector<float> gray(width * height);
    for (int i = 0; i < width * height; i++) {
        if (channels == 1) {
            gray[i] = pixel_bytes[i] / 255.0f;
        } else if (channels == 3) {
            gray[i] = (0.299f * pixel_bytes[i*3] + 0.587f * pixel_bytes[i*3+1]
                       + 0.114f * pixel_bytes[i*3+2]) / 255.0f;
        } else if (channels == 4) {
            gray[i] = (0.299f * pixel_bytes[i*4] + 0.587f * pixel_bytes[i*4+1]
                       + 0.114f * pixel_bytes[i*4+2]) / 255.0f;
        }
    }

    return ppformulanet_ocr_recognize(ctx, gray.data(), width, height, out_len);
}

const char* ppformulanet_ocr_recognize_chw(ppformulanet_ocr_context* ctx,
                                            const float* chw_data,
                                            int* out_len) {
    if (!ctx || !chw_data) return nullptr;
    const int S = ctx->hparams.image_size;

    // Input is already preprocessed CHW [3, S, S] normalized data
    run_encoder(ctx, chw_data, S, S);
    project_encoder(ctx);
    precompute_cross_kv(ctx);
    auto tokens = greedy_decode(ctx);

    // Detokenize: replace GPT-2 BPE Ġ (U+0120, bytes C4 A0) with space
    ctx->result_buf.clear();
    for (int tok : tokens) {
        if (tok < 0 || tok >= (int)ctx->vocab.size()) continue;
        const auto& s = ctx->vocab[tok];
        for (size_t i = 0; i < s.size(); ) {
            if (i + 1 < s.size() &&
                (unsigned char)s[i] == 0xC4 && (unsigned char)s[i+1] == 0xA0) {
                ctx->result_buf += ' ';
                i += 2;
            } else {
                ctx->result_buf += s[i];
                i++;
            }
        }
    }

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const float* ppformulanet_ocr_get_encoder_output(
    const ppformulanet_ocr_context* ctx,
    int* out_n_tokens, int* out_hidden) {
    if (!ctx || ctx->enc_out.empty()) return nullptr;
    if (out_n_tokens) *out_n_tokens = ctx->n_enc_tokens;
    if (out_hidden) *out_hidden = ctx->hparams.enc_hidden;
    return ctx->enc_out.data();
}
