// bttr_ocr.cpp — BTTR Handwritten Math OCR via ggml.
//
// Architecture: DenseNet encoder + Transformer decoder.
// BTTR = "Bidirectionally Trained Transformer" (ICDAR 2021).
//
// Encoder: DenseNet (growth=24, 16 layers × 3 blocks, 1ch input, BN folded)
//          + Conv1×1 projection + LayerNorm + 2D sinusoidal pos encoding
// Decoder: nn.TransformerDecoder (3 layers, 8 heads, d=256, FFN=1024)
//          Post-LN, fused QKV, sinusoidal word pos encoding

#include "bttr_ocr.h"
#include "core/cpu_ops.h"
#include "core/gguf_loader.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>


#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

struct dense_layer_bttr {
    struct ggml_tensor * conv1_w;  // (bottleneck, in_ch) — BN folded, flattened 2D
    struct ggml_tensor * conv1_b;
    struct ggml_tensor * conv2_w;  // (growth, bottleneck*9) — BN folded, flattened 2D
    struct ggml_tensor * conv2_b;
};

struct transition_bttr {
    struct ggml_tensor * conv_w;   // BN folded
    struct ggml_tensor * conv_b;
};

struct decoder_layer_bttr {
    // Self-attention (fused QKV)
    struct ggml_tensor * sa_qkv_w;   // (768, 256) — Q,K,V packed
    struct ggml_tensor * sa_qkv_b;   // (768,)
    struct ggml_tensor * sa_out_w;   // (256, 256)
    struct ggml_tensor * sa_out_b;
    // Cross-attention (fused QKV)
    struct ggml_tensor * ca_qkv_w;   // (768, 256)
    struct ggml_tensor * ca_qkv_b;
    struct ggml_tensor * ca_out_w;
    struct ggml_tensor * ca_out_b;
    // FFN
    struct ggml_tensor * ff_up_w;    // (1024, 256)
    struct ggml_tensor * ff_up_b;
    struct ggml_tensor * ff_down_w;  // (256, 1024)
    struct ggml_tensor * ff_down_b;
    // LayerNorms (post-LN)
    struct ggml_tensor * ln1_w, * ln1_b;  // after self-attn
    struct ggml_tensor * ln2_w, * ln2_b;  // after cross-attn
    struct ggml_tensor * ln3_w, * ln3_b;  // after FFN
};

struct bttr_ocr_context {
    bttr_ocr_hparams hparams;

    // Encoder stem (conv1+norm1 folded)
    struct ggml_tensor * stem_conv_w;
    struct ggml_tensor * stem_conv_b;

    std::vector<dense_layer_bttr> block1, block2, block3;
    transition_bttr trans1, trans2;

    struct ggml_tensor * post_norm_scale;
    struct ggml_tensor * post_norm_offset;

    // Feature projection
    struct ggml_tensor * feat_proj_w;  // (256, 684)
    struct ggml_tensor * feat_proj_b;
    struct ggml_tensor * enc_ln_w;     // LayerNorm after projection
    struct ggml_tensor * enc_ln_b;

    // Decoder
    struct ggml_tensor * word_embed_w;
    struct ggml_tensor * word_embed_ln_w;
    struct ggml_tensor * word_embed_ln_b;
    struct ggml_tensor * pos_enc;       // (500, 256) sinusoidal
    struct ggml_tensor * proj_w;        // (113, 256) output projection
    struct ggml_tensor * proj_b;

    std::vector<decoder_layer_bttr> dec_layers;

    // Tokenizer
    std::vector<std::string> vocab;

    // 2D positional encoding cache
    std::vector<float> pe_cache;  // (n_pos * D) — PE values to add to encoder output
    int pe_cache_h = 0, pe_cache_w = 0;

    // Dequant cache
    core_cpu::DequantCache dequant_cache;

    // GGUF state
    core_gguf::WeightLoad wl;
    int n_threads;

    // Inference state
    std::string result_buf;
    std::vector<float> char_confidences; // per-token softmax probabilities
    std::vector<float> encoder_output; // (n_pos, d_model)
    int n_enc_pos;

    bool bench;
};

// ---------------------------------------------------------------------------
// Helpers — core_cpu shared + local unique ops
// ---------------------------------------------------------------------------

// Dequant: delegate to DequantCache (replaces per-context dequant_cache map)
static const float * tf32(bttr_ocr_context * ctx, struct ggml_tensor * t) {
    return ctx->dequant_cache.get(t);
}

// conv2d, relu, linear, layernorm → delegate to core_cpu (SIMD-accelerated)
static void conv2d(const float * in, int ic, int ih, int iw,
                   const float * W, const float * B,
                   int oc, int kH, int kW, int stride, int pad,
                   float * out, int /*oh*/, int /*ow*/) {
    core_cpu::conv2d_cpu(in, out, W, B, ic, oc, ih, iw, kH, kW, stride, pad);
}

static void relu_ip(float * d, int n) { core_cpu::relu_inplace(d, n); }

static void maxpool2d_ceil(const float * in, int ch, int ih, int iw,
                           int k, int s, float * out, int oh, int ow) {
    for (int c = 0; c < ch; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                float mx = -1e30f;
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++) {
                        int iy = y*s + ky, ix = x*s + kx;
                        if (iy < ih && ix < iw) {
                            float v = in[c*ih*iw + iy*iw + ix];
                            if (v > mx) mx = v;
                        }
                    }
                out[c*oh*ow + y*ow + x] = mx;
            }
}

static void avgpool2d_ceil(const float * in, int ch, int ih, int iw,
                           int k, int s, float * out, int oh, int ow) {
    for (int c = 0; c < ch; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                float sum = 0; int cnt = 0;
                for (int ky = 0; ky < k; ky++)
                    for (int kx = 0; kx < k; kx++) {
                        int iy = y*s + ky, ix = x*s + kx;
                        if (iy < ih && ix < iw) {
                            sum += in[c*ih*iw + iy*iw + ix];
                            cnt++;
                        }
                    }
                out[c*oh*ow + y*ow + x] = cnt > 0 ? sum / cnt : 0;
            }
}

static void apply_bn(float * d, int ch, int sp,
                     const float * scale, const float * offset) {
    for (int c = 0; c < ch; c++) {
        float s = scale[c], o = offset[c];
        for (int i = 0; i < sp; i++) d[c*sp + i] = d[c*sp + i] * s + o;
    }
}

static void layernorm(float * x, int d, const float * w, const float * b) {
    // in-place: use x as both input and output
    core_cpu::layernorm_cpu(x, x, d, w, b, 1e-5f);
}

static void linear(const float * x, int in_d,
                   const float * W, const float * B, int out_d,
                   float * out) {
    core_cpu::linear_cpu(x, out, in_d, out_d, W, B);
}

// ---------------------------------------------------------------------------
// Tensor mapping
// ---------------------------------------------------------------------------

static struct ggml_tensor * find(const std::map<std::string, ggml_tensor*> & m,
                                 const char * name) {
    auto it = m.find(name);
    return it != m.end() ? it->second : nullptr;
}

static bool map_tensors(bttr_ocr_context * ctx) {
    const auto & m = ctx->wl.tensors;
    char buf[256];
    const auto & hp = ctx->hparams;

    ctx->stem_conv_w = find(m, "enc.stem.conv.weight");
    ctx->stem_conv_b = find(m, "enc.stem.conv.bias");

    auto map_block = [&](int bi, std::vector<dense_layer_bttr> & layers) {
        layers.resize(hp.num_layers);
        for (int li = 0; li < hp.num_layers; li++) {
            auto & l = layers[li];
            auto T = [&](const char * s) {
                snprintf(buf, sizeof(buf), "enc.block%d.layer%d.%s", bi, li, s);
                return find(m, buf);
            };
            l.conv1_w = T("conv1.weight"); l.conv1_b = T("conv1.bias");
            l.conv2_w = T("conv2.weight"); l.conv2_b = T("conv2.bias");
        }
    };
    map_block(1, ctx->block1);
    map_block(2, ctx->block2);
    map_block(3, ctx->block3);

    ctx->trans1.conv_w = find(m, "enc.trans1.conv.weight");
    ctx->trans1.conv_b = find(m, "enc.trans1.conv.bias");
    ctx->trans2.conv_w = find(m, "enc.trans2.conv.weight");
    ctx->trans2.conv_b = find(m, "enc.trans2.conv.bias");

    ctx->post_norm_scale  = find(m, "enc.post_norm.scale");
    ctx->post_norm_offset = find(m, "enc.post_norm.offset");

    ctx->feat_proj_w = find(m, "enc.feature_proj.weight");
    ctx->feat_proj_b = find(m, "enc.feature_proj.bias");
    ctx->enc_ln_w    = find(m, "enc.norm.weight");
    ctx->enc_ln_b    = find(m, "enc.norm.bias");

    // Decoder
    ctx->word_embed_w    = find(m, "dec.word_embed.weight");
    ctx->word_embed_ln_w = find(m, "dec.word_embed_ln.weight");
    ctx->word_embed_ln_b = find(m, "dec.word_embed_ln.bias");
    ctx->pos_enc         = find(m, "dec.pos_enc");
    ctx->proj_w          = find(m, "dec.proj.weight");
    ctx->proj_b          = find(m, "dec.proj.bias");

    ctx->dec_layers.resize(hp.num_decoder_layers);
    for (int li = 0; li < hp.num_decoder_layers; li++) {
        auto & l = ctx->dec_layers[li];
        auto T = [&](const char * s) {
            snprintf(buf, sizeof(buf), "dec.layers.%d.%s", li, s);
            return find(m, buf);
        };
        l.sa_qkv_w = T("self_attn.in_proj_weight");
        l.sa_qkv_b = T("self_attn.in_proj_bias");
        l.sa_out_w = T("self_attn.out_proj.weight");
        l.sa_out_b = T("self_attn.out_proj.bias");
        l.ca_qkv_w = T("cross_attn.in_proj_weight");
        l.ca_qkv_b = T("cross_attn.in_proj_bias");
        l.ca_out_w = T("cross_attn.out_proj.weight");
        l.ca_out_b = T("cross_attn.out_proj.bias");
        l.ff_up_w  = T("ffn.up.weight");  l.ff_up_b  = T("ffn.up.bias");
        l.ff_down_w = T("ffn.down.weight"); l.ff_down_b = T("ffn.down.bias");
        l.ln1_w = T("norm1.weight"); l.ln1_b = T("norm1.bias");
        l.ln2_w = T("norm2.weight"); l.ln2_b = T("norm2.bias");
        l.ln3_w = T("norm3.weight"); l.ln3_b = T("norm3.bias");
    }

    if (!ctx->stem_conv_w || !ctx->word_embed_w || !ctx->proj_w) {
        fprintf(stderr, "bttr_ocr: missing critical tensors\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

bttr_ocr_context * bttr_ocr_init(const char * model_path, int n_threads) {
    auto ctx = std::make_unique<bttr_ocr_context>();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    gguf_context * gctx = core_gguf::open_metadata(model_path);
    if (!gctx) return nullptr;

    auto & hp = ctx->hparams;
    hp.growth_rate        = core_gguf::kv_u32(gctx, "bttr.encoder.growth_rate", 24);
    hp.num_layers         = core_gguf::kv_u32(gctx, "bttr.encoder.num_layers", 16);
    hp.input_channels     = core_gguf::kv_u32(gctx, "bttr.encoder.input_channels", 1);
    hp.d_model            = core_gguf::kv_u32(gctx, "bttr.decoder.d_model", 256);
    hp.nhead              = core_gguf::kv_u32(gctx, "bttr.decoder.nhead", 8);
    hp.num_decoder_layers = core_gguf::kv_u32(gctx, "bttr.decoder.num_layers", 3);
    hp.dim_feedforward    = core_gguf::kv_u32(gctx, "bttr.decoder.dim_feedforward", 1024);
    hp.vocab_size         = core_gguf::kv_u32(gctx, "bttr.decoder.vocab_size", 113);
    hp.max_len            = core_gguf::kv_u32(gctx, "bttr.decoder.max_len", 200);
    hp.pad_token          = core_gguf::kv_u32(gctx, "bttr.decoder.pad_token", 0);
    hp.sos_token          = core_gguf::kv_u32(gctx, "bttr.decoder.sos_token", 1);
    hp.eos_token          = core_gguf::kv_u32(gctx, "bttr.decoder.eos_token", 2);

    ctx->vocab = core_gguf::kv_str_array(gctx, "tokenizer.tokens");
    core_gguf::free_metadata(gctx);

    fprintf(stderr, "bttr_ocr: growth=%d layers=%d d=%d heads=%d dec=%d vocab=%d(%zu)\n",
            hp.growth_rate, hp.num_layers, hp.d_model, hp.nhead,
            hp.num_decoder_layers, hp.vocab_size, ctx->vocab.size());

    // Prefer GPU backend — weights read via ggml_backend_tensor_get (GPU-safe).
    bool force_cpu = (getenv("BTTR_OCR_FORCE_CPU") && atoi(getenv("BTTR_OCR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, backend, "bttr_ocr", ctx->wl)) {
        ggml_backend_free(backend);
        return nullptr;
    }
    if (!map_tensors(ctx.get())) return nullptr;

    ctx->bench = (std::getenv("CRISPEMBED_BTTR_BENCH") != nullptr);

    return ctx.release();
}

void bttr_ocr_free(bttr_ocr_context * ctx) {
    if (!ctx) return;
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

const bttr_ocr_hparams * bttr_ocr_get_hparams(const bttr_ocr_context * ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

// ---------------------------------------------------------------------------
// DenseNet encoder
// ---------------------------------------------------------------------------

static void run_encoder(bttr_ocr_context * ctx, const float * gray, int W, int H) {
    const auto & hp = ctx->hparams;
    const int gr = hp.growth_rate;     // 24
    const int bn_size = 4;             // bottleneck multiplier
    const int init_ch = 2 * gr;        // 48

    // Stem: Conv(1→48, 7×7, s=2, p=3) + BN(folded) + ReLU
    int h1 = (H + 2*3 - 7) / 2 + 1, w1 = (W + 2*3 - 7) / 2 + 1;
    std::vector<float> feat(init_ch * h1 * w1);
    conv2d(gray, 1, H, W, tf32(ctx, ctx->stem_conv_w), tf32(ctx, ctx->stem_conv_b),
           init_ch, 7, 7, 2, 3, feat.data(), h1, w1);
    relu_ip(feat.data(), init_ch * h1 * w1);

    // MaxPool(2, s=2, ceil_mode=True)
    int h2 = (h1 + 1) / 2, w2 = (w1 + 1) / 2;  // ceil_mode
    std::vector<float> pooled(init_ch * h2 * w2);
    maxpool2d_ceil(feat.data(), init_ch, h1, w1, 2, 2, pooled.data(), h2, w2);

    int cur_ch = init_ch, cur_h = h2, cur_w = w2;
    std::vector<float> features = std::move(pooled);

    // Dense blocks + transitions
    auto run_block = [&](const std::vector<dense_layer_bttr> & layers) {
        int sp = cur_h * cur_w;
        for (const auto & l : layers) {
            if (!l.conv1_w) continue;
            int bn_ch = bn_size * gr;  // 96

            // Conv1(cur_ch→96, 1×1) + BN(folded) + ReLU
            std::vector<float> bot(bn_ch * sp);
            // conv1_w is flattened to (96, cur_ch) — treat as 1×1 conv
            conv2d(features.data(), cur_ch, cur_h, cur_w,
                   tf32(ctx, l.conv1_w), tf32(ctx, l.conv1_b),
                   bn_ch, 1, 1, 1, 0, bot.data(), cur_h, cur_w);
            relu_ip(bot.data(), bn_ch * sp);

            // Conv2(96→24, 3×3) + BN(folded) + ReLU
            std::vector<float> nf(gr * sp);
            conv2d(bot.data(), bn_ch, cur_h, cur_w,
                   tf32(ctx, l.conv2_w), tf32(ctx, l.conv2_b),
                   gr, 3, 3, 1, 1, nf.data(), cur_h, cur_w);
            relu_ip(nf.data(), gr * sp);

            // Concat
            features.resize((cur_ch + gr) * sp);
            memcpy(features.data() + cur_ch * sp, nf.data(), gr * sp * sizeof(float));
            cur_ch += gr;
        }
    };

    auto run_trans = [&](const transition_bttr & t) {
        int sp = cur_h * cur_w;
        int out_ch = cur_ch / 2;  // reduction=0.5

        // Conv1×1(cur_ch→out_ch) + BN(folded) + ReLU
        std::vector<float> conv_out(out_ch * sp);
        conv2d(features.data(), cur_ch, cur_h, cur_w,
               tf32(ctx, t.conv_w), tf32(ctx, t.conv_b),
               out_ch, 1, 1, 1, 0, conv_out.data(), cur_h, cur_w);
        relu_ip(conv_out.data(), out_ch * sp);

        // AvgPool(2, s=2, ceil_mode=True)
        int oh = (cur_h + 1) / 2, ow = (cur_w + 1) / 2;
        std::vector<float> pool_out(out_ch * oh * ow);
        avgpool2d_ceil(conv_out.data(), out_ch, cur_h, cur_w,
                       2, 2, pool_out.data(), oh, ow);

        features = std::move(pool_out);
        cur_ch = out_ch; cur_h = oh; cur_w = ow;
    };

    // Block 1: 48 → 48+16*24 = 432
    run_block(ctx->block1);
    run_trans(ctx->trans1);  // 432 → 216

    // Block 2: 216 → 216+16*24 = 600
    run_block(ctx->block2);
    run_trans(ctx->trans2);  // 600 → 300

    // Block 3: 300 → 300+16*24 = 684
    run_block(ctx->block3);

    // Post-norm (BN as scale+offset)
    int sp = cur_h * cur_w;
    apply_bn(features.data(), cur_ch, sp,
             tf32(ctx, ctx->post_norm_scale), tf32(ctx, ctx->post_norm_offset));

    // Feature projection: Conv1×1(684→256) + ReLU
    const int D = hp.d_model;
    std::vector<float> proj(D * sp);
    conv2d(features.data(), cur_ch, cur_h, cur_w,
           tf32(ctx, ctx->feat_proj_w), tf32(ctx, ctx->feat_proj_b),
           D, 1, 1, 1, 0, proj.data(), cur_h, cur_w);
    relu_ip(proj.data(), D * sp);

    // Transpose CHW → HW×D
    int n_pos = cur_h * cur_w;
    ctx->encoder_output.resize(n_pos * D);
    for (int c = 0; c < D; c++)
        for (int i = 0; i < n_pos; i++)
            ctx->encoder_output[i * D + c] = proj[c * n_pos + i];

    // LayerNorm per position
    const float * ln_w = tf32(ctx, ctx->enc_ln_w);
    const float * ln_b = tf32(ctx, ctx->enc_ln_b);
    for (int i = 0; i < n_pos; i++)
        layernorm(ctx->encoder_output.data() + i * D, D, ln_w, ln_b);

    // 2D positional encoding (DETR-style, computed on-the-fly)
    // Matches PyTorch ImgPosEnc with normalize=True, scale=2*pi:
    //   inv_freq[i] = 1 / (10000 ^ (i / half_d))  for i = 0..half_d-1
    //   pos_x = stack(sin(x * inv_freq[0::2]), cos(x * inv_freq[1::2])).flatten
    //   pos_y = stack(sin(y * inv_freq[0::2]), cos(y * inv_freq[1::2])).flatten
    //   pos = cat(pos_x, pos_y)
    // Cache the PE buffer: it depends only on (cur_h, cur_w, D).
    {
        if (cur_h != ctx->pe_cache_h || cur_w != ctx->pe_cache_w) {
            // Recompute and store PE into cache.
            const int half_d = D / 2;           // 128
            const int quarter_d = half_d / 2;   // 64
            const float scale = 2.0f * (float)M_PI;

            // Precompute inv_freq: (half_d,) = 1/10000^(i/half_d)
            std::vector<float> inv_freq(half_d);
            for (int i = 0; i < half_d; i++)
                inv_freq[i] = 1.0f / powf(10000.0f, (float)i / (float)half_d);

            ctx->pe_cache.assign(n_pos * D, 0.0f);
            for (int y = 0; y < cur_h; y++) {
                for (int x = 0; x < cur_w; x++) {
                    // Normalized coordinates (cumsum-based, 1-indexed, then /max * scale)
                    float y_norm = scale * (float)(y + 1) / (float)cur_h;
                    float x_norm = scale * (float)(x + 1) / (float)cur_w;

                    int pos_idx = y * cur_w + x;
                    float * pe = ctx->pe_cache.data() + pos_idx * D;

                    // pos_x occupies pe[0..half_d-1]
                    // pos_x = stack(sin(x*inv_freq[even]), cos(x*inv_freq[odd])).flatten
                    // flatten of stack([A0,A1,...A63], [B0,B1,...B63], dim=-1)
                    // = [A0,B0, A1,B1, ..., A63,B63]
                    for (int i = 0; i < quarter_d; i++) {
                        float sx = x_norm * inv_freq[2 * i];      // even inv_freq → sin
                        float cx = x_norm * inv_freq[2 * i + 1];  // odd inv_freq → cos
                        pe[2 * i]     = sinf(sx);
                        pe[2 * i + 1] = cosf(cx);
                    }
                    // pos_y occupies pe[half_d..D-1]
                    for (int i = 0; i < quarter_d; i++) {
                        float sy = y_norm * inv_freq[2 * i];
                        float cy = y_norm * inv_freq[2 * i + 1];
                        pe[half_d + 2 * i]     = sinf(sy);
                        pe[half_d + 2 * i + 1] = cosf(cy);
                    }
                }
            }
            ctx->pe_cache_h = cur_h;
            ctx->pe_cache_w = cur_w;
        }

        // Add cached PE to encoder output.
        for (int i = 0; i < n_pos * D; i++)
            ctx->encoder_output[i] += ctx->pe_cache[i];
    }

    ctx->n_enc_pos = n_pos;
    fprintf(stderr, "bttr_ocr: encoder: (%d, %d, %d) → %d positions × %d\n",
            cur_ch, cur_h, cur_w, n_pos, D);

}

// ---------------------------------------------------------------------------
// Transformer decoder (greedy, single-step with KV cache)
// ---------------------------------------------------------------------------

static std::string greedy_decode(bttr_ocr_context * ctx) {
    ctx->char_confidences.clear();
    const auto & hp = ctx->hparams;
    const int D = hp.d_model;
    const int V = hp.vocab_size;
    const int n_enc = ctx->n_enc_pos;
    const int head_dim = D / hp.nhead;

    // KV caches for self-attention
    std::vector<std::vector<float>> kv_k(hp.num_decoder_layers);
    std::vector<std::vector<float>> kv_v(hp.num_decoder_layers);

    // Precompute cross-attention K/V from encoder output (constant across steps)
    std::vector<std::vector<float>> ca_ck(hp.num_decoder_layers);
    std::vector<std::vector<float>> ca_cv(hp.num_decoder_layers);
    for (int li = 0; li < hp.num_decoder_layers; li++) {
        const auto & l = ctx->dec_layers[li];
        const float * ca_W = tf32(ctx, l.ca_qkv_w);
        const float * ca_B = tf32(ctx, l.ca_qkv_b);
        const float * Wk = ca_W + D * D;
        const float * Bk = ca_B + D;
        const float * Wv = ca_W + 2 * D * D;
        const float * Bv = ca_B + 2 * D;
        ca_ck[li].resize(n_enc * D);
        ca_cv[li].resize(n_enc * D);
        for (int p = 0; p < n_enc; p++) {
            linear(ctx->encoder_output.data() + p * D, D,
                   Wk, Bk, D, ca_ck[li].data() + p * D);
            linear(ctx->encoder_output.data() + p * D, D,
                   Wv, Bv, D, ca_cv[li].data() + p * D);
        }
    }

    std::vector<int> tokens;
    int prev_token = hp.sos_token;

    for (int step = 0; step < hp.max_len; step++) {
        // Token embedding + LayerNorm
        std::vector<float> x(D);
        if (ctx->word_embed_w && prev_token >= 0 && prev_token < V) {
            const float * emb = tf32(ctx, ctx->word_embed_w);
            memcpy(x.data(), emb + prev_token * D, D * sizeof(float));
        }
        layernorm(x.data(), D,
                  tf32(ctx, ctx->word_embed_ln_w),
                  tf32(ctx, ctx->word_embed_ln_b));

        // Add positional encoding
        if (ctx->pos_enc && step < 500) {
            const float * pe = tf32(ctx, ctx->pos_enc);
            for (int i = 0; i < D; i++) x[i] += pe[step * D + i];
        }

        // Decoder layers
        for (int li = 0; li < hp.num_decoder_layers; li++) {
            const auto & l = ctx->dec_layers[li];

            // --- Causal self-attention ---
            // Fused QKV: in_proj_weight = [Wq; Wk; Wv], each (D, D)
            std::vector<float> qkv(3 * D);
            linear(x.data(), D, tf32(ctx, l.sa_qkv_w), tf32(ctx, l.sa_qkv_b), 3*D, qkv.data());
            float * q = qkv.data(), * k = qkv.data() + D, * v = qkv.data() + 2*D;

            // Append K, V to cache
            kv_k[li].insert(kv_k[li].end(), k, k + D);
            kv_v[li].insert(kv_v[li].end(), v, v + D);
            int n_past = step + 1;

            // Multi-head attention
            std::vector<float> attn_out(D, 0.0f);
            float scale = 1.0f / sqrtf((float)head_dim);
            for (int h = 0; h < hp.nhead; h++) {
                std::vector<float> scores(n_past);
                float max_s = -1e9f;
                for (int ki = 0; ki < n_past; ki++) {
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++)
                        dot += q[h*head_dim + d] * kv_k[li][ki*D + h*head_dim + d];
                    scores[ki] = dot * scale;
                    if (scores[ki] > max_s) max_s = scores[ki];
                }
                float sum_exp = 0;
                for (int ki = 0; ki < n_past; ki++) {
                    scores[ki] = expf(scores[ki] - max_s);
                    sum_exp += scores[ki];
                }
                for (int ki = 0; ki < n_past; ki++) scores[ki] /= sum_exp;
                for (int d = 0; d < head_dim; d++) {
                    float s = 0;
                    for (int ki = 0; ki < n_past; ki++)
                        s += scores[ki] * kv_v[li][ki*D + h*head_dim + d];
                    attn_out[h*head_dim + d] = s;
                }
            }
            // Output projection + residual
            std::vector<float> sa_proj(D);
            linear(attn_out.data(), D, tf32(ctx, l.sa_out_w), tf32(ctx, l.sa_out_b), D, sa_proj.data());
            for (int i = 0; i < D; i++) x[i] += sa_proj[i];
            // Post-LN
            layernorm(x.data(), D, tf32(ctx, l.ln1_w), tf32(ctx, l.ln1_b));

            // --- Cross-attention ---
            // Q from decoder, K/V from encoder (fused QKV but only Q uses decoder state)
            // Actually, nn.TransformerDecoderLayer's multihead_attn uses:
            //   Q = decoder_hidden, K = V = encoder_output
            // But with fused in_proj_weight, we need to split:
            //   Q = x @ Wq + bq (first D rows)
            //   K = enc @ Wk + bk (next D rows)
            //   V = enc @ Wv + bv (last D rows)
            {
                // Q from decoder state (first D rows of fused QKV)
                const float * ca_W = tf32(ctx, l.ca_qkv_w);
                const float * ca_B = tf32(ctx, l.ca_qkv_b);
                std::vector<float> cq(D);
                linear(x.data(), D, ca_W, ca_B, D, cq.data());

                // K, V from precomputed cache
                const float * ck = ca_ck[li].data();
                const float * cv = ca_cv[li].data();

                // MHA
                std::vector<float> ca_out(D, 0.0f);
                for (int h = 0; h < hp.nhead; h++) {
                    std::vector<float> scores(n_enc);
                    float max_s = -1e9f;
                    for (int ki = 0; ki < n_enc; ki++) {
                        float dot = 0;
                        for (int d = 0; d < head_dim; d++)
                            dot += cq[h*head_dim + d] * ck[ki*D + h*head_dim + d];
                        scores[ki] = dot * scale;
                        if (scores[ki] > max_s) max_s = scores[ki];
                    }
                    float sum_exp = 0;
                    for (auto & s : scores) { s = expf(s - max_s); sum_exp += s; }
                    for (auto & s : scores) s /= sum_exp;
                    for (int d = 0; d < head_dim; d++) {
                        float s = 0;
                        for (int ki = 0; ki < n_enc; ki++)
                            s += scores[ki] * cv[ki*D + h*head_dim + d];
                        ca_out[h*head_dim + d] = s;
                    }
                }
                std::vector<float> ca_proj(D);
                linear(ca_out.data(), D, tf32(ctx, l.ca_out_w), tf32(ctx, l.ca_out_b), D, ca_proj.data());
                for (int i = 0; i < D; i++) x[i] += ca_proj[i];
                layernorm(x.data(), D, tf32(ctx, l.ln2_w), tf32(ctx, l.ln2_b));
            }

            // --- FFN ---
            {
                const int F = hp.dim_feedforward;
                std::vector<float> up(F);
                linear(x.data(), D, tf32(ctx, l.ff_up_w), tf32(ctx, l.ff_up_b), F, up.data());
                relu_ip(up.data(), F);
                std::vector<float> down(D);
                linear(up.data(), F, tf32(ctx, l.ff_down_w), tf32(ctx, l.ff_down_b), D, down.data());
                for (int i = 0; i < D; i++) x[i] += down[i];
                layernorm(x.data(), D, tf32(ctx, l.ln3_w), tf32(ctx, l.ln3_b));
            }
        }

        // Output projection
        std::vector<float> logits(V);
        linear(x.data(), D, tf32(ctx, ctx->proj_w), tf32(ctx, ctx->proj_b), V, logits.data());

        // Argmax
        int best = 0;
        float best_score = logits[0];
        for (int v = 1; v < V; v++)
            if (logits[v] > best_score) { best_score = logits[v]; best = v; }

        if (best == hp.eos_token || best == hp.pad_token) break;

        // Confidence: softmax of winning token
        {
            float max_l = logits[0];
            for (int v = 1; v < V; v++) if (logits[v] > max_l) max_l = logits[v];
            float sum_e = 0;
            for (int v = 0; v < V; v++) sum_e += expf(logits[v] - max_l);
            ctx->char_confidences.push_back(expf(logits[best] - max_l) / sum_e);
        }

        tokens.push_back(best);
        prev_token = best;
    }

    // Detokenize
    std::string result;
    for (int tok : tokens) {
        if (tok >= 0 && tok < (int)ctx->vocab.size()) {
            if (!result.empty()) result += ' ';
            result += ctx->vocab[tok];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Beam search decoder
// ---------------------------------------------------------------------------

struct Beam {
    std::vector<int> tokens;
    float score;      // log-probability sum
    int prev_token;
    bool finished;
    // Per-layer self-attention KV caches
    std::vector<std::vector<float>> kv_k;
    std::vector<std::vector<float>> kv_v;
};

// Run one decoder step for a single beam, returning logits.
// Modifies beam's KV caches in place.
static void decode_step(bttr_ocr_context * ctx,
                        Beam & beam, int step,
                        const std::vector<std::vector<float>> & ca_ck,
                        const std::vector<std::vector<float>> & ca_cv,
                        std::vector<float> & logits_out) {
    const auto & hp = ctx->hparams;
    const int D = hp.d_model;
    const int V = hp.vocab_size;
    const int n_enc = ctx->n_enc_pos;
    const int head_dim = D / hp.nhead;
    const float scale = 1.0f / sqrtf((float)head_dim);

    // Token embedding + LayerNorm
    std::vector<float> x(D);
    if (ctx->word_embed_w && beam.prev_token >= 0 && beam.prev_token < V) {
        const float * emb = tf32(ctx, ctx->word_embed_w);
        memcpy(x.data(), emb + beam.prev_token * D, D * sizeof(float));
    }
    layernorm(x.data(), D, tf32(ctx, ctx->word_embed_ln_w),
              tf32(ctx, ctx->word_embed_ln_b));

    if (ctx->pos_enc && step < 500) {
        const float * pe = tf32(ctx, ctx->pos_enc);
        for (int i = 0; i < D; i++) x[i] += pe[step * D + i];
    }

    for (int li = 0; li < hp.num_decoder_layers; li++) {
        const auto & l = ctx->dec_layers[li];

        // Self-attention
        std::vector<float> qkv(3 * D);
        linear(x.data(), D, tf32(ctx, l.sa_qkv_w), tf32(ctx, l.sa_qkv_b),
               3 * D, qkv.data());
        float * q = qkv.data(), * k = qkv.data() + D, * v = qkv.data() + 2*D;

        beam.kv_k[li].insert(beam.kv_k[li].end(), k, k + D);
        beam.kv_v[li].insert(beam.kv_v[li].end(), v, v + D);
        int n_past = step + 1;

        std::vector<float> attn_out(D, 0.0f);
        for (int h = 0; h < hp.nhead; h++) {
            std::vector<float> scores(n_past);
            float max_s = -1e9f;
            for (int ki = 0; ki < n_past; ki++) {
                float dot = 0;
                for (int d = 0; d < head_dim; d++)
                    dot += q[h*head_dim+d] * beam.kv_k[li][ki*D+h*head_dim+d];
                scores[ki] = dot * scale;
                if (scores[ki] > max_s) max_s = scores[ki];
            }
            float sum_exp = 0;
            for (auto & s : scores) { s = expf(s - max_s); sum_exp += s; }
            for (auto & s : scores) s /= sum_exp;
            for (int d = 0; d < head_dim; d++) {
                float s = 0;
                for (int ki = 0; ki < n_past; ki++)
                    s += scores[ki] * beam.kv_v[li][ki*D+h*head_dim+d];
                attn_out[h*head_dim+d] = s;
            }
        }
        std::vector<float> sa_proj(D);
        linear(attn_out.data(), D, tf32(ctx, l.sa_out_w),
               tf32(ctx, l.sa_out_b), D, sa_proj.data());
        for (int i = 0; i < D; i++) x[i] += sa_proj[i];
        layernorm(x.data(), D, tf32(ctx, l.ln1_w), tf32(ctx, l.ln1_b));

        // Cross-attention
        {
            const float * ca_W = tf32(ctx, l.ca_qkv_w);
            const float * ca_B = tf32(ctx, l.ca_qkv_b);
            std::vector<float> cq(D);
            linear(x.data(), D, ca_W, ca_B, D, cq.data());
            const float * ck = ca_ck[li].data();
            const float * cv = ca_cv[li].data();
            std::vector<float> ca_out(D, 0.0f);
            for (int h = 0; h < hp.nhead; h++) {
                std::vector<float> scores(n_enc);
                float max_s = -1e9f;
                for (int ki = 0; ki < n_enc; ki++) {
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++)
                        dot += cq[h*head_dim+d] * ck[ki*D+h*head_dim+d];
                    scores[ki] = dot * scale;
                    if (scores[ki] > max_s) max_s = scores[ki];
                }
                float sum_exp = 0;
                for (auto & s : scores) { s = expf(s - max_s); sum_exp += s; }
                for (auto & s : scores) s /= sum_exp;
                for (int d = 0; d < head_dim; d++) {
                    float s = 0;
                    for (int ki = 0; ki < n_enc; ki++)
                        s += scores[ki] * cv[ki*D+h*head_dim+d];
                    ca_out[h*head_dim+d] = s;
                }
            }
            std::vector<float> ca_proj(D);
            linear(ca_out.data(), D, tf32(ctx, l.ca_out_w),
                   tf32(ctx, l.ca_out_b), D, ca_proj.data());
            for (int i = 0; i < D; i++) x[i] += ca_proj[i];
            layernorm(x.data(), D, tf32(ctx, l.ln2_w), tf32(ctx, l.ln2_b));
        }

        // FFN
        {
            const int F = hp.dim_feedforward;
            std::vector<float> up(F);
            linear(x.data(), D, tf32(ctx, l.ff_up_w),
                   tf32(ctx, l.ff_up_b), F, up.data());
            relu_ip(up.data(), F);
            std::vector<float> down(D);
            linear(up.data(), F, tf32(ctx, l.ff_down_w),
                   tf32(ctx, l.ff_down_b), D, down.data());
            for (int i = 0; i < D; i++) x[i] += down[i];
            layernorm(x.data(), D, tf32(ctx, l.ln3_w), tf32(ctx, l.ln3_b));
        }
    }

    logits_out.resize(V);
    linear(x.data(), D, tf32(ctx, ctx->proj_w), tf32(ctx, ctx->proj_b),
           V, logits_out.data());
}

static std::string beam_decode(bttr_ocr_context * ctx, int beam_width) {
    const auto & hp = ctx->hparams;
    const int D = hp.d_model;
    const int n_enc = ctx->n_enc_pos;

    // Precompute cross-attention K/V (shared across all beams)
    std::vector<std::vector<float>> ca_ck(hp.num_decoder_layers);
    std::vector<std::vector<float>> ca_cv(hp.num_decoder_layers);
    for (int li = 0; li < hp.num_decoder_layers; li++) {
        const auto & l = ctx->dec_layers[li];
        const float * ca_W = tf32(ctx, l.ca_qkv_w);
        const float * ca_B = tf32(ctx, l.ca_qkv_b);
        const float * Wk = ca_W + D * D;
        const float * Bk = ca_B + D;
        const float * Wv = ca_W + 2 * D * D;
        const float * Bv = ca_B + 2 * D;
        ca_ck[li].resize(n_enc * D);
        ca_cv[li].resize(n_enc * D);
        for (int p = 0; p < n_enc; p++) {
            linear(ctx->encoder_output.data() + p * D, D,
                   Wk, Bk, D, ca_ck[li].data() + p * D);
            linear(ctx->encoder_output.data() + p * D, D,
                   Wv, Bv, D, ca_cv[li].data() + p * D);
        }
    }

    // Initialize beams
    std::vector<Beam> beams(1);
    beams[0].score = 0.0f;
    beams[0].prev_token = hp.sos_token;
    beams[0].finished = false;
    beams[0].kv_k.resize(hp.num_decoder_layers);
    beams[0].kv_v.resize(hp.num_decoder_layers);

    std::vector<Beam> completed;

    for (int step = 0; step < hp.max_len; step++) {
        struct Candidate {
            int beam_idx;
            int token;
            float score;
        };
        std::vector<Candidate> candidates;

        for (int bi = 0; bi < (int)beams.size(); bi++) {
            if (beams[bi].finished) continue;

            std::vector<float> logits;
            decode_step(ctx, beams[bi], step, ca_ck, ca_cv, logits);

            // Log-softmax
            float max_l = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0;
            for (auto & l : logits) { l = expf(l - max_l); sum_exp += l; }
            float log_sum = logf(sum_exp) + max_l;
            for (int v = 0; v < hp.vocab_size; v++) {
                float log_p = logits[v] > 0 ? logf(logits[v]) + max_l - log_sum : -100.0f;
                candidates.push_back({bi, v, beams[bi].score + log_p});
            }
        }

        if (candidates.empty()) break;

        // Sort by score descending, keep top beam_width
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate & a, const Candidate & b) {
                      return a.score > b.score;
                  });

        int keep = std::min(beam_width, (int)candidates.size());
        std::vector<Beam> new_beams;
        new_beams.reserve(keep);

        for (int i = 0; i < keep; i++) {
            const auto & c = candidates[i];
            Beam nb;
            nb.tokens = beams[c.beam_idx].tokens;
            nb.kv_k = beams[c.beam_idx].kv_k;
            nb.kv_v = beams[c.beam_idx].kv_v;
            nb.score = c.score;

            if (c.token == hp.eos_token || c.token == hp.pad_token) {
                nb.finished = true;
                nb.prev_token = c.token;
                // Length-normalize score
                float len = (float)nb.tokens.size();
                nb.score = nb.score / (len > 0 ? len : 1.0f);
                completed.push_back(std::move(nb));
            } else {
                nb.tokens.push_back(c.token);
                nb.prev_token = c.token;
                nb.finished = false;
                new_beams.push_back(std::move(nb));
            }
        }

        beams = std::move(new_beams);
        if (beams.empty()) break;
        if ((int)completed.size() >= beam_width) break;
    }

    // Add any unfinished beams
    for (auto & b : beams) {
        float len = (float)b.tokens.size();
        b.score = b.score / (len > 0 ? len : 1.0f);
        completed.push_back(std::move(b));
    }

    // Pick best completed beam
    if (completed.empty()) return "";
    auto best = std::max_element(completed.begin(), completed.end(),
        [](const Beam & a, const Beam & b) { return a.score < b.score; });

    std::string result;
    for (int tok : best->tokens) {
        if (tok >= 0 && tok < (int)ctx->vocab.size()) {
            if (!result.empty()) result += ' ';
            result += ctx->vocab[tok];
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Scale image down to fit within max_pixels, preserving aspect ratio.
// Uses bilinear interpolation. Returns true if scaling was applied.
static bool scale_to_fit(const float * src, int sw, int sh,
                         std::vector<float> & dst, int & dw, int & dh,
                         int max_pixels = 100000) {
    if (sw * sh <= max_pixels) return false;
    float ratio = sqrtf((float)max_pixels / (sw * sh));
    dw = std::max(1, (int)(sw * ratio));
    dh = std::max(1, (int)(sh * ratio));
    dst.resize(dw * dh);
    for (int y = 0; y < dh; y++) {
        float sy = (y + 0.5f) * sh / dh - 0.5f;
        int y0 = std::max(0, std::min((int)sy, sh - 1));
        int y1 = std::min(y0 + 1, sh - 1);
        float fy = sy - y0;
        for (int x = 0; x < dw; x++) {
            float sx = (x + 0.5f) * sw / dw - 0.5f;
            int x0 = std::max(0, std::min((int)sx, sw - 1));
            int x1 = std::min(x0 + 1, sw - 1);
            float fx = sx - x0;
            dst[y * dw + x] =
                src[y0*sw+x0]*(1-fx)*(1-fy) + src[y0*sw+x1]*fx*(1-fy) +
                src[y1*sw+x0]*(1-fx)*fy     + src[y1*sw+x1]*fx*fy;
        }
    }
    return true;
}

const char * bttr_ocr_recognize(
    bttr_ocr_context * ctx,
    const float * pixels, int width, int height,
    int * out_len
) {
    if (!ctx || !pixels || width <= 0 || height <= 0) return nullptr;

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Auto-invert if needed
    auto t0 = std::chrono::steady_clock::now();
    const int n = width * height;
    float mean = 0;
    for (int i = 0; i < n; i++) mean += pixels[i];
    mean /= n;

    std::vector<float> work;
    const float * input = pixels;
    if (mean > 0.5f) {
        work.resize(n);
        for (int i = 0; i < n; i++) work[i] = 1.0f - pixels[i];
        input = work.data();
        fprintf(stderr, "bttr_ocr: auto-inverted (mean=%.2f)\n", mean);
    }

    // Scale down large images to fit within 100K pixels
    int w = width, h = height;
    std::vector<float> scaled;
    if (scale_to_fit(input, width, height, scaled, w, h)) {
        input = scaled.data();
        fprintf(stderr, "bttr_ocr: scaled %dx%d → %dx%d\n", width, height, w, h);
    }
    if (bench) fprintf(stderr, "[bttr-bench] preprocess: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    t0 = std::chrono::steady_clock::now();
    run_encoder(ctx, input, w, h);
    if (bench) fprintf(stderr, "[bttr-bench] encoder: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    t0 = std::chrono::steady_clock::now();
    ctx->result_buf = greedy_decode(ctx);
    if (bench) fprintf(stderr, "[bttr-bench] decoder greedy: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    if (bench) fprintf(stderr, "[bttr-bench] total: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_total).count());

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const char * bttr_ocr_recognize_beam(
    bttr_ocr_context * ctx,
    const float * pixels, int width, int height,
    int beam_width,
    int * out_len
) {
    if (!ctx || !pixels || width <= 0 || height <= 0) return nullptr;
    if (beam_width <= 1) return bttr_ocr_recognize(ctx, pixels, width, height, out_len);

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    auto t0 = std::chrono::steady_clock::now();
    const int n = width * height;
    float mean = 0;
    for (int i = 0; i < n; i++) mean += pixels[i];
    mean /= n;

    std::vector<float> work;
    const float * input = pixels;
    if (mean > 0.5f) {
        work.resize(n);
        for (int i = 0; i < n; i++) work[i] = 1.0f - pixels[i];
        input = work.data();
    }

    int w = width, h = height;
    std::vector<float> scaled;
    if (scale_to_fit(input, width, height, scaled, w, h)) {
        input = scaled.data();
    }
    if (bench) fprintf(stderr, "[bttr-bench] preprocess: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    t0 = std::chrono::steady_clock::now();
    run_encoder(ctx, input, w, h);
    if (bench) fprintf(stderr, "[bttr-bench] encoder: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    t0 = std::chrono::steady_clock::now();
    ctx->result_buf = beam_decode(ctx, beam_width);
    if (bench) fprintf(stderr, "[bttr-bench] decoder beam: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    if (bench) fprintf(stderr, "[bttr-bench] total: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_total).count());

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const char * bttr_ocr_recognize_raw(
    bttr_ocr_context * ctx,
    const uint8_t * pixel_bytes, int width, int height, int channels,
    int * out_len
) {
    if (!ctx || !pixel_bytes || width <= 0 || height <= 0) return nullptr;
    std::vector<float> gray(width * height);
    for (int i = 0; i < width * height; i++) {
        if (channels == 1) gray[i] = pixel_bytes[i] / 255.0f;
        else if (channels >= 3) {
            int b = i * channels;
            gray[i] = (0.299f*pixel_bytes[b] + 0.587f*pixel_bytes[b+1] + 0.114f*pixel_bytes[b+2]) / 255.0f;
        }
    }
    return bttr_ocr_recognize(ctx, gray.data(), width, height, out_len);
}

const float * bttr_ocr_confidences(const bttr_ocr_context * ctx, int * n_chars) {
    if (!ctx || ctx->char_confidences.empty()) {
        if (n_chars) *n_chars = 0;
        return nullptr;
    }
    if (n_chars) *n_chars = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float bttr_ocr_mean_confidence(const bttr_ocr_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float c : ctx->char_confidences) sum += c;
    return (float)(sum / ctx->char_confidences.size());
}
