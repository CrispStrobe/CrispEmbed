// posformer_ocr.cpp — PosFormer Handwritten Math OCR via ggml.
//
// Architecture: DenseNet encoder + Transformer decoder + ARM.
// PosFormer = BTTR + Attention Refinement Module (ARM) for coverage.
//
// Encoder: identical to BTTR (DenseNet, growth=24, 16 layers × 3 blocks)
// Decoder: same as BTTR + ARM between decoder layers
// ARM: Conv2d(2*nhead→dc, 5×5) + ReLU + Conv2d(dc→nhead, 1×1, BN-folded)
//      Applied to layers 1 and 2 (not layer 0) using coverage from previous layer.

#include "posformer_ocr.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Structures
// ---------------------------------------------------------------------------

struct dense_layer_pf {
    struct ggml_tensor * conv1_w;
    struct ggml_tensor * conv1_b;
    struct ggml_tensor * conv2_w;
    struct ggml_tensor * conv2_b;
};

struct transition_pf {
    struct ggml_tensor * conv_w;
    struct ggml_tensor * conv_b;
};

struct decoder_layer_pf {
    struct ggml_tensor * sa_qkv_w;
    struct ggml_tensor * sa_qkv_b;
    struct ggml_tensor * sa_out_w;
    struct ggml_tensor * sa_out_b;
    struct ggml_tensor * ca_qkv_w;
    struct ggml_tensor * ca_qkv_b;
    struct ggml_tensor * ca_out_w;
    struct ggml_tensor * ca_out_b;
    struct ggml_tensor * ff_up_w;
    struct ggml_tensor * ff_up_b;
    struct ggml_tensor * ff_down_w;
    struct ggml_tensor * ff_down_b;
    struct ggml_tensor * ln1_w, * ln1_b;
    struct ggml_tensor * ln2_w, * ln2_b;
    struct ggml_tensor * ln3_w, * ln3_b;
};

struct arm_weights {
    struct ggml_tensor * conv_w;   // (dc, 2*nhead, 5, 5) flattened to (dc, 2*nhead*25)
    struct ggml_tensor * conv_b;   // (dc,)
    struct ggml_tensor * proj_w;   // (nhead, dc, 1, 1) flattened to (nhead, dc)
    struct ggml_tensor * proj_b;   // (nhead,) — BN folded in
};

struct posformer_ocr_context {
    posformer_ocr_hparams hparams;

    struct ggml_tensor * stem_conv_w;
    struct ggml_tensor * stem_conv_b;

    std::vector<dense_layer_pf> block1, block2, block3;
    transition_pf trans1, trans2;

    struct ggml_tensor * post_norm_scale;
    struct ggml_tensor * post_norm_offset;

    struct ggml_tensor * feat_proj_w;
    struct ggml_tensor * feat_proj_b;
    struct ggml_tensor * enc_ln_w;
    struct ggml_tensor * enc_ln_b;

    struct ggml_tensor * word_embed_w;
    struct ggml_tensor * word_embed_ln_w;
    struct ggml_tensor * word_embed_ln_b;
    struct ggml_tensor * pos_enc;
    struct ggml_tensor * proj_w;
    struct ggml_tensor * proj_b;

    std::vector<decoder_layer_pf> dec_layers;
    arm_weights arm;

    std::vector<std::string> vocab;
    std::map<const void *, std::vector<float>> dequant_cache;

    core_gguf::WeightLoad wl;
    int n_threads;

    std::string result_buf;
    std::vector<float> encoder_output;
    int n_enc_pos;
    int enc_h, enc_w;  // spatial dims of encoder output (needed for ARM)
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// GPU-safe: uses ggml_backend_tensor_get instead of direct tensor->data access
static const float * tf32(posformer_ocr_context * ctx, struct ggml_tensor * t) {
    if (!t) return nullptr;
    auto it = ctx->dequant_cache.find(t->data);
    if (it != ctx->dequant_cache.end()) return it->second.data();
    const int64_t n = ggml_nelements(t);
    auto & buf = ctx->dequant_cache[t->data];
    buf.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    } else {
        size_t raw_sz = ggml_nbytes(t);
        std::vector<uint8_t> raw(raw_sz);
        ggml_backend_tensor_get(t, raw.data(), 0, raw_sz);
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits->to_float) traits->to_float(raw.data(), buf.data(), n);
        else std::fill(buf.begin(), buf.end(), 0.0f);
    }
    return buf.data();
}

static void conv2d(const float * in, int ic, int ih, int iw,
                   const float * W, const float * B,
                   int oc, int kH, int kW, int stride, int pad,
                   float * out, int oh, int ow) {
    for (int o = 0; o < oc; o++) {
        float b = B ? B[o] : 0.0f;
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                float sum = b;
                for (int c = 0; c < ic; c++)
                    for (int ky = 0; ky < kH; ky++)
                        for (int kx = 0; kx < kW; kx++) {
                            int iy = y * stride - pad + ky;
                            int ix = x * stride - pad + kx;
                            if (iy >= 0 && iy < ih && ix >= 0 && ix < iw)
                                sum += in[c*ih*iw + iy*iw + ix] *
                                       W[o*ic*kH*kW + c*kH*kW + ky*kW + kx];
                        }
                out[o*oh*ow + y*ow + x] = sum;
            }
    }
}

static void relu_ip(float * d, int n) {
    for (int i = 0; i < n; i++) if (d[i] < 0) d[i] = 0;
}

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
    float mean = 0, var = 0;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= d;
    for (int i = 0; i < d; i++) { float v = x[i] - mean; var += v*v; }
    var /= d;
    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < d; i++) x[i] = (x[i] - mean) * inv * w[i] + b[i];
}

static void linear(const float * x, int in_d,
                   const float * W, const float * B, int out_d,
                   float * out) {
    for (int o = 0; o < out_d; o++) {
        float s = B ? B[o] : 0.0f;
        for (int i = 0; i < in_d; i++) s += x[i] * W[o*in_d + i];
        out[o] = s;
    }
}

// ---------------------------------------------------------------------------
// Tensor mapping
// ---------------------------------------------------------------------------

static struct ggml_tensor * find(const std::map<std::string, ggml_tensor*> & m,
                                 const char * name) {
    auto it = m.find(name);
    return it != m.end() ? it->second : nullptr;
}

static bool map_tensors(posformer_ocr_context * ctx) {
    const auto & m = ctx->wl.tensors;
    char buf[256];
    const auto & hp = ctx->hparams;

    ctx->stem_conv_w = find(m, "enc.stem.conv.weight");
    ctx->stem_conv_b = find(m, "enc.stem.conv.bias");

    auto map_block = [&](int bi, std::vector<dense_layer_pf> & layers) {
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

    // ARM
    ctx->arm.conv_w = find(m, "arm.conv.weight");
    ctx->arm.conv_b = find(m, "arm.conv.bias");
    ctx->arm.proj_w = find(m, "arm.proj.weight");
    ctx->arm.proj_b = find(m, "arm.proj.bias");

    if (!ctx->stem_conv_w || !ctx->word_embed_w || !ctx->proj_w || !ctx->arm.conv_w) {
        fprintf(stderr, "posformer_ocr: missing critical tensors\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

posformer_ocr_context * posformer_ocr_init(const char * model_path, int n_threads) {
    auto ctx = std::make_unique<posformer_ocr_context>();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    gguf_context * gctx = core_gguf::open_metadata(model_path);
    if (!gctx) return nullptr;

    auto & hp = ctx->hparams;
    hp.growth_rate        = core_gguf::kv_u32(gctx, "posformer.encoder.growth_rate", 24);
    hp.num_layers         = core_gguf::kv_u32(gctx, "posformer.encoder.num_layers", 16);
    hp.input_channels     = core_gguf::kv_u32(gctx, "posformer.encoder.input_channels", 1);
    hp.d_model            = core_gguf::kv_u32(gctx, "posformer.decoder.d_model", 256);
    hp.nhead              = core_gguf::kv_u32(gctx, "posformer.decoder.nhead", 8);
    hp.num_decoder_layers = core_gguf::kv_u32(gctx, "posformer.decoder.num_layers", 3);
    hp.dim_feedforward    = core_gguf::kv_u32(gctx, "posformer.decoder.dim_feedforward", 1024);
    hp.vocab_size         = core_gguf::kv_u32(gctx, "posformer.decoder.vocab_size", 113);
    hp.max_len            = core_gguf::kv_u32(gctx, "posformer.decoder.max_len", 200);
    hp.pad_token          = core_gguf::kv_u32(gctx, "posformer.decoder.pad_token", 0);
    hp.sos_token          = core_gguf::kv_u32(gctx, "posformer.decoder.sos_token", 1);
    hp.eos_token          = core_gguf::kv_u32(gctx, "posformer.decoder.eos_token", 2);
    hp.arm_dc             = core_gguf::kv_u32(gctx, "posformer.arm.dc", 32);

    ctx->vocab = core_gguf::kv_str_array(gctx, "tokenizer.tokens");
    core_gguf::free_metadata(gctx);

    fprintf(stderr, "posformer_ocr: growth=%d layers=%d d=%d heads=%d dec=%d vocab=%d(%zu) dc=%d\n",
            hp.growth_rate, hp.num_layers, hp.d_model, hp.nhead,
            hp.num_decoder_layers, hp.vocab_size, ctx->vocab.size(), hp.arm_dc);

    // Prefer GPU backend — weights read via ggml_backend_tensor_get (GPU-safe).
    bool force_cpu = (getenv("POSFORMER_OCR_FORCE_CPU") && atoi(getenv("POSFORMER_OCR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, backend, "posformer_ocr", ctx->wl)) {
        ggml_backend_free(backend);
        return nullptr;
    }
    if (!map_tensors(ctx.get())) return nullptr;

    return ctx.release();
}

void posformer_ocr_free(posformer_ocr_context * ctx) {
    if (!ctx) return;
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

const posformer_ocr_hparams * posformer_ocr_get_hparams(const posformer_ocr_context * ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

// ---------------------------------------------------------------------------
// DenseNet encoder (identical to BTTR)
// ---------------------------------------------------------------------------

static void run_encoder(posformer_ocr_context * ctx, const float * gray, int W, int H) {
    const auto & hp = ctx->hparams;
    const int gr = hp.growth_rate;
    const int bn_size = 4;
    const int init_ch = 2 * gr;

    int h1 = (H + 2*3 - 7) / 2 + 1, w1 = (W + 2*3 - 7) / 2 + 1;
    std::vector<float> feat(init_ch * h1 * w1);
    conv2d(gray, 1, H, W, tf32(ctx, ctx->stem_conv_w), tf32(ctx, ctx->stem_conv_b),
           init_ch, 7, 7, 2, 3, feat.data(), h1, w1);
    relu_ip(feat.data(), init_ch * h1 * w1);

    int h2 = (h1 + 1) / 2, w2 = (w1 + 1) / 2;
    std::vector<float> pooled(init_ch * h2 * w2);
    maxpool2d_ceil(feat.data(), init_ch, h1, w1, 2, 2, pooled.data(), h2, w2);

    int cur_ch = init_ch, cur_h = h2, cur_w = w2;
    std::vector<float> features = std::move(pooled);

    auto run_block = [&](const std::vector<dense_layer_pf> & layers) {
        int sp = cur_h * cur_w;
        for (const auto & l : layers) {
            if (!l.conv1_w) continue;
            int bn_ch = bn_size * gr;
            std::vector<float> bot(bn_ch * sp);
            conv2d(features.data(), cur_ch, cur_h, cur_w,
                   tf32(ctx, l.conv1_w), tf32(ctx, l.conv1_b),
                   bn_ch, 1, 1, 1, 0, bot.data(), cur_h, cur_w);
            relu_ip(bot.data(), bn_ch * sp);
            std::vector<float> nf(gr * sp);
            conv2d(bot.data(), bn_ch, cur_h, cur_w,
                   tf32(ctx, l.conv2_w), tf32(ctx, l.conv2_b),
                   gr, 3, 3, 1, 1, nf.data(), cur_h, cur_w);
            relu_ip(nf.data(), gr * sp);
            features.resize((cur_ch + gr) * sp);
            memcpy(features.data() + cur_ch * sp, nf.data(), gr * sp * sizeof(float));
            cur_ch += gr;
        }
    };

    auto run_trans = [&](const transition_pf & t) {
        int sp = cur_h * cur_w;
        int out_ch = cur_ch / 2;
        std::vector<float> conv_out(out_ch * sp);
        conv2d(features.data(), cur_ch, cur_h, cur_w,
               tf32(ctx, t.conv_w), tf32(ctx, t.conv_b),
               out_ch, 1, 1, 1, 0, conv_out.data(), cur_h, cur_w);
        relu_ip(conv_out.data(), out_ch * sp);
        int oh = (cur_h + 1) / 2, ow = (cur_w + 1) / 2;
        std::vector<float> pool_out(out_ch * oh * ow);
        avgpool2d_ceil(conv_out.data(), out_ch, cur_h, cur_w,
                       2, 2, pool_out.data(), oh, ow);
        features = std::move(pool_out);
        cur_ch = out_ch; cur_h = oh; cur_w = ow;
    };

    run_block(ctx->block1);
    run_trans(ctx->trans1);
    run_block(ctx->block2);
    run_trans(ctx->trans2);
    run_block(ctx->block3);

    int sp = cur_h * cur_w;
    apply_bn(features.data(), cur_ch, sp,
             tf32(ctx, ctx->post_norm_scale), tf32(ctx, ctx->post_norm_offset));

    const int D = hp.d_model;
    std::vector<float> proj(D * sp);
    conv2d(features.data(), cur_ch, cur_h, cur_w,
           tf32(ctx, ctx->feat_proj_w), tf32(ctx, ctx->feat_proj_b),
           D, 1, 1, 1, 0, proj.data(), cur_h, cur_w);
    // No ReLU here — PyTorch's feature_proj is a plain Conv2d

    int n_pos = cur_h * cur_w;
    ctx->encoder_output.resize(n_pos * D);
    // Transpose from [D, h*w] to [h*w, D]
    for (int c = 0; c < D; c++)
        for (int i = 0; i < n_pos; i++)
            ctx->encoder_output[i * D + c] = proj[c * n_pos + i];

    // 2D positional encoding (DETR-style) — applied BEFORE LayerNorm
    {
        const int half_d = D / 2;        // 128
        const int quarter_d = half_d / 2; // 64
        const float scale = 2.0f * (float)M_PI;
        // 64 frequencies matching PyTorch's arange(0, half_d, 2)
        std::vector<float> inv_freq(quarter_d);
        for (int i = 0; i < quarter_d; i++)
            inv_freq[i] = 1.0f / powf(10000.0f, (float)(2 * i) / (float)half_d);
        for (int y = 0; y < cur_h; y++)
            for (int x = 0; x < cur_w; x++) {
                float y_norm = scale * (float)(y + 1) / (float)cur_h;
                float x_norm = scale * (float)(x + 1) / (float)cur_w;
                int pos_idx = y * cur_w + x;
                float * enc = ctx->encoder_output.data() + pos_idx * D;
                for (int i = 0; i < quarter_d; i++) {
                    float freq = inv_freq[i];
                    enc[2*i]     += sinf(x_norm * freq);
                    enc[2*i + 1] += cosf(x_norm * freq);  // same freq for sin/cos pair
                }
                for (int i = 0; i < quarter_d; i++) {
                    float freq = inv_freq[i];
                    enc[half_d + 2*i]     += sinf(y_norm * freq);
                    enc[half_d + 2*i + 1] += cosf(y_norm * freq);  // same freq
                }
            }
    }

    // LayerNorm — applied AFTER positional encoding
    const float * ln_w = tf32(ctx, ctx->enc_ln_w);
    const float * ln_b = tf32(ctx, ctx->enc_ln_b);
    for (int i = 0; i < n_pos; i++)
        layernorm(ctx->encoder_output.data() + i * D, D, ln_w, ln_b);

    ctx->n_enc_pos = n_pos;
    ctx->enc_h = cur_h;
    ctx->enc_w = cur_w;
    fprintf(stderr, "posformer_ocr: encoder: (%d, %d, %d) → %d positions × %d\n",
            cur_ch, cur_h, cur_w, n_pos, D);

    // Dump encoder output if requested
    const char * dump_dir = getenv("POSFORMER_DUMP");
    if (dump_dir) {
        char path[256];
        snprintf(path, sizeof(path), "%s/cpp_enc_output.f32", dump_dir);
        FILE * f = fopen(path, "wb");
        if (f) { fwrite(ctx->encoder_output.data(), sizeof(float), n_pos * D, f); fclose(f); }
        fprintf(stderr, "posformer_ocr: dumped encoder output to %s\n", path);
    }
}

// ---------------------------------------------------------------------------
// ARM (Attention Refinement Module) — incremental coverage computation
//
// For batch inference, ARM computes:
//   attns = cat(prev_attn, curr_attn) along head dim → [2*nhead, t, l]
//   mask out structural tokens (^, _, })
//   coverage = cumsum(attns, dim=t) - attns  (prefix sum excluding current)
//   reshape to spatial [2*nhead, h, w]
//   Conv2d(5x5) + ReLU + Conv2d(1x1, BN-folded)
//   → coverage_bias [nhead, l] to subtract from cross-attn logits
//
// For incremental decoding (one token at a time):
//   coverage at step t = running sum of masked attention from steps 0..t-1
//   After ARM, update running sum += masked attention at step t
// ---------------------------------------------------------------------------

// Token IDs excluded from coverage: ^ (82), _ (83), } (110)
static bool is_structural_token(int token_id) {
    return token_id == 82 || token_id == 83 || token_id == 110;
}

// Compute ARM coverage bias for one timestep.
// coverage_accum: [2*nhead * n_enc] running sum from all prior steps
// prev_attn: [nhead * n_enc] cross-attn weights from previous layer at current step
// curr_attn: [nhead * n_enc] first-pass cross-attn weights from current layer
// token_id: current token being decoded (for vocab masking)
// Returns coverage_bias [nhead * n_enc] to subtract from cross-attn logits.
// Updates coverage_accum in place.
static void compute_arm(posformer_ocr_context * ctx,
                        float * coverage_accum,
                        const float * prev_attn, const float * curr_attn,
                        int token_id,
                        float * coverage_bias) {
    const auto & hp = ctx->hparams;
    const int nhead = hp.nhead;
    const int n_enc = ctx->n_enc_pos;
    const int h = ctx->enc_h;
    const int w = ctx->enc_w;
    const int dc = hp.arm_dc;
    const int in_ch = 2 * nhead;  // 16

    // The coverage input is the accumulated attention from all previous steps.
    // We need it in spatial layout [2*nhead, h, w] for the conv.
    // coverage_accum is already [2*nhead, n_enc] where n_enc = h*w.

    // Run ARM conv pipeline on the accumulated coverage
    // Conv2d(in_ch=16, dc=32, k=5, pad=2) + ReLU
    const float * conv_w = tf32(ctx, ctx->arm.conv_w);
    const float * conv_b = tf32(ctx, ctx->arm.conv_b);
    std::vector<float> conv_out(dc * h * w);
    conv2d(coverage_accum, in_ch, h, w, conv_w, conv_b,
           dc, 5, 5, 1, 2, conv_out.data(), h, w);
    relu_ip(conv_out.data(), dc * h * w);

    // Conv2d(dc=32, nhead=8, k=1) + BN-folded bias
    const float * proj_w = tf32(ctx, ctx->arm.proj_w);
    const float * proj_b = tf32(ctx, ctx->arm.proj_b);
    // Output is [nhead, h, w], flatten to [nhead, n_enc]
    conv2d(conv_out.data(), dc, h, w, proj_w, proj_b,
           nhead, 1, 1, 1, 0, coverage_bias, h, w);

    // Update coverage accumulator with current step's attention
    // Concat prev_attn and curr_attn into [2*nhead, n_enc]
    // Apply vocab mask (zero out if structural token)
    if (!is_structural_token(token_id)) {
        for (int i = 0; i < nhead * n_enc; i++) {
            coverage_accum[i] += prev_attn[i];                  // prev_attn channels
            coverage_accum[nhead * n_enc + i] += curr_attn[i];  // curr_attn channels
        }
    }
    // If structural token, don't accumulate (masking = zero contribution)
}

// ---------------------------------------------------------------------------
// Transformer decoder with ARM (greedy, incremental)
// ---------------------------------------------------------------------------

static std::string greedy_decode(posformer_ocr_context * ctx) {
    const auto & hp = ctx->hparams;
    const int D = hp.d_model;
    const int V = hp.vocab_size;
    const int n_enc = ctx->n_enc_pos;
    const int nhead = hp.nhead;
    const int head_dim = D / nhead;
    const float scale = 1.0f / sqrtf((float)head_dim);

    // KV caches for self-attention
    std::vector<std::vector<float>> kv_k(hp.num_decoder_layers);
    std::vector<std::vector<float>> kv_v(hp.num_decoder_layers);

    // Precompute cross-attention K/V from encoder output
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

    // ARM coverage accumulators.
    // ARM is applied between layers: layer 1 gets coverage from layer 0,
    // layer 2 gets coverage from layer 1. That's 2 ARM invocations.
    // Each accumulator is [2*nhead, n_enc] = [16, n_enc].
    const int n_arm = hp.num_decoder_layers - 1;  // 2
    std::vector<std::vector<float>> arm_accum(n_arm);
    for (int i = 0; i < n_arm; i++)
        arm_accum[i].assign(2 * nhead * n_enc, 0.0f);

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

        if (ctx->pos_enc && step < 500) {
            const float * pe = tf32(ctx, ctx->pos_enc);
            for (int i = 0; i < D; i++) x[i] += pe[step * D + i];
        }

        // Cross-attention weights from previous layer (for ARM)
        std::vector<float> prev_layer_ca_weights(nhead * n_enc);

        for (int li = 0; li < hp.num_decoder_layers; li++) {
            const auto & l = ctx->dec_layers[li];

            // --- Self-attention ---
            std::vector<float> qkv(3 * D);
            linear(x.data(), D, tf32(ctx, l.sa_qkv_w), tf32(ctx, l.sa_qkv_b),
                   3 * D, qkv.data());
            float * q = qkv.data(), * k = qkv.data() + D, * v = qkv.data() + 2*D;

            kv_k[li].insert(kv_k[li].end(), k, k + D);
            kv_v[li].insert(kv_v[li].end(), v, v + D);
            int n_past = step + 1;

            std::vector<float> attn_out(D, 0.0f);
            for (int h = 0; h < nhead; h++) {
                std::vector<float> scores(n_past);
                float max_s = -1e9f;
                for (int ki = 0; ki < n_past; ki++) {
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++)
                        dot += q[h*head_dim+d] * kv_k[li][ki*D+h*head_dim+d];
                    scores[ki] = dot * scale;
                    if (scores[ki] > max_s) max_s = scores[ki];
                }
                float sum_exp = 0;
                for (auto & s : scores) { s = expf(s - max_s); sum_exp += s; }
                for (auto & s : scores) s /= sum_exp;
                for (int d = 0; d < head_dim; d++) {
                    float s = 0;
                    for (int ki = 0; ki < n_past; ki++)
                        s += scores[ki] * kv_v[li][ki*D+h*head_dim+d];
                    attn_out[h*head_dim+d] = s;
                }
            }
            std::vector<float> sa_proj(D);
            linear(attn_out.data(), D, tf32(ctx, l.sa_out_w),
                   tf32(ctx, l.sa_out_b), D, sa_proj.data());
            for (int i = 0; i < D; i++) x[i] += sa_proj[i];
            layernorm(x.data(), D, tf32(ctx, l.ln1_w), tf32(ctx, l.ln1_b));

            // --- Cross-attention (with ARM for layers > 0) ---
            {
                const float * ca_W = tf32(ctx, l.ca_qkv_w);
                const float * ca_B = tf32(ctx, l.ca_qkv_b);
                std::vector<float> cq(D);
                linear(x.data(), D, ca_W, ca_B, D, cq.data());

                const float * ck = ca_ck[li].data();
                const float * cv = ca_cv[li].data();

                // Compute raw attention scores
                std::vector<float> raw_scores(nhead * n_enc);
                for (int h = 0; h < nhead; h++) {
                    float max_s = -1e9f;
                    for (int ki = 0; ki < n_enc; ki++) {
                        float dot = 0;
                        for (int d = 0; d < head_dim; d++)
                            dot += cq[h*head_dim+d] * ck[ki*D+h*head_dim+d];
                        raw_scores[h*n_enc+ki] = dot * scale;
                        if (raw_scores[h*n_enc+ki] > max_s)
                            max_s = raw_scores[h*n_enc+ki];
                    }
                    // First softmax to get initial attention weights
                    float sum_exp = 0;
                    for (int ki = 0; ki < n_enc; ki++) {
                        raw_scores[h*n_enc+ki] = expf(raw_scores[h*n_enc+ki] - max_s);
                        sum_exp += raw_scores[h*n_enc+ki];
                    }
                    for (int ki = 0; ki < n_enc; ki++)
                        raw_scores[h*n_enc+ki] /= sum_exp;
                }

                // curr_attn = raw_scores after first softmax (for ARM input)
                // These are the "initial attention weights" before ARM refinement.

                if (li > 0) {
                    // Apply ARM: compute coverage bias and subtract from logits
                    std::vector<float> cov_bias(nhead * n_enc);
                    compute_arm(ctx, arm_accum[li - 1].data(),
                                prev_layer_ca_weights.data(),
                                raw_scores.data(),
                                prev_token, cov_bias.data());

                    // Recompute attention: subtract coverage bias from raw logits, re-softmax
                    // We need the raw logits (pre-softmax). Recompute them.
                    for (int h = 0; h < nhead; h++) {
                        float max_s = -1e9f;
                        for (int ki = 0; ki < n_enc; ki++) {
                            float dot = 0;
                            for (int d = 0; d < head_dim; d++)
                                dot += cq[h*head_dim+d] * ck[ki*D+h*head_dim+d];
                            float logit = dot * scale - cov_bias[h*n_enc+ki];
                            raw_scores[h*n_enc+ki] = logit;
                            if (logit > max_s) max_s = logit;
                        }
                        float sum_exp = 0;
                        for (int ki = 0; ki < n_enc; ki++) {
                            raw_scores[h*n_enc+ki] = expf(raw_scores[h*n_enc+ki] - max_s);
                            sum_exp += raw_scores[h*n_enc+ki];
                        }
                        for (int ki = 0; ki < n_enc; ki++)
                            raw_scores[h*n_enc+ki] /= sum_exp;
                    }
                }

                // Save this layer's attention weights for next layer's ARM
                memcpy(prev_layer_ca_weights.data(), raw_scores.data(),
                       nhead * n_enc * sizeof(float));

                // Weighted sum of values
                std::vector<float> ca_out(D, 0.0f);
                for (int h = 0; h < nhead; h++)
                    for (int d = 0; d < head_dim; d++) {
                        float s = 0;
                        for (int ki = 0; ki < n_enc; ki++)
                            s += raw_scores[h*n_enc+ki] * cv[ki*D+h*head_dim+d];
                        ca_out[h*head_dim+d] = s;
                    }

                std::vector<float> ca_proj(D);
                linear(ca_out.data(), D, tf32(ctx, l.ca_out_w),
                       tf32(ctx, l.ca_out_b), D, ca_proj.data());
                for (int i = 0; i < D; i++) x[i] += ca_proj[i];
                layernorm(x.data(), D, tf32(ctx, l.ln2_w), tf32(ctx, l.ln2_b));
            }

            // --- FFN ---
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

        // Output projection
        std::vector<float> logits(V);
        linear(x.data(), D, tf32(ctx, ctx->proj_w), tf32(ctx, ctx->proj_b),
               V, logits.data());

        int best = 0;
        float best_score = logits[0];
        for (int v = 1; v < V; v++)
            if (logits[v] > best_score) { best_score = logits[v]; best = v; }

        // Debug: dump top-5 logits
        {
            std::vector<std::pair<float,int>> scored(V);
            for (int v = 0; v < V; v++) scored[v] = {logits[v], v};
            std::sort(scored.begin(), scored.end(), [](auto &a, auto &b){ return a.first > b.first; });
            fprintf(stderr, "  step %d: token=%d '%s' | top5:", step, best,
                    best < (int)ctx->vocab.size() ? ctx->vocab[best].c_str() : "?");
            for (int i = 0; i < 5 && i < V; i++)
                fprintf(stderr, " %s(%.2f)",
                        scored[i].second < (int)ctx->vocab.size() ? ctx->vocab[scored[i].second].c_str() : "?",
                        scored[i].first);
            fprintf(stderr, "\n");

            // Dump logits to file if env POSFORMER_DUMP is set
            const char * dump_dir = getenv("POSFORMER_DUMP");
            if (dump_dir) {
                char path[256];
                snprintf(path, sizeof(path), "%s/cpp_step%03d_logits.f32", dump_dir, step);
                FILE * f = fopen(path, "wb");
                if (f) { fwrite(logits.data(), sizeof(float), V, f); fclose(f); }
            }
        }

        if (best == hp.eos_token || best == hp.pad_token) break;
        tokens.push_back(best);
        prev_token = best;
    }

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
// Public API
// ---------------------------------------------------------------------------

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

const char * posformer_ocr_recognize(
    posformer_ocr_context * ctx,
    const float * pixels, int width, int height,
    int * out_len
) {
    if (!ctx || !pixels || width <= 0 || height <= 0) return nullptr;

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
        fprintf(stderr, "posformer_ocr: auto-inverted (mean=%.2f)\n", mean);
    }

    int w = width, h = height;
    std::vector<float> scaled;
    if (scale_to_fit(input, width, height, scaled, w, h)) {
        input = scaled.data();
        fprintf(stderr, "posformer_ocr: scaled %dx%d → %dx%d\n", width, height, w, h);
    }

    run_encoder(ctx, input, w, h);
    ctx->result_buf = greedy_decode(ctx);

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const char * posformer_ocr_recognize_raw(
    posformer_ocr_context * ctx,
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
    return posformer_ocr_recognize(ctx, gray.data(), width, height, out_len);
}
