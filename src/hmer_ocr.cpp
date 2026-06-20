// hmer_ocr.cpp — Handwritten Math Expression Recognition via ggml.
//
// Architecture: DenseNet-121 encoder + GRU attention decoder (with coverage).
// Loads from GGUF produced by convert-hmer-to-gguf.py.
//
// Implements:
//   1. Tensor mapping from GGUF
//   2. DenseNet-121 forward pass (conv, BN-as-scale+offset, concat, pool)
//   3. GRU + Bahdanau attention + coverage decoder
//   4. Greedy decoding loop
//   5. Detokenization

#include "hmer_ocr.h"

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
#include <memory>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

struct dense_layer {
    // Pre-activation BN (precomputed scale+offset)
    struct ggml_tensor * bn1_scale;   // (in_ch,)
    struct ggml_tensor * bn1_offset;  // (in_ch,)
    struct ggml_tensor * conv1_w;     // (128, in_ch, 1, 1)

    struct ggml_tensor * bn2_scale;   // (128,)
    struct ggml_tensor * bn2_offset;  // (128,)
    struct ggml_tensor * conv2_w;     // (32, 128, 3, 3)
};

struct transition_layer {
    struct ggml_tensor * bn_scale;    // (in_ch,)
    struct ggml_tensor * bn_offset;   // (in_ch,)
    struct ggml_tensor * conv_w;      // (out_ch, in_ch, 1, 1)
};

struct hmer_ocr_context {
    hmer_ocr_hparams hparams;

    // Encoder: stem (conv0_m + norm0 folded)
    struct ggml_tensor * stem_conv_w;  // (64, 2, 7, 7) — BN folded in
    struct ggml_tensor * stem_conv_b;  // (64,)

    // Encoder: dense blocks
    std::vector<dense_layer> block1;  // 6 layers
    std::vector<dense_layer> block2;  // 12 layers
    std::vector<dense_layer> block3;  // 24 layers

    // Encoder: transitions
    transition_layer trans1;
    transition_layer trans2;

    // Encoder: final BN (precomputed scale+offset)
    struct ggml_tensor * final_bn_scale;   // (1024,)
    struct ggml_tensor * final_bn_offset;  // (1024,)

    // Decoder
    struct ggml_tensor * embedding_w;      // (112, 256)

    struct ggml_tensor * gru1_w_ih;        // (768, 256)
    struct ggml_tensor * gru1_w_hh;        // (768, 256)
    struct ggml_tensor * gru1_b_ih;        // (768,)
    struct ggml_tensor * gru1_b_hh;        // (768,)

    struct ggml_tensor * gru_w_ih;         // (768, 1024)
    struct ggml_tensor * gru_w_hh;         // (768, 256)
    struct ggml_tensor * gru_b_ih;         // (768,)
    struct ggml_tensor * gru_b_hh;         // (768,)

    struct ggml_tensor * hidden_w;         // (256, 256) — query projection
    struct ggml_tensor * hidden_b;         // (256,)
    struct ggml_tensor * hidden2_w;        // (128, 256)
    struct ggml_tensor * hidden2_b;        // (128,)
    struct ggml_tensor * emb2_w;           // (128, 256)
    struct ggml_tensor * emb2_b;           // (128,)

    struct ggml_tensor * ua_w;             // (256, 1024) — encoder key proj
    struct ggml_tensor * ua_b;             // (256,)
    struct ggml_tensor * uf_w;             // (256, 1) — coverage proj
    struct ggml_tensor * uf_b;             // (256,)
    struct ggml_tensor * v_w;              // (1, 256) — energy scalar
    struct ggml_tensor * v_b;              // (1,)
    struct ggml_tensor * wc_w;             // (128, 1024) — context proj
    struct ggml_tensor * wc_b;             // (128,)
    struct ggml_tensor * out_w;            // (112, 128) — output logits
    struct ggml_tensor * out_b;            // (112,)

    struct ggml_tensor * conv1_w;          // (1, 1, 3, 3) — coverage conv
    struct ggml_tensor * conv1_b;          // (1,)
    struct ggml_tensor * conv_tan_w;       // (256, 256, 3, 3) — attention conv
    struct ggml_tensor * conv_tan_b;       // (256,)
    struct ggml_tensor * bn1_scale;        // (256,) — attention BN
    struct ggml_tensor * bn1_offset;       // (256,)

    // Tokenizer
    std::vector<std::string> vocab;

    // GGUF loader state
    core_gguf::WeightLoad wl;
    int n_threads;

    // Dequantized weight cache (for quantized models)
    core_cpu::DequantCache dequant_cache;

    // Inference state
    std::string result_buf;
    std::vector<float> char_confidences; // per-token softmax probabilities

    // Cached encoder output: (enc_h * enc_w, 1024)
    std::vector<float> encoder_output;
    int enc_h;  // spatial height after encoder
    int enc_w;  // spatial width after encoder

    // Pre-allocated decoder scratch (avoids per-step heap allocs)
    struct dec_scratch {
        std::vector<float> embedded, st, hidden1;
        std::vector<float> et, et_chw, ct_out, ct_hwc;
        std::vector<float> energy, alpha, context;
        std::vector<float> h2, e2, c2, combined;
        std::vector<float> logits;
        bool allocated = false;
    } ds;

    bool bench;
};

// ---------------------------------------------------------------------------
// Tensor mapping
// ---------------------------------------------------------------------------

static struct ggml_tensor * find(
    const std::unordered_map<std::string, ggml_tensor *> & m,
    const char * name
) {
    auto it = m.find(name);
    return it != m.end() ? it->second : nullptr;
}

static bool map_tensors(hmer_ocr_context * ctx) {
    const auto & m = ctx->wl.tensors;
    char buf[256];

    // Stem
    ctx->stem_conv_w = find(m, "enc.stem.conv.weight");
    ctx->stem_conv_b = find(m, "enc.stem.conv.bias");
    if (!ctx->stem_conv_w) {
        fprintf(stderr, "hmer_ocr: missing enc.stem.conv.weight\n");
        return false;
    }

    // Dense blocks
    auto map_block = [&](int bi, int n_layers, std::vector<dense_layer> & layers) {
        layers.resize(n_layers);
        for (int li = 0; li < n_layers; li++) {
            auto & l = layers[li];
            auto T = [&](const char * suffix) -> struct ggml_tensor * {
                snprintf(buf, sizeof(buf), "enc.block%d.layer%d.%s", bi, li + 1, suffix);
                return find(m, buf);
            };
            l.bn1_scale  = T("bn1.scale");
            l.bn1_offset = T("bn1.offset");
            l.conv1_w    = T("conv1.weight");
            l.bn2_scale  = T("bn2.scale");
            l.bn2_offset = T("bn2.offset");
            l.conv2_w    = T("conv2.weight");
        }
    };

    map_block(1, ctx->hparams.block_config[0], ctx->block1);
    map_block(2, ctx->hparams.block_config[1], ctx->block2);
    map_block(3, ctx->hparams.block_config[2], ctx->block3);

    // Transitions
    auto map_trans = [&](int ti, transition_layer & t) {
        snprintf(buf, sizeof(buf), "enc.trans%d.bn.scale", ti);
        t.bn_scale = find(m, buf);
        snprintf(buf, sizeof(buf), "enc.trans%d.bn.offset", ti);
        t.bn_offset = find(m, buf);
        snprintf(buf, sizeof(buf), "enc.trans%d.conv.weight", ti);
        t.conv_w = find(m, buf);
    };
    map_trans(1, ctx->trans1);
    map_trans(2, ctx->trans2);

    ctx->final_bn_scale  = find(m, "enc.final_bn.scale");
    ctx->final_bn_offset = find(m, "enc.final_bn.offset");

    // Decoder
    ctx->embedding_w = find(m, "dec.embedding.weight");
    ctx->gru1_w_ih   = find(m, "dec.gru1.weight_ih");
    ctx->gru1_w_hh   = find(m, "dec.gru1.weight_hh");
    ctx->gru1_b_ih   = find(m, "dec.gru1.bias_ih");
    ctx->gru1_b_hh   = find(m, "dec.gru1.bias_hh");
    ctx->gru_w_ih    = find(m, "dec.gru.weight_ih");
    ctx->gru_w_hh    = find(m, "dec.gru.weight_hh");
    ctx->gru_b_ih    = find(m, "dec.gru.bias_ih");
    ctx->gru_b_hh    = find(m, "dec.gru.bias_hh");
    ctx->hidden_w    = find(m, "dec.hidden.weight");
    ctx->hidden_b    = find(m, "dec.hidden.bias");
    ctx->hidden2_w   = find(m, "dec.hidden2.weight");
    ctx->hidden2_b   = find(m, "dec.hidden2.bias");
    ctx->emb2_w      = find(m, "dec.emb2.weight");
    ctx->emb2_b      = find(m, "dec.emb2.bias");
    ctx->ua_w        = find(m, "dec.ua.weight");
    ctx->ua_b        = find(m, "dec.ua.bias");
    ctx->uf_w        = find(m, "dec.uf.weight");
    ctx->uf_b        = find(m, "dec.uf.bias");
    ctx->v_w         = find(m, "dec.v.weight");
    ctx->v_b         = find(m, "dec.v.bias");
    ctx->wc_w        = find(m, "dec.wc.weight");
    ctx->wc_b        = find(m, "dec.wc.bias");
    ctx->out_w       = find(m, "dec.out.weight");
    ctx->out_b       = find(m, "dec.out.bias");
    ctx->conv1_w     = find(m, "dec.conv1.weight");
    ctx->conv1_b     = find(m, "dec.conv1.bias");
    ctx->conv_tan_w  = find(m, "dec.conv_tan.weight");
    ctx->conv_tan_b  = find(m, "dec.conv_tan.bias");
    ctx->bn1_scale   = find(m, "dec.bn1.scale");
    ctx->bn1_offset  = find(m, "dec.bn1.offset");

    if (!ctx->embedding_w || !ctx->gru1_w_ih || !ctx->gru_w_ih) {
        fprintf(stderr, "hmer_ocr: missing critical decoder tensors\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

hmer_ocr_context * hmer_ocr_init(const char * model_path, int n_threads) {
    auto ctx = std::make_unique<hmer_ocr_context>();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    // Phase 1: metadata
    gguf_context * gctx = core_gguf::open_metadata(model_path);
    if (!gctx) {
        fprintf(stderr, "hmer_ocr: failed to open %s\n", model_path);
        return nullptr;
    }

    auto & hp = ctx->hparams;
    hp.num_init_features = core_gguf::kv_u32(gctx, "hmer.encoder.num_init_features", 64);
    hp.growth_rate       = core_gguf::kv_u32(gctx, "hmer.encoder.growth_rate", 32);
    hp.input_channels    = core_gguf::kv_u32(gctx, "hmer.encoder.input_channels", 2);
    hp.output_channels   = core_gguf::kv_u32(gctx, "hmer.encoder.output_channels", 1024);
    hp.hidden_size       = core_gguf::kv_u32(gctx, "hmer.decoder.hidden_size", 256);
    hp.output_size       = core_gguf::kv_u32(gctx, "hmer.decoder.output_size", 112);
    hp.sos_token         = core_gguf::kv_u32(gctx, "hmer.decoder.sos_token", 111);
    hp.eol_token         = core_gguf::kv_u32(gctx, "hmer.decoder.eol_token", 0);
    hp.max_seq_len       = core_gguf::kv_u32(gctx, "hmer.decoder.max_seq_len", 48);

    // block_config array — read from GGUF or use defaults
    hp.block_config[0] = 6;
    hp.block_config[1] = 12;
    hp.block_config[2] = 24;
    // TODO: read from "hmer.encoder.block_config" array if present

    // Tokenizer
    ctx->vocab = core_gguf::kv_str_array(gctx, "tokenizer.tokens");

    core_gguf::free_metadata(gctx);

    fprintf(stderr, "hmer_ocr: blocks=[%d,%d,%d] hidden=%d vocab=%d(%zu)\n",
            hp.block_config[0], hp.block_config[1], hp.block_config[2],
            hp.hidden_size, hp.output_size, ctx->vocab.size());

    // Phase 2: load weights — prefer GPU backend (weights read via ggml_backend_tensor_get)
    bool force_cpu = (getenv("HMER_OCR_FORCE_CPU") && atoi(getenv("HMER_OCR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, backend, "hmer_ocr", ctx->wl)) {
        ggml_backend_free(backend);
        fprintf(stderr, "hmer_ocr: failed to load weights from %s\n", model_path);
        return nullptr;
    }

    if (!map_tensors(ctx.get())) return nullptr;

    int mapped = 0;
    for (const auto & l : ctx->block1) if (l.conv1_w) mapped++;
    for (const auto & l : ctx->block2) if (l.conv1_w) mapped++;
    for (const auto & l : ctx->block3) if (l.conv1_w) mapped++;
    fprintf(stderr, "hmer_ocr: mapped %d/42 dense layers, %zu vocab tokens\n",
            mapped, ctx->vocab.size());

    ctx->bench = (std::getenv("CRISPEMBED_HMER_BENCH") != nullptr);

    return ctx.release();
}

void hmer_ocr_free(hmer_ocr_context * ctx) {
    if (!ctx) return;
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

const hmer_ocr_hparams * hmer_ocr_get_hparams(const hmer_ocr_context * ctx) {
    return ctx ? &ctx->hparams : nullptr;
}

// ---------------------------------------------------------------------------
// Helpers — core_cpu shared + local unique ops
// ---------------------------------------------------------------------------

static const float * tensor_f32(hmer_ocr_context * ctx, struct ggml_tensor * t) {
    return ctx->dequant_cache.get(t);
}

static void linear(const float * x, int in_dim,
                   const float * W, const float * B, int out_dim,
                   float * out) {
    core_cpu::linear_cpu(x, out, in_dim, out_dim, W, B);
}

// BN scale+offset (unique — not in core_cpu)
static void apply_bn_scale(float * data, int channels, int spatial,
                           const float * scale, const float * offset) {
    for (int c = 0; c < channels; c++) {
        float s = scale[c], o = offset[c];
        float * row = data + c * spatial;
        for (int i = 0; i < spatial; i++) row[i] = row[i] * s + o;
    }
}

static void relu_inplace(float * data, int n) { core_cpu::relu_inplace(data, n); }

static void conv2d(const float * input, int in_ch, int in_h, int in_w,
                   const float * weight, const float * bias,
                   int out_ch, int kH, int kW, int stride, int pad,
                   float * output, int /*out_h*/, int /*out_w*/) {
    core_cpu::conv2d_cpu(input, output, weight, bias, in_ch, out_ch, in_h, in_w, kH, kW, stride, pad);
}

// ---------------------------------------------------------------------------
// Helper: MaxPool2d
// ---------------------------------------------------------------------------

static void maxpool2d(const float * input, int ch, int in_h, int in_w,
                      int ksize, int stride, int pad,
                      float * output, int out_h, int out_w) {
    for (int c = 0; c < ch; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float maxv = -1e30f;
                for (int kh = 0; kh < ksize; kh++) {
                    for (int kw = 0; kw < ksize; kw++) {
                        int ih = oh * stride - pad + kh;
                        int iw = ow * stride - pad + kw;
                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            float v = input[c * in_h * in_w + ih * in_w + iw];
                            if (v > maxv) maxv = v;
                        }
                    }
                }
                output[c * out_h * out_w + oh * out_w + ow] = maxv;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: AvgPool2d
// ---------------------------------------------------------------------------

static void avgpool2d(const float * input, int ch, int in_h, int in_w,
                      int ksize, int stride,
                      float * output, int out_h, int out_w) {
    float inv_k2 = 1.0f / (ksize * ksize);
    for (int c = 0; c < ch; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                float sum = 0;
                for (int kh = 0; kh < ksize; kh++) {
                    for (int kw = 0; kw < ksize; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;
                        if (ih < in_h && iw < in_w) {
                            sum += input[c * in_h * in_w + ih * in_w + iw];
                        }
                    }
                }
                output[c * out_h * out_w + oh * out_w + ow] = sum * inv_k2;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DenseNet-121 encoder forward pass
// ---------------------------------------------------------------------------

static void run_encoder(hmer_ocr_context * ctx,
                        const float * gray, int W, int H) {
    // Input: 2-channel (gray + mask), CHW layout
    // Channel 0 = gray [0,1], Channel 1 = mask (all 1.0)
    int spatial = W * H;
    std::vector<float> input(2 * spatial);
    memcpy(input.data(), gray, spatial * sizeof(float));
    std::fill(input.data() + spatial, input.data() + 2 * spatial, 1.0f);

    // Stem: Conv2d(2→64, 7×7, stride=2, pad=3) + BN (folded) + ReLU + MaxPool
    int h1 = (H + 2 * 3 - 7) / 2 + 1;
    int w1 = (W + 2 * 3 - 7) / 2 + 1;
    int ch1 = 64;
    std::vector<float> stem(ch1 * h1 * w1);
    conv2d(input.data(), 2, H, W,
           tensor_f32(ctx, ctx->stem_conv_w), tensor_f32(ctx, ctx->stem_conv_b),
           ch1, 7, 7, 2, 3, stem.data(), h1, w1);
    relu_inplace(stem.data(), ch1 * h1 * w1);

    // MaxPool(3, stride=2, pad=1)
    int h2 = (h1 + 2 * 1 - 3) / 2 + 1;
    int w2 = (w1 + 2 * 1 - 3) / 2 + 1;
    std::vector<float> pooled(ch1 * h2 * w2);
    maxpool2d(stem.data(), ch1, h1, w1, 3, 2, 1, pooled.data(), h2, w2);

    // Current feature map state
    int cur_ch = ch1;
    int cur_h = h2, cur_w = w2;
    std::vector<float> features = std::move(pooled);

    // Dense blocks + transitions
    auto run_dense_block = [&](const std::vector<dense_layer> & layers) {
        int spatial = cur_h * cur_w;
        for (const auto & l : layers) {
            if (!l.conv1_w) continue;

            // BN1 + ReLU on current features
            std::vector<float> normed(cur_ch * spatial);
            memcpy(normed.data(), features.data(), cur_ch * spatial * sizeof(float));
            apply_bn_scale(normed.data(), cur_ch, spatial,
                           tensor_f32(ctx, l.bn1_scale), tensor_f32(ctx, l.bn1_offset));
            relu_inplace(normed.data(), cur_ch * spatial);

            // Conv1: 1×1 bottleneck → 128 channels
            int bn_ch = 128;
            std::vector<float> bottleneck(bn_ch * spatial);
            conv2d(normed.data(), cur_ch, cur_h, cur_w,
                   tensor_f32(ctx, l.conv1_w), nullptr,
                   bn_ch, 1, 1, 1, 0, bottleneck.data(), cur_h, cur_w);

            // BN2 + ReLU
            apply_bn_scale(bottleneck.data(), bn_ch, spatial,
                           tensor_f32(ctx, l.bn2_scale), tensor_f32(ctx, l.bn2_offset));
            relu_inplace(bottleneck.data(), bn_ch * spatial);

            // Conv2: 3×3 → growth_rate (32) channels
            int gr = ctx->hparams.growth_rate;
            std::vector<float> new_features(gr * spatial);
            conv2d(bottleneck.data(), bn_ch, cur_h, cur_w,
                   tensor_f32(ctx, l.conv2_w), nullptr,
                   gr, 3, 3, 1, 1, new_features.data(), cur_h, cur_w);

            // Concat: features = cat(features, new_features) along channel dim
            features.resize((cur_ch + gr) * spatial);
            memcpy(features.data() + cur_ch * spatial,
                   new_features.data(), gr * spatial * sizeof(float));
            cur_ch += gr;
        }
    };

    auto run_transition = [&](const transition_layer & t) {
        int spatial = cur_h * cur_w;

        // BN + ReLU
        apply_bn_scale(features.data(), cur_ch, spatial,
                       tensor_f32(ctx, t.bn_scale), tensor_f32(ctx, t.bn_offset));
        relu_inplace(features.data(), cur_ch * spatial);

        // Conv 1×1: cur_ch → cur_ch/2
        int out_ch = cur_ch / 2;
        std::vector<float> conv_out(out_ch * spatial);
        conv2d(features.data(), cur_ch, cur_h, cur_w,
               tensor_f32(ctx, t.conv_w), nullptr,
               out_ch, 1, 1, 1, 0, conv_out.data(), cur_h, cur_w);

        // AvgPool 2×2
        int oh = cur_h / 2, ow = cur_w / 2;
        std::vector<float> pool_out(out_ch * oh * ow);
        avgpool2d(conv_out.data(), out_ch, cur_h, cur_w,
                  2, 2, pool_out.data(), oh, ow);

        features = std::move(pool_out);
        cur_ch = out_ch;
        cur_h = oh;
        cur_w = ow;
    };

    // Block 1 (6 layers): 64 → 64+6*32 = 256
    run_dense_block(ctx->block1);
    run_transition(ctx->trans1);  // 256 → 128, spatial /2

    // Block 2 (12 layers): 128 → 128+12*32 = 512
    run_dense_block(ctx->block2);
    run_transition(ctx->trans2);  // 512 → 256, spatial /2

    // Block 3 (24 layers): 256 → 256+24*32 = 1024
    run_dense_block(ctx->block3);

    // Final BN + ReLU
    int final_spatial = cur_h * cur_w;
    apply_bn_scale(features.data(), cur_ch, final_spatial,
                   tensor_f32(ctx, ctx->final_bn_scale),
                   tensor_f32(ctx, ctx->final_bn_offset));
    relu_inplace(features.data(), cur_ch * final_spatial);

    // Store encoder output in (H'*W', 1024) layout for the decoder
    // Transpose from CHW to HW×C
    ctx->enc_h = cur_h;
    ctx->enc_w = cur_w;
    int n_positions = cur_h * cur_w;
    ctx->encoder_output.resize(n_positions * cur_ch);
    for (int c = 0; c < cur_ch; c++) {
        for (int i = 0; i < n_positions; i++) {
            ctx->encoder_output[i * cur_ch + c] = features[c * n_positions + i];
        }
    }

    fprintf(stderr, "hmer_ocr: encoder output: (%d, %d, %d) = %d positions\n",
            cur_ch, cur_h, cur_w, n_positions);

}

// ---------------------------------------------------------------------------
// GRU cell
// ---------------------------------------------------------------------------

static void gru_cell(const float * x, int x_dim,
                     const float * h, int h_dim,
                     const float * W_ih, const float * W_hh,
                     const float * b_ih, const float * b_hh,
                     float * h_out) {
    // W_ih: (3*h_dim, x_dim), W_hh: (3*h_dim, h_dim)
    // Gates: z (update), r (reset), n (new)
    std::vector<float> gates_ih(3 * h_dim), gates_hh(3 * h_dim);

    // gates_ih = W_ih @ x + b_ih
    for (int o = 0; o < 3 * h_dim; o++) {
        float sum = b_ih[o];
        for (int i = 0; i < x_dim; i++) {
            sum += x[i] * W_ih[o * x_dim + i];
        }
        gates_ih[o] = sum;
    }

    // gates_hh = W_hh @ h + b_hh
    for (int o = 0; o < 3 * h_dim; o++) {
        float sum = b_hh[o];
        for (int i = 0; i < h_dim; i++) {
            sum += h[i] * W_hh[o * h_dim + i];
        }
        gates_hh[o] = sum;
    }

    // PyTorch GRU weight order: [reset, update, new] (r, z, n)
    // r = sigmoid(gates_ih[0:H] + gates_hh[0:H])
    // z = sigmoid(gates_ih[H:2H] + gates_hh[H:2H])
    // n = tanh(gates_ih[2H:3H] + r * gates_hh[2H:3H])
    // h' = (1 - z) * n + z * h
    for (int i = 0; i < h_dim; i++) {
        float r = 1.0f / (1.0f + expf(-(gates_ih[i] + gates_hh[i])));
        float z = 1.0f / (1.0f + expf(-(gates_ih[h_dim + i] + gates_hh[h_dim + i])));
        float n = tanhf(gates_ih[2 * h_dim + i] + r * gates_hh[2 * h_dim + i]);
        h_out[i] = (1.0f - z) * n + z * h[i];
    }
}

// ---------------------------------------------------------------------------
// Decoder: single step
// ---------------------------------------------------------------------------

struct decoder_state {
    std::vector<float> hidden;           // (256,)
    std::vector<float> attention_sum;    // (enc_h * enc_w,)  cumulative coverage
    std::vector<float> decoder_attention; // (enc_h * enc_w,) prev step attention

    // Precomputed once from encoder output
    std::vector<float> enc_ua;           // (enc_h * enc_w, 256) = ua(encoder)
};

static int decoder_step(hmer_ocr_context * ctx,
                        int prev_token,
                        decoder_state & state) {
    const int H = ctx->hparams.hidden_size;  // 256
    const int V = ctx->hparams.output_size;  // 112
    const int enc_n = ctx->enc_h * ctx->enc_w;
    const int enc_ch = ctx->hparams.output_channels; // 1024

    auto & ds = ctx->ds;

    // 1. Embed previous token → (256,)
    memset(ds.embedded.data(), 0, H * sizeof(float));
    if (ctx->embedding_w && prev_token >= 0 && prev_token < V) {
        const float * emb = tensor_f32(ctx, ctx->embedding_w);
        memcpy(ds.embedded.data(), emb + prev_token * H, H * sizeof(float));
    }

    // 2. GRU1: st = gru1(embedded, hidden) → (256,)
    gru_cell(ds.embedded.data(), H,
             state.hidden.data(), H,
             tensor_f32(ctx, ctx->gru1_w_ih), tensor_f32(ctx, ctx->gru1_w_hh),
             tensor_f32(ctx, ctx->gru1_b_ih), tensor_f32(ctx, ctx->gru1_b_hh),
             ds.st.data());

    // 3. Query projection: hidden1 = Linear_hidden(st) → (256,)
    linear(ds.st.data(), H,
           tensor_f32(ctx, ctx->hidden_w), tensor_f32(ctx, ctx->hidden_b), H,
           ds.hidden1.data());

    // 4. Coverage conv: decoder_attention = conv1(prev_attention)
    {
        // Use alpha buffer as temp (will be overwritten later)
        conv2d(state.decoder_attention.data(), 1, ctx->enc_h, ctx->enc_w,
               tensor_f32(ctx, ctx->conv1_w), tensor_f32(ctx, ctx->conv1_b),
               1, 3, 3, 1, 1, ds.alpha.data(), ctx->enc_h, ctx->enc_w);
        memcpy(state.decoder_attention.data(), ds.alpha.data(), enc_n * sizeof(float));
    }

    // 5. attention_sum += decoder_attention
    for (int i = 0; i < enc_n; i++)
        state.attention_sum[i] += state.decoder_attention[i];

    // 6. Compute attention energy
    {
        const float * uf_W = tensor_f32(ctx, ctx->uf_w);
        const float * uf_B = tensor_f32(ctx, ctx->uf_b);

        for (int p = 0; p < enc_n; p++) {
            float as = state.attention_sum[p];
            for (int h = 0; h < H; h++) {
                float enc_part = state.enc_ua[p * H + h];
                float cov_part = as * uf_W[h] + uf_B[h];
                ds.et[p * H + h] = ds.hidden1[h] + enc_part + cov_part;
            }
        }
    }

    // Transpose et to CHW for conv_tan
    for (int c = 0; c < H; c++)
        for (int i = 0; i < enc_n; i++)
            ds.et_chw[c * enc_n + i] = ds.et[i * H + c];

    // conv_tan: (256, 256, 3, 3)
    conv2d(ds.et_chw.data(), H, ctx->enc_h, ctx->enc_w,
           tensor_f32(ctx, ctx->conv_tan_w), tensor_f32(ctx, ctx->conv_tan_b),
           H, 3, 3, 1, 1, ds.ct_out.data(), ctx->enc_h, ctx->enc_w);

    // BN1 + tanh
    apply_bn_scale(ds.ct_out.data(), H, enc_n,
                   tensor_f32(ctx, ctx->bn1_scale), tensor_f32(ctx, ctx->bn1_offset));
    for (int i = 0; i < H * enc_n; i++)
        ds.ct_out[i] = tanhf(ds.ct_out[i]);

    // Transpose back to (enc_n, 256)
    for (int c = 0; c < H; c++)
        for (int i = 0; i < enc_n; i++)
            ds.ct_hwc[i * H + c] = ds.ct_out[c * enc_n + i];

    // v: Linear(256, 1) → energy per position (SIMD dot product)
    {
        const float * v_W = tensor_f32(ctx, ctx->v_w);
        float v_B = tensor_f32(ctx, ctx->v_b)[0];
        for (int p = 0; p < enc_n; p++)
            ds.energy[p] = core_cpu::dot_product(&ds.ct_hwc[p * H], v_W, H) + v_B;
    }

    // Softmax → attention weights
    float max_e = ds.energy[0];
    for (int i = 1; i < enc_n; i++)
        if (ds.energy[i] > max_e) max_e = ds.energy[i];
    float sum_exp = 0;
    for (int i = 0; i < enc_n; i++) {
        ds.alpha[i] = expf(ds.energy[i] - max_e);
        sum_exp += ds.alpha[i];
    }
    float inv_sum = 1.0f / (sum_exp + 1e-8f);
    for (int i = 0; i < enc_n; i++)
        ds.alpha[i] *= inv_sum;

    // Store attention for next step's coverage
    memcpy(state.decoder_attention.data(), ds.alpha.data(), enc_n * sizeof(float));

    // Context vector: ct = sum(alpha * encoder_outputs) → (1024,)
    memset(ds.context.data(), 0, enc_ch * sizeof(float));
    for (int p = 0; p < enc_n; p++) {
        float a = ds.alpha[p];
        const float * enc = ctx->encoder_output.data() + p * enc_ch;
        for (int c = 0; c < enc_ch; c++)
            ds.context[c] += a * enc[c];
    }

    // GRU2: hidden_next = gru(context, st)
    gru_cell(ds.context.data(), enc_ch,
             ds.st.data(), H,
             tensor_f32(ctx, ctx->gru_w_ih), tensor_f32(ctx, ctx->gru_w_hh),
             tensor_f32(ctx, ctx->gru_b_ih), tensor_f32(ctx, ctx->gru_b_hh),
             state.hidden.data());

    // Output: out(hidden2 + embedded2 + ct2)
    linear(state.hidden.data(), H,
           tensor_f32(ctx, ctx->hidden2_w), tensor_f32(ctx, ctx->hidden2_b), 128,
           ds.h2.data());
    linear(ds.embedded.data(), H,
           tensor_f32(ctx, ctx->emb2_w), tensor_f32(ctx, ctx->emb2_b), 128,
           ds.e2.data());
    linear(ds.context.data(), enc_ch,
           tensor_f32(ctx, ctx->wc_w), tensor_f32(ctx, ctx->wc_b), 128,
           ds.c2.data());

    for (int i = 0; i < 128; i++)
        ds.combined[i] = ds.h2[i] + ds.e2[i] + ds.c2[i];

    linear(ds.combined.data(), 128,
           tensor_f32(ctx, ctx->out_w), tensor_f32(ctx, ctx->out_b), V,
           ds.logits.data());

    // Argmax
    int best = 0;
    float best_score = ds.logits[0];
    for (int v = 1; v < V; v++)
        if (ds.logits[v] > best_score) { best_score = ds.logits[v]; best = v; }

    // Confidence
    {
        float sum_e = 0;
        for (int v = 0; v < V; v++) sum_e += expf(ds.logits[v] - best_score);
        ctx->char_confidences.push_back(1.0f / sum_e);
    }

    return best;
}

// ---------------------------------------------------------------------------
// Allocate decoder scratch buffers (once per image, reused across all steps)
// ---------------------------------------------------------------------------

static void ensure_dec_scratch(hmer_ocr_context * ctx) {
    if (ctx->ds.allocated) return;
    const int H = ctx->hparams.hidden_size;  // 256
    const int V = ctx->hparams.output_size;  // 112
    const int enc_n = ctx->enc_h * ctx->enc_w;
    const int enc_ch = ctx->hparams.output_channels; // 1024

    ctx->ds.embedded.resize(H);
    ctx->ds.st.resize(H);
    ctx->ds.hidden1.resize(H);
    ctx->ds.et.resize(enc_n * H);
    ctx->ds.et_chw.resize(H * enc_n);
    ctx->ds.ct_out.resize(H * enc_n);
    ctx->ds.ct_hwc.resize(enc_n * H);
    ctx->ds.energy.resize(enc_n);
    ctx->ds.alpha.resize(enc_n);
    ctx->ds.context.resize(enc_ch);
    ctx->ds.h2.resize(128);
    ctx->ds.e2.resize(128);
    ctx->ds.c2.resize(128);
    ctx->ds.combined.resize(128);
    ctx->ds.logits.resize(V);
    ctx->ds.allocated = true;
}

// ---------------------------------------------------------------------------
// Greedy decoding loop
// ---------------------------------------------------------------------------

static std::string greedy_decode(hmer_ocr_context * ctx) {
    ctx->char_confidences.clear();
    const auto & hp = ctx->hparams;
    const int H = hp.hidden_size;
    const int enc_n = ctx->enc_h * ctx->enc_w;
    const int enc_ch = hp.output_channels;

    // Initialize decoder state
    decoder_state state;
    state.hidden.assign(H, 0.0f);  // zeros for inference
    state.attention_sum.assign(enc_n, 0.0f);
    state.decoder_attention.assign(enc_n, 0.0f);

    // Pre-allocate decoder scratch buffers
    ctx->ds.allocated = false;
    ensure_dec_scratch(ctx);

    // Precompute enc_ua = ua(encoder_output) → (enc_n, 256)
    state.enc_ua.resize(enc_n * H);
    {
        const float * ua_W = tensor_f32(ctx, ctx->ua_w);
        const float * ua_B = tensor_f32(ctx, ctx->ua_b);
        for (int p = 0; p < enc_n; p++) {
            core_cpu::linear_cpu(ctx->encoder_output.data() + p * enc_ch,
                                 state.enc_ua.data() + p * H,
                                 enc_ch, H, ua_W, ua_B);
        }
    }

    std::vector<int> tokens;
    int prev_token = hp.sos_token;

    for (int step = 0; step < hp.max_seq_len; step++) {
        int tok = decoder_step(ctx, prev_token, state);
        if (tok == hp.eol_token) {
            // Pop the EOL confidence pushed inside decoder_step
            if (!ctx->char_confidences.empty())
                ctx->char_confidences.pop_back();
            break;
        }
        tokens.push_back(tok);
        prev_token = tok;
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
// Public API
// ---------------------------------------------------------------------------

// Scale image to fit within max_pixels, preserving aspect ratio (bilinear).
static bool hmer_scale_to_fit(const float * src, int sw, int sh,
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

const char * hmer_ocr_recognize(
    hmer_ocr_context * ctx,
    const float * pixels, int width, int height,
    int * out_len
) {
    if (!ctx || !pixels || width <= 0 || height <= 0) return nullptr;

    const bool bench = ctx->bench;
    auto t_total = std::chrono::steady_clock::now();

    // Auto-detect polarity and invert if needed
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
        fprintf(stderr, "hmer_ocr: auto-inverted (mean=%.2f)\n", mean);
    }

    // Scale down large images (matches PyTorch training constraint)
    int w = width, h = height;
    std::vector<float> scaled;
    if (hmer_scale_to_fit(input, width, height, scaled, w, h)) {
        input = scaled.data();
        fprintf(stderr, "hmer_ocr: scaled %dx%d → %dx%d\n", width, height, w, h);
    }
    if (bench) fprintf(stderr, "[hmer-bench] preprocess: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    // Run encoder
    t0 = std::chrono::steady_clock::now();
    run_encoder(ctx, input, w, h);
    if (bench) fprintf(stderr, "[hmer-bench] encoder: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    // Run decoder
    t0 = std::chrono::steady_clock::now();
    ctx->result_buf = greedy_decode(ctx);
    if (bench) fprintf(stderr, "[hmer-bench] decoder: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count());

    if (bench) fprintf(stderr, "[hmer-bench] total: %.1f ms\n",
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t_total).count());

    if (out_len) *out_len = (int)ctx->result_buf.size();
    return ctx->result_buf.c_str();
}

const char * hmer_ocr_recognize_raw(
    hmer_ocr_context * ctx,
    const uint8_t * pixel_bytes, int width, int height, int channels,
    int * out_len
) {
    if (!ctx || !pixel_bytes || width <= 0 || height <= 0) return nullptr;

    std::vector<float> gray(width * height);
    for (int i = 0; i < width * height; i++) {
        if (channels == 1) {
            gray[i] = pixel_bytes[i] / 255.0f;
        } else if (channels >= 3) {
            int base = i * channels;
            gray[i] = (0.299f * pixel_bytes[base] +
                        0.587f * pixel_bytes[base + 1] +
                        0.114f * pixel_bytes[base + 2]) / 255.0f;
        }
    }
    return hmer_ocr_recognize(ctx, gray.data(), width, height, out_len);
}

const float * hmer_ocr_confidences(const hmer_ocr_context * ctx, int * n_chars) {
    if (!ctx || ctx->char_confidences.empty()) {
        if (n_chars) *n_chars = 0;
        return nullptr;
    }
    if (n_chars) *n_chars = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float hmer_ocr_mean_confidence(const hmer_ocr_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float c : ctx->char_confidences) sum += c;
    return (float)(sum / ctx->char_confidences.size());
}
