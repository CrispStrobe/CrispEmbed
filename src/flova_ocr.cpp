// flova_ocr.cpp — Flova/omr_transformer handwritten/whiteboard OMR via ggml.
//
// Donut VisionEncoderDecoder → LilyPond "simple notes":
//   Encoder : DonutSwin (Swin-Base scale) — patch 4, window 10, embed_dim 128,
//             depths [2,2,14,2], heads [4,8,16,32], hidden 1024, image 583×409.
//             CPU-scalar windowed attention (identical to src/mixtex_ocr.cpp's
//             DonutSwin; only the config differs), batched LN/linear via a small
//             ggml CPU graph. Final LayerNorm produces the cross-attn memory.
//   Decoder : mBART 4-layer PRE-norm — d_model 1024, 16 heads, ffn 4096, vocab 75,
//             learned positions (offset +2), scale_embedding (×√1024), GELU (erf).
//             decoder_start/bos 56 (<s>), eos 54 (</s>), pad 55.
//
// Loaded from a GGUF produced by models/convert-flova-to-gguf.py (arch "flova_ocr").
//
// KEY POINTS (validated vs tools/dump_flova_reference.py, CPU only):
//   • Patch embed pads H,W UP to a multiple of patch_size (583→584, 409→412);
//     grid 146×103 → stage0 downsample pads W 103→104 → 73×52 = 3796 tokens.
//   • Swin LayerNorm eps 1e-5, GELU = erf. Downsample = PatchMerging [TL,BL,TR,BR].
//   • mBART is PRE-norm (LN before each sublayer, residual around); layernorm eps
//     1e-5; embed = tokens·√1024 + positions[pos+2], then layernorm_embedding.
//   • eos is the tokenizer's </s> = 54 (NOT generation_config's stale mBART 2).

#include "flova_ocr.h"
#include "core/cpu_ops.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

extern "C" {
typedef unsigned char stbi_uc;
stbi_uc * stbi_load(char const * filename, int * x, int * y, int * channels_in_file, int desired_channels);
void stbi_image_free(void * retval_from_stbi_load);
}

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

using core_cpu::gelu_erf;
using core_cpu::layernorm_cpu;
using core_cpu::linear_cpu;
using core_cpu::softmax;
using core_cpu::to_f32;

// ---------------------------------------------------------------------------
// Hparams
// ---------------------------------------------------------------------------
struct flova_hparams {
    // encoder (DonutSwin)
    int patch_size = 4;
    int window_size = 10;
    int embed_dim = 128;
    int enc_hidden = 1024;
    int image_h = 583;
    int image_w = 409;
    int enc_depths[4] = { 2, 2, 14, 2 };
    int enc_heads[4] = { 4, 8, 16, 32 };
    // decoder (mBART)
    int dec_hidden = 1024;
    int dec_layers = 4;
    int dec_heads = 16;
    int dec_ffn = 4096;
    int vocab_size = 75;
    int max_position = 1536;
    int scale_embedding = 1;
    int bos_token = 56;
    int eos_token = 54;
    int pad_token = 55;
    int unk_token = 0;
    float image_mean = 0.5f;
    float image_std = 0.5f;
};

// ---------------------------------------------------------------------------
// Swin window helpers (identical DonutSwin math to mixtex_ocr.cpp)
// ---------------------------------------------------------------------------

// Window partition: [H, W, C] → [nWindows, ws², C]
static void window_partition(const float * x, float * out, int H, int W, int C, int ws) {
    int nH = H / ws, nW = W / ws;
    for (int wh = 0; wh < nH; wh++)
        for (int ww = 0; ww < nW; ww++) {
            int win_idx = wh * nW + ww;
            for (int y = 0; y < ws; y++)
                for (int x_pos = 0; x_pos < ws; x_pos++) {
                    int src_y = wh * ws + y;
                    int src_x = ww * ws + x_pos;
                    int token_idx = y * ws + x_pos;
                    memcpy(out + (win_idx * ws * ws + token_idx) * C, x + (src_y * W + src_x) * C, C * sizeof(float));
                }
        }
}

// Window reverse: [nWindows, ws², C] → [H, W, C]
static void window_reverse(const float * windows, float * out, int H, int W, int C, int ws) {
    int nH = H / ws, nW = W / ws;
    for (int wh = 0; wh < nH; wh++)
        for (int ww = 0; ww < nW; ww++) {
            int win_idx = wh * nW + ww;
            for (int y = 0; y < ws; y++)
                for (int x_pos = 0; x_pos < ws; x_pos++) {
                    int dst_y = wh * ws + y;
                    int dst_x = ww * ws + x_pos;
                    int token_idx = y * ws + x_pos;
                    memcpy(out + (dst_y * W + dst_x) * C, windows + (win_idx * ws * ws + token_idx) * C,
                           C * sizeof(float));
                }
        }
}

// Cyclic shift [H,W,C] by (shift_h, shift_w) with wrap. cyclic_shift(+s)=torch.roll(-s).
static void cyclic_shift(const float * in, float * out, int H, int W, int C, int shift_h, int shift_w) {
    for (int y = 0; y < H; y++) {
        int src_y = (y + shift_h + H) % H;
        for (int x = 0; x < W; x++) {
            int src_x = (x + shift_w + W) % W;
            memcpy(out + (y * W + x) * C, in + (src_y * W + src_x) * C, C * sizeof(float));
        }
    }
}

// Window multi-head self-attention with relative position bias + optional mask.
static void window_mhsa(const float * tokens, float * out, int n_tokens, int D, int n_heads, const float * q_w,
                        const float * q_b, const float * k_w, const float * k_b, const float * v_w, const float * v_b,
                        const float * out_w, const float * out_b, const float * rpb_table, const float * rpb_index,
                        int rpb_table_len, const float * attn_mask = nullptr) {
    int hd = D / n_heads;
    float scale = 1.0f / sqrtf((float)hd);

    std::vector<float> Q(n_tokens * D), K(n_tokens * D), V(n_tokens * D);
    for (int t = 0; t < n_tokens; t++) {
        linear_cpu(tokens + t * D, Q.data() + t * D, D, D, q_w, q_b);
        linear_cpu(tokens + t * D, K.data() + t * D, D, D, k_w, k_b);
        linear_cpu(tokens + t * D, V.data() + t * D, D, D, v_w, v_b);
    }

    std::vector<float> attn_out(n_tokens * D);
    for (int h = 0; h < n_heads; h++) {
        int off = h * hd;
        std::vector<float> scores(n_tokens * n_tokens);
        for (int i = 0; i < n_tokens; i++)
            for (int j = 0; j < n_tokens; j++) {
                float dot = core_cpu::dot_product(&Q[i * D + off], &K[j * D + off], hd);
                scores[i * n_tokens + j] = dot * scale;
                if (rpb_table && rpb_index) {
                    int idx = (int)rpb_index[i * n_tokens + j];
                    if (idx >= 0 && idx < rpb_table_len) scores[i * n_tokens + j] += rpb_table[idx * n_heads + h];
                }
                if (attn_mask) scores[i * n_tokens + j] += attn_mask[i * n_tokens + j];
            }
        for (int i = 0; i < n_tokens; i++) softmax(scores.data() + i * n_tokens, n_tokens);
        for (int i = 0; i < n_tokens; i++)
            for (int d = 0; d < hd; d++) {
                float sum = 0;
                for (int j = 0; j < n_tokens; j++) sum += scores[i * n_tokens + j] * V[j * D + off + d];
                attn_out[i * D + off + d] = sum;
            }
    }
    for (int t = 0; t < n_tokens; t++) linear_cpu(attn_out.data() + t * D, out + t * D, D, D, out_w, out_b);
}

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------
struct swin_block_weights {
    ggml_tensor *ln1_w, *ln1_b;
    ggml_tensor *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *out_w, *out_b;
    ggml_tensor *rpb_table, *rpb_index;
    ggml_tensor *ln2_w, *ln2_b;
    ggml_tensor *ffn_up_w, *ffn_up_b, *ffn_down_w, *ffn_down_b;
};
struct swin_downsample_weights {
    ggml_tensor *norm_w, *norm_b;
    ggml_tensor * reduction_w; // [2C, 4C], no bias
};
struct dec_layer_weights {
    ggml_tensor *self_ln_w, *self_ln_b;
    ggml_tensor *self_q_w, *self_q_b, *self_k_w, *self_k_b, *self_v_w, *self_v_b, *self_out_w, *self_out_b;
    ggml_tensor *cross_ln_w, *cross_ln_b;
    ggml_tensor *cross_q_w, *cross_q_b, *cross_k_w, *cross_k_b, *cross_v_w, *cross_v_b, *cross_out_w, *cross_out_b;
    ggml_tensor *ffn_ln_w, *ffn_ln_b;
    ggml_tensor *ffn_up_w, *ffn_up_b, *ffn_down_w, *ffn_down_b;
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
struct flova_ocr_context {
    flova_hparams hp;
    int n_threads = 4;

    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;

    // encoder
    ggml_tensor *patch_w, *patch_b, *patch_norm_w, *patch_norm_b;
    std::vector<swin_block_weights> stage_blocks[4];
    swin_downsample_weights downsample[3];
    ggml_tensor *enc_final_norm_w, *enc_final_norm_b;

    // decoder
    ggml_tensor *embed_tokens_w, *embed_pos_w;
    ggml_tensor *embed_ln_w, *embed_ln_b;
    dec_layer_weights dec_layers[4];
    ggml_tensor *dec_final_norm_w, *dec_final_norm_b;
    ggml_tensor * lm_head_w;

    std::vector<std::string> vocab;
    std::string output_text;

    // batched-matmul infrastructure for the encoder (CPU)
    ggml_backend_t enc_backend = nullptr;
    ggml_backend_sched_t enc_sched = nullptr;
    std::vector<uint8_t> enc_meta;
};

static ggml_tensor * find(const std::unordered_map<std::string, ggml_tensor *> & m, const char * name) {
    return core_gguf::try_get(m, name);
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------
flova_ocr_context * flova_ocr_init(const char * model_path, int n_threads) {
    auto * ctx = new flova_ocr_context{};
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    gguf_context * gc = core_gguf::open_metadata(model_path);
    if (!gc) {
        delete ctx;
        return nullptr;
    }
    auto & hp = ctx->hp;
    hp.patch_size = core_gguf::kv_u32(gc, "flova.encoder.patch_size", 4);
    hp.window_size = core_gguf::kv_u32(gc, "flova.encoder.window_size", 10);
    hp.embed_dim = core_gguf::kv_u32(gc, "flova.encoder.embed_dim", 128);
    hp.enc_hidden = core_gguf::kv_u32(gc, "flova.encoder.hidden_size", 1024);
    hp.image_h = core_gguf::kv_u32(gc, "flova.encoder.image_h", 583);
    hp.image_w = core_gguf::kv_u32(gc, "flova.encoder.image_w", 409);
    {
        auto d = core_gguf::kv_i32_array(gc, "flova.encoder.depths");
        auto h = core_gguf::kv_i32_array(gc, "flova.encoder.num_heads");
        for (int i = 0; i < 4; i++) {
            if ((int)d.size() == 4) hp.enc_depths[i] = d[i];
            if ((int)h.size() == 4) hp.enc_heads[i] = h[i];
        }
    }
    hp.dec_hidden = core_gguf::kv_u32(gc, "flova.decoder.hidden_size", 1024);
    hp.dec_layers = core_gguf::kv_u32(gc, "flova.decoder.num_layers", 4);
    hp.dec_heads = core_gguf::kv_u32(gc, "flova.decoder.num_heads", 16);
    hp.dec_ffn = core_gguf::kv_u32(gc, "flova.decoder.ffn_dim", 4096);
    hp.vocab_size = core_gguf::kv_u32(gc, "flova.decoder.vocab_size", 75);
    hp.max_position = core_gguf::kv_u32(gc, "flova.decoder.max_position", 1536);
    hp.scale_embedding = core_gguf::kv_u32(gc, "flova.decoder.scale_embedding", 1);
    hp.bos_token = core_gguf::kv_u32(gc, "flova.decoder_start_token", 56);
    hp.eos_token = core_gguf::kv_u32(gc, "flova.eos_token", 54);
    hp.pad_token = core_gguf::kv_u32(gc, "flova.pad_token", 55);
    hp.unk_token = core_gguf::kv_u32(gc, "flova.unk_token", 0);
    {
        auto mean = core_gguf::kv_f32_array(gc, "flova.image_mean");
        auto std = core_gguf::kv_f32_array(gc, "flova.image_std");
        if (!mean.empty()) hp.image_mean = mean[0];
        if (!std.empty()) hp.image_std = std[0];
    }
    ctx->vocab = core_gguf::kv_str_array(gc, "tokenizer.tokens");
    core_gguf::free_metadata(gc);

    fprintf(stderr, "flova_ocr: enc patch=%d win=%d embed=%d hidden=%d depths=[%d,%d,%d,%d] heads=[%d,%d,%d,%d]\n",
            hp.patch_size, hp.window_size, hp.embed_dim, hp.enc_hidden, hp.enc_depths[0], hp.enc_depths[1],
            hp.enc_depths[2], hp.enc_depths[3], hp.enc_heads[0], hp.enc_heads[1], hp.enc_heads[2], hp.enc_heads[3]);
    fprintf(stderr, "flova_ocr: dec hidden=%d layers=%d heads=%d ffn=%d vocab=%d(%zu) bos=%d eos=%d\n", hp.dec_hidden,
            hp.dec_layers, hp.dec_heads, hp.dec_ffn, hp.vocab_size, ctx->vocab.size(), hp.bos_token, hp.eos_token);

    // Weights load on CPU: the scalar encoder/decoder read them via to_f32
    // (dcache), and the small batched LN/linear graphs run on the CPU sched.
    ctx->backend = ggml_backend_cpu_init();
    if (!ctx->backend) {
        delete ctx;
        return nullptr;
    }
    ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    if (!core_gguf::load_weights(model_path, ctx->backend, "flova_ocr", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }

    const auto & m = ctx->wl.tensors;
    char buf[256];
    auto T = [&](const char * fmt, ...) -> ggml_tensor * {
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        return find(m, buf);
    };

    ctx->patch_w = find(m, "enc.patch.weight");
    ctx->patch_b = find(m, "enc.patch.bias");
    ctx->patch_norm_w = find(m, "enc.patch_norm.weight");
    ctx->patch_norm_b = find(m, "enc.patch_norm.bias");
    for (int s = 0; s < 4; s++) {
        ctx->stage_blocks[s].resize(hp.enc_depths[s]);
        for (int b = 0; b < hp.enc_depths[s]; b++) {
            auto & blk = ctx->stage_blocks[s][b];
            blk.ln1_w = T("enc.stage%d.block%d.ln1.weight", s, b);
            blk.ln1_b = T("enc.stage%d.block%d.ln1.bias", s, b);
            blk.q_w = T("enc.stage%d.block%d.attn.q.weight", s, b);
            blk.q_b = T("enc.stage%d.block%d.attn.q.bias", s, b);
            blk.k_w = T("enc.stage%d.block%d.attn.k.weight", s, b);
            blk.k_b = T("enc.stage%d.block%d.attn.k.bias", s, b);
            blk.v_w = T("enc.stage%d.block%d.attn.v.weight", s, b);
            blk.v_b = T("enc.stage%d.block%d.attn.v.bias", s, b);
            blk.out_w = T("enc.stage%d.block%d.attn.out.weight", s, b);
            blk.out_b = T("enc.stage%d.block%d.attn.out.bias", s, b);
            blk.rpb_table = T("enc.stage%d.block%d.attn.rpb_table", s, b);
            blk.rpb_index = T("enc.stage%d.block%d.attn.rpb_index", s, b);
            blk.ln2_w = T("enc.stage%d.block%d.ln2.weight", s, b);
            blk.ln2_b = T("enc.stage%d.block%d.ln2.bias", s, b);
            blk.ffn_up_w = T("enc.stage%d.block%d.ffn.up.weight", s, b);
            blk.ffn_up_b = T("enc.stage%d.block%d.ffn.up.bias", s, b);
            blk.ffn_down_w = T("enc.stage%d.block%d.ffn.down.weight", s, b);
            blk.ffn_down_b = T("enc.stage%d.block%d.ffn.down.bias", s, b);
        }
        if (s < 3) {
            ctx->downsample[s].norm_w = T("enc.stage%d.downsample.norm.weight", s);
            ctx->downsample[s].norm_b = T("enc.stage%d.downsample.norm.bias", s);
            ctx->downsample[s].reduction_w = T("enc.stage%d.downsample.reduction.weight", s);
        }
    }
    ctx->enc_final_norm_w = find(m, "enc.final_norm.weight");
    ctx->enc_final_norm_b = find(m, "enc.final_norm.bias");

    ctx->embed_tokens_w = find(m, "dec.embed_tokens.weight");
    ctx->embed_pos_w = find(m, "dec.embed_positions.weight");
    ctx->embed_ln_w = find(m, "dec.embed_ln.weight");
    ctx->embed_ln_b = find(m, "dec.embed_ln.bias");
    if (ctx->embed_tokens_w) hp.vocab_size = (int)ctx->embed_tokens_w->ne[1];
    for (int i = 0; i < hp.dec_layers; i++) {
        auto & l = ctx->dec_layers[i];
        l.self_ln_w = T("dec.layers.%d.self_ln.weight", i);
        l.self_ln_b = T("dec.layers.%d.self_ln.bias", i);
        l.self_q_w = T("dec.layers.%d.self_q.weight", i);
        l.self_q_b = T("dec.layers.%d.self_q.bias", i);
        l.self_k_w = T("dec.layers.%d.self_k.weight", i);
        l.self_k_b = T("dec.layers.%d.self_k.bias", i);
        l.self_v_w = T("dec.layers.%d.self_v.weight", i);
        l.self_v_b = T("dec.layers.%d.self_v.bias", i);
        l.self_out_w = T("dec.layers.%d.self_out.weight", i);
        l.self_out_b = T("dec.layers.%d.self_out.bias", i);
        l.cross_ln_w = T("dec.layers.%d.cross_ln.weight", i);
        l.cross_ln_b = T("dec.layers.%d.cross_ln.bias", i);
        l.cross_q_w = T("dec.layers.%d.cross_q.weight", i);
        l.cross_q_b = T("dec.layers.%d.cross_q.bias", i);
        l.cross_k_w = T("dec.layers.%d.cross_k.weight", i);
        l.cross_k_b = T("dec.layers.%d.cross_k.bias", i);
        l.cross_v_w = T("dec.layers.%d.cross_v.weight", i);
        l.cross_v_b = T("dec.layers.%d.cross_v.bias", i);
        l.cross_out_w = T("dec.layers.%d.cross_out.weight", i);
        l.cross_out_b = T("dec.layers.%d.cross_out.bias", i);
        l.ffn_ln_w = T("dec.layers.%d.ffn_ln.weight", i);
        l.ffn_ln_b = T("dec.layers.%d.ffn_ln.bias", i);
        l.ffn_up_w = T("dec.layers.%d.ffn.up.weight", i);
        l.ffn_up_b = T("dec.layers.%d.ffn.up.bias", i);
        l.ffn_down_w = T("dec.layers.%d.ffn.down.weight", i);
        l.ffn_down_b = T("dec.layers.%d.ffn.down.bias", i);
    }
    ctx->dec_final_norm_w = find(m, "dec.final_norm.weight");
    ctx->dec_final_norm_b = find(m, "dec.final_norm.bias");
    ctx->lm_head_w = find(m, "dec.lm_head.weight");

    // ggml encoder batched-matmul infrastructure (CPU)
    ctx->enc_backend = ggml_backend_cpu_init();
    if (ctx->enc_backend) {
        ggml_backend_cpu_set_n_threads(ctx->enc_backend, ctx->n_threads);
        ggml_backend_t backends[] = { ctx->enc_backend };
        ctx->enc_sched = ggml_backend_sched_new(backends, nullptr, 1, 4096, false, false);
    }

    fprintf(stderr, "flova_ocr: loaded %zu tensors from %s\n", ctx->wl.tensors.size(), model_path);
    return ctx;
}

void flova_ocr_free(flova_ocr_context * ctx) {
    if (!ctx) return;
    if (ctx->enc_sched) ggml_backend_sched_free(ctx->enc_sched);
    if (ctx->enc_backend) ggml_backend_free(ctx->enc_backend);
    core_gguf::free_weights(ctx->wl);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

// ---------------------------------------------------------------------------
// Batched linear / layernorm via a tiny ggml CPU graph (encoder hot path)
// ---------------------------------------------------------------------------
static void batch_linear(flova_ocr_context * ctx, const float * input, int N, int in_D, ggml_tensor * weight,
                         ggml_tensor * bias, int out_D, float * output) {
    if (!ctx->enc_sched) {
        auto wf = to_f32(weight);
        auto bf = bias ? to_f32(bias) : std::vector<float>();
        for (int i = 0; i < N; i++)
            linear_cpu(input + i * in_D, output + i * out_D, in_D, out_D, wf.data(), bf.empty() ? nullptr : bf.data());
        return;
    }
    int max_nodes = 32;
    size_t buf_size = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ctx->enc_meta.resize(std::max(ctx->enc_meta.size(), buf_size));
    ggml_init_params ip = { buf_size, ctx->enc_meta.data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(g, max_nodes, false);

    ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, in_D, N);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_tensor * out = ggml_mul_mat(g, weight, x); // [out_D, N]
    if (bias) {
        ggml_tensor * b_input = ggml_new_tensor_2d(g, GGML_TYPE_F32, out_D, N);
        ggml_set_name(b_input, "b");
        ggml_set_input(b_input);
        out = ggml_add(g, out, b_input);
    }
    ggml_set_name(out, "out");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    ggml_backend_sched_reset(ctx->enc_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->enc_sched, gf)) {
        fprintf(stderr, "flova: batch_linear alloc failed\n");
        ggml_free(g);
        return;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), input, 0, (size_t)N * in_D * sizeof(float));
    if (bias) {
        auto bv = to_f32(bias);
        std::vector<float> b_data((size_t)N * out_D);
        for (int i = 0; i < N; i++) memcpy(b_data.data() + i * out_D, bv.data(), out_D * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "b"), b_data.data(), 0, (size_t)N * out_D * sizeof(float));
    }
    ggml_backend_sched_graph_compute(ctx->enc_sched, gf);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out"), output, 0, (size_t)N * out_D * sizeof(float));
    ggml_free(g);
}

static void batch_layernorm(flova_ocr_context * ctx, const float * input, int N, int D, ggml_tensor * w_t,
                            ggml_tensor * b_t, float * output) {
    if (!ctx->enc_sched) {
        auto wf = to_f32(w_t), bf = to_f32(b_t);
        for (int i = 0; i < N; i++) layernorm_cpu(input + i * D, output + i * D, D, wf.data(), bf.data(), 1e-5f);
        return;
    }
    int max_nodes = 32;
    size_t buf_size = ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false);
    ctx->enc_meta.resize(std::max(ctx->enc_meta.size(), buf_size));
    ggml_init_params ip = { buf_size, ctx->enc_meta.data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(g, max_nodes, false);

    ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, N);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    ggml_tensor * norm = ggml_norm(g, x, 1e-5f);
    ggml_tensor * w = (w_t->type != GGML_TYPE_F32) ? ggml_cast(g, w_t, GGML_TYPE_F32) : w_t;
    ggml_tensor * b = (b_t->type != GGML_TYPE_F32) ? ggml_cast(g, b_t, GGML_TYPE_F32) : b_t;
    norm = ggml_mul(g, norm, w);
    norm = ggml_add(g, norm, b);
    ggml_set_name(norm, "out");
    ggml_set_output(norm);
    ggml_build_forward_expand(gf, norm);

    ggml_backend_sched_reset(ctx->enc_sched);
    if (!ggml_backend_sched_alloc_graph(ctx->enc_sched, gf)) {
        ggml_free(g);
        return;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "x"), input, 0, (size_t)N * D * sizeof(float));
    ggml_backend_sched_graph_compute(ctx->enc_sched, gf);
    ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "out"), output, 0, (size_t)N * D * sizeof(float));
    ggml_free(g);
}

// ---------------------------------------------------------------------------
// DonutSwin encoder (CPU scalar windowed attention). Returns [N, 1024].
// When dref is set, compares enc_stage0..3 + enc_output and counts failures.
// ---------------------------------------------------------------------------
static std::vector<float> run_swin_encoder(flova_ocr_context * ctx, const float * pixels_chw, int img_h, int img_w,
                                           crispembed_diff::Ref * dref, int * fails) {
    auto & hp = ctx->hp;
    int D = hp.embed_dim;    // 128
    int ws = hp.window_size; // 10
    int ps = hp.patch_size;  // 4

    // Patch embed: Conv2d(3, D, ps×ps, stride ps). DonutSwin pads H,W UP to a
    // multiple of patch_size (zero pad), so use ceil-div and guard reads.
    int pH = (img_h + ps - 1) / ps;
    int pW = (img_w + ps - 1) / ps;
    int N = pH * pW;

    auto patch_w = to_f32(ctx->patch_w); // [D, 3, ps, ps]
    auto patch_b = to_f32(ctx->patch_b); // [D]

    std::vector<float> patches((size_t)N * D);
    for (int py = 0; py < pH; py++)
        for (int px = 0; px < pW; px++) {
            int pos = py * pW + px;
            for (int oc = 0; oc < D; oc++) {
                float sum = patch_b[oc];
                for (int ic = 0; ic < 3; ic++)
                    for (int ky = 0; ky < ps; ky++)
                        for (int kx = 0; kx < ps; kx++) {
                            int iy = py * ps + ky;
                            int ix = px * ps + kx;
                            if (iy >= img_h || ix >= img_w) continue; // zero-padded region
                            float pixel = pixels_chw[(size_t)ic * img_h * img_w + (size_t)iy * img_w + ix];
                            sum += pixel * patch_w[((size_t)oc * 3 + ic) * ps * ps + ky * ps + kx];
                        }
                patches[(size_t)pos * D + oc] = sum;
            }
        }

    batch_layernorm(ctx, patches.data(), N, D, ctx->patch_norm_w, ctx->patch_norm_b, patches.data());

    int H = pH, W = pW;
    std::vector<float> x = std::move(patches);

    for (int stage = 0; stage < 4; stage++) {
        int n_blocks = hp.enc_depths[stage];
        int n_heads = hp.enc_heads[stage];

        for (int bi = 0; bi < n_blocks; bi++) {
            auto & blk = ctx->stage_blocks[stage][bi];
            int HW = H * W;

            std::vector<float> normed((size_t)HW * D);
            batch_layernorm(ctx, x.data(), HW, D, blk.ln1_w, blk.ln1_b, normed.data());

            bool shifted = (bi % 2 == 1);
            int shift = shifted ? ws / 2 : 0;

            int pad_h = (ws - H % ws) % ws;
            int pad_w = (ws - W % ws) % ws;
            int pH2 = H + pad_h, pW2 = W + pad_w;

            std::vector<float> shifted_x;
            if (pad_h > 0 || pad_w > 0) {
                shifted_x.assign((size_t)pH2 * pW2 * D, 0.0f);
                for (int y = 0; y < H; y++)
                    memcpy(shifted_x.data() + (size_t)y * pW2 * D, normed.data() + (size_t)y * W * D,
                           (size_t)W * D * sizeof(float));
            } else {
                shifted_x = normed;
            }

            if (shifted) {
                std::vector<float> tmp((size_t)pH2 * pW2 * D);
                cyclic_shift(shifted_x.data(), tmp.data(), pH2, pW2, D, shift, shift);
                shifted_x = std::move(tmp);
            }

            std::vector<float> attn_mask_all;
            if (shifted) {
                std::vector<float> img_mask((size_t)pH2 * pW2, 0.0f);
                int h_cuts[4] = { 0, pH2 - ws, pH2 - shift, pH2 };
                int w_cuts[4] = { 0, pW2 - ws, pW2 - shift, pW2 };
                int region = 0;
                for (int hi = 0; hi < 3; hi++)
                    for (int wi = 0; wi < 3; wi++) {
                        for (int y = h_cuts[hi]; y < h_cuts[hi + 1]; y++)
                            for (int xx = w_cuts[wi]; xx < w_cuts[wi + 1]; xx++)
                                img_mask[(size_t)y * pW2 + xx] = (float)region;
                        region++;
                    }
                int nH_m = pH2 / ws, nW_m = pW2 / ws, n_win = nH_m * nW_m;
                std::vector<float> mask_windows((size_t)n_win * ws * ws);
                for (int wh = 0; wh < nH_m; wh++)
                    for (int ww = 0; ww < nW_m; ww++) {
                        int win = wh * nW_m + ww;
                        for (int y = 0; y < ws; y++)
                            for (int xx = 0; xx < ws; xx++)
                                mask_windows[(size_t)win * ws * ws + y * ws + xx] =
                                    img_mask[(size_t)(wh * ws + y) * pW2 + ww * ws + xx];
                    }
                int tpw = ws * ws;
                attn_mask_all.assign((size_t)n_win * tpw * tpw, 0.0f);
                for (int w = 0; w < n_win; w++)
                    for (int i = 0; i < tpw; i++)
                        for (int j = 0; j < tpw; j++)
                            if (mask_windows[(size_t)w * tpw + i] != mask_windows[(size_t)w * tpw + j])
                                attn_mask_all[(size_t)w * tpw * tpw + i * tpw + j] = -100.0f;
            }

            int nH = pH2 / ws, nW = pW2 / ws;
            int n_windows = nH * nW;
            int tokens_per_win = ws * ws;
            std::vector<float> windows((size_t)n_windows * tokens_per_win * D);
            window_partition(shifted_x.data(), windows.data(), pH2, pW2, D, ws);

            auto q_w = to_f32(blk.q_w), q_b = to_f32(blk.q_b);
            auto k_w = to_f32(blk.k_w), k_b = to_f32(blk.k_b);
            auto v_w = to_f32(blk.v_w), v_b = to_f32(blk.v_b);
            auto out_w = to_f32(blk.out_w), out_b = to_f32(blk.out_b);
            auto rpb_t = to_f32(blk.rpb_table);
            auto rpb_i = to_f32(blk.rpb_index);
            int rpb_len = blk.rpb_table ? (int)blk.rpb_table->ne[1] : 0;

            std::vector<float> attn_out((size_t)n_windows * tokens_per_win * D);
            auto run_window = [&](int w) {
                const float * win_mask = (!attn_mask_all.empty())
                                             ? attn_mask_all.data() + (size_t)w * tokens_per_win * tokens_per_win
                                             : nullptr;
                window_mhsa(windows.data() + (size_t)w * tokens_per_win * D,
                            attn_out.data() + (size_t)w * tokens_per_win * D, tokens_per_win, D, n_heads, q_w.data(),
                            q_b.data(), k_w.data(), k_b.data(), v_w.data(), v_b.data(), out_w.data(), out_b.data(),
                            rpb_t.empty() ? nullptr : rpb_t.data(), rpb_i.empty() ? nullptr : rpb_i.data(), rpb_len,
                            win_mask);
            };
            int n_thr = std::min(ctx->n_threads, n_windows);
            if (n_thr <= 1) {
                for (int w = 0; w < n_windows; w++) run_window(w);
            } else {
                std::atomic<int> next{ 0 };
                std::vector<std::thread> pool;
                pool.reserve(n_thr);
                for (int t = 0; t < n_thr; t++)
                    pool.emplace_back([&]() {
                        int w;
                        while ((w = next.fetch_add(1)) < n_windows) run_window(w);
                    });
                for (auto & th : pool) th.join();
            }

            std::vector<float> merged((size_t)pH2 * pW2 * D);
            window_reverse(attn_out.data(), merged.data(), pH2, pW2, D, ws);

            if (shifted) {
                std::vector<float> unshifted((size_t)pH2 * pW2 * D);
                cyclic_shift(merged.data(), unshifted.data(), pH2, pW2, D, -shift, -shift);
                merged = std::move(unshifted);
            }
            if (pad_h > 0 || pad_w > 0) {
                std::vector<float> unpadded((size_t)H * W * D);
                for (int y = 0; y < H; y++)
                    memcpy(unpadded.data() + (size_t)y * W * D, merged.data() + (size_t)y * pW2 * D,
                           (size_t)W * D * sizeof(float));
                merged = std::move(unpadded);
            }

            for (size_t i = 0; i < (size_t)HW * D; i++) x[i] += merged[i];

            // FFN: LN → up → GELU → down → residual
            int ffn_dim = (int)blk.ffn_up_w->ne[1];
            std::vector<float> ln_out((size_t)HW * D);
            batch_layernorm(ctx, x.data(), HW, D, blk.ln2_w, blk.ln2_b, ln_out.data());
            std::vector<float> up((size_t)HW * ffn_dim);
            batch_linear(ctx, ln_out.data(), HW, D, blk.ffn_up_w, blk.ffn_up_b, ffn_dim, up.data());
            for (size_t j = 0; j < (size_t)HW * ffn_dim; j++) up[j] = gelu_erf(up[j]);
            std::vector<float> down((size_t)HW * D);
            batch_linear(ctx, up.data(), HW, ffn_dim, blk.ffn_down_w, blk.ffn_down_b, D, down.data());
            for (size_t i = 0; i < (size_t)HW * D; i++) x[i] += down[i];
        }

        // Downsample (PatchMerging) between stages
        if (stage < 3) {
            auto & ds = ctx->downsample[stage];
            int new_D = D * 2;
            int padH = H + (H % 2);
            int padW = W + (W % 2);
            std::vector<float> padded;
            const float * merge_src = x.data();
            if (padH != H || padW != W) {
                padded.assign((size_t)padH * padW * D, 0.0f);
                for (int y = 0; y < H; y++)
                    memcpy(padded.data() + (size_t)y * padW * D, x.data() + (size_t)y * W * D,
                           (size_t)W * D * sizeof(float));
                merge_src = padded.data();
            }
            int newH = padH / 2, newW = padW / 2;
            int newN = newH * newW;
            std::vector<float> merged((size_t)newN * 4 * D);
            for (int y = 0; y < newH; y++)
                for (int xi = 0; xi < newW; xi++) {
                    int dst = y * newW + xi;
                    int s0 = (2 * y) * padW + (2 * xi);         // top-left
                    int s1 = (2 * y + 1) * padW + (2 * xi);     // bottom-left
                    int s2 = (2 * y) * padW + (2 * xi + 1);     // top-right
                    int s3 = (2 * y + 1) * padW + (2 * xi + 1); // bottom-right
                    memcpy(merged.data() + (size_t)dst * 4 * D + 0 * D, merge_src + (size_t)s0 * D, D * sizeof(float));
                    memcpy(merged.data() + (size_t)dst * 4 * D + 1 * D, merge_src + (size_t)s1 * D, D * sizeof(float));
                    memcpy(merged.data() + (size_t)dst * 4 * D + 2 * D, merge_src + (size_t)s2 * D, D * sizeof(float));
                    memcpy(merged.data() + (size_t)dst * 4 * D + 3 * D, merge_src + (size_t)s3 * D, D * sizeof(float));
                }
            batch_layernorm(ctx, merged.data(), newN, 4 * D, ds.norm_w, ds.norm_b, merged.data());
            std::vector<float> reduced((size_t)newN * new_D);
            batch_linear(ctx, merged.data(), newN, 4 * D, ds.reduction_w, nullptr, new_D, reduced.data());
            x = std::move(reduced);
            H = newH;
            W = newW;
            D = new_D;
        }

        if (dref && fails) {
            char nm[32];
            snprintf(nm, sizeof(nm), "enc_stage%d", stage);
            if (dref->has(nm)) {
                auto r = dref->compare(nm, x.data(), (size_t)H * W * D, 0);
                fprintf(stderr, "[flova-diff] %-11s (%dx%dx%d) cos_min=%.6f cos_mean=%.6f max_abs=%.2e %s\n", nm, H, W,
                        D, r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                if (!r.is_pass()) (*fails)++;
            }
        }
    }

    // Final LayerNorm → cross-attn memory
    if (ctx->enc_final_norm_w) {
        int N_out = H * W;
        batch_layernorm(ctx, x.data(), N_out, D, ctx->enc_final_norm_w, ctx->enc_final_norm_b, x.data());
    }

    if (dref && fails && dref->has("enc_output")) {
        auto r = dref->compare("enc_output", x.data(), (size_t)H * W * D, 0);
        fprintf(stderr, "[flova-diff] %-11s (%dx%d)      cos_min=%.6f cos_mean=%.6f max_abs=%.2e %s\n", "enc_output",
                H * W, D, r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        if (!r.is_pass()) (*fails)++;
    }
    return x; // [H*W, 1024]
}

// ---------------------------------------------------------------------------
// mBART decoder (scalar, pre-norm). Teacher-forced over the full id sequence.
// ---------------------------------------------------------------------------

// Multi-head attention: q[Lq*D], k[Lk*D], v[Lk*D] pre-projected → out[Lq*D].
// causal masks j>i (valid when query pos i aligns with key pos i).
static void mha_full(const float * q, const float * k, const float * v, float * out, int Lq, int Lk, int D, int nh,
                     bool causal) {
    int hd = D / nh;
    float scale = 1.0f / sqrtf((float)hd);
    std::vector<float> scores(Lk);
    for (int h = 0; h < nh; h++) {
        int off = h * hd;
        for (int i = 0; i < Lq; i++) {
            int jmax = causal ? i + 1 : Lk;
            for (int j = 0; j < jmax; j++)
                scores[j] = core_cpu::dot_product(&q[(size_t)i * D + off], &k[(size_t)j * D + off], hd) * scale;
            softmax(scores.data(), jmax);
            for (int d = 0; d < hd; d++) {
                float s = 0;
                for (int j = 0; j < jmax; j++) s += scores[j] * v[(size_t)j * D + off + d];
                out[(size_t)i * D + off + d] = s;
            }
        }
    }
}

struct CrossKV {
    std::vector<float> K, V; // [enc_len * D]
};

// Cache of per-layer dequantized decoder weights (constant across greedy steps).
struct DecW {
    std::vector<float> sq_w, sq_b, sk_w, sk_b, sv_w, sv_b, so_w, so_b, sln_w, sln_b;
    std::vector<float> cq_w, cq_b, co_w, co_b, cln_w, cln_b;
    std::vector<float> fu_w, fu_b, fd_w, fd_b, fln_w, fln_b;
};

// Forward the decoder over ids[L] with cached cross-KV. Fills logits_all[L*vocab].
// When dref is set, compares dec_block0..3 + logits and counts failures.
static void forward_decoder(flova_ocr_context * ctx, const std::vector<int> & ids, const float * enc_output,
                            int enc_len, const std::vector<CrossKV> & ckv, const std::vector<DecW> & dw,
                            const std::vector<float> & tok_w, const std::vector<float> & pos_w,
                            const std::vector<float> & eln_w, const std::vector<float> & eln_b,
                            const std::vector<float> & fn_w, const std::vector<float> & fn_b,
                            const std::vector<float> & lm_w, std::vector<float> & logits_all,
                            crispembed_diff::Ref * dref, int * fails) {
    auto & hp = ctx->hp;
    int D = hp.dec_hidden;
    int L = (int)ids.size();
    int nh = hp.dec_heads;
    int vocab = hp.vocab_size;
    float embed_scale = hp.scale_embedding ? sqrtf((float)D) : 1.0f;

    // Embedding: tokens·scale + positions[pos+2], then layernorm_embedding.
    std::vector<float> hidden((size_t)L * D);
    for (int i = 0; i < L; i++) {
        int tid = ids[i];
        int pos = i + 2; // mBART learned-position offset
        for (int d = 0; d < D; d++)
            hidden[(size_t)i * D + d] = tok_w[(size_t)tid * D + d] * embed_scale + pos_w[(size_t)pos * D + d];
        layernorm_cpu(&hidden[(size_t)i * D], &hidden[(size_t)i * D], D, eln_w.data(), eln_b.data(), 1e-5f);
    }

    std::vector<float> xn((size_t)L * D), q((size_t)L * D), k((size_t)L * D), vv((size_t)L * D), attn((size_t)L * D),
        proj((size_t)L * D);
    for (int li = 0; li < hp.dec_layers; li++) {
        const DecW & d = dw[li];

        // --- self-attention (pre-norm) ---
        for (int i = 0; i < L; i++)
            layernorm_cpu(&hidden[(size_t)i * D], &xn[(size_t)i * D], D, d.sln_w.data(), d.sln_b.data(), 1e-5f);
        for (int i = 0; i < L; i++) {
            linear_cpu(&xn[(size_t)i * D], &q[(size_t)i * D], D, D, d.sq_w.data(), d.sq_b.data());
            linear_cpu(&xn[(size_t)i * D], &k[(size_t)i * D], D, D, d.sk_w.data(), d.sk_b.data());
            linear_cpu(&xn[(size_t)i * D], &vv[(size_t)i * D], D, D, d.sv_w.data(), d.sv_b.data());
        }
        mha_full(q.data(), k.data(), vv.data(), attn.data(), L, L, D, nh, /*causal*/ true);
        for (int i = 0; i < L; i++)
            linear_cpu(&attn[(size_t)i * D], &proj[(size_t)i * D], D, D, d.so_w.data(), d.so_b.data());
        for (size_t j = 0; j < (size_t)L * D; j++) hidden[j] += proj[j];

        // --- cross-attention (pre-norm, no mask) ---
        for (int i = 0; i < L; i++)
            layernorm_cpu(&hidden[(size_t)i * D], &xn[(size_t)i * D], D, d.cln_w.data(), d.cln_b.data(), 1e-5f);
        for (int i = 0; i < L; i++)
            linear_cpu(&xn[(size_t)i * D], &q[(size_t)i * D], D, D, d.cq_w.data(), d.cq_b.data());
        mha_full(q.data(), ckv[li].K.data(), ckv[li].V.data(), attn.data(), L, enc_len, D, nh, /*causal*/ false);
        for (int i = 0; i < L; i++)
            linear_cpu(&attn[(size_t)i * D], &proj[(size_t)i * D], D, D, d.co_w.data(), d.co_b.data());
        for (size_t j = 0; j < (size_t)L * D; j++) hidden[j] += proj[j];

        // --- FFN (pre-norm) ---
        int ffn = hp.dec_ffn;
        std::vector<float> up((size_t)L * ffn), fdown((size_t)L * D);
        for (int i = 0; i < L; i++)
            layernorm_cpu(&hidden[(size_t)i * D], &xn[(size_t)i * D], D, d.fln_w.data(), d.fln_b.data(), 1e-5f);
        for (int i = 0; i < L; i++) {
            linear_cpu(&xn[(size_t)i * D], &up[(size_t)i * ffn], D, ffn, d.fu_w.data(), d.fu_b.data());
            for (int j = 0; j < ffn; j++) up[(size_t)i * ffn + j] = gelu_erf(up[(size_t)i * ffn + j]);
            linear_cpu(&up[(size_t)i * ffn], &fdown[(size_t)i * D], ffn, D, d.fd_w.data(), d.fd_b.data());
        }
        for (size_t j = 0; j < (size_t)L * D; j++) hidden[j] += fdown[j];

        if (dref && fails) {
            char nm[24];
            snprintf(nm, sizeof(nm), "dec_block%d", li);
            if (dref->has(nm)) {
                auto r = dref->compare(nm, hidden.data(), (size_t)L * D, 0);
                fprintf(stderr, "[flova-diff] %-11s (%dx%d)       cos_min=%.6f cos_mean=%.6f max_abs=%.2e %s\n", nm, L,
                        D, r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                if (!r.is_pass()) (*fails)++;
            }
        }
    }

    // Final norm → lm_head
    std::vector<float> fn((size_t)L * D);
    for (int i = 0; i < L; i++)
        layernorm_cpu(&hidden[(size_t)i * D], &fn[(size_t)i * D], D, fn_w.data(), fn_b.data(), 1e-5f);
    logits_all.assign((size_t)L * vocab, 0.0f);
    for (int i = 0; i < L; i++)
        linear_cpu(&fn[(size_t)i * D], &logits_all[(size_t)i * vocab], D, vocab, lm_w.data(), nullptr);

    if (dref && fails && dref->has("logits")) {
        auto r = dref->compare("logits", logits_all.data(), (size_t)L * vocab, 0);
        fprintf(stderr, "[flova-diff] %-11s (%dx%d)         cos_min=%.6f cos_mean=%.6f max_abs=%.2e %s\n", "logits", L,
                vocab, r.cos_min, r.cos_mean, r.max_abs, r.is_pass() ? "PASS" : "FAIL");
        if (!r.is_pass()) (*fails)++;
    }
}

static void precompute_cross(flova_ocr_context * ctx, const float * enc_output, int enc_len,
                             const std::vector<DecW> & dw, std::vector<CrossKV> & ckv) {
    auto & hp = ctx->hp;
    int D = hp.dec_hidden;
    ckv.resize(hp.dec_layers);
    for (int li = 0; li < hp.dec_layers; li++) {
        auto & l = ctx->dec_layers[li];
        auto ck_w = to_f32(l.cross_k_w), ck_b = to_f32(l.cross_k_b);
        auto cv_w = to_f32(l.cross_v_w), cv_b = to_f32(l.cross_v_b);
        ckv[li].K.resize((size_t)enc_len * D);
        ckv[li].V.resize((size_t)enc_len * D);
        for (int t = 0; t < enc_len; t++) {
            linear_cpu(enc_output + (size_t)t * D, &ckv[li].K[(size_t)t * D], D, D, ck_w.data(), ck_b.data());
            linear_cpu(enc_output + (size_t)t * D, &ckv[li].V[(size_t)t * D], D, D, cv_w.data(), cv_b.data());
        }
    }
}

// Dequantize all constant decoder weights once.
static void build_decw(flova_ocr_context * ctx, std::vector<DecW> & dw) {
    dw.resize(ctx->hp.dec_layers);
    for (int li = 0; li < ctx->hp.dec_layers; li++) {
        auto & l = ctx->dec_layers[li];
        auto & d = dw[li];
        d.sq_w = to_f32(l.self_q_w), d.sq_b = to_f32(l.self_q_b);
        d.sk_w = to_f32(l.self_k_w), d.sk_b = to_f32(l.self_k_b);
        d.sv_w = to_f32(l.self_v_w), d.sv_b = to_f32(l.self_v_b);
        d.so_w = to_f32(l.self_out_w), d.so_b = to_f32(l.self_out_b);
        d.sln_w = to_f32(l.self_ln_w), d.sln_b = to_f32(l.self_ln_b);
        d.cq_w = to_f32(l.cross_q_w), d.cq_b = to_f32(l.cross_q_b);
        d.co_w = to_f32(l.cross_out_w), d.co_b = to_f32(l.cross_out_b);
        d.cln_w = to_f32(l.cross_ln_w), d.cln_b = to_f32(l.cross_ln_b);
        d.fu_w = to_f32(l.ffn_up_w), d.fu_b = to_f32(l.ffn_up_b);
        d.fd_w = to_f32(l.ffn_down_w), d.fd_b = to_f32(l.ffn_down_b);
        d.fln_w = to_f32(l.ffn_ln_w), d.fln_b = to_f32(l.ffn_ln_b);
    }
}

static int argmax(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

// ---------------------------------------------------------------------------
// Detokenize: concat token strings (skip specials), </w> → space, strip.
// ---------------------------------------------------------------------------
static std::string detok(flova_ocr_context * ctx, const std::vector<int> & ids) {
    auto & hp = ctx->hp;
    std::string raw;
    for (size_t i = 1; i < ids.size(); i++) { // skip seeded bos
        int tid = ids[i];
        if (tid < 0 || tid >= (int)ctx->vocab.size()) continue;
        if (tid == hp.bos_token || tid == hp.eos_token || tid == hp.pad_token || tid == hp.unk_token) continue;
        raw += ctx->vocab[tid];
    }
    // Replace the "</w>" word-boundary marker with a space.
    std::string out;
    const std::string mark = "</w>";
    for (size_t i = 0; i < raw.size();) {
        if (raw.compare(i, mark.size(), mark) == 0) {
            out += ' ';
            i += mark.size();
        } else {
            out += raw[i++];
        }
    }
    size_t a = out.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = out.find_last_not_of(" \t\r\n");
    return out.substr(a, b - a + 1);
}

// ---------------------------------------------------------------------------
// Greedy decode: start at bos, argmax last-position logits, stop on eos.
// ---------------------------------------------------------------------------
static std::string decode_greedy(flova_ocr_context * ctx, const float * enc_output, int enc_len) {
    auto & hp = ctx->hp;
    std::vector<DecW> dw;
    build_decw(ctx, dw);
    std::vector<CrossKV> ckv;
    precompute_cross(ctx, enc_output, enc_len, dw, ckv);

    auto tok_w = to_f32(ctx->embed_tokens_w);
    auto pos_w = to_f32(ctx->embed_pos_w);
    auto eln_w = to_f32(ctx->embed_ln_w), eln_b = to_f32(ctx->embed_ln_b);
    auto fn_w = to_f32(ctx->dec_final_norm_w), fn_b = to_f32(ctx->dec_final_norm_b);
    auto lm_w = to_f32(ctx->lm_head_w);

    std::vector<int> ids = { hp.bos_token };
    int cap = std::min(hp.max_position, 256);
    std::vector<float> logits_all;
    for (int step = 0; step < cap; step++) {
        forward_decoder(ctx, ids, enc_output, enc_len, ckv, dw, tok_w, pos_w, eln_w, eln_b, fn_w, fn_b, lm_w,
                        logits_all, nullptr, nullptr);
        int L = (int)ids.size();
        int best = argmax(&logits_all[(size_t)(L - 1) * hp.vocab_size], hp.vocab_size);
        if (best == hp.eos_token) break;
        ids.push_back(best);
    }
    return detok(ctx, ids);
}

// ---------------------------------------------------------------------------
// Preprocessing (DonutImageProcessor) → CHW float [-1,1] at (image_h, image_w).
// Validated end-to-end against ref pixel_values via flova_ocr_run_diff; the
// native file path mirrors align_long_axis → thumbnail → pad → normalize.
// ---------------------------------------------------------------------------
static float bilinear_sample(const std::vector<float> & img, int H, int W, int C, float fy, float fx, int c) {
    int y0 = (int)std::floor(fy), x0 = (int)std::floor(fx);
    int y1 = std::min(y0 + 1, H - 1), x1 = std::min(x0 + 1, W - 1);
    y0 = std::max(0, std::min(y0, H - 1));
    x0 = std::max(0, std::min(x0, W - 1));
    float wy = fy - std::floor(fy), wx = fx - std::floor(fx);
    auto px = [&](int y, int x) { return img[((size_t)y * W + x) * C + c]; };
    float top = px(y0, x0) + wx * (px(y0, x1) - px(y0, x0));
    float bot = px(y1, x0) + wx * (px(y1, x1) - px(y1, x0));
    return top + wy * (bot - top);
}

static std::vector<float> preprocess_donut(flova_ocr_context * ctx, const uint8_t * pixels, int w, int h, int ch) {
    auto & hp = ctx->hp;
    int TH = hp.image_h, TW = hp.image_w;

    // to float RGB HWC
    std::vector<float> img((size_t)h * w * 3);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const uint8_t * s = pixels + ((size_t)y * w + x) * ch;
            float r, g, b;
            if (ch == 1) {
                r = g = b = s[0];
            } else {
                r = s[0];
                g = s[1];
                b = s[2];
            }
            img[((size_t)y * w + x) * 3 + 0] = r;
            img[((size_t)y * w + x) * 3 + 1] = g;
            img[((size_t)y * w + x) * 3 + 2] = b;
        }

    // align_long_axis: target is portrait (TH>TW); if input is landscape rotate 90° CW.
    if ((TW < TH) && (w > h)) {
        std::vector<float> rot((size_t)w * h * 3); // new dims: H'=w, W'=h
        int nh = w, nw = h;
        for (int y = 0; y < nh; y++)
            for (int x = 0; x < nw; x++) {
                // rot90(k=3): out[y,x] = in[H-1-x, y]  (90° clockwise)
                int sy = h - 1 - x, sx = y;
                for (int c = 0; c < 3; c++) rot[((size_t)y * nw + x) * 3 + c] = img[((size_t)sy * w + sx) * 3 + c];
            }
        img = std::move(rot);
        h = nh;
        w = nw;
    }

    // thumbnail: shrink (never enlarge) preserving aspect so h≤TH and w≤TW.
    double scale = std::min({ 1.0, (double)TH / h, (double)TW / w });
    int rh = std::max(1, (int)std::lround(h * scale));
    int rw = std::max(1, (int)std::lround(w * scale));
    std::vector<float> resized((size_t)rh * rw * 3);
    for (int y = 0; y < rh; y++) {
        float fy = (h > 1 && rh > 1) ? (float)y * (h - 1) / (rh - 1) : 0.0f;
        for (int x = 0; x < rw; x++) {
            float fx = (w > 1 && rw > 1) ? (float)x * (w - 1) / (rw - 1) : 0.0f;
            for (int c = 0; c < 3; c++)
                resized[((size_t)y * rw + x) * 3 + c] = bilinear_sample(img, h, w, 3, fy, fx, c);
        }
    }

    // pad to (TH,TW): delta//2 top/left (HF Donut center-ish pad), rescale+normalize.
    int pad_top = (TH - rh) / 2;
    int pad_left = (TW - rw) / 2;
    std::vector<float> out((size_t)3 * TH * TW, (0.0f - hp.image_mean) / hp.image_std); // zero-pixel → normalized
    for (int y = 0; y < rh; y++)
        for (int x = 0; x < rw; x++) {
            int oy = y + pad_top, ox = x + pad_left;
            if (oy < 0 || oy >= TH || ox < 0 || ox >= TW) continue;
            for (int c = 0; c < 3; c++) {
                float v = resized[((size_t)y * rw + x) * 3 + c] / 255.0f;
                out[(size_t)c * TH * TW + (size_t)oy * TW + ox] = (v - hp.image_mean) / hp.image_std;
            }
        }
    return out;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
const char * flova_ocr_recognize_raw(flova_ocr_context * ctx, const uint8_t * pixels, int width, int height,
                                     int channels, int * out_len) {
    if (!ctx || !pixels) {
        if (out_len) *out_len = 0;
        return nullptr;
    }
    auto input = preprocess_donut(ctx, pixels, width, height, channels);
    auto enc = run_swin_encoder(ctx, input.data(), ctx->hp.image_h, ctx->hp.image_w, nullptr, nullptr);
    int enc_len = (int)(enc.size() / ctx->hp.enc_hidden);
    ctx->output_text = decode_greedy(ctx, enc.data(), enc_len);
    if (out_len) *out_len = (int)ctx->output_text.size();
    return ctx->output_text.c_str();
}

const char * flova_ocr_recognize_file(flova_ocr_context * ctx, const char * image_path, int * out_len) {
    int w = 0, h = 0, c = 0;
    stbi_uc * data = stbi_load(image_path, &w, &h, &c, 3);
    if (!data) {
        if (out_len) *out_len = 0;
        return nullptr;
    }
    const char * r = flova_ocr_recognize_raw(ctx, data, w, h, 3, out_len);
    stbi_image_free(data);
    return r;
}

// ---------------------------------------------------------------------------
// Per-stage parity harness vs tools/dump_flova_reference.py
// ---------------------------------------------------------------------------
int flova_ocr_run_diff(flova_ocr_context * ctx, const char * ref_gguf_path) {
    crispembed_diff::Ref ref;
    if (!ref.load(ref_gguf_path)) return 1;

    auto pvshape = ref.shape("pixel_values"); // ne = [W, H, 3]
    if (pvshape.size() < 2) {
        fprintf(stderr, "flova_diff: no pixel_values in ref\n");
        return 1;
    }
    int W = (int)pvshape[0], H = (int)pvshape[1];
    auto pv = ref.get_f32("pixel_values");
    fprintf(stderr, "flova_diff: pixel_values W=%d H=%d (%zu floats)\n", W, H, pv.second);

    int fails = 0;
    // ---- encoder (feed reference pixel_values) ----
    auto enc = run_swin_encoder(ctx, pv.first, H, W, &ref, &fails);
    int enc_len = (int)(enc.size() / ctx->hp.enc_hidden);

    // ---- decoder (teacher-forced on ref ids) ----
    auto idf = ref.get_f32("ids");
    if (!idf.first || idf.second == 0) {
        fprintf(stderr, "flova_diff: no ids — skipping decoder\n");
        return fails ? 1 : 0;
    }
    std::vector<int> ids(idf.second);
    for (size_t i = 0; i < idf.second; i++) ids[i] = (int)llround(idf.first[i]);
    fprintf(stderr, "flova_diff: teacher-forcing %zu tokens\n", ids.size());

    std::vector<DecW> dw;
    build_decw(ctx, dw);
    std::vector<CrossKV> ckv;
    precompute_cross(ctx, enc.data(), enc_len, dw, ckv);
    auto tok_w = to_f32(ctx->embed_tokens_w);
    auto pos_w = to_f32(ctx->embed_pos_w);
    auto eln_w = to_f32(ctx->embed_ln_w), eln_b = to_f32(ctx->embed_ln_b);
    auto fn_w = to_f32(ctx->dec_final_norm_w), fn_b = to_f32(ctx->dec_final_norm_b);
    auto lm_w = to_f32(ctx->lm_head_w);
    std::vector<float> logits_all;
    forward_decoder(ctx, ids, enc.data(), enc_len, ckv, dw, tok_w, pos_w, eln_w, eln_b, fn_w, fn_b, lm_w, logits_all,
                    &ref, &fails);

    // Per-position argmax agreement (the decode-relevant metric).
    if (ref.has("logits")) {
        auto rf = ref.get_f32("logits");
        int V = ctx->hp.vocab_size, L = (int)ids.size(), ok = 0;
        for (int i = 0; i < L; i++)
            if (argmax(&logits_all[(size_t)i * V], V) == argmax(&rf.first[(size_t)i * V], V)) ok++;
        fprintf(stderr, "[flova-diff] argmax logits  %d/%d positions agree\n", ok, L);
    }

    // Greedy decode → compare LilyPond to the model card.
    std::string ly = decode_greedy(ctx, enc.data(), enc_len);
    fprintf(stderr, "[flova-diff] greedy decode: \"%s\"\n", ly.c_str());

    fprintf(stderr, "flova_diff: %s (%d stage failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
