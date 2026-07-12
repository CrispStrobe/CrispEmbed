// smt_ocr.cpp — Sheet Music Transformer (SMT/SMT++) OMR via ggml graph compute.
//
// ConvNext encoder + cross-attention Transformer decoder → bekern tokens.
// Blueprint: SMT-plusplus/smt_model/modeling_smt.py (read line-by-line).
// KEY DEVIATIONS from a vanilla VED (verified in source, not the abstract):
//   • attention QK^T is UNSCALED (MHA.scale_factor defined but never applied);
//   • cross-attn KEY = enc features + 2D PE, VALUE = raw enc features;
//   • decoder FFN is 1× (256→256), ReLU; token emb NOT scaled by sqrt(d);
//   • encoder last_hidden_state is pre-pooler-LN (encoder.layernorm is dead);
//   • LM head = 1×1 Conv1d (stored [V,256,1], squeezed to Linear by the converter).

#include "smt_ocr.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gpu_backend_pref.h"
#include "core/gguf_loader.h"
#include "core/cpu_ops.h"
#include "crispembed_diff.h"

extern "C" {
typedef unsigned char stbi_uc;
stbi_uc * stbi_load(char const * filename, int * x, int * y, int * channels_in_file, int desired_channels);
void stbi_image_free(void * retval_from_stbi_load);
}

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// dequant helper (device-safe)
// ---------------------------------------------------------------------------
static std::vector<float> to_f32(const ggml_tensor * t) {
    if (!t) return {};
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    std::vector<uint8_t> raw;
    const void * src;
    if (t->buffer) {
        raw.resize(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        src = raw.data();
    } else {
        src = t->data;
    }
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), src, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *)src;
        for (int i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(s[i]);
    } else {
        const auto * tr = ggml_get_type_traits(t->type);
        if (tr && tr->to_float)
            tr->to_float(src, out.data(), n);
        else
            memset(out.data(), 0, n * sizeof(float));
    }
    return out;
}

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------
struct smt_enc_layer {
    ggml_tensor *dw_w, *dw_b, *ln_w, *ln_b, *pw1_w, *pw1_b, *pw2_w, *pw2_b, *gamma;
};
struct smt_enc_stage {
    ggml_tensor *ds_ln_w, *ds_ln_b, *ds_conv_w, *ds_conv_b; // downsampling (null for stage 0)
    std::vector<smt_enc_layer> layers;
};
struct smt_dec_layer {
    ggml_tensor *sa_q_w, *sa_q_b, *sa_k_w, *sa_k_b, *sa_v_w, *sa_v_b, *sa_o_w, *sa_o_b;
    ggml_tensor *n1_w, *n1_b;
    ggml_tensor *ca_q_w, *ca_q_b, *ca_k_w, *ca_k_b, *ca_v_w, *ca_v_b, *ca_o_w, *ca_o_b;
    ggml_tensor *n2_w, *n2_b;
    ggml_tensor *ff0_w, *ff0_b, *ff3_w, *ff3_b;
    ggml_tensor *n3_w, *n3_b;
};

struct smt_ocr_context {
    smt_ocr_hparams hp;
    // encoder
    ggml_tensor *stem_w, *stem_b, *stem_ln_w, *stem_ln_b;
    std::vector<int> stage_sizes, stage_depths;
    std::vector<smt_enc_stage> stages;
    // decoder
    ggml_tensor *tok_embed, *pos1d, *lm_w, *lm_b;
    std::vector<smt_dec_layer> dec;

    std::vector<std::string> vocab;
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr, backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    int n_threads = 4;
    std::string result;

    // cached encoder memory (host) from the last run_encoder
    std::vector<float> mem_value, mem_key; // [n_enc * C], token-major (c fastest)
    int n_enc = 0, enc_h = 0, enc_w = 0;
    // decode KV cache: cross K/V are constant per image (precomputed once);
    // self K/V grow one token per step.
    std::vector<std::vector<float>> cross_k_host, cross_v_host; // [nl][n_enc*C]
    std::vector<std::vector<float>> self_k_host, self_v_host;   // [nl][t*C]
};

// ---------------------------------------------------------------------------
static ggml_tensor * F(const std::unordered_map<std::string, ggml_tensor *> & m, const std::string & n) {
    auto it = m.find(n);
    return it != m.end() ? it->second : nullptr;
}

static void map_tensors(smt_ocr_context * ctx) {
    const auto & m = ctx->wl.tensors;
    const auto & hp = ctx->hp;
    char b[256];

    ctx->stem_w = F(m, "encoder.embeddings.patch_embeddings.weight");
    ctx->stem_b = F(m, "encoder.embeddings.patch_embeddings.bias");
    ctx->stem_ln_w = F(m, "encoder.embeddings.layernorm.weight");
    ctx->stem_ln_b = F(m, "encoder.embeddings.layernorm.bias");

    ctx->stages.resize(hp.enc_num_stages);
    for (int s = 0; s < hp.enc_num_stages; s++) {
        auto & S = ctx->stages[s];
        snprintf(b, sizeof(b), "encoder.encoder.stages.%d.downsampling_layer.0.weight", s);
        S.ds_ln_w = F(m, b);
        snprintf(b, sizeof(b), "encoder.encoder.stages.%d.downsampling_layer.0.bias", s);
        S.ds_ln_b = F(m, b);
        snprintf(b, sizeof(b), "encoder.encoder.stages.%d.downsampling_layer.1.weight", s);
        S.ds_conv_w = F(m, b);
        snprintf(b, sizeof(b), "encoder.encoder.stages.%d.downsampling_layer.1.bias", s);
        S.ds_conv_b = F(m, b);
        S.layers.resize(ctx->stage_depths[s]);
        for (int i = 0; i < ctx->stage_depths[s]; i++) {
            auto & L = S.layers[i];
            auto E = [&](const char * suf) {
                snprintf(b, sizeof(b), "encoder.encoder.stages.%d.layers.%d.%s", s, i, suf);
                return F(m, b);
            };
            L.dw_w = E("dwconv.weight");
            L.dw_b = E("dwconv.bias");
            L.ln_w = E("layernorm.weight");
            L.ln_b = E("layernorm.bias");
            L.pw1_w = E("pwconv1.weight");
            L.pw1_b = E("pwconv1.bias");
            L.pw2_w = E("pwconv2.weight");
            L.pw2_b = E("pwconv2.bias");
            L.gamma = E("layer_scale_parameter");
        }
    }

    ctx->tok_embed = F(m, "decoder.embedding.weight");
    ctx->pos1d = F(m, "smt.positional_1d");
    ctx->lm_w = F(m, "decoder.out_layer.weight");
    ctx->lm_b = F(m, "decoder.out_layer.bias");

    ctx->dec.resize(hp.dec_layers);
    for (int i = 0; i < hp.dec_layers; i++) {
        auto & L = ctx->dec[i];
        auto D = [&](const char * suf) {
            snprintf(b, sizeof(b), "decoder.decoder.layers.%d.%s", i, suf);
            return F(m, b);
        };
        L.sa_q_w = D("input_attention.lq.weight");
        L.sa_q_b = D("input_attention.lq.bias");
        L.sa_k_w = D("input_attention.lk.weight");
        L.sa_k_b = D("input_attention.lk.bias");
        L.sa_v_w = D("input_attention.lv.weight");
        L.sa_v_b = D("input_attention.lv.bias");
        L.sa_o_w = D("input_attention.out_proj.weight");
        L.sa_o_b = D("input_attention.out_proj.bias");
        L.n1_w = D("norm1.weight");
        L.n1_b = D("norm1.bias");
        L.ca_q_w = D("cross_attention.lq.weight");
        L.ca_q_b = D("cross_attention.lq.bias");
        L.ca_k_w = D("cross_attention.lk.weight");
        L.ca_k_b = D("cross_attention.lk.bias");
        L.ca_v_w = D("cross_attention.lv.weight");
        L.ca_v_b = D("cross_attention.lv.bias");
        L.ca_o_w = D("cross_attention.out_proj.weight");
        L.ca_o_b = D("cross_attention.out_proj.bias");
        L.n2_w = D("norm2.weight");
        L.n2_b = D("norm2.bias");
        L.ff0_w = D("ffNet.0.weight");
        L.ff0_b = D("ffNet.0.bias");
        L.ff3_w = D("ffNet.3.weight");
        L.ff3_b = D("ffNet.3.bias");
        L.n3_w = D("norm3.weight");
        L.n3_b = D("norm3.bias");
    }
}

// ---------------------------------------------------------------------------
// graph helpers
// ---------------------------------------------------------------------------
static ggml_tensor * f32(ggml_context * g, ggml_tensor * t) {
    return (!t || t->type == GGML_TYPE_F32) ? t : ggml_cast(g, t, GGML_TYPE_F32);
}
// LayerNorm over ne[0] (channels-last). eps=1e-6 (ConvNext + decoder).
static ggml_tensor * ln0(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_norm(g, x, 1e-6f);
    x = ggml_mul(g, x, f32(g, w));
    if (b) x = ggml_add(g, x, f32(g, b));
    return x;
}
// Linear: W[in,out] (ggml ne), x[in,...] → [out,...]
static ggml_tensor * lin(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_mul_mat(g, w, x);
    if (b) x = ggml_add(g, x, f32(g, b));
    return x;
}
// per-channel bias for a conv map [W,H,OC,N]: bias[OC] → [1,1,OC,1]
static ggml_tensor * add_cbias(ggml_context * g, ggml_tensor * x, ggml_tensor * bias) {
    return ggml_add(g, x, ggml_reshape_4d(g, f32(g, bias), 1, 1, bias->ne[0], 1));
}
// channels-first LayerNorm on a map [W,H,C,N] (ConvNext stem/downsample LN)
static ggml_tensor * ln_cf(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * xc = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3)); // [C,W,H,N]
    xc = ln0(g, xc, w, b);
    return ggml_cont(g, ggml_permute(g, xc, 2, 0, 1, 3)); // back to [W,H,C,N]
}
// conv2d with in-graph F16 kernel cast; kernel [KW,KH,IC,OC], x=[W,H,IC,N].
// crispembed-quantize flattens 4D conv kernels to 2D [IC*KH*KW, OC] in the GGUF
// header (data bytes unchanged); reshape back to 4D so ggml_conv_2d sees a valid
// kernel for both the F32 converter output and quantized GGUFs.
static ggml_tensor * conv(ggml_context * g, ggml_tensor * a, ggml_tensor * x, int s, int p, bool dw, int kw, int kh,
                          int ic, int oc) {
    if (ggml_n_dims(a) != 4) a = ggml_reshape_4d(g, a, kw, kh, ic, oc);
    ggml_tensor * k = ggml_cast(g, a, GGML_TYPE_F16);
    return dw ? ggml_conv_2d_dw(g, k, x, s, s, p, p, 1, 1) : ggml_conv_2d(g, k, x, s, s, p, p, 1, 1);
}
// flatten a feature map [W,H,C,N=1] → token tensor [C, W*H] (c fastest per token)
static ggml_tensor * to_tokens(ggml_context * g, ggml_tensor * map) {
    int W = map->ne[0], H = map->ne[1], C = map->ne[2];
    ggml_tensor * t = ggml_cont(g, ggml_permute(g, map, 1, 2, 0, 3)); // [C,W,H,N]
    return ggml_reshape_2d(g, t, C, W * H);                           // [C, W*H]
}
// unscaled multi-head attention core (returns concat-heads [C,Lq], no out_proj)
static ggml_tensor * mha_core(ggml_context * g, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int nh, int Lq,
                              int Lk, ggml_tensor * mask) {
    int C = q->ne[0];
    int hd = C / nh;
    ggml_tensor * Q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, q, hd, nh, Lq), 0, 2, 1, 3)); // [hd,Lq,nh]
    ggml_tensor * K = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, k, hd, nh, Lk), 0, 2, 1, 3)); // [hd,Lk,nh]
    ggml_tensor * V = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, v, hd, nh, Lk), 0, 2, 1, 3)); // [hd,Lk,nh]
    ggml_tensor * scores = ggml_mul_mat(g, K, Q);                                                   // [Lk,Lq,nh]
    scores = ggml_soft_max_ext(g, scores, mask, 1.0f, 0.0f);         // UNSCALED (scale=1.0)
    ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3)); // [Lk,hd,nh]
    ggml_tensor * a = ggml_mul_mat(g, Vt, scores);                   // [hd,Lq,nh]
    a = ggml_cont(g, ggml_permute(g, a, 0, 2, 1, 3));                // [hd,nh,Lq]
    return ggml_reshape_2d(g, a, C, Lq);
}

// ---------------------------------------------------------------------------
// Encoder graph: input [W,H,1,1] → per-stage token snapshots + enc_output tokens
// ---------------------------------------------------------------------------
static ggml_cgraph * build_encoder(smt_ocr_context * ctx, ggml_context * g, int W, int H) {
    ggml_cgraph * gf = ggml_new_graph_custom(g, 8192, false);
    if (getenv("SMT_OCR_DEBUG")) {
        fprintf(stderr, "[dbg] stem_w=%p stem_b=%p stem_ln_w=%p tok_embed=%p pos1d=%p lm_w=%p lm_b=%p\n",
                (void *)ctx->stem_w, (void *)ctx->stem_b, (void *)ctx->stem_ln_w, (void *)ctx->tok_embed,
                (void *)ctx->pos1d, (void *)ctx->lm_w, (void *)ctx->lm_b);
        for (int s = 0; s < ctx->hp.enc_num_stages; s++) {
            auto & S = ctx->stages[s];
            fprintf(stderr, "[dbg] stage%d ds_conv=%p nlayers=%zu L0.dw=%p L0.pw1=%p L0.gamma=%p\n", s,
                    (void *)S.ds_conv_w, S.layers.size(), S.layers.empty() ? nullptr : (void *)S.layers[0].dw_w,
                    S.layers.empty() ? nullptr : (void *)S.layers[0].pw1_w,
                    S.layers.empty() ? nullptr : (void *)S.layers[0].gamma);
        }
    }
    ggml_tensor * x = ggml_new_tensor_4d(g, GGML_TYPE_F32, W, H, ctx->hp.enc_num_channels, 1);
    ggml_set_name(x, "input");
    ggml_set_input(x);

    // stem: Conv2d(in_ch->C0, k, s=k) + channels-first LN
    const int K = ctx->hp.enc_stem_kernel;
    int cur_ch = ctx->stage_sizes[0];
    ggml_tensor * cur = conv(g, ctx->stem_w, x, K, 0, false, K, K, ctx->hp.enc_num_channels, cur_ch);
    cur = add_cbias(g, cur, ctx->stem_b);
    cur = ln_cf(g, cur, ctx->stem_ln_w, ctx->stem_ln_b);

    for (int s = 0; s < ctx->hp.enc_num_stages; s++) {
        auto & S = ctx->stages[s];
        int Cs = ctx->stage_sizes[s];
        if (S.ds_conv_w) { // downsampling: channels-first LN → Conv2d(Cprev->Cs, k2, s2)
            cur = ln_cf(g, cur, S.ds_ln_w, S.ds_ln_b);
            cur = conv(g, S.ds_conv_w, cur, 2, 0, false, 2, 2, cur_ch, Cs);
            cur = add_cbias(g, cur, S.ds_conv_b);
        }
        cur_ch = Cs;
        for (auto & L : S.layers) {
            ggml_tensor * inp = cur;
            ggml_tensor * y = conv(g, L.dw_w, cur, 1, 3, true, 7, 7, 1, Cs); // depthwise 7×7
            y = add_cbias(g, y, L.dw_b);
            ggml_tensor * yc = ggml_cont(g, ggml_permute(g, y, 1, 2, 0, 3)); // [C,W,H,N]
            yc = ln0(g, yc, L.ln_w, L.ln_b);
            yc = lin(g, yc, L.pw1_w, L.pw1_b); // [4C,...]
            yc = ggml_gelu_erf(g, yc);
            yc = lin(g, yc, L.pw2_w, L.pw2_b);                 // [C,...]
            yc = ggml_mul(g, yc, f32(g, L.gamma));             // layer scale (broadcast over ne0=C)
            y = ggml_cont(g, ggml_permute(g, yc, 2, 0, 1, 3)); // back to [W,H,C,N]
            cur = ggml_add(g, inp, y);
        }
        ggml_tensor * tok = to_tokens(g, cur);
        char nm[24];
        snprintf(nm, sizeof(nm), "enc_stage%d", s);
        ggml_set_name(tok, nm);
        ggml_set_output(tok);
        ggml_build_forward_expand(gf, tok); // off-trunk branch — expand explicitly
    }
    ggml_tensor * out = to_tokens(g, cur); // [C, n_enc]
    ggml_set_name(out, "enc_output");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);
    return gf;
}

// generate the 2D sinusoidal PE for (h_max=eh, w_max=ew) as token-major [C*n] (c fastest)
static std::vector<float> gen_pe2d(int C, int eh, int ew) {
    int n = eh * ew;
    std::vector<float> pe((size_t)n * C, 0.0f);
    int half = C / 2; // 128
    for (int t = 0; t < n; t++) {
        int h = t / ew, w = t % ew;
        for (int c = 0; c < C; c++) {
            float val;
            if (c < half) {
                int k = c / 2;
                float d = std::exp(-(float)(2 * k) / C * std::log(10000.0f));
                val = (c % 2 == 0) ? std::sin(h * d) : std::cos(h * d);
            } else {
                int cc = c - half, k = cc / 2;
                float d = std::exp(-(float)(2 * k) / C * std::log(10000.0f));
                val = (cc % 2 == 0) ? std::sin(w * d) : std::cos(w * d);
            }
            pe[(size_t)t * C + c] = val;
        }
    }
    return pe;
}

// run encoder on a preprocessed image [C_in,H,W] (row-major, W fastest) → fills mem_*
static bool run_encoder(smt_ocr_context * ctx, const float * input, int W, int H, ggml_context ** keep_g,
                        ggml_cgraph ** keep_gf) {
    size_t meta = 32 * 1024 * 1024;
    std::vector<uint8_t> * buf = new std::vector<uint8_t>(meta);
    ggml_init_params ip = { meta, buf->data(), true };
    ggml_context * g = ggml_init(ip);
    bool dbg = getenv("SMT_OCR_DEBUG");
    ggml_cgraph * gf = build_encoder(ctx, g, W, H);
    if (dbg) fprintf(stderr, "[dbg] encoder graph built: %d nodes\n", ggml_graph_n_nodes(gf));

    ggml_backend_sched_reset(ctx->sched);
    if (dbg) fprintf(stderr, "[dbg] sched reset ok, allocating...\n");
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "smt_ocr: encoder alloc failed\n");
        ggml_free(g);
        delete buf;
        return false;
    }
    ggml_tensor * inp = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(inp, input, 0, (size_t)W * H * ctx->hp.enc_num_channels * sizeof(float));
    if (dbg) fprintf(stderr, "[dbg] input set, computing...\n");
    ggml_backend_sched_graph_compute(ctx->sched, gf);
    if (dbg) fprintf(stderr, "[dbg] encoder compute done\n");

    ggml_tensor * out = ggml_graph_get_tensor(gf, "enc_output");
    int C = out->ne[0];
    ctx->n_enc = out->ne[1];
    ctx->enc_h = H / ctx->hp.enc_reduction;
    ctx->enc_w = W / ctx->hp.enc_reduction;
    ctx->mem_value.resize((size_t)ctx->n_enc * C);
    ggml_backend_tensor_get(out, ctx->mem_value.data(), 0, ctx->mem_value.size() * sizeof(float));

    // mem_key = mem_value + 2D PE (added to keys only)
    std::vector<float> pe = gen_pe2d(C, ctx->enc_h, ctx->enc_w);
    ctx->mem_key.resize(ctx->mem_value.size());
    for (size_t i = 0; i < ctx->mem_value.size(); i++) ctx->mem_key[i] = ctx->mem_value[i] + pe[i];

    if (keep_g && keep_gf) {
        *keep_g = g;
        *keep_gf = gf;
    } // caller frees
    else {
        ggml_free(g);
        delete buf;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Decoder graph (teacher-forced over L tokens; causal self-attn, unscaled)
// ---------------------------------------------------------------------------
static ggml_cgraph * build_decoder(smt_ocr_context * ctx, ggml_context * g, int L) {
    const auto & hp = ctx->hp;
    int C = hp.d_model, nh = hp.num_heads, nE = ctx->n_enc;
    ggml_cgraph * gf = ggml_new_graph_custom(g, hp.dec_layers * 64 + 256, false);

    ggml_tensor * tokens = ggml_new_tensor_1d(g, GGML_TYPE_I32, L);
    ggml_set_name(tokens, "dec_tokens");
    ggml_set_input(tokens);
    ggml_tensor * mk = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nE);
    ggml_set_name(mk, "mem_key");
    ggml_set_input(mk);
    ggml_tensor * mv = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nE);
    ggml_set_name(mv, "mem_value");
    ggml_set_input(mv);
    ggml_tensor * mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, L, L);
    ggml_set_name(mask, "dec_mask");
    ggml_set_input(mask);

    // token embedding + 1D PE
    ggml_tensor * cur = ggml_get_rows(g, ctx->tok_embed, tokens); // [C,L]
    ggml_tensor * pe = ggml_view_2d(g, ctx->pos1d, C, L, ctx->pos1d->nb[1], 0);
    cur = ggml_add(g, cur, pe);
    ggml_set_name(cur, "dec_tok_emb");
    ggml_set_output(cur);

    for (int i = 0; i < hp.dec_layers; i++) {
        auto & Lr = ctx->dec[i];
        // self-attention (causal, unscaled)
        ggml_tensor * q = lin(g, cur, Lr.sa_q_w, Lr.sa_q_b);
        ggml_tensor * k = lin(g, cur, Lr.sa_k_w, Lr.sa_k_b);
        ggml_tensor * v = lin(g, cur, Lr.sa_v_w, Lr.sa_v_b);
        ggml_tensor * sa = mha_core(g, q, k, v, nh, L, L, mask);
        sa = lin(g, sa, Lr.sa_o_w, Lr.sa_o_b);
        cur = ggml_add(g, cur, sa);
        cur = ln0(g, cur, Lr.n1_w, Lr.n1_b);
        // cross-attention (unscaled, no mask; KEY=mem_key, VALUE=mem_value)
        ggml_tensor * aq = cur;
        ggml_tensor * cq = lin(g, aq, Lr.ca_q_w, Lr.ca_q_b);
        ggml_tensor * ck = lin(g, mk, Lr.ca_k_w, Lr.ca_k_b);
        ggml_tensor * cv = lin(g, mv, Lr.ca_v_w, Lr.ca_v_b);
        ggml_tensor * ca = mha_core(g, cq, ck, cv, nh, L, nE, nullptr);
        ca = lin(g, ca, Lr.ca_o_w, Lr.ca_o_b);
        cur = ggml_add(g, aq, ca);
        cur = ln0(g, cur, Lr.n2_w, Lr.n2_b);
        // FFN (1×, ReLU)
        ggml_tensor * ff = lin(g, cur, Lr.ff0_w, Lr.ff0_b);
        ff = ggml_relu(g, ff);
        ff = lin(g, ff, Lr.ff3_w, Lr.ff3_b);
        cur = ggml_add(g, cur, ff);
        cur = ln0(g, cur, Lr.n3_w, Lr.n3_b);
        char nm[24];
        snprintf(nm, sizeof(nm), "dec_layer%d", i);
        ggml_set_name(cur, nm);
        ggml_set_output(cur);
    }
    // logits = out_layer(relu(output))
    ggml_tensor * logits = lin(g, ggml_relu(g, cur), ctx->lm_w, ctx->lm_b); // [V,L]
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

// build a causal mask [L,L]: mask[j + i*L] = 0 if key j <= query i else -inf
static std::vector<float> causal_mask(int L) {
    std::vector<float> m((size_t)L * L, 0.0f);
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++) m[(size_t)i * L + j] = (j <= i) ? 0.0f : -INFINITY;
    return m;
}

// run decoder teacher-forced over token ids; returns logits [V,L] if want_logits
static bool run_decoder(smt_ocr_context * ctx, const std::vector<int32_t> & ids, ggml_context ** keep_g,
                        ggml_cgraph ** keep_gf) {
    int L = (int)ids.size();
    size_t meta = 64 * 1024 * 1024;
    std::vector<uint8_t> * buf = new std::vector<uint8_t>(meta);
    ggml_init_params ip = { meta, buf->data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = build_decoder(ctx, g, L);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "smt_ocr: decoder alloc failed\n");
        ggml_free(g);
        delete buf;
        return false;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_tokens"), ids.data(), 0, L * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mem_key"), ctx->mem_key.data(), 0,
                            ctx->mem_key.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mem_value"), ctx->mem_value.data(), 0,
                            ctx->mem_value.size() * sizeof(float));
    std::vector<float> mask = causal_mask(L);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_mask"), mask.data(), 0, mask.size() * sizeof(float));
    ggml_backend_sched_graph_compute(ctx->sched, gf);

    if (keep_g && keep_gf) {
        *keep_g = g;
        *keep_gf = gf;
    } else {
        ggml_free(g);
        delete buf;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Incremental (KV-cached) decode: cross K/V precomputed once, self K/V grown
// one token per step. Mathematically identical to the full-recompute path
// (causal self-attn + constant cross memory), but O(L) work per step.
// ---------------------------------------------------------------------------
static void precompute_cross_kv(smt_ocr_context * ctx) {
    const int C = ctx->hp.d_model, nE = ctx->n_enc, nl = ctx->hp.dec_layers;
    ctx->cross_k_host.assign(nl, {});
    ctx->cross_v_host.assign(nl, {});
    size_t meta = 32 * 1024 * 1024;
    std::vector<uint8_t> buf(meta);
    ggml_init_params ip = { meta, buf.data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(g, nl * 8 + 16, false);
    ggml_tensor * mk = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nE);
    ggml_set_name(mk, "mk");
    ggml_set_input(mk);
    ggml_tensor * mv = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nE);
    ggml_set_name(mv, "mv");
    ggml_set_input(mv);
    for (int i = 0; i < nl; i++) {
        auto & L = ctx->dec[i];
        char nm[16];
        ggml_tensor * ck = lin(g, mk, L.ca_k_w, L.ca_k_b);
        snprintf(nm, sizeof(nm), "ck_%d", i);
        ggml_set_name(ck, nm);
        ggml_set_output(ck);
        ggml_build_forward_expand(gf, ck);
        ggml_tensor * cv = lin(g, mv, L.ca_v_w, L.ca_v_b);
        snprintf(nm, sizeof(nm), "cv_%d", i);
        ggml_set_name(cv, nm);
        ggml_set_output(cv);
        ggml_build_forward_expand(gf, cv);
    }
    ggml_backend_sched_reset(ctx->sched);
    ggml_backend_sched_alloc_graph(ctx->sched, gf);
    ggml_backend_tensor_set(mk, ctx->mem_key.data(), 0, ctx->mem_key.size() * sizeof(float));
    ggml_backend_tensor_set(mv, ctx->mem_value.data(), 0, ctx->mem_value.size() * sizeof(float));
    ggml_backend_sched_graph_compute(ctx->sched, gf);
    for (int i = 0; i < nl; i++) {
        char nm[16];
        ctx->cross_k_host[i].resize((size_t)nE * C);
        ctx->cross_v_host[i].resize((size_t)nE * C);
        snprintf(nm, sizeof(nm), "ck_%d", i);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, nm), ctx->cross_k_host[i].data(), 0,
                                (size_t)nE * C * sizeof(float));
        snprintf(nm, sizeof(nm), "cv_%d", i);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, nm), ctx->cross_v_host[i].data(), 0,
                                (size_t)nE * C * sizeof(float));
    }
    ggml_free(g);
}

// One decode step for the token at position n_cached (0-based). Self K/V for the
// n_cached prior tokens come from the host cache; the new token's K/V are emitted
// as "sk_i"/"sv_i" for the caller to append.
static ggml_cgraph * build_decode_step(smt_ocr_context * ctx, ggml_context * g, int n_cached) {
    const auto & hp = ctx->hp;
    const int C = hp.d_model, nh = hp.num_heads, nE = ctx->n_enc;
    ggml_cgraph * gf = ggml_new_graph_custom(g, hp.dec_layers * 128 + 512, false);

    ggml_tensor * token = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(token, "step_token");
    ggml_set_input(token);
    ggml_tensor * cur = ggml_get_rows(g, ctx->tok_embed, token); // [C,1]
    ggml_tensor * pe = ggml_view_2d(g, ctx->pos1d, C, 1, ctx->pos1d->nb[1], (size_t)n_cached * ctx->pos1d->nb[1]);
    cur = ggml_add(g, cur, pe);

    for (int i = 0; i < hp.dec_layers; i++) {
        auto & Lr = ctx->dec[i];
        char nm[16];
        ggml_tensor * q = lin(g, cur, Lr.sa_q_w, Lr.sa_q_b);
        ggml_tensor * knew = lin(g, cur, Lr.sa_k_w, Lr.sa_k_b); // [C,1]
        ggml_tensor * vnew = lin(g, cur, Lr.sa_v_w, Lr.sa_v_b);
        snprintf(nm, sizeof(nm), "sk_%d", i);
        ggml_set_name(knew, nm);
        ggml_set_output(knew);
        ggml_build_forward_expand(gf, knew);
        snprintf(nm, sizeof(nm), "sv_%d", i);
        ggml_set_name(vnew, nm);
        ggml_set_output(vnew);
        ggml_build_forward_expand(gf, vnew);
        ggml_tensor *kfull = knew, *vfull = vnew;
        if (n_cached > 0) {
            ggml_tensor * skin = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, n_cached);
            snprintf(nm, sizeof(nm), "skin_%d", i);
            ggml_set_name(skin, nm);
            ggml_set_input(skin);
            ggml_tensor * svin = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, n_cached);
            snprintf(nm, sizeof(nm), "svin_%d", i);
            ggml_set_name(svin, nm);
            ggml_set_input(svin);
            kfull = ggml_concat(g, skin, knew, 1); // [C, n_cached+1]
            vfull = ggml_concat(g, svin, vnew, 1);
        }
        ggml_tensor * sa = mha_core(g, q, kfull, vfull, nh, 1, n_cached + 1, nullptr);
        sa = lin(g, sa, Lr.sa_o_w, Lr.sa_o_b);
        cur = ggml_add(g, cur, sa);
        cur = ln0(g, cur, Lr.n1_w, Lr.n1_b);
        // cross-attention against precomputed (already-projected) K/V
        ggml_tensor * ck = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nE);
        snprintf(nm, sizeof(nm), "ck_%d", i);
        ggml_set_name(ck, nm);
        ggml_set_input(ck);
        ggml_tensor * cv = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nE);
        snprintf(nm, sizeof(nm), "cv_%d", i);
        ggml_set_name(cv, nm);
        ggml_set_input(cv);
        ggml_tensor * aq = cur;
        ggml_tensor * cq = lin(g, aq, Lr.ca_q_w, Lr.ca_q_b);
        ggml_tensor * ca = mha_core(g, cq, ck, cv, nh, 1, nE, nullptr);
        ca = lin(g, ca, Lr.ca_o_w, Lr.ca_o_b);
        cur = ggml_add(g, aq, ca);
        cur = ln0(g, cur, Lr.n2_w, Lr.n2_b);
        ggml_tensor * ff = lin(g, cur, Lr.ff0_w, Lr.ff0_b);
        ff = ggml_relu(g, ff);
        ff = lin(g, ff, Lr.ff3_w, Lr.ff3_b);
        cur = ggml_add(g, cur, ff);
        cur = ln0(g, cur, Lr.n3_w, Lr.n3_b);
    }
    ggml_tensor * logits = lin(g, ggml_relu(g, cur), ctx->lm_w, ctx->lm_b); // [V,1]
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

// KV-cached greedy decode. Assumes run_encoder already populated mem_*.
static void decode_greedy_kv(smt_ocr_context * ctx) {
    const auto & hp = ctx->hp;
    const int C = hp.d_model, nl = hp.dec_layers, V = hp.vocab_size;
    precompute_cross_kv(ctx);
    ctx->self_k_host.assign(nl, {});
    ctx->self_v_host.assign(nl, {});
    ctx->result.clear();
    int id = hp.bos_token;
    for (int pos = 0; pos < hp.maxlen; pos++) {
        size_t meta = 24 * 1024 * 1024;
        std::vector<uint8_t> buf(meta);
        ggml_init_params ip = { meta, buf.data(), true };
        ggml_context * g = ggml_init(ip);
        ggml_cgraph * gf = build_decode_step(ctx, g, pos);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            ggml_free(g);
            break;
        }
        int32_t tok = id;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "step_token"), &tok, 0, sizeof(int32_t));
        for (int l = 0; l < nl; l++) {
            char nm[16];
            snprintf(nm, sizeof(nm), "ck_%d", l);
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, nm), ctx->cross_k_host[l].data(), 0,
                                    ctx->cross_k_host[l].size() * sizeof(float));
            snprintf(nm, sizeof(nm), "cv_%d", l);
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, nm), ctx->cross_v_host[l].data(), 0,
                                    ctx->cross_v_host[l].size() * sizeof(float));
            if (pos > 0) {
                snprintf(nm, sizeof(nm), "skin_%d", l);
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, nm), ctx->self_k_host[l].data(), 0,
                                        ctx->self_k_host[l].size() * sizeof(float));
                snprintf(nm, sizeof(nm), "svin_%d", l);
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, nm), ctx->self_v_host[l].data(), 0,
                                        ctx->self_v_host[l].size() * sizeof(float));
            }
        }
        ggml_backend_sched_graph_compute(ctx->sched, gf);
        std::vector<float> logits(V);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logits"), logits.data(), 0, V * sizeof(float));
        for (int l = 0; l < nl; l++) {
            char nm[16];
            float kn[4096], vn[4096]; // C <= 4096
            snprintf(nm, sizeof(nm), "sk_%d", l);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, nm), kn, 0, C * sizeof(float));
            snprintf(nm, sizeof(nm), "sv_%d", l);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, nm), vn, 0, C * sizeof(float));
            ctx->self_k_host[l].insert(ctx->self_k_host[l].end(), kn, kn + C);
            ctx->self_v_host[l].insert(ctx->self_v_host[l].end(), vn, vn + C);
        }
        ggml_free(g);
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < V; i++)
            if (logits[i] > bv) {
                bv = logits[i];
                best = i;
            }
        if (best == hp.eos_token) break;
        if (best >= 0 && best < (int)ctx->vocab.size()) {
            if (!ctx->result.empty()) ctx->result += ' ';
            ctx->result += ctx->vocab[best];
        }
        id = best;
    }
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------
smt_ocr_context * smt_ocr_init(const char * model_path, int n_threads) {
    auto ctx = new smt_ocr_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    gguf_context * gc = core_gguf::open_metadata(model_path);
    if (!gc) {
        fprintf(stderr, "smt_ocr: can't open %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    auto & hp = ctx->hp;
    hp.enc_num_stages = core_gguf::kv_u32(gc, "smt.encoder.num_stages", 3);
    hp.enc_reduction = core_gguf::kv_u32(gc, "smt.encoder.reduction", 16);
    hp.enc_num_channels = core_gguf::kv_u32(gc, "smt.encoder.num_channels", 1);
    hp.enc_stem_kernel = core_gguf::kv_u32(gc, "smt.encoder.stem_kernel", 4);
    hp.dec_layers = core_gguf::kv_u32(gc, "smt.decoder.num_layers", 8);
    hp.d_model = core_gguf::kv_u32(gc, "smt.decoder.d_model", 256);
    hp.num_heads = core_gguf::kv_u32(gc, "smt.decoder.num_heads", 4);
    hp.dim_ff = core_gguf::kv_u32(gc, "smt.decoder.dim_ff", 256);
    hp.vocab_size = core_gguf::kv_u32(gc, "smt.decoder.vocab_size", 20578);
    hp.maxlen = core_gguf::kv_u32(gc, "smt.decoder.maxlen", 1281);
    hp.maxh = core_gguf::kv_u32(gc, "smt.decoder.maxh", 256);
    hp.maxw = core_gguf::kv_u32(gc, "smt.decoder.maxw", 3056);
    hp.bos_token = core_gguf::kv_u32(gc, "smt.bos_token_id", 4426);
    hp.eos_token = core_gguf::kv_u32(gc, "smt.eos_token_id", 8822);
    hp.pad_token = core_gguf::kv_u32(gc, "smt.pad_token_id", 0);
    {
        int idx = gguf_find_key(gc, "smt.scale_attention");
        hp.scale_attention = (idx < 0) ? 0 : (gguf_get_val_bool(gc, idx) ? 1 : 0);
    }
    ctx->stage_sizes = { 64, 128, 256 };
    ctx->stage_depths = { 3, 3, 9 };
    ctx->vocab = core_gguf::kv_str_array(gc, "tokenizer.tokens");
    core_gguf::free_metadata(gc);

    fprintf(stderr, "smt_ocr: enc %d stages /%d, dec %dL/%dH/d%d ff%d vocab=%d(%zu) scale_attn=%d\n", hp.enc_num_stages,
            hp.enc_reduction, hp.dec_layers, hp.num_heads, hp.d_model, hp.dim_ff, hp.vocab_size, ctx->vocab.size(),
            hp.scale_attention);

    bool force_cpu = (getenv("SMT_OCR_FORCE_CPU") && atoi(getenv("SMT_OCR_FORCE_CPU")));
    ctx->backend = force_cpu ? ggml_backend_cpu_init() : crispasr_init_gpu_backend();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx->backend)) ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    ctx->backend_cpu = ggml_backend_is_cpu(ctx->backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);

    if (!core_gguf::load_weights(model_path, ctx->backend, "smt_ocr", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    std::vector<ggml_backend_t> backends = { ctx->backend };
    if (ctx->backend_cpu) backends.push_back(ctx->backend_cpu);
    ctx->sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), 8192, false, false);

    map_tensors(ctx);
    fprintf(stderr, "smt_ocr: loaded %zu tensors, init complete\n", ctx->wl.tensors.size());
    return ctx;
}

void smt_ocr_free(smt_ocr_context * ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

const smt_ocr_hparams * smt_ocr_get_hparams(const smt_ocr_context * ctx) {
    return ctx ? &ctx->hp : nullptr;
}

// ---------------------------------------------------------------------------
// Greedy inference
// ---------------------------------------------------------------------------
// Full-recompute greedy decode (O(L²)); kept as an A/B reference behind
// SMT_OCR_FULL_DECODE=1. The KV-cached path (decode_greedy_kv) is the default.
static void decode_greedy_full(smt_ocr_context * ctx) {
    std::vector<int32_t> ids = { ctx->hp.bos_token };
    ctx->result.clear();
    int V = ctx->hp.vocab_size;
    for (int step = 0; step < ctx->hp.maxlen; step++) {
        ggml_context * g = nullptr;
        ggml_cgraph * gf = nullptr;
        if (!run_decoder(ctx, ids, &g, &gf)) break;
        ggml_tensor * lt = ggml_graph_get_tensor(gf, "logits"); // [V,L]
        int L = (int)ids.size();
        std::vector<float> last(V);
        ggml_backend_tensor_get(lt, last.data(), (size_t)(L - 1) * V * sizeof(float), V * sizeof(float));
        ggml_free(g);
        int best = 0;
        float bv = last[0];
        for (int i = 1; i < V; i++)
            if (last[i] > bv) {
                bv = last[i];
                best = i;
            }
        if (best == ctx->hp.eos_token) break;
        if (best >= 0 && best < (int)ctx->vocab.size()) {
            if (!ctx->result.empty()) ctx->result += ' ';
            ctx->result += ctx->vocab[best];
        }
        ids.push_back(best);
    }
}

const char * smt_ocr_recognize(smt_ocr_context * ctx, const float * pixels, int width, int height, int * out_len) {
    // pixels: grayscale [0,1] row-major, already inverted+resized by the caller.
    if (!run_encoder(ctx, pixels, width, height, nullptr, nullptr)) return nullptr;
    if (getenv("SMT_OCR_FULL_DECODE"))
        decode_greedy_full(ctx);
    else
        decode_greedy_kv(ctx);
    if (out_len) *out_len = (int)ctx->result.size();
    return ctx->result.c_str();
}

const char * smt_ocr_recognize_raw(smt_ocr_context * ctx, const uint8_t * data, int w, int h, int ch, int * out_len) {
    // Mirror the SMT single-system inference pipeline (SMT-main data.py
    // prepare_data + convert_img_to_tensor), reduce_ratio=1.0:
    //   width = min(w, 3056); height = max(h, 256); cv2.resize (bilinear, uint8)
    //   → Grayscale (RGB ITU-R 601 luma) → /255. NO invert (SMT-main's
    //   convert_img_to_tensor is Grayscale→ToTensor; inverting drops accuracy
    //   from ~96% to ~30%), NO channel swap (HF dataset feeds RGB).
    if (!data || w <= 0 || h <= 0 || ch <= 0) return nullptr;
    int rw = w > 3056 ? 3056 : w;
    int rh = h < 256 ? 256 : h;
    float sx = (float)w / rw, sy = (float)h / rh;
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    int nk = ch >= 3 ? 3 : 1;
    std::vector<float> gray((size_t)rw * rh);
    for (int y = 0; y < rh; y++) {
        float fy = (y + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(fy);
        float wy = fy - y0;
        int y0c = clampi(y0, 0, h - 1), y1c = clampi(y0 + 1, 0, h - 1);
        for (int x = 0; x < rw; x++) {
            float fx = (x + 0.5f) * sx - 0.5f;
            int x0 = (int)std::floor(fx);
            float wx = fx - x0;
            int x0c = clampi(x0, 0, w - 1), x1c = clampi(x0 + 1, 0, w - 1);
            float pix[3] = { 0, 0, 0 };
            for (int k = 0; k < nk; k++) {
                float p00 = data[((size_t)y0c * w + x0c) * ch + k];
                float p01 = data[((size_t)y0c * w + x1c) * ch + k];
                float p10 = data[((size_t)y1c * w + x0c) * ch + k];
                float p11 = data[((size_t)y1c * w + x1c) * ch + k];
                float top = p00 + wx * (p01 - p00), bot = p10 + wx * (p11 - p10);
                pix[k] = std::round(top + wy * (bot - top)); // cv2 returns uint8
            }
            float luma = (nk >= 3) ? 0.299f * pix[0] + 0.587f * pix[1] + 0.114f * pix[2] // RGB luma, no invert
                                   : pix[0];
            gray[(size_t)y * rw + x] = std::round(luma) / 255.0f;
        }
    }
    return smt_ocr_recognize(ctx, gray.data(), rw, rh, out_len);
}

const char * smt_ocr_recognize_file(smt_ocr_context * ctx, const char * image_path, int * out_len) {
    int w = 0, h = 0, c = 0;
    stbi_uc * data = stbi_load(image_path, &w, &h, &c, 3); // force RGB
    if (!data) return nullptr;
    const char * r = smt_ocr_recognize_raw(ctx, data, w, h, 3, out_len);
    stbi_image_free(data);
    return r;
}

// ---------------------------------------------------------------------------
// Per-stage parity harness
// ---------------------------------------------------------------------------
int smt_ocr_run_diff(smt_ocr_context * ctx, const char * ref_path) {
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) return 1;
    int C = ctx->hp.d_model;

    // input_tensor stored (C_in,H,W) → ggml [W,H,C_in,1] flat is identical order
    auto inshape = ref.shape("input_tensor"); // gguf ne = [W,H,C_in]
    if (inshape.size() < 2) {
        fprintf(stderr, "smt_ocr_diff: no input_tensor\n");
        return 1;
    }
    int W = (int)inshape[0], H = (int)inshape[1];
    auto in = ref.get_f32("input_tensor");
    fprintf(stderr, "smt_ocr_diff: input W=%d H=%d (%zu floats)\n", W, H, in.second);

    int fails = 0;
    auto report = [&](const char * nm, const crispembed_diff::Report & r) {
        bool pass = r.is_pass();
        fprintf(stderr, "[smt-diff] %-12s cos_min=%.6f cos_mean=%.6f max_abs=%.2e %s\n", nm, r.cos_min, r.cos_mean,
                r.max_abs, pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    };

    // ---- encoder ----
    ggml_context * eg = nullptr;
    ggml_cgraph * egf = nullptr;
    if (!run_encoder(ctx, in.first, W, H, &eg, &egf)) return 1;
    for (int s = 0; s < ctx->hp.enc_num_stages; s++) {
        char nm[24];
        snprintf(nm, sizeof(nm), "enc_stage%d", s);
        if (!ref.has(nm)) continue;
        ggml_tensor * t = ggml_graph_get_tensor(egf, nm);
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        report(nm, ref.compare(nm, buf.data(), buf.size(), 0)); // row_dim=0 → D=C
    }
    if (ref.has("enc_output"))
        report("enc_output", ref.compare("enc_output", ctx->mem_value.data(), ctx->mem_value.size(), 0));
    if (ref.has("mem_key")) report("mem_key", ref.compare("mem_key", ctx->mem_key.data(), ctx->mem_key.size(), 0));
    ggml_free(eg);

    // ---- decoder (teacher-forced on ref token_ids) ----
    auto tid = ref.get_f32("token_ids"); // stored I32 → floats
    std::vector<int32_t> ids(tid.second);
    for (size_t i = 0; i < tid.second; i++) ids[i] = (int32_t)llround(tid.first[i]);
    int L = (int)ids.size();
    if (L == 0) {
        fprintf(stderr, "smt_ocr_diff: no token_ids in ref — skipping decoder stages\n");
        return fails ? 1 : 0;
    }
    fprintf(stderr, "smt_ocr_diff: teacher-forcing %d tokens\n", L);

    ggml_context * dg = nullptr;
    ggml_cgraph * dgf = nullptr;
    if (!run_decoder(ctx, ids, &dg, &dgf)) return 1;
    auto readcmp = [&](const char * nm, int rowD) {
        if (!ref.has(nm)) return;
        ggml_tensor * t = ggml_graph_get_tensor(dgf, nm);
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        report(nm, ref.compare(nm, buf.data(), buf.size(), rowD));
    };
    readcmp("dec_tok_emb", 0);
    for (int i = 0; i < ctx->hp.dec_layers; i++) {
        char nm[24];
        snprintf(nm, sizeof(nm), "dec_layer%d", i);
        readcmp(nm, 0);
    }
    readcmp("logits", 0);
    ggml_free(dg);

    fprintf(stderr, "smt_ocr_diff: %s (%d stage failures)\n", fails ? "FAIL" : "PASS", fails);
    (void)C;
    return fails ? 1 : 0;
}
