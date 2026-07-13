// transcoda_ocr.cpp — Transcoda-59M zero-shot OMR via ggml graph compute.
//
// ConvNeXt-V2-Tiny encoder + 2-layer projector + 2D-sinusoidal-PE bridge +
// 8-layer pre-LN RoPE cross-attention Transformer decoder → Humdrum **kern.
//
// Clean-room: written from the paper (arXiv 2605.10835), the HF config/data
// files, and an oracle activation dump. The AGPL reference source is NOT read.
//
// Facts pinned against the oracle (tools/dump_transcoda_reference.py):
//   • pixel_values normalized to [-1,1] via (x/255-0.5)/0.5 (NOT ImageNet).
//   • ConvNeXt-V2 block: dwconv7x7 → LN(1e-6) → pwconv1(×4) → GELU-erf → GRN →
//     pwconv2, residual, NO LayerScale. Encoder final layernorm is UNUSED (dead).
//   • projector: fc1(768→2048) → GELU-erf → fc2(2048→512), on the flattened grid.
//   • memory: enc_pos = projector_out + 2D-PE (cross-attn KEY); enc_raw =
//     projector_out (cross-attn VALUE). 2D PE = half rows(h) / half cols(w),
//     freq exp(-2k/512·ln1e4).
//   • decoder self-attn: fused qkv, RoPE (torchtune, adjacent-pair = NORMAL,
//     θ=1e4, hd=64), causal, scale 1/√64; embedding UNscaled.
//   • cross-attn: scale 1/√64, no mask (all encoder tokens valid on a full page).
//   • pre-LN (x += Sub(LN(x))); FFN 512→1024→512 GELU-erf; final_norm; untied
//     LM head (vocab_projection, with bias). greedy: repetition_penalty 1.1.

#include "transcoda_ocr.h"
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
struct tc_enc_layer {
    ggml_tensor *dw_w, *dw_b, *ln_w, *ln_b, *pw1_w, *pw1_b, *grn_w, *grn_b, *pw2_w, *pw2_b;
};
struct tc_enc_stage {
    ggml_tensor *ds_ln_w, *ds_ln_b, *ds_conv_w, *ds_conv_b; // downsampling (null for stage 0)
    std::vector<tc_enc_layer> layers;
};
struct tc_dec_layer {
    ggml_tensor *qkv_w, *qkv_b, *sa_o_w, *sa_o_b; // fused self-attn
    ggml_tensor *ca_q_w, *ca_q_b, *ca_k_w, *ca_k_b, *ca_v_w, *ca_v_b, *ca_o_w, *ca_o_b;
    ggml_tensor *n0_w, *n0_b, *n1_w, *n1_b, *n2_w, *n2_b; // pre-LN
    ggml_tensor *ff0_w, *ff0_b, *ff3_w, *ff3_b;
};

struct transcoda_ocr_context {
    transcoda_ocr_hparams hp;
    // encoder
    ggml_tensor *stem_w, *stem_b, *stem_ln_w, *stem_ln_b;
    std::vector<int> stage_sizes, stage_depths;
    std::vector<tc_enc_stage> stages;
    // projector
    ggml_tensor *pj1_w, *pj1_b, *pj2_w, *pj2_b;
    // decoder
    ggml_tensor *tok_embed, *final_w, *final_b, *lm_w, *lm_b;
    std::vector<tc_dec_layer> dec;

    std::vector<std::string> vocab;
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr, backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    int n_threads = 4;
    std::string result;

    // cached encoder memory (host): C=d_model, token-major (c fastest)
    std::vector<float> mem_pos, mem_raw; // [n_enc * C]
    int n_enc = 0, enc_h = 0, enc_w = 0;
    // KV cache: cross K from mem_pos, cross V from mem_raw (constant per image);
    // self K/V (already RoPE'd) grow one token per step.
    std::vector<std::vector<float>> cross_k_host, cross_v_host;
    std::vector<std::vector<float>> self_k_host, self_v_host;

    // Persistent (device-resident) KV cache for the fast decode path. Cross K/V
    // are computed once; self K/V are written in-graph per step — no host
    // round-trip, no per-step re-upload (the host path re-uploaded ~48 MB/step).
    ggml_context * kv_ctx = nullptr;
    ggml_backend_buffer_t kv_buf = nullptr;
    ggml_tensor *pk_self_k = nullptr, *pk_self_v = nullptr;   // [C, max_seq, nl]
    ggml_tensor *pk_cross_k = nullptr, *pk_cross_v = nullptr; // [C, nE, nl]
    int pk_max_seq = 0, pk_nE = 0;
    std::vector<uint8_t> compute_meta;
};

// ---------------------------------------------------------------------------
static ggml_tensor * F(const std::unordered_map<std::string, ggml_tensor *> & m, const std::string & n) {
    auto it = m.find(n);
    return it != m.end() ? it->second : nullptr;
}

static void map_tensors(transcoda_ocr_context * ctx) {
    const auto & m = ctx->wl.tensors;
    const auto & hp = ctx->hp;
    char b[128];

    ctx->stem_w = F(m, "enc.embed.patch.weight");
    ctx->stem_b = F(m, "enc.embed.patch.bias");
    ctx->stem_ln_w = F(m, "enc.embed.ln.weight");
    ctx->stem_ln_b = F(m, "enc.embed.ln.bias");

    ctx->stages.resize(hp.enc_num_stages);
    for (int s = 0; s < hp.enc_num_stages; s++) {
        auto & S = ctx->stages[s];
        snprintf(b, sizeof(b), "enc.st%d.ds.ln.weight", s);
        S.ds_ln_w = F(m, b);
        snprintf(b, sizeof(b), "enc.st%d.ds.ln.bias", s);
        S.ds_ln_b = F(m, b);
        snprintf(b, sizeof(b), "enc.st%d.ds.conv.weight", s);
        S.ds_conv_w = F(m, b);
        snprintf(b, sizeof(b), "enc.st%d.ds.conv.bias", s);
        S.ds_conv_b = F(m, b);
        S.layers.resize(ctx->stage_depths[s]);
        for (int i = 0; i < ctx->stage_depths[s]; i++) {
            auto & L = S.layers[i];
            auto E = [&](const char * suf) {
                snprintf(b, sizeof(b), "enc.st%d.l%d.%s", s, i, suf);
                return F(m, b);
            };
            L.dw_w = E("dw.weight");
            L.dw_b = E("dw.bias");
            L.ln_w = E("ln.weight");
            L.ln_b = E("ln.bias");
            L.pw1_w = E("pw1.weight");
            L.pw1_b = E("pw1.bias");
            L.grn_w = E("grn.weight");
            L.grn_b = E("grn.bias");
            L.pw2_w = E("pw2.weight");
            L.pw2_b = E("pw2.bias");
        }
    }

    ctx->pj1_w = F(m, "proj.fc1.weight");
    ctx->pj1_b = F(m, "proj.fc1.bias");
    ctx->pj2_w = F(m, "proj.fc2.weight");
    ctx->pj2_b = F(m, "proj.fc2.bias");

    ctx->tok_embed = F(m, "dec.tok_embed.weight");
    ctx->final_w = F(m, "dec.final_norm.weight");
    ctx->final_b = F(m, "dec.final_norm.bias");
    ctx->lm_w = F(m, "dec.lm_head.weight");
    ctx->lm_b = F(m, "dec.lm_head.bias");

    ctx->dec.resize(hp.n_layers);
    for (int i = 0; i < hp.n_layers; i++) {
        auto & L = ctx->dec[i];
        auto D = [&](const char * suf) {
            snprintf(b, sizeof(b), "dec.l%d.%s", i, suf);
            return F(m, b);
        };
        L.qkv_w = D("qkv.weight");
        L.qkv_b = D("qkv.bias");
        L.sa_o_w = D("sa_out.weight");
        L.sa_o_b = D("sa_out.bias");
        L.ca_q_w = D("ca_q.weight");
        L.ca_q_b = D("ca_q.bias");
        L.ca_k_w = D("ca_k.weight");
        L.ca_k_b = D("ca_k.bias");
        L.ca_v_w = D("ca_v.weight");
        L.ca_v_b = D("ca_v.bias");
        L.ca_o_w = D("ca_out.weight");
        L.ca_o_b = D("ca_out.bias");
        L.n0_w = D("n0.weight");
        L.n0_b = D("n0.bias");
        L.n1_w = D("n1.weight");
        L.n1_b = D("n1.bias");
        L.n2_w = D("n2.weight");
        L.n2_b = D("n2.bias");
        L.ff0_w = D("ff0.weight");
        L.ff0_b = D("ff0.bias");
        L.ff3_w = D("ff3.weight");
        L.ff3_b = D("ff3.bias");
    }
}

// ---------------------------------------------------------------------------
// graph helpers
// ---------------------------------------------------------------------------
static ggml_tensor * f32(ggml_context * g, ggml_tensor * t) {
    return (!t || t->type == GGML_TYPE_F32) ? t : ggml_cast(g, t, GGML_TYPE_F32);
}
// LayerNorm over ne[0] (channels-last). eps=1e-6.
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
// channels-first LayerNorm on a map [W,H,C,N]
static ggml_tensor * ln_cf(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    ggml_tensor * xc = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3)); // [C,W,H,N]
    xc = ln0(g, xc, w, b);
    return ggml_cont(g, ggml_permute(g, xc, 2, 0, 1, 3)); // back to [W,H,C,N]
}
// conv2d with in-graph F16 kernel cast; kernel [KW,KH,IC,OC], x=[W,H,IC,N].
static ggml_tensor * conv(ggml_context * g, ggml_tensor * a, ggml_tensor * x, int s, int p, bool dw, int kw, int kh,
                          int ic, int oc) {
    if (ggml_n_dims(a) != 4) a = ggml_reshape_4d(g, a, kw, kh, ic, oc);
    ggml_tensor * k = ggml_cast(g, a, GGML_TYPE_F16);
    return dw ? ggml_conv_2d_dw(g, k, x, s, s, p, p, 1, 1) : ggml_conv_2d(g, k, x, s, s, p, p, 1, 1);
}
// flatten a feature map [W,H,C,N=1] → token tensor [C, W*H] (c fastest per token,
// token index t = h*W + w — matches the oracle's flatten(2).transpose).
static ggml_tensor * to_tokens(ggml_context * g, ggml_tensor * map) {
    int W = map->ne[0], H = map->ne[1], C = map->ne[2];
    ggml_tensor * t = ggml_cont(g, ggml_permute(g, map, 1, 2, 0, 3)); // [C,W,H,N]
    return ggml_reshape_2d(g, t, C, W * H);                           // [C, W*H]
}
// ConvNeXt-V2 Global Response Normalization on x=[C,W,H,N] (channels-first-ish:
// ne0=channels). GRN(x)=x + γ·(x·Nx)+β, Nx=L2(x over W,H per channel)/mean_c.
static ggml_tensor * grn(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    int C = x->ne[0], Wd = x->ne[1], Hd = x->ne[2], S = Wd * Hd;
    ggml_tensor * xf = ggml_reshape_2d(g, x, C, S);         // [C,S]
    ggml_tensor * xt = ggml_cont(g, ggml_transpose(g, xf)); // [S,C]
    ggml_tensor * ss = ggml_sum_rows(g, ggml_sqr(g, xt));   // [1,C] sum_S x^2
    ggml_tensor * Gx = ggml_sqrt(g, ss);                    // [1,C] L2 per channel
    ggml_tensor * mean = ggml_scale(g, ggml_sum_rows(g, ggml_cont(g, ggml_transpose(g, Gx))),
                                    1.0f / (float)C);        // [1,1] mean over channels
    ggml_tensor * Nx = ggml_div(g, Gx, mean);                // [1,C] broadcast [1,1]
    ggml_tensor * Nxc = ggml_cont(g, ggml_transpose(g, Nx)); // [C,1]
    ggml_tensor * xn = ggml_mul(g, xf, Nxc);                 // [C,S] · [C,1]
    ggml_tensor * gw = ggml_reshape_2d(g, f32(g, w), C, 1);
    ggml_tensor * gb = ggml_reshape_2d(g, f32(g, b), C, 1);
    ggml_tensor * out = ggml_add(g, ggml_add(g, ggml_mul(g, xn, gw), gb), xf); // γ(x·Nx)+β+x
    return ggml_reshape_4d(g, out, C, Wd, Hd, 1);
}
// scaled multi-head attention core (q,k,v are [C,L]; returns concat-heads [C,Lq]).
static ggml_tensor * mha(ggml_context * g, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int nh, int Lq, int Lk,
                         ggml_tensor * mask, float scale) {
    int C = q->ne[0], hd = C / nh;
    ggml_tensor * Q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, q, hd, nh, Lq), 0, 2, 1, 3)); // [hd,Lq,nh]
    ggml_tensor * K = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, k, hd, nh, Lk), 0, 2, 1, 3)); // [hd,Lk,nh]
    ggml_tensor * V = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, v, hd, nh, Lk), 0, 2, 1, 3)); // [hd,Lk,nh]
    ggml_tensor * scores = ggml_mul_mat(g, K, Q);                                                   // [Lk,Lq,nh]
    scores = ggml_soft_max_ext(g, scores, mask, scale, 0.0f);
    ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3)); // [Lk,hd,nh]
    ggml_tensor * a = ggml_mul_mat(g, Vt, scores);                   // [hd,Lq,nh]
    a = ggml_cont(g, ggml_permute(g, a, 0, 2, 1, 3));                // [hd,nh,Lq]
    return ggml_reshape_2d(g, a, C, Lq);
}
// self-attention with RoPE (torchtune adjacent-pair = GGML_ROPE_TYPE_NORMAL).
// q,k passed [C,L]; roped in [hd,nh,L] layout, then attention (scale, causal mask).
static ggml_tensor * self_attn_rope(ggml_context * g, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int nh, int Lq,
                                    int Lk, ggml_tensor * pos, float theta, ggml_tensor * mask, float scale) {
    int C = q->ne[0], hd = C / nh;
    ggml_tensor * Q = ggml_reshape_3d(g, q, hd, nh, Lq);
    ggml_tensor * K = ggml_reshape_3d(g, k, hd, nh, Lk);
    Q = ggml_rope_ext(g, Q, pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(g, K, pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3)); // [hd,Lq,nh]
    K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3)); // [hd,Lk,nh]
    ggml_tensor * V = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, v, hd, nh, Lk), 0, 2, 1, 3));
    ggml_tensor * scores = ggml_mul_mat(g, K, Q); // [Lk,Lq,nh]
    scores = ggml_soft_max_ext(g, scores, mask, scale, 0.0f);
    ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
    ggml_tensor * a = ggml_mul_mat(g, Vt, scores);
    a = ggml_cont(g, ggml_permute(g, a, 0, 2, 1, 3));
    return ggml_reshape_2d(g, a, C, Lq);
}

// ---------------------------------------------------------------------------
// Encoder + projector graph: input [W,H,3,1] → enc_grid [768,n] + enc_raw [512,n]
// ---------------------------------------------------------------------------
static ggml_cgraph * build_encoder(transcoda_ocr_context * ctx, ggml_context * g, int W, int H) {
    ggml_cgraph * gf = ggml_new_graph_custom(g, 16384, false);
    ggml_tensor * x = ggml_new_tensor_4d(g, GGML_TYPE_F32, W, H, ctx->hp.enc_num_channels, 1);
    ggml_set_name(x, "input");
    ggml_set_input(x);

    const int K = ctx->hp.enc_stem_kernel;
    int cur_ch = ctx->stage_sizes[0];
    ggml_tensor * cur = conv(g, ctx->stem_w, x, K, 0, false, K, K, ctx->hp.enc_num_channels, cur_ch);
    cur = add_cbias(g, cur, ctx->stem_b);
    cur = ln_cf(g, cur, ctx->stem_ln_w, ctx->stem_ln_b);

    for (int s = 0; s < ctx->hp.enc_num_stages; s++) {
        auto & S = ctx->stages[s];
        int Cs = ctx->stage_sizes[s];
        if (S.ds_conv_w) { // channels-first LN → Conv2d(Cprev→Cs, k2, s2)
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
            yc = grn(g, yc, L.grn_w, L.grn_b);                 // GRN over [4C,W,H] (V2)
            yc = lin(g, yc, L.pw2_w, L.pw2_b);                 // [C,...]
            y = ggml_cont(g, ggml_permute(g, yc, 2, 0, 1, 3)); // back to [W,H,C,N]
            cur = ggml_add(g, inp, y);                         // residual, NO LayerScale
        }
    }
    ggml_tensor * grid = to_tokens(g, cur); // [768, n_enc]
    ggml_set_name(grid, "enc_grid");
    ggml_set_output(grid);
    ggml_build_forward_expand(gf, grid);

    // projector: fc1(768→2048) → GELU-erf → fc2(2048→512)
    ggml_tensor * mem = lin(g, grid, ctx->pj1_w, ctx->pj1_b);
    mem = ggml_gelu_erf(g, mem);
    mem = lin(g, mem, ctx->pj2_w, ctx->pj2_b); // [512, n_enc]
    ggml_set_name(mem, "enc_raw");
    ggml_set_output(mem);
    ggml_build_forward_expand(gf, mem);
    return gf;
}

// 2D sinusoidal PE for grid (eh rows × ew cols) at dim C, token-major [n*C]
// (c fastest, t=h*ew+w). First C/2 dims encode row h, second C/2 encode col w;
// freq d = exp(-(2k)/C · ln 1e4).
static std::vector<float> gen_pe2d(int C, int eh, int ew) {
    int n = eh * ew, half = C / 2;
    std::vector<float> pe((size_t)n * C, 0.0f);
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

// run encoder on a preprocessed image [3,H,W] → fills mem_pos / mem_raw
static bool run_encoder(transcoda_ocr_context * ctx, const float * input, int W, int H, ggml_context ** keep_g,
                        ggml_cgraph ** keep_gf) {
    size_t meta = 128 * 1024 * 1024;
    std::vector<uint8_t> * buf = new std::vector<uint8_t>(meta);
    ggml_init_params ip = { meta, buf->data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = build_encoder(ctx, g, W, H);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "transcoda_ocr: encoder alloc failed\n");
        ggml_free(g);
        delete buf;
        return false;
    }
    ggml_tensor * inp = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(inp, input, 0, (size_t)W * H * ctx->hp.enc_num_channels * sizeof(float));
    ggml_backend_sched_graph_compute(ctx->sched, gf);

    ggml_tensor * mem = ggml_graph_get_tensor(gf, "enc_raw");
    int C = mem->ne[0];
    ctx->n_enc = mem->ne[1];
    ctx->enc_h = H / ctx->hp.enc_reduction;
    ctx->enc_w = W / ctx->hp.enc_reduction;
    ctx->mem_raw.resize((size_t)ctx->n_enc * C);
    ggml_backend_tensor_get(mem, ctx->mem_raw.data(), 0, ctx->mem_raw.size() * sizeof(float));

    // enc_pos = enc_raw + 2D PE (over the actual grid dims from n_enc)
    std::vector<float> pe = gen_pe2d(C, ctx->enc_h, ctx->enc_w);
    if ((int)(pe.size() / C) != ctx->n_enc) {
        // grid dims from reduction disagree with the conv output length; recompute
        // ew from n_enc using the observed width, fall back to raw.
        fprintf(stderr, "transcoda_ocr: PE grid %d*%d=%zu != n_enc=%d\n", ctx->enc_h, ctx->enc_w, pe.size() / C,
                ctx->n_enc);
    }
    ctx->mem_pos.resize(ctx->mem_raw.size());
    for (size_t i = 0; i < ctx->mem_raw.size() && i < pe.size(); i++) ctx->mem_pos[i] = ctx->mem_raw[i] + pe[i];

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
// Teacher-forced decoder graph (full L; pre-LN, RoPE causal self-attn, dual-mem
// cross-attn). Emits dec_block0..7 + logits.
// ---------------------------------------------------------------------------
static ggml_cgraph * build_decoder(transcoda_ocr_context * ctx, ggml_context * g, int L) {
    const auto & hp = ctx->hp;
    int C = hp.d_model, nh = hp.n_heads, nE = ctx->n_enc;
    float scale = 1.0f / std::sqrt((float)(C / nh));
    ggml_cgraph * gf = ggml_new_graph_custom(g, hp.n_layers * 128 + 512, false);

    ggml_tensor * tokens = ggml_new_tensor_1d(g, GGML_TYPE_I32, L);
    ggml_set_name(tokens, "dec_tokens");
    ggml_set_input(tokens);
    ggml_tensor * pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, L);
    ggml_set_name(pos, "dec_pos");
    ggml_set_input(pos);
    ggml_tensor * mk = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nE);
    ggml_set_name(mk, "mem_pos");
    ggml_set_input(mk);
    ggml_tensor * mv = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nE);
    ggml_set_name(mv, "mem_raw");
    ggml_set_input(mv);
    ggml_tensor * mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, L, L);
    ggml_set_name(mask, "dec_mask");
    ggml_set_input(mask);

    ggml_tensor * cur = ggml_get_rows(g, ctx->tok_embed, tokens); // [C,L] (unscaled)
    ggml_set_name(cur, "dec_tok_emb");
    ggml_set_output(cur);

    for (int i = 0; i < hp.n_layers; i++) {
        auto & Lr = ctx->dec[i];
        // pre-LN self-attention (RoPE, causal)
        ggml_tensor * h = ln0(g, cur, Lr.n0_w, Lr.n0_b);
        ggml_tensor * qkv = lin(g, h, Lr.qkv_w, Lr.qkv_b); // [3C,L]
        size_t es = ggml_element_size(qkv);
        ggml_tensor * q = ggml_cont(g, ggml_view_2d(g, qkv, C, L, qkv->nb[1], 0));
        ggml_tensor * k = ggml_cont(g, ggml_view_2d(g, qkv, C, L, qkv->nb[1], (size_t)C * es));
        ggml_tensor * v = ggml_cont(g, ggml_view_2d(g, qkv, C, L, qkv->nb[1], (size_t)2 * C * es));
        ggml_tensor * sa = self_attn_rope(g, q, k, v, nh, L, L, pos, hp.rope_theta, mask, scale);
        sa = lin(g, sa, Lr.sa_o_w, Lr.sa_o_b);
        cur = ggml_add(g, cur, sa);
        // pre-LN cross-attention (key=mem_pos, value=mem_raw, no mask)
        h = ln0(g, cur, Lr.n1_w, Lr.n1_b);
        ggml_tensor * cq = lin(g, h, Lr.ca_q_w, Lr.ca_q_b);
        ggml_tensor * ck = lin(g, mk, Lr.ca_k_w, Lr.ca_k_b);
        ggml_tensor * cv = lin(g, mv, Lr.ca_v_w, Lr.ca_v_b);
        ggml_tensor * ca = mha(g, cq, ck, cv, nh, L, nE, nullptr, scale);
        ca = lin(g, ca, Lr.ca_o_w, Lr.ca_o_b);
        cur = ggml_add(g, cur, ca);
        // pre-LN FFN (GELU-erf)
        h = ln0(g, cur, Lr.n2_w, Lr.n2_b);
        ggml_tensor * ff = lin(g, h, Lr.ff0_w, Lr.ff0_b);
        ff = ggml_gelu_erf(g, ff);
        ff = lin(g, ff, Lr.ff3_w, Lr.ff3_b);
        cur = ggml_add(g, cur, ff);
        char nm[24];
        snprintf(nm, sizeof(nm), "dec_block%d", i);
        ggml_set_name(cur, nm);
        ggml_set_output(cur);
    }
    ggml_tensor * normed = ln0(g, cur, ctx->final_w, ctx->final_b);
    ggml_tensor * logits = lin(g, normed, ctx->lm_w, ctx->lm_b); // [V,L]
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

static bool run_decoder(transcoda_ocr_context * ctx, const std::vector<int32_t> & ids, ggml_context ** keep_g,
                        ggml_cgraph ** keep_gf) {
    int L = (int)ids.size(), C = ctx->hp.d_model;
    size_t meta = 96 * 1024 * 1024;
    std::vector<uint8_t> * buf = new std::vector<uint8_t>(meta);
    ggml_init_params ip = { meta, buf->data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = build_decoder(ctx, g, L);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "transcoda_ocr: decoder alloc failed\n");
        ggml_free(g);
        delete buf;
        return false;
    }
    std::vector<int32_t> posv(L);
    for (int i = 0; i < L; i++) posv[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_tokens"), ids.data(), 0, L * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_pos"), posv.data(), 0, L * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mem_pos"), ctx->mem_pos.data(), 0,
                            ctx->mem_pos.size() * sizeof(float));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mem_raw"), ctx->mem_raw.data(), 0,
                            ctx->mem_raw.size() * sizeof(float));
    std::vector<float> mask = causal_mask(L);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "dec_mask"), mask.data(), 0, mask.size() * sizeof(float));
    ggml_backend_sched_graph_compute(ctx->sched, gf);
    (void)C;
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
// Full-recompute greedy decode (O(L²) per step) with repetition_penalty=1.1.
// Correctness reference; gated behind default vs the KV-cached path below.
// ---------------------------------------------------------------------------
static int argmax_with_penalty(const float * logits, int V, const std::vector<int> & seen, float penalty) {
    std::vector<float> lg(logits, logits + V);
    if (penalty != 1.0f) {
        // HF RepetitionPenaltyLogitsProcessor penalizes each token that appears in
        // the running sequence exactly ONCE (regardless of how often it repeats) —
        // applying it per-occurrence over-suppresses frequent tokens (e.g. the '\n'
        // kern record separator), derailing the decode.
        std::vector<char> done(V, 0);
        for (int id : seen)
            if (id >= 0 && id < V && !done[id]) {
                lg[id] = lg[id] > 0 ? lg[id] / penalty : lg[id] * penalty;
                done[id] = 1;
            }
    }
    int best = 0;
    float bv = lg[0];
    for (int i = 1; i < V; i++)
        if (lg[i] > bv) {
            bv = lg[i];
            best = i;
        }
    return best;
}

static void decode_greedy_full(transcoda_ocr_context * ctx) {
    const auto & hp = ctx->hp;
    int V = hp.vocab_size;
    std::vector<int32_t> ids = { hp.bos_token };
    std::vector<int> seen = { hp.bos_token };
    ctx->result.clear();
    for (int step = 0; step < hp.max_seq_len; step++) {
        ggml_context * g = nullptr;
        ggml_cgraph * gf = nullptr;
        if (!run_decoder(ctx, ids, &g, &gf)) break;
        ggml_tensor * lt = ggml_graph_get_tensor(gf, "logits"); // [V,L]
        int L = (int)ids.size();
        std::vector<float> last(V);
        ggml_backend_tensor_get(lt, last.data(), (size_t)(L - 1) * V * sizeof(float), V * sizeof(float));
        ggml_free(g);
        int best = argmax_with_penalty(last.data(), V, seen, 1.1f);
        if (best == hp.eos_token) break;
        if (best >= 0 && best < (int)ctx->vocab.size()) {
            // Concatenate tokens directly: **kern structure lives in the literal
            // '\n'/'\t' vocab tokens, and tokens can contain '/' (e.g. "*M2/4"),
            // so no separator — this reconstructs the exact kern text.
            ctx->result += ctx->vocab[best];
        }
        ids.push_back(best);
        seen.push_back(best);
    }
}

// ---------------------------------------------------------------------------
// KV-cached greedy decode. cross K/V precomputed once (K from mem_pos, V from
// mem_raw); self K/V (RoPE'd at their absolute position) grow one per step.
// ---------------------------------------------------------------------------
static void precompute_cross_kv(transcoda_ocr_context * ctx) {
    const int C = ctx->hp.d_model, nE = ctx->n_enc, nl = ctx->hp.n_layers;
    ctx->cross_k_host.assign(nl, {});
    ctx->cross_v_host.assign(nl, {});
    size_t meta = 48 * 1024 * 1024;
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
    ggml_backend_tensor_set(mk, ctx->mem_pos.data(), 0, ctx->mem_pos.size() * sizeof(float));
    ggml_backend_tensor_set(mv, ctx->mem_raw.data(), 0, ctx->mem_raw.size() * sizeof(float));
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

// One decode step at absolute position n_cached. Self K/V for prior tokens come
// from the host cache; the new token's RoPE'd K/V are emitted as sk_i/sv_i.
static ggml_cgraph * build_decode_step(transcoda_ocr_context * ctx, ggml_context * g, int n_cached) {
    const auto & hp = ctx->hp;
    const int C = hp.d_model, nh = hp.n_heads, hd = C / nh, nE = ctx->n_enc;
    float scale = 1.0f / std::sqrt((float)hd);
    ggml_cgraph * gf = ggml_new_graph_custom(g, hp.n_layers * 160 + 512, false);

    ggml_tensor * token = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(token, "step_token");
    ggml_set_input(token);
    ggml_tensor * pos = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(pos, "step_pos");
    ggml_set_input(pos);
    ggml_tensor * cur = ggml_get_rows(g, ctx->tok_embed, token); // [C,1]

    for (int i = 0; i < hp.n_layers; i++) {
        auto & Lr = ctx->dec[i];
        char nm[16];
        ggml_tensor * h = ln0(g, cur, Lr.n0_w, Lr.n0_b);
        ggml_tensor * qkv = lin(g, h, Lr.qkv_w, Lr.qkv_b); // [3C,1]
        size_t es = ggml_element_size(qkv);
        ggml_tensor * q = ggml_cont(g, ggml_view_2d(g, qkv, C, 1, qkv->nb[1], 0));
        ggml_tensor * knew = ggml_cont(g, ggml_view_2d(g, qkv, C, 1, qkv->nb[1], (size_t)C * es));
        ggml_tensor * vnew = ggml_cont(g, ggml_view_2d(g, qkv, C, 1, qkv->nb[1], (size_t)2 * C * es));
        // RoPE q and knew at absolute position (single-token)
        ggml_tensor * Qr = ggml_rope_ext(g, ggml_reshape_3d(g, q, hd, nh, 1), pos, nullptr, hd, GGML_ROPE_TYPE_NORMAL,
                                         0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * Kr = ggml_rope_ext(g, ggml_reshape_3d(g, knew, hd, nh, 1), pos, nullptr, hd,
                                         GGML_ROPE_TYPE_NORMAL, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        // cont: this roped K is both consumed in-graph AND read back to the host
        // KV cache; a bare reshape view of Kr would be clobbered post-compute
        // (set_output on a view reads stale), corrupting every later step's self-K.
        ggml_tensor * krow = ggml_cont(g, ggml_reshape_2d(g, Kr, C, 1)); // roped K for this token
        snprintf(nm, sizeof(nm), "sk_%d", i);
        ggml_set_name(krow, nm);
        ggml_set_output(krow);
        ggml_build_forward_expand(gf, krow);
        snprintf(nm, sizeof(nm), "sv_%d", i);
        ggml_set_name(vnew, nm);
        ggml_set_output(vnew);
        ggml_build_forward_expand(gf, vnew);
        // assemble full (roped) K and V over all cached + current
        ggml_tensor *kfull = krow, *vfull = vnew;
        if (n_cached > 0) {
            ggml_tensor * skin = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, n_cached);
            snprintf(nm, sizeof(nm), "skin_%d", i);
            ggml_set_name(skin, nm);
            ggml_set_input(skin);
            ggml_tensor * svin = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, n_cached);
            snprintf(nm, sizeof(nm), "svin_%d", i);
            ggml_set_name(svin, nm);
            ggml_set_input(svin);
            kfull = ggml_concat(g, skin, krow, 1); // [C, n_cached+1]
            vfull = ggml_concat(g, svin, vnew, 1);
        }
        // attention with Q already roped: reuse mha but Q/K need [hd,L,nh] from
        // [C,·]; kfull/vfull are plain [C,Lk] (K already roped), Qr is [hd,nh,1].
        int Lk = n_cached + 1;
        ggml_tensor * Qp = ggml_cont(g, ggml_permute(g, Qr, 0, 2, 1, 3)); // [hd,1,nh]
        ggml_tensor * Kp = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, kfull, hd, nh, Lk), 0, 2, 1, 3));
        ggml_tensor * Vp = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, vfull, hd, nh, Lk), 0, 2, 1, 3));
        ggml_tensor * sc = ggml_mul_mat(g, Kp, Qp); // [Lk,1,nh]
        sc = ggml_soft_max_ext(g, sc, nullptr, scale, 0.0f);
        ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, Vp, 1, 0, 2, 3));
        ggml_tensor * sa = ggml_mul_mat(g, Vt, sc); // [hd,1,nh]
        sa = ggml_cont(g, ggml_permute(g, sa, 0, 2, 1, 3));
        sa = ggml_reshape_2d(g, sa, C, 1);
        sa = lin(g, sa, Lr.sa_o_w, Lr.sa_o_b);
        cur = ggml_add(g, cur, sa);
        // cross-attention against precomputed K/V
        ggml_tensor * ck = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nE);
        snprintf(nm, sizeof(nm), "ck_%d", i);
        ggml_set_name(ck, nm);
        ggml_set_input(ck);
        ggml_tensor * cv = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nE);
        snprintf(nm, sizeof(nm), "cv_%d", i);
        ggml_set_name(cv, nm);
        ggml_set_input(cv);
        h = ln0(g, cur, Lr.n1_w, Lr.n1_b);
        ggml_tensor * cq = lin(g, h, Lr.ca_q_w, Lr.ca_q_b);
        ggml_tensor * ca = mha(g, cq, ck, cv, nh, 1, nE, nullptr, scale);
        ca = lin(g, ca, Lr.ca_o_w, Lr.ca_o_b);
        cur = ggml_add(g, cur, ca);
        h = ln0(g, cur, Lr.n2_w, Lr.n2_b);
        ggml_tensor * ff = lin(g, h, Lr.ff0_w, Lr.ff0_b);
        ff = ggml_gelu_erf(g, ff);
        ff = lin(g, ff, Lr.ff3_w, Lr.ff3_b);
        cur = ggml_add(g, cur, ff);
    }
    ggml_tensor * normed = ln0(g, cur, ctx->final_w, ctx->final_b);
    ggml_tensor * logits = lin(g, normed, ctx->lm_w, ctx->lm_b); // [V,1]
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

static void decode_greedy_kv(transcoda_ocr_context * ctx) {
    const auto & hp = ctx->hp;
    const int C = hp.d_model, nl = hp.n_layers, V = hp.vocab_size;
    precompute_cross_kv(ctx);
    ctx->self_k_host.assign(nl, {});
    ctx->self_v_host.assign(nl, {});
    ctx->result.clear();
    int id = hp.bos_token;
    std::vector<int> seen = { hp.bos_token };
    for (int pos = 0; pos < hp.max_seq_len; pos++) {
        size_t meta = 32 * 1024 * 1024;
        std::vector<uint8_t> buf(meta);
        ggml_init_params ip = { meta, buf.data(), true };
        ggml_context * g = ggml_init(ip);
        ggml_cgraph * gf = build_decode_step(ctx, g, pos);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            ggml_free(g);
            break;
        }
        int32_t tok = id, pp = pos;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "step_token"), &tok, 0, sizeof(int32_t));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "step_pos"), &pp, 0, sizeof(int32_t));
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
            std::vector<float> kn(C), vn(C);
            snprintf(nm, sizeof(nm), "sk_%d", l);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, nm), kn.data(), 0, C * sizeof(float));
            snprintf(nm, sizeof(nm), "sv_%d", l);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, nm), vn.data(), 0, C * sizeof(float));
            ctx->self_k_host[l].insert(ctx->self_k_host[l].end(), kn.begin(), kn.end());
            ctx->self_v_host[l].insert(ctx->self_v_host[l].end(), vn.begin(), vn.end());
        }
        ggml_free(g);
        int best = argmax_with_penalty(logits.data(), V, seen, 1.1f);
        if (best == hp.eos_token) break;
        if (best >= 0 && best < (int)ctx->vocab.size()) {
            // Concatenate tokens directly: **kern structure lives in the literal
            // '\n'/'\t' vocab tokens, and tokens can contain '/' (e.g. "*M2/4"),
            // so no separator — this reconstructs the exact kern text.
            ctx->result += ctx->vocab[best];
        }
        id = best;
        seen.push_back(best);
    }
}

// ---------------------------------------------------------------------------
// Persistent (device-resident) KV-cache decode. Mathematically identical to
// decode_greedy_kv, but the cross K/V (constant per image) and the growing self
// K/V live in a backend buffer and are written IN-GRAPH — no per-step host
// round-trip (the host path re-uploaded ~48 MB/step). Default path; the host
// path stays available behind TRANSCODA_OCR_HOST_KV=1 for A/B.
// ---------------------------------------------------------------------------
static bool pk_alloc(transcoda_ocr_context * ctx, int nE) {
    const int C = ctx->hp.d_model, nl = ctx->hp.n_layers, max_seq = ctx->hp.max_seq_len;
    if (ctx->kv_buf && ctx->pk_nE == nE && ctx->pk_max_seq == max_seq) {
        ggml_backend_buffer_clear(ctx->kv_buf, 0);
        return true;
    }
    if (ctx->kv_buf) ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx) ggml_free(ctx->kv_ctx);
    ggml_init_params ip = { 8 * ggml_tensor_overhead() + 256, nullptr, true };
    ctx->kv_ctx = ggml_init(ip);
    ctx->pk_self_k = ggml_new_tensor_3d(ctx->kv_ctx, GGML_TYPE_F32, C, max_seq, nl);
    ctx->pk_self_v = ggml_new_tensor_3d(ctx->kv_ctx, GGML_TYPE_F32, C, max_seq, nl);
    ctx->pk_cross_k = ggml_new_tensor_3d(ctx->kv_ctx, GGML_TYPE_F32, C, nE, nl);
    ctx->pk_cross_v = ggml_new_tensor_3d(ctx->kv_ctx, GGML_TYPE_F32, C, nE, nl);
    ctx->kv_buf = ggml_backend_alloc_ctx_tensors(ctx->kv_ctx, ctx->backend);
    if (!ctx->kv_buf) return false;
    ggml_backend_buffer_clear(ctx->kv_buf, 0);
    ctx->pk_nE = nE;
    ctx->pk_max_seq = max_seq;
    return true;
}

// compute cross K (from mem_pos) / V (from mem_raw) once into the persistent buffer
static void pk_precompute_cross(transcoda_ocr_context * ctx) {
    const int C = ctx->hp.d_model, nl = ctx->hp.n_layers, nE = ctx->n_enc;
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
        ggml_tensor * ck = lin(g, mk, L.ca_k_w, L.ca_k_b); // [C,nE]
        ggml_tensor * kdst =
            ggml_view_2d(g, ctx->pk_cross_k, C, nE, ctx->pk_cross_k->nb[1], (size_t)i * ctx->pk_cross_k->nb[2]);
        ggml_build_forward_expand(gf, ggml_cpy(g, ck, kdst));
        ggml_tensor * cv = lin(g, mv, L.ca_v_w, L.ca_v_b);
        ggml_tensor * vdst =
            ggml_view_2d(g, ctx->pk_cross_v, C, nE, ctx->pk_cross_v->nb[1], (size_t)i * ctx->pk_cross_v->nb[2]);
        ggml_build_forward_expand(gf, ggml_cpy(g, cv, vdst));
    }
    ggml_backend_sched_reset(ctx->sched);
    ggml_backend_sched_alloc_graph(ctx->sched, gf);
    ggml_backend_tensor_set(mk, ctx->mem_pos.data(), 0, ctx->mem_pos.size() * sizeof(float));
    ggml_backend_tensor_set(mv, ctx->mem_raw.data(), 0, ctx->mem_raw.size() * sizeof(float));
    ggml_backend_sched_graph_compute(ctx->sched, gf);
    ggml_free(g);
}

// One decode step reading/writing the persistent cache. Writes this token's
// RoPE'd self-K + V into the cache at column `pos`, then attends over [0..pos].
static ggml_cgraph * build_step_persistent(transcoda_ocr_context * ctx, ggml_context * g, int pos) {
    const auto & hp = ctx->hp;
    const int C = hp.d_model, nh = hp.n_heads, hd = C / nh, nE = ctx->n_enc, Lk = pos + 1;
    float scale = 1.0f / std::sqrt((float)hd);
    ggml_cgraph * gf = ggml_new_graph_custom(g, hp.n_layers * 96 + 256, false);

    ggml_tensor * token = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(token, "step_token");
    ggml_set_input(token);
    ggml_tensor * ppos = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(ppos, "step_pos");
    ggml_set_input(ppos);
    ggml_tensor * cur = ggml_get_rows(g, ctx->tok_embed, token); // [C,1]

    for (int i = 0; i < hp.n_layers; i++) {
        auto & Lr = ctx->dec[i];
        ggml_tensor * h = ln0(g, cur, Lr.n0_w, Lr.n0_b);
        ggml_tensor * qkv = lin(g, h, Lr.qkv_w, Lr.qkv_b); // [3C,1]
        size_t es = ggml_element_size(qkv);
        ggml_tensor * q = ggml_cont(g, ggml_view_2d(g, qkv, C, 1, qkv->nb[1], 0));
        ggml_tensor * knew = ggml_cont(g, ggml_view_2d(g, qkv, C, 1, qkv->nb[1], (size_t)C * es));
        ggml_tensor * vnew = ggml_cont(g, ggml_view_2d(g, qkv, C, 1, qkv->nb[1], (size_t)2 * C * es));
        ggml_tensor * Qr = ggml_rope_ext(g, ggml_reshape_3d(g, q, hd, nh, 1), ppos, nullptr, hd, GGML_ROPE_TYPE_NORMAL,
                                         0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * Kr = ggml_rope_ext(g, ggml_reshape_3d(g, knew, hd, nh, 1), ppos, nullptr, hd,
                                         GGML_ROPE_TYPE_NORMAL, 0, hp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        ggml_tensor * krow = ggml_cont(g, ggml_reshape_2d(g, Kr, C, 1));
        // write this token's roped-K and V into the self cache at column `pos`
        size_t k_ls = ctx->pk_self_k->nb[2], k_cs = ctx->pk_self_k->nb[1];
        ggml_tensor * kdst = ggml_view_2d(g, ctx->pk_self_k, C, 1, k_cs, (size_t)i * k_ls + (size_t)pos * k_cs);
        ggml_build_forward_expand(gf, ggml_cpy(g, krow, kdst));
        size_t v_ls = ctx->pk_self_v->nb[2], v_cs = ctx->pk_self_v->nb[1];
        ggml_tensor * vdst = ggml_view_2d(g, ctx->pk_self_v, C, 1, v_cs, (size_t)i * v_ls + (size_t)pos * v_cs);
        ggml_build_forward_expand(gf, ggml_cpy(g, vnew, vdst));
        // read full self cache [C, Lk] for this layer
        ggml_tensor * kfull = ggml_view_2d(g, ctx->pk_self_k, C, Lk, k_cs, (size_t)i * k_ls);
        ggml_tensor * vfull = ggml_view_2d(g, ctx->pk_self_v, C, Lk, v_cs, (size_t)i * v_ls);
        // self-attention: Q(roped) [hd,1,nh] vs cache
        ggml_tensor * Qp = ggml_cont(g, ggml_permute(g, Qr, 0, 2, 1, 3)); // [hd,1,nh]
        ggml_tensor * Kp = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, kfull, hd, nh, Lk), 0, 2, 1, 3));
        ggml_tensor * Vp = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, vfull, hd, nh, Lk), 0, 2, 1, 3));
        ggml_tensor * sc = ggml_mul_mat(g, Kp, Qp); // [Lk,1,nh]
        sc = ggml_soft_max_ext(g, sc, nullptr, scale, 0.0f);
        ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, Vp, 1, 0, 2, 3));
        ggml_tensor * sa = ggml_mul_mat(g, Vt, sc); // [hd,1,nh]
        sa = ggml_cont(g, ggml_permute(g, sa, 0, 2, 1, 3));
        sa = ggml_reshape_2d(g, sa, C, 1);
        sa = lin(g, sa, Lr.sa_o_w, Lr.sa_o_b);
        cur = ggml_add(g, cur, sa);
        // cross-attention against the persistent cross cache
        ggml_tensor * ck =
            ggml_view_2d(g, ctx->pk_cross_k, C, nE, ctx->pk_cross_k->nb[1], (size_t)i * ctx->pk_cross_k->nb[2]);
        ggml_tensor * cv =
            ggml_view_2d(g, ctx->pk_cross_v, C, nE, ctx->pk_cross_v->nb[1], (size_t)i * ctx->pk_cross_v->nb[2]);
        h = ln0(g, cur, Lr.n1_w, Lr.n1_b);
        ggml_tensor * cq = lin(g, h, Lr.ca_q_w, Lr.ca_q_b);
        ggml_tensor * ca = mha(g, cq, ck, cv, nh, 1, nE, nullptr, scale);
        ca = lin(g, ca, Lr.ca_o_w, Lr.ca_o_b);
        cur = ggml_add(g, cur, ca);
        h = ln0(g, cur, Lr.n2_w, Lr.n2_b);
        ggml_tensor * ff = lin(g, h, Lr.ff0_w, Lr.ff0_b);
        ff = ggml_gelu_erf(g, ff);
        ff = lin(g, ff, Lr.ff3_w, Lr.ff3_b);
        cur = ggml_add(g, cur, ff);
    }
    ggml_tensor * normed = ln0(g, cur, ctx->final_w, ctx->final_b);
    ggml_tensor * logits = lin(g, normed, ctx->lm_w, ctx->lm_b); // [V,1]
    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);
    return gf;
}

static void decode_greedy_persistent(transcoda_ocr_context * ctx) {
    const auto & hp = ctx->hp;
    const int V = hp.vocab_size;
    if (!pk_alloc(ctx, ctx->n_enc)) {
        decode_greedy_kv(ctx);
        return;
    }
    pk_precompute_cross(ctx);
    if (ctx->compute_meta.size() < 24 * 1024 * 1024) ctx->compute_meta.resize(24 * 1024 * 1024);
    ctx->result.clear();
    int id = hp.bos_token;
    std::vector<int> seen = { hp.bos_token };
    for (int pos = 0; pos < hp.max_seq_len; pos++) {
        ggml_init_params ip = { ctx->compute_meta.size(), ctx->compute_meta.data(), true };
        ggml_context * g = ggml_init(ip);
        ggml_cgraph * gf = build_step_persistent(ctx, g, pos);
        ggml_backend_sched_reset(ctx->sched);
        if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
            ggml_free(g);
            break;
        }
        int32_t tok = id, pp = pos;
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "step_token"), &tok, 0, sizeof(int32_t));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "step_pos"), &pp, 0, sizeof(int32_t));
        ggml_backend_sched_graph_compute(ctx->sched, gf);
        std::vector<float> logits(V);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logits"), logits.data(), 0, V * sizeof(float));
        ggml_free(g);
        int best = argmax_with_penalty(logits.data(), V, seen, 1.1f);
        if (best == hp.eos_token) break;
        if (best >= 0 && best < (int)ctx->vocab.size()) ctx->result += ctx->vocab[best];
        id = best;
        seen.push_back(best);
    }
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------
transcoda_ocr_context * transcoda_ocr_init(const char * model_path, int n_threads) {
    auto ctx = new transcoda_ocr_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    gguf_context * gc = core_gguf::open_metadata(model_path);
    if (!gc) {
        fprintf(stderr, "transcoda_ocr: can't open %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    auto & hp = ctx->hp;
    hp.d_model = core_gguf::kv_u32(gc, "transcoda.d_model", 512);
    hp.n_layers = core_gguf::kv_u32(gc, "transcoda.n_layers", 8);
    hp.n_heads = core_gguf::kv_u32(gc, "transcoda.n_heads", 8);
    hp.dim_ff = core_gguf::kv_u32(gc, "transcoda.dim_ff", 1024);
    hp.vocab_size = core_gguf::kv_u32(gc, "transcoda.vocab_size", 3000);
    hp.max_seq_len = core_gguf::kv_u32(gc, "transcoda.max_seq_len", 2048);
    hp.rope_theta = core_gguf::kv_f32(gc, "transcoda.rope_theta", 10000.0f);
    hp.enc_num_stages = core_gguf::kv_u32(gc, "transcoda.enc_num_stages", 4);
    hp.enc_num_channels = core_gguf::kv_u32(gc, "transcoda.enc_num_channels", 3);
    hp.enc_stem_kernel = core_gguf::kv_u32(gc, "transcoda.enc_stem_kernel", 4);
    hp.enc_reduction = core_gguf::kv_u32(gc, "transcoda.enc_reduction", 32);
    hp.fixed_h = core_gguf::kv_u32(gc, "transcoda.fixed_height", 1485);
    hp.fixed_w = core_gguf::kv_u32(gc, "transcoda.fixed_width", 1050);
    hp.bos_token = core_gguf::kv_u32(gc, "transcoda.bos_token_id", 1);
    hp.eos_token = core_gguf::kv_u32(gc, "transcoda.eos_token_id", 2);
    hp.pad_token = core_gguf::kv_u32(gc, "transcoda.pad_token_id", 0);
    {
        auto mean = core_gguf::kv_f32_array(gc, "transcoda.image_mean");
        auto std = core_gguf::kv_f32_array(gc, "transcoda.image_std");
        hp.mean0 = mean.size() > 0 ? mean[0] : 0.5f;
        hp.mean1 = mean.size() > 1 ? mean[1] : 0.5f;
        hp.mean2 = mean.size() > 2 ? mean[2] : 0.5f;
        hp.std0 = std.size() > 0 ? std[0] : 0.5f;
        hp.std1 = std.size() > 1 ? std[1] : 0.5f;
        hp.std2 = std.size() > 2 ? std[2] : 0.5f;
    }
    ctx->vocab = core_gguf::kv_str_array(gc, "tokenizer.tokens");
    core_gguf::free_metadata(gc);

    ctx->stage_sizes = { 96, 192, 384, 768 };
    ctx->stage_depths = { 3, 3, 9, 3 };

    fprintf(stderr, "transcoda_ocr: dec %dL/%dH/d%d ff%d vocab=%d(%zu), enc /%d %dch, img %dx%d\n", hp.n_layers,
            hp.n_heads, hp.d_model, hp.dim_ff, hp.vocab_size, ctx->vocab.size(), hp.enc_reduction, hp.enc_num_channels,
            hp.fixed_h, hp.fixed_w);

    bool force_cpu = (getenv("TRANSCODA_OCR_FORCE_CPU") && atoi(getenv("TRANSCODA_OCR_FORCE_CPU")));
    ctx->backend = force_cpu ? ggml_backend_cpu_init() : crispasr_init_gpu_backend();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx->backend)) ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    ctx->backend_cpu = ggml_backend_is_cpu(ctx->backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);

    if (!core_gguf::load_weights(model_path, ctx->backend, "transcoda_ocr", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    std::vector<ggml_backend_t> backends = { ctx->backend };
    if (ctx->backend_cpu) backends.push_back(ctx->backend_cpu);
    ctx->sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), 16384, false, false);

    map_tensors(ctx);
    fprintf(stderr, "transcoda_ocr: loaded %zu tensors, init complete\n", ctx->wl.tensors.size());
    return ctx;
}

void transcoda_ocr_free(transcoda_ocr_context * ctx) {
    if (!ctx) return;
    if (ctx->kv_buf) ggml_backend_buffer_free(ctx->kv_buf);
    if (ctx->kv_ctx) ggml_free(ctx->kv_ctx);
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

const transcoda_ocr_hparams * transcoda_ocr_get_hparams(const transcoda_ocr_context * ctx) {
    return ctx ? &ctx->hp : nullptr;
}

// ---------------------------------------------------------------------------
// Preprocessing + recognize
// ---------------------------------------------------------------------------
// Reproduce the model card's preprocessing exactly:
//   resize to width fixed_w preserving aspect (bilinear) → top-crop or white-pad
//   (255) the bottom to fixed_h → (x/255-0.5)/0.5. Output [3,H,W] (c,h,w order).
static std::vector<float> preprocess(const transcoda_ocr_context * ctx, const uint8_t * data, int w, int h, int ch,
                                     int * outW, int * outH) {
    const int TW = ctx->hp.fixed_w, TH = ctx->hp.fixed_h;
    int nh = std::max(1, (int)((double)h * ((double)TW / w)));
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    int nk = ch >= 3 ? 3 : 1;
    // bilinear resize (w,h)->(TW,nh) into RGB uint8-rounded, then crop/pad to TH.
    std::vector<float> out((size_t)3 * TH * TW, 0.0f); // [c][y][x]
    float sx = (float)w / TW, sy = (float)h / nh;
    const float mean[3] = { ctx->hp.mean0, ctx->hp.mean1, ctx->hp.mean2 };
    const float stdv[3] = { ctx->hp.std0, ctx->hp.std1, ctx->hp.std2 };
    for (int y = 0; y < TH; y++) {
        for (int x = 0; x < TW; x++) {
            float rgb[3];
            if (y < nh) {
                float fy = (y + 0.5f) * sy - 0.5f;
                int y0 = (int)std::floor(fy);
                float wy = fy - y0;
                int y0c = clampi(y0, 0, h - 1), y1c = clampi(y0 + 1, 0, h - 1);
                float fx = (x + 0.5f) * sx - 0.5f;
                int x0 = (int)std::floor(fx);
                float wx = fx - x0;
                int x0c = clampi(x0, 0, w - 1), x1c = clampi(x0 + 1, 0, w - 1);
                for (int k = 0; k < 3; k++) {
                    int src = nk >= 3 ? k : 0;
                    float p00 = data[((size_t)y0c * w + x0c) * ch + src];
                    float p01 = data[((size_t)y0c * w + x1c) * ch + src];
                    float p10 = data[((size_t)y1c * w + x0c) * ch + src];
                    float p11 = data[((size_t)y1c * w + x1c) * ch + src];
                    float top = p00 + wx * (p01 - p00), bot = p10 + wx * (p11 - p10);
                    rgb[k] = top + wy * (bot - top);
                }
            } else {
                rgb[0] = rgb[1] = rgb[2] = 255.0f; // white bottom pad
            }
            for (int k = 0; k < 3; k++) out[((size_t)k * TH + y) * TW + x] = (rgb[k] / 255.0f - mean[k]) / stdv[k];
        }
    }
    *outW = TW;
    *outH = TH;
    return out;
}

const char * transcoda_ocr_recognize_raw(transcoda_ocr_context * ctx, const uint8_t * data, int w, int h, int ch,
                                         int * out_len) {
    if (!data || w <= 0 || h <= 0 || ch <= 0) return nullptr;
    int W = 0, H = 0;
    std::vector<float> px = preprocess(ctx, data, w, h, ch, &W, &H);
    if (const char * dp = getenv("TRANSCODA_OCR_DUMP_PX")) {
        FILE * f = fopen(dp, "wb");
        if (f) {
            fwrite(px.data(), sizeof(float), px.size(), f);
            fclose(f);
            fprintf(stderr, "transcoda_ocr: dumped %zu px floats [%dx%d] to %s\n", px.size(), W, H, dp);
        }
    }
    if (!run_encoder(ctx, px.data(), W, H, nullptr, nullptr)) return nullptr;
    if (getenv("TRANSCODA_OCR_FULL_DECODE"))
        decode_greedy_full(ctx); // O(L²) reference (no KV cache)
    else if (getenv("TRANSCODA_OCR_HOST_KV"))
        decode_greedy_kv(ctx); // host-shuttled KV cache (A/B reference)
    else
        decode_greedy_persistent(ctx); // device-resident KV cache (default, fast)
    if (out_len) *out_len = (int)ctx->result.size();
    return ctx->result.c_str();
}

const char * transcoda_ocr_recognize_file(transcoda_ocr_context * ctx, const char * image_path, int * out_len) {
    int w = 0, h = 0, c = 0;
    stbi_uc * data = stbi_load(image_path, &w, &h, &c, 3);
    if (!data) return nullptr;
    const char * r = transcoda_ocr_recognize_raw(ctx, data, w, h, 3, out_len);
    stbi_image_free(data);
    return r;
}

// ---------------------------------------------------------------------------
// Per-stage parity harness
// ---------------------------------------------------------------------------
int transcoda_ocr_run_diff(transcoda_ocr_context * ctx, const char * ref_path) {
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) return 1;

    auto inshape = ref.shape("input_tensor"); // ggml ne = [W,H,C_in]
    if (inshape.size() < 2) {
        fprintf(stderr, "transcoda_diff: no input_tensor\n");
        return 1;
    }
    int W = (int)inshape[0], H = (int)inshape[1];
    auto in = ref.get_f32("input_tensor");
    fprintf(stderr, "transcoda_diff: input W=%d H=%d\n", W, H);

    int fails = 0;
    auto report = [&](const char * nm, const crispembed_diff::Report & r) {
        bool pass = r.is_pass();
        fprintf(stderr, "[tc-diff] %-12s cos_min=%.6f cos_mean=%.6f max_abs=%.2e %s\n", nm, r.cos_min, r.cos_mean,
                r.max_abs, pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    };

    // ---- encoder ----
    ggml_context * eg = nullptr;
    ggml_cgraph * egf = nullptr;
    if (!run_encoder(ctx, in.first, W, H, &eg, &egf)) return 1;
    if (ref.has("enc_grid")) {
        ggml_tensor * t = ggml_graph_get_tensor(egf, "enc_grid");
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        report("enc_grid", ref.compare("enc_grid", buf.data(), buf.size(), 0));
    }
    if (ref.has("enc_raw")) report("enc_raw", ref.compare("enc_raw", ctx->mem_raw.data(), ctx->mem_raw.size(), 0));
    if (ref.has("enc_pos")) report("enc_pos", ref.compare("enc_pos", ctx->mem_pos.data(), ctx->mem_pos.size(), 0));
    ggml_free(eg);

    // ---- decoder (teacher-forced on ref token_ids) ----
    auto tid = ref.get_f32("token_ids");
    std::vector<int32_t> ids(tid.second);
    for (size_t i = 0; i < tid.second; i++) ids[i] = (int32_t)llround(tid.first[i]);
    int L = (int)ids.size();
    if (L == 0) {
        fprintf(stderr, "transcoda_diff: no token_ids — skipping decoder\n");
        return fails ? 1 : 0;
    }
    fprintf(stderr, "transcoda_diff: teacher-forcing %d tokens\n", L);
    ggml_context * dg = nullptr;
    ggml_cgraph * dgf = nullptr;
    if (!run_decoder(ctx, ids, &dg, &dgf)) return 1;
    auto readcmp = [&](const char * nm) {
        if (!ref.has(nm)) return;
        ggml_tensor * t = ggml_graph_get_tensor(dgf, nm);
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        report(nm, ref.compare(nm, buf.data(), buf.size(), 0));
    };
    readcmp("dec_tok_emb");
    for (int i = 0; i < ctx->hp.n_layers; i++) {
        char nm[24];
        snprintf(nm, sizeof(nm), "dec_block%d", i);
        readcmp(nm);
    }
    readcmp("logits");
    // Per-position argmax agreement: does my logits[i] top-1 == the oracle's next
    // token ids[i+1]? Near-tie flips (F32 op-order) show up here; a high rate with
    // only a few flips confirms the port is correct despite greedy AR divergence.
    if (ref.has("logits")) {
        ggml_tensor * lt = ggml_graph_get_tensor(dgf, "logits");
        int V = ctx->hp.vocab_size;
        std::vector<float> lg(ggml_nelements(lt));
        ggml_backend_tensor_get(lt, lg.data(), 0, lg.size() * sizeof(float));
        int agree = 0, cmp = 0;
        for (int i = 0; i + 1 < L; i++) {
            int best = 0;
            float bv = lg[(size_t)i * V];
            for (int v = 1; v < V; v++)
                if (lg[(size_t)i * V + v] > bv) {
                    bv = lg[(size_t)i * V + v];
                    best = v;
                }
            cmp++;
            if (best == ids[i + 1]) agree++;
        }
        fprintf(stderr, "[tc-diff] argmax agreement vs oracle next-token: %d/%d = %.4f\n", agree, cmp,
                cmp ? (double)agree / cmp : 1.0);
    }
    ggml_free(dg);

    fprintf(stderr, "transcoda_diff: %s (%d stage failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
