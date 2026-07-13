// tromr_ocr.cpp — Polyphonic-TrOMR optical music recognition via ggml graphs.
//
// Blueprint: github.com/NetEase/Polyphonic-TrOMR (Apache-2.0), read line-by-line.
//   Encoder : timm hybrid ViT — ResNetV2 backbone (StdConv2dSame + GroupNorm,
//             layers [2,3,7], 1→64→256→512→1024, /16) → HybridEmbed 1×1 proj
//             (1024→256) → ViT (depth 4, 8 heads, dim 256, cls token, custom 2D
//             pos-index). The 40 StdConv backbone kernels are pre-standardized by
//             the converter (timm F.batch_norm, eps 1e-6) so we run plain convs.
//   Decoder : x_transformers Decoder — 12 sublayers = ('a','c','f')×4
//             (self-attn → cross-attn → GLU-FF). attn-on-attn to_out = SIGLU,
//             ff = GEGLU. Input = rhythm+pitch+lift+abs-pos embeddings (pos scaled
//             by d^-0.5, token embs unscaled). 4 heads (rhythm/pitch/lift/note),
//             autoregressive over the 3 streams (argmax, stop on rhythm==eos).
//
// Loaded from a GGUF produced by models/convert-tromr-to-gguf.py (arch "tromr_ocr").
//
// KEY POINTS (validated vs tools/dump_tromr_reference.py, CPU only):
//   • SAME padding is asymmetric — computed per input size (ggml_pad_ext(W,H)).
//   • pos_emb IS scaled by 256^-0.5; token embeddings are NOT.
//   • ViT head_dim = 256/8 = 32 → scale 32^-0.5 (NOT 64^-0.5); decoder head_dim
//     = 512/8 = 64 → scale 64^-0.5.
//   • ViT/decoder self-attn masks: ViT none, decoder self causal, cross none.
//   • GroupNorm 32 groups eps 1e-5; ViT/ff/proj GELU = erf.
//   • dec_block{i} in the ref archive is the sublayer block output PRE-residual.

#include "tromr_ocr.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gpu_backend_pref.h"
#include "core/gguf_loader.h"
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
// Structs
// ---------------------------------------------------------------------------
struct tromr_bottleneck {
    ggml_tensor *c1_w, *c2_w, *c3_w;                                   // conv weights (1×1, 3×3, 1×1)
    ggml_tensor *n1_w, *n1_b, *n2_w, *n2_b, *n3_w, *n3_b;              // GroupNorm affine
    ggml_tensor *ds_w = nullptr, *ds_n_w = nullptr, *ds_n_b = nullptr; // downsample (block 0)
    int mid, out, in;                                                  // channel counts
};
struct tromr_bb_stage {
    std::vector<tromr_bottleneck> blocks;
    int stride;
};
struct tromr_vit_block {
    ggml_tensor *n1_w, *n1_b, *n2_w, *n2_b;
    ggml_tensor *qkv_w, *qkv_b, *proj_w, *proj_b;
    ggml_tensor *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};
struct tromr_dec_layer {
    char kind;                                  // 'a' self-attn, 'c' cross-attn, 'f' ff
    ggml_tensor *ln_w, *ln_b;                   // pre-norm (.0.0)
    ggml_tensor *q_w, *k_w, *v_w, *out_w;       // attn (no bias)
    ggml_tensor *ff0_w, *ff0_b, *ff3_w, *ff3_b; // ff
};

struct tromr_ocr_context {
    tromr_ocr_hparams hp;
    // encoder backbone
    ggml_tensor *stem_w, *stem_n_w, *stem_n_b;
    std::vector<tromr_bb_stage> stages;
    ggml_tensor *proj_w, *proj_b;
    // encoder ViT
    ggml_tensor *cls_token, *pos_embed, *enc_norm_w, *enc_norm_b;
    std::vector<tromr_vit_block> vit;
    // decoder
    ggml_tensor *rhythm_emb, *pitch_emb, *lift_emb, *pos_emb;
    ggml_tensor *dec_norm_w, *dec_norm_b;
    ggml_tensor *lg_r_w, *lg_r_b, *lg_p_w, *lg_p_b, *lg_l_w, *lg_l_b, *lg_n_w, *lg_n_b;
    std::vector<tromr_dec_layer> dec;

    std::vector<std::string> rhythm_tok, pitch_tok, lift_tok;
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr, backend_cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    int n_threads = 4;
    std::string result;

    // cached ViT memory (host) from the last run_encoder: [n_ctx * C], token-major
    std::vector<float> enc_ctx;
    int n_ctx = 0;
};

// ---------------------------------------------------------------------------
static ggml_tensor * F(const std::unordered_map<std::string, ggml_tensor *> & m, const std::string & n) {
    auto it = m.find(n);
    return it != m.end() ? it->second : nullptr;
}

static void map_tensors(tromr_ocr_context * ctx) {
    const auto & m = ctx->wl.tensors;
    char b[256];
    const char * BB = "enc.bb"; // converter shortens this prefix (GGML_MAX_NAME=64)

    ctx->stem_w = F(m, std::string(BB) + ".stem.conv.weight");
    ctx->stem_n_w = F(m, std::string(BB) + ".stem.norm.weight");
    ctx->stem_n_b = F(m, std::string(BB) + ".stem.norm.bias");

    // 3 stages: [2 blk out256 mid64 s1], [3 blk out512 mid128 s2], [7 blk out1024 mid256 s2]
    const int n_blk[3] = { 2, 3, 7 };
    const int out_c[3] = { 256, 512, 1024 };
    const int mid_c[3] = { 64, 128, 256 };
    const int strd[3] = { 1, 2, 2 };
    int in_c = 64; // stem output channels
    ctx->stages.resize(3);
    for (int s = 0; s < 3; s++) {
        ctx->stages[s].stride = strd[s];
        ctx->stages[s].blocks.resize(n_blk[s]);
        for (int bi = 0; bi < n_blk[s]; bi++) {
            auto & B = ctx->stages[s].blocks[bi];
            auto G = [&](const char * suf) {
                snprintf(b, sizeof(b), "%s.stages.%d.blocks.%d.%s", BB, s, bi, suf);
                return F(m, b);
            };
            B.c1_w = G("conv1.weight");
            B.c2_w = G("conv2.weight");
            B.c3_w = G("conv3.weight");
            B.n1_w = G("norm1.weight");
            B.n1_b = G("norm1.bias");
            B.n2_w = G("norm2.weight");
            B.n2_b = G("norm2.bias");
            B.n3_w = G("norm3.weight");
            B.n3_b = G("norm3.bias");
            B.in = (bi == 0) ? in_c : out_c[s];
            B.mid = mid_c[s];
            B.out = out_c[s];
            if (bi == 0) {
                B.ds_w = G("downsample.conv.weight");
                B.ds_n_w = G("downsample.norm.weight");
                B.ds_n_b = G("downsample.norm.bias");
            }
        }
        in_c = out_c[s];
    }
    ctx->proj_w = F(m, "encoder.patch_embed.proj.weight");
    ctx->proj_b = F(m, "encoder.patch_embed.proj.bias");

    ctx->cls_token = F(m, "encoder.cls_token");
    ctx->pos_embed = F(m, "encoder.pos_embed");
    ctx->enc_norm_w = F(m, "encoder.norm.weight");
    ctx->enc_norm_b = F(m, "encoder.norm.bias");
    ctx->vit.resize(ctx->hp.encoder_depth);
    for (int i = 0; i < ctx->hp.encoder_depth; i++) {
        auto & V = ctx->vit[i];
        auto E = [&](const char * suf) {
            snprintf(b, sizeof(b), "encoder.blocks.%d.%s", i, suf);
            return F(m, b);
        };
        V.n1_w = E("norm1.weight");
        V.n1_b = E("norm1.bias");
        V.n2_w = E("norm2.weight");
        V.n2_b = E("norm2.bias");
        V.qkv_w = E("attn.qkv.weight");
        V.qkv_b = E("attn.qkv.bias");
        V.proj_w = E("attn.proj.weight");
        V.proj_b = E("attn.proj.bias");
        V.fc1_w = E("mlp.fc1.weight");
        V.fc1_b = E("mlp.fc1.bias");
        V.fc2_w = E("mlp.fc2.weight");
        V.fc2_b = E("mlp.fc2.bias");
    }

    ctx->rhythm_emb = F(m, "decoder.net.rhythm_emb.emb.weight");
    ctx->pitch_emb = F(m, "decoder.net.pitch_emb.emb.weight");
    ctx->lift_emb = F(m, "decoder.net.lift_emb.emb.weight");
    ctx->pos_emb = F(m, "decoder.net.pos_emb.emb.weight");
    ctx->dec_norm_w = F(m, "decoder.net.norm.weight");
    ctx->dec_norm_b = F(m, "decoder.net.norm.bias");
    ctx->lg_r_w = F(m, "decoder.net.to_logits_rhythm.weight");
    ctx->lg_r_b = F(m, "decoder.net.to_logits_rhythm.bias");
    ctx->lg_p_w = F(m, "decoder.net.to_logits_pitch.weight");
    ctx->lg_p_b = F(m, "decoder.net.to_logits_pitch.bias");
    ctx->lg_l_w = F(m, "decoder.net.to_logits_lift.weight");
    ctx->lg_l_b = F(m, "decoder.net.to_logits_lift.bias");
    ctx->lg_n_w = F(m, "decoder.net.to_logits_note.weight");
    ctx->lg_n_b = F(m, "decoder.net.to_logits_note.bias");

    // 12 sublayers, kind = ('a','c','f') cycling
    const char kinds[3] = { 'a', 'c', 'f' };
    int n_sub = ctx->hp.decoder_depth * 3;
    ctx->dec.resize(n_sub);
    for (int i = 0; i < n_sub; i++) {
        auto & L = ctx->dec[i];
        L.kind = kinds[i % 3];
        auto D = [&](const char * suf) {
            snprintf(b, sizeof(b), "decoder.net.attn_layers.layers.%d.%s", i, suf);
            return F(m, b);
        };
        L.ln_w = D("0.0.weight");
        L.ln_b = D("0.0.bias");
        if (L.kind == 'f') {
            L.ff0_w = D("1.net.0.proj.weight");
            L.ff0_b = D("1.net.0.proj.bias");
            L.ff3_w = D("1.net.3.weight");
            L.ff3_b = D("1.net.3.bias");
        } else {
            L.q_w = D("1.to_q.weight");
            L.k_w = D("1.to_k.weight");
            L.v_w = D("1.to_v.weight");
            L.out_w = D("1.to_out.0.weight");
        }
    }
}

// ---------------------------------------------------------------------------
// graph helpers
// ---------------------------------------------------------------------------
static ggml_tensor * f32(ggml_context * g, ggml_tensor * t) {
    return (!t || t->type == GGML_TYPE_F32) ? t : ggml_cast(g, t, GGML_TYPE_F32);
}
// LayerNorm over ne[0] (channels-last), weight + bias.
static ggml_tensor * ln(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * bi, float eps) {
    x = ggml_norm(g, x, eps);
    x = ggml_mul(g, x, f32(g, w));
    if (bi) x = ggml_add(g, x, f32(g, bi));
    return x;
}
// Linear: W (ggml ne [in,out]), x [in,...] → [out,...]
static ggml_tensor * lin(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * bi) {
    x = ggml_mul_mat(g, w, x);
    if (bi) x = ggml_add(g, x, f32(g, bi));
    return x;
}
// per-channel bias for a conv map [W,H,OC,N]: bias[OC] → [1,1,OC,1]
static ggml_tensor * add_cbias(ggml_context * g, ggml_tensor * x, ggml_tensor * bias) {
    return ggml_add(g, x, ggml_reshape_4d(g, f32(g, bias), 1, 1, bias->ne[0], 1));
}
// GroupNorm (32 groups, eps 1e-5) + per-channel affine on a map [W,H,C,N].
static ggml_tensor * gnorm(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * bi) {
    int C = x->ne[2];
    x = ggml_group_norm(g, x, 32, 1e-5f);
    x = ggml_mul(g, x, ggml_reshape_4d(g, f32(g, w), 1, 1, C, 1));
    x = ggml_add(g, x, ggml_reshape_4d(g, f32(g, bi), 1, 1, C, 1));
    return x;
}
// prepare a conv kernel: reshape flattened-quantized 2D → 4D, cast to F16.
static ggml_tensor * prep_conv(ggml_context * g, ggml_tensor * a, int kw, int kh, int ic, int oc) {
    if (ggml_n_dims(a) != 4) a = ggml_reshape_4d(g, a, kw, kh, ic, oc);
    return ggml_cast(g, a, GGML_TYPE_F16);
}
// SAME padding (asymmetric) for kernel k stride s over input `in`: returns (lp,rp).
static void same_pad(int in, int k, int s, int * lp, int * rp) {
    int out = (in + s - 1) / s;
    int tot = std::max((out - 1) * s + k - in, 0);
    *lp = tot / 2;
    *rp = tot - *lp;
}
// StdConv2dSame: SAME-pad then plain conv. x=[W,H,IC,N], w kernel (already standardized).
static ggml_tensor * conv_same(ggml_context * g, ggml_tensor * w, ggml_tensor * x, int k, int s, int ic, int oc) {
    int lpw, rpw, lph, rph;
    same_pad(x->ne[0], k, s, &lpw, &rpw);
    same_pad(x->ne[1], k, s, &lph, &rph);
    if (lpw || rpw || lph || rph) x = ggml_pad_ext(g, x, lpw, rpw, lph, rph, 0, 0, 0, 0);
    ggml_tensor * kf = prep_conv(g, w, k, k, ic, oc);
    return ggml_conv_2d(g, kf, x, s, s, 0, 0, 1, 1);
}
// flatten map [W,H,C,1] → tokens [C, W*H] (h-major, w-minor; matches flatten(2).T)
static ggml_tensor * to_tokens(ggml_context * g, ggml_tensor * map) {
    int W = map->ne[0], H = map->ne[1], C = map->ne[2];
    ggml_tensor * t = ggml_cont(g, ggml_permute(g, map, 1, 2, 0, 3)); // [C,W,H,N]
    return ggml_reshape_2d(g, t, C, W * H);
}
// scaled multi-head attention (returns concat-heads [inner, Lq], no out_proj).
static ggml_tensor * attn(ggml_context * g, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, int nh, float scale,
                          ggml_tensor * mask) {
    int inner = q->ne[0], Lq = q->ne[1], Lk = k->ne[1], hd = inner / nh;
    ggml_tensor * Q = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, q, hd, nh, Lq), 0, 2, 1, 3)); // [hd,Lq,nh]
    ggml_tensor * K = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, k, hd, nh, Lk), 0, 2, 1, 3)); // [hd,Lk,nh]
    ggml_tensor * V = ggml_cont(g, ggml_permute(g, ggml_reshape_3d(g, v, hd, nh, Lk), 0, 2, 1, 3)); // [hd,Lk,nh]
    ggml_tensor * scores = ggml_mul_mat(g, K, Q);                                                   // [Lk,Lq,nh]
    scores = ggml_soft_max_ext(g, scores, mask, scale, 0.0f);
    ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3)); // [Lk,hd,nh]
    ggml_tensor * a = ggml_mul_mat(g, Vt, scores);                   // [hd,Lq,nh]
    a = ggml_cont(g, ggml_permute(g, a, 0, 2, 1, 3));                // [hd,nh,Lq]
    return ggml_reshape_2d(g, a, inner, Lq);
}
// GLU: split ne[0] into value(first half) + gate(second half); result = value * act(gate).
enum glu_act { GLU_SIGMOID, GLU_GELU };
static ggml_tensor * glu(ggml_context * g, ggml_tensor * x, glu_act act) {
    int half = x->ne[0] / 2, T = x->ne[1];
    ggml_tensor * value = ggml_cont(g, ggml_view_2d(g, x, half, T, x->nb[1], 0));
    ggml_tensor * gate = ggml_cont(g, ggml_view_2d(g, x, half, T, x->nb[1], (size_t)half * x->nb[0]));
    gate = (act == GLU_SIGMOID) ? ggml_sigmoid(g, gate) : ggml_gelu_erf(g, gate);
    return ggml_mul(g, value, gate);
}

// ---------------------------------------------------------------------------
// Encoder graph: input [W,H,1,1] → enc_backbone (ResNet map) + enc_context (ViT).
// ---------------------------------------------------------------------------
static ggml_cgraph * build_encoder(tromr_ocr_context * ctx, ggml_context * g, int W, int H) {
    const auto & hp = ctx->hp;
    ggml_cgraph * gf = ggml_new_graph_custom(g, 8192, false);
    const int Cenc = hp.encoder_dim;                          // 256
    const int nh = hp.encoder_heads;                          // 8
    const float scale = 1.0f / std::sqrt((float)(Cenc / nh)); // head_dim = 32 → 32^-0.5

    ggml_tensor * x = ggml_new_tensor_4d(g, GGML_TYPE_F32, W, H, 1, 1);
    ggml_set_name(x, "input");
    ggml_set_input(x);

    // ---- Stem: StdConv 1→64 k7 s2 SAME → GN(32,64)+ReLU → MaxPool k3 s2 SAME ----
    ggml_tensor * cur = conv_same(g, ctx->stem_w, x, 7, 2, 1, 64);
    cur = gnorm(g, cur, ctx->stem_n_w, ctx->stem_n_b);
    cur = ggml_relu(g, cur);
    { // MaxPool SAME (post-ReLU ≥0 → zero-pad == -inf-pad here)
        int lpw, rpw, lph, rph;
        same_pad(cur->ne[0], 3, 2, &lpw, &rpw);
        same_pad(cur->ne[1], 3, 2, &lph, &rph);
        if (lpw || rpw || lph || rph) cur = ggml_pad_ext(g, cur, lpw, rpw, lph, rph, 0, 0, 0, 0);
        cur = ggml_pool_2d(g, cur, GGML_OP_POOL_MAX, 3, 3, 2, 2, 0, 0);
    }

    // ---- 3 ResNetV2 stages of non-preact Bottleneck blocks ----
    for (auto & S : ctx->stages) {
        for (size_t bi = 0; bi < S.blocks.size(); bi++) {
            auto & B = S.blocks[bi];
            int s = (bi == 0) ? S.stride : 1; // stride on block 0's conv2/downsample
            ggml_tensor * shortcut = cur;
            if (B.ds_w) { // block 0: downsample = StdConv 1×1 (stride) → GN (no act)
                shortcut = conv_same(g, B.ds_w, cur, 1, s, B.in, B.out);
                shortcut = gnorm(g, shortcut, B.ds_n_w, B.ds_n_b);
            }
            ggml_tensor * y = conv_same(g, B.c1_w, cur, 1, 1, B.in, B.mid); // 1×1
            y = ggml_relu(g, gnorm(g, y, B.n1_w, B.n1_b));
            y = conv_same(g, B.c2_w, y, 3, s, B.mid, B.mid); // 3×3, stride
            y = ggml_relu(g, gnorm(g, y, B.n2_w, B.n2_b));
            y = conv_same(g, B.c3_w, y, 1, 1, B.mid, B.out); // 1×1, no act
            y = gnorm(g, y, B.n3_w, B.n3_b);
            cur = ggml_relu(g, ggml_add(g, y, shortcut));
        }
    }
    ggml_set_name(cur, "enc_backbone");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // ---- HybridEmbed proj: 1×1 Conv 1024→256 as a token matmul ----
    ggml_tensor * tok = to_tokens(g, cur);                          // [1024, N]
    ggml_tensor * pw = ggml_reshape_2d(g, ctx->proj_w, 1024, Cenc); // [1024,256]
    ggml_tensor * xt = lin(g, tok, pw, ctx->proj_b);                // [256, N]
    int N = xt->ne[1];

    // ---- ViT: prepend cls, add custom 2D pos-index, 4 blocks, final norm ----
    ggml_tensor * cls = ggml_reshape_2d(g, ctx->cls_token, Cenc, 1); // [256,1]
    ggml_tensor * seq = ggml_concat(g, cls, xt, 1);                  // [256, N+1]
    int T = N + 1;
    ggml_tensor * pos_ind = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(pos_ind, "pos_ind");
    ggml_set_input(pos_ind);
    ggml_tensor * pe2d = ggml_reshape_2d(g, ctx->pos_embed, Cenc, ctx->pos_embed->ne[1]); // [256,641]
    ggml_tensor * pos = ggml_get_rows(g, pe2d, pos_ind);                                  // [256, T]
    ggml_tensor * cur2 = ggml_add(g, seq, pos);

    for (auto & V : ctx->vit) {
        ggml_tensor * xn = ln(g, cur2, V.n1_w, V.n1_b, 1e-6f);
        ggml_tensor * qkv = lin(g, xn, V.qkv_w, V.qkv_b); // [768, T]
        ggml_tensor * q = ggml_cont(g, ggml_view_2d(g, qkv, Cenc, T, qkv->nb[1], 0));
        ggml_tensor * k = ggml_cont(g, ggml_view_2d(g, qkv, Cenc, T, qkv->nb[1], (size_t)Cenc * qkv->nb[0]));
        ggml_tensor * v = ggml_cont(g, ggml_view_2d(g, qkv, Cenc, T, qkv->nb[1], (size_t)2 * Cenc * qkv->nb[0]));
        ggml_tensor * a = attn(g, q, k, v, nh, scale, nullptr); // [256, T]
        a = lin(g, a, V.proj_w, V.proj_b);
        cur2 = ggml_add(g, cur2, a);
        ggml_tensor * hn = ln(g, cur2, V.n2_w, V.n2_b, 1e-6f);
        ggml_tensor * h = lin(g, hn, V.fc1_w, V.fc1_b); // [1024, T]
        h = ggml_gelu_erf(g, h);
        h = lin(g, h, V.fc2_w, V.fc2_b); // [256, T]
        cur2 = ggml_add(g, cur2, h);
    }
    cur2 = ln(g, cur2, ctx->enc_norm_w, ctx->enc_norm_b, 1e-6f);
    ggml_set_name(cur2, "enc_context");
    ggml_set_output(cur2);
    ggml_build_forward_expand(gf, cur2);
    return gf;
}

// Build the ViT custom pos-index on host. h=H/16 rows, w=W/16 cols (W already /16
// multiple). pos_ind[t=hh*w+ww] = hh*80 + ww + 1; cls at index 0.
static std::vector<int32_t> build_pos_index(int W, int H) {
    int h = H / 16, w = W / 16;
    std::vector<int32_t> ind;
    ind.reserve(h * w + 1);
    ind.push_back(0); // cls
    for (int hh = 0; hh < h; hh++)
        for (int ww = 0; ww < w; ww++) ind.push_back(hh * 80 + ww + 1);
    return ind;
}

// Run encoder on a preprocessed image [1,H,W] (row-major, W fastest). Fills enc_ctx.
// `buf` is caller-owned graph-meta scratch (kept alive as long as *keep_g is used).
static bool run_encoder(tromr_ocr_context * ctx, const float * input, int W, int H, std::vector<uint8_t> & buf,
                        ggml_context ** keep_g, ggml_cgraph ** keep_gf) {
    size_t meta = 128 * 1024 * 1024;
    buf.assign(meta, 0);
    ggml_init_params ip = { meta, buf.data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = build_encoder(ctx, g, W, H);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "tromr_ocr: encoder alloc failed\n");
        ggml_free(g);
        return false;
    }
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "input"), input, 0, (size_t)W * H * sizeof(float));
    std::vector<int32_t> ind = build_pos_index(W, H);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ind"), ind.data(), 0, ind.size() * sizeof(int32_t));
    ggml_backend_sched_graph_compute(ctx->sched, gf);

    ggml_tensor * out = ggml_graph_get_tensor(gf, "enc_context");
    int C = out->ne[0];
    ctx->n_ctx = out->ne[1];
    ctx->enc_ctx.resize((size_t)ctx->n_ctx * C);
    ggml_backend_tensor_get(out, ctx->enc_ctx.data(), 0, ctx->enc_ctx.size() * sizeof(float));

    if (keep_g && keep_gf) {
        *keep_g = g;
        *keep_gf = gf;
    } else {
        ggml_free(g);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Decoder graph (teacher-forced over L steps; causal self, cross vs enc_context)
// ---------------------------------------------------------------------------
static ggml_cgraph * build_decoder(tromr_ocr_context * ctx, ggml_context * g, int L) {
    const auto & hp = ctx->hp;
    const int C = hp.decoder_dim, nh = hp.decoder_heads, nE = ctx->n_ctx;
    const float scale = 1.0f / std::sqrt((float)(512 / nh)); // head_dim 64 → 64^-0.5
    ggml_cgraph * gf = ggml_new_graph_custom(g, (int)ctx->dec.size() * 64 + 512, false);

    ggml_tensor * r_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, L);
    ggml_set_name(r_ids, "r_ids");
    ggml_set_input(r_ids);
    ggml_tensor * p_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, L);
    ggml_set_name(p_ids, "p_ids");
    ggml_set_input(p_ids);
    ggml_tensor * l_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, L);
    ggml_set_name(l_ids, "l_ids");
    ggml_set_input(l_ids);
    ggml_tensor * pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, L);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);
    ggml_tensor * mem = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nE);
    ggml_set_name(mem, "mem");
    ggml_set_input(mem);
    ggml_tensor * mask = ggml_new_tensor_2d(g, GGML_TYPE_F32, L, L);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    // input embedding: rhythm + pitch + lift (unscaled) + pos (scaled d^-0.5)
    ggml_tensor * cur = ggml_get_rows(g, ctx->rhythm_emb, r_ids); // [C,L]
    cur = ggml_add(g, cur, ggml_get_rows(g, ctx->pitch_emb, p_ids));
    cur = ggml_add(g, cur, ggml_get_rows(g, ctx->lift_emb, l_ids));
    ggml_tensor * pos = ggml_get_rows(g, ctx->pos_emb, pos_ids);
    pos = ggml_scale(g, pos, 1.0f / std::sqrt((float)C));
    cur = ggml_add(g, cur, pos);
    ggml_set_name(cur, "dec_tok_emb");
    ggml_set_output(cur);

    for (size_t i = 0; i < ctx->dec.size(); i++) {
        auto & Lr = ctx->dec[i];
        ggml_tensor * xn = ln(g, cur, Lr.ln_w, Lr.ln_b, 1e-5f);
        ggml_tensor * blk;
        if (Lr.kind == 'f') {
            ggml_tensor * h = lin(g, xn, Lr.ff0_w, Lr.ff0_b); // [2048,L]
            h = glu(g, h, GLU_GELU);                          // GEGLU → [1024,L]
            blk = lin(g, h, Lr.ff3_w, Lr.ff3_b);              // [256,L]
        } else {
            ggml_tensor * kv_src = (Lr.kind == 'c') ? mem : xn;
            ggml_tensor * q = ggml_mul_mat(g, Lr.q_w, xn);     // [512,L]
            ggml_tensor * k = ggml_mul_mat(g, Lr.k_w, kv_src); // [512,Lk]
            ggml_tensor * v = ggml_mul_mat(g, Lr.v_w, kv_src);
            ggml_tensor * a = attn(g, q, k, v, nh, scale, (Lr.kind == 'a') ? mask : nullptr); // [512,L]
            a = ggml_mul_mat(g, Lr.out_w, a);                                                 // [512,L]
            blk = glu(g, a, GLU_SIGMOID); // attn-on-attn SIGLU → [256,L]
        }
        char nm[24];
        snprintf(nm, sizeof(nm), "dec_block%d", (int)i);
        ggml_set_name(blk, nm);
        ggml_set_output(blk);
        ggml_build_forward_expand(gf, blk);
        cur = ggml_add(g, cur, blk); // residual
    }
    cur = ln(g, cur, ctx->dec_norm_w, ctx->dec_norm_b, 1e-5f);

    struct {
        const char * nm;
        ggml_tensor *w, *b;
    } heads[4] = {
        { "logits_rhythm", ctx->lg_r_w, ctx->lg_r_b },
        { "logits_pitch", ctx->lg_p_w, ctx->lg_p_b },
        { "logits_lift", ctx->lg_l_w, ctx->lg_l_b },
        { "logits_note", ctx->lg_n_w, ctx->lg_n_b },
    };
    for (auto & h : heads) {
        ggml_tensor * lg = lin(g, cur, h.w, h.b);
        ggml_set_name(lg, h.nm);
        ggml_set_output(lg);
        ggml_build_forward_expand(gf, lg);
    }
    return gf;
}

static std::vector<float> causal_mask(int L) {
    std::vector<float> m((size_t)L * L, 0.0f);
    for (int i = 0; i < L; i++)
        for (int j = 0; j < L; j++) m[(size_t)i * L + j] = (j <= i) ? 0.0f : -INFINITY;
    return m;
}

// Run decoder teacher-forced over the 3 id streams. Keeps graph for read-back.
// `buf` is caller-owned graph-meta scratch (kept alive as long as *keep_g is used).
static bool run_decoder(tromr_ocr_context * ctx, const std::vector<int32_t> & r, const std::vector<int32_t> & p,
                        const std::vector<int32_t> & l, std::vector<uint8_t> & buf, ggml_context ** keep_g,
                        ggml_cgraph ** keep_gf) {
    int L = (int)r.size();
    size_t meta = 64 * 1024 * 1024;
    buf.assign(meta, 0);
    ggml_init_params ip = { meta, buf.data(), true };
    ggml_context * g = ggml_init(ip);
    ggml_cgraph * gf = build_decoder(ctx, g, L);

    ggml_backend_sched_reset(ctx->sched);
    if (!ggml_backend_sched_alloc_graph(ctx->sched, gf)) {
        fprintf(stderr, "tromr_ocr: decoder alloc failed\n");
        ggml_free(g);
        return false;
    }
    std::vector<int32_t> pos(L);
    for (int i = 0; i < L; i++) pos[i] = i;
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "r_ids"), r.data(), 0, L * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "p_ids"), p.data(), 0, L * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "l_ids"), l.data(), 0, L * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"), pos.data(), 0, L * sizeof(int32_t));
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mem"), ctx->enc_ctx.data(), 0,
                            ctx->enc_ctx.size() * sizeof(float));
    std::vector<float> mask = causal_mask(L);
    ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask"), mask.data(), 0, mask.size() * sizeof(float));
    ggml_backend_sched_graph_compute(ctx->sched, gf);

    if (keep_g && keep_gf) {
        *keep_g = g;
        *keep_gf = gf;
    } else {
        ggml_free(g);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Detokenize (staff2score.detokenize): 'Ġ'→space, strip, drop [BOS]/[EOS]/[PAD].
// ---------------------------------------------------------------------------
static std::string detok(const std::vector<std::string> & vocab, int id) {
    if (id < 0 || id >= (int)vocab.size()) return "";
    std::string t = vocab[id];
    if (t == "[BOS]" || t == "[EOS]" || t == "[PAD]") return "";
    // 'Ġ' is UTF-8 0xC4 0xA0 (GPT-2 space marker) → space, then strip.
    std::string s;
    for (size_t i = 0; i < t.size();) {
        if (i + 1 < t.size() && (uint8_t)t[i] == 0xC4 && (uint8_t)t[i + 1] == 0xA0) {
            s += ' ';
            i += 2;
        } else {
            s += t[i++];
        }
    }
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// ---------------------------------------------------------------------------
// Init / free
// ---------------------------------------------------------------------------
tromr_ocr_context * tromr_ocr_init(const char * model_path, int n_threads) {
    auto ctx = new tromr_ocr_context();
    ctx->n_threads = n_threads > 0 ? n_threads : 4;

    gguf_context * gc = core_gguf::open_metadata(model_path);
    if (!gc) {
        fprintf(stderr, "tromr_ocr: can't open %s\n", model_path);
        delete ctx;
        return nullptr;
    }
    auto & hp = ctx->hp;
    hp.channels = core_gguf::kv_u32(gc, "tromr.channels", 1);
    hp.patch_size = core_gguf::kv_u32(gc, "tromr.patch_size", 16);
    hp.max_height = core_gguf::kv_u32(gc, "tromr.max_height", 128);
    hp.max_width = core_gguf::kv_u32(gc, "tromr.max_width", 1280);
    hp.max_seq_len = core_gguf::kv_u32(gc, "tromr.max_seq_len", 256);
    hp.encoder_dim = core_gguf::kv_u32(gc, "tromr.encoder_dim", 256);
    hp.encoder_depth = core_gguf::kv_u32(gc, "tromr.encoder_depth", 4);
    hp.encoder_heads = core_gguf::kv_u32(gc, "tromr.encoder_heads", 8);
    hp.decoder_dim = core_gguf::kv_u32(gc, "tromr.decoder_dim", 256);
    hp.decoder_depth = core_gguf::kv_u32(gc, "tromr.decoder_depth", 4);
    hp.decoder_heads = core_gguf::kv_u32(gc, "tromr.decoder_heads", 8);
    hp.num_rhythm_tokens = core_gguf::kv_u32(gc, "tromr.num_rhythm_tokens", 260);
    hp.num_pitch_tokens = core_gguf::kv_u32(gc, "tromr.num_pitch_tokens", 71);
    hp.num_lift_tokens = core_gguf::kv_u32(gc, "tromr.num_lift_tokens", 7);
    hp.num_note_tokens = core_gguf::kv_u32(gc, "tromr.num_note_tokens", 2);
    hp.bos_token = core_gguf::kv_u32(gc, "tromr.bos_token", 1);
    hp.eos_token = core_gguf::kv_u32(gc, "tromr.eos_token", 2);
    hp.pad_token = core_gguf::kv_u32(gc, "tromr.pad_token", 0);
    hp.nonote_token = core_gguf::kv_u32(gc, "tromr.nonote_token", 0);
    hp.norm_mean = core_gguf::kv_f32(gc, "tromr.norm_mean", 0.7931f);
    hp.norm_std = core_gguf::kv_f32(gc, "tromr.norm_std", 0.1738f);
    ctx->rhythm_tok = core_gguf::kv_str_array(gc, "tromr.rhythm_tokens");
    ctx->pitch_tok = core_gguf::kv_str_array(gc, "tromr.pitch_tokens");
    ctx->lift_tok = core_gguf::kv_str_array(gc, "tromr.lift_tokens");
    core_gguf::free_metadata(gc);

    fprintf(stderr, "tromr_ocr: enc d%d/%dL/%dH dec d%d/%dx3/%dH vocab r%zu p%zu l%zu\n", hp.encoder_dim,
            hp.encoder_depth, hp.encoder_heads, hp.decoder_dim, hp.decoder_depth, hp.decoder_heads,
            ctx->rhythm_tok.size(), ctx->pitch_tok.size(), ctx->lift_tok.size());

    bool force_cpu = (getenv("TROMR_OCR_FORCE_CPU") && atoi(getenv("TROMR_OCR_FORCE_CPU")));
    ctx->backend = force_cpu ? ggml_backend_cpu_init() : crispasr_init_gpu_backend();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (ggml_backend_is_cpu(ctx->backend)) ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    ctx->backend_cpu = ggml_backend_is_cpu(ctx->backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx->backend_cpu) ggml_backend_cpu_set_n_threads(ctx->backend_cpu, ctx->n_threads);

    if (!core_gguf::load_weights(model_path, ctx->backend, "tromr_ocr", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    std::vector<ggml_backend_t> backends = { ctx->backend };
    if (ctx->backend_cpu) backends.push_back(ctx->backend_cpu);
    ctx->sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), 8192, false, false);

    map_tensors(ctx);
    fprintf(stderr, "tromr_ocr: loaded %zu tensors, init complete\n", ctx->wl.tensors.size());
    return ctx;
}

void tromr_ocr_free(tromr_ocr_context * ctx) {
    if (!ctx) return;
    if (ctx->sched) ggml_backend_sched_free(ctx->sched);
    if (ctx->backend_cpu) ggml_backend_free(ctx->backend_cpu);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    core_gguf::free_weights(ctx->wl);
    delete ctx;
}

const tromr_ocr_hparams * tromr_ocr_get_hparams(const tromr_ocr_context * ctx) {
    return ctx ? &ctx->hp : nullptr;
}

// ---------------------------------------------------------------------------
// Greedy decode (argmax over the 3 streams, stop on rhythm==eos)
// ---------------------------------------------------------------------------
static int argmax(const float * v, int n) {
    int best = 0;
    float bv = v[0];
    for (int i = 1; i < n; i++)
        if (v[i] > bv) {
            bv = v[i];
            best = i;
        }
    return best;
}

static void decode_greedy(tromr_ocr_context * ctx) {
    const auto & hp = ctx->hp;
    std::vector<int32_t> r = { hp.bos_token }, p = { hp.nonote_token }, l = { hp.nonote_token };
    ctx->result.clear();
    int maxlen = hp.max_seq_len > 0 ? hp.max_seq_len : 256;
    std::vector<uint8_t> buf; // graph-meta scratch, reused (freed) across steps
    for (int step = 0; step < maxlen; step++) {
        ggml_context * g = nullptr;
        ggml_cgraph * gf = nullptr;
        if (!run_decoder(ctx, r, p, l, buf, &g, &gf)) break;
        int L = (int)r.size();
        auto last = [&](const char * nm, int V) {
            ggml_tensor * t = ggml_graph_get_tensor(gf, nm);
            std::vector<float> buf(V);
            ggml_backend_tensor_get(t, buf.data(), (size_t)(L - 1) * V * sizeof(float), V * sizeof(float));
            return argmax(buf.data(), V);
        };
        int rr = last("logits_rhythm", hp.num_rhythm_tokens);
        int pp = last("logits_pitch", hp.num_pitch_tokens);
        int ll = last("logits_lift", hp.num_lift_tokens);
        ggml_free(g);
        if (rr == hp.eos_token) break;
        r.push_back(rr);
        p.push_back(pp);
        l.push_back(ll);
    }
    // Canonical stream merge (Polyphonic-TrOMR inference.py). Streams exclude the
    // seed at index 0. rhythm drives structure: "|" chains a chord (replaces the
    // trailing '+'); a "note" duration merges pitch(+lift)_duration; anything else
    // (clef/key/barline/rest) is emitted verbatim.
    std::vector<std::string> pr, ppc, pl;
    for (size_t i = 1; i < r.size(); i++) {
        pr.push_back(detok(ctx->rhythm_tok, r[i]));
        ppc.push_back(detok(ctx->pitch_tok, p[i]));
        pl.push_back(detok(ctx->lift_tok, l[i]));
    }
    ctx->result.clear();
    if (pr.empty()) return;
    std::string merge = pr[0] + "+";
    for (size_t j = 1; j < pr.size(); j++) {
        if (pr[j] == "|") {
            if (!merge.empty()) merge.pop_back(); // drop trailing '+'
            merge += "|";
        } else if (pr[j].find("note") != std::string::npos) {
            std::string lift;
            if (pl[j] == "lift_##" || pl[j] == "lift_#" || pl[j] == "lift_bb" || pl[j] == "lift_b" || pl[j] == "lift_N")
                lift = pl[j].substr(pl[j].find('_') + 1);
            std::string suf = pr[j];
            size_t pos = suf.rfind("note-");
            if (pos != std::string::npos) suf = suf.substr(pos + 5);
            merge += ppc[j] + lift + "_" + suf + "+";
        } else {
            merge += pr[j] + "+";
        }
    }
    if (!merge.empty() && merge.back() == '+') merge.pop_back();
    ctx->result = merge;
}

// ---------------------------------------------------------------------------
// Preprocessing (staff2score.readimg + transform), then encode + decode.
//   RGB → resize (h=128, w=int(128/h*w)//16*16, bilinear) → luma → normalize.
// ---------------------------------------------------------------------------
const char * tromr_ocr_recognize_raw(tromr_ocr_context * ctx, const uint8_t * data, int w, int h, int ch,
                                     int * out_len) {
    if (!data || w <= 0 || h <= 0 || ch <= 0) return nullptr;
    const int nh = ctx->hp.max_height; // 128
    int nw = (int)((double)nh / h * w) / 16 * 16;
    if (nw < 16) nw = 16;
    if (nw > ctx->hp.max_width) nw = ctx->hp.max_width;

    float sx = (float)w / nw, sy = (float)h / nh;
    auto clampi = [](int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); };
    int nk = ch >= 3 ? 3 : 1;
    std::vector<float> img((size_t)nw * nh);
    const float mean = ctx->hp.norm_mean, istd = 1.0f / ctx->hp.norm_std;
    for (int y = 0; y < nh; y++) {
        float fy = (y + 0.5f) * sy - 0.5f;
        int y0 = (int)std::floor(fy);
        float wy = fy - y0;
        int y0c = clampi(y0, 0, h - 1), y1c = clampi(y0 + 1, 0, h - 1);
        for (int x = 0; x < nw; x++) {
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
                pix[k] = std::round(top + wy * (bot - top)); // cv2 resize → uint8
            }
            float luma = (nk >= 3) ? std::round(0.299f * pix[0] + 0.587f * pix[1] + 0.114f * pix[2]) : pix[0];
            img[(size_t)y * nw + x] = (luma / 255.0f - mean) * istd; // NON-inverted
        }
    }
    std::vector<uint8_t> ebuf;
    if (!run_encoder(ctx, img.data(), nw, nh, ebuf, nullptr, nullptr)) return nullptr;
    decode_greedy(ctx);
    if (out_len) *out_len = (int)ctx->result.size();
    return ctx->result.c_str();
}

const char * tromr_ocr_recognize_file(tromr_ocr_context * ctx, const char * image_path, int * out_len) {
    int w = 0, h = 0, c = 0;
    stbi_uc * data = stbi_load(image_path, &w, &h, &c, 3); // force RGB
    if (!data) return nullptr;
    const char * r = tromr_ocr_recognize_raw(ctx, data, w, h, 3, out_len);
    stbi_image_free(data);
    return r;
}

// ---------------------------------------------------------------------------
// Per-stage parity harness vs tools/dump_tromr_reference.py
// ---------------------------------------------------------------------------
int tromr_ocr_run_diff(tromr_ocr_context * ctx, const char * ref_path) {
    crispembed_diff::Ref ref;
    if (!ref.load(ref_path)) return 1;

    auto inshape = ref.shape("input_tensor"); // ne = [W,H,1]
    if (inshape.size() < 2) {
        fprintf(stderr, "tromr_diff: no input_tensor\n");
        return 1;
    }
    int W = (int)inshape[0], H = (int)inshape[1];
    auto in = ref.get_f32("input_tensor");
    fprintf(stderr, "tromr_diff: input W=%d H=%d (%zu floats)\n", W, H, in.second);

    int fails = 0;
    auto report = [&](const char * nm, const crispembed_diff::Report & r) {
        bool pass = r.is_pass();
        fprintf(stderr, "[tromr-diff] %-14s cos_min=%.6f cos_mean=%.6f max_abs=%.2e %s\n", nm, r.cos_min, r.cos_mean,
                r.max_abs, pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    };
    auto readcmp = [&](ggml_cgraph * gf, const char * nm, int rowD) {
        if (!ref.has(nm)) return;
        ggml_tensor * t = ggml_graph_get_tensor(gf, nm);
        if (!t) {
            fprintf(stderr, "[tromr-diff] %-14s MISSING in graph\n", nm);
            fails++;
            return;
        }
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        report(nm, ref.compare(nm, buf.data(), buf.size(), rowD));
    };

    // ---- encoder ----
    std::vector<uint8_t> ebuf;
    ggml_context * eg = nullptr;
    ggml_cgraph * egf = nullptr;
    if (!run_encoder(ctx, in.first, W, H, ebuf, &eg, &egf)) return 1;
    // enc_backbone is a sparse ResNet map (post-ReLU) with many all-zero spatial
    // rows, so a per-row cosine divides by ~0 → spurious 0.0. Report a single
    // GLOBAL cosine (+max_abs), which is the honest structural gate here.
    if (ref.has("enc_backbone")) {
        ggml_tensor * t = ggml_graph_get_tensor(egf, "enc_backbone");
        std::vector<float> buf(ggml_nelements(t));
        ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
        auto rf = ref.get_f32("enc_backbone");
        size_t n = std::min(buf.size(), rf.second);
        double dot = 0, na = 0, nb = 0, mx = 0;
        for (size_t i = 0; i < n; i++) {
            dot += (double)buf[i] * rf.first[i];
            na += (double)buf[i] * buf[i];
            nb += (double)rf.first[i] * rf.first[i];
            mx = std::max(mx, (double)std::fabs(buf[i] - rf.first[i]));
        }
        float cos = (na > 0 && nb > 0) ? (float)(dot / (std::sqrt(na) * std::sqrt(nb))) : 0.0f;
        bool pass = cos >= 0.999f;
        fprintf(stderr, "[tromr-diff] %-14s cos_glob=%.6f max_abs=%.2e %s\n", "enc_backbone", cos, mx,
                pass ? "PASS" : "FAIL");
        if (!pass) fails++;
    }
    readcmp(egf, "enc_context", 0); // [256, N+1] → per-token
    ggml_free(eg);

    // ---- decoder (teacher-forced on ref id streams) ----
    auto to_ids = [&](const char * nm) {
        auto f = ref.get_f32(nm);
        std::vector<int32_t> v(f.second);
        for (size_t i = 0; i < f.second; i++) v[i] = (int32_t)llround(f.first[i]);
        return v;
    };
    std::vector<int32_t> r = to_ids("ids_rhythm"), p = to_ids("ids_pitch"), l = to_ids("ids_lift");
    if (r.empty()) {
        fprintf(stderr, "tromr_diff: no ids_rhythm — skipping decoder\n");
        return fails ? 1 : 0;
    }
    fprintf(stderr, "tromr_diff: teacher-forcing %zu tokens\n", r.size());
    std::vector<uint8_t> dbuf;
    ggml_context * dg = nullptr;
    ggml_cgraph * dgf = nullptr;
    if (!run_decoder(ctx, r, p, l, dbuf, &dg, &dgf)) return 1;
    readcmp(dgf, "dec_tok_emb", 0);
    for (size_t i = 0; i < ctx->dec.size(); i++) {
        char nm[24];
        snprintf(nm, sizeof(nm), "dec_block%d", (int)i);
        readcmp(dgf, nm, 0);
    }
    readcmp(dgf, "logits_rhythm", 0);
    readcmp(dgf, "logits_pitch", 0);
    readcmp(dgf, "logits_lift", 0);
    readcmp(dgf, "logits_note", 0);

    // Per-position argmax agreement (the decode-relevant metric): under identical
    // teacher-forcing, how often does my argmax match the reference's? 100% ⇒ the
    // greedy path is bit-faithful; any gap is a near-tie flip (F16 conv-cast noise).
    auto agree = [&](const char * nm, int V) {
        if (!ref.has(nm)) return;
        ggml_tensor * t = ggml_graph_get_tensor(dgf, nm);
        std::vector<float> mine(ggml_nelements(t));
        ggml_backend_tensor_get(t, mine.data(), 0, mine.size() * sizeof(float));
        auto rf = ref.get_f32(nm);
        int L = (int)(mine.size() / V), ok = 0;
        for (int i = 0; i < L; i++) {
            if (argmax(&mine[(size_t)i * V], V) == argmax(&rf.first[(size_t)i * V], V)) ok++;
        }
        fprintf(stderr, "[tromr-diff] argmax %-8s %d/%d positions agree\n", nm + 7, ok, L);
    };
    agree("logits_rhythm", ctx->hp.num_rhythm_tokens);
    agree("logits_pitch", ctx->hp.num_pitch_tokens);
    agree("logits_lift", ctx->hp.num_lift_tokens);
    ggml_free(dg);

    fprintf(stderr, "tromr_diff: %s (%d stage failures)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
