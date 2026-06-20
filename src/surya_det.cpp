// surya_det.cpp — surya-ocr-2 text detector (EfficientViT segformer)
//
// CPU-scalar forward pass with BN pre-folded into conv weights.
// All convolution weights stored as [OC, IC, KH, KW] (or [OC, 1, KH, KW]
// for depthwise). The GGUF converter handles BN folding.
//
// Debug: set env SURYA_DET_DUMP=1 to print per-layer stats.

#include "surya_det.h"
#include "core/gguf_loader.h"
#include "core/cpu_ops.h"
#include "ggml-cpu.h"

#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <map>
#include <unordered_map>
#include <string>
#include <vector>

using core_cpu::to_f32;
using core_cpu::conv2d_cpu;
using core_cpu::hardswish_inplace;
using core_cpu::relu6_inplace;
using core_cpu::relu_inplace;
using core_cpu::linear_cpu;

// Bilinear interpolation for upsampling [C, H, W] → [C, tH, tW]
static void bilinear_upsample(const float* in, float* out,
                               int C, int H, int W, int tH, int tW) {
    for (int c = 0; c < C; c++) {
        for (int oy = 0; oy < tH; oy++) {
            float iy_f = (oy + 0.5f) * H / tH - 0.5f;
            int iy0 = (int)floorf(iy_f);
            int iy1 = iy0 + 1;
            float fy = iy_f - iy0;
            iy0 = std::max(0, std::min(iy0, H - 1));
            iy1 = std::max(0, std::min(iy1, H - 1));
            for (int ox = 0; ox < tW; ox++) {
                float ix_f = (ox + 0.5f) * W / tW - 0.5f;
                int ix0 = (int)floorf(ix_f);
                int ix1 = ix0 + 1;
                float fx = ix_f - ix0;
                ix0 = std::max(0, std::min(ix0, W - 1));
                ix1 = std::max(0, std::min(ix1, W - 1));

                float v00 = in[c * H * W + iy0 * W + ix0];
                float v01 = in[c * H * W + iy0 * W + ix1];
                float v10 = in[c * H * W + iy1 * W + ix0];
                float v11 = in[c * H * W + iy1 * W + ix1];

                float v = (1 - fy) * ((1 - fx) * v00 + fx * v01)
                        + fy * ((1 - fx) * v10 + fx * v11);
                out[c * tH * tW + oy * tW + ox] = v;
            }
        }
    }
}

// Sigmoid in-place
static void sigmoid_inplace(float* data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = 1.0f / (1.0f + expf(-data[i]));
}

// ---------------------------------------------------------------------------
// Conv layer helpers (with dequant from ggml tensors)
// ---------------------------------------------------------------------------
struct conv_layer {
    ggml_tensor* weight;
    ggml_tensor* bias;
};

// Apply conv + activation (BN already folded)
static std::vector<float> apply_conv(const float* in, const conv_layer& l,
                                      int in_ch, int H, int W,
                                      int out_ch, int kh, int kw,
                                      int stride, int pad, int groups,
                                      int act) { // 0=none, 1=hardswish, 2=relu6, 3=relu
    int out_H = (H + 2 * pad - kh) / stride + 1;
    int out_W = (W + 2 * pad - kw) / stride + 1;
    std::vector<float> out(out_ch * out_H * out_W);
    auto wv = to_f32(l.weight);
    auto bv = to_f32(l.bias);
    conv2d_cpu(in, out.data(), wv.data(), bv.data(),
               in_ch, out_ch, H, W, kh, kw, stride, pad, groups);
    int n = out_ch * out_H * out_W;
    if (act == 1) hardswish_inplace(out.data(), n);
    else if (act == 2) relu6_inplace(out.data(), n);
    else if (act == 3) relu_inplace(out.data(), n);
    return out;
}

// ---------------------------------------------------------------------------
// Weight structures
// ---------------------------------------------------------------------------
struct fused_mbconv_weights {
    conv_layer spatial;  // [expand*ch, in_ch, kh, kw] or grouped
    conv_layer point;    // [out_ch, expand*ch, 1, 1]
};

struct mbconv_weights {
    conv_layer inverted;  // [mid_ch, in_ch, 1, 1] pointwise expand
    conv_layer depth;     // [mid_ch, 1, 3, 3] depthwise
    conv_layer point;     // [out_ch, mid_ch, 1, 1] pointwise project
};

struct litemla_weights {
    conv_layer qkv;       // [3*total_dim, in_ch, 1, 1]
    conv_layer agg_dw;    // [3*total_dim, 1, 5, 5] depthwise
    conv_layer agg_pw;    // [3*total_dim, groups_dim, 1, 1] grouped pointwise
    conv_layer proj;      // [out_ch, total_dim*(1+n_scales), 1, 1]
};

struct evitvit_block_weights {
    litemla_weights ctx;
    mbconv_weights local;
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------
struct surya_det_context {
    surya_det_hparams hp;
    int n_threads;

    // Model storage
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;

    // Stem
    conv_layer stem_in_conv;
    conv_layer stem_res0_conv1;
    conv_layer stem_res0_conv2;

    // Stage 0-1: FusedMBConv blocks
    fused_mbconv_weights stage01[2][2]; // stage01[stage][block]

    // Stage 2: MBConv blocks (7)
    mbconv_weights stage2[7];

    // Stage 3: block0 = MBConv, blocks 1-6 = EfficientVitBlock
    mbconv_weights stage3_block0;
    evitvit_block_weights stage3_vit[6]; // blocks 1-6

    // Decode head
    ggml_tensor* dec_proj_w[4];
    ggml_tensor* dec_proj_b[4];
    conv_layer dec_fuse;       // Conv1x1 + folded BN
    conv_layer dec_classifier; // Conv1x1 → 2 classes

    // Output buffers
    std::vector<float> heatmap;
    int heatmap_h, heatmap_w;
    std::vector<surya_det_bbox> boxes;

    // Persistent allocator
    ggml_gallocr_t galloc = nullptr;

    // Debug
    bool dump;
    bool bench;
    std::map<std::string, std::vector<float>> debug_tensors;
};

// Debug helper
static void dump_stats(const char* name, const float* data, int n) {
    float mn = data[0], mx = data[0];
    double sum = 0;
    for (int i = 0; i < n; i++) {
        if (data[i] < mn) mn = data[i];
        if (data[i] > mx) mx = data[i];
        sum += data[i];
    }
    fprintf(stderr, "  %-30s: min=%8.4f max=%8.4f mean=%8.4f  [n=%d]\n",
            name, mn, mx, (float)(sum / n), n);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
// Helper: find tensor in map
static ggml_tensor* find(const std::unordered_map<std::string, ggml_tensor*>& m, const char* name) {
    return core_gguf::try_get(m, name);
}

surya_det_context * surya_det_init(const char * model_path, int n_threads) {
    auto* ctx = new surya_det_context{};
    ctx->n_threads = n_threads > 0 ? n_threads : 4;
    ctx->dump  = (getenv("SURYA_DET_DUMP") != nullptr);
    ctx->bench = (std::getenv("CRISPEMBED_SURYA_DET_BENCH") != nullptr);

    // Pass 1: metadata
    gguf_context* gctx = core_gguf::open_metadata(model_path);
    if (!gctx) { delete ctx; return nullptr; }
    // Could read hparams from GGUF keys here, but we hardcode for now
    core_gguf::free_metadata(gctx);

    // Pass 2: load weights. Use the best available backend (Metal on Apple
    // Silicon, CUDA elsewhere) so the stage 0-2 / decode-head graph runs on
    // GPU; set SURYA_DET_FORCE_CPU=1 to pin to CPU for parity debugging.
    const char* force_cpu = std::getenv("SURYA_DET_FORCE_CPU");
    bool want_cpu = force_cpu && force_cpu[0] && std::strcmp(force_cpu, "0") != 0;
    ctx->backend = want_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!ctx->backend) ctx->backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(model_path, ctx->backend, "surya_det", ctx->wl)) {
        ggml_backend_free(ctx->backend);
        delete ctx;
        return nullptr;
    }
    // n_threads only affects the CPU backend; on GPU backends this is a no-op.
    if (ggml_backend_is_cpu(ctx->backend)) {
        ggml_backend_cpu_set_n_threads(ctx->backend, ctx->n_threads);
    }
    {
        ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend);
        fprintf(stderr, "surya_det: backend = %s\n",
                dev ? ggml_backend_dev_name(dev) : ggml_backend_name(ctx->backend));
    }

    // Hyperparams
    auto& hp = ctx->hp;
    hp.input_h = 1200; hp.input_w = 1200;
    hp.num_classes = 2;
    hp.stem_ch = 32;
    hp.stage_ch[0] = 64; hp.stage_ch[1] = 128;
    hp.stage_ch[2] = 256; hp.stage_ch[3] = 512;
    hp.head_dim = 32;
    hp.dec_hidden = 512;
    hp.dec_layer_hidden = 128;

    const auto& m = ctx->wl.tensors;

    // Helper to find tensor by name
    auto F = [&](const char* name) -> ggml_tensor* {
        return find(m, name);
    };

    char buf[256];
    auto T = [&](const char* fmt, ...) -> ggml_tensor* {
        va_list args;
        va_start(args, fmt);
        vsnprintf(buf, sizeof(buf), fmt, args);
        va_end(args);
        return F(buf);
    };

    // Load weights
    ctx->stem_in_conv  = { F("stem.in_conv.weight"),  F("stem.in_conv.bias") };
    ctx->stem_res0_conv1 = { F("stem.res0.conv1.weight"), F("stem.res0.conv1.bias") };
    ctx->stem_res0_conv2 = { F("stem.res0.conv2.weight"), F("stem.res0.conv2.bias") };

    // Stages 0-1: FusedMBConv
    for (int s = 0; s < 2; s++) {
        for (int b = 0; b < 2; b++) {
            ctx->stage01[s][b].spatial = {
                T("stage%d.block%d.spatial.weight", s, b),
                T("stage%d.block%d.spatial.bias", s, b)
            };
            ctx->stage01[s][b].point = {
                T("stage%d.block%d.point.weight", s, b),
                T("stage%d.block%d.point.bias", s, b)
            };
        }
    }

    // Stage 2: MBConv
    for (int b = 0; b < 7; b++) {
        ctx->stage2[b].inverted = {
            T("stage2.block%d.inverted.weight", b),
            T("stage2.block%d.inverted.bias", b)
        };
        ctx->stage2[b].depth = {
            T("stage2.block%d.depth.weight", b),
            T("stage2.block%d.depth.bias", b)
        };
        ctx->stage2[b].point = {
            T("stage2.block%d.point.weight", b),
            T("stage2.block%d.point.bias", b)
        };
    }

    // Stage 3 block 0: MBConv
    ctx->stage3_block0.inverted = {
        F("stage3.block0.inverted.weight"), F("stage3.block0.inverted.bias")
    };
    ctx->stage3_block0.depth = {
        F("stage3.block0.depth.weight"), F("stage3.block0.depth.bias")
    };
    ctx->stage3_block0.point = {
        F("stage3.block0.point.weight"), F("stage3.block0.point.bias")
    };

    // Stage 3 blocks 1-6: EfficientVitBlock
    for (int b = 0; b < 6; b++) {
        int bi = b + 1;
        auto& vit = ctx->stage3_vit[b];
        vit.ctx.qkv    = { T("stage3.block%d.ctx.qkv.weight", bi),
                            T("stage3.block%d.ctx.qkv.bias", bi) };
        vit.ctx.agg_dw = { T("stage3.block%d.ctx.agg_dw.weight", bi),
                            T("stage3.block%d.ctx.agg_dw.bias", bi) };
        vit.ctx.agg_pw = { T("stage3.block%d.ctx.agg_pw.weight", bi),
                            T("stage3.block%d.ctx.agg_pw.bias", bi) };
        vit.ctx.proj   = { T("stage3.block%d.ctx.proj.weight", bi),
                            T("stage3.block%d.ctx.proj.bias", bi) };

        vit.local.inverted = { T("stage3.block%d.local.inverted.weight", bi),
                                T("stage3.block%d.local.inverted.bias", bi) };
        vit.local.depth    = { T("stage3.block%d.local.depth.weight", bi),
                                T("stage3.block%d.local.depth.bias", bi) };
        vit.local.point    = { T("stage3.block%d.local.point.weight", bi),
                                T("stage3.block%d.local.point.bias", bi) };
    }

    // Decode head
    for (int i = 0; i < 4; i++) {
        ctx->dec_proj_w[i] = T("dec.proj%d.weight", i);
        ctx->dec_proj_b[i] = T("dec.proj%d.bias", i);
    }
    ctx->dec_fuse       = { F("dec.fuse.weight"), F("dec.fuse.bias") };
    ctx->dec_classifier = { F("dec.classifier.weight"), F("dec.classifier.bias") };

    ctx->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(ctx->backend));

    fprintf(stderr, "surya_det: loaded %s (%d threads)\n", model_path, n_threads);
    return ctx;
}

void surya_det_free(surya_det_context * ctx) {
    if (ctx) {
        if (ctx->galloc) ggml_gallocr_free(ctx->galloc);
        core_gguf::free_weights(ctx->wl);
        if (ctx->backend) ggml_backend_free(ctx->backend);
        delete ctx;
    }
}

const surya_det_hparams * surya_det_get_hparams(const surya_det_context * ctx) {
    return ctx ? &ctx->hp : nullptr;
}

// ---------------------------------------------------------------------------
// Forward pass: FusedMBConv
// ---------------------------------------------------------------------------
// FusedMBConv: spatial_conv(kh×kw, stride, groups=1) + act → point_conv(1×1) (no act)
// Block 0 of each stage: stride=2 (downsample), no residual
// Block 1+: stride=1, with residual
// (FusedMBConv is not used as a generic function — stages 0-1 are handled
// inline in the main forward pass since their expand ratios differ per block)

// ---------------------------------------------------------------------------
// Forward pass: MBConv
// ---------------------------------------------------------------------------
// MBConv: inverted(1x1,act) → depth(3x3,dw,act) → point(1x1,no act)
static std::vector<float> mbconv_fwd(const float* in, const mbconv_weights& w,
                                      int in_ch, int H, int W,
                                      int mid_ch, int out_ch,
                                      int stride, bool residual,
                                      int act_type) { // 1=hardswish, 2=relu6
    // inverted: expand [mid_ch, in_ch, 1, 1]
    auto mid = apply_conv(in, w.inverted, in_ch, H, W,
                          mid_ch, 1, 1, 1, 0, 1, act_type);

    // depth: depthwise [mid_ch, 1, 3, 3]
    int pad = 1;
    int kh = 3;
    // For block0 of stages, stride may be 2
    auto dw = apply_conv(mid.data(), w.depth, mid_ch, H, W,
                          mid_ch, kh, kh, stride, pad, mid_ch, act_type);
    int oH = (H + 2 * pad - kh) / stride + 1;
    int oW = (W + 2 * pad - kh) / stride + 1;

    // point: project [out_ch, mid_ch, 1, 1], no activation
    auto out = apply_conv(dw.data(), w.point, mid_ch, oH, oW,
                          out_ch, 1, 1, 1, 0, 1, 0);

    if (residual && in_ch == out_ch && stride == 1) {
        int n = out_ch * oH * oW;
        for (int i = 0; i < n; i++) out[i] += in[i];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Forward pass: LiteMLA (lightweight multi-scale linear attention)
// ---------------------------------------------------------------------------
static std::vector<float> litemla_fwd(const float* in, const litemla_weights& w,
                                       int in_ch, int H, int W, int head_dim,
                                       bool dump_debug = false, const char* prefix = "") {
    int heads = in_ch / head_dim;
    int total_dim = heads * head_dim; // == in_ch
    int HW = H * W;

    // 1. QKV projection: Conv1x1 → [3*total_dim, H, W]
    auto qkv = apply_conv(in, w.qkv, in_ch, H, W,
                           3 * total_dim, 1, 1, 1, 0, 1, 0);

    if (dump_debug) {
        char name[128];
        snprintf(name, sizeof(name), "%s_qkv", prefix);
        dump_stats(name, qkv.data(), 3 * total_dim * HW);
    }

    // 2. Multi-scale aggregation: depthwise 5x5 + grouped 1x1
    auto agg_qkv = apply_conv(qkv.data(), w.agg_dw, 3 * total_dim, H, W,
                               3 * total_dim, 5, 5, 1, 2, 3 * total_dim, 0);

    if (dump_debug) {
        char name[128];
        snprintf(name, sizeof(name), "%s_agg_dw", prefix);
        dump_stats(name, agg_qkv.data(), 3 * total_dim * HW);
    }

    auto agg_qkv2 = apply_conv(agg_qkv.data(), w.agg_pw,
                                3 * total_dim, H, W,
                                3 * total_dim, 1, 1, 1, 0, 3 * heads, 0);

    if (dump_debug) {
        char name[128];
        snprintf(name, sizeof(name), "%s_agg_pw", prefix);
        dump_stats(name, agg_qkv2.data(), 3 * total_dim * HW);
    }

    // 3. Concat original qkv + aggregated → [2 * 3*total_dim, H, W]
    //    Then reshape to [n_heads*2, 3*head_dim, HW] and transpose last two dims
    int n_scales = 1; // only 1 scale (5x5) in the default config
    int concat_dim = total_dim * (1 + n_scales); // per q/k/v channel count

    // Reshape: concatenate qkv and agg_qkv2 along channel dimension
    // Then split into Q, K, V each of size [concat_dim, HW]
    // Then chunk by head_dim: Q,K,V each [n_heads*(1+n_scales), head_dim, HW]

    // Actually the PyTorch code does:
    //   multi_scale_qkv = cat([qkv, agg_qkv2], dim=1)  → [B, 2*3*total_dim, H, W]
    //   reshape to [B, -1, 3*head_dim, HW] → [B, heads*2, 3*head_dim, HW]
    //   transpose(-1,-2) → [B, heads*2, HW, 3*head_dim]
    //   chunk 3 on dim=-1 → q,k,v each [B, heads*2, HW, head_dim]

    int total_ch = 2 * 3 * total_dim;
    std::vector<float> concat(total_ch * HW);
    // Copy qkv (3*total_dim channels)
    memcpy(concat.data(), qkv.data(), 3 * total_dim * HW * sizeof(float));
    // Copy agg_qkv2 (3*total_dim channels)
    memcpy(concat.data() + 3 * total_dim * HW, agg_qkv2.data(),
           3 * total_dim * HW * sizeof(float));

    // Reshape to [heads*2, 3*head_dim, HW], then transpose to [heads*2, HW, 3*head_dim]
    int n_groups = heads * (1 + n_scales); // heads * 2
    // chunk 3 on last dim → q,k,v each [n_groups, HW, head_dim]

    // Allocate Q, K, V: [n_groups, HW, head_dim]
    std::vector<float> Q(n_groups * HW * head_dim);
    std::vector<float> K(n_groups * HW * head_dim);
    std::vector<float> V(n_groups * HW * head_dim);

    // Read from concat in [n_groups, 3*head_dim, HW] layout
    for (int g = 0; g < n_groups; g++) {
        for (int hw = 0; hw < HW; hw++) {
            for (int d = 0; d < head_dim; d++) {
                int src_ch = g * 3 * head_dim + d;
                Q[g * HW * head_dim + hw * head_dim + d] =
                    concat[src_ch * HW + hw];
            }
            for (int d = 0; d < head_dim; d++) {
                int src_ch = g * 3 * head_dim + head_dim + d;
                K[g * HW * head_dim + hw * head_dim + d] =
                    concat[src_ch * HW + hw];
            }
            for (int d = 0; d < head_dim; d++) {
                int src_ch = g * 3 * head_dim + 2 * head_dim + d;
                V[g * HW * head_dim + hw * head_dim + d] =
                    concat[src_ch * HW + hw];
            }
        }
    }

    // 4. Apply kernel function (ReLU) to Q and K
    relu_inplace(Q.data(), n_groups * HW * head_dim);
    relu_inplace(K.data(), n_groups * HW * head_dim);

    // 5. Pad V with 1: append column of 1s → [n_groups, HW, head_dim+1]
    int vd = head_dim + 1;
    std::vector<float> V_pad(n_groups * HW * vd);
    for (int g = 0; g < n_groups; g++) {
        for (int hw = 0; hw < HW; hw++) {
            for (int d = 0; d < head_dim; d++)
                V_pad[g * HW * vd + hw * vd + d] = V[g * HW * head_dim + hw * head_dim + d];
            V_pad[g * HW * vd + hw * vd + head_dim] = 1.0f;
        }
    }

    // 6. Linear attention: out = Q @ (K^T @ V_pad) / (out[..., -1:] + eps)
    //    K^T @ V_pad: [head_dim, HW] × [HW, vd] → [head_dim, vd]
    //    Q @ KTV: [HW, head_dim] × [head_dim, vd] → [HW, vd]
    float eps = 1e-5f;
    std::vector<float> attn_out(n_groups * HW * head_dim);

    for (int g = 0; g < n_groups; g++) {
        // KTV = K^T @ V_pad: [head_dim, vd]
        std::vector<float> KTV(head_dim * vd, 0.0f);
        for (int d = 0; d < head_dim; d++) {
            for (int v = 0; v < vd; v++) {
                float sum = 0.0f;
                for (int hw = 0; hw < HW; hw++) {
                    sum += K[g * HW * head_dim + hw * head_dim + d]
                         * V_pad[g * HW * vd + hw * vd + v];
                }
                KTV[d * vd + v] = sum;
            }
        }

        // out = Q @ KTV: [HW, head_dim] × [head_dim, vd] → [HW, vd]
        for (int hw = 0; hw < HW; hw++) {
            std::vector<float> row(vd, 0.0f);
            for (int d = 0; d < head_dim; d++) {
                float qval = Q[g * HW * head_dim + hw * head_dim + d];
                for (int v = 0; v < vd; v++) {
                    row[v] += qval * KTV[d * vd + v];
                }
            }
            // Normalize by last element + eps, take first head_dim elements
            float norm = row[head_dim] + eps;
            for (int d = 0; d < head_dim; d++) {
                attn_out[g * HW * head_dim + hw * head_dim + d] = row[d] / norm;
            }
        }
    }

    // 7. Reshape back to [concat_dim, H, W] for projection
    // attn_out is [n_groups, HW, head_dim], need [n_groups * head_dim, HW]
    // = [concat_dim, H, W]
    std::vector<float> proj_in(concat_dim * HW);
    for (int g = 0; g < n_groups; g++) {
        for (int hw = 0; hw < HW; hw++) {
            for (int d = 0; d < head_dim; d++) {
                proj_in[(g * head_dim + d) * HW + hw] =
                    attn_out[g * HW * head_dim + hw * head_dim + d];
            }
        }
    }

    // 8. Final projection: Conv1x1 [out_ch, concat_dim, 1, 1] + BN(folded)
    auto out = apply_conv(proj_in.data(), w.proj, concat_dim, H, W,
                           in_ch, 1, 1, 1, 0, 1, 0);

    if (dump_debug) {
        char name[128];
        snprintf(name, sizeof(name), "%s_proj", prefix);
        dump_stats(name, out.data(), in_ch * HW);
    }

    return out;
}

// ---------------------------------------------------------------------------
// Forward pass: EfficientVitBlock = context(LiteMLA) + local(MBConv)
// Both have residual connections
// ---------------------------------------------------------------------------
static std::vector<float> evitvit_block_fwd(const float* in,
                                              const evitvit_block_weights& w,
                                              int ch, int H, int W,
                                              int head_dim,
                                              bool dump_debug = false,
                                              const char* block_prefix = "") {
    // Context module (LiteMLA with residual)
    auto ctx_out = litemla_fwd(in, w.ctx, ch, H, W, head_dim, dump_debug, block_prefix);
    int n = ch * H * W;
    for (int i = 0; i < n; i++) ctx_out[i] += in[i]; // residual

    // Local module (MBConv with residual)
    int mid_ch = ch * 6; // expand_ratio=6
    auto local_out = mbconv_fwd(ctx_out.data(), w.local, ch, H, W,
                                 mid_ch, ch, 1, false, 1 /*hardswish*/);
    for (int i = 0; i < n; i++) local_out[i] += ctx_out[i]; // residual

    return local_out;
}

// ---------------------------------------------------------------------------
// Forward pass: Decode head
// ---------------------------------------------------------------------------
static std::vector<float> decode_head_fwd(surya_det_context* ctx,
                                           const std::vector<float>* stage_outputs,
                                           const int stage_H[4], const int stage_W[4]) {
    int target_H = stage_H[0], target_W = stage_W[0];
    int dec_layer_hidden = ctx->hp.dec_layer_hidden; // 128

    // Project each stage output through linear, upsample to stage_0 size
    std::vector<std::vector<float>> projected(4);

    for (int s = 0; s < 4; s++) {
        int C = ctx->hp.stage_ch[s];
        int H = stage_H[s], W = stage_W[s];
        int HW = H * W;

        // Linear projection: flatten [C, H, W] → [HW, C], multiply by [128, C]
        auto wv = to_f32(ctx->dec_proj_w[s]);
        auto bv = to_f32(ctx->dec_proj_b[s]);

        // Output: [HW, 128]
        std::vector<float> proj(HW * dec_layer_hidden);
        for (int hw = 0; hw < HW; hw++) {
            for (int o = 0; o < dec_layer_hidden; o++) {
                float sum = bv.empty() ? 0.0f : bv[o];
                for (int ic = 0; ic < C; ic++) {
                    sum += stage_outputs[s][ic * HW + hw] * wv[o * C + ic];
                }
                proj[hw * dec_layer_hidden + o] = sum;
            }
        }

        // Permute from [HW, 128] to [128, H, W]
        std::vector<float> perm(dec_layer_hidden * HW);
        for (int c = 0; c < dec_layer_hidden; c++) {
            for (int hw = 0; hw < HW; hw++) {
                perm[c * HW + hw] = proj[hw * dec_layer_hidden + c];
            }
        }

        if (ctx->dump) {
            char name[64];
            snprintf(name, sizeof(name), "decode_proj_%d_pre_up", s);
            dump_stats(name, perm.data(), dec_layer_hidden * HW);
        }

        // Bilinear upsample to target_H × target_W
        if (H == target_H && W == target_W) {
            projected[s] = std::move(perm);
        } else {
            projected[s].resize(dec_layer_hidden * target_H * target_W);
            bilinear_upsample(perm.data(), projected[s].data(),
                              dec_layer_hidden, H, W, target_H, target_W);
        }
    }

    // Concatenate in REVERSE order (stages [3,2,1,0]) → [4*128, tH, tW]
    int cat_ch = 4 * dec_layer_hidden;
    int tHW = target_H * target_W;
    std::vector<float> cat(cat_ch * tHW);
    for (int s = 3; s >= 0; s--) {
        int offset = (3 - s) * dec_layer_hidden;
        for (int c = 0; c < dec_layer_hidden; c++) {
            memcpy(cat.data() + (offset + c) * tHW,
                   projected[s].data() + c * tHW,
                   tHW * sizeof(float));
        }
    }

    if (ctx->dump) {
        for (int s = 0; s < 4; s++) {
            char name[64];
            snprintf(name, sizeof(name), "decode_proj_%d", s);
            dump_stats(name, projected[s].data(), dec_layer_hidden * target_H * target_W);
        }
        dump_stats("decode_cat", cat.data(), cat_ch * tHW);
    }

    // Conv1x1 fuse (with folded BN) → [512, tH, tW]
    auto fused = apply_conv(cat.data(), ctx->dec_fuse, cat_ch, target_H, target_W,
                             ctx->hp.dec_hidden, 1, 1, 1, 0, 1, 3 /*relu*/);

    if (ctx->dump) dump_stats("decode_fused", fused.data(), ctx->hp.dec_hidden * tHW);

    // Classifier: Conv1x1 → [2, tH, tW]
    auto logits = apply_conv(fused.data(), ctx->dec_classifier,
                              ctx->hp.dec_hidden, target_H, target_W,
                              ctx->hp.num_classes, 1, 1, 1, 0, 1, 0);

    if (ctx->dump) dump_stats("decode_logits", logits.data(), ctx->hp.num_classes * tHW);

    // Sigmoid
    sigmoid_inplace(logits.data(), ctx->hp.num_classes * tHW);

    return logits;
}

// ---------------------------------------------------------------------------
// Image preprocessing
// ---------------------------------------------------------------------------
static std::vector<float> preprocess_image(const uint8_t* pixels, int w, int h, int ch,
                                            int target_h, int target_w) {
    // ImageNet mean/std
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_[3] = {0.229f, 0.224f, 0.225f};

    // Bilinear resize + normalize
    std::vector<float> out(3 * target_h * target_w);
    for (int c = 0; c < 3; c++) {
        for (int oy = 0; oy < target_h; oy++) {
            float fy = (float)oy * h / target_h;
            int y0 = (int)fy, y1 = std::min(y0 + 1, h - 1);
            float wy = fy - y0;
            for (int ox = 0; ox < target_w; ox++) {
                float fx = (float)ox * w / target_w;
                int x0 = (int)fx, x1 = std::min(x0 + 1, w - 1);
                float wx = fx - x0;
                auto px = [&](int y, int x) -> float {
                    if (ch == 1) return pixels[y * w + x] / 255.0f;
                    return pixels[(y * w + x) * ch + c] / 255.0f;
                };
                float pixel = (1 - wy) * ((1 - wx) * px(y0, x0) + wx * px(y0, x1))
                            +      wy  * ((1 - wx) * px(y1, x0) + wx * px(y1, x1));
                out[c * target_h * target_w + oy * target_w + ox] = (pixel - mean[c]) / std_[c];
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Graph-accelerated forward pass (ggml BLAS/SIMD)
// ---------------------------------------------------------------------------

// Helper: Conv2d + bias + activation via ggml graph
static ggml_tensor* g_conv(ggml_context* g, ggml_tensor* x,
                            const conv_layer& cl, int IC, int KH, int KW,
                            int stride, int pad, int groups, int act) {
    ggml_tensor* w = cl.weight;
    if (!w) return x;

    // Dequant quantized weights before reshape (quantized types have block
    // alignment on ne[0] that reshape can violate). ggml only supports
    // Q→F32 cast, not Q→F16, so go via F32 then to F16.
    if (ggml_is_quantized(w->type)) {
        w = ggml_cast(g, w, GGML_TYPE_F32);
    }
    // Ensure weight is 4D [KW, KH, IC/groups, OC]
    if (ggml_n_dims(w) == 2) {
        int64_t OC = w->ne[1];
        int64_t IC_g = IC / groups;
        w = ggml_reshape_4d(g, w, KW, KH, IC_g, OC);
    }
    // ggml_conv_2d requires F16 weights
    if (w->type != GGML_TYPE_F16) {
        w = ggml_cast(g, w, GGML_TYPE_F16);
    }

    if (groups > 1 && groups == IC) {
        // Depthwise conv
        x = ggml_conv_2d_dw(g, w, x, stride, stride, pad, pad, 1, 1);
    } else {
        x = ggml_conv_2d(g, w, x, stride, stride, pad, pad, 1, 1);
    }

    // Bias
    if (cl.bias) {
        ggml_tensor* b = cl.bias;
        if (b->type != GGML_TYPE_F32) b = ggml_cast(g, b, GGML_TYPE_F32);
        int64_t OC = b->ne[0];
        b = ggml_reshape_3d(g, b, 1, 1, OC);
        x = ggml_add(g, x, b);
    }

    // Activation
    if (act == 1) x = ggml_hardswish(g, x);
    else if (act == 3) x = ggml_relu(g, x);
    // act == 2 (relu6) → clamp(relu(x), 0, 6)... not needed with hardswish fix

    return x;
}

// Graph-based MBConv: inverted(1x1,act) → depth(3x3,dw,act) → point(1x1)
static ggml_tensor* g_mbconv(ggml_context* g, ggml_tensor* x,
                              const mbconv_weights& w, int in_ch,
                              int mid_ch, int stride, bool residual, int act) {
    ggml_tensor* identity = residual ? x : nullptr;
    x = g_conv(g, x, w.inverted, in_ch, 1, 1, 1, 0, 1, act);
    x = g_conv(g, x, w.depth, mid_ch, 3, 3, stride, 1, mid_ch, act);
    x = g_conv(g, x, w.point, mid_ch, 1, 1, 1, 0, 1, 0);
    if (identity) x = ggml_add(g, x, identity);
    return x;
}

// Graph-based LiteMLA linear attention.
// Linear attention formula: out = (Q @ (K^T @ V)) / (Q · K_sum)
// where K_sum[d] = sum_hw K[hw,d] — equivalent to the "pad V with 1" trick.
static ggml_tensor* g_litemla(ggml_context* g, ggml_tensor* x,
                               const litemla_weights& w,
                               int in_ch, int head_dim) {
    int heads    = in_ch / head_dim;
    int n_groups = 2 * heads;          // original scale + 1 aggregated scale
    int HW       = (int)(x->ne[0] * x->ne[1]);
    int W        = (int)x->ne[0];
    int H        = (int)x->ne[1];
    int concat_dim = n_groups * head_dim; // == 2 * in_ch

    // 1. QKV Conv1x1 + multi-scale aggregation
    ggml_tensor* qkv = g_conv(g, x,   w.qkv,    in_ch,       1, 1, 1, 0, 1,       0);
    ggml_tensor* agg = g_conv(g, qkv, w.agg_dw, 3 * in_ch,   5, 5, 1, 2, 3*in_ch, 0);
    agg              = g_conv(g, agg,  w.agg_pw, 3 * in_ch,   1, 1, 1, 0, 3*heads, 0);

    // 2. Concat [qkv, agg] along channels → [W, H, n_groups*3*head_dim]
    ggml_tensor* multi = ggml_concat(g, qkv, agg, 2);

    // 3. Reshape → [HW, 3*head_dim, n_groups]
    //    multi[c, hw] where c = g*3*head_dim + qkv_d → multi_r[hw, qkv_d, g]
    ggml_tensor* multi_r = ggml_reshape_3d(g, multi, HW, 3 * head_dim, n_groups);

    // 4. Extract Q/K/V views: each [HW, head_dim, n_groups]
    //    Strides come from multi_r (contiguous): nb1=HW*f32, nb2=HW*3*hd*f32
    size_t f = sizeof(float);
    size_t nb1 = (size_t)HW * f;
    size_t nb2 = (size_t)HW * 3 * head_dim * f;
    ggml_tensor* Q = ggml_view_3d(g, multi_r, HW, head_dim, n_groups, nb1, nb2, 0);
    ggml_tensor* K = ggml_view_3d(g, multi_r, HW, head_dim, n_groups, nb1, nb2,
                                   (size_t)head_dim * nb1);
    ggml_tensor* V = ggml_view_3d(g, multi_r, HW, head_dim, n_groups, nb1, nb2,
                                   (size_t)(2 * head_dim) * nb1);

    // 5. ReLU kernel on Q and K (ggml_relu output is contiguous)
    Q = ggml_relu(g, Q);
    K = ggml_relu(g, K);
    V = ggml_cont(g, V); // contiguous copy for matmul

    // 6. KTV = K^T @ V → [head_dim, head_dim, n_groups]
    //    ggml_mul_mat(A, B): A=[K, M], B=[K, N] → [M, N]; K=HW, M=head_dim, N=head_dim
    ggml_tensor* KTV = ggml_mul_mat(g, K, V);

    // 7. K_sum = sum_hw K[hw, :, :] → [1, head_dim, n_groups]
    ggml_tensor* K_sum = ggml_sum_rows(g, K); // sums ne[0]=HW → [1, head_dim, n_groups]

    // 8. Q_T = permute Q [HW, head_dim, n_groups] → [head_dim, HW, n_groups]
    ggml_tensor* Q_T = ggml_cont(g, ggml_permute(g, Q, 1, 0, 2, 3));

    // 9. out_unnorm = KTV @ Q_T → [head_dim, HW, n_groups]
    //    A=KTV [head_dim, head_dim, n_g], B=Q_T [head_dim, HW, n_g] → [head_dim, HW, n_g]
    ggml_tensor* out_unnorm = ggml_mul_mat(g, KTV, Q_T);

    // 10. norm = K_sum^T @ Q_T → [1, HW, n_groups]
    //    Permute K_sum [1, head_dim, n_g] → [head_dim, 1, n_g]
    ggml_tensor* K_sum_T = ggml_cont(g, ggml_permute(g, K_sum, 1, 0, 2, 3));
    ggml_tensor* norm    = ggml_mul_mat(g, K_sum_T, Q_T); // [1, HW, n_groups]

    // 11. Clamp norm to eps for numerical stability, broadcast-divide out_unnorm
    ggml_tensor* norm_c   = ggml_clamp(g, norm, 1e-5f, 1e30f);
    ggml_tensor* norm_rep = ggml_repeat(g, norm_c, out_unnorm); // [head_dim, HW, n_g]
    ggml_tensor* out_norm = ggml_div(g, out_unnorm, norm_rep);

    // 12. Permute [head_dim, HW, n_groups] → [HW, head_dim, n_groups]
    //     then reshape to [W, H, concat_dim] for the projection conv
    ggml_tensor* out_T   = ggml_cont(g, ggml_permute(g, out_norm, 1, 0, 2, 3));
    ggml_tensor* proj_in = ggml_reshape_3d(g, out_T, W, H, concat_dim);

    // 13. Final Conv1x1 projection
    return g_conv(g, proj_in, w.proj, concat_dim, 1, 1, 1, 0, 1, 0);
}

// Graph-based EfficientVitBlock = LiteMLA (context) + MBConv (local), both with residual.
static ggml_tensor* g_evitvit_block(ggml_context* g, ggml_tensor* x,
                                      const evitvit_block_weights& w,
                                      int ch, int head_dim) {
    // Context module: LiteMLA + residual
    ggml_tensor* ctx_out = g_litemla(g, x, w.ctx, ch, head_dim);
    ctx_out = ggml_add(g, ctx_out, x);

    // Local module: MBConv (expand_ratio=6) + residual
    int mid_ch = ch * 6;
    ggml_tensor* local_out = g_mbconv(g, ctx_out, w.local, ch, mid_ch, 1, false, 1);
    return ggml_add(g, local_out, ctx_out);
}

static const float * run_forward_graph(surya_det_context * ctx,
                                        const float * input_data, int H, int W,
                                        int * out_h, int * out_w) {
    auto& hp = ctx->hp;
    const bool bench = ctx->bench;
    auto t_total_graph = std::chrono::steady_clock::now();

    // The LiteMLA attention in stage 3 is hard to express as a ggml graph.
    // Strategy: use ggml graph for stages 0-2 + decode head (the bottlenecks),
    // and CPU-scalar for stage 3 (which is fast since it's at 38×38).

    // --- Phase 1: ggml graph for stages 0-2 ---
    size_t ctx_size = 1024 * 1024 * 1024; // 1GB context
    struct ggml_init_params params = {ctx_size, nullptr, true};
    ggml_context* g = ggml_init(params);
    if (!g) return nullptr;

    int max_nodes = 4096;

    // Input tensor [W, H, 3]  (ggml uses [ne0=W, ne1=H, ne2=C])
    ggml_tensor* inp = ggml_new_tensor_3d(g, GGML_TYPE_F32, W, H, 3);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // === Stem ===
    ggml_tensor* x = g_conv(g, inp, ctx->stem_in_conv, 3, 3, 3, 2, 1, 1, 1);
    // res0: conv1(hardswish) + conv2(none) + residual
    ggml_tensor* res = g_conv(g, x, ctx->stem_res0_conv1, hp.stem_ch, 3, 3, 1, 1, 1, 1);
    res = g_conv(g, res, ctx->stem_res0_conv2, hp.stem_ch, 3, 3, 1, 1, 1, 0);
    x = ggml_add(g, res, x);

    // === Stage 0 ===
    ggml_tensor* s0 = g_conv(g, x, ctx->stage01[0][0].spatial,
                              hp.stem_ch, 3, 3, 2, 1, 1, 1);
    s0 = g_conv(g, s0, ctx->stage01[0][0].point,
                 hp.stage_ch[0] * 8, 1, 1, 1, 0, 1, 0);
    // Block 1 with residual
    ggml_tensor* s0b1 = g_conv(g, s0, ctx->stage01[0][1].spatial,
                                hp.stage_ch[0], 3, 3, 1, 1, 1, 1);
    s0b1 = g_conv(g, s0b1, ctx->stage01[0][1].point,
                   hp.stage_ch[0] * 4, 1, 1, 1, 0, 1, 0);
    s0 = ggml_add(g, s0b1, s0);

    // === Stage 1 ===
    ggml_tensor* s1 = g_conv(g, s0, ctx->stage01[1][0].spatial,
                              hp.stage_ch[0], 3, 3, 2, 1, 1, 1);
    s1 = g_conv(g, s1, ctx->stage01[1][0].point,
                 hp.stage_ch[0] * 16, 1, 1, 1, 0, 1, 0);
    ggml_tensor* s1b1 = g_conv(g, s1, ctx->stage01[1][1].spatial,
                                hp.stage_ch[1], 3, 3, 1, 1, 1, 1);
    s1b1 = g_conv(g, s1b1, ctx->stage01[1][1].point,
                   hp.stage_ch[1] * 4, 1, 1, 1, 0, 1, 0);
    s1 = ggml_add(g, s1b1, s1);

    // === Stage 2 ===
    ggml_tensor* s2 = g_mbconv(g, s1, ctx->stage2[0], hp.stage_ch[1],
                                hp.stage_ch[1] * 16, 2, false, 1);
    for (int b = 1; b < 7; b++) {
        s2 = g_mbconv(g, s2, ctx->stage2[b], hp.stage_ch[2],
                       hp.stage_ch[2] * 4, 1, true, 1);
    }

    // Mark s0, s1, s2 as outputs (needed for decode head)
    ggml_set_name(s0, "stage_0");
    ggml_set_name(s1, "stage_1");
    ggml_set_name(s2, "stage_2");
    ggml_set_output(s0);
    ggml_set_output(s1);
    ggml_set_output(s2);

    // Build and compute graph for stages 0-2
    ggml_cgraph* gf = ggml_new_graph_custom(g, max_nodes, false);
    ggml_build_forward_expand(gf, s2);
    // Also need s0, s1 in the graph
    ggml_build_forward_expand(gf, s0);
    ggml_build_forward_expand(gf, s1);

    if (!ggml_gallocr_alloc_graph(ctx->galloc, gf)) {
        fprintf(stderr, "surya_det: graph allocation failed\n");
        ggml_free(g);
        return nullptr;
    }

    // Set input data
    ggml_tensor* inp_t = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(inp_t, input_data, 0, 3 * H * W * sizeof(float));

    // Compute stages 0-2
    auto t0 = std::chrono::steady_clock::now();
    ggml_backend_graph_compute(ctx->backend, gf);
    auto t1 = std::chrono::steady_clock::now();
    double enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (ctx->dump) fprintf(stderr, "surya_det: stages 0-2 graph compute: %.1f ms\n", enc_ms);
    if (bench) fprintf(stderr, "[surya_det-bench] stages 0-2: %.1f ms\n", enc_ms);

    // Read stage outputs
    ggml_tensor* s0_out = ggml_graph_get_tensor(gf, "stage_0");
    ggml_tensor* s1_out = ggml_graph_get_tensor(gf, "stage_1");
    ggml_tensor* s2_out = ggml_graph_get_tensor(gf, "stage_2");

    int s0_H = (int)s0_out->ne[1], s0_W = (int)s0_out->ne[0];
    int s1_H = (int)s1_out->ne[1], s1_W = (int)s1_out->ne[0];
    int s2_H = (int)s2_out->ne[1], s2_W = (int)s2_out->ne[0];
    int s0_C = (int)s0_out->ne[2], s1_C = (int)s1_out->ne[2], s2_C = (int)s2_out->ne[2];

    // Copy stage outputs to CPU vectors
    std::vector<float> s0_data(s0_C * s0_H * s0_W);
    std::vector<float> s1_data(s1_C * s1_H * s1_W);
    std::vector<float> s2_data(s2_C * s2_H * s2_W);
    ggml_backend_tensor_get(s0_out, s0_data.data(), 0, s0_data.size() * sizeof(float));
    ggml_backend_tensor_get(s1_out, s1_data.data(), 0, s1_data.size() * sizeof(float));
    ggml_backend_tensor_get(s2_out, s2_data.data(), 0, s2_data.size() * sizeof(float));

    ggml_free(g);

    if (ctx->dump) {
        dump_stats("stage_0 (graph)", s0_data.data(), s0_data.size());
        dump_stats("stage_1 (graph)", s1_data.data(), s1_data.size());
        dump_stats("stage_2 (graph)", s2_data.data(), s2_data.size());
    }

    // --- Phase 2: stage3 block0 (MBConv) + 6 EfficientViT blocks (LiteMLA+MBConv) ---
    {
        struct ggml_init_params p2 = {16 * 1024 * 1024, nullptr, true};
        ggml_context* g2 = ggml_init(p2);
        ggml_tensor* s2_inp = ggml_new_tensor_3d(g2, GGML_TYPE_F32, s2_W, s2_H, s2_C);
        ggml_set_name(s2_inp, "s2_in");
        ggml_set_input(s2_inp);

        // Stage3 block0: stride-2 MBConv 256→6144→512
        ggml_tensor* s3 = g_mbconv(g2, s2_inp, ctx->stage3_block0,
                                    hp.stage_ch[2], hp.stage_ch[2] * 24,
                                    2, false, 1);

        // 6 EfficientViT blocks (LiteMLA context + MBConv local, both with residual)
        int ch = hp.stage_ch[3];
        for (int b = 0; b < 6; b++) {
            s3 = g_evitvit_block(g2, s3, ctx->stage3_vit[b], ch, hp.head_dim);
        }
        ggml_set_name(s3, "s3_out");
        ggml_set_output(s3);

        ggml_cgraph* gf2 = ggml_new_graph_custom(g2, 2048, false);
        ggml_build_forward_expand(gf2, s3);

        if (!ggml_gallocr_alloc_graph(ctx->galloc, gf2)) {
            fprintf(stderr, "surya_det: stage3 graph alloc failed\n");
            ggml_free(g2);
            return nullptr;
        }

        ggml_tensor* s2_t = ggml_graph_get_tensor(gf2, "s2_in");
        ggml_backend_tensor_set(s2_t, s2_data.data(), 0, s2_data.size() * sizeof(float));

        auto t2 = std::chrono::steady_clock::now();
        ggml_backend_graph_compute(ctx->backend, gf2);
        auto t3 = std::chrono::steady_clock::now();
        if (bench) fprintf(stderr, "[surya_det-bench] stage3 (MBConv+6×ViT ggml): %.1f ms\n",
                           std::chrono::duration<double, std::milli>(t3 - t2).count());

        ggml_tensor* s3_t = ggml_graph_get_tensor(gf2, "s3_out");
        int sH = (int)s3_t->ne[1], sW = (int)s3_t->ne[0];
        int s3_ch = (int)s3_t->ne[2];
        std::vector<float> s3_data(s3_ch * sH * sW);
        ggml_backend_tensor_get(s3_t, s3_data.data(), 0, s3_data.size() * sizeof(float));

        ggml_free(g2);

        // --- Phase 3: decode head ---
        int stage_H[4] = { s0_H, s1_H, s2_H, sH };
        int stage_W[4] = { s0_W, s1_W, s2_W, sW };
        std::vector<float> stage_outputs[4];
        stage_outputs[0] = std::move(s0_data);
        stage_outputs[1] = std::move(s1_data);
        stage_outputs[2] = std::move(s2_data);
        stage_outputs[3] = std::move(s3_data);

        auto t_dec = std::chrono::steady_clock::now();
        auto heatmap = decode_head_fwd(ctx, stage_outputs, stage_H, stage_W);
        if (bench) {
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_dec).count();
            fprintf(stderr, "[surya_det-bench] decode head: %.1f ms\n", ms);
        }

        int out_hm_h = stage_H[0], out_hm_w = stage_W[0];
        ctx->heatmap = std::move(heatmap);
        ctx->heatmap_h = out_hm_h;
        ctx->heatmap_w = out_hm_w;

        if (ctx->dump) dump_stats("heatmap (graph)", ctx->heatmap.data(), 2 * out_hm_h * out_hm_w);

        if (bench) {
            double total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_total_graph).count();
            fprintf(stderr, "[surya_det-bench] total: %.1f ms\n", total_ms);
        }

        if (out_h) *out_h = out_hm_h;
        if (out_w) *out_w = out_hm_w;
        return ctx->heatmap.data();
    }

}

// ---------------------------------------------------------------------------
// Core forward pass (shared between detect and detect_raw)
// ---------------------------------------------------------------------------
static const float * run_forward(surya_det_context * ctx,
                                  const float * input_data, int H, int W,
                                  int * out_h, int * out_w) {
    // Use graph-accelerated path unless SURYA_DET_SCALAR is set
    if (!getenv("SURYA_DET_SCALAR")) {
        return run_forward_graph(ctx, input_data, H, W, out_h, out_w);
    }

    auto& hp = ctx->hp;
    bool dump = ctx->dump;
    const bool bench = ctx->bench;
    auto t_total_scalar = std::chrono::steady_clock::now();

    if (dump) {
        fprintf(stderr, "surya_det: input [3, %d, %d]\n", H, W);
        dump_stats("input", input_data, 3 * H * W);
    }

    auto t_stages012 = std::chrono::steady_clock::now();
    // === Stem ===
    // in_conv: Conv3x3 s2 + hardswish
    auto stem = apply_conv(input_data, ctx->stem_in_conv, 3, H, W,
                            hp.stem_ch, 3, 3, 2, 1, 1, 1 /*hardswish*/);
    H /= 2; W /= 2; // 600x600

    // res0: ConvBlock (conv1 + conv2) with residual
    auto r1 = apply_conv(stem.data(), ctx->stem_res0_conv1, hp.stem_ch, H, W,
                          hp.stem_ch, 3, 3, 1, 1, 1, 1 /*hardswish*/);
    auto r2 = apply_conv(r1.data(), ctx->stem_res0_conv2, hp.stem_ch, H, W,
                          hp.stem_ch, 3, 3, 1, 1, 1, 0 /*none*/);
    // Residual
    int n = hp.stem_ch * H * W;
    for (int i = 0; i < n; i++) r2[i] += stem[i];
    stem = std::move(r2);

    if (dump) dump_stats("stem", stem.data(), hp.stem_ch * H * W);

    // === Stage 0 ===
    int ch = hp.stage_ch[0]; // 64
    // Block 0: stride=2, no residual (in_ch=32 != out_ch=64)
    auto s0 = apply_conv(stem.data(), ctx->stage01[0][0].spatial,
                          hp.stem_ch, H, W, ch * 8, 3, 3, 2, 1, 1, 1);
    H /= 2; W /= 2; // 300x300
    s0 = apply_conv(s0.data(), ctx->stage01[0][0].point,
                     ch * 8, H, W, ch, 1, 1, 1, 0, 1, 0);

    // Block 1: stride=1, residual
    auto s0b1 = apply_conv(s0.data(), ctx->stage01[0][1].spatial,
                            ch, H, W, ch * 4, 3, 3, 1, 1, 1, 1);
    s0b1 = apply_conv(s0b1.data(), ctx->stage01[0][1].point,
                       ch * 4, H, W, ch, 1, 1, 1, 0, 1, 0);
    n = ch * H * W;
    for (int i = 0; i < n; i++) s0b1[i] += s0[i]; // residual
    s0 = std::move(s0b1);

    int stage_H[4], stage_W[4];
    stage_H[0] = H; stage_W[0] = W;
    if (dump) dump_stats("stage_0", s0.data(), ch * H * W);

    // === Stage 1 ===
    ch = hp.stage_ch[1]; // 128
    int prev_ch = hp.stage_ch[0]; // 64
    // Block 0: stride=2, no residual
    auto s1 = apply_conv(s0.data(), ctx->stage01[1][0].spatial,
                          prev_ch, H, W, prev_ch * 16, 3, 3, 2, 1, 1, 1);
    H /= 2; W /= 2; // 150x150
    s1 = apply_conv(s1.data(), ctx->stage01[1][0].point,
                     prev_ch * 16, H, W, ch, 1, 1, 1, 0, 1, 0);

    // Block 1: stride=1, residual
    auto s1b1 = apply_conv(s1.data(), ctx->stage01[1][1].spatial,
                            ch, H, W, ch * 4, 3, 3, 1, 1, 1, 1);
    s1b1 = apply_conv(s1b1.data(), ctx->stage01[1][1].point,
                       ch * 4, H, W, ch, 1, 1, 1, 0, 1, 0);
    n = ch * H * W;
    for (int i = 0; i < n; i++) s1b1[i] += s1[i];
    s1 = std::move(s1b1);

    stage_H[1] = H; stage_W[1] = W;
    if (dump) dump_stats("stage_1", s1.data(), ch * H * W);

    // === Stage 2 ===
    prev_ch = hp.stage_ch[1]; // 128
    ch = hp.stage_ch[2]; // 256

    // Block 0: MBConv stride=2, no residual (128 → 256)
    // Stage 2 uses Hardswish activation (fewer_norm=true for i>=2)
    auto s2 = mbconv_fwd(s1.data(), ctx->stage2[0], prev_ch, H, W,
                          prev_ch * 16, ch, 2, false, 1 /*hardswish*/);
    H /= 2; W /= 2; // 75x75

    // Blocks 1-6: MBConv stride=1, residual
    for (int b = 1; b < 7; b++) {
        s2 = mbconv_fwd(s2.data(), ctx->stage2[b], ch, H, W,
                          ch * 4, ch, 1, true, 1 /*hardswish*/);
    }

    stage_H[2] = H; stage_W[2] = W;
    if (dump) dump_stats("stage_2", s2.data(), ch * H * W);
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_stages012).count();
        fprintf(stderr, "[surya_det-bench] stages 0-2 (scalar): %.1f ms\n", ms);
    }

    // === Stage 3 ===
    prev_ch = hp.stage_ch[2]; // 256
    ch = hp.stage_ch[3]; // 512

    // Block 0: MBConv stride=2, no residual (256 → 512)
    // Stage 3 uses Hardswish activation (not ReLU6)
    auto s3 = mbconv_fwd(s2.data(), ctx->stage3_block0, prev_ch, H, W,
                          prev_ch * 24, ch, 2, false, 1 /*hardswish*/);
    // MBConv stride-2: inverted 1x1 keeps H,W, then depthwise 3x3 stride=2 pad=1:
    // oH = (H + 2*1 - 3) / 2 + 1
    H = (H + 2 - 3) / 2 + 1;  // 75 → 38
    W = (W + 2 - 3) / 2 + 1;  // 75 → 38

    if (dump) dump_stats("stage_3_block_0", s3.data(), ch * H * W);
    {
        auto t_s3b0 = std::chrono::steady_clock::now();
        // Timing already captured above; emit here for stage3 block0 scalar
        if (bench) {
            double ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_s3b0).count();
            (void)ms; // block0 was timed as part of the MBConv call above
        }
    }

    // Blocks 1-6: EfficientVitBlock (LiteMLA + MBConv, both with residual)
    auto t_vit_scalar = std::chrono::steady_clock::now();
    for (int b = 0; b < 6; b++) {
        bool dump_block = dump && (b == 0); // dump internals for first block only
        char blk_pfx[32];
        snprintf(blk_pfx, sizeof(blk_pfx), "b%d", b + 1);
        s3 = evitvit_block_fwd(s3.data(), ctx->stage3_vit[b], ch, H, W, hp.head_dim,
                                dump_block, blk_pfx);
        if (dump) {
            char name[64];
            snprintf(name, sizeof(name), "stage_3_block_%d", b + 1);
            dump_stats(name, s3.data(), ch * H * W);
        }
    }
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_vit_scalar).count();
        fprintf(stderr, "[surya_det-bench] ViT blocks (LiteMLA x6, scalar): %.1f ms\n", ms);
    }

    stage_H[3] = H; stage_W[3] = W;
    if (dump) dump_stats("stage_3", s3.data(), ch * H * W);

    // === Decode head ===
    std::vector<float> stage_outputs[4];
    stage_outputs[0] = std::move(s0);
    stage_outputs[1] = std::move(s1);
    stage_outputs[2] = std::move(s2);
    stage_outputs[3] = std::move(s3);

    auto t_dec_scalar = std::chrono::steady_clock::now();
    auto heatmap = decode_head_fwd(ctx, stage_outputs, stage_H, stage_W);
    if (bench) {
        double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_dec_scalar).count();
        fprintf(stderr, "[surya_det-bench] decode head (scalar): %.1f ms\n", ms);
    }

    int out_hm_h = stage_H[0], out_hm_w = stage_W[0]; // 300x300
    ctx->heatmap = std::move(heatmap);
    ctx->heatmap_h = out_hm_h;
    ctx->heatmap_w = out_hm_w;

    if (dump) dump_stats("heatmap", ctx->heatmap.data(), 2 * out_hm_h * out_hm_w);

    if (bench) {
        double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_total_scalar).count();
        fprintf(stderr, "[surya_det-bench] total: %.1f ms\n", total_ms);
    }

    if (out_h) *out_h = out_hm_h;
    if (out_w) *out_w = out_hm_w;
    return ctx->heatmap.data();
}

// ---------------------------------------------------------------------------
// Public API entry points
// ---------------------------------------------------------------------------
const float * surya_det_detect(surya_det_context * ctx,
                                const uint8_t * pixels, int width, int height, int channels,
                                int * out_h, int * out_w) {
    if (!ctx || !pixels) return nullptr;
    auto input = preprocess_image(pixels, width, height, channels,
                                   ctx->hp.input_h, ctx->hp.input_w);
    return run_forward(ctx, input.data(), ctx->hp.input_h, ctx->hp.input_w, out_h, out_w);
}

const float * surya_det_detect_raw(surya_det_context * ctx,
                                    const float * preprocessed, int H, int W,
                                    int * out_h, int * out_w) {
    if (!ctx || !preprocessed) return nullptr;
    return run_forward(ctx, preprocessed, H, W, out_h, out_w);
}

const float * surya_det_get_heatmap(surya_det_context * ctx, int * out_h, int * out_w) {
    if (!ctx || ctx->heatmap.empty()) return nullptr;
    if (out_h) *out_h = ctx->heatmap_h;
    if (out_w) *out_w = ctx->heatmap_w;
    return ctx->heatmap.data();
}

// ---------------------------------------------------------------------------
// Heatmap → bounding box extraction
// ---------------------------------------------------------------------------
// Port of surya's detect_boxes: dynamic thresholds, connected components,
// morphological expansion, axis-aligned bounding boxes.
const surya_det_bbox * surya_det_get_boxes(surya_det_context * ctx,
                                            int orig_w, int orig_h,
                                            float text_threshold, float low_threshold,
                                            int * n_boxes) {
    if (!ctx || ctx->heatmap.empty()) { if (n_boxes) *n_boxes = 0; return nullptr; }

    int map_h = ctx->heatmap_h, map_w = ctx->heatmap_w;
    // Channel 0 is the text line heatmap
    const float* linemap = ctx->heatmap.data();
    int map_n = map_h * map_w;

    // Dynamic thresholds (from surya)
    {
        // Top 10% intensity average
        std::vector<float> flat(linemap, linemap + map_n);
        std::nth_element(flat.begin(), flat.begin() + (int)(map_n * 0.9f), flat.end());
        float avg_top10 = 0;
        int top_count = 0;
        for (int i = (int)(map_n * 0.9f); i < map_n; i++) {
            avg_top10 += flat[i];
            top_count++;
        }
        if (top_count > 0) avg_top10 /= top_count;

        float scale = std::min(1.0f, powf(avg_top10 / 0.7f, 0.5f));
        scale = std::max(0.0f, scale);
        low_threshold = std::max(0.1f, std::min(0.6f, low_threshold * scale));
        text_threshold = std::max(0.15f, std::min(0.8f, text_threshold * scale));
    }

    // Binarize
    std::vector<uint8_t> binary(map_n, 0);
    for (int i = 0; i < map_n; i++) {
        if (linemap[i] > low_threshold) binary[i] = 1;
    }

    // Connected components (4-connected flood fill)
    std::vector<int> labels(map_n, 0);
    int next_label = 1;
    std::vector<int> stack;

    struct ComponentInfo {
        int min_x, max_x, min_y, max_y;
        float sum_prob;
        int count;
    };
    std::vector<ComponentInfo> components;

    for (int y = 0; y < map_h; y++) {
        for (int x = 0; x < map_w; x++) {
            int idx = y * map_w + x;
            if (!binary[idx] || labels[idx]) continue;

            int label = next_label++;
            stack.clear();
            stack.push_back(idx);
            labels[idx] = label;

            ComponentInfo ci = {x, x, y, y, 0, 0};

            while (!stack.empty()) {
                int cur = stack.back(); stack.pop_back();
                int cx = cur % map_w;
                int cy = cur / map_w;

                ci.sum_prob += linemap[cur];
                ci.count++;
                ci.min_x = std::min(ci.min_x, cx);
                ci.max_x = std::max(ci.max_x, cx);
                ci.min_y = std::min(ci.min_y, cy);
                ci.max_y = std::max(ci.max_y, cy);

                const int dx[] = {-1, 1, 0, 0};
                const int dy[] = {0, 0, -1, 1};
                for (int d = 0; d < 4; d++) {
                    int nx = cx + dx[d], ny = cy + dy[d];
                    if (nx >= 0 && nx < map_w && ny >= 0 && ny < map_h) {
                        int ni = ny * map_w + nx;
                        if (binary[ni] && !labels[ni]) {
                            labels[ni] = label;
                            stack.push_back(ni);
                        }
                    }
                }
            }
            components.push_back(ci);
        }
    }

    // Extract boxes
    ctx->boxes.clear();
    float scale_x = (float)orig_w / map_w;
    float scale_y = (float)orig_h / map_h;
    float max_conf = 0;

    for (auto& ci : components) {
        if (ci.count < 10) continue;

        float line_max = 0;
        // Scan the component region for max
        for (int y = ci.min_y; y <= ci.max_y; y++) {
            for (int x = ci.min_x; x <= ci.max_x; x++) {
                int idx = y * map_w + x;
                if (linemap[idx] > line_max) line_max = linemap[idx];
            }
        }
        if (line_max < text_threshold) continue;

        // Expand box (morphological dilation approximation)
        int w = ci.max_x - ci.min_x + 1;
        int h = ci.max_y - ci.min_y + 1;
        int niter = (int)sqrtf((float)std::min(w, h));
        int buffer = 1 + niter;

        float x0 = std::max(0.0f, (float)(ci.min_x - buffer)) * scale_x;
        float y0 = std::max(0.0f, (float)(ci.min_y - buffer)) * scale_y;
        float x1 = std::min((float)(map_w - 1), (float)(ci.max_x + buffer)) * scale_x;
        float y1 = std::min((float)(map_h - 1), (float)(ci.max_y + buffer)) * scale_y;

        max_conf = std::max(max_conf, line_max);
        ctx->boxes.push_back({x0, y0, x1, y1, line_max});
    }

    // Normalize confidences
    if (max_conf > 0) {
        for (auto& b : ctx->boxes) b.confidence /= max_conf;
    }

    if (n_boxes) *n_boxes = (int)ctx->boxes.size();
    return ctx->boxes.empty() ? nullptr : ctx->boxes.data();
}

const float * surya_det_get_debug(surya_det_context * ctx, const char * name,
                                   int * n_elements) {
    if (!ctx) return nullptr;
    auto it = ctx->debug_tensors.find(name);
    if (it == ctx->debug_tensors.end()) return nullptr;
    if (n_elements) *n_elements = (int)it->second.size();
    return it->second.data();
}
