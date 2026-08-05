// deepseek_ocr2.cpp — DeepSeek-OCR-2 engine: SAM ViT-B + Qwen2 encoder + MoE decoder.
//
// Vision: per-layer ggml graph with CPU window partition (SAM ViT-B pattern).
// Qwen2 encoder: CPU-scalar bidirectional transformer (no causal mask).
// LLM decoder: ggml graph with KV cache, MoE layers use CPU-scalar expert dispatch.

#include "deepseek_ocr2.h"
#include "crispembed_diff.h"
#include "core/gguf_loader.h"
#include "core/bpe.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/gpu_backend_pref.h"
#include "core/no_repeat_ngram.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <thread>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

struct sam_hparams {
    int depth = 12, hidden = 768, heads = 12, head_dim = 64;
    int patch_size = 16, image_size = 1024, window_size = 14;
    int neck_out = 256;
    std::vector<int> global_attn_indexes{ 2, 5, 8, 11 };
    // DeepSeek-OCR2 BasicImageTransform: simple [-1,1] normalization (mean=std=0.5),
    // NOT CLIP normalization. (processor_config.json image_mean/std = 0.5.)
    float image_mean[3] = { 0.5f, 0.5f, 0.5f };
    float image_std[3] = { 0.5f, 0.5f, 0.5f };
};

struct qwen2_enc_hparams {
    int depth = 24, hidden = 896, heads = 14, kv_heads = 2;
    int intermediate = 4864;
    float rms_eps = 1e-6f;
    float rope_theta = 1000000.0f;
};

struct llm_hparams {
    int vocab_size = 129280, hidden = 1280, heads = 10, kv_heads = 10;
    int head_dim = 128, n_layers = 12;
    int dense_intermediate = 6848;  // layer 0
    int expert_intermediate = 896;  // routed experts
    int shared_intermediate = 1792; // shared experts (896*2)
    int n_experts = 64, n_experts_top = 6, n_shared_experts = 2;
    float rms_eps = 1e-6f, rope_theta = 10000.0f;
    float routed_scaling_factor = 1.0f;
    int eos_token_id = 1;
    int max_position_embeddings = 4096;
};

// ---------------------------------------------------------------------------
// Weight storage
// ---------------------------------------------------------------------------

struct sam_block_w {
    ggml_tensor *ln1_w{}, *ln1_b{}, *ln2_w{}, *ln2_b{};
    ggml_tensor *qkv_w{}, *qkv_b{}, *proj_w{}, *proj_b{};
    ggml_tensor *rel_pos_h{}, *rel_pos_w{};
    ggml_tensor *ffn_up_w{}, *ffn_up_b{}, *ffn_down_w{}, *ffn_down_b{};
    bool is_global = false;
};

struct qwen2_enc_layer_w {
    ggml_tensor *in_ln_w{}, *post_ln_w{};
    ggml_tensor *q_w{}, *q_b{}, *k_w{}, *k_b{}, *v_w{}, *v_b{}, *o_w{};
    ggml_tensor *gate_w{}, *up_w{}, *down_w{};
};

struct moe_expert_w {
    ggml_tensor *gate_w{}, *up_w{}, *down_w{};
};

struct llm_layer_w {
    ggml_tensor *in_ln_w{}, *post_ln_w{};
    ggml_tensor *q_w{}, *k_w{}, *v_w{}, *o_w{};
    // Dense FFN (layer 0)
    ggml_tensor *ffn_gate_w{}, *ffn_up_w{}, *ffn_down_w{};
    // MoE (layers 1-11)
    ggml_tensor * router_w{}; // mlp.gate.weight
    std::vector<moe_expert_w> experts;
    moe_expert_w shared_experts[2];
    // Single shared expert (combined)
    ggml_tensor *shared_gate_w{}, *shared_up_w{}, *shared_down_w{};
    // Experts stacked as [in, out, n_exp] for ggml_mul_mat_id (Metal MoE path).
    // Built at load by stack_moe_experts() when the graph MoE path is active.
    ggml_tensor *gate_exps{}, *up_exps{}, *down_exps{};
};

struct model_weights {
    sam_hparams shp;
    qwen2_enc_hparams qhp;
    llm_hparams lhp;

    // SAM
    ggml_tensor *patch_embed_w{}, *patch_embed_b{}, *pos_embed{};
    std::vector<sam_block_w> sam_blocks;
    ggml_tensor *neck_conv1_w{}, *neck_ln1_w{}, *neck_ln1_b{};
    ggml_tensor *neck_conv2_w{}, *neck_ln2_w{}, *neck_ln2_b{};
    ggml_tensor *net_2_w{}, *net_3_w{};

    // Qwen2 encoder
    std::vector<qwen2_enc_layer_w> qwen2_layers;
    ggml_tensor *query_768{}, *query_1024{}, *qe_output_norm{};

    // Projector
    ggml_tensor *projector_w{}, *projector_b{};

    // View separator
    ggml_tensor * view_separator{};

    // LLM
    ggml_tensor *embed_tokens{}, *output_norm_w{}, *lm_head_w{};
    std::vector<llm_layer_w> llm_layers;
};

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct ds_ocr2_ctx {
    model_weights m;
    ggml_context * model_ctx{};        // alias into model_wl (do not free separately)
    ggml_backend_buffer_t model_buf{}; // alias into model_wl
    core_gguf::WeightLoad model_wl;    // owns ctx/buf (+ the mmap on the no-copy path)
    ggml_backend_t backend{}, backend_cpu{};
    ggml_backend_sched_t sched{};
    std::vector<uint8_t> compute_meta;

    // Stacked MoE expert weights ([in,out,n_exp]) for the ggml_mul_mat_id path.
    ggml_context * moe_ctx{};
    ggml_backend_buffer_t moe_buf{};
    bool moe_metal = false;        // true once experts are stacked + the graph path is on
    bool moe_prestacked = false;   // GGUF already ships ffn_*_exps (converter #4) → skip runtime stacking
    ggml_context * moe_view_ctx{}; // per-expert views into prestacked tensors (DS_MOE_CPU fallback only)

    // Tokenizer
    std::vector<std::string> id_to_piece;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
    int tok_vocab_size = 0;

    // KV cache for LLM decoder — persistent device-side tensors.
    // Layout: [kv_dim, max_seq, n_layers] where kv_dim = kv_heads * head_dim.
    struct {
        ggml_context * ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor * k = nullptr; // [kv_dim, max_seq, n_layers]
        ggml_tensor * v = nullptr;
        int max_seq = 0;
        int n_past = 0;
        bool allocated = false;
    } kvc;

    // Precomputed RPE tables (default grid = image_size/patch_size)
    std::vector<std::vector<float>> rp_h_per_layer, rp_w_per_layer;

    // Crop-mode (DS2_CROP_MODE) lazily-built caches for the 768² tile grid:
    // global-attn RPE tables at the tile grid and the bicubic-resampled SAM
    // position embedding. grid==0 means not built yet.
    std::vector<std::vector<float>> rp_h_crop, rp_w_crop;
    int rp_crop_grid = 0;
    std::vector<float> pos_embed_crop;
    int pos_crop_grid = 0;

    int n_threads = 4, verbosity = 1;
    std::string diff_ref_path;
};

// Forward declarations for KV cache management
static void free_ds_kv_cache(ds_ocr2_ctx & c);
static bool alloc_ds_kv_cache(ds_ocr2_ctx & c, int max_seq);

// ---------------------------------------------------------------------------
// CPU helpers (shared with got_ocr.cpp pattern)
// ---------------------------------------------------------------------------

static std::vector<float> to_f32(const ggml_tensor * t) {
    if (!t) return {};
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    // Read the raw bytes through the tensor's backend buffer. A weight tensor
    // resident on CUDA (or any non-host backend) has a DEVICE pointer in
    // t->data that must NOT be dereferenced on the host — doing so segfaults.
    // ggml_backend_tensor_get performs the correct device->host copy; only fall
    // back to a direct read for buffer-less host tensors (e.g. scratch tensors).
    size_t nb = ggml_nbytes(t);
    std::vector<uint8_t> raw(nb);
    const void * src_bytes;
    if (t->buffer) {
        ggml_backend_tensor_get(t, raw.data(), 0, nb);
        src_bytes = raw.data();
    } else {
        src_bytes = t->data;
    }
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), src_bytes, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * src = (const ggml_fp16_t *)src_bytes;
        for (int i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float)
            traits->to_float(src_bytes, out.data(), n);
        else
            memset(out.data(), 0, n * sizeof(float));
    }
    return out;
}

static void layernorm_cpu(const float * in, float * out, int D, const float * w, const float * b, float eps = 1e-6f) {
    double mean = 0;
    for (int i = 0; i < D; i++) mean += in[i];
    mean /= D;
    double var = 0;
    for (int i = 0; i < D; i++) {
        double d = in[i] - mean;
        var += d * d;
    }
    var /= D;
    float s = 1.0f / sqrtf((float)var + eps);
    for (int i = 0; i < D; i++) out[i] = ((in[i] - (float)mean) * s) * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
}

static void layernorm2d_cpu(const float * in, float * out, int C, int H, int W, const float * w, const float * b,
                            float eps = 1e-6f) {
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            double mean = 0;
            for (int c = 0; c < C; c++) mean += in[c * H * W + y * W + x];
            mean /= C;
            double var = 0;
            for (int c = 0; c < C; c++) {
                double d = in[c * H * W + y * W + x] - mean;
                var += d * d;
            }
            var /= C;
            float s = 1.0f / sqrtf((float)var + eps);
            for (int c = 0; c < C; c++) {
                float v = (in[c * H * W + y * W + x] - (float)mean) * s;
                out[c * H * W + y * W + x] = v * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);
            }
        }
}

static void rmsnorm_cpu(const float * in, float * out, int D, const float * w, float eps = 1e-6f) {
    double ss = 0;
    for (int i = 0; i < D; i++) ss += (double)in[i] * in[i];
    float s = 1.0f / sqrtf((float)(ss / D) + eps);
    for (int i = 0; i < D; i++) out[i] = in[i] * s * (w ? w[i] : 1.0f);
}

static void linear_cpu(const float * in, float * out, int in_dim, int out_dim, const float * w, const float * b) {
    for (int o = 0; o < out_dim; o++) {
        float s = b ? b[o] : 0.0f;
        for (int i = 0; i < in_dim; i++) s += in[i] * w[o * in_dim + i];
        out[o] = s;
    }
}

static void conv2d_cpu(const float * in, float * out, const float * weight, const float * bias, int in_ch, int out_ch,
                       int H, int W, int kh, int kw, int stride, int pad, int n_threads = 1) {
    int oH = (H + 2 * pad - kh) / stride + 1;
    int oW = (W + 2 * pad - kw) / stride + 1;
    // Each output channel writes its own plane → parallelize over oc. This is
    // the SAM neck/downsample hot path (~10 s scalar); threading it is exact.
    auto plane = [&](int oc0, int oc1) {
        for (int oc = oc0; oc < oc1; oc++) {
            float b = bias ? bias[oc] : 0.0f;
            for (int oy = 0; oy < oH; oy++)
                for (int ox = 0; ox < oW; ox++) {
                    float sum = b;
                    for (int ic = 0; ic < in_ch; ic++)
                        for (int ky2 = 0; ky2 < kh; ky2++)
                            for (int kx2 = 0; kx2 < kw; kx2++) {
                                int iy = oy * stride - pad + ky2;
                                int ix = ox * stride - pad + kx2;
                                if (iy >= 0 && iy < H && ix >= 0 && ix < W)
                                    sum += in[ic * H * W + iy * W + ix] *
                                           weight[oc * (in_ch * kh * kw) + ic * kh * kw + ky2 * kw + kx2];
                            }
                    out[oc * oH * oW + oy * oW + ox] = sum;
                }
        }
    };
    int nt = std::max(1, std::min(n_threads, out_ch));
    if (nt <= 1) {
        plane(0, out_ch);
        return;
    }
    std::vector<std::thread> pool;
    int chunk = (out_ch + nt - 1) / nt;
    for (int t = 0; t < nt; t++) {
        int o0 = t * chunk, o1 = std::min(out_ch, o0 + chunk);
        if (o0 < o1) pool.emplace_back(plane, o0, o1);
    }
    for (auto & th : pool) th.join();
}

static void silu_cpu(float * x, int n) {
    for (int i = 0; i < n; i++) x[i] = x[i] / (1.0f + expf(-x[i]));
}

static void swiglu_ffn_cpu(const float * in, float * out, int D, int inter, const float * gate_w, const float * up_w,
                           const float * down_w) {
    std::vector<float> gate(inter), up(inter);
    linear_cpu(in, gate.data(), D, inter, gate_w, nullptr);
    linear_cpu(in, up.data(), D, inter, up_w, nullptr);
    silu_cpu(gate.data(), inter);
    for (int i = 0; i < inter; i++) gate[i] *= up[i];
    linear_cpu(gate.data(), out, inter, D, down_w, nullptr);
}

// ---------------------------------------------------------------------------
// SAM window partition / unpartition + RPE (same as got_ocr.cpp)
// ---------------------------------------------------------------------------

static void window_partition(const float * h, float * wo, int nP, int ws, int C) {
    int pad_h = (ws - nP % ws) % ws, pad_w = (ws - nP % ws) % ws;
    int pH = nP + pad_h, pW = nP + pad_w;
    int nWh = pH / ws, nWw = pW / ws, wN = ws * ws;
    memset(wo, 0, (size_t)nWh * nWw * wN * C * sizeof(float));
    for (int wh = 0; wh < nWh; wh++)
        for (int ww = 0; ww < nWw; ww++) {
            int wi = wh * nWw + ww;
            for (int y = 0; y < ws; y++) {
                int sy = wh * ws + y;
                if (sy >= nP) continue;
                for (int x = 0; x < ws; x++) {
                    int sx = ww * ws + x;
                    if (sx >= nP) continue;
                    memcpy(wo + (wi * wN + y * ws + x) * C, h + (sy * nP + sx) * C, C * sizeof(float));
                }
            }
        }
}

static void window_unpartition(const float * wi, float * h, int nP, int ws, int C) {
    int pad_h = (ws - nP % ws) % ws, pad_w = (ws - nP % ws) % ws;
    int pH = nP + pad_h, pW = nP + pad_w;
    int nWh = pH / ws, nWw = pW / ws, wN = ws * ws;
    for (int wh = 0; wh < nWh; wh++)
        for (int ww = 0; ww < nWw; ww++) {
            int widx = wh * nWw + ww;
            for (int y = 0; y < ws; y++) {
                int sy = wh * ws + y;
                if (sy >= nP) continue;
                for (int x = 0; x < ws; x++) {
                    int sx = ww * ws + x;
                    if (sx >= nP) continue;
                    memcpy(h + (sy * nP + sx) * C, wi + (widx * wN + y * ws + x) * C, C * sizeof(float));
                }
            }
        }
}

static std::vector<float> get_rel_pos(int q_size, int k_size, const float * rel_pos, int L, int hd) {
    int max_rd = 2 * std::max(q_size, k_size) - 1;
    std::vector<float> resized(hd * max_rd);
    for (int c = 0; c < hd; c++)
        for (int i = 0; i < max_rd; i++) {
            // F.interpolate(mode='linear') default align_corners=False:
            // src = (i+0.5)*L/out - 0.5, clamped. The 64-grid tables have
            // L == max_rd (no resize), so only the 48-grid crop tables hit this.
            float src = ((float)i + 0.5f) * (float)L / (float)max_rd - 0.5f;
            src = std::min(std::max(src, 0.0f), (float)(L - 1));
            int lo = (int)src, hi = std::min(lo + 1, L - 1);
            float frac = src - lo;
            resized[i * hd + c] = rel_pos[lo * hd + c] * (1.0f - frac) + rel_pos[hi * hd + c] * frac;
        }
    float qs = std::max((float)k_size / q_size, 1.0f);
    float ks = std::max((float)q_size / k_size, 1.0f);
    float off = (float)(k_size - 1) * qs;
    std::vector<float> result(q_size * k_size * hd);
    for (int qi = 0; qi < q_size; qi++)
        for (int ki = 0; ki < k_size; ki++) {
            int idx = std::max(0, std::min((int)(qi * qs - ki * ks + off), max_rd - 1));
            for (int c = 0; c < hd; c++) result[(qi * k_size + ki) * hd + c] = resized[idx * hd + c];
        }
    return result;
}

static void reformat_rp_table(const float * rp_in, float * rp_out, int aH, int hd) {
    for (int q = 0; q < aH; q++)
        for (int k = 0; k < aH; k++)
            for (int d = 0; d < hd; d++) rp_out[d + k * hd + q * aH * hd] = rp_in[(q * aH + k) * hd + d];
}

// Antialiased separable bicubic 1-D weights (PIL/torch downscale algorithm:
// kernel support and argument scaled by src/dst). `a` selects the cubic:
// torch F.interpolate(mode='bicubic', antialias=True) uses a=-0.75 (the SAM
// pos-embed 64→48 resize); PIL Image.resize(BICUBIC) uses a=-0.5 (the crop
// tile resize in dynamic_preprocess).
struct bicubic_taps {
    std::vector<int> lo;  // first source index per output index
    std::vector<int> n;   // tap count per output index
    std::vector<float> w; // taps, max_taps stride
    int max_taps = 0;
};

static bicubic_taps bicubic_aa_taps(int src, int dst, float a) {
    auto cubic = [a](float x) {
        x = fabsf(x);
        if (x <= 1.0f) return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f;
        if (x < 2.0f) return (((x - 5.0f) * x + 8.0f) * x - 4.0f) * a;
        return 0.0f;
    };
    float fs = std::max((float)src / dst, 1.0f); // filter scale (>=1 downscale)
    float support = 2.0f * fs;
    bicubic_taps t;
    t.max_taps = (int)ceilf(support) * 2 + 2;
    t.lo.resize(dst);
    t.n.resize(dst);
    t.w.assign((size_t)dst * t.max_taps, 0.0f);
    for (int i = 0; i < dst; i++) {
        float center = ((float)i + 0.5f) * src / dst;
        int xmin = std::max(0, (int)(center - support + 0.5f));
        int xmax = std::min(src, (int)(center + support + 0.5f));
        t.lo[i] = xmin;
        t.n[i] = xmax - xmin;
        float sum = 0.0f;
        for (int k = xmin; k < xmax; k++) {
            float wv = cubic(((float)k + 0.5f - center) / fs);
            t.w[(size_t)i * t.max_taps + (k - xmin)] = wv;
            sum += wv;
        }
        if (sum != 0.0f)
            for (int k = 0; k < t.n[i]; k++) t.w[(size_t)i * t.max_taps + k] /= sum;
    }
    return t;
}

// Resample a [srcH x srcW] plane (row-major, stride `stride` floats between
// consecutive samples along W, `row_stride` between rows) into dst (compact).
static void bicubic_aa_resample_plane(const float * in, int srcW, int srcH, float * out, int dstW, int dstH, float a) {
    bicubic_taps tw = bicubic_aa_taps(srcW, dstW, a);
    bicubic_taps th = bicubic_aa_taps(srcH, dstH, a);
    std::vector<float> tmp((size_t)srcH * dstW); // horizontal pass first
    for (int y = 0; y < srcH; y++)
        for (int x = 0; x < dstW; x++) {
            float acc = 0.0f;
            for (int k = 0; k < tw.n[x]; k++)
                acc += tw.w[(size_t)x * tw.max_taps + k] * in[(size_t)y * srcW + tw.lo[x] + k];
            tmp[(size_t)y * dstW + x] = acc;
        }
    for (int y = 0; y < dstH; y++)
        for (int x = 0; x < dstW; x++) {
            float acc = 0.0f;
            for (int k = 0; k < th.n[y]; k++)
                acc += th.w[(size_t)y * th.max_taps + k] * tmp[(size_t)(th.lo[y] + k) * dstW + x];
            out[(size_t)y * dstW + x] = acc;
        }
}

// ---------------------------------------------------------------------------
// ggml graph helpers
// ---------------------------------------------------------------------------

static ggml_tensor * ensure_f32(ggml_context * g, ggml_tensor * t) {
    if (!t || t->type == GGML_TYPE_F32) return t;
    return ggml_cast(g, t, GGML_TYPE_F32);
}

static ggml_tensor * g_ln(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps = 1e-6f) {
    if (!w) return x;
    x = ggml_norm(g, x, eps);
    x = ggml_mul(g, x, ensure_f32(g, w));
    if (b) x = ggml_add(g, x, ensure_f32(g, b));
    return x;
}

static ggml_tensor * g_linear(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    if (!w) return x;
    x = ggml_mul_mat(g, w, x);
    if (b) x = ggml_add(g, x, ensure_f32(g, b));
    return x;
}

// ---------------------------------------------------------------------------
// Load model
// ---------------------------------------------------------------------------

static bool load_hparams(ds_ocr2_ctx & ctx, const char * path) {
    gguf_context * g = core_gguf::open_metadata(path);
    if (!g) return false;

    auto u32 = [&](const char * k, uint32_t d) { return core_gguf::kv_u32(g, k, d); };
    auto f32v = [&](const char * k, float d) { return core_gguf::kv_f32(g, k, d); };

    auto & s = ctx.m.shp;
    s.depth = u32("deepseek_ocr2.sam.depth", s.depth);
    s.hidden = u32("deepseek_ocr2.sam.hidden_size", s.hidden);
    s.heads = u32("deepseek_ocr2.sam.num_heads", s.heads);
    s.head_dim = s.hidden / s.heads;
    s.patch_size = u32("deepseek_ocr2.sam.patch_size", s.patch_size);
    s.image_size = u32("deepseek_ocr2.sam.image_size", s.image_size);
    s.window_size = u32("deepseek_ocr2.sam.window_size", s.window_size);
    s.neck_out = u32("deepseek_ocr2.sam.neck_out_channels", s.neck_out);

    int key_id = gguf_find_key(g, "deepseek_ocr2.sam.global_attn_indexes");
    if (key_id >= 0) {
        int n = (int)gguf_get_arr_n(g, key_id);
        s.global_attn_indexes.resize(n);
        memcpy(s.global_attn_indexes.data(), gguf_get_arr_data(g, key_id), n * sizeof(int32_t));
    }

    key_id = gguf_find_key(g, "deepseek_ocr2.sam.image_mean");
    if (key_id >= 0 && gguf_get_arr_n(g, key_id) >= 3)
        memcpy(s.image_mean, gguf_get_arr_data(g, key_id), 3 * sizeof(float));
    key_id = gguf_find_key(g, "deepseek_ocr2.sam.image_std");
    if (key_id >= 0 && gguf_get_arr_n(g, key_id) >= 3)
        memcpy(s.image_std, gguf_get_arr_data(g, key_id), 3 * sizeof(float));

    auto & q = ctx.m.qhp;
    q.depth = u32("deepseek_ocr2.qwen2_enc.depth", q.depth);
    q.hidden = u32("deepseek_ocr2.qwen2_enc.hidden_size", q.hidden);
    q.heads = u32("deepseek_ocr2.qwen2_enc.num_heads", q.heads);
    q.kv_heads = u32("deepseek_ocr2.qwen2_enc.num_kv_heads", q.kv_heads);
    q.intermediate = u32("deepseek_ocr2.qwen2_enc.intermediate_size", q.intermediate);
    q.rms_eps = f32v("deepseek_ocr2.qwen2_enc.rms_norm_eps", q.rms_eps);

    auto & l = ctx.m.lhp;
    l.vocab_size = u32("deepseek_ocr2.vocab_size", l.vocab_size);
    l.hidden = u32("deepseek_ocr2.hidden_size", l.hidden);
    l.heads = u32("deepseek_ocr2.num_attention_heads", l.heads);
    l.kv_heads = u32("deepseek_ocr2.num_key_value_heads", l.kv_heads);
    l.head_dim = l.hidden / l.heads;
    l.n_layers = u32("deepseek_ocr2.num_hidden_layers", l.n_layers);
    l.dense_intermediate = u32("deepseek_ocr2.dense_intermediate_size", l.dense_intermediate);
    l.expert_intermediate = u32("deepseek_ocr2.expert_intermediate_size", l.expert_intermediate);
    l.shared_intermediate = u32("deepseek_ocr2.shared_intermediate_size", l.shared_intermediate);
    l.n_experts = u32("deepseek_ocr2.n_routed_experts", l.n_experts);
    l.n_experts_top = u32("deepseek_ocr2.num_experts_per_tok", l.n_experts_top);
    l.n_shared_experts = u32("deepseek_ocr2.n_shared_experts", l.n_shared_experts);
    l.rms_eps = f32v("deepseek_ocr2.rms_norm_eps", l.rms_eps);
    l.rope_theta = f32v("deepseek_ocr2.rope_theta", l.rope_theta);
    l.routed_scaling_factor = f32v("deepseek_ocr2.routed_scaling_factor", l.routed_scaling_factor);
    l.eos_token_id = u32("deepseek_ocr2.eos_token_id", l.eos_token_id);

    // Tokenizer
    int vocab_idx = gguf_find_key(g, "tokenizer.ggml.tokens");
    if (vocab_idx >= 0) {
        int n = (int)gguf_get_arr_n(g, vocab_idx);
        ctx.id_to_piece.resize(n);
        ctx.token_to_id.reserve(n * 2);
        for (int i = 0; i < n; i++) {
            ctx.id_to_piece[i] = gguf_get_arr_str(g, vocab_idx, i);
            ctx.token_to_id[ctx.id_to_piece[i]] = i;
        }
        ctx.tok_vocab_size = n;
    }
    int merges_idx = gguf_find_key(g, "tokenizer.ggml.merges");
    if (merges_idx >= 0) {
        int n = (int)gguf_get_arr_n(g, merges_idx);
        ctx.merge_rank.reserve(n * 2);
        for (int i = 0; i < n; i++) ctx.merge_rank[gguf_get_arr_str(g, merges_idx, i)] = i;
    }

    core_gguf::free_metadata(g);
    return true;
}

static bool load_tensors(ds_ocr2_ctx & ctx, const char * path) {
    // DS_MMAP=1 opts into the no-copy mmap load (Metal/CPU unified memory):
    // point the weight buffer at the mmap'd file instead of copying ~2 GB into
    // a fresh buffer — halves resident memory. On Metal the pages still get
    // wired on first use, so it's a memory win more than a first-load-time win;
    // default stays the proven copy path. Falls back automatically if
    // unsupported. Validated equal by tests/test_gguf_loader_mmap.
    bool try_mmap = getenv("DS_MMAP") != nullptr;
    if (!core_gguf::load_weights(path, ctx.backend, "deepseek_ocr2", ctx.model_wl, try_mmap)) return false;

    ctx.model_ctx = ctx.model_wl.ctx;
    ctx.model_buf = ctx.model_wl.buf;
    auto & t = ctx.model_wl.tensors;
    auto F = [&](const char * n) -> ggml_tensor * {
        auto it = t.find(n);
        return it != t.end() ? it->second : nullptr;
    };

    auto & m = ctx.m;
    auto & s = m.shp;

    // SAM
    m.patch_embed_w = F("v.patch_embed.weight");
    m.patch_embed_b = F("v.patch_embed.bias");
    m.pos_embed = F("v.pos_embed");

    m.sam_blocks.resize(s.depth);
    for (int i = 0; i < s.depth; i++) {
        char buf[128];
        auto & blk = m.sam_blocks[i];
        blk.is_global = false;
        for (int gi : s.global_attn_indexes)
            if (gi == i) {
                blk.is_global = true;
                break;
            }
        auto BF = [&](const char * sfx) -> ggml_tensor * {
            snprintf(buf, sizeof(buf), "v.blk.%d.%s", i, sfx);
            return F(buf);
        };
        blk.ln1_w = BF("ln1.weight");
        blk.ln1_b = BF("ln1.bias");
        blk.ln2_w = BF("ln2.weight");
        blk.ln2_b = BF("ln2.bias");
        blk.qkv_w = BF("attn_qkv.weight");
        blk.qkv_b = BF("attn_qkv.bias");
        blk.proj_w = BF("attn_proj.weight");
        blk.proj_b = BF("attn_proj.bias");
        blk.rel_pos_h = BF("attn_rel_pos_h");
        blk.rel_pos_w = BF("attn_rel_pos_w");
        blk.ffn_up_w = BF("ffn_up.weight");
        blk.ffn_up_b = BF("ffn_up.bias");
        blk.ffn_down_w = BF("ffn_down.weight");
        blk.ffn_down_b = BF("ffn_down.bias");
    }

    m.neck_conv1_w = F("v.neck_conv1.weight");
    m.neck_ln1_w = F("v.neck_ln1.weight");
    m.neck_ln1_b = F("v.neck_ln1.bias");
    m.neck_conv2_w = F("v.neck_conv2.weight");
    m.neck_ln2_w = F("v.neck_ln2.weight");
    m.neck_ln2_b = F("v.neck_ln2.bias");
    m.net_2_w = F("v.net_2.weight");
    m.net_3_w = F("v.net_3.weight");

    // Qwen2 encoder
    auto & qhp = m.qhp;
    m.qwen2_layers.resize(qhp.depth);
    for (int i = 0; i < qhp.depth; i++) {
        char buf[128];
        auto & ly = m.qwen2_layers[i];
        auto QF = [&](const char * sfx) -> ggml_tensor * {
            snprintf(buf, sizeof(buf), "qe.blk.%d.%s", i, sfx);
            return F(buf);
        };
        ly.in_ln_w = QF("input_layernorm.weight");
        ly.post_ln_w = QF("post_attention_layernorm.weight");
        ly.q_w = QF("attn_q.weight");
        ly.q_b = QF("attn_q.bias");
        ly.k_w = QF("attn_k.weight");
        ly.k_b = QF("attn_k.bias");
        ly.v_w = QF("attn_v.weight");
        ly.v_b = QF("attn_v.bias");
        ly.o_w = QF("attn_o.weight");
        ly.gate_w = QF("ffn_gate.weight");
        ly.up_w = QF("ffn_up.weight");
        ly.down_w = QF("ffn_down.weight");
    }
    m.query_768 = F("qe.query_768");
    m.query_1024 = F("qe.query_1024");
    m.qe_output_norm = F("qe.output_norm.weight");

    // Projector
    m.projector_w = F("proj.weight");
    m.projector_b = F("proj.bias");

    // View separator
    m.view_separator = F("v.view_separator");

    // LLM
    m.embed_tokens = F("l.embed_tokens.weight");
    m.output_norm_w = F("l.output_norm.weight");
    m.lm_head_w = F("l.lm_head.weight");

    auto & lhp = m.lhp;
    m.llm_layers.resize(lhp.n_layers);
    for (int i = 0; i < lhp.n_layers; i++) {
        char buf[128];
        auto & ly = m.llm_layers[i];
        auto LF = [&](const char * sfx) -> ggml_tensor * {
            snprintf(buf, sizeof(buf), "l.blk.%d.%s", i, sfx);
            return F(buf);
        };
        ly.in_ln_w = LF("input_layernorm.weight");
        ly.post_ln_w = LF("post_attention_layernorm.weight");
        ly.q_w = LF("attn_q.weight");
        ly.k_w = LF("attn_k.weight");
        ly.v_w = LF("attn_v.weight");
        ly.o_w = LF("attn_o.weight");

        if (i == 0) {
            // Dense FFN
            ly.ffn_gate_w = LF("ffn_gate.weight");
            ly.ffn_up_w = LF("ffn_up.weight");
            ly.ffn_down_w = LF("ffn_down.weight");
        } else {
            // MoE
            ly.router_w = LF("mlp_gate.weight");

            // Prefer PRESTACKED experts (converter #4): l.blk.{i}.ffn_{gate,up,down}_exps.weight
            // are [in,out,n_exp] tensors byte-identical to what stack_moe_experts()
            // builds — load them directly (no per-expert copies, no stacking pass, saves
            // ~1.3 GB of duplicated resident weights). Fall back to the per-expert layout
            // for legacy GGUFs.
            ly.gate_exps = LF("ffn_gate_exps.weight");
            ly.up_exps = LF("ffn_up_exps.weight");
            ly.down_exps = LF("ffn_down_exps.weight");
            if (ly.gate_exps && ly.up_exps && ly.down_exps) {
                ctx.moe_prestacked = true;
                // The graph MoE path consumes gate_exps/up_exps/down_exps directly.
                // Only the DS_MOE_CPU fallback needs per-expert tensors — build them as
                // views into the stacked buffer (shared storage, no copy). The view's own
                // ->buffer is null, so set it to the parent's: ggml_backend_tensor_get
                // (used by to_f32) reads via view_src->buffer, but to_f32's fast path gates
                // on ->buffer and would otherwise deref a raw device pointer (Metal segfault).
                if (getenv("DS_MOE_CPU")) {
                    if (!ctx.moe_view_ctx) {
                        ggml_init_params vp = {
                            (size_t)lhp.n_layers * 3 * lhp.n_experts * ggml_tensor_overhead() + 4096, nullptr, true
                        };
                        ctx.moe_view_ctx = ggml_init(vp);
                    }
                    auto mkview = [&](ggml_tensor * st, int e) -> ggml_tensor * {
                        ggml_tensor * v =
                            ggml_view_2d(ctx.moe_view_ctx, st, st->ne[0], st->ne[1], st->nb[1], (size_t)e * st->nb[2]);
                        v->buffer = st->buffer;
                        return v;
                    };
                    ly.experts.resize(lhp.n_experts);
                    for (int j = 0; j < lhp.n_experts; j++) {
                        ly.experts[j].gate_w = mkview(ly.gate_exps, j);
                        ly.experts[j].up_w = mkview(ly.up_exps, j);
                        ly.experts[j].down_w = mkview(ly.down_exps, j);
                    }
                }
            } else {
                ly.experts.resize(lhp.n_experts);
                for (int j = 0; j < lhp.n_experts; j++) {
                    auto EF = [&](const char * sfx) -> ggml_tensor * {
                        snprintf(buf, sizeof(buf), "l.blk.%d.exp.%d.%s", i, j, sfx);
                        return F(buf);
                    };
                    ly.experts[j].gate_w = EF("ffn_gate.weight");
                    ly.experts[j].up_w = EF("ffn_up.weight");
                    ly.experts[j].down_w = EF("ffn_down.weight");
                }
            }
            ly.shared_gate_w = LF("shared_exp.ffn_gate.weight");
            ly.shared_up_w = LF("shared_exp.ffn_up.weight");
            ly.shared_down_w = LF("shared_exp.ffn_down.weight");
        }
    }

    return true;
}

static void precompute_rpe_tables(ds_ocr2_ctx & ctx) {
    auto & s = ctx.m.shp;
    int hd = s.head_dim, nP = s.image_size / s.patch_size, ws = s.window_size;
    ctx.rp_h_per_layer.resize(s.depth);
    ctx.rp_w_per_layer.resize(s.depth);
    for (int li = 0; li < s.depth; li++) {
        auto & blk = ctx.m.sam_blocks[li];
        if (!blk.rel_pos_h || !blk.rel_pos_w) continue;
        auto rph = to_f32(blk.rel_pos_h), rpw = to_f32(blk.rel_pos_w);
        int L_h = (int)blk.rel_pos_h->ne[1], L_w = (int)blk.rel_pos_w->ne[1];
        int aH = blk.is_global ? nP : ws, aW = aH;
        ctx.rp_h_per_layer[li] = get_rel_pos(aH, aH, rph.data(), L_h, hd);
        ctx.rp_w_per_layer[li] = get_rel_pos(aW, aW, rpw.data(), L_w, hd);
    }
}

// Crop-mode grid caches. Windowed layers' RPE tables are grid-invariant
// (aH == window_size); only the global-attn layers need tables at the tile
// grid (get_rel_pos then really interpolates: 127 rows → 2*nP-1).
static void ensure_crop_rpe_tables(ds_ocr2_ctx & ctx, int nP) {
    if (ctx.rp_crop_grid == nP) return;
    auto & s = ctx.m.shp;
    int hd = s.head_dim;
    ctx.rp_h_crop.assign(s.depth, {});
    ctx.rp_w_crop.assign(s.depth, {});
    for (int li = 0; li < s.depth; li++) {
        auto & blk = ctx.m.sam_blocks[li];
        if (!blk.rel_pos_h || !blk.rel_pos_w || !blk.is_global) continue;
        auto rph = to_f32(blk.rel_pos_h), rpw = to_f32(blk.rel_pos_w);
        int L_h = (int)blk.rel_pos_h->ne[1], L_w = (int)blk.rel_pos_w->ne[1];
        ctx.rp_h_crop[li] = get_rel_pos(nP, nP, rph.data(), L_h, hd);
        ctx.rp_w_crop[li] = get_rel_pos(nP, nP, rpw.data(), L_w, hd);
    }
    ctx.rp_crop_grid = nP;
}

// get_abs_pos_sam: bicubic (torch a=-0.75), antialias=True, align_corners=False.
static void ensure_crop_pos_embed(ds_ocr2_ctx & ctx, const std::vector<float> & pos_default, int nP0, int nP, int C) {
    if (ctx.pos_crop_grid == nP) return;
    ctx.pos_embed_crop.assign((size_t)nP * nP * C, 0.0f);
    std::vector<float> plane_in((size_t)nP0 * nP0), plane_out((size_t)nP * nP);
    for (int c = 0; c < C; c++) {
        for (int t = 0; t < nP0 * nP0; t++) plane_in[t] = pos_default[(size_t)t * C + c];
        bicubic_aa_resample_plane(plane_in.data(), nP0, nP0, plane_out.data(), nP, nP, -0.75f);
        for (int t = 0; t < nP * nP; t++) ctx.pos_embed_crop[(size_t)t * C + c] = plane_out[t];
    }
    ctx.pos_crop_grid = nP;
}

// ---------------------------------------------------------------------------
// SAM ViT-B per-layer ggml graph (same pattern as got_ocr.cpp)
// ---------------------------------------------------------------------------

static ggml_cgraph * build_sam_layer_graph(ggml_context * g, ds_ocr2_ctx * ctx, int li, int C, int T, int aH, int aW,
                                           int nW, int n_heads, bool skip_ln1) {
    auto & layer = ctx->m.sam_blocks[li];
    int hd = C / n_heads, wN = aH * aW, batch = n_heads * nW;
    float attn_scale = 1.0f / sqrtf((float)hd);
    ggml_cgraph * gf = ggml_new_graph_custom(g, 512, false);

    ggml_tensor * inp = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, T);
    ggml_set_name(inp, "layer_input");
    ggml_set_input(inp);

    ggml_tensor * res_inp = nullptr;
    if (skip_ln1) {
        res_inp = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, T);
        ggml_set_name(res_inp, "residual_input");
        ggml_set_input(res_inp);
    }

    ggml_tensor * rp_h = ggml_new_tensor_3d(g, GGML_TYPE_F32, hd, aH, aH);
    ggml_set_name(rp_h, "rp_h");
    ggml_set_input(rp_h);
    ggml_tensor * rp_w = ggml_new_tensor_3d(g, GGML_TYPE_F32, hd, aW, aW);
    ggml_set_name(rp_w, "rp_w");
    ggml_set_input(rp_w);

    ggml_tensor * cur = skip_ln1 ? inp : g_ln(g, inp, layer.ln1_w, layer.ln1_b, 1e-6f);
    ggml_tensor * qkv = g_linear(g, cur, layer.qkv_w, layer.qkv_b);

    ggml_tensor * Q = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1], 0));
    ggml_tensor * K = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1], (size_t)C * sizeof(float)));
    ggml_tensor * V = ggml_cont(g, ggml_view_2d(g, qkv, C, T, qkv->nb[1], (size_t)2 * C * sizeof(float)));

    Q = ggml_reshape_4d(g, Q, hd, n_heads, wN, nW);
    Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));
    Q = ggml_reshape_3d(g, Q, hd, wN, batch);
    K = ggml_reshape_4d(g, K, hd, n_heads, wN, nW);
    K = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3));
    K = ggml_reshape_3d(g, K, hd, wN, batch);
    V = ggml_reshape_4d(g, V, hd, n_heads, wN, nW);
    V = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3));
    V = ggml_reshape_3d(g, V, hd, wN, batch);

    ggml_tensor * scores = ggml_mul_mat(g, K, Q);
    scores = ggml_scale(g, scores, attn_scale);

    // Decomposed RPE
    ggml_tensor * Q_4d = ggml_reshape_4d(g, Q, hd, aW, aH, batch);
    ggml_tensor * rp_h_4d = ggml_reshape_4d(g, rp_h, hd, aH, aH, 1);
    ggml_tensor * rel_h = ggml_mul_mat(g, rp_h_4d, Q_4d);
    rel_h = ggml_reshape_3d(g, rel_h, aH, wN, batch);
    rel_h = ggml_reshape_4d(g, rel_h, 1, aH, wN, batch);

    ggml_tensor * Q_w = ggml_cont(g, ggml_permute(g, Q_4d, 0, 2, 1, 3));
    ggml_tensor * rp_w_4d = ggml_reshape_4d(g, rp_w, hd, aW, aW, 1);
    ggml_tensor * rel_w2 = ggml_mul_mat(g, rp_w_4d, Q_w);
    rel_w2 = ggml_cont(g, ggml_permute(g, rel_w2, 0, 2, 1, 3));
    rel_w2 = ggml_reshape_3d(g, rel_w2, aW, wN, batch);
    rel_w2 = ggml_reshape_4d(g, rel_w2, aW, 1, wN, batch);

    scores = ggml_reshape_4d(g, scores, aW, aH, wN, batch);
    scores = ggml_add(g, scores, rel_h);
    scores = ggml_add(g, scores, rel_w2);
    scores = ggml_reshape_3d(g, scores, wN, wN, batch);

    scores = ggml_soft_max_ext(g, scores, nullptr, 1.0f, 0.0f);

    ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, V, 1, 0, 2, 3));
    ggml_tensor * attn = ggml_mul_mat(g, Vt, scores);
    attn = ggml_reshape_4d(g, attn, hd, wN, n_heads, nW);
    attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
    attn = ggml_reshape_2d(g, attn, C, T);
    attn = g_linear(g, attn, layer.proj_w, layer.proj_b);

    cur = ggml_add(g, skip_ln1 ? res_inp : inp, attn);

    ggml_tensor * residual = cur;
    cur = g_ln(g, cur, layer.ln2_w, layer.ln2_b, 1e-6f);
    ggml_tensor * up = g_linear(g, cur, layer.ffn_up_w, layer.ffn_up_b);
    up = ggml_gelu(g, up);
    cur = g_linear(g, up, layer.ffn_down_w, layer.ffn_down_b);
    cur = ggml_add(g, residual, cur);

    ggml_set_name(cur, "layer_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);
    return gf;
}

// ---------------------------------------------------------------------------
// SAM vision encoder
// ---------------------------------------------------------------------------

// Metal graph for the SAM neck + downsample (conv 768->256 1x1, LN2d, conv
// 256->256 3x3, LN2d, conv 256->512 3x3 s2, conv 512->896 3x3 s2). Replaces the
// CPU conv2d_cpu chain. Conv kernels are fed as F32 inputs (the GGUF stores them
// Q8_0 — can't reshape a quantized tensor to [1,1,IC,OC]). Default path;
// DS_SAM_CONV_CPU=1 restores the CPU chain. Validated equal via DS_REF sam_output.
// Metal graph for the SAM patch embedding (conv 3->C, PS×PS, stride PS) + the
// absolute position embedding. Replaces the scalar/threaded per-patch matmul.
// Produces hidden as [C, N] (== hidden[tok*C+c]). Kernel/pos fed as F32.
static ggml_cgraph * build_sam_patch_graph(ggml_context * g, int imgS, int PS, int C, int nP) {
    ggml_cgraph * gf = ggml_new_graph(g);
    ggml_tensor * px = ggml_new_tensor_4d(g, GGML_TYPE_F32, imgS, imgS, 3, 1);
    ggml_set_name(px, "px");
    ggml_set_input(px);
    ggml_tensor * w = ggml_new_tensor_4d(g, GGML_TYPE_F32, PS, PS, 3, C);
    ggml_set_name(w, "w_patch");
    ggml_set_input(w);
    ggml_tensor * bias = ggml_new_tensor_1d(g, GGML_TYPE_F32, C);
    ggml_set_name(bias, "pe_b");
    ggml_set_input(bias);
    ggml_tensor * pos = ggml_new_tensor_2d(g, GGML_TYPE_F32, C, nP * nP);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    ggml_tensor * x = ggml_conv_2d(g, w, px, PS, PS, 0, 0, 1, 1); // [nP,nP,C]
    x = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3));             // [C,nP,nP]
    x = ggml_reshape_2d(g, x, C, nP * nP);                        // [C, N]
    x = ggml_add(g, x, bias);                                     // + per-channel bias
    x = ggml_add(g, x, pos);
    ggml_set_name(x, "patch_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

static ggml_cgraph * build_sam_neck_graph(ggml_context * g, int nP, int C, int nC, int ds1_ch, int ds2_ch) {
    ggml_cgraph * gf = ggml_new_graph(g);
    ggml_tensor * chw = ggml_new_tensor_4d(g, GGML_TYPE_F32, nP, nP, C, 1);
    ggml_set_name(chw, "chw");
    ggml_set_input(chw);
    auto in4 = [&](const char * nm, int kw, int kh, int ic, int oc) {
        ggml_tensor * t = ggml_new_tensor_4d(g, GGML_TYPE_F32, kw, kh, ic, oc);
        ggml_set_name(t, nm);
        ggml_set_input(t);
        return t;
    };
    auto in1 = [&](const char * nm, int n) {
        ggml_tensor * t = ggml_new_tensor_1d(g, GGML_TYPE_F32, n);
        ggml_set_name(t, nm);
        ggml_set_input(t);
        return t;
    };
    ggml_tensor * w_nc1 = in4("w_nc1", 1, 1, C, nC);
    ggml_tensor * w_nc2 = in4("w_nc2", 3, 3, nC, nC);
    ggml_tensor * w_n2 = in4("w_n2", 3, 3, nC, ds1_ch);
    ggml_tensor * w_n3 = in4("w_n3", 3, 3, ds1_ch, ds2_ch);
    ggml_tensor *ln1w = in1("ln1w", nC), *ln1b = in1("ln1b", nC);
    ggml_tensor *ln2w = in1("ln2w", nC), *ln2b = in1("ln2b", nC);

    // LayerNorm over the channel axis (ne[2]) at each spatial position.
    auto ln2d = [&](ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        ggml_tensor * xp = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3)); // [C,W,H]
        xp = ggml_norm(g, xp, 1e-6f);
        xp = ggml_add(g, ggml_mul(g, xp, w), b);
        return ggml_cont(g, ggml_permute(g, xp, 2, 0, 1, 3)); // [W,H,C]
    };

    ggml_tensor * x = ggml_conv_2d(g, w_nc1, chw, 1, 1, 0, 0, 1, 1); // [nP,nP,nC]
    x = ln2d(x, ln1w, ln1b);
    x = ggml_conv_2d(g, w_nc2, x, 1, 1, 1, 1, 1, 1); // [nP,nP,nC]
    x = ln2d(x, ln2w, ln2b);
    x = ggml_conv_2d(g, w_n2, x, 2, 2, 1, 1, 1, 1); // [ds1,ds1,ds1_ch]
    x = ggml_conv_2d(g, w_n3, x, 2, 2, 1, 1, 1, 1); // [ds2,ds2,ds2_ch]
    int ds2 = (nP + 2 - 3) / 2 + 1;
    ds2 = (ds2 + 2 - 3) / 2 + 1;
    x = ggml_cont(g, ggml_permute(g, x, 1, 2, 0, 3)); // [C, W, H]
    x = ggml_reshape_2d(g, x, ds2_ch, ds2 * ds2);     // [C, n_vis] = out_features
    ggml_set_name(x, "neck_out");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

// `imgS_in` selects the input resolution: 0 (default) = shp.image_size (the
// 1024² global view); crop mode passes 768 for the dynamic-preprocess tiles.
static bool encode_sam(ds_ocr2_ctx & ctx, const float * pixels, std::vector<float> & out_features, int & out_n_tokens,
                       int & out_dim, int imgS_in = 0) {
    auto & s = ctx.m.shp;
    const int imgS = imgS_in > 0 ? imgS_in : s.image_size;
    int C = s.hidden, PS = s.patch_size, nP = imgS / PS;
    const int nP0 = s.image_size / PS; // grid the pos-embed / RPE tables ship at
    const bool crop_grid = nP != nP0;
    int N = nP * nP, hd = s.head_dim, ws = s.window_size;
    if (crop_grid) ensure_crop_rpe_tables(ctx, nP);
    auto _sam_t = std::chrono::steady_clock::now();
    auto sam_mark = [&](const char * w) {
        if (!getenv("DS_DBG")) return;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "  [time] sam.%s %lldms\n", w,
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - _sam_t).count());
        _sam_t = now;
    };

    // Patch embedding
    auto pe_w = to_f32(ctx.m.patch_embed_w);
    auto pe_b = to_f32(ctx.m.patch_embed_b);
    auto pos = to_f32(ctx.m.pos_embed);
    if (crop_grid && !pos.empty()) {
        ensure_crop_pos_embed(ctx, pos, nP0, nP, C);
        pos = ctx.pos_embed_crop;
    }
    int patch_dim = 3 * PS * PS;
    std::vector<float> hidden(N * C);

    if (!getenv("DS_SAM_CONV_CPU")) {
        // Patch embed on Metal (conv 3->C, PS×PS stride PS) + position embed.
        size_t meta_sz = 8 * 1024 * 1024;
        std::vector<uint8_t> mb(meta_sz);
        ggml_init_params ip = { meta_sz, mb.data(), true };
        ggml_context * gc = ggml_init(ip);
        ggml_cgraph * gf = build_sam_patch_graph(gc, imgS, PS, C, nP);
        ggml_backend_sched_reset(ctx.sched);
        ggml_backend_sched_alloc_graph(ctx.sched, gf);
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "px"), pixels, 0, (size_t)3 * imgS * imgS * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "w_patch"), pe_w.data(), 0, pe_w.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pe_b"), pe_b.data(), 0, pe_b.size() * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos"), pos.data(), 0, pos.size() * sizeof(float));
        ggml_backend_sched_graph_compute(ctx.sched, gf);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "patch_out"), hidden.data(), 0,
                                (size_t)N * C * sizeof(float));
        ggml_free(gc);
    } else {
        // Threaded CPU per-patch matmul (exact fallback).
        auto patch_rows = [&](int py0, int py1) {
            std::vector<float> patch(patch_dim);
            for (int py = py0; py < py1; py++)
                for (int px = 0; px < nP; px++) {
                    int tok = py * nP + px;
                    for (int c = 0; c < 3; c++)
                        for (int ky = 0; ky < PS; ky++)
                            for (int kx = 0; kx < PS; kx++)
                                patch[c * PS * PS + ky * PS + kx] =
                                    pixels[c * imgS * imgS + (py * PS + ky) * imgS + (px * PS + kx)];
                    for (int o = 0; o < C; o++) {
                        float sv = pe_b.empty() ? 0.0f : pe_b[o];
                        for (int i = 0; i < patch_dim; i++) sv += pe_w[o * patch_dim + i] * patch[i];
                        hidden[tok * C + o] = sv + (pos.empty() ? 0.0f : pos[tok * C + o]);
                    }
                }
        };
        int nt = std::max(1, std::min(ctx.n_threads, nP));
        if (nt <= 1)
            patch_rows(0, nP);
        else {
            std::vector<std::thread> pool;
            int chunk = (nP + nt - 1) / nt;
            for (int t = 0; t < nt; t++) {
                int y0 = t * chunk, y1 = std::min(nP, y0 + chunk);
                if (y0 < y1) pool.emplace_back(patch_rows, y0, y1);
            }
            for (auto & th : pool) th.join();
        }
    }

    sam_mark("patch_embed");
    // Pre-dequant LN weights for windowed layers
    std::vector<std::vector<float>> ln1_ws(s.depth), ln1_bs(s.depth);
    for (int li = 0; li < s.depth; li++)
        if (!ctx.m.sam_blocks[li].is_global) {
            ln1_ws[li] = to_f32(ctx.m.sam_blocks[li].ln1_w);
            ln1_bs[li] = to_f32(ctx.m.sam_blocks[li].ln1_b);
        }

    // Per-layer ggml graph
    for (int li = 0; li < s.depth; li++) {
        auto _slt = std::chrono::steady_clock::now();
        auto & blk = ctx.m.sam_blocks[li];
        bool is_global = blk.is_global;
        int aH = is_global ? nP : ws, aW = aH, wN = aH * aW;
        int nW, T;
        if (is_global) {
            nW = 1;
            T = N;
        } else {
            int ph = (ws - nP % ws) % ws, pw = (ws - nP % ws) % ws;
            nW = ((nP + ph) / ws) * ((nP + pw) / ws);
            T = wN * nW;
        }

        bool skip_ln1 = !is_global;
        std::vector<float> ln1_hidden;
        if (skip_ln1) {
            ln1_hidden.resize(N * C);
            for (int n = 0; n < N; n++)
                layernorm_cpu(hidden.data() + n * C, ln1_hidden.data() + n * C, C, ln1_ws[li].data(), ln1_bs[li].data(),
                              1e-6f);
        }

        std::vector<float> graph_input, residual_input;
        if (is_global)
            graph_input.assign(hidden.begin(), hidden.end());
        else {
            graph_input.resize(T * C, 0.0f);
            window_partition(ln1_hidden.data(), graph_input.data(), nP, ws, C);
            residual_input.resize(T * C, 0.0f);
            window_partition(hidden.data(), residual_input.data(), nP, ws, C);
        }

        // Global layers on the crop grid use the tables interpolated for nP;
        // windowed layers' tables (aH == ws) are grid-invariant.
        const auto & rp_h_src = (crop_grid && is_global) ? ctx.rp_h_crop[li] : ctx.rp_h_per_layer[li];
        const auto & rp_w_src = (crop_grid && is_global) ? ctx.rp_w_crop[li] : ctx.rp_w_per_layer[li];
        if (getenv("DS_DBG"))
            fprintf(stderr, "  [dbg] sam li=%d is_global=%d aH=%d nW=%d T=%d rp_h.sz=%zu\n", li, is_global, aH, nW, T,
                    rp_h_src.size());
        std::vector<float> rp_h_ggml(aH * aH * hd), rp_w_ggml(aW * aW * hd);
        reformat_rp_table(rp_h_src.data(), rp_h_ggml.data(), aH, hd);
        reformat_rp_table(rp_w_src.data(), rp_w_ggml.data(), aW, hd);
        if (getenv("DS_DBG")) fprintf(stderr, "  [dbg] sam li=%d reformat ok, building graph\n", li);

        size_t meta_sz = 8 * 1024 * 1024;
        std::vector<uint8_t> mb(meta_sz);
        ggml_init_params ip = { meta_sz, mb.data(), true };
        ggml_context * gc = ggml_init(ip);

        ggml_cgraph * gf = build_sam_layer_graph(gc, &ctx, li, C, T, aH, aW, nW, s.heads, skip_ln1);
        ggml_backend_sched_reset(ctx.sched);
        ggml_backend_sched_alloc_graph(ctx.sched, gf);

        ggml_tensor * inp_t = ggml_graph_get_tensor(gf, "layer_input");
        ggml_backend_tensor_set(inp_t, graph_input.data(), 0, (size_t)T * C * sizeof(float));
        if (skip_ln1) {
            ggml_tensor * res_t = ggml_graph_get_tensor(gf, "residual_input");
            ggml_backend_tensor_set(res_t, residual_input.data(), 0, (size_t)T * C * sizeof(float));
        }
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rp_h"), rp_h_ggml.data(), 0,
                                (size_t)aH * aH * hd * sizeof(float));
        ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "rp_w"), rp_w_ggml.data(), 0,
                                (size_t)aW * aW * hd * sizeof(float));

        ggml_backend_sched_graph_compute(ctx.sched, gf);

        std::vector<float> graph_output(T * C);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "layer_output"), graph_output.data(), 0,
                                (size_t)T * C * sizeof(float));
        ggml_free(gc);

        if (is_global)
            memcpy(hidden.data(), graph_output.data(), N * C * sizeof(float));
        else
            window_unpartition(graph_output.data(), hidden.data(), nP, ws, C);

        if (ctx.verbosity >= 2)
            fprintf(stderr, "deepseek_ocr2: sam_layer_%d done (%s, T=%d)\n", li, is_global ? "global" : "window", T);
        if (getenv("DS_DBG"))
            fprintf(stderr, "  [time] sam_li=%d (%s T=%d) %lldms\n", li, is_global ? "global" : "window", T,
                    (long long)std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
                                                                                     _slt)
                        .count());
    }

    // Diff: pre-neck ViT output (4096x768). The reference dump does not capture
    // this intermediate; named distinctly so it never collides with the final
    // SAM output below.
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("sam_vit_output")) {
            auto r = ref.compare("sam_vit_output", hidden.data(), N * C);
            fprintf(stderr, "  sam_vit_output: cos_min=%.6f max_abs=%.6f %s\n", r.cos_min, r.max_abs,
                    r.is_pass() ? "PASS" : "FAIL");
        }
    }

    sam_mark("layers");
    // Neck: Conv(768->256,1x1) -> LN2d -> Conv(256->256,3x3,p1) -> LN2d
    int nC = s.neck_out;
    std::vector<float> chw(C * nP * nP);
    for (int tok = 0; tok < N; tok++) {
        int y = tok / nP, x = tok % nP;
        for (int c = 0; c < C; c++) chw[c * nP * nP + y * nP + x] = hidden[tok * C + c];
    }

    // net_2 = 256->512, net_3 = 512->896 (= Qwen2 dim); derive channels from the
    // weights (ne[1]), not the config's nominal [512,1024] (which overruns).
    int ds1_ch = (int)ctx.m.net_2_w->ne[1];
    int ds2_ch = (int)ctx.m.net_3_w->ne[1];
    int ds1_H = (nP + 2 - 3) / 2 + 1;
    int ds2_H = (ds1_H + 2 - 3) / 2 + 1, ds2_W = ds2_H;
    int n_vis = ds2_H * ds2_W, vis_D = ds2_ch;
    out_features.resize((size_t)n_vis * vis_D);

    if (!getenv("DS_SAM_CONV_CPU")) {
        // Neck + downsample on Metal (ggml_conv_2d), ~20-40x vs the CPU convs and
        // no thread-scheduling variance. Conv kernels fed as F32 (GGUF stores them
        // Q8_0; can't reshape a quantized tensor to [1,1,IC,OC]). DS_SAM_CONV_CPU=1
        // restores the threaded CPU chain. Validated equal via DS_REF sam_output.
        auto nc1 = to_f32(ctx.m.neck_conv1_w), nc2 = to_f32(ctx.m.neck_conv2_w);
        auto n2 = to_f32(ctx.m.net_2_w), n3 = to_f32(ctx.m.net_3_w);
        auto l1w = to_f32(ctx.m.neck_ln1_w), l1b = to_f32(ctx.m.neck_ln1_b);
        auto l2w = to_f32(ctx.m.neck_ln2_w), l2b = to_f32(ctx.m.neck_ln2_b);
        size_t meta_sz = 16 * 1024 * 1024;
        std::vector<uint8_t> mb(meta_sz);
        ggml_init_params ip = { meta_sz, mb.data(), true };
        ggml_context * gc = ggml_init(ip);
        ggml_cgraph * gf = build_sam_neck_graph(gc, nP, C, nC, ds1_ch, ds2_ch);
        ggml_backend_sched_reset(ctx.sched);
        ggml_backend_sched_alloc_graph(ctx.sched, gf);
        auto setn = [&](const char * nm, const std::vector<float> & v) {
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, nm), v.data(), 0, v.size() * sizeof(float));
        };
        setn("chw", chw);
        setn("w_nc1", nc1);
        setn("w_nc2", nc2);
        setn("w_n2", n2);
        setn("w_n3", n3);
        setn("ln1w", l1w);
        setn("ln1b", l1b);
        setn("ln2w", l2w);
        setn("ln2b", l2b);
        ggml_backend_sched_graph_compute(ctx.sched, gf);
        ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "neck_out"), out_features.data(), 0,
                                (size_t)n_vis * vis_D * sizeof(float));
        ggml_free(gc);
    } else {
        auto nc1_w = to_f32(ctx.m.neck_conv1_w);
        std::vector<float> neck1(nC * nP * nP);
        conv2d_cpu(chw.data(), neck1.data(), nc1_w.data(), nullptr, C, nC, nP, nP, 1, 1, 1, 0, ctx.n_threads);
        auto nln1_w = to_f32(ctx.m.neck_ln1_w), nln1_b = to_f32(ctx.m.neck_ln1_b);
        std::vector<float> neck1_ln(nC * nP * nP);
        layernorm2d_cpu(neck1.data(), neck1_ln.data(), nC, nP, nP, nln1_w.data(), nln1_b.data());
        auto nc2_w = to_f32(ctx.m.neck_conv2_w);
        std::vector<float> neck2(nC * nP * nP);
        conv2d_cpu(neck1_ln.data(), neck2.data(), nc2_w.data(), nullptr, nC, nC, nP, nP, 3, 3, 1, 1, ctx.n_threads);
        auto nln2_w = to_f32(ctx.m.neck_ln2_w), nln2_b = to_f32(ctx.m.neck_ln2_b);
        std::vector<float> neck2_ln(nC * nP * nP);
        layernorm2d_cpu(neck2.data(), neck2_ln.data(), nC, nP, nP, nln2_w.data(), nln2_b.data());
        auto n2_w = to_f32(ctx.m.net_2_w);
        std::vector<float> ds1((size_t)ds1_ch * ds1_H * ds1_H);
        conv2d_cpu(neck2_ln.data(), ds1.data(), n2_w.data(), nullptr, nC, ds1_ch, nP, nP, 3, 3, 2, 1, ctx.n_threads);
        auto n3_w = to_f32(ctx.m.net_3_w);
        std::vector<float> ds2((size_t)ds2_ch * ds2_H * ds2_W);
        conv2d_cpu(ds1.data(), ds2.data(), n3_w.data(), nullptr, ds1_ch, ds2_ch, ds1_H, ds1_H, 3, 3, 2, 1,
                   ctx.n_threads);
        for (int tok = 0; tok < n_vis; tok++) {
            int y = tok / ds2_W, x = tok % ds2_W;
            for (int c = 0; c < vis_D; c++) out_features[tok * vis_D + c] = ds2[c * ds2_H * ds2_W + y * ds2_W + x];
        }
    }

    sam_mark("neck_downsample");
    out_n_tokens = n_vis;
    out_dim = vis_D;

    // Diff: final SAM output (post neck + downsample), [256, 896] — this is the
    // tensor the reference dump's "sam_output" corresponds to (sam_model output).
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("sam_output")) {
            auto r = ref.compare("sam_output", out_features.data(), (size_t)out_n_tokens * out_dim);
            fprintf(stderr, "  sam_output: cos_min=%.6f max_abs=%.6f %s\n", r.cos_min, r.max_abs,
                    r.is_pass() ? "PASS" : "FAIL");
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Qwen2 bidirectional encoder (CPU-scalar)
// ---------------------------------------------------------------------------

// NEOX (rotate_half) RoPE applied in-place to one head_dim vector at `pos`.
static void apply_rope_neox(float * v, int hd, int pos, float theta) {
    int half = hd / 2;
    for (int j = 0; j < half; j++) {
        float freq = 1.0f / powf(theta, (float)(2 * j) / hd);
        float a = pos * freq, c = cosf(a), s = sinf(a);
        float x1 = v[j], x2 = v[j + half];
        v[j] = x1 * c - x2 * s;
        v[j + half] = x2 * c + x1 * s;
    }
}

// Graph-based qwen2 encoder layer (runs on ctx.sched / Metal). One layer over
// all T = n_vis + n_query tokens at once — bidirectional, no KV cache. Mirrors
// build_llm_layer_attn (NEOX RoPE, GQA interleave, soft_max_ext+mask) but adds
// q/k/v biases and an always-on SwiGLU FFN. The graph is built + executed +
// freed within one scope by the caller (SAM pattern), so the meta buffer never
// outlives the context. Gated by DS_QWEN2_SCALAR=1 (falls back to the CPU path).
static ggml_cgraph * build_qwen2_enc_layer_graph(ggml_context * g, ds_ocr2_ctx * ctx, int li, int T) {
    auto & qhp = ctx->m.qhp;
    auto & ly = ctx->m.qwen2_layers[li];
    int D = qhp.hidden, nh = qhp.heads, nkv = qhp.kv_heads;
    int hd = D / nh, kv_repeat = nh / nkv;
    float eps = qhp.rms_eps;

    ggml_cgraph * gf = ggml_new_graph_custom(g, 2048, false);
    auto rmsnorm = [&](ggml_tensor * t, ggml_tensor * w) {
        return ggml_mul(g, ggml_rms_norm(g, t, eps), ensure_f32(g, w));
    };

    ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "layer_input");
    ggml_set_input(x);
    ggml_tensor * pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);
    ggml_tensor * mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, T, T); // [keys, queries]
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    // Pre-norm + Q/K/V (+ bias)
    ggml_tensor * h = rmsnorm(x, ly.in_ln_w);
    ggml_tensor * Q = ggml_mul_mat(g, ly.q_w, h);
    ggml_tensor * K = ggml_mul_mat(g, ly.k_w, h);
    ggml_tensor * V = ggml_mul_mat(g, ly.v_w, h);
    if (ly.q_b) Q = ggml_add(g, Q, ensure_f32(g, ly.q_b));
    if (ly.k_b) K = ggml_add(g, K, ensure_f32(g, ly.k_b));
    if (ly.v_b) V = ggml_add(g, V, ensure_f32(g, ly.v_b));

    Q = ggml_reshape_3d(g, Q, hd, nh, T);
    K = ggml_reshape_3d(g, K, hd, nkv, T);
    V = ggml_reshape_3d(g, V, hd, nkv, T);
    Q = ggml_rope_ext(g, Q, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, qhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(g, K, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, qhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Attention. Default = manual masked GQA (soft_max_ext + explicit interleave
    // repeat) — the verified-correct path. Opt-in DS_QWEN2_ENC_FLASH=1 selects
    // ggml_flash_attn_ext (native GQA broadcast, no repeat; output is already
    // [hd,nh,T] so reshape straight to [D,T] — NO trailing permute; see the
    // Jun-20 perf-sweep regression). A/B-verify the flash path's decoded output
    // equals the manual path before making it the default (per the dev-guide rule).
    ggml_tensor * attn;
    if (getenv("DS_QWEN2_ENC_FLASH")) {
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3));                // [hd, T, nh]
        ggml_tensor * Kp = ggml_cont(g, ggml_permute(g, K, 0, 2, 1, 3)); // [hd, T, nkv]
        ggml_tensor * Vp = ggml_cont(g, ggml_permute(g, V, 0, 2, 1, 3)); // [hd, T, nkv]
        attn = ggml_flash_attn_ext(g, Q, Kp, Vp, mask, 1.0f / sqrtf((float)hd), 0.0f, 0.0f);
        attn = ggml_reshape_2d(g, attn, D, T); // flash output already [hd,nh,T]
    } else {
        ggml_tensor * Kfull = ggml_cont(g, K);
        ggml_tensor * Vfull = ggml_cont(g, V);
        if (kv_repeat > 1) { // GQA interleave (not tile)
            Kfull = ggml_reshape_4d(g, Kfull, hd, 1, nkv, T);
            Kfull = ggml_repeat(g, Kfull, ggml_new_tensor_4d(g, Kfull->type, hd, kv_repeat, nkv, T));
            Kfull = ggml_reshape_3d(g, Kfull, hd, nh, T);
            Vfull = ggml_reshape_4d(g, Vfull, hd, 1, nkv, T);
            Vfull = ggml_repeat(g, Vfull, ggml_new_tensor_4d(g, Vfull->type, hd, kv_repeat, nkv, T));
            Vfull = ggml_reshape_3d(g, Vfull, hd, nh, T);
        }
        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3)); // [hd, T, nh]
        Kfull = ggml_cont(g, ggml_permute(g, Kfull, 0, 2, 1, 3));
        Vfull = ggml_cont(g, ggml_permute(g, Vfull, 0, 2, 1, 3));
        ggml_tensor * scores = ggml_mul_mat(g, Kfull, Q); // [T(keys), T(queries), nh]
        scores = ggml_soft_max_ext(g, scores, mask, 1.0f / sqrtf((float)hd), 0.0f);
        ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, Vfull, 1, 0, 2, 3));
        attn = ggml_mul_mat(g, Vt, scores);
        attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(g, attn, D, T);
    }
    attn = ggml_mul_mat(g, ly.o_w, attn);
    x = ggml_add(g, x, attn);

    // SwiGLU FFN + residual
    ggml_tensor * res = x;
    h = rmsnorm(x, ly.post_ln_w);
    ggml_tensor * gate = ggml_silu(g, ggml_mul_mat(g, ly.gate_w, h));
    ggml_tensor * up = ggml_mul_mat(g, ly.up_w, h);
    x = ggml_add(g, res, ggml_mul_mat(g, ly.down_w, ggml_mul(g, gate, up)));

    ggml_set_name(x, "layer_output");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    return gf;
}

static bool encode_qwen2(ds_ocr2_ctx & ctx, const float * vis_features, int n_vis, int vis_dim,
                         std::vector<float> & out_enc, int & out_n_tokens, int & out_dim) {
    auto & qhp = ctx.m.qhp;
    int D = qhp.hidden, nh = qhp.heads, nkv = qhp.kv_heads;
    int hd = D / nh, kv_repeat = nh / nkv, inter = qhp.intermediate;
    float eps = qhp.rms_eps;

    // Build token sequence: vis features + query tokens. The blueprint
    // (Qwen2Decoder2Encoder.forward) selects the query bank by the SAM token
    // count: n_query==144 → query_768 (crop tiles), n_query==256 → query_1024
    // (the 1024² global view). Fall back to the other bank if the matching one
    // is absent (pre-crop GGUFs always carried both).
    ggml_tensor * query_t = ctx.m.query_1024 ? ctx.m.query_1024 : ctx.m.query_768;
    if (n_vis == 144 && ctx.m.query_768) query_t = ctx.m.query_768;
    auto query_data = to_f32(query_t);
    int n_query = query_data.empty() ? 0 : (int)(query_data.size() / D);

    // Total tokens = n_query + n_vis
    // The visual features from SAM are 1024-dim, but Qwen2 encoder is 896-dim.
    // A projection from vis_dim(1024) -> D(896) is needed (this is the query mechanism).
    // Actually, the query tokens are learnable embeddings that attend to vis features
    // via cross-attention. But in DeepSeek-OCR-2, the Qwen2 encoder is self-attention
    // over concatenated [query_tokens, visual_features].
    // The vis features need to be projected to D first if dims don't match.
    // For now assume vis_dim == D or handle the mismatch.

    // Blueprint (CustomQwen2): x_combined = cat([visual, queries]) — VISUAL
    // FIRST, then the learned query tokens. token_type 0=visual (non-causal),
    // 1=query (causal). RoPE applied at positions 0..T-1. Returns the query
    // half (y[:, n_vis:]).
    int T = n_vis + n_query;
    std::vector<float> hidden(T * D, 0.0f);
    if (vis_dim == D)
        memcpy(hidden.data(), vis_features, (size_t)n_vis * D * sizeof(float));
    else
        for (int t = 0; t < n_vis; t++)
            for (int d = 0; d < D; d++) hidden[t * D + d] = (d < vis_dim) ? vis_features[t * vis_dim + d] : 0.0f;
    if (n_query > 0) memcpy(hidden.data() + (size_t)n_vis * D, query_data.data(), (size_t)n_query * D * sizeof(float));

    // Run the 24 bidirectional transformer layers. Default: ggml graph on
    // ctx.sched (Metal). DS_QWEN2_SCALAR=1 forces the CPU-scalar reference path.
    if (!getenv("DS_QWEN2_SCALAR")) {
        std::vector<int32_t> pos(T);
        for (int t = 0; t < T; t++) pos[t] = t;
        // Bidirectional-vis + causal-query mask, shared across layers. Layout
        // matches soft_max_ext: [keys, queries], mask[qi*T + ki].
        std::vector<ggml_fp16_t> emask((size_t)T * T);
        const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ninf = ggml_fp32_to_fp16(-INFINITY);
        for (int qi = 0; qi < T; qi++) {
            bool qv = qi < n_vis;
            for (int ki = 0; ki < T; ki++) {
                bool ok = qv ? (ki < n_vis) : (ki < n_vis || ki <= qi);
                emask[(size_t)qi * T + ki] = ok ? z : ninf;
            }
        }
        for (int li = 0; li < qhp.depth; li++) {
            size_t meta_sz = 16 * 1024 * 1024;
            std::vector<uint8_t> mb(meta_sz);
            ggml_init_params ip = { meta_sz, mb.data(), true };
            ggml_context * gc = ggml_init(ip);
            ggml_cgraph * gf = build_qwen2_enc_layer_graph(gc, &ctx, li, T);
            ggml_backend_sched_reset(ctx.sched);
            ggml_backend_sched_alloc_graph(ctx.sched, gf);
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "layer_input"), hidden.data(), 0,
                                    (size_t)T * D * sizeof(float));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"), pos.data(), 0, (size_t)T * sizeof(int32_t));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "mask"), emask.data(), 0,
                                    (size_t)T * T * sizeof(ggml_fp16_t));
            ggml_backend_sched_graph_compute(ctx.sched, gf);
            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "layer_output"), hidden.data(), 0,
                                    (size_t)T * D * sizeof(float));
            ggml_free(gc);

            if (!ctx.diff_ref_path.empty()) {
                char nm[32];
                snprintf(nm, sizeof(nm), "qwen2_layer_%d", li);
                crispembed_diff::Ref ref;
                if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(nm)) {
                    auto r = ref.compare(nm, hidden.data(), (size_t)T * D);
                    fprintf(stderr, "  %s: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n", nm, r.cos_min, r.cos_mean,
                            r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                }
            }
        }
    } else
        // Run bidirectional transformer layers (CPU-scalar reference path)
        for (int li = 0; li < qhp.depth; li++) {
            auto & ly = ctx.m.qwen2_layers[li];
            auto in_ln = to_f32(ly.in_ln_w), post_ln = to_f32(ly.post_ln_w);
            auto qw = to_f32(ly.q_w), qb = to_f32(ly.q_b);
            auto kw = to_f32(ly.k_w), kb = to_f32(ly.k_b);
            auto vw = to_f32(ly.v_w), vb = to_f32(ly.v_b);
            auto ow = to_f32(ly.o_w);
            auto gw = to_f32(ly.gate_w), uw = to_f32(ly.up_w), dw = to_f32(ly.down_w);

            int q_dim = nh * hd, kv_dim = nkv * hd;

            // Pre-norm
            std::vector<float> normed(T * D);
            for (int t = 0; t < T; t++) rmsnorm_cpu(hidden.data() + t * D, normed.data() + t * D, D, in_ln.data(), eps);

            // Q/K/V projections
            std::vector<float> Q(T * q_dim), K(T * kv_dim), V(T * kv_dim);
            for (int t = 0; t < T; t++) {
                linear_cpu(normed.data() + t * D, Q.data() + t * q_dim, D, q_dim, qw.data(),
                           qb.empty() ? nullptr : qb.data());
                linear_cpu(normed.data() + t * D, K.data() + t * kv_dim, D, kv_dim, kw.data(),
                           kb.empty() ? nullptr : kb.data());
                linear_cpu(normed.data() + t * D, V.data() + t * kv_dim, D, kv_dim, vw.data(),
                           vb.empty() ? nullptr : vb.data());
            }

            // NEOX RoPE at positions 0..T-1 (Qwen2, rope_theta from config).
            for (int t = 0; t < T; t++) {
                for (int h = 0; h < nh; h++) apply_rope_neox(Q.data() + t * q_dim + h * hd, hd, t, qhp.rope_theta);
                for (int h = 0; h < nkv; h++) apply_rope_neox(K.data() + t * kv_dim + h * hd, hd, t, qhp.rope_theta);
            }

            // Multi-head attention with the token-type mask: visual tokens
            // (< n_vis) attend to visual only (bidirectional); query tokens (>=
            // n_vis) attend to all visual + causally to earlier/equal queries.
            float attn_scale = 1.0f / sqrtf((float)hd);
            std::vector<float> attn_out(T * D, 0.0f);

            for (int h = 0; h < nh; h++) {
                int kv_h = h / kv_repeat; // GQA mapping

                // Compute scores for all query positions
                for (int qi = 0; qi < T; qi++) {
                    bool qi_vis = qi < n_vis;
                    std::vector<float> scores(T);
                    for (int ki = 0; ki < T; ki++) {
                        bool allowed = qi_vis ? (ki < n_vis) : (ki < n_vis || ki <= qi);
                        if (!allowed) {
                            scores[ki] = -INFINITY;
                            continue;
                        }
                        float dot = 0;
                        for (int d = 0; d < hd; d++) dot += Q[qi * q_dim + h * hd + d] * K[ki * kv_dim + kv_h * hd + d];
                        scores[ki] = dot * attn_scale;
                    }
                    // Softmax
                    float max_s = *std::max_element(scores.begin(), scores.end());
                    float sum_exp = 0;
                    for (int ki = 0; ki < T; ki++) {
                        scores[ki] = expf(scores[ki] - max_s);
                        sum_exp += scores[ki];
                    }
                    for (int ki = 0; ki < T; ki++) scores[ki] /= sum_exp;

                    // Weighted sum of values
                    for (int ki = 0; ki < T; ki++)
                        for (int d = 0; d < hd; d++)
                            attn_out[qi * D + h * hd + d] += scores[ki] * V[ki * kv_dim + kv_h * hd + d];
                }
            }

            // Output projection + residual
            std::vector<float> proj(T * D);
            for (int t = 0; t < T; t++)
                linear_cpu(attn_out.data() + t * D, proj.data() + t * D, D, D, ow.data(), nullptr);
            for (int i = 0; i < T * D; i++) hidden[i] += proj[i];

            // Diff: post-attention hidden (residual + attn, pre-FFN). Splits each
            // layer into attention-half vs FFN-half to localize the divergence.
            if (!ctx.diff_ref_path.empty()) {
                char nm[40];
                snprintf(nm, sizeof(nm), "qwen2_layer_%d_postattn", li);
                crispembed_diff::Ref ref;
                if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(nm)) {
                    auto r = ref.compare(nm, hidden.data(), (size_t)T * D);
                    fprintf(stderr, "  %s: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n", nm, r.cos_min, r.cos_mean,
                            r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                }
            }

            // Post-attn norm + SwiGLU FFN + residual
            for (int t = 0; t < T; t++) {
                rmsnorm_cpu(hidden.data() + t * D, normed.data() + t * D, D, post_ln.data(), eps);
                std::vector<float> ffn_out(D);
                swiglu_ffn_cpu(normed.data() + t * D, ffn_out.data(), D, inter, gw.data(), uw.data(), dw.data());
                for (int d = 0; d < D; d++) hidden[t * D + d] += ffn_out[d];
            }

            if (ctx.verbosity >= 2) fprintf(stderr, "deepseek_ocr2: qwen2_enc_layer_%d done\n", li);

            // Diff: per-layer qwen2 hidden state (full [vis+query] seq, pre-final-norm)
            // for bisecting the encoder divergence.
            if (!ctx.diff_ref_path.empty()) {
                char nm[32];
                snprintf(nm, sizeof(nm), "qwen2_layer_%d", li);
                crispembed_diff::Ref ref;
                if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(nm)) {
                    auto r = ref.compare(nm, hidden.data(), (size_t)T * D);
                    fprintf(stderr, "  %s: cos_min=%.6f cos_mean=%.6f max_abs=%.6f %s\n", nm, r.cos_min, r.cos_mean,
                            r.max_abs, r.is_pass() ? "PASS" : "FAIL");
                }
            }
        }

    // Final RMSNorm (Qwen2Model.norm) over all tokens.
    if (ctx.m.qe_output_norm) {
        auto fn = to_f32(ctx.m.qe_output_norm);
        std::vector<float> tmp(D);
        for (int t = 0; t < T; t++) {
            rmsnorm_cpu(hidden.data() + t * D, tmp.data(), D, fn.data(), eps);
            memcpy(hidden.data() + t * D, tmp.data(), D * sizeof(float));
        }
    }

    // Output = the query half (blueprint y[:, n_vis:]): positions n_vis..T-1.
    out_n_tokens = (n_query > 0) ? n_query : T;
    out_dim = D;
    out_enc.resize((size_t)out_n_tokens * D);
    memcpy(out_enc.data(), hidden.data() + (size_t)n_vis * D, (size_t)out_n_tokens * D * sizeof(float));

    // Diff: Qwen2 encoder output (post final-norm, query half) — this is the
    // tensor the reference dump's "qwen2_enc_output" corresponds to.
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("qwen2_enc_output")) {
            auto r = ref.compare("qwen2_enc_output", out_enc.data(), (size_t)out_n_tokens * D);
            fprintf(stderr, "  qwen2_enc_output: cos_min=%.6f max_abs=%.6f %s\n", r.cos_min, r.max_abs,
                    r.is_pass() ? "PASS" : "FAIL");
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Projector: Linear(896, 1280)
// ---------------------------------------------------------------------------

static bool project_to_llm(ds_ocr2_ctx & ctx, const float * enc_out, int n_tokens, int enc_dim,
                           std::vector<float> & proj_out) {
    auto & lhp = ctx.m.lhp;
    int out_dim = lhp.hidden;
    auto pw = to_f32(ctx.m.projector_w);
    auto pb = to_f32(ctx.m.projector_b);

    proj_out.resize(n_tokens * out_dim);
    for (int t = 0; t < n_tokens; t++)
        linear_cpu(enc_out + t * enc_dim, proj_out.data() + t * out_dim, enc_dim, out_dim, pw.data(),
                   pb.empty() ? nullptr : pb.data());

    // Diff: projector output
    if (!ctx.diff_ref_path.empty()) {
        crispembed_diff::Ref ref;
        if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("projector_output")) {
            auto r = ref.compare("projector_output", proj_out.data(), n_tokens * out_dim);
            fprintf(stderr, "  projector_output: cos_min=%.6f max_abs=%.6f %s\n", r.cos_min, r.max_abs,
                    r.is_pass() ? "PASS" : "FAIL");
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// LLM decoder — ggml graph for attention + CPU-scalar MoE FFN
// ---------------------------------------------------------------------------

// Stack the 64 per-expert weights of each MoE layer into [in, out, n_exp]
// tensors so the decode graph can dispatch them with ggml_mul_mat_id on Metal
// (instead of the per-token CPU-scalar moe_ffn_cpu). Keeps the quantized blocks
// as-is — each expert is the same shape/type, so its bytes are a contiguous
// slice. Allocates a dedicated backend buffer (the per-expert tensors stay in
// model_buf for the DS_MOE_CPU fallback). Returns false on any failure → caller
// falls back to the CPU MoE.
static bool stack_moe_experts(ds_ocr2_ctx & ctx) {
    int n_exp = ctx.m.lhp.n_experts;
    int n_moe = 0;
    for (auto & ly : ctx.m.llm_layers)
        if ((int)ly.experts.size() == n_exp && ly.experts[0].gate_w) n_moe++;
    if (n_moe == 0) return false;

    ggml_init_params ip = { (size_t)n_moe * 3 * ggml_tensor_overhead() + 4096, nullptr, true };
    ctx.moe_ctx = ggml_init(ip);
    if (!ctx.moe_ctx) return false;

    for (auto & ly : ctx.m.llm_layers) {
        if ((int)ly.experts.size() != n_exp || !ly.experts[0].gate_w) continue;
        auto & e0 = ly.experts[0];
        ly.gate_exps = ggml_new_tensor_3d(ctx.moe_ctx, e0.gate_w->type, e0.gate_w->ne[0], e0.gate_w->ne[1], n_exp);
        ly.up_exps = ggml_new_tensor_3d(ctx.moe_ctx, e0.up_w->type, e0.up_w->ne[0], e0.up_w->ne[1], n_exp);
        ly.down_exps = ggml_new_tensor_3d(ctx.moe_ctx, e0.down_w->type, e0.down_w->ne[0], e0.down_w->ne[1], n_exp);
    }

    ctx.moe_buf = ggml_backend_alloc_ctx_tensors(ctx.moe_ctx, ctx.backend);
    if (!ctx.moe_buf) {
        ggml_free(ctx.moe_ctx);
        ctx.moe_ctx = nullptr;
        return false;
    }

    std::vector<uint8_t> tmp;
    auto fill = [&](ggml_tensor * stacked, const std::vector<moe_expert_w> & exps,
                    ggml_tensor * moe_expert_w::*member) {
        for (int e = 0; e < n_exp; e++) {
            ggml_tensor * src = exps[e].*member;
            size_t nb = ggml_nbytes(src);
            if (nb != stacked->nb[2]) return false; // slice size must match
            tmp.resize(nb);
            ggml_backend_tensor_get(src, tmp.data(), 0, nb);
            ggml_backend_tensor_set(stacked, tmp.data(), (size_t)e * stacked->nb[2], nb);
        }
        return true;
    };
    for (auto & ly : ctx.m.llm_layers) {
        if (!ly.gate_exps) continue;
        if (!fill(ly.gate_exps, ly.experts, &moe_expert_w::gate_w) ||
            !fill(ly.up_exps, ly.experts, &moe_expert_w::up_w) ||
            !fill(ly.down_exps, ly.experts, &moe_expert_w::down_w))
            return false;
    }
    return true;
}

struct llm_attn_graph {
    ggml_cgraph * gf{};
    ggml_context * gctx{};
    // The no-alloc ggml context places its tensor/graph metadata in `meta`, so
    // the buffer must outlive the returned graph. Holding it here (moved out with
    // the struct) fixes a use-after-free: a local meta buffer was freed on return,
    // leaving gf/gctx dangling — a latent crash that surfaced once the fast
    // (graph) qwen2 encoder let the decoder prefill actually run.
    std::vector<uint8_t> meta;
};

// Build attention-only graph for one LLM layer (no FFN — MoE done on CPU)
// For layer 0 (dense), includes the FFN in the graph.
static llm_attn_graph build_llm_layer_attn(ds_ocr2_ctx & ctx, int li, int T, int n_past, bool include_ffn,
                                           bool include_moe = false) {
    auto & lhp = ctx.m.lhp;
    auto & ly = ctx.m.llm_layers[li];
    int D = lhp.hidden, nh = lhp.heads, nkv = lhp.kv_heads;
    int hd = lhp.head_dim;
    int Lk = n_past + T;
    float eps = lhp.rms_eps;

    size_t meta_sz = 4 * 1024 * 1024;
    llm_attn_graph lag;
    lag.meta.resize(meta_sz); // owned by lag; survives the return (move preserves data ptr)
    ggml_init_params ip = { meta_sz, lag.meta.data(), true };
    lag.gctx = ggml_init(ip);
    auto * g = lag.gctx;
    lag.gf = ggml_new_graph_custom(g, 4096, false);

    auto rmsnorm = [&](ggml_tensor * t, ggml_tensor * w) -> ggml_tensor * {
        // ensure_f32: the norm weight is F16 in an all-F16 GGUF, and ggml's
        // elementwise mul does not support an f32×f16 operand pair.
        return ggml_mul(g, ggml_rms_norm(g, t, eps), ensure_f32(g, w));
    };

    // Input hidden states
    ggml_tensor * x = ggml_new_tensor_2d(g, GGML_TYPE_F32, D, T);
    ggml_set_name(x, "layer_input");
    ggml_set_input(x);

    // Position IDs
    ggml_tensor * pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // Persistent KV cache (device-side, no per-step upload)
    int kv_dim = nkv * hd;

    // Pre-norm
    ggml_tensor * h = rmsnorm(x, ly.in_ln_w);

    // Q/K/V
    ggml_tensor * Q = ggml_mul_mat(g, ly.q_w, h);
    ggml_tensor * K = ggml_mul_mat(g, ly.k_w, h);
    ggml_tensor * V = ggml_mul_mat(g, ly.v_w, h);

    // Reshape for RoPE
    Q = ggml_reshape_3d(g, Q, hd, nh, T);
    K = ggml_reshape_3d(g, K, hd, nkv, T);
    V = ggml_reshape_3d(g, V, hd, nkv, T);

    // Standard RoPE
    Q = ggml_rope_ext(g, Q, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    K = ggml_rope_ext(g, K, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Materialize K/V once for the attention path. K/V come from rope as
    // [hd, nkv, T] (memory: hd fastest, then nkv, then T) — exactly the layout
    // the cache reload expects (reshape_3d(hd, nkv, n_past)). Do NOT permute the
    // token/head axes (an earlier permute(0,2,1,3) transposed T<->nkv and
    // scrambled the cache).
    ggml_tensor * Kc = ggml_cont(g, K); // [hd, nkv, T]
    ggml_tensor * Vc = ggml_cont(g, V);

    // Write new K/V into persistent device cache at position n_past.
    // Kc/Vc are [hd, nkv, T]. Reshape to [kv_dim, T] for the cache write.
    ggml_tensor * K_flat = ggml_cont(g, ggml_reshape_2d(g, Kc, kv_dim, T));
    ggml_tensor * V_flat = ggml_cont(g, ggml_reshape_2d(g, Vc, kv_dim, T));

    size_t layer_off_k = (size_t)li * ctx.kvc.k->nb[2];
    size_t layer_off_v = (size_t)li * ctx.kvc.v->nb[2];

    ggml_tensor * k_dst =
        ggml_view_2d(g, ctx.kvc.k, kv_dim, T, ctx.kvc.k->nb[1], layer_off_k + (size_t)n_past * ctx.kvc.k->nb[1]);
    ggml_tensor * v_dst =
        ggml_view_2d(g, ctx.kvc.v, kv_dim, T, ctx.kvc.v->nb[1], layer_off_v + (size_t)n_past * ctx.kvc.v->nb[1]);

    ggml_build_forward_expand(lag.gf, ggml_cpy(g, K_flat, k_dst));
    ggml_build_forward_expand(lag.gf, ggml_cpy(g, V_flat, v_dst));

    // Read full K/V history [0..n_past+T) from persistent cache.
    ggml_tensor * Kfull =
        ggml_reshape_3d(g, ggml_view_2d(g, ctx.kvc.k, kv_dim, Lk, ctx.kvc.k->nb[1], layer_off_k), hd, nkv, Lk);
    ggml_tensor * Vfull =
        ggml_reshape_3d(g, ggml_view_2d(g, ctx.kvc.v, kv_dim, Lk, ctx.kvc.v->nb[1], layer_off_v), hd, nkv, Lk);

    // GQA repeat if needed
    int kv_repeat = nh / nkv;
    if (kv_repeat > 1) {
        Kfull = ggml_reshape_4d(g, Kfull, hd, 1, nkv, Lk);
        ggml_tensor * K_tgt = ggml_new_tensor_4d(g, Kfull->type, hd, kv_repeat, nkv, Lk);
        Kfull = ggml_repeat(g, Kfull, K_tgt);
        Kfull = ggml_reshape_3d(g, Kfull, hd, nh, Lk);

        Vfull = ggml_reshape_4d(g, Vfull, hd, 1, nkv, Lk);
        ggml_tensor * V_tgt = ggml_new_tensor_4d(g, Vfull->type, hd, kv_repeat, nkv, Lk);
        Vfull = ggml_repeat(g, Vfull, V_tgt);
        Vfull = ggml_reshape_3d(g, Vfull, hd, nh, Lk);
    }

    // Attention
    Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3)); // [hd, T, nh]
    Kfull = ggml_cont(g, ggml_permute(g, Kfull, 0, 2, 1, 3));
    Vfull = ggml_cont(g, ggml_permute(g, Vfull, 0, 2, 1, 3));

    // Causal mask
    ggml_tensor * mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, Lk, T);
    ggml_set_name(mask, "mask");
    ggml_set_input(mask);

    float attn_scale = 1.0f / sqrtf((float)hd);
    // Default = manual masked attention (verified-correct). Opt-in DS_LLM_FLASH=1
    // uses ggml_flash_attn_ext (output already [hd,nh,T] → reshape straight, NO
    // trailing permute). A/B-verify decoded output before flipping the default.
    ggml_tensor * attn;
    if (getenv("DS_LLM_FLASH")) {
        attn = ggml_flash_attn_ext(g, Q, Kfull, Vfull, mask, attn_scale, 0.0f, 0.0f);
        attn = ggml_reshape_2d(g, attn, D, T);
    } else {
        ggml_tensor * scores = ggml_mul_mat(g, Kfull, Q);
        scores = ggml_soft_max_ext(g, scores, mask, attn_scale, 0.0f);
        ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, Vfull, 1, 0, 2, 3));
        attn = ggml_mul_mat(g, Vt, scores);
        attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
        attn = ggml_reshape_2d(g, attn, D, T);
    }

    attn = ggml_mul_mat(g, ly.o_w, attn);
    x = ggml_add(g, x, attn);

    if (include_ffn) {
        // Dense SwiGLU FFN (layer 0)
        ggml_tensor * residual = x;
        h = rmsnorm(x, ly.post_ln_w);
        ggml_tensor * gate = ggml_silu(g, ggml_mul_mat(g, ly.ffn_gate_w, h));
        ggml_tensor * up = ggml_mul_mat(g, ly.ffn_up_w, h);
        ggml_tensor * ffn = ggml_mul_mat(g, ly.ffn_down_w, ggml_mul(g, gate, up));
        x = ggml_add(g, residual, ffn);
    } else if (include_moe) {
        // DeepSeek-V2 MoE FFN on Metal via ggml_mul_mat_id. Router → softmax →
        // top-k (raw probs; norm_topk_prob=False, routed_scaling_factor=1.0) →
        // per-expert SwiGLU dispatch → weighted sum + a (combined) shared expert.
        int n_exp = lhp.n_experts, K = lhp.n_experts_top;
        ggml_tensor * residual = x;
        ggml_tensor * hn = rmsnorm(x, ly.post_ln_w); // [D, T]

        ggml_tensor * logits = ggml_mul_mat(g, ly.router_w, hn); // [n_exp, T]
        ggml_tensor * probs = ggml_soft_max(g, logits);
        ggml_tensor * ids = ggml_top_k(g, probs, K); // [K, T] I32
        ggml_tensor * p3 = ggml_reshape_3d(g, probs, 1, n_exp, T);
        ggml_tensor * top_w = ggml_reshape_2d(g, ggml_get_rows(g, p3, ids), K, T); // [K, T]
        top_w = ggml_scale(g, top_w, lhp.routed_scaling_factor);

        ggml_tensor * hn3 = ggml_reshape_3d(g, hn, D, 1, T);
        ggml_tensor * hnK = ggml_repeat(g, hn3, ggml_new_tensor_3d(g, hn->type, D, K, T));
        ggml_tensor * gate = ggml_silu(g, ggml_mul_mat_id(g, ly.gate_exps, hnK, ids)); // [inter,K,T]
        ggml_tensor * up = ggml_mul_mat_id(g, ly.up_exps, hnK, ids);
        ggml_tensor * down = ggml_mul_mat_id(g, ly.down_exps, ggml_mul(g, gate, up), ids); // [D,K,T]

        // Weighted sum over the K experts: down [D,K,T] → [K,D,T], w [K,1,T] → [1,D,T].
        ggml_tensor * down_p = ggml_cont(g, ggml_permute(g, down, 1, 0, 2, 3));
        ggml_tensor * w_col = ggml_reshape_3d(g, top_w, K, 1, T);
        ggml_tensor * routed = ggml_reshape_2d(g, ggml_mul_mat(g, w_col, down_p), D, T);

        // Combined shared expert (always active), SwiGLU on the same normed input.
        ggml_tensor * sg = ggml_silu(g, ggml_mul_mat(g, ly.shared_gate_w, hn));
        ggml_tensor * su = ggml_mul_mat(g, ly.shared_up_w, hn);
        ggml_tensor * shared = ggml_mul_mat(g, ly.shared_down_w, ggml_mul(g, sg, su));

        x = ggml_add(g, residual, ggml_add(g, routed, shared));
    }

    ggml_set_name(x, "layer_output");
    ggml_set_output(x);
    ggml_build_forward_expand(lag.gf, x);
    // KV cache writes (ggml_cpy) were already expanded above.
    return lag;
}

// CPU-scalar MoE FFN for one layer
static void moe_ffn_cpu(ds_ocr2_ctx & ctx, int li, float * hidden, int T) {
    auto & lhp = ctx.m.lhp;
    auto & ly = ctx.m.llm_layers[li];
    int D = lhp.hidden, inter_e = lhp.expert_intermediate;
    int inter_s = lhp.shared_intermediate;
    int n_exp = lhp.n_experts, top_k = lhp.n_experts_top;
    float scale = lhp.routed_scaling_factor;
    float eps = lhp.rms_eps;

    auto post_ln = to_f32(ly.post_ln_w);
    auto router = to_f32(ly.router_w);

    // Dequant shared expert weights
    auto sh_gw = to_f32(ly.shared_gate_w);
    auto sh_uw = to_f32(ly.shared_up_w);
    auto sh_dw = to_f32(ly.shared_down_w);

    struct exp_w {
        std::vector<float> gw, uw, dw;
    };
    std::vector<exp_w> exp_ws(n_exp);

    // Pass 1 (serial, cheap): RMSNorm + route every token, recording its
    // normed vector and its top-k experts/weights. Also note which experts get
    // used so we dequant only those.
    std::vector<float> normed_all((size_t)T * D);
    std::vector<std::array<int, 16>> tk_idx(T);
    std::vector<std::array<float, 16>> tk_w(T);
    std::vector<char> used(n_exp, 0);
    for (int t = 0; t < T; t++) {
        float * normed = normed_all.data() + (size_t)t * D;
        rmsnorm_cpu(hidden + t * D, normed, D, post_ln.data(), eps);
        std::vector<float> logits(n_exp);
        for (int e = 0; e < n_exp; e++) {
            float dot = 0;
            for (int d = 0; d < D; d++) dot += normed[d] * router[e * D + d];
            logits[e] = dot;
        }
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum_exp = 0;
        for (int e = 0; e < n_exp; e++) {
            logits[e] = expf(logits[e] - max_l);
            sum_exp += logits[e];
        }
        for (int e = 0; e < n_exp; e++) logits[e] /= sum_exp;
        std::vector<std::pair<float, int>> scored(n_exp);
        for (int e = 0; e < n_exp; e++) scored[e] = { logits[e], e };
        std::partial_sort(scored.begin(), scored.begin() + top_k, scored.end(),
                          [](auto & a, auto & b) { return a.first > b.first; });
        // norm_topk_prob=False, routed_scaling_factor=1.0 → raw top-k softmax probs.
        for (int k = 0; k < top_k; k++) {
            tk_idx[t][k] = scored[k].second;
            tk_w[t][k] = scored[k].first * scale;
            used[scored[k].second] = 1;
        }
    }

    // Dequant the used routed experts once (parallel-unsafe lazy dequant is gone).
    for (int e = 0; e < n_exp; e++)
        if (used[e]) {
            exp_ws[e].gw = to_f32(ly.experts[e].gate_w);
            exp_ws[e].uw = to_f32(ly.experts[e].up_w);
            exp_ws[e].dw = to_f32(ly.experts[e].down_w);
        }

    // Pass 2 (parallel over tokens): each token's expert FFNs are independent
    // and write only to its own row, so split the token range across threads.
    int nthreads = std::max(1, ctx.n_threads);
    if (nthreads > T) nthreads = std::max(1, T);
    auto worker = [&](int t0, int t1) {
        for (int t = t0; t < t1; t++) {
            const float * normed = normed_all.data() + (size_t)t * D;
            float * tok = hidden + t * D;
            std::vector<float> routed_out(D, 0.0f), expert_out(D);
            for (int k = 0; k < top_k; k++) {
                int eid = tk_idx[t][k];
                float w = tk_w[t][k];
                swiglu_ffn_cpu(normed, expert_out.data(), D, inter_e, exp_ws[eid].gw.data(), exp_ws[eid].uw.data(),
                               exp_ws[eid].dw.data());
                for (int d = 0; d < D; d++) routed_out[d] += w * expert_out[d];
            }
            std::vector<float> shared_out(D);
            swiglu_ffn_cpu(normed, shared_out.data(), D, inter_s, sh_gw.data(), sh_uw.data(), sh_dw.data());
            for (int d = 0; d < D; d++) tok[d] += routed_out[d] + shared_out[d];
        }
    };
    if (nthreads <= 1)
        worker(0, T);
    else {
        std::vector<std::thread> pool;
        int chunk = (T + nthreads - 1) / nthreads;
        for (int ti = 0; ti < nthreads; ti++) {
            int t0 = ti * chunk, t1 = std::min(T, t0 + chunk);
            if (t0 < t1) pool.emplace_back(worker, t0, t1);
        }
        for (auto & th : pool) th.join();
    }
}

// ---------------------------------------------------------------------------
// Full LLM decoder forward
// ---------------------------------------------------------------------------

// Runs the MoE decoder. `prompt_embeds` is the fully-assembled prompt embedding
// matrix [n_prompt x D] (bos + image features + view-separator + instruction
// token embeddings), built by the caller. Generation continues until EOS.
// DS_PROFILE=1 profiling accumulators: graph build+alloc vs compute time across
// the whole decode. Used to decide whether the persistent-single-graph decode
// port is worth it (overhead-bound → yes; compute/MoE-bound → marginal).
static long long g_ds_build_us = 0, g_ds_compute_us = 0;

// Stage-bench accumulators for the `[deepseek-ocr2-stage-bench]` line. Split so
// the prefill pass (one graph over the whole prompt) is never folded into the
// per-token decode figure the T14 A/B moves — mixing them would let a prefill
// change masquerade as a decode win.
static long long g_ds_prefill_us = 0, g_ds_decode_us = 0;
static int g_ds_n_generated = 0;
// Which arm actually ran, so a silently-blocked persistent path can never be
// mistaken for a measured persistent-path number in an A/B table.
static const char * g_ds_decode_path = "per-layer";
static const char * g_ds_kv_dtype = "f32";

// ---------------------------------------------------------------------------
// T14: persistent decode-step graph
// ---------------------------------------------------------------------------
//
// The legacy decode builds, allocates, computes and frees ONE GRAPH PER LAYER
// PER TOKEN (12 graphs/token), bouncing the hidden state host<->device 24 times
// per token, then runs the final norm on the host and the LM head as yet
// another graph. For a 1280-dim/12-layer decoder that is launch- and
// transfer-bound, which is precisely the case the dev guide says a PERSISTENT
// graph fixes ("the fix is a PERSISTENT graph, not per-step rebuild").
//
// This path mirrors `qwen2vl_ocr.cpp::build_decode_step_graph`: one graph for
// the whole step (embedding lookup -> 12 layers -> final norm -> logits), with
// every tensor shape CONSTANT across steps so `ggml_backend_sched_alloc_graph`
// takes its no-realloc fast path from step 2 on. Constant shapes are why the
// KV read view spans the full `max_seq` and an F16 `kv_mask` (0 / -INF) hides
// the not-yet-written slots, rather than a per-step `[.., n_past+1]` view.
//
// NOTHING IS DELETED: the legacy per-layer path stays selectable. Commit
// c75b95d on this very engine replaced the manual masked GQA with
// flash_attn_ext with NO gate, mishandled the custom bidirectional mask, and
// shipped garbage OCR unnoticed for weeks precisely because there was nothing
// left to A/B against.
//
// Env gates:
//   DS2_LEGACY_DECODE=1  force the legacy per-layer/per-token decode
//   DS2_FAST_DECODE=1    explicitly select this path. It is the default where
//                        usable, so this only documents intent — it exists so
//                        an A/B harness can name the arm rather than select it
//                        by omission, which would silently follow the default
//                        if the default ever changes again
//   DS2_KV_BUCKET=<n>    KV read-view depth granularity for this path
//                        (default 256; 0 = read the whole allocation, which
//                        is the qwen2vl behaviour and measured 2.42x SLOWER)
//   DS2_KV_F16=1         allocate the KV cache as F16 instead of F32. This is
//                        a PRECISION change, not a pure perf refactor, so it
//                        is a SEPARATE opt-in gate: the persistent-graph
//                        byte-identity gate is run with the cache dtype held
//                        fixed, otherwise a text diff could not be attributed.
//   DS_LLM_FLASH=1       (existing) flash_attn_ext. On this path it is only
//                        honoured together with DS2_KV_F16=1, since
//                        flash_attn_ext wants an F16 K/V.
//   DS2_NO_REPEAT_NGRAM=<n>  ngram size for the no-repeat decode guard
//                        (default 20 = the reference contract's
//                        no_repeat_ngram_size; 0 restores the historical
//                        unguarded argmax). Applied at the single argmax site
//                        both decode arms share, so the arms stay comparable.

// The KV cache dtype is read once per process: prefill and decode share the
// cache, so they must agree on it. Value-parsed (G6 finding): the old
// presence-based check made DS2_KV_F16=0 ENABLE f16, inverting the repo's
// gate convention.
static ggml_type ds_kv_type() {
    const char * e = getenv("DS2_KV_F16");
    return (e && *e && strcmp(e, "0") != 0) ? GGML_TYPE_F16 : GGML_TYPE_F32;
}

// No-repeat-ngram size for the greedy decode. The reference contract
// (tests/regression/gold/deepseek-ocr2/contract.json) generates with
// no_repeat_ngram_size=20; without the guard 2 of the 5 cc0 gold pages spiral
// into the max_new cap repeating one phrase. DS2_NO_REPEAT_NGRAM=0 restores
// the historical plain argmax.
static int ds_no_repeat_ngram() {
    if (const char * e = getenv("DS2_NO_REPEAT_NGRAM")) {
        int v = atoi(e);
        return v > 0 ? v : 0;
    }
    return 20;
}

// Greedy argmax with HF-style no-repeat-ngram banning
// (core/no_repeat_ngram.h, hermetically tested): a candidate is banned when
// the last (ngram-1) generated tokens plus the candidate would repeat an
// ngram already present in the generated history. HF bans over prompt +
// generation; here the history is generation-only (the prompt reaches the
// decoder as embeddings), which for ngram=20 over a 4-token text prompt
// cannot differ. Shared with qwen2vl_ocr.cpp / internvl2_ocr.cpp.
using core_decode::argmax_no_repeat_ngram;

// Depth of the KV read view used by the persistent decode graph.
//
// qwen2vl reads the FULL allocated max_seq every step and lets the mask hide
// the tail. Measured here that is a disaster: max_seq is n_prompt+max_new+64
// (1408 for a 261-token prompt) while only ~300-480 slots are ever live, so
// every layer of every token attends over ~3x too many slots AND materialises
// three full `cont(permute(...))` copies of a [kv_dim x 1408] K/V. Interleaved
// on Metal that cost 2.37x the legacy decode.
//
// Bucketing keeps the property the persistent graph actually needs — a shape
// that is CONSTANT across consecutive steps, so sched_alloc keeps its
// no-realloc fast path — while only ever reading a little more than is live.
// The shape changes once per bucket boundary, which is rare.
// DS2_KV_BUCKET=0 restores the read-everything behaviour.
static int ds_kv_bucket() {
    if (const char * e = getenv("DS2_KV_BUCKET")) {
        int v = atoi(e);
        return v > 0 ? v : 0;
    }
    return 256;
}

static int ds_kv_view_len(int n_kv, int max_seq) {
    const int b = ds_kv_bucket();
    if (b <= 0) return max_seq;
    long long want = (long long)n_kv + 1;
    long long rounded = (want + b - 1) / b * b;
    if (rounded > max_seq) rounded = max_seq;
    return (int)rounded;
}

// Why the persistent path is or is not usable. Returns nullptr when usable,
// else a human-readable reason (reported once, so a silent fallback can never
// be mistaken for a measured result).
static const char * ds_persistent_decode_blocker(const ds_ocr2_ctx & ctx) {
    // Value-parsed (G2b, same audit class as the DS2_KV_F16 fix 73beea9f):
    // DS2_LEGACY_DECODE=0 must NOT select the legacy path.
    if (const char * e = getenv("DS2_LEGACY_DECODE"); e && strcmp(e, "0") != 0) return "DS2_LEGACY_DECODE=1";
    // DEFAULT IS THE PERSISTENT PATH, on measured evidence (Metal, M1,
    // 2026-08-05, interleaved + load-gated, 9 scored pairs on a 217-token
    // page): decode median 11474 ms -> 8192 ms (1.40x) and total 15815 ms ->
    // 12785 ms (1.24x), with the decoded text byte-identical on all 25 gold
    // fixtures and prefill/sam/qwen2_enc unchanged (461/2754/379 ms vs
    // 462/2822/376 ms). DS2_LEGACY_DECODE=1 restores the per-layer path.
    //
    // The win is NOT the graph-build amortisation this task was scoped
    // around — that is only ~1-6% of decode here. It is that one graph per
    // token replaces 13 backend dispatches and 24 host<->device hidden-state
    // transfers per token. Getting there required NOT copying qwen2vl's
    // read-the-whole-max_seq KV view (see ds_kv_view_len): verbatim, that
    // made decode 2.42x SLOWER than legacy.
    // DS_NO_KV reprocesses the whole growing sequence every step; there is no
    // single-token step graph to make persistent.
    if (getenv("DS_NO_KV")) return "DS_NO_KV=1";
    // The CPU MoE fallback runs host code BETWEEN layers, so the layers cannot
    // live in one graph.
    if (!ctx.moe_metal) return "CPU MoE (DS_MOE_CPU=1 or expert stacking failed)";
    return nullptr;
}

// Build the single-token decode graph. `n_kv` is the slot this step writes;
// `max_seq` is the (constant) allocated cache depth.
static ggml_cgraph * build_ds_decode_step_graph(ds_ocr2_ctx & ctx, ggml_context * g, int n_kv, int max_seq,
                                                int kv_len) {
    const auto & lhp = ctx.m.lhp;
    const int D = lhp.hidden, nh = lhp.heads, nkv = lhp.kv_heads, hd = lhp.head_dim;
    const int n_layers = lhp.n_layers;
    const int kv_dim = nkv * hd;
    const float eps = lhp.rms_eps;
    const float attn_scale = 1.0f / sqrtf((float)hd);
    const bool kv_f16 = (ctx.kvc.k->type == GGML_TYPE_F16);
    const bool use_flash = kv_f16 && getenv("DS_LLM_FLASH");

    ggml_cgraph * gf = ggml_new_graph_custom(g, 16384, false);

    // Inputs. ALL of them are re-set on every step by the caller: gallocr hands
    // an input's buffer to a later intermediate once that input is dead, so even
    // the "constant" kv_mask is corrupted from step 2 on if it is written once.
    ggml_tensor * tok_id = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(tok_id, "tok_id");
    ggml_set_input(tok_id);

    ggml_tensor * pos_ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, 1);
    ggml_set_name(pos_ids, "pos_ids");
    ggml_set_input(pos_ids);

    // 0.0 for populated slots, -INF for the rest. The caller unmasks slot n_kv
    // before the step so this token attends to its own freshly-written K/V.
    ggml_tensor * kv_mask = ggml_new_tensor_2d(g, GGML_TYPE_F16, kv_len, 1);
    ggml_set_name(kv_mask, "kv_mask");
    ggml_set_input(kv_mask);

    auto rmsnorm = [&](ggml_tensor * t, ggml_tensor * w) -> ggml_tensor * {
        // ensure_f32: the norm weight is F16 in an all-F16 GGUF and ggml's
        // elementwise mul rejects an f32xf16 operand pair.
        return ggml_mul(g, ggml_rms_norm(g, t, eps), ensure_f32(g, w));
    };

    // Embedding lookup runs in-graph (get_rows works for every quant type and
    // returns F32), which removes the legacy path's host dequant + upload.
    ggml_tensor * x = ggml_reshape_2d(g, ggml_get_rows(g, ctx.m.embed_tokens, tok_id), D, 1);

    for (int li = 0; li < n_layers; li++) {
        const auto & ly = ctx.m.llm_layers[li];
        ggml_tensor * residual = x;
        ggml_tensor * h = rmsnorm(x, ly.in_ln_w);

        ggml_tensor * Q = ggml_mul_mat(g, ly.q_w, h);
        ggml_tensor * K = ggml_mul_mat(g, ly.k_w, h);
        ggml_tensor * V = ggml_mul_mat(g, ly.v_w, h);

        Q = ggml_reshape_3d(g, Q, hd, nh, 1);
        K = ggml_reshape_3d(g, K, hd, nkv, 1);
        V = ggml_reshape_3d(g, V, hd, nkv, 1);

        Q = ggml_rope_ext(g, Q, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);
        K = ggml_rope_ext(g, K, pos_ids, nullptr, hd, GGML_ROPE_TYPE_NEOX, 0, lhp.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f,
                          0.0f);

        // ggml_cont before the cache write: a bare rope/reshape VIEW written
        // into (or read back from) the cache reads stale data.
        ggml_tensor * K_flat = ggml_reshape_2d(g, ggml_cont(g, K), kv_dim, 1);
        ggml_tensor * V_flat = ggml_reshape_2d(g, ggml_cont(g, V), kv_dim, 1);

        const size_t off_k = (size_t)li * ctx.kvc.k->nb[2];
        const size_t off_v = (size_t)li * ctx.kvc.v->nb[2];

        // Write shape is constant (kv_dim,1); only the byte offset moves, which
        // keeps the sched_alloc compatibility check on its fast path.
        ggml_tensor * k_write =
            ggml_view_2d(g, ctx.kvc.k, kv_dim, 1, ctx.kvc.k->nb[1], off_k + (size_t)n_kv * ctx.kvc.k->nb[1]);
        ggml_tensor * v_write =
            ggml_view_2d(g, ctx.kvc.v, kv_dim, 1, ctx.kvc.v->nb[1], off_v + (size_t)n_kv * ctx.kvc.v->nb[1]);
        ggml_build_forward_expand(gf, ggml_cpy(g, K_flat, k_write));
        ggml_build_forward_expand(gf, ggml_cpy(g, V_flat, v_write));

        // Fixed-depth read view: constant shape across every step, with kv_mask
        // hiding the unwritten tail.
        ggml_tensor * Kfull =
            ggml_reshape_3d(g, ggml_view_2d(g, ctx.kvc.k, kv_dim, kv_len, ctx.kvc.k->nb[1], off_k), hd, nkv, kv_len);
        ggml_tensor * Vfull =
            ggml_reshape_3d(g, ggml_view_2d(g, ctx.kvc.v, kv_dim, kv_len, ctx.kvc.v->nb[1], off_v), hd, nkv, kv_len);

        // This decoder is MHA (heads == kv_heads), so there is no GQA repeat.
        // Guard rather than assume: a future checkpoint with nkv < nh would
        // otherwise attend to the wrong heads silently.
        if (nh != nkv) {
            const int kv_repeat = nh / nkv;
            Kfull = ggml_reshape_4d(g, Kfull, hd, 1, nkv, kv_len);
            Kfull = ggml_repeat(g, Kfull, ggml_new_tensor_4d(g, Kfull->type, hd, kv_repeat, nkv, kv_len));
            Kfull = ggml_reshape_3d(g, Kfull, hd, nh, kv_len);
            Vfull = ggml_reshape_4d(g, Vfull, hd, 1, nkv, kv_len);
            Vfull = ggml_repeat(g, Vfull, ggml_new_tensor_4d(g, Vfull->type, hd, kv_repeat, nkv, kv_len));
            Vfull = ggml_reshape_3d(g, Vfull, hd, nh, kv_len);
        }

        Q = ggml_cont(g, ggml_permute(g, Q, 0, 2, 1, 3)); // [hd, 1, nh]
        Kfull = ggml_cont(g, ggml_permute(g, Kfull, 0, 2, 1, 3));
        Vfull = ggml_cont(g, ggml_permute(g, Vfull, 0, 2, 1, 3));

        ggml_tensor * attn;
        if (use_flash) {
            // flash_attn_ext already returns [hd, nh, T] — reshape straight, and
            // never append a trailing permute(0,2,1,3).
            attn = ggml_flash_attn_ext(g, Q, Kfull, Vfull, kv_mask, attn_scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);
            attn = ggml_reshape_2d(g, attn, D, 1);
        } else {
            // Same manual masked attention the legacy path uses by default.
            ggml_tensor * scores = ggml_mul_mat(g, Kfull, Q);
            scores = ggml_soft_max_ext(g, scores, kv_mask, attn_scale, 0.0f);
            ggml_tensor * Vt = ggml_cont(g, ggml_permute(g, Vfull, 1, 0, 2, 3));
            attn = ggml_mul_mat(g, Vt, scores);
            attn = ggml_cont(g, ggml_permute(g, attn, 0, 2, 1, 3));
            attn = ggml_reshape_2d(g, attn, D, 1);
        }

        attn = ggml_mul_mat(g, ly.o_w, attn);
        x = ggml_add(g, residual, attn);

        // FFN — identical math to build_llm_layer_attn at T == 1.
        residual = x;
        ggml_tensor * hn = rmsnorm(x, ly.post_ln_w);
        if (li == 0) {
            ggml_tensor * gate = ggml_silu(g, ggml_mul_mat(g, ly.ffn_gate_w, hn));
            ggml_tensor * up = ggml_mul_mat(g, ly.ffn_up_w, hn);
            x = ggml_add(g, residual, ggml_mul_mat(g, ly.ffn_down_w, ggml_mul(g, gate, up)));
        } else {
            const int n_exp = lhp.n_experts, K = lhp.n_experts_top;
            ggml_tensor * logits = ggml_mul_mat(g, ly.router_w, hn);
            ggml_tensor * probs = ggml_soft_max(g, logits);
            ggml_tensor * ids = ggml_top_k(g, probs, K);
            ggml_tensor * p3 = ggml_reshape_3d(g, probs, 1, n_exp, 1);
            ggml_tensor * top_w = ggml_reshape_2d(g, ggml_get_rows(g, p3, ids), K, 1);
            top_w = ggml_scale(g, top_w, lhp.routed_scaling_factor);

            ggml_tensor * hn3 = ggml_reshape_3d(g, hn, D, 1, 1);
            ggml_tensor * hnK = ggml_repeat(g, hn3, ggml_new_tensor_3d(g, hn->type, D, K, 1));
            ggml_tensor * gate = ggml_silu(g, ggml_mul_mat_id(g, ly.gate_exps, hnK, ids));
            ggml_tensor * up = ggml_mul_mat_id(g, ly.up_exps, hnK, ids);
            ggml_tensor * down = ggml_mul_mat_id(g, ly.down_exps, ggml_mul(g, gate, up), ids);

            ggml_tensor * down_p = ggml_cont(g, ggml_permute(g, down, 1, 0, 2, 3));
            ggml_tensor * w_col = ggml_reshape_3d(g, top_w, K, 1, 1);
            ggml_tensor * routed = ggml_reshape_2d(g, ggml_mul_mat(g, w_col, down_p), D, 1);

            ggml_tensor * sg = ggml_silu(g, ggml_mul_mat(g, ly.shared_gate_w, hn));
            ggml_tensor * su = ggml_mul_mat(g, ly.shared_up_w, hn);
            ggml_tensor * shared = ggml_mul_mat(g, ly.shared_down_w, ggml_mul(g, sg, su));

            x = ggml_add(g, residual, ggml_add(g, routed, shared));
        }
    }

    // Final norm + LM head in the SAME graph — the legacy path reads the hidden
    // state back to the host, norms it there and dispatches the head as another
    // graph. (The host rmsnorm and the in-graph one reduce in a different order,
    // so per-token logits differ in the last bits; the acceptance gate is the
    // decoded text, which is what argmax actually consumes.)
    x = ggml_mul(g, ggml_rms_norm(g, x, eps), ensure_f32(g, ctx.m.output_norm_w));
    ggml_tensor * lm_w = ctx.m.lm_head_w ? ctx.m.lm_head_w : ctx.m.embed_tokens;
    x = ggml_mul_mat(g, lm_w, x);
    ggml_set_name(x, "logits");
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);

    return gf;
}

static bool run_llm_decoder(ds_ocr2_ctx & ctx, const float * prompt_embeds, int n_prompt, int max_new,
                            std::vector<int32_t> & out_ids, std::vector<float> & out_confs) {
    g_ds_build_us = 0;
    g_ds_compute_us = 0;
    g_ds_prefill_us = 0;
    g_ds_decode_us = 0;
    g_ds_n_generated = 0;
    auto & lhp = ctx.m.lhp;
    int D = lhp.hidden, V = lhp.vocab_size;
    int nh = lhp.heads, nkv = lhp.kv_heads, hd = lhp.head_dim;
    int n_layers = lhp.n_layers;
    int kv_dim = nkv * hd;

    // Allocate persistent device-side KV cache. Rounded up to a multiple of 64
    // so the fixed-depth kv_mask the persistent decode graph uses satisfies
    // flash_attn_ext's KQ-mask padding when DS2_KV_F16=1 + DS_LLM_FLASH=1 select
    // it; harmless for the manual path.
    int max_seq = ((n_prompt + max_new + 64) + 63) / 64 * 64;
    if (!alloc_ds_kv_cache(ctx, max_seq)) {
        fprintf(stderr, "deepseek_ocr2: KV cache allocation failed\n");
        return false;
    }

    // Per-row embedding: dequant only the requested row on demand instead of
    // the whole 128k×1280 table (~655 MB held for the entire decode). The decode
    // touches ~one row per generated token, so the table f32 copy was pure waste.
    ggml_tensor * emb_t = ctx.m.embed_tokens;
    const auto * emb_tt = ggml_get_type_traits(emb_t->type);
    const size_t emb_row_bytes = ggml_row_size(emb_t->type, D);
    std::vector<uint8_t> emb_row;
    auto get_embedding = [&](int32_t tok_id, float * out_emb) {
        const size_t off = (size_t)tok_id * emb_row_bytes;
        if (emb_t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(emb_t, out_emb, off, (size_t)D * sizeof(float));
        } else {
            emb_row.resize(emb_row_bytes);
            ggml_backend_tensor_get(emb_t, emb_row.data(), off, emb_row_bytes);
            emb_tt->to_float(emb_row.data(), out_emb, D);
        }
    };

    // Dequant weights needed for LM head
    auto norm_w = to_f32(ctx.m.output_norm_w);
    // LM head: default runs as a Metal quantized mul_mat (DS_LMHEAD_CPU=1 forces
    // the scalar path). Only dequant the 662 MB head weight for the CPU path.
    ggml_tensor * lm_w = ctx.m.lm_head_w ? ctx.m.lm_head_w : ctx.m.embed_tokens;
    bool lmhead_cpu = getenv("DS_LMHEAD_CPU") != nullptr;
    std::vector<float> head_w;
    if (lmhead_cpu) head_w = to_f32(lm_w);

    // Diagnostic: DS_NO_KV disables the KV cache and re-runs the entire growing
    // sequence each step (n_past always 0). Slow but a ground-truth reference to
    // isolate cache bugs from prefill bugs.
    bool no_kv = getenv("DS_NO_KV") != nullptr;
    std::vector<float> full_emb(prompt_embeds, prompt_embeds + (size_t)n_prompt * D);

    // T14: persistent single-graph decode for the T==1 steps. Prefill keeps the
    // per-layer path unconditionally — it runs once, so it is not what the A/B
    // moves, and leaving it alone keeps the prefill/vision stage times identical
    // by construction.
    const char * persist_blocker = ds_persistent_decode_blocker(ctx);
    const bool use_persistent = (persist_blocker == nullptr);
    g_ds_decode_path = use_persistent ? "persistent" : "per-layer";
    g_ds_kv_dtype = (ctx.kvc.k->type == GGML_TYPE_F16) ? "f16" : "f32";
    if (ctx.verbosity >= 1) {
        if (use_persistent)
            fprintf(stderr, "deepseek_ocr2: decode path = persistent step graph (kv=%s%s)\n",
                    ctx.kvc.k->type == GGML_TYPE_F16 ? "f16" : "f32",
                    (ctx.kvc.k->type == GGML_TYPE_F16 && getenv("DS_LLM_FLASH")) ? ", flash" : "");
        else
            fprintf(stderr, "deepseek_ocr2: decode path = legacy per-layer (%s)\n", persist_blocker);
    }

    // kv_mask: 0.0 for populated slots, -INF for the rest. Rebuilt/re-uploaded
    // every step (gallocr aliasing), never written once at build time.
    std::vector<ggml_fp16_t> kv_mask_data((size_t)max_seq, ggml_fp32_to_fp16(-INFINITY));

    // Logits buffer, reused by both arms.
    std::vector<float> logits(V);

    // Run generation loop
    int n_generated = 0;
    std::vector<int32_t> cur_tokens; // tokens generated so far (single-token steps)
    int n_past = 0;

    while (n_generated < max_new) {
        // Prefill (n_past==0) processes the whole assembled prompt; subsequent
        // steps process one freshly-generated token at a time. In no_kv mode the
        // whole sequence is reprocessed every step.
        const bool is_prefill = (n_past == 0);
        const auto _step_t0 = std::chrono::steady_clock::now();

        if (use_persistent && !is_prefill) {
            // ── T14 persistent single-graph decode step ──
            // One graph: embedding lookup -> every layer -> final norm -> logits.
            const int32_t tok = cur_tokens.empty() ? 0 : cur_tokens.back();
            const int32_t tok_pos = n_past;

            // Read depth for this step: bucketed, so the shape stays constant
            // across consecutive steps but never spans the whole allocation.
            const int kv_len = ds_kv_view_len(n_past, max_seq);

            // Unmask everything written so far plus this step's own slot, which
            // the graph writes before it reads the cache back.
            for (int i = 0; i <= n_past && i < max_seq; i++) kv_mask_data[i] = ggml_fp32_to_fp16(0.0f);

            const auto _tb0 = std::chrono::steady_clock::now();
            ggml_init_params ip{ ctx.compute_meta.size(), ctx.compute_meta.data(), true };
            ggml_context * g = ggml_init(ip);
            if (!g) {
                fprintf(stderr, "deepseek_ocr2: persistent decode ggml_init failed\n");
                return false;
            }
            ggml_cgraph * gf = build_ds_decode_step_graph(ctx, g, n_past, max_seq, kv_len);
            ggml_backend_sched_reset(ctx.sched);
            if (!ggml_backend_sched_alloc_graph(ctx.sched, gf)) {
                fprintf(stderr, "deepseek_ocr2: persistent decode alloc failed\n");
                ggml_free(g);
                return false;
            }
            const auto _tb1 = std::chrono::steady_clock::now();
            g_ds_build_us += std::chrono::duration_cast<std::chrono::microseconds>(_tb1 - _tb0).count();

            // Re-set EVERY input on EVERY step. gallocr may hand an input's
            // buffer to a later intermediate once that input is dead, so the
            // "constant" kv_mask is corrupted from step 2 on if it is uploaded
            // only once — the failure mode is step 0 matching and everything
            // after diverging.
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "tok_id"), &tok, 0, sizeof(int32_t));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "pos_ids"), &tok_pos, 0, sizeof(int32_t));
            ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "kv_mask"), kv_mask_data.data(), 0,
                                    (size_t)kv_len * sizeof(ggml_fp16_t));

            const auto _t0 = std::chrono::steady_clock::now();
            if (ggml_backend_sched_graph_compute(ctx.sched, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "deepseek_ocr2: persistent decode compute failed\n");
                ggml_free(g);
                return false;
            }
            const auto _t1 = std::chrono::steady_clock::now();
            g_ds_compute_us += std::chrono::duration_cast<std::chrono::microseconds>(_t1 - _t0).count();

            ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "logits"), logits.data(), 0, (size_t)V * sizeof(float));
            ggml_free(g);
            n_past += 1;
        } else {
            int T = no_kv ? (int)(full_emb.size() / D) : ((n_past == 0) ? n_prompt : (int)cur_tokens.size());
            if (getenv("DS_DBG"))
                fprintf(stderr, "  [dbg] decode step gen=%d n_past=%d T=%d\n", n_generated, n_past, T);

            // Build input embeddings
            std::vector<float> input_emb(T * D);
            if (no_kv) {
                memcpy(input_emb.data(), full_emb.data(), (size_t)T * D * sizeof(float));
            } else if (n_past == 0) {
                memcpy(input_emb.data(), prompt_embeds, (size_t)T * D * sizeof(float));
            } else {
                for (int t = 0; t < T; t++) get_embedding(cur_tokens[t], input_emb.data() + t * D);
            }

            // Process each layer
            std::vector<float> hidden(input_emb);

            for (int li = 0; li < n_layers; li++) {
                bool is_dense = (li == 0);
                // MoE in-graph (Metal) when experts were stacked; else CPU fallback.
                bool moe_in_graph = ctx.moe_metal && !is_dense;

                // Build and run attention graph
                auto _tb0 = std::chrono::steady_clock::now();
                auto lag = build_llm_layer_attn(ctx, li, T, n_past, is_dense, moe_in_graph);
                ggml_backend_sched_reset(ctx.sched);
                if (!ggml_backend_sched_alloc_graph(ctx.sched, lag.gf)) {
                    ggml_free(lag.gctx);
                    return false;
                }
                auto _tb1 = std::chrono::steady_clock::now();
                g_ds_build_us += std::chrono::duration_cast<std::chrono::microseconds>(_tb1 - _tb0).count();

                // Set inputs
                ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "layer_input"), hidden.data(), 0,
                                        T * D * sizeof(float));

                std::vector<int32_t> pos(T);
                for (int t = 0; t < T; t++) pos[t] = n_past + t;
                ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "pos_ids"), pos.data(), 0, T * sizeof(int32_t));

                // KV cache is now persistent on device — no per-step upload needed.
                // The graph writes new K/V via ggml_cpy and reads history via ggml_view.

                // Causal mask
                int Lk = n_past + T;
                std::vector<ggml_fp16_t> mask(Lk * T);
                for (int qi = 0; qi < T; qi++)
                    for (int ki = 0; ki < Lk; ki++)
                        mask[qi * Lk + ki] = ggml_fp32_to_fp16(ki > n_past + qi ? -INFINITY : 0.0f);
                ggml_backend_tensor_set(ggml_graph_get_tensor(lag.gf, "mask"), mask.data(), 0,
                                        Lk * T * sizeof(ggml_fp16_t));

                auto _t0 = std::chrono::steady_clock::now();
                ggml_backend_sched_graph_compute(ctx.sched, lag.gf);
                auto _t1 = std::chrono::steady_clock::now();
                g_ds_compute_us += std::chrono::duration_cast<std::chrono::microseconds>(_t1 - _t0).count();

                // Read outputs
                ggml_backend_tensor_get(ggml_graph_get_tensor(lag.gf, "layer_output"), hidden.data(), 0,
                                        T * D * sizeof(float));

                // KV cache updated in-graph via ggml_cpy (no readback needed).

                ggml_free(lag.gctx);

                // MoE FFN (layers 1-11): in-graph on Metal (done above) or CPU here.
                auto _t2 = std::chrono::steady_clock::now();
                if (!is_dense && !moe_in_graph) {
                    moe_ffn_cpu(ctx, li, hidden.data(), T);
                }
                auto _t3 = std::chrono::steady_clock::now();
                if (getenv("DS_DBG"))
                    fprintf(stderr, "  [dbg] llm li=%d attn=%lldms moe=%lldms (n_threads=%d)\n", li,
                            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(_t1 - _t0).count(),
                            (long long)std::chrono::duration_cast<std::chrono::milliseconds>(_t3 - _t2).count(),
                            ctx.n_threads);

                // Diff comparison
                if (!ctx.diff_ref_path.empty() && n_past == 0) {
                    char name[64];
                    snprintf(name, sizeof(name), "llm_layer_%d", li);
                    crispembed_diff::Ref ref;
                    if (ref.load(ctx.diff_ref_path.c_str()) && ref.has(name)) {
                        auto r = ref.compare(name, hidden.data(), T * D);
                        fprintf(stderr, "  %s: cos_min=%.6f max_abs=%.6f %s\n", name, r.cos_min, r.max_abs,
                                r.is_pass() ? "PASS" : "FAIL");
                    }
                }
            }

            if (!no_kv) n_past += T;

            // Final norm + LM head (CPU)
            std::vector<float> last_hidden(D);
            rmsnorm_cpu(hidden.data() + (T - 1) * D, last_hidden.data(), D, norm_w.data(), lhp.rms_eps);

            if (lmhead_cpu) {
                linear_cpu(last_hidden.data(), logits.data(), D, V, head_w.data(), nullptr);
            } else {
                // logits = lm_w @ last_hidden on Metal (quantized; ~165M mults/token
                // off the scalar path). Build+run+free in scope (no dangling graph).
                size_t meta_sz = 1 * 1024 * 1024;
                std::vector<uint8_t> mb(meta_sz);
                ggml_init_params ip = { meta_sz, mb.data(), true };
                ggml_context * gc = ggml_init(ip);
                ggml_cgraph * gf = ggml_new_graph(gc);
                ggml_tensor * in = ggml_new_tensor_2d(gc, GGML_TYPE_F32, D, 1);
                ggml_set_name(in, "lmh_in");
                ggml_set_input(in);
                ggml_tensor * out = ggml_mul_mat(gc, lm_w, in); // [V, 1]
                ggml_set_name(out, "lmh_out");
                ggml_set_output(out);
                ggml_build_forward_expand(gf, out);
                ggml_backend_sched_reset(ctx.sched);
                ggml_backend_sched_alloc_graph(ctx.sched, gf);
                ggml_backend_tensor_set(ggml_graph_get_tensor(gf, "lmh_in"), last_hidden.data(), 0,
                                        (size_t)D * sizeof(float));
                ggml_backend_sched_graph_compute(ctx.sched, gf);
                ggml_backend_tensor_get(ggml_graph_get_tensor(gf, "lmh_out"), logits.data(), 0,
                                        (size_t)V * sizeof(float));
                ggml_free(gc);
            }
        } // end legacy per-layer step

        // Diff: logits
        if (!ctx.diff_ref_path.empty() && n_generated == 0) {
            crispembed_diff::Ref ref;
            if (ref.load(ctx.diff_ref_path.c_str()) && ref.has("logits")) {
                auto r = ref.compare("logits", logits.data(), V);
                fprintf(stderr, "  logits: cos_min=%.6f max_abs=%.6f %s\n", r.cos_min, r.max_abs,
                        r.is_pass() ? "PASS" : "FAIL");
            }
        }

        // Argmax, with the contract's no-repeat-ngram guard (both decode arms
        // reach this same site, so they stay comparable per arm).
        const int nrn = ds_no_repeat_ngram();
        int next = (nrn > 0) ? argmax_no_repeat_ngram(logits.data(), V, out_ids, nrn)
                             : (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());

        // Confidence: softmax mass of the emitted token, stabilised on the
        // global max so a guard-redirected pick cannot overflow the
        // exponentials. When the guard does not fire, logits[next] IS the
        // global max and this reduces bit-for-bit to the historical 1/sum_e.
        float max_l = *std::max_element(logits.begin(), logits.end());
        float sum_e = 0;
        for (int v = 0; v < V; v++) sum_e += expf(logits[v] - max_l);
        out_confs.push_back(expf(logits[next] - max_l) / sum_e);

        out_ids.push_back(next);
        n_generated++;

        {
            const auto _step_us =
                std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - _step_t0)
                    .count();
            // The prefill step also emits the first token, so it is counted as
            // prefill only; every later step is decode. `break` paths below run
            // after this, so no step is ever dropped from the totals.
            (is_prefill ? g_ds_prefill_us : g_ds_decode_us) += _step_us;
        }
        g_ds_n_generated = n_generated;

        if (getenv("DS_DBG")) {
            const char * pc = (next >= 0 && next < ctx.tok_vocab_size) ? ctx.id_to_piece[next].c_str() : "?";
            fprintf(stderr, "  [gen %d] id=%d piece=%s\n", n_generated - 1, next, pc);
        }

        if (next == lhp.eos_token_id) break;

        // Next step: single token (or, in no_kv mode, append to the full sequence)
        cur_tokens = { (int32_t)next };
        if (no_kv) {
            size_t off = full_emb.size();
            full_emb.resize(off + D);
            get_embedding(next, full_emb.data() + off);
        }
    }

    if (getenv("DS_PROFILE"))
        fprintf(stderr,
                "[ds-profile] decode: %d tokens, graph-build+alloc=%lldms, compute=%lldms "
                "(build is %.0f%% of build+compute)\n",
                n_generated, g_ds_build_us / 1000, g_ds_compute_us / 1000,
                100.0 * g_ds_build_us / (double)(g_ds_build_us + g_ds_compute_us + 1));

    return true;
}

// ---------------------------------------------------------------------------
// Tokenizer decode
// ---------------------------------------------------------------------------

static std::string decode_tokens(const ds_ocr2_ctx & ctx, const int32_t * ids, int n) {
    // Concatenate the byte-encoded vocab pieces (skipping special markers),
    // then map the utf-8 codepoints back to raw bytes via the shared
    // core_bpe decoder (inverse of byte_encoder(); 'Ġ' -> space, etc.).
    std::string merged;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id == ctx.m.lhp.eos_token_id) continue;
        if (id < 0 || id >= ctx.tok_vocab_size) continue;
        const auto & piece = ctx.id_to_piece[id];
        // Skip special marker tokens like <｜...｜>.
        if (piece.size() >= 2 && piece[0] == '<' && piece.back() == '>') continue;
        merged += piece;
    }

    return core_bpe::unicode_to_bytes(merged);
}

// ---------------------------------------------------------------------------
// C ABI wrappers
// ---------------------------------------------------------------------------

struct deepseek_ocr2_context {
    ds_ocr2_ctx inner;
    std::string result;
    std::vector<float> char_confidences;
};

deepseek_ocr2_context * deepseek_ocr2_init(const char * model_path, int n_threads) {
    auto * c = new deepseek_ocr2_context;
    auto & ctx = c->inner;
    ctx.n_threads = n_threads;

    // Parity harness: when DS_REF points at a crispembed_diff GGUF dump, each
    // stage (sam_output, qwen2_enc_output, projector_output, llm logits) is
    // compared against the reference and a cos_min/max_abs line is printed.
    // See tools/dump_deepseek_ocr2_reference.py.
    if (const char * ref = getenv("DS_REF")) ctx.diff_ref_path = ref;

    if (!load_hparams(ctx, model_path)) {
        fprintf(stderr, "deepseek_ocr2: failed to load hparams\n");
        delete c;
        return nullptr;
    }

    // DS2_FORCE_CPU=1 runs the whole engine on the CPU backend. `--gpu-backend
    // cpu` cannot do this: crispasr_init_gpu_backend() only scans GPU/iGPU
    // devices, so a "cpu" preference falls through to ggml_backend_init_best()
    // and silently returns Metal. That gap is T18's to fix globally; this is the
    // engine-local lever the T14 gate needs to A/B both decode paths on both
    // backends, and it follows the CRISPEMBED_TESSERACT_FORCE_CPU precedent.
    if (getenv("DS2_FORCE_CPU")) {
        ctx.backend = ggml_backend_cpu_init();
        if (ctx.backend) ggml_backend_cpu_set_n_threads(ctx.backend, n_threads);
        fprintf(stderr, "deepseek_ocr2: DS2_FORCE_CPU=1 — CPU backend\n");
    } else {
        ctx.backend = crispasr_init_gpu_backend();
    }
    if (!ctx.backend) {
        ctx.backend = ggml_backend_cpu_init();
        if (ctx.backend) ggml_backend_cpu_set_n_threads(ctx.backend, n_threads);
    }
    if (!ctx.backend) {
        delete c;
        return nullptr;
    }

    ctx.backend_cpu = ggml_backend_is_cpu(ctx.backend) ? nullptr : ggml_backend_cpu_init();
    if (ctx.backend_cpu) ggml_backend_cpu_set_n_threads(ctx.backend_cpu, n_threads);

    std::vector<ggml_backend_t> backends;
    backends.push_back(ctx.backend);
    if (ctx.backend_cpu) backends.push_back(ctx.backend_cpu);
    ctx.sched = ggml_backend_sched_new(backends.data(), nullptr, (int)backends.size(), 32768, false, false);
    ctx.compute_meta.resize(16 * 1024 * 1024);

    auto _it = std::chrono::steady_clock::now();
    auto init_ms = [&](const char * w) {
        if (!getenv("DS_DBG")) return;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "  [time] init.%s %lldms\n", w,
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - _it).count());
        _it = now;
    };
    if (!load_tensors(ctx, model_path)) {
        fprintf(stderr, "deepseek_ocr2: failed to load tensors\n");
        delete c;
        return nullptr;
    }
    init_ms("load_tensors");

    precompute_rpe_tables(ctx);

    // Stack MoE experts for the Metal ggml_mul_mat_id decode path (default).
    // DS_MOE_CPU=1 keeps the per-token CPU-scalar moe_ffn_cpu (slower, but the
    // reference path / fallback for platforms where mul_mat_id misbehaves).
    // A prestacked GGUF (converter #4) already loaded gate_exps/up_exps/down_exps
    // directly, so skip the runtime copy — the graph path is ready as-is.
    if (!getenv("DS_MOE_CPU")) {
        if (ctx.moe_prestacked) {
            ctx.moe_metal = true;
            fprintf(stderr, "deepseek_ocr2: using prestacked MoE experts (no runtime stacking)\n");
        } else {
            ctx.moe_metal = stack_moe_experts(ctx);
            if (!ctx.moe_metal) fprintf(stderr, "deepseek_ocr2: MoE expert stacking failed — using CPU MoE\n");
        }
    }
    init_ms("stack_moe_experts");

    if (ctx.verbosity >= 1) {
        auto & s = ctx.m.shp;
        auto & q = ctx.m.qhp;
        auto & l = ctx.m.lhp;
        fprintf(stderr, "deepseek_ocr2: loaded %s\n", model_path);
        fprintf(stderr, "  sam: %dL %dd %dH patch=%d img=%d ws=%d\n", s.depth, s.hidden, s.heads, s.patch_size,
                s.image_size, s.window_size);
        fprintf(stderr, "  qwen2_enc: %dL %dd %dH/%dKV inter=%d\n", q.depth, q.hidden, q.heads, q.kv_heads,
                q.intermediate);
        fprintf(stderr, "  llm: %dL %dd %dH/%dKV vocab=%d n_exp=%d top_%d\n", l.n_layers, l.hidden, l.heads, l.kv_heads,
                l.vocab_size, l.n_experts, l.n_experts_top);
        fprintf(stderr, "  tokenizer: %d tokens\n", ctx.tok_vocab_size);
    }

    return c;
}

static void free_ds_kv_cache(ds_ocr2_ctx & c) {
    if (c.kvc.buf) {
        ggml_backend_buffer_free(c.kvc.buf);
        c.kvc.buf = nullptr;
    }
    if (c.kvc.ctx) {
        ggml_free(c.kvc.ctx);
        c.kvc.ctx = nullptr;
    }
    c.kvc.allocated = false;
    c.kvc.n_past = 0;
}

static bool alloc_ds_kv_cache(ds_ocr2_ctx & c, int max_seq) {
    auto & kv = c.kvc;
    const ggml_type kt = ds_kv_type();
    if (kv.allocated && kv.max_seq >= max_seq && kv.k && kv.k->type == kt) {
        kv.n_past = 0;
        if (kv.buf) ggml_backend_buffer_clear(kv.buf, 0);
        return true;
    }
    free_ds_kv_cache(c);

    int kv_dim = c.m.lhp.kv_heads * c.m.lhp.head_dim;
    int nl = c.m.lhp.n_layers;

    size_t mem = 2 * ggml_tensor_overhead() + ggml_graph_overhead();
    ggml_init_params ip = { mem, nullptr, true };
    kv.ctx = ggml_init(ip);
    if (!kv.ctx) return false;

    // DS2_KV_F16=1 halves the cache and its per-token read traffic. It is a
    // PRECISION change, so it is gated separately from the persistent-graph
    // refactor: mixing them would make a text diff unattributable.
    kv.k = ggml_new_tensor_3d(kv.ctx, kt, kv_dim, max_seq, nl);
    kv.v = ggml_new_tensor_3d(kv.ctx, kt, kv_dim, max_seq, nl);

    kv.buf = ggml_backend_alloc_ctx_tensors(kv.ctx, c.backend);
    if (!kv.buf) {
        fprintf(stderr, "deepseek_ocr2: KV cache alloc failed\n");
        ggml_free(kv.ctx);
        kv.ctx = nullptr;
        return false;
    }
    ggml_backend_buffer_clear(kv.buf, 0);

    kv.max_seq = max_seq;
    kv.n_past = 0;
    kv.allocated = true;

    size_t bytes = ggml_backend_buffer_get_size(kv.buf);
    fprintf(stderr, "deepseek_ocr2: KV cache: %d layers, max_seq=%d, kv_dim=%d, %s, %.1f MB\n", nl, max_seq, kv_dim,
            kt == GGML_TYPE_F16 ? "f16" : "f32", (float)bytes / 1024 / 1024);
    return true;
}

void deepseek_ocr2_free(deepseek_ocr2_context * ctx) {
    if (!ctx) return;
    auto & c = ctx->inner;
    free_ds_kv_cache(c);
    if (c.sched) ggml_backend_sched_free(c.sched);
    if (c.moe_buf) ggml_backend_buffer_free(c.moe_buf);
    if (c.moe_ctx) ggml_free(c.moe_ctx);
    if (c.moe_view_ctx) ggml_free(c.moe_view_ctx);
    // model_buf/model_ctx alias model_wl — free via free_weights (also unmaps).
    c.model_buf = nullptr;
    c.model_ctx = nullptr;
    core_gguf::free_weights(c.model_wl);
    if (c.backend) ggml_backend_free(c.backend);
    if (c.backend_cpu) ggml_backend_free(c.backend_cpu);
    delete ctx;
}

const char * deepseek_ocr2_recognize_raw(deepseek_ocr2_context * ctx, const uint8_t * px, int w, int h, int ch,
                                         int * out_len) {
    if (!ctx || !px) {
        if (out_len) *out_len = 0;
        return "";
    }

    // Isolation test: DS_TEXT_TEST runs the LLM decoder as a pure language
    // model (no vision) to verify the decoder/MoE/rope in isolation.
    if (const char * tt = getenv("DS_TEXT_TEST")) {
        auto & mdl = ctx->inner.m;
        int D = mdl.lhp.hidden;
        auto embed_w = to_f32(mdl.embed_tokens);
        std::vector<int32_t> ids = { 0 }; // bos
        auto more = core_bpe::legacy_whitespace()
                        ? core_bpe::tokenize_simple(ctx->inner.token_to_id, ctx->inner.merge_rank, tt)
                        : core_bpe::tokenize_deepseek(ctx->inner.token_to_id, ctx->inner.merge_rank, tt);
        ids.insert(ids.end(), more.begin(), more.end());
        std::vector<float> pe((size_t)ids.size() * D);
        for (size_t i = 0; i < ids.size(); i++)
            memcpy(pe.data() + i * D, embed_w.data() + (size_t)ids[i] * D, D * sizeof(float));
        fprintf(stderr, "  [TEXT_TEST] prompt=\"%s\" ids:", tt);
        for (int id : ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
        std::vector<int32_t> g;
        std::vector<float> gc;
        run_llm_decoder(ctx->inner, pe.data(), (int)ids.size(), 40, g, gc);
        ctx->result = decode_tokens(ctx->inner, g.data(), (int)g.size());
        if (out_len) *out_len = (int)ctx->result.size();
        return ctx->result.c_str();
    }

    // `[deepseek-ocr2-stage-bench]` (CRISPEMBED_DEEPSEEK_OCR2_BENCH=1). Every
    // span below is measured from its OWN stage start and the clock starts here,
    // inside recognize — model load happened in deepseek_ocr2_init and is not in
    // any of these figures, so `total` is net-of-load by construction (the
    // ppocrv6 convention, without ppocrv6's stage-entry correction: no stage
    // here loads anything).
    const bool ds_bench = getenv("CRISPEMBED_DEEPSEEK_OCR2_BENCH") != nullptr;
    const auto _b_start = std::chrono::steady_clock::now();

    auto & s = ctx->inner.m.shp;
    int imgS = s.image_size;

    // Preprocess like the HF reference: ImageOps.pad(image, (imgS, imgS)) —
    // resize preserving aspect ratio to fit inside imgS×imgS, center, and pad
    // the borders with the mean colour (gray 127). Then normalize to [-1,1]
    // (mean=std=0.5). The padded border normalizes to exactly 0.
    float scale = std::min((float)imgS / w, (float)imgS / h);
    int rw = std::max(1, (int)lroundf(w * scale));
    int rh = std::max(1, (int)lroundf(h * scale));
    int ox = (imgS - rw) / 2;
    int oy = (imgS - rh) / 2;

    std::vector<float> pixels(3 * imgS * imgS);
    for (int c = 0; c < 3; c++) {
        int ci = std::min(c, ch - 1);
        for (int y = 0; y < imgS; y++) {
            for (int x = 0; x < imgS; x++) {
                float val;
                if (x < ox || x >= ox + rw || y < oy || y >= oy + rh) {
                    val = s.image_mean[c]; // gray padding -> normalizes to 0
                } else {
                    // Bilinear sample from source at the un-scaled position.
                    float sx = (x - ox + 0.5f) / scale - 0.5f;
                    float sy = (y - oy + 0.5f) / scale - 0.5f;
                    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
                    float dx = sx - x0, dy = sy - y0;
                    int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
                    x0 = std::min(std::max(x0, 0), w - 1);
                    y0 = std::min(std::max(y0, 0), h - 1);
                    auto P = [&](int xx, int yy) { return (float)px[(yy * w + xx) * ch + ci] / 255.0f; };
                    val = P(x0, y0) * (1 - dx) * (1 - dy) + P(x1, y0) * dx * (1 - dy) + P(x0, y1) * (1 - dx) * dy +
                          P(x1, y1) * dx * dy;
                }
                pixels[c * imgS * imgS + y * imgS + x] = (val - s.image_mean[c]) / s.image_std[c];
            }
        }
    }

    // G2 (F5): dynamic-crop mode, DEFAULT ON since G2b (DS2_CROP_MODE=0
    // restores the single-view path). Mirrors the reference infer() with
    // crop_mode=True — the contract's own configuration: an image over 768 px
    // in either dimension additionally yields N=2..6 local 768² tiles
    // (dynamic_preprocess: closest-aspect grid (gw,gh) with 2<=gw*gh<=6, whole
    // image resized to (768*gw, 768*gh), split in raster order). The 1024²
    // padded global view above is unchanged in both modes. G2b flip evidence
    // (tests/results/g2b/): cc0 raw CER mean Metal 0.657→0.236, CPU
    // 0.279→0.185 (beats the A4 reference's 0.187); the two G2-recorded
    // regressions are formatting-only drift (receipt_historical: one
    // bold-vs-plain near-tie at char 82 then markdown-list self-conditioning,
    // alnum-content CER flat; synth_01_noise: four inserted colons).
    const int tileS = 768;
    const bool crop_mode = [] {
        const char * e = getenv("DS2_CROP_MODE");
        return !e || strcmp(e, "0") != 0;
    }();
    int crop_gw = 1, crop_gh = 1;
    std::vector<std::vector<float>> crop_px; // per tile: 3*768*768, normalized
    if (crop_mode && (w > tileS || h > tileS)) {
        // find_closest_aspect_ratio over all (i,j) with 2 <= i*j <= 6, sorted
        // by i*j (ties among equal products broken by i — the reference's set
        // iteration order is unspecified there; only exact-tie targets differ).
        std::vector<std::pair<int, int>> ratios;
        for (int i = 1; i <= 6; i++)
            for (int j = 1; j <= 6; j++)
                if (i * j >= 2 && i * j <= 6) ratios.push_back({ i, j });
        std::stable_sort(ratios.begin(), ratios.end(),
                         [](auto & a, auto & b) { return a.first * a.second < b.first * b.second; });
        float ar = (float)w / h;
        float best_diff = 1e30f;
        std::pair<int, int> best = { 1, 1 };
        for (auto & r : ratios) {
            float rd = fabsf(ar - (float)r.first / r.second);
            if (rd < best_diff) {
                best_diff = rd;
                best = r;
            } else if (rd == best_diff && (float)w * h > 0.5f * tileS * tileS * r.first * r.second) {
                best = r;
            }
        }
        crop_gw = best.first;
        crop_gh = best.second;
        const int tw = tileS * crop_gw, th = tileS * crop_gh;
        // PIL Image.resize default = antialiased bicubic a=-0.5 on uint8 (the
        // blueprint resizes the PIL image, then ToTensor+Normalize). Round and
        // clamp to byte like PIL before normalizing.
        std::vector<float> plane_in((size_t)w * h), plane_out((size_t)tw * th);
        std::vector<std::vector<float>> resized(3, std::vector<float>((size_t)tw * th));
        for (int c = 0; c < 3; c++) {
            int ci = std::min(c, ch - 1);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) plane_in[(size_t)y * w + x] = (float)px[((size_t)y * w + x) * ch + ci];
            bicubic_aa_resample_plane(plane_in.data(), w, h, plane_out.data(), tw, th, -0.5f);
            for (size_t i = 0; i < plane_out.size(); i++) {
                float v = roundf(plane_out[i]);
                resized[c][i] = std::min(255.0f, std::max(0.0f, v));
            }
        }
        crop_px.resize((size_t)crop_gw * crop_gh);
        for (int ty = 0; ty < crop_gh; ty++)
            for (int tx = 0; tx < crop_gw; tx++) {
                auto & tile = crop_px[(size_t)ty * crop_gw + tx];
                tile.resize((size_t)3 * tileS * tileS);
                for (int c = 0; c < 3; c++)
                    for (int y = 0; y < tileS; y++)
                        for (int x = 0; x < tileS; x++) {
                            float v01 = resized[c][(size_t)(ty * tileS + y) * tw + (tx * tileS + x)] / 255.0f;
                            tile[(size_t)c * tileS * tileS + (size_t)y * tileS + x] =
                                (v01 - s.image_mean[c]) / s.image_std[c];
                        }
            }
        if (getenv("DS_DBG"))
            fprintf(stderr, "  [dbg] crop_mode: %dx%d source -> %dx%d grid (%zu tiles)\n", w, h, crop_gw, crop_gh,
                    crop_px.size());
    }

    const auto _b_prep = std::chrono::steady_clock::now();

    bool dbg_t = getenv("DS_DBG") != nullptr;
    auto _ts = std::chrono::steady_clock::now();
    auto stage_ms = [&](const char * name) {
        if (!dbg_t) return;
        auto now = std::chrono::steady_clock::now();
        fprintf(stderr, "  [time] %s %lldms\n", name,
                (long long)std::chrono::duration_cast<std::chrono::milliseconds>(now - _ts).count());
        _ts = now;
    };

    // 1. SAM vision encoder
    std::vector<float> sam_features;
    int n_sam_tokens, sam_dim;
    if (!encode_sam(ctx->inner, pixels.data(), sam_features, n_sam_tokens, sam_dim)) {
        fprintf(stderr, "deepseek_ocr2: SAM encoding failed\n");
        if (out_len) *out_len = 0;
        return "";
    }
    stage_ms("sam");
    const auto _b_sam = std::chrono::steady_clock::now();

    // 2. Qwen2 bidirectional encoder
    std::vector<float> enc_out;
    int n_enc_tokens, enc_dim;
    if (!encode_qwen2(ctx->inner, sam_features.data(), n_sam_tokens, sam_dim, enc_out, n_enc_tokens, enc_dim)) {
        fprintf(stderr, "deepseek_ocr2: Qwen2 encoder failed\n");
        if (out_len) *out_len = 0;
        return "";
    }
    stage_ms("qwen2_enc");
    const auto _b_enc = std::chrono::steady_clock::now();

    // 3. Project to LLM dimension
    std::vector<float> proj_out;
    if (!project_to_llm(ctx->inner, enc_out.data(), n_enc_tokens, enc_dim, proj_out)) {
        fprintf(stderr, "deepseek_ocr2: projection failed\n");
        if (out_len) *out_len = 0;
        return "";
    }
    stage_ms("projector");

    // Crop tiles through the same SAM → qwen2(query_768) → projector stack.
    // Reference embedding order (masked_scatter fill): local tile features
    // FIRST (raster order), then the 256 global features, then view_seperator.
    std::vector<float> crops_proj;
    int n_crop_tokens = 0;
    for (size_t t = 0; t < crop_px.size(); t++) {
        std::vector<float> c_sam, c_enc, c_proj;
        int c_n_sam, c_sam_dim, c_n_enc, c_enc_dim;
        if (!encode_sam(ctx->inner, crop_px[t].data(), c_sam, c_n_sam, c_sam_dim, tileS) ||
            !encode_qwen2(ctx->inner, c_sam.data(), c_n_sam, c_sam_dim, c_enc, c_n_enc, c_enc_dim) ||
            !project_to_llm(ctx->inner, c_enc.data(), c_n_enc, c_enc_dim, c_proj)) {
            fprintf(stderr, "deepseek_ocr2: crop tile %zu encoding failed\n", t);
            if (out_len) *out_len = 0;
            return "";
        }
        crops_proj.insert(crops_proj.end(), c_proj.begin(), c_proj.end());
        n_crop_tokens += c_n_enc;
        if (getenv("DS_DBG")) fprintf(stderr, "  [dbg] crop tile %zu: sam=%d qwen2=%d tokens\n", t, c_n_sam, c_n_enc);
    }
    if (!crop_px.empty()) stage_ms("crops");
    const auto _b_proj = std::chrono::steady_clock::now();

    fprintf(stderr, "deepseek_ocr2: stages done — sam=%d/%d qwen2=%d/%d proj=%d image tokens (+%d crop tokens)\n",
            n_sam_tokens, sam_dim, n_enc_tokens, enc_dim, n_enc_tokens, n_crop_tokens);

    // 4. Assemble the LLM prompt embeddings. The HF reference (infer + plain
    //    template, prompt "<image>\nFree OCR.") builds the token sequence:
    //        [bos] + <image>*N + <view_sep> + tokenize("\nFree OCR.")
    //    where the N image placeholders + the view-separator placeholder are
    //    masked-scatter-replaced by [global_features (N), view_seperator (1)].
    //    We build the embedding matrix directly: text positions use
    //    embed_tokens, image positions use the projected vision features, and
    //    the separator position uses the learned v.view_separator embedding.
    auto & mdl = ctx->inner.m;
    auto & lhp = mdl.lhp;
    int D = lhp.hidden;
    auto embed_w = to_f32(mdl.embed_tokens);
    auto vsep = to_f32(mdl.view_separator); // [D]

    // Instruction text after <image>. DeepSeek-OCR2 plain prompt: "\nFree OCR."
    //
    // `tokenize_simple` DELETED that leading newline, so this emitted 3 ids
    // where the reference tokenizer emits 4: HF gives [201, 21431, 126041, 16]
    // ("\n" "Free" " OCR" "."), we sent [21431, 126041, 16]. The prompt is
    // fixed text, so the loss was deterministic on every page. Now transcribes
    // the pre_tokenizer DeepSeek-OCR-2's tokenizer.json declares and matches HF
    // exactly. CRISPEMBED_BPE_LEGACY_WHITESPACE=1 restores the old 3-id prompt.
    std::vector<int32_t> instr_ids =
        core_bpe::legacy_whitespace()
            ? core_bpe::tokenize_simple(ctx->inner.token_to_id, ctx->inner.merge_rank, "\nFree OCR.")
            : core_bpe::tokenize_deepseek(ctx->inner.token_to_id, ctx->inner.merge_rank, "\nFree OCR.");

    // Crop mode: the reference's contiguous image-token block is filled in
    // feature order [local tiles (144·N), global (256), view_seperator], so the
    // sequence is [bos][crop features][global features][vsep][instr]. With no
    // crops this reduces to the historical layout exactly.
    int n_img_tokens = n_enc_tokens; // 256 global features
    int n_prompt = 1 /*bos*/ + n_crop_tokens + n_img_tokens + 1 /*view_sep*/ + (int)instr_ids.size();
    std::vector<float> prompt_embeds((size_t)n_prompt * D);

    int row = 0;
    auto put_tok = [&](int32_t id) {
        memcpy(prompt_embeds.data() + (size_t)row * D, embed_w.data() + (size_t)id * D, D * sizeof(float));
        row++;
    };
    put_tok(0); // bos = <｜begin▁of▁sentence｜>
    for (int i = 0; i < n_crop_tokens; i++) {
        memcpy(prompt_embeds.data() + (size_t)row * D, crops_proj.data() + (size_t)i * D, D * sizeof(float));
        row++;
    }
    for (int i = 0; i < n_img_tokens; i++) {
        memcpy(prompt_embeds.data() + (size_t)row * D, proj_out.data() + (size_t)i * D, D * sizeof(float));
        row++;
    }
    memcpy(prompt_embeds.data() + (size_t)row * D, vsep.data(), D * sizeof(float)); // view separator
    row++;
    for (int32_t id : instr_ids) put_tok(id);

    if (getenv("DS_DBG")) {
        fprintf(stderr,
                "  [dbg] prompt: bos + %d crop + %d img + sep + %zu instr = %d tokens; instr_ids:", n_crop_tokens,
                n_img_tokens, instr_ids.size(), n_prompt);
        for (int32_t id : instr_ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }

    const auto _b_prompt = std::chrono::steady_clock::now();

    // 5. LLM decoder
    std::vector<int32_t> gen_ids;
    std::vector<float> gen_confs;
    if (!run_llm_decoder(ctx->inner, prompt_embeds.data(), n_prompt, 1024, gen_ids, gen_confs)) {
        fprintf(stderr, "deepseek_ocr2: LLM decode failed\n");
        if (out_len) *out_len = 0;
        return "";
    }

    if (getenv("DS_DBG")) {
        fprintf(stderr, "  [dbg] gen_ids (%zu):", gen_ids.size());
        for (int id : gen_ids) fprintf(stderr, " %d", id);
        fprintf(stderr, "\n");
    }
    ctx->result = decode_tokens(ctx->inner, gen_ids.data(), (int)gen_ids.size());
    ctx->char_confidences = std::move(gen_confs);

    if (ds_bench) {
        const auto _b_end = std::chrono::steady_clock::now();
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        // `prompt` covers the embed-table dequant + prompt assembly; `prefill`
        // and `decode` are reported by the decoder itself so the split survives
        // whichever decode path ran. decode_path names the arm under A/B.
        fprintf(stderr,
                "[deepseek-ocr2-stage-bench] preprocess=%.1f ms sam=%.1f ms qwen2_enc=%.1f ms projector=%.1f ms "
                "prompt=%.1f ms prefill=%.1f ms decode=%.1f ms total=%.1f ms image_tokens=%d crop_tokens=%d "
                "prompt_tokens=%d gen_tokens=%d decode_path=%s kv=%s\n",
                ms(_b_start, _b_prep), ms(_b_prep, _b_sam), ms(_b_sam, _b_enc), ms(_b_enc, _b_proj),
                ms(_b_proj, _b_prompt), g_ds_prefill_us / 1000.0, g_ds_decode_us / 1000.0, ms(_b_start, _b_end),
                n_img_tokens, n_crop_tokens, n_prompt, g_ds_n_generated, g_ds_decode_path, g_ds_kv_dtype);
    }

    if (out_len) *out_len = (int)ctx->result.size();
    return ctx->result.c_str();
}

const char * deepseek_ocr2_recognize(deepseek_ocr2_context * ctx, const float * px, int w, int h, int * out_len) {
    if (!ctx || !px) {
        if (out_len) *out_len = 0;
        return "";
    }
    std::vector<uint8_t> rgb(w * h * 3);
    for (int i = 0; i < w * h; i++) {
        uint8_t v = (uint8_t)std::min(255.0f, std::max(0.0f, px[i] * 255.0f));
        rgb[i * 3] = v;
        rgb[i * 3 + 1] = v;
        rgb[i * 3 + 2] = v;
    }
    return deepseek_ocr2_recognize_raw(ctx, rgb.data(), w, h, 3, out_len);
}

const float * deepseek_ocr2_confidences(const deepseek_ocr2_context * ctx, int * n_tokens) {
    if (!ctx || ctx->char_confidences.empty()) {
        if (n_tokens) *n_tokens = 0;
        return nullptr;
    }
    if (n_tokens) *n_tokens = (int)ctx->char_confidences.size();
    return ctx->char_confidences.data();
}

float deepseek_ocr2_mean_confidence(const deepseek_ocr2_context * ctx) {
    if (!ctx || ctx->char_confidences.empty()) return 0.0f;
    double sum = 0;
    for (float c : ctx->char_confidences) sum += c;
    return (float)(sum / ctx->char_confidences.size());
}
