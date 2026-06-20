// instructir.cpp — InstructIR all-in-one restoration (CPU-scalar).
//
// NAFNet U-Net backbone with ICB (Instruction Condition Block) text injection.
// Encoder: 4 levels [2,2,4,8 NAFBlocks] + ICB + Conv2d(k=2,s=2) downsample
// Middle: 4 NAFBlocks at 512ch
// Decoder: 4 levels [upsample + skip + 2 NAFBlocks + ICB]
// Intro: Conv3x3(3→32), Ending: Conv3x3(32→3) + global residual
//
// ICB: sigmoid(Linear(text_embd→C)) * (x*gamma+beta) → NAFBlock → +x

#include "instructir.h"
#include "core/gguf_loader.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ── Helpers (same as nafnet_denoise.cpp / safmn_sr.cpp) ──

// GPU-safe: uses ggml_backend_tensor_get instead of direct tensor->data
static const float * to_f32(const ggml_tensor * t, std::vector<float> & buf) {
    if (!t) return nullptr;
    int64_t n = ggml_nelements(t);
    buf.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; i++) buf[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        size_t raw_sz = ggml_nbytes(t);
        std::vector<uint8_t> raw(raw_sz);
        ggml_backend_tensor_get(t, raw.data(), 0, raw_sz);
        const auto * traits = ggml_get_type_traits(t->type);
        if (traits && traits->to_float) traits->to_float(raw.data(), buf.data(), n);
        else memset(buf.data(), 0, n * sizeof(float));
    }
    return buf.data();
}

static void conv2d(const float * in, int ic, int h, int w,
                   const float * wt, const float * bi,
                   int oc, int kh, int kw, int pad, int stride, int groups,
                   float * out) {
    int oh = (h + 2 * pad - kh) / stride + 1;
    int ow = (w + 2 * pad - kw) / stride + 1;
    int ic_pg = ic / groups, oc_pg = oc / groups;
    for (int g = 0; g < groups; g++)
        for (int o = 0; o < oc_pg; o++) {
            int oc_a = g * oc_pg + o;
            float b = bi ? bi[oc_a] : 0.0f;
            for (int oy = 0; oy < oh; oy++)
                for (int ox = 0; ox < ow; ox++) {
                    float sum = b;
                    for (int c = 0; c < ic_pg; c++) {
                        int ic_a = g * ic_pg + c;
                        for (int ky = 0; ky < kh; ky++)
                            for (int kx = 0; kx < kw; kx++) {
                                int iy = oy * stride + ky - pad, ix = ox * stride + kx - pad;
                                if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                    sum += in[ic_a * h * w + iy * w + ix]
                                         * wt[oc_a * ic_pg * kh * kw + c * kh * kw + ky * kw + kx];
                            }
                    }
                    out[oc_a * oh * ow + oy * ow + ox] = sum;
                }
        }
}

static void layernorm2d(const float * in, int c, int h, int w,
                        const float * wt, const float * bi, float * out) {
    int hw = h * w;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            float mean = 0;
            for (int ch = 0; ch < c; ch++) mean += in[ch * hw + y * w + x];
            mean /= c;
            float var = 0;
            for (int ch = 0; ch < c; ch++) {
                float d = in[ch * hw + y * w + x] - mean; var += d * d;
            }
            var /= c;
            float inv = 1.0f / sqrtf(var + 1e-6f);
            for (int ch = 0; ch < c; ch++)
                out[ch * hw + y * w + x] =
                    (in[ch * hw + y * w + x] - mean) * inv * wt[ch] + bi[ch];
        }
}

static void simple_gate(const float * in, int c2, int hw, float * out) {
    int c = c2 / 2;
    for (int ch = 0; ch < c; ch++)
        for (int i = 0; i < hw; i++)
            out[ch * hw + i] = in[ch * hw + i] * in[(ch + c) * hw + i];
}

static void pixel_shuffle(const float * in, int c_in, int h, int w,
                           int r, float * out) {
    int c_out = c_in / (r * r), oh = h * r, ow = w * r;
    for (int c = 0; c < c_out; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                int ic = c * r * r + (y % r) * r + (x % r);
                out[c * oh * ow + y * ow + x] = in[ic * h * w + (y / r) * w + (x / r)];
            }
}

// ── NAFBlock forward ──

struct nafblock_wt {
    ggml_tensor * beta, * gamma;
    ggml_tensor * norm1_w, * norm1_b;
    ggml_tensor * conv1_w, * conv1_b; // 1x1, C→2C
    ggml_tensor * conv2_w, * conv2_b; // DW 3x3
    ggml_tensor * sca_w, * sca_b;     // 1x1
    ggml_tensor * conv3_w, * conv3_b; // 1x1, C→C
    ggml_tensor * norm2_w, * norm2_b;
    ggml_tensor * conv4_w, * conv4_b; // 1x1, C→2C
    ggml_tensor * conv5_w, * conv5_b; // 1x1, C→C
};

static void nafblock_forward(float * x, int c, int h, int w,
                              const nafblock_wt & wt,
                              std::vector<float> & dq1, std::vector<float> & dq2) {
    int hw = h * w, c2 = c * 2;
    if (!wt.beta || !wt.gamma || !wt.norm1_w || !wt.conv1_w || !wt.conv2_w ||
        !wt.sca_w || !wt.conv3_w || !wt.norm2_w || !wt.conv4_w || !wt.conv5_w) {
        fprintf(stderr, "nafblock: NULL tensor! beta=%p gamma=%p n1w=%p c1w=%p c2w=%p sca=%p c3w=%p n2w=%p c4w=%p c5w=%p\n",
                (void*)wt.beta, (void*)wt.gamma, (void*)wt.norm1_w, (void*)wt.conv1_w,
                (void*)wt.conv2_w, (void*)wt.sca_w, (void*)wt.conv3_w,
                (void*)wt.norm2_w, (void*)wt.conv4_w, (void*)wt.conv5_w);
        return;
    }
    const float * beta = to_f32(wt.beta, dq1);
    const float * gamma = to_f32(wt.gamma, dq2);

    // Spatial mixing
    std::vector<float> t1(c * hw), t2(c2 * hw), t3(c * hw);
    layernorm2d(x, c, h, w, to_f32(wt.norm1_w, dq1), to_f32(wt.norm1_b, dq2), t1.data());
    conv2d(t1.data(), c, h, w, to_f32(wt.conv1_w, dq1), to_f32(wt.conv1_b, dq2),
           c2, 1, 1, 0, 1, 1, t2.data());
    // DW conv outputs c2 channels — need a c2-sized buffer
    std::vector<float> dw_out(c2 * hw);
    conv2d(t2.data(), c2, h, w, to_f32(wt.conv2_w, dq1), to_f32(wt.conv2_b, dq2),
           c2, 3, 3, 1, 1, c2, dw_out.data());
    simple_gate(dw_out.data(), c2, hw, t3.data());
    // SCA
    std::vector<float> pool(c, 0.0f);
    for (int ch = 0; ch < c; ch++) {
        for (int i = 0; i < hw; i++) pool[ch] += t3[ch * hw + i];
        pool[ch] /= hw;
    }
    std::vector<float> sca(c);
    for (int o = 0; o < c; o++) {
        float sum = to_f32(wt.sca_b, dq2)[o];
        for (int i = 0; i < c; i++) sum += to_f32(wt.sca_w, dq1)[o * c + i] * pool[i];
        sca[o] = sum;
    }
    for (int ch = 0; ch < c; ch++)
        for (int i = 0; i < hw; i++) t3[ch * hw + i] *= sca[ch];
    conv2d(t3.data(), c, h, w, to_f32(wt.conv3_w, dq1), to_f32(wt.conv3_b, dq2),
           c, 1, 1, 0, 1, 1, t1.data());
    for (int ch = 0; ch < c; ch++)
        for (int i = 0; i < hw; i++)
            x[ch * hw + i] += t1[ch * hw + i] * beta[ch];

    // Channel mixing
    layernorm2d(x, c, h, w, to_f32(wt.norm2_w, dq1), to_f32(wt.norm2_b, dq2), t1.data());
    conv2d(t1.data(), c, h, w, to_f32(wt.conv4_w, dq1), to_f32(wt.conv4_b, dq2),
           c2, 1, 1, 0, 1, 1, t2.data());
    simple_gate(t2.data(), c2, hw, t3.data());
    conv2d(t3.data(), c, h, w, to_f32(wt.conv5_w, dq1), to_f32(wt.conv5_b, dq2),
           c, 1, 1, 0, 1, 1, t1.data());
    for (int ch = 0; ch < c; ch++)
        for (int i = 0; i < hw; i++)
            x[ch * hw + i] += t1[ch * hw + i] * gamma[ch];
}

// ── ICB forward ──

struct icb_wt {
    ggml_tensor * beta, * gamma;
    ggml_tensor * fc_w, * fc_b;  // Linear(256→C)
    nafblock_wt block;
};

static void icb_forward(float * x, int c, int h, int w,
                         const float * text_embd, int emb_dim,
                         const icb_wt & wt,
                         std::vector<float> & dq1, std::vector<float> & dq2) {
    int hw = h * w;
    const float * beta = to_f32(wt.beta, dq1);
    const float * gamma = to_f32(wt.gamma, dq2);

    // Gating from text embedding
    std::vector<float> gate(c);
    const float * fc_w = to_f32(wt.fc_w, dq1);
    const float * fc_b = to_f32(wt.fc_b, dq2);
    for (int o = 0; o < c; o++) {
        float sum = fc_b[o];
        for (int i = 0; i < emb_dim; i++) sum += fc_w[o * emb_dim + i] * text_embd[i];
        gate[o] = 1.0f / (1.0f + expf(-sum)); // sigmoid
    }

    // y = x * gamma + beta, then y *= gate
    std::vector<float> y(c * hw);
    for (int ch = 0; ch < c; ch++)
        for (int i = 0; i < hw; i++)
            y[ch * hw + i] = (x[ch * hw + i] * gamma[ch] + beta[ch]) * gate[ch];

    // NAFBlock refinement
    nafblock_forward(y.data(), c, h, w, wt.block, dq1, dq2);

    // Residual
    for (int i = 0; i < c * hw; i++) x[i] += y[i];
}

// ── Model context ──

struct instructir_context {
    ggml_backend_t backend = nullptr;
    ggml_context * gguf_ctx;
    ggml_backend_buffer_t gguf_buf;

    int n_tasks, emb_dim;
    bool bench;
    ggml_tensor * task_embeddings; // [n_tasks, 256]

    ggml_tensor * intro_w, * intro_b;
    ggml_tensor * ending_w, * ending_b;

    // 4 encoder levels, each with N NAFBlocks + 1 ICB + downsample
    struct enc_level {
        std::vector<nafblock_wt> blocks;
        icb_wt cond;
        ggml_tensor * down_w, * down_b;
    } enc[4];
    int enc_n_blocks[4];

    // 4 middle NAFBlocks
    std::vector<nafblock_wt> middle;

    // 4 decoder levels: upsample + N NAFBlocks + ICB
    struct dec_level {
        ggml_tensor * up_w, * up_b; // Conv1x1(C→4C) for PixelShuffle
        std::vector<nafblock_wt> blocks;
        icb_wt cond;
    } dec[4];
    int dec_n_blocks[4];
};

static void load_nafblock(core_gguf::WeightLoad & wl, const char * pfx, nafblock_wt & b) {
    auto g = [&](const char * s) {
        char buf[256]; snprintf(buf, sizeof(buf), "%s.%s", pfx, s);
        return core_gguf::try_get(wl.tensors, buf);
    };
    b.beta = g("beta"); b.gamma = g("gamma");
    b.norm1_w = g("norm1.weight"); b.norm1_b = g("norm1.bias");
    b.conv1_w = g("conv1.weight"); b.conv1_b = g("conv1.bias");
    b.conv2_w = g("conv2.weight"); b.conv2_b = g("conv2.bias");
    b.sca_w = g("sca.1.weight"); b.sca_b = g("sca.1.bias");
    b.conv3_w = g("conv3.weight"); b.conv3_b = g("conv3.bias");
    b.norm2_w = g("norm2.weight"); b.norm2_b = g("norm2.bias");
    b.conv4_w = g("conv4.weight"); b.conv4_b = g("conv4.bias");
    b.conv5_w = g("conv5.weight"); b.conv5_b = g("conv5.bias");
}

static void load_icb(core_gguf::WeightLoad & wl, const char * pfx, icb_wt & ic) {
    auto g = [&](const char * s) {
        char buf[256]; snprintf(buf, sizeof(buf), "%s.%s", pfx, s);
        return core_gguf::try_get(wl.tensors, buf);
    };
    ic.beta = g("beta"); ic.gamma = g("gamma");
    ic.fc_w = g("fc.weight"); ic.fc_b = g("fc.bias");
    char blk[256]; snprintf(blk, sizeof(blk), "%s.block", pfx);
    load_nafblock(wl, blk, ic.block);
}

instructir_context * instructir_init(const char * model_path, int n_threads) {
    (void)n_threads;
    if (!model_path) return nullptr;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) return nullptr;
    int n_tasks = (int)core_gguf::kv_u32(meta, "instructir.n_tasks", 7);
    int emb_dim = (int)core_gguf::kv_u32(meta, "instructir.emb_dim", 256);
    core_gguf::free_metadata(meta);

    bool force_cpu = (getenv("INSTRUCTIR_FORCE_CPU") && atoi(getenv("INSTRUCTIR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    if (!backend) return nullptr;
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, backend, "instructir", wl)) {
        ggml_backend_free(backend); return nullptr;
    }

    auto * ctx = new instructir_context;
    ctx->backend = backend;
    ctx->gguf_ctx = wl.ctx; ctx->gguf_buf = wl.buf;
    ctx->n_tasks = n_tasks; ctx->emb_dim = emb_dim;
    ctx->task_embeddings = core_gguf::require(wl.tensors, "task_embeddings", "instructir");
    ctx->intro_w = core_gguf::try_get(wl.tensors, "intro.weight");
    ctx->intro_b = core_gguf::try_get(wl.tensors, "intro.bias");
    ctx->ending_w = core_gguf::try_get(wl.tensors, "ending.weight");
    ctx->ending_b = core_gguf::try_get(wl.tensors, "ending.bias");

    int enc_blks[] = {2, 2, 4, 8};
    int dec_blks[] = {2, 2, 2, 2};
    for (int lvl = 0; lvl < 4; lvl++) {
        ctx->enc_n_blocks[lvl] = enc_blks[lvl];
        ctx->enc[lvl].blocks.resize(enc_blks[lvl]);
        for (int i = 0; i < enc_blks[lvl]; i++) {
            char pfx[64]; snprintf(pfx, sizeof(pfx), "encoders.%d.%d", lvl, i);
            load_nafblock(wl, pfx, ctx->enc[lvl].blocks[i]);
        }
        char cpfx[64]; snprintf(cpfx, sizeof(cpfx), "enc_cond.%d", lvl);
        load_icb(wl, cpfx, ctx->enc[lvl].cond);
        char dw[64], db[64];
        snprintf(dw, sizeof(dw), "downs.%d.weight", lvl);
        snprintf(db, sizeof(db), "downs.%d.bias", lvl);
        ctx->enc[lvl].down_w = core_gguf::try_get(wl.tensors, dw);
        ctx->enc[lvl].down_b = core_gguf::try_get(wl.tensors, db);

        ctx->dec_n_blocks[lvl] = dec_blks[lvl];
        ctx->dec[lvl].blocks.resize(dec_blks[lvl]);
        char uw[64], ub[64];
        snprintf(uw, sizeof(uw), "ups.%d.0.weight", lvl);
        snprintf(ub, sizeof(ub), "ups.%d.0.bias", lvl);
        ctx->dec[lvl].up_w = core_gguf::try_get(wl.tensors, uw);
        ctx->dec[lvl].up_b = core_gguf::try_get(wl.tensors, ub);
        for (int i = 0; i < dec_blks[lvl]; i++) {
            char pfx[64]; snprintf(pfx, sizeof(pfx), "decoders.%d.%d", lvl, i);
            load_nafblock(wl, pfx, ctx->dec[lvl].blocks[i]);
        }
        char dcpfx[64]; snprintf(dcpfx, sizeof(dcpfx), "dec_cond.%d", lvl);
        load_icb(wl, dcpfx, ctx->dec[lvl].cond);
    }

    ctx->middle.resize(4);
    for (int i = 0; i < 4; i++) {
        char pfx[64]; snprintf(pfx, sizeof(pfx), "middle_blks.%d", i);
        load_nafblock(wl, pfx, ctx->middle[i]);
    }

    ctx->bench = (std::getenv("CRISPEMBED_INSTRUCTIR_BENCH") != nullptr);
    return ctx;
}

void instructir_free(instructir_context * ctx) {
    if (!ctx) return;
    core_gguf::WeightLoad wl;
    wl.ctx = ctx->gguf_ctx; wl.buf = ctx->gguf_buf;
    core_gguf::free_weights(wl);
    if (ctx->backend) ggml_backend_free(ctx->backend);
    delete ctx;
}

int instructir_get_n_tasks(const instructir_context * ctx) {
    return ctx ? ctx->n_tasks : 0;
}

int instructir_process_float(instructir_context * ctx, int task,
                             const float * input_chw, int width, int height,
                             float * output_chw) {
    if (!ctx || !input_chw || !output_chw || task < 0 || task >= ctx->n_tasks)
        return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    int H = height, W = width;
    std::vector<float> dq1, dq2;

    // Get task embedding [256]
    const float * all_emb = to_f32(ctx->task_embeddings, dq1);
    const float * text_embd = all_emb + task * ctx->emb_dim;

    // Intro conv
    int ch = 32;
    std::vector<float> x(ch * H * W);
    conv2d(input_chw, 3, H, W, to_f32(ctx->intro_w, dq1), to_f32(ctx->intro_b, dq2),
           ch, 3, 3, 1, 1, 1, x.data());

    // Encoder
    std::vector<std::vector<float>> skips;
    int cur_h = H, cur_w = W;
    int channels[] = {32, 64, 128, 256};
    for (int lvl = 0; lvl < 4; lvl++) {
        auto t_enc = std::chrono::steady_clock::now();
        for (int i = 0; i < ctx->enc_n_blocks[lvl]; i++) {
            nafblock_forward(x.data(), ch, cur_h, cur_w, ctx->enc[lvl].blocks[i], dq1, dq2);
        }
        icb_forward(x.data(), ch, cur_h, cur_w, text_embd, ctx->emb_dim,
                    ctx->enc[lvl].cond, dq1, dq2);
        skips.push_back(std::vector<float>(x.begin(), x.end()));
        // Downsample: Conv2d(C→2C, k=2, s=2)
        int next_ch = ch * 2;
        int nh = cur_h / 2, nw = cur_w / 2;
        std::vector<float> ds(next_ch * nh * nw);
        conv2d(x.data(), ch, cur_h, cur_w,
               to_f32(ctx->enc[lvl].down_w, dq1), to_f32(ctx->enc[lvl].down_b, dq2),
               next_ch, 2, 2, 0, 2, 1, ds.data());
        x = std::move(ds);
        ch = next_ch; cur_h = nh; cur_w = nw;
        if (bench) {
            auto t_enc_end = std::chrono::steady_clock::now();
            fprintf(stderr, "[instructir-bench] enc level %d: %.1f ms\n",
                    lvl, ms_f(t_enc_end - t_enc).count());
        }
    }

    // Middle
    auto t_mid = std::chrono::steady_clock::now();
    for (int i = 0; i < 4; i++)
        nafblock_forward(x.data(), ch, cur_h, cur_w, ctx->middle[i], dq1, dq2);
    if (bench) {
        auto t_mid_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[instructir-bench] middle: %.1f ms\n",
                ms_f(t_mid_end - t_mid).count());
    }

    // Decoder
    for (int lvl = 0; lvl < 4; lvl++) {
        auto t_dec = std::chrono::steady_clock::now();
        // Upsample: Conv1x1(C→4C') + PixelShuffle(2)
        int next_ch = ch / 2;
        int up_ch = next_ch * 4; // after conv1x1, before shuffle
        std::vector<float> up(up_ch * cur_h * cur_w);
        conv2d(x.data(), ch, cur_h, cur_w,
               to_f32(ctx->dec[lvl].up_w, dq1), to_f32(ctx->dec[lvl].up_b, dq2),
               up_ch, 1, 1, 0, 1, 1, up.data());
        int nh = cur_h * 2, nw = cur_w * 2;
        x.resize(next_ch * nh * nw);
        pixel_shuffle(up.data(), up_ch, cur_h, cur_w, 2, x.data());
        ch = next_ch; cur_h = nh; cur_w = nw;

        // Skip connection
        auto & sk = skips[3 - lvl];
        for (int i = 0; i < ch * cur_h * cur_w; i++) x[i] += sk[i];

        for (int i = 0; i < ctx->dec_n_blocks[lvl]; i++)
            nafblock_forward(x.data(), ch, cur_h, cur_w, ctx->dec[lvl].blocks[i], dq1, dq2);
        icb_forward(x.data(), ch, cur_h, cur_w, text_embd, ctx->emb_dim,
                    ctx->dec[lvl].cond, dq1, dq2);
        if (bench) {
            auto t_dec_end = std::chrono::steady_clock::now();
            fprintf(stderr, "[instructir-bench] dec level %d: %.1f ms\n",
                    lvl, ms_f(t_dec_end - t_dec).count());
        }
    }

    // Ending conv + global residual
    conv2d(x.data(), ch, cur_h, cur_w,
           to_f32(ctx->ending_w, dq1), to_f32(ctx->ending_b, dq2),
           3, 3, 3, 1, 1, 1, output_chw);
    for (int i = 0; i < 3 * H * W; i++) output_chw[i] += input_chw[i];

    if (bench) {
        auto t_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[instructir-bench] total: %.1f ms\n",
                ms_f(t_end - t_total).count());
    }
    return 0;
}

static int instructir_process_tile(instructir_context * ctx, int task,
                                   const uint8_t * input, int width, int height,
                                   uint8_t * output) {
    int hw = width * height;
    std::vector<float> in_chw(3 * hw);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                in_chw[c * hw + y * width + x] = (float)input[(y * width + x) * 3 + c] / 255.0f;
    std::vector<float> out_chw(3 * hw);
    int ret = instructir_process_float(ctx, task, in_chw.data(), width, height, out_chw.data());
    if (ret != 0) return ret;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++) {
                float v = out_chw[c * hw + y * width + x] * 255.0f;
                output[(y * width + x) * 3 + c] = (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
    return 0;
}

static void iir_blend_win(int ts, int ov, std::vector<float> & w) {
    w.resize(ts * ts);
    for (int y = 0; y < ts; y++) {
        float wy = 1.0f;
        if (y < ov) wy = 0.5f - 0.5f * cosf((float)M_PI * y / ov);
        else if (y >= ts - ov) wy = 0.5f - 0.5f * cosf((float)M_PI * (ts - 1 - y) / ov);
        for (int x = 0; x < ts; x++) {
            float wx = 1.0f;
            if (x < ov) wx = 0.5f - 0.5f * cosf((float)M_PI * x / ov);
            else if (x >= ts - ov) wx = 0.5f - 0.5f * cosf((float)M_PI * (ts - 1 - x) / ov);
            w[y * ts + x] = wy * wx;
        }
    }
}

int instructir_process(instructir_context * ctx, int task,
                       const uint8_t * input, int width, int height,
                       uint8_t * output) {
    if (!ctx || !input || !output) return -1;
    int tile_size = 256;
    int tile_overlap = 32;
    const char * ts_env = std::getenv("CRISPEMBED_INSTRUCTIR_TILE");
    if (ts_env) tile_size = std::max(64, (atoi(ts_env) / 8) * 8);
    tile_overlap = std::min(tile_overlap, tile_size / 4);

    if (width <= tile_size && height <= tile_size)
        return instructir_process_tile(ctx, task, input, width, height, output);

    std::vector<float> accum(3 * height * width, 0.0f);
    std::vector<float> wmap(height * width, 0.0f);
    std::vector<float> bwin;
    iir_blend_win(tile_size, tile_overlap, bwin);
    int step = tile_size - tile_overlap;
    int ntx = std::max(1, (width + step - 1) / step);
    int nty = std::max(1, (height + step - 1) / step);
    fprintf(stderr, "instructir: %dx%d, tiles=%dx%d (size=%d)\n", width, height, ntx, nty, tile_size);
    for (int ty = 0; ty < nty; ty++)
        for (int tx = 0; tx < ntx; tx++) {
            int x0 = std::min(tx*step, std::max(0, width-tile_size));
            int y0 = std::min(ty*step, std::max(0, height-tile_size));
            int tw = std::min(tile_size, width-x0), th = std::min(tile_size, height-y0);
            std::vector<uint8_t> ti(tw*th*3), to(tw*th*3);
            for (int y=0;y<th;y++) memcpy(ti.data()+y*tw*3, input+((y0+y)*width+x0)*3, tw*3);
            if (instructir_process_tile(ctx, task, ti.data(), tw, th, to.data()) != 0) return -1;
            for (int y=0;y<th;y++)
                for (int x=0;x<tw;x++) {
                    float w = (tw==tile_size&&th==tile_size) ? bwin[y*tile_size+x] : 1.0f;
                    if (tw!=tile_size||th!=tile_size) {
                        if (x0>0&&x<tile_overlap) w*=0.5f-0.5f*cosf((float)M_PI*x/tile_overlap);
                        if (y0>0&&y<tile_overlap) w*=0.5f-0.5f*cosf((float)M_PI*y/tile_overlap);
                    }
                    int dy=y0+y, dx=x0+x;
                    for (int c=0;c<3;c++) accum[c*height*width+dy*width+dx]+=to[(y*tw+x)*3+c]*w;
                    wmap[dy*width+dx]+=w;
                }
        }
    for (int y=0;y<height;y++)
        for (int x=0;x<width;x++) {
            float wt=wmap[y*width+x]; if(wt<=0)wt=1;
            for (int c=0;c<3;c++) {
                float v=accum[c*height*width+y*width+x]/wt;
                output[(y*width+x)*3+c]=(uint8_t)std::max(0.f,std::min(255.f,v+0.5f));
            }
        }
    return 0;
}
