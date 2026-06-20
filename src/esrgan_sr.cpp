// esrgan_sr.cpp — Real-ESRGAN SRVGGNetCompact (CPU-scalar).
//
// Forward pass: x → [Conv3x3 + PReLU] × 17 → Conv3x3 → PixelShuffle(4)
//               + nearest-upsample(input) → output
//
// Body layout: body.0=Conv(3→64), body.1=PReLU, body.2=Conv(64→64),
// body.3=PReLU, ..., body.32=Conv(64→64), body.33=PReLU, body.34=Conv(64→48)

#include "esrgan_sr.h"
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

static const float * to_f32(const ggml_tensor * t, std::vector<float> & buf) {
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

static void conv2d(const float * input, int ic, int ih, int iw,
                   const float * weight, const float * bias,
                   int oc, float * output) {
    // 3×3, pad=1, stride=1, groups=1
    for (int o = 0; o < oc; o++) {
        float b = bias[o];
        for (int oy = 0; oy < ih; oy++) {
            for (int ox = 0; ox < iw; ox++) {
                float sum = b;
                for (int c = 0; c < ic; c++) {
                    for (int ky = 0; ky < 3; ky++) {
                        int iy = oy + ky - 1;
                        if (iy < 0 || iy >= ih) continue;
                        for (int kx = 0; kx < 3; kx++) {
                            int ix = ox + kx - 1;
                            if (ix < 0 || ix >= iw) continue;
                            sum += input[c * ih * iw + iy * iw + ix]
                                 * weight[o * ic * 9 + c * 9 + ky * 3 + kx];
                        }
                    }
                }
                output[o * ih * iw + oy * iw + ox] = sum;
            }
        }
    }
}

// PReLU: y = max(0,x) + slope * min(0,x). Per-channel slopes.
static void prelu(float * data, int c, int hw, const float * slopes) {
    for (int ch = 0; ch < c; ch++) {
        float s = slopes[ch];
        for (int i = 0; i < hw; i++) {
            float & v = data[ch * hw + i];
            if (v < 0) v *= s;
        }
    }
}

static void pixel_shuffle(const float * input, int c_in, int h, int w,
                           int r, float * output) {
    int c_out = c_in / (r * r), oh = h * r, ow = w * r;
    for (int c = 0; c < c_out; c++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                int ic = c * r * r + (y % r) * r + (x % r);
                output[c * oh * ow + y * ow + x] =
                    input[ic * h * w + (y / r) * w + (x / r)];
            }
}

static void nearest_upsample(const float * input, int c, int h, int w,
                              int scale, float * output) {
    int oh = h * scale, ow = w * scale;
    for (int ch = 0; ch < c; ch++)
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                output[ch * oh * ow + y * ow + x] =
                    input[ch * h * w + (y / scale) * w + (x / scale)];
}

// ── Context ────────────────────────────────────────────────────────

struct conv_layer { ggml_tensor * w; ggml_tensor * b; };
struct prelu_layer { ggml_tensor * slope; };

struct esrgan_context {
    ggml_context * gguf_ctx;
    ggml_backend_buffer_t gguf_buf;
    int scale, num_feat, num_conv;
    bool bench;
    // body.0=conv, body.1=prelu, body.2=conv, ..., body.34=conv
    std::vector<conv_layer> convs;   // 18 convolutions
    std::vector<prelu_layer> prelus; // 17 PReLU layers
};

esrgan_context * esrgan_init(const char * model_path, int n_threads) {
    (void)n_threads;
    if (!model_path) return nullptr;

    gguf_context * meta = core_gguf::open_metadata(model_path);
    if (!meta) return nullptr;
    int scale    = (int)core_gguf::kv_u32(meta, "esrgan.scale", 4);
    int num_feat = (int)core_gguf::kv_u32(meta, "esrgan.num_feat", 64);
    int num_conv = (int)core_gguf::kv_u32(meta, "esrgan.num_conv", 16);
    core_gguf::free_metadata(meta);

    bool force_cpu = (getenv("ESRGAN_SR_FORCE_CPU") && atoi(getenv("ESRGAN_SR_FORCE_CPU")));
    ggml_backend_t backend = force_cpu ? ggml_backend_cpu_init() : ggml_backend_init_best();
    if (!backend) backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad wl;
    if (!core_gguf::load_weights(model_path, backend, "esrgan", wl)) {
        ggml_backend_free(backend); return nullptr;
    }
    ggml_backend_free(backend);

    auto * ctx = new esrgan_context;
    ctx->gguf_ctx = wl.ctx;
    ctx->gguf_buf = wl.buf;
    ctx->scale = scale;
    ctx->num_feat = num_feat;
    ctx->num_conv = num_conv;

    // 18 convs (entry + 16 body + output), 17 PReLUs (entry + 16 body)
    int n_total = 2 * num_conv + 2 + 1; // body indices 0..34
    for (int i = 0; i < n_total; i++) {
        char wn[64], bn[64];
        snprintf(wn, sizeof(wn), "body.%d.weight", i);
        snprintf(bn, sizeof(bn), "body.%d.bias", i);
        ggml_tensor * w = core_gguf::try_get(wl.tensors, wn);
        ggml_tensor * b = core_gguf::try_get(wl.tensors, bn);
        if (b) {
            // Conv layer (has bias)
            ctx->convs.push_back({w, b});
        } else if (w) {
            // PReLU layer (weight = slopes, no bias)
            ctx->prelus.push_back({w});
        }
    }

    ctx->bench = (std::getenv("CRISPEMBED_ESRGAN_BENCH") != nullptr);
    return ctx;
}

void esrgan_free(esrgan_context * ctx) {
    if (!ctx) return;
    core_gguf::WeightLoad wl;
    wl.ctx = ctx->gguf_ctx;
    wl.buf = ctx->gguf_buf;
    core_gguf::free_weights(wl);
    delete ctx;
}

int esrgan_get_scale(const esrgan_context * ctx) {
    return ctx ? ctx->scale : 0;
}

int esrgan_process_float(esrgan_context * ctx,
                         const float * input_chw, int width, int height,
                         float * output_chw) {
    if (!ctx || !input_chw || !output_chw) return -1;

    const bool bench = ctx->bench;
    using ms_f = std::chrono::duration<double, std::milli>;
    auto t_total = std::chrono::steady_clock::now();

    const int H = height, W = width, hw = H * W;
    std::vector<float> dq1, dq2;

    // Two ping-pong buffers
    std::vector<float> buf_a(input_chw, input_chw + 3 * hw);
    std::vector<float> buf_b;

    int ic = 3;
    int conv_idx = 0, prelu_idx = 0;

    // Process: conv → prelu pairs, then final conv (no prelu)
    int n_convs = (int)ctx->convs.size();
    for (int ci = 0; ci < n_convs; ci++) {
        auto t_conv = std::chrono::steady_clock::now();
        int oc = (ci == n_convs - 1) ? 3 * ctx->scale * ctx->scale : ctx->num_feat;
        buf_b.resize(oc * hw);
        conv2d(buf_a.data(), ic, H, W,
               to_f32(ctx->convs[ci].w, dq1),
               to_f32(ctx->convs[ci].b, dq2),
               oc, buf_b.data());

        // PReLU after every conv except the last
        if (ci < n_convs - 1 && prelu_idx < (int)ctx->prelus.size()) {
            prelu(buf_b.data(), oc, hw,
                  to_f32(ctx->prelus[prelu_idx].slope, dq1));
            prelu_idx++;
        }

        std::swap(buf_a, buf_b);
        ic = oc;
        if (bench) {
            auto t_conv_end = std::chrono::steady_clock::now();
            fprintf(stderr, "[esrgan-bench] conv %d: %.1f ms\n", ci,
                    ms_f(t_conv_end - t_conv).count());
        }
    }

    // PixelShuffle
    auto t_ps = std::chrono::steady_clock::now();
    int out_h = H * ctx->scale, out_w = W * ctx->scale;
    int out_hw = out_h * out_w;
    std::vector<float> sr(3 * out_hw);
    pixel_shuffle(buf_a.data(), ic, H, W, ctx->scale, sr.data());

    // Global residual: nearest-upsample input + sr
    nearest_upsample(input_chw, 3, H, W, ctx->scale, output_chw);
    for (int i = 0; i < 3 * out_hw; i++)
        output_chw[i] += sr[i];
    if (bench) {
        auto t_ps_end = std::chrono::steady_clock::now();
        fprintf(stderr, "[esrgan-bench] pixelshuffle+residual: %.1f ms\n",
                ms_f(t_ps_end - t_ps).count());
        fprintf(stderr, "[esrgan-bench] total: %.1f ms\n",
                ms_f(t_ps_end - t_total).count());
    }

    return 0;
}

// Build a 2D raised-cosine (Hann) blending window for tile overlap regions.
static void build_blend_window(int tile_size, int overlap, std::vector<float> & win) {
    win.resize(tile_size * tile_size);
    for (int y = 0; y < tile_size; y++) {
        float wy = 1.0f;
        if (y < overlap)
            wy = 0.5f - 0.5f * cosf((float)M_PI * y / overlap);
        else if (y >= tile_size - overlap)
            wy = 0.5f - 0.5f * cosf((float)M_PI * (tile_size - 1 - y) / overlap);
        for (int x = 0; x < tile_size; x++) {
            float wx = 1.0f;
            if (x < overlap)
                wx = 0.5f - 0.5f * cosf((float)M_PI * x / overlap);
            else if (x >= tile_size - overlap)
                wx = 0.5f - 0.5f * cosf((float)M_PI * (tile_size - 1 - x) / overlap);
            win[y * tile_size + x] = wy * wx;
        }
    }
}

int esrgan_process(esrgan_context * ctx,
                   const uint8_t * input, int width, int height,
                   uint8_t * output) {
    if (!ctx || !input || !output) return -1;

    int r = ctx->scale;
    // Tile parameters — 128 is conservative for 8GB RAM with 64-channel model
    int tile_size = 128;
    int tile_overlap = 16;
    const char * ts_env = std::getenv("CRISPEMBED_ESRGAN_TILE");
    if (ts_env) tile_size = std::max(32, atoi(ts_env));
    tile_overlap = std::min(tile_overlap, tile_size / 4);

    // Convert full input to CHW float [0,1]
    int hw = width * height;
    std::vector<float> full_input(3 * hw);
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            for (int c = 0; c < 3; c++)
                full_input[c * hw + y * width + x] =
                    (float)input[(y * width + x) * 3 + c] / 255.0f;

    int ow = width * r, oh = height * r;
    int out_tile = tile_size * r;
    int out_overlap = tile_overlap * r;

    // Small image: process in one shot (no tiling overhead)
    if (width <= tile_size && height <= tile_size) {
        std::vector<float> out_chw(3 * oh * ow);
        int ret = esrgan_process_float(ctx, full_input.data(), width, height, out_chw.data());
        if (ret != 0) return ret;
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++)
                for (int c = 0; c < 3; c++) {
                    float v = out_chw[c * oh * ow + y * ow + x] * 255.0f;
                    output[(y * ow + x) * 3 + c] =
                        (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
                }
        return 0;
    }

    // Tiled processing with Hann-window blending
    std::vector<float> accum(3 * oh * ow, 0.0f);
    std::vector<float> weight_map(oh * ow, 0.0f);

    std::vector<float> blend_win;
    build_blend_window(out_tile, out_overlap, blend_win);

    int step = tile_size - tile_overlap;
    int n_tiles_x = std::max(1, (width + step - 1) / step);
    int n_tiles_y = std::max(1, (height + step - 1) / step);

    fprintf(stderr, "esrgan: %dx%d -> %dx%d (%dx), tiles=%dx%d (size=%d, overlap=%d)\n",
            width, height, ow, oh, r, n_tiles_x, n_tiles_y, tile_size, tile_overlap);

    for (int ty = 0; ty < n_tiles_y; ty++) {
        for (int tx = 0; tx < n_tiles_x; tx++) {
            int x0 = std::min(tx * step, std::max(0, width - tile_size));
            int y0 = std::min(ty * step, std::max(0, height - tile_size));
            int tw = std::min(tile_size, width - x0);
            int th = std::min(tile_size, height - y0);

            // Extract tile
            std::vector<float> tile_in(3 * th * tw);
            for (int c = 0; c < 3; c++)
                for (int y = 0; y < th; y++)
                    for (int x = 0; x < tw; x++)
                        tile_in[c * th * tw + y * tw + x] =
                            full_input[c * height * width + (y0 + y) * width + (x0 + x)];

            // SR forward pass on tile
            int otw = tw * r, oth = th * r;
            std::vector<float> tile_out(3 * oth * otw);
            int ret = esrgan_process_float(ctx, tile_in.data(), tw, th, tile_out.data());
            if (ret != 0) return ret;

            // Blend into accumulator
            int ox0 = x0 * r, oy0 = y0 * r;
            for (int y = 0; y < oth; y++) {
                for (int x = 0; x < otw; x++) {
                    float w = 1.0f;
                    if (tw == tile_size && th == tile_size)
                        w = blend_win[y * out_tile + x];
                    else {
                        if (x0 > 0 && x < out_overlap)
                            w *= 0.5f - 0.5f * cosf((float)M_PI * x / out_overlap);
                        if (y0 > 0 && y < out_overlap)
                            w *= 0.5f - 0.5f * cosf((float)M_PI * y / out_overlap);
                    }
                    int dy = oy0 + y, dx = ox0 + x;
                    if (dy >= oh || dx >= ow) continue;
                    for (int c = 0; c < 3; c++)
                        accum[c * oh * ow + dy * ow + dx] +=
                            tile_out[c * oth * otw + y * otw + x] * w;
                    weight_map[dy * ow + dx] += w;
                }
            }
        }
    }

    // Normalize by weights and convert to uint8
    for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++) {
            float wt = weight_map[y * ow + x];
            if (wt <= 0.0f) wt = 1.0f;
            for (int c = 0; c < 3; c++) {
                float v = accum[c * oh * ow + y * ow + x] / wt * 255.0f;
                output[(y * ow + x) * 3 + c] =
                    (uint8_t)std::max(0.0f, std::min(255.0f, v + 0.5f));
            }
        }
    return 0;
}
