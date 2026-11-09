/*
Source-compatible arithmetic/graph adapted from llama.cpp6c01 (MIT).
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include "lighton4b_physical_f16_vision.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#if defined(CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI) && !defined(__EMSCRIPTEN__)
#include "qwen35_math.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "llamafile/sgemm.h"
#include <atomic>
#include <deque>
#include <memory>
#include <stdexcept>
extern "C" void ggml_vec_dot_f16(int, float *, size_t, ggml_fp16_t *, size_t, ggml_fp16_t *, size_t, int);
#endif
namespace qwen35_ocr {
bool lighton4b_physical_f16_vision_available() {
    return lighton4b_physical_f16_attention_available();
}
#if defined(CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI) && !defined(__EMSCRIPTEN__)
namespace {
void require(bool ok) {
    if (!ok) throw std::runtime_error("unsupported physical F16 vision operand");
}
bool finite(const std::vector<float> & x) {
    return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
}
struct product_state {
    ggml_threadpool * pool;
    std::atomic_bool accepted{ true };
    explicit product_state(ggml_threadpool * p) : pool(p) {}
};
void product(ggml_tensor * dst, const ggml_tensor *, const ggml_tensor * a, const ggml_tensor * b, int ith, int nth,
             void * user) {
    auto & s = *static_cast<product_state *>(user);
    if (nth != 4 || a->type != GGML_TYPE_F16 || b->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 ||
        a->ne[0] != b->ne[0] || a->ne[2] != 1 || b->ne[2] != 1 || a->ne[3] != 1 || b->ne[3] != 1 ||
        !ggml_is_contiguous(a) || !ggml_is_contiguous(b) || !ggml_is_contiguous(dst)) {
        s.accepted = false;
        return;
    }
    const int m = int(a->ne[1]), n = int(b->ne[1]), k = int(a->ne[0]);
    auto * left = static_cast<ggml_fp16_t *>(a->data);
    auto * right = static_cast<ggml_fp16_t *>(b->data);
    auto * out = static_cast<float *>(dst->data);
    if (n == 1) {
        for (int row = ith; row < m; row += nth)
            ggml_vec_dot_f16(k, out + row, 0, left + size_t(row) * k, 0, right, 0, 1);
        return;
    }
    ggml_compute_params params{};
    params.ith = ith;
    params.nth = nth;
    params.threadpool = s.pool;
    if (!llamafile_sgemm(&params, m, n, k, left, k, right, k, out, m, GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32))
        s.accepted = false;
}
struct graph {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx{ ggml_init({ 512ull << 20, nullptr, false }), ggml_free };
    std::deque<product_state> states;
    ggml_backend_t backend;
    ggml_threadpool * pool;
    graph(ggml_backend_t b, ggml_threadpool * p) : backend(b), pool(p) { require(bool(ctx)); }
    ggml_context * c() { return ctx.get(); }
    ggml_tensor * f32(int width, int rows, const std::vector<float> & data) {
        require(data.size() == size_t(width) * rows && finite(data));
        auto * t = ggml_new_tensor_2d(c(), GGML_TYPE_F32, width, rows);
        std::copy(data.begin(), data.end(), static_cast<float *>(t->data));
        return t;
    }
    std::vector<float> values(ggml_tensor * t, int width, int rows) {
        require(t && t->type == GGML_TYPE_F32 && t->ne[0] == width && t->ne[1] == rows && t->ne[2] == 1 &&
                t->ne[3] == 1 && ggml_is_contiguous(t));
        std::vector<float> x(size_t(width) * rows);
        require(tensor_floats(t, x.data(), 4) && finite(x));
        return x;
    }
    ggml_tensor * parameter(ggml_tensor * t, int width, int rows = 1) {
        return f32(width, rows, values(t, width, rows));
    }
    ggml_tensor * half(ggml_tensor * t, int width, int rows) {
        auto x = values(t, width, rows);
        auto * y = ggml_new_tensor_2d(c(), GGML_TYPE_F16, width, rows);
        auto * p = static_cast<ggml_fp16_t *>(y->data);
        for (size_t i = 0; i < x.size(); ++i) {
            p[i] = ggml_fp32_to_fp16(x[i]);
            require(std::isfinite(ggml_fp16_to_fp32(p[i])));
        }
        return y;
    }
    ggml_tensor * mm(ggml_tensor * a, ggml_tensor * b) {
        require(a && b && a->type == GGML_TYPE_F16 && a->ne[0] == b->ne[0]);
        if (b->type == GGML_TYPE_F32) b = ggml_cast(c(), b, GGML_TYPE_F16);
        require(b->type == GGML_TYPE_F16);
        auto * shape = ggml_new_tensor_2d(c(), GGML_TYPE_F32, a->ne[1], b->ne[1]);
        states.emplace_back(pool);
        return ggml_map_custom3(c(), shape, a, b, product, 4, &states.back());
    }
    ggml_tensor * linear(ggml_tensor * w, ggml_tensor * b, ggml_tensor * x, int width, int rows) {
        auto * y = mm(half(w, width, rows), x);
        return ggml_add(c(), y, parameter(b, rows));
    }
    ggml_tensor * norm(ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
        return ggml_add(c(), ggml_mul(c(), ggml_norm(c(), x, 1e-6f), parameter(w, 1024)), parameter(b, 1024));
    }
    void compute(ggml_tensor * out) {
        auto * gf = ggml_new_graph_custom(c(), 1024, false);
        ggml_build_forward_expand(gf, out);
        require(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);
        for (auto & s : states) require(s.accepted);
    }
    std::vector<float> copy(ggml_tensor * t) {
        require(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
        auto * p = static_cast<float *>(t->data);
        std::vector<float> x(p, p + ggml_nelements(t));
        require(finite(x));
        return x;
    }
};
ggml_tensor * spatial(graph & g, ggml_tensor * t, int gh, int gw) {
    t = ggml_permute(g.c(), t, 1, 2, 0, 3);
    t = ggml_cont_4d(g.c(), t, 2048, gw / 2, gh, 1);
    t = ggml_reshape_4d(g.c(), t, 2048, gw / 2, 2, gh / 2);
    t = ggml_permute(g.c(), t, 0, 2, 1, 3);
    return ggml_cont_3d(g.c(), t, 1024, gh * gw, 1);
}
ggml_tensor * spatial_position(graph & g, ggml_tensor * t, int gh, int gw) {
    t = ggml_cont_4d(g.c(), t, 2048, gw / 2, gh, 1);
    t = ggml_reshape_4d(g.c(), t, 2048, gw / 2, 2, gh / 2);
    t = ggml_permute(g.c(), t, 0, 2, 1, 3);
    return ggml_cont_3d(g.c(), t, 1024, gh * gw, 1);
}
ggml_tensor * conv(graph & g, ggml_tensor * kernel, ggml_tensor * pixels) {
    // Explicit F16 im2col bypasses the preserved fork's F32 conv promotion.
    auto * col = ggml_im2col(g.c(), kernel, pixels, 16, 16, 0, 0, 1, 1, true, GGML_TYPE_F16);
    auto * a = ggml_reshape_2d(g.c(), col, col->ne[0], col->ne[1] * col->ne[2] * col->ne[3]);
    auto * b = ggml_reshape_2d(g.c(), kernel, 768, 1024);
    auto * y = g.mm(a, b);
    y = ggml_reshape_4d(g.c(), y, col->ne[1], col->ne[2], col->ne[3], 1024);
    return ggml_cont(g.c(), ggml_permute(g.c(), y, 0, 1, 3, 2));
}
} // namespace
#endif
bool lighton4b_physical_f16_vision_cpu(const vision_weights & w, const vision_config & cfg, const image_inputs & image,
                                       int threads, bool physical, vision_outputs & out) {
    out = {};
#if defined(CRISPEMBED_LIGHTON4B_F16_SGEMM_ABI) && !defined(__EMSCRIPTEN__)
    if (!lighton4b_physical_f16_vision_mode(physical) || !lighton4b_physical_f16_vision_valid(physical) ||
        threads != 4 || !core_env::on("CRISPEMBED_LIGHTON4B_UPSTREAM_PHYSICAL") ||
        !core_env::on("CRISPEMBED_QWEN35_OCR") || !core_env::on("CRISPEMBED_LIGHTON4B_PHYSICAL_PILLOW_RESIZE") ||
        !core_env::on("CRISPEMBED_LIGHTON4B_IMAGE_COMPOSE") || cfg.width != 1024 || cfg.heads != 16 ||
        cfg.intermediate != 4096 || cfg.output != 2560 || cfg.patch_width != 1536 || cfg.merge != 2 ||
        cfg.position_side != 48 || cfg.eps != 1e-6f || cfg.theta != 10000.f || w.layers.size() != 24 ||
        image.grid.size() != 3 || image.grid[0] != 1 || image.grid[1] < 2 || image.grid[2] < 2 || image.grid[1] % 2 ||
        image.grid[2] % 2 || image.grid[1] > 1024 / image.grid[2] || image.height != image.grid[1] * 16 ||
        image.width != image.grid[2] * 16)
        return false;
    const int gh = image.grid[1], gw = image.grid[2], n = gh * gw;
    if (image.patches.size() != size_t(n) * 1536 || image.resized_rgb.size() != size_t(image.height) * image.width * 3)
        return false;
    try {
        auto pp = ggml_threadpool_params_default(4);
        std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(ggml_threadpool_new(&pp),
                                                                               ggml_threadpool_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        require(pool && backend);
        ggml_backend_cpu_set_n_threads(backend.get(), 4);
        ggml_backend_cpu_set_threadpool(backend.get(), pool.get());
        vision_outputs result;
        std::vector<float> pixels(size_t(image.height) * image.width * 3);
        for (int y = 0; y < image.height; ++y)
            for (int x = 0; x < image.width; ++x)
                for (int c = 0; c < 3; ++c) {
                    const float scaled = float(image.resized_rgb[(size_t(y) * image.width + x) * 3 + c]) / 255.0f;
                    pixels[size_t(c) * image.height * image.width + size_t(y) * image.width + x] = (scaled - .5f) / .5f;
                }
        size_t index = 0;
        for (int by = 0; by < gh / 2; ++by)
            for (int bx = 0; bx < gw / 2; ++bx)
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx)
                        for (int c = 0; c < 3; ++c)
                            for (int t = 0; t < 2; ++t)
                                for (int py = 0; py < 16; ++py)
                                    for (int px = 0; px < 16; ++px) {
                                        const float expected = pixels[size_t(c) * image.height * image.width +
                                                                      size_t(by * 32 + dy * 16 + py) * image.width +
                                                                      bx * 32 + dx * 16 + px];
                                        require(image.patches[index++] == expected);
                                    }
        {
            graph g(backend.get(), pool.get());
            auto * p = ggml_new_tensor_4d(g.c(), GGML_TYPE_F32, image.width, image.height, 3, 1);
            std::copy(pixels.begin(), pixels.end(), static_cast<float *>(p->data));
            auto source = g.values(w.patch, 1536, 1024);
            ggml_tensor * kernels[2];
            for (int time = 0; time < 2; ++time) {
                kernels[time] = ggml_new_tensor_4d(g.c(), GGML_TYPE_F16, 16, 16, 3, 1024);
                auto * dst = static_cast<ggml_fp16_t *>(kernels[time]->data);
                for (int row = 0; row < 1024; ++row)
                    for (int c = 0; c < 3; ++c)
                        for (int q = 0; q < 256; ++q) {
                            const auto half = ggml_fp32_to_fp16(source[size_t(row) * 1536 + c * 512 + time * 256 + q]);
                            require(std::isfinite(ggml_fp16_to_fp32(half)));
                            dst[size_t(row) * 768 + c * 256 + q] = half;
                        }
            }
            auto * patch = spatial(g, ggml_add(g.c(), conv(g, kernels[0], p), conv(g, kernels[1], p)), gh, gw);
            patch = ggml_add(g.c(), patch, g.parameter(w.patch_bias, 1024));
            auto * pos = g.parameter(w.position, 1024, 2304);
            pos = ggml_reshape_3d(g.c(), pos, 1024, 48, 48);
            pos = ggml_permute(g.c(), pos, 2, 0, 1, 3);
            pos =
                ggml_interpolate(g.c(), pos, gw, gh, 1024, 1, GGML_SCALE_MODE_BILINEAR | GGML_SCALE_FLAG_ALIGN_CORNERS);
            pos = ggml_permute(g.c(), pos, 1, 2, 0, 3);
            pos = ggml_cont_2d(g.c(), pos, 1024, n);
            pos = spatial_position(g, pos, gh, gw);
            auto * embedded = ggml_add(g.c(), patch, pos);
            g.compute(embedded);
            result.patch = g.copy(patch);
            result.position = g.copy(pos);
            result.embedded = g.copy(embedded);
        }
        auto hidden = result.embedded;
        for (const auto & layer : w.layers) {
            graph g(backend.get(), pool.get());
            auto * x = g.f32(1024, n, hidden);
            auto * normalized = g.norm(x, layer.norm1, layer.norm1_bias);
            auto * qkv = g.linear(layer.qkv, layer.qkv_bias, normalized, 1024, 3072);
            auto * Q = ggml_view_3d(g.c(), qkv, 64, 16, n, 64 * 4, qkv->nb[1], 0);
            auto * K = ggml_view_3d(g.c(), qkv, 64, 16, n, 64 * 4, qkv->nb[1], 1024 * 4);
            auto * V = ggml_view_3d(g.c(), qkv, 64, 16, n, 64 * 4, qkv->nb[1], 2048 * 4);
            auto * positions = ggml_new_tensor_1d(g.c(), GGML_TYPE_I32, n * 4);
            auto * axes = static_cast<int32_t *>(positions->data);
            int at = 0;
            for (int by = 0; by < gh / 2; ++by)
                for (int bx = 0; bx < gw / 2; ++bx)
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx) {
                            axes[at] = axes[2 * n + at] = by * 2 + dy;
                            axes[n + at] = axes[3 * n + at] = bx * 2 + dx;
                            ++at;
                        }
            int sections[4] = { 16, 16, 16, 16 };
            Q = ggml_rope_multi(g.c(), Q, positions, nullptr, 32, sections, GGML_ROPE_TYPE_VISION, 32768, 10000.f, 1, 0,
                                1, 32, 1);
            K = ggml_rope_multi(g.c(), K, positions, nullptr, 32, sections, GGML_ROPE_TYPE_VISION, 32768, 10000.f, 1, 0,
                                1, 32, 1);
            auto * q = ggml_permute(g.c(), Q, 0, 2, 1, 3);
            auto * k = ggml_permute(g.c(), K, 0, 2, 1, 3);
            auto * v = ggml_cont(g.c(), ggml_permute(g.c(), V, 1, 2, 0, 3));
            auto * probs = ggml_soft_max_ext(g.c(), ggml_mul_mat(g.c(), k, q), nullptr, .125f, 0.f);
            auto * pooled = ggml_mul_mat(g.c(), v, probs);
            pooled = ggml_cont_2d(g.c(), ggml_permute(g.c(), pooled, 0, 2, 1, 3), 1024, n);
            auto * attention = g.linear(layer.out, layer.out_bias, pooled, 1024, 1024);
            auto * residual = ggml_add(g.c(), attention, x);
            auto * norm2 = g.norm(residual, layer.norm2, layer.norm2_bias);
            auto * up = g.linear(layer.up, layer.up_bias, norm2, 1024, 4096);
            auto * activation = ggml_gelu(g.c(), up);
            auto * down = g.linear(layer.down, layer.down_bias, activation, 4096, 1024);
            auto * final = ggml_add(g.c(), residual, down);
            g.compute(final);
            hidden = g.copy(final);
            result.hidden.push_back(hidden);
        }
        {
            graph g(backend.get(), pool.get());
            auto * normalized = g.norm(g.f32(1024, n, hidden), w.merger_norm, w.merger_norm_bias);
            g.compute(normalized);
            result.merger_norm = g.copy(normalized);
        }
        {
            graph g(backend.get(), pool.get());
            auto * x = g.f32(4096, n / 4, result.merger_norm);
            auto * up = g.linear(w.merger_up, w.merger_up_bias, x, 4096, 4096);
            auto * activation = ggml_gelu(g.c(), up);
            auto * final = g.linear(w.merger_down, w.merger_down_bias, activation, 4096, 2560);
            g.compute(final);
            result.merged = g.copy(final);
        }
        require(result.merged.size() == size_t(n / 4) * 2560 && finite(result.merged));
        out = std::move(result);
        return true;
    } catch (const std::exception &) {
        out = {};
        return false;
    }
#else
    (void)w;
    (void)cfg;
    (void)image;
    (void)threads;
    (void)physical;
    return false;
#endif
}
} // namespace qwen35_ocr
