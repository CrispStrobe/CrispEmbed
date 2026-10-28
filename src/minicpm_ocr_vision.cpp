// minicpm_ocr_vision.cpp - MiniCPM-OCR-1.0 vision tower + 4x projector (ggml, CPU).
//
// SigLIP block ported from smoldocling_ocr.cpp's sd_vision_forward (pre-LN,
// biased q/k/v/out, gelu_pytorch_tanh MLP, post_layernorm), with:
//   * MiniCPM-V bucketed 70x70 learned positions (minicpm_ocr_tiles.h),
//   * one independent sequence per image (overview and each slice), i.e. the
//     reference's cu_seqlens / per-image attention mask without cross-image terms,
//   * the 4x merger: 2x2 row-major concat -> LayerNorm(4608) -> Linear -> GELU -> Linear.
// vit_merger (16x mode only) is never used here.
#include "minicpm_ocr_vision.h"

#include "core/env_gate.h"
#include "core/gelu_tanh_f32.h"
#include "core/torch_bicubic_u8.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>

namespace minicpm_ocr {
namespace {

bool has_shape(const ggml_tensor * t, int64_t a, int64_t b = 1, int64_t c = 1, int64_t d = 1) {
    return t && t->ne[0] == a && t->ne[1] == b && t->ne[2] == c && t->ne[3] == d;
}

ggml_tensor * as_f32(ggml_context * g, ggml_tensor * t) {
    if (!t || t->type == GGML_TYPE_F32) return t;
    return ggml_cast(g, t, GGML_TYPE_F32);
}

ggml_tensor * layer_norm(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b, float eps) {
    x = ggml_norm(g, x, eps);
    x = ggml_mul(g, x, as_f32(g, w));
    if (b) x = ggml_add(g, x, as_f32(g, b));
    return x;
}

ggml_tensor * linear(ggml_context * g, ggml_tensor * x, ggml_tensor * w, ggml_tensor * b) {
    x = ggml_mul_mat(g, w, x);
    if (b) x = ggml_add(g, x, as_f32(g, b));
    return x;
}

struct graph_deleter {
    void operator()(ggml_context * c) const { ggml_free(c); }
};
struct gallocr_deleter {
    void operator()(ggml_gallocr * a) const { ggml_gallocr_free(a); }
};

void read_rows(ggml_tensor * t, std::vector<float> & out) {
    const size_t n = size_t(ggml_nelements(t));
    const size_t base = out.size();
    out.resize(base + n);
    ggml_backend_tensor_get(t, out.data() + base, 0, n * sizeof(float));
}

} // namespace

bool map_vision(const core_gguf::tensor_map & tensors, vision_weights & w, std::string & error) {
    w = {};
    auto get = [&](const std::string & name) { return core_gguf::try_get(tensors, name.c_str()); };
    auto need = [&](ggml_tensor * t, const std::string & name, bool ok) {
        if (!t || !ok) {
            if (error.empty()) error = "vision tensor missing or mis-shaped: " + name;
            return false;
        }
        return true;
    };
    const int D = w.width, F = 4304, P = w.patch;
    bool ok = true;
    w.patch_w = get("v.emb.patch_embedding.weight");
    w.patch_b = get("v.emb.patch_embedding.bias");
    w.pos = get("v.emb.position_embedding.weight");
    ok &= need(w.patch_w, "v.emb.patch_embedding.weight", has_shape(w.patch_w, P, P, 3, D));
    ok &= need(w.patch_b, "v.emb.patch_embedding.bias", has_shape(w.patch_b, D));
    ok &= need(w.pos, "v.emb.position_embedding.weight", has_shape(w.pos, D, w.pos_grid * w.pos_grid));
    for (int i = 0; ok && i < 27; ++i) {
        const std::string p = "v.blk." + std::to_string(i) + ".";
        vision_layer l;
        struct item {
            ggml_tensor ** slot;
            const char * name;
            int64_t a, b;
        };
        const item items[] = {
            { &l.ln1_w, "layer_norm1.weight", D, 1 },      { &l.ln1_b, "layer_norm1.bias", D, 1 },
            { &l.ln2_w, "layer_norm2.weight", D, 1 },      { &l.ln2_b, "layer_norm2.bias", D, 1 },
            { &l.q_w, "self_attn.q_proj.weight", D, D },   { &l.q_b, "self_attn.q_proj.bias", D, 1 },
            { &l.k_w, "self_attn.k_proj.weight", D, D },   { &l.k_b, "self_attn.k_proj.bias", D, 1 },
            { &l.v_w, "self_attn.v_proj.weight", D, D },   { &l.v_b, "self_attn.v_proj.bias", D, 1 },
            { &l.o_w, "self_attn.out_proj.weight", D, D }, { &l.o_b, "self_attn.out_proj.bias", D, 1 },
            { &l.fc1_w, "mlp.fc1.weight", D, F },          { &l.fc1_b, "mlp.fc1.bias", F, 1 },
            { &l.fc2_w, "mlp.fc2.weight", F, D },          { &l.fc2_b, "mlp.fc2.bias", D, 1 },
        };
        for (const auto & it : items) {
            *it.slot = get(p + it.name);
            ok &= need(*it.slot, p + it.name, has_shape(*it.slot, it.a, it.b));
        }
        w.layers.push_back(l);
    }
    w.post_w = get("v.post_ln.weight");
    w.post_b = get("v.post_ln.bias");
    ok &= need(w.post_w, "v.post_ln.weight", has_shape(w.post_w, D));
    ok &= need(w.post_b, "v.post_ln.bias", has_shape(w.post_b, D));
    // merger (model.merger.mlp.0 = MiniCPMOcrDownsampleMLP, modeling_minicpm_ocr.py:494-510):
    // pre_norm = LayerNorm(4608, eps=1e-6) -> linear_1 -> nn.GELU() (erf) -> linear_2.
    const int M = D * w.merge * w.merge;
    w.m1_w = get("m.0.linear_1.weight");
    w.m1_b = get("m.0.linear_1.bias");
    w.m2_w = get("m.0.linear_2.weight");
    w.m2_b = get("m.0.linear_2.bias");
    ok &= need(w.m1_w, "m.0.linear_1.weight", has_shape(w.m1_w, M, M));
    ok &= need(w.m2_w, "m.0.linear_2.weight", has_shape(w.m2_w, M, w.out_width));
    if (w.m1_b && !has_shape(w.m1_b, M)) ok &= need(nullptr, "m.0.linear_1.bias", false);
    if (w.m2_b && !has_shape(w.m2_b, w.out_width)) ok &= need(nullptr, "m.0.linear_2.bias", false);
    w.mln_w = get("m.0.pre_norm.weight");
    w.mln_b = get("m.0.pre_norm.bias");
    ok &= need(w.mln_w, "m.0.pre_norm.weight", has_shape(w.mln_w, M));
    ok &= need(w.mln_b, "m.0.pre_norm.bias", has_shape(w.mln_b, M));
    return ok;
}

bool preprocess(const uint8_t * rgb, int width, int height, const minicpm_ocr_tiles::config & cfg,
                minicpm_ocr_tiles::plan & plan, std::vector<image_part> & parts) {
    parts.clear();
    if (!rgb || !minicpm_ocr_tiles::make_plan(width, height, cfg, plan)) return false;
    // MiniCPMOcrImageProcessor is a TorchvisionBackend (image_processing_minicpm_ocr.py:139-140):
    // uint8 CHW tensor -> tvF.resize(BICUBIC, antialias=True) (image_processing_backends.py:230-283)
    // = torch CPU uint8 AA bicubic, shared with the qwen35 frontend.
    auto resize = [&](int ow, int oh, std::vector<uint8_t> & hwc) {
        core_torch_resize::resize_rgb(rgb, height, width, oh, ow, hwc);
    };
    // Fused rescale+normalize (image_processing_backends.py:325-360): mean/std are
    // scaled by 1/rescale_factor = 255 -> (float32(v) - 127.5f) / 127.5f in float32.
    auto to_chw = [](const std::vector<uint8_t> & hwc, int full_w, int x0, int y0, int w, int h,
                     std::vector<float> & chw) {
        chw.resize(size_t(3) * w * h);
        for (int c = 0; c < 3; ++c)
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    const float v = float(hwc[(size_t(y0 + y) * full_w + (x0 + x)) * 3 + c]);
                    chw[(size_t(c) * h + y) * w + x] = (v - 127.5f) / 127.5f;
                }
    };
    std::vector<uint8_t> hwc;
    for (const auto & p : plan.parts) {
        image_part out;
        out.width = p.size.w;
        out.height = p.size.h;
        out.tokens = p.tokens;
        out.overview = p.overview;
        out.row = p.row;
        out.col = p.col;
        if (p.overview) {
            resize(p.size.w, p.size.h, hwc);
            to_chw(hwc, p.size.w, 0, 0, p.size.w, p.size.h, out.chw);
        } else {
            if (p.row == 0 && p.col == 0) resize(plan.refine.w, plan.refine.h, hwc);
            to_chw(hwc, plan.refine.w, p.x0, p.y0, p.size.w, p.size.h, out.chw);
        }
        parts.push_back(std::move(out));
    }
    return !parts.empty();
}

bool encode(const vision_weights & w, ggml_backend_t backend, const image_part & image, int threads,
            std::vector<float> & merged, vision_debug * debug) {
    const int P = w.patch, D = w.width, H = w.heads, hd = D / H, S = w.merge;
    if (!backend || image.width % (P * S) || image.height % (P * S) ||
        image.chw.size() != size_t(3) * image.width * image.height)
        return false;
    const int ph = image.height / P, pw = image.width / P, T = ph * pw, K = 3 * P * P;
    if (T < 1 || T > 16384) return false;

    // im2col in conv-weight order (c, ky, kx); row-major patches.
    std::vector<float> cols(size_t(T) * K);
    for (int py = 0; py < ph; ++py)
        for (int px = 0; px < pw; ++px) {
            float * dst = cols.data() + size_t(py * pw + px) * K;
            for (int c = 0; c < 3; ++c)
                for (int ky = 0; ky < P; ++ky)
                    for (int kx = 0; kx < P; ++kx)
                        dst[(c * P + ky) * P + kx] =
                            image.chw[(size_t(c) * image.height + py * P + ky) * image.width + px * P + kx];
        }
    const auto pos_ids = minicpm_ocr_tiles::position_ids(ph, pw, w.pos_grid);

    static constexpr int kMaxNodes = 8192;
    const size_t meta = ggml_tensor_overhead() * kMaxNodes + ggml_graph_overhead_custom(kMaxNodes, false);
    ggml_init_params ip = { meta, nullptr, true };
    std::unique_ptr<ggml_context, graph_deleter> gctx(ggml_init(ip));
    if (!gctx) return false;
    ggml_context * g = gctx.get();

    ggml_tensor * in = ggml_new_tensor_2d(g, GGML_TYPE_F32, K, T);
    ggml_set_name(in, "im2col");
    ggml_set_input(in);
    ggml_tensor * ids = ggml_new_tensor_1d(g, GGML_TYPE_I32, T);
    ggml_set_name(ids, "pos_ids");
    ggml_set_input(ids);

    ggml_tensor * x = linear(g, in, ggml_reshape_2d(g, w.patch_w, K, D), w.patch_b);
    x = ggml_add(g, x, ggml_get_rows(g, w.pos, ids));
    const bool dbg = debug && debug->enabled;
    std::vector<ggml_tensor *> taps;
    auto tap = [&](ggml_tensor * t) {
        if (dbg) {
            ggml_set_output(t);
            taps.push_back(t);
        }
    };
    tap(x);
    const float scale = 1.0f / std::sqrt(float(hd));
    // Default-off performance gate (not a semantics switch): fused flash attention
    // instead of the explicit softmax(QK^T)V. Promote only after A/B on the capture.
    const bool flash = core_env::on("CRISPEMBED_MINICPM_OCR_VISION_FLASH");
    for (size_t li = 0; li < w.layers.size(); ++li) {
        const auto & l = w.layers[li];
        ggml_tensor * h = layer_norm(g, x, l.ln1_w, l.ln1_b, w.eps);
        ggml_tensor * q = ggml_permute(g, ggml_reshape_3d(g, linear(g, h, l.q_w, l.q_b), hd, H, T), 0, 2, 1, 3);
        ggml_tensor * k = ggml_permute(g, ggml_reshape_3d(g, linear(g, h, l.k_w, l.k_b), hd, H, T), 0, 2, 1, 3);
        ggml_tensor * att;
        if (flash) {
            // Same fused bidirectional path as lfm2_vl_ocr.cpp (LFM2_VL_FLASH_ATTN): output
            // is already [hd, H, T], so no trailing permute.
            ggml_tensor * v = ggml_permute(g, ggml_reshape_3d(g, linear(g, h, l.v_w, l.v_b), hd, H, T), 0, 2, 1, 3);
            att = ggml_flash_attn_ext(g, q, k, v, nullptr, scale, 0.0f, 0.0f);
            ggml_flash_attn_ext_set_prec(att, GGML_PREC_F32);
        } else {
            ggml_tensor * v = ggml_cont(
                g, ggml_permute(g, ggml_reshape_3d(g, linear(g, h, l.v_w, l.v_b), hd, H, T), 1, 2, 0, 3)); // [T, hd, H]
            ggml_tensor * kq = ggml_mul_mat(g, k, q);                                                      // [T, T, H]
            kq = ggml_soft_max_ext(g, kq, nullptr, scale, 0.0f);
            att = ggml_mul_mat(g, v, kq); // [hd, T, H]
            att = ggml_cont(g, ggml_permute(g, att, 0, 2, 1, 3));
        }
        att = linear(g, ggml_reshape_2d(g, att, D, T), l.o_w, l.o_b);
        x = ggml_add(g, x, att);
        h = layer_norm(g, x, l.ln2_w, l.ln2_b, w.eps);
        h = linear(g, h, l.fc1_w, l.fc1_b);
        h = core_cpu::gelu_tanh_f32(g, ggml_cont(g, h)); // gelu_pytorch_tanh without the FP16 LUT
        h = linear(g, h, l.fc2_w, l.fc2_b);
        x = ggml_add(g, x, h);
        if (li == 0 || li == 6 || li == 26) tap(x);
    }
    x = layer_norm(g, x, w.post_w, w.post_b, w.eps);
    tap(x);
    // 2x2 merge (MiniCPMOcrMerger, modeling_minicpm_ocr.py:540-546):
    // view(mh, 2, mw, 2, D).permute(0, 2, 1, 3, 4) -> per merged cell the
    // order is (dy, dx) row-major: [(0,0),(0,1),(1,0),(1,1)] x D; cells row-major.
    ggml_tensor * m = ggml_reshape_4d(g, x, D * S, pw / S, S, ph / S);
    m = ggml_cont(g, ggml_permute(g, m, 0, 2, 1, 3)); // [2D, 2(dy), pw/2, ph/2]
    m = ggml_reshape_2d(g, m, D * S * S, (pw / S) * (ph / S));
    m = layer_norm(g, m, w.mln_w, w.mln_b, w.merger_eps);
    m = linear(g, m, w.m1_w, w.m1_b);
    m = ggml_gelu_erf(g, m); // nn.GELU() default approximate="none" (erf), modeling_minicpm_ocr.py:502
    m = linear(g, m, w.m2_w, w.m2_b);
    ggml_set_name(m, "merged");
    ggml_set_output(m);

    ggml_cgraph * gf = ggml_new_graph_custom(g, kMaxNodes, false);
    for (auto * t : taps) ggml_build_forward_expand(gf, t);
    ggml_build_forward_expand(gf, m);
    std::unique_ptr<ggml_gallocr, gallocr_deleter> alloc(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)));
    if (!alloc || !ggml_gallocr_alloc_graph(alloc.get(), gf)) return false;
    ggml_backend_tensor_set(in, cols.data(), 0, cols.size() * sizeof(float));
    ggml_backend_tensor_set(ids, pos_ids.data(), 0, pos_ids.size() * sizeof(int32_t));
    if (ggml_backend_is_cpu(backend)) ggml_backend_cpu_set_n_threads(backend, threads);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) return false;

    const size_t before = merged.size();
    read_rows(m, merged);
    if (merged.size() - before != size_t(image.tokens) * w.out_width) return false;
    if (!std::all_of(merged.begin() + before, merged.end(), [](float f) { return std::isfinite(f); })) return false;
    if (dbg && taps.size() == 5) {
        read_rows(taps[0], debug->embeddings);
        read_rows(taps[1], debug->layer0);
        read_rows(taps[2], debug->layer6);
        read_rows(taps[3], debug->layer26);
        read_rows(taps[4], debug->post_ln);
        debug->merged.insert(debug->merged.end(), merged.begin() + before, merged.end());
    }
    return true;
}

} // namespace minicpm_ocr
