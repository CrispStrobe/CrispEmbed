#include "ppocrv6_ocr.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/cpu_ops.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "crispembed_diff.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

using core_cpu::conv2d_cpu;
using core_cpu::gelu;
using core_cpu::hardswish_inplace;
using core_cpu::linear_cpu;
using core_cpu::softmax;
using core_cpu::to_f32;

struct pp_conv {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
    int in_ch = 0, out_ch = 0, kh = 1, kw = 1, stride = 1, stride_h = 1, stride_w = 1, pad_h = 0, pad_w = 0, groups = 1;
    // Recognizer crops reuse the same convolution weights. Cache the CPU
    // representation once, matching the detector's weight policy; otherwise
    // every detected line repeats a full GGUF→F32 dequantization.
    mutable std::vector<float> wf, bf;
};

struct pp_block {
    pp_conv dw, cm1, cm2, se1, se2;
    bool se = false;
    bool residual = false;
    bool silu_act = false;
};

struct pp_svtr {
    ggml_tensor *ln1_w = nullptr, *ln1_b = nullptr;
    ggml_tensor *qkv_w = nullptr, *qkv_b = nullptr;
    ggml_tensor *proj_w = nullptr, *proj_b = nullptr;
    ggml_tensor *ln2_w = nullptr, *ln2_b = nullptr;
    ggml_tensor *fc1_w = nullptr, *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr, *fc2_b = nullptr;
    std::vector<float> ln1_wf, ln1_bf, qkv_wf, qkv_bf, proj_wf, proj_bf;
    std::vector<float> ln2_wf, ln2_bf, fc1_wf, fc1_bf, fc2_wf, fc2_bf;
};

struct pp_graph_state {
    ggml_backend_t backend = nullptr;
    ggml_backend_t cpu_backend = nullptr;
    ggml_backend_sched_t sched = nullptr;
    ggml_context * graph_ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * output = nullptr;
    bool logits_output = false;
    bool svtr_prefix_output = false;
    bool svtr_decoder_output = false;
    std::vector<uint8_t> graph_meta;
    std::unordered_map<const ggml_tensor *, ggml_tensor *> resident;
    std::vector<ggml_context *> resident_ctxs;
    std::vector<ggml_backend_buffer_t> resident_bufs;
    std::vector<std::pair<const char *, ggml_tensor *>> debug_taps;
    bool attempted = false;
    bool ready = false;
    // The recognizer graph has fixed crop dimensions. Keep its scheduler
    // allocation alive across line crops; rebuilding the allocation for every
    // crop defeats the persistent-graph design and needlessly re-plans backend
    // buffers.
    bool allocated = false;
};

struct ppocrv6_ocr_context {
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    std::vector<std::string> vocab;
    std::string result;
    std::vector<float> scratch;
    std::string variant;
    bool large_stem = false;
    int hidden = 0;
    int vocab_size = 0;
    int last_ch = 0;
    std::vector<pp_conv> stem;
    std::vector<std::vector<pp_block>> stages;
    pp_conv head_dw, head_pw;
    ggml_tensor * fc1_w = nullptr;
    ggml_tensor * fc1_b = nullptr;
    ggml_tensor * fc2_w = nullptr;
    ggml_tensor * fc2_b = nullptr;
    std::vector<float> fc1_wf, fc1_bf, fc2_wf, fc2_bf;
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr, *norm1_mean = nullptr, *norm1_var = nullptr;
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr, *norm2_mean = nullptr, *norm2_var = nullptr;
    ggml_tensor *svtr_norm_w = nullptr, *svtr_norm_b = nullptr;
    ggml_tensor *svtr_head_w = nullptr, *svtr_head_b = nullptr;
    std::vector<pp_svtr> svtr;
    std::vector<float> svtr_norm_wf, svtr_norm_bf, svtr_head_wf, svtr_head_bf;
    std::unique_ptr<crispembed_diff::Ref> diff;
    pp_graph_state graph;
};

static ggml_tensor * get(const core_gguf::tensor_map & m, const std::string & n) {
    return core_gguf::try_get(m, n.c_str());
}

static pp_conv conv(const core_gguf::tensor_map & m, const std::string & n, int in, int out, int k, int stride,
                    int groups = 1, int stride_w = 0) {
    pp_conv c;
    c.w = get(m, n + ".weight");
    c.b = get(m, n + ".bias");
    c.in_ch = in;
    c.out_ch = out;
    c.kh = c.kw = k;
    c.stride = stride;
    c.stride_h = stride;
    c.stride_w = stride_w ? stride_w : stride;
    c.pad_h = c.pad_w = k / 2;
    c.groups = groups;
    return c;
}

static bool apply_conv(const pp_conv & c, const std::vector<float> & in, int h, int w, std::vector<float> & out,
                       int & oh, int & ow) {
    if (!c.w) {
        fprintf(stderr, "ppocrv6: missing convolution weight (%d -> %d, k=%dx%d)\n", c.in_ch, c.out_ch, c.kh, c.kw);
        return false;
    }
    oh = (h + 2 * c.pad_h - c.kh) / c.stride_h + 1;
    ow = (w + 2 * c.pad_w - c.kw) / c.stride_w + 1;
    out.assign((size_t)c.out_ch * oh * ow, 0.0f);
    if (c.wf.empty()) c.wf = to_f32(c.w);
    if (c.b && c.bf.empty()) c.bf = to_f32(c.b);
    const auto & ww = c.wf;
    const auto & bb = c.bf;
    // Even 2x2 kernels in the large recognizer stem use asymmetric effective
    // borders (one branch is valid, the other restores the spatial size).
    // Keep this small path explicit; it also avoids backend-specific
    // even-kernel padding behavior in the generic CPU helper.
    if (c.kh == 2 && c.kw == 2) {
        const int ci = c.in_ch / c.groups, co = c.out_ch / c.groups;
        for (int g = 0; g < c.groups; ++g)
            for (int oc = 0; oc < co; ++oc)
                for (int oy = 0; oy < oh; ++oy)
                    for (int ox = 0; ox < ow; ++ox) {
                        float sum = bb.empty() ? 0.0f : bb[g * co + oc];
                        for (int ic = 0; ic < ci; ++ic)
                            for (int ky = 0; ky < 2; ++ky)
                                for (int kx = 0; kx < 2; ++kx) {
                                    const int iy = oy * c.stride_h + ky - c.pad_h;
                                    const int ix = ox * c.stride_w + kx - c.pad_w;
                                    if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                        sum += in[(g * ci + ic) * h * w + iy * w + ix] *
                                               ww[(g * co + oc) * ci * 4 + ic * 4 + ky * 2 + kx];
                                }
                        out[(g * co + oc) * oh * ow + oy * ow + ox] = sum;
                    }
        return true;
    }
    if (c.pad_h == c.pad_w && c.stride_h == c.stride_w) {
        conv2d_cpu(in.data(), out.data(), ww.data(), bb.empty() ? nullptr : bb.data(), c.in_ch, c.out_ch, h, w, c.kh,
                   c.kw, c.stride, c.pad_h, c.groups);
    } else {
        // cpu_ops uses one symmetric padding value.  Materialize the
        // rectangular padding used by the SVTR 1x7 convolution instead.
        const int ph = c.pad_h, pw = c.pad_w;
        std::vector<float> padded((size_t)c.in_ch * (h + 2 * ph) * (w + 2 * pw), 0.0f);
        const int pwid = w + 2 * pw;
        for (int ch = 0; ch < c.in_ch; ++ch)
            for (int yy = 0; yy < h; ++yy)
                std::memcpy(padded.data() + (size_t)ch * (h + 2 * ph) * pwid + (yy + ph) * pwid + pw,
                            in.data() + (size_t)ch * h * w + yy * w, sizeof(float) * w);
        if (c.stride_h == c.stride_w) {
            conv2d_cpu(padded.data(), out.data(), ww.data(), bb.empty() ? nullptr : bb.data(), c.in_ch, c.out_ch,
                       h + 2 * ph, w + 2 * pw, c.kh, c.kw, c.stride_h, 0, c.groups);
        } else {
            const int cin = c.in_ch / c.groups, cout = c.out_ch / c.groups, ks = cin * c.kh * c.kw;
            for (int g = 0; g < c.groups; ++g)
                for (int oc = 0; oc < cout; ++oc)
                    for (int oy = 0; oy < oh; ++oy)
                        for (int ox = 0; ox < ow; ++ox) {
                            float sum = bb.empty() ? 0.0f : bb[g * cout + oc];
                            const float * wt = ww.data() + (g * cout + oc) * ks;
                            int kk = 0;
                            for (int ic = 0; ic < cin; ++ic)
                                for (int ky = 0; ky < c.kh; ++ky)
                                    for (int kx = 0; kx < c.kw; ++kx, ++kk)
                                        sum += padded[(size_t)(g * cin + ic) * (h + 2 * ph) * (w + 2 * pw) +
                                                      (oy * c.stride_h + ky) * (w + 2 * pw) + ox * c.stride_w + kx] *
                                               wt[kk];
                            out[(size_t)(g * cout + oc) * oh * ow + oy * ow + ox] = sum;
                        }
        }
    }
    return true;
}

static void activate(std::vector<float> & x, bool hs) {
    if (hs)
        hardswish_inplace(x.data(), (int)x.size());
    else
        for (float & v : x) v = gelu(v);
}

static void silu(std::vector<float> & x) {
    for (float & v : x) v = v / (1.0f + std::exp(-v));
}

static void relu(std::vector<float> & x) {
    for (float & v : x) v = std::max(0.0f, v);
}

static void pad_right_bottom(const std::vector<float> & in, int channels, int h, int w, std::vector<float> & out,
                             int & oh, int & ow) {
    oh = h + 1;
    ow = w + 1;
    out.assign((size_t)channels * oh * ow, 0.0f);
    for (int c = 0; c < channels; ++c)
        for (int y = 0; y < h; ++y)
            std::memcpy(out.data() + (size_t)c * oh * ow + y * ow, in.data() + (size_t)c * h * w + y * w,
                        sizeof(float) * w);
}

static void maxpool2x2_stride1(const std::vector<float> & in, int channels, int h, int w, std::vector<float> & out,
                               int & oh, int & ow) {
    oh = std::max(1, h - 1);
    ow = std::max(1, w - 1);
    out.assign((size_t)channels * oh * ow, 0.0f);
    for (int c = 0; c < channels; ++c)
        for (int y = 0; y < oh; ++y)
            for (int x = 0; x < ow; ++x) {
                const float * p = in.data() + (size_t)c * h * w + y * w + x;
                out[(size_t)c * oh * ow + y * ow + x] = std::max(std::max(p[0], p[1]), std::max(p[w], p[w + 1]));
            }
}

static void layernorm_tokens(std::vector<float> & x, int tokens, int channels, ggml_tensor * w, ggml_tensor * b) {
    auto ww = to_f32(w), bb = to_f32(b);
    for (int t = 0; t < tokens; ++t) {
        float mean = 0.0f;
        for (int c = 0; c < channels; ++c) mean += x[t * channels + c];
        mean /= channels;
        float var = 0.0f;
        for (int c = 0; c < channels; ++c) {
            float d = x[t * channels + c] - mean;
            var += d * d;
        }
        var /= channels;
        for (int c = 0; c < channels; ++c)
            x[t * channels + c] = (x[t * channels + c] - mean) / std::sqrt(var + 1e-5f) * ww[c] + bb[c];
    }
}

static void layernorm_tokens(std::vector<float> & x, int tokens, int channels, const std::vector<float> & w,
                             const std::vector<float> & b) {
    for (int t = 0; t < tokens; ++t) {
        float mean = 0.0f;
        for (int c = 0; c < channels; ++c) mean += x[t * channels + c];
        mean /= channels;
        float var = 0.0f;
        for (int c = 0; c < channels; ++c) {
            const float d = x[t * channels + c] - mean;
            var += d * d;
        }
        var /= channels;
        for (int c = 0; c < channels; ++c)
            x[t * channels + c] = (x[t * channels + c] - mean) / std::sqrt(var + 1e-5f) * w[c] + b[c];
    }
}

static void linear_vec(const std::vector<float> & x, std::vector<float> & y, ggml_tensor * w, ggml_tensor * b) {
    auto ww = to_f32(w), bb = to_f32(b);
    const int out = (int)bb.size(), in = (int)x.size();
    y.resize(out);
    linear_cpu(x.data(), y.data(), in, out, ww.data(), bb.data());
}

static void linear_vec(const std::vector<float> & x, std::vector<float> & y, const std::vector<float> & w,
                       const std::vector<float> & b) {
    y.resize(b.size());
    linear_cpu(x.data(), y.data(), (int)x.size(), (int)b.size(), w.data(), b.data());
}

static void bn1d(std::vector<float> & x, int channels, int length, ggml_tensor * w, ggml_tensor * b, ggml_tensor * mean,
                 ggml_tensor * var) {
    if (!w || !b || !mean || !var) return;
    auto ww = to_f32(w), bb = to_f32(b), mm = to_f32(mean), vv = to_f32(var);
    for (int c = 0; c < channels; ++c) {
        const float scale = ww[c] / std::sqrt(vv[c] + 1e-5f);
        const float shift = bb[c] - mm[c] * scale;
        for (int i = 0; i < length; ++i) x[c * length + i] = x[c * length + i] * scale + shift;
    }
}

static bool run_block(pp_block & b, std::vector<float> & x, int & h, int & w, std::vector<float> * tap_dw = nullptr,
                      std::vector<float> * tap_cm1 = nullptr) {
    std::vector<float> y, z;
    int oh, ow;
    if (!apply_conv(b.dw, x, h, w, y, oh, ow)) return false;
    if (tap_dw) *tap_dw = y;
    if (b.se) {
        // The SE convolutions operate on the global average, then scale the
        // depthwise output. Both gates are kept at F16/F32 by the quantizer.
        std::vector<float> pooled(b.dw.out_ch, 0.0f);
        for (int c = 0; c < b.dw.out_ch; ++c)
            for (int i = 0; i < oh * ow; ++i) pooled[c] += y[c * oh * ow + i];
        for (float & v : pooled) v /= float(oh * ow);
        std::vector<float> gate;
        int gh, gw;
        if (!apply_conv(b.se1, pooled, 1, 1, gate, gh, gw)) return false;
        for (float & v : gate) v = std::max(v, 0.0f);
        if (!apply_conv(b.se2, gate, 1, 1, pooled, gh, gw)) return false;
        for (float & v : pooled) v = std::min(1.0f, std::max(0.0f, (v + 3.0f) / 6.0f));
        for (int c = 0; c < b.dw.out_ch; ++c)
            for (int i = 0; i < oh * ow; ++i) y[c * oh * ow + i] *= pooled[c];
    }
    if (!apply_conv(b.cm1, y, oh, ow, z, oh, ow)) return false;
    if (b.silu_act)
        silu(z);
    else
        activate(z, false);
    if (tap_cm1) *tap_cm1 = z;
    std::vector<float> out;
    int nh, nw;
    if (!apply_conv(b.cm2, z, oh, ow, out, nh, nw)) return false;
    if (b.residual && out.size() == y.size())
        for (size_t i = 0; i < out.size(); ++i) out[i] += y[i];
    x.swap(out);
    h = nh;
    w = nw;
    return true;
}

static bool map_model(ppocrv6_ocr_context * c) {
    const auto & m = c->wl.tensors;
    const bool tiny = c->variant == "tiny";
    const int s = tiny ? 24 : (c->variant == "small" ? 48 : 64);
    const int stem2 = tiny ? 48 : (c->variant == "small" ? 96 : 128);
    c->hidden = tiny ? 80 : (c->variant == "small" ? 120 : 192);
    if (c->vocab_size == 0) c->vocab_size = (int)c->vocab.size() + 1;
    if (tiny) {
        c->stem.push_back(conv(m, "rec.bb.stem.conv1.conv", 3, s, 3, 2));
        c->stem.push_back(conv(m, "rec.bb.stem.conv2.conv", s, stem2, 3, 2));
    } else {
        c->large_stem = true;
        c->stem.push_back(conv(m, "rec.bb.stem.stem1.conv", 3, s, 3, 2));
        c->stem.push_back(conv(m, "rec.bb.stem.stem2a.conv", s, s / 2, 2, 1));
        c->stem.back().pad_h = c->stem.back().pad_w = 0;
        c->stem.push_back(conv(m, "rec.bb.stem.stem2b.conv", s / 2, s, 2, 1));
        c->stem.back().pad_h = c->stem.back().pad_w = 0;
        c->stem.push_back(conv(m, "rec.bb.stem.stem3.conv", s * 2, s, 3, 2));
        c->stem.push_back(conv(m, "rec.bb.stem.stem4.conv", s, stem2, 1, 1));
    }
    int widths[4];
    if (tiny) {
        widths[0] = 48;
        widths[1] = 48;
        widths[2] = 96;
        widths[3] = 160;
    } else if (c->variant == "small") {
        widths[0] = 96;
        widths[1] = 96;
        widths[2] = 192;
        widths[3] = 384;
    } else {
        widths[0] = 128;
        widths[1] = 256;
        widths[2] = 512;
        widths[3] = 768;
    }
    c->stages.resize(4);
    for (int si = 0; si < 4; ++si) {
        const std::string p = "rec.bb.blk." + std::to_string(si) + ".b.";
        // The converter preserves the source block order; use tensor presence
        // to discover the number of blocks, making this work for all three
        // official width variants without a second model-specific registry.
        for (int bi = 0; bi < 16; ++bi) {
            std::string q = p + std::to_string(bi);
            if (!get(m, q + ".cm1.weight")) break;
            int in = bi == 0 && si > 0 ? widths[si - 1] : widths[si];
            int out = widths[si];
            pp_block b;
            const bool down = bi == 0 && (si == 2 || si == 3);
            b.dw = conv(m, q + ".dw", in, in, 3, down ? 2 : 1, in, down && !tiny ? 1 : 0);
            b.cm1 = conv(m, q + ".cm1", in, in * 2, 1, 1);
            b.cm2 = conv(m, q + ".cm2", in * 2, out, 1, 1);
            b.se = get(m, q + ".se1.weight") != nullptr;
            if (b.se) {
                b.se1 = conv(m, q + ".se1", in, std::max(1, in / 4), 1, 1);
                b.se2 = conv(m, q + ".se2", std::max(1, in / 4), in, 1, 1);
            }
            b.residual = in == out && b.dw.stride == 1;
            // PPLCNetV4 uses the configured activation in the stem (SILU for
            // PP-OCRv6) but the channel mixer activation is GELU.
            b.silu_act = false;
            c->stages[si].push_back(b);
        }
    }
    c->head_dw = conv(m, "rec.head.conv1", widths[3], widths[3], 5, 1, widths[3]);
    c->head_dw.kh = 1;
    c->head_dw.kw = 5;
    // cpu_ops::conv2d uses symmetric padding for both axes; the subsequent
    // sequence path accepts the valid-width result of this 1-D convolution.
    c->head_dw.pad_h = 0;
    c->head_dw.pad_w = 0;
    c->head_pw = conv(m, "rec.head.conv2", widths[3], widths[3], 1, 1);
    c->fc1_w = get(m, "rec.head.fc1.weight");
    c->fc1_b = get(m, "rec.head.fc1.bias");
    c->fc2_w = get(m, "rec.head.fc2.weight");
    c->fc2_b = get(m, "rec.head.fc2.bias");
    c->fc1_wf = to_f32(c->fc1_w);
    c->fc1_bf = to_f32(c->fc1_b);
    c->fc2_wf = to_f32(c->fc2_w);
    c->fc2_bf = to_f32(c->fc2_b);
    c->norm1_w = get(m, "rec.head.norm1.weight");
    c->norm1_b = get(m, "rec.head.norm1.bias");
    c->norm1_mean = get(m, "rec.head.norm1.running_mean");
    c->norm1_var = get(m, "rec.head.norm1.running_var");
    c->norm2_w = get(m, "rec.head.norm2.weight");
    c->norm2_b = get(m, "rec.head.norm2.bias");
    c->norm2_mean = get(m, "rec.head.norm2.running_mean");
    c->norm2_var = get(m, "rec.head.norm2.running_var");
    if (!tiny) {
        const std::string p = "rec.head.encoder.";
        auto load_svtr = [&](int i) {
            pp_svtr b;
            const std::string q = p + "svtr_block." + std::to_string(i) + ".";
            b.ln1_w = get(m, q + "layer_norm1.weight");
            b.ln1_b = get(m, q + "layer_norm1.bias");
            b.qkv_w = get(m, q + "self_attn.qkv.weight");
            b.qkv_b = get(m, q + "self_attn.qkv.bias");
            b.proj_w = get(m, q + "self_attn.projection.weight");
            b.proj_b = get(m, q + "self_attn.projection.bias");
            b.ln2_w = get(m, q + "layer_norm2.weight");
            b.ln2_b = get(m, q + "layer_norm2.bias");
            b.fc1_w = get(m, q + "mlp.fc1.weight");
            b.fc1_b = get(m, q + "mlp.fc1.bias");
            b.fc2_w = get(m, q + "mlp.fc2.weight");
            b.fc2_b = get(m, q + "mlp.fc2.bias");
            b.ln1_wf = to_f32(b.ln1_w);
            b.ln1_bf = to_f32(b.ln1_b);
            b.qkv_wf = to_f32(b.qkv_w);
            b.qkv_bf = to_f32(b.qkv_b);
            b.proj_wf = to_f32(b.proj_w);
            b.proj_bf = to_f32(b.proj_b);
            b.ln2_wf = to_f32(b.ln2_w);
            b.ln2_bf = to_f32(b.ln2_b);
            b.fc1_wf = to_f32(b.fc1_w);
            b.fc1_bf = to_f32(b.fc1_b);
            b.fc2_wf = to_f32(b.fc2_w);
            b.fc2_bf = to_f32(b.fc2_b);
            c->svtr.push_back(b);
        };
        load_svtr(0);
        load_svtr(1);
        c->svtr_norm_w = get(m, p + "norm.weight");
        c->svtr_norm_b = get(m, p + "norm.bias");
        c->svtr_head_w = get(m, "rec.head.head.weight");
        c->svtr_head_b = get(m, "rec.head.head.bias");
        c->svtr_norm_wf = to_f32(c->svtr_norm_w);
        c->svtr_norm_bf = to_f32(c->svtr_norm_b);
        c->svtr_head_wf = to_f32(c->svtr_head_w);
        c->svtr_head_bf = to_f32(c->svtr_head_b);
    }
    return !c->stem.empty() && (tiny ? c->fc2_w != nullptr : c->svtr_head_w != nullptr && c->svtr.size() == 2);
}

// The GGUF loader owns the original leaves.  A graph backend must not consume
// those leaves directly when they live on another backend, and convolution
// kernels need an explicit [KW, KH, IC/group, OC] view.  Materialize each leaf
// once on the graph backend, preserving the contiguous GGUF order.
static ggml_tensor * pp_graph_resident(ppocrv6_ocr_context * c, const ggml_tensor * src, ggml_type type, int64_t ne0,
                                       int64_t ne1, int64_t ne2, int64_t ne3) {
    if (!src || !c->graph.backend) return nullptr;
    auto it = c->graph.resident.find(src);
    if (it != c->graph.resident.end()) return it->second;
    std::vector<float> data = to_f32(src);
    ggml_init_params ip = { ggml_tensor_overhead() + 64, nullptr, true };
    ggml_context * wc = ggml_init(ip);
    if (!wc) return nullptr;
    ggml_tensor * dst = ggml_new_tensor_4d(wc, type, ne0, ne1, ne2, ne3);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wc, c->graph.backend);
    if (!dst || !buf) {
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(wc);
        return nullptr;
    }
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(data.size());
        ggml_fp32_to_fp16_row(data.data(), half.data(), (int64_t)half.size());
        ggml_backend_tensor_set(dst, half.data(), 0, ggml_nbytes(dst));
    } else {
        ggml_backend_tensor_set(dst, data.data(), 0, ggml_nbytes(dst));
    }
    c->graph.resident[src] = dst;
    c->graph.resident_ctxs.push_back(wc);
    c->graph.resident_bufs.push_back(buf);
    return dst;
}

// GGUF preserves the converter's contiguous Paddle [OC, IC/G, KH, KW]
// convolution bytes, while ggml_conv_2d consumes [KW, KH, IC/G, OC]. The
// dimensions alone cannot express this layout change; materialize the
// reordered tensor once on the selected graph backend.
static ggml_tensor * pp_graph_resident_conv(ppocrv6_ocr_context * c, const pp_conv & p, ggml_type type) {
    if (!p.w || !c->graph.backend) return nullptr;
    auto it = c->graph.resident.find(p.w);
    if (it != c->graph.resident.end()) return it->second;
    const int icg = p.in_ch / std::max(1, p.groups);
    const size_t n = (size_t)p.kw * p.kh * icg * p.out_ch;
    const std::vector<float> src = to_f32(p.w);
    if (src.size() < n) return nullptr;
    std::vector<float> data(n);
    if (p.groups == p.in_ch) {
        // The converter's depthwise tensors are already channel-major
        // [OC, KH, KW], which is the layout consumed by ggml's depthwise
        // im2col path after its [KW, KH, 1, OC] view.
        data = src;
    } else {
        for (int oc = 0; oc < p.out_ch; ++oc)
            for (int ic = 0; ic < icg; ++ic)
                for (int ky = 0; ky < p.kh; ++ky)
                    for (int kx = 0; kx < p.kw; ++kx) {
                        const size_t paddle = ((size_t)oc * icg + ic) * p.kh * p.kw + (size_t)ky * p.kw + kx;
                        const size_t ggml = (size_t)kx + (size_t)p.kw * (ky + (size_t)p.kh * (ic + (size_t)icg * oc));
                        data[ggml] = src[paddle];
                    }
    }
    ggml_init_params ip = { ggml_tensor_overhead() + 64, nullptr, true };
    ggml_context * wc = ggml_init(ip);
    if (!wc) return nullptr;
    ggml_tensor * dst = ggml_new_tensor_4d(wc, type, p.kw, p.kh, icg, p.out_ch);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(wc, c->graph.backend);
    if (!dst || !buf) {
        if (buf) ggml_backend_buffer_free(buf);
        ggml_free(wc);
        return nullptr;
    }
    if (type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(data.size());
        ggml_fp32_to_fp16_row(data.data(), half.data(), (int64_t)half.size());
        ggml_backend_tensor_set(dst, half.data(), 0, ggml_nbytes(dst));
    } else {
        ggml_backend_tensor_set(dst, data.data(), 0, ggml_nbytes(dst));
    }
    if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG"))
        fprintf(stderr, "[ppocrv6-graph-weight] conv=%dx%dx%dx%d src0=%.7g dst0=%.7g\n", p.kw, p.kh, icg, p.out_ch,
                src[0], data[0]);
    c->graph.resident[p.w] = dst;
    c->graph.resident_ctxs.push_back(wc);
    c->graph.resident_bufs.push_back(buf);
    return dst;
}

static ggml_tensor * pp_graph_conv(ppocrv6_ocr_context * c, ggml_context * g, ggml_tensor * x, const pp_conv & p) {
    if (!p.w) {
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG"))
            fprintf(stderr, "ppocrv6 graph unsupported conv weights=%d\n", p.w ? 1 : 0);
        return nullptr;
    }
    const bool dw = p.groups == p.in_ch;
    const int icg = dw ? 1 : p.in_ch / p.groups;
    // Depthwise im2col requires F16; regular convolutions stay F32 to preserve
    // the scalar reference's accumulation precision.
    // Keep the CPU diagnostic graph in F32 so it can be compared against the
    // scalar reference without introducing an avoidable half-precision seam;
    // Metal depthwise im2col still uses F16 for its supported fast path.
    const ggml_type graph_type = dw && !ggml_backend_is_cpu(c->graph.backend) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    ggml_tensor * w = pp_graph_resident_conv(c, p, graph_type);
    if (!w) {
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG"))
            fprintf(stderr, "ppocrv6 graph resident conv allocation failed\n");
        return nullptr;
    }
    // Keep the generic depthwise im2col path here: it supports the explicit
    // [W,H,C] graph layout on both CPU and accelerator backends. The direct
    // depthwise kernel was tested but did not improve parity and remains out
    // of the accepted path until its layout contract is independently gated.
    // ggml names the spatial arguments x/y ([W,H]); pp_conv stores them as
    // height/width to match the scalar reference.
    ggml_tensor * y = dw ? ggml_conv_2d_dw(g, w, x, p.stride_w, p.stride_h, p.pad_w, p.pad_h, 1, 1)
                         : ggml_conv_2d(g, w, x, p.stride_w, p.stride_h, p.pad_w, p.pad_h, 1, 1);
    if (p.b) {
        ggml_tensor * b = pp_graph_resident(c, p.b, GGML_TYPE_F32, p.out_ch, 1, 1, 1);
        if (!b) return nullptr;
        y = ggml_add(g, y, ggml_reshape_3d(g, b, 1, 1, p.out_ch));
    }
    return y;
}

static ggml_tensor * pp_graph_block(ppocrv6_ocr_context * c, ggml_context * g, ggml_tensor * x, const pp_block & b) {
    ggml_tensor * y = pp_graph_conv(c, g, x, b.dw);
    if (!y) return nullptr;
    if (b.se) {
        ggml_tensor * pooled = ggml_pool_2d(g, y, GGML_OP_POOL_AVG, y->ne[0], y->ne[1], y->ne[0], y->ne[1], 0, 0);
        ggml_tensor * gate = pp_graph_conv(c, g, pooled, b.se1);
        if (!gate) return nullptr;
        gate = ggml_relu(g, gate);
        gate = pp_graph_conv(c, g, gate, b.se2);
        if (!gate) return nullptr;
        gate = ggml_hardsigmoid(g, gate);
        y = ggml_mul(g, y, gate);
    }
    ggml_tensor * z = pp_graph_conv(c, g, y, b.cm1);
    if (!z) return nullptr;
    z = b.silu_act ? ggml_silu(g, z) : ggml_gelu(g, z);
    ggml_tensor * out = pp_graph_conv(c, g, z, b.cm2);
    if (!out) return nullptr;
    if (b.residual) out = ggml_add(g, out, y);
    return out;
}

static ggml_tensor * pp_graph_bn(ppocrv6_ocr_context * c, ggml_context * g, ggml_tensor * x, ggml_tensor * w,
                                 ggml_tensor * b, ggml_tensor * mean, ggml_tensor * var, int channels) {
    if (!w || !b || !mean || !var) return x;
    ggml_tensor * rw = pp_graph_resident(c, w, GGML_TYPE_F32, channels, 1, 1, 1);
    ggml_tensor * rb = pp_graph_resident(c, b, GGML_TYPE_F32, channels, 1, 1, 1);
    ggml_tensor * rm = pp_graph_resident(c, mean, GGML_TYPE_F32, channels, 1, 1, 1);
    ggml_tensor * rv = pp_graph_resident(c, var, GGML_TYPE_F32, channels, 1, 1, 1);
    if (!rw || !rb || !rm || !rv) return nullptr;
    ggml_tensor * scale = ggml_div(g, rw, ggml_sqrt(g, ggml_scale_bias(g, rv, 1.0f, 1e-5f)));
    ggml_tensor * shift = ggml_sub(g, rb, ggml_mul(g, rm, scale));
    scale = ggml_reshape_3d(g, scale, 1, 1, channels);
    shift = ggml_reshape_3d(g, shift, 1, 1, channels);
    return ggml_add(g, ggml_mul(g, x, scale), shift);
}

static ggml_tensor * pp_graph_layernorm(ppocrv6_ocr_context * c, ggml_context * g, ggml_tensor * x, ggml_tensor * w_src,
                                        ggml_tensor * b_src, int channels) {
    if (!w_src || !b_src) return x;
    ggml_tensor * w = pp_graph_resident(c, w_src, GGML_TYPE_F32, channels, 1, 1, 1);
    ggml_tensor * b = pp_graph_resident(c, b_src, GGML_TYPE_F32, channels, 1, 1, 1);
    if (!w || !b) return nullptr;
    w = ggml_reshape_2d(g, w, channels, 1);
    b = ggml_reshape_2d(g, b, channels, 1);
    return ggml_add(g, ggml_mul(g, ggml_norm(g, x, 1e-5f), w), b);
}

static ggml_tensor * pp_graph_linear(ppocrv6_ocr_context * c, ggml_context * g, ggml_tensor * x, ggml_tensor * wt,
                                     ggml_tensor * bias) {
    if (!wt || ggml_n_dims(wt) < 2) {
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) fprintf(stderr, "ppocrv6 graph missing linear weights\n");
        return nullptr;
    }
    ggml_tensor * w = pp_graph_resident(c, wt, GGML_TYPE_F32, wt->ne[0], wt->ne[1], 1, 1);
    if (!w) return nullptr;
    if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG"))
        fprintf(stderr, "ppocrv6 graph linear wt=%lldx%lld x=%lldx%lld\n", (long long)wt->ne[0], (long long)wt->ne[1],
                (long long)x->ne[0], (long long)x->ne[1]);
    ggml_tensor * y = ggml_mul_mat(g, w, x);
    if (bias) {
        ggml_tensor * b = pp_graph_resident(c, bias, GGML_TYPE_F32, bias->ne[0], 1, 1, 1);
        if (!b) return nullptr;
        y = ggml_add(g, y, b);
    }
    return y;
}

static bool pp_graph_build(ppocrv6_ocr_context * c) {
    if (c->graph.attempted) return c->graph.ready;
    c->graph.attempted = true;
    if (!std::getenv("CRISPEMBED_PPOCRV6_GRAPH")) return false;
    c->graph.backend = c->backend;
    if (!c->graph.backend) return false;
    if (ggml_backend_is_cpu(c->graph.backend)) {
        ggml_backend_t backends[] = { c->graph.backend };
        c->graph.sched = ggml_backend_sched_new(backends, nullptr, 1, 4096, false, false);
    } else {
        c->graph.cpu_backend = ggml_backend_cpu_init();
        ggml_backend_t backends[] = { c->graph.backend, c->graph.cpu_backend };
        c->graph.sched = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, false);
    }
    if (!c->graph.sched) {
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) fprintf(stderr, "ppocrv6 graph scheduler creation failed\n");
        return false;
    }
    constexpr int H = 48, W = 320;
    const size_t meta_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false);
    c->graph.graph_meta.resize(meta_size);
    ggml_init_params ip = { meta_size, c->graph.graph_meta.data(), true };
    c->graph.graph_ctx = ggml_init(ip);
    if (!c->graph.graph_ctx) {
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) fprintf(stderr, "ppocrv6 graph context creation failed\n");
        return false;
    }
    c->graph.graph = ggml_new_graph_custom(c->graph.graph_ctx, 4096, false);
    c->graph.input = ggml_new_tensor_4d(c->graph.graph_ctx, GGML_TYPE_F32, W, H, 3, 1);
    ggml_set_name(c->graph.input, "ppocrv6_graph_input");
    ggml_set_input(c->graph.input);
    ggml_tensor * x = c->graph.input;
    if (c->large_stem) {
        // PPLCNetV4 large stem: the two even-kernel branches use explicit
        // right/bottom padding before valid convolution, then concatenate
        // with ceil-mode max-pool output along the channel axis.
        auto large_tap = [&](const char * name, ggml_tensor * value) {
            if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
                c->graph.debug_taps.push_back({ name, value });
                ggml_set_output(value);
            }
        };
        x = pp_graph_conv(c, c->graph.graph_ctx, x, c->stem[0]);
        if (!x) return false;
        x = ggml_relu(c->graph.graph_ctx, x);
        large_tap("large_stem1", x);
        ggml_tensor * padded = ggml_pad_ext(c->graph.graph_ctx, x, 0, 1, 0, 1, 0, 0, 0, 0);
        ggml_tensor * branch = pp_graph_conv(c, c->graph.graph_ctx, padded, c->stem[1]);
        if (!branch) return false;
        branch = ggml_relu(c->graph.graph_ctx, branch);
        large_tap("large_stem2a", branch);
        branch = ggml_pad_ext(c->graph.graph_ctx, branch, 0, 1, 0, 1, 0, 0, 0, 0);
        branch = pp_graph_conv(c, c->graph.graph_ctx, branch, c->stem[2]);
        if (!branch) return false;
        branch = ggml_relu(c->graph.graph_ctx, branch);
        large_tap("large_stem2b", branch);
        ggml_tensor * pooled = ggml_pool_2d(c->graph.graph_ctx, padded, GGML_OP_POOL_MAX, 2, 2, 1, 1, 0, 0);
        x = ggml_concat(c->graph.graph_ctx, pooled, branch, 2);
        large_tap("large_cat", x);
        x = pp_graph_conv(c, c->graph.graph_ctx, x, c->stem[3]);
        if (!x) return false;
        x = ggml_relu(c->graph.graph_ctx, x);
        large_tap("large_stem3", x);
        x = pp_graph_conv(c, c->graph.graph_ctx, x, c->stem[4]);
        if (!x) return false;
        x = ggml_relu(c->graph.graph_ctx, x);
        large_tap("large_stem", x);
    } else {
        for (size_t stem_idx = 0; stem_idx < c->stem.size(); ++stem_idx) {
            const pp_conv & p = c->stem[stem_idx];
            x = pp_graph_conv(c, c->graph.graph_ctx, x, p);
            if (!x) {
                if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) fprintf(stderr, "ppocrv6 graph stem failed\n");
                return false;
            }
            if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
                const char * pre_name = stem_idx == 0 ? "stem1_pre" : "stem2_pre";
                c->graph.debug_taps.push_back({ pre_name, x });
                ggml_set_output(x);
            }
            // PP-OCRv6 applies GELU after the first stem convolution only; the
            // second stem feeds the first backbone block directly.
            if (stem_idx == 0) x = ggml_gelu(c->graph.graph_ctx, x);
            if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
                const char * name = stem_idx == 0 ? "stem1" : "stem2";
                c->graph.debug_taps.push_back({ name, x });
                ggml_set_output(x);
            }
        }
    }
    for (size_t stage_idx = 0; stage_idx < c->stages.size(); ++stage_idx) {
        const auto & stage = c->stages[stage_idx];
        for (const pp_block & b : stage) {
            x = pp_graph_block(c, c->graph.graph_ctx, x, b);
            if (!x) {
                if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) fprintf(stderr, "ppocrv6 graph block failed\n");
                return false;
            }
        }
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
            static const char * names[] = { "stage1", "stage2", "stage3", "stage4" };
            if (stage_idx < sizeof(names) / sizeof(names[0])) {
                c->graph.debug_taps.push_back({ names[stage_idx], x });
                ggml_set_output(x);
            }
        }
    }
    if (c->large_stem && std::getenv("CRISPEMBED_PPOCRV6_SVTR_GRAPH")) {
        // Keep the SVTR attention/MLP decoder on the scalar reference path,
        // but move its convolutional tokenization onto the persistent graph.
        // The output is [hidden,tokens] with token-major contiguous storage.
        const std::string p = "rec.head.encoder.conv_block.";
        pp_conv c0 = conv(c->wl.tensors, p + "0.conv", c->stages.back().back().cm2.out_ch, c->hidden, 1, 1);
        pp_conv c1 = conv(c->wl.tensors, p + "1.conv", c->stages.back().back().cm2.out_ch, c->hidden, 1, 1);
        pp_conv c2 = conv(c->wl.tensors, p + "2.conv", c->hidden, c->hidden, 1, 1, c->hidden);
        c2.kh = 1;
        c2.kw = 7;
        c2.pad_w = 3;
        ggml_tensor * pooled = ggml_pool_2d(c->graph.graph_ctx, x, GGML_OP_POOL_AVG, 2, 3, 2, 3, 0, 0);
        ggml_tensor * residual = pp_graph_conv(c, c->graph.graph_ctx, pooled, c0);
        ggml_tensor * branch = pp_graph_conv(c, c->graph.graph_ctx, pooled, c1);
        if (!residual || !branch) return false;
        residual = ggml_silu(c->graph.graph_ctx, residual);
        branch = ggml_silu(c->graph.graph_ctx, branch);
        branch = pp_graph_conv(c, c->graph.graph_ctx, branch, c2);
        if (!branch) return false;
        branch = ggml_silu(c->graph.graph_ctx, branch);
        x = ggml_add(c->graph.graph_ctx, residual, branch);
        x = ggml_cont(c->graph.graph_ctx, ggml_permute(c->graph.graph_ctx, x, 2, 0, 1, 3));
        // On ggml's convolution layout the singleton spatial dimension can
        // remain as the fastest axis after the permute. The bytes are already
        // token-major; expose them as the decoder's [hidden,tokens] matrix.
        const int64_t tokens = x->ne[2];
        x = ggml_reshape_2d(c->graph.graph_ctx, x, c->hidden, tokens);
        c->graph.svtr_prefix_output = true;
    }
    if (c->large_stem && c->graph.svtr_prefix_output && std::getenv("CRISPEMBED_PPOCRV6_SVTR_DECODER_GRAPH")) {
        const int heads = 8;
        const int head_dim = c->hidden / heads;
        const int tokens = (int)x->ne[1];
        for (const auto & b : c->svtr) {
            ggml_tensor * residual = x;
            x = pp_graph_layernorm(c, c->graph.graph_ctx, x, b.ln1_w, b.ln1_b, c->hidden);
            if (!x) return false;
            ggml_tensor * qkv = pp_graph_linear(c, c->graph.graph_ctx, x, b.qkv_w, b.qkv_b);
            if (!qkv) return false;
            const size_t slice = (size_t)c->hidden * ggml_type_size(qkv->type);
            ggml_tensor * q =
                ggml_cont(c->graph.graph_ctx, ggml_view_2d(c->graph.graph_ctx, qkv, c->hidden, tokens, qkv->nb[1], 0));
            ggml_tensor * k = ggml_cont(c->graph.graph_ctx,
                                        ggml_view_2d(c->graph.graph_ctx, qkv, c->hidden, tokens, qkv->nb[1], slice));
            ggml_tensor * v = ggml_cont(
                c->graph.graph_ctx, ggml_view_2d(c->graph.graph_ctx, qkv, c->hidden, tokens, qkv->nb[1], 2 * slice));
            q = ggml_permute(c->graph.graph_ctx, ggml_reshape_3d(c->graph.graph_ctx, q, head_dim, heads, tokens), 0, 2,
                             1, 3);
            k = ggml_permute(c->graph.graph_ctx, ggml_reshape_3d(c->graph.graph_ctx, k, head_dim, heads, tokens), 0, 2,
                             1, 3);
            v = ggml_permute(c->graph.graph_ctx, ggml_reshape_3d(c->graph.graph_ctx, v, head_dim, heads, tokens), 0, 2,
                             1, 3);
            const float scale = 1.0f / std::sqrt((float)head_dim);
            ggml_tensor * scores =
                ggml_mul_mat(c->graph.graph_ctx, ggml_cont(c->graph.graph_ctx, k), ggml_cont(c->graph.graph_ctx, q));
            scores = ggml_soft_max_ext(c->graph.graph_ctx, scores, nullptr, scale, 0.0f);
            ggml_tensor * vt = ggml_cont(c->graph.graph_ctx, ggml_permute(c->graph.graph_ctx, v, 1, 0, 2, 3));
            ggml_tensor * attn = ggml_mul_mat(c->graph.graph_ctx, vt, scores);
            attn = ggml_cont(c->graph.graph_ctx, ggml_permute(c->graph.graph_ctx, attn, 0, 2, 1, 3));
            attn = ggml_reshape_2d(c->graph.graph_ctx, attn, c->hidden, tokens);
            attn = pp_graph_linear(c, c->graph.graph_ctx, attn, b.proj_w, b.proj_b);
            if (!attn) return false;
            x = ggml_add(c->graph.graph_ctx, residual, attn);
            residual = x;
            x = pp_graph_layernorm(c, c->graph.graph_ctx, x, b.ln2_w, b.ln2_b, c->hidden);
            if (!x) return false;
            x = pp_graph_linear(c, c->graph.graph_ctx, x, b.fc1_w, b.fc1_b);
            if (!x) return false;
            x = ggml_silu(c->graph.graph_ctx, x);
            x = pp_graph_linear(c, c->graph.graph_ctx, x, b.fc2_w, b.fc2_b);
            if (!x) return false;
            x = ggml_add(c->graph.graph_ctx, residual, x);
        }
        x = pp_graph_layernorm(c, c->graph.graph_ctx, x, c->svtr_norm_w, c->svtr_norm_b, c->hidden);
        if (!x) return false;
        c->graph.svtr_decoder_output = true;
    }
    // Complete the tiny/small recognizer graph through logits.  The large
    // SVTR decoder remains on its established CPU path for now.  Keeping the
    // logits as the graph output also avoids copying the backbone feature map
    // back to CPU merely to run two small projection layers.
    if (!c->large_stem) {
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
            c->graph.debug_taps.push_back({ "backbone", x });
            ggml_set_output(x);
        }
        // ggml addresses image dimensions as [W,H], while the reference
        // recognizer pools [H,W] with a 3x2 kernel and 3x2 stride. Swap the
        // arguments here so graph pooling matches the CPU sequence contract.
        x = ggml_pool_2d(c->graph.graph_ctx, x, GGML_OP_POOL_AVG, 2, 3, 2, 3, 0, 0);
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
            c->graph.debug_taps.push_back({ "pool", x });
            ggml_set_output(x);
        }
        pp_conv graph_head_dw = c->head_dw;
        // The CPU reference materializes two zero columns on both sides and
        // then performs a valid 1x5 convolution; express that equivalently as
        // explicit graph padding.
        graph_head_dw.pad_w = 2;
        x = pp_graph_conv(c, c->graph.graph_ctx, x, graph_head_dw);
        if (!x) {
            if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) fprintf(stderr, "ppocrv6 graph head failed\n");
            return false;
        }
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
            c->graph.debug_taps.push_back({ "head_conv1", x });
            ggml_set_output(x);
        }
        x = pp_graph_bn(c, c->graph.graph_ctx, x, c->norm1_w, c->norm1_b, c->norm1_mean, c->norm1_var,
                        c->head_dw.out_ch);
        x = ggml_hardswish(c->graph.graph_ctx, x);
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
            c->graph.debug_taps.push_back({ "head_act1", x });
            ggml_set_output(x);
        }
        x = pp_graph_conv(c, c->graph.graph_ctx, x, c->head_pw);
        if (!x) return false;
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
            c->graph.debug_taps.push_back({ "head_conv2", x });
            ggml_set_output(x);
        }
        x = pp_graph_bn(c, c->graph.graph_ctx, x, c->norm2_w, c->norm2_b, c->norm2_mean, c->norm2_var,
                        c->head_pw.out_ch);
        x = ggml_hardswish(c->graph.graph_ctx, x);
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
            c->graph.debug_taps.push_back({ "head_act2", x });
            ggml_set_output(x);
        }
        x = ggml_cont(c->graph.graph_ctx, ggml_permute(c->graph.graph_ctx, x, 2, 0, 1, 3));
        // The asymmetric 1x5 head changes the spatial width; derive the token
        // count from the actual element count instead of the pre-padding view.
        const int64_t tokens = ggml_nelements(x) / c->head_pw.out_ch;
        x = ggml_reshape_2d(c->graph.graph_ctx, x, c->head_pw.out_ch, tokens);
        x = pp_graph_linear(c, c->graph.graph_ctx, x, c->fc1_w, c->fc1_b);
        if (!x) return false;
        // The tiny recognizer head is two linear projections with no
        // activation between them (the reference decoder is F.linear twice).
        x = pp_graph_linear(c, c->graph.graph_ctx, x, c->fc2_w, c->fc2_b);
        if (!x) return false;
        c->graph.logits_output = true;
    }
    c->graph.output = x;
    ggml_set_name(x, "ppocrv6_graph_output");
    ggml_set_output(x);
    ggml_build_forward_expand(c->graph.graph, x);
    ggml_backend_sched_reset(c->graph.sched);
    if (!ggml_backend_sched_alloc_graph(c->graph.sched, c->graph.graph)) {
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) fprintf(stderr, "ppocrv6 graph allocation failed\n");
        return false;
    }
    c->graph.allocated = true;
    c->graph.ready = true;
    const char * output_kind =
        c->graph.logits_output ? "logits" : (c->graph.svtr_decoder_output ? "svtr-decoder" : "backbone");
    fprintf(stderr, "ppocrv6: persistent GGML graph ready (%s, %s, %lldx%lldx%lld)\n",
            ggml_backend_name(c->graph.backend), output_kind, (long long)x->ne[0], (long long)x->ne[1],
            (long long)x->ne[2]);
    return true;
}

static bool pp_graph_run(ppocrv6_ocr_context * c, const std::vector<float> & input, std::vector<float> & output,
                         int & h, int & w) {
    // The accepted graph is deliberately static at the Paddle minimum width.
    // Line recognition now preserves wider crops so CTC has enough timesteps;
    // never upload a dynamic-width tensor into this 320-wide graph. The CPU
    // reference path below handles those crops correctly until a dynamic
    // graph is implemented.
    constexpr size_t static_input = 3u * 48u * 320u;
    if (input.size() != static_input) {
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG"))
            fprintf(stderr, "ppocrv6: static graph bypassed for input elements=%zu (requires %zu)\n", input.size(),
                    static_input);
        return false;
    }
    if (!pp_graph_build(c)) {
        if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_BENCH"))
            fprintf(stderr, "[ppocrv6-graph-bench] graph unavailable large_stem=%d backend=%s\n", c->large_stem ? 1 : 0,
                    c->backend ? ggml_backend_name(c->backend) : "none");
        return false;
    }
    // resize_normalize and the scalar reference use channel-plane storage
    // [C,H,W].  A contiguous ggml tensor shaped [W,H,C,N] has the same byte
    // order: dimension 0 (x), then dimension 1 (y), then channels.  No
    // pixel-interleaving transpose is needed at this backend boundary.
    constexpr int H = 48, W = 320, C = 3;
    std::vector<float> graph_input = input;
    ggml_backend_tensor_set(c->graph.input, graph_input.data(), 0, graph_input.size() * sizeof(float));
    if (c->diff && std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
        std::vector<float> staged(graph_input.size());
        ggml_backend_tensor_get(c->graph.input, staged.data(), 0, staged.size() * sizeof(float));
        const auto report = c->diff->compare("ppocrv6.input", staged.data(), staged.size(), -1);
        fprintf(stderr, "[ppocrv6-graph-diff] staged-input cos=%.6f global=%.6f %s\n", report.cos_min,
                report.cos_global, report.is_pass() ? "PASS" : "FAIL");
    }
    // Reuse of mixed Metal/CPU scheduler allocations is unstable on the
    // current pre-tensor Apple backend across repeated line crops. Rebuild
    // the static allocation per crop until backend reuse is validated.
    // This is a static-shape graph, so retain the scheduler allocation between
    // crops. If a future dynamic-shape path invalidates it, it must clear
    // `allocated` before reaching this function.
    if (!c->graph.allocated) {
        ggml_backend_sched_reset(c->graph.sched);
        if (!ggml_backend_sched_alloc_graph(c->graph.sched, c->graph.graph)) return false;
        c->graph.allocated = true;
    }
    const auto started = std::chrono::steady_clock::now();
    if (ggml_backend_sched_graph_compute(c->graph.sched, c->graph.graph) != GGML_STATUS_SUCCESS) return false;
    const auto finished = std::chrono::steady_clock::now();
    output.resize(ggml_nelements(c->graph.output));
    ggml_backend_tensor_get(c->graph.output, output.data(), 0, output.size() * sizeof(float));
    if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
        for (const auto & tap : c->graph.debug_taps) {
            std::vector<float> values(ggml_nelements(tap.second));
            ggml_backend_tensor_get(tap.second, values.data(), 0, values.size() * sizeof(float));
            fprintf(stderr, "[ppocrv6-graph-tap] %s shape=%lldx%lldx%lldx%lld first=%.7g %.7g %.7g %.7g\n", tap.first,
                    (long long)tap.second->ne[0], (long long)tap.second->ne[1], (long long)tap.second->ne[2],
                    (long long)tap.second->ne[3], values.size() > 0 ? values[0] : 0.0f,
                    values.size() > 1 ? values[1] : 0.0f, values.size() > 2 ? values[2] : 0.0f,
                    values.size() > 3 ? values[3] : 0.0f);
            if (c->diff && tap.second->ne[2] > 1) {
                const int64_t tw = tap.second->ne[0], th = tap.second->ne[1], tc = tap.second->ne[2];
                const char * ref_name = nullptr;
                if (std::strcmp(tap.first, "stem1") == 0) ref_name = "ppocrv6.stem1";
                if (std::strcmp(tap.first, "stem1_pre") == 0) ref_name = "ppocrv6.stem1_pre";
                if (std::strcmp(tap.first, "stem2") == 0) ref_name = "ppocrv6.stem2";
                if (std::strcmp(tap.first, "stage1") == 0) ref_name = "ppocrv6.stage1";
                if (std::strcmp(tap.first, "stage2") == 0) ref_name = "ppocrv6.stage2";
                if (std::strcmp(tap.first, "stage3") == 0) ref_name = "ppocrv6.stage3";
                if (std::strcmp(tap.first, "stage4") == 0) ref_name = "ppocrv6.stage4";
                if (std::strcmp(tap.first, "backbone") == 0) ref_name = "ppocrv6.stage4";
                if (std::strcmp(tap.first, "head_act1") == 0) ref_name = "ppocrv6.head_conv1";
                if (std::strcmp(tap.first, "head_act2") == 0) ref_name = "ppocrv6.head_input";
                if (std::strcmp(tap.first, "large_stem1") == 0) ref_name = "ppocrv6.large_stem1";
                if (std::strcmp(tap.first, "large_stem2a") == 0) ref_name = "ppocrv6.large_stem2a";
                if (std::strcmp(tap.first, "large_stem2b") == 0) ref_name = "ppocrv6.large_stem2b";
                if (std::strcmp(tap.first, "large_cat") == 0) ref_name = "ppocrv6.large_cat";
                if (std::strcmp(tap.first, "large_stem3") == 0) ref_name = "ppocrv6.large_stem3";
                if (std::strcmp(tap.first, "large_stem") == 0) ref_name = "ppocrv6.large_stem";
                if (ref_name) {
                    std::vector<float> token_major;
                    const float * compare_data = values.data();
                    if (std::strcmp(tap.first, "head_act2") == 0) {
                        token_major.resize(values.size());
                        for (int64_t t = 0; t < tw; ++t)
                            for (int64_t ch = 0; ch < tc; ++ch)
                                token_major[(size_t)t * tc + ch] = values[(size_t)ch * tw + t];
                        compare_data = token_major.data();
                    }
                    if (std::strcmp(tap.first, "backbone") == 0) {
                        const auto raw_report = c->diff->compare(ref_name, compare_data, values.size(), -1);
                        fprintf(stderr, "[ppocrv6-graph-diff] %s raw-as-%s cos=%.6f global=%.6f %s\n", tap.first,
                                ref_name, raw_report.cos_min, raw_report.cos_global,
                                raw_report.is_pass() ? "PASS" : "FAIL");
                    }
                    const auto report = c->diff->compare(ref_name, compare_data, values.size(), -1);
                    fprintf(stderr, "[ppocrv6-graph-diff] %s as %s cos=%.6f global=%.6f %s\n", tap.first, ref_name,
                            report.cos_min, report.cos_global, report.is_pass() ? "PASS" : "FAIL");
                    if (std::strcmp(tap.first, "stem1_pre") == 0 || std::strcmp(tap.first, "stem1") == 0 ||
                        std::strcmp(tap.first, "stem2") == 0) {
                        const auto ref = c->diff->get_f32(ref_name);
                        fprintf(stderr, "[ppocrv6-graph-debug] %s graph=%.7g %.7g %.7g %.7g ref=%.7g %.7g %.7g %.7g\n",
                                tap.first, values.size() > 0 ? values[0] : 0.0f, values.size() > 1 ? values[1] : 0.0f,
                                values.size() > 2 ? values[2] : 0.0f, values.size() > 3 ? values[3] : 0.0f,
                                ref.second > 0 ? ref.first[0] : 0.0f, ref.second > 1 ? ref.first[1] : 0.0f,
                                ref.second > 2 ? ref.first[2] : 0.0f, ref.second > 3 ? ref.first[3] : 0.0f);
                    }
                }
            }
        }
    }
    w = (int)c->graph.output->ne[0];
    h = (int)c->graph.output->ne[1];
    if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_BENCH")) {
        const double ms = std::chrono::duration<double, std::milli>(finished - started).count();
        fprintf(stderr, "[ppocrv6-graph-bench] backend=%s graph_ms=%.3f output=%dx%d\n",
                ggml_backend_name(c->graph.backend), ms, w, h);
    }
    return true;
}

// Returns the padded input width, which is NOT fixed. PaddleOCR
// tools/infer/predict_rec.py seeds max_wh_ratio with imgW/imgH (320/48) and
// then grows it to the widest crop in the batch, taking
// imgW = int(imgH * max_wh_ratio). So 320 is a floor: a 520x35 line becomes
// 713 px, not a squeeze. Capping at 320 crushed a 44-character line into 40
// CTC timesteps, which no recognizer can decode. CRISPEMBED_PPOCRV6_FIXED_WIDTH
// restores the old cap for bisection.
static int resize_normalize(const uint8_t * px, int w, int h, int ch, std::vector<float> & out) {
    constexpr int H = 48, W_MIN = 320;
    const bool fixed = std::getenv("CRISPEMBED_PPOCRV6_FIXED_WIDTH") != nullptr;
    const float ratio = w / float(std::max(1, h));
    const int W = fixed ? W_MIN : std::max(W_MIN, int(H * std::max(W_MIN / float(H), ratio)));
    out.assign(3 * H * W, 0.0f);
    // The shipped Paddle inference contract is BGR, while the HF processor
    // advertises RGB conversion. Keep BGR as the production default (it is
    // the contract used by our official-source gold archives), but expose an
    // explicit diagnostic switch so the two source formats can be compared
    // without rebuilding the runtime.
    const bool rgb = std::getenv("CRISPEMBED_PPOCRV6_RGB") != nullptr;
    const int rw = std::max(1, std::min(W, int(std::round(w * (H / float(std::max(1, h)))))));
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < rw; ++x) {
            // Match Paddle/PIL's pixel-center resize used by the reference
            // dumper. Endpoint-aligned interpolation produces visible drift
            // on narrow text crops.
            const float fy = std::max(0.0f, (y + 0.5f) * h / float(H) - 0.5f);
            const float fx = std::max(0.0f, (x + 0.5f) * w / float(rw) - 0.5f);
            const int y0 = std::clamp((int)std::floor(fy), 0, h - 1), y1 = std::clamp(y0 + 1, 0, h - 1);
            const int x0 = std::clamp((int)std::floor(fx), 0, w - 1), x1 = std::clamp(x0 + 1, 0, w - 1);
            const float wy = std::clamp(fy - std::floor(fy), 0.0f, 1.0f);
            const float wx = std::clamp(fx - std::floor(fx), 0.0f, 1.0f);
            for (int c = 0; c < 3; ++c) {
                // stbi_load delivers RGB. The normal path swaps at the model
                // boundary for Paddle's DecodeImage(img_mode=BGR); the
                // diagnostic RGB path preserves the public channel order.
                int sc = ch == 1 ? 0 : (ch >= 3 && !rgb ? (2 - c) : std::min(c, ch - 1));
                const float a = px[(y0 * w + x0) * ch + sc] * (1 - wx) + px[(y0 * w + x1) * ch + sc] * wx;
                const float b = px[(y1 * w + x0) * ch + sc] * (1 - wx) + px[(y1 * w + x1) * ch + sc] * wx;
                out[c * H * W + y * W + x] = ((a * (1 - wy) + b * wy) / 255.0f - 0.5f) / 0.5f;
            }
        }
    return W;
}

static const char * recognize_svtr(ppocrv6_ocr_context * c, std::vector<float> & x, int h, int w, int * out_len,
                                   bool prefix_encoded = false, bool decoder_encoded = false) {
    const int in_ch = c->stages.back().back().cm2.out_ch;
    int ow = 0;
    std::vector<float> tok;
    // skip_conv output, held in token layout until after the SVTR blocks.
    std::vector<float> skip_tokens;
    if (prefix_encoded || decoder_encoded) {
        // pp_graph_run exposes [hidden,tokens] as w=hidden,h=tokens.
        ow = h;
        tok = x;
    } else {
        const int ph = std::max(1, h / 3), pw = std::max(1, w / 2);
        std::vector<float> pooled((size_t)in_ch * ph * pw, 0.0f);
        for (int cc = 0; cc < in_ch; ++cc)
            for (int yy = 0; yy < ph; ++yy)
                for (int xx = 0; xx < pw; ++xx) {
                    float sum = 0.0f;
                    for (int ky = 0; ky < 3; ++ky)
                        for (int kx = 0; kx < 2; ++kx) sum += x[(size_t)cc * h * w + (yy * 3 + ky) * w + xx * 2 + kx];
                    pooled[(size_t)cc * ph * pw + yy * pw + xx] = sum / 6.0f;
                }
        int oh, ow_full;
        std::vector<float> y, residual;
        const std::string p = "rec.head.encoder.conv_block.";
        pp_conv c0 = conv(c->wl.tensors, p + "0.conv", in_ch, c->hidden, 1, 1);
        pp_conv c1 = conv(c->wl.tensors, p + "1.conv", in_ch, c->hidden, 1, 1);
        pp_conv c2 = conv(c->wl.tensors, p + "2.conv", c->hidden, c->hidden, 1, 1, c->hidden);
        c2.kh = 1;
        c2.kw = 7;
        c2.pad_h = 0;
        c2.pad_w = 3;
        // PaddleOCR ppocr/modeling/necks/rnn.py, EncoderWithLightSVTR::forward:
        //     skip = skip_conv(x); z = conv_reduce(x); z = z + local_conv(z)
        //     ... svtr blocks ...; z = norm(z); z = z + skip
        // conv_block.0 is skip_conv and lands AFTER the blocks and the final
        // norm, not here; the local [1,7] conv is a residual on conv_reduce.
        if (!apply_conv(c0, pooled, ph, pw, residual, oh, ow_full)) return nullptr;
        silu(residual);
        if (!apply_conv(c1, pooled, ph, pw, y, oh, ow_full)) return nullptr;
        silu(y);
        std::vector<float> z;
        if (!apply_conv(c2, y, ph, pw, z, oh, ow_full)) return nullptr;
        silu(z);
        for (size_t i = 0; i < z.size() && i < y.size(); ++i) z[i] += y[i];
        ow = ow_full;
        tok.resize((size_t)ow * c->hidden);
        skip_tokens.resize((size_t)ow * c->hidden);
        for (int t = 0; t < ow; ++t)
            for (int cc = 0; cc < c->hidden; ++cc) {
                tok[(size_t)t * c->hidden + cc] = z[(size_t)cc * oh * ow + t];
                skip_tokens[(size_t)t * c->hidden + cc] = residual[(size_t)cc * oh * ow + t];
            }
    }
    if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG"))
        fprintf(stderr, "[ppocrv6-svtr-prefix] mode=%s tokens=%d hidden=%d first=%.7g %.7g %.7g %.7g\n",
                prefix_encoded ? "graph" : "cpu", ow, c->hidden, tok.size() > 0 ? tok[0] : 0.0f,
                tok.size() > 1 ? tok[1] : 0.0f, tok.size() > 2 ? tok[2] : 0.0f, tok.size() > 3 ? tok[3] : 0.0f);
    const int heads = 8, hd = c->hidden / heads;
    if (!decoder_encoded)
        for (const auto & b : c->svtr) {
            std::vector<float> a = tok, qkv, attn((size_t)ow * c->hidden, 0.0f), proj, mlp;
            for (int t = 0; t < ow; ++t) {
                std::vector<float> one(tok.begin() + (size_t)t * c->hidden, tok.begin() + (size_t)(t + 1) * c->hidden);
                layernorm_tokens(one, 1, c->hidden, b.ln1_wf, b.ln1_bf);
                std::vector<float> qrow;
                linear_vec(one, qrow, b.qkv_wf, b.qkv_bf);
                qkv.insert(qkv.end(), qrow.begin(), qrow.end());
            }
            for (int head = 0; head < heads; ++head)
                for (int t = 0; t < ow; ++t) {
                    std::vector<float> scores(ow);
                    for (int u = 0; u < ow; ++u) {
                        float s = 0.0f;
                        for (int k = 0; k < hd; ++k)
                            s += qkv[(size_t)t * 3 * c->hidden + head * hd + k] *
                                 qkv[(size_t)u * 3 * c->hidden + c->hidden + head * hd + k];
                        scores[u] = s / std::sqrt(float(hd));
                    }
                    softmax(scores.data(), ow);
                    for (int u = 0; u < ow; ++u)
                        for (int k = 0; k < hd; ++k)
                            attn[(size_t)t * c->hidden + head * hd + k] +=
                                scores[u] * qkv[(size_t)u * 3 * c->hidden + 2 * c->hidden + head * hd + k];
                }
            for (int t = 0; t < ow; ++t) {
                std::vector<float> one(attn.begin() + (size_t)t * c->hidden,
                                       attn.begin() + (size_t)(t + 1) * c->hidden),
                    out;
                linear_vec(one, out, b.proj_wf, b.proj_bf);
                for (int k = 0; k < c->hidden; ++k) tok[(size_t)t * c->hidden + k] += out[k];
            }
            for (int t = 0; t < ow; ++t) {
                std::vector<float> one(tok.begin() + (size_t)t * c->hidden, tok.begin() + (size_t)(t + 1) * c->hidden),
                    n, out;
                layernorm_tokens(one, 1, c->hidden, b.ln2_wf, b.ln2_bf);
                linear_vec(one, n, b.fc1_wf, b.fc1_bf);
                silu(n);
                linear_vec(n, out, b.fc2_wf, b.fc2_bf);
                for (int k = 0; k < c->hidden; ++k) tok[(size_t)t * c->hidden + k] += out[k];
            }
        }
    if (!decoder_encoded) layernorm_tokens(tok, ow, c->hidden, c->svtr_norm_wf, c->svtr_norm_bf);
    // z = z + skip, after the blocks and the neck norm (rnn.py). Only the CPU
    // neck fills skip_tokens; the graph prefix paths still fold it in-graph.
    if (!skip_tokens.empty() && skip_tokens.size() == tok.size())
        for (size_t i = 0; i < tok.size(); ++i) tok[i] += skip_tokens[i];
    if (c->diff) {
        auto r = c->diff->compare("ppocrv6.head_input", tok.data(), tok.size(), -1);
        fprintf(stderr, "[ppocrv6-diff] ppocrv6.head_input cos=%.6f |mine|=%.6g %s\n", r.cos_min,
                std::sqrt(std::inner_product(tok.begin(), tok.end(), tok.begin(), 0.0)), r.is_pass() ? "PASS" : "FAIL");
    }
    c->result.clear();
    int last = -1;
    std::vector<float> logits((size_t)ow * c->vocab_size);
    for (int t = 0; t < ow; ++t) {
        linear_cpu(tok.data() + (size_t)t * c->hidden, logits.data() + (size_t)t * c->vocab_size, c->hidden,
                   c->vocab_size, c->svtr_head_wf.data(), c->svtr_head_bf.data());
        int best = int(std::max_element(logits.begin() + (size_t)t * c->vocab_size,
                                        logits.begin() + (size_t)(t + 1) * c->vocab_size) -
                       (logits.begin() + (size_t)t * c->vocab_size));
        if (best > 0 && best != last && best - 1 < (int)c->vocab.size()) c->result += c->vocab[best - 1];
        last = best;
    }
    if (c->diff) {
        auto r = c->diff->compare("ppocrv6.logits", logits.data(), logits.size(), -1);
        fprintf(stderr, "[ppocrv6-diff] logits cos=%.6f |mine|=%.6g %s\n", r.cos_min,
                std::sqrt(std::inner_product(logits.begin(), logits.end(), logits.begin(), 0.0)),
                r.is_pass() ? "PASS" : "FAIL");
    }
    if (out_len) *out_len = (int)c->result.size();
    return c->result.c_str();
}

static const char * recognize_nchw(ppocrv6_ocr_context * c, const std::vector<float> & input, int * out_len,
                                   int input_w) {
    std::vector<float> x = input, y;
    int h = 48, w = input_w;
    std::vector<float> graph_out;
    bool graph_done = pp_graph_run(c, input, graph_out, h, w);
    if (graph_done && c->graph.logits_output && std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG")) {
        const int tokens = h;
        const int classes = w;
        fprintf(stderr, "[ppocrv6-graph-decode] tokens=%d classes=%d vocab=%zu\n", tokens, classes, c->vocab.size());
        for (int t = 0; t < std::min(tokens, 4); ++t) {
            const float * row = graph_out.data() + (size_t)t * classes;
            const int best = int(std::max_element(row, row + classes) - row);
            fprintf(stderr, "[ppocrv6-graph-decode] t=%d best=%d value=%.7g blank=%.7g\n", t, best, row[best], row[0]);
        }
    }
    if (graph_done && c->graph.logits_output && !std::getenv("CRISPEMBED_PPOCRV6_GRAPH_ACCEPT")) {
        fprintf(stderr, "ppocrv6: recognizer graph is diagnostic-only; using CPU reference\n");
        graph_done = false;
        h = 48;
        w = input_w;
    }
    if (graph_done) {
        if (c->graph.logits_output) {
            const int tokens = h;
            const int classes = w;
            c->result.clear();
            int last = -1;
            for (int t = 0; t < tokens; ++t) {
                const float * row = graph_out.data() + (size_t)t * classes;
                const int best = int(std::max_element(row, row + classes) - row);
                if (best > 0 && best != last && best - 1 < (int)c->vocab.size()) c->result += c->vocab[best - 1];
                last = best;
            }
            if (out_len) *out_len = (int)c->result.size();
            return c->result.c_str();
        }
        if (graph_done) {
            x.swap(graph_out);
            if (c->diff && !c->graph.svtr_prefix_output) {
                // The compact reference archives predate the explicit
                // graph_backbone alias; stage4 is the same tensor at this
                // seam. Prefer the alias when a future archive provides it.
                const char * ref_name =
                    c->diff->has("ppocrv6.graph_backbone") ? "ppocrv6.graph_backbone" : "ppocrv6.stage4";
                auto r = c->diff->compare(ref_name, x.data(), x.size(), -1);
                fprintf(stderr, "[ppocrv6-diff] graph_backbone as %s cos=%.6f %s\n", ref_name, r.cos_min,
                        r.is_pass() ? "PASS" : "FAIL");
            }
        }
    } else if (c->large_stem) {
        int oh, ow;
        auto diff_stem = [&](const char * name, const std::vector<float> & v) {
            if (!c->diff) return;
            auto r = c->diff->compare(name, v.data(), v.size(), -1);
            fprintf(stderr, "[ppocrv6-diff] %s cos=%.6f %s\n", name, r.cos_min, r.is_pass() ? "PASS" : "FAIL");
        };
        if (!apply_conv(c->stem[0], x, h, w, y, oh, ow)) return nullptr;
        relu(y);
        x.swap(y);
        h = oh;
        w = ow;
        diff_stem("ppocrv6.large_stem1", x);
        std::vector<float> padded_stem;
        int ph, pw;
        pad_right_bottom(x, c->stem[0].out_ch, h, w, padded_stem, ph, pw);
        std::vector<float> branch;
        if (!apply_conv(c->stem[1], padded_stem, ph, pw, branch, oh, ow)) return nullptr;
        relu(branch);
        diff_stem("ppocrv6.large_stem2a", branch);
        std::vector<float> padded_branch;
        int bph, bpw;
        pad_right_bottom(branch, c->stem[1].out_ch, oh, ow, padded_branch, bph, bpw);
        if (!apply_conv(c->stem[2], padded_branch, bph, bpw, y, oh, ow)) return nullptr;
        relu(y);
        branch.swap(y);
        diff_stem("ppocrv6.large_stem2b", branch);
        std::vector<float> pooled;
        int poh, pow;
        maxpool2x2_stride1(padded_stem, c->stem[0].out_ch, ph, pw, pooled, poh, pow);
        std::vector<float> cat((size_t)(pooled.size() + branch.size()));
        std::memcpy(cat.data(), pooled.data(), pooled.size() * sizeof(float));
        std::memcpy(cat.data() + pooled.size(), branch.data(), branch.size() * sizeof(float));
        diff_stem("ppocrv6.large_cat", cat);
        if (!apply_conv(c->stem[3], cat, poh, pow, y, oh, ow)) return nullptr;
        relu(y);
        x.swap(y);
        h = oh;
        w = ow;
        diff_stem("ppocrv6.large_stem3", x);
        if (!apply_conv(c->stem[4], x, h, w, y, oh, ow)) return nullptr;
        relu(y);
        x.swap(y);
        h = oh;
        w = ow;
        if (c->diff) {
            auto r = c->diff->compare("ppocrv6.large_stem", x.data(), x.size(), -1);
            fprintf(stderr, "[ppocrv6-diff] ppocrv6.large_stem cos=%.6f |mine|=%.6g %s\n", r.cos_min,
                    std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0)), r.is_pass() ? "PASS" : "FAIL");
        }
    } else {
        for (const auto & s : c->stem) {
            int oh, ow;
            if (!apply_conv(s, x, h, w, y, oh, ow)) return nullptr;
            if (c->diff && s.w == c->stem.front().w) {
                auto r = c->diff->compare("ppocrv6.stem1_pre", y.data(), y.size(), -1);
                fprintf(stderr, "[ppocrv6-diff] stem1_pre cos=%.6f |mine|=%.6g %s\n", r.cos_min,
                        std::sqrt(std::inner_product(y.begin(), y.end(), y.begin(), 0.0)),
                        r.is_pass() ? "PASS" : "FAIL");
            }
            if (s.w == c->stem.front().w) activate(y, false);
            x.swap(y);
            h = oh;
            w = ow;
            if (c->diff) {
                const char * name = s.w == c->stem.front().w ? "ppocrv6.stem1" : "ppocrv6.stem2";
                auto r = c->diff->compare(name, x.data(), x.size(), -1);
                auto refv = c->diff->get_f32(name);
                fprintf(stderr, "[ppocrv6-diff] %s cos=%.6f |mine|=%.6g %s\n", name, r.cos_min,
                        std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0)),
                        r.is_pass() ? "PASS" : "FAIL");
            }
        }
    }
    if (!graph_done)
        for (size_t si = 0; si < c->stages.size(); ++si) {
            for (size_t bi = 0; bi < c->stages[si].size(); ++bi) {
                std::vector<float> tap_dw, tap_cm1;
                if (!run_block(c->stages[si][bi], x, h, w, si == 0 && bi == 0 ? &tap_dw : nullptr,
                               si == 0 && bi == 0 ? &tap_cm1 : nullptr))
                    return nullptr;
                if (c->diff && si == 0 && bi == 0) {
                    for (auto pair :
                         { std::pair<const char *, const std::vector<float> *>("ppocrv6.block0_dw", &tap_dw),
                           std::pair<const char *, const std::vector<float> *>("ppocrv6.block0_cm1", &tap_cm1) }) {
                        auto r = c->diff->compare(pair.first, pair.second->data(), pair.second->size(), -1);
                        fprintf(stderr, "[ppocrv6-diff] %s cos=%.6f |mine|=%.6g %s\n", pair.first, r.cos_min,
                                std::sqrt(std::inner_product(pair.second->begin(), pair.second->end(),
                                                             pair.second->begin(), 0.0)),
                                r.is_pass() ? "PASS" : "FAIL");
                    }
                }
            }
            if (c->diff) {
                std::string name = "ppocrv6.stage" + std::to_string(si + 1);
                auto r = c->diff->compare(name, x.data(), x.size(), -1);
                fprintf(stderr, "[ppocrv6-diff] %s cos=%.6f |mine|=%.6g %s\n", name.c_str(), r.cos_min,
                        std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0)),
                        r.is_pass() ? "PASS" : "FAIL");
            }
        }
    if (c->large_stem)
        return recognize_svtr(c, x, h, w, out_len, c->graph.svtr_prefix_output, c->graph.svtr_decoder_output);
    if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG"))
        fprintf(stderr, "[ppocrv6-cpu-tap] backbone shape=%dx%d first=%.7g %.7g %.7g %.7g\n", h, w,
                x.size() > 0 ? x[0] : 0.0f, x.size() > 1 ? x[1] : 0.0f, x.size() > 2 ? x[2] : 0.0f,
                x.size() > 3 ? x[3] : 0.0f);
    const int pooled_h = std::max(1, h / 3), pooled_w = std::max(1, w / 2);
    std::vector<float> pooled((size_t)c->head_dw.in_ch * pooled_h * pooled_w, 0.0f);
    for (int cc = 0; cc < c->head_dw.in_ch; ++cc)
        for (int yy = 0; yy < pooled_h; ++yy)
            for (int xx = 0; xx < pooled_w; ++xx) {
                float sum = 0.0f;
                for (int ky = 0; ky < 3; ++ky)
                    for (int kx = 0; kx < 2; ++kx) sum += x[cc * h * w + (yy * 3 + ky) * w + xx * 2 + kx];
                pooled[cc * pooled_h * pooled_w + yy * pooled_w + xx] = sum / 6.0f;
            }
    x.swap(pooled);
    h = pooled_h;
    w = pooled_w;
    if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG"))
        fprintf(stderr, "[ppocrv6-cpu-tap] pool shape=%dx%d first=%.7g %.7g %.7g %.7g\n", h, w,
                x.size() > 0 ? x[0] : 0.0f, x.size() > 1 ? x[1] : 0.0f, x.size() > 2 ? x[2] : 0.0f,
                x.size() > 3 ? x[3] : 0.0f);
    std::vector<float> padded((size_t)c->head_dw.in_ch * h * (w + 4), 0.0f);
    for (int cc = 0; cc < c->head_dw.in_ch; ++cc)
        for (int yy = 0; yy < h; ++yy)
            std::memcpy(padded.data() + cc * h * (w + 4) + yy * (w + 4) + 2, x.data() + cc * h * w + yy * w,
                        sizeof(float) * w);
    int oh, ow;
    if (!apply_conv(c->head_dw, padded, h, w + 4, y, oh, ow)) return nullptr;
    bn1d(y, c->head_dw.out_ch, oh * ow, c->norm1_w, c->norm1_b, c->norm1_mean, c->norm1_var);
    hardswish_inplace(y.data(), (int)y.size());
    x.swap(y);
    h = oh;
    w = ow;
    if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG"))
        fprintf(stderr, "[ppocrv6-cpu-tap] head_act1 shape=%dx%d first=%.7g %.7g %.7g %.7g\n", h, w,
                x.size() > 0 ? x[0] : 0.0f, x.size() > 1 ? x[1] : 0.0f, x.size() > 2 ? x[2] : 0.0f,
                x.size() > 3 ? x[3] : 0.0f);
    if (c->diff) {
        auto r = c->diff->compare("ppocrv6.head_conv1", x.data(), x.size(), -1);
        fprintf(stderr, "[ppocrv6-diff] ppocrv6.head_conv1 cos=%.6f |mine|=%.6g %s\n", r.cos_min,
                std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0)), r.is_pass() ? "PASS" : "FAIL");
    }
    if (!apply_conv(c->head_pw, x, h, w, y, oh, ow)) return nullptr;
    if (c->diff) {
        auto r = c->diff->compare("ppocrv6.head_conv2_pre", y.data(), y.size(), -1);
        fprintf(stderr, "[ppocrv6-diff] ppocrv6.head_conv2_pre cos=%.6f |mine|=%.6g %s\n", r.cos_min,
                std::sqrt(std::inner_product(y.begin(), y.end(), y.begin(), 0.0)), r.is_pass() ? "PASS" : "FAIL");
    }
    bn1d(y, c->head_pw.out_ch, oh * ow, c->norm2_w, c->norm2_b, c->norm2_mean, c->norm2_var);
    if (c->diff) {
        auto r = c->diff->compare("ppocrv6.head_norm2", y.data(), y.size(), -1);
        fprintf(stderr, "[ppocrv6-diff] ppocrv6.head_norm2 cos=%.6f |mine|=%.6g %s\n", r.cos_min,
                std::sqrt(std::inner_product(y.begin(), y.end(), y.begin(), 0.0)), r.is_pass() ? "PASS" : "FAIL");
    }
    hardswish_inplace(y.data(), (int)y.size());
    x.swap(y);
    h = oh;
    w = ow;
    if (std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG"))
        fprintf(stderr, "[ppocrv6-cpu-tap] head_act2 shape=%dx%d first=%.7g %.7g %.7g %.7g\n", h, w,
                x.size() > 0 ? x[0] : 0.0f, x.size() > 1 ? x[1] : 0.0f, x.size() > 2 ? x[2] : 0.0f,
                x.size() > 3 ? x[3] : 0.0f);
    const int pw = w;
    std::vector<float> seq((size_t)pw * c->head_dw.in_ch);
    for (int t = 0; t < pw; ++t)
        for (int cc = 0; cc < c->head_dw.in_ch; ++cc) seq[t * c->head_dw.in_ch + cc] = x[cc * h * w + t];
    if (c->diff) {
        auto r = c->diff->compare("ppocrv6.head_input", seq.data(), seq.size(), -1);
        fprintf(stderr, "[ppocrv6-diff] ppocrv6.head_input cos=%.6f |mine|=%.6g %s\n", r.cos_min,
                std::sqrt(std::inner_product(seq.begin(), seq.end(), seq.begin(), 0.0)), r.is_pass() ? "PASS" : "FAIL");
    }
    c->result.clear();
    std::vector<float> all_logits;
    all_logits.reserve((size_t)pw * c->vocab_size);
    int last = -1;
    for (int t = 0; t < pw; ++t) {
        std::vector<float> hidden(c->hidden), logits(c->vocab_size);
        linear_cpu(seq.data() + t * c->head_dw.in_ch, hidden.data(), c->head_dw.in_ch, c->hidden, c->fc1_wf.data(),
                   c->fc1_bf.data());
        linear_cpu(hidden.data(), logits.data(), c->hidden, c->vocab_size, c->fc2_wf.data(), c->fc2_bf.data());
        all_logits.insert(all_logits.end(), logits.begin(), logits.end());
        int best = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (best > 0 && best != last && best - 1 < (int)c->vocab.size()) c->result += c->vocab[best - 1];
        last = best;
    }
    if (!graph_out.empty() && c->graph.logits_output && std::getenv("CRISPEMBED_PPOCRV6_GRAPH_DEBUG") &&
        graph_out.size() == all_logits.size()) {
        double dot = 0.0, gn = 0.0, cn = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < all_logits.size(); ++i) {
            dot += double(graph_out[i]) * all_logits[i];
            gn += double(graph_out[i]) * graph_out[i];
            cn += double(all_logits[i]) * all_logits[i];
            max_abs = std::max(max_abs, std::fabs(graph_out[i] - all_logits[i]));
        }
        fprintf(stderr, "[ppocrv6-graph-decode] cpu_cos=%.7f max_abs=%.7g\n", float(dot / (std::sqrt(gn * cn) + 1e-30)),
                max_abs);
    }
    if (c->diff) {
        auto r = c->diff->compare("ppocrv6.logits", all_logits.data(), all_logits.size(), 1);
        fprintf(stderr, "[ppocrv6-diff] logits cos=%.6f |mine|=%.6g |ref|=%.6g %s\n", r.cos_min,
                std::sqrt(std::inner_product(all_logits.begin(), all_logits.end(), all_logits.begin(), 0.0)),
                std::sqrt(std::inner_product(c->diff->get_f32("ppocrv6.logits").first,
                                             c->diff->get_f32("ppocrv6.logits").first +
                                                 c->diff->get_f32("ppocrv6.logits").second,
                                             c->diff->get_f32("ppocrv6.logits").first, 0.0)),
                r.is_pass() ? "PASS" : "FAIL");
    }
    if (out_len) *out_len = (int)c->result.size();
    return c->result.c_str();
}

extern "C" ppocrv6_ocr_context * ppocrv6_ocr_init(const char * path, int) {
    auto * c = new ppocrv6_ocr_context();
    const bool force_cpu = std::getenv("CRISPEMBED_PPOCRV6_FORCE_CPU") != nullptr;
    c->backend = force_cpu ? ggml_backend_cpu_init() : crispasr_init_gpu_backend();
    if (!c->backend) c->backend = ggml_backend_cpu_init();
    gguf_context * meta = core_gguf::open_metadata(path);
    if (!meta) {
        delete c;
        return nullptr;
    }
    c->variant = core_gguf::kv_str(meta, "ppocrv6.variant", "tiny");
    c->vocab = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
    c->vocab_size = (int)core_gguf::kv_u32(meta, "ppocrv6.vocab_size", 0);
    // The PP-OCRv6 configs set use_space_char, so PaddleOCR's label list is
    // blank + the dict + ' ' -- which is where vocab_size (18710) gets its two
    // extra classes over the 18708-entry dict the converter emits. Without the
    // trailing space every inter-word class decodes to nothing and the output
    // is one run-on token.
    if (c->vocab_size == (int)c->vocab.size() + 2) c->vocab.push_back(" ");
    if (const char * ref = std::getenv("PPOCRV6_REF")) {
        c->diff = std::make_unique<crispembed_diff::Ref>();
        if (!c->diff->load(ref)) c->diff.reset();
    }
    core_gguf::free_metadata(meta);
    if (!core_gguf::load_weights(path, c->backend, "ppocrv6", c->wl)) {
        fprintf(stderr, "ppocrv6: failed to load recognizer weights: %s (variant=%s)\n", path, c->variant.c_str());
        ppocrv6_ocr_free(c);
        return nullptr;
    }
    if (!map_model(c)) {
        fprintf(stderr, "ppocrv6: recognizer tensor map is incompatible: %s (variant=%s)\n", path, c->variant.c_str());
        ppocrv6_ocr_free(c);
        return nullptr;
    }
    return c;
}

extern "C" void ppocrv6_ocr_free(ppocrv6_ocr_context * c) {
    if (!c) return;
    for (auto * buf : c->graph.resident_bufs)
        if (buf) ggml_backend_buffer_free(buf);
    for (auto * ctx : c->graph.resident_ctxs)
        if (ctx) ggml_free(ctx);
    if (c->graph.sched) ggml_backend_sched_free(c->graph.sched);
    // graph.backend is normally c->backend; it is freed below.
    if (c->graph.cpu_backend) ggml_backend_free(c->graph.cpu_backend);
    if (c->graph.graph_ctx) ggml_free(c->graph.graph_ctx);
    core_gguf::free_weights(c->wl);
    if (c->backend) ggml_backend_free(c->backend);
    delete c;
}

extern "C" const char * ppocrv6_ocr_recognize_raw(ppocrv6_ocr_context * c, const uint8_t * px, int w, int h, int ch,
                                                  int * out_len) {
    if (!c || !px || w <= 0 || h <= 0 || (ch != 1 && ch != 3 && ch != 4)) return nullptr;
    std::vector<float> input;
    const int input_w = resize_normalize(px, w, h, ch, input);
    if (c->diff) {
        auto r = c->diff->compare("ppocrv6.input", input.data(), input.size(), -1);
        fprintf(stderr, "[ppocrv6-diff] input cos=%.6f |mine|=%.6g %s\n", r.cos_min,
                std::sqrt(std::inner_product(input.begin(), input.end(), input.begin(), 0.0)),
                r.is_pass() ? "PASS" : "FAIL");
    }
    return recognize_nchw(c, input, out_len, input_w);
}

extern "C" const char * ppocrv6_ocr_recognize(ppocrv6_ocr_context * c, const float * px, int w, int h, int * out_len) {
    if (!c || !px) return nullptr;
    std::vector<uint8_t> u((size_t)w * h);
    for (size_t i = 0; i < u.size(); ++i) u[i] = (uint8_t)std::clamp(int(px[i] * 255.0f + 0.5f), 0, 255);
    return ppocrv6_ocr_recognize_raw(c, u.data(), w, h, 1, out_len);
}
