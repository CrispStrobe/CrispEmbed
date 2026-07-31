#include "ppocrv6_ocr.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/cpu_ops.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <string>
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
    ggml_tensor *norm1_w = nullptr, *norm1_b = nullptr, *norm1_mean = nullptr, *norm1_var = nullptr;
    ggml_tensor *norm2_w = nullptr, *norm2_b = nullptr, *norm2_mean = nullptr, *norm2_var = nullptr;
    ggml_tensor *svtr_norm_w = nullptr, *svtr_norm_b = nullptr;
    ggml_tensor *svtr_head_w = nullptr, *svtr_head_b = nullptr;
    std::vector<pp_svtr> svtr;
    std::unique_ptr<crispembed_diff::Ref> diff;
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
    auto ww = to_f32(c.w);
    auto bb = to_f32(c.b);
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

static void linear_vec(const std::vector<float> & x, std::vector<float> & y, ggml_tensor * w, ggml_tensor * b) {
    auto ww = to_f32(w), bb = to_f32(b);
    const int out = (int)bb.size(), in = (int)x.size();
    y.resize(out);
    linear_cpu(x.data(), y.data(), in, out, ww.data(), bb.data());
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
            const bool down = bi == 0 && (si == 2 || (!tiny && si == 3));
            b.dw = conv(m, q + ".dw", in, in, 3, down ? 2 : 1, in, down && !tiny ? 1 : 0);
            b.cm1 = conv(m, q + ".cm1", in, in * 2, 1, 1);
            b.cm2 = conv(m, q + ".cm2", in * 2, out, 1, 1);
            b.se = get(m, q + ".se1.weight") != nullptr;
            if (b.se) {
                b.se1 = conv(m, q + ".se1", in, std::max(1, in / 4), 1, 1);
                b.se2 = conv(m, q + ".se2", std::max(1, in / 4), in, 1, 1);
            }
            b.residual = in == out && b.dw.stride == 1;
            b.silu_act = !tiny;
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
            c->svtr.push_back(b);
        };
        load_svtr(0);
        load_svtr(1);
        c->svtr_norm_w = get(m, p + "norm.weight");
        c->svtr_norm_b = get(m, p + "norm.bias");
        c->svtr_head_w = get(m, "rec.head.head.weight");
        c->svtr_head_b = get(m, "rec.head.head.bias");
    }
    return !c->stem.empty() && (tiny ? c->fc2_w != nullptr : c->svtr_head_w != nullptr && c->svtr.size() == 2);
}

static void resize_normalize(const uint8_t * px, int w, int h, int ch, std::vector<float> & out) {
    constexpr int H = 48, W = 320;
    out.assign(3 * H * W, 0.0f);
    const int rw = std::max(1, std::min(W, int(std::round(w * (H / float(std::max(1, h)))))));
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < rw; ++x) {
            const float fy = y * (h - 1.0f) / std::max(1, H - 1);
            const float fx = x * (w - 1.0f) / std::max(1, rw - 1);
            const int y0 = std::clamp((int)std::floor(fy), 0, h - 1), y1 = std::clamp(y0 + 1, 0, h - 1);
            const int x0 = std::clamp((int)std::floor(fx), 0, w - 1), x1 = std::clamp(x0 + 1, 0, w - 1);
            const float wy = std::clamp(fy - std::floor(fy), 0.0f, 1.0f);
            const float wx = std::clamp(fx - std::floor(fx), 0.0f, 1.0f);
            for (int c = 0; c < 3; ++c) {
                int sc = ch == 1 ? 0 : std::min(c, ch - 1);
                const float a = px[(y0 * w + x0) * ch + sc] * (1 - wx) + px[(y0 * w + x1) * ch + sc] * wx;
                const float b = px[(y1 * w + x0) * ch + sc] * (1 - wx) + px[(y1 * w + x1) * ch + sc] * wx;
                out[c * H * W + y * W + x] = ((a * (1 - wy) + b * wy) / 255.0f - 0.5f) / 0.5f;
            }
        }
}

static const char * recognize_svtr(ppocrv6_ocr_context * c, std::vector<float> & x, int h, int w, int * out_len) {
    const int in_ch = c->stages.back().back().cm2.out_ch;
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
    int oh, ow;
    std::vector<float> y, residual;
    pp_conv c0, c1, c2;
    const std::string p = "rec.head.encoder.conv_block.";
    c0 = conv(c->wl.tensors, p + "0.conv", in_ch, c->hidden, 1, 1);
    c1 = conv(c->wl.tensors, p + "1.conv", in_ch, c->hidden, 1, 1);
    c2 = conv(c->wl.tensors, p + "2.conv", c->hidden, c->hidden, 1, 1, c->hidden);
    c2.kh = 1;
    c2.kw = 7;
    c2.pad_h = 0;
    c2.pad_w = 3;
    if (!apply_conv(c0, pooled, ph, pw, residual, oh, ow)) return nullptr;
    silu(residual);
    if (!apply_conv(c1, pooled, ph, pw, y, oh, ow)) return nullptr;
    silu(y);
    std::vector<float> z;
    if (!apply_conv(c2, y, ph, pw, z, oh, ow)) return nullptr;
    silu(z);
    for (size_t i = 0; i < z.size(); ++i) z[i] += residual[i];
    std::vector<float> tok((size_t)ow * c->hidden);
    for (int t = 0; t < ow; ++t)
        for (int cc = 0; cc < c->hidden; ++cc) tok[(size_t)t * c->hidden + cc] = z[(size_t)cc * oh * ow + t];
    const int heads = 8, hd = c->hidden / heads;
    for (const auto & b : c->svtr) {
        std::vector<float> a = tok, qkv, attn((size_t)ow * c->hidden, 0.0f), proj, mlp;
        for (int t = 0; t < ow; ++t) {
            std::vector<float> one(tok.begin() + (size_t)t * c->hidden, tok.begin() + (size_t)(t + 1) * c->hidden);
            layernorm_tokens(one, 1, c->hidden, b.ln1_w, b.ln1_b);
            std::vector<float> qrow;
            linear_vec(one, qrow, b.qkv_w, b.qkv_b);
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
            std::vector<float> one(attn.begin() + (size_t)t * c->hidden, attn.begin() + (size_t)(t + 1) * c->hidden),
                out;
            linear_vec(one, out, b.proj_w, b.proj_b);
            for (int k = 0; k < c->hidden; ++k) tok[(size_t)t * c->hidden + k] += out[k];
        }
        for (int t = 0; t < ow; ++t) {
            std::vector<float> one(tok.begin() + (size_t)t * c->hidden, tok.begin() + (size_t)(t + 1) * c->hidden), n,
                out;
            layernorm_tokens(one, 1, c->hidden, b.ln2_w, b.ln2_b);
            linear_vec(one, n, b.fc1_w, b.fc1_b);
            silu(n);
            linear_vec(n, out, b.fc2_w, b.fc2_b);
            for (int k = 0; k < c->hidden; ++k) tok[(size_t)t * c->hidden + k] += out[k];
        }
    }
    layernorm_tokens(tok, ow, c->hidden, c->svtr_norm_w, c->svtr_norm_b);
    c->result.clear();
    int last = -1;
    auto hw = to_f32(c->svtr_head_w), hb = to_f32(c->svtr_head_b);
    std::vector<float> logits((size_t)ow * c->vocab_size);
    for (int t = 0; t < ow; ++t) {
        linear_cpu(tok.data() + (size_t)t * c->hidden, logits.data() + (size_t)t * c->vocab_size, c->hidden,
                   c->vocab_size, hw.data(), hb.data());
        int best = int(std::max_element(logits.begin() + (size_t)t * c->vocab_size,
                                        logits.begin() + (size_t)(t + 1) * c->vocab_size) -
                       (logits.begin() + (size_t)t * c->vocab_size));
        if (best > 0 && best != last && best - 1 < (int)c->vocab.size()) c->result += c->vocab[best - 1];
        last = best;
    }
    if (out_len) *out_len = (int)c->result.size();
    return c->result.c_str();
}

static const char * recognize_nchw(ppocrv6_ocr_context * c, const std::vector<float> & input, int * out_len) {
    std::vector<float> x = input, y;
    int h = 48, w = 320;
    if (c->large_stem) {
        int oh, ow;
        if (!apply_conv(c->stem[0], x, h, w, y, oh, ow)) return nullptr;
        silu(y);
        x.swap(y);
        h = oh;
        w = ow;
        std::vector<float> branch;
        if (!apply_conv(c->stem[1], x, h, w, branch, oh, ow)) return nullptr;
        silu(branch);
        if (!apply_conv(c->stem[2], branch, oh, ow, y, oh, ow)) return nullptr;
        silu(y);
        branch.swap(y);
        std::vector<float> cat((size_t)(x.size() + branch.size()));
        std::memcpy(cat.data(), x.data(), x.size() * sizeof(float));
        std::memcpy(cat.data() + x.size(), branch.data(), branch.size() * sizeof(float));
        if (!apply_conv(c->stem[3], cat, h, w, y, oh, ow)) return nullptr;
        silu(y);
        x.swap(y);
        h = oh;
        w = ow;
        if (!apply_conv(c->stem[4], x, h, w, y, oh, ow)) return nullptr;
        silu(y);
        x.swap(y);
        h = oh;
        w = ow;
    } else
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
                            std::sqrt(std::inner_product(pair.second->begin(), pair.second->end(), pair.second->begin(),
                                                         0.0)),
                            r.is_pass() ? "PASS" : "FAIL");
                }
            }
        }
        if (c->diff) {
            std::string name = "ppocrv6.stage" + std::to_string(si + 1);
            auto r = c->diff->compare(name, x.data(), x.size(), -1);
            fprintf(stderr, "[ppocrv6-diff] %s cos=%.6f |mine|=%.6g %s\n", name.c_str(), r.cos_min,
                    std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0)), r.is_pass() ? "PASS" : "FAIL");
        }
    }
    if (c->large_stem) return recognize_svtr(c, x, h, w, out_len);
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
    const int pw = w;
    std::vector<float> seq((size_t)pw * c->head_dw.in_ch);
    for (int t = 0; t < pw; ++t)
        for (int cc = 0; cc < c->head_dw.in_ch; ++cc) seq[t * c->head_dw.in_ch + cc] = x[cc * h * w + t];
    if (c->diff) {
        auto r = c->diff->compare("ppocrv6.head_input", seq.data(), seq.size(), -1);
        fprintf(stderr, "[ppocrv6-diff] ppocrv6.head_input cos=%.6f |mine|=%.6g %s\n", r.cos_min,
                std::sqrt(std::inner_product(seq.begin(), seq.end(), seq.begin(), 0.0)), r.is_pass() ? "PASS" : "FAIL");
    }
    auto f1 = to_f32(c->fc1_w), b1 = to_f32(c->fc1_b), f2 = to_f32(c->fc2_w), b2 = to_f32(c->fc2_b);
    c->result.clear();
    std::vector<float> all_logits;
    all_logits.reserve((size_t)pw * c->vocab_size);
    int last = -1;
    for (int t = 0; t < pw; ++t) {
        std::vector<float> hidden(c->hidden), logits(c->vocab_size);
        linear_cpu(seq.data() + t * c->head_dw.in_ch, hidden.data(), c->head_dw.in_ch, c->hidden, f1.data(), b1.data());
        linear_cpu(hidden.data(), logits.data(), c->hidden, c->vocab_size, f2.data(), b2.data());
        all_logits.insert(all_logits.end(), logits.begin(), logits.end());
        int best = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (best > 0 && best != last && best - 1 < (int)c->vocab.size()) c->result += c->vocab[best - 1];
        last = best;
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
    c->backend = ggml_backend_cpu_init();
    gguf_context * meta = core_gguf::open_metadata(path);
    if (!meta) {
        delete c;
        return nullptr;
    }
    c->variant = core_gguf::kv_str(meta, "ppocrv6.variant", "tiny");
    c->vocab = core_gguf::kv_str_array(meta, "tokenizer.ggml.tokens");
    c->vocab_size = (int)core_gguf::kv_u32(meta, "ppocrv6.vocab_size", 0);
    if (const char * ref = std::getenv("PPOCRV6_REF")) {
        c->diff = std::make_unique<crispembed_diff::Ref>();
        if (!c->diff->load(ref)) c->diff.reset();
    }
    core_gguf::free_metadata(meta);
    if (!core_gguf::load_weights(path, c->backend, "ppocrv6", c->wl) || !map_model(c)) {
        ppocrv6_ocr_free(c);
        return nullptr;
    }
    return c;
}

extern "C" void ppocrv6_ocr_free(ppocrv6_ocr_context * c) {
    if (!c) return;
    core_gguf::free_weights(c->wl);
    if (c->backend) ggml_backend_free(c->backend);
    delete c;
}

extern "C" const char * ppocrv6_ocr_recognize_raw(ppocrv6_ocr_context * c, const uint8_t * px, int w, int h, int ch,
                                                  int * out_len) {
    if (!c || !px || w <= 0 || h <= 0 || (ch != 1 && ch != 3 && ch != 4)) return nullptr;
    std::vector<float> input;
    resize_normalize(px, w, h, ch, input);
    if (c->diff) {
        auto r = c->diff->compare("ppocrv6.input", input.data(), input.size(), -1);
        fprintf(stderr, "[ppocrv6-diff] input cos=%.6f |mine|=%.6g %s\n", r.cos_min,
                std::sqrt(std::inner_product(input.begin(), input.end(), input.begin(), 0.0)),
                r.is_pass() ? "PASS" : "FAIL");
    }
    return recognize_nchw(c, input, out_len);
}

extern "C" const char * ppocrv6_ocr_recognize(ppocrv6_ocr_context * c, const float * px, int w, int h, int * out_len) {
    if (!c || !px) return nullptr;
    std::vector<uint8_t> u((size_t)w * h);
    for (size_t i = 0; i < u.size(); ++i) u[i] = (uint8_t)std::clamp(int(px[i] * 255.0f + 0.5f), 0, 255);
    return ppocrv6_ocr_recognize_raw(c, u.data(), w, h, 1, out_len);
}
