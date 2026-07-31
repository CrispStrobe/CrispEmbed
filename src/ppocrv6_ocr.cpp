#include "ppocrv6_ocr.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "core/cpu_ops.h"
#include "core/gguf_loader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
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
    int in_ch = 0, out_ch = 0, kh = 1, kw = 1, stride = 1, pad_h = 0, pad_w = 0, groups = 1;
};

struct pp_block {
    pp_conv dw, cm1, cm2, se1, se2;
    bool se = false;
    bool residual = false;
};

struct ppocrv6_ocr_context {
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    std::vector<std::string> vocab;
    std::string result;
    std::vector<float> scratch;
    std::string variant;
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
};

static ggml_tensor * get(const core_gguf::tensor_map & m, const std::string & n) {
    return core_gguf::try_get(m, n.c_str());
}

static pp_conv conv(const core_gguf::tensor_map & m, const std::string & n, int in, int out, int k, int stride,
                    int groups = 1) {
    pp_conv c;
    c.w = get(m, n + ".weight");
    c.b = get(m, n + ".bias");
    c.in_ch = in;
    c.out_ch = out;
    c.kh = c.kw = k;
    c.stride = stride;
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
    oh = (h + 2 * c.pad_h - c.kh) / c.stride + 1;
    ow = (w + 2 * c.pad_w - c.kw) / c.stride + 1;
    out.assign((size_t)c.out_ch * oh * ow, 0.0f);
    auto ww = to_f32(c.w);
    auto bb = to_f32(c.b);
    conv2d_cpu(in.data(), out.data(), ww.data(), bb.empty() ? nullptr : bb.data(), c.in_ch, c.out_ch, h, w, c.kh, c.kw,
               c.stride, c.pad_h, c.groups);
    return true;
}

static void activate(std::vector<float> & x, bool hs) {
    if (hs)
        hardswish_inplace(x.data(), (int)x.size());
    else
        for (float & v : x) v = gelu(v);
}

static bool run_block(pp_block & b, std::vector<float> & x, int & h, int & w) {
    std::vector<float> y, z;
    int oh, ow;
    if (!apply_conv(b.dw, x, h, w, y, oh, ow)) return false;
    activate(y, true);
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
    activate(z, false);
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
    c->stem.push_back(conv(m, "rec.bb.stem.conv1.conv", 3, s, 3, 2));
    c->stem.push_back(conv(m, "rec.bb.stem.conv2.conv", s, stem2, 3, 2));
    const int widths[4] = { tiny ? 48 : stem2, tiny ? 48 : stem2, tiny ? 96 : stem2 * 2,
                            tiny ? 160 : (c->variant == "small" ? 384 : 768) };
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
            b.dw = conv(m, q + ".dw", in, in, 3, (bi == 0 && (si == 2 || si == 3)) ? 2 : 1, in);
            b.cm1 = conv(m, q + ".cm1", in, in * 2, 1, 1);
            b.cm2 = conv(m, q + ".cm2", in * 2, out, 1, 1);
            b.se = get(m, q + ".se1.weight") != nullptr;
            if (b.se) {
                b.se1 = conv(m, q + ".se1", in, std::max(1, in / 4), 1, 1);
                b.se2 = conv(m, q + ".se2", std::max(1, in / 4), in, 1, 1);
            }
            b.residual = in == out && b.dw.stride == 1;
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
    return !c->stem.empty() && c->fc2_w;
}

static void resize_normalize(const uint8_t * px, int w, int h, int ch, std::vector<float> & out) {
    constexpr int H = 48, W = 320;
    out.assign(3 * H * W, 0.0f);
    const float scale = std::min(1.0f, float(W) / std::max(1, w));
    const int rw = std::max(1, std::min(W, int(std::round(w * scale))));
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < rw; ++x) {
            int sy = std::min(h - 1, int(y * h / float(H)));
            int sx = std::min(w - 1, int(x * w / float(rw)));
            for (int c = 0; c < 3; ++c) {
                int sc = ch == 1 ? 0 : std::min(c, ch - 1);
                out[c * H * W + y * W + x] = (px[(sy * w + sx) * ch + sc] / 255.0f - 0.5f) / 0.5f;
            }
        }
}

static const char * recognize_nchw(ppocrv6_ocr_context * c, const std::vector<float> & input, int * out_len) {
    std::vector<float> x = input, y;
    int h = 48, w = 320;
    for (const auto & s : c->stem) {
        int oh, ow;
        if (!apply_conv(s, x, h, w, y, oh, ow)) return nullptr;
        if (s.w == c->stem.front().w) activate(y, false);
        x.swap(y);
        h = oh;
        w = ow;
    }
    for (auto & stage : c->stages)
        for (auto & b : stage)
            if (!run_block(b, x, h, w)) return nullptr;
    int oh, ow;
    if (!apply_conv(c->head_dw, x, h, w, y, oh, ow)) return nullptr;
    hardswish_inplace(y.data(), (int)y.size());
    x.swap(y);
    h = oh;
    w = ow;
    if (!apply_conv(c->head_pw, x, h, w, y, oh, ow)) return nullptr;
    hardswish_inplace(y.data(), (int)y.size());
    x.swap(y);
    h = oh;
    w = ow;
    // Pooling is (3,2), then the tiny head's Conv1D keeps the sequence width.
    const int ph = std::max(1, h / 3), pw = std::max(1, w / 2);
    std::vector<float> seq((size_t)c->head_dw.in_ch * pw, 0.0f);
    for (int cc = 0; cc < c->head_dw.in_ch; ++cc)
        for (int xx = 0; xx < pw; ++xx)
            for (int yy = 0; yy < ph; ++yy) seq[cc * pw + xx] += x[cc * h * w + (yy * 3) * w + xx * 2];
    for (float & v : seq) v /= float(ph);
    auto f1 = to_f32(c->fc1_w), b1 = to_f32(c->fc1_b), f2 = to_f32(c->fc2_w), b2 = to_f32(c->fc2_b);
    c->result.clear();
    int last = -1;
    for (int t = 0; t < pw; ++t) {
        std::vector<float> hidden(c->hidden), logits(c->vocab_size);
        linear_cpu(seq.data() + t, hidden.data(), c->head_dw.in_ch, c->hidden, f1.data(), b1.data());
        linear_cpu(hidden.data(), logits.data(), c->hidden, c->vocab_size, f2.data(), b2.data());
        int best = int(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (best > 0 && best != last && best - 1 < (int)c->vocab.size()) c->result += c->vocab[best - 1];
        last = best;
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
    return recognize_nchw(c, input, out_len);
}

extern "C" const char * ppocrv6_ocr_recognize(ppocrv6_ocr_context * c, const float * px, int w, int h, int * out_len) {
    if (!c || !px) return nullptr;
    std::vector<uint8_t> u((size_t)w * h);
    for (size_t i = 0; i < u.size(); ++i) u[i] = (uint8_t)std::clamp(int(px[i] * 255.0f + 0.5f), 0, 255);
    return ppocrv6_ocr_recognize_raw(c, u.data(), w, h, 1, out_len);
}
