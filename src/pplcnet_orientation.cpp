#include "pplcnet_orientation.h"

#include "core/cpu_ops.h"
#include "core/gguf_loader.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
unsigned char * stbi_load(const char *, int *, int *, int *, int);
void stbi_image_free(void *);
}

namespace pplcnet_orientation {

using core_cpu::conv2d_cpu;
using core_cpu::linear_cpu;
using core_cpu::to_f32;

struct conv {
    ggml_tensor * w = nullptr;
    ggml_tensor * b = nullptr;
    int in = 0, out = 0, k = 1, stride = 1, groups = 1;
};

struct bn {
    ggml_tensor * mean = nullptr;
    ggml_tensor * var = nullptr;
    ggml_tensor * scale = nullptr;
    ggml_tensor * bias = nullptr;
};

struct block {
    conv dw, pw;
    bn dw_bn, pw_bn;
    bool se = false;
    conv se1, se2;
    bool residual = false;
};

struct context {
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    conv stem, head;
    bn stem_bn;
    std::vector<block> blocks;
    ggml_tensor * fc_w = nullptr;
    ggml_tensor * fc_b = nullptr;
};

static ggml_tensor * prefix(const core_gguf::tensor_map & m, const std::string & p) {
    for (const auto & it : m)
        if (it.first.rfind(p, 0) == 0) return it.second;
    return nullptr;
}

static conv make_conv(const core_gguf::tensor_map & m, int id, int in, int out, int k, int stride, int groups) {
    const std::string p = "ori.conv2d_" + std::to_string(id) + ".";
    conv c;
    c.w = prefix(m, p + "w_0");
    c.b = prefix(m, p + "b_0");
    c.in = in;
    c.out = out;
    c.k = k;
    c.stride = stride;
    c.groups = groups;
    return c;
}

static bn make_bn(const core_gguf::tensor_map & m, int id) {
    const std::string p = "ori.batch_norm2d_" + std::to_string(id) + ".";
    // Paddle's BatchNorm inputs are mean, variance, scale, bias.  The PIR
    // variable names are w_1, w_2, w_0, b_0 respectively.
    return { prefix(m, p + "w_1"), prefix(m, p + "w_2"), prefix(m, p + "w_0"), prefix(m, p + "b_0") };
}

static int bn_id_for_conv(int id) {
    if (id <= 23) return id;
    if (id == 26) return 24;
    if (id == 27) return 25;
    if (id == 30) return 26;
    return -1;
}

static bool apply_conv(const conv & c, const std::vector<float> & x, int h, int w, std::vector<float> & y, int & oh,
                       int & ow) {
    if (!c.w) return false;
    oh = (h + c.k - 1 - c.k) / c.stride + 1 + (c.k - 1) / c.stride; // replaced below for clarity
    const int pad = c.k / 2;
    oh = (h + 2 * pad - c.k) / c.stride + 1;
    ow = (w + 2 * pad - c.k) / c.stride + 1;
    y.assign((size_t)c.out * oh * ow, 0.0f);
    auto weights = to_f32(c.w);
    auto bias = to_f32(c.b);
    conv2d_cpu(x.data(), y.data(), weights.data(), bias.empty() ? nullptr : bias.data(), c.in, c.out, h, w, c.k, c.k,
               c.stride, pad, c.groups);
    return true;
}

static void apply_bn(std::vector<float> & x, const bn & b, int channels) {
    if (!b.mean || !b.var || !b.scale || !b.bias) return;
    const auto mean = to_f32(b.mean), var = to_f32(b.var), scale = to_f32(b.scale), bias = to_f32(b.bias);
    const size_t spatial = x.size() / channels;
    for (int c = 0; c < channels; ++c) {
        const float inv = scale[c] / std::sqrt(var[c] + 1e-5f);
        for (size_t i = 0; i < spatial; ++i)
            x[(size_t)c * spatial + i] = (x[(size_t)c * spatial + i] - mean[c]) * inv + bias[c];
    }
}

static void hardswish(std::vector<float> & x) {
    for (float & v : x) v *= std::clamp(v + 3.0f, 0.0f, 6.0f) / 6.0f;
}

static void relu(std::vector<float> & x) {
    for (float & v : x) v = std::max(0.0f, v);
}

static bool run_conv_bn(const conv & c, const bn & b, std::vector<float> & x, int & h, int & w) {
    std::vector<float> y;
    int oh = 0, ow = 0;
    if (!apply_conv(c, x, h, w, y, oh, ow)) return false;
    apply_bn(y, b, c.out);
    h = oh;
    w = ow;
    hardswish(y);
    x.swap(y);
    return true;
}

static bool run_block(const block & b, std::vector<float> & x, int & h, int & w) {
    std::vector<float> input = x;
    if (!run_conv_bn(b.dw, b.dw_bn, x, h, w)) return false;
    if (b.se) {
        const int channels = b.dw.out;
        std::vector<float> pooled(channels, 0.0f);
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < h * w; ++i) pooled[c] += x[(size_t)c * h * w + i] / float(h * w);
        std::vector<float> gate;
        int gh = 0, gw = 0;
        if (!apply_conv(b.se1, pooled, 1, 1, gate, gh, gw)) return false;
        relu(gate);
        if (!apply_conv(b.se2, gate, 1, 1, pooled, gh, gw)) return false;
        for (float & v : pooled) v = std::clamp((v + 3.0f) / 6.0f, 0.0f, 1.0f);
        for (int c = 0; c < channels; ++c)
            for (int i = 0; i < h * w; ++i) x[(size_t)c * h * w + i] *= pooled[c];
    }
    if (!run_conv_bn(b.pw, b.pw_bn, x, h, w)) return false;
    if (b.residual && input.size() == x.size())
        for (size_t i = 0; i < x.size(); ++i) x[i] += input[i];
    return true;
}

static std::vector<float> preprocess(const uint8_t * px, int width, int height, int channels) {
    constexpr int H = 80, W = 160;
    std::vector<float> out((size_t)3 * H * W);
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int sx = std::min(width - 1, std::max(0, (int)std::floor((x + 0.5f) * width / float(W))));
            const int sy = std::min(height - 1, std::max(0, (int)std::floor((y + 0.5f) * height / float(H))));
            const uint8_t * p = px + ((size_t)sy * width + sx) * channels;
            constexpr float mean[3] = { 0.485f, 0.456f, 0.406f };
            constexpr float stdev[3] = { 0.229f, 0.224f, 0.225f };
            for (int c = 0; c < 3; ++c)
                out[(size_t)c * H * W + y * W + x] = (p[std::min(c, channels - 1)] / 255.0f - mean[c]) / stdev[c];
        }
    return out;
}

context * init(const char * path, int) {
    auto * c = new context();
    c->backend = ggml_backend_cpu_init();
    if (!core_gguf::load_weights(path, c->backend, "pplcnet_orientation", c->wl)) {
        free(c);
        return nullptr;
    }
    const auto & m = c->wl.tensors;
    c->stem = make_conv(m, 0, 3, 16, 3, 2, 1);
    c->stem_bn = make_bn(m, 0);
    struct spec {
        int dw, pw, in, out, k, stride;
        bool se;
        int se1, se2;
    };
    const spec specs[] = {
        { 1, 2, 16, 32, 3, 1, false, 0, 0 },      { 3, 4, 32, 64, 3, 2, false, 0, 0 },
        { 5, 6, 64, 64, 3, 1, false, 0, 0 },      { 7, 8, 64, 128, 3, 2, false, 0, 0 },
        { 9, 10, 128, 128, 3, 1, false, 0, 0 },   { 11, 12, 128, 256, 3, 2, false, 0, 0 },
        { 13, 14, 256, 256, 5, 1, false, 0, 0 },  { 15, 16, 256, 256, 5, 1, false, 0, 0 },
        { 17, 18, 256, 256, 5, 1, false, 0, 0 },  { 19, 20, 256, 256, 5, 1, false, 0, 0 },
        { 21, 22, 256, 256, 5, 1, false, 0, 0 },  { 23, 26, 256, 512, 5, 2, true, 24, 25 },
        { 27, 30, 512, 512, 5, 1, true, 28, 29 },
    };
    for (const auto & s : specs) {
        block b;
        b.dw = make_conv(m, s.dw, s.in, s.in, s.k, s.stride, s.in);
        b.dw_bn = make_bn(m, bn_id_for_conv(s.dw));
        b.pw = make_conv(m, s.pw, s.in, s.out, 1, 1, 1);
        b.pw_bn = make_bn(m, bn_id_for_conv(s.pw));
        b.se = s.se;
        b.residual = s.in == s.out && s.stride == 1;
        if (b.se) {
            b.se1 = make_conv(m, s.se1, s.in, s.in / 4, 1, 1, 1);
            b.se2 = make_conv(m, s.se2, s.in / 4, s.in, 1, 1, 1);
        }
        c->blocks.push_back(b);
    }
    c->head = make_conv(m, 31, 512, 1280, 1, 1, 1);
    c->fc_w = prefix(m, "ori.linear_0.w_0");
    c->fc_b = prefix(m, "ori.linear_0.b_0");
    return c;
}

void free(context * c) {
    if (!c) return;
    core_gguf::free_weights(c->wl);
    if (c->backend) ggml_backend_free(c->backend);
    delete c;
}

result classify_raw(context * c, const uint8_t * px, int width, int height, int channels) {
    result r;
    if (!c || !px || width <= 0 || height <= 0 || channels <= 0) return r;
    auto x = preprocess(px, width, height, channels);
    int h = 80, w = 160;
    if (!run_conv_bn(c->stem, c->stem_bn, x, h, w)) return r;
    for (const auto & b : c->blocks)
        if (!run_block(b, x, h, w)) return r;
    std::vector<float> pooled(512, 0.0f);
    for (int ch = 0; ch < 512; ++ch)
        for (int i = 0; i < h * w; ++i) pooled[ch] += x[(size_t)ch * h * w + i] / float(h * w);
    std::vector<float> y;
    int oh = 0, ow = 0;
    if (!apply_conv(c->head, pooled, 1, 1, y, oh, ow)) return r;
    hardswish(y);
    auto fw = to_f32(c->fc_w), fb = to_f32(c->fc_b);
    float logits[2] = {};
    linear_cpu(y.data(), logits, 1280, 2, fw.data(), fb.data());
    const float mx = std::max(logits[0], logits[1]);
    const float e0 = std::exp(logits[0] - mx), e1 = std::exp(logits[1] - mx);
    const float z = e0 + e1;
    r.probabilities[0] = e0 / z;
    r.probabilities[1] = e1 / z;
    r.angle = r.probabilities[1] > r.probabilities[0] ? 180 : 0;
    r.confidence = std::max(r.probabilities[0], r.probabilities[1]);
    return r;
}

result classify_file(context * c, const char * path) {
    int w = 0, h = 0, ch = 0;
    auto * px = stbi_load(path, &w, &h, &ch, 3);
    if (!px) return {};
    auto r = classify_raw(c, px, w, h, 3);
    stbi_image_free(px);
    return r;
}

} // namespace pplcnet_orientation
