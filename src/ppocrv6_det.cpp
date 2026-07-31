#include "ppocrv6_det.h"

#include "core/cpu_ops.h"
#include "core/gguf_loader.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <queue>
#include <string>
#include <unordered_map>

extern "C" {
unsigned char * stbi_load(const char *, int *, int *, int *, int);
void stbi_image_free(void *);
}

namespace ppocrv6_det {
using core_cpu::conv2d_cpu;
using core_cpu::to_f32;

struct conv {
    ggml_tensor *w = nullptr, *b = nullptr;
    int ic = 0, oc = 0, kh = 1, kw = 1, sh = 1, sw = 1, ph = 0, pw = 0, groups = 1;
};

struct se {
    conv c1, c2;
    bool valid = false;
};

struct block {
    conv dw, cm1, cm2;
    se gate;
    bool residual = false;
};

struct neck_feature {
    conv insert, insert_se1, insert_se2;
    conv dw, pw, se1, se2;
};

struct medium_ic {
    conv reduce, vertical[3], horizontal[3], symmetric[3], final;
};

struct context {
    core_gguf::WeightLoad wl;
    ggml_backend_t backend = nullptr;
    std::string variant;
    int neck = 0;
    int stage_channels[4] = {};
    std::vector<conv> stem;
    std::vector<std::vector<block>> stages;
    std::vector<neck_feature> features;
    std::vector<conv> med_adjust, med_project, med_bottom, med_lateral;
    std::vector<medium_ic> med_ic;
    conv head_down, head_up, head_final;
    std::vector<float> last_prob;
    int last_h = 0, last_w = 0;
    std::unordered_map<std::string, std::vector<float>> last_stages;
};

static ggml_tensor * get(const core_gguf::tensor_map & m, const std::string & n) {
    return core_gguf::try_get(m, n.c_str());
}

static conv make_conv(const core_gguf::tensor_map & m, const std::string & n, int ic, int oc, int kh, int kw = 0,
                      int sh = 1, int sw = 0, int groups = 1, int ph = -1, int pw = -1) {
    conv c;
    c.w = get(m, n + ".weight");
    c.b = get(m, n + ".bias");
    c.ic = ic;
    c.oc = oc;
    c.kh = kh;
    c.kw = kw ? kw : kh;
    c.sh = sh;
    c.sw = sw ? sw : sh;
    c.ph = ph >= 0 ? ph : c.kh / 2;
    c.pw = pw >= 0 ? pw : c.kw / 2;
    c.groups = groups;
    return c;
}

static bool apply_conv(const conv & c, const std::vector<float> & x, int h, int w, std::vector<float> & y, int & oh,
                       int & ow) {
    if (!c.w) return false;
    if (x.size() < (size_t)c.ic * h * w) {
        fprintf(stderr, "ppocrv6-det: input underflow ic=%d h=%d w=%d size=%zu\n", c.ic, h, w, x.size());
        return false;
    }
    oh = (h + 2 * c.ph - c.kh) / c.sh + 1;
    ow = (w + 2 * c.pw - c.kw) / c.sw + 1;
    if (oh <= 0 || ow <= 0) return false;
    y.assign((size_t)c.oc * oh * ow, 0.0f);
    auto ww = to_f32(c.w), bb = to_f32(c.b);
    if (ww.size() < (size_t)c.oc * (c.ic / c.groups) * c.kh * c.kw) {
        fprintf(stderr, "ppocrv6-det: weight underflow ic=%d oc=%d k=%dx%d groups=%d size=%zu\n", c.ic, c.oc, c.kh,
                c.kw, c.groups, ww.size());
        return false;
    }
    if (c.sh == c.sw && c.ph == c.pw) {
        conv2d_cpu(x.data(), y.data(), ww.data(), bb.empty() ? nullptr : bb.data(), c.ic, c.oc, h, w, c.kh, c.kw, c.sh,
                   c.ph, c.groups);
        return true;
    }
    const int cin = c.ic / c.groups, cout = c.oc / c.groups, ks = cin * c.kh * c.kw;
    for (int g = 0; g < c.groups; ++g)
        for (int oc = 0; oc < cout; ++oc)
            for (int oy = 0; oy < oh; ++oy)
                for (int ox = 0; ox < ow; ++ox) {
                    float sum = bb.empty() ? 0.0f : bb[g * cout + oc];
                    const float * wt = ww.data() + (g * cout + oc) * ks;
                    int k = 0;
                    for (int ic = 0; ic < cin; ++ic)
                        for (int ky = 0; ky < c.kh; ++ky)
                            for (int kx = 0; kx < c.kw; ++kx, ++k) {
                                int iy = oy * c.sh - c.ph + ky, ix = ox * c.sw - c.pw + kx;
                                if (iy >= 0 && iy < h && ix >= 0 && ix < w)
                                    sum += x[(size_t)(g * cin + ic) * h * w + iy * w + ix] * wt[k];
                            }
                    y[(size_t)(g * cout + oc) * oh * ow + oy * ow + ox] = sum;
                }
    return true;
}

static bool apply_deconv2(const conv & c, const std::vector<float> & x, int h, int w, std::vector<float> & y, int & oh,
                          int & ow) {
    if (!c.w || c.kh != 2 || c.kw != 2) return false;
    oh = h * 2;
    ow = w * 2;
    y.assign((size_t)c.oc * oh * ow, 0.0f);
    auto ww = to_f32(c.w), bb = to_f32(c.b);
    for (int oc = 0; oc < c.oc; ++oc)
        for (int iy = 0; iy < h; ++iy)
            for (int ix = 0; ix < w; ++ix) {
                for (int ic = 0; ic < c.ic; ++ic)
                    for (int ky = 0; ky < 2; ++ky)
                        for (int kx = 0; kx < 2; ++kx)
                            y[(size_t)oc * oh * ow + (iy * 2 + ky) * ow + ix * 2 + kx] +=
                                x[(size_t)ic * h * w + iy * w + ix] * ww[((size_t)oc * c.ic + ic) * 4 + ky * 2 + kx];
            }
    if (!bb.empty())
        for (int oc = 0; oc < c.oc; ++oc)
            for (int i = 0; i < oh * ow; ++i) y[(size_t)oc * oh * ow + i] += bb[oc];
    return true;
}

static void relu(std::vector<float> & x) {
    for (float & v : x) v = std::max(0.0f, v);
}
static void gelu(std::vector<float> & x) {
    for (float & v : x) v = 0.5f * v * (1.0f + std::erf(v * 0.7071067811865475f));
}

static void pad_bottom_right(const std::vector<float> & x, int c, int h, int w, std::vector<float> & y) {
    y.assign((size_t)c * (h + 1) * (w + 1), 0.0f);
    for (int ch = 0; ch < c; ++ch)
        for (int yy = 0; yy < h; ++yy)
            std::memcpy(y.data() + (size_t)ch * (h + 1) * (w + 1) + (size_t)yy * (w + 1),
                        x.data() + (size_t)ch * h * w + (size_t)yy * w, (size_t)w * sizeof(float));
}

static void maxpool2_stride1(const std::vector<float> & x, int c, int h, int w, std::vector<float> & y) {
    const int oh = h - 1, ow = w - 1;
    y.assign((size_t)c * oh * ow, 0.0f);
    for (int ch = 0; ch < c; ++ch)
        for (int yy = 0; yy < oh; ++yy)
            for (int xx = 0; xx < ow; ++xx) {
                float m = -INFINITY;
                for (int ky = 0; ky < 2; ++ky)
                    for (int kx = 0; kx < 2; ++kx) m = std::max(m, x[(size_t)ch * h * w + (yy + ky) * w + xx + kx]);
                y[(size_t)ch * oh * ow + yy * ow + xx] = m;
            }
}

static void upsample2(const std::vector<float> & x, int c, int h, int w, std::vector<float> & y) {
    y.assign((size_t)c * h * 2 * w * 2, 0.0f);
    for (int cc = 0; cc < c; ++cc)
        for (int yy = 0; yy < h; ++yy)
            for (int xx = 0; xx < w; ++xx) {
                float v = x[(size_t)cc * h * w + yy * w + xx];
                for (int dy = 0; dy < 2; ++dy)
                    for (int dx = 0; dx < 2; ++dx)
                        y[(size_t)cc * h * 2 * w * 2 + (yy * 2 + dy) * w * 2 + xx * 2 + dx] = v;
            }
}

static void resize_nearest(const std::vector<float> & x, int c, int h, int w, int oh, int ow, std::vector<float> & y) {
    y.assign((size_t)c * oh * ow, 0.0f);
    for (int cc = 0; cc < c; ++cc)
        for (int yy = 0; yy < oh; ++yy)
            for (int xx = 0; xx < ow; ++xx) {
                int sy = std::min(h - 1, yy * h / oh), sx = std::min(w - 1, xx * w / ow);
                y[(size_t)cc * oh * ow + yy * ow + xx] = x[(size_t)cc * h * w + sy * w + sx];
            }
}

static void add_inplace(std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return;
    for (size_t i = 0; i < a.size(); ++i) a[i] += b[i];
}

static bool run_se(const se & s, std::vector<float> & x, int channels, std::vector<float> * gate_out = nullptr) {
    if (!s.valid) return true;
    std::vector<float> p(channels, 0.0f), g;
    int gh, gw;
    for (int c = 0; c < channels; ++c) {
        // x is a 1x1 tensor for the SE path.
        p[c] = x[c];
    }
    if (!apply_conv(s.c1, p, 1, 1, g, gh, gw)) return false;
    relu(g);
    if (!apply_conv(s.c2, g, 1, 1, p, gh, gw)) return false;
    if (gate_out) gate_out->resize(channels);
    for (int c = 0; c < channels; ++c) {
        const float gate = std::clamp(p[c] / 6.0f + 0.5f, 0.0f, 1.0f);
        if (gate_out) (*gate_out)[c] = gate;
        x[c] *= gate;
    }
    return true;
}

static bool run_block(const block & b, std::vector<float> & x, int & h, int & w, context * debug = nullptr,
                      const std::string & prefix = {}) {
    std::vector<float> y, z, out;
    int oh, ow, nh, nw;
    if (!apply_conv(b.dw, x, h, w, y, oh, ow)) return false;
    if (debug && !prefix.empty()) debug->last_stages[prefix + "_dw"] = y;
    if (b.gate.valid) {
        std::vector<float> pooled(b.dw.ic, 0.0f);
        for (int c = 0; c < b.dw.ic; ++c)
            for (int i = 0; i < oh * ow; ++i) pooled[c] += y[(size_t)c * oh * ow + i] / float(oh * ow);
        const std::vector<float> pooled_before = pooled;
        std::vector<float> gate;
        if (!run_se(b.gate, pooled, b.dw.ic, debug && !prefix.empty() ? &gate : nullptr)) return false;
        if (debug && !prefix.empty()) {
            debug->last_stages[prefix + "_pool"] = pooled_before;
            debug->last_stages[prefix + "_gate"] = gate;
        }
        for (int c = 0; c < b.dw.ic; ++c)
            for (int i = 0; i < oh * ow; ++i) y[(size_t)c * oh * ow + i] *= pooled[c];
        if (debug && !prefix.empty()) debug->last_stages[prefix + "_se"] = y;
    }
    if (!apply_conv(b.cm1, y, oh, ow, z, nh, nw)) return false;
    gelu(z);
    if (debug && !prefix.empty()) debug->last_stages[prefix + "_cm1"] = z;
    if (!apply_conv(b.cm2, z, nh, nw, out, nh, nw)) return false;
    if (b.residual && out.size() == y.size()) add_inplace(out, y);
    x.swap(out);
    if (debug && !prefix.empty()) debug->last_stages[prefix + "_out"] = x;
    h = nh;
    w = nw;
    return true;
}

static bool run_stem(context * c, const std::vector<float> & input, int h, int w, std::vector<float> & out, int & oh,
                     int & ow) {
    std::vector<float> x = input, y, branch;
    int H = h, W = w;
    if (!apply_conv(c->stem[0], x, H, W, y, oh, ow)) return false;
    relu(y);
    c->last_stages["stem1"] = y;
    x.swap(y);
    H = oh;
    W = ow;
    std::vector<float> padded;
    pad_bottom_right(x, c->stem[0].oc, H, W, padded);
    std::vector<float> embedding_padded = padded;
    if (!apply_conv(c->stem[1], padded, H + 1, W + 1, branch, oh, ow)) return false;
    relu(branch);
    pad_bottom_right(branch, c->stem[1].oc, oh, ow, padded);
    if (!apply_conv(c->stem[2], padded, oh + 1, ow + 1, y, oh, ow)) return false;
    relu(y);
    branch.swap(y);
    c->last_stages["stem2b"] = branch;
    std::vector<float> pooled;
    maxpool2_stride1(embedding_padded, c->stem[0].oc, H + 1, W + 1, pooled);
    c->last_stages["stem_pooled"] = pooled;
    const int cat_h = H, cat_w = W;
    std::vector<float> cat(pooled.size() + branch.size());
    std::memcpy(cat.data(), pooled.data(), pooled.size() * sizeof(float));
    std::memcpy(cat.data() + pooled.size(), branch.data(), branch.size() * sizeof(float));
    if (!apply_conv(c->stem[3], cat, cat_h, cat_w, y, oh, ow)) return false;
    relu(y);
    x.swap(y);
    c->last_stages["stem3"] = x;
    H = oh;
    W = ow;
    if (!apply_conv(c->stem[4], x, H, W, out, oh, ow)) return false;
    relu(out);
    c->last_stages["stem4"] = out;
    return true;
}

static bool run_medium_ic(const medium_ic & b, std::vector<float> & x, int h, int w) {
    std::vector<float> r;
    int rh, rw;
    if (!apply_conv(b.reduce, x, h, w, r, rh, rw)) return false;
    std::vector<float> a[3];
    for (int j = 0; j < 3; ++j) {
        std::vector<float> v, q, s;
        int vh, vw;
        const std::vector<float> & source = j == 0 ? r : a[j - 1];
        if (!apply_conv(b.vertical[j], source, h, w, v, vh, vw) ||
            !apply_conv(b.horizontal[j], source, h, w, q, vh, vw) ||
            !apply_conv(b.symmetric[j], source, h, w, s, vh, vw))
            return false;
        a[j].resize(s.size());
        for (size_t k = 0; k < s.size(); ++k) a[j][k] = v[k] + q[k] + s[k];
    }
    std::vector<float> z = a[2];
    std::vector<float> y;
    if (!apply_conv(b.final, z, h, w, y, rh, rw)) return false;
    for (size_t k = 0; k < y.size(); ++k) y[k] += x[k];
    x.swap(y);
    return true;
}

static bool run_medium_neck(const context * c, const std::vector<std::vector<float>> & stages,
                            const std::vector<int> & hs, const std::vector<int> & ws, std::vector<float> & neck,
                            int & nh, int & nw) {
    std::vector<std::vector<float>> adjusted(4), top(4), projected(4), bottom(4), lateral(4), refined(4);
    std::vector<int> ah(4), aw(4);
    for (int i = 0; i < 4; ++i)
        if (!apply_conv(c->med_adjust[i], stages[i], hs[i], ws[i], adjusted[i], ah[i], aw[i])) return false;
    top[3] = adjusted[3];
    for (int i = 2; i >= 0; --i) {
        std::vector<float> up;
        resize_nearest(top[i + 1], c->neck, ah[i + 1], aw[i + 1], ah[i], aw[i], up);
        top[i] = adjusted[i];
        add_inplace(top[i], up);
    }
    for (int i = 0; i < 4; ++i) {
        const auto & source = i < 3 ? top[i] : adjusted[3];
        if (!apply_conv(c->med_project[i], source, ah[i], aw[i], projected[i], ah[i], aw[i])) return false;
    }
    bottom[0] = projected[0];
    for (int i = 1; i < 4; ++i) {
        std::vector<float> down;
        int dh, dw;
        if (!apply_conv(c->med_bottom[i - 1], bottom[i - 1], ah[i - 1], aw[i - 1], down, dh, dw)) return false;
        resize_nearest(down, c->neck / 4, dh, dw, (int)projected[i].size() / (c->neck / 4) / aw[i], aw[i], down);
        bottom[i] = projected[i];
        add_inplace(bottom[i], down);
    }
    for (int i = 0; i < 4; ++i) {
        const auto & source = i == 0 ? projected[0] : bottom[i];
        int sh = ah[i], sw = aw[i];
        if (!apply_conv(c->med_lateral[i], source, sh, sw, lateral[i], sh, sw)) return false;
        refined[i] = lateral[i];
        if (!run_medium_ic(c->med_ic[i], refined[i], sh, sw)) return false;
    }
    nh = ah[0];
    nw = aw[0];
    neck.clear();
    for (int i = 3; i >= 0; --i) {
        std::vector<float> up;
        resize_nearest(refined[i], c->neck / 4, ah[i], aw[i], nh, nw, up);
        neck.insert(neck.end(), up.begin(), up.end());
    }
    return true;
}

static void append_component(const std::vector<float> & prob, int h, int w, float threshold, std::vector<box> & out) {
    std::vector<uint8_t> seen((size_t)h * w, 0);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            if (seen[(size_t)y * w + x] || prob[(size_t)y * w + x] < threshold) continue;
            std::queue<std::pair<int, int>> q;
            q.push({ x, y });
            seen[(size_t)y * w + x] = 1;
            int x0 = x, x1 = x, y0 = y, y1 = y, n = 0;
            float sum = 0;
            while (!q.empty()) {
                auto [cx, cy] = q.front();
                q.pop();
                ++n;
                x0 = std::min(x0, cx);
                x1 = std::max(x1, cx);
                y0 = std::min(y0, cy);
                y1 = std::max(y1, cy);
                sum += prob[(size_t)cy * w + cx];
                for (auto [dx, dy] : { std::pair<int, int>{ 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } }) {
                    int nx = cx + dx, ny = cy + dy;
                    if (nx >= 0 && nx < w && ny >= 0 && ny < h && !seen[(size_t)ny * w + nx] &&
                        prob[(size_t)ny * w + nx] >= threshold) {
                        seen[(size_t)ny * w + nx] = 1;
                        q.push({ nx, ny });
                    }
                }
            }
            if (n >= 4) out.push_back({ float(x0), float(y0), float(x1 - x0 + 1), float(y1 - y0 + 1), sum / n });
        }
}

context * init(const char * path, int) {
    auto * c = new context();
    c->backend = ggml_backend_cpu_init();
    auto * meta = core_gguf::open_metadata(path);
    if (!meta) {
        delete c;
        return nullptr;
    }
    c->variant = core_gguf::kv_str(meta, "ppocrv6.variant", "tiny");
    core_gguf::free_metadata(meta);
    if (!core_gguf::load_weights(path, c->backend, "ppocrv6", c->wl)) {
        free(c);
        return nullptr;
    }
    const auto & m = c->wl.tensors;
    const bool tiny = c->variant == "tiny", medium = c->variant == "medium";
    int stem = medium ? 64 : (tiny ? 16 : 24);
    int stage[4] = { medium ? 128 : (tiny ? 32 : 48), medium ? 256 : (tiny ? 48 : 96), medium ? 512 : (tiny ? 64 : 192),
                     medium ? 896 : (tiny ? 160 : 384) };
    c->stage_channels[0] = stage[0];
    c->stage_channels[1] = stage[1];
    c->stage_channels[2] = stage[2];
    c->stage_channels[3] = stage[3];
    c->neck = medium ? 256 : (tiny ? 64 : 96);
    c->stem = { make_conv(m, "det.bb.stem.stem1.conv", 3, stem, 3, 3, 2),
                make_conv(m, "det.bb.stem.stem2a.conv", stem, stem / 2, 2, 2, 1, 1, 1, 0, 0),
                make_conv(m, "det.bb.stem.stem2b.conv", stem / 2, stem, 2, 2, 1, 1, 1, 0, 0),
                make_conv(m, "det.bb.stem.stem3.conv", stem * 2, stem, 3, 3, 2),
                make_conv(m, "det.bb.stem.stem4.conv", stem, stage[0], 1) };
    c->stages.resize(4);
    for (int si = 0; si < 4; ++si)
        for (int bi = 0; bi < 16; ++bi) {
            std::string q = "det.bb.blk." + std::to_string(si) + ".b." + std::to_string(bi);
            if (!get(m, q + ".cm1.weight")) break;
            int ic = bi == 0 && si ? stage[si - 1] : stage[si], stride = bi == 0 && si > 0 ? 2 : 1;
            block b;
            b.dw = make_conv(m, q + ".dw", ic, ic, 3, 3, stride, stride, ic);
            b.cm1 = make_conv(m, q + ".cm1", ic, ic * 2, 1);
            b.cm2 = make_conv(m, q + ".cm2", ic * 2, stage[si], 1);
            b.residual = ic == stage[si] && stride == 1;
            b.gate.valid = get(m, q + ".se1.weight") != nullptr;
            if (b.gate.valid) {
                b.gate.c1 = make_conv(m, q + ".se1", ic, ic / 4, 1);
                b.gate.c2 = make_conv(m, q + ".se2", ic / 4, ic, 1);
            }
            c->stages[si].push_back(b);
        }
    if (medium) {
        c->med_adjust.resize(4);
        c->med_project.resize(4);
        c->med_bottom.resize(3);
        c->med_lateral.resize(4);
        for (int i = 0; i < 4; ++i) {
            c->med_adjust[i] = make_conv(m, "det.neck.input_channel_adjustment_convolution." + std::to_string(i),
                                         stage[i], c->neck, 1);
            c->med_project[i] = make_conv(m, "det.neck.input_feature_projection_convolution." + std::to_string(i),
                                          c->neck, c->neck / 4, 9, 9, 1, 1, 1, 4, 4);
            c->med_lateral[i] = make_conv(m, "det.neck.path_aggregation_lateral_convolution." + std::to_string(i),
                                          c->neck / 4, c->neck / 4, 9, 9, 1, 1, 1, 4, 4);
            if (i > 0)
                c->med_bottom[i - 1] =
                    make_conv(m, "det.neck.path_aggregation_head_convolution." + std::to_string(i - 1), c->neck / 4,
                              c->neck / 4, 3, 3, 2, 2, 1, 1, 1);
        }
        c->med_ic.resize(4);
        for (int i = 0; i < 4; ++i) {
            auto & b = c->med_ic[i];
            const std::string q = "det.nk.ic." + std::to_string(i);
            b.reduce = make_conv(m, q + ".conv_reduce_channel", 64, 32, 1, 1, 1, 1, 1, 0, 0);
            const int ks[3] = { 7, 5, 3 }, ps[3] = { 3, 2, 1 };
            for (int j = 0; j < 3; ++j) {
                b.vertical[j] = make_conv(m,
                                          q + ".vl" +
                                              (j == 0   ? "l"
                                               : j == 1 ? "m"
                                                        : "s"),
                                          32, 32, ks[j], 1, 1, 1, 1, ps[j], 0);
                b.horizontal[j] = make_conv(m,
                                            q + ".hs" +
                                                (j == 0   ? "l"
                                                 : j == 1 ? "m"
                                                          : "s"),
                                            32, 32, 1, ks[j], 1, 1, 1, 0, ps[j]);
                b.symmetric[j] = make_conv(m,
                                           q + ".sl" +
                                               (j == 0   ? "l"
                                                : j == 1 ? "m"
                                                         : "s"),
                                           32, 32, ks[j], ks[j], 1, 1, 1, ps[j], ps[j]);
            }
            b.final = make_conv(m, q + ".conv_final.conv", 32, 64, 1, 1, 1, 1, 1, 0, 0);
        }
    }
    c->features.resize(4);
    for (int i = 0; i < 4; ++i) {
        auto & f = c->features[i];
        f.insert = make_conv(m, "det.neck.insert_conv." + std::to_string(i) + ".in_conv", stage[i], c->neck, 1);
        f.insert_se1 = make_conv(m, "det.neck.insert_conv." + std::to_string(i) + ".squeeze_excitation_block.conv1",
                                 c->neck, c->neck / 4, 1);
        f.insert_se2 = make_conv(m, "det.neck.insert_conv." + std::to_string(i) + ".squeeze_excitation_block.conv2",
                                 c->neck / 4, c->neck, 1);
        int k = tiny ? 5 : 7;
        f.dw = make_conv(m, "det.neck.input_conv." + std::to_string(i) + ".depthwise_convolution", c->neck, c->neck, k,
                         k, 1, 1, c->neck);
        f.pw = make_conv(m, "det.neck.input_conv." + std::to_string(i) + ".pointwise_convolution", c->neck, c->neck / 4,
                         1);
        f.se1 = make_conv(m, "det.neck.input_conv." + std::to_string(i) + ".squeeze_excitation_module.conv1",
                          c->neck / 4, c->neck / 16, 1);
        f.se2 = make_conv(m, "det.neck.input_conv." + std::to_string(i) + ".squeeze_excitation_module.conv2",
                          c->neck / 16, c->neck / 4, 1);
    }
    c->head_down = make_conv(m, "det.head.conv_down.conv", c->neck, c->neck / 4, 3);
    c->head_up = make_conv(m, "det.head.conv_up.conv", c->neck / 4, c->neck / 4, 2, 2, 2, 2, 1, 0, 0);
    c->head_final = make_conv(m, "det.head.conv_final", c->neck / 4, 1, 2, 2, 2, 2, 1, 0, 0);
    return c;
}

void free(context * c) {
    if (!c) return;
    core_gguf::free_weights(c->wl);
    if (c->backend) ggml_backend_free(c->backend);
    delete c;
}

std::vector<box> detect_raw(context * c, const uint8_t * px, int w, int h, int channels, float threshold) {
    if (!c || !px) return {};
    static constexpr float mean[3] = { 0.485f, 0.456f, 0.406f };
    static constexpr float stdev[3] = { 0.229f, 0.224f, 0.225f };
    std::vector<float> input((size_t)3 * h * w);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++)
            for (int ch = 0; ch < 3; ch++) {
                const int src_ch = 2 - ch; // official detector transform decodes BGR
                input[(size_t)ch * h * w + y * w + x] =
                    (px[((size_t)y * w + x) * channels + std::min(src_ch, channels - 1)] / 255.0f - mean[ch]) /
                    stdev[ch];
            }
    std::vector<float> x;
    int H, W;
    if (!run_stem(c, input, h, w, x, H, W)) return {};
    std::vector<std::vector<float>> stages;
    std::vector<int> hs, ws;
    for (size_t si = 0; si < c->stages.size(); ++si) {
        auto & ss = c->stages[si];
        for (size_t bi = 0; bi < ss.size(); ++bi)
            if (!run_block(ss[bi], x, H, W, c, si == 0 && bi == 0 ? "block0" : "")) return {};
        stages.push_back(x);
        hs.push_back(H);
        ws.push_back(W);
        c->last_stages["backbone_stage" + std::to_string(stages.size() - 1)] = x;
    }
    if (c->variant == "medium") {
        std::vector<float> neck;
        int nh, nw;
        if (!run_medium_neck(c, stages, hs, ws, neck, nh, nw)) return {};
        std::vector<float> y, z;
        int oh, ow;
        if (!apply_conv(c->head_down, neck, nh, nw, y, oh, ow)) return {};
        relu(y);
        if (!apply_deconv2(c->head_up, y, oh, ow, z, oh, ow)) return {};
        relu(z);
        if (!apply_deconv2(c->head_final, z, oh, ow, y, oh, ow)) return {};
        c->last_stages["neck_output"] = neck;
        for (float & v : y) v = 1 / (1 + std::exp(-v));
        c->last_prob = y;
        c->last_h = oh;
        c->last_w = ow;
        std::vector<box> out;
        append_component(y, oh, ow, threshold, out);
        float sx = float(w) / ow, sy = float(h) / oh;
        for (auto & b : out) {
            b.x *= sx;
            b.w *= sx;
            b.y *= sy;
            b.h *= sy;
        }
        return out;
    }
    std::vector<std::vector<float>> fused(4);
    std::vector<int> fh(4), fw(4);
    for (int i = 0; i < 4; i++) {
        if (!apply_conv(c->features[i].insert, stages[i], hs[i], ws[i], fused[i], fh[i], fw[i])) return {};
        std::vector<float> p(c->neck);
        for (int k = 0; k < c->neck; k++)
            for (int j = 0; j < fh[i] * fw[i]; j++) p[k] += fused[i][(size_t)k * fh[i] * fw[i] + j] / (fh[i] * fw[i]);
        std::vector<float> g;
        int gh, gw;
        if (!apply_conv(c->features[i].insert_se1, p, 1, 1, g, gh, gw)) return {};
        relu(g);
        if (!apply_conv(c->features[i].insert_se2, g, 1, 1, p, gh, gw)) return {};
        for (int k = 0; k < c->neck; k++)
            for (int j = 0; j < fh[i] * fw[i]; j++)
                fused[i][(size_t)k * fh[i] * fw[i] + j] +=
                    fused[i][(size_t)k * fh[i] * fw[i] + j] * std::clamp(0.2f * p[k] + 0.5f, 0.f, 1.f);
    }
    for (int i = 2; i >= 0; i--) {
        std::vector<float> u;
        resize_nearest(fused[i + 1], c->neck, fh[i + 1], fw[i + 1], fh[i], fw[i], u);
        add_inplace(fused[i], u);
    }
    std::vector<std::vector<float>> proc(4);
    std::vector<int> ph(4), pw(4);
    for (int i = 0; i < 4; i++) {
        auto & f = c->features[i];
        std::vector<float> z;
        int a, b;
        if (!apply_conv(f.dw, fused[i], fh[i], fw[i], z, a, b)) return {};
        std::vector<float> q;
        int qa, qb;
        if (!apply_conv(f.pw, z, a, b, q, qa, qb)) return {};
        std::vector<float> pooled(c->neck / 4, 0.0f), gate, seout;
        for (int ch = 0; ch < c->neck / 4; ++ch)
            for (int j = 0; j < qa * qb; ++j) pooled[ch] += q[(size_t)ch * qa * qb + j] / float(qa * qb);
        int gh, gw;
        if (!apply_conv(f.se1, pooled, 1, 1, gate, gh, gw)) return {};
        relu(gate);
        if (!apply_conv(f.se2, gate, 1, 1, seout, gh, gw)) return {};
        for (int ch = 0; ch < c->neck / 4; ++ch) {
            const float scale = std::clamp(0.2f * seout[ch] + 0.5f, 0.0f, 1.0f);
            for (int j = 0; j < qa * qb; ++j) q[(size_t)ch * qa * qb + j] += q[(size_t)ch * qa * qb + j] * scale;
        }
        proc[i] = q;
        ph[i] = qa;
        pw[i] = qb;
    }
    std::vector<float> neck;
    int nh = ph[0], nw = pw[0];
    for (int i = 3; i >= 0; i--) {
        std::vector<float> u = proc[i];
        int uh = ph[i], uw = pw[i];
        std::vector<float> resized;
        resize_nearest(u, c->neck / 4, uh, uw, nh, nw, resized);
        u.swap(resized);
        neck.insert(neck.end(), u.begin(), u.end());
    }
    c->last_stages["neck_output"] = neck;
    std::vector<float> y, z;
    int oh, ow;
    if (!apply_conv(c->head_down, neck, nh, nw, y, oh, ow)) return {};
    relu(y);
    if (!apply_deconv2(c->head_up, y, oh, ow, z, oh, ow)) return {};
    relu(z);
    if (!apply_deconv2(c->head_final, z, oh, ow, y, oh, ow)) return {};
    for (float & v : y) v = 1 / (1 + std::exp(-v));
    c->last_prob = y;
    c->last_h = oh;
    c->last_w = ow;
    std::vector<box> out;
    append_component(y, oh, ow, threshold, out);
    float sx = float(w) / ow, sy = float(h) / oh;
    for (auto & b : out) {
        b.x *= sx;
        b.w *= sx;
        b.y *= sy;
        b.h *= sy;
    }
    return out;
}

const float * last_probability(const context * c, int * height, int * width) {
    if (!c || c->last_prob.empty()) return nullptr;
    if (height) *height = c->last_h;
    if (width) *width = c->last_w;
    return c->last_prob.data();
}

const float * last_stage(const context * c, const char * name, size_t * n) {
    if (!c || !name) return nullptr;
    auto it = c->last_stages.find(name);
    if (it == c->last_stages.end()) return nullptr;
    if (n) *n = it->second.size();
    return it->second.data();
}

std::vector<box> detect_file(context * c, const char * path, float threshold) {
    int w, h, ch;
    auto * p = stbi_load(path, &w, &h, &ch, 3);
    if (!p) return {};
    float scale = std::min(1.0f, 960.0f / float(std::max(w, h)));
    int rw = std::max(32, int(std::ceil(w * scale / 32.0f)) * 32);
    int rh = std::max(32, int(std::ceil(h * scale / 32.0f)) * 32);
    std::vector<uint8_t> resized((size_t)rw * rh * 3);
    for (int y = 0; y < rh; ++y)
        for (int x = 0; x < rw; ++x) {
            const float fy = std::max(0.0f, (y + 0.5f) * h / rh - 0.5f);
            const float fx = std::max(0.0f, (x + 0.5f) * w / rw - 0.5f);
            const int y0 = std::min(h - 1, int(fy)), y1 = std::min(h - 1, y0 + 1);
            const int x0 = std::min(w - 1, int(fx)), x1 = std::min(w - 1, x0 + 1);
            const float wy = fy - y0, wx = fx - x0;
            for (int ch = 0; ch < 3; ++ch) {
                const float a = p[((size_t)y0 * w + x0) * 3 + ch] * (1 - wx) + p[((size_t)y0 * w + x1) * 3 + ch] * wx;
                const float b = p[((size_t)y1 * w + x0) * 3 + ch] * (1 - wx) + p[((size_t)y1 * w + x1) * 3 + ch] * wx;
                resized[((size_t)y * rw + x) * 3 + ch] = (uint8_t)std::clamp(a * (1 - wy) + b * wy, 0.0f, 255.0f);
            }
        }
    auto r = detect_raw(c, resized.data(), rw, rh, 3, threshold);
    for (auto & b : r) {
        b.x *= float(w) / rw;
        b.w *= float(w) / rw;
        b.y *= float(h) / rh;
        b.h *= float(h) / rh;
    }
    stbi_image_free(p);
    return r;
}
} // namespace ppocrv6_det
