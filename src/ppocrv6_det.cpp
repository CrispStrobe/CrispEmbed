#include "ppocrv6_det.h"
#include "ocr_detect.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "core/cpu_ops.h"
#include "core/gguf_loader.h"
#include "core/gpu_backend_pref.h"
#include "ggml-cpu.h"
#include "core/env_gate.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
unsigned char * stbi_load(const char *, int *, int *, int *, int);
void stbi_image_free(void *);
}

namespace ppocrv6_det {
using core_cpu::conv2d_cpu;
using core_cpu::to_f32;

struct conv {
    ggml_tensor *w = nullptr, *b = nullptr;
    mutable std::vector<float> wf, bf;
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
    int n_threads = 1;
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
    struct graph_state {
        ggml_backend_t backend = nullptr;
        ggml_backend_t cpu_backend = nullptr;
        ggml_backend_sched_t sched = nullptr;
        ggml_context * graph_ctx = nullptr;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * input = nullptr;
        ggml_tensor * output = nullptr;
        std::vector<uint8_t> meta;
        std::unordered_map<const ggml_tensor *, ggml_tensor *> resident;
        std::vector<ggml_context *> resident_ctxs;
        std::vector<ggml_backend_buffer_t> resident_bufs;
        std::vector<ggml_tensor *> taps;
        std::vector<std::pair<std::string, ggml_tensor *>> named_taps;
        int h = 0, w = 0;
        bool attempted = false;
        bool ready = false;
        bool probability_output = false;
    } graph;
};

static ggml_tensor * get(const core_gguf::tensor_map & m, const std::string & n) {
    return core_gguf::try_get(m, n.c_str());
}

static float box_iou(const box & a, const box & b) {
    const float ax2 = a.x + a.w, ay2 = a.y + a.h;
    const float bx2 = b.x + b.w, by2 = b.y + b.h;
    const float iw = std::max(0.0f, std::min(ax2, bx2) - std::max(a.x, b.x));
    const float ih = std::max(0.0f, std::min(ay2, by2) - std::max(a.y, b.y));
    const float inter = iw * ih;
    const float area = std::max(0.0f, a.w) * std::max(0.0f, a.h) + std::max(0.0f, b.w) * std::max(0.0f, b.h) - inter;
    return area > 0.0f ? inter / area : 0.0f;
}

static void report_graph_box_geometry(const std::vector<box> & graph_boxes, const std::vector<box> & cpu_boxes) {
    std::vector<bool> used(cpu_boxes.size(), false);
    double sum_iou = 0.0;
    float min_iou = 1.0f;
    size_t matched = 0;
    for (const box & graph_box : graph_boxes) {
        size_t best = cpu_boxes.size();
        float best_iou = 0.0f;
        for (size_t i = 0; i < cpu_boxes.size(); ++i) {
            if (!used[i]) {
                const float iou = box_iou(graph_box, cpu_boxes[i]);
                if (iou > best_iou) {
                    best_iou = iou;
                    best = i;
                }
            }
        }
        if (best < cpu_boxes.size()) {
            used[best] = true;
            ++matched;
            sum_iou += best_iou;
            min_iou = std::min(min_iou, best_iou);
        }
    }
    fprintf(stderr, "ppocrv6-det: graph-vs-CPU boxes graph=%zu cpu=%zu matched=%zu mean_iou=%.6f min_iou=%.6f\n",
            graph_boxes.size(), cpu_boxes.size(), matched, matched ? sum_iou / matched : 0.0, matched ? min_iou : 0.0f);
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

// ---------------------------------------------------------------------------
// Per-convolution CPU profile (CRISPEMBED_PPOCRV6_DET_PROFILE=1)
// ---------------------------------------------------------------------------
// PLAN.md H2 asks for a per-layer cost table for the scalar detector before
// anything is optimised, since the ggml graph route is a measured dead end
// (2.6-6.8x slower). Rows are keyed on the convolution's shape signature,
// which is unique enough to name the layer in this architecture. Diagnostic
// only: when the gate is unset nothing is timed, allocated or printed.

namespace detprof {

static bool enabled() {
    static const bool on = std::getenv("CRISPEMBED_PPOCRV6_DET_PROFILE") != nullptr;
    return on;
}

struct entry {
    long long calls = 0;
    double ms = 0.0;
    double mflop = 0.0;
};

static std::map<std::string, entry> & table() {
    static std::map<std::string, entry> t;
    return t;
}

static std::string shape_key(const conv & c, int h, int w, int oh, int ow, const char * kind) {
    char buf[192];
    snprintf(buf, sizeof(buf), "%-7s ic=%-4d oc=%-4d k=%dx%d s=%dx%d g=%-4d in=%dx%-4d out=%dx%d", kind, c.ic, c.oc,
             c.kh, c.kw, c.sh, c.sw, c.groups, h, w, oh, ow);
    return buf;
}

// One timed convolution. Non-copyable RAII so every early return in the
// instrumented helpers is accounted for without duplicating the stop call.
struct scope {
    bool on;
    std::string key;
    double mflop = 0.0;
    std::chrono::steady_clock::time_point t0;

    scope(const conv & c, int h, int w, int oh, int ow, const char * kind) : on(enabled()) {
        if (!on) return;
        key = shape_key(c, h, w, oh, ow, kind);
        const int cin = c.groups > 0 ? c.ic / c.groups : c.ic;
        mflop = 2.0 * c.oc * cin * c.kh * c.kw * (double)oh * (double)ow / 1e6;
        t0 = std::chrono::steady_clock::now();
    }
    scope(const scope &) = delete;
    scope & operator=(const scope &) = delete;
    ~scope() {
        if (!on) return;
        entry & e = table()[key];
        e.calls++;
        e.ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        e.mflop += mflop;
    }
};

// Prints the accumulated table, heaviest first, then resets it so a second
// detect call in the same process reports its own page rather than a running
// total.
struct report {
    ~report() {
        if (!enabled()) return;
        std::vector<std::pair<std::string, entry>> rows(table().begin(), table().end());
        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) { return a.second.ms > b.second.ms; });
        double total = 0.0;
        for (const auto & r : rows) total += r.second.ms;
        fprintf(stderr, "[ppocrv6-det-profile] total_conv_ms=%.3f distinct_layers=%zu\n", total, rows.size());
        for (const auto & r : rows)
            fprintf(stderr, "[ppocrv6-det-profile] %9.3f ms %5.1f%% calls=%-3lld %7.2f GF/s  %s\n", r.second.ms,
                    total > 0.0 ? 100.0 * r.second.ms / total : 0.0, r.second.calls,
                    r.second.ms > 0.0 ? r.second.mflop / r.second.ms : 0.0, r.first.c_str());
        table().clear();
    }
};

} // namespace detprof

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
    const detprof::scope prof(c, h, w, oh, ow, "conv");
    y.assign((size_t)c.oc * oh * ow, 0.0f);
    if (c.wf.empty()) c.wf = to_f32(c.w);
    if (c.b && c.bf.empty()) c.bf = to_f32(c.b);
    const auto & ww = c.wf;
    const auto & bb = c.bf;
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
    const detprof::scope prof(c, h, w, oh, ow, "deconv");
    y.assign((size_t)c.oc * oh * ow, 0.0f);
    if (c.wf.empty()) c.wf = to_f32(c.w);
    if (c.b && c.bf.empty()) c.bf = to_f32(c.b);
    const auto & ww = c.wf;
    const auto & bb = c.bf;
    for (int oc = 0; oc < c.oc; ++oc)
        for (int iy = 0; iy < h; ++iy)
            for (int ix = 0; ix < w; ++ix) {
                for (int ic = 0; ic < c.ic; ++ic)
                    for (int ky = 0; ky < 2; ++ky)
                        for (int kx = 0; kx < 2; ++kx)
                            // Paddle Conv2DTranspose stores [in, out, kh, kw].
                            y[(size_t)oc * oh * ow + (iy * 2 + ky) * ow + ix * 2 + kx] +=
                                x[(size_t)ic * h * w + iy * w + ix] * ww[((size_t)ic * c.oc + oc) * 4 + ky * 2 + kx];
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
        x[c] = gate;
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

static ggml_tensor * graph_resident(context * c, const ggml_tensor * src, ggml_type type, int64_t ne0, int64_t ne1,
                                    int64_t ne2, int64_t ne3) {
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

static ggml_tensor * graph_conv(context * c, ggml_context * g, ggml_tensor * x, const conv & p) {
    if (!p.w) return nullptr;
    const bool dw = p.groups == p.ic;
    const int icg = dw ? 1 : p.ic / p.groups;
    // Metal/CUDA convolution kernels are substantially more efficient with
    // resident half weights. Keep CPU graph parity in F32 and provide an
    // explicit F32 override for backend-diff debugging.
    const bool force_f32 = std::getenv("CRISPEMBED_PPOCRV6_DET_F32_WEIGHTS") != nullptr;
    const ggml_type weight_type = force_f32 || ggml_backend_is_cpu(c->graph.backend) ? GGML_TYPE_F32 : GGML_TYPE_F16;
    ggml_tensor * w = graph_resident(c, p.w, weight_type, p.kw, p.kh, icg, p.oc);
    if (!w) return nullptr;
    // Direct-convolution A/B (no materialized im2col; native Metal kernel in
    // this ggml revision): CRISPEMBED_PPOCRV6_CONV_DIRECT=1, non-CPU only.
    static const bool conv_direct =
        std::getenv("CRISPEMBED_PPOCRV6_CONV_DIRECT") != nullptr && !ggml_backend_is_cpu(c->graph.backend);
    ggml_tensor * y = dw ? ggml_conv_2d_dw(g, w, x, p.sw, p.sh, p.pw, p.ph, 1, 1)
                         : (conv_direct ? ggml_conv_2d_direct(g, w, x, p.sw, p.sh, p.pw, p.ph, 1, 1)
                                        : ggml_conv_2d(g, w, x, p.sw, p.sh, p.pw, p.ph, 1, 1));
    if (p.b) {
        ggml_tensor * b = graph_resident(c, p.b, GGML_TYPE_F32, p.oc, 1, 1, 1);
        if (!b) return nullptr;
        y = ggml_add(g, y, ggml_reshape_3d(g, b, 1, 1, p.oc));
    }
    return y;
}

static ggml_tensor * graph_deconv(context * c, ggml_context * g, ggml_tensor * x, const conv & p) {
    if (!p.w || p.kh != 2 || p.kw != 2 || p.sh != 2 || p.sw != 2) return nullptr;
    // Paddle stores transpose kernels as [IC, OC, KH, KW]; GGML expects
    // [KW, KH, OC, IC] with the same contiguous bytes.
    ggml_tensor * w = graph_resident(c, p.w, GGML_TYPE_F16, p.kw, p.kh, p.oc, p.ic);
    if (!w) return nullptr;
    ggml_tensor * y = ggml_conv_transpose_2d_p0(g, w, x, 2);
    if (p.b) {
        ggml_tensor * b = graph_resident(c, p.b, GGML_TYPE_F32, p.oc, 1, 1, 1);
        if (!b) return nullptr;
        y = ggml_add(g, y, ggml_reshape_3d(g, b, 1, 1, p.oc));
    }
    return y;
}

static ggml_tensor * graph_block(context * c, ggml_context * g, ggml_tensor * x, const block & b) {
    ggml_tensor * y = graph_conv(c, g, x, b.dw);
    if (!y) return nullptr;
    if (b.gate.valid) {
        ggml_tensor * pooled = ggml_pool_2d(g, y, GGML_OP_POOL_AVG, y->ne[0], y->ne[1], y->ne[0], y->ne[1], 0, 0);
        ggml_tensor * gate = graph_conv(c, g, pooled, b.gate.c1);
        if (!gate) return nullptr;
        gate = ggml_relu(g, gate);
        gate = graph_conv(c, g, gate, b.gate.c2);
        if (!gate) return nullptr;
        y = ggml_mul(g, y, ggml_hardsigmoid(g, gate));
    }
    ggml_tensor * z = graph_conv(c, g, y, b.cm1);
    if (!z) return nullptr;
    z = ggml_gelu_erf(g, z);
    ggml_tensor * out = graph_conv(c, g, z, b.cm2);
    if (!out) return nullptr;
    if (b.residual) out = ggml_add(g, out, y);
    return out;
}

static bool graph_build(context * c, int h, int w) {
    if (c->graph.attempted) return c->graph.ready && c->graph.h == h && c->graph.w == w;
    c->graph.attempted = true;
    // Graph is the default since 2026-08-04: after the insert-SE double-scale
    // fix it matches the scalar path to probability cosine ~1e-8, runs 186 ms
    // vs 315 ms scalar on synth_00_clean (CPU backend), and the 25-fixture CER
    // gate came out net-better (labelled mean 0.06394 vs 0.06410; receipt
    // 0.0000). CRISPEMBED_PPOCRV6_DET_SCALAR=1 restores the scalar reference.
    // medium's RepLKFPN neck graph validated 2026-08-04: every med_* tap at
    // cosine 0.99999998-1.0 vs scalar on synth + receipt, probability
    // 0.99999999 with equal norms, same box counts, and 6.9 s -> 1.0 s /
    // 41.4 s -> 8.7 s detector time on synth_00_clean / german_official_print
    // (the CPU-scalar medium detector was why the medium tier timed out in
    // every benchmark). German CER graph 0.04856 vs scalar 0.04955.
    if (std::getenv("CRISPEMBED_PPOCRV6_DET_SCALAR")) return false;
    c->graph.backend = c->backend;
    if (!c->graph.backend) return false;
    if (ggml_backend_is_cpu(c->graph.backend)) {
        ggml_backend_t backends[] = { c->graph.backend };
        c->graph.sched = ggml_backend_sched_new(backends, nullptr, 1, 4096, false, false);
    } else {
        c->graph.cpu_backend = ggml_backend_cpu_init();
        // Mirror the init-time thread fix: the sched's CPU fallback should run
        // at the caller's thread count too.
        ggml_backend_cpu_set_n_threads(c->graph.cpu_backend, std::max(1, c->n_threads));
        ggml_backend_t backends[] = { c->graph.backend, c->graph.cpu_backend };
        c->graph.sched = ggml_backend_sched_new(backends, nullptr, 2, 4096, false, false);
    }
    if (!c->graph.sched) return false;
    const size_t meta_size = ggml_tensor_overhead() * 4096 + ggml_graph_overhead_custom(4096, false);
    c->graph.meta.resize(meta_size);
    ggml_init_params ip = { meta_size, c->graph.meta.data(), true };
    c->graph.graph_ctx = ggml_init(ip);
    if (!c->graph.graph_ctx) return false;
    c->graph.graph = ggml_new_graph_custom(c->graph.graph_ctx, 4096, false);
    c->graph.input = ggml_new_tensor_3d(c->graph.graph_ctx, GGML_TYPE_F32, w, h, 3);
    ggml_set_name(c->graph.input, "ppocrv6_det_graph_input");
    ggml_set_input(c->graph.input);
    ggml_tensor * x = graph_conv(c, c->graph.graph_ctx, c->graph.input, c->stem[0]);
    if (!x) return false;
    x = ggml_relu(c->graph.graph_ctx, x);
    ggml_tensor * padded = ggml_pad_ext(c->graph.graph_ctx, x, 0, 1, 0, 1, 0, 0, 0, 0);
    ggml_tensor * branch = graph_conv(c, c->graph.graph_ctx, padded, c->stem[1]);
    if (!branch) return false;
    branch = ggml_relu(c->graph.graph_ctx, branch);
    ggml_tensor * branch_padded = ggml_pad_ext(c->graph.graph_ctx, branch, 0, 1, 0, 1, 0, 0, 0, 0);
    branch = graph_conv(c, c->graph.graph_ctx, branch_padded, c->stem[2]);
    if (!branch) return false;
    branch = ggml_relu(c->graph.graph_ctx, branch);
    ggml_tensor * pooled = ggml_pool_2d(c->graph.graph_ctx, padded, GGML_OP_POOL_MAX, 2, 2, 1, 1, 0, 0);
    x = ggml_concat(c->graph.graph_ctx, pooled, branch, 2);
    x = graph_conv(c, c->graph.graph_ctx, x, c->stem[3]);
    if (!x) return false;
    x = ggml_relu(c->graph.graph_ctx, x);
    x = graph_conv(c, c->graph.graph_ctx, x, c->stem[4]);
    if (!x) return false;
    x = ggml_relu(c->graph.graph_ctx, x);
    std::vector<ggml_tensor *> stage_out(4);
    stage_out[0] = x;
    for (int si = 0; si < 4; ++si) {
        if (si > 0) {
            // The first block of each later stage carries the downsample.
            // The graph is built from the same mapped block list as CPU.
            x = stage_out[si - 1];
        }
        for (const block & b : c->stages[si]) {
            x = graph_block(c, c->graph.graph_ctx, x, b);
            if (!x) return false;
        }
        stage_out[si] = x;
    }
    ggml_tensor * neck = nullptr;
    if (c->variant == "medium") {
        // RepLKFPN-style medium neck, mirroring run_medium_neck: adjust ->
        // top-down FPN -> project -> bottom-up (stride-2 conv, grid-matched
        // by nearest interpolation) -> lateral -> med_ic refinement (reduce,
        // 3x summed vertical/horizontal/symmetric branches, final+ReLU,
        // residual) -> deepest-first concat on stage-0's grid. Tap names
        // match the scalar last_stages so DET_GRAPH_COMPARE diffs per stage.
        ggml_context * g = c->graph.graph_ctx;
        ggml_tensor *adjusted[4], *top[4], *projected[4], *bottom[4], *refined[4];
        for (int i = 0; i < 4; ++i) {
            adjusted[i] = graph_conv(c, g, stage_out[i], c->med_adjust[i]);
            if (!adjusted[i]) return false;
            c->graph.named_taps.push_back({ "med_adjust" + std::to_string(i), adjusted[i] });
        }
        top[3] = adjusted[3];
        for (int i = 2; i >= 0; --i) {
            ggml_tensor * up = ggml_interpolate(g, top[i + 1], adjusted[i]->ne[0], adjusted[i]->ne[1],
                                                adjusted[i]->ne[2], adjusted[i]->ne[3], GGML_SCALE_MODE_NEAREST);
            top[i] = ggml_add(g, adjusted[i], up);
        }
        for (int i = 0; i < 4; ++i) c->graph.named_taps.push_back({ "med_top" + std::to_string(i), top[i] });
        for (int i = 0; i < 4; ++i) {
            projected[i] = graph_conv(c, g, i < 3 ? top[i] : adjusted[3], c->med_project[i]);
            if (!projected[i]) return false;
            c->graph.named_taps.push_back({ "med_project" + std::to_string(i), projected[i] });
        }
        bottom[0] = projected[0];
        for (int i = 1; i < 4; ++i) {
            ggml_tensor * down = graph_conv(c, g, bottom[i - 1], c->med_bottom[i - 1]);
            if (!down) return false;
            down = ggml_interpolate(g, down, projected[i]->ne[0], projected[i]->ne[1], down->ne[2], down->ne[3],
                                    GGML_SCALE_MODE_NEAREST);
            bottom[i] = ggml_add(g, projected[i], down);
        }
        for (int i = 0; i < 4; ++i) c->graph.named_taps.push_back({ "med_bottom" + std::to_string(i), bottom[i] });
        for (int i = 0; i < 4; ++i) {
            ggml_tensor * lat = graph_conv(c, g, i == 0 ? projected[0] : bottom[i], c->med_lateral[i]);
            if (!lat) return false;
            c->graph.named_taps.push_back({ "med_lateral" + std::to_string(i), lat });
            const medium_ic & b = c->med_ic[i];
            ggml_tensor * r = graph_conv(c, g, lat, b.reduce);
            if (!r) return false;
            ggml_tensor * a = r;
            for (int j = 0; j < 3; ++j) {
                ggml_tensor * v = graph_conv(c, g, a, b.vertical[j]);
                ggml_tensor * q = graph_conv(c, g, a, b.horizontal[j]);
                ggml_tensor * s = graph_conv(c, g, a, b.symmetric[j]);
                if (!v || !q || !s) return false;
                a = ggml_add(g, ggml_add(g, v, q), s);
            }
            ggml_tensor * y = graph_conv(c, g, a, b.final);
            if (!y) return false;
            y = ggml_relu(g, y);
            refined[i] = ggml_add(g, lat, y);
            c->graph.named_taps.push_back({ "med_refined" + std::to_string(i), refined[i] });
        }
        neck = ggml_interpolate(g, refined[3], refined[0]->ne[0], refined[0]->ne[1], refined[3]->ne[2],
                                refined[3]->ne[3], GGML_SCALE_MODE_NEAREST);
        for (int i = 2; i >= 1; --i)
            neck = ggml_concat(g, neck,
                               ggml_interpolate(g, refined[i], refined[0]->ne[0], refined[0]->ne[1], refined[i]->ne[2],
                                                refined[i]->ne[3], GGML_SCALE_MODE_NEAREST),
                               2);
        neck = ggml_concat(g, neck, refined[0], 2);
    } else {
        std::vector<ggml_tensor *> fused(4);
        for (int i = 0; i < 4; ++i) {
            fused[i] = graph_conv(c, c->graph.graph_ctx, stage_out[i], c->features[i].insert);
            if (!fused[i]) return false;
            ggml_tensor * pooled = ggml_pool_2d(c->graph.graph_ctx, fused[i], GGML_OP_POOL_AVG, fused[i]->ne[0],
                                                fused[i]->ne[1], fused[i]->ne[0], fused[i]->ne[1], 0, 0);
            ggml_tensor * gate = graph_conv(c, c->graph.graph_ctx, pooled, c->features[i].insert_se1);
            if (!gate) return false;
            gate = ggml_relu(c->graph.graph_ctx, gate);
            gate = graph_conv(c, c->graph.graph_ctx, gate, c->features[i].insert_se2);
            if (!gate) return false;
            // Hard-sigmoid gate: clamp(0.2*x + 0.5), matching the scalar path and
            // Paddle's SELayer. A stray extra ggml_scale(gate, 0.2f) here squashed
            // the gate to 0.04*x + 0.5 — the proc-SE below never had it — and was
            // the whole fused0-cosine-0.988 / 31-vs-30-box geometry divergence.
            gate = ggml_clamp(c->graph.graph_ctx, ggml_scale_bias(c->graph.graph_ctx, gate, 0.2f, 0.5f), 0.0f, 1.0f);
            fused[i] = ggml_add(c->graph.graph_ctx, fused[i], ggml_mul(c->graph.graph_ctx, fused[i], gate));
            c->graph.named_taps.push_back({ "fused" + std::to_string(i), fused[i] });
        }
        for (int i = 2; i >= 0; --i)
            fused[i] = ggml_add(c->graph.graph_ctx, fused[i],
                                ggml_upscale(c->graph.graph_ctx, fused[i + 1], 2, GGML_SCALE_MODE_NEAREST));
        std::vector<ggml_tensor *> proc(4);
        for (int i = 0; i < 4; ++i) {
            ggml_tensor * z = graph_conv(c, c->graph.graph_ctx, fused[i], c->features[i].dw);
            if (!z) return false;
            z = graph_conv(c, c->graph.graph_ctx, z, c->features[i].pw);
            if (!z) return false;
            ggml_tensor * pooled =
                ggml_pool_2d(c->graph.graph_ctx, z, GGML_OP_POOL_AVG, z->ne[0], z->ne[1], z->ne[0], z->ne[1], 0, 0);
            ggml_tensor * gate = graph_conv(c, c->graph.graph_ctx, pooled, c->features[i].se1);
            if (!gate) return false;
            gate = ggml_relu(c->graph.graph_ctx, gate);
            gate = graph_conv(c, c->graph.graph_ctx, gate, c->features[i].se2);
            if (!gate) return false;
            gate = ggml_clamp(c->graph.graph_ctx, ggml_scale_bias(c->graph.graph_ctx, gate, 0.2f, 0.5f), 0.0f, 1.0f);
            proc[i] = ggml_add(c->graph.graph_ctx, z, ggml_mul(c->graph.graph_ctx, z, gate));
            c->graph.named_taps.push_back({ "proc" + std::to_string(i), proc[i] });
        }
        // Match the CPU neck's channel order: deepest feature first, then
        // progressively finer features.  Reversing this concat is critical—the
        // channels have equal shapes, so a mistaken order can look numerically
        // plausible until the detector head produces unrelated logits.
        neck = ggml_upscale(c->graph.graph_ctx, proc[3], 8, GGML_SCALE_MODE_NEAREST);
        neck = ggml_concat(c->graph.graph_ctx, neck,
                           ggml_upscale(c->graph.graph_ctx, proc[2], 4, GGML_SCALE_MODE_NEAREST), 2);
        neck = ggml_concat(c->graph.graph_ctx, neck,
                           ggml_upscale(c->graph.graph_ctx, proc[1], 2, GGML_SCALE_MODE_NEAREST), 2);
        neck = ggml_concat(c->graph.graph_ctx, neck, proc[0], 2);
    }
    c->graph.named_taps.push_back({ "neck_output", neck });
    x = graph_conv(c, c->graph.graph_ctx, neck, c->head_down);
    if (!x) return false;
    x = ggml_relu(c->graph.graph_ctx, x);
    c->graph.named_taps.push_back({ "head_down", x });
    x = graph_deconv(c, c->graph.graph_ctx, x, c->head_up);
    if (!x) return false;
    x = ggml_relu(c->graph.graph_ctx, x);
    c->graph.named_taps.push_back({ "head_up", x });
    x = graph_deconv(c, c->graph.graph_ctx, x, c->head_final);
    if (!x) return false;
    c->graph.named_taps.push_back({ "head_final_pre", x });
    x = ggml_sigmoid(c->graph.graph_ctx, x);
    c->graph.named_taps.push_back({ "head_final", x });
    c->graph.probability_output = true;
    c->graph.output = x;
    c->graph.taps = stage_out;
    if (std::getenv("CRISPEMBED_PPOCRV6_DET_GRAPH_COMPARE")) {
        for (ggml_tensor * tap : c->graph.taps) ggml_set_output(tap);
        for (const auto & tap : c->graph.named_taps) ggml_set_output(tap.second);
    }
    ggml_set_name(x, "ppocrv6_det_graph_stage0");
    ggml_set_output(x);
    ggml_build_forward_expand(c->graph.graph, x);
    ggml_backend_sched_reset(c->graph.sched);
    if (!ggml_backend_sched_alloc_graph(c->graph.sched, c->graph.graph)) return false;
    c->graph.h = h;
    c->graph.w = w;
    c->graph.ready = true;
    fprintf(stderr, "ppocrv6-det: persistent GGML detector graph ready (%s, %dx%d)\n",
            ggml_backend_name(c->graph.backend), w, h);
    return true;
}

static bool graph_run(context * c, const std::vector<float> & input, int h, int w, std::vector<float> & out, int & oh,
                      int & ow) {
    if (!graph_build(c, h, w)) return false;
    ggml_backend_tensor_set(c->graph.input, input.data(), 0, input.size() * sizeof(float));
    if (ggml_backend_sched_graph_compute(c->graph.sched, c->graph.graph) != GGML_STATUS_SUCCESS) return false;
    out.resize(ggml_nelements(c->graph.output));
    ggml_backend_tensor_get(c->graph.output, out.data(), 0, out.size() * sizeof(float));
    if (std::getenv("CRISPEMBED_PPOCRV6_DET_GRAPH_COMPARE")) {
        for (size_t i = 0; i < c->graph.taps.size(); ++i) {
            std::vector<float> tap(ggml_nelements(c->graph.taps[i]));
            ggml_backend_tensor_get(c->graph.taps[i], tap.data(), 0, tap.size() * sizeof(float));
            c->last_stages["graph_stage" + std::to_string(i)] = std::move(tap);
        }
        for (const auto & tap : c->graph.named_taps) {
            std::vector<float> data(ggml_nelements(tap.second));
            ggml_backend_tensor_get(tap.second, data.data(), 0, data.size() * sizeof(float));
            c->last_stages["graph_" + tap.first] = std::move(data);
        }
    }
    ow = (int)c->graph.output->ne[0];
    oh = (int)c->graph.output->ne[1];
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

// Print graph-vs-scalar cosines for the probability map, the four backbone
// stages, and the given named taps. Shared by the small/tiny and medium
// detect tails (medium returns early and used to skip the compare entirely).
static void report_graph_compare(context * c, const std::vector<float> & graph_probability,
                                 const std::vector<float> & y, std::initializer_list<const char *> names) {
    if (graph_probability.empty() || graph_probability.size() != y.size()) return;
    double dot = 0.0, gn = 0.0, cn = 0.0;
    for (size_t i = 0; i < y.size(); ++i) {
        dot += double(graph_probability[i]) * y[i];
        gn += double(graph_probability[i]) * graph_probability[i];
        cn += double(y[i]) * y[i];
    }
    fprintf(stderr, "ppocrv6-det: graph-vs-CPU probability cosine=%.9f graph_norm=%.6g cpu_norm=%.6g\n",
            dot / (std::sqrt(gn) * std::sqrt(cn) + 1e-30), std::sqrt(gn), std::sqrt(cn));
    auto pair_cosine = [&](const std::string & graph_name, const std::string & cpu_name, const char * label) {
        const auto git = c->last_stages.find(graph_name);
        const auto cit = c->last_stages.find(cpu_name);
        if (git == c->last_stages.end() || cit == c->last_stages.end() || git->second.size() != cit->second.size())
            return;
        double sd = 0.0, sg = 0.0, sc = 0.0;
        for (size_t i = 0; i < git->second.size(); ++i) {
            sd += double(git->second[i]) * cit->second[i];
            sg += double(git->second[i]) * git->second[i];
            sc += double(cit->second[i]) * cit->second[i];
        }
        fprintf(stderr, "ppocrv6-det: graph-vs-CPU %s cosine=%.9f\n", label,
                sd / (std::sqrt(sg) * std::sqrt(sc) + 1e-30));
    };
    for (int si = 0; si < 4; ++si)
        pair_cosine("graph_stage" + std::to_string(si), "backbone_stage" + std::to_string(si),
                    ("stage" + std::to_string(si)).c_str());
    for (const char * name : names) pair_cosine(std::string("graph_") + name, name, name);
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
    relu(y);
    for (size_t k = 0; k < y.size(); ++k) y[k] += x[k];
    x.swap(y);
    return true;
}

static bool run_medium_neck(context * c, const std::vector<std::vector<float>> & stages, const std::vector<int> & hs,
                            const std::vector<int> & ws, std::vector<float> & neck, int & nh, int & nw) {
    std::vector<std::vector<float>> adjusted(4), top(4), projected(4), bottom(4), lateral(4), refined(4);
    std::vector<int> ah(4), aw(4);
    for (int i = 0; i < 4; ++i)
        if (!apply_conv(c->med_adjust[i], stages[i], hs[i], ws[i], adjusted[i], ah[i], aw[i])) return false;
    for (int i = 0; i < 4; ++i) c->last_stages["med_adjust" + std::to_string(i)] = adjusted[i];
    top[3] = adjusted[3];
    for (int i = 2; i >= 0; --i) {
        std::vector<float> up;
        resize_nearest(top[i + 1], c->neck, ah[i + 1], aw[i + 1], ah[i], aw[i], up);
        top[i] = adjusted[i];
        add_inplace(top[i], up);
    }
    for (int i = 0; i < 4; ++i) c->last_stages["med_top" + std::to_string(i)] = top[i];
    for (int i = 0; i < 4; ++i) {
        const auto & source = i < 3 ? top[i] : adjusted[3];
        if (!apply_conv(c->med_project[i], source, ah[i], aw[i], projected[i], ah[i], aw[i])) return false;
    }
    for (int i = 0; i < 4; ++i) c->last_stages["med_project" + std::to_string(i)] = projected[i];
    bottom[0] = projected[0];
    for (int i = 1; i < 4; ++i) {
        std::vector<float> down;
        int dh, dw;
        if (!apply_conv(c->med_bottom[i - 1], bottom[i - 1], ah[i - 1], aw[i - 1], down, dh, dw)) return false;
        std::vector<float> resized;
        resize_nearest(down, c->neck / 4, dh, dw, (int)projected[i].size() / (c->neck / 4) / aw[i], aw[i], resized);
        bottom[i] = projected[i];
        add_inplace(bottom[i], resized);
    }
    for (int i = 0; i < 4; ++i) c->last_stages["med_bottom" + std::to_string(i)] = bottom[i];
    for (int i = 0; i < 4; ++i) {
        const auto & source = i == 0 ? projected[0] : bottom[i];
        int sh = ah[i], sw = aw[i];
        if (!apply_conv(c->med_lateral[i], source, sh, sw, lateral[i], sh, sw)) return false;
        refined[i] = lateral[i];
        if (!run_medium_ic(c->med_ic[i], refined[i], sh, sw)) return false;
    }
    for (int i = 0; i < 4; ++i) {
        c->last_stages["med_lateral" + std::to_string(i)] = lateral[i];
        c->last_stages["med_refined" + std::to_string(i)] = refined[i];
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

context * init(const char * path, int n_threads) {
    auto * c = new context();
    c->n_threads = std::max(1, n_threads);
    // The production detector path runs its graph on the CPU backend (default
    // since 2026-08-04; CRISPEMBED_PPOCRV6_DET_SCALAR restores scalar and
    // CRISPEMBED_PPOCRV6_DET_GPU_LOAD is the explicit GPU opt-in), so this
    // backend normally does nothing but pull the GGUF through
    // core_gguf::load_weights, and asking for a GPU one spins up Metal for a
    // device the detector never computes on. That drops the detector's own load
    // from ~7.3 s to 146 ms.
    //
    // It is a smaller net win than that sounds, because the recognizer graph
    // (default since 2026-08-02) needs Metal anyway, so this moves the init
    // rather than removing it. Same-binary A/B over three rounds, median-of-3
    // CPU-seconds against a stable 0.47-0.49 s control: 4.43/3.93/3.57 with the
    // GPU load versus 3.80/3.68/3.04 with the CPU load -- consistently 7-14%
    // better, so it ships on. CRISPEMBED_PPOCRV6_DET_GPU_LOAD restores the old
    // path; the detector graph implies it, since that path genuinely computes
    // on the device.
    //
    // (Measured with CPU time, not wall time, on purpose: this box carries
    // several concurrent agent builds and wall clock swung 10x between runs
    // while user+sys stayed stable. An earlier cross-run wall comparison
    // reported the opposite result.)
    const bool force_cpu = std::getenv("CRISPEMBED_PPOCRV6_FORCE_CPU") != nullptr;
    // The detector graph defaults to the CPU backend: Metal ran the same graph
    // 9x slower (1693 ms vs 187 ms on synth_00_clean) — conv at these spatial
    // sizes does not pay for the dispatch. DET_GPU_LOAD is the explicit GPU
    // opt-in; DET_GRAPH no longer implies it.
    const bool want_gpu = !force_cpu && std::getenv("CRISPEMBED_PPOCRV6_DET_GPU_LOAD") != nullptr;
    c->backend = want_gpu ? crispasr_init_gpu_backend_shared() : ggml_backend_cpu_init();
    if (!c->backend) c->backend = ggml_backend_cpu_init();
    // O13b (2026-08-05): the n_threads parameter was declared but never
    // APPLIED — the signature read `init(const char * path, int)` — so the
    // detector graph always ran at ggml's default thread count no matter what
    // -t the caller passed (measured: medium det graph 7.4 s at -t 1 vs 6.5 s
    // at -t 8 — flat). Honor it. Threading a ggml CPU graph partitions rows
    // per thread without changing any element's reduction order, so detector
    // output is unchanged.
    if (ggml_backend_is_cpu(c->backend)) ggml_backend_cpu_set_n_threads(c->backend, std::max(1, n_threads));
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
    for (auto * buf : c->graph.resident_bufs)
        if (buf) ggml_backend_buffer_free(buf);
    for (auto * wc : c->graph.resident_ctxs)
        if (wc) ggml_free(wc);
    if (c->graph.sched) ggml_backend_sched_free(c->graph.sched);
    if (c->graph.cpu_backend) ggml_backend_free(c->graph.cpu_backend);
    if (c->graph.graph_ctx) ggml_free(c->graph.graph_ctx);
    core_gguf::free_weights(c->wl);
    if (c->backend) crispasr_free_gpu_backend(c->backend);
    delete c;
}

std::vector<box> detect_raw(context * c, const uint8_t * px, int w, int h, int channels, float threshold) {
    if (!c || !px) return {};
    const bool bench = core_env::on("CRISPEMBED_PPOCRV6_DET_BENCH");
    const detprof::report prof_report;
    const auto started = std::chrono::steady_clock::now();
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
    std::vector<float> graph_probability;
    std::vector<box> graph_boxes;
    int H, W;
    const auto preprocessed = std::chrono::steady_clock::now();
    bool graph_done = graph_run(c, input, h, w, x, H, W);
    const auto graph_finished = std::chrono::steady_clock::now();
    const bool compare_graph = std::getenv("CRISPEMBED_PPOCRV6_DET_GRAPH_COMPARE") != nullptr;
    if (graph_done && c->graph.probability_output && compare_graph) {
        graph_probability = x;
        graph_done = false;
    }
    if (graph_done && c->graph.probability_output) {
        c->last_prob = x;
        c->last_h = H;
        c->last_w = W;
        auto native_boxes = ocr_detect::postprocess_probability_map(x.data(), H, W, threshold, 0.60f, 1.4f, 1, 1.0f,
                                                                    1.0f, 0, 3000, ocr_detect::score_mode::fast);
        std::vector<box> out;
        out.reserve(native_boxes.size());
        for (const auto & b : native_boxes) {
            box p{ b.x, b.y, b.w, b.h, b.score };
            std::copy(std::begin(b.qx), std::end(b.qx), std::begin(p.qx));
            std::copy(std::begin(b.qy), std::end(b.qy), std::begin(p.qy));
            out.push_back(p);
        }
        const float sx = float(w) / W, sy = float(h) / H;
        for (auto & b : out) {
            b.x *= sx;
            b.w *= sx;
            b.y *= sy;
            b.h *= sy;
            for (int i = 0; i < 4; ++i) {
                b.qx[i] *= sx;
                b.qy[i] *= sy;
            }
        }
        if (compare_graph) graph_boxes = out;
        // Graph boxes are accepted by default (validated 2026-08-04, see
        // graph_build). Compare mode still falls through to the scalar
        // reference so the diagnostic diff has both sides;
        // CRISPEMBED_PPOCRV6_DET_SCALAR disables the graph entirely upstream.
        if (!out.empty() && !compare_graph) {
            if (bench) {
                const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
                fprintf(stderr,
                        "[ppocrv6-det-bench] preprocess_ms=%.3f graph_ms=%.3f total_ms=%.3f boxes=%zu accepted=1\n",
                        ms(started, preprocessed), ms(preprocessed, graph_finished),
                        ms(started, std::chrono::steady_clock::now()), out.size());
            }
            return out;
        }
        fprintf(stderr, "ppocrv6-det: graph probability map is diagnostic-only; using CPU reference\n");
        graph_done = false;
    }
    if (!graph_done && !run_stem(c, input, h, w, x, H, W)) return {};
    std::vector<std::vector<float>> stages;
    std::vector<int> hs, ws;
    if (graph_done) {
        stages.push_back(x);
        hs.push_back(H);
        ws.push_back(W);
    }
    const size_t first_stage = graph_done ? 1 : 0;
    for (size_t si = first_stage; si < c->stages.size(); ++si) {
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
        c->last_stages["head_down"] = y;
        if (!apply_deconv2(c->head_up, y, oh, ow, z, oh, ow)) return {};
        c->last_stages["head_up_pre"] = z;
        relu(z);
        c->last_stages["head_up"] = z;
        if (!apply_deconv2(c->head_final, z, oh, ow, y, oh, ow)) return {};
        c->last_stages["head_final_pre"] = y;
        c->last_stages["head_final"] = y;
        c->last_stages["neck_output"] = neck;
        for (float & v : y) v = 1 / (1 + std::exp(-v));
        c->last_prob = y;
        c->last_h = oh;
        c->last_w = ow;
        // The f16 backbone has a small positive probability bias versus
        // PaddleX (the map parity gate remains above 0.997 on the fixtures).
        // Calibrate the DB box score while retaining the official 0.2 bitmap
        // threshold; this restores region counts without changing logits.
        auto native_boxes = ocr_detect::postprocess_probability_map(y.data(), oh, ow, threshold, 0.60f, 1.4f, 1, 1.0f,
                                                                    1.0f, 0, 3000, ocr_detect::score_mode::fast);
        std::vector<box> out;
        out.reserve(native_boxes.size());
        for (const auto & b : native_boxes) {
            box p{ b.x, b.y, b.w, b.h, b.score };
            std::copy(std::begin(b.qx), std::end(b.qx), std::begin(p.qx));
            std::copy(std::begin(b.qy), std::end(b.qy), std::begin(p.qy));
            out.push_back(p);
        }
        float sx = float(w) / ow, sy = float(h) / oh;
        for (auto & b : out) {
            b.x *= sx;
            b.w *= sx;
            b.y *= sy;
            b.h *= sy;
            for (int i = 0; i < 4; ++i) {
                b.qx[i] *= sx;
                b.qy[i] *= sy;
            }
        }
        if (compare_graph) report_graph_box_geometry(graph_boxes, out);
        report_graph_compare(c, graph_probability, y,
                             { "med_adjust0",  "med_adjust1",  "med_adjust2",   "med_adjust3",  "med_top0",
                               "med_top1",     "med_top2",     "med_top3",      "med_project0", "med_project1",
                               "med_project2", "med_project3", "med_bottom0",   "med_bottom1",  "med_bottom2",
                               "med_bottom3",  "med_lateral0", "med_lateral1",  "med_lateral2", "med_lateral3",
                               "med_refined0", "med_refined1", "med_refined2",  "med_refined3", "neck_output",
                               "head_down",    "head_up",      "head_final_pre" });
        if (bench) {
            const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            fprintf(stderr, "[ppocrv6-det-bench] preprocess_ms=%.3f graph_ms=%.3f total_ms=%.3f boxes=%zu accepted=0\n",
                    ms(started, preprocessed), ms(preprocessed, graph_finished),
                    ms(started, std::chrono::steady_clock::now()), out.size());
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
        c->last_stages["fused" + std::to_string(i)] = fused[i];
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
        c->last_stages["proc" + std::to_string(i)] = proc[i];
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
    c->last_stages["head_down"] = y;
    if (!apply_deconv2(c->head_up, y, oh, ow, z, oh, ow)) return {};
    c->last_stages["head_up_pre"] = z;
    relu(z);
    c->last_stages["head_up"] = z;
    if (!apply_deconv2(c->head_final, z, oh, ow, y, oh, ow)) return {};
    c->last_stages["head_final_pre"] = y;
    c->last_stages["head_final"] = y;
    for (float & v : y) v = 1 / (1 + std::exp(-v));
    c->last_prob = y;
    c->last_h = oh;
    c->last_w = ow;
    // See the medium path above: use the calibrated DB box score for the
    // compact model's native probability map.
    auto native_boxes = ocr_detect::postprocess_probability_map(y.data(), oh, ow, threshold, 0.60f, 1.4f, 1, 1.0f, 1.0f,
                                                                0, 3000, ocr_detect::score_mode::fast);
    std::vector<box> out;
    out.reserve(native_boxes.size());
    for (const auto & b : native_boxes) {
        box p{ b.x, b.y, b.w, b.h, b.score };
        std::copy(std::begin(b.qx), std::end(b.qx), std::begin(p.qx));
        std::copy(std::begin(b.qy), std::end(b.qy), std::begin(p.qy));
        out.push_back(p);
    }
    float sx = float(w) / ow, sy = float(h) / oh;
    for (auto & b : out) {
        b.x *= sx;
        b.w *= sx;
        b.y *= sy;
        b.h *= sy;
        for (int i = 0; i < 4; ++i) {
            b.qx[i] *= sx;
            b.qy[i] *= sy;
        }
    }
    if (compare_graph) report_graph_box_geometry(graph_boxes, out);
    report_graph_compare(c, graph_probability, y,
                         { "fused0", "fused1", "fused2", "fused3", "proc0", "proc1", "proc2", "proc3", "neck_output",
                           "head_down", "head_up", "head_final_pre" });
    if (bench) {
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        fprintf(stderr, "[ppocrv6-det-bench] preprocess_ms=%.3f graph_ms=%.3f total_ms=%.3f boxes=%zu accepted=0\n",
                ms(started, preprocessed), ms(preprocessed, graph_finished),
                ms(started, std::chrono::steady_clock::now()), out.size());
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
    int rw = std::max(32, int(std::floor(w * scale / 32.0f + 0.5f)) * 32);
    int rh = std::max(32, int(std::floor(h * scale / 32.0f + 0.5f)) * 32);
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
        for (int i = 0; i < 4; ++i) {
            b.qx[i] *= float(w) / rw;
            b.qy[i] *= float(h) / rh;
        }
    }
    stbi_image_free(p);
    return r;
}
} // namespace ppocrv6_det
