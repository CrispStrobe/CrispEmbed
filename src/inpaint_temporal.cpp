// Temporally consistent inpainting of a frame sequence (see inpaint_temporal.h).
#include "inpaint_temporal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <vector>

namespace inpaint_temporal {
namespace {

constexpr int ring_radius = 24; // ring used by the static tests
constexpr int flat_radius = 6;  // known pixels that decide the flat fill
constexpr int cell = 8;         // static test cell size
constexpr int strip = 64;       // flat decision per strip of the hole's width
constexpr float flat_low = 4.0f, flat_high = 8.0f;
constexpr double harmonic_budget = 2e7; // SOR pixel updates per frame (bounds the flat fill's cost)

struct box {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    bool empty() const { return x1 <= x0; }
    int width() const { return x1 - x0; }
    bool operator==(const box & o) const { return x0 == o.x0 && y0 == o.y0 && x1 == o.x1 && y1 == o.y1; }
};

struct frame {
    std::vector<uint8_t> rgb;
    std::vector<uint8_t> mask; // 0/1, bounds-sized (empty without a mask)
    box bounds;
    bool masked(int x, int y) const {
        return x >= bounds.x0 && x < bounds.x1 && y >= bounds.y0 && y < bounds.y1 &&
               mask[size_t(y - bounds.y0) * bounds.width() + x - bounds.x0];
    }
};

using output = std::shared_ptr<const std::vector<uint8_t>>;

float luma(const uint8_t * p) {
    return 0.299f * p[0] + 0.587f * p[1] + 0.114f * p[2];
}

} // namespace

struct stream::state {
    int w = 0, h = 0;
    settings opt;
    fill_fn fill;
    std::deque<frame> buf; // frames [base, base + buf.size())
    int64_t base = 0, pushed = 0, next = 0;
    bool finished = false;
    std::string error;
    output prev_out; // output of frame next - 1 (shared with `ready`, not copied)
    std::deque<output> ready;
    stats st;

    // Per-frame scratch (region-local except mask255/engine_out).
    box region, region_harm;
    std::vector<int> integral, unknown, span;
    std::vector<uint8_t> ring, near, mask255, engine_out;
    std::vector<float> diffs, values, strip_weight, harm;

    const frame * get(int64_t i) const {
        return i < base || i >= base + int64_t(buf.size()) ? nullptr : &buf[size_t(i - base)];
    }

    // ring / near: unmasked pixels of the region within ring_radius / flat_radius
    // of a masked pixel (box sums of an integral image; no mask lies outside).
    void make_rings(const frame & f) {
        const int W = region.width(), H = region.y1 - region.y0;
        integral.assign(size_t(W + 1) * (H + 1), 0);
        for (int y = 0; y < H; y++) {
            int row = 0;
            for (int x = 0; x < W; x++) {
                row += f.masked(x + region.x0, y + region.y0);
                integral[size_t(y + 1) * (W + 1) + x + 1] = integral[size_t(y) * (W + 1) + x + 1] + row;
            }
        }
        auto sum = [&](int x, int y, int r) {
            const int xa = std::max(0, x - r), xb = std::min(W, x + r + 1);
            const int ya = std::max(0, y - r), yb = std::min(H, y + r + 1);
            return integral[size_t(yb) * (W + 1) + xb] - integral[size_t(ya) * (W + 1) + xb] -
                   integral[size_t(yb) * (W + 1) + xa] + integral[size_t(ya) * (W + 1) + xa];
        };
        ring.assign(size_t(W) * H, 0);
        near.assign(size_t(W) * H, 0);
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                if (f.masked(x + region.x0, y + region.y0) || !sum(x, y, ring_radius)) continue;
                ring[size_t(y) * W + x] = 1;
                near[size_t(y) * W + x] = sum(x, y, flat_radius) > 0;
            }
    }

    // Static test of the ring between two frames (see the header). mean/p90 are -1
    // when there are too few usable cells to judge.
    bool ring_static(const frame & a, const frame & b, double & mean, double & p90) {
        diffs.clear();
        const int W = region.width();
        for (int cy = region.y0; cy < region.y1; cy += cell)
            for (int cx = region.x0; cx < region.x1; cx += cell) {
                int n = 0, sa[3] = { 0, 0, 0 }, sb[3] = { 0, 0, 0 };
                for (int y = cy; y < std::min(region.y1, cy + cell); y++)
                    for (int x = cx; x < std::min(region.x1, cx + cell); x++) {
                        if (!ring[size_t(y - region.y0) * W + x - region.x0] || b.masked(x, y)) continue;
                        const size_t i = (size_t(y) * w + x) * 3;
                        for (int c = 0; c < 3; c++) {
                            sa[c] += a.rgb[i + c];
                            sb[c] += b.rgb[i + c];
                        }
                        n++;
                    }
                if (n >= 8)
                    diffs.push_back(float(std::abs(sa[0] - sb[0]) + std::abs(sa[1] - sb[1]) + std::abs(sa[2] - sb[2])) /
                                    (3.0f * n));
            }
        mean = p90 = -1;
        if (diffs.size() < 8) return false;
        double total = 0;
        for (float d : diffs) total += d;
        mean = total / double(diffs.size());
        std::nth_element(diffs.begin(), diffs.begin() + long(diffs.size() * 9 / 10), diffs.end());
        p90 = diffs[diffs.size() * 9 / 10];
        return mean <= opt.static_threshold && p90 <= 2.0 * opt.static_threshold;
    }

    // Flat-fill weight per strip of the mask's width: 1 at trimmed (10..90 %) luma
    // std <= flat_low of the near pixels, 0 from flat_high. Returns the largest.
    float strip_weights(const frame & f) {
        const box & b = f.bounds;
        const int n = (b.width() + strip - 1) / strip, W = region.width();
        strip_weight.assign(size_t(n), 0.0f);
        float best = 0;
        for (int s = 0; s < n; s++) {
            const int xa = std::max(region.x0, b.x0 + s * strip - flat_radius);
            const int xb = std::min(region.x1, b.x0 + (s + 1) * strip + flat_radius);
            values.clear();
            for (int y = region.y0; y < region.y1; y++)
                for (int x = xa; x < xb; x++)
                    if (near[size_t(y - region.y0) * W + x - region.x0])
                        values.push_back(luma(&f.rgb[(size_t(y) * w + x) * 3]));
            if (values.size() < 32) continue;
            const size_t lo = values.size() / 10, hi = values.size() * 9 / 10;
            std::nth_element(values.begin(), values.begin() + long(lo), values.end());
            std::nth_element(values.begin() + long(lo), values.begin() + long(hi), values.end());
            double mean = 0, var = 0;
            for (size_t i = lo; i < hi; i++) mean += values[i];
            mean /= double(hi - lo);
            for (size_t i = lo; i < hi; i++) var += (values[i] - mean) * (values[i] - mean);
            const float sd = float(std::sqrt(var / double(hi - lo)));
            strip_weight[size_t(s)] = std::clamp((flat_high - sd) / (flat_high - flat_low), 0.0f, 1.0f);
            best = std::max(best, strip_weight[size_t(s)]);
        }
        return best;
    }

    // Weight at column x, linear between strip centres (no seams between strips).
    float flat_weight(const frame & f, int x) const {
        const float p = (x - f.bounds.x0 + 0.5f) / strip - 0.5f;
        const int n = int(strip_weight.size()), i = std::clamp(int(std::floor(p)), 0, n - 1);
        const int j = std::min(n - 1, i + 1);
        const float t = std::clamp(p - i, 0.0f, 1.0f);
        return strip_weight[size_t(i)] * (1 - t) + strip_weight[size_t(j)] * t;
    }

    // Harmonic (Laplace) fill of the mask from img over the mask's bounds plus a 1 px
    // border. The start is the row and column linear interpolations, weighted towards
    // the shorter span (close to harmonic for subtitle-shaped holes); SOR then settles
    // it within a fixed per-frame work budget.
    void harmonic(const frame & f, const std::vector<uint8_t> & img) {
        const box e{ std::max(0, f.bounds.x0 - 1), std::max(0, f.bounds.y0 - 1), std::min(w, f.bounds.x1 + 1),
                     std::min(h, f.bounds.y1 + 1) };
        const int W = e.width(), H = e.y1 - e.y0;
        region_harm = e;
        harm.resize(size_t(W) * H * 3);
        span.assign(size_t(W) * H, 0);
        unknown.clear();
        auto known = [&](int x, int y) { return !f.masked(x + e.x0, y + e.y0); };
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                if (known(x, y))
                    for (int c = 0; c < 3; c++)
                        harm[(size_t(y) * W + x) * 3 + c] = img[(size_t(y + e.y0) * w + x + e.x0) * 3 + c];
        // Linear interpolation of the unknown runs along a line of n pixels (start
        // `first`, `step` apart). The vertical pass blends into the horizontal one.
        auto interpolate = [&](int first, int step, int n, bool vertical) {
            int last = -1;
            for (int k = 0; k <= n; k++) {
                const int i = first + k * step;
                const bool is_known = k == n || known(i % W, i / W);
                if (!is_known) continue;
                const int len = k - last - 1;
                for (int u = last + 1; u < k; u++) {
                    const int p = first + u * step;
                    const float t = float(u - last) / float(k - last);
                    for (int c = 0; c < 3; c++) {
                        const float l = last >= 0 ? harm[size_t(first + last * step) * 3 + c] : -1;
                        const float r = k < n ? harm[size_t(i) * 3 + c] : -1;
                        const float v = l < 0 ? std::max(r, 0.0f) : r < 0 ? l : l + (r - l) * t;
                        float & out = harm[size_t(p) * 3 + c];
                        // Weight each direction by the other's span: shorter span wins.
                        out = vertical ? (out * len + v * span[size_t(p)]) / float(len + span[size_t(p)]) : v;
                    }
                    if (!vertical) {
                        span[size_t(p)] = len;
                        unknown.push_back(p);
                    }
                }
                last = k;
            }
        };
        for (int y = 0; y < H; y++) interpolate(y * W, 1, W, false);
        for (int x = 0; x < W; x++) interpolate(x, W, H, true);
        constexpr float omega = 1.8f;
        const int sweeps = unknown.empty() ? 0 : int(std::clamp(harmonic_budget / double(unknown.size()), 20.0, 500.0));
        for (int iteration = 0; iteration < sweeps; iteration++) {
            float change = 0;
            for (int i : unknown) {
                const int x = i % W, y = i / W;
                const size_t l = size_t(x > 0 ? i - 1 : i), r = size_t(x + 1 < W ? i + 1 : i);
                const size_t u = size_t(y > 0 ? i - W : i), d = size_t(y + 1 < H ? i + W : i);
                for (int c = 0; c < 3; c++) {
                    float & v = harm[size_t(i) * 3 + c];
                    const float target =
                        0.25f * (harm[l * 3 + c] + harm[r * 3 + c] + harm[u * 3 + c] + harm[d * 3 + c]);
                    const float delta = omega * (target - v);
                    v += delta;
                    change = std::max(change, std::fabs(delta));
                }
            }
            if (change < 0.05f) break;
        }
    }

    // Steps 3 and 4 for the pixels of f's mask that are still open in img.
    bool fill_rest(const frame & f, std::vector<uint8_t> & img, bool same, double ring_mean, double ring_p90) {
        const box & b = f.bounds;
        const bool flat = opt.flat_fill && strip_weights(f) > 0;
        if (flat) harmonic(f, img);
        float lowest = flat ? 1.0f : 0.0f;
        if (flat)
            for (int x = b.x0; x < b.x1 && lowest > 0; x++) lowest = std::min(lowest, flat_weight(f, x));
        const bool engine = lowest < 1.0f;
        if (engine) {
            mask255.resize(size_t(w) * h);
            engine_out.resize(img.size());
            for (int y = b.y0; y < b.y1; y++)
                for (int x = b.x0; x < b.x1; x++) mask255[size_t(y) * w + x] = f.masked(x, y) ? 255 : 0;
            const bool ok = fill(img.data(), mask255.data(), engine_out.data(), error);
            for (int y = b.y0; y < b.y1; y++) std::memset(&mask255[size_t(y) * w + b.x0], 0, size_t(b.width()));
            if (!ok) {
                if (error.empty()) error = "inpainting engine failed";
                return false;
            }
            st.frames_engine++;
        }
        // Temporal blend weight for engine pixels (same mask, moderate ring change).
        float blend = 0;
        if (engine && same && ring_mean >= 0 && opt.blend_threshold > opt.static_threshold &&
            ring_mean < opt.blend_threshold && ring_p90 < opt.blend_threshold) {
            const double t =
                std::max(0.0, (ring_mean - opt.static_threshold) / (opt.blend_threshold - opt.static_threshold));
            blend = float(0.6 * (1.0 - t));
        }
        for (int y = b.y0; y < b.y1; y++)
            for (int x = b.x0; x < b.x1; x++) {
                if (!f.masked(x, y)) continue;
                const size_t i = size_t(y) * w + x;
                const float wf = flat ? flat_weight(f, x) : 0.0f;
                for (int c = 0; c < 3; c++) {
                    float v = engine ? engine_out[i * 3 + c] : 0.0f;
                    if (wf > 0) {
                        const size_t k = (size_t(y - region_harm.y0) * region_harm.width() + x - region_harm.x0) * 3;
                        v = wf * harm[k + c] + (1 - wf) * v;
                    }
                    if (blend > 0 && wf < 1) v = v * (1 - blend) + (*prev_out)[i * 3 + c] * blend;
                    img[i * 3 + c] = uint8_t(std::clamp(v + 0.5f, 0.0f, 255.0f));
                }
                if (wf >= 1)
                    st.px_flat++;
                else {
                    st.px_engine++;
                    st.px_blended += blend > 0;
                }
            }
        return true;
    }

    bool process(int64_t t) {
        const frame & f = *get(t);
        std::vector<uint8_t> img = f.rgb;
        st.frames++;
        if (!f.bounds.empty()) {
            const box & b = f.bounds;
            st.frames_masked++;
            region = { std::max(0, b.x0 - ring_radius), std::max(0, b.y0 - ring_radius),
                       std::min(w, b.x1 + ring_radius), std::min(h, b.y1 + ring_radius) };
            make_rings(f);
            int64_t open = 0;
            for (uint8_t v : f.mask) open += v;
            st.px_masked += open;
            auto copy_from = [&](const std::vector<uint8_t> & source) {
                for (int y = b.y0; y < b.y1; y++)
                    for (int x = b.x0; x < b.x1; x++)
                        if (f.masked(x, y))
                            std::memcpy(&img[(size_t(y) * w + x) * 3], &source[(size_t(y) * w + x) * 3], 3);
            };

            // 1. Real background from the nearest frame with no mask (stricter than the
            // reuse test: frames are further apart and slow motion would be pasted).
            for (int d = 1; d <= opt.window && open; d++)
                for (int sign = -1; sign <= 1 && open; sign += 2) {
                    const frame * k = get(t + sign * d);
                    double mean, p90;
                    if (!k || !k->bounds.empty() || !ring_static(f, *k, mean, p90) ||
                        mean > 0.5 * opt.static_threshold || p90 > opt.static_threshold)
                        continue;
                    copy_from(k->rgb);
                    st.px_propagated += open;
                    open = 0;
                }

            // 2. The same subtitle as the previous frame on a static ring: reuse its output.
            const frame * prev = opt.window > 0 && prev_out ? get(t - 1) : nullptr;
            const bool same = prev && prev->bounds == b && prev->mask == f.mask;
            double ring_mean = -1, ring_p90 = -1;
            if (open && same && ring_static(f, *prev, ring_mean, ring_p90)) {
                copy_from(*prev_out);
                st.px_reused += open;
                open = 0;
            }

            // 3-4. Flat fill and the engine, with everything filled so far as context.
            if (open && !fill_rest(f, img, same, ring_mean, ring_p90)) return false;
        }
        prev_out = std::make_shared<const std::vector<uint8_t>>(std::move(img));
        ready.push_back(prev_out);
        return true;
    }

    bool advance() {
        while (error.empty() && next < pushed && (finished || pushed - 1 >= next + opt.window)) {
            if (!process(next)) return false;
            next++;
            // Keep `window` frames behind (at least 1 for the static-span test).
            while (!buf.empty() && base < next - std::max(1, opt.window)) {
                buf.pop_front();
                base++;
            }
        }
        return error.empty();
    }
};

stream::stream(int width, int height, const settings & options, fill_fn fill) : s_(new state) {
    s_->w = std::max(1, width);
    s_->h = std::max(1, height);
    s_->opt = options;
    // 2 * window + 1 buffered frames must fit the memory budget.
    const size_t frame_bytes = size_t(s_->w) * s_->h * 3;
    const size_t fit = std::max<size_t>(1, options.max_buffer_bytes / frame_bytes);
    s_->opt.window = int(std::min<size_t>(size_t(std::max(0, options.window)), (fit - 1) / 2));
    s_->fill = std::move(fill);
}

stream::~stream() = default;

int stream::window() const {
    return s_->opt.window;
}

bool stream::push(const uint8_t * rgb, const uint8_t * mask, std::string & error) {
    state & s = *s_;
    if (!s.error.empty() || s.finished) {
        error = s.error.empty() ? "stream already finished" : s.error;
        return false;
    }
    const int W = s.w, H = s.h;
    frame f;
    f.rgb.assign(rgb, rgb + size_t(W) * H * 3);
    if (mask) {
        box b{ W, H, 0, 0 };
        for (int y = 0; y < H; y++) {
            const uint8_t * row = mask + size_t(y) * W;
            int first = 0, last = W - 1;
            while (first < W && row[first] < 128) first++;
            if (first == W) continue;
            while (row[last] < 128) last--;
            b = { std::min(b.x0, first), std::min(b.y0, y), std::max(b.x1, last + 1), std::max(b.y1, y + 1) };
        }
        if (!b.empty()) {
            f.bounds = b;
            f.mask.resize(size_t(b.width()) * (b.y1 - b.y0));
            for (int y = b.y0; y < b.y1; y++)
                for (int x = b.x0; x < b.x1; x++)
                    f.mask[size_t(y - b.y0) * b.width() + x - b.x0] = mask[size_t(y) * W + x] >= 128;
        }
    }
    s.buf.push_back(std::move(f));
    s.pushed++;
    if (!s.advance()) {
        error = s.error;
        return false;
    }
    return true;
}

bool stream::finish(std::string & error) {
    s_->finished = true;
    if (!s_->advance()) {
        error = s_->error;
        return false;
    }
    return true;
}

bool stream::pop(uint8_t * out) {
    if (s_->ready.empty()) return false;
    std::memcpy(out, s_->ready.front()->data(), s_->ready.front()->size());
    s_->ready.pop_front();
    return true;
}

const stats & stream::counters() const {
    return s_->st;
}

} // namespace inpaint_temporal
