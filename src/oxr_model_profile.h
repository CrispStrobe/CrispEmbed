// Pure OXR-1.0 preprocessing and request contracts. The reference is the
// pinned SoMarkAI/OXR-1.0 custom modeling_oxr.py and vLLM processor plugin.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace oxr_profile {
constexpr const char * family = "oxr-1.0";
constexpr const char * revision = "c708407f9d2b4df658f2543d9132b62bb7221df4";
constexpr const char * gate = "CRISPEMBED_OXR";
constexpr int context_length = 32768;
constexpr int min_pixels = 200704;
constexpr int vllm_max_pixels = 921600;
constexpr int hf_max_pixels = 2408448;
constexpr int visual_atom = -300;
constexpr int image_begin = -301;
constexpr int image_end = -302;
constexpr int vocab = 151936;
constexpr int real_vocab = 151669;
constexpr int visual_vocab = 65536;
constexpr int visual_logits = visual_vocab - 4;
constexpr int patch = 16;
constexpr int stride = 2;

inline int64_t round_even(double value) {
    const double lo = std::floor(value);
    const double frac = value - lo;
    return int64_t(lo + (frac > 0.5 || (frac == 0.5 && std::fmod(lo, 2.0) != 0.0)));
}

inline std::pair<int, int> smart_resize(int h, int w, int lo = min_pixels, int hi = vllm_max_pixels) {
    if (h <= 0 || w <= 0 || lo <= 0 || hi < lo || hi > hf_max_pixels)
        throw std::invalid_argument("OXR invalid dimensions or pixel limits");
    constexpr int factor = patch * stride;
    int64_t height = h, width = w;
    if (height < factor || width < factor) {
        if (height < width) {
            width = round_even(double(factor) / double(height) * double(width));
            height = factor;
        } else {
            height = round_even(double(factor) / double(width) * double(height));
            width = factor;
        }
    } else if (double(std::max(height, width)) / double(std::min(height, width)) > 200.0) {
        if (height > width)
            height = 200 * width;
        else
            width = 200 * height;
    }
    int64_t rh = round_even(double(height) / factor) * factor;
    int64_t rw = round_even(double(width) / factor) * factor;
    if (double(rh) * double(rw) > hi) {
        const double beta = std::sqrt(double(height) * double(width) / hi);
        rh = int64_t(std::floor(double(height) / beta / factor)) * factor;
        rw = int64_t(std::floor(double(width) / beta / factor)) * factor;
    } else if (double(rh) * double(rw) < lo) {
        const double beta = std::sqrt(double(lo) / (double(height) * double(width)));
        rh = int64_t(std::ceil(double(height) * beta / factor)) * factor;
        rw = int64_t(std::ceil(double(width) * beta / factor)) * factor;
    }
    if (rh <= 0 || rw <= 0 || rh > 65536 || rw > 65536 || rh * rw > int64_t(hi) + int64_t(65536) * 64)
        throw std::invalid_argument("OXR smart_resize produced unsupported dimensions");
    return { int(rh), int(rw) };
}

// Pillow's bilinear 8-bit path: double coefficients, 22-bit quantization,
// clipped/rounded uint8 intermediate between horizontal and vertical passes.
struct pil_axis {
    int taps = 0;
    std::vector<int> first, count;
    std::vector<int32_t> weights;
};
inline pil_axis bilinear_axis(int src, int dst) {
    pil_axis a;
    const double scale = double(src) / dst, filter = std::max(1.0, scale);
    a.taps = int(std::ceil(filter)) * 2 + 1;
    a.first.resize(dst);
    a.count.resize(dst);
    a.weights.resize(size_t(dst) * a.taps);
    std::vector<double> weights(a.taps);
    for (int i = 0; i < dst; ++i) {
        const double center = (double(i) + 0.5) * scale;
        const int first = std::max(0, int(center - filter + 0.5));
        const int end = std::min(src, int(center + filter + 0.5));
        const int count = std::max(0, end - first);
        double sum = 0.0;
        for (int j = 0; j < count; ++j) {
            const double x = std::fabs((double(j + first) - center + 0.5) / filter);
            sum += weights[j] = std::max(0.0, 1.0 - x);
        }
        for (int j = 0; j < count; ++j)
            a.weights[size_t(i) * a.taps + j] = int32_t(0.5 + (weights[j] / sum) * double(1 << 22));
        a.first[i] = first;
        a.count[i] = count;
    }
    return a;
}
inline uint8_t clip8(int64_t x) {
    return uint8_t(std::clamp<int64_t>(x >> 22, 0, 255));
}

inline std::vector<uint8_t> resize_rgb_pil_bilinear(const uint8_t * src, int w, int h, int channels, int rw, int rh) {
    if (!src || w <= 0 || h <= 0 || rw <= 0 || rh <= 0 || (channels != 1 && channels != 3 && channels != 4))
        throw std::invalid_argument("OXR expects nonempty gray, RGB or RGBA pixels");
    const pil_axis ax = bilinear_axis(w, rw), ay = bilinear_axis(h, rh);
    std::vector<uint8_t> mid(size_t(h) * rw * 3), out(size_t(rh) * rw * 3);
    constexpr int64_t half = int64_t(1) << 21;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < rw; ++x)
            for (int c = 0; c < 3; ++c) {
                int64_t sum = half;
                for (int j = 0; j < ax.count[x]; ++j)
                    sum += int64_t(ax.weights[size_t(x) * ax.taps + j]) *
                           src[(size_t(y) * w + ax.first[x] + j) * channels + (channels == 1 ? 0 : c)];
                mid[(size_t(y) * rw + x) * 3 + c] = clip8(sum);
            }
    for (int y = 0; y < rh; ++y)
        for (int x = 0; x < rw; ++x)
            for (int c = 0; c < 3; ++c) {
                int64_t sum = half;
                for (int j = 0; j < ay.count[y]; ++j)
                    sum +=
                        int64_t(ay.weights[size_t(y) * ay.taps + j]) * mid[(size_t(ay.first[y] + j) * rw + x) * 3 + c];
                out[(size_t(y) * rw + x) * 3 + c] = clip8(sum);
            }
    return out;
}

struct image_input {
    std::vector<float> patches;
    int32_t grid[3] = { 1, 0, 0 };
    int width = 0, height = 0;
};
inline image_input preprocess(const uint8_t * src, int w, int h, int channels, int lo, int hi) {
    const auto dims = smart_resize(h, w, lo, hi);
    image_input result;
    result.height = dims.first;
    result.width = dims.second;
    result.grid[1] = result.height / patch;
    result.grid[2] = result.width / patch;
    const auto rgb = resize_rgb_pil_bilinear(src, w, h, channels, result.width, result.height);
    result.patches.resize(size_t(result.height) * result.width * 3);
    size_t index = 0;
    for (int by = 0; by < result.grid[1] / stride; ++by)
        for (int bx = 0; bx < result.grid[2] / stride; ++bx)
            for (int sy = 0; sy < stride; ++sy)
                for (int sx = 0; sx < stride; ++sx)
                    for (int c = 0; c < 3; ++c)
                        for (int y = 0; y < patch; ++y)
                            for (int x = 0; x < patch; ++x) {
                                const int py = (by * stride + sy) * patch + y;
                                const int px = (bx * stride + sx) * patch + x;
                                // HF rescale uses float64 before cast to float32;
                                // normalization then acts on the float32 value.
                                const float scaled =
                                    float(double(rgb[(size_t(py) * result.width + px) * 3 + c]) * (1.0 / 255.0));
                                result.patches[index++] = (scaled - 0.5f) / 0.5f;
                            }
    return result;
}

inline std::vector<int32_t> window_groups(int h, int w) {
    if (h <= 0 || w <= 0 || h % 2 || w % 2) throw std::invalid_argument("invalid OXR patch grid");
    std::vector<int32_t> result;
    result.reserve(size_t(h) * w / 4);
    // 112 // hidden_stride(2) // patch(16) == 3 merged tokens.
    for (int y = 0; y < h / 2; y += 3)
        for (int x = 0; x < w / 2; x += 3)
            for (int dy = 0; dy < 3 && y + dy < h / 2; ++dy)
                for (int dx = 0; dx < 3 && x + dx < w / 2; ++dx) result.push_back((y + dy) * (w / 2) + x + dx);
    return result;
}

inline void cubic_coeff(float t, float (&c)[4]) {
    constexpr float a = -0.75f;
    const auto cubic1 = [](float x) { return ((a + 2.0f) * x - (a + 3.0f)) * x * x + 1.0f; };
    const auto cubic2 = [](float x) { return ((a * x - 5.0f * a) * x + 8.0f * a) * x - 4.0f * a; };
    c[0] = cubic2(t + 1.0f);
    c[1] = cubic1(t);
    c[2] = cubic1(1.0f - t);
    c[3] = cubic2(2.0f - t);
}

// F.interpolate(mode='bicubic',align_corners=False,antialias=False) on the
// learned 32x32 position grid, followed by the model's 2x2 merge-major order.
inline std::vector<float> position_embeddings(const float * table, int h, int w, int dim = 1152) {
    std::vector<float> out(size_t(h) * w * dim);
    const float sh = 32.0f / h, sw = 32.0f / w;
    for (int y = 0; y < h; ++y) {
        const float fy = sh * (float(y) + 0.5f) - 0.5f;
        const int iy = int(std::floor(fy));
        float cy[4];
        cubic_coeff(fy - float(iy), cy);
        for (int x = 0; x < w; ++x) {
            const float fx = sw * (float(x) + 0.5f) - 0.5f;
            const int ix = int(std::floor(fx));
            float cx[4];
            cubic_coeff(fx - float(ix), cx);
            const int row = ((y / 2) * (w / 2) + x / 2) * 4 + (y % 2) * 2 + x % 2;
            for (int c = 0; c < dim; ++c) {
                float rows[4];
                for (int j = 0; j < 4; ++j) {
                    const int sy = std::clamp(iy - 1 + j, 0, 31);
                    rows[j] = 0.0f;
                    for (int k = 0; k < 4; ++k) {
                        const int sx = std::clamp(ix - 1 + k, 0, 31);
                        rows[j] += table[(size_t(sy) * 32 + sx) * dim + c] * cx[k];
                    }
                }
                out[size_t(row) * dim + c] = rows[0] * cy[0] + rows[1] * cy[1] + rows[2] * cy[2] + rows[3] * cy[3];
            }
        }
    }
    return out;
}

inline bool eos(int32_t token) {
    return token == 151643 || token == 151645;
}
inline bool special(int32_t token) {
    return token >= 151643 && token <= 151656;
}
inline int32_t vllm_penalty_id(int32_t token) {
    if (token == image_begin) return 151652;
    if (token == image_end) return 151653;
    return token == visual_atom ? 151655 : token;
}
inline void repetition_penalty(std::vector<float> & logits, const std::vector<int32_t> & history, float penalty) {
    if (!std::isfinite(penalty) || penalty <= 0.0f) throw std::invalid_argument("invalid repetition penalty");
    if (penalty == 1.0f) return;
    std::vector<uint8_t> seen(logits.size(), 0);
    for (int32_t id : history)
        if (id >= 0 && size_t(id) < logits.size() && !seen[size_t(id)]) {
            seen[size_t(id)] = 1;
            float & value = logits[size_t(id)];
            value = value < 0.0f ? value * penalty : value / penalty;
        }
}
} // namespace oxr_profile
