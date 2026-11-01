#include "inpaint_graph_spectral.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace inpaint_graph {
namespace {

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr int64_t max_fft_axis = 128;
constexpr int64_t gla_chunk = 64;

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(std::string("inpainting spectral graph: ") + message);
}

struct ComplexTensor {
    ggml_tensor * re;
    ggml_tensor * im;
};

bool power_of_two(int64_t n) {
    return n > 0 && (n & (n - 1)) == 0;
}

ggml_tensor * contiguous(ggml_context * ctx, ggml_tensor * x) {
    return ggml_is_contiguous(x) ? x : ggml_cont(ctx, x);
}

ggml_tensor * matmul(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * result = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}

ComplexTensor multiply(ggml_context * ctx, ComplexTensor a, ComplexTensor b) {
    return { ggml_sub(ctx, ggml_mul(ctx, a.re, b.re), ggml_mul(ctx, a.im, b.im)),
             ggml_add(ctx, ggml_mul(ctx, a.re, b.im), ggml_mul(ctx, a.im, b.re)) };
}

// Gather along the FFT axis with one GET_ROWS for both complex components.
// The only host-supplied values are fixed geometry indices, not activations.
ComplexTensor gather_axis0(Builder & builder, ComplexTensor x, const std::string & name,
                           const std::vector<int32_t> & indices) {
    ggml_context * ctx = builder.ctx;
    const int64_t columns = x.re->ne[1], selected = int64_t(indices.size());
    require(selected > 0 && x.re->ne[2] == 1 && x.re->ne[3] == 1, "invalid complex gather geometry");
    ggml_tensor * both = ggml_concat(ctx, x.re, x.im, 1); // [N,2*columns]
    both = ggml_cont(ctx, ggml_transpose(ctx, both));     // [2*columns,N]
    ggml_tensor * index = builder.constant_i32(name, { selected }, indices);
    both = ggml_get_rows(ctx, both, index);
    both = ggml_cont(ctx, ggml_transpose(ctx, both)); // [selected,2*columns]
    return { ggml_view_2d(ctx, both, selected, columns, size_t(selected) * sizeof(float), 0),
             ggml_view_2d(ctx, both, selected, columns, size_t(selected) * sizeof(float),
                          size_t(selected * columns) * sizeof(float)) };
}

// Unnormalized forward transform of columns stored as [N,columns]. Every
// butterfly is a view, vector operation or concat executed by the backend.
ComplexTensor radix2(Builder & builder, ComplexTensor x) {
    ggml_context * ctx = builder.ctx;
    const int64_t n = x.re->ne[0], columns = x.re->ne[1];
    require(power_of_two(n) && n <= 256 && x.re->type == GGML_TYPE_F32 && x.im->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(x.re, x.im),
            "invalid radix-2 complex tensor");
    if (n == 1) return x;
    if (n > 2) {
        std::vector<int32_t> reverse(size_t(n), 0);
        for (int64_t i = 0; i < n; ++i) {
            int64_t value = i, reversed = 0;
            for (int64_t bit = n >> 1; bit; bit >>= 1) {
                reversed = (reversed << 1) | (value & 1);
                value >>= 1;
            }
            reverse[size_t(i)] = int32_t(reversed);
        }
        x = gather_axis0(builder, x, "spectral.fft.bit_reverse." + std::to_string(n), reverse);
    }
    for (int64_t length = 2; length <= n; length <<= 1) {
        const int64_t half = length / 2, groups = n / length;
        const size_t row_stride = size_t(length) * sizeof(float), column_stride = size_t(n) * sizeof(float);
        auto part = [&](ggml_tensor * component, size_t offset) {
            return ggml_view_3d(ctx, component, half, groups, columns, row_stride, column_stride, offset);
        };
        ComplexTensor even{ part(x.re, 0), part(x.im, 0) };
        ComplexTensor odd{ part(x.re, size_t(half) * sizeof(float)), part(x.im, size_t(half) * sizeof(float)) };
        std::vector<float> cosine(size_t(half), 0), sine(size_t(half), 0);
        for (int64_t k = 0; k < half; ++k) {
            const double phase = -2.0 * pi * double(k) / double(length);
            cosine[size_t(k)] = float(std::cos(phase));
            sine[size_t(k)] = float(std::sin(phase));
        }
        const std::string key = "spectral.fft.twiddle." + std::to_string(length);
        ComplexTensor twiddle{ builder.constant(key + ".re", { half }, cosine),
                               builder.constant(key + ".im", { half }, sine) };
        odd = multiply(ctx, odd, twiddle);
        x.re = ggml_reshape_2d(ctx, ggml_concat(ctx, ggml_add(ctx, even.re, odd.re), ggml_sub(ctx, even.re, odd.re), 0),
                               n, columns);
        x.im = ggml_reshape_2d(ctx, ggml_concat(ctx, ggml_add(ctx, even.im, odd.im), ggml_sub(ctx, even.im, odd.im), 0),
                               n, columns);
    }
    return x;
}

ComplexTensor inverse_radix2(Builder & builder, ComplexTensor x) {
    const float scale = 1.0f / float(x.re->ne[0]);
    x.im = ggml_neg(builder.ctx, x.im);
    x = radix2(builder, x);
    return { ggml_scale(builder.ctx, x.re, scale), ggml_scale(builder.ctx, x.im, -scale) };
}

// A Bluestein convolution kernel depends only on integer transform length.
// This small host calculation initializes constant graph parameters. It never
// receives, computes on, downloads or uploads a model activation.
std::vector<std::complex<double>> bluestein_kernel(int64_t n, int64_t padded) {
    std::vector<std::complex<double>> values(size_t(padded), { 0, 0 });
    for (int64_t k = 0; k < n; ++k) {
        const double phase = pi * double((k * k) % (2 * n)) / double(n);
        values[size_t(k)] = std::polar(1.0, phase);
        if (k) values[size_t(padded - k)] = values[size_t(k)];
    }
    for (size_t i = 1, j = 0; i < values.size(); ++i) {
        size_t bit = values.size() >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(values[i], values[j]);
    }
    for (size_t length = 2; length <= values.size(); length <<= 1) {
        const auto unit = std::polar(1.0, -2.0 * pi / double(length));
        for (size_t start = 0; start < values.size(); start += length) {
            std::complex<double> phase(1, 0);
            for (size_t k = 0; k < length / 2; ++k) {
                const auto even = values[start + k], odd = values[start + k + length / 2] * phase;
                values[start + k] = even + odd;
                values[start + k + length / 2] = even - odd;
                phase *= unit;
            }
        }
    }
    return values;
}

ComplexTensor forward_fft(Builder & builder, ComplexTensor x) {
    const int64_t n = x.re->ne[0], columns = x.re->ne[1];
    require(n >= 1 && n <= max_fft_axis, "FFT axis exceeds [1,128]");
    if (power_of_two(n)) return radix2(builder, x);
    int64_t padded = 1;
    while (padded < 2 * n - 1) padded <<= 1;
    std::vector<float> chirp_re(size_t(n), 0), chirp_im(size_t(n), 0);
    for (int64_t k = 0; k < n; ++k) {
        const double phase = -pi * double((k * k) % (2 * n)) / double(n);
        chirp_re[size_t(k)] = float(std::cos(phase));
        chirp_im[size_t(k)] = float(std::sin(phase));
    }
    const std::string key = "spectral.fft.bluestein." + std::to_string(n);
    ComplexTensor chirp{ builder.constant(key + ".chirp.re", { n }, chirp_re),
                         builder.constant(key + ".chirp.im", { n }, chirp_im) };
    const auto kernel = bluestein_kernel(n, padded);
    std::vector<float> kernel_re(size_t(padded), 0), kernel_im(size_t(padded), 0);
    for (int64_t k = 0; k < padded; ++k) {
        kernel_re[size_t(k)] = float(kernel[size_t(k)].real());
        kernel_im[size_t(k)] = float(kernel[size_t(k)].imag());
    }
    ComplexTensor transformed_kernel{ builder.constant(key + ".kernel.re", { padded }, kernel_re),
                                      builder.constant(key + ".kernel.im", { padded }, kernel_im) };
    ggml_context * ctx = builder.ctx;
    x = multiply(ctx, x, chirp);
    x = { ggml_pad(ctx, x.re, int(padded - n), 0, 0, 0), ggml_pad(ctx, x.im, int(padded - n), 0, 0, 0) };
    x = radix2(builder, x);
    x = inverse_radix2(builder, multiply(ctx, x, transformed_kernel));
    x = { ggml_view_2d(ctx, x.re, n, columns, size_t(padded) * sizeof(float), 0),
          ggml_view_2d(ctx, x.im, n, columns, size_t(padded) * sizeof(float), 0) };
    return multiply(ctx, x, chirp);
}

ComplexTensor fft_axis(Builder & builder, ComplexTensor x, int axis, bool inverse) {
    ggml_context * ctx = builder.ctx;
    require(axis == 0 || axis == 1, "only spatial FFT axes are supported");
    if (axis == 1) {
        x = { ggml_cont(ctx, ggml_transpose(ctx, x.re)), ggml_cont(ctx, ggml_transpose(ctx, x.im)) };
    }
    const int64_t n = x.re->ne[0], other = x.re->ne[1], channels = x.re->ne[2], batch = x.re->ne[3];
    x = { ggml_reshape_2d(ctx, contiguous(ctx, x.re), n, other * channels * batch),
          ggml_reshape_2d(ctx, contiguous(ctx, x.im), n, other * channels * batch) };
    if (inverse) x.im = ggml_neg(ctx, x.im);
    x = forward_fft(builder, x);
    if (inverse) x = { ggml_scale(ctx, x.re, 1.0f / float(n)), ggml_scale(ctx, x.im, -1.0f / float(n)) };
    x = { ggml_reshape_4d(ctx, x.re, n, other, channels, batch),
          ggml_reshape_4d(ctx, x.im, n, other, channels, batch) };
    if (axis == 1) x = { ggml_cont(ctx, ggml_transpose(ctx, x.re)), ggml_cont(ctx, ggml_transpose(ctx, x.im)) };
    return x;
}

void valid_image(ggml_tensor * x, bool packed) {
    require(x && x->type == GGML_TYPE_F32 && x->ne[0] >= 1 && x->ne[0] <= max_fft_axis && x->ne[1] >= 1 &&
                x->ne[1] <= max_fft_axis && x->ne[2] >= 1 && x->ne[3] >= 1 && x->ne[2] <= 8192 && x->ne[3] <= 8 &&
                x->ne[2] * x->ne[3] <= 8192 && (!packed || x->ne[2] % 2 == 0),
            "invalid F32 WHCN FFT geometry");
}

ggml_tensor * token_chunk(ggml_context * ctx, ggml_tensor * x, int64_t begin, int64_t length) {
    return ggml_cont(ctx,
                     ggml_view_3d(ctx, x, length, x->ne[1], x->ne[2], x->nb[1], x->nb[2], size_t(begin) * x->nb[0]));
}

ggml_tensor * concatenate_chunks(ggml_context * ctx, std::vector<ggml_tensor *> parts) {
    require(!parts.empty(), "empty GLA token sequence");
    // A balanced tree avoids copying an ever-growing prefix once per chunk.
    while (parts.size() > 1) {
        std::vector<ggml_tensor *> next;
        next.reserve((parts.size() + 1) / 2);
        for (size_t i = 0; i < parts.size(); i += 2)
            next.push_back(i + 1 < parts.size() ? ggml_concat(ctx, parts[i], parts[i + 1], 1) : parts[i]);
        parts = std::move(next);
    }
    return parts.front();
}

} // namespace

ggml_tensor * rfft2_ortho(Builder & builder, ggml_tensor * real) {
    valid_image(real, false);
    ggml_context * ctx = builder.ctx;
    real = contiguous(ctx, real);
    const int64_t width = real->ne[0], height = real->ne[1], channels = real->ne[2], batch = real->ne[3];
    const int64_t half_width = width / 2 + 1;
    ComplexTensor x = fft_axis(builder, { real, ggml_scale(ctx, real, 0.0f) }, 0, false);
    auto half = [&](ggml_tensor * component) {
        return ggml_cont(ctx, ggml_view_4d(ctx, component, half_width, height, channels, batch, component->nb[1],
                                           component->nb[2], component->nb[3], 0));
    };
    x = fft_axis(builder, { half(x.re), half(x.im) }, 1, false);
    ggml_tensor * both = ggml_concat(ctx, ggml_reshape_4d(ctx, x.re, half_width * height, 1, channels, batch),
                                     ggml_reshape_4d(ctx, x.im, half_width * height, 1, channels, batch), 1);
    both = ggml_reshape_4d(ctx, both, half_width, height, 2 * channels, batch);
    return ggml_scale(ctx, both, float(1.0 / std::sqrt(double(width * height))));
}

ggml_tensor * irfft2_ortho(Builder & builder, ggml_tensor * packed, int original_width) {
    valid_image(packed, true);
    require(original_width >= 1 && original_width <= max_fft_axis && packed->ne[0] == original_width / 2 + 1,
            "packed FFT width differs from the explicit real width");
    ggml_context * ctx = builder.ctx;
    packed = contiguous(ctx, packed);
    const int64_t half_width = packed->ne[0], height = packed->ne[1], channels = packed->ne[2] / 2;
    const int64_t batch = packed->ne[3], width = original_width;
    auto component = [&](size_t offset) {
        return ggml_cont(ctx, ggml_view_4d(ctx, packed, half_width, height, channels, batch, packed->nb[1],
                                           2 * packed->nb[2], packed->nb[3], offset));
    };
    ComplexTensor x{ component(0), component(packed->nb[2]) };
    x = fft_axis(builder, x, 1, true);
    if (width > half_width) {
        const int64_t columns = height * channels * batch;
        ComplexTensor flat{ ggml_reshape_2d(ctx, x.re, half_width, columns),
                            ggml_reshape_2d(ctx, x.im, half_width, columns) };
        std::vector<int32_t> reverse(size_t(width - half_width), 0);
        for (int64_t i = 0; i < width - half_width; ++i) reverse[size_t(i)] = int32_t(width - half_width - i);
        ComplexTensor mirror = gather_axis0(builder, flat, "spectral.fft.hermitian." + std::to_string(width), reverse);
        x = { ggml_reshape_4d(ctx, ggml_concat(ctx, flat.re, mirror.re, 0), width, height, channels, batch),
              ggml_reshape_4d(ctx, ggml_concat(ctx, flat.im, ggml_neg(ctx, mirror.im), 0), width, height, channels,
                              batch) };
    }
    x = fft_axis(builder, x, 0, true);
    return ggml_scale(ctx, x.re, float(std::sqrt(double(width * height))));
}

ggml_tensor * gla(Builder & builder, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, ggml_tensor * gate_logits,
                  int heads, float normalizer) {
    require(q && k && v && gate_logits && q->type == GGML_TYPE_F32 && k->type == GGML_TYPE_F32 &&
                v->type == GGML_TYPE_F32 && gate_logits->type == GGML_TYPE_F32 && ggml_are_same_shape(q, k) &&
                ggml_are_same_shape(k, gate_logits) && q->ne[0] == v->ne[0] && q->ne[1] == v->ne[1] &&
                q->ne[3] == v->ne[3] && heads > 0 && heads <= 32 && q->ne[2] % heads == 0 && v->ne[2] % heads == 0 &&
                q->ne[0] > 0 && q->ne[1] > 0 && q->ne[0] * q->ne[1] <= 4096 && q->ne[3] >= 1 && q->ne[3] <= 8 &&
                std::isfinite(normalizer) && normalizer > 0 && std::isfinite(1.0f / normalizer),
            "invalid aligned rectangular GLA tensors or normalizer");
    const int64_t width = q->ne[0], height = q->ne[1], tokens = width * height, batch = q->ne[3];
    const int64_t dk = q->ne[2] / heads, dv = v->ne[2] / heads, head_batches = heads * batch;
    require(dk >= 1 && dk <= 256 && dv >= 1 && dv <= 256, "GLA head dimensions exceed the bounded profile");
    ggml_context * ctx = builder.ctx;
    q = ggml_scale(ctx, contiguous(ctx, q), 1.0f / std::sqrt(float(dk)));
    ggml_tensor * log_gate =
        ggml_scale(ctx, ggml_softplus(ctx, ggml_neg(ctx, contiguous(ctx, gate_logits))), -1.0f / normalizer);
    // exp(-120) is already zero in F32. Keeping log prefixes finite preserves
    // the recurrence's zero-decay behavior for extreme finite negative logits
    // and prevents (-inf)-(-inf) in the causal pairwise differences.
    log_gate = ggml_clamp(ctx, log_gate, -120.0f, 0.0f);
    q = ggml_reshape_3d(ctx, q, tokens, dk, head_batches);
    k = ggml_reshape_3d(ctx, contiguous(ctx, k), tokens, dk, head_batches);
    v = ggml_reshape_3d(ctx, contiguous(ctx, v), tokens, dv, head_batches);
    log_gate = ggml_reshape_3d(ctx, log_gate, tokens, dk, head_batches);
    ggml_tensor * state = nullptr; // Independent zero state for every invocation/head/batch.
    std::vector<ggml_tensor *> outputs;
    for (int64_t begin = 0; begin < tokens; begin += gla_chunk) {
        const int64_t length = std::min(gla_chunk, tokens - begin);
        ggml_tensor * qc = ggml_cont(ctx, ggml_transpose(ctx, token_chunk(ctx, q, begin, length)));
        ggml_tensor * kc = ggml_cont(ctx, ggml_transpose(ctx, token_chunk(ctx, k, begin, length)));
        ggml_tensor * vc = token_chunk(ctx, v, begin, length); // [length,dv,head_batches]
        ggml_tensor * prefix_tokens = ggml_cumsum(ctx, token_chunk(ctx, log_gate, begin, length));
        ggml_tensor * prefix = ggml_cont(ctx, ggml_transpose(ctx, prefix_tokens)); // [dk,length,HB]
        ggml_tensor * query_prefix = ggml_reshape_4d(ctx, prefix, dk, 1, length, head_batches);
        ggml_tensor * key_prefix = ggml_reshape_4d(ctx, prefix, dk, length, 1, head_batches);
        ggml_tensor * difference =
            ggml_sub(ctx, ggml_repeat_4d(ctx, query_prefix, dk, length, length, head_batches), key_prefix);
        // Future entries are discarded below; clamp them BEFORE exp so no
        // overflow/NaN is created and subsequently multiplied by a zero mask.
        ggml_tensor * decay = ggml_exp(ctx, ggml_clamp(ctx, difference, -std::numeric_limits<float>::max(), 0.0f));
        ggml_tensor * query = ggml_reshape_4d(ctx, qc, dk, 1, length, head_batches);
        ggml_tensor * key = ggml_reshape_4d(ctx, kc, dk, length, 1, head_batches);
        ggml_tensor * products = ggml_mul(ctx, ggml_repeat_4d(ctx, query, dk, length, length, head_batches), key);
        products = ggml_mul(ctx, products, decay);
        ggml_tensor * attention = ggml_reshape_3d(ctx, ggml_sum_rows(ctx, products), length, length, head_batches);
        std::vector<float> causal(size_t(length * length), 0);
        for (int64_t t = 0; t < length; ++t)
            for (int64_t s = 0; s <= t; ++s) causal[size_t(s + t * length)] = 1;
        attention =
            ggml_mul(ctx, attention,
                     builder.constant("spectral.gla.causal." + std::to_string(length), { length, length }, causal));
        ggml_tensor * output = matmul(ctx, vc, attention); // [dv,length,HB]
        if (state) output = ggml_add(ctx, output, matmul(ctx, state, ggml_mul(ctx, qc, ggml_exp(ctx, prefix))));
        outputs.push_back(output);

        if (begin + length < tokens) {
            ggml_tensor * last =
                ggml_cont(ctx, ggml_view_3d(ctx, prefix_tokens, 1, dk, head_batches, prefix_tokens->nb[1],
                                            prefix_tokens->nb[2], size_t(length - 1) * sizeof(float)));
            last = ggml_reshape_3d(ctx, last, dk, 1, head_batches);
            ggml_tensor * remaining = ggml_sub(ctx, ggml_repeat(ctx, last, prefix), prefix);
            remaining = ggml_exp(ctx, ggml_clamp(ctx, remaining, -std::numeric_limits<float>::max(), 0.0f));
            ggml_tensor * weighted_key = ggml_cont(ctx, ggml_transpose(ctx, ggml_mul(ctx, kc, remaining)));
            ggml_tensor * update = matmul(ctx, weighted_key, vc); // [dk,dv,HB]
            state = state ? ggml_add(ctx, ggml_mul(ctx, state, ggml_exp(ctx, last)), update) : update;
        }
    }
    ggml_tensor * result = ggml_cont(ctx, ggml_transpose(ctx, concatenate_chunks(ctx, std::move(outputs))));
    return ggml_reshape_4d(ctx, result, width, height, dv * heads, batch);
}

} // namespace inpaint_graph
