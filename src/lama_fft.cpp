#include "lama_fft.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <vector>

namespace lama_fft {
namespace {

using complex = std::complex<double>;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr int error_invalid = 1;
constexpr int error_allocation = 2;
constexpr int error_nonfinite = 3;
constexpr int error_internal = 4;
constexpr size_t max_axis = 8192;
constexpr size_t max_plane = 1024 * 1024;
constexpr size_t max_elements = 64 * 1024 * 1024;
constexpr size_t scratch_budget = 128 * 1024 * 1024;

void fail(execution_status * status, int code) noexcept {
    if (!status) return;
    int expected = 0;
    status->code.compare_exchange_strong(expected, code, std::memory_order_relaxed);
}

bool shape_ok(int64_t width, int64_t height, int64_t channels, int64_t batch) noexcept {
    if (width < 1 || height < 1 || channels < 1 || batch < 1 || uint64_t(width) > max_axis ||
        uint64_t(height) > max_axis)
        return false;
    const size_t area = size_t(width) * size_t(height);
    if (area > max_plane || uint64_t(channels) > max_elements / area) return false;
    return uint64_t(batch) <= max_elements / (area * size_t(channels));
}

bool power_of_two(size_t n) noexcept {
    return (n & (n - 1)) == 0;
}

size_t convolution_length(size_t n) noexcept {
    size_t result = 1;
    while (result < 2 * n - 1) result *= 2;
    return result;
}

// In-place radix-2 Cooley-Tukey. The inverse here includes 1/n, solely for
// Bluestein's convolution; fft_plan applies unnormalized transforms in either
// direction and the 2D caller applies 1/sqrt(width*height) exactly once.
void radix2(complex * data, size_t n, bool inverse) noexcept {
    for (size_t i = 1, reversed = 0; i < n; ++i) {
        size_t bit = n >> 1;
        while (reversed & bit) {
            reversed ^= bit;
            bit >>= 1;
        }
        reversed ^= bit;
        if (i < reversed) std::swap(data[i], data[reversed]);
    }
    for (size_t block = 2; block <= n; block *= 2) {
        const double angle = (inverse ? 2.0 : -2.0) * pi / double(block);
        const complex step(std::cos(angle), std::sin(angle));
        for (size_t start = 0; start < n; start += block) {
            complex phase(1.0, 0.0);
            for (size_t j = 0; j < block / 2; ++j) {
                const complex even = data[start + j];
                const complex odd = data[start + j + block / 2] * phase;
                data[start + j] = even + odd;
                data[start + j + block / 2] = even - odd;
                phase *= step;
            }
        }
    }
    if (inverse)
        for (size_t i = 0; i < n; ++i) data[i] /= double(n);
}

class fft_plan {
public:
    explicit fft_plan(size_t n) : n_(n), length_(power_of_two(n) ? 0 : convolution_length(n)) {
        if (!length_) return;
        chirp_.resize(n_);
        kernel_.assign(length_, complex(0.0, 0.0));
        for (size_t i = 0; i < n_; ++i) {
            // Reducing i^2 modulo 2n keeps trigonometric arguments well bounded.
            const double angle = pi * double((uint64_t(i) * i) % (2 * n_)) / double(n_);
            chirp_[i] = complex(std::cos(angle), -std::sin(angle));
            kernel_[i] = std::conj(chirp_[i]);
            if (i) kernel_[length_ - i] = kernel_[i];
        }
        radix2(kernel_.data(), length_, false);
    }

    size_t work_size() const noexcept { return length_; }

    void apply(complex * data, bool inverse, std::vector<complex> & work) const noexcept {
        if (inverse)
            for (size_t i = 0; i < n_; ++i) data[i] = std::conj(data[i]);
        if (!length_) {
            radix2(data, n_, false);
        } else {
            std::fill(work.begin(), work.begin() + length_, complex(0.0, 0.0));
            for (size_t i = 0; i < n_; ++i) work[i] = data[i] * chirp_[i];
            radix2(work.data(), length_, false);
            for (size_t i = 0; i < length_; ++i) work[i] *= kernel_[i];
            radix2(work.data(), length_, true);
            for (size_t i = 0; i < n_; ++i) data[i] = work[i] * chirp_[i];
        }
        if (inverse)
            for (size_t i = 0; i < n_; ++i) data[i] = std::conj(data[i]);
    }

private:
    size_t n_, length_;
    std::vector<complex> chirp_, kernel_;
};

class transform {
public:
    transform(size_t width, size_t height)
        : width_(width), height_(height), rows_(width), columns_(height), plane_(width * height), column_(height),
          work_(std::max(rows_.work_size(), columns_.work_size())) {}

    template <typename Read, typename Write> bool forward(Read read, Write write) {
        for (size_t y = 0; y < height_; ++y)
            for (size_t x = 0; x < width_; ++x) {
                const float value = read(x, y, false);
                if (!std::isfinite(value)) return false;
                plane_[y * width_ + x] = complex(value, 0.0);
            }
        compute(false);
        const double scale = 1.0 / std::sqrt(double(width_ * height_));
        for (size_t y = 0; y < height_; ++y)
            for (size_t x = 0; x <= width_ / 2; ++x) {
                const complex value = plane_[y * width_ + x] * scale;
                if (!std::isfinite(value.real()) || !std::isfinite(value.imag()) ||
                    std::abs(value.real()) > std::numeric_limits<float>::max() ||
                    std::abs(value.imag()) > std::numeric_limits<float>::max())
                    return false;
                const float real = float(value.real()), imag = float(value.imag());
                write(x, y, false, real);
                write(x, y, true, imag);
            }
        return true;
    }

    template <typename Read, typename Write> bool inverse(Read read, Write write) {
        for (size_t y = 0; y < height_; ++y)
            for (size_t x = 0; x <= width_ / 2; ++x) {
                const float real = read(x, y, false), imag = read(x, y, true);
                if (!std::isfinite(real) || !std::isfinite(imag)) return false;
                plane_[y * width_ + x] = complex(real, imag);
            }
        // Negative horizontal frequencies come from the conjugated *opposite
        // vertical frequency*. Mirroring only within a row is incorrect in 2D.
        for (size_t y = 0; y < height_; ++y)
            for (size_t x = width_ / 2 + 1; x < width_; ++x)
                plane_[y * width_ + x] = std::conj(plane_[((height_ - y) % height_) * width_ + width_ - x]);
        compute(true);
        const double scale = 1.0 / std::sqrt(double(width_ * height_));
        for (size_t y = 0; y < height_; ++y)
            for (size_t x = 0; x < width_; ++x) {
                // irfftn returns real data. In particular, arbitrary imaginary
                // DC/Nyquist values introduced by the spectral 1x1 convolution
                // must be discarded through this real projection.
                const double value = plane_[y * width_ + x].real() * scale;
                if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max()) return false;
                write(x, y, false, float(value));
            }
        return true;
    }

private:
    void compute(bool inverse) noexcept {
        for (size_t y = 0; y < height_; ++y) rows_.apply(plane_.data() + y * width_, inverse, work_);
        for (size_t x = 0; x < width_; ++x) {
            for (size_t y = 0; y < height_; ++y) column_[y] = plane_[y * width_ + x];
            columns_.apply(column_.data(), inverse, work_);
            for (size_t y = 0; y < height_; ++y) plane_[y * width_ + x] = column_[y];
        }
    }

    size_t width_, height_;
    fft_plan rows_, columns_;
    std::vector<complex> plane_, column_, work_;
};

// A conservative upper bound including both chirp/kernel plans, one shared
// convolution buffer, the spatial plane and column buffer. It also covers
// power-of-two dimensions, whose plans actually need no chirp/kernel storage.
size_t workspace_bytes(size_t width, size_t height) noexcept {
    return sizeof(complex) *
           (width * height + width + 2 * height + convolution_length(width) + convolution_length(height) +
            std::max(convolution_length(width), convolution_length(height)));
}

float tensor_read(const ggml_tensor * tensor, size_t x, size_t y, size_t channel, size_t batch) noexcept {
    float value;
    const char * address = static_cast<const char *>(tensor->data) + x * tensor->nb[0] + y * tensor->nb[1] +
                           channel * tensor->nb[2] + batch * tensor->nb[3];
    std::memcpy(&value, address, sizeof(value));
    return value;
}

void tensor_write(ggml_tensor * tensor, size_t x, size_t y, size_t channel, size_t batch, float value) noexcept {
    char * address = static_cast<char *>(tensor->data) + x * tensor->nb[0] + y * tensor->nb[1] +
                     channel * tensor->nb[2] + batch * tensor->nb[3];
    std::memcpy(address, &value, sizeof(value));
}

void nan_plane(ggml_tensor * tensor, size_t channel, size_t batch, bool packed) noexcept {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const size_t first = packed ? 2 * channel : channel;
    for (size_t c = first; c < first + (packed ? 2 : 1); ++c)
        for (size_t y = 0; y < size_t(tensor->ne[1]); ++y)
            for (size_t x = 0; x < size_t(tensor->ne[0]); ++x) tensor_write(tensor, x, y, c, batch, nan);
}

template <bool Inverse> void compute_node(ggml_tensor * dst, int ith, int nth, void * opaque) noexcept {
    auto * status = static_cast<execution_status *>(opaque);
    const ggml_tensor * src = dst->src[0];
    const size_t width = size_t(Inverse ? dst->ne[0] : src->ne[0]);
    const size_t height = size_t(dst->ne[1]);
    const size_t channels = size_t(Inverse ? dst->ne[2] : src->ne[2]);
    const size_t batch = size_t(dst->ne[3]);
    const size_t planes = channels * batch;
    const size_t workers =
        std::max<size_t>(1, std::min({ size_t(nth), planes, scratch_budget / workspace_bytes(width, height) }));
    if (size_t(ith) >= workers) return;
    try {
        transform operation(width, height);
        for (size_t index = size_t(ith); index < planes; index += workers) {
            const size_t channel = index % channels, batch_index = index / channels;
            auto read = [&](size_t x, size_t y, bool imaginary) {
                return tensor_read(src, x, y, Inverse ? 2 * channel + size_t(imaginary) : channel, batch_index);
            };
            auto write = [&](size_t x, size_t y, bool imaginary, float value) {
                tensor_write(dst, x, y, Inverse ? channel : 2 * channel + size_t(imaginary), batch_index, value);
            };
            const bool ok = Inverse ? operation.inverse(read, write) : operation.forward(read, write);
            if (!ok) {
                fail(status, error_nonfinite);
                nan_plane(dst, channel, batch_index, !Inverse);
            }
        }
    } catch (const std::bad_alloc &) {
        fail(status, error_allocation);
        for (size_t index = size_t(ith); index < planes; index += workers)
            nan_plane(dst, index % channels, index / channels, !Inverse);
    } catch (...) {
        fail(status, error_internal);
        for (size_t index = size_t(ith); index < planes; index += workers)
            nan_plane(dst, index % channels, index / channels, !Inverse);
    }
}

template <bool Inverse>
bool contiguous(const float * input, size_t input_count, float * output, size_t output_count, int64_t width,
                int64_t height, int64_t channels, int64_t batch, std::string & error) {
    error.clear();
    if (!input || !output || !shape_ok(width, height, channels, batch)) {
        error = "LaMa FFT: invalid dimensions or null storage";
        return false;
    }
    const size_t w = size_t(width), h = size_t(height), c = size_t(channels), b = size_t(batch);
    const size_t real_plane = w * h, packed_plane = (w / 2 + 1) * h;
    const size_t real_count = real_plane * c * b, packed_count = 2 * packed_plane * c * b;
    const size_t needed_input = Inverse ? packed_count : real_count;
    const size_t needed_output = Inverse ? real_count : packed_count;
    if (input_count < needed_input || output_count < needed_output) {
        error = "LaMa FFT: buffer is smaller than the declared shape";
        return false;
    }
    const uintptr_t input_start = reinterpret_cast<uintptr_t>(input);
    const uintptr_t output_start = reinterpret_cast<uintptr_t>(output);
    const bool overlap = input_start <= output_start ? output_start - input_start < needed_input * sizeof(float)
                                                     : input_start - output_start < needed_output * sizeof(float);
    if (overlap) {
        error = "LaMa FFT: input and output storage overlap";
        return false;
    }
    try {
        transform operation(w, h);
        for (size_t plane = 0; plane < c * b; ++plane) {
            auto read = [&](size_t x, size_t y, bool imaginary) {
                const size_t offset = Inverse ? (2 * plane + size_t(imaginary)) * packed_plane + y * (w / 2 + 1) + x
                                              : plane * real_plane + y * w + x;
                return input[offset];
            };
            auto write = [&](size_t x, size_t y, bool imaginary, float value) {
                const size_t offset = Inverse ? plane * real_plane + y * w + x
                                              : (2 * plane + size_t(imaginary)) * packed_plane + y * (w / 2 + 1) + x;
                output[offset] = value;
            };
            if (!(Inverse ? operation.inverse(read, write) : operation.forward(read, write))) {
                error = "LaMa FFT: nonfinite input or result";
                return false;
            }
        }
        return true;
    } catch (const std::bad_alloc &) {
        error = "LaMa FFT: scratch allocation failed";
    } catch (const std::exception & exception) {
        error = std::string("LaMa FFT: ") + exception.what();
    } catch (...) {
        error = "LaMa FFT: unexpected transform failure";
    }
    return false;
}

} // namespace

const char * execution_status::message() const noexcept {
    switch (code.load(std::memory_order_relaxed)) {
    case 0:
        return "";
    case error_invalid:
        return "LaMa FFT: unsupported tensor type or dimensions";
    case error_allocation:
        return "LaMa FFT: scratch allocation failed";
    case error_nonfinite:
        return "LaMa FFT: nonfinite input or result";
    default:
        return "LaMa FFT: unexpected transform failure";
    }
}

ggml_tensor * rfft2(ggml_context * ctx, ggml_tensor * input, execution_status * status) {
    if (!ctx || !input || input->type != GGML_TYPE_F32 ||
        !shape_ok(input->ne[0], input->ne[1], input->ne[2], input->ne[3])) {
        fail(status, error_invalid);
        return nullptr;
    }
    ggml_tensor * arguments[] = { input };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, input->ne[0] / 2 + 1, input->ne[1], 2 * input->ne[2], input->ne[3],
                          arguments, 1, compute_node<false>, GGML_N_TASKS_MAX, status);
}

ggml_tensor * irfft2(ggml_context * ctx, ggml_tensor * input, int64_t original_width, execution_status * status) {
    if (!ctx || !input || input->type != GGML_TYPE_F32 || input->ne[2] % 2 || original_width < 1 ||
        input->ne[0] != original_width / 2 + 1 ||
        !shape_ok(original_width, input->ne[1], input->ne[2] / 2, input->ne[3])) {
        fail(status, error_invalid);
        return nullptr;
    }
    ggml_tensor * arguments[] = { input };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, original_width, input->ne[1], input->ne[2] / 2, input->ne[3], arguments,
                          1, compute_node<true>, GGML_N_TASKS_MAX, status);
}

bool rfft2_f32(const float * input, size_t input_count, float * output, size_t output_count, int64_t width,
               int64_t height, int64_t channels, int64_t batch, std::string & error) {
    return contiguous<false>(input, input_count, output, output_count, width, height, channels, batch, error);
}

bool irfft2_f32(const float * input, size_t input_count, float * output, size_t output_count, int64_t width,
                int64_t height, int64_t channels, int64_t batch, std::string & error) {
    return contiguous<true>(input, input_count, output, output_count, width, height, channels, batch, error);
}

} // namespace lama_fft
