#include "lama_fft.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>
#include <vector>

namespace lama {
namespace detail {
namespace {
using Complex = std::complex<double>;
constexpr double pi = 3.141592653589793238462643383279502884;
constexpr int max_fft_axis = 256;

void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

bool power_of_two(size_t n) {
    return n && (n & (n - 1)) == 0;
}

// Unnormalized forward transform; inverse normalization is explicit below.
void radix2(std::vector<Complex> & data) {
    const size_t n = data.size();
    require(power_of_two(n), "LaMa FFT: radix-2 length is invalid");
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
    for (size_t length = 2; length <= n; length <<= 1) {
        const Complex unit = std::polar(1.0, -2.0 * pi / double(length));
        for (size_t start = 0; start < n; start += length) {
            Complex phase(1.0, 0.0);
            for (size_t j = 0; j < length / 2; ++j) {
                const Complex even = data[start + j], odd = data[start + j + length / 2] * phase;
                data[start + j] = even + odd;
                data[start + j + length / 2] = even - odd;
                phase *= unit;
            }
        }
    }
}

void inverse_radix2(std::vector<Complex> & data) {
    for (auto & value : data) value = std::conj(value);
    radix2(data);
    const double scale = 1.0 / double(data.size());
    for (auto & value : data) value = std::conj(value) * scale;
}

class Plan {
    size_t n_;
    size_t convolution_size_ = 0;
    std::vector<Complex> chirp_, transformed_kernel_;

    void forward(std::vector<Complex> & data) const {
        if (!convolution_size_) {
            radix2(data);
            return;
        }
        std::vector<Complex> work(convolution_size_);
        for (size_t i = 0; i < n_; ++i) work[i] = data[i] * chirp_[i];
        radix2(work);
        for (size_t i = 0; i < work.size(); ++i) work[i] *= transformed_kernel_[i];
        inverse_radix2(work);
        for (size_t i = 0; i < n_; ++i) data[i] = work[i] * chirp_[i];
    }

public:
    explicit Plan(size_t n) : n_(n) {
        require(n > 0 && n <= max_fft_axis, "LaMa FFT: spatial dimension exceeds the bounded profile");
        if (power_of_two(n)) return;
        convolution_size_ = 1;
        while (convolution_size_ < 2 * n - 1) convolution_size_ <<= 1;
        chirp_.resize(n);
        transformed_kernel_.resize(convolution_size_);
        for (size_t i = 0; i < n; ++i) {
            // Reduce the phase before evaluating trig; n is bounded, and the
            // i*i product is exact in this range.
            const double angle = pi * double((i * i) % (2 * n)) / double(n);
            chirp_[i] = std::polar(1.0, -angle);
            const Complex value = std::conj(chirp_[i]);
            transformed_kernel_[i] = value;
            if (i) transformed_kernel_[convolution_size_ - i] = value;
        }
        radix2(transformed_kernel_);
    }

    void transform(std::vector<Complex> & data, bool inverse) const {
        require(data.size() == n_, "LaMa FFT: plan/input length mismatch");
        if (!inverse) {
            forward(data);
            return;
        }
        for (auto & value : data) value = std::conj(value);
        forward(data);
        const double scale = 1.0 / double(n_);
        for (auto & value : data) value = std::conj(value) * scale;
    }
};

void transform_plane(std::vector<Complex> & plane, int height, int width, const Plan & row_plan,
                     const Plan & column_plan, bool inverse) {
    std::vector<Complex> row(width), column(height);
    for (int y = 0; y < height; ++y) {
        std::copy_n(plane.begin() + size_t(y) * width, width, row.begin());
        row_plan.transform(row, inverse);
        std::copy(row.begin(), row.end(), plane.begin() + size_t(y) * width);
    }
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) column[y] = plane[size_t(y) * width + x];
        column_plan.transform(column, inverse);
        for (int y = 0; y < height; ++y) plane[size_t(y) * width + x] = column[y];
    }
}

void valid(const Tensor & tensor) {
    require(latent_inpaint::finite(tensor), "LaMa FFT: invalid tensor shape, storage or nonfinite value");
    require(tensor.h <= max_fft_axis && tensor.w <= max_fft_axis,
            "LaMa FFT: spatial dimension exceeds the bounded profile");
}
} // namespace

Tensor rfft2_ortho(const Tensor & real) {
    valid(real);
    require(real.c <= 8192, "LaMa FFT: too many real channels");
    Tensor result(real.c * 2, real.h, real.w / 2 + 1);
    const Plan rows(real.w), columns(real.h);
    std::vector<Complex> plane(real.spatial());
    const double scale = 1.0 / std::sqrt(double(real.spatial()));
    for (int ch = 0; ch < real.c; ++ch) {
        for (size_t p = 0; p < real.spatial(); ++p) plane[p] = Complex(real.v[size_t(ch) * real.spatial() + p], 0);
        transform_plane(plane, real.h, real.w, rows, columns, false);
        for (int y = 0; y < real.h; ++y)
            for (int x = 0; x < result.w; ++x) {
                const Complex value = plane[size_t(y) * real.w + x] * scale;
                result.at(2 * ch, y, x) = float(value.real());
                result.at(2 * ch + 1, y, x) = float(value.imag());
            }
    }
    require(latent_inpaint::finite(result), "LaMa FFT: forward transform overflow");
    return result;
}

Tensor irfft2_ortho(const Tensor & packed, int real_width) {
    valid(packed);
    require(packed.c % 2 == 0 && real_width > 0 && real_width <= max_fft_axis && packed.w == real_width / 2 + 1,
            "LaMa FFT: invalid packed spectrum or requested real width");
    Tensor result(packed.c / 2, packed.h, real_width);
    const Plan rows(real_width), columns(packed.h);
    std::vector<Complex> plane(result.spatial());
    const double scale = std::sqrt(double(result.spatial()));
    for (int ch = 0; ch < result.c; ++ch) {
        for (int y = 0; y < result.h; ++y)
            for (int x = 0; x < packed.w; ++x)
                plane[size_t(y) * real_width + x] = Complex(packed.at(2 * ch, y, x), packed.at(2 * ch + 1, y, x));
        for (int y = 0; y < result.h; ++y)
            for (int x = packed.w; x < real_width; ++x) {
                const int mirror_y = (result.h - y) % result.h, mirror_x = real_width - x;
                plane[size_t(y) * real_width + x] = std::conj(
                    Complex(packed.at(2 * ch, mirror_y, mirror_x), packed.at(2 * ch + 1, mirror_y, mirror_x)));
            }
        transform_plane(plane, result.h, real_width, rows, columns, true);
        // Real projection also matches irfftn's treatment of imaginary DC and
        // Nyquist components after the learned spectral convolution/BN/ReLU.
        for (size_t p = 0; p < result.spatial(); ++p)
            result.v[size_t(ch) * result.spatial() + p] = float(plane[p].real() * scale);
    }
    require(latent_inpaint::finite(result), "LaMa FFT: inverse transform overflow");
    return result;
}

} // namespace detail
} // namespace lama
