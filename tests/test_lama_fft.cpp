// Model-free numerical contracts for LaMa's orthonormal Fourier operators.
// Expected values use a direct DFT, independent of the production FFT algorithm.
#include "lama_fft.h"
#include "core/clean_exit.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr double pi = 3.141592653589793238462643383279502884;
int checks = 0, failures = 0;
double largest_forward_error = 0, largest_inverse_error = 0, largest_roundtrip_error = 0;

void check(bool condition, const char * description) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", description);
    }
}

double difference(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double result = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) return std::numeric_limits<double>::infinity();
        result = std::max(result, std::abs(double(a[i]) - b[i]));
    }
    return result;
}

std::vector<float> fixture(size_t width, size_t height, size_t channels, size_t batch) {
    std::vector<float> values(width * height * channels * batch);
    for (size_t plane = 0; plane < channels * batch; ++plane)
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x)
                values[plane * width * height + y * width + x] =
                    float(std::sin(double(13 * x + 7 * y + 3 * plane)) * 0.8 +
                          std::cos(double(5 * x + 11 * y + 17 * plane)) * 0.3 + double(plane) * 0.07);
    return values;
}

std::vector<float> direct_forward(const std::vector<float> & input, size_t width, size_t height, size_t planes) {
    const size_t half_width = width / 2 + 1, packed_plane = half_width * height;
    std::vector<float> output(2 * packed_plane * planes);
    const double scale = 1.0 / std::sqrt(double(width * height));
    for (size_t plane = 0; plane < planes; ++plane)
        for (size_t ky = 0; ky < height; ++ky)
            for (size_t kx = 0; kx < half_width; ++kx) {
                std::complex<double> value(0.0, 0.0);
                for (size_t y = 0; y < height; ++y)
                    for (size_t x = 0; x < width; ++x) {
                        const double angle = -2.0 * pi * (double(kx * x) / width + double(ky * y) / height);
                        value += double(input[plane * width * height + y * width + x]) *
                                 std::complex<double>(std::cos(angle), std::sin(angle));
                    }
                output[(2 * plane) * packed_plane + ky * half_width + kx] = float(value.real() * scale);
                output[(2 * plane + 1) * packed_plane + ky * half_width + kx] = float(value.imag() * scale);
            }
    return output;
}

// Each interior horizontal frequency represents a conjugate pair and therefore
// contributes twice its real component. The DC/even-Nyquist columns contribute
// once. This directly evaluates irfftn, including non-Hermitian endpoint values
// introduced by FourierUnit's learned spectral convolution.
std::vector<float> direct_inverse(const std::vector<float> & input, size_t width, size_t height, size_t planes) {
    const size_t half_width = width / 2 + 1, packed_plane = half_width * height;
    const double scale = 1.0 / std::sqrt(double(width * height));
    std::vector<float> output(width * height * planes);
    for (size_t plane = 0; plane < planes; ++plane)
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x) {
                double value = 0;
                for (size_t ky = 0; ky < height; ++ky)
                    for (size_t kx = 0; kx < half_width; ++kx) {
                        const double real = input[(2 * plane) * packed_plane + ky * half_width + kx];
                        const double imag = input[(2 * plane + 1) * packed_plane + ky * half_width + kx];
                        const double angle = 2.0 * pi * (double(kx * x) / width + double(ky * y) / height);
                        const double multiplicity = (kx == 0 || (width % 2 == 0 && kx == width / 2)) ? 1.0 : 2.0;
                        value += multiplicity * (real * std::cos(angle) - imag * std::sin(angle));
                    }
                output[plane * width * height + y * width + x] = float(value * scale);
            }
    return output;
}

void numerical_contracts() {
    std::string error;
    for (const auto & dimensions :
         { std::make_pair(1, 1), std::make_pair(1, 7), std::make_pair(8, 1), std::make_pair(2, 2), std::make_pair(3, 5),
           std::make_pair(4, 6), std::make_pair(7, 8), std::make_pair(11, 13), std::make_pair(16, 16) }) {
        const size_t width = dimensions.first, height = dimensions.second, channels = 3, batch = 2;
        const auto input = fixture(width, height, channels, batch);
        std::vector<float> packed((width / 2 + 1) * height * 2 * channels * batch);
        std::vector<float> restored(input.size());
        error = "stale";
        check(lama_fft::rfft2_f32(input.data(), input.size(), packed.data(), packed.size(), width, height, channels,
                                  batch, error) &&
                  error.empty(),
              "forward transform accepts odd/even/non-power-of-two dimensions and clears stale error");
        const double forward_error = difference(packed, direct_forward(input, width, height, channels * batch));
        largest_forward_error = std::max(largest_forward_error, forward_error);
        check(forward_error < 2e-5, "all packed real/imag channel planes match independent orthonormal DFT");
        check(lama_fft::irfft2_f32(packed.data(), packed.size(), restored.data(), restored.size(), width, height,
                                   channels, batch, error),
              "inverse transform accepts the exact original width");
        const double roundtrip_error = difference(restored, input);
        largest_roundtrip_error = std::max(largest_roundtrip_error, roundtrip_error);
        check(roundtrip_error < 2e-6, "round trip preserves every channel and batch plane");

        // Learned complex coefficients need not obey the endpoint symmetry of
        // an actual rFFT. Test that situation rather than only round trips.
        for (size_t i = 0; i < packed.size(); ++i) packed[i] = float(std::sin(double(17 * i + 2)));
        check(lama_fft::irfft2_f32(packed.data(), packed.size(), restored.data(), restored.size(), width, height,
                                   channels, batch, error),
              "inverse accepts arbitrary finite spectral coefficients");
        const double inverse_error = difference(restored, direct_inverse(packed, width, height, channels * batch));
        largest_inverse_error = std::max(largest_inverse_error, inverse_error);
        check(inverse_error < 2e-5, "inverse matches direct real projection with asymmetric DC/Nyquist columns");
    }

    for (const auto & dimensions : { std::make_pair(37, 24), std::make_pair(129, 65), std::make_pair(512, 64) }) {
        const size_t width = dimensions.first, height = dimensions.second;
        const auto input = fixture(width, height, 2, 1);
        std::vector<float> packed((width / 2 + 1) * height * 4), restored(input.size());
        check(
            lama_fft::rfft2_f32(input.data(), input.size(), packed.data(), packed.size(), width, height, 2, 1, error) &&
                lama_fft::irfft2_f32(packed.data(), packed.size(), restored.data(), restored.size(), width, height, 2,
                                     1, error),
            "larger radix-2 and Bluestein transforms complete");
        const double roundtrip_error = difference(restored, input);
        largest_roundtrip_error = std::max(largest_roundtrip_error, roundtrip_error);
        check(roundtrip_error < 2e-6, "larger transform round trip meets unchanged absolute error budget");
    }
}

void normalization_contracts() {
    constexpr size_t width = 8, height = 6, half_width = width / 2 + 1;
    std::vector<float> input(width * height, 1.0f), packed(2 * half_width * height), restored(input.size());
    std::string error;
    check(lama_fft::rfft2_f32(input.data(), input.size(), packed.data(), packed.size(), width, height, 1, 1, error),
          "constant input transforms");
    check(std::abs(packed[0] - std::sqrt(double(width * height))) < 1e-6,
          "constant DC amplitude uses orthonormal scaling");
    bool zero = true;
    for (size_t i = 1; i < packed.size(); ++i) zero &= std::abs(packed[i]) < 1e-6;
    check(zero, "constant input has no other real or imaginary frequency");

    for (size_t y = 0; y < height; ++y)
        for (size_t x = 0; x < width; ++x)
            input[y * width + x] = float(std::cos(2 * pi * (double(x) / width + 2.0 * y / height)));
    check(lama_fft::rfft2_f32(input.data(), input.size(), packed.data(), packed.size(), width, height, 1, 1, error),
          "two-axis sinusoid transforms");
    check(std::abs(packed[2 * half_width + 1] - std::sqrt(double(width * height)) / 2) < 1e-6,
          "two-axis sinusoid lands at its expected positive frequency");
    double input_energy = 0, spectrum_energy = 0;
    for (float value : input) input_energy += double(value) * value;
    for (size_t y = 0; y < height; ++y)
        for (size_t x = 0; x < half_width; ++x) {
            const double real = packed[y * half_width + x];
            const double imag = packed[half_width * height + y * half_width + x];
            spectrum_energy += (x == 0 || x == width / 2 ? 1.0 : 2.0) * (real * real + imag * imag);
        }
    check(std::abs(input_energy - spectrum_energy) < 1e-5, "half-spectrum Parseval energy matches spatial energy");

    std::fill(packed.begin(), packed.end(), 0.0f);
    packed[half_width * height] = 51.0f;
    packed[half_width * height + width / 2] = -32.0f;
    check(lama_fft::irfft2_f32(packed.data(), packed.size(), restored.data(), restored.size(), width, height, 1, 1,
                               error),
          "inverse accepts imaginary self-conjugate bins");
    check(std::all_of(restored.begin(), restored.end(), [](float value) { return std::abs(value) < 1e-6; }),
          "imaginary DC and Nyquist self-conjugate bins do not affect real output");
}

void validation_contracts() {
    std::vector<float> input(36, 1), output(48, 2);
    const auto previous = output;
    std::string error;
    check(!lama_fft::rfft2_f32(nullptr, 36, output.data(), 48, 6, 6, 1, 1, error), "null input rejects");
    check(!lama_fft::rfft2_f32(input.data(), 36, nullptr, 48, 6, 6, 1, 1, error), "null output rejects");
    check(!lama_fft::rfft2_f32(input.data(), 35, output.data(), 48, 6, 6, 1, 1, error), "short real buffer rejects");
    check(!lama_fft::rfft2_f32(input.data(), 36, output.data(), 47, 6, 6, 1, 1, error),
          "short spectrum buffer rejects");
    for (int64_t bad : { int64_t(-1), int64_t(0), int64_t(8193), std::numeric_limits<int64_t>::max() })
        check(!lama_fft::rfft2_f32(input.data(), 36, output.data(), 48, bad, 6, 1, 1, error),
              "invalid or overflowing width rejects before reading storage");
    check(!lama_fft::rfft2_f32(input.data(), 36, output.data(), 48, 8192, 8192, 1, 1, error),
          "excessive plane allocation rejects");
    check(
        !lama_fft::rfft2_f32(input.data(), 36, output.data(), 48, 6, 6, std::numeric_limits<int64_t>::max(), 1, error),
        "overflowing channel count rejects");
    check(output == previous, "shape/storage preflight leaves output unchanged");
    check(!lama_fft::rfft2_f32(output.data(), output.size(), output.data(), output.size(), 6, 6, 1, 1, error),
          "overlapping contiguous storage rejects");
    input[3] = std::numeric_limits<float>::infinity();
    check(!lama_fft::rfft2_f32(input.data(), 36, output.data(), 48, 6, 6, 1, 1, error) &&
              error.find("nonfinite") != std::string::npos,
          "nonfinite real input rejects with a useful error");
    std::fill(input.begin(), input.end(), std::numeric_limits<float>::max());
    check(!lama_fft::rfft2_f32(input.data(), 36, output.data(), 48, 6, 6, 1, 1, error),
          "finite input whose DC coefficient exceeds F32 range rejects before narrowing");
    output[7] = std::numeric_limits<float>::quiet_NaN();
    check(!lama_fft::irfft2_f32(output.data(), 48, input.data(), 36, 6, 6, 1, 1, error),
          "nonfinite spectral input rejects");
}

void graph_contracts() {
    const ggml_init_params parameters = { 16 * 1024 * 1024, nullptr, false };
    ggml_context * context = ggml_init(parameters);
    check(context != nullptr, "ggml test context allocates");
    if (!context) return;
    lama_fft::execution_status status;
    constexpr int width = 7, height = 6, channels = 3, batch = 2;
    const auto original = fixture(width, height, channels, batch);
    ggml_tensor * input = ggml_new_tensor_4d(context, GGML_TYPE_F32, width, height, channels, batch);
    std::memcpy(input->data, original.data(), original.size() * sizeof(float));
    ggml_tensor * packed = lama_fft::rfft2(context, input, &status);
    ggml_tensor * restored = lama_fft::irfft2(context, packed, width, &status);
    check(packed && restored && status.ok(), "graph helpers construct custom-shape CPU nodes");
    if (packed && restored) {
        ggml_cgraph * graph = ggml_new_graph(context);
        ggml_build_forward_expand(graph, restored);
        for (int threads : { 1, 4 }) {
            status.reset();
            check(ggml_graph_compute_with_ctx(context, graph, threads) == GGML_STATUS_SUCCESS && status.ok(),
                  "CPU callbacks compute with one or multiple workers");
            std::vector<float> actual(original.size());
            std::memcpy(actual.data(), restored->data, actual.size() * sizeof(float));
            check(difference(actual, original) < 2e-6, "CPU callback round trip preserves batch/channel layout");
            std::vector<float> spectrum(size_t(width / 2 + 1) * height * channels * batch * 2);
            std::memcpy(spectrum.data(), packed->data, spectrum.size() * sizeof(float));
            check(difference(spectrum, direct_forward(original, width, height, channels * batch)) < 2e-5,
                  "CPU callback spectrum matches independent direct DFT");
        }
        static_cast<float *>(input->data)[0] = std::numeric_limits<float>::quiet_NaN();
        status.reset();
        check(ggml_graph_compute_with_ctx(context, graph, 4) == GGML_STATUS_SUCCESS && !status.ok() &&
                  std::string(status.message()).find("nonfinite") != std::string::npos,
              "callback reports numerical failure without throwing through ggml");
        check(std::isnan(static_cast<float *>(restored->data)[0]), "failed callback plane is explicitly invalid");
        std::memcpy(input->data, original.data(), original.size() * sizeof(float));
        status.reset();
        check(ggml_graph_compute_with_ctx(context, graph, 4) == GGML_STATUS_SUCCESS && status.ok(),
              "status reset permits successful graph reuse after failure");
    }

    ggml_tensor * transposed = ggml_permute(context, input, 1, 0, 2, 3);
    status.reset();
    ggml_tensor * transposed_spectrum = lama_fft::rfft2(context, transposed, &status);
    if (transposed_spectrum) {
        ggml_cgraph * graph = ggml_new_graph(context);
        ggml_build_forward_expand(graph, transposed_spectrum);
        check(ggml_graph_compute_with_ctx(context, graph, 4) == GGML_STATUS_SUCCESS && status.ok(),
              "CPU callback accepts a strided transpose view");
        std::vector<float> transposed_input(original.size());
        for (size_t plane = 0; plane < channels * batch; ++plane)
            for (size_t y = 0; y < height; ++y)
                for (size_t x = 0; x < width; ++x)
                    transposed_input[plane * width * height + x * height + y] =
                        original[plane * width * height + y * width + x];
        std::vector<float> actual(size_t(height / 2 + 1) * width * channels * batch * 2);
        std::memcpy(actual.data(), transposed_spectrum->data, actual.size() * sizeof(float));
        check(difference(actual, direct_forward(transposed_input, height, width, channels * batch)) < 2e-5,
              "strided view spectrum matches the independently transposed source");
    } else {
        check(false, "strided FFT graph constructs");
    }

    status.reset();
    check(lama_fft::irfft2(context, packed, width + 4, &status) == nullptr && !status.ok(),
          "inconsistent inverse original-width metadata rejects");
    status.reset();
    ggml_tensor * wrong_type = ggml_new_tensor_2d(context, GGML_TYPE_F16, 4, 4);
    check(lama_fft::rfft2(context, wrong_type, &status) == nullptr && !status.ok(), "non-F32 graph input rejects");
    status.reset();
    check(lama_fft::rfft2(nullptr, input, &status) == nullptr && !status.ok(), "null graph context rejects");
    ggml_free(context);
}

} // namespace

static int crispembed_test_main() {
    numerical_contracts();
    normalization_contracts();
    validation_contracts();
    graph_contracts();
    std::printf("LAMA_FFT_CONTRACT {\"checks\":%d,\"failures\":%d,\"passed\":%s,"
                "\"max_forward_abs\":%.9g,\"max_inverse_abs\":%.9g,\"max_roundtrip_abs\":%.9g}\n",
                checks, failures, failures ? "false" : "true", largest_forward_error, largest_inverse_error,
                largest_roundtrip_error);
    return failures ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
