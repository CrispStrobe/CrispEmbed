// Hosted, model-free graph controls. No learned checkpoint is needed.
// Compare the selected CPU/CUDA/Metal graph against the existing independent
// CPU FFT/GLA implementations and a tiny direct-DFT oracle. An unavailable or
// unsupported explicitly requested GPU is a failure, never a CPU fallback.
#include "inpaint_graph_spectral.h"
#include "lama_fft.h"
#include "latent_inpaint_unet.h"
#include "core/clean_exit.h"
#include "core/gpu_backend_pref.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using latent_inpaint::Tensor;
using Complex = std::complex<double>;
constexpr double pi = 3.141592653589793238462643383279502884;
// Declared before any backend run. These are component controls, not model
// qualification or substitutes for the unchanged original-model image gates.
constexpr double fft_max_abs = 1e-4;
constexpr double gla_max_abs = 1e-4;
constexpr double gla_max_relative_l2 = 0.002;
constexpr double gla_min_cosine = 0.99999;
int checks = 0, failures = 0;

void check(bool condition, const std::string & label) {
    ++checks;
    if (!condition) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", label.c_str());
    }
}

template <class Function> void rejects(Function function, const char * label) {
    bool rejected = false;
    try {
        function();
    } catch (const std::exception &) {
        rejected = true;
    }
    check(rejected, label);
}

std::vector<float> values(size_t count, int seed) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) result[i] = float(int((i * 37 + size_t(seed) * 19) % 101) - 50) / 128.0f;
    return result;
}

struct Metrics {
    double maximum = 0, relative_l2 = 0, cosine = 0;
    bool finite = true;
};

Metrics compare(const std::vector<float> & actual, const std::vector<float> & expected) {
    if (actual.size() != expected.size()) throw std::runtime_error("comparison element count differs");
    Metrics result;
    double error2 = 0, a2 = 0, b2 = 0, dot = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i], b = expected[i], error = a - b;
        if (!std::isfinite(a) || !std::isfinite(b)) {
            result.finite = false;
            return result;
        }
        result.maximum = std::max(result.maximum, std::abs(error));
        error2 += error * error;
        a2 += a * a;
        b2 += b * b;
        dot += a * b;
    }
    result.relative_l2 = std::sqrt(error2) / std::max(std::sqrt(b2), 1e-12);
    result.cosine = a2 == 0 || b2 == 0 ? (a2 == 0 && b2 == 0 ? 1 : 0) : dot / std::sqrt(a2 * b2);
    return result;
}

void graph_contract(ggml_tensor * output, const char * label) {
    std::set<ggml_tensor *> visited;
    std::vector<ggml_tensor *> pending{ output };
    bool portable = true;
    while (!pending.empty()) {
        ggml_tensor * node = pending.back();
        pending.pop_back();
        if (!node || !visited.insert(node).second) continue;
        portable = portable && node->op != GGML_OP_CUSTOM && node->op != GGML_OP_MAP_CUSTOM1 &&
                   node->op != GGML_OP_MAP_CUSTOM2 && node->op != GGML_OP_MAP_CUSTOM3 &&
                   node->op != GGML_OP_GATED_LINEAR_ATTN;
        for (ggml_tensor * source : node->src)
            if (source) pending.push_back(source);
    }
    check(portable, std::string(label) + " contains only portable graph operators, no callbacks/square-head GLA");
}

Tensor batch_tensor(const std::vector<float> & source, int width, int height, int channels, int batch) {
    Tensor result(channels, height, width);
    const size_t offset = size_t(batch) * result.v.size();
    std::copy_n(source.begin() + offset, result.v.size(), result.v.begin());
    return result;
}

std::vector<float> fft_reference(const std::vector<float> & source, int width, int height, int channels, int batches) {
    std::vector<float> output;
    for (int batch = 0; batch < batches; ++batch) {
        auto result = lama::detail::rfft2_ortho(batch_tensor(source, width, height, channels, batch));
        output.insert(output.end(), result.v.begin(), result.v.end());
    }
    return output;
}

Complex direct_frequency(const Tensor & input, int channel, int ky, int kx) {
    Complex result = 0;
    for (int y = 0; y < input.h; ++y)
        for (int x = 0; x < input.w; ++x) {
            const double angle = -2 * pi * (double(ky) * y / input.h + double(kx) * x / input.w);
            result += double(input.at(channel, y, x)) * std::polar(1.0, angle);
        }
    return result / std::sqrt(double(input.spatial()));
}

void fft_case(inpaint_graph::Device & device, int width, int height, int channels, int batches) {
    const std::string label = "fft-" + std::to_string(width) + "x" + std::to_string(height) + "-c" +
                              std::to_string(channels) + "-b" + std::to_string(batches);
    inpaint_graph::Builder builder(device);
    auto * input = builder.parameter("input", { width, height, channels, batches });
    auto * spectrum = inpaint_graph::rfft2_ortho(builder, input);
    auto * restored = inpaint_graph::irfft2_ortho(builder, spectrum, width);
    graph_contract(spectrum, label.c_str());
    graph_contract(restored, label.c_str());
    check(spectrum->ne[0] == width / 2 + 1 && spectrum->ne[1] == height && spectrum->ne[2] == 2 * channels &&
              spectrum->ne[3] == batches,
          label + " preserves interleaved real/imag channel and batch geometry");
    builder.finish({ spectrum, restored }, label);
    const auto source = values(size_t(width) * height * channels * batches, width + height);
    builder.upload(input, source);
    const uint64_t uploads = builder.upload_bytes();
    builder.compute();
    check(builder.upload_bytes() == uploads && builder.download_bytes() == 0,
          label + " performs no activation upload/download during graph computation");
    auto actual = builder.download(spectrum);
    auto expected = fft_reference(source, width, height, channels, batches);
    const Metrics forward = compare(actual, expected);
    check(forward.finite && forward.maximum <= fft_max_abs, label + " matches independent double-precision CPU FFT");
    const Metrics inverse = compare(builder.download(restored), source);
    check(inverse.finite && inverse.maximum <= fft_max_abs, label + " orthonormal round trip");
    if (width * height <= 160) {
        double maximum = 0;
        const int half_width = width / 2 + 1;
        const size_t plane = size_t(half_width) * height;
        for (int batch = 0; batch < batches; ++batch) {
            const Tensor image = batch_tensor(source, width, height, channels, batch);
            for (int channel = 0; channel < channels; ++channel)
                for (int y = 0; y < height; ++y)
                    for (int x = 0; x < half_width; ++x) {
                        const Complex oracle = direct_frequency(image, channel, y, x);
                        const size_t index = size_t(batch) * plane * channels * 2 + size_t(channel * 2) * plane +
                                             size_t(y) * half_width + x;
                        maximum = std::max(maximum, std::abs(double(actual[index]) - oracle.real()));
                        maximum = std::max(maximum, std::abs(double(actual[index + plane]) - oracle.imag()));
                    }
        }
        check(maximum <= fft_max_abs, label + " also agrees with the independent direct DFT");
    }
    std::printf("spectral_case=%s backend=%s max_abs=%.9g roundtrip_max_abs=%.9g nodes=%zu bytes=%zu\n", label.c_str(),
                device.name.c_str(), forward.maximum, inverse.maximum, builder.nodes(), builder.allocation_bytes());
}

std::vector<float> direct_inverse(const std::vector<float> & source, int width, int height, int channels, int batches) {
    const int half_width = width / 2 + 1;
    std::vector<float> result(size_t(width) * height * channels * batches);
    for (int batch = 0; batch < batches; ++batch) {
        const Tensor packed = batch_tensor(source, half_width, height, 2 * channels, batch);
        for (int ch = 0; ch < channels; ++ch)
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) {
                    Complex sum = 0;
                    for (int ky = 0; ky < height; ++ky)
                        for (int kx = 0; kx < width; ++kx) {
                            const Complex frequency =
                                kx < half_width
                                    ? Complex(packed.at(2 * ch, ky, kx), packed.at(2 * ch + 1, ky, kx))
                                    : std::conj(Complex(packed.at(2 * ch, (height - ky) % height, width - kx),
                                                        packed.at(2 * ch + 1, (height - ky) % height, width - kx)));
                            const double angle = 2 * pi * (double(ky) * y / height + double(kx) * x / width);
                            sum += frequency * std::polar(1.0, angle);
                        }
                    const size_t index = ((size_t(batch) * channels + ch) * height + y) * width + x;
                    result[index] = float(sum.real() / std::sqrt(double(width * height)));
                }
    }
    return result;
}

void learned_spectrum_case(inpaint_graph::Device & device, int width, int height) {
    constexpr int channels = 2, batches = 2;
    inpaint_graph::Builder builder(device);
    auto * input = builder.parameter("learned-spectrum", { width / 2 + 1, height, channels * 2, batches });
    auto * output = inpaint_graph::irfft2_ortho(builder, input, width);
    builder.finish({ output }, "arbitrary-learned-spectrum");
    const auto source = values(size_t(ggml_nelements(input)), 17);
    builder.upload(input, source);
    builder.compute();
    const Metrics metric = compare(builder.download(output), direct_inverse(source, width, height, channels, batches));
    check(metric.finite && metric.maximum <= fft_max_abs,
          "irfft real projection matches arbitrary non-Hermitian DC/Nyquist columns, both batches");
    const size_t nodes = builder.nodes(), allocation = builder.allocation_bytes();
    std::vector<float> imaginary_dc(source.size(), 0);
    const size_t plane = size_t(width / 2 + 1) * height;
    // Imaginary DC at ky=0 contributes no real signal. A nonzero answer would
    // reveal an incorrect Re/Im channel layout or real-projection contract.
    for (int batch = 0; batch < batches; ++batch)
        for (int ch = 0; ch < channels; ++ch)
            imaginary_dc[size_t(batch) * plane * channels * 2 + size_t(ch * 2 + 1) * plane] = 7.0f;
    builder.upload(input, imaginary_dc);
    const uint64_t uploads = builder.upload_bytes(), downloads = builder.download_bytes();
    builder.compute();
    check(builder.runs() == 2 && builder.nodes() == nodes && builder.allocation_bytes() == allocation &&
              builder.upload_bytes() == uploads && builder.download_bytes() == downloads,
          "FFT graph reuses its compiled allocation with no internal host transfers");
    const auto zeros = builder.download(output);
    check(
        std::all_of(zeros.begin(), zeros.end(), [](float x) { return std::isfinite(x) && std::abs(x) <= fft_max_abs; }),
        "imaginary DC projects to zero on a reused inverse graph");
}

std::vector<float> gla_reference(const std::vector<float> & q, const std::vector<float> & k,
                                 const std::vector<float> & v, const std::vector<float> & gates, int width, int height,
                                 int dk, int dv, int heads, int batches, float normalizer) {
    std::vector<float> expected;
    for (int batch = 0; batch < batches; ++batch) {
        auto result = latent_inpaint::detail::gla_recurrent(
            batch_tensor(q, width, height, dk * heads, batch), batch_tensor(k, width, height, dk * heads, batch),
            batch_tensor(v, width, height, dv * heads, batch), batch_tensor(gates, width, height, dk * heads, batch),
            heads, normalizer);
        expected.insert(expected.end(), result.v.begin(), result.v.end());
    }
    return expected;
}

void gla_case(inpaint_graph::Device & device, int width, int height, int dk, int dv, int heads, int batches,
              bool extreme, float normalizer = 16.0f) {
    const std::string label = "gla-" + std::to_string(width * height) + "-k" + std::to_string(dk) + "-v" +
                              std::to_string(dv) + "-h" + std::to_string(heads) + "-b" + std::to_string(batches) +
                              (extreme ? "-extreme" : "");
    inpaint_graph::Builder builder(device);
    auto * q = builder.parameter("q", { width, height, dk * heads, batches });
    auto * k = builder.parameter("k", { width, height, dk * heads, batches });
    auto * v = builder.parameter("v", { width, height, dv * heads, batches });
    auto * gates = builder.parameter("gates", { width, height, dk * heads, batches });
    auto * output = inpaint_graph::gla(builder, q, k, v, gates, heads, normalizer);
    graph_contract(output, label.c_str());
    check(output->ne[0] == width && output->ne[1] == height && output->ne[2] == dv * heads && output->ne[3] == batches,
          label + " output retains rectangular heads and independent batch geometry");
    builder.finish({ output }, label);
    auto qv = values(size_t(ggml_nelements(q)), 3), kv = values(size_t(ggml_nelements(k)), 5);
    auto vv = values(size_t(ggml_nelements(v)), 7), gv = values(size_t(ggml_nelements(gates)), 11);
    for (size_t i = 0; i < gv.size(); ++i) {
        gv[i] = gv[i] * 5 - 1;
        if (extreme) {
            if (i % 7 == 0) gv[i] = -1000;
            if (i % 11 == 0) gv[i] = 1000;
            if (i % 37 == 0) gv[i] = -std::numeric_limits<float>::max();
        }
    }
    builder.upload(q, qv);
    builder.upload(k, kv);
    builder.upload(v, vv);
    builder.upload(gates, gv);
    const uint64_t uploads = builder.upload_bytes();
    builder.compute();
    check(builder.upload_bytes() == uploads && builder.download_bytes() == 0,
          label + " carries recurrent state entirely on the selected backend");
    auto actual = builder.download(output);
    const auto expected = gla_reference(qv, kv, vv, gv, width, height, dk, dv, heads, batches, normalizer);
    const Metrics metric = compare(actual, expected);
    check(metric.finite && metric.maximum <= gla_max_abs && metric.relative_l2 <= gla_max_relative_l2 &&
              metric.cosine >= gla_min_cosine,
          label + " matches independent token-by-token F32 recurrence");

    const int tokens = width * height;
    // Cross the 64-token chunk boundary as well as checking ordinary causal
    // masking within a chunk. The 65-token case also exercises its one-token tail.
    const int boundary = tokens > 64 ? std::min(tokens - 1, 65) : std::max(1, tokens / 2);
    for (int batch = 0; batch < batches; ++batch)
        for (int ch = 0; ch < dv * heads; ++ch)
            for (int token = boundary; token < tokens; ++token)
                vv[(size_t(batch) * dv * heads + ch) * tokens + token] += 0.25f;
    const size_t nodes = builder.nodes(), allocation = builder.allocation_bytes();
    builder.upload(v, vv);
    const uint64_t repeated_uploads = builder.upload_bytes(), downloads = builder.download_bytes();
    builder.compute();
    check(builder.runs() == 2 && builder.nodes() == nodes && builder.allocation_bytes() == allocation &&
              builder.upload_bytes() == repeated_uploads && builder.download_bytes() == downloads,
          label + " reuses the graph and resets recurrence without hidden host transfers");
    const auto changed = builder.download(output);
    bool causal = true;
    for (int batch = 0; batch < batches; ++batch)
        for (int ch = 0; ch < dv * heads; ++ch)
            for (int token = 0; token < boundary; ++token) {
                const size_t index = (size_t(batch) * dv * heads + ch) * tokens + token;
                causal = causal && changed[index] == actual[index];
            }
    check(causal, label + " future values cannot change a prior raster token");
    const Metrics repeated =
        compare(changed, gla_reference(qv, kv, vv, gv, width, height, dk, dv, heads, batches, normalizer));
    check(repeated.finite && repeated.maximum <= gla_max_abs && repeated.relative_l2 <= gla_max_relative_l2 &&
              repeated.cosine >= gla_min_cosine,
          label + " reused graph has no stale carried state");
    std::printf("spectral_case=%s backend=%s max_abs=%.9g relative_l2=%.9g cosine=%.12g nodes=%zu bytes=%zu\n",
                label.c_str(), device.name.c_str(), metric.maximum, metric.relative_l2, metric.cosine, builder.nodes(),
                builder.allocation_bytes());
}

void rejected_geometry(inpaint_graph::Device & device) {
    inpaint_graph::Builder builder(device);
    auto * too_wide = builder.parameter("too-wide", { 129, 1, 1, 1 });
    rejects([&] { (void)inpaint_graph::rfft2_ortho(builder, too_wide); }, "FFT axis beyond128 is rejected");
    auto * invalid_packed = builder.parameter("invalid-packed", { 3, 3, 3, 1 });
    rejects([&] { (void)inpaint_graph::irfft2_ortho(builder, invalid_packed, 5); },
            "odd complex channel count is rejected");
    auto * packed = builder.parameter("packed", { 3, 3, 2, 1 });
    rejects([&] { (void)inpaint_graph::irfft2_ortho(builder, packed, 6); }, "inconsistent original width is rejected");
    auto * q = builder.parameter("q", { 5, 2, 6, 2 });
    auto * v = builder.parameter("v", { 5, 2, 10, 2 });
    rejects([&] { (void)inpaint_graph::gla(builder, q, q, v, q, 0); }, "zero GLA heads are rejected");
    rejects([&] { (void)inpaint_graph::gla(builder, q, q, v, q, 4); }, "nondivisible GLA heads are rejected");
    rejects([&] { (void)inpaint_graph::gla(builder, q, q, v, q, 2, 0); }, "zero GLA normalizer is rejected");
    rejects([&] { (void)inpaint_graph::gla(builder, q, q, v, q, 2, std::numeric_limits<float>::quiet_NaN()); },
            "nonfinite GLA normalizer is rejected");
    rejects([&] { (void)inpaint_graph::gla(builder, q, q, v, q, 2, std::numeric_limits<float>::denorm_min()); },
            "GLA normalizer with an overflowing reciprocal is rejected");
    auto * wrong_batch = builder.parameter("wrong-batch", { 5, 2, 10, 1 });
    rejects([&] { (void)inpaint_graph::gla(builder, q, q, wrong_batch, q, 2); }, "mixed GLA batch counts are rejected");
}

int run(int argc, char ** argv) {
    std::string backend = "cpu";
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if ((argument == "--backend" || argument == "--gpu-backend") && i + 1 < argc)
            backend = argv[++i];
        else
            throw std::runtime_error("usage: test-inpaint-graph-spectral [--backend cpu|cuda|metal]");
    }
    if (backend != "cpu" && backend != "cuda" && backend != "metal")
        throw std::runtime_error("unsupported explicit test backend");
    crispasr_set_gpu_backend_pref(backend.c_str());
    inpaint_graph::Device device(2);
    check(device.gpu == (backend != "cpu"), "explicit backend request is honored without CPU fallback");
    rejected_geometry(device);
    fft_case(device, 1, 1, 2, 2);
    fft_case(device, 8, 8, 3, 2);
    fft_case(device, 5, 3, 2, 2);
    fft_case(device, 10, 7, 2, 1);
    fft_case(device, 9, 16, 1, 2);
    fft_case(device, 127, 128, 1, 1);
    learned_spectrum_case(device, 5, 3);
    learned_spectrum_case(device, 6, 5);
    gla_case(device, 1, 1, 3, 5, 2, 2, false);
    gla_case(device, 13, 7, 5, 3, 2, 2, false);
    gla_case(device, 33, 5, 7, 11, 3, 1, false, 4.0f);
    gla_case(device, 8, 9, 20, 40, 8, 2, false);
    gla_case(device, 65, 1, 4, 7, 2, 2, true);
    std::printf("{\"suite\":\"inpaint-graph-spectral\",\"backend\":\"%s\",\"checks\":%d,\"failures\":%d,"
                "\"native_model_qualified\":false}\n",
                device.name.c_str(), checks, failures);
    return failures ? 1 : 0;
}
} // namespace

int main(int argc, char ** argv) {
    int status = 1;
    try {
        status = run(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
    }
    core_util::clean_exit(status);
}
