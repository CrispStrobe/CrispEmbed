// Model-free graph controls and explicitly requested operator benchmarks.
// No learned checkpoint is needed. Benchmark timings are component evidence,
// not full-model latency or final-image qualification.
// Compare the selected CPU/CUDA/Metal graph against the existing independent
// CPU FFT/GLA implementations and a tiny direct-DFT oracle. An unavailable or
// unsupported explicitly requested GPU is a failure, never a CPU fallback.
#include "inpaint_graph_spectral.h"
#include "lama_fft.h"
#include "latent_inpaint_unet.h"
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "core/gpu_backend_pref.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
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

struct GraphOperators {
    size_t native_recurrences = 0;
    size_t cpu_ffts = 0;
};

GraphOperators graph_contract(ggml_tensor * output, const char * label, bool native_cpu_gla = false,
                              size_t expected_cpu_ffts = 0) {
    std::set<ggml_tensor *> visited;
    std::vector<ggml_tensor *> pending{ output };
    bool declared_callbacks_only = true;
    GraphOperators operators;
    while (!pending.empty()) {
        ggml_tensor * node = pending.back();
        pending.pop_back();
        if (!node || !visited.insert(node).second) continue;
        declared_callbacks_only = declared_callbacks_only && node->op != GGML_OP_MAP_CUSTOM1 &&
                                  node->op != GGML_OP_MAP_CUSTOM2 && node->op != GGML_OP_MAP_CUSTOM3;
        if (node->op == GGML_OP_CUSTOM) {
            ++operators.cpu_ffts;
            declared_callbacks_only = declared_callbacks_only && expected_cpu_ffts > 0 &&
                                      (std::strcmp(node->name, "spectral.cpu.rfft2") == 0 ||
                                       std::strcmp(node->name, "spectral.cpu.irfft2") == 0);
        }
        if (node->op == GGML_OP_GATED_LINEAR_ATTN) ++operators.native_recurrences;
        for (ggml_tensor * source : node->src)
            if (source) pending.push_back(source);
    }
    check(declared_callbacks_only && operators.cpu_ffts == expected_cpu_ffts,
          std::string(label) + " contains only the explicitly selected CPU FFT callbacks, or none in portable mode");
    check(operators.native_recurrences == (native_cpu_gla ? 1u : 0u),
          std::string(label) + " uses precisely the declared portable or native CPU recurrence");
    return operators;
}

bool cpu_gla_requested(const inpaint_graph::Device & device) {
    return !device.gpu && core_env::on("CRISPEMBED_INPAINT_CPU_GLA");
}

bool cpu_fft_requested(const inpaint_graph::Device & device) {
    return !device.gpu && core_env::on("CRISPEMBED_INPAINT_CPU_FFT");
}

// Snapshot real allocated leaf bytes outside execution/timing. Merely marking
// a tensor INPUT does not preserve it after its last use in ggml's allocator:
// a repeated FFT used to overwrite its bit-reversal indices with activations.
// Check values, including F32 twiddles and I32 indices, rather than checking an
// implementation-specific flag. A failed check also avoids a later GET_ROWS
// assertion so the test can report the underlying lifetime defect clearly.
struct ResidentInputs {
    std::vector<std::pair<ggml_tensor *, std::vector<uint8_t>>> tensors;

    explicit ResidentInputs(ggml_tensor * output) {
        std::set<ggml_tensor *> visited;
        std::vector<ggml_tensor *> pending{ output };
        while (!pending.empty()) {
            ggml_tensor * node = pending.back();
            pending.pop_back();
            if (!node || !visited.insert(node).second) continue;
            if (node->op == GGML_OP_NONE && (node->flags & GGML_TENSOR_FLAG_INPUT)) {
                std::vector<uint8_t> bytes(ggml_nbytes(node));
                ggml_backend_tensor_get(node, bytes.data(), 0, bytes.size());
                tensors.emplace_back(node, std::move(bytes));
            }
            for (ggml_tensor * source : node->src)
                if (source) pending.push_back(source);
        }
    }

    bool unchanged(const std::string & label) const {
        bool preserved = true;
        for (const auto & entry : tensors) {
            std::vector<uint8_t> actual(entry.second.size());
            ggml_backend_tensor_get(entry.first, actual.data(), 0, actual.size());
            if (actual != entry.second) {
                preserved = false;
                std::fprintf(stderr, "resident leaf changed: %s (%s)\n", entry.first->name, label.c_str());
            }
        }
        check(preserved, label + " preserves resident inputs, F32 constants and I32 indices byte for byte");
        return preserved;
    }
};

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
    graph_contract(spectrum, label.c_str(), false, cpu_fft_requested(device) ? 1 : 0);
    graph_contract(restored, label.c_str(), false, cpu_fft_requested(device) ? 2 : 0);
    check(spectrum->ne[0] == width / 2 + 1 && spectrum->ne[1] == height && spectrum->ne[2] == 2 * channels &&
              spectrum->ne[3] == batches,
          label + " preserves interleaved real/imag channel and batch geometry");
    builder.finish({ spectrum, restored }, label);
    const auto source = values(size_t(width) * height * channels * batches, width + height);
    builder.upload(input, source);
    const ResidentInputs resident(restored);
    const uint64_t uploads = builder.upload_bytes();
    builder.compute();
    if (!resident.unchanged(label)) return;
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
    builder.compute();
    if (!resident.unchanged(label + " repeat")) return;
    check(builder.download(spectrum) == actual, label + " repeats exactly without reuploading its input or constants");
    const Metrics repeated = compare(builder.download(restored), source);
    check(repeated.finite && repeated.maximum <= fft_max_abs, label + " repeated orthonormal round trip");
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
    graph_contract(output, "arbitrary-learned-spectrum", false, cpu_fft_requested(device) ? 1 : 0);
    builder.finish({ output }, "arbitrary-learned-spectrum");
    const auto source = values(size_t(ggml_nelements(input)), 17);
    builder.upload(input, source);
    const ResidentInputs resident(output);
    builder.compute();
    if (!resident.unchanged("arbitrary-learned-spectrum")) return;
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
    const ResidentInputs updated(output);
    const uint64_t uploads = builder.upload_bytes(), downloads = builder.download_bytes();
    builder.compute();
    if (!updated.unchanged("arbitrary-learned-spectrum repeat")) return;
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
              bool extreme, float normalizer = 16.0f, int chunk_size = 64) {
    const bool native_cpu = cpu_gla_requested(device);
    const std::string label = "gla-" + std::to_string(width * height) + "-k" + std::to_string(dk) + "-v" +
                              std::to_string(dv) + "-h" + std::to_string(heads) + "-b" + std::to_string(batches) +
                              (native_cpu ? "-cpu-recurrent" : "-chunk" + std::to_string(chunk_size)) +
                              (extreme ? "-extreme" : "");
    inpaint_graph::Builder builder(device);
    auto * q = builder.parameter("q", { width, height, dk * heads, batches });
    auto * k = builder.parameter("k", { width, height, dk * heads, batches });
    auto * v = builder.parameter("v", { width, height, dv * heads, batches });
    auto * gates = builder.parameter("gates", { width, height, dk * heads, batches });
    auto * output = inpaint_graph::gla(builder, q, k, v, gates, heads, normalizer, chunk_size);
    graph_contract(output, label.c_str(), native_cpu);
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
    const ResidentInputs resident(output);
    const uint64_t uploads = builder.upload_bytes();
    builder.compute();
    if (!resident.unchanged(label)) return;
    check(builder.upload_bytes() == uploads && builder.download_bytes() == 0,
          label + " carries recurrent state entirely on the selected backend");
    auto actual = builder.download(output);
    const auto expected = gla_reference(qv, kv, vv, gv, width, height, dk, dv, heads, batches, normalizer);
    const Metrics metric = compare(actual, expected);
    check(metric.finite && metric.maximum <= gla_max_abs && metric.relative_l2 <= gla_max_relative_l2 &&
              metric.cosine >= gla_min_cosine,
          label + " matches independent token-by-token F32 recurrence");

    const int tokens = width * height;
    // Cross the selected chunk boundary as well as checking ordinary causal
    // masking within a chunk. The 65-token case exercises every allowed tail.
    const int boundary = tokens > chunk_size ? std::min(tokens - 1, chunk_size + 1) : std::max(1, tokens / 2);
    for (int batch = 0; batch < batches; ++batch)
        for (int ch = 0; ch < dv * heads; ++ch)
            for (int token = boundary; token < tokens; ++token)
                vv[(size_t(batch) * dv * heads + ch) * tokens + token] += 0.25f;
    const size_t nodes = builder.nodes(), allocation = builder.allocation_bytes();
    builder.upload(v, vv);
    const ResidentInputs updated(output);
    const uint64_t repeated_uploads = builder.upload_bytes(), downloads = builder.download_bytes();
    builder.compute();
    if (!updated.unchanged(label + " repeat")) return;
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
    rejects([&] { (void)inpaint_graph::gla(builder, q, q, v, q, 2, 16.0f, 3); },
            "unbounded GLA chunk selection is rejected");
}

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct BenchmarkOptions {
    std::string profile;
    std::string operation = "all";
    std::vector<int> chunks{ 64 };
    int threads = 2;
    int warmup = 2;
    int iterations = 5;
};

int bounded_integer(const std::string & text, int minimum, int maximum, const char * label) {
    if (text.empty() || text.size() > 6 || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(std::string(label) + " requires a bounded nonnegative integer");
    const int result = std::stoi(text);
    if (result < minimum || result > maximum)
        throw std::runtime_error(std::string(label) + " exceeds its permitted range");
    return result;
}

double median(std::vector<double> data) {
    std::sort(data.begin(), data.end());
    const size_t middle = data.size() / 2;
    return data.size() % 2 ? data[middle] : (data[middle - 1] + data[middle]) / 2;
}

bool fixed_budget(const Metrics & metric, bool attention) {
    return metric.finite && metric.maximum <= (attention ? gla_max_abs : fft_max_abs) &&
           (!attention || (metric.relative_l2 <= gla_max_relative_l2 && metric.cosine >= gla_min_cosine));
}

void benchmark_graph(inpaint_graph::Builder & builder, ggml_tensor * output, const std::string & label,
                     const std::vector<std::pair<ggml_tensor *, const std::vector<float> *>> & inputs,
                     const std::vector<float> & expected, Clock::time_point construction_started,
                     const BenchmarkOptions & options, bool attention, int chunk_size, double reference_ms) {
    const GraphOperators operators =
        graph_contract(output, label.c_str(), attention && cpu_gla_requested(builder.device),
                       !attention && cpu_fft_requested(builder.device) ? 1 : 0);
    const char * implementation = operators.native_recurrences ? "cpu-recurrent"
                                  : operators.cpu_ffts         ? "cpu-fft"
                                  : attention                  ? "portable-chunk"
                                                               : "portable-fft";
    const char * arithmetic = operators.cpu_ffts ? "f32-storage-f64-fft" : "f32";
    builder.finish({ output }, label);
    const double build_ms = elapsed_ms(construction_started);
    const auto upload_started = Clock::now();
    for (const auto & entry : inputs) builder.upload(entry.first, *entry.second);
    const double upload_ms = elapsed_ms(upload_started);
    const auto cold_started = Clock::now();
    builder.compute();
    const double cold_ms = elapsed_ms(cold_started);
    const auto download_started = Clock::now();
    const auto first = builder.download(output);
    double download_ms = elapsed_ms(download_started);
    const Metrics cold = compare(first, expected);
    const bool cold_pass = fixed_budget(cold, attention);
    check(cold_pass, label + " cold output meets the unchanged independent-reference budget");

    const size_t nodes = builder.nodes(), allocation = builder.allocation_bytes();
    const uint64_t uploaded = builder.upload_bytes(), downloaded = builder.download_bytes();
    for (int i = 0; i < options.warmup; ++i) builder.compute();
    std::vector<double> samples;
    samples.reserve(size_t(options.iterations));
    for (int i = 0; i < options.iterations; ++i) {
        const auto started = Clock::now();
        builder.compute();
        samples.push_back(elapsed_ms(started));
    }
    const bool reused = builder.nodes() == nodes && builder.allocation_bytes() == allocation &&
                        builder.upload_bytes() == uploaded && builder.download_bytes() == downloaded &&
                        builder.runs() == uint64_t(1 + options.warmup + options.iterations);
    check(reused, label + " warm timing reuses the graph without activation transfers");
    const auto final_download = Clock::now();
    const auto last = builder.download(output);
    download_ms += elapsed_ms(final_download);
    const Metrics warm = compare(last, expected);
    const bool warm_pass = fixed_budget(warm, attention);
    check(warm_pass, label + " warm output meets the same independent-reference budget");
    const bool bit_exact =
        first.size() == last.size() && std::memcmp(first.data(), last.data(), first.size() * sizeof(float)) == 0;
    // The scalar reference duration is reported separately. It includes its
    // wrapper allocation/copies and is not a directly comparable speedup.
    std::printf("SPECTRAL_BENCHMARK {\"case\":\"%s\",\"backend\":\"%s\",\"arithmetic\":\"%s\","
                "\"implementation\":\"%s\",\"native_recurrent_nodes\":%zu,\"native_fft_nodes\":%zu,"
                "\"cpu_fft_scratch_bytes_per_worker\":%zu,\"graph_allocation_excludes_cpu_fft_plans_and_stack\":%s,"
                "\"threads\":%d,\"chunk_size\":%d,\"warmup\":%d,\"iterations\":%d,"
                "\"build_ms\":%.9g,\"input_upload_ms\":%.9g,\"cold_compute_ms\":%.9g,"
                "\"warm_median_compute_ms\":%.9g,\"warm_min_compute_ms\":%.9g,\"warm_max_compute_ms\":%.9g,"
                "\"two_output_downloads_ms\":%.9g,\"independent_reference_ms\":%.9g,"
                "\"nodes\":%zu,\"allocation_bytes\":%zu,\"input_bytes\":%llu,\"output_bytes\":%zu,"
                "\"cold_max_abs\":%.9g,\"warm_max_abs\":%.9g,\"warm_relative_l2\":%.9g,"
                "\"warm_cosine\":%.12g,\"repeat_bit_exact\":%s,\"allocation_reused\":%s,\"passed\":%s,"
                "\"model_inference_performed\":false,\"full_model_qualified\":false,\"samples_compute_ms\":[",
                label.c_str(), builder.device.name.c_str(), arithmetic, implementation, operators.native_recurrences,
                operators.cpu_ffts, operators.cpu_ffts ? (size_t(128 * 128 + 128 + 256) * sizeof(Complex)) : 0,
                operators.cpu_ffts ? "true" : "false", options.threads, operators.native_recurrences ? 0 : chunk_size,
                options.warmup, options.iterations, build_ms, upload_ms, cold_ms, median(samples),
                *std::min_element(samples.begin(), samples.end()), *std::max_element(samples.begin(), samples.end()),
                download_ms, reference_ms, nodes, allocation, (unsigned long long)uploaded, last.size() * sizeof(float),
                cold.maximum, warm.maximum, warm.relative_l2, warm.cosine, bit_exact ? "true" : "false",
                reused ? "true" : "false", cold_pass && warm_pass && reused ? "true" : "false");
    for (size_t i = 0; i < samples.size(); ++i) std::printf("%s%.9g", i ? "," : "", samples[i]);
    std::printf("]}\n");
    std::fflush(stdout);
}

void benchmark_fft(inpaint_graph::Device & device, int width, int height, int channels,
                   const BenchmarkOptions & options) {
    constexpr int batches = 1;
    const std::string geometry =
        std::to_string(width) + "x" + std::to_string(height) + "-c" + std::to_string(channels) + "-b1";
    const auto real = values(size_t(width) * height * channels, 41);
    const auto reference_started = Clock::now();
    const auto spectrum = fft_reference(real, width, height, channels, batches);
    const double reference_ms = elapsed_ms(reference_started);
    {
        const auto started = Clock::now();
        inpaint_graph::Builder builder(device);
        auto * input = builder.parameter("real", { width, height, channels, batches });
        auto * output = inpaint_graph::rfft2_ortho(builder, input);
        benchmark_graph(builder, output, "fft.forward-" + geometry, { { input, &real } }, spectrum, started, options,
                        false, 0, reference_ms);
    }
    // A learned Fourier layer may destroy the Hermitian DC/Nyquist relation;
    // benchmark the actual inverse contract rather than only a round trip.
    const auto learned = values(spectrum.size(), 43);
    const auto inverse_reference_started = Clock::now();
    const auto expected =
        lama::detail::irfft2_ortho(batch_tensor(learned, width / 2 + 1, height, 2 * channels, 0), width);
    const double inverse_reference_ms = elapsed_ms(inverse_reference_started);
    {
        const auto started = Clock::now();
        inpaint_graph::Builder builder(device);
        auto * input = builder.parameter("learned-spectrum", { width / 2 + 1, height, channels * 2, batches });
        auto * output = inpaint_graph::irfft2_ortho(builder, input, width);
        benchmark_graph(builder, output, "fft.inverse-" + geometry, { { input, &learned } }, expected.v, started,
                        options, false, 0, inverse_reference_ms);
    }
}

void benchmark_gla(inpaint_graph::Device & device, int width, int height, int dk, int dv, int heads, int batches,
                   const BenchmarkOptions & options) {
    const size_t tokens = size_t(width) * height;
    const auto qv = values(tokens * dk * heads * batches, 47);
    const auto kv = values(qv.size(), 53);
    const auto vv = values(tokens * dv * heads * batches, 59);
    auto gv = values(qv.size(), 61);
    for (float & value : gv) value = 5 * value - 1;
    const auto reference_started = Clock::now();
    const auto expected = gla_reference(qv, kv, vv, gv, width, height, dk, dv, heads, batches, 16.0f);
    const double reference_ms = elapsed_ms(reference_started);
    const std::string geometry = std::to_string(tokens) + "-k" + std::to_string(dk) + "-v" + std::to_string(dv) + "-h" +
                                 std::to_string(heads) + "-b" + std::to_string(batches);
    const bool native_cpu = cpu_gla_requested(device);
    const auto chunks = native_cpu ? std::vector<int>{ 64 } : options.chunks;
    for (int chunk : chunks) {
        const auto started = Clock::now();
        inpaint_graph::Builder builder(device);
        auto * q = builder.parameter("q", { width, height, dk * heads, batches });
        auto * k = builder.parameter("k", { width, height, dk * heads, batches });
        auto * v = builder.parameter("v", { width, height, dv * heads, batches });
        auto * gate = builder.parameter("gate", { width, height, dk * heads, batches });
        auto * output = inpaint_graph::gla(builder, q, k, v, gate, heads, 16.0f, chunk);
        benchmark_graph(builder, output,
                        "gla-" + geometry + (native_cpu ? "-cpu-recurrent" : "-chunk" + std::to_string(chunk)),
                        { { q, &qv }, { k, &kv }, { v, &vv }, { gate, &gv } }, expected, started, options, true, chunk,
                        reference_ms);
    }
}

void benchmark(inpaint_graph::Device & device, const BenchmarkOptions & options) {
    const bool models = options.profile == "model-shapes";
    if (options.operation != "gla") {
        if (models) {
            // Actual LaMa Fourier-unit geometry at 256/512-pixel input plus an
            // odd bottleneck width. Only one graph is live at any one time.
            benchmark_fft(device, 32, 32, 192, options);
            benchmark_fft(device, 64, 64, 192, options);
            benchmark_fft(device, 33, 32, 192, options);
        } else {
            benchmark_fft(device, 32, 32, 16, options);
            benchmark_fft(device, 33, 32, 8, options);
        }
    }
    if (options.operation != "fft") {
        if (models) {
            // PixelHacker's three real head widths, including CFG batch two.
            // Largest paired-gate tensor is ~20MiB (dk80/chunk64/16 heads).
            benchmark_gla(device, 64, 64, 20, 40, 8, 2, options);
            benchmark_gla(device, 32, 32, 40, 80, 8, 2, options);
            benchmark_gla(device, 16, 16, 80, 160, 8, 2, options);
        } else {
            benchmark_gla(device, 16, 16, 20, 40, 8, 2, options);
            benchmark_gla(device, 33, 5, 7, 11, 3, 2, options);
        }
    }
}

int run(int argc, char ** argv) {
    std::string backend = "cpu";
    BenchmarkOptions options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if ((argument == "--backend" || argument == "--gpu-backend") && i + 1 < argc)
            backend = argv[++i];
        else if (argument == "--benchmark" && i + 1 < argc)
            options.profile = argv[++i];
        else if (argument == "--benchmark-op" && i + 1 < argc)
            options.operation = argv[++i];
        else if (argument == "--threads" && i + 1 < argc)
            options.threads = bounded_integer(argv[++i], 1, 64, "--threads");
        else if (argument == "--warmup" && i + 1 < argc)
            options.warmup = bounded_integer(argv[++i], 0, 100, "--warmup");
        else if (argument == "--iterations" && i + 1 < argc)
            options.iterations = bounded_integer(argv[++i], 1, 100, "--iterations");
        else if (argument == "--gla-chunk" && i + 1 < argc) {
            const std::string choice = argv[++i];
            options.chunks = choice == "all" ? std::vector<int>{ 8, 16, 32, 64 }
                                             : std::vector<int>{ bounded_integer(choice, 8, 64, "--gla-chunk") };
            for (int chunk : options.chunks)
                if (chunk != 8 && chunk != 16 && chunk != 32 && chunk != 64)
                    throw std::runtime_error("--gla-chunk requires 8, 16, 32, 64 or all");
        } else
            throw std::runtime_error("usage: test-inpaint-graph-spectral [--backend cpu|cuda|metal] [--threads 1..64]\n"
                                     "  [--benchmark small|model-shapes] [--benchmark-op fft|gla|all]\n"
                                     "  [--gla-chunk 8|16|32|64|all] [--warmup 0..100] [--iterations 1..100]");
    }
    if (backend != "cpu" && backend != "cuda" && backend != "metal")
        throw std::runtime_error("unsupported explicit test backend");
    if ((!options.profile.empty() && options.profile != "small" && options.profile != "model-shapes") ||
        (options.operation != "all" && options.operation != "fft" && options.operation != "gla"))
        throw std::runtime_error("unsupported benchmark profile or operation");
    crispasr_set_gpu_backend_pref(backend.c_str());
    inpaint_graph::Device device(options.threads);
    check(device.gpu == (backend != "cpu"), "explicit backend request is honored without CPU fallback");
    if (!options.profile.empty()) {
        benchmark(device, options);
        std::printf("{\"suite\":\"inpaint-graph-spectral-benchmark\",\"backend\":\"%s\","
                    "\"checks\":%d,\"failures\":%d,\"native_model_qualified\":false}\n",
                    device.name.c_str(), checks, failures);
        return failures ? 1 : 0;
    }
    rejected_geometry(device);
    fft_case(device, 1, 1, 2, 2);
    fft_case(device, 8, 8, 3, 2);
    fft_case(device, 5, 3, 2, 2);
    fft_case(device, 10, 7, 2, 1);
    fft_case(device, 9, 16, 1, 2);
    fft_case(device, 127, 128, 1, 1);
    learned_spectrum_case(device, 5, 3);
    learned_spectrum_case(device, 6, 5);
    const auto chunks = cpu_gla_requested(device) ? std::vector<int>{ 64 } : std::vector<int>{ 8, 16, 32, 64 };
    for (int chunk : chunks) {
        gla_case(device, 3, 3, 1, 3, 1, 2, false, 16.0f, chunk);
        gla_case(device, 1, 1, 3, 5, 2, 2, false, 16.0f, chunk);
        gla_case(device, 13, 7, 5, 3, 2, 2, false, 16.0f, chunk);
        gla_case(device, 33, 5, 7, 11, 3, 1, false, 4.0f, chunk);
        gla_case(device, 8, 9, 20, 40, 8, 2, false, 16.0f, chunk);
        gla_case(device, 3, 5, 17, 33, 2, 2, false, 16.0f, chunk);
        gla_case(device, 65, 1, 4, 7, 2, 2, true, 16.0f, chunk);
    }
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
