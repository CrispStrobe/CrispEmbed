// Hosted, model-free checks of the actual UNet device-graph kernels. Expected
// values use direct scalar definitions, independent of graph transpositions.
#include "latent_inpaint_graph_unet.h"
#include "core/clean_exit.h"
#include "core/gpu_backend_pref.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using inpaint_graph::Builder;
using inpaint_graph::Device;

void check(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

void rejects(const std::function<void()> & fn) {
    try {
        fn();
    } catch (const std::exception &) {
        return;
    }
    throw std::runtime_error("malformed graph input was accepted");
}

struct Fixture {
    int w = 3, h = 2, tokens = 5, dk = 3, dv = 4, heads = 2, batch = 2;
    std::vector<float> q, k, v, positional, relative, gate, scale;

    int n() const { return w * h; }
    size_t index(int sample, int channel, int position, int channels, int count) const {
        return (size_t(sample) * channels + channel) * count + position;
    }

    explicit Fixture(int head_count, int width = 3, int height = 2, int key_channels = 3, int value_channels = 4,
                     int token_count = 5, int samples = 2)
        : w(width), h(height), tokens(token_count), dk(key_channels), dv(value_channels), heads(head_count),
          batch(samples) {
        q.resize(size_t(n()) * heads * dk * batch);
        k.resize(size_t(tokens) * dk * batch);
        v.resize(size_t(tokens) * dv * batch);
        positional.resize(size_t(n()) * dk * dv * batch);
        relative.resize(size_t(n()) * tokens * dk);
        gate.resize(size_t(n()) * heads * dv * batch);
        scale.resize(size_t(dv));
        fill(q, 23, 0.07f, 0.03f);
        fill(k, 17, 0.13f, -0.11f);
        fill(v, 19, 0.09f, 0.17f);
        fill(positional, 29, 0.011f, -0.07f);
        fill(relative, 31, 0.019f, 0.09f);
        fill(gate, 13, 0.21f, -0.04f);
        fill(scale, 7, 0.11f, 0.63f);
        // A zero query at one position exercises RMSNorm's epsilon branch.
        for (int sample = 0; sample < batch; ++sample)
            for (int c = 0; c < heads * dk; ++c) q[index(sample, c, 0, heads * dk, n())] = 0;
    }

    static void fill(std::vector<float> & values, int modulus, float factor, float offset) {
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = float(int((i * 7 + 3) % size_t(modulus)) - modulus / 2) * factor + offset;
    }

    std::vector<float> content() const {
        std::vector<float> out(size_t(n()) * heads * dv * batch);
        std::vector<double> normalized(size_t(tokens) * dk);
        for (int sample = 0; sample < batch; ++sample) {
            for (int d = 0; d < dk; ++d) {
                double maximum = -1e100, total = 0;
                for (int token = 0; token < tokens; ++token)
                    maximum = std::max(maximum, double(k[index(sample, d, token, dk, tokens)]));
                for (int token = 0; token < tokens; ++token) {
                    const double value = std::exp(double(k[index(sample, d, token, dk, tokens)]) - maximum);
                    normalized[size_t(d) * tokens + token] = value;
                    total += value;
                }
                for (int token = 0; token < tokens; ++token) normalized[size_t(d) * tokens + token] /= total;
            }
            for (int head = 0; head < heads; ++head)
                for (int value = 0; value < dv; ++value)
                    for (int p = 0; p < n(); ++p) {
                        double sum = 0;
                        for (int key = 0; key < dk; ++key)
                            for (int token = 0; token < tokens; ++token)
                                sum += double(q[index(sample, head * dk + key, p, heads * dk, n())]) *
                                       normalized[size_t(key) * tokens + token] *
                                       v[index(sample, value, token, dv, tokens)];
                        out[index(sample, head * dv + value, p, heads * dv, n())] = float(sum);
                    }
        }
        return out;
    }

    std::vector<float> local() const {
        std::vector<float> out(size_t(n()) * heads * dv * batch);
        for (int sample = 0; sample < batch; ++sample)
            for (int head = 0; head < heads; ++head)
                for (int value = 0; value < dv; ++value)
                    for (int p = 0; p < n(); ++p) {
                        double sum = 0;
                        for (int key = 0; key < dk; ++key) {
                            const size_t pi = ((size_t(sample) * dv + value) * dk + key) * n() + p;
                            sum += double(q[index(sample, head * dk + key, p, heads * dk, n())]) * positional[pi];
                        }
                        out[index(sample, head * dv + value, p, heads * dv, n())] = float(sum);
                    }
        return out;
    }

    std::vector<float> learned() const {
        std::vector<float> out(size_t(n()) * heads * dv * batch);
        for (int sample = 0; sample < batch; ++sample)
            for (int head = 0; head < heads; ++head)
                for (int value = 0; value < dv; ++value)
                    for (int p = 0; p < n(); ++p) {
                        double sum = 0;
                        for (int key = 0; key < dk; ++key)
                            for (int token = 0; token < tokens; ++token)
                                sum += double(q[index(sample, head * dk + key, p, heads * dk, n())]) *
                                       relative[(size_t(p) * tokens + token) * dk + key] *
                                       v[index(sample, value, token, dv, tokens)];
                        out[index(sample, head * dv + value, p, heads * dv, n())] = float(sum);
                    }
        return out;
    }

    // Direct double-precision definition for selected large-model positions.
    // This does not use either matrix association from the tested graphs.
    float learned_at(size_t flat_index) const {
        const int p = int(flat_index % size_t(n()));
        const int channel = int((flat_index / size_t(n())) % size_t(heads * dv));
        const int sample = int(flat_index / (size_t(n()) * heads * dv));
        const int head = channel / dv, value = channel % dv;
        double sum = 0;
        for (int key = 0; key < dk; ++key)
            for (int token = 0; token < tokens; ++token)
                sum += double(q[index(sample, head * dk + key, p, heads * dk, n())]) *
                       relative[(size_t(p) * tokens + token) * dk + key] * v[index(sample, value, token, dv, tokens)];
        return float(sum);
    }

    std::vector<float> normalized(const std::vector<float> & x) const {
        std::vector<float> out(x.size());
        for (int sample = 0; sample < batch; ++sample)
            for (int head = 0; head < heads; ++head)
                for (int p = 0; p < n(); ++p) {
                    double square_sum = 0;
                    for (int value = 0; value < dv; ++value) {
                        const float a = x[index(sample, head * dv + value, p, heads * dv, n())];
                        square_sum += double(a) * a;
                    }
                    const double inv_rms = 1.0 / std::sqrt(square_sum / dv + 1e-6);
                    for (int value = 0; value < dv; ++value) {
                        const size_t i = index(sample, head * dv + value, p, heads * dv, n());
                        out[i] =
                            float(double(x[i]) * inv_rms * scale[value] * gate[i] / (1.0 + std::exp(-double(gate[i]))));
                    }
                }
        return out;
    }
};

struct LocalFixture {
    int w, h, dk, dv, heads, side, batch = 2;
    std::vector<float> q, v, kernel, bias;

    LocalFixture(int width, int height, int keys, int values, int head_count, int kernel_side, bool with_bias)
        : w(width), h(height), dk(keys), dv(values), heads(head_count), side(kernel_side) {
        q.resize(size_t(w) * h * dk * heads * batch);
        v.resize(size_t(w) * h * dv * batch);
        kernel.resize(size_t(side) * side * dk);
        if (with_bias) bias.resize(size_t(dk));
        Fixture::fill(q, 23, 0.071f, -0.03f);
        Fixture::fill(v, 19, 0.089f, 0.17f);
        Fixture::fill(kernel, 31, 0.013f, -0.027f);
        Fixture::fill(bias, 7, 0.043f, 0.12f);
    }

    size_t elements() const { return size_t(w) * h * heads * dv * batch; }

    // Literal zero-padded convolution followed by the learned query reduction.
    // It does not use an im2col buffer or either graph contraction layout.
    float at(size_t flat_index) const {
        const size_t spatial = size_t(w) * h;
        const int x = int(flat_index % spatial) % w, y = int(flat_index % spatial) / w;
        const int channel = int((flat_index / spatial) % size_t(heads * dv));
        const int sample = int(flat_index / (spatial * heads * dv));
        const int head = channel / dv, value = channel % dv;
        double sum = 0;
        for (int key = 0; key < dk; ++key) {
            double convolution = bias.empty() ? 0.0 : double(bias[size_t(key)]);
            for (int ky = 0; ky < side; ++ky) {
                const int iy = y + ky - side / 2;
                if (iy < 0 || iy >= h) continue;
                for (int kx = 0; kx < side; ++kx) {
                    const int ix = x + kx - side / 2;
                    if (ix < 0 || ix >= w) continue;
                    const size_t wi = (size_t(key) * side + ky) * side + kx;
                    const size_t vi = (size_t(sample) * dv + value) * spatial + size_t(iy) * w + ix;
                    convolution += double(kernel[wi]) * v[vi];
                }
            }
            const size_t qi = (size_t(sample) * heads * dk + head * dk + key) * spatial + size_t(y) * w + x;
            sum += double(q[qi]) * convolution;
        }
        return float(sum);
    }

    std::vector<float> expected() const {
        std::vector<float> result(elements());
        for (size_t i = 0; i < result.size(); ++i) result[i] = at(i);
        return result;
    }
};

void compare(const std::string & name, const std::vector<float> & actual, const std::vector<float> & expected,
             bool fast) {
    check(actual.size() == expected.size(), name + " element count differs");
    // Fixed before execution. The wider mixed-profile component diagnostic
    // does not alter the independent full-model parity acceptance budgets.
    const float absolute = fast ? 3e-3f : 3e-5f;
    const float relative = fast ? 3e-3f : 3e-5f;
    for (size_t i = 0; i < actual.size(); ++i) {
        const float budget = absolute + relative * std::abs(expected[i]);
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > budget)
            throw std::runtime_error(name + " scalar-reference mismatch at element " + std::to_string(i) + ": actual=" +
                                     std::to_string(actual[i]) + " expected=" + std::to_string(expected[i]));
    }
}

void kernels(Device & device, int heads) {
    using namespace latent_inpaint::graph_detail;
    Fixture f(heads);
    Builder b(device);
    auto * q = b.parameter("query", { f.w, f.h, f.heads * f.dk, f.batch });
    auto * k = b.constant("key", { f.tokens, 1, f.dk, f.batch }, f.k);
    auto * v = b.constant("value", { f.tokens, 1, f.dv, f.batch }, f.v);
    auto * pos = b.constant("position", { f.w, f.h, f.dk, f.dv * f.batch }, f.positional);
    auto * rel = b.constant("relative", { 1, f.dk, f.tokens, f.n() }, f.relative);
    auto * gate = b.constant("gate", { f.w, f.h, f.heads * f.dv, f.batch }, f.gate);
    auto * scale = b.constant("scale", { f.dv }, f.scale);
    auto * content = lambda_content(b, q, k, v, f.heads);
    auto * local = lambda_position(b, q, pos, f.heads, f.dv);
    auto * learned = lambda_relative(b, q, v, rel, f.heads);
    auto * reassociated = lambda_relative_reassociated(b, q, v, rel, f.heads);
    auto * norm = gated_rms_norm(b, content, gate, scale, f.heads);
    for (auto * output : { content, local, learned, reassociated, norm }) {
        check(output->ne[0] == f.w && output->ne[1] == f.h && output->ne[2] == f.heads * f.dv &&
                  output->ne[3] == f.batch && ggml_is_contiguous(output),
              "graph kernel WHCN output layout differs");
    }
    b.finish({ content, local, learned, reassociated, norm }, "latent-unet-analytical");
    const size_t nodes = b.nodes(), allocation = b.allocation_bytes();
    check(nodes > 0 && allocation > 0, "graph did not allocate reusable backend storage");
    const uint64_t initial_upload = b.upload_bytes();
    for (int run = 0; run < 2; ++run) {
        if (run == 1)
            for (size_t i = 0; i < f.q.size(); ++i) f.q[i] = f.q[i] * -0.73f + float(i % 5) * 0.021f;
        b.upload(q, f.q);
        b.compute();
        const auto expected_content = f.content();
        compare("lambda content", b.download(content), expected_content, device.fast);
        compare("local positional lambda", b.download(local), f.local(), device.fast);
        const auto expected_relative = f.learned();
        compare("learned relative lambda", b.download(learned), expected_relative, device.fast);
        compare("reassociated relative lambda", b.download(reassociated), expected_relative, device.fast);
        compare("per-head gated RMSNorm", b.download(norm), f.normalized(expected_content), device.fast);
        // INPUT schedules an upload before execution; it does not by itself
        // forbid gallocr from reclaiming the tensor after its last consumer.
        // Persistent graphs must preserve every uploaded-once input exactly,
        // including constants used by in-place-capable softmax/SiLU nodes.
        check(b.download(q) == f.q && b.download(k) == f.k && b.download(v) == f.v && b.download(pos) == f.positional &&
                  b.download(rel) == f.relative && b.download(gate) == f.gate && b.download(scale) == f.scale,
              "persistent graph input or constant was overwritten during execution");
        check(b.runs() == uint64_t(run + 1) && b.nodes() == nodes && b.allocation_bytes() == allocation,
              "graph execution rebuilt nodes or changed its persistent allocation");
        check(b.upload_bytes() == initial_upload + uint64_t(run + 1) * f.q.size() * sizeof(float),
              "a graph replay uploaded more than its changed query input");
    }
}

struct LocalGraph {
    struct KernelWeights {
        std::filesystem::path directory;
        std::unique_ptr<inpaint_graph::Weights> loaded;

        KernelWeights(Device & device, const LocalFixture & f) {
            directory =
                std::filesystem::temp_directory_path() /
                ("latent-local-kernel-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
            check(std::filesystem::create_directory(directory), "cannot create local-kernel fixture directory");
            try {
                const auto path = directory / "kernel.gguf";
                const size_t bytes = (size_t(1) << 20) + (f.kernel.size() + f.bias.size()) * sizeof(float);
                std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ bytes, nullptr, false }),
                                                                        ggml_free);
                std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
                check(bool(ctx) && bool(file), "cannot allocate local-kernel GGUF fixture");
                auto * kernel = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, f.side, f.side, 1, f.dk);
                ggml_set_name(kernel, "position.weight");
                std::memcpy(kernel->data, f.kernel.data(), f.kernel.size() * sizeof(float));
                gguf_add_tensor(file.get(), kernel);
                inpaint_graph::ShapeMap schema{ { "position.weight", { f.dk, 1, f.side, f.side } } };
                if (!f.bias.empty()) {
                    auto * bias = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, f.dk);
                    ggml_set_name(bias, "position.bias");
                    std::memcpy(bias->data, f.bias.data(), f.bias.size() * sizeof(float));
                    gguf_add_tensor(file.get(), bias);
                    schema.emplace("position.bias", std::vector<int64_t>{ f.dk });
                }
                check(gguf_write_to_file(file.get(), path.string().c_str(), false),
                      "cannot write local-kernel GGUF fixture");
                loaded = std::make_unique<inpaint_graph::Weights>(device, path.string().c_str(), schema);
            } catch (...) {
                std::error_code error;
                std::filesystem::remove_all(directory, error);
                throw;
            }
        }

        ~KernelWeights() {
            loaded.reset();
            std::error_code error;
            std::filesystem::remove_all(directory, error);
        }
    } weights;
    Builder builder;
    ggml_tensor *query = nullptr, *value = nullptr, *kernel = nullptr, *bias = nullptr, *output = nullptr;
    std::vector<uint8_t> kernel_bytes, bias_bytes;

    static std::vector<uint8_t> bytes(ggml_tensor * tensor) {
        std::vector<uint8_t> result(tensor ? ggml_nbytes(tensor) : 0);
        if (tensor) ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
        return result;
    }

    LocalGraph(Device & device, const LocalFixture & f, bool reassociate, bool pad_cpu_gemm = false)
        : weights(device, f), builder(device, weights.loaded.get()) {
        using namespace latent_inpaint::graph_detail;
        query = builder.parameter("query", { f.w, f.h, f.heads * f.dk, f.batch });
        value = builder.constant("value", { f.w, f.h, f.dv, f.batch }, f.v);
        kernel = builder.weight("position.weight");
        if (!f.bias.empty()) bias = builder.weight("position.bias");
        kernel_bytes = bytes(kernel);
        bias_bytes = bytes(bias);
        if (reassociate) {
            output = lambda_local_reassociated(builder, query, value, kernel, bias, f.heads, pad_cpu_gemm);
        } else {
            check(!pad_cpu_gemm, "padding requires local reassociation");
            auto * planes = ggml_reshape_4d(builder.ctx, value, f.w, f.h, 1, f.dv * f.batch);
            auto * positional = builder.conv("position", planes, 1, f.side / 2);
            output = lambda_position(builder, query, positional, f.heads, f.dv);
        }
        check(output->ne[0] == f.w && output->ne[1] == f.h && output->ne[2] == f.heads * f.dv &&
                  output->ne[3] == f.batch && ggml_is_contiguous(output),
              "local positional graph output layout differs");
        builder.finish({ output }, pad_cpu_gemm  ? "moebius-local-padded"
                                   : reassociate ? "moebius-local-reassociated"
                                                 : "moebius-local-baseline");
        builder.upload(query, f.q);
    }

    void unchanged_inputs(const LocalFixture & f) {
        check(builder.download(query) == f.q && builder.download(value) == f.v && bytes(kernel) == kernel_bytes &&
                  bytes(bias) == bias_bytes,
              "local positional graph overwrote persistent input or kernel bytes");
    }
};

void local_kernels(Device & device) {
    for (const LocalFixture & f : { LocalFixture(5, 3, 3, 2, 2, 3, true), LocalFixture(3, 4, 5, 2, 2, 15, false),
                                    LocalFixture(17, 31, 2, 65, 2, 15, true) }) {
        // The final geometry crosses the real 32 MiB CPU strip budget, with
        // a partial final strip. Its independent scalar oracle covers every
        // output, including edges, the strip seam, both heads and both samples.
        LocalGraph baseline(device, f, false), candidate(device, f, true);
        std::unique_ptr<LocalGraph> padded;
        if (!device.gpu) padded = std::make_unique<LocalGraph>(device, f, true, true);
        std::vector<LocalGraph *> graphs{ &baseline, &candidate };
        if (padded) graphs.push_back(padded.get());
        const auto expected = f.expected();
        std::vector<size_t> allocations;
        std::vector<uint64_t> uploads;
        std::vector<std::vector<float>> first(graphs.size());
        for (const auto * graph : graphs) {
            allocations.push_back(graph->builder.allocation_bytes());
            uploads.push_back(graph->builder.upload_bytes());
        }
        for (int replay = 0; replay < 2; ++replay) {
            for (size_t variant = 0; variant < graphs.size(); ++variant) {
                LocalGraph & graph = *graphs[variant];
                graph.builder.compute();
                const auto actual = graph.builder.download(graph.output);
                compare("local convolution variant " + std::to_string(variant), actual, expected, device.fast);
                if (replay == 0)
                    first[variant] = actual;
                else
                    check(actual == first[variant], "local convolution replay output is not bit-exact");
                graph.unchanged_inputs(f);
                check(graph.builder.allocation_bytes() == allocations[variant] &&
                          graph.builder.upload_bytes() == uploads[variant],
                      "local convolution replay rebuilt storage or uploaded unchanged inputs");
            }
        }
    }
}

void invalid_inputs(Device & device) {
    using namespace latent_inpaint::graph_detail;
    Builder b(device);
    auto * q = b.parameter("query", { 3, 2, 6, 2 });
    auto * k = b.parameter("key", { 5, 1, 3, 2 });
    auto * v = b.parameter("value", { 5, 1, 4, 2 });
    auto * wrong_batch = b.parameter("wrong_batch", { 5, 1, 4, 1 });
    auto * wrong_relative = b.parameter("wrong_relative", { 1, 3, 5, 5 });
    auto * wrong_position = b.parameter("wrong_position", { 3, 2, 3, 7 });
    auto * wrong_scale = b.parameter("wrong_scale", { 8 });
    auto * local_value = b.parameter("local_value", { 3, 2, 4, 2 });
    auto * local_kernel = b.parameter("local_kernel", { 3, 3, 1, 3 });
    auto * even_kernel = b.parameter("even_kernel", { 2, 2, 1, 3 });
    auto * wrong_local_bias = b.parameter("wrong_local_bias", { 4 });
    rejects([&] { lambda_content(b, q, k, v, 0); });
    rejects([&] { lambda_content(b, q, k, v, 4); });
    rejects([&] { lambda_content(b, q, k, wrong_batch, 2); });
    rejects([&] { lambda_relative(b, q, v, wrong_relative, 2); });
    rejects([&] { lambda_relative_reassociated(b, q, v, wrong_relative, 2); });
    rejects([&] { lambda_relative_reassociated(b, q, wrong_batch, wrong_relative, 2); });
    rejects([&] { lambda_relative_reassociated(b, q, v, wrong_relative, 0); });
    rejects([&] { lambda_position(b, q, wrong_position, 2, 4); });
    rejects([&] { lambda_local_reassociated(b, q, local_value, local_kernel, nullptr, 0); });
    rejects([&] { lambda_local_reassociated(b, q, v, local_kernel, nullptr, 2); });
    rejects([&] { lambda_local_reassociated(b, q, local_value, even_kernel, nullptr, 2); });
    rejects([&] { lambda_local_reassociated(b, q, local_value, local_kernel, wrong_local_bias, 2); });
    if (device.gpu) {
        rejects([&] { lambda_local_reassociated(b, q, local_value, local_kernel, nullptr, 2, true); });
    } else {
        auto * wide_query = b.parameter("wide_local_query", { 64, 1, 2, 2 });
        auto * wide_values = b.parameter("wide_local_values", { 64, 1, 256, 2 });
        auto * wide_kernel = b.parameter("wide_local_kernel", { 15, 15, 1, 1 });
        rejects([&] { lambda_local_reassociated(b, wide_query, wide_values, wide_kernel, nullptr, 2, true); });
    }
    rejects([&] { gated_rms_norm(b, q, q, wrong_scale, 2); });
    rejects([&] { gated_rms_norm(b, q, v, wrong_scale, 2); });
    // Reject malformed model boundaries before looking up any checkpoint.
    auto * time = b.parameter("time", { 1, 1, 320, 2 });
    auto * tokens = b.parameter("tokens", { 10, 1, 3072, 2 });
    rejects([&] { latent_inpaint::graph_denoise(b, latent_inpaint::Variant::moebius, q, time, tokens); });
    auto * latent = b.parameter("latent", { 64, 64, 9, 2 });
    auto * one_time = b.parameter("one_time", { 1, 1, 320, 1 });
    rejects([&] { latent_inpaint::graph_denoise(b, latent_inpaint::Variant::pixelhacker, latent, one_time, tokens); });
}

struct RelativeGraph {
    Builder builder;
    ggml_tensor * output = nullptr;

    RelativeGraph(Device & device, const Fixture & f, bool reassociate) : builder(device) {
        auto * q = builder.parameter("query", { f.w, f.h, f.heads * f.dk, f.batch });
        auto * v = builder.constant("value", { f.tokens, 1, f.dv, f.batch }, f.v);
        auto * relative = builder.constant("relative", { 1, f.dk, f.tokens, f.n() }, f.relative);
        using namespace latent_inpaint::graph_detail;
        output = reassociate ? lambda_relative_reassociated(builder, q, v, relative, f.heads)
                             : lambda_relative(builder, q, v, relative, f.heads);
        builder.finish({ output }, reassociate ? "moebius-relative-reassociated" : "moebius-relative-baseline");
        builder.upload(q, f.q);
    }
};

double median(std::vector<double> values) {
    check(!values.empty(), "empty benchmark timing sample");
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) * 0.5;
}

void timing_samples(const char * name, const std::vector<double> & values) {
    std::cout << ',' << std::quoted(name) << ":[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

void benchmark_relative(Device & device, int iterations, int threads) {
    // The three actual Moebius resolutions, shared-head feature widths,
    // ten learned tokens, and both CFG branches. Model weights are not needed.
    for (const int side : { 64, 32, 16 }) {
        const int feature = 2560 / side;
        Fixture f(8, side, side, feature, feature, 10, 2);
        RelativeGraph baseline(device, f, false), candidate(device, f, true);
        // Warm both persistent graphs before timing. Allocation and download
        // are excluded from the compute-only samples but reported separately.
        baseline.builder.compute();
        candidate.builder.compute();
        auto original = baseline.builder.download(baseline.output);
        auto optimized = candidate.builder.download(candidate.output);
        compare("model-shape relative reassociation", optimized, original, device.fast);
        std::vector<float> samples_original, samples_optimized, samples_reference;
        constexpr size_t sample_count = 64;
        for (size_t sample = 0; sample < sample_count; ++sample) {
            const size_t index = sample * (original.size() - 1) / (sample_count - 1);
            samples_original.push_back(original[index]);
            samples_optimized.push_back(optimized[index]);
            samples_reference.push_back(f.learned_at(index));
        }
        compare("model-shape baseline scalar samples", samples_original, samples_reference, device.fast);
        compare("model-shape reassociation scalar samples", samples_optimized, samples_reference, device.fast);
        double max_absolute = 0, difference2 = 0, reference2 = 0;
        for (size_t i = 0; i < original.size(); ++i) {
            const double error = double(optimized[i]) - original[i];
            max_absolute = std::max(max_absolute, std::abs(error));
            difference2 += error * error;
            reference2 += double(original[i]) * original[i];
        }
        std::vector<double> baseline_times, candidate_times;
        const size_t baseline_nodes = baseline.builder.nodes(), candidate_nodes = candidate.builder.nodes();
        const size_t baseline_bytes = baseline.builder.allocation_bytes(),
                     candidate_bytes = candidate.builder.allocation_bytes();
        const uint64_t baseline_upload = baseline.builder.upload_bytes(),
                       candidate_upload = candidate.builder.upload_bytes();
        auto timed = [](Builder & builder, std::vector<double> & times) {
            const auto start = std::chrono::steady_clock::now();
            builder.compute();
            times.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        };
        // Alternate order to reduce systematic first/second-run cache bias.
        for (int run = 0; run < iterations; ++run) {
            if (run % 2) {
                timed(candidate.builder, candidate_times);
                timed(baseline.builder, baseline_times);
            } else {
                timed(baseline.builder, baseline_times);
                timed(candidate.builder, candidate_times);
            }
        }
        check(baseline.builder.allocation_bytes() == baseline_bytes &&
                  candidate.builder.allocation_bytes() == candidate_bytes &&
                  baseline.builder.nodes() == baseline_nodes && candidate.builder.nodes() == candidate_nodes &&
                  baseline.builder.upload_bytes() == baseline_upload &&
                  candidate.builder.upload_bytes() == candidate_upload,
              "relative benchmark rebuilt graphs or uploaded unchanged inputs");
        // Repeated executions must preserve their exact first warmed result.
        check(baseline.builder.download(baseline.output) == original &&
                  candidate.builder.download(candidate.output) == optimized,
              "relative benchmark repeated graph output changed");
        const double baseline_ms = median(baseline_times), candidate_ms = median(candidate_times);
        std::cout
            << std::setprecision(9)
            << "{\"schema\":\"crispembed-latent-graph-component-benchmark-v1\",\"component\":\"moebius-relative\","
            << "\"backend\":" << std::quoted(device.name)
            << ",\"precision\":" << std::quoted(device.fast ? "mixed-f16-diagnostic" : "strict-f32")
            << ",\"fixture_weight_type\":\"f32\""
            << ",\"threads\":" << threads << ",\"width\":" << side << ",\"height\":" << side << ",\"dk\":" << feature
            << ",\"dv\":" << feature << ",\"heads\":8,\"tokens\":10,\"batch\":2"
            << ",\"warmups\":1,\"iterations\":" << iterations << ",\"baseline_median_ms\":" << baseline_ms
            << ",\"reassociated_median_ms\":" << candidate_ms << ",\"speedup\":" << baseline_ms / candidate_ms
            << ",\"baseline_allocation_bytes\":" << baseline_bytes
            << ",\"reassociated_allocation_bytes\":" << candidate_bytes
            << ",\"max_absolute_difference\":" << max_absolute
            << ",\"relative_l2_difference\":" << std::sqrt(difference2 / std::max(reference2, 1e-300))
            << ",\"scalar_samples\":" << sample_count;
        timing_samples("baseline_samples_ms", baseline_times);
        timing_samples("reassociated_samples_ms", candidate_times);
        std::cout << ",\"repeated_output_bit_exact\":true,\"full_model_parity\":\"not_run\"}\n" << std::flush;
    }
}

void benchmark_local(Device & device, int iterations, int threads) {
    check(!device.gpu, "the three-variant local GEMM-padding benchmark requires a CPU backend");
    for (const int side : { 64, 32, 16 }) {
        const int feature = 2560 / side;
        LocalFixture f(side, side, feature, feature, 8, 15, true);
        LocalGraph baseline(device, f, false), candidate(device, f, true), padded(device, f, true, true);
        std::array<LocalGraph *, 3> graphs{ &baseline, &candidate, &padded };
        std::array<std::vector<float>, 3> initial;
        std::array<size_t, 3> allocations{}, nodes{};
        std::array<uint64_t, 3> uploads{};
        for (size_t variant = 0; variant < graphs.size(); ++variant) {
            auto & graph = *graphs[variant];
            graph.builder.compute();
            initial[variant] = graph.builder.download(graph.output);
            allocations[variant] = graph.builder.allocation_bytes();
            nodes[variant] = graph.builder.nodes();
            uploads[variant] = graph.builder.upload_bytes();
        }
        const auto & original = initial[0];
        compare("model-shape local reassociation", initial[1], original, device.fast);
        compare("model-shape local padded reassociation", initial[2], original, device.fast);
        std::array<std::vector<float>, 3> samples;
        std::vector<float> samples_reference;
        std::vector<size_t> scalar_indices;
        constexpr size_t distributed_samples = 64;
        for (size_t sample = 0; sample < distributed_samples; ++sample)
            scalar_indices.push_back(sample * (original.size() - 1) / (distributed_samples - 1));
        const size_t strip_budget = size_t(32) << 20;
        auto rows_for = [&](bool pad_gemm) {
            const size_t area = size_t(f.side) * f.side, aligned_area = pad_gemm ? (area + 15) / 16 * 16 : area;
            const size_t patches = aligned_area * f.dv + (aligned_area != area ? area * f.dv : 0);
            const size_t row_bytes = (patches + aligned_area * f.heads) * f.w * f.batch * sizeof(float);
            check(row_bytes <= strip_budget, "benchmark row exceeds local workspace budget");
            return std::max(1, std::min(f.h, int(strip_budget / row_bytes)));
        };
        const int strip_rows = rows_for(false), padded_strip_rows = rows_for(true);
        // Explicitly sample both sides of every actual strip seam, with left,
        // center and right columns and both CFG samples. Distributed indices
        // alone need not happen to land on a row-strip boundary.
        for (const int rows : { strip_rows, padded_strip_rows })
            for (int start = rows; start < f.h; start += rows)
                for (int y : { start - 1, start })
                    for (int x : { 0, f.w / 2, f.w - 1 })
                        for (int sample = 0; sample < f.batch; ++sample) {
                            const int head = (sample + start) % f.heads, value = (x + start) % f.dv;
                            scalar_indices.push_back((size_t(sample) * f.heads * f.dv + head * f.dv + value) * f.w *
                                                         f.h +
                                                     size_t(y) * f.w + x);
                        }
        std::sort(scalar_indices.begin(), scalar_indices.end());
        scalar_indices.erase(std::unique(scalar_indices.begin(), scalar_indices.end()), scalar_indices.end());
        for (const size_t index : scalar_indices) {
            for (size_t variant = 0; variant < graphs.size(); ++variant)
                samples[variant].push_back(initial[variant][index]);
            samples_reference.push_back(f.at(index));
        }
        for (size_t variant = 0; variant < graphs.size(); ++variant)
            compare("model-shape local scalar samples variant " + std::to_string(variant), samples[variant],
                    samples_reference, device.fast);
        std::array<double, 3> maximum{}, difference2{};
        double reference2 = 0;
        for (size_t i = 0; i < original.size(); ++i) {
            for (size_t variant = 1; variant < graphs.size(); ++variant) {
                const double error = double(initial[variant][i]) - original[i];
                maximum[variant] = std::max(maximum[variant], std::abs(error));
                difference2[variant] += error * error;
            }
            reference2 += double(original[i]) * original[i];
        }
        std::array<std::vector<double>, 3> times;
        auto timed = [](Builder & builder, std::vector<double> & times) {
            const auto start = std::chrono::steady_clock::now();
            builder.compute();
            times.push_back(
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
        };
        for (int iteration = 0; iteration < iterations; ++iteration) {
            // Rotate all three variants through the first/second/third slot.
            for (size_t slot = 0; slot < graphs.size(); ++slot) {
                const size_t variant = (size_t(iteration) + slot) % graphs.size();
                timed(graphs[variant]->builder, times[variant]);
            }
        }
        for (size_t variant = 0; variant < graphs.size(); ++variant) {
            auto & graph = *graphs[variant];
            graph.unchanged_inputs(f);
            check(graph.builder.download(graph.output) == initial[variant],
                  "local positional benchmark repeated output changed");
            check(graph.builder.allocation_bytes() == allocations[variant] && graph.builder.nodes() == nodes[variant] &&
                      graph.builder.upload_bytes() == uploads[variant],
                  "local positional benchmark rebuilt graphs or uploaded unchanged inputs");
        }
        const double baseline_ms = median(times[0]), candidate_ms = median(times[1]), padded_ms = median(times[2]);
        std::cout << std::setprecision(9)
                  << "{\"schema\":\"crispembed-latent-graph-component-benchmark-v1\",\"component\":\"moebius-local\","
                  << "\"backend\":" << std::quoted(device.name)
                  << ",\"precision\":" << std::quoted(device.fast ? "mixed-f16-diagnostic" : "strict-f32")
                  << ",\"fixture_weight_type\":\"f32\",\"resident_weight_type\":"
                  << std::quoted(ggml_type_name(baseline.kernel->type)) << ",\"threads\":" << threads
                  << ",\"width\":" << side << ",\"height\":" << side << ",\"dk\":" << feature << ",\"dv\":" << feature
                  << ",\"heads\":8,\"kernel_side\":15,\"bias\":true,\"batch\":2,\"warmups\":1,\"iterations\":"
                  << iterations << ",\"baseline_median_ms\":" << baseline_ms
                  << ",\"reassociated_median_ms\":" << candidate_ms << ",\"speedup\":" << baseline_ms / candidate_ms
                  << ",\"padded_median_ms\":" << padded_ms << ",\"padded_speedup\":" << baseline_ms / padded_ms
                  << ",\"baseline_allocation_bytes\":" << allocations[0]
                  << ",\"reassociated_allocation_bytes\":" << allocations[1]
                  << ",\"padded_allocation_bytes\":" << allocations[2] << ",\"baseline_nodes\":" << nodes[0]
                  << ",\"reassociated_nodes\":" << nodes[1] << ",\"padded_nodes\":" << nodes[2]
                  << ",\"patch_and_coefficient_strip_budget_bytes\":" << strip_budget
                  << ",\"max_absolute_difference\":" << maximum[1]
                  << ",\"relative_l2_difference\":" << std::sqrt(difference2[1] / std::max(reference2, 1e-300))
                  << ",\"padded_max_absolute_difference\":" << maximum[2]
                  << ",\"padded_relative_l2_difference\":" << std::sqrt(difference2[2] / std::max(reference2, 1e-300))
                  << ",\"scalar_samples\":" << scalar_indices.size() << ",\"strip_rows\":" << strip_rows
                  << ",\"padded_strip_rows\":" << padded_strip_rows << ",\"padded_neighborhood_width\":240"
                  << ",\"padded_query_width\":" << (feature + 15) / 16 * 16;
        timing_samples("baseline_samples_ms", times[0]);
        timing_samples("reassociated_samples_ms", times[1]);
        timing_samples("padded_samples_ms", times[2]);
        std::cout << ",\"repeated_output_bit_exact\":true,\"full_model_parity\":\"not_run\"}\n" << std::flush;
    }
}

int run(int argc, char ** argv) {
    try {
        std::string requested = "cpu";
        int benchmark = 0, benchmark_local_iterations = 0, threads = 2;
        for (int i = 1; i < argc; ++i) {
            const std::string option = argv[i];
            if (option == "--backend" && i + 1 < argc)
                requested = argv[++i];
            else if ((option == "--benchmark" || option == "--benchmark-local" || option == "--threads") &&
                     i + 1 < argc) {
                const std::string value = argv[++i];
                check(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos,
                      "benchmark/thread count must be an integer");
                const int count = std::stoi(value);
                check(count >= 1 && count <= (option == "--threads" ? 8 : 10), "benchmark/thread count out of range");
                if (option == "--threads")
                    threads = count;
                else if (option == "--benchmark-local")
                    benchmark_local_iterations = count;
                else
                    benchmark = count;
            } else
                throw std::runtime_error("usage: test-latent-inpaint-graph-unet [--backend cpu|cuda|metal|mps] "
                                         "[--threads 1..8] [--benchmark 1..10] [--benchmark-local 1..10]");
        }
        check(requested == "cpu" || requested == "cuda" || requested == "metal" || requested == "mps",
              "unsupported requested backend");
        crispasr_set_gpu_backend_pref(requested.c_str());
        Device device(threads);
        check((requested == "cpu") != device.gpu, "requested backend and selected device class differ");
        kernels(device, 2);
        kernels(device, 8);
        local_kernels(device);
        invalid_inputs(device);
        if (benchmark) benchmark_relative(device, benchmark, threads);
        if (benchmark_local_iterations) benchmark_local(device, benchmark_local_iterations, threads);
        std::cout
            << "UNet graph lambda content/local/relative and per-head gated RMSNorm scalar references passed; backend="
            << device.name << "; requested_backend=" << requested
            << "; profile=" << (device.fast ? "mixed-f16-diagnostic" : "strict-f32")
            << "; component_atol=" << (device.fast ? "0.003" : "0.00003")
            << "; component_rtol=" << (device.fast ? "0.003" : "0.00003")
            << "; batch=2; replay_count=2; full_model_parity=not_run\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}

} // namespace

int main(int argc, char ** argv) {
    const int result = run(argc, argv);
    std::cout.flush();
    std::cerr.flush();
    core_util::clean_exit(result);
}
