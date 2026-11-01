// Hosted, model-free checks of the actual UNet device-graph kernels. Expected
// values use direct scalar definitions, independent of graph transpositions.
#include "latent_inpaint_graph_unet.h"
#include "core/clean_exit.h"
#include "core/gpu_backend_pref.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
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

    explicit Fixture(int head_count) : heads(head_count) {
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
    auto * norm = gated_rms_norm(b, content, gate, scale, f.heads);
    for (auto * output : { content, local, learned, norm }) {
        check(output->ne[0] == f.w && output->ne[1] == f.h && output->ne[2] == f.heads * f.dv &&
                  output->ne[3] == f.batch && ggml_is_contiguous(output),
              "graph kernel WHCN output layout differs");
    }
    b.finish({ content, local, learned, norm }, "latent-unet-analytical");
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
        compare("learned relative lambda", b.download(learned), f.learned(), device.fast);
        compare("per-head gated RMSNorm", b.download(norm), f.normalized(expected_content), device.fast);
        check(b.runs() == uint64_t(run + 1) && b.nodes() == nodes && b.allocation_bytes() == allocation,
              "graph execution rebuilt nodes or changed its persistent allocation");
        check(b.upload_bytes() == initial_upload + uint64_t(run + 1) * f.q.size() * sizeof(float),
              "a graph replay uploaded more than its changed query input");
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
    rejects([&] { lambda_content(b, q, k, v, 0); });
    rejects([&] { lambda_content(b, q, k, v, 4); });
    rejects([&] { lambda_content(b, q, k, wrong_batch, 2); });
    rejects([&] { lambda_relative(b, q, v, wrong_relative, 2); });
    rejects([&] { lambda_position(b, q, wrong_position, 2, 4); });
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

int run(int argc, char ** argv) {
    try {
        std::string requested = "cpu";
        if (argc == 3 && std::string(argv[1]) == "--backend")
            requested = argv[2];
        else if (argc != 1)
            throw std::runtime_error("usage: test-latent-inpaint-graph-unet [--backend cpu|cuda|metal|mps]");
        check(requested == "cpu" || requested == "cuda" || requested == "metal" || requested == "mps",
              "unsupported requested backend");
        crispasr_set_gpu_backend_pref(requested.c_str());
        Device device(2);
        check((requested == "cpu") != device.gpu, "requested backend and selected device class differ");
        kernels(device, 2);
        kernels(device, 8);
        invalid_inputs(device);
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
