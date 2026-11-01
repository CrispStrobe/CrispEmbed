// Tiny synthetic weights test actual ggml adapters against direct scalar
// formulae. No downloaded checkpoint, framework or quality claim is involved.
#include "latent_inpaint_ops.h"
#include "core/clean_exit.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using latent_inpaint::Engine;
using latent_inpaint::Tensor;
struct Value {
    std::vector<int64_t> shape;
    std::vector<float> data;
};
using Weights = std::map<std::string, Value>;

void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
void close(const Tensor & actual, const Tensor & expected) {
    require(actual.c == expected.c && actual.h == expected.h && actual.w == expected.w, "operator shape differs");
    for (size_t i = 0; i < actual.v.size(); ++i)
        if (!std::isfinite(actual.v[i]) || std::abs(actual.v[i] - expected.v[i]) > 2e-5f)
            throw std::runtime_error("operator scalar reference differs at " + std::to_string(i));
}

struct Fixture {
    std::filesystem::path directory, path;
    Weights weights;
    latent_inpaint::ShapeMap schema;
    Fixture() {
        const auto tick = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        directory = std::filesystem::temp_directory_path() / ("crispembed-ops-" + std::to_string(tick));
        require(std::filesystem::create_directory(directory), "cannot create new operator fixture directory");
        path = directory / "tiny.gguf";
    }
    ~Fixture() {
        std::error_code ignored;
        std::filesystem::remove_all(directory, ignored);
    }
    void add(const std::string & name, std::vector<int64_t> shape, float offset) {
        size_t count = 1;
        for (int64_t n : shape) count *= size_t(n);
        Value value{ shape, std::vector<float>(count) };
        for (size_t i = 0; i < count; ++i) value.data[i] = offset + float(int(i % 13) - 6) * 0.03137f;
        weights.emplace(name, std::move(value));
        schema.emplace(name, std::move(shape));
    }
    void save() {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 1024 * 1024, nullptr, false }), &ggml_free);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), &gguf_free);
        require(bool(ctx) && bool(file), "fixture context failed");
        for (const auto & row : weights) {
            int64_t ne[4] = { 1, 1, 1, 1 };
            for (size_t i = 0; i < row.second.shape.size(); ++i)
                ne[row.second.shape.size() - 1 - i] = row.second.shape[i];
            auto * tensor = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, int(row.second.shape.size()), ne);
            ggml_set_name(tensor, row.first.c_str());
            std::memcpy(tensor->data, row.second.data.data(), row.second.data.size() * sizeof(float));
            gguf_add_tensor(file.get(), tensor);
        }
        require(gguf_write_to_file(file.get(), path.string().c_str(), false), "cannot save tiny GGUF fixture");
    }
};

Tensor convolution(const Fixture & f, const std::string & name, const Tensor & x, int stride, int padding, int groups,
                   int dilation) {
    const auto & kernel = f.weights.at(name + ".weight");
    const int oc = int(kernel.shape[0]), ic = int(kernel.shape[1]), kh = int(kernel.shape[2]),
              kw = int(kernel.shape[3]);
    Tensor out(oc, (x.h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1,
               (x.w + 2 * padding - dilation * (kw - 1) - 1) / stride + 1);
    for (int o = 0; o < oc; ++o)
        for (int y = 0; y < out.h; ++y)
            for (int z = 0; z < out.w; ++z) {
                double sum = f.weights.at(name + ".bias").data[size_t(o)];
                for (int i = 0; i < ic; ++i)
                    for (int ky = 0; ky < kh; ++ky)
                        for (int kx = 0; kx < kw; ++kx) {
                            const int iy = y * stride + ky * dilation - padding,
                                      ix = z * stride + kx * dilation - padding;
                            if (iy >= 0 && iy < x.h && ix >= 0 && ix < x.w)
                                sum += double(x.at((o / (oc / groups)) * ic + i, iy, ix)) *
                                       kernel.data[((size_t(o) * ic + i) * kh + ky) * kw + kx];
                        }
                out.at(o, y, z) = float(sum);
            }
    return out;
}

void convolution_controls(Engine & engine, const Fixture & f, const Tensor & x) {
    close(engine.conv("dense", x, 2, 1), convolution(f, "dense", x, 2, 1, 1, 1));
    close(engine.conv("depthwise", x, 1, 2, 2, 2), convolution(f, "depthwise", x, 1, 2, 2, 2));
    close(engine.conv("grouped", x, 1, 0, 2), convolution(f, "grouped", x, 1, 0, 2, 1));
    const auto & kernel = f.weights.at("transpose.weight");
    Tensor expected(3, x.h * 2, x.w * 2);
    for (int o = 0; o < 3; ++o)
        for (size_t p = 0; p < expected.spatial(); ++p)
            expected.v[size_t(o) * expected.spatial() + p] = f.weights.at("transpose.bias").data[size_t(o)];
    // Independent scatter of each input sample, including the extra bottom
    // and right output cells selected by output_padding=1.
    for (int i = 0; i < x.c; ++i)
        for (int y = 0; y < x.h; ++y)
            for (int z = 0; z < x.w; ++z)
                for (int o = 0; o < 3; ++o)
                    for (int ky = 0; ky < 3; ++ky)
                        for (int kx = 0; kx < 3; ++kx) {
                            const int oy = 2 * y + ky - 1, ox = 2 * z + kx - 1;
                            if (oy >= 0 && oy < expected.h && ox >= 0 && ox < expected.w)
                                expected.at(o, oy, ox) +=
                                    x.at(i, y, z) * kernel.data[((size_t(i) * 3 + o) * 3 + ky) * 3 + kx];
                        }
    close(engine.conv_transpose("transpose", x), expected);
}

void linear_controls(Engine & engine, const Fixture & f, const Tensor & x) {
    Tensor expected(3, x.h, x.w);
    const auto & kernel = f.weights.at("linear.weight").data;
    for (int o = 0; o < 3; ++o)
        for (size_t p = 0; p < x.spatial(); ++p) {
            double sum = f.weights.at("linear.bias").data[size_t(o)];
            for (int i = 0; i < x.c; ++i) sum += double(x.v[size_t(i) * x.spatial() + p]) * kernel[size_t(o) * x.c + i];
            expected.v[size_t(o) * x.spatial() + p] = float(sum);
        }
    close(engine.linear("linear", x), expected);
    Tensor token(2, 1, 1), one(3, 1, 1);
    for (int i = 0; i < 2; ++i) token.v[size_t(i)] = x.v[size_t(i) * x.spatial()];
    for (int o = 0; o < 3; ++o) one.v[size_t(o)] = expected.v[size_t(o) * expected.spatial()];
    close(engine.linear("linear", token), one);
}

void attention_controls(Engine & engine, int queries, int keys) {
    Tensor q(16, 1, queries), k(16, 1, keys), v(32, 1, keys);
    for (size_t i = 0; i < q.v.size(); ++i) q.v[i] = float(int(i % 7) - 3) * 0.2f;
    for (size_t i = 0; i < k.v.size(); ++i) k.v[i] = float(int(i % 11) - 5) * 0.13f;
    for (size_t i = 0; i < v.v.size(); ++i) v.v[i] = float(int(i % 17) - 8) * 0.17f;
    Tensor expected(32, 1, queries);
    for (int head = 0; head < 2; ++head)
        for (int p = 0; p < queries; ++p) {
            std::vector<double> scores(size_t(keys), 0.0);
            double total = 0;
            for (int t = 0; t < keys; ++t) {
                double dot = 0;
                for (int d = 0; d < 8; ++d) dot += double(q.at(head * 8 + d, 0, p)) * k.at(head * 8 + d, 0, t);
                scores[t] = std::exp(dot / std::sqrt(8.0));
                total += scores[t];
            }
            for (int d = 0; d < 16; ++d) {
                double sum = 0;
                for (int t = 0; t < keys; ++t) sum += scores[t] / total * v.at(head * 16 + d, 0, t);
                expected.at(head * 16 + d, 0, p) = float(sum);
            }
        }
    close(engine.attention(q, k, v, 2), expected);
}

int run() {
    try {
        Fixture f;
        f.add("dense.weight", { 3, 2, 3, 2 }, 0.01f);
        f.add("dense.bias", { 3 }, 0.1f);
        f.add("depthwise.weight", { 2, 1, 3, 3 }, -0.02f);
        f.add("depthwise.bias", { 2 }, 0.05f);
        f.add("grouped.weight", { 4, 1, 2, 2 }, 0.03f);
        f.add("grouped.bias", { 4 }, -0.1f);
        f.add("transpose.weight", { 2, 3, 3, 3 }, 0.05f);
        f.add("transpose.bias", { 3 }, 0.02f);
        f.add("linear.weight", { 3, 2 }, 0.02f);
        f.add("linear.bias", { 3 }, 0.03f);
        f.save();
        Engine engine(f.path.string().c_str(), 1, f.schema);
        Tensor x(2, 4, 5);
        for (size_t i = 0; i < x.v.size(); ++i) x.v[i] = float(int(i % 23) - 11) * 0.137f;
        convolution_controls(engine, f, x);
        linear_controls(engine, f, x);
        attention_controls(engine, 3, 5);
        // Cross the ggml CPU query-tile boundary and include a tail in both
        // dimensions; the production VAE uses the tiled attention branch.
        attention_controls(engine, 65, 67);
        std::cout << "bounded native convolution, transpose, linear and attention scalar references passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
} // namespace

int main() {
    const int result = run();
    std::cout.flush();
    std::cerr.flush();
    core_util::clean_exit(result);
}
