// Hosted model-free checks of real persistent graph operators. All numerical
// expectations below are scalar definitions, not another ggml decomposition.
#include "inpaint_graph.h"
#include "core/clean_exit.h"
#include "core/gpu_backend_pref.h"
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
#include <unordered_set>
#include <vector>

namespace {
using inpaint_graph::Builder;
void check(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}
struct Values {
    std::vector<int64_t> shape;
    std::vector<float> data;
};
struct Fixture {
    std::filesystem::path directory, path;
    std::map<std::string, Values> values;
    inpaint_graph::ShapeMap schema;
    Fixture() {
        directory = std::filesystem::temp_directory_path() /
                    ("inpaint-graph-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        check(std::filesystem::create_directory(directory), "cannot create owned fixture");
        path = directory / "tiny.gguf";
    }
    ~Fixture() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
    void add(const std::string & name, std::vector<int64_t> shape, float bias) {
        size_t n = 1;
        for (auto x : shape) n *= size_t(x);
        Values out{ shape, std::vector<float>(n) };
        for (size_t i = 0; i < n; ++i) out.data[i] = bias + float(int(i % 11) - 5) * .017f;
        values.emplace(name, std::move(out));
        schema.emplace(name, std::move(shape));
    }
    void save() {
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({ 1 << 20, nullptr, false }), ggml_free);
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
        check(bool(ctx) && bool(file), "cannot allocate tiny fixture metadata");
        for (const auto & entry : values) {
            int64_t ne[4] = { 1, 1, 1, 1 };
            for (size_t i = 0; i < entry.second.shape.size(); ++i)
                ne[entry.second.shape.size() - 1 - i] = entry.second.shape[i];
            ggml_tensor * t = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, ne);
            ggml_set_name(t, entry.first.c_str());
            std::memcpy(t->data, entry.second.data.data(), entry.second.data.size() * sizeof(float));
            gguf_add_tensor(file.get(), t);
        }
        check(gguf_write_to_file(file.get(), path.string().c_str(), false), "cannot write tiny model");
    }
};
size_t ix(int b, int c, int y, int x, int channels, int h, int w) {
    return ((size_t(b) * channels + c) * h + y) * w + x;
}
double worst = 0;
void close_values(const std::vector<float> & actual, const std::vector<float> & expected, double absolute = 4e-5,
                  double relative = 0, const std::string & label = "operator") {
    check(actual.size() == expected.size(), "output size differs");
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(double(actual[i]) - expected[i]);
        worst = std::max(worst, error);
        const double budget = absolute + relative * std::abs(double(expected[i]));
        check(std::isfinite(actual[i]) && std::isfinite(expected[i]) && error <= budget,
              label + " scalar oracle differs at " + std::to_string(i) + " error=" + std::to_string(error) +
                  " budget=" + std::to_string(budget));
    }
}
std::vector<float> conv_ref(const Fixture & f, const std::string & name, const std::vector<float> & input, int w, int h,
                            int c, int batch, int stride, int pad, bool dw) {
    const auto & kernel = f.values.at(name + ".weight");
    const int oc = int(kernel.shape[0]), ic = int(kernel.shape[1]), kh = int(kernel.shape[2]),
              kw = int(kernel.shape[3]);
    const int ow = (w + 2 * pad - kw) / stride + 1, oh = (h + 2 * pad - kh) / stride + 1;
    std::vector<float> out(size_t(batch) * oc * oh * ow);
    for (int b = 0; b < batch; ++b)
        for (int o = 0; o < oc; ++o)
            for (int y = 0; y < oh; ++y)
                for (int x = 0; x < ow; ++x) {
                    double sum = f.values.at(name + ".bias").data[size_t(o)];
                    for (int i = 0; i < ic; ++i)
                        for (int ky = 0; ky < kh; ++ky)
                            for (int kx = 0; kx < kw; ++kx) {
                                const int sy = y * stride + ky - pad, sx = x * stride + kx - pad;
                                if (sy >= 0 && sy < h && sx >= 0 && sx < w)
                                    sum += double(input[ix(b, dw ? o : i, sy, sx, c, h, w)]) *
                                           kernel.data[((size_t(o) * ic + i) * kh + ky) * kw + kx];
                            }
                    out[ix(b, o, y, x, oc, oh, ow)] = float(sum);
                }
    return out;
}
void operators(inpaint_graph::Device & device) {
    Fixture f;
    for (const auto & n : { "group", "layer", "bn" }) {
        f.add(std::string(n) + ".weight", { 4 }, .9f);
        f.add(std::string(n) + ".bias", { 4 }, .2f);
    }
    f.add("bn.running_mean", { 4 }, .3f);
    f.add("bn.running_var", { 4 }, .8f);
    f.add("dense.weight", { 3, 4, 3, 3 }, .01f);
    f.add("dense.bias", { 3 }, .04f);
    f.add("point.weight", { 6, 4, 1, 1 }, .08f);
    f.add("point.bias", { 6 }, .01f);
    f.add("dw.weight", { 4, 1, 3, 3 }, .03f);
    f.add("dw.bias", { 4 }, .02f);
    f.save();
    inpaint_graph::Weights weights(device, f.path.string().c_str(), f.schema);
    Builder b(device, &weights, 4096);
    auto * input = b.parameter("input", { 7, 5, 4, 2 });
    auto * dense = b.conv("dense", input, 2, 1);
    auto * point = b.conv("point", input);
    auto * dw = b.conv("dw", input, 1, 1, 4);
    auto * gn = b.group_norm("group", input, 2);
    auto * ln = b.layer_norm("layer", input);
    auto * bn = b.batch_norm("bn", input);
    auto * padded = b.pad(input, 1, 2, 2, 1);
    auto * reflected = b.reflect_pad(input, 1);
    b.finish({ dense, point, dw, gn, ln, bn, padded, reflected }, "operator-oracles");
    const size_t nodes = b.nodes(), bytes = b.allocation_bytes();
    for (int run = 0; run < 2; ++run) {
        std::vector<float> x(7 * 5 * 4 * 2);
        for (size_t i = 0; i < x.size(); ++i) x[i] = float(int((i * 7 + run * 3) % 37) - 18) * .031f;
        b.upload(input, x);
        b.compute();
        close_values(b.download(dense), conv_ref(f, "dense", x, 7, 5, 4, 2, 2, 1, false), device.fast ? 2e-3 : 4e-5);
        close_values(b.download(point), conv_ref(f, "point", x, 7, 5, 4, 2, 1, 0, false), device.fast ? 2e-3 : 4e-5);
        close_values(b.download(dw), conv_ref(f, "dw", x, 7, 5, 4, 2, 1, 1, true));
        std::vector<float> expected(x.size());
        for (int sample = 0; sample < 2; ++sample)
            for (int group = 0; group < 2; ++group) {
                double mean = 0, var = 0;
                for (int c = group * 2; c < group * 2 + 2; ++c)
                    for (int p = 0; p < 35; ++p) mean += x[(sample * 4 + c) * 35 + p];
                mean /= 70;
                for (int c = group * 2; c < group * 2 + 2; ++c)
                    for (int p = 0; p < 35; ++p) {
                        const double d = x[(sample * 4 + c) * 35 + p] - mean;
                        var += d * d;
                    }
                var /= 70;
                for (int c = group * 2; c < group * 2 + 2; ++c)
                    for (int p = 0; p < 35; ++p) {
                        const size_t i = size_t(sample * 4 + c) * 35 + p;
                        expected[i] =
                            float((x[i] - mean) / std::sqrt(var + 1e-5) * f.values.at("group.weight").data[c] +
                                  f.values.at("group.bias").data[c]);
                    }
            }
        close_values(b.download(gn), expected);
        for (int sample = 0; sample < 2; ++sample)
            for (int p = 0; p < 35; ++p) {
                double mean = 0, var = 0;
                for (int c = 0; c < 4; ++c) mean += x[(sample * 4 + c) * 35 + p];
                mean /= 4;
                for (int c = 0; c < 4; ++c) {
                    double d = x[(sample * 4 + c) * 35 + p] - mean;
                    var += d * d;
                }
                var /= 4;
                for (int c = 0; c < 4; ++c) {
                    size_t i = size_t(sample * 4 + c) * 35 + p;
                    expected[i] = float((x[i] - mean) / std::sqrt(var + 1e-5) * f.values.at("layer.weight").data[c] +
                                        f.values.at("layer.bias").data[c]);
                }
            }
        close_values(b.download(ln), expected);
        for (int sample = 0; sample < 2; ++sample)
            for (int c = 0; c < 4; ++c)
                for (int p = 0; p < 35; ++p) {
                    size_t i = size_t(sample * 4 + c) * 35 + p;
                    expected[i] = (x[i] - f.values.at("bn.running_mean").data[c]) /
                                      std::sqrt(f.values.at("bn.running_var").data[c] + 1e-5f) *
                                      f.values.at("bn.weight").data[c] +
                                  f.values.at("bn.bias").data[c];
                }
        close_values(b.download(bn), expected);
        std::vector<float> zeros(10 * 8 * 4 * 2, 0), reflection(9 * 7 * 4 * 2);
        for (int sample = 0; sample < 2; ++sample)
            for (int c = 0; c < 4; ++c) {
                for (int y = 0; y < 5; ++y)
                    for (int xx = 0; xx < 7; ++xx)
                        zeros[ix(sample, c, y + 2, xx + 1, 4, 8, 10)] = x[ix(sample, c, y, xx, 4, 5, 7)];
                for (int y = 0; y < 7; ++y)
                    for (int xx = 0; xx < 9; ++xx) {
                        int sy = y - 1, sx = xx - 1;
                        if (sy < 0) sy = -sy;
                        if (sy >= 5) sy = 8 - sy;
                        if (sx < 0) sx = -sx;
                        if (sx >= 7) sx = 12 - sx;
                        reflection[ix(sample, c, y, xx, 4, 7, 9)] = x[ix(sample, c, sy, sx, 4, 5, 7)];
                    }
            }
        close_values(b.download(padded), zeros, 0);
        close_values(b.download(reflected), reflection, 0);
        check(b.nodes() == nodes && b.allocation_bytes() == bytes, "persistent graph changed allocation on replay");
    }
    check(b.runs() == 2 && b.upload_bytes() == 2 * 7 * 5 * 4 * 2 * sizeof(float),
          "execution/transfer accounting differs");
}

std::vector<float> transpose_ref(const Fixture & f, const std::vector<float> & input, int width, int height,
                                 int padding, int output_padding, int stride, int batch) {
    const auto & kernel = f.values.at("transpose.weight");
    const int ic = int(kernel.shape[0]), oc = int(kernel.shape[1]);
    const int kh = int(kernel.shape[2]), kw = int(kernel.shape[3]);
    const int ow = (width - 1) * stride - 2 * padding + kw + output_padding;
    const int oh = (height - 1) * stride - 2 * padding + kh + output_padding;
    std::vector<double> sums(size_t(ow) * oh * oc * batch);
    for (int sample = 0; sample < batch; ++sample)
        for (int out = 0; out < oc; ++out)
            for (int y = 0; y < oh; ++y)
                for (int x = 0; x < ow; ++x)
                    sums[ix(sample, out, y, x, oc, oh, ow)] = f.values.at("transpose.bias").data[out];
    // Scatter the original PyTorch [input,output,ky,kx] weights. This does not
    // derive the expected result from ggml's uncropped intermediate tensor.
    for (int sample = 0; sample < batch; ++sample)
        for (int in = 0; in < ic; ++in)
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    for (int out = 0; out < oc; ++out)
                        for (int ky = 0; ky < kh; ++ky)
                            for (int kx = 0; kx < kw; ++kx) {
                                const int oy = stride * y + ky - padding, ox = stride * x + kx - padding;
                                if (oy >= 0 && oy < oh && ox >= 0 && ox < ow)
                                    sums[ix(sample, out, oy, ox, oc, oh, ow)] +=
                                        double(input[ix(sample, in, y, x, ic, height, width)]) *
                                        kernel.data[((size_t(in) * oc + out) * kh + ky) * kw + kx];
                            }
    return std::vector<float>(sums.begin(), sums.end());
}

void transpose_convolution(inpaint_graph::Device & device, int batch) {
    Fixture f;
    f.add("transpose.weight", { 2, 3, 3, 3 }, .013f);
    f.add("transpose.bias", { 3 }, -.021f);
    f.save();
    inpaint_graph::Weights weights(device, f.path.string().c_str(), f.schema);
    Builder b(device, &weights, 1024);
    auto * input = b.parameter("transpose.input", { 4, 3, 2, batch });
    struct Case {
        int stride, padding, output_padding;
    };
    const std::vector<Case> cases{ { 2, 1, 0 }, { 2, 1, 1 }, { 2, 0, 0 }, { 1, 1, 0 },
                                   { 3, 1, 2 }, { 4, 0, 1 }, { 4, 2, 0 } };
    std::vector<ggml_tensor *> outputs;
    for (const auto & profile : cases) {
        auto * output = b.conv_transpose("transpose", input, profile.stride, profile.padding, profile.output_padding);
        check(output->ne[0] == 3 * profile.stride - 2 * profile.padding + 3 + profile.output_padding &&
                  output->ne[1] == 2 * profile.stride - 2 * profile.padding + 3 + profile.output_padding &&
                  output->ne[2] == 3 && output->ne[3] == batch,
              "transposed convolution crop/output-padding geometry differs");
        outputs.push_back(output);
    }
    // Verify this is the optimized graph, including its once-only prepared
    // kernel. A call to the old scalar direct backend would pass the numerical
    // oracle but would not satisfy the graph implementation requirement.
    std::unordered_set<ggml_tensor *> visited, prepared_weights;
    std::vector<ggml_tensor *> pending = outputs;
    size_t matrix_nodes = 0;
    while (!pending.empty()) {
        auto * tensor = pending.back();
        pending.pop_back();
        if (!tensor || !visited.insert(tensor).second) continue;
        check(tensor->op != GGML_OP_CONV_TRANSPOSE_2D, "transpose graph still uses the direct scalar operation");
        if (tensor->op == GGML_OP_MUL_MAT) ++matrix_nodes;
        if (tensor->op == GGML_OP_IM2COL) prepared_weights.insert(tensor->src[0]);
        for (auto * source : tensor->src) pending.push_back(source);
    }
    check(matrix_nodes >= cases.size() && prepared_weights.size() == 1,
          "transpose graph did not share one transformed kernel across GEMM evaluations");
    auto * prepared = *prepared_weights.begin();
    check(prepared->ne[0] == 3 && prepared->ne[1] == 3 && prepared->ne[2] == 2 && prepared->ne[3] == 3 &&
              prepared->type == (device.fast ? GGML_TYPE_F16 : GGML_TYPE_F32),
          "transpose prepared kernel layout or precision differs");
    auto reject = [&](int stride, int padding, int output_padding) {
        bool rejected = false;
        try {
            (void)b.conv_transpose("transpose", input, stride, padding, output_padding);
        } catch (const std::exception &) {
            rejected = true;
        }
        check(rejected, "unsupported transpose convolution parameters were accepted");
    };
    reject(0, 1, 0);
    reject(5, 1, 0);
    reject(2, 3, 0);
    reject(2, 1, -1);
    reject(2, 1, 2);
    b.finish(outputs, "transpose-convolution-oracles");
    const size_t nodes = b.nodes(), bytes = b.allocation_bytes();
    for (int run = 0; run < 2; ++run) {
        std::vector<float> x(size_t(4 * 3 * 2 * batch));
        for (size_t i = 0; i < x.size(); ++i) x[i] = float(int((i * 5 + run * 3) % 19) - 9) * .093f;
        b.upload(input, x);
        b.compute();
        const double budget = device.fast ? 2e-3 : 4e-5;
        for (size_t i = 0; i < cases.size(); ++i) {
            const auto & profile = cases[i];
            close_values(
                b.download(outputs[i]),
                transpose_ref(f, x, 4, 3, profile.padding, profile.output_padding, profile.stride, batch), budget, 0,
                "transpose stride" + std::to_string(profile.stride) + " padding" + std::to_string(profile.padding) +
                    " output_padding" + std::to_string(profile.output_padding));
        }
        check(b.nodes() == nodes && b.allocation_bytes() == bytes, "transpose graph changed on replay");
    }
    check(b.runs() == 2 && b.upload_bytes() == size_t(2 * 4 * 3 * 2 * batch) * sizeof(float),
          "transpose replay transfers differ");
    std::cout << "transpose_convolution batch=" << batch << " cases=" << cases.size()
              << " matrix_nodes=" << matrix_nodes << " prepared_weights=" << prepared_weights.size()
              << " prepared_type=" << ggml_type_name(prepared->type) << " runs=" << b.runs() << '\n';
}

std::vector<float> attention_ref(const std::vector<float> & q, const std::vector<float> & k,
                                 const std::vector<float> & v, int queries, int keys, int dk, int dv, int heads,
                                 int batch) {
    std::vector<float> result(size_t(batch) * heads * dv * queries);
    std::vector<double> scores(size_t(keys), 0);
    for (int sample = 0; sample < batch; ++sample)
        for (int head = 0; head < heads; ++head)
            for (int query = 0; query < queries; ++query) {
                double maximum = -1e100;
                for (int key = 0; key < keys; ++key) {
                    double dot = 0;
                    for (int d = 0; d < dk; ++d)
                        dot += double(q[(size_t(sample * heads + head) * dk + d) * queries + query]) *
                               k[(size_t(sample * heads + head) * dk + d) * keys + key];
                    scores[size_t(key)] = dot / std::sqrt(double(dk));
                    maximum = std::max(maximum, scores[size_t(key)]);
                }
                double total = 0;
                for (double & value : scores) {
                    value = std::exp(value - maximum);
                    total += value;
                }
                for (int d = 0; d < dv; ++d) {
                    double sum = 0;
                    for (int key = 0; key < keys; ++key)
                        sum += scores[size_t(key)] / total * v[(size_t(sample * heads + head) * dv + d) * keys + key];
                    result[(size_t(sample * heads + head) * dv + d) * queries + query] = float(sum);
                }
            }
    return result;
}

void attention(inpaint_graph::Device & device, int queries, int keys) {
    // DK8/DV16 exercises rectangular F32 CPU flash and strict GPU GEMMs. The
    // explicit mixed profile uses the real model's DK40/DV40, supported by
    // both pinned GPU flash implementations. Neither case silently falls back.
    const int dk = device.fast ? 40 : 8, dv = device.fast ? 40 : 16;
    constexpr int heads = 2, batch = 2;
    const int height = queries == 129 ? 3 : 1, width = queries / height;
    std::vector<float> q(size_t(batch) * heads * dk * queries), k(size_t(batch) * heads * dk * keys);
    std::vector<float> v(size_t(batch) * heads * dv * keys);
    for (size_t i = 0; i < q.size(); ++i) q[i] = float(int((i * 7 + 3) % 23) - 11) * .13f;
    for (size_t i = 0; i < k.size(); ++i) k[i] = float(int((i * 5 + 1) % 29) - 14) * .07f;
    for (size_t i = 0; i < v.size(); ++i) v[i] = float(int((i * 11 + 2) % 31) - 15) * .09f;
    // Make the final key significant to the final query, including the tile
    // tail. This catches dropping a tail even when most random scores are flat.
    for (int sample = 0; sample < batch; ++sample)
        for (int head = 0; head < heads; ++head) {
            for (int d = 0; d < dk; ++d) {
                q[(size_t(sample * heads + head) * dk + d) * queries + queries - 1] = 2.0f;
                k[(size_t(sample * heads + head) * dk + d) * keys + keys - 1] = 1.5f;
            }
            for (int d = 0; d < dv; ++d)
                v[(size_t(sample * heads + head) * dv + d) * keys + keys - 1] += float(sample + head + 1);
        }
    Builder b(device, nullptr, 4096);
    auto * query = b.parameter("attention.query", { width, height, heads * dk, batch });
    auto * key = b.parameter("attention.key", { keys, 1, heads * dk, batch });
    auto * value = b.parameter("attention.value", { keys, 1, heads * dv, batch });
    auto * output = b.attention(query, key, value, heads);
    check(output->ne[0] == width && output->ne[1] == height && output->ne[2] == heads * dv && output->ne[3] == batch &&
              ggml_is_contiguous(output),
          "attention WHCN output differs");
    b.finish({ output }, "attention-oracle-" + std::to_string(queries) + "-" + std::to_string(keys));
    const size_t nodes = b.nodes(), bytes = b.allocation_bytes();
    std::vector<float> previous;
    for (int run = 0; run < 2; ++run) {
        if (run == 1) {
            // Change only sample 1; a batch-stride error then changes sample 0.
            for (size_t i = q.size() / batch; i < q.size(); ++i) q[i] *= .81f;
            for (size_t i = v.size() / batch; i < v.size(); ++i) v[i] += .37f;
        }
        b.upload(query, q);
        b.upload(key, k);
        b.upload(value, v);
        b.compute();
        auto actual = b.download(output);
        // Budgets are fixed before any hosted execution. Mixed diagnostics do
        // not relax the separate end-to-end original-model parity contract.
        const double budget = device.fast ? 2e-3 : 4e-5;
        close_values(actual, attention_ref(q, k, v, queries, keys, dk, dv, heads, batch), budget, budget,
                     "attention " + std::to_string(queries) + "/" + std::to_string(keys));
        if (run == 1) {
            const size_t one = actual.size() / batch;
            close_values(std::vector<float>(actual.begin(), actual.begin() + one),
                         std::vector<float>(previous.begin(), previous.begin() + one), budget, 0,
                         "attention independent sample 0");
        }
        previous = std::move(actual);
        check(b.nodes() == nodes && b.allocation_bytes() == bytes, "attention graph changed on replay");
    }
    check(b.runs() == 2 && b.upload_bytes() == 2 * (q.size() + k.size() + v.size()) * sizeof(float),
          "attention replay transfers differ");
    std::cout << "attention queries=" << queries << " keys=" << keys << " dk=" << dk << " dv=" << dv
              << " heads=" << heads << " batch=" << batch << " nodes=" << nodes << " allocation_bytes=" << bytes
              << " runs=" << b.runs() << '\n';
}

void tiled_convolution(inpaint_graph::Device & device) {
    // Force a real row-strip seam under the production allocation limits:
    // CPU uses 32 MiB, GPUs 128 MiB. Mixed F16 patches need twice the height
    // to cross the latter limit. Largest input is ~32 MiB; output has only two
    // channels. All output pixels, including every seam and border, are checked.
    const int width = device.gpu ? 256 : 64, height = device.fast ? 129 : 65;
    constexpr int channels = 128, batch = 2, output_channels = 2;
    Fixture f;
    f.add("tiled.weight", { output_channels, channels, 3, 3 }, 0);
    f.add("tiled.bias", { output_channels }, -.07f);
    auto & kernel = f.values.at("tiled.weight").data;
    for (size_t i = 0; i < kernel.size(); ++i) kernel[i] = float(int((i * 7 + 3) % 23) - 11) * .0013f;
    f.save();
    inpaint_graph::Weights weights(device, f.path.string().c_str(), f.schema);
    Builder b(device, &weights, 4096);
    auto * input = b.parameter("tiled.input", { width, height, channels, batch });
    auto * output = b.conv("tiled", input, 1, 1);
    size_t im2col_nodes = 0, maximum_patch_bytes = 0;
    std::unordered_set<ggml_tensor *> visited;
    std::vector<ggml_tensor *> pending{ output };
    while (!pending.empty()) {
        auto * tensor = pending.back();
        pending.pop_back();
        if (!tensor || !visited.insert(tensor).second) continue;
        if (tensor->op == GGML_OP_IM2COL) {
            ++im2col_nodes;
            maximum_patch_bytes = std::max(maximum_patch_bytes, ggml_nbytes(tensor));
        }
        for (auto * source : tensor->src) pending.push_back(source);
    }
    check(im2col_nodes >= 2, "tiled-convolution fixture did not exercise a strip seam");
    check(maximum_patch_bytes <= (device.gpu ? size_t(128) : size_t(32)) * 1024 * 1024,
          "convolution strip exceeds the production patch-memory limit");
    check(output->ne[0] == width && output->ne[1] == height && output->ne[2] == output_channels &&
              output->ne[3] == batch,
          "tiled convolution geometry differs");
    b.finish({ output }, "convolution-strip-seam-oracle");
    std::vector<float> x(size_t(width) * height * channels * batch);
    for (int sample = 0; sample < batch; ++sample)
        for (int channel = 0; channel < channels; ++channel)
            for (int y = 0; y < height; ++y)
                for (int xx = 0; xx < width; ++xx)
                    x[ix(sample, channel, y, xx, channels, height, width)] =
                        float((xx * 3 + y * 5 + channel * 7 + sample * 11) % 37 - 18) * .023f + float(y) * .001f +
                        float(sample) * .015f + (y > 33 ? .5f : 0.0f);
    b.upload(input, x);
    b.compute();
    const double budget = device.fast ? 2e-3 : 4e-5;
    close_values(b.download(output), conv_ref(f, "tiled", x, width, height, channels, batch, 1, 1, false), budget,
                 budget, "tiled convolution including seams");
    check(b.runs() == 1 && b.upload_bytes() == x.size() * sizeof(float), "tiled convolution transfers differ");
    std::cout << "tiled_convolution width=" << width << " height=" << height << " input_channels=" << channels
              << " output_channels=" << output_channels << " batch=" << batch << " im2col_nodes=" << im2col_nodes
              << " maximum_patch_bytes=" << maximum_patch_bytes << " allocation_bytes=" << b.allocation_bytes() << '\n';
}

void recurrent_state(inpaint_graph::Device & device) {
    inpaint_graph::State resident(device);
    auto * value = resident.tensor("resident", { 16 });
    resident.allocate();
    resident.upload(value, std::vector<float>(16, 1));
    Builder b(device, nullptr, 256);
    auto * delta = b.parameter("delta", { 1 });
    auto * result = ggml_add(b.ctx, value, delta);
    b.finish({ ggml_cpy(b.ctx, result, value) }, "resident-state-replay");
    for (int step = 0; step < 4; ++step) {
        b.upload(delta, { float(step + 1) });
        b.compute();
    }
    close_values(b.download(value), std::vector<float>(16, 11), 0);
    check(b.runs() == 4 && b.upload_bytes() == 4 * sizeof(float) && b.download_bytes() == 16 * sizeof(float),
          "resident loop transferred activations");
}
int run(int argc, char ** argv) {
    std::string backend = "cpu";
    if (argc == 3 && std::string(argv[1]) == "--backend")
        backend = argv[2];
    else
        check(argc == 1, "usage: test-inpaint-graph [--backend cpu|cuda|metal]");
    check(backend == "cpu" || backend == "cuda" || backend == "metal" || backend == "mps", "invalid backend");
    crispasr_set_gpu_backend_pref(backend.c_str());
    inpaint_graph::Device device(2);
    check((backend == "cpu") != device.gpu, "requested and selected backend classes differ");
    operators(device);
    transpose_convolution(device, 1);
    transpose_convolution(device, 2);
    attention(device, 3, 5);
    attention(device, 129, 131);
    attention(device, 257, 131); // odd three-tile balanced concatenation on strict GPUs
    tiled_convolution(device);
    recurrent_state(device);
    std::cout << "PASS backend=" << device.name << " requested_backend=" << backend
              << " precision=" << (device.fast ? "mixed-f16-diagnostic" : "f32")
              << " profile_atol=" << (device.fast ? "0.002" : "0.00004")
              << " profile_rtol=" << (device.fast ? "0.002" : "0.00004")
              << " normalization_atol=0.00004 worst_abs=" << worst << " full_model_parity=not_run\n";
    return 0;
}
} // namespace
int main(int argc, char ** argv) {
    int status = 0;
    try {
        status = run(argc, argv);
    } catch (const std::exception & e) {
        std::cerr << e.what() << "\n";
        status = 1;
    }
    std::cout.flush();
    std::cerr.flush();
    core_util::clean_exit(status);
}
