// SPDX-License-Identifier: Apache-2.0
// TeleOCR FFN diagnostic adapted from the model equations.
// Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
// Source revision: b3f6a0c084db72f971e19a779d2fd8a6bc9e3b8d
// See NOTICE.teleocr and models/licenses/apache-2.0/LICENSE.
// This executable uses original post-attention inputs and selected block weights;
// it does not load a complete model or change the production arithmetic.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "core/clean_exit.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

static int run(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s PROBE.gguf OUTPUT_DIRECTORY\n", argv[0]);
        return 2;
    }
    const std::filesystem::path output_dir(argv[2]);
    std::error_code error;
    std::filesystem::create_directories(output_dir, error);
    if (error || !std::filesystem::is_directory(output_dir)) {
        std::fprintf(stderr, "Cannot create output directory\n");
        return 2;
    }

    ggml_context * weights = nullptr;
    gguf_context * file = gguf_init_from_file(argv[1], { false, &weights });
    ggml_context * graph_context = nullptr;
    ggml_backend_t backend = nullptr;
    auto finish = [&](int status) {
        if (backend) ggml_backend_free(backend);
        if (graph_context) ggml_free(graph_context);
        if (weights) ggml_free(weights);
        if (file) gguf_free(file);
        return status;
    };
    if (!file || !weights) {
        std::fprintf(stderr, "Cannot load the source-generated FFN probe\n");
        return finish(2);
    }
    auto tensor = [&](const char * name) { return ggml_get_tensor(weights, name); };
    for (const char * name : { "input", "norm_w", "gate_w", "gate_b", "up_w", "up_b", "down_w", "down_b", "norm",
                               "gate", "up", "silu", "product", "down", "output" }) {
        if (!tensor(name) || tensor(name)->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor(name))) {
            std::fprintf(stderr, "Missing or invalid float32 tensor: %s\n", name);
            return finish(2);
        }
    }
    const int64_t hidden = tensor("input")->ne[0], tokens = tensor("input")->ne[1];
    const int64_t intermediate = tensor("gate_w")->ne[1];
    auto shape = [&](const char * name, int64_t columns, int64_t rows = 1) {
        const auto * value = tensor(name);
        return value->ne[0] == columns && value->ne[1] == rows && value->ne[2] == 1 && value->ne[3] == 1;
    };
    const bool valid =
        (hidden == 1280 || hidden == 1024) && tokens > 0 && tokens <= 1024 &&
        ((hidden == 1280 && intermediate == 3420) || (hidden == 1024 && intermediate == 3072)) &&
        shape("input", hidden, tokens) && shape("norm_w", hidden) && shape("gate_w", hidden, intermediate) &&
        shape("gate_b", intermediate) && shape("up_w", hidden, intermediate) && shape("up_b", intermediate) &&
        shape("down_w", intermediate, hidden) && shape("down_b", hidden) && shape("norm", hidden, tokens) &&
        shape("gate", intermediate, tokens) && shape("up", intermediate, tokens) &&
        shape("silu", intermediate, tokens) && shape("product", intermediate, tokens) &&
        shape("down", hidden, tokens) && shape("output", hidden, tokens);
    if (!valid) {
        std::fprintf(stderr, "Invalid probe shapes; this bounded diagnostic supports at most 1024 input rows\n");
        return finish(2);
    }

    graph_context = ggml_init({ 256 * 1024 * 1024, nullptr, false });
    backend = ggml_backend_cpu_init();
    if (!graph_context || !backend) return finish(2);
    ggml_backend_cpu_set_n_threads(backend, 2);
    auto * g = graph_context;
    std::vector<std::pair<const char *, ggml_tensor *>> outputs;
    auto keep = [&](const char * name, ggml_tensor * value) {
        outputs.push_back({ name, value });
        return value;
    };
    auto norm = keep("norm", ggml_mul(g, ggml_rms_norm(g, tensor("input"), 1e-6f), tensor("norm_w")));
    auto multiply = [&](ggml_tensor * weight, ggml_tensor * input) {
        auto * result = ggml_mul_mat(g, weight, input);
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
        return result;
    };
    auto gate = keep("gate", ggml_add(g, multiply(tensor("gate_w"), norm), tensor("gate_b")));
    auto up = keep("up", ggml_add(g, multiply(tensor("up_w"), norm), tensor("up_b")));
    auto silu = keep("silu", ggml_silu(g, gate));
    auto product = keep("product", ggml_mul(g, silu, up));
    auto down = keep("down", ggml_add(g, multiply(tensor("down_w"), product), tensor("down_b")));
    auto output = keep("output", ggml_add(g, tensor("input"), down));
    auto graph = ggml_new_graph(g);
    ggml_build_forward_expand(graph, output);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) return finish(2);

    bool all_pass = true;
    for (const auto & item : outputs) {
        const auto * reference = tensor(item.first);
        const auto * actual = static_cast<const float *>(item.second->data);
        const auto * expected = static_cast<const float *>(reference->data);
        const int64_t count = ggml_nelements(reference), width = reference->ne[0];
        double maximum = 0.0, minimum_cosine = 1.0;
        int64_t at = 0, over_budget = 0;
        bool finite = true;
        for (int64_t row = 0; row < count; row += width) {
            double aa = 0.0, bb = 0.0, ab = 0.0;
            for (int64_t i = row; i < row + width; ++i) {
                const double a = actual[i], b = expected[i], difference = std::fabs(a - b);
                finite = finite && std::isfinite(a) && std::isfinite(b);
                if (difference > maximum) {
                    maximum = difference;
                    at = i;
                }
                over_budget += difference > 0.002;
                aa += a * a;
                bb += b * b;
                ab += a * b;
            }
            const double cosine = aa == 0.0 && bb == 0.0 ? 1.0 : aa == 0.0 || bb == 0.0 ? 0.0 : ab / std::sqrt(aa * bb);
            minimum_cosine = std::min(minimum_cosine, cosine);
        }
        const bool pass = finite && maximum <= 0.002 && minimum_cosine >= 0.99999;
        all_pass = all_pass && pass;
        std::printf("%s max_abs=%.9g row=%lld column=%lld native=%.9g ref=%.9g cos_min=%.12g over_0.002=%lld %s\n",
                    item.first, maximum, (long long)(at / width), (long long)(at % width), actual[at], expected[at],
                    minimum_cosine, (long long)over_budget, pass ? "PASS" : "FAIL");
        const auto path = output_dir / (std::string("native_") + item.first + ".f32");
        if (std::filesystem::exists(path)) {
            std::fprintf(stderr, "Refusing to overwrite an existing native stage: %s\n", path.string().c_str());
            return finish(2);
        }
        auto * stream = std::fopen(path.string().c_str(), "wb");
        if (!stream) return finish(2);
        const size_t written = std::fwrite(actual, sizeof(float), (size_t)count, stream);
        const int close_status = std::fclose(stream);
        if (written != (size_t)count || close_status != 0) return finish(2);
    }
    return finish(all_pass ? 0 : 1);
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
