// Included only inside migan::test in the observer-enabled translation unit.
// These tiny cases exercise the actual production builder and CPU callbacks.
#pragma once

static void cpu_kernel_regression(int n_threads) {
    const int cases[][6] = { { 131, 9, 3, 3, 1, 1 }, { 137, 11, 5, 4, 2, 1 }, { 17, 13, 2, 4, 1, 0 },
                             { 3, 2, 1, 3, 1, 1 },   { 2, 2, 3, 4, 2, 1 },    { 263, 7, 2, 3, 2, 1 } };
    double max_raw_error = 0, max_activation_error = 0;
    for (const auto & dimensions : cases) {
        const int width = dimensions[0], height = dimensions[1], channels = dimensions[2];
        const int kernel_size = dimensions[3], stride = dimensions[4], padding = dimensions[5];
        const int out_width = (width + 2 * padding - kernel_size) / stride + 1;
        const int out_height = (height + 2 * padding - kernel_size) / stride + 1;
        context model;
        std::string error;
        model.persistent = model.cpu_kernels = true;
        model.stats.persistent = model.stats.cpu_kernels = true;
        model.backend = inpaint_graph::select_backend(n_threads, model.stats.backend, error);
        if (!model.backend || !core_cpu_backend::is_cpu(model.backend))
            throw std::runtime_error("MI-GAN CPU kernel control selected a non-CPU backend: " + error);
        model.threadpool = std::make_unique<inpaint_graph::CpuPool>(model.backend, n_threads);
        model.weights.ctx = ggml_init({ 2 * ggml_tensor_overhead(), nullptr, true });
        if (!model.weights.ctx) throw std::runtime_error("MI-GAN CPU kernel control metadata allocation failed");
        ggml_tensor * kernel =
            ggml_new_tensor_4d(model.weights.ctx, GGML_TYPE_F32, kernel_size, kernel_size, 1, channels);
        ggml_tensor * bias = ggml_new_tensor_1d(model.weights.ctx, GGML_TYPE_F32, channels);
        model.weights.tensors.emplace("dw.weight", kernel);
        model.weights.tensors.emplace("dw.bias", bias);
        model.weights.buf = ggml_backend_alloc_ctx_tensors(model.weights.ctx, model.backend);
        if (!model.weights.buf) throw std::runtime_error("MI-GAN CPU kernel control weight allocation failed");
        std::vector<float> weights(size_t(channels) * kernel_size * kernel_size), biases(channels);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = float(int((i * 7 + 3) % 17) - 8) * 0.019f;
        for (int c = 0; c < channels; ++c) biases[c] = float(c - 2) * 0.037f;
        ggml_backend_tensor_set(kernel, weights.data(), 0, weights.size() * sizeof(float));
        ggml_backend_tensor_set(bias, biases.data(), 0, biases.size() * sizeof(float));
        graph_builder graph(model);
        graph.retain = true;
        graph.input = ggml_new_tensor_4d(graph.ctx, GGML_TYPE_F32, width, height, channels, 1);
        ggml_set_input(graph.input);
        ggml_tensor * raw = graph.convolution("dw", graph.input, stride, padding, true, true);
        graph.finish(graph.activate(raw));
        std::vector<float> input(size_t(width) * height * channels);
        std::vector<float> output(size_t(out_width) * out_height * channels), raw_output(output.size());
        for (int repeat = 0; repeat < 2; ++repeat) {
            for (size_t i = 0; i < input.size(); ++i) input[i] = float(int((i * 5 + repeat) % 23) - 11) / 13.0f;
            input.front() = repeat ? -65536.0f : 65536.0f;
            if (!graph.compute(input.data(), output.data(), error)) throw std::runtime_error(error);
            ggml_backend_tensor_get(raw, raw_output.data(), 0, raw_output.size() * sizeof(float));
            for (int c = 0; c < channels; ++c)
                for (int y = 0; y < out_height; ++y)
                    for (int x = 0; x < out_width; ++x) {
                        // Independent double-precision scalar convolution.
                        double expected = 0;
                        for (int ky = 0; ky < kernel_size; ++ky)
                            for (int kx = 0; kx < kernel_size; ++kx) {
                                const int sy = y * stride + ky - padding, sx = x * stride + kx - padding;
                                if (sy >= 0 && sy < height && sx >= 0 && sx < width)
                                    expected += double(input[(size_t(c) * height + sy) * width + sx]) *
                                                weights[(size_t(c) * kernel_size + ky) * kernel_size + kx];
                            }
                        expected += biases[c];
                        const size_t index = (size_t(c) * out_height + y) * out_width + x;
                        const double raw_error = std::abs(raw_output[index] - expected);
                        max_raw_error = std::max(max_raw_error, raw_error);
                        if (!std::isfinite(raw_output[index]) || raw_error > 8e-5 + 2e-6 * std::abs(expected))
                            throw std::runtime_error("MI-GAN specialized depthwise differs from scalar oracle");
                        const float cast = float(expected);
                        const float activated = std::max(
                            -256.0f, std::min(256.0f, (cast < 0 ? cast * 0.2f : cast) * 1.4142135623730950488f));
                        const double activation_error = std::abs(double(output[index]) - activated);
                        max_activation_error = std::max(max_activation_error, activation_error);
                        if (!std::isfinite(output[index]) || activation_error > 8e-5 + 2e-6 * std::abs(activated))
                            throw std::runtime_error("MI-GAN fused activation differs from scalar oracle");
                    }
        }
        std::vector<float> actual(weights.size());
        ggml_backend_tensor_get(kernel, actual.data(), 0, actual.size() * sizeof(float));
        if (std::memcmp(actual.data(), weights.data(), weights.size() * sizeof(float)))
            throw std::runtime_error("MI-GAN CPU kernel control changed immutable weights");
    }
    std::printf("MIGAN_GRAPH_CPU_KERNELS {\"cases\":6,\"repeats_per_case\":2,\"max_raw_abs\":%.9g,"
                "\"max_activation_abs\":%.9g,\"border_stride_tail_oracles_passed\":true,\"passed\":true}\n",
                max_raw_error, max_activation_error);
}
