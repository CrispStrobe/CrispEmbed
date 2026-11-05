// Diagnostic only: replay checkpoint layer0 on an authenticated retained input.
#include "lighton4b_linear_block.h"
#include "core/clean_exit.h"
#include "ggml-cpu.h"
#include <fstream>
#include <iterator>
#include <map>
// Diagnostic scalar reconstruction of the pinned Torch CPU inner-sum order.
// No reference values are embedded; 8/16 lanes test the two source vector widths.
static float diagnostic_tree_mean(const float * input, int lanes) {
    float partial[4][16]{};
    const int rows = 2560 / (4 * lanes);
    for (int first = 0; first < rows; first += 16) {
        float block[4][16]{};
        for (int row = first; row < std::min(rows, first + 16); ++row)
            for (int stream = 0; stream < 4; ++stream)
                for (int lane = 0; lane < lanes; ++lane) {
                    const float value = input[(row * 4 + stream) * lanes + lane];
                    block[stream][lane] += value * value;
                }
        for (int stream = 0; stream < 4; ++stream)
            for (int lane = 0; lane < lanes; ++lane) partial[stream][lane] += block[stream][lane];
    }
    for (int stream = 1; stream < 4; ++stream)
        for (int lane = 0; lane < lanes; ++lane) partial[0][lane] += partial[stream][lane];
    float sum = 0;
    for (int lane = 0; lane < lanes; ++lane) sum += partial[0][lane];
    return sum / 2560.0f;
}
static int run(int argc, char ** argv) {
    if (argc != 4 && argc != 5 && argc != 7 && argc != 8) return 2;
    // RNE tie cases distinguish BF16 from truncation and FP16 conversion.
    if (qwen35_ocr::lighton4b_bf16_round(1.00390625f) != 1.0f ||
        qwen35_ocr::lighton4b_bf16_round(1.01171875f) != 1.015625f ||
        qwen35_ocr::lighton4b_bf16_round(-1.00390625f) != -1.0f || qwen35_ocr::lighton4b_bf16_round(0.0f) != 0.0f)
        return 6;
    if (!qwen35_ocr::lighton4b_bf16_linear_enabled()) {
        const float zero = 0;
        std::vector<float> rejected_output{ 1 }, rejected_state{ 1 };
        if (qwen35_ocr::delta_pre_normalized_cpu(&zero, &zero, &zero, &zero, &zero, nullptr, 1, 1, 1, 1, 1,
                                                 rejected_output, rejected_state) ||
            !rejected_output.empty() || !rejected_state.empty())
            return 8;
    }
    auto backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad weights;
    if (!backend || !core_gguf::load_weights(argv[1], backend, "layer0-source-fed", weights)) return 2;
    auto get = [&](const char * name) { return core_gguf::try_get(weights.tensors, name); };
    qwen35_ocr::decoder_layer_weights w;
    w.input_norm = get("input_layernorm.weight");
    w.post_norm = get("post_attention_layernorm.weight");
    w.gate = get("mlp.gate_proj.weight");
    w.up = get("mlp.up_proj.weight");
    w.down = get("mlp.down_proj.weight");
    w.linear = { get("linear_attn.in_proj_qkv.weight"),
                 get("linear_attn.in_proj_z.weight"),
                 get("linear_attn.in_proj_a.weight"),
                 get("linear_attn.in_proj_b.weight"),
                 get("linear_attn.out_proj.weight"),
                 get("linear_attn.conv1d.weight"),
                 get("linear_attn.A_log"),
                 get("linear_attn.dt_bias"),
                 get("linear_attn.norm.weight") };
    std::ifstream stream(argv[2], std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != 154 * 2560 * 4) return 2;
    std::vector<float> input(154 * 2560);
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(input.data()), input.size() * 4);
    if (!stream) return 2;
    if (argc == 5 || argc == 7 || argc == 8) {
        if (std::string(argv[4]) != "norm-only") return 2;
        std::vector<float> norm(2560);
        if (!qwen35_ocr::tensor_floats(w.input_norm, norm.data())) return 2;
        for (const char * precise : { "0", "1" }) {
#ifdef _WIN32
            _putenv_s("CRISPEMBED_QWEN35_PRECISE_RMS", precise);
#else
            setenv("CRISPEMBED_QWEN35_PRECISE_RMS", precise, 1);
#endif
            std::vector<float> values(input.size()), scales(154);
            for (int token = 0; token < 154; ++token) {
                const size_t base = size_t(token) * 2560;
                const float scale = qwen35_ocr::rms_scale_cpu(input.data() + base, 2560, 1e-6f);
                scales[token] = scale;
                for (int lane = 0; lane < 2560; ++lane)
                    values[base + lane] = input[base + lane] * scale * (1.0f + norm[lane]);
            }
            std::ofstream scale_file(std::string(argv[3]) + "/norm-precise" + precise + ".scale.f32", std::ios::binary);
            scale_file.write(reinterpret_cast<const char *>(scales.data()), scales.size() * 4);
            if (!scale_file) return 4;
            for (const char * dtype : { "f32", "bf16" }) {
                if (std::string(dtype) == "bf16") qwen35_ocr::lighton4b_bf16_round(values);
                std::ofstream output(std::string(argv[3]) + "/norm-precise" + precise + "." + dtype + ".f32",
                                     std::ios::binary);
                output.write(reinterpret_cast<const char *>(values.data()), values.size() * 4);
                if (!output) return 4;
            }
        }
        if (argc == 7 || argc == 8) {
            std::vector<float> source_means(154), source_scales(154);
            auto read = [&](const char * path, std::vector<float> & values) {
                std::ifstream f(path, std::ios::binary | std::ios::ate);
                if (!f || f.tellg() != 154 * 4) return false;
                f.seekg(0);
                f.read(reinterpret_cast<char *>(values.data()), values.size() * 4);
                return bool(f);
            };
            if (!read(argv[5], source_means) || !read(argv[6], source_scales)) return 2;
            for (const char * variant : { "mean-f32-sqrt", "mean-f64-rsqrt", "scale", "sum-round-mean", "sum-tree8",
                                          "sum-tree16", "candidate" }) {
                std::vector<float> values(input.size()), scales(154), means(154);
                for (int token = 0; token < 154; ++token) {
                    const size_t base = size_t(token) * 2560;
                    float mean = source_means[token];
                    const std::string kind(variant);
                    if (kind == "sum-round-mean") {
                        double sum = 0;
                        for (int lane = 0; lane < 2560; ++lane) sum += double(input[base + lane] * input[base + lane]);
                        mean = float(sum) / 2560.0f;
                    } else if (kind == "sum-tree8" || kind == "sum-tree16") {
                        mean = diagnostic_tree_mean(input.data() + base, kind == "sum-tree8" ? 8 : 16);
                    }
                    means[token] = mean;
                    const float denominator = mean + 1e-6f;
                    const float scale =
                        kind == "candidate" ? qwen35_ocr::lighton4b_bf16_rms_scale(input.data() + base, 2560, 1e-6f)
                        : std::string(variant) == "scale"          ? source_scales[token]
                        : std::string(variant) == "mean-f64-rsqrt" ? float(1.0 / std::sqrt(double(denominator)))
                                                                   : 1.0f / std::sqrt(denominator);
                    scales[token] = scale;
                    for (int lane = 0; lane < 2560; ++lane)
                        values[base + lane] = input[base + lane] * scale * (1.0f + norm[lane]);
                }
                std::ofstream mean_file(std::string(argv[3]) + "/norm-source-" + variant + ".mean.f32",
                                        std::ios::binary);
                mean_file.write(reinterpret_cast<const char *>(means.data()), means.size() * 4);
                if (!mean_file) return 4;
                std::ofstream scale_file(std::string(argv[3]) + "/norm-source-" + variant + ".scale.f32",
                                         std::ios::binary);
                scale_file.write(reinterpret_cast<const char *>(scales.data()), scales.size() * 4);
                if (!scale_file) return 4;
                for (const char * dtype : { "f32", "bf16" }) {
                    if (std::string(dtype) == "bf16") qwen35_ocr::lighton4b_bf16_round(values);
                    std::ofstream output(std::string(argv[3]) + "/norm-source-" + variant + "." + dtype + ".f32",
                                         std::ios::binary);
                    output.write(reinterpret_cast<const char *>(values.data()), values.size() * 4);
                    if (!output) return 4;
                }
            }
        }
        if (argc == 8) {
            std::vector<float> gdn(154 * 32 * 128);
            std::ifstream f(argv[7], std::ios::binary | std::ios::ate);
            if (!f || f.tellg() != std::streamoff(gdn.size() * 4)) return 2;
            f.seekg(0);
            f.read(reinterpret_cast<char *>(gdn.data()), gdn.size() * 4);
            if (!f) return 2;
            for (int row = 0; row < 154 * 32; ++row) {
                const size_t base = size_t(row) * 128;
                const float scale = qwen35_ocr::lighton4b_bf16_rms_scale(gdn.data() + base, 128, 1e-6f);
                for (int lane = 0; lane < 128; ++lane)
                    gdn[base + lane] = qwen35_ocr::lighton4b_bf16_round(gdn[base + lane] * scale);
            }
            std::ofstream output(std::string(argv[3]) + "/gdn-candidate-normalized.f32", std::ios::binary);
            output.write(reinterpret_cast<const char *>(gdn.data()), gdn.size() * 4);
            if (!output) return 4;
        }
        core_gguf::free_weights(weights);
        ggml_backend_free(backend);
        return 0;
    }
    qwen35_ocr::lighton4b_linear_cache cache;
    std::map<std::string, std::vector<float>> stages;
    for (int offset = 0; offset < 154; offset += 16) {
        qwen35_ocr::lighton4b_linear_block_outputs b;
        if (!qwen35_ocr::lighton4b_linear_block_cpu(w, input.data() + offset * 2560, std::min(16, 154 - offset), 2,
                                                    offset ? &cache : nullptr, b))
            return 3;
        auto append = [&](const char * name, const std::vector<float> & v) {
            auto & target = stages[name];
            target.insert(target.end(), v.begin(), v.end());
        };
        if (offset == 0) {
            auto incompatible = b.cache;
            incompatible.bf16_activations = !incompatible.bf16_activations;
            qwen35_ocr::lighton4b_linear_block_outputs rejected;
            if (qwen35_ocr::lighton4b_linear_block_cpu(w, input.data(), 1, 2, &incompatible, rejected) ||
                !rejected.hidden.empty())
                return 7;
        }
        qwen35_ocr::projected_linear_outputs linear;
        const bool replay =
            qwen35_ocr::lighton4b_bf16_linear_enabled()
                ? qwen35_ocr::lighton4b_bf16_grouped_cpu(w.linear, b.input_norm.data(), std::min(16, 154 - offset), 2,
                                                         offset ? cache.conv.data() : nullptr,
                                                         offset ? cache.recurrent.data() : nullptr, linear)
                : qwen35_ocr::linear_grouped_cpu(w.linear, 16, b.input_norm.data(), 1, std::min(16, 154 - offset), 2,
                                                 offset ? cache.conv.data() : nullptr,
                                                 offset ? cache.recurrent.data() : nullptr, linear);
        if (!replay || linear.attention != b.attention || linear.core.state != b.cache.recurrent ||
            linear.core.conv_history != b.cache.conv)
            return 5;
        append("input_norm", b.input_norm);
        append("qkv", linear.qkv);
        append("z", linear.z);
        append("a", linear.a);
        append("b", linear.b);
        append("gdn_norm", linear.core.values);
        append("attention", b.attention);
        append("attention_residual", b.attention_residual);
        append("post_norm", b.post_norm);
        append("gate", b.gate);
        append("up", b.up);
        append("activation", b.activation);
        append("down", b.down);
        append("hidden", b.hidden);
        cache = std::move(b.cache);
    }
    stages["conv"] = cache.conv;
    stages["recurrent"] = cache.recurrent;
    for (auto & item : stages) {
        std::ofstream output(std::string(argv[3]) + "/" + item.first + ".f32", std::ios::binary);
        output.write(reinterpret_cast<const char *>(item.second.data()), item.second.size() * 4);
        if (!output) return 4;
    }
    core_gguf::free_weights(weights);
    ggml_backend_free(backend);
    return 0;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
