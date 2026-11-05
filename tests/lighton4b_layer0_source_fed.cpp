// Diagnostic only: replay checkpoint layer0 on an authenticated retained input.
#include "lighton4b_linear_block.h"
#include "core/clean_exit.h"
#include "ggml-cpu.h"
#include <fstream>
#include <iterator>
#include <map>
static int run(int argc, char ** argv) {
    if (argc != 4 && argc != 5) return 2;
    // RNE tie cases distinguish BF16 from truncation and FP16 conversion.
    if (qwen35_ocr::lighton4b_bf16_round(1.00390625f) != 1.0f ||
        qwen35_ocr::lighton4b_bf16_round(1.01171875f) != 1.015625f ||
        qwen35_ocr::lighton4b_bf16_round(-1.00390625f) != -1.0f || qwen35_ocr::lighton4b_bf16_round(0.0f) != 0.0f)
        return 6;
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
    if (argc == 5) {
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
