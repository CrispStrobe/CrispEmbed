#include "lighton4b_linear_block.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static int run(int argc, char ** argv) {
    if (argc != 4) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad weights;
    if (!backend || !core_gguf::load_weights(argv[1], backend, "lighton4b-block-component", weights)) return 2;
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
    crispembed_diff::Ref reference;
    if (!reference.load(argv[2])) return 2;
    const char * gates[]{ "CRISPEMBED_LIGHTON4B_LINEAR_BLOCK", "CRISPEMBED_QWEN35_OCR",
                          "CRISPEMBED_QWEN35_GDN_GROUPED_HEADS" };
    for (auto * g : gates) gate(g, "1");
    bool ok = true;
    size_t controls = 0;
    auto check = [&](bool value) {
        ++controls;
        if (!value) std::printf("FAIL block-control=%zu\n", controls);
        ok &= value;
    };
    auto input = reference.get_f32("prefill.input");
    if (!input.first || input.second != 5120) return 2;
    qwen35_ocr::lighton4b_linear_block_outputs output;
    auto invoke = [&](int tokens, const qwen35_ocr::lighton4b_linear_cache * previous) {
        return qwen35_ocr::lighton4b_linear_block_cpu(w, input.first, tokens, 2, previous, output);
    };
    for (auto * g : gates) {
        for (auto * value : { "", "0" }) {
            gate(g, value);
            check(!invoke(2, nullptr) && output.hidden.empty());
        }
        gate(g, "1");
    }
    check(!invoke(0, nullptr));
    check(!invoke(17, nullptr));
    auto saved = w.input_norm->ne[0];
    ++w.input_norm->ne[0];
    check(!invoke(2, nullptr));
    w.input_norm->ne[0] = saved;
    w.full_attention = true;
    check(!invoke(2, nullptr));
    w.full_attention = false;
    qwen35_ocr::lighton4b_linear_cache cache;
    auto emit = [&](const std::string & phase, const std::string & name, const std::vector<float> & values) {
        const auto path = std::string(argv[3]) + "/" + phase + "." + name + ".f32";
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        check(bool(f));
    };
    for (auto * phase : { "prefill", "cached0", "full_prefix" }) {
        const auto x = reference.get_f32((std::string(phase) + ".input").c_str());
        if (!x.first || x.second % 2560) return 2;
        const auto previous = cache;
        const bool cached = std::string(phase) == "cached0";
        check(qwen35_ocr::lighton4b_linear_block_cpu(w, x.first, int(x.second / 2560), 2, cached ? &cache : nullptr,
                                                     output));
        if (output.hidden.empty()) return 1;
        if (cached)
            check(cache.tokens == previous.tokens && cache.conv == previous.conv &&
                  cache.recurrent == previous.recurrent);
        emit(phase, "input_norm", output.input_norm);
        emit(phase, "attention", output.attention);
        emit(phase, "attention_residual", output.attention_residual);
        emit(phase, "post_norm", output.post_norm);
        emit(phase, "gate", output.gate);
        emit(phase, "up", output.up);
        emit(phase, "activation", output.activation);
        emit(phase, "down", output.down);
        emit(phase, "hidden", output.hidden);
        emit(phase, "conv", output.cache.conv);
        emit(phase, "recurrent", output.cache.recurrent);
        if (std::string(phase) == "prefill") cache = output.cache;
    }
    auto bad = cache;
    bad.conv.pop_back();
    check(!invoke(1, &bad));
    bad = cache;
    bad.recurrent.pop_back();
    check(!invoke(1, &bad));
    bad = cache;
    bad.tokens = 0;
    check(!invoke(1, &bad));
    bad = cache;
    bad.tokens = 4096;
    check(!invoke(1, &bad));
    check(cache.tokens == 2 && cache.conv.size() == 32768 && cache.recurrent.size() == 524288);
    core_gguf::free_weights(weights);
    ggml_backend_free(backend);
    std::printf("LINEAR_BLOCK controls=%zu passed=%d full_model_inference=0 logits=0\n", controls, ok);
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
