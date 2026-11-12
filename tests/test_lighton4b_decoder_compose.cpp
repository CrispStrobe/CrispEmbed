#include "lighton4b_decoder_compose.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
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
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights(argv[1], backend, "lighton4b-shared-synthetic-decoder", loaded)) return 2;
    auto layer = [&](const std::string & prefix, bool full) {
        auto get = [&](const char * name) { return core_gguf::try_get(loaded.tensors, (prefix + name).c_str()); };
        qwen35_ocr::decoder_layer_weights w;
        w.full_attention = full;
        w.input_norm = get("input_layernorm.weight");
        w.post_norm = get("post_attention_layernorm.weight");
        w.gate = get("mlp.gate_proj.weight");
        w.up = get("mlp.up_proj.weight");
        w.down = get("mlp.down_proj.weight");
        if (full)
            w.attention = { get("self_attn.q_proj.weight"), get("self_attn.k_proj.weight"),
                            get("self_attn.v_proj.weight"), get("self_attn.o_proj.weight"),
                            get("self_attn.q_norm.weight"), get("self_attn.k_norm.weight") };
        else
            w.linear = { get("linear_attn.in_proj_qkv.weight"),
                         get("linear_attn.in_proj_z.weight"),
                         get("linear_attn.in_proj_a.weight"),
                         get("linear_attn.in_proj_b.weight"),
                         get("linear_attn.out_proj.weight"),
                         get("linear_attn.conv1d.weight"),
                         get("linear_attn.A_log"),
                         get("linear_attn.dt_bias"),
                         get("linear_attn.norm.weight") };
        return w;
    };
    qwen35_ocr::decoder_weights w;
    w.norm = core_gguf::try_get(loaded.tensors, "norm.weight");
    for (int i = 0; i < 32; ++i) w.layers.push_back(layer(i % 4 == 3 ? "full." : "linear.", i % 4 == 3));
    crispembed_diff::Ref reference;
    if (!reference.load(argv[2])) return 2;
    const char * gates[]{ "CRISPEMBED_LIGHTON4B_DECODER_COMPOSE", "CRISPEMBED_LIGHTON4B_LINEAR_BLOCK",
                          "CRISPEMBED_LIGHTON4B_ATTENTION_BLOCK", "CRISPEMBED_QWEN35_GDN_GROUPED_HEADS",
                          "CRISPEMBED_QWEN35_OCR" };
    for (auto * g : gates) gate(g, "1");
    bool ok = true;
    size_t controls = 0;
    auto check = [&](bool value) {
        ++controls;
        if (!value) std::printf("FAIL decoder-control=%zu\n", controls);
        ok &= value;
    };
    auto input = reference.get_f32("prefill.input");
    auto positions = reference.get_f32("prefill.positions");
    if (!input.first || input.second != 5120 || !positions.first || positions.second != 6) return 2;
    qwen35_ocr::lighton4b_decoder_outputs output;
    auto invoke = [&](int tokens, const qwen35_ocr::lighton4b_decoder_cache * previous) {
        return qwen35_ocr::lighton4b_decoder_hidden_cpu(w, input.first, positions.first, tokens, 2, previous, output);
    };
    auto reject = [&](int tokens, const qwen35_ocr::lighton4b_decoder_cache * previous) {
        return !invoke(tokens, previous) && output.hidden.empty() && output.norm.empty() && output.cache.layers.empty();
    };
    for (auto * g : gates) {
        for (auto * value : { "", "0" }) {
            gate(g, value);
            check(reject(2, nullptr));
        }
        gate(g, "1");
    }
    check(reject(0, nullptr));
    check(reject(17, nullptr));
    auto last = w.layers.back();
    w.layers.pop_back();
    check(reject(2, nullptr));
    w.layers.push_back(last);
    w.layers[0].full_attention = true;
    check(reject(2, nullptr));
    w.layers[0].full_attention = false;
    qwen35_ocr::lighton4b_decoder_cache cache;
    auto emit = [&](const std::string & name, const std::vector<float> & value) {
        std::ofstream f(std::string(argv[3]) + "/" + name + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(value.data()), value.size() * sizeof(float));
        check(bool(f));
    };
    for (auto * label : { "prefill", "cached0", "full_prefix" }) {
        std::string phase = label;
        auto x = reference.get_f32((phase + ".input").c_str());
        auto pos = reference.get_f32((phase + ".positions").c_str());
        if (!x.first || !pos.first || pos.second != 3 * x.second / 2560) return 2;
        const auto previous = cache;
        const bool cached = phase == "cached0";
        check(qwen35_ocr::lighton4b_decoder_hidden_cpu(w, x.first, pos.first, int(x.second / 2560), 2,
                                                       cached ? &cache : nullptr, output));
        if (output.hidden.size() != 32 || output.cache.layers.size() != 32) return 1;
        for (size_t i = 0; i < 32; ++i) {
            const auto & c = output.cache.layers[i];
            emit(phase + ".layer." + std::to_string(i), output.hidden[i]);
            const auto p = phase + ".cache." + std::to_string(i) + ".";
            if (c.full_attention) {
                emit(p + "keys", c.attention.keys);
                emit(p + "values", c.attention.values);
            } else {
                emit(p + "conv", c.linear.conv);
                emit(p + "recurrent", c.linear.recurrent);
            }
            if (cached) {
                const auto & a = cache.layers[i];
                const auto & b = previous.layers[i];
                check(a.attention.keys == b.attention.keys && a.attention.values == b.attention.values &&
                      a.linear.conv == b.linear.conv && a.linear.recurrent == b.linear.recurrent);
            }
        }
        emit(phase + ".norm", output.norm);
        if (phase == "prefill") cache = std::move(output.cache);
    }
    // A cache made under one dequantized weight dtype cannot cross modes.
    for (auto * g : { "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT", "CRISPEMBED_LIGHTON4B_QUANT", "CRISPEMBED_QWEN35_QUANT" })
        gate(g, "1");
    gate("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", "1");
    check(reject(1, &cache));
    check(invoke(2, nullptr));
    const auto rounded_cache = output.cache;
    check(rounded_cache.bf16_quant_weights);
    check(invoke(1, &rounded_cache));
    gate("CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", "0");
    check(reject(1, &rounded_cache));
    check(invoke(1, &cache));
    // New arithmetic identity must be rejected at outer/full/inactive layers.
    auto wrong_rope = cache;
    wrong_rope.physical_rope = true;
    check(reject(1, &wrong_rope));
    wrong_rope = cache;
    wrong_rope.layers[3].attention.physical_rope = true;
    check(reject(1, &wrong_rope));
    wrong_rope = cache;
    wrong_rope.layers[0].attention.physical_rope = true;
    check(reject(1, &wrong_rope));
    auto bad = cache;
    bad.tokens = 0;
    check(reject(1, &bad));
    bad.tokens = 2;
    bad.layers[31].full_attention = false;
    check(reject(1, &bad));
    bad.layers[31].full_attention = true;
    bad.layers[3].linear.tokens = 1;
    check(reject(1, &bad));
    bad.layers[3].linear.tokens = 0;
    bad.layers[0].linear.tokens = 1;
    check(reject(1, &bad));
    bad.layers[0].linear.tokens = 2;
    bad.layers[3].attention.tokens = 1;
    check(reject(1, &bad));
    bad.layers[3].attention.tokens = 2;
    bad.layers[0].linear.conv.pop_back();
    check(reject(1, &bad));
    check(cache.tokens == 2 && cache.layers.size() == 32);
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    std::printf("DECODER_COMPOSE controls=%zu passed=%d shared_synthetic_weight_sets=2 checkpoint=0 logits=0\n",
                controls, ok);
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
