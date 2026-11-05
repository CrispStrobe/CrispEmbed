#include "lighton4b_linear_block.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
static void gate(const char * name) {
#ifdef _WIN32
    _putenv_s(name, "1");
#else
    setenv(name, "1", 1);
#endif
}
static int run(int argc, char ** argv) {
    if (argc != 4) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad weights;
    if (!backend || !core_gguf::load_weights(argv[1], backend, "lighton4b-retained-upstream", weights)) return 2;
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
    for (auto * g :
         { "CRISPEMBED_LIGHTON4B_LINEAR_BLOCK", "CRISPEMBED_QWEN35_OCR", "CRISPEMBED_QWEN35_GDN_GROUPED_HEADS" })
        gate(g);
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    bool ok = true;
    auto emit = [&](const std::string & name, const std::vector<float> & value) {
        std::ofstream f(std::string(argv[3]) + "/" + name + ".f32", std::ios::binary);
        f.write(reinterpret_cast<const char *>(value.data()), value.size() * sizeof(float));
        ok &= bool(f);
    };
    for (auto * label : { "source_input", "native_input" }) {
        const std::string p = label;
        auto x = ref.get_f32((p + ".input").c_str());
        auto norm = ref.get_f32((p + ".input_norm").c_str());
        if (!x.first || x.second != 5120 || !norm.first || norm.second != 5120) return 2;
        qwen35_ocr::lighton4b_linear_block_outputs out;
        if (!qwen35_ocr::lighton4b_linear_block_cpu(w, x.first, 2, 2, nullptr, out)) return 1;
        emit(p + ".input_norm", out.input_norm);
        emit(p + ".attention", out.attention);
        emit(p + ".attention_residual", out.attention_residual);
        emit(p + ".post_norm", out.post_norm);
        emit(p + ".gate", out.gate);
        emit(p + ".up", out.up);
        emit(p + ".activation", out.activation);
        emit(p + ".down", out.down);
        emit(p + ".hidden", out.hidden);
        emit(p + ".conv", out.cache.conv);
        emit(p + ".recurrent", out.cache.recurrent);
        // Source-fed stage boundary exists only in this test. No source cache is fed.
        qwen35_ocr::projected_linear_outputs core;
        if (!qwen35_ocr::linear_grouped_cpu(w.linear, 16, norm.first, 1, 2, 2, nullptr, nullptr, core)) return 1;
        emit(p + ".source_norm.conv", core.core.conv_history);
        emit(p + ".source_norm.recurrent", core.core.state);
        auto post = ref.get_f32((p + ".post_norm").c_str());
        auto gate_input = ref.get_f32((p + ".gate").c_str());
        auto up_input = ref.get_f32((p + ".up").c_str());
        auto activation = ref.get_f32((p + ".activation").c_str());
        if (!post.first || post.second != 5120 || !gate_input.first || gate_input.second != 18432 || !up_input.first ||
            up_input.second != 18432 || !activation.first || activation.second != 18432)
            return 2;
        std::vector<std::vector<float>> ffn;
        if (!qwen35_ocr::project_cpu({ w.gate, w.up }, post.first, 2560, 2, 2, ffn)) return 1;
        emit(p + ".source_post_norm.gate", ffn[0]);
        emit(p + ".source_post_norm.up", ffn[1]);
        std::vector<float> activated(18432);
        for (size_t i = 0; i < activated.size(); ++i) {
            const float g = gate_input.first[i];
            activated[i] = (g / (1.0f + std::exp(-g))) * up_input.first[i];
        }
        emit(p + ".source_gate_up.activation", activated);
        ffn.clear();
        if (!qwen35_ocr::project_cpu({ w.down }, activation.first, 9216, 2, 2, ffn)) return 1;
        emit(p + ".source_activation.down", ffn[0]);
    }
    core_gguf::free_weights(weights);
    ggml_backend_free(backend);
    std::printf("RETAINED_UPSTREAM native_complete=%d full_decoder_repeated=0 checkpoint=0\n", ok);
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
