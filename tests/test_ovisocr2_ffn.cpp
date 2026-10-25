// Validation-only observations of the actual decoder source. Never linked into production.
#include "core/clean_exit.h"
#include "qwen35_decoder.h"
#include "qwen35_math.h"
#include "qwen35_parity.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace {
const qwen35_ocr::decoder_layer_weights * observed_layer;
const crispembed_diff::Ref * targets;
std::string mode;
std::map<std::string, std::vector<float>> captured;
std::vector<float> get(const std::string & name) {
    auto x = targets->get_f32(name);
    if (!x.first || !x.second) throw std::runtime_error("Missing reference " + name);
    return { x.first, x.first + x.second };
}
bool audit_project(const std::vector<ggml_tensor *> & matrices, const float * input, int width, int tokens, int threads,
                   std::vector<std::vector<float>> & output, qwen35_ocr::converted_weight_cache * cache) {
    const bool gu = matrices.size() == 2 && matrices[0] == observed_layer->gate && matrices[1] == observed_layer->up;
    const bool down = matrices.size() == 1 && matrices[0] == observed_layer->down;
    std::vector<float> replacement;
    if (gu) {
        captured["natural_input"] = { input, input + size_t(width) * tokens };
        if (mode != "natural") {
            replacement = get("quant.input");
            input = replacement.data();
        }
        if (!replacement.empty() && replacement.size() != size_t(width) * tokens) return false;
        captured["input"] = { input, input + size_t(width) * tokens };
    }
    if (down) {
        captured[mode == "source_silu" ? "silu" : "product"] = { input, input + size_t(width) * tokens };
        if (mode == "source_product") {
            replacement = get("quant.product");
            input = replacement.data();
        }
    }
    if ((!replacement.empty() && replacement.size() != size_t(width) * tokens) ||
        !qwen35_ocr::project_cpu(matrices, input, width, tokens, threads, output, cache))
        return false;
    if (gu) {
        captured["gate"] = output[0];
        captured["up"] = output[1];
        if (mode == "source_gate_up" || mode == "source_silu") {
            output[0] = get("quant.gate");
            output[1] = get("quant.up");
            if (mode == "source_silu") std::fill(output[1].begin(), output[1].end(), 1.0f);
            if (output[0].size() != size_t(tokens) * matrices[0]->ne[1] || output[1].size() != output[0].size())
                return false;
        }
    }
    if (down) captured["down"] = output[0];
    return true;
}
} // namespace
// Rename external entry points only; calls inside the real decoder use the observer.
#define project_cpu audit_project
#define decoder_cpu ffn_observed_decoder_cpu
#define embed_cpu ffn_observed_embed_cpu
#define map_small_decoder ffn_observed_map_small_decoder
#include "../src/qwen35_decoder.cpp"
#undef map_small_decoder
#undef embed_cpu
#undef decoder_cpu
#undef project_cpu

static bool include(const char * name, void *) {
    return std::string(name).find("l.layers.0.") == 0 || std::string(name) == "l.norm.weight";
}
static bool exact(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
static int run(int argc, char ** argv) {
    if (argc != 3) return 2;
    crispembed_diff::Ref ref;
    targets = &ref;
    if (!ref.load(argv[2])) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights_filtered(argv[1], backend, include, nullptr, "ffn-audit", loaded))
        return 2;
    auto field = [&](const char * name) {
        return core_gguf::try_get(loaded.tensors, (std::string("l.layers.0.") + name).c_str());
    };
    qwen35_ocr::decoder_layer_weights layer;
    layer.input_norm = field("input_layernorm.weight");
    layer.post_norm = field("post_attention_layernorm.weight");
    layer.gate = field("mlp.gate_proj.weight");
    layer.up = field("mlp.up_proj.weight");
    layer.down = field("mlp.down_proj.weight");
    layer.linear = { field("linear_attn.in_proj_qkv.weight"),
                     field("linear_attn.in_proj_z.weight"),
                     field("linear_attn.in_proj_a.weight"),
                     field("linear_attn.in_proj_b.weight"),
                     field("linear_attn.out_proj.weight"),
                     field("linear_attn.conv1d.weight"),
                     field("linear_attn.A_log"),
                     field("linear_attn.dt_bias"),
                     field("linear_attn.norm.weight") };
    observed_layer = &layer;
    bool ok = true;
    for (auto pair :
         { std::make_pair("gate", layer.gate), std::make_pair("up", layer.up), std::make_pair("down", layer.down) }) {
        if (!pair.second || pair.second->type != GGML_TYPE_Q4_K) return 2;
        std::vector<float> x(ggml_nelements(pair.second));
        if (!qwen35_ocr::tensor_floats(pair.second, x.data(), 2)) return 2;
        const auto expected = get("weight." + std::string(pair.first));
        const bool same = exact(x, expected);
        std::printf("%s dequant.%s bit_exact=%d elements=%zu\n", same ? "PASS" : "FAIL", pair.first, same, x.size());
        ok &= same;
        ok &= qwen35_compare(ref, "weight." + std::string(pair.first), x);
    }
    // The real first-layer computation needs neither the vocabulary embedding nor
    // full LM head. A test-only zero head satisfies the outer decoder interface.
    ggml_init_params params = { ggml_tensor_overhead() + 1024, nullptr, true };
    auto * ctx = ggml_init(params);
    if (!ctx) return 2;
    auto * dummy = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1024, 1);
    auto * dummy_buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!dummy_buffer) return 2;
    const std::vector<float> zeros(1024, 0);
    ggml_backend_tensor_set(dummy, zeros.data(), 0, 4096);
    qwen35_ocr::decoder_weights w;
    w.layers = { layer };
    w.norm = core_gguf::try_get(loaded.tensors, "l.norm.weight");
    w.embedding = dummy;
    w.head = dummy;
    const auto input = get("native.embeddings"), original_input = get("original.embeddings"),
               position = get("original.positions");
    if (original_input.size() != input.size() || input.size() % 1024 || input.empty() || input.size() / 1024 > 4096 ||
        position.size() != 3 * (input.size() / 1024))
        return 2;
    const int tokens = int(input.size() / 1024);
    qwen35_ocr::decoder_outputs ordinary;
    qwen35_ocr::attention_config config;
    if (!qwen35_ocr::decoder_cpu(w, config, input.data(), position.data(), 1, tokens, 2, nullptr, ordinary, nullptr))
        return 2;
    const char * dump_env = std::getenv("CRISPEMBED_DIFF_DUMP_DIR");
    const std::string dump_root = dump_env ? dump_env : "";
    for (const char * arm : { "natural", "source_input", "source_gate_up", "source_silu", "source_product" }) {
        if (!dump_root.empty()) {
            const auto path = dump_root + "/" + arm;
            std::filesystem::create_directories(path);
#ifdef _WIN32
            _putenv_s("CRISPEMBED_DIFF_DUMP_DIR", path.c_str());
#else
            setenv("CRISPEMBED_DIFF_DUMP_DIR", path.c_str(), 1);
#endif
        }
        mode = arm;
        captured.clear();
        qwen35_ocr::decoder_outputs result;
        const float * supplied_input = mode == "natural" ? input.data() : original_input.data();
        if (!qwen35_ocr::ffn_observed_decoder_cpu(w, config, supplied_input, position.data(), 1, tokens, 2, nullptr,
                                                  result, nullptr))
            return 2;
        if (mode == "natural") {
            const bool neutral = exact(ordinary.hidden[0], result.hidden[0]) &&
                                 exact(ordinary.cache.layers[0].conv, result.cache.layers[0].conv) &&
                                 exact(ordinary.cache.layers[0].recurrent, result.cache.layers[0].recurrent);
            std::printf("%s observer_neutrality bit_exact=%d\n", neutral ? "PASS" : "FAIL", neutral);
            ok &= neutral;
            ok &= qwen35_compare(ref, "native.layer0", result.hidden[0]);
            for (auto & row : captured) {
                // Observed natural-input drift is diagnostic, never retargeted.
                if (row.first == "natural_input") continue;
                qwen35_compare(ref, "quant." + row.first, row.second);
            }
            qwen35_compare(ref, "original.layer0", result.hidden[0]); // Intended precision loss, not a new reference.
        } else if (mode == "source_input") {
            const bool input_exact = exact(captured["input"], get("quant.input"));
            std::printf("%s source_input bit_exact=%d\n", input_exact ? "PASS" : "FAIL", input_exact);
            ok &= input_exact;
            for (const char * stage : { "input", "gate", "up", "product", "down" })
                ok &= qwen35_compare(ref, "quant." + std::string(stage), captured[stage]);
            ok &= qwen35_compare(ref, "quant.layer0", result.hidden[0]);
        } else if (mode == "source_gate_up") {
            ok &= qwen35_compare(ref, "quant.product", captured["product"]);
            ok &= qwen35_compare(ref, "quant.down", captured["down"]);
        } else if (mode == "source_silu")
            ok &= qwen35_compare(ref, "quant.silu", captured["silu"]);
        else
            ok &= qwen35_compare(ref, "quant.down", captured["down"]);
        std::printf("ARM %s complete\n", arm);
    }
    ggml_backend_buffer_free(dummy_buffer);
    ggml_free(ctx);
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(run(argc, argv));
}
