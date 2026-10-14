#include "core/clean_exit.h"
#include "qwen35_attention.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static bool compare(const crispembed_diff::Ref & ref, const std::string & name, const std::vector<float> & values) {
    const auto report = ref.compare(name, values.data(), values.size());
    const bool ok = report.found && ref.get_f32(name).second == values.size() && report.n_elem == values.size() &&
                    report.cos_min >= 0.99999f && report.max_abs <= 0.002f && std::isfinite(report.max_abs) &&
                    std::isfinite(report.cos_min);
    std::printf("%s %s: cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g norms=%.9g/%.9g\n",
                ok ? "PASS" : "FAIL", name.c_str(), report.cos_min, report.cos_global, report.max_abs, report.mean_abs,
                report.mine_norm, report.ref_norm);
    return ok;
}


static bool include(const char * name, void *) {
    const std::string value(name);
    return value.find("l.layers.") == 0 && value.find(".self_attn.") != std::string::npos;
}
static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", value);
#else
    setenv("CRISPEMBED_QWEN35_OCR", value, 1);
#endif
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 3) return 2;
    gate("1");
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights_filtered(argv[1], backend, include, nullptr, "qwen35-attention", loaded))
        return 2;
    bool ok = loaded.tensors.size() == 6 * 6;
    for (int index = 3; index < 24; index += 4) {
        auto tensor = [&](const char * suffix) {
            const std::string name = "l.layers." + std::to_string(index) + ".self_attn." + suffix;
            return core_gguf::try_get(loaded.tensors, name.c_str());
        };
        const qwen35_ocr::attention_weights weights{ tensor("q_proj.weight"), tensor("k_proj.weight"),
                                                     tensor("v_proj.weight"), tensor("o_proj.weight"),
                                                     tensor("q_norm.weight"), tensor("k_norm.weight") };
        const qwen35_ocr::attention_config config;
        qwen35_ocr::attention_cache previous;
        for (const std::string phase : { "prefill", "cached", "full_prefix" }) {
            const auto label = phase + ".attention." + std::to_string(index);
            const auto hidden = ref.get_f32(label + ".input");
            const auto shape = ref.shape(label + ".input");
            const auto coords = ref.get_f32(phase + ".decoder.position_ids");
            const auto coord_shape = ref.shape(phase + ".decoder.position_ids");
            if (!hidden.first || shape.size() != 3 || hidden.second != size_t(shape[0] * shape[1] * shape[2]) ||
                !coords.first || coord_shape.size() != 3) {
                ok = false;
                break;
            }
            const int batch = int(shape[2]), tokens = int(shape[1]), count = batch * tokens;
            std::vector<float> positions(3 * count);
            // GGUF shapes reverse HF axes. The text chronology plane is omitted;
            // cached one-plane positions broadcast to the three rotary axes.
            // Shape is [tokens,batch,planes], unlike hidden [width,tokens,batch].
            const int actual_planes = int(coord_shape[2]);
            if (coord_shape[0] != tokens || coord_shape[1] != batch ||
                (actual_planes != 1 && actual_planes != 3 && actual_planes != 4) ||
                coords.second != size_t(actual_planes * count)) {
                ok = false;
                break;
            }
            for (int axis = 0; axis < 3; ++axis) {
                const int source = actual_planes == 1 ? 0 : actual_planes - 3 + axis;
                std::copy_n(coords.first + source * count, count, positions.data() + axis * count);
            }
            qwen35_ocr::attention_outputs current;
            const auto * cache = phase == "cached" ? &previous : nullptr;
            if (!qwen35_ocr::attention_cpu(weights, config, hidden.first, positions.data(), batch, tokens, 4, cache,
                                           current)) {
                std::printf("FAIL native attention %s\n", label.c_str());
                ok = false;
                break;
            }
            ok &= compare(ref, label + ".q", current.q);
            ok &= compare(ref, label + ".k", current.k);
            ok &= compare(ref, label + ".v", current.v);
            ok &= compare(ref, label + ".q_norm", current.q_norm);
            ok &= compare(ref, label + ".k_norm", current.k_norm);
            ok &= compare(ref, label + ".pooled", current.pooled);
            ok &= compare(ref, label + ".output", current.attention);
            if (phase != "full_prefix") {
                ok &= compare(ref, phase + ".cache." + std::to_string(index) + ".keys", current.cache.keys);
                ok &= compare(ref, phase + ".cache." + std::to_string(index) + ".values", current.cache.values);
            }
            if (phase == "prefill") {
                qwen35_ocr::attention_outputs rejected;
                gate("0");
                ok &= !qwen35_ocr::attention_cpu(weights, config, hidden.first, positions.data(), batch, tokens, 4,
                                                 nullptr, rejected);
                gate("1");
                auto bad = config;
                bad.rotary_dim = config.head_dim + 2;
                ok &= !qwen35_ocr::attention_cpu(weights, bad, hidden.first, positions.data(), batch, tokens, 4,
                                                 nullptr, rejected);
                auto invalid = current.cache;
                invalid.keys.pop_back();
                ok &= !qwen35_ocr::attention_cpu(weights, config, hidden.first, positions.data(), batch, tokens, 4,
                                                 &invalid, rejected);
                auto wrong = current.pooled;
                for (auto & value : wrong) value += 0.1f;
                const auto difference = ref.compare(label + ".pooled", wrong.data(), wrong.size());
                ok &= difference.found && difference.max_abs > 0.002f;
                previous = std::move(current.cache);
            }
        }
    }
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    std::printf(
        "%s GGUF-weighted Qwen3.5 six full-attention blocks, native KV carry and fresh full-prefix; full OCR pending\n",
        ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
