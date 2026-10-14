#include "qwen35_decoder.h"
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
    return std::string(name).find("l.") == 0 || std::string(name) == "lm_head.weight";
}
static bool positions(const crispembed_diff::Ref & ref, const std::string & label, int batch, int tokens,
                      std::vector<float> & output) {
    const auto values = ref.get_f32(label + ".decoder.position_ids");
    const auto shape = ref.shape(label + ".decoder.position_ids");
    if (!values.first || shape.size() != 3 || shape[0] != tokens || shape[1] != batch ||
        (shape[2] != 1 && shape[2] != 3 && shape[2] != 4) || values.second != size_t(shape[2]) * batch * tokens)
        return false;
    const size_t count = size_t(batch) * tokens;
    output.resize(3 * count);
    for (int axis = 0; axis < 3; ++axis) {
        const int source = shape[2] == 1 ? 0 : int(shape[2]) - 3 + axis;
        std::copy_n(values.first + source * count, count, output.data() + axis * count);
    }
    return true;
}
static int argmax(const std::vector<float> & x) {
    return int(std::max_element(x.begin(), x.end()) - x.begin());
}
int main(int argc, char ** argv) {
    if (argc != 3) return 2;
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", "1");
#else
    setenv("CRISPEMBED_QWEN35_OCR", "1", 1);
#endif
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights_filtered(argv[1], backend, include, nullptr, "qwen35-decoder", loaded))
        return 2;
    qwen35_ocr::decoder_weights weights;
    if (!qwen35_ocr::map_small_decoder(loaded.tensors, weights)) return 2;
    const qwen35_ocr::attention_config config;
    bool ok = true;
    qwen35_ocr::decoder_cache prefill_cache;
    std::vector<float> first_logits;
    for (const std::string phase : { "prefill", "cached", "full_prefix" }) {
        const auto hidden = ref.get_f32(phase + ".decoder.inputs_embeds");
        const auto shape = ref.shape(phase + ".decoder.inputs_embeds");
        if (!hidden.first || shape.size() != 3 || shape[0] != 1024 || shape[2] != 1 ||
            hidden.second != size_t(shape[0] * shape[1] * shape[2])) {
            ok = false;
            break;
        }
        std::vector<float> coords;
        const int batch = int(shape[2]), tokens = int(shape[1]);
        if (!positions(ref, phase, batch, tokens, coords)) {
            ok = false;
            break;
        }
        qwen35_ocr::decoder_outputs current;
        if (!qwen35_ocr::decoder_cpu(weights, config, hidden.first, coords.data(), batch, tokens, 4,
                                     phase == "cached" ? &prefill_cache : nullptr, current)) {
            ok = false;
            break;
        }
        for (size_t i = 0; i < current.hidden.size(); ++i) {
            if (!compare(ref, phase + ".decoder.layer." + std::to_string(i), current.hidden[i])) {
                ok = false;
                // Keep the earliest divergence visible; later layers are not proof.
                break;
            }
        }
        ok &= compare(ref, phase + ".decoder.norm", current.norm);
        ok &= compare(ref, phase + ".logits", current.logits);
        const auto oracle = ref.get_f32(phase + ".logits");
        ok &= oracle.first && argmax(current.logits) ==
                                  int(std::max_element(oracle.first, oracle.first + oracle.second) - oracle.first);
        for (size_t i = 0; i < current.cache.layers.size(); ++i) {
            const auto label = phase + ".cache." + std::to_string(i);
            if (phase == "full_prefix") continue;
            const auto & cache = current.cache.layers[i];
            if (weights.layers[i].full_attention) {
                ok &= compare(ref, label + ".keys", cache.attention.keys);
                ok &= compare(ref, label + ".values", cache.attention.values);
            } else {
                ok &= compare(ref, label + ".conv_states", cache.conv);
                ok &= compare(ref, label + ".recurrent_states", cache.recurrent);
            }
        }
        if (phase == "prefill") {
            prefill_cache = std::move(current.cache);
            first_logits = std::move(current.logits);
        }
    }
    // A free-running decoder: native argmax selects the next embedding. Only
    // initial merged embeddings and multimodal position geometry come from HF.
    const auto expected = ref.get_f32("generated.ids");
    std::vector<int> ids;
    auto cache = std::move(prefill_cache);
    auto logits = std::move(first_logits);
    if (!expected.first || expected.second == 0 || expected.second > 12 || logits.empty())
        ok = false;
    else
        for (size_t step = 0; step < expected.second; ++step) {
            ok &= compare(ref, "generated.scores." + std::to_string(step), logits);
            const int id = argmax(logits);
            ids.push_back(id);
            if (id != int(expected.first[step])) {
                ok = false;
                break;
            }
            if (id == 248046) {
                ok &= step + 1 == expected.second;
                break;
            }
            if (step + 1 == expected.second) break;
            std::vector<float> hidden, coords;
            if (!qwen35_ocr::embed_cpu(weights.embedding, { id }, hidden) ||
                !positions(ref, "generated." + std::to_string(step + 1), 1, 1, coords)) {
                ok = false;
                break;
            }
            qwen35_ocr::decoder_outputs current;
            if (!qwen35_ocr::decoder_cpu(weights, config, hidden.data(), coords.data(), 1, 1, 4, &cache, current)) {
                ok = false;
                break;
            }
            cache = std::move(current.cache);
            logits = std::move(current.logits);
        }
    ok &= ids.size() == expected.second;
    std::printf("Native generated IDs:");
    for (int id : ids) std::printf(" %d", id);
    std::printf("\n");
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    std::printf("%s composed 24-layer native decoder and greedy IDs; official merged embeddings/positions supplied, "
                "native frontend pending\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
