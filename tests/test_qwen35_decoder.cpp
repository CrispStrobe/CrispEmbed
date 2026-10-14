#include "qwen35_decoder.h"
#include "qwen35_vision.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "qwen35_parity.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

static bool include(const char * name, void * user) {
    return std::string(name).find("l.") == 0 || std::string(name) == "lm_head.weight" ||
           (user && std::string(name).find("v.") == 0);
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
    if (argc != 3 && argc != 4) return 2;
    const bool native_vision = argc == 4;
    if (native_vision && std::string(argv[3]) != "--native-vision") return 2;
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", "1");
#else
    setenv("CRISPEMBED_QWEN35_OCR", "1", 1);
#endif
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights_filtered(argv[1], backend, include,
                                                      native_vision ? const_cast<bool *>(&native_vision) : nullptr,
                                                      "qwen35-decoder", loaded))
        return 2;
    qwen35_ocr::decoder_weights weights;
    if (!qwen35_ocr::map_small_decoder(loaded.tensors, weights)) return 2;
    qwen35_ocr::vision_outputs vision;
    std::vector<int> input_ids;
    if (native_vision) {
        const auto pixels = ref.get_f32("input.pixel_values");
        const auto pixel_shape = ref.shape("input.pixel_values");
        const auto grid = ref.get_f32("input.image_grid_thw");
        const auto ids = ref.get_f32("input.input_ids");
        if (!pixels.first || pixel_shape.size() != 2 || pixel_shape[0] != 1536 ||
            pixels.second != size_t(pixel_shape[0] * pixel_shape[1]) || !grid.first || grid.second % 3 || !ids.first)
            return 2;
        std::vector<int> dimensions;
        for (size_t i = 0; i < grid.second; ++i) {
            if (!std::isfinite(grid.first[i]) || grid.first[i] < 1 || grid.first[i] > 1024 ||
                std::floor(grid.first[i]) != grid.first[i])
                return 2;
            dimensions.push_back(int(grid.first[i]));
        }
        for (size_t i = 0; i < ids.second; ++i) {
            if (!std::isfinite(ids.first[i]) || ids.first[i] < 0 || ids.first[i] >= 248320 ||
                std::floor(ids.first[i]) != ids.first[i])
                return 2;
            input_ids.push_back(int(ids.first[i]));
        }
        qwen35_ocr::vision_weights visual;
        if (!qwen35_ocr::map_small_vision(loaded.tensors, visual) ||
            !qwen35_ocr::vision_cpu(visual, {}, pixels.first, int(pixel_shape[1]), dimensions, 4, vision))
            return 2;
    }
    const qwen35_ocr::attention_config config;
    bool ok = true;
    qwen35_ocr::decoder_cache prefill_cache;
    std::vector<float> first_logits;
    int first_selected = -1;
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
        std::vector<float> native_hidden;
        const float * decoder_input = hidden.first;
        if (native_vision) {
            std::vector<int> ids = phase == "cached" ? std::vector<int>{ first_selected } : input_ids;
            if (phase == "full_prefix") ids.push_back(first_selected);
            if (ids.size() != size_t(tokens) || !qwen35_ocr::embed_cpu(weights.embedding, ids, native_hidden)) {
                ok = false;
                break;
            }
            if (phase != "cached") {
                size_t image = 0;
                for (size_t i = 0; i < ids.size(); ++i)
                    if (ids[i] == 248056) {
                        if ((image + 1) * 1024 > vision.merged.size()) {
                            ok = false;
                            break;
                        }
                        std::copy_n(vision.merged.data() + image * 1024, 1024, native_hidden.data() + i * 1024);
                        ++image;
                    }
                ok &= image * 1024 == vision.merged.size();
            }
            ok &= qwen35_compare(ref, phase + ".decoder.inputs_embeds", native_hidden);
            decoder_input = native_hidden.data();
        }
        qwen35_ocr::decoder_outputs current;
        if (!qwen35_ocr::decoder_cpu(weights, config, decoder_input, coords.data(), batch, tokens, 4,
                                     phase == "cached" ? &prefill_cache : nullptr, current)) {
            ok = false;
            break;
        }
        for (size_t i = 0; i < current.hidden.size(); ++i) {
            if (!qwen35_compare(ref, phase + ".decoder.layer." + std::to_string(i), current.hidden[i])) {
                ok = false;
                // Keep the earliest divergence visible; later layers are not proof.
                break;
            }
        }
        ok &= qwen35_compare(ref, phase + ".decoder.norm", current.norm);
        ok &= qwen35_compare(ref, phase + ".logits", current.logits);
        const auto oracle = ref.get_f32(phase + ".logits");
        ok &= oracle.first && argmax(current.logits) ==
                                  int(std::max_element(oracle.first, oracle.first + oracle.second) - oracle.first);
        for (size_t i = 0; i < current.cache.layers.size(); ++i) {
            const auto label = phase + ".cache." + std::to_string(i);
            if (phase == "full_prefix") continue;
            const auto & cache = current.cache.layers[i];
            if (weights.layers[i].full_attention) {
                ok &= qwen35_compare(ref, label + ".keys", cache.attention.keys);
                ok &= qwen35_compare(ref, label + ".values", cache.attention.values);
            } else {
                ok &= qwen35_compare(ref, label + ".conv_states", cache.conv);
                ok &= qwen35_compare(ref, label + ".recurrent_states", cache.recurrent);
            }
        }
        if (phase == "prefill") {
            prefill_cache = std::move(current.cache);
            first_selected = argmax(current.logits);
            first_logits = std::move(current.logits);
        }
    }
    // Free-running native argmax selects the next embedding. The connected arm
    // starts from native vision/text embeddings; packed pixels/IDs/positions
    // still come from the official frontend.
    const auto expected = ref.get_f32("generated.ids");
    std::vector<int> ids;
    auto cache = std::move(prefill_cache);
    auto logits = std::move(first_logits);
    if (!expected.first || expected.second == 0 || expected.second > 12 || logits.empty())
        ok = false;
    else
        for (size_t step = 0; step < expected.second; ++step) {
            ok &= qwen35_compare(ref, "generated.scores." + std::to_string(step), logits);
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
    std::printf("%s composed 24-layer native decoder and greedy IDs; %s, native frontend pending\n",
                ok ? "PASS" : "FAIL",
                native_vision ? "native vision/text embeddings from official packed pixels/IDs/positions"
                              : "official merged embeddings/positions supplied");
    return ok ? 0 : 1;
}
