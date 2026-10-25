#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "qwen35_decoder.h"
#include "qwen35_vision.h"
#include "qwen35_positions.h"
#include "qwen35_frontend.h"
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
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) return 2;
    const bool native_frontend = argc == 4 && std::string(argv[3]) == "--native-frontend";
    const bool audit_all_layers = core_env::on("CRISPEMBED_QWEN35_AUDIT_ALL_LAYERS");
    const bool native_positions = native_frontend || (argc == 4 && std::string(argv[3]) == "--native-positions");
    const bool native_vision = argc == 4;
    if (native_vision && !native_positions && std::string(argv[3]) != "--native-vision") return 2;
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
    qwen35_ocr::frontend_tokenizer tokenizer;
    qwen35_ocr::image_inputs image;
    std::vector<int> input_ids, modalities, dimensions;
    qwen35_ocr::position_outputs prefill_positions;
    if (native_vision) {
        const auto pixels = ref.get_f32("input.pixel_values");
        const auto pixel_shape = ref.shape("input.pixel_values");
        const auto grid = ref.get_f32("input.image_grid_thw");
        const auto ids = ref.get_f32("input.input_ids");
        if (!pixels.first || pixel_shape.size() != 2 || pixel_shape[0] != 1536 ||
            pixels.second != size_t(pixel_shape[0] * pixel_shape[1]) || !grid.first || grid.second % 3 || !ids.first)
            return 2;

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
        std::vector<int> native_modalities;
        if (native_frontend) {
            auto * metadata = core_gguf::open_metadata(argv[1]);
            const bool ready = tokenizer.load(metadata);
            // The original OvisOCR3 reference stores the source task text. The
            // fixed-task frontend accepts an empty caller prompt and renders
            // the reviewed template itself; bind the source text before use.
            const std::string task_prompt =
                metadata ? core_gguf::kv_str(metadata, "qwen35_ocr.profile.task_prompt", "") : "";
            const std::string rendered_prompt =
                metadata ? core_gguf::kv_str(metadata, "qwen35_ocr.profile.rendered_prompt", "") : "";
            if (metadata) core_gguf::free_metadata(metadata);
            const auto rgb = ref.get_f32("input.rgb");
            const auto rgb_shape = ref.shape("input.rgb");
            const auto prompt_bytes = ref.get_f32("input.prompt_utf8");
            const auto prompt_length = ref.get_f32("input.prompt_length");
            if (!ready || !rgb.first || rgb_shape.size() != 3 || rgb_shape[0] != 3 ||
                rgb.second != size_t(3 * rgb_shape[1] * rgb_shape[2]) || !prompt_bytes.first || !prompt_length.first ||
                prompt_length.second != 1 || !std::isfinite(prompt_length.first[0]) || prompt_length.first[0] < 0 ||
                std::floor(prompt_length.first[0]) != prompt_length.first[0] ||
                prompt_length.first[0] > prompt_bytes.second)
                return 2;
            std::vector<uint8_t> bytes(rgb.second);
            for (size_t i = 0; i < rgb.second; ++i) {
                if (rgb.first[i] < 0 || rgb.first[i] > 255 || std::floor(rgb.first[i]) != rgb.first[i]) return 2;
                bytes[i] = uint8_t(rgb.first[i]);
            }
            std::string prompt, rendered;
            for (int i = 0; i < int(prompt_length.first[0]); ++i) {
                if (!std::isfinite(prompt_bytes.first[i]) || prompt_bytes.first[i] < 0 || prompt_bytes.first[i] > 255 ||
                    std::floor(prompt_bytes.first[i]) != prompt_bytes.first[i])
                    return 2;
                prompt.push_back(char(prompt_bytes.first[i]));
            }
            const bool fixed_task = tokenizer.is_ovisocr3();
            if (fixed_task && (prompt != task_prompt || rendered_prompt.empty())) return 1;
            std::vector<int> derived_ids;
            if (!qwen35_ocr::preprocess_rgb(bytes.data(), int(rgb_shape[2]), int(rgb_shape[1]),
                                            tokenizer.image_limits(), image) ||
                !tokenizer.image_prompt(fixed_task ? std::string{} : prompt, int(image.patches.size() / 1536 / 4),
                                        derived_ids, native_modalities, rendered))
                return 2;
            if (fixed_task && rendered != rendered_prompt) return 1;
            if (image.grid != dimensions || derived_ids != input_ids) return 1;
            input_ids = std::move(derived_ids);
            dimensions = image.grid;
            std::printf("PASS native raw RGB grid and prompt IDs exact\n");
            if (!qwen35_compare(ref, "input.pixel_values", image.patches)) return 1;
        }
        if (native_positions) {
            const auto types = ref.get_f32("input.mm_token_type_ids");
            if (!types.first || types.second != input_ids.size()) return 2;
            for (size_t i = 0; i < types.second; ++i) {
                if (types.first[i] != 0 && types.first[i] != 1) return 2;
                modalities.push_back(int(types.first[i]));
            }
            if (native_frontend) {
                if (native_modalities != modalities) return 1;
                modalities = std::move(native_modalities);
            }
            if (!qwen35_ocr::image_positions(modalities, dimensions, 2, prefill_positions)) return 2;
        }
        qwen35_ocr::vision_weights visual;
        if (!qwen35_ocr::map_small_vision(loaded.tensors, visual) ||
            !qwen35_ocr::vision_cpu(visual, {}, native_frontend ? image.patches.data() : pixels.first,
                                    int(pixel_shape[1]), dimensions, 4, vision))
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
        if (native_positions) {
            std::vector<float> derived;
            if (phase == "cached") {
                if (!qwen35_ocr::cached_positions(prefill_cache.tokens, tokens, prefill_positions.rope_delta, derived))
                    return 2;
            } else {
                auto types = modalities;
                if (phase == "full_prefix") types.push_back(0);
                qwen35_ocr::position_outputs positions;
                if (!qwen35_ocr::image_positions(types, dimensions, 2, positions)) return 2;
                derived = std::move(positions.axes);
                ok &= qwen35_compare(ref, phase + ".decoder.position_ids", positions.generation);
            }
            ok &= derived == coords;
            std::printf("%s native positions %s delta=%d\n", derived == coords ? "PASS" : "FAIL", phase.c_str(),
                        prefill_positions.rope_delta);
            coords = std::move(derived);
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
        if (native_frontend) {
            for (const auto & entry : current.attention_inputs) {
                const auto name = phase + ".attention." + std::to_string(entry.first) + ".input";
                const auto metric = ref.compare(name, entry.second.data(), entry.second.size());
                std::printf("TRACE %s max_abs=%.9g cos=%.9f norms=%.9g/%.9g\n", name.c_str(), metric.max_abs,
                            metric.cos_min, metric.mine_norm, metric.ref_norm);
            }
            for (const auto & entry : current.attention_keys) {
                const auto name = phase + ".attention." + std::to_string(entry.first) + ".k_norm";
                const auto metric = ref.compare(name, entry.second.data(), entry.second.size());
                std::printf("TRACE %s max_abs=%.9g cos=%.9f norms=%.9g/%.9g\n", name.c_str(), metric.max_abs,
                            metric.cos_min, metric.mine_norm, metric.ref_norm);
            }
        }
        for (size_t i = 0; i < current.hidden.size(); ++i) {
            if (!qwen35_compare(ref, phase + ".decoder.layer." + std::to_string(i), current.hidden[i])) {
                ok = false;
                // The diagnostic arm measures amplification after the earliest failure.
                if (!audit_all_layers) break;
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
    // still come from the official frontend in the baseline arm. The position
    // arm derives coordinates from modalities/grids and its own carried cache.
    const auto expected = ref.get_f32("generated.ids");
    std::vector<int> ids;
    double native_confidence = 0, reference_confidence = 0;
    auto cache = std::move(prefill_cache);
    auto logits = std::move(first_logits);
    if (!expected.first || expected.second == 0 || expected.second > 64 || logits.empty())
        ok = false;
    else
        for (size_t step = 0; step < expected.second; ++step) {
            ok &= qwen35_compare(ref, "generated.scores." + std::to_string(step), logits);
            const int id = argmax(logits);
            if (native_frontend) {
                const auto reference_score = ref.get_f32("generated.scores." + std::to_string(step));
                double total = 0, reference_total = 0;
                for (float value : logits) total += std::exp(double(value) - logits[id]);
                if (!reference_score.first || reference_score.second != logits.size()) return 2;
                const float highest =
                    *std::max_element(reference_score.first, reference_score.first + reference_score.second);
                for (size_t i = 0; i < reference_score.second; ++i)
                    reference_total += std::exp(double(reference_score.first[i]) - highest);
                native_confidence += 1 / total;
                reference_confidence += 1 / reference_total;
            }

            ids.push_back(id);
            if (id != int(expected.first[step])) {
                ok = false;
                break;
            }
            if (native_frontend ? tokenizer.is_stop(id) : id == 248046) {
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
            if (native_positions) {
                std::vector<float> derived;
                if (!qwen35_ocr::cached_positions(cache.tokens, 1, prefill_positions.rope_delta, derived)) return 2;
                ok &= derived == coords;
                std::printf("%s native positions generated.%zu cache=%d\n", derived == coords ? "PASS" : "FAIL",
                            step + 1, cache.tokens);
                coords = std::move(derived);
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
    if (native_frontend) {
        std::string text, oracle;
        const auto bytes = ref.get_f32("input.decoded_utf8");
        if (!bytes.first || !tokenizer.decode(ids, text)) return 2;
        const auto length = ref.get_f32("input.decoded_length");
        size_t count = bytes.second;
        if (length.first) {
            if (length.second != 1 || !std::isfinite(length.first[0]) || length.first[0] < 0 ||
                length.first[0] > bytes.second || std::floor(length.first[0]) != length.first[0])
                return 2;
            count = size_t(length.first[0]);
        }
        for (size_t i = 0; i < count; ++i) oracle.push_back(char(bytes.first[i]));
        ok &= text == oracle;
        std::printf("Native decoded UTF8 hex:");
        for (unsigned char byte : text) std::printf("%02x", byte);
        std::printf("\nNative/reference mean token confidence including EOS: %.9g/%.9g\n",
                    native_confidence / ids.size(), reference_confidence / ids.size());

        std::printf("%s native decoded text exact bytes=%zu: %s\n", text == oracle ? "PASS" : "FAIL", text.size(),
                    text.c_str());
    }
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    std::printf("%s composed 24-layer native decoder and greedy IDs; %s, production integration pending\n",
                ok ? "PASS" : "FAIL",
                native_frontend
                    ? "native raw RGB/BPE/prompt/vision/positions/decoder/text; official arrays only compared"
                : native_positions ? "native vision/text embeddings and positions; official packed pixels/IDs"
                : native_vision    ? "native vision/text embeddings from official packed pixels/IDs/positions"
                                   : "official merged embeddings/positions supplied");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
