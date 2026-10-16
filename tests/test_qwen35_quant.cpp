#include "core/clean_exit.h"
#include "qwen35_ocr.h"
#include "qwen35_frontend.h"
#include "qwen35_decoder.h"
#include "qwen35_positions.h"
#include "qwen35_vision.h"
#include "crispembed_diff.h"
#include "qwen35_parity.h"
#include "ggml-cpu.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../ggml/examples/stb_image.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

static void gate(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
static bool strict_prefill = false, strict_scores = false;
static int strict_prefill_failures = 0, strict_score_failures = 0;
static bool measure(const crispembed_diff::Ref & ref, const std::string & name, const std::vector<float> & values) {
    const auto r = ref.compare(name, values.data(), values.size());
    if (!r.found || r.n_elem != values.size() || values.empty() ||
        !std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }) ||
        !std::isfinite(r.cos_min) || !std::isfinite(r.cos_global) || !std::isfinite(r.max_abs))
        return false;
    // Diagnostic measurements, never replace or relax the F16 parity gates.
    std::printf("METRIC %s n=%zu cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g rms=%.9g norms=%.9g/%.9g\n",
                name.c_str(), r.n_elem, r.cos_min, r.cos_global, r.max_abs, r.mean_abs, r.rms, r.mine_norm, r.ref_norm);
    if (strict_prefill && !qwen35_compare(ref, name, values)) ++strict_prefill_failures;
    if (strict_scores && name.find("generated.scores.") == 0 && !qwen35_compare(ref, name, values))
        ++strict_score_failures;
    return true;
}
static int best(const std::vector<float> & scores) {
    return int(std::max_element(scores.begin(), scores.end()) - scores.begin());
}
static bool requested_cap(const char * value, int & cap) {
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (!end || end == value || *end || parsed < 1 || parsed > 1024) return false;
    cap = int(parsed);
    return true;
}
struct diagnostic_mode {
    bool valid = false, reference_embeddings = false, prefill_only = false, strict_scores = false, scores_only = false;
};
static diagnostic_mode decode_mode(const std::string & value) {
    if (value.empty()) return { true, false, false, false, false };
    if (value == "--prefill-only") return { true, false, true, false, false };
    if (value == "--prefill-reference-embeddings") return { true, true, true, false, false };
    if (value == "--full-quality") return { true, false, false, true, false };
    if (value == "--scores-only") return { true, false, false, true, true };
    if (value == "--scores-reference-embeddings") return { true, true, false, true, true };
    return {};
}
static bool probe_measure(const crispembed_diff::Ref & ref, const std::string & name,
                          const std::vector<float> & values) {
    const auto r = ref.compare(name, values.data(), values.size());
    if (!r.found || r.n_elem != values.size() || values.empty() ||
        !std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); }) ||
        !std::isfinite(r.cos_min) || !std::isfinite(r.cos_global) || !std::isfinite(r.max_abs))
        return false;
    std::printf(
        "PROBE_METRIC %s n=%zu cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g rms=%.9g norms=%.9g/%.9g\n",
        name.c_str(), r.n_elem, r.cos_min, r.cos_global, r.max_abs, r.mean_abs, r.rms, r.mine_norm, r.ref_norm);
    qwen35_compare(ref, name, values); // Keep failed numerical records; driver aggregates them separately.
    return true;
}
static int request_controls() {
    int cap = 64;
    if (!requested_cap("1", cap) || cap != 1 || !requested_cap("512", cap) || cap != 512 ||
        !requested_cap("1024", cap) || cap != 1024)
        return 1;
    for (const char * value : { "", "0", "-1", "1025", "1024x", "nan", "99999999999999999999999999999" })
        if (requested_cap(value, cap)) return 1;
    for (const std::string value : { "", "--prefill-only", "--prefill-reference-embeddings", "--full-quality",
                                     "--scores-only", "--scores-reference-embeddings" }) {
        const auto mode = decode_mode(value);
        if (!mode.valid ||
            mode.reference_embeddings !=
                (value == "--prefill-reference-embeddings" || value == "--scores-reference-embeddings") ||
            mode.prefill_only != (value == "--prefill-only" || value == "--prefill-reference-embeddings") ||
            mode.scores_only != (value == "--scores-only" || value == "--scores-reference-embeddings") ||
            mode.strict_scores != (value == "--full-quality" || mode.scores_only))
            return 1;
    }
    for (const std::string value : { "--scores-reference-embeddings=0", "--scores", "--unknown" })
        if (decode_mode(value).valid) return 1;
    std::puts("DECODE_MODE_CONTROLS modes=6 invalid=3 test_only=1");
    std::puts("RECEIPT_CONTROLS cap_min=1 cap_max=1024 invalid=1 default_cap=64");
    return 0;
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--request-controls") return request_controls();
    if (argc < 5 || argc > 7) return 2; // model, image, reference, prompt, optional cap and diagnostic mode
    const auto mode = decode_mode(argc == 7 ? argv[6] : "");
    if (!mode.valid) return 2;
    const bool reference_embeddings = mode.reference_embeddings;
    const bool prefill_only = mode.prefill_only;
    strict_scores = mode.strict_scores;
    strict_prefill = prefill_only;
    int token_cap = 64;
    if (argc >= 6) {
        if (!requested_cap(argv[5], token_cap)) return 2;
    }
    gate("CRISPEMBED_QWEN35_OCR", "1");
    gate("CRISPEMBED_QWEN35_QUANT", "1");
    crispembed_diff::Ref ref;
    if (!ref.load(argv[3])) return 2;
    const auto expected = ref.get_f32("generated.ids");
    const auto expected_bytes = ref.get_f32("input.decoded_utf8");
    const auto expected_length = ref.get_f32("input.decoded_length");
    if (!expected.first || !expected.second || expected.second > size_t(token_cap) || !expected_bytes.first) return 2;
    int w = 0, h = 0, channels = 0;
    std::unique_ptr<unsigned char, decltype(&stbi_image_free)> rgb(stbi_load(argv[2], &w, &h, &channels, 3),
                                                                   stbi_image_free);
    if (!rgb) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights(argv[1], backend, "qwen35-quant", loaded)) return 2;
    bool quantized = false;
    for (const auto & tensor : loaded.tensors) quantized |= ggml_is_quantized(tensor.second->type);
    qwen35_ocr::decoder_weights decoder;
    qwen35_ocr::vision_weights visual;
    qwen35_ocr::frontend_tokenizer tokenizer;
    auto * metadata = core_gguf::open_metadata(argv[1]);
    const bool ready = tokenizer.load(metadata);
    if (metadata) core_gguf::free_metadata(metadata);
    if (!ready || !qwen35_ocr::map_small_decoder(loaded.tensors, decoder) ||
        !qwen35_ocr::map_small_vision(loaded.tensors, visual))
        return 2;
    qwen35_ocr::image_inputs image;
    std::vector<int> ids, modalities;
    std::string rendered;
    if (!qwen35_ocr::preprocess_rgb(rgb.get(), h, w, {}, image) ||
        !tokenizer.image_prompt(argv[4], int(image.patches.size() / 1536 / 4), ids, modalities, rendered))
        return 2;
    const auto official_ids = ref.get_f32("input.input_ids");
    if (!official_ids.first || official_ids.second != ids.size()) return 2;
    for (size_t i = 0; i < ids.size(); ++i)
        if (ids[i] != int(official_ids.first[i])) return 1;
    qwen35_ocr::position_outputs positions;
    qwen35_ocr::vision_outputs vision;
    if (!qwen35_ocr::image_positions(modalities, image.grid, 2, positions) ||
        !qwen35_ocr::vision_cpu(visual, {}, image.patches.data(), int(image.patches.size() / 1536), image.grid, 4,
                                vision))
        return 2;
    bool ok = measure(ref, "input.pixel_values", image.patches);
    ok &= measure(ref, "prefill.vision.merger", vision.merged);
    std::vector<float> hidden;
    if (!qwen35_ocr::embed_cpu(decoder.embedding, ids, hidden)) return 2;
    size_t image_token = 0;
    for (size_t i = 0; i < ids.size(); ++i)
        if (modalities[i] == 1) {
            if ((image_token + 1) * 1024 > vision.merged.size()) return 2;
            std::copy_n(vision.merged.data() + image_token * 1024, 1024, hidden.data() + i * 1024);
            ++image_token;
        }
    if (image_token * 1024 != vision.merged.size()) return 2;
    if (reference_embeddings) {
        // Test-only same-input localization, never a production frontend or
        // composed-model parity claim. Keep the independently measured vision
        // result above, but isolate decoder arithmetic from its input error.
        const auto original = ref.get_f32("prefill.decoder.inputs_embeds");
        if (!original.first || original.second != hidden.size()) return 2;
        hidden.assign(original.first, original.first + original.second);
        std::printf("ISOLATION reference_embeddings=1 test_only=1\n");
    }
    ok &= measure(ref, "prefill.decoder.inputs_embeds", hidden);
    qwen35_ocr::decoder_outputs current;
    if (!qwen35_ocr::decoder_cpu(decoder, {}, hidden.data(), positions.axes.data(), 1, int(ids.size()), 4, nullptr,
                                 current))
        return 2;
    for (size_t i = 0; i < current.hidden.size(); ++i)
        ok &= measure(ref, "prefill.decoder.layer." + std::to_string(i), current.hidden[i]);
    ok &= measure(ref, "prefill.decoder.norm", current.norm);
    ok &= measure(ref, "prefill.logits", current.logits);
    for (size_t i = 0; i < current.cache.layers.size(); ++i) {
        const auto label = "prefill.cache." + std::to_string(i);
        const auto & c = current.cache.layers[i];
        if (decoder.layers[i].full_attention) {
            ok &= measure(ref, label + ".keys", c.attention.keys);
            ok &= measure(ref, label + ".values", c.attention.values);
        } else {
            ok &= measure(ref, label + ".conv_states", c.conv);
            ok &= measure(ref, label + ".recurrent_states", c.recurrent);
        }
    }
    if (prefill_only) {
        std::printf("PREFILL_ONLY finite=%d metrics=77 selected=%d reference=%d strict_failures=%d\n", ok,
                    best(current.logits), int(expected.first[0]), strict_prefill_failures);
        core_gguf::free_weights(loaded);
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    }
    std::vector<int> probe_steps;
    const auto requested_probes = ref.get_f32("diagnostic.decode_probe_steps");
    if (mode.scores_only || (strict_scores && requested_probes.first)) {
        if (!requested_probes.first || requested_probes.second < 1 || requested_probes.second > 4) return 2;
        for (size_t i = 0; i < requested_probes.second; ++i) {
            const float value = requested_probes.first[i];
            if (!std::isfinite(value) || value != std::floor(value) || value <= 0 || value >= expected.second ||
                (!probe_steps.empty() && value <= probe_steps.back()))
                return 2;
            probe_steps.push_back(int(value));
        }
    }
    size_t probe_count = 0;
    // Teacher forcing is confined to this diagnostic arm. It keeps each score
    // comparison on the original Python prefix even after a quantized argmax differs.
    // The independent owning-context call below generates freely from image/prompt.
    for (size_t step = 0; step < expected.second; ++step) {
        ok &= measure(ref, "generated.scores." + std::to_string(step), current.logits);
        if (std::find(probe_steps.begin(), probe_steps.end(), int(step)) != probe_steps.end()) {
            const auto prefix = "generated.probe." + std::to_string(step) + ".";
            auto observe = [&](const std::string & label, const std::vector<float> & values) {
                ++probe_count;
                ok &= probe_measure(ref, prefix + label, values);
            };
            observe("decoder.inputs_embeds", hidden);
            for (size_t i = 0; i < current.hidden.size(); ++i)
                observe("decoder.layer." + std::to_string(i), current.hidden[i]);
            observe("decoder.norm", current.norm);
            observe("logits", current.logits);
            for (size_t i = 0; i < current.cache.layers.size(); ++i) {
                const auto label = "cache." + std::to_string(i);
                const auto & cache = current.cache.layers[i];
                if (decoder.layers[i].full_attention) {
                    observe(label + ".keys", cache.attention.keys);
                    observe(label + ".values", cache.attention.values);
                } else {
                    observe(label + ".conv_states", cache.conv);
                    observe(label + ".recurrent_states", cache.recurrent);
                }
            }
        }
        const int original = int(expected.first[step]), selected = best(current.logits);
        std::printf("TEACHER step=%zu selected=%d reference=%d same=%d\n", step, selected, original,
                    selected == original);
        if (step + 1 == expected.second) break;
        std::vector<float> coords;
        auto cache = std::move(current.cache);
        if (!qwen35_ocr::embed_cpu(decoder.embedding, { original }, hidden) ||
            !qwen35_ocr::cached_positions(cache.tokens, 1, positions.rope_delta, coords) ||
            !qwen35_ocr::decoder_cpu(decoder, {}, hidden.data(), coords.data(), 1, 1, 4, &cache, current))
            return 2;
    }
    if (strict_scores)
        std::printf("TEACHER_QUALITY stages=%zu strict_failures=%d\n", expected.second, strict_score_failures);
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    if (mode.scores_only) {
        if (probe_count != 75 * probe_steps.size()) return 2;
        std::printf(
            "DECODE_ISOLATION finite=%d reference_embeddings=%d test_only=1 probes=%zu scores=%zu free_generation=0\n",
            ok, reference_embeddings, probe_count, expected.second);
        return ok ? 0 : 1;
    }
    gate("CRISPEMBED_QWEN35_QUANT", "0");
    using owner = std::unique_ptr<qwen35_ocr_context, decltype(&qwen35_ocr_free)>;
    owner disabled(qwen35_ocr_init(argv[1], 4), qwen35_ocr_free);
    ok &= quantized ? !disabled : bool(disabled);
    disabled.reset();
    gate("CRISPEMBED_QWEN35_QUANT", "1");
    owner context(qwen35_ocr_init(argv[1], 4), qwen35_ocr_free);
    if (!context || !qwen35_ocr_set_max_tokens(context.get(), token_cap)) return 2;
    int length = 0;
    const auto started = std::chrono::steady_clock::now();
    const char * text = qwen35_ocr_recognize_file(context.get(), argv[2], argv[4], &length);
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    int count = 0;
    const auto * generated = qwen35_ocr_token_ids(context.get(), &count);
    int probability_count = 0;
    const auto * probabilities = qwen35_ocr_confidences(context.get(), &probability_count);
    if (!text || !generated || !probabilities || count < 1 || probability_count != count || length < 0) return 2;
    bool exact_ids = count == int(expected.second);
    for (int i = 0; i < count && i < int(expected.second); ++i) exact_ids &= generated[i] == int(expected.first[i]);
    std::string reference;
    size_t reference_length = expected_bytes.second;
    if (expected_length.first) {
        if (expected_length.second != 1 || !std::isfinite(expected_length.first[0]) || expected_length.first[0] < 0 ||
            expected_length.first[0] > float(expected_bytes.second))
            return 2;
        reference_length = size_t(expected_length.first[0]);
    }
    for (size_t i = 0; i < reference_length; ++i) reference.push_back(char(expected_bytes.first[i]));
    std::printf("FREE ids:");
    for (int i = 0; i < count; ++i) std::printf(" %d", generated[i]);
    std::printf("\nFREE confidences:");
    for (int i = 0; i < count; ++i) {
        if (!std::isfinite(probabilities[i]) || probabilities[i] < 0 || probabilities[i] > 1) return 2;
        std::printf(" %.9g", probabilities[i]);
    }
    std::printf("\nFREE utf8_hex:");
    for (int i = 0; i < length; ++i) std::printf("%02x", static_cast<unsigned char>(text[i]));
    std::printf("\nFREE ids_exact=%d text_exact=%d stop=%d mean_confidence=%.9g bytes=%d\n", exact_ids,
                std::string(text, size_t(length)) == reference, qwen35_ocr_last_stop(context.get()),
                qwen35_ocr_mean_confidence(context.get()), length);
    std::printf("FREE time_ms=%.9g\n", elapsed_ms);
    std::printf("%s finite quantization measurements, native input IDs and literal0 quant gate control\n",
                ok ? "PASS" : "FAIL");
    // Quality differences are recorded, not hidden behind relaxed parity gates.
    return ok ? 0 : 1;
}
int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
