#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "qwen35_vision.h"
#include "crispembed_diff.h"
#include "qwen35_parity.h"
#include "ggml-cpu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static bool include(const char * name, void *) {
    return std::string(name).find("v.") == 0;
}
static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", value);
#else
    setenv("CRISPEMBED_QWEN35_OCR", value, 1);
#endif
}
class pause_diff_dump {
    std::string previous;
    static void set(const char * value) {
#ifdef _WIN32
        _putenv_s("CRISPEMBED_DIFF_DUMP_DIR", value);
#else
        setenv("CRISPEMBED_DIFF_DUMP_DIR", value, 1);
#endif
    }

public:
    pause_diff_dump() {
        const char * value = std::getenv("CRISPEMBED_DIFF_DUMP_DIR");
        if (value && *value) {
            previous = value;
            set("");
        }
    }
    ~pause_diff_dump() {
        if (!previous.empty()) set(previous.c_str());
    }
};
static bool actual_dump_intact(const char * directory, const char * name, const std::vector<float> & actual) {
    const std::string path = std::string(directory) + "/" + name + ".f32";
    FILE * file = std::fopen(path.c_str(), "rb");
    if (!file) return false;
    std::vector<float> saved(actual.size());
    const size_t count = std::fread(saved.data(), sizeof(float), saved.size(), file);
    const bool exact = count == actual.size() && std::fgetc(file) == EOF &&
                       std::memcmp(saved.data(), actual.data(), actual.size() * sizeof(float)) == 0;
    std::fclose(file);
    return exact;
}
static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 3 && argc != 4) return 2;
    if (argc == 4 && std::string(argv[3]) != "--synthetic") return 2;
    gate("1");
    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 2;
    auto * backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    if (!backend || !core_gguf::load_weights_filtered(argv[1], backend, include, nullptr, "qwen35-vision", loaded))
        return 2;
    qwen35_ocr::vision_weights weights;
    if (!qwen35_ocr::map_small_vision(loaded.tensors, weights)) return 2;
    qwen35_ocr::vision_config config;
    if (argc == 4) {
        config.width = 24;
        config.heads = 3;
        config.intermediate = 48;
        config.output = 16;
        config.patch_width = 12;
        config.position_side = 7;
    }
    const auto patches = ref.get_f32("input.pixel_values");
    const auto shape = ref.shape("input.pixel_values");
    const auto grid = ref.get_f32("input.image_grid_thw");
    if (!patches.first || shape.size() != 2 || shape[0] != config.patch_width ||
        patches.second != size_t(shape[0] * shape[1]) || !grid.first || grid.second % 3)
        return 2;
    std::vector<int> dimensions;
    for (size_t i = 0; i < grid.second; ++i) {
        if (!std::isfinite(grid.first[i]) || grid.first[i] < 1 || grid.first[i] > 1024 ||
            std::floor(grid.first[i]) != grid.first[i])
            return 2;
        dimensions.push_back(int(grid.first[i]));
    }
    qwen35_ocr::vision_outputs output;
    bool ok = loaded.tensors.size() == 153 &&
              qwen35_ocr::vision_cpu(weights, config, patches.first, int(shape[1]), dimensions, 4, output);
    if (!ok) {
        std::printf("FAIL native vision initialization/execution\n");
        return 1;
    }
    ok &= qwen35_compare(ref, "prefill.vision.patch_embed", output.patch);
    ok &= qwen35_compare(ref, "prefill.vision.position", output.position);
    ok &= qwen35_compare(ref, "prefill.vision.embedded", output.embedded);
    ok &= qwen35_compare(ref, "prefill.vision.rotary_cos", output.rotary_cos);
    ok &= qwen35_compare(ref, "prefill.vision.rotary_sin", output.rotary_sin);
    for (size_t i = 0; i < output.hidden.size(); ++i) {
        if (!qwen35_compare(ref, "prefill.vision.layer." + std::to_string(i), output.hidden[i])) {
            ok = false;
            if (!core_env::on("CRISPEMBED_QWEN35_AUDIT_ALL_LAYERS")) break;
        }
    }
    for (size_t i = 0; i < output.probes.size(); ++i) {
        const auto & probe = output.probes[i];
        if (probe.norm1.empty() || !ref.get_f32("prefill.vision.probe." + std::to_string(i) + ".norm1").first) continue;
        const std::string prefix = "prefill.vision.probe." + std::to_string(i) + ".";
        ok &= qwen35_compare(ref, prefix + "norm1", probe.norm1);
        ok &= qwen35_compare(ref, prefix + "qkv", probe.qkv);
        ok &= qwen35_compare(ref, prefix + "pooled", probe.pooled);
        ok &= qwen35_compare(ref, prefix + "attention", probe.attention);
        ok &= qwen35_compare(ref, prefix + "norm2", probe.norm2);
        ok &= qwen35_compare(ref, prefix + "up", probe.up);
        ok &= qwen35_compare(ref, prefix + "activation", probe.activation);
        ok &= qwen35_compare(ref, prefix + "down", probe.down);
    }
    ok &= qwen35_compare(ref, "prefill.vision.merger.norm", output.merger_norm);
    ok &= qwen35_compare(ref, "prefill.vision.merger.up", output.merger_up);
    ok &= qwen35_compare(ref, "prefill.vision.merger.activation", output.merger_activation);
    ok &= qwen35_compare(ref, "prefill.vision.merger", output.merged);
    // Repeated calls reset all temporary geometry/state. Fail-closed controls.
    qwen35_ocr::vision_outputs repeated;
    ok &= qwen35_ocr::vision_cpu(weights, config, patches.first, int(shape[1]), dimensions, 4, repeated);
    ok &= repeated.merged == output.merged;
    gate("0");
    ok &= !qwen35_ocr::vision_cpu(weights, config, patches.first, int(shape[1]), dimensions, 4, repeated);
    gate("1");
    auto invalid = dimensions;
    invalid[1] += 1;
    ok &= !qwen35_ocr::vision_cpu(weights, config, patches.first, int(shape[1]), invalid, 4, repeated);
    invalid = dimensions;
    invalid[0] += 1;
    ok &= !qwen35_ocr::vision_cpu(weights, config, patches.first, int(shape[1]), invalid, 4, repeated);
    auto missing = weights;
    missing.patch = nullptr;
    ok &= !qwen35_ocr::vision_cpu(missing, config, patches.first, int(shape[1]), dimensions, 4, repeated);
    {
        // Deliberately wrong vectors must never replace the real trace files.
        pause_diff_dump pause;
        auto perturbed = output.position;
        for (float & x : perturbed) x += 0.1f;
        ok &= ref.compare("prefill.vision.position", perturbed.data(), perturbed.size()).max_abs > 0.002f;
        const auto oracle = ref.get_f32("prefill.vision.layer.11");
        const auto control = ref.get_f32("backend_off.vision.layer.11");
        if (control.first && oracle.first && control.second == oracle.second) {
            std::vector<float> corrupted(oracle.first, oracle.first + oracle.second);
            size_t worst = 0;
            for (size_t i = 1; i < oracle.second; ++i)
                if (std::abs(control.first[i] - oracle.first[i]) > std::abs(control.first[worst] - oracle.first[worst]))
                    worst = i;
            const float budget = 0.002f + 2.0f * std::abs(control.first[worst] - oracle.first[worst]);
            corrupted[worst] += 2.0f * budget + 0.1f;
            ok &= !qwen35_compare(ref, "prefill.vision.layer.11", corrupted, false);
        }
    }
    if (const char * directory = std::getenv("CRISPEMBED_DIFF_DUMP_DIR"); directory && *directory) {
        bool intact = actual_dump_intact(directory, "prefill.vision.position", output.position);
        if (output.hidden.size() > 11)
            intact &= actual_dump_intact(directory, "prefill.vision.layer.11", output.hidden[11]);
        std::printf("%s VISION_RAW_DUMP_INTEGRITY stages=%d\n", intact ? "PASS" : "FAIL",
                    output.hidden.size() > 11 ? 2 : 1);
        ok &= intact;
    }
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    std::printf("%s native composed Qwen3.5 vision from packed pixels, positional geometry, 12 blocks and merger; "
                "image frontend pending\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
