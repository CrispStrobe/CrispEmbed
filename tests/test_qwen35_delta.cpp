#include "core/clean_exit.h"
#include "qwen35_delta.h"
#include "crispembed_diff.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", value);
#else
    setenv("CRISPEMBED_QWEN35_OCR", value, 1);
#endif
}

class pause_diff_dump {
    std::string previous;

public:
    pause_diff_dump() {
        if (const char * value = std::getenv("CRISPEMBED_DIFF_DUMP_DIR"); value && *value) {
            previous = value;
#ifdef _WIN32
            _putenv_s("CRISPEMBED_DIFF_DUMP_DIR", "");
#else
            setenv("CRISPEMBED_DIFF_DUMP_DIR", "", 1);
#endif
        }
    }
    ~pause_diff_dump() {
        if (!previous.empty()) {
#ifdef _WIN32
            _putenv_s("CRISPEMBED_DIFF_DUMP_DIR", previous.c_str());
#else
            setenv("CRISPEMBED_DIFF_DUMP_DIR", previous.c_str(), 1);
#endif
        }
    }
};

class scoped_diff_dump {
    std::string previous;

public:
    explicit scoped_diff_dump(const std::string & variant) {
        if (const char * value = std::getenv("CRISPEMBED_DIFF_DUMP_DIR"); value && *value) {
            previous = value;
            const std::string directory = previous + "/" + variant;
            std::filesystem::create_directories(directory);
#ifdef _WIN32
            _putenv_s("CRISPEMBED_DIFF_DUMP_DIR", directory.c_str());
#else
            setenv("CRISPEMBED_DIFF_DUMP_DIR", directory.c_str(), 1);
#endif
        }
    }
    ~scoped_diff_dump() {
        if (!previous.empty()) {
#ifdef _WIN32
            _putenv_s("CRISPEMBED_DIFF_DUMP_DIR", previous.c_str());
#else
            setenv("CRISPEMBED_DIFF_DUMP_DIR", previous.c_str(), 1);
#endif
        }
    }
};

static bool actual_dump_intact(const char * directory, const std::string & stage, const std::vector<float> & actual) {
    const std::string path = std::string(directory) + "/" + stage + ".f32";
    FILE * file = std::fopen(path.c_str(), "rb");
    if (!file) return false;
    std::vector<float> saved(actual.size());
    const size_t count = std::fread(saved.data(), sizeof(float), saved.size(), file);
    const bool exact = count == actual.size() && std::fgetc(file) == EOF &&
                       std::memcmp(saved.data(), actual.data(), actual.size() * sizeof(float)) == 0;
    std::fclose(file);
    return exact;
}

static bool compare(const crispembed_diff::Ref & ref, const std::string & name, const std::vector<float> & values) {
    const auto report = ref.compare(name, values.data(), values.size());
    const bool ok = ref.get_f32(name).second == values.size() && report.n_elem == values.size() && report.found &&
                    report.cos_min >= 0.99999f && report.max_abs <= 0.002f && std::isfinite(report.max_abs) &&
                    std::isfinite(report.cos_min);
    std::printf("%s %s: cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g norms=%.9g/%.9g\n",
                ok ? "PASS" : "FAIL", name.c_str(), report.cos_min, report.cos_global, report.max_abs, report.mean_abs,
                report.mine_norm, report.ref_norm);
    return ok;
}

static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 2) return 2;
    crispembed_diff::Ref ref;
    if (!ref.load(argv[1])) return 2;
    std::vector<float> previous_state, output, state;
    qwen35_ocr::linear_outputs previous_linear;
    std::vector<std::pair<std::string, std::vector<float>>> primary_dumps;
    bool ok = true;
    for (const std::string phase : { "prefill", "cached" }) {
        auto tensor = [&](const char * field) { return ref.get_f32(phase + ".delta." + field); };
        auto shape = ref.shape(phase + ".delta.q");
        if (shape.size() != 4) return 2;
        const int width = int(shape[0]), heads = int(shape[1]), tokens = int(shape[2]), batch = int(shape[3]);
        const auto q = tensor("q"), k = tensor("k"), v = tensor("v"), g = tensor("g"), beta = tensor("beta");
        const size_t elements = size_t(width) * heads * tokens * batch;
        const size_t rows = size_t(heads) * tokens * batch;
        if (!q.first || !k.first || !v.first || !g.first || !beta.first || q.second != elements ||
            k.second != elements || v.second != elements || g.second != rows || beta.second != rows)
            return 2;
        const float * initial = previous_state.empty() ? nullptr : previous_state.data();
        gate("0");
        ok &= !qwen35_ocr::delta_cpu(q.first, k.first, v.first, g.first, beta.first, initial, batch, tokens, heads,
                                     width, 4, output, state);
        ok &= output.empty() && state.empty();
        gate("1");
        if (!qwen35_ocr::delta_cpu(q.first, k.first, v.first, g.first, beta.first, initial, batch, tokens, heads, width,
                                   4, output, state))
            return 1;
        ok &= compare(ref, phase + ".delta.output", output);
        ok &= compare(ref, phase + ".delta.final_state", state);
        primary_dumps.emplace_back(phase + ".delta.output", output);
        primary_dumps.emplace_back(phase + ".delta.final_state", state);
        previous_state = state;
        if (ref.has(phase + ".linear.qkv")) {
            auto field = [&](const std::string & name) { return ref.get_f32(name).first; };
            const auto conv_shape = ref.shape("linear.conv_weights");
            if (conv_shape.size() != 3 || conv_shape[1] != 1 || conv_shape[2] != 3 * heads * width) return 2;
            const int kernel = int(conv_shape[0]);
            if (ref.get_f32(phase + ".linear.qkv").second != 3 * elements ||
                ref.get_f32(phase + ".linear.z").second != elements ||
                ref.get_f32(phase + ".linear.a").second != rows || ref.get_f32(phase + ".linear.b").second != rows ||
                ref.get_f32("linear.conv_weights").second != size_t(3 * heads * width) * kernel ||
                ref.get_f32("linear.a_log").second != size_t(heads) ||
                ref.get_f32("linear.dt_bias").second != size_t(heads) ||
                ref.get_f32("linear.norm_weights").second != size_t(width))
                return 2;
            const qwen35_ocr::linear_inputs input{
                batch,
                tokens,
                heads,
                width,
                kernel,
                4,
                1e-6f,
                field(phase + ".linear.qkv"),
                field(phase + ".linear.z"),
                field(phase + ".linear.a"),
                field(phase + ".linear.b"),
                field("linear.conv_weights"),
                field("linear.a_log"),
                field("linear.dt_bias"),
                field("linear.norm_weights"),
                previous_linear.conv_history.empty() ? nullptr : previous_linear.conv_history.data(),
                previous_linear.state.empty() ? nullptr : previous_linear.state.data(),
            };
            qwen35_ocr::linear_outputs current;
            if (!qwen35_ocr::linear_core_cpu(input, current)) return 1;
            ok &= compare(ref, phase + ".linear.norm", current.values);
            ok &= compare(ref, phase + ".cache.0.conv_states", current.conv_history);
            {
                scoped_diff_dump variant("linear-core");
                ok &= compare(ref, phase + ".delta.final_state", current.state);
            }
            previous_linear = std::move(current);
        }
    }
    // A fresh request must reset to zero state, not inherit the previous context.
    const auto q = ref.get_f32("prefill.delta.q"), k = ref.get_f32("prefill.delta.k"),
               v = ref.get_f32("prefill.delta.v");
    const auto g = ref.get_f32("prefill.delta.g"), beta = ref.get_f32("prefill.delta.beta");
    const auto shape = ref.shape("prefill.delta.q");
    if (!qwen35_ocr::delta_cpu(q.first, k.first, v.first, g.first, beta.first, nullptr, int(shape[3]), int(shape[2]),
                               int(shape[1]), int(shape[0]), 4, output, state))
        return 1;
    {
        scoped_diff_dump variant("reset");
        ok &= compare(ref, "prefill.delta.output", output);
        ok &= compare(ref, "prefill.delta.final_state", state);
    }
    // Prove that the comparator rejects corruption even when cosine is high.
    auto perturbed = output;
    perturbed.front() += 1.0f;
    {
        pause_diff_dump pause;
        const auto bad = ref.compare("prefill.delta.output", perturbed.data(), perturbed.size());
        ok &= bad.found && bad.max_abs > 0.5f;
    }
    if (const char * directory = std::getenv("CRISPEMBED_DIFF_DUMP_DIR"); directory && *directory) {
        bool intact = primary_dumps.size() == 4;
        for (const auto & [stage, values] : primary_dumps) intact &= actual_dump_intact(directory, stage, values);
        intact &= actual_dump_intact((std::string(directory) + "/reset").c_str(), "prefill.delta.output", output);
        intact &= actual_dump_intact((std::string(directory) + "/reset").c_str(), "prefill.delta.final_state", state);
        std::printf("%s DELTA_RAW_DUMP_INTEGRITY stages=4 variants=linear-core,reset\n", intact ? "PASS" : "FAIL");
        ok &= intact;
    }
    ok &= !qwen35_ocr::delta_cpu(nullptr, k.first, v.first, g.first, beta.first, nullptr, 1, 1, 16, 128, 4, output,
                                 state);
    std::printf("%s Qwen3.5 delta recurrence + continuation + reset + disabled/invalid controls\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
