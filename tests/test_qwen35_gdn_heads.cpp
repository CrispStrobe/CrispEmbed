#include "core/clean_exit.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "qwen35_linear.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>
#include <vector>

// Synthetic original-module fixtures only. This executable never loads a model,
// admits an OCR profile, or supplies original cache values to the projected lane.
namespace {
constexpr const char * grouped_gate = "CRISPEMBED_QWEN35_GDN_GROUPED_HEADS";
constexpr const char * ocr_gate = "CRISPEMBED_QWEN35_OCR";
constexpr const char * dump_gate = "CRISPEMBED_DIFF_DUMP_DIR";
constexpr int threads = 2;

struct environment {
    std::string name, saved;
    bool present;
    static void set(const char * name, const char * value) {
#ifdef _WIN32
        _putenv_s(name, value ? value : "");
#else
        if (value)
            setenv(name, value, 1);
        else
            unsetenv(name);
#endif
    }
    environment(const char * key, const char * value) : name(key), present(std::getenv(key) != nullptr) {
        if (present) saved = std::getenv(key);
        set(key, value);
    }
    ~environment() { set(name.c_str(), present ? saved.c_str() : nullptr); }
};

struct weight_owner {
    ggml_backend_t backend = ggml_backend_cpu_init();
    core_gguf::WeightLoad loaded;
    ~weight_owner() {
        core_gguf::free_weights(loaded);
        if (backend) ggml_backend_free(backend);
    }
};

bool control(const std::string & name, bool pass) {
    std::printf("CONTROL name=%s pass=%d\n", name.c_str(), pass ? 1 : 0);
    return pass;
}

bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}
bool same(const qwen35_ocr::linear_outputs & a, const qwen35_ocr::linear_outputs & b) {
    return same(a.values, b.values) && same(a.conv_history, b.conv_history) && same(a.state, b.state);
}
bool same(const qwen35_ocr::projected_linear_outputs & a, const qwen35_ocr::projected_linear_outputs & b) {
    return same(a.qkv, b.qkv) && same(a.z, b.z) && same(a.a, b.a) && same(a.b, b.b) && same(a.attention, b.attention) &&
           same(a.core, b.core);
}
bool empty(const qwen35_ocr::linear_outputs & value) {
    return value.values.empty() && value.conv_history.empty() && value.state.empty();
}
bool empty(const qwen35_ocr::projected_linear_outputs & value) {
    return value.qkv.empty() && value.z.empty() && value.a.empty() && value.b.empty() && value.attention.empty() &&
           empty(value.core);
}
const float * optional_data(const std::vector<float> & value) {
    return value.empty() ? nullptr : value.data();
}

bool compare(const crispembed_diff::Ref & ref, const std::string & name, const std::vector<float> & values) {
    const auto report = ref.compare(name, values.data(), values.size());
    const bool finite = std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
    const bool ok = finite && report.found && !values.empty() && ref.get_f32(name).second == values.size() &&
                    report.n_elem == values.size() && report.cos_min >= 0.99999f && report.max_abs <= 0.002f &&
                    std::isfinite(report.max_abs) && std::isfinite(report.cos_min) && std::isfinite(report.mean_abs);
    std::printf("%s %s: cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g norms=%.9g/%.9g\n",
                ok ? "PASS" : "FAIL", name.c_str(), report.cos_min, report.cos_global, report.max_abs, report.mean_abs,
                report.mine_norm, report.ref_norm);
    return ok;
}

const float * require(const crispembed_diff::Ref & ref, const std::string & name, const std::vector<int64_t> & shape,
                      bool & ok) {
    size_t count = 1;
    for (const auto size : shape) count *= size_t(size);
    const auto value = ref.get_f32(name);
    const bool valid =
        value.first && value.second == count && ref.shape(name) == shape &&
        std::all_of(value.first, value.first + value.second, [](float item) { return std::isfinite(item); });
    if (!valid) std::fprintf(stderr, "Invalid original fixture field: %s\n", name.c_str());
    ok &= valid;
    return valid ? value.first : nullptr;
}

struct phase_input {
    std::string name;
    int batch = 0, tokens = 0;
    const float * hidden = nullptr;
    qwen35_ocr::linear_inputs core{};
    const float * q = nullptr;
    const float * k = nullptr;
    const float * v = nullptr;
    const float * g = nullptr;
    const float * beta = nullptr;
};

bool dump_matches(const std::string & path, const std::vector<float> & expected) {
    FILE * file = std::fopen(path.c_str(), "rb");
    if (!file) return false;
    std::vector<float> value(expected.size());
    const bool read = std::fread(value.data(), sizeof(float), value.size(), file) == value.size();
    const bool at_end = std::fgetc(file) == EOF;
    std::fclose(file);
    return read && at_end && same(value, expected);
}
} // namespace

static int crispembed_test_main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "Usage: %s weights.gguf reference.gguf key_heads (requires %s)\n", argv[0], dump_gate);
        return 2;
    }
    char * end = nullptr;
    const long parsed = std::strtol(argv[3], &end, 10);
    if (!end || end == argv[3] || *end || parsed < 1 || parsed > 128) return 2;
    const int keys = int(parsed);
    const char * dump_value = std::getenv(dump_gate);
    if (!dump_value || !*dump_value) return 2;
    const std::string dump_root(dump_value), core_dump = dump_root + "/core", delta_dump = dump_root + "/delta";
    for (const auto & directory : { dump_root, core_dump, delta_dump }) {
        std::error_code error;
        std::filesystem::create_directories(directory, error);
        if (error) return 2;
    }
    environment enable_ocr(ocr_gate, "1"), enable_grouped(grouped_gate, "1");
    crispembed_diff::Ref ref;
    weight_owner owner;
    if (!ref.load(argv[2]) || !owner.backend ||
        !core_gguf::load_weights(argv[1], owner.backend, "qwen35-gdn-heads", owner.loaded))
        return 2;
    const auto tensor = [&](const char * suffix) {
        return core_gguf::try_get(owner.loaded.tensors, (std::string("l.layers.0.linear_attn.") + suffix).c_str());
    };
    const qwen35_ocr::linear_weights weights{
        tensor("in_proj_qkv.weight"),
        tensor("in_proj_z.weight"),
        tensor("in_proj_a.weight"),
        tensor("in_proj_b.weight"),
        tensor("out_proj.weight"),
        tensor("conv1d.weight"),
        tensor("A_log"),
        tensor("dt_bias"),
        tensor("norm.weight"),
    };
    bool source_ok = owner.loaded.tensors.size() == 9 && ref.tensor_names().size() == 68;
    for (auto * weight : { weights.qkv, weights.z, weights.a, weights.b, weights.out, weights.conv, weights.a_log,
                           weights.dt_bias, weights.norm })
        source_ok &= weight && weight->type == GGML_TYPE_F32 && ggml_is_contiguous(weight);
    if (!source_ok) {
        control("source_contract", false);
        return 2;
    }
    const auto heads64 = weights.a_log->ne[0], width64 = weights.norm->ne[0], hidden64 = weights.qkv->ne[0];
    const auto kernel64 = weights.conv->ne[0];
    if (heads64 < keys || heads64 > 128 || heads64 % keys || width64 != 128 || hidden64 != 16 || kernel64 != 4) {
        control("source_contract", false);
        return 2;
    }
    const int heads = int(heads64), width = int(width64), hidden = int(hidden64), kernel = int(kernel64);
    const int channels = (2 * keys + heads) * width, group = heads * width;
    const auto shape = [](const ggml_tensor * weight, int64_t n0, int64_t n1 = 1, int64_t n2 = 1) {
        return weight->ne[0] == n0 && weight->ne[1] == n1 && weight->ne[2] == n2 && weight->ne[3] == 1;
    };
    source_ok &= shape(weights.qkv, hidden, channels) && shape(weights.z, hidden, group) &&
                 shape(weights.a, hidden, heads) && shape(weights.b, hidden, heads) &&
                 shape(weights.out, group, hidden) && shape(weights.conv, kernel, 1, channels) &&
                 shape(weights.a_log, heads) && shape(weights.dt_bias, heads) && shape(weights.norm, width);
    const auto * conv = require(ref, "linear.conv_weights", { kernel, 1, channels }, source_ok);
    const auto * a_log = require(ref, "linear.a_log", { heads }, source_ok);
    const auto * dt_bias = require(ref, "linear.dt_bias", { heads }, source_ok);
    const auto * norm = require(ref, "linear.norm_weights", { width }, source_ok);
    std::vector<phase_input> inputs;
    for (const std::string name : { "prefill", "cached0", "cached1", "full_prefix" }) {
        const std::string linear = name + ".linear.0.", cache = name + ".cache.0.", delta = name + ".delta.";
        const auto dims = ref.shape(linear + "input");
        if (dims.size() != 3 || dims[0] != hidden || dims[1] < 1 || dims[1] > 4096 || dims[2] < 1 || dims[2] > 4) {
            source_ok = false;
            break;
        }
        phase_input input;
        input.name = name;
        input.tokens = int(dims[1]);
        input.batch = int(dims[2]);
        const int t = input.tokens, b = input.batch;
        input.hidden = require(ref, linear + "input", { hidden, t, b }, source_ok);
        input.core = { b,
                       t,
                       heads,
                       width,
                       kernel,
                       threads,
                       1e-6f,
                       require(ref, linear + "qkv", { channels, t, b }, source_ok),
                       require(ref, linear + "z", { group, t, b }, source_ok),
                       require(ref, linear + "a", { heads, t, b }, source_ok),
                       require(ref, linear + "b", { heads, t, b }, source_ok),
                       conv,
                       a_log,
                       dt_bias,
                       norm,
                       nullptr,
                       nullptr };
        require(ref, linear + "norm", { width, heads, t, b }, source_ok);
        require(ref, linear + "output", { hidden, t, b }, source_ok);
        require(ref, cache + "conv_states", { kernel, channels, b }, source_ok);
        require(ref, cache + "recurrent_states", { width, width, heads, b }, source_ok);
        input.q = require(ref, delta + "q", { width, heads, t, b }, source_ok);
        input.k = require(ref, delta + "k", { width, heads, t, b }, source_ok);
        input.v = require(ref, delta + "v", { width, heads, t, b }, source_ok);
        input.g = require(ref, delta + "g", { heads, t, b }, source_ok);
        input.beta = require(ref, delta + "beta", { heads, t, b }, source_ok);
        require(ref, delta + "output", { width, heads, t, b }, source_ok);
        require(ref, delta + "final_state", { width, width, heads, b }, source_ok);
        inputs.push_back(input);
    }
    if (inputs.size() == 4) {
        source_ok &= inputs[1].tokens == 1 && inputs[2].tokens == 1 && inputs[3].tokens == inputs[0].tokens + 2;
        for (const auto & input : inputs) source_ok &= input.batch == inputs[0].batch;
    } else
        source_ok = false;
    if (!control("source_contract", source_ok)) return 2;
    // A cleared vector may retain a nonnull allocation. Unqualified data(v)
    // would select std::data(v) by ADL instead of a const-reference helper.
    std::vector<float> cleared_state(128, 1.0f);
    cleared_state.clear();
    bool ok = control("empty_cache_is_null", optional_data(cleared_state) == nullptr);

    const auto & first = inputs.front();
    const auto raw_rejects = [&](const qwen35_ocr::linear_inputs & input, int key_heads) {
        qwen35_ocr::linear_outputs out{ { 1 }, { 1 }, { 1 } };
        return !qwen35_ocr::linear_grouped_core_cpu(input, key_heads, out) && empty(out);
    };
    const auto projected_rejects = [&](const qwen35_ocr::linear_weights & w, const float * values, int tokens,
                                       int key_heads) {
        qwen35_ocr::projected_linear_outputs out{ { 1 }, { 1 }, { 1 }, { 1 }, { 1 }, { { 1 }, { 1 }, { 1 } } };
        return !qwen35_ocr::linear_grouped_cpu(w, key_heads, values, first.batch, tokens, threads, nullptr, nullptr,
                                               out) &&
               empty(out);
    };
    const auto both_reject = [&]() {
        const bool raw = raw_rejects(first.core, keys);
        const bool projected = projected_rejects(weights, first.hidden, first.tokens, keys);
        return raw && projected;
    };
    {
        environment gate(grouped_gate, nullptr);
        ok &= control("gate_group_unset", both_reject());
    }
    {
        environment gate(grouped_gate, "");
        ok &= control("gate_group_empty", both_reject());
    }
    {
        environment gate(grouped_gate, "0");
        ok &= control("gate_group_zero", both_reject());
    }
    {
        environment gate(ocr_gate, "0");
        ok &= control("gate_ocr_zero", both_reject());
    }
    ok &= control("reject_key_zero",
                  raw_rejects(first.core, 0) & projected_rejects(weights, first.hidden, first.tokens, 0));
    ok &= control("reject_key_nondivisor", raw_rejects(first.core, heads - 1) &
                                               projected_rejects(weights, first.hidden, first.tokens, heads - 1));
    auto bad = first.core;
    bad.tokens = 4097;
    ok &= control("reject_tokens_over_limit",
                  raw_rejects(bad, keys) & projected_rejects(weights, first.hidden, 4097, keys));
    bad = first.core;
    bad.width = 257;
    ok &= control("reject_width_over_limit", raw_rejects(bad, keys));
    bad = first.core;
    bad.qkv = nullptr;
    ok &= control("reject_missing_qkv", raw_rejects(bad, keys));
    bad = first.core;
    bad.norm_eps = 0;
    ok &= control("reject_bad_norm_epsilon", raw_rejects(bad, keys));
    std::vector<float> invalid_norm(norm, norm + width);
    invalid_norm[0] = std::numeric_limits<float>::quiet_NaN();
    bad = first.core;
    bad.norm_weights = invalid_norm.data();
    ok &= control("reject_nonfinite_norm", raw_rejects(bad, keys));
    ok &= control("reject_missing_hidden", projected_rejects(weights, nullptr, first.tokens, keys));
    std::vector<float> invalid_hidden(first.hidden, first.hidden + size_t(first.batch) * first.tokens * hidden);
    invalid_hidden[0] = std::numeric_limits<float>::quiet_NaN();
    ok &= control("reject_nonfinite_hidden", projected_rejects(weights, invalid_hidden.data(), first.tokens, keys));
    ggml_tensor invalid_weight = *weights.qkv;
    invalid_weight.type = GGML_TYPE_Q8_0;
    auto invalid_weights = weights;
    invalid_weights.qkv = &invalid_weight;
    ok &= control("reject_quantized_weight", projected_rejects(invalid_weights, first.hidden, first.tokens, keys));
    if (keys != heads) {
        for (const char * gate_value : { "0", "1" }) {
            environment gate(grouped_gate, gate_value);
            qwen35_ocr::projected_linear_outputs out;
            ok &= control(std::string("legacy_rejects_grouped_") + (*gate_value == '0' ? "zero" : "one"),
                          !qwen35_ocr::linear_cpu(weights, first.hidden, first.batch, first.tokens, threads, nullptr,
                                                  nullptr, out));
        }
    }

    qwen35_ocr::projected_linear_outputs previous, first_projected;
    qwen35_ocr::linear_outputs previous_core, first_core;
    std::vector<float> previous_delta;
    for (const auto & input : inputs) {
        if (input.name == "prefill" || input.name == "full_prefix") {
            previous = {};
            previous_core = {};
            previous_delta.clear();
        }
        qwen35_ocr::projected_linear_outputs current;
        qwen35_ocr::linear_outputs current_core;
        std::vector<float> delta_output, delta_state;
        auto raw = input.core;
        raw.conv_history = optional_data(previous_core.conv_history);
        raw.state = optional_data(previous_core.state);
        const bool projected_ok =
            qwen35_ocr::linear_grouped_cpu(weights, keys, input.hidden, input.batch, input.tokens, threads,
                                           optional_data(previous.core.conv_history), optional_data(previous.core.state), current);
        const bool raw_ok = qwen35_ocr::linear_grouped_core_cpu(raw, keys, current_core);
        const bool delta_ok =
            qwen35_ocr::delta_cpu(input.q, input.k, input.v, input.g, input.beta, optional_data(previous_delta), input.batch,
                                  input.tokens, heads, width, threads, delta_output, delta_state);
        if (!control("execute_" + input.name, projected_ok && raw_ok && delta_ok)) return 1;
        if (keys == heads) {
            environment gate(grouped_gate, "0");
            qwen35_ocr::projected_linear_outputs legacy;
            qwen35_ocr::linear_outputs legacy_core;
            const bool legacy_ok =
                qwen35_ocr::linear_cpu(weights, input.hidden, input.batch, input.tokens, threads,
                                       optional_data(previous.core.conv_history), optional_data(previous.core.state), legacy);
            const bool legacy_core_ok = qwen35_ocr::linear_core_cpu(raw, legacy_core);
            ok &= control("equal_projected_" + input.name, legacy_ok && same(current, legacy));
            ok &= control("equal_core_" + input.name, legacy_core_ok && same(current_core, legacy_core));
        }
        const std::string linear = input.name + ".linear.0.", cache = input.name + ".cache.0.";
        ok &= compare(ref, linear + "qkv", current.qkv);
        ok &= compare(ref, linear + "z", current.z);
        ok &= compare(ref, linear + "a", current.a);
        ok &= compare(ref, linear + "b", current.b);
        ok &= compare(ref, linear + "norm", current.core.values);
        ok &= compare(ref, linear + "output", current.attention);
        ok &= compare(ref, cache + "conv_states", current.core.conv_history);
        ok &= compare(ref, cache + "recurrent_states", current.core.state);
        {
            environment dump(dump_gate, core_dump.c_str());
            ok &= compare(ref, linear + "norm", current_core.values);
            ok &= compare(ref, cache + "conv_states", current_core.conv_history);
            ok &= compare(ref, cache + "recurrent_states", current_core.state);
        }
        {
            environment dump(dump_gate, delta_dump.c_str());
            ok &= compare(ref, input.name + ".delta.output", delta_output);
            ok &= compare(ref, input.name + ".delta.final_state", delta_state);
        }
        if (input.name == "prefill") {
            first_projected = current;
            first_core = current_core;
        }
        previous = std::move(current);
        previous_core = std::move(current_core);
        previous_delta = std::move(delta_state);
    }
    qwen35_ocr::projected_linear_outputs reset;
    qwen35_ocr::linear_outputs reset_core;
    const bool reset_ok = qwen35_ocr::linear_grouped_cpu(weights, keys, first.hidden, first.batch, first.tokens,
                                                         threads, nullptr, nullptr, reset);
    ok &= control("reset_projected", reset_ok && same(reset, first_projected));
    const bool reset_core_ok = qwen35_ocr::linear_grouped_core_cpu(first.core, keys, reset_core);
    ok &= control("reset_core", reset_core_ok && same(reset_core, first_core));
    auto corrupted = first_projected.attention;
    corrupted[0] += 1.0f;
    {
        // A diagnostic must not replace the actual projected output artifact.
        environment pause_dump(dump_gate, "");
        const auto report = ref.compare("prefill.linear.0.output", corrupted.data(), corrupted.size());
        ok &= control("comparator_rejects_corruption", report.found && report.max_abs > 0.5f);
    }
    ok &=
        control("dump_preserved", dump_matches(dump_root + "/prefill.linear.0.output.f32", first_projected.attention));
    std::printf("%s synthetic original-module GDN component: K=%d V=%d D=%d; model qualification remains pending\n",
                ok ? "PASS" : "FAIL", keys, heads, width);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
