#include "qwen35_linear.h"
#include "core/gguf_loader.h"
#include "crispembed_diff.h"
#include "ggml-cpu.h"

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
    return std::string(name).find("l.layers.0.linear_attn.") == 0;
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
    if (!backend || !core_gguf::load_weights_filtered(argv[1], backend, include, nullptr, "qwen35-linear", loaded))
        return 2;
    auto tensor = [&](const char * suffix) {
        const std::string name = std::string("l.layers.0.linear_attn.") + suffix;
        return core_gguf::try_get(loaded.tensors, name.c_str());
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
    qwen35_ocr::projected_linear_outputs previous;
    bool ok = loaded.tensors.size() == 9;
    for (const std::string phase : { "prefill", "cached" }) {
        const auto hidden = ref.get_f32(phase + ".linear.input");
        const auto shape = ref.shape(phase + ".linear.input");
        if (!hidden.first || shape.size() != 3 || hidden.second != size_t(shape[0] * shape[1] * shape[2])) {
            ok = false;
            break;
        }
        qwen35_ocr::projected_linear_outputs current;
        if (!qwen35_ocr::linear_cpu(weights, hidden.first, int(shape[2]), int(shape[1]), 4,
                                    previous.core.conv_history.empty() ? nullptr : previous.core.conv_history.data(),
                                    previous.core.state.empty() ? nullptr : previous.core.state.data(), current)) {
            ok = false;
            break;
        }
        ok &= compare(ref, phase + ".linear.qkv", current.qkv);
        ok &= compare(ref, phase + ".linear.z", current.z);
        ok &= compare(ref, phase + ".linear.a", current.a);
        ok &= compare(ref, phase + ".linear.b", current.b);
        ok &= compare(ref, phase + ".linear.norm", current.core.values);
        ok &= compare(ref, phase + ".cache.0.conv_states", current.core.conv_history);
        ok &= compare(ref, phase + ".delta.final_state", current.core.state);
        ok &= compare(ref, phase + ".linear.output", current.attention);
        previous = std::move(current);
    }
    core_gguf::free_weights(loaded);
    ggml_backend_free(backend);
    std::printf("%s actual-checkpoint Qwen3.5 first linear-attention block; full OCR remains pending\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
