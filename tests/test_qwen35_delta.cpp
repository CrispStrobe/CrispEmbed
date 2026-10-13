#include "qwen35_delta.h"
#include "crispembed_diff.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_QWEN35_OCR", value);
#else
    setenv("CRISPEMBED_QWEN35_OCR", value, 1);
#endif
}

static bool compare(const crispembed_diff::Ref & ref, const std::string & name, const std::vector<float> & values) {
    const auto report = ref.compare(name, values.data(), values.size());
    const bool ok = report.found && report.cos_min >= 0.99999f && report.max_abs <= 0.002f &&
                    std::isfinite(report.max_abs) && std::isfinite(report.cos_min);
    std::printf("%s %s: cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g norms=%.9g/%.9g\n",
                ok ? "PASS" : "FAIL", name.c_str(), report.cos_min, report.cos_global, report.max_abs, report.mean_abs,
                report.mine_norm, report.ref_norm);
    return ok;
}

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    crispembed_diff::Ref ref;
    if (!ref.load(argv[1])) return 2;
    std::vector<float> previous_state, output, state;
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
        previous_state = state;
    }
    // A fresh request must reset to zero state, not inherit the previous context.
    const auto q = ref.get_f32("prefill.delta.q"), k = ref.get_f32("prefill.delta.k"),
               v = ref.get_f32("prefill.delta.v");
    const auto g = ref.get_f32("prefill.delta.g"), beta = ref.get_f32("prefill.delta.beta");
    const auto shape = ref.shape("prefill.delta.q");
    if (!qwen35_ocr::delta_cpu(q.first, k.first, v.first, g.first, beta.first, nullptr, int(shape[3]), int(shape[2]),
                               int(shape[1]), int(shape[0]), 4, output, state))
        return 1;
    ok &= compare(ref, "prefill.delta.output", output);
    ok &= compare(ref, "prefill.delta.final_state", state);
    ok &= !qwen35_ocr::delta_cpu(nullptr, k.first, v.first, g.first, beta.first, nullptr, 1, 1, 16, 128, 4, output,
                                 state);
    std::printf("%s Qwen3.5 delta recurrence + continuation + reset + disabled/invalid controls\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
