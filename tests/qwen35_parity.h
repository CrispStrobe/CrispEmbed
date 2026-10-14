#pragma once
#include "crispembed_diff.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

// Composed Qwen3.5 vision has amplified states for which two official F32 CPU
// convolution backends differ beyond a uniform 0.002 absolute gate. Only vision
// boundaries and the final decoder norm may use the independently captured
// backend variation per named stage (vision), per element (final norm).
// Vocabulary logits, hidden states and caches
// retain the original absolute gate. Reference arrays are test-only artifacts.
static bool qwen35_compare(const crispembed_diff::Ref & ref, const std::string & name,
                           const std::vector<float> & values, bool print = true) {
    const auto report = ref.compare(name, values.data(), values.size());
    const auto oracle = ref.get_f32(name);
    const bool sensitive = name.find(".vision.") != std::string::npos ||
                           (name.size() >= 13 && name.substr(name.size() - 13) == ".decoder.norm");
    std::string control_name;
    if (sensitive && name.find("prefill.") == 0) control_name = "backend_off." + name.substr(8);
    if (sensitive && name.find("full_prefix.") == 0) control_name = "backend_off_full_prefix." + name.substr(12);
    const auto control = ref.get_f32(control_name);
    bool ok = report.found && oracle.first && oracle.second == values.size() && report.n_elem == values.size() &&
              report.cos_min >= 0.99999f && std::isfinite(report.max_abs) && std::isfinite(report.cos_min);
    if (control.first && (control.second != values.size() || ref.shape(control_name) != ref.shape(name))) ok = false;
    const auto shape = ref.shape(name);
    const bool patch_vector = control.first && name.find(".vision.") != std::string::npos;
    const size_t row_width = patch_vector && !shape.empty() ? size_t(shape[0]) : 1;
    if (row_width < 1 || values.size() % row_width) ok = false;
    double stage_variation = 0;
    if (ok && patch_vector) {
        for (size_t i = 0; i < values.size(); ++i)
            stage_variation = std::max(stage_variation, std::abs(double(control.first[i]) - double(oracle.first[i])));
    }
    double budget_ratio = 0, self_max_abs = 0;
    if (ok) {
        for (size_t i = 0; i < values.size(); ++i) {
            if (!std::isfinite(values[i]) || !std::isfinite(oracle.first[i]) ||
                (control.first && !std::isfinite(control.first[i]))) {
                ok = false;
                break;
            }
            const double self = control.first ? std::abs(double(control.first[i]) - double(oracle.first[i])) : 0.0;
            const double delta = std::abs(double(values[i]) - double(oracle.first[i]));
            self_max_abs = std::max(self_max_abs, self);
            const double variation = patch_vector ? stage_variation : self;
            budget_ratio = std::max(budget_ratio, delta / (0.002 + 2.0 * variation));
        }
        ok &= budget_ratio <= 1.0;
    }
    if (print) {
        std::printf("%s %s: cos_min=%.9f cos_global=%.9f max_abs=%.9g mean_abs=%.9g norms=%.9g/%.9g\n",
                    ok ? "PASS" : "FAIL", name.c_str(), report.cos_min, report.cos_global, report.max_abs,
                    report.mean_abs, report.mine_norm, report.ref_norm);
        std::printf("BUDGET %s: strict_absolute_pass=%d calibrated=%d self_max_abs=%.9g max_ratio=%.9g\n", name.c_str(),
                    report.found && report.max_abs <= 0.002f, control.first != nullptr, self_max_abs, budget_ratio);
    }
    return ok;
}
