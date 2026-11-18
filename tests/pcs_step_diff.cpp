// Compare PCS archives using the existing shared crispembed-diff implementation.
#include "crispembed_diff.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: crispembed-diff-pcs REF.gguf ACTUAL.gguf inventory.txt ABS_BUDGET\n");
        return 2;
    }
    crispembed_diff::Ref reference, actual;
    if (!reference.load(argv[1]) || !actual.load(argv[2])) return 2;
    char * end = nullptr;
    const float budget = strtof(argv[4], &end);
    if (!end || *end || !std::isfinite(budget) || budget <= 0) return 2;
    std::ifstream inventory(argv[3]);
    if (!inventory) return 2;
    int failures = 0, count = 0;
    std::string name;
    std::cout << "stage\tcos_min\tmax_abs\tmean_abs\trms\tactual_norm\tref_norm\tactual_peak\tref_peak\tworst_index\tactual_at_worst\tref_at_worst\tresult\n";
    while (std::getline(inventory, name)) {
        if (name.empty()) continue;
        ++count;
        const auto data = actual.get_f32(name);
        const bool shape = reference.has(name) && actual.has(name) && reference.shape(name) == actual.shape(name);
        if (!shape || !data.first || !data.second) {
            std::cout << name << "\tMISSING_OR_SHAPE_MISMATCH\n";
            ++failures;
            continue;
        }
        const auto expected = reference.get_f32(name);
        const bool finite = std::all_of(data.first, data.first + data.second, [](float x) { return std::isfinite(x); }) &&
                            std::all_of(expected.first, expected.first + expected.second, [](float x) { return std::isfinite(x); });
        const auto result = reference.compare(name, data.first, data.second, 0);
        float actual_peak = 0, ref_peak = 0, worst_error = -1;
        size_t worst = 0;
        for (size_t i = 0; i < data.second; ++i) {
            actual_peak = std::max(actual_peak, std::fabs(data.first[i]));
            ref_peak = std::max(ref_peak, std::fabs(expected.first[i]));
            const float error = std::fabs(data.first[i] - expected.first[i]);
            if (error > worst_error) { worst_error = error; worst = i; }
        }
        // Keep shared cosine policy. Absolute error additionally catches large
        // deviations hidden by cosine; integer input identities require exactness.
        const bool pass = finite && result.is_pass() && std::isfinite(result.max_abs) &&
                          result.max_abs <= (name.rfind("input.", 0) == 0 ? 0.0f : budget);
        std::cout << name << '\t' << std::setprecision(10) << result.cos_min << '\t'
                  << result.max_abs << '\t' << result.mean_abs << '\t' << result.rms << '\t'
                  << result.mine_norm << '\t' << result.ref_norm << '\t' << actual_peak << '\t' << ref_peak << '\t'
                  << worst << '\t' << data.first[worst] << '\t' << expected.first[worst] << '\t'
                  << (pass ? "PASS" : "FAIL") << '\n';
        if (!pass) ++failures;
    }
    std::cerr << "stages=" << count << " failures=" << failures << " absolute_budget=" << budget << '\n';
    return count && !failures ? 0 : 1;
}
