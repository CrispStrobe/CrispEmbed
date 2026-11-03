#include "lightonocr3_profile.h"
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include <cmath>
#include <cstdio>
#include <vector>

static int crispembed_test_main() {
    int failures = 0;
    auto check = [&](bool result, const char * message) {
        if (!result) {
            ++failures;
            fprintf(stderr, "%s\n", message);
        }
    };
    // Official floor/ceil policy: odd patch grids must not become even grids.
    check(lightonocr3::image_size(317, 452) == std::make_pair(322, 462), "odd patch grid changed");
    check(lightonocr3::image_size(3080, 1540) == std::make_pair(1540, 770), "aspect downscale changed");
    check(lightonocr3::image_size(28, 28) == std::make_pair(28, 28), "aligned input changed");
    bool rejected = false;
    try {
        lightonocr3::image_size(0, 28);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "zero image accepted");
    rejected = false;
    try {
        lightonocr3::image_size(1, 100);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "empty merge accepted");
    std::vector<uint8_t> rgb(28 * 28 * 3, 255);
    int ph, pw;
    const auto normalized = lightonocr3::preprocess(rgb.data(), 28, 28, ph, pw);
    check(ph == 2 && pw == 2 && normalized.size() == rgb.size(), "processor shape mismatch");
    check(std::abs(normalized[0] - 1.9303362f) < 1e-6f, "white/red normalization drift");
    check(std::abs(normalized[784] - 2.0748837f) < 1e-6f, "white/green normalization drift");
    check(std::abs(normalized[1568] - 2.145897f) < 1e-6f, "white/blue normalization drift");
    std::vector<int32_t> prefix{ 151644, 8948, 151645, 198, 151644, 872, 198 };
    const std::vector<int32_t> suffix{ 151645, 198, 151644, 77091, 198 };
    const std::vector<int32_t> grounding{ 1951, 287, 198, 151645, 198, 151644, 77091, 198 };
    const std::vector<int32_t> stops{ 151645, 151643 };
    std::vector<int32_t> special;
    for (int id = 151643; id <= 151656; ++id) special.push_back(id);
    const char * model = "lightonai/LightOnOCR-3-1B";
    const char * revision = "23d544085c886bbdace647bcfab91203299417e2";
    check(lightonocr3::token_contract_matches(model, revision, prefix, suffix, prefix, grounding, stops, special),
          "pinned token contract rejected");
    check(!lightonocr3::token_contract_matches("different/repo", revision, prefix, suffix, prefix, grounding, stops,
                                               special),
          "substituted model repo accepted");
    auto changed = prefix;
    changed[1] += 1;
    check(!lightonocr3::token_contract_matches(model, revision, changed, suffix, prefix, grounding, stops, special),
          "in-range but substituted prompt accepted");
    special.pop_back();
    check(!lightonocr3::token_contract_matches(model, revision, prefix, suffix, prefix, grounding, stops, special),
          "incomplete special-ID set accepted");
    return failures ? 1 : 0;
}
int main() {
    core_util::clean_exit(crispembed_test_main());
}
