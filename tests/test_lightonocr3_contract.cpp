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
    return failures ? 1 : 0;
}
int main() {
    core_util::clean_exit(crispembed_test_main());
}
