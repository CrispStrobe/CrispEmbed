// Weight-free frontend/decoding edge cases anchored in the upstream contract.
#include "mineru_model_profile.h"
#include "core/clean_exit.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int crispembed_test_main() {
    using namespace mineru25_profile;
    bool ok = round_factor_even(70) == 56 && round_factor_even(98) == 112;
    int h = 0, w = 0;
    ok = ok && smart_resize(70, 784, 50176, 1605632, h, w) && h == 84 && w == 756;
    ok = ok && !smart_resize(1, 201, 50176, 1605632, h, w);
    ok = ok && !smart_resize(0, 28, 50176, 1605632, h, w);

    image_preproc::config cfg;
    cfg.patch_size = 14;
    cfg.temporal_patch_size = cfg.merge_size = 2;
    cfg.min_pixels = 50176;
    cfg.max_pixels = 1605632;
    for (int c = 0; c < 3; ++c) {
        cfg.mean[c] = image_preproc::kCLIPMean[c];
        cfg.std[c] = image_preproc::kCLIPStd[c];
    }
    std::vector<uint8_t> rgb(224 * 224 * 3), rgba(224 * 224 * 4);
    for (size_t i = 0; i < 224 * 224; ++i) {
        for (size_t c = 0; c < 3; ++c) rgb[i * 3 + c] = rgba[i * 4 + c] = uint8_t((i * 7 + c * 31) % 256);
        rgba[i * 4 + 3] = uint8_t(i % 256);
    }
    image_preproc::result a, b;
    ok = ok && mineru25_profile::preprocess_rgb(rgb.data(), 224, 224, 3, cfg, a) &&
         mineru25_profile::preprocess_rgb(rgba.data(), 224, 224, 4, cfg, b) && a.patches == b.patches;
    ok = ok && a.n_patches == 256 && a.row_dim == 1176 && a.grid_thw[1] == 16 && a.grid_thw[2] == 16;
    // First patch: channel0 temporal frames are duplicated, before channel1.
    for (int i = 0; ok && i < 14 * 14; ++i) ok = a.patches[i] == a.patches[14 * 14 + i];
    const float expected = (float(rgb[0]) - cfg.mean[0] * 255.0f) / (cfg.std[0] * 255.0f);
    ok = ok && a.patches[0] == expected;

    float logits[4] = { 0, 1, 3, 2 };
    std::vector<int32_t> history(100, 2);
    auto choice = greedy_choice(logits, 4, history);
    ok = ok && choice.id == 3 && choice.probability > 0.5f && choice.probability < 1.0f;
    history.resize(99);
    ok = ok && greedy_choice(logits, 4, history).id == 2;
    ok = ok && is_eos(151643) && is_eos(151645) && !is_eos(151644);
    ok = ok && replace_invalid_utf8("a\xe2\x82") == "a\xef\xbf\xbd";
    ok = ok && replace_invalid_utf8("\xed\xa0\x80") == "\xef\xbf\xbd\xef\xbf\xbd\xef\xbf\xbd";
    std::puts(ok ? "PASS MinerU frontend/decoding contract" : "FAIL MinerU frontend/decoding contract");
    return ok ? 0 : 1;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
