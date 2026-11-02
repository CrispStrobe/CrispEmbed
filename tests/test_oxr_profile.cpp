#include "oxr_model_profile.h"
#include "core/clean_exit.h"
#include "core/nfc.h"
#include "fixtures/oxr/preprocess_gold.h"
#include <cstdio>
#include <cstring>
#include <numeric>
#include <set>

static int crispembed_test_main() {
    int checks = 0, failed = 0;
    const auto check = [&](bool ok, const char * name) {
        ++checks;
        if (!ok) {
            ++failed;
            std::fprintf(stderr, "FAIL %s\n", name);
        }
    };
    for (const auto & c : oxr_gold::resizes) {
        const auto actual = oxr_profile::resize_rgb_pil_bilinear(c.input.data(), c.w, c.h, c.ch, c.rw, c.rh);
        check(actual == c.expected, "Pillow bilinear golden pixels, including RGBA conversion");
    }
    for (const auto & c : oxr_gold::sizes) {
        const auto actual = oxr_profile::smart_resize(c.h, c.w);
        check(actual.first == c.rh && actual.second == c.rw, "pinned original smart_resize golden size");
    }
    const auto patches = oxr_profile::preprocess(oxr_gold::patch_input.data(), oxr_gold::patch_w, oxr_gold::patch_h,
                                                 oxr_gold::patch_ch, 1024, 1024);
    check(patches.height == oxr_gold::patch_rh && patches.width == oxr_gold::patch_rw, "patch resize dimensions");
    check(patches.patches.size() == oxr_gold::patch_expected_bits.size(), "patch element count");
    bool exact = patches.patches.size() == oxr_gold::patch_expected_bits.size();
    for (size_t i = 0; exact && i < patches.patches.size(); ++i) {
        uint32_t bits;
        std::memcpy(&bits, &patches.patches[i], sizeof(bits));
        exact = bits == oxr_gold::patch_expected_bits[i];
    }
    check(exact, "all 3072 source-normalized and patch-ordered float bits");
    check(oxr_profile::round_even(2.5) == 2 && oxr_profile::round_even(3.5) == 4, "ties-to-even");
    const auto order = oxr_profile::window_groups(8, 10);
    check(order == std::vector<int32_t>({ 0, 1, 2, 5, 6, 7, 10, 11, 12, 3, 4, 8, 9, 13, 14, 15, 16, 17, 18, 19 }),
          "3x3 merged-window permutation");
    check(std::set<int32_t>(order.begin(), order.end()).size() == 20,
          "window permutation complete without duplication");
    std::vector<float> table(32 * 32);
    std::iota(table.begin(), table.end(), 0.0f);
    const auto pos = oxr_profile::position_embeddings(table.data(), 32, 32, 1);
    bool identity = true;
    for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x)
            identity = identity && pos[size_t(((y / 2) * 16 + x / 2) * 4 + (y % 2) * 2 + x % 2)] == float(y * 32 + x);
    check(identity, "learned positions identity resize and merge-major placement");
    check(core_nfc::normalize("e\xcc\x81") == "\xc3\xa9", "NFC normalization");
    std::vector<float> logits{ 3.0f, -3.0f, 2.0f };
    oxr_profile::repetition_penalty(logits, { 0, 1, 0, -300, 100 }, 1.5f);
    check(logits == std::vector<float>({ 2.0f, -4.5f, 2.0f }),
          "penalty applied once, sign-correct, non-token placeholders excluded");
    check(oxr_profile::vllm_penalty_id(-301) == 151652 && oxr_profile::vllm_penalty_id(-302) == 151653 &&
              oxr_profile::vllm_penalty_id(-300) == 151655,
          "vLLM expanded image placeholder history");
    check(oxr_profile::special(151656) && !oxr_profile::special(151657) && !oxr_profile::special(151668),
          "actual added token special flags");
    check(oxr_profile::eos(151643) && oxr_profile::eos(151645) && !oxr_profile::eos(151644), "two EOS IDs");
    bool rejected = false;
    try {
        (void)oxr_profile::smart_resize(0, 32);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "invalid dimensions rejected");
    rejected = false;
    try {
        oxr_profile::repetition_penalty(logits, {}, 0.0f);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "zero repetition penalty rejected");
    std::printf("OXR profile: %d checks, %d failures\n", checks, failed);
    return failed ? 1 : 0;
}
int main() {
    core_util::clean_exit(crispembed_test_main());
}
