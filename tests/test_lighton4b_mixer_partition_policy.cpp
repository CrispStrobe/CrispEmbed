#include "lighton4b_mixer_partition_policy.h"
#include "core/clean_exit.h"
#include <set>
#include <iostream>
static int crispembed_test_main() {
    qwen35_ocr::decoder_weights w;
    w.layers.resize(32);
    for (int i = 0; i < 32; ++i) w.layers[i].full_attention = i % 4 == 3;
    auto slots = sensitivity::slots(w);
    if (slots.size() != 248) return 1;
    std::set<std::string> names;
    for (auto & s : slots)
        if (!names.insert(s.name).second) return 2;
    int expected[] = { 0, 120, 32, 152, 184, 248 };
    int index = 0;
    for (auto g : { sensitivity::group::q4, sensitivity::group::linear, sensitivity::group::attention,
                    sensitivity::group::mixers, sensitivity::group::mixers_down, sensitivity::group::q8 }) {
        int count = 0;
        for (auto & s : slots) count += sensitivity::selected(g, s.role);
        if (count != expected[index++]) return 3;
    }
    for (const auto & slot : slots) {
        const bool linear = sensitivity::selected(sensitivity::group::linear, slot.role);
        const bool attention = sensitivity::selected(sensitivity::group::attention, slot.role);
        if (linear && attention) return 5;
        if ((linear || attention) != sensitivity::selected(sensitivity::group::mixers, slot.role)) return 6;
        if (sensitivity::selected(sensitivity::group::mixers_down, slot.role) !=
            (linear || attention || slot.role == 2))
            return 7;
    }
    w.layers.pop_back();
    if (!sensitivity::slots(w).empty()) return 4;
    std::cout
        << "MIXER_PARTITION_POLICY passed=1 unique_slots=248 counts=0,120,32,152,184,248 heldout=0 model_payload=0\n";
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
