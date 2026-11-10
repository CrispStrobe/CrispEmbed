#include "lighton4b_gdn_projection_policy.h"
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
    int expected[] = { 0, 24, 24, 48, 24, 120, 184, 248 };
    int index = 0;
    for (auto g : { sensitivity::group::q4, sensitivity::group::qkv, sensitivity::group::z, sensitivity::group::ab,
                    sensitivity::group::out, sensitivity::group::linear, sensitivity::group::mixers_down,
                    sensitivity::group::q8 }) {
        int count = 0;
        for (auto & s : slots) count += sensitivity::selected(g, s.role);
        if (count != expected[index++]) return 3;
    }
    for (const auto & slot : slots) {
        int partitions = 0;
        for (auto g :
             { sensitivity::group::qkv, sensitivity::group::z, sensitivity::group::ab, sensitivity::group::out })
            partitions += sensitivity::selected(g, slot.role);
        const bool linear = sensitivity::selected(sensitivity::group::linear, slot.role);
        if (partitions != int(linear)) return 5;
        if (sensitivity::selected(sensitivity::group::mixers_down, slot.role) !=
            (linear || slot.role == 3 || slot.role == 2))
            return 6;
    }
    w.layers.pop_back();
    if (!sensitivity::slots(w).empty()) return 4;
    std::cout << "GDN_PROJECTION_POLICY passed=1 unique_slots=248 counts=0,24,24,48,24,120,184,248 heldout=0 "
                 "model_payload=0\n";
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
