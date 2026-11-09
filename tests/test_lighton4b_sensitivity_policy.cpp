#include "lighton4b_sensitivity_policy.h"
#include <set>
#include <iostream>
int main() {
    qwen35_ocr::decoder_weights w;
    w.layers.resize(32);
    for (int i = 0; i < 32; ++i) w.layers[i].full_attention = i % 4 == 3;
    auto slots = sensitivity::slots(w);
    if (slots.size() != 248) return 1;
    std::set<std::string> names;
    for (auto & s : slots)
        if (!names.insert(s.name).second) return 2;
    int expected[] = { 0, 152, 32, 96, 248 };
    int index = 0;
    for (auto g : { sensitivity::group::q4, sensitivity::group::mixers, sensitivity::group::down,
                    sensitivity::group::ffn, sensitivity::group::q8 }) {
        int count = 0;
        for (auto & s : slots) count += sensitivity::selected(g, s.role);
        if (count != expected[index++]) return 3;
    }
    w.layers.pop_back();
    if (!sensitivity::slots(w).empty()) return 4;
    std::cout << "SENSITIVITY_POLICY passed=1 unique_slots=248 counts=0,152,32,96,248 heldout=0 model_payload=0\n";
}
