#include "lighton4b_vision_bf16.h"
#include "core/clean_exit.h"
#include <cmath>
#include <iostream>
#include <limits>
static int crispembed_test_main() {
    using qwen35_ocr::lighton4b_bf16;
    if (lighton4b_bf16(1.00390625f) != 1.0f || lighton4b_bf16(1.01171875f) != 1.015625f ||
        lighton4b_bf16(-1.00390625f) != -1.0f || lighton4b_bf16(0.0f) != 0.0f || !std::signbit(lighton4b_bf16(-0.0f)) ||
        !std::isinf(lighton4b_bf16(INFINITY)) || !std::isnan(lighton4b_bf16(std::numeric_limits<float>::quiet_NaN())))
        return 1;
    for (uint32_t word = 0; word < 65536; ++word) {
        uint32_t bits = word << 16;
        float source;
        std::memcpy(&source, &bits, 4);
        if (std::isfinite(source) && lighton4b_bf16(source) != source) return 2;
    }
    std::cout << "BF16_ROUND passed=1 finite_bf16_identity=1 ties_even=1 special=1 model_inference=0\n";
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
