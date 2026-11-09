#include "lighton4b_physical_f16_attention.h"
#include "qwen35_attention.h"
#include "core/clean_exit.h"
#include <cstdlib>
#include <cstdio>
using namespace qwen35_ocr;
static int run() {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", "1");
    _putenv_s("CRISPEMBED_QWEN35_OCR", "0");
#else
    setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", "1", 1);
    setenv("CRISPEMBED_QWEN35_OCR", "0", 1);
#endif
    // Force both actual main-library objects into the link, without a model.
    attention_weights weights{};
    attention_config config;
    attention_outputs output;
    if (attention_cpu(weights, config, nullptr, nullptr, 1, 1, 4, nullptr, output)) return 1;
    if (lighton4b_physical_f16_attention_available() || lighton4b_physical_f16_attention_valid(true) ||
        !lighton4b_physical_f16_attention_valid(false))
        return 2;
    std::vector<float> pooled{ 1 };
    attention_cache input, canonical;
    canonical.tokens = 99;
    if (lighton4b_physical_f16_attention_cpu({}, {}, input, 0, 1, 4, pooled, canonical) || !pooled.empty() ||
        canonical.tokens)
        return 3;
    std::puts("MAIN_SDK_F16_CLOSED_STUB_LINK PASS 5");
    return 0;
}
int main() {
    core_util::clean_exit(run());
}
