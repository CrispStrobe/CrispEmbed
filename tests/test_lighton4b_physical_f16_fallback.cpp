#include "lighton4b_physical_f16_attention.h"
#include "qwen35_attention.h"
#include <cstdlib>
#include <cstdio>
using namespace qwen35_ocr;
int main() {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", "1");
#else
    setenv("CRISPEMBED_LIGHTON4B_PHYSICAL_F16_ATTENTION", "1", 1);
#endif
    if (lighton4b_physical_f16_attention_available() || lighton4b_physical_f16_attention_valid(true) ||
        !lighton4b_physical_f16_attention_valid(false) || lighton4b_physical_f16_attention_mode(false))
        return 1;
    std::vector<float> result{ 1 };
    attention_cache cache, output;
    output.tokens = 99;
    if (lighton4b_physical_f16_attention_cpu({}, {}, cache, 0, 1, 4, result, output) || !result.empty() ||
        output.tokens)
        return 1;
    std::puts("PHYSICAL_F16_UNSUPPORTED_FALLBACK PASS 5");
    return 0;
}
