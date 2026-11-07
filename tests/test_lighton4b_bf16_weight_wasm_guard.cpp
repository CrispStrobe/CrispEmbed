// Preprocessor fallback control only: this is not a WebAssembly runtime test.
#define __EMSCRIPTEN__ 1
#include "lighton4b_bf16_dequant_weights.h"
#include "core/clean_exit.h"
#include <cstdio>
#include <cstdlib>
static int run() {
    for (const char * name : { "CRISPEMBED_LIGHTON4B_BF16_QUANT_WEIGHTS", "CRISPEMBED_LIGHTON4B_NATIVE_CONTEXT",
                               "CRISPEMBED_LIGHTON4B_QUANT", "CRISPEMBED_QWEN35_QUANT" }) {
#ifdef _WIN32
        _putenv_s(name, "1");
#else
        setenv(name, "1", 1);
#endif
    }
    if (qwen35_ocr::lighton4b_bf16_dequant_weights_mode()) return 1;
    std::puts("BF16_WEIGHT_WASM_GUARD passed=1 simulated_preprocessor=1 actual_wasm_runtime=0");
    return 0;
}
int main() {
    core_util::clean_exit(run());
}
