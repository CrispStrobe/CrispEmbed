// test_granite_vision_diff.cpp — crispembed-diff parity for Granite Vision.
//
// Compares C++ vision encoder + projector against Python reference.
// Usage: test-granite-vision-diff <model.gguf> <ref.gguf>

#include "granite_vision_ocr.h"
#include "crispembed_diff.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <map>

static crispembed_diff::Ref * g_ref = nullptr;
static int g_pass = 0, g_fail = 0;

static void dump_cb(const char * name, const float * data, int n, void * ud) {
    (void)ud;
    printf("  C++ stage: %s (%d elements)\n", name, n);
    if (g_ref && g_ref->has(name)) {
        auto r = g_ref->compare(name, data, n);
        bool pass = r.cos_min >= 0.99f;  // lower threshold for Q4_K
        printf("    cos_min=%.6f  max_abs=%.2e  %s\n",
               r.cos_min, r.max_abs, pass ? "PASS" : "FAIL");
        if (pass) g_pass++; else g_fail++;
    } else {
        printf("    (no reference for this stage)\n");
    }
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("Usage: test-granite-vision-diff <model.gguf> <ref.gguf>\n");
        return 1;
    }

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) { printf("Failed to load ref\n"); return 1; }
    g_ref = &ref;

    printf("Reference stages:\n");
    for (auto & name : ref.tensor_names()) {
        auto shape = ref.shape(name);
        printf("  %s: [", name.c_str());
        for (size_t i = 0; i < shape.size(); i++)
            printf("%s%lld", i ? "," : "", (long long)shape[i]);
        printf("]\n");
    }

    granite_vision_context * ctx = granite_vision_init(argv[1], 2);
    if (!ctx) { printf("Failed to load model\n"); return 1; }

    // Vision encoder + projector parity (ref must contain "input").
    auto [ref_input, ref_n] = ref.get_f32("input");
    if (ref_input) {
        printf("\nRunning vision encoder + projector...\n");
        granite_vision_dump_vision(ctx, ref_input, 384, 384, dump_cb, nullptr);
    }

    // LLM decode parity (ref must contain "llm_logits"). The token sequence
    // MUST match tools/dump_granite_llm_reference.py.
    // Pass "graph" as argv[3] to test the ggml LLM path (gv_run_llm_body)
    // instead of the scalar decode.
    if (ref.has("llm_logits")) {
        const int tokens[] = {12, 345, 678, 901, 234, 56, 789};
        const int n = (int)(sizeof(tokens) / sizeof(tokens[0]));
        bool use_graph = (argc > 3 && std::string(argv[3]) == "graph");
        printf("\nRunning LLM decode (%s, fixed token sequence)...\n",
               use_graph ? "ggml graph" : "scalar");
        if (use_graph) granite_vision_dump_llm_graph(ctx, tokens, n, dump_cb, nullptr);
        else           granite_vision_dump_llm(ctx, tokens, n, dump_cb, nullptr);
    }

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    granite_vision_free(ctx);
    return g_fail > 0 ? 1 : 0;
}
