// tests/test_decoder_embed_diff.cpp — decoder-embedding (Qwen3/Gemma3) parity via crispembed-diff.
// Usage: ./test-decoder-embed-diff model.gguf ref.gguf
//
// Set CRISPEMBED_DUMP_LAYERS_GGUF=/path/native.gguf to compare every graph
// boundary captured by tools/dump_decoder_embed_reference.py.

#include "crispembed.h"
#include "core/clean_exit.h"
#include "crispembed_diff.h"
#include <cstdlib>
#include <cstdio>
#include <string>

#define GREEN "\033[32m"
#define RED "\033[31m"
#define RESET "\033[0m"
static int n_pass = 0, n_fail = 0;
static void check(const char * n, bool c) {
    if (c) {
        printf("  %s[PASS]%s %s\n", GREEN, RESET, n);
        n_pass++;
    } else {
        printf("  %s[FAIL]%s %s\n", RED, RESET, n);
        n_fail++;
    }
}

static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s model.gguf ref.gguf\n", argv[0]);
        return 1;
    }
    printf("decoder_embed (Qwen3/Gemma3) — parity test\n");
    printf("  Model: %s\n  Ref:   %s\n\n", argv[1], argv[2]);

    crispembed_diff::Ref ref;
    if (!ref.load(argv[2])) return 1;
    const std::string ref_text = ref.meta("ref.text");
    const char * text =
        argc >= 4 ? argv[3] : (ref_text.empty() ? "The quick brown fox jumps over the lazy dog" : ref_text.c_str());

    crispembed_context * ctx = crispembed_init(argv[1], 0);
    check("model loads", ctx != nullptr);
    if (!ctx) return 1;

    int dim = 0;
    const float * emb = crispembed_encode(ctx, text, &dim);
    check("encode returns non-null", emb != nullptr);
    printf("  embedding dim: %d\n", dim);

    if (emb && dim > 0) {
        auto r = ref.compare("final_embedding", emb, dim);
        // 0.98 floor: q8_0 GGUF vs f32 HF ref → ~0.99; a scramble craters to ~0.
        printf("  final_embedding: cos_min=%.6f max_abs=%.6f |mine|=%.6f |ref|=%.6f  %s\n", r.cos_min, r.max_abs,
               r.mine_norm, r.ref_norm, r.is_pass(0.98f) ? "PASS" : "FAIL");
        check("final_embedding cos >= 0.98", r.is_pass(0.98f));
    }

    const char * native_path = std::getenv("CRISPEMBED_DUMP_LAYERS_GGUF");
    if (native_path && *native_path) {
        float stage_floor = 0.995f;
        if (const char * floor_env = std::getenv("CRISPEMBED_DIFF_MIN_COS"))
            stage_floor = std::strtof(floor_env, nullptr);
        crispembed_diff::Ref native;
        check("native stage dump loads", native.load(native_path));
        if (native.has("post_embed")) {
            printf("\n=== Per-stage crispembed-diff ===\n");
            for (const std::string & name : native.tensor_names()) {
                auto [data, n] = native.get_f32(name);
                const auto shape = native.shape(name);
                const int row_dim = shape.size() >= 2 ? 0 : -1;
                auto r = ref.compare(name, data, n, row_dim);
                if (!r.found) continue;
                const bool ok = r.is_pass(stage_floor);
                printf("  %-20s cos_min=%.6f cos_global=%.6f max_abs=%.3e |mine|=%.6f |ref|=%.6f %s\n", name.c_str(),
                       r.cos_min, r.cos_global, r.max_abs, r.mine_norm, r.ref_norm, ok ? "PASS" : "FAIL");
                check((name + " cos >= " + std::to_string(stage_floor)).c_str(), ok);
            }
        }
    }

    crispembed_free(ctx);
    printf("\n=== Results: %d passed, %d failed ===\n", n_pass, n_fail);
    return n_fail > 0 ? 1 : 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
