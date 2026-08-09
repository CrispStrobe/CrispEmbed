// Decoded-output A/B for the fireredpunc tokenizer fix: punctuate each input
// line and print the result. Arm is selected by CRISPEMBED_FIREREDPUNC_HF_TOK.
#include "fireredpunc.h"

#include "core/clean_exit.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <corpus.txt>\n", argv[0]);
        return 2;
    }
    fireredpunc_context * ctx = fireredpunc_init(argv[1]);
    if (!ctx) {
        fprintf(stderr, "error: load failed\n");
        return 1;
    }
    // FIREREDPUNC_DUMP_IDS=1 prints the token ids instead of the punctuated
    // text, so tests/firered_tokenizer_parity.py can compare them against
    // HuggingFace's BertTokenizer on the same vocab.
    const bool dump_ids = std::getenv("FIREREDPUNC_DUMP_IDS") != nullptr;

    std::ifstream f(argv[2]);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (dump_ids) {
            int n = 0;
            const int * ids = fireredpunc_debug_token_ids(ctx, line.c_str(), &n);
            for (int i = 0; i < n; i++) printf("%s%d", i ? " " : "", ids[i]);
            printf("\n");
            continue;
        }
        char * out = fireredpunc_process(ctx, line.c_str());
        printf("%s\n", out ? out : "(null)");
        if (out) free(out);
    }
    fireredpunc_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
