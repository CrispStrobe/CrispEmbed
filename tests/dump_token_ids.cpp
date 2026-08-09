// tests/dump_token_ids.cpp — dump the token ids a real GGUF actually produces.
//
// The WordPiece parity work (docs/LANGUAGES.md) could drive the tokenizer
// directly from a vocab.txt, because WordPieceTokenizer is self-contained.
// SentencePiece and BPE models cannot: their vocab, merges, precompiled
// charsmap and pre-tokenizer selection all live in the GGUF. This loads the
// real model through the public C API and prints what the SHIPPING runtime
// tokenizes, so a parity check measures the runtime rather than a
// reimplementation of it.
//
//   dump-token-ids <model.gguf> <corpus.txt> [--strings]
//
// One line of space-separated ids per input line; `--strings` adds a second
// line with the surface forms. Used by tests/embed_tokenizer_parity.py.

#include "crispembed.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <corpus.txt> [--strings]\n", argv[0]);
        return 2;
    }
    const bool want_strings = (argc > 3 && strcmp(argv[3], "--strings") == 0);

    crispembed_context * ctx = crispembed_init(argv[1], 4);
    if (!ctx) {
        fprintf(stderr, "error: failed to load %s\n", argv[1]);
        return 1;
    }
    fprintf(stderr, "tokenizer_kind=%d\n", crispembed_tokenizer_kind(ctx));

    std::ifstream f(argv[2]);
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", argv[2]);
        crispembed_free(ctx);
        return 2;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        int n_tokens = 0, dim = 0;
        // encode_tokens is what populates last_token_ids; the vectors are
        // ignored here, only the ids matter.
        if (!crispembed_encode_tokens(ctx, line.c_str(), &n_tokens, &dim)) {
            printf("\n");
            if (want_strings) printf("\n");
            continue;
        }
        const int32_t * ids = crispembed_last_token_ids(ctx);
        for (int i = 0; i < n_tokens; i++) printf("%s%d", i ? " " : "", ids ? ids[i] : -1);
        printf("\n");
        if (want_strings) {
            for (int i = 0; i < n_tokens; i++) {
                const char * s = ids ? crispembed_token_str(ctx, ids[i]) : "";
                printf("%s%s", i ? " " : "", s ? s : "");
            }
            printf("\n");
        }
    }
    crispembed_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
