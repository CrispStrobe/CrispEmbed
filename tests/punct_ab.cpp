// punct_ab.cpp — drive ANY punctuation model through the public C API.
//
// tests/firered_punct_ab.cpp does this for FireRedPunc by calling
// fireredpunc_* directly, which cannot reach the PCS engine at all — and PCS is
// four registry entries with no parity harness. `crispembed_punct_init`
// dispatches on `general.architecture`, so one driver covers both, and a future
// third engine needs no new binary.
//
// Prints one punctuated line per input line, so a parity script can diff it
// against a reference. The engines' own diff hooks (PCS_DUMP_LOGITS,
// FIREREDPUNC_DUMP_LOGITS) work unchanged underneath.

#include "core/clean_exit.h"
#include "crispembed.h"

#include <cstdio>
#include <fstream>
#include <string>

static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <corpus.txt>\n", argv[0]);
        return 2;
    }
    void * ctx = crispembed_punct_init(argv[1], 0);
    if (!ctx) {
        std::fprintf(stderr, "error: failed to load %s\n", argv[1]);
        return 1;
    }
    std::ifstream f(argv[2]);
    if (!f) {
        std::fprintf(stderr, "error: cannot read %s\n", argv[2]);
        crispembed_punct_free(ctx);
        return 1;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const char * out = crispembed_punct_process(ctx, line.c_str());
        std::printf("%s\n", out ? out : "");
        std::fflush(stdout);
    }
    crispembed_punct_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
