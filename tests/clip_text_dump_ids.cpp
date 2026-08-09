// tests/clip_text_dump_ids.cpp — dump the token ids the clip_text engine
// produces for a real CLIP/SigLIP text GGUF.
//
// The generic dump-token-ids driver cannot be used here: clip_text is a
// standalone engine with its own loader (crispembed_init rejects these GGUFs),
// and SigLIP's tokenizer lives behind clip_text's own SentencePiece path.
//
//   clip-text-dump-ids <clip-or-siglip-text.gguf> <corpus.txt>
//
// One line of space-separated ids per input line. Used by
// tests/clip_text_tokenizer_parity.py.

#include "clip_text_embed.h"

#include "core/clean_exit.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// Exposed by clip_text_embed.cpp for parity testing: tokenize without running
// the transformer.
namespace clip_text {
std::vector<int32_t> tokenize_only(context * ctx, const char * text);
}

static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <corpus.txt>\n", argv[0]);
        return 2;
    }
    clip_text::context * ctx = nullptr;
    if (!clip_text::load(&ctx, argv[1], 4) || !ctx) {
        fprintf(stderr, "error: failed to load %s\n", argv[1]);
        return 1;
    }
    std::ifstream f(argv[2]);
    if (!f) {
        fprintf(stderr, "error: cannot open %s\n", argv[2]);
        clip_text::free(ctx);
        return 2;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const std::vector<int32_t> ids = clip_text::tokenize_only(ctx, line.c_str());
        for (size_t i = 0; i < ids.size(); i++) printf("%s%d", i ? " " : "", (int)ids[i]);
        printf("\n");
    }
    clip_text::free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
