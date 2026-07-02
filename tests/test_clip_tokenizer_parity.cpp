// tests/test_clip_tokenizer_parity.cpp — CLIP BPE tokenizer parity harness.
//
// Replays HuggingFace CLIPTokenizerFast token IDs against CrispEmbed's
// clip_style BPETokenizer path and asserts an exact match. This guards the
// CLIP `</w>` end-of-word word-boundary tokenizer (fixed in
// "fix(clip_text): CLIP BPE word-boundary tokenizer"): the old code routed
// CLIP through the GPT-2 byte-level path, emitting a standalone `Ġ` (220)
// space token instead of `</w>`, so every token id was wrong and text-image
// cosine dropped to ~0.79.
//
// Self-contained — links only against src/tokenizer_bpe.cpp (no ggml/GGUF).
// Generate the reference files from any CLIP checkpoint first:
//
//   python tests/gen_clip_tokenizer_reference.py \
//       --model openai/clip-vit-base-patch32 --out-dir /tmp/clip-tok-ref
//
// Then build and run:
//
//   c++ -std=c++17 -O1 -Isrc tests/test_clip_tokenizer_parity.cpp \
//       src/tokenizer_bpe.cpp -o /tmp/test-clip-tok
//   /tmp/test-clip-tok /tmp/clip-tok-ref/{vocab,merges,expected}.tsv
//
// Exit code 0 == all probes match HF; non-zero == a tokenizer regression.

#include "tokenizer.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<std::string> read_lines(const char * path) {
    std::vector<std::string> out;
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return out;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        out.push_back(line);
    }
    return out;
}

// Unescape the \n and \t written by the generator.
static std::string unescape(const std::string & s) {
    std::string out;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            if (s[i + 1] == 'n') { out.push_back('\n'); i++; continue; }
            if (s[i + 1] == 't') { out.push_back('\t'); i++; continue; }
        }
        out.push_back(s[i]);
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s vocab.tsv merges.tsv expected.tsv\n", argv[0]);
        return 2;
    }
    auto vocab = read_lines(argv[1]);
    auto merges = read_lines(argv[2]);
    auto expected = read_lines(argv[3]);
    if (vocab.empty() || merges.empty() || expected.empty()) {
        fprintf(stderr, "missing/empty reference files — run gen_clip_tokenizer_reference.py\n");
        return 2;
    }

    BPETokenizer tok;
    // CLIP: eos=49407, pad=49407, suffix=-1, bos=49406, spm=false, max=77,
    // spm_dummy_prefix=false, clip_style=true.
    tok.load(vocab, merges, 49407, 49407, -1, 49406, false, 77, false, true);

    int pass = 0, fail = 0;
    for (const auto & row : expected) {
        auto tab = row.find('\t');
        if (tab == std::string::npos) continue;
        std::string text = unescape(row.substr(0, tab));

        std::vector<int> exp;
        std::stringstream ss(row.substr(tab + 1));
        std::string num;
        while (std::getline(ss, num, ',')) {
            if (!num.empty()) exp.push_back(std::stoi(num));
        }

        embed_tokens got = tok.encode(text);
        bool ok = got.ids.size() == exp.size();
        for (size_t i = 0; ok && i < exp.size(); i++) ok = got.ids[i] == exp[i];

        printf("%s '%s'\n", ok ? "PASS" : "FAIL", row.substr(0, tab).c_str());
        if (!ok) {
            printf("   expected:");
            for (int x : exp) printf(" %d", x);
            printf("\n   got:     ");
            for (int x : got.ids) printf(" %d", x);
            printf("\n");
        }
        ok ? pass++ : fail++;
    }
    printf("\n=== %d passed, %d failed ===\n", pass, fail);
    return fail ? 1 : 0;
}
