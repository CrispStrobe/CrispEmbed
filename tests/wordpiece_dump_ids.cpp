// tests/wordpiece_dump_ids.cpp — dump WordPiece token ids for a real vocab.
//
// The C++ half of tests/wordpiece_hf_parity.py: it drives the SHIPPING
// WordPieceTokenizer (same split_words / wordpiece code the runtime uses) so
// the parity check measures the runtime, not a reimplementation.
//
//   c++ -std=c++17 -O1 -Isrc tests/wordpiece_dump_ids.cpp src/tokenizer.cpp \
//       -o build/wordpiece-dump-ids
//   build/wordpiece-dump-ids vocab.txt corpus.txt      # one id-list per line
//
// Honors CRISPEMBED_WORDPIECE_HF_NORM, so the same binary produces both A/B
// arms.

#include "tokenizer.h"

#include <cstdio>
#include "core/clean_exit.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

static int crispembed_test_main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <vocab.txt> <corpus.txt> [cls sep unk pad]\n", argv[0]);
        return 2;
    }
    std::vector<std::string> vocab;
    {
        std::ifstream f(argv[1]);
        if (!f) {
            fprintf(stderr, "cannot open vocab %s\n", argv[1]);
            return 2;
        }
        std::string line;
        while (std::getline(f, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            vocab.push_back(line);
        }
    }
    std::unordered_map<std::string, int> id_of;
    for (int i = 0; i < (int)vocab.size(); i++) id_of.emplace(vocab[i], i);
    auto id = [&](const char * t, int dflt) {
        auto it = id_of.find(t);
        return it == id_of.end() ? dflt : it->second;
    };

    // Uncased detection mirrors crispembed.cpp: a single uppercase letter
    // token anywhere in the vocab means the model is cased.
    bool do_lower_case = true;
    for (const auto & t : vocab) {
        if (t.size() == 1 && t[0] >= 'A' && t[0] <= 'Z') {
            do_lower_case = false;
            break;
        }
    }

    // Special ids may be overridden on the command line: MPNet wraps with
    // <s>/</s> rather than [CLS]/[SEP], and guessing wrong makes every
    // sequence differ by its first and last token in BOTH arms — which reads
    // as a tokenizer bug when it is only a harness bug.
    int cls_id = id("[CLS]", 101), sep_id = id("[SEP]", 102);
    int unk_id = id("[UNK]", 100), pad_id = id("[PAD]", 0);
    if (argc >= 7) {
        cls_id = atoi(argv[3]);
        sep_id = atoi(argv[4]);
        unk_id = atoi(argv[5]);
        pad_id = atoi(argv[6]);
    }

    WordPieceTokenizer tok;
    tok.load(vocab, cls_id, sep_id, unk_id, pad_id, 512, do_lower_case);

    std::ifstream f(argv[2]);
    if (!f) {
        fprintf(stderr, "cannot open corpus %s\n", argv[2]);
        return 2;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const auto enc = tok.encode(line);
        bool first = true;
        for (size_t i = 0; i < enc.ids.size(); i++) {
            if (!enc.attn_mask[i]) break;
            printf("%s%d", first ? "" : " ", (int)enc.ids[i]);
            first = false;
        }
        printf("\n");
    }
    return 0;
}

int main(int argc, char ** argv) {
    core_util::clean_exit(crispembed_test_main(argc, argv));
}
