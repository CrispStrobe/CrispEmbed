// Ministral3 ByteLevel pre-tokenizer parity. Golden splits were captured from
// malteos/most-embed-de/tokenizer.json with tokenizers.Tokenizer.

#include "core/bpe.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <string>
#include <vector>

struct Case {
    const char * text;
    std::vector<std::string> splits;
};

static const std::vector<Case> k_cases = {
    { "query: Wie \xc3\xa4ndere ich meine Bestellung 2026?",
      { "query", ":", "\304\240Wie", "\304\240\303\203\302\244ndere", "\304\240ich", "\304\240meine",
        "\304\240Bestellung", "\304\240", "2", "0", "2", "6", "?" } },
    { "don't DON'T can't THEY'RE", { "don", "'t", "\304\240DON", "'T", "\304\240can", "'t", "\304\240THEY", "'RE" } },
    { "camelCaseIdentifier XMLHttpRequest HTTPServer",
      { "camel", "Case", "Identifier", "\304\240XMLHttp", "Request", "\304\240HTTPServer" } },
    { "a\n\nb", { "a", "\304\212\304\212", "b" } },
    { "1234567", { "1", "2", "3", "4", "5", "6", "7" } },
};

static int crispembed_test_main(int, char **) {
    int failed = 0;
    for (const auto & tc : k_cases) {
        std::vector<std::string> got;
        for (const auto & piece : core_bpe::ministral_pretokenize(tc.text))
            got.push_back(core_bpe::bytes_to_unicode(piece.data(), piece.size()));
        if (got != tc.splits) {
            std::fprintf(stderr, "FAIL: %s\n", tc.text);
            failed++;
        }
    }
    std::printf("ministral pretokenize: %zu cases, %d failed\n", k_cases.size(), failed);
    return failed ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main(0, nullptr));
}
