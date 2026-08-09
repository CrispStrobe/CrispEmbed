// tests/test_spm_norm.cpp — core/spm_norm.h: the SentencePiece `Precompiled`
// (nmt_nfkc charsmap) normalizer that every XLM-R-family Unigram model
// declares and that CrispEmbed implemented nowhere.
//
// Hermetic: no vocab, no GGUF, no network.
//
// Goldens captured from HuggingFace's OWN normalizer — the Precompiled
// component in isolation, which is the part that belongs in the table (the
// trailing " " -> "▁" Replace is SentencePieceTokenizer's own job):
//
//     import base64, json
//     from tokenizers.normalizers import Precompiled
//     from tokenizers import NormalizedString
//     cm = [x for x in tj["normalizer"]["normalizers"]
//           if x["type"] == "Precompiled"][0]["precompiled_charsmap"]
//     n = Precompiled(base64.b64decode(cm))
//     ns = NormalizedString(text); n.normalize(ns); str(ns)
//
// What this charsmap does and does NOT do — the rows that matter:
//
//     …  ‥          -> ...  ..
//     ＡＢＣ １２３   -> ABC 123      every fullwidth form
//     U+3000        -> ' '          ideographic space
//     ﬁ ﬂ           -> fi fl
//     ① ② Ⅳ ㎏ ㈱  -> 1 2 IV kg (株)
//     café Müller   -> UNCHANGED    it does NOT strip accents
//     ΟΔΟΣ, 日本語  -> UNCHANGED
//
// The accent row is the one to keep straight: BertNormalizer strips accents,
// this charsmap does not. Two different normalizers, two different tables.
//
//   c++ -std=c++17 -O1 -Isrc tests/test_spm_norm.cpp -o /tmp/test-spm-norm
//   /tmp/test-spm-norm
//
// Exit 0 == every golden and every invariant matches.

#include "core/clean_exit.h"
#include "core/spm_norm.h"

#include <cctype>
#include <cstdio>
#include <string>
#include <vector>

struct Case {
    const char * in;
    const char * want;
};

// clang-format off
static const std::vector<Case> k_goldens = {
    { "The quick brown fox jumps over the lazy dog",
      "The quick brown fox jumps over the lazy dog" },
    { "Prices went up 15% in Q3 2024!",
      "Prices went up 15% in Q3 2024!" },
    { "He said hello then left\xe2""\x80""\xa6""",
      "He said hello then left..." },
    { "continue\xe2""\x80""\xa5""and\xe2""\x80""\xa6""done",
      "continue..and...done" },
    { "\xef""\xbc""\xa1""\xef""\xbc""\xa2""\xef""\xbc""\xa3""\xe3""\x80""\x80""\xef""\xbc""\x91""\xef""\xbc""\x92""\xef""\xbc""\x93""",
      "ABC 123" },
    { "\xe4""\xbe""\xa1""\xe6""\xa0""\xbc""\xe3""\x81""\xaf""\xef""\xbf""\xa5""\xef""\xbc""\x91""\xef""\xbc""\x92""\xef""\xbc""\x93""\xef""\xbc""\x94""\xe3""\x81""\xa7""\xe3""\x81""\x99""",
      "\xe4""\xbe""\xa1""\xe6""\xa0""\xbc""\xe3""\x81""\xaf""\xc2""\xa5""1234\xe3""\x81""\xa7""\xe3""\x81""\x99""" },
    { "\xef""\xbd""\x88""\xef""\xbd""\x85""\xef""\xbd""\x8c""\xef""\xbd""\x8c""\xef""\xbd""\x8f""\xe3""\x80""\x80""\xef""\xbd""\x97""\xef""\xbd""\x8f""\xef""\xbd""\x92""\xef""\xbd""\x8c""\xef""\xbd""\x84""",
      "hello world" },
    { "\xef""\xac""\x81""le \xef""\xac""\x82""ow",
      "file flow" },
    { "item \xe2""\x91""\xa0"" and \xe2""\x91""\xa1"" and \xe2""\x85""\xa3""",
      "item 1 and 2 and IV" },
    { "\xe9""\x87""\x8d""\xe3""\x81""\x95""\xe3""\x8e""\x8f""\xe3""\x81""\xa8""\xe3""\x88""\xb1""\xe3""\x81""\xae""\xe8""\xa1""\xa8""\xe8""\xa8""\x98""",
      "\xe9""\x87""\x8d""\xe3""\x81""\x95""kg\xe3""\x81""\xa8""(\xe6""\xa0""\xaa"")\xe3""\x81""\xae""\xe8""\xa1""\xa8""\xe8""\xa8""\x98""" },
    { "\xe6""\x97""\xa5""\xe6""\x9c""\xac""\xe8""\xaa""\x9e""\xe3""\x81""\xae""\xe3""\x83""\x86""\xe3""\x82""\xad""\xe3""\x82""\xb9""\xe3""\x83""\x88""",
      "\xe6""\x97""\xa5""\xe6""\x9c""\xac""\xe8""\xaa""\x9e""\xe3""\x81""\xae""\xe3""\x83""\x86""\xe3""\x82""\xad""\xe3""\x82""\xb9""\xe3""\x83""\x88""" },
    // It does NOT strip accents — that is BertNormalizer's job, not this one.
    { "caf\xc3""\xa9"" M\xc3""\xbc""ller \xc3""\x98""ystein",
      "caf\xc3""\xa9"" M\xc3""\xbc""ller \xc3""\x98""ystein" },
    { "\xce""\x9f""\xce""\x94""\xce""\x9f""\xce""\xa3"" \xce""\x95""\xce""\xbb""\xce""\xbb""\xce""\xac""\xce""\xb4""\xce""\xb1""",
      "\xce""\x9f""\xce""\x94""\xce""\x9f""\xce""\xa3"" \xce""\x95""\xce""\xbb""\xce""\xbb""\xce""\xac""\xce""\xb4""\xce""\xb1""" },
    { "tab\x09""here",
      "tab here" },
};
// clang-format on

static int crispembed_test_main() {
    int fails = 0;

    // ---- 1. Goldens against HF's own Precompiled normalizer -----------
    for (const auto & c : k_goldens) {
        const std::string got = core_spm::normalize(c.in);
        if (got != c.want) {
            fails++;
            printf("FAIL golden    in=%s\n               got=%s\n              want=%s\n", c.in, got.c_str(), c.want);
        }
    }

    // ---- 2. THE PRINTABLE-ASCII INVARIANT ------------------------------
    // 0x20-0x7E passes through untouched. This is what makes enabling the
    // normalizer by default safe for English text.
    for (int cp = 0x20; cp < 0x7F; cp++) {
        const std::string in(1, (char)cp);
        const std::string got = core_spm::normalize(in);
        if (got != in) {
            fails++;
            printf("FAIL ascii-inv U+%04X '%c' -> %s\n", cp, (char)cp, got.c_str());
        }
    }

    // ---- 3. The rest of ASCII, pinned rather than glossed over ---------
    // "ASCII is untouched" would be FALSE: \t \n \f \r fold to a space and
    // the other C0 controls plus DEL are deleted. Both match HF, and both
    // are harmless for a tokenizer that splits on whitespace — but they are
    // asserted, not assumed.
    for (int cp : { 0x09, 0x0A, 0x0C, 0x0D }) {
        const std::string got = core_spm::normalize(std::string(1, (char)cp));
        if (got != " ") {
            fails++;
            printf("FAIL ascii-ws  U+%04X should fold to a space, got %zu byte(s)\n", cp, got.size());
        }
    }
    for (int cp : { 0x01, 0x08, 0x0B, 0x1F, 0x7F }) {
        const std::string got = core_spm::normalize(std::string(1, (char)cp));
        if (!got.empty()) {
            fails++;
            printf("FAIL ascii-ctl U+%04X should be deleted, got %zu byte(s)\n", cp, got.size());
        }
    }

    // ---- 4. Idempotence ------------------------------------------------
    for (const auto & c : k_goldens) {
        const std::string once = core_spm::normalize(c.in);
        const std::string twice = core_spm::normalize(once);
        if (once != twice) {
            fails++;
            printf("FAIL idempot   %s -> %s -> %s\n", c.in, once.c_str(), twice.c_str());
        }
    }

    // ---- 5. Binary-search preconditions --------------------------------
    for (int i = 1; i < core_unicode_spm::N_ROWS; i++) {
        if (core_unicode_spm::ROWS[i].cp <= core_unicode_spm::ROWS[i - 1].cp) {
            fails++;
            printf("FAIL table-ord row %d: U+%04X after U+%04X\n", i, core_unicode_spm::ROWS[i].cp,
                   core_unicode_spm::ROWS[i - 1].cp);
            break;
        }
    }
    for (int i = 0; i < core_unicode_spm::N_ROWS; i++) {
        const uint32_t cp = core_unicode_spm::ROWS[i].cp;
        if (cp >= 0x20 && cp < 0x7F) {
            fails++;
            printf("FAIL table-shadow row %d U+%04X is unreachable behind the ASCII fast path\n", i, cp);
            break;
        }
    }

    // ---- 6. SigLIP's canonicalization on top of the charsmap -----------
    // Goldens from the tokenizer that ACTUALLY RUNS: there is no fast SigLIP
    // tokenizer (`use_fast=True` still returns the slow SiglipTokenizer), so
    // `canonicalize_text` — lower, translate away ALL of string.punctuation,
    // collapse whitespace, strip — is the authority. tokenizer.json's Replace
    // regex keeps `/ < >` and is NOT what executes; trusting it cost a wrong
    // implementation that scored 16/17 instead of 17/17 against HF.
    // clang-format off
    static const std::vector<Case> k_siglip = {
        { "A photo of a CAT.",
          "a photo of a cat" },
        { "Hello, World! How are you?",
          "hello world how are you" },
        { "caf\xc3""\xa9"" M\xc3""\xbc""ller",
          "caf\xc3""\xa9"" m\xc3""\xbc""ller" },   // lowercases, does NOT strip accents
        { "a dog & a cat (running) - fast!",
          "a dog a cat running fast" },
        { "the a/b test with <tags> and >arrows<",
          "the ab test with tags and arrows" },    // / < > ARE stripped
        { "under_score and hyphen-ated",
          "underscore and hyphenated" },
        { "  leading and trailing  ",
          "leading and trailing" },
        { "multiple    inner     spaces",
          "multiple inner spaces" },
        { "\xe6""\x97""\xa5""\xe6""\x9c""\xac""\xe8""\xaa""\x9e""\xe3""\x81""\xae""\xe3""\x83""\x86""\xe3""\x82""\xad""\xe3""\x82""\xb9""\xe3""\x83""\x88""",
          "\xe6""\x97""\xa5""\xe6""\x9c""\xac""\xe8""\xaa""\x9e""\xe3""\x81""\xae""\xe3""\x83""\x86""\xe3""\x82""\xad""\xe3""\x82""\xb9""\xe3""\x83""\x88""" },
        { "MiXeD CaSe WoRdS",
          "mixed case words" },
        { "", "" },
        { "!!!", "" },
    };
    // clang-format on
    for (const auto & c : k_siglip) {
        const std::string got = core_spm::siglip_normalize(c.in);
        if (got != c.want) {
            fails++;
            printf("FAIL siglip    in=%s\n               got=%s\n              want=%s\n", c.in, got.c_str(), c.want);
        }
    }

    printf("spm-norm: %zu goldens + 95 ascii-inv + 9 ascii-ctl + %zu idempot + table + %zu siglip, %d failure(s)\n",
           k_goldens.size(), k_goldens.size(), k_siglip.size(), fails);
    return fails ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
