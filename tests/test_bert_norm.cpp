// tests/test_bert_norm.cpp — core/bert_norm.h: HuggingFace BertNormalizer's
// lowercase + strip-accents stage for uncased WordPiece models.
//
// Hermetic: no vocab file, no GGUF, no network.
//
// Goldens captured verbatim from HuggingFace's OWN normalizer — the RUST one,
// which is what BertTokenizerFast actually runs for every affected model:
//
//     from tokenizers import NormalizedString
//     from tokenizers.normalizers import BertNormalizer
//     n = BertNormalizer(clean_text=False, handle_chinese_chars=False,
//                        strip_accents=None, lowercase=True)
//     ns = NormalizedString(text); n.normalize(ns); str(ns)
//
// (`strip_accents=None` + `lowercase=True` is literally what
// all-MiniLM-L6-v2's and all-mpnet-base-v2's tokenizer.json declare; HF
// resolves it as `strip_accents.unwrap_or(lowercase)` = true.)
//
// The rows that a hand-written accent-stripping table gets WRONG, and the
// reason this table is generated from a real NFD instead:
//
//     Øystein -> øystein   (NOT oystein)   Ø has no canonical decomposition
//     Łódź    -> łodz      (NOT lodz)      Ł survives, ó and ź do not
//     Đà Nẵng -> đa nang   (NOT da nang)
//     Straße  -> straße    ß survives; but STRAẞE -> straße (U+1E9E lowers)
//     ırmak   -> ırmak     dotless ı survives; İstanbul -> istanbul
//     ﬁnal    -> ﬁnal      a COMPATIBILITY ligature: NFD leaves it alone
//
// and one that pins the reference implementation itself:
//
//     ΟΔΟΣ    -> οδοσ      NOT οδος. The Rust normalizer is context-free and
//                          does NOT apply Final_Sigma; Python's slow
//                          BasicTokenizer does. The models run the Rust one.
//
//   c++ -std=c++17 -O1 -Isrc tests/test_bert_norm.cpp -o /tmp/test-bert-norm
//   /tmp/test-bert-norm
//
// Exit 0 == every golden, every invariant and the end-to-end WordPiece
// behaviour match.

#include "core/bert_norm.h"
#include "core/clean_exit.h"
#include "tokenizer.h"

#include <algorithm>
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
    { "caf\xc3""\xa9""",
      "cafe" },
    { "M\xc3""\xbc""ller",
      "muller" },
    { "na\xc3""\xaf""ve",
      "naive" },
    { "r\xc3""\xa9""sum\xc3""\xa9""",
      "resume" },
    { "\xc3""\xbc""ber",
      "uber" },
    { "\xc3""\x81""ngel",
      "angel" },
    { "Fran\xc3""\xa7""ois",
      "francois" },
    { "\xc3""\x98""ystein",
      "\xc3""\xb8""ystein" },
    { "\xc5""\x81""\xc3""\xb3""d\xc5""\xba""",
      "\xc5""\x82""odz" },
    { "\xc4""\x90""\xc3""\xa0"" N\xe1""\xba""\xb5""ng",
      "\xc4""\x91""a nang" },
    { "Stra\xc3""\x9f""e",
      "stra\xc3""\x9f""e" },
    { "\xc3""\x86""ther",
      "\xc3""\xa6""ther" },
    { "Hr\xc3""\xa5""kon",
      "hrakon" },
    { "\xc3""\x9e""\xc3""\xb3""rr",
      "\xc3""\xbe""orr" },
    { "Die B\xc3""\xa4""ckerei an der Stra\xc3""\x9f""e verkauft s\xc3""\xbc""\xc3""\x9f""e Br\xc3""\xb6""tchen",
      "die backerei an der stra\xc3""\x9f""e verkauft su\xc3""\x9f""e brotchen" },
    { "Le gar\xc3""\xa7""on a d\xc3""\xa9""j\xc3""\xa0"" mang\xc3""\xa9"" son d\xc3""\xa9""jeuner \xc3""\xa0"" l'h\xc3""\xb4""tel",
      "le garcon a deja mange son dejeuner a l'hotel" },
    { "El ni\xc3""\xb1""o peque\xc3""\xb1""o compr\xc3""\xb3"" una pi\xc3""\xb1""ata en el mercado",
      "el nino pequeno compro una pinata en el mercado" },
    { "A informa\xc3""\xa7""\xc3""\xa3""o est\xc3""\xa1"" dispon\xc3""\xad""vel na p\xc3""\xa1""gina tr\xc3""\xaa""s",
      "a informacao esta disponivel na pagina tres" },
    // Already-decomposed input: the combining mark arrives in the source text.
    { "cafe\xcc""\x81""",
      "cafe" },
    { "Mu\xcc""\x88""ller",
      "muller" },
    { "A\xcc""\x8a""",
      "a" },
    { "\xc4""\xb0""stanbul",
      "istanbul" },
    { "ISPARTA",
      "isparta" },
    { "\xc4""\xb1""rmak",
      "\xc4""\xb1""rmak" },
    { "STRA\xe1""\xba""\x9e""E",
      "stra\xc3""\x9f""e" },
    { "\xef""\xac""\x81""nal",
      "\xef""\xac""\x81""nal" },
    { "\xef""\xac""\x82""ower",
      "\xef""\xac""\x82""ower" },
    { "\xce""\x9f""\xce""\x94""\xce""\x9f""\xce""\xa3""",
      "\xce""\xbf""\xce""\xb4""\xce""\xbf""\xcf""\x83""" },
    { "\xce""\x95""\xce""\xbb""\xce""\xbb""\xce""\xac""\xce""\xb4""\xce""\xb1""",
      "\xce""\xb5""\xce""\xbb""\xce""\xbb""\xce""\xb1""\xce""\xb4""\xce""\xb1""" },
    { "\xe1""\xbc""\x88""\xce""\xb8""\xe1""\xbf""\x86""\xce""\xbd""\xce""\xb1""\xce""\xb9""",
      "\xce""\xb1""\xce""\xb8""\xce""\xb7""\xce""\xbd""\xce""\xb1""\xce""\xb9""" },
    { "\xd0""\x9c""\xd0""\x9e""\xd0""\xa1""\xd0""\x9a""\xd0""\x92""\xd0""\x90""",
      "\xd0""\xbc""\xd0""\xbe""\xd1""\x81""\xd0""\xba""\xd0""\xb2""\xd0""\xb0""" },
    { "\xd0""\x9a""\xd0""\xb8""\xd1""\x80""\xd0""\xb8""\xd0""\xbb""\xd0""\xbb""\xd0""\xb8""\xd1""\x86""\xd0""\xb0""",
      "\xd0""\xba""\xd0""\xb8""\xd1""\x80""\xd0""\xb8""\xd0""\xbb""\xd0""\xbb""\xd0""\xb8""\xd1""\x86""\xd0""\xb0""" },
    // Hangul: the arithmetic L/V/T decomposition path, not the table.
    { "\xed""\x95""\x9c""\xea""\xb5""\xad""\xec""\x96""\xb4""",
      "\xe1""\x84""\x92""\xe1""\x85""\xa1""\xe1""\x86""\xab""\xe1""\x84""\x80""\xe1""\x85""\xae""\xe1""\x86""\xa8""\xe1""\x84""\x8b""\xe1""\x85""\xa5""" },
    { "\xec""\x84""\x9c""\xec""\x9a""\xb8""",
      "\xe1""\x84""\x89""\xe1""\x85""\xa5""\xe1""\x84""\x8b""\xe1""\x85""\xae""\xe1""\x86""\xaf""" },
    { "\xed""\x95""\x9c""",
      "\xe1""\x84""\x92""\xe1""\x85""\xa1""\xe1""\x86""\xab""" },
    // CJK ideographs and kana are untouched by this stage.
    { "\xe6""\x97""\xa5""\xe6""\x9c""\xac""\xe8""\xaa""\x9e""\xe3""\x81""\xae""\xe3""\x83""\x86""\xe3""\x82""\xad""\xe3""\x82""\xb9""\xe3""\x83""\x88""",
      "\xe6""\x97""\xa5""\xe6""\x9c""\xac""\xe8""\xaa""\x9e""\xe3""\x81""\xae""\xe3""\x83""\x86""\xe3""\x82""\xad""\xe3""\x82""\xb9""\xe3""\x83""\x88""" },
    { "\xe4""\xb8""\xad""\xe6""\x96""\x87""\xe6""\x96""\x87""\xe6""\x9c""\xac""",
      "\xe4""\xb8""\xad""\xe6""\x96""\x87""\xe6""\x96""\x87""\xe6""\x9c""\xac""" },
    { "Der Preis ist 100\xe2""\x82""\xac"" f\xc3""\xbc""r Cafe-Besucher",
      "der preis ist 100\xe2""\x82""\xac"" fur cafe-besucher" },
    { "The quick brown fox jumps over the lazy dog",
      "the quick brown fox jumps over the lazy dog" },
    { "ALL CAPS ASCII 12345 !@#$%",
      "all caps ascii 12345 !@#$%" },
};
// clang-format on

static int crispembed_test_main() {
    int fails = 0;

    // ---- 1. Goldens against HF's own normalizer -----------------------
    for (const auto & c : k_goldens) {
        const std::string got = core_bert::lower_strip_accents(c.in);
        if (got != c.want) {
            fails++;
            printf("FAIL golden    in=%s\n               got=%s\n              want=%s\n", c.in, got.c_str(), c.want);
        }
    }

    // ---- 2. THE ASCII INVARIANT ---------------------------------------
    // Exactly std::tolower over all of ASCII. This is what makes the default
    // flip safe: no pure-ASCII text can tokenize differently than it did
    // before, so no shipped English embedding moves.
    for (int cp = 0; cp < 0x80; cp++) {
        const std::string in(1, (char)cp);
        const std::string got = core_bert::lower_strip_accents(in);
        const std::string want(1, (char)std::tolower(cp));
        if (got != want) {
            fails++;
            printf("FAIL ascii-inv U+%04X got=%s want=%s\n", cp, got.c_str(), want.c_str());
        }
    }

    // ---- 3. Idempotence ------------------------------------------------
    // The output of a normalizer must be a fixed point of it. A table row
    // whose replacement is itself uppercase or accented breaks this.
    for (const auto & c : k_goldens) {
        const std::string once = core_bert::lower_strip_accents(c.in);
        const std::string twice = core_bert::lower_strip_accents(once);
        if (once != twice) {
            fails++;
            printf("FAIL idempot   %s -> %s -> %s\n", c.in, once.c_str(), twice.c_str());
        }
    }

    // ---- 4. Binary-search precondition ---------------------------------
    // lower_strip_cp binary-searches ROWS. If the generator ever emits an
    // unsorted or duplicated table the lookup silently misses instead of
    // crashing, so assert the ordering rather than trusting it.
    for (int i = 1; i < core_unicode_norm::N_ROWS; i++) {
        if (core_unicode_norm::ROWS[i].cp <= core_unicode_norm::ROWS[i - 1].cp) {
            fails++;
            printf("FAIL table-ord row %d: U+%04X after U+%04X\n", i, core_unicode_norm::ROWS[i].cp,
                   core_unicode_norm::ROWS[i - 1].cp);
            break;
        }
    }
    // The table must not carry rows the arithmetic/ASCII fast paths shadow.
    for (int i = 0; i < core_unicode_norm::N_ROWS; i++) {
        const uint32_t cp = core_unicode_norm::ROWS[i].cp;
        const bool shadowed = cp < 0x80 || (cp >= core_unicode_norm::HANGUL_SBASE &&
                                            cp < core_unicode_norm::HANGUL_SBASE + core_unicode_norm::HANGUL_SCOUNT);
        if (shadowed) {
            fails++;
            printf("FAIL table-shadow row %d U+%04X is unreachable behind a fast path\n", i, cp);
            break;
        }
    }

    // ---- 5. End-to-end: the [UNK] explosion is gone --------------------
    // A synthetic uncased vocab holding only the STRIPPED forms, exactly as a
    // real uncased WordPiece vocab does. Before the fix every accented word
    // below tokenized to a prefix + [UNK].
    const std::vector<std::string> vocab = { "[PAD]", "[UNK]",  "[CLS]",    "[SEP]", "cafe",  "muller", "uber", "naive",
                                             "angel", "resume", "francois", "the",   "quick", "brown",  "fox",  "##s" };
    const int unk_id = 1, cls_id = 2, sep_id = 3;
    WordPieceTokenizer tok;
    tok.load(vocab, cls_id, sep_id, unk_id, /*pad_id=*/0, /*max_length=*/32, /*do_lower_case=*/true);

    struct E2E {
        const char * text;
        const char * want_token;
    };
    const std::vector<E2E> k_e2e = {
        { "caf\xc3"
          "\xa9"
          "",
          "cafe" },
        { "M\xc3"
          "\xbc"
          "ller",
          "muller" },
        { "\xc3"
          "\xbc"
          "ber",
          "uber" },
        { "na\xc3"
          "\xaf"
          "ve",
          "naive" },
        { "\xc3"
          "\x81"
          "ngel",
          "angel" },
        { "r\xc3"
          "\xa9"
          "sum\xc3"
          "\xa9"
          "",
          "resume" },
        { "Fran\xc3"
          "\xa7"
          "ois",
          "francois" },
    };
    for (const auto & c : k_e2e) {
        const auto enc = tok.encode(c.text);
        // ids: [CLS] <word pieces> [SEP] (+ padding)
        int n_real = 0;
        bool has_unk = false;
        for (size_t i = 0; i < enc.ids.size(); i++) {
            if (enc.attn_mask[i] == 0) break;
            const int id = enc.ids[i];
            if (id == cls_id || id == sep_id) continue;
            n_real++;
            if (id == unk_id) has_unk = true;
        }
        const int want_id = (int)(std::find(vocab.begin(), vocab.end(), c.want_token) - vocab.begin());
        const bool ok = (n_real == 1 && !has_unk && enc.ids[1] == want_id);
        if (!ok) {
            fails++;
            printf("FAIL e2e       %-12s -> %d piece(s)%s, ids[1]=%d, want single id %d (%s)\n", c.text, n_real,
                   has_unk ? " incl [UNK]" : "", enc.ids.size() > 1 ? enc.ids[1] : -1, want_id, c.want_token);
        }
    }

    // Pure ASCII through the same tokenizer must be untouched.
    {
        const auto enc = tok.encode("The quick brown fox");
        const std::vector<int32_t> want = { cls_id, 11, 12, 13, 14, sep_id };
        std::vector<int32_t> got;
        for (size_t i = 0; i < enc.ids.size() && enc.attn_mask[i]; i++) got.push_back(enc.ids[i]);
        if (got != want) {
            fails++;
            printf("FAIL e2e-ascii pure-ASCII tokenization changed\n");
        }
    }

    printf("bert-norm: %zu goldens + 128 ascii-inv + %zu idempot + table + %zu e2e cases, %d failure(s)\n",
           k_goldens.size(), k_goldens.size(), k_e2e.size(), fails);
    return fails ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
