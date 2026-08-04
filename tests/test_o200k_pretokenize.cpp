// tests/test_o200k_pretokenize.cpp — o200k ByteLevel pre-tokenizer parity.
//
// Hermetic: no vocab, no merges, no GGUF, no network. Pre-tokenization is pure
// string splitting, so the golden splits below are HuggingFace's own
// `tokenizer.pre_tokenizer.pre_tokenize_str()` output (byte-level encoded, so
// U+0120 'G-dot' is a space and U+010A 'C-dot' a newline), captured verbatim
// from ibm-granite/granite-embedding-97m-multilingual-r2 — the o200k_base split
// that model's tokenizer.json declares:
//
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//  |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//  |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Two properties separate this from the Qwen split next door, and a typo in
// either is invisible on plain lowercase English:
//   * it branches on LETTER CASE (alternative 1 is "uppercase run then
//     lowercase run", alternative 2 the reverse), which is why camelCase
//     splits and ALL-CAPS does not — and why the repo's historical
//     "every byte >= 0x80 is a letter" approximation is not good enough:
//     it splits all-caps German ("ÄRGER") after the umlaut. core_unicode
//     carries the real Lu/Ll/Lo/M/N/White_Space table for that reason.
//   * contractions are a SUFFIX of the word token ("don't" is one pre-token,
//     not "don" + "'t"), digits come in groups of up to THREE (Qwen: one),
//     and a punctuation run may swallow trailing '/' as well as CR/LF.
//
//   c++ -std=c++17 -O1 -Isrc tests/test_o200k_pretokenize.cpp -o /tmp/test-o200k-pretok
//   /tmp/test-o200k-pretok
//
// Exit 0 == every split matches HF.

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
    { "The quick brown fox jumps over the lazy dog.",
      { "The",
        "\xc4"
        "\xa0"
        "quick",
        "\xc4"
        "\xa0"
        "brown",
        "\xc4"
        "\xa0"
        "fox",
        "\xc4"
        "\xa0"
        "jumps",
        "\xc4"
        "\xa0"
        "over",
        "\xc4"
        "\xa0"
        "the",
        "\xc4"
        "\xa0"
        "lazy",
        "\xc4"
        "\xa0"
        "dog",
        "." } },
    { "Wie hoch ist der Mount Everest?",
      { "Wie",
        "\xc4"
        "\xa0"
        "hoch",
        "\xc4"
        "\xa0"
        "ist",
        "\xc4"
        "\xa0"
        "der",
        "\xc4"
        "\xa0"
        "Mount",
        "\xc4"
        "\xa0"
        "Everest",
        "?" } },
    { "Gr\xc3"
      "\xbc"
      "\xc3"
      "\x9f"
      "e aus M\xc3"
      "\xbc"
      "nchen \xe2"
      "\x80"
      "\x94"
      " Stra\xc3"
      "\x9f"
      "enbahn, \xc3"
      "\x9c"
      "bermut, \xc3"
      "\x96"
      "l und \xc3"
      "\x84"
      "pfel.",
      { "Gr\xc3"
        "\x83"
        "\xc2"
        "\xbc"
        "\xc3"
        "\x83"
        "\xc5"
        "\x81"
        "e",
        "\xc4"
        "\xa0"
        "aus",
        "\xc4"
        "\xa0"
        "M\xc3"
        "\x83"
        "\xc2"
        "\xbc"
        "nchen",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xa2"
        "\xc4"
        "\xa2"
        "\xc4"
        "\xb6"
        "",
        "\xc4"
        "\xa0"
        "Stra\xc3"
        "\x83"
        "\xc5"
        "\x81"
        "enbahn",
        ",",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x83"
        "\xc4"
        "\xbe"
        "bermut",
        ",",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x83"
        "\xc4"
        "\xb8"
        "l",
        "\xc4"
        "\xa0"
        "und",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x83"
        "\xc4"
        "\xa6"
        "pfel",
        "." } },
    { "\xc3"
      "\x84"
      "RGER \xc3"
      "\x9c"
      "BER \xc3"
      "\x96"
      "STERREICH: GROSSE STRASSE",
      { "\xc3"
        "\x83"
        "\xc4"
        "\xa6"
        "RGER",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x83"
        "\xc4"
        "\xbe"
        "BER",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x83"
        "\xc4"
        "\xb8"
        "STERREICH",
        ":",
        "\xc4"
        "\xa0"
        "GROSSE",
        "\xc4"
        "\xa0"
        "STRASSE" } },
    { "camelCaseIdentifier XMLHttpRequest HTTPServer parseJSONValue",
      { "camel", "Case", "Identifier",
        "\xc4"
        "\xa0"
        "XMLHttp",
        "Request",
        "\xc4"
        "\xa0"
        "HTTPServer",
        "\xc4"
        "\xa0"
        "parse",
        "JSONValue" } },
    { "don't can't I'M He'Ll they've we're it'd",
      { "don't",
        "\xc4"
        "\xa0"
        "can't",
        "\xc4"
        "\xa0"
        "I'M",
        "\xc4"
        "\xa0"
        "He'Ll",
        "\xc4"
        "\xa0"
        "they've",
        "\xc4"
        "\xa0"
        "we're",
        "\xc4"
        "\xa0"
        "it'd" } },
    { "a\n\nb\tc\r\nd",
      { "a",
        "\xc4"
        "\x8a"
        "\xc4"
        "\x8a"
        "",
        "b",
        "\xc4"
        "\x89"
        "c",
        "\xc4"
        "\x8d"
        "\xc4"
        "\x8a"
        "",
        "d" } },
    { "multiple    spaces   and\ttabs",
      { "multiple",
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "",
        "\xc4"
        "\xa0"
        "spaces",
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "",
        "\xc4"
        "\xa0"
        "and",
        "\xc4"
        "\x89"
        "tabs" } },
    { "def f(x):\n    if x > 0:\n        return x/2\n    return 0\n",
      { "def",
        "\xc4"
        "\xa0"
        "f",
        "(x",
        "):\xc4"
        "\x8a"
        "",
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "",
        "\xc4"
        "\xa0"
        "if",
        "\xc4"
        "\xa0"
        "x",
        "\xc4"
        "\xa0"
        ">",
        "\xc4"
        "\xa0"
        "",
        "0",
        ":\xc4"
        "\x8a"
        "",
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "",
        "\xc4"
        "\xa0"
        "return",
        "\xc4"
        "\xa0"
        "x",
        "/",
        "2",
        "\xc4"
        "\x8a"
        "",
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "",
        "\xc4"
        "\xa0"
        "return",
        "\xc4"
        "\xa0"
        "",
        "0",
        "\xc4"
        "\x8a"
        "" } },
    { "Zahlen: 1 12 123 1234 12345 007 3.14159",
      { "Zahlen",
        ":",
        "\xc4"
        "\xa0"
        "",
        "1",
        "\xc4"
        "\xa0"
        "",
        "12",
        "\xc4"
        "\xa0"
        "",
        "123",
        "\xc4"
        "\xa0"
        "",
        "123",
        "4",
        "\xc4"
        "\xa0"
        "",
        "123",
        "45",
        "\xc4"
        "\xa0"
        "",
        "007",
        "\xc4"
        "\xa0"
        "",
        "3",
        ".",
        "141",
        "59" } },
    { "\xe2"
      "\x80"
      "\x9e"
      "Anf\xc3"
      "\xbc"
      "hrungszeichen\xe2"
      "\x80"
      "\x9c"
      " \xe2"
      "\x80"
      "\x93"
      " Halbgeviertstrich \xe2"
      "\x80"
      "\xa6"
      " \xe2"
      "\x80"
      "\x9a"
      "einfach' \xc2"
      "\xbb"
      "guillemets\xc2"
      "\xab"
      "",
      { "\xc3"
        "\xa2"
        "\xc4"
        "\xa2"
        "\xc5"
        "\x80"
        "Anf\xc3"
        "\x83"
        "\xc2"
        "\xbc"
        "hrungszeichen",
        "\xc3"
        "\xa2"
        "\xc4"
        "\xa2"
        "\xc4"
        "\xbe"
        "",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xa2"
        "\xc4"
        "\xa2"
        "\xc4"
        "\xb5"
        "",
        "\xc4"
        "\xa0"
        "Halbgeviertstrich",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xa2"
        "\xc4"
        "\xa2"
        "\xc2"
        "\xa6"
        "",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xa2"
        "\xc4"
        "\xa2"
        "\xc4"
        "\xbc"
        "",
        "einfach", "'",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x82"
        "\xc2"
        "\xbb"
        "",
        "guillemets",
        "\xc3"
        "\x82"
        "\xc2"
        "\xab"
        "" } },
    { "Donaudampfschifffahrtsgesellschaftskapitaen Rindfleischetikettierungsueberwachungsaufgabenuebertragungsgesetz",
      { "Donaudampfschifffahrtsgesellschaftskapitaen",
        "\xc4"
        "\xa0"
        "Rindfleischetikettierungsueberwachungsaufgabenuebertragungsgesetz" } },
    { "\xe6"
      "\x97"
      "\xa5"
      "\xe6"
      "\x9c"
      "\xac"
      "\xe8"
      "\xaa"
      "\x9e"
      "\xe3"
      "\x81"
      "\xae"
      "\xe3"
      "\x83"
      "\x86"
      "\xe3"
      "\x82"
      "\xad"
      "\xe3"
      "\x82"
      "\xb9"
      "\xe3"
      "\x83"
      "\x88"
      "\xe3"
      "\x81"
      "\xa7"
      "\xe3"
      "\x81"
      "\x99"
      "\xe3"
      "\x80"
      "\x82"
      "\xe4"
      "\xb8"
      "\xad"
      "\xe6"
      "\x96"
      "\x87"
      "\xe6"
      "\xb5"
      "\x8b"
      "\xe8"
      "\xaf"
      "\x95"
      "\xe3"
      "\x80"
      "\x82"
      "\xed"
      "\x95"
      "\x9c"
      "\xea"
      "\xb5"
      "\xad"
      "\xec"
      "\x96"
      "\xb4"
      " \xed"
      "\x85"
      "\x8c"
      "\xec"
      "\x8a"
      "\xa4"
      "\xed"
      "\x8a"
      "\xb8"
      ".",
      { "\xc3"
        "\xa6"
        "\xc4"
        "\xb9"
        "\xc2"
        "\xa5"
        "\xc3"
        "\xa6"
        "\xc4"
        "\xbe"
        "\xc2"
        "\xac"
        "\xc3"
        "\xa8"
        "\xc2"
        "\xaa"
        "\xc5"
        "\x80"
        "\xc3"
        "\xa3"
        "\xc4"
        "\xa3"
        "\xc2"
        "\xae"
        "\xc3"
        "\xa3"
        "\xc4"
        "\xa5"
        "\xc4"
        "\xa8"
        "\xc3"
        "\xa3"
        "\xc4"
        "\xa4"
        "\xc5"
        "\x83"
        "\xc3"
        "\xa3"
        "\xc4"
        "\xa4"
        "\xc2"
        "\xb9"
        "\xc3"
        "\xa3"
        "\xc4"
        "\xa5"
        "\xc4"
        "\xaa"
        "\xc3"
        "\xa3"
        "\xc4"
        "\xa3"
        "\xc2"
        "\xa7"
        "\xc3"
        "\xa3"
        "\xc4"
        "\xa3"
        "\xc4"
        "\xbb"
        "",
        "\xc3"
        "\xa3"
        "\xc4"
        "\xa2"
        "\xc4"
        "\xa4"
        "\xc3"
        "\xa4"
        "\xc2"
        "\xb8"
        "\xc5"
        "\x83"
        "\xc3"
        "\xa6"
        "\xc4"
        "\xb8"
        "\xc4"
        "\xa9"
        "\xc3"
        "\xa6"
        "\xc2"
        "\xb5"
        "\xc4"
        "\xad"
        "\xc3"
        "\xa8"
        "\xc2"
        "\xaf"
        "\xc4"
        "\xb7"
        "",
        "\xc3"
        "\xa3"
        "\xc4"
        "\xa2"
        "\xc4"
        "\xa4"
        "\xc3"
        "\xad"
        "\xc4"
        "\xb7"
        "\xc4"
        "\xbe"
        "\xc3"
        "\xaa"
        "\xc2"
        "\xb5"
        "\xc5"
        "\x83"
        "\xc3"
        "\xac"
        "\xc4"
        "\xb8"
        "\xc2"
        "\xb4"
        "",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xad"
        "\xc4"
        "\xa7"
        "\xc4"
        "\xae"
        "\xc3"
        "\xac"
        "\xc4"
        "\xac"
        "\xc2"
        "\xa4"
        "\xc3"
        "\xad"
        "\xc4"
        "\xac"
        "\xc2"
        "\xb8"
        "",
        "." } },
    { "\xd0"
      "\x9f"
      "\xd1"
      "\x80"
      "\xd0"
      "\xb8"
      "\xd0"
      "\xb2"
      "\xd0"
      "\xb5"
      "\xd1"
      "\x82"
      ", \xd0"
      "\xbc"
      "\xd0"
      "\xb8"
      "\xd1"
      "\x80"
      "! \xd0"
      "\x9c"
      "\xd0"
      "\x9e"
      "\xd0"
      "\xa1"
      "\xd0"
      "\x9a"
      "\xd0"
      "\x92"
      "\xd0"
      "\x90"
      " \xd0"
      "\xb8"
      " \xd0"
      "\xa1"
      "\xd0"
      "\xb0"
      "\xd0"
      "\xbd"
      "\xd0"
      "\xba"
      "\xd1"
      "\x82"
      "-\xd0"
      "\x9f"
      "\xd0"
      "\xb5"
      "\xd1"
      "\x82"
      "\xd0"
      "\xb5"
      "\xd1"
      "\x80"
      "\xd0"
      "\xb1"
      "\xd1"
      "\x83"
      "\xd1"
      "\x80"
      "\xd0"
      "\xb3"
      ".",
      { "\xc3"
        "\x90"
        "\xc5"
        "\x81"
        "\xc3"
        "\x91"
        "\xc4"
        "\xa2"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb8"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb2"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb5"
        "\xc3"
        "\x91"
        "\xc4"
        "\xa4"
        "",
        ",",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x90"
        "\xc2"
        "\xbc"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb8"
        "\xc3"
        "\x91"
        "\xc4"
        "\xa2"
        "",
        "!",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x90"
        "\xc4"
        "\xbe"
        "\xc3"
        "\x90"
        "\xc5"
        "\x80"
        "\xc3"
        "\x90"
        "\xc2"
        "\xa1"
        "\xc3"
        "\x90"
        "\xc4"
        "\xbc"
        "\xc3"
        "\x90"
        "\xc4"
        "\xb4"
        "\xc3"
        "\x90"
        "\xc4"
        "\xb2"
        "",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb8"
        "",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x90"
        "\xc2"
        "\xa1"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb0"
        "\xc3"
        "\x90"
        "\xc2"
        "\xbd"
        "\xc3"
        "\x90"
        "\xc2"
        "\xba"
        "\xc3"
        "\x91"
        "\xc4"
        "\xa4"
        "",
        "-\xc3"
        "\x90"
        "\xc5"
        "\x81"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb5"
        "\xc3"
        "\x91"
        "\xc4"
        "\xa4"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb5"
        "\xc3"
        "\x91"
        "\xc4"
        "\xa2"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb1"
        "\xc3"
        "\x91"
        "\xc4"
        "\xa5"
        "\xc3"
        "\x91"
        "\xc4"
        "\xa2"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb3"
        "",
        "." } },
    { "emoji \xf0"
      "\x9f"
      "\x9a"
      "\x80"
      "\xf0"
      "\x9f"
      "\x87"
      "\xa9"
      "\xf0"
      "\x9f"
      "\x87"
      "\xaa"
      " test \xe2"
      "\x98"
      "\x95"
      "\xef"
      "\xb8"
      "\x8f"
      "",
      { "emoji",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xb0"
        "\xc5"
        "\x81"
        "\xc4"
        "\xbc"
        "\xc4"
        "\xa2"
        "\xc3"
        "\xb0"
        "\xc5"
        "\x81"
        "\xc4"
        "\xa9"
        "\xc2"
        "\xa9"
        "\xc3"
        "\xb0"
        "\xc5"
        "\x81"
        "\xc4"
        "\xa9"
        "\xc2"
        "\xaa"
        "",
        "\xc4"
        "\xa0"
        "test",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xa2"
        "\xc4"
        "\xba"
        "\xc4"
        "\xb7"
        "\xc3"
        "\xaf"
        "\xc2"
        "\xb8"
        "\xc4"
        "\xb1"
        "" } },
    { "path/to/file.txt and http://example.com/a/b?c=1&d=2",
      { "path", "/to", "/file", ".txt",
        "\xc4"
        "\xa0"
        "and",
        "\xc4"
        "\xa0"
        "http",
        "://", "example", ".com", "/a", "/b", "?c", "=", "1", "&d", "=", "2" } },
    { "  leading and trailing   ",
      { "\xc4"
        "\xa0"
        "",
        "\xc4"
        "\xa0"
        "leading",
        "\xc4"
        "\xa0"
        "and",
        "\xc4"
        "\xa0"
        "trailing",
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "" } },
    { "na\xc3"
      "\xaf"
      "ve caf\xc3"
      "\xa9"
      " r\xc3"
      "\xa9"
      "sum\xc3"
      "\xa9"
      " \xc3"
      "\x89"
      "COLE",
      { "na\xc3"
        "\x83"
        "\xc2"
        "\xaf"
        "ve",
        "\xc4"
        "\xa0"
        "caf\xc3"
        "\x83"
        "\xc2"
        "\xa9"
        "",
        "\xc4"
        "\xa0"
        "r\xc3"
        "\x83"
        "\xc2"
        "\xa9"
        "sum\xc3"
        "\x83"
        "\xc2"
        "\xa9"
        "",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x83"
        "\xc4"
        "\xab"
        "COLE" } },
    { "x=1;y=2//comment\n/* block */\n",
      { "x", "=", "1", ";y", "=", "2", "//", "comment",
        "\xc4"
        "\x8a"
        "",
        "/*",
        "\xc4"
        "\xa0"
        "block",
        "\xc4"
        "\xa0"
        "*/\xc4"
        "\x8a"
        "" } },
    { "\n\n\n",
      { "\xc4"
        "\x8a"
        "\xc4"
        "\x8a"
        "\xc4"
        "\x8a"
        "" } },
    { "1234567890", { "123", "456", "789", "0" } },
    { "\xc3"
      "\x84"
      "rzteschaft's \xc3"
      "\x9c"
      "bung's",
      { "\xc3"
        "\x83"
        "\xc4"
        "\xa6"
        "rzteschaft's",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x83"
        "\xc4"
        "\xbe"
        "bung's" } },
    { "Das kostet 1.234,56 \xe2"
      "\x82"
      "\xac"
      " statt \xc2"
      "\xa3"
      "999 oder \xc2"
      "\xa5"
      "120000 \xe2"
      "\x80"
      "\x93"
      " ein Schn\xc3"
      "\xa4"
      "ppchen!",
      { "Das",
        "\xc4"
        "\xa0"
        "kostet",
        "\xc4"
        "\xa0"
        "",
        "1",
        ".",
        "234",
        ",",
        "56",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xa2"
        "\xc4"
        "\xa4"
        "\xc2"
        "\xac"
        "",
        "\xc4"
        "\xa0"
        "statt",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x82"
        "\xc2"
        "\xa3"
        "",
        "999",
        "\xc4"
        "\xa0"
        "oder",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x82"
        "\xc2"
        "\xa5"
        "",
        "120",
        "000",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xa2"
        "\xc4"
        "\xa2"
        "\xc4"
        "\xb5"
        "",
        "\xc4"
        "\xa0"
        "ein",
        "\xc4"
        "\xa0"
        "Schn\xc3"
        "\x83"
        "\xc2"
        "\xa4"
        "ppchen",
        "!" } },
    { "sagte \xe2"
      "\x80"
      "\x9e"
      "Hallo\xe2"
      "\x80"
      "\x9c"
      " heute und \xc2"
      "\xbb"
      "Tsch\xc3"
      "\xbc"
      "ss\xc2"
      "\xab"
      " morgen",
      { "sagte",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xa2"
        "\xc4"
        "\xa2"
        "\xc5"
        "\x80"
        "",
        "Hallo",
        "\xc3"
        "\xa2"
        "\xc4"
        "\xa2"
        "\xc4"
        "\xbe"
        "",
        "\xc4"
        "\xa0"
        "heute",
        "\xc4"
        "\xa0"
        "und",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x82"
        "\xc2"
        "\xbb"
        "",
        "Tsch\xc3"
        "\x83"
        "\xc2"
        "\xbc"
        "ss",
        "\xc3"
        "\x82"
        "\xc2"
        "\xab"
        "",
        "\xc4"
        "\xa0"
        "morgen" } },
    { "NBSP:\xc2"
      "\xa0"
      "hier\xc2"
      "\xa0"
      "dort  soft\xc2"
      "\xad"
      "hyphen   100\xc2"
      "\xa0"
      "%",
      { "NBSP", ":",
        "\xc3"
        "\x82"
        "\xc5"
        "\x82"
        "hier",
        "\xc3"
        "\x82"
        "\xc5"
        "\x82"
        "dort",
        "\xc4"
        "\xa0"
        "",
        "\xc4"
        "\xa0"
        "soft",
        "\xc3"
        "\x82"
        "\xc5"
        "\x83"
        "hyphen",
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "",
        "\xc4"
        "\xa0"
        "",
        "100",
        "\xc3"
        "\x82"
        "\xc5"
        "\x82"
        "",
        "%" } },
    { "\xc3"
      "\x84"
      "hnlich: \xc3"
      "\x96"
      "PNV, \xc3"
      "\x9c"
      "STRA, \xe1"
      "\xba"
      "\x9e"
      " und \xc3"
      "\x9f"
      ", \xc5"
      "\x92"
      "uvre, \xc3"
      "\x86"
      "r\xc3"
      "\xb8"
      "",
      { "\xc3"
        "\x83"
        "\xc4"
        "\xa6"
        "hnlich",
        ":",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x83"
        "\xc4"
        "\xb8"
        "PNV",
        ",",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x83"
        "\xc4"
        "\xbe"
        "STRA",
        ",",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xa1"
        "\xc2"
        "\xba"
        "\xc5"
        "\x80"
        "",
        "\xc4"
        "\xa0"
        "und",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x83"
        "\xc5"
        "\x81"
        "",
        ",",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x85"
        "\xc4"
        "\xb4"
        "uvre",
        ",",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x83"
        "\xc4"
        "\xa8"
        "r\xc3"
        "\x83"
        "\xc2"
        "\xb8"
        "" } },
    { "MwSt. 19 % \xc2"
      "\xb7"
      " Art.-Nr. 4711/A \xc2"
      "\xb7"
      " \xc2"
      "\xb1"
      "0,5 \xc2"
      "\xb0"
      "C \xc2"
      "\xb7"
      " \xc2"
      "\xbd"
      " Liter",
      { "Mw",
        "St",
        ".",
        "\xc4"
        "\xa0"
        "",
        "19",
        "\xc4"
        "\xa0"
        "%",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x82"
        "\xc2"
        "\xb7"
        "",
        "\xc4"
        "\xa0"
        "Art",
        ".-",
        "Nr",
        ".",
        "\xc4"
        "\xa0"
        "",
        "471",
        "1",
        "/A",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x82"
        "\xc2"
        "\xb7"
        "",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x82"
        "\xc2"
        "\xb1"
        "",
        "0",
        ",",
        "5",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x82"
        "\xc2"
        "\xb0"
        "",
        "C",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x82"
        "\xc2"
        "\xb7"
        "",
        "\xc4"
        "\xa0"
        "",
        "\xc3"
        "\x82"
        "\xc2"
        "\xbd"
        "",
        "\xc4"
        "\xa0"
        "Liter" } },
    { "Stra\xc3"
      "\x9f"
      "e\xe2"
      "\x80"
      "\xa8"
      "Zeile",
      { "Stra\xc3"
        "\x83"
        "\xc5"
        "\x81"
        "e",
        "\xc3"
        "\xa2"
        "\xc4"
        "\xa2"
        "\xc2"
        "\xa8"
        "Zeile" } },
    { "5G-Netz iPhone15Pro Version2.0.1 IPv6",
      { "5", "G", "-Netz",
        "\xc4"
        "\xa0"
        "i",
        "Phone", "15", "Pro",
        "\xc4"
        "\xa0"
        "Version",
        "2", ".", "0", ".", "1",
        "\xc4"
        "\xa0"
        "IPv",
        "6" } },
};

// Unicode-category spot checks. These pin the exact class boundaries the
// alternation branches on; a table regenerated against the wrong category set
// (or an "all non-ASCII is a letter" shortcut) fails here before any split does.
struct CatCase {
    uint32_t cp;
    uint8_t cat;
    const char * what;
};

static const std::vector<CatCase> k_cats = {
    { 'A', core_unicode::CAT_LU, "ASCII A" },
    { 'z', core_unicode::CAT_LL, "ASCII z" },
    { '7', core_unicode::CAT_N, "ASCII 7" },
    { ' ', core_unicode::CAT_WS, "ASCII space" },
    { '\n', core_unicode::CAT_WS, "newline" },
    { '-', core_unicode::CAT_P, "hyphen-minus" },
    { 0x00C4, core_unicode::CAT_LU, "A-umlaut" },
    { 0x00E4, core_unicode::CAT_LL, "a-umlaut" },
    { 0x00DF, core_unicode::CAT_LL, "sharp s" },
    { 0x00A0, core_unicode::CAT_WS, "no-break space" },
    { 0x00D7, core_unicode::CAT_P, "multiplication sign (NOT a letter)" },
    { 0x2013, core_unicode::CAT_P, "en dash" },
    { 0x201E, core_unicode::CAT_P, "low-9 quote" },
    { 0x0410, core_unicode::CAT_LU, "Cyrillic A" },
    { 0x0430, core_unicode::CAT_LL, "Cyrillic a" },
    { 0x0391, core_unicode::CAT_LU, "Greek Alpha" },
    { 0x03B1, core_unicode::CAT_LL, "Greek alpha" },
    { 0x4E2D, core_unicode::CAT_LO, "CJK zhong" },
    { 0x3042, core_unicode::CAT_LO, "Hiragana a" },
    { 0x0301, core_unicode::CAT_M, "combining acute" },
    { 0x00B2, core_unicode::CAT_N, "superscript two" },
    { 0x0660, core_unicode::CAT_N, "Arabic-Indic zero" },
    { 0x1F680, core_unicode::CAT_P, "rocket emoji (So)" },
    { 0x00AD, core_unicode::CAT_C, "soft hyphen (Cf)" },
    { 0x2028, core_unicode::CAT_WS, "line separator" },
    { 0x01C5, core_unicode::CAT_LU, "titlecase Dz" },
};

static int crispembed_test_main() {
    int failures = 0;
    int checked = 0;

    for (const auto & c : k_cats) {
        checked++;
        const uint8_t got = core_unicode::category(c.cp);
        if (got == c.cat) continue;
        failures++;
        fprintf(stderr, "FAIL: category(U+%04X) [%s] want %u got %u\n", c.cp, c.what, c.cat, got);
    }

    for (const auto & c : k_cases) {
        // o200k_pretokenize returns raw substrings; the golden splits are the
        // byte-level-encoded form, so encode before comparing.
        std::vector<std::string> got;
        for (const auto & pt : core_bpe::o200k_pretokenize(c.text))
            got.push_back(core_bpe::bytes_to_unicode(pt.data(), pt.size()));
        checked++;
        if (got == c.splits) continue;
        failures++;
        fprintf(stderr, "FAIL: %s\n  want(%zu):", c.text, c.splits.size());
        for (const auto & s : c.splits) fprintf(stderr, " [%s]", s.c_str());
        fprintf(stderr, "\n  got (%zu):", got.size());
        for (const auto & s : got) fprintf(stderr, " [%s]", s.c_str());
        fprintf(stderr, "\n");
    }

    // The empty string must yield no pre-tokens at all.
    if (!core_bpe::o200k_pretokenize("").empty()) {
        fprintf(stderr, "FAIL: empty input produced pre-tokens\n");
        failures++;
    }
    checked++;

    // Invariant a typo cannot satisfy: concatenating the raw splits must
    // reproduce the input byte-for-byte (the regex partitions, never drops).
    for (const auto & c : k_cases) {
        std::string joined;
        for (const auto & pt : core_bpe::o200k_pretokenize(c.text)) joined += pt;
        checked++;
        if (joined != c.text) {
            fprintf(stderr, "FAIL: lossy split for [%s]\n  rejoined [%s]\n", c.text, joined.c_str());
            failures++;
        }
    }

    printf("test-o200k-pretokenize: %d checks, %d failures\n", checked, failures);
    return failures == 0 ? 0 : 1;
}

// tools/check_test_clean_exit.sh: a one-shot binary must not run ggml's
// static GPU-device destructor at exit (it aborts on Metal / faults on CUDA).
int main() {
    core_util::clean_exit(crispembed_test_main());
}
