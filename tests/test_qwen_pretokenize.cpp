// tests/test_qwen_pretokenize.cpp — Qwen2/Qwen3 ByteLevel pre-tokenizer parity.
//
// Hermetic: no vocab, no merges, no GGUF, no network. Pre-tokenization is pure
// string splitting, so the golden splits below are HuggingFace's own
// `tokenizer.pre_tokenizer.pre_tokenize_str()` output (byte-level encoded, so
// U+0120 'G-dot' is a space and U+010A 'C-dot' a newline), captured from
// codefuse-ai/F2LLM-v2-160M — the same regex every Qwen2/Qwen3 vocab declares.
//
// The guard exists because `core_bpe::tokenize_simple` split on whitespace and
// rejoined the runs with a single space, silently deleting newlines and
// indentation: "a\n\nb" and "a b" produced identical ids. Measured against the
// HF reference on F2LLM-v2-160M that cost cosine 0.9803 on a code snippet and
// 0.9907 on this model family's own "Instruct: ...\nQuery: " prompt, while
// newline-free text was unaffected at 1.000000 — i.e. exactly the inputs a
// retrieval workload actually sends. The cases below cover the alternations
// that defect hid: newline runs, indentation, punctuation+newline, digits
// (Qwen emits ONE token per digit), case-insensitive contractions, CJK,
// Cyrillic, emoji, and CRLF.
//
//   c++ -std=c++17 -O1 -Isrc tests/test_qwen_pretokenize.cpp -o /tmp/test-qwen-pretok
//   /tmp/test-qwen-pretok
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
    { "Instruct: Given a question, retrieve passages that can help answer the question.\nQuery: Wie hoch ist der Mount "
      "Everest?",
      { "Instruct",
        ":",
        "\xc4"
        "\xa0"
        "Given",
        "\xc4"
        "\xa0"
        "a",
        "\xc4"
        "\xa0"
        "question",
        ",",
        "\xc4"
        "\xa0"
        "retrieve",
        "\xc4"
        "\xa0"
        "passages",
        "\xc4"
        "\xa0"
        "that",
        "\xc4"
        "\xa0"
        "can",
        "\xc4"
        "\xa0"
        "help",
        "\xc4"
        "\xa0"
        "answer",
        "\xc4"
        "\xa0"
        "the",
        "\xc4"
        "\xa0"
        "question",
        ".\xc4"
        "\x8a"
        "",
        "Query",
        ":",
        "\xc4"
        "\xa0"
        "Wie",
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
    { "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1)",
      { "def",
        "\xc4"
        "\xa0"
        "fibonacci",
        "(n",
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
        "return",
        "\xc4"
        "\xa0"
        "n",
        "\xc4"
        "\xa0"
        "if",
        "\xc4"
        "\xa0"
        "n",
        "\xc4"
        "\xa0"
        "<",
        "\xc4"
        "\xa0"
        "",
        "2",
        "\xc4"
        "\xa0"
        "else",
        "\xc4"
        "\xa0"
        "fibonacci",
        "(n", "-", "1", ")" } },
    { "a\n\nb",
      { "a",
        "\xc4"
        "\x8a"
        "\xc4"
        "\x8a"
        "",
        "b" } },
    { "a\n\n\n  b",
      { "a",
        "\xc4"
        "\x8a"
        "\xc4"
        "\x8a"
        "\xc4"
        "\x8a"
        "",
        "\xc4"
        "\xa0"
        "",
        "\xc4"
        "\xa0"
        "b" } },
    { "  leading and trailing  ",
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
        "" } },
    { "tabs\tand\t\tmore",
      { "tabs",
        "\xc4"
        "\x89"
        "and",
        "\xc4"
        "\x89"
        "",
        "\xc4"
        "\x89"
        "more" } },
    { "Quarterly revenue grew by 12% while costs 2026 stayed flat",
      { "Quarterly",
        "\xc4"
        "\xa0"
        "revenue",
        "\xc4"
        "\xa0"
        "grew",
        "\xc4"
        "\xa0"
        "by",
        "\xc4"
        "\xa0"
        "",
        "1", "2", "%",
        "\xc4"
        "\xa0"
        "while",
        "\xc4"
        "\xa0"
        "costs",
        "\xc4"
        "\xa0"
        "",
        "2", "0", "2", "6",
        "\xc4"
        "\xa0"
        "stayed",
        "\xc4"
        "\xa0"
        "flat" } },
    { "don't DON'T can't THEY'RE we've I'll he'd",
      { "don", "'t",
        "\xc4"
        "\xa0"
        "DON",
        "'T",
        "\xc4"
        "\xa0"
        "can",
        "'t",
        "\xc4"
        "\xa0"
        "THEY",
        "'RE",
        "\xc4"
        "\xa0"
        "we",
        "'ve",
        "\xc4"
        "\xa0"
        "I",
        "'ll",
        "\xc4"
        "\xa0"
        "he",
        "'d" } },
    { "Die Katze schl\xc3"
      "\xa4"
      "ft; der Hund l\xc3"
      "\xa4"
      "uft \xe2"
      "\x80"
      "\x94"
      " schnell!",
      { "Die",
        "\xc4"
        "\xa0"
        "Katze",
        "\xc4"
        "\xa0"
        "schl\xc3"
        "\x83"
        "\xc2"
        "\xa4"
        "ft",
        ";",
        "\xc4"
        "\xa0"
        "der",
        "\xc4"
        "\xa0"
        "Hund",
        "\xc4"
        "\xa0"
        "l\xc3"
        "\x83"
        "\xc2"
        "\xa4"
        "uft",
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
        "schnell",
        "!" } },
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
      " \xd0"
      "\xbc"
      "\xd0"
      "\xb8"
      "\xd1"
      "\x80"
      ", \xd0"
      "\xba"
      "\xd0"
      "\xb0"
      "\xd0"
      "\xba"
      " \xd0"
      "\xb4"
      "\xd0"
      "\xb5"
      "\xd0"
      "\xbb"
      "\xd0"
      "\xb0"
      "?",
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
        ",",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x90"
        "\xc2"
        "\xba"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb0"
        "\xc3"
        "\x90"
        "\xc2"
        "\xba"
        "",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb4"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb5"
        "\xc3"
        "\x90"
        "\xc2"
        "\xbb"
        "\xc3"
        "\x90"
        "\xc2"
        "\xb0"
        "",
        "?" } },
    { "\xe4"
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
      "\xe6"
      "\x96"
      "\x87"
      "\xe6"
      "\x9c"
      "\xac"
      "\xe4"
      "\xb8"
      "\x80"
      "\xe4"
      "\xba"
      "\x8c"
      "\xe4"
      "\xb8"
      "\x89"
      "",
      { "\xc3"
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
        "\xc3"
        "\xa6"
        "\xc4"
        "\xb8"
        "\xc4"
        "\xa9"
        "\xc3"
        "\xa6"
        "\xc4"
        "\xbe"
        "\xc2"
        "\xac"
        "\xc3"
        "\xa4"
        "\xc2"
        "\xb8"
        "\xc4"
        "\xa2"
        "\xc3"
        "\xa4"
        "\xc2"
        "\xba"
        "\xc4"
        "\xae"
        "\xc3"
        "\xa4"
        "\xc2"
        "\xb8"
        "\xc4"
        "\xab"
        "" } },
    { "emoji \xf0"
      "\x9f"
      "\x9a"
      "\x80"
      " test \xf0"
      "\x9f"
      "\x91"
      "\x8d"
      "\xf0"
      "\x9f"
      "\x8f"
      "\xbd"
      " done",
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
        "",
        "\xc4"
        "\xa0"
        "test",
        "\xc4"
        "\xa0"
        "\xc3"
        "\xb0"
        "\xc5"
        "\x81"
        "\xc4"
        "\xb3"
        "\xc4"
        "\xaf"
        "\xc3"
        "\xb0"
        "\xc5"
        "\x81"
        "\xc4"
        "\xb1"
        "\xc2"
        "\xbd"
        "",
        "\xc4"
        "\xa0"
        "done" } },
    { "#include <stdio.h>\r\nint main(void) { return 0; }",
      { "#include",
        "\xc4"
        "\xa0"
        "<",
        "stdio", ".h",
        ">\xc4"
        "\x8d"
        "\xc4"
        "\x8a"
        "",
        "int",
        "\xc4"
        "\xa0"
        "main",
        "(void", ")",
        "\xc4"
        "\xa0"
        "{",
        "\xc4"
        "\xa0"
        "return",
        "\xc4"
        "\xa0"
        "",
        "0", ";",
        "\xc4"
        "\xa0"
        "}" } },
    { "trailing newline\n",
      { "trailing",
        "\xc4"
        "\xa0"
        "newline",
        "\xc4"
        "\x8a"
        "" } },
    { "\n\nleading newlines",
      { "\xc4"
        "\x8a"
        "\xc4"
        "\x8a"
        "",
        "leading",
        "\xc4"
        "\xa0"
        "newlines" } },
    { "multi  space   run",
      { "multi",
        "\xc4"
        "\xa0"
        "",
        "\xc4"
        "\xa0"
        "space",
        "\xc4"
        "\xa0"
        "\xc4"
        "\xa0"
        "",
        "\xc4"
        "\xa0"
        "run" } },
    { "x=1;y=2;z=x+y", { "x", "=", "1", ";y", "=", "2", ";z", "=x", "+y" } },
    { "\xe2"
      "\x82"
      "\xac"
      "100 costs $50 and \xc2"
      "\xa3"
      "20",
      { "\xc3"
        "\xa2"
        "\xc4"
        "\xa4"
        "\xc2"
        "\xac"
        "",
        "1", "0", "0",
        "\xc4"
        "\xa0"
        "costs",
        "\xc4"
        "\xa0"
        "$",
        "5", "0",
        "\xc4"
        "\xa0"
        "and",
        "\xc4"
        "\xa0"
        "\xc3"
        "\x82"
        "\xc2"
        "\xa3"
        "",
        "2", "0" } },
    { "a\r\nb\rc",
      { "a",
        "\xc4"
        "\x8d"
        "\xc4"
        "\x8a"
        "",
        "b",
        "\xc4"
        "\x8d"
        "",
        "c" } },
};

static int crispembed_test_main() {
    int failures = 0;
    int checked = 0;
    for (const auto & c : k_cases) {
        // qwen_pretokenize returns raw substrings; the golden splits are the
        // byte-level-encoded form, so encode before comparing.
        std::vector<std::string> got;
        for (const auto & pt : core_bpe::qwen_pretokenize(c.text))
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
    if (!core_bpe::qwen_pretokenize("").empty()) {
        fprintf(stderr, "FAIL: empty input produced pre-tokens\n");
        failures++;
    }
    checked++;

    // Invariant a typo cannot satisfy: concatenating the raw splits must
    // reproduce the input byte-for-byte (the regex partitions, never drops).
    for (const auto & c : k_cases) {
        std::string joined;
        for (const auto & pt : core_bpe::qwen_pretokenize(c.text)) joined += pt;
        checked++;
        if (joined != c.text) {
            fprintf(stderr, "FAIL: lossy split for [%s]\n  rejoined [%s]\n", c.text, joined.c_str());
            failures++;
        }
    }

    printf("test-qwen-pretokenize: %d checks, %d failures\n", checked, failures);
    return failures == 0 ? 0 : 1;
}

// tools/check_test_clean_exit.sh: a one-shot binary must not run ggml's
// static GPU-device destructor at exit (it aborts on Metal / faults on CUDA).
int main() {
    core_util::clean_exit(crispembed_test_main());
}
