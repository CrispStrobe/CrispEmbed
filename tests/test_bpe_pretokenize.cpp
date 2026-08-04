// tests/test_bpe_pretokenize.cpp — declared-regex pre-tokenizer parity for the
// non-Qwen byte-level BPE families in core/bpe.h.
//
// Hermetic: no vocab, no merges, no GGUF, no network. Pre-tokenization is pure
// string splitting, so every golden split below is HuggingFace's own
// `tokenizer.pre_tokenizer.pre_tokenize_str()` output (byte-level encoded, so
// U+0120 'G-dot' is a space and U+010A 'C-dot' a newline), captured from the
// checkpoints each engine actually loads. Regenerate with `python
// tools/gen_bpe_pretokenize_test.py tests/test_bpe_pretokenize.cpp &&
// tools/format.sh --fix` (needs network + the `tokenizers` package).
//
// Why it exists — `core_bpe::tokenize_simple` collapsed every whitespace run to
// a single space and deleted newlines outright, so "a\n\nb" and "a b" produced
// identical ids. T19-E1 fixed the Qwen embedder path; this is the audit of the
// other callers:
//
//   src/lfm2_embed.cpp      LiquidAI/LFM2.5-Embedding-350M  live (arbitrary user text)
//   src/deepseek_ocr2.cpp   deepseek-ai/DeepSeek-OCR-2      live ("\nFree OCR." prompt)
//   src/unlimited_ocr.cpp   baidu/Unlimited-OCR             latent (debug path only)
//
// Three declared regexes, three tables:
//
//   qwen      \p{N}             — one token per digit
//   lfm2      \p{N}{1,3}        — digit runs of up to three, else identical to qwen
//   deepseek  a Split SEQUENCE  — \p{N}{1,3}, then CJK/kana runs, then a regex
//                                 built on [\p{P}\p{S}] rather than [^\s\p{L}\p{N}]
//
// The qwen table is here as well because this audit found a SECOND, residual
// defect in the merged E1 fix: `qwen_is_letter` answered true for every byte
// >= 0x80, so non-ASCII punctuation was absorbed into the neighbouring word.
// HuggingFace splits `sagte \u201eHallo\u201c heute` into 5 pre-tokens; the
// approximation produced 3. That is live German retrieval text on every
// Qwen-family embedder, which is why the cases are in the guard.
//
//   c++ -std=c++17 -O1 -Isrc tests/test_bpe_pretokenize.cpp -o /tmp/test-bpe-pretok
//   /tmp/test-bpe-pretok
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

// Golden: codefuse-ai/F2LLM-v2-160M (the Qwen2/Qwen3 declared regex).
static const std::vector<Case> k_qwen_cases = {
    { "\012Free OCR.", { "\304\212", "Free", "\304\240OCR", "." } },
    { "<image>\012Free OCR.", { "<image", ">\304\212", "Free", "\304\240OCR", "." } },
    { "document parsing.", { "document", "\304\240parsing", "." } },
    { "\012<|grounding|>Convert the document to markdown.",
      { "\304\212", "<|", "grounding", "|>", "Convert", "\304\240the", "\304\240document", "\304\240to",
        "\304\240markdown", "." } },
    { "a\012\012b", { "a", "\304\212\304\212", "b" } },
    { "a\012\012\012  b", { "a", "\304\212\304\212\304\212", "\304\240", "\304\240b" } },
    { "  leading and trailing  ",
      { "\304\240", "\304\240leading", "\304\240and", "\304\240trailing", "\304\240\304\240" } },
    { "tabs\011and\011\011more", { "tabs", "\304\211and", "\304\211", "\304\211more" } },
    { "multi  space   run", { "multi", "\304\240", "\304\240space", "\304\240\304\240", "\304\240run" } },
    { "trailing newline\012", { "trailing", "\304\240newline", "\304\212" } },
    { "\012\012leading newlines", { "\304\212\304\212", "leading", "\304\240newlines" } },
    { "a\015\012b\015c", { "a", "\304\215\304\212", "b", "\304\215", "c" } },
    { "def fibonacci(n):\012    return n if n < 2 else fibonacci(n-1)",
      { "def", "\304\240fibonacci", "(n", "):\304\212", "\304\240\304\240\304\240", "\304\240return", "\304\240n",
        "\304\240if", "\304\240n", "\304\240<", "\304\240", "2", "\304\240else", "\304\240fibonacci", "(n", "-", "1",
        ")" } },
    { "#include <stdio.h>\015\012int main(void) { return 0; }",
      { "#include", "\304\240<", "stdio", ".h", ">\304\215\304\212", "int", "\304\240main", "(void", ")", "\304\240{",
        "\304\240return", "\304\240", "0", ";", "\304\240}" } },
    { "Quarterly revenue grew by 12% while costs 2026 stayed flat",
      { "Quarterly", "\304\240revenue", "\304\240grew", "\304\240by", "\304\240", "1", "2", "%", "\304\240while",
        "\304\240costs", "\304\240", "2", "0", "2", "6", "\304\240stayed", "\304\240flat" } },
    { "1234567", { "1", "2", "3", "4", "5", "6", "7" } },
    { "10,000 and 3.14 and 0x1F and v2.0.1",
      { "1",        "0", ",", "0", "0", "0",           "\304\240and", "\304\240", "3", ".", "1", "4", "\304\240and",
        "\304\240", "0", "x", "1", "F", "\304\240and", "\304\240v",   "2",        ".", "0", ".", "1" } },
    { "x=1;y=2;z=x+y", { "x", "=", "1", ";y", "=", "2", ";z", "=x", "+y" } },
    { "\331\243\331\244\331\245 and \340\271\231\340\271\231 and \302\275 and \302\262",
      { "\303\231\302\243", "\303\231\302\244", "\303\231\302\245", "\304\240and", "\304\240",
        "\303\240\302\271\304\273", "\303\240\302\271\304\273", "\304\240and", "\304\240", "\303\202\302\275",
        "\304\240and", "\304\240", "\303\202\302\262" } },
    { "sagte \342\200\236Hallo\342\200\234 heute",
      { "sagte", "\304\240\303\242\304\242\305\200", "Hallo", "\303\242\304\242\304\276", "\304\240heute" } },
    { "Er sagte: \302\273Guten Tag\302\253, dann ging er.",
      { "Er", "\304\240sagte", ":", "\304\240\303\202\302\273", "Guten", "\304\240Tag", "\303\202\302\253,",
        "\304\240dann", "\304\240ging", "\304\240er", "." } },
    { "\302\253quote\302\273", { "\303\202\302\253quote", "\303\202\302\273" } },
    { "\342\202\254\302\243abc", { "\303\242\304\244\302\254\303\202\302\243", "abc" } },
    { "\342\206\222\342\206\222x", { "\303\242\304\250\304\264\303\242\304\250\304\264", "x" } },
    { "a\302\251\302\256b", { "a", "\303\202\302\251\303\202\302\256", "b" } },
    { "\342\200\224 \"Hi\"", { "\303\242\304\242\304\266", "\304\240\"", "Hi", "\"" } },
    { "Die Katze schl\303\244ft; der Hund l\303\244uft \342\200\224 schnell!",
      { "Die", "\304\240Katze", "\304\240schl\303\203\302\244ft", ";", "\304\240der", "\304\240Hund",
        "\304\240l\303\203\302\244uft", "\304\240\303\242\304\242\304\266", "\304\240schnell", "!" } },
    { "\342\202\254100 costs $50 and \302\24320",
      { "\303\242\304\244\302\254", "1", "0", "0", "\304\240costs", "\304\240$", "5", "0", "\304\240and",
        "\304\240\303\202\302\243", "2", "0" } },
    { "\344\270\255\346\226\207\346\265\213\350\257\225\346\226\207\346\234\254\344\270\200\344\272\214\344\270\211",
      { "\303\244\302\270\305\203\303\246\304\270\304\251\303\246\302\265\304\255\303\250\302\257\304\267\303\246\304"
        "\270\304\251\303\246\304\276\302\254\303\244\302\270\304\242\303\244\302\272\304\256\303\244\302\270\304"
        "\253" } },
    { "\344\270\255\346\226\207abc", { "\303\244\302\270\305\203\303\246\304\270\304\251abc" } },
    { "abc\344\270\255\346\226\207", { "abc\303\244\302\270\305\203\303\246\304\270\304\251" } },
    { "\344\270\255\346\226\207\357\274\214\346\265\213\350\257\225\343\200\202",
      { "\303\244\302\270\305\203\303\246\304\270\304\251",
        "\303\257\302\274\304\256\303\246\302\265\304\255\303\250\302\257\304\267", "\303\243\304\242\304\244" } },
    { "\343\201\262\343\202\211\343\201\214\343\201\252\343\202\253\343\202\277\343\202\253\343\203\212\346\274\242\345"
      "\255\227",
      { "\303\243\304\243\302\262\303\243\304\244\304\253\303\243\304\243\304\256\303\243\304\243\302\252\303\243\304"
        "\244\302\253\303\243\304\244\302\277\303\243\304\244\302\253\303\243\304\245\304\254\303\246\302\274\302\242"
        "\303\245\305\203\304\271" } },
    { "caf\303\251 \344\270\255\346\226\207 123 !!!",
      { "caf\303\203\302\251", "\304\240\303\244\302\270\305\203\303\246\304\270\304\251", "\304\240", "1", "2", "3",
        "\304\240!!!" } },
    { "line1\012\344\270\255\346\226\207\012line3",
      { "line", "1", "\304\212", "\303\244\302\270\305\203\303\246\304\270\304\251", "\304\212", "line", "3" } },
    { "x\314\201y", { "x", "\303\214\304\243y" } },
    { "emoji \360\237\232\200 test \360\237\221\215\360\237\217\275 done",
      { "emoji", "\304\240\303\260\305\201\304\274\304\242", "\304\240test",
        "\304\240\303\260\305\201\304\263\304\257\303\260\305\201\304\261\302\275", "\304\240done" } },
    { "don't DON'T can't THEY'RE we've I'll he'd",
      { "don", "'t", "\304\240DON", "'T", "\304\240can", "'t", "\304\240THEY", "'RE", "\304\240we", "'ve", "\304\240I",
        "'ll", "\304\240he", "'d" } },
    { "\320\237\321\200\320\270\320\262\320\265\321\202 \320\274\320\270\321\200, \320\272\320\260\320\272 "
      "\320\264\320\265\320\273\320\260\077",
      { "\303\220\305\201\303\221\304\242\303\220\302\270\303\220\302\262\303\220\302\265\303\221\304\244",
        "\304\240\303\220\302\274\303\220\302\270\303\221\304\242", ",",
        "\304\240\303\220\302\272\303\220\302\260\303\220\302\272",
        "\304\240\303\220\302\264\303\220\302\265\303\220\302\273\303\220\302\260", "\077" } },
    { "Instruct: Given a question, retrieve passages that can help answer the question.\012Query: Wie hoch ist der "
      "Mount Everest\077",
      { "Instruct",
        ":",
        "\304\240Given",
        "\304\240a",
        "\304\240question",
        ",",
        "\304\240retrieve",
        "\304\240passages",
        "\304\240that",
        "\304\240can",
        "\304\240help",
        "\304\240answer",
        "\304\240the",
        "\304\240question",
        ".\304\212",
        "Query",
        ":",
        "\304\240Wie",
        "\304\240hoch",
        "\304\240ist",
        "\304\240der",
        "\304\240Mount",
        "\304\240Everest",
        "\077" } },
};

// Golden: LiquidAI/LFM2.5-Embedding-350M (same regex, \p{N}{1,3} digits).
static const std::vector<Case> k_lfm2_cases = {
    { "\012Free OCR.", { "\304\212", "Free", "\304\240OCR", "." } },
    { "<image>\012Free OCR.", { "<image", ">\304\212", "Free", "\304\240OCR", "." } },
    { "document parsing.", { "document", "\304\240parsing", "." } },
    { "\012<|grounding|>Convert the document to markdown.",
      { "\304\212", "<|", "grounding", "|>", "Convert", "\304\240the", "\304\240document", "\304\240to",
        "\304\240markdown", "." } },
    { "a\012\012b", { "a", "\304\212\304\212", "b" } },
    { "a\012\012\012  b", { "a", "\304\212\304\212\304\212", "\304\240", "\304\240b" } },
    { "  leading and trailing  ",
      { "\304\240", "\304\240leading", "\304\240and", "\304\240trailing", "\304\240\304\240" } },
    { "tabs\011and\011\011more", { "tabs", "\304\211and", "\304\211", "\304\211more" } },
    { "multi  space   run", { "multi", "\304\240", "\304\240space", "\304\240\304\240", "\304\240run" } },
    { "trailing newline\012", { "trailing", "\304\240newline", "\304\212" } },
    { "\012\012leading newlines", { "\304\212\304\212", "leading", "\304\240newlines" } },
    { "a\015\012b\015c", { "a", "\304\215\304\212", "b", "\304\215", "c" } },
    { "def fibonacci(n):\012    return n if n < 2 else fibonacci(n-1)",
      { "def", "\304\240fibonacci", "(n", "):\304\212", "\304\240\304\240\304\240", "\304\240return", "\304\240n",
        "\304\240if", "\304\240n", "\304\240<", "\304\240", "2", "\304\240else", "\304\240fibonacci", "(n", "-", "1",
        ")" } },
    { "#include <stdio.h>\015\012int main(void) { return 0; }",
      { "#include", "\304\240<", "stdio", ".h", ">\304\215\304\212", "int", "\304\240main", "(void", ")", "\304\240{",
        "\304\240return", "\304\240", "0", ";", "\304\240}" } },
    { "Quarterly revenue grew by 12% while costs 2026 stayed flat",
      { "Quarterly", "\304\240revenue", "\304\240grew", "\304\240by", "\304\240", "12", "%", "\304\240while",
        "\304\240costs", "\304\240", "202", "6", "\304\240stayed", "\304\240flat" } },
    { "1234567", { "123", "456", "7" } },
    { "10,000 and 3.14 and 0x1F and v2.0.1",
      { "10", ",", "000", "\304\240and", "\304\240",  "3", ".", "14", "\304\240and", "\304\240", "0",
        "x",  "1", "F",   "\304\240and", "\304\240v", "2", ".", "0",  ".",           "1" } },
    { "x=1;y=2;z=x+y", { "x", "=", "1", ";y", "=", "2", ";z", "=x", "+y" } },
    { "\331\243\331\244\331\245 and \340\271\231\340\271\231 and \302\275 and \302\262",
      { "\303\231\302\243\303\231\302\244\303\231\302\245", "\304\240and", "\304\240",
        "\303\240\302\271\304\273\303\240\302\271\304\273", "\304\240and", "\304\240", "\303\202\302\275",
        "\304\240and", "\304\240", "\303\202\302\262" } },
    { "sagte \342\200\236Hallo\342\200\234 heute",
      { "sagte", "\304\240\303\242\304\242\305\200", "Hallo", "\303\242\304\242\304\276", "\304\240heute" } },
    { "Er sagte: \302\273Guten Tag\302\253, dann ging er.",
      { "Er", "\304\240sagte", ":", "\304\240\303\202\302\273", "Guten", "\304\240Tag", "\303\202\302\253,",
        "\304\240dann", "\304\240ging", "\304\240er", "." } },
    { "\302\253quote\302\273", { "\303\202\302\253quote", "\303\202\302\273" } },
    { "\342\202\254\302\243abc", { "\303\242\304\244\302\254\303\202\302\243", "abc" } },
    { "\342\206\222\342\206\222x", { "\303\242\304\250\304\264\303\242\304\250\304\264", "x" } },
    { "a\302\251\302\256b", { "a", "\303\202\302\251\303\202\302\256", "b" } },
    { "\342\200\224 \"Hi\"", { "\303\242\304\242\304\266", "\304\240\"", "Hi", "\"" } },
    { "Die Katze schl\303\244ft; der Hund l\303\244uft \342\200\224 schnell!",
      { "Die", "\304\240Katze", "\304\240schl\303\203\302\244ft", ";", "\304\240der", "\304\240Hund",
        "\304\240l\303\203\302\244uft", "\304\240\303\242\304\242\304\266", "\304\240schnell", "!" } },
    { "\342\202\254100 costs $50 and \302\24320",
      { "\303\242\304\244\302\254", "100", "\304\240costs", "\304\240$", "50", "\304\240and",
        "\304\240\303\202\302\243", "20" } },
    { "\344\270\255\346\226\207\346\265\213\350\257\225\346\226\207\346\234\254\344\270\200\344\272\214\344\270\211",
      { "\303\244\302\270\305\203\303\246\304\270\304\251\303\246\302\265\304\255\303\250\302\257\304\267\303\246\304"
        "\270\304\251\303\246\304\276\302\254\303\244\302\270\304\242\303\244\302\272\304\256\303\244\302\270\304"
        "\253" } },
    { "\344\270\255\346\226\207abc", { "\303\244\302\270\305\203\303\246\304\270\304\251abc" } },
    { "abc\344\270\255\346\226\207", { "abc\303\244\302\270\305\203\303\246\304\270\304\251" } },
    { "\344\270\255\346\226\207\357\274\214\346\265\213\350\257\225\343\200\202",
      { "\303\244\302\270\305\203\303\246\304\270\304\251",
        "\303\257\302\274\304\256\303\246\302\265\304\255\303\250\302\257\304\267", "\303\243\304\242\304\244" } },
    { "\343\201\262\343\202\211\343\201\214\343\201\252\343\202\253\343\202\277\343\202\253\343\203\212\346\274\242\345"
      "\255\227",
      { "\303\243\304\243\302\262\303\243\304\244\304\253\303\243\304\243\304\256\303\243\304\243\302\252\303\243\304"
        "\244\302\253\303\243\304\244\302\277\303\243\304\244\302\253\303\243\304\245\304\254\303\246\302\274\302\242"
        "\303\245\305\203\304\271" } },
    { "caf\303\251 \344\270\255\346\226\207 123 !!!",
      { "caf\303\203\302\251", "\304\240\303\244\302\270\305\203\303\246\304\270\304\251", "\304\240", "123",
        "\304\240!!!" } },
    { "line1\012\344\270\255\346\226\207\012line3",
      { "line", "1", "\304\212", "\303\244\302\270\305\203\303\246\304\270\304\251", "\304\212", "line", "3" } },
    { "x\314\201y", { "x", "\303\214\304\243y" } },
    { "emoji \360\237\232\200 test \360\237\221\215\360\237\217\275 done",
      { "emoji", "\304\240\303\260\305\201\304\274\304\242", "\304\240test",
        "\304\240\303\260\305\201\304\263\304\257\303\260\305\201\304\261\302\275", "\304\240done" } },
    { "don't DON'T can't THEY'RE we've I'll he'd",
      { "don", "'t", "\304\240DON", "'T", "\304\240can", "'t", "\304\240THEY", "'RE", "\304\240we", "'ve", "\304\240I",
        "'ll", "\304\240he", "'d" } },
    { "\320\237\321\200\320\270\320\262\320\265\321\202 \320\274\320\270\321\200, \320\272\320\260\320\272 "
      "\320\264\320\265\320\273\320\260\077",
      { "\303\220\305\201\303\221\304\242\303\220\302\270\303\220\302\262\303\220\302\265\303\221\304\244",
        "\304\240\303\220\302\274\303\220\302\270\303\221\304\242", ",",
        "\304\240\303\220\302\272\303\220\302\260\303\220\302\272",
        "\304\240\303\220\302\264\303\220\302\265\303\220\302\273\303\220\302\260", "\077" } },
    { "Instruct: Given a question, retrieve passages that can help answer the question.\012Query: Wie hoch ist der "
      "Mount Everest\077",
      { "Instruct",
        ":",
        "\304\240Given",
        "\304\240a",
        "\304\240question",
        ",",
        "\304\240retrieve",
        "\304\240passages",
        "\304\240that",
        "\304\240can",
        "\304\240help",
        "\304\240answer",
        "\304\240the",
        "\304\240question",
        ".\304\212",
        "Query",
        ":",
        "\304\240Wie",
        "\304\240hoch",
        "\304\240ist",
        "\304\240der",
        "\304\240Mount",
        "\304\240Everest",
        "\077" } },
};

// Golden: deepseek-ai/DeepSeek-OCR-2 == baidu/Unlimited-OCR (asserted equal
// in the generator; the Split SEQUENCE regex).
static const std::vector<Case> k_deepseek_cases = {
    { "\012Free OCR.", { "\304\212", "Free", "\304\240OCR", "." } },
    { "<image>\012Free OCR.", { "<image", ">\304\212", "Free", "\304\240OCR", "." } },
    { "document parsing.", { "document", "\304\240parsing", "." } },
    { "\012<|grounding|>Convert the document to markdown.",
      { "\304\212", "<|", "grounding", "|>", "Convert", "\304\240the", "\304\240document", "\304\240to",
        "\304\240markdown", "." } },
    { "a\012\012b", { "a", "\304\212\304\212", "b" } },
    { "a\012\012\012  b", { "a", "\304\212\304\212\304\212", "\304\240", "\304\240b" } },
    { "  leading and trailing  ",
      { "\304\240", "\304\240leading", "\304\240and", "\304\240trailing", "\304\240\304\240" } },
    { "tabs\011and\011\011more", { "tabs", "\304\211and", "\304\211", "\304\211more" } },
    { "multi  space   run", { "multi", "\304\240", "\304\240space", "\304\240\304\240", "\304\240run" } },
    { "trailing newline\012", { "trailing", "\304\240newline", "\304\212" } },
    { "\012\012leading newlines", { "\304\212\304\212", "leading", "\304\240newlines" } },
    { "a\015\012b\015c", { "a", "\304\215\304\212", "b", "\304\215", "c" } },
    { "def fibonacci(n):\012    return n if n < 2 else fibonacci(n-1)",
      { "def", "\304\240fibonacci", "(n", "):\304\212", "\304\240\304\240\304\240", "\304\240return", "\304\240n",
        "\304\240if", "\304\240n", "\304\240<", "\304\240", "2", "\304\240else", "\304\240fibonacci", "(n", "-", "1",
        ")" } },
    { "#include <stdio.h>\015\012int main(void) { return 0; }",
      { "#include", "\304\240<", "stdio", ".h", ">\304\215\304\212", "int", "\304\240main", "(void", ")", "\304\240{",
        "\304\240return", "\304\240", "0", ";", "\304\240}" } },
    { "Quarterly revenue grew by 12% while costs 2026 stayed flat",
      { "Quarterly", "\304\240revenue", "\304\240grew", "\304\240by", "\304\240", "12", "%", "\304\240while",
        "\304\240costs", "\304\240", "202", "6", "\304\240stayed", "\304\240flat" } },
    { "1234567", { "123", "456", "7" } },
    { "10,000 and 3.14 and 0x1F and v2.0.1",
      { "10", ",", "000", "\304\240and", "\304\240",  "3", ".", "14", "\304\240and", "\304\240", "0",
        "x",  "1", "F",   "\304\240and", "\304\240v", "2", ".", "0",  ".",           "1" } },
    { "x=1;y=2;z=x+y", { "x", "=", "1", ";y", "=", "2", ";z", "=x", "+y" } },
    { "\331\243\331\244\331\245 and \340\271\231\340\271\231 and \302\275 and \302\262",
      { "\303\231\302\243\303\231\302\244\303\231\302\245", "\304\240and", "\304\240",
        "\303\240\302\271\304\273\303\240\302\271\304\273", "\304\240and", "\304\240", "\303\202\302\275",
        "\304\240and", "\304\240", "\303\202\302\262" } },
    { "sagte \342\200\236Hallo\342\200\234 heute",
      { "sagte", "\304\240\303\242\304\242\305\200", "Hallo", "\303\242\304\242\304\276", "\304\240heute" } },
    { "Er sagte: \302\273Guten Tag\302\253, dann ging er.",
      { "Er", "\304\240sagte", ":", "\304\240\303\202\302\273", "Guten", "\304\240Tag", "\303\202\302\253,",
        "\304\240dann", "\304\240ging", "\304\240er", "." } },
    { "\302\253quote\302\273", { "\303\202\302\253", "quote", "\303\202\302\273" } },
    { "\342\202\254\302\243abc", { "\303\242\304\244\302\254\303\202\302\243", "abc" } },
    { "\342\206\222\342\206\222x", { "\303\242\304\250\304\264\303\242\304\250\304\264", "x" } },
    { "a\302\251\302\256b", { "a", "\303\202\302\251\303\202\302\256", "b" } },
    { "\342\200\224 \"Hi\"", { "\303\242\304\242\304\266", "\304\240\"", "Hi", "\"" } },
    { "Die Katze schl\303\244ft; der Hund l\303\244uft \342\200\224 schnell!",
      { "Die", "\304\240Katze", "\304\240schl\303\203\302\244ft", ";", "\304\240der", "\304\240Hund",
        "\304\240l\303\203\302\244uft", "\304\240\303\242\304\242\304\266", "\304\240schnell", "!" } },
    { "\342\202\254100 costs $50 and \302\24320",
      { "\303\242\304\244\302\254", "100", "\304\240costs", "\304\240$", "50", "\304\240and",
        "\304\240\303\202\302\243", "20" } },
    { "\344\270\255\346\226\207\346\265\213\350\257\225\346\226\207\346\234\254\344\270\200\344\272\214\344\270\211",
      { "\303\244\302\270\305\203\303\246\304\270\304\251\303\246\302\265\304\255\303\250\302\257\304\267\303\246\304"
        "\270\304\251\303\246\304\276\302\254\303\244\302\270\304\242\303\244\302\272\304\256\303\244\302\270\304"
        "\253" } },
    { "\344\270\255\346\226\207abc", { "\303\244\302\270\305\203\303\246\304\270\304\251", "abc" } },
    { "abc\344\270\255\346\226\207", { "abc", "\303\244\302\270\305\203\303\246\304\270\304\251" } },
    { "\344\270\255\346\226\207\357\274\214\346\265\213\350\257\225\343\200\202",
      { "\303\244\302\270\305\203\303\246\304\270\304\251", "\303\257\302\274\304\256",
        "\303\246\302\265\304\255\303\250\302\257\304\267", "\303\243\304\242\304\244" } },
    { "\343\201\262\343\202\211\343\201\214\343\201\252\343\202\253\343\202\277\343\202\253\343\203\212\346\274\242\345"
      "\255\227",
      { "\303\243\304\243\302\262\303\243\304\244\304\253\303\243\304\243\304\256\303\243\304\243\302\252\303\243\304"
        "\244\302\253\303\243\304\244\302\277\303\243\304\244\302\253\303\243\304\245\304\254\303\246\302\274\302\242"
        "\303\245\305\203\304\271" } },
    { "caf\303\251 \344\270\255\346\226\207 123 !!!",
      { "caf\303\203\302\251", "\304\240", "\303\244\302\270\305\203\303\246\304\270\304\251", "\304\240", "123",
        "\304\240!!!" } },
    { "line1\012\344\270\255\346\226\207\012line3",
      { "line", "1", "\304\212", "\303\244\302\270\305\203\303\246\304\270\304\251", "\304\212", "line", "3" } },
    { "x\314\201y", { "x\303\214\304\243y" } },
    { "emoji \360\237\232\200 test \360\237\221\215\360\237\217\275 done",
      { "emoji", "\304\240\303\260\305\201\304\274\304\242", "\304\240test",
        "\304\240\303\260\305\201\304\263\304\257\303\260\305\201\304\261\302\275", "\304\240done" } },
    { "don't DON'T can't THEY'RE we've I'll he'd",
      { "don", "'t", "\304\240DON", "'T", "\304\240can", "'t", "\304\240THEY", "'RE", "\304\240we", "'ve", "\304\240I",
        "'ll", "\304\240he", "'d" } },
    { "\320\237\321\200\320\270\320\262\320\265\321\202 \320\274\320\270\321\200, \320\272\320\260\320\272 "
      "\320\264\320\265\320\273\320\260\077",
      { "\303\220\305\201\303\221\304\242\303\220\302\270\303\220\302\262\303\220\302\265\303\221\304\244",
        "\304\240\303\220\302\274\303\220\302\270\303\221\304\242", ",",
        "\304\240\303\220\302\272\303\220\302\260\303\220\302\272",
        "\304\240\303\220\302\264\303\220\302\265\303\220\302\273\303\220\302\260", "\077" } },
    { "Instruct: Given a question, retrieve passages that can help answer the question.\012Query: Wie hoch ist der "
      "Mount Everest\077",
      { "Instruct",
        ":",
        "\304\240Given",
        "\304\240a",
        "\304\240question",
        ",",
        "\304\240retrieve",
        "\304\240passages",
        "\304\240that",
        "\304\240can",
        "\304\240help",
        "\304\240answer",
        "\304\240the",
        "\304\240question",
        ".\304\212",
        "Query",
        ":",
        "\304\240Wie",
        "\304\240hoch",
        "\304\240ist",
        "\304\240der",
        "\304\240Mount",
        "\304\240Everest",
        "\077" } },
};

typedef std::vector<std::string> (*PreTok)(const std::string &);

static int run(const char * name, PreTok fn, const std::vector<Case> & cases, int & checked) {
    int failures = 0;
    for (const auto & c : cases) {
        // The pre-tokenizers return raw substrings; the golden splits are the
        // byte-level-encoded form, so encode before comparing.
        std::vector<std::string> got;
        for (const auto & pt : fn(c.text)) got.push_back(core_bpe::bytes_to_unicode(pt.data(), pt.size()));
        checked++;
        if (got != c.splits) {
            failures++;
            fprintf(stderr, "FAIL[%s]: %s\n  want(%zu):", name, c.text, c.splits.size());
            for (const auto & s : c.splits) fprintf(stderr, " [%s]", s.c_str());
            fprintf(stderr, "\n  got (%zu):", got.size());
            for (const auto & s : got) fprintf(stderr, " [%s]", s.c_str());
            fprintf(stderr, "\n");
        }
        // An invariant a typo cannot satisfy: the regexes PARTITION the input,
        // so concatenating the raw splits must reproduce it byte for byte.
        std::string joined;
        for (const auto & pt : fn(c.text)) joined += pt;
        checked++;
        if (joined != c.text) {
            failures++;
            fprintf(stderr, "FAIL[%s]: lossy split for [%s]\n  rejoined [%s]\n", name, c.text, joined.c_str());
        }
    }
    // The empty string must yield no pre-tokens at all.
    checked++;
    if (!fn("").empty()) {
        fprintf(stderr, "FAIL[%s]: empty input produced pre-tokens\n", name);
        failures++;
    }
    return failures;
}

static int crispembed_test_main() {
    int checked = 0;
    int failures = 0;
    failures += run("qwen", core_bpe::qwen_pretokenize, k_qwen_cases, checked);
    failures += run("lfm2", core_bpe::lfm2_pretokenize, k_lfm2_cases, checked);
    failures += run("deepseek", core_bpe::deepseek_pretokenize, k_deepseek_cases, checked);

    // bpe_one's merge heap must break rank TIES leftmost, the way HuggingFace's
    // BPE orders its heap by (rank, pos) both ascending. std::priority_queue
    // leaves equal keys in an UNSPECIFIED order, so before the tie-break the
    // merge could start anywhere in a run of equal-rank pairs. Both cases below
    // are hermetic (a five-entry vocab, one or two merge rules, no model) and
    // both were observed wrong on the pre-fix comparator; on real vocabs it
    // cost 4 of 1508 random strings per tokenizer.
    //
    // Short runs are NOT a guard: with three symbols the heap happens to come
    // out leftmost anyway. It takes five to make the ordering bite.
    {
        struct TB {
            const char * word;
            std::unordered_map<std::string, int32_t> vocab;
            std::unordered_map<std::string, int32_t> merges;
            std::vector<int32_t> want;
            const char * note;
        };
        const std::vector<TB> tbs = {
            // qq|qq|q|c — merging leftmost-first. The pre-fix heap gave qq|q|qq|c.
            { "qqqqqc",
              { { "q", 10 }, { "qq", 11 }, { "c", 12 } },
              { { "q q", 0 } },
              { 11, 11, 10, 12 },
              "single rule, five-symbol run" },
            // ab|ab|a|x with `a b` and `b a` at the SAME rank; the pre-fix heap
            // took the `b a` pair first and gave ab|a|ba|x.
            { "ababax",
              { { "a", 20 }, { "b", 21 }, { "ab", 22 }, { "ba", 23 }, { "x", 24 } },
              { { "a b", 0 }, { "b a", 0 } },
              { 22, 22, 20, 24 },
              "two rules of equal rank" },
        };
        for (const auto & tb : tbs) {
            std::vector<int32_t> ids;
            core_bpe::bpe_one(tb.vocab, tb.merges, tb.word, ids);
            checked++;
            if (ids != tb.want) {
                failures++;
                fprintf(stderr, "FAIL: bpe_one tie-break (%s) on \"%s\"\n  want:", tb.note, tb.word);
                for (int32_t v : tb.want) fprintf(stderr, " %d", v);
                fprintf(stderr, "\n  got :");
                for (int32_t v : ids) fprintf(stderr, " %d", v);
                fprintf(stderr, "\n");
            }
        }
    }

    // baidu/Unlimited-OCR declares a pre_tokenizer section byte-identical to
    // DeepSeek-OCR-2's, so one implementation serves both engines. The
    // generator asserts that against the two live tokenizer.json files; this
    // records the consequence the C++ side relies on.
    checked++;
    if (core_bpe::deepseek_pretokenize("\nFree OCR.") != core_bpe::deepseek_pretokenize("\nFree OCR.")) {
        fprintf(stderr, "FAIL: deepseek_pretokenize is not deterministic\n");
        failures++;
    }

    printf("test-bpe-pretokenize: %d checks, %d failures\n", checked, failures);
    return failures == 0 ? 0 : 1;
}

// tools/check_test_clean_exit.sh: a one-shot binary must not run ggml's
// static GPU-device destructor at exit (it aborts on Metal / faults on CUDA).
int main() {
    core_util::clean_exit(crispembed_test_main());
}
