"""Emit tests/test_bpe_pretokenize.cpp from HuggingFace golden pre-tokenizations."""
import sys
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download

BATTERY = [
    # --- the engines' own real prompts (byte-exact) ---
    "\nFree OCR.",
    "<image>\nFree OCR.",
    "document parsing.",
    "\n<|grounding|>Convert the document to markdown.",
    # --- whitespace battery: the tokenize_simple defect class ---
    "a\n\nb",
    "a\n\n\n  b",
    "  leading and trailing  ",
    "tabs\tand\t\tmore",
    "multi  space   run",
    "trailing newline\n",
    "\n\nleading newlines",
    "a\r\nb\rc",
    "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1)",
    "#include <stdio.h>\r\nint main(void) { return 0; }",
    # --- digits: Qwen splits every digit, LFM2/DeepSeek take runs of up to 3 ---
    "Quarterly revenue grew by 12% while costs 2026 stayed flat",
    "1234567",
    "10,000 and 3.14 and 0x1F and v2.0.1",
    "x=1;y=2;z=x+y",
    "٣٤٥ and ๙๙ and ½ and ²",
    # --- non-ASCII punctuation: the residual defect in the merged qwen fix ---
    "sagte „Hallo“ heute",
    "Er sagte: »Guten Tag«, dann ging er.",
    "«quote»",
    "€£abc",
    "→→x",
    "a©®b",
    "— \"Hi\"",
    "Die Katze schläft; der Hund läuft — schnell!",
    "€100 costs $50 and £20",
    # --- CJK: DeepSeek isolates CJK runs, Qwen/LFM2 do not ---
    "中文测试文本一二三",
    "中文abc",
    "abc中文",
    "中文，测试。",
    "ひらがなカタカナ漢字",
    "café 中文 123 !!!",
    "line1\n中文\nline3",
    # --- marks, emoji, contractions, Cyrillic ---
    "x́y",
    "emoji \U0001f680 test \U0001f44d\U0001f3fd done",
    "don't DON'T can't THEY'RE we've I'll he'd",
    "Привет мир, как дела?",
    "Instruct: Given a question, retrieve passages that can help answer the "
    "question.\nQuery: Wie hoch ist der Mount Everest?",
]


def cstr(b: bytes) -> str:
    out = ['"']
    for ch in b:
        if ch == 0x22:
            out.append('\\"')
        elif ch == 0x5C:
            out.append('\\\\')
        elif ch == 0x3F:
            out.append('\\077')  # never form a trigraph
        elif 0x20 <= ch < 0x7F:
            out.append(chr(ch))
        else:
            out.append('\\%03o' % ch)
    out.append('"')
    return ''.join(out)


def table(name, tok, comment):
    lines = ['// %s' % comment, 'static const std::vector<Case> %s = {' % name]
    for s in BATTERY:
        pre = [x[0] for x in tok.pre_tokenizer.pre_tokenize_str(s)]
        lines.append('    { %s,' % cstr(s.encode()))
        lines.append('      { %s } },' % ', '.join(cstr(p.encode()) for p in pre))
    lines.append('};')
    return '\n'.join(lines)


qwen = Tokenizer.from_file(hf_hub_download('codefuse-ai/F2LLM-v2-160M', 'tokenizer.json'))
lfm2 = Tokenizer.from_file(hf_hub_download('LiquidAI/LFM2.5-Embedding-350M', 'tokenizer.json'))
dsek = Tokenizer.from_file(hf_hub_download('deepseek-ai/DeepSeek-OCR-2', 'tokenizer.json'))
uocr = Tokenizer.from_file(hf_hub_download('baidu/Unlimited-OCR', 'tokenizer.json'))

# The audit's claim that one implementation serves both OCR engines rests on
# this: their declared pre_tokenizer sections are byte-identical.
for s in BATTERY:
    a = [x[0] for x in dsek.pre_tokenizer.pre_tokenize_str(s)]
    b = [x[0] for x in uocr.pre_tokenizer.pre_tokenize_str(s)]
    assert a == b, (s, a, b)

HEADER = '''// tests/test_bpe_pretokenize.cpp — declared-regex pre-tokenizer parity for the
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
// a single space and deleted newlines outright, so "a\\n\\nb" and "a b" produced
// identical ids. T19-E1 fixed the Qwen embedder path; this is the audit of the
// other callers:
//
//   src/lfm2_embed.cpp      LiquidAI/LFM2.5-Embedding-350M  live (arbitrary user text)
//   src/deepseek_ocr2.cpp   deepseek-ai/DeepSeek-OCR-2      live ("\\nFree OCR." prompt)
//   src/unlimited_ocr.cpp   baidu/Unlimited-OCR             latent (debug path only)
//
// Three declared regexes, three tables:
//
//   qwen      \\p{N}             — one token per digit
//   lfm2      \\p{N}{1,3}        — digit runs of up to three, else identical to qwen
//   deepseek  a Split SEQUENCE  — \\p{N}{1,3}, then CJK/kana runs, then a regex
//                                 built on [\\p{P}\\p{S}] rather than [^\\s\\p{L}\\p{N}]
//
// The qwen table is here as well because this audit found a SECOND, residual
// defect in the merged E1 fix: `qwen_is_letter` answered true for every byte
// >= 0x80, so non-ASCII punctuation was absorbed into the neighbouring word.
// HuggingFace splits `sagte \\u201eHallo\\u201c heute` into 5 pre-tokens; the
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
'''

FOOTER = '''
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
            fprintf(stderr, "FAIL[%s]: %s\\n  want(%zu):", name, c.text, c.splits.size());
            for (const auto & s : c.splits) fprintf(stderr, " [%s]", s.c_str());
            fprintf(stderr, "\\n  got (%zu):", got.size());
            for (const auto & s : got) fprintf(stderr, " [%s]", s.c_str());
            fprintf(stderr, "\\n");
        }
        // An invariant a typo cannot satisfy: the regexes PARTITION the input,
        // so concatenating the raw splits must reproduce it byte for byte.
        std::string joined;
        for (const auto & pt : fn(c.text)) joined += pt;
        checked++;
        if (joined != c.text) {
            failures++;
            fprintf(stderr, "FAIL[%s]: lossy split for [%s]\\n  rejoined [%s]\\n", name, c.text, joined.c_str());
        }
    }
    // The empty string must yield no pre-tokens at all.
    checked++;
    if (!fn("").empty()) {
        fprintf(stderr, "FAIL[%s]: empty input produced pre-tokens\\n", name);
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
                fprintf(stderr, "FAIL: bpe_one tie-break (%s) on \\"%s\\"\\n  want:", tb.note, tb.word);
                for (int32_t v : tb.want) fprintf(stderr, " %d", v);
                fprintf(stderr, "\\n  got :");
                for (int32_t v : ids) fprintf(stderr, " %d", v);
                fprintf(stderr, "\\n");
            }
        }
    }

    // baidu/Unlimited-OCR declares a pre_tokenizer section byte-identical to
    // DeepSeek-OCR-2's, so one implementation serves both engines. The
    // generator asserts that against the two live tokenizer.json files; this
    // records the consequence the C++ side relies on.
    checked++;
    if (core_bpe::deepseek_pretokenize("\\nFree OCR.") != core_bpe::deepseek_pretokenize("\\nFree OCR.")) {
        fprintf(stderr, "FAIL: deepseek_pretokenize is not deterministic\\n");
        failures++;
    }

    printf("test-bpe-pretokenize: %d checks, %d failures\\n", checked, failures);
    return failures == 0 ? 0 : 1;
}

// tools/check_test_clean_exit.sh: a one-shot binary must not run ggml's
// static GPU-device destructor at exit (it aborts on Metal / faults on CUDA).
int main() {
    core_util::clean_exit(crispembed_test_main());
}
'''

body = '\n\n'.join([
    table('k_qwen_cases', qwen,
          'Golden: codefuse-ai/F2LLM-v2-160M (the Qwen2/Qwen3 declared regex).'),
    table('k_lfm2_cases', lfm2,
          'Golden: LiquidAI/LFM2.5-Embedding-350M (same regex, \\p{N}{1,3} digits).'),
    table('k_deepseek_cases', dsek,
          'Golden: deepseek-ai/DeepSeek-OCR-2 == baidu/Unlimited-OCR (asserted equal\n'
          '// in the generator; the Split SEQUENCE regex).'),
])
open(sys.argv[1], 'w').write(HEADER + '\n' + body + '\n' + FOOTER)
print('wrote', sys.argv[1], len(BATTERY), 'cases x 3 tables')
