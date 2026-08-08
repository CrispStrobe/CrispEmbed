// tests/test_bert_pretokenize.cpp — BERT pre-tokenizer + tokenizer-family
// routing parity (the F8 LaBSE-class WordPiece fix).
//
// Hermetic: no vocab, no GGUF, no network. Two independent defect classes are
// pinned here:
//
// 1. core_bert::pretokenize (src/core/bert_pretok.h) versus HuggingFace's own
//    BertNormalizer + BertPreTokenizer. Goldens captured verbatim from
//    sentence-transformers/LaBSE's tokenizer.json via
//        t = Tokenizer.from_file(".../LaBSE/tokenizer.json")
//        ns = t.normalizer.normalize_str(text)
//        [w for w, _ in t.pre_tokenizer.pre_tokenize_str(ns)]
//    The historical per-byte isspace/ispunct splitter fails most of these
//    (CJK stays glued, Unicode punctuation rides inside the word, an NBSP is
//    a letter, a soft hyphen survives) — which is exactly why that path stays
//    frozen for shipped GGUFs and this one is opt-in via
//    `tokenizer.ggml.pre = "bert"`.
//
//    The subtle rows: HF `is_bert_punc` covers ASCII punctuation AND ASCII
//    symbols plus Unicode \p{P} — but NOT Unicode symbols, so "$50" splits
//    while "€100"/"£20"/"¥3000" stay whole. CJK IDEOGRAPHS split
//    per-character but kana do not ("日本語の..." -> "日","本","語",
//    "のテキストと..."). Control/format chars (soft hyphen, ZWSP, word
//    joiner) vanish; every Unicode whitespace (NBSP included) is a break.
//
// 2. resolve_tokenizer_family (src/tokenizer.h). The legacy `vocab > 100k ->
//    SentencePiece` heuristic must yield to an EXPLICIT numeric
//    `tokenizer.ggml.type` — before the fix an explicit WordPiece(0) with
//    LaBSE's 501k vocab was still routed into the SPM tokenizer (bos=0/eos=2
//    instead of [CLS]/[SEP], literal "▁" tokens for spaces, 0/20 HF id
//    parity). GGUFs WITHOUT the numeric key must keep the historical
//    behavior bit-for-bit, including the known-wrong corner (community
//    "bert"-string GGUF with a >100k vocab still goes SPM).
//
//   c++ -std=c++17 -O1 -Isrc tests/test_bert_pretokenize.cpp -o /tmp/test-bert-pretok
//   /tmp/test-bert-pretok
//
// Exit 0 == every split and every routing decision matches.

#include "core/bert_pretok.h"
#include "core/clean_exit.h"
#include "tokenizer.h"

#include <cstdio>
#include <string>
#include <vector>

struct Case {
    const char * text;
    std::vector<std::string> words;
};

static const std::vector<Case> k_cases = {
    { "Die Bäckerei an der Straße verkauft süße Brötchen.",
      { "Die", "Bäckerei", "an", "der", "Straße", "verkauft", "süße", "Brötchen", "." } },
    { "ÄRGER MIT GROSSEN BUCHSTABEN UND ÜBERMUT", { "ÄRGER", "MIT", "GROSSEN", "BUCHSTABEN", "UND", "ÜBERMUT" } },
    { "multi  space   run    here", { "multi", "space", "run", "here" } },
    { "line1\nline2\n\nline4", { "line1", "line2", "line4" } },
    { "tabs\tand\t\tmore\ttabs", { "tabs", "and", "more", "tabs" } },
    { "a\r\nb\rc\nd", { "a", "b", "c", "d" } },
    { "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1)",
      { "def", "fibonacci", "(", "n", ")", ":", "return", "n", "if", "n", "<", "2", "else", "fibonacci", "(", "n", "-",
        "1", ")" } },
    { "Er sagte: »Guten Tag«, dann ging er — „wirklich“.",
      { "Er", "sagte", ":", "»", "Guten", "Tag", "«", ",", "dann", "ging", "er", "—", "„", "wirklich", "“", "." } },
    // ASCII symbols split ($), Unicode currency symbols do NOT (euro/pound/yen)
    { "€100 costs $50 and £20 or ¥3000", { "€100", "costs", "$", "50", "and", "£20", "or", "¥3000" } },
    // NBSP is whitespace; the soft hyphen (Cf) vanishes
    { "hard\u00a0space and soft\u00adhyphen test", { "hard", "space", "and", "softhyphen", "test" } },
    { "中文测试文本，一二三。", { "中", "文", "测", "试", "文", "本", "，", "一", "二", "三", "。" } },
    // ideographs split per-char, the kana run stays ONE word
    { "日本語のテキストとひらがなカタカナ", { "日", "本", "語", "のテキストとひらがなカタカナ" } },
    { "Русский текст для "
      "проверки "
      "токенизации",
      { "Русский", "текст", "для", "проверки", "токенизации" } },
    { "Привет, мир! Как дела?", { "Привет", ",", "мир", "!", "Как", "дела", "?" } },
    // emoji are Unicode SYMBOLS (not punctuation): they stay words / glued runs
    { "emoji \U0001f680 test \U0001f44d\U0001f3fd done \U0001f1e9\U0001f1ea",
      { "emoji", "\U0001f680", "test", "\U0001f44d\U0001f3fd", "done", "\U0001f1e9\U0001f1ea" } },
    { "Donaudampfschifffahrtsgesellschaftskapitän", { "Donaudampfschifffahrtsgesellschaftskapitän" } },
    { "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz",
      { "Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz" } },
    { "don't DON'T can't THEY'RE we've I'll he'd", { "don", "'",  "t", "DON", "'", "T", "can", "'",  "t", "THEY", "'",
                                                     "RE",  "we", "'", "ve",  "I", "'", "ll",  "he", "'", "d" } },
    { "1234567 and 10,000 and 3.14 and 1.000.000,50",
      { "1234567", "and", "10", ",", "000", "and", "3", ".", "14", "and", "1", ".", "000", ".", "000", ",", "50" } },
    { "café naïve résumé ½ ² ٣٤٥", { "café", "naïve", "résumé", "½", "²", "٣٤٥" } },
    // zero-width space and word joiner are FORMAT chars: removed, words fuse
    { "x\u200by z\u2060w", { "xy", "zw" } },
    // symbol-versus-punctuation fine print: euro/yen/copyright attach,
    // section sign (Po) isolates
    { "a€b £5 ¥¥ §7 ©x", { "a€b", "£5", "¥¥", "§", "7", "©x" } },
    { "中文abc漢字123かな", { "中", "文", "abc", "漢", "字", "123かな" } },
    { "…!?—–‐", { "…", "!", "?", "—", "–", "‐" } },
};

struct RouteCase {
    const char * name;
    bool type_key_present;
    int declared_type;
    const char * model_str;
    int64_t vocab_n;
    int expect; // 0=WordPiece 1=BPE 2=SentencePiece
};

static const std::vector<RouteCase> k_routes = {
    // The F8 defect: explicit WordPiece must win over the >100k heuristic
    { "labse: explicit type=0, 501k vocab", true, 0, "", 501153, 0 },
    { "explicit type=2 (XLM-R crisp GGUF)", true, 2, "", 250002, 2 },
    { "explicit type=1 (granite-r2 BPE, 180k)", true, 1, "", 180000, 1 },
    { "explicit type=0, small vocab (MiniLM)", true, 0, "", 30522, 0 },
    // Absent numeric key == the HISTORICAL behavior, bit for bit
    { "no keys, 30k vocab -> WordPiece", false, 0, "", 30522, 0 },
    { "no keys, 250k vocab -> SPM heuristic", false, 0, "", 250002, 2 },
    { "model=gpt2 -> BPE", false, 0, "gpt2", 50368, 1 },
    { "model=bpe -> BPE", false, 0, "bpe", 50368, 1 },
    { "model=llama -> SPM", false, 0, "llama", 32000, 2 },
    { "model=t5 -> SPM", false, 0, "t5", 32100, 2 },
    { "model=unigram -> SPM", false, 0, "unigram", 250002, 2 },
    { "model=bert, small -> WordPiece", false, 0, "bert", 30522, 0 },
    { "model=wordpiece, small -> WordPiece", false, 0, "wordpiece", 30522, 0 },
    // Known-wrong historical corner, deliberately preserved: a community
    // "bert" GGUF with a LaBSE-sized vocab has no CrispEmbed metadata, so it
    // keeps taking the legacy heuristic into SPM.
    { "model=bert, 501k -> SPM (frozen legacy)", false, 0, "bert", 501153, 2 },
    { "model=gpt2, 180k -> BPE (string authoritative)", false, 0, "gpt2", 180000, 1 },
};

// ---------- E5 guard: full WordPiece pipeline on CJK -------------------------
//
// The pretokenize cases above verify that core_bert::pretokenize splits CJK
// ideographs and isolates Unicode punctuation. But the E5 defect is that the
// HISTORICAL per-byte path (used by shipped all-MiniLM-L6-v2 GGUFs, which
// lack `tokenizer.ggml.pre = "bert"`) collapses entire Japanese strings into
// a single word → a single [UNK]. Both Japanese fixture sentences produce
// bit-identical embeddings because they produce bit-identical token sequences:
//    [CLS] [UNK] [SEP]
//
// HF's BasicTokenizer with do_lower_case=True produces DIFFERENT sequences:
//    ja_cat_a: [UNK],[UNK],上,て,[UNK],っ,##て,##い,##る,。  (10 tokens)
//    ja_cat_b: [UNK],[UNK],か,[UNK],て,##い,##ま,##す,。     ( 9 tokens)
//
// The difference: HF applies NFD + accent stripping (strips dakuten from kana:
// が→か, で→て) THEN CJK ideograph splitting, so kana runs decompose further
// and each kanji becomes its own word.
//
// This test pins both behaviors:
//   1. Historical per-byte split_words: both JA sentences → 1 word each
//   2. core_bert::pretokenize on the same JA text: ideographs split,
//      kana glued, Unicode punct isolated — different from HF (no NFD strip)
//      but the two sentences DO produce different pre-token lists
//
// The full WordPiece guard (with a real vocab) is exercised via the C++ API
// by tests/wordpiece_cjk_parity.py's documentation; the hermetic C++ cases
// here test only the pretokenizer/splitter, which is the root cause.

struct HistSplitCase {
    const char * text;
    std::vector<std::string> words;
};

// Historical per-byte split (simulate: ASCII isspace/ispunct only, lowercased)
static std::vector<std::string> historical_split(const std::string & text) {
    std::vector<std::string> words;
    std::string current;
    for (size_t i = 0; i < text.size(); i++) {
        unsigned char c = text[i];
        if (std::isspace(c)) {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
        } else if (std::ispunct(c)) {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }
            words.push_back(std::string(1, (char)c));
        } else {
            current += (char)std::tolower(c);
        }
    }
    if (!current.empty()) words.push_back(current);
    return words;
}

// The E5 fixture sentences
static const char * JA_CAT_A = "猫がソファの上で眠っている。";
static const char * JA_CAT_B = "ソファーで猫が寝ています。";
static const char * JA_WEATHER = "明日の東京の天気は雨でしょう。";

// E5 guard cases: core_bert::pretokenize on the JA fixture
static const std::vector<Case> k_e5_pretok_cases = {
    // CJK ideographs split per-char, kana stays glued, Unicode punct isolated
    { "猫がソファの上で眠っている。", { "猫", "がソファの", "上", "で", "眠", "っている", "。" } },
    { "ソファーで猫が寝ています。", { "ソファーで", "猫", "が", "寝", "ています", "。" } },
    { "明日の東京の天気は雨でしょう。",
      { "明", "日", "の", "東", "京", "の", "天", "気", "は", "雨", "でしょう", "。" } },
};

// E5 guard cases: historical per-byte split on the same JA text
// The entire UTF-8 string is one word (no byte matches ASCII space/punct,
// since 。 is U+3002 = 0xE3 0x80 0x82, none of which are ASCII punctuation)
static const std::vector<HistSplitCase> k_e5_hist_cases = {
    // lowercased bytes, but Japanese bytes are all >0x7F so tolower is identity
    { "猫がソファの上で眠っている。", { "猫がソファの上で眠っている。" } },
    { "ソファーで猫が寝ています。", { "ソファーで猫が寝ています。" } },
    { "明日の東京の天気は雨でしょう。", { "明日の東京の天気は雨でしょう。" } },
};

static std::string join(const std::vector<std::string> & v) {
    std::string s;
    for (size_t i = 0; i < v.size(); i++) {
        if (i) s += "|";
        s += v[i];
    }
    return s;
}

static int crispembed_test_main() {
    int fails = 0;

    for (const auto & c : k_cases) {
        const std::vector<std::string> got = core_bert::pretokenize(c.text);
        if (got != c.words) {
            fails++;
            printf("FAIL pretok  %-40.40s\n  want %s\n  got  %s\n", c.text, join(c.words).c_str(), join(got).c_str());
        }
    }

    for (const auto & r : k_routes) {
        const int got = resolve_tokenizer_family(r.type_key_present, r.declared_type, r.model_str, r.vocab_n);
        if (got != r.expect) {
            fails++;
            printf("FAIL route   %s: want %d got %d\n", r.name, r.expect, got);
        }
    }

    // E5: core_bert::pretokenize on JA fixture (CJK split, kana glued, no NFD)
    for (const auto & c : k_e5_pretok_cases) {
        const std::vector<std::string> got = core_bert::pretokenize(c.text);
        if (got != c.words) {
            fails++;
            printf("FAIL E5-pretok %-40.40s\n  want %s\n  got  %s\n", c.text, join(c.words).c_str(), join(got).c_str());
        }
    }

    // E5: historical per-byte split on the same JA text — pins the degenerate
    // behavior (entire string is one word, leading to single-[UNK] collapse)
    for (const auto & c : k_e5_hist_cases) {
        const std::vector<std::string> got = historical_split(c.text);
        if (got != c.words) {
            fails++;
            printf("FAIL E5-hist   %-40.40s\n  want %s\n  got  %s\n", c.text, join(c.words).c_str(), join(got).c_str());
        }
    }

    // E5: the two JA cat sentences produce DIFFERENT pretokenize results
    // (the historical path produces the SAME single-word result — that is
    // what causes the bit-identical embeddings)
    {
        auto a = core_bert::pretokenize(JA_CAT_A);
        auto b = core_bert::pretokenize(JA_CAT_B);
        if (a == b) {
            fails++;
            printf("FAIL E5-diff   pretokenize(ja_cat_a) == pretokenize(ja_cat_b) — should differ\n");
        }
        auto ha = historical_split(JA_CAT_A);
        auto hb = historical_split(JA_CAT_B);
        // Historical path: both are single words (but different content)
        if (ha.size() != 1 || hb.size() != 1) {
            fails++;
            printf("FAIL E5-hist-1word  historical split should produce 1 word per JA sentence\n");
        }
    }

    printf("bert-pretokenize: %zu split + %zu route + %zu E5-pretok + %zu E5-hist cases, %d failure(s)\n",
           k_cases.size(), k_routes.size(), k_e5_pretok_cases.size(), k_e5_hist_cases.size(), fails);
    return fails ? 1 : 0;
}

int main() {
    core_util::clean_exit(crispembed_test_main());
}
