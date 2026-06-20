// tests/test_tokenizer.cpp — Unit tests for src/tokenizer.{h,cpp,_bpe.cpp,_spm.cpp}
//
// Pure in-memory tests: constructs tokenizers from small synthetic vocabs.
// No GGUF model files needed.
//
// Usage: ./build/test-tokenizer
// Exit 0 = all pass, non-zero = failure.

#include "tokenizer.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "  FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while (0)

// ===========================================================================
// WordPieceTokenizer
// ===========================================================================

static void test_wordpiece_basic() {
    printf("test_wordpiece_basic...\n");

    // Minimal vocab: [PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 hello=4 world=5
    std::vector<std::string> vocab = {"[PAD]","[UNK]","[CLS]","[SEP]","hello","world"};
    WordPieceTokenizer tok;
    tok.load(vocab, /*cls*/2, /*sep*/3, /*unk*/1, /*pad*/0, /*max_len*/8);

    auto r = tok.encode("hello world");
    // The implementation pads to seq_len (not max_length) when seq_len <= max_length.
    // "hello world" → [CLS, hello, world, SEP] = 4 tokens
    CHECK(r.ids.size() == 4, "encode: output size == seq_len (no trailing pads when short)");
    CHECK(r.ids[0] == 2, "encode: ids[0] == CLS");
    CHECK(r.ids[1] == 4, "encode: ids[1] == hello");
    CHECK(r.ids[2] == 5, "encode: ids[2] == world");
    CHECK(r.ids[3] == 3, "encode: ids[3] == SEP");
    // Attention mask: 1 for all 4 real tokens (no trailing PAD in short seqs)
    CHECK(r.attn_mask[0] == 1, "encode: mask[0] == 1 (CLS)");
    CHECK(r.attn_mask[1] == 1, "encode: mask[1] == 1 (hello)");
    CHECK(r.attn_mask[2] == 1, "encode: mask[2] == 1 (world)");
    CHECK(r.attn_mask[3] == 1, "encode: mask[3] == 1 (SEP)");
}

static void test_wordpiece_subword() {
    printf("test_wordpiece_subword...\n");

    // Vocab includes a subword continuation
    // good=4  ##bye=5  →  "goodbye" → [good, ##bye]
    std::vector<std::string> vocab = {"[PAD]","[UNK]","[CLS]","[SEP]","good","##bye"};
    WordPieceTokenizer tok;
    tok.load(vocab, 2, 3, 1, 0, 8);

    auto r = tok.encode("goodbye");
    CHECK(r.ids[1] == 4, "subword: ids[1] == good");
    CHECK(r.ids[2] == 5, "subword: ids[2] == ##bye");
    CHECK(r.ids[3] == 3, "subword: ids[3] == SEP");
}

static void test_wordpiece_unk() {
    printf("test_wordpiece_unk...\n");

    std::vector<std::string> vocab = {"[PAD]","[UNK]","[CLS]","[SEP]","hello"};
    WordPieceTokenizer tok;
    tok.load(vocab, 2, 3, 1, 0, 8);

    // "xyz" has no match → UNK
    auto r = tok.encode("xyz");
    CHECK(r.ids[1] == 1, "unk: unrecognised word → UNK id");
    CHECK(r.ids[2] == 3, "unk: SEP after UNK");
    CHECK(r.attn_mask[1] == 1, "unk: UNK token is real (mask=1)");
}

static void test_wordpiece_lowercase() {
    printf("test_wordpiece_lowercase...\n");

    std::vector<std::string> vocab = {"[PAD]","[UNK]","[CLS]","[SEP]","hello"};
    WordPieceTokenizer tok;
    tok.load(vocab, 2, 3, 1, 0, 8, /*do_lower_case*/true);

    // "Hello" with do_lower_case=true should map to hello=4
    auto r = tok.encode("Hello");
    CHECK(r.ids[1] == 4, "lowercase: Hello→hello maps to id 4");

    // With do_lower_case=false: "Hello" not in vocab → UNK
    WordPieceTokenizer tok2;
    tok2.load(vocab, 2, 3, 1, 0, 8, /*do_lower_case*/false);
    auto r2 = tok2.encode("Hello");
    CHECK(r2.ids[1] == 1, "no-lowercase: Hello not found → UNK");
}

static void test_wordpiece_punctuation() {
    printf("test_wordpiece_punctuation...\n");

    // Punctuation splits into its own token
    std::vector<std::string> vocab = {"[PAD]","[UNK]","[CLS]","[SEP]","hello","world",","};
    WordPieceTokenizer tok;
    tok.load(vocab, 2, 3, 1, 0, 16);

    auto r = tok.encode("hello,world");
    // Expected: [CLS=2, hello=4, ,=6, world=5, SEP=3, ...]
    CHECK(r.ids[1] == 4, "punct: ids[1] == hello");
    CHECK(r.ids[2] == 6, "punct: ids[2] == comma");
    CHECK(r.ids[3] == 5, "punct: ids[3] == world");
}

static void test_wordpiece_encode_pair() {
    printf("test_wordpiece_encode_pair...\n");

    std::vector<std::string> vocab = {"[PAD]","[UNK]","[CLS]","[SEP]","hello","world"};
    WordPieceTokenizer tok;
    tok.load(vocab, 2, 3, 1, 0, 16);

    auto r = tok.encode_pair("hello", "world");
    // Expected: [CLS=2, hello=4, SEP=3, world=5, SEP=3, ...]
    CHECK(r.ids[0] == 2, "pair: ids[0] == CLS");
    CHECK(r.ids[1] == 4, "pair: ids[1] == hello");
    CHECK(r.ids[2] == 3, "pair: ids[2] == SEP");
    CHECK(r.ids[3] == 5, "pair: ids[3] == world");
    CHECK(r.ids[4] == 3, "pair: ids[4] == SEP");
    // type_ids: segment A = 0, segment B = 1
    CHECK(r.type_ids[1] == 0, "pair: type_ids[1] == 0 (seg A)");
    CHECK(r.type_ids[3] == 1, "pair: type_ids[3] == 1 (seg B)");
    CHECK(r.type_ids[4] == 1, "pair: type_ids[4] == 1 (trailing SEP in B)");
}

static void test_wordpiece_ollama_vocab() {
    printf("test_wordpiece_ollama_vocab...\n");

    // Ollama format: whole-word tokens prefixed with ▁, subwords have no ##
    // ▁hello → hello, ing → ##ing (conversion happens in load)
    static const std::string SP = "\xe2\x96\x81";  // ▁ (U+2581)
    std::vector<std::string> vocab = {
        "[PAD]",  // 0
        "[UNK]",  // 1
        "[CLS]",  // 2
        "[SEP]",  // 3
        SP + "hello",   // 4 → should become "hello"
        SP + "world",   // 5 → should become "world"
        "ing",    // 6 → should become "##ing"
    };
    WordPieceTokenizer tok;
    tok.load(vocab, 2, 3, 1, 0, 16);

    // "hello world" should tokenize the same as with the standard format
    auto r = tok.encode("hello world");
    CHECK(r.ids[1] == 4, "ollama: hello → id 4");
    CHECK(r.ids[2] == 5, "ollama: world → id 5");

    // "##ing" should be accessible internally (won't appear at word start)
    // Test subword by checking tok.token_str
    CHECK(tok.token_str(4) == "hello", "ollama: token_str(4) == hello");
    CHECK(tok.token_str(6) == "##ing", "ollama: token_str(6) == ##ing");
}

static void test_wordpiece_maxlength_truncation() {
    printf("test_wordpiece_maxlength_truncation...\n");

    std::vector<std::string> vocab = {"[PAD]","[UNK]","[CLS]","[SEP]","a","b","c","d","e"};
    WordPieceTokenizer tok;
    tok.load(vocab, 2, 3, 1, 0, /*max_length*/5);

    // "a b c d e" → 5 ids: [CLS, a, b, c, SEP] (d,e truncated to fit max_length=5)
    auto r = tok.encode("a b c d e");
    CHECK((int)r.ids.size() == 5, "truncate: output size == max_length");
    CHECK(r.ids[0] == 2, "truncate: ids[0] == CLS");
    CHECK(r.ids[1] == 4, "truncate: ids[1] == a");
    CHECK(r.ids[2] == 5, "truncate: ids[2] == b");
    CHECK(r.ids[3] == 6, "truncate: ids[3] == c");
    CHECK(r.ids[4] == 3, "truncate: ids[4] == SEP (not d/e)");
}

// ===========================================================================
// SentencePieceTokenizer
// ===========================================================================

static void test_spm_basic() {
    printf("test_spm_basic...\n");

    // SPM vocab: ▁ markers on word-start tokens
    static const std::string SP = "\xe2\x96\x81";  // ▁ (U+2581)
    std::vector<std::string> vocab = {
        "<pad>",        // 0
        "<s>",          // 1  (BOS)
        "</s>",         // 2  (EOS)
        "<unk>",        // 3
        SP + "hello",   // 4
        SP + "world",   // 5
    };
    std::vector<float> scores(vocab.size(), 0.0f);
    SentencePieceTokenizer tok;
    tok.load(vocab, scores, /*bos*/1, /*eos*/2, /*unk*/3, /*pad*/0, /*max_len*/16);

    auto r = tok.encode("hello world");
    // Expected: [BOS=1, ▁hello=4, ▁world=5, EOS=2, PAD=0, ...]
    CHECK(r.ids[0] == 1, "spm basic: ids[0] == BOS");
    CHECK(r.ids[1] == 4, "spm basic: ids[1] == ▁hello");
    CHECK(r.ids[2] == 5, "spm basic: ids[2] == ▁world");
    CHECK(r.ids[3] == 2, "spm basic: ids[3] == EOS");
    CHECK(r.attn_mask[1] == 1, "spm basic: mask[1] == 1 (▁hello)");
    CHECK(r.attn_mask[3] == 1, "spm basic: mask[3] == 1 (EOS)");
}

static void test_spm_unk() {
    printf("test_spm_unk...\n");

    static const std::string SP = "\xe2\x96\x81";
    std::vector<std::string> vocab = {"<pad>","<s>","</s>","<unk>", SP + "hello"};
    std::vector<float> scores(vocab.size(), 0.0f);
    SentencePieceTokenizer tok;
    tok.load(vocab, scores, 1, 2, 3, 0, 16);

    // "xyz" → prepend space → "▁xyz" (6 bytes: 3-byte ▁ + x y z).
    // Byte fallback fires per byte → 6 UNK tokens (one per byte), then EOS.
    auto r = tok.encode("xyz");
    CHECK(r.ids[0] == 1, "spm unk: ids[0] == BOS");
    CHECK(r.ids[1] == 3, "spm unk: ids[1] == UNK (byte fallback)");
    // EOS should appear somewhere after the UNK run
    bool has_eos = false;
    for (int id : r.ids) if (id == 2) { has_eos = true; break; }
    CHECK(has_eos, "spm unk: EOS present after UNK byte-fallback tokens");
}

static void test_spm_subword() {
    printf("test_spm_subword...\n");

    // SPM with sub-word vocab: ▁go + od → ▁good, or ▁go + ##od
    // In SPM (unigram), subwords don't have ## prefix; they just don't have ▁
    static const std::string SP = "\xe2\x96\x81";
    std::vector<std::string> vocab = {
        "<pad>","<s>","</s>","<unk>",
        SP + "good",   // 4
        SP + "go",     // 5
        "od",          // 6  (subword without ▁ = continuation)
        SP + "world",  // 7
    };
    std::vector<float> scores = {0.0f, 0.0f, 0.0f, 0.0f, -1.0f, -2.0f, -3.0f, -1.0f};
    SentencePieceTokenizer tok;
    tok.load(vocab, scores, 1, 2, 3, 0, 16);

    // "good world" — should prefer ▁good (longer match = higher score) over ▁go+od
    auto r = tok.encode("good world");
    CHECK(r.ids[1] == 4, "spm subword: ids[1] == ▁good (longest match)");
    CHECK(r.ids[2] == 7, "spm subword: ids[2] == ▁world");
}

// ===========================================================================
// BPETokenizer (SentencePiece style — simpler than GPT-2 byte-level)
// ===========================================================================

static void test_bpe_spm_basic() {
    printf("test_bpe_spm_basic...\n");

    static const std::string SP = "\xe2\x96\x81";
    // Vocab: individual chars + merged forms
    // hello: h e l l o  → merge h+e → he, he+l → hel, hel+l → hell, hell+o → hello
    std::vector<std::string> vocab = {
        "<pad>",           // 0
        "<bos>",           // 1
        "<eos>",           // 2
        SP,                // 3  (space marker alone)
        "h", "e", "l", "o",   // 4 5 6 7
        "he", "hel", "hell", "hello",  // 8 9 10 11
        SP + "w", SP + "wo", SP + "wor", SP + "worl", SP + "world",  // 12..16
    };
    std::vector<std::string> merges = {
        "h e",
        "he l",
        "hel l",
        "hell o",
        "\xe2\x96\x81 w",
        "\xe2\x96\x81w o",
        "\xe2\x96\x81wo r",
        "\xe2\x96\x81wor l",
        "\xe2\x96\x81worl d",
    };

    BPETokenizer tok;
    // spm_style=true, bos_id=1, suffix_id (EOS)=2
    tok.load(vocab, merges, /*eos*/2, /*pad*/0, /*suffix*/2,
             /*bos*/1, /*spm_style*/true, /*max_len*/16);

    auto r = tok.encode("hello");
    // Expected: BOS + hello + EOS = [1, 11, 2, 0, ...]
    CHECK(r.ids[0] == 1, "bpe spm: ids[0] == BOS");
    CHECK(r.ids[1] == 11, "bpe spm: ids[1] == hello (fully merged)");
    CHECK(r.ids[2] == 2, "bpe spm: ids[2] == EOS");
}

// ===========================================================================
// main
// ===========================================================================

int main() {
    printf("=== Tokenizer Unit Tests ===\n\n");

    test_wordpiece_basic();
    test_wordpiece_subword();
    test_wordpiece_unk();
    test_wordpiece_lowercase();
    test_wordpiece_punctuation();
    test_wordpiece_encode_pair();
    test_wordpiece_ollama_vocab();
    test_wordpiece_maxlength_truncation();

    test_spm_basic();
    test_spm_unk();
    test_spm_subword();

    test_bpe_spm_basic();

    printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
