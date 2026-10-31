// Hermetic guard for core_wordlevel (no model file), plus an optional parity
// mode driven by tests/test_surya25_tokenizer_parity.py:
//   test-wordlevel-added <vocab.bin> <cases.bin> <out.txt>
// vocab.bin: "WLA1", u32 n_tokens, n x (u32 len, bytes), then five i32 arrays
// (u32 count + values): model ids, raw added, normalized added, special, [unk].
// cases.bin: u32 count, count x (u32 len, bytes). out.txt: one line per case,
// space-separated ids, or "REJECT".
#include "core/wordlevel_added.h"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int failures = 0;
#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

static std::vector<int32_t> enc(const core_wordlevel::tokenizer & t, const std::string & s) {
    std::vector<int32_t> ids;
    if (!t.encode(s, ids)) return { -99 };
    return ids;
}

static void hermetic() {
    // ids: 0 <unk>(special) 1 <|im_end|>(special) 2 "a" 3 "b" 4 "c" 5 "<" 6 "." 7 "|" 8 "ü"
    //      9 "ab"(raw added) 10 "abc"(raw added) 11 ".<"(raw added) 12 "bc"(normalized added)
    //      13 "\n" 14 "ca"(normalized) 15 "c"(raw added duplicate of model text)
    std::vector<std::string> tokens = { "<unk>", "<|im_end|>", "a", "b", "c", "<", ".", "|", "\xc3\xbc",
                                        "ab", "abc", ".<", "bc", "\n", "ca", "c" };
    core_wordlevel::tokenizer t;
    CHECK(t.load(tokens, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 13 }, { 1, 9, 10, 11, 15 }, { 12, 14 }, { 0, 1 }, 0));
    // Longest at the leftmost start.
    CHECK((enc(t, "abc") == std::vector<int32_t>{ 10 }));
    CHECK((enc(t, "ab") == std::vector<int32_t>{ 9 }));
    // Leftmost wins over a longer later match: ".<" swallows the '<' of <|im_end|>.
    CHECK((enc(t, ".<|im_end|>") == std::vector<int32_t>{ 11, 7, 0, 0, 0, 0, 0, 0, 7, 0 }));
    CHECK((enc(t, "x<|im_end|>") == std::vector<int32_t>{ 0, 1 }));
    // Raw pass precedes normalized pass: raw "c" splits "bc" before the normalized trie sees it.
    CHECK((enc(t, "bc") == std::vector<int32_t>{ 3, 15 }));
    // Normalized tokens apply only inside raw-unmatched pieces.
    CHECK((enc(t, "b\n") == std::vector<int32_t>{ 3, 13 }));
    // Code-point fallback and <unk>.
    CHECK((enc(t, "\xc3\xbc") == std::vector<int32_t>{ 8 }));
    CHECK((enc(t, "\xe2\x82\xac") == std::vector<int32_t>{ 0 }));
    CHECK((enc(t, "") == std::vector<int32_t>{}));
    // Invalid UTF-8 rejects.
    CHECK((enc(t, "\xc3") == std::vector<int32_t>{ -99 }));
    CHECK((enc(t, "\xed\xa0\x80") == std::vector<int32_t>{ -99 }));
    std::string text;
    CHECK(t.decode({ 10, 1, 11, 0, 8 }, text) && text == "abc.<\xc3\xbc");
    CHECK(t.decode({ 10, 1 }, text, false) && text == "abc<|im_end|>");
    CHECK(!t.decode({ 99 }, text) && text.empty());
    // Load rejects duplicates, unknown ids and a non-special unk.
    core_wordlevel::tokenizer bad;
    CHECK(!bad.load(tokens, { 0, 2, 2 }, {}, {}, { 0 }, 0));
    CHECK(!bad.load(tokens, { 0 }, { 9, 9 }, {}, { 0 }, 0));
    CHECK(!bad.load(tokens, { 0 }, { 99 }, {}, { 0 }, 0));
    CHECK(!bad.load(tokens, { 0 }, {}, {}, {}, 0));
}

static bool read_u32(std::ifstream & f, uint32_t & v) { return bool(f.read(reinterpret_cast<char *>(&v), 4)); }
static bool read_str(std::ifstream & f, std::string & s) {
    uint32_t n = 0;
    if (!read_u32(f, n) || n > (64u << 20)) return false;
    s.resize(n);
    return n == 0 || bool(f.read(&s[0], n));
}
static bool read_ids(std::ifstream & f, std::vector<int32_t> & v) {
    uint32_t n = 0;
    if (!read_u32(f, n) || n > (16u << 20)) return false;
    v.resize(n);
    return n == 0 || bool(f.read(reinterpret_cast<char *>(v.data()), std::streamsize(n) * 4));
}

static int parity(const char * vocab_path, const char * cases_path, const char * out_path) {
    std::ifstream v(vocab_path, std::ios::binary), c(cases_path, std::ios::binary);
    char magic[4];
    uint32_t n = 0;
    if (!v.read(magic, 4) || std::memcmp(magic, "WLA1", 4) || !read_u32(v, n)) return 2;
    std::vector<std::string> tokens(n);
    for (auto & s : tokens)
        if (!read_str(v, s)) return 2;
    std::vector<int32_t> model, raw, normalized, special, unk;
    if (!read_ids(v, model) || !read_ids(v, raw) || !read_ids(v, normalized) || !read_ids(v, special) ||
        !read_ids(v, unk) || unk.size() != 1)
        return 2;
    core_wordlevel::tokenizer t;
    if (!t.load(std::move(tokens), model, raw, normalized, special, unk[0])) return 3;
    uint32_t count = 0;
    if (!read_u32(c, count)) return 2;
    std::FILE * out = std::fopen(out_path, "wb");
    if (!out) return 2;
    for (uint32_t i = 0; i < count; ++i) {
        std::string s;
        if (!read_str(c, s)) return 2;
        std::vector<int32_t> ids;
        if (!t.encode(s, ids)) {
            std::fputs("REJECT\n", out);
            continue;
        }
        for (size_t k = 0; k < ids.size(); ++k) std::fprintf(out, k ? " %d" : "%d", ids[k]);
        std::fputc('\n', out);
    }
    std::fclose(out);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 4) return parity(argv[1], argv[2], argv[3]);
    hermetic();
    std::printf("test-wordlevel-added: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
