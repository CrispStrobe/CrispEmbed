// Model-free candidate A/B executable. Original diagnostic source stays frozen.
#include "tokenizer.h"
#include "ultrabert_tokenizer_mode.h"
#include "core/clean_exit.h"
#include "core/env_gate.h"
#include "core/json.h"
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace {
void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
uint32_t number(std::istream & stream) {
    unsigned char bytes[4];
    stream.read(reinterpret_cast<char *>(bytes), 4);
    require(bool(stream), "truncated packet integer");
    return uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 | uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24;
}
std::string string(std::istream & stream, size_t bound) {
    const uint32_t n = number(stream);
    require(n <= bound, "packet string exceeds bound");
    std::string value(n, '\0');
    stream.read(value.data(), n);
    require(bool(stream), "truncated packet string");
    return value;
}
void ids(const std::vector<int32_t> & values) {
    printf("[");
    for (size_t i = 0; i < values.size(); ++i) printf("%s%d", i ? "," : "", values[i]);
    printf("]");
}
void unicode_gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN", value, 1);
    else
        unsetenv("CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN");
#endif
}
void controls() {
    unicode_gate("0");
    BPETokenizer a;
    ultrabert_profile::configure_tokenizer_mode(a);
    unicode_gate("1");
    BPETokenizer b;
    ultrabert_profile::configure_tokenizer_mode(b);
    unicode_gate("0");
    BPETokenizer c;
    ultrabert_profile::configure_tokenizer_mode(c);
    unicode_gate("1");
    const std::string quotes = u8" ‘stem";
    const std::vector<std::string> old = { quotes }, fixed = { u8" ‘", "stem" };
    require(a.diagnostic_gpt2_pretokens(quotes) == old && b.diagnostic_gpt2_pretokens(quotes) == fixed &&
                c.diagnostic_gpt2_pretokens(quotes) == old,
            "context A0/B1/A0 isolation failed");
    BPETokenizer generic;
    generic.set_gpt2_regex_pretok(true);
    require(generic.diagnostic_gpt2_pretokens(quotes) == old, "generic GPT2 changed under global UltraBERT flag");
    const auto clip = generic.diagnostic_clip_pretokens(u8"Sagte „Hallo“ heute");
    generic.set_ultrabert_unicode_gpt2(true);
    require(generic.diagnostic_clip_pretokens(u8"Sagte „Hallo“ heute") == clip, "CLIP splitter changed");
    require(generic.load({ "owned" }, {}, -1, 0, -1, -1, false, 1024), "generic reload failed");
    require(generic.diagnostic_gpt2_pretokens(quotes) == old, "generic reload inherited UltraBERT mode");
    const std::vector<std::pair<std::string, std::vector<std::string>>> cases = {
        { u8"a ‘stem cells’ in", { "a", u8" ‘", "stem", " cells", u8"’", " in" } },
        { u8"a (“caustic”) text", { "a", u8" (“", "caustic", u8"”)", " text" } },
        { u8"cafe\u0301 €42²", { "cafe", u8"\u0301", u8" €", u8"42²" } },
        { "x \t\n y", { "x", " \t\n", " y" } },
        { u8"x \t\u00A0y  ", { "x", " \t", u8"\u00A0", "y", "  " } },
        { "I'm I'M they've WE'RE", { "I", "'m", " I", "'", "M", " they", "'ve", " WE", "'", "RE" } },
        { u8"x👩‍🔬y", { "x", u8"👩‍🔬", "y" } },
        { u8"αβ ١٢Ⅳ 3.14", { u8"αβ", u8" ١٢Ⅳ", " 3", ".", "14" } },
        { u8"x\u2003é", { "x", u8"\u2003", u8"é" } },
        { u8"\u0378x", { u8"\u0378", "x" } },
        { u8"क्", { u8"क", u8"्" } },
        { "x\x1cy", { "x", "\x1c", "y" } },
        { " \t\n", { " \t\n" } },
        { "", {} }
    };
    for (const auto & item : cases) {
        const auto observed = b.diagnostic_gpt2_pretokens(item.first);
        require(observed == item.second, "owned Unicode GPT2 splitter control failed");
        std::string reconstructed;
        for (const auto & token : observed) reconstructed += token;
        require(reconstructed == item.first, "owned splitter lost input bytes");
    }
    std::printf("{\"context_isolation\":true,\"generic_unchanged\":true,\"clip_unchanged\":true,\"owned_cases\":%zu,"
                "\"passed\":true}\n",
                cases.size());
}
int checked_main(int argc, char ** argv) {
    if (!core_env::on("CRISPEMBED_SCIFACT_TOKENIZER_DIAGNOSTIC")) return 2;
    try {
        if (argc == 2 && std::string(argv[1]) == "--controls") {
            controls();
            return 0;
        }
        require(argc == 2, "expected metadata packet");
        std::ifstream stream(argv[1], std::ios::binary);
        char magic[8];
        stream.read(magic, 8);
        require(stream && std::string(magic, 8) == std::string("SCITOK1\0", 8), "wrong packet magic");
        std::vector<std::string> vocab, merges;
        for (auto * values : { &vocab, &merges }) {
            uint32_t n = number(stream);
            require(n == (values == &vocab ? 49152u : 48900u), "wrong vocabulary/merge count");
            for (uint32_t i = 0; i < n; ++i) values->push_back(string(stream, 4096));
        }
        require(vocab[0] == "<|endoftext|>" && vocab[15] == "<|sep|>" && vocab[16] == "<|mask|>", "wrong special IDs");
        BPETokenizer raw;
        require(raw.load(vocab, merges, -1, 0, -1, -1, false, 1048576), "load failed");
        ultrabert_profile::configure_tokenizer_mode(raw);
        const uint32_t rows = number(stream);
        require(rows == 5483, "incomplete frozen panel");
        for (uint32_t row = 0; row < rows; ++row) {
            const auto id = string(stream, 64), text = string(stream, 65536);
            const bool detailed = number(stream) != 0;
            const auto ordinary_tokens = raw.encode(text);
            const auto & ordinary = ordinary_tokens.ids;
            require(ordinary_tokens.attn_mask.size() == ordinary.size(), "native mask length differs");
            std::vector<std::string> pretokens;
            if (detailed) pretokens = raw.diagnostic_gpt2_pretokens(text);
            const auto observed = raw.encode(text);
            require(ordinary == observed.ids && ordinary_tokens.attn_mask == observed.attn_mask,
                    "observer changed native IDs/mask");
            std::vector<int32_t> full = { 0 };
            full.insert(full.end(), ordinary.begin(), ordinary.end());
            std::vector<int32_t> clipped(full.begin(), full.begin() + std::min(full.size(), size_t(512)));
            printf("{\"row\":%u,\"id\":\"%s\",\"observer_bit_exact\":true,\"ids\":", row,
                   core_json::json_escape_strict(id).c_str());
            ids(clipped);
            std::vector<int32_t> mask = { 1 };
            mask.insert(mask.end(), ordinary_tokens.attn_mask.begin(), ordinary_tokens.attn_mask.end());
            mask.resize(clipped.size());
            printf(",\"attention_mask\":");
            ids(mask);
            printf(",\"untruncated_ids\":");
            ids(full);
            if (detailed) {
                std::string reconstructed;
                printf(",\"actual_native_pretokens\":[");
                for (size_t i = 0; i < pretokens.size(); ++i) {
                    reconstructed += pretokens[i];
                    printf("%s\"%s\"", i ? "," : "", core_json::json_escape_strict(pretokens[i]).c_str());
                }
                printf("]");
                require(reconstructed == text, "native split did not preserve literal bytes");
            }
            printf("}\n");
        }
        require(stream.peek() == std::char_traits<char>::eof(), "extra packet bytes");
        return 0;
    } catch (const std::exception & error) {
        fprintf(stderr, "SciFact tokenizer diagnostic: %s\n", error.what());
        return 2;
    }
}
} // namespace
int main(int argc, char ** argv) {
    core_util::clean_exit(checked_main(argc, argv));
}
