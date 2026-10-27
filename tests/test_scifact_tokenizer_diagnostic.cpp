// Model-free test executable. Uses only frozen e35 BPE sources, no inference library.
#include "tokenizer.h"
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
int checked_main(int argc, char ** argv) {
    if (!core_env::on("CRISPEMBED_SCIFACT_TOKENIZER_DIAGNOSTIC")) return 2;
    try {
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
        raw.set_gpt2_regex_pretok(true);
        const uint32_t rows = number(stream);
        require(rows == 5483, "incomplete frozen panel");
        for (uint32_t row = 0; row < rows; ++row) {
            const auto id = string(stream, 64), text = string(stream, 65536);
            const bool detailed = number(stream) != 0;
            const auto ordinary = raw.encode(text).ids;
            std::vector<std::string> pretokens;
            if (detailed) pretokens = raw.diagnostic_gpt2_pretokens(text);
            const auto observed = raw.encode(text).ids;
            require(ordinary == observed, "observer changed native IDs");
            std::vector<int32_t> full = { 0 };
            full.insert(full.end(), ordinary.begin(), ordinary.end());
            std::vector<int32_t> clipped(full.begin(), full.begin() + std::min(full.size(), size_t(512)));
            printf("{\"row\":%u,\"id\":\"%s\",\"observer_bit_exact\":true,\"ids\":", row,
                   core_json::json_escape_strict(id).c_str());
            ids(clipped);
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
