// Test-only token preflight; opens metadata, allocates no inference backend.
#include "core/clean_exit.h"
#include "core/json.h"
#include "moedern_text_frontend.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
namespace {
void require(bool ok, const char * why) {
    if (!ok) throw std::runtime_error(why);
}
std::string quote(const std::string & value) {
    return "\"" + core_json::json_escape_strict(value) + "\"";
}
template <class T> void values(std::ostream & out, const char * name, const std::vector<T> & input) {
    out << ",\"" << name << "\":[";
    for (size_t i = 0; i < input.size(); ++i) {
        if (i) out << ',';
        out << +input[i];
    }
    out << ']';
}
std::string unhex(const std::string & input) {
    if (input == "-") return {};
    require(input.size() % 2 == 0 && input.size() <= 131072, "bounded even UTF8 hex required");
    auto nibble = [](char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        throw std::runtime_error("invalid hex");
    };
    std::string result;
    result.reserve(input.size() / 2);
    for (size_t i = 0; i < input.size(); i += 2) result.push_back(char(nibble(input[i]) * 16 + nibble(input[i + 1])));
    return result;
}
int run(int argc, char ** argv) {
    require(argc == 5, "MODEL RAW-TEXT.tsv OUTPUT.jsonl f32|q4k");
    const std::string mode = argv[4];
    require(mode == "f32" || mode == "q4k", "explicit profile required");
    moedern_colbert::text_frontend frontend;
    std::string error;
    const bool loaded = mode == "q4k" ? frontend.load_packed_q4k(argv[1], error) : frontend.load_packed(argv[1], error);
    require(loaded, error.c_str());
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3]);
    require(bool(input) && bool(output), "token protocol file unavailable");
    std::string line;
    bool rejected = false;
    while (std::getline(input, line)) {
        const auto a = line.find('\t'), b = line.find('\t', a == std::string::npos ? a : a + 1);
        require(a != std::string::npos && b != std::string::npos && line.find('\t', b + 1) == std::string::npos,
                "three raw-text columns required");
        const auto key = line.substr(0, a), role = line.substr(a + 1, b - a - 1);
        output << "{\"key\":" << quote(key) << ",\"role\":" << quote(role);
        try {
            require(role == "query" || role == "document", "explicit role required");
            moedern_colbert::text_tokens tokens;
            const auto text = unhex(line.substr(b + 1));
            const bool ok = frontend.encode(
                text, role == "query" ? moedern_colbert::role::query : moedern_colbert::role::document, tokens, error);
            require(ok, error.c_str());
            values(output, "ids", tokens.ids);
            values(output, "supplied_mask", tokens.supplied_mask);
            values(output, "effective_mask", tokens.effective_mask);
            values(output, "scoring_mask", tokens.scoring_mask);
            values(output, "expansion_positions", tokens.expansion_positions);
            output << ",\"completed\":true";
        } catch (const std::exception & e) {
            output << ",\"completed\":false,\"error\":" << quote(e.what());
            rejected = true;
        }
        output << "}\n";
        require(bool(output), "token journal write failed");
    }
    return rejected ? 1 : 0;
}
} // namespace
int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        core_util::clean_exit(2);
    }
}
