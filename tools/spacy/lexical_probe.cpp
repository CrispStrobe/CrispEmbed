// Developer-only lexical differential adapter; never part of installed runtime.
#include "spacy_lexical.h"
#include "core/clean_exit.h"
#include "core/json.h"
#include "core/json_value.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace {
std::string quoted(const std::string & text) {
    return "\"" + core_json::json_escape_strict(text) + "\"";
}
int run(int argc, char ** argv) {
    if (argc != 2) throw std::invalid_argument("Usage: lexical-probe authenticated-resources.json");
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) throw std::invalid_argument("Required tokenizer resource missing");
    std::ostringstream bytes;
    bytes << file.rdbuf();
    crispembed::spacy::LexicalModel model(bytes.str());
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.size() > 8 * 1024 * 1024) throw std::invalid_argument("Probe JSON line exceeds bound");
        const auto input = core_json_value::parse(line);
        if (input.kind != core_json_value::value::string_t) throw std::invalid_argument("Expected JSON string input");
        const auto tokens = model.tokenize(input.s);
        std::cout << '[';
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const auto & token = tokens[i];
            if (i) std::cout << ',';
            std::cout << "{\"text\":" << quoted(token.text) << ",\"norm\":" << quoted(token.norm)
                      << ",\"prefix\":" << quoted(token.prefix) << ",\"suffix\":" << quoted(token.suffix)
                      << ",\"shape\":" << quoted(token.shape)
                      << ",\"orth_id\":" << quoted(std::to_string(token.orth_id))
                      << ",\"start_byte\":" << token.byte_start << ",\"end_byte\":" << token.byte_end
                      << ",\"start_codepoint\":" << token.scalar_start << ",\"end_codepoint\":" << token.scalar_end
                      << ",\"start_utf16\":" << token.utf16_start << ",\"end_utf16\":" << token.utf16_end
                      << ",\"whitespace\":" << quoted(token.space_after ? " " : "")
                      << ",\"is_space\":" << (token.is_space ? "true" : "false") << ",\"features\":[";
            for (std::size_t j = 0; j < token.features.size(); ++j) {
                if (j) std::cout << ',';
                std::cout << quoted(std::to_string(token.features[j]));
            }
            std::cout << "]}";
        }
        std::cout << ']' << std::endl;
    }
    return 0;
}
} // namespace
int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "LEXICAL_PROBE_FAILED: " << error.what() << '\n';
        core_util::clean_exit(1);
    }
}
