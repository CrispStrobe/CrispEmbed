#include "spacy_lexical.h"
#include "core/clean_exit.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
using namespace crispembed::spacy;
std::size_t assertions = 0;
void require(bool condition, const char * message) {
    ++assertions;
    if (!condition) throw std::runtime_error(message);
}
template <class Exception, class Function> void rejects(Function operation, const char * message) {
    bool caught = false;
    try {
        operation();
    } catch (const Exception &) {
        caught = true;
    }
    require(caught, message);
}

int run(int argc, char ** argv) {
    require(spacy_string_id("") == 0, "Empty StringStore ID");
    require(spacy_string_id("X") == 101 && spacy_string_id("NORM") == 67, "Reserved StringStore symbols");
    require(spacy_string_id("Haus") == 18326629320658941346ULL, "Murmur original spelling");
    require(spacy_string_id("haus") == 17820212624685240070ULL, "Murmur lowercase spelling");
    require(spacy_string_id("Äpfel") == 9486589612853130994ULL, "Murmur original UTF-8 bytes");
    require(spacy_string_id(std::string_view("vor\0nach", 8)) == 5656654113200839298ULL, "Murmur embedded NUL");
    require(spacy_string_id(" ") == 8532415787641010193ULL, "Single space hash");
    require(utf8_boundaries("") == std::vector<std::size_t>{ 0 }, "Empty UTF-8 boundaries");
    require(utf8_boundaries("Aä👩‍🔬Z") == std::vector<std::size_t>({ 0, 1, 3, 7, 10, 14, 15 }),
            "Scalar boundaries preserve supplementary characters and ZWJ");
    require(utf8_boundaries(std::string_view("a\0b", 3)) == std::vector<std::size_t>({ 0, 1, 2, 3 }), "NUL scalar");
    for (const auto & malformed :
         { std::string("\x80"), std::string("\xC0\x80"), std::string("\xC1\xBF"), std::string("\xE0\x80\x80"),
           std::string("\xED\xA0\x80"), std::string("\xED\xBF\xBF"), std::string("\xF0\x80\x80\x80"),
           std::string("\xF4\x90\x80\x80"), std::string("\xF5\x80\x80\x80"), std::string("\xF8\x88\x80\x80\x80"),
           std::string("\xE2\x82"), std::string("\xC2x") }) {
        rejects<std::invalid_argument>([&] { utf8_boundaries(malformed); }, "Malformed UTF-8 accepted");
    }
    for (const auto & malformed : { "{}", "{\"x\":0,\"x\":1}", "{\"x\":NaN}", "{\"x\":\"\\ud800\"}",
                                    "{\"tokenizer\":{\"prefix_search\":[]}}" }) {
        rejects<std::invalid_argument>([&] { LexicalModel model(malformed); }, "Malformed resource accepted");
    }
    require(argc == 1 || argc == 2, "Usage: test-spacy-lexical [authenticated-resources.json]");
    const auto model_free_count = assertions;
    if (argc == 2) {
        std::ifstream file(argv[1], std::ios::binary);
        require(static_cast<bool>(file), "Required resource fixture missing");
        std::ostringstream bytes;
        bytes << file.rdbuf();
        LexicalModel model(bytes.str());
        require(model.tokenize("").empty(), "Explicit empty tokenization");
        rejects<NumericCancelled>([&] { model.tokenize("Ein Haus.", [] { return true; }); }, "Cancellation accepted");
        require(model.tokenize("Ein Haus.").size() == 3, "Cancellation recovery");
        std::string infixes = "a";
        for (unsigned i = 0; i < 10000; ++i) infixes += ",a";
        unsigned cancel_calls = 0;
        rejects<NumericCancelled>([&] { model.tokenize(infixes, [&] { return ++cancel_calls == 100; }); },
                                  "Infix collection must cancel before exhausting its match-count limit");
        require(model.tokenize("Ein Haus.").size() == 3, "Infix cancellation recovery");
        rejects<std::invalid_argument>([&] { model.tokenize(std::string(1024 * 1024 + 1, 'a')); },
                                       "Byte bound ignored");
        std::string many;
        for (unsigned i = 0; i < 2049; ++i) many += "a ";
        rejects<std::invalid_argument>([&] { model.tokenize(many); }, "Token bound ignored");
        rejects<std::invalid_argument>([&] { model.tokenize("\xED\xA0\x80"); }, "UTF-8 tokenization guard ignored");
        const auto tokens = model.tokenize("ΟΣ ΟΣΑ İ Σ ² X ");
        require(tokens.size() == 6, "Unicode lexical control count");
        require(tokens[0].norm == "ος" && tokens[1].norm == "οσα" && tokens[2].norm == "i̇", "Contextual lowercase");
        require(tokens[3].norm == "σ" && tokens[4].shape == "d" && tokens[5].features[3] == 101, "Digit/shape symbols");
        require(tokens[5].space_after && tokens[5].byte_end + 1 == std::string("ΟΣ ΟΣΑ İ Σ ² X ").size(),
                "Original trailing ASCII space");
        const auto nul = model.tokenize(std::string_view("a\0b", 3));
        require(nul.size() == 1 && nul[0].text.size() == 3 && nul[0].byte_end == 3, "Counted NUL tokenization");
    }
    std::cout << "SPACY_LEXICAL_CONTROLS model_free_assertions=" << model_free_count
              << " resource_assertions=" << assertions - model_free_count
              << " trained_inference=0 acceptance_inputs=0\n";
    return 0;
}
} // namespace

int main(int argc, char ** argv) {
    try {
        core_util::clean_exit(run(argc, argv));
    } catch (const std::exception & error) {
        std::cerr << "SPACY_LEXICAL_CONTROLS FAILED: " << error.what() << '\n';
        core_util::clean_exit(1);
    }
}
