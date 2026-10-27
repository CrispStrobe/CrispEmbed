#include "josev2b_text_frontend.h"
#include "core/clean_exit.h"
#include "core/decision_schema.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdlib>

namespace {
void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_JOSEV_TEXT_FRONTEND", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_JOSEV_TEXT_FRONTEND", value, 1);
    else
        unsetenv("CRISPEMBED_JOSEV_TEXT_FRONTEND");
#endif
}
void check(bool v, const char * e) {
    if (!v) throw std::runtime_error(e);
}
std::string q(const std::string & s) {
    return "\"" + core_json::json_escape_strict(s) + "\"";
}
template <class T> void numbers(std::ostream & o, const std::vector<T> & v) {
    o << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) o << ',';
        o << v[i];
    }
    o << ']';
}
void span(std::ostream & o, josev2b::text_span s) {
    o << '[' << s.start << ',' << s.end << ']';
}
void dump(std::ostream & o, const josev2b::encoded_text & v) {
    o << "{\"input_ids\":";
    numbers(o, v.input_ids);
    o << ",\"attention_mask\":";
    numbers(o, v.attention_mask);
    o << ",\"record_id\":" << q(v.record_id) << ",\"media\":null,\"questions\":[";
    for (size_t i = 0; i < v.questions.size(); ++i) {
        if (i) o << ',';
        const auto & f = v.questions[i];
        o << "{\"question_id\":" << q(f.question_id) << ",\"question_type\":" << f.question_type
          << ",\"question_span\":";
        span(o, f.question_span);
        o << ",\"option_ids\":[";
        for (size_t j = 0; j < f.option_ids.size(); ++j) {
            if (j) o << ',';
            o << q(f.option_ids[j]);
        }
        o << "],\"option_spans\":[";
        for (size_t j = 0; j < f.option_spans.size(); ++j) {
            if (j) o << ',';
            span(o, f.option_spans[j]);
        }
        o << "],\"lexical_ids\":[";
        for (size_t j = 0; j < f.lexical_ids.size(); ++j) {
            if (j) o << ',';
            numbers(o, f.lexical_ids[j]);
        }
        o << "]}";
    }
    o << "]}";
}
int run(int argc, char ** argv) {
    check(argc >= 3, "usage: test-josev-text --gates GGUF | GGUF REQUESTS_JSONL OUTPUT_JSONL");
    if (std::string(argv[1]) == "--gates") {
        std::string error;
        for (const char * v : { static_cast<const char *>(nullptr), "", "0" }) {
            gate(v);
            josev2b::text_frontend disabled;
            check(!disabled.load(argv[2], error) && error == "Josev text frontend gate is off",
                  "off gate loaded or wrong diagnostic");
            josev2b::encoded_text out;
            out.input_ids = { 123 };
            out.record_id = "stale";
            check(!disabled.encode_record("{}", out, error) && out.input_ids.empty() && out.record_id.empty(),
                  "off output stale");
        }
        gate("1");
        josev2b::text_frontend enabled;
        const bool loaded = enabled.load(argv[2], error);
        check(loaded, error.c_str());
        gate("0");
        check(enabled.enabled(), "context gate changed with environment");
        josev2b::encoded_text kept;
        check(enabled.encode_record(R"({"state":"x","questions":{"q":{"type":"noul"}}})", kept, error) &&
                  !kept.input_ids.empty(),
              "existing context disabled by gate change");
        check(!enabled.load("does-not-exist", error) && !enabled.enabled(), "reload did not clear context");
        std::cout << "Josev unset/empty/0 rejection and context gate controls PASS\n";
        return 0;
    }
    check(argc == 4, "wrong arguments");
    josev2b::text_frontend frontend;
    std::string error;
    const bool loaded = frontend.load(argv[1], error);
    check(loaded, error.c_str());
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3]);
    check(bool(input) && bool(output), "file open failed");
    std::string line;
    size_t count = 0;
    while (std::getline(input, line)) {
        const auto entry = core_decision::Parser(line).parse();
        const auto & request = entry.get("request");
        check(request.kind == core_decision::Value::string, "request must be raw JSON string");
        const auto & length = entry.get("max_length");
        check(length.kind == core_decision::Value::number &&
                  length.text.find_first_not_of("0123456789") == std::string::npos,
              "max_length must be an unsigned integer");
        josev2b::encoded_text result;
        result.input_ids = { 123 };
        result.record_id = "stale";
        const bool ok = frontend.encode_record(request.text, result, error, std::stoi(length.text));
        output << "{\"id\":" << q(entry.get("id").text) << ",\"accepted\":" << (ok ? "true" : "false");
        if (ok) {
            output << ",\"encoded\":";
            dump(output, result);
        } else {
            check(result.input_ids.empty() && result.attention_mask.empty() && result.questions.empty() &&
                      result.record_id.empty(),
                  "rejected encode left stale output");
            output << ",\"error\":" << q(error);
        }
        output << "}\n";
        ++count;
    }
    check(count > 0 && output.good(), "empty requests or failed write");
    return 0;
}
} // namespace
int main(int argc, char ** argv) {
    int rc = 0;
    try {
        rc = run(argc, argv);
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        rc = 2;
    }
    core_util::clean_exit(rc);
}
