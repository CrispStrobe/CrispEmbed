#include "josev2b_answer_serializer.h"
#include "core/clean_exit.h"
#include "core/decision_schema.h"
#include <cmath>
#include <cfenv>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using V = core_decision::Value;
void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
std::string quote(const std::string & s) {
    return "\"" + core_json::json_escape_strict(s) + "\"";
}
void gate(const char * value) {
#ifdef _WIN32
    _putenv_s("CRISPEMBED_JOSEV_ANSWER_SERIALIZER", value ? value : "");
#else
    if (value)
        setenv("CRISPEMBED_JOSEV_ANSWER_SERIALIZER", value, 1);
    else
        unsetenv("CRISPEMBED_JOSEV_ANSWER_SERIALIZER");
#endif
}
long long integer(const V & value) {
    check(value.kind == V::number && value.text.find_first_of(".eE") == std::string::npos, "expected integer");
    size_t end = 0;
    const auto number = std::stoll(value.text, &end);
    check(end == value.text.size(), "integer suffix");
    return number;
}
int narrow(const V & value) {
    auto n = integer(value);
    check(n >= INT32_MIN && n <= INT32_MAX, "int32 bounds");
    return int(n);
}
std::string string(const V & value) {
    check(value.kind == V::string, "expected string");
    return value.text;
}
const std::vector<V> & array(const V & value) {
    check(value.kind == V::array, "expected array");
    return value.list;
}
std::vector<int32_t> ints(const V & value) {
    std::vector<int32_t> result;
    for (const auto & item : array(value)) result.push_back(narrow(item));
    return result;
}
josev2b::text_span span(const V & value) {
    const auto & items = array(value);
    check(items.size() == 2, "span cardinality");
    return { narrow(items[0]), narrow(items[1]) };
}
josev2b::encoded_text encoded(const V & value) {
    josev2b::encoded_text result;
    result.input_ids = ints(value.get("input_ids"));
    result.attention_mask = ints(value.get("attention_mask"));
    result.record_id = string(value.get("record_id"));
    for (const auto & row : array(value.get("questions"))) {
        josev2b::text_field field;
        field.question_id = string(row.get("question_id"));
        field.question_type = narrow(row.get("question_type"));
        field.question_span = span(row.get("question_span"));
        for (const auto & item : array(row.get("option_ids"))) field.option_ids.push_back(string(item));
        for (const auto & item : array(row.get("option_spans"))) field.option_spans.push_back(span(item));
        for (const auto & item : array(row.get("lexical_ids"))) field.lexical_ids.push_back(ints(item));
        result.questions.push_back(field);
    }
    return result;
}
std::vector<std::vector<float>> probabilities(const V & value) {
    std::vector<std::vector<float>> result;
    for (const auto & row : array(value)) {
        std::vector<float> field;
        for (const auto & item : array(row)) {
            const auto n = integer(item);
            check(n >= 0 && n <= UINT32_MAX, "F32 bits bounds");
            const uint32_t bits = uint32_t(n);
            float p;
            std::memcpy(&p, &bits, sizeof(p));
            field.push_back(p);
        }
        result.push_back(field);
    }
    return result;
}
void gate_controls(const V & first) {
    const auto text = string(first.get("request"));
    const auto meta = encoded(first.get("encoded"));
    const auto probs = probabilities(first.get("probabilities_u32"));
    std::string result, error;
    for (const char * value : { static_cast<const char *>(nullptr), "", "0" }) {
        gate(value);
        josev2b::answer_serializer off;
        result = "stale";
        check(!off.serialize(text, meta, probs, result, error) && result.empty() &&
                  error == "Josev answer serializer gate is off",
              "off gate/output failed");
        gate("1");
        result = "stale";
        check(!off.serialize(text, meta, probs, result, error) && result.empty(),
              "disabled context enabled by environment");
    }
    gate("1");
    josev2b::answer_serializer on;
    const bool ok = on.serialize(text, meta, probs, result, error);
    check(ok, error.c_str());
    const auto expected = result;
    gate("0");
    result = "stale";
    check(on.serialize(text, meta, probs, result, error) && result == expected,
          "enabled context changed with environment");
    check(std::fesetround(FE_UPWARD) == 0, "cannot exercise non-nearest mode");
    result = "stale";
    const bool wrong_mode = on.serialize(text, meta, probs, result, error);
    check(std::fesetround(FE_TONEAREST) == 0, "cannot restore nearest mode");
    check(!wrong_mode && result.empty() && error == "binary64 nearest rounding mode required",
          "unsupported rounding mode accepted");
    check(on.serialize(text, meta, probs, result, error) && result == expected,
          "rounding-mode rejection recovery failed");
    std::cout << "unset/empty/0, immutable context/output and rounding-mode recovery controls PASS\n";
}
int run(int argc, char ** argv) {
    check(argc == 4, "usage: test-josev-answer --round|--answers|--gates INPUT_JSONL OUTPUT_JSONL");
    std::ifstream input(argv[2]);
    std::ofstream output(argv[3]);
    check(bool(input) && bool(output), "probe files unavailable");
    const std::string mode = argv[1];
    check(mode == "--round" || mode == "--answers" || mode == "--gates", "probe mode");
    std::string line;
    size_t count = 0;
    josev2b::answer_serializer context;
    while (std::getline(input, line)) {
        const auto entry = core_decision::Parser(line).parse();
        const auto id = string(entry.get("id"));
        if (mode == "--gates") {
            gate_controls(entry);
            return 0;
        }
        output << "{\"id\":" << quote(id);
        try {
            if (mode == "--round") {
#ifdef CRISPEMBED_JOSEV_ANSWER_TESTING
                const auto hex = string(entry.get("binary64_hex"));
                check(hex.size() == 16 && hex.find_first_not_of("0123456789abcdef") == std::string::npos,
                      "binary64 hex");
                const uint64_t bits = std::stoull(hex, nullptr, 16);
                double value;
                std::memcpy(&value, &bits, sizeof(value));
                const auto rounded = josev2b::answer_test::rounded4(value);
                output << ",\"accepted\":true,\"rounded_json\":" << quote(rounded);
#else
                throw std::runtime_error("round helper is absent from ordinary binary");
#endif
            } else {
                const auto meta = encoded(entry.get("encoded"));
                const auto probs = probabilities(entry.get("probabilities_u32"));
                std::string result = "stale", error;
                const bool ok = context.serialize(string(entry.get("request")), meta, probs, result, error);
                check(ok || result.empty(), "failed serializer retained output");
                output << ",\"accepted\":" << (ok ? "true" : "false");
                if (ok)
                    output << ",\"answer_json\":" << quote(result);
                else
                    output << ",\"error\":" << quote(error) << ",\"output_empty\":true";
            }
        } catch (const std::exception & error) {
            output << ",\"accepted\":false,\"probe_error\":" << quote(error.what());
        }
        output << "}\n";
        ++count;
    }
    check(count > 0 && output.good(), "empty requests/write failure");
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
