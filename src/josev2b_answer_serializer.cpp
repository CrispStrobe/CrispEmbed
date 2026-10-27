// Independent publisher SystemOne formatting: Josev-2B@6f0a31c8, Apache-2.0.
// Original prompts/formatting license: docs/josev2b-tokenizer-schema-LICENSE.txt.
#include "josev2b_answer_serializer.h"
#include "core/decision_schema.h"
#include "core/env_gate.h"
#include <algorithm>
#include <cmath>
#include <cfenv>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace josev2b {
namespace {
using V = core_decision::Value;
void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
const V & member(const V & value, const char * name) {
    require(value.kind == V::object, "Josev answer expected object");
    for (const auto & p : value.fields)
        if (p.first == name) return p.second;
    throw std::runtime_error(std::string("missing Josev answer member: ") + name);
}
void scalar(const std::string & s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = s[i++];
        if (c < 128) continue;
        const int n = c >= 0xc2 && c <= 0xdf ? 1 : c >= 0xe0 && c <= 0xef ? 2 : c >= 0xf0 && c <= 0xf4 ? 3 : 0;
        require(n && i + n <= s.size(), "invalid Unicode answer input");
        for (int k = 0; k < n; ++k) {
            const unsigned char d = s[i++];
            require(d >= 0x80 && d <= 0xbf &&
                        (k || !((c == 0xe0 && d < 0xa0) || (c == 0xed && d >= 0xa0) || (c == 0xf0 && d < 0x90) ||
                                (c == 0xf4 && d >= 0x90))),
                    "invalid Unicode answer input");
        }
    }
}
void validate_strings(const V & v) {
    if (v.kind == V::string) scalar(v.text);
    for (const auto & p : v.fields) {
        scalar(p.first);
        validate_strings(p.second);
    }
    for (const auto & item : v.list) validate_strings(item);
}
void strict_escapes(const std::string & s) {
    bool inside = false;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '"')
            inside = !inside;
        else if (inside && s[i] == '\\') {
            require(++i < s.size() && std::string("\"\\/bfnrtu").find(s[i]) != std::string::npos,
                    "invalid JSON escape");
            if (s[i] == 'u')
                for (int k = 0; k < 4; ++k)
                    require(++i < s.size() && std::string("0123456789abcdefABCDEF").find(s[i]) != std::string::npos,
                            "invalid Unicode escape");
        }
    }
}
std::string quote(const std::string & s) {
    scalar(s);
    return "\"" + core_json::json_escape_strict(s) + "\"";
}
std::string json_value(const V & v) {
    if (v.kind == V::string) return quote(v.text);
    if (v.kind == V::number) return core_decision::python_str(v);
    if (v.kind == V::null_value) return "null";
    if (v.kind == V::boolean) return v.flag ? "true" : "false";
    std::string s = v.kind == V::array ? "[" : "{";
    if (v.kind == V::array)
        for (size_t i = 0; i < v.list.size(); ++i) {
            if (i) s += ',';
            s += json_value(v.list[i]);
        }
    else
        for (size_t i = 0; i < v.fields.size(); ++i) {
            if (i) s += ',';
            s += quote(v.fields[i].first) + ":" + json_value(v.fields[i].second);
        }
    return s + (v.kind == V::array ? "]" : "}");
}
// Round the ACTUAL positive binary64 rational to 4 decimal places, ties even.
// Avoid x*10000 rounded in binary64. mantissa*625 is <2^63; 128 probabilities
// in [0,1] bound a non-normalized ordinal sum by 0+...+127=8128.
std::string rounded4(double value) {
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559 &&
                      std::numeric_limits<double>::digits == 53,
                  "IEEE binary64 required");
    require(std::isfinite(value) && value >= 0 && value <= 8128, "round4 value outside bounded positive binary64");
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const bool negative = (bits >> 63) != 0;
    const int exponent = int((bits >> 52) & 2047);
    uint64_t rounded = 0;
    if (exponent) {
        const uint64_t mantissa = (bits & ((uint64_t(1) << 52) - 1)) | (uint64_t(1) << 52);
        const uint64_t scaled = mantissa * 625;
        const int shift = 48 - (exponent - 1023);
        require(shift >= 36, "round4 exponent outside bounds");
        if (shift < 64) {
            rounded = scaled >> shift;
            const uint64_t remainder = scaled & ((uint64_t(1) << shift) - 1);
            const uint64_t midpoint = uint64_t(1) << (shift - 1);
            if (remainder > midpoint || (remainder == midpoint && (rounded & 1))) ++rounded;
        }
    }
    std::string fraction = std::to_string(10000 + rounded % 10000).substr(1);
    while (fraction.size() > 1 && fraction.back() == '0') fraction.pop_back();
    return (negative ? "-" : "") + std::to_string(rounded / 10000) + "." + fraction;
}
// Independent expression of CPython3.13.15's ordered float sum (Neumaier),
// not a plain fold or fsum. Every F32*index product is exact in binary64.
double expectation(const std::vector<float> & probabilities) {
    double result = 0.0, compensation = 0.0;
    for (size_t i = 0; i < probabilities.size(); ++i) {
        const double x = double(i) * double(probabilities[i]);
        const double t = result + x;
        if (std::abs(result) >= std::abs(x))
            compensation += (result - t) + x;
        else
            compensation += (x - t) + result;
        result = t;
    }
    if (compensation && std::isfinite(compensation)) result += compensation;
    return result;
}
std::vector<std::string> option_ids(const V & q, int kind) {
    if (kind == 0) {
        const auto & criteria = q.get("criteria");
        require(!criteria.truth() || criteria.kind == V::object, "noul criteria schema");
        return { "true", "false" };
    }
    const auto & criteria = member(q, "criteria");
    require(criteria.kind == (kind == 1 ? V::object : V::array), "criteria schema differs");
    std::vector<std::string> result;
    if (kind == 1) {
        for (const auto & p : criteria.fields) result.push_back(p.first);
        std::sort(result.begin(), result.end());
    } else
        for (size_t i = 0; i < criteria.list.size(); ++i) result.push_back(std::to_string(i));
    require(!result.empty(), "criteria must not be empty");
    return result;
}
void span(text_span s, size_t n, int previous) {
    require(s.start >= previous && s.end > s.start && s.end <= int(n), "encoded span bounds/order differs");
}
} // namespace
answer_serializer::answer_serializer() : enabled_(core_env::on("CRISPEMBED_JOSEV_ANSWER_SERIALIZER")) {}
bool answer_serializer::serialize(const std::string & text, const encoded_text & encoded,
                                  const std::vector<std::vector<float>> & probabilities, std::string & output,
                                  std::string & error) const {
    output.clear();
    error.clear();
    try {
        static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559 &&
                          std::numeric_limits<float>::digits == 24,
                      "native F32 required");
        require(enabled_, "Josev answer serializer gate is off");
        require(std::fegetround() == FE_TONEAREST, "binary64 nearest rounding mode required");
        require(text.size() <= 65536, "request exceeds bounded text profile");
        scalar(text);
        strict_escapes(text);
        const auto root = core_decision::Parser(text).parse();
        validate_strings(root);
        const auto & model = member(root, "model");
        member(root, "state");
        require(model.kind == V::string, "model must be a string");
        for (const auto & p : root.fields)
            require(p.first == "model" || p.first == "state" || p.first == "questions" || p.first == "id",
                    "unknown text request schema member");
        const auto & questions = member(root, "questions");
        require(questions.kind == V::object && !questions.fields.empty() && questions.fields.size() <= 16 &&
                    encoded.questions.size() == questions.fields.size() &&
                    probabilities.size() == encoded.questions.size(),
                "field count differs");
        require(!encoded.input_ids.empty() && encoded.input_ids.size() <= 512 &&
                    encoded.attention_mask.size() == encoded.input_ids.size(),
                "encoded token/mask count differs");
        for (size_t i = 0; i < encoded.input_ids.size(); ++i)
            require(encoded.input_ids[i] >= 0 && encoded.input_ids[i] < 248320 && encoded.attention_mask[i] == 1,
                    "encoded token/mask differs");
        bool has_id = false;
        for (const auto & p : root.fields)
            if (p.first == "id") has_id = true;
        require(encoded.record_id == (has_id ? core_decision::python_str(root.get("id")) : "unknown"),
                "encoded record ID differs");
        std::string answers = "{";
        size_t total_options = 0;
        int previous = 0;
        for (size_t i = 0; i < questions.fields.size(); ++i) {
            const auto & qid = questions.fields[i].first;
            const auto & q = questions.fields[i].second;
            const auto & type = member(q, "type");
            require(type.kind == V::string && (type.text == "noul" || type.text == "choice" || type.text == "score"),
                    "unknown question type");
            for (const auto & p : q.fields)
                require(p.first == "type" || p.first == "instructions" || p.first == "criteria",
                        "unknown question schema member");
            const int kind = type.text == "noul" ? 0 : type.text == "choice" ? 1 : 2;
            const auto & field = encoded.questions[i];
            const auto & probs = probabilities[i];
            const auto expected = option_ids(q, kind);
            require(!qid.empty() && field.question_id == qid && field.question_type == kind &&
                        field.option_ids == expected,
                    "encoded field/type/option IDs differ");
            require((total_options += expected.size()) <= 128 && probs.size() == expected.size() &&
                        field.option_spans.size() == expected.size() && field.lexical_ids.size() == expected.size(),
                    "option/probability count differs");
            span(field.question_span, encoded.input_ids.size(), previous);
            previous = field.question_span.end;
            for (size_t j = 0; j < probs.size(); ++j) {
                require(std::isfinite(probs[j]) && probs[j] >= 0 && probs[j] <= 1,
                        "probability must be finite F32 in [0,1]");
                const auto s = field.option_spans[j];
                span(s, encoded.input_ids.size(), previous);
                previous = s.end;
                require(field.lexical_ids[j] == std::vector<int32_t>(encoded.input_ids.begin() + s.start,
                                                                     encoded.input_ids.begin() + s.end),
                        "lexical IDs differ from option span");
            }
            std::string answer;
            if (kind == 0)
                answer = "{\"type\":\"noul\",\"noul\":" + rounded4(double(probs[0])) + "}";
            else {
                size_t winner = 0;
                for (size_t j = 1; j < probs.size(); ++j)
                    if (probs[j] > probs[winner]) winner = j;
                if (kind == 1) {
                    size_t choice = 0;
                    bool first = true;
                    for (const auto & p : q.get("criteria").fields) {
                        const size_t j =
                            size_t(std::find(expected.begin(), expected.end(), p.first) - expected.begin());
                        if (first || probs[j] > probs[choice]) {
                            choice = j;
                            first = false;
                        }
                    }
                    answer = "{\"type\":\"choice\",\"choice\":" + quote(expected[choice]) +
                             ",\"confidence\":" + rounded4(double(probs[choice]));
                } else {
                    answer = "{\"type\":\"score\",\"score\":" + rounded4(expectation(probs)) +
                             ",\"confidence\":" + rounded4(double(probs[winner])) + ",\"legend\":{";
                    const auto & criteria = q.get("criteria");
                    for (size_t j = 0; j < expected.size(); ++j) {
                        if (j) answer += ',';
                        answer += quote(expected[j]) + ":" + json_value(criteria.list[j]);
                    }
                    answer += '}';
                }
                answer += ",\"probabilities\":{";
                if (kind == 1) {
                    const auto & original_order = q.get("criteria").fields;
                    for (size_t j = 0; j < original_order.size(); ++j) {
                        if (j) answer += ',';
                        const auto & id = original_order[j].first;
                        const size_t index = size_t(std::find(expected.begin(), expected.end(), id) - expected.begin());
                        answer += quote(id) + ":" + rounded4(double(probs[index]));
                    }
                } else
                    for (size_t j = 0; j < expected.size(); ++j) {
                        if (j) answer += ',';
                        answer += quote(expected[j]) + ":" + rounded4(double(probs[j]));
                    }
                answer += "}}";
            }
            if (i) answers += ',';
            answers += quote(qid) + ":" + answer;
        }
        output = "{\"model\":" + quote(model.text) + ",\"answers\":" + answers +
                 "},\"usage\":{\"input_tokens\":" + std::to_string(encoded.input_ids.size()) + ",\"output_tokens\":0}}";
        return true;
    } catch (const std::exception & e) {
        error = e.what();
        return false;
    }
}
#ifdef CRISPEMBED_JOSEV_ANSWER_TESTING
namespace answer_test {
std::string rounded4(double value) {
    return josev2b::rounded4(value);
}
} // namespace answer_test
#endif
} // namespace josev2b
