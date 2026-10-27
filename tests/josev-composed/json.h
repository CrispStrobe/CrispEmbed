#pragma once
#include "core/decision_schema.h"
#include "josev2b_text_frontend.h"
#include <ostream>
namespace composed_json {
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
} // namespace composed_json
