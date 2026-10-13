#pragma once
// Ordered JSON and Python-compatible text rendering for the GemmaDecision
// inference schema. No training/gold-label logic is included.
#include "json.h"
#include "shortest_double.h"
#include "python_printable.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace core_decision {
struct Value {
    enum Kind { null_value, boolean, number, string, array, object } kind = null_value;
    std::string text;
    bool flag = false;
    std::vector<Value> list;
    std::vector<std::pair<std::string, Value>> fields;
    const Value & get(const std::string & key) const {
        static const Value empty;
        for (const auto & p : fields)
            if (p.first == key) return p.second;
        return empty;
    }
    bool truth() const {
        switch (kind) {
        case null_value:
            return false;
        case boolean:
            return flag;
        case number:
            return std::strtod(text.c_str(), nullptr) != 0;
        case string:
            return !text.empty();
        case array:
            return !list.empty();
        case object:
            return !fields.empty();
        }
        return false;
    }
};
class Parser {
    const std::string & input;
    size_t at = 0;
    void space() {
        while (at < input.size() && (input[at] == ' ' || input[at] == '\n' || input[at] == '\r' || input[at] == '\t'))
            ++at;
    }
    [[noreturn]] void bad() const { throw std::runtime_error("invalid decision JSON"); }
    std::string str() {
        size_t end = 0;
        std::string value;
        if (!core_json::json_decode_string(input, at, value, end)) bad();
        for (size_t i = at + 1; i + 1 < end; ++i)
            if ((unsigned char)input[i] < 32) bad();
        at = end;
        return value;
    }
    Value value(int depth) {
        if (depth > 64) throw std::runtime_error("decision JSON nesting limit exceeded");
        space();
        if (at >= input.size()) bad();
        Value v;
        char ch = input[at];
        if (ch == '"') {
            v.kind = Value::string;
            v.text = str();
            return v;
        }
        if (ch == '[' || ch == '{') {
            ++at;
            v.kind = ch == '[' ? Value::array : Value::object;
            const char close = ch == '[' ? ']' : '}';
            space();
            if (at < input.size() && input[at] == close) {
                ++at;
                return v;
            }
            for (;;) {
                space();
                std::string key;
                if (v.kind == Value::object) {
                    if (at >= input.size() || input[at] != '"') bad();
                    key = str();
                    space();
                    if (at >= input.size() || input[at++] != ':') bad();
                }
                auto item = value(depth + 1);
                if (v.kind == Value::array)
                    v.list.push_back(std::move(item));
                else {
                    auto it =
                        std::find_if(v.fields.begin(), v.fields.end(), [&](const auto & p) { return p.first == key; });
                    if (it == v.fields.end())
                        v.fields.emplace_back(key, std::move(item));
                    else
                        it->second = std::move(item);
                }
                space();
                if (at >= input.size()) bad();
                char sep = input[at++];
                if (sep == close) break;
                if (sep != ',') bad();
            }
            return v;
        }
        for (const auto & literal : { std::string("true"), std::string("false"), std::string("null") }) {
            if (input.compare(at, literal.size(), literal) == 0) {
                at += literal.size();
                v.kind = literal == "null" ? Value::null_value : Value::boolean;
                v.flag = literal == "true";
                return v;
            }
        }
        const size_t begin = at;
        if (input[at] == '-') ++at;
        if (at >= input.size()) bad();
        if (input[at] == '0')
            ++at;
        else {
            if (input[at] < '1' || input[at] > '9') bad();
            while (at < input.size() && input[at] >= '0' && input[at] <= '9') ++at;
        }
        if (at < input.size() && input[at] == '.') {
            ++at;
            size_t before = at;
            while (at < input.size() && input[at] >= '0' && input[at] <= '9') ++at;
            if (at == before) bad();
        }
        if (at < input.size() && (input[at] == 'e' || input[at] == 'E')) {
            ++at;
            if (at < input.size() && (input[at] == '+' || input[at] == '-')) ++at;
            size_t before = at;
            while (at < input.size() && input[at] >= '0' && input[at] <= '9') ++at;
            if (at == before) bad();
        }
        v.kind = Value::number;
        v.text = input.substr(begin, at - begin);
        if (v.text.find_first_of(".eE") != std::string::npos) {
            if (!std::isfinite(std::strtod(v.text.c_str(), nullptr))) bad();
        } else if (v.text.size() - (v.text[0] == '-') > 4300)
            throw std::runtime_error("integer exceeds Python reference's 4300-digit bound");
        return v;
    }

public:
    explicit Parser(const std::string & s) : input(s) {}
    Value parse() {
        if (input.size() > 1024 * 1024) throw std::runtime_error("decision JSON exceeds 1 MiB");
        auto v = value(0);
        space();
        if (at != input.size()) bad();
        return v;
    }
};
inline std::string quote(const std::string & s) {
    return "\"" + core_json::json_escape(s) + "\"";
}
inline std::string python_quote(const std::string & s) {
    const char q = s.find('\'') != std::string::npos && s.find('"') == std::string::npos ? '"' : '\'';
    std::string out(1, q);
    for (size_t i = 0; i < s.size();) {
        const size_t start = i;
        const uint32_t c = core_unicode::utf8_next(s.data(), s.size(), i);
        if (c == '\\' || c == (unsigned char)q) {
            out += '\\';
            out += (char)c;
        } else if (c == '\n')
            out += "\\n";
        else if (c == '\r')
            out += "\\r";
        else if (c == '\t')
            out += "\\t";
        else if (!core_unicode::python_is_printable15(c)) {
            const char * hex = "0123456789abcdef";
            const int digits = c <= 255 ? 2 : c <= 65535 ? 4 : 8;
            out += c <= 255 ? "\\x" : c <= 65535 ? "\\u" : "\\U";
            for (int shift = digits * 4 - 4; shift >= 0; shift -= 4) out += hex[(c >> shift) & 15];
        } else
            out.append(s, start, i - start);
    }
    out += q;
    return out;
}
inline std::string python_str(const Value & v, bool repr = false) {
    if (v.kind == Value::null_value) return "None";
    if (v.kind == Value::boolean) return v.flag ? "True" : "False";
    if (v.kind == Value::string) return repr ? python_quote(v.text) : v.text;
    if (v.kind == Value::number) {
        if (v.text.find_first_of(".eE") == std::string::npos) return v.text == "-0" ? "0" : v.text;
        const double x = std::strtod(v.text.c_str(), nullptr);
        char buf[64];
        const int length = crispembed_ryu_d2s_buffered_n(x, buf);
        std::string s(buf, (size_t)length);
        const auto marker = s.find('E');
        const int exponent = std::stoi(s.substr(marker + 1));
        if (exponent < -4 || exponent >= 16) {
            const int magnitude = std::abs(exponent);
            return s.substr(0, marker) + "e" + (exponent < 0 ? "-" : "+") + (magnitude < 10 ? "0" : "") +
                   std::to_string(magnitude);
        }
        std::string digits = s.substr(0, marker);
        const bool negative = digits[0] == '-';
        if (negative) digits.erase(0, 1);
        const auto dot = digits.find('.');
        if (dot != std::string::npos) digits.erase(dot, 1);
        const int point = 1 + exponent;
        if (point <= 0)
            s = "0." + std::string(-point, '0') + digits;
        else if (point >= (int)digits.size())
            s = digits + std::string(point - (int)digits.size(), '0') + ".0";
        else
            s = digits.substr(0, point) + "." + digits.substr(point);
        return negative ? "-" + s : s;
    }
    std::string out = v.kind == Value::array ? "[" : "{";
    if (v.kind == Value::array)
        for (size_t i = 0; i < v.list.size(); ++i) {
            if (i) out += ", ";
            out += python_str(v.list[i], true);
        }
    else
        for (size_t i = 0; i < v.fields.size(); ++i) {
            if (i) out += ", ";
            out += python_quote(v.fields[i].first) + ": " + python_str(v.fields[i].second, true);
        }
    out += v.kind == Value::array ? "]" : "}";
    return out;
}
inline Value unpack(const Value & v) {
    return v.kind == Value::string ? Parser(v.text).parse() : v;
}
inline std::string state_text(const std::string & workflow, Value state) {
    if (state.kind == Value::string) {
        size_t at = 0;
        while (at < state.text.size()) {
            size_t next = at;
            const auto cp = core_unicode::utf8_next(state.text.data(), state.text.size(), next);
            if (core_unicode::category(cp) != core_unicode::CAT_WS && !(cp >= 0x1c && cp <= 0x1f)) break;
            at = next;
        }
        // Python's empty stripped string also satisfies `prefix in "{["`;
        // preserve its JSON-parse failure instead of embedding a different state.
        if (at == state.text.size() || state.text[at] == '{' || state.text[at] == '[')
            state = Parser(state.text).parse();
    }
    if (state.kind != Value::object) return "workflow: " + workflow + " | state: " + python_str(state);
    std::vector<std::string> lines;
    auto walk = [&](auto && self, const std::string & prefix, const Value & v) -> void {
        if (v.kind == Value::object)
            for (const auto & p : v.fields) self(self, prefix.empty() ? p.first : prefix + " · " + p.first, p.second);
        else {
            std::string key = prefix;
            std::replace(key.begin(), key.end(), '_', ' ');
            lines.push_back(key + ": " + python_str(v));
        }
    };
    walk(walk, "", state);
    std::string out = "workflow: " + workflow + "\n";
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i) out += '\n';
        out += lines[i];
    }
    return out;
}
struct Question {
    std::string id, kind, instructions;
    std::vector<std::string> options, descriptions;
    int kind_id = 0;
};
inline bool integer(const std::string & s) {
    size_t i = 0;
    while (i < s.size() && s[i] == '-') ++i;
    if (i == s.size()) return false;
    for (; i < s.size(); ++i)
        if (s[i] < '0' || s[i] > '9') return false;
    return true;
}
inline long long level(const std::string & s) {
    size_t n;
    long long v = std::stoll(s, &n);
    if (n != s.size()) throw std::runtime_error("invalid score level");
    return v;
}
inline Question question(const std::string & id, const Value & data) {
    if (data.kind != Value::object) throw std::runtime_error("question must be an object");
    Question q;
    q.id = id;
    q.kind = data.get("type").truth() ? python_str(data.get("type")) : "choice";
    if (q.kind != "score" && q.kind != "noul") q.kind = "choice";
    q.kind_id = q.kind == "score" ? 1 : q.kind == "noul" ? 2 : 0;
    q.instructions = data.get("instructions").truth() ? python_str(data.get("instructions")) : id;
    const auto & criteria = data.get("criteria");
    if (q.kind == "noul") {
        q.options = { "false", "true" };
        q.descriptions = { "no", "yes" };
    } else if (q.kind == "score") {
        if (criteria.kind == Value::array)
            for (size_t i = 0; i < criteria.list.size(); ++i) {
                q.options.push_back(std::to_string(i));
                q.descriptions.push_back(python_str(criteria.list[i]));
            }
        else if (criteria.kind == Value::object) {
            bool all = true;
            for (const auto & p : criteria.fields) all &= integer(p.first);
            auto pairs = criteria.fields;
            if (all)
                std::stable_sort(pairs.begin(), pairs.end(),
                                 [](const auto & a, const auto & b) { return level(a.first) < level(b.first); });
            for (size_t i = 0; i < pairs.size(); ++i) {
                q.options.push_back(all ? pairs[i].first : std::to_string(i));
                q.descriptions.push_back(all ? python_str(pairs[i].second)
                                             : pairs[i].first + ": " + python_str(pairs[i].second));
            }
        }
    } else {
        if (criteria.kind == Value::object)
            for (const auto & p : criteria.fields) {
                q.options.push_back(p.first);
                q.descriptions.push_back(python_str(p.second));
            }
        else if (criteria.kind == Value::array)
            for (const auto & v : criteria.list) {
                q.options.push_back(python_str(v));
                q.descriptions.push_back(python_str(v));
            }
    }
    if (q.options.empty() || q.options.size() > 4096) throw std::runtime_error("question must have 1 to 4096 options");
    return q;
}
inline std::string question_text(const Question & q) {
    return q.kind + " question: " + q.instructions;
}
inline std::string option_text(const Question & q, size_t i) {
    std::string answer = q.descriptions[i];
    if (q.kind == "score")
        answer = "level " + q.options[i] + " of " + std::to_string(q.options.size() - 1) + ": " + answer;
    else if (q.kind == "choice")
        answer = q.options[i] == answer ? answer : q.options[i] + ": " + answer;
    return "question: " + q.instructions + " | answer: " + answer;
}
} // namespace core_decision
