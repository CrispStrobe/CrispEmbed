// src/core/json_value.h — small strict JSON DOM preserving Python's int/float
// distinction (json.loads yields int for "3" and float for "3.0"/"3e0").
// Used where a native port must validate data exactly as Python helpers do.
#pragma once

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace core_json_value {

struct value {
    enum kind_t { null_t, bool_t, int_t, float_t, string_t, array_t, object_t } kind = null_t;
    bool b = false;
    long long i = 0;
    double f = 0;
    std::string s;
    std::vector<value> a;
    std::vector<std::pair<std::string, value>> o; // insertion order, last duplicate wins on lookup

    bool is_number() const { return kind == int_t || kind == float_t; }
    double number() const { return kind == int_t ? double(i) : f; }
    const value * get(const std::string & key) const {
        const value * found = nullptr;
        for (const auto & kv : o)
            if (kv.first == key) found = &kv.second;
        return found;
    }
};

class parser {
public:
    explicit parser(const std::string & text) : t(text) {}
    value parse() {
        value v = parse_value(0);
        ws();
        if (p != t.size()) fail("trailing characters");
        return v;
    }

private:
    const std::string & t;
    size_t p = 0;
    [[noreturn]] void fail(const char * what) {
        throw std::invalid_argument(std::string("invalid JSON: ") + what + " at byte " + std::to_string(p));
    }
    void ws() {
        while (p < t.size() && (t[p] == ' ' || t[p] == '\t' || t[p] == '\n' || t[p] == '\r')) ++p;
    }
    bool lit(const char * w) {
        size_t n = 0;
        while (w[n]) ++n;
        if (t.compare(p, n, w) == 0) {
            p += n;
            return true;
        }
        return false;
    }
    static void put_utf8(std::string & out, unsigned cp) {
        if (cp < 0x80)
            out += char(cp);
        else if (cp < 0x800) {
            out += char(0xC0 | (cp >> 6));
            out += char(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            out += char(0xE0 | (cp >> 12));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        } else {
            out += char(0xF0 | (cp >> 18));
            out += char(0x80 | ((cp >> 12) & 0x3F));
            out += char(0x80 | ((cp >> 6) & 0x3F));
            out += char(0x80 | (cp & 0x3F));
        }
    }
    unsigned hex4() {
        if (p + 4 > t.size()) fail("short \\u escape");
        unsigned v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = t[p++];
            v <<= 4;
            if (c >= '0' && c <= '9')
                v |= unsigned(c - '0');
            else if (c >= 'a' && c <= 'f')
                v |= unsigned(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                v |= unsigned(c - 'A' + 10);
            else
                fail("bad \\u escape");
        }
        return v;
    }
    std::string parse_string() {
        if (t[p] != '"') fail("expected string");
        ++p;
        std::string out;
        while (true) {
            if (p >= t.size()) fail("unterminated string");
            const unsigned char c = (unsigned char)t[p++];
            if (c == '"') break;
            if (c < 0x20) fail("control character in string");
            if (c != '\\') {
                out += char(c);
                continue;
            }
            if (p >= t.size()) fail("bad escape");
            const char e = t[p++];
            switch (e) {
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            case '/':
                out += '/';
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u': {
                unsigned cp = hex4();
                if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 <= t.size() && t[p] == '\\' && t[p + 1] == 'u') {
                    const size_t save = p;
                    p += 2;
                    const unsigned lo = hex4();
                    if (lo >= 0xDC00 && lo <= 0xDFFF)
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    else
                        p = save;
                }
                if (cp >= 0xD800 && cp <= 0xDFFF) fail("lone surrogate is not representable as UTF-8");
                put_utf8(out, cp);
                break;
            }
            default:
                fail("bad escape");
            }
        }
        return out;
    }
    value parse_value(int depth) {
        if (depth > 256) fail("nesting too deep");
        ws();
        if (p >= t.size()) fail("unexpected end");
        value v;
        const char c = t[p];
        if (c == '{') {
            ++p;
            v.kind = value::object_t;
            ws();
            if (p < t.size() && t[p] == '}') {
                ++p;
                return v;
            }
            while (true) {
                ws();
                if (p >= t.size()) fail("unexpected end");
                std::string key = parse_string();
                ws();
                if (p >= t.size() || t[p] != ':') fail("expected ':'");
                ++p;
                v.o.emplace_back(std::move(key), parse_value(depth + 1));
                ws();
                if (p < t.size() && t[p] == ',') {
                    ++p;
                    continue;
                }
                if (p < t.size() && t[p] == '}') {
                    ++p;
                    return v;
                }
                fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            ++p;
            v.kind = value::array_t;
            ws();
            if (p < t.size() && t[p] == ']') {
                ++p;
                return v;
            }
            while (true) {
                v.a.push_back(parse_value(depth + 1));
                ws();
                if (p < t.size() && t[p] == ',') {
                    ++p;
                    continue;
                }
                if (p < t.size() && t[p] == ']') {
                    ++p;
                    return v;
                }
                fail("expected ',' or ']'");
            }
        }
        if (c == '"') {
            v.kind = value::string_t;
            v.s = parse_string();
            return v;
        }
        if (lit("true")) {
            v.kind = value::bool_t;
            v.b = true;
            return v;
        }
        if (lit("false")) {
            v.kind = value::bool_t;
            return v;
        }
        if (lit("null")) return v;
        if (lit("NaN")) {
            v.kind = value::float_t;
            v.f = std::nan("");
            return v;
        }
        if (lit("Infinity")) {
            v.kind = value::float_t;
            v.f = HUGE_VAL;
            return v;
        }
        if (lit("-Infinity")) {
            v.kind = value::float_t;
            v.f = -HUGE_VAL;
            return v;
        }
        const size_t start = p;
        if (t[p] == '-') ++p;
        if (p >= t.size() || !(t[p] >= '0' && t[p] <= '9')) fail("bad number");
        if (t[p] == '0')
            ++p;
        else
            while (p < t.size() && t[p] >= '0' && t[p] <= '9') ++p;
        bool is_float = false;
        if (p < t.size() && t[p] == '.') {
            is_float = true;
            ++p;
            if (p >= t.size() || !(t[p] >= '0' && t[p] <= '9')) fail("bad fraction");
            while (p < t.size() && t[p] >= '0' && t[p] <= '9') ++p;
        }
        if (p < t.size() && (t[p] == 'e' || t[p] == 'E')) {
            is_float = true;
            ++p;
            if (p < t.size() && (t[p] == '+' || t[p] == '-')) ++p;
            if (p >= t.size() || !(t[p] >= '0' && t[p] <= '9')) fail("bad exponent");
            while (p < t.size() && t[p] >= '0' && t[p] <= '9') ++p;
        }
        const std::string num = t.substr(start, p - start);
        if (is_float) {
            v.kind = value::float_t;
            v.f = std::strtod(num.c_str(), nullptr);
        } else {
            v.kind = value::int_t;
            errno = 0;
            v.i = std::strtoll(num.c_str(), nullptr, 10);
            if (errno) { // Python ints are unbounded; keep the magnitude as a float view
                v.kind = value::float_t;
                v.f = std::strtod(num.c_str(), nullptr);
            }
        }
        return v;
    }
};

inline value parse(const std::string & text) {
    return parser(text).parse();
}

} // namespace core_json_value
