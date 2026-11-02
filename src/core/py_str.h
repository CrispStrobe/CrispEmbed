// src/core/py_str.h — CPython str/float/path semantics for exact native ports.
//
// Document helpers translated from Python (OXR, PaddleX XY-Cut++) operate on
// code points, Python's Unicode str predicates, banker's rounding, float
// floor division, repr() of floats and posixpath. These helpers reproduce
// those semantics so a C++ port can emit byte-identical output.
#pragma once

#include "py_str_tables.h"
#include "shortest_double.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace py_str {

using u32 = std::u32string;

inline bool in_ranges(const py_cp_range * table, size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        if (cp < table[mid].lo)
            hi = mid;
        else if (cp > table[mid].hi)
            lo = mid + 1;
        else
            return true;
    }
    return false;
}
#define PY_STR_TABLE(name) name, sizeof(name) / sizeof(name[0])
inline bool isspace(char32_t c) {
    return in_ranges(PY_STR_TABLE(k_isspace), uint32_t(c));
}
inline bool isalpha(char32_t c) {
    return in_ranges(PY_STR_TABLE(k_isalpha), uint32_t(c));
}
inline bool isalnum(char32_t c) {
    return in_ranges(PY_STR_TABLE(k_isalnum), uint32_t(c));
}
inline bool isdecimal(char32_t c) {
    return in_ranges(PY_STR_TABLE(k_isdecimal), uint32_t(c));
}
#undef PY_STR_TABLE
// re's Unicode \w for str patterns: alphanumeric or underscore.
inline bool is_word(char32_t c) {
    return c == U'_' || isalnum(c);
}
inline bool is_ascii_alpha(char32_t c) {
    return (c >= U'A' && c <= U'Z') || (c >= U'a' && c <= U'z');
}
inline bool is_ascii_digit(char32_t c) {
    return c >= U'0' && c <= U'9';
}

// Strict UTF-8 decode (Python bytes.decode("utf-8") without errors=...).
inline u32 from_utf8(const std::string & s) {
    u32 out;
    out.reserve(s.size());
    size_t i = 0;
    while (i < s.size()) {
        const unsigned char c = (unsigned char)s[i];
        uint32_t cp;
        int n;
        if (c < 0x80) {
            cp = c;
            n = 0;
        } else if (c >= 0xC2 && c <= 0xDF) {
            cp = c & 0x1F;
            n = 1;
        } else if (c >= 0xE0 && c <= 0xEF) {
            cp = c & 0x0F;
            n = 2;
        } else if (c >= 0xF0 && c <= 0xF4) {
            cp = c & 0x07;
            n = 3;
        } else {
            throw std::invalid_argument("invalid UTF-8 text");
        }
        if (n && i + size_t(n) >= s.size()) throw std::invalid_argument("truncated UTF-8 text");
        for (int k = 1; k <= n; ++k) {
            const unsigned char d = (unsigned char)s[i + size_t(k)];
            if ((d & 0xC0) != 0x80) throw std::invalid_argument("invalid UTF-8 continuation");
            cp = (cp << 6) | (d & 0x3F);
        }
        if ((n == 2 && cp < 0x800) || (n == 3 && (cp < 0x10000 || cp > 0x10FFFF)) || (cp >= 0xD800 && cp <= 0xDFFF))
            throw std::invalid_argument("invalid UTF-8 scalar");
        out.push_back(char32_t(cp));
        i += size_t(n) + 1;
    }
    return out;
}

inline void append_utf8(std::string & out, char32_t c) {
    const uint32_t cp = uint32_t(c);
    if (cp < 0x80) {
        out.push_back(char(cp));
    } else if (cp < 0x800) {
        out.push_back(char(0xC0 | (cp >> 6)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out.push_back(char(0xE0 | (cp >> 12)));
        out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(char(0xF0 | (cp >> 18)));
        out.push_back(char(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(char(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(char(0x80 | (cp & 0x3F)));
    }
}
inline std::string to_utf8(const u32 & s) {
    std::string out;
    out.reserve(s.size());
    for (char32_t c : s) append_utf8(out, c);
    return out;
}
inline u32 U(const char * ascii) {
    u32 out;
    for (const char * p = ascii; *p; ++p) out.push_back(char32_t((unsigned char)*p));
    return out;
}

// str.strip() / lstrip() / rstrip() with no arguments.
inline u32 strip(const u32 & s) {
    size_t a = 0, b = s.size();
    while (a < b && isspace(s[a])) ++a;
    while (b > a && isspace(s[b - 1])) --b;
    return s.substr(a, b - a);
}
inline u32 lstrip(const u32 & s) {
    size_t a = 0;
    while (a < s.size() && isspace(s[a])) ++a;
    return s.substr(a);
}
inline u32 rstrip(const u32 & s) {
    size_t b = s.size();
    while (b > 0 && isspace(s[b - 1])) --b;
    return s.substr(0, b);
}
// "".join(s.split())
inline u32 remove_space(const u32 & s) {
    u32 out;
    for (char32_t c : s)
        if (!isspace(c)) out.push_back(c);
    return out;
}
inline bool startswith(const u32 & s, const u32 & p, size_t at = 0) {
    return at <= s.size() && s.size() - at >= p.size() && s.compare(at, p.size(), p) == 0;
}
inline bool endswith(const u32 & s, const u32 & p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}
inline bool contains(const u32 & s, const u32 & p) {
    return s.find(p) != u32::npos;
}
// str.isspace(): nonempty and every code point is whitespace.
inline bool all_space(const u32 & s) {
    if (s.empty()) return false;
    for (char32_t c : s)
        if (!isspace(c)) return false;
    return true;
}

// Python 3 round(x) for a float: nearest integer, ties to even.
inline long long round_half_even(double x) {
    if (!std::isfinite(x)) throw std::invalid_argument("cannot round a non-finite float");
    double r = std::floor(x);
    const double diff = x - r;
    if (diff > 0.5 || (diff == 0.5 && std::fmod(r, 2.0) != 0.0)) r += 1.0;
    return (long long)r;
}

// CPython float_floor_div (also numpy npy_floor_divide for float64).
inline double floordiv(double vx, double wx) {
    if (wx == 0.0) throw std::domain_error("float floor division by zero");
    double mod = std::fmod(vx, wx);
    double div = (vx - mod) / wx;
    if (mod) {
        if ((wx < 0) != (mod < 0)) div -= 1.0;
    }
    double floordiv;
    if (div) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, vx / wx);
    }
    return floordiv;
}
// Python int // int.
inline long long ifloordiv(long long a, long long b) {
    if (b == 0) throw std::domain_error("integer division by zero");
    long long q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
}

// repr(float): shortest round-trip digits in CPython's 'r' layout.
inline std::string float_repr(double v) {
    if (std::isnan(v)) return "NaN"; // json.dumps spelling
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    if (v == 0.0) return std::signbit(v) ? "-0.0" : "0.0";
    char buf[40];
    const int n = crispembed_ryu_d2s_buffered_n(v, buf);
    std::string s(buf, size_t(n)); // e.g. "-1.2345E2"
    bool neg = false;
    if (!s.empty() && s[0] == '-') {
        neg = true;
        s.erase(0, 1);
    }
    const size_t e = s.find('E');
    std::string mant = s.substr(0, e);
    const int exp10 = std::stoi(s.substr(e + 1));
    std::string digits;
    for (char c : mant)
        if (c != '.') digits.push_back(c);
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int decpt = exp10 + 1; // digits d1 d2 ... with the point after decpt digits
    std::string out = neg ? "-" : "";
    if (decpt <= -4 || decpt > 16) {
        out += digits.substr(0, 1);
        if (digits.size() > 1) out += "." + digits.substr(1);
        const int x = decpt - 1;
        char eb[16];
        std::snprintf(eb, sizeof eb, "e%c%02d", x < 0 ? '-' : '+', x < 0 ? -x : x);
        out += eb;
    } else if (decpt <= 0) {
        out += "0." + std::string(size_t(-decpt), '0') + digits;
    } else if (size_t(decpt) >= digits.size()) {
        out += digits + std::string(size_t(decpt) - digits.size(), '0') + ".0";
    } else {
        out += digits.substr(0, size_t(decpt)) + "." + digits.substr(size_t(decpt));
    }
    return out;
}

// posixpath.basename / splitext / join.
inline std::string basename(const std::string & p) {
    const size_t i = p.rfind('/');
    return i == std::string::npos ? p : p.substr(i + 1);
}
inline std::pair<std::string, std::string> splitext(const std::string & p) {
    const long sep = p.rfind('/') == std::string::npos ? -1 : long(p.rfind('/'));
    const long dot = p.rfind('.') == std::string::npos ? -1 : long(p.rfind('.'));
    if (dot > sep) {
        for (long i = sep + 1; i < dot; ++i)
            if (p[size_t(i)] != '.') return { p.substr(0, size_t(dot)), p.substr(size_t(dot)) };
    }
    return { p, "" };
}
inline std::string join(const std::string & a, const std::string & b) {
    if (!b.empty() && b[0] == '/') return b;
    if (a.empty() || a.back() == '/') return a + b;
    return a + "/" + b;
}

// SHA-256 (FIPS 180-4) hex digest, used for request trace keys.
inline std::string sha256_hex(const uint8_t * data, size_t len) {
    static const uint32_t k[64] = { 0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
                                    0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
                                    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
                                    0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
                                    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
                                    0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
                                    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
                                    0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                                    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
                                    0xc67178f2 };
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    auto block = [&](const uint8_t * p) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) |
                   uint32_t(p[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            const uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    };
    size_t i = 0;
    for (; i + 64 <= len; i += 64) block(data + i);
    uint8_t tail[128] = { 0 };
    const size_t rem = len - i;
    if (rem) std::memcpy(tail, data + i, rem);
    tail[rem] = 0x80;
    const size_t total = rem + 1 + 8 <= 64 ? 64 : 128;
    const uint64_t bits = uint64_t(len) * 8;
    for (int b = 0; b < 8; ++b) tail[total - 1 - size_t(b)] = uint8_t(bits >> (8 * b));
    block(tail);
    if (total == 128) block(tail + 64);
    static const char * hex = "0123456789abcdef";
    std::string out;
    for (uint32_t v : h)
        for (int s = 28; s >= 0; s -= 4) out.push_back(hex[(v >> s) & 0xF]);
    return out;
}

} // namespace py_str
