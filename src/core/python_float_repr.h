// python_float_repr.h — CPython repr(float) for finite doubles (shortest
// round-trip digits via Ryu, then CPython's fixed/scientific layout rule),
// so native JSON matches json.dumps of the same float32-derived values.
#pragma once

#include "core/shortest_double.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace core_pyrepr {

// Python repr(float(x)) for a finite double (CPython 'r' formatting).
inline void append_python_float(std::string & out, double value) {
    if (!std::isfinite(value)) throw std::runtime_error("non-finite layout value (JSON allow_nan=False)");
    char buf[64];
    const int n = crispembed_ryu_d2s_buffered_n(value, buf);
    std::string s(buf, (size_t)n);
    bool neg = false;
    if (!s.empty() && s[0] == '-') {
        neg = true;
        s.erase(0, 1);
    }
    const size_t e = s.find('E');
    const std::string mant = s.substr(0, e);
    const int exp10 = std::atoi(s.c_str() + e + 1);
    std::string digits;
    for (char c : mant)
        if (c != '.') digits += c;
    if (neg) out += '-';
    if (exp10 >= -4 && exp10 < 16) { // repr: fixed unless exponent < -4 or >= 16
        if (exp10 < 0) {
            out += "0.";
            out.append((size_t)(-exp10 - 1), '0');
            out += digits;
        } else if ((int)digits.size() <= exp10 + 1) {
            out += digits;
            out.append((size_t)(exp10 + 1 - (int)digits.size()), '0');
            out += ".0";
        } else {
            out += digits.substr(0, (size_t)exp10 + 1);
            out += '.';
            out += digits.substr((size_t)exp10 + 1);
        }
    } else {
        out += digits[0];
        if (digits.size() > 1) {
            out += '.';
            out += digits.substr(1);
        }
        char ex[16];
        std::snprintf(ex, sizeof(ex), "e%c%02d", exp10 < 0 ? '-' : '+', std::abs(exp10));
        out += ex;
    }
}


inline std::string python_float(double value) {
    std::string s;
    append_python_float(s, value);
    return s;
}

} // namespace core_pyrepr
