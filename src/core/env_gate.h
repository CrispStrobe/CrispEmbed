// env_gate.h — one value-parsed boolean environment gate for the whole
// codebase.
//
// The defect this exists to prevent: `getenv("X") != nullptr` (or bare
// `if (getenv("X"))`) treats `X=0` as ENABLED. Every operator who writes
// `CRISPEMBED_FOO_BENCH=0` to turn an instrument OFF gets it turned ON, and
// for a diagnostic gate the only symptom is unexplained output — which is
// exactly how the DS_* inversion survived until the G2b/G6 audits.
//
// Semantics (identical to deepseek_ocr2.cpp's `ds_env_on`, which established
// them, and to core_initbench::enabled): the gate is ON when the variable is
// set to a non-empty value that is not an off-word. Unset, empty, and the
// exact (ASCII case-insensitive) words "0", "false", "off", "no" are all OFF.
//
//     const bool bench = core_env::on("CRISPEMBED_FOO_BENCH");
//
// Guarded hermetically by tests/test_env_gate.cpp.
#pragma once

#include <cstdlib>
#include <cstring>

namespace core_env {

// Exact, ASCII-case-insensitive spellings of "no" (MSVC-safe: no strcasecmp).
// "0", "false", "off", "no" -- nothing else ("00", "0 ", "nope" stay values).
inline bool is_off_word(const char * e) {
    static const char * const words[] = { "0", "false", "off", "no" };
    for (const char * w : words) {
        const char * a = e;
        const char * b = w;
        while (*a && *b) {
            char c = *a;
            if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
            if (c != *b) break;
            ++a;
            ++b;
        }
        if (!*a && !*b) return true;
    }
    return false;
}

inline bool on(const char * name) {
    if (!name) return false;
    const char * e = std::getenv(name);
    return e && *e && !is_off_word(e);
}

// The other half of a tri-state gate: EXPLICITLY set to "0". `on()` cannot
// express it, because `on()` folds unset and "0" together — fine for an
// instrument that defaults off, wrong for a knob that defaults ON and needs an
// opt-OUT. Use the pair when the default is on:
//
//     if (core_env::explicitly_off("CRISPEMBED_FOO_CLEANUP")) { /* legacy */ }
//
// Unset and empty are NOT off (they mean "default"); only an off-word
// ("0", "false", "off", "no", case-insensitive) is.
inline bool explicitly_off(const char * name) {
    if (!name) return false;
    const char * e = std::getenv(name);
    return e && *e && is_off_word(e);
}

// Value-carrying variables (paths, ints, strings): the raw value, with EMPTY
// folded into unset. `X=` (an exported-but-blank variable) must not switch on
// a path-driven debug dump or override an int default with atoi("") == 0.
//
//     if (const char * ref = core_env::value("CRISPEMBED_FOO_REF")) { ... }
inline const char * value(const char * name) {
    if (!name) return nullptr;
    const char * e = std::getenv(name);
    return (e && *e) ? e : nullptr;
}

} // namespace core_env
