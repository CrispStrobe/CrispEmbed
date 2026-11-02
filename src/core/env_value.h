// env_value.h — value-carrying environment variables (paths, ints, strings).
//
// CrispEmbed-only companion to env_gate.h (which must stay code-identical to
// CrispASR's copy, so it is not extended here). `X=` (exported but blank) must
// not switch on a path-driven debug dump or override an int default with
// atoi("") == 0, so EMPTY is folded into unset:
//
//     if (const char * ref = core_env::value("CRISPEMBED_FOO_REF")) { ... }
//
// Guarded hermetically by tests/test_env_gate.cpp.
#pragma once

#include <cstdlib>

namespace core_env {

inline const char * value(const char * name) {
    if (!name) return nullptr;
    const char * e = std::getenv(name);
    return (e && *e) ? e : nullptr;
}

} // namespace core_env
