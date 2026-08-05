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
// set to a non-empty value that is not exactly "0". Unset, empty, and "0" are
// all OFF.
//
//     const bool bench = core_env::on("CRISPEMBED_FOO_BENCH");
//
// Guarded hermetically by tests/test_env_gate.cpp.
#pragma once

#include <cstdlib>
#include <cstring>

namespace core_env {

inline bool on(const char * name) {
    if (!name) return false;
    const char * e = std::getenv(name);
    return e && *e && std::strcmp(e, "0") != 0;
}

} // namespace core_env
