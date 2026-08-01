// src/core/hparam_keys.h — architecture-driven GGUF hyperparameter key resolution.
//
// Background (issue #33): the encoder loader hardcoded `bert.*` / `xlmr.*` key
// chains. A GGUF exported by llama.cpp/Ollama writes its hyperparameters under
// `<general.architecture>.<field>` — e.g. nomic-embed-text-v2-moe writes
// `nomic-bert-moe.embedding_length`. Teaching the loader one model at a time
// (append `nomic-bert-moe.*` to every chain) is an alias list that grows
// forever and is easy to leave half-done — the upstream PR for #33 added only
// the two expert-count keys and would still have loaded at the defaults.
//
// Instead: read `general.architecture` and derive the prefix. Every future
// community GGUF then resolves with no new code.
//
// The sharper half — SILENT DEFAULTS. When no candidate key matches, the loader
// falls back to a default (384-dim / 6-layer / 1e-12 eps). #33 got lucky and
// failed loudly on a missing tensor; for an architecture whose tensor names DO
// resolve, a wrong default silently produces a garbage embedding with exit code
// 0. `resolve()` reports whether it actually found the key so the caller can
// hard-fail under CRISPEMBED_STRICT_HPARAMS=1.
//
// Env gates (see PLAN.md "Ecosystem-compat + input-parsing hardening"):
//   CRISPEMBED_ARCH_HPARAMS=0    disable arch-derived lookup (default ON —
//                                purely additive; only fires when the legacy
//                                bert.*/xlmr.* keys are absent)
//   CRISPEMBED_STRICT_HPARAMS=1  hard-fail on a missing REQUIRED hparam instead
//                                of silently defaulting (default OFF — a
//                                hard-fail could break a model that legitimately
//                                relies on a default)
//
// Header-only so the pure key/resolution logic is unit-testable without a GGUF
// (tests/test_hparam_keys.cpp).
#pragma once

#include <cstdlib>
#include <string>
#include <vector>

namespace core_hparams {

// True unless CRISPEMBED_ARCH_HPARAMS=0. Read once.
inline bool arch_keys_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("CRISPEMBED_ARCH_HPARAMS");
        return !(e && e[0] == '0');
    }();
    return on;
}

// True when CRISPEMBED_STRICT_HPARAMS=1. Read once.
inline bool strict_hparams_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("CRISPEMBED_STRICT_HPARAMS");
        return e && e[0] == '1';
    }();
    return on;
}

// Build "<arch>.<field>", or "" when arch-derived lookup is disabled or the GGUF
// declares no architecture. An empty key is skipped by resolve(), so passing the
// result straight into a candidate list is safe.
inline std::string arch_key(const std::string & arch, const char * field, bool enabled) {
    if (!enabled || arch.empty() || field == nullptr || field[0] == '\0') return std::string();
    return arch + "." + field;
}

// Resolve `out` from the first candidate key that exists, in order. Empty keys
// are skipped (a disabled/absent arch key). Returns true if a key was found —
// the caller uses this to distinguish "read a real value" from "kept the
// default", which is what makes strict mode possible.
//
// `lookup(key, out)` must return true and assign `out` iff `key` is present.
template <typename T, typename Lookup>
inline bool resolve(const Lookup & lookup, const std::vector<std::string> & keys, T & out) {
    for (const std::string & k : keys) {
        if (k.empty()) continue;
        if (lookup(k, out)) return true;
    }
    return false;
}

} // namespace core_hparams
