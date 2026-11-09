// license_gate.h — the one decision rule for restricted-licence downloads.
//
// A registry model under a restricted licence (cc-by-nc-*, gemma, lfm1.0, ...)
// downloads only when the caller has accepted that licence. Acceptance is
// matched against the model's SPDX-style tag. Two callers share this rule:
//
//   * crispembed_resolve_model / the CLI's --accept-license: the accepted value
//     may be an exact tag, or the blanket "all" / "*".
//   * crispembed_resolve_model_accepting: the app passes the one tag the user
//     consented to for one model. Blanket values are refused there, so an app
//     cannot accept every licence through a per-model consent call.
//
// In both cases CRISPEMBED_ACCEPT_LICENSE is still consulted as before; that
// variable is the operator's own process-wide choice and keeps its meaning.
//
// Header-only so tests/test_license_consent.cpp can check the rule without a
// model, a network or the registry.
#pragma once

#include <string>

namespace core_license {

// "all" and "*" accept every restricted licence.
inline bool is_blanket(const std::string & accepted) {
    return accepted == "all" || accepted == "*";
}

// Does one accepted value cover `spdx`? Blanket values count only when
// `allow_blanket` is set.
inline bool covers(const std::string & accepted, const char * spdx, bool allow_blanket) {
    if (accepted.empty()) return false;
    if (is_blanket(accepted)) return allow_blanket;
    return spdx && accepted == spdx;
}

// Gate decision: the explicit value first, then the environment value
// (nullptr when unset). The environment always allows blanket values, so its
// behaviour does not depend on which API the caller used.
inline bool accepted(const char * spdx, const std::string & accepted_arg, bool allow_blanket_arg,
                     const char * env_value) {
    if (covers(accepted_arg, spdx, allow_blanket_arg)) return true;
    if (env_value && covers(env_value, spdx, true)) return true;
    return false;
}

} // namespace core_license
