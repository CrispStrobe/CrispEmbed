// test_env_gate.cpp — hermetic guard for core/env_gate.h, the shared
// value-parsed boolean environment gate. No weights, no GGUF, no network.
//
// The defect under guard is the presence-based inversion the DS_* audit found
// and this sweep removed codebase-wide: `getenv("X") != nullptr` reports
// ENABLED for `X=0`, so an operator who spells a gate off turns it on. The
// `=0` case below is the one that fails against presence semantics; the rest
// pin the contract so a future "simplification" back to truthiness is caught.
#include "core/env_gate.h"
#include "core/clean_exit.h"

#include <cstdio>
#include <cstdlib>

static int g_checks = 0, g_failures = 0;

static void expect(bool got, bool want, const char * what) {
    g_checks++;
    if (got != want) {
        g_failures++;
        fprintf(stderr, "FAIL %s: got %s want %s\n", what, got ? "on" : "off", want ? "on" : "off");
    }
}

static void set_var(const char * value) {
    if (value)
        setenv("CRISPEMBED_TEST_ENV_GATE", value, 1);
    else
        unsetenv("CRISPEMBED_TEST_ENV_GATE");
}

static int crispembed_test_main() {
    const char * V = "CRISPEMBED_TEST_ENV_GATE";

    // The three spellings every gate in the codebase must honour.
    set_var(nullptr);
    expect(core_env::on(V), false, "absent => off");
    set_var("0");
    expect(core_env::on(V), false, "\"0\" => off (THE FIX: presence semantics said on)");
    set_var("1");
    expect(core_env::on(V), true, "\"1\" => on");

    // Empty string is a set-but-blank variable (`FOO= cmd`): off, because the
    // operator supplied no value to turn anything on with.
    set_var("");
    expect(core_env::on(V), false, "\"\" => off");

    // Any other non-empty value is on — gates are booleans, not enums, and a
    // typo must not silently disable an instrument the operator asked for.
    set_var("2");
    expect(core_env::on(V), true, "\"2\" => on");
    set_var("yes");
    expect(core_env::on(V), true, "\"yes\" => on");
    set_var("00");
    expect(core_env::on(V), true, "\"00\" => on (only exactly \"0\" is off)");
    set_var("0 ");
    expect(core_env::on(V), true, "\"0 \" => on (only exactly \"0\" is off)");

    // A null name is never on (defensive: callers pass string literals).
    expect(core_env::on(nullptr), false, "nullptr name => off");

    // A variable that was never in the environment at all.
    expect(core_env::on("CRISPEMBED_TEST_ENV_GATE_NEVER_SET"), false, "unset name => off");

    set_var(nullptr);

    printf("env-gate: %d checks, %d failure(s)\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}

// tools/check_test_clean_exit.sh: a one-shot binary must not run ggml's
// static GPU-device destructor at exit (it aborts on Metal / faults on CUDA).
int main() {
    core_util::clean_exit(crispembed_test_main());
}
