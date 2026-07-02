#!/usr/bin/env bash
# CI guard: every one-shot binary that defines `int main` (the test-*-diff/etc.
# harnesses under tests/, plus the CLI) must route its exit through
# core_util::clean_exit. Otherwise the ggml v0.10.0 GPU-device static-destructor
# teardown crashes it on exit — Metal aborts on the residency-set assert, CUDA
# hits a use-after-free (SIGSEGV/SIGABRT) — turning a passing run (correct output
# already printed) into a false failure with a corrupted exit code.
#
# See src/core/clean_exit.h and LEARNINGS.md "ggml v0.10.0 GPU-teardown
# regressions". To fix an offender: rename its `int main(...)` body to
# `static int crispembed_test_main(...)`, add `#include "core/clean_exit.h"`, and
# add a thin `int main(...) { core_util::clean_exit(crispembed_test_main(...)); }`.
#
# Long-lived hosts (examples/server, bindings) intentionally do NOT use this — they
# free via crispembed_free on shutdown — so they are excluded below.
set -euo pipefail
cd "$(dirname "$0")/.."

has_main='^[[:space:]]*int[[:space:]]+main[[:space:]]*\('
fail=0
checked=0

check_file() {
    local f="$1"
    grep -qE "$has_main" "$f" || return 0
    checked=$((checked + 1))
    if ! grep -q 'clean_exit' "$f"; then
        echo "::error file=$f::main() does not route through core_util::clean_exit (see src/core/clean_exit.h)"
        fail=1
    fi
}

for f in tests/*.cpp; do check_file "$f"; done
check_file examples/cli/main.cpp

if [ "$fail" -ne 0 ]; then
    echo "FAIL: the above one-shot binaries bypass core_util::clean_exit and will crash on GPU-device teardown at exit."
    exit 1
fi
echo "OK: all $checked one-shot main()s route through core_util::clean_exit."
