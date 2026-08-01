#!/usr/bin/env bash
# CI guard: every toolchain-installing workflow step must PIN its version.
#
# Why this exists — a real 2-day outage (2026-07-14 → 07-16). Both WASM
# workflows used `mymindstorm/setup-emsdk` with no `version:`, so they resolved
# the `latest` alias at run time. `latest` moved 6.0.2 → 6.0.3, and 6.0.3's
# clang (23.0.0git) dies with an internal compiler crash (SIGSEGV in ParseAST)
# compiling src/layout_detect.cpp — a file that had not changed. main went red
# and stayed red until someone happened to push onto it. Nothing in the repo
# had changed; the toolchain drifted underneath us.
#
# An unpinned toolchain makes the build a function of wall-clock time, so a green
# run proves nothing about tomorrow and a red one implicates innocent code. Pin
# it, and drift becomes an explicit, reviewable commit instead of a silent break.
#
# A deliberate exception must say so, on the `uses:` line or the line above it:
#     uses: some/setup-thing@v1  # pin-exempt: <reason>
#
# Usage: bash tools/check_workflow_pins.sh [dir]   (default .github/workflows)
set -euo pipefail
cd "$(dirname "$0")/.."

DIR="${1:-.github/workflows}"

# Actions that install a toolchain whose version can drift.
TOOLS_RE='(setup-emsdk|setup-python|setup-node|setup-java|setup-go|setup-dotnet|setup-ninja|setup-cmake|rust-toolchain|install-nix)'
# Keys these actions use to pin a version.
PIN_RE='(version|python-version|node-version|java-version|go-version|dotnet-version|toolchain|cmake-version|ninja-version):'

fail=0
checked=0

# Emits: <file>:<lineno>:<uses-line> for each toolchain step.
while IFS= read -r hit; do
    [ -z "$hit" ] && continue
    file="${hit%%:*}"
    rest="${hit#*:}"
    lineno="${rest%%:*}"
    line="${rest#*:}"
    checked=$((checked + 1))

    # Exemption on the uses: line or the line immediately above.
    prev=$(sed -n "$((lineno > 1 ? lineno - 1 : 1))p" "$file")
    if printf '%s\n%s\n' "$prev" "$line" | grep -q 'pin-exempt:'; then
        continue
    fi

    # A pin must appear in this step's `with:` block — i.e. after the `uses:`
    # line and before the next step (a line starting with "- " at any indent).
    # NOTE: awk runs END even on `exit`, and an exit status set there WINS — so
    # the match is recorded in a flag and END alone decides the status.
    if awk -v start="$((lineno + 1))" -v pin="$PIN_RE" '
        NR < start { next }
        /^[[:space:]]*-[[:space:]]/ { exit }       # next step began: not found
        $0 ~ pin { found = 1; exit }               # pinned
        NR > start + 12 { exit }                   # give up: not found
        END { exit found ? 0 : 1 }
    ' "$file"; then
        continue
    fi

    echo "::error file=$file,line=$lineno::toolchain step is not version-pinned — 'latest' drifts and silently breaks the build (see tools/check_workflow_pins.sh). Add a 'version:' under 'with:', or '# pin-exempt: <reason>'."
    echo "    $file:$lineno: ${line#"${line%%[![:space:]]*}"}"
    fail=1
done < <(grep -rnE "uses:.*$TOOLS_RE" "$DIR" 2>/dev/null || true)

if [ "$fail" -ne 0 ]; then
    echo "FAIL: the above workflow steps install a toolchain at an unpinned 'latest'."
    exit 1
fi
echo "OK: all $checked toolchain step(s) in $DIR are version-pinned."
