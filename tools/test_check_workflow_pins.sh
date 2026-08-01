#!/usr/bin/env bash
# Self-test for tools/check_workflow_pins.sh (A4).
#
# A guard that cannot fail is decoration. This drives the checker over fixture
# workflows and asserts BOTH arms: it accepts pinned steps and rejects the exact
# shape that caused the 2026-07-14 emsdk outage (an unpinned setup-emsdk).
#
# No network, no build. Usage: bash tools/test_check_workflow_pins.sh
set -uo pipefail
cd "$(dirname "$0")/.."

CHECKER="tools/check_workflow_pins.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fails=0
ok()   { echo "  [PASS] $1"; }
bad()  { echo "  [FAIL] $1"; fails=$((fails + 1)); }

expect() { # expect <want_rc> <name> <dir>
    local want="$1" name="$2" dir="$3"
    bash "$CHECKER" "$dir" >/dev/null 2>&1
    local got=$?
    if [ "$got" -eq "$want" ]; then ok "$name"; else bad "$name (rc=$got, want $want)"; fi
}

echo "test_check_workflow_pins"

# --- Arm A: the real outage shape — unpinned setup-emsdk MUST be rejected ---
mkdir -p "$TMP/bad"
cat > "$TMP/bad/w.yml" <<'YML'
jobs:
  build:
    steps:
      - uses: actions/checkout@v4
      - name: Setup emsdk
        uses: mymindstorm/setup-emsdk@v14
      - name: Build
        run: ./build-wasm.sh
YML
expect 1 "rejects unpinned setup-emsdk (the 07-14 outage shape)" "$TMP/bad"

# --- Arm B: the fix — a pinned version is accepted ---
mkdir -p "$TMP/good"
cat > "$TMP/good/w.yml" <<'YML'
jobs:
  build:
    steps:
      - uses: actions/checkout@v4
      - name: Setup emsdk
        uses: mymindstorm/setup-emsdk@v14
        with:
          version: '6.0.2'
      - name: Build
        run: ./build-wasm.sh
YML
expect 0 "accepts pinned setup-emsdk" "$TMP/good"

# --- setup-python pinned via its own key name ---
mkdir -p "$TMP/py"
cat > "$TMP/py/w.yml" <<'YML'
jobs:
  t:
    steps:
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
YML
expect 0 "accepts setup-python pinned via python-version" "$TMP/py"

# --- unpinned setup-python is rejected ---
mkdir -p "$TMP/pybad"
cat > "$TMP/pybad/w.yml" <<'YML'
jobs:
  t:
    steps:
      - uses: actions/setup-python@v5
YML
expect 1 "rejects unpinned setup-python" "$TMP/pybad"

# --- explicit exemption is honoured ---
mkdir -p "$TMP/exempt"
cat > "$TMP/exempt/w.yml" <<'YML'
jobs:
  t:
    steps:
      - uses: actions/setup-node@v4  # pin-exempt: docs-only job, version irrelevant
YML
expect 0 "honours '# pin-exempt:' escape hatch" "$TMP/exempt"

# --- a pin belonging to the NEXT step must not count for this one ---
mkdir -p "$TMP/leak"
cat > "$TMP/leak/w.yml" <<'YML'
jobs:
  t:
    steps:
      - uses: actions/setup-node@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
YML
expect 1 "does not let the next step's pin satisfy an unpinned one" "$TMP/leak"

# --- non-toolchain actions are ignored (no false positives) ---
mkdir -p "$TMP/other"
cat > "$TMP/other/w.yml" <<'YML'
jobs:
  t:
    steps:
      - uses: actions/checkout@v6
      - uses: actions/upload-artifact@v7
YML
expect 0 "ignores non-toolchain actions" "$TMP/other"

# --- the repo's own workflows must be clean ---
expect 0 "the repo's .github/workflows are all pinned" ".github/workflows"

if [ "$fails" -ne 0 ]; then echo "FAILED ($fails failure(s))"; exit 1; fi
echo "OK (0 failures)"
