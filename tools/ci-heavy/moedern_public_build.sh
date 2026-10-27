#!/usr/bin/env bash
# Model-free compile/ABI controls for the MoEdern public draft (heavy-cpu.yml).
# Separate shared library from draft sources only; no model, no root wiring.
set -euo pipefail
git fetch -q --unshallow origin 2>/dev/null || git fetch -q origin 1a053116908224cf1c9358d7d24c85ba36e82eff
R="$HEAVY_OUT"; B="$HEAVY_SCRATCH/mp"; mkdir -p "$R"
python tools/moedern-public/public_contract.py | tee "$R/source.json"
python tests/test_moedern_public_text.py -v 2>&1 | tee "$R/stdlib.txt"
cmake -S tests/moedern-public -B "$B" -DCMAKE_BUILD_TYPE=Release > "$R/configure.txt"
cmake --build "$B" --parallel 2 > "$R/build.txt" 2>&1 || { tail -50 "$R/build.txt"; exit 1; }
"$B/moedern-public-controls" | tee "$R/controls.txt"
nm -D --defined-only "$B/libmoedern-public.so" | grep ' T crispembed_' | tee "$R/exports.txt"
test "$(grep -c crispembed_moedern_ "$R/exports.txt")" = 5
if nm -C "$B/libmoedern-public.so" | grep -E 'composed_test|observe'; then echo "observer leaked"; exit 1; fi
cp "$B/compile_commands.json" "$R/"
python - "$R/compile_commands.json" <<'P'
import json,sys
rows=[r for r in json.load(open(sys.argv[1])) if 'moedern_text' in r['file']]
assert len(rows)==4 and all('-ffp-contract=off' in r['command'] and 'TESTING' not in r['command'] for r in rows), rows
print('flags ok', len(rows))
P
# Rust: runner-local module registration for type-checking only.
echo 'pub mod moedern_text;' >> crispembed/src/lib.rs
cargo check -p crispembed 2>&1 | tee "$R/rust.txt"
git checkout crispembed/src/lib.rs
# Dart: exact SDK and ffi pin.
curl -sSLo "$HEAVY_SCRATCH/dart.zip" https://storage.googleapis.com/dart-archive/channels/stable/release/3.13.5/sdk/dartsdk-linux-x64-release.zip
unzip -q "$HEAVY_SCRATCH/dart.zip" -d "$HEAVY_SCRATCH"
D="$HEAVY_SCRATCH/dt"; mkdir -p "$D/lib"; cp flutter/crispembed/lib/src/moedern_text.dart "$D/lib/"
printf 'name: moedern_audit\nenvironment:\n  sdk: ">=3.0.0 <4.0.0"\ndependencies:\n  ffi: 2.1.4\n' > "$D/pubspec.yaml"
( cd "$D" && "$HEAVY_SCRATCH/dart-sdk/bin/dart" pub get && "$HEAVY_SCRATCH/dart-sdk/bin/dart" analyze --fatal-infos ) 2>&1 | tee "$R/dart.txt"
echo "### MoEdern public draft: model-free compile/ABI controls passed" > "$R/summary.md"
