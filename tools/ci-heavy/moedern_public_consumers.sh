#!/usr/bin/env bash
# heavy-cpu entry: exact Dart SDK, then the MoEdern public consumer worker.
set -euo pipefail
git fetch -q --unshallow origin 2>/dev/null || true
curl -sSLo "$HEAVY_SCRATCH/dart.zip" https://storage.googleapis.com/dart-archive/channels/stable/release/3.13.5/sdk/dartsdk-linux-x64-release.zip
unzip -q "$HEAVY_SCRATCH/dart.zip" -d "$HEAVY_SCRATCH"
export DART="$HEAVY_SCRATCH/dart-sdk/bin/dart"
python tests/test_moedern_public_root.py -v
python -I tools/ci-heavy/moedern_public_consumers.py
