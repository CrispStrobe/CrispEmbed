#!/usr/bin/env bash
# Usage: scripts/bump-version.sh <version>   e.g.  scripts/bump-version.sh 0.13.0
#
# Updates VERSION, propagates to Cargo.toml / pyproject.toml / pubspec.yaml
# via sync-version.py, commits, and creates an annotated tag — all in one step.
# Mirrors CrispASR's scripts/bump-version.sh.
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>  (e.g. $0 0.13.0)" >&2
    exit 1
fi

VERSION="$1"
REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO_ROOT"

echo "$VERSION" > VERSION
python scripts/sync-version.py

# Stage only the files sync-version.py touches (plus VERSION itself).
git add VERSION
for f in \
    bindings/javascript/package.json \
    python/pyproject.toml \
    crispembed/Cargo.toml \
    crispembed-sys/Cargo.toml \
    flutter/crispembed/pubspec.yaml; do
    [ -f "$f" ] && git add "$f" || true
done

git commit -m "release: bump VERSION to $VERSION"
git tag -a "v$VERSION" -m "Release v$VERSION"

echo ""
echo "Created commit + annotated tag v$VERSION."
echo "Push with:  git push && git push origin v$VERSION"
