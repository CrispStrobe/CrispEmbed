#!/usr/bin/env bash
# Vendor the C/C++ sources into the crispembed-sys crate.
#
# Why this exists: `cargo package` only archives files under the crate root, so
# build.rs's `manifest_dir.parent()` — the repository root it runs cmake on —
# does not exist in the published tarball. Without a vendored copy, `cargo
# publish` uploads a crate that cannot build for anyone, with no warning.
# Run this before `cargo package`/`cargo publish` (CI does it automatically).
#
# What is vendored: CMakeLists.txt, VERSION (CMakeLists file(READ)s it), cmake/,
# src/, examples/ (the library itself compiles examples/cli/model_mgr.cpp), the
# ggml submodule, and the test *sources* — but NOT the 33 MB of test fixtures,
# which would blow the crates.io size limit.
#
# The test sources have to be here even though nothing links them: the top-level
# CMakeLists declares add_executable(test-... tests/foo.cpp) unconditionally, and
# cmake fails at CONFIGURE time on a missing source file. build.rs then builds
# only `--target crispembed-shared`, so no consumer ever compiles them. Guarding
# those targets with a CMake option instead was tried and reverted: the block
# nests inside `if(NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")`, so a hand-placed
# if/endif silently paired with the wrong endif and left the tests ungated.
#
# Only git-TRACKED files are copied, so stray build dirs, .o files and model
# blobs in a dirty working tree can never leak into a published crate.
#
# The vendored tree is generated, not authored — it is gitignored, and build.rs
# prefers the repository copy so an in-tree build never uses a stale snapshot.
set -euo pipefail

repo_root=$(cd "$(dirname "$0")/.." && pwd)
vendor="$repo_root/crispembed-sys/vendor"

if [ ! -f "$repo_root/ggml/CMakeLists.txt" ]; then
    echo "error: ggml submodule not checked out — run: git submodule update --init ggml" >&2
    exit 1
fi

rm -rf "$vendor"
mkdir -p "$vendor"

copy_tracked() {   # $1 = git dir, $2 = destination prefix, rest = pathspecs
    local gitdir="$1" prefix="$2"; shift 2
    git -C "$gitdir" ls-files -z "$@" | while IFS= read -r -d '' f; do
        mkdir -p "$vendor/$prefix$(dirname "$f")"
        cp "$gitdir/$f" "$vendor/$prefix$f"
    done
}

copy_tracked "$repo_root" ""      CMakeLists.txt VERSION cmake src examples
# Sources only from tests/ and tools/ — their fixtures/models are the 33 MB we
# must not ship. `:(glob)` so nested directories are matched too.
for ext in cpp cc c h hpp; do
    copy_tracked "$repo_root" "" ":(glob)tests/**/*.$ext" ":(glob)tools/**/*.$ext"
done
copy_tracked "$repo_root/ggml" "ggml/"

n=$(find "$vendor" -type f | wc -l | tr -d ' ')
echo "vendored $n tracked files -> ${vendor#"$repo_root"/}"

# Guard against a partial copy shipping a crate that cannot build.
for required in CMakeLists.txt VERSION ggml/CMakeLists.txt; do
    if [ ! -f "$vendor/$required" ]; then
        echo "error: vendored tree is incomplete — missing $required" >&2
        exit 1
    fi
done
