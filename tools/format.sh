#!/usr/bin/env bash
# tools/format.sh — clang-format wrapper that ALWAYS uses v18.
#
# Ported from CrispASR. Pins clang-format-18 because wrapping decisions drift
# between v14/18/20/22 and cause "passed locally, failed elsewhere" churn. The
# style config lives in .clang-format at the repo root (tuned to CrispEmbed's
# existing hand-written style: 4-space indent, 120 cols, `T * p` / `T & r`
# pointer-middle, short if/loop bodies inline). NOTE: CrispEmbed's tree is not
# yet fully clang-format-clean; scope this to the files you actually changed
# rather than reformatting the whole tree in one go.
#
# Usage:
#   ./tools/format.sh                 # check-only, prints any violations
#   ./tools/format.sh --fix           # rewrite files in place
#   ./tools/format.sh src/foo.cpp     # operate on specific files only
#
# Prefer `git clang-format` for touching only changed *lines* of an otherwise
# unformatted file (avoids whole-file churn).
set -euo pipefail

# Locate clang-format-18. Refuse to fall back to anything else — the whole
# point is to fail loudly when v18 isn't available, not silently use v22.
find_clang_format_18() {
    local mac_path="/opt/homebrew/opt/llvm@18/bin/clang-format"
    [[ -x "$mac_path" ]] && { echo "$mac_path"; return 0; }
    if command -v clang-format-18 >/dev/null 2>&1; then
        command -v clang-format-18; return 0
    fi
    if command -v clang-format >/dev/null 2>&1; then
        local v
        v=$(clang-format --version 2>/dev/null | sed -nE 's/.*version ([0-9]+).*/\1/p' | head -1)
        if [[ "$v" == "18" ]]; then
            command -v clang-format; return 0
        fi
    fi
    return 1
}

if ! CLANG_FMT=$(find_clang_format_18); then
    cat <<'EOF' >&2
error: clang-format-18 not found.

v17/v19/v20/v22 produce different wrapping and cause formatting drift.

Install:
  macOS:  brew install llvm@18
          (binary lands at /opt/homebrew/opt/llvm@18/bin/clang-format)
  Ubuntu: sudo apt install clang-format-18
  Conda:  conda install -c conda-forge clang-format=18
EOF
    exit 2
fi

mode="check"
files=()
for arg in "$@"; do
    case "$arg" in
        --fix) mode="fix" ;;
        --check) mode="check" ;;
        --help|-h)
            sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        -*) echo "error: unknown flag: $arg" >&2; exit 2 ;;
        *) files+=("$arg") ;;
    esac
done

if [[ ${#files[@]} -eq 0 ]]; then
    # Default scope: project-owned C++ only. NEVER the vendored ggml/ submodule
    # or bundled single-header libraries.
    while IFS= read -r f; do files+=("$f"); done < <(
        find src examples tests \
             \( -name '*.cpp' -o -name '*.h' -o -name '*.c' -o -name '*.hpp' \) \
             ! -path '*/ggml/*' \
             ! -name 'httplib.h' \
             ! -name 'miniaudio.h' \
             ! -name 'json.hpp' \
             ! -name 'stb_image.h' \
             ! -name 'stb_image_write.h' \
             ! -name 'stb_vorbis.c'
    )
fi

fmt_files=()
skipped=()
for f in "${files[@]}"; do
    case "$f" in
        */ggml/*) skipped+=("$f") ;;                       # never touch the submodule
        *.c|*.cc|*.cpp|*.cxx|*.h|*.hh|*.hpp|*.hxx) fmt_files+=("$f") ;;
        *) skipped+=("$f") ;;
    esac
done
files=()
if [[ ${#fmt_files[@]} -gt 0 ]]; then
    files=("${fmt_files[@]}")
fi

if [[ ${#files[@]} -eq 0 ]]; then
    if [[ ${#skipped[@]} -gt 0 ]]; then
        echo "format.sh: skipped ${#skipped[@]} non-C/C++ or vendored file(s)"
    fi
    exit 0
fi

if [[ "$mode" == "fix" ]]; then
    "$CLANG_FMT" -i "${files[@]}"
    echo "format.sh: rewrote ${#files[@]} file(s) with $("$CLANG_FMT" --version | head -1)"
else
    out=$("$CLANG_FMT" --dry-run --Werror "${files[@]}" 2>&1 || true)
    n=$(echo "$out" | grep -c 'error:.*clang-format' || true)
    if [[ "$n" -ne 0 ]]; then
        echo "$out" | grep 'error:.*clang-format' | head -50
        echo ""
        echo "format.sh: $n violation(s). Fix with: ./tools/format.sh --fix"
        exit 1
    fi
    echo "format.sh: OK ($("$CLANG_FMT" --version | head -1), ${#files[@]} files)"
fi
