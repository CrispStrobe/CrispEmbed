#!/bin/sh
# Flatten a directory of CrispEmbed/ggml macOS dylibs to ONE real file per
# library, named and identified by its UNVERSIONED name (libX.dylib), with every
# inter-library @rpath reference rewritten to match.
#
#   flatten_dylibs.sh [--check] DIR [EXTRA_MACHO ...]
#
# Why: CMake's VERSION/SOVERSION give each dylib the install name
# @rpath/libX.0.dylib and ship it as libX.<M>.<m>.<p>.dylib plus libX.0.dylib
# and libX.dylib SYMLINKS. Symlinks do not survive the paths these files take
# into an app: CocoaPods dereferences vendored_libraries when it embeds them
# (Flutter macOS), upload-artifact copies them as separate files, and earlier
# packaging steps kept whichever names they happened to glob. The embedded set
# then never matched both the name the Dart side dlopen()s (libcrispembed.dylib)
# and the @rpath/libX.0.dylib names the load commands ask for, and the app
# aborted with "Library not loaded". With unversioned install names there is
# exactly one name per library, so whatever copies the files keeps them loadable.
#
# EXTRA_MACHO: executables (or other images outside DIR) whose @rpath references
# to the flattened libraries must be rewritten too, e.g. the CLI in a tarball.
#
# --check makes no changes: it fails unless every dylib in DIR is a regular file
# whose install name is @rpath/<its own file name>, and every @rpath reference
# of every image (DIR's dylibs and EXTRA_MACHO) names a regular file in DIR.
# Flattening runs the same check at the end. Idempotent.
set -eu

check_only=0
if [ "${1:-}" = "--check" ]; then check_only=1; shift; fi
if [ $# -lt 1 ] || [ ! -d "$1" ]; then
  echo "usage: $0 [--check] DIR [EXTRA_MACHO ...]" >&2
  exit 2
fi
dir=$1; shift

# libggml-base.0.10.2.dylib / libggml-base.0.dylib / libggml-base.dylib -> libggml-base
stem_of() { basename "$1" .dylib | sed -E 's/(\.[0-9]+)+$//'; }

rpath_refs() {  # every @rpath/... load command of a Mach-O image (not its id)
  id=$(otool -D "$1" 2>/dev/null | sed -n 2p)
  otool -L "$1" | sed 1d | awk '{print $1}' | grep '^@rpath/' | grep -vxF "${id:-<none>}" || true
}

if [ "$check_only" = 0 ]; then
  # 1. Alias symlinks go: they are either dereferenced into duplicates or
  #    dropped by whatever copies them next, never preserved.
  for f in "$dir"/*.dylib; do
    [ -L "$f" ] && rm -f "$f"
  done
  # 2. One real file per library under its unversioned name. Several real
  #    copies of the same library (an upload-artifact or `cp -L` of a symlink
  #    chain) collapse to one.
  for f in "$dir"/*.dylib; do
    [ -f "$f" ] || continue
    target="$dir/$(stem_of "$f").dylib"
    [ "$f" = "$target" ] && continue
    if [ -f "$target" ]; then rm -f "$f"; else mv -f "$f" "$target"; fi
  done
  # 3. Install names and cross-references -> @rpath/libX.dylib, then re-sign
  #    (install_name_tool invalidates the signature; arm64 refuses unsigned
  #    code). Xcode signs again with the app identity when it embeds them.
  patch_one() {  # $1 = image; its own positional list collects the edits
    f=$1; set --
    case "$f" in
      "$dir"/*.dylib)
        id=$(otool -D "$f" | sed -n 2p)
        [ "$id" = "@rpath/$(basename "$f")" ] || set -- -id "@rpath/$(basename "$f")" ;;
    esac
    for ref in $(rpath_refs "$f"); do
      want="$(stem_of "$ref").dylib"
      [ -f "$dir/$want" ] || continue
      [ "$ref" = "@rpath/$want" ] || set -- "$@" -change "$ref" "@rpath/$want"
    done
    [ $# -gt 0 ] || return 0
    install_name_tool "$@" "$f"
    if command -v codesign >/dev/null 2>&1; then
      codesign --force --sign - "$f"
    fi
  }
  for f in "$dir"/*.dylib "$@"; do
    if [ -f "$f" ]; then patch_one "$f"; fi
  done
fi

status=0
found=0
for f in "$dir"/*.dylib; do
  [ -e "$f" ] || [ -L "$f" ] || continue
  found=1
  if [ -L "$f" ]; then
    echo "FAIL: $f is a symlink; it will not survive embedding" >&2; status=1; continue
  fi
  id=$(otool -D "$f" | sed -n 2p)
  if [ "$id" != "@rpath/$(basename "$f")" ]; then
    echo "FAIL: $f has install name '$id', expected @rpath/$(basename "$f")" >&2; status=1
  fi
done
if [ "$found" = 0 ]; then echo "FAIL: no dylibs in $dir" >&2; exit 1; fi
if [ ! -f "$dir/libcrispembed.dylib" ]; then
  echo "FAIL: $dir/libcrispembed.dylib is missing (the Dart side dlopen()s that name)" >&2; status=1
fi
for f in "$dir"/*.dylib "$@"; do
  [ -f "$f" ] || continue
  for ref in $(rpath_refs "$f"); do
    if [ ! -f "$dir/${ref#@rpath/}" ] || [ -L "$dir/${ref#@rpath/}" ]; then
      echo "FAIL: $f references $ref, which is not a regular file in $dir" >&2; status=1
    fi
  done
done
if [ "$status" = 0 ]; then
  echo "flatten_dylibs: OK — every @rpath reference resolves within $dir:"
  for f in "$dir"/*.dylib; do echo "  $(basename "$f")"; done
fi
exit $status
