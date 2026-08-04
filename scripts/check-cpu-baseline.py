#!/usr/bin/env python3
"""Assert that a configured build tree targets a portable CPU baseline.

Run this right after `cmake -S . -B build ...` in any workflow that produces a
*redistributable* artifact (release archives, Python wheels). It fails the job
if the build was configured to target the builder's own CPU instead of a fixed
baseline.

Why this exists
---------------
Issue #41: the v0.16.1 `crispembed-windows-x86_64.zip` (cpu) crashed with
"Illegal instruction" on an i9-14900KF (AVX2, no AVX-512), immediately after
tokenizer load. The same model ran fine on the previous release and on the
cuda/vulkan artifacts' CPU fallback.

Cause: the cpu leg left `GGML_NATIVE` at its default (ON). On MSVC that makes
ggml include `FindSIMD.cmake`, which *runs* probe programs on the build machine
via `check_c_source_runs` and sets AVX-512 -> `/arch:AVX512` whenever the runner
supports it. GitHub's `windows-latest` pool mixes AVX-512-capable Intel hosts
with AVX2-only AMD hosts, so whether a release was usable on consumer hardware
came down to which runner picked up the job. The cuda leg already pinned
`-DGGML_NATIVE=OFF`, which is exactly why its CPU fallback worked. The ARM legs
have the same hazard via `-mcpu=native` probes for dotprod/i8mm/sve/sme.

The failure is silent at build time and only reproduces on hardware the CI
fleet does not have, so it needs a configure-time gate rather than a test.

Two independent checks
----------------------
1. **Cache options** — `GGML_NATIVE` / `CRISPEMBED_NATIVE` must be OFF, and on
   x86 the ISA options must describe the intended baseline. This is the
   check with an actionable message.
2. **Generated compiler flags** — scan what CMake actually wrote into
   `build.ninja` / `*.vcxproj` / `flags.make` for banned tokens. Necessary
   because `FindSIMD.cmake` sets `GGML_AVX512` as a *normal* variable that
   shadows the cache entry: with NATIVE on, the cache can read `OFF` while the
   compile line says `/arch:AVX512`. This check sees the real flags.

Usage
-----
    python scripts/check-cpu-baseline.py build
    python scripts/check-cpu-baseline.py build --arch x86_64
"""

from __future__ import annotations

import argparse
import platform
import re
import sys
from pathlib import Path

# Must hold for every redistributable build, on any architecture. This is the
# root cause of #41 and the one knob a caller has to set.
REQUIRED_ANY_ARCH = {
    "GGML_NATIVE": "OFF",
    "CRISPEMBED_NATIVE": "OFF",
}

# x86_64 only, and only meaningful once NATIVE is OFF (see module docstring).
# AVX2/FMA/F16C is the shipped floor: Haswell (2013) and Excavator (2015)
# onward. AVX-512 is absent from every current Intel consumer part (12th gen
# onward) and from all pre-Zen4 AMD, so it can never be the baseline.
REQUIRED_X86 = {
    "GGML_AVX2": "ON",
    "GGML_AVX512": "OFF",
    "GGML_AVX512_VBMI": "OFF",
    "GGML_AVX512_VNNI": "OFF",
    "GGML_AVX512_BF16": "OFF",
    "GGML_AVX_VNNI": "OFF",
    "GGML_AMX_TILE": "OFF",
    "GGML_AMX_INT8": "OFF",
    "GGML_AMX_BF16": "OFF",
}

X86_ALIASES = {"x86_64", "amd64", "x64", "win64", "i386", "i686"}

# Tokens that must never appear on a compile line in a redistributable build.
# `native` is banned on every architecture; the rest are x86 extensions above
# the baseline. AVX-VNNI is included because it is Alder-Lake-and-newer only.
BANNED_FLAGS_ANY_ARCH = [
    "-march=native",
    "-mcpu=native",
    "-mtune=native",
    "/arch:native",
]
BANNED_FLAGS_X86 = [
    "/arch:AVX512",
    "/arch:AVX10",
    "AdvancedVectorExtensions512",  # MSVC generator's XML spelling of /arch:AVX512
    "-mavx512",
    "-mamx-",
    "-mavxvnni",
]

# Where CMake records the actual compile lines, per generator.
FLAG_FILE_GLOBS = ("build.ninja", "**/*.vcxproj", "**/flags.make")


def read_cache(cache_path: Path) -> dict[str, str]:
    """Parse CMakeCache.txt into {NAME: VALUE}, dropping the :TYPE suffix."""
    entries: dict[str, str] = {}
    for raw in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key, _, value = line.partition("=")
        entries[key.split(":", 1)[0]] = value
    return entries


def detect_arch(build_dir: Path, entries: dict[str, str]) -> str:
    """Best-effort target architecture for a configured build tree.

    CMAKE_SYSTEM_PROCESSOR is a normal variable, not a cache entry, so it is
    absent from CMakeCache.txt on most generators. CMake does write it to
    CMakeFiles/<ver>/CMakeSystem.cmake, which is authoritative and stays
    correct for cross-compiles. Fall back to the host only as a last resort.
    """
    cached = entries.get("CMAKE_SYSTEM_PROCESSOR")
    if cached:
        return cached
    for sysfile in sorted(build_dir.glob("CMakeFiles/*/CMakeSystem.cmake")):
        text = sysfile.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'set\s*\(\s*CMAKE_SYSTEM_PROCESSOR\s+"?([^")\s]+)"?', text)
        if m:
            return m.group(1)
    return platform.machine()


def is_x86(arch: str) -> bool:
    return arch.lower() in X86_ALIASES


def scan_generated_flags(build_dir: Path, banned: list[str]) -> list[str]:
    """Return "<file>: <token>" for every banned token found in a build file."""
    hits: list[str] = []
    seen: set[tuple[str, str]] = set()
    for pattern in FLAG_FILE_GLOBS:
        for path in build_dir.glob(pattern):
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for token in banned:
                if token in text and (path.name, token) not in seen:
                    seen.add((path.name, token))
                    hits.append(f"{path.relative_to(build_dir)}: {token}")
    return sorted(hits)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("build_dir", help="configured CMake build directory")
    ap.add_argument(
        "--arch",
        default=None,
        help="target architecture (default: CMAKE_SYSTEM_PROCESSOR from the build tree)",
    )
    args = ap.parse_args()

    build_dir = Path(args.build_dir)
    cache_path = build_dir / "CMakeCache.txt"
    if not cache_path.is_file():
        print(f"error: {cache_path} not found — configure the build first", file=sys.stderr)
        return 2

    entries = read_cache(cache_path)
    arch = args.arch or detect_arch(build_dir, entries)
    x86 = is_x86(arch)

    print(f"CPU baseline check: {cache_path} (arch={arch or 'unknown'})")

    # ── 1. cache options ────────────────────────────────────────────────
    # A missing entry means the option was never declared for this build
    # (GGML_FMA/F16C are MSVC-gated, for instance). Absent is not a violation;
    # only present-and-wrong is.
    def check(expected: dict[str, str]) -> list[str]:
        bad = []
        for name, want in sorted(expected.items()):
            got = entries.get(name)
            print(f"  {name:<20} = {got if got is not None else '<not declared>'}")
            if got is not None and got.upper() != want:
                bad.append(f"  {name} = {got}  (expected {want})")
        return bad

    failures = check(REQUIRED_ANY_ARCH)
    if failures:
        print(
            "\nERROR: this build is NOT redistributable — it targets the build "
            "machine's CPU.\nOffending cache entries:\n" + "\n".join(failures) + "\n\n"
            "Pass -DGGML_NATIVE=OFF at configure time; CRISPEMBED_NATIVE follows it.\n"
            "See issue #41.",
            file=sys.stderr,
        )
        return 1

    # Only meaningful with NATIVE off: with it on, FindSIMD's normal variables
    # shadow these cache entries and the values below are not the truth.
    if x86:
        failures = check(REQUIRED_X86)

    # ── 2. generated compile lines (authoritative) ──────────────────────
    banned = list(BANNED_FLAGS_ANY_ARCH) + (BANNED_FLAGS_X86 if x86 else [])
    flag_hits = scan_generated_flags(build_dir, banned)

    if failures or flag_hits:
        msg = ["\nERROR: this build is NOT redistributable."]
        if failures:
            msg.append("Offending cache entries:\n" + "\n".join(failures))
        if flag_hits:
            msg.append(
                "Above-baseline instruction sets on the generated compile lines:\n"
                + "\n".join(f"  {h}" for h in flag_hits)
            )
        msg.append("See issue #41 and the CPU baseline notes in release.yml.")
        print("\n\n".join(msg), file=sys.stderr)
        return 1

    print(f"\nOK: portable CPU baseline (scanned generated flags for {len(banned)} banned tokens).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
