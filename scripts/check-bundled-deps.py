#!/usr/bin/env python3
"""Assert that a packaged Linux directory can actually start on a clean machine.

Run this on the staged `pkg/` directory of any Linux release leg, before it is
tarred. It parses each ELF's dynamic section and fails if a `DT_NEEDED` entry is
neither bundled alongside it nor part of the base system every glibc Linux has.

Why this exists
---------------
SubtitleEdit#13205: `crispembed-server` from `crispembed-linux-x86_64.tar.gz`
exited with code 127 and printed nothing, on two unrelated distros. 127 from a
dynamically-linked binary means the loader could not resolve a library, so the
process dies before `main()`. The chain was:

    crispembed-server -> libggml.so.0      (bundled)
    libggml.so.0      -> libggml-blas.so.0 (bundled)
    libggml-blas.so.0 -> libopenblas.so.0  <-- NOT bundled, not installed by default

`-DGGML_BLAS=ON` put a hard link-time dependency on OpenBLAS into the artifact.
The CI runner had `libopenblas-dev` installed (the workflow apt-gets it so the
BLAS probe succeeds), so the build and every smoke check on the runner passed.
It could only ever fail on a user's machine — and it did, for every Linux user
of every release, on both x86_64 and arm64.

This is the same shape as issue #41: a property of the BUILD environment
leaking into a shipped artifact, invisible to a CI that shares that
environment. Hence a packaging-time gate rather than a test.

Usage
-----
    python scripts/check-bundled-deps.py pkg
    python scripts/check-bundled-deps.py pkg --max-glibc 2.35
"""

from __future__ import annotations

import argparse
import collections
import struct
import sys
from pathlib import Path

# Present on every glibc Linux — part of the C/C++ runtime a user already has.
# Deliberately NOT here: libgomp.so.1 (OpenMP; ships with gcc, absent on
# minimal installs), libopenblas/libblas/liblapack, libgfortran, libcuda,
# libvulkan. Anything in that second group must be bundled or not linked.
SYSTEM_LIBS = {
    "libc.so.6",
    "libm.so.6",
    "libdl.so.2",
    "libpthread.so.0",
    "librt.so.1",
    "libutil.so.1",
    "libgcc_s.so.1",
    "libstdc++.so.6",
    "libresolv.so.2",
}
SYSTEM_PREFIXES = ("ld-linux",)

DT_NEEDED, DT_STRTAB, DT_RPATH, DT_RUNPATH = 1, 5, 15, 29
DT_VERNEED, DT_VERNEEDNUM = 0x6FFFFFFE, 0x6FFFFFFF
PT_DYNAMIC, SHT_NOBITS = 2, 8

MACHINES = {0x3E: "x86-64", 0xB7: "aarch64", 0x28: "arm", 0xF3: "riscv"}


def is_system(lib: str) -> bool:
    return lib in SYSTEM_LIBS or lib.startswith(SYSTEM_PREFIXES)


def parse_elf(data: bytes) -> dict | None:
    """DT_NEEDED / RUNPATH / required symbol versions. None if not an ELF."""
    if len(data) < 64 or data[:4] != b"\x7fELF":
        return None
    is64 = data[4] == 2
    en = "<" if data[5] == 1 else ">"
    (machine,) = struct.unpack_from(en + "H", data, 0x12)
    if is64:
        (e_phoff,) = struct.unpack_from(en + "Q", data, 0x20)
        e_phentsize, e_phnum = struct.unpack_from(en + "HH", data, 0x36)
        (e_shoff,) = struct.unpack_from(en + "Q", data, 0x28)
        e_shentsize, e_shnum = struct.unpack_from(en + "HH", data, 0x3A)
    else:
        (e_phoff,) = struct.unpack_from(en + "I", data, 0x1C)
        e_phentsize, e_phnum = struct.unpack_from(en + "HH", data, 0x2A)
        (e_shoff,) = struct.unpack_from(en + "I", data, 0x20)
        e_shentsize, e_shnum = struct.unpack_from(en + "HH", data, 0x2E)

    # Section headers give the vaddr -> file-offset mapping the dynamic
    # section's pointers are expressed in.
    secs = []
    for i in range(e_shnum):
        o = e_shoff + i * e_shentsize
        if o + e_shentsize > len(data):
            break
        if is64:
            _n, typ, _f, addr, off, size = struct.unpack_from(en + "IIQQQQ", data, o)
        else:
            _n, typ, _f, addr, off, size = struct.unpack_from(en + "IIIIII", data, o)
        secs.append((addr, off, size, typ))

    def v2o(vaddr: int):
        for addr, off, size, typ in secs:
            if typ != SHT_NOBITS and addr <= vaddr < addr + size:
                return off + (vaddr - addr)
        return None

    dyn = None
    for i in range(e_phnum):
        o = e_phoff + i * e_phentsize
        if is64:
            (p_type,) = struct.unpack_from(en + "I", data, o)
            p_offset = struct.unpack_from(en + "Q", data, o + 8)[0]
            p_filesz = struct.unpack_from(en + "Q", data, o + 32)[0]
        else:
            p_type, p_offset = struct.unpack_from(en + "II", data, o)[:2]
            p_filesz = struct.unpack_from(en + "I", data, o + 16)[0]
        if p_type == PT_DYNAMIC:
            dyn = (p_offset, p_filesz)
    if dyn is None:
        return {"machine": machine, "static": True, "needed": [], "runpath": [], "versions": {}}

    ent = 16 if is64 else 8
    tags = []
    for o in range(dyn[0], dyn[0] + dyn[1], ent):
        if is64:
            d_tag, d_val = struct.unpack_from(en + "Qq", data, o)
        else:
            d_tag, d_val = struct.unpack_from(en + "Ii", data, o)
        if d_tag == 0:
            break
        tags.append((d_tag, d_val))
    d = dict(tags)
    strtab = v2o(d.get(DT_STRTAB, 0))
    if strtab is None:
        return {"machine": machine, "static": True, "needed": [], "runpath": [], "versions": {}}

    def s(off: int) -> str:
        end = data.index(b"\0", strtab + off)
        return data[strtab + off : end].decode("utf-8", "replace")

    versions: dict = collections.defaultdict(set)
    if DT_VERNEED in d:
        off = v2o(d[DT_VERNEED])
        for _ in range(d.get(DT_VERNEEDNUM, 0)):
            if off is None:
                break
            _v, cnt, file_off, aux, nxt = struct.unpack_from(en + "HHIII", data, off)
            lib = s(file_off)
            a = off + aux
            for _ in range(cnt):
                _h, _fl, _o, name_off, anext = struct.unpack_from(en + "IHHII", data, a)
                versions[lib].add(s(name_off))
                if not anext:
                    break
                a += anext
            if not nxt:
                break
            off += nxt

    return {
        "machine": machine,
        "static": False,
        "needed": [s(v) for t, v in tags if t == DT_NEEDED],
        "runpath": [s(d[t]) for t in (DT_RUNPATH, DT_RPATH) if t in d],
        "versions": dict(versions),
    }


def max_version(versions: dict, prefix: str) -> str:
    vals = []
    for names in versions.values():
        for n in names:
            if n.startswith(prefix):
                try:
                    vals.append(tuple(int(x) for x in n[len(prefix) :].split(".")))
                except ValueError:
                    pass
    return ".".join(map(str, max(vals))) if vals else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pkg_dir", help="staged package directory (contents of the archive)")
    ap.add_argument("--max-glibc", default=None,
                    help="fail if any binary requires a glibc newer than this (e.g. 2.35)")
    args = ap.parse_args()

    root = Path(args.pkg_dir)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    # A DT_NEEDED entry is satisfied by any name present in the tree, including
    # the SONAME symlinks cmake emits (libggml.so.0 -> libggml.so.0.10.2).
    provided = {p.name for p in root.rglob("*") if p.is_file() or p.is_symlink()}

    unbundled: list[tuple[str, str]] = []
    glibc_floor = ("", "")
    scanned = 0

    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            info = parse_elf(path.read_bytes())
        except (OSError, struct.error, ValueError) as exc:
            print(f"  {path.relative_to(root)}: unreadable as ELF ({exc})", file=sys.stderr)
            continue
        if info is None:
            continue
        scanned += 1
        rel = str(path.relative_to(root))
        missing = [n for n in info["needed"] if not is_system(n) and n not in provided]
        g = max_version(info["versions"], "GLIBC_")
        gxx = max_version(info["versions"], "GLIBCXX_")
        if g and tuple(int(x) for x in g.split(".")) > tuple(
            int(x) for x in (glibc_floor[0] or "0").split(".")
        ):
            glibc_floor = (g, rel)
        mark = "FAIL" if missing else " ok "
        print(f"  [{mark}] {rel}  [{MACHINES.get(info['machine'], hex(info['machine']))}]"
              f"  RUNPATH={','.join(info['runpath']) or '-'}"
              f"  GLIBC<={g or '-'} GLIBCXX<={gxx or '-'}")
        for m in missing:
            unbundled.append((rel, m))

    print(f"\nscanned {scanned} ELF file(s); highest glibc requirement: "
          f"{glibc_floor[0] or 'none'}" + (f" (from {glibc_floor[1]})" if glibc_floor[0] else ""))

    rc = 0
    if unbundled:
        print("\nERROR: these dependencies are neither bundled nor part of a base glibc system.\n"
              "The loader will fail before main() and the process exits 127 with no output:",
              file=sys.stderr)
        for who, lib in unbundled:
            print(f"  {who}  needs  {lib}", file=sys.stderr)
        print("\nEither stop linking it (preferred — check whether it actually buys anything)\n"
              "or copy it plus its own dependencies into the package directory.\n"
              "See SubtitleEdit#13205.", file=sys.stderr)
        rc = 1

    if args.max_glibc and glibc_floor[0]:
        want = tuple(int(x) for x in args.max_glibc.split("."))
        got = tuple(int(x) for x in glibc_floor[0].split("."))
        if got > want:
            print(f"\nERROR: requires GLIBC_{glibc_floor[0]} (from {glibc_floor[1]}) but the "
                  f"declared floor is {args.max_glibc}. Build on an older base image.",
                  file=sys.stderr)
            rc = 1

    if rc == 0:
        print("OK: every non-system dependency is bundled.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
