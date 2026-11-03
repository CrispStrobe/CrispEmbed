#!/usr/bin/env python3
"""Attribute a WebAssembly binary's size to sections and functions.

Usage: wasm_size_report.py MODULE.wasm [SYMBOL_MAP] [--top N]

SYMBOL_MAP is the file written by emcc --emit-symbol-map (lines
"<function index>:<mangled name>"). Without it, functions are reported by
index (or by the name section, if the module still has one). Names are
demangled with c++filt / llvm-cxxfilt when one is on PATH.

The report lists section sizes, the largest functions, and totals grouped by
a coarse owner key (C++ namespace / class, or the C-name prefix before the
first underscore-delimited word), which is enough to see which engines or
libraries dominate a build.
"""
import shutil
import subprocess
import sys
from collections import defaultdict

SECTION_NAMES = {0: "custom", 1: "type", 2: "import", 3: "function", 4: "table", 5: "memory",
                 6: "global", 7: "export", 8: "start", 9: "element", 10: "code", 11: "data",
                 12: "datacount", 13: "tag"}


def uleb(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def read_name(buf, pos):
    n, pos = uleb(buf, pos)
    return buf[pos:pos + n].decode("utf-8", "replace"), pos + n


def parse(buf):
    if buf[:4] != b"\0asm":
        raise SystemExit("not a wasm module")
    pos = 8
    sections = []
    n_func_imports = 0
    bodies = []
    names = {}
    while pos < len(buf):
        sid = buf[pos]
        size, p = uleb(buf, pos + 1)
        end = p + size
        label = SECTION_NAMES.get(sid, str(sid))
        if sid == 0:
            cname, _ = read_name(buf, p)
            label = "custom:" + cname
            if cname == "name":
                q = p + 1 + len(cname)
                while q < end:
                    sub = buf[q]
                    ssize, q = uleb(buf, q + 1)
                    if sub == 1:
                        cnt, r = uleb(buf, q)
                        for _ in range(cnt):
                            idx, r = uleb(buf, r)
                            nm, r = read_name(buf, r)
                            names[idx] = nm
                    q += ssize
        elif sid == 2:
            cnt, q = uleb(buf, p)
            for _ in range(cnt):
                _, q = read_name(buf, q)
                _, q = read_name(buf, q)
                kind = buf[q]
                q += 1
                if kind == 0:
                    n_func_imports += 1
                    _, q = uleb(buf, q)
                elif kind == 1:
                    q += 1
                    flags, q = uleb(buf, q)
                    _, q = uleb(buf, q)
                    if flags & 1:
                        _, q = uleb(buf, q)
                elif kind == 2:
                    flags, q = uleb(buf, q)
                    _, q = uleb(buf, q)
                    if flags & 1:
                        _, q = uleb(buf, q)
                elif kind == 3:
                    q += 2
                elif kind == 4:
                    q += 1
                    _, q = uleb(buf, q)
        elif sid == 10:
            cnt, q = uleb(buf, p)
            for i in range(cnt):
                bsize, r = uleb(buf, q)
                bodies.append((n_func_imports + i, r - q + bsize))
                q = r + bsize
        sections.append((label, end - pos))
        pos = end
    return sections, bodies, names


def demangle(names):
    tool = shutil.which("c++filt") or shutil.which("llvm-cxxfilt")
    if not tool or not names:
        return names
    keys = list(names)
    out = subprocess.run([tool], input="\n".join(names[k] for k in keys), capture_output=True,
                         text=True).stdout.split("\n")
    if len(out) < len(keys):
        return names
    return {k: out[i] for i, k in enumerate(keys)}


def owner(name):
    base = name.split("(")[0]
    if "::" in base:
        parts = [p for p in base.split("::") if p and not p.startswith("(anonymous")]
        if base.startswith("(anonymous namespace)::") and len(parts) > 1:
            return "anon::" + parts[0]
        return "::".join(parts[:2]) if parts[0] in ("std", "core", "ocr_pipeline") and len(parts) > 2 \
            else parts[0]
    if base.startswith("$") or base.isdigit():
        return "(unnamed)"
    head = base.lstrip("_").split("_")
    return "_".join(head[:2]) if head[0] in ("crispembed", "ggml", "wasm", "core") and len(head) > 1 \
        else head[0]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--top")]
    top = 60
    for a in sys.argv[1:]:
        if a.startswith("--top="):
            top = int(a.split("=", 1)[1])
    with open(args[0], "rb") as f:
        buf = f.read()
    sections, bodies, names = parse(buf)
    if len(args) > 1 and not __import__("os").path.exists(args[1]):
        print(f"symbol map {args[1]} not found; reporting by index")
    elif len(args) > 1:
        with open(args[1]) as f:
            for line in f:
                if ":" in line:
                    idx, nm = line.rstrip("\n").split(":", 1)
                    if idx.isdigit():
                        names[int(idx)] = nm
    names = demangle(names)
    print(f"total {len(buf)} bytes")
    for label, size in sorted(sections, key=lambda s: -s[1]):
        print(f"  section {label:24s} {size:>10d}")
    code_total = sum(s for _, s in bodies)
    print(f"\ncode bodies: {len(bodies)} functions, {code_total} bytes")
    groups = defaultdict(lambda: [0, 0])
    named = []
    for idx, size in bodies:
        nm = names.get(idx, f"${idx}")
        named.append((size, nm))
        g = groups[owner(nm)]
        g[0] += size
        g[1] += 1
    print(f"\ntop {top} owners (bytes, functions):")
    for k, (s, n) in sorted(groups.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"  {s:>9d} {n:>5d}  {k}")
    print(f"\ntop {top} functions:")
    for s, nm in sorted(named, reverse=True)[:top]:
        print(f"  {s:>9d}  {nm[:150]}")


if __name__ == "__main__":
    main()
