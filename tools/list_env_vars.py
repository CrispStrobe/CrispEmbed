#!/usr/bin/env python3
"""Inventory every runtime environment variable read by CrispEmbed code.

Scans src/, examples/, python/, crispembed/, crispembed-sys/ and flutter/ for
string literals naming an environment variable and classifies how each is read:

  on        core_env::on(...)              bool gate, default OFF (unset/""/0/false/off/no = off)
  off       core_env::explicitly_off(...)  bool gate, default ON  (only 0/false/off/no = off)
  presence  raw getenv(X) truth test       forbidden (X=0 would mean ON); see --presence
  value     getenv/os.environ/env::var     value (int/float/path/string) or raw presence
  set       setenv/_putenv_s/set_var       written by the CLI/orchestrator, not a user knob

Every quoted "CRISPEMBED_*" literal is included. Variables without the prefix
(legacy per-engine debug switches, GGML_*, HOME, ...) are included when the
line also contains an environment accessor.

Usage:
  python tools/list_env_vars.py              # print the generated Markdown table
  python tools/list_env_vars.py --write      # refresh the generated block in
                                             # docs/environment-variables.md
  python tools/list_env_vars.py --check      # exit 1 if the block is stale
  python tools/list_env_vars.py --names      # one CRISPEMBED_* name per line
  python tools/list_env_vars.py --presence   # raw presence truth tests (must be none)

The generated block sits between the BEGIN/END GENERATED markers; everything
outside it is hand-written and preserved.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "environment-variables.md"
SCAN_DIRS = ["src", "examples", "python", "crispembed", "crispembed-sys", "flutter"]
EXTS = {".c", ".cc", ".cpp", ".h", ".hpp", ".m", ".mm", ".py", ".rs", ".dart"}
SKIP_PARTS = {"build", "target", ".dart_tool", "__pycache__", "third_party", "vendor"}
BEGIN = "<!-- BEGIN GENERATED: tools/list_env_vars.py -->"
END = "<!-- END GENERATED: tools/list_env_vars.py -->"

PREFIXED = re.compile(r'"(CRISPEMBED_[A-Z0-9_]*[A-Z0-9])"')
ANY_NAME = re.compile(r'"([A-Z][A-Z0-9_]{2,}[A-Z0-9])"')
ACCESSOR = re.compile(
    r"getenv|core_env::|os\.environ|env::var|Platform\.environment|env_flag|env_int|env_float|"
    r"got_gate|setenv|_putenv_s|set_var|set_env_if_unset"
)
# CRISPEMBED_FOO_* are placeholder examples in src/core/env_gate.h comments.
NOT_ENV = {"CRISPEMBED_API", "CRISPEMBED_FOO_BENCH", "CRISPEMBED_FOO_CLEANUP"}
# Names in ANY_NAME position that are not environment variables.
NON_PREFIX_DENY = re.compile(r"^(GGUF|GGML_TYPE|F16|F32|Q\d|BF16|JSON|POST|GET|UTF|RGB|PNG|USAGE)")


def iter_files():
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.suffix in EXTS and p.is_file() and not (SKIP_PARTS & set(p.relative_to(ROOT).parts)):
                yield p


def classify(line: str, name: str) -> str:
    q = re.escape(f'"{name}"')
    if re.search(r"core_env::explicitly_off\(\s*" + q, line):
        return "off"
    if re.search(r"core_env::on\(\s*" + q, line) or re.search(r"env_flag\(\s*" + q, line):
        return "on"
    if re.search(r"(setenv|_putenv_s|set_var|set_env_if_unset)\(\s*" + q, line):
        return "set"
    if re.search(r"got_gate\(\s*" + q, line):
        return "on/off"
    # Raw presence test: getenv("X") used directly as a boolean. Any value,
    # including "0", turns it on -- the defect class core_env exists to avoid.
    if re.search(r"getenv\(\s*" + q + r"\s*\)\s*([!=]=\s*(nullptr|NULL)|\)|&&|\|\||\?)", line) and not re.search(
            r"(const char\s*\*\s*\w+\s*=|auto\s*\*?\s*\w+\s*=|ato[ifl]\(|strto\w+\(|sto\w+\(|std::string\()\s*(std::)?getenv\(\s*" + q, line):
        return "presence"
    return "value"


def presence_sites():
    """(rel:line, name, source) for every raw presence truth test still in the tree."""
    out = []
    for p in iter_files():
        rel = p.relative_to(ROOT).as_posix()
        try:
            text = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(text):
            if "getenv" not in line:
                continue
            for n in sorted((set(PREFIXED.findall(line)) | set(ANY_NAME.findall(line))) - NOT_ENV):
                if classify(line, n) == "presence":
                    out.append((f"{rel}:{i + 1}", n, line.strip()))
    return out


def value_type(name: str, lines: list[str]) -> str:
    blob = " ".join(lines)
    if re.search(r"(_DIR|_PATH|_FILE|_ROOT|_LIB|_MODEL|_REF|_OUT)$", name):
        return "path"
    if re.search(r"ato[f]|strtof|strtod|stof|stod|float\(", blob):
        return "float"
    if re.search(r"atoi|atol|strtol|strtoul|stoi|stoul|stoll|int\(", blob):
        return "int"
    if re.search(r"strcmp|== *std::string|== *\"|\bin \(", blob):
        return "string"
    return "value"


def area_of(name: str, files: list[str]) -> str:
    # Prefer the engine source that reads it over the CLI/server that forwards it.
    f = next((x for x in files if x.startswith("src/") and not x.startswith("src/crispembed.")), files[0])
    stem = Path(f).stem
    top = f.split("/")[0]
    if top != "src":
        return {"examples": "CLI / server / examples", "python": "Python bindings",
                "crispembed": "Rust bindings", "crispembed-sys": "Rust bindings",
                "flutter": "Dart / Flutter"}.get(top, top)
    if f.startswith("src/core/") or stem in {"crispembed", "model_mgr"}:
        return "core"
    return stem


def comment_hint(lines: list[str], idx: int) -> str:
    line = lines[idx]
    m = re.search(r"//\s*(.+)$", line)
    if m and '"' not in m.group(1)[:1]:
        return m.group(1).strip()
    out = []
    j = idx - 1
    while j >= 0 and len(out) < 2:
        s = lines[j].strip()
        if s.startswith("//") or s.startswith("#") and not s.startswith("#include") and not s.startswith("#if"):
            out.insert(0, s.lstrip("/#! ").strip())
            j -= 1
        else:
            break
    return " ".join(out)


def collect():
    recs = defaultdict(lambda: {"kinds": set(), "locs": [], "lines": [], "hint": ""})
    for p in iter_files():
        rel = p.relative_to(ROOT).as_posix()
        try:
            text = p.read_text(errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(text):
            names = set(PREFIXED.findall(line))
            if ACCESSOR.search(line):
                names |= {n for n in ANY_NAME.findall(line) if not NON_PREFIX_DENY.match(n)}
            for n in names - NOT_ENV:
                r = recs[n]
                r["kinds"].add(classify(line, n))
                r["locs"].append(f"{rel}:{i + 1}")
                r["lines"].append(" ".join(text[i:i + 2]))
                if not r["hint"]:
                    r["hint"] = comment_hint(text, i)
    # Non-prefixed names must actually be read somewhere (not just mentioned).
    for n in [n for n in recs if not n.startswith("CRISPEMBED_")]:
        if not any(ACCESSOR.search(l) for l in recs[n]["lines"]):
            del recs[n]
    return recs


def default_of(kinds: set[str]) -> str:
    if "off" in kinds and "on" not in kinds:
        return "ON (opt-out `=0`)"
    if "on" in kinds and "off" not in kinds:
        return "off"
    if {"on", "off"} <= kinds:
        return "mixed (see code)"
    if "on/off" in kinds:
        return "per-gate default"
    if kinds == {"set"}:
        return "set internally"
    if "presence" in kinds:
        return "off (any value, even `0`, = on)"
    return "unset"


def status_of(name: str, kinds: set[str]) -> str:
    if kinds == {"presence"} and not re.search(r"DUMP|DEBUG|TRACE|BENCH", name):
        return "legacy presence gate"
    if re.search(r"DUMP|DEBUG|_DBG|BENCH|TRACE|TIMING|PROFILE|VERBOSE|INJECT|PROBE|OBSERV|_LOG\b|STATS", name):
        return "debug"
    if kinds == {"set"}:
        return "internal"
    if "off" in kinds and "on" not in kinds:
        return "production default (opt-out)"
    if "on" in kinds:
        return "default-off experimental / opt-in"
    return "tuning / config"


def sanitize(s: str) -> str:
    s = s.replace("|", "\\|").replace("`", "'")
    return (s[:157] + "...") if len(s) > 160 else s


def render(recs) -> str:
    groups = defaultdict(list)
    for n, r in recs.items():
        files = sorted({l.rsplit(":", 1)[0] for l in r["locs"]})
        groups[area_of(n, files) if n.startswith("CRISPEMBED_") else "non-prefixed (legacy / third-party)"].append((n, r, files))
    out = [BEGIN, "",
           f"{sum(1 for n in recs if n.startswith('CRISPEMBED_'))} `CRISPEMBED_*` variables and "
           f"{sum(1 for n in recs if not n.startswith('CRISPEMBED_'))} non-prefixed variables. "
           "Kind: `on` = `core_env::on` (default off), `off` = `core_env::explicitly_off` "
           "(default on, `=0`/`false`/`off`/`no` opts out), `presence` = raw `getenv(X)` truth test (forbidden: "
           "`tests/test_env_vars_documented.py` fails on one), `value` = value read via getenv/os.environ, `set` = written by the "
           "CLI/orchestrator. Hint = nearest source comment (may be empty).", ""]
    for g in sorted(groups, key=lambda k: (k.startswith("non-prefixed"), k)):
        out += [f"### {g}", "", "| Variable | Kind | Default | Type | Status | Where | Hint |",
                "|---|---|---|---|---|---|---|"]
        for n, r, files in sorted(groups[g]):
            kinds = r["kinds"]
            typ = "bool" if kinds & {"on", "off", "on/off"} and "value" not in kinds else value_type(n, r["lines"])
            where = ", ".join(f"`{f}`" for f in files[:3]) + (f" +{len(files) - 3}" if len(files) > 3 else "")
            out.append(f"| `{n}` | {'/'.join(sorted(kinds))} | {default_of(kinds)} | {typ} | "
                       f"{status_of(n, kinds)} | {where} | {sanitize(r['hint'])} |")
        out.append("")
    out.append(END)
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--names", action="store_true")
    g.add_argument("--presence", action="store_true",
                   help="list raw getenv presence truth tests (should be empty; use core_env::on/value)")
    a = ap.parse_args()
    if a.presence:
        for loc, n, src in presence_sites():
            print(f"{loc}\t{n}\t{src}")
        return 0
    recs = collect()
    if a.names:
        print("\n".join(sorted(n for n in recs if n.startswith("CRISPEMBED_"))))
        return 0
    block = render(recs)
    if not (a.write or a.check):
        print(block)
        return 0
    doc = DOC.read_text()
    if BEGIN not in doc or END not in doc:
        print(f"{DOC.relative_to(ROOT)} lacks generated markers", file=sys.stderr)
        return 1
    new = doc[:doc.index(BEGIN)] + block + doc[doc.index(END) + len(END):]
    if a.check:
        if new != doc:
            print("docs/environment-variables.md generated block is stale; run tools/list_env_vars.py --write")
            return 1
        print("OK")
        return 0
    DOC.write_text(new)
    print(f"wrote {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
