#!/usr/bin/env python3
"""Guard: every CRISPEMBED_* environment variable read by code is documented.

tools/list_env_vars.py inventories the quoted "CRISPEMBED_*" names in src/,
examples/, python/, crispembed/, crispembed-sys/ and flutter/. This test fails
when one of them does not appear in docs/environment-variables.md, or when the
generated inventory block there is stale. Fix with:

    python tools/list_env_vars.py --write

It also fails on any raw presence truth test (`getenv("X") != nullptr`,
`if (getenv("X"))`, `!getenv("X")`, `getenv("X") ? a : b`): those turn the
gate ON for `X=0`. Use core_env::on / core_env::explicitly_off (bool gates) or
core_env::value (paths/ints/strings; empty == unset) from src/core/env_gate.h.
A deliberate exception must be listed in PRESENCE_ALLOWLIST with a reason.

Source-parsing only; no build, no models.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "list_env_vars.py"
DOC = ROOT / "docs" / "environment-variables.md"


# Line-independent exceptions: {(relative path, VARIABLE): reason}.
# Keep empty unless a raw presence test is genuinely required (e.g. mirroring a
# third-party library's own presence semantics).
PRESENCE_ALLOWLIST: dict[tuple[str, str], str] = {}


def check_presence() -> int:
    sys.path.insert(0, str(TOOL.parent))
    import list_env_vars  # noqa: E402

    # Self-test: the detector must still see every presence spelling, and must
    # not flag the helper forms, or the guard below would pass vacuously.
    must = ['if (getenv("A_B"))', 'bool x = std::getenv("A_B") != nullptr;', 'if (!getenv("A_B")) f();',
            'x = getenv("A_B") ? 1 : 0;', 'if (std::getenv("A_B") == nullptr) f();', 'if (y && getenv("A_B")) f();']
    must_not = ['if (core_env::on("A_B"))', 'const char * v = std::getenv("A_B");',
                'if (const char * v = core_env::value("A_B")) f(v);', 'int n = atoi(getenv("A_B"));']
    for line in must:
        if list_env_vars.classify(line, "A_B") != "presence":
            print(f"FAIL: presence detector missed: {line}")
            return 1
    for line in must_not:
        if list_env_vars.classify(line, "A_B") == "presence":
            print(f"FAIL: presence detector false positive: {line}")
            return 1

    bad = [(loc, n, src) for loc, n, src in list_env_vars.presence_sites()
           if (loc.rsplit(":", 1)[0], n) not in PRESENCE_ALLOWLIST]
    if bad:
        print(f"FAIL: {len(bad)} raw getenv presence test(s) -- X=0 would turn these ON:")
        for loc, n, src in bad:
            print(f"  {loc}: {n}: {src[:140]}")
        print("use core_env::on / core_env::explicitly_off / core_env::value (src/core/env_gate.h)")
        return 1
    return 0


def main() -> int:
    if check_presence():
        return 1
    names = subprocess.run([sys.executable, str(TOOL), "--names"], check=True, capture_output=True,
                           text=True).stdout.split()
    if len(names) < 100:
        print(f"FAIL: inventory suspiciously short ({len(names)} names)")
        return 1
    doc = DOC.read_text()
    missing = [n for n in names if f"`{n}`" not in doc]
    if missing:
        print(f"FAIL: {len(missing)} CRISPEMBED_* variable(s) missing from docs/environment-variables.md:")
        for n in missing:
            print(f"  {n}")
        print("run: python tools/list_env_vars.py --write")
        return 1
    check = subprocess.run([sys.executable, str(TOOL), "--check"], capture_output=True, text=True)
    if check.returncode != 0:
        print("FAIL: " + check.stdout.strip())
        return 1
    print(f"OK: {len(names)} CRISPEMBED_* variables documented")
    return 0


if __name__ == "__main__":
    sys.exit(main())
