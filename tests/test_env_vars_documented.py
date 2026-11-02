#!/usr/bin/env python3
"""Guard: every CRISPEMBED_* environment variable read by code is documented.

tools/list_env_vars.py inventories the quoted "CRISPEMBED_*" names in src/,
examples/, python/, crispembed/, crispembed-sys/ and flutter/. This test fails
when one of them does not appear in docs/environment-variables.md, or when the
generated inventory block there is stale. Fix with:

    python tools/list_env_vars.py --write

Source-parsing only; no build, no models.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "list_env_vars.py"
DOC = ROOT / "docs" / "environment-variables.md"


def main() -> int:
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
