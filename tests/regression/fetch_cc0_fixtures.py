#!/usr/bin/env python3
"""Fetch the small, explicitly CC0/public-domain fixture seed set.

The source catalog is intentionally separate from the regression manifest:
these images have no reliable machine-readable gold transcription yet.  They
are still useful for live robustness checks, while annotations are added one
fixture at a time after human verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = Path(__file__).with_name("cc0_sources.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "tests/regression/images/cc0"))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    records = []
    for src in json.loads(SOURCES.read_text()):
        dest = out / src["file"]
        if args.force or not dest.exists():
            print(f"downloading {src['name']} -> {dest}", flush=True)
            request = urllib.request.Request(src["url"], headers={"User-Agent": "CrispEmbed-regression-fixtures/1.0"})
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, context=ctx, timeout=60) as response:
                        dest.write_bytes(response.read())
                    break
                except urllib.error.HTTPError:
                    if attempt == 2:
                        print(f"warning: unable to fetch {src['name']}; leaving it unvendored", flush=True)
                    else:
                        time.sleep(3 * (attempt + 1))
        if not dest.exists():
            continue
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        records.append({**src, "local_file": str(dest.relative_to(ROOT)),
                        "sha256": digest, "size": dest.stat().st_size})
    (out / "MANIFEST.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"fetched={len(records)} manifest={out / 'MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
