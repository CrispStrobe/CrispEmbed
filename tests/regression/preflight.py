#!/usr/bin/env python3
"""Tier 1 preflight: HEAD-check every pinned HF artifact in the manifest.

No downloads — just a network HEAD against each GGUF (and each diff ref
that is *expected* to exist). Catches a dead pin / typo'd filename / moved
repo in ~5 s of PR CI, before the nightly job spends minutes downloading.

A diff `ref` that 404s is reported as a NOTE (not a failure): refs are
opt-in, so a missing ref just means that model's exact diff check is not
enabled yet. A missing GGUF-under-test IS a failure.

Usage:  python tests/regression/preflight.py
Exit:   0 = all required artifacts reachable, 1 = at least one missing.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"


def resolve_url(repo: str, file: str, revision: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{file}"


def head_ok(url: str) -> tuple[bool, int]:
    token = os.environ.get("HF_TOKEN")
    req = urllib.request.Request(url, method="HEAD")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return (200 <= r.status < 400), r.status
    except urllib.error.HTTPError as e:
        # HF returns 302 → CDN for real files; urllib follows it. A 401/403
        # on a public repo without token still means the object exists.
        if e.code in (401, 403):
            return True, e.code
        return False, e.code
    except Exception:  # noqa: BLE001
        return False, 0


def main() -> int:
    manifest = json.loads(MANIFEST.read_text())
    failures = 0
    for m in manifest["models"]:
        g = m["gguf"]
        url = resolve_url(g["repo"], g["file"], g.get("revision", "main"))
        ok, code = head_ok(url)
        print(f"[{'OK ' if ok else 'MISS'}] {m['name']} gguf {g['repo']}/{g['file']}@{g.get('revision','main')} ({code})")
        if not ok:
            failures += 1
        diff = m.get("diff")
        if diff:
            r = diff["ref"]
            rurl = resolve_url(r["repo"], r["file"], r.get("revision", "main"))
            rok, rcode = head_ok(rurl)
            tag = "OK " if rok else "note"
            print(f"[{tag}] {m['name']} diff-ref {r['repo']}/{r['file']} ({rcode})"
                  + ("" if rok else "  — diff check not enabled until this ref is uploaded"))
    print(f"\n=== preflight: {failures} required artifact(s) missing ===")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
