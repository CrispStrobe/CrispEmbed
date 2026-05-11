#!/usr/bin/env python3
"""Verify every HF model ID + GGUF download URL referenced by CrispEmbed's
registries actually resolves on huggingface.co.

Scans five registry locations:

  1. examples/cli/model_mgr.cpp     — k_registry entries (download URLs)
  2. models/upload_to_hf.py         — MODELS["...base_model..."] (HF IDs)
  3. tests/debug_model.py           — GGUF_TO_HF (HF IDs for auto-detect)
  4. tests/test_all_models.py       — MODEL_MAP (HF IDs)
  5. tests/benchmark.py             — MODEL_MAP (HF IDs)

For each HF model ID, calls `HfApi().model_info(repo_id)` — the token-aware
path that can distinguish:
    OK         → repo exists and is accessible
    GATED      → exists but needs license acceptance / org membership
    NOT_FOUND  → repo really doesn't exist (caught a bogus org name)
    ERROR      → network / unexpected exception

Anonymous HTTP HEAD on the HF API returns 401 for both gated repos *and*
nonexistent ones, which is why an earlier no-auth version of this script
silently let `CrispStrobe/PIXIE-Rune-v1.0` through (no such HF org — the
GitHub user is CrispStrobe, the HF user is cstr).

For each download URL, follows redirects (HF resolve → CDN) using the
authed urllib opener. NOT_FOUND on `/resolve/main/<file>.gguf` typically
means the file name doesn't match what's actually uploaded.

Exit code: number of NOT_FOUND or error rows.

Usage:
    HF_HOME=/Volumes/backups/ai/huggingface-hub \\
    HUGGINGFACE_HUB_CACHE=/Volumes/backups/ai/huggingface-hub \\
        python tests/check_registry_urls.py
    python tests/check_registry_urls.py --only urls
    python tests/check_registry_urls.py --only ids
    python tests/check_registry_urls.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────────
# Source parsers
# ────────────────────────────────────────────────────────────────────────

def parse_cpp_registry(path: Path) -> List[Tuple[str, str, str]]:
    """examples/cli/model_mgr.cpp — k_registry struct array."""
    text = path.read_text()
    pattern = re.compile(
        r'\{\s*"([^"]+)"\s*,\s*"([^"]+\.gguf)"\s*,\s*"(https?://[^"]+)"',
        re.MULTILINE,
    )
    return [(m.group(1), m.group(3), path.name) for m in pattern.finditer(text)]


def parse_python_dict(path: Path, var_name: str,
                      value_kind: str) -> List[Tuple[str, str, str]]:
    """Generic parser for `<var> = { "key": "value", … }` or nested
    base_model fields."""
    text = path.read_text()
    out: List[Tuple[str, str, str]] = []
    if value_kind == "flat":
        m = re.search(rf'{var_name}\s*=\s*\{{(.+?)\n\}}', text, re.DOTALL)
        if not m:
            return out
        body = m.group(1)
        for k, v in re.findall(r'"([\w\-.]+)"\s*:\s*\n?\s*"([^"]+)"', body):
            out.append((k, v, path.name))
    elif value_kind == "base_model_nested":
        for k, v in re.findall(
                r'"([\w\-.]+)"\s*:\s*\{[^{}]*?'
                r'"base_model"\s*:\s*"([^"]+)"',
                text, re.DOTALL):
            out.append((k, v, path.name))
    return out


# ────────────────────────────────────────────────────────────────────────
# HuggingFace check (token-aware, distinguishes NOT_FOUND from GATED)
# ────────────────────────────────────────────────────────────────────────

def check_hf_model_id_authed(repo_id: str, api) -> str:
    from huggingface_hub.errors import (
        RepositoryNotFoundError, GatedRepoError, HfHubHTTPError,
    )
    try:
        api.model_info(repo_id)
        return "OK"
    except GatedRepoError:
        return "GATED"
    except RepositoryNotFoundError:
        return "NOT_FOUND"
    except HfHubHTTPError as e:
        # If we hit a 401 even with token, it means the user's token can't
        # see the repo — could be private, deleted, or really gated. Treat
        # as ambiguous.
        if e.response is not None and e.response.status_code == 401:
            return "AUTH_REQUIRED"
        return f"HTTP_{getattr(e.response, 'status_code', '?')}"
    except Exception as e:
        return f"ERROR_{type(e).__name__}"


# ────────────────────────────────────────────────────────────────────────
# Download URL check
# ────────────────────────────────────────────────────────────────────────

def _make_opener(token):
    """urllib opener that adds an Authorization header so HF returns
    accurate status codes for cstr/* private repos (the user is logged in
    as cstr and these are their own uploads — non-existent should be 404,
    not 401)."""
    handlers = []
    opener = urllib.request.build_opener(*handlers)
    headers = [("User-Agent", "crispembed-registry-check/0.3")]
    if token:
        headers.append(("Authorization", f"Bearer {token}"))
    opener.addheaders = headers
    return opener


def head_with_redirects(opener, url: str, max_hops: int = 5) -> Tuple[int, str]:
    """Follow redirects manually so we see the final tier's status."""
    for _ in range(max_hops):
        req = urllib.request.Request(url, method="HEAD")
        try:
            resp = opener.open(req, timeout=20)
            if 300 <= resp.status < 400:
                loc = resp.headers.get("Location")
                if not loc:
                    return resp.status, url
                url = loc
                continue
            return resp.status, url
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                loc = e.headers.get("Location") if e.headers else None
                if not loc:
                    return e.code, url
                url = loc
                continue
            return e.code, url
        except Exception as e:
            return -1, f"{type(e).__name__}: {e}"
    return -2, url


def check_gguf_url(opener, url: str) -> str:
    status, _ = head_with_redirects(opener, url)
    if status == 200: return "OK"
    if status == 401: return "AUTH_REQUIRED"
    if status == 403: return "GATED"
    if status == 404: return "NOT_FOUND"
    if status == -1:  return "NETWORK_ERR"
    return f"HTTP_{status}"


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["ids", "urls"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # Token-aware HF API (uses ~/.cache/huggingface/token or HF_HOME)
    from huggingface_hub import HfApi, get_token
    api = HfApi()
    token = get_token()
    if not token:
        print("WARNING: no HF token found — HF will return 401 for both "
              "gated repos and non-existent repos, so NOT_FOUND will be "
              "indistinguishable from GATED. Set HF_HOME or run "
              "`huggingface-cli login` for accurate results.",
              file=sys.stderr)
    opener = _make_opener(token)

    ids:  List[Tuple[str, str, str]] = []
    urls: List[Tuple[str, str, str]] = []

    if args.only != "ids":
        urls += parse_cpp_registry(REPO / "examples/cli/model_mgr.cpp")

    if args.only != "urls":
        ids += parse_python_dict(REPO / "models/upload_to_hf.py",
                                  "MODELS", "base_model_nested")
        ids += parse_python_dict(REPO / "tests/debug_model.py",
                                  "GGUF_TO_HF", "flat")
        ids += parse_python_dict(REPO / "tests/test_all_models.py",
                                  "MODEL_MAP", "flat")
        ids += parse_python_dict(REPO / "tests/benchmark.py",
                                  "MODEL_MAP", "flat")

    rows: List[Dict[str, str]] = []

    for name, repo_id, src in ids:
        st = check_hf_model_id_authed(repo_id, api)
        rows.append({"kind": "model_id", "source": src,
                     "name": name, "target": repo_id, "status": st})

    for name, url, src in urls:
        st = check_gguf_url(opener, url)
        rows.append({"kind": "url", "source": src,
                     "name": name, "target": url, "status": st})

    if args.json:
        print(json.dumps(rows, indent=2))
        bad = sum(1 for r in rows if r["status"] not in ("OK", "GATED"))
        return bad

    # Text table.
    print(f"{'':<2} {'kind':<8s} {'source':<24s} {'name':<42s} → "
          f"{'target':<70s} {'status'}")
    print("─" * 175)
    for r in rows:
        badge = ("✓" if r["status"] == "OK"
                 else "·" if r["status"] == "GATED"
                 else "✗")
        target = r["target"].replace("https://huggingface.co/", "")
        if len(target) > 68:
            target = target[:65] + "..."
        print(f"{badge:<2} {r['kind']:<8s} {r['source']:<24s} "
              f"{r['name']:<42s} → {target:<70s} {r['status']}")

    n_ok     = sum(1 for r in rows if r["status"] == "OK")
    n_gated  = sum(1 for r in rows if r["status"] == "GATED")
    n_bad    = sum(1 for r in rows if r["status"] not in ("OK", "GATED"))
    print("")
    print(f"  {n_ok} OK   {n_gated} gated   {n_bad} broken   "
          f"({len(rows)} total)")
    if n_bad:
        print("\nBROKEN entries:")
        for r in rows:
            if r["status"] not in ("OK", "GATED"):
                print(f"  {r['kind']:<8s} {r['source']:<24s} "
                      f"{r['name']:<42s} → {r['target']}   [{r['status']}]")
    return n_bad


if __name__ == "__main__":
    sys.exit(main())
