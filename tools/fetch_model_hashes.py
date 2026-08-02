#!/usr/bin/env python3
"""Generate examples/cli/model_hashes.h — SHA-256 pins for every auto-download
model in CrispEmbed's registry.

The registry in examples/cli/model_mgr.cpp hands curl/wget an https URL and
renames whatever comes back into the model cache. Without a pin, a compromised
or swapped re-host silently replaces the weights of an already-trusted model
name. GGUF is a graph plus tensor data that the process then executes; "it
downloaded fine" is not an integrity statement.

Hashes come from HuggingFace's tree API, which reports the SHA-256 of every
LFS-backed file without transferring it:

    GET /api/models/<owner>/<repo>/tree/<rev>?recursive=1
        -> [{"path": ..., "lfs": {"oid": "<sha256>"}}, ...]

Only `lfs.oid` is a SHA-256. A non-LFS file's `oid` is a git blob SHA-1 and is
NOT interchangeable — such files are reported as unpinnable rather than pinned
with the wrong digest.

Usage:
    python tools/fetch_model_hashes.py              # rewrite the header
    python tools/fetch_model_hashes.py --check      # CI: fail if stale
    python tools/fetch_model_hashes.py --json       # dump what was resolved
    python tools/fetch_model_hashes.py --check-sizes  # CI: declared vs real size

The same tree call also carries each file's size, so the declared `approx_size`
in the registry can be checked against reality for free. That matters because
the size is what a user is shown before agreeing to a download, and nothing
else verifies it: this check found four wrong entries on its first run, on top
of the four the pinning audit had already found by hand.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_CPP = REPO_ROOT / "examples" / "cli" / "model_mgr.cpp"
HEADER_OUT = REPO_ROOT / "examples" / "cli" / "model_hashes.h"

# https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path-with-slashes>
URL_RE = re.compile(
    r"https://huggingface\.co/"
    r"(?P<owner>[^/\"]+)/(?P<repo>[^/\"]+)"
    r"/resolve/(?P<rev>[^/\"]+)/(?P<path>[^\"]+)"
)

API = "https://huggingface.co/api/models/{owner}/{repo}/tree/{rev}?recursive=1"


def registry_urls(cpp_path: Path) -> "OrderedDict[str, dict]":
    """Every distinct resolve-URL in the registry, in source order."""
    text = cpp_path.read_text(encoding="utf-8")
    # Confine the scan to the k_registry array so unrelated URLs in comments
    # or helper code do not become pins.
    start = text.index("static const ModelEntry k_registry[]")
    end = text.index("\n};", start)
    body = text[start:end]

    out: "OrderedDict[str, dict]" = OrderedDict()
    for m in URL_RE.finditer(body):
        out.setdefault(m.group(0), m.groupdict())
    return out


# Each ModelEntry is { name, filename, url, desc, approx_size, license, card }.
# The size is the only field shaped like "<number> <unit>", which makes it
# findable without parsing C++ — and it is the field most likely to be a
# guess: the audit that added pinning found four entries off by 1.5x to 250x
# (pix2struct 300->467 MB, german-ocr 1301->1684, h2ovl-800m 398->644, and
# glotlid claiming 3.3 MB for an 848 MB model).
ENTRY_RE = re.compile(r"\{\s*\"(?P<name>[^\"]+)\"(?P<body>.*?)\}\s*,", re.S)
SIZE_RE = re.compile(r"\"(?P<size>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>[KMG]B)\"")

_UNIT = {"KB": 1e3, "MB": 1e6, "GB": 1e9}


def registry_sizes(cpp_path: Path) -> "OrderedDict[str, tuple]":
    """url -> (model_name, declared_bytes, declared_text) for every entry that
    declares both a download URL and a size."""
    text = cpp_path.read_text(encoding="utf-8")
    start = text.index("static const ModelEntry k_registry[]")
    end = text.index("\n};", start)
    body = text[start:end]

    out: "OrderedDict[str, tuple]" = OrderedDict()
    for m in ENTRY_RE.finditer(body):
        blob = m.group("body")
        url_m = URL_RE.search(blob)
        size_m = SIZE_RE.search(blob)
        if not url_m or not size_m:
            continue
        declared = float(size_m.group("size")) * _UNIT[size_m.group("unit")]
        out.setdefault(url_m.group(0), (m.group("name"), declared, size_m.group(0).strip('"')))
    return out


def fetch_tree(owner: str, repo: str, rev: str) -> dict:
    """path -> (sha256, size_bytes), for LFS-backed files only."""
    url = API.format(owner=owner, repo=repo, rev=rev)
    req = urllib.request.Request(url, headers={"User-Agent": "crispembed-hash-pin"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        entries = json.load(resp)
    return {
        e["path"]: (e["lfs"]["oid"], int(e["lfs"].get("size") or e.get("size") or 0))
        for e in entries
        if isinstance(e, dict) and e.get("type") == "file" and e.get("lfs")
    }


def resolve(urls: "OrderedDict[str, dict]", verbose: bool = True):
    """Returns (pins, unpinned) — unpinned carries a human-readable reason."""
    trees: dict = {}
    pins: "OrderedDict[str, str]" = OrderedDict()
    unpinned: list = []

    for url, parts in urls.items():
        key = (parts["owner"], parts["repo"], parts["rev"])
        if key not in trees:
            try:
                trees[key] = fetch_tree(*key)
                if verbose:
                    print(f"  {key[0]}/{key[1]}@{key[2]}: {len(trees[key])} LFS files",
                          file=sys.stderr)
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError) as exc:
                trees[key] = {}
                if verbose:
                    print(f"  {key[0]}/{key[1]}@{key[2]}: FAILED ({exc})", file=sys.stderr)

        entry = trees[key].get(parts["path"])
        sha = entry[0] if entry else None
        if sha:
            pins[url] = sha
        else:
            unpinned.append((url, "not an LFS file, or repo/path unreachable"))
    return pins, unpinned


def render_header(pins: "OrderedDict[str, str]", unpinned: list) -> str:
    lines = [
        "// model_hashes.h — GENERATED, do not edit by hand.",
        "//",
        "// SHA-256 pins for CrispEmbed's auto-download registry. Regenerate with:",
        "//     python tools/fetch_model_hashes.py",
        "// and verify coverage in CI with:",
        "//     python tools/fetch_model_hashes.py --check",
        "//",
        "// Digests are HuggingFace LFS object IDs, which are the SHA-256 of the file",
        "// content. download_file() refuses to install a payload whose digest does not",
        "// match the pin for its URL, so a swapped re-host fails closed instead of",
        "// being executed as a graph.",
        "#pragma once",
        "",
        "#include <cstring>",
        "",
        "namespace crispembed_mgr {",
        "",
        "struct ModelHash {",
        "    const char * url;",
        "    const char * sha256;",
        "};",
        "",
        "// clang-format off",
        "static const ModelHash k_model_hashes[] = {",
    ]
    for url, sha in pins.items():
        lines.append(f'    {{ "{url}",')
        lines.append(f'      "{sha}" }},')
    lines.append("    { nullptr, nullptr },")
    lines.append("};")
    lines.append("// clang-format on")
    lines.append("")
    if unpinned:
        lines.append("// Unpinned at generation time (integrity NOT enforced for these):")
        for url, why in unpinned:
            lines.append(f"//   {url}  — {why}")
        lines.append("")
    lines += [
        "// Returns the pinned SHA-256 for a download URL, or nullptr when the URL is",
        "// not pinned. A nullptr is not a pass: the caller decides whether an unpinned",
        "// download is tolerable, and says so out loud.",
        "inline const char * model_pinned_sha256(const char * url) {",
        "    if (!url) return nullptr;",
        "    for (const ModelHash * h = k_model_hashes; h->url; h++) {",
        "        if (strcmp(h->url, url) == 0) return h->sha256;",
        "    }",
        "    return nullptr;",
        "}",
        "",
        "} // namespace crispembed_mgr",
        "",
    ]
    return "\n".join(lines)


def check_sizes(tolerance: float = 0.25, floor_bytes: int = 2_000_000,
                verbose: bool = True) -> int:
    """Compare each registry entry's declared size against the real file.

    The declared size is what a user sees before agreeing to a download, so a
    wrong one is a small honesty problem and, at 250x, a real one. Tolerance is
    generous because these are deliberately round numbers ("19 MB"); it only
    fires on genuine drift, not rounding.
    """
    sizes = registry_sizes(REGISTRY_CPP)
    urls = registry_urls(REGISTRY_CPP)
    trees: dict = {}
    bad, unknown = [], []

    for url, (name, declared, text) in sizes.items():
        parts = urls.get(url)
        if not parts:
            continue
        key = (parts["owner"], parts["repo"], parts["rev"])
        if key not in trees:
            try:
                trees[key] = fetch_tree(*key)
            except (urllib.error.URLError, urllib.error.HTTPError, ValueError):
                trees[key] = {}
        entry = trees[key].get(parts["path"])
        if not entry or not entry[1]:
            unknown.append((name, text))
            continue
        actual = entry[1]
        # Ratio in both directions, so 3.3 MB vs 848 MB is as loud as the reverse.
        # Both a relative and an absolute gate: "976 KB" vs 1.0 MB is a
        # rounding artifact, not a wrong number, and failing CI on it would
        # train people to ignore this check.
        delta = abs(actual - declared)
        if delta > floor_bytes and delta / max(actual, 1) > tolerance:
            bad.append((name, text, actual, declared))

    for name, text, actual, declared in sorted(bad, key=lambda r: -abs(r[2] - r[3])):
        factor = actual / max(declared, 1)
        print(f"  SIZE  {name:34s} declared {text:>9s}  actual {actual/1e6:9.1f} MB  ({factor:.2f}x)",
              file=sys.stderr)
    if verbose and unknown:
        print(f"  ({len(unknown)} entries not size-checked: unreachable or not LFS)", file=sys.stderr)

    if bad:
        print(f"\n{len(bad)} registry size(s) wrong by more than {tolerance:.0%}. "
              f"These are shown to users before they agree to a download.", file=sys.stderr)
        return 1
    print(f"registry sizes: {len(sizes) - len(unknown)} checked, all within "
          f"{tolerance:.0%} (or under the {floor_bytes/1e6:.0f} MB absolute floor)",
          file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit non-zero if the header is stale or coverage regressed")
    ap.add_argument("--json", action="store_true", help="dump resolved pins as JSON")
    ap.add_argument("--check-sizes", action="store_true",
                    help="verify each entry's declared size against the real file")
    args = ap.parse_args()

    if args.check_sizes:
        return check_sizes()

    urls = registry_urls(REGISTRY_CPP)
    print(f"registry: {len(urls)} distinct download URLs", file=sys.stderr)

    pins, unpinned = resolve(urls, verbose=not args.json)

    if args.json:
        json.dump({"pinned": pins, "unpinned": [u for u, _ in unpinned]},
                  sys.stdout, indent=2)
        print()
        return 0

    print(f"pinned: {len(pins)}   unpinned: {len(unpinned)}", file=sys.stderr)

    header = render_header(pins, unpinned)

    if args.check:
        current = HEADER_OUT.read_text(encoding="utf-8") if HEADER_OUT.exists() else ""
        if current != header:
            print("model_hashes.h is stale — re-run tools/fetch_model_hashes.py",
                  file=sys.stderr)
            return 1
        print("model_hashes.h is current", file=sys.stderr)
        return 0

    HEADER_OUT.write_text(header, encoding="utf-8")
    print(f"wrote {HEADER_OUT.relative_to(REPO_ROOT)}", file=sys.stderr)
    for url, why in unpinned:
        print(f"  UNPINNED {url} — {why}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
