#!/usr/bin/env python3
"""Fetch the declared license for every model in CrispEmbed's auto-download
registry, cross-check against both the cstr/*-GGUF re-host and the upstream
base model on huggingface.co, and flag NC / mismatches.

Sources:

  1. examples/cli/model_mgr.cpp   — k_registry struct (download URLs, the
                                    re-host repo we ship to users)
  2. models/upload_to_hf.py       — MODELS dict (declared `license` +
                                    upstream `base_model`)

For each entry we report three license tags side-by-side:

    declared    — what upload_to_hf.py claims when it pushes the README
                  card to cstr/<name>-GGUF
    rehost      — what cstr/<name>-GGUF actually shows on HF today
    upstream    — what the upstream `base_model` repo shows on HF today

Plus a classification (PERMISSIVE / NC / GEMMA / OTHER / UNKNOWN) derived
from the upstream tag — which is the legally authoritative one, since the
re-host is a derivative work and inherits upstream's restrictions.

Exit code: number of rows where classification == NC/OTHER and declared !=
upstream (i.e. legal-exposure rows that need fixing).

Usage:
    HF_HOME=/Volumes/backups/ai/huggingface-hub \\
        python tests/check_registry_licenses.py
    python tests/check_registry_licenses.py --json
    python tests/check_registry_licenses.py --emit-cpp   # paste-ready
                                                         # SPDX strings for
                                                         # ModelEntry
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────────────
# Source parsers
# ────────────────────────────────────────────────────────────────────────

def parse_cpp_registry(path: Path) -> List[Tuple[str, str, str]]:
    """Return [(name, gguf_url, gguf_filename), ...]."""
    text = path.read_text()
    pattern = re.compile(
        r'\{\s*"([^"]+)"\s*,\s*"([^"]+\.gguf)"\s*,\s*"(https?://[^"]+)"',
        re.MULTILINE,
    )
    return [(m.group(1), m.group(3), m.group(2)) for m in pattern.finditer(text)]


def parse_upload_to_hf(path: Path) -> Dict[str, Dict[str, str]]:
    """Return {name: {base_model, license}} from upload_to_hf.py's MODELS
    dict. Uses AST so nested braces / lists inside per-entry dicts don't
    confuse the parser.
    """
    import ast
    tree = ast.parse(path.read_text())
    out: Dict[str, Dict[str, str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not (len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "MODELS"):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        for k_node, v_node in zip(node.value.keys, node.value.values):
            if not (isinstance(k_node, ast.Constant)
                    and isinstance(k_node.value, str)):
                continue
            name = k_node.value
            if not isinstance(v_node, ast.Dict):
                continue
            fields: Dict[str, str] = {}
            for fk, fv in zip(v_node.keys, v_node.values):
                if not (isinstance(fk, ast.Constant)
                        and isinstance(fk.value, str)):
                    continue
                if isinstance(fv, ast.Constant) and isinstance(fv.value, str):
                    fields[fk.value] = fv.value
            if "base_model" in fields:
                out[name] = {
                    "base_model": fields["base_model"],
                    "license": fields.get("license", ""),
                }
        break
    return out


def derive_rehost_repo(url: str) -> Optional[str]:
    """Extract `cstr/foo-GGUF` from a `https://huggingface.co/<repo>/resolve/...` URL."""
    m = re.match(r'https?://huggingface\.co/([^/]+/[^/]+)/resolve/', url)
    return m.group(1) if m else None


# ────────────────────────────────────────────────────────────────────────
# License classification
# ────────────────────────────────────────────────────────────────────────

# Set membership uses normalized lower-case license tag.
PERMISSIVE = {
    "apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause",
    "bsd-3-clause-clear", "cc0-1.0", "cc-by-4.0", "cc-by-3.0",
    "cdla-permissive-1.0", "cdla-permissive-2.0", "cdla-sharing-1.0",
    "openrail", "openrail++", "bigscience-openrail-m",
    "bigscience-bloom-rail-1.0", "creativeml-openrail-m",
    "unlicense", "wtfpl", "isc", "lgpl-3.0", "agpl-3.0", "gpl-3.0",
    "mpl-2.0", "epl-2.0", "artistic-2.0",
}

# CC BY-NC and friends, or anything ending in `-nc` / `-nc-*`.
NC_PATTERNS = (
    "cc-by-nc-",  # cc-by-nc-4.0, cc-by-nc-sa-4.0, cc-by-nc-nd-4.0
    "-nc-",
    "-noncommercial",
)
NC_EXACT = {"cc-by-nc-2.0", "cc-by-nc-3.0", "cc-by-nc-4.0"}

# Gated / model-specific licenses that allow commercial use under their own
# acceptable-use policy. Treated as RESTRICTED — distinct from pure NC, but
# still requires the user to accept upstream terms before redistribution.
GEMMA_LIKE = {
    "gemma", "llama2", "llama3", "llama3.1", "llama3.2", "llama3.3",
    "llama4", "mistral-ai-research", "qwen", "qwen-research", "deepseek",
    "intel-research-use-license",
}


def classify(license_tag: str) -> str:
    if not license_tag:
        return "UNKNOWN"
    tag = license_tag.strip().lower()
    if tag in NC_EXACT or any(p in tag for p in NC_PATTERNS):
        return "NC"
    if tag in GEMMA_LIKE:
        return "GEMMA"
    if tag in PERMISSIVE:
        return "PERMISSIVE"
    if tag == "other":
        return "OTHER"
    # Unknown SPDX-like tag → conservative: mark as OTHER so the user looks
    # at the model card manually instead of assuming permissive.
    return "OTHER"


# ────────────────────────────────────────────────────────────────────────
# HuggingFace fetch
# ────────────────────────────────────────────────────────────────────────

def fetch_hf_license(api, repo_id: str) -> Tuple[str, str]:
    """Return (license_tag, status). license_tag is "" if the repo exists
    but has no `license:` in its card metadata."""
    from huggingface_hub.errors import (
        RepositoryNotFoundError, GatedRepoError, HfHubHTTPError,
    )
    try:
        info = api.model_info(repo_id)
    except GatedRepoError:
        return "", "GATED"
    except RepositoryNotFoundError:
        return "", "NOT_FOUND"
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", "?")
        return "", f"HTTP_{code}"
    except Exception as e:
        return "", f"ERROR_{type(e).__name__}"

    card = getattr(info, "card_data", None)
    if card is None:
        return "", "OK_NO_CARD"

    # card_data.license can be a string ("apache-2.0") or, rarely, a list
    # ("['apache-2.0', 'cc-by-nc-4.0']") for dual-licensed models.
    lic = getattr(card, "license", None)
    if isinstance(lic, list):
        lic = ",".join(str(x) for x in lic)
    return (lic or ""), "OK"


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true",
                    help="emit machine-readable rows")
    ap.add_argument("--emit-cpp", action="store_true",
                    help="print SPDX tags as a paste-ready C++ block "
                         "for ModelEntry.license")
    ap.add_argument("--skip-rehost", action="store_true",
                    help="don't fetch cstr/*-GGUF re-host licenses (faster, "
                         "only checks upstream)")
    args = ap.parse_args()

    from huggingface_hub import HfApi, get_token
    api = HfApi()
    if not get_token():
        print("WARNING: no HF token — some private/gated repos will show "
              "GATED instead of their real license. Run "
              "`huggingface-cli login` if you need them resolved.",
              file=sys.stderr)

    cpp_entries = parse_cpp_registry(REPO / "examples/cli/model_mgr.cpp")
    upload_map = parse_upload_to_hf(REPO / "models/upload_to_hf.py")

    rows: List[Dict[str, str]] = []
    for name, url, filename in cpp_entries:
        rehost = derive_rehost_repo(url) or ""
        # upload_to_hf.py keys don't always match the cpp registry name (e.g.
        # cpp "granite-embedding-278m" → upload "granite-embedding-278m-
        # multilingual"). Fall back to filename stem, then to the rehost
        # repo's basename minus the -GGUF suffix.
        upload = upload_map.get(name)
        if upload is None and filename:
            stem = filename.rsplit(".gguf", 1)[0]
            # strip common quant suffixes
            for suffix in ("-q4_k", "-q5_k", "-q6_k", "-q8_0", "-f16"):
                if stem.endswith(suffix):
                    stem = stem[:-len(suffix)]
            upload = upload_map.get(stem)
        if upload is None and rehost.startswith("cstr/"):
            rstem = rehost[len("cstr/"):]
            if rstem.endswith("-GGUF"):
                rstem = rstem[:-len("-GGUF")]
            upload = upload_map.get(rstem)
        upload = upload or {}
        upstream = upload.get("base_model", "")
        declared = upload.get("license", "")

        rehost_lic, rehost_status = ("", "SKIPPED")
        if rehost and not args.skip_rehost:
            rehost_lic, rehost_status = fetch_hf_license(api, rehost)

        upstream_lic, upstream_status = ("", "")
        if upstream:
            upstream_lic, upstream_status = fetch_hf_license(api, upstream)

        cls = classify(upstream_lic) if upstream_lic else classify(declared)

        rows.append({
            "name": name,
            "filename": filename,
            "rehost": rehost,
            "upstream": upstream,
            "declared": declared,
            "rehost_lic": rehost_lic,
            "rehost_status": rehost_status,
            "upstream_lic": upstream_lic,
            "upstream_status": upstream_status,
            "class": cls,
        })

    # ── JSON output ────────────────────────────────────────────────────
    if args.json:
        print(json.dumps(rows, indent=2))
        return _exit_code(rows)

    # ── Paste-ready C++ snippet ────────────────────────────────────────
    if args.emit_cpp:
        print("// Paste into the ModelEntry initializers in")
        print("// examples/cli/model_mgr.cpp. Verify each tag against the")
        print("// upstream model card — `OTHER` and `UNKNOWN` need a human.")
        print()
        max_name = max(len(r["name"]) for r in rows)
        for r in rows:
            lic = r["upstream_lic"] or r["declared"] or "UNKNOWN"
            card_url = (
                f"https://huggingface.co/{r['upstream']}"
                if r["upstream"] else ""
            )
            print(f'    // {r["name"]:<{max_name}}  '
                  f'class={r["class"]:<10}  upstream={r["upstream"]}')
            print(f'    //   license: "{lic}"   model_card_url: "{card_url}"')
        return _exit_code(rows)

    # ── Text table ─────────────────────────────────────────────────────
    print(f'{"":<2} {"name":<40s}  {"class":<10}  '
          f'{"declared":<18}  {"rehost":<18}  {"upstream":<18}  notes')
    print("─" * 140)
    for r in rows:
        badge, note = _badge_and_note(r)
        decl = r["declared"] or "(none)"
        reh = r["rehost_lic"] or f"({r['rehost_status']})"
        ups = r["upstream_lic"] or f"({r['upstream_status']})"
        print(f'{badge:<2} {r["name"]:<40s}  {r["class"]:<10}  '
              f'{decl:<18}  {reh:<18}  {ups:<18}  {note}')

    # ── Summary ────────────────────────────────────────────────────────
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print("")
    print("  classification: " + "  ".join(
        f"{k}={v}" for k, v in sorted(counts.items())))

    dangerous = [r for r in rows
                 if _rehost_more_permissive_than_upstream(r)]
    if dangerous:
        print("")
        print(f"  ⚠  {len(dangerous)} LEGAL-EXPOSURE entries — the cstr/*-"
              "GGUF re-host declares a more permissive license than "
              "the upstream model. Re-push the README card with the correct "
              "tag IMMEDIATELY:")
        for r in dangerous:
            print(f"    - {r['name']:<40s}  "
                  f"rehost={r['rehost_lic']:<14s}  "
                  f"upstream={r['upstream_lic']}")

    rehost_misses = [r for r in rows
                     if not _rehost_more_permissive_than_upstream(r)
                     and _rehost_mismatch(r)]
    if rehost_misses:
        print("")
        print(f"  {len(rehost_misses)} rehost-license mismatch(es) (not "
              "necessarily dangerous, but should match upstream):")
        for r in rehost_misses:
            print(f"    - {r['name']:<40s}  "
                  f"rehost={r['rehost_lic']:<14s}  "
                  f"upstream={r['upstream_lic']}")

    nc_or_other = [r for r in rows if r["class"] in ("NC", "OTHER", "GEMMA")]
    if nc_or_other:
        print("")
        print(f"  {len(nc_or_other)} restricted-license entries — these "
              "must carry the correct SPDX tag in ModelEntry AND in the "
              "cstr/*-GGUF README, and must trigger the --accept-license "
              "gate before auto-download:")
        for r in nc_or_other:
            ups = r["upstream_lic"] or "?"
            print(f"    - {r['name']:<40s}  {r['class']:<8}  {ups}")

    declared_misses = [r for r in rows if _declared_mismatch(r)]
    if declared_misses:
        print("")
        print(f"  {len(declared_misses)} upload_to_hf.py license claim(s) "
              "differ from upstream — fix the MODELS dict and re-push:")
        for r in declared_misses:
            print(f"    - {r['name']:<40s}  "
                  f"declared={r['declared'] or '(none)'}  "
                  f"upstream={r['upstream_lic'] or '?'}")

    return _exit_code(rows)


def _badge_and_note(r: Dict[str, str]) -> Tuple[str, str]:
    """Return (badge, note) for table display.

    Priority of warnings (worst first):
      1. rehost claims MORE permissive license than upstream (legal exposure)
      2. rehost != upstream in any direction (misrepresentation)
      3. declared (upload_to_hf.py) != upstream (re-push needed to fix)
      4. NC / Gemma classification (correct but needs gating)
      5. unknown / needs manual review
    """
    if _rehost_more_permissive_than_upstream(r):
        return "✗", "REHOST CLAIMS PERMISSIVE OVER NC UPSTREAM"
    if _rehost_mismatch(r):
        return "✗", "rehost != upstream"
    if r["upstream_status"] not in ("OK", "OK_NO_CARD", ""):
        return "?", f"upstream {r['upstream_status']}"
    if _declared_mismatch(r):
        return "✗", "declared != upstream (re-push card)"
    if r["class"] == "NC":
        return "·", "non-commercial"
    if r["class"] == "GEMMA":
        return "·", "gated terms"
    if r["class"] == "OTHER":
        return "?", "needs manual review"
    if r["class"] == "UNKNOWN":
        return "?", "no license tag"
    return "✓", ""


def _declared_mismatch(r: Dict[str, str]) -> bool:
    decl = (r["declared"] or "").strip().lower()
    ups = (r["upstream_lic"] or "").strip().lower()
    return bool(decl and ups and decl != ups)


def _rehost_mismatch(r: Dict[str, str]) -> bool:
    reh = (r["rehost_lic"] or "").strip().lower()
    ups = (r["upstream_lic"] or "").strip().lower()
    return bool(reh and ups and reh != ups)


def _rehost_more_permissive_than_upstream(r: Dict[str, str]) -> bool:
    """The legally dangerous direction: rehost says apache-2.0 but upstream
    is cc-by-nc-4.0. Means we're (inadvertently) granting users rights we
    don't have."""
    return (classify(r["rehost_lic"]) == "PERMISSIVE"
            and classify(r["upstream_lic"]) in ("NC", "GEMMA", "OTHER"))


def _exit_code(rows: List[Dict[str, str]]) -> int:
    """Nonzero on any of:
      - rehost claims more permissive license than upstream (legal exposure)
      - rehost != upstream
      - declared != upstream for restricted-license rows
    """
    bad = 0
    for r in rows:
        if _rehost_more_permissive_than_upstream(r):
            bad += 1
            continue
        if _rehost_mismatch(r):
            bad += 1
            continue
        if r["class"] in ("NC", "OTHER", "GEMMA") and _declared_mismatch(r):
            bad += 1
    return bad


if __name__ == "__main__":
    sys.exit(main())
