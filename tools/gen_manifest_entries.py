#!/usr/bin/env python3
"""
Generate tests/regression/manifest.json smoke+bench entries from the CLI model
registry (examples/cli/model_mgr.cpp), so the regression + benchmark suites can
cover ~all shippable backends without hand-writing ~130 entries.

Each registry row is {name, file, url, description, size, license, src}. We
classify the modality from the description + name + section comment, then emit a
`smoke` entry with the correct `crispembed` CLI invocation for that modality and
a conservative `expect` check (embedding shape for encoders, else exit-0 +
non-empty stdout — "it loads and produces output"). Image modalities reference
the committed tests/regression/sample.bmp.

Invocations the CLI does NOT expose standalone (LID + punctuation are only
reachable inside --ocr-pipeline; LiLT needs a bespoke JSON input; text
detectors need a recognizer) are emitted as tier "skip" with a reason instead
of a guessed (and falsely-failing) command.

Curated entries already in the manifest (the diff-tier ones, and any with
"curated": true) are preserved; generated entries are added only for names not
already present.

Usage:
    python tools/gen_manifest_entries.py            # rewrite manifest (pin SHAs)
    python tools/gen_manifest_entries.py --no-pin   # skip HF SHA lookup (use "main")
    python tools/gen_manifest_entries.py --print     # print summary, don't write
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
REGISTRY = REPO / "examples" / "cli" / "model_mgr.cpp"
MANIFEST = REPO / "tests" / "regression" / "manifest.json"
SAMPLE = "tests/regression/sample.bmp"

ENTRY_RE = re.compile(
    r'\{"([a-z0-9.\-]+)",\s*"([^"]+\.gguf)",\s*"(https://[^"]+)",\s*"([^"]*)"', re.S)


def parse_registry() -> list[dict]:
    txt = REGISTRY.read_text()
    out = []
    for name, file, url, desc in ENTRY_RE.findall(txt):
        m = re.search(r"cstr/[^/]+", url)
        out.append({"name": name, "file": file,
                    "repo": m.group(0) if m else None, "desc": desc})
    return out


# ── modality classification ──────────────────────────────────────────────────
# Explicit name → modality overrides where the description is ambiguous.
OVERRIDE = {
    "instructir": "restore", "adair-5d": "restore",
    "pix2struct-base": "pix2struct", "tps-loc": "skip",
    "scunet": "denoise", "nafnet-sidd": "denoise",
}


def classify(name: str, desc: str) -> str:
    n, d = name.lower(), desc.lower()
    if name in OVERRIDE:
        return OVERRIDE[name]
    if "rerank" in d or "cross-encoder" in d or "reranker" in n:
        return "reranker"
    if "colbert" in d or "colbert" in n:
        return "colbert"
    if "sparse" in d or "splade" in n:
        return "sparse"
    if "lid" in n or "glotlid" in n or "cld3" in n or "fasttext" in n:
        return "lid"          # no standalone CLI → skip
    if "punc" in n or "punct" in d or "fullstop" in n:
        return "punct"        # no standalone CLI → skip
    if "ner" in n or "gliner" in n:
        return "ner"
    if "lilt" in n:
        return "lilt"         # bespoke JSON input → skip
    if "layout" in n:
        return "layout"
    if any(k in n for k in ("scrfd", "yunet")):
        return "detect-face"
    if any(k in n for k in ("auraface", "sface")) or "arcface" in d:
        return "face"
    if any(k in n for k in ("dbnet", "surya-det")):
        return "detect-text"  # needs a recognizer → skip standalone
    if "clip-text" in n or "siglip-text" in n:
        return "embedding"    # text encoder
    if any(k in n for k in ("clip", "siglip")):
        return "vision-embed"
    for eng in ("pan", "hat", "dat", "safmn", "esrgan", "swinir", "tbsrn"):
        if n.startswith(eng) or f"-{eng}" in n or n == eng:
            return f"sr:{eng}"
    if "restormer" in n:
        return "restore"
    if any(k in n for k in ("nafnet", "scunet")) or "denoise" in d:
        return "denoise"
    if any(k in n for k in ("ocr", "docling", "got", "glm-ocr", "qwen2vl",
                            "qwen3vl", "internvl2", "nanonets", "qari",
                            "firered-ocr", "lighton", "deepseek-ocr",
                            "trocr", "parseq", "tesseract", "h2ovl",
                            "german-ocr", "paddleocr")):
        return "ocr"
    if any(k in n for k in ("pix2tex", "texteller", "posformer", "hmer",
                            "bttr", "ppformulanet", "mixtex", "texo", "mfr")):
        return "math-ocr"
    if "pix2struct" in n:
        return "pix2struct"
    # default: a text embedding model (BERT/XLM-R/Qwen3 Nd encoders)
    return "embedding"


def dim_from_desc(desc: str) -> int | None:
    m = re.search(r"(\d{3,4})\s*d\b", desc.lower())
    return int(m.group(1)) if m else None


# modality → (tier, args-template, expect, skip-reason)
# args use {model} (the GGUF) and {sample} (the committed BMP). All include an
# explicit model flag (no auto -m rewrite).
def build_entry(e: dict, sha: str) -> dict:
    name, mod = e["name"], classify(e["name"], e["desc"])
    base = {"name": name, "engine_id": name.replace("-", "_"),
            "modality": mod,
            "gguf": {"repo": e["repo"], "revision": sha, "file": e["file"]}}

    SKIP = {
        "lid": "no standalone LID CLI (only via --ocr-pipeline --lid-model)",
        "punct": "no standalone punctuation CLI (only via --ocr-pipeline --punct-model)",
        "lilt": "needs a bespoke LiLT JSON input, not a plain image",
        "detect-text": "text detector needs a paired recognizer (--ocr-det/--ocr-rec)",
        "skip": "utility/sub-model, no standalone smoke",
    }
    if mod in SKIP:
        base.update(tier="skip", skip_reason=SKIP[mod])
        return base

    txt = "The quick brown fox jumps over the lazy dog."
    if mod == "embedding":
        dim = dim_from_desc(e["desc"])
        exp = {"kind": "embedding"}
        if dim:
            exp["dim"] = dim
        smoke = {"args": ["-m", "{model}", "--json", txt], "expect": exp}
    elif mod == "reranker":
        smoke = {"args": ["-m", "{model}", "--rerank", "what is a fox?",
                          "a fox is an animal", "the sky is blue"],
                 "expect": {"kind": "nonempty"}}
    elif mod == "sparse":
        smoke = {"args": ["-m", "{model}", "--sparse", txt],
                 "expect": {"kind": "nonempty"}}
    elif mod == "colbert":
        smoke = {"args": ["-m", "{model}", "--colbert", txt],
                 "expect": {"kind": "nonempty"}}
    elif mod == "ner":
        smoke = {"args": ["-m", "{model}", "--ner",
                          "Barack Obama visited Paris last week."],
                 "expect": {"kind": "nonempty"}}
    elif mod == "vision-embed":
        smoke = {"args": ["-m", "{model}", "--image", "{sample}", "--json"],
                 "expect": {"kind": "embedding"}}
    elif mod in ("ocr", "math-ocr"):
        smoke = {"args": ["-m", "{model}", "--ocr", "{sample}"],
                 "expect": {"kind": "nonempty"}}
    elif mod == "pix2struct":
        smoke = {"args": ["-m", "{model}", "--pix2struct", "{sample}"],
                 "expect": {"kind": "nonempty"}}
    elif mod == "layout":
        smoke = {"args": ["-m", "{model}", "--layout", "{sample}"],
                 "expect": {"kind": "nonempty"}}
    elif mod == "face":
        smoke = {"args": ["-m", "{model}", "--face", "{sample}"],
                 "expect": {"kind": "nonempty"}}
    elif mod == "detect-face":
        smoke = {"args": ["-m", "{model}", "--detect", "{sample}"],
                 "expect": {"kind": "nonempty"}}
    elif mod in ("denoise", "restore"):
        flag = {"scunet": "--scunet-denoise", "nafnet-sidd": "--scunet-denoise",
                "restormer-denoise": "--restormer", "instructir": "--instructir",
                "adair-5d": "--adair"}.get(name, "--restormer")
        smoke = {"args": ["-m", "{model}", flag, "{sample}"],
                 "expect": {"kind": "nonempty"}}
    elif mod.startswith("sr:"):
        eng = mod.split(":")[1]
        smoke = {"args": [f"--{eng}-sr", "{sample}", f"--{eng}-model", "{model}"],
                 "expect": {"kind": "nonempty"}}
    else:
        base.update(tier="skip", skip_reason=f"unhandled modality {mod}")
        return base

    base.update(tier="smoke", smoke=smoke,
                bench={"args": smoke["args"]})  # benchmark reuses the smoke argv
    return base


# ── SHA pinning ──────────────────────────────────────────────────────────────
def hf_token() -> str | None:
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    p = Path.home() / ".cache" / "huggingface" / "token"
    return p.read_text().strip() if p.exists() else None


def pin_shas(repos: set[str], tok: str | None) -> dict[str, str]:
    out = {}
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    for r in sorted(repos):
        try:
            req = urllib.request.Request(
                f"https://huggingface.co/api/models/{r}", headers=headers)
            d = json.load(urllib.request.urlopen(req, timeout=20))
            out[r] = d.get("sha", "main")
        except Exception as exc:
            print(f"  pin {r}: {getattr(exc,'code',exc)} -> main", file=sys.stderr)
            out[r] = "main"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--print", dest="printonly", action="store_true")
    args = ap.parse_args()

    reg = [e for e in parse_registry() if e["repo"]]
    manifest = json.loads(MANIFEST.read_text())
    # Curated = hand-written entries (no "modality" key). Previously-generated
    # entries carry "modality" and are rebuilt, so re-running is idempotent.
    curated_entries = [b for b in manifest["backends"] if "modality" not in b]
    curated = {b["name"] for b in curated_entries}

    shas = {}
    if not args.no_pin:
        repos = {e["repo"] for e in reg if e["name"] not in curated}
        print(f"pinning SHAs for {len(repos)} repos…", file=sys.stderr)
        shas = pin_shas(repos, hf_token())

    from collections import Counter
    cnt = Counter()
    new = []
    for e in reg:
        if e["name"] in curated:
            continue
        sha = shas.get(e["repo"], "main")
        entry = build_entry(e, sha)
        cnt[entry.get("tier", "?") + ":" + entry.get("modality", "?")] += 1
        new.append(entry)

    print(f"registry: {len(reg)} models, {len(curated)} curated kept, "
          f"{len(new)} generated", file=sys.stderr)
    for k in sorted(cnt):
        print(f"  {k:24s} {cnt[k]}", file=sys.stderr)
    n_smoke = sum(1 for e in new if e.get("tier") == "smoke")
    n_skip = sum(1 for e in new if e.get("tier") == "skip")
    print(f"  => {n_smoke} smoke, {n_skip} skip", file=sys.stderr)

    if args.printonly:
        return 0

    manifest["backends"] = curated_entries + new
    manifest["_generated_note"] = (
        "Entries below the curated set are generated by "
        "tools/gen_manifest_entries.py from examples/cli/model_mgr.cpp. "
        "smoke = best-effort CLI invocation + exit-0/non-empty (or embedding) "
        "check; the Kaggle sweep empirically validates which invocations are "
        "correct. tier=skip entries have no standalone CLI mode.")
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {MANIFEST} ({len(manifest['backends'])} total entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
