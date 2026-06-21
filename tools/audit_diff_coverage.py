#!/usr/bin/env python3
"""
Audit `crispembed-diff` coverage across all engines that have (or could have)
a ground-truth diff harness.

CrispEmbed has no single `REGISTERED_BACKENDS` dict like CrispASR — its diff
coverage is spread across three on-disk artifacts per engine, so this tool
discovers and crosses them:

  - a C++ diff harness   `tests/test_<engine>_diff.cpp`     (+ a CMake
    `test-<engine>-diff` executable target that actually builds it)
  - a Python dumper      `tools/dump_<engine>_reference*.py`  that produces the
    frozen `<engine>-ref.gguf` (a `*_from_gguf` variant bakes a self-consistent
    ref from the GGUF alone — no source-model download)
  - a frozen ref archive `<engine>-ref.gguf` uploaded to that model's HF GGUF
    repo (conventionally under `diff-harness-ref/`), tracked in
    `tests/regression/manifest.json`

With `--probe-hf` it additionally queries the HF API (token from
`~/.cache/huggingface/token` or `$HF_TOKEN`) to see which of the model repos
actually carries a `*-ref.gguf` right now. Offline (default), it reports ref
presence from the manifest + the built-in KNOWN_REFS map.

Output: a markdown table at `docs/diff-harness-coverage.md` plus a stdout
summary. An engine is **fully wired** when it has a harness + a CMake target +
a reachable ref. The gap lists ("ref but no harness", "harness but no ref",
"neither") drive the PLAN.md todos.

Usage:
    python tools/audit_diff_coverage.py                  # write doc + print
    python tools/audit_diff_coverage.py --print-only      # don't write the doc
    python tools/audit_diff_coverage.py --probe-hf        # live HF ref check
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TESTS = REPO / "tests"
MANIFEST_PATH = TESTS / "regression" / "manifest.json"
CMAKE = REPO / "CMakeLists.txt"

# Engine names use underscores in tests/tools (glm_ocr) but hyphens in CMake
# targets and HF slugs (glm-ocr). Canonicalise on the underscore form.
def canon(name: str) -> str:
    return name.replace("-", "_")


def display(name: str) -> str:
    return canon(name).replace("_", "-")


# Built-in record of model GGUF repos that already carry a `*-ref.gguf`,
# captured by a token-authed HF probe (2026-06-21). The manifest is the
# source of truth once populated; this map keeps the audit useful before
# every entry is pinned and lets `--probe-hf` know which repos to inspect.
# engine (canonical) -> (hf_repo, [ref filenames present])
KNOWN_REFS: dict[str, tuple[str, list[str]]] = {
    "instructir":     ("cstr/InstructIR-GGUF", ["instructir-ref.gguf"]),
    "granite_vision": ("cstr/granite-vision-crispembed-GGUF",
                       ["granite-vision-ref.gguf", "granite-llm-ref.gguf"]),
    "hat":            ("cstr/text-super-resolution-gguf", ["hat-ref.gguf"]),
    "pan":            ("cstr/text-super-resolution-gguf", ["pan-ref.gguf"]),
    # Refs uploaded but with NO C++ harness yet (tracked as a gap):
    "firered_ocr":    ("cstr/firered-ocr-crispembed-GGUF", ["firered-ocr-ref.gguf"]),
    "qari_ocr":       ("cstr/qari-ocr-crispembed-GGUF",
                       ["diff-harness-ref/qari-ocr-ref.gguf", "qari-ocr-ref.gguf"]),
    "qwen3vl":        ("cstr/qwen3-vl-2b-crispembed-gguf", ["qwen3-vl-2b-diff-ref.gguf"]),
}


def discover_harnesses() -> set[str]:
    """Engines with a tests/test_<engine>_diff.cpp file."""
    out = set()
    for p in TESTS.glob("test_*_diff.cpp"):
        out.add(canon(p.name[len("test_"):-len("_diff.cpp")]))
    return out


def discover_cmake_targets() -> set[str]:
    """Engines wired as a `test-<engine>-diff` add_executable target."""
    if not CMAKE.exists():
        return set()
    text = CMAKE.read_text()
    return {canon(m) for m in re.findall(r"add_executable\(test-([a-z0-9-]+)-diff", text)}


def discover_dumpers() -> dict[str, dict]:
    """Engines with a tools/dump_<engine>_reference*.py dumper.

    Returns {engine: {"path": rel, "from_gguf": bool}}. A `*_from_gguf`
    dumper bakes a self-consistent ref from the GGUF alone (no source model).
    """
    out: dict[str, dict] = {}
    for p in HERE.glob("dump_*_reference*.py"):
        stem = p.name[len("dump_"):]
        from_gguf = stem.endswith("_reference_from_gguf.py")
        eng = stem[:-len("_reference_from_gguf.py")] if from_gguf \
            else stem[:-len("_reference.py")]
        eng = canon(eng)
        # Prefer recording the from_gguf flag if either variant exists.
        rec = out.setdefault(eng, {"path": f"tools/{p.name}", "from_gguf": False})
        if from_gguf:
            rec["from_gguf"] = True
            rec["path"] = f"tools/{p.name}"
    return out


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        return json.loads(MANIFEST_PATH.read_text())
    except Exception as exc:  # pragma: no cover
        print(f"WARN: could not parse {MANIFEST_PATH}: {exc}", file=sys.stderr)
        return {}


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    cand = Path.home() / ".cache" / "huggingface" / "token"
    if cand.exists():
        try:
            return cand.read_text().strip() or None
        except Exception:
            return None
    return None


def probe_hf_refs(repos: set[str], tok: str | None) -> dict[str, list[str]]:
    """For each repo, list its files and return the `*-ref.gguf` ones.

    repo -> [ref filenames] (empty list = repo reachable, no ref; absent key
    = repo unreachable / 404).
    """
    import urllib.request

    out: dict[str, list[str]] = {}
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    for repo in sorted(repos):
        url = f"https://huggingface.co/api/models/{repo}?full=true"
        try:
            req = urllib.request.Request(url, headers=headers)
            data = json.load(urllib.request.urlopen(req, timeout=25))
        except Exception as exc:
            print(f"  probe {repo}: ERR {getattr(exc, 'code', exc)}", file=sys.stderr)
            continue
        files = [s["rfilename"] for s in data.get("siblings", [])]
        refs = [f for f in files if "ref" in f.lower() and f.endswith(".gguf")]
        out[repo] = refs
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO / "docs" / "diff-harness-coverage.md")
    p.add_argument("--print-only", action="store_true")
    p.add_argument("--probe-hf", action="store_true",
                   help="query the HF API for the actual ref.gguf presence")
    args = p.parse_args()

    harnesses = discover_harnesses()
    cmake_targets = discover_cmake_targets()
    dumpers = discover_dumpers()
    manifest = load_manifest()
    today = dt.date.today().isoformat()

    # Map engine -> manifest entry (by `engine_id`, canonicalised).
    man_by_engine: dict[str, dict] = {}
    for b in manifest.get("backends", []):
        man_by_engine[canon(b.get("engine_id", b.get("name", "")))] = b

    # Resolve ref presence. Start from KNOWN_REFS + manifest, optionally
    # refresh from a live HF probe.
    ref_repo: dict[str, str] = {}
    ref_files: dict[str, list[str]] = {}
    for eng, (repo, files) in KNOWN_REFS.items():
        ref_repo[eng] = repo
        ref_files[eng] = list(files)
    for eng, entry in man_by_engine.items():
        r = entry.get("ref")
        if r and r.get("repo"):
            ref_repo[eng] = r["repo"]
            ref_files.setdefault(eng, [])
            if r.get("path") and r["path"] not in ref_files[eng]:
                ref_files[eng].append(r["path"])

    if args.probe_hf:
        tok = hf_token()
        if not tok:
            print("WARN: --probe-hf but no HF token found; results may be partial",
                  file=sys.stderr)
        repos = set(ref_repo.values())
        # also probe any gguf repos named in the manifest
        for entry in man_by_engine.values():
            g = entry.get("gguf")
            if g and g.get("repo"):
                repos.add(g["repo"])
        live = probe_hf_refs(repos, tok)
        # rewrite ref_files from live truth for probed repos
        repo_to_engines: dict[str, list[str]] = {}
        for eng, repo in ref_repo.items():
            repo_to_engines.setdefault(repo, []).append(eng)
        for repo, refs in live.items():
            engs = repo_to_engines.get(repo, [])
            if len(engs) == 1:
                # Sole engine for this repo owns every ref it carries.
                ref_files[engs[0]] = list(refs)
            else:
                # Shared repo (e.g. the SR repo holds hat + pan): match each
                # engine's refs by basename token.
                for eng in engs:
                    tok = display(eng).split("-")[0]
                    ref_files[eng] = [f for f in refs if tok in f.lower()]

    engines = sorted(harnesses | set(dumpers) | set(ref_repo) | set(man_by_engine))

    rows = []
    for eng in engines:
        has_harness = eng in harnesses
        has_target = eng in cmake_targets
        d = dumpers.get(eng)
        has_dumper = d is not None
        from_gguf = bool(d and d["from_gguf"])
        refs = ref_files.get(eng, [])
        repo = ref_repo.get(eng, "")
        has_ref = bool(refs)
        tier = man_by_engine.get(eng, {}).get("tier", "")
        fully = has_harness and has_target and has_ref
        rows.append({
            "engine": eng, "harness": has_harness, "target": has_target,
            "dumper": has_dumper, "from_gguf": from_gguf,
            "ref_repo": repo, "refs": refs, "has_ref": has_ref,
            "tier": tier, "fully": fully,
        })

    n_harness = sum(r["harness"] for r in rows)
    n_ref = sum(r["has_ref"] for r in rows)
    n_fully = sum(r["fully"] for r in rows)
    ref_no_harness = [r for r in rows if r["has_ref"] and not r["harness"]]
    harness_no_ref = [r for r in rows if r["harness"] and not r["has_ref"]]

    print(f"crispembed-diff coverage audit ({today}):")
    print(f"  {len(rows)} engines tracked")
    print(f"  {n_harness} have a C++ diff harness")
    print(f"  {n_ref} have a reachable *-ref.gguf")
    print(f"  {n_fully} are FULLY wired (harness + CMake target + ref)")
    print(f"  {len(ref_no_harness)} have a ref but NO harness")
    print(f"  {len(harness_no_ref)} have a harness but NO ref")
    if harness_no_ref:
        quick = [r["engine"] for r in harness_no_ref if r["from_gguf"]]
        if quick:
            print(f"  quick wins (self-consistent *_from_gguf dumper): {', '.join(quick)}")

    if args.print_only:
        return 0

    lines: list[str] = []
    lines.append("# crispembed-diff coverage")
    lines.append("")
    lines.append(f"Auto-generated by `tools/audit_diff_coverage.py` on {today}. "
                 "Do not edit by hand — re-run the tool to refresh "
                 "(`--probe-hf` for a live ref check).")
    lines.append("")
    lines.append(f"- {len(rows)} engines tracked")
    lines.append(f"- {n_harness} have a C++ diff harness (`tests/test_<e>_diff.cpp` + CMake target)")
    lines.append(f"- {n_ref} have a reachable `*-ref.gguf`")
    lines.append(f"- **{n_fully} are fully wired** (harness + target + ref)")
    lines.append("")
    lines.append("Refs live in each model's HF GGUF repo (conventionally under "
                 "`diff-harness-ref/<engine>-ref.gguf`); pins are in "
                 "`tests/regression/manifest.json`. A `*_from_gguf` dumper can "
                 "bake a self-consistent ref from the GGUF alone (no source-model "
                 "download) — the cheapest path to close a harness-but-no-ref gap.")
    lines.append("")
    lines.append("| engine | harness | target | dumper | ref repo | ref files | tier |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        dcell = ("yes (from_gguf)" if r["from_gguf"] else "yes") if r["dumper"] else "—"
        refcell = ", ".join(f"`{x}`" for x in r["refs"]) if r["refs"] else "—"
        repocell = f"`{r['ref_repo']}`" if r["ref_repo"] else "—"
        lines.append(
            f"| `{display(r['engine'])}` | {'yes' if r['harness'] else '**no**'} | "
            f"{'yes' if r['target'] else '**no**'} | {dcell} | {repocell} | "
            f"{refcell} | {r['tier'] or '—'} |")
    lines.append("")

    if ref_no_harness:
        lines.append("## Ref uploaded but no C++ harness")
        lines.append("")
        lines.append("A frozen ref exists on HF but there is no `test-<engine>-diff` "
                     "harness to diff against it. Wire one in `tests/` + `CMakeLists.txt`.")
        lines.append("")
        for r in ref_no_harness:
            lines.append(f"- `{display(r['engine'])}` — ref in `{r['ref_repo']}` "
                         f"({', '.join(r['refs'])})")
        lines.append("")

    if harness_no_ref:
        lines.append("## Harness but no ref archive")
        lines.append("")
        lines.append("Harness builds but has nothing to diff against. Bake + upload a "
                     "`*-ref.gguf` to the model's repo under `diff-harness-ref/`. "
                     "Engines with a `*_from_gguf` dumper are quick wins (no source "
                     "model download).")
        lines.append("")
        for r in harness_no_ref:
            tag = "  _(quick win: `*_from_gguf` dumper)_" if r["from_gguf"] else \
                  ("" if r["dumper"] else "  _(needs a dumper too)_")
            lines.append(f"- `{display(r['engine'])}`{tag}")
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
