#!/usr/bin/env python3
"""
Run one (or all) CrispEmbed regression entries from tests/regression/manifest.json.

Two tiers (see manifest + README):

  * **diff**  — download the pinned GGUF + ref archive, run the engine's
                `test-<engine>-diff <model> <ref>` binary, parse its
                `cos_min=… max_abs=…` lines, and assert each stage clears its
                manifest threshold (`diff_thresholds`, with `"*"` as the
                global default and per-stage overrides; `require_stages` lists
                stages that must be present).
  * **smoke** — download the pinned GGUF, run the `crispembed` CLI, and assert
                the output is sane (a finite, correctly-shaped embedding) or
                matches a captured golden. The "some other confirmation that
                it works" tier for engines without a ref yet.

Importable: the Kaggle regression kernel reuses `parse_diff_stdout`,
`evaluate_diff`, `run_diff`, `run_smoke`, and `hf_download` directly.

Usage:
    python tests/regression/run_one.py --list
    python tests/regression/run_one.py --dry-run
    python tests/regression/run_one.py --backend pan --build-dir build
    python tests/regression/run_one.py --tier diff --build-dir build
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
MANIFEST_PATH = HERE / "manifest.json"

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# A line carrying a comparison metric, e.g.
#   "  output  cos_min=0.999997  max_abs=6.6e-04  PASS"       (pan)
#   "    cos_min=0.998903  max_abs=1.2e-03  PASS"             (granite, name above)
_COS = re.compile(r"cos_min=([0-9.eE+\-]+)")
_MAXABS = re.compile(r"max_abs=([0-9.eE+\-]+)")
# Stage-label lines that precede a metric line without a leading name:
#   "  C++ stage: vision_out (1234 elements)"   (granite dump_cb)
_STAGE_LABEL = re.compile(r"C\+\+ stage:\s*(\S+)")
#   "  [PASS] enc_layer_0"                       (instructir/scunet check())
_VERDICT = re.compile(r"\[(PASS|FAIL)\]\s+(\S+)")
_IDENT = re.compile(r"^[A-Za-z][\w.\-]*$")


def die(msg: str, code: int = 1):
    print(f"\033[31mERROR\033[0m  {msg}", file=sys.stderr)
    sys.exit(code)


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


# ── HF download ──────────────────────────────────────────────────────────────
def hf_download(repo: str, file_in_repo: str, revision: str,
                dest_dir: Path) -> Path:
    """Download one file from HF at a pinned revision; honour HF_HOME if set."""
    from huggingface_hub import hf_hub_download
    import time as _time

    cache_dir = None
    if not (os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")):
        cache_dir = str(dest_dir / "hf_cache")
    print(f"  download  {repo}@{revision[:8]} :: {file_in_repo}", flush=True)
    for attempt in range(5):
        try:
            return Path(hf_hub_download(repo_id=repo, filename=file_in_repo,
                                        revision=revision, cache_dir=cache_dir))
        except Exception as e:
            if "429" in str(e) and attempt < 4:
                wait = 2 ** (attempt + 1) + (hash(repo) % 5)
                print(f"  429 rate limit — retry {attempt+1}/4 in {wait}s", flush=True)
                _time.sleep(wait)
            else:
                raise


# ── diff parsing / evaluation ────────────────────────────────────────────────
def parse_diff_stdout(stdout: str) -> dict[str, dict]:
    """Parse a CrispEmbed diff binary's stdout into {stage: {cos_min, max_abs}}.

    Handles both layouts: a leading stage name on the metric line (pan,
    instructir) and a `C++ stage: <name>` label line preceding a bare metric
    line (granite). Falls back to a positional `stage_<i>` key. Also records
    standalone `[PASS]/[FAIL] <name>` verdicts under `_verdicts` so callers
    can fall back to them when a harness emits no cos_min lines.
    """
    out: dict[str, dict] = {}
    verdicts: dict[str, bool] = {}
    last_label: str | None = None
    idx = 0
    for raw in stdout.splitlines():
        line = _ANSI.sub("", raw)
        ms = _STAGE_LABEL.search(line)
        if ms:
            last_label = ms.group(1)
        mv = _VERDICT.search(line)
        if mv and not _COS.search(line):
            verdicts[mv.group(2)] = (mv.group(1) == "PASS")
            last_label = mv.group(2)
        mc = _COS.search(line)
        if not mc:
            continue
        cos = float(mc.group(1))
        mm = _MAXABS.search(line)
        max_abs = float(mm.group(1)) if mm else None
        # Stage name: leading identifier token on this line, else the verdict
        # name on this line, else the most recent label, else positional.
        head = line[:mc.start()].strip()
        name = None
        if mv:
            name = mv.group(2)
        elif head:
            tok = head.split()[0]
            if _IDENT.match(tok) and tok != "cos_min":
                name = tok
        if not name:
            name = last_label or f"stage_{idx}"
        out[name] = {"cos_min": cos, "max_abs": max_abs}
        idx += 1
    if verdicts:
        out["_verdicts"] = verdicts
    return out


def evaluate_diff(stages: dict[str, dict], thresholds: dict[str, float],
                  require_stages: list[str] | None = None):
    """Apply thresholds. `thresholds["*"]` is the global default applied to
    every parsed stage; explicit keys override it. Returns
    (passes, fails, missing) where each pass/fail is (stage, cos_min, thr)."""
    require_stages = require_stages or []
    glob = thresholds.get("*")
    real_stages = {k: v for k, v in stages.items() if k != "_verdicts"}
    passes, fails = [], []
    for stage, metrics in real_stages.items():
        thr = thresholds.get(stage, glob)
        if thr is None:
            continue  # no global, no explicit → not graded
        cos = metrics["cos_min"]
        (passes if cos >= thr else fails).append((stage, cos, thr))
    missing = [s for s in require_stages if s not in real_stages]
    missing += [s for s in thresholds if s != "*" and s not in real_stages]
    return passes, fails, sorted(set(missing))


def run_diff(diff_bin: Path, argv_tmpl: str, model: Path, ref: Path) -> dict:
    """Run a diff binary and return parsed stages. argv_tmpl uses {model}/{ref}."""
    argv = argv_tmpl.format(model=str(model), ref=str(ref)).split()
    proc = subprocess.run([str(diff_bin), *argv], capture_output=True,
                          text=True, check=False, timeout=900)
    if proc.returncode < 0:
        die(f"{diff_bin.name} died from signal {-proc.returncode}\n"
            f"  stderr tail: {proc.stderr[-400:]}")
    stages = parse_diff_stdout(proc.stdout)
    real = {k: v for k, v in stages.items() if k != "_verdicts"}
    if not real and "_verdicts" not in stages:
        die(f"{diff_bin.name} produced no parseable stage lines.\n"
            f"  stdout tail: {proc.stdout[-600:]}\n"
            f"  stderr tail: {proc.stderr[-400:]}")
    return stages


# ── smoke parsing ────────────────────────────────────────────────────────────
def parse_embedding(stdout: str) -> list[float] | None:
    """Extract the first embedding vector from `crispembed --json` output:
    `[{"text": ..., "embedding": [ ... ]}]`."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        emb = data[0].get("embedding")
        if isinstance(emb, list):
            return [float(x) for x in emb]
    if isinstance(data, dict) and isinstance(data.get("embedding"), list):
        return [float(x) for x in data["embedding"]]
    return None


def run_smoke(cli_bin: Path, args: list[str], model: Path, expect: dict) -> dict:
    """Run the crispembed CLI and validate the output against `expect`.

    expect.kind:
      * "embedding" — JSON embedding of length `dim`, all finite, non-zero norm.
      * "golden"    — stdout (stripped) equals `value`.
      * "nonempty"  — any non-empty stdout (engine loaded + produced output).
    Returns {ok, detail, ...}.
    """
    argv = [a.format(model=str(model)) for a in args]
    proc = subprocess.run([str(cli_bin), *_with_model(argv, model)],
                          capture_output=True, text=True, check=False, timeout=600)
    if proc.returncode != 0 and not proc.stdout.strip():
        return {"ok": False, "detail": f"exit={proc.returncode} "
                f"stderr={proc.stderr[-300:]}"}
    kind = expect.get("kind", "nonempty")
    if kind == "embedding":
        emb = parse_embedding(proc.stdout)
        if emb is None:
            return {"ok": False, "detail": "no parseable embedding in stdout"}
        dim = expect.get("dim")
        if dim is not None and len(emb) != dim:
            return {"ok": False, "detail": f"dim {len(emb)} != expected {dim}"}
        if not all(math.isfinite(x) for x in emb):
            return {"ok": False, "detail": "embedding has non-finite values"}
        norm = math.sqrt(sum(x * x for x in emb))
        if norm <= 0:
            return {"ok": False, "detail": "zero-norm embedding"}
        return {"ok": True, "detail": f"dim={len(emb)} norm={norm:.4f}"}
    if kind == "golden":
        ok = proc.stdout.strip() == expect.get("value", "").strip()
        return {"ok": ok, "detail": "golden match" if ok else
                f"stdout!=golden: {proc.stdout.strip()[:120]!r}"}
    # nonempty
    ok = bool(proc.stdout.strip())
    return {"ok": ok, "detail": "non-empty stdout" if ok else "empty stdout"}


def _with_model(argv: list[str], model: Path) -> list[str]:
    """If the smoke args already contain the model path (via {model}), pass
    them through; otherwise the CLI takes the model via `-m <gguf>`. The
    manifest's `--json {model} "text"` form positions {model} as a bare arg,
    but crispembed wants `-m`; rewrite a lone {model}-substituted gguf to
    `-m <gguf>`."""
    out: list[str] = []
    mstr = str(model)
    for a in argv:
        if a == mstr:
            out += ["-m", mstr]
        else:
            out.append(a)
    return out


# ── per-entry driver ─────────────────────────────────────────────────────────
def regression_for(name: str, manifest: dict, work_dir: Path,
                   build_dir: Path) -> dict:
    entry = next((b for b in manifest["backends"] if b["name"] == name), None)
    if entry is None:
        die(f"backend '{name}' not in manifest")
    tier = entry.get("tier", "diff")

    gguf = hf_download(entry["gguf"]["repo"], entry["gguf"]["file"],
                       entry["gguf"]["revision"], work_dir)

    if tier == "diff":
        ref = hf_download(entry["ref"]["repo"], entry["ref"]["path"],
                          entry["ref"]["revision"], work_dir)
        # Companion refs (e.g. granite-llm-ref.gguf) must sit next to the main
        # ref so the harness finds them by the names it loads internally.
        for comp in entry["ref"].get("companion_paths", []):
            hf_download(entry["ref"]["repo"], comp, entry["ref"]["revision"],
                        work_dir)
        diff_bin = build_dir / entry["diff_bin"]
        if not diff_bin.exists():
            diff_bin = build_dir / "bin" / entry["diff_bin"]
        if not diff_bin.exists():
            die(f"diff binary not built: {entry['diff_bin']} (build it with "
                f"`cmake --build {build_dir} --target {entry['diff_bin']}`)")
        stages = run_diff(diff_bin, entry.get("argv", "{model} {ref}"), gguf, ref)
        passes, fails, missing = evaluate_diff(
            stages, entry.get("diff_thresholds", {}), entry.get("require_stages"))
        ok = not fails and not missing
        return {"backend": name, "tier": tier, "ok": ok,
                "passes": len(passes), "fails": fails, "missing": missing,
                "stages": {k: v["cos_min"] for k, v in stages.items()
                           if k != "_verdicts"}}

    # smoke
    cli_bin = build_dir / "crispembed"
    if not cli_bin.exists():
        cli_bin = build_dir / "bin" / "crispembed"
    if not cli_bin.exists():
        die(f"crispembed CLI not built in {build_dir}")
    res = run_smoke(cli_bin, entry["smoke"]["args"], gguf, entry["smoke"]["expect"])
    return {"backend": name, "tier": tier, "ok": res["ok"], "detail": res["detail"]}


def dry_run(manifest: dict, tier: str | None) -> int:
    print(f"manifest version {manifest['version']} — "
          f"{len(manifest['backends'])} entries")
    bad = 0
    for b in manifest["backends"]:
        if tier and b.get("tier") != tier:
            continue
        g = b["gguf"]
        line = f"  {b['name']:24s} [{b.get('tier','?'):5s}]  {g['repo']}@{g['revision'][:8]} :: {g['file']}"
        if b.get("tier") == "diff":
            r = b.get("ref", {})
            line += f"  ref={r.get('repo','?')}::{r.get('path','?')}"
            if "*" not in b.get("diff_thresholds", {}) and not b.get("require_stages") \
               and not any(k != "*" for k in b.get("diff_thresholds", {})):
                print(line + "  \033[31m(no thresholds!)\033[0m"); bad += 1; continue
        print(line)
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", help="run a single entry by name")
    p.add_argument("--tier", choices=["diff", "smoke"], help="restrict to a tier")
    p.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/crispembed-regression"))
    p.add_argument("--list", action="store_true", help="list entries and exit")
    p.add_argument("--dry-run", action="store_true",
                   help="validate manifest pins without downloading/running")
    args = p.parse_args()

    manifest = load_manifest()
    if args.list:
        for b in manifest["backends"]:
            print(f"{b['name']:24s} {b.get('tier','?')}")
        return 0
    if args.dry_run:
        return dry_run(manifest, args.tier)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    names = [args.backend] if args.backend else [
        b["name"] for b in manifest["backends"]
        if not args.tier or b.get("tier") == args.tier]

    results = []
    for name in names:
        print(f"\n========== {name} ==========")
        try:
            r = regression_for(name, manifest, args.work_dir, args.build_dir)
        except SystemExit:
            raise
        except Exception as exc:
            r = {"backend": name, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        flag = "\033[32m✓\033[0m" if r.get("ok") else "\033[31m✗\033[0m"
        print(f"  {flag} {r}")

    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\nSUMMARY  ok={n_ok}/{len(results)}")
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
