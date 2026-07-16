"""Community-GGUF import matrix driver (A3).

Loads THIRD-PARTY (llama.cpp / Ollama) GGUFs of models we claim to support and
checks they actually work — the gap that produced issue #33, where our own
cstr/* conversion loaded fine while the community GGUF aborted at load.

Checks per model:
  1. loads            — CLI exits 0
  2. shape            — reported n_layer / dim match the manifest (catches the
                        silent-default trap: #33 would give 384-dim / 6-layer)
  3. dim              — returned vector length matches
  4. garbage guard    — cos(query, related) must beat cos(query, unrelated) by
                        min_margin. A model loaded with fabricated hparams still
                        emits floats, so rc==0 + right dim prove nothing on their
                        own; this is the "judge the decoded output" gate.
  5. cross-conversion — optional A/B: cosine(community gguf, our gguf) of the
                        same text must be >= min_ref_cos.

The pure logic (parse_load_banner / cosine / evaluate) is unit-tested with canned
data by tests/test_community_gguf_smoke.py — no models, no network, runs in CI.

Env:
  CRISPEMBED_BIN          path to the crispembed CLI (default: build/crispembed)
  CRISPEMBED_MODELS_DIR   where to look for GGUFs (default: ~/crispembed-live-cache)
  CRISPEMBED_FETCH_MODELS=1  allow downloading a missing GGUF from HuggingFace
                             (off by default so CI never pulls GBs implicitly)

Usage:
  python tests/run_community_gguf.py --list
  python tests/run_community_gguf.py --name nomic-embed-text-v2-moe
  CRISPEMBED_FETCH_MODELS=1 python tests/run_community_gguf.py --all
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
REPO = HERE.parent
MANIFEST = HERE / "community_gguf_matrix.json"

# "crispembed: loaded 12 layers, 768 dims, 250048 vocab"
_BANNER = re.compile(r"loaded\s+(\d+)\s+layers,\s+(\d+)\s+dims")


def parse_load_banner(stderr: str) -> dict:
    """Pull the reported shape out of the CLI's load banner.

    Returns {} when the banner is absent (model failed to load before printing).
    """
    m = _BANNER.search(stderr or "")
    if not m:
        return {}
    return {"n_layer": int(m.group(1)), "dim": int(m.group(2))}


def cosine(a, b) -> float:
    if not a or not b or len(a) != len(b):
        return float("nan")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return dot / (na * nb)


def evaluate(entry: dict, banner: dict, dim: int, cos_related: float, cos_unrelated: float,
             ref_cos: float | None = None) -> list[tuple[str, bool, str]]:
    """Pure pass/fail logic. Returns [(check_name, ok, detail)].

    Kept free of I/O so it can be unit-tested with canned values.
    """
    out: list[tuple[str, bool, str]] = []

    want_layer, want_dim = entry.get("n_layer"), entry.get("dim")
    if want_layer is not None:
        got = banner.get("n_layer")
        out.append(("n_layer", got == want_layer, f"got {got}, want {want_layer}"))
    if want_dim is not None:
        got = banner.get("dim")
        out.append(("banner_dim", got == want_dim, f"got {got}, want {want_dim}"))
        out.append(("vector_dim", dim == want_dim, f"got {dim}, want {want_dim}"))

    margin = entry.get("min_margin", 0.05)
    delta = (cos_related - cos_unrelated) if not (math.isnan(cos_related) or math.isnan(cos_unrelated)) else float("nan")
    ok = (not math.isnan(delta)) and delta >= margin
    out.append(("garbage_guard", ok,
                f"cos(related)={cos_related:.4f} cos(unrelated)={cos_unrelated:.4f} "
                f"margin={delta:.4f} (need >={margin})"))

    if ref_cos is not None:
        need = entry.get("min_ref_cos", 0.90)
        out.append(("cross_conversion", ref_cos >= need, f"cos={ref_cos:.4f} (need >={need})"))

    return out


def load_manifest() -> dict:
    with open(MANIFEST) as f:
        return json.load(f)


def _models_dir() -> Path:
    return Path(os.environ.get("CRISPEMBED_MODELS_DIR", str(Path.home() / "crispembed-live-cache")))


def _bin() -> Path:
    env = os.environ.get("CRISPEMBED_BIN")
    return Path(env) if env else REPO / "build" / "crispembed"


def resolve_model(entry: dict) -> Path | None:
    """Find the GGUF locally; optionally fetch it when CRISPEMBED_FETCH_MODELS=1."""
    p = _models_dir() / entry["file"]
    if p.is_file():
        return p
    if os.environ.get("CRISPEMBED_FETCH_MODELS") != "1":
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(entry["repo"], entry["file"])
    import shutil

    shutil.copy(src, p)
    return p


def embed(binary: Path, model: Path, text: str) -> tuple[list[float], str, int]:
    """Run the CLI once. Returns (vector, stderr, returncode)."""
    r = subprocess.run(
        [str(binary), "-m", str(model), "--prefix", "", "--json", text],
        capture_output=True, text=True, timeout=300,
    )
    vec: list[float] = []
    if r.returncode == 0 and r.stdout.strip():
        try:
            vec = json.loads(r.stdout)[0]["embedding"]
        except Exception:
            vec = []
    return vec, r.stderr, r.returncode


def run_one(entry: dict) -> tuple[bool, list[tuple[str, bool, str]]]:
    binary, model = _bin(), resolve_model(entry)
    if not binary.is_file():
        return False, [("binary", False, f"{binary} not found — build crispembed-cli")]
    if model is None:
        return False, [("model", False,
                        f"{entry['file']} not in {_models_dir()} "
                        f"(set CRISPEMBED_FETCH_MODELS=1 to download {entry['repo']})")]

    qp, dp = entry.get("query_prefix", ""), entry.get("doc_prefix", "")
    q_vec, stderr, rc = embed(binary, model, qp + entry["query"])
    if rc != 0 or not q_vec:
        return False, [("loads", False, f"rc={rc}; stderr tail: {(stderr or '')[-300:]}")]
    rel_vec, _, _ = embed(binary, model, dp + entry["related"])
    unrel_vec, _, _ = embed(binary, model, dp + entry["unrelated"])

    ref_cos = None
    ref_file = entry.get("ref_file")
    if ref_file:
        ref_path = _models_dir() / ref_file
        if ref_path.is_file():
            ref_vec, _, ref_rc = embed(binary, ref_path, qp + entry["query"])
            if ref_rc == 0 and ref_vec:
                ref_cos = cosine(q_vec, ref_vec)

    results = [("loads", True, "rc=0")]
    results += evaluate(entry, parse_load_banner(stderr), len(q_vec),
                        cosine(q_vec, rel_vec), cosine(q_vec, unrel_vec), ref_cos)
    return all(ok for _, ok, _ in results), results


def main() -> int:
    ap = argparse.ArgumentParser(description="Community-GGUF import matrix")
    ap.add_argument("--name", help="run a single manifest entry")
    ap.add_argument("--all", action="store_true", help="run every entry")
    ap.add_argument("--list", action="store_true", help="list entries")
    args = ap.parse_args()

    models = load_manifest()["models"]
    if args.list:
        for e in models:
            print(f"{e['name']:32s} {e['repo']}/{e['file']}")
        return 0

    todo = [e for e in models if e["name"] == args.name] if args.name else (models if args.all else [])
    if not todo:
        ap.print_help()
        return 2

    failed = skipped = 0
    for e in todo:
        print(f"\n=== {e['name']} ({e['repo']}) ===")
        ok, results = run_one(e)
        for name, good, detail in results:
            if not good and name in ("model", "binary"):
                print(f"  [SKIP] {name}: {detail}")
                skipped += 1
                break
            print(f"  [{'PASS' if good else 'FAIL'}] {name}: {detail}")
        else:
            if not ok:
                failed += 1
    print(f"\n{'FAILED' if failed else 'OK'} ({failed} failed, {skipped} skipped)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
