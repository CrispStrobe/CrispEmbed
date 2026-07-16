"""Precision-control check: prove a low q4_k cosine is quantization, not a bug.

A low crispembed-vs-HF cosine on a shipped q4_k GGUF (e.g. nomic-v1.5 at 0.9515)
never distinguishes "quant floor" from "our encoder graph is wrong". The only
thing that does is re-running the SAME code path at higher precision: if the
f16/f32 GGUF of the same model matches the ORIGINAL Python model to ~1.0 at
EVERY stage, the graph is exact and the entire q4_k gap is quantization.

This automates the manual f16/f32 control that separated those two for issue #33.
For each manifest entry that declares a `control_file` it:
  1. dumps the control (f16/f32) GGUF's per-stage activations
     (CRISPEMBED_DUMP_LAYERS_GGUF)
  2. dumps the HF/PyTorch reference for the same text
     (tools/dump_encoder_reference.py)
  3. compares per-stage (tests/test_encoder_diff.py) and asserts every stage
     >= control_min_cos

Env:
  CRISPEMBED_BIN, CRISPEMBED_MODELS_DIR   (as run_community_gguf.py)
  CRISPEMBED_FETCH_MODELS=1               allow downloading a missing control GGUF
  HF_HOME                                  writable HF cache (the default symlink
                                           here points at an unmounted volume)

Usage:
  HF_HOME=/tmp/hf python tests/prove_quant_control.py --name bge-small-en-v1.5
  HF_HOME=/tmp/hf python tests/prove_quant_control.py --all
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import run_community_gguf as drv  # noqa: E402

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DUMPER = REPO / "tools" / "dump_encoder_reference.py"
DIFF = HERE / "test_encoder_diff.py"


def _control_path(entry: dict) -> Path | None:
    p = Path(os.environ.get("CRISPEMBED_MODELS_DIR", str(Path.home() / "crispembed-live-cache"))) / entry["control_file"]
    if p.is_file():
        return p
    if os.environ.get("CRISPEMBED_FETCH_MODELS") != "1":
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    import shutil

    p.parent.mkdir(parents=True, exist_ok=True)
    src = hf_hub_download(entry["repo"], entry["control_file"])
    shutil.copy(src, p)
    return p


def run(entry: dict, workdir: Path) -> bool | None:
    binary = drv._bin()
    control = _control_path(entry)
    if not binary.is_file():
        print(f"  [SKIP] {entry['name']}: crispembed binary missing")
        return None
    if control is None:
        print(f"  [SKIP] {entry['name']}: control {entry['control_file']} missing "
              f"(set CRISPEMBED_FETCH_MODELS=1 to download)")
        return None

    text = entry.get("query_prefix", "") + entry["query"]
    ours = workdir / f"{entry['name']}.control.gguf"
    ref = workdir / f"{entry['name']}.ref.gguf"

    # 1. control GGUF per-stage dump (our code path, high precision).
    env = dict(os.environ, CRISPEMBED_DUMP_LAYERS="1", CRISPEMBED_DUMP_LAYERS_GGUF=str(ours))
    r = subprocess.run([str(binary), "-m", str(control), "--prefix", "", "--json", text],
                       capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0 or not ours.is_file():
        print(f"  [FAIL] {entry['name']}: control dump failed rc={r.returncode}\n{r.stderr[-300:]}")
        return False

    # 2. HF reference per-stage dump (original Python model).
    cmd = [sys.executable, str(DUMPER), "--model", entry["hf_repo"], "--text", text, "--output", str(ref)]
    if entry.get("hf_trust_remote_code"):
        cmd.append("--trust-remote-code")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or not ref.is_file():
        print(f"  [FAIL] {entry['name']}: HF reference dump failed rc={r.returncode}\n{r.stderr[-400:]}")
        return False

    # 3. per-stage compare — every stage must clear control_min_cos.
    floor = str(entry.get("control_min_cos", 0.999))
    r = subprocess.run([sys.executable, str(DIFF), "--ours", str(ours), "--ref", str(ref),
                        "--min-cos", floor, "--gate-cos", "0.999"],
                       capture_output=True, text=True, timeout=300)
    ok = r.returncode == 0
    # Surface the per-stage line summary either way.
    tail = [ln for ln in r.stdout.splitlines() if "cos=" in ln or "OK" in ln or "FAILED" in ln]
    print(f"  [{'PASS' if ok else 'FAIL'}] {entry['name']} (control={entry['control_file']}, floor={floor})")
    for ln in tail[-3:]:
        print("        " + ln.strip())
    if ok:
        print(f"        -> graph exact at high precision; q4_k gap (measured {entry.get('measured_hf_cos')}) "
              f"is quantization, not a bug.")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Prove shipped-quant cosine is quantization, not a bug")
    ap.add_argument("--name")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    models = [m for m in drv.load_manifest()["models"] if "control_file" in m]
    todo = [m for m in models if m["name"] == a.name] if a.name else (models if a.all else [])
    if not todo:
        print("no matching entries with a control_file")
        ap.print_help()
        return 2

    print("precision-control: crispembed f16/f32 vs HF fp32, per stage")
    bad = 0
    ran = 0
    with tempfile.TemporaryDirectory() as td:
        for e in todo:
            res = run(e, Path(td))
            if res is None:
                continue
            ran += 1
            if not res:
                bad += 1
    if ran == 0:
        print("SKIP (no controls available)")
        return 0
    print("FAILED" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
