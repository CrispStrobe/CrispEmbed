#!/usr/bin/env python3
"""F1 acceptance analysis over the run_matrix.sh output dirs.

Gates (HANDOVER §F1):
  A. Both decode arms byte-identical per arm (guard on), Metal AND CPU.
  B. Synth 20/20 unchanged guard-on vs guard-off (the guard must be a no-op
     where nothing spirals) — byte compare, plus CER via the shared scorer.
  C. Spiral pages terminate before the 1024 cap (gen_tokens < 1024).
  D. cc0 CER moves materially toward the reference (0.18743 raw / 0.11063
     stripped) from the pre-guard baseline.
Decoded text judged, never cosine.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent  # worktree root
SYNTH_IMAGES = Path.home() / "crispembed-ocr-synth"
CC0_IMAGES = ROOT / "tests/regression/images/cc0"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def compare_dirs(a: Path, b: Path, label: str) -> bool:
    """Byte-compare all fixture transcripts between two run dirs."""
    ok = True
    txts = sorted(p.name for p in a.glob("*.txt"))
    if not txts:
        print(f"  {label}: NO TRANSCRIPTS in {a}")
        return False
    for name in txts:
        pa, pb = a / name, b / name
        if not pb.exists():
            print(f"  {label}: {name} MISSING in {b}")
            ok = False
            continue
        if pa.read_bytes() != pb.read_bytes():
            print(f"  {label}: {name} DIFFERS ({sha(pa)} vs {sha(pb)})")
            ok = False
    n_same = sum(1 for n in txts if (b / n).exists() and (a / n).read_bytes() == (b / n).read_bytes())
    print(f"  {label}: {n_same}/{len(txts)} byte-identical")
    return ok


def gen_tokens(run_dir: Path) -> dict:
    rows = json.loads((run_dir / "runs.json").read_text())["rows"]
    return {r["fixture"]: (r.get("gen_tokens"), r.get("ok"), r.get("chars")) for r in rows}


def score(run_dir: Path, images: Path, out_json: Path) -> dict:
    cmd = [sys.executable, str(ROOT / "tests/score_gold_transcripts.py"),
           "--images", str(images), "--gold", str(run_dir),
           "--engine", "native", "--strip-markup", "--output", str(out_json)]
    subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if not out_json.exists():
        return {}
    doc = json.loads(out_json.read_text())
    per = {}
    for fx in doc["fixtures"]:
        e = fx["engines"]["native"]
        per[fx["fixture"]] = {
            "cer": e.get("cer"),
            "cer_stripped": (e.get("alt_scores") or {}).get("markup_stripped", {}).get("cer"),
        }
    cers = [v["cer"] for v in per.values() if v["cer"] is not None]
    strp = [v["cer_stripped"] for v in per.values() if v["cer_stripped"] is not None]
    return {"per_fixture": per,
            "mean_cer": sum(cers) / len(cers) if cers else None,
            "mean_cer_stripped": sum(strp) / len(strp) if strp else None}


def main():
    print("== Gate A: arm byte-identity (guard on) ==")
    a1 = compare_dirs(HERE / "m-guard-persist-synth", HERE / "m-guard-legacy-synth", "Metal synth persist-vs-legacy")
    a2 = compare_dirs(HERE / "m-guard-persist-cc0", HERE / "m-guard-legacy-cc0", "Metal cc0   persist-vs-legacy")
    a3 = compare_dirs(HERE / "c-guard-persist-cc0", HERE / "c-guard-legacy-cc0", "CPU   cc0   persist-vs-legacy")
    a4 = compare_dirs(HERE / "c-guard-persist-synth", HERE / "c-guard-legacy-synth", "CPU   synth persist-vs-legacy")

    print("\n== Gate B: synth guard-on vs guard-off (no-op where nothing spirals) ==")
    b1 = compare_dirs(HERE / "m-guard-persist-synth", HERE / "m-base-persist-synth", "Metal synth guard-vs-base")
    b2 = compare_dirs(HERE / "c-guard-persist-synth", HERE / "c-base-persist-synth", "CPU   synth guard-vs-base")

    print("\n== Gate C: spiral termination (gen_tokens, guard arms) ==")
    for d in ("m-guard-persist-cc0", "m-guard-legacy-cc0", "c-guard-persist-cc0", "c-guard-legacy-cc0",
              "m-base-persist-cc0"):
        p = HERE / d
        if not (p / "runs.json").exists():
            print(f"  {d}: missing")
            continue
        gt = gen_tokens(p)
        caps = {k: v for k, v in gt.items() if v[0] and v[0] >= 1024}
        print(f"  {d}: capped pages = {list(caps) if caps else 'NONE'}; "
              + "; ".join(f"{k}={v[0]:.0f}tok" for k, v in sorted(gt.items())))

    print("\n== Gate D: cc0 CER (reference A4: raw 0.18743 / stripped 0.11063) ==")
    for d in ("m-base-persist-cc0", "m-guard-persist-cc0", "m-guard-legacy-cc0",
              "c-guard-persist-cc0", "c-guard-legacy-cc0"):
        p = HERE / d
        if not p.exists():
            continue
        s = score(p, CC0_IMAGES, p / "cer.json")
        if not s:
            print(f"  {d}: scoring failed")
            continue
        print(f"  {d}: mean CER raw={s['mean_cer']:.5f} stripped={s['mean_cer_stripped']:.5f}")
        for fx, v in sorted(s["per_fixture"].items()):
            print(f"      {fx:36s} raw={v['cer']:.4f} stripped={v['cer_stripped']:.4f}")

    print("\n== Synth CER (gate B corroboration) ==")
    for d in ("m-base-persist-synth", "m-guard-persist-synth"):
        p = HERE / d
        if not p.exists():
            continue
        s = score(p, SYNTH_IMAGES, p / "cer.json")
        if s:
            print(f"  {d}: mean CER raw={s['mean_cer']:.5f} stripped={s['mean_cer_stripped']:.5f}")

    print("\nGate A pass:", all([a1, a2, a3, a4]), " Gate B byte-pass:", all([b1, b2]))


if __name__ == "__main__":
    main()
