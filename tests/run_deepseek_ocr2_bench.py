#!/usr/bin/env python3
"""Run the deepseek-ocr2 lane over a fixture directory, one process per page.

T14 needs two things the generic parity harness does not give it: the decoded
text of *one* lane saved per fixture so two arms can be diffed byte-for-byte,
and the `decode=` field of `[deepseek-ocr2-stage-bench]` rather than its
`total=` (the vision tower dominates `total`, and no decode change moves it).

Each page runs as a SEPARATE process on purpose.  A second inference spawned
immediately after the first in the same shell has been observed to exit at 0 s
on a GPU/resource-release race, which mints a fake "win"; the returncode and a
non-empty transcript are both checked before any timing is recorded.

    python tests/run_deepseek_ocr2_bench.py \
        --binary build/crispembed --model ~/.cache/crispembed-local/ds.gguf \
        --images ~/crispembed-ocr-synth --out tests/results/t14/legacy-synth \
        --env DS2_LEGACY_DECODE=1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ocr_external_parity import DEEPSEEK_OCR2_DECODE_RE, REGIONS_RE, STAGE_BENCH  # noqa: E402

FIELD_RE = re.compile(r"\[deepseek-ocr2-stage-bench\] (.*)")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_bench(err: str) -> dict:
    """Pull every `name=value` pair off the stage-bench line."""
    m = FIELD_RE.search(err)
    if not m:
        return {}
    out: dict[str, float | str] = {}
    for k, v in re.findall(r"(\w+)=([-\w.]+)", m.group(1)):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def run_once(binary: Path, model: Path, image: Path, env: dict, gpu_backend: str | None):
    cmd = [str(binary), "--ocr-pipeline", str(image),
           "--ocr-engine", "deepseek-ocr2", "--ocr-rec", str(model)]
    if gpu_backend:
        cmd += ["--gpu-backend", gpu_backend]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=3600)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    text = REGIONS_RE.sub("", p.stdout).strip()
    return text, wall_ms, parse_bench(p.stderr), p.returncode, p.stderr


def interleave(args, base_env: dict) -> int:
    """Alternate the two arms on one fixture, one process per run.

    Interleaved rather than blocked because a 16 GB box shared with other
    agents drifts: running all of arm A then all of arm B measures the drift,
    not the change.  The first pair is discarded as cold, and each run is its
    own process (a second inference spawned immediately after the first in the
    same shell has exited at 0 s on a resource-release race).
    """
    img = Path(args.interleave)
    # The persistent arm is selected EXPLICITLY via DS2_FAST_DECODE=1 rather
    # than by omitting the legacy gate, because the shipped default is legacy.
    arms = {"legacy": dict(base_env, DS2_LEGACY_DECODE="1"),
            "persistent": dict(base_env, DS2_FAST_DECODE="1")}
    arms["persistent"].pop("DS2_LEGACY_DECODE", None)
    samples: dict[str, list[dict]] = {k: [] for k in arms}
    texts: dict[str, set[str]] = {k: set() for k in arms}
    pairs_out, dropped = [], []

    for pair in range(args.pairs + 1):  # +1: pair 0 is the discarded cold pair
        load1 = float(os.popen("sysctl -n vm.loadavg").read().split()[1])
        rec_pair = {"pair": pair, "load_at_start": load1}
        for arm, env in arms.items():
            text, wall_ms, bench, rc, err = run_once(
                args.binary, args.model, img, env, args.gpu_backend)
            ok = rc == 0 and bool(text)
            texts[arm].add(text)
            rec = {"pair": pair, "ok": ok, "wall_ms": round(wall_ms, 1), **bench}
            rec_pair[arm] = rec
            if pair > 0 and ok and load1 <= args.max_load:
                samples[arm].append(rec)
            print(f"pair {pair} {arm:11s} ok={ok} decode={bench.get('decode')} "
                  f"total={bench.get('total')} chars={len(text)} load1={load1}", flush=True)
        # The user is on this machine; a fully quiet window may never arrive, so
        # pairs started under heavy load are recorded but excluded from the
        # verdict rather than silently averaged in.
        both_ok = all(rec_pair[a].get("ok") for a in arms)
        if pair > 0 and both_ok:
            ld = rec_pair["legacy"].get("decode")
            pd = rec_pair["persistent"].get("decode")
            if ld and pd:
                rec_pair["ratio_persistent_over_legacy"] = round(pd / ld, 4)
        if pair > 0 and load1 > args.max_load:
            dropped.append(pair)
            rec_pair["dropped_reason"] = f"load {load1} > {args.max_load}"
        pairs_out.append(rec_pair)

    def stats(vals: list[float]) -> dict:
        vals = sorted(vals)
        n = len(vals)
        if not n:
            return {}
        med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        spread = (vals[-1] - vals[0]) / med if med else 0.0
        return {"n": n, "median": round(med, 1), "min": round(vals[0], 1),
                "max": round(vals[-1], 1), "spread_frac": round(spread, 3)}

    summary = {}
    for arm, rows in samples.items():
        summary[arm] = {
            "decode_ms": stats([r["decode"] for r in rows if "decode" in r]),
            "total_ms": stats([r["total"] for r in rows if "total" in r]),
            "prefill_ms": stats([r["prefill"] for r in rows if "prefill" in r]),
            "sam_ms": stats([r["sam"] for r in rows if "sam" in r]),
            "qwen2_enc_ms": stats([r["qwen2_enc"] for r in rows if "qwen2_enc" in r]),
            "wall_ms": stats([r["wall_ms"] for r in rows]),
            "distinct_texts": len(texts[arm]),
        }
    ratios = [p["ratio_persistent_over_legacy"] for p in pairs_out
              if p["pair"] > 0 and "ratio_persistent_over_legacy" in p
              and p["load_at_start"] <= args.max_load]
    ratio_stats = stats(ratios) if ratios else {}
    doc = {"fixture": img.name, "pairs": args.pairs, "max_load": args.max_load,
           "gpu_backend": args.gpu_backend or "default",
           "base_env": args.env, "samples": samples, "summary": summary,
           "per_pair": pairs_out, "dropped_pairs_high_load": dropped,
           "ratio_persistent_over_legacy": ratio_stats,
           "text_identical_across_arms": len(texts["legacy"] | texts["persistent"]) == 1}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"interleaved_{img.stem}.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"per-pair persistent/legacy decode ratios (load-gated): {ratios}")
    print(f"ratio stats: {ratio_stats}")
    print(f"dropped for load > {args.max_load}: {dropped}")
    print(f"text identical across arms: {doc['text_identical_across_arms']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True, type=Path)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--env", action="append", default=[],
                    help="KEY=VALUE applied to every run (repeatable)")
    ap.add_argument("--gpu-backend", default=None,
                    help="passed through as --gpu-backend (e.g. cpu)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--labelled-only", action="store_true",
                    help="restrict to fixtures present in the directory's "
                         "ground_truth.json (the CC0 dir holds 14 images but "
                         "only 5 are transcribed, and the rest are Fraktur / "
                         "Arabic / handwriting the English gate does not score)")
    ap.add_argument("--interleave", type=Path, default=None,
                    help="A/B one fixture: alternate legacy/persistent arms, "
                         "one process per run, instead of sweeping a corpus")
    ap.add_argument("--max-load", type=float, default=8.0,
                    help="drop pairs whose 1-min load average at pair start "
                         "exceeds this; the machine has an interactive user, so "
                         "a fully quiet window may never arrive")
    ap.add_argument("--pairs", type=int, default=5,
                    help="interleave mode: scored pairs (a cold pair 0 is "
                         "always run first and discarded)")
    args = ap.parse_args()

    env = os.environ.copy()
    env["CRISPEMBED_DEEPSEEK_OCR2_BENCH"] = "1"
    for kv in args.env:
        k, _, v = kv.partition("=")
        env[k] = v

    if args.interleave:
        return interleave(args, env)

    args.out.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.labelled_only:
        gt = args.images / "ground_truth.json"
        keep = {r["file"] for r in json.loads(gt.read_text())["records"]}
        images = [p for p in images if p.name in keep]
    if args.limit:
        images = images[:args.limit]

    rows, failures = [], []
    for img in images:
        text, wall_ms, bench, rc, err = run_once(
            args.binary, args.model, img, env, args.gpu_backend)
        # A crash or an empty transcript is never timed as a win.
        ok = rc == 0 and bool(text)
        if not ok:
            failures.append({"fixture": img.name, "returncode": rc,
                             "stderr_tail": err[-800:]})
        (args.out / (img.stem + ".txt")).write_text(text)
        rows.append({"fixture": img.name, "ok": ok, "returncode": rc,
                     "chars": len(text), "wall_ms": round(wall_ms, 1), **bench})
        print(f"{img.name:36s} ok={ok} chars={len(text):5d} "
              f"decode={bench.get('decode')} total={bench.get('total')} "
              f"path={bench.get('decode_path')} kv={bench.get('kv')}", flush=True)

    doc = {"binary": str(args.binary), "model": str(args.model),
           "images": str(args.images), "env": args.env,
           "gpu_backend": args.gpu_backend or "default",
           "rows": rows, "failures": failures}
    (args.out / "runs.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\n{len(rows)} pages, {len(failures)} failures -> {args.out}")
    # Referenced so the shared regexes stay the single source of truth.
    assert "deepseek-ocr2" in STAGE_BENCH and DEEPSEEK_OCR2_DECODE_RE
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
