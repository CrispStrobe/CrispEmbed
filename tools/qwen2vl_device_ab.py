#!/usr/bin/env python3
"""CPU/GPU device A/B for TeleOCR and MinerU2.5-Pro (qwen2vl_ocr).

Runs every committed case of one model at one precision on one device through
the resident-context runners (test-teleocr-run / test-mineru25-run), compares
each run's completion IDs with the committed original-Python reference IDs,
and writes one receipt JSON per (model, precision, device).

    python tools/qwen2vl_device_ab.py --model teleocr --gguf teleocr-f16.gguf \
        --precision f16 --device metal --build build --out tests/results/teleocr/metal-m1

Device control uses only the production gates: cpu sets QWEN2VL_OCR_FORCE_CPU=1;
metal leaves the GPU selection to the library (TeleOCR additionally needs its
opt-in CRISPEMBED_TELEOCR_GPU=1 until a default flip). Extra KEY=VALUE pairs
from --env are applied to every run and recorded in the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "tests/regression/images"


def teleocr_cases():
    parity = json.loads((ROOT / "tests/results/teleocr/hosted-parity.json").read_text())
    cap = parity["max_new_tokens"]
    out = []
    for case in parity["cases"]:
        meta = json.loads((ROOT / f"tests/results/teleocr/reference/{case['id']}.json").read_text())
        out.append({
            "id": case["id"],
            "image": ROOT / "tests/results/teleocr/reference" / meta["image"],
            "prompt": meta["prompt"],
            "max_tokens": cap,
            "reference_ids": case["reference_ids"],
            "reference_text": case["reference_text"],
        })
    return out


def mineru_cases():
    parity = json.loads((ROOT / "tests/results/mineru25/native-parity.json").read_text())
    out = []
    for case in parity["cases"]:
        out.append({
            "id": case["id"],
            "image": IMAGES / case["image"],
            "task": case["task"],
            "max_tokens": case["max_tokens"],
            "reference_ids": case["reference_ids"],
            "reference_text": case["reference_text"],
        })
    return out


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["teleocr", "mineru"], required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--precision", required=True)
    ap.add_argument("--device", choices=["cpu", "metal", "gpu"], required=True)
    ap.add_argument("--build", default="build")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--cases", default="", help="comma-separated subset")
    ap.add_argument("--env", action="append", default=[], help="extra KEY=VALUE")
    ap.add_argument("--label", default="", help="receipt name suffix")
    ap.add_argument("--dump", default="", help="stage dump root (one subdir per case)")
    ap.add_argument("--hash", action="store_true", help="record the GGUF sha256")
    ap.add_argument("--force-reference", action="store_true",
                    help="teacher-force the committed reference IDs and dump the logits at every generated "
                         "position (final-output cosine per docs/precision-acceptance.md)")
    ap.add_argument("--free-then-forced", action="store_true",
                    help="one model load: run 0 free-running (decoded agreement), run 1 teacher-forced with "
                         "all-position logit dumps (needs --dump; implies --repeat 2)")
    ap.add_argument("--merge", action="store_true",
                    help="merge cases into an existing receipt (one lock per case on a shared host)")
    args = ap.parse_args()
    if args.free_then_forced:
        if not args.dump:
            ap.error("--free-then-forced needs --dump")
        args.repeat = 2

    env = dict(os.environ)
    env.update({
        "CRISPEMBED_TELEOCR": "1",
        "CRISPEMBED_ACCEPT_QWEN_RESEARCH_LICENSE": "1",
        "CRISPEMBED_MINERU25_PRO": "1",
    })
    env.pop("QWEN2VL_OCR_FORCE_CPU", None)
    if args.device == "cpu":
        env["QWEN2VL_OCR_FORCE_CPU"] = "1"
    elif args.model == "teleocr":
        env.setdefault("CRISPEMBED_TELEOCR_GPU", "1")
    extra = {}
    for kv in args.env:
        k, _, v = kv.partition("=")
        extra[k] = v
    env.update(extra)

    cases = teleocr_cases() if args.model == "teleocr" else mineru_cases()
    if args.cases:
        wanted = set(args.cases.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    build = Path(args.build)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.precision}-{args.device}" + (f"-{args.label}" if args.label else "")

    results = []
    for case in cases:
        run_env = dict(env)
        if args.dump:
            d = Path(args.dump) / case["id"]
            d.mkdir(parents=True, exist_ok=True)
            run_env["CRISPEMBED_QWEN2VL_STAGE_DUMP"] = str(d)
            if args.force_reference or args.free_then_forced:
                forced = d / "forced_ids.txt"
                forced.write_text(" ".join(str(i) for i in case["reference_ids"]) + "\n")
                run_env["CRISPEMBED_QWEN2VL_FORCE_IDS"] = str(forced)
                run_env["CRISPEMBED_QWEN2VL_STAGE_DUMP_STEPS"] = "100000"
                if args.free_then_forced:
                    run_env["CRISPEMBED_QWEN2VL_FORCE_FROM_CALL"] = "1"
                    run_env["CRISPEMBED_QWEN2VL_STAGE_DUMP_CALL"] = "1"
        if args.model == "teleocr":
            cmd = [str(build / "test-teleocr-run"), args.gguf, str(case["image"]), case["prompt"],
                   str(case["max_tokens"]), str(args.threads), str(args.repeat)]
        else:
            cmd = [str(build / "test-mineru25-run"), "--model", args.gguf, "--image", str(case["image"]),
                   "--task", case["task"], "--max-tokens", str(case["max_tokens"]), "--threads",
                   str(args.threads), "--repeat", str(args.repeat)]
        # Peak RSS: /usr/bin/time when present (absent on Kaggle images), else
        # getrusage(RUSAGE_CHILDREN), the high-water mark over children so far.
        use_time = os.path.exists("/usr/bin/time")
        if use_time:
            cmd = ["/usr/bin/time", "-l" if sys.platform == "darwin" else "-v"] + cmd
        started = time.time()
        proc = subprocess.run(cmd, env=run_env, capture_output=True, text=True)
        wall = time.time() - started
        log = out_dir / f"{name}-{case['id']}.log"
        log.write_text(proc.stderr)
        rec = {"id": case["id"], "exit": proc.returncode, "wall_s": wall}
        if not use_time:
            import resource
            peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
            rec["max_rss_mib"] = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
            rec["max_rss_source"] = "getrusage(RUSAGE_CHILDREN) high-water mark"
        for line in proc.stderr.splitlines():
            if "maximum resident set size" in line:  # macOS: bytes
                rec["max_rss_mib"] = int(line.split()[0]) / (1024 * 1024)
            elif "Maximum resident set size (kbytes)" in line:  # GNU time: KiB
                rec["max_rss_mib"] = int(line.split()[-1]) / 1024
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            data = None
        if data is None:
            rec["error"] = "no JSON output"
            results.append(rec)
            print(f"{case['id']}: FAILED exit={proc.returncode}", file=sys.stderr)
            continue
        if args.model == "teleocr":
            runs = [{"ms": r["ms"], "ids": r["ids"], "text": r["text"], "eos": r["eos"]} for r in data["runs"]]
            rec["load_ms"] = data["load_ms"]
        else:
            runs = [{"ms": ms, "ids": r["token_ids"], "text": r.get("raw_text", r.get("text", "")),
                     "eos": r.get("stopped_eos", r.get("eos"))}
                    for ms, r in zip(data["timing"]["runs_ms"], data["runs"])]
            rec["load_ms"] = data["timing"]["load_ms"]
        ref = case["reference_ids"]
        if args.free_then_forced and len(runs) == 2:
            runs[1]["teacher_forced"] = True
            rec["forced_run"] = runs.pop(1)
        for r in runs:
            r["ids_exact"] = r["ids"] == ref
            r["first_divergence"] = first_divergence(r["ids"], ref)
        rec["runs"] = runs
        rec["n_ids"] = len(runs[0]["ids"])
        rec["n_reference_ids"] = len(ref)
        rec["ids_exact_all_runs"] = all(r["ids_exact"] for r in runs)
        rec["runs_consistent"] = all(r["ids"] == runs[0]["ids"] for r in runs)
        rec["first_divergence"] = runs[0]["first_divergence"]
        warm = [r["ms"] for r in runs[1:]] or ([rec["forced_run"]["ms"]] if "forced_run" in rec else [runs[0]["ms"]])
        rec["cold_ms"] = runs[0]["ms"]
        rec["warm_ms_min"] = min(warm)
        rec["warm_ms_mean"] = sum(warm) / len(warm)
        results.append(rec)
        print(f"{case['id']}: exact={rec['ids_exact_all_runs']} n={rec['n_ids']}/{len(ref)} "
              f"div={rec['first_divergence']} cold={rec['cold_ms']:.0f} warm={rec['warm_ms_min']:.0f}ms",
              file=sys.stderr)

    target = out_dir / f"{name}.json"
    if args.merge and target.exists():
        previous = json.loads(target.read_text())
        mine = {r["id"] for r in results}
        results = [c for c in previous["cases"] if c["id"] not in mine] + results
    receipt = {
        "schema": "crispembed.qwen2vl.device-ab.v1",
        "model": args.model,
        "precision": args.precision,
        "device": args.device,
        "gguf": Path(args.gguf).name,
        "gguf_sha256": sha256(Path(args.gguf)) if args.hash else None,
        "threads": args.threads,
        "repeat": args.repeat,
        "extra_env": extra,
        "teacher_forced": bool(args.force_reference),
        "free_then_forced": bool(args.free_then_forced),
        "commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                 text=True).stdout.strip(),
        "dirty": bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "src", "ggml"],
                                     capture_output=True, text=True).stdout.strip()),
        "host": "Apple M1 (8 GPU cores, 16 GiB)",
        "all_exact": all(r.get("ids_exact_all_runs") for r in results),
        "cases": results,
    }
    target.write_text(json.dumps(receipt, indent=1, ensure_ascii=False) + "\n")
    return 0 if receipt["all_exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
