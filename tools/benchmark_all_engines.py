#!/usr/bin/env python3
"""
Perf sweep across CrispEmbed engines — the benchmark companion to the
regression suite. For every entry in tests/regression/manifest.json (or a
subset / a directory of GGUFs), time:

  * **load_ms**   — `crispembed --dim -m <gguf>`: load the model + print the
                    embedding dim and exit. Pure model-load + init cost.
  * **total_ms**  — `crispembed --json -m <gguf> "<text>"`: load + encode.
  * **encode_ms** — total_ms - load_ms (the marginal inference cost).

Each timing is the median of `--repeat` runs (default 3). Output is a JSON Lines
file + a markdown table. Mirrors CrispASR's tools/benchmark_all_backends; the
Kaggle wrapper (tools/kaggle/crispembed-benchmark/) runs this and pushes results
to the cstr/crispembed-kaggle-progress HF dataset after each engine so partial
results survive a crash.

Only text/embedding engines are timed via the `--dim`/`--json` path (it needs a
text input + an embedding output). Image/OCR/SR engines need a per-engine bench
spec — add a `bench` block to the manifest entry; until then they're reported as
skipped, not silently dropped.

Usage:
    python tools/benchmark_all_engines.py --build-dir build
    python tools/benchmark_all_engines.py --build-dir build --backend bge-small-en-v1.5
    python tools/benchmark_all_engines.py --gguf-dir /path/to/ggufs --build-dir build
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
MANIFEST_PATH = REPO_ROOT / "tests" / "regression" / "manifest.json"
sys.path.insert(0, str(REPO_ROOT / "tests" / "regression"))

DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog. Benchmark sentence for embedding latency."
SAMPLE = REPO_ROOT / "tests" / "regression" / "sample.bmp"


def cli_bin(build_dir: Path) -> Path:
    for c in (build_dir / "crispembed", build_dir / "bin" / "crispembed"):
        if c.exists():
            return c
    sys.exit(f"crispembed CLI not found in {build_dir}")


def _time(cmd: list[str], repeat: int) -> tuple[float | None, str]:
    """Median wall time (ms) over `repeat` runs, or (None, err) on failure.
    Captures bytes (SR/image engines emit binary to stdout)."""
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, timeout=900)
        dt = (time.perf_counter() - t0) * 1000.0
        if p.returncode != 0:
            return None, (p.stderr or b"").decode("utf-8", "replace").strip()[-200:]
        times.append(dt)
    return statistics.median(times), ""


def bench_one(cli: Path, entry: dict, gguf: Path, sample: Path, repeat: int) -> dict:
    """Time the entry's bench invocation. For embeddings, also time --dim
    (load only) and report the encode delta; other modalities report total."""
    spec = entry.get("bench") or entry.get("smoke") or {}
    args = [a.format(model=str(gguf), sample=str(sample))
            for a in spec["args"]]
    total_ms, err = _time([str(cli), *args], repeat)
    if total_ms is None:
        return {"ok": False, "error": err}
    out = {"ok": True, "total_ms": round(total_ms, 1)}
    if entry.get("modality") == "embedding":
        load_ms, lerr = _time([str(cli), "-m", str(gguf), "--dim"], repeat)
        if load_ms is not None:
            out["load_ms"] = round(load_ms, 1)
            out["encode_ms"] = round(max(0.0, total_ms - load_ms), 1)
    return out


def resolve_gguf(entry: dict, gguf_dir: Path | None, work_dir: Path) -> Path | None:
    """Find the entry's GGUF: prefer a local --gguf-dir hit, else HF download."""
    fname = entry["gguf"]["file"]
    if gguf_dir is not None:
        cand = gguf_dir / fname
        if cand.exists():
            return cand
    try:
        from run_one import hf_download
        return hf_download(entry["gguf"]["repo"], fname,
                           entry["gguf"]["revision"], work_dir)
    except Exception as exc:
        print(f"  resolve failed for {fname}: {exc}", file=sys.stderr)
        return None


def benchable(entry: dict) -> bool:
    """Benchable if it has a `bench` block (generated entries) or a `smoke`
    block (curated entries) to time. tier=skip / diff-only entries have neither."""
    for k in ("bench", "smoke"):
        if isinstance(entry.get(k), dict) and "args" in entry[k]:
            return True
    return False


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--build-dir", type=Path, default=REPO_ROOT / "build")
    p.add_argument("--gguf-dir", type=Path, default=None,
                   help="look here for GGUFs before downloading from HF")
    p.add_argument("--work-dir", type=Path, default=Path("/tmp/crispembed-bench"))
    p.add_argument("--backend", help="benchmark a single manifest entry")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--out", type=Path, default=None,
                   help="results JSONL (default tools/benchmark_all_engines.results.jsonl)")
    p.add_argument("--md", type=Path, default=None,
                   help="markdown table (default tools/benchmark_all_engines.results.md)")
    args = p.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    cli = cli_bin(args.build_dir)
    entries = [b for b in manifest["backends"]
               if not args.backend or b["name"] == args.backend]

    out_jsonl = args.out or HERE / "benchmark_all_engines.results.jsonl"
    results: list[dict] = []
    with out_jsonl.open("w") as fout:
        for entry in entries:
            name = entry["name"]
            if not benchable(entry):
                rec = {"backend": name, "ok": False, "skipped": True,
                       "reason": f"tier={entry.get('tier')} (no bench block)"}
                print(f"  ⊘ {name:24s} skipped ({rec['reason']})")
                results.append(rec); fout.write(json.dumps(rec) + "\n"); fout.flush()
                continue
            gguf = resolve_gguf(entry, args.gguf_dir, args.work_dir)
            if gguf is None:
                rec = {"backend": name, "ok": False, "error": "gguf unavailable"}
            else:
                r = bench_one(cli, entry, gguf, SAMPLE, args.repeat)
                rec = {"backend": name, "engine_id": entry.get("engine_id"),
                       "modality": entry.get("modality"),
                       "gguf": entry["gguf"]["file"], **r}
            results.append(rec)
            fout.write(json.dumps(rec) + "\n"); fout.flush()  # crash-survivable
            if rec.get("ok"):
                extra = (f"load={rec['load_ms']:.0f} encode={rec['encode_ms']:.0f} "
                         if "load_ms" in rec else "")
                print(f"  ✓ {name:24s} {extra}total={rec['total_ms']:.0f}ms")
            elif not rec.get("skipped"):
                print(f"  ✗ {name:24s} {rec.get('error','')}")

    md = args.md or HERE / "benchmark_all_engines.results.md"
    lines = ["# CrispEmbed engine benchmark", "",
             f"`crispembed` CLI, median of {args.repeat} runs. "
             "`load` = model load+init (`--dim`); `encode` = marginal encode "
             "(`total - load`).", "",
             "| engine | modality | gguf | load (ms) | encode (ms) | total (ms) |",
             "|---|---|---|---:|---:|---:|"]
    for r in results:
        if r.get("ok"):
            lines.append(f"| `{r['backend']}` | {r.get('modality','')} | "
                         f"`{r['gguf']}` | {r.get('load_ms','')} | "
                         f"{r.get('encode_ms','')} | {r['total_ms']} |")
        elif r.get("skipped"):
            lines.append(f"| `{r['backend']}` | — | — | _skipped_ | | |")
        else:
            lines.append(f"| `{r['backend']}` | — | — | _error_ | | |")
    md.write_text("\n".join(lines) + "\n")

    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"\nwrote {out_jsonl}\nwrote {md}\nbenchmarked {n_ok}/{len(results)} OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
