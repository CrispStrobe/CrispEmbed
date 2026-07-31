#!/usr/bin/env python3
"""Measure OCR preprocessing effects on real fixtures.

The harness deliberately keeps preprocessing and OCR measurements separate:
each image stage is materialized first, then (optionally) sent through the
same document pipeline.  This makes a strong recognizer unable to hide a
harmful preprocessing step and makes missing optional models visible.

Examples:
  python3 tests/ocr_preprocessor_benchmark.py --only receipt_example.png
  python3 tests/ocr_preprocessor_benchmark.py --pipeline-binary build/crispembed \
      --det /Volumes/backups/ai/crispembed-gguf/dbnet-ic15-f16.gguf \
      --rec /Volumes/backups/ai/crispembed-gguf/trocr-small-printed-q8_0.gguf
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import struct
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tests/regression/images"
DEFAULT_FIXTURES = ROOT / "tests/regression/corpus_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dimensions(path: Path) -> tuple[int | None, int | None, int | None]:
    """Read common image dimensions without adding a Python imaging dependency."""
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 26:
        return (*struct.unpack(">II", data[16:24]), 4)
    if data.startswith((b"P5", b"P6")):
        fields = re.findall(rb"\d+", data[:256])
        if len(fields) >= 2:
            return int(fields[0]), int(fields[1]), 1 if data[:2] == b"P5" else 3
    if data.startswith(b"\xff\xd8"):
        p = 2
        while p + 9 < len(data):
            if data[p] != 0xFF:
                p += 1
                continue
            marker = data[p + 1]
            p += 2
            if marker in (0xD8, 0xD9):
                continue
            if p + 2 > len(data):
                break
            n = struct.unpack(">H", data[p:p + 2])[0]
            if marker in range(0xC0, 0xC4) and p + 7 <= len(data):
                h, w = struct.unpack(">HH", data[p + 3:p + 7])
                return w, h, data[p + 7] if p + 7 < len(data) else None
            p += n
    return None, None, None


def ppm_stats(data: bytes) -> dict:
    """Parse the small PPM/PGM header and calculate cheap pixel statistics."""
    m = re.match(rb"P([56])\s+(?:#.*\n\s*)?(\d+)\s+(\d+)\s+(\d+)\s", data)
    if not m:
        return {}
    mode, w, h, maxval = int(m[1]), int(m[2]), int(m[3]), int(m[4])
    payload = data[m.end():]
    if not payload:
        return {"width": w, "height": h, "channels": 3 if mode == 6 else 1}
    sample = payload[::max(1, len(payload) // 200000)]
    mean = sum(sample) / len(sample)
    return {"width": w, "height": h, "channels": 3 if mode == 6 else 1,
            "mean": round(mean / maxval, 6),
            "min": min(sample) / maxval, "max": max(sample) / maxval}


def run(cmd: list[str], timeout: float, *, capture_stdout: bool = True) -> dict:
    started = time.perf_counter()
    try:
        p = subprocess.run(cmd, cwd=ROOT,
                           stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=timeout, env=os.environ.copy())
        timed_out = False
    except subprocess.TimeoutExpired as e:
        p = subprocess.CompletedProcess(cmd, 124, e.stdout or b"", e.stderr or b"")
        timed_out = True
    elapsed = (time.perf_counter() - started) * 1000
    stdout = p.stdout or b""
    stderr = p.stderr or b""
    if isinstance(stdout, str): stdout = stdout.encode()
    if isinstance(stderr, str): stderr = stderr.encode()
    return {"ms": round(elapsed, 3), "returncode": p.returncode,
            "timed_out": timed_out, "stdout": stdout,
            "stderr_tail": stderr.decode("utf-8", "replace")[-1200:]}


def normalize(s: str) -> str:
    s = re.sub(r"(?m)^regions=\d+\s+mean_conf=[0-9.]+.*$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def cer(got: str, want: str | None) -> dict:
    if not want:
        return {"status": "unscored", "text_delta": normalize(got)}
    a, b = normalize(got), normalize(want)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    d = prev[-1]
    return {"status": "exact" if a == b else "cer", "exact": a == b,
            "cer": d / max(1, len(b)), "edit_distance": d}


def effect(candidate: dict, baseline: dict | None) -> str:
    """Classify an OCR stage against raw input with conservative margins."""
    if not candidate.get("ocr") or not baseline or not baseline.get("ocr"):
        return "unavailable"
    a, b = candidate["ocr"], baseline["ocr"]
    if a.get("status") != "ok" or b.get("status") != "ok":
        return "error"
    if a.get("cer") is not None and b.get("cer") is not None:
        delta = a["cer"] - b["cer"]
        if delta < -0.01:
            return "helped"
        if delta > 0.01:
            return "harmed"
        return "neutral"
    # Without verified gold text, only call a stage helpful when it improves
    # both confidence and usable text yield; a confidence drop or empty text
    # is harmful.  Otherwise the result stays neutral.
    ac, bc = a.get("mean_confidence") or 0.0, b.get("mean_confidence") or 0.0
    al, bl = len(a.get("text", "").strip()), len(b.get("text", "").strip())
    if not al:
        return "harmed"
    if ac >= bc + 0.03 and al >= max(1, bl - 2):
        return "helped"
    if ac + 0.03 < bc or al + 8 < bl:
        return "harmed"
    return "neutral"


def fixture_paths(args: argparse.Namespace) -> list[Path]:
    if args.only:
        result = []
        for name in args.only:
            p = Path(name)
            if p.is_absolute():
                result.append(p)
            else:
                direct, cc0 = CORPUS / name, CORPUS / "cc0" / name
                result.append(direct if direct.exists() else cc0)
        return result
    manifest = json.loads(DEFAULT_FIXTURES.read_text())
    names = set()
    for values in manifest.get("stages", {}).values():
        names.update(values)
    return [CORPUS / n for n in sorted(names)]


def cleanup_stage(binary: Path, image: Path, flags: list[str], out: Path,
                  timeout: float) -> dict:
    # JSON gives dimensions without mixing binary PPM data into diagnostics.
    meta = run([str(binary), "--json", "--cleanup-only", str(image), *flags], timeout)
    if meta["returncode"] != 0:
        return {"status": "error", "ms": meta["ms"],
                "stderr_tail": meta["stderr_tail"]}
    try:
        info = json.loads(meta["stdout"].decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return {"status": "error", "ms": meta["ms"],
                "reason": "invalid cleanup JSON", "stderr_tail": meta["stderr_tail"]}
    # Re-run without JSON to materialize the exact bytes that were measured.
    material = run([str(binary), "--cleanup-only", str(image), *flags], timeout)
    if material["returncode"] != 0:
        return {"status": "error", "ms": meta["ms"] + material["ms"],
                "stderr_tail": material["stderr_tail"]}
    out.write_bytes(material["stdout"])
    stats = ppm_stats(material["stdout"])
    return {"status": "ok", "ms": round(meta["ms"] + material["ms"], 3),
            "input": {"width": info.get("original_width"),
                      "height": info.get("original_height")},
            "output": stats or {"width": info.get("width"), "height": info.get("height")},
            "checksum": sha256(out), "stderr_tail": material["stderr_tail"]}


def ocr_stage(binary: Path, image: Path, args: argparse.Namespace) -> dict:
    cmd = [str(binary), "--json", "--ocr-pipeline", str(image),
           "--ocr-engine", args.engine, "--ocr-det", str(args.det),
           "--ocr-rec", str(args.rec)]
    result = run(cmd, args.timeout)
    text = result["stdout"].decode("utf-8", "replace")
    row = {"status": "ok" if result["returncode"] == 0 else "error",
           "ms": result["ms"], "stderr_tail": result["stderr_tail"]}
    try:
        payload = json.loads(text)
        row.update({"regions": payload.get("n_regions"),
                    "mean_confidence": payload.get("mean_confidence"),
                    "text": payload.get("full_text", "")})
        row.update(cer(row["text"], args.expected))
    except (ValueError, TypeError):
        row["reason"] = "invalid OCR JSON"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default="build/crispembed")
    ap.add_argument("--only", nargs="*", help="fixture filenames under images/")
    ap.add_argument("--output", default="", help="write JSON report")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--pipeline-binary", default="", help="also run OCR per stage")
    ap.add_argument("--det", default="", help="DB detector GGUF for --pipeline-binary")
    ap.add_argument("--rec", default="", help="recognizer GGUF for --pipeline-binary")
    ap.add_argument("--engine", default="trocr")
    ap.add_argument("--expected", default=None)
    ap.add_argument("--include-dewarp", action="store_true")
    ap.add_argument("--stage", action="append", default=[],
                    help="limit the run to named stages; repeat (default: all)")
    ap.add_argument("--model", action="append", default=[], metavar="STAGE=MODEL",
                    help="optional learned stage: nafnet,pan,dat,hat,safmn,esrgan,swinir,tbsrn")
    args = ap.parse_args()
    binary = Path(args.binary)
    if not binary.exists():
        ap.error(f"missing binary: {binary}")
    if bool(args.pipeline_binary) != bool(args.det and args.rec):
        ap.error("--pipeline-binary requires both --det and --rec")
    models = dict(x.split("=", 1) for x in args.model if "=" in x)
    stages = [("raw", [])]
    stages += [("cleanup", []), ("cleanup-binarize", ["--binarize"]),
               ("cleanup-no-deskew", ["--no-deskew"]),
               ("cleanup-no-crop-borders", ["--no-crop-borders"]),
               ("cleanup-no-whiten", ["--no-whiten"]),
               ("cleanup-no-consensus", ["--no-deskew-consensus"])]
    if args.include_dewarp:
        stages.append(("dewarp", ["--dewarp"]))
    if args.stage:
        wanted = set(args.stage) | {"raw"}
        stages = [(name, flags) for name, flags in stages if name in wanted]
    rows = []
    with tempfile.TemporaryDirectory(prefix="crispembed-preproc-") as tmp:
        tmpdir = Path(tmp)
        for image in fixture_paths(args):
            if not image.exists():
                rows.append({"fixture": str(image), "stage": "raw",
                             "status": "unavailable", "reason": "missing fixture"})
                continue
            start = len(rows)
            iw, ih, ic = dimensions(image)
            expected = args.expected
            for name, flags in stages:
                row = {"fixture": str(image.relative_to(ROOT) if image.is_relative_to(ROOT) else image),
                       "stage": name, "input_checksum": sha256(image),
                       "input": {"width": iw, "height": ih, "channels": ic}}
                if name == "raw":
                    row.update({"status": "ok", "output": row["input"],
                                "checksum": row["input_checksum"], "ms": 0.0})
                    out = image
                elif name == "dewarp":
                    outpath = tmpdir / f"{len(rows)}.pgm"
                    # Classical dewarp writes a PGM to stdout.
                    result = run([str(binary), "--dewarp", str(image)], args.timeout)
                    if result["returncode"] == 0:
                        outpath.write_bytes(result["stdout"])
                        row.update({"status": "ok", "ms": result["ms"],
                                    "output": ppm_stats(result["stdout"]),
                                    "checksum": sha256(outpath),
                                    "stderr_tail": result["stderr_tail"]})
                        out = outpath
                    else:
                        not_applicable = "too few textlines" in result["stderr_tail"].lower()
                        row.update({"status": "unavailable" if not_applicable else "error",
                                    "ms": result["ms"],
                                    "reason": "not applicable: too few textlines" if not_applicable else "dewarp failed",
                                    "stderr_tail": result["stderr_tail"]})
                        out = None
                else:
                    outpath = tmpdir / f"{len(rows)}.ppm"
                    row.update(cleanup_stage(binary, image, flags, outpath, args.timeout))
                    out = outpath if row["status"] == "ok" else None
                if args.pipeline_binary and out is not None and row["status"] == "ok":
                    ocr = ocr_stage(Path(args.pipeline_binary), out, args)
                    row["ocr"] = ocr
                rows.append(row)
                print(f"{row['fixture']:48} {name:26} {row['status']:11} "
                      f"{row.get('ms', 0):8.1f}ms" +
                      (f" cer={row['ocr'].get('cer'):.3f}" if row.get('ocr', {}).get('cer') is not None else ""),
                      flush=True)
            for stage_name, model in models.items():
                # Learned stage invocation is intentionally explicit: each
                # model has a stable CLI flag and cannot be accidentally used
                # merely because an unrelated file is present in the cache.
                flag = {"nafnet": "--nafnet-denoise", "pan": "--pan-sr",
                        "dat": "--dat-sr", "hat": "--hat-sr", "safmn": "--safmn-sr",
                        "esrgan": "--esrgan-sr", "swinir": "--swinir-sr",
                        "tbsrn": "--tbsrn-sr"}.get(stage_name)
                model_flag = {"nafnet": "--nafnet-model", "pan": "--pan-model",
                              "dat": "--dat-model", "hat": "--hat-model",
                              "safmn": "--safmn-model", "esrgan": "--esrgan-model",
                              "swinir": "--swinir-model", "tbsrn": "--tbsrn-model"}.get(stage_name)
                if not flag or not model_flag:
                    continue
                result = {"fixture": str(image), "stage": stage_name,
                          "input_checksum": sha256(image)}
                if not flag or not model_flag:
                    result.update({"status": "unavailable", "reason": "unsupported stage"})
                    rows.append(result)
                    continue
                model_path = Path(model)
                if not model_path.exists():
                    result.update({"status": "unavailable", "reason": "model missing",
                                   "model": str(model_path)})
                    rows.append(result)
                    continue
                outpath = tmpdir / f"{len(rows)}.ppm"
                command = [str(binary), flag, str(image), model_flag, str(model_path)]
                measured = run(command, args.timeout)
                result["ms"] = measured["ms"]
                result["model"] = str(model_path)
                if measured["returncode"] != 0:
                    result.update({"status": "error", "reason": "model stage failed",
                                   "stderr_tail": measured["stderr_tail"]})
                else:
                    outpath.write_bytes(measured["stdout"])
                    result.update({"status": "ok", "output": ppm_stats(measured["stdout"]),
                                   "checksum": sha256(outpath),
                                   "stderr_tail": measured["stderr_tail"]})
                    if args.pipeline_binary:
                        result["ocr"] = ocr_stage(Path(args.pipeline_binary), outpath, args)
                rows.append(result)
            baseline = next((r for r in rows[start:] if r.get("stage") == "raw"), None)
            for result in rows[start:]:
                result["effect_vs_raw"] = "baseline" if result is baseline else effect(result, baseline)
    report = {"version": 1, "binary": str(binary), "rows": rows,
              "policy": "raw and each preprocessor are measured independently; missing stages are explicit"}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"completed={sum(r.get('status') == 'ok' for r in rows)} "
          f"unavailable={sum(r.get('status') == 'unavailable' for r in rows)} "
          f"errors={sum(r.get('status') == 'error' for r in rows)}")
    return 0 if not any(r.get("status") == "error" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
