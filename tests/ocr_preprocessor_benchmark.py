#!/usr/bin/env python3
"""Measure OCR before and after the available classical preprocessing stages.

The harness deliberately uses the real CLI twice per row: ``--cleanup-only``
produces the transformed PNM, then the normal detector/recognizer consumes
that PNM.  This makes preprocessing effects observable instead of inferring
them from pixel statistics alone.

Example:
  python3 tests/ocr_preprocessor_benchmark.py \
    --models-dir /Volumes/backups/ai/crispembed-gguf \
    --build-dir build --image tests/regression/images/cc0/receipt_example.png \
    --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from difflib import SequenceMatcher




def run(cmd: list[str], env: dict[str, str], binary: bool = False):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          env=env, check=False, text=not binary)


def normalise(text: str) -> str:
    return re.sub(r"[^\w]+", "", text, flags=re.UNICODE).lower()


def delta(raw: str, value: str) -> float | None:
    a, b = normalise(raw), normalise(value)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def image_info(path: Path) -> dict:
    try:
        from PIL import Image  # optional benchmark-only dependency
        with Image.open(path) as im:
            pixels = list(im.convert("L").getdata())
            return {"width": im.width, "height": im.height, "channels": len(im.getbands()),
                    "mean_gray": sum(pixels) / len(pixels),
                    "min_gray": min(pixels), "max_gray": max(pixels),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    except Exception:  # noqa: BLE001 - metadata is best effort
        return {"width": None, "height": None, "channels": None,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None}


def parse_results(stdout: str) -> tuple[str, int]:
    texts = []
    for line in stdout.splitlines():
        m = re.search(r'conf=[0-9.]+\s+"(.*)"$', line)
        if m:
            texts.append(m.group(1))
    if not texts and "(no text detected)" not in stdout:
        texts = [line.strip() for line in stdout.splitlines() if line.strip()]
    return "\n".join(texts), len(texts)


def outcome(row: dict, raw: dict) -> str:
    if row.get("status") != "ok":
        return "error"
    if row.get("cleanup") == ["raw"]:
        return "neutral"
    change = row.get("text_delta_vs_raw")
    if change is None:
        return "unavailable"
    # Without a verified transcription, only a near-identical output is safe
    # to call neutral. A changed output needs gold text before it can be
    # promoted to helped or harmed.
    return "neutral" if change <= 0.02 else "unavailable"


def one(cli: Path, det: Path, rec: Path, image: Path, env: dict[str, str], cleanup: list[str], tmp: Path) -> dict:
    source = image
    generated = None
    if cleanup:
        p = run([str(cli), "--cleanup-only", str(image), *cleanup], env, binary=True)
        # PNG by default, raw Netpbm under CRISPEMBED_IMAGE_FORMAT=ppm. Accept
        # both and name the file for what it actually is — PIL reads either, and
        # a .pnm holding PNG bytes would mislead anyone inspecting the tmp dir.
        magic = p.stdout[:8]
        is_png = magic.startswith(b"\x89PNG\r\n\x1a\n")
        is_pnm = magic.startswith((b"P5", b"P6"))
        if p.returncode == 0 and (is_png or is_pnm):
            generated = tmp / (image.stem + "-" + "-".join(cleanup[1:]) + (".png" if is_png else ".pnm"))
            generated.write_bytes(p.stdout)
            source = generated
        else:
            return {"status": "error", "cleanup": cleanup, "reason": "cleanup-only failed",
                    "returncode": p.returncode, "stderr_tail": p.stderr.decode(errors="replace")[-800:]}
    started = time.monotonic()
    p = run([str(cli), "-m", str(rec), "--ocr-det", str(det), "--ocr-rec", str(rec), "--ocr", str(source)], env)
    elapsed = (time.monotonic() - started) * 1000.0
    text, regions = parse_results(p.stdout)
    return {"status": "ok" if p.returncode == 0 else "error", "cleanup": cleanup or ["raw"],
            "source": str(source), "elapsed_ms": round(elapsed, 2), "regions": regions,
            "output": image_info(source),
            "text": text, "returncode": p.returncode,
            "stderr_tail": p.stderr[-800:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", required=True)
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--det", default="dbnet-ic15-f16.gguf")
    ap.add_argument("--rec", default="trocr-small-printed-q8_0.gguf")
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--include-derived", action="store_true",
                    help="also benchmark deterministic CC0/PD-derived robustness fixtures")
    ap.add_argument("--corpus-manifest", default="tests/regression/corpus_manifest.json")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    images = [Path(x) for x in args.image]
    if not images:
        corpus = json.loads((root / args.corpus_manifest).read_text())
        images = [root / "tests/regression/images" / value
                  for values in corpus["stages"].values() for value in values]
    if args.include_derived:
        derived_manifest = root / "tests/regression/images/derived/MANIFEST.json"
        if derived_manifest.exists():
            derived = json.loads(derived_manifest.read_text())
            images.extend(root / record["file"] for record in derived["records"])
    images = list(dict.fromkeys(images))
    cli = Path(args.build_dir) / "crispembed"
    det, rec = Path(args.models_dir) / args.det, Path(args.models_dir) / args.rec
    if not cli.exists() or not det.exists() or not rec.exists():
        ap.error(f"missing CLI/model: {cli}, {det}, or {rec}")
    env = os.environ.copy()
    rows = []
    with tempfile.TemporaryDirectory(prefix="crispembed-preproc-") as work:
        tmp = Path(work)
        for image in images:
            if not image.exists():
                rows.append({"image": str(image), "status": "unavailable", "reason": "missing fixture"})
                continue
            raw = one(cli, det, rec, image, env, [], tmp)
            raw_text = raw.get("text", "")
            variants = [raw, one(cli, det, rec, image, env, ["--cleanup"], tmp),
                        one(cli, det, rec, image, env, ["--cleanup", "--binarize"], tmp)]
            for row in variants:
                row["image"] = str(image)
                row["input"] = image_info(image)
                row["text_delta_vs_raw"] = delta(raw_text, row.get("text", ""))
                row["outcome"] = outcome(row, raw)
                rows.append(row)
    result = {"version": 1, "detector": str(det), "recognizer": str(rec), "rows": rows}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for row in rows:
            print(f"{row['image']} {','.join(row['cleanup'])}: {row['status']} "
                  f"{row.get('elapsed_ms', 0):.1f} ms regions={row.get('regions', 0)} "
                  f"delta={row.get('text_delta_vs_raw')}")
    return 0 if all(row["status"] in {"ok", "unavailable"} for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
