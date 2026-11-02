#!/usr/bin/env python3
"""EasyOCR native vs Python parity, and page reading order (hosted x86 CPU).

Residual gaps after PR #100 (PLAN): native DBNet + EasyOCR line text matched
Python EasyOCR's own recognizer on the same crops for only 1/2 (fox), 6/11
(scan_strip) and 21/36 (scan_page_pd) lines, and scan_page_pd's page text
differed between the M1 and x86/CUDA.

  references  tools/easyocr_python_parity.py per fixture: Reader(['en'])
              readtext end to end (int8 default and F32), Python recognize()
              on every native crop (path and array input), and per-stage
              preprocessing variants.
  recognizer  test-easyocr-pipeline manifests, F16 and F32 GGUF, arms:
                legacy   FLOOR_WIDTH=0 CONTRAST_RETRY=0 (pre-fix behaviour)
                floor    CONTRAST_RETRY=0
                retry    FLOOR_WIDTH=0
                default  (int width + contrast retry)
                libpng   default + LIBPNG_GREY=1 (readtext(<png path>) grey)
  page lane   crispembed --ocr-pipeline --ocr-engine easyocr on the regression
              and cc0 ground-truth fixtures: reading order legacy vs rows,
              recognizer legacy vs default, cleanup on vs off; CER vs ground
              truth where it exists.

  gh workflow run heavy-cpu.yml --ref fix/easyocr-python-parity \
      -f script=tools/ci-heavy/easyocr_python_parity.py \
      -f pip="torch torchvision easyocr gguf" -f isolated_lane=easyocr-parity
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(os.environ["HEAVY_OUT"])
S = Path(os.environ["HEAVY_SCRATCH"])
ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "tests" / "regression" / "images"
M = S / "models"
M.mkdir(parents=True, exist_ok=True)
REPORT: dict = {"source": None, "build": {}, "python": {}, "recognizer": {}, "page": {}, "verdict": {}}
FIXTURES = ["fox", "scan_strip", "scan_page_pd"]
PAGES = [IMG / f"{f}.png" for f in FIXTURES] + [
    IMG / "cc0" / n for n in ("commons_test_ocr_document.jpg", "receipt_historical.png", "commons_example_receipt.png",
                              "simple_form.png", "german_official_print.jpg")]


def sh(cmd, env=None, timeout=3600):
    e = dict(os.environ)
    e.update(env or {})
    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], env=e, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr, round(time.time() - t0, 2)


def build():
    b = S / "build"
    rc, o, e, _ = sh(["cmake", "-S", ROOT, "-B", b, "-DCMAKE_BUILD_TYPE=Release", "-DGGML_METAL=OFF", "-DGGML_CUDA=OFF",
                      "-DGGML_VULKAN=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF", "-DGGML_NATIVE=OFF",
                      "-DCRISPEMBED_NATIVE=OFF"])
    (OUT / "configure.log").write_text(o + e)
    if rc:
        sys.exit("configure failed")
    rc, o, e, secs = sh(["cmake", "--build", b, "--parallel", "4", "--target", "crispembed-cli",
                         "test-easyocr-pipeline", "test-easyocr-postprocess"], timeout=7200)
    (OUT / "build.log").write_text((o + e)[-20000:])
    REPORT["build"] = {"rc": rc, "seconds": secs}
    if rc:
        sys.exit("build failed")
    rc, o, e, _ = sh([b / "test-easyocr-postprocess"])
    REPORT["build"]["unit"] = (o + e).strip()[-300:]
    return b


def fetch_models():
    dbnet = M / "dbnet-ic15-q8_0.gguf"
    if not dbnet.exists():
        rc, _, e, _ = sh(["curl", "-sfL", "--retry", "3", "-o", dbnet,
                          "https://huggingface.co/cstr/dbnet-ic15-GGUF/resolve/main/dbnet-ic15-q8_0.gguf"])
        if rc:
            sys.exit(f"dbnet download failed: {e[-300:]}")
    md = S / "easyocr-models"
    md.mkdir(parents=True, exist_ok=True)
    for name, url in (("english_g2", "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"),
                      ("craft_mlt_25k", "https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip")):
        if not (md / f"{name}.pth").exists():
            zp = md / f"{name}.zip"
            rc, _, e, _ = sh(["curl", "-sfL", "--retry", "3", "-o", zp, url])
            if rc:
                sys.exit(f"{name} download failed: {e[-300:]}")
            shutil.unpack_archive(str(zp), str(md))
    import easyocr.config as cfg

    chars = M / "english_g2_chars.txt"
    chars.write_text(cfg.recognition_models["gen2"]["english_g2"]["characters"], encoding="utf-8")
    recs = {}
    for prec, extra in (("f16", ["--fp16"]), ("f32", [])):
        rec = M / f"easyocr-english-g2-{prec}.gguf"
        if not rec.exists():
            rc, o, e, _ = sh([sys.executable, ROOT / "models" / "convert-easyocr-to-gguf.py", "--checkpoint",
                              md / "english_g2.pth", "--charset", chars, "--output", rec, *extra])
            if rc:
                sys.exit(f"convert failed: {(o + e)[-600:]}")
        recs[prec] = rec
    return dbnet, recs, md


ARMS = {
    "legacy": {"CRISPEMBED_EASYOCR_FLOOR_WIDTH": "0", "CRISPEMBED_EASYOCR_CONTRAST_RETRY": "0"},
    "floor": {"CRISPEMBED_EASYOCR_CONTRAST_RETRY": "0"},
    "retry": {"CRISPEMBED_EASYOCR_FLOOR_WIDTH": "0"},
    "default": {},
    "libpng": {"CRISPEMBED_EASYOCR_LIBPNG_GREY": "1"},
}


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    rc, o, _, _ = sh(["git", "-C", ROOT, "rev-parse", "HEAD"])
    REPORT["source"] = o.strip()
    REPORT["cpu"] = sh(["bash", "-c", "lscpu | grep -E 'Model name' | cut -c1-200"])[1].strip()
    import cv2

    REPORT["cv2_hal"] = [ln.strip() for ln in cv2.getBuildInformation().splitlines() if "HAL" in ln or "IPP" in ln]
    b = build()
    dbnet, recs, md = fetch_models()
    v = REPORT["verdict"]

    # 1. Native manifests per arm.
    for f in FIXTURES:
        for prec, rec in recs.items():
            for arm, env in ARMS.items():
                dest = OUT / f"manifest-{f}-{prec}-{arm}.json"
                r = sh([b / "test-easyocr-pipeline", dbnet, rec, IMG / f"{f}.png", "lines", dest], env=env)
                if r[0] or not dest.exists():
                    v[f"manifest.{f}.{prec}.{arm}"] = f"FAIL rc={r[0]}"

    # 2. Python references on the default-arm crops (crops are arm-independent).
    for f in FIXTURES:
        dest = OUT / f"python-{f}.json"
        r = sh([sys.executable, ROOT / "tools" / "easyocr_python_parity.py", "--image", IMG / f"{f}.png", "--manifest",
                OUT / f"manifest-{f}-f16-default.json", "--model-dir", md, "--output", dest], timeout=5400)
        REPORT["python"][f] = {"rc": r[0], "seconds": r[3], "stderr": r[2][-800:] if r[0] else ""}
        if r[0]:
            continue
        py = json.loads(dest.read_text())["crops"]
        for prec in recs:
            for arm in ARMS:
                mp = OUT / f"manifest-{f}-{prec}-{arm}.json"
                if not mp.exists():
                    continue
                nat = json.loads(mp.read_text())["records"]
                same_crops = [n["recognizer_crop"] for n in nat] == [p["crop"] for p in py]
                for ref in ("python_rgb_array", "python_path"):
                    eq = sum(n["text"] == p[ref]["text"] for n, p in zip(nat, py))
                    dc = [abs(n["confidence"] - p[ref]["confidence"]) for n, p in zip(nat, py)
                          if n["text"] == p[ref]["text"]]
                    v[f"lines.{f}.{prec}.{arm}.vs_{ref}"] = (
                        f"{eq}/{len(py)} max_dconf={max(dc) if dc else float('nan'):.5f} same_crops={same_crops}")

    # 3. Page lane A/B.
    gt = {r["file"]: r["text"] for r in json.loads((IMG / "cc0" / "ground_truth.json").read_text())["records"]}
    rec = recs["f16"]
    page_arms = {
        "legacy-all": {"CRISPEMBED_OCR_READING_ORDER_ROWS": "0", **ARMS["legacy"]},
        "rows-only": ARMS["legacy"],
        "default": {},
        "default-nocleanup": {"CRISPEMBED_EASYOCR_CLEANUP": "0"},
    }
    for img in PAGES:
        for arm, env in page_arms.items():
            r = sh([b / "crispembed", "-t", "4", "--json", "-m", rec, "--ocr-pipeline", img, "--ocr-engine", "easyocr",
                    "--ocr-det", dbnet, "--ocr-rec", rec], env=env)
            try:
                j = json.loads(r[1].strip().splitlines()[-1])
            except Exception:  # noqa: BLE001
                j = {"error": r[2][-400:]}
            item = {"rc": r[0], "seconds": r[3], "json": j}
            if img.name in gt and "full_text" in j:
                g = gt[img.name]
                item["cer"] = round(lev(j["full_text"], g) / max(1, len(g)), 4)
            REPORT["page"].setdefault(img.name, {})[arm] = item
            v[f"page.{img.name}.{arm}"] = (f"rc={r[0]} n={j.get('n_regions')} cer={item.get('cer')} "
                                           f"sha={hash_text(j.get('full_text', ''))}")
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=1, ensure_ascii=False))
    with open(OUT / "summary.md", "w") as fh:
        fh.write(f"### EasyOCR Python parity ({REPORT['source'][:8]}, {REPORT['cpu']})\n```\n")
        for k, val in v.items():
            fh.write(f"{k}: {val}\n")
        fh.write("```\n")
    print((OUT / "summary.md").read_text())
    return 1 if any("FAIL" in str(x) for x in v.values()) else 0


def hash_text(s):
    import hashlib

    return hashlib.sha256(s.encode()).hexdigest()[:8]


if __name__ == "__main__":
    sys.exit(main())
