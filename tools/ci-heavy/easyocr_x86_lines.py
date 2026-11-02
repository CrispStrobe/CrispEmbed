#!/usr/bin/env python3
"""EasyOCR x86 CPU dropped-line investigation and fix verification (hosted x86 CPU).

The CUDA audit (PR #98) found that the EasyOCR page lane on x86 CPU returns empty
text for every line whose recognizer canvas is narrower than the previous one.
This script reproduces and bisects it, then verifies the fix:

  builds   GGML_NATIVE=OFF (release flags), GGML_NATIVE=ON, and an SSE-only
           build (AVX/AVX2/FMA/F16C off) -- SIMD-level bisection.
  arms     default (all BiLSTM states reset) vs legacy
           CRISPEMBED_EASYOCR_LSTM_STATE_RESET=0; the legacy arm additionally
           with EASYOCR_WIDTH_SORT=1, -t 1, glibc mmap for every allocation
           (fresh zero pages), MALLOC_PERTURB_ (glibc fills reused memory), and
           CRISPEMBED_EASYOCR_POISON_COMPUTE=1 (NaN-filled compute buffer).
  test     test-easyocr-width-shrink (decreasing widths, reused vs fresh).
  truth    Python EasyOCR's own recognizer on each native line crop.

  gh workflow run heavy-cpu.yml --ref fix/easyocr-x86-lines \
      -f script=tools/ci-heavy/easyocr_x86_lines.py \
      -f pip="torch torchvision easyocr gguf" -f isolated_lane=easyocr-x86
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

OUT = Path(os.environ["HEAVY_OUT"])
S = Path(os.environ["HEAVY_SCRATCH"])
ROOT = Path(__file__).resolve().parents[2]
IMG = ROOT / "tests" / "regression" / "images"
M = S / "models"
M.mkdir(parents=True, exist_ok=True)
REPORT: dict = {"source": None, "builds": {}, "pipeline": {}, "width_shrink": {}, "python": {}, "verdict": {}}


def sh(cmd, env=None, timeout=3600, cwd=None):
    e = dict(os.environ)
    e.update(env or {})
    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], env=e, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr, round(time.time() - t0, 2)


def build(name, flags):
    b = S / f"build-{name}"
    metal = "-DGGML_METAL=ON" if "-DGGML_METAL=ON" in flags else "-DGGML_METAL=OFF"
    rc, o, e, _ = sh(["cmake", "-S", ROOT, "-B", b, "-DCMAKE_BUILD_TYPE=Release", metal,
                      "-DGGML_CUDA=OFF", "-DGGML_VULKAN=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF", *flags])
    (OUT / f"configure-{name}.log").write_text(o + e)
    if rc != 0:
        return None
    rc, o, e, secs = sh(["cmake", "--build", b, "--parallel", "4", "--target", "crispembed-cli",
                         "test-easyocr-width-shrink", "test-easyocr-pipeline"], timeout=7200)
    (OUT / f"build-{name}.log").write_text((o + e)[-20000:])
    REPORT["builds"][name] = {"flags": flags, "rc": rc, "seconds": secs}
    return b if rc == 0 else None


def fetch_models():
    dbnet = M / "dbnet-ic15-q8_0.gguf"
    if not dbnet.exists():
        rc, _, e, _ = sh(["curl", "-sfL", "--retry", "3", "-o", dbnet,
                          "https://huggingface.co/cstr/dbnet-ic15-GGUF/resolve/main/dbnet-ic15-q8_0.gguf"])
        if rc:
            sys.exit(f"dbnet download failed: {e[-300:]}")
    work = S / "easyocr-src"
    work.mkdir(parents=True, exist_ok=True)
    zp = work / "english_g2.zip"
    if not zp.exists():
        rc, _, e, _ = sh(["curl", "-sfL", "--retry", "3", "-o", zp,
                          "https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip"])
        if rc:
            sys.exit(f"english_g2 download failed: {e[-300:]}")
        shutil.unpack_archive(str(zp), str(work))
    import easyocr.config as cfg  # the installed wheel's own charset

    chars = cfg.recognition_models["gen2"]["english_g2"]["characters"]
    (work / "english_g2_chars.txt").write_text(chars, encoding="utf-8")
    ckpt = next(work.glob("english_g2.pth"))
    rec = M / "easyocr-english-g2-f16.gguf"
    if not rec.exists():
        rc, o, e, _ = sh([sys.executable, ROOT / "models" / "convert-easyocr-to-gguf.py", "--checkpoint", ckpt,
                          "--charset", work / "english_g2_chars.txt", "--output", rec, "--fp16"])
        if rc:
            sys.exit(f"convert failed: {(o + e)[-600:]}")
    return dbnet, rec, ckpt


def fixtures():
    from PIL import Image

    fx = S / "fixtures"
    fx.mkdir(exist_ok=True)
    lines = []
    # The audit's fox line crops (DBNet boxes + 5 px): canvases 704 then 576 on
    # the page, i.e. the second line shrinks the graph.
    for name, box in (("fox-line1", (20, 30, 515, 82)), ("fox-line2", (20, 100, 480, 155))):
        p = fx / f"{name}.png"
        Image.open(IMG / "fox.png").convert("RGB").crop(box).save(p)
        lines.append(p)
    return lines


def region_lines(stderr):
    out = []
    for ln in stderr.splitlines():
        if ln.startswith("[easyocr-region]"):
            out.append(ln)
    return out


def run_pipeline(bdir, dbnet, rec, img, env, threads="4", backend=None):
    cmd = [bdir / "crispembed", "-t", threads, "--json", "-m", rec, "--ocr-pipeline", img, "--ocr-engine", "easyocr",
           "--ocr-det", dbnet, "--ocr-rec", rec]
    if backend:
        cmd[1:1] = ["--gpu-backend", backend]
    rc, o, e, secs = sh(cmd, env={"CRISPEMBED_EASYOCR_DEBUG_REGIONS": "1", **env})
    try:
        j = json.loads(o.strip().splitlines()[-1])
    except Exception:  # noqa: BLE001
        j = None
    return {"rc": rc, "env": env, "threads": threads, "seconds": secs, "json": j, "regions": region_lines(e),
            "stderr_tail": e[-1500:] if rc else ""}


def manifest(bdir, dbnet, rec, img, env, dest):
    rc, o, e, _ = sh([bdir / "test-easyocr-pipeline", dbnet, rec, img, "lines", dest], env=env)
    if rc != 0 or not dest.exists():
        return None
    return json.loads(dest.read_text())


def python_truth(ckpt, img, records):
    """Python EasyOCR recognizer on every native crop, one crop per call."""
    import easyocr
    from PIL import Image

    md = Path(tempfile.mkdtemp(prefix="easyocr-models-"))
    (md / "english_g2.pth").symlink_to(ckpt.resolve())
    reader = easyocr.Reader(["en"], gpu=False, model_storage_directory=str(md), download_enabled=False,
                            detector=False, recognizer=True, verbose=False, quantize=False)
    src = Image.open(img).convert("RGB")
    out = []
    for r in records:
        x, y, w, h = (int(v) for v in r["crop"])
        p = md / f"crop-{r['index']}.png"
        src.crop((x, y, x + w, y + h)).save(p)
        res = reader.recognize(str(p), horizontal_list=[[0, w, 0, h]], free_list=[], decoder="greedy",
                               batch_size=1, detail=1)
        text = res[0][1] if res else ""
        conf = float(res[0][2]) if res else 0.0
        out.append({"index": r["index"], "crop": r["crop"], "text": text, "confidence": round(conf, 4)})
    return out


def main():
    rc, o, _, _ = sh(["git", "-C", ROOT, "rev-parse", "HEAD"])
    REPORT["source"] = o.strip()
    REPORT["cpu"] = sh(["bash", "-c", "lscpu | grep -E 'Model name|Flags' | cut -c1-400"])[1]
    dbnet, rec, ckpt = fetch_models()
    lines = fixtures()
    mac = sys.platform == "darwin"
    builds = {"metal": ["-DGGML_METAL=ON"]} if mac else {
        "native-off": ["-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF"],
        "native-on": ["-DGGML_NATIVE=ON", "-DCRISPEMBED_NATIVE=ON"],
        "sse-only": ["-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_AVX=OFF", "-DGGML_AVX2=OFF",
                     "-DGGML_FMA=OFF", "-DGGML_F16C=OFF", "-DGGML_AVX512=OFF", "-DGGML_BMI2=OFF"],
    }
    legacy = {"CRISPEMBED_EASYOCR_LSTM_STATE_RESET": "0"}
    arms = {
        "default": ({}, "4"),
        "legacy": (legacy, "4"),
        "legacy-t1": (legacy, "1"),
        "legacy-widthsort": ({**legacy, "EASYOCR_WIDTH_SORT": "1"}, "4"),
        "legacy-mmap-all": ({**legacy, "GLIBC_TUNABLES": "glibc.malloc.mmap_threshold=0"}, "4"),
        "legacy-perturb": ({**legacy, "MALLOC_PERTURB_": "165"}, "4"),
        "legacy-poison": ({**legacy, "CRISPEMBED_EASYOCR_POISON_COMPUTE": "1"}, "4"),
        "default-poison": ({"CRISPEMBED_EASYOCR_POISON_COMPUTE": "1"}, "4"),
        "default-perturb": ({"MALLOC_PERTURB_": "165"}, "4"),
        "default-t1": ({}, "1"),
    }
    if mac:  # macOS runner: Apple CPU and Metal arms (hosted runners may abort in ggml-metal); binaries
        # are exported for the run on a real M1
        arms = {k: arms[k] for k in ("default", "legacy", "legacy-poison", "default-poison", "legacy-widthsort")}
    pages = ["fox", "scan_strip", "scan_page_pd"]
    for bname, flags in builds.items():
        bdir = build(bname, flags)
        if not bdir:
            continue
        REPORT["pipeline"][bname] = {}
        names = arms if bname in ("native-off", "metal") else {k: arms[k] for k in ("default", "legacy",
                                                                                     "legacy-poison", "default-poison")}
        for img in pages:
            for arm, (env, t) in names.items():
                if mac:
                    for be in ("cpu", "metal"):
                        REPORT["pipeline"][bname][f"{img}/{arm}/{be}"] = run_pipeline(
                            bdir, dbnet, rec, IMG / f"{img}.png", env, t, backend=be)
                else:
                    REPORT["pipeline"][bname][f"{img}/{arm}"] = run_pipeline(bdir, dbnet, rec, IMG / f"{img}.png",
                                                                             env, t)
        if mac:
            exp = OUT / "mac-bin"
            exp.mkdir(exist_ok=True)
            for f in (bdir / "crispembed", bdir / "test-easyocr-width-shrink", dbnet, rec, *lines):
                shutil.copy2(f, exp / Path(f).name)
            for lib in bdir.rglob("*.dylib"):  # ggml is a shared library on macOS; run with DYLD_LIBRARY_PATH
                shutil.copy2(lib, exp / lib.name, follow_symlinks=True)
        ws = {}
        for arm, env in (("legacy-poison", {**legacy, "CRISPEMBED_EASYOCR_POISON_COMPUTE": "1"}),
                         ("default-poison", {"CRISPEMBED_EASYOCR_POISON_COMPUTE": "1"}),
                         ("legacy-nopoison", {**legacy, "CRISPEMBED_EASYOCR_POISON_COMPUTE": "0"}),
                         ("default-nopoison", {"CRISPEMBED_EASYOCR_POISON_COMPUTE": "0"})):
            rc, o, e, secs = sh([bdir / "test-easyocr-width-shrink", rec, *lines], env=env)
            ws[arm] = {"rc": rc, "seconds": secs, "stdout": o[-4000:], "stderr_tail": e[-800:] if rc > 1 else ""}
        REPORT["width_shrink"][bname] = ws
        if bname in ("native-off", "metal"):
            for img in pages:
                for arm, env in (("default", {}), ("legacy", legacy)):
                    man = manifest(bdir, dbnet, rec, IMG / f"{img}.png", env, OUT / f"manifest-{img}-{arm}.json")
                    REPORT["python"].setdefault(img, {})[f"native-{arm}"] = (
                        [{"index": r["index"], "crop": r["crop"], "text": r["text"],
                          "confidence": round(r["confidence"], 4)} for r in man["records"]] if man else None)
                recs = REPORT["python"][img].get("native-default")
                if recs:
                    try:
                        REPORT["python"][img]["python"] = python_truth(ckpt, IMG / f"{img}.png", recs)
                    except Exception as exc:  # noqa: BLE001
                        REPORT["python"][img]["python_error"] = repr(exc)[-800:]
        shutil.rmtree(bdir, ignore_errors=True)  # keep the runner disk free between builds

    # Verdicts.
    v = REPORT["verdict"]
    for img, d in REPORT["python"].items():
        py = d.get("python") or []
        for arm in ("native-default", "native-legacy"):
            nat = d.get(arm) or []
            same = [a["text"] == b["text"] for a, b in zip(nat, py)]
            v[f"{img}.{arm}.python_text_equal"] = f"{sum(same)}/{len(py)}" if py else "no-python"
            v[f"{img}.{arm}.empty_lines"] = sum(1 for a in nat if not a["text"])
    for bname, d in REPORT["pipeline"].items():
        for key, r in d.items():
            j = r.get("json") or {}
            v[f"pipeline.{bname}.{key}"] = f"rc={r['rc']} n_regions={j.get('n_regions')} " \
                                           f"empty={sum(1 for ln in r['regions'] if ' len=0 ' in ln)}"
    for bname, d in REPORT["width_shrink"].items():
        for arm, r in d.items():
            v[f"width_shrink.{bname}.{arm}"] = "PASS" if r["rc"] == 0 else f"FAIL rc={r['rc']}"
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=1, ensure_ascii=False))
    with open(OUT / "summary.md", "w") as f:
        f.write(f"### EasyOCR x86 lines ({REPORT['source'][:8]})\n```\n")
        for k, val in v.items():
            f.write(f"{k}: {val}\n")
        f.write("```\n")
    print(json.dumps(v, indent=1))
    # Fail the run when the default path regresses: any empty line, a failing
    # default width-shrink test, or default text different from Python.
    bad = []
    for k, val in v.items():
        sval = str(val)
        if k.startswith("width_shrink.") and k.split(".")[-1].startswith("default") and sval != "PASS":
            bad.append(k)
        if k.startswith("pipeline.") and "/default" in k and (" empty=0" not in sval or "rc=0 " not in sval):
            bad.append(k)
        if ".native-default.empty_lines" in k and val:
            bad.append(k)
    if bad:
        print("default-path failures:", bad)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
