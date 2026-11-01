#!/usr/bin/env python3
"""OXR-1.0 CUDA variants on a Kaggle GPU: F16 (published), Q8_0 and Q4_K (quantized here).

    python tools/kaggle/oxr-gpu/run.py BUILD_DIR

For every committed case in tests/results/oxr/hosted-parity.json, runs the native
CUDA runner (CRISPEMBED_OXR_GPU=1) with the exact reference request and compares
its token IDs with the original model's. Writes /kaggle/working/oxr-gpu/report.json
after every step. Native output goes to files (never unread pipes) and every step
has a timeout, so one hang cannot stall the session.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_VERSION = "oxr-kaggle-gpu-v1"
F16_URL = ("https://huggingface.co/cstr/OXR-1.0-GGUF/resolve/f1e262e276349aa2647a85da6fc6326dd6aaf271/"
           "oxr-1.0-f16.gguf")
F16_SHA = "eac8754f4886ba4197416083d49089a3d86c8644f0e5650f44fcf69059d81a57"
QUANTS = ("q8_0", "q4_k")
SCRATCH = Path("/kaggle/temp/oxr")
OUT = Path("/kaggle/working/oxr-gpu")
ENV = dict(os.environ, CRISPEMBED_OXR="1", CRISPEMBED_ACCEPT_OXR_LICENSE="1", CRISPEMBED_OXR_GPU="1")
REPORT = {"script_version": SCRIPT_VERSION, "commit": os.environ.get("OXR_COMMIT"),
          "gpu": os.environ.get("OXR_GPU"), "variants": {}, "cases": {}}


def save(phase):
    REPORT["phase"] = phase
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=2, ensure_ascii=False) + "\n")
    print(f"== {phase}", flush=True)


def run(command, timeout=3600, env=None):
    """Run with stdout/stderr in files; returns (returncode, stdout, stderr, seconds)."""
    print("+", *command, flush=True)
    out_p, err_p = SCRATCH / "step.out", SCRATCH / "step.err"
    started = time.monotonic()
    with open(out_p, "w") as out_f, open(err_p, "w") as err_f:
        try:
            rc = subprocess.run([str(c) for c in command], stdout=out_f, stderr=err_f, env=env or ENV,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = -9
    return rc, out_p.read_text(errors="replace"), err_p.read_text(errors="replace"), time.monotonic() - started


def sha256(path):
    rc, out, _, _ = run(["sha256sum", path], timeout=600)
    return out.split()[0] if rc == 0 and out else None


def main(argv):
    build = Path(argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    runner = next(build.rglob("test-oxr-run"))
    quant = next(build.rglob("crispembed-quantize"))
    parity = json.loads(Path("tests/results/oxr/hosted-parity.json").read_text())
    save("setup")

    models = {}
    f16 = SCRATCH / "oxr-1.0-f16.gguf"
    rc, _, err, secs = run(["curl", "-sfL", "-o", f16, F16_URL], timeout=3600)
    ok = rc == 0 and sha256(f16) == F16_SHA
    REPORT["variants"]["f16"] = {"source": F16_URL, "sha256_ok": ok, "seconds": round(secs, 1), "error": err[-500:]}
    if ok:
        models["f16"] = f16
    save("f16 download")

    rc, out, err, secs = run([sys.executable, "-c", (
        "import json,sys\nfrom huggingface_hub import snapshot_download\n"
        "m=json.load(open('models/oxr-source.json'))\n"
        "snapshot_download(m['model_id'],revision=m['revision'],allow_patterns=list(m['files']),"
        "local_dir=sys.argv[1])"), SCRATCH / "checkpoint"], timeout=3600)
    f32 = SCRATCH / "oxr-1.0-f32.gguf"
    if rc == 0:
        rc, out, err, secs = run([sys.executable, "models/convert-oxr-to-gguf.py", "--model", SCRATCH / "checkpoint",
                                  "--dtype", "f32", "--output", f32], timeout=3600)
    REPORT["variants"]["f32-source"] = {"returncode": rc, "seconds": round(secs, 1), "log_tail": (out + err)[-800:]}
    save("f32 conversion")
    if rc == 0:
        for q in QUANTS:
            path = SCRATCH / f"oxr-1.0-{q}.gguf"
            rc, out, err, secs = run([quant, f32, path, q], timeout=3600)
            REPORT["variants"][q] = {"returncode": rc, "bytes": path.stat().st_size if path.exists() else 0,
                                     "log_tail": (out + err)[-600:]}
            if rc == 0:
                models[q] = path
            save(f"quantize {q}")
        f32.unlink(missing_ok=True)

    from PIL import Image
    for case in parity["cases"]:
        cid, lane = case["id"], case["lane"]
        ref = json.loads(Path(f"tests/results/oxr/{lane}/{cid}/python.json").read_text())
        image = SCRATCH / f"{cid}.png"
        source = Path("tests/regression/images") / {
            "formula": "formula_quadratic.png", "table": "cc0/simple_table.jpg", "japanese": "japanese_print.png",
            "receipt": "cc0/receipt_example.png", "german": "cc0/german_official_print.jpg"}[cid]
        Image.open(source).convert("RGB").save(image)  # decode JPEGs once, as the reference did
        entry = REPORT["cases"].setdefault(cid, {"reference_tokens": len(case["reference_ids"]), "arms": {}})
        for name, model in models.items():
            rc, out, err, secs = run([runner, "--model", model, "--image", image, "--prompt", ref["prompt"],
                                      "--max-tokens", ref["max_tokens"], "--threads", "4", "--repeat", "2",
                                      "--repetition-penalty", ref["repetition_penalty"], "--hf-generation", "1",
                                      "--min-pixels", ref["min_pixels"], "--max-pixels", ref["max_pixels"]],
                                     timeout=1800)
            arm = {"returncode": rc, "seconds": round(secs, 2)}
            try:
                native = json.loads(out)
                ids = native.get("token_ids", [])
                diff = next((i for i, (a, b) in enumerate(zip(ids, case["reference_ids"])) if a != b), None)
                if diff is None and len(ids) != len(case["reference_ids"]):
                    diff = min(len(ids), len(case["reference_ids"]))
                arm.update(ids_exact=ids == case["reference_ids"], text_exact=native.get("text") == ref["text"],
                           tokens=len(ids), first_divergence=diff, eos=native.get("eos"),
                           inference_ms=native.get("inference_ms"), device=native.get("device"),
                           text=native.get("text", "")[:400])
            except Exception as exc:  # retained verbatim: a failed arm is evidence too
                arm.update(error=f"{type(exc).__name__}: {exc}", stderr_tail=err[-1500:])
            entry["arms"][name] = arm
            save(f"{cid} {name}")
    save("done")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
