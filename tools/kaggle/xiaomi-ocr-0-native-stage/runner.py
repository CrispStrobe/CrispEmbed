#!/usr/bin/env python3
"""Hosted first-stage CPU/CUDA ggml audit, not full OCR or a benchmark."""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zlib

BUNDLE = "__CRISPEMBED_SOURCE_BUNDLE__"
OUT = Path("/kaggle/working/xiaomi-patch-audit")
ROOT = Path("/kaggle/temp/xiaomi-patch-audit/source")
MODEL_SHA = "1759b8da2a6fa2651de9cacdb5e90fe0551d7ab8b1fe783198f7339389451235"
REF_SHA = "2a58fc3fc2a4ae2d603ff600371cf71fa1f54e40ebd260d7680bfa4c1f868334"
BASE = "15adb672f71f8470a23db1284dafa102515adae9"
GGML = "a06fc0bb46b73cc10011fc90ae4054aee8438aeb"
ALLOWED = {
    "CMakeLists.txt", "src/qwen35_frontend.h", "src/qwen35_ocr.cpp",
    "src/qwen35_tokenizer.cpp", "src/xiaomi_profile_pin.h",
    "tests/test_xiaomi_ocr.cpp", "tests/test_xiaomi_patch_embed.cpp",
    "tests/test_xiaomi_patch_embed_cuda.cpp", "models/verify-gguf-export.py",
    "tests/results/xiaomi-ocr-0/kaggle-export-v2-manifest.json",
}
REPORT = {"scope": "test-only patch projection: native CPU and standalone ggml CUDA, no OCR parity",
          "native_cuda_engine": False, "status": "running"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT["phase"] = phase
    (OUT / "report.json").write_text(json.dumps(REPORT, indent=2, allow_nan=False) + "\n")
    print("===", phase, flush=True)


def unpack(root):
    if BUNDLE == "__CRISPEMBED_SOURCE_BUNDLE__":
        raise ValueError("source bundle was not staged")
    data = json.loads(zlib.decompress(base64.b64decode(BUNDLE, validate=True)))
    if data["base_commit"] != BASE or data["ggml_commit"] != GGML or set(data["files"]) != ALLOWED:
        raise ValueError("source closure differs")
    for name, entry in data["files"].items():
        content = base64.b64decode(entry["content"], validate=True)
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError("source digest differs: " + name)
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    return {"base_commit": BASE, "ggml_commit": GGML,
            "runner_template_sha256": data["runner_template_sha256"],
            "files_sha256": {name: entry["sha256"] for name, entry in data["files"].items()}}


def find_input(name, expected):
    matches = [path for path in Path("/kaggle/input").rglob(name) if digest(path) == expected]
    if len(matches) != 1:
        raise ValueError(f"{name}: expected exactly one input with pinned hash, found {len(matches)}")
    return matches[0]


def command(label, args, cwd=None, env=None, allowed=(0,)):
    print("RUN", label, flush=True)
    result = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    (OUT / (label + ".log")).write_text(result.stdout)
    print(result.stdout[-12000:], flush=True)
    if result.returncode not in allowed:
        raise RuntimeError(f"{label} failed with exit {result.returncode}")
    return result


def main():
    save("admission")
    if not Path("/kaggle/working").is_dir() or shutil.disk_usage("/kaggle/working").free < 6 * 1024**3:
        raise RuntimeError("insufficient hosted workspace")
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU fallback forbidden")
    capability = torch.cuda.get_device_capability()
    if capability < (7, 0):
        raise RuntimeError("CUDA compute capability below 7.0")
    REPORT["device"] = torch.cuda.get_device_name()
    REPORT["cuda_capability"] = list(capability)
    REPORT["cuda_architecture"] = str(capability[0]) + str(capability[1])
    command("nvidia-smi", ["nvidia-smi"])
    model = find_input("xiaomi-ocr-0-f32-experimental.gguf", MODEL_SHA)
    reference = find_input("qianziwen-prefill.gguf", REF_SHA)
    REPORT["input_sha256"] = {"model": MODEL_SHA, "reference": REF_SHA}
    save("checkout")
    ROOT.parent.mkdir(parents=True, exist_ok=True)
    command("clone", ["git", "clone", "--no-checkout", "https://github.com/CrispStrobe/CrispEmbed.git", str(ROOT)])
    command("checkout", ["git", "checkout", "--detach", BASE], cwd=ROOT)
    command("submodule", ["git", "submodule", "update", "--init", "--recursive"], cwd=ROOT)
    for name, cwd, expected in (("repo", ROOT, BASE), ("ggml", ROOT / "ggml", GGML)):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()
        if actual != expected:
            raise ValueError(name + " commit mismatch")
    REPORT["source_closure"] = unpack(ROOT)
    REPORT["source_closure"]["staged_runner_sha256"] = digest(__file__)
    (OUT / "source-closure.json").write_text(json.dumps(REPORT["source_closure"], indent=2) + "\n")
    save("verify-model")
    command("verify-model", [sys.executable, str(ROOT / "models/verify-gguf-export.py"), str(model),
            str(ROOT / "tests/results/xiaomi-ocr-0/kaggle-export-v2-manifest.json")])
    save("build")
    build = ROOT / "build-patch-audit"
    command("configure", ["cmake", "-S", str(ROOT), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_NATIVE=OFF", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_CUDA=ON",
            "-DGGML_CUDA_NO_VMM=ON", "-DGGML_METAL=OFF", "-DGGML_BLAS=OFF",
            "-DCRISPEMBED_BUILD_SHARED=OFF", "-DCMAKE_CUDA_ARCHITECTURES=" + REPORT["cuda_architecture"]])
    command("build", ["cmake", "--build", str(build), "--target", "test-xiaomi-patch-embed",
            "test-xiaomi-patch-embed-cuda", "--parallel", "2"])
    save("execute")
    dump = OUT / "cpu-trace"
    dump.mkdir()
    env = dict(os.environ, CRISPEMBED_DIFF_DUMP_DIR=str(dump))
    cpu = command("cpu", [str(build / "test-xiaomi-patch-embed"), str(model), str(reference)], env=env, allowed=(0, 1))
    cuda_path = OUT / "cuda-patch.f32"
    gpu = command("cuda", [str(build / "test-xiaomi-patch-embed-cuda"), str(model), str(reference), str(cuda_path)],
                  allowed=(0, 1))
    if "AUDIT_BACKEND" not in gpu.stdout or "CUDA" not in gpu.stdout:
        raise RuntimeError("CUDA backend evidence missing")
    a = np.fromfile(dump / "prefill.vision.patch_embed.f32", dtype="<f4")
    b = np.fromfile(cuda_path, dtype="<f4")
    if a.size != 1984 * 768 or b.size != a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("native CPU/CUDA outputs have invalid sizes or non-finite values")
    aa, bb = a.astype(np.float64).reshape(-1, 768), b.astype(np.float64).reshape(-1, 768)
    d = aa - bb
    norms = np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1)
    if (norms <= 0).any():
        raise ValueError("zero magnitude row in CPU/CUDA comparison")
    metrics = {"max_abs": float(np.abs(d).max()), "mean_abs": float(np.abs(d).mean()),
               "rms": float(np.sqrt(np.square(d).mean())), "cos_min": float(np.min((aa * bb).sum(axis=1) / norms)),
               "cpu_norm": float(np.linalg.norm(aa)), "cuda_norm": float(np.linalg.norm(bb))}
    metrics["pass"] = metrics["max_abs"] <= 0.002 and metrics["cos_min"] >= 0.99999
    REPORT["cpu_vs_original_strict_exit"] = cpu.returncode
    REPORT["cuda_vs_original_strict_exit"] = gpu.returncode
    REPORT["cpu_vs_cuda_f32"] = metrics
    REPORT["output_sha256"] = {"cpu": digest(dump / "prefill.vision.patch_embed.f32"), "cuda": digest(cuda_path)}
    REPORT["status"] = "completed patch projection only; decoded and full vision parity not tested"
    save("complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        REPORT["status"] = "failed"
        REPORT["error_type"] = type(error).__name__
        REPORT["error"] = str(error)[:1000]
        REPORT["traceback"] = traceback.format_exc()[-5000:]
        save(REPORT.get("phase", "startup"))
        raise
