#!/usr/bin/env python3
"""Original Moebius / PixelHacker CUDA reference capture on a Kaggle GPU worker.

usage: run.py moebius|pixelhacker   (from the CrispEmbed checkout root)

1. Detect the GPU. PixelHacker's original Flash Linear Attention (Triton)
   needs compute capability >= 7.0; on a P100 this exits at once so the
   session does not idle (no re-push loop: the operator decides).
2. Provision a separate interpreter with the pinned original framework
   versions from docs/latent-inpaint-source-contract.json (uv venv).
3. Download the pinned original source files and checkpoints, verify them.
4. Run tools/dump_latent_inpaint_reference.py (unchanged) on cuda:0 for the
   full four-case panel, then write the captures as gzip parts in
   /kaggle/working for the hosted CPU native comparison.
Every step writes its own log file and has a timeout; no unread pipes.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tools"))
from latent_inpaint_contract import load_contract, model_record, verify_checkpoint, verify_source  # noqa: E402

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
PROGRESS = WORK / "progress.txt"
MODELS = ("moebius", "pixelhacker")
TORCH = {"moebius": ("2.7.1", "0.22.1", "https://download.pytorch.org/whl/cu126"),
         "pixelhacker": ("2.3.0", "0.18.0", "https://download.pytorch.org/whl/cu121")}
PACKAGES = {
    "moebius": ["diffusers==0.38.0", "accelerate==1.14.0", "transformers", "flash-linear-attention==0.3.2",
                "safetensors", "pillow", "numpy", "opencv-python-headless", "pyyaml", "omegaconf", "timm",
                "tqdm", "toml", "einops", "scipy"],
    "pixelhacker": ["diffusers==0.30.2", "accelerate==0.34.0", "transformers==4.40.0", "tokenizers==0.19.0",
                    "einops==0.7.0", "timm==0.9.11", "omegaconf==2.3.0", "pyyaml", "numpy<2",
                    "opencv-python-headless", "pillow", "safetensors", "tqdm", "scipy"],
}
# Upstream pins triton 3.2.0 beside torch 2.3.0 (whose metadata asks for 2.3.0),
# so these two are installed after resolution, exactly as the author's list.
# FLA 0.3.2 ships its `fla` ops in the separate fla-core distribution.
NO_DEPS = {"moebius": [], "pixelhacker": ["flash-linear-attention==0.3.2", "fla-core==0.3.2", "triton==3.2.0"]}


def progress(message):
    line = time.strftime("%H:%M:%S ") + message
    print(line, flush=True)
    with PROGRESS.open("a") as stream:
        stream.write(line + "\n")


def step(name, argv, timeout, env=None):
    progress("start " + name)
    start = time.monotonic()
    with (WORK / (name + ".log")).open("wb") as log:
        try:
            rc = subprocess.run([str(a) for a in argv], stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                                env=dict(os.environ, **(env or {}))).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    progress(f"end {name} rc={rc} seconds={time.monotonic() - start:.0f}")
    if rc != 0:
        raise SystemExit(f"{name} failed ({rc}); see {name}.log")


def download(url, path, size, digest):
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "CrispEmbed-latent-capture/1"})
    h, count = hashlib.sha256(), 0
    with urllib.request.urlopen(request, timeout=120) as response, path.open("xb") as out:
        for block in iter(lambda: response.read(1 << 20), b""):
            count += len(block)
            h.update(block)
            out.write(block)
    if count != size or h.hexdigest() != digest:
        raise SystemExit(f"pinned download differs: {url}")


def main(model):
    WORK.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    progress(f"model={model} commit={os.environ.get('CAPTURE_COMMIT')} gpu={gpu or 'none'}")
    if not gpu:
        raise SystemExit("NO_GPU")
    capability = float(gpu.splitlines()[0].split(",")[1])
    if model == "pixelhacker" and capability < 7.0:
        raise SystemExit("NEEDS_SM70: original Triton FLA cannot run on this GPU; not retrying automatically")
    contract = load_contract()
    # 1. Interpreter with the original framework pins.
    venv = TEMP / ("venv-" + model)
    python = venv / "bin" / "python"
    step("uv-install", [sys.executable, "-m", "pip", "install", "-q", "uv"], 600)
    step("venv", [sys.executable, "-m", "uv", "venv", "--python", "3.11", str(venv)], 600)
    torch_version, vision_version, index = TORCH[model]
    step("pip-torch", [sys.executable, "-m", "uv", "pip", "install", "--python", python,
                       f"torch=={torch_version}", f"torchvision=={vision_version}", "--index-url", index], 1800)
    step("pip-deps", [sys.executable, "-m", "uv", "pip", "install", "--python", python,
                      f"torch=={torch_version}", *PACKAGES[model], "--extra-index-url", index,
                      "--index-strategy", "unsafe-best-match"], 1800)
    if NO_DEPS[model]:
        step("pip-nodeps", [sys.executable, "-m", "uv", "pip", "install", "--python", python, "--no-deps",
                            *NO_DEPS[model]], 900)
    step("pip-freeze", [sys.executable, "-m", "uv", "pip", "freeze", "--python", python], 120)
    # 2. Exact original sources and checkpoints.
    record = contract["source"][model]
    source = TEMP / "original" / model
    repository = urllib.parse.urlsplit(record["repository"]).path
    for relative, expected in record["files"].items():
        url = ("https://raw.githubusercontent.com" + repository + "/" + record["commit"] + "/" +
               urllib.parse.quote(relative, safe="/"))
        download(url, source / relative, expected["bytes"], expected["sha256"])
    verify_source(source, model, contract)
    progress("sources verified")
    checkpoints = {}
    for name in (model, "sdxl_vae"):
        row = model_record(name, contract)["checkpoint"]
        path = TEMP / "checkpoints" / (name + ".bin")
        download(row["url"], path, row["bytes"], row["sha256"])
        verify_checkpoint(path, model_record(name, contract))
        checkpoints[name] = path
    progress("checkpoints verified")
    # 3. Unchanged original capture.
    env = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": str(ROOT / "tools"), "PYTHONUNBUFFERED": "1"}
    step("import-preflight", [python, "-c",
                              "import sys,torch,diffusers;sys.path.insert(0,sys.argv[1]);"
                              + ("import removal.v1_2.removal_model, removal.v1_2.pipeline"
                                 if model == "moebius" else "import utils, pipeline")
                              + ";print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))",
                              str(source)], 900, env)
    output = TEMP / ("capture-" + model)
    step("capture-original", [python, "tools/dump_latent_inpaint_reference.py", "--model", model,
                              "--source-dir", source, "--checkpoint", checkpoints[model],
                              "--vae-checkpoint", checkpoints["sdxl_vae"], "--output", output,
                              "--device", "cuda:0", "--threads", "2"], 5400, env)
    report = json.loads((output / "report.json").read_text())
    (WORK / "original-report.json").write_text(json.dumps(report, indent=2) + "\n")
    for case in report["cases"]:
        (WORK / (case["id"] + "-reference.json")).write_bytes((output / case["id"] / "reference.json").read_bytes())
    # 4. Hand the raw captures to the hosted CPU native replay. The private HF
    #    account is at its storage limit, so the archive is written as gzip
    #    parts below Kaggle's 1 GB output-file limit; the hosted job downloads
    #    this kernel's output and checks every part and the whole digest.
    archive = TEMP / f"{model}-original-capture.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        tar.add(output, arcname=f"{model}-original-capture")
    hasher, parts = hashlib.sha256(), []
    with archive.open("rb") as stream:
        for index in range(1000):
            block = stream.read(900 * 1024 * 1024)
            if not block:
                break
            hasher.update(block)
            part = WORK / f"{archive.name}.part{index:02d}"
            part.write_bytes(block)
            parts.append({"file": part.name, "bytes": len(block), "sha256": hashlib.sha256(block).hexdigest()})
    archive.unlink()
    manifest = {"model": model, "commit": os.environ.get("CAPTURE_COMMIT"), "gpu": gpu,
                "archive": archive.name, "bytes": sum(p["bytes"] for p in parts), "sha256": hasher.hexdigest(),
                "parts": parts, "report_sha256": hashlib.sha256((output / "report.json").read_bytes()).hexdigest(),
                "complete": report.get("complete"), "cases": [c["id"] for c in report["cases"]]}
    (WORK / "capture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    progress(f"archive {manifest['bytes']} bytes in {len(parts)} parts sha256={manifest['sha256']}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in MODELS:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1]))
