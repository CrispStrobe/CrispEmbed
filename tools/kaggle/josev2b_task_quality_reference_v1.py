#!/usr/bin/env python3
"""NEW20 authored-task original-only GPU recipe. No old-four recapture.

Run inside an approved GPU session, after installing the documented pinned env.
No CPU fallback, native inference, publication, training or calibration fitting.
"""
import argparse
import json
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from josev2b_task_quality_v1 import CONTRACT, ROOT, environment_versions, panel, protocol, require_cuda_capability, sha256, source_guard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-source", required=True)
    parser.add_argument("--reviewed-contract-sha256", required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = protocol()
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != args.reviewed_source:
        raise ValueError("checkout must equal explicitly reviewed source")
    subprocess.run(["git", "diff", "--exit-code", "HEAD"], cwd=ROOT, check=True)
    if args.reviewed_contract_sha256 != sha256(CONTRACT):
        raise ValueError("reviewed contract digest differs")
    source_guard()
    panel()
    if args.scratch.exists() or args.out.exists():
        raise ValueError("use fresh scratch/output; retain earlier attempts")
    # Detect unsupported P100 before importing torch or accessing any model payload.
    capabilities = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                                          text=True).strip().splitlines()
    if not capabilities:
        raise ValueError("CUDA hardware capability unavailable")
    try:
        for item in capabilities:
            require_cuda_capability(tuple(map(int, item.strip().split("."))))
    except ValueError:
        args.out.mkdir(parents=True)
        (args.out / "preflight.json").write_text(json.dumps({"status": "unsupported_gpu", "conclusive": False,
            "retry_permitted": False, "compute_capabilities": capabilities, "model_payload_accessed": False,
            "source": args.reviewed_source, "contract_sha256": args.reviewed_contract_sha256}, indent=2) + "\n",
            encoding="utf-8")
        raise
    environment_versions()
    args.scratch.mkdir(parents=True)
    if shutil.disk_usage(args.scratch).free < 18 * 2**30:
        raise ValueError("need 18 GiB free scratch")
    # Resource barrier BEFORE any checkpoint/tokenizer download.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if any(importlib.util.find_spec(name) is not None for name in ("fla", "causal_conv1d")):
        raise ValueError("use an isolated env without optional attention/conv kernels")
    import torch
    if not torch.cuda.is_available():
        raise ValueError("approved CUDA session required; do not run on CPU")
    require_cuda_capability(torch.cuda.get_device_capability())
    free, total = torch.cuda.mem_get_info()
    if free < cfg["scope"]["min_free_cuda_gib"] * 2**30 or total < cfg["scope"]["min_total_cuda_gib"] * 2**30:
        raise ValueError("GPU memory bound not met")
    from huggingface_hub import hf_hub_download
    snapshot = args.scratch / "snapshot"
    # Tokenizer/source freeze precedes ALL weight payloads. No original output
    # is obtained during this stage, and no state facts may be clipped.
    def download(filename):
        path = Path(hf_hub_download(cfg["repository"], filename, revision=cfg["revision"], local_dir=snapshot))
        expected = cfg["files"][filename]
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"pinned download failed: {filename}")
    for filename in cfg["files"]:
        if not filename.endswith(".safetensors"):
            download(filename)
    producer = TOOLS / "dump_josev2b_task_quality_v1.py"
    worker_args = ["--snapshot", str(snapshot), "--out", str(args.out),
                   "--reviewed-contract-sha256", args.reviewed_contract_sha256]
    subprocess.run([sys.executable, str(producer), "prepare", *worker_args], cwd=ROOT,
                   timeout=300, check=True)
    preflight = json.loads((args.out / "preflight.json").read_text())
    if preflight.get("status") != "input-freeze-completed" or preflight.get("model_weights_loaded") is not False or len(preflight["cases"]) != 20:
        raise ValueError("new20 tokenizer/source freeze failed before weight access")
    frozen_sha = sha256(args.out / "input-freeze.json")
    if preflight["input_freeze_sha256"] != frozen_sha:
        raise ValueError("preflight/frozen actual input SHA differs")
    for filename in cfg["files"]:
        if filename.endswith(".safetensors"):
            download(filename)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run([sys.executable, str(producer), "capture", *worker_args,
                   "--reviewed-input-freeze-sha256", frozen_sha], cwd=ROOT,
                   timeout=cfg["scope"]["max_wall_seconds"], check=True)


if __name__ == "__main__":
    main()
