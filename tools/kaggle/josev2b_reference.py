#!/usr/bin/env python3
"""Reviewed-source GPU recipe only; no submission/account configuration here.

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
from josev2b_contract import CONTRACT, ROOT, environment_versions, panel, protocol, require_cuda_capability, sha256


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
    for filename in cfg["files"]:
        path = Path(hf_hub_download(cfg["repository"], filename, revision=cfg["revision"], local_dir=snapshot))
        expected = cfg["files"][filename]
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"pinned download failed: {filename}")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    subprocess.run([sys.executable, str(TOOLS / "dump_josev2b_reference.py"),
        "--snapshot", str(snapshot), "--out", str(args.out),
        "--reviewed-contract-sha256", args.reviewed_contract_sha256], cwd=ROOT,
        timeout=cfg["scope"]["max_wall_seconds"], check=True)


if __name__ == "__main__":
    main()
