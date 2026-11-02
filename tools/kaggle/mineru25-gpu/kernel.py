"""Kaggle GPU kernel: MinerU2.5-Pro-2604 native CUDA run of every precision.

Kaggle script kernels upload only this file, so it clones CrispEmbed at the
exact commit rendered by stage.py (@COMMIT@), builds the CUDA targets for the
GPU actually drawn, and runs tools/kaggle/mineru25-gpu/run.py from that same
checkout. The original Python reference runs on this host's CPU, so a P100
draw (sm_60, unsupported by current torch) does not invalidate the lane.
Results are measured and reported; the publication gate is the hosted CPU
receipt, never this kernel.
"""
import os
import subprocess
import sys

COMMIT = "@COMMIT@"
REPO_DIR = "/kaggle/temp/ce"
BUILD = "/kaggle/temp/build"
PYTHON = "/kaggle/temp/venv/bin/python"


def sh(*command, **kwargs):
    print("+", *command, flush=True)
    subprocess.run(list(command), check=True, **kwargs)


def main():
    cap = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print("GPU:", cap or "none", flush=True)
    if not cap:
        raise SystemExit("NO_GPU: this lane measures the CUDA backend")
    arch = cap.split(",")[-1].strip().replace(".", "")

    sh("git", "clone", "-q", "--filter=blob:none", "https://github.com/CrispStrobe/CrispEmbed.git", REPO_DIR)
    os.chdir(REPO_DIR)
    sh("git", "checkout", "-q", COMMIT)
    sh("git", "submodule", "update", "-q", "--init", "--depth", "1", "ggml")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"checked out {head}, expected {COMMIT}")

    sh("cmake", "-S", ".", "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release", "-DCRISPEMBED_NATIVE=OFF",
       "-DGGML_NATIVE=OFF", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
    sh("cmake", "--build", BUILD, "--parallel", "4", "--target", "test-mineru25-run", "crispembed-quantize")

    # Kaggle's preinstalled stack is unpinned; the reference protocol pins its own.
    sh(sys.executable, "-m", "pip", "install", "-q", "uv==0.8.17")
    sh(sys.executable, "-m", "uv", "venv", "-q", "--python", "3.12", "--seed", "/kaggle/temp/venv")
    sh(PYTHON, "-m", "pip", "install", "-q", "torch==2.9.0", "torchvision==0.24.0",
       "--index-url", "https://download.pytorch.org/whl/cpu")
    sh(PYTHON, "-m", "pip", "install", "-q", "-r", "tools/mineru25-reference-requirements.txt")

    finished = subprocess.run([PYTHON, "tools/kaggle/mineru25-gpu/run.py", BUILD],
                              env=dict(os.environ, MINERU25_COMMIT=COMMIT, MINERU25_GPU=cap,
                                       HF_HUB_DISABLE_IMPLICIT_TOKEN="1"))
    return finished.returncode


if __name__ == "__main__":
    sys.exit(main())
