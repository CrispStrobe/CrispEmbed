"""Kaggle GPU kernel: OXR-1.0 native CUDA run of F16 and the qualifying quantized recipes.

Kaggle script kernels upload only this file, so it clones CrispEmbed at the
exact commit rendered by stage.py (@COMMIT@), builds the CUDA targets for the
GPU actually drawn, and runs tools/kaggle/oxr-gpu/run.py from that checkout.
No Python reference runs here: native token IDs are compared with the original
model's IDs committed in tests/results/oxr/hosted-parity.json. Results are
measured and reported; publication gates stay on the hosted CPU receipts.
"""
import os
import subprocess
import sys

COMMIT = "@COMMIT@"
REPO_DIR = "/kaggle/temp/ce"
BUILD = "/kaggle/temp/build"


def sh(*command, **kwargs):
    print("+", *command, flush=True)
    subprocess.run(list(command), check=True, **kwargs)


def main():
    cap = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print("GPU:", cap or "none", flush=True)
    if not cap:
        raise SystemExit("NO_GPU: this lane measures the CUDA backend")
    arch = cap.splitlines()[0].split(",")[-1].strip().replace(".", "")

    sh("git", "clone", "-q", "--filter=blob:none", "https://github.com/CrispStrobe/CrispEmbed.git", REPO_DIR)
    os.chdir(REPO_DIR)
    sh("git", "checkout", "-q", COMMIT)
    sh("git", "submodule", "update", "-q", "--init", "--depth", "1", "ggml")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"checked out {head}, expected {COMMIT}")

    sh("cmake", "-S", ".", "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release", "-DCRISPEMBED_NATIVE=OFF",
       "-DGGML_NATIVE=OFF", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
    sh("cmake", "--build", BUILD, "--parallel", "4", "--target", "test-oxr-run", "test-oxr-stage-dump",
       "crispembed-quantize")
    sh(sys.executable, "-m", "pip", "install", "-q", "-r", "tools/oxr-conversion-requirements.txt")

    finished = subprocess.run([sys.executable, "tools/kaggle/oxr-gpu/run.py", BUILD],
                              env=dict(os.environ, OXR_COMMIT=COMMIT, OXR_GPU=cap,
                                       HF_HUB_DISABLE_IMPLICIT_TOKEN="1"))
    return finished.returncode


if __name__ == "__main__":
    sys.exit(main())
