"""Kaggle GPU kernel: Unlimited-OCR-2 Q4_K-class CUDA final-output cosine.

Clones CrispEmbed at @COMMIT@ (render with: sed "s/@COMMIT@/<40-hex pushed commit>/"),
builds the CUDA targets for the GPU drawn (portable host code, GGML_NATIVE=OFF), and runs
tools/ci-heavy/unlimited_ocr2_q4k_sweep.sh with DEVICE=gpu: the F16 reference is
teacher-forced on the CPU, the F16 control and every recipe run on CUDA, teacher-forced on
the original IDs (final-output cosine) and free-running (decoded IDs/text).
docs/unlimited-ocr2-q4k-recipes.md. Outputs: /kaggle/working/out/.
"""
import os
import subprocess
import sys

COMMIT = "@COMMIT@"
RECIPES = "q4kn,q8"
FREE = "f16,f16cuda,q4kn,q8"
REPO_DIR = "/kaggle/temp/ce"


def sh(*command, **kwargs):
    print("+", *command, flush=True)
    return subprocess.run(list(command), **kwargs)


def main():
    cap = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print("GPU:", cap or "none", flush=True)
    if not cap:
        raise SystemExit("NO_GPU: this lane measures the CUDA backend")
    arch = cap.split(",")[-1].strip().replace(".", "")
    sh("git", "clone", "-q", "--filter=blob:none", "https://github.com/CrispStrobe/CrispEmbed.git", REPO_DIR, check=True)
    os.chdir(REPO_DIR)
    sh("git", "checkout", "-q", COMMIT, check=True)
    sh("git", "submodule", "update", "-q", "--init", "--depth", "1", "ggml", check=True)
    sh(sys.executable, "-m", "pip", "install", "-q", "numpy==2.2.6", "gguf==0.17.1", "pillow==11.3.0",
       "matplotlib==3.10.7", check=True)
    out, scratch = "/kaggle/working/out", "/kaggle/temp/scratch"
    os.makedirs(out, exist_ok=True)
    os.makedirs(scratch, exist_ok=True)
    with open(f"{out}/gpu.txt", "w") as f:
        f.write(cap + "\n")
    env = dict(os.environ, DEVICE="gpu", HEAVY_OUT=out, HEAVY_SCRATCH=scratch,
               CMAKE_EXTRA=f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch}")
    done = sh("bash", "tools/ci-heavy/unlimited_ocr2_q4k_sweep.sh", RECIPES, FREE, env=env)
    sh("rm", "-rf", scratch)
    return done.returncode


if __name__ == "__main__":
    sys.exit(main())
