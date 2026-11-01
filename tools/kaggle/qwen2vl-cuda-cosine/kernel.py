"""Kaggle GPU kernel: TeleOCR and MinerU2.5-Pro CUDA final-output cosine.

Clones CrispEmbed at @COMMIT@ (rendered by stage.py), builds the CUDA targets
for the GPU drawn, and runs tools/ci-heavy/qwen2vl_cpu_cosine.sh with
DEVICE=gpu for both models: the reference (MinerU F32, TeleOCR F16-on-CPU)
is teacher-forced on the CPU, every variant runs free on CUDA for decoded
agreement and teacher-forced on CUDA for the logits at every generated
position (docs/precision-acceptance.md). Outputs: /kaggle/working/<model>/.
"""
import os
import subprocess
import sys

COMMIT = "@COMMIT@"
REPO_DIR = "/kaggle/temp/ce"
# Version 2 completed the MinerU lane; version 3 reruns TeleOCR only, after the
# fix that keeps the GPU F16 candidate dumps apart from the CPU F16 reference.
RUNS = [("teleocr", "f16,q8_0,q4_k-m,q4_k,q4_k-attn8")]


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
    sh(sys.executable, "-m", "pip", "install", "-q", "numpy==2.2.6", "safetensors==0.6.2",
       "huggingface_hub==0.36.0", "gguf==0.17.1", check=True)
    status = 0
    for model, variants in RUNS:
        out = f"/kaggle/working/{model}"
        scratch = f"/kaggle/temp/{model}"
        os.makedirs(out, exist_ok=True)
        os.makedirs(scratch, exist_ok=True)
        env = dict(os.environ, DEVICE="gpu", HEAVY_OUT=out, HEAVY_SCRATCH=scratch, HF_HOME=f"{scratch}/hf",
                   CMAKE_EXTRA=f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch} -DGGML_NATIVE=OFF "
                               "-DCRISPEMBED_NATIVE=OFF", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
        done = sh("bash", "tools/ci-heavy/qwen2vl_cpu_cosine.sh", model, variants, env=env)
        status = status or done.returncode
        sh("rm", "-rf", scratch)
    return status


if __name__ == "__main__":
    sys.exit(main())
