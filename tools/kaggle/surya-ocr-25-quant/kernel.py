"""Kaggle CPU kernel: Surya OCR 2.5 F16 conversion and Q8_0/Q4_K quantization.

Clones this repository at the staged commit (@COMMIT@ is substituted by
stage.py before `kaggle kernels push`), builds only the two targets this lane
needs, then runs tools/kaggle/surya-ocr-25-quant/run.py in a clean uv Python
3.12 venv:

  1. anonymous pinned snapshot of datalab-to/surya-ocr-2.5 (metadata pins first),
  2. F16 conversion with --profile surya25,
  3. crispembed-quantize F16 -> q8_0 and -> q4_k,
  4. stored-type census of every arm (no norm/bias may be quantized),
  5. the six owned crops on every arm, compared with the frozen original.

The F16 arm is the only gate: if it no longer reproduces the frozen original
ids and text on this host, the quantized comparison is reported as
non-meaningful rather than qualified. Nothing here changes a default, writes to
the registry, or publishes a model.
"""
import base64
import hashlib
import os
from pathlib import Path
import subprocess
import sys

COMMIT = "@COMMIT@"
RUNNER_B64 = "@RUNNER_B64@"
RUNNER_SHA256 = "@RUNNER_SHA256@"
BUILD = "/kaggle/temp/build"
REPO_DIR = "/kaggle/temp/ce"
PYTHON = "/kaggle/temp/venv/bin/python"


def sh(*command, **kwargs):
    print("+", *command, flush=True)
    subprocess.run(list(command), check=True, **kwargs)


def main():
    sh("git", "clone", "-q", "--filter=blob:none",
       "https://github.com/CrispStrobe/CrispEmbed.git", REPO_DIR)
    os.chdir(REPO_DIR)
    sh("git", "checkout", "-q", COMMIT)
    sh("git", "submodule", "update", "-q", "--init", "--depth", "1", "ggml")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"checked out {head}, expected {COMMIT}")
    runner = base64.b64decode(RUNNER_B64, validate=True)
    if hashlib.sha256(runner).hexdigest() != RUNNER_SHA256:
        raise SystemExit("embedded runner digest mismatch")
    runner_path = Path(REPO_DIR) / "tools/kaggle/surya-ocr-25-quant/run.py"
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_bytes(runner)

    # The CI reference crops use DejaVu Sans; Kaggle CPU images do not ship it.
    # The runner independently checks every rendered PNG against the frozen hash.
    sh("apt-get", "update", "-qq")
    sh("apt-get", "install", "-y", "-qq", "fonts-dejavu-core")

    sh("cmake", "-S", ".", "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release",
       "-DCRISPEMBED_NATIVE=OFF", "-DGGML_NATIVE=OFF", "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF")
    sh("cmake", "--build", BUILD, "--parallel", "4",
       "--target", "crispembed-quantize", "test-surya25-run")

    # Kaggle's preinstalled stack is unpinned and its interpreter lacks ensurepip.
    sh(sys.executable, "-m", "pip", "install", "-q", "uv==0.8.17")
    sh(sys.executable, "-m", "uv", "venv", "-q", "--python", "3.12", "--seed", "/kaggle/temp/venv")
    sh(PYTHON, "-m", "pip", "install", "-q", "numpy==2.2.6", "gguf==0.19.0",
       "safetensors==0.8.0", "huggingface-hub==1.33.0", "pillow==12.1.0", "tokenizers==0.23.3",
       "transformers==5.16.1")
    sh(PYTHON, "-m", "pip", "install", "-q", "torch==2.14.0", "torchvision==0.29.0",
       "--index-url", "https://download.pytorch.org/whl/cpu")

    finished = subprocess.run([PYTHON, "tools/kaggle/surya-ocr-25-quant/run.py"],
                              env=dict(os.environ, SURYA25_COMMIT=COMMIT,
                                       HF_HUB_DISABLE_IMPLICIT_TOKEN="1"))
    os.makedirs("/kaggle/working", exist_ok=True)
    with open("/kaggle/working/commit.txt", "w") as handle:
        handle.write(COMMIT + "\n")
    return finished.returncode


if __name__ == "__main__":
    sys.exit(main())
