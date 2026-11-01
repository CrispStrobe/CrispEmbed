"""Kaggle GPU kernel: original Moebius/PixelHacker CUDA reference capture.

Script kernels upload only this file, so it clones CrispEmbed at the exact
commit rendered by stage.py and runs tools/kaggle/latent-inpaint-capture/run.py
from that checkout. All output goes to files under /kaggle/working, including
the raw captures as verified gzip parts that the hosted CPU native comparison
(tools/ci-heavy/latent_native.sh) downloads. No native code runs here.
"""
import os
import subprocess
import sys

COMMIT = "@COMMIT@"
MODEL = "@MODEL@"
REPO_DIR = "/kaggle/temp/ce"


def main():
    os.makedirs("/kaggle/working", exist_ok=True)
    with open("/kaggle/working/clone.log", "w") as log:
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "https://github.com/CrispStrobe/CrispEmbed.git",
                        REPO_DIR], check=True, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        subprocess.run(["git", "checkout", "-q", COMMIT], cwd=REPO_DIR, check=True, stdout=log,
                       stderr=subprocess.STDOUT, timeout=300)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, capture_output=True, text=True,
                          check=True).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"checked out {head}, expected {COMMIT}")
    return subprocess.run([sys.executable, "tools/kaggle/latent-inpaint-capture/run.py", MODEL], cwd=REPO_DIR,
                          env=dict(os.environ, CAPTURE_COMMIT=COMMIT)).returncode


if __name__ == "__main__":
    sys.exit(main())
