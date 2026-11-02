"""Kaggle GPU kernel: CrispEmbed CUDA-vs-CPU engine audit (feat/cuda-audit).

Kaggle script kernels upload only this file, so it clones CrispEmbed at the
exact commit rendered by stage.py (@COMMIT@), builds crispembed-cli and the
diff binaries for the GPU actually drawn, and runs
tools/kaggle/cuda-audit/run.py from that checkout for batch @BATCH@.
Every arm is a separate process of the same binary; the CPU arm hides the GPU
(CUDA_VISIBLE_DEVICES=""), so it is the CPU reference of the same build.
"""
import json
import os
import subprocess
import sys
import time

COMMIT = "@COMMIT@"
BATCH = "@BATCH@"
SCRIPT_VERSION = "cuda-audit-kernel-v1"
REPO_DIR = "/kaggle/temp/ce"
BUILD = "/kaggle/temp/build"
OUT = "/kaggle/working/cuda-audit"


def note(phase, **kw):
    os.makedirs(OUT, exist_ok=True)
    rec = {"script_version": SCRIPT_VERSION, "commit": COMMIT, "batch": BATCH, "phase": phase,
           "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **kw}
    with open(f"{OUT}/kernel-phase.json", "w") as f:
        json.dump(rec, f, indent=1)
    print("==", phase, kw, flush=True)


def sh(*command, log=None, timeout=5400):
    print("+", *command, flush=True)
    if log:
        with open(log, "w") as f:
            rc = subprocess.run(list(command), stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
    else:
        rc = subprocess.run(list(command), timeout=timeout).returncode
    if rc != 0:
        if log:
            print(open(log, errors="replace").read()[-6000:], flush=True)
        note("failed", step=" ".join(command)[:300], rc=rc)
        raise SystemExit(rc)


def main():
    cap = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    print("GPU:", cap or "none", flush=True)
    note("start", gpu=cap, nproc=os.cpu_count())
    if not cap:
        note("no-gpu")
        raise SystemExit("NO_GPU: this lane measures the CUDA backend")
    arch = cap.splitlines()[0].split(",")[1].strip().replace(".", "")

    sh("git", "clone", "-q", "--filter=blob:none", "https://github.com/CrispStrobe/CrispEmbed.git", REPO_DIR)
    os.chdir(REPO_DIR)
    sh("git", "checkout", "-q", COMMIT)
    sh("git", "submodule", "update", "-q", "--init", "--depth", "1", "ggml")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"checked out {head}, expected {COMMIT}")
    sys.path.insert(0, f"{REPO_DIR}/tools/kaggle/cuda-audit")
    import run as audit  # noqa: E402  (from the pinned checkout)
    targets = audit.build_targets(BATCH)
    note("configure", arch=arch, targets=targets)
    os.makedirs(OUT, exist_ok=True)
    sh("cmake", "-S", ".", "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release", "-DCRISPEMBED_NATIVE=ON",
       "-DGGML_NATIVE=ON", "-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
       "-DCRISPEMBED_BUILD_SHARED=OFF", log=f"{OUT}/configure.log")
    t0 = time.time()
    sh("cmake", "--build", BUILD, "--parallel", str(max(2, os.cpu_count() or 4)), "--target", *targets,
       log=f"{OUT}/build.log", timeout=7200)
    note("built", build_s=round(time.time() - t0, 1))
    rc = audit.main(BUILD, BATCH, commit=COMMIT, gpu=cap)
    note("done", rc=rc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
