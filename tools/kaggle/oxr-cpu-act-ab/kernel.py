"""Kaggle CPU kernel: OXR-1.0 decode activation-quantization A/B (CRISPEMBED_OXR_DECODE_F32_ACT).

Kaggle script kernels upload only this file, so it clones CrispEmbed at the
exact commit rendered by stage.py (@COMMIT@), builds the CPU targets and runs
tools/kaggle/oxr-cpu-act-ab/run.py from that checkout.
No Python reference runs here: native token IDs are compared with the native
CPU F16 arm and with the original model's IDs in tests/results/oxr/hosted-parity.json.
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
    cap = subprocess.run(["nproc"], capture_output=True, text=True).stdout.strip()
    print("CPUs:", cap, flush=True)
    sh("git", "clone", "-q", "--filter=blob:none", "https://github.com/CrispStrobe/CrispEmbed.git", REPO_DIR)
    os.chdir(REPO_DIR)
    sh("git", "checkout", "-q", COMMIT)
    sh("git", "submodule", "update", "-q", "--init", "--depth", "1", "ggml")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"checked out {head}, expected {COMMIT}")

    try:  # ce-ccache-bootstrap: CrispEmbed harness from the pinned clone warms + auto-refreshes the seed (docs/kaggle-ccache.md)
        import glob as _ccg, importlib.util as _ccu, os as _cco
        for _cch in ["kaggle_harness.py"] + sorted(_ccg.glob("/kaggle/working/*/kaggle_harness.py")) + sorted(_ccg.glob("/kaggle/temp/*/kaggle_harness.py")):
            if _cco.path.isfile(_cco.path.join(_cco.path.dirname(_cco.path.abspath(_cch)), "CMakeLists.txt")) and _cco.path.isfile(_cch):
                _ccs = _ccu.spec_from_file_location("ce_kaggle_harness", _cch)
                _ccm = _ccu.module_from_spec(_ccs)
                _ccs.loader.exec_module(_ccm)
                _ccm.ensure_build_cache()
                break
    except Exception as _cce:
        print("ce-ccache: bootstrap skipped:", _cce, flush=True)
    sh("cmake", "-S", ".", "-B", BUILD, "-DCMAKE_BUILD_TYPE=Release", "-DCRISPEMBED_NATIVE=OFF",
       "-DGGML_NATIVE=ON")
    sh("cmake", "--build", BUILD, "--parallel", "4", "--target", "test-oxr-run", "test-oxr-stage-dump",
       "crispembed-quantize")

    finished = subprocess.run([sys.executable, "tools/kaggle/oxr-cpu-act-ab/run.py", BUILD],
                              env=dict(os.environ, OXR_COMMIT=COMMIT, OXR_CPUS=cap,
                                       HF_HUB_DISABLE_IMPLICIT_TOKEN="1"))
    return finished.returncode


if __name__ == "__main__":
    sys.exit(main())
