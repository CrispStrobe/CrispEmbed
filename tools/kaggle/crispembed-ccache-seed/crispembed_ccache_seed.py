"""Refresh seed for the chr1s4/crispembed-ccache dataset.

The dataset has been dead weight since 2026-06-21: it was uploaded rooted
at the cache CONTENTS (raw `0/…` tree, no `.ccache/` prefix), a layout
neither the harness warmer nor the script-local warmers probe — every
CrispEmbed kernel since has printed `ccache: cold build` and paid the full
compile (~19 min CUDA, ~10 min CPU per run, measured 2026-08-06).

This kernel produces a CORRECT seed: standard CUDA build of the CLI on
current main, then kh.export_ccache_tar() → ONE /kaggle/working/ccache.tar
rooted at `.ccache/`. Kaggle auto-extracts dataset tars on upload, so
versioning the dataset with that file mounts as `<slug>/.ccache/…`, which
`kaggle_harness._warm_ccache_from_dataset` consumes with no code changes.

After the run, locally:
    kaggle kernels output chr1s4/crispembed-ccache-seed -p <dir>
    kaggle datasets version -p <dir-with-only-ccache.tar> \
        -m "seed from crispembed-ccache-seed <date>" --dir-mode tar
(keep ONLY ccache.tar in the staging dir).
"""
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)

REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "main")
EMBED_DIR = TEMP / "CrispEmbed"
BUILD_DIR = EMBED_DIR / "build"

_CRISPASR = TEMP / "CrispASR"
if not _CRISPASR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", CRISPASR_URL, str(_CRISPASR)])
        sys.path.insert(0, str(_CRISPASR / "tools" / "kaggle"))
    except Exception:
        pass
if str(_CRISPASR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.step("clone")
if not EMBED_DIR.exists():
    kh.sh(f"git clone --depth 1 --recursive -b {BRANCH} {REPO_URL} {EMBED_DIR}")
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
print(f"CUDA arch: {arch}", flush=True)
kh.step("configure")
BUILD_DIR.mkdir(exist_ok=True)
kh.sh(f"cd {BUILD_DIR} && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release "
      + " ".join(kh.cuda_build_flags(arch) + kh.cache_and_link_flags()) + " ..", check=True)
kh.step("build")
with kh.build_heartbeat("ninja", 30):
    kh.sh(f"cd {BUILD_DIR} && ninja -j{kh.safe_build_jobs(gpu=True)} crispembed-cli test-core-cpu-ops", check=True)
kh.step("stats+export")
subprocess.run("ccache -s | head -8", shell=True)
tar = kh.export_ccache_tar()
print(f"seed: {tar}", flush=True)
print("done", flush=True)
