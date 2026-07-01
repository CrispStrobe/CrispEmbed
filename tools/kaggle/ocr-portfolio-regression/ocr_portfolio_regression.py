#!/usr/bin/env python3
"""CrispEmbed OCR portfolio regression — GPU (Kaggle).

Builds CrispEmbed `main` with CUDA (ccache-warmed) and runs the whole OCR
portfolio through the shared driver `tests/regression/run_one.py`, one model
per subprocess, then prints a pass/fail summary + writes results.json. GPU
counterpart to the CPU nightly in .github/workflows/regression.yml.

Per model (tests/regression/README.md): no-garbage guard (the colorcolor…
degeneration 3fb1f8e shipped), lenient CER text match vs expected_text, and an
optional diff-harness cos vs <model>-ref.gguf if that ref is on HF.

Follows tools/kaggle regime (../kaggle_usage.md): chr1s4 account; clone CrispASR
+ import kaggle_harness (bundled fallback); kh.init_progress; kh.resolve_hf_token
(3-tier: env→Secret→dataset); kh.install_build_toolchain + kh.cuda_build_flags +
kh.cache_and_link_flags + kh.safe_build_jobs; kh.build_heartbeat +
kh.sh_with_progress around long steps; warm chr1s4/crispembed-ccache. Datasets
attached: chr1s4/crispasr-hf-token + chr1s4/crispembed-ccache.
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

# ── run.log tee (kernels_output does NOT expose stderr — gotcha #15 — so tee
#    everything to a downloadable working file, incl. any fatal traceback). ──
_LOG = open(WORK / "run.log", "w", buffering=1)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def _excepthook(exc_type, exc, tb):
    _LOG.write("\n=== FATAL ===\n")
    traceback.print_exception(exc_type, exc, tb, file=_LOG)
    _LOG.flush()
    traceback.print_exception(exc_type, exc, tb, file=sys.__stderr__)


sys.excepthook = _excepthook

REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "main")
REBAKE = "--rebake" in sys.argv
EMBED_DIR = WORK / "CrispEmbed"
BUILD_DIR = EMBED_DIR / "build"

# ── Clone CrispASR + import kaggle_harness (canonical snippet from
#    kaggle_usage.md); bundled kaggle_harness.py is the fallback. ──
_CRISPASR_DIR = WORK / "CrispASR"
if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
                               CRISPASR_URL, str(_CRISPASR_DIR)])
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass  # fall through to bundled copy
if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()


def _mask(t):
    return f"{t[:3]}…{t[-3:]} (len={len(t)})" if t else "<none>"  # never print full token (#14)


# ── Resolve HF token (reference pattern: kh.step + kh.resolve_hf_token),
#    and show up front IF and HOW it was obtained (masked). Cheap probes only
#    — no separate kaggle_secret() call (it ConnectionErrors + retries, #13). ──
kh.step("resolve HF token")
print("=" * 60, "\nHF token resolution (env → Kaggle Secret → dataset)\n", "=" * 60, flush=True)
_env_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
_ds_paths = ["/kaggle/input/crispasr-hf-token/hf_token.txt",
             "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]
_ds_found = next((p for p in _ds_paths if Path(p).exists()), None)
print(f"  env HF_TOKEN present : {_env_present}", flush=True)
print(f"  dataset token file   : {_ds_found or 'absent (' + ', '.join(_ds_paths) + ')'}", flush=True)

HF_TOKEN = kh.resolve_hf_token()   # env → Kaggle Secret(retry) → dataset; sets env + HF_TRANSFER
if HF_TOKEN:
    src = "env" if _env_present else (f"dataset {_ds_found}" if _ds_found else "Kaggle Secret")
    print(f"  ==> HF_TOKEN resolved via {src}: {_mask(HF_TOKEN)}  "
          f"[HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER','')}]", flush=True)
else:
    print("  ==> no HF_TOKEN resolved — cstr/* repos are public so downloads "
          "should still work; gated repos would fail", flush=True)

# ── Step 1: clone CrispEmbed + build (CUDA, ccache, heartbeat) ──
kh.step("clone.crispembed")
if not EMBED_DIR.exists():
    kh.sh(f"git clone --depth 1 --recursive -b {BRANCH} {REPO_URL} {EMBED_DIR}")
BUILD_DIR.mkdir(exist_ok=True)

gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader",
                     shell=True, capture_output=True, text=True)
has_gpu = gpu.returncode == 0 and len(gpu.stdout.strip()) > 3
print(f"GPU: {gpu.stdout.strip() or 'none'}", flush=True)

# small deps only (gotcha #11: never pip install torch). hf_transfer is needed
# because resolve_hf_token enabled HF_HUB_ENABLE_HF_TRANSFER.
kh.sh("pip install -q huggingface_hub hf_transfer gguf safetensors pillow || true",
      check=False)

kh.install_build_toolchain()           # ninja + ccache + mold; sets CCACHE_DIR


def _warm_crispembed_ccache():
    """Warm ccache from the attached chr1s4/crispembed-ccache dataset. The
    harness only knows the crispasr-ccache paths, so CrispEmbed warms its own
    per-project seed (kaggle_usage.md #17). Cold build if absent."""
    import tarfile
    ccache_dir = Path(os.environ.get("CCACHE_DIR", str(WORK / ".ccache")))
    ccache_dir.mkdir(parents=True, exist_ok=True)
    for base in (Path("/kaggle/input/crispembed-ccache"),
                 Path("/kaggle/input/datasets/chr1s4/crispembed-ccache")):
        tar = base / "ccache.tar"
        if tar.exists():
            try:
                with tarfile.open(tar) as tf:
                    tf.extractall(str(ccache_dir))
                n = sum(1 for _ in ccache_dir.rglob("*") if _.is_file())
                print(f"  crispembed-ccache: warmed from {tar} ({n} files)", flush=True)
                return
            except Exception as e:  # noqa: BLE001
                print(f"  crispembed-ccache: extract failed: {e}", flush=True)
    print("  crispembed-ccache: no seed found (cold build)", flush=True)


_warm_crispembed_ccache()

arch = kh.detect_cuda_arch()
build_flags = (kh.cuda_build_flags(arch) if has_gpu else []) + kh.cache_and_link_flags()

manifest = json.loads((EMBED_DIR / "tests/regression/manifest.json").read_text())
diff_targets = sorted({m["diff"]["binary"] for m in manifest["models"] if "diff" in m})
targets = "crispembed-cli " + " ".join(diff_targets)

os.chdir(str(BUILD_DIR))
kh.step("cmake.configure")
with kh.build_heartbeat("cmake.configure"):
    try:
        kh.sh_with_progress(
            f"cmake {EMBED_DIR} -G Ninja -DCMAKE_BUILD_TYPE=Release "
            + " ".join(build_flags))
    except subprocess.CalledProcessError:
        print("CUDA/ninja configure failed → CPU-only reconfigure", flush=True)
        kh.sh("rm -f CMakeCache.txt && rm -rf CMakeFiles", check=False)
        kh.sh_with_progress(
            f"cmake {EMBED_DIR} -G Ninja -DCMAKE_BUILD_TYPE=Release "
            + " ".join(kh.cache_and_link_flags()))

kh.step("cmake.build")
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build . --target {targets} "
        f"-j{kh.safe_build_jobs(gpu=has_gpu)}")

# ── Step 2: portfolio through the shared driver ──
print("\n" + "=" * 60, "\nStep 2: portfolio regression\n", "=" * 60, flush=True)
env = dict(os.environ)
env["BUILD_DIR"] = str(BUILD_DIR)
env["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{env.get('LD_LIBRARY_PATH','')}"
env["REGRESSION_WORK"] = str(WORK / "models")
driver = str(EMBED_DIR / "tests/regression/run_one.py")

results = {}
for m in manifest["models"]:
    name = m["name"]
    cmd = [sys.executable, driver, "--name", name] + (["--rebake"] if REBAKE else [])
    print(f"\n----- {name} -----", flush=True)
    kh.step(f"model.{name}")
    t0 = time.time()
    with kh.build_heartbeat(f"model.{name}", rss=True):
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
    results[name] = {
        "exit": proc.returncode,
        "time_s": round(dt, 1),
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "tail": (proc.stdout.strip().splitlines() or [""])[-1],
    }

# ── Step 3: summary ──
print("\n" + "=" * 60, "\nStep 3: summary\n", "=" * 60)
print(f"{'Model':<18} {'Time':>7} {'Status':>7}")
print("-" * 36)
n_fail = 0
for name, r in results.items():
    n_fail += r["status"] == "FAIL"
    print(f"{name:<18} {r['time_s']:>6.1f}s {r['status']:>7}")
(WORK / "results.json").write_text(json.dumps(results, indent=2))
print(f"\n{len(results)} models, {n_fail} FAIL", flush=True)

# refresh ccache.tar as a downloadable output for dataset versioning (#17)
try:
    os.chdir(str(WORK))
    subprocess.run("tar cf ccache.tar .ccache/ 2>/dev/null", shell=True, check=False)
except Exception:
    pass

if not REBAKE:
    sys.exit(1 if n_fail else 0)
