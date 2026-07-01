#!/usr/bin/env python3
"""CrispEmbed OCR portfolio regression — GPU (Kaggle).

Builds CrispEmbed from `main` with CUDA (ccache-warmed), then runs the whole
OCR portfolio through the shared driver `tests/regression/run_one.py` — one
model per subprocess for isolation — and prints a pass/fail summary. GPU
counterpart to the CPU-subset nightly in .github/workflows/regression.yml.

Per model (see tests/regression/README.md): (1) no-garbage guard (the
colorcolor… degeneration 3fb1f8e shipped), (2) lenient CER text match vs
manifest expected_text, (3) optional diff-harness cos vs <model>-ref.gguf
if that ref is on HF.

Modes: default = validate; `--rebake` = print captured OCR text (never fails)
so you can eyeball + paste expected_text into the manifest.

Follows kaggle_usage.md: chr1s4 account, kaggle_harness auth + ccache toolchain
(chr1s4/crispasr-hf-token + chr1s4/crispasr-ccache attached). Runs on P100/T4.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "main")
REBAKE = "--rebake" in sys.argv

EMBED_DIR = WORK / "CrispEmbed"
BUILD_DIR = EMBED_DIR / "build"

# ── Auth + toolchain via kaggle_harness (regime: clone CrispASR + import;
#    bundled kaggle_harness.py is the CPU-worker/no-internet fallback) ──
_CRISPASR_DIR = WORK / "CrispASR"
if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
                               CRISPASR_URL, str(_CRISPASR_DIR)])
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass
if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()

# HF token: env → Kaggle secret → attached dataset (harness handles all three)
try:
    tok = kh.hf_token() if hasattr(kh, "hf_token") else None
except Exception:
    tok = None
if not tok:
    for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
              "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
        if Path(p).exists():
            tok = Path(p).read_text().strip()
            break
if tok:
    os.environ["HF_TOKEN"] = tok


def run(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


# ── Step 1: clone + build with CUDA (ccache-warmed) ──────────────────
print("=" * 60, "\nStep 1: clone CrispEmbed + build (CUDA, ccache)\n", "=" * 60)
if not EMBED_DIR.exists():
    run(f"git clone --depth 1 --recursive -b {BRANCH} {REPO_URL} {EMBED_DIR}")
BUILD_DIR.mkdir(exist_ok=True)

gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader",
                     shell=True, capture_output=True, text=True)
has_gpu = gpu.returncode == 0 and len(gpu.stdout.strip()) > 3
print(f"GPU: {gpu.stdout.strip() or 'none'}", flush=True)

run("pip install -q gguf safetensors huggingface_hub pillow")

kh.install_build_toolchain()                 # ninja + ccache + mold; sets CCACHE_DIR


def _warm_crispembed_ccache():
    """Warm the ccache from the attached chr1s4/crispembed-ccache dataset.
    kaggle_harness only knows the crispasr-ccache paths, so do our own —
    CrispEmbed keeps its own ccache seed (kaggle_usage.md: per-project,
    per-account). Falls back silently to a cold build if absent."""
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
extra_flags = kh.cache_and_link_flags()      # -DCMAKE_*_COMPILER_LAUNCHER=ccache, mold
cuda_flags = []
if has_gpu:
    import glob
    cuda_flags = ["-DGGML_CUDA=ON",
                  f"-DCMAKE_CUDA_ARCHITECTURES={kh.detect_cuda_arch()}"]
    stubs = glob.glob("/usr/local/cuda/targets/*/lib/stubs/libcuda.so")
    if stubs:
        cuda_flags.append(f"-DCMAKE_LIBRARY_PATH={os.path.dirname(stubs[0])}")

os.chdir(str(BUILD_DIR))
gen = "-G Ninja" if kh.sh("which ninja", check=False) == 0 else "-G 'Unix Makefiles'"
cmake = f"cmake {EMBED_DIR} {gen} -DCMAKE_BUILD_TYPE=Release " + \
        " ".join(cuda_flags + extra_flags)
try:
    run(cmake)
except subprocess.CalledProcessError:
    print("CUDA cmake failed → CPU-only build", flush=True)
    run(f"cmake {EMBED_DIR} {gen} -DCMAKE_BUILD_TYPE=Release " + " ".join(extra_flags))

manifest = json.loads((EMBED_DIR / "tests/regression/manifest.json").read_text())
diff_targets = sorted({m["diff"]["binary"] for m in manifest["models"] if "diff" in m})
run(f"cmake --build . -j$(nproc) --target crispembed-cli " + " ".join(diff_targets))

# ── Step 2: run the portfolio through the shared driver ──────────────
print("\n" + "=" * 60, "\nStep 2: portfolio regression\n", "=" * 60)
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
    kh.step(f"model:{name}")
    t0 = time.time()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-1500:])
    results[name] = {
        "exit": proc.returncode,
        "time_s": round(dt, 1),
        "status": "PASS" if proc.returncode == 0 else "FAIL",
        "tail": (proc.stdout.strip().splitlines() or [""])[-1],
    }

# ── Step 3: summary ──────────────────────────────────────────────────
print("\n" + "=" * 60, "\nStep 3: summary\n", "=" * 60)
print(f"{'Model':<18} {'Time':>7} {'Status':>7}")
print("-" * 36)
n_fail = 0
for name, r in results.items():
    n_fail += r["status"] == "FAIL"
    print(f"{name:<18} {r['time_s']:>6.1f}s {r['status']:>7}")
(WORK / "results.json").write_text(json.dumps(results, indent=2))
print(f"\n{len(results)} models, {n_fail} FAIL", flush=True)

# refresh ccache tar as a downloadable output (per kaggle_usage.md)
try:
    os.chdir(str(WORK))
    subprocess.run("tar cf ccache.tar .ccache/ 2>/dev/null", shell=True, check=False)
except Exception:
    pass

if not REBAKE:
    sys.exit(1 if n_fail else 0)
