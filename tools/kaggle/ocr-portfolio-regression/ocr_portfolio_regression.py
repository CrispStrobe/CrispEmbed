#!/usr/bin/env python3
"""CrispEmbed OCR portfolio regression — GPU (Kaggle).

Builds CrispEmbed from `main` with CUDA, then runs the whole OCR portfolio
through the shared driver `tests/regression/run_one.py` — one model per
subprocess for isolation — and prints a pass/fail summary. This is the GPU
counterpart to the CPU-subset nightly in .github/workflows/regression.yml.

What it checks per model (see tests/regression/README.md):
  1. no-garbage guard  (the colorcolor… degeneration that 3fb1f8e shipped)
  2. lenient CER text match vs manifest expected_text (if pinned)
  3. optional diff-harness cos vs <model>-ref.gguf (if the ref is on HF)

Modes:
  default   — validate every model against the manifest.
  --rebake  — print captured OCR text for models whose expected_text is
              null, so you can eyeball + paste it into the manifest.

Runs on P100 (sm_60) or T4 (sm_75). Needs the `chr1s4/crispasr-hf-token`
dataset attached for HF auth.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "main")
REBAKE = "--rebake" in sys.argv

EMBED_DIR = WORK / "CrispEmbed"
BUILD_DIR = EMBED_DIR / "build"


def run(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=True, **kw)


# ── HF token (attached dataset) ──────────────────────────────────────
hf_token = None
for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
          "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
    if Path(p).exists():
        hf_token = Path(p).read_text().strip()
        break
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# ── Step 1: clone + build with CUDA ──────────────────────────────────
print("=" * 60, "\nStep 1: clone CrispEmbed + build (CUDA)\n", "=" * 60)
if not EMBED_DIR.exists():
    run(f"git clone --depth 1 --recursive -b {BRANCH} {REPO_URL} {EMBED_DIR}")
BUILD_DIR.mkdir(exist_ok=True)

gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader",
                     shell=True, capture_output=True, text=True)
has_gpu = gpu.returncode == 0 and len(gpu.stdout.strip()) > 3
print(f"GPU: {gpu.stdout.strip() or 'none'}")

run("pip install -q gguf safetensors huggingface_hub pillow")

cuda_flag = ""
if has_gpu:
    import glob
    cuda_flag = "-DGGML_CUDA=ON"
    stubs = glob.glob("/usr/local/cuda/targets/*/lib/stubs/libcuda.so")
    if stubs:
        cuda_flag += f" -DCMAKE_LIBRARY_PATH={os.path.dirname(stubs[0])}"

os.chdir(str(BUILD_DIR))
try:
    run(f"cmake {EMBED_DIR} -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release {cuda_flag}")
except subprocess.CalledProcessError:
    print("CUDA cmake failed → CPU-only build")
    run(f"cmake {EMBED_DIR} -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release")

# Build the CLI + every diff harness the manifest might use.
manifest = json.loads((EMBED_DIR / "tests/regression/manifest.json").read_text())
diff_targets = sorted({m["diff"]["binary"] for m in manifest["models"] if "diff" in m})
run(f"make -j$(nproc) crispembed-cli " + " ".join(diff_targets))

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
    cmd = [sys.executable, driver, "--name", name]
    if REBAKE:
        cmd.append("--rebake")
    print(f"\n----- {name} -----", flush=True)
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
print(f"\n{len(results)} models, {n_fail} FAIL")
if not REBAKE:
    sys.exit(1 if n_fail else 0)
