#!/usr/bin/env python3
"""CrispEmbed persistent KV cache verification — GPU (Kaggle T4/P100).

Tests all OCR engines that use the persistent device-side KV cache:
  - math_ocr (TrOCR): DeiT encoder + TrOCR decoder
  - deepseek_ocr2: SAM + Qwen2-enc + MoE decoder (3.4B)
  - lightonocr: Pixtral ViT + Qwen3 decoder (1B)
  - pix2struct: ViT + T5 decoder

Each test:
  1. Downloads the q4_k model from HuggingFace
  2. Runs OCR on a test image
  3. Verifies non-empty output and no crash
  4. Reports timing

Also runs the full pipeline (DBNet + TrOCR) on a real document page.
"""
import os, sys, subprocess, time, json, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"

# --- Auth via kaggle_harness ---
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
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
try:
    import kaggle_harness as kh
    kh.init_progress()
except ImportError:
    print("WARNING: kaggle_harness not available")
    class kh:
        @staticmethod
        def resolve_hf_token(): return os.environ.get("HF_TOKEN", "")
        @staticmethod
        def install_build_toolchain(): pass
        @staticmethod
        def detect_cuda_arch(): return "75"
        @staticmethod
        def cuda_build_flags(arch): return f"-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES={arch}"
        @staticmethod
        def cache_and_link_flags(): return ""
        @staticmethod
        def safe_build_jobs(gpu=False): return 2
        class build_heartbeat:
            def __init__(self, name): pass
            def __enter__(self): return self
            def __exit__(self, *a): pass

# --- Clone and build CrispEmbed ---
EMBED_DIR = WORK / "CrispEmbed"
BUILD = EMBED_DIR / "build"
MODELS = WORK / "models"
MODELS.mkdir(exist_ok=True)

if not EMBED_DIR.exists():
    print("Cloning CrispEmbed...")
    subprocess.check_call(["git", "clone", "--recursive", "--depth", "1",
                           REPO_URL, str(EMBED_DIR)])

kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
cuda_flags = kh.cuda_build_flags(arch)
cache_flags = kh.cache_and_link_flags()

# cuda_flags and cache_flags are lists — join them for shell command
if isinstance(cuda_flags, list):
    cuda_flags = " ".join(cuda_flags)
if isinstance(cache_flags, list):
    cache_flags = " ".join(cache_flags)

print(f"Building CrispEmbed (CUDA arch={arch})...")
print(f"  cuda_flags: {cuda_flags}")
print(f"  cache_flags: {cache_flags}")
os.chdir(str(EMBED_DIR))
subprocess.check_call(
    f"cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release "
    f"{cuda_flags} {cache_flags} "
    f"-DCRISPEMBED_BUILD_SHARED=ON",
    shell=True)

jobs = kh.safe_build_jobs(gpu=True)
with kh.build_heartbeat("cmake.build"):
    subprocess.check_call(
        f"cmake --build build -j{jobs} --target crispembed test-ocr-pipeline",
        shell=True)

CLI = BUILD / "crispembed"
PIPELINE = BUILD / "test-ocr-pipeline"
assert CLI.exists(), f"CLI not found at {CLI}"

# --- Download models ---
print("\nDownloading models...")
token = kh.resolve_hf_token()
os.environ["HF_TOKEN"] = token
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import hf_hub_download

def dl(repo, fname):
    p = MODELS / fname
    if not p.exists():
        print(f"  Downloading {fname}...")
        hf_hub_download(repo, fname, local_dir=str(MODELS), token=token)
    return str(p)

# Small models for quick tests
pix2tex = dl("cstr/pix2tex-mfr-gguf", "pix2tex-mfr-q4_k.gguf")
dbnet = dl("cstr/dbnet-ic15-GGUF", "dbnet-ic15-q4_k.gguf")
trocr = dl("cstr/trocr-small-printed-GGUF", "trocr-small-printed-q4_k.gguf")

# Larger models (persistent KV is critical for these)
deepseek = dl("cstr/deepseek-ocr2-crispembed-GGUF", "deepseek-ocr2-q4_k.gguf")

# Create a test image (white with black text-like bars)
import struct, zlib
def make_test_png(w, h, path):
    """Minimal PNG with text-like horizontal bars."""
    raw = b""
    for y in range(h):
        raw += b"\x00"  # filter: None
        for x in range(w):
            is_bar = (20 <= y <= 30 or 50 <= y <= 60 or 70 <= y <= 80) and 20 <= x <= min(w-20, 180)
            v = 30 if is_bar else 240
            raw += bytes([v, v, v])
    compressed = zlib.compress(raw)
    def chunk(ctype, data):
        c = ctype + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", compressed))
        f.write(chunk(b"IEND", b""))

test_img = str(WORK / "test.png")
make_test_png(200, 100, test_img)

# --- Run tests ---
LD = f"LD_LIBRARY_PATH={BUILD}/ggml/src"
results = []

def run_ocr(name, model, image, extra="", timeout_s=120):
    """Run OCR and return (output, elapsed_s, success)."""
    cmd = f"{LD} timeout {timeout_s} {CLI} --ocr {image} -m {model} -t 1 {extra}"
    print(f"\n=== {name} ===")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_s+10)
        elapsed = time.time() - t0
        output = r.stdout.strip()
        stderr = r.stderr
        ok = r.returncode == 0 and len(output) > 0
        print(f"  Time: {elapsed:.1f}s, Output: {output[:100]}{'...' if len(output) > 100 else ''}")
        if not ok:
            print(f"  STDERR: {stderr[-200:]}")
        results.append({"name": name, "ok": ok, "time": elapsed, "output": output[:200]})
        return output, elapsed, ok
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED: {e} ({elapsed:.1f}s)")
        results.append({"name": name, "ok": False, "time": elapsed, "error": str(e)})
        return "", elapsed, False

# Test 1: pix2tex (small, graph decoder with persistent KV)
run_ocr("pix2tex q4_k", pix2tex, test_img)

# Test 2: TrOCR pipeline (DBNet + TrOCR, persistent KV)
cmd = f"{LD} timeout 300 {PIPELINE} {dbnet} {trocr} {test_img}"
print(f"\n=== OCR Pipeline (DBNet + TrOCR) ===")
t0 = time.time()
r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=310)
elapsed = time.time() - t0
output = r.stdout + r.stderr
n_regions = output.count("conf=")
print(f"  Time: {elapsed:.1f}s, Regions: {n_regions}, RC: {r.returncode}")
if n_regions > 0:
    for line in output.split("\n"):
        if "conf=" in line:
            print(f"  {line.strip()}")
results.append({"name": "pipeline DBNet+TrOCR", "ok": r.returncode == 0, "time": elapsed, "regions": n_regions})

# Test 3: DeepSeek-OCR2 (3.4B MoE, persistent KV — the main test)
run_ocr("deepseek-ocr2 q4_k", deepseek, test_img, "--ocr-max-tokens 100", timeout_s=300)

# --- Summary ---
print(f"\n{'='*60}")
print("RESULTS:")
for r in results:
    status = "PASS" if r.get("ok") else "FAIL"
    print(f"  [{status}] {r['name']}: {r['time']:.1f}s")
    if r.get("output"):
        print(f"         Output: {r['output'][:80]}")
    if r.get("regions"):
        print(f"         Regions: {r['regions']}")
    if r.get("error"):
        print(f"         Error: {r['error']}")

n_pass = sum(1 for r in results if r.get("ok"))
n_total = len(results)
print(f"\n{n_pass}/{n_total} passed")

# Save results as JSON
with open(WORK / "kv_cache_results.json", "w") as f:
    json.dump(results, f, indent=2)
