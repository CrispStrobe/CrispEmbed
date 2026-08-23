#!/usr/bin/env python3
"""LFM2.5-VL-3B reference dump — Kaggle kernel (chr1s4).

Runs tools/dump_lfm2_vl_reference.py on a CC0 test image, producing a
per-stage reference GGUF for the crispembed parity loop, and uploads it
to HF. Needs a T4/P100 GPU (model is loaded in f32, ~12 GB).

Follows the Kaggle regime: kaggle_harness heartbeat/progress, HF token
from the private chr1s4/crispasr-hf-token dataset, uploads to
cstr/crispembed-regression-fixtures.
"""
import json, os, subprocess, sys, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
REPO = WORK / "CrispEmbed"
REF_GGUF = WORK / "lfm2-vl-ref.gguf"
PROGRESS = WORK / "progress.txt"
RESULTS = WORK / "lfm2_vl_refdump_results.json"

# Test image: use the CC0 receipt from the repo
TEST_IMAGE = "tests/regression/images/cc0/commons_example_receipt.png"

HF_REF_REPO = "cstr/crispembed-regression-fixtures"
HF_REF_PATH = "lfm2_vl/commons_example_receipt/ref.gguf"

# ── Setup ────────────────────────────────────────────────────────────

def log(msg):
    print(msg, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(str(msg) + "\n")

log("=== LFM2.5-VL-3B reference dump ===")

# Clone repo (shallow + submodules for gguf Python package)
if not REPO.exists():
    log("Cloning CrispEmbed...")
    subprocess.check_call([
        "git", "clone", "--depth", "1", "--recursive", "--shallow-submodules",
        REPO_URL, str(REPO)
    ])

# Add gguf package to path (from the ggml submodule)
sys.path.insert(0, str(REPO / "ggml" / "scripts"))

# HF token
sys.path.insert(0, str(REPO / "tools" / "kaggle" / "crispembed-ref-gen"))
try:
    # Try kaggle_harness from the repo clone
    crispasr_harness = WORK / "CrispASR" / "tools" / "kaggle" / "kaggle_harness.py"
    if not crispasr_harness.exists():
        # Clone CrispASR for the harness
        crispasr_url = "https://github.com/CrispStrobe/CrispASR.git"
        crispasr_dir = WORK / "CrispASR"
        if not crispasr_dir.exists():
            subprocess.check_call([
                "git", "clone", "--depth", "1", crispasr_url, str(crispasr_dir)
            ])
    sys.path.insert(0, str(WORK / "CrispASR" / "tools" / "kaggle"))
    import kaggle_harness as kh
    kh.init_progress()
    tok = kh.resolve_hf_token(require=True)
except Exception as e:
    log(f"kaggle_harness import failed ({e}); falling back to env HF_TOKEN")
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        # Try the private dataset
        token_path = Path("/kaggle/input/crispasr-hf-token/hf_token.txt")
        if token_path.exists():
            tok = token_path.read_text().strip()

if tok:
    os.environ["HF_TOKEN"] = tok
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
    log("HF token set")
else:
    log("WARNING: no HF token — upload will fail")

# Install deps — LFM2.5-VL needs transformers>=5.0.0 (Lfm2VlForConditionalGeneration)
log("Installing dependencies...")
# LFM2.5-VL requires transformers >=5.0.0 for Lfm2VlForConditionalGeneration.
# Do NOT reinstall torch — Kaggle's pre-installed CUDA torch is matched to the
# GPU (P100/sm_60 needs the pre-installed build; pip-upgraded torch drops sm_60).
subprocess.run("pip install -q --no-deps 'transformers>=5.0.0'", shell=True)
subprocess.run("pip install -q accelerate gguf Pillow", shell=True)
log(f"transformers version: {subprocess.check_output('python -c \"import transformers; print(transformers.__version__)\"', shell=True).decode().strip()}")

# ── Run the dumper ───────────────────────────────────────────────────

image_path = str(REPO / TEST_IMAGE)
if not Path(image_path).exists():
    log(f"FATAL: test image not found at {image_path}")
    sys.exit(1)

dumper = str(REPO / "tools" / "dump_lfm2_vl_reference.py")
cmd = [
    sys.executable, dumper,
    "--model", "LiquidAI/LFM2.5-VL-3B",
    "--image", image_path,
    "--output", str(REF_GGUF),
    "--max-vis-layers", "4",
    "--max-llm-layers", "4",
    "--prompt", "OCR this image. Output the text content.",
]

log(f"Running: {' '.join(cmd)}")
rc = subprocess.call(cmd, env={**os.environ, "PYTHONUNBUFFERED": "1"})
log(f"Dumper exit code: {rc}")

if rc != 0 or not REF_GGUF.exists():
    log("FATAL: dumper failed or produced no output")
    json.dump({"status": "FAIL", "rc": rc}, open(RESULTS, "w"))
    sys.exit(1)

ref_size = REF_GGUF.stat().st_size
log(f"Reference GGUF: {ref_size / 1024 / 1024:.1f} MB")

# ── Upload to HF ────────────────────────────────────────────────────

log(f"Uploading to {HF_REF_REPO}/{HF_REF_PATH}...")
try:
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    api.upload_file(
        path_or_fileobj=str(REF_GGUF),
        path_in_repo=HF_REF_PATH,
        repo_id=HF_REF_REPO,
        repo_type="dataset",
    )
    log("Upload complete")
    status = "PASS"
except Exception as e:
    log(f"Upload failed: {e}")
    status = "FAIL_UPLOAD"

# ── Results ──────────────────────────────────────────────────────────

results = {
    "status": status,
    "ref_size_bytes": ref_size,
    "hf_repo": HF_REF_REPO,
    "hf_path": HF_REF_PATH,
    "image": TEST_IMAGE,
}
json.dump(results, open(RESULTS, "w"), indent=2)
log(f"\nDone: {status}")
log(json.dumps(results, indent=2))
