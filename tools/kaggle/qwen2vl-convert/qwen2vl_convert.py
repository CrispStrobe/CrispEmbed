import os, subprocess, sys, shutil, time, traceback
from pathlib import Path

# Earliest possible breadcrumb — if we see this file, the script started
_WORK = "/kaggle/working" if os.path.exists("/kaggle/working") else "/tmp/qwen2vl-convert"
os.makedirs(_WORK, exist_ok=True)
with open(os.path.join(_WORK, "progress.txt"), "w") as _f:
    _f.write(f"script_started at {time.time()}\n")
    _f.write(f"python={sys.version}\n")
    _f.write(f"cwd={os.getcwd()}\n")

os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

WORK = Path(_WORK)

REPO = WORK / "CrispEmbed"
BRANCH = "feat/keyven-german-ocr"
HF_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
HF_REPO = "cstr/qwen2.5-vl-3b-crispembed-GGUF"
OUT_F16 = WORK / "qwen2.5-vl-3b-f16.gguf"

def run(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)

LOG = WORK / "progress.txt"

def step(msg):
    line = f"[{time.time():.0f}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

try:
    # ── Step 1: Clone CrispEmbed ──────────────────────────────────
    step("1. Cloning CrispEmbed")
    if REPO.exists():
        shutil.rmtree(REPO)
    run(f"git clone --depth 1 --branch {BRANCH} https://github.com/CrispStrobe/CrispEmbed.git {REPO}")
    step("1. Clone done")

    # ── Step 2: Install deps ──────────────────────────────────────
    step("2. Installing dependencies")
    # torch is pre-installed on Kaggle — only install small deps
    run(f"{sys.executable} -m pip install --quiet safetensors gguf huggingface_hub transformers hf_transfer")
    step("2. Deps installed")

    # ── Step 3: Resolve HF token ──────────────────────────────────
    step("3. Resolving HF token")
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        # Try Kaggle Secrets
        try:
            from kaggle_secrets import UserSecretsClient
            client = UserSecretsClient()
            hf_token = client.get_secret("HF_TOKEN")
        except Exception as e:
            print(f"   Kaggle secret failed: {e}", flush=True)
    if not hf_token:
        # Try dataset file
        for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
                   "/kaggle/input/crispasr-hf-token/token.txt"]:
            if os.path.exists(p):
                hf_token = open(p).read().strip()
                break
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        step(f"3. HF token resolved ({len(hf_token)} chars)")
    else:
        step("3. No HF token — will skip upload")

    # ── Step 4: Download model ────────────────────────────────────
    step("4. Downloading model")
    # Use scratch dir for large downloads
    for candidate in ["/kaggle/temp", "/tmp"]:
        if os.path.isdir(candidate):
            scratch = Path(candidate) / "qwen25vl-cache"
            break
    scratch.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(scratch).free / (1024**3)
    print(f"   Scratch: {scratch} (free: {free_gb:.1f} GiB)", flush=True)

    from huggingface_hub import snapshot_download
    t0 = time.time()
    src = snapshot_download(repo_id=HF_MODEL, cache_dir=str(scratch))
    dt = time.time() - t0
    step(f"4. Downloaded to {src} ({dt:.0f}s)")

    # ── Step 5: Convert to F16 GGUF ──────────────────────────────
    step("5. Converting to F16 GGUF")
    converter = REPO / "models" / "convert-qwen2vl-to-gguf.py"
    t0 = time.time()
    run(f"{sys.executable} {converter} --model {src} --output {OUT_F16} --dtype f16 --load-dtype bfloat16")
    dt = time.time() - t0
    size_gb = OUT_F16.stat().st_size / (1024**3)
    step(f"5. F16 done: {size_gb:.2f} GiB ({dt:.0f}s)")

    # ── Step 6: Upload to HuggingFace ─────────────────────────────
    if hf_token and OUT_F16.exists():
        step("6. Uploading to HuggingFace")
        from huggingface_hub import HfApi
        api = HfApi(token=hf_token)
        try:
            api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
        except Exception as e:
            print(f"   Repo create: {e}", flush=True)

        print(f"   Uploading {OUT_F16.name} ({size_gb:.1f} GiB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(OUT_F16),
            path_in_repo="qwen2.5-vl-3b-f16.gguf",
            repo_id=HF_REPO, repo_type="model",
            commit_message="Add F16 GGUF (vision + 36-layer LLM)",
        )
        step("6. Upload complete")
    else:
        step("6. Skipped upload (no token or no output)")

    step("DONE")

except Exception:
    tb = traceback.format_exc()
    print(tb, flush=True)
    step(f"FAILED: {tb}")
    sys.exit(1)
