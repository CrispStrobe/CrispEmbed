# %% [markdown]
# # CrispEmbed — Unlimited-OCR GGUF reconvert (stacked MoE experts)
#
# Reconverts `baidu/Unlimited-OCR` with the converter that emits the 64 routed
# experts as STACKED 3D tensors (`l.blk.{i}.ffn_{gate,up,down}_exps.weight`,
# ggml ne=[in,out,n_exp]) so the runtime loads them directly and skips the
# ~1.3 GB per-expert→stacked duplication (same as deepseek-ocr2 #4).
#
# Validates the stacked bytes are byte-identical to the source experts, quantizes
# q4_k, and uploads to `cstr/unlimited-ocr-crispembed-GGUF` under NEW `-stacked`
# filenames (non-clobbering).

# %% [code]
import os, sys, subprocess, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "feat/uocr-stacked-experts")
EMBED_DIR = WORK / "CrispEmbed"
BUILD_DIR = EMBED_DIR / "build"

SCRATCH = None
for c in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(c):
        SCRATCH = Path(c) / "uocr"
        break
SCRATCH.mkdir(parents=True, exist_ok=True)

_CRISPASR = WORK / "CrispASR"
if not _CRISPASR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", CRISPASR_URL, str(_CRISPASR)])
    except Exception:
        pass
sys.path.insert(0, str(_CRISPASR / "tools" / "kaggle"))
if str(_CRISPASR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
hf_token = kh.resolve_hf_token(require=True)  # upload-bearing: fail fast before any compute (G7d)
kh.step("harness_ready", hf_token_ok=bool(hf_token))

# %% [code]
kh.step("clone.crispembed", branch=BRANCH)
if EMBED_DIR.exists():
    shutil.rmtree(EMBED_DIR)
kh.sh(f"git clone --depth 1 --recursive -b {BRANCH} {REPO_URL} {EMBED_DIR}")
kh.sh("pip install -q huggingface_hub hf_transfer gguf safetensors numpy torch || true", check=False)
kh.install_build_toolchain()

BUILD_DIR.mkdir(exist_ok=True)
kh.step("cmake.configure")
kh.sh(f"cd {BUILD_DIR} && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_METAL=OFF "
      + " ".join(kh.cache_and_link_flags()) + " ..", check=True)
kh.step("build.quantize")
kh.sh(f"cd {BUILD_DIR} && ninja -j{kh.safe_build_jobs(gpu=False)} crispembed-quantize", check=True)
QUANTIZE = next((p for p in (BUILD_DIR / "crispembed-quantize", BUILD_DIR / "bin" / "crispembed-quantize")
                 if p.exists()), BUILD_DIR / "crispembed-quantize")
kh.step("quantize_built", path=str(QUANTIZE))

# %% [code]
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import snapshot_download  # noqa: E402

HF_SRC = "baidu/Unlimited-OCR"
kh.step("download.source", model=HF_SRC)
with kh.build_heartbeat("model.download"):
    SRC = Path(snapshot_download(repo_id=HF_SRC, cache_dir=str(SCRATCH),
                                 allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt",
                                                 "tokenizer*", "*.py"]))
kh.step("downloaded", src=str(SRC))

# %% [code]
converter = EMBED_DIR / "models" / "convert-unlimited-ocr-to-gguf.py"
F16 = SCRATCH / "unlimited-ocr-f16-stacked.gguf"
kh.step("convert.f16")
# Prefix single-thread BLAS/OMP + unbuffered (dev-guide HARD rule: numpy bf16→f16
# casts + the large expert accumulate/stack deadlock/thrash under multithreaded
# OpenBLAS/OMP otherwise). TMPDIR off /tmp for GGUFWriter's use_temp_file spill.
conv_env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "TMPDIR": str(SCRATCH)}
with kh.build_heartbeat("convert.f16"):
    subprocess.check_call([sys.executable, str(converter),
                           "--model-dir", str(SRC), "--output", str(F16), "--fp16"], env=conv_env)
print(f"[f16] {F16.stat().st_size/1e9:.2f} GB", flush=True)
kh.step("f16_done", size_gb=round(F16.stat().st_size / 1e9, 2))

# %% [code]
# ── VALIDATION: stacked layout + byte-equivalence vs source experts ──
import numpy as np, json  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from safetensors import safe_open  # noqa: E402

def _fail(msg):
    kh.step("VALIDATION_FAILED", reason=msg)
    raise SystemExit(f"VALIDATION FAILED: {msg}")

r = GGUFReader(str(F16))
names = {t.name for t in r.tensors}
exps = sorted(n for n in names if n.endswith("_exps.weight") and ".ffn_" in n)
peri = [n for n in names if ".exp." in n]
print(f"[val] {len(exps)} stacked expert tensors, {len(peri)} per-expert", flush=True)
if not exps:
    _fail("no stacked expert tensors emitted")
if peri:
    _fail(f"per-expert tensors leaked: {peri[:5]}")

def gguf_tensor(name):
    return next(t for t in r.tensors if t.name == name)

# Map source expert tensor name -> shard file (index.json weight_map, else single file).
idx_path = SRC / "model.safetensors.index.json"
if idx_path.exists():
    weight_map = json.load(open(idx_path))["weight_map"]
else:
    single = next(SRC.glob("*.safetensors"))
    weight_map = None
    SINGLE = str(single)

def read_src_expert(li, proj, e):
    key = f"model.layers.{li}.mlp.experts.{e}.{proj}_proj.weight"
    shard = str(SRC / weight_map[key]) if weight_map else SINGLE
    with safe_open(shard, framework="pt") as f:  # np fails on bf16; pt->float is bit-exact widen
        t = f.get_tensor(key)
    return t.float().numpy().astype(np.float16)  # [out, in]

# Determine which layers are MoE from the emitted names.
moe_layers = sorted({int(n.split(".")[2]) for n in exps})
checks = []
for li in (moe_layers[0], moe_layers[len(moe_layers) // 2], moe_layers[-1]):
    for proj, e in (("gate", 0), ("up", 5), ("down", 1)):
        checks.append((li, proj, e))
for (li, proj, e) in checks:
    gt = gguf_tensor(f"l.blk.{li}.ffn_{proj}_exps.weight")
    inn, out, ne = (int(x) for x in gt.shape)  # ggml ne=[in,out,n_exp]
    slice_e = np.array(gt.data, dtype=np.float16).reshape(ne, out, inn)[e]  # [out, in]
    src = read_src_expert(li, proj, e)
    if slice_e.shape != src.shape:
        _fail(f"shape mismatch {li} {proj} e{e}: gguf {slice_e.shape} vs src {src.shape}")
    if not np.array_equal(slice_e, src):
        d = int(np.sum(slice_e.view(np.uint16) != src.view(np.uint16)))
        _fail(f"expert bytes differ {li} {proj} e{e}: {d}/{slice_e.size}")
    print(f"[val] layer {li} {proj} e{e} {slice_e.shape}: byte-identical ✓", flush=True)
kh.step("validated", stacked=len(exps), checks=len(checks))

# %% [code]
Q4 = WORK / "unlimited-ocr-q4_k-stacked.gguf"
kh.step("quantize.q4k")
with kh.build_heartbeat("quantize.q4k"):
    subprocess.check_call([str(QUANTIZE), str(F16), str(Q4), "q4_k"])
print(f"[q4_k] {Q4.stat().st_size/1e9:.2f} GB", flush=True)
rq = GGUFReader(str(Q4))
qexps = [t for t in rq.tensors if t.name.endswith("_exps.weight")]
if len(qexps) != len(exps):
    _fail("q4_k lost stacked expert tensors")
print(f"[q4_k] expert types: {sorted({t.tensor_type.name for t in qexps})}", flush=True)
kh.step("q4k_done", size_gb=round(Q4.stat().st_size / 1e9, 2))

# %% [code]
HF_REPO = "cstr/unlimited-ocr-crispembed-GGUF"
if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    try:
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[up] repo: {e}", flush=True)
    for path, name, msg in [
        (Q4, "unlimited-ocr-q4_k-stacked.gguf",
         "q4_k stacked MoE experts — ffn_*_exps 3D tensors, byte-identical to per-expert"),
        (F16, "unlimited-ocr-f16-stacked.gguf", "f16 stacked MoE experts — canonical source for requant"),
    ]:
        if path.exists():
            print(f"[up] uploading {name} ({path.stat().st_size/1e9:.1f} GB)", flush=True)
            with kh.build_heartbeat(f"upload.{name}"):
                api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                                repo_id=HF_REPO, repo_type="model", commit_message=msg)
            print(f"[up] uploaded {name}", flush=True)
    kh.step("uploaded")
else:
    print("[up] no HF token — skipping upload", flush=True)
    kh.step("upload_skipped")

kh.step("done")
print("DONE", flush=True)
