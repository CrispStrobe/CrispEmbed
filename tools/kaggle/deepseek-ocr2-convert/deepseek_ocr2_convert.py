# %% [markdown]
# # CrispEmbed — DeepSeek-OCR-2 GGUF reconvert (stacked MoE experts, #4)
#
# Reconverts `deepseek-ai/DeepSeek-OCR-2` with the new converter that emits the
# 64 routed experts as STACKED 3D tensors (`l.blk.{i}.ffn_{gate,up,down}_exps.weight`,
# ggml ne=[in,out,n_exp]) instead of per-expert 2D tensors. The runtime then loads
# them directly and skips the +1.3 GB per-expert→stacked duplication.
#
# Produces f16 + q4_k, validates the stacked expert bytes are byte-identical to the
# source experts, and uploads to `cstr/deepseek-ocr2-crispembed-GGUF` under NEW
# `-stacked` filenames (does NOT clobber the rev-pinned regression GGUF).

# %% [code]
import os, sys, subprocess, shutil
from pathlib import Path

WORK = Path("/kaggle/working")
REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "feat/ds-ocr2-stacked-experts")
EMBED_DIR = WORK / "CrispEmbed"
BUILD_DIR = EMBED_DIR / "build"

# scratch with space for the ~6 GB source + ~6 GB f16
SCRATCH = None
for c in ("/kaggle/temp", "/tmp"):
    if os.path.isdir(c):
        SCRATCH = Path(c) / "ds-ocr2"
        break
SCRATCH.mkdir(parents=True, exist_ok=True)

# ── clone CrispASR for kaggle_harness ──
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
kh.sh("pip install -q huggingface_hub hf_transfer gguf safetensors pillow numpy || true", check=False)
kh.install_build_toolchain()

# ── build crispembed-quantize (CPU only; no CUDA needed) ──
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

HF_SRC = "deepseek-ai/DeepSeek-OCR-2"
kh.step("download.source", model=HF_SRC)
with kh.build_heartbeat("model.download"):
    SRC = snapshot_download(repo_id=HF_SRC, cache_dir=str(SCRATCH),
                            allow_patterns=["config.json", "tokenizer.json", "*.safetensors", "*.safetensors.index.json"])
SRC = Path(SRC)
SRC_ST = SRC / "model-00001-of-000001.safetensors"
assert SRC_ST.exists(), f"source safetensors not found in {SRC}: {list(SRC.iterdir())}"
kh.step("downloaded", src=str(SRC))

# %% [code]
converter = EMBED_DIR / "models" / "convert-deepseek-ocr2-to-gguf.py"
F16 = SCRATCH / "deepseek-ocr2-f16-stacked.gguf"
kh.step("convert.f16")
with kh.build_heartbeat("convert.f16"):
    subprocess.check_call([sys.executable, str(converter),
                           "--model", str(SRC_ST),
                           "--config", str(SRC / "config.json"),
                           "--tokenizer", str(SRC / "tokenizer.json"),
                           "--output", str(F16), "--fp16"])
print(f"[f16] {F16.stat().st_size/1e9:.2f} GB", flush=True)
kh.step("f16_done", size_gb=round(F16.stat().st_size / 1e9, 2))

# %% [code]
# ── VALIDATION: stacked layout + byte-equivalence vs source experts ──
import numpy as np, json, struct  # noqa: E402
from gguf import GGUFReader  # noqa: E402

def _fail(msg):
    kh.step("VALIDATION_FAILED", reason=msg)
    raise SystemExit(f"VALIDATION FAILED: {msg}")

r = GGUFReader(str(F16))
names = {t.name for t in r.tensors}
exps = sorted(n for n in names if n.endswith("_exps.weight") and ".ffn_" in n)
peri = [n for n in names if ".exp." in n]
print(f"[val] {len(exps)} stacked expert tensors, {len(peri)} per-expert (want 33 / 0)", flush=True)
if len(exps) != 33:
    _fail(f"expected 33 stacked expert tensors (11 MoE layers x 3), got {len(exps)}: {exps}")
if peri:
    _fail(f"per-expert tensors leaked into GGUF: {peri[:5]}")

def gguf_tensor(name):
    return next(t for t in r.tensors if t.name == name)

# gate/up project hidden(1280)->moe_inter(896): ne=[in=1280,out=896,n_exp=64].
# down projects moe_inter(896)->hidden(1280): ne=[in=896,out=1280,n_exp=64].
t0 = gguf_tensor("l.blk.1.ffn_gate_exps.weight")
if list(int(x) for x in t0.shape) != [1280, 896, 64]:
    _fail(f"gate_exps shape {list(t0.shape)} != [1280,896,64] (ggml ne=[in,out,n_exp])")
td = gguf_tensor("l.blk.1.ffn_down_exps.weight")
if list(int(x) for x in td.shape) != [896, 1280, 64]:
    _fail(f"down_exps shape {list(td.shape)} != [896,1280,64] (ggml ne=[in,out,n_exp])")

# Byte-equivalence: read source experts directly (bf16 -> f32 -> f16, same path as
# the converter) and compare to the corresponding stacked slice e.
with open(SRC_ST, "rb") as f:
    hlen = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(hlen))
hdr.pop("__metadata__", None)
base = 8 + hlen

def read_src_expert(li, proj, e):
    key = f"model.layers.{li}.mlp.experts.{e}.{proj}_proj.weight"
    info = hdr[key]
    o0, o1 = info["data_offsets"]
    with open(SRC_ST, "rb") as f:
        f.seek(base + o0)
        raw = f.read(o1 - o0)
    dt = info["dtype"]
    if dt == "BF16":
        u16 = np.frombuffer(raw, dtype=np.uint16)
        arr = (u16.astype(np.uint32) << 16).view(np.float32)
    elif dt == "F16":
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    elif dt == "F32":
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        _fail(f"unexpected source dtype {dt}")
    return arr.reshape(info["shape"]).astype(np.float16)  # [out, in]

checks = [(1, "gate", 0), (1, "up", 5), (6, "down", 63), (11, "gate", 30), (11, "down", 1)]
for (li, proj, e) in checks:
    gt = gguf_tensor(f"l.blk.{li}.ffn_{proj}_exps.weight")
    inn, out, ne = (int(x) for x in gt.shape)  # ggml ne=[in,out,n_exp]
    slice_e = np.array(gt.data, dtype=np.float16).reshape(ne, out, inn)[e]  # [out, in]
    src = read_src_expert(li, proj, e)  # source [out, in]
    if slice_e.shape != src.shape:
        _fail(f"shape mismatch layer {li} {proj} e{e}: gguf {slice_e.shape} vs src {src.shape}")
    if not np.array_equal(slice_e, src):
        d = int(np.sum(slice_e.view(np.uint16) != src.view(np.uint16)))
        _fail(f"expert bytes differ at layer {li} {proj} e{e}: {d}/{slice_e.size} elems")
    print(f"[val] layer {li} {proj} e{e} {slice_e.shape}: byte-identical ✓", flush=True)
kh.step("validated", stacked=len(exps), checks=len(checks))

# %% [code]
# ── quantize -> q4_k, sanity-check it loads + keeps stacked shape ──
Q4 = WORK / "deepseek-ocr2-q4_k-stacked.gguf"
kh.step("quantize.q4k")
with kh.build_heartbeat("quantize.q4k"):
    subprocess.check_call([str(QUANTIZE), str(F16), str(Q4), "q4_k"])
print(f"[q4_k] {Q4.stat().st_size/1e9:.2f} GB", flush=True)

rq = GGUFReader(str(Q4))
qnames = {t.name for t in rq.tensors}
qt = next(t for t in rq.tensors if t.name == "l.blk.1.ffn_gate_exps.weight")
print(f"[q4_k] gate_exps type={qt.tensor_type.name} shape={list(int(x) for x in qt.shape)}", flush=True)
if list(int(x) for x in qt.shape) != [1280, 896, 64]:
    _fail(f"q4_k stacked shape changed: {list(qt.shape)}")
if not str(qt.tensor_type.name).startswith("Q4_K"):
    _fail(f"q4_k expert tensor not quantized (type={qt.tensor_type.name})")
if sum(1 for n in qnames if n.endswith("_exps.weight")) != 33:
    _fail("q4_k lost stacked expert tensors")
kh.step("q4k_done", size_gb=round(Q4.stat().st_size / 1e9, 2))

# %% [code]
# ── non-clobbering upload to cstr/deepseek-ocr2-crispembed-GGUF ──
HF_REPO = "cstr/deepseek-ocr2-crispembed-GGUF"
if hf_token:
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    try:
        api.create_repo(HF_REPO, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"[up] repo: {e}", flush=True)
    for path, name, msg in [
        (Q4, "deepseek-ocr2-q4_k-stacked.gguf",
         "q4_k stacked MoE experts (#4) — ffn_*_exps 3D tensors, byte-identical to per-expert"),
        (F16, "deepseek-ocr2-f16-stacked.gguf",
         "f16 stacked MoE experts (#4) — canonical source for requant"),
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
