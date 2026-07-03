#!/usr/bin/env python3
"""CrispEmbed — splade-pp-en-v1 reconvert (with MLM head) + quantize + upload.

The uploaded splade-pp GGUFs contain ONLY the encoder — the MLM/vocab-decode head
(cls.predictions.*) was dropped, so `has_sparse=false` and --sparse fails. SPLADE
*is* that head, so the model was functionally broken. convert-bert-to-gguf.py now
auto-detects the MLM head (AutoModelForMaskedLM + _checkpoint_has_mlm_head), so a
clean reconvert restores sparse. This kernel reconverts, VERIFIES --sparse works
(fails loudly otherwise — never re-upload a still-broken GGUF), quantizes, uploads.
"""
import os, subprocess, sys
from pathlib import Path

WORK = Path("/kaggle/working")
_CRISPASR_DIR = WORK / "CrispASR"
if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
            "https://github.com/CrispStrobe/CrispASR.git", str(_CRISPASR_DIR)])
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    except Exception:
        pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh
kh.init_progress()
hf_token = kh.resolve_hf_token()
kh.step("harness_ready", hf_token_ok=bool(hf_token))

SRC_ID  = "prithivida/Splade_PP_en_v1"
HF_REPO = "cstr/splade-pp-en-v1-GGUF"
PREFIX  = "splade-pp-en-v1"

subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
    "torch", "transformers", "safetensors", "gguf", "huggingface_hub", "hf_transfer"])
kh.step("deps_installed")

REPO = WORK / "CrispEmbed"
if not REPO.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", "main",
        "https://github.com/CrispStrobe/CrispEmbed.git", str(REPO)])
    subprocess.check_call(["git", "-C", str(REPO), "submodule", "update", "--init", "--recursive"])
kh.step("cloned")

from huggingface_hub import snapshot_download, HfApi
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
scratch = Path("/tmp/splade-src"); scratch.mkdir(parents=True, exist_ok=True)
with kh.build_heartbeat("download.model"):
    src = snapshot_download(repo_id=SRC_ID, cache_dir=str(scratch), token=hf_token)
kh.step("model_downloaded", src=src)

# --- Diagnostic: what MLM-head keys does the SOURCE checkpoint actually have? ---
import torch, glob
diag = []
def _log(s):
    print(s, flush=True); diag.append(str(s))
sd_keys = []
for bp in glob.glob(str(Path(src) / "*.bin")) + glob.glob(str(Path(src) / "*.safetensors")):
    try:
        if bp.endswith(".bin"):
            sd_keys = list(torch.load(bp, map_location="cpu", weights_only=False).keys())
        else:
            from safetensors import safe_open
            with safe_open(bp, framework="pt") as f: sd_keys = list(f.keys())
        break
    except Exception as e:
        _log(f"probe {bp}: {e}")
_log(f"source has {len(sd_keys)} keys; MLM-ish keys: " +
     str([k for k in sd_keys if any(x in k for x in ('cls.predict', 'lm_head', 'predictions', 'decoder', 'mlm'))][:20]))

# --- Convert (auto-detects the SPLADE MLM head) — capture stdout ---
OUT_BASE = WORK / f"{PREFIX}.gguf"
with kh.build_heartbeat("convert"):
    cvt = subprocess.run([sys.executable, str(REPO / "models" / "convert-bert-to-gguf.py"),
        "--model", str(src), "--output", str(OUT_BASE), "--crisp", "--dtype", "f32"],
        capture_output=True, text=True)
_log("converter stdout tail:\n" + cvt.stdout[-1500:])
_log("converter stderr tail:\n" + cvt.stderr[-800:])
if cvt.returncode != 0:
    raise RuntimeError(f"convert rc={cvt.returncode}:\n{cvt.stderr[-1200:]}")
# inspect produced GGUF tensors for the MLM head
import gguf as _g
_r = _g.GGUFReader(str(OUT_BASE))
_names = [t.name for t in _r.tensors]
_head = [n for n in _names if any(x in n for x in ('mlm', 'sparse', 'cls'))]
_log(f"GGUF has {len(_names)} tensors; head tensors: {_head}")
# always upload the diagnostic so we see it even if a later step fails
try:
    (WORK / "splade-diag.txt").write_text("\n".join(diag))
    if hf_token:
        from huggingface_hub import HfApi as _A
        _A(token=hf_token).upload_file(path_or_fileobj=str(WORK / "splade-diag.txt"),
            path_in_repo="splade-diag.txt", repo_id=HF_REPO, repo_type="model",
            commit_message="splade reconvert diagnostic")
except Exception as e:
    print("diag upload failed:", e, flush=True)
kh.step("converted", size_mb=round(OUT_BASE.stat().st_size / 1e6, 1), head_tensors=_head)

# --- Build CLI + quantizer ---
kh.install_build_toolchain()
BUILD = REPO / "build"; BUILD.mkdir(exist_ok=True)
kh.sh_with_progress(f"cmake -G Ninja -S {REPO} -B {BUILD} -DCMAKE_BUILD_TYPE=Release "
                    f"-DGGML_CUDA=OFF " + " ".join(kh.cache_and_link_flags()))
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"cmake --build {BUILD} --target crispembed-cli crispembed-quantize "
                        f"-j{kh.safe_build_jobs(gpu=False)}")
kh.step("built")
CLI = BUILD / "crispembed"; QUANT = BUILD / "crispembed-quantize"

# --- VERIFY sparse works on the reconverted base (fail loudly if still broken) ---
r = subprocess.run([str(CLI), "-m", str(OUT_BASE), "--json", "--sparse", "machine learning models"],
                   capture_output=True, text=True)
ok = r.returncode == 0 and '"sparse"' in r.stdout and "does not support sparse" not in r.stderr
kh.step("sparse_check", ok=ok, stderr_tail=r.stderr[-300:])
if not ok:
    raise RuntimeError(f"reconverted splade STILL has no sparse head; rc={r.returncode}; "
                       f"stderr:\n{r.stderr[-800:]}\nstdout head:\n{r.stdout[:200]}")
print("[verify] sparse OK:", r.stdout[:180], flush=True)

# --- Quantize ---
OUT_Q8 = WORK / f"{PREFIX}-q8_0.gguf"; OUT_Q4 = WORK / f"{PREFIX}-q4_k.gguf"
with kh.build_heartbeat("quantize.q8"):
    subprocess.check_call([str(QUANT), str(OUT_BASE), str(OUT_Q8), "q8_0"])
with kh.build_heartbeat("quantize.q4k"):
    subprocess.check_call([str(QUANT), str(OUT_BASE), str(OUT_Q4), "q4_k"])
# sanity: sparse must still work on q8 (dequant-safe head read)
rq = subprocess.run([str(CLI), "-m", str(OUT_Q8), "--json", "--sparse", "test query"],
                    capture_output=True, text=True)
if not ('"sparse"' in rq.stdout and rq.returncode == 0):
    raise RuntimeError(f"q8 sparse broken after quantize; stderr:\n{rq.stderr[-600:]}")
kh.step("quantized_and_verified")

# --- Upload (replace the broken base + add quants) ---
if hf_token:
    api = HfApi(token=hf_token)
    for path, name, msg in [
        (OUT_BASE, f"{PREFIX}.gguf",      "F32 reconvert WITH MLM head (restores sparse)"),
        (OUT_Q8,   f"{PREFIX}-q8_0.gguf", "Q8_0 reconvert WITH MLM head (restores sparse)"),
        (OUT_Q4,   f"{PREFIX}-q4_k.gguf", "Q4_K reconvert WITH MLM head (restores sparse)"),
    ]:
        sz = path.stat().st_size / 1e6
        with kh.build_heartbeat(f"upload.{name}"):
            api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                            repo_id=HF_REPO, repo_type="model", commit_message=msg)
        print(f"[upload] {name} ({sz:.1f} MB)", flush=True)
    kh.step("uploaded")
else:
    print("[upload] SKIP — no HF token", flush=True)

kh.step("all_done")
print("[DONE] splade-pp reconvert + sparse restored", flush=True)
