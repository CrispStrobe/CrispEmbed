#!/usr/bin/env python3
"""LFM2.5-VL: our decode vs the PYTHON BLUEPRINT's, across real documents.

Every check so far compared our output to a hand transcription. That measures
two things at once — whether the port is faithful, and whether the MODEL is any
good at the page — and a CER of 0.02 cannot tell you which. This kernel removes
the ambiguity by running the blueprint itself:

    text_hf   = Lfm2VlForConditionalGeneration.generate(...)   # the blueprint
    text_ours = crispembed --ocr ...                           # the port

and reporting CER(ours, hf) alongside CER(ours, gt) and CER(hf, gt). CER(ours,
hf) is the port-fidelity number; the gap between CER(hf, gt) and CER(ours, gt)
is how much of any error is the model's rather than ours.

Nine CC0 fixtures actually split under the NaFlex trigger, and they are varied
on purpose: Latin print, German Fraktur, Arabic, handwriting, a formula
photograph, a receipt, and a 10-image page. Ground truth exists for only three,
which is exactly why the blueprint comparison matters — the other six are still
fully usable as a fidelity test.

Greedy decode on both sides, same prompt, same max tokens. A sampled decode
would make the comparison meaningless.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/lfm2vl_bp")
SCRATCH.mkdir(parents=True, exist_ok=True)

EMBED_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
EMBED_BRANCH = "feat/lfm2vl-multitile"
EMBED = SCRATCH / "CrispEmbed"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
# Kept out of the sibling position, but no longer because it BREAKS: that was
# CrispEmbed issue #50 and it is fixed (the four shared libraries now compile
# against CrispEmbed's src/core, verified on a P100 in
# chr1s4/crispembed-punc-rerank-cuda, CUDA included). This kernel does not use
# audio / punc / lid / truecase, so staying out of the sibling layout just
# avoids building four libraries it has no use for.
CRISPASR = Path("/tmp/lfm2vl_bp_harness") / "CrispASR"
BUILD = EMBED / "build"

PROGRESS = WORK / "progress.txt"
RESULTS = WORK / "lfm2_vl_blueprint_ab_results.json"

MODEL_REPO = "LiquidAI/LFM2.5-VL-3B-GGUF"
HF_MODEL = "LiquidAI/LFM2.5-VL-3B"
PROMPT = "OCR this image. Output the text content."
MAX_TOKENS = 1600

# Every CC0 fixture that SPLITS, i.e. exercises multi-tile.
FIXTURES = [
    "commons_test_ocr_document.jpg",   # 1920x2485, 2x3, Latin print, GT 2981
    "german_official_print.jpg",       # 1920x2518, 2x3, Fraktur, GT 1009
    "receipt_historical.png",          # 768x1552, 2x4, receipt, GT 768
    "german_official_document.jpg",    # 960x1280, 2x3
    "handwritten_letter.jpg",          # 1920x1920, 3x3, 10 images
    "arabic_handwriting.jpg",          # 1275x1650, 2x3, RTL
    "public_domain_formula_photo.jpg", # 1600x1200, 3x2
    "german_kurrent_handwriting.jpg",  # 1064x672, 3x2
]
FIXTURE_DIR = "tests/regression/images/cc0"

results = {"status": "RUNNING", "fixtures": {}}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def sh(cmd, check=True, capture=False, env=None):
    log(f"$ {cmd}")
    if capture:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
        if check and r.returncode != 0:
            log(f"  rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            raise RuntimeError(cmd)
        return r
    r = subprocess.run(cmd, shell=True, env=env)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd} -> {r.returncode}")
    return r


def save():
    results["elapsed_s"] = round(time.time() - T0, 1)
    RESULTS.write_text(json.dumps(results, indent=2))


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def score(hyp, ref):
    h, r_ = norm(hyp), norm(ref)
    if not r_:
        return {}
    hw, rw = h.split(), r_.split()
    return {"cer": round(levenshtein(h, r_) / max(len(r_), 1), 4),
            "wer": round(levenshtein(hw, rw) / max(len(rw), 1), 4)}


T0 = time.time()
log("=== LFM2.5-VL: port vs Python blueprint, real documents ===")

if not CRISPASR.exists():
    CRISPASR.parent.mkdir(parents=True, exist_ok=True)
    sh(f"git clone --depth 1 {CRISPASR_URL} {CRISPASR}")
sys.path.insert(0, str(CRISPASR / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
HF_TOKEN = kh.resolve_hf_token(require=True)
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

if not EMBED.exists():
    sh(f"git clone --depth 1 --recursive --shallow-submodules -b {EMBED_BRANCH} {EMBED_URL} {EMBED}")
results["commit"] = sh(f"git -C {EMBED} rev-parse HEAD", capture=True).stdout.strip()
log(f"CrispEmbed @ {results['commit']}")

# ── Build ────────────────────────────────────────────────────────────────
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
# Not a workaround any more — see the clone comment above. Pinning these to a
# path that cannot exist skips four libraries this kernel does not use.
NO_SIBLING = "/nonexistent/crispasr"
flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags() + [
    f"-DCRISP_AUDIO_DIR={NO_SIBLING}/crisp_audio",
    f"-DCRISP_PUNC_DIR={NO_SIBLING}/crisp_punc",
    f"-DCRISP_LID_DIR={NO_SIBLING}/crisp_lid",
    f"-DCRISP_TRUECASE_DIR={NO_SIBLING}/crisp_truecase",
]
BUILD.mkdir(exist_ok=True)
sh(f"cmake -S {EMBED} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release " + " ".join(flags))
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispembed "
                        f"-j{kh.safe_build_jobs(gpu=True)}")
CRISPEMBED = BUILD / "crispembed"
if not CRISPEMBED.exists():
    CRISPEMBED = BUILD / "bin" / "crispembed"
log(f"built: {CRISPEMBED}")

# ── Models ───────────────────────────────────────────────────────────────
sh("pip install -q gguf Pillow huggingface_hub", check=False)
sh("pip install -q --no-deps 'transformers>=5.0.0'", check=False)
sh("pip install -q --no-deps accelerate", check=False)
sh("pip install -q torchvision", check=False)
from huggingface_hub import hf_hub_download  # noqa: E402

MODELS = SCRATCH / "models"
MODELS.mkdir(exist_ok=True)
Q4 = Path(hf_hub_download(MODEL_REPO, "LFM2.5-VL-3B-Q4_K_M.gguf", local_dir=str(MODELS), token=HF_TOKEN))
hf_hub_download(MODEL_REPO, "mmproj-LFM2.5-VL-3B-F16.gguf", local_dir=str(MODELS), token=HF_TOKEN)

# ── The blueprint side ───────────────────────────────────────────────────
log("loading the HF model (the blueprint)...")
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    try:
        torch.zeros(1, device="cuda")
    except Exception as e:
        log(f"CUDA probe failed ({e}); blueprint runs on CPU")
        device = "cpu"
if device == "cuda":
    major, _ = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if major >= 8 else torch.float16
else:
    dtype = torch.float32
log(f"blueprint device={device} dtype={dtype}")
results["blueprint_device"] = str(device)
results["blueprint_dtype"] = str(dtype)

processor = AutoProcessor.from_pretrained(HF_MODEL)
model = AutoModelForImageTextToText.from_pretrained(
    HF_MODEL, torch_dtype=dtype, device_map=device if device == "cuda" else None)
if device == "cpu":
    model = model.to(device)
model.eval()


def blueprint_ocr(path):
    img = Image.open(path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": PROMPT}]}]
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = PROMPT
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
    n_prompt = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        # Greedy on both sides or the comparison means nothing.
        out = model.generate(**inputs, max_new_tokens=MAX_TOKENS,
                             do_sample=False, num_beams=1)
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True), n_prompt


# ── Ground truth ─────────────────────────────────────────────────────────
gt_raw = json.loads((EMBED / FIXTURE_DIR / "ground_truth.json").read_text())
GT = {}
for rec in gt_raw.get("records", []):
    n = rec.get("name") or rec.get("file")
    if n:
        GT[n] = rec.get("text") or rec.get("ground_truth") or ""

# ── Run both sides ───────────────────────────────────────────────────────
BASE_ENV = {**os.environ, "CRISPEMBED_ACCEPT_LFM_LICENSE": "1", "GGML_CUDA_NO_VMM": "1"}

for name in FIXTURES:
    path = EMBED / FIXTURE_DIR / name
    if not path.exists():
        log(f"SKIP {name}: not in the repo")
        continue
    log(f"--- {name} ---")
    entry = {}

    t0 = time.time()
    try:
        hf_text, n_prompt = blueprint_ocr(path)
        entry["blueprint_seconds"] = round(time.time() - t0, 1)
        entry["blueprint_prompt_tokens"] = n_prompt
        entry["blueprint_text"] = hf_text
    except Exception as e:
        log(f"  blueprint FAILED: {e}")
        entry["blueprint_error"] = str(e)[:400]
        hf_text = None

    # Two arms of OUR decode. The blueprint's generation_config.json carries no
    # no_repeat_ngram_size (so: no constraint) while we default to 5, which
    # FORCES a different token wherever the model legitimately repeats — table
    # rows, a formula, unreadable handwriting. That is a decode-recipe
    # divergence, not a numerical one, and it has to be measured rather than
    # argued about.
    def run_ours(ngram):
        env = dict(BASE_ENV)
        if ngram is not None:
            env["LFM2_VL_NO_REPEAT_NGRAM"] = str(ngram)
        cmd = f"{CRISPEMBED} --ocr {path} -m {Q4} --ocr-max-tokens {MAX_TOKENS} -v"
        t0 = time.time()
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
        return r, round(time.time() - t0, 1)

    r, secs = run_ours(None)
    entry["ours_seconds"] = secs
    entry["ours_rc"] = r.returncode
    ours = r.stdout.strip()
    entry["ours_text"] = ours
    m = re.search(r"prompt: (\d+) tokens", r.stderr)
    if m:
        entry["ours_prompt_tokens"] = int(m.group(1))

    r0, secs0 = run_ours(0)
    ours_ng0 = r0.stdout.strip()
    entry["ours_ngram0_text"] = ours_ng0
    entry["ours_ngram0_seconds"] = secs0
    entry["ours_ngram0_rc"] = r0.returncode

    gt = GT.get(name, "")
    if hf_text is not None and ours:
        entry["ours_vs_blueprint"] = score(ours, hf_text)
    if hf_text is not None and ours_ng0:
        entry["ours_ngram0_vs_blueprint"] = score(ours_ng0, hf_text)
    if gt and ours_ng0:
        entry["ours_ngram0_vs_gt"] = score(ours_ng0, gt)
    if gt:
        entry["ours_vs_gt"] = score(ours, gt)
        if hf_text is not None:
            entry["blueprint_vs_gt"] = score(hf_text, gt)
    entry["chars"] = {"ours": len(norm(ours)),
                      "blueprint": len(norm(hf_text or "")),
                      "gt": len(norm(gt))}

    log(f"  prompt tokens: ours={entry.get('ours_prompt_tokens')} blueprint={entry.get('blueprint_prompt_tokens')}")
    log(f"  ours_vs_blueprint={entry.get('ours_vs_blueprint')} "
        f"ours_vs_gt={entry.get('ours_vs_gt')} blueprint_vs_gt={entry.get('blueprint_vs_gt')}")
    log(f"  ngram0: vs_blueprint={entry.get('ours_ngram0_vs_blueprint')} "
        f"vs_gt={entry.get('ours_ngram0_vs_gt')}")
    log(f"  ours[:120]      = {ours[:120]!r}")
    log(f"  blueprint[:120] = {(hf_text or '')[:120]!r}")
    results["fixtures"][name] = entry
    save()

# ── Summary ──────────────────────────────────────────────────────────────
ok = [v for v in results["fixtures"].values() if v.get("ours_vs_blueprint")]
if ok:
    cers = [v["ours_vs_blueprint"]["cer"] for v in ok]
    ok0 = [v for v in results["fixtures"].values() if v.get("ours_ngram0_vs_blueprint")]
    cers0 = [v["ours_ngram0_vs_blueprint"]["cer"] for v in ok0]
    results["summary"] = {
        "n_compared": len(ok),
        "cer_ours_vs_blueprint_mean": round(sum(cers) / len(cers), 4),
        "cer_ours_vs_blueprint_max": round(max(cers), 4),
        "cer_ngram0_vs_blueprint_mean": round(sum(cers0) / len(cers0), 4) if cers0 else None,
        "cer_ngram0_vs_blueprint_max": round(max(cers0), 4) if cers0 else None,
        "prompt_tokens_match": all(
            v.get("ours_prompt_tokens") == v.get("blueprint_prompt_tokens") for v in ok),
    }
results["status"] = "DONE"
save()
log("=== summary ===")
log(json.dumps(results.get("summary", {}), indent=2))
