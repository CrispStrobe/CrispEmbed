#!/usr/bin/env python3
"""LFM2.5-VL multi-tile NaFlex — acceptance run on Kaggle (chr1s4).

Why this is not on the VPS: the vision encoder costs ~300 s per 1024-patch
image there, and a split A4 page is SEVEN images. One arm of this A/B is
~40 minutes of encode alone before the prefill, and the gate needs four arms
(multi-tile off/on x Q4_K/F16, dev-guide rule 4.2). On a T4/P100 the same work
is minutes.

What it does, in the order the dev guide's port pipeline prescribes:

  1. Build crispembed with CUDA.
  2. Dump a per-stage reference from the Python blueprint for a SPLITTING
     fixture, and upload it to HF immediately -- the upload IS the checkpoint,
     so a later failure never loses the GPU-hours already spent.
  3. Prompt-token parity FIRST. The runtime's prompt ids must be byte-identical
     to the HF processor's. Everything the tiling decides lands there and
     nowhere a cosine can see it; if this fails, every per-stage number below
     is the wrong prompt agreeing with the wrong prompt.
  4. Per-stage diff from the earliest stage, per tile.
  5. The only acceptance test that counts (HARD RULE 3): the decoded text,
     both arms, both quants, scored as CER/WER against the fixture's
     transcribed ground truth.

`kernels_output` never returns stdout/stderr, so everything worth reading is
mirrored into /kaggle/working/progress.txt and the results JSON.
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SCRATCH = Path("/tmp/lfm2vl_mt")  # big artifacts off /kaggle/working (~20 GB)
SCRATCH.mkdir(parents=True, exist_ok=True)

EMBED_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
EMBED_BRANCH = "feat/lfm2vl-multitile"
EMBED = SCRATCH / "CrispEmbed"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
# ⚠ NOT a sibling of CrispEmbed. CrispEmbed's CMakeLists probes
# `${CMAKE_CURRENT_SOURCE_DIR}/../CrispASR/{crisp_audio,crisp_punc,crisp_lid,
# crisp_truecase}` and builds them when present — the dev-machine layout. On a
# kernel that clones both repos side by side that fires by accident, and
# CrispASR's crisp_punc then compiles against CrispEmbed's src/core and dies on
# a missing `core/ggml_cpu_backend.h`. Cost one run (kernel v1) to learn. We
# only want the harness from CrispASR, so keep it out of the probe's reach --
# and pin the four dirs explicitly below so the layout cannot matter either way.
CRISPASR = Path("/tmp/lfm2vl_harness") / "CrispASR"
BUILD = EMBED / "build"

PROGRESS = WORK / "progress.txt"
RESULTS = WORK / "lfm2_vl_multitile_results.json"

# The fixture must SPLIT and must have a transcribed ground truth, or neither
# half of this run means anything. commons_test_ocr_document.jpg is 1920x2485
# -> a 2x3 grid + thumbnail, 7 encoded images, 1788 image tokens, and carries a
# 2981-character manual transcription in ground_truth.json.
FIXTURE = "tests/regression/images/cc0/commons_test_ocr_document.jpg"
FIXTURE_STEM = "commons_test_ocr_document"

# The 500x650 receipt does NOT split (327680 rounded px < the 524288 trigger),
# so it is the regression canary: it must come out byte-identical with the gate
# on and off. It is not the multi-tile test.
CANARY = "tests/regression/images/cc0/commons_example_receipt.png"
CANARY_EXPECT = "Jackson-Washington\n6640 Ortiz Cove, Markmouth"

MODEL_REPO = "LiquidAI/LFM2.5-VL-3B-GGUF"
HF_REF_REPO = "cstr/crispembed-regression-fixtures"
HF_REF_PATH = f"lfm2_vl/{FIXTURE_STEM}/ref.gguf"

PROMPT = "OCR this image. Output the text content."
MAX_TOKENS = 1600  # 900 truncated at ~2015 of the 2981 ground-truth chars,
                   # which inflates CER for BOTH arms and muddies the absolute number

results = {"status": "RUNNING", "stages": {}}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def sh(cmd, check=True, capture=False, env=None, timeout=None):
    log(f"$ {cmd}")
    if capture:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           env=env, timeout=timeout)
        if check and r.returncode != 0:
            log(f"  rc={r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
            raise RuntimeError(f"command failed: {cmd}")
        return r
    r = subprocess.run(cmd, shell=True, env=env, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {cmd}")
    return r


def save_results():
    results["elapsed_s"] = round(time.time() - T0, 1)
    RESULTS.write_text(json.dumps(results, indent=2))


T0 = time.time()
log("=== LFM2.5-VL multi-tile NaFlex acceptance run ===")

# ── 0. Clone + harness + token, before anything expensive ─────────────────
#
# resolve_hf_token(require=True) comes FIRST: a missing token then aborts in
# seconds instead of a finished run losing every artifact to an upload 401.
CRISPASR.parent.mkdir(parents=True, exist_ok=True)
if not CRISPASR.exists():
    sh(f"git clone --depth 1 {CRISPASR_URL} {CRISPASR}")
sys.path.insert(0, str(CRISPASR / "tools" / "kaggle"))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
HF_TOKEN = kh.resolve_hf_token(require=True)
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
log("HF token resolved")

if not EMBED.exists():
    sh(f"git clone --depth 1 --recursive --shallow-submodules "
       f"-b {EMBED_BRANCH} {EMBED_URL} {EMBED}")
results["commit"] = sh(f"git -C {EMBED} rev-parse HEAD", capture=True).stdout.strip()
log(f"CrispEmbed @ {results['commit']} ({EMBED_BRANCH})")

# ── 1. Build with CUDA ────────────────────────────────────────────────────
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
log(f"CUDA arch: {arch}")
results["cuda_arch"] = arch

# Belt and braces on the sibling-probe trap above: point every optional
# CrispASR subdirectory at a path that cannot exist. Each has a clean
# "not found" fallback, and none of punctuation / LID / truecasing is on the
# LFM2-VL OCR path, so this costs nothing and removes a whole class of
# layout-dependent build failures.
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
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target crispembed test-lfm2-tiling "
        f"-j{kh.safe_build_jobs(gpu=True)}")

CRISPEMBED = BUILD / "crispembed"
if not CRISPEMBED.exists():
    CRISPEMBED = BUILD / "bin" / "crispembed"
if not CRISPEMBED.exists():
    log("FATAL: crispembed binary not found after build")
    results["status"] = "FAIL_BUILD"
    save_results()
    sys.exit(1)
log(f"built: {CRISPEMBED}")

# The hermetic layout guard. It needs no weights, so run it before spending a
# single GPU-second: if the tiling math is wrong here, nothing below is worth
# measuring.
r = sh(f"{BUILD}/test-lfm2-tiling", capture=True, check=False)
log(r.stdout.strip()[-2000:])
results["stages"]["tiling_guard"] = {
    "rc": r.returncode,
    "tail": r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "",
}
if r.returncode != 0:
    log("FATAL: the hermetic tiling guard failed — stopping before the GPU work")
    results["status"] = "FAIL_TILING_GUARD"
    save_results()
    sys.exit(1)
save_results()

# ── 2. Models ─────────────────────────────────────────────────────────────
MODELS = SCRATCH / "models"
MODELS.mkdir(exist_ok=True)
sh("pip install -q gguf Pillow huggingface_hub", check=False)
from huggingface_hub import hf_hub_download, HfApi  # noqa: E402


def fetch(repo, fname):
    log(f"downloading {repo}/{fname}")
    p = hf_hub_download(repo, fname, local_dir=str(MODELS), token=HF_TOKEN)
    log(f"  -> {p} ({Path(p).stat().st_size / 1e9:.2f} GB)")
    return Path(p)


Q4 = fetch(MODEL_REPO, "LFM2.5-VL-3B-Q4_K_M.gguf")
MMPROJ = fetch(MODEL_REPO, "mmproj-LFM2.5-VL-3B-F16.gguf")
# The F16 LLM is the second A/B arm (quant amplifies divergence, rule 4.2).
try:
    F16 = fetch(MODEL_REPO, "LFM2.5-VL-3B-F16.gguf")
except Exception as e:
    log(f"F16 LLM unavailable ({e}); falling back to Q8_0 for the second arm")
    try:
        F16 = fetch(MODEL_REPO, "LFM2.5-VL-3B-Q8_0.gguf")
    except Exception as e2:
        log(f"Q8_0 unavailable too ({e2}); running the Q4_K arm only")
        F16 = None

# ── 3. Reference dump from the Python blueprint, then upload immediately ──
log("=== reference dump (splitting fixture) ===")
# LFM2.5-VL needs transformers >= 5.0 for Lfm2VlForConditionalGeneration.
# --no-deps so pip cannot swap Kaggle's GPU-matched torch (a pip-upgraded
# torch drops sm_60 and kills P100).
sh("pip install -q --no-deps 'transformers>=5.0.0'", check=False)
sh("pip install -q --no-deps accelerate", check=False)
sh("pip install -q torchvision", check=False)  # the fast image processor needs it

REF = SCRATCH / "lfm2-vl-multitile-ref.gguf"
dump_cmd = (
    f"{sys.executable} {EMBED}/tools/dump_lfm2_vl_reference.py "
    f"--model LiquidAI/LFM2.5-VL-3B "
    f"--image {EMBED}/{FIXTURE} "
    f"--output {REF} "
    f"--max-vis-layers 4 --max-llm-layers 4 "
    f"--prompt {json.dumps(PROMPT)}"
)
r = sh(dump_cmd, check=False, capture=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
log(r.stdout[-6000:])

# A failed reference dump costs the per-stage diff, NOT the acceptance test.
# The decoded-output A/B is what HARD RULE 3 actually gates on and it needs no
# reference at all -- so carry on without it rather than throwing away a build
# and a 9 GB download. The likeliest failure here is bf16 on a pre-sm_80 GPU
# (Kaggle hands out P100s), which says nothing about the runtime.
REF_OK = r.returncode == 0 and REF.exists()
if not REF_OK:
    log(f"reference dump FAILED rc={r.returncode} — continuing WITHOUT per-stage parity")
    log(r.stderr[-4000:])
    results["stages"]["refdump"] = {
        "rc": r.returncode, "ok": False,
        "stderr_tail": r.stderr[-3000:],
        "note": "per-stage diff and prompt-token parity unavailable this run; "
                "the decoded-output A/B below still ran",
    }
else:
    results["stages"]["refdump"] = {
        "rc": 0, "ok": True, "size_mb": round(REF.stat().st_size / 1e6, 1),
    }
    log(f"reference: {REF.stat().st_size / 1e6:.1f} MB")

# Upload BEFORE the next crash-prone step — the upload is the checkpoint.
if REF_OK:
  try:
    HfApi(token=HF_TOKEN).upload_file(
        path_or_fileobj=str(REF), path_in_repo=HF_REF_PATH,
        repo_id=HF_REF_REPO, repo_type="dataset")
    log(f"reference uploaded to {HF_REF_REPO}/{HF_REF_PATH}")
    results["stages"]["refdump"]["hf_path"] = f"{HF_REF_REPO}/{HF_REF_PATH}"
  except Exception as e:
    log(f"WARNING: reference upload failed: {e}")
    results["stages"]["refdump"]["upload_error"] = str(e)
save_results()

# ── 4. Runner ─────────────────────────────────────────────────────────────
BASE_ENV = {
    **os.environ,
    "CRISPEMBED_ACCEPT_LFM_LICENSE": "1",
    "GGML_CUDA_NO_VMM": "1",
}


def run_ocr(model, image, multi_tile, diff_ref=None, max_tokens=MAX_TOKENS, extra=None, bicubic=False):
    env = dict(BASE_ENV)
    env["LFM2_VL_MULTI_TILE"] = "1" if multi_tile else "0"
    env["LFM2_VL_BICUBIC"] = "1" if bicubic else "0"
    if diff_ref:
        env["LFM2_VL_DIFF_REF"] = str(diff_ref)
    if extra:
        env.update(extra)
    cmd = (f"{CRISPEMBED} --ocr {image} -m {model} "
           f"--ocr-max-tokens {max_tokens} -v")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    dt = time.time() - t0
    return {
        "cmd": cmd,
        "multi_tile": bool(multi_tile),
        "rc": r.returncode,
        "seconds": round(dt, 1),
        "stdout": r.stdout,
        "stderr": r.stderr,
    }


def text_of(run):
    """The transcript is stdout; stderr carries the diagnostics."""
    return run["stdout"].strip()


def parse_diffs(stderr):
    out = {}
    for m in re.finditer(
            r"DIFF\s+(\S+)\s+cos_min=([0-9.]+)\s+max_abs=(\S+)\s+"
            r"\|mine\|=([0-9.]+)\s+\|ref\|=([0-9.]+)\s+(PASS|FAIL)", stderr):
        out[m.group(1)] = {
            "cos_min": float(m.group(2)), "max_abs": m.group(3),
            "mine_norm": float(m.group(4)), "ref_norm": float(m.group(5)),
            "verdict": m.group(6),
        }
    m = re.search(r"DIFF\s+prompt_token_ids\s+(PASS|FAIL)([^\n]*)", stderr)
    if m:
        out["prompt_token_ids"] = {"verdict": m.group(1), "detail": m.group(2).strip()}
    return out


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
    return re.sub(r"\s+", " ", s.strip())


def score(hyp, ref):
    h, r_ = norm(hyp), norm(ref)
    cer = levenshtein(h, r_) / max(len(r_), 1)
    hw, rw = h.split(), r_.split()
    wer = levenshtein(hw, rw) / max(len(rw), 1)
    return {"cer": round(cer, 4), "wer": round(wer, 4),
            "chars": len(h), "ref_chars": len(r_),
            "words": len(hw), "ref_words": len(rw)}


gt = json.loads(Path(EMBED / "tests/regression/images/cc0/ground_truth.json").read_text())
GROUND_TRUTH = ""
for rec in gt.get("records", []):
    # The records key the fixture by "name"/"file"; there is no "image" key.
    # Getting this wrong silently yields an empty ground truth, which then
    # suppresses every CER/WER number without failing anything -- exactly the
    # kind of quiet nothing rule 4a warns about. Assert it landed.
    if FIXTURE_STEM in (str(rec.get("name", "")) + str(rec.get("file", ""))):
        GROUND_TRUTH = rec.get("text") or rec.get("ground_truth") or ""
        break
if not GROUND_TRUTH:
    log(f"FATAL: no ground truth for {FIXTURE_STEM} — the A/B could not be scored")
    results["status"] = "FAIL_NO_GROUND_TRUTH"
    save_results()
    sys.exit(1)
log(f"ground truth: {len(GROUND_TRUTH)} chars")
results["ground_truth_chars"] = len(GROUND_TRUTH)

# ── 5. The regression canary, BOTH arms ───────────────────────────────────
#
# The receipt does not split, so multi-tile on and off must produce the same
# 45 characters. If they differ, the gate is changing a page it should not be
# touching and every multi-tile number below is suspect.
log("=== canary: 500x650, must not split, must be identical in both arms ===")
canary = {}
for arm, mt in (("off", False), ("on", True)):
    run = run_ocr(Q4, EMBED / CANARY, mt, max_tokens=15)
    txt = text_of(run)
    canary[arm] = {"rc": run["rc"], "seconds": run["seconds"],
                   "text": txt, "chars": len(txt),
                   "matches_expected": txt == CANARY_EXPECT}
    log(f"  canary multi_tile={arm}: rc={run['rc']} {len(txt)} chars "
        f"{'MATCH' if txt == CANARY_EXPECT else 'DIFFERS'} :: {txt!r}")
canary["identical_across_arms"] = canary["off"]["text"] == canary["on"]["text"]
results["stages"]["canary"] = canary
save_results()

# ── 6. Per-stage parity on the SPLITTING fixture ──────────────────────────
if REF_OK:
    log("=== per-stage diff vs the blueprint reference (multi-tile ON) ===")
    run = run_ocr(Q4, EMBED / FIXTURE, True, diff_ref=REF, max_tokens=8)
    diffs = parse_diffs(run["stderr"])
    for name, d in diffs.items():
        log(f"  {name}: {d}")
    results["stages"]["parity_q4k"] = {
        "rc": run["rc"], "seconds": run["seconds"], "diffs": diffs,
        "stderr_tail": run["stderr"][-4000:],
    }
    # The reference resamples with PIL BICUBIC (processor_config resample: 3);
    # our default is align-corners bilinear with no antialiasing. At ~1x
    # scaling the two agree, but a tile grid downscales 1.6x and the thumbnail
    # 4.3x, where bilinear-without-antialias aliases badly. Run the SAME diff
    # with LFM2_VL_BICUBIC=1: if the projector cosines climb, the residual gap
    # is the resample and not the tiling. This is the whole reason that gate
    # exists, and it costs one extra forward pass to answer.
    # F16 as well as Q4_K. The dev guide is explicit: run the runtime at F16 to
    # isolate STRUCTURE from quantization damage, then at the shipped quant.
    # Note the vision tower is F16 in both arms (it comes from the mmproj), so
    # any projector-stage gap is structural by construction and a Q4_K-only
    # reading would have muddied that.
    if F16:
        log("=== per-stage diff at F16 (structure, not quant) ===")
        runf = run_ocr(F16, EMBED / FIXTURE, True, diff_ref=REF, max_tokens=8)
        diffsf = parse_diffs(runf["stderr"])
        for name, dd in diffsf.items():
            log(f"  {name}: {dd}")
        results["stages"]["parity_f16"] = {
            "rc": runf["rc"], "seconds": runf["seconds"], "diffs": diffsf,
            "stderr_tail": runf["stderr"][-6000:],
        }
        save_results()

    log("=== per-stage diff, same reference, LFM2_VL_BICUBIC=0 (old bilinear) ===")
    runb = run_ocr(Q4, EMBED / FIXTURE, True, diff_ref=REF, max_tokens=8, bicubic=False)
    diffsb = parse_diffs(runb["stderr"])
    for name, dd in diffsb.items():
        log(f"  {name}: {dd}")
    results["stages"]["parity_q4k_bicubic"] = {
        "rc": runb["rc"], "seconds": runb["seconds"], "diffs": diffsb,
        "stderr_tail": runb["stderr"][-4000:],
    }
    # State the comparison rather than leaving two tables to eyeball.
    delta = {}
    for name, dd in diffsb.items():
        base = diffs.get(name, {})
        if "cos_min" in dd and "cos_min" in base:
            delta[name] = {"bicubic_default": base["cos_min"], "bilinear_legacy": dd["cos_min"],
                           "delta": round(base["cos_min"] - dd["cos_min"], 6)}
    results["stages"]["resample_cos_delta"] = delta
    log("=== resample effect on per-stage cosine ===")
    log(json.dumps(delta, indent=2))
else:
    log("=== per-stage diff SKIPPED (no reference) ===")
    results["stages"]["parity_q4k"] = {"skipped": "reference dump failed"}
save_results()

# ── 7. The acceptance test: decoded output, both arms, both quants ────────
log("=== decoded-output A/B on the splitting fixture ===")
ab = {}
arms = [("q4_k", Q4)] + ([("f16", F16)] if F16 else [])
for quant, model in arms:
    # bicubic is the default since the last run measured it as the dominant
    # per-stage term; the informative third arm is now the OLD bilinear path.
    for arm, mt, bic in (("off", False, True), ("on", True, True), ("on_bilinear", True, False)):
        key = f"{quant}_multitile_{arm}"
        log(f"  running {key} ...")
        run = run_ocr(model, EMBED / FIXTURE, mt, bicubic=bic)
        txt = text_of(run)
        entry = {"rc": run["rc"], "seconds": run["seconds"], "text": txt,
                 "chars": len(norm(txt))}
        if run["rc"] == 0 and txt and GROUND_TRUTH:
            entry.update(score(txt, GROUND_TRUTH))
        # Proof of work: a crash or an empty transcript must never be scored
        # as a win (dev-guide rule 4a).
        entry["valid"] = bool(run["rc"] == 0 and txt)
        m = re.search(r"(\d+) image tokens", run["stderr"])
        if m:
            entry["image_tokens"] = int(m.group(1))
        m = re.search(r"prompt: (\d+) tokens", run["stderr"])
        if m:
            entry["prompt_tokens"] = int(m.group(1))
        entry["bicubic"] = bic
        ab[key] = entry
        log(f"    {key}: rc={run['rc']} {entry.get('chars', 0)} chars "
            f"CER={entry.get('cer')} WER={entry.get('wer')} "
            f"img_tok={entry.get('image_tokens')} {run['seconds']}s")
        # Read the text (HARD RULE 3). A CER number cannot tell you that the
        # single-tile arm hallucinated fluent Latin while the multi-tile arm
        # transcribed the page -- and that is the whole finding.
        log(f"      text: {txt[:200]!r}")
        results["stages"]["ab"] = ab
        save_results()

# ── 8. Verdict ────────────────────────────────────────────────────────────
verdict = {}
for quant, _ in arms:
    off = ab.get(f"{quant}_multitile_off", {})
    on = ab.get(f"{quant}_multitile_on", {})
    onb = ab.get(f"{quant}_multitile_on_bilinear", {})
    if off.get("valid") and on.get("valid"):
        verdict[quant] = {
            "cer_off": off.get("cer"), "cer_on": on.get("cer"),
            "cer_on_bilinear": onb.get("cer"), "wer_on_bilinear": onb.get("wer"),
            "wer_off": off.get("wer"), "wer_on": on.get("wer"),
            "cer_delta": (round(on["cer"] - off["cer"], 4)
                          if off.get("cer") is not None and on.get("cer") is not None else None),
            "seconds_off": off.get("seconds"), "seconds_on": on.get("seconds"),
            "image_tokens_off": off.get("image_tokens"),
            "image_tokens_on": on.get("image_tokens"),
            "multi_tile_better": (on.get("cer") is not None and off.get("cer") is not None
                                  and on["cer"] < off["cer"]),
        }
    else:
        verdict[quant] = {"error": "one or both arms did not produce a transcript"}

results["verdict"] = verdict
results["status"] = "DONE" if REF_OK else "DONE_NO_PARITY"
save_results()

log("=== verdict ===")
log(json.dumps(verdict, indent=2))
log("NOTE: flipping LFM2_VL_MULTI_TILE on by default requires the canary "
    "identical across arms AND a CER improvement on the splitting fixture, "
    "stated as a number (dev-guide rule 3).")
log(f"results: {RESULTS}")
