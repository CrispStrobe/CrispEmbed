#!/usr/bin/env python3
"""Vision-language OCR parity arm on a Kaggle GPU (brief A2).

Runs `tests/ocr_external_parity.py` with the `--qwen` arm over both parity
corpora and brings back the per-fixture transcripts that become the CER gate
for the native VL lane (`src/qwen2vl_ocr.cpp`).

Why this runs here and not on the dev Mac: the reference checkpoint is 7B at
16-bit, i.e. ~15.5 GiB of weights, against 16 GiB of *total* unified memory on
that machine.  There is no configuration in which the weights and a page of
vision tokens both fit; the run would be measuring swap.  Quality (CER/WER)
transfers across hosts, timing does not — the harness JSON records the hardware
and the numbers from this kernel must never be quoted next to Mac timings.

Placement is handed to accelerate (`--qwen-device-map auto`) because Kaggle
hands out either one 16 GB card or two 15 GB cards at random and only the
sharded arrangement fits on both.  dtype follows the hardware: bf16 needs
compute capability >= 8.0, and the cards here are older than that, so the run
records whichever it actually used rather than claiming bf16.

Corpora:
  * synth  — the 20-image synthetic corpus, shipped as a dataset because it is
             rendered from macOS system fonts and cannot be regenerated here.
  * cc0    — the 5 labelled CC0 fixtures, which live in the repo.

Outputs under /kaggle/working: parity_{synth,cc0}.json, the matching markdown,
gold/<corpus>/*.txt transcripts + manifest.json, and run.log.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
WORK.mkdir(parents=True, exist_ok=True)

# kernels_output does not expose stderr (kaggle_usage gotcha #15) — tee
# everything, including a fatal traceback, into a downloadable file.
_LOG = open(WORK / "run.log", "w", buffering=1)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def _excepthook(exc_type, exc, tb):
    _LOG.write("\n=== FATAL ===\n")
    traceback.print_exception(exc_type, exc, tb, file=_LOG)
    _LOG.flush()
    traceback.print_exception(exc_type, exc, tb, file=sys.__stderr__)


sys.excepthook = _excepthook

REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "feat/parity-qwenvl")
MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
REPO = WORK / "CrispEmbed"
GOLD = WORK / "gold"

# The 15.5 GiB checkpoint must not land in /kaggle/working (20 GB output cap);
# the ephemeral layer under /tmp has ~70 GB.
os.environ["HF_HOME"] = "/tmp/hf"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# A dense page asks for one multi-GiB activation; with the default allocator that
# request has to find a single contiguous block, which fragmentation can deny even
# when the total is available.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def sh(cmd, check=True, cwd=None):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check, cwd=cwd)


def find_dataset(*names) -> Path | None:
    roots = [Path("/kaggle/input")]
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            for cand in root.rglob(name):
                if cand.is_dir():
                    return cand
    return None


print("=" * 70)
print("Step 1: environment")
print("=" * 70)
sh("nvidia-smi || true", check=False)
sh("pip install -q -U 'transformers>=4.51' accelerate 2>&1 | tail -3", check=False)
sh("apt-get -qq update >/dev/null 2>&1 && apt-get -qq install -y tesseract-ocr "
   ">/dev/null 2>&1 || true", check=False)

import torch  # noqa: E402  (after the pip upgrade above)
import transformers  # noqa: E402

n_gpu = torch.cuda.device_count()
caps = [torch.cuda.get_device_capability(i) for i in range(n_gpu)]
names = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
vram = [torch.cuda.get_device_properties(i).total_memory / 2 ** 30 for i in range(n_gpu)]
arch_list = torch.cuda.get_arch_list()
# bf16 arithmetic needs Ampere or newer; on an older card torch will accept the
# dtype and then run it through a slow emulated path, so choose fp16 explicitly
# instead of silently paying for that.
dtype = "bfloat16" if caps and all(c[0] >= 8 for c in caps) else "float16"
hardware = f"Kaggle {n_gpu}x {names[0] if names else 'CPU'} ({sum(vram):.0f} GiB VRAM total)"
print(f"torch={torch.__version__} transformers={transformers.__version__}")
print(f"gpus={names} caps={caps} vram={[round(v, 1) for v in vram]}")
print(f"arch_list={arch_list}")
print(f"dtype={dtype}  hardware={hardware}")

# `torch.cuda.is_available()` is True on a card the installed wheel has no SASS
# for; the failure only surfaces at the first launch, as
# `cudaErrorNoKernelImageForDevice`, ~90 s into a run.  Check the card against
# the wheel's compiled arch list up front so a bad draw is one line, not a
# wasted GPU-hour.  (Measured: a P100 draw died exactly this way — cu128 wheels
# ship no sm_60.)
missing = [f"sm_{c[0]}{c[1]}" for c in caps if f"sm_{c[0]}{c[1]}" not in arch_list]
if missing:
    raise SystemExit(
        f"torch {torch.__version__} has no kernels for {missing} "
        f"(compiled for {arch_list}); re-push pinning a supported accelerator "
        f"via kernel-metadata machine_shape")
# 7B at 16-bit is 15.45 GiB of weights.  A single 16 GiB card leaves nothing for
# a page's vision tokens, so accelerate silently offloads layers to CPU and every
# generated token then streams them back over PCIe.  That still produces correct
# transcripts, but the timing means something completely different — refuse it
# rather than publish a number nobody can interpret.
WEIGHTS_GIB = 15.45
if sum(vram) < WEIGHTS_GIB + 4:
    raise SystemExit(
        f"{sum(vram):.1f} GiB VRAM cannot hold {WEIGHTS_GIB} GiB of weights plus "
        f"activations without CPU offload; re-push pinning a larger accelerator")

print("=" * 70)
print("Step 2: repo + fixtures")
print("=" * 70)
if not REPO.exists():
    sh(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {REPO}")
sh("git log --oneline -1", cwd=REPO)

synth = find_dataset("crispembed-ocr-synth")
cc0 = REPO / "tests" / "regression" / "images" / "cc0"
print(f"synth fixtures: {synth}")
print(f"cc0 fixtures:   {cc0}")
if synth is None:
    raise SystemExit("crispembed-ocr-synth dataset not mounted")

# The dataset mount is read-only and the harness only writes next to its
# outputs, but copy anyway so nothing depends on that staying true.
synth_local = WORK / "synth"
if not synth_local.exists():
    shutil.copytree(synth, synth_local)

token_dir = find_dataset("crispasr-hf-token")
if token_dir is not None:
    tok = token_dir / "hf_token.txt"
    if tok.exists():
        os.environ["HF_TOKEN"] = tok.read_text().strip()
        print("HF_TOKEN loaded from dataset")

print("=" * 70)
print("Step 3: parity runs (sequential — one heavy process at a time)")
print("=" * 70)
# accelerate fills device 0 with weights before moving to device 1, and the
# inputs land on device 0 too, so the busiest device ends up with the least free
# memory.  Measured on 2x T4: a full page asked for 3.98 GiB against 2.26 GiB
# free on device 0 while device 1 sat half empty.  Cap the *weight* budget per
# device so the activation headroom exists where it is needed; the synthetic
# corpus never noticed, only the dense CC0 scans did.
max_memory = ""
if n_gpu > 1:
    head = 5.0  # GiB of activation headroom to keep free on the input device
    max_memory = ",".join(
        f"{i}={int(max(vram[i] - (head if i == 0 else 1.0), 4.0))}GiB"
        for i in range(n_gpu))
    print(f"max_memory={max_memory}")

results = {}
for corpus, images in (("synth", synth_local), ("cc0", cc0)):
    out_json = WORK / f"parity_{corpus}.json"
    out_md = WORK / f"parity_{corpus}.md"
    gold = GOLD / corpus
    cmd = [
        sys.executable, str(REPO / "tests" / "ocr_external_parity.py"),
        "--images", str(images), "--require-truth", "--repeats", "1",
        "--qwen", "--qwen-model", MODEL,
        "--qwen-dtype", dtype, "--qwen-device-map", "auto",
        *(["--qwen-max-memory", max_memory] if max_memory else []),
        "--qwen-transcripts", str(gold),
        "--hardware", hardware,
        "--skip", "docling-py", "--skip", "easyocr-py", "--skip", "paddleocr-py",
        "--output", str(out_json), "--markdown", str(out_md),
    ]
    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    print(p.stdout[-20000:], flush=True)
    if p.returncode != 0:
        print(f"!! {corpus} exit={p.returncode}\n{p.stderr[-8000:]}", flush=True)
    print(f"[{corpus}] wall={time.time() - t0:.1f}s exit={p.returncode}", flush=True)
    if out_json.exists():
        results[corpus] = json.loads(out_json.read_text())["summary"]

print("=" * 70)
print("Summary")
print("=" * 70)
print(json.dumps(results, indent=2))
(WORK / "summary.json").write_text(json.dumps(
    {"hardware": hardware, "dtype": dtype, "model": MODEL, "branch": BRANCH,
     "torch": torch.__version__, "transformers": transformers.__version__,
     "summary": results}, indent=2) + "\n")

# The clone and the HF cache are not outputs; leaving them would blow the
# 20 GB /kaggle/working cap.
shutil.rmtree(REPO, ignore_errors=True)
shutil.rmtree(WORK / "synth", ignore_errors=True)
print("done")
