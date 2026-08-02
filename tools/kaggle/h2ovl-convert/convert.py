#!/usr/bin/env python3
"""Kaggle kernel: convert H2OVL-Mississippi-2B to CrispEmbed GGUF + upload.

Runs off-box because the local dev machine cannot hold this one: 4.3 GB of
upstream safetensors, a 4.4 GB f16 GGUF and a 1.4 GB q4_k on top — more than
the free space there, and a local attempt ran the disk out mid-write.

What makes this model special is `use_msac` (Multi-Scale Adaptive Cropping).
The converter records the flag and the runtime honours it by tiling the page
at two scales; without that the model loads fine and answers with fluent
nonsense. Its 800m sibling has use_msac=false, which is why that one worked
before MSAC existed. Both facts are asserted below rather than assumed —
every failure mode this model has is silent.

Follows the kaggle_usage.md contract:
  - kaggle_harness for auth + toolchain, cloned from CrispASR with the
    bundled copy beside this file as the fallback (a CPU worker may have no
    internet, and then the clone is what fails first)
  - init_progress() before anything else: Kaggle buffers parent stdout
    heavily, so without it a hang is invisible until the kernel is killed
  - build_heartbeat() around every long silent block, so a stall is
    distinguishable from slow progress
  - both datasets attached: crispasr-hf-token (upload auth) and
    crispasr-ccache (warm build, ~20 min -> ~3 min)

Attach datasets: chr1str/crispasr-hf-token, chr1str/crispasr-ccache
"""
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
MODEL = "h2oai/h2ovl-mississippi-2b"
NAME = "h2ovl-mississippi-2b"
REPO = "cstr/h2ovl-mississippi-2b-crispembed-GGUF"

# ── harness ────────────────────────────────────────────────────────────
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
_CRISPASR_DIR = WORK / "CrispASR"
if not _CRISPASR_DIR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", CRISPASR_URL, str(_CRISPASR_DIR)])
        sys.path.insert(0, str(_CRISPASR_DIR / "tools" / "kaggle"))
    except Exception as exc:  # noqa: BLE001 — fall through to bundled copy
        print(f"CrispASR clone failed ({exc}); using bundled harness", flush=True)

if str(_CRISPASR_DIR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.step("start", model=MODEL, repo=REPO)

hf_token = kh.resolve_hf_token()
kh.step("auth", have_token=bool(hf_token))
if not hf_token:
    sys.exit("no HF token from env/secret/dataset — cannot upload, refusing to "
             "burn an hour of compute for nothing")

# ── source ─────────────────────────────────────────────────────────────
os.chdir(WORK)
if not (WORK / "CrispEmbed").exists():
    with kh.build_heartbeat("clone"):
        kh.sh("git clone --recursive https://github.com/CrispStrobe/CrispEmbed.git")
os.chdir(WORK / "CrispEmbed")
kh.sh("git log --oneline -1", check=False)
kh.step("cloned")

# Not torch — Kaggle pre-installs it (gotcha #11); only the small deps.
kh.sh("pip install -q gguf safetensors transformers", check=False)

# ── convert ────────────────────────────────────────────────────────────
f16 = WORK / f"{NAME}-f16.gguf"
with kh.build_heartbeat("convert"):
    kh.sh(f"python models/convert-internvl2-to-gguf.py --model {MODEL} "
          f"--dtype f16 --output {f16}")
kh.step("converted", mb=round(f16.stat().st_size / 1024**2))

# The flag must have survived into the GGUF, or the runtime single-scale-tiles
# a model trained on two scales — which does not error, it just lies.
from gguf import GGUFReader  # noqa: E402

reader = GGUFReader(str(f16))
msac = None
for key, field in reader.fields.items():
    if key.endswith("use_msac"):
        msac = bool(field.parts[field.data[0]][0])

# And the LLM matrices must exist: the old model_type-based export branch
# emitted only per-layer norms for this checkpoint, giving a GGUF that loads
# and then segfaults in ggml_mul_mat on a null tensor.
names = {t.name for t in reader.tensors}
missing = [n for n in ("l.blk.0.attn_q.weight", "l.blk.0.ffn_gate.weight") if n not in names]
kh.step("gguf_checked", use_msac=msac, tensors=len(names), missing=missing)
del reader
if msac is not True:
    sys.exit(f"internvl2.use_msac is {msac!r} — the runtime would mis-tile this model")
if missing:
    sys.exit(f"LLM weights missing from GGUF: {missing}")

# ── build + quantize ───────────────────────────────────────────────────
kh.install_build_toolchain()  # warms /kaggle/working/.ccache from the dataset
flags = kh.cache_and_link_flags()
jobs = kh.safe_build_jobs(gpu=False)
with kh.build_heartbeat("build"):
    kh.sh("cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release " + " ".join(flags))
    kh.sh_with_progress(f"ninja -C build -j{jobs} crispembed-quantize crispembed-cli test-msac-tiling")
kh.step("built")

# Geometry check before the model check: if the tiler is wrong, a bad OCR
# result would be blamed on the weights.
kh.sh("./build/test-msac-tiling")
kh.step("msac_geometry_ok")

q4 = WORK / f"{NAME}-q4_k.gguf"
with kh.build_heartbeat("quantize"):
    kh.sh(f"./build/crispembed-quantize {f16} {q4} q4_k")
kh.step("quantized", mb=round(q4.stat().st_size / 1024**2))

# ── smoke test ─────────────────────────────────────────────────────────
img = Path("tests/regression/images/scan_page_pd.png")
if img.exists():
    with kh.build_heartbeat("ocr", interval_s=60.0):
        proc = subprocess.run(["./build/crispembed", "-m", str(q4), "--ocr", str(img)],
                              capture_output=True, text=True, timeout=7200)
    tiles = [ln for ln in proc.stderr.splitlines() if "MSAC" in ln or "tiles (" in ln]
    text = proc.stdout.strip()
    kh.step("ocr", rc=proc.returncode, chars=len(text), tiling=tiles)
    print("\n".join(tiles), flush=True)
    print(text[:800], flush=True)
    if proc.returncode != 0 or len(text) < 200:
        sys.exit("OCR smoke test failed — refusing to upload a model that cannot read a page")
    if not any("MSAC" in t for t in tiles):
        sys.exit("MSAC tiling did not run — the GGUF flag is not reaching the runtime")

# ── upload ─────────────────────────────────────────────────────────────
from huggingface_hub import HfApi, create_repo  # noqa: E402

api = HfApi(token=hf_token)
create_repo(REPO, repo_type="model", exist_ok=True, token=hf_token)
card = f"""---
license: apache-2.0
base_model: {MODEL}
tags: [gguf, ocr, crispembed, internvl, h2ovl]
---

# H2OVL-Mississippi-2B — CrispEmbed GGUF

H2OVL-Mississippi-2B (InternViT-300M + H2O-Danube2-1.8B, OCRBench 782) in the
single-file **CrispEmbed** GGUF layout, for the `internvl2_ocr` engine.

**Requires MSAC.** This model sets `use_msac`, so the page is tiled at two
scales and concatenated `fine[:-1] + coarse[:-1] + fine[-1:]`. CrispEmbed
implements this. A runtime that single-scale-tiles it does not error — it
returns confident nonsense, so check your engine supports MSAC before trusting
any output. Not interchangeable with llama.cpp GGUFs.

## Usage

```bash
crispembed -m {NAME} --ocr document.png
```

## Attribution & licence

Upstream © H2O.ai, Apache-2.0 — see [{MODEL}](https://huggingface.co/{MODEL});
vision tower InternViT-300M is MIT. Conversion and quantization do not
relicense it. See [CrispEmbed](https://github.com/CrispStrobe/CrispEmbed) and
its `POLICY.md`: OCR output is a probabilistic reconstruction, not a faithful
copy, and VLM engines confabulate through a smudge rather than leave it blank.
"""
(WORK / "README.md").write_text(card)
api.upload_file(path_or_fileobj=str(WORK / "README.md"), path_in_repo="README.md",
                repo_id=REPO, token=hf_token, commit_message="Model card")
for f in (q4, f16):
    if f.exists():
        with kh.build_heartbeat(f"upload.{f.name}", interval_s=60.0):
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=REPO,
                            token=hf_token,
                            commit_message="CrispEmbed-format GGUF (MSAC two-scale tiling required)")
        kh.step("uploaded", file=f.name)

kh.step("done")
print("DONE")
