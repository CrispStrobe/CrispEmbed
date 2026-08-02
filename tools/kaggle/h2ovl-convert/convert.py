#!/usr/bin/env python3
"""Kaggle kernel: convert H2OVL-Mississippi-2B to CrispEmbed GGUF + upload.

Runs off-box because the local dev machine cannot hold this one: the upstream
safetensors are 4.3 GB, the f16 GGUF another 4.4 GB, and the q4_k 1.4 GB on top
— more than the free space there, and the download alone saturates it.

What makes this model special is `use_msac` (Multi-Scale Adaptive Cropping).
The converter records the flag and the runtime honours it by tiling the page
twice; without that the model loads fine and answers with fluent nonsense. The
800m sibling has use_msac=false, which is why it worked before MSAC existed.

Attach datasets: chr1s4/crispasr-hf-token
"""
import gc
import os
import subprocess
import sys

MODEL = "h2oai/h2ovl-mississippi-2b"
NAME = "h2ovl-mississippi-2b"
REPO = "cstr/h2ovl-mississippi-2b-crispembed-GGUF"
WORK = "/kaggle/working"

os.chdir(WORK)
subprocess.run("git clone --recursive https://github.com/CrispStrobe/CrispEmbed.git", shell=True, check=True)
os.chdir("CrispEmbed")
subprocess.run("git log --oneline -1", shell=True, check=True)

subprocess.run([sys.executable, "-m", "pip", "install", "gguf", "safetensors", "transformers", "-q"], check=True)

hf_token = None
for p in ["/kaggle/input/crispasr-hf-token/hf_token.txt",
          "/kaggle/input/datasets/chr1s4/crispasr-hf-token/hf_token.txt"]:
    if os.path.exists(p):
        hf_token = open(p).read().strip()
        break
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    print(f"HF token loaded ({len(hf_token)} chars)", flush=True)
else:
    print("WARNING: no HF token — will convert but not upload", flush=True)

f16 = f"{WORK}/{NAME}-f16.gguf"
print(f"Converting {MODEL}...", flush=True)
subprocess.run([sys.executable, "models/convert-internvl2-to-gguf.py",
                "--model", MODEL, "--dtype", "f16", "--output", f16], check=True)
print(f"F16: {os.path.getsize(f16) / 1024**2:.0f} MB", flush=True)

# The whole point of this conversion: the flag must have survived into the
# GGUF, or the runtime will single-scale-tile a model that cannot read that.
from gguf import GGUFReader  # noqa: E402

r = GGUFReader(f16)
msac = None
for k, field in r.fields.items():
    if k.endswith("use_msac"):
        msac = bool(field.parts[field.data[0]][0])
print(f"internvl2.use_msac in GGUF: {msac}", flush=True)
assert msac is True, "use_msac missing or false — the runtime would mis-tile this model"

# Sanity: the LLM weight matrices must be present. A model_type-based branch
# used to silently export only the per-layer norms, producing a GGUF that
# loads and then segfaults in ggml_mul_mat on a null tensor.
names = {t.name for t in r.tensors}
missing = [n for n in ("l.blk.0.attn_q.weight", "l.blk.0.ffn_gate.weight") if n not in names]
assert not missing, f"LLM weights missing from GGUF: {missing}"
print(f"tensors: {len(names)} (LLM attention + FFN present)", flush=True)
del r
gc.collect()

subprocess.run("cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release", shell=True, check=True)
subprocess.run("ninja -C build crispembed-quantize", shell=True, check=True)

q4 = f"{WORK}/{NAME}-q4_k.gguf"
subprocess.run(["build/crispembed-quantize", f16, q4, "q4_k"], check=True)
print(f"q4_k: {os.path.getsize(q4) / 1024**2:.0f} MB", flush=True)

# Smoke-test the real path on Kaggle, so a broken artifact is never uploaded.
# The MSAC line in the log is the thing to look for: it proves the two-scale
# tiler ran rather than the single-scale one.
subprocess.run("ninja -C build crispembed-cli", shell=True, check=True)
img = "tests/regression/images/scan_page_pd.png"
if os.path.exists(img):
    print("=== OCR smoke test (q4_k) ===", flush=True)
    p = subprocess.run(["build/crispembed", "-m", q4, "--ocr", img],
                       capture_output=True, text=True, timeout=3600)
    tiles = [ln for ln in p.stderr.splitlines() if "MSAC" in ln or "tiles (" in ln]
    print("\n".join(tiles), flush=True)
    text = p.stdout.strip()
    print(f"rc={p.returncode} chars={len(text)}", flush=True)
    print(text[:600], flush=True)
    if p.returncode != 0 or len(text) < 200:
        sys.exit("OCR smoke test failed — refusing to upload a model that cannot read a page")
    if not any("MSAC" in t for t in tiles):
        sys.exit("MSAC tiling did not run — the GGUF flag is not reaching the runtime")

if hf_token:
    from huggingface_hub import HfApi, create_repo

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
scales and the tiles concatenated `fine[:-1] + coarse[:-1] + fine[-1:]`.
CrispEmbed implements this; a runtime that single-scale-tiles it will get
fluent nonsense rather than a transcription. Needs CrispEmbed with MSAC
support (`internvl2.use_msac` honoured in `image_preprocess`).

Not interchangeable with llama.cpp GGUFs — different tensor naming, one file.

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
    open(f"{WORK}/README.md", "w").write(card)
    api.upload_file(path_or_fileobj=f"{WORK}/README.md", path_in_repo="README.md",
                    repo_id=REPO, token=hf_token, commit_message="Model card")
    for f in (f"{NAME}-q4_k.gguf", f"{NAME}-f16.gguf"):
        path = f"{WORK}/{f}"
        if os.path.exists(path):
            print(f"Uploading {f}...", flush=True)
            api.upload_file(path_or_fileobj=path, path_in_repo=f, repo_id=REPO, token=hf_token,
                            commit_message="CrispEmbed-format GGUF (MSAC two-scale tiling required)")
    print("Upload complete", flush=True)

print("DONE")
