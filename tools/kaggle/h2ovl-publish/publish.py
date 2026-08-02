"""h2ovl-mississippi-2b — publish the DIAGNOSTIC artifacts for local debugging.

Deliberately separate from `h2ovl-parity`, and deliberately NOT a release.

The model is known broken: `h2ovl-2b-convert` measured 29 characters for a full
page and refused to publish, and `h2ovl-2b-parity` then showed 27 stages at
cos_min 0.999972 against the Python blueprint — so the ported compute is right
and the fault is downstream of the logits. The next step is a local
edit/rebuild/re-run loop on the Mac, which needs the q4_k (1.4 GB, fits) and the
reference on HF.

So this uploads artifacts whose ONLY purpose is that debugging loop:

  * repo is PRIVATE — a model that cannot read a page must not be published
  * the card says UNVALIDATED in the first line, with the measured failure
  * no registry entry, no SHA pin: `crispembed --list-models` must not offer it

Upload order is smallest-first (q4_k, then q8_0, then f16) so the artifact
actually needed locally lands first — the dev guide's "checkpoint each artifact
the moment it exists" applied to a run that could still be cut short.
"""
import os
import subprocess
import sys
from pathlib import Path

MODEL = "h2oai/h2ovl-mississippi-2b"
NAME = "h2ovl-mississippi-2b"
REPO = f"cstr/{NAME}-crispembed-GGUF"

TEMP = Path("/kaggle/temp")
BIG = Path("/tmp/h2ovl")
for d in (TEMP, BIG):
    d.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(BIG / "hf")

_CRISPASR = TEMP / "CrispASR"
if not _CRISPASR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
                               "https://github.com/CrispStrobe/CrispASR.git", str(_CRISPASR)])
    except Exception as e:
        print(f"CrispASR clone failed ({e}) — bundled harness fallback")
if (_CRISPASR / "tools" / "kaggle").is_dir():
    sys.path.insert(0, str(_CRISPASR / "tools" / "kaggle"))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.step("start", model=MODEL, repo=REPO, purpose="diagnostic artifacts, private, unvalidated")

hf_token = kh.resolve_hf_token()
kh.step("auth", have_token=bool(hf_token))
if not hf_token:
    sys.exit("no HF token — nothing to publish to")

CE = TEMP / "CrispEmbed"
if not CE.exists():
    with kh.build_heartbeat("clone"):
        kh.sh(f"git clone --recursive https://github.com/CrispStrobe/CrispEmbed.git {CE}")
os.chdir(CE)
kh.sh("git log --oneline -1", check=False)
kh.step("cloned")
kh.sh("pip install -q gguf safetensors huggingface_hub", check=False)

f16 = BIG / f"{NAME}-f16.gguf"
with kh.build_heartbeat("convert"):
    kh.sh(f"python models/convert-internvl2-to-gguf.py --model {MODEL} "
          f"--dtype f16 --output {f16}")
kh.step("converted", mb=round(f16.stat().st_size / 1024**2))

kh.install_build_toolchain()
flags = kh.cache_and_link_flags()
jobs = kh.safe_build_jobs(gpu=False)
with kh.build_heartbeat("build"):
    kh.sh("cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release " + " ".join(flags))
    kh.sh_with_progress(f"ninja -C build -j{jobs} crispembed-quantize")
kh.step("built")

from huggingface_hub import HfApi, create_repo  # noqa: E402

api = HfApi()
create_repo(REPO, repo_type="model", exist_ok=True, token=hf_token, private=True)
kh.step("repo_created", private=True)

card = f"""---
license: apache-2.0
base_model: {MODEL}
tags: [gguf, crispembed, unvalidated, diagnostic]
---

# {NAME} — CrispEmbed GGUF (**UNVALIDATED — DO NOT USE**)

**This model does not work.** It is published private, for debugging only, and
must not be added to the CrispEmbed model registry.

Measured on `tests/regression/images/scan_page_pd.png`:

| check | result |
|---|---|
| decoded output | **29 characters for a full page** — fails |
| per-stage cosine vs Python blueprint | 27 stages, cos_min `0.999972` — passes |

The ported compute is correct; the fault is downstream of the logits. Root cause
identified as the chat template: this checkpoint declares `template: h2ogpt2`
(`<|prompt|>...<|end|><|answer|>`, eos `<|end|>` = 32009, and
`generation_config.eos_token_id = [2, 32009]`), whereas `src/internvl2_ocr.cpp`
builds InternVL2 ChatML unconditionally. `<|im_start|>`/`<|im_end|>` are absent
from this vocab, so those markers are silently dropped and the model receives an
unmarked prompt it was never trained on.

Upstream (c) H2O.ai, Apache-2.0 — see [{MODEL}](https://huggingface.co/{MODEL}).
Conversion and quantization do not relicense it.
"""
(BIG / "README.md").write_text(card)
api.upload_file(path_or_fileobj=str(BIG / "README.md"), path_in_repo="README.md",
                repo_id=REPO, token=hf_token, commit_message="Card: UNVALIDATED, diagnostic only")
kh.step("card_uploaded")

# Smallest first: q4_k is the one needed on a 16 GB Mac with ~6 GB free.
for prec in ("q4_k", "q8_0"):
    out = BIG / f"{NAME}-{prec}.gguf"
    with kh.build_heartbeat(f"quantize.{prec}"):
        kh.sh(f"./build/crispembed-quantize {f16} {out} {prec}")
    kh.step("quantized", precision=prec, mb=round(out.stat().st_size / 1024**2))
    with kh.build_heartbeat(f"upload.{prec}", interval_s=60.0):
        api.upload_file(path_or_fileobj=str(out), path_in_repo=out.name, repo_id=REPO,
                        token=hf_token, commit_message=f"{prec} (UNVALIDATED — diagnostic)")
    kh.step("uploaded", file=out.name)
    out.unlink(missing_ok=True)          # keep /tmp clear

with kh.build_heartbeat("upload.f16", interval_s=60.0):
    api.upload_file(path_or_fileobj=str(f16), path_in_repo=f16.name, repo_id=REPO,
                    token=hf_token, commit_message="f16 (UNVALIDATED — diagnostic)")
kh.step("uploaded", file=f16.name)

kh.step("done")
print("DONE")
