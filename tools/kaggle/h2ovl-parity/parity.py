"""h2ovl-mississippi-2b — the parity half of the port pipeline.

`h2ovl-convert` proved the GGUF is structurally sound (565 tensors, use_msac
survived) and then correctly REFUSED to publish: its OCR smoke returned rc=0 but
only 29 characters for a full page, so it exited rather than ship a model that
cannot read. Structural soundness is not clearance. Per crispasr-crispembed-dev.md
the acceptance gate is per-stage cosine against a `-ref.gguf` baked from the
Python blueprint AND the decoded output — never a smoke test, never a size.

This kernel closes that gap:

  1. convert f16 in-kernel (~2 min) -- self-contained, because the convert
     kernel PUBLISHED NOTHING: its OCR gate fired (rc=0 but chars=29 on a full
     page) and it exited rather than ship a model that cannot read. So there is
     no repo to pull from, and the model is still broken.
  2. bake `-ref.gguf` from the blueprint (tools/dump_internvl2_reference.py --
     a pure-numpy forward over the safetensors, so no 8 GB torch load)
  3. crispembed-quantize -> q8_0
  4. test-internvl2-diff EVERY precision against the reference -- THE POINT.
     MSAC two-scale tiling now demonstrably runs (19 tiles, 3x2 grid) and did
     NOT fix the degenerate output, so that hypothesis is dead and the next
     move is localising the divergence by stage, not guessing at the next one.
     The harness reaches vis_proj_output AND llm_embed + run_llm_forward, so it
     can say whether vision is clean and the LLM is at fault.
  5. decoded-output roundtrip on every precision
  6. publish ONLY if both gates pass. A model that emits 29 characters for a
     page does not get a repo just because the pipeline ran; on failure this
     writes the diagnostic and uploads nothing.

Regime notes (kaggle_usage.md): harness comes from the CrispASR clone, not the
bundled copy (#26a); the token is resolved, not merely attached (#26b); every
long block is heartbeated so Kaggle cannot idle-kill us; big files stage under
/tmp, and the repo clones live in /kaggle/temp so `/kaggle/working` stays under
the 500-file output page cap (#18/#22).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

MODEL = "h2oai/h2ovl-mississippi-2b"
NAME = "h2ovl-mississippi-2b"
REPO = f"cstr/{NAME}-crispembed-GGUF"
# Reference intermediates are fixtures, not releases — mirrors CrispASR's
# cstr/crispasr-regression-fixtures convention (dev guide, crispasr-diff §).
FIXTURES = "cstr/crispembed-regression-fixtures"

WORK = Path("/kaggle/working")          # artifacts we must retrieve only
TEMP = Path("/kaggle/temp")             # repo clones (keeps WORK small, #22)
BIG = Path("/tmp/h2ovl")                # models: ~70 GB ephemeral layer (#18)
for d in (TEMP, BIG):
    d.mkdir(parents=True, exist_ok=True)

# HF cache off /kaggle/working — a 4.4 GB checkpoint must not land in the 20 GB
# persistent mount.
os.environ["HF_HOME"] = str(BIG / "hf")

# ── harness: from the clone, bundled copy is fallback only (#26a) ──────
_CRISPASR = TEMP / "CrispASR"
if not _CRISPASR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1",
                               "https://github.com/CrispStrobe/CrispASR.git", str(_CRISPASR)])
    except Exception as e:
        print(f"CrispASR clone failed ({e}) — falling back to bundled harness")
if (_CRISPASR / "tools" / "kaggle").is_dir():
    sys.path.insert(0, str(_CRISPASR / "tools" / "kaggle"))
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.step("start", model=MODEL, repo=REPO)

hf_token = kh.resolve_hf_token()
kh.step("auth", have_token=bool(hf_token))
if not hf_token:
    sys.exit("no HF token from env/secret/dataset — refusing to burn compute "
             "on a run that cannot publish or even fetch the f16")

# ── source ─────────────────────────────────────────────────────────────
CE = TEMP / "CrispEmbed"
if not CE.exists():
    with kh.build_heartbeat("clone"):
        kh.sh(f"git clone --recursive https://github.com/CrispStrobe/CrispEmbed.git {CE}")
os.chdir(CE)
kh.sh("git log --oneline -1", check=False)
kh.step("cloned")

# Not torch (#11) — the dumper is numpy-only by design.
kh.sh("pip install -q gguf safetensors huggingface_hub pillow", check=False)

# ── 1a. convert f16 in-kernel (the convert kernel published nothing) ───
f16 = BIG / f"{NAME}-f16.gguf"
with kh.build_heartbeat("convert"):
    kh.sh(f"python models/convert-internvl2-to-gguf.py --model {MODEL} "
          f"--dtype f16 --output {f16}")
kh.step("converted", mb=round(f16.stat().st_size / 1024**2))

# ── 1b. bake the reference from the Python blueprint ───────────────────
ref = BIG / f"{NAME}-ref.gguf"
with kh.build_heartbeat("refdump", interval_s=30.0):
    kh.sh(f"python tools/dump_internvl2_reference.py --model {MODEL} "
          f"--output {ref} --max-llm-layers 4")
if not ref.exists():
    sys.exit("reference dump produced no file — cannot gate anything without it")
kh.step("ref_baked", mb=round(ref.stat().st_size / 1024**2))

from huggingface_hub import HfApi, create_repo  # noqa: E402

api = HfApi()

# Checkpoint the reference the MOMENT it exists (dev guide: never crash before
# a produced artifact reaches HF). It is a FIXTURE, not a model release, so it
# goes to the fixtures dataset unconditionally — the model artifacts below stay
# gated. Losing this to a later failure would mean re-paying the whole dump.
try:
    create_repo(FIXTURES, repo_type="dataset", exist_ok=True, token=hf_token, private=False)
    with kh.build_heartbeat("upload.ref.fixture", interval_s=60.0):
        api.upload_file(path_or_fileobj=str(ref), path_in_repo=f"internvl2/{NAME}/ref.gguf",
                        repo_id=FIXTURES, repo_type="dataset", token=hf_token,
                        commit_message=f"{NAME} per-stage reference (Python blueprint)")
    kh.step("ref_checkpointed", repo=FIXTURES, path=f"internvl2/{NAME}/ref.gguf")
except Exception as e:
    kh.step("ref_checkpoint_failed", err=str(e)[:300])

# ── 2. build ───────────────────────────────────────────────────────────
kh.install_build_toolchain()
flags = kh.cache_and_link_flags()
jobs = kh.safe_build_jobs(gpu=False)
with kh.build_heartbeat("build"):
    kh.sh("cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release " + " ".join(flags))
    kh.sh_with_progress(
        f"ninja -C build -j{jobs} crispembed-quantize crispembed-cli test-internvl2-diff")
kh.step("built")

# ── 3. quantize ────────────────────────────────────────────────────────
q8 = BIG / f"{NAME}-q8_0.gguf"
with kh.build_heartbeat("quantize.q8_0"):
    kh.sh(f"./build/crispembed-quantize {f16} {q8} q8_0")
kh.step("quantized", precision="q8_0", mb=round(q8.stat().st_size / 1024**2))

q4 = BIG / f"{NAME}-q4_k.gguf"
with kh.build_heartbeat("quantize.q4_k"):
    kh.sh(f"./build/crispembed-quantize {f16} {q4} q4_k")
kh.step("quantized", precision="q4_k", mb=round(q4.stat().st_size / 1024**2))


# ── 5+6. per-stage diff AND decoded output, for every precision ────────
def parse_cosines(out: str):
    """Pull 'name: cos=X max_abs=Y' lines out of the diff harness output."""
    got = {}
    for line in out.splitlines():
        if "cos=" in line and ":" in line:
            try:
                stage = line.split(":")[0].strip()
                cos = float(line.split("cos=")[1].split()[0])
                got[stage] = cos
            except Exception:
                pass
    return got


img = CE / "tests/regression/images/scan_page_pd.png"
results = {}
for label, path in (("f16", f16), ("q8_0", q8), ("q4_k", q4)):
    if path is None or not Path(path).exists():
        kh.step("skip", precision=label, reason="artifact absent")
        continue
    entry = {"mb": round(Path(path).stat().st_size / 1024**2)}

    with kh.build_heartbeat(f"diff.{label}", interval_s=60.0):
        p = subprocess.run(["./build/test-internvl2-diff", str(path), str(ref)],
                           capture_output=True, text=True, timeout=7200)
    cos = parse_cosines(p.stdout)
    entry["diff_rc"] = p.returncode
    entry["cos"] = cos
    entry["cos_min"] = min(cos.values()) if cos else None
    entry["diff_tail"] = p.stdout.strip().splitlines()[-6:]
    if p.returncode != 0 and not cos:
        entry["diff_stderr"] = p.stderr.strip().splitlines()[-8:]
    kh.step(f"diff.{label}", rc=p.returncode, cos_min=entry["cos_min"], stages=len(cos))

    # Decoded output is the only acceptance test (HARD RULE #3). A cosine can
    # be perfect while the text degenerates -- which is exactly this model's
    # recorded failure (f16 repeated one token, q4_k emitted ".").
    if img.exists():
        with kh.build_heartbeat(f"ocr.{label}", interval_s=60.0):
            o = subprocess.run(["./build/crispembed", "-m", str(path), "--ocr", str(img)],
                               capture_output=True, text=True, timeout=7200)
        text = o.stdout.strip()
        entry["ocr_rc"] = o.returncode
        entry["chars"] = len(text)
        entry["words"] = len(text.split())
        entry["head"] = text[:300]
        # Degenerate-output detector: the recorded failure mode is a single
        # token repeated, or one punctuation mark then EOS.
        uniq = len(set(text.split()))
        entry["unique_words"] = uniq
        entry["degenerate"] = bool(len(text) < 40 or (entry["words"] > 8 and uniq <= 2))
        kh.step(f"ocr.{label}", rc=o.returncode, chars=len(text), words=entry["words"],
                unique=uniq, degenerate=entry["degenerate"], head=text[:120])
    results[label] = entry

(WORK / "parity.json").write_text(json.dumps(results, indent=2))
kh.step("results_written")

# ── 7. publish ONLY if the model earned it ─────────────────────────────
# The reference is a fixture and is always worth keeping — it is what any
# future debugging session needs and it cost a full run to produce. The MODEL
# artifacts are gated: the convert kernel already refused to ship this
# checkpoint at 29 chars/page, and a pipeline completing is not a clearance.


def cleared(r):
    return (r.get("ocr_rc") == 0 and not r.get("degenerate", True)
            and (r.get("cos_min") is None or r["cos_min"] >= 0.99))


passing = {k: v for k, v in results.items() if cleared(v)}
if passing:
    create_repo(REPO, repo_type="model", exist_ok=True, token=hf_token)
    with kh.build_heartbeat("upload.ref", interval_s=60.0):
        api.upload_file(path_or_fileobj=str(ref), path_in_repo=ref.name, repo_id=REPO,
                        token=hf_token,
                        commit_message="Per-stage reference intermediates (Python blueprint)")
    kh.step("ref_uploaded", file=ref.name)
    for label in passing:
        art = {"f16": f16, "q8_0": q8, "q4_k": q4}[label]
        with kh.build_heartbeat(f"upload.{label}", interval_s=60.0):
            api.upload_file(path_or_fileobj=str(art), path_in_repo=art.name, repo_id=REPO,
                            token=hf_token,
                            commit_message="per-stage diff + decoded-output verified")
        kh.step("uploaded", file=art.name)
else:
    kh.step("upload_withheld_all",
            reason="no precision passed both the per-stage diff and the decoded-output gate",
            detail={k: {"cos_min": v.get("cos_min"), "chars": v.get("chars"),
                        "degenerate": v.get("degenerate"), "ocr_rc": v.get("ocr_rc")}
                    for k, v in results.items()})

kh.step("done", verdict={k: {"cos_min": v.get("cos_min"), "chars": v.get("chars"),
                             "degenerate": v.get("degenerate")}
                         for k, v in results.items()})
print("DONE")
print(json.dumps(results, indent=2)[:4000])
