#!/usr/bin/env python3
"""T19/F7b imatrix re-run — post-F7 QKV-coverage fix (Kaggle, chr1s4).

A thin driver, deliberately: ALL the logic lives in
`tools/kaggle/crispembed-imatrix-quant/imatrix_quant.py` and is executed FROM THE
CLONE, so there is exactly one copy of the pipeline and the calibration corpora
travel with it (a Kaggle *script* kernel ships only its `code_file` — bundled
siblings are unreadable at runtime, kaggle_usage #26/#19; that is precisely the
bug that made every earlier imatrix quant calibrate on a 10-sentence English
fallback).

F7b context: F7 (main `68033e8d`) names the pre-merged BERT QKV tensor
`enc.<N>.attn.qkv_merged.weight` and aliases `attn.{q,k,v}.weight` to it in
tools/quantize.cpp, so BERT-family imatrix coverage goes 36→72/74-class instead
of losing every q/k/v to ggml's auto `leaf_N`. Every imatrix artifact published
BEFORE that fix carries the defect. Targets:
  arctic-embed-m-v2  — PRIMARY: the only BERT-family model with published
                       imatrix artifacts (T19-E3); re-collect + re-quant.
                       Its published -q4_k-imatrix/-iq4_xs SHAs are PINNED in
                       examples/cli/model_hashes.h → uploads use "-f7" names.
  f2llm-v2-80m       — no-change CONTROL: decoder path (never pre-merged,
                       coverage was already 56/57); its numbers must reproduce
                       T19-E3 within noise. Also pinned → "-f7" names.
  granite-embedding-{97m,311m}-multilingual-r2 — BERT/ModernBERT pre-merge path
                       (converter splits Wqkv into F32 q/k/v; runtime re-merges)
                       and had NO imatrix calibrated yet (T19-E4) → first-time
                       artifacts under canonical names, nothing to collide with.

Nothing here decides a default: quants upload under distinct filenames and the
registry is untouched. Promotion is the coordinator's call.
"""
import os
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
if not WORK.exists():
    WORK = Path("/tmp/crisp-imatrix-t19")
    WORK.mkdir(parents=True, exist_ok=True)

# main carries F7 (68033e8d); the old feat/imatrix-quants branch predates it.
BRANCH = os.environ.get("CRISP_BRANCH", "main")
os.environ["CRISP_BRANCH"] = BRANCH
os.environ["MODELS"] = ("arctic-embed-m-v2,f2llm-v2-80m,"
                        "granite-embedding-97m-multilingual-r2,"
                        "granite-embedding-311m-multilingual-r2")
# arctic + f2llm-80m already have imatrix files, so the idempotence skip must be off.
os.environ["FORCE"] = "1"

# ── HF token: glob BOTH mount layouts before handing over to the harness ─────
# Run 1 of this kernel died with `hf_token_ok: False` and a 401 on every upload
# after a full 21-minute pipeline. The log says why:
#   "HF auth: /kaggle/input contains 1 entries: ['datasets']"
# On that worker the attached datasets mounted ONLY under the long path
# /kaggle/input/datasets/<acct>/<slug>/ (kaggle_usage #19). The harness's ccache
# warm globs that layout — `_warm_ccache_from_dataset` found
# /kaggle/input/datasets/chr1s4/crispasr-ccache/.ccache and warmed 2974 files —
# but `resolve_hf_token()` does not, so it returned None, `HfApi(token=None)`
# went out unauthenticated, and a *public* repo answered 401/RepositoryNotFound.
# resolve_hf_token() checks the environment FIRST, so exporting it here fixes
# the run without patching CrispASR (a harness fix is filed separately).
def _find_hf_token():
    import glob
    pats = ["/kaggle/input/*/hf_token.txt", "/kaggle/input/datasets/*/*/hf_token.txt"]
    for pat in pats:
        for p in sorted(glob.glob(pat)):
            try:
                tok = Path(p).read_text().strip()
            except OSError:
                continue
            if tok:
                print(f"[t19] HF token found at {p} (len {len(tok)})", flush=True)
                return tok
    print(f"[t19] WARNING: no hf_token.txt under {pats}", flush=True)
    return None


if not os.environ.get("HF_TOKEN"):
    _tok = _find_hf_token()
    if _tok:
        os.environ["HF_TOKEN"] = _tok
        os.environ["HUGGING_FACE_HUB_TOKEN"] = _tok
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    else:
        # Never burn 21 minutes of quota to fail at the first upload again.
        raise SystemExit("[t19] FATAL: no HF token — uploads would 401; aborting early")

REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
repo = WORK / "CrispEmbed"
if not repo.exists():
    subprocess.check_call(["git", "clone", "--depth", "1", "--branch", BRANCH,
                           REPO_URL, str(repo)])
    subprocess.check_call(["git", "-C", str(repo), "submodule", "update",
                           "--init", "--recursive"])
print(f"[t19] cloned {BRANCH} at "
      + subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                text=True).strip(), flush=True)

# F7 guard: this run is POINTLESS on a pre-F7 tree — the collector would file
# QKV statistics under leaf_N again and re-publish the exact defect F7b exists
# to repair. Abort before burning quota.
for rel in ("src/crispembed.cpp", "tools/quantize.cpp"):
    if "qkv_merged" not in (repo / rel).read_text(encoding="utf-8"):
        raise SystemExit(f"[t19] FATAL: clone lacks the F7 qkv_merged fix in {rel} "
                         f"(branch {BRANCH}) — refusing to re-publish the leaf_N defect")
print("[t19] F7 guard OK: qkv_merged present in crispembed.cpp + quantize.cpp", flush=True)

kdir = repo / "tools" / "kaggle" / "crispembed-imatrix-quant"
script = kdir / "imatrix_quant.py"
if not script.exists():
    raise SystemExit(f"[t19] FATAL: {script} missing in the clone")
for corpus in ("calib_corpus.jsonl", "eval_corpus.jsonl"):
    if not (kdir / corpus).exists():
        raise SystemExit(f"[t19] FATAL: {kdir / corpus} missing — refusing to "
                         f"calibrate on a fallback corpus")

# Exec the pipeline WITHOUT auto-running main(), then override the upload names
# for the two models whose canonical imatrix artifacts are SHA-pinned in
# examples/cli/model_hashes.h (arctic-embed-m-v2, f2llm-v2-80m — both published
# by T19-E3). "-f7" marks the post-F7 re-collection; the corpus is unchanged
# (the c2-generation jsonl committed in the clone). Granite r2 keeps the default
# canonical names — it has no prior imatrix artifacts to protect.
sys.path.insert(0, str(kdir))
ns = {"__name__": "__t19_f7b__", "__file__": str(script)}
exec(compile(script.read_text(), str(script), "exec"), ns)

QS_F7 = [
    ("q8_0",   False, None),                             # A/B reference
    ("q4_k",   False, None),                             # baseline, shows the delta
    ("q4_k",   True,  "{prefix}-q4_k-imatrix-f7.gguf"),
    ("iq4_xs", True,  "{prefix}-iq4_xs-f7.gguf"),
]
ns["OVERRIDES"]["arctic-embed-m-v2"] = {
    "meta_prefix": "arctic-embed-m-v2-f7", "quants": QS_F7,
}
ns["OVERRIDES"]["f2llm-v2-80m"] = {
    "meta_prefix": "f2llm-v2-80m-f7", "quants": QS_F7,
}
ns["main"]()
