#!/usr/bin/env python3
"""T19 imatrix run — arctic-embed-m-v2 + the F2LLM-v2 family (Kaggle, chr1s4).

A thin driver, deliberately: ALL the logic lives in
`tools/kaggle/crispembed-imatrix-quant/imatrix_quant.py` and is executed FROM THE
CLONE, so there is exactly one copy of the pipeline and the calibration corpora
travel with it (a Kaggle *script* kernel ships only its `code_file` — bundled
siblings are unreadable at runtime, kaggle_usage #26/#19; that is precisely the
bug that made every earlier imatrix quant calibrate on a 10-sentence English
fallback).

Targets, in ascending cost so the most important result lands first:
  arctic-embed-m-v2  — PRIMARY: its q4_k without imatrix is weak (T19-E2:
                       cos_min 0.954), q8_0 is today's registry default
  f2llm-v2-80m/160m/330m — no imatrix quants exist yet
  f2llm-v2-0.6b      — re-calibration; publishes under "-c2" names because the
                       existing -q4_k-imatrix / -iq4_xs SHAs are PINNED in
                       examples/cli/model_hashes.h

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

BRANCH = os.environ.get("CRISP_BRANCH", "feat/imatrix-quants")
os.environ["CRISP_BRANCH"] = BRANCH
os.environ["MODELS"] = ("arctic-embed-m-v2,f2llm-v2-80m,f2llm-v2-160m,"
                        "f2llm-v2-330m,f2llm-v2-0.6b")
# f2llm-v2-0.6b already has imatrix files, so the idempotence skip must be off.
os.environ["FORCE"] = "1"

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

kdir = repo / "tools" / "kaggle" / "crispembed-imatrix-quant"
script = kdir / "imatrix_quant.py"
if not script.exists():
    raise SystemExit(f"[t19] FATAL: {script} missing in the clone")
for corpus in ("calib_corpus.jsonl", "eval_corpus.jsonl"):
    if not (kdir / corpus).exists():
        raise SystemExit(f"[t19] FATAL: {kdir / corpus} missing — refusing to "
                         f"calibrate on a fallback corpus")

sys.path.insert(0, str(kdir))
exec(compile(script.read_text(), str(script), "exec"),
     {"__name__": "__main__", "__file__": str(script)})
