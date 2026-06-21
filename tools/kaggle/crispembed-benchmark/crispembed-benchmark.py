# CrispEmbed — engine perf sweep (Kaggle)
#
# Companion to tools/kaggle/crispembed-regression/. Builds the crispembed CLI
# and runs tools/benchmark_all_engines.py once per manifest entry, pushing each
# engine's timing to the cstr/crispembed-kaggle-progress HF dataset right after
# it completes — so a crashed/timed-out sweep still leaves every finished
# engine's numbers on HF.
#
# Env: CRISPEMBED_REF (default main), CRISPEMBED_BENCH_BACKENDS (csv filter),
#      CRISPEMBED_BENCH_BUILD (cpu|cuda), CRISPEMBED_BENCH_REPEAT (default 3).
# Attach chr1s4/crispasr-hf-token + chr1s4/crispasr-ccache. Internet ON.

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "CrispEmbed"
BUILD = WORK / "build"
HF_CACHE = WORK / "hf_cache"
RESULTS = WORK / "results"
for d in (HF_CACHE, RESULTS):
    d.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)

PROGRESS_REPO = "cstr/crispembed-kaggle-progress"
# Defaults to the regression-suite branch (where this suite lives) until merged.
REF = os.environ.get("CRISPEMBED_REF", "regression-suite")
FILTER = {b.strip() for b in os.environ.get("CRISPEMBED_BENCH_BACKENDS", "").split(",") if b.strip()}
FLAVOUR = os.environ.get("CRISPEMBED_BENCH_BUILD", "cpu")
REPEAT = os.environ.get("CRISPEMBED_BENCH_REPEAT", "3")
URL = "https://github.com/CrispStrobe/CrispEmbed.git"

# Clone FIRST, then import kaggle_harness from the clone — a Kaggle `script`
# kernel doesn't ship sibling .py files at runtime.
def _sh(cmd, cwd=None):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True, cwd=str(cwd) if cwd else None)

if not REPO.exists():
    _sh(f"git clone --recursive {URL} {REPO}")
_sh(f"git fetch origin && git checkout {REF}", cwd=REPO)
_sh("git submodule update --init --recursive", cwd=REPO)

sys.path.insert(0, str(REPO / "tools" / "kaggle" / "crispembed-benchmark"))
import kaggle_harness as kh  # noqa: E402 — from the cloned repo
kh.init_progress(progress_path=str(WORK / "progress.jsonl"),
                 hf_progress_repo=PROGRESS_REPO)
kh.step("script.start", ref=REF)

kh.step("auth")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "huggingface_hub"])
tok = kh.resolve_hf_token("HF_TOKEN")
if tok:
    os.environ["HF_TOKEN"] = tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = tok

kh.step("toolchain")
os.environ["CCACHE_BASEDIR"] = str(WORK)
os.environ.setdefault("CCACHE_SLOPPINESS",
                      "locale,time_macros,include_file_ctime,include_file_mtime")
kh.install_build_toolchain()
flags = kh.cache_and_link_flags()
flags += kh.cuda_build_flags(kh.detect_cuda_arch()) if FLAVOUR == "cuda" else ["-DGGML_BLAS=ON"]

kh.step("build")
with kh.build_heartbeat("cmake.configure"):
    kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release "
          f"-DCRISPEMBED_BUILD_SHARED=OFF " + " ".join(flags))
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(f"stdbuf -oL -eL cmake --build {BUILD} --target crispembed-cli "
                        f"-j{kh.safe_build_jobs(gpu=(FLAVOUR == 'cuda'))}")
# Export the populated ccache to seed/refresh chr1s4/crispembed-ccache.
try:
    # Pack .ccache CONTENTS (top-level hash dirs); a tar of `.ccache/` would
    # double-nest on extract and stay unrecognized (the CrispASR ccache bug).
    kh.sh(f"tar -C {WORK}/.ccache -cf {WORK}/ccache.tar . && ls -la {WORK}/ccache.tar")
    kh.sh(f"rm -rf {WORK}/.ccache")
    kh.step("ccache.exported")
    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi
        HfApi(token=os.environ["HF_TOKEN"]).upload_file(
            path_or_fileobj=str(WORK / "ccache.tar"),
            path_in_repo="ccache/ccache.tar",
            repo_id=PROGRESS_REPO, repo_type="dataset",
            commit_message=f"ccache seed ({REF})")
        kh.step("ccache.uploaded_hf")
except Exception as _e:
    print(f"  ccache export skipped: {_e}", flush=True)

MANIFEST = json.loads((REPO / "tests" / "regression" / "manifest.json").read_text())
backends = [b["name"] for b in MANIFEST["backends"] if not FILTER or b["name"] in FILTER]

RESULTS_JSONL = RESULTS / f"benchmark-{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl"
per_backend = RESULTS / "one.jsonl"


def publish(rec: dict):
    kh.step("bench", backend=rec.get("backend"), ok=rec.get("ok"),
            total_ms=rec.get("total_ms"))
    kh._push_progress_to_hf(force=True)
    with RESULTS_JSONL.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    if os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=str(RESULTS_JSONL),
                path_in_repo=f"benchmarks/{RESULTS_JSONL.name}",
                repo_id=PROGRESS_REPO, repo_type="dataset",
                commit_message=f"bench {rec.get('backend')}")
        except Exception as exc:
            print(f"  (upload skipped: {type(exc).__name__})", flush=True)


all_results = []
for name in backends:
    print(f"\n========== bench :: {name} ==========")
    t0 = time.time()
    rc = subprocess.run(
        [sys.executable, str(REPO / "tools" / "benchmark_all_engines.py"),
         "--backend", name, "--build-dir", str(BUILD),
         "--repeat", REPEAT, "--out", str(per_backend)],
        capture_output=True, text=True)
    print(rc.stdout[-800:])
    rec = {"backend": name, "ok": False, "error": rc.stderr[-200:]}
    if per_backend.exists():
        try:
            rec = json.loads(per_backend.read_text().strip().splitlines()[-1])
        except Exception:
            pass
    rec["wall_s"] = round(time.time() - t0, 1)
    all_results.append(rec)
    publish(rec)
    # Free the HF cache after each model so a 130+ model sweep fits ~20 GB.
    import shutil
    for d in (HF_CACHE, Path.home() / ".cache" / "huggingface"):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    HF_CACHE.mkdir(parents=True, exist_ok=True)

n_ok = sum(1 for r in all_results if r.get("ok"))
print(f"\nSUMMARY  benchmarked {n_ok}/{len(all_results)} OK")
kh.step("done", ok=n_ok, total=len(all_results))
kh._push_progress_to_hf(force=True)
