# CrispEmbed — full-suite regression / re-bake (Kaggle)
#
# Mirrors CrispASR's tools/kaggle/crispasr-regression.py, adapted to
# CrispEmbed's per-engine `test-<engine>-diff` harnesses + the `crispembed`
# CLI. Two modes (env CRISPEMBED_REGRESSION_MODE):
#
#   MODE="validate" (default) — for each tests/regression/manifest.json entry:
#       diff tier : download pinned GGUF + ref, run test-<engine>-diff, assert
#                   every stage's cos_min >= its manifest threshold.
#       smoke tier: download pinned GGUF, run the crispembed CLI, assert the
#                   output is sane (finite, correctly-shaped embedding).
#   MODE="rebake"  — regenerate refs for entries that carry a `rebake` recipe;
#                    stage to /kaggle/working/rebake-stage/<repo>/<path>;
#                    optional gated upload to the per-model GGUF repo's
#                    diff-harness-ref/ (UPLOAD=1).
#
# CRASH-SURVIVABLE RESULTS (the whole point of running on Kaggle): after EVERY
# engine we (a) step() the full verdict into progress.jsonl and force-push it to
# the cstr/crispembed-kaggle-progress HF dataset, and (b) append to + upload a
# rolling results-<mode>.jsonl to the same dataset. A kernel that OOMs / times
# out at engine #12 still leaves engines #1..#11's verdicts on HF.
#
# Requirements: Internet ON. Attach chr1s4/crispasr-hf-token (HF token) and
# chr1s4/crispasr-ccache (warm build cache). HF token only needs write scope
# for MODE=rebake UPLOAD=1; read scope is enough for validate (public repos).

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── workspace + config (no kaggle_harness yet; it ships in the repo we clone) ─
WORK = Path("/kaggle/working")
REPO = WORK / "CrispEmbed"
BUILD = WORK / "build"
HF_CACHE = WORK / "hf_cache"
RESULTS = WORK / "results"
REBAKE_STAGE = WORK / "rebake-stage"
for d in (HF_CACHE, RESULTS, REBAKE_STAGE):
    d.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE)

PROGRESS_REPO = "cstr/crispembed-kaggle-progress"
MODE = os.environ.get("CRISPEMBED_REGRESSION_MODE", "validate")
UPLOAD = os.environ.get("CRISPEMBED_REGRESSION_UPLOAD", "0") == "1"
BACKEND_FILTER = os.environ.get("CRISPEMBED_REGRESSION_BACKENDS", "").strip()
TIER_FILTER = os.environ.get("CRISPEMBED_REGRESSION_TIER", "").strip()  # diff|smoke|""
BUILD_FLAVOUR = os.environ.get("CRISPEMBED_REGRESSION_BUILD", "cpu")    # cpu|cuda
# Default ref is the regression-suite branch (where this suite lives) until it
# merges to main; override with $CRISPEMBED_REF.
CRISPEMBED_REF = os.environ.get("CRISPEMBED_REF", "regression-suite")
CRISPEMBED_URL = "https://github.com/CrispStrobe/CrispEmbed.git"

print(f"crispembed-regression {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"  MODE={MODE}  BUILD={BUILD_FLAVOUR}  REF={CRISPEMBED_REF}  "
      f"BACKENDS={BACKEND_FILTER or '(all)'}  TIER={TIER_FILTER or '(all)'}  "
      f"UPLOAD={UPLOAD}", flush=True)

# ── clone the repo FIRST, then import kaggle_harness from it ──────────────────
# A Kaggle `script` kernel only ships its code_file at runtime — sibling .py
# files in the push dir are NOT importable. So kaggle_harness must come from the
# clone, and the pre-clone phase uses a plain inline shell helper.
def _sh(cmd, cwd=None):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True, cwd=str(cwd) if cwd else None)

if not REPO.exists():
    _sh(f"git clone --recursive {CRISPEMBED_URL} {REPO}")
_sh(f"git fetch origin && git checkout {CRISPEMBED_REF}", cwd=REPO)
_sh("git submodule update --init --recursive", cwd=REPO)

sys.path.insert(0, str(REPO / "tools" / "kaggle" / "crispembed-regression"))
import kaggle_harness as kh  # noqa: E402 — from the cloned repo
kh.init_progress(progress_path=str(WORK / "progress.jsonl"),
                 hf_progress_repo=PROGRESS_REPO)
step = kh.step
step("script.start", ref=CRISPEMBED_REF)

# ── HF auth ──────────────────────────────────────────────────────────────────
step("auth")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                       "huggingface_hub"])
hf_token = kh.resolve_hf_token("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    print("HF auth: token present")
else:
    print("HF auth: anonymous (validate reads public repos OK; rebake+upload "
          "needs a write-scoped token via chr1s4/crispasr-hf-token).")

step("toolchain")
# Relocatable ccache hashes: rewrite /kaggle/working/* absolute paths to
# relative so a saved chr1s4/crispembed-ccache seed hits across runs (and even
# across the CrispASR/CrispEmbed dir split for shared ggml TUs).
os.environ["CCACHE_BASEDIR"] = str(WORK)
os.environ.setdefault("CCACHE_SLOPPINESS",
                      "locale,time_macros,include_file_ctime,include_file_mtime")
tc = kh.install_build_toolchain()
print(f"  toolchain: {tc}")

# Load the manifest now so we only build the diff binaries we actually need.
MANIFEST = json.loads((REPO / "tests" / "regression" / "manifest.json").read_text())
want = {b.strip() for b in BACKEND_FILTER.split(",") if b.strip()}
BACKENDS = [b for b in MANIFEST["backends"]
            if (not want or b["name"] in want)
            and (not TIER_FILTER or b.get("tier") == TIER_FILTER)]
diff_bins = sorted({b["diff_bin"] for b in BACKENDS if b.get("tier") == "diff"})

build_flags = kh.cache_and_link_flags()
if BUILD_FLAVOUR == "cuda":
    build_flags += kh.cuda_build_flags(kh.detect_cuda_arch())
else:
    build_flags += ["-DGGML_BLAS=ON"]

step("cmake.configure", n_backends=len(BACKENDS), diff_bins=diff_bins)
with kh.build_heartbeat("cmake.configure"):
    kh.sh(f"cmake -S {REPO} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release "
          f"-DCRISPEMBED_BUILD_SHARED=OFF " + " ".join(build_flags))

targets = "crispembed-cli " + " ".join(diff_bins)
step("cmake.build", targets=targets)
with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} --target {targets} "
        f"-j{kh.safe_build_jobs(gpu=(BUILD_FLAVOUR == 'cuda'))}")

# Export the populated ccache as a downloadable output so it can seed
# chr1s4/crispembed-ccache (warm subsequent builds). Refresh the dataset with:
#   kaggle datasets version -p <dir-with-ccache.tar> -m "refresh crispembed ccache"
try:
    kh.sh("ccache -s | grep -E 'cache hit|cache miss|files in cache' || true")
    # Pack the CONTENTS of .ccache (hash dirs at top level), NOT the .ccache dir
    # itself — the harness extracts into /kaggle/working/.ccache, so a tar of
    # `.ccache/` would double-nest and stay unrecognized (the CrispASR bug).
    kh.sh(f"tar -C {WORK}/.ccache -cf {WORK}/ccache.tar . && ls -la {WORK}/ccache.tar")
    # Remove the loose .ccache tree from /kaggle/working so the kernel OUTPUT is
    # just ccache.tar (one-page download to seed chr1s4/crispembed-ccache).
    kh.sh(f"rm -rf {WORK}/.ccache")
    step("ccache.exported")
except Exception as _e:
    print(f"  ccache export skipped: {_e}", flush=True)

# ── import the in-repo runner library ────────────────────────────────────────
sys.path.insert(0, str(REPO / "tests" / "regression"))
import run_one  # noqa: E402

RESULTS_JSONL = RESULTS / f"results-{MODE}-{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl"


def _publish_result(rec: dict) -> None:
    """Persist one engine's verdict everywhere, immediately. progress.jsonl
    (force-pushed) is the crash-survivable log; results-*.jsonl is the clean
    artifact, uploaded to the progress dataset after each engine."""
    step("result", **{k: rec[k] for k in ("backend", "tier", "ok") if k in rec})
    kh._push_progress_to_hf(force=True)
    with RESULTS_JSONL.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    if os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=str(RESULTS_JSONL),
                path_in_repo=f"results/{RESULTS_JSONL.name}",
                repo_id=PROGRESS_REPO, repo_type="dataset",
                commit_message=f"{rec['backend']} -> ok={rec.get('ok')}")
        except Exception as exc:
            print(f"  (results upload skipped: {type(exc).__name__})", flush=True)


# ── validate ─────────────────────────────────────────────────────────────────
def run_validate() -> list[dict]:
    results = []
    for entry in BACKENDS:
        name = entry["name"]
        print(f"\n========== validate :: {name} ({entry.get('tier')}) ==========")
        t0 = time.time()
        try:
            # Heartbeat so multi-GB downloads + heavy diffs (e.g. granite q8_0
            # ~2.7 GB) keep emitting liveness instead of a silent log freeze.
            with kh.build_heartbeat(f"validate.{name}"):
                r = run_one.regression_for(name, MANIFEST, HF_CACHE, BUILD)
            r["elapsed_s"] = round(time.time() - t0, 2)
            r["mode"] = "validate"
        except SystemExit as exc:
            r = {"backend": name, "tier": entry.get("tier"), "ok": False,
                 "mode": "validate", "elapsed_s": round(time.time() - t0, 2),
                 "error": f"SystemExit: {exc}"}
        except Exception as exc:
            r = {"backend": name, "tier": entry.get("tier"), "ok": False,
                 "mode": "validate", "elapsed_s": round(time.time() - t0, 2),
                 "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        print(f"  -> {r}")
        _publish_result(r)
    return results


# ── rebake ───────────────────────────────────────────────────────────────────
def run_rebake() -> list[dict]:
    """Regenerate refs for entries carrying a `rebake` recipe:
        "rebake": {"cmd": "python tools/dump_<e>_reference_from_gguf.py {model} {out}",
                   "needs_gguf": true}
    {model} = downloaded GGUF, {out} = staged ref path. Entries without a
    recipe are reported (a PLAN.md todo), not silently skipped."""
    results = []
    for entry in BACKENDS:
        name = entry["name"]
        print(f"\n========== rebake :: {name} ==========")
        t0 = time.time()
        recipe = entry.get("rebake")
        if not recipe or entry.get("tier") != "diff":
            results.append({"backend": name, "ok": False, "mode": "rebake",
                            "elapsed_s": 0.0,
                            "error": "no rebake recipe (add entry['rebake'])"})
            print("  -> SKIP (no rebake recipe)")
            _publish_result(results[-1])
            continue
        try:
          with kh.build_heartbeat(f"rebake.{name}"):
            model = run_one.hf_download(entry["gguf"]["repo"], entry["gguf"]["file"],
                                        entry["gguf"]["revision"], HF_CACHE) \
                if recipe.get("needs_gguf") else ""
            out_path = REBAKE_STAGE / entry["ref"]["repo"] / entry["ref"]["path"]
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = recipe["cmd"].format(model=model, out=str(out_path))
            subprocess.check_call(cmd, shell=True, cwd=str(REPO))
            r = {"backend": name, "ok": out_path.exists(), "mode": "rebake",
                 "elapsed_s": round(time.time() - t0, 2),
                 "out_path": str(out_path),
                 "out_size_b": out_path.stat().st_size if out_path.exists() else 0,
                 "ref_repo": entry["ref"]["repo"], "ref_path": entry["ref"]["path"]}
        except Exception as exc:
            r = {"backend": name, "ok": False, "mode": "rebake",
                 "elapsed_s": round(time.time() - t0, 2),
                 "error": f"{type(exc).__name__}: {exc}"}
        results.append(r)
        print(f"  -> {r}")
        _publish_result(r)

    if UPLOAD and os.environ.get("HF_TOKEN"):
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ["HF_TOKEN"])
        for r in [x for x in results if x.get("ok") and x.get("out_path")]:
            try:
                api.upload_file(path_or_fileobj=r["out_path"],
                                path_in_repo=r["ref_path"], repo_id=r["ref_repo"],
                                repo_type="model",
                                commit_message=f"rebake {r['backend']} ref "
                                               f"(crispembed {CRISPEMBED_REF})")
                print(f"  uploaded {r['ref_repo']}::{r['ref_path']}")
            except Exception as exc:
                print(f"  upload FAILED {r['ref_repo']}: {exc}")
    return results


# ── dispatch ─────────────────────────────────────────────────────────────────
step("dispatch", mode=MODE)
if MODE == "validate":
    RESULTS_DATA = run_validate()
elif MODE == "rebake":
    RESULTS_DATA = run_rebake()
else:
    raise SystemExit(f"unknown MODE={MODE!r}")

n_ok = sum(1 for r in RESULTS_DATA if r.get("ok"))
n_fail = len(RESULTS_DATA) - n_ok
print(f"\nSUMMARY  mode={MODE}  ok={n_ok}/{len(RESULTS_DATA)}  fail={n_fail}")
for r in RESULTS_DATA:
    print(f"  {'✓' if r.get('ok') else '✗'} {r['backend']:24s} "
          f"{r.get('elapsed_s', 0):6.1f}s  {r.get('error', '')}")
step("done", ok=n_ok, fail=n_fail)
kh._push_progress_to_hf(force=True)

# Non-zero exit so a Kaggle scheduled run shows red when anything regressed.
sys.exit(n_fail)
