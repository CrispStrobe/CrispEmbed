"""Seed / probe for the ${KAGGLE_ACCOUNT}/crispembed-ccache dataset (CPU + CUDA builds).

Clones an exact SHA (pusher substitutes __SHA__) into a path that contains the
SHA, so consecutive runs build from different absolute paths, then imports
kaggle_harness from THAT clone (Kaggle uploads only this script, so a side-file
harness never reaches the kernel). install_build_toolchain() warms the cache
from the attached dataset when present. Two builds, each measured with
`ccache -z` / `ccache --print-stats`:
  cpu   portable CPU Release (frozen-replay flags), crispembed-cli + test-ovisocr2-quality-run
  cuda      GGML_CUDA at the detected arch (build-cuda)
  cuda-<a>  GGML_CUDA for every other arch in CE_CCACHE_SEED_ARCHS (default "75,60":
            T4 and P100; nvcc needs no matching GPU), build-cuda-<a> at the same depth
  CUDA targets: crispembed-cli test-ovisocr2-quality-run test-qwen35-profile
Then maybe_export_ccache() writes ONE /kaggle/working/ccache.tar only when the
seed is stale (combined hit rate below CE_CCACHE_MIN_HIT, or seed too old);
`kpush.py <dir> --sha <sha> --refresh-ccache` then versions the dataset from it.
summary.json carries times and hit/miss counts (the cold-vs-warm evidence).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SHA = "__SHA__"
REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
WORK, TEMP = Path("/kaggle/working"), Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)
SRC = TEMP / f"ce-{SHA[:12]}"
summary = {"sha": SHA, "src": str(SRC), "builds": {}}


def save():
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))


subprocess.check_call(f"git clone -q {REPO_URL} {SRC} && git -C {SRC} checkout -q {SHA} && "
                      f"git -C {SRC} submodule update --init --recursive -q", shell=True)
sys.path.insert(0, str(SRC))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
tool = kh.install_build_toolchain()
summary["toolchain"] = tool
summary["ccache_env"] = {k: v for k, v in os.environ.items() if k.startswith("CCACHE_")}
summary["seed_files"] = sum(1 for p in Path(os.environ.get("CCACHE_DIR", "/nonexistent")).rglob("*") if p.is_file())
arch = kh.detect_cuda_arch()
summary["cuda_arch"] = arch
seed_archs = [a.strip() for a in os.environ.get("CE_CCACHE_SEED_ARCHS", "75,60").split(",") if a.strip()]
# CUDA 13 dropped offline compilation for sm_50..sm_72 (P100 is sm_60); keep only
# arches this nvcc can target, and record the ones skipped.
_nvcc_archs = subprocess.run("/usr/local/cuda/bin/nvcc --list-gpu-arch", shell=True, capture_output=True,
                             text=True).stdout.split()
extra_archs = [a for a in seed_archs if a != arch and (not _nvcc_archs or f"compute_{a}" in _nvcc_archs)]
summary["cuda_archs_skipped"] = [a for a in seed_archs if a != arch and a not in extra_archs]
os.environ["CE_CCACHE_CUDA_ARCH"] = ",".join([arch] + extra_archs)
summary["toolchain_fingerprint"] = kh.toolchain_fingerprint() if hasattr(kh, "toolchain_fingerprint") else None
common = "-G Ninja -DCMAKE_BUILD_TYPE=Release " + " ".join(kh.cache_and_link_flags())
builds = [("cpu", "-DCRISPEMBED_NATIVE=OFF -DGGML_NATIVE=OFF -DGGML_BLAS=OFF -DBUILD_SHARED_LIBS=OFF "
           "-DCRISPEMBED_BUILD_SHARED=OFF", "crispembed-cli test-ovisocr2-quality-run", "$(nproc)")]
if Path("/usr/local/cuda/bin/nvcc").exists():
    cuda_targets = "crispembed-cli test-ovisocr2-quality-run test-qwen35-profile"
    builds.append(("cuda", " ".join(kh.cuda_build_flags(arch)), cuda_targets, kh.safe_build_jobs(gpu=True)))
    for a in extra_archs:
        builds.append((f"cuda-{a}", " ".join(kh.cuda_build_flags(a)), cuda_targets, kh.safe_build_jobs(gpu=True)))
for name, flags, targets, jobs in builds:
    bdir = SRC / f"build-{name}"
    subprocess.run("ccache -z", shell=True, capture_output=True)
    t = time.time()
    rc = subprocess.run(f"cmake -S {SRC} -B {bdir} {common} {flags}", shell=True).returncode
    t_conf = time.time() - t
    with kh.build_heartbeat(name, 60):
        rc = rc or subprocess.run(f"cmake --build {bdir} --target {targets} -j{jobs}", shell=True).returncode
    stats = subprocess.run("ccache --print-stats", shell=True, capture_output=True, text=True).stdout
    kv = dict(l.split("\t", 1) for l in stats.splitlines() if "\t" in l)
    hits = int(kv.get("direct_cache_hit", 0)) + int(kv.get("preprocessed_cache_hit", 0))
    misses = int(kv.get("cache_miss", 0))
    summary["builds"][name] = {"rc": rc, "configure_s": round(t_conf, 1), "total_s": round(time.time() - t, 1),
                               "hits": hits, "misses": misses,
                               "hit_rate": round(hits / max(1, hits + misses), 4),
                               "direct_hits": int(kv.get("direct_cache_hit", 0)),
                               "ccache_s": subprocess.run("ccache -s", shell=True, capture_output=True,
                                                          text=True).stdout[-1200:]}
    print(name, summary["builds"][name], flush=True)
    save()
tot_h = sum(b["hits"] for b in summary["builds"].values())
tot_m = sum(b["misses"] for b in summary["builds"].values())
summary["export"] = kh.maybe_export_ccache(tot_h, tot_m, repo=SRC)
save()
print("done", flush=True)
