"""CPU-image ccache probe: the standalone CMake trees of tools/kaggle/minicpm-ocr-graph, nothing else.

Clones the exact source (__SHA__) to the same paths that kernel uses and builds, through the harness
(`ensure_build_cache`: CCACHE_BASEDIR=/kaggle, NOHASHDIR, COMPILERCHECK=content, env compiler launchers):
  tests/minicpm-graph -> /kaggle/temp/g   (all targets; then runs minicpm-graph-controls: link + run proof)
  root CPU tree       -> /kaggle/temp/b   (crispembed-cli crispembed-quantize test-ultrabert-profile)
  tests/josev-quant   -> /kaggle/temp/jq  (josev-quant-controls)
with the same flags as that kernel. Records ccache hits/misses per tree. Push once with --refresh-ccache
(cold: exports and versions the seed), then once more (warm): the warm run's hit rate is the evidence.
Output: summary.json (+ ccache-export.json / ccache.tar from the harness when the seed is stale).
"""
import json, os, subprocess, sys, time
from pathlib import Path

SHA = "__SHA__"
SRC = Path("/kaggle/temp/ce")
OUT = Path("/kaggle/working")
summary = {"schema": "crispembed-ccache-cpu-probe-v1", "sha": SHA, "builds": [], "started": time.time()}


def save():
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))


def sh(cmd, cwd=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    t = time.time()
    r = subprocess.run(list(map(str, cmd)), cwd=cwd)
    return r.returncode, round(time.time() - t, 1)


try:
    Path("/kaggle/temp").mkdir(parents=True, exist_ok=True)
    assert sh(["git", "clone", "-q", "--filter=blob:none", "https://github.com/CrispStrobe/CrispEmbed.git", SRC])[0] == 0
    os.chdir(SRC)
    assert sh(["git", "checkout", "-q", SHA])[0] == 0
    assert sh(["git", "submodule", "update", "-q", "--init", "--depth", "1", "ggml"])[0] == 0
    sys.path.insert(0, str(SRC))
    import kaggle_harness as kh
    kh.ensure_build_cache()
    summary["env"] = {k: os.environ.get(k) for k in ("CCACHE_BASEDIR", "CCACHE_NOHASHDIR", "CCACHE_COMPILERCHECK",
                                                      "CCACHE_SLOPPINESS", "CMAKE_C_COMPILER_LAUNCHER",
                                                      "CMAKE_CXX_COMPILER_LAUNCHER")}
    summary["toolchain"] = kh.toolchain_fingerprint()
    summary["seed"] = {k: v for k, v in kh._seed_info.items() if k != "toolchain"}
    trees = [
        ("minicpm-graph", ["cmake", "-S", "tests/minicpm-graph", "-B", "/kaggle/temp/g", "-DCMAKE_BUILD_TYPE=Release",
                           "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF"],
         ["cmake", "--build", "/kaggle/temp/g", "--parallel", "4"]),
        ("root-cpu", ["cmake", "-S", ".", "-B", "/kaggle/temp/b", "-DCMAKE_BUILD_TYPE=Release",
                      "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_NATIVE=OFF",
                      "-DGGML_LLAMAFILE=ON", "-DGGML_CUDA=OFF"],
         ["cmake", "--build", "/kaggle/temp/b", "--target", "crispembed-cli", "crispembed-quantize",
          "test-ultrabert-profile", "--parallel", "4"]),
        ("josev-quant", ["cmake", "-S", "tests/josev-quant", "-B", "/kaggle/temp/jq", "-DCMAKE_BUILD_TYPE=Release"],
         ["cmake", "--build", "/kaggle/temp/jq", "--target", "josev-quant-controls", "--parallel", "4"]),
    ]
    for name, conf, build in trees:
        before = kh.ccache_counters()
        rc1, s1 = sh(conf)
        rc2, s2 = sh(build) if rc1 == 0 else (None, 0)
        after = kh.ccache_counters()
        h, m = after["hits"] - before["hits"], after["misses"] - before["misses"]
        summary["builds"].append({"tree": name, "configure_rc": rc1, "build_rc": rc2, "build_s": s2, "hits": h,
                                  "misses": m, "hit_rate": round(h / max(1, h + m), 4)})
        save()
    rc, _ = sh(["/kaggle/temp/g/minicpm-graph-controls"])
    summary["minicpm_graph_controls_rc"] = rc
    tot = kh.ccache_counters()
    summary["total"] = dict(tot, hit_rate=round(tot["hits"] / max(1, tot["hits"] + tot["misses"]), 4))
except BaseException as e:
    summary["error"] = repr(e)[:800]
    raise
finally:
    summary["elapsed_s"] = round(time.time() - summary.pop("started", time.time()), 1)
    save()
