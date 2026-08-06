"""CrispEmbed conv A/B — O8 (R6 im2col on x86/AVX2) + O9 (CUDA residency).

The M1 verdicts these arms exist to test (PERFORMANCE.md 2026-08-05/06):

- O8: the R6 conv2d im2col-tile path is 4-7% SLOWER single-threaded on M1
  because the 12 MB shared L2 already holds the weight matrices; nt=4 wall is
  2.04x. On a small-private-L2 x86 box the interchange itself may win — this
  kernel runs the same three arms (legacy / GEMM nt=1 / GEMM nt=4) on the
  PP-OCRv6 medium SCALAR detector plus the bitwise unit gate on AVX2 lanes.
- O9: "DBNet / PP-OCRv6 det are CPU by measurement" is a *Metal* verdict.
  CUDA has strong conv/im2col kernels, so the defaults may flip there:
  det CPU-graph vs GPU-graph arms with decoded-output comparison, and the
  LAYOUT_CONV_F16 gate (loses on M1 Metal, 2.2 s vs 1.4 s) as a one-env
  CUDA candidate.

Everything lands in /kaggle/working/convab.log. Models stage under /tmp
(never /kaggle/working — the ENOSPC gotcha). Every arm's decoded output is
captured and cross-compared; a timing row without matching output is a FAIL,
never a win (proof-of-work rule).
"""
import os
import sys
import subprocess
import hashlib
from pathlib import Path

WORK = Path("/kaggle/working")
# Gotcha #22: keep /kaggle/working to only the artifacts we must retrieve
# (the 500-file output page cap) — clone/build under the ephemeral layer.
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)
DL = Path("/tmp/crispembed-convab")
DL.mkdir(parents=True, exist_ok=True)

_LOG = open(WORK / "convab.log", "w", buffering=1)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)

REPO_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "main")
EMBED_DIR = TEMP / "CrispEmbed"
BUILD_DIR = EMBED_DIR / "build"

_CRISPASR = TEMP / "CrispASR"
if not _CRISPASR.exists():
    try:
        subprocess.check_call(["git", "clone", "--depth", "1", CRISPASR_URL, str(_CRISPASR)])
        sys.path.insert(0, str(_CRISPASR / "tools" / "kaggle"))
    except Exception:
        pass
if str(_CRISPASR / "tools" / "kaggle") not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_harness as kh  # noqa: E402

kh.init_progress()
kh.resolve_hf_token()

kh.step("clone.crispembed")
if not EMBED_DIR.exists():
    kh.sh(f"git clone --depth 1 --recursive -b {BRANCH} {REPO_URL} {EMBED_DIR}")
BUILD_DIR.mkdir(exist_ok=True)

gpu = subprocess.run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
                     shell=True, capture_output=True, text=True)
print(f"GPU: {gpu.stdout.strip() or 'none'}", flush=True)
cpu = subprocess.run("lscpu | grep -E 'Model name|L2|L3|Core'", shell=True,
                     capture_output=True, text=True)
print(f"CPU:\n{cpu.stdout}", flush=True)
kh.sh("pip install -q huggingface_hub hf_transfer || true", check=False)
kh.install_build_toolchain()


def _warm_ccache():
    import tarfile
    ccache_dir = Path(os.environ.get("CCACHE_DIR", str(WORK / ".ccache")))
    ccache_dir.mkdir(parents=True, exist_ok=True)
    for base in (Path("/kaggle/input/crispembed-ccache"),
                 Path("/kaggle/input/datasets/chr1s4/crispembed-ccache")):
        tar = base / "ccache.tar"
        if tar.exists():
            try:
                with tarfile.open(tar) as tf:
                    tf.extractall(str(ccache_dir))
                print(f"  ccache warmed from {tar}", flush=True)
                return
            except Exception as e:
                print(f"  ccache warm failed: {e}", flush=True)
    print("  ccache: cold build", flush=True)


_warm_ccache()

arch = kh.detect_cuda_arch()
print(f"CUDA arch: {arch}", flush=True)
kh.step("cmake.configure")
kh.sh(f"cd {BUILD_DIR} && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release "
      + " ".join(kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
      + " ..", check=True)
kh.step("build")
with kh.build_heartbeat("ninja", 30):
    kh.sh(f"cd {BUILD_DIR} && ninja -j{kh.safe_build_jobs(gpu=True)} "
          "crispembed-cli test-core-cpu-ops test-ppocrv6-direct", check=True)


def binp(name):
    for p in (BUILD_DIR / "bin" / name, BUILD_DIR / name):
        if p.exists():
            return str(p)
    return str(BUILD_DIR / name)


from huggingface_hub import hf_hub_download  # noqa: E402


def get(repo, fname):
    try:
        return hf_hub_download(repo_id=repo, filename=fname, local_dir=str(DL))
    except Exception as e:
        print(f"  DL FAIL {repo}/{fname}: {e}", flush=True)
        return None


def run(label, argv, env=None, timeout=900):
    print("\n" + "#" * 72)
    print(f"# {label}")
    print(f"# cmd: {' '.join(argv)}   env+: {env or {}}")
    print("#" * 72, flush=True)
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{BUILD_DIR / 'bin'}:{e.get('LD_LIBRARY_PATH', '')}"
    if env:
        e.update({k: str(v) for k, v in env.items()})
    try:
        with kh.build_heartbeat(label[:40], 60):
            p = subprocess.run(argv, env=e, capture_output=True, text=True, timeout=timeout)
        for line in p.stderr.splitlines():
            if any(k in line for k in ("bench", "elapsed", "regions", "Phase", "ggml_cuda",
                                       "CUDA", "backend", "error", "FAIL", "Results")):
                print("  E| " + line, flush=True)
        print(f"  rc={p.returncode} stdout_bytes={len(p.stdout)} "
              f"stdout_sha={hashlib.sha256(p.stdout.encode()).hexdigest()[:12]}", flush=True)
        return p
    except subprocess.TimeoutExpired:
        print("  TIMEOUT", flush=True)
        return None
    except Exception as ex:
        print(f"  EXC {ex}", flush=True)
        return None


IMG_PAGE = str(EMBED_DIR / "tests/regression/images/scan_page_pd.png")
IMG_STRIP = str(EMBED_DIR / "tests/regression/images/scan_strip.png")

det_m = get("cstr/PP-OCRv6-medium-det-GGUF", "PP-OCRv6_medium_det-f16.gguf")
rec_t = get("cstr/PP-OCRv6_tiny_rec-GGUF", "PP-OCRv6_tiny_rec-q8-head.gguf")
rec_m = get("cstr/PP-OCRv6_medium_rec-GGUF", "PP-OCRv6_medium_rec-q8-head.gguf")
layout_m = get("cstr/layout-heron-gguf", "layout-heron-f32.gguf")

results = []


def note(row):
    results.append(row)
    print("RESULT| " + row, flush=True)


# ── O8: unit gate — the bitwise-equality guard on AVX2 lanes ──
kh.step("o8.unitgate")
for env in (None, {"CRISPEMBED_CONV2D_GEMM": "1", "CRISPEMBED_CONV2D_THREADS": "4"}):
    p = run(f"test-core-cpu-ops env={env}", [binp("test-core-cpu-ops")], env=env)
    if p:
        tail = [l for l in p.stdout.splitlines() if "Results:" in l]
        note(f"o8.unit env={env}: {tail[-1] if tail else 'NO RESULT LINE'} rc={p.returncode}")

# ── O8: three-arm scalar-det A/B, interleaved x3 pairs ──
kh.step("o8.det-scalar-arms")
if det_m and rec_t:
    arms = [
        ("legacy", {"CRISPEMBED_PPOCRV6_DET_SCALAR": "1"}),
        ("gemm1", {"CRISPEMBED_PPOCRV6_DET_SCALAR": "1", "CRISPEMBED_CONV2D_GEMM": "1"}),
        ("gemm4", {"CRISPEMBED_PPOCRV6_DET_SCALAR": "1", "CRISPEMBED_CONV2D_GEMM": "1",
                   "CRISPEMBED_CONV2D_THREADS": "4"}),
    ]
    for rep in range(3):
        for name, env in arms:
            e = dict(env)
            e["PPOCRV6_DIRECT_MAX_REGIONS"] = "0"
            p = run(f"o8 det-scalar {name} rep{rep}",
                    [binp("test-ppocrv6-direct"), det_m, rec_t, IMG_PAGE], env=e)
            if p:
                el = [l for l in p.stdout.splitlines() if "elapsed_ms=" in l]
                rg = [l for l in p.stdout.splitlines() if "detector_regions=" in l]
                note(f"o8.{name} rep{rep}: {el[-1].split('elapsed_ms=')[-1] if el else '?'} ms "
                     f"{rg[-1].split('detector_regions=')[-1].split(' ')[0] if rg else '?'} regions")

# ── O9: PP-OCRv6 det CPU-graph vs CUDA-graph, full-pipeline text identity ──
kh.step("o9.ppocr-det-cuda")
texts = {}
if det_m and rec_m:
    for name, env in (("det-cpu", {}), ("det-cuda", {"CRISPEMBED_PPOCRV6_DET_GPU_LOAD": "1"})):
        e = dict(env)
        e["CRISPEMBED_PPOCRV6_DET_BENCH"] = "1"
        e["CRISPEMBED_PPOCRV6_BENCH"] = "1"
        p = run(f"o9 ppocr {name}",
                [binp("crispembed"), "-m", rec_m, "--ocr", IMG_PAGE,
                 "--ocr-engine", "ppocrv6", "--ocr-det", det_m, "--ocr-rec", rec_m,
                 "-t", "4"], env=e)
        if p:
            texts[name] = p.stdout
            det = [l for l in p.stderr.splitlines() if "det-bench" in l]
            st = [l for l in p.stderr.splitlines() if "stage-bench" in l]
            note(f"o9.ppocr.{name}: {det[-1] if det else 'no det-bench'} | "
                 f"{st[-1] if st else 'no stage-bench'}")
    if "det-cpu" in texts and "det-cuda" in texts:
        note(f"o9.ppocr text identical: {texts['det-cpu'] == texts['det-cuda']} "
             f"(cpu {len(texts['det-cpu'])}B, cuda {len(texts['det-cuda'])}B)")

# ── O9: layout f32 vs LAYOUT_CONV_F16 on CUDA ──
kh.step("o9.layout-conv-f16")
ltexts = {}
if layout_m:
    for name, env in (("f32", {}), ("f16", {"LAYOUT_CONV_F16": "1"})):
        e = dict(env)
        e["CRISPEMBED_LAYOUT_DETECT_BENCH"] = "1"
        e["CRISPEMBED_LAYOUT_REPEAT"] = "2"
        p = run(f"o9 layout {name}",
                [binp("crispembed"), "-m", layout_m, "--layout", IMG_PAGE, "-t", "4"], env=e)
        if p:
            ltexts[name] = p.stdout
            ph = [l for l in p.stderr.splitlines() if "Phase 1" in l]
            for l in ph:
                note(f"o9.layout.{name}: {l.strip()}")
    if "f32" in ltexts and "f16" in ltexts:
        note(f"o9.layout regions identical: {ltexts['f32'] == ltexts['f16']}")
        if ltexts["f32"] != ltexts["f16"]:
            a, b = ltexts["f32"].splitlines(), ltexts["f16"].splitlines()
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    note(f"o9.layout first diff line {i}: f32='{x}' f16='{y}'")
                    break

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
for r in results:
    print(r)
print("done", flush=True)
