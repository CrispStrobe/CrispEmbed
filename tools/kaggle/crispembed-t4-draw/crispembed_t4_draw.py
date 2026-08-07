"""CrispEmbed T4 draw — the LAYOUT_CONV_F16 tensor-core arm, chr1str pool.

Round N+4 queue #4: LAYOUT_CONV_F16 loses on M1 Metal AND on P100 (region
drift 20->19, time-neutral); only tensor-core hardware (T4) remains untested.
Four chr1s4 draws on 2026-08-07 all came up P100 (per-day sticky pool), so
this kernel runs under the chr1str account for an independent draw.
chr1str/crispembed-ccache is a same-day clone of the chr1s4 seed.

If line 2 says P100 again, the T4 question stays open — record and stop.
Trimmed to the layout arm only (the det/dbnet verdicts are already
replicated twice; no need to spend chr1str quota on them).

Everything lands in /kaggle/working/t4draw.log.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)
DL = Path("/tmp/crispembed-t4draw")
DL.mkdir(parents=True, exist_ok=True)

_LOG = open(WORK / "t4draw.log", "w", buffering=1)


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
try:
    kh.resolve_hf_token()
except Exception as e:
    print(f"hf token resolve failed (layout model is public, continuing): {e}", flush=True)

gpu = subprocess.run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
                     shell=True, capture_output=True, text=True)
GPU_NAME = gpu.stdout.strip() or "none"
print(f"GPU: {GPU_NAME}", flush=True)
print(f"T4-DRAW: {'YES - tensor-core arm is decisive' if 'T4' in GPU_NAME else 'NO - P100/other, LAYOUT_CONV_F16 stays open'}",
      flush=True)

kh.step("clone.crispembed")
if not EMBED_DIR.exists():
    kh.sh(f"git clone --depth 1 --recursive -b {BRANCH} {REPO_URL} {EMBED_DIR}")
BUILD_DIR.mkdir(exist_ok=True)
kh.sh("pip install -q huggingface_hub hf_transfer || true", check=False)
kh.install_build_toolchain()


def _warm_ccache():
    import tarfile
    ccache_dir = Path(os.environ.get("CCACHE_DIR", str(WORK / ".ccache")))
    ccache_dir.mkdir(parents=True, exist_ok=True)
    for base in (Path("/kaggle/input/crispembed-ccache"),
                 Path("/kaggle/input/datasets/chr1str/crispembed-ccache"),
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
        if (base / ".ccache").exists():
            print(f"  ccache: bare tree at {base / '.ccache'} (harness warmer handles it)", flush=True)
            return
    print("  ccache: cold build (~19 min on this account — acceptable, the draw is the point)", flush=True)


_warm_ccache()

arch = kh.detect_cuda_arch()
print(f"CUDA arch: {arch}", flush=True)
kh.step("cmake.configure")
kh.sh(f"cd {BUILD_DIR} && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release "
      + " ".join(kh.cuda_build_flags(arch) + kh.cache_and_link_flags())
      + " ..", check=True)
kh.step("build")
with kh.build_heartbeat("ninja", 30):
    kh.sh(f"cd {BUILD_DIR} && ninja -j{kh.safe_build_jobs(gpu=True)} crispembed-cli", check=True)
kh.sh("ccache -s | grep -iE 'hit|miss' | head -4 || true", check=False)


def binp(name):
    for p in (BUILD_DIR / "bin" / name, BUILD_DIR / name):
        if p.exists():
            return str(p)
    return str(BUILD_DIR / name)


from huggingface_hub import hf_hub_download  # noqa: E402

results = []


def note(row):
    results.append(row)
    print("RESULT| " + row, flush=True)


def run(label, argv, env=None, timeout=900):
    print("\n" + "#" * 72)
    print(f"# {label}   env+: {env or {}}")
    print("#" * 72, flush=True)
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{BUILD_DIR / 'bin'}:{e.get('LD_LIBRARY_PATH', '')}"
    if env:
        e.update({k: str(v) for k, v in env.items()})
    try:
        with kh.build_heartbeat(label[:40], 60):
            p = subprocess.run(argv, env=e, capture_output=True, text=True, timeout=timeout)
        for line in p.stderr.splitlines():
            if any(k in line for k in ("Phase", "bench", "error", "FAIL")):
                print("  E| " + line, flush=True)
        print(f"  rc={p.returncode} stdout_bytes={len(p.stdout)}", flush=True)
        return p
    except subprocess.TimeoutExpired:
        print("  TIMEOUT", flush=True)
        return None


IMG_PAGE = str(EMBED_DIR / "tests/regression/images/scan_page_pd.png")

kh.step("layout-conv-f16")
try:
    layout_m = hf_hub_download(repo_id="cstr/layout-heron-gguf", filename="layout-heron-f32.gguf",
                               local_dir=str(DL))
except Exception as e:
    layout_m = None
    note(f"MODEL DL FAIL: {e}")

ltexts = {}
if layout_m:
    for name, env in (("f32", {}), ("f16", {"LAYOUT_CONV_F16": "1"})):
        e = dict(env)
        e["CRISPEMBED_LAYOUT_DETECT_BENCH"] = "1"
        e["CRISPEMBED_LAYOUT_REPEAT"] = "2"
        p = run(f"layout {name}",
                [binp("crispembed"), "-m", layout_m, "--layout", IMG_PAGE, "-t", "4"], env=e)
        if p and p.returncode == 0 and p.stdout.strip():
            ltexts[name] = p.stdout
            for l in (x for x in p.stderr.splitlines() if "Phase 1" in x):
                note(f"layout.{name} [{GPU_NAME}]: {l.strip()}")
        else:
            note(f"layout.{name}: rc={p.returncode if p else 'none'} "
                 f"bytes={len(p.stdout) if p else 0} FAIL — not evidence")
    if "f32" in ltexts and "f16" in ltexts:
        note(f"layout regions identical on {GPU_NAME}: {ltexts['f32'] == ltexts['f16']}")
        if ltexts["f32"] != ltexts["f16"]:
            a, b = ltexts["f32"].splitlines(), ltexts["f16"].splitlines()
            note(f"layout line counts: f32={len(a)} f16={len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    note(f"layout first diff line {i}: f32='{x}' f16='{y}'")
                    break

print("\n" + "=" * 72)
print(f"SUMMARY (GPU: {GPU_NAME})")
print("=" * 72)
for r in results:
    print(r)
print("done", flush=True)
