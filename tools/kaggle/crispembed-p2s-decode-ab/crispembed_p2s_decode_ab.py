"""CrispEmbed pix2struct decode A/B v1 — CUDA-first ggml decode-step graph.

Round N+4 queue #1 (PLAN.md): the decoder is ~80-90 ms/tok of per-layer small
matvecs on CPU (profile in HISTORY's fix/pix2struct-decoder row — informed,
not re-derived). CRISPEMBED_PIX2STRUCT_GGML_DECODE=1 swaps the scalar loop
for a single-backend ggml step graph with device-resident self/cross KV
(got_ocr pattern). Locally proven decoded-output byte-identical on CPU and
Metal (f16 + q8_0, fox + scan_strip). This kernel supplies the CUDA half:
LEARNING 35 — a CUDA default claim needs a CUDA decoded-text roundtrip.

Arms (each lever varied alone, interleaved reps):
  base     : default scalar decode, CPU encoder
  encgpu   : CRISPEMBED_PIX2STRUCT_ENC_GPU=1        (encoder on CUDA, scalar decode)
  ggmlcpu  : CRISPEMBED_PIX2STRUCT_GGML_DECODE=1    (CPU ggml decode graph)
  cuda     : ENC_GPU=1 + GGML_DECODE=1              (the CUDA-first target)

Proof-of-work: every timed row carries stdout bytes + sha; an empty stdout at
rc=0 is a FAIL, never a timing (the --ocr filename-autodetect trap minted six
fake "timed" runs once). Decoded text is compared across ALL arms; the
[pix2struct-bench] encoder/decoder stage lines are the timing signal.

Everything lands in /kaggle/working/p2sab.log. Model stages under /tmp.
The ccache line MUST read "ccache warmed from" (or the harness's own line) —
a cold build means the chr1s4/crispembed-ccache seed regressed.
"""
import os
import sys
import subprocess
import hashlib
import re
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)
DL = Path("/tmp/crispembed-p2sab")
DL.mkdir(parents=True, exist_ok=True)

_LOG = open(WORK / "p2sab.log", "w", buffering=1)


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
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "perf/pix2struct-cuda-decode")
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
GPU_NAME = gpu.stdout.strip() or "none"
print(f"GPU: {GPU_NAME}", flush=True)
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
        if (base / ".ccache").exists():
            print(f"  ccache: bare tree at {base / '.ccache'} (harness warmer handles it)", flush=True)
            return
    print("  ccache: cold build (no tar, no tree — reseed via crispembed-ccache-seed)", flush=True)


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


def get(repo, fname):
    try:
        return hf_hub_download(repo_id=repo, filename=fname, local_dir=str(DL))
    except Exception as e:
        print(f"  DL FAIL {repo}/{fname}: {e}", flush=True)
        return None


BENCH_RE = re.compile(r"\[pix2struct-bench\] (\w+): ([\d.]+) ms")
PATH_RE = re.compile(r"\[pix2struct-bench\] decode path: (\w+)")

results = []


def note(row):
    results.append(row)
    print("RESULT| " + row, flush=True)


def run(label, model, img, env=None, timeout=1200):
    print("\n" + "#" * 72)
    print(f"# {label}")
    print(f"# env+: {env or {}}")
    print("#" * 72, flush=True)
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{BUILD_DIR / 'bin'}:{e.get('LD_LIBRARY_PATH', '')}"
    e["CRISPEMBED_PIX2STRUCT_BENCH"] = "1"
    if env:
        e.update({k: str(v) for k, v in env.items()})
    argv = [binp("crispembed"), "-m", model, "--pix2struct", img, "-t", "4"]
    try:
        with kh.build_heartbeat(label[:40], 60):
            p = subprocess.run(argv, env=e, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT", flush=True)
        return None
    for line in p.stderr.splitlines():
        if any(k in line for k in ("bench", "falling back", "failed", "error", "CUDA")):
            print("  E| " + line, flush=True)
    stages = {m.group(1): float(m.group(2)) for m in BENCH_RE.finditer(p.stderr)}
    path = PATH_RE.search(p.stderr)
    sha = hashlib.sha256(p.stdout.encode()).hexdigest()[:12]
    nbytes = len(p.stdout.strip())
    # Proof-of-work: rc=0 with empty stdout is a FAIL, never a timing.
    ok = (p.returncode == 0 and nbytes > 0)
    note(f"{label}: rc={p.returncode} bytes={nbytes} sha={sha} "
         f"path={path.group(1) if path else '?'} "
         f"enc={stages.get('encoder', -1):.0f}ms dec={stages.get('decoder', -1):.0f}ms "
         f"total={stages.get('total', -1):.0f}ms {'OK' if ok else 'FAIL'}")
    return (p.stdout, stages, ok) if ok else None


ARMS = (
    ("base", {}),
    ("encgpu", {"CRISPEMBED_PIX2STRUCT_ENC_GPU": "1"}),
    ("ggmlcpu", {"CRISPEMBED_PIX2STRUCT_GGML_DECODE": "1"}),
    ("cuda", {"CRISPEMBED_PIX2STRUCT_ENC_GPU": "1", "CRISPEMBED_PIX2STRUCT_GGML_DECODE": "1"}),
)

IMG_FOX = str(EMBED_DIR / "tests/regression/images/fox.png")
IMG_STRIP = str(EMBED_DIR / "tests/regression/images/scan_strip.png")

# ── Main A/B: q8_0 (the shipped quant), fox, 3 interleaved reps × 4 arms ──
kh.step("p2s.q8.fox")
q8 = get("cstr/pix2struct-GGUF", "pix2struct-textcaps-q8_0.gguf")
texts = {}
if q8:
    for rep in range(3):
        for name, env in ARMS:
            r = run(f"q8.fox.{name}.rep{rep}", q8, IMG_FOX, env)
            if r:
                texts.setdefault(name, []).append(r)
    if "base" in texts:
        ref = texts["base"][0][0]
        for name, _ in ARMS:
            if name in texts:
                same = all(t[0] == ref for t in texts[name])
                note(f"q8.fox decoded-text {name} == base: {same}")
                if not same:
                    note(f"q8.fox {name} FIRST-DIFF: base={ref[:120]!r} vs {texts[name][0][0][:120]!r}")
        for name, _ in ARMS:
            if name in texts:
                decs = ["%.0f" % t[1].get("decoder", -1) for t in texts[name]]
                encs = ["%.0f" % t[1].get("encoder", -1) for t in texts[name]]
                note(f"q8.fox {name}: dec_ms={decs} enc_ms={encs}")

# ── Identity on the second fixture (1 rep × arms) ──
kh.step("p2s.q8.strip")
if q8:
    stexts = {}
    for name, env in ARMS:
        r = run(f"q8.strip.{name}", q8, IMG_STRIP, env)
        if r:
            stexts[name] = r[0]
    if "base" in stexts:
        for name in stexts:
            note(f"q8.strip decoded-text {name} == base: {stexts[name] == stexts['base']}")

# ── f16 identity pair (base vs cuda) ──
kh.step("p2s.f16.fox")
f16 = get("cstr/pix2struct-GGUF", "pix2struct-textcaps-f16.gguf")
if f16:
    ftexts = {}
    for name in ("base", "cuda"):
        env = dict(ARMS[0][1] if name == "base" else ARMS[3][1])
        r = run(f"f16.fox.{name}", f16, IMG_FOX, env)
        if r:
            ftexts[name] = r[0]
    if len(ftexts) == 2:
        note(f"f16.fox decoded-text cuda == base: {ftexts['cuda'] == ftexts['base']}")

print("\n" + "=" * 72)
print(f"SUMMARY (GPU: {GPU_NAME})")
print("=" * 72)
for r in results:
    print(r)
print("done", flush=True)
