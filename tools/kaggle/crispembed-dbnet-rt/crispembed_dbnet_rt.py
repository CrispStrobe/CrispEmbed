"""CrispEmbed dbnet auto-CUDA roundtrip — the LAST gate before the O11-style default.

Round N+4 queue #2 (PLAN.md): det-only DBNet on CUDA is ~6x and box-equivalent
(Δ≤1.0px, 0 unmatched, deterministic — replicated twice, conv-ab v3/v4). The
remaining question is whether the ±1px crop movement moves the RECOGNIZER:
this kernel runs the full dbnet+TrOCR pipeline with det on CPU vs CUDA and
compares DECODED TEXT (LEARNING 35: a CUDA default needs a CUDA decoded-text
roundtrip). v2's lesson is encoded: per-arm determinism reps + a ground-truth
CER for fox.png ("The quick brown fox jumps over the lazy dog. 12345"), so a
non-byte-equal result still gets an honest quality verdict instead of a bare
"differs".

Flip rule (decided by the reader, evidenced here): CUDA det is flip-ready if
its text is byte-equal to the CPU arm, OR its ground-truth CER is <= the CPU
arm's with per-arm determinism holding. Anything else stays opt-in.

Everything lands in /kaggle/working/dbnetrt.log. Models stage under /tmp.
"""
import os
import sys
import subprocess
import hashlib
import time
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)
DL = Path("/tmp/crispembed-dbnetrt")
DL.mkdir(parents=True, exist_ok=True)

_LOG = open(WORK / "dbnetrt.log", "w", buffering=1)


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


def cer(ref, hyp):
    """Plain Levenshtein / len(ref) on whitespace-normalized text."""
    r = " ".join(ref.split())
    h = " ".join(hyp.split())
    if not r:
        return 0.0 if not h else 1.0
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hc in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(r)


results = []


def note(row):
    results.append(row)
    print("RESULT| " + row, flush=True)


def run(label, argv, env=None, timeout=1800):
    print("\n" + "#" * 72)
    print(f"# {label}")
    print(f"# env+: {env or {}}")
    print("#" * 72, flush=True)
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{BUILD_DIR / 'bin'}:{e.get('LD_LIBRARY_PATH', '')}"
    if env:
        e.update({k: str(v) for k, v in env.items()})
    t0 = time.monotonic()
    try:
        with kh.build_heartbeat(label[:40], 60):
            p = subprocess.run(argv, env=e, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT", flush=True)
        return None, 0.0
    wall = time.monotonic() - t0
    for line in p.stderr.splitlines():
        if any(k in line for k in ("bench", "error", "FAIL", "CUDA", "detect")):
            print("  E| " + line, flush=True)
    sha = hashlib.sha256(p.stdout.encode()).hexdigest()[:12]
    nbytes = len(p.stdout.strip())
    ok = (p.returncode == 0 and nbytes > 0)
    note(f"{label}: rc={p.returncode} wall={wall:.1f}s bytes={nbytes} sha={sha} {'OK' if ok else 'FAIL'}")
    return (p.stdout if ok else None), wall


FOX_GT = "The quick brown fox jumps over the lazy dog. 12345"

dbnet_m = get("cstr/dbnet-ic15-GGUF", "dbnet-ic15-q8_0.gguf")
trocr_m = get("cstr/trocr-small-printed-GGUF", "trocr-small-printed-q8_0.gguf")

FIXTURES = (
    ("fox", str(EMBED_DIR / "tests/regression/images/fox.png"), FOX_GT),
    ("scan_page", str(EMBED_DIR / "tests/regression/images/scan_page_pd.png"), None),
)

if dbnet_m and trocr_m:
    for fx_name, img, gt in FIXTURES:
        kh.step(f"rt.{fx_name}")
        arms = {}
        # Interleaved 2 reps per arm: determinism per arm + cross-arm compare.
        for rep in range(2):
            for name, env in (("detcpu", {}), ("detcuda", {"OCR_DETECT_USE_GPU": "1"})):
                e = dict(env)
                e["OCR_DETECT_THREADS"] = "4"
                out, wall = run(f"rt.{fx_name}.{name}.rep{rep}",
                                [binp("crispembed"), "-m", trocr_m, "--ocr", img,
                                 "--ocr-det", dbnet_m, "--ocr-rec", trocr_m, "-t", "4"], env=e)
                if out is not None:
                    arms.setdefault(name, []).append(out)
        for name, outs in arms.items():
            det = all(o == outs[0] for o in outs)
            note(f"rt.{fx_name}.{name} deterministic across {len(outs)} reps: {det}")
        if "detcpu" in arms and "detcuda" in arms:
            a, b = arms["detcpu"][0], arms["detcuda"][0]
            note(f"rt.{fx_name} decoded text detcuda == detcpu: {a == b}")
            if a != b:
                al, bl = a.splitlines(), b.splitlines()
                note(f"rt.{fx_name} line counts: cpu={len(al)} cuda={len(bl)}")
                shown = 0
                for i in range(max(len(al), len(bl))):
                    x = al[i] if i < len(al) else "<missing>"
                    y = bl[i] if i < len(bl) else "<missing>"
                    if x != y:
                        note(f"rt.{fx_name} diff L{i}: cpu={x!r} cuda={y!r}")
                        shown += 1
                        if shown >= 8:
                            note(f"rt.{fx_name} (further diffs suppressed)")
                            break
                note(f"rt.{fx_name} arm-vs-arm CER: {cer(a, b):.4f}")
            if gt:
                note(f"rt.{fx_name} CER vs ground truth: cpu={cer(gt, a):.4f} cuda={cer(gt, b):.4f}")
else:
    note("MODEL DOWNLOAD FAILED — no evidence produced")

print("\n" + "=" * 72)
print(f"SUMMARY (GPU: {GPU_NAME})")
print("=" * 72)
for r in results:
    print(r)
print("done", flush=True)
