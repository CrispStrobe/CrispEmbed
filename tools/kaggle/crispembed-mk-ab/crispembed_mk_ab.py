"""R6 micro-kernel x86 A/B (Fable task 3, branch perf/r6-gemm-microkernel).

M1 verdict is in (CPU time -34% on the medium scalar det, decoded text
byte-identical). This kernel answers the x86 side on the small-private-L2
CPU where the plain interchange already won 13% nt=1 / 1.76x nt=4:

  0. unit gate: test-core-cpu-ops on AVX2 — the mk arm's FIRST run on x86
     (the AVX2 conv2d_mk_block8 was authored on an ARM box);
  1. arms on the PP-OCRv6 medium scalar det page, CPU-time verdicts:
     legacy / gemm nt=1 / gemm nt=4 / mk nt=1 / mk nt=4;
  2. decoded text cross-compared over all arms (proof-of-work: a timing row
     without matching output is a FAIL, never a win).

Everything lands in /kaggle/working/mkab.log.
"""
import os
import sys
import subprocess
import hashlib
import resource
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)
DL = Path("/tmp/crispembed-mkab")
DL.mkdir(parents=True, exist_ok=True)

_LOG = open(WORK / "mkab.log", "w", buffering=1)


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
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "perf/r6-gemm-microkernel")
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

cpu = subprocess.run("lscpu | grep -E 'Model name|L2|L3|avx2'", shell=True, capture_output=True, text=True)
print(f"CPU:\n{cpu.stdout}", flush=True)
flags = subprocess.run("grep -o 'avx2\\|fma' /proc/cpuinfo | sort -u", shell=True, capture_output=True, text=True)
print(f"isa: {flags.stdout.strip()}", flush=True)
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

# CPU-only build: the arms under test are the scalar/GEMM/MK CPU conv paths.
kh.step("cmake.configure")
kh.sh(f"cd {BUILD_DIR} && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release "
      + " ".join(kh.cache_and_link_flags()) + " ..", check=True)
kh.step("build")
with kh.build_heartbeat("ninja", 30):
    kh.sh(f"cd {BUILD_DIR} && ninja -j{kh.safe_build_jobs(gpu=False)} crispembed-cli test-core-cpu-ops", check=True)


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


results = []


def note(row):
    results.append(row)
    print("RESULT| " + row, flush=True)


kh.step("unit.gate")
p = subprocess.run([binp("test-core-cpu-ops")], capture_output=True, text=True, timeout=600)
tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "(no output)"
note(f"unit.cpu-ops(AVX2): rc={p.returncode} {tail}")
if p.returncode != 0:
    print(p.stdout[-3000:], flush=True)
    print("UNIT GATE FAILED — timing arms skipped", flush=True)
    sys.exit(1)

IMG = str(EMBED_DIR / "tests/regression/images/scan_page_pd.png")
det_m = get("cstr/PP-OCRv6-medium-det-GGUF", "PP-OCRv6_medium_det-f16.gguf")
rec_m = get("cstr/PP-OCRv6_medium_rec-GGUF", "PP-OCRv6_medium_rec-f16.gguf")

ARMS = [
    ("legacy", {}),
    ("gemm-nt1", {"CRISPEMBED_CONV2D_GEMM": "1", "CRISPEMBED_CONV2D_THREADS": "1"}),
    ("gemm-nt4", {"CRISPEMBED_CONV2D_GEMM": "1", "CRISPEMBED_CONV2D_THREADS": "4"}),
    ("mk-nt1", {"CRISPEMBED_CONV2D_MK": "1", "CRISPEMBED_CONV2D_THREADS": "1"}),
    ("mk-nt4", {"CRISPEMBED_CONV2D_MK": "1", "CRISPEMBED_CONV2D_THREADS": "4"}),
]

kh.step("arms")
outputs = {}
for rnd in range(3):
    for name, env in ARMS:
        e = dict(os.environ)
        e.update({"CRISPEMBED_PPOCRV6_DET_SCALAR": "1", "CRISPEMBED_PPOCRV6_DET_GPU": "0"})
        e.update(env)
        before = resource.getrusage(resource.RUSAGE_CHILDREN)
        p = subprocess.run([binp("crispembed"), "--ocr-pipeline", IMG, "--ocr-engine", "ppocrv6",
                            "--ocr-det", det_m, "--ocr-rec", rec_m, "-t", "4"],
                           env=e, capture_output=True, text=True, timeout=1800)
        after = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu_s = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
        sha = hashlib.sha256(p.stdout.encode()).hexdigest()[:12]
        outputs.setdefault(name, sha)
        ok = p.returncode == 0 and len(p.stdout) > 100
        note(f"r{rnd} {name}: cpu={cpu_s:.2f}s rc={p.returncode} stdout_sha={sha} "
             f"{'OK' if ok else 'FAIL'}")

kh.step("verdict")
shas = {n: s for n, s in outputs.items()}
base = shas.get("legacy")
for n, s in shas.items():
    note(f"output {n}: {s} {'== legacy' if s == base else '!= legacy  <-- INSPECT'}")

print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
for r in results:
    print(r)
print("done", flush=True)
