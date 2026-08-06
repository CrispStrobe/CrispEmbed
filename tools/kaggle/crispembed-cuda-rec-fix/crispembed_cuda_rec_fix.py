"""CUDA-rec 0-results fix proof (Fable task 1, branch fix/cuda-rec-zero-results).

The conv-ab v2 capture showed PP-OCRv6 rec on CUDA running its fused batch
graphs yet emitting ZERO results (boxes=38 results=0, rc=0). Root cause
(reproduced on M1 Metal with the same q8-head artifact — it was never
CUDA-specific): pp_graph_resident uploaded raw F32 bytes into a q8_0-typed
resident for the native-quant head path; the misread f16 scale bytes land on
Inf/NaN, every logit goes NaN, max_element's NaN-poisoned compare returns
index 0 = the CTC blank, and every crop decodes empty.

This kernel proves the fix WITH DECODED TEXT ON CUDA (LEARNING 35):
  arm A: q8-head rec, CUDA fused graph (default)      -> must yield results
  arm B: q8-head rec, CRISPEMBED_PPOCRV6_NO_GRAPH=1   -> scalar CPU reference
  arm C: f16 rec, CUDA fused graph                    -> no-regression control
PASS = A.results == B.results > 0 and A text ~= B text (>=0.95 similarity;
CUDA-F32 vs CPU-scalar reduction order may legitimately flip near-ties).
Everything lands in /kaggle/working/cudarecfix.log.
"""
import os
import sys
import subprocess
import hashlib
import difflib
from pathlib import Path

WORK = Path("/kaggle/working")
TEMP = Path("/kaggle/temp")
TEMP.mkdir(parents=True, exist_ok=True)
DL = Path("/tmp/crispembed-cudarecfix")
DL.mkdir(parents=True, exist_ok=True)

_LOG = open(WORK / "cudarecfix.log", "w", buffering=1)


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
BRANCH = os.environ.get("CRISPEMBED_BRANCH", "fix/cuda-rec-zero-results")
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
    kh.sh(f"cd {BUILD_DIR} && ninja -j{kh.safe_build_jobs(gpu=True)} crispembed-cli", check=True)


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
            if any(k in line for k in ("bench", "regions", "ggml_cuda", "backend", "error", "FAIL")):
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
rec_q8 = get("cstr/PP-OCRv6_medium_rec-GGUF", "PP-OCRv6_medium_rec-q8-head.gguf")
rec_f16 = get("cstr/PP-OCRv6_medium_rec-GGUF", "PP-OCRv6_medium_rec-f16.gguf")

results = []


def note(row):
    results.append(row)
    print("RESULT| " + row, flush=True)


def ocr_argv(img, rec):
    return [binp("crispembed"), "--ocr-pipeline", img, "--ocr-engine", "ppocrv6",
            "--ocr-det", det_m, "--ocr-rec", rec, "-t", "4"]


def region_count(stdout):
    for line in stdout.splitlines():
        if line.startswith("regions="):
            try:
                return int(line.split()[0].split("=")[1])
            except Exception:
                return -1
    return -1


def body(stdout):
    return "\n".join(stdout.splitlines()[1:])


kh.step("proof")
for img_name, img in (("strip", IMG_STRIP), ("page", IMG_PAGE)):
    a = run(f"A.{img_name} q8 CUDA graph", ocr_argv(img, rec_q8),
            env={"CRISPEMBED_PPOCRV6_BENCH": "1", "CRISPEMBED_PPOCRV6_GRAPH_BENCH": "1"})
    b = run(f"B.{img_name} q8 scalar reference", ocr_argv(img, rec_q8),
            env={"CRISPEMBED_PPOCRV6_NO_GRAPH": "1"})
    c = run(f"C.{img_name} f16 CUDA graph", ocr_argv(img, rec_f16))
    ra = region_count(a.stdout) if a else -1
    rb = region_count(b.stdout) if b else -1
    rc_ = region_count(c.stdout) if c else -1
    sim = difflib.SequenceMatcher(None, body(a.stdout), body(b.stdout)).ratio() if a and b else 0.0
    ok = a is not None and b is not None and ra == rb and ra > 0 and sim >= 0.95
    note(f"{img_name}: A(q8-graph)={ra} B(q8-scalar)={rb} C(f16-graph)={rc_} "
         f"A~B_sim={sim:.4f} byte_eq={a is not None and b is not None and body(a.stdout) == body(b.stdout)} "
         f"=> {'PASS' if ok else 'FAIL'}")
    if a:
        print(f"---- A.{img_name} full stdout ----\n{a.stdout}", flush=True)
    if b:
        print(f"---- B.{img_name} full stdout ----\n{b.stdout}", flush=True)

kh.step("summary")
print("\n" + "=" * 72)
print("SUMMARY")
print("=" * 72)
for r in results:
    print(r)
print("done", flush=True)
