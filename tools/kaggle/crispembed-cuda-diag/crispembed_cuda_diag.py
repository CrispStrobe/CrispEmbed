"""CrispEmbed CUDA diagnostic — the 4 remaining Turing/Pascal-only FAILs.

Builds CrispEmbed `main` with CUDA (ccache-warmed) and, on the actual T4/P100
box, exercises the existing env gates for glm-ocr / internvl2 / granite-vision /
layout-heron so we can localize each divergence WITHOUT that HW locally. It runs
the per-stage diff binaries (input is baked into each ref GGUF) plus FORCE_CPU /
*_SCALAR isolation runs, capturing FULL stdout+stderr for each. See
`PLAN.md → "CUDA regression — the 4 remaining FAILs: diagnosis & fix plan"`.

Everything downloadable ends up in /kaggle/working/diag.log (mirror of stdout).
Model + ref GGUFs stage under /tmp (never /kaggle/working — the ENOSPC gotcha).
"""
import os
import sys
import subprocess
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
DL = Path("/tmp/crispembed-diag")
DL.mkdir(parents=True, exist_ok=True)

_LOG = open(WORK / "diag.log", "w", buffering=1)


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
EMBED_DIR = WORK / "CrispEmbed"
BUILD_DIR = EMBED_DIR / "build"

# ── clone CrispASR for kaggle_harness (bundled fallback) ──
_CRISPASR = WORK / "CrispASR"
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
kh.resolve_hf_token()  # env → Kaggle Secret → dataset; cstr/* are public anyway

# ── clone + CUDA build (ccache-warmed) ──
kh.step("clone.crispembed")
if not EMBED_DIR.exists():
    kh.sh(f"git clone --depth 1 --recursive -b {BRANCH} {REPO_URL} {EMBED_DIR}")
BUILD_DIR.mkdir(exist_ok=True)

gpu = subprocess.run("nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader",
                     shell=True, capture_output=True, text=True)
print(f"GPU: {gpu.stdout.strip() or 'none'}", flush=True)
kh.sh("pip install -q huggingface_hub hf_transfer gguf safetensors pillow || true", check=False)
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
# only the four diff binaries + CLI we need (fast partial build)
kh.step("build")
targets = "test-layout-diff test-granite-vision-diff test-glm-ocr-diff test-internvl2-diff crispembed-cli"
kh.sh(f"cd {BUILD_DIR} && ninja -j{kh.safe_build_jobs(gpu=True)} {targets}", check=True)


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


def run(label, argv, env=None):
    """Run one diagnostic command, capture FULL stdout+stderr."""
    print("\n" + "#" * 72)
    print(f"# {label}")
    print(f"# cmd: {' '.join(argv)}   env+: {env or {}}")
    print("#" * 72, flush=True)
    e = dict(os.environ)
    e["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{BUILD_DIR / 'bin'}:{e.get('LD_LIBRARY_PATH','')}"
    if env:
        e.update({k: str(v) for k, v in env.items()})
    try:
        p = subprocess.run(argv, env=e, capture_output=True, text=True, timeout=600)
        sys.stdout.write(p.stdout)
        if p.stderr:
            print("---- stderr ----")
            sys.stdout.write(p.stderr)
        print(f"---- exit={p.returncode}"
              + (f"  (signal {-p.returncode})" if p.returncode < 0 else "") + " ----", flush=True)
    except Exception as ex:
        print(f"  RUN EXCEPTION: {ex}", flush=True)


LD = binp("test-layout-diff")
GD = binp("test-granite-vision-diff")
MD = binp("test-glm-ocr-diff")
CLI = binp("crispembed")

# ── download models + refs (input is baked into each ref) ──
kh.step("download")
lay = get("cstr/layout-heron-gguf", "layout-heron-f32.gguf")
layref = get("cstr/layout-heron-gguf", "layout-ref.gguf")
gv = get("cstr/granite-vision-crispembed-GGUF", "granite-vision-3.3-2b-q8_0.gguf")
gvref = get("cstr/granite-vision-crispembed-GGUF", "granite-vision-ref.gguf")
glm = get("cstr/glm-ocr-crispembed-GGUF", "glm-ocr-q8_0.gguf")
glmref = get("cstr/glm-ocr-crispembed-GGUF", "glm-ocr-ref-full.gguf")
iv = get("cstr/internvl2-1b-crispembed-GGUF", "internvl2-1b-q4_k.gguf")

# ── DIAGNOSTIC MATRIX ──────────────────────────────────────────────────
kh.step("diag.layout")
# (3) layout-heron: capture the full GGML_ASSERT (SIGABRT before output on CUDA),
#     then confirm CPU passes on the SAME box (isolates the CUDA backend).
if lay and layref:
    run("LAYOUT / CUDA (default) — capture the abort + full stderr", [LD, lay, layref])
    run("LAYOUT / CPU (LAYOUT_DETECT_FORCE_CPU=1) — should PASS",
        [LD, lay, layref], {"LAYOUT_DETECT_FORCE_CPU": 1})

kh.step("diag.granite")
# (2) granite: baseline projector cos on CUDA, then isolate — VIS_SCALAR runs the
#     projector in scalar CPU-math; if that PASSES the drift is in the ggml-CUDA
#     projector graph. Full CPU is the control.
if gv and gvref:
    run("GRANITE / CUDA (default) — baseline projector cos", [GD, gv, gvref])
    run("GRANITE / VIS_SCALAR (scalar vision+projector) — isolates the ggml graph",
        [GD, gv, gvref], {"CRISPEMBED_GRANITE_VIS_SCALAR": 1})
    run("GRANITE / full CPU (CRISPEMBED_GRANITE_CPU=1) — control, should PASS",
        [GD, gv, gvref], {"CRISPEMBED_GRANITE_CPU": 1})

kh.step("diag.glm")
# (1a) glm-ocr Class-B: per-stage vision diff on CUDA localizes WHICH stage
#      craters; CPU run is the freshness/control check for the ref.
if glm and glmref:
    run("GLM / CUDA (default) — per-stage vision cos (localize the crater)", [MD, glm, glmref])
    run("GLM / CPU (GLM_OCR_FORCE_CPU=1) — control (confirms ref fresh + CPU good)",
        [MD, glm, glmref], {"GLM_OCR_FORCE_CPU": 1})

kh.step("diag.internvl2")
# (1b) internvl2 Class-B: no ref uploaded → run the OCR discriminator instead.
#      A synthetic 'fox' image; CUDA vs INTERNVL2_OCR_FORCE_CPU. If CPU reads it
#      and CUDA is garbage, the CUDA vision backend is the cause.
if iv:
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new("RGB", (640, 96), "white")
        d = ImageDraw.Draw(img)
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        except Exception:
            f = ImageFont.load_default()
        d.text((16, 28), "The quick brown fox jumps over the lazy dog. 12345", fill="black", font=f)
        fox = str(DL / "fox.png")
        img.save(fox)
        run("INTERNVL2 / CUDA (default) --ocr", [CLI, "-m", iv, "--ocr", fox])
        run("INTERNVL2 / CPU (INTERNVL2_OCR_FORCE_CPU=1) --ocr — should read it",
            [CLI, "-m", iv, "--ocr", fox], {"INTERNVL2_OCR_FORCE_CPU": 1})
        # bonus: same discriminator for glm (text path)
        if glm:
            run("GLM / CUDA (default) --ocr", [CLI, "-m", glm, "--ocr", fox])
            run("GLM / CPU (GLM_OCR_FORCE_CPU=1) --ocr", [CLI, "-m", glm, "--ocr", fox],
                {"GLM_OCR_FORCE_CPU": 1})
    except Exception as ex:
        print(f"  internvl2 discriminator skipped: {ex}", flush=True)

print("\n=== DIAG DONE — full transcript in /kaggle/working/diag.log ===", flush=True)
