"""CrispEmbed conv A/B v3 — O9d det-only DBNet CPU-vs-CUDA (box-level compare)
+ the LAYOUT_CONV_F16 tensor-core arm (wants a T4 draw).

Why v3 (PERFORMANCE.md 2026-08-06, board row): the v2 whole-pipeline DBNet
wall was TrOCR-rec-dominated (114.3 vs 111.7 s), so det was never isolated —
and the two arms' decoded TEXT differed because a det backend change moves
boxes. This version runs detection ONLY (`test-ocr-detect`) and compares at
the box level: count, per-box coordinate delta vs the CPU arm, score delta,
prob-map stats. A timing row without its matching box list is a FAIL, never
a win (proof-of-work rule).

The LAYOUT_CONV_F16 arm re-runs from v2 unchanged: it LOSES on M1 Metal
(2.2 vs 1.4 s) AND on P100 (region drift 20->19); the only open question is
tensor-core hardware. GPU IS A DRAW — if this log says P100 again, the T4
question stays open (delete-then-push for a re-draw when quota allows).

Everything lands in /kaggle/working/convab.log. Models stage under /tmp.
The ccache line MUST read "ccache warmed from" — a cold-build line means the
chr1s4/crispembed-ccache seed regressed again (re-run crispembed-ccache-seed).
"""
import os
import sys
import re
import subprocess
import hashlib
import time
from pathlib import Path

WORK = Path("/kaggle/working")
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
GPU_NAME = gpu.stdout.strip() or "none"
print(f"GPU: {GPU_NAME}", flush=True)
print(f"T4-DRAW: {'YES - tensor-core arm is decisive' if 'T4' in GPU_NAME else 'NO - P100/other, LAYOUT_CONV_F16 stays open'}",
      flush=True)
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
        # Kaggle auto-extracts a dataset's ccache.tar into a bare .ccache/
        # tree, which the harness warmer (install_build_toolchain) already
        # consumed — v3's local "ccache: cold build" line here was a FALSE
        # alarm over exactly that (829 files warm, 2.3 min build).
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
    kh.sh(f"cd {BUILD_DIR} && ninja -j{kh.safe_build_jobs(gpu=True)} "
          "crispembed-cli test-ocr-detect", check=True)
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
            if any(k in line for k in ("bench", "elapsed", "Phase", "ggml_cuda", "CUDA",
                                       "backend", "error", "FAIL")):
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
IMG_FOX = str(EMBED_DIR / "tests/regression/images/fox.png")

results = []


def note(row):
    results.append(row)
    print("RESULT| " + row, flush=True)


BOX_RE = re.compile(r"\[(\d+)\] \(([-\d.]+), ([-\d.]+)\)-\(([-\d.]+), ([-\d.]+)\) score=([\d.]+)")
COUNT_RE = re.compile(r"Detected (\d+) text regions")
PROB_RE = re.compile(r"Prob map: (\d+)x(\d+), range \[([-\d.]+), ([-\d.]+)\], (\d+) pixels")


def parse_boxes(stdout):
    boxes = [tuple(float(g) for g in m.groups()[1:]) for m in BOX_RE.finditer(stdout)]
    count = COUNT_RE.search(stdout)
    prob = PROB_RE.search(stdout)
    return (int(count.group(1)) if count else -1), boxes, (prob.groups() if prob else None)


def compare_boxes(ref, cand):
    """Box-level compare vs the CPU reference arm: greedy nearest match by
    center distance; report count delta, max coordinate delta over matched
    pairs, and unmatched counts. A det backend that moves geometry shows up
    here directly instead of through a recognizer's text."""
    if len(ref) == 0 and len(cand) == 0:
        return "both-empty"
    used = set()
    max_d = 0.0
    max_s = 0.0
    unmatched = 0
    for r in ref:
        rcx, rcy = (r[0] + r[2]) / 2, (r[1] + r[3]) / 2
        best, bd = None, 1e18
        for j, c in enumerate(cand):
            if j in used:
                continue
            ccx, ccy = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
            d = (rcx - ccx) ** 2 + (rcy - ccy) ** 2
            if d < bd:
                bd, best = d, j
        if best is None:
            unmatched += 1
            continue
        used.add(best)
        c = cand[best]
        max_d = max(max_d, max(abs(r[k] - c[k]) for k in range(4)))
        max_s = max(max_s, abs(r[4] - c[4]))
    return (f"count {len(ref)}->{len(cand)} max_coord_delta={max_d:.1f}px "
            f"max_score_delta={max_s:.3f} unmatched_ref={unmatched} "
            f"extra_cand={len(cand) - len(used)}")


# ── O9d v3: det-only DBNet CPU vs CUDA, 3 interleaved reps, box-level ──
kh.step("o9d3.dbnet-det-only")
dbnet_m = get("cstr/dbnet-ic15-GGUF", "dbnet-ic15-q8_0.gguf")
if dbnet_m:
    for img_name, img in (("scan_page", IMG_PAGE), ("fox", IMG_FOX)):
        arm_out = {}
        for rep in range(3):
            for name, env in (("cpu", {"OCR_DETECT_FORCE_CPU": "1"}),
                              ("cuda", {"OCR_DETECT_USE_GPU": "1"})):
                e = dict(env)
                e["CRISPEMBED_OCR_DETECT_BENCH"] = "1"
                t0 = time.monotonic()
                p = run(f"o9d3 {img_name} {name} rep{rep}",
                        [binp("test-ocr-detect"), dbnet_m, img], env=e)
                wall = time.monotonic() - t0
                if p and p.returncode == 0:
                    count, boxes, prob = parse_boxes(p.stdout)
                    if count < 0 or (count > 0 and not boxes):
                        note(f"o9d3.{img_name}.{name} rep{rep}: PARSE-FAIL (no box lines) — not evidence")
                        continue
                    arm_out.setdefault(name, []).append((wall, count, boxes, prob))
                    note(f"o9d3.{img_name}.{name} rep{rep}: wall={wall:.2f}s boxes={count} "
                         f"prob={prob}")
                else:
                    note(f"o9d3.{img_name}.{name} rep{rep}: rc={p.returncode if p else 'none'} FAIL")
        if "cpu" in arm_out and "cuda" in arm_out:
            cw = [r[0] for r in arm_out["cpu"]]
            gw = [r[0] for r in arm_out["cuda"]]
            note(f"o9d3.{img_name} walls: cpu={['%.2f' % w for w in cw]} "
                 f"cuda={['%.2f' % w for w in gw]}")
            note(f"o9d3.{img_name} box-compare (cpu ref vs cuda): "
                 + compare_boxes(arm_out["cpu"][0][2], arm_out["cuda"][0][2]))
            cpu_stable = all(r[2] == arm_out["cpu"][0][2] for r in arm_out["cpu"])
            cuda_stable = all(r[2] == arm_out["cuda"][0][2] for r in arm_out["cuda"])
            note(f"o9d3.{img_name} intra-arm determinism: cpu={cpu_stable} cuda={cuda_stable}")

# ── LAYOUT_CONV_F16 tensor-core arm (decisive only on T4) ──
kh.step("layout-conv-f16")
layout_m = get("cstr/layout-heron-gguf", "layout-heron-f32.gguf")
ltexts = {}
if layout_m:
    for name, env in (("f32", {}), ("f16", {"LAYOUT_CONV_F16": "1"})):
        e = dict(env)
        e["CRISPEMBED_LAYOUT_DETECT_BENCH"] = "1"
        e["CRISPEMBED_LAYOUT_REPEAT"] = "2"
        p = run(f"layout {name}",
                [binp("crispembed"), "-m", layout_m, "--layout", IMG_PAGE, "-t", "4"], env=e)
        if p:
            ltexts[name] = p.stdout
            for l in (x for x in p.stderr.splitlines() if "Phase 1" in x):
                note(f"layout.{name} [{GPU_NAME}]: {l.strip()}")
    if "f32" in ltexts and "f16" in ltexts:
        note(f"layout regions identical on {GPU_NAME}: {ltexts['f32'] == ltexts['f16']}")
        if ltexts["f32"] != ltexts["f16"]:
            a, b = ltexts["f32"].splitlines(), ltexts["f16"].splitlines()
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
