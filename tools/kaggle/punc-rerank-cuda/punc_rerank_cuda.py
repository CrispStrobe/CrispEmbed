#!/usr/bin/env python3
"""Three things that cannot be settled on the VPS, on one clean CUDA box.

1. THE SIBLING BUILD (CrispEmbed issue #50). Every earlier kernel clones
   CrispASR somewhere that is deliberately NOT a sibling of CrispEmbed, with a
   comment saying the sibling layout fails — "cost a run once". That was true;
   it is not any more. This kernel clones CrispASR AS A SIBLING on purpose, so
   the four shared libraries (crisp_audio / crisp_punc / crisp_lid /
   crisp_truecase) are compiled against CrispEmbed's src/core, on a fresh
   machine that has never seen either repo. If the fix is real this build
   succeeds and the workaround can be deleted from the other kernels.

2. fireredpunc PARITY ON CUDA, and imatrix vs plain q4_k. The port was just
   fixed against the Python blueprint — it appended a [SEP] the blueprint never
   emits, worth f16 cos_min 0.931090 -> 1.000000 — but that was measured on CPU.
   The dev guide is explicit that "correct on CPU AND Metal" is not sufficient,
   because CUDA has stricter per-op contiguity asserts, so a graph change must
   be re-run on real CUDA before it is trusted there. The 9 KB reference is
   checked into the repo, so this needs no torch and no 407 MB checkpoint.
   It also measures fireredpunc-q4_k-imatrix.gguf — the artifact the model
   registry actually downloads by default — against ground truth for the first
   time. Plain q4_k scores cos_min 0.935078 / 118-119 preds on the VPS; whether
   the imatrix build is better has never been checked against the blueprint,
   only against itself.

3. RERANK LATENCY ON A QUIET BOX (issue #51 follow-up). The VPS is shared with
   other agents and ran at load 13-25 throughout; interleaved medians there had
   a 2.4x within-arm spread, which is larger than any between-arm difference, so
   it could rule out a large regression and nothing more. A Kaggle box is idle,
   which is the only way to get a number worth recording. Both routes, warmed,
   interleaved, median of N, absolute ms — and the ordering-equality check that
   keeps the sigmoid honest.

Proof-of-work discipline (dev guide 4a): every timed call is checked for a
non-zero exit and a non-empty result before it is allowed to contribute a
number, because a crash that exits in 0.5 s otherwise mints a fake speedup.
"""

import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

WORK = Path("/kaggle/working")
# /tmp is the big layer (~70 GB); /kaggle/working is ~20 GB and holds outputs.
SCRATCH = Path("/tmp/punc_rerank")
SCRATCH.mkdir(parents=True, exist_ok=True)

EMBED_URL = "https://github.com/CrispStrobe/CrispEmbed.git"
EMBED_BRANCH = "main"
CRISPASR_URL = "https://github.com/CrispStrobe/CrispASR.git"

# ⚠ DELIBERATE SIBLING LAYOUT — this is item 1, not an oversight. CrispEmbed's
# CMakeLists defaults CRISP_*_DIR to ../CrispASR/crisp_*, so putting them side
# by side is what activates the code path under test.
EMBED = SCRATCH / "CrispEmbed"
CRISPASR = SCRATCH / "CrispASR"
BUILD = EMBED / "build"

PROGRESS = WORK / "progress.txt"
RESULTS = WORK / "punc_rerank_cuda_results.json"

PUNC_REPO = "cstr/fireredpunc-GGUF"
PUNC_FILES = [
    ("f16", "fireredpunc.gguf", 0.99),
    ("q8_0", "fireredpunc-q8_0.gguf", 0.99),
    ("q4_k", "fireredpunc-q4_k.gguf", 0.90),
    ("q4_k-imatrix", "fireredpunc-q4_k-imatrix.gguf", 0.90),
    ("iq4_xs", "fireredpunc-iq4_xs.gguf", 0.90),
]
RERANK_REPO = "cstr/ettin-reranker-150m-v1-GGUF"
RERANK_FILE = "ettin-reranker-150m-v1-q8_0.gguf"

RERANK_QUERY = "What is the capital of France?"
RERANK_DOCS = [
    "Bananas are a good source of potassium.",
    "Paris is the capital and most populous city of France.",
    "The Rust compiler enforces memory safety without a garbage collector.",
]
RELEVANT_INDEX = 1
ROUNDS = 9

results = {"status": "RUNNING"}
T0 = time.time()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a") as f:
        f.write(line + "\n")


def sh(cmd, check=True, capture=False, env=None, cwd=None):
    log(f"$ {cmd}")
    if capture:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           env=env, cwd=cwd)
        if check and r.returncode != 0:
            log(f"  rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            raise RuntimeError(cmd)
        return r
    r = subprocess.run(cmd, shell=True, env=env, cwd=cwd)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd} -> {r.returncode}")
    return r


def save():
    results["elapsed_s"] = round(time.time() - T0, 1)
    RESULTS.write_text(json.dumps(results, indent=2))


# ── harness ──────────────────────────────────────────────────────────────
log("=== CrispEmbed: sibling build + fireredpunc CUDA parity + rerank latency ===")
if not CRISPASR.exists():
    sh(f"git clone --depth 1 {CRISPASR_URL} {CRISPASR}")
sys.path.insert(0, str(CRISPASR / "tools" / "kaggle"))
try:
    import kaggle_harness as kh
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kaggle_harness as kh
kh.init_progress()
HF_TOKEN = kh.resolve_hf_token(require=True)
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

if not EMBED.exists():
    sh(f"git clone --depth 1 --recursive --shallow-submodules -b {EMBED_BRANCH} "
       f"{EMBED_URL} {EMBED}")
results["crispembed_commit"] = sh(f"git -C {EMBED} rev-parse HEAD", capture=True).stdout.strip()
results["crispasr_commit"] = sh(f"git -C {CRISPASR} rev-parse HEAD", capture=True).stdout.strip()
log(f"CrispEmbed @ {results['crispembed_commit']}")
log(f"CrispASR   @ {results['crispasr_commit']} (SIBLING of CrispEmbed, on purpose)")
results["gpu"] = sh("nvidia-smi --query-gpu=name --format=csv,noheader",
                    capture=True, check=False).stdout.strip()
log(f"GPU: {results['gpu']}")
save()

# ── 1. the sibling build ─────────────────────────────────────────────────
kh.install_build_toolchain()
arch = kh.detect_cuda_arch()
flags = kh.cuda_build_flags(arch) + kh.cache_and_link_flags()
# NOTE: no -DCRISP_*_DIR overrides. The defaults point at ../CrispASR/crisp_*,
# which is exactly what we want exercised.
BUILD.mkdir(exist_ok=True)
cfg = sh(f"cmake -S {EMBED} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release "
         + " ".join(flags), capture=True)
picked = re.findall(r"linking (crisp_\w+) from", cfg.stdout)
results["siblings_picked_up"] = picked
log(f"sibling libraries picked up: {picked}")
# A silent "not found — disabled" would make the whole point of this kernel
# evaporate while still going green, so fail loudly instead.
assert len(picked) == 4, f"expected 4 sibling libs, cmake reported {picked}"

with kh.build_heartbeat("cmake.build"):
    kh.sh_with_progress(
        f"stdbuf -oL -eL cmake --build {BUILD} "
        f"--target crispembed crispembed-server firered-punct-ab "
        f"-j{kh.safe_build_jobs(gpu=True)}")


def find_bin(name):
    for p in (BUILD / name, BUILD / "bin" / name):
        if p.exists():
            return p
    raise FileNotFoundError(name)


CRISPEMBED = find_bin("crispembed")
SERVER = find_bin("crispembed-server")
PUNCT_AB = find_bin("firered-punct-ab")
results["sibling_build"] = "OK"
log(f"sibling build OK: {CRISPEMBED.name}, {SERVER.name}, {PUNCT_AB.name}")
save()

# ── models ───────────────────────────────────────────────────────────────
sh("pip install -q huggingface_hub", check=False)
from huggingface_hub import hf_hub_download  # noqa: E402

MODELS = SCRATCH / "models"
MODELS.mkdir(exist_ok=True)
punc_paths = {}
for tag, fname, _floor in PUNC_FILES:
    try:
        punc_paths[tag] = Path(hf_hub_download(PUNC_REPO, fname,
                                               local_dir=str(MODELS), token=HF_TOKEN))
        log(f"got {fname}")
    except Exception as e:
        log(f"SKIP {fname}: {type(e).__name__}")
RERANK_GGUF = Path(hf_hub_download(RERANK_REPO, RERANK_FILE,
                                   local_dir=str(MODELS), token=HF_TOKEN))
log(f"got {RERANK_FILE}")

# ── 2. fireredpunc parity vs the blueprint, CPU and CUDA ─────────────────
REF = EMBED / "tests/regression/fireredpunc/blueprint_ref.txt"
PARITY = EMBED / "tests/firered_punc_parity.py"
results["punc_parity"] = {}

if not REF.exists() or not PARITY.exists():
    log(f"SKIP parity: reference/harness missing at {REF}")
else:
    # Backend probe. The parity script swallows the engine's stderr, so run the
    # binary once directly per arm and keep the load lines. Without this, a CUDA
    # arm that silently fell back to CPU would produce identical numbers to the
    # CPU arm and read as "CUDA is perfect" — the exact failure mode the dev
    # guide warns about when a gate mis-fires.
    probe_corpus = SCRATCH / "probe.txt"
    probe_corpus.write_text("hello world this is a test\n")
    results["backend_probe"] = {}
    for device in ("cpu", "gpu"):
        pr = subprocess.run([str(PUNCT_AB), str(punc_paths.get("f16", RERANK_GGUF)),
                             str(probe_corpus)],
                            capture_output=True, text=True,
                            env=dict(os.environ, FIREREDPUNC_BACKEND=device))
        tail = (pr.stderr or "")[-1200:]
        results["backend_probe"][device] = {"rc": pr.returncode, "stderr": tail}
        log(f"--- backend probe {device} (rc={pr.returncode}) ---\n{tail}")
    save()

    for tag, _fname, floor in PUNC_FILES:
        if tag not in punc_paths:
            continue
        for device in ("cpu", "cuda"):
            # FIREREDPUNC_BACKEND is the engine's own gate (read from
            # fireredpunc.cpp, not guessed): "cpu" forces ggml_backend_cpu_init,
            # "gpu" forces crispasr_init_gpu_backend. Forcing BOTH explicitly
            # rather than letting one arm take the default means the CUDA arm is
            # provably on CUDA — an arm that silently fell back to CPU would
            # otherwise "pass" and prove nothing.
            env = dict(os.environ, FIREREDPUNC_BACKEND=("cpu" if device == "cpu" else "gpu"))
            r = subprocess.run(
                [sys.executable, str(PARITY), str(BUILD), str(punc_paths[tag]),
                 str(REF), str(floor)],
                capture_output=True, text=True, env=env, cwd=str(EMBED))
            out = r.stdout
            log(f"--- fireredpunc {tag} / {device} (floor {floor}) ---\n{out.strip()}")
            m_cos = re.search(r"cos_min ([0-9.]+), max_abs ([0-9.]+)", out)
            m_pred = re.search(r"preds\s+: (\d+)/(\d+)", out)
            m_ids = re.search(r"token ids\s+: (\d+)/(\d+)", out)
            results["punc_parity"][f"{tag}/{device}"] = {
                "rc": r.returncode,
                "verdict": "PASS" if "PASS" in out else ("FAIL" if "FAIL" in out else "?"),
                "cos_min": float(m_cos.group(1)) if m_cos else None,
                "max_abs": float(m_cos.group(2)) if m_cos else None,
                "preds": f"{m_pred.group(1)}/{m_pred.group(2)}" if m_pred else None,
                "token_ids": f"{m_ids.group(1)}/{m_ids.group(2)}" if m_ids else None,
                # Kept for BOTH outcomes, not just failures: on the CUDA arm
                # this is where a "found 1 CUDA devices" line (or its absence)
                # shows whether the GPU was actually used.
                "stderr_tail": (r.stderr or "")[-600:],
            }
            save()

# ── 3. rerank latency on an idle box ─────────────────────────────────────
PORT = 8137
BASE = f"http://127.0.0.1:{PORT}"


def post(path, payload, timeout=300):
    req = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, (time.perf_counter() - t) * 1000.0


log("starting crispembed-server for the rerank A/B...")
srv_log = open(SCRATCH / "server.log", "w")
srv = subprocess.Popen([str(SERVER), "-m", str(RERANK_GGUF),
                        "--host", "127.0.0.1", "--port", str(PORT)],
                       stdout=srv_log, stderr=subprocess.STDOUT)
try:
    deadline = time.time() + 300
    ready = False
    while time.time() < deadline:
        if srv.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            urllib.request.urlopen(BASE + "/health", timeout=3).read()
            ready = True
            break
        except Exception:
            time.sleep(2)
    if not ready:
        raise RuntimeError("server not ready in time")
    health = json.loads(urllib.request.urlopen(BASE + "/health", timeout=5).read())
    log(f"server ready: {health}")
    assert health.get("reranker"), "loaded model is not a reranker"

    payload = {"query": RERANK_QUERY, "documents": RERANK_DOCS}
    # Warm-up, discarded: the first call pays classifier-weight caching, and a
    # cold shape would otherwise fake a collapse in round 1.
    post("/rerank", payload)
    post("/v1/rerank", payload)

    t_native, t_v1 = [], []
    for i in range(ROUNDS):
        b1, ms1 = post("/rerank", payload)
        b2, ms2 = post("/v1/rerank", dict(payload, model="ettin"))
        # Proof of work: an empty or mis-ranked result must not be timed.
        assert len(b1["results"]) == len(RERANK_DOCS), b1
        assert len(b2["results"]) == len(RERANK_DOCS), b2
        assert b1["results"][0]["index"] == RELEVANT_INDEX, b1
        assert b2["results"][0]["index"] == RELEVANT_INDEX, b2
        t_native.append(ms1)
        t_v1.append(ms2)
        log(f"  round {i+1}: /rerank {ms1:8.1f} ms   /v1/rerank {ms2:8.1f} ms")

    native_order = [r["index"] for r in b1["results"]]
    v1_order = [r["index"] for r in b2["results"]]
    v1_scores = [r["relevance_score"] for r in b2["results"]]
    native_scores = [r["score"] for r in b1["results"]]

    def summarise(a):
        return {"median_ms": round(statistics.median(a), 1),
                "min_ms": round(min(a), 1), "max_ms": round(max(a), 1),
                "spread": round(max(a) / min(a), 3),
                "stdev_pct": round(100.0 * statistics.pstdev(a) / statistics.mean(a), 2)}

    results["rerank"] = {
        "rounds": ROUNDS,
        "native": summarise(t_native),
        "v1": summarise(t_v1),
        "median_ratio_v1_over_native": round(
            statistics.median(t_v1) / statistics.median(t_native), 4),
        "native_order": native_order,
        "v1_order": v1_order,
        "orders_agree": native_order == v1_order,
        "native_scores": native_scores,
        "v1_relevance_scores": v1_scores,
        "sigmoid_matches": [
            round(abs(1.0 / (1.0 + pow(2.718281828459045, -s)) - v), 9)
            for s, v in zip(native_scores, v1_scores)],
    }
    log(f"rerank: native {results['rerank']['native']}")
    log(f"rerank: v1     {results['rerank']['v1']}")
    log(f"orders agree: {results['rerank']['orders_agree']}")
finally:
    srv.terminate()
    try:
        srv.wait(timeout=15)
    except subprocess.TimeoutExpired:
        srv.kill()
    srv_log.close()

# ── verdict ──────────────────────────────────────────────────────────────
fails = []
if results.get("sibling_build") != "OK":
    fails.append("sibling build")
for k, v in results.get("punc_parity", {}).items():
    if v["verdict"] != "PASS":
        fails.append(f"punc {k}")
if not results.get("rerank", {}).get("orders_agree", False):
    fails.append("rerank ordering")
results["status"] = "OK" if not fails else "FAIL"
results["failures"] = fails
save()
log(f"=== {results['status']} === failures={fails}")

# Refresh the ccache dataset payload (gotcha #17: MUST come from a real Kaggle
# build, and MUST be refreshed after every successful one or the next run gets a
# 100% miss and a ~25 min build).
try:
    kh.export_ccache_tar()
    log("ccache.tar exported for the dataset refresh")
except Exception as e:
    log(f"ccache export skipped: {type(e).__name__}")
