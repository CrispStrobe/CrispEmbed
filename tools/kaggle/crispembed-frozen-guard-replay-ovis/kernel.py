"""Fresh OvisOCR2 final-integrated replay (frozen 14-arm or eleven-case F16 quality).

Mirrors .github/workflows/ovisocr2-final-integrated-v2.yml for one gate and
one phase, on an exact full-SHA clone, through
tools/ci-heavy/frozen_guard_replay_ovis.py (records instead of requiring the
frozen source identity; every numerical/decoded check unchanged).

GitHub Actions artifacts are served by a replay-only `gh` shim from an
attached private dataset of previously recorded `gh api` responses (index.tsv
+ files named by a hash of the API path). The shim never contacts GitHub and
fails on any unrecorded request; no token is present in this kernel.

Pusher substitutes __SHA__, __MODE__ (frozen|quality), __GATE__ (0|1) and
__DATASET__. Output: small reports only (no raw vectors, no models).
"""
import json, os, shutil, subprocess, sys, tarfile, time
from pathlib import Path

SHA = "__SHA__"
MODE = "__MODE__"
GATE = "__GATE__"
DATASET = "__DATASET__"
REPO = "https://github.com/CrispStrobe/CrispEmbed.git"
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
SRC = TMP / "src"; OUT = TMP / "out"; SCRATCH = TMP / "scratch"; BIN = TMP / "bin"; SHIM = TMP / "shim"
STARTED = time.time()
log = []


def sh(cmd, check=True, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    t = time.time()
    r = subprocess.run(cmd, shell=isinstance(cmd, str), **kw)
    log.append({"cmd": cmd if isinstance(cmd, str) else " ".join(map(str, cmd))[:300], "rc": r.returncode, "s": round(time.time() - t, 1)})
    if check and r.returncode:
        raise SystemExit("command failed: %s" % r.returncode)
    return r


def find_cache():
    for p in Path("/kaggle/input").rglob("index.tsv"):
        return p.parent
    raise SystemExit("recorded gh cache dataset not attached: " + DATASET)


def install_shim(cache):
    SHIM.mkdir(parents=True, exist_ok=True)
    (SHIM / "gh").write_text('''#!/usr/bin/env python3
# Replay-only gh: serves recorded `gh api repos/CrispStrobe/CrispEmbed/<path>` responses.
import hashlib, shutil, sys
from pathlib import Path
CACHE = Path(%r); PREFIX = "repos/CrispStrobe/CrispEmbed/"
args = [a for a in sys.argv[1:]]
if not args or args[0] != "api":
    sys.exit("gh shim: only `gh api` is supported, got %%r" %% args)
paths = [a for a in args[1:] if not a.startswith("-")]
if len(paths) != 1 or len(args) != 2 or not paths[0].startswith(PREFIX):
    sys.exit("gh shim: unsupported arguments %%r" %% args)
path = paths[0][len(PREFIX):]
key = hashlib.sha256(path.encode()).hexdigest()[:32]
f = CACHE / key
with open("/kaggle/working/gh-shim.log", "a") as log:
    log.write(("hit " if f.exists() else "MISS ") + path + "\\n")
if not f.exists():
    sys.exit("gh shim: unrecorded request " + path)
with f.open("rb") as src:
    shutil.copyfileobj(src, sys.stdout.buffer, 1 << 20)
''' % str(cache))
    (SHIM / "gh").chmod(0o755)


cache = find_cache()
install_shim(cache)
summary = {"sha": SHA, "mode": MODE, "gate": int(GATE), "cache_entries": sum(1 for _ in open(cache / "index.tsv"))}
try:
    sh(["git", "clone", "-q", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "-q", SHA])
    sh(["git", "-C", str(SRC), "submodule", "update", "--init", "--recursive", "-q"])
    for d in (OUT, SCRATCH, BIN):
        d.mkdir(parents=True, exist_ok=True)
    # Managed CPython 3.12 venv, as in the hosted workflow (setup-python 3.12).
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    venv = TMP / "venv"
    sh([sys.executable, "-m", "uv", "venv", "-q", "--seed", "--python-preference", "only-managed", "--python", "3.12", str(venv)])
    py = str(venv / "bin" / "python")
    pip = [sys.executable, "-m", "uv", "pip", "install", "-q", "--python", py]
    sh(pip + ["wrapt", "cmake==3.31.6"])
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HF_", "HUGGING", "KAGGLE_", "CRISPEMBED_", "GH_", "GITHUB_"))
           and k not in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE")}
    env.update(PATH=os.pathsep.join([str(SHIM), str(venv / "bin"), env.get("PATH", "")]),
               GITHUB_ACTIONS="true", GITHUB_SHA=SHA, HEAVY_OUT=str(OUT), HEAVY_SCRATCH=str(SCRATCH),
               AUDIT_BIN_DIR=str(BIN), HF_HOME=str(SCRATCH / "hf"),
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    replay = [py, "tools/ci-heavy/frozen_guard_replay_ovis.py"]
    run = lambda c, check=True, e=None: sh(c, check=check, cwd=SRC, env=e or env)
    run(replay + ["--phase", "preflight"])
    if MODE == "frozen":
        cpu = subprocess.run("lscpu | grep -E 'Model name|Flags' | cut -c1-400", shell=True, capture_output=True, text=True).stdout
        summary["cpu"] = cpu
        build = TMP / "build"
        run(["cmake", "-S", ".", "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DCRISPEMBED_NATIVE=OFF", "-DGGML_NATIVE=OFF",
             "-DGGML_BLAS=OFF", "-DGGML_METAL=OFF", "-DBUILD_SHARED_LIBS=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF"])
        run([py, "scripts/check-cpu-baseline.py", str(build)])
        names = ["test-qwen35-decoder", "test-qwen35-vision", "test-qwen35-delta", "test-ovisocr2-context",
                 "test-ovisocr2-quality-run", "test-ovisocr2-image-bounds"]
        run(["cmake", "--build", str(build), "--target", *names, "-j4"])
        import hashlib
        for n in names:
            shutil.copy2(build / n, BIN / n)
        record = {"source_commit": SHA, "protocol_sha256": hashlib.sha256((SRC / "docs/ovisocr2-final-integrated-v2.json").read_bytes()).hexdigest(),
                  "portable_cpu_baseline_checked": True,
                  "binary_sha256": {n: hashlib.sha256((BIN / n).read_bytes()).hexdigest() for n in names}}
        (BIN / "binaries.json").write_text(json.dumps(record, indent=2) + "\n")
        sh(pip + ["numpy==2.2.6", "pillow==12.1.0", "gguf==0.19.0", "huggingface-hub==1.33.0"])
        r = run(replay + ["--phase", "frozen", "--gate", GATE], check=False)
        summary["frozen_rc"] = r.returncode
    else:
        qenv = dict(env, CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION=GATE, CRISPEMBED_QWEN35_WEIGHT_CACHE="0",
                    CRISPEMBED_QWEN35_PARALLEL_F16="0", CRISPEMBED_QWEN35_PRECISE_RMS="0", CRISPEMBED_QWEN35_IMATRIX="0",
                    CRISPEMBED_QWEN35_PROFILE="0", CRISPEMBED_WASM_Q8_0_ROUND="0")
        run([py, "tools/ci-heavy/ovisocr2_quality_run.py", "--phase", "f16", "--original-run", "40054043629", "--prepare-documents"], e=qenv)
        sh(pip + ["torch==2.11.0", "torchvision==0.26.0", "--index-url", "https://download.pytorch.org/whl/cpu"])
        sh(pip + ["transformers==5.16.1", "tokenizers==0.23.3", "numpy==2.2.6", "pillow==12.1.0", "huggingface-hub==1.33.0",
                  "gguf==0.19.0", "safetensors==0.8.0"])
        r = run([py, "tools/ci-heavy/ovisocr2_quality_run.py", "--phase", "f16", "--original-run", "40054043629",
                 "--native-document-timeout", "3600"], check=False, e=qenv)
        summary["quality_run_rc"] = r.returncode
        r = run(replay + ["--phase", "quality-check", "--gate", GATE], check=False, e=qenv)
        summary["quality_check_rc"] = r.returncode
finally:
    summary["log"] = log
    summary["elapsed_s"] = round(time.time() - STARTED)
    # Keep outputs small: JSON/logs from the out dir only (raw vectors stay in scratch).
    dest = WORK / "out"; dest.mkdir(exist_ok=True)
    if OUT.exists():
        for p in OUT.rglob("*"):
            if p.is_file() and p.suffix in (".json", ".log", ".txt") and p.stat().st_size <= 8 * 1024**2:
                t = dest / p.relative_to(OUT); t.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, t)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "log"}, default=str)[:2000])
