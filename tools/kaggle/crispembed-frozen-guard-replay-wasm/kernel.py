"""Fresh replay of the WASM Q8 rounding A/B (run 40391544739) on an exact full-SHA clone.

Kaggle fallback for .github/workflows/frozen-guard-replay-wasm-q8.yml when the
hosted queue stalls. Same steps and pins (emsdk 6.0.2, node 22, playwright
1.56.0, same driver phases via tools/ci-heavy/frozen_guard_replay_wasm_q8.py).
One recorded deviation: the operator-control barrier uses receipts produced
earlier in this session (`--operator-receipts`) instead of a same-run GitHub
artifact, with the same operator_receipt checks.

GitHub artifacts come from a replay-only `gh` shim over an attached private
dataset of recorded `gh api` responses; no token is present. Pusher
substitutes __SHA__. Output: small JSON/log reports only.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path

SHA = "__SHA__"
REPO = "https://github.com/CrispStrobe/CrispEmbed.git"
WORK = Path("/kaggle/working")
TMP = Path("/kaggle/temp"); TMP.mkdir(parents=True, exist_ok=True)
SRC = TMP / "src"; OUT = TMP / "rounding-ab"; CTL = TMP / "rounding-controls"; SHIM = TMP / "shim"
STARTED = time.time(); log = []


def sh(cmd, check=True, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    t = time.time(); r = subprocess.run(cmd, shell=isinstance(cmd, str), **kw)
    log.append({"cmd": (cmd if isinstance(cmd, str) else " ".join(map(str, cmd)))[:300], "rc": r.returncode, "s": round(time.time() - t, 1)})
    if check and r.returncode:
        raise SystemExit("command failed: %s" % r.returncode)
    return r


def install_shim():
    caches = [p.parent for p in Path("/kaggle/input").rglob("index.tsv")]
    if not caches:
        raise SystemExit("recorded gh cache dataset not attached")
    SHIM.mkdir(parents=True, exist_ok=True)
    (SHIM / "gh").write_text('''#!/usr/bin/env python3
# Replay-only gh: serves recorded `gh api repos/CrispStrobe/CrispEmbed/<path>` responses.
import hashlib, shutil, sys
from pathlib import Path
CACHES = [Path(p) for p in %r]; PREFIX = "repos/CrispStrobe/CrispEmbed/"
args = sys.argv[1:]
if len(args) != 2 or args[0] != "api" or not args[1].startswith(PREFIX):
    sys.exit("gh shim: unsupported arguments %%r" %% args)
path = args[1][len(PREFIX):]
key = hashlib.sha256(path.encode()).hexdigest()[:32]
hits = [c / key for c in CACHES if (c / key).exists()]
with open("/kaggle/working/gh-shim.log", "a") as log:
    log.write(("hit " if hits else "MISS ") + path + "\\n")
if not hits:
    sys.exit("gh shim: unrecorded request " + path)
with hits[0].open("rb") as src:
    shutil.copyfileobj(src, sys.stdout.buffer, 1 << 20)
''' % [str(c) for c in caches])
    (SHIM / "gh").chmod(0o755)


install_shim()
summary = {"sha": SHA}
try:
    sh(["git", "clone", "-q", REPO, str(SRC)])
    sh(["git", "-C", str(SRC), "checkout", "-q", SHA])
    sh(["git", "-C", str(SRC), "submodule", "update", "--init", "--recursive", "-q"])
    # Node 22 and emsdk 6.0.2, as pinned by the hosted workflow.
    node = TMP / "node"
    sh("curl -fsSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz | tar -xJ -C %s && mv %s/node-v22.11.0-linux-x64 %s" % (TMP, TMP, node))
    sh(["git", "clone", "-q", "https://github.com/emscripten-core/emsdk.git", str(TMP / "emsdk")])
    sh("cd %s && ./emsdk install 6.0.2 >/dev/null && ./emsdk activate 6.0.2 >/dev/null" % (TMP / "emsdk"))
    sh([sys.executable, "-m", "pip", "install", "-q", "uv"])
    venv = TMP / "venv"
    sh([sys.executable, "-m", "uv", "venv", "-q", "--seed", "--python-preference", "only-managed", "--python", "3.12", str(venv)])
    py = str(venv / "bin" / "python")
    sh([sys.executable, "-m", "uv", "pip", "install", "-q", "--python", py, "wrapt", "cmake==3.31.6"])
    em = subprocess.run("bash -c 'source %s/emsdk_env.sh >/dev/null 2>&1 && env'" % (TMP / "emsdk"), shell=True, capture_output=True, text=True).stdout
    emenv = dict(line.split("=", 1) for line in em.splitlines() if "=" in line)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("HF_", "HUGGING", "KAGGLE_", "CRISPEMBED_", "GH_", "GITHUB_"))
           and k not in ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE")}
    env.update({k: emenv[k] for k in ("EMSDK", "EMSDK_NODE") if k in emenv})
    env.update(PATH=os.pathsep.join([str(SHIM), str(node / "bin"), str(venv / "bin"), emenv.get("PATH", env.get("PATH", ""))]),
               GITHUB_ACTIONS="true", GITHUB_SHA=SHA, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2",
               MKL_NUM_THREADS="2", CMAKE_BUILD_PARALLEL_LEVEL="2")
    run = lambda c, check=True, **kw: sh(c, check=check, cwd=SRC, env=env, **kw)
    # --- operator controls (same commands as successor-wasm-rounding-controls.yml) ---
    CTL.mkdir()
    r = run("python tests/test_wasm_q8_rounding_source.py -v > %s/frozen-source-test.log 2>&1" % CTL, check=False)
    summary["frozen_source_test_rc"] = r.returncode
    run([py, "-c", """
import hashlib,json,os,subprocess,sys
from pathlib import Path
spec=json.loads(Path('docs/wasm-q8-rounding-candidate.json').read_text())
actual=subprocess.check_output(['git','-C','ggml','rev-parse','HEAD'],text=True).strip()
assert actual==subprocess.check_output(['git','rev-parse','HEAD:ggml'],text=True).strip()
files={n:{'frozen':i['candidate_sha256'],'current':hashlib.sha256((Path('ggml')/n).read_bytes()).hexdigest()} for n,i in spec['files'].items()}
receipt={'schema':'crispembed-wasm-rounding-operator-source-v1','source':os.environ['GITHUB_SHA'],'ggml':actual,'declaration':spec,'model_access':False,'qualification':False,
 'replay':{'frozen_ggml':spec['ggml_candidate'],'dependency_files':files,'encoder_sha256':hashlib.sha256(Path('src/crispembed.cpp').read_bytes()).hexdigest(),
 'frozen_encoder_sha256':spec['crispembed_encoder_candidate_sha256']}}
Path(sys.argv[1],'source.json').write_text(json.dumps(receipt,indent=2)+'\\n')
""", str(CTL)])
    build = TMP / "rounding-build"
    run("emcmake cmake -S ggml/tests/wasm-q8-rounding -B %s -DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm -DCMAKE_BUILD_TYPE=Release "
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BLAS=OFF -DGGML_LLAMAFILE=OFF "
        "-DGGML_CUDA=OFF -DGGML_METAL=OFF -DGGML_VULKAN=OFF -DCMAKE_C_FLAGS='-msimd128 -pthread' -DCMAKE_CXX_FLAGS='-msimd128 -pthread' "
        "-DCMAKE_EXE_LINKER_FLAGS='-msimd128 -pthread -sPTHREAD_POOL_SIZE=2 -sENVIRONMENT=node -sALLOW_MEMORY_GROWTH=1 -sEXIT_RUNTIME=1' "
        "> %s/configure.log 2>&1" % (build, CTL))
    run("cmake --build %s --target test-wasm-q8-rounding -j2 > %s/build.log 2>&1" % (build, CTL))
    run("timeout 120s node %s/test-wasm-q8-rounding.js > %s/runtime.log 2>&1" % (build, CTL))
    run([py, "-c", """
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
rows=[json.loads(l) for l in (root/'runtime.log').read_text().splitlines() if l.startswith('{')]
assert len(rows)==5 and all(r['passed'] for r in rows)
assert rows[0]['control']=='converter'
assert {(r['type'],r['indirect']) for r in rows[1:]}=={(8,False),(8,True),(20,False),(20,True)}
assert all(r['legacy']==158 and r['nearest']==189 for r in rows[1:])
(root/'controls.json').write_text(json.dumps({'schema':'crispembed-wasm-rounding-operator-result-v1','passed':True,'controls':rows,'qualification':False,
 'limits':['Weight-free operator and backend-isolation controls only; no model inference, source-cause or default promotion.']},indent=2)+'\\n')
""", str(CTL)])
    receipts = TMP / "operator-receipts"; receipts.mkdir()
    for n in ("source.json", "controls.json"):
        shutil.copy2(CTL / n, receipts / n)
    # --- A/B (same phases as wasm-q8-rounding-ab-v1.yml) ---
    r = run("python tests/test_wasm_q8_rounding_ab_v1.py -v > %s/frozen-ab-test.log 2>&1" % TMP, check=False)
    summary["frozen_ab_test_rc"] = r.returncode
    replay = [py, "tools/ci-heavy/frozen_guard_replay_wasm_q8.py"]
    run(replay + ["preflight", "--output", str(OUT), "--operator-receipts", str(receipts)])
    run(replay + ["prepare", "--output", str(OUT)])
    tools = TMP / "browser-tools"
    run(["npm", "install", "--prefix", str(tools), "--no-save", "playwright@1.56.0"])
    run([str(tools / "node_modules/.bin/playwright"), "install", "--with-deps", "chromium"])
    benv = dict(env, NODE_PATH=str(tools / "node_modules"))
    run(replay + ["native-reference", "--output", str(OUT)])
    sh(["node", "tests/test_wasm_q8_rounding_ab_v1.cjs", str(OUT / "browser"), "reference"], cwd=SRC, env=benv)
    run(replay + ["freeze-old", "--output", str(OUT)])
    run(replay + ["build", "--output", str(OUT)])
    sh(["node", "tests/test_wasm_q8_rounding_ab_v1.cjs", str(OUT / "browser"), "candidate"], cwd=SRC, env=benv)
    r = run(replay + ["compare", "--output", str(OUT)], check=False)
    summary["compare_rc"] = r.returncode
    frozen = TMP / "frozen-ab"; frozen.mkdir()
    run("gh api repos/CrispStrobe/CrispEmbed/actions/artifacts/12692262412/zip > %s/review.zip && "
        "echo '6185d3a2b0776a157ab0dc10f25ddcc22406ba847dbb70b0b2638d43483bb047  %s/review.zip' | sha256sum -c && "
        "cd %s && python -c \"import zipfile;zipfile.ZipFile('review.zip').extractall('.',['ab.json','old-panel-freeze.json'])\"" % (frozen, frozen, frozen))
    r = run(replay + ["replay-compare", "--output", str(OUT), "--frozen", str(frozen / "ab.json")], check=False)
    summary["replay_compare_rc"] = r.returncode
finally:
    summary["log"] = log; summary["elapsed_s"] = round(time.time() - STARTED)
    dest = WORK / "out"; dest.mkdir(exist_ok=True)
    for base in (OUT, CTL):
        if base.exists():
            for p in base.glob("*"):
                if p.is_file() and p.suffix in (".json", ".log", ".txt") and p.stat().st_size <= 8 * 1024**2:
                    shutil.copy2(p, dest / (base.name + "-" + p.name))
    for p in (TMP / "frozen-ab-test.log",):
        if p.exists(): shutil.copy2(p, dest / p.name)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "log"}, default=str)[:2000])
