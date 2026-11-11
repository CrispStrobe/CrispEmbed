"""OvisOCR2 CPU speed A/B against the frozen first-F16 receipt (Kaggle CPU).

Builds test-ovisocr2-quality-run at an exact full SHA with the portable flags of
the frozen replay, downloads the published F16 GGUF (pinned revision + SHA-256)
and the three reviewed realistic documents (pinned prepared.zip), then runs each
arm and compares every token ID and printed confidence with
tests/results/qwen35-ocr/ovisocr2-f16-quality.json (native_cases.f16), which
is the bit-exact ground truth of the default path at vision gate 0.

Arms (env on top of the Ovis gates; CRISPEMBED_QWEN35_PROFILE=1 for stage times):
  base       default path                                  inventory-table only
  vt         CRISPEMBED_QWEN35_VISION_THREADED_ATTN=1        inventory-table only
  graph-f32  vt + CRISPEMBED_QWEN35_OCR_GRAPH_DECODER=1 (GRAPH_MATMUL=f32)   all three
  graph-nat  vt + graph decoder, GRAPH_MATMUL=native                         all three
Pusher substitutes __SHA__. Output: summary.json only (small).
"""
import hashlib, json, os, re, shutil, subprocess, sys, time, zipfile
from pathlib import Path

SHA = "__SHA__"
REPO = "https://github.com/CrispStrobe/CrispEmbed.git"
MODEL_REPO, MODEL_REV = "cstr/OvisOCR2-GGUF", "b7229e46f31bcf342d0999966bae8a03bf02bb07"
MODEL_FILE, MODEL_SHA = "ovisocr2-f16.gguf", "d3a733a5d5ba5d63f5275500fad4cff585b82e2fb90d1def2fe767d04edf57db"
PREPARED = ("https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/resolve/"
            "057e12668bf8109ec83c65929c9b8d6cb99ca566/ocr-quality-v1/prepared-38879605915/prepared.zip")
THREADS = int(os.environ.get("AB_THREADS", "4"))
WORK, TMP = Path("/kaggle/working"), Path("/kaggle/temp")
TMP.mkdir(parents=True, exist_ok=True)
SRC, BUILD = TMP / "CrispEmbed", TMP / "CrispEmbed" / "build"
STARTED = time.time()
summary = {"sha": SHA, "threads": THREADS, "runs": [], "steps": []}


def save():
    summary["elapsed_s"] = round(time.time() - STARTED, 1)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))


def sh(cmd, **kw):
    t = time.time()
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, **kw)
    summary["steps"].append({"cmd": cmd[:200], "rc": r.returncode, "s": round(time.time() - t, 1)})
    save()
    if r.returncode:
        raise SystemExit(f"failed: {cmd}")
    return r


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


try:
    summary["cpu"] = subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)'", shell=True, capture_output=True,
                                    text=True).stdout
    sh(f"git clone -q {REPO} {SRC} && git -C {SRC} checkout -q {SHA} && "
       f"git -C {SRC} submodule update --init --recursive -q")
    sys.path.insert(0, str(SRC))
    import kaggle_harness as kh  # the pinned clone's harness (ccache settings under test)
    kh.install_build_toolchain()
    t = time.time()
    sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPEMBED_NATIVE=OFF -DGGML_NATIVE=OFF "
       "-DGGML_BLAS=OFF -DGGML_METAL=OFF -DBUILD_SHARED_LIBS=OFF -DCRISPEMBED_BUILD_SHARED=OFF "
       + " ".join(kh.cache_and_link_flags()))
    sh(f"cmake --build {BUILD} --target test-ovisocr2-quality-run -j4")
    summary["build_s"] = round(time.time() - t, 1)
    summary["ccache"] = subprocess.run("ccache -s 2>&1 | head -20", shell=True, capture_output=True, text=True).stdout
    binary = BUILD / "test-ovisocr2-quality-run"

    sh(f"{sys.executable} -m pip install -q huggingface_hub")
    from huggingface_hub import hf_hub_download
    model = hf_hub_download(MODEL_REPO, MODEL_FILE, revision=MODEL_REV, cache_dir=str(TMP / "hf"))
    if sha256(model) != MODEL_SHA:
        raise SystemExit("model SHA-256 differs")
    docs = TMP / "docs"
    docs.mkdir(exist_ok=True)
    sh(f"curl -sSL -o {TMP}/prepared.zip {PREPARED}")
    with zipfile.ZipFile(TMP / "prepared.zip") as z:
        z.extractall(docs)
    frozen = json.loads((SRC / "tests/results/qwen35-ocr/ovisocr2-f16-quality.json").read_text())
    frozen = frozen["producer"]["native_cases"]["f16"]
    cases = {}
    for cid in ("inventory-table", "two-column-page", "historical-receipt"):
        png = docs / f"{cid}.png"
        if sha256(png) != frozen[cid]["input_png_sha256"]:
            raise SystemExit(f"{cid} PNG differs from the frozen input")
        cases[cid] = png
    base_env = {k: v for k, v in os.environ.items() if not k.startswith("CRISPEMBED_")}
    base_env.update(CRISPEMBED_QWEN35_OCR="1", CRISPEMBED_OVISOCR2_FRONTEND="1", CRISPEMBED_OVISOCR2_INFERENCE="1",
                    CRISPEMBED_QWEN35_PROFILE="1", OMP_NUM_THREADS=str(THREADS))
    vt = {"CRISPEMBED_QWEN35_VISION_THREADED_ATTN": "1"}
    graph = dict(vt, CRISPEMBED_QWEN35_OCR_GRAPH_DECODER="1")
    arms = [("graph-f32", dict(graph, CRISPEMBED_QWEN35_GRAPH_MATMUL="f32"), list(cases)),
            ("vt", vt, ["inventory-table"]),
            ("base", {}, ["inventory-table"]),
            ("graph-nat", dict(graph, CRISPEMBED_QWEN35_GRAPH_MATMUL="native"), list(cases))]
    only = os.environ.get("AB_ARMS")
    for arm, extra, ids in arms:
        if only and arm not in only.split(","):
            continue
        for cid in ids:
            env = dict(base_env, **extra)
            t = time.time()
            r = subprocess.run(["/usr/bin/time", "-f", "MAXRSS_KIB=%M", str(binary), model, str(cases[cid]), "1024",
                                str(THREADS)], env=env, capture_output=True, text=True, timeout=6 * 3600)
            row = {"arm": arm, "case": cid, "rc": r.returncode, "wall_s": round(time.time() - t, 2),
                   "stderr_tail": r.stderr[-600:]}
            m = re.search(r"MAXRSS_KIB=(\d+)", r.stderr)
            row["max_rss_kib"] = int(m.group(1)) if m else None
            line = next((l for l in r.stdout.splitlines() if l.startswith("OVIS_QUALITY_RESULT ")), None)
            prof = {}
            for l in r.stdout.splitlines():
                m = re.match(r"PROFILE phase=(\S+) event=(\S+) calls=(\d+) nanoseconds=(\d+)", l)
                if m and m.group(2) in ("vision", "prefill", "cached_decode", "selection", "request_file"):
                    prof[m.group(2)] = {"calls": int(m.group(3)), "s": int(m.group(4)) / 1e9}
            row["profile"] = prof
            if line:
                res = json.loads(line.split(" ", 1)[1])
                ref = frozen[cid]
                fids = json.loads(ref["token_ids"]) if isinstance(ref["token_ids"], str) else ref["token_ids"]
                fconf = json.loads(ref["confidences"]) if isinstance(ref["confidences"], str) else ref["confidences"]
                ids_out, conf = res["token_ids"], res["confidences"]
                row.update(tokens=len(ids_out), ids_exact=ids_out == fids, text_exact=
                           bytes.fromhex(res["raw_text_utf8_hex"]).decode("utf-8", "replace") == ref["raw_text"],
                           stop_kind=res["stop_kind"], recognize_s=res["wall_seconds"], init_s=res["init_seconds"])
                common = min(len(conf), len(fconf))
                row["conf_max_abs"] = max((abs(float(a) - float(b)) for a, b in zip(conf, fconf)), default=None)
                row["conf_exact"] = len(conf) == len(fconf) and all(float(a) == float(b) for a, b in zip(conf, fconf))
                first = next((i for i in range(common) if ids_out[i] != fids[i]), None)
                row["first_id_divergence"] = first
                dec = prof.get("cached_decode")
                if dec and dec["calls"]:
                    row["decode_ms_per_token"] = round(1000 * dec["s"] / dec["calls"], 1)
                    row["decode_tok_s"] = round(dec["calls"] / dec["s"], 2)
                row["frozen_hosted_wall_s"] = float(ref["wall_seconds"])
            summary["runs"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "stderr_tail"}), flush=True)
            save()
except BaseException as e:
    summary["error"] = repr(e)[:1000]
    raise
finally:
    save()
