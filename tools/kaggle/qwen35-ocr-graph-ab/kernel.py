"""Qwen3.5-OCR (LightOnOCR-3-0.8B, qwen35_ocr engine) graph-decoder A/B on Kaggle CPU.

OvisOCR2 and LightOnOCR-3-0.8B share src/qwen35_ocr.cpp, so a default flip of
CRISPEMBED_QWEN35_OCR_GRAPH_DECODER / CRISPEMBED_QWEN35_VISION_THREADED_ATTN reaches
both. This kernel reconstructs the qualified LightOn F16 GGUF with the pinned frozen
converter (tools/ci-heavy/ovisocr2_lighton_converter_frozen.py, same vision-F32 recipe
as tests/results/qwen35-ocr/receipt.json; byte identity is checked and recorded) and
runs the three frozen realistic documents (prepared-38879605915, cap 1024, empty
prompt, as tests/results/qwen35-ocr/quality.json) through test-qwen35-profile at its
maximum cap of 512 tokens (the same cap in every arm; a capped page compares its prefix).

Ground truth is the old default path in the same kernel (graph decoder off, threaded
vision attention off). Arms:
  base       CRISPEMBED_QWEN35_OCR_GRAPH_DECODER=0, ..._VISION_THREADED_ATTN=0   all three
  vt         graph off, threaded vision on                                       inventory-table
  graph-nat  threaded on, graph decoder, GRAPH_MATMUL=native                     all three
  graph-f32  threaded on, graph decoder, GRAPH_MATMUL=f32                        all three
Each non-base row reports exact IDs/text/stop vs base and the confidence drift.
Pusher substitutes __SHA__. Output: summary.json only (small).
"""
import hashlib, json, os, re, subprocess, sys, time, zipfile
from pathlib import Path

SHA = "__SHA__"
REPO = "https://github.com/CrispStrobe/CrispEmbed.git"
PREPARED = ("https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/resolve/"
            "057e12668bf8109ec83c65929c9b8d6cb99ca566/ocr-quality-v1/prepared-38879605915/prepared.zip")
THREADS = int(os.environ.get("AB_THREADS", "4"))
CAP = 512  # test-qwen35-profile maximum; every arm uses the same cap
WORK, TMP = Path("/kaggle/working"), Path("/kaggle/temp")
TMP.mkdir(parents=True, exist_ok=True)
SRC, BUILD = TMP / "CrispEmbed", TMP / "CrispEmbed" / "build"
STARTED = time.time()
summary = {"sha": SHA, "threads": THREADS, "cap": CAP, "runs": [], "steps": []}


def save():
    summary["elapsed_s"] = round(time.time() - STARTED, 1)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))


def sh(cmd):
    t = time.time()
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    summary["steps"].append({"cmd": cmd[:200], "rc": r.returncode, "s": round(time.time() - t, 1)})
    save()
    if r.returncode:
        raise SystemExit(f"failed: {cmd}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(binary, model, image, env, nthreads):
    t = time.time()
    r = subprocess.run([str(binary), str(model), str(image), str(CAP), str(nthreads)], env=env,
                       capture_output=True, text=True)
    out = r.stdout.splitlines()
    bench = {l.split(":", 1)[0][6:]: l.split(":", 1)[1].split() for l in out
             if l.startswith(("BENCH ids:", "BENCH confidences:"))}
    hexline = next((l.split(":", 1)[1] for l in out if l.startswith("BENCH utf8_hex:")), "")
    stop = next((l for l in out if l.startswith("BENCH stop=")), "")
    prof = {}
    for l in out:
        m = re.match(r"PROFILE phase=(\S+) event=(\S+) calls=(\d+) nanoseconds=(\d+)", l)
        if m and m.group(2) in ("vision", "prefill", "cached_decode"):
            prof[m.group(2)] = {"calls": int(m.group(3)), "s": int(m.group(4)) / 1e9}
    row = {"rc": r.returncode, "wall_s": round(time.time() - t, 2), "stderr_tail": r.stderr[-400:],
           "ids": [int(x) for x in bench.get("ids", [])], "confidences": [float(x) for x in bench.get("confidences", [])],
           "text": bytes.fromhex(hexline.strip()).decode("utf-8", "replace") if hexline.strip() else "",
           "stop": re.search(r"stop=(\S+)", stop).group(1) if stop else None, "profile": prof}
    dec = prof.get("cached_decode")
    if dec and dec["calls"]:
        row["decode_ms_per_token"] = round(1000 * dec["s"] / dec["calls"], 1)
    return row


try:
    summary["cpu"] = subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)'", shell=True, capture_output=True,
                                    text=True).stdout
    sh(f"git clone -q {REPO} {SRC} && git -C {SRC} checkout -q {SHA} && "
       f"git -C {SRC} submodule update --init --recursive -q")
    sys.path.insert(0, str(SRC))
    import kaggle_harness as kh
    kh.install_build_toolchain()
    sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCRISPEMBED_NATIVE=OFF -DGGML_NATIVE=OFF "
       "-DGGML_BLAS=OFF -DGGML_METAL=OFF -DBUILD_SHARED_LIBS=OFF -DCRISPEMBED_BUILD_SHARED=OFF "
       + " ".join(kh.cache_and_link_flags()))
    sh(f"cmake --build {BUILD} --target test-qwen35-profile -j4")
    binary = BUILD / "test-qwen35-profile"

    # Reconstruct the qualified LightOn F16 GGUF with the pinned frozen converter.
    sh(f"{sys.executable} -m pip install -q gguf==0.19.0 safetensors==0.8.0 transformers==5.5.4 "
       "tokenizers==0.22.2 huggingface_hub")
    sys.path[:0] = [str(SRC / "tools"), str(SRC / "tools/ci-heavy")]
    import ovisocr2_lighton_fixture as fx
    from huggingface_hub import snapshot_download
    receipt = json.loads(fx.RECEIPT.read_text())
    converter = fx.frozen_converter()
    keep = fx.receipt_vision_recipe(receipt, converter)
    snap = Path(snapshot_download(fx.MODEL, revision=fx.REVISION, cache_dir=str(TMP / "hf"),
                                  allow_patterns=list(fx.SOURCE_FILES)))
    model = TMP / fx.GGUF_NAME
    t = time.time()
    converter.convert(snap, model, fx.MODEL, fx.REVISION, dtype="f16", keep_f32=keep)
    summary["gguf"] = {"sha256": sha256(model), "bytes": model.stat().st_size, "convert_s": round(time.time() - t, 1)}
    summary["gguf"]["receipt_identical"] = summary["gguf"]["sha256"] == fx.GGUF_SHA
    save()

    docs = TMP / "docs"
    docs.mkdir(exist_ok=True)
    sh(f"curl -sSL -o {TMP}/prepared.zip {PREPARED}")
    with zipfile.ZipFile(TMP / "prepared.zip") as z:
        z.extractall(docs)
    cases = {cid: docs / f"{cid}.png" for cid in ("inventory-table", "two-column-page", "historical-receipt")}
    summary["inputs"] = {cid: sha256(p) for cid, p in cases.items()}

    base_env = {k: v for k, v in os.environ.items() if not k.startswith("CRISPEMBED_")}
    base_env.update(CRISPEMBED_QWEN35_OCR="1", CRISPEMBED_QWEN35_PROFILE="1", OMP_NUM_THREADS=str(THREADS))
    arms = [("graph-nat", {"CRISPEMBED_QWEN35_OCR_GRAPH_DECODER": "1", "CRISPEMBED_QWEN35_VISION_THREADED_ATTN": "1",
                           "CRISPEMBED_QWEN35_GRAPH_MATMUL": "native"}, list(cases)),
            ("graph-f32", {"CRISPEMBED_QWEN35_OCR_GRAPH_DECODER": "1", "CRISPEMBED_QWEN35_VISION_THREADED_ATTN": "1",
                           "CRISPEMBED_QWEN35_GRAPH_MATMUL": "f32"}, list(cases)),
            ("base", {"CRISPEMBED_QWEN35_OCR_GRAPH_DECODER": "0", "CRISPEMBED_QWEN35_VISION_THREADED_ATTN": "0"},
             list(cases)),
            ("vt", {"CRISPEMBED_QWEN35_OCR_GRAPH_DECODER": "0", "CRISPEMBED_QWEN35_VISION_THREADED_ATTN": "1"},
             ["inventory-table"])]
    results = {}
    for arm, extra, ids in arms:
        for cid in ids:
            row = run(binary, model, cases[cid], dict(base_env, **extra), THREADS)
            row.update(arm=arm, case=cid, tokens=len(row["ids"]))
            results[(arm, cid)] = row
            summary["runs"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("ids", "confidences", "stderr_tail")}),
                  flush=True)
            save()
    for row in summary["runs"]:
        ref = results.get(("base", row["case"]))
        if row["arm"] == "base" or not ref:
            continue
        a, b = row["ids"], ref["ids"]
        row["vs_base"] = {"both_ran": row["rc"] == 0 and ref["rc"] == 0 and bool(a) and bool(b), "ids_exact": a == b, "text_exact": row["text"] == ref["text"], "stop_exact": row["stop"] == ref["stop"],
                          "first_id_divergence": next((i for i in range(min(len(a), len(b))) if a[i] != b[i]),
                                                      None if len(a) == len(b) else min(len(a), len(b))),
                          "conf_exact": row["confidences"] == ref["confidences"],
                          "conf_max_abs": max((abs(x - y) for x, y in zip(row["confidences"], ref["confidences"])),
                                              default=None)}
    for row in summary["runs"]:
        row.pop("confidences_full", None)
    save()
except BaseException as e:
    summary["error"] = repr(e)[:1000]
    raise
finally:
    save()
