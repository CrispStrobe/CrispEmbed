"""OvisOCR2 GPU confidence drift: per-stage precision bisect of CRISPEMBED_QWEN35_OCR_GPU (Kaggle T4 / P100).

Same build, model, documents and frozen CPU ground truth as tools/kaggle/ovisocr2-gpu-bench. Arms (all
three documents each, cap 1024):
  gpu        CRISPEMBED_QWEN35_OCR_GPU=1 (stored F16 matrices on the device)
  gpu-f32v   + CRISPEMBED_QWEN35_OCR_GPU_F32_VISION=1   (vision matrices widened to F32 on the device)
  gpu-f32d   + CRISPEMBED_QWEN35_OCR_GPU_F32_DECODER=1  (decoder matrices + head widened to F32)
  gpu-f32vd  both gates
  ggml-cpu   the same graphs on the CPU backend (CUDA hidden): the CPU-path drift level
  cpu-default  default CPU path, CUDA hidden (inventory-table only; timing reference)
Each row: IDs/text/stop vs the frozen receipt, max |conf - frozen|, max |conf - ggml-cpu| and the step
where the max occurs. Pusher substitutes __SHA__; the accelerator is chosen at push time.
Output: summary.json only (small).
"""
import hashlib, json, os, re, subprocess, sys, time, zipfile
from pathlib import Path

SHA = "__SHA__"
REPO = "https://github.com/CrispStrobe/CrispEmbed.git"
MODEL_REPO, MODEL_REV = "cstr/OvisOCR2-GGUF", "b7229e46f31bcf342d0999966bae8a03bf02bb07"
MODEL_FILE, MODEL_SHA = "ovisocr2-f16.gguf", "d3a733a5d5ba5d63f5275500fad4cff585b82e2fb90d1def2fe767d04edf57db"
PREPARED = ("https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/resolve/"
            "057e12668bf8109ec83c65929c9b8d6cb99ca566/ocr-quality-v1/prepared-38879605915/prepared.zip")
THREADS = 4
WORK, TMP = Path("/kaggle/working"), Path("/kaggle/temp")
TMP.mkdir(parents=True, exist_ok=True)
SRC = TMP / f"ce-{SHA[:12]}"
BUILD = SRC / "build-cuda"
STARTED = time.time()
summary = {"sha": SHA, "threads": THREADS, "runs": [], "steps": []}


def save():
    summary["elapsed_s"] = round(time.time() - STARTED, 1)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))


def sh(cmd):
    t = time.time()
    print("+", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    summary["steps"].append({"cmd": cmd[:240], "rc": r.returncode, "s": round(time.time() - t, 1)})
    save()
    if r.returncode:
        raise SystemExit(f"failed: {cmd}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def as_list(v):
    return json.loads(v) if isinstance(v, str) else v


try:
    summary["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader",
                                    shell=True, capture_output=True, text=True).stdout.strip()
    summary["cpu"] = subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)'", shell=True, capture_output=True,
                                    text=True).stdout
    sh(f"git clone -q {REPO} {SRC} && git -C {SRC} checkout -q {SHA} && "
       f"git -C {SRC} submodule update --init --recursive -q")
    sys.path.insert(0, str(SRC))
    import kaggle_harness as kh
    kh.install_build_toolchain()
    arch = kh.detect_cuda_arch()
    summary["cuda_arch"] = arch
    t = time.time()
    sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release " + " ".join(kh.cache_and_link_flags()) + " "
       + " ".join(kh.cuda_build_flags(arch)))
    sh(f"cmake --build {BUILD} --target test-ovisocr2-quality-run test-qwen35-profile -j{kh.safe_build_jobs(gpu=True)}")
    summary["build_s"] = round(time.time() - t, 1)
    summary["ccache"] = subprocess.run("ccache -s 2>&1 | head -12", shell=True, capture_output=True, text=True).stdout
    save()
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
    gpu = {"CRISPEMBED_QWEN35_OCR_GPU": "1", "CRISPEMBED_QWEN35_OCR_GPU_LOG": "1"}
    hide = {"CUDA_VISIBLE_DEVICES": ""}
    f32v, f32d = {"CRISPEMBED_QWEN35_OCR_GPU_F32_VISION": "1"}, {"CRISPEMBED_QWEN35_OCR_GPU_F32_DECODER": "1"}
    arms = [("ggml-cpu", dict(gpu, **hide), list(cases)), ("gpu", gpu, list(cases)),
            ("gpu-f32v", dict(gpu, **f32v), list(cases)), ("gpu-f32d", dict(gpu, **f32d), list(cases)),
            ("gpu-f32vd", dict(gpu, **f32v, **f32d), list(cases)), ("cpu-default", hide, ["inventory-table"])]
    ggml_cpu_conf = {}
    for arm, extra, ids in arms:
        for cid in ids:
            env = dict(base_env, **extra)
            t = time.time()
            r = subprocess.run([str(binary), model, str(cases[cid]), "1024", str(THREADS)], env=env,
                               capture_output=True, text=True)
            row = {"arm": arm, "case": cid, "rc": r.returncode, "wall_s": round(time.time() - t, 2),
                   "placement": [l for l in r.stderr.splitlines() if l.startswith("qwen35_ocr ggml-ops")],
                   "stderr_tail": r.stderr[-500:]}
            prof = {}
            for l in r.stdout.splitlines():
                m = re.match(r"PROFILE phase=(\S+) event=(\S+) calls=(\d+) nanoseconds=(\d+)", l)
                if m and m.group(2) in ("vision", "prefill", "cached_decode", "selection"):
                    prof[m.group(2)] = {"calls": int(m.group(3)), "s": int(m.group(4)) / 1e9}
            row["profile"] = prof
            line = next((l for l in r.stdout.splitlines() if l.startswith("OVIS_QUALITY_RESULT ")), None)
            if line:
                res = json.loads(line.split(" ", 1)[1])
                ref = frozen[cid]
                fids, fconf = as_list(ref["token_ids"]), as_list(ref["confidences"])
                ids_out, conf = res["token_ids"], res["confidences"]
                text = bytes.fromhex(res["raw_text_utf8_hex"]).decode("utf-8", "replace")
                row.update(tokens=len(ids_out), frozen_tokens=len(fids), ids_exact=ids_out == fids,
                           text_exact=text == ref["raw_text"], stop_kind=res["stop_kind"],
                           stop_exact=res["stop_kind"] == ref["stop_kind"],
                           recognize_s=res["wall_seconds"], init_s=res["init_seconds"],
                           conf_max_abs=max((abs(float(a) - float(b)) for a, b in zip(conf, fconf)), default=None),
                           mean_conf=sum(map(float, conf)) / len(conf) if conf else None,
                           frozen_mean_conf=ref["mean_confidence"],
                           first_id_divergence=next((i for i in range(min(len(ids_out), len(fids)))
                                                     if ids_out[i] != fids[i]), None))
                diffs = [abs(float(a) - float(b)) for a, b in zip(conf, fconf)]
                if diffs:
                    row["conf_max_step"] = diffs.index(max(diffs))
                    row["conf_mean_abs"] = sum(diffs) / len(diffs)
                if arm == "ggml-cpu":
                    ggml_cpu_conf[cid] = conf
                elif cid in ggml_cpu_conf:
                    d2 = [abs(float(a) - float(b)) for a, b in zip(conf, ggml_cpu_conf[cid])]
                    row["conf_max_abs_vs_ggml_cpu"] = max(d2) if d2 else None
                    row["conf_mean_abs_vs_ggml_cpu"] = sum(d2) / len(d2) if d2 else None
                if not row["text_exact"]:
                    row["text"] = text[:3000]
                dec = prof.get("cached_decode")
                if dec and dec["calls"]:
                    row["decode_ms_per_token"] = round(1000 * dec["s"] / dec["calls"], 2)
                    row["decode_tok_s"] = round(dec["calls"] / dec["s"], 2)
                if res.get("wall_seconds"):
                    row["end_to_end_tok_s"] = round(len(ids_out) / float(res["wall_seconds"]), 2)
            summary["runs"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("stderr_tail", "text")}), flush=True)
            save()
except BaseException as e:
    summary["error"] = repr(e)[:1000]
    raise
finally:
    save()
