"""OCR engine GPU readiness: same three documents on Kaggle CPU and Kaggle GPU (CUDA).

Engines (published GGUFs at pinned revisions; LightOnOCR-3-1B converted here from the pinned original):
  lighton2-1b   cstr/lightonocr-GGUF lightonocr-1b-q4_k.gguf          (registry alias `lightonocr`)
  lighton3-1b   lightonai/LightOnOCR-3-1B @23d54408 -> F16 (CRISPEMBED_LIGHTONOCR3=1)
  deepseek-ocr2 cstr/deepseek-ocr2-crispembed-GGUF q4_k-stacked       (alias `deepseek-ocr2`)
  uocr2         cstr/Unlimited-OCR-2-GGUF F16 (alias `unlimited-ocr2`), FAST decoder + TILES
  minicpm-ocr   cstr/MiniCPM-OCR-1.0-GGUF F16 (alias `minicpm-ocr`), prompt "Text Recognition:" (reference)
LightOn4B is not run: the public loader rejects its experimental formats (docs/lighton4b-production-fence.md).

Documents: inventory-table, two-column-page, historical-receipt from the pinned prepared.zip used by
tests/results/qwen35-ocr/ovisocr2-graph-speed-ab.json (PNG SHA-256 checked against the Ovis frozen receipt).

DEVICE (rendered): `cpu` = portable CPU build, CPU kernel.  `gpu` = CUDA build (ccache seed attached);
arms `cuda` (default backend) then `cpu` (same binary, `crispembed_set_gpu_backend("cpu")` and
CUDA_VISIBLE_DEVICES="") while the time budget lasts. A small driver (written below, linked against the static
library with the CLI's own ninja compile/link lines) prints text, per-token confidences and init/recognize
wall time; engine stage lines (bench/timing env) and generated IDs are taken from stderr. On the GPU a
`sched` probe per engine (inventory-table, 8 tokens, GGML_SCHED_DEBUG=2) counts graph nodes per backend
and the ops left on CPU; nvidia-smi is sampled during every run (mean utilisation, peak memory).
Pusher substitutes __SHA__ and __DEVICE__. Output: summary.json only (small).
"""
import hashlib, json, os, re, subprocess, sys, threading, time, zipfile
from collections import Counter
from pathlib import Path

SHA, DEVICE = "__SHA__", "__DEVICE__"
REPO = "https://github.com/CrispStrobe/CrispEmbed.git"
PREPARED = ("https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/resolve/"
            "057e12668bf8109ec83c65929c9b8d6cb99ca566/ocr-quality-v1/prepared-38879605915/prepared.zip")
CASES = ("inventory-table", "two-column-page", "historical-receipt")
MAX_NEW, THREADS = 1024, 4
BUDGET_S = float(os.environ.get("AB_BUDGET_S", str(10.5 * 3600)))
WORK, TMP = Path("/kaggle/working"), Path("/kaggle/temp")
TMP.mkdir(parents=True, exist_ok=True)
SRC, BUILD = TMP / "CrispEmbed", TMP / "CrispEmbed" / "build"
STARTED = time.time()
HF = "https://huggingface.co/{}/resolve/{}/{}"
ENGINES = [  # name, (repo, revision, file) or None, env, prompt
    ("lighton2-1b", ("cstr/lightonocr-GGUF", "f7c097777d5e971df3813e6b28a4ab5647553ffd", "lightonocr-1b-q4_k.gguf"),
     {"CRISPEMBED_LIGHTONOCR_BENCH": "1"}, ""),
    ("lighton3-1b", None, {"CRISPEMBED_LIGHTONOCR3": "1", "CRISPEMBED_LIGHTONOCR_BENCH": "1"}, ""),
    ("deepseek-ocr2", ("cstr/deepseek-ocr2-crispembed-GGUF", "731d9377ea2a2f32b663267130ded422e8fe06f2",
                       "deepseek-ocr2-q4_k-stacked.gguf"),
     {"CRISPEMBED_DEEPSEEK_OCR2_BENCH": "1", "DS_DBG": "1"}, ""),
    ("uocr2", ("cstr/Unlimited-OCR-2-GGUF", "e3af0c0c4bfa241c161db64ca36f3c63d3ca8378", "unlimited-ocr-2-f16.gguf"),
     {"CRISPEMBED_UNLIMITED_OCR2": "1", "CRISPEMBED_UNLIMITED_OCR2_FAST": "1", "CRISPEMBED_UNLIMITED_OCR2_TILES": "1",
      "UOCR_DBG": "1", "CRISPEMBED_UNLIMITED_OCR_BENCH": "1"}, ""),
    ("uocr2-fused", ("cstr/Unlimited-OCR-2-GGUF", "e3af0c0c4bfa241c161db64ca36f3c63d3ca8378", "unlimited-ocr-2-f16.gguf"),
     {"CRISPEMBED_UNLIMITED_OCR2": "1", "CRISPEMBED_UNLIMITED_OCR2_FAST": "1", "CRISPEMBED_UNLIMITED_OCR2_TILES": "1",
      "UOCR_DBG": "1", "CRISPEMBED_UNLIMITED_OCR_BENCH": "1", "CRISPEMBED_UNLIMITED_OCR2_FUSED_STEP": "1"}, ""),
    ("minicpm-ocr", ("cstr/MiniCPM-OCR-1.0-GGUF", "6705b0652744a880e79c8ec856fac73ce56e1105", "minicpm-ocr-1.0-f16.gguf"),
     {"CRISPEMBED_MINICPM_OCR": "1", "CRISPEMBED_QWEN35_OCR": "1", "CRISPEMBED_MINICPM_OCR_TIMING": "1"},
     "Text Recognition:"),
]
ONLY = [e for e in os.environ.get("AB_ENGINES", "__ENGINES__".strip("_") if "__ENGINES__"[0] != "_" else "").split(",") if e]
summary = {"schema": "crispembed-ocr-gpu-readiness-run-v1", "sha": SHA, "device": DEVICE, "threads": THREADS,
           "max_new": MAX_NEW, "models": {}, "runs": [], "sched": {}, "steps": []}

DRIVER = r'''
#include "crispembed.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
// argv: model rgb_file width height max_tokens threads backend(auto|cpu|cuda) [prompt]
int main(int argc, char ** argv) {
    if (argc < 8) return 64;
    const int w = atoi(argv[3]), h = atoi(argv[4]);
    if (strcmp(argv[7], "auto") != 0) crispembed_set_gpu_backend(argv[7]);
    std::vector<unsigned char> px((size_t) w * h * 3);
    FILE * f = fopen(argv[2], "rb");
    if (!f || fread(px.data(), 1, px.size(), f) != px.size()) return 65;
    fclose(f);
    auto t0 = std::chrono::steady_clock::now();
    void * c = crispembed_ocr_model_init(argv[1], atoi(argv[6]));
    auto t1 = std::chrono::steady_clock::now();
    if (!c) { printf("AB_INIT_FAIL\n"); return 2; }
    crispembed_ocr_model_set_max_tokens(c, atoi(argv[5]));
    if (argc > 8 && argv[8][0]) printf("AB_PROMPT_ACCEPTED %d\n", crispembed_ocr_model_set_prompt(c, argv[8]));
    int n = 0;
    const char * txt = crispembed_ocr_model_recognize(c, px.data(), w, h, 3, &n);
    auto t2 = std::chrono::steady_clock::now();
    auto s = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    printf("AB_INIT_S %.4f\nAB_RECOG_S %.4f\nAB_TEXT_HEX ", s(t0, t1), s(t1, t2));
    for (int i = 0; txt && i < n; ++i) printf("%02x", (unsigned char) txt[i]);
    int nt = 0;
    const float * cf = crispembed_ocr_model_confidences(c, &nt);
    printf("\nAB_CONF %d", nt);
    for (int i = 0; cf && i < nt; ++i) printf(" %.9g", cf[i]);
    printf("\n");
    fflush(stdout);
    crispembed_ocr_model_free(c);
    return 0;
}
'''


def el():
    return time.time() - STARTED


def save():
    summary["elapsed_s"] = round(el(), 1)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))


def sh(cmd, check=True, **kw):
    t = time.time()
    print("+", cmd[:300], flush=True)
    r = subprocess.run(cmd, shell=True, **kw)
    summary["steps"].append({"cmd": cmd[:160], "rc": r.returncode, "s": round(time.time() - t, 1)})
    save()
    if check and r.returncode:
        raise SystemExit(f"failed: {cmd[:200]}")
    return r


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


class GpuMon:
    def __init__(self):
        self.proc, self.out = None, TMP / "smi.csv"

    def __enter__(self):
        if DEVICE == "gpu":
            self.f = open(self.out, "w")
            self.proc = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                          "--format=csv,noheader,nounits", "-lms", "500"], stdout=self.f,
                                         stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *a):
        self.stats = {}
        if self.proc:
            self.proc.terminate(); self.proc.wait(); self.f.close()
            rows = [l.split(",") for l in self.out.read_text().splitlines() if l.count(",") >= 1]
            util = [float(r[0]) for r in rows if r[0].strip().replace(".", "").isdigit()]
            mem = [float(r[1]) for r in rows if r[1].strip().replace(".", "").isdigit()]
            if util:
                self.stats = {"gpu_util_mean": round(sum(util) / len(util), 1), "gpu_util_samples": len(util),
                              "gpu_busy_frac": round(sum(u > 5 for u in util) / len(util), 3),
                              "gpu_mem_peak_mib": max(mem) if mem else None}


STAGE_PREFIXES = ("[lightonocr-bench]", "[deepseek-ocr2-stage-bench]", "[unlimited_ocr-bench]", "[uocr2-fast] prefill done",
                  "[uocr2-fast] decode done", "[decode] total=", "[time] minicpm_ocr", "lightonocr:", "deepseek_ocr2:",
                  "unlimited_ocr", "minicpm_ocr:", "ggml_cuda_init", "Device 0")


def run_case(binary, model, rgb, w, h, env, prompt, backend, max_new, stderr_path):
    cmd = [str(binary), model, str(rgb), str(w), str(h), str(max_new), str(THREADS), backend] + ([prompt] if prompt else [])
    t = time.time()
    with open(TMP / "o.txt", "w") as fo, open(stderr_path, "w") as fe, GpuMon() as mon:
        proc = subprocess.Popen(cmd, env=env, stdout=fo, stderr=fe)
        _, status, usage = os.wait4(proc.pid, 0)
    row = {"rc": os.waitstatus_to_exitcode(status), "wall_s": round(time.time() - t, 2),
           "max_rss_mib": round(usage.ru_maxrss / 1024, 1), **mon.stats}
    out = (TMP / "o.txt").read_text(errors="replace")
    for l in out.splitlines():
        if l.startswith("AB_INIT_S "): row["init_s"] = float(l.split()[1])
        elif l.startswith("AB_RECOG_S "): row["recognize_s"] = float(l.split()[1])
        elif l.startswith("AB_TEXT_HEX "):
            hx = l.split(" ", 1)[1].strip()
            row["text"] = bytes.fromhex(hx).decode("utf-8", "replace")
            row["text_sha256"] = hashlib.sha256(bytes.fromhex(hx)).hexdigest()
        elif l.startswith("AB_CONF "):
            p = l.split()
            row["confidences"] = [float(x) for x in p[2:]]
        elif l.startswith("AB_PROMPT_ACCEPTED"): row["prompt_accepted"] = int(l.split()[1])
        elif l.startswith("AB_INIT_FAIL"): row["init_fail"] = True
    return row


def parse_stderr(path, row):
    stage, ids = [], None
    with open(path, errors="replace") as f:
        for l in f:
            s = l.strip()
            m = re.search(r"gen_ids \((\d+)\):(.*)", s)
            if m:
                ids = [int(x) for x in m.group(2).split()]
                continue
            if s.startswith(STAGE_PREFIXES) and len(stage) < 60 and "decode_step[" not in s:
                stage.append(s[:300])
    row["stage_lines"] = stage
    if ids is not None:
        row["ids"] = ids
        row["ids_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    tail = Path(path).read_text(errors="replace")[-1500:]
    if row["rc"]:
        row["stderr_tail"] = tail


NODE = re.compile(r"node #\s*\d+ \(\s*([^)]*?)\s*\):\s+(\S+) \(\s*[^)]*\) \[\s*(\S+)")
SPLIT = re.compile(r"## SPLIT #\d+: (\S+) # (\d+) inputs")


def sched_summary(path):
    by_backend, cpu_ops, splits, graphs, names = Counter(), Counter(), Counter(), 0, Counter()
    with open(path, errors="replace") as f:
        for l in f:
            m = SPLIT.search(l)
            if m:
                splits[m.group(1)] += 1
                if "SPLIT #0:" in l:
                    graphs += 1
                continue
            m = NODE.search(l)
            if m:
                op, name, be = m.group(1), m.group(2), m.group(3)
                by_backend[be] += 1
                if be.upper().startswith("CPU"):
                    cpu_ops[op] += 1
                    names[re.sub(r"\d+", "#", name)[:40]] += 1
    return {"graphs": graphs, "nodes_by_backend": dict(by_backend), "splits_by_backend": dict(splits),
            "cpu_ops": dict(cpu_ops.most_common(30)), "cpu_node_names": dict(names.most_common(30)),
            "log_bytes": os.path.getsize(path)}


try:
    summary["cpu"] = subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)'", shell=True, capture_output=True,
                                    text=True).stdout
    summary["gpu"] = subprocess.run("nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader",
                                    shell=True, capture_output=True, text=True).stdout.strip()
    sh(f"git clone -q {REPO} {SRC} && git -C {SRC} checkout -q {SHA} && "
       f"git -C {SRC} submodule update --init --recursive -q")
    sys.path.insert(0, str(SRC))
    import kaggle_harness as kh
    kh.install_build_toolchain()
    t = time.time()
    if DEVICE == "gpu":
        flags, jobs = " ".join(kh.cuda_build_flags(kh.detect_cuda_arch())), kh.safe_build_jobs(gpu=True)
    else:
        flags, jobs = ("-DCRISPEMBED_NATIVE=OFF -DGGML_NATIVE=OFF -DGGML_BLAS=OFF -DBUILD_SHARED_LIBS=OFF "
                       "-DCRISPEMBED_BUILD_SHARED=OFF"), "4"
    sh(f"cmake -S {SRC} -B {BUILD} -G Ninja -DCMAKE_BUILD_TYPE=Release {flags} " + " ".join(kh.cache_and_link_flags()))
    sh(f"cmake --build {BUILD} --target crispembed-cli -j{jobs}")
    summary["build_s"] = round(time.time() - t, 1)
    summary["ccache"] = subprocess.run("ccache -s 2>&1 | head -12", shell=True, capture_output=True, text=True).stdout
    # Driver: reuse the CLI's exact compile and link commands with main.cpp swapped for the driver.
    (BUILD / "ab_driver.cpp").write_text(DRIVER)
    cmds = subprocess.run(f"ninja -C {BUILD} -t commands crispembed-cli", shell=True, capture_output=True,
                          text=True).stdout.splitlines()
    comp = next(c for c in cmds if "examples/cli/main.cpp" in c and " -c " in c)
    link = cmds[-1]
    obj = re.search(r"-o (\S*main\.cpp\.o)", comp).group(1)
    comp = comp.replace(re.search(r"(\S*examples/cli/main\.cpp)(?=\s|$)", comp).group(1), str(BUILD / "ab_driver.cpp"))
    comp = comp.replace(f"-o {obj}", f"-o {BUILD}/ab_driver.o").replace(f"-MF {obj}.d", f"-MF {BUILD}/ab_driver.o.d")
    out_bin = re.search(r"-o (\S*crispembed)(?=\s|$)", link).group(1)
    link = link.replace(obj, f"{BUILD}/ab_driver.o").replace(f"-o {out_bin} ", f"-o {BUILD}/ab-driver ")
    summary["driver_link"] = link[:400]
    sh(f"cd {BUILD} && {comp}")
    sh(f"cd {BUILD} && {link}")
    binary = BUILD / "ab-driver"

    sh(f"{sys.executable} -m pip install -q huggingface_hub pillow")
    from huggingface_hub import hf_hub_download
    from PIL import Image
    docs = TMP / "docs"
    docs.mkdir(exist_ok=True)
    sh(f"curl -sSL -o {TMP}/prepared.zip {PREPARED}")
    with zipfile.ZipFile(TMP / "prepared.zip") as z:
        z.extractall(docs)
    frozen = json.loads((SRC / "tests/results/qwen35-ocr/ovisocr2-f16-quality.json").read_text())
    frozen = frozen["producer"]["native_cases"]["f16"]
    cases = {}
    for cid in CASES:
        png = docs / f"{cid}.png"
        if sha256(png) != frozen[cid]["input_png_sha256"]:
            raise SystemExit(f"{cid} PNG differs from the frozen input")
        im = Image.open(png).convert("RGB")
        rgb = TMP / f"{cid}.rgb"
        rgb.write_bytes(im.tobytes())
        cases[cid] = (rgb, im.width, im.height)
        summary.setdefault("documents", {})[cid] = {"png_sha256": frozen[cid]["input_png_sha256"],
                                                    "w": im.width, "h": im.height}

    models = {}
    for name, src, _, _ in ENGINES:
        if ONLY and name not in ONLY or name.endswith("-fused") and "-fused" not in ",".join(ONLY):
            continue
        t = time.time()
        try:
            if src is None:  # LightOnOCR-3-1B: convert the pinned original to F16 here (converter verifies pins).
                venv = TMP / "lvenv"
                sh(f"{sys.executable} -m pip install -q uv==0.8.17 && {sys.executable} -m uv venv -q --python 3.12 "
                   f"--seed {venv} && {venv}/bin/python -m pip install -q -r "
                   f"{SRC}/models/requirements-lightonocr3-reference.txt")
                path = str(TMP / "lightonocr3-1b-f16.gguf")
                sh(f"cd {SRC} && HF_HOME={TMP}/hfl {venv}/bin/python models/convert-lightonocr-to-gguf.py --model "
                   f"lightonai/LightOnOCR-3-1B --revision 23d544085c886bbdace647bcfab91203299417e2 --profile "
                   f"lightonocr3-1b --dtype f16 --output {path}")
                sh(f"rm -rf {TMP}/hfl", check=False)
                src_meta = {"repo": "lightonai/LightOnOCR-3-1B", "revision": "23d544085c886bbdace647bcfab91203299417e2",
                            "converted": "f16"}
            else:
                path = hf_hub_download(src[0], src[2], revision=src[1], cache_dir=str(TMP / "hf"))
                src_meta = {"repo": src[0], "revision": src[1], "file": src[2]}
            models[name] = path
            summary["models"][name] = dict(src_meta, sha256=sha256(path), bytes=os.path.getsize(path),
                                           fetch_s=round(time.time() - t, 1))
        except BaseException as e:
            summary["models"][name] = {"error": repr(e)[:500]}
        save()

    base_env = {k: v for k, v in os.environ.items() if not k.startswith(("CRISPEMBED_", "GGML_SCHED"))}
    base_env.update(OMP_NUM_THREADS=str(THREADS))
    arms = ["cuda", "cpu"] if DEVICE == "gpu" else ["cpu"]
    # GPU: scheduler probe first (cheap, 8 tokens), then the timed arms.
    if DEVICE == "gpu":
        for name, _, eenv, prompt in ENGINES:
            if name not in models:
                continue
            rgb, w, h = cases["inventory-table"]
            env = dict(base_env, **eenv, GGML_SCHED_DEBUG="2")
            log = TMP / f"sched-{name}.log"
            row = run_case(binary, models[name], rgb, w, h, env, prompt, "auto", 8, log)
            summary["sched"][name] = dict(rc=row["rc"], wall_s=row["wall_s"], **sched_summary(log),
                                          **{k: row[k] for k in ("gpu_util_mean", "gpu_mem_peak_mib") if k in row})
            log.unlink()
            print("sched", name, json.dumps(summary["sched"][name])[:1500], flush=True)
            save()
    for arm in arms:
        for name, _, eenv, prompt in ENGINES:
            if name not in models:
                continue
            for cid in CASES:
                row = {"engine": name, "arm": arm, "case": cid}
                if el() > BUDGET_S:
                    row["skipped"] = "time budget"
                    summary["runs"].append(row); save(); continue
                env = dict(base_env, **eenv)
                if arm == "cpu" and DEVICE == "gpu":
                    env["CUDA_VISIBLE_DEVICES"] = ""
                rgb, w, h = cases[cid]
                errp = TMP / "e.txt"
                row.update(run_case(binary, models[name], rgb, w, h, env, prompt,
                                    "cpu" if arm == "cpu" else "auto", MAX_NEW, errp))
                parse_stderr(errp, row)
                summary["runs"].append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ("confidences", "ids", "text", "stage_lines")}),
                      flush=True)
                save()
except BaseException as e:
    summary["error"] = repr(e)[:1000]
    raise
finally:
    save()
