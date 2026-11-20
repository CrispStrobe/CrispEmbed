"""DeepSeek-OCR2 determinism bisect (21 November 2026).

The 20 November ocr-gpu-readiness receipt found DeepSeek-OCR2 (Q4_K stacked) not device-exact on two of
three documents. That kernel's CPU arms differed in two ways at once: the GPU image's binary is a NATIVE
build (GGML_NATIVE unset) on one Xeon, the CPU image's a PORTABLE build (GGML_NATIVE=OFF, fixed
AVX2/FMA/F16C) on another. This kernel separates the factors on one host: build ISA (portable vs native),
thread count (4 vs 1), repeat (same binary, same input), weight precision (Q4_K vs Q8_0 vs F16) and, on a
GPU, CUDA. Every run sets CRISPEMBED_DEEPSEEK_OCR2_TRACE_OUT, so each decode step records the top-2 logit
margin and each MoE layer the router's K-th/(K+1)-th probability gap; the receipt reports both at the
first divergence of every pair. Pusher substitutes __SHA__ and __DEVICE__ (gpu|cpu). Output: summary.json.
"""
import hashlib, json, os, re, subprocess, sys, time, zipfile
from pathlib import Path

SHA, DEVICE = "__SHA__", "__DEVICE__"
REPO = "https://github.com/CrispStrobe/CrispEmbed.git"
PREPARED = ("https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/resolve/"
            "057e12668bf8109ec83c65929c9b8d6cb99ca566/ocr-quality-v1/prepared-38879605915/prepared.zip")
CASES = ("two-column-page", "historical-receipt", "inventory-table")
MAX_NEW = 1024
WORK, TMP = Path("/kaggle/working"), Path("/kaggle/temp")
TMP.mkdir(parents=True, exist_ok=True)
SRC = TMP / "CrispEmbed"
MREPO, MREV = "cstr/deepseek-ocr2-crispembed-GGUF", "731d9377ea2a2f32b663267130ded422e8fe06f2"
MODELS = {"q4k": "deepseek-ocr2-q4_k-stacked.gguf", "q8": "deepseek-ocr2-q8_0.gguf", "f16": "deepseek-ocr2-f16-stacked.gguf"}
# arm: (build, model, threads, backend)
if DEVICE == "gpu":
    ARMS = {"cuda": ("cuda", "q4k", 4, "auto"), "cuda-rep": ("cuda", "q4k", 4, "auto"),
            "cuda-hostcpu": ("cuda", "q4k", 4, "cpu"), "portable": ("portable", "q4k", 4, "cpu"),
            "native": ("native", "q4k", 4, "cpu"), "cuda-f16": ("cuda", "f16", 4, "auto"),
            "portable-q8": ("portable", "q8", 4, "cpu")}
else:
    ARMS = {"portable": ("portable", "q4k", 4, "cpu"), "portable-rep": ("portable", "q4k", 4, "cpu"),
            "portable-t1": ("portable", "q4k", 1, "cpu"), "native": ("native", "q4k", 4, "cpu"),
            "portable-q8": ("portable", "q8", 4, "cpu"), "portable-f16": ("portable", "f16", 4, "cpu")}
ONLY = [a for a in os.environ.get("DS_ARMS", "").split(",") if a]
summary = {"schema": "crispembed-deepseek-ocr2-determinism-v1", "sha": SHA, "device": DEVICE, "arms": ARMS,
           "runs": [], "steps": []}
STARTED = time.time()

DRIVER = r'''
#include "crispembed.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
int main(int argc, char ** argv) {
    if (argc < 8) return 64;
    const int w = atoi(argv[3]), h = atoi(argv[4]);
    if (strcmp(argv[7], "auto") != 0) crispembed_set_gpu_backend(argv[7]);
    std::vector<unsigned char> px((size_t) w * h * 3);
    FILE * f = fopen(argv[2], "rb");
    if (!f || fread(px.data(), 1, px.size(), f) != px.size()) return 65;
    fclose(f);
    void * c = crispembed_ocr_model_init(argv[1], atoi(argv[6]));
    if (!c) { printf("AB_INIT_FAIL\n"); return 2; }
    crispembed_ocr_model_set_max_tokens(c, atoi(argv[5]));
    int n = 0;
    const char * txt = crispembed_ocr_model_recognize(c, px.data(), w, h, 3, &n);
    printf("AB_TEXT_HEX ");
    for (int i = 0; txt && i < n; ++i) printf("%02x", (unsigned char) txt[i]);
    printf("\n");
    fflush(stdout);
    crispembed_ocr_model_free(c);
    return 0;
}
'''


def save():
    summary["elapsed_s"] = round(time.time() - STARTED, 1)
    (WORK / "summary.json").write_text(json.dumps(summary, indent=1))


def sh(cmd, check=True):
    t = time.time()
    print("+", cmd[:300], flush=True)
    r = subprocess.run(cmd, shell=True)
    summary["steps"].append({"cmd": cmd[:160], "rc": r.returncode, "s": round(time.time() - t, 1)})
    save()
    if check and r.returncode:
        raise SystemExit(f"failed: {cmd[:200]}")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build(kind, kh):
    b = SRC / f"build-{kind}"
    base = "-DGGML_BLAS=OFF -DBUILD_SHARED_LIBS=OFF -DCRISPEMBED_BUILD_SHARED=OFF"
    if kind == "portable":
        flags, jobs = base + " -DCRISPEMBED_NATIVE=OFF -DGGML_NATIVE=OFF", "4"
    elif kind == "native":
        flags, jobs = base + " -DCRISPEMBED_NATIVE=ON -DGGML_NATIVE=ON", "4"
    else:  # the 20 November GPU arm: CUDA flags, native default
        flags, jobs = " ".join(kh.cuda_build_flags(kh.detect_cuda_arch())), kh.safe_build_jobs(gpu=True)
    sh(f"cmake -S {SRC} -B {b} -G Ninja -DCMAKE_BUILD_TYPE=Release {flags} " + " ".join(kh.cache_and_link_flags()))
    sh(f"cmake --build {b} --target crispembed-cli -j{jobs}")
    (b / "ab_driver.cpp").write_text(DRIVER)
    cmds = subprocess.run(f"ninja -C {b} -t commands crispembed-cli", shell=True, capture_output=True,
                          text=True).stdout.splitlines()
    comp = next(c for c in cmds if "examples/cli/main.cpp" in c and " -c " in c)
    link = cmds[-1]
    obj = re.search(r"-o (\S*main\.cpp\.o)", comp).group(1)
    comp = comp.replace(re.search(r"(\S*examples/cli/main\.cpp)(?=\s|$)", comp).group(1), str(b / "ab_driver.cpp"))
    comp = comp.replace(f"-o {obj}", f"-o {b}/ab_driver.o").replace(f"-MF {obj}.d", f"-MF {b}/ab_driver.o.d")
    out_bin = re.search(r"-o (\S*crispembed)(?=\s|$)", link).group(1)
    link = link.replace(obj, f"{b}/ab_driver.o").replace(f"-o {out_bin} ", f"-o {b}/ab-driver ")
    sh(f"cd {b} && {comp}")
    sh(f"cd {b} && {link}")
    return b / "ab-driver"


def parse_trace(path):
    steps, router = {}, {}
    for l in Path(path).read_text().splitlines():
        p = l.split()
        if p and p[0] == "S":
            steps[int(p[1])] = [int(p[2]), int(p[4]), float(p[6])]  # top1, top2, margin
        elif p and p[0] == "R":
            st, li, gap = int(p[1]), int(p[2]), float(p[4]) - float(p[6])
            if st not in router or gap < router[st][1]:
                router[st] = [li, gap, int(p[3]), int(p[5])]
    n = max(steps) + 1 if steps else 0
    return [steps.get(i) for i in range(n)], [router.get(i) for i in range(n)]


try:
    summary["cpu"] = subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)'; grep -o -w -E 'avx512f|avx512_vnni|avx2|"
                                    "amx_tile|avx_vnni' /proc/cpuinfo | sort | uniq -c", shell=True,
                                    capture_output=True, text=True).stdout
    sh(f"git clone -q {REPO} {SRC} && git -C {SRC} checkout -q {SHA} && git -C {SRC} submodule update --init --recursive -q")
    sys.path.insert(0, str(SRC))
    os.environ.setdefault("CE_CCACHE_EXPORT", "never")
    import kaggle_harness as kh
    kh.install_build_toolchain()
    arms = {a: v for a, v in ARMS.items() if not ONLY or a in ONLY}
    bins = {k: build(k, kh) for k in sorted({v[0] for v in arms.values()})}
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
        (TMP / f"{cid}.rgb").write_bytes(im.tobytes())
        cases[cid] = (TMP / f"{cid}.rgb", im.width, im.height)
    models = {}
    for m in sorted({v[1] for v in arms.values()}):
        path = hf_hub_download(MREPO, MODELS[m], revision=MREV, cache_dir=str(TMP / "hf"))
        models[m] = path
        summary.setdefault("models", {})[m] = {"file": MODELS[m], "revision": MREV, "sha256": sha256(path)}
        save()
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(("CRISPEMBED_", "GGML_SCHED"))}
    for cid, (rgb, w, h) in cases.items():
        for arm, (bk, m, thr, backend) in arms.items():
            env = dict(base_env, OMP_NUM_THREADS=str(thr), DS_DBG="1")
            if backend == "cpu" and DEVICE == "gpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            tr = TMP / "trace.txt"
            env["CRISPEMBED_DEEPSEEK_OCR2_TRACE_OUT"] = str(tr)
            t = time.time()
            with open(TMP / "o.txt", "w") as fo, open(TMP / "e.txt", "w") as fe:
                rc = subprocess.run([str(bins[bk]), models[m], str(rgb), str(w), str(h), str(MAX_NEW), str(thr),
                                     backend], env=env, stdout=fo, stderr=fe).returncode
            row = {"case": cid, "arm": arm, "rc": rc, "wall_s": round(time.time() - t, 1)}
            for l in (TMP / "o.txt").read_text(errors="replace").splitlines():
                if l.startswith("AB_TEXT_HEX "):
                    hx = l.split(" ", 1)[1].strip()
                    row["text"] = bytes.fromhex(hx).decode("utf-8", "replace")
            err = (TMP / "e.txt").read_text(errors="replace")
            mm = re.findall(r"gen_ids \((\d+)\):(.*)", err)
            if mm:
                row["ids"] = [int(x) for x in mm[-1][1].split()]
            if rc:
                row["stderr_tail"] = err[-1500:]
            if tr.exists():
                row["top2"], row["router_min"] = parse_trace(tr)
                tr.unlink()
            summary["runs"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("text", "ids", "top2", "router_min")}),
                  flush=True)
            save()
    # Pairwise first divergence with the margins at that step on both sides.
    comps = []
    by = {}
    for r in summary["runs"]:
        by.setdefault(r["case"], {})[r["arm"]] = r
    for cid, rs in by.items():
        names = list(rs)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                A, B = rs[a].get("ids") or [], rs[b].get("ids") or []
                d = next((k for k, (x, y) in enumerate(zip(A, B)) if x != y), None if len(A) == len(B) else min(len(A), len(B)))
                c = {"case": cid, "pair": [a, b], "ids_exact": A == B, "first_div": d, "n": [len(A), len(B)]}
                if d is not None:
                    for side, r in (("a", rs[a]), ("b", rs[b])):
                        t2, rm = r.get("top2") or [], r.get("router_min") or []
                        c[side] = {"top2": t2[d] if d < len(t2) else None,
                                   "min_margin_before": min((s[2] for s in t2[:d] if s), default=None),
                                   "router_min_gap_upto": min((x[1] for x in rm[:d + 1] if x), default=None)}
                comps.append(c)
    summary["comparisons"] = comps
except BaseException as e:
    summary["error"] = repr(e)[:1000]
    raise
finally:
    save()
