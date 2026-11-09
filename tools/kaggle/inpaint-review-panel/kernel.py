"""Kaggle CPU kernel: issue-71 owner review panel for the published MI-GAN engines.

Clones CrispEmbed at the rendered commit, builds the CPU CLI, downloads the
SHA-pinned published MI-GAN512 F32/F16/Q8_0/Q4_K GGUFs (registry URLs and digests
from examples/cli/model_mgr.cpp / model_hashes.h) and reproduces the reviewed
MI-GAN256 F32 model with tools/prepare_migan_inpaint.py. MI-GAN256 has no
published F16/Q8_0/Q4_K variant, so its only column is F32.

Inputs: the two Wikimedia Commons photos already used by
tools/ci-heavy/inpaint_live_inputs.sh (licence read from the Commons API and
recorded), subtitled with the same renderer as tools/inpaint_live_eval.py.
Cases: two 512 still crops with exact text masks, the street crop with a DBNet
mask (the known Q4_K caveat case), and a four-frame synthetic pan across the
street photo with a subtitle change at frame 3. Writes /kaggle/working/report.json
and JPEG thumbnails. Evaluation only: no gold, no gate, no receipt is altered.
"""
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request

COMMIT = "@COMMIT@"
TMP = "/kaggle/temp"
CE = TMP + "/ce"
OUT = "/kaggle/working"
NT = str(os.cpu_count() or 4)
UA = {"User-Agent": "CrispEmbed-inpaint-review/1"}
HF = "https://huggingface.co/cstr/MI-GAN-512-GGUF/resolve/e1481f21784f99ec9d53ee935b978264be36abbb/"
M512 = {
    "F32": ("migan-512-places2-f32.gguf", "4bec681ef10f8ca914c5331f007a0605773cd09106d4fd6d9fcc0992f0a61739"),
    "F16": ("migan-512-places2-f16.gguf", "932cfbd35db478dba4372f7ad9889ee68c84eac1c0b46025cf7336c8d55c0033"),
    "Q8_0": ("migan-512-places2-q8_0.gguf", "efdf1b08207f2c47c4710ca66831e6aff86ea31fca68593337f2ce605db70280"),
    "Q4_K": ("migan-512-places2-q4_k.gguf", "bd90218cd6f86a9b215fa3528ad5400536b7cf883a76e2450f22c835e376af46"),
}
CKPT512 = "1d6087eee0aac8923ad2606be5d8caeb4824d3e4de331995e420c74e124a466a"
M256_SHA = "b8fcefc3de7e04f96867b9d15bd84a2cf7e0c444d2925644a7e8b15081809c8b"
DBNET = ("https://huggingface.co/cstr/dbnet-ic15-GGUF/resolve/main/dbnet-ic15-q8_0.gguf",
         "163c3b3d1211fd47b4ea323e2e684287bdc75ae68f0ffea397ccfcdc29a74de2")
PHOTOS = {
    "landscape": ("Tour_Eiffel_Wikimedia_Commons.jpg",
                  "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/Tour_Eiffel_Wikimedia_Commons.jpg/960px-Tour_Eiffel_Wikimedia_Commons.jpg"),
    "street": ("New_york_times_square-terabass.jpg",
               "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/New_york_times_square-terabass.jpg/1280px-New_york_times_square-terabass.jpg"),
}
BARS = {"F16": 0.98, "Q8_0": 0.97, "Q4_K": 0.94}
LOG = open(OUT + "/run.log", "a") if os.path.isdir(OUT) else None


def log(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    if LOG:
        LOG.write(msg + "\n"); LOG.flush()


def sh(*cmd, cwd=CE, env=None, check=True):
    log("$", " ".join(cmd))
    p = subprocess.run(list(cmd), cwd=cwd, env=env, capture_output=True, text=True)
    if p.returncode and check:
        log(p.stdout[-3000:], p.stderr[-3000:])
        raise SystemExit(f"failed: {cmd[0]} rc={p.returncode}")
    return p


def fetch(url, dest, sha=None):
    data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300).read()
    got = hashlib.sha256(data).hexdigest()
    if sha and got != sha:
        raise SystemExit(f"digest mismatch for {url}: {got}")
    open(dest, "wb").write(data)
    return got


def commons_licence(title):
    url = ("https://commons.wikimedia.org/w/api.php?action=query&format=json&prop=imageinfo&iiprop=extmetadata"
           "&titles=File:" + urllib.request.quote(title))
    pages = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))["query"]["pages"]
    meta = next(iter(pages.values()))["imageinfo"][0]["extmetadata"]
    strip = lambda v: re.sub(r"<[^>]+>", "", (meta.get(v) or {}).get("value", "")).strip()
    return {"source_page": "https://commons.wikimedia.org/wiki/File:" + title, "licence": strip("LicenseShortName"),
            "licence_url": strip("LicenseUrl"), "artist": strip("Artist"), "credit": strip("Credit")[:200]}


def main():
    os.makedirs(OUT + "/thumbs", exist_ok=True)
    os.makedirs(TMP, exist_ok=True)
    t0 = time.time()
    sh("git", "clone", "-q", "--filter=blob:none", "https://github.com/CrispStrobe/CrispEmbed.git", CE, cwd=TMP)
    sh("git", "checkout", "-q", COMMIT)
    head = sh("git", "rev-parse", "HEAD").stdout.strip()
    if head != COMMIT:
        raise SystemExit(f"checked out {head}, expected {COMMIT}")
    sh("git", "submodule", "update", "-q", "--init", "--depth", "1", "ggml")
    sh("cmake", "-S", ".", "-B", TMP + "/b", "-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=OFF",
       "-DCRISPEMBED_NATIVE=OFF", "-DGGML_CUDA=OFF", "-DGGML_METAL=OFF", "-DGGML_VULKAN=OFF", "-DGGML_BLAS=OFF",
       "-DBUILD_SHARED_LIBS=OFF", "-DCRISPEMBED_BUILD_SHARED=OFF")
    sh("cmake", "--build", TMP + "/b", "--target", "crispembed-cli", "--parallel", NT)
    binary = TMP + "/b/crispembed"
    log("build done", round(time.time() - t0), "s")

    import numpy as np
    from PIL import Image
    sys.path.insert(0, CE + "/tools")
    import inpaint_live_eval as live  # same subtitle renderer and LINES as the earlier live panel

    models = {}
    for v, (name, sha) in M512.items():
        fetch(HF + name, TMP + "/" + name, sha)
        models["512-" + v] = (TMP + "/" + name, sha)
    m256 = None
    p = sh(sys.executable, "tools/prepare_migan_inpaint.py", "--download", "--output-dir", TMP + "/m256", check=False)
    cand = TMP + "/m256/migan-256-places2-f32-experimental.gguf"
    if p.returncode == 0 and os.path.exists(cand) and hashlib.sha256(open(cand, "rb").read()).hexdigest() == M256_SHA:
        m256 = cand
    else:
        log("MI-GAN256 preparation failed:", p.stderr[-1500:])
    dbnet = TMP + "/dbnet.gguf"
    try:
        fetch(DBNET[0], dbnet, DBNET[1])
    except Exception as e:  # recorded; the DBNet case is then skipped
        log("dbnet fetch failed", e); dbnet = None

    inputs = {}
    for key, (title, url) in PHOTOS.items():
        dest = TMP + "/" + key + ".jpg"
        inputs[key] = {"url": url, "sha256": fetch(url, dest), **commons_licence(title)}
        if not re.search(r"CC|public domain|PD", inputs[key]["licence"], re.I):
            raise SystemExit(f"unexpected licence for {title}: {inputs[key]['licence']}")

    # Cases: (id, photo, left offset from centre, subtitle lines, mask kind, clip)
    lines_b = ["Zweite Zeile: die Kamera schwenkt", "langsam nach rechts – Bild 3"]
    cases = [("still-landscape", "landscape", 0, live.LINES, "exact", None),
             ("still-street", "street", 0, live.LINES, "exact", None),
             ("still-street-dbnet", "street", 0, live.LINES, "dbnet", None)]
    for i, dx in enumerate((-24, -8, 8, 24)):
        cases.append((f"pan-street-f{i + 1}", "street", dx, live.LINES if i < 2 else lines_b, "exact", "pan-street"))

    def subtitled(clean, lines):
        saved = live.LINES
        live.LINES = lines
        try:
            return live.subtitle(clean)
        finally:
            live.LINES = saved

    def line_masks(sel):
        rows, out, start = sel.any(axis=1), [], None
        for y, on in enumerate(list(rows) + [False]):
            if on and start is None:
                start = y
            elif not on and start is not None:
                band = np.zeros_like(sel); band[start:y] = sel[start:y]; out.append(band); start = None
        return out

    def thumb(img, name):
        img.save(f"{OUT}/thumbs/{name}.jpg", "JPEG", quality=85)

    report = {"schema": "crispembed-inpaint-owner-review-panel-v1", "commit": COMMIT, "backend": "CPU (Kaggle)",
              "threads": NT, "inputs": inputs, "bars_cosine_vs_f32": BARS,
              "models": {k: {"file": os.path.basename(p), "sha256": s} for k, (p, s) in models.items()},
              "migan256": {"file": "migan-256-places2-f32-experimental.gguf", "sha256": M256_SHA,
                           "available": bool(m256),
                           "note": "MI-GAN256 has no published F16/Q8_0/Q4_K variant; F32 only"},
              "cases": []}
    variants = ["256-F32", "512-F32", "512-F16", "512-Q8_0", "512-Q4_K"]
    arrays = {}
    for cid, photo, dx, lines, kind, clip in cases:
        img = Image.open(TMP + "/" + photo + ".jpg").convert("RGB")
        if min(img.size) < 512:
            s = 512 / min(img.size)
            img = img.resize((max(512, round(img.width * s)), max(512, round(img.height * s))), Image.LANCZOS)
        left, top = (img.width - 512) // 2 + dx, (img.height - 512) // 2
        clean = img.crop((left, top, left + 512, top + 512))
        subbed, mask = subtitled(clean, lines)
        d = f"{TMP}/cases/{cid}"
        os.makedirs(d, exist_ok=True)
        subbed.save(d + "/input.png"); mask.save(d + "/mask.png")
        thumb(subbed, cid + ".input"); thumb(mask.convert("RGB"), cid + ".mask")
        c, s_, mk = (np.asarray(x) for x in (clean, subbed, mask))
        sel = mk > 0
        case = {"id": cid, "photo": photo, "pan_dx": dx, "clip": clip, "mask": kind, "subtitle": lines,
                "masked_pixels": int(sel.sum()), "outputs": {}}
        for v in variants:
            if v == "256-F32":
                if not m256:
                    case["outputs"][v] = {"error": "model unavailable"}; continue
                eargs, env = ["--inpaint-model", m256], dict(os.environ, CRISPEMBED_MIGAN="1")
            else:
                path, sha = models[v]
                eargs = ["--inpaint-engine", "migan512", "--inpaint-model", path, "--inpaint-model-sha256", sha,
                         "--inpaint-checkpoint-sha256", CKPT512]
                env = dict(os.environ, CRISPEMBED_MIGAN="1", CRISPEMBED_MIGAN512="1")
                if not v.endswith("F32"):
                    env["CRISPEMBED_INPAINT_QUANT"] = "1"
            margs = ["--mask", d + "/mask.png"] if kind == "exact" else \
                ["--mask-from-text", "--mask-text-model", dbnet or "missing", "--mask-dilate", "6"]
            dest = f"{d}/{v}.png"
            t = time.time()
            p = sh(binary, *eargs, "--inpaint", d + "/input.png", *margs, "--inpaint-output", dest,
                   "--threads", NT, env=env, check=False)
            entry = {"returncode": p.returncode, "seconds": round(time.time() - t, 2)}
            if p.returncode or not os.path.exists(dest):
                entry["stderr_tail"] = p.stderr[-800:]
            else:
                r = np.asarray(Image.open(dest).convert("RGB"))
                arrays[(cid, v)] = r
                entry["vs_clean"] = live.scores(r, c, mk)
                entry["outside_exact"] = bool(np.array_equal(r[~sel], s_[~sel])) if kind == "exact" else None
                # Applied mask = pixels the engine changed (unmasked bytes are kept
                # exactly), reported as coverage of each exact subtitle line.
                changed = np.any(r != s_, axis=2)
                entry["applied_mask"] = {"changed_px": int(changed.sum()),
                                         "line_coverage": [round(float(changed[ln].sum()) / max(1, int(ln.sum())), 4)
                                                           for ln in line_masks(sel)]}
                if v.startswith("512-") and v != "512-F32" and (cid, "512-F32") in arrays:
                    a = live.agreement(r, arrays[(cid, "512-F32")], mk)
                    a["bar"] = BARS[v[4:]]
                    a["below_bar"] = a["rgb_cosine"] is not None and a["rgb_cosine"] <= BARS[v[4:]]
                    entry["vs_f32"] = a
                thumb(Image.fromarray(r), f"{cid}.{v}")
            case["outputs"][v] = entry
            log(cid, v, json.dumps({k: entry.get(k) for k in ("returncode", "vs_clean", "vs_f32")}))
        arrays[(cid, "clean")], arrays[(cid, "mask")] = c, sel
        report["cases"].append(case)

    # Temporal error proxy (docs/inpaint-video-quality.md): first difference of
    # reconstruction error over the union of adjacent masks, mean abs per channel.
    clip_ids = [c[0] for c in cases if c[5] == "pan-street"]
    temporal = {}
    for v in variants:
        vals = []
        for a, b in zip(clip_ids, clip_ids[1:]):
            if (a, v) not in arrays or (b, v) not in arrays:
                vals.append(None); continue
            ea = arrays[(a, v)].astype(np.int32) - arrays[(a, "clean")].astype(np.int32)
            eb = arrays[(b, v)].astype(np.int32) - arrays[(b, "clean")].astype(np.int32)
            u = arrays[(a, "mask")] | arrays[(b, "mask")]
            vals.append(round(float(np.mean(np.abs(eb[u] - ea[u]))), 3))
        temporal[v] = vals
    report["temporal_error_mae"] = {"clip": "pan-street", "pairs": [f"{a}->{b}" for a, b in zip(clip_ids, clip_ids[1:])],
                                    "by_variant": temporal,
                                    "note": "pan is 16 px/frame; camera motion is discounted by subtracting clean frames"}
    report["elapsed_s"] = round(time.time() - t0)
    json.dump(report, open(OUT + "/report.json", "w"), indent=1)
    log("done", report["elapsed_s"], "s")


if __name__ == "__main__":
    main()
