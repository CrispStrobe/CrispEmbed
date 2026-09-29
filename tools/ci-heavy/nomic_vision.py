#!/usr/bin/env python3
"""nomic-embed-vision-v1.5 (Apache-2.0): CrispEmbed vit_embed port vs transformers.

Converts the checkpoint (models/convert-nomic-vision-to-gguf.py, F32 and F16),
builds crispembed, and compares image embeddings with upstream
(trust_remote_code NomicVisionModel, embedding = L2-normalised token 0 of
last_hidden_state, as on the model card):

  raw     C++ --image-raw on the processor's exact pixel_values -> model parity
          (gate: cos >= 0.9999 F32, >= 0.999 F16)
  file    C++ --image on a 224x224 PNG (resize is an identity on both sides)
          -> preprocessing parity (reported, gate cos >= 0.999)

Also reports the retrieval sanity check the pair exists for: cosine of each
image to nomic-embed-text-v1.5 "search_query: " captions, upstream vs C++.

    gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/nomic_vision.py \\
        -f pip="torch torchvision transformers<5 einops pillow gguf safetensors"
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

OUT = Path(os.environ.get("HEAVY_OUT", "out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", "scratch"))
OUT.mkdir(parents=True, exist_ok=True)
SCR.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[2]
MODEL = "nomic-ai/nomic-embed-vision-v1.5"
TEXT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
SRC_IMAGES = ["face.png", "fox.png", "scan_page_pd.png", "staff_flova.png"]
CAPTIONS = ["a photo of a person's face", "a line of printed text", "a page from a book",
            "sheet music"]
res = {"model": MODEL, "images": {}, "errors": []}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def cos(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


try:
    import torch
    import torch.nn.functional as F
    from huggingface_hub import snapshot_download
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    snap = Path(snapshot_download(MODEL))
    ggufs = {}
    for tag, extra in (("f32", []), ("f16", ["--fp16"])):
        out = SCR / f"nomic-embed-vision-v1.5-{tag}.gguf"
        subprocess.check_call([sys.executable, str(REPO / "models/convert-nomic-vision-to-gguf.py"),
                               "--model-dir", str(snap), "--output", str(out), *extra])
        ggufs[tag] = out
    res["gguf_mb"] = {k: round(v.stat().st_size / 1e6, 1) for k, v in ggufs.items()}
    log("building crispembed-cli")
    b = SCR / "build"
    subprocess.check_call(["cmake", "-S", str(REPO), "-B", str(b), "-DCMAKE_BUILD_TYPE=Release"], stdout=subprocess.DEVNULL)
    subprocess.check_call(["cmake", "--build", str(b), "--target", "crispembed-cli", f"-j{os.cpu_count() or 4}"],
                          stdout=subprocess.DEVNULL)
    exe = next(p for p in b.rglob("crispembed") if p.is_file() and os.access(p, os.X_OK))

    proc = AutoImageProcessor.from_pretrained(MODEL)
    vis = AutoModel.from_pretrained(MODEL, trust_remote_code=True).eval()
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    txt = AutoModel.from_pretrained(TEXT_MODEL, trust_remote_code=True).eval()
    with torch.no_grad():
        enc = tok(["search_query: " + c for c in CAPTIONS], padding=True, return_tensors="pt")
        te = txt(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        te = (te * m).sum(1) / m.sum(1)
        te = F.normalize(F.layer_norm(te, (te.shape[1],)), dim=-1).numpy()  # model-card recipe

    def cpp(args):
        r = subprocess.run([str(exe), *args, "-t", str(os.cpu_count() or 4)], capture_output=True, text=True,
                           timeout=600)
        line = next((l for l in r.stdout.splitlines() if l.strip()), "")
        try:
            return [float(v) for v in line.split()], r.stderr[-800:]
        except ValueError:
            return None, (r.stdout[-400:] + r.stderr[-800:])

    for name in SRC_IMAGES:
        im = Image.open(REPO / "tests/regression/images" / name).convert("RGB")
        side = min(im.size)
        im = im.crop(((im.width - side) // 2, (im.height - side) // 2, (im.width + side) // 2,
                      (im.height + side) // 2)).resize((224, 224), Image.BICUBIC)
        png = SCR / f"{Path(name).stem}_224.png"
        im.save(png)
        pv = proc(im, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            ref = F.normalize(vis(pixel_values=pv).last_hidden_state[:, 0], p=2, dim=1)[0].numpy()
        raw = SCR / f"{Path(name).stem}.f32"
        pv[0].numpy().astype(np.float32).tofile(raw)
        e = {}
        for tag, gp in ggufs.items():
            emb, err = cpp(["-m", str(gp), "--image-raw", str(raw)])
            e[f"raw_{tag}"] = cos(emb, ref) if emb else err
            emb2, err2 = cpp(["-m", str(gp), "--image", str(png)])
            e[f"file_{tag}"] = cos(emb2, ref) if emb2 else err2
            if tag == "f32" and emb:
                e["caption_sims_cpp"] = [round(cos(emb, t), 4) for t in te]
        e["caption_sims_ref"] = [round(cos(ref, t), 4) for t in te]
        res["images"][name] = e
        log(f"{name}: {e}")
        (OUT / "result.json").write_text(json.dumps(res, indent=1))
    if "--keep-gguf" in sys.argv:  # artifact + SHA256SUMS, for upload with the HF write token elsewhere
        import hashlib
        import shutil
        sums = []
        for gp in ggufs.values():
            sums.append(f"{hashlib.sha256(gp.read_bytes()).hexdigest()}  {gp.name}")
            shutil.copy(gp, OUT / gp.name)
        (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")
except Exception:
    res["errors"].append(traceback.format_exc())
    print(res["errors"][-1], file=sys.stderr)
finally:
    (OUT / "result.json").write_text(json.dumps(res, indent=1))

num = lambda v: f"{v:.6f}" if isinstance(v, float) else f"ERR {str(v)[-120:]!r}"
lines = [f"### {MODEL}: C++ vit_embed vs transformers (cosine of L2-normalised embeddings)\n",
         f"GGUF sizes (MB): {res.get('gguf_mb')}\n",
         "| image | raw F32 | raw F16 | file F32 | file F16 | caption sims ref / C++ |", "|---|---|---|---|---|---|"]
for n, e in res["images"].items():
    lines.append(f"| {n} | {num(e['raw_f32'])} | {num(e['raw_f16'])} | {num(e['file_f32'])} | {num(e['file_f16'])} | "
                 f"{e['caption_sims_ref']} / {e.get('caption_sims_cpp')} |")
lines.append(f"\ncaptions: {CAPTIONS}")
if res["errors"]:
    lines.append("\n**errors:**\n```\n" + res["errors"][-1][-2000:] + "\n```")
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
isnum = lambda v: isinstance(v, float)
ok = (not res["errors"] and len(res["images"]) == len(SRC_IMAGES) and all(
    isnum(e["raw_f32"]) and e["raw_f32"] >= 0.9999 and isnum(e["raw_f16"]) and e["raw_f16"] >= 0.999
    and isnum(e["file_f32"]) and e["file_f32"] >= 0.999 for e in res["images"].values()))
sys.exit(0 if ok else 1)
