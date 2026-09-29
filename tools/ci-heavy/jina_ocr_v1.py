#!/usr/bin/env python3
"""jina-ocr-v1 (CC BY-NC 4.0) through CrispEmbed's DeepSeek-OCR v1 engine.

jina-ocr-v1 is a DeepSeek-OCR fine-tune: SAM ViT-B + CLIP-L/14 DeepEncoder,
linear 2048->1280 projector, DeepSeek-V2 MoE decoder (12 layers, 64 routed +
2 shared experts, top-6) plus MTP speculative heads (not needed for greedy).
That is exactly what src/unlimited_ocr.cpp implements, so this checks whether
models/convert-unlimited-ocr-to-gguf.py + that engine reproduce upstream.

Arms per image:
  hf_global   transformers (trust_remote_code), crop_mode=False: one 1024 view -
              what the C++ engine does (it has no tiling)
  hf_tiled    transformers default (crop_mode=True: global view + up to 9 tiles)
  cpp         crispembed, F16 GGUF, instruction tokens = upstream's own ids

Exit 0 only if cpp == hf_global on every fixture. Writes the F16 GGUF to
$HEAVY_SCRATCH; with --keep-gguf it is also copied to $HEAVY_OUT.

    gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/jina_ocr_v1.py \\
        -f pip="torch torchvision transformers accelerate pillow gguf safetensors einops addict easydict"
"""
import difflib
import json
import os
import shutil
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
MODEL = "jinaai/jina-ocr-v1"
IMAGES = {n: REPO / "tests/regression/images" / f"{n}.png" for n in ("fox", "scan_page_pd")}
QUICK = "--quick" in sys.argv  # prompt/prefix diagnosis: one image, no tiled arm, no quants
if QUICK:
    IMAGES = {"scan_page_pd": IMAGES["scan_page_pd"]}
MAX_NEW = 256
IMAGE_TOKEN = 128815
res = {"model": MODEL, "images": {}, "errors": []}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def norm(s):
    return " ".join(s.split())


def save():
    (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


gguf = SCR / "jina-ocr-v1-f16.gguf"
try:
    from huggingface_hub import snapshot_download

    snap = Path(snapshot_download(MODEL, local_dir=str(SCR / "jina-ocr-v1")))
    # The prompt layout comes from upstream's own processor + chat template:
    # "<|User|>:\n" <image block> "{recommended prompt}<|Assistant|>:\n", no BOS.
    import torch
    from PIL import Image
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(str(snap), trust_remote_code=True)
    probe = proc.prepare_ocr_inputs(Image.new("RGB", (64, 64), "white"), device=torch.device("cpu"),
                                    crop_mode=False, base_size=1024, image_size=1024)["input_ids"][0].tolist()
    first = probe.index(IMAGE_TOKEN)
    last = len(probe) - 1 - probe[::-1].index(IMAGE_TOKEN)
    res["prompt_prefix_ids"], res["prompt_instr_ids"] = probe[:first], probe[last + 1:]
    log(f"prompt prefix {res['prompt_prefix_ids']} instr {res['prompt_instr_ids']}")
    log("converting")
    subprocess.check_call([sys.executable, str(REPO / "models/convert-unlimited-ocr-to-gguf.py"),
                           "--model-dir", str(snap), "--output", str(gguf), "--fp16", "--name", "jina-ocr-v1",
                           "--license", "cc-by-nc-4.0", "--source", "https://huggingface.co/jinaai/jina-ocr-v1",
                           "--prompt-prefix-ids", ",".join(map(str, res["prompt_prefix_ids"])),
                           "--prompt-instr-ids", ",".join(map(str, res["prompt_instr_ids"])),
                           "--no-repeat-ngram", "0"])
    res["gguf_gb"] = round(gguf.stat().st_size / 1e9, 2)
    log("building crispembed-cli")
    b = SCR / "build"
    subprocess.check_call(["cmake", "-S", str(REPO), "-B", str(b), "-DCMAKE_BUILD_TYPE=Release"], stdout=subprocess.DEVNULL)
    subprocess.check_call(["cmake", "--build", str(b), "--target", "crispembed-cli", "crispembed-quantize",
                           f"-j{os.cpu_count() or 4}"], stdout=subprocess.DEVNULL)
    qexe = next(p for p in b.rglob("crispembed-quantize") if p.is_file() and os.access(p, os.X_OK))
    GGUFS = {"f16": gguf}
    for q in (() if QUICK else ("q8_0", "q4_k")):
        dst = SCR / f"jina-ocr-v1-{q}.gguf"
        subprocess.check_call([str(qexe), str(gguf), str(dst), q], stdout=subprocess.DEVNULL)
        GGUFS[q] = dst
    res["gguf_gb"] = {q: round(p.stat().st_size / 1e9, 2) for q, p in GGUFS.items()}
    exe = next(p for p in b.rglob("crispembed") if p.is_file() and os.access(p, os.X_OK))

    from transformers import AutoModelForCausalLM

    torch.set_num_threads(os.cpu_count() or 4)
    model = AutoModelForCausalLM.from_pretrained(str(snap), dtype=torch.bfloat16, trust_remote_code=True).eval()

    def hf(img, crop_mode):
        # Global arm at 1024 (base_size = image_size = 1024): the one view the C++
        # engine encodes (273 vision tokens); crop_mode=False alone gives 640.
        extra = {} if crop_mode else {"base_size": 1024, "image_size": 1024}
        inputs = proc.prepare_ocr_inputs(img, device=torch.device("cpu"), crop_mode=crop_mode, **extra)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
        return proc.decode_ocr(out, inputs["input_ids"]).strip(), inputs["input_ids"][0].tolist()

    hf_out = {}
    for name, path in IMAGES.items():
        img = Image.open(path).convert("RGB")
        g_text, g_ids = hf(img, False)
        t_text = "" if QUICK else hf(img, True)[0]
        last_img = max(i for i, t in enumerate(g_ids) if t == IMAGE_TOKEN)
        instr = g_ids[last_img + 1:]
        first_img = min(i for i, t in enumerate(g_ids) if t == IMAGE_TOKEN)
        hf_out[name] = {"hf_global": g_text, "hf_tiled": t_text, "hf_n_prompt": len(g_ids),
                        "hf_prefix_ids": g_ids[:first_img], "hf_prefix_text": proc.tokenizer.decode(g_ids[:first_img]),
                        "hf_between_ids": [t for t in g_ids[first_img:last_img + 1] if t != IMAGE_TOKEN],
                        "hf_n_image_tokens": sum(t == IMAGE_TOKEN for t in g_ids), "instr_ids": instr,
                        "bos": g_ids[0]}
        log(f"[{name}] hf_global: {g_text[:200]!r}")
    del model

    for name, path in IMAGES.items():
        e = hf_out[name]
        # No UOCR_INSTR: the GGUF's own prompt_prefix_ids / prompt_instr_ids are under test.
        env = dict(os.environ, UOCR_DBG="1", UOCR_DUMP_PIXELS=str(SCR / f"pix_{name}.f32"))
        for q, gp in GGUFS.items():
            r = subprocess.run([str(exe), "-m", str(gp), "--ocr", str(path), "--ocr-max-tokens", str(MAX_NEW),
                                "-t", str(os.cpu_count() or 4)], capture_output=True, text=True, env=env, timeout=3600)
            key = "cpp" if q == "f16" else f"cpp_{q}"
            e[key] = r.stdout.strip()
            e[key + "_ratio"] = round(difflib.SequenceMatcher(None, e["hf_global"].split(), e[key].split()).ratio(), 4)
            if q == "f16":
                e["cpp_prompt_line"] = next((l.strip() for l in r.stderr.splitlines() if "[dbg] prompt:" in l), None)
                (OUT / f"cpp_stderr_{name}.txt").write_text(r.stderr[-20000:])
        e["cpp_matches_global"] = norm(e["cpp"]) == norm(e["hf_global"])
        res["images"][name] = e
        save()
        log(f"[{name}] cpp: {e['cpp'][:200]!r} ratios f16/q8/q4: {e['cpp_ratio']}/{e.get('cpp_q8_0_ratio')}/{e.get('cpp_q4_k_ratio')}")
    # Same-pixels arm: upstream on the C++ engine's exact preprocessed global view.
    # If this equals the C++ text, the remaining gap is preprocessing rounding.
    # float32 here: the default arms run upstream in bf16 (8-bit mantissa), which
    # alone can flip near-tie tokens (curly vs straight quotes) - C++ is F16
    # weights with F32 accumulation. Same pixels + fp32 isolates the port.
    if QUICK:
        model = AutoModelForCausalLM.from_pretrained(str(snap), dtype=torch.float32, trust_remote_code=True).eval()

        def swap(obj, t):
            if torch.is_tensor(obj) and tuple(obj.shape[-3:]) == (3, 1024, 1024) and obj.numel() == t.numel():
                return t.reshape(obj.shape).to(obj.dtype), 1
            if isinstance(obj, (list, tuple)):
                out, n = [], 0
                for o in obj:
                    r, k = swap(o, t)
                    out.append(r)
                    n += k
                return type(obj)(out) if not isinstance(obj, tuple) else tuple(out), n
            if isinstance(obj, dict):
                out, n = {}, 0
                for k2, v in obj.items():
                    out[k2], k = swap(v, t)
                    n += k
                return out, n
            return obj, 0

        for name, path in IMAGES.items():
            pix = torch.from_numpy(np.fromfile(SCR / f"pix_{name}.f32", dtype=np.float32))
            inputs = proc.prepare_ocr_inputs(Image.open(path).convert("RGB"), device=torch.device("cpu"),
                                             crop_mode=False, base_size=1024, image_size=1024)
            inputs, n_swapped = swap(dict(inputs), pix)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
            t = proc.decode_ocr(out, inputs["input_ids"]).strip()
            e = res["images"][name]
            e["hf_on_cpp_pixels"], e["n_swapped"] = t, n_swapped
            e["hf_on_cpp_pixels_vs_cpp"] = round(difflib.SequenceMatcher(None, t.split(), e["cpp"].split()).ratio(), 4)
            e["hf_on_cpp_pixels_exact"] = norm(t) == norm(e["cpp"])
            log(f"[{name}] HF fp32 on C++ pixels (swapped {n_swapped}): exact={e['hf_on_cpp_pixels_exact']} "
                f"ratio={e['hf_on_cpp_pixels_vs_cpp']}")
        save()
    if "--keep-gguf" in sys.argv:
        for gp in GGUFS.values():
            shutil.copy(gp, OUT / gp.name)
except Exception:
    res["errors"].append(traceback.format_exc())
    print(res["errors"][-1], file=sys.stderr)
finally:
    save()

cell = lambda s: "`" + (s or "").replace("\n", "⏎").replace("|", "\\|")[:160] + "`"
lines = [f"### {MODEL} via the DeepSeek-OCR v1 engine (F16 GGUF, CPU) vs transformers (bf16)\n",
         f"GGUF: {res.get('gguf_gb')} GB\n", "| image | HF global view (1024) | C++ f16 | exact | word ratio f16 / q8_0 / q4_k | HF tiled (default) |",
         "|---|---|---|---|---|---|"]
for n, e in res["images"].items():
    lines.append(f"| {n} | {cell(e['hf_global'])} | {cell(e['cpp'])} | {e['cpp_matches_global']} | "
                 f"{e.get('cpp_ratio')} / {e.get('cpp_q8_0_ratio')} / {e.get('cpp_q4_k_ratio')} | {cell(e['hf_tiled'])} |")
    lines.append(f"| | prompt HF {e['hf_n_prompt']} tok ({e['hf_n_image_tokens']} image), prefix {e.get('hf_prefix_ids')} "
                 f"{e.get('hf_prefix_text')!r}, non-image ids inside the image block {e.get('hf_between_ids')}; "
                 f"C++ {e['cpp_prompt_line']} | | | | |")
if res["errors"]:
    lines.append("\n**errors:**\n```\n" + res["errors"][-1][-2000:] + "\n```")
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
ok = (not res["errors"] and len(res["images"]) == len(IMAGES)
      and all(e["cpp_matches_global"] for e in res["images"].values()))
sys.exit(0 if ok else 1)
