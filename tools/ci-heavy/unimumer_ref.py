#!/usr/bin/env python3
"""Uni-MuMER-Qwen3-VL-2B: C++ (crispembed) LaTeX vs upstream transformers.

The three Qwen3-VL port fixes of 2026-09-29 (qwen3vl.* normalisation keys,
deepstack after the layer, no Qwen2-VL system block) change Uni-MuMER's
input too - it is a Qwen3-VL-2B fine-tune - but it has no regression entry.
It also still gets the Qwen2-VL system block (exempted when the fix went in),
which upstream's chat template does not add. This settles both:

  hf_greedy     phxember/Uni-MuMER-Qwen3-VL-2B via its own processor + template
  cpp_default   crispembed, F16 GGUF, current default prompt construction
  cpp_nosys     same with CRISPEMBED_QARI_NO_SYSTEM=1 (no system block)

Exit 0 only if cpp_default == hf_greedy on every fixture.

    gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/unimumer_ref.py \\
        -f pip="torch torchvision transformers>=4.57 accelerate pillow"
"""
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

OUT = Path(os.environ.get("HEAVY_OUT", "out"))
SCR = Path(os.environ.get("HEAVY_SCRATCH", "scratch"))
OUT.mkdir(parents=True, exist_ok=True)
SCR.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[2]
MODEL = "phxember/Uni-MuMER-Qwen3-VL-2B"
GGUF_REPO, GGUF_FILE = "cstr/uni-mumer-qwen3-vl-2b-GGUF", "uni-mumer-qwen3-vl-2b-f16.gguf"
# qwen2vl_ocr.cpp's Uni-MuMER prompt (paper, Appendix A)
PROMPT = ("I have an image of a handwritten mathematical expression. Please write out the expression of "
          "the formula in the image using LaTeX format.")
MAX_NEW = 256
IMAGES = {n: REPO / "tests/regression/images" / f"{n}.png" for n in ("formula_quadratic", "mixtex_pow")}
res = {"model": MODEL, "images": {}, "errors": []}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def norm(s):
    return " ".join(s.split())


try:
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.set_num_threads(os.cpu_count() or 4)
    b = SCR / "build"
    log("building crispembed-cli")
    subprocess.check_call(["cmake", "-S", str(REPO), "-B", str(b), "-DCMAKE_BUILD_TYPE=Release"], stdout=subprocess.DEVNULL)
    subprocess.check_call(["cmake", "--build", str(b), "--target", "crispembed-cli", f"-j{os.cpu_count() or 4}"],
                          stdout=subprocess.DEVNULL)
    exe = next(p for p in b.rglob("crispembed") if p.is_file() and os.access(p, os.X_OK))
    gg = hf_hub_download(GGUF_REPO, GGUF_FILE)
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(MODEL)

    def cpp(img, extra_env=None):
        r = subprocess.run([str(exe), "-m", gg, "--ocr", str(img), "--ocr-max-tokens", str(MAX_NEW),
                            "-t", str(os.cpu_count() or 4)], capture_output=True, text=True, timeout=3600,
                           env=dict(os.environ, **(extra_env or {})))
        return r.stdout.strip()

    for name, img in IMAGES.items():
        im = Image.open(img).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": PROMPT}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False)
        e = {"hf_greedy": proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip(),
             "hf_prompt_tokens": int(inputs["input_ids"].shape[1]),
             "cpp_default": cpp(img), "cpp_nosys": cpp(img, {"CRISPEMBED_QARI_NO_SYSTEM": "1"})}
        e["default_matches"] = norm(e["cpp_default"]) == norm(e["hf_greedy"])
        e["nosys_matches"] = norm(e["cpp_nosys"]) == norm(e["hf_greedy"])
        res["images"][name] = e
        log(json.dumps({name: e}, ensure_ascii=False))
        (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
except Exception:
    res["errors"].append(traceback.format_exc())
    print(res["errors"][-1], file=sys.stderr)
finally:
    (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

cell = lambda s: "`" + s.replace("\n", "⏎").replace("|", "\\|")[:120] + "`"
lines = [f"### Uni-MuMER-Qwen3-VL-2B: C++ vs transformers\n",
         "| image | HF greedy | C++ default | C++ no system | default == HF | no-sys == HF |", "|---|---|---|---|---|---|"]
for n, e in res["images"].items():
    lines.append(f"| {n} | {cell(e['hf_greedy'])} | {cell(e['cpp_default'])} | {cell(e['cpp_nosys'])} | "
                 f"{e['default_matches']} | {e['nosys_matches']} |")
if res["errors"]:
    lines.append("\n**errors:**\n```\n" + res["errors"][-1][-1500:] + "\n```")
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
ok = (not res["errors"] and len(res["images"]) == len(IMAGES)
      and all(e["default_matches"] for e in res["images"].values()))
sys.exit(0 if ok else 1)
