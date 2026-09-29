#!/usr/bin/env python3
"""Qwen3-VL-2B reference OCR (transformers, CPU fp32) on the fixtures where
CrispEmbed's port spells large display text letter by letter.

Question (2026-09-29, after CrispEmbed #56): on an ALPHA / OMEGA image the C++
qwen2vl_ocr engine outputs "A L P H A H A L P E R A L E M E G A ..." (CPU and
CUDA alike) while fox.png reads perfectly. Is that the model, or the port?
The port also always applies no-repeat-ngram=3 in its greedy decode, and a
model that spells letters one per line cannot repeat a trigram - so the ban
alone could scramble the letters.

Arms, per image, same prompt and resize as the C++ engine:
  hf_greedy          plain greedy (transformers default)
  hf_greedy_ngram3   greedy + no_repeat_ngram_size=3 (what the C++ engine does)
Compared against the C++ outputs recorded below (crispembed CPU, q4_k).

    gh workflow run heavy-cpu.yml -f script=tools/ci-heavy/qwen3vl_ref_ocr.py \\
        -f pip="torch transformers>=4.57 accelerate pillow"
"""
import json
import os
import sys
import time
import traceback
from pathlib import Path

OUT = Path(os.environ.get("HEAVY_OUT", "out"))
OUT.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MODEL = "Qwen/Qwen3-VL-2B-Instruct"
# qwen2vl_ocr.cpp default for the qwen3vl architecture
PROMPT = "Read all the text in this image. Output the exact text content only."
MAX_NEW = 48

FIXTURES = {
    "fox": REPO / "tests/regression/images/fox.png",
    "alpha_omega_1000x420": HERE / "fixtures/alpha_omega_1000x420.png",
    "alpha_omega_800x192": HERE / "fixtures/alpha_omega_800x192.png",
}
# crispembed (build of 2026-08-24, CPU, qwen3-vl-2b-q4_k.gguf, -t 4), same PNGs.
CPP = {
    "fox": "The quick brown fox jumps\nover the lazy dog. 12345",
    "alpha_omega_1000x420": "A L P H A H A L P E R A L E M E G A A L H A A M E L A H E M A L A L M E A L L A",
    "alpha_omega_800x192": "ALP\nH\nA\nO\nME\nG\nA",
}

res = {"model": MODEL, "prompt": PROMPT, "max_new_tokens": MAX_NEW, "runs": {}, "errors": []}


def save():
    (OUT / "result.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


def norm(s):
    return " ".join(s.upper().split())


try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    torch.set_num_threads(os.cpu_count() or 4)
    res["versions"] = {"torch": torch.__version__, "transformers": __import__("transformers").__version__}
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(MODEL)
    res["load_s"] = round(time.time() - t0, 1)
    save()

    for name, path in FIXTURES.items():
        img = Image.open(path).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                          return_tensors="pt")
        grid = inputs.get("image_grid_thw")
        entry = {"size": list(img.size), "grid_thw": grid.tolist() if grid is not None else None, "cpp": CPP[name]}
        for arm, extra in (("hf_greedy", {}), ("hf_greedy_ngram3", {"no_repeat_ngram_size": 3})):
            t1 = time.time()
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, **extra)
            text = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            entry[arm] = text
            entry[arm + "_s"] = round(time.time() - t1, 1)
            print(f"[{name}] {arm} ({entry[arm + '_s']} s): {text!r}", flush=True)
        entry["cpp_matches_hf_greedy"] = norm(CPP[name]) == norm(entry["hf_greedy"])
        entry["cpp_matches_hf_ngram3"] = norm(CPP[name]) == norm(entry["hf_greedy_ngram3"])
        res["runs"][name] = entry
        save()
except Exception:
    res["errors"].append(traceback.format_exc())
    print(res["errors"][-1], file=sys.stderr)
finally:
    save()

# Summary: the question is answered by the table, not by pass/fail. Only an
# error (or a missing arm) makes the run red.
lines = [f"### Qwen3-VL-2B reference OCR vs CrispEmbed\n", f"`{MODEL}`, CPU fp32, prompt: _{PROMPT}_\n",
         "| image | C++ (ngram 3) | HF greedy | HF greedy + ngram 3 |", "|---|---|---|---|"]
for name, e in res["runs"].items():
    cell = lambda s: "`" + s.replace("\n", "⏎").replace("|", "\\|")[:90] + "`"
    lines.append(f"| {name} {e['size'][0]}x{e['size'][1]} | {cell(e['cpp'])} | {cell(e.get('hf_greedy', ''))} | "
                 f"{cell(e.get('hf_greedy_ngram3', ''))} |")
if res["errors"]:
    lines.append("\n**errors:**\n```\n" + res["errors"][-1][-1500:] + "\n```")
(OUT / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
complete = len(res["runs"]) == len(FIXTURES) and all("hf_greedy_ngram3" in e for e in res["runs"].values())
sys.exit(0 if complete and not res["errors"] else 1)
