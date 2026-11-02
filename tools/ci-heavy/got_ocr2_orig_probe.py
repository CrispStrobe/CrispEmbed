#!/usr/bin/env python3
"""GOT-OCR2 ORIGINAL pipeline capture (stepfun-ai/GOT-OCR2_0 remote code, CPU, F32).

Reproduces GOTQwenForCausalLM.chat(ocr_type='ocr') step for step, except that
it runs on CPU in float32 (chat() hard-codes .cuda() + bf16 autocast). For each
image it records:
  * the exact preprocessing tensor (GOTImageEvalProcessor(1024): torchvision
    Resize((1024,1024), BICUBIC) on the PIL image -> ToTensor -> Normalize),
    saved as <out>/orig.<case>.pixels.npy plus its sha256/statistics,
  * the exact prompt token IDs (MPT conversation template + QWenTokenizer),
  * the generation config chat() uses, and
  * the greedy token IDs/text for each requested no_repeat_ngram_size
    (chat() uses 20; 0 = off is captured as an attribution control).

Needs transformers==4.37.2 (the remote code's pinned version), tiktoken, torchvision.

    python got_ocr2_orig_probe.py <out_dir> <max_new_tokens> <ngram,ngram> <case=img> [...]
"""

import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

REPO = "stepfun-ai/GOT-OCR2_0"
# Verbatim from modeling_GOT.py chat(): the triple-quoted literal keeps the
# 8-space source indentation of its second line.
SYSTEM = """<|im_start|>system
        You should follow the instructions carefully and explain your answers in detail."""


def main():
    out, max_new, ngrams = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
    cases = [a.split("=", 1) for a in sys.argv[4:]]
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 4)
    tok = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model = AutoModel.from_pretrained(REPO, trust_remote_code=True, low_cpu_mem_usage=True,
                                      use_safetensors=True, pad_token_id=151643,
                                      torch_dtype=torch.float32).eval()
    mod = sys.modules[type(model).__module__]
    chat_src = inspect.getsource(type(model).chat)
    # Guard: the template/processor/generation settings below are the ones chat() uses.
    for needle in ('You should follow the instructions carefully and explain your answers in detail."""',
                   "GOTImageEvalProcessor(image_size=1024)", "qs = 'OCR: '", "no_repeat_ngram_size = 20",
                   "do_sample=False", "num_beams = 1", 'roles=("<|im_start|>user\\n", "<|im_start|>assistant\\n")',
                   "sep_style=SeparatorStyle.MPT"):
        assert needle in chat_src, f"chat() source changed: {needle!r} not found"
    assert SYSTEM in chat_src, "system literal (with indentation) not found verbatim in chat()"

    proc = mod.GOTImageEvalProcessor(image_size=1024)
    conv = mod.Conversation(system=SYSTEM, roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
                            version="mpt", messages=(), offset=0, sep_style=mod.SeparatorStyle.MPT,
                            sep="<|im_end|>")
    qs = mod.DEFAULT_IM_START_TOKEN + mod.DEFAULT_IMAGE_PATCH_TOKEN * 256 + mod.DEFAULT_IM_END_TOKEN + "\n" + "OCR: "
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = torch.as_tensor(tok([prompt]).input_ids)
    prompt_ids = input_ids[0].tolist()

    rec = {"model": REPO, "transformers_remote_code": True, "device": "cpu", "dtype": "float32",
           "note": "chat() runs CUDA bf16 autocast with a .half() image; this capture is the same pipeline in F32 on CPU",
           "prompt": prompt, "prompt_ids": prompt_ids,
           "prompt_ids_no_imgpad": [i for i in prompt_ids if i != 151859],
           "token_ocr": tok("OCR").input_ids, "token_space_ocr": tok(" OCR").input_ids,
           "generation": {"do_sample": False, "num_beams": 1, "no_repeat_ngram_size": 20,
                          "max_new_tokens_chat": 4096, "max_new_tokens_capture": max_new,
                          "stopping": "KeywordsStoppingCriteria(['<|im_end|>']) + eos 151643"},
           "processor": {"resize": "torchvision.transforms.Resize((1024,1024), InterpolationMode.BICUBIC) on PIL",
                         "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]},
           "cases": {}}
    import torchvision, PIL, transformers
    rec["versions"] = {"torch": torch.__version__, "torchvision": torchvision.__version__,
                       "pillow": PIL.__version__, "transformers": transformers.__version__}

    for cid, path in cases:
        image = Image.open(path).convert("RGB")
        px = proc(image)  # (3, 1024, 1024) float32
        arr = px.numpy().astype(np.float32)
        np.save(os.path.join(out, f"orig.{cid}.pixels.npy"), arr)
        c = {"image": path, "image_size": list(image.size), "pixels_shape": list(arr.shape),
             "pixels_sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
             "pixels_mean": float(arr.mean()), "pixels_min": float(arr.min()), "pixels_max": float(arr.max()),
             "runs": {}}
        for ng in ngrams:
            stop = mod.KeywordsStoppingCriteria(["<|im_end|>"], tok, input_ids)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(input_ids, images=[px.unsqueeze(0)], do_sample=False, num_beams=1,
                                     no_repeat_ngram_size=ng, max_new_tokens=max_new,
                                     stopping_criteria=[stop])
            dt = time.time() - t0
            ids = gen[0, input_ids.shape[1]:].tolist()
            text = tok.decode(ids).strip()
            if text.endswith("<|im_end|>"):
                text = text[:-len("<|im_end|>")]
            text = text.strip()
            c["runs"][str(ng)] = {"token_ids": ids, "text": text, "n_tokens": len(ids), "seconds": round(dt, 2),
                                  "hit_cap": len(ids) >= max_new and ids[-1] not in (151643, 151645)}
            print(f"{cid} ngram={ng} n={len(ids)} {dt:.1f}s {text[:80]!r}", flush=True)
        rec["cases"][cid] = c
    with open(os.path.join(out, "orig.json"), "w") as f:
        json.dump(rec, f, indent=1)


if __name__ == "__main__":
    main()
