#!/usr/bin/env python3
"""Original-first CPU F32 reference for baidu/Unlimited-OCR-2 (hosted only).

Runs the publisher's own `infer` path (DASH block-sparse prefill and decode) on
frozen synthetic pages, with every pinned file verified before remote code runs.
Captures, for the native port: projector output, each decoder layer's prefill
output, each dash_router/indexer first output, first-step logits, generated IDs
and text. Triton paths are CUDA-only, so the torch fallback runs; `.cuda()` is
made the identity and bf16 image tensors are cast to the F32 model dtype.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT/'docs/unlimited-ocr2-source.json'
PAGES = [
    ('para-en', ['Unlimited OCR 2 reference page.', 'The quick brown fox jumps over the lazy dog.',
                 'Invoice 2026-10-27: total 1,234.56 EUR', 'Zeile mit Umlauten: Grüße, Straße, Äpfel.']),
    ('table', ['Item | Qty | Price', 'Apple | 3 | 1.20', 'Pear | 12 | 0.85', 'Total | 15 | 13.80']),
]


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(8 << 20): h.update(b)
    return h.hexdigest()


def render(lines, path):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', (1024, 768), 'white'); d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=34)
    for i, line in enumerate(lines): d.text((48, 60 + 90*i), line, fill='black', font=font)
    img.save(path, format='PNG', optimize=False)
    return sha(path)


def fetch(cfg, directory):
    from huggingface_hub import hf_hub_download
    for name, pin in cfg['files'].items():
        p = Path(hf_hub_download(cfg['repository'], name, revision=cfg['revision'], local_dir=directory))
        if p.stat().st_size != pin['bytes'] or ('sha256' in pin and sha(p) != pin['sha256']):
            raise ValueError('pinned original file differs: '+name)


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--max-new-tokens', type=int, default=384)
    args = a.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true' and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        a.error('original model execution is hosted-only')
    cfg = json.loads(CONTRACT.read_text())
    out = args.output; arrays = out/'arrays'; arrays.mkdir(parents=True, exist_ok=False)
    report = dict(schema='crispembed-unlimited-ocr2-reference-v1', completed=False, revision=cfg['revision'],
                  contract_sha256=sha(CONTRACT), driver_sha256=sha(__file__), cases=[], arrays={})
    save = lambda: (out/'reference.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    model_dir = out.parent/'checkpoint'
    fetch(cfg, model_dir); report['files_verified'] = len(cfg['files']); save()

    import numpy as np
    import torch
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.Tensor.cuda = lambda self, *a, **k: self          # CPU reference: no device moves
    import transformers
    from transformers import AutoModel, AutoTokenizer
    report['versions'] = dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__)
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, use_safetensors=True,
                                      torch_dtype=torch.float32).eval()
    mod = sys.modules[type(model).__module__]
    build = mod._uocr_build_inputs
    def cast(x):
        if isinstance(x, torch.Tensor): return x.float() if x.dtype == torch.bfloat16 else x
        if isinstance(x, (list, tuple)): return type(x)(cast(v) for v in x)
        return x
    mod._uocr_build_inputs = lambda *a, **k: cast(build(*a, **k))

    captured = {}
    def keep(name):
        def hook(_m, _i, o):
            if name in captured: return                      # first call = prefill
            t = o[0] if isinstance(o, (tuple, list)) else o
            if isinstance(t, torch.Tensor): captured[name] = t.detach().float().cpu().numpy()
        return hook
    handles = []
    for name, m in model.named_modules():
        last = name.rsplit('.', 1)[-1]
        if (name.endswith('projector') or last in ('dash_router', 'indexer', 'lm_head', 'norm')
                or (name.rsplit('.', 2)[-2:-1] == ['layers'] and last.isdigit() and 'vision' not in name and 'sam' not in name)):
            handles.append(m.register_forward_hook(keep(name)))
    report['hooked_modules'] = len(handles)

    for cid, lines in PAGES:
        captured.clear()
        image = out/(cid+'.png'); image_sha = render(lines, image)
        started = time.time()
        with torch.inference_mode():
            res = model.infer(tok, image_file=str(image), eval_mode=True, return_dict=True,
                              max_new_tokens=args.max_new_tokens, temperature=0.0)
        row = dict(id=cid, lines=lines, image_sha256=image_sha, seconds=time.time()-started,
                   prompt=res.get('prompt'), text=res.get('text'), arrays={})
        for k in ('ids', 'token_ids', 'output_ids', 'input_ids'):
            if k in res and res[k] is not None:
                v = res[k]; row[k] = v.tolist() if hasattr(v, 'tolist') else list(v)
        row['cap_hit'] = len(row.get('ids') or row.get('token_ids') or row.get('output_ids') or []) >= args.max_new_tokens
        for name, value in captured.items():
            f = arrays/cid/(name.replace('.', '_')+'.npy'); f.parent.mkdir(parents=True, exist_ok=True)
            np.save(f, value); row['arrays'][name] = dict(path=str(f.relative_to(out)), shape=list(value.shape), sha256=sha(f))
        report['cases'].append(row); save()
    report['completed'] = True; save()


if __name__ == '__main__':
    main()
