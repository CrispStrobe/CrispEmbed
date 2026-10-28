#!/usr/bin/env python3
"""Original-first CPU F32 reference for openbmb/MiniCPM-OCR-1.0 (hosted only).

Runs the native Transformers `minicpm_ocr` model (pinned version from
docs/minicpm-ocr-source.json) on frozen synthetic regions, after verifying every
pinned file. Captures, for the native port: rendered prompt, full input_ids,
pixel_values/target_sizes, vision embeddings and selected vision layers,
post_layernorm, merger output, each decoder layer's prefill output, final norm,
first-step logits, greedy generated IDs and text.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT/'docs/minicpm-ocr-source.json'
CASES = [  # id, lines, size, task prompt
    ('line', ['MiniCPM-OCR reference line 2026-10-28'], (560, 64), 'Text Recognition:'),
    ('para-en', ['MiniCPM-OCR reference page.', 'The quick brown fox jumps over the lazy dog.',
                 'Invoice 2026-10-27: total 1,234.56 EUR', 'Zeile mit Umlauten: Grüße, Straße, Äpfel.'],
     (1024, 768), 'Text Recognition:'),
    ('table', ['Item | Qty | Price', 'Apple | 3 | 1.20', 'Pear | 12 | 0.85', 'Total | 15 | 13.80'],
     (1024, 768), 'Table Recognition:'),
    ('tall', ['Tall page line %d: slicing check.' % i for i in range(1, 25)], (700, 2100), 'Text Recognition:'),
    ('tiny', ['42'], (60, 30), 'Text Recognition:'),
]
VISION_LAYERS = (0, 6, 26)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(8 << 20): h.update(b)
    return h.hexdigest()


def render(lines, size, path):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new('RGB', size, 'white'); d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=26)
    for i, line in enumerate(lines): d.text((16, 14 + 80*i if size[1] > 100 else 14), line, fill='black', font=font)
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
    a.add_argument('--max-new-tokens', type=int, default=1024)
    args = a.parse_args()
    if os.environ.get('GITHUB_ACTIONS') != 'true' and not os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
        a.error('original model execution is hosted-only')
    cfg = json.loads(CONTRACT.read_text())
    out = args.output; arrays = out/'arrays'; arrays.mkdir(parents=True, exist_ok=False)
    report = dict(schema='crispembed-minicpm-ocr-reference-v1', completed=False, revision=cfg['revision'],
                  contract_sha256=sha(CONTRACT), driver_sha256=sha(__file__), cases=[])
    save = lambda: (out/'reference.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    model_dir = out.parent/'checkpoint'
    fetch(cfg, model_dir); report['files_verified'] = len(cfg['files']); save()

    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor
    want = cfg['reference_runtime']['version']
    if transformers.__version__ != want:
        raise SystemExit(f'transformers {transformers.__version__} != pinned {want}')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    torch.use_deterministic_algorithms(True, warn_only=True)
    report['versions'] = dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__)
    proc = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.float32).eval()
    report['model_class'] = type(model).__name__

    captured = {}
    def keep(name):
        def hook(_m, _i, o):
            if name in captured: return                      # first call = prefill
            t = o[0] if isinstance(o, (tuple, list)) else o
            if isinstance(t, torch.Tensor): captured[name] = t.detach().float().cpu().numpy()
        return hook
    targets = {'model.vision_tower.embeddings', 'model.vision_tower.post_layernorm', 'model.merger',
               'model.language_model.norm', 'lm_head'}
    targets |= {f'model.vision_tower.encoder.layers.{i}' for i in VISION_LAYERS}
    targets |= {f'model.language_model.layers.{i}' for i in range(24)}
    handles = [m.register_forward_hook(keep(n)) for n, m in model.named_modules() if n in targets]
    report['hooked_modules'] = sorted(n for n, _ in model.named_modules() if n in targets)
    if len(handles) != len(targets):
        raise SystemExit('module names differ from audited layout: %s' % sorted(targets - set(report['hooked_modules'])))

    def dump(cid, name, value, row):
        f = arrays/cid/(name.replace('.', '_')+'.npy'); f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, value); row['arrays'][name] = dict(path=str(f.relative_to(out)), shape=list(value.shape), sha256=sha(f))

    for cid, lines, size, task in CASES:
        captured.clear()
        image_path = out/(cid+'.png'); image_sha = render(lines, size, image_path)
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': task}]}]
        prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=[prompt], images=[[image]], return_tensors='pt')
        row = dict(id=cid, lines=lines, size=list(size), task=task, image_sha256=image_sha, prompt=prompt,
                   input_ids=inputs['input_ids'][0].tolist(),
                   input_text=proc.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False),
                   input_keys=sorted(inputs.keys()), arrays={})
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and k not in ('input_ids', 'attention_mask'):
                dump(cid, 'inputs.'+k, v.float().numpy() if v.is_floating_point() else v.numpy(), row)
        started = time.time()
        with torch.inference_mode():
            gen = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens)
        ids = gen[0, inputs['input_ids'].shape[1]:].tolist()
        row.update(seconds=time.time()-started, ids=ids, text=proc.tokenizer.decode(ids, skip_special_tokens=True),
                   eos=bool(ids) and ids[-1] in (248044, 248046), cap_hit=len(ids) >= args.max_new_tokens)
        for name, value in captured.items():
            if name == 'lm_head': value = value[..., -1, :]   # first-step logits only
            dump(cid, name, value, row)
        report['cases'].append(row); save()
    report['completed'] = True; save()


if __name__ == '__main__':
    main()
