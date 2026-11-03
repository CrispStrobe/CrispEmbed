#!/usr/bin/env python3
"""MiniCPM-OCR-1.0 quantization sweep helper (hosted only; docs/minicpm-ocr-q4k.md).

Subcommands (driven by tools/ci-heavy/minicpm_q4k_sweep.sh):

  fetch DIR            frozen original report (SHA256 aeb1756d...), the five case PNGs,
                       first-step lm_head and pixel_values arrays from the original
                       reference kernel output; every file is checked against the report.
  original REF CKPT OUT
                       teacher-forced ORIGINAL logits: one Transformers 5.21.0 F32 forward
                       per case over frozen prompt + frozen generated IDs[:-1]; keeps the
                       vocabulary scores at every generated position. Checks input_ids and
                       pixel_values bit-equal to the frozen capture, row 0 against the frozen
                       first-step lm_head and the argmax at every position against the frozen IDs.
  argv RECIPES NAME    print the crispembed-quantize argv tail for a named recipe
  env RECIPES NAME     print the recipe's quantizer environment (named policies)
  census GGUF          stored-type census (tensor count/bytes per type, file bytes)
  score REF ORIG ARM DIR
                       per-position cosine of the native arm's teacher-forced logits
                       (DIR/<case>.f32) against the original ones, argmax agreement and
                       first divergence; free-running results (DIR/<case>.free.json) if present.
  stages REF FROZEN NATIVE
                       prefill decoder layers/final norm of a native dump against the frozen
                       original arrays (worst stage by minimum row cosine).
  table OUT...         markdown table of every arm's score JSON.

The frozen original is the native-parity reference (tests/results/minicpm-ocr/native-parity.json):
Transformers 5.21.0, torch 2.13.0+cpu, F32, greedy, openbmb/MiniCPM-OCR-1.0 at 6796197b.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPORT_SHA256 = 'aeb1756dcbcfb72d72c13cf5a8d8131bb90e8d8884cf07c70723cc1fe4a64c32'
KERNEL = 'crispembed-minicpm-ocr-reference'
CASES = ('line', 'para-en', 'table', 'tall', 'tiny')
VOCAB = 248100


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def require(ok, why):
    if not ok:
        raise SystemExit('minicpm q4k sweep: ' + why)


def fetch(out, layers=''):
    out = Path(out)
    account = os.environ.get('KAGGLE_ACCOUNT', '')
    require(account and os.environ.get('KAGGLE_API_TOKEN'), 'KAGGLE_ACCOUNT/KAGGLE_API_TOKEN required')
    pattern = r'(reference\.json|\.png|lm_head\.npy|inputs_pixel_values\.npy'
    if layers:  # prefill decoder layers and final norm, for the worst-stage comparison
        pattern += r'|model_language_model_(layers_\d+|norm)\.npy'
    pattern += ')$'
    subprocess.run(['kaggle', 'kernels', 'output', f'{account}/{KERNEL}', '-p', str(out), '--file-pattern', pattern],
                   check=True, stdout=subprocess.DEVNULL)
    base = out/'minicpm-ocr'
    require(sha(base/'reference.json') == REPORT_SHA256, 'frozen original report hash differs')
    ref = json.loads((base/'reference.json').read_text())
    for c in ref['cases']:
        require(sha(base/(c['id']+'.png')) == c['image_sha256'], 'frozen image differs: '+c['id'])
        names = ['lm_head', 'inputs.pixel_values']
        if layers:
            names += [n for n in c['arrays'] if n.startswith('model.language_model.')]
        for name in names:
            a = c['arrays'][name]
            require(sha(base/a['path']) == a['sha256'], f'frozen array differs: {c["id"]}/{name}')
    print('frozen original verified:', base)


def load_ref(ref):
    ref = Path(ref)
    require(sha(ref) == REPORT_SHA256, 'frozen original report hash differs')
    return json.loads(ref.read_text())


def cos(a, b):
    import numpy as np
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def original(ref_path, ckpt, out):
    import numpy as np
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor
    sys.path.insert(0, str(ROOT/'tools/minicpm-ocr'))
    import reference as frozen_driver
    ref = load_ref(ref_path); base = Path(ref_path).parent; out = Path(out); out.mkdir(parents=True, exist_ok=True)
    require(transformers.__version__ == '5.21.0', 'transformers must be the pinned 5.21.0')
    cfg = json.loads((ROOT/'docs/minicpm-ocr-source.json').read_text())
    frozen_driver.fetch(cfg, ckpt)
    torch.set_num_threads(os.cpu_count() or 4)
    proc = AutoProcessor.from_pretrained(ckpt)
    model = AutoModelForImageTextToText.from_pretrained(ckpt, dtype=torch.float32).eval()
    hidden = {}
    norm = dict(model.named_modules())['model.language_model.norm']
    norm.register_forward_hook(lambda _m, _i, o: hidden.__setitem__('h', o[0] if isinstance(o, (tuple, list)) else o))
    summary = dict(schema='crispembed-minicpm-ocr-teacher-forced-original-v1', report_sha256=REPORT_SHA256,
                   versions=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__),
                   frozen_versions=ref['versions'], cases=[])
    for c in ref['cases']:
        image = Image.open(base/(c['id']+'.png')).convert('RGB')
        inputs = proc(text=[c['prompt']], images=[[image]], return_tensors='pt')
        require(inputs['input_ids'][0].tolist() == c['input_ids'], 'input_ids differ from frozen: '+c['id'])
        frozen_px = np.load(base/c['arrays']['inputs.pixel_values']['path'])
        require(np.array_equal(inputs['pixel_values'].float().numpy(), frozen_px), 'pixel_values differ: '+c['id'])
        ids = c['ids']; P = len(c['input_ids']); G = len(ids)
        full = torch.tensor([c['input_ids'] + ids[:-1]], dtype=inputs['input_ids'].dtype)
        feed = {k: v for k, v in inputs.items() if k not in ('input_ids', 'attention_mask')}
        hidden.clear()
        with torch.inference_mode():
            try:  # logits for the last position only; the hooked final norm carries every position
                model(input_ids=full, attention_mask=torch.ones_like(full), logits_to_keep=1, **feed)
            except TypeError:
                model(input_ids=full, attention_mask=torch.ones_like(full), **feed)
            h = hidden['h'][0, P-1:P+G-1]
            logits = model.lm_head(h).float().numpy()
        require(logits.shape == (G, VOCAB), 'unexpected logits shape')
        np.save(out/(c['id']+'.npy'), logits)
        first = np.load(base/c['arrays']['lm_head']['path']).reshape(-1)
        argmax = logits.argmax(-1).tolist()
        row = dict(id=c['id'], prompt_tokens=P, generated=G, input_ids_exact=True, pixel_values_exact=True,
                   first_step_cosine_vs_frozen=cos(logits[0], first),
                   first_step_max_abs_vs_frozen=float(np.abs(logits[0]-first).max()),
                   argmax_matches_frozen_ids=sum(int(a == b) for a, b in zip(argmax, ids)),
                   first_argmax_mismatch=next((i for i, (a, b) in enumerate(zip(argmax, ids)) if a != b), None),
                   sha256=sha(out/(c['id']+'.npy')))
        summary['cases'].append(row); print(json.dumps(row), flush=True)
    summary['all_argmax_exact'] = all(r['first_argmax_mismatch'] is None for r in summary['cases'])
    summary['min_first_step_cosine'] = min(r['first_step_cosine_vs_frozen'] for r in summary['cases'])
    (out/'original.json').write_text(json.dumps(summary, indent=2)+'\n')
    require(summary['all_argmax_exact'], 'teacher-forced original argmax differs from the frozen IDs')
    require(summary['min_first_step_cosine'] > 0.99999, 'teacher-forced original row 0 differs from the frozen lm_head')


def recipe_argv(recipes, name):
    r = json.loads(Path(recipes).read_text())['recipes'][name]
    if 'type' not in r:
        return ''
    argv = [r['type']]
    for rule in r.get('tensor_type', []):
        argv += ['--tensor-type', rule]
    return ' '.join(argv)


def census(path):
    import gguf
    rd = gguf.GGUFReader(path)
    by = {}
    for t in rd.tensors:
        k = t.tensor_type.name
        e = by.setdefault(k, dict(tensors=0, bytes=0))
        e['tensors'] += 1; e['bytes'] += int(t.n_bytes)
    groups = {}
    for t in rd.tensors:
        n = t.name
        g = ('embedding' if n == 'l.embed_tokens.weight' else 'decoder' if n.startswith('l.') else
             'vision_blocks' if n.startswith('v.blk.') else 'vit_merger' if n.startswith('v.vmerge.') else
             'merger' if n.startswith('m.') else 'vision_other')
        e = groups.setdefault(g, {})
        e[t.tensor_type.name] = e.get(t.tensor_type.name, 0) + int(t.n_bytes)
    h = hashlib.sha256()  # tensor payloads only: a policy stamp changes metadata, not tensor data
    for t in rd.tensors:
        h.update(t.name.encode()); h.update(t.tensor_type.name.encode()); h.update(bytes(t.data))
    return dict(file_bytes=os.path.getsize(path), tensor_data_sha256=h.hexdigest(), file_mib=round(os.path.getsize(path)/2**20, 1), types=by,
                group_bytes=groups,
                patch_embedding=next((dict(type=t.tensor_type.name, shape=[int(x) for x in t.shape])
                                      for t in rd.tensors if t.name == 'v.emb.patch_embedding.weight'), None))


def score(ref_path, orig_dir, arm, d):
    import numpy as np
    ref = load_ref(ref_path); d = Path(d)
    res = dict(arm=arm, cases=[])
    for c in ref['cases']:
        o = np.load(Path(orig_dir)/(c['id']+'.npy'))
        raw = np.fromfile(d/(c['id']+'.f32'), dtype='<f4')
        require(raw.size == o.size, f'{arm}/{c["id"]}: native logits {raw.size} != original {o.size}')
        n = raw.reshape(o.shape).astype(np.float64); o = o.astype(np.float64)
        cs = (n*o).sum(1) / (np.linalg.norm(n, axis=1)*np.linalg.norm(o, axis=1))
        am = n.argmax(1).tolist(); ids = c['ids']
        mism = [i for i, (a, b) in enumerate(zip(am, ids)) if a != b]
        row = dict(id=c['id'], positions=len(ids), cos_min=float(cs.min()), cos_mean=float(cs.mean()),
                   cos_argmin=int(cs.argmin()), max_abs=float(np.abs(n-o).max()),
                   argmax_agree=len(ids)-len(mism), first_divergence=mism[0] if mism else None,
                   forced_seconds=None)
        t = d/(c['id']+'.forced.json')
        if t.exists():
            row['forced_seconds'] = json.loads(t.read_text()).get('seconds')
        f = d/(c['id']+'.free.json')
        if f.exists():
            fr = json.loads(f.read_text())
            row.update(free_ids_exact=fr['ids'] == ids, free_text_exact=fr['text'] == c['text'],
                       free_eos=bool(fr['ids']) and fr['ids'][-1] in (248044, 248046), free_generated=len(fr['ids']),
                       free_text=fr['text'] if fr['text'] != c['text'] else None, free_seconds=fr.get('seconds'))
        else:
            # Greedy decoding is a deterministic function of the prefix: when every teacher-forced
            # argmax equals the frozen ID, free running reproduces the frozen IDs exactly.
            row['decoded_exact_by_forced_argmax'] = not mism
        res['cases'].append(row)
    res['cos_min'] = min(r['cos_min'] for r in res['cases'])
    res['worst_case'] = min(res['cases'], key=lambda r: r['cos_min'])['id']
    res['decoded_exact_cases'] = sum(1 for r in res['cases'] if r['first_divergence'] is None)
    return res


def stages(ref_path, frozen_base, native):
    import numpy as np
    ref = load_ref(ref_path); out = []
    for c in ref['cases']:
        nd = Path(native)/c['id']
        if not nd.exists():
            continue
        for name, a in c['arrays'].items():
            if not (name.startswith('model.language_model.layers.') or name == 'model.language_model.norm'):
                continue
            p = Path(frozen_base)/a['path']; q = nd/(name.replace('.', '_')+'.npy')
            if not p.exists() or not q.exists():
                continue
            o = np.load(p).reshape(-1, 1024).astype(np.float64); n = np.load(q).reshape(-1, 1024).astype(np.float64)
            if o.shape != n.shape:
                out.append(dict(case=c['id'], stage=name, structural=f'{o.shape} vs {n.shape}')); continue
            rc = (o*n).sum(1)/(np.linalg.norm(o, axis=1)*np.linalg.norm(n, axis=1))
            out.append(dict(case=c['id'], stage=name, min_row_cosine=float(rc.min()), global_cosine=cos(o.ravel(), n.ravel()),
                            max_abs=float(np.abs(o-n).max()), original_abs_max=float(np.abs(o).max())))
    ok = [s for s in out if 'min_row_cosine' in s]
    return dict(stages=out, worst=min(ok, key=lambda s: s['min_row_cosine']) if ok else None)


def receipt(score_path, dtype, commit, build_mode_txt, stages_path=''):
    """Publication precision receipt (schema read by tools/ci-heavy/minicpm_ocr_publish.py)."""
    r = json.loads(Path(score_path).read_text())
    bm = dict(line.split('=', 1) for line in Path(build_mode_txt).read_text().splitlines() if '=' in line)
    worst = json.loads(Path(stages_path).read_text())['worst'] if stages_path else None
    cases = [dict(id=c['id'], final_cos_min=c['cos_min'], final_cos_mean=c['cos_mean'], positions=c['positions'],
                  ids_exact=c['first_divergence'] is None and c.get('free_ids_exact', True),
                  first_divergence=c['first_divergence'], free_text=c.get('free_text'),
                  forced_seconds=c['forced_seconds']) for c in r['cases']]
    return dict(schema='crispembed.minicpm.q4k-precision.v1', model='openbmb/MiniCPM-OCR-1.0',
                reference_revision='6796197bb3a8e224b2f64ae462bfbb3f8a15ac4b',
                metric='minimum final-logit cosine over every generated position, teacher-forced on the frozen original '
                       'IDs, against the original Transformers 5.21.0 F32 logits recomputed from the frozen inputs',
                reference='original model (frozen report sha256 ' + REPORT_SHA256 + ')', bar=0.94, commit=commit,
                worst_final_cos=min(c['final_cos_min'] for c in cases),
                arms=[dict(name='cpu-hosted-portable', dtype=dtype, device='cpu', build_mode='portable',
                           cpu=bm.get('cpu'), gguf_sha256=r.get('gguf_sha256'),
                           file_mib=r.get('census', {}).get('file_mib'),
                           tensor_data_sha256=r.get('census', {}).get('tensor_data_sha256'),
                           worst_stage=worst, cases=cases)])


def table(paths):
    rows = ['| Arm | MiB | Final cos min (worst case) | Decoded exact | First divergence |', '|---|---|---|---|---|']
    for p in paths:
        r = json.loads(Path(p).read_text())
        if 'failed' in r:
            rows.append(f"| {r['arm']} | - | failed: {r['failed']} | - | - |"); continue
        div = ', '.join(f"{c['id']}@{c['first_divergence']}" for c in r['cases'] if c['first_divergence'] is not None)
        mib = r.get('census', {}).get('file_mib', '-')
        rows.append(f"| {r['arm']} | {mib} | {r['cos_min']:.4f} ({r['worst_case']}) | {r['decoded_exact_cases']}/5 | {div or '-'} |")
    return '\n'.join(rows)


def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == 'fetch':
        fetch(*args)
    elif cmd == 'original':
        original(*args)
    elif cmd == 'argv':
        print(recipe_argv(*args))
    elif cmd == 'env':
        r = json.loads(Path(args[0]).read_text())['recipes'][args[1]]
        print(' '.join(f'{k}={v}' for k, v in sorted(r.get('env', {}).items())))
    elif cmd == 'census':
        print(json.dumps(census(*args)))
    elif cmd == 'score':
        ref, orig, arm, d = args[:4]
        res = score(ref, orig, arm, d)
        for extra in args[4:]:
            k, v = extra.split('=', 1)
            res[k] = json.loads(Path(v).read_text()) if v.endswith('.json') else v
        print(json.dumps(res, indent=2))
    elif cmd == 'receipt':
        print(json.dumps(receipt(*args), indent=2))
    elif cmd == 'stages':
        print(json.dumps(stages(*args), indent=2))
    elif cmd == 'table':
        print(table(args))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
