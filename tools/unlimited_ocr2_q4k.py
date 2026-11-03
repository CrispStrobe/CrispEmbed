#!/usr/bin/env python3
"""Unlimited-OCR-2 Q4_K recipe helper (docs/unlimited-ocr2-q4k-recipes.md).

Subcommands (all used by tools/ci-heavy/unlimited_ocr2_q4k_sweep.sh):
  render <pages-dir>              the frozen trial renderer, with matplotlib's DejaVu fonts (the
                                  fonts of the original Kaggle capture) rather than system fonts
  cases <pages-dir> <out-dir>     verify the rendered PNGs against the frozen original capture
                                  (tests/results/unlimited-ocr2/trial.json) and write <case>.ids
  args <recipe>                   crispembed-quantize argv tail (base type + --tensor-type rules)
  census <gguf> <out.json>        stored-type census, bytes and the protected-tensor check
  cosine <ref.f32> <cand.f32> <n_ids> <out.json>
                                  per-position cosine of teacher-forced logits (V = 129280)
  free <stderr-log> <case> <out.json>   free-running gen_ids vs the original IDs
  summary <out-dir>               markdown table over every recipe receipt in <out-dir>
"""
import glob
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'tests/results/unlimited-ocr2/trial.json'
RECIPES = ROOT / 'tools/unlimited_ocr2_quant_recipes.json'
CASES = ['paragraphs', 'table', 'receipt', 'german']
VOCAB = 129280
# Tensors that must keep their source floating type and payload in every recipe.
PROTECTED = ['v.pos_embed', 'v.patch_embed.weight', 'c.patch_embed.weight', 'v.image_newline',
             'v.view_separator', 'c.cls_token', 'l.output_norm.weight']
TYPE_NAMES = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0', 10: 'Q2_K',
              11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 20: 'IQ4_NL', 23: 'IQ4_XS', 30: 'BF16'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def frozen():
    trial = json.loads(TRIAL.read_text())
    refs = {c['id']: c for c in trial['reference']['cases'] if c['id'] in CASES}
    pages = {p['id']: p['sha256'] for p in trial['pages'] if p['id'] in CASES}
    out = {}
    for case in CASES:
        ids = refs[case]['ids']
        ids = json.loads(ids) if isinstance(ids, str) else ids
        out[case] = dict(ids=[int(x) for x in ids], text=refs[case]['text'], png_sha256=pages[case])
    return out


def cmd_cases(pages, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = {}
    ok = True
    for case, ref in frozen().items():
        got = sha(Path(pages) / f'{case}.png')
        rows[case] = dict(png_sha256=got, frozen_png_sha256=ref['png_sha256'], match=got == ref['png_sha256'],
                          original_ids=len(ref['ids']))
        ok &= got == ref['png_sha256']
        (outdir / f'{case}.ids').write_text(' '.join(map(str, ref['ids'])) + '\n')
    json.dump(dict(cases=rows, all_match=ok), sys.stdout, indent=1)
    print()
    return 0 if ok else 1


def cmd_render(outdir):
    import importlib.util
    import matplotlib
    fonts = Path(matplotlib.get_data_path()) / 'fonts/ttf'
    spec = importlib.util.spec_from_file_location(
        'trial_worker', ROOT / 'tools/kaggle/unlimited-ocr2-trial/trial_worker.py')
    tw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tw)

    def font(size, mono=False):
        from PIL import ImageFont
        path = str(fonts / ('DejaVuSansMono.ttf' if mono else 'DejaVuSans.ttf'))
        return ImageFont.truetype(path, size), path

    tw.font = font
    tw.render(outdir)
    return 0


def cmd_args(name):
    r = json.loads(RECIPES.read_text())[name]
    print(r['type'])
    for rule in r['rules']:
        print('--tensor-type')
        print(rule)
    return 0


def inventory(path):
    from gguf import GGUFReader
    rows = {}
    for t in GGUFReader(str(path)).tensors:
        rows[t.name] = dict(type=int(t.tensor_type), shape=[int(x) for x in t.shape], bytes=int(t.n_bytes),
                            sha256=hashlib.sha256(memoryview(t.data).cast('B')).hexdigest()
                            if t.name in PROTECTED else None)
    return rows


def semantic(shape):
    s = list(shape)
    while len(s) > 1 and s[-1] == 1:
        s.pop()
    return s


def cmd_census(path, out, source=None):
    rows = inventory(path)
    counts, nbytes = {}, {}
    for r in rows.values():
        n = TYPE_NAMES.get(r['type'], str(r['type']))
        counts[n] = counts.get(n, 0) + 1
        nbytes[n] = nbytes.get(n, 0) + r['bytes']
    rec = dict(file=os.path.basename(path), file_bytes=os.path.getsize(path),
               file_mib=round(os.path.getsize(path) / 2**20, 1), tensors=len(rows), type_counts=counts,
               type_bytes=nbytes,
               types={k: TYPE_NAMES.get(v['type'], v['type']) for k, v in rows.items() if '.blk.' not in k or
                      k.startswith(('l.blk.0.', 'l.blk.1.', 'l.blk.11.'))})
    if source:
        src = inventory(source)
        prot = {}
        for name in PROTECTED:
            a, b = src.get(name), rows.get(name)
            if a is None:
                continue
            prot[name] = dict(type=TYPE_NAMES.get(b['type']), source_shape=a['shape'], stored_shape=b['shape'],
                              byte_exact=a['type'] == b['type'] and a['sha256'] == b['sha256'],
                              semantic_shape_equal=semantic(a['shape']) == semantic(b['shape']))
        rec['protected'] = prot
        rec['protected_ok'] = all(v['byte_exact'] and v['semantic_shape_equal'] for v in prot.values())
        rec['q4_0_tensors'] = sorted(k for k, v in rows.items() if v['type'] == 2)
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(f"{rec['file']}: {rec['file_mib']} MiB {counts} protected_ok={rec.get('protected_ok')}")
    return 0 if rec.get('protected_ok', True) else 1


def cmd_cosine(ref, cand, n_ids, out):
    import numpy as np
    a = np.fromfile(ref, dtype=np.float32)
    b = np.fromfile(cand, dtype=np.float32)
    n = int(n_ids) + 1  # every forced ID plus the position that should predict EOS
    ok = a.size == n * VOCAB and b.size == n * VOCAB
    rec = dict(positions=n, ref_floats=int(a.size), cand_floats=int(b.size), complete=bool(ok))
    if ok:
        a = a.reshape(n, VOCAB).astype(np.float64)
        b = b.reshape(n, VOCAB).astype(np.float64)
        cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        top = (a.argmax(1) == b.argmax(1))
        rec.update(final_cos_min=float(cos.min()), final_cos_mean=float(cos.mean()),
                   worst_position=int(cos.argmin()), top1_agreement=float(top.mean()),
                   top1_disagreements=[int(i) for i in np.nonzero(~top)[0]][:20],
                   max_abs=float(np.abs(a - b).max()))
    Path(out).write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(rec))
    return 0 if ok else 1


def cmd_free(log, case, out):
    ref = frozen()[case]['ids']
    m = re.search(r'gen_ids \(\d+\):([^\n]*)', Path(log).read_text(errors='replace'))
    ids = [int(x) for x in m.group(1).split()] if m else None
    first = None
    if ids is not None:
        first = next((i for i, (x, y) in enumerate(zip(ids, ref)) if x != y),
                     None if len(ids) == len(ref) else min(len(ids), len(ref)))
    rec = dict(case=case, n_ids=len(ids) if ids else None, original_n=len(ref), ids_exact=ids == ref,
               first_difference=first, ids=ids)
    Path(out).write_text(json.dumps(rec) + '\n')
    print(f"{case}: ids_exact={rec['ids_exact']} first_diff={first} n={rec['n_ids']}/{len(ref)}")
    return 0


def cmd_summary(outdir):
    outdir = Path(outdir)
    lines = ['| recipe | MiB | final cos min (case) | per case | top1 agree min | free-run IDs exact | wall s |',
             '|---|---|---|---|---|---|---|']
    names = sorted({Path(p).name.split('.')[0] for p in glob.glob(str(outdir / '*.census.json'))})
    names = ['f16'] + [n for n in names if n != 'f16']
    for n in names:
        cen = outdir / f'{n}.census.json'
        mib = json.loads(cen.read_text())['file_mib'] if cen.exists() else '-'
        per = {}
        for c in CASES:
            f = outdir / f'{n}.{c}.cosine.json'
            if f.exists():
                per[c] = json.loads(f.read_text())
        mins = [(v['final_cos_min'], c) for c, v in per.items() if v.get('complete')]
        worst = min(mins) if mins else None
        top = min((v['top1_agreement'] for v in per.values() if v.get('complete')), default=None)
        free = []
        for c in CASES:
            f = outdir / f'{n}.{c}.free.json'
            if f.exists():
                free.append(c if json.loads(f.read_text())['ids_exact'] else f'{c}x')
        wall = outdir / f'{n}.wall'
        lines.append(f"| {n} | {mib} | {f'{worst[0]:.4f} ({worst[1]})' if worst else '-'} | "
                     f"{' '.join(f'{c}:{v:.4f}' for v, c in sorted(mins, key=lambda x: CASES.index(x[1])))} | "
                     f"{f'{top:.4f}' if top is not None else '-'} | "
                     f"{sum(not x.endswith('x') for x in free)}/{len(free) if free else '-'} | "
                     f"{wall.read_text().strip() if wall.exists() else '-'} |")
    print('\n'.join(lines))
    return 0


if __name__ == '__main__':
    c, a = sys.argv[1], sys.argv[2:]
    fn = {'render': cmd_render, 'cases': cmd_cases, 'args': cmd_args, 'census': cmd_census, 'cosine': cmd_cosine, 'free': cmd_free,
          'summary': cmd_summary}[c]
    sys.exit(fn(*a))
