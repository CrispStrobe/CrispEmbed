#!/usr/bin/env python3
"""Portfolio Q4_K lane for MoEdern 1.5M/27M/115M (heavy-cpu.yml, hosted CPU only).

  heavy-cpu.yml -f script=tools/ci-heavy/moedern_q4k_lanes.py \
      -f args="115M f32 q8 q4 q8+attn q8+exps ..." -f pip="numpy==2.2.6 gguf==0.17.1"

For one variant: authenticate the frozen ORIGINAL token vectors (report SHA pinned
in docs/moedern-q4k-padded.json, every array SHA from that report), the published
self-contained F32 package, build tests/moedern-q4k (GGML_NATIVE=OFF, portable),
then run each requested arm through the actual C consumer on the frozen ten-case
raw-text panel and compare to the ORIGINAL model:

  f32   published F32 (control; FAST family graph for 27M/115M as in the padded panel)
  q8    named Q8_0/F32-critical profile (quantize_q8_critical.py, CRISPEMBED_MOEDERN_Q8_0)
  q4    plain padded Q4_K profile (quantize_q4k_padded.py, CRISPEMBED_MOEDERN_Q4_K_PADDED)
  other mixed recipe/policy (quantize_q4k_mixed.py, + CRISPEMBED_MOEDERN_Q4_K_MIXED)

Metric (docs/precision-acceptance.md): final output = every token's 128-d ColBERT
vector; `final_cos_min` is the minimum row cosine over every token of every case
(all tokens and the selected scoring vectors), plus per-case flattened cosine,
MaxSim absolute error and exact IDs/masks. Strict 0.002/0.99999 flags are recorded.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/moedern-colbert'))
import quantize_q4k_padded as plain  # noqa: E402
import quantize_q4k_mixed as mixed  # noqa: E402
import quantize_q8_critical as q8  # noqa: E402

FIXTURES = ('https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/resolve/'
            '3528982465132c835ea0fc7c0b8b4b90ea99ecae/moedern-q4k-originals/')
GATES = ('CRISPEMBED_MOEDERN_PUBLIC_TEXT', 'CRISPEMBED_MOEDERN_PACKED', 'CRISPEMBED_MOEDERN_TEXT_COMPOSED',
         'CRISPEMBED_MOEDERN_TEXT_FRONTEND', 'CRISPEMBED_MOEDERN_COLBERT', 'CRISPEMBED_MOEDERN_FAMILY')
STAGES = ('ordinary.features.token_embeddings', 'ordinary.selected')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    return plain.sha_file(path)


def fetch(url, dest, sha256=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r, dest.open('wb') as f:
        while block := r.read(1 << 20):
            f.write(block)
    require(sha256 is None or sha(dest) == sha256, 'download SHA differs: ' + dest.name)


def originals(variant, root):
    pin = json.loads((ROOT / 'docs/moedern-q4k-padded.json').read_text())['originals'][variant]
    report_path = root / 'review/original.json'
    fetch(FIXTURES + variant + '/review/original.json', report_path, pin['report_sha256'])
    report = json.loads(report_path.read_text())
    arrays = {}
    for row in report['stage_arrays']:
        if row['stage'] not in STAGES:
            continue
        rel = PurePosixPath(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'unsafe array path')
        path = root / rel
        fetch(FIXTURES + variant + '/' + row['path'], path, row['sha256'])
        arrays[row['case'], row['stage']] = (path, row['shape'])
    require(len(arrays) == 2 * len(report['cases']) == 20, 'twenty original vectors required')
    return report, arrays


def build(out, build_dir):
    log = out / 'build.log'
    with log.open('w') as f:
        for args in (['cmake', '-S', ROOT / 'tests/moedern-q4k', '-B', build_dir, '-DCMAKE_BUILD_TYPE=Release',
                      '-DGGML_NATIVE=OFF'],
                     ['cmake', '--build', build_dir, '--parallel', str(os.cpu_count() or 2)]):
            subprocess.run(list(map(str, args)), stdout=f, stderr=subprocess.STDOUT, check=True, timeout=3600)
    cache = (build_dir / 'CMakeCache.txt').read_text()
    require('GGML_NATIVE:BOOL=OFF' in cache, 'portable build required')
    cpu = ''
    try:
        cpu = next(l.split(':', 1)[1].strip() for l in subprocess.check_output(['lscpu'], text=True).splitlines()
                   if l.startswith('Model name'))
    except Exception:
        cpu = platform.processor()
    return dict(mode='portable (GGML_NATIVE=OFF, GGML_LLAMAFILE=ON, -ffp-contract=off)', cpu=cpu,
                heavy_cpu_build_mode=os.environ.get('CRISPEMBED_BUILD_MODE', 'unset (pre-#120 workflow; -DGGML_NATIVE=OFF passed explicitly)'),
                nproc=os.cpu_count(), source=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                                                       text=True).strip())


def metrics(a, b):
    import numpy as np
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    n1, n2 = np.linalg.norm(a64, axis=1), np.linalg.norm(b64, axis=1)
    cos = np.divide((a64 * b64).sum(axis=1), n1 * n2, out=np.ones_like(n1), where=(n1 * n2) != 0)
    flat = float((a64 * b64).sum() / (np.linalg.norm(a64) * np.linalg.norm(b64)))
    d = np.abs(a64 - b64)
    return dict(rows=int(a.shape[0]), min_row_cosine=float(cos.min()), mean_row_cosine=float(cos.mean()),
                worst_row=int(cos.argmin()), flattened_cosine=flat, max_absolute=float(d.max()),
                strict_passed=bool(d.max() <= 0.002 and cos.min() >= 0.99999))


def evaluate(name, model, env, report, arrays, consumer, cases, pairs, raw):
    import numpy as np
    start = time.monotonic()
    run = subprocess.run([str(consumer), str(model), str(cases), str(pairs)], env=env, text=True,
                         capture_output=True, timeout=1800)
    seconds = time.monotonic() - start
    (raw / (name.replace('+', 'P').replace('/', '_') + '.stderr')).write_text(run.stderr)
    if run.returncode:
        return dict(completed=False, error=run.stderr.strip()[-500:], exit=run.returncode)
    out = [json.loads(l) for l in run.stdout.splitlines() if l.strip()]
    cs = [r for r in out if r['kind'] == 'case']
    ss = [r for r in out if r['kind'] == 'score']
    require([r['id'] for r in cs] == [c['id'] for c in report['cases']] and len(ss) == 3, 'ten cases/three scores')
    per_case, ids_exact, worst = {}, True, (2.0, None)
    for target, native in zip(report['cases'], cs):
        same = (native['ids'] == target['token_ids']
                and native['supplied_mask'] == list(map(int, target['supplied_attention_mask']))
                and native['scoring_mask'] == list(map(int, target['scoring_mask'])))
        ids_exact &= same
        row = dict(ids_exact=same)
        for kind, stage in (('all', STAGES[0]), ('selected', STAGES[1])):
            path, shape = arrays[target['id'], stage]
            a = np.fromfile(path, dtype='<f4').reshape(shape)
            b = np.asarray(native[kind + '_vectors'], dtype='<f4')
            require(b.size == a.size, 'vector extent differs: ' + target['id'])
            row[kind] = m = metrics(a, b.reshape(shape))
            if m['min_row_cosine'] < worst[0]:
                worst = (m['min_row_cosine'], target['id'] + '/' + kind)
        per_case[target['id']] = row
    byid = {r['id']: r for r in cs}
    scores = []
    for target, native in zip(report['scores'], ss):
        q = np.asarray(byid[target['query']]['selected_vectors'], dtype=np.float64).reshape(-1, 128)
        d = np.asarray(byid[target['document']]['selected_vectors'], dtype=np.float64).reshape(-1, 128)
        scores.append(dict(id=native['id'], native=float(native['score']), original=target['original_maxsim'],
                           independent=float((q @ d.T).max(axis=1).sum()),
                           absolute_error=abs(float(native['score']) - target['original_maxsim'])))
    return dict(completed=True, seconds=seconds, final_cos_min=worst[0], worst_case=worst[1],
                flattened_cos_min=min(min(c['all']['flattened_cosine'], c['selected']['flattened_cosine'])
                                      for c in per_case.values()),
                ids_exact=ids_exact, maxsim_max_abs=max(s['absolute_error'] for s in scores),
                strict_passed=all(c[k]['strict_passed'] for c in per_case.values() for k in ('all', 'selected')),
                cases=per_case, scores=scores)


def main():
    require(os.environ.get('GITHUB_ACTIONS') == 'true', 'hosted execution required')
    variant, arms = sys.argv[1], sys.argv[2:] or ['f32', 'q8', 'q4']
    require(variant in plain.PROFILES, 'variant must be one of ' + ','.join(plain.PROFILES))
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH'])
    raw = out / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    result = dict(schema='crispembed-moedern-q4k-lane-v1', variant=variant, arms={}, completed=False,
                  reference='frozen original model token vectors (docs/moedern-q4k-padded.json pins)',
                  bars={'Q8_0': 0.97, 'Q4_K': 0.94}, publication_qualified=False)
    save = lambda: (out / ('lane-' + variant + '.json')).write_text(json.dumps(result, indent=2) + '\n')  # noqa
    result['build'] = build(out, scratch / 'build')
    save()
    report, arrays = originals(variant, scratch / 'original')
    pub = plain.publication(variant)
    f32 = scratch / 'published-f32.gguf'
    fetch(pub['url'], f32, pub['sha256'])
    result['f32_model'] = dict(url=pub['url'], sha256=pub['sha256'], bytes=pub['bytes'])
    hx = lambda s: s.encode().hex() or '-'  # noqa: E731
    text = {c['id']: c['text'] for c in report['cases']}
    cases, pairs = raw / 'cases.tsv', raw / 'pairs.tsv'
    cases.write_text(''.join(c['id'] + '\t' + c['role'] + '\t' + hx(c['text']) + '\n' for c in report['cases']))
    pairs.write_text(''.join(s['query'] + '|' + s['document'] + '\t' + hx(text[s['query']]) + '\t'
                             + hx(text[s['document']]) + '\n' for s in report['scores']))
    b = scratch / 'build'
    consumer, quantizer = b / 'moedern-q4k-consumer', b / 'moedern-q4k-quantize'
    env0 = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    env0.update({g: '1' for g in GATES}, OMP_NUM_THREADS='2')
    quant = dict(env0, CRISPEMBED_MOEDERN_FAST='1', CRISPEMBED_MOEDERN_QUANT='1')
    for arm in arms:
        model, conv = None, None
        try:
            if arm == 'f32':
                model, env = f32, dict(env0, **({} if variant == '1.5M' else {'CRISPEMBED_MOEDERN_FAST': '1'}))
            elif arm == 'q8':
                model = scratch / 'm-q8.gguf'
                conv = q8.quantize(f32, model, variant, quantizer)
                env = dict(quant, CRISPEMBED_MOEDERN_Q8_0='1')
            elif arm == 'q4':
                model = scratch / 'm-q4.gguf'
                conv = plain.quantize(f32, model, variant, quantizer)
                env = dict(quant, CRISPEMBED_MOEDERN_Q4_K_PADDED='1')
            else:
                model = scratch / ('m-' + hashlib.sha256(arm.encode()).hexdigest()[:12] + '.gguf')
                conv = mixed.quantize(f32, model, variant, quantizer, arm)
                env = dict(quant, CRISPEMBED_MOEDERN_Q4_K_PADDED='1', CRISPEMBED_MOEDERN_Q4_K_MIXED='1')
            row = evaluate(arm, model, env, report, arrays, consumer, cases, pairs, raw)
            row['bytes'] = model.stat().st_size
            if conv:
                row['conversion'] = {k: conv.get(k) for k in ('output_sha256', 'output_bytes', 'counts',
                                                              'bytes_by_type', 'recipe_spec', 'policy',
                                                              'q4k_tensors', 'f32_tensors', 'actual_q8_tensors')}
            if arm != 'f32':
                for leftover in scratch.glob(model.name + '*'):
                    leftover.unlink()
        except Exception as error:  # keep every failure on record and continue the sweep
            row = dict(completed=False, error=type(error).__name__ + ': ' + str(error)[:500])
        result['arms'][arm] = row
        save()
    result['completed'] = True
    save()
    q8b = result['arms'].get('q8', {}).get('bytes')
    lines = ['| arm | MiB | vs Q8_0 | final cos min (worst) | flat cos min | MaxSim max abs | IDs | s |',
             '|---|---|---|---|---|---|---|---|']
    for arm, r in result['arms'].items():
        if not r.get('completed'):
            lines.append(f"| {arm} | - | - | FAILED: {r.get('error','')[:80]} | | | | |")
            continue
        rel = f"{r['bytes'] / q8b:.3f}" if q8b else '-'
        lines.append(f"| {arm} | {r['bytes'] / 2**20:.2f} | {rel} | {r['final_cos_min']:.4f} ({r['worst_case']}) | "
                     f"{r['flattened_cos_min']:.4f} | {r['maxsim_max_abs']:.4f} | {r['ids_exact']} | {r['seconds']:.2f} |")
    (out / 'summary.md').write_text(f"## MoEdern {variant} Q4_K lane ({result['build']['mode']}; "
                                    f"{result['build']['cpu']})\n\n" + '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
