#!/usr/bin/env python3
"""Josev Q4_K sensitivity sweep against the frozen original reference (hosted CPU).

Owner bar (docs/precision-acceptance.md): final-output cosine against the original
model, minimum over every frozen case, Q8_0 > 0.97 and Q4_K (incl. mixed) > 0.94.
For Josev the final output is the per-case head output that produces the typed
response; this sweep reports, per case:

* ``logits_cos``: cosine of the concatenated per-field head logits (the judged
  metric, the vector the typed probabilities are a softmax of),
* ``probabilities_cos`` and maxima of the per-field probabilities,
* ``final_norm_cos``: global cosine of the backbone final-norm hidden state
  (seq x 2048), the stricter representation-level diagnostic used for v4,
* the worst strict stage (minimum row cosine) and typed-response exactness.

Cosines are derived from the actual comparator's per-stage moments
(original/native L2 and RMS of the difference), the same derivation used by the
v4 review. The original reference is the frozen v5 capture; no original forward
runs. Every recipe keeps the 292 critical tensors F32 and stores projections as
real Q8_0/Q4_K (the C quantizer rejects any fallback). The josev-q4k CMake
project forces GGML_NATIVE=OFF (portable build); lscpu is recorded.
"""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import josev2b_q4k_contract as contract
import quantize_josev2b_protected as prot
import josev2b_composed_panel as panel

spec = importlib.util.spec_from_file_location('josev_frozen_q4_worker', ROOT / 'tools/kaggle/josev-q4k/worker.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
require = contract.require
RECIPES = json.loads((ROOT / 'tools/josev2b_q4k_recipes.json').read_text())['recipes']
TYPE_ID = {'f32': 0, 'f16': 1, 'q8_0': 8, 'q4_k': 12}
ROLE_SUFFIX = {
    'attn': ('.self_attn.q_proj.weight', '.self_attn.k_proj.weight', '.self_attn.v_proj.weight',
             '.self_attn.o_proj.weight', '.linear_attn.in_proj_qkv.weight', '.linear_attn.in_proj_z.weight',
             '.linear_attn.out_proj.weight'),
    'gate': ('.mlp.gate_proj.weight',), 'up': ('.mlp.up_proj.weight',), 'down': ('.mlp.down_proj.weight',),
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n')


def plan(rows, identity):
    """Per-tensor storage: critical F32, projections Q8_0 unless a selector lowers them."""
    projections = prot.old.recipe(rows)
    result = {}
    for name in rows:
        if projections[name] == 0:
            result[name] = 0
            continue
        kind = 8
        layer = int(name.split('.')[2])
        for rule in RECIPES[identity]:
            lo, hi = rule.get('layers', [0, 23])
            if lo <= layer <= hi and any(name.endswith(s) for r in rule['roles'] for s in ROLE_SUFFIX[r]):
                kind = TYPE_ID[rule['type']]
        result[name] = kind
    require(sum(v == 0 for v in result.values()) == 292 and sum(v != 0 for v in result.values()) == 150,
            'critical/projection inventory differs')
    return result


def quantize(source, output, executable, identity, rows):
    wanted = plan(rows, identity)
    counts = {t: sum(v == t for v in wanted.values()) for t in prot.TYPES}
    # Reuse the protected-recipe verifier (F32 byte identity, finite scales, typed
    # metadata) with this sweep's plan. The header is Q8_0 only without Q4_K.
    tag = ('q8-' if counts[12] == 0 else 'sweep-') + identity
    prot.COUNTS[tag] = counts
    prot.recipe = lambda r, i, _w=wanted: _w
    method = 'q8_0' if counts[12] == 0 else 'q4_k'
    target = TYPE_ID[method]
    args = [str(executable), str(source), str(output), method,
            '--metadata-string', 'josev2b.packed.format=' + prot.FORMAT, '--standard-file-type']
    for name, kind in wanted.items():
        if kind != target:
            args += ['--tensor-type', name + '=' + prot.TYPES[kind].lower()]
    log = output.with_suffix('.quantize.log')
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    from ci_process import run
    process = run(args, log, env=env, timeout=2400)
    require(not process['timed_out'] and process['exit'] == 0, 'C quantizer rejected ' + identity)
    require('(fallback ' not in log.read_text(), 'quantizer fallback forbidden')
    verified = prot.verify_conversion(source, output, rows, tag, vector=False)
    verified.pop('tensors')
    return dict(identity=identity, counts={prot.TYPES[t]: n for t, n in counts.items()}, method=method,
                quantize_seconds=process['seconds'], output_sha256=prot.digest(output),
                output_mib=round(output.stat().st_size / 2**20, 1), **verified)


def moments(m):
    a, b = m['original_l2'], m['native_l2']
    return (a * a + b * b - m['elements'] * m['rms'] ** 2) / 2, a * a, b * b


def final_outputs(comparison, typed):
    cases = []
    exact = {c['id']: c['complete_typed_json_exact'] for c in typed['cases']}
    argmax = {c['id']: all(f['original_argmax'] == f['native_argmax'] for f in c['fields']) for c in typed['cases']}
    for c in comparison['cases']:
        row = dict(id=c['id'])
        for kind in ('logits', 'probabilities'):
            d = x = y = 0.0
            stages = [m for m in c['metrics'] if m['stage'].startswith(kind + '/')]
            for m in stages:
                p, q, r = moments(m)
                d, x, y = d + p, x + q, y + r
            row[kind + '_cos'] = d / math.sqrt(x * y)
            row[kind + '_max_abs'] = max(m['max_absolute'] for m in stages)
        fn = next(m for m in c['metrics'] if m['stage'] == 'backbone/final_norm')
        p, q, r = moments(fn)
        row['final_norm_cos'] = p / math.sqrt(q * r)
        row['final_norm_min_row_cos'] = fn['min_row_cosine']
        worst = min(c['metrics'], key=lambda m: m['min_row_cosine'])
        row['worst_stage'] = dict(stage=worst['stage'], min_row_cosine=worst['min_row_cosine'])
        row['typed_json_exact'] = exact[c['id']]
        row['argmax_exact'] = argmax[c['id']]
        cases.append(row)
    summary = {k: min(r[k] for r in cases) for k in ('logits_cos', 'probabilities_cos', 'final_norm_cos')}
    summary['probabilities_max_abs'] = max(r['probabilities_max_abs'] for r in cases)
    summary['typed_json_exact'] = sum(r['typed_json_exact'] for r in cases)
    summary['argmax_exact'] = sum(r['argmax_exact'] for r in cases)
    return cases, summary


def fetch_reference(destination):
    spec2 = importlib.util.spec_from_file_location('josev_frozen_fetch', ROOT / 'tools/ci-heavy/josev2b_composed.py')
    helper = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(helper)
    pin = json.loads((ROOT / 'docs/josev2b-composed-dependencies.json').read_text())['fixed']['head']
    report = json.loads((ROOT / 'tests/results/josev2b/original-reference.json').read_text())
    names = {'review/artifact.json', 'original/reference.json', 'original/input-freeze.json'} | \
            {'original/' + r['file'] for r in report['arrays']}
    destination.mkdir(parents=True, exist_ok=False)
    directory, proof = helper.fetch(pin, destination, names)
    receipt = json.loads((directory / 'review/artifact.json').read_text())
    require(receipt['completed'] is True and receipt['arrays'] == 251 and receipt['reference_sha256'] == contract.REFERENCE_SHA,
            'original archive receipt differs')
    for name in names - {'review/artifact.json'}:
        path = directory / name
        require(path.is_file() and prot.digest(path) == receipt['files'][name]['sha256'], 'original member differs: ' + name)
    return proof


def lscpu():
    try:
        out = subprocess.check_output(['lscpu'], text=True)
        return {k.strip(): v.strip() for k, v in (l.split(':', 1) for l in out.splitlines() if ':' in l)
                if k.strip() in ('Model name', 'Architecture', 'CPU(s)', 'Flags')}
    except Exception as e:  # pragma: no cover
        return dict(error=str(e))


def main(a):
    a.scratch.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    review, raw = a.out / 'review', a.scratch / 'raw'
    review.mkdir(exist_ok=True)
    raw.mkdir(exist_ok=True)
    identities = a.recipes.split(',')
    require(all(i in RECIPES for i in identities), 'unknown recipe')
    result = dict(schema='crispembed-josev-q4k-sweep-v1', source=os.environ.get('GITHUB_SHA'),
                  build=dict(mode='portable', cmake='tests/josev-q4k forces GGML_NATIVE=OFF, GGML_LLAMAFILE=ON',
                             lscpu=lscpu(), platform=platform.platform()),
                  bars=dict(q8_0=0.97, q4_k=0.94, metric='min over 4 frozen cases of concatenated head-logit cosine; '
                            'final-norm global cosine reported as the stricter diagnostic'),
                  original_rerun=False, publication_qualified=False, recipes={})
    save = lambda: write(review / 'sweep.json', result)
    result['reference_artifact'] = fetch_reference(a.scratch / 'reference')
    mount = a.scratch / 'reference'
    freeze, gold, result['original'] = old.original(mount, a.scratch / 'original')
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    build = a.scratch / 'build'
    for name, cmd in [('configure', ['cmake', '-S', ROOT / 'tests/josev-q4k', '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                                     '-DGGML_NATIVE=OFF']),
                      ('build', ['cmake', '--build', build, '--parallel', str(os.cpu_count() or 2)])]:
        old.command(cmd, review / (name + '.log'), env)
    cache = (build / 'CMakeCache.txt').read_text()
    require('GGML_NATIVE:BOOL=OFF' in cache, 'portable build required')
    result['build']['ggml_native'] = 'OFF'
    save()
    model = a.scratch / 'published-f32.gguf'
    result['model'] = old.fetch_f32(model)
    requests = raw / 'requests.json'
    write(requests, [dict(id=r['id'], request_json=json.dumps(r['request'], ensure_ascii=False, allow_nan=False,
                                                              separators=(',', ':'))) for r in freeze['records']])
    enabled = dict(env, **{k: '1' for k in old.GATES})
    quant_env = dict(enabled, **{k: '1' for k in old.QUANT})
    rows = prot.schema()

    def execute(label, path, active):
        outputs, timing = {}, {}
        for mode in ('plain', 'observed', 'repeat'):
            destination = raw / (label + '-' + mode)
            outputs[mode] = destination
            exe = build / ('josev-q4k-' + ('plain' if mode == 'plain' else 'observed') + '-probe')
            proc = old.command([exe, path, requests, destination], review / (label + '-' + mode + '.log'), active, allowed=(0, 1))
            require(proc['exit'] == 0, 'native run failed: ' + label)
            timing[mode + '_seconds'] = proc['seconds']
        comparison = panel.compare(a.scratch / 'original', freeze, gold, outputs, build / 'josev-q4k-diff',
                                   raw / (label + '-comparison'))
        write(review / (label + '-panel.json'), comparison)
        typed = old.typed_decisions(freeze, gold, outputs['plain'])
        write(review / (label + '-typed.json'), typed)
        cases, summary = final_outputs(comparison, typed)
        for d in outputs.values():
            shutil.rmtree(d, ignore_errors=True)
        shutil.rmtree(raw / (label + '-comparison'), ignore_errors=True)
        return dict(cases=cases, summary=summary, timing=timing,
                    strict_stage_failures=sum(not m['passed'] for c in comparison['cases'] for m in c['metrics']))

    if not a.skip_f32:
        result['f32_control'] = execute('f32', model, enabled)
        require(result['f32_control']['summary']['logits_cos'] > 0.999999, 'F32 control failed')
        save()
    for identity in identities:
        entry = result['recipes'][identity] = dict(phase='convert')
        save()
        candidate = raw / ('josev2b-' + identity + '.gguf')
        try:
            entry['conversion'] = quantize(model, candidate, build / 'josev-q4k-quantize', identity, rows)
            entry.update(execute(identity, candidate, quant_env))
            s = entry['summary']
            bar = 0.94 if entry['conversion']['counts']['Q4_K'] else 0.97
            entry['bar'] = bar
            entry['passes_logits_bar'] = s['logits_cos'] > bar
            entry['passes_final_norm_bar'] = s['final_norm_cos'] > bar
            entry['phase'] = 'complete'
        except Exception as e:
            entry.update(phase='rejected', error=type(e).__name__ + ': ' + str(e))
        finally:
            for p in raw.glob('josev2b-' + identity + '*'):
                p.unlink()
        save()
    table = ['| recipe | Q4_K/Q8_0/F32 | MiB | logits cos min | final-norm cos min | typed exact | argmax | worst stage |',
             '|---|---|---|---|---|---|---|---|']
    for k, v in result['recipes'].items():
        if v.get('phase') != 'complete':
            table.append(f"| {k} | rejected: {v.get('error')} |||||||")
            continue
        c, s = v['conversion'], v['summary']
        worst = min(v['cases'], key=lambda r: r['worst_stage']['min_row_cosine'])['worst_stage']
        table.append(f"| {k} | {c['counts']['Q4_K']}/{c['counts']['Q8_0']}/{c['counts']['F32']} | {c['output_mib']} | "
                     f"{s['logits_cos']:.5f} | {s['final_norm_cos']:.5f} | {s['typed_json_exact']}/4 | {s['argmax_exact']}/4 | "
                     f"{worst['stage']} {worst['min_row_cosine']:.4f} |")
    (a.out / 'summary.md').write_text('\n'.join(table) + '\n')
    result['completed'] = True
    save()
    require(all(v.get('phase') == 'complete' for v in result['recipes'].values()), 'a recipe was rejected')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--recipes', required=True)
    p.add_argument('--scratch', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--skip-f32', action='store_true')
    main(p.parse_args())
