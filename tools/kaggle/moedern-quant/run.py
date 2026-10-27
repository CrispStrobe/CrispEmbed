#!/usr/bin/env python3
"""MoEdern 27M/115M quantized-weight diagnostic (hosted only: Kaggle).

  run.py PROBE MOUNT OUT

For every frozen original panel under MOUNT (review/original.json + model-F32.gguf),
quantizes the F32 GGUF to Q8_0 and Q4_0 (tools/moedern-colbert/quantize_gguf.py) and
runs the family probe on the fast path for F32, Q8_0 and Q4_0 at 2 and 4 threads:

* accuracy: tests/moedern-family/compare.py (same frozen IDs/masks, max-abs and min
  row cosine vs the frozen original vectors, MaxSim differences). F32-fast must pass
  the unchanged 0.002 / 0.99999 gate; quantized arms are reported, not gated.
* speed: REPS probe runs over the document cases only; docs/s = documents / median
  total encode seconds.

Exit 0 only if every F32-fast arm passes and every quantized arm completes.
"""
import glob
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools/moedern-colbert'))
from quantize_gguf import quantize  # noqa: E402

REPS = int(os.environ.get('MOEDERN_QUANT_REPS', '5'))
PRECISIONS = ('f32', 'q8_0', 'q4_0')
THREADS = (2, 4)


def worst(cases, part, key, pick):
    vals = [c[part][key] for c in cases.values() if part in c]
    return pick(vals) if vals else None


def env_for(precision, threads):
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_MOEDERN')}
    env.update(CRISPEMBED_MOEDERN_FAMILY='1', CRISPEMBED_MOEDERN_COLBERT='1', CRISPEMBED_MOEDERN_FAST='1',
               MOEDERN_PROBE_THREADS=str(threads))
    if precision != 'f32':
        env['CRISPEMBED_MOEDERN_QUANT'] = '1'
    return env


def speed(probe, model, docs_tsv, n_docs, scratch, env, threads):
    totals = []
    for _ in range(REPS):
        r = subprocess.run([str(probe), str(model), str(docs_tsv), str(scratch), str(threads)], env=env,
                           capture_output=True, text=True)
        if r.returncode:
            return dict(error=r.stderr[-2000:])
        totals += [float(l.split()[1]) for l in r.stdout.splitlines() if l.startswith('total_encode_seconds')]
    med = statistics.median(totals)
    return dict(reps=totals, median_seconds=med, documents=n_docs, docs_per_second=n_docs / med)


def panel(probe, original, out):
    root = original.parent.parent
    report = json.loads(original.read_text())
    variant = report['variant']
    vout = out / variant
    vout.mkdir(parents=True, exist_ok=True)
    models = {'f32': root / 'model-F32.gguf'}
    quant = {}
    for kind in ('q8_0', 'q4_0'):
        models[kind] = vout / ('model-%s.gguf' % kind.upper())
        rep = quantize(str(models['f32']), str(models[kind]), kind)
        rep['bytes'] = models[kind].stat().st_size
        quant[kind] = rep
    (vout / 'quantization.json').write_text(json.dumps(quant, indent=1) + '\n')
    docs = [c for c in report['cases'] if c['role'] != 'query']
    docs_tsv = vout / 'docs.tsv'
    docs_tsv.write_text(''.join('%s\t%s\t%s\t%s\n' % (c['id'], c['role'], ','.join(map(str, c['token_ids'])),
                                                        ','.join(map(str, c['supplied_attention_mask']))) for c in docs))
    (vout / 'speed-native').mkdir(exist_ok=True)
    arms, ok = {}, True
    for precision in PRECISIONS:
        for threads in THREADS:
            arm = '%s-t%d' % (precision, threads)
            env = env_for(precision, threads)
            env['MOEDERN_PROBE_MODEL'] = str(models[precision])
            subprocess.run([sys.executable, str(ROOT / 'tests/moedern-family/compare.py'), str(probe), str(root),
                            str(vout / arm)], env=env)
            path = vout / arm / 'parity.json'
            if not path.exists():
                arms[arm] = dict(completed=False)
                ok = False
                continue
            res = json.loads(path.read_text())
            row = dict(precision=precision, threads=threads, completed=res['probe_exit'] == 0,
                       parity_gate_passed=res['passed'],
                       all_max_absolute=worst(res['cases'], 'all', 'max_absolute', max),
                       all_min_row_cosine=worst(res['cases'], 'all', 'min_row_cosine', min),
                       selected_max_absolute=worst(res['cases'], 'selected', 'max_absolute', max),
                       selected_min_row_cosine=worst(res['cases'], 'selected', 'min_row_cosine', min),
                       maxsim_max_abs_diff=max((abs(s['native'] - s['original']) for s in res['scores']), default=None),
                       speed=speed(probe, models[precision], docs_tsv, len(docs), vout / 'speed-native', env, threads))
            ok = ok and row['completed'] and 'error' not in row['speed'] and (precision != 'f32' or res['passed'])
            arms[arm] = row
    for threads in THREADS:
        base = arms.get('f32-t%d' % threads, {}).get('speed', {}).get('docs_per_second')
        for precision in PRECISIONS:
            a = arms.get('%s-t%d' % (precision, threads), {})
            if base and a.get('speed', {}).get('docs_per_second'):
                a['speedup_vs_f32_fast'] = a['speed']['docs_per_second'] / base
    for kind in ('q8_0', 'q4_0'):
        quant[kind].pop('tensors')
    return variant, dict(revision=report['revision'], quantization=quant, arms=arms, passed=ok)


def main(probe, mount, out):
    out.mkdir(parents=True, exist_ok=True)
    originals = sorted(Path(p) for p in glob.glob(str(mount / '**/review/original.json'), recursive=True))
    summary = dict(schema='crispembed-moedern-quant-v1', reps=REPS, commit=os.environ.get('MOEDERN_COMMIT', ''),
                   variants={})
    for original in originals:
        variant, res = panel(probe, original, out)
        summary['variants'][variant] = res
    summary['passed'] = bool(originals) and all(v['passed'] for v in summary['variants'].values())
    (out / 'quant.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=1))
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
