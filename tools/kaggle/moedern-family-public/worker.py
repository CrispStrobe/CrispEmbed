#!/usr/bin/env python3
"""Hosted MoEdern 27M/115M public-text proof on the actual root library (Kaggle CPU).

Modelled on tools/ci-heavy/moedern_public_consumers.py (1.5M). Inputs are the
frozen original Kaggle capture (mounted read-only via kernel_sources) and the
reviewed Unigram tokenizer component (sha256-pinned, identical for every size).
Authenticate -> byte-preserving family pack -> root build (LLAMAFILE ON) ->
actual C/Python/CLI consumers on the 10 frozen raw-text cases and 3 MaxSim pairs
-> exact IDs, three masks and expansion; unchanged 0.002/0.99999 vector and 0.002
MaxSim gates; gate-off rejection and repeat identity. No original recapture.

Usage: worker.py VARIANT MOUNT OUT
"""
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tools'))
import pack_moedern_text_f32 as packer  # noqa: E402
BASE_GATES = ('CRISPEMBED_MOEDERN_PUBLIC_TEXT', 'CRISPEMBED_MOEDERN_PACKED', 'CRISPEMBED_MOEDERN_TEXT_COMPOSED',
              'CRISPEMBED_MOEDERN_TEXT_FRONTEND', 'CRISPEMBED_MOEDERN_COLBERT')
FAMILY_GATE = 'CRISPEMBED_MOEDERN_FAMILY'
LIMITS = dict(max_absolute=0.002, min_row_cosine=0.99999, maxsim_max_absolute=0.002)


def require(ok, why):
    if not ok: raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(1 << 20): h.update(b)
    return h.hexdigest()


def write(path, value): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def flag(v): return [int(x) for x in (json.loads(v.replace('True', 'true').replace('False', 'false')) if isinstance(v, str) else v)]


def locate(mount, variant):
    """Frozen original capture root and the pinned tokenizer component."""
    pin = packer.FAMILY[variant]
    roots = [Path(p).parent.parent for p in glob.glob(str(mount/'**/review/original.json'), recursive=True)
             if sha(p) == pin['original_report_sha256']]
    require(len(roots) == 1, 'exactly one pinned original report required')
    root = roots[0]
    tokenizers = [Path(p) for p in glob.glob(str(mount/'**/*.gguf'), recursive=True)
                  if Path(p).stat().st_size == packer.TOKENIZER['bytes'] and sha(p) == packer.TOKENIZER['sha256']]
    require(tokenizers, 'pinned tokenizer component (sha256 %s) not mounted' % packer.TOKENIZER['sha256'])
    return root, tokenizers[0]


def frozen(root, report):
    vectors = {}
    for row in report['stage_arrays']:
        if row['stage'] not in ('ordinary.features.token_embeddings', 'ordinary.selected'): continue
        path = root/row['path']
        if 'sha256' in row: require(sha(path) == row['sha256'], 'frozen array differs: '+row['path'])
        data = path.read_bytes(); n = len(data)//4; v = struct.unpack('<%df' % n, data)
        vectors[(row['case'], row['stage'])] = [list(v[i:i+128]) for i in range(0, n, 128)]
    require(len(vectors) == 2*len(report['cases']) == 20, 'twenty frozen vectors required')
    expected = {}
    for c in report['cases']:
        supplied = flag(c['supplied_attention_mask'])
        expected[c['id']] = dict(ids=[int(x) for x in c['token_ids']], supplied_mask=supplied,
                                 effective_mask=flag(c['effective_attention_mask']), scoring_mask=flag(c['scoring_mask']),
                                 expansion_positions=[1-x for x in supplied] if c['role'] == 'query' else [])
    require(len(report['scores']) == 3, 'three MaxSim pairs required')
    return vectors, expected


def metrics(native, original):
    require(len(native) == len(original) and all(len(a) == 128 for a in native), 'vector shape differs')
    worst, cosine = 0.0, 1.0
    for a, b in zip(native, original):
        worst = max(worst, max(abs(x-y) for x, y in zip(a, b)))
        na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(y*y for y in b))
        cosine = min(cosine, sum(x*y for x, y in zip(a, b))/(na*nb) if na and nb else float(na == nb))
    return dict(max_absolute=worst, min_row_cosine=cosine, passed=worst <= LIMITS['max_absolute'] and cosine >= LIMITS['min_row_cosine'])


def maxsim(q, d): return sum(max(sum(x*y for x, y in zip(a, b)) for b in d) for a in q)
def rows(flat): return [flat[i:i+128] for i in range(0, len(flat), 128)]
def f32(v): return struct.unpack('<f', struct.pack('<f', v))[0]


def judge(outputs, report, vectors, expected):
    cases = {r['id']: r for r in outputs if r['kind'] == 'case'}; scores = {r['id']: r['score'] for r in outputs if r['kind'] == 'score'}
    require(list(cases) == list(expected) and len(scores) == 3, 'complete case/pair coverage required')
    result = dict(cases={}, scores={}, failures=[])
    for cid, row in cases.items():
        e = expected[cid]
        typed = {k: row[k] == e[k] for k in e}
        try:
            all_m = metrics(rows(row['all_vectors']), vectors[(cid, 'ordinary.features.token_embeddings')])
            sel_m = metrics(rows(row['selected_vectors']), vectors[(cid, 'ordinary.selected')])
        except ValueError as error:
            all_m = sel_m = dict(passed=False, error=str(error))
        result['cases'][cid] = dict(typed=typed, all_vectors=all_m, selected_vectors=sel_m)
        if not all(typed.values()) or not all_m['passed'] or not sel_m['passed']: result['failures'].append(cid)
    for pair in report['scores']:
        pid = pair['query']+'|'+pair['document']; native = scores[pid]
        independent = maxsim(rows(cases[pair['query']]['selected_vectors']), rows(cases[pair['document']]['selected_vectors']))
        ok = abs(native-pair['original_maxsim']) <= LIMITS['maxsim_max_absolute'] and abs(native-independent) <= LIMITS['maxsim_max_absolute']
        result['scores'][pid] = dict(native=native, original=pair['original_maxsim'], independent=independent, passed=ok)
        if not ok: result['failures'].append(pid)
    result['passed'] = not result['failures']
    return result


def canonical(outputs):
    keys = ('kind', 'id', 'ids', 'supplied_mask', 'effective_mask', 'scoring_mask', 'expansion_positions')
    out = []
    for r in outputs:
        row = {k: r[k] for k in keys if k in r}
        for k in ('all_vectors', 'selected_vectors'):
            if k in r: row[k] = [f32(x) for x in r[k]]
        if 'score' in r: row['score'] = f32(r['score'])
        out.append(row)
    return out


def run(cmd, log, env=None, timeout=3600):
    with open(log, 'w') as err:
        r = subprocess.run(list(map(str, cmd)), cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=err, timeout=timeout, text=True)
    return r.returncode, r.stdout


def main(variant, mount, out):
    require(variant in packer.FAMILY, 'variant must be one of '+', '.join(packer.FAMILY))
    scratch = Path(os.environ.get('MOEDERN_SCRATCH', '/kaggle/temp/moedern-family')); scratch.mkdir(parents=True, exist_ok=True)
    report = dict(schema='crispembed-moedern-family-public-consumers-v1', variant=variant, source=os.environ.get('MOEDERN_COMMIT'),
                  completed=False, public_model_proof=False, default_enabled=False, release_qualified=False, model_quality=False, limits=LIMITS)
    try:
        root, tokenizer = locate(mount, variant)
        original = json.loads((root/'review/original.json').read_text())
        require(original['variant'] == variant and original['revision'] == packer.FAMILY[variant]['revision'], 'original variant differs')
        vectors, expected = frozen(root, original)
        packed = scratch/('moedern-%s-text-f32.gguf' % variant.lower())
        if packed.exists(): packed.unlink()
        report['pack'] = packer.pack_family(variant, root/'model-F32.gguf', tokenizer, root/'review/original.json', packed)
        report['pack']['output'].pop('tensors', None)
        write(out/'pack.json', report['pack'])
        hexed = lambda s: s.encode('utf-8').hex() or '-'
        texts = {c['id']: c['text'] for c in original['cases']}
        cases, pairs = scratch/'cases.tsv', scratch/'pairs.tsv'
        cases.write_text(''.join(f"{c['id']}\t{c['role']}\t{hexed(c['text'])}\n" for c in original['cases']))
        pairs.write_text(''.join(f"{s['query']}|{s['document']}\t{hexed(texts[s['query']])}\t{hexed(texts[s['document']])}\n" for s in original['scores']))
        build = scratch/'build'
        for name, cmd in [('configure', ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DCRISPEMBED_BUILD_SHARED=ON',
                                         '-DBUILD_SHARED_LIBS=OFF', '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF', '-DGGML_BLAS=OFF',
                                         '-DGGML_METAL=OFF', '-DGGML_CUDA=OFF', '-DGGML_LLAMAFILE=ON']),
                          ('build', ['cmake', '--build', build, '--target', 'crispembed-shared', 'crispembed-cli', '--parallel', '4']),
                          ('c-compile', ['cc', '-O2', '-I', ROOT/'src', ROOT/'tests/moedern-public/consumer.c', '-L', build, '-lcrispembed',
                                         '-Wl,-rpath,'+str(build), '-o', scratch/'c-consumer'])]:
            code, _ = run(cmd, out/(name+'.log'), timeout=7200); require(code == 0, name+' failed')
        lib = build/'libcrispembed.so'
        env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
        env.update({g: '1' for g in BASE_GATES + (FAMILY_GATE,)}, CRISPEMBED_OFFLINE='1', OMP_NUM_THREADS='2',
                   CRISPEMBED_SYS_LIB_DIR=str(build), LD_LIBRARY_PATH=str(build))
        # Gate-off controls: every model gate unset, then only the family gate unset.
        off_all = {k: v for k, v in env.items() if k not in BASE_GATES + (FAMILY_GATE,)}
        off_family = {k: v for k, v in env.items() if k != FAMILY_GATE}
        report['gate_off_rejected'] = {}
        for name, e in (('all', off_all), ('family', off_family)):
            code, _ = run([scratch/'c-consumer', packed, cases, pairs], out/('gate-off-'+name+'.log'), env=e)
            report['gate_off_rejected'][name] = code != 0
        require(all(report['gate_off_rejected'].values()), 'gate-off library accepted the family model')
        consumers = {
            'c': [scratch/'c-consumer', packed, cases, pairs],
            'c-repeat': [scratch/'c-consumer', packed, cases, pairs],
            'python': [sys.executable, ROOT/'tests/moedern-public/python_consumer.py', packed, lib, cases, pairs, ROOT/'python']}
        results, raw = {}, {}
        for name, cmd in consumers.items():
            code, stdout = run(cmd, out/(name+'.log'), env=env)
            if code != 0: results[name] = dict(exit=code, passed=False); continue
            raw[name] = [json.loads(l) for l in stdout.splitlines() if l.strip()]
            results[name] = dict(exit=0, **judge(raw[name], original, vectors, expected))
            write(out/'consumers.json', report | dict(consumers=results))
        try:
            rows_ = []
            for c in original['cases']:
                code, stdout = run([build/'crispembed', '-m', packed, '--moedern', c['role'], c['text']], out/'cli.log', env=env)
                require(code == 0, 'CLI encode failed: '+c['id']); rows_.append(dict(json.loads(stdout), kind='case', id=c['id']))
            for p in original['scores']:
                code, stdout = run([build/'crispembed', '-m', packed, '--moedern-score', texts[p['query']], texts[p['document']]], out/'cli.log', env=env)
                require(code == 0, 'CLI score failed'); rows_.append(dict(kind='score', id=p['query']+'|'+p['document'], score=json.loads(stdout)['score']))
            # CLI prints selected vectors only; all_vectors come from C for the shape check (as in the 1.5M worker).
            for r in rows_:
                if r['kind'] == 'case': r['all_vectors'] = next(x for x in raw['c'] if x.get('id') == r['id'])['all_vectors']
            raw['cli'] = rows_; results['cli'] = dict(exit=0, **judge(rows_, original, vectors, expected))
        except Exception as e: results['cli'] = dict(passed=False, error=str(e))
        report['consumers'] = results
        report['repeat_byte_identical'] = 'c' in raw and raw.get('c-repeat') == raw['c']
        report['cross_consumer_float32_identical'] = {n: canonical(raw[n]) == canonical(raw['c']) for n in raw} if 'c' in raw else {}
        passed = (all(r.get('passed') for r in results.values()) and report['repeat_byte_identical']
                  and bool(report['cross_consumer_float32_identical']) and all(report['cross_consumer_float32_identical'].values()))
        report.update(completed=True, all_consumers_parity=passed)
        write(out/'consumers.json', report)
        print(json.dumps({k: report[k] for k in ('variant', 'all_consumers_parity', 'gate_off_rejected', 'repeat_byte_identical')}, indent=1))
        return int(not passed)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error).replace(str(ROOT), '<caller>'))
        write(out/'consumers.json', report); raise


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])))
