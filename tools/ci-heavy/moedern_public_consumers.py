#!/usr/bin/env python3
"""Hosted MoEdern1.5M public-text proof on the actual root library (heavy-cpu.yml).

Authenticate immutable reviewed producers -> byte-preserving pack -> root build
(LLAMAFILE ON) -> actual C/Python/Rust/Dart consumers on raw text -> exact IDs,
three masks and expansion, old 20 vectors and 3 MaxSim scores under unchanged
0.002/0.99999 gates, cross-consumer and repeat identity. No original recapture.
"""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
GATES = {g: '1' for g in ('CRISPEMBED_MOEDERN_PUBLIC_TEXT', 'CRISPEMBED_MOEDERN_PACKED', 'CRISPEMBED_MOEDERN_TEXT_COMPOSED',
                          'CRISPEMBED_MOEDERN_TEXT_FRONTEND', 'CRISPEMBED_MOEDERN_COLBERT')}
LIMITS = dict(max_absolute=0.002, min_row_cosine=0.99999, maxsim_max_absolute=0.002)
COMPOSED = dict(run=40455093591, source='1a053116908224cf1c9358d7d24c85ba36e82eff', id=12711284127, bytes=84683,
                sha256='a5ba93289e7fa3fa7cde8e3b587f4cc0934ae87d7a73fa6b2423d3878a67d5bd',
                report_sha256='98af43c6780bbd11b1ec9ec4f26c7b2e3a2348507072408408bf726b2e89482e')


def require(ok, why):
    if not ok: raise ValueError(why)


def sha(data): return hashlib.sha256(data).hexdigest()
def write(path, value): path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
def api(path): return json.loads(subprocess.check_output(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/'+path], timeout=90))


def download(run, source, pin, scratch, cap):
    row = api('actions/runs/'+str(run))
    require((row['head_sha'], row['conclusion'], row['run_attempt']) == (source, 'success', 1), 'producer run differs')
    a = api('actions/artifacts/'+str(pin['id']))
    require(a['workflow_run']['id'] == run and a['size_in_bytes'] == pin['bytes'] and a['digest'] == 'sha256:'+pin['sha256']
            and not a['expired'] and 0 < pin['bytes'] <= cap, 'artifact identity differs')
    path = scratch/(str(pin['id'])+'.zip')
    with path.open('xb') as s: subprocess.run(['gh', 'api', f'repos/CrispStrobe/CrispEmbed/actions/artifacts/{pin["id"]}/zip'], stdout=s, check=True, timeout=600)
    require(sha(path.read_bytes()) == pin['sha256'], 'ZIP bytes differ')
    z = zipfile.ZipFile(path)
    for i in z.infolist():
        p = PurePosixPath(i.filename)
        require(not p.is_absolute() and '..' not in p.parts and '\\' not in i.filename and (i.external_attr >> 16) & 0o170000 != 0o120000, 'unsafe member')
    return z, {k: a[k] for k in ('id', 'name', 'size_in_bytes', 'digest', 'expires_at')}


def prepare(scratch, inputs):
    """Every immutable input, authenticated before any build or forward."""
    cfg = json.loads(subprocess.check_output(['git', 'show', COMPOSED['source']+':docs/moedern-text-composed.json'], cwd=ROOT))
    proof = {}
    z, proof['composed'] = download(COMPOSED['run'], COMPOSED['source'], COMPOSED, scratch, 4*2**20)
    hits = [n for n in z.namelist() if sha(z.read(n)) == COMPOSED['report_sha256']]
    require(len(hits) == 1, 'reviewed composition report member not unique'); (inputs/'composed.json').write_bytes(z.read(hits[0]))
    u = cfg['unigram']
    z, proof['tokenizer'] = download(u['run'], u['source'], u['metadata'], scratch, 64*2**20)
    component, gold = z.read(u['component_member']), z.read(u['gold_member'])
    require(sha(component) == u['component_sha256'], 'tokenizer component differs')
    (inputs/'tokenizer.gguf').write_bytes(component)
    p = cfg['primary']
    z, proof['primary_review'] = download(p['run'], p['source'], p['review_artifact'], scratch, 2**20)
    data = z.read('moedern-original/review/original.json'); require(sha(data) == p['receipt_sha256'], 'primary receipt differs')
    report = json.loads(data)
    z, proof['primary_raw'] = download(p['run'], p['source'], p['raw_artifact'], scratch, 32*2**20)
    model = z.read('moedern-original/model-F32.gguf'); require(sha(model) == p['gguf_sha256'], 'old F32 model differs')
    (inputs/'model.gguf').write_bytes(model)
    vectors = {}
    for row in report['stage_arrays']:
        if row['stage'] not in ('ordinary.features.token_embeddings', 'ordinary.selected'): continue
        payload = z.read('moedern-original/'+row['path']); require(sha(payload) == row['sha256'] and len(payload) == row['bytes'], 'old vector differs')
        vectors[(row['case'], row['stage'])] = (row['shape'], payload)
    require(len(vectors) == 20, 'twenty old vectors required')
    encoded = {c['id']: c['encoded'] for c in json.loads(gold)['cases'][:10]}
    require(list(encoded) == [c['id'] for c in report['cases']], 'tokenizer/model lineage differs')
    return report, vectors, encoded, proof


def floats(shape, payload): return [list(struct.unpack_from('<%df' % shape[1], payload, 4*shape[1]*r)) for r in range(shape[0])]


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


def judge(outputs, report, vectors, encoded):
    cases = {r['id']: r for r in outputs if r['kind'] == 'case'}; scores = {r['id']: r['score'] for r in outputs if r['kind'] == 'score'}
    require(list(cases) == list(encoded) and len(scores) == 3, 'complete case/pair coverage required')
    result = dict(cases={}, scores={}, failures=[])
    for cid, row in cases.items():
        e = encoded[cid]
        typed = dict(ids=row['ids'] == e['ids'], supplied_mask=row['supplied_mask'] == e['supplied_mask'],
                     effective_mask=row['effective_mask'] == e['effective_mask'], scoring_mask=row['scoring_mask'] == e['scoring_mask'],
                     expansion_positions=row['expansion_positions'] == e['expansion_positions'])
        all_m = metrics(rows(row['all_vectors']), floats(*vectors[(cid, 'ordinary.features.token_embeddings')]))
        sel_m = metrics(rows(row['selected_vectors']), floats(*vectors[(cid, 'ordinary.selected')]))
        result['cases'][cid] = dict(typed=typed, all_vectors=all_m, selected_vectors=sel_m)
        if not all(typed.values()) or not all_m['passed'] or not sel_m['passed']: result['failures'].append(cid)
    for i, pair in enumerate(report['scores']):
        pid = pair['query']+'|'+pair['document']; native = scores[pid]
        independent = maxsim(rows(cases[pair['query']]['selected_vectors']), rows(cases[pair['document']]['selected_vectors']))
        ok = abs(native-pair['original_maxsim']) <= LIMITS['maxsim_max_absolute'] and abs(native-independent) <= LIMITS['maxsim_max_absolute']
        result['scores'][pid] = dict(native=native, original=pair['original_maxsim'], independent=independent, passed=ok)
        if not ok: result['failures'].append(pid)
    result['passed'] = not result['failures']
    return result


def canonical(outputs):
    """Float32-exact comparison key across consumers with different printers."""
    keys = ('kind', 'id', 'ids', 'supplied_mask', 'effective_mask', 'scoring_mask', 'expansion_positions')
    out = []
    for r in outputs:
        row = {k: r[k] for k in keys if k in r}
        for k in ('all_vectors', 'selected_vectors'):
            if k in r: row[k] = [f32(x) for x in r[k]]
        if 'score' in r: row['score'] = f32(r['score'])
        out.append(row)
    return out


def run(cmd, log, cwd=ROOT, env=None, timeout=1800):
    with open(log, 'w') as err:
        r = subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=err, timeout=timeout, text=True)
    return r.returncode, r.stdout


def main():
    out, scratch = Path(os.environ['HEAVY_OUT']), Path(os.environ['HEAVY_SCRATCH'])/'moedern'
    require(os.environ.get('GITHUB_ACTIONS') == 'true' or bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE')), 'hosted only')
    import importlib.util
    spec = importlib.util.spec_from_file_location('root_guard', ROOT/'tools/moedern-public/root_contract.py')
    guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard); closure = guard.source()
    scratch.mkdir(parents=True); inputs = scratch/'inputs'; inputs.mkdir()
    report = dict(schema='crispembed-moedern-public-consumers-v1', source=os.environ.get('GITHUB_SHA'), closure_base=closure['base'],
                  completed=False, public_model_proof=False, default_enabled=False, release_qualified=False, model_quality=False, limits=LIMITS)
    try:
        primary, vectors, encoded, report['service'] = prepare(scratch, inputs)
        import pack_moedern_text_f32 as packer
        packed = scratch/'moedern-1.5m-text-f32.gguf'
        report['pack'] = packer.pack(inputs/'model.gguf', inputs/'tokenizer.gguf', inputs/'composed.json', packed)
        report['pack']['output'].pop('tensors', None); report['pack']['tensor_count'] = 32
        write(out/'pack.json', report['pack'])
        hexed = lambda s: s.encode('utf-8').hex() or '-'
        cases, pairs = scratch/'cases.tsv', scratch/'pairs.tsv'
        cases.write_text(''.join(f"{c['id']}\t{c['role']}\t{hexed(c['text'])}\n" for c in primary['cases']))
        texts = {c['id']: c['text'] for c in primary['cases']}
        pairs.write_text(''.join(f"{s['query']}|{s['document']}\t{hexed(texts[s['query']])}\t{hexed(texts[s['document']])}\n" for s in primary['scores']))
        build = scratch/'build'
        for name, cmd in [('configure', ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DCRISPEMBED_BUILD_SHARED=ON',
                                         '-DBUILD_SHARED_LIBS=OFF', '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF', '-DGGML_BLAS=OFF',
                                         '-DGGML_METAL=OFF', '-DGGML_CUDA=OFF', '-DGGML_LLAMAFILE=ON']),
                          ('build', ['cmake', '--build', build, '--target', 'crispembed-shared', 'crispembed-cli', 'crispembed-server', '--parallel', '2']),
                          ('c-compile', ['cc', '-O2', '-I', ROOT/'src', ROOT/'tests/moedern-public/consumer.c', '-L', build, '-lcrispembed',
                                         '-Wl,-rpath,'+str(build), '-o', scratch/'c-consumer'])]:
            code, _ = run(cmd, out/(name+'.log')); require(code == 0, name+' failed')
        lib = build/'libcrispembed.so'
        rust = scratch/'rust'; (rust/'src').mkdir(parents=True)
        (rust/'Cargo.toml').write_text('[package]\nname="moedern-consumer"\nversion="0.0.0"\nedition="2021"\n[dependencies]\ncrispembed={path="'+str(ROOT/'crispembed')+'"}\n[workspace]\n')
        shutil.copyfile(ROOT/'tests/moedern-public/rust_consumer.rs', rust/'src/main.rs')
        dart = scratch/'dart'; (dart/'lib').mkdir(parents=True); (dart/'bin').mkdir()
        shutil.copyfile(ROOT/'flutter/crispembed/lib/src/moedern_text.dart', dart/'lib/moedern_text.dart')
        shutil.copyfile(ROOT/'tests/moedern-public/dart_consumer.dart', dart/'bin/main.dart')
        (dart/'pubspec.yaml').write_text('name: moedern_audit\nenvironment:\n  sdk: ">=3.0.0 <4.0.0"\ndependencies:\n  ffi: 2.1.4\n')
        dart_bin = os.environ.get('DART', 'dart')
        env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
        env.update(GATES, CRISPEMBED_OFFLINE='1', OMP_NUM_THREADS='2', CRISPEMBED_SYS_LIB_DIR=str(build), LD_LIBRARY_PATH=str(build))
        # Gate-off control first: the actual library must reject with every model gate unset.
        off = {k: v for k, v in env.items() if k not in GATES}
        code, _ = run([scratch/'c-consumer', packed, cases, pairs], out/'gate-off.log', env=off)
        require(code != 0, 'gate-off library accepted the model'); report['gate_off_rejected'] = True
        run([dart_bin, 'pub', 'get'], out/'dart-pub.log', cwd=dart, env=env)
        consumers = {
            'c': [scratch/'c-consumer', packed, cases, pairs],
            'c-repeat': [scratch/'c-consumer', packed, cases, pairs],
            'python': [sys.executable, ROOT/'tests/moedern-public/python_consumer.py', packed, lib, cases, pairs, ROOT/'python'],
            'rust': ['cargo', 'run', '--release', '--quiet', '--manifest-path', rust/'Cargo.toml', '--', packed, cases, pairs],
            'dart': [dart_bin, 'run', 'bin/main.dart', packed, lib, cases, pairs]}
        results, raw = {}, {}
        for name, cmd in consumers.items():
            code, stdout = run(cmd, out/(name+'.log'), cwd=dart if name == 'dart' else ROOT, env=env, timeout=3600)
            (out/(name+'.jsonl')).write_text(stdout) if name in ('c',) else None
            if code != 0: results[name] = dict(exit=code, passed=False); write(out/'consumers.json', report | dict(consumers=results)); continue
            raw[name] = [json.loads(l) for l in stdout.splitlines() if l.strip()]
            results[name] = dict(exit=0, **judge(raw[name], primary, vectors, encoded))
            write(out/'consumers.json', report | dict(consumers=results))
        # Actual CLI and HTTP arms on the same packed file and raw texts.
        texts_by_id = {c['id']: (c['role'], c['text']) for c in primary['cases']}
        try:
            rows = []
            for cid, (role, txt) in texts_by_id.items():
                code, stdout = run([build/'crispembed', '-m', packed, '--moedern', role, txt], out/'cli.log', env=env)
                require(code == 0, 'CLI encode failed: '+cid); rows.append(dict(json.loads(stdout), kind='case', id=cid))
            for pair in primary['scores']:
                code, stdout = run([build/'crispembed', '-m', packed, '--moedern-score', texts[pair['query']], texts[pair['document']]], out/'cli.log', env=env)
                require(code == 0, 'CLI score failed'); rows.append(dict(kind='score', id=pair['query']+'|'+pair['document'], score=json.loads(stdout)['score']))
            # CLI prints only selected vectors; judge them and reuse C's all-vectors only for the shape check.
            for r in rows:
                if r['kind'] == 'case': r['all_vectors'] = next(x for x in raw['c'] if x.get('id') == r['id'])['all_vectors']
            raw['cli'] = rows; results['cli'] = dict(exit=0, **judge(rows, primary, vectors, encoded))
        except Exception as e: results['cli'] = dict(passed=False, error=str(e))
        import urllib.request, time as _t
        server = subprocess.Popen([build/'crispembed-server', '--moedern-model', packed, '--port', '18431'], env=env,
                                  stdout=open(out/'http.log', 'w'), stderr=subprocess.STDOUT)
        try:
            def post(path, body):
                req = urllib.request.Request('http://127.0.0.1:18431'+path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=120) as r: return json.load(r)
            deadline = _t.time()+60
            while True:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18431/health/ready', timeout=2) as r:
                        if json.load(r)['status'] == 'ready': break
                except OSError: pass
                require(server.poll() is None and _t.time() < deadline, 'server not ready'); _t.sleep(.3)
            with urllib.request.urlopen('http://127.0.0.1:18431/capabilities', timeout=5) as r:
                require(json.load(r)['routes']['moedern'] is True, 'moedern route not advertised')
            try: post('/moedern/encode', {'text': 'x', 'role': 'other'}); require(False, 'invalid role accepted')
            except urllib.error.HTTPError as e: require(e.code == 400, 'unexpected status')
            rows = [dict(post('/moedern/encode', {'text': txt, 'role': role}), kind='case', id=cid) for cid, (role, txt) in texts_by_id.items()]
            rows += [dict(kind='score', id=p['query']+'|'+p['document'], score=post('/moedern/score', {'query': texts[p['query']], 'document': texts[p['document']]})['score']) for p in primary['scores']]
            raw['http'] = rows; results['http'] = dict(exit=0, **judge(rows, primary, vectors, encoded))
        except Exception as e: results['http'] = dict(passed=False, error=str(e))
        finally: server.terminate(); server.wait(timeout=15)
        report['consumers'] = results
        report['repeat_byte_identical'] = 'c' in raw and raw.get('c-repeat') == raw['c']
        if 'c' in raw:
            report['cross_consumer_float32_identical'] = {n: canonical(raw[n]) == canonical(raw['c']) for n in raw}
        passed = (all(r.get('passed') for r in results.values()) and report['repeat_byte_identical']
                  and all(report['cross_consumer_float32_identical'].values()))
        report.update(completed=True, all_consumers_parity=passed)
        write(out/'consumers.json', report)
        lines = [f"- {n}: {'PASS' if r.get('passed') else 'FAIL'}" for n, r in results.items()]
        (out/'summary.md').write_text('### MoEdern1.5M public consumers\n'+'\n'.join(lines)+'\n')
        return int(not passed)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error).replace(str(ROOT), '<caller>'))
        write(out/'consumers.json', report); raise


if __name__ == '__main__': sys.exit(main())
