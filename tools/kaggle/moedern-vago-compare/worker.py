#!/usr/bin/env python3
"""MoEdern 27M/115M: CrispEmbed runtime vs the model author's (VAGOsolutions) runtime.

Hosted only (Kaggle CPU kernel). Diagnostic comparison, not a qualification:
no gate, budget, pin or default changes follow from it.

1. Pinned author files (HF revision + sha256 / git blob sha1 from
   docs/moedern-colbert-source-contract.json), ggml at the pinned commit, their
   engine built as shipped (GGML_NATIVE ON, forced in their CMake) and with a
   copy whose CMake forces GGML_NATIVE OFF.
2. CrispEmbed root library (Release, LLAMAFILE ON) portable and native, plus the
   byte-preserving F32 family pack from the frozen original capture.
3. OUTPUTS on the ten frozen texts and three MaxSim pairs (author F32/F16/Q8_0/Q4_0
   and ours F32 vs the frozen original and vs each other, token IDs).
4. SPEED on an identical deterministic workload (256 documents, 256 queries),
   2 and 4 threads, warm-up excluded, 3 timed repetitions, median.
5. "ours-fast" arms: the same CrispEmbed builds with CRISPEMBED_MOEDERN_FAST=1
   (single-graph family path, top-2 experts only, up to 64 threads); outputs at
   2 threads and speed at 2 and 4 threads. Its acceptance gate is the unchanged
   0.002 / 0.99999 vector and 0.002 MaxSim gate vs the frozen originals with
   identical token IDs; float differences from the default path are expected.

Usage: worker.py VARIANT MOUNT OUT
"""
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT/'tools'))
import compare_lib as lib  # noqa: E402
import pack_moedern_text_f32 as packer  # noqa: E402

GGML_REPO, GGML_COMMIT = 'https://github.com/ggml-org/ggml.git', '3612c0af93f2cfc8f048295300a61187723b12ae'
GATES = ('CRISPEMBED_MOEDERN_PUBLIC_TEXT', 'CRISPEMBED_MOEDERN_PACKED', 'CRISPEMBED_MOEDERN_TEXT_COMPOSED',
         'CRISPEMBED_MOEDERN_TEXT_FRONTEND', 'CRISPEMBED_MOEDERN_COLBERT', 'CRISPEMBED_MOEDERN_FAMILY')
THREADS = (2, 4)
REPS = int(os.environ.get('MOEDERN_VAGO_REPS', '3'))
CONFIG_TIMEOUT = int(os.environ.get('MOEDERN_VAGO_CONFIG_TIMEOUT', '5400'))
DEADLINE = time.time() + float(os.environ.get('MOEDERN_VAGO_BUDGET_SECONDS', str(10.5 * 3600)))
QUANTS = ('F32', 'F16', 'Q8_0', 'Q4_0')


def require(ok, why):
    if not ok: raise ValueError(why)


def public_worker():
    spec = importlib.util.spec_from_file_location('moedern_family_public_worker', ROOT/'tools/kaggle/moedern-family-public/worker.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def run(cmd, log, env=None, timeout=7200, cwd=None):
    with open(log, 'w') as err:
        r = subprocess.run(list(map(str, cmd)), cwd=cwd or ROOT, env=env, stdout=subprocess.PIPE, stderr=err, timeout=timeout, text=True)
    return r.returncode, r.stdout


def capture(cmd):
    try: return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e: return 'unavailable: %s' % e


def host():
    flags = ''
    try: flags = next(l for l in open('/proc/cpuinfo') if l.startswith('flags')).split(':', 1)[1].split()
    except Exception: pass
    model = capture(['sh', '-c', "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"])
    return dict(cpu=model, nproc=os.cpu_count(), machine=platform.machine(), python=platform.python_version(),
                simd={f: f in flags for f in ('avx', 'avx2', 'fma', 'f16c', 'avx512f', 'avx512bw', 'avx512vnni', 'avx_vnni')},
                cc=capture(['cc', '--version']).splitlines()[0:1], cmake=capture(['cmake', '--version']).splitlines()[0:1],
                lscpu=capture(['lscpu']))


def fetch_author(variant, dest, log):
    """Download every pinned author file at the pinned revision and verify it."""
    contract = json.loads((ROOT/'docs/moedern-colbert-source-contract.json').read_text())
    pin = contract['variants'][variant]['author_gguf']
    result = dict(repository=pin['repository'], revision=pin['revision'], files={})
    for name, want in pin['files'].items():
        path = dest/name; path.parent.mkdir(parents=True, exist_ok=True)
        url = 'https://huggingface.co/%s/resolve/%s/%s' % (pin['repository'], pin['revision'], name)
        for attempt in range(4):
            try:
                tmp = path.with_name(path.name + '.partial')
                with urllib.request.urlopen(url, timeout=600) as r, open(tmp, 'wb') as f: shutil.copyfileobj(r, f, 1 << 20)
                tmp.rename(path); break
            except Exception as e:
                log.write('download %s attempt %d: %s\n' % (name, attempt, e)); time.sleep(10 * (attempt + 1))
        require(path.is_file(), 'author download failed: ' + name)
        got = dict(bytes=path.stat().st_size, sha256=lib.sha256_file(path))
        ok = got['bytes'] == want['bytes'] and (got['sha256'] == want['sha256'] if want.get('sha256') else True)
        if want.get('git_blob_sha1'):
            got['git_blob_sha1'] = lib.git_blob_sha1(path); ok = ok and got['git_blob_sha1'] == want['git_blob_sha1']
        require(ok, 'author file differs from pin: ' + name)
        result['files'][name] = dict(got, verified=True)
    return result


def build_author(vago, ggml, scratch, native, out):
    name = 'author-' + ('native' if native else 'portable')
    src = scratch/(name + '-src'); shutil.rmtree(src, ignore_errors=True); shutil.copytree(vago/'engine', src)
    cm = src/'CMakeLists.txt'; text = cm.read_text()
    shipped = 'set(GGML_NATIVE ON CACHE BOOL "" FORCE)'
    require(shipped in text, 'author CMake no longer forces GGML_NATIVE ON')
    if not native: cm.write_text(text.replace(shipped, 'set(GGML_NATIVE OFF CACHE BOOL "" FORCE)'))
    build = scratch/(name + '-build')
    for step, cmd in [('configure', ['cmake', '-S', src, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DGGML_DIR=' + str(ggml)]),
                      ('build', ['cmake', '--build', build, '--target', 'ggml_infer_v10', '--parallel', '4'])]:
        code, _ = run(cmd, out/('%s-%s.log' % (name, step)), timeout=7200); require(code == 0, '%s %s failed' % (name, step))
    return build/'ggml_infer_v10'


def build_ours(scratch, native, out):
    name = 'ours-' + ('native' if native else 'portable'); flag = 'ON' if native else 'OFF'
    build = scratch/(name + '-build')
    for step, cmd in [('configure', ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release', '-DCRISPEMBED_BUILD_SHARED=ON',
                                     '-DBUILD_SHARED_LIBS=OFF', '-DGGML_NATIVE=' + flag, '-DCRISPEMBED_NATIVE=' + flag, '-DGGML_BLAS=OFF',
                                     '-DGGML_METAL=OFF', '-DGGML_CUDA=OFF', '-DGGML_LLAMAFILE=ON']),
                      ('build', ['cmake', '--build', build, '--target', 'crispembed-shared', '--parallel', '4']),
                      ('bench-compile', ['cc', '-O2', '-I', ROOT/'src', HERE/'bench_ours.c', '-L', build, '-lcrispembed',
                                         '-Wl,-rpath,' + str(build), '-o', build/'bench_ours'])]:
        code, _ = run(cmd, out/('%s-%s.log' % (name, step)), timeout=7200); require(code == 0, '%s %s failed' % (name, step))
    return build/'libcrispembed.so', build/'bench_ours'


def ours_env(build_dir, threads, fast=False):
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    env.update({g: '1' for g in GATES}, CRISPEMBED_OFFLINE='1', OMP_NUM_THREADS=str(threads),
               CRISPEMBED_SYS_LIB_DIR=str(build_dir), LD_LIBRARY_PATH=str(build_dir))
    if fast: env['CRISPEMBED_MOEDERN_FAST'] = '1'
    return env


def plain_env(threads): return dict(os.environ, OMP_NUM_THREADS=str(threads), TOKENIZERS_PARALLELISM='false')


def json_run(cmd, log, env, out_json=None, timeout=CONFIG_TIMEOUT):
    code, stdout = run(cmd, log, env=env, timeout=timeout)
    if code != 0: raise RuntimeError('exit %d (see %s)' % (code, Path(log).name))
    return json.loads(Path(out_json).read_text()) if out_json else json.loads(stdout)


# ----------------------------------------------------------------------------- outputs
def compare_outputs(original, vectors, author, ours):
    """author: {quant/build: runner outputs}; ours: {build: runner outputs}."""
    cases = {c['id']: c for c in original['cases']}
    res = dict(tokenization={}, vs_original={}, author_vs_ours={}, maxsim={}, ours_native_vs_portable={})
    ref_ours = ours.get('portable') or next(iter(ours.values()), None)
    for key, a in author.items():
        tok, vo, vs = {}, {}, {}
        for cid, c in cases.items():
            r = a['cases'][cid]; orig_ids = [int(x) for x in c['token_ids']]
            tok[cid] = lib.token_diff(r['ids'], orig_ids)
            sel = vectors[(cid, 'ordinary.selected')]; full = vectors[(cid, 'ordinary.features.token_embeddings')]
            row = dict(selected=lib.vector_metrics(r['vectors'], sel),
                       engine_rows_vs_original_token_embeddings=lib.vector_metrics(r['engine_rows'], full))
            if len(r['keep']) == len(full):  # author's own skiplist applied to original rows (only meaningful if IDs are equal)
                row['author_selection_of_original_rows'] = lib.vector_metrics(r['vectors'], lib.select(full, r['keep']))
            vo[cid] = row
            if ref_ours:
                o = ref_ours['cases'][cid]
                vs[cid] = dict(tokens=lib.token_diff(r['ids'], o['ids']),
                               selected=lib.vector_metrics(r['vectors'], lib.rows(o['selected_vectors'])),
                               engine_rows_vs_ours_all=lib.vector_metrics(r['engine_rows'], lib.rows(o['all_vectors'])))
        sc = {}
        for p in original['scores']:
            pid = p['query'] + '|' + p['document']
            s = lib.maxsim(a['cases'][p['query']]['vectors'], a['cases'][p['document']]['vectors'])
            sc[pid] = dict(author=s, original=p['original_maxsim'], abs_diff_original=abs(s - p['original_maxsim']))
            if ref_ours: sc[pid].update(ours=ref_ours['scores'][pid], abs_diff_ours=abs(s - ref_ours['scores'][pid]))
        res['tokenization']['author-' + key] = tok; res['vs_original']['author-' + key] = vo
        res['author_vs_ours'][key] = vs; res['maxsim']['author-' + key] = sc
    for key, o in ours.items():
        tok, vo, sc = {}, {}, {}
        for cid, c in cases.items():
            r = o['cases'][cid]
            tok[cid] = lib.token_diff(r['ids'], c['token_ids'])
            vo[cid] = dict(selected=lib.vector_metrics(lib.rows(r['selected_vectors']), vectors[(cid, 'ordinary.selected')]),
                           all_vs_original_token_embeddings=lib.vector_metrics(lib.rows(r['all_vectors']), vectors[(cid, 'ordinary.features.token_embeddings')]))
        for p in original['scores']:
            pid = p['query'] + '|' + p['document']; s = o['scores'][pid]
            sc[pid] = dict(ours=s, original=p['original_maxsim'], abs_diff_original=abs(s - p['original_maxsim']))
        res['tokenization']['ours-' + key] = tok; res['vs_original']['ours-' + key] = vo; res['maxsim']['ours-' + key] = sc
    if 'portable' in ours and 'native' in ours:
        res['ours_native_vs_portable'] = {cid: lib.vector_metrics(lib.rows(ours['native']['cases'][cid]['selected_vectors']),
                                                                  lib.rows(ours['portable']['cases'][cid]['selected_vectors'])) for cid in cases}
    res['digest'] = digest(res)
    return res


def _worst(metrics):
    vals = [m for m in metrics if m and m.get('max_absolute') is not None]
    return dict(max_absolute=max((m['max_absolute'] for m in vals), default=None),
                min_row_cosine=min((m['min_row_cosine'] for m in vals), default=None),
                row_count_mismatches=sum(1 for m in metrics if m and not m['row_count_match']))


def digest(res):
    out = {}
    for key, per in res['vs_original'].items():
        tok = res['tokenization'][key]
        out[key] = dict(token_ids_equal_original=sum(t['equal'] for t in tok.values()), cases=len(tok),
                        selected_vs_original=_worst([v['selected'] for v in per.values()]),
                        maxsim_max_abs_diff_original=max(s['abs_diff_original'] for s in res['maxsim'][key].values()))
    for key, per in res['author_vs_ours'].items():
        if per: out['author-%s_vs_ours' % key] = dict(token_ids_equal=sum(v['tokens']['equal'] for v in per.values()),
                                                     selected=_worst([v['selected'] for v in per.values()]),
                                                     engine_rows_vs_ours_all=_worst([v['engine_rows_vs_ours_all'] for v in per.values()]))
    return out


# ----------------------------------------------------------------------------- speed
def speed_row(meta, raw):
    s = lib.summarize_timings(raw['reps'], raw['n_documents'], raw['n_queries'], raw['document_tokens'], raw['query_tokens'])
    if 'engine_peak_rss_kib' in raw:
        rss = (raw['engine_peak_rss_kib'] + raw['python_peak_rss_kib']) / 1024
    else:
        rss = raw.get('python_peak_rss_kib', raw.get('process_peak_rss_kib', 0)) / 1024
    return dict(meta, summary=s, peak_rss_mb=rss, raw=raw)


def main(variant, mount, out):
    out.mkdir(parents=True, exist_ok=True)
    require(variant in packer.FAMILY, 'variant must be 27M or 115M')
    scratch = Path(os.environ.get('MOEDERN_SCRATCH', '/kaggle/temp/moedern-vago')); scratch.mkdir(parents=True, exist_ok=True)
    report = dict(schema='crispembed-moedern-vago-runtime-compare-v1', variant=variant, source=os.environ.get('MOEDERN_COMMIT'),
                  diagnostic_only=True, qualification=False, completed=False, ggml_commit=GGML_COMMIT, reps=REPS, threads=list(THREADS))
    jpath = out/'compare.json'
    try:
        report['host'] = host(); write(jpath, report)
        pw = public_worker()
        root, tokenizer = pw.locate(mount, variant)
        original = json.loads((root/'review/original.json').read_text())
        require(original['variant'] == variant and original['revision'] == packer.FAMILY[variant]['revision'], 'original variant differs')
        vectors, _ = pw.frozen(root, original)
        # 1. author files, ggml, author builds
        vago = scratch/'vago'
        with open(out/'download.log', 'w') as log: report['author_files'] = fetch_author(variant, vago, log)
        ggml = scratch/'ggml'
        if not ggml.exists():
            code, _ = run(['git', 'clone', '-q', '--filter=blob:none', GGML_REPO, ggml], out/'ggml-clone.log'); require(code == 0, 'ggml clone failed')
        code, _ = run(['git', '-C', ggml, 'checkout', '-q', GGML_COMMIT], out/'ggml-checkout.log'); require(code == 0, 'ggml checkout failed')
        report['ggml_head'] = capture(['git', '-C', str(ggml), 'rev-parse', 'HEAD']); require(report['ggml_head'] == GGML_COMMIT, 'ggml commit differs')
        engines = {'native': build_author(vago, ggml, scratch, True, out), 'portable': build_author(vago, ggml, scratch, False, out)}
        # 2. ours
        ours_builds = {'portable': build_ours(scratch, False, out), 'native': build_ours(scratch, True, out)}
        packed = scratch/('moedern-%s-text-f32.gguf' % variant.lower())
        if packed.exists(): packed.unlink()
        pack = packer.pack_family(variant, root/'model-F32.gguf', tokenizer, root/'review/original.json', packed)
        pack['output'].pop('tensors', None)
        proven = ROOT/('tests/results/moedern-colbert/public-consumers-%s.json' % variant.lower())
        pack['proven_sha256'] = json.loads(proven.read_text())['pack']['sha256'] if proven.exists() else None
        if pack['proven_sha256']: require(pack['output']['sha256'] == pack['proven_sha256'], 'packed model differs from the proven pack')
        report['ours_pack'] = pack; write(jpath, report)
        # inputs
        cases = [dict(id=c['id'], role=c['role'], text=c['text']) for c in original['cases']]
        (scratch/'cases.json').write_text(json.dumps(cases))
        (scratch/'pairs.json').write_text(json.dumps([dict(query=p['query'], document=p['document']) for p in original['scores']]))
        work = lib.make_workload([c['text'] for c in cases if c['role'] == 'document'], [c['text'] for c in cases if c['role'] == 'query'])
        (scratch/'workload.json').write_text(json.dumps(work))
        (scratch/'workload.tsv').write_text(''.join('%s\t%s\n' % (role, t.encode('utf-8').hex() or '-')
                                                    for role, texts in (('document', work['documents']), ('query', work['queries'])) for t in texts))
        report['workload'] = dict(seed=work['seed'], sha256=work['sha256'], n_documents=len(work['documents']), n_queries=len(work['queries']),
                                  document_characters=sum(map(len, work['documents'])))
        py = [sys.executable, '-I']
        model_file = lambda q: vago/('SauerkrautLM-MoEdernColBERT-%s-%s.gguf' % (variant, q))
        tok_json = vago/'tokenizer.json'
        # 3. outputs
        author_out, ours_out, errors = {}, {}, {}
        for key, engine, quant in [('F32-native', engines['native'], 'F32'), ('F32-portable', engines['portable'], 'F32')] + \
                [(q + '-native', engines['native'], q) for q in QUANTS[1:]]:
            o = scratch/('out-author-%s.json' % key)
            try: author_out[key] = json_run(py + [HERE/'author_runner.py', 'outputs', vago, engine, model_file(quant), tok_json, 2, scratch/'cases.json', o],
                                            out/('outputs-author-%s.log' % key), plain_env(2), o)
            except Exception as e: errors['author-' + key] = str(e)
        for fast in (False, True):
            for build, (so, _) in ours_builds.items():
                key = ('fast-' if fast else '') + build
                o = scratch/('out-ours-%s.json' % key)
                try: ours_out[key] = json_run(py + [HERE/'ours_runner.py', 'outputs', ROOT/'python', so, packed, 2, scratch/'cases.json', scratch/'pairs.json', o],
                                              out/('outputs-ours-%s.log' % key), ours_env(so.parent, 2, fast), o)
                except Exception as e: errors['ours-' + key] = str(e)
        report['outputs'] = compare_outputs(original, vectors, author_out, ours_out); report['outputs']['errors'] = errors
        report['outputs']['ours_fast_vs_default'] = {
            b: _worst([lib.vector_metrics(lib.rows(ours_out['fast-' + b]['cases'][c['id']]['selected_vectors']),
                                          lib.rows(ours_out[b]['cases'][c['id']]['selected_vectors'])) for c in original['cases']])
            for b in ours_builds if b in ours_out and 'fast-' + b in ours_out}
        report['outputs']['author_skiplist_ids'] = next(iter(author_out.values()))['skiplist_ids'] if author_out else None
        write(jpath, report)
        # 4. speed
        configs = []
        for t in THREADS:
            for build in ('native', 'portable'):
                configs.append(dict(runtime='author', build=build, model='F32', threads=t, batching='wrapper default (docs 16, queries 64)', db=0, qb=0))
                configs.append(dict(runtime='author', build=build, model='F32', threads=t, batching='batch_size=1', db=1, qb=1))
            for prefix in ('ours', 'ours-fast'):
                for build in ('portable', 'native'):
                    configs.append(dict(runtime=prefix + '-python', build=build, model='F32', threads=t, batching='one call per text (no batching)'))
                    configs.append(dict(runtime=prefix + '-c', build=build, model='F32', threads=t, batching='one call per text (no batching)'))
        for t in THREADS:
            for q in QUANTS[1:]:
                configs.append(dict(runtime='author', build='native', model=q, threads=t, batching='wrapper default (docs 16, queries 64)', db=0, qb=0))
        report['speed'] = []
        for i, c in enumerate(configs):
            meta = {k: v for k, v in c.items() if k not in ('db', 'qb')}
            if time.time() > DEADLINE: report['speed'].append(dict(meta, error='skipped: time budget exhausted')); continue
            o = scratch/('bench-%d.json' % i); log = out/('bench-%02d-%s-%s-%s-t%d.log' % (i, c['runtime'], c['build'], c['model'], c['threads']))
            try:
                if c['runtime'] == 'author':
                    raw = json_run(py + [HERE/'author_runner.py', 'bench', vago, engines[c['build']], model_file(c['model']), tok_json, c['threads'],
                                         scratch/'workload.json', c['db'], c['qb'], REPS, o], log, plain_env(c['threads']), o)
                elif c['runtime'].endswith('-python'):
                    so = ours_builds[c['build']][0]
                    raw = json_run(py + [HERE/'ours_runner.py', 'bench', ROOT/'python', so, packed, c['threads'], scratch/'workload.json', REPS, o],
                                   log, ours_env(so.parent, c['threads'], c['runtime'].startswith('ours-fast')), o)
                else:
                    so, exe = ours_builds[c['build']]
                    raw = json_run([exe, packed, c['threads'], scratch/'workload.tsv', REPS], log,
                                   ours_env(so.parent, c['threads'], c['runtime'].startswith('ours-fast')))
                report['speed'].append(speed_row(meta, raw))
            except Exception as e:
                report['speed'].append(dict(meta, error=str(e)))
            write(jpath, report)
        report['completed'] = True
        write(jpath, report)
        summary(report, out/'summary.md')
        return 0
    except Exception as e:
        report.update(error_type=type(e).__name__, error=str(e).replace(str(ROOT), '<repo>'))
        write(jpath, report); raise


def summary(report, path):
    o = report.get('outputs', {}).get('digest', {})
    lines = ['# MoEdern %s: CrispEmbed vs author runtime (diagnostic)' % report['variant'], '',
             'Source `%s`, ggml `%s`, workload sha256 `%s`, %d timed repetitions (median), warm-up excluded.' % (
                 report.get('source'), GGML_COMMIT[:12], report.get('workload', {}).get('sha256', '')[:16], REPS), '',
             'Batching: the author wrapper batches (documents 16, queries 64 per engine request by default; also measured with batch_size=1). '
             'CrispEmbed has no batch API: one encode call per text. CrispEmbed\'s default path accepts 1-2 threads only, so its 4-thread rows are expected to fail; '
             'ours-fast rows (CRISPEMBED_MOEDERN_FAST=1) accept up to 64 threads under the unchanged parity gates.', '',
             '## Outputs vs frozen original', '', '| arm | IDs equal | selected max abs | min cosine | row-count mismatches | MaxSim max abs diff |', '|---|---|---|---|---|---|']
    for k, v in o.items():
        if 'selected_vs_original' in v:
            s = v['selected_vs_original']
            lines.append('| %s | %d/%d | %s | %s | %d | %.6g |' % (k, v['token_ids_equal_original'], v['cases'], s['max_absolute'], s['min_row_cosine'],
                                                                 s['row_count_mismatches'], v['maxsim_max_abs_diff_original']))
    fast = report.get('outputs', {}).get('ours_fast_vs_default', {})
    if fast:
        lines += ['', '## CrispEmbed fast vs default path (same build, selected vectors)', '', '| build | max abs | min cosine |', '|---|---|---|']
        lines += ['| %s | %s | %s |' % (k, v['max_absolute'], v['min_row_cosine']) for k, v in fast.items()]
    lines += ['', '## Author vs CrispEmbed (portable F32)', '', '| arm | IDs equal | selected max abs | min cosine | row-count mismatches |', '|---|---|---|---|---|']
    for k, v in o.items():
        if k.endswith('_vs_ours'):
            lines.append('| %s | %d | %s | %s | %d |' % (k, v['token_ids_equal'], v['selected']['max_absolute'], v['selected']['min_row_cosine'],
                                                          v['selected']['row_count_mismatches']))
    lines += ['', '## Speed', '', lib.speed_table(report.get('speed', []))]
    path.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])))
