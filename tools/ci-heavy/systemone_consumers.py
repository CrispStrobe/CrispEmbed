#!/usr/bin/env python3
"""Hosted SystemOne public consumers on the actual root build profile.

Order: closed public source -> reviewed packed producer pin (rejects before any
service/payload access while pending) -> authenticated packed GGUF -> root build
with GGML_LLAMAFILE=ON and opt-in audit targets -> 227-stage root-profile parity
-> C/Python/CLI/HTTP/Rust/Dart consumers on the same packed file.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time
import zipfile
ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'docs/systemone-public-consumers-v1.json'
OWN = (PROTOCOL, 'tools/ci-heavy/systemone_consumers.py', 'tests/test_systemone_consumers_worker.py',
       '.github/workflows/systemone-consumers.yml', 'tests/systemone/consumer.py',
       'tests/systemone/rust_consumer.rs', 'tests/systemone/dart_consumer.dart', 'tests/systemone/CMakeLists.txt')
PROFILE = 'official-pytorch-text-batch1-512-no-extra-temperature'
REPO = 'repos/CrispStrobe/CrispEmbed/'


def require(ok, why):
    if not ok: raise ValueError(why)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def load(path): return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(4*2**20): h.update(b)
    return h.hexdigest()


def git(*args): return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()
def api(path): return json.loads(subprocess.check_output(['gh', 'api', REPO+path], timeout=90))


def pending(cfg):
    """Return the reasons the packed producer pin cannot yet be used; no I/O."""
    pin = cfg['packed_producer']; why = []
    if pin.get('reviewed') is not True: why.append('packed producer not independently reviewed')
    for key in ('source', 'run', 'review_artifact', 'review_receipt_sha256', 'asset_artifact',
                'model_sha256', 'model_bytes', 'original_archive'):
        if pin.get(key) in (None, '', 0): why.append('pending packed pin: '+key)
    return why


def policy(hosted=True):
    if hosted:
        require(os.environ.get('GITHUB_ACTIONS') == 'true', 'Hosted public consumers only')
        require(git('rev-parse', 'HEAD') == os.environ.get('GITHUB_SHA'), 'Caller/source service identity differs')
    cfg = load(ROOT/PROTOCOL)
    require(cfg['schema'] == 'crispembed-systemone-public-consumers-v1', 'Consumer protocol differs')
    require(cfg['expected']['compared_stages'] == 227 and cfg['expected']['cases'] == 4, 'Fixed numerical scope changed')
    require(not any(cfg['qualification'].values()), 'Qualification flags may not be pre-set')
    require('-DGGML_LLAMAFILE=ON' in cfg['root_profile']['cmake'] and '-DCRISPEMBED_SYSTEMONE_AUDIT=ON' in cfg['root_profile']['cmake'],
            'Actual root profile must keep llamafile and opt-in audit targets')
    for path in OWN:
        require(git('hash-object', path) == git('rev-parse', 'HEAD:'+path), 'Uncommitted consumer source: '+path)
    sys.path.insert(0, str(ROOT/'tools'))
    import systemone_api_contract
    systemone_api_contract.verify()
    return cfg


def fetch(pin, scratch, include=None):
    """Exact retained artifact by service identity; selected members only."""
    run = api('actions/runs/'+str(pin['run']))
    require(run['head_sha'].startswith(pin['source']) and run['run_attempt'] == 1 and run['status'] == 'completed'
            and run['conclusion'] in pin.get('conclusions', ['success']), 'Producer run identity differs')
    a = api('actions/artifacts/'+str(pin['id']))
    require(a['workflow_run']['id'] == pin['run'] and a['name'] == pin['name'] and a['size_in_bytes'] == pin['bytes']
            and a['digest'] == 'sha256:'+pin['sha256'] and not a['expired'], 'Retained artifact identity differs')
    require(0 < pin['bytes'] <= 12*2**30, 'Declared artifact download budget exceeded')
    target = scratch/str(pin['id']); target.mkdir(); archive = scratch/(str(pin['id'])+'.zip')
    with archive.open('xb') as stream:
        subprocess.run(['gh', 'api', REPO+'actions/artifacts/'+str(pin['id'])+'/zip'], stdout=stream, check=True, timeout=3600)
    require(archive.stat().st_size == pin['bytes'] and sha(archive) == pin['sha256'], 'Downloaded ZIP bytes differ')
    with zipfile.ZipFile(archive) as z:
        items = z.infolist(); names = [r.filename for r in items]
        require(len(set(names)) == len(names) and len(items) <= 8192, 'Archive member budget exceeded')
        for row in items:
            p = PurePosixPath(row.filename)
            require(row.filename and not p.is_absolute() and '..' not in p.parts and '\\' not in row.filename
                    and (row.external_attr >> 16) & 0o170000 != 0o120000, 'Unsafe retained member')
        if include is not None: require(set(include) <= set(names), 'Pinned member missing')
        for row in items:
            if row.is_dir() or include is not None and row.filename not in include: continue
            path = target/row.filename; path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(row) as src, path.open('xb') as dst: shutil.copyfileobj(src, dst, 4*2**20)
    archive.unlink()
    return target, {k: a[k] for k in ('id', 'name', 'size_in_bytes', 'digest', 'expires_at')}


def command(args, log, *, cwd=ROOT, env=None, timeout=1800, stdout=None):
    start = time.monotonic()
    with log.open('w') as err:
        r = subprocess.run(list(map(str, args)), cwd=cwd, env=env, stdout=stdout or err, stderr=err, timeout=timeout)
    return dict(exit=r.returncode, elapsed_seconds=time.monotonic()-start, log_sha256=sha(log))


def comparison_helpers(cfg, scratch):
    directory = scratch/'comparison-tools'; directory.mkdir()
    for path, digest in cfg['reference_helpers'].items():
        data = subprocess.check_output(['git', 'show', cfg['reference_source']+':'+path], cwd=ROOT)
        require(hashlib.sha256(data).hexdigest() == digest, 'Exact original comparison helper source differs')
        (directory/Path(path).name).write_bytes(data)
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location('systemone_original_comparison', directory/'josev2b_composed_panel.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def reviewed_producer(cfg, scratch, review):
    """Small review receipt first; the GGUF is fetched only after it passes."""
    pin = cfg['packed_producer']
    directory, service = fetch(dict(pin['review_artifact'], run=pin['run'], source=pin['source']), scratch)
    receipt = directory/'hosted.json'
    require(sha(receipt) == pin['review_receipt_sha256'], 'Reviewed packing receipt bytes differ')
    report = load(receipt)
    require(report['completed'] is True and report['packed_numeric_and_complete_typed_parity'] is True
            and report['compared_stages'] == 227 and not report['failures'], 'Packing receipt is not a full pass')
    require(report['packed_asset']['sha256'] == pin['model_sha256'] and report['packed_asset']['bytes'] == pin['model_bytes'],
            'Packed asset identity differs from reviewed receipt')
    shutil.copyfile(receipt, review/'packed-prerequisite.json')
    asset, asset_service = fetch(dict(pin['asset_artifact'], run=pin['run'], source=pin['source']), scratch,
                                 include={pin['model_member']})
    model = asset/pin['model_member']
    require(model.stat().st_size == pin['model_bytes'] and sha(model) == pin['model_sha256'], 'Packed GGUF bytes differ')
    return model, dict(review=service, asset=asset_service)


def original_reference(cfg, scratch):
    pin = cfg['packed_producer']['original_archive']
    directory, service = fetch(pin, scratch, include=set(pin['members']))
    reference = directory/pin['reference_dir']
    require(sha(reference/'reference.json') == cfg['original']['reference_sha256']
            and sha(reference/'input-freeze.json') == cfg['original']['input_freeze_sha256'], 'Frozen original bytes differ')
    return reference, service


def root_build(cfg, build, review, report, env):
    for name, args in [('configure', ['cmake', '-S', ROOT, '-B', build, *cfg['root_profile']['cmake']]),
                       ('build', ['cmake', '--build', build, '--parallel', '2', '--target', *cfg['root_profile']['targets']])]:
        report['commands'][name] = command(args, review/(name+'.log'), env=env)
        write(review/'hosted.json', report); require(report['commands'][name]['exit'] == 0, 'Root build rejected: '+name)
    cache = (build/'CMakeCache.txt').read_text()
    require('GGML_LLAMAFILE:BOOL=ON' in cache and 'CRISPEMBED_SYSTEMONE_AUDIT:BOOL=ON' in cache, 'Actual root cache profile differs')
    rows = [r for r in load(build/'compile_commands.json') if Path(r['file']).name in
            {'josev2b_composed.cpp', 'josev2b_head.cpp', 'josev2b_backbone.cpp', 'josev2b_text_frontend.cpp',
             'josev2b_answer_serializer.cpp', 'qwen35_decoder.cpp'}]
    require(rows and not any(f in r['command'] for r in rows for f in ('-ffast-math', '-Ofast', '-ffp-contract=fast')),
            'Root arithmetic flags differ')
    write(review/'compile-commands.json', rows)
    archives = {}
    for kind in ('plain', 'observed'):
        archive = build/'tests/systemone'/('libsystemone-audit-'+kind+'.a')
        symbols = subprocess.check_output(['nm', '-C', archive], text=True)
        observed = 'josev2b::composed_test::observe' in symbols
        require(observed == (kind == 'observed'), 'Observer presence differs in '+kind+' archive')
        archives[kind] = dict(sha256=sha(archive), bytes=archive.stat().st_size)
    production = subprocess.check_output(['nm', '-D', '-C', build/'libcrispembed.so'], text=True)
    require('composed_test::' not in production and 'run_observed' not in production, 'Observer leaked into production library')
    report['archives'] = archives; report['production_excludes_observers'] = True


def root_parity(cfg, build, model, reference, raw, review, report, env, scratch):
    panel = comparison_helpers(cfg, scratch)
    panel.ref_controls(build/'tests/systemone/systemone-audit-diff', review/'ref-controls')
    freeze, original = load(reference/'input-freeze.json'), load(reference/'reference.json')
    requests = raw/'requests.json'
    write(requests, [dict(id=r['id'], request_json=json.dumps(r['request'], ensure_ascii=False, allow_nan=False,
                                                              separators=(',', ':'))) for r in freeze['records']])
    outputs = {}
    for label, variant in [('plain', 'plain'), ('observed', 'observed'), ('repeat', 'observed')]:
        outputs[label] = raw/label
        report['commands']['root-'+label] = command([build/'tests/systemone'/('systemone-audit-'+variant+'-probe'), model, requests,
                                                     outputs[label]], review/('root-'+label+'.log'), env=env)
        write(review/'hosted.json', report)
    require(all(report['commands']['root-'+k]['exit'] == 0 for k in outputs), 'Root-profile probe rejected')
    result = panel.compare(reference, freeze, original, outputs, build/'tests/systemone/systemone-audit-diff', raw/'comparison')
    shutil.copyfile(raw/'comparison/panel.json', review/'root-panel.json')
    passed = result['numeric_and_complete_typed_parity'] and result['compared_stages'] == 227
    report.update(root_compared_stages=result['compared_stages'], root_failures=result['failures'], root_parity=passed)
    require(passed, 'Root-profile 227-stage parity failed; consumers not run')
    return freeze


def consumers(cfg, build, model, reference, freeze, raw, review, report, env, scratch):
    gold = {c['id']: c['official_response'] for c in load(reference/'reference.json')['cases']}
    tsv = raw/'requests.tsv'
    tsv.write_text(''.join(r['id']+'\t'+json.dumps(r['request'], ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n'
                           for r in freeze['records']))
    env = dict(env, **cfg['gates'], CRISPEMBED_OFFLINE='1')
    for mode in ('c', 'python', 'cli', 'http'):
        report['commands']['consumer-'+mode] = command(
            [sys.executable, ROOT/'tests/systemone/consumer.py', '--mode', mode, '--model', model,
             '--library', build/'libcrispembed.so', '--cli', build/'crispembed', '--server', build/'crispembed-server',
             '--freeze', reference/'input-freeze.json', '--reference', reference/'reference.json', '--output', raw/('consumer-'+mode)],
            review/('consumer-'+mode+'.log'), env=env)
        shutil.copyfile(raw/('consumer-'+mode)/'consumer.json', review/('consumer-'+mode+'.json'))
        write(review/'hosted.json', report)
    # Rust and Dart print id<TAB>response; compare complete typed JSON with gold.
    rust = scratch/'rust-consumer'; (rust/'src').mkdir(parents=True)
    (rust/'Cargo.toml').write_text('[package]\nname="systemone-consumer"\nversion="0.0.0"\nedition="2021"\n'
                                   '[dependencies]\ncrispembed={path="'+str(ROOT/'crispembed')+'"}\n[workspace]\n')
    shutil.copyfile(ROOT/'tests/systemone/rust_consumer.rs', rust/'src/main.rs')
    dart = scratch/'dart-consumer'; (dart/'lib').mkdir(parents=True); (dart/'bin').mkdir()
    shutil.copyfile(ROOT/'flutter/crispembed/lib/src/systemone.dart', dart/'lib/systemone.dart')
    shutil.copyfile(ROOT/'tests/systemone/dart_consumer.dart', dart/'bin/main.dart')
    (dart/'pubspec.yaml').write_text('name: systemone_audit\nenvironment:\n  sdk: ">=3.0.0 <4.0.0"\ndependencies:\n  ffi: 2.1.4\n')
    lib_env = dict(env, CRISPEMBED_SYS_LIB_DIR=str(build), LD_LIBRARY_PATH=str(build))
    for mode, args, cwd in [('rust', ['cargo', 'run', '--release', '--quiet', '--', model, tsv], rust),
                            ('dart-pub', ['dart', 'pub', 'get'], dart),
                            ('dart', ['dart', 'run', 'bin/main.dart', model, build/'libcrispembed.so', tsv], dart)]:
        out = raw/(mode+'.out')
        with out.open('w') as stream:
            report['commands']['consumer-'+mode] = command(args, review/('consumer-'+mode+'.log'), cwd=cwd, env=lib_env, stdout=stream)
        write(review/'hosted.json', report)
        if mode == 'dart-pub': continue
        rows = [line.split('\t', 1) for line in out.read_text().splitlines()]
        exact = [len(r) == 2 and json.loads(r[1]) == gold.get(r[0]) for r in rows]
        write(review/('consumer-'+mode+'.json'), dict(mode=mode, cases=len(rows), complete_typed_exact=exact,
                                                       exit=report['commands']['consumer-'+mode]['exit']))
    passed = {m: report['commands']['consumer-'+m]['exit'] == 0 for m in cfg['expected']['consumers']}
    for m in ('rust', 'dart'):
        row = load(review/('consumer-'+m+'.json'))
        passed[m] = passed[m] and row['cases'] == 4 and all(row['complete_typed_exact'])
    report['consumers'] = passed
    return all(passed.values())


def run(out, scratch):
    cfg = policy()
    require(not out.exists() and not scratch.exists(), 'Fresh task-owned output/scratch required')
    out.mkdir(parents=True); scratch.mkdir(parents=True)
    review = out/'review'; raw = out/'raw'; review.mkdir(); raw.mkdir()
    report = dict(schema='crispembed-systemone-public-consumers-report-v1', completed=False, source=os.environ.get('GITHUB_SHA'),
                  worker_files={p: sha(ROOT/p) for p in OWN}, commands={}, **cfg['qualification'])
    blocked = pending(cfg)
    if blocked:
        report.update(rejected_before_payload_access=True, pending=blocked); write(review/'hosted.json', report)
        return 1
    available = int(next(r.split()[1] for r in Path('/proc/meminfo').read_text().splitlines() if r.startswith('MemAvailable:')))*1024
    require(available >= cfg['resources']['minimum_ram_bytes'] and shutil.disk_usage(scratch).free >= cfg['resources']['minimum_disk_bytes'],
            'Hosted RAM/disk budget not met')
    env = {k: v for k, v in os.environ.items() if not k.startswith('CRISPEMBED_')}
    env.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    write(review/'hosted.json', report)
    try:
        build = scratch/'build'
        root_build(cfg, build, review, report, env)
        model, report['producer'] = reviewed_producer(cfg, scratch, review)
        reference, report['original_archive'] = original_reference(cfg, scratch)
        freeze = root_parity(cfg, build, model, reference, raw, review, report, dict(env, **cfg['gates']), scratch)
        passed = consumers(cfg, build, model, reference, freeze, raw, review, report, env, scratch)
        report.update(completed=True, all_consumers_complete_typed_parity=passed)
        write(review/'hosted.json', report); return int(not passed)
    except Exception as error:
        message = str(error)
        for path, label in ((str(ROOT), '<caller>'), (str(scratch), '<scratch>'), (str(out), '<out>')):
            message = message.replace(path, label)
        report.update(error_type=type(error).__name__, error=message); write(review/'hosted.json', report); return 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--scratch', type=Path, required=True)
    a = p.parse_args()
    try: return run(a.output.resolve(), a.scratch.resolve())
    except Exception as error:
        if not a.output.exists() or not any(a.output.iterdir()):
            write(a.output/'review/initial-rejection.json', dict(schema='crispembed-systemone-public-consumers-report-v1',
                  completed=False, error_type=type(error).__name__, error=str(error).replace(str(ROOT), '<caller>')))
        return 1


if __name__ == '__main__': sys.exit(main())
