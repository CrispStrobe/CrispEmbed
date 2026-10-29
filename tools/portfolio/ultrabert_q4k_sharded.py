#!/usr/bin/env python3
"""Hosted retained-F32 audit and explicitly activated UltraBERT Q4 shards.

Frozen018 supplies arithmetic, conversion, ranking and Ref adapters. This new
controller never calls its whole-panel run() or native() (no F32/original replay).
"""
import argparse
import ctypes
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import shlex
import subprocess
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/ultrabert-q4k-sharded-v1.json'
RUNTIME = '018cf85b7bb6ee2ac98011738aaa289d63b896ac'
F32_SHA = '89478b9cbbad2fd1121be4031d760f1529c989fedf23b6b19251ba2c87422566'
ROWS, WIDTH, SHARDS = 5483, 768, 8
FIRST_RECIPE = 'q4k-last-four-ffn'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    return dict(bytes=Path(path).stat().st_size, sha256=sha(path))


def atomic(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    with temporary.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value):
    atomic(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def indices(shard):
    require(type(shard) is int and 0 <= shard < SHARDS, 'original strided shard0..7 required')
    return list(range(shard, ROWS, SHARDS))


def checked_file(path, pin):
    require(identity(path) == pin, 'exact file SHA/extent differs: ' + Path(path).name)


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def source(runtime, reviewed):
    protocol = read(PROTOCOL)
    require(re.fullmatch('[0-9a-f]{40}', reviewed or '') is not None and git('rev-parse', 'HEAD') == reviewed,
            'reviewed exact audit source differs')
    require(os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_SHA') == reviewed
            and os.environ.get('RUNNER_ENVIRONMENT') == 'github-hosted', 'genuine hosted GitHub source required')
    expected_owned={'tools/portfolio/ultrabert_q4k_sharded.py','tests/test_ultrabert_q4k_sharded.py','.github/workflows/ultrabert-q4k-sharded-v1.yml','docs/ultrabert-q4k-sharded.md','tests/results/ultrabert/q4k-sharded-initial-freeze-rejection.json'}
    require(set(protocol['owned_sha256'])==expected_owned,'exact audit source closure required')
    require(not git('status', '--porcelain', '--', str(PROTOCOL.relative_to(ROOT)), *protocol['owned_sha256']), 'dirty audit source')
    for name, digest in protocol['owned_sha256'].items():
        require(sha(ROOT / name) == digest, 'owned source differs: ' + name)
    require(git('rev-parse', 'HEAD', cwd=runtime) == RUNTIME
            and not git('status', '--porcelain', '--untracked-files=all', cwd=runtime), 'frozen018 source must be exact and clean')
    require(git('rev-parse', 'HEAD', cwd=runtime / 'ggml') == protocol['ggml']
            and not git('status', '--porcelain', '--untracked-files=all', cwd=runtime / 'ggml'), 'frozen ggml source differs')
    for path, obj in protocol['runtime_git_objects'].items():
        require(git('rev-parse', RUNTIME + ':' + path, cwd=runtime) == obj, 'runtime Git object differs: ' + path)
    return dict(audit_source=reviewed, runtime_source=RUNTIME, protocol_sha256=sha(PROTOCOL),
                runtime_git_objects=protocol['runtime_git_objects'], ggml=protocol['ggml'],
                controller_sha256=sha(Path(__file__)))


def symbol_names(text):
    names = []
    for line in text.splitlines():
        if not line.strip(): continue
        row = re.fullmatch(r'\s*(?:[0-9a-fA-F]+\s+)?[A-Za-z?]\s+(.+)', line)
        require(row is not None, 'unparsed nm symbol inventory')
        names.append(row[1])
    return names


def symbol_text_controls(whole, object_symbols, object_strings, pinned):
    names = symbol_names(whole)
    actual = sorted(name for name in names if 'crispembed_diff::' in name)
    require(actual == pinned['known_unrelated_diff_symbols'], 'whole-library pre-existing comparator symbol inventory changed')
    leak = lambda value: re.search(r'ultrabert[^\n]*(?:observ|probe|dump|testing)', value, re.I) is not None
    require(not any(leak(name) for name in names), 'Ultra observer/probe in whole-library symbols')
    object_names = symbol_names(object_symbols)
    require(not any('crispembed_diff' in name or leak(name) for name in object_names), 'Ultra object contains observer/comparator symbol')
    require('crispembed_diff' not in object_strings and not leak(object_strings), 'Ultra object contains observer/comparator string')
    require(not any(line.strip() in {'emb_out', 'post_attn_residual', 'final_hidden'} for line in object_strings.splitlines()), 'Ultra object contains reference-only stage label')
    required = {'ultrabert_init', 'ultrabert_free', 'ultrabert_encode', 'ultrabert_hparams', 'ultrabert_token', 'ultrabert_query_prefix'}
    found = {name.split('(', 1)[0] for name in object_names if name.startswith('ultrabert_')}
    require(required <= found, 'actual Ultra production object exports incomplete')
    return dict(known_unrelated_diff_symbols=len(actual), ultra_exports=sorted(required),
                object_symbols_no_observers=True, object_strings_no_observers=True,
                whole_library_ultra_symbols_no_observers=True)


def symbol_audit(runtime, build, out):
    """Inspect the actual shared-target object; no surrogate test archive/build."""
    out.mkdir(parents=True, exist_ok=True)
    report = dict(schema='crispembed-ultrabert-production-symbol-audit-v1', completed=False,
                  runtime_source=RUNTIME, original_forwards=0, model_access=False)
    receipt = out / 'symbol-audit.json'
    try:
        pin = read(PROTOCOL)['symbol_audit']
        rows = read(build / 'compile_commands.json')
        object_path = build / 'CMakeFiles/crispembed-shared.dir/src/ultrabert_embed.cpp.o'
        candidates = []
        for row in rows:
            if Path(row['file']).resolve() != (runtime / 'src/ultrabert_embed.cpp').resolve(): continue
            argv = row.get('arguments') or shlex.split(row['command'])
            require(argv.count('-o') == 1, 'unambiguous actual Ultra object output required')
            target = Path(argv[argv.index('-o') + 1])
            if not target.is_absolute(): target = Path(row['directory']) / target
            if target.resolve() == object_path.resolve(): candidates.append((row, argv))
        require(len(candidates) == 1 and object_path.is_file() and not object_path.is_symlink(), 'exact shared-target Ultra object missing/ambiguous')
        row, argv = candidates[0]
        require(not any(re.search(r'^-D.*(?:ULTRABERT.*(?:TEST|OBSERV|PROBE)|CRISPEMBED_TEST)', a, re.I) for a in argv), 'test observer macro in production object command')
        checked_file(runtime / 'src/ultrabert_embed.cpp', pin['ultra_source'])
        for name, digest in pin['known_unrelated_source_sha256'].items():
            require(sha(runtime / name) == digest, 'known unrelated comparator source changed: ' + name)
        whole_path = out / 'symbols.txt'
        for tool, target, destination in [('nm', build / 'libcrispembed.so', whole_path),
                                         ('nm', object_path, out / 'ultrabert-object-symbols.txt'),
                                         ('strings', object_path, out / 'ultrabert-object-strings.txt')]:
            with destination.open('wb') as stream:
                subprocess.run([tool, '-C' if tool == 'nm' else '-a', str(target)], stdout=stream, stderr=subprocess.STDOUT, check=True)
        report['controls'] = symbol_text_controls(whole_path.read_text(), (out / 'ultrabert-object-symbols.txt').read_text(),
                                                  (out / 'ultrabert-object-strings.txt').read_text(), pin)
        report.update(completed=True, production_object='CMakeFiles/crispembed-shared.dir/src/ultrabert_embed.cpp.o',
                      object=identity(object_path), library=identity(build / 'libcrispembed.so'),
                      compile_command=row, source=identity(runtime / 'src/ultrabert_embed.cpp'),
                      inventories={name: identity(out / name) for name in ['symbols.txt', 'ultrabert-object-symbols.txt', 'ultrabert-object-strings.txt']},
                      initial_failure_run=pin['initial_failure_run'], initial_source_unchanged=True)
        write(receipt, report)
        return report
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error); write(receipt, report); raise


def build_proof(args):
    build = args.library.parent
    cache = (build / 'CMakeCache.txt').read_text(encoding='utf-8')
    expected = {'CMAKE_BUILD_TYPE': 'Release', 'CRISPEMBED_BUILD_SHARED': 'ON',
                'BUILD_SHARED_LIBS': 'OFF', 'GGML_NATIVE': 'OFF', 'CRISPEMBED_NATIVE': 'OFF',
                'GGML_LLAMAFILE': 'ON', 'GGML_METAL': 'OFF', 'GGML_CUDA': 'OFF', 'GGML_BLAS': 'OFF'}
    for key, value in expected.items():
        require(re.search(r'^' + key + r':[^=]+=' + value + r'$', cache, re.M), 'actual root build profile differs: ' + key)
    versions = {name: importlib.metadata.version(name) for name in ['numpy', 'gguf', 'huggingface-hub', 'safetensors']}
    require(versions == {'numpy': '2.2.6', 'gguf': '0.17.1', 'huggingface-hub': '0.36.0', 'safetensors': '0.6.2'},
            'frozen numerical/conversion tooling versions differ')
    audit=read(build.parent/'ultra-output/symbol-audit.json')
    require(audit['completed'] is True and audit['runtime_source']==RUNTIME and audit['library']==identity(args.library) and audit['object']==identity(build/'CMakeFiles/crispembed-shared.dir/src/ultrabert_embed.cpp.o'), 'actual Ultra object symbol audit missing/changed')
    return dict(symbol_audit=audit, profile=expected, packages=versions, library=identity(args.library), quantizer=identity(args.quantizer),
                comparator=identity(args.diff), cache=identity(build / 'CMakeCache.txt'),
                compile_commands=identity(build / 'compile_commands.json'),
                actual_root_runtime=True, original_or_f32_forward=False)


def old_module(runtime):
    # Only after source() authenticated the entire frozen checkout. No guard or
    # ROOT monkeypatch, no old run()/hosted_source()/native() execution.
    sys.path.insert(0, str(runtime / 'tools'))
    path = runtime / 'tools/portfolio/ultrabert_q4k.py'
    spec = importlib.util.spec_from_file_location('frozen_ultrabert_q4k_018', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def admission(out):
    out.mkdir(parents=True, exist_ok=True)
    ram = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                   if x.startswith('MemAvailable:'))) * 1024
    disk = shutil.disk_usage(out).free
    require(ram >= 14 * 2**30 and disk >= 24 * 2**30, 'unchanged14GiB RAM/24GiB disk admission failed')
    return dict(available_ram_bytes=ram, disk_free_bytes=disk, threads=2, cpu_only=True)


def kaggle_outputs(pin):
    # Typed SDK requests bind v2 on Get, Status and EVERY output page. Account
    # identity and signed URLs stay in memory, never in public report/log output.
    require(os.environ.get('KAGGLE_ACCOUNT') and os.environ.get('KAGGLE_API_TOKEN'), 'existing Kaggle secrets absent')
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    mod = importlib.import_module('kagglesdk.kernels.types.kernels_api_service')
    def request(kind):
        q = kind(); q.user_name = os.environ['KAGGLE_ACCOUNT']; q.kernel_slug = pin['kernel_slug']; q.version_label = pin['version_label']
        require(json.loads(str(q)).get('versionLabel') == pin['version_label'], 'SDK lost exact version')
        return q
    try:
        with api.build_kaggle_client() as client:
            service = client.kernels.kernels_api_client
            body = json.loads(str(service.get_kernel(request(mod.ApiGetKernelRequest))))
            submitted = body['blob']['source']
            require(hashlib.sha256(submitted.encode()).hexdigest() == pin['submitted_script_sha256']
                    and pin['source'] in submitted, 'submitted v2 source differs')
            status = str(service.get_kernel_session_status(request(mod.ApiGetKernelSessionStatusRequest)).status).split('.')[-1]
            require(status == 'ERROR', 'retained v2 terminal state differs')
            files = {}; token = None; pages = 0
            while True:
                q = request(mod.ApiListKernelSessionOutputRequest); q.page_size = 200
                if token: q.page_token = token
                r = service.list_kernel_session_output(q); pages += 1
                require(pages <= 100, 'output page bound exceeded')
                for item in r.files or []:
                    require(item.file_name not in files, 'duplicate output member')
                    files[item.file_name] = item.url
                token = r.next_page_token
                if not token: break
    except ValueError:
        raise
    except Exception:
        raise ValueError('exact-version service access rejected') from None
    return files, dict(kernel_slug=pin['kernel_slug'], version_label=pin['version_label'], status=status,
                       submitted_script_sha256=pin['submitted_script_sha256'], source=pin['source'], pages=pages,
                       indexed_members=sorted(files), actual_sdk_version_bound_on_every_request=True)


def download(url, path, maximum, expected=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    require(url.startswith('https://'), 'HTTPS source required')
    part = path.with_name(path.name + '.download')
    h = hashlib.sha256(); n = 0
    try:
        with urllib.request.urlopen(url, timeout=180) as response, part.open('xb') as f:
            header = response.headers.get('Content-Length')
            require(header is None or 0 <= int(header) <= maximum, 'remote member extent exceeds bound')
            while block := response.read(2**20):
                n += len(block); require(n <= maximum, 'remote member extent exceeds bound')
                h.update(block); f.write(block)
            f.flush(); os.fsync(f.fileno())
    except Exception:
        raise ValueError('bounded immutable transfer rejected: ' + path.name) from None
    result = dict(bytes=n, sha256=h.hexdigest())
    require(expected is None or result == expected, 'remote member exact identity differs: ' + path.name)
    os.replace(part, path)
    return result


def retained_freeze(args, old, report):
    import numpy as np
    pin = read(PROTOCOL)['retained_v2']
    files, proof = kaggle_outputs(pin)
    require(set(files) == set(pin['indexed_members']), 'retained v2 member coverage differs')
    report['retained_service'] = proof
    prefix = 'ultrabert-q4k/panel/'
    saved = args.out / 'retained-review.json'
    download(files[prefix + 'review.json'], saved, 2**20, pin['report'])
    setup_path = args.out / 'retained-setup.json'
    download(files['ultrabert-q4k/setup.json'], setup_path, 2**20, pin['setup'])
    setup = read(setup_path)
    require(setup['source'] == RUNTIME and setup['completed'] is False and setup['timed_out_command'] == 12
            and setup['commands'][-1]['exit'] == -15, 'initial nine-hour wrapper rejection differs')
    report['initial_setup'] = identity(setup_path)
    producer = read(saved); arm = producer['arms']['f32']
    require(producer['source'] == RUNTIME and not producer['completed'] and not producer['original_rerun']
            and set(producer['arms']) == {'f32'} and arm['qualified'] is True and arm['all_ids_exact'] is True
            and arm['sha256'] == F32_SHA and arm['model_bytes'] == 609437696
            and arm['raw']['f32_gate_passed'] and arm['normalized']['f32_gate_passed'], 'frozen F32 arm differs')
    frozen, data, expected, original_raw, original_norm = originals(args, old, report)
    native_raw = np.empty((ROWS, WIDTH), dtype='<f4'); native_norm = np.empty_like(native_raw)
    report['reference_controls'] = old.reference_controls(args.diff, args.scratch)
    report['streams'] = {}; report['actual_ref'] = []
    for i in range(SHARDS):
        folder = args.out / 'f32' / str(i); folder.mkdir(parents=True)
        take = indices(i); count = len(take)
        for kind, original, native in [('raw', original_raw, native_raw), ('normalized', original_norm, native_norm)]:
            for label in ['original', 'native']:
                key = f'f32/{i}/{label}-{kind}.f32'; path = args.out / key
                ident = download(files[prefix + key], path, count * WIDTH * 4)
                require(ident['bytes'] == count * WIDTH * 4, 'saved F32 stream shape differs')
                values = np.memmap(path, mode='r', dtype='<f4', shape=(count, WIDTH))
                require(np.isfinite(values).all(), 'saved F32 stream nonfinite')
                if label == 'original': require(np.array_equal(values.view('<u4'), original[take].view('<u4')), 'original403756 stream differs')
                else: native[take] = values
                report['streams'][key] = ident
        saved_result = args.out / f'f32/{i}/old-diff.json'
        download(files[prefix + f'f32/{i}/diff.json'], saved_result, 2**20)
        require(read(saved_result) == arm['actual_ref'][i]['result'], 'saved Ref result differs from retained producer')
        for label in ['original', 'native']:
            key = f'f32/{i}/{label}.gguf'; path = args.out / key
            report['streams'][key] = download(files[prefix + key], path, 8 * 2**20)
            rebuilt = folder / ('reconstructed-' + label + '.gguf')
            specs = [dict(name=kind, path=str(folder / f'{label}-{kind}.f32'), shape=[count, WIDTH],
                          sha256=sha(folder / f'{label}-{kind}.f32')) for kind in ['raw', 'normalized']]
            old.write_gguf(rebuilt, specs)
            require(identity(rebuilt) == identity(path), 'saved adapter not losslessly reconstructible from frozen streams')
            rebuilt.unlink()  # task-owned duplicate, after canonical payload/adapter SHA proof
        require(arm['actual_ref'][i]['shard'] == i, 'saved Ref shard order differs')
        result = compare(old, args.diff, folder, original_raw[take], original_norm[take], native_raw[take], native_norm[take])
        require(result['exit'] == 0, 'retained F32 Ref rejected')
        metric_agreement(result['independent'], arm['actual_ref'][i]['independent'])
        report['actual_ref'].append(result)
        write(args.out / 'freeze.json', report)
    report['raw'] = old.metrics(original_raw, native_raw); report['normalized'] = old.metrics(original_norm, native_norm)
    metric_agreement({k: report[k] for k in ['raw', 'normalized']}, {k: arm[k] for k in ['raw', 'normalized']})
    report['quality'] = old.quality(old.rankings(data, original_norm), old.rankings(data, native_norm))
    require(report['quality'] == arm['quality'] and report['quality']['passed'], 'saved full300 ranking gate differs')
    report['data'] = identity(args.out / 'dataset.json'); report['lengths'] = identity(args.out / 'lengths.json')
    report['completed'] = True; report['f32_qualified'] = True
    report['model_or_original_forwards'] = 0


def originals(args, old, report):
    import numpy as np
    runtime = args.runtime
    frozen = read(runtime / 'tests/results/ultrabert/scifact-original-freeze.json')
    prep = read(runtime / 'tests/results/ultrabert/scifact-canonical-preparation.json')
    needed = {k: prep['files'][k] for k in ['dataset.json', 'lengths.json']}
    report['preparation'] = old.artifact(frozen['preparation'], args.scratch / 'prepared', needed)
    data = read(args.scratch / 'prepared/dataset.json'); lengths = read(args.scratch / 'prepared/lengths.json')
    rows = data['queries'] + data['documents']; expected = {r['id']: r for r in lengths}
    require(len(rows) == len(expected) == ROWS and len(data['queries']) == 300
            and len(set(r['id'] for r in rows)) == ROWS, 'all5483 stable IDs required')
    shutil.copyfile(args.scratch / 'prepared/dataset.json', args.out / 'dataset.json')
    shutil.copyfile(args.scratch / 'prepared/lengths.json', args.out / 'lengths.json')
    raw = np.empty((ROWS, WIDTH), dtype='<f4'); norm = np.empty_like(raw)
    report['originals'] = []
    for item in frozen['shards']:
        i = item['shard']; folder = args.scratch / ('original-' + str(i)); take = indices(i)
        members = {k: item['raw_member_identities'][k] for k in ['raw.npy', 'normalized.npy', 'ids.json']}
        report['originals'].append(old.artifact(item['raw_artifact_lineage'], folder, members))
        r = np.load(folder / 'raw.npy', allow_pickle=False); n = np.load(folder / 'normalized.npy', allow_pickle=False)
        ids = read(folder / 'ids.json')
        require(r.dtype == n.dtype == np.dtype('<f4') and r.shape == n.shape == (len(take), WIDTH)
                and [v['id'] for v in ids] == [rows[j]['id'] for j in take]
                and all(v['ids'] == expected[v['id']]['ids'] for v in ids), 'original array/IDs layout differs')
        raw[take] = r; norm[take] = n
    return frozen, data, expected, raw, norm


def metric_agreement(actual, expected):
    for kind in ['raw', 'normalized']:
        for key in ['max_absolute', 'mean_absolute', 'rms', 'min_row_cosine', 'original_l2', 'native_l2']:
            a, b = actual[kind][key], expected[kind][key]
            require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= 2e-7 * max(1., abs(a), abs(b)),
                    'unchanged Ref/independent metric disagreement: ' + kind + '/' + key)


def compare(old, binary, folder, original_raw, original_norm, native_raw, native_norm):
    # Source018's same lossless GGUF adapter and actual crispembed_diff::Ref.
    folder.mkdir(parents=True, exist_ok=True)
    take = len(original_raw)
    for label, r, n in [('original', original_raw, original_norm), ('native', native_raw, native_norm)]:
        specs = []
        for kind, values in [('raw', r), ('normalized', n)]:
            path = folder / f'{label}-{kind}.f32'
            if path.exists():
                require(path.read_bytes() == values.astype('<f4').tobytes(), 'retained stream changed before comparator')
            else: values.astype('<f4').tofile(path)
            specs.append(dict(name=kind, path=str(path), shape=[take, WIDTH], sha256=sha(path)))
        adapter = folder / (label + '.gguf')
        if not adapter.exists(): old.write_gguf(adapter, specs)
    result_path = folder / 'diff.json'
    with (folder / 'diff.log').open('wb') as log:
        call = subprocess.run([str(binary.resolve()), str(folder / 'original.gguf'), str(folder / 'native.gguf'),
                               str(result_path)], stdout=log, stderr=subprocess.STDOUT, timeout=180)
    require(call.returncode in (0, 1), 'actual Ref structural rejection')
    result = read(result_path)
    require({s['name'] for s in result['stages']} == {'raw', 'normalized'} and len(result['stages']) == 2,
            'actual Ref stage coverage differs')
    independent = dict(raw=old.metrics(original_raw, native_raw), normalized=old.metrics(original_norm, native_norm))
    metric_agreement(independent, {s['name']: s for s in result['stages']})
    require(all(s['passed'] == independent[s['name']]['f32_gate_passed'] for s in result['stages']), 'actual Ref/numpy gate disagreement')
    return dict(exit=call.returncode, result=result, independent=independent,
                log=identity(folder / 'diff.log'), adapters={n: identity(folder / n) for n in ['original.gguf', 'native.gguf']})


def manifest(root):
    return {p.relative_to(root).as_posix(): identity(p) for p in sorted(root.rglob('*'))
            if p.is_file() and p.name not in ('manifest.json',) and not p.name.endswith('.pending')}


def validate_manifest(root):
    saved = read(root / 'manifest.json'); require(saved == manifest(root), 'authenticated artifact file manifest differs')
    return saved


def artifact(old, ident, target, expected, maximum):
    require(type(ident) is int and ident > 0, 'explicit reviewed artifact ID required')
    row = old.api('actions/artifacts/' + str(ident))
    run = old.api('actions/runs/' + str(row['workflow_run']['id']))
    require(not row['expired'] and row['workflow_run']['head_sha'] == expected['source']
            and run['head_sha'] == expected['source'] and run['run_attempt'] == 1 and run['conclusion'] == expected.get('conclusion', 'success')
            and ident == expected['artifact_id'] and row['digest'] == 'sha256:' + expected['sha256']
            and row['size_in_bytes'] == expected['bytes'] and row['name'] == expected['name'], 'reviewed artifact service identity differs')
    archive = target.with_suffix('.zip'); old.download_archive(ident, archive, maximum)
    checked_file(archive, {k: expected[k] for k in ['bytes', 'sha256']})
    target.mkdir()
    with zipfile.ZipFile(archive) as z:
        names = [r.filename for r in z.infolist()]
        require(len(names) == len(set(names)) and len(names) <= 2048 and sum(r.file_size for r in z.infolist()) <= maximum,
                'artifact duplicate/extent bound exceeded')
        for r in z.infolist():
            p = PurePosixPath(r.filename)
            require(not p.is_absolute() and '..' not in p.parts and '\\' not in r.filename
                    and ((r.external_attr >> 16) & 0o170000) != 0o120000, 'unsafe artifact member')
            if r.is_dir(): continue
            out = target / p; out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(r) as src, out.open('xb') as dst: shutil.copyfileobj(src, dst, 2**20)
    archive.unlink()
    validate_manifest(target)
    return target


def approval(kind):
    item = read(PROTOCOL)['reviewed_dependencies'].get(kind)
    require(isinstance(item, dict) and item.get('reviewed') is True, 'pending reviewed ' + kind + ' dependency; no model access')
    require(all(k in item for k in ['artifact_id', 'source', 'sha256', 'bytes', 'name', 'receipt_sha256', 'receipt_bytes']), 'reviewed dependency identity incomplete')
    return item


def frozen_inputs(args, old):
    pin = approval('freeze')
    root = artifact(old, args.freeze_artifact, args.scratch / 'freeze', pin, 256 * 2**20)
    checked_file(root / 'freeze.json', dict(bytes=pin['receipt_bytes'], sha256=pin['receipt_sha256']))
    receipt = read(root / 'freeze.json'); require(receipt['completed'] and receipt['f32_qualified'] and receipt['model_or_original_forwards'] == 0,
                                               'retained zero-forward F32 freeze is not complete')
    return root, receipt


def prepare_model(args, old, report):
    from huggingface_hub import snapshot_download
    from gguf import GGUFReader
    frozen, freeze = frozen_inputs(args, old)
    # No module/model execution: reproduce the authenticated conversion only.
    snapshot = snapshot_download(old.MODEL, revision=old.REVISION,
                                 allow_patterns=list(old.profile()['inspected_file_sha256']) + ['model.safetensors'])
    path = args.runtime / 'models/convert-ultrabert-to-gguf.py'
    spec = importlib.util.spec_from_file_location('frozen_ultrabert_converter', path)
    converter = importlib.util.module_from_spec(spec); spec.loader.exec_module(converter)
    f32 = args.scratch / 'ultrabert-f32.gguf'; converter.convert(snapshot, f32, 'f32')
    checked_file(f32, dict(bytes=609437696, sha256=F32_SHA))
    model = args.out / 'model.gguf'
    rules = old.recipes()[FIRST_RECIPE]
    with (args.out / 'quantizer.log').open('wb') as log:
        call = subprocess.run([str(args.quantizer.resolve()), str(f32), str(model), 'q4_k', *rules],
                              stdout=log, stderr=subprocess.STDOUT, timeout=1200)
    require(call.returncode == 0, 'actual frozen C quantizer failed')
    a = GGUFReader(f32); b = GGUFReader(model)
    original = {t.name: t for t in a.tensors}; inventory = {}; types = {}
    require(len(original) == len(b.tensors) == 218, 'exact218 tensor inventory required')
    for tensor in b.tensors:
        require(tensor.name in original and list(map(int, tensor.shape)) == list(map(int, original[tensor.name].shape)), 'semantic tensor shape changed')
        typ = tensor.tensor_type.name; types[typ] = types.get(typ, 0) + 1
        wanted = 'Q4_K' if re.fullmatch(r'blk\.(20|21|22|23)\.ffn_(gate|up|down)\.weight', tensor.name) else 'F32'
        require(typ == wanted, 'actual role precision differs: ' + tensor.name)
        checksum = hashlib.sha256(tensor.data).hexdigest()
        if typ == 'F32': require(checksum == hashlib.sha256(original[tensor.name].data).hexdigest(), 'critical F32 payload changed')
        inventory[tensor.name] = dict(type=typ, shape=list(map(int, tensor.shape)), payload_sha256=checksum)
    require(types == {'F32': 206, 'Q4_K': 12}, 'declared twelve genuine Q4_K/206F32 tensors required')
    report.update(completed=True, recipe=FIRST_RECIPE, model=identity(model), original_f32=dict(bytes=609437696, sha256=F32_SHA),
                  inventory=inventory, actual_stored_types=types, quantizer=identity(args.quantizer), quantizer_log=identity(args.out / 'quantizer.log'),
                  freeze_receipt_sha256=sha(frozen / 'freeze.json'), model_or_original_forwards=0,
                  legacy_header_from_frozen018_quantizer=True, standard_header_publication_qualified=False)


def row_identity(row, expected):
    return hashlib.sha256(json.dumps(dict(id=row['id'], text=row['text'], ids=expected[row['id']]['ids']),
                                     ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def resume_prefix(out, binding, rows):
    journal = out / 'rows.jsonl'; state_path = out / 'binding.json'
    if not state_path.exists():
        require(not journal.exists() and not (out / 'raw.f32').exists() and not (out / 'normalized.f32').exists(), 'unbound resume outputs rejected')
        write(state_path, binding); atomic(journal, b''); atomic(out / 'raw.f32', b''); atomic(out / 'normalized.f32', b'')
    require(read(state_path) == binding, 'resume source/model/input/library/binary binding differs')
    # A crash may leave uncommitted vector tails, never accepted as completed.
    saved = journal.read_bytes(); require(not saved or saved.endswith(b'\n'), 'torn row journal rejected')
    records = [json.loads(s) for s in saved.splitlines()]
    require(len(records) <= len(rows), 'journal has extra rows')
    for i, record in enumerate(records):
        require(record['index'] == i and record['id'] == rows[i]['id'] and record['input_sha256'] == rows[i]['input_sha256'], 'resume row order/input differs')
        for kind in ['raw', 'normalized']:
            with (out / (kind + '.f32')).open('rb') as f: f.seek(i * WIDTH * 4); data = f.read(WIDTH * 4)
            require(len(data) == WIDTH * 4 and hashlib.sha256(data).hexdigest() == record[kind + '_sha256'], 'resume committed vector span differs')
    for kind in ['raw', 'normalized']:
        path = out / (kind + '.f32'); require(path.stat().st_size >= len(records) * WIDTH * 4, 'resume stream truncated')
        with path.open('r+b') as f: f.truncate(len(records) * WIDTH * 4); f.flush(); os.fsync(f.fileno())
    return len(records)


def commit_row(out, record, raw, norm):
    require(len(raw) == len(norm) == WIDTH * 4, 'exact768F32 output row required')
    for kind, data in [('raw', raw), ('normalized', norm)]:
        with (out / (kind + '.f32')).open('ab') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        record[kind + '_sha256'] = hashlib.sha256(data).hexdigest()
    # Journal commit follows BOTH durable vectors. A killed forward/tail remains
    # partial and is never counted; resume authenticates committed spans first.
    atomic(out / 'rows.jsonl', (out / 'rows.jsonl').read_bytes() +
           (json.dumps(record, sort_keys=True, allow_nan=False) + '\n').encode())


def shard(args, old, report):
    import numpy as np
    frozen, freeze = frozen_inputs(args, old)
    pin = approval('model')
    prepared = artifact(old, args.model_artifact, args.scratch / 'model', pin, 1024 * 2**20)
    checked_file(prepared / 'model.json', dict(bytes=pin['receipt_bytes'], sha256=pin['receipt_sha256']))
    model_record = read(prepared / 'model.json'); model = prepared / 'model.gguf'
    require(model_record['completed'] and model_record['recipe'] == FIRST_RECIPE and model_record['actual_stored_types'] == {'F32': 206, 'Q4_K': 12}, 'reviewed recipe preparation differs')
    checked_file(model, model_record['model'])
    activation = read(PROTOCOL)['activation']
    require(activation['reviewed'] is True and args.shard in activation['shards'] and activation['recipe'] == FIRST_RECIPE,
            'shard not explicitly activated; no native forwards')
    data = read(frozen / 'dataset.json'); expected = {r['id']: r for r in read(frozen / 'lengths.json')}
    allrows = data['queries'] + data['documents']; require(len(allrows) == ROWS, 'all5483 frozen inputs required')
    take = indices(args.shard); rows = [dict(r, input_sha256=row_identity(r, expected)) for r in (allrows[j] for j in take)]
    binding = dict(source={k: report['source'][k] for k in ['runtime_source', 'runtime_git_objects', 'ggml', 'controller_sha256']}, library=identity(args.library), comparator=identity(args.diff),
                   frozen_receipt_sha256=sha(frozen / 'freeze.json'), model=model_record['model'],
                   model_receipt_sha256=sha(prepared / 'model.json'), shard=args.shard, recipe=FIRST_RECIPE,
                   inputs=[dict(id=r['id'], input_sha256=r['input_sha256']) for r in rows])
    if args.resume_artifact:
        resume_pin = approval('resume')
        prior = artifact(old, args.resume_artifact, args.scratch / 'resume', resume_pin, 128 * 2**20)
        checked_file(prior / 'shard.json', dict(bytes=resume_pin['receipt_bytes'], sha256=resume_pin['receipt_sha256']))
        require(not read(prior / 'shard.json')['completed'], 'resume requires declared incomplete shard evidence')
        for name in ['binding.json', 'raw.f32', 'normalized.f32', 'rows.jsonl']:
            require(not (args.out / name).exists(), 'resume import would overwrite existing task output')
            shutil.copyfile(prior / name, args.out / name)
        report['resume_receipt_sha256'] = sha(prior / 'shard.json')
    start = resume_prefix(args.out, binding, rows)
    # Workflow launches a fresh Python with the existing exact build/lib path.
    allowed = {'CRISPEMBED_ULTRABERT': '1', 'CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN': '1',
               'CRISPEMBED_ULTRABERT_TRUNCATE': '1', 'CRISPEMBED_ULTRABERT_QUANT': '1'}
    require({k: v for k, v in os.environ.items() if k.startswith('CRISPEMBED_')} == allowed, 'exact four native gates required')
    lib = ctypes.CDLL(str(args.library.resolve()))
    lib.crispembed_init.argtypes = [ctypes.c_char_p, ctypes.c_int]; lib.crispembed_init.restype = ctypes.c_void_p
    lib.crispembed_free.argtypes = [ctypes.c_void_p]; lib.crispembed_free.restype = None
    lib.crispembed_encode_tokens_raw.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]; lib.crispembed_encode_tokens_raw.restype = ctypes.POINTER(ctypes.c_float)
    lib.crispembed_encode.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]; lib.crispembed_encode.restype = ctypes.POINTER(ctypes.c_float)
    lib.crispembed_last_token_ids.argtypes = [ctypes.c_void_p]; lib.crispembed_last_token_ids.restype = ctypes.POINTER(ctypes.c_int32)
    # Retain default/unset, empty and literal0 rejection before gate1 model load.
    report['quant_gate_rejections'] = {}
    for value in [None, '', '0']:
        if value is None: os.environ.pop('CRISPEMBED_ULTRABERT_QUANT', None)
        else: os.environ['CRISPEMBED_ULTRABERT_QUANT'] = value
        bad = lib.crispembed_init(str(model).encode(), 2)
        if bad: lib.crispembed_free(bad)
        require(not bad, 'quant model accepted with quant gate off')
        report['quant_gate_rejections'][str(value)] = True
    os.environ['CRISPEMBED_ULTRABERT_QUANT'] = '1'
    ctx = lib.crispembed_init(str(model).encode(), 2); require(ctx, 'reviewed genuine Q4 model rejected')
    began = time.monotonic()
    try:
        for i in range(start, len(rows)):
            require(time.monotonic() - began < 18000, 'bounded five-hour shard deadline; committed rows retained')
            require(shutil.disk_usage(args.out).free >= 2 * 2**30, 'output disk reserve exhausted; committed rows retained')
            row = rows[i]; text = ((old.QUERY if row['id'].startswith('q:') else '') + row['text']).encode()
            count = ctypes.c_int(); dim = ctypes.c_int()
            ptr = lib.crispembed_encode_tokens_raw(ctx, text, ctypes.byref(count), ctypes.byref(dim))
            require(bool(ptr) and dim.value == WIDTH, 'raw public CLS rejected: ' + row['id'])
            ids = lib.crispembed_last_token_ids(ctx)
            require(bool(ids) and count.value == len(expected[row['id']]['ids']) and list(ids[:count.value]) == expected[row['id']]['ids'], 'public raw token IDs differ')
            raw = np.ctypeslib.as_array(ptr, shape=(count.value * WIDTH,))[:WIDTH].copy()
            ptr = lib.crispembed_encode(ctx, text, ctypes.byref(dim)); require(bool(ptr) and dim.value == WIDTH, 'normalized public output rejected')
            norm = np.ctypeslib.as_array(ptr, shape=(WIDTH,)).copy()
            ids = lib.crispembed_last_token_ids(ctx)
            require(bool(ids) and list(ids[:count.value]) == expected[row['id']]['ids'] and np.isfinite(raw).all() and np.isfinite(norm).all(), 'normalized IDs/nonfinite output rejected')
            commit_row(args.out, dict(index=i, global_index=take[i], id=row['id'], input_sha256=row['input_sha256'], actual_ids_exact=True),
                       raw.astype('<f4').tobytes(), norm.astype('<f4').tobytes())
            report.update(completed_rows=i+1, total_rows=len(rows), elapsed_this_process=time.monotonic()-began, resumed_committed_rows=start)
            write(args.out / 'shard.json', report)
    finally: lib.crispembed_free(ctx)
    original_raw = np.memmap(frozen / f'f32/{args.shard}/original-raw.f32', mode='r', dtype='<f4', shape=(len(rows), WIDTH))
    original_norm = np.memmap(frozen / f'f32/{args.shard}/original-normalized.f32', mode='r', dtype='<f4', shape=(len(rows), WIDTH))
    native_raw = np.memmap(args.out / 'raw.f32', mode='r', dtype='<f4', shape=(len(rows), WIDTH))
    native_norm = np.memmap(args.out / 'normalized.f32', mode='r', dtype='<f4', shape=(len(rows), WIDTH))
    report['actual_ref'] = compare(old, args.diff, args.out / 'comparison', original_raw, original_norm, native_raw, native_norm)
    report.update(completed=True, completed_rows=len(rows), total_rows=len(rows), recipe=FIRST_RECIPE, shard=args.shard, all_ids_exact=True,
                  streams={k: identity(args.out / k) for k in ['raw.f32', 'normalized.f32', 'rows.jsonl']},
                  strict_f32_vector_flags=report['actual_ref']['exit'] == 0, quantized_qualified=False,
                  original_forwards=0, f32_forwards=0)


def aggregate(args, old, report):
    import numpy as np
    frozen, freeze = frozen_inputs(args, old)
    model_pin = approval('model')
    pins = read(PROTOCOL)['reviewed_dependencies'].get('shards')
    require(isinstance(pins, list) and len(pins) == SHARDS and len(args.shard_artifacts) == SHARDS,
            'eight explicit reviewed complete shard artifacts required')
    raw = np.empty((ROWS, WIDTH), dtype='<f4'); norm = np.empty_like(raw)
    original_raw = np.empty_like(raw); original_norm = np.empty_like(raw)
    report['shards'] = []
    data = read(frozen / 'dataset.json'); allrows = data['queries'] + data['documents']
    expected = {r['id']: r for r in read(frozen / 'lengths.json')}
    for i, (ident, pin) in enumerate(zip(args.shard_artifacts, pins)):
        require(pin.get('reviewed') is True and pin.get('shard') == i, 'reviewed shard ordering differs')
        root = artifact(old, ident, args.scratch / ('shard-' + str(i)), pin, 128 * 2**20)
        checked_file(root / 'shard.json', dict(bytes=pin['receipt_bytes'], sha256=pin['receipt_sha256']))
        r = read(root / 'shard.json'); take = indices(i)
        require(r['completed'] and r['recipe'] == FIRST_RECIPE and r['shard'] == i and r['completed_rows'] == len(take)
                and r['all_ids_exact'] and not r['quantized_qualified'], 'incomplete or mismatched shard')
        binding = read(root / 'binding.json')
        require(binding['frozen_receipt_sha256'] == sha(frozen / 'freeze.json')
                and binding['model_receipt_sha256'] == model_pin['receipt_sha256']
                and binding['source']['runtime_source'] == RUNTIME
                and binding['source']['runtime_git_objects'] == read(PROTOCOL)['runtime_git_objects'], 'shard source/freeze/model binding differs')
        require(binding['inputs'] == [dict(id=allrows[j]['id'], input_sha256=row_identity(allrows[j], expected)) for j in take], 'shard frozen input identities differ')
        records = [json.loads(x) for x in (root / 'rows.jsonl').read_text().splitlines()]
        require(len(records) == len(take), 'all shard journal rows required')
        for j, record in enumerate(records): require(record['global_index'] == take[j] and record['index'] == j and record['id'] == binding['inputs'][j]['id']
                                                    and record['input_sha256'] == binding['inputs'][j]['input_sha256'] and record['actual_ids_exact'], 'shard row coverage differs')
        for kind, target, original in [('raw', raw, original_raw), ('normalized', norm, original_norm)]:
            checked_file(root / (kind + '.f32'), r['streams'][kind + '.f32'])
            require((root / (kind + '.f32')).stat().st_size == len(take)*WIDTH*4, 'shard extent differs')
            target[take] = np.memmap(root / (kind + '.f32'), mode='r', dtype='<f4', shape=(len(take), WIDTH))
            original[take] = np.memmap(frozen / f'f32/{i}/original-{kind}.f32', mode='r', dtype='<f4', shape=(len(take), WIDTH))
        metric_agreement(r['actual_ref']['independent'], dict(raw=old.metrics(original_raw[take], raw[take]), normalized=old.metrics(original_norm[take], norm[take])))
        require({s['name'] for s in r['actual_ref']['result']['stages']} == {'raw', 'normalized'}, 'shard actual Ref coverage differs')
        report['shards'].append(dict(shard=i, receipt_sha256=sha(root / 'shard.json'), model=binding['model'], actual_ref=r['actual_ref']))
    require(len({r['model']['sha256'] for r in report['shards']}) == 1, 'mixed recipe model SHAs rejected')
    data = read(frozen / 'dataset.json')
    report['raw'] = old.metrics(original_raw, raw); report['normalized'] = old.metrics(original_norm, norm)
    report['quality'] = old.quality(old.rankings(data, original_norm), old.rankings(data, norm))
    report.update(completed=True, rows=ROWS, queries=300, recipe=FIRST_RECIPE,
                  strict_f32_vector_flags=report['raw']['f32_gate_passed'] and report['normalized']['f32_gate_passed'],
                  quantized_task_qualified=report['quality']['passed'], release_qualified=False, publication_qualified=False,
                  original_forwards=0, f32_forwards=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['freeze', 'model', 'shard', 'aggregate'])
    for name in ['runtime', 'out', 'scratch', 'diff']: p.add_argument('--' + name, type=Path, required=True)
    for name in ['library', 'quantizer']: p.add_argument('--' + name, type=Path)
    p.add_argument('--reviewed-source', required=True)
    p.add_argument('--freeze-artifact', type=int, default=0); p.add_argument('--model-artifact', type=int, default=0)
    p.add_argument('--resume-artifact', type=int, default=0)
    p.add_argument('--shard', type=int, default=0); p.add_argument('--shard-artifacts', type=int, nargs='*', default=[])
    args = p.parse_args(); args.scratch.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(schema='crispembed-ultrabert-q4k-sharded-v1', phase=args.phase, completed=False,
                  original_forwards=0, f32_forwards=0, release_qualified=False, publication_qualified=False)
    receipt = args.out / ('freeze.json' if args.phase == 'freeze' else args.phase + '.json')
    try:
        report['source'] = source(args.runtime, args.reviewed_source)
        report['resources'] = admission(args.out)
        report['build'] = build_proof(args)
        # Dependencies reject before optional numerical imports, model fetch or API.
        if args.phase != 'freeze': approval('freeze')
        if args.phase == 'shard':
            approval('model')
            active = read(PROTOCOL)['activation']
            require(active['reviewed'] is True and active['recipe'] == FIRST_RECIPE and args.shard in active['shards'],
                    'pending explicit shard activation; no model access')
        old = old_module(args.runtime)
        globals()[{'freeze': 'retained_freeze', 'model': 'prepare_model', 'shard': 'shard', 'aggregate': 'aggregate'}[args.phase]](args, old, report)
        write(receipt, report); write(args.out / 'manifest.json', manifest(args.out))
        return 0
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        # Preserve the journal and streams on every structural/numerical failure.
        write(receipt, report); write(args.out / 'manifest.json', manifest(args.out)); return 1


if __name__ == '__main__':
    raise SystemExit(main())
