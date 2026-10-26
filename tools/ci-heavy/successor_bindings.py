"""Hosted Linux prebuilt Rust/Dart consumers; never compile native sources/download models."""
import argparse
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools/ci-heavy'))
import successor_assets as assets
from successor_asset_manifest import sha, safe_name

TEXTS = [assets.TEXT, 'A second packaged binding check.', 'Unicode café Ελληνικά']
BATCH_ORDER = [2, 0, 1, 0]
CONTRACTS = {'dimensions', 'repeat_exact', 'offline_cached_path', 'missing_offline_rejected', 'actual_prebuilt_mapped'}
FFI_SHA = '289279317b4b16eb2bb7e271abccd4bf84ec9bdcbe999e278a94b804f5630418'
RUST_FILES = ('crispembed/Cargo.toml', 'crispembed/src/lib.rs', 'crispembed-sys/Cargo.toml',
              'crispembed-sys/build.rs', 'crispembed-sys/src/lib.rs', 'README.md', 'LICENSE')
DART_FILES = ('flutter/crispembed/lib/src/crispembed.dart', 'flutter/crispembed/lib/src/crispembed_bindings.dart')
AUDIT_FILES = ('tests/successor_bindings.rs', 'tests/successor_bindings.dart',
               'tools/ci-heavy/successor_bindings.py', '.github/workflows/successor-bindings.yml',
               'tools/ci-heavy/successor_assets.py', 'tools/successor_asset_manifest.py', 'Cargo.lock', 'src/crispembed.h')


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def compare(a, b):
    if (len(a) != 384 or len(b) != 384 or any(type(v) not in (float, int) or not math.isfinite(v) for v in a + b)):
        raise ValueError('Invalid/missing/nonfinite384-vector')
    an = math.sqrt(math.fsum(v * v for v in a)); bn = math.sqrt(math.fsum(v * v for v in b))
    if min(an, bn) <= 0 or abs(an - 1) > .002 or abs(bn - 1) > .002:
        raise ValueError('Zero/non-normalized vector')
    error = [abs(x - y) for x, y in zip(a, b)]
    cosine = math.fsum(x * y for x, y in zip(a, b)) / (an * bn)
    maximum = max(error)
    return dict(max_abs=maximum, cosine=cosine, mean_abs=math.fsum(error) / 384,
                rms=math.sqrt(math.fsum(v * v for v in error) / 384),
                reference_norm=an, native_norm=bn, passed=maximum <= .002 and cosine >= .99999)


def comparisons(reference, result):
    if (set(result) != {'scalar', 'repeat', 'batch', 'contracts'} or set(result['contracts']) != CONTRACTS
            or not all(v is True for v in result['contracts'].values())
            or len(result['scalar']) != 3 or len(result['repeat']) != 1 or len(result['batch']) != 4):
        raise ValueError('Incomplete actual binding vectors/contracts')
    if result['repeat'][0] != result['scalar'][0]:
        raise ValueError('Repeat changed actual output')
    rows = [dict(mode='scalar', row=i, **compare(reference['scalar'][i], v)) for i, v in enumerate(result['scalar'])]
    rows += [dict(mode='repeat', row=0, **compare(reference['scalar'][0], result['repeat'][0]))]
    for i, index in enumerate(BATCH_ORDER):
        rows.append(dict(mode='batch_vs_cabi_scalar', row=i, source_row=index,
                         **compare(reference['scalar'][index], result['batch'][i])))
        rows.append(dict(mode='batch_vs_cabi_batch', row=i, source_row=index,
                         **compare(reference['batch'][i], result['batch'][i])))
    return rows


def checked_ldd(text, package, library):
    found = []
    for line in text.splitlines():
        if 'not found' in line:
            raise ValueError('Unresolved prebuilt dependency')
        words = line.split()
        if words and (words[0].startswith('libcrispembed.so') or words[0].startswith('libggml')):
            if len(words) < 3 or words[1] != '=>':
                raise ValueError('Missing resolved package library')
            target = Path(words[2]).resolve()
            if target.parent != package.resolve():
                raise ValueError('Package dependency fallback')
            if words[0].startswith('libcrispembed.so'):
                found.append(target)
    if found != [library.resolve()]:
        raise ValueError('Rust did not resolve exactly the candidate prebuilt')


def cabi(model, library, output):
    # Isolated child avoids native global teardown in the orchestrator process.
    lib = C.CDLL(str(library.resolve()))
    lib.crispembed_set_offline.argtypes = [C.c_int]
    lib.crispembed_set_offline(1)
    lib.crispembed_is_offline.restype = C.c_int
    if lib.crispembed_is_offline() != 1:
        raise ValueError('C ABI offline state absent')
    lib.crispembed_init.argtypes = [C.c_char_p, C.c_int]; lib.crispembed_init.restype = C.c_void_p
    lib.crispembed_free.argtypes = [C.c_void_p]
    lib.crispembed_encode.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int)]
    lib.crispembed_encode.restype = C.POINTER(C.c_float)
    lib.crispembed_encode_batch.argtypes = [C.c_void_p, C.POINTER(C.c_char_p), C.c_int, C.POINTER(C.c_int)]
    lib.crispembed_encode_batch.restype = C.POINTER(C.c_float)
    ctx = lib.crispembed_init(os.fsencode(model), 2)
    if not ctx:
        raise ValueError('Candidate C ABI model initialization failed')
    try:
        def encode(text):
            dim = C.c_int()
            pointer = lib.crispembed_encode(ctx, text.encode(), C.byref(dim))
            if not pointer or dim.value != 384:
                raise ValueError('C ABI scalar shape failed')
            return [float(pointer[i]) for i in range(384)]
        scalar = [encode(text) for text in TEXTS]; repeat = encode(TEXTS[0])
        dim = C.c_int(); texts = [TEXTS[i].encode() for i in BATCH_ORDER]
        ptr = lib.crispembed_encode_batch(ctx, (C.c_char_p * 4)(*texts), 4, C.byref(dim))
        if not ptr or dim.value != 384 or repeat != scalar[0]:
            raise ValueError('C ABI batch/repeat contract failed')
        batch = [[float(ptr[i * 384 + j]) for j in range(384)] for i in range(4)]
        reference = dict(scalar=scalar, repeat=[repeat], batch=batch)
        for vector in scalar + [repeat] + batch:
            compare(vector, vector)
        write(output, reference)
    finally:
        lib.crispembed_free(ctx)


def fixture(run_id, producer, destination):
    service = assets.api('actions/runs/' + str(run_id))
    if (service['path'] not in ('.github/workflows/successor-assets.yml', '.github/workflows/ovisocr2-metadata.yml')
            or service['event'] != 'workflow_dispatch'):
        raise ValueError('Wrong consumer fixture producer workflow/event')
    if str(run_id) != os.environ['GITHUB_RUN_ID'] and (service['status'] != 'completed' or service['conclusion'] != 'success'):
        raise ValueError('Prior fixture producer must complete successfully')
    destination.mkdir()
    identity = assets.artifact(run_id, 'successor-minilm-fixture-' + str(run_id), destination / 'fixture.zip', 96 * 1024 ** 2)
    with zipfile.ZipFile(destination / 'fixture.zip') as z:
        names = z.namelist()
        if (len(names) != len(set(names)) or sum(i.file_size for i in z.infolist()) > 128 * 1024 ** 2
                or any(safe_name(n) != n or n.endswith('/') for n in names)):
            raise ValueError('Unsafe fixture ZIP')
        for info in z.infolist():
            if info.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError('Fixture ZIP symlink forbidden')
        z.extractall(destination)
    models = list(destination.rglob('all-MiniLM-L6-v2-iq4_xs.gguf'))
    receipts = list(destination.rglob('consumer.json'))
    if len(models) != 1 or len(receipts) != 1 or len(names) != 2 or sha(models[0]) != assets.MODEL_SHA:
        raise ValueError('Retained fixture model/member identity differs')
    receipt = json.loads(receipts[0].read_text())
    if (receipt.get('schema') != 'crispembed-successor-fresh-consumer-v2'
            or receipt.get('consumer_source') != service['head_sha'] or receipt.get('source_commit') != producer['source_commit']
            or receipt.get('producer_run') != producer['producer_run'] or receipt.get('payload_artifact') != producer['payload_artifact']
            or receipt.get('model_sha256') != assets.MODEL_SHA or receipt.get('text') != TEXTS[0]
            or receipt.get('desktop_bounded_consumer_positive') is not True or receipt.get('offline_reuse') is not True
            or receipt.get('empty_cache_download') is not True or receipt.get('empty_cache_offline_rejected') is not True
            or not all(v.get('passed') is True for v in receipt.get('comparisons', {}).values())
            or set(receipt.get('comparisons', {})) != {'cli_to_cabi', 'offline_cli', 'python_to_cabi', 'compiled_C_to_cabi'}):
        raise ValueError('Fixture is not the actual qualified candidate desktop output')
    compare(receipt['vector'], receipt['vector'])
    return models[0].resolve(), receipt, dict(run=int(run_id), head=service['head_sha'], artifact=identity,
        consumer_receipt_sha256=sha(receipts[0]))


def execute(args):
    if os.environ.get('GITHUB_ACTIONS') != 'true' or platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise ValueError('Hosted Linux x86_64 execution only')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    scratch = args.scratch.resolve(); scratch.mkdir(parents=True, exist_ok=False)
    report = dict(schema='crispembed-successor-prebuilt-bindings-v1', status='started',
        audit_source=os.environ['GITHUB_SHA'], source_sha256={p:sha(ROOT / p) for p in RUST_FILES + DART_FILES + AUDIT_FILES},
        protocol=dict(model_sha256=assets.MODEL_SHA, texts=TEXTS, batch_order=BATCH_ORDER, threads=2,
                      dimension=384, max_abs=.002, cosine_min=.99999, repeat='bit-exact scalar repeat'),
        limits=['Actual Linux CPU MiniLM prebuilt binding consumers only; no native source build.',
                'No new model download; verified retained consumer fixture reused offline.',
                'No experimental-model, task-quality, GPU, release publication or application-pin qualification.'],
        surfaces={})
    def save():
        write(out / 'successor-bindings.json', report)
    def run(command, name, *, cwd=ROOT, env=None, timeout=1200):
        with (out / (name + '.log')).open('w') as log, (out / (name + '.stderr.log')).open('w') as errors:
            result = subprocess.run([str(x) for x in command], cwd=cwd, env=env,
                                    stdout=log, stderr=errors, timeout=timeout)
        for suffix in ('.log', '.stderr.log'):
            report.setdefault('logs', {})[name + suffix] = dict(sha256=sha(out / (name + suffix)), exit_code=result.returncode)
        save()
        if result.returncode:
            raise ValueError(name + ' failed; actual log retained')
        return (out / (name + '.log')).read_text()
    save()
    try:
        producer = assets.prepare(args.native_run, 'native', 'linux-x86_64', scratch / 'candidate')
        if producer['consumer_source'] != os.environ['GITHUB_SHA']:
            raise ValueError('Audit checkout differs from actual workflow source')
        report['producer'] = producer
        model, prior, lineage = fixture(args.consumer_run, producer, scratch / 'fixture')
        if prior['runtime_source_closure']['objects'] != producer['runtime_source_closure']['objects']:
            raise ValueError('Fixture production closure differs')
        report['fixture'] = lineage
        report['fixture_reference_sha256'] = hashlib.sha256(json.dumps(prior['vector'], separators=(',', ':')).encode()).hexdigest()
        packages = list((scratch / 'candidate').glob('crispembed-linux-x86_64'))
        if len(packages) != 1:
            raise ValueError('Exactly one packaged Linux CPU library required')
        package = packages[0]; library = (package / 'libcrispembed.so').resolve(strict=True)
        if library.parent != package.resolve() or sha(model) != assets.MODEL_SHA:
            raise ValueError('Library/model path escapes the verified candidate')
        report.update(library_sha256=sha(library), model_sha256=sha(model))
        cache = scratch / 'empty-offline-cache'; cache.mkdir()
        request = scratch / 'requests.txt'; request.write_text('\n'.join(TEXTS) + '\n')
        report['request_sha256'] = sha(request)
        env = {k:v for k,v in os.environ.items() if not k.startswith('CRISPEMBED_')}
        env.update(CRISPEMBED_SYS_LIB_DIR=str(package.resolve()), CRISPEMBED_OFFLINE='1', HF_HUB_OFFLINE='1',
                   CRISPEMBED_FORCE_CPU='1', CRISPEMBED_CACHE_DIR=str(cache), LD_LIBRARY_PATH=str(package.resolve()),
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', CARGO_BUILD_JOBS='2')
        report['environment'] = {k:env[k] for k in ('CRISPEMBED_OFFLINE','HF_HUB_OFFLINE','CRISPEMBED_FORCE_CPU',
            'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CARGO_BUILD_JOBS')}
        reference_file = out / 'cabi-reference.json'
        run([sys.executable, __file__, '--cabi', '--model', model, '--library', library, '--output', reference_file], 'cabi', env=env)
        reference = json.loads(reference_file.read_text())
        prior_check = compare(prior['vector'], reference['scalar'][0])
        report['frozen_consumer_cabi_replay'] = prior_check
        if not prior_check['passed']:
            raise ValueError('Current C ABI differs from retained exact-candidate reference')
        # Clone only unchanged Rust wrapper sources/manifests. No native source,
        # CMakeLists or ggml is present at either build.rs source-root probe.
        wrappers = scratch / 'wrappers'; wrappers.mkdir()
        for name in RUST_FILES:
            target = wrappers / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
            if sha(target) != report['source_sha256'][name]:
                raise ValueError('Copied real Rust wrapper bytes differ')
        rust = scratch / 'rust'; (rust / 'src').mkdir(parents=True)
        shutil.copy2(ROOT / 'tests/successor_bindings.rs', rust / 'src/main.rs')
        (rust / 'Cargo.toml').write_text('[package]\nname="successor-binding-probe"\nversion="0.0.0"\nedition="2021"\n'
            '[dependencies]\ncrispembed={path="../wrappers/crispembed"}\n[workspace]\n')
        # Preserve the actual root dependency lock, adding only this test package.
        (rust / 'Cargo.lock').write_bytes((ROOT / 'Cargo.lock').read_bytes() + b'\n[[package]]\nname = "successor-binding-probe"\nversion = "0.0.0"\ndependencies = [\n "crispembed",\n]\n')
        shutil.copy2(rust / 'Cargo.lock', out / 'consumer-Cargo.lock')
        shutil.copy2(rust / 'Cargo.toml', out / 'consumer-Cargo.toml')
        deny = scratch / 'deny-native-build'; deny.mkdir()
        shim = deny / 'cmake'; shim.write_text('#!/bin/sh\nprintf "%s\\n" "forbidden native CMake fallback" >&2\nexit 97\n'); shim.chmod(0o700)
        env['PATH'] = str(deny) + os.pathsep + env['PATH']
        run(['cargo','fetch','--locked','--manifest-path',rust / 'Cargo.toml'], 'rust-dependencies', env=env)
        run(['cargo','build','--locked','--offline','--manifest-path',rust / 'Cargo.toml'], 'rust-build', env=env)
        binary = rust / 'target/debug/successor-binding-probe'
        if list(rust.rglob('crispembed-build')) or list(wrappers.rglob('CMakeLists.txt')) or list(wrappers.rglob('*.cpp')):
            raise ValueError('Native source fallback/build detected')
        checked_ldd(run(['ldd', binary], 'rust-ldd', env=env), package, library)
        build_outputs = list((rust / 'target/debug/build').glob('crispembed-sys-*/output'))
        if len(build_outputs) != 1 or 'cargo:LIB_DIR=' + str(package.resolve()) not in build_outputs[0].read_text().splitlines():
            raise ValueError('Actual sys build script did not select the prebuilt package')
        shutil.copy2(build_outputs[0], out / 'rust-sys-build-output.txt')
        shutil.copy2(rust / 'Cargo.lock', out / 'consumer-Cargo.lock')
        shutil.copy2(rust / 'Cargo.toml', out / 'consumer-Cargo.toml')
        shutil.copy2(binary, out / 'successor-rust-probe')
        dart = scratch / 'dart'; (dart / 'lib').mkdir(parents=True)
        for name in DART_FILES:
            target = dart / 'lib' / Path(name).name; shutil.copy2(ROOT / name, target)
            if sha(target) != report['source_sha256'][name]:
                raise ValueError('Copied real Dart wrapper bytes differ')
        shutil.copy2(ROOT / 'tests/successor_bindings.dart', dart / 'main.dart')
        (dart / 'pubspec.yaml').write_text("name: successor_binding_probe\nenvironment:\n  sdk: '>=3.7.0 <4.0.0'\ndependencies:\n  ffi: 2.1.4\n")
        (dart / 'pubspec.lock').write_text('packages:\n  ffi:\n    dependency: "direct main"\n    description:\n      name: ffi\n'
            '      sha256: "' + FFI_SHA + '"\n      url: "https://pub.dev"\n    source: hosted\n    version: "2.1.4"\nsdks:\n  dart: ">=3.7.0 <4.0.0"\n')
        shutil.copy2(dart / 'pubspec.lock', out / 'consumer-pubspec.lock')
        shutil.copy2(dart / 'pubspec.yaml', out / 'consumer-pubspec.yaml')
        env['PUB_CACHE'] = str(scratch / 'pub-cache')
        run(['dart','--disable-analytics'], 'dart-analytics', env=env)
        run(['dart','pub','get','--enforce-lockfile'], 'dart-dependencies', env=env, cwd=dart)
        run(['dart','analyze','--no-fatal-warnings'], 'dart-analyze', env=env, cwd=dart)
        lock = (dart / 'pubspec.lock').read_text()
        if FFI_SHA not in lock or 'version: "2.1.4"' not in lock or 'flutter:' in lock:
            raise ValueError('Dart dependency lock differs or pulls Flutter')
        shutil.copy2(dart / 'pubspec.lock', out / 'consumer-pubspec.lock')
        shutil.copy2(dart / 'pubspec.yaml', out / 'consumer-pubspec.yaml')
        dart_binary = dart / 'successor-dart-probe'
        run(['dart','compile','exe',dart / 'main.dart','-o',dart_binary], 'dart-build', env=env, cwd=dart)
        shutil.copy2(dart_binary, out / 'successor-dart-probe')
        report['dart_binary_sha256'] = sha(dart_binary)
        report['compilers'] = dict(rust=run(['rustc','--version','--verbose'], 'rust-version', env=env).strip(),
            cargo=run(['cargo','--version'], 'cargo-version', env=env).strip(),
            dart=run(['dart','--version'], 'dart-version', env=env).strip())
        if not report['compilers']['dart']:
            report['compilers']['dart'] = (out / 'dart-version.stderr.log').read_text().strip()
        for name in ('rustc','cargo'):
            tool = Path(run(['rustup','which',name], name + '-path', env=env).strip())
            report['compilers'][name + '_binary_sha256'] = sha(tool)
        report['compilers']['dart_binary_sha256'] = sha(Path(shutil.which('dart', path=env['PATH'])).resolve())
        report['locks'] = {name:sha(out / name) for name in ('consumer-Cargo.lock','consumer-Cargo.toml','consumer-pubspec.lock','consumer-pubspec.yaml')}
        report['native_build_fallback'] = False
        for language, command in [('rust',[binary]), ('dart',[dart_binary])]:
            log = run(command + [model, request, library], language + '-actual', env=env, cwd=dart if language == 'dart' else rust)
            # Compiled consumers emit JSON on stdout; native diagnostics remain
            # in their separately retained stderr logs. No package lookup at inference.
            actual = json.loads(log)
            write(out / (language + '-actual.json'), actual)
            measured = comparisons(reference, actual)
            report['surfaces'][language] = dict(comparisons=measured, contracts=actual['contracts'],
                passed=all(r['passed'] for r in measured), actual_sha256=sha(out / (language + '-actual.json')))
            save()
        if list(cache.iterdir()) or sha(model) != assets.MODEL_SHA or sha(library) != report['library_sha256']:
            raise ValueError('Binding execution mutated fixture/library or populated empty offline cache')
        report.update(status='completed bounded prebuilt Rust/Dart consumers',
            passed=all(v['passed'] for v in report['surfaces'].values()), offline_cache_unchanged=True,
            rust_binary_sha256=sha(binary), cabi_reference_sha256=sha(reference_file))
        save()
        return int(not report['passed'])
    except BaseException as error:
        message = str(error)
        for path,label in ((str(ROOT),'<repo>'),(str(scratch),'<scratch>'),(str(out),'<out>')):
            message = message.replace(path,label)
        report.update(status='failed prebuilt binding producer', error_type=type(error).__name__, error=message)
        save()
        raise


class Controls(unittest.TestCase):
    def test_vectors_and_retained_numeric_failure(self):
        unit = [1.] + [0.] * 383
        self.assertTrue(compare(unit, unit)['passed'])
        changed = [math.sqrt(1 - .01 ** 2), .01] + [0.] * 382
        self.assertFalse(compare(unit, changed)['passed'])
        for bad in ([0.] * 384, [float('nan')] * 384, [1.] * 383):
            with self.assertRaises(ValueError): compare(unit,bad)
    def test_actual_coverage_contracts_and_batch_order(self):
        vectors = [[float(i == n) for i in range(384)] for n in range(3)]
        ref = dict(scalar=vectors,batch=[vectors[i] for i in BATCH_ORDER])
        result = dict(scalar=vectors,repeat=[vectors[0]],batch=ref['batch'],contracts={k:True for k in CONTRACTS})
        self.assertEqual(len(comparisons(ref,result)),12)
        self.assertTrue(all(r['passed'] for r in comparisons(ref,result)))
        swapped = dict(result,batch=list(reversed(ref['batch'])))
        self.assertFalse(all(r['passed'] for r in comparisons(ref,swapped)))
        with self.assertRaises(ValueError): comparisons(ref,dict(result,repeat=[]))
        with self.assertRaises(ValueError): comparisons(ref,dict(result,contracts={}))
    def test_ldd_missing_fallback_and_expected_package(self):
        with tempfile.TemporaryDirectory() as d:
            package=Path(d); library=package / 'libcrispembed.so.0.17.14'; library.touch()
            good='libcrispembed.so.0 => ' + str(library) + ' (0x123)\nlibggml.so => ' + str(package / 'libggml.so') + ' (0x234)'
            checked_ldd(good,package,library)
            for bad in ('libcrispembed.so.0 => not found',good.replace(str(library),'/other/libcrispembed.so.0'), ''):
                with self.assertRaises(ValueError): checked_ldd(bad,package,library)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-run',type=int); parser.add_argument('--consumer-run',type=int)
    parser.add_argument('--output',type=Path); parser.add_argument('--scratch',type=Path)
    parser.add_argument('--self-test',action='store_true'); parser.add_argument('--cabi',action='store_true')
    parser.add_argument('--model',type=Path); parser.add_argument('--library',type=Path)
    args=parser.parse_args()
    if args.self_test:
        return not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls)).wasSuccessful()
    if args.cabi:
        if os.environ.get('GITHUB_ACTIONS')!='true' or platform.system()!='Linux':
            raise ValueError('Hosted C ABI child only')
        cabi(args.model,args.library,args.output)
        sys.stdout.flush();sys.stderr.flush();os._exit(0)
    if not all((args.native_run,args.consumer_run,args.output,args.scratch)):
        parser.error('native-run, consumer-run, output and scratch are required')
    return execute(args)


if __name__=='__main__': raise SystemExit(main())
