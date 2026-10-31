#!/usr/bin/env python3
"""Kaggle CPU-only OvisOCR2 direct-F16 decoded regression; no build or publish.

Requires a separately staged checkout at the pinned base, with only the pinned
qwen35_linear.cpp overlay in the build closure, initialized pinned ggml, and a
fresh CPU build of test-ovisocr2-quality-run under /kaggle/temp. Set
OVIS_BUILD to its build directory. Install Pillow==12.1.0 and huggingface-hub.
The companion kernel must retain its own staging/build commands and digests;
this script cannot attest that a prebuilt binary came from the checked sources.
"""
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
TEMP = Path('/kaggle/temp')
WORKING = Path('/kaggle/working')
BASE = '9498e67e3558a9a0b0ec20f93c40698b2434204e'
GGML = 'a06fc0bb46b73cc10011fc90ae4054aee8438aeb'
MODEL_REPO = 'cstr/OvisOCR2-GGUF'
MODEL_REVISION = 'b7229e46f31bcf342d0999966bae8a03bf02bb07'
MODEL_NAME = 'ovisocr2-f16.gguf'
MODEL_SHA = 'd3a733a5d5ba5d63f5275500fad4cff585b82e2fb90d1def2fe767d04edf57db'
MODEL_BYTES = 1717222848
RECEIPT = 'tests/results/qwen35-ocr/ovisocr2-f16-quality.json'
PANEL = 'tests/results/qwen35-ocr/ovisocr2-quality-panel.json'
CONTRACT = 'tools/ci-heavy/ovisocr2_quality_contract.py'
SOURCE_PINS = {
    'src/qwen35_linear.cpp': '38e342810caff1aca42dfdae1bff67ed4fddc60049fcfbd68d7f01f076275af3',
    RECEIPT: 'ff68a1cd4c4cc74b5092b748a69bd61c29b80c823b1d4219985ee3595c508bd6',
    PANEL: '3c576337645193e4347d833c185e70c81b81e3b986d8b3e8664a4fd6cdd4f8a9',
    CONTRACT: 'c1e77fd2a20bb2aeae3e09f030df13389891d188957cb7db78488c9d4460b8b3',
    'tools/ci-heavy/ovisocr2_vision_env.py': 'bfa6f9f6b22720d9bae8a8b1f6eb3a32ae2598e1453560edec5b34c2f6b54a06',
    'tests/ovisocr2_quality_run.cpp': '07034383eb2d9f19af5b0fe5f08094f77b311025a091a10b2b832a92bbe5368e',
}
PRODUCER_SHA = '0eb71f5281d781c043013a367c0217ac2e4ba57fb80defe395b003dff1975239'
CASES = ('dev-hello', 'dev-invoice', 'dev-menu', 'dev-section',
         'hold-batch', 'hold-paid', 'hold-date', 'hold-code')
GATES = {
    'CRISPEMBED_QWEN35_OCR': '1', 'CRISPEMBED_OVISOCR2_FRONTEND': '1',
    'CRISPEMBED_OVISOCR2_INFERENCE': '1', 'CRISPEMBED_QWEN35_QUANT': '0',
    'CRISPEMBED_QWEN35_WEIGHT_CACHE': '0', 'CRISPEMBED_QWEN35_PARALLEL_F16': '0',
    'CRISPEMBED_QWEN35_PRECISE_RMS': '0', 'CRISPEMBED_QWEN35_IMATRIX': '0',
    'CRISPEMBED_QWEN35_PROFILE': '0', 'CRISPEMBED_QWEN35_TOKENIZER_MARKS': '0',
    'CRISPEMBED_QWEN35_VISION_PRECISE_NORM': '1',
    'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN': '1',
    'CRISPEMBED_QWEN35_VISION_CASCADE_NORM': '0',
    'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
}
REPORT = {
    'schema': 'ovisocr2-direct-f16-decoded-regression-v1', 'status': 'incomplete',
    'phase': 'startup', 'cases': {}, 'requested_cases': list(CASES),
    'scope': 'Eight self-authored CC0 synthetic text images, CPU, two threads, cache off; exact decoded regression only.',
    'limitations': [
        'This is a new dirty-source experiment, not the historical F16 source-closure validation or eleven-case host control.',
        'Only eight synthetic images are run; the three reviewed documents and broad OCR accuracy are not tested.',
        'The Ovis quality executable does not expose operator dispatch counters; setting the direct gate does not establish that the direct path ran.',
        'No speed or profiler measurement, intermediate numerical parity, other model profiles, release qualification, or default change is established.',
        'Prebuilt binary digest and build configuration are recorded, but source-to-binary linkage is not independently attested by this runner.',
        'Processed text is not computed; raw text, token IDs and EOS are the strict comparison.',
    ],
}
REPORT_PATH = None


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def save(phase):
    REPORT['phase'] = phase
    if REPORT_PATH is not None:
        tmp = REPORT_PATH.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(REPORT, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        tmp.replace(REPORT_PATH)
        print('OVIS_DIRECT_F16_PHASE', phase, flush=True)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()


def import_contract():
    spec = importlib.util.spec_from_file_location('ovis_quality_contract', ROOT / CONTRACT)
    if spec is None or spec.loader is None:
        raise RuntimeError('quality contract import unavailable')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preflight_source(build):
    head = git(ROOT, 'rev-parse', 'HEAD')
    link = git(ROOT, 'rev-parse', 'HEAD:ggml')
    ggml_head = git(ROOT / 'ggml', 'rev-parse', 'HEAD')
    if head != BASE or link != GGML or ggml_head != GGML:
        raise ValueError('base or initialized ggml revision differs')
    if git(ROOT / 'ggml', 'status', '--porcelain', '--untracked-files=all'):
        raise ValueError('ggml submodule has modifications or untracked files')
    # Require the checked-out base plus the single intended tracked overlay.
    # The standalone runner itself may be staged as an untracked file.
    changed = git(ROOT, 'diff', '--name-only', 'HEAD').splitlines()
    untracked_all = git(ROOT, 'ls-files', '--others', '--exclude-standard').splitlines()
    runner_rel = str(Path(__file__).resolve().relative_to(ROOT))
    untracked = [name for name in untracked_all if name != runner_rel]
    if changed != ['src/qwen35_linear.cpp'] or untracked:
        raise ValueError('unexpected checkout changes: ' + repr((changed, untracked)))
    files = {}
    for rel, expected in SOURCE_PINS.items():
        actual = sha(ROOT / rel)
        files[rel] = {'sha256': actual, 'expected_sha256': expected}
        if actual != expected:
            raise ValueError('source pin differs: ' + rel)
    # Record every tracked src file rather than pretending the old receipt's
    # source-closure validator applied to the dirty candidate.
    tracked_src = git(ROOT, 'ls-files', '--', 'src').splitlines()
    source_files = {name: sha(ROOT / name) for name in tracked_src if (ROOT / name).is_file()}
    binary = build / 'test-ovisocr2-quality-run'
    cache = build / 'CMakeCache.txt'
    if not binary.is_file() or not cache.is_file():
        raise ValueError('prebuilt Ovis binary or CMakeCache.txt absent')
    settings = {}
    for line in cache.read_text(errors='strict').splitlines():
        if '=' in line and ':' in line.split('=', 1)[0] and not line.startswith(('#', '//')):
            key, value = line.split('=', 1)
            settings[key.split(':', 1)[0]] = value
    required = {'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_HOME_DIRECTORY': str(ROOT),
                'CRISPEMBED_NATIVE': 'OFF', 'GGML_NATIVE': 'OFF', 'GGML_BLAS': 'OFF',
                'GGML_METAL': 'OFF', 'GGML_CUDA': 'OFF'}
    if any(settings.get(key) != value for key, value in required.items()):
        raise ValueError('build cache does not match required portable CPU configuration: ' +
                         repr({key: settings.get(key) for key in required}))
    REPORT['provenance'] = {
        'base_commit': head, 'ggml_gitlink': link, 'ggml_checkout': ggml_head,
        'build_closure_dirty_paths': changed, 'build_closure_untracked_paths': untracked,
        'source_pins': files, 'tracked_src_sha256': source_files,
        'cmake_source_sha256': sha(ROOT / 'CMakeLists.txt'),
        'runner_sha256': sha(Path(__file__)), 'binary_sha256': sha(binary),
        'binary_path': str(binary), 'cmake_cache_sha256': sha(cache),
        'cmake_required_values': required,
        'source_to_binary_attested': False,
        'stage_bundle_sha256_claim': os.environ.get('OVIS_SOURCE_BUNDLE_SHA256'),
    }
    return binary


def check_output(data, cap):
    ids, confidence = data['token_ids'], data['confidences']
    if (not isinstance(ids, list) or not 0 < len(ids) <= cap
            or any(type(x) is not int or not 0 <= x < 248320 for x in ids)
            or not isinstance(confidence, list) or len(confidence) != len(ids)
            or any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in confidence)
            or type(data['mean_confidence']) not in (int, float)
            or not math.isfinite(data['mean_confidence'])
            or not math.isclose(data['mean_confidence'], sum(confidence) / len(confidence), abs_tol=1e-6, rel_tol=0)
            or data['cap'] != cap or data['stop_kind'] != 1 or data['stop_id'] not in (248044, 248046)
            or ids[-1] != data['stop_id'] or any(x in (248044, 248046) for x in ids[:-1])):
        raise ValueError('invalid IDs, EOS or selected-token confidences')
    data['raw_text'] = bytes.fromhex(data['raw_text_utf8_hex']).decode('utf-8', 'strict')
    if not data['raw_text']:
        raise ValueError('empty native raw text')
    return data


def native(binary, model, image, cap, direct, log_path):
    env = {k: os.environ[k] for k in ('PATH', 'LD_LIBRARY_PATH', 'LANG', 'LC_ALL') if k in os.environ}
    env.update(GATES)
    env['CRISPEMBED_QWEN35_DIRECT_F16'] = str(direct)
    cmd = [str(binary), str(model), str(image), str(cap), '2']
    result = {'command': cmd, 'gates': {**GATES, 'CRISPEMBED_QWEN35_DIRECT_F16': str(direct)}}
    try:
        with log_path.open('wb') as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=1800)
        result['returncode'] = proc.returncode
    except subprocess.TimeoutExpired:
        result.update(returncode=None, error='native process timed out')
    result['log'] = log_path.name
    result['log_sha256'] = sha(log_path)
    if result.get('returncode') != 0:
        result.setdefault('error', 'native process failed')
        return result
    try:
        lines = [line[len('OVIS_QUALITY_RESULT '):] for line in log_path.read_text(errors='strict').splitlines()
                 if line.startswith('OVIS_QUALITY_RESULT ')]
        if len(lines) != 1:
            raise ValueError('expected exactly one native result')
        result['output'] = check_output(json.loads(lines[0]), cap)
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        result['error'] = 'invalid native output: ' + str(exc)
    return result


def exact(actual, reference):
    # Missing values must not compare equal; EOS is checked independently.
    keys = ('token_ids', 'raw_text', 'stop_id')
    if any(key not in actual or key not in reference for key in keys):
        raise ValueError('missing comparison field')
    return {key + '_exact': actual[key] == reference[key] for key in keys}


def main():
    global REPORT_PATH
    if not WORKING.is_dir() or not TEMP.is_dir() or not ROOT.resolve().is_relative_to(TEMP.resolve()):
        raise RuntimeError('requires Kaggle directories and source checkout under /kaggle/temp')
    out = Path(tempfile.mkdtemp(prefix='ovis-direct-f16-', dir=WORKING))
    REPORT_PATH = out / 'report.json'
    save('admission')
    scratch = Path(tempfile.mkdtemp(prefix='ovis-direct-f16-', dir=TEMP))
    os.environ.update(TMPDIR=str(scratch), HF_HOME=str(scratch / 'hf-home'),
                      HF_HUB_CACHE=str(scratch / 'hf-cache'), HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    tempfile.tempdir = str(scratch)
    REPORT['scratch'] = str(scratch)
    REPORT['free_bytes'] = {'temp': shutil.disk_usage(scratch).free, 'working': shutil.disk_usage(out).free}
    REPORT['host'] = {'platform': platform.platform(), 'machine': platform.machine(),
                      'logical_cpus': os.cpu_count(), 'threads': 2, 'python': sys.version.split()[0]}
    REPORT['versions'] = {name: importlib.metadata.version(name) for name in ('Pillow', 'huggingface-hub')}
    if REPORT['versions']['Pillow'] != '12.1.0':
        raise ValueError('Pillow 12.1.0 required')
    build = Path(os.environ.get('OVIS_BUILD', '/kaggle/temp/build-ovis')).resolve()
    if not build.is_relative_to(TEMP.resolve()):
        raise ValueError('build must be under /kaggle/temp')
    binary = preflight_source(build)
    save('source-and-build-provenance')
    contract = import_contract()
    panel = contract.load_panel()
    if tuple(row['id'] for row in panel['cases']) != CASES or panel['max_new_tokens'] != 256:
        raise ValueError('panel coverage or cap differs')
    receipt = json.loads((ROOT / RECEIPT).read_text())
    producer = receipt['producer']
    review = receipt['independent_review']
    if (canonical(producer) != PRODUCER_SHA or review['producer_canonical_sha256'] != PRODUCER_SHA
            or review['quality_positive'] is not True or review['f16_11_complete'] is not True
            or producer['models']['f16'] != {'sha256': MODEL_SHA, 'bytes': MODEL_BYTES}):
        raise ValueError('frozen independent producer or model proof differs')
    selected = producer['selected_vision_preset']
    if selected['name'] != 'both' or any(selected['values'].get(k) != GATES[k] for k in (
            'CRISPEMBED_QWEN35_VISION_PRECISE_NORM', 'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN',
            'CRISPEMBED_QWEN35_VISION_CASCADE_NORM')):
        raise ValueError('frozen vision selection differs')
    frozen_inputs = {row['id']: row for row in producer['inputs']}
    frozen = producer['native_cases']['f16']
    original = producer['original_cases']
    if len(producer['inputs']) != 11 or len(frozen_inputs) != 11 or set(frozen) != set(frozen_inputs):
        raise ValueError('historical 11-case producer coverage differs')
    rendered = contract.render_panel(scratch / 'synthetic', panel)
    if tuple(row['id'] for row in rendered['cases']) != CASES:
        raise ValueError('rendered case set differs')
    for spec, row in zip(panel['cases'], rendered['cases']):
        name = row['id']
        historical, ref, hf = frozen_inputs[name], frozen[name], original[name]
        if (historical['id'] != name or historical['split'] != spec['split']
                or historical['gold'] != spec['gold'] or historical['license'] != 'CC0-1.0'
                or historical['cap'] != 256 or any(row[key] != historical[key] for key in ('png_sha256', 'rgb_sha256'))
                or any(ref['input_' + key] != row[key] or hf['input_' + key] != row[key]
                       for key in ('png_sha256', 'rgb_sha256'))):
            raise ValueError('rendered bytes differ from frozen independent producer: ' + name)
        for item in (ref, hf):
            if (item['cap'] != 256 or item['stop_id'] != 248046 or item['finish_reason'] != 'stop'
                    or not item['token_ids'] or item['token_ids'][-1] != 248046
                    or not isinstance(item['raw_text'], str) or not item['raw_text']):
                raise ValueError('frozen output incomplete: ' + name)
        if any(value is not True for value in exact(ref, hf).values()):
            raise ValueError('frozen F16 and original decoded records differ: ' + name)
        REPORT['cases'][name] = {'input': {'png_sha256': row['png_sha256'], 'rgb_sha256': row['rgb_sha256'],
                                           'file': row['file'], 'license': 'CC0-1.0', 'cap': 256},
                                 'frozen_f16': {key: ref[key] for key in ('token_ids', 'raw_text', 'stop_id', 'finish_reason')},
                                 'arms': {}}
        save('render-' + name)
    REPORT['frozen_producer'] = {'receipt_sha256': SOURCE_PINS[RECEIPT], 'canonical_sha256': PRODUCER_SHA,
                                 'run': receipt['run'], 'source_commit': receipt['source_commit'],
                                 'panel_canonical_sha256': rendered['panel_canonical_sha256']}
    save('frozen-inputs-verified')
    from huggingface_hub import hf_hub_download
    REPORT['model'] = {'repo': MODEL_REPO, 'revision': MODEL_REVISION, 'filename': MODEL_NAME,
                       'expected_bytes': MODEL_BYTES, 'expected_sha256': MODEL_SHA, 'token': False,
                       'license': 'Apache-2.0', 'base_model_license': 'Apache-2.0'}
    save('download-model')
    model = Path(hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_NAME, revision=MODEL_REVISION,
                                 token=False, cache_dir=str(scratch / 'hub'))).resolve()
    if not model.is_relative_to(scratch.resolve()):
        raise ValueError('model resolved outside task scratch')
    REPORT['model'].update(actual_bytes=model.stat().st_size, actual_sha256=sha(model), path=str(model))
    save('verify-model')
    if REPORT['model']['actual_bytes'] != MODEL_BYTES or REPORT['model']['actual_sha256'] != MODEL_SHA:
        raise ValueError('downloaded GGUF bytes or sha256 differ')
    for row in rendered['cases']:
        name = row['id']
        case = REPORT['cases'][name]
        image = scratch / 'synthetic' / row['file']
        # Verify again immediately before invoking both arms on the same file.
        from PIL import Image
        with Image.open(image) as opened:
            rgb = opened.convert('RGB')
            rgb_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
        if sha(image) != row['png_sha256'] or rgb_hash != row['rgb_sha256']:
            raise ValueError('input changed before inference: ' + name)
        for arm, gate in (('off', 0), ('on', 1)):
            save('run-' + name + '-' + arm)
            result = native(binary, model, image, 256, gate, out / (name + '-' + arm + '.log'))
            case['arms'][arm] = result
            save('captured-' + name + '-' + arm)
            if 'output' in result:
                result['vs_frozen_f16'] = exact(result['output'], case['frozen_f16'])
                save('compared-' + name + '-' + arm)
        off = case['arms']['off']
        on = case['arms']['on']
        if 'output' in off and 'output' in on:
            case['on_vs_off'] = exact(on['output'], off['output'])
        save('case-' + name)
    passed = (set(REPORT['cases']) == set(CASES) and all(
        set(case['arms']) == {'off', 'on'}
        and all('error' not in case['arms'][arm] and 'output' in case['arms'][arm]
                for arm in ('off', 'on'))
        and all(value is True for value in case['arms']['off'].get('vs_frozen_f16', {}).values())
        and len(case['arms']['off'].get('vs_frozen_f16', {})) == 3
        and len(case.get('on_vs_off', {})) == 3
        and all(value is True for value in case['on_vs_off'].values())
        for case in REPORT['cases'].values()))
    REPORT['decoded_gate_passed'] = passed
    REPORT['operator_dispatch_observed'] = None
    REPORT['status'] = 'completed: bounded decoded gate passed' if passed else 'failed: decoded gate or required run'
    save('complete' if passed else 'decoded-failure')
    return 0 if passed else 1


if __name__ == '__main__':
    REPORT['started_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    try:
        code = main()
    except BaseException as exc:
        REPORT.update(status='failed', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc()[-12000:])
        code = 2
    REPORT['exit_code'] = code
    REPORT['finished_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    try:
        save(REPORT['phase'])
    except BaseException:
        print('REPORT_WRITE_FAILED', traceback.format_exc(), file=sys.stderr, flush=True)
        code = 2
    print('OVIS_DIRECT_F16_RESULT', json.dumps({'status': REPORT['status'], 'exit_code': code,
          'report': str(REPORT_PATH) if REPORT_PATH else None}), flush=True)
    raise SystemExit(code)
