#!/usr/bin/env python3
"""Hosted, diagnostic OvisOCR2 vision A/B on the existing native gates.

The source, reference, full F32/F16 models and comparator budgets are frozen.
Every arm is measured independently. An A/B pass does not qualify an alias.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[2]
SOURCE_RUN = 40041958521
SOURCE_HEAD = 'cc038b0041e6ace09cf413dc18aafccdd1b31c35'
SOURCE_REPORT_ARTIFACT = (12540483904, 'ovisocr2-model-reports-40041958521', 271125,
                          '86ea94b185eed85fd3bd09320124a0860fb985021af0cc4253e7eadae59b19a1')
SOURCE_REPORT_SHA = 'bdcfc725ccdf4a13d4f16ed3f7238d12f103ddc4d0e47a861c82122a9c7b98ca'
SOURCE_REFERENCE_ARTIFACT = (12540836880, 'ovisocr2-model-diff-reference-40041958521', 1303553123,
                             '54c823899bdde33448ff93137fbbf960981fa031877a6e29f76c91b0065d36a4')
SOURCE_REFERENCE_GGUF = ('official-delta.gguf', 1454224192,
                         'e8e20c654645092ef11e8632db6d84393761df3f8b1d34bc48ae1ddac668e588')
SOURCE_REFERENCE_JSON = ('ovisocr2-original.json', 86906,
                         '7534b151236cb61b943d2ad1a769fb15a65f0e83f48034d483e423b81d18a658')
EXPORT_RUN = 39992662147
EXPORT_HEAD = '5e197baee331d9b5745c6d906c21893e8211a3fd'
EXPORT_ARTIFACT = (12519953155, 'ovisocr2-experimental-gguf-39992662147', 2832545300,
                   '1bc1b6556e65fc40a497110e5229ad79402d9358aa9640aaf7fda863bc8a5c3f')
MODELS = {
    'f32': ('ovisocr2-f32.gguf', 3421945280,
            '7f04dbc597e122fc3f1db2d96a64e8d7759be9f62dca4bc4a472fe5f0db9bfd9'),
    'f16': ('ovisocr2-f16.gguf', 1717222848,
            'd3a733a5d5ba5d63f5275500fad4cff585b82e2fb90d1def2fe767d04edf57db'),
}
ARM_GATES = {
    'baseline': (),
    'precise_norm': ('CRISPEMBED_QWEN35_VISION_PRECISE_NORM',),
    'precise_attn': ('CRISPEMBED_QWEN35_VISION_PRECISE_ATTN',),
    'both': ('CRISPEMBED_QWEN35_VISION_PRECISE_NORM', 'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'),
    'cascade': ('CRISPEMBED_QWEN35_VISION_CASCADE_NORM',),
    'cascade_precise_attn': ('CRISPEMBED_QWEN35_VISION_CASCADE_NORM',
                            'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN'),
}
BASE_ENV = {name: '1' for name in ('CRISPEMBED_QWEN35_OCR', 'CRISPEMBED_OVISOCR2_FRONTEND',
                                   'CRISPEMBED_OVISOCR2_INFERENCE', 'CRISPEMBED_QWEN35_AUDIT_ALL_LAYERS')}
BASE_ENV.update({name: '0' for name in ('CRISPEMBED_QWEN35_PRECISE_RMS',
    'CRISPEMBED_QWEN35_VISION_PRECISE_NORM', 'CRISPEMBED_QWEN35_VISION_PRECISE_ATTN',
    'CRISPEMBED_QWEN35_VISION_CASCADE_NORM', 'CRISPEMBED_QWEN35_IMATRIX',
    'CRISPEMBED_QWEN35_PARALLEL_F16', 'CRISPEMBED_QWEN35_PROFILE',
    'CRISPEMBED_QWEN35_QUANT', 'CRISPEMBED_QWEN35_TOKENIZER_MARKS')})
SOURCE_CODE = ('src/qwen35_vision.cpp', 'src/crispembed_diff.h',
               'tests/qwen35_parity.h')
SOURCE_TEST_SHA = '33ff0ad62c83a56af7bff466a79408a83437eda13cec5a76e56bbfbda7ab9f74'
TRACE_TEST_COMMIT = 'a566399ac4c13705f19626bd05da12e396aed6c3'
TRACE_TEST_SHA = '41f74671971086d0fc245c413dd3d0997697b37e59071a0221243d389561b1c5'
VISION_PROBES = ('norm1', 'qkv', 'pooled', 'attention', 'norm2', 'up', 'activation', 'down')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(out, report):
    (out / 'ovisocr2-vision-audit.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def api_json(path, call=subprocess.check_output):
    return json.loads(call(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/' + path], timeout=60))


def service_source(run_id, head, artifact_pin, *, conclusion):
    run = api_json('actions/runs/' + str(run_id))
    ident, name, size, digest = artifact_pin
    artifact = api_json('actions/artifacts/' + str(ident))
    if (run.get('id') != run_id or run.get('status') != 'completed'
            or run.get('conclusion') != conclusion or run.get('head_sha') != head
            or run.get('path') != '.github/workflows/ovisocr2-metadata.yml'
            or artifact.get('id') != ident or artifact.get('name') != name
            or artifact.get('expired') is not False or artifact.get('size_in_bytes') != size
            or artifact.get('digest') != 'sha256:' + digest
            or artifact.get('workflow_run', {}).get('id') != run_id
            or artifact.get('workflow_run', {}).get('head_sha') != head):
        raise ValueError('Pinned source run/artifact service proof differs: ' + name)
    return {'run': run_id, 'head': head, 'conclusion': conclusion, 'artifact_id': ident,
            'artifact_name': name, 'artifact_bytes': size, 'artifact_zip_sha256': digest}


def download_artifact(pin, destination, *, timeout=1800):
    ident, name, expected_bytes, expected_sha = pin
    start = time.monotonic()
    print('START artifact', name, flush=True)
    with Path(destination).open('wb') as output:
        process = subprocess.Popen(['gh', 'api',
            'repos/CrispStrobe/CrispEmbed/actions/artifacts/' + str(ident) + '/zip'],
            stdout=output, stderr=subprocess.PIPE, start_new_session=True)
        while process.poll() is None:
            if time.monotonic() - start >= timeout:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise TimeoutError('Artifact download timed out: ' + name)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print('HEARTBEAT artifact', name, Path(destination).stat().st_size, 'bytes', flush=True)
        error = process.stderr.read().decode(errors='replace')[:1000]
    if process.returncode:
        raise RuntimeError('Artifact download failed: ' + name + ': ' + error)
    if Path(destination).stat().st_size != expected_bytes or sha(destination) != expected_sha:
        raise ValueError('Downloaded artifact ZIP size/SHA differs: ' + name)
    print('END artifact', name, expected_bytes, 'bytes', flush=True)


def zip_members(archive):
    names = [entry.filename for entry in archive.infolist()]
    if (len(names) != len(set(names)) or any(Path(name).name != name or name.startswith('.')
                                             or name.endswith('/') for name in names)):
        raise ValueError('Artifact ZIP contains duplicate or unsafe paths')
    return set(names)


def extract_member(archive, name, destination, expected_bytes, expected_sha):
    if name not in zip_members(archive):
        raise ValueError('Pinned artifact member missing: ' + name)
    with archive.open(name) as source, Path(destination).open('wb') as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    if Path(destination).stat().st_size != expected_bytes or sha(destination) != expected_sha:
        raise ValueError('Extracted artifact member size/SHA differs: ' + name)


def verify_source_report(data, files):
    """The failed model run is a source of frozen inputs, never a parity pass."""
    if (data.get('source_commit') != SOURCE_HEAD or data.get('status') != 'failed hosted model audit'
            or data.get('failures') != ['f32-vision', 'f16-vision']
            or data.get('prior_reference_comparison', {}).get('passed') is not True
            or data.get('prior_reference_comparison', {}).get('unchanged_prior_members') != 871
            or data.get('prior_reference_comparison', {}).get('added_cache_members') != 72
            or data.get('prior_reference_comparison', {}).get('fresh_stage_count') != 943
            or data.get('reference_artifacts', {}).get('gguf', {}).get('sha256') != SOURCE_REFERENCE_GGUF[2]
            or data.get('reference_artifacts', {}).get('json', {}).get('sha256') != SOURCE_REFERENCE_JSON[2]
            or data.get('native_environment') != BASE_ENV):
        raise ValueError('Pinned failed source report status/reference/gates differ')
    commands = data.get('commands', {})
    expected_arms = {f'{precision}-{suffix}' for precision in MODELS for suffix in
        ('delta', 'vision', 'decoder-prepared', 'decoder-native-vision',
         'decoder-native-positions', 'decoder-native-frontend', 'context')}
    failed = {'f32-vision', 'f16-vision'}
    if (not expected_arms.issubset(commands)
            or any((commands[name].get('exit_code') == 0) != (name not in failed)
                   or commands[name].get('passed') is not (name not in failed)
                   for name in expected_arms)):
        raise ValueError('Pinned failed source did not have exactly two vision failures')
    for precision, (_, count, digest) in MODELS.items():
        source = data.get('exports', {}).get(precision, {})
        if source.get('model_sha256') != digest or source.get('bytes') != count:
            raise ValueError('Pinned source export differs: ' + precision)
    for name in SOURCE_CODE:
        expected = data.get('native_source_sha256', {}).get(name)
        if expected != sha(ROOT / name):
            raise ValueError('Vision/comparator source differs from pinned failed run: ' + name)
    if (data.get('native_source_sha256', {}).get('tests/test_qwen35_vision.cpp') != SOURCE_TEST_SHA
            or sha(ROOT / 'tests/test_qwen35_vision.cpp') != TRACE_TEST_SHA):
        raise ValueError('Vision test-only dump-preservation source differs')
    for precision in MODELS:
        label = precision + '-vision'
        if files[label + '.log'] != commands[label]['sha256']:
            raise ValueError('Frozen vision failure log SHA differs: ' + label)
        stages = parse_vision_log((ROOT / 'unused'), text=files[label + '.text'])
        if stages['first_failing_operation'] != ('prefill.vision.probe.11.up' if precision == 'f32'
                                                 else 'prefill.vision.probe.11.norm2'):
            raise ValueError('Frozen source earliest vision operation differs: ' + label)
    return {'source_commit': SOURCE_HEAD, 'failed_only_vision': True,
            'old_npy_exact': 871, 'new_cache_states': 72, 'native_other_arms_passed': 12,
            'original_vision_test_sha256': SOURCE_TEST_SHA,
            'trace_vision_test_commit': TRACE_TEST_COMMIT,
            'trace_vision_test_sha256': TRACE_TEST_SHA,
            'vision_and_comparator_implementation_unchanged': True}


def operation_key(name):
    if name in ('prefill.vision.patch_embed', 'prefill.vision.position', 'prefill.vision.embedded',
                'prefill.vision.rotary_cos', 'prefill.vision.rotary_sin'):
        return (0, ('prefill.vision.patch_embed', 'prefill.vision.position',
                    'prefill.vision.embedded', 'prefill.vision.rotary_cos',
                    'prefill.vision.rotary_sin').index(name))
    match = re.fullmatch(r'prefill\.vision\.probe\.(5|11)\.([a-z0-9_]+)', name)
    if match and match[2] in VISION_PROBES:
        return (1, int(match[1]), VISION_PROBES.index(match[2]))
    match = re.fullmatch(r'prefill\.vision\.layer\.(\d+)', name)
    if match:
        return (1, int(match[1]), len(VISION_PROBES))
    if name.startswith('prefill.vision.merger'):
        return (2, ('prefill.vision.merger.norm', 'prefill.vision.merger.up',
                    'prefill.vision.merger.activation', 'prefill.vision.merger').index(name))
    raise ValueError('Unexpected vision metric stage: ' + name)


METRIC = re.compile(r'^(PASS|FAIL) (prefill\.vision\.[a-z0-9_.]+): cos_min=([\deE+.-]+) '
                    r'cos_global=([\deE+.-]+) max_abs=([\deE+.-]+) mean_abs=([\deE+.-]+) '
                    r'norms=([\deE+.-]+)/([\deE+.-]+)$')
BUDGET = re.compile(r'^BUDGET (prefill\.vision\.[a-z0-9_.]+): strict_absolute_pass=([01]) '
                    r'calibrated=([01]) self_max_abs=([\deE+.-]+) max_ratio=([\deE+.-]+)$')


def parse_vision_log(path, *, text=None, require_dump_integrity=False):
    text = Path(path).read_text(errors='replace') if text is None else text
    metrics, budgets = {}, {}
    for line in text.splitlines():
        if match := METRIC.fullmatch(line):
            _, name, *values = match.groups()
            if name in metrics:
                raise ValueError('Repeated vision metric: ' + name)
            numbers = [float(value) for value in values]
            if (any(not math.isfinite(value) for value in numbers)
                    or any(value < 0 for value in numbers[2:])
                    or any(abs(value) > 1.00001 for value in numbers[:2])):
                raise ValueError('Nonfinite/negative vision metric: ' + name)
            metrics[name] = {'passed': match[1] == 'PASS', 'cos_min': numbers[0],
                             'cos_global': numbers[1], 'max_abs': numbers[2],
                             'mean_abs': numbers[3], 'native_norm': numbers[4],
                             'reference_norm': numbers[5]}
        elif match := BUDGET.fullmatch(line):
            name = match[1]
            if name in budgets:
                raise ValueError('Repeated vision budget: ' + name)
            ratio, self_abs = float(match[5]), float(match[4])
            if not math.isfinite(ratio) or not math.isfinite(self_abs) or ratio < 0 or self_abs < 0:
                raise ValueError('Nonfinite/negative vision budget: ' + name)
            budgets[name] = {'strict_absolute_pass': match[2] == '1',
                             'calibrated': match[3] == '1',
                             'self_max_abs': self_abs, 'max_ratio': ratio}
    if not metrics or set(metrics) != set(budgets):
        raise ValueError('Vision metrics/budgets incomplete')
    for name in metrics:
        operation_key(name)
        metrics[name].update(budgets[name])
    ordered = sorted(metrics, key=operation_key)
    failures = [name for name in ordered if not metrics[name]['passed']]
    required = {f'prefill.vision.layer.{index}' for index in range(12)}
    required |= {f'prefill.vision.probe.{index}.{field}' for index in (5, 11) for field in VISION_PROBES}
    required |= {'prefill.vision.patch_embed', 'prefill.vision.position', 'prefill.vision.embedded',
                 'prefill.vision.rotary_cos', 'prefill.vision.rotary_sin',
                 'prefill.vision.merger.norm', 'prefill.vision.merger.up',
                 'prefill.vision.merger.activation', 'prefill.vision.merger'}
    if set(metrics) != required:
        raise ValueError('Vision stage metric coverage incomplete')
    marker = 'PASS VISION_RAW_DUMP_INTEGRITY stages=2'
    if require_dump_integrity and text.splitlines().count(marker) != 1:
        raise ValueError('Actual native vision raw-dump integrity marker absent')
    return {'stage_count': len(metrics), 'first_failing_operation': failures[0] if failures else None,
            'failing_operations': failures, 'stages': metrics,
            'merger': metrics['prefill.vision.merger'],
            'actual_dump_integrity_passed': text.splitlines().count(marker) == 1}

def vector_manifest(folder, source_shapes, parsed, label):
    """Hash every actual native vector; the test preserves two sensitive dumps."""
    if parsed['actual_dump_integrity_passed'] is not True:
        raise ValueError('Vision raw-vector preservation was not proved: ' + label)
    expected = set(parsed['stages'])
    paths = {path.name.removesuffix('.f32'): path for path in Path(folder).glob('*.f32')}
    if set(paths) != expected:
        raise ValueError('Vision raw-vector coverage differs: ' + label)
    records = []
    for stage in sorted(expected, key=operation_key):
        shape = source_shapes.get(stage)
        if not isinstance(shape, list) or not shape or any(type(x) is not int or x < 1 for x in shape):
            raise ValueError('Original vision stage shape missing: ' + stage)
        path = paths[stage]
        expected_bytes = math.prod(shape) * 4
        if path.stat().st_size != expected_bytes:
            raise ValueError('Vision raw-vector F32 byte size differs: ' + stage)
        records.append({'stage': stage, 'file': 'vision-vectors/' + label + '/' + path.name,
                        'bytes': expected_bytes, 'sha256': sha(path), 'original_shape': shape,
                        'provenance': 'actual_primary_comparison'})
    return {'arm': label, 'all_metric_files_present': True,
            'actual_primary_vector_count': len(records),
            'test_byte_exact_integrity_marker': True, 'files': records,
            'scope': 'All raw native F32 comparison vectors retained remotely; test confirms two sensitive dumps byte-exact after negative controls'}


def process_command(label, args, out, report, *, env=None, timeout=1800):
    path = out / (label + '.log')
    start = time.monotonic()
    timed_out = False
    print('START', label, flush=True)
    with path.open('w') as log:
        child = subprocess.Popen([str(arg) for arg in args], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while child.poll() is None:
            remaining = timeout - (time.monotonic() - start)
            if remaining <= 0:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                break
            try:
                child.wait(timeout=min(30, remaining))
            except subprocess.TimeoutExpired:
                print('HEARTBEAT', label, int(time.monotonic() - start), 'seconds', flush=True)
    entry = {'argv': [str(arg) for arg in args], 'exit_code': child.returncode,
             'timed_out': timed_out, 'wall_seconds': time.monotonic() - start,
             'log': path.name, 'log_sha256': sha(path)}
    report.setdefault('commands', {})[label] = entry
    save(out, report)
    print('END', label, 'exit', child.returncode, flush=True)
    return entry


def preflight(out, scratch, report):
    report['phase'] = 'source-service'; save(out, report)
    report['source_services'] = {
        'failed_model_reports': service_source(SOURCE_RUN, SOURCE_HEAD, SOURCE_REPORT_ARTIFACT,
                                               conclusion='failure'),
        'failed_model_reference': service_source(SOURCE_RUN, SOURCE_HEAD, SOURCE_REFERENCE_ARTIFACT,
                                                 conclusion='failure'),
        'qualified_export': service_source(EXPORT_RUN, EXPORT_HEAD, EXPORT_ARTIFACT,
                                           conclusion='success')}
    report['phase'] = 'small-source-report'; save(out, report)
    archive_path = scratch / 'source-reports.zip'
    download_artifact(SOURCE_REPORT_ARTIFACT, archive_path, timeout=300)
    with zipfile.ZipFile(archive_path) as archive:
        names = zip_members(archive)
        required = {'ovisocr2-model-audit.json', 'f32-vision.log', 'f16-vision.log'}
        if not required.issubset(names):
            raise ValueError('Pinned small source report members missing')
        source_bytes = archive.read('ovisocr2-model-audit.json')
        if hashlib.sha256(source_bytes).hexdigest() != SOURCE_REPORT_SHA:
            raise ValueError('Pinned failed model producer report bytes differ')
        source = json.loads(source_bytes)
        log_files = {}
        for label in ('f32-vision', 'f16-vision'):
            content = archive.read(label + '.log')
            (out / ('source-' + label + '.log')).write_bytes(content)
            log_files[label + '.log'] = hashlib.sha256(content).hexdigest()
            log_files[label + '.text'] = content.decode(errors='replace')
        report['source_contract'] = verify_source_report(source, log_files)
        (out / 'source-ovisocr2-model-audit.json').write_bytes(source_bytes)
    archive_path.unlink()  # Verified task-owned small transport only.
    report['source_report_sha256'] = SOURCE_REPORT_SHA
    report['source_logs_sha256'] = {name: value for name, value in log_files.items() if name.endswith('.log')}
    report['phase'] = 'preflight-complete'; save(out, report)
    return source


def checked_archive(pin, scratch, out, label, report):
    path = scratch / (label + '.zip')
    download_artifact(pin, path)
    with zipfile.ZipFile(path) as archive:
        names = zip_members(archive)
        if label == 'reference':
            member_specs = (SOURCE_REFERENCE_GGUF, SOURCE_REFERENCE_JSON)
        elif label == 'export':
            member_specs = tuple(MODELS.values())
        else:
            raise ValueError('Unknown pinned archive')
        if not {spec[0] for spec in member_specs}.issubset(names):
            raise ValueError('Pinned required archive members missing: ' + label)
        extracted = {}
        for name, size, digest in member_specs:
            target = scratch / name
            extract_member(archive, name, target, size, digest)
            extracted[name] = {'bytes': size, 'sha256': digest}
    path.unlink()  # Full ZIP and selected payloads verified; exact artifact retained remotely.
    report.setdefault('extracted', {})[label] = {'artifact_id': pin[0],
        'zip_sha256': pin[3], 'verified_zip_removed': True, 'files': extracted}
    save(out, report)


def main(preflight_only=False):
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'ovis-vision-ab'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report = {'phase': 'preflight', 'status': 'started', 'source_commit': os.environ.get('GITHUB_SHA'),
              'driver_sha256': sha(__file__), 'commands': {}, 'arms': {},
              'scope': 'Diagnostic 6-gate A/B for F32 and F16 native vision only; no comparator, tolerance, model or default changes',
              'qualification': {'native_alias': False, 'decoded_quality': False,
                                'reason': 'Vision component A/B is diagnostic; existing model audit still has two failures'}}
    save(out, report)
    try:
        source = preflight(out, scratch, report)
        if preflight_only:
            report['status'] = 'preflight passed; no weights, model, or build loaded'
            save(out, report)
            return report
        report['phase'] = 'large-artifacts'; save(out, report)
        report['disk_free_before_large_bytes'] = shutil.disk_usage(scratch).free
        save(out, report)
        if report['disk_free_before_large_bytes'] < 9 * 1024 ** 3:
            raise ValueError('Hosted scratch has under 9 GiB free before bounded artifacts')
        # Extract the two models and remove their verified ZIP first. Keeping
        # the reference file during that ZIP's extraction adds ~1.45 GB peak.
        checked_archive(EXPORT_ARTIFACT, scratch, out, 'export', report)
        checked_archive(SOURCE_REFERENCE_ARTIFACT, scratch, out, 'reference', report)
        report['disk_free_after_extraction_bytes'] = shutil.disk_usage(scratch).free
        save(out, report)
        report['phase'] = 'portable-native-build'; save(out, report)
        build = scratch / 'build'
        configure = ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                     '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF',
                     '-DGGML_METAL=OFF', '-DGGML_BLAS=OFF', '-DCRISPEMBED_BUILD_SHARED=OFF']
        if process_command('cmake-configure', configure, out, report, timeout=600)['exit_code'] != 0:
            raise ValueError('Portable CMake configure failed')
        if process_command('cpu-baseline-check', [sys.executable, ROOT / 'scripts/check-cpu-baseline.py', build],
                           out, report, timeout=120)['exit_code'] != 0:
            raise ValueError('Portable CPU baseline checker failed')
        if process_command('vision-target-build', ['cmake', '--build', build, '--config', 'Release',
                                                  '--target', 'test-qwen35-vision', '-j2'],
                           out, report, timeout=1800)['exit_code'] != 0:
            raise ValueError('Vision-only native target build failed')
        binary = build / 'test-qwen35-vision'
        report['native_binary_sha256'] = sha(binary)
        save(out, report)
        source_shapes = json.loads((scratch / SOURCE_REFERENCE_JSON[0]).read_text())['stages']
        report['phase'] = 'vision-ab'; save(out, report)
        diagnostic_errors = []
        for precision in MODELS:
            model = scratch / MODELS[precision][0]
            for arm, enabled in ARM_GATES.items():
                label = precision + '-' + arm
                env = dict(os.environ, **BASE_ENV)
                env.pop('CRISPEMBED_DIFF_DUMP_DIR', None)
                for name in enabled:
                    env[name] = '1'
                vectors = scratch / 'vision-vectors' / label
                vectors.mkdir(parents=True, exist_ok=True)
                env['CRISPEMBED_DIFF_DUMP_DIR'] = str(vectors)
                entry = process_command(label, [binary, model, scratch / SOURCE_REFERENCE_GGUF[0]],
                                        out, report, env=env, timeout=1800)
                arm_result = {'precision': precision, 'enabled_gates': list(enabled),
                              'all_other_gates': BASE_ENV, 'exit_code': entry['exit_code'],
                              'timed_out': entry['timed_out'], 'log_sha256': entry['log_sha256']}
                try:
                    parsed = parse_vision_log(out / entry['log'], require_dump_integrity=True)
                    manifest = vector_manifest(vectors, source_shapes, parsed, label)
                    manifest_path = out / (label + '-raw-vector-manifest.json')
                    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
                    arm_result.update(metrics=parsed, raw_vector_manifest=manifest_path.name,
                                      raw_vector_manifest_sha256=sha(manifest_path),
                                      raw_vector_actual_count=manifest['actual_primary_vector_count'])
                    if entry['exit_code'] not in (0, 1) or entry['timed_out']:
                        raise ValueError('Vision test execution did not finish normally')
                except (OSError, ValueError) as error:
                    arm_result['diagnostic_error'] = type(error).__name__ + ': ' + str(error)
                    diagnostic_errors.append(label)
                report['arms'][label] = arm_result
                save(out, report)
            if precision == 'f32' and len([key for key in report['arms'] if key.startswith('f32-')]) == 6:
                # The pinned source export remains durable remotely. Remove
                # only this verified task-owned F32 reconstruction after use.
                if sha(model) != MODELS['f32'][2]:
                    raise ValueError('F32 model changed before task-owned cleanup')
                model.unlink()
                report['f32_local_cleanup'] = {'removed_after_six_arms': True,
                    'model_sha256': MODELS['f32'][2], 'source_export_run': EXPORT_RUN}
                save(out, report)
        if len(report['arms']) != 12:
            raise ValueError('Vision A/B arm coverage incomplete')
        report['diagnostic_errors'] = diagnostic_errors
        if diagnostic_errors:
            raise ValueError('Vision A/B diagnostic arm integrity failed: ' + ', '.join(diagnostic_errors))
        report['all_arms_pass_existing_comparator'] = all(row['exit_code'] == 0 and
            row['metrics']['first_failing_operation'] is None for row in report['arms'].values())
        report['passing_arms'] = [label for label, row in report['arms'].items()
            if row['exit_code'] == 0 and row['metrics']['first_failing_operation'] is None]
        report['status'] = 'completed diagnostic vision A/B; no qualification'
        report['phase'] = 'complete'; save(out, report)
        return report
    except BaseException as error:
        report['status'] = 'failed diagnostic vision A/B infrastructure'
        report['error_type'] = type(error).__name__
        report['error'] = str(error)
        report['error_phase'] = report['phase']
        save(out, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    main(preflight_only=args.preflight)
