#!/usr/bin/env python3
"""Hosted fallback: replay frozen Ovis block-11 norm2/up in all A/B arms.

This is an operator diagnosis, never a model or decoded-quality qualification.
The A/B service and artifact pins stay unset until its completed run is reviewed.
"""

import argparse
import hashlib
import importlib.util
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
AB_HELPER = ROOT / 'tools/ci-heavy/ovisocr2_vision_audit.py'
AB_HELPER_SHA256 = 'a812910212ac030e13aa2e77bbab26460b6375f8978a802bf44c829c95f0a9c3'
TARGET_SOURCE_SHA256 = '3af6a2fd37e5fbc089056fae527f12c211f420823b3ce5959e23478216085b15'
OPERATOR_HEADER_SHA256 = '8c90bbd42464fb7998e4ffec251a6dc7d50839e8e05a99437a82971e3ace0723'
AB_RUN = None
AB_HEAD = None
AB_REPORT_ARTIFACT = None  # (id, name, ZIP bytes, ZIP SHA256)
AB_VECTOR_ARTIFACT = None  # (id, name, ZIP bytes, ZIP SHA256)
AB_REPORT_SHA256 = None  # ovisocr2-vision-audit.json member SHA256
AB_VECTOR_ARCHIVE_ROOT = None  # Exact ZIP prefix, e.g. '' or 'vision-vectors/'
SELECTED = ('prefill.vision.layer.10', 'prefill.vision.probe.11.attention',
            'prefill.vision.probe.11.norm2', 'prefill.vision.probe.11.up')
COMPLETE = re.compile(r'^LOCAL_OP_DIAGNOSTIC_COMPLETE complete=1 qualification_claim=0 '
                      r'precise_norm=(0|1) cascade_norm=(0|1) precise_attn=(0|1)$', re.M)
METRIC = re.compile(r'^LOCAL_OP_METRIC (native_input|same_original_input) '
                    r'(prefill\.vision\.probe\.11\.(?:norm2|up)) n=(\d+) '
                    r'max_abs=([\deE+.-]+) mean_abs=([\deE+.-]+) cos_min=([\deE+.-]+) '
                    r'worst_index=(\d+) row=(\d+) candidate=([\deE+.-]+) '
                    r'original=([\deE+.-]+) existing_budget_pass=([01])$', re.M)
BUDGET = re.compile(r'^BUDGET (prefill\.vision\.probe\.11\.(?:norm2|up)): '
                    r'strict_absolute_pass=([01]) calibrated=([01]) '
                    r'self_max_abs=([\deE+.-]+) max_ratio=([\deE+.-]+)$', re.M)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(out, report):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ovisocr2-vision-local-ops.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def require_pins():
    if (AB_RUN is None or AB_HEAD is None or AB_REPORT_ARTIFACT is None
            or AB_VECTOR_ARTIFACT is None or AB_REPORT_SHA256 is None
            or AB_VECTOR_ARCHIVE_ROOT is None):
        raise ValueError('Reviewed A/B run, report, vectors, and ZIP root pins are unset')
    if (type(AB_RUN) is not int or AB_RUN < 1 or not re.fullmatch(r'[0-9a-f]{40}', AB_HEAD)
            or not re.fullmatch(r'[0-9a-f]{64}', AB_REPORT_SHA256)
            or AB_VECTOR_ARCHIVE_ROOT not in ('', 'vision-vectors/')):
        raise ValueError('A/B source pin format differs')
    for pin in (AB_REPORT_ARTIFACT, AB_VECTOR_ARTIFACT):
        if (not isinstance(pin, tuple) or len(pin) != 4 or type(pin[0]) is not int or pin[0] < 1
                or not isinstance(pin[1], str) or not pin[1].startswith('ovisocr2-vision-ab-')
                or type(pin[2]) is not int or pin[2] < 1
                or not isinstance(pin[3], str) or not re.fullmatch(r'[0-9a-f]{64}', pin[3])):
            raise ValueError('A/B artifact pin format differs')
    if (AB_REPORT_ARTIFACT[1] != f'ovisocr2-vision-ab-reports-{AB_RUN}'
            or AB_VECTOR_ARTIFACT[1] != f'ovisocr2-vision-ab-vectors-{AB_RUN}'):
        raise ValueError('A/B artifact run/name pairing differs')


def load_ab_helper():
    if sha(AB_HELPER) != AB_HELPER_SHA256:
        raise ValueError('Pinned A/B source helper changed')
    if (sha(ROOT / 'tests/test_ovisocr2_vision_local_ops.cpp') != TARGET_SOURCE_SHA256
            or sha(ROOT / 'tests/qwen35_vision_operators_impl.h') != OPERATOR_HEADER_SHA256):
        raise ValueError('Reviewed local operator test source changed')
    spec = importlib.util.spec_from_file_location('frozen_ovis_vision_ab', AB_HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def api_json(path, call=subprocess.check_output):
    return json.loads(call(['gh', 'api', 'repos/CrispStrobe/CrispEmbed/' + path], timeout=60))


def service_ab(pin):
    run = api_json('actions/runs/' + str(AB_RUN))
    ident, name, size, digest = pin
    artifact = api_json('actions/artifacts/' + str(ident))
    if (run.get('id') != AB_RUN or run.get('status') != 'completed' or run.get('conclusion') != 'success'
            or run.get('head_sha') != AB_HEAD
            or run.get('path') != '.github/workflows/ovisocr2-vision-audit.yml'
            or artifact.get('id') != ident or artifact.get('name') != name
            or artifact.get('expired') is not False or artifact.get('size_in_bytes') != size
            or artifact.get('digest') != 'sha256:' + digest
            or artifact.get('workflow_run', {}).get('id') != AB_RUN
            or artifact.get('workflow_run', {}).get('head_sha') != AB_HEAD):
        raise ValueError('A/B service run/artifact proof differs: ' + name)
    return {'run': AB_RUN, 'head': AB_HEAD, 'artifact_id': ident,
            'artifact_name': name, 'artifact_bytes': size, 'artifact_zip_sha256': digest}


def flat_members(archive):
    infos = archive.infolist()
    names = [entry.filename for entry in infos]
    if (len(names) != len(set(names)) or any('/' in name or '\\' in name or name.startswith('.')
                                             or name.endswith('/') for name in names)):
        raise ValueError('A/B report ZIP contains unsafe or duplicate members')
    return set(names)


def required_labels(helper):
    return [precision + '-' + arm for precision in helper.MODELS for arm in helper.ARM_GATES]


def validate_ab_outcome(report, labels):
    passing = []
    for label in labels:
        arm = report['arms'][label]
        exit_code = arm.get('exit_code')
        first = arm.get('metrics', {}).get('first_failing_operation')
        if (type(exit_code) is not int or exit_code not in (0, 1)
                or arm.get('timed_out') is not False or (exit_code == 0) != (first is None)):
            raise ValueError('A/B arm does not have a complete comparator outcome: ' + label)
        if exit_code == 0:
            passing.append(label)
    if (report.get('passing_arms') != passing
            or report.get('all_arms_pass_existing_comparator') is not (len(passing) == len(labels))):
        raise ValueError('A/B reported passing arms differ from comparator outcomes')
    f32 = {label.removeprefix('f32-') for label in passing if label.startswith('f32-')}
    f16 = {label.removeprefix('f16-') for label in passing if label.startswith('f16-')}
    if f32 & f16:
        raise ValueError('A/B has a common passing F32/F16 preset; full replay is preferred')
    return passing


def check_ab_report(report, manifests, logs, helper):
    labels = required_labels(helper)
    if (report.get('source_commit') != AB_HEAD or report.get('driver_sha256') != AB_HELPER_SHA256
            or report.get('status') != 'completed diagnostic vision A/B; no qualification'
            or report.get('phase') != 'complete' or report.get('diagnostic_errors') != []
            or set(report.get('arms', {})) != set(labels)):
        raise ValueError('A/B result is incomplete or not frozen')
    validate_ab_outcome(report, labels)
    contract = report.get('source_contract', {})
    if (contract.get('source_commit') != helper.SOURCE_HEAD or contract.get('old_npy_exact') != 871
            or contract.get('new_cache_states') != 72 or contract.get('native_other_arms_passed') != 12):
        raise ValueError('A/B source does not attest frozen 943-stage original')
    services = report.get('source_services', {})
    for name, pin in (('failed_model_reports', helper.SOURCE_REPORT_ARTIFACT),
                      ('failed_model_reference', helper.SOURCE_REFERENCE_ARTIFACT),
                      ('qualified_export', helper.EXPORT_ARTIFACT)):
        item = services.get(name, {})
        if item.get('artifact_id') != pin[0] or item.get('artifact_zip_sha256') != pin[3]:
            raise ValueError('A/B frozen service source differs: ' + name)
    if (report.get('extracted', {}).get('reference', {}).get('files', {}).get(
            helper.SOURCE_REFERENCE_GGUF[0], {}).get('sha256') != helper.SOURCE_REFERENCE_GGUF[2]
            or report.get('extracted', {}).get('export', {}).get('files', {}).get(
                helper.MODELS['f32'][0], {}).get('sha256') != helper.MODELS['f32'][2]
            or report.get('extracted', {}).get('export', {}).get('files', {}).get(
                helper.MODELS['f16'][0], {}).get('sha256') != helper.MODELS['f16'][2]):
        raise ValueError('A/B extracted reference/export differs')
    records = {}
    for label in labels:
        arm = report['arms'][label]
        precision, suffix = label.split('-', 1)
        command = report.get('commands', {}).get(label, {})
        manifest = manifests[label]
        log = logs[label]
        if (arm.get('precision') != precision or arm.get('enabled_gates') != list(helper.ARM_GATES[suffix])
                or arm.get('all_other_gates') != helper.BASE_ENV
                or arm.get('timed_out') is not False
                or arm.get('log_sha256') != hashlib.sha256(log).hexdigest()
                or command.get('log_sha256') != arm['log_sha256']
                or command.get('exit_code') != arm['exit_code']
                or command.get('timed_out') is not False
                or arm.get('raw_vector_manifest_sha256') != hashlib.sha256(
                    json.dumps(manifest, indent=2, allow_nan=False).encode() + b'\n').hexdigest()
                or arm.get('raw_vector_actual_count') != 37
                or arm.get('metrics', {}).get('stage_count') != 37):
            raise ValueError('A/B arm/command/manifest differs: ' + label)
        parsed = helper.parse_vision_log(ROOT / 'unused', text=log.decode(errors='replace'),
                                         require_dump_integrity=True)
        if parsed != arm['metrics']:
            raise ValueError('A/B metric transcript differs: ' + label)
        if (manifest.get('arm') != label or manifest.get('actual_primary_vector_count') != 37
                or manifest.get('all_metric_files_present') is not True
                or manifest.get('test_byte_exact_integrity_marker') is not True
                or len(manifest.get('files', [])) != 37):
            raise ValueError('A/B raw-vector manifest coverage differs: ' + label)
        files = {}
        for item in manifest['files']:
            stage = item.get('stage')
            shape = item.get('original_shape')
            if (stage in files or stage not in parsed['stages']
                    or item.get('file') != f'vision-vectors/{label}/{stage}.f32'
                    or type(item.get('bytes')) is not int or item['bytes'] < 4
                    or not isinstance(shape, list) or not shape
                    or any(type(axis) is not int or axis < 1 for axis in shape)
                    or item['bytes'] != math.prod(shape) * 4
                    or not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', ''))
                    or item.get('provenance') != 'actual_primary_comparison'):
                raise ValueError('A/B raw-vector member differs: ' + label)
            files[stage] = item
        if set(files) != set(parsed['stages']):
            raise ValueError('A/B raw-vector stage names differ: ' + label)
        records[label] = files
    return records


def read_ab_reports(path, out, helper):
    with zipfile.ZipFile(path) as archive:
        members = flat_members(archive)
        labels = required_labels(helper)
        required = {'ovisocr2-vision-audit.json'} | {label + '.log' for label in labels} | {
            label + '-raw-vector-manifest.json' for label in labels}
        if not required.issubset(members):
            raise ValueError('A/B report artifact is missing result/log/manifest')
        raw = archive.read('ovisocr2-vision-audit.json')
        if hashlib.sha256(raw).hexdigest() != AB_REPORT_SHA256:
            raise ValueError('Pinned A/B report member SHA differs')
        report = json.loads(raw)
        manifests = {}
        logs = {}
        for label in labels:
            name = label + '-raw-vector-manifest.json'
            raw_manifest = archive.read(name)
            if hashlib.sha256(raw_manifest).hexdigest() != report.get('arms', {}).get(label, {}).get(
                    'raw_vector_manifest_sha256'):
                raise ValueError('A/B raw-vector manifest SHA differs: ' + label)
            manifests[label] = json.loads(raw_manifest)
            logs[label] = archive.read(label + '.log')
        records = check_ab_report(report, manifests, logs, helper)
        (out / 'source-ovisocr2-vision-audit.json').write_bytes(raw)
        for label in labels:
            (out / ('source-' + label + '-raw-vector-manifest.json')).write_text(
                json.dumps(manifests[label], indent=2, allow_nan=False) + '\n')
    return report, records


def vector_zip_members(archive):
    infos = archive.infolist()
    names = [entry.filename for entry in infos]
    if len(names) != len(set(names)):
        raise ValueError('A/B vector ZIP has duplicate members')
    for info in infos:
        name = info.filename
        parts = name.split('/')
        if (name.startswith('/') or '\\' in name or name.endswith('/') or len(parts) < 2
                or any(part in ('', '.', '..') for part in parts)
                or (info.external_attr >> 16) & 0o170000 == 0o120000):
            raise ValueError('A/B vector ZIP has unsafe member: ' + name)
    return set(names)


def verify_and_extract_vectors(path, scratch, records):
    expected = {}
    selected = {}
    for label, stages in records.items():
        for stage, item in stages.items():
            name = AB_VECTOR_ARCHIVE_ROOT + label + '/' + stage + '.f32'
            expected[name] = item
            if stage in SELECTED:
                selected[name] = scratch / 'vision-vectors' / label / (stage + '.f32')
    if len(selected) != 48 or len(expected) != 12 * 37:
        raise ValueError('A/B local vector selection is incomplete')
    with zipfile.ZipFile(path) as archive:
        if vector_zip_members(archive) != set(expected):
            raise ValueError('A/B vector ZIP member set differs from all 12 manifests')
        for info in archive.infolist():
            item = expected[info.filename]
            if info.file_size != item['bytes']:
                raise ValueError('A/B vector ZIP declared size differs: ' + info.filename)
            target = selected.get(info.filename)
            if target:
                target.parent.mkdir(parents=True, exist_ok=True)
                output = target.open('wb')
            else:
                output = None
            digest = hashlib.sha256()
            size = 0
            try:
                with archive.open(info) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        digest.update(block)
                        size += len(block)
                        if output:
                            output.write(block)
            finally:
                if output:
                    output.close()
            if size != item['bytes'] or digest.hexdigest() != item['sha256']:
                raise ValueError('A/B raw-vector member SHA/size differs: ' + info.filename)
    return {'verified_member_count': len(expected), 'selected_member_count': len(selected),
            'all_manifest_sha256_match': True, 'extracted_only_selected': True}


def extract_frozen(pin, members, scratch, helper):
    name = pin[1].split('-')[1]
    archive_path = scratch / (name + '-transport.zip')
    helper.download_artifact(pin, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = helper.zip_members(archive)
        if not {item[0] for item in members}.issubset(names):
            raise ValueError('Frozen source ZIP member missing: ' + name)
        for filename, size, digest in members:
            helper.extract_member(archive, filename, scratch / filename, size, digest)
    archive_path.unlink()  # The verified exact source artifact remains remote.
    return {'artifact_id': pin[0], 'zip_sha256': pin[3], 'files': {
        filename: {'bytes': size, 'sha256': digest} for filename, size, digest in members}}


def process_command(label, args, out, report, *, env=None, timeout=1800):
    path = out / (label + '.log')
    start = time.monotonic()
    timed_out = False
    with path.open('w') as stream:
        child = subprocess.Popen([str(arg) for arg in args], cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
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
    item = {'argv': [str(arg) for arg in args], 'exit_code': child.returncode,
            'timed_out': timed_out, 'wall_seconds': time.monotonic() - start,
            'log': path.name, 'log_sha256': sha(path)}
    report.setdefault('commands', {})[label] = item
    save(out, report)
    return item


def parse_local_log(text):
    completed = COMPLETE.findall(text)
    controls = re.findall(r'^LOCAL_OP_CONTROL native\.(norm2|up) exact=1 n=(\d+)/(\d+)$', text, re.M)
    inputs = re.findall(r'^LOCAL_OP_INPUT norm2 n=(\d+) max_abs=([\deE+.-]+) '
                        r'mean_abs=([\deE+.-]+) worst_index=(\d+) row=(\d+) '
                        r'native=([\deE+.-]+) original=([\deE+.-]+)$', text, re.M)
    metrics = METRIC.findall(text)
    budgets = BUDGET.findall(text)
    required = {(mode, stage) for mode in ('native_input', 'same_original_input')
                for stage in ('prefill.vision.probe.11.norm2', 'prefill.vision.probe.11.up')}
    if (len(completed) != 1 or len(controls) != 2 or {row[0] for row in controls} != {'norm2', 'up'}
            or any(row[1] != row[2] for row in controls) or len(inputs) != 1
            or len(metrics) != 4 or {(row[0], row[1]) for row in metrics} != required
            or len(budgets) != 4 or [row[0] for row in budgets] != [row[1] for row in metrics]):
        raise ValueError('Local operator transcript incomplete')
    for row in inputs:
        if (any(not math.isfinite(float(value)) for value in (row[1], row[2], row[5], row[6]))
                or int(row[0]) < 1 or float(row[1]) < 0 or float(row[2]) < 0
                or int(row[3]) >= int(row[0])):
            raise ValueError('Nonfinite local operator input metric')
    result = {}
    for row, budget in zip(metrics, budgets):
        mode, stage, count, maximum, mean, cosine, worst, token, actual, original, passed = row
        if (int(count) < 1 or int(worst) >= int(count)
                or any(not math.isfinite(float(value)) for value in
                       (maximum, mean, cosine, actual, original, budget[3], budget[4]))
                or float(maximum) < 0 or float(mean) < 0 or not -1.00001 <= float(cosine) <= 1.00001
                or float(budget[3]) < 0 or float(budget[4]) < 0):
            raise ValueError('Nonfinite or invalid local operator metric')
        result[mode + ':' + stage] = {'n': int(count), 'max_abs': float(maximum),
            'mean_abs': float(mean), 'cos_min': float(cosine), 'worst_index': int(worst),
            'row': int(token), 'candidate': float(actual), 'original': float(original),
            'existing_budget_pass': passed == '1',
            'strict_absolute_pass': budget[1] == '1', 'calibrated': budget[2] == '1',
            'self_max_abs': float(budget[3]), 'max_ratio': float(budget[4])}
    return {'native_replay_exact': True, 'input': {'n': int(inputs[0][0]),
            'max_abs': float(inputs[0][1]), 'mean_abs': float(inputs[0][2]),
            'worst_index': int(inputs[0][3]), 'row': int(inputs[0][4])},
            'metrics': result, 'gates': {'precise_norm': int(completed[0][0]),
                                        'cascade_norm': int(completed[0][1]),
                                        'precise_attn': int(completed[0][2])},
            'qualification_claim': False}


def main(preflight_only=False):
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'ovis-vision-local-ops'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    report = {'phase': 'pins', 'status': 'started', 'source_commit': os.environ.get('GITHUB_SHA'),
              'driver_sha256': sha(__file__), 'qualification_claim': False, 'arms': {}, 'commands': {},
              'scope': 'Frozen block-11 norm2/up diagnostic for all 12 original A/B arms'}
    save(out, report)
    try:
        require_pins()  # Must fail before gh auth, framework imports, builds, or weights.
        helper = load_ab_helper()
        report['phase'] = 'service'; save(out, report)
        report['ab_service'] = {'reports': service_ab(AB_REPORT_ARTIFACT),
                                'vectors': service_ab(AB_VECTOR_ARTIFACT)}
        report['source_service'] = {
            'model_reports': helper.service_source(helper.SOURCE_RUN, helper.SOURCE_HEAD,
                                                  helper.SOURCE_REPORT_ARTIFACT, conclusion='failure'),
            'model_reference': helper.service_source(helper.SOURCE_RUN, helper.SOURCE_HEAD,
                                                    helper.SOURCE_REFERENCE_ARTIFACT, conclusion='failure'),
            'export': helper.service_source(helper.EXPORT_RUN, helper.EXPORT_HEAD,
                                           helper.EXPORT_ARTIFACT, conclusion='success')}
        source_out = out / 'frozen-model-source'
        source_out.mkdir(exist_ok=True)
        source_gate = {'phase': 'source-preflight'}
        helper.preflight(source_out, scratch, source_gate)
        report['frozen_model_contract'] = source_gate['source_contract']
        report['target_source_sha256'] = TARGET_SOURCE_SHA256
        report['operator_header_sha256'] = OPERATOR_HEADER_SHA256
        report['phase'] = 'ab-reports'; save(out, report)
        ab_zip = scratch / 'ab-reports-transport.zip'
        helper.download_artifact(AB_REPORT_ARTIFACT, ab_zip, timeout=300)
        ab_report, records = read_ab_reports(ab_zip, out, helper)
        if ab_report['source_contract'] != report['frozen_model_contract']:
            raise ValueError('A/B source contract differs from independently replayed model preflight')
        ab_zip.unlink()
        report['ab_report_sha256'] = AB_REPORT_SHA256
        report['ab_driver_sha256'] = ab_report['driver_sha256']
        report['ab_passing_arms'] = ab_report['passing_arms']
        report['ab_arm_count'] = len(records)
        report['phase'] = 'preflight-complete'; save(out, report)
        if preflight_only:
            report['status'] = 'preflight passed; no vectors, models, build, or inference loaded'
            save(out, report)
            return report
        report['phase'] = 'ab-vectors'; save(out, report)
        vectors_zip = scratch / 'ab-vectors-transport.zip'
        helper.download_artifact(AB_VECTOR_ARTIFACT, vectors_zip)
        report['vector_integrity'] = verify_and_extract_vectors(vectors_zip, scratch, records)
        vectors_zip.unlink()  # Verified task-owned transport; full vectors remain remote.
        save(out, report)
        report['phase'] = 'frozen-source'; save(out, report)
        if shutil.disk_usage(scratch).free < 9 * 1024 ** 3:
            raise ValueError('Hosted scratch has under 9 GiB free before frozen weights')
        report['frozen_export'] = extract_frozen(helper.EXPORT_ARTIFACT, tuple(helper.MODELS.values()),
                                                 scratch, helper)
        report['frozen_reference'] = extract_frozen(helper.SOURCE_REFERENCE_ARTIFACT,
            (helper.SOURCE_REFERENCE_GGUF, helper.SOURCE_REFERENCE_JSON), scratch, helper)
        source_shapes = json.loads((scratch / helper.SOURCE_REFERENCE_JSON[0]).read_text())['stages']
        for stages in records.values():
            for stage, item in stages.items():
                if (item.get('original_shape') != source_shapes.get(stage)
                        or item['bytes'] != math.prod(item['original_shape']) * 4):
                    raise ValueError('A/B manifest shape differs from frozen original: ' + stage)
        report['phase'] = 'build'; save(out, report)
        build = scratch / 'build'
        configure = ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                     '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF', '-DGGML_METAL=OFF',
                     '-DGGML_BLAS=OFF', '-DCRISPEMBED_BUILD_SHARED=OFF']
        for label, command, timeout in (
                ('cmake-configure', configure, 600),
                ('cpu-baseline-check', [sys.executable, ROOT / 'scripts/check-cpu-baseline.py', build], 120),
                ('local-op-target-build', ['cmake', '--build', build, '--config', 'Release',
                                           '--target', 'test-ovisocr2-vision-local-ops', '-j2'], 1800)):
            if process_command(label, command, out, report, timeout=timeout)['exit_code'] != 0:
                raise ValueError('Hosted local operator build/check failed: ' + label)
        binary = build / 'test-ovisocr2-vision-local-ops'
        report['binary_sha256'] = sha(binary)
        report['phase'] = 'local-operators'; save(out, report)
        errors = []
        for precision in helper.MODELS:
            model = scratch / helper.MODELS[precision][0]
            for arm, enabled in helper.ARM_GATES.items():
                label = precision + '-' + arm
                env = dict(os.environ, **helper.BASE_ENV)
                env.pop('CRISPEMBED_DIFF_DUMP_DIR', None)
                for name in enabled:
                    env[name] = '1'
                entry = process_command(label, [binary, model, scratch / helper.SOURCE_REFERENCE_GGUF[0],
                    scratch / 'vision-vectors' / label], out, report, env=env)
                arm_report = {'precision': precision, 'enabled_gates': list(enabled),
                              'exit_code': entry['exit_code'], 'timed_out': entry['timed_out'],
                              'log_sha256': entry['log_sha256']}
                try:
                    if entry['exit_code'] != 0 or entry['timed_out']:
                        raise ValueError('Local operator replay exited before complete diagnostic')
                    parsed = parse_local_log((out / entry['log']).read_text(errors='replace'))
                    gates = {'precise_norm': int('CRISPEMBED_QWEN35_VISION_PRECISE_NORM' in enabled),
                             'cascade_norm': int('CRISPEMBED_QWEN35_VISION_CASCADE_NORM' in enabled),
                             'precise_attn': int('CRISPEMBED_QWEN35_VISION_PRECISE_ATTN' in enabled)}
                    if parsed['gates'] != gates:
                        raise ValueError('Local operator gate transcript differs from A/B arm')
                    for stage in (k for k in SELECTED if k.endswith(('.norm2', '.up'))):
                        count = records[label][stage]['bytes'] // 4
                        for mode in ('native_input', 'same_original_input'):
                            if parsed['metrics'][mode + ':' + stage]['n'] != count:
                                raise ValueError('Local operator metric size differs from A/B manifest')
                    if parsed['input']['n'] != records[label]['prefill.vision.probe.11.norm2']['bytes'] // 4:
                        raise ValueError('Local operator input size differs from A/B manifest')
                    arm_report.update(parsed)
                except (OSError, ValueError) as error:
                    arm_report['diagnostic_error'] = type(error).__name__ + ': ' + str(error)
                    errors.append(label)
                report['arms'][label] = arm_report
                save(out, report)
            if precision == 'f32':
                if sha(model) != helper.MODELS['f32'][2]:
                    raise ValueError('Frozen F32 model changed before task-owned cleanup')
                model.unlink()
                report['f32_cleanup'] = {'verified_frozen_file_removed': True,
                                         'durable_artifact_id': helper.EXPORT_ARTIFACT[0]}
                save(out, report)
        report['diagnostic_errors'] = errors
        report['all_twelve_complete'] = len(report['arms']) == 12 and not errors
        report['status'] = 'completed frozen local operator diagnosis; no qualification'
        report['phase'] = 'complete'; save(out, report)
        if not report['all_twelve_complete']:
            raise ValueError('One or more local operator arms lacked nonvacuous diagnostics')
        return report
    except BaseException as error:
        report['status'] = 'failed frozen local operator diagnostic'
        report['error_type'] = type(error).__name__
        report['error'] = str(error)
        report['error_phase'] = report['phase']
        save(out, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    arguments = parser.parse_args()
    main(preflight_only=arguments.preflight)
