#!/usr/bin/env python3
"""Hosted replay of frozen OvisOCR2 source intermediates against all native arms.

This lane never loads Transformers, recaptures HF tensors, or reconverts weights.
Its selected existing vision gate preset must be approved after the A/B report.
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
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
AB_HELPER = ROOT / 'tools/ci-heavy/ovisocr2_vision_audit.py'
AB_HELPER_SHA = 'a812910212ac030e13aa2e77bbab26460b6375f8978a802bf44c829c95f0a9c3'
MODEL_HELPER = ROOT / 'tools/ci-heavy/ovisocr2_model_audit.py'
MODEL_HELPER_SHA = '34f6e764810c9f9425f29f0e9f6389d37b00f062036c3dc3ca4347f731d096b5'
SOURCE_DELTA_TEST_SHA = 'd8552b2cc6999e42202c80ef4108be6a598679d23b775583375ab53dc83faf20'
TRACE_DELTA_TEST_SHA = 'd74964fc6b6fe3b35435b2f419cfe34acd8fdb9309b12ec4fc1ea39f2d96340a'
SOURCE_REPORT_SHA = 'bdcfc725ccdf4a13d4f16ed3f7238d12f103ddc4d0e47a861c82122a9c7b98ca'
READABLE_SHA = 'fba91697e616a1308653ec37c9e0d56bbec20f0c202a4d6098c31c25c92b9d0a'
LICENSE_SHA = 'bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a'
FROZEN_IDS = [9419, 77796, 271, 16, 17, 18, 19, 20, 248046]
FROZEN_TEXT = 'Hello OCR\n\n12345'
# Set only after the hosted six-arm A/B artifact is independently reviewed.
# No implicit baseline or production default is selected by this recipe.
APPROVED_PRESET = 'both'
APPROVED_AB_RUN = 40047163838
APPROVED_AB_REPORT_SHA = 'd1c75b079a45691493ccdd837332a88912ae66fb5c91f492d473d3ca5e2600b4'
APPROVED_AB_HEAD = 'df65e911c08c2f8f87fd6eae96f907d6f5f9e8d4'
APPROVED_AB_ARTIFACT_ID = 12542806209
APPROVED_AB_ARTIFACT_SHA = 'df9c8197012dc12521d24e01fd2b54b50079c091658146192f95c4232b3ec1cf'
APPROVED_AB_ARTIFACT_BYTES = 155975


def save_report(out, report):
    (Path(out) / 'ovisocr2-model-audit.json').write_text(
        json.dumps(report, indent=2, allow_nan=False) + '\n')


def load_helper(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_helpers():
    import hashlib
    def sha(path):
        value = hashlib.sha256()
        with Path(path).open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                value.update(block)
        return value.hexdigest()
    if (sha(AB_HELPER) != AB_HELPER_SHA or sha(MODEL_HELPER) != MODEL_HELPER_SHA
            or sha(ROOT / 'tests/test_qwen35_delta.cpp') != TRACE_DELTA_TEST_SHA):
        raise ValueError('Frozen A/B/model helper or test-only delta trace source differs')
    return load_helper('ovis_replay_ab_helper', AB_HELPER), load_helper('ovis_replay_model_helper', MODEL_HELPER)


def approved_preset(requested, ab):
    if (APPROVED_PRESET is None or APPROVED_AB_RUN is None or APPROVED_AB_REPORT_SHA is None):
        raise ValueError('Independently reviewed A/B preset and run are not pinned in source')
    if requested != APPROVED_PRESET or requested not in ab.ARM_GATES:
        raise ValueError('Requested vision math preset differs from reviewed source selection')
    return {'name': requested, 'enabled_gates': list(ab.ARM_GATES[requested]),
            'reviewed_ab_run': APPROVED_AB_RUN, 'reviewed_ab_report_sha256': APPROVED_AB_REPORT_SHA,
            'scope': 'Existing runtime gates only; no model, comparator, tolerance or default change'}


def native_source_closure(source, ab):
    """The source report fixes every implementation and comparison file, not only vision."""
    original = source.get('native_source_sha256')
    exceptions = {'tests/test_qwen35_vision.cpp': (ab.SOURCE_TEST_SHA, ab.TRACE_TEST_SHA),
                  'tests/test_qwen35_delta.cpp': (SOURCE_DELTA_TEST_SHA, TRACE_DELTA_TEST_SHA)}
    if (not isinstance(original, dict) or len(original) != 29
            or not set(exceptions).issubset(original)
            or not {'src/qwen35_norm.cpp', 'src/qwen35_math.h', 'src/qwen35_linear.cpp',
                    'src/qwen35_attention.cpp', 'src/crispembed_diff.h',
                    'tests/qwen35_parity.h', 'tests/test_qwen35_decoder.cpp',
                    'tests/test_ovisocr2_context.cpp'}.issubset(original)):
        raise ValueError('Frozen native implementation/comparison source set differs')
    discovered = {str(path.relative_to(ROOT)) for path in (ROOT / 'src').glob('qwen35*')
                  if path.is_file() and path.suffix in ('.cpp', '.h')}
    discovered |= {'src/crispembed_diff.h', 'tests/qwen35_parity.h',
                   'tests/test_qwen35_decoder.cpp', 'tests/test_qwen35_vision.cpp',
                   'tests/test_qwen35_delta.cpp', 'tests/test_ovisocr2_context.cpp',
                   'tests/ovisocr2_quality_run.cpp'}
    if set(original) != discovered:
        raise ValueError('Current native implementation/comparison source set differs')
    reviewed = {}
    for name, old_sha in original.items():
        if (not isinstance(name, str) or not re.fullmatch(r'(?:src|tests)/[A-Za-z0-9_./-]+\.(?:h|cpp)', name)
                or '..' in Path(name).parts or not re.fullmatch('[0-9a-f]{64}', old_sha)):
            raise ValueError('Frozen native source path/hash differs')
        current_sha = ab.sha(ROOT / name)
        if name in exceptions:
            if (old_sha, current_sha) != exceptions[name]:
                raise ValueError('Reviewed test-only trace fix source differs: ' + name)
            reviewed[name] = {'original_sha256': old_sha, 'replay_sha256': current_sha,
                              'scope': 'test-only raw-dump preservation'}
        elif current_sha != old_sha:
            raise ValueError('Native implementation/comparator changed since frozen source: ' + name)
    return {'all_original_source_files_checked': len(original),
            'unchanged_files': len(original) - len(exceptions),
            'reviewed_test_only_differences': reviewed,
            'implementation_and_comparison_source_unchanged': True}


def source_reference(ab, scratch, out, report, source):
    pin = ab.SOURCE_REFERENCE_ARTIFACT
    archive_path = scratch / 'frozen-reference.zip'
    ab.download_artifact(pin, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = ab.zip_members(archive)
        required = {'official-delta.gguf', 'ovisocr2-original.json', 'owned-readable.png', 'LICENSE'}
        if names != required:
            raise ValueError('Frozen numerical reference artifact members differ')
        files = {}
        for name, count, digest in (ab.SOURCE_REFERENCE_GGUF, ab.SOURCE_REFERENCE_JSON):
            ab.extract_member(archive, name, scratch / name, count, digest)
            files[name] = {'bytes': count, 'sha256': digest}
        for name, digest in (('owned-readable.png', READABLE_SHA), ('LICENSE', LICENSE_SHA)):
            info = archive.getinfo(name)
            ab.extract_member(archive, name, scratch / name, info.file_size, digest)
            files[name] = {'bytes': info.file_size, 'sha256': digest}
    archive_path.unlink()
    reference = json.loads((scratch / 'ovisocr2-original.json').read_text())
    if (reference.get('generated_ids') != [FROZEN_IDS]
            or reference.get('decoded') != [FROZEN_TEXT]
            or reference.get('raw_decoded') != [FROZEN_TEXT + '<|im_end|>']
            or len(reference.get('stages', {})) != 943
            or reference.get('missing_stages') != []
            or source.get('reference_artifacts', {}).get('gguf', {}).get('sha256') != files['official-delta.gguf']['sha256']
            or source.get('reference_artifacts', {}).get('json', {}).get('sha256') != files['ovisocr2-original.json']['sha256']
            or source.get('readable', {}).get('png_sha256') != files['owned-readable.png']['sha256']):
        raise ValueError('Frozen original numerical reference output/source differs')
    report['frozen_reference'] = {'source_run': ab.SOURCE_RUN, 'source_head': ab.SOURCE_HEAD,
        'artifact_id': pin[0], 'artifact_zip_sha256': pin[3], 'verified_zip_removed': True,
        'files': files, 'stage_count': len(reference['stages']),
        'old_npy_exact': 871, 'new_cache_states': 72,
        'original_npz_sha256_retained_at_source_run': source['reference_artifacts']['npz']['sha256']}
    return reference


def parse_metric_names(log, *, allow_delta_repeat=False):
    pass_fail = re.findall(r'^(?:PASS|FAIL) ([A-Za-z0-9_.]+): cos_min=', log, re.M)
    trace = re.findall(r'^TRACE ([A-Za-z0-9_.]+) max_abs=', log, re.M)
    if not pass_fail:
        raise ValueError('Native metric stages absent')
    if len(pass_fail) != len(set(pass_fail)) and not allow_delta_repeat:
        raise ValueError('Repeated native metric stages')
    return set(pass_fail) | set(trace), {'pass_fail': pass_fail, 'trace': trace}


def raw_vector_manifest(folder, source_shapes, log, arm, ab):
    """Retain all Ref.compare raw vectors, including decoder TRACE observations."""
    metric_names, observed = parse_metric_names(log, allow_delta_repeat=arm.endswith('-delta'))
    if arm in ('f32-vision', 'f16-vision'):
        parsed = ab.parse_vision_log('unused', text=log, require_dump_integrity=True)
        if metric_names != set(parsed['stages']) or len(metric_names) != 37:
            raise ValueError('Vision raw metric set differs from 37 preserved stages')
        marker = 'PASS VISION_RAW_DUMP_INTEGRITY stages=2'
    elif arm.endswith('-delta'):
        if log.splitlines().count('PASS DELTA_RAW_DUMP_INTEGRITY stages=4 variants=linear-core,reset') != 1:
            raise ValueError('Actual delta raw output integrity marker absent')
        if not {'prefill.delta.output', 'cached.delta.output'}.issubset(metric_names):
            raise ValueError('Delta output metric coverage differs')
        marker = 'PASS DELTA_RAW_DUMP_INTEGRITY stages=4 variants=linear-core,reset'
    else:
        marker = None
        required_layers = {f'{phase}.decoder.layer.{layer}' for phase in
                           ('prefill', 'cached', 'full_prefix') for layer in range(24)}
        required_cache = {name for name in source_shapes
                          if re.fullmatch(r'(prefill|cached)\.cache\.\d+\.(keys|values|conv_states|recurrent_states)', name)}
        if (not required_layers.issubset(metric_names)
                or not required_cache.issubset(metric_names)
                or not {f'generated.scores.{i}' for i in range(len(FROZEN_IDS))}.issubset(metric_names)):
            raise ValueError('Decoder layers/cache/full-vocabulary score coverage differs')
        if arm.endswith('-decoder-native-frontend') and not observed['trace']:
            raise ValueError('Native frontend decoder attention TRACE observations absent')
    files = {path.name.removesuffix('.f32'): path for path in Path(folder).glob('*.f32')}
    if set(files) != metric_names:
        raise ValueError('Actual raw Ref.compare stage file set differs: ' + arm)
    variants = {}
    if arm.endswith('-delta'):
        expected_variants = {
            'linear-core': {'prefill.delta.final_state', 'cached.delta.final_state'},
            'reset': {'prefill.delta.output', 'prefill.delta.final_state'},
        }
        for variant, stages in expected_variants.items():
            found = {p.name.removesuffix('.f32'): p for p in (Path(folder) / variant).glob('*.f32')}
            if set(found) != stages:
                raise ValueError('Delta repeated-positive variant dumps differ: ' + variant)
            variants[variant] = found
        if {p.relative_to(folder).as_posix() for p in Path(folder).rglob('*.f32')} != (
                {p.name for p in files.values()}
                | {variant + '/' + stage + '.f32' for variant, stages in expected_variants.items() for stage in stages}):
            raise ValueError('Unknown or missing delta raw variant dumps')
    elif set(Path(folder).rglob('*.f32')) != set(files.values()):
        raise ValueError('Unexpected nested raw dump for non-delta arm')
    rows = []
    for variant, stage_files in [('', files), *sorted(variants.items())]:
      for stage in sorted(stage_files):
        shape = source_shapes.get(stage)
        if not isinstance(shape, list) or not shape or any(type(x) is not int or x < 1 for x in shape):
            raise ValueError('Frozen source stage shape missing: ' + stage)
        expected_bytes = math.prod(shape) * 4
        path = stage_files[stage]
        if path.stat().st_size != expected_bytes:
            raise ValueError('Actual raw F32 stage byte count differs: ' + stage)
        rows.append({'stage': stage, 'variant': variant or 'primary',
                     'original_stage': stage, 'file': 'native-diff/' + arm + '/' +
                     (variant + '/' if variant else '') + path.name,
                     'bytes': expected_bytes, 'sha256': ab.sha(path), 'source_shape': shape})
    return {'arm': arm, 'metric_stage_count': len(metric_names), 'trace_stage_count': len(observed['trace']),
            'raw_vector_count': len(rows), 'coverage_complete': True,
            'variant_stage_count': len(rows) - len(metric_names),
            'dump_integrity_marker': marker, 'files': rows,
            'scope': 'All raw Ref.compare F32 vectors retained remotely; context uses C ABI and has no Ref.compare vectors'}


def parse_ids(log):
    match = re.search(r'^Native generated IDs:((?: \d+)+)$', log, re.M)
    return [int(x) for x in match[1].split()] if match else None


def verify_arm_log(arm, log, source_shapes, ab):
    if arm.endswith('-context'):
        ids = re.search(r'^IDS raw:((?: \d+)+)$', log, re.M)
        encoded = re.search(r'^UTF8_HEX raw:([0-9a-f]+)$', log, re.M)
        if (ids is None or [int(x) for x in ids[1].split()] != FROZEN_IDS
                or encoded is None or bytes.fromhex(encoded[1]).decode() != FROZEN_TEXT
                or 'PASS OVIS_CONTEXT raw ' not in log
                or 'PASS OVIS_CONTEXT repeat ' not in log
                or 'stop=1' not in log):
            raise ValueError('Native context IDs/text/EOS/repeat controls differ')
        return {'ids': FROZEN_IDS, 'decoded': FROZEN_TEXT, 'eos_stop': 248046,
                'context_raw_repeat': True, 'comparison_coverage': True}
    names, _ = parse_metric_names(log, allow_delta_repeat=arm.endswith('-delta'))
    if arm.endswith('-decoder-prepared') or '-decoder-native-' in arm:
        if parse_ids(log) != FROZEN_IDS:
            raise ValueError('Native decoder generated IDs differ')
        if arm.endswith('-decoder-native-frontend'):
            encoded = re.search(r'^Native decoded UTF8 hex:([0-9a-f]+)$', log, re.M)
            if encoded is None or bytes.fromhex(encoded[1]).decode() != FROZEN_TEXT:
                raise ValueError('Native decoded UTF8 text differs')
        return {'ids': FROZEN_IDS, 'decoded': FROZEN_TEXT if arm.endswith('-decoder-native-frontend') else None,
                'decoder_layers': len([x for x in names if re.fullmatch(
                    r'(prefill|cached|full_prefix)\.decoder\.layer\.\d+', x)]),
                'generated_score_count': len([x for x in names if x.startswith('generated.scores.')]),
                'comparison_coverage': True}
    if arm in ('f32-vision', 'f16-vision'):
        stage = ab.parse_vision_log('unused', text=log, require_dump_integrity=True)
        return {'first_failing_operation': stage['first_failing_operation'],
                'merger': stage['merger'], 'comparison_coverage': stage['stage_count'] == 37}
    if arm.endswith('-delta'):
        return {'delta_prefill_cached': {'prefill.delta.output', 'cached.delta.output'}.issubset(names),
                'comparison_coverage': True}
    raise ValueError('Unknown native replay arm')


def source_exports(ab, scratch, source, report):
    pin = ab.EXPORT_ARTIFACT
    archive_path = scratch / 'qualified-exports.zip'
    ab.download_artifact(pin, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = ab.zip_members(archive)
        if not {row[0] for row in ab.MODELS.values()}.issubset(names):
            raise ValueError('Qualified F32/F16 model artifact members differ')
        for precision, (name, size, digest) in ab.MODELS.items():
            ab.extract_member(archive, name, scratch / name, size, digest)
            old = source.get('exports', {}).get(precision, {})
            if old.get('model_sha256') != digest or old.get('bytes') != size:
                raise ValueError('Frozen source export evidence differs: ' + precision)
    archive_path.unlink()
    report['export_source'] = {'run': ab.EXPORT_RUN, 'head': ab.EXPORT_HEAD,
        'artifact_id': pin[0], 'artifact_zip_sha256': pin[3],
        'models': source['exports'], 'verified_zip_removed': True}


def reviewed_ab_selection(ab, scratch, preset):
    # The review pins are deliberately absent until the actual A/B run finishes.
    if any(value is None for value in (APPROVED_AB_RUN, APPROVED_AB_REPORT_SHA,
                                       APPROVED_AB_ARTIFACT_ID, APPROVED_AB_ARTIFACT_SHA,
                                       APPROVED_AB_ARTIFACT_BYTES, APPROVED_AB_HEAD)):
        raise ValueError('Reviewed hosted vision A/B result/artifact pins are not yet set')
    pin = (APPROVED_AB_ARTIFACT_ID, f'ovisocr2-vision-ab-reports-{APPROVED_AB_RUN}',
           APPROVED_AB_ARTIFACT_BYTES, APPROVED_AB_ARTIFACT_SHA)
    service = ab.service_source(APPROVED_AB_RUN, APPROVED_AB_HEAD, pin, conclusion='success')
    archive_path = scratch / 'reviewed-ab-reports.zip'
    ab.download_artifact(pin, archive_path, timeout=600)
    with zipfile.ZipFile(archive_path) as archive:
        names = ab.zip_members(archive)
        if 'ovisocr2-vision-audit.json' not in names:
            raise ValueError('Reviewed A/B producer report missing')
        payload = archive.read('ovisocr2-vision-audit.json')
    if hashlib.sha256(payload).hexdigest() != APPROVED_AB_REPORT_SHA:
        raise ValueError('Reviewed A/B producer report SHA differs')
    data = json.loads(payload)
    if (data.get('status') != 'completed diagnostic vision A/B; no qualification'
            or data.get('source_commit') != APPROVED_AB_HEAD
            or data.get('source_contract', {}).get('failed_only_vision') is not True
            or data.get('qualification', {}).get('native_alias') is not False):
        raise ValueError('Reviewed A/B producer status/source differs')
    for precision in ('f32', 'f16'):
        arm = data.get('arms', {}).get(precision + '-' + preset, {})
        if (arm.get('exit_code') != 0 or arm.get('metrics', {}).get('first_failing_operation') is not None
                or arm.get('metrics', {}).get('actual_dump_integrity_passed') is not True
                or arm.get('raw_vector_actual_count') != 37
                or arm.get('enabled_gates') != list(ab.ARM_GATES[preset])):
            raise ValueError('Chosen A/B vision arm lacks exact positive evidence: ' + precision)
    archive_path.unlink()
    return {'service': service, 'report_sha256': APPROVED_AB_REPORT_SHA,
            'preset': preset, 'f32_f16_vision_positive': True,
            'scope': 'A/B result selects an existing gate only; full native replay remains independent'}


def recorded_command(label, args, out, report, ab, *, env=None, timeout=1800):
    # Reuse the bounded heartbeat/process-group implementation, then copy its
    # fields into the established model-audit command schema.
    logdir = out / 'arm-logs'
    logdir.mkdir(exist_ok=True)
    temporary = {'commands': {}}
    entry = ab.process_command(label, args, logdir, temporary, env=env, timeout=timeout)
    (logdir / 'ovisocr2-vision-audit.json').unlink(missing_ok=True)
    path = logdir / entry['log']
    lines = path.read_text(errors='replace').splitlines()
    metrics = [line for line in lines if re.search(
        r'\b(?:PASS|FAIL|BUDGET|DECODED|IDS|UTF8_HEX|logits|cos_min|decoded|confidence|cache|argmax|TRACE)\b',
        line)]
    failures = [line for line in lines if re.search(r'\b(?:FAIL|ERROR|Traceback|Error:)\b', line)]
    record = {'argv': entry['argv'], 'log': 'arm-logs/' + entry['log'], 'exit_code': entry['exit_code'],
              'bytes': path.stat().st_size, 'sha256': entry['log_sha256'],
              'timed_out': entry['timed_out'], 'wall_seconds': entry['wall_seconds'],
              'first_failure': failures[0] if failures else None, 'metric_lines': metrics,
              'passed': entry['exit_code'] == 0 and not entry['timed_out']}
    report['commands'][label] = record
    return record


def full_replay(out, scratch, source, reference, ab, model, preset, report):
    report['phase'] = 'portable-native-build'; save_report(out, report)
    build = scratch / 'build'
    configure = ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                 '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF', '-DGGML_METAL=OFF',
                 '-DGGML_BLAS=OFF', '-DCRISPEMBED_BUILD_SHARED=OFF']
    for label, command, timeout in (
        ('cmake-configure', configure, 600),
        ('cpu-baseline-check', [sys.executable, ROOT / 'scripts/check-cpu-baseline.py', build], 120),
        ('native-target-build', ['cmake', '--build', build, '--config', 'Release', '--target',
                                 *model.NATIVE_TARGETS, '-j2'], 1800)):
        entry = recorded_command(label, command, out, report, ab, timeout=timeout)
        save_report(out, report)
        if not entry['passed']:
            raise ValueError('Portable native preparation failed: ' + label)
    report['native_binary_sha256'] = {name: ab.sha(build / name) for name in model.NATIVE_TARGETS}
    report['phase'] = 'fourteen-native-arms'; save_report(out, report)
    env_base = dict(os.environ, **model.NATIVE_ENV)
    env_base.pop('CRISPEMBED_DIFF_DUMP_DIR', None)
    for name in ab.ARM_GATES[preset]:
        env_base[name] = '1'
    report['native_environment'] = {name: env_base[name] for name in model.NATIVE_ENV}
    report['selected_vision_preset'] = {'name': preset,
        'enabled_gates': list(ab.ARM_GATES[preset]),
        'all_other_native_gates_unchanged': True,
        'no_production_default_promoted': True}
    source_shapes = reference['stages']
    if source_shapes != source['original_capture']['stages']:
        raise ValueError('Frozen source report/reference stage shapes differ')
    arm_errors = []
    for precision in ('f32', 'f16'):
        file = scratch / ab.MODELS[precision][0]
        arms = model.native_arms(precision, build, file, scratch / 'owned-readable.png',
                                 scratch / 'official-delta.gguf')
        if set(arms) != {precision + '-' + suffix for suffix in
                        ('delta', 'vision', 'decoder-prepared', 'decoder-native-vision',
                         'decoder-native-positions', 'decoder-native-frontend', 'context')}:
            raise ValueError('Original native arm set differs')
        for label, args in arms.items():
            env = env_base.copy()
            vector_dir = scratch / 'native-diff' / label
            vector_dir.mkdir(parents=True, exist_ok=True)
            env['CRISPEMBED_DIFF_DUMP_DIR'] = str(vector_dir)
            entry = recorded_command(label, args, out, report, ab, env=env, timeout=1800)
            log = (out / entry['log']).read_text(errors='replace')
            try:
                observation = verify_arm_log(label, log, source_shapes, ab)
                entry['observation'] = observation
                if label.endswith('-context'):
                    if list(vector_dir.glob('*.f32')):
                        raise ValueError('Context C ABI arm unexpectedly dumped Ref.compare vectors')
                    entry['raw_diff_not_applicable'] = 'C ABI context uses IDs/text/stops/confidence, no Ref.compare'
                else:
                    manifest = raw_vector_manifest(vector_dir, source_shapes, log, label, ab)
                    path = out / (label + '-raw-diff-manifest.json')
                    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
                    entry['raw_diff_manifest'] = path.name
                    entry['raw_diff_manifest_sha256'] = ab.sha(path)
                    entry['raw_diff_coverage_complete'] = True
                entry['comparison_coverage'] = observation['comparison_coverage']
                if not entry['comparison_coverage']:
                    raise ValueError('Native comparison coverage incomplete')
            except (OSError, ValueError, KeyError) as error:
                entry['passed'] = False
                entry['integrity_error'] = type(error).__name__ + ': ' + str(error)
                arm_errors.append(label + '-integrity')
            if entry['exit_code'] != 0 or not entry['passed']:
                report['failures'].append(label)
            save_report(out, report)
        if precision == 'f32':
            if ab.sha(file) != ab.MODELS['f32'][2]:
                raise ValueError('Verified task-owned F32 source changed before cleanup')
            file.unlink()
            report['f32_local_cleanup'] = {'removed_after_seven_arms': True,
                'source_model_sha256': ab.MODELS['f32'][2], 'durable_export_run': ab.EXPORT_RUN}
            save_report(out, report)
    report['arm_integrity_errors'] = arm_errors
    if len([key for key in report['commands'] if key.startswith(('f32-', 'f16-'))]) != 14:
        raise ValueError('Full native arm coverage incomplete')


def main(*, preflight_only=False, requested_preset=None):
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'ovis-native-replay'
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    ab, model = source_helpers()
    report = {'status': 'started', 'phase': 'preflight', 'source_commit': os.environ.get('GITHUB_SHA'),
        'source_sha256': {'tools/ci-heavy/ovisocr2_native_replay.py': ab.sha(__file__),
                          'tools/ci-heavy/ovisocr2_vision_audit.py': AB_HELPER_SHA,
                          'tools/ci-heavy/ovisocr2_model_audit.py': MODEL_HELPER_SHA,
                          'tests/test_qwen35_delta.cpp': TRACE_DELTA_TEST_SHA,
                          'tests/test_qwen35_vision.cpp': ab.TRACE_TEST_SHA},
        'status_scope': 'Frozen original CPU F32 reference versus experimental native F32/F16; no HF recapture/conversion',
        'commands': {}, 'failures': [], 'qualification': {'native_alias': False,
           'decoded_quality': False, 'reason': 'Requires independent model, quality and release review'}}
    save_report(out, report)
    try:
        preout = out / 'source-preflight'
        prescratch = scratch / 'source-preflight'
        preout.mkdir(exist_ok=True)
        prescratch.mkdir(exist_ok=True)
        ab.preflight(preout, prescratch, {'commands': {}})
        source_path = preout / 'source-ovisocr2-model-audit.json'
        if ab.sha(source_path) != SOURCE_REPORT_SHA:
            raise ValueError('Pinned source model report differs')
        source = json.loads(source_path.read_text())
        report['native_source_closure'] = native_source_closure(source, ab)
        report['source_reference_provenance'] = {'run': ab.SOURCE_RUN, 'head': ab.SOURCE_HEAD,
            'report_sha256': SOURCE_REPORT_SHA, 'original_871_npy_exact': True,
            'new_cache_states': 72, 'original_report_status': source['status'],
            'frozen_original_ids': FROZEN_IDS, 'frozen_original_text': FROZEN_TEXT}
        report['gpu_prerequisite'] = source['gpu_prerequisite']
        report['public_evidence'] = source['public_evidence']
        report['readable'] = source['readable']
        report['exports'] = source['exports']
        report['reference_artifacts'] = source['reference_artifacts']
        original = source['original_capture']
        report['original_capture'] = {key: original[key] for key in (
            'sha256', 'generation_result', 'cached_vs_full_prefix_max_abs',
            'cached_vs_full_prefix_same_argmax', 'observational_control',
            'backend_off_control', 'generated_ids', 'decoded', 'raw_decoded')}
        report['original_capture']['provenance'] = 'Reused exact frozen CPU F32 source; not recaptured in replay'
        report['phase'] = 'preflight-complete'; save_report(out, report)
        if preflight_only:
            report['status'] = 'preflight passed; no weights, build, or inference loaded'
            save_report(out, report)
            return report
        selection = approved_preset(requested_preset, ab)
        report['reviewed_ab_selection'] = reviewed_ab_selection(ab, scratch, selection['name'])
        report['selected_vision_preset'] = selection
        report['phase'] = 'frozen-model-and-reference-artifacts'; save_report(out, report)
        report['disk_free_before_large_bytes'] = shutil.disk_usage(scratch).free
        if report['disk_free_before_large_bytes'] < 9 * 1024 ** 3:
            raise ValueError('Hosted scratch has under 9 GiB free before bounded artifacts')
        source_exports(ab, scratch, source, report)
        reference = source_reference(ab, scratch, out, report, source)
        save_report(out, report)
        full_replay(out, scratch, source, reference, ab, model, selection['name'], report)
        report['status'] = ('completed diagnostic audit' if not report['failures']
                            else 'completed with failures')
        report['phase'] = 'complete'; save_report(out, report)
        return report
    except BaseException as error:
        report['status'] = 'failed frozen native replay infrastructure'
        report['error_type'] = type(error).__name__
        report['error_phase'] = report['phase']
        save_report(out, report)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--preset', choices=('baseline', 'precise_norm', 'precise_attn', 'both',
                                             'cascade', 'cascade_precise_attn'))
    args = parser.parse_args()
    if not args.preflight and args.preset is None:
        parser.error('Full replay requires an explicit reviewed --preset')
    result = main(preflight_only=args.preflight, requested_preset=args.preset)
    if result['status'] == 'completed with failures':
        sys.exit(1)
