"""Hosted replay of retained quantized candidates into immutable original probes.

No original inference, conversion, quality-panel rerun, or Hugging Face client.
Only reviewed candidate bytes and frozen references are used on hosted CI.
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
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'tests/results/qwen35-ocr'
RECIPES = {'q8': 'q8-protected', 'q4': 'q4-attention-q8',
           'q4-calibrated': 'q4-attention-q8'}
PHASES = tuple(RECIPES)
WORKFLOWS = {'.github/workflows/ovisocr2-quality.yml',
             '.github/workflows/ovisocr2-metadata.yml'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def hex_value(value, size):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{%d}' % size, value) is not None


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def checked_receipt(receipt, phase, run_id):
    """A reviewed failed numerical attempt may still be replayed diagnostically."""
    if phase not in PHASES or type(run_id) is not int or run_id <= 0:
        raise ValueError('Exact reviewed quantized precision and run required')
    producer = receipt.get('producer', {})
    review = receipt.get('independent_review', {})
    artifact = receipt.get('candidate_artifact', {})
    canonical = hashlib.sha256(json.dumps(producer, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    model = producer.get('models', {}).get(phase, {})
    cases = producer.get('native_cases', {}).get(phase, {})
    inputs = producer.get('inputs', [])
    if (receipt.get('run') != run_id or producer.get('requested_phase') != phase
            or not hex_value(receipt.get('source_commit'), 40)
            or producer.get('source_commit') != receipt['source_commit']
            or review.get('quant_diagnostic_complete') is not True
            or review.get('producer_canonical_sha256') != canonical
            or not hex_value(receipt.get('report_sha256'), 64)
            or type(receipt.get('artifact_id')) is not int or receipt['artifact_id'] <= 0
            or receipt.get('artifact_name') != f'ovisocr2-quality-reports-{run_id}'
            or not hex_value(receipt.get('artifact_zip_sha256'), 64)
            or review.get('artifact_zip_sha256') != receipt['artifact_zip_sha256']
            or model.get('recipe') != RECIPES[phase]
            or not hex_value(model.get('sha256'), 64)
            or type(model.get('bytes')) is not int or model['bytes'] < 200_000_000
            or not isinstance(inputs, list) or len(inputs) != 11
            or len({row.get('id') for row in inputs if isinstance(row, dict)}) != 11
            or not isinstance(cases, dict) or set(cases) != {row['id'] for row in inputs}
            or phase not in producer.get('scores', {})
            or not isinstance(producer.get('quant_numerical'), dict)
            or not hex_value(producer['quant_numerical'].get('log_sha256'), 64)
            or type(artifact.get('id')) is not int or artifact['id'] <= 0
            or artifact.get('name') != f'ovisocr2-quality-candidate-{run_id}'
            or not hex_value(artifact.get('zip_sha256'), 64)
            or review.get('candidate_artifact_zip_sha256') != artifact['zip_sha256']):
        raise ValueError('Reviewed quantized candidate, complete panel or source proof differs')
    for case in cases.values():
        ids, cap = case.get('token_ids'), case.get('cap')
        scores = case.get('confidences')
        if (type(cap) is not int or not isinstance(ids, list) or not 0 < len(ids) <= cap
                or any(type(token) is not int or not 0 <= token < 248320 for token in ids)
                or not isinstance(scores, list) or len(scores) != len(ids)
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or not 0 <= value <= 1 for value in scores)):
            raise ValueError('Quantized candidate case IDs/confidences incomplete')
    return {'phase': phase, 'run': run_id, 'source_commit': producer['source_commit'],
            'model_sha256': model['sha256'], 'model_bytes': model['bytes'],
            'filename': f'ovisocr2-{phase}.gguf', 'candidate_artifact': artifact,
            'report_artifact_id': receipt['artifact_id'],
            'report_artifact_zip_sha256': receipt['artifact_zip_sha256']}


def service_proof(pins, api):
    run = api(f'actions/runs/{pins["run"]}')
    if (run.get('id') != pins['run'] or run.get('head_sha') != pins['source_commit']
            or run.get('status') != 'completed' or run.get('path') not in WORKFLOWS
            or run.get('conclusion') not in ('success', 'failure')):
        raise ValueError('Reviewed candidate service source/conclusion differs')
    records = {}
    for label, artifact_id, name, digest in (
            ('reports', pins['report_artifact_id'],
             f'ovisocr2-quality-reports-{pins["run"]}', pins['report_artifact_zip_sha256']),
            ('candidate', pins['candidate_artifact']['id'],
             pins['candidate_artifact']['name'], pins['candidate_artifact']['zip_sha256'])):
        artifact = api(f'actions/artifacts/{artifact_id}')
        if (artifact.get('id') != artifact_id or artifact.get('name') != name
                or artifact.get('expired') is not False
                or artifact.get('workflow_run', {}).get('id') != pins['run']
                or artifact.get('digest') != 'sha256:' + digest):
            raise ValueError('Reviewed candidate artifact service identity differs: ' + label)
        records[label] = {'id': artifact_id, 'name': name, 'zip_sha256': digest}
    return {'run': {key: run[key] for key in
                    ('id', 'head_sha', 'status', 'conclusion', 'path')},
            'artifacts': records}


def capacity(scratch, candidate_bytes):
    free = shutil.disk_usage(scratch.parent).free
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, _, value = line.partition(':')
        if name in ('MemAvailable', 'SwapFree'):
            memory[name + '_kib'] = int(value.split()[0])
    required_disk = candidate_bytes + 5_000_000_000
    if free < required_disk or memory.get('MemAvailable_kib', 0) < 6 * 1024 * 1024:
        raise ValueError('Hosted raw replay needs candidate/reference disk margin and 6 GiB available RAM')
    return {'disk_free_bytes': free, 'required_disk_free_bytes': required_disk,
            'load_average': os.getloadavg(), **memory}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', required=True, choices=PHASES)
    parser.add_argument('--candidate-run-id', required=True, type=int)
    parser.add_argument('--artifact-dir', type=Path)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'ovis-quant-raw'
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'ovisocr2-quant-raw-replay.json'
    report = {'status': 'started quantized raw replay', 'phase': 'preflight',
              'source_commit': os.environ.get('GITHUB_SHA'), 'precision': args.precision,
              'commands': {}, 'limits': [
                  'Retained candidate and immutable original arrays only; no original recapture or panel rerun.',
                  'Raw completeness is independent of numerical qualification.',
                  'Large model/vector bytes remain on hosted resources.']}
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    def api(path):
        return json.loads(subprocess.check_output(
            ['gh', 'api', 'repos/CrispStrobe/CrispEmbed/' + path], timeout=60))
    def command(label, argv, timeout=2400):
        path = out / (label + '.log')
        started = time.monotonic()
        code, timed_out, error = None, False, None
        with path.open('wb') as stream:
            try:
                completed = subprocess.run([str(value) for value in argv], stdout=stream,
                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
                code = completed.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
            except OSError as exc:
                error = type(exc).__name__
        report['commands'][label] = {'exit_code': code, 'timed_out': timed_out,
            'process_error': error, 'log': path.name,
            'log_sha256': sha(path), 'wall_seconds': time.monotonic() - started}
        save()
        if code != 0 or timed_out or error:
            raise ValueError('Hosted replay build/control failed: ' + label)
    save()
    try:
        driver = module('ovis_quality_raw_replay', 'tools/ci-heavy/ovisocr2_quality_run.py')
        receipt_path = EVIDENCE / f'ovisocr2-{args.precision}-quality.json'
        receipt = json.loads(receipt_path.read_text())
        pins = checked_receipt(receipt, args.precision, args.candidate_run_id)
        original = json.loads(driver.ORIGINAL_REVIEW.read_text())
        baseline = driver.require_baseline(args.precision, original['run'])
        if receipt['producer'].get('prerequisite', {}).get('baseline_sha256') != baseline['baseline_sha256']:
            raise ValueError('Candidate native baseline changed')
        first = json.loads(driver.F16_REVIEW.read_text())
        if (receipt['producer'].get('inputs') != first['producer'].get('inputs')
                or receipt['producer'].get('selected_vision_preset') != baseline['selected_vision_preset']):
            raise ValueError('Candidate first-F16 inputs or selected math differs')
        contract = module('ovis_raw_tensor_contract', 'tools/ci-heavy/ovisocr2_quality_contract.py')
        records = receipt['producer'].get('tensor_records', {})
        expected_audit = contract.validate_quant_tensors(records.get('before', {}),
            records.get('after', {}), receipt['producer']['models'][args.precision]['recipe'])
        if expected_audit != receipt['producer'].get('tensor_audit'):
            raise ValueError('Candidate retained tensor audit differs')
        report['prerequisite'] = baseline
        report['candidate'] = pins
        report['candidate_receipt_sha256'] = sha(receipt_path)
        report['candidate_service'] = service_proof(pins, api)
        report['tensor_records'] = records
        report['first_linear_weights'] = driver.first_linear_weight_records(records)
        report['f16_source_closure'] = driver.f16_source_closure(
            first['producer'], first['source_commit'], report['source_commit'])
        report['cpu_runtime'] = module('ovis_raw_cpu',
            'tools/ci-heavy/ovisocr2_model_audit.py').cpu_runtime_attribution()
        report['source_sha256'] = {name: sha(ROOT / name) for name in (
            'tools/ci-heavy/ovisocr2_quant_raw_replay.py',
            'tools/ci-heavy/ovisocr2_quality_run.py', 'tests/test_qwen35_linear.cpp',
            'tests/test_qwen35_decoder.cpp', 'tests/qwen35_parity.h', 'src/crispembed_diff.h')}
        if os.environ.get('GITHUB_ACTIONS') == 'true':
            report['capacity'] = capacity(scratch, pins['model_bytes'])
        report['phase'] = 'preflight-complete'
        save()
        if args.preflight:
            report['status'] = 'preflight passed; no candidate/reference weights accessed'
            save()
            return
        if os.environ.get('GITHUB_ACTIONS') != 'true':
            raise ValueError('Heavy raw replay runs only in the hosted workflow')
        if args.artifact_dir is None:
            raise ValueError('Retained reviewed candidate artifact directory required')
        scratch.mkdir(parents=True, exist_ok=True)
        candidates = list(args.artifact_dir.rglob(pins['filename']))
        if (len(candidates) != 1 or candidates[0].stat().st_size != pins['model_bytes']
                or sha(candidates[0]) != pins['model_sha256']):
            raise ValueError('Actual retained candidate SHA/bytes differ')
        report['phase'] = 'reuse-frozen-original-reference'
        save()
        model = json.loads(driver.BASELINE.read_text())['producer']
        reference, proof = driver.original_diff_reference(
            baseline['baseline_pins'], model, scratch,
            receipt['producer']['metadata_source_sha256']['LICENSE'])
        report['numerical_reference'] = proof
        build = scratch / 'build'
        command('cmake-configure', ['cmake', '-S', ROOT, '-B', build,
            '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF', '-DCRISPEMBED_NATIVE=OFF',
            '-DGGML_BLAS=OFF', '-DGGML_METAL=OFF', '-DCRISPEMBED_BUILD_SHARED=ON'])
        command('cpu-baseline-check', [sys.executable, ROOT / 'scripts/check-cpu-baseline.py', build])
        command('native-build', ['cmake', '--build', build, '--config', 'Release', '--target',
            'test-qwen35-decoder', 'test-qwen35-linear', '-j2'])
        report['native_binary_sha256'] = {name: sha(build / name)
            for name in ('test-qwen35-decoder', 'test-qwen35-linear')}
        report['phase'] = 'retained-candidate-raw-diagnostics'
        save()
        diagnostics = driver.quant_numerical_diagnostics(
            build, candidates[0], reference, args.precision, out, report, save,
            baseline['selected_vision_preset'], scratch)
        if diagnostics.get('raw_complete') is not True:
            raise ValueError('Observed raw diagnostic coverage incomplete; partial evidence retained')
        report['status'] = 'completed quantized raw replay; numerical verdict retained'
        report['phase'] = 'complete'
        save()
    except BaseException as exc:
        report['status'] = 'failed quantized raw replay or preflight'
        report['error'] = str(exc).replace(str(ROOT), '<repo>').replace(str(scratch), '<scratch>')[:400]
        save()
        raise


if __name__ == '__main__':
    main()
