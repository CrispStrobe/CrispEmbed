"""Hosted end-user OvisOCR2 alias audit after independent publication review.

The preflight is stdlib only. It rejects absent or mismatched receipts, registry
entries, and SHA pins before build, remote access, or model access.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / 'tests/results/qwen35-ocr'
PUBLICATION = EVIDENCE / 'ovisocr2-publication.json'
PRESERVATION = EVIDENCE / 'ovisocr2-vllm-reference-preservation.json'
REGISTRY = ROOT / 'examples/cli/model_mgr.cpp'
HASHES = ROOT / 'examples/cli/model_hashes.h'
GATES = ('CRISPEMBED_QWEN35_OCR', 'CRISPEMBED_OVISOCR2_FRONTEND',
         'CRISPEMBED_OVISOCR2_INFERENCE')
BASELINE_ZERO = ('CRISPEMBED_QWEN35_PRECISE_RMS', 'CRISPEMBED_QWEN35_IMATRIX',
                 'CRISPEMBED_QWEN35_PARALLEL_F16', 'CRISPEMBED_QWEN35_PROFILE',
                 'CRISPEMBED_QWEN35_TOKENIZER_MARKS', 'CRISPEMBED_QWEN35_AUDIT_ALL_LAYERS')
MAX_TOKENS = 12
FROZEN_ARCHIVE_REVISION = '602a36454331cc9a02ed942402fcfa3a06affaa5'
FROZEN_ARCHIVE_SHA = 'cf45bfd5d9c0dd995391ff6501372a600501bf7eac23e5c81f4fc51844082ac7'
FROZEN_ARCHIVE_BYTES = 500157
FROZEN_IMAGE_SHA = 'fba91697e616a1308653ec37c9e0d56bbec20f0c202a4d6098c31c25c92b9d0a'
FROZEN_IDS = [9419, 77796, 271, 16, 17, 18, 19, 20, 248046]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def hex_field(value, n):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{%d}' % n, value) is not None


def module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def registry_entry(registry, hashes, alias, filename, url, expected_sha):
    """Require one literal C++ registry row and one generated URL pin."""
    # clang-format may wrap an immutable URL into adjacent C++ string literals.
    registry = re.sub(r'"\s*"', '', registry)
    hashes = re.sub(r'"\s*"', '', hashes)
    row = re.compile(r'^\s*\{\s*"' + re.escape(alias)
                     + r'"\s*,\s*"([^"\n]+)"\s*,\s*"([^"\n]+)"\s*,', re.MULTILINE)
    rows = row.findall(registry)
    if rows != [(filename, url)]:
        raise ValueError('Exact published alias/filename/immutable URL is absent or ambiguous')
    pin = re.compile(r'^\s*\{\s*"' + re.escape(url)
                     + r'"\s*,\s*"([0-9a-f]{64})"\s*\}', re.MULTILINE)
    pins = pin.findall(hashes)
    if pins != [expected_sha]:
        raise ValueError('Generated registry SHA pin absent, ambiguous, or differs')
    return {'alias': alias, 'filename': filename, 'url': url, 'sha256': expected_sha}


def require_publication(receipt, pins, registry, hashes):
    """Check reviewed publisher output against current independently gated pins."""
    review = receipt.get('independent_review') if isinstance(receipt, dict) else None
    if not isinstance(review, dict) or review.get('publication_positive') is not True:
        raise ValueError('Independently positive publication receipt required')
    producer = receipt.get('producer')
    if not isinstance(producer, dict):
        raise ValueError('Original publisher producer required')
    precision = pins['precision']
    filename = f'ovisocr2-{precision}.gguf'
    commit = producer.get('published_commit')
    remote = producer.get('remote', {})
    small = remote.get('small_files_sha256') if isinstance(remote, dict) else None
    if not hex_field(commit, 40):
        raise ValueError('Immutable published commit is missing')
    url = f'https://huggingface.co/{pins["destination"]}/resolve/{commit}/{filename}'
    if (producer.get('status') != 'published and immutable remote file reverified'
            or not hex_field(producer.get('source_commit'), 40)
            or producer.get('qualification') != pins
            or producer.get('destination') != pins['destination']
            or not isinstance(remote, dict) or remote.get('repo') != pins['destination']
            or remote.get('commit') != commit or remote.get('filename') != filename
            or remote.get('sha256') != pins['candidate_sha256']
            or type(remote.get('bytes')) is not int or remote['bytes'] != pins['candidate_bytes']
            or not isinstance(small, dict) or not hex_field(small.get('manifest.json'), 64)):
        raise ValueError('Published candidate, destination, qualification, or readback differs')
    if pins['selected_vision_preset'].get('name') != 'both':
        raise ValueError('Consumer requires the independently selected both vision preset')
    entry = registry_entry(registry, hashes, f'ovisocr2-{precision}', filename,
                           url, pins['candidate_sha256'])
    return {'precision': precision, 'commit': commit, 'remote': remote, 'registry': entry,
            'qualification': pins}


def require_frozen_image(preservation, model_receipt):
    image = preservation.get('source', {}).get('files', {}).get('owned-readable.png', {})
    model = model_receipt.get('producer', {})
    frozen = model.get('frozen_reference', {}).get('files', {}).get('owned-readable.png', {})
    readable = model.get('readable', {})
    expected = model.get('original_capture', {})
    ids = expected.get('generated_ids', [])
    expected_url = (f'https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/resolve/'
                    f'{FROZEN_ARCHIVE_REVISION}/qwen35-ovisocr2/'
                    'controlled-vllm-t4-5fe3fc90/reference.zip')
    if (preservation.get('anonymous_immutable_readback_verified') is not True
            or preservation.get('repo') != 'cstr/crispembed-regression-fixtures'
            or preservation.get('revision') != FROZEN_ARCHIVE_REVISION
            or preservation.get('sha256') != FROZEN_ARCHIVE_SHA
            or preservation.get('bytes') != FROZEN_ARCHIVE_BYTES
            or preservation.get('url') != expected_url
            or image != frozen or image.get('sha256') != readable.get('png_sha256')
            or image.get('sha256') != FROZEN_IMAGE_SHA or image.get('bytes') != 5247
            or not hex_field(readable.get('raw_rgb_sha256'), 64)
            or ids != [FROZEN_IDS]
            or expected.get('generation_result', {}).get('max_new_tokens') != MAX_TOKENS
            or expected.get('generation_result', {}).get('stop_id') != 248046
            or expected.get('decoded') != ['Hello OCR\n\n12345']):
        raise ValueError('Frozen public PNG or nine-ID native proof differs')
    return {'url': preservation['url'], 'archive_sha256': preservation['sha256'],
            'archive_bytes': preservation['bytes'], 'png_sha256': image['sha256'],
            'png_bytes': image['bytes'], 'rgb_sha256': readable['raw_rgb_sha256'],
            'expected_text': expected['decoded'][0], 'native_ids': ids[0]}


def preflight(precision):
    # Crucially, this call remains before any packages, auth, build, or weights.
    publisher = module('ovis_publish_consumer', 'tools/ci-heavy/ovisocr2_publish.py')
    pins = publisher.require_receipts(precision)
    if not PUBLICATION.is_file():
        raise ValueError('Reviewed immutable publication receipt absent')
    receipt = json.loads(PUBLICATION.read_text())
    publication = require_publication(receipt, pins, REGISTRY.read_text(), HASHES.read_text())
    publication['receipt_sha256'] = sha(PUBLICATION)
    model_receipt = json.loads((EVIDENCE / 'ovisocr2-model-audit.json').read_text())
    preservation = json.loads(PRESERVATION.read_text())
    fixture = require_frozen_image(preservation, model_receipt)
    quality_path = EVIDENCE / f'ovisocr2-{precision}-quality.json'
    if sha(quality_path) != pins['receipt_sha256']['quality']:
        raise ValueError('Selected quality receipt SHA differs from publisher gate')
    quality = json.loads(quality_path.read_text())
    reference = require_quality_confidence(quality, pins, fixture)
    return publication, fixture, reference


def require_quality_confidence(receipt, pins, fixture):
    """Bind the selected candidate's reviewed mean to the exact frozen PNG."""
    if not isinstance(receipt, dict):
        raise ValueError('Selected positive quality receipt absent')
    producer = receipt.get('producer')
    review = receipt.get('independent_review')
    if not isinstance(producer, dict) or not isinstance(review, dict):
        raise ValueError('Selected quality producer/review absent')
    precision = pins['precision']
    case = producer.get('native_cases', {}).get(precision, {}).get('dev-hello', {})
    smoke = producer.get('cabi_smoke', {}).get(precision, {})
    mean = case.get('mean_confidence')
    smoke_mean = smoke.get('mean_confidence')
    if (review.get('quality_positive') is not True
            or producer.get('models', {}).get(precision, {}).get('sha256') != pins['candidate_sha256']
            or case.get('input_png_sha256') != fixture['png_sha256']
            or case.get('input_rgb_sha256') != fixture['rgb_sha256']
            or case.get('raw_text') != fixture['expected_text']
            or case.get('stop_id') != 248046
            or case.get('finish_reason') != 'stop'
            or case.get('token_ids') != FROZEN_IDS
            or not isinstance(case.get('cap'), int) or case['cap'] < len(FROZEN_IDS)
            or not isinstance(case.get('confidences'), list)
            or len(case['confidences']) != len(FROZEN_IDS)
            or smoke.get('case') != 'dev-hello'
            or smoke.get('raw_text') != fixture['expected_text']
            or type(mean) not in (int, float) or not math.isfinite(mean)
            or type(smoke_mean) not in (int, float) or not math.isfinite(smoke_mean)
            or not 0 <= mean <= 1 or abs(mean - smoke_mean) > 1e-5):
        raise ValueError('Selected quality dev-hello native/C ABI confidence or exact image differs')
    return {'mean_confidence': mean, 'c_abi_mean_confidence': smoke_mean,
            'case': 'dev-hello', 'input_png_sha256': fixture['png_sha256'],
            'input_rgb_sha256': fixture['rgb_sha256'], 'token_ids': FROZEN_IDS,
            'quality_receipt_sha256': pins['receipt_sha256']['quality']}


def bounded_get(url, limit, opener=urllib.request.urlopen):
    with opener(url, timeout=90) as response:
        result = response.read(limit + 1)
    if len(result) > limit:
        raise ValueError('Small public provenance object exceeds bound')
    return result


def fetch_small_provenance(publication, fixture, scratch):
    remote = publication['remote']
    base = publication['registry']['url'].rsplit('/', 1)[0]
    manifest = bounded_get(base + '/manifest.json', 1_000_000)
    if hashlib.sha256(manifest).hexdigest() != remote['small_files_sha256']['manifest.json']:
        raise ValueError('Published immutable manifest SHA differs')
    parsed = json.loads(manifest)
    candidate = parsed.get('candidate', {})
    pins = publication['qualification']
    if (candidate != {'filename': publication['registry']['filename'],
                       'sha256': pins['candidate_sha256'], 'bytes': pins['candidate_bytes'],
                       'precision': pins['precision'], 'recipe': pins['recipe']}
            or parsed.get('selected_vision_preset') != pins['selected_vision_preset']
            or parsed.get('receipts') != pins['receipt_sha256']):
        raise ValueError('Published immutable manifest qualification differs')
    payload = bounded_get(fixture['url'], 1_000_000)
    if (len(payload) != fixture['archive_bytes']
            or hashlib.sha256(payload).hexdigest() != fixture['archive_sha256']):
        raise ValueError('Preserved public GPU fixture archive differs')
    archive = scratch / 'frozen-fixture.zip'
    archive.write_bytes(payload)
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)) or names.count('owned-readable.png') != 1:
            raise ValueError('Preserved PNG archive member absent or ambiguous')
        info = source.getinfo('owned-readable.png')
        if info.file_size != fixture['png_bytes']:
            raise ValueError('Preserved PNG member size differs')
        png = source.read(info)
    if hashlib.sha256(png).hexdigest() != fixture['png_sha256']:
        raise ValueError('Preserved PNG member SHA differs')
    image = scratch / 'owned-readable.png'
    image.write_bytes(png)
    archive.unlink()
    return image, {'manifest_sha256': hashlib.sha256(manifest).hexdigest(),
                   'fixture_archive_sha256': fixture['archive_sha256'],
                   'png_sha256': fixture['png_sha256']}


def env_for(cache, selection, precision):
    vision = module('ovis_vision_consumer', 'tools/ci-heavy/ovisocr2_vision_env.py')
    env = vision.apply_selection(os.environ, selection)
    env.update({gate: '1' for gate in GATES})
    env.update({gate: '0' for gate in BASELINE_ZERO})
    env.update(CRISPEMBED_CACHE_DIR=str(cache), OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               CRISPEMBED_ALLOW_UNPINNED_MODEL='0', HF_HUB_OFFLINE='0',
               CRISPEMBED_OFFLINE='0',
               CRISPEMBED_QWEN35_QUANT='1' if precision != 'f16' else '0')
    return env


def cpu_record():
    model = None
    info = Path('/proc/cpuinfo')
    if info.is_file():
        for line in info.read_text().splitlines():
            if line.startswith('model name'):
                model = line.split(':', 1)[1].strip()
                break
    return {'machine': platform.machine(), 'cpu_model': model,
            'logical_count': os.cpu_count(),
            'affinity_count': len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'threads': {'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}}


def capacity_record(scratch, candidate_bytes):
    usage = shutil.disk_usage(scratch.parent)
    memory = {}
    info = Path('/proc/meminfo')
    if info.is_file():
        for line in info.read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ('MemAvailable', 'SwapFree'):
                memory[key + '_kib'] = int(value.strip().split()[0])
    minimum = candidate_bytes * 2 + 4_000_000_000
    if usage.free < minimum:
        raise ValueError('Hosted disk cannot hold two fresh model caches and portable build')
    return {'disk_free_bytes_before_build': usage.free, 'required_free_bytes': minimum,
            'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None,
            **memory}


def run(label, command, out, env=None, timeout=2400):
    path = out / (label + '.log')
    start = time.monotonic()
    with path.open('wb') as log:
        try:
            completed = subprocess.run([str(value) for value in command], cwd=ROOT, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout,
                                       check=False)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
    return {'exit_code': code, 'seconds': round(time.monotonic() - start, 3),
            'log': path.name, 'log_sha256': sha(path), 'log_bytes': path.stat().st_size,
            'peak_child_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss}


def parse_cli(log, expected):
    records = [json.loads(line) for line in Path(log).read_text(errors='replace').splitlines()
               if line.startswith('{')]
    if len(records) != 1:
        raise ValueError('Exactly one CLI JSON OCR result required')
    value = records[0]
    confidence = value.get('mean_confidence')
    if (value.get('full_text') != expected or value.get('n_regions') != 1
            or type(confidence) not in (int, float) or not math.isfinite(confidence)
            or not 0 <= confidence <= 1):
        raise ValueError('CLI decoded text, region count, or confidence differs')
    return {'text': value['full_text'], 'n_regions': 1, 'mean_confidence': confidence,
            'max_new_tokens_argument': MAX_TOKENS,
            'token_ids_exposed_by_public_api': False}


def check_cached(path, remote):
    if (not path.is_file() or path.stat().st_size != remote['bytes']
            or sha(path) != remote['sha256']):
        raise ValueError('Downloaded registry cache SHA/size differs from remote readback')
    return {'sha256': remote['sha256'], 'bytes': remote['bytes']}


def compare_public_answers(online, offline, http, reference):
    mean = reference['mean_confidence']
    values = (mean, online['mean_confidence'], offline['mean_confidence'], http['mean_confidence'])
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in values):
        raise ValueError('Public confidence mean missing, nonfinite, or out of range')
    cli_error = abs(online['mean_confidence'] - mean)
    offline_error = abs(offline['mean_confidence'] - mean)
    http_error = abs(http['mean_confidence'] - mean)
    if (online['text'] != offline['text'] or online['text'] != http['text']
            or online['n_regions'] != offline['n_regions'] or online['n_regions'] != http['n_regions']
            or online['mean_confidence'] != offline['mean_confidence']
            or cli_error > .0006 or offline_error > .0006 or http_error > .00006):
        raise ValueError('CLI fresh/offline or HTTP decoded output/confidence differs')
    return {'fresh_offline_confidence_exact': True,
            'reference_mean_confidence': mean,
            'cli_mean_confidence_abs': cli_error,
            'offline_mean_confidence_abs': offline_error,
            'http_mean_confidence_abs': http_error,
            'cli_budget': .0006, 'http_budget': .00006}


def http_arm(build, alias, image, expected, cache, selection, precision, out):
    env = env_for(cache, selection, precision)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    log_path = out / 'http-alias.log'
    start = time.monotonic()
    with log_path.open('wb') as log:
        proc = subprocess.Popen([str(build / 'crispembed-server'), '--ocr-pipeline',
                                 '--ocr-engine', 'ovisocr2', '--ocr-rec', alias,
                                 '--ocr-max-tokens', str(MAX_TOKENS), '--host', '127.0.0.1',
                                 '--port', str(port), '--image-root', str(image.parent)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 900
            while True:
                if proc.poll() is not None:
                    raise ValueError('HTTP alias service exited before readiness')
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2):
                        break
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise ValueError('HTTP alias service readiness timeout')
                    time.sleep(.3)
            request = urllib.request.Request(f'http://127.0.0.1:{port}/ocr/pipeline',
                                             json.dumps({'image': str(image)}).encode(),
                                             {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=600) as response:
                answer = json.load(response)
            confidence = answer.get('mean_confidence')
            if (answer.get('text') != expected or answer.get('n_regions') != 1
                    or type(confidence) not in (int, float) or not math.isfinite(confidence)
                    or not 0 <= confidence <= 1):
                raise ValueError('HTTP alias decoded text, regions, or confidence differ')
            return {'text': answer['text'], 'n_regions': 1, 'mean_confidence': confidence,
                    'max_new_tokens_argument': MAX_TOKENS,
                    'token_ids_exposed_by_public_api': False,
                    'seconds': round(time.monotonic() - start, 3)}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--precision', required=True, choices=('f16', 'q8', 'q4', 'q4-calibrated'))
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    out = Path(os.environ['HEAVY_OUT'])
    scratch = Path(os.environ['HEAVY_SCRATCH']) / 'ovis-consumer'
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / 'ovisocr2-published-consumer.json'
    report = {'status': 'started', 'phase': 'preflight', 'precision': args.precision,
              'source_commit': os.environ.get('GITHUB_SHA'), 'cpu': cpu_record(),
              'source_sha256': {name: sha(ROOT / name) for name in (
                  'tools/ci-heavy/ovisocr2_published_consumer.py',
                  'tools/ci-heavy/ovisocr2_publish.py',
                  '.github/workflows/ovisocr2-published-consumer.yml',
                  'examples/cli/model_mgr.cpp', 'examples/cli/model_hashes.h')},
              'arms': {}, 'limits': ['Public CLI/HTTP APIs expose text/confidence but no token IDs or stop reason; '
                                     'nine generated IDs and EOS are separately frozen in the reviewed native receipt.',
                                     'This checks one frozen 448x448 readable image and a 12-token request cap.',
                                     'CLI and HTTP means use existing public printing budgets .0006 and .00006 '
                                     'against the selected quality dev-hello native mean on the byte-identical PNG/RGB.']}
    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    try:
        publication, fixture, reference = preflight(args.precision)
        report['publication'] = {'commit': publication['commit'],
                                 'registry': publication['registry'],
                                 'receipt_sha256': publication['receipt_sha256'],
                                 'candidate_bytes': publication['remote']['bytes'],
                                 'selected_vision_preset': publication['qualification']['selected_vision_preset']}
        report['native_nine_ids'] = fixture['native_ids']
        report['selected_quality_confidence_reference'] = reference
        report['phase'] = 'preflight-complete'
        save()
        if args.preflight:
            report['status'] = 'preflight passed; no build, model access, or remote access'
            save()
            return
        report['capacity'] = capacity_record(scratch, publication['remote']['bytes'])
        save()
        scratch.mkdir(parents=True, exist_ok=True)
        image, provenance = fetch_small_provenance(publication, fixture, scratch)
        report['small_provenance'] = provenance
        report['phase'] = 'portable-build'
        save()
        build = scratch / 'build'
        for label, command in (
                ('cmake-configure', ['cmake', '-S', ROOT, '-B', build, '-DCMAKE_BUILD_TYPE=Release',
                                     '-DCRISPEMBED_NATIVE=OFF', '-DGGML_NATIVE=OFF',
                                     '-DGGML_BLAS=OFF', '-DGGML_METAL=OFF']),
                ('cpu-baseline-check', [sys.executable, ROOT / 'scripts/check-cpu-baseline.py', build]),
                ('consumer-target-build', ['cmake', '--build', build, '--config', 'Release',
                                           '--target', 'crispembed-cli', 'crispembed-server', '-j2'])):
            result = run(label, command, out)
            report['arms'][label] = result
            save()
            if result['exit_code'] != 0 or result['log_bytes'] == 0:
                raise ValueError(label + ' failed or returned empty diagnostics')
        alias = publication['registry']['alias']
        filename = publication['registry']['filename']
        selection = publication['qualification']['selected_vision_preset']
        cli_cache, http_cache = scratch / 'cli-cache', scratch / 'http-cache'
        if cli_cache.exists() or http_cache.exists():
            raise ValueError('Two genuinely empty model caches required')
        env = env_for(cli_cache, selection, args.precision)
        cli = build / 'crispembed'
        args_cli = [cli, '--ocr-pipeline', image, '--ocr-engine', 'ovisocr2',
                    '--ocr-rec', alias, '--ocr-max-tokens', str(MAX_TOKENS), '--json', '-t', '2']
        report['phase'] = 'online-cli-alias'
        save()
        first = run('cli-online-alias', args_cli, out, env=env, timeout=1200)
        report['arms']['cli-online-alias'] = first
        save()
        if first['exit_code'] != 0:
            raise ValueError('Fresh CLI registry alias download/inference failed')
        first['result'] = parse_cli(out / first['log'], fixture['expected_text'])
        first['cache'] = check_cached(cli_cache / filename, publication['remote'])
        save()
        env['CRISPEMBED_OFFLINE'] = '1'
        repeat = run('cli-offline-cached', [*args_cli, '--offline'], out, env=env, timeout=600)
        report['arms']['cli-offline-cached'] = repeat
        save()
        if repeat['exit_code'] != 0:
            raise ValueError('Offline cached CLI repeat failed')
        repeat['result'] = parse_cli(out / repeat['log'], fixture['expected_text'])
        repeat['cache'] = check_cached(cli_cache / filename, publication['remote'])
        save()
        report['phase'] = 'online-http-separate-cache'
        save()
        try:
            report['arms']['http-online-alias'] = http_arm(build, alias, image,
                fixture['expected_text'], http_cache, selection, args.precision, out)
        finally:
            http_log = out / 'http-alias.log'
            if http_log.exists():
                report['arms'].setdefault('http-online-alias', {}).update(
                    log=http_log.name, log_sha256=sha(http_log), log_bytes=http_log.stat().st_size)
            save()
        report['arms']['http-online-alias']['cache'] = check_cached(http_cache / filename,
                                                                    publication['remote'])
        report['public_comparison'] = compare_public_answers(first['result'], repeat['result'],
                                                              report['arms']['http-online-alias'], reference)
        report['status'] = 'completed published consumer alias audit'
        report['phase'] = 'complete'
        save()
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise


if __name__ == '__main__':
    main()
