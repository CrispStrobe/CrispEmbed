#!/usr/bin/env python3
"""Pinned, fail-closed OvisOCR3 original vLLM T4 reference launcher.

The reviewed CPU metadata pins below must be filled after the hosted metadata
run and independent review. With sentinels this script exits before packages,
checkpoint access, or GPU work. Kaggle deploys only this code_file.
"""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import venv


OUT = Path('/kaggle/working')
SCRATCH = Path('/tmp/ovisocr3-vllm-reference')
LOGS = OUT / 'ovisocr3-vllm-logs'
VENV = SCRATCH / 'venv'
MODEL = 'ATH-MaaS/OvisOCR3'
REVISION = '5f6ce6932efee9ec10c30746db55bd780214b7fd'
CHECKPOINT_SHA = 'aa2311c6b22b07cf3afe9aac6e8bcb02a2b19dd8671b2a91d0b144071577e600'
CHECKPOINT_BYTES = 1706030528
SOURCE_REPORT_SHA = '8ca73ca6a349b67a7533c36de6c14ac066e1152f2c8d517af849b2655087a331'
SOURCE_EOS = [248044, 248046]
SOURCE_PROMPT_SHA = 'e037cf0013a757e19eae6d4d701f3b90848449fbebe44d1f8fa7d8ca4a5af9a7'
METADATA_RUN = 0
METADATA_HEAD = '0' * 40
METADATA_ARTIFACT_ID = 0
METADATA_ARTIFACT_ZIP_SHA = '0' * 64
METADATA_REPORT_SHA = '0' * 64
PROOF_COMMIT = '0' * 40
PRODUCER_SHA = '0' * 64
WORKER_COMMIT = '0' * 40
WORKER_SHA = '0' * 64
WHEEL_URL = ('https://github.com/vllm-project/vllm/releases/download/v0.22.1/'
             'vllm-0.22.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl')
WHEEL_NAME = 'vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
WHEEL_SHA = '365ee929afd73bb5d146235b65053fa948788ec2ee00a2c3e957d3f43bf2b0cd'
WHEEL_BYTES = 473003163
DRIVER_MIN = (575, 51, 3)
PIP_VERSION = '26.2.2'


def configured():
    """Reject incomplete reviewed pins before service, hardware or packages."""
    if (METADATA_RUN <= 0 or METADATA_ARTIFACT_ID <= 0
            or not all(re.fullmatch('[0-9a-f]{40}', value) and value != '0' * 40
                       for value in (METADATA_HEAD, PROOF_COMMIT, WORKER_COMMIT))
            or not all(re.fullmatch('[0-9a-f]{64}', value) and value != '0' * 64
                       for value in (METADATA_ARTIFACT_ZIP_SHA, METADATA_REPORT_SHA,
                                     PRODUCER_SHA, WORKER_SHA))):
        raise ValueError('Reviewed OvisOCR3 metadata or immutable worker pins are unresolved')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


class StripCrossHostAuth(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (redirected is not None and
                urllib.parse.urlsplit(req.full_url).hostname != urllib.parse.urlsplit(newurl).hostname):
            redirected.remove_header('Authorization')
        return redirected


OPENER = urllib.request.build_opener(StripCrossHostAuth())


def request(url, accept='application/vnd.github+json'):
    return urllib.request.Request(url, headers={'User-Agent': 'CrispEmbed-OvisOCR3-reference',
                                                 'Accept': accept})


def get_json(url):
    with OPENER.open(request(url), timeout=30) as response:
        return json.load(response)


def fetch(url, destination, timeout=300):
    started = time.monotonic()
    last = started
    count = 0
    print('START fetch', destination.name, flush=True)
    with OPENER.open(request(url, 'application/octet-stream'), timeout=timeout) as response:
        with destination.open('wb') as output:
            while block := response.read(1024 * 1024):
                output.write(block)
                count += len(block)
                now = time.monotonic()
                if now - last >= 30:
                    print('HEARTBEAT fetch', destination.name, count, 'bytes', flush=True)
                    last = now
    print('END fetch', destination.name, count, 'bytes',
          int(time.monotonic() - started), 'seconds', flush=True)


def save(report):
    (OUT / 'ovisocr3-vllm-reference.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def run(name, argv, report, timeout=600, env=None):
    log_path = LOGS / (name + '.log')
    started = time.monotonic()
    timed_out = False
    with log_path.open('w') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
        while child.poll() is None:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                timed_out = True
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                break
            try:
                child.wait(timeout=min(30, remaining))
            except subprocess.TimeoutExpired:
                print('HEARTBEAT', name, int(time.monotonic() - started), 'seconds', flush=True)
    record = {'name': name, 'exit_code': child.returncode, 'timed_out': timed_out,
              'wall_seconds': time.monotonic() - started, 'log': log_path.name,
              'log_sha256': digest(log_path)}
    report['commands'].append(record)
    save(report)
    if timed_out or child.returncode:
        raise RuntimeError(name + ' failed; see retained log')
    return log_path


def metadata_proof(report):
    service = get_json('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/runs/'
                       + str(METADATA_RUN))
    if (service['id'] != METADATA_RUN or service['head_sha'] != METADATA_HEAD
            or service['status'] != 'completed' or service['conclusion'] != 'success'
            or service['path'] != '.github/workflows/ovisocr3-metadata.yml'):
        raise ValueError('OvisOCR3 metadata service completion/source differs')
    url = ('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/' + PROOF_COMMIT
           + '/tests/results/qwen35-ocr/ovisocr3-metadata.json')
    wrapper = get_json(url)
    producer = wrapper['producer']
    review = wrapper['independent_review']
    artifact = wrapper['artifact']
    if (wrapper['run'] != METADATA_RUN or wrapper['head_sha'] != METADATA_HEAD
            or wrapper['source_commit'] != METADATA_HEAD
            or artifact['id'] != METADATA_ARTIFACT_ID
            or artifact['zip_sha256'] != METADATA_ARTIFACT_ZIP_SHA
            or wrapper['report_sha256'] != METADATA_REPORT_SHA
            or review['metadata_positive'] is not True
            or review['producer_canonical_sha256'] != PRODUCER_SHA
            or canonical_sha(producer) != PRODUCER_SHA
            or producer['source_commit'] != METADATA_HEAD
            or producer['status'] != 'completed weight-free OvisOCR3 HF metadata/source probe; no model forward or native qualification'
            or producer['source_report_sha256'] != SOURCE_REPORT_SHA
            or producer['source_model'] != MODEL or producer['source_revision'] != REVISION
            or producer['task_prompt_sha256'] != SOURCE_PROMPT_SHA
            or producer['tokenizer_eos'] != 248046
            or producer['effective_stops_source_probe']['all_stop_ids'] != SOURCE_EOS
            or producer['schema']['checkpoint_tensors'] != 473
            or len(producer['inputs']) != 3
            or {row['case'] for row in producer['inputs']} != {'square', 'odd', 'small-control'}):
        raise ValueError('Reviewed OvisOCR3 metadata producer differs')
    report['service_metadata'] = {'run': METADATA_RUN, 'head_sha': METADATA_HEAD,
                                  'path': service['path'], 'conclusion': service['conclusion']}
    report['metadata_proof'] = {'url': url, 'producer_sha256': PRODUCER_SHA,
                                'artifact_id': METADATA_ARTIFACT_ID,
                                'artifact_zip_sha256': METADATA_ARTIFACT_ZIP_SHA}
    save(report)
    return wrapper


def gpu_gate(report):
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,compute_cap,driver_version,memory.total',
         '--format=csv,noheader,nounits'], text=True, timeout=15).strip().splitlines()
    if len(rows) != 1:
        raise ValueError('Expected exactly one visible T4; no device redraw')
    parts = [part.strip() for part in rows[0].split(',')]
    if len(parts) != 4:
        raise ValueError('GPU identity unavailable')
    name, capability, version, memory = parts
    match = re.fullmatch(r'(\d+)\.(\d+)', capability)
    driver = tuple(int(x) for x in version.split('.')[:3])
    driver += (0,) * (3 - len(driver))
    report['hardware'] = {'name': name, 'capability': capability,
                          'driver': version, 'memory_mib': int(memory)}
    save(report)
    if not match or 'T4' not in name or tuple(map(int, match.groups())) != (7, 5):
        raise ValueError('Selected GPU is not pinned T4; no fallback')
    if driver < DRIVER_MIN:
        raise ValueError('Driver below CUDA 12.9 GA gate; no fallback')


def cuda_link_environment(report, env):
    driver = ctypes.CDLL('libcuda.so.1')
    version = ctypes.c_int()
    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDriverGetVersion.restype = ctypes.c_int
    if driver.cuDriverGetVersion(ctypes.byref(version)) != 0 or version.value < 12090:
        raise ValueError('Real CUDA driver API unavailable or below 12.9')
    candidates = []
    for line in Path('/proc/self/maps').read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and Path(parts[5]).name.startswith('libcuda.so'):
            path = Path(parts[5]).resolve(strict=True)
            if 'stubs' not in path.parts and path.is_file():
                candidates.append(path)
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError('Expected one real CUDA driver library')
    directory = SCRATCH / 'cuda-driver-link'
    directory.mkdir(exist_ok=True)
    link = directory / 'libcuda.so'
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise ValueError('Unexpected task link file')
    link.symlink_to(candidates[0])
    result = dict(env)
    result['LIBRARY_PATH'] = str(directory) + (
        ':' + result['LIBRARY_PATH'] if result.get('LIBRARY_PATH') else '')
    report['cuda_driver_link'] = {'api_version': version.value,
                                  'driver_sha256': digest(candidates[0]),
                                  'system_files_modified': False}
    save(report)
    return result


def isolated_environment(report):
    venv.EnvBuilder(with_pip=False, clear=True).create(VENV)
    python = str(VENV / 'bin/python')
    run('bootstrap-isolated-pip', [sys.executable, '-m', 'pip', '--python', python,
        'install', 'pip==' + PIP_VERSION], report, timeout=300)
    return python


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    report = {'recipe': 'ovisocr3-controlled-vllm-t4-triton-attn-v1',
              'status': 'started', 'phase': 'reviewed-pins',
              'scope': 'Original OvisOCR3 plain OCR; controlled TRITON_ATTN T4 FP16; no auto-backend/BF16/native/quality claim',
              'pins': {'model': MODEL, 'revision': REVISION,
                       'checkpoint_sha256': CHECKPOINT_SHA, 'checkpoint_bytes': CHECKPOINT_BYTES,
                       'source_report_sha256': SOURCE_REPORT_SHA,
                       'vllm': '0.22.1', 'wheel_sha256': WHEEL_SHA,
                       'torch': '2.11.0', 'transformers': '5.16.1', 'tokenizers': '0.23.3'},
              'commands': [],
              'limits': ['README max_tokens=16384 is explicitly bounded to 256.',
                         'Original Triton GDN retained; decoder attention explicitly TRITON_ATTN after OvisOCR2 T4 auto-FlashInfer failure.',
                         'Torch 2.11 GPU processor output is checked against reviewed Torch 2.14 CPU metadata; mismatch fails before checkpoint.']}
    save(report)
    try:
        configured()
        report['script_sha256'] = digest(__file__)
        report['phase'] = 'metadata-proof'; save(report)
        metadata_proof(report)
        report['phase'] = 'hardware'; save(report)
        gpu_gate(report)
        driver_env = cuda_link_environment(report, os.environ)
        report['phase'] = 'pinned-worker'; save(report)
        worker = SCRATCH / 'worker.py'
        worker_url = ('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/' + WORKER_COMMIT
                      + '/tools/kaggle/ovisocr3-vllm-reference/worker.py')
        fetch(worker_url, worker, timeout=60)
        if digest(worker) != WORKER_SHA:
            raise ValueError('Pinned worker source bytes differ')
        report['worker_sha256'] = WORKER_SHA; save(report)
        report['phase'] = 'isolated-packages'; save(report)
        python = isolated_environment(report)
        wheel = SCRATCH / WHEEL_NAME
        fetch(WHEEL_URL, wheel, timeout=900)
        if wheel.stat().st_size != WHEEL_BYTES or digest(wheel) != WHEEL_SHA:
            raise ValueError('Pinned vLLM CUDA 12.9 wheel differs')
        report['wheel_sha256'] = digest(wheel); save(report)
        install_report = SCRATCH / 'pip-install-report.json'
        run('install-pinned-stack', [python, '-m', 'pip', 'install', '--report', str(install_report),
             str(wheel), 'torch==2.11.0', 'torchvision==0.26.0', 'transformers==5.16.1',
             'tokenizers==0.23.3', 'numpy==2.2.6', 'pillow==12.1.0',
             'huggingface-hub==1.33.0', '--extra-index-url',
             'https://download.pytorch.org/whl/cu129'], report, timeout=1800)
        installs = json.loads(install_report.read_text())['install']
        report['package_archive_sha256'] = {
            entry['metadata']['name']: {'version': entry['metadata']['version'],
                'sha256': entry.get('download_info', {}).get('archive_info', {}).get('hashes', {}).get('sha256')}
            for entry in installs}
        report['package_install_report_sha256'] = digest(install_report); save(report)
        freeze = run('package-freeze', [python, '-m', 'pip', 'freeze', '--all'], report, timeout=120)
        report['package_freeze_sha256'] = digest(freeze); save(report)
        report['phase'] = 'original-gpu-reference'; save(report)
        env = dict(driver_env, CUDA_VISIBLE_DEVICES='0', OVIS_OUT=str(OUT),
                   OVIS_PROOF_COMMIT=PROOF_COMMIT, OVIS_PRODUCER_SHA=PRODUCER_SHA,
                   OVIS_CHECKPOINT_SHA=CHECKPOINT_SHA, OVIS_CHECKPOINT_BYTES=str(CHECKPOINT_BYTES))
        run('original-vllm-worker', [python, str(worker)], report, env=env, timeout=3600)
        child = json.loads((OUT / 'ovisocr3-vllm-worker.json').read_text())
        report['worker'] = child
        if child['status'] != 'completed bounded original OvisOCR3 vLLM reference':
            raise ValueError('Original OvisOCR3 worker did not complete')
        report['status'] = 'completed bounded original OvisOCR3 vLLM T4 reference'
        report['phase'] = 'complete'; save(report)
    except BaseException as error:
        report['status'] = 'inconclusive' if report['phase'] in ('reviewed-pins', 'hardware') else 'failed'
        report['error_type'] = type(error).__name__
        report['error_message'] = str(error)[:300]
        save(report)
        raise


if __name__ == '__main__':
    main()
