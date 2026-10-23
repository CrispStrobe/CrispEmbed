#!/usr/bin/env python3
"""One pinned original OvisOCR2 vLLM reference on a qualified T4.

The original README uses dtype=auto. On T4 vLLM resolves that to FP16; this
receipt is deliberately not a BF16 reference or a task-quality benchmark.
"""

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.parse
import venv
import zipfile


HERE = Path(__file__).resolve().parent
OUT = Path('/kaggle/working')
LOGS = OUT / 'ovisocr2-vllm-logs'
SCRATCH = Path('/tmp/ovisocr2-vllm-reference')
VENV = SCRATCH / 'venv'
METADATA_RUN = 39983101540
ORIGINAL_ARTIFACT_ID = 12515557390
METADATA_HEAD = 'ca3d096fe673f3554e95ef8e5f534b597564233e'
FRONTEND_RUN = 40017014043
FRONTEND_HEAD = 'ccb6059a6b07b1339abc4de7dd085b0f08af8493'
PROOF_COMMIT = '8f13e82fe592978a6dd47d812163842019b47eb2'
PRODUCER_SHA = '233dc37fdaa2ff6b4a8b3ca3832b578305b5f3b6ed4d0d88d54c73591626b372'
MODEL = 'ATH-MaaS/OvisOCR2'
REVISION = 'cbdede82e048659b83310476ff807da825f8fe78'
CHECKPOINT_SHA = '9270560288656ece5cb3a6989001afcf5af8d223bceed4a423c33a008861d009'
CHECKPOINT_BYTES = 1706030496
VLLM_COMMIT = '0decac0d96c42b49572498019f0a0e3600f50398'
WHEEL_NAME = 'vllm-0.22.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl'
WHEEL_URL = 'https://github.com/vllm-project/vllm/releases/download/v0.22.1/vllm-0.22.1%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl'
WHEEL_SHA = '365ee929afd73bb5d146235b65053fa948788ec2ee00a2c3e957d3f43bf2b0cd'
WHEEL_BYTES = 473003163
MIRROR_REPO = 'cstr/crispembed-regression-fixtures'
MIRROR_REVISION = '50b63a86f2d706491c3cfb953da28b143324fa48'
MIRROR_URL = (f'https://huggingface.co/datasets/{MIRROR_REPO}/resolve/{MIRROR_REVISION}'
              '/qwen35-ovisocr2/original-processor/metadata.zip')
MIRROR_BYTES = 524058
MIRROR_SHA = 'e3ddf289cbac24d48d4cb4818d7cc17a1ed46be3dd2afd361c6ad747d2c7dde5'
WORKER_COMMIT = '2dc537902e72a5f706850cd75afc3dc14e498295'
WORKER_SHA = 'dcc731b512e949b3a82dee4e6706a4560b8efb704ec664d5615d0175f4f3c2af'
WORKER_URL = (f'https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/{WORKER_COMMIT}'
              '/tools/kaggle/ovisocr2-vllm-reference/worker.py')
DRIVER_MIN = (575, 51, 3)  # NVIDIA CUDA 12.9 GA corresponding Linux driver.
SOURCES = {
    'readme': f'https://huggingface.co/{MODEL}/blob/{REVISION}/README.md',
    'vllm_release': 'https://github.com/vllm-project/vllm/releases/tag/v0.22.1',
    'driver': 'https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html',
    'gpu': 'https://docs.vllm.ai/en/v0.22.1/getting_started/installation/gpu/',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                         allow_nan=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def request(url, auth='', accept='application/vnd.github+json'):
    headers = {'User-Agent': 'CrispEmbed-OvisOCR2-reference', 'Accept': accept}
    if auth:
        headers['Authorization'] = 'Bearer ' + auth
    return urllib.request.Request(url, headers=headers)


class StripCrossHostAuth(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if (redirected is not None and
                urllib.parse.urlsplit(req.full_url).hostname != urllib.parse.urlsplit(newurl).hostname):
            redirected.remove_header('Authorization')
        return redirected


OPENER = urllib.request.build_opener(StripCrossHostAuth())


def get_json(url, auth=''):
    with OPENER.open(request(url, auth), timeout=30) as response:
        return json.load(response)


def fetch(url, destination, auth='', timeout=300):
    started = time.monotonic()
    last = started
    count = 0
    print('START fetch', destination.name, flush=True)
    with OPENER.open(request(url, auth, 'application/octet-stream'), timeout=timeout) as response:
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


def run(name, argv, report, timeout=600, env=None):
    path = LOGS / (name + '.log')
    start = time.monotonic()
    timed_out = False
    print('START', name, flush=True)
    with path.open('w') as log:
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
        while child.poll() is None:
            remaining = timeout - (time.monotonic() - start)
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
                print('HEARTBEAT', name, int(time.monotonic() - start), 'seconds', flush=True)
    entry = {'name': name, 'exit_code': child.returncode, 'timed_out': timed_out,
             'timeout_seconds': timeout, 'wall_seconds': time.monotonic() - start,
             'log': path.name, 'log_sha256': digest(path)}
    report['commands'].append(entry)
    save(report)
    print('END', name, 'exit', child.returncode, 'timeout', timed_out, flush=True)
    if timed_out or child.returncode:
        raise RuntimeError(name + ' failed; see retained log')
    return path


def save(report):
    (OUT / 'ovisocr2-vllm-reference.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def service_gate(run_id, expected_head, label):
    data = get_json(f'https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/runs/{run_id}')
    if (data['id'] != run_id or data['head_sha'] != expected_head
            or data['status'] != 'completed' or data['conclusion'] != 'success'
            or data['path'] != '.github/workflows/ovisocr2-metadata.yml'):
        raise ValueError(label + ' has not completed successfully at pinned source')
    return {'run': run_id, 'head_sha': data['head_sha'], 'conclusion': data['conclusion'],
            'workflow_path': data['path']}


def metadata_proof(report):
    base = f'https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/{PROOF_COMMIT}/tests/results/qwen35-ocr/ovisocr2-metadata.json'
    evidence = get_json(base)
    producer = evidence['producer']
    review = evidence['independent_review']
    if (evidence['run'] != METADATA_RUN or evidence['artifact']['id'] != ORIGINAL_ARTIFACT_ID
            or producer['source_commit'] != METADATA_HEAD
            or canonical_sha(producer) != PRODUCER_SHA
            or review['producer_canonical_sha256'] != PRODUCER_SHA
            or evidence['artifact']['zip_sha256'] != review['artifact_zip_sha256']
            or any(review[key] is not True for key in (
                'schema_all_473_shapes_exact', 'missing_only_tied_head',
                'loaded_processor_image_tokenizer_match_upstream_tag',
                'all_three_rgb_id_grid_and_packed_field_hashes_checked',
                'default_counterfactual_fields_checked',
                'image_placeholder_counts_match_merge_geometry'))
            or producer['resolved_generation']['eos_token_id'] != 248044
            or producer['tokenizer_eos'] != 248046
            or producer['effective_stops_source_probe']['all_stop_ids'] != [248044, 248046]
            or producer['schema']['checkpoint_tensors'] != 473):
        raise ValueError('Original metadata producer proof differs')
    declaration = evidence['checkpoint_lfs_declaration']['file']
    if (declaration['lfs']['sha256'] != CHECKPOINT_SHA or declaration['size'] != CHECKPOINT_BYTES
            or declaration['rfilename'] != 'model.safetensors'):
        raise ValueError('Original checkpoint declaration differs')
    report['metadata_proof'] = {'url': base, 'producer_sha256': PRODUCER_SHA,
                                'artifact_id': evidence['artifact']['id']}
    save(report)
    return evidence


def original_arrays(evidence, report):
    artifact_id = evidence['artifact']['id']
    if artifact_id != ORIGINAL_ARTIFACT_ID:
        raise ValueError('Original metadata artifact identity differs')
    archive = SCRATCH / 'original-metadata.zip'
    if evidence['artifact']['zip_sha256'] != MIRROR_SHA:
        raise ValueError('Public mirror identity differs from original ZIP proof')
    fetch(MIRROR_URL, archive)
    expected = evidence['artifact']['zip_sha256']
    if archive.stat().st_size != MIRROR_BYTES or digest(archive) != expected:
        raise ValueError('Original metadata artifact SHA256 differs')
    root = SCRATCH / 'original'
    root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any(Path(n).name != n or n.endswith('/') for n in names):
            raise ValueError('Unexpected original metadata ZIP paths')
        for row in evidence['producer']['inputs']:
            name = row['case'] + '-processor.npz'
            if name not in names:
                raise ValueError('Original processor archive missing: ' + name)
            (root / name).write_bytes(z.read(name))
            if digest(root / name) != row['archive_sha256']:
                raise ValueError('Original processor archive SHA256 differs: ' + name)
    report['original_artifact'] = {'id': artifact_id, 'sha256': expected,
                                   'transport': {'repo': MIRROR_REPO, 'revision': MIRROR_REVISION,
                                                 'url': MIRROR_URL, 'bytes': MIRROR_BYTES},
                                   'files': {p.name: digest(p) for p in root.iterdir()}}
    save(report)
    return root


def gpu_gate(report):
    command = ['nvidia-smi', '--query-gpu=name,compute_cap,driver_version,memory.total',
               '--format=csv,noheader,nounits']
    raw = subprocess.check_output(command, text=True, timeout=15).strip().splitlines()
    report['gpu_query'] = raw
    if not raw:
        raise ValueError('No GPU visible; no device redraw or fallback')
    # A Kaggle session may expose T4 x2. Pin the first visible GPU only;
    # never search for a different device after a failure.
    parts = [part.strip() for part in raw[0].split(',')]
    if len(parts) != 4:
        raise ValueError('Could not identify GPU capability and driver')
    name, capability, version, memory = parts
    match = re.fullmatch(r'(\d+)\.(\d+)', capability)
    driver = tuple(int(x) for x in version.split('.')[:3])
    driver += (0,) * (3 - len(driver))
    report['hardware'] = {'name': name, 'capability': capability, 'driver': version,
                          'memory_mib': int(memory), 'required_driver': '.'.join(map(str, DRIVER_MIN)),
                          'wheel_cuda': '12.9'}
    try:
        nvcc = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=15, check=False)
        report['hardware']['nvcc_version_output'] = nvcc.stdout.strip() if nvcc.returncode == 0 else None
    except FileNotFoundError:
        report['hardware']['nvcc_version_output'] = None
    save(report)
    if not match or 'T4' not in name or (int(match[1]), int(match[2])) != (7, 5):
        raise ValueError('Inconclusive: selected GPU is not the pinned T4; no redraw')
    if driver < DRIVER_MIN:
        raise ValueError('Inconclusive: driver below conservative CUDA 12.9 gate; no minor-compatibility assumption')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    report = {'recipe': 'ovisocr2-original-vllm-t4-v1', 'status': 'started', 'phase': 'frontend-gate',
              'scope': 'Original vLLM auto-FP16 T4 path and bounded greedy prefix; no BF16 parity or task-quality claim',
              'pins': {'model': MODEL, 'revision': REVISION, 'metadata_run': METADATA_RUN,
                       'frontend_run': FRONTEND_RUN, 'proof_commit': PROOF_COMMIT,
                       'vllm_version': '0.22.1', 'vllm_commit': VLLM_COMMIT,
                       'worker_commit': WORKER_COMMIT, 'worker_sha256': WORKER_SHA,
                       'vllm_wheel_sha256': WHEEL_SHA, 'vllm_wheel_bytes': WHEEL_BYTES,
                       'checkpoint_sha256': CHECKPOINT_SHA, 'torch': '2.11.0',
                       'original_mirror_revision': MIRROR_REVISION,
                       'original_mirror_zip_sha256': MIRROR_SHA,
                       'transformers': '5.16.1', 'tokenizers': '0.23.3'},
              'sources': SOURCES, 'commands': [],
              'limits': ['Published vLLM v0.22.1 release has no cu128 wheel; this recipe uses pinned cu129 only after an actual qualified driver check.',
                         'The README leaves dtype=auto; vLLM resolves Ovis BF16 config to FP16 on T4.',
                         'A bounded 256-token output is an explicitly labeled cap versus README max_tokens=16384.']}
    save(report)
    try:
        report['script_sha256'] = digest(__file__)
        report['service_frontend'] = service_gate(FRONTEND_RUN, FRONTEND_HEAD, 'Corrected frontend')
        report['service_metadata'] = service_gate(METADATA_RUN, METADATA_HEAD, 'Original metadata')
        report['phase'] = 'metadata-proof'; save(report)
        evidence = metadata_proof(report)
        report['phase'] = 'hardware'; save(report)
        gpu_gate(report)
        report['phase'] = 'pinned-worker'; save(report)
        worker = SCRATCH / 'worker.py'
        fetch(WORKER_URL, worker, timeout=60)
        if digest(worker) != WORKER_SHA:
            raise ValueError('Pinned original worker source bytes differ')
        report['worker_sha256'] = digest(worker); save(report)
        report['phase'] = 'original-artifact'; save(report)
        original_arrays(evidence, report)
        report['phase'] = 'isolated-packages'; save(report)
        venv.EnvBuilder(with_pip=True, clear=True).create(VENV)
        python = str(VENV / 'bin/python')
        wheel = SCRATCH / WHEEL_NAME
        fetch(WHEEL_URL, wheel, timeout=900)
        if wheel.stat().st_size != WHEEL_BYTES or digest(wheel) != WHEEL_SHA:
            raise ValueError('Published vLLM CUDA 12.9 wheel differs')
        report['downloaded_wheel_sha256'] = digest(wheel); save(report)
        install_report = SCRATCH / 'pip-install-report.json'
        run('install-pinned-stack', [python, '-m', 'pip', 'install', '--report', str(install_report), str(wheel),
             'torch==2.11.0', 'torchvision==0.26.0', 'transformers==5.16.1',
             'tokenizers==0.23.3', 'numpy==2.2.6', 'pillow==12.1.0',
             'huggingface-hub==1.33.0', '--extra-index-url',
             'https://download.pytorch.org/whl/cu129'], report, timeout=1800)
        installs = json.loads(install_report.read_text())['install']
        report['package_archive_sha256'] = {
            entry['metadata']['name']: {
                'version': entry['metadata']['version'],
                'sha256': entry.get('download_info', {}).get('archive_info', {}).get('hashes', {}).get('sha256')}
            for entry in installs
        }
        report['package_install_report_sha256'] = digest(install_report)
        save(report)
        freeze = run('package-freeze', [python, '-m', 'pip', 'freeze', '--all'], report, timeout=120)
        report['package_freeze_sha256'] = digest(freeze)
        report['phase'] = 'original-gpu-reference'; save(report)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OVIS_OUT=str(OUT), OVIS_SCRATCH=str(SCRATCH),
                   OVIS_PROOF_COMMIT=PROOF_COMMIT, OVIS_CHECKPOINT_SHA=CHECKPOINT_SHA)
        run('original-vllm-worker', [python, str(worker)], report, env=env, timeout=3600)
        worker_result = json.loads((OUT / 'ovisocr2-vllm-worker.json').read_text())
        report['worker'] = worker_result
        if worker_result['status'] != 'completed bounded original vLLM reference':
            raise ValueError('Original worker did not complete')
        report['status'] = 'completed bounded original vLLM T4 reference'
        report['phase'] = 'complete'; save(report)
    except BaseException as error:
        report['status'] = 'inconclusive' if report['phase'] in ('frontend-gate', 'hardware') else 'failed'
        report['error_type'] = type(error).__name__
        save(report)
        raise


if __name__ == '__main__':
    main()
