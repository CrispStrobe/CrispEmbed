#!/usr/bin/env python3
"""Pinned original OvisOCR2 GPU quality capture on a qualified T4.

The original README uses dtype=auto. On T4 vLLM resolves that to FP16. This
small frozen panel is separate from BF16, native parity, and public benchmarks.
"""

import hashlib
import ctypes
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
LOGS = OUT / 'ovisocr2-vllm-quality-logs'
SCRATCH = Path('/tmp/ovisocr2-vllm-quality-reference')
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
WORKER_COMMIT = 'e6f3c708cfe24fc8a6e77e0e3b9c27709e74f333'
WORKER_SHA = '0726210027850e96bfd6da2c9a3ff100b5a4992fdf972fab8ca8c1ba44d31794'
WORKER_URL = (f'https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/{WORKER_COMMIT}'
              '/tools/kaggle/ovisocr2-vllm-quality-reference/worker.py')
DRIVER_MIN = (575, 51, 3)  # NVIDIA CUDA 12.9 GA corresponding Linux driver.
ISOLATED_PIP_VERSION = '26.2.2'
SOURCES = {
    'readme': f'https://huggingface.co/{MODEL}/blob/{REVISION}/README.md',
    'vllm_release': 'https://github.com/vllm-project/vllm/releases/tag/v0.22.1',
    'driver': 'https://docs.nvidia.com/cuda/archive/12.9.0/cuda-toolkit-release-notes/index.html',
    'gpu': 'https://docs.vllm.ai/en/v0.22.1/getting_started/installation/gpu/',
    'isolated_pip': 'https://pip.pypa.io/en/stable/topics/python-option/',
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
    (OUT / 'ovisocr2-vllm-quality-reference.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


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



def cuda_link_environment(report, env):
    """Expose the installed real driver to JIT linkers without system changes."""
    # NVIDIA container mounts need not appear in the ldconfig cache. Ask the
    # dynamic loader for the actual driver, then inspect its mapped file.
    driver = ctypes.CDLL('libcuda.so.1')
    version = ctypes.c_int()
    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDriverGetVersion.restype = ctypes.c_int
    if driver.cuDriverGetVersion(ctypes.byref(version)) != 0 or version.value < 12090:
        raise ValueError('Actual CUDA driver API is unavailable or below CUDA 12.9')
    mapped = Path('/proc/self/maps').read_text()
    candidates = []
    for line in mapped.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6 and Path(parts[5]).name.startswith('libcuda.so'):
            path = Path(parts[5]).resolve(strict=True)
            if 'stubs' not in path.parts and path.is_file():
                candidates.append(path)
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise ValueError('Expected one dynamically loaded real CUDA driver library')
    target = candidates[0]
    directory = SCRATCH / 'cuda-driver-link'
    directory.mkdir(exist_ok=True)
    link = directory / 'libcuda.so'
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise ValueError('Unexpected file in task-owned CUDA link directory')
    link.symlink_to(target)
    result = dict(env)
    result['LIBRARY_PATH'] = str(directory) + (
        ':' + result['LIBRARY_PATH'] if result.get('LIBRARY_PATH') else '')
    report['cuda_driver_link'] = {'driver_api_version': version.value, 'target': str(target), 'sha256': digest(target),
                                  'link': str(link), 'system_files_modified': False}
    save(report)
    return result


def isolated_environment(report):
    # Debian's interpreter can create a venv even when its ensurepip is absent.
    # The base pip --python option targets only this task-owned environment.
    venv.EnvBuilder(with_pip=False, clear=True).create(VENV)
    python = str(VENV / 'bin/python')
    run('bootstrap-isolated-pip', [sys.executable, '-m', 'pip', '--python', python,
        'install', 'pip==' + ISOLATED_PIP_VERSION], report, timeout=300)
    report['isolated_pip_version'] = ISOLATED_PIP_VERSION
    save(report)
    return python



QUALITY_SOURCE_COMMIT = 'e50eed66e4608252a0e9ecea916d5baaf91f54ad'
V6_RECEIPT_SHA = 'fbec0f4b75576e93bc32db26e645be0764b23be3a5a6be3e84e04854db1bab4c'
V6_RECEIPT_PATH = 'tests/results/qwen35-ocr/ovisocr2-vllm-reference.json'
SOURCE_FILES = {
    'tests/results/qwen35-ocr/ovisocr2-quality-panel.json': '3c576337645193e4347d833c185e70c81b81e3b986d8b3e8664a4fd6cdd4f8a9',
    'tests/results/ocr-quality-prepared.json': '5b770c6e2f8a3ba307ff92b7e9d3c2c064cf478d90b4506211663d9d831d8d25',
    'tests/regression/ocr-quality/panel.json': 'b7bd7e4b3d84266367d4f96d95aa7be572172a8d883d0e53398d29bc0a761803',
    'tests/regression/ocr-quality/historical-receipt.gold.json': '125c47bc521ded86512c6ba3bc85feb1a9a19a371a64bf8a9e8ccb3453d7705e',
    'tests/regression/ocr-quality/two-column-page.gold.json': 'abf6dfbf6d2461f79dd6911b5e7bd259a56fb0fc6534155bc34eb5fce41044b7',
    'tests/regression/ocr-quality/inventory.gold.json': '801171c1459131e5d34479d6820fcc3dd260c9ea7110b0cbb4a6705e0f729e76',
    'tools/ci-heavy/ovisocr2_quality_contract.py': 'f1609993d77bd038586b620b5c99655aef22837b107fac8e4e9dd36ddebda9fb',
    'tools/ci-heavy/ovisocr2_quality_run.py': '8824305d2408e30c574efdbdfc6f0cc7de7c2a1c678dcacf76a623758def0b82',
    'tools/ci-heavy/qwen35_quant_calibration.py': 'bf03b5a66fd6d42700eaf5e4dcc9a5705ea208df3e903c2187f576b7b45ebed6',
    'tools/ocr_quality.py': 'de93f01d26759e052e4c99c224b3184a39bdf3bbd11f768318c672ccad7fa0c5',
}
# The reviewed preparation archive is mirrored byte-for-byte to a public dataset.
# Fill only after the mirror's immutable revision, artifact identity, and ZIP SHA are reviewed.
QUALITY_MIRROR_URL = ('https://huggingface.co/datasets/cstr/crispembed-regression-fixtures/'
                      'resolve/057e12668bf8109ec83c65929c9b8d6cb99ca566/'
                      'ocr-quality-v1/prepared-38879605915/prepared.zip')
QUALITY_MIRROR_SHA = 'a890c3ad8151706948d3d89529e5f5716405e5f790a3749436f36d78f20ca9cc'
QUALITY_MIRROR_BYTES = 724951
QUALITY_ORIGINAL_ARTIFACT_ID = 12037153992
EXPECTED_PNGS = ('historical-receipt.png', 'two-column-page.png', 'inventory-table.png')


def v6_positive_gate(report):
    path = SCRATCH / 'reviewed-v6-receipt.json'
    fetch('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/'
          + QUALITY_SOURCE_COMMIT + '/' + V6_RECEIPT_PATH, path, timeout=60)
    if digest(path) != V6_RECEIPT_SHA:
        raise ValueError('Reviewed positive original GPU receipt bytes differ')
    receipt = json.loads(path.read_text())
    review = receipt['independent_review']
    worker = receipt['producer']['worker']
    if (receipt['recipe_source'] != '5fe3fc90032259981112bf673e227ec66a07c83e'
            or receipt['worker_source'] != '4b4293d92a4f388315ebf5be4e5cd56f44e31816'
            or review['genuine_gpu_forward_completed'] is not True
            or review['all_three_processor_arrays_rehashed'] is not True
            or review['raw_ids_prompt_text_completion_checked'] is not True
            or review['actual_dtype'] != 'torch.float16'
            or review['decoder_attention_backend'] != 'TRITON_ATTN'
            or review['stops'] != [248044, 248046]
            or review['recipe_sha256'] != '95e0406fc60e121f3c499d8227a50bbe812d0e13418683202f1f133823d5ff4e'
            or review['worker_sha256'] != 'daaade86dbae60b7aecd5ecd9dc1e87a0e49e091361fa00441a437e27f2ee1df'):
        raise ValueError('Reviewed positive original GPU fields differ')
    if (worker['status'] != 'completed bounded original vLLM reference'
            or worker['checkpoint_sha256'] != CHECKPOINT_SHA
            or worker['runtime_stop_probe']['all_stop_ids'] != [248044, 248046]):
        raise ValueError('Original GPU worker proof differs')
    report['reviewed_v6'] = {'source_commit': QUALITY_SOURCE_COMMIT, 'receipt_sha256': V6_RECEIPT_SHA,
                             'recipe_source': receipt['recipe_source'],
                             'worker_source': receipt['worker_source'],
                             'reviewed_genuine_forward': True}
    save(report)


def frozen_sources(report):
    root = SCRATCH / 'source'
    for name, expected in SOURCE_FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        fetch('https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/'
              + QUALITY_SOURCE_COMMIT + '/' + name, path, timeout=60)
        if digest(path) != expected:
            raise ValueError('Frozen quality source differs: ' + name)
    report['quality_source'] = {'commit': QUALITY_SOURCE_COMMIT, 'files_sha256': SOURCE_FILES}
    save(report)
    return root


def frozen_realistic_images(report):
    archive = SCRATCH / 'reviewed-realistic-images.zip'
    fetch(QUALITY_MIRROR_URL, archive, timeout=300)
    if archive.stat().st_size != QUALITY_MIRROR_BYTES or digest(archive) != QUALITY_MIRROR_SHA:
        raise ValueError('Reviewed realistic image artifact bytes differ')
    manifest = json.loads((SCRATCH / 'source/tests/results/ocr-quality-prepared.json').read_text())
    expected = {row['file']: row['png_sha256'] for row in manifest['cases']}
    if tuple(expected) != EXPECTED_PNGS:
        raise ValueError('Reviewed realistic image order differs')
    root = SCRATCH / 'realistic'
    root.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or set(names) != set(EXPECTED_PNGS) | {'prepared.json'}:
            raise ValueError('Reviewed realistic ZIP paths differ')
        report['original_prepared_json_sha256'] = hashlib.sha256(z.read('prepared.json')).hexdigest()
        if report['original_prepared_json_sha256'] != manifest['artifact_report_sha256']:
            raise ValueError('Original prepared report bytes differ')
        for name in EXPECTED_PNGS:
            (root / name).write_bytes(z.read(name))
            if digest(root / name) != expected[name]:
                raise ValueError('Reviewed realistic PNG payload differs: ' + name)
    report['realistic_images'] = {'original_artifact_id': QUALITY_ORIGINAL_ARTIFACT_ID,
                                  'mirror_url': QUALITY_MIRROR_URL, 'zip_sha256': QUALITY_MIRROR_SHA,
                                  'files_sha256': expected}
    save(report)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    report = {'recipe': 'ovisocr2-controlled-original-vllm-quality-v1',
              'phase': 'reviewed-v6-gate', 'status': 'started', 'commands': [],
              'scope': 'Eleven frozen original GPU quality images on one qualified T4; controlled decoder TRITON_ATTN, vLLM auto-FP16. No native parity or auto-FlashInfer equivalence claim.',
              'pins': {'model': MODEL, 'revision': REVISION, 'checkpoint_sha256': CHECKPOINT_SHA,
                       'vllm': '0.22.1+cu129', 'vllm_commit': VLLM_COMMIT,
                       'torch': '2.11.0', 'transformers': '5.16.1', 'tokenizers': '0.23.3',
                       'pillow': '12.1.0', 'worker_commit': WORKER_COMMIT,
                       'worker_sha256': WORKER_SHA, 'quality_source_commit': QUALITY_SOURCE_COMMIT,
                       'v6_receipt_sha256': V6_RECEIPT_SHA,
                       'realistic_mirror_sha256': QUALITY_MIRROR_SHA},
              'limits': ['Original README auto-FlashInfer T4 failure remains unqualified; controlled decoder TRITON_ATTN only.',
                         'Synthetic cases are smoke, three separately reviewed documents are a small licensed panel; not a public OCR benchmark.',
                         'No CPU model reference, native comparison, throughput claim, heldout calibration, or weight publication.']}
    save(report)
    try:
        report['script_sha256'] = digest(__file__)
        v6_positive_gate(report)
        report['service_frontend'] = service_gate(FRONTEND_RUN, FRONTEND_HEAD, 'Corrected frontend')
        report['service_metadata'] = service_gate(METADATA_RUN, METADATA_HEAD, 'Original metadata')
        report['phase'] = 'metadata-proof'; save(report)
        metadata_proof(report)
        report['phase'] = 'frozen-sources'; save(report)
        frozen_sources(report)
        frozen_realistic_images(report)
        report['phase'] = 'hardware'; save(report)
        gpu_gate(report)
        driver_env = cuda_link_environment(report, os.environ)
        report['phase'] = 'worker-source'; save(report)
        worker = SCRATCH / 'worker.py'
        fetch(WORKER_URL, worker, timeout=60)
        if digest(worker) != WORKER_SHA:
            raise ValueError('Pinned original GPU quality worker source differs')
        report['worker_sha256'] = digest(worker); save(report)
        report['phase'] = 'isolated-packages'; save(report)
        python = isolated_environment(report)
        wheel = SCRATCH / WHEEL_NAME
        fetch(WHEEL_URL, wheel, timeout=900)
        if wheel.stat().st_size != WHEEL_BYTES or digest(wheel) != WHEEL_SHA:
            raise ValueError('Published pinned CUDA 12.9 wheel differs')
        install_report = SCRATCH / 'pip-install-report.json'
        run('install-pinned-stack', [python, '-m', 'pip', 'install', '--report', str(install_report), str(wheel),
             'torch==2.11.0', 'torchvision==0.26.0', 'transformers==5.16.1',
             'tokenizers==0.23.3', 'numpy==2.2.6', 'pillow==12.1.0',
             'huggingface-hub==1.33.0', '--extra-index-url',
             'https://download.pytorch.org/whl/cu129'], report, timeout=1800)
        report['package_install_report_sha256'] = digest(install_report)
        installs = json.loads(install_report.read_text())['install']
        report['package_archive_sha256'] = {entry['metadata']['name']: {
            'version': entry['metadata']['version'],
            'sha256': entry.get('download_info', {}).get('archive_info', {}).get('hashes', {}).get('sha256')}
            for entry in installs}
        save(report)
        report['phase'] = 'original-gpu-quality'; save(report)
        env = dict(driver_env, CUDA_VISIBLE_DEVICES='0', OVIS_OUT=str(OUT), OVIS_SCRATCH=str(SCRATCH),
                   OVIS_PROOF_COMMIT=PROOF_COMMIT, OVIS_CHECKPOINT_SHA=CHECKPOINT_SHA)
        run('original-vllm-quality-worker', [python, str(worker)], report, env=env, timeout=10800)
        result = json.loads((OUT / 'ovisocr2-vllm-quality-worker.json').read_text())
        report['worker'] = result
        if (result['status'] != 'completed bounded original vLLM quality reference'
                or len(result['outputs']) != 11):
            raise ValueError('Original GPU quality output coverage incomplete')
        report['status'] = 'completed bounded original vLLM GPU quality reference'
        report['phase'] = 'complete'; save(report)
    except BaseException as error:
        report['status'] = 'inconclusive' if report['phase'] in ('reviewed-v6-gate', 'hardware') else 'failed'
        report['error_type'] = type(error).__name__
        report['error_phase'] = report['phase']
        save(report)
        raise


if __name__ == '__main__':
    main()
