"""Render @COMMIT@; mount the immutable stopped-v3 output privately.

Read/hash audit only: no compilation, quantization or model forward.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

COMMIT = '@COMMIT@'
OUT = Path('/kaggle/working/unlimited-retained-v3')
SCRATCH = Path('/kaggle/temp/unlimited-retained-v3')


def main():
    if not (Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir() and
            os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Batch', 'Interactive')):
        raise ValueError('Genuine Kaggle CPU audit required')
    if not re.fullmatch('[0-9a-f]{40}', COMMIT):
        raise ValueError('Exact reviewed source required')
    OUT.mkdir(exist_ok=False)
    SCRATCH.mkdir(parents=True, exist_ok=False)
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:'))) * 1024
    if available < 10 * 2**30 or shutil.disk_usage(SCRATCH).free < 2 * 2**30:
        raise ValueError('10GiB RAM/2GiB scratch required before setup')
    if shutil.which('nvidia-smi'):
        gpu = subprocess.run(['nvidia-smi', '-L'], capture_output=True, timeout=15)
        if gpu.returncode == 0 and gpu.stdout.strip():
            raise ValueError('Read-only CPU audit must not reserve GPU resources')
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'PIP_TARGET', 'PIP_PREFIX')}
    env.update(PYTHONNOUSERSITE='1', PIP_CONFIG_FILE='/dev/null', PIP_USER='0',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    result = dict(source=COMMIT, completed=False, commands=[], native_forwards=0, quantizer_runs=0,
                  executed_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        (OUT / 'setup.json').write_text(json.dumps(result, indent=2) + '\n')
    def run(name, args, timeout=1800):
        log = OUT / (name + '.log')
        started = time.monotonic()
        with log.open('wb') as stream:
            p = subprocess.run(list(map(str, args)), env=env, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=timeout)
        result['commands'].append(dict(name=name, exit=p.returncode,
                                       seconds=time.monotonic() - started,
                                       log_sha256=hashlib.sha256(log.read_bytes()).hexdigest()))
        save()
        if p.returncode:
            raise ValueError('Retained audit command rejected: ' + name)
    save()
    try:
        root = SCRATCH / 'source'
        run('clone', ['git', 'clone', '--quiet', '--filter=blob:none', '--no-checkout',
                      'https://github.com/CrispStrobe/CrispEmbed.git', root])
        run('checkout', ['git', '-C', root, 'checkout', '--quiet', COMMIT])
        run('backend-source', ['git', '-C', root, 'submodule', 'update', '--quiet', '--init', '--depth', '1', 'ggml'])
        run('small-controls', [sys.executable, root / 'tests/test_unlimited_ocr2_q4k.py'])
        # The hosted system Python lacks a usable ensurepip. Use the same
        # pinned uv/CPython3.12 bootstrap as the completed source410 producer.
        run('install-uv', [sys.executable, '-m', 'pip', 'install', '--quiet', 'uv==0.8.17'])
        run('create-venv', [sys.executable, '-m', 'uv', 'venv', '--quiet',
                            '--python', '3.12', '--seed', SCRATCH / 'venv'])
        python = SCRATCH / 'venv/bin/python'
        run('locked-reader', [python, '-I', '-m', 'pip', 'install', '--quiet',
                              'numpy==2.2.6', 'gguf==0.17.1'])
        run('retained-audit', [python, '-I', root / 'tools/unlimited-ocr2-q4k/retained_v3.py',
                              '--source', COMMIT, '--mount', '/kaggle/input', '--out', OUT / 'review'])
        result['completed'] = True
        save()
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        save()
        raise


if __name__ == '__main__':
    main()
