"""Private Kaggle CPU genuine-scan crop A/B; no publication."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

COMMIT = '@COMMIT@'
SOURCES_B64 = '@SOURCES_B64@'
SOURCES_SHA256 = '@SOURCES_SHA256@'
UTIL_URL = 'https://raw.githubusercontent.com/datalab-to/surya/9d5aadfebb07939755c24e8a80d0e61b3be80895/surya/inference/util.py'
UTIL_SHA256 = 'd853246db9767d01c8648058b779047ad0d93b3c74c0f7783dcf9faeef33b11b'
REPO = Path('/kaggle/temp/ce-real-scan')
BUILD = '/kaggle/temp/build-real-scan'
PYTHON = '/kaggle/temp/venv-real-scan/bin/python'
OUT = Path('/kaggle/working')


def sh(*args, **kwargs):
    print('+', *args, flush=True)
    subprocess.run(list(args), check=True, **kwargs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, object] = {'commit': COMMIT, 'source_bundle_sha256': SOURCES_SHA256,
                                  'phase': 'checkout', 'status': 'failed'}
    def save():
        (OUT / 'scan-bootstrap.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    try:
        sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', str(REPO))
        os.chdir(REPO)
        sh('git', 'checkout', '-q', COMMIT)
        sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
            raise ValueError('commit mismatch')
        raw = base64.b64decode(SOURCES_B64, validate=True)
        if hashlib.sha256(raw).hexdigest() != SOURCES_SHA256:
            raise ValueError('source bundle checksum mismatch')
        bundle = json.loads(raw)
        expected = {'src/qwen35_linear.cpp', 'tools/kaggle/surya-ocr-25-quant/run.py',
                    'tools/kaggle/surya25-real-scan/run.py'}
        if set(bundle) != expected:
            raise ValueError('unexpected source bundle paths')
        for name, entry in bundle.items():
            source = base64.b64decode(entry['base64'], validate=True)
            if hashlib.sha256(source).hexdigest() != entry['sha256']:
                raise ValueError('source mismatch: ' + name)
            path = REPO / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source)
        receipt['source_sha256'] = {name: entry['sha256'] for name, entry in bundle.items()}
        receipt['phase'] = 'surya-util'; save()
        with urllib.request.urlopen(urllib.request.Request(UTIL_URL, headers={'User-Agent': 'CrispEmbed-audit/1.0'}), timeout=60) as response:
            util = response.read()
        if hashlib.sha256(util).hexdigest() != UTIL_SHA256:
            raise ValueError('pinned Surya package util mismatch')
        util_path = Path('/kaggle/temp/surya-pinned-util.py')
        util_path.write_bytes(util)
        receipt['package_util_sha256'] = UTIL_SHA256
        receipt['phase'] = 'build'; save()
        sh('cmake', '-S', '.', '-B', BUILD, '-DCMAKE_BUILD_TYPE=Release',
           '-DCRISPEMBED_NATIVE=OFF', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_METAL=OFF')
        sh('cmake', '--build', BUILD, '--parallel', '4', '--target', 'test-surya25-run')
        receipt['phase'] = 'python'; save()
        sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
        sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv-real-scan')
        sh(PYTHON, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'gguf==0.19.0',
           'safetensors==0.8.0', 'huggingface-hub==1.33.0', 'pillow==12.1.0',
           'tokenizers==0.23.3', 'transformers==5.16.1')
        sh(PYTHON, '-m', 'pip', 'install', '-q', 'torch==2.14.0', 'torchvision==0.29.0',
           '--index-url', 'https://download.pytorch.org/whl/cpu')
        receipt['phase'] = 'execute'; save()
        result = subprocess.run([PYTHON, 'tools/kaggle/surya25-real-scan/run.py'],
                                env=dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                                         SURYA25_COMMIT=COMMIT, SCAN_SOURCE_BUNDLE_SHA256=SOURCES_SHA256,
                                         SURYA25_NATIVE=BUILD + '/test-surya25-run',
                                         SURYA_UTIL_PATH=str(util_path)))
        receipt.update(phase='complete', status='complete' if result.returncode == 0 else 'failed',
                       exit_code=result.returncode)
        save()
        return result.returncode
    except Exception as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == '__main__':
    sys.exit(main())
