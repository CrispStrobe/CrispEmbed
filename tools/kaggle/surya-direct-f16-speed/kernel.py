"""Private Kaggle CPU direct-F16 paired speed and decoded comparison."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT = "@COMMIT@"
SOURCES_B64 = "@SOURCES_B64@"
SOURCES_SHA256 = "@SOURCES_SHA256@"
REPO = Path('/kaggle/temp/ce-speed')
BUILD = '/kaggle/temp/build-speed'
PYTHON = '/kaggle/temp/venv-speed/bin/python'
WORKING = Path('/kaggle/working')


def sh(*args, **kwargs):
    print('+', *args, flush=True)
    subprocess.run(list(args), check=True, **kwargs)


def main():
    WORKING.mkdir(exist_ok=True, parents=True)
    receipt: dict[str, object] = {'commit': COMMIT, 'source_bundle_sha256': SOURCES_SHA256,
               'phase': 'checkout', 'status': 'failed'}
    def save():
        (WORKING / 'speed-kernel-bootstrap.json').write_text(json.dumps(receipt, indent=2) + '\n')
    save()
    try:
        sh('git', 'clone', '-q', '--filter=blob:none', 'https://github.com/CrispStrobe/CrispEmbed.git', str(REPO))
        os.chdir(REPO)
        sh('git', 'checkout', '-q', COMMIT)
        sh('git', 'submodule', 'update', '-q', '--init', '--depth', '1', 'ggml')
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() != COMMIT:
            raise ValueError('commit mismatch')
        if COMMIT != '9498e67e3558a9a0b0ec20f93c40698b2434204e':
            raise ValueError('benchmark baseline mismatch')
        if subprocess.check_output(['git', '-C', 'ggml', 'rev-parse', 'HEAD'], text=True).strip() != 'a06fc0bb46b73cc10011fc90ae4054aee8438aeb':
            raise ValueError('ggml commit mismatch')
        bundle_bytes = base64.b64decode(SOURCES_B64, validate=True)
        if hashlib.sha256(bundle_bytes).hexdigest() != SOURCES_SHA256:
            raise ValueError('embedded bundle checksum mismatch')
        bundle = json.loads(bundle_bytes)
        expected = {'src/qwen35_linear.cpp', 'tools/kaggle/surya-ocr-25-quant/run.py',
                    'tools/kaggle/surya-direct-f16-speed/run.py'}
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
        receipt['phase'] = 'build'; save()
        sh('apt-get', 'update', '-qq')
        sh('apt-get', 'install', '-y', '-qq', 'fonts-dejavu-core')
        sh('cmake', '-S', '.', '-B', BUILD, '-DCMAKE_BUILD_TYPE=Release',
           '-DCRISPEMBED_NATIVE=OFF', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_METAL=OFF')
        sh('cmake', '--build', BUILD, '--parallel', '4', '--target', 'test-surya25-run')
        receipt['phase'] = 'python'; save()
        sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
        sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv-speed')
        sh(PYTHON, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'gguf==0.19.0',
           'safetensors==0.8.0', 'huggingface-hub==1.33.0', 'pillow==12.1.0',
           'tokenizers==0.23.3', 'transformers==5.16.1')
        sh(PYTHON, '-m', 'pip', 'install', '-q', 'torch==2.14.0', 'torchvision==0.29.0',
           '--index-url', 'https://download.pytorch.org/whl/cpu')
        receipt['phase'] = 'execute'; save()
        result = subprocess.run([PYTHON, 'tools/kaggle/surya-direct-f16-speed/run.py'],
                                env=dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                                         SURYA25_COMMIT=COMMIT, SPEED_SOURCE_BUNDLE_SHA256=SOURCES_SHA256,
                                         SPEED_BUILD=BUILD))
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
