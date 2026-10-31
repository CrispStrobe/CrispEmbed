"""Private CPU OvisOCR2 direct-F16 decoded regression bootstrap."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

COMMIT = '@COMMIT@'
SOURCES_B64 = '@SOURCES_B64@'
SOURCES_SHA256 = '@SOURCES_SHA256@'
REPO = Path('/kaggle/temp/ce-ovis-regression')
BUILD = '/kaggle/temp/build-ovis-regression'
PYTHON = '/kaggle/temp/venv-ovis-regression/bin/python'
OUT = Path('/kaggle/working')


def sh(*args, **kwargs):
    print('+', *args, flush=True)
    subprocess.run(list(args), check=True, **kwargs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, object] = {'commit': COMMIT, 'source_bundle_sha256': SOURCES_SHA256,
                                  'phase': 'checkout', 'status': 'failed'}
    def save():
        (OUT / 'ovis-direct-bootstrap.json').write_text(json.dumps(receipt, indent=2) + '\n')
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
        expected = {'src/qwen35_linear.cpp', 'tools/kaggle/ovisocr2-direct-f16-regression/run.py'}
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
        sh('cmake', '-S', '.', '-B', BUILD, '-DCMAKE_BUILD_TYPE=Release',
           '-DCRISPEMBED_NATIVE=OFF', '-DGGML_NATIVE=OFF', '-DGGML_BLAS=OFF', '-DGGML_METAL=OFF')
        sh('cmake', '--build', BUILD, '--parallel', '4', '--target', 'test-ovisocr2-quality-run')
        receipt['phase'] = 'python'; save()
        sh(sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.8.17')
        sh(sys.executable, '-m', 'uv', 'venv', '-q', '--python', '3.12', '--seed', '/kaggle/temp/venv-ovis-regression')
        sh(PYTHON, '-m', 'pip', 'install', '-q', 'pillow==12.1.0', 'huggingface-hub==1.33.0')
        receipt['phase'] = 'execute'; save()
        result = subprocess.run([PYTHON, 'tools/kaggle/ovisocr2-direct-f16-regression/run.py'],
                                env=dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
                                         OVIS_REGRESSION_COMMIT=COMMIT, OVIS_SOURCE_BUNDLE_SHA256=SOURCES_SHA256,
                                         OVIS_BUILD=BUILD, PYTHONDONTWRITEBYTECODE='1'))
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
