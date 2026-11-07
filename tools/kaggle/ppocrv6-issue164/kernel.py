"""Issue 164: live medium PP-OCRv6 detector/recognizer A/B on CUDA."""
import glob
import os
from pathlib import Path
import subprocess
import sys

COMMIT = '@COMMIT@'
BRANCH = 'fix/ppocrv6-det-issue164-20261106'
REPO = Path('/kaggle/temp/ppocrv6-issue164')
OUT = Path('/kaggle/working/out')
SCRATCH = Path('/kaggle/temp/ppocrv6-issue164-scratch')


def main():
    if len(COMMIT) != 40 or not all(c in '0123456789abcdef' for c in COMMIT):
        raise SystemExit('Expected an immutable source commit')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'], text=True).strip()
    if not gpu:
        raise SystemExit('CUDA GPU required')
    print('GPU:', gpu, flush=True)
    arch = gpu.splitlines()[0].split(',')[-1].strip().replace('.', '')
    subprocess.run(['git', 'clone', '--filter=blob:none', '--branch', BRANCH,
                    'https://github.com/CrispStrobe/CrispEmbed.git', str(REPO)], check=True)
    subprocess.run(['git', '-C', str(REPO), 'checkout', '--detach', COMMIT], check=True)
    assert subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT
    subprocess.run(['git', '-C', str(REPO), 'submodule', 'update', '--init', '--depth', '1', 'ggml'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy==2.2.6', 'pillow==11.3.0'], check=True)
    OUT.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    flags = ['-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=' + arch]
    stubs = glob.glob('/usr/local/cuda/targets/*/lib/stubs/libcuda.so')
    if stubs:
        flags.append('-DCMAKE_LIBRARY_PATH=' + str(Path(stubs[0]).parent))
    env = dict(os.environ, HEAVY_OUT=str(OUT), HEAVY_SCRATCH=str(SCRATCH), DEVICE='gpu', CMAKE_EXTRA=' '.join(flags))
    return subprocess.run([sys.executable, str(REPO / 'tools/ci-heavy/ppocrv6_issue164.py')], env=env, cwd=REPO).returncode


if __name__ == '__main__':
    sys.exit(main())
