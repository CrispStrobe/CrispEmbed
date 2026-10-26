#!/usr/bin/env python3
"""Single bounded original Josev CUDA reference; no native build or CPU fallback.

Upload this one script as a private GPU kernel. All executed repository code is
fetched at WORKER_SOURCE, independently of this bootstrap's own version.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import venv
import zipfile

WORKER_SOURCE = '1e723a96f8fed0ea34b0e071e88f29561a315aaa'
CONTRACT_SHA = '39129e437588ff2177a3ef491685d0a58ac3c168147cc5e98891f9ad46619460'
WHEELS = {'torch': {'url': 'https://files.pythonhosted.org/packages/d4/5a/7c50ba1b7b713d71d34669c6d13dab0a11531a3eceb0307a5162dbfec0f7/torch-2.13.0-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=a3a9a21312872af8a26950b2c15680335a386a1f56ed03e780653d78b9607e9e', 'filename': 'torch-2.13.0-cp313-cp313-manylinux_2_28_x86_64.whl', 'sha256': 'a3a9a21312872af8a26950b2c15680335a386a1f56ed03e780653d78b9607e9e', 'bytes': 526602329}, 'torchvision': {'url': 'https://files.pythonhosted.org/packages/32/db/062cdb5a84380a60439775311fff34d89229760d2a50680393dc18699956/torchvision-0.28.0-cp313-cp313-manylinux_2_28_x86_64.whl#sha256=ad7b3a439265cc3739a4ab5b4c998c0e38ea99c0ee7ca4dea35c5d0b099ec237', 'filename': 'torchvision-0.28.0-cp313-cp313-manylinux_2_28_x86_64.whl', 'sha256': 'ad7b3a439265cc3739a4ab5b4c998c0e38ea99c0ee7ca4dea35c5d0b099ec237', 'bytes': 7674669}}
PIP_BOOTSTRAP = {'url': 'https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70', 'sha256': 'fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70', 'bytes': 1819850, 'filename': 'pip-26.2.2-py3-none-any.whl'}
OUT = Path('/kaggle/working/josev2b-original')
SCRATCH = Path('/tmp/crispembed-josev2b-original-v1')
REPORT = {'schema': 'crispembed-josev2b-bootstrap-v1', 'conclusive': False,
          'status': 'preflight', 'retry_permitted': False, 'worker_source': WORKER_SOURCE,
          'runtime_profile': 'cuda-f32-python313-v2', 'python_expected': '3.13',
          'contract_sha256': CONTRACT_SHA, 'commands': [], 'model_payload_accessed': False,
          'limits': 'Original-only four owned text cases; no native or quality qualification.'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def save():
    (OUT/'bootstrap.json').write_text(json.dumps(REPORT, indent=2, allow_nan=False)+'\n')


def run(name, args, *, timeout=600, cwd=None, env=None):
    path = OUT/(name+'.log')
    start = time.monotonic()
    print('START', name, flush=True)
    with path.open('w') as stream:
        child = subprocess.Popen([str(x) for x in args], cwd=cwd, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        while child.poll() is None:
            remaining = timeout-(time.monotonic()-start)
            if remaining <= 0:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                break
            try:
                child.wait(timeout=min(30, remaining))
            except subprocess.TimeoutExpired:
                print('HEARTBEAT', name, int(time.monotonic()-start), flush=True)
    REPORT['commands'].append({'name': name, 'exit_code': child.returncode, 'timed_out': timed_out,
        'seconds': time.monotonic()-start, 'log': path.name, 'sha256': digest(path)})
    save()
    print('END', name, child.returncode, flush=True)
    if child.returncode or timed_out:
        raise RuntimeError(name+' failed; retained log')
    return path


def isolated_environment(env):
    # Reuse the established Ovis bootstrap: Debian ensurepip may be unavailable.
    # Base pip --python installs only pinned pip into this fresh task-owned venv.
    venv.EnvBuilder(with_pip=False, system_site_packages=False).create(SCRATCH/'venv')
    python = SCRATCH/'venv/bin/python'
    run('bootstrap-isolated-pip', [sys.executable,'-m','pip','--python',python,
        'install','--index-url','https://pypi.org/simple',PIP_BOOTSTRAP['url']], env=env, timeout=300)
    REPORT['isolated_pip'] = PIP_BOOTSTRAP
    save()
    return python


def main():
    if OUT.exists() or SCRATCH.exists():
        raise ValueError('fresh task-owned paths required; earlier attempts must be preserved')
    OUT.mkdir(parents=True)
    REPORT['bootstrap_sha256'] = digest(__file__)
    save()
    try:
        REPORT['python'] = sys.version
        save()
        # Hardware proof precedes Python-profile rejection, installation and payload access.
        data = subprocess.check_output(['nvidia-smi',
            '--query-gpu=name,compute_cap,memory.total,memory.free,driver_version',
            '--format=csv,noheader,nounits'], text=True, timeout=20)
        rows = [line.split(',') for line in data.strip().splitlines()]
        if not rows or any(len(row) != 5 for row in rows):
            raise ValueError('GPU hardware declaration unavailable')
        REPORT['hardware'] = [[part.strip() for part in row] for row in rows]
        for _, cap, total, free, driver in rows:
            # CUDA13 wheels require Turing or newer and the CUDA13 driver baseline.
            if tuple(map(int, cap.strip().split('.'))) < (7, 5):
                raise ValueError('unsupported GPU for pinned CUDA13 wheels; no retry')
            if tuple(map(int, driver.strip().split('.'))) < (580, 65, 6):
                raise ValueError('driver is below CUDA13.0 Linux baseline; no retry')
            if int(total) < 14*1024 or int(free) < 11*1024:
                raise ValueError('GPU memory bound not met')
        if sys.version_info[:2] != (3, 13):
            raise ValueError('reviewed wheel/runtime profile requires Python3.13')
        if shutil.disk_usage('/tmp').free < 25*2**30 or shutil.disk_usage(OUT).free < 3*2**30:
            raise ValueError('need25GiB scratch before install and3GiB result space')
        SCRATCH.mkdir()
        save()
        repo = SCRATCH/'repo'
        run('git-init', ['git','init',repo], timeout=30)
        run('git-remote', ['git','remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git'], cwd=repo, timeout=30)
        run('git-fetch', ['git','-c','gc.auto=0','fetch','--depth=1','origin',WORKER_SOURCE], cwd=repo, timeout=300)
        run('git-checkout', ['git','checkout','--detach','FETCH_HEAD'], cwd=repo, timeout=60)
        if digest(repo/'docs/josev2b-source-contract.json') != CONTRACT_SHA:
            raise ValueError('worker contract SHA differs')
        # Always a new isolated environment; no base FLA/causal-conv import leakage.
        env = os.environ.copy()
        for key in ('PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','PYTHONPATH','PYTHONHOME'):
            env.pop(key, None)
        env.update(PIP_NO_CACHE_DIR='1', PYTHONNOUSERSITE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   HF_HOME=str(SCRATCH/'hf'), OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')
        env.update(PIP_CONFIG_FILE=os.devnull, PIP_USER='0')
        python = isolated_environment(env)
        requirements = [wheel['url'] for wheel in WHEELS.values()]
        cfg = json.loads((repo/'docs/josev2b-source-contract.json').read_text())
        requirements += [name+'=='+version for name,version in cfg['runtime'].items()
                         if name not in ('python_minor','torch','torchvision')]
        run('install-original', [python,'-m','pip','install','--index-url','https://pypi.org/simple',*requirements],
            timeout=1500, env=env)
        # pip check and exact source checks run before checkpoint access.
        run('pip-check', [python,'-m','pip','check'], env=env, timeout=60)
        cuda_smoke = """import json, torch
assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')
a=torch.ones((2,2),device='cuda',dtype=torch.float32)
b=a@a
torch.cuda.synchronize()
assert b.cpu().tolist()==[[2.,2.],[2.,2.]]
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,
 'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(),
 'compiled_architectures':torch.cuda.get_arch_list(),'tiny_f32_cuda_gemm_exact':True}))
"""
        run('cuda-runtime-smoke', [python,'-c',cuda_smoke], env=env, timeout=60)
        run('source-controls', [python,'tests/test_josev2b_contract.py','-v'], cwd=repo, env=env, timeout=60)
        (OUT/'wheel-pins.json').write_text(json.dumps(WHEELS,indent=2)+'\n')
        # Preserve logs and partial reference outputs; model downloads stay in scratch.
        REPORT['model_payload_accessed'] = 'worker_phase_entered; see worker receipt for actual access'
        save()
        run('original-reference', [python,'tools/kaggle/josev2b_reference.py',
            '--reviewed-source',WORKER_SOURCE,'--reviewed-contract-sha256',CONTRACT_SHA,
            '--scratch',SCRATCH/'payload','--out',OUT/'reference'], cwd=repo, env=env, timeout=2400)
        reference = json.loads((OUT/'reference/reference.json').read_text())
        if reference.get('conclusive') is not True or reference.get('status') != 'completed':
            raise ValueError('original worker did not complete')
        REPORT.update(status='completed-original', conclusive=True,
            reference_sha256=digest(OUT/'reference/reference.json'))
    except Exception as exc:
        REPORT.update(status='rejected', error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        # Preserve an index of all result bytes. Never upload venv, model or tokenizer.
        REPORT['files'] = {str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':digest(p)}
            for p in OUT.rglob('*') if p.is_file() and p.name != 'bootstrap.json'}
        save()


if __name__ == '__main__':
    main()
