#!/usr/bin/env python3
"""Private single GPU submission, fetching a reviewed immutable worker source."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
import venv

WORKER_SOURCE = 'a69b3fcac4389b52d56cd4bb7560bfe3ae75dbad'
TOOLING_SHA = '7ca30c5842b7e7ba797a2d54e1b0294a8264c12b459a681f5cf1e827b06c45be'
OUT = Path('/kaggle/working/josev2b-head')
SCRATCH = Path('/tmp/crispembed-josev2b-head-v1')
REPORT = dict(schema='crispembed-josev-head-bootstrap-v1',completed=False,passed=False,
              retry_permitted=False,worker_source=WORKER_SOURCE,commands=[])


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def save():
    (OUT/'bootstrap.json').write_text(json.dumps(REPORT,indent=2)+'\n')


def run(name,args,*,timeout=600,cwd=None,env=None):
    path=OUT/(name+'.log');start=time.monotonic()
    print('START',name,flush=True)
    with path.open('w') as stream:
        child=subprocess.Popen([str(x) for x in args],cwd=cwd,env=env,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        timed_out=False
        while child.poll() is None:
            remaining=timeout-(time.monotonic()-start)
            if remaining<=0:
                timed_out=True;os.killpg(child.pid,signal.SIGKILL);child.wait();break
            try:child.wait(timeout=min(30,remaining))
            except subprocess.TimeoutExpired:print('HEARTBEAT',name,int(time.monotonic()-start),flush=True)
    REPORT['commands'].append(dict(name=name,exit_code=child.returncode,timed_out=timed_out,
        seconds=time.monotonic()-start,log=path.name,sha256=digest(path)))
    save()
    if child.returncode or timed_out:raise RuntimeError(name+' failed; retained log')
    return path


def install(python,repo,env):
    lock=repo/'tools/kaggle/josev2b-head/tooling-lock.json'
    if digest(lock)!=TOOLING_SHA:raise ValueError('Reviewed head tooling lock differs')
    cfg=json.loads(lock.read_text())
    if cfg['python_minor']!='3.13' or len(cfg['packages'])!=4:raise ValueError('Tooling profile differs')
    wheels=SCRATCH/'wheels';wheels.mkdir()
    lines=[];proof=[]
    for row in cfg['packages']:
        filename=row['filename']
        if Path(filename).name!=filename or not filename.endswith('.whl'):raise ValueError('Invalid wheel name')
        target=wheels/filename;h=hashlib.sha256();count=0
        with urllib.request.urlopen(row['url'],timeout=60) as response,target.open('xb') as stream:
            while chunk:=response.read(1024*1024):
                count+=len(chunk)
                if count>row['bytes']:raise ValueError('Wheel bound exceeded')
                h.update(chunk);stream.write(chunk)
        if count!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Wheel identity differs')
        lines.append(str(target)+' --hash=sha256:'+row['sha256'])
        proof.append(dict(name=row['name'],version=row['version'],bytes=count,sha256=h.hexdigest()))
    requirements=SCRATCH/'requirements.lock';requirements.write_text('\n'.join(lines)+'\n')
    run('install-head-tooling',[sys.executable,'-m','pip','--python',python,'install',
        '--no-index','--no-deps','--require-hashes','-r',requirements],env=env,timeout=300)
    (OUT/'tooling-archives.json').write_text(json.dumps(proof,indent=2)+'\n')
    run('tooling-pip-check',[sys.executable,'-m','pip','--python',python,'check'],env=env,timeout=60)


def main():
    if OUT.exists() or SCRATCH.exists():raise ValueError('Fresh task-owned paths required')
    OUT.mkdir(parents=True);save()
    try:
        # Verify real hardware BEFORE any tooling/code/model/payload downloads.
        hardware=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap,memory.total,memory.free,driver_version',
            '--format=csv,noheader,nounits'],text=True,timeout=20)
        rows=[[p.strip() for p in line.split(',')] for line in hardware.strip().splitlines()]
        if not rows or any(len(r)!=5 or tuple(map(int,r[1].split('.')))<(7,5) or int(r[2])<14336 or int(r[3])<11264 or tuple(map(int,r[4].split('.')))<(580,65,6) for r in rows):
            raise ValueError('Actual supported GPU required; no CPU fallback or resource retry')
        REPORT['hardware']=rows;save()
        if sys.version_info[:2]!=(3,13):raise ValueError('Reviewed tooling requires Python3.13')
        SCRATCH.mkdir();repo=SCRATCH/'repo'
        run('git-init',['git','init',repo],timeout=30)
        run('git-remote',['git','remote','add','origin','https://github.com/CrispStrobe/CrispEmbed.git'],cwd=repo,timeout=30)
        run('git-fetch',['git','-c','gc.auto=0','fetch','--depth=1','origin',WORKER_SOURCE],cwd=repo,timeout=300)
        run('git-checkout',['git','checkout','--detach','FETCH_HEAD'],cwd=repo,timeout=60)
        run('frozen-head-fetch',['git','-c','gc.auto=0','fetch','--depth=1','origin',
            'd8b99a9c8e3a5cf0f6699fc026d3396d26e03763'],cwd=repo,timeout=300)
        run('original-source-fetch',['git','-c','gc.auto=0','fetch','--depth=1','origin',
            'b2edf8c83c9aca1f8a9210198d47798ac13828b8'],cwd=repo,timeout=300)
        run('ggml-init',['git','submodule','update','--init','--depth=1','ggml'],cwd=repo,timeout=300)
        env=os.environ.copy()
        for key in ('PIP_INDEX_URL','PIP_EXTRA_INDEX_URL','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
        env.update(PIP_CONFIG_FILE=os.devnull,PIP_NO_CACHE_DIR='1',PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
            CRISPEMBED_JOSEV_GPU_HEAD_AUDIT='1')
        venv.EnvBuilder(with_pip=False,system_site_packages=False).create(SCRATCH/'venv')
        python=SCRATCH/'venv/bin/python';install(python,repo,env)
        run('head-cpu-cuda',[python,'tools/kaggle/josev2b-head/worker.py',
            '--output',OUT/'audit','--build',SCRATCH/'build'],cwd=repo,env=env,timeout=5400)
        result=json.loads((OUT/'audit/gpu.json').read_text())
        if result.get('passed') is not True or result.get('completed') is not True:
            raise ValueError('Head audit did not complete parity')
        REPORT.update(completed=True,passed=True)
    except Exception as error:
        REPORT.update(error_type=type(error).__name__,error=str(error));raise
    finally:save()


if __name__=='__main__':main()
