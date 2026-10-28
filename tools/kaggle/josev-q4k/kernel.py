"""Render @COMMIT@ to a reviewed immutable source, attach frozen v5 privately.

CPU2 model execution only. No submission, credential dataset or GPU allocation.
The input metadata stays outside Git; the worker rejects any non-v5 bytes.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import venv

COMMIT='@COMMIT@'
OUT=Path('/kaggle/working/josev-q4k')
SCRATCH=Path('/kaggle/temp/crispembed-josev-q4k')


def main():
    if not (Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()and
            os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in('Batch','Interactive')):
        raise ValueError('genuine hosted Kaggle CPU model task required')
    if not re.fullmatch('[0-9a-f]{40}',COMMIT):raise ValueError('reviewed exact worker source required before network')
    if platform.system()!='Linux'or platform.machine()!='x86_64'or sys.version_info[:2]!=(3,13):
        raise ValueError('locked Linux x64 CPython3.13 required')
    OUT.mkdir(exist_ok=False);SCRATCH.mkdir(parents=True,exist_ok=False)
    report=dict(schema='crispembed-josev-q4k-bootstrap-v1',source=COMMIT,completed=False,commands=[],
                execution='CPU2 actual F32/Q4_K model comparison',gpu_execution=False,publication_qualified=False,
                executed_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():(OUT/'setup.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,args,timeout=1200,env=None):
        log=OUT/(name+'.log');start=time.monotonic()
        with log.open('w')as stream:
            child=subprocess.Popen(list(map(str,args)),stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            while child.poll()is None:
                if time.monotonic()-start>timeout:
                    # Give the worker's handler time to kill/reap its own
                    # measured child group before terminating the wrapper.
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait()
                    break
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:print('HEARTBEAT',name,int(time.monotonic()-start),flush=True)
        report['commands'].append(dict(name=name,exit=child.returncode,seconds=time.monotonic()-start,
             log=log.name,sha256=hashlib.sha256(log.read_bytes()).hexdigest()));save()
        if child.returncode:raise RuntimeError('setup/model task rejected in '+name)
    try:
        available=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024
        report['resources']=dict(available_ram_bytes=available,free_disk_bytes=shutil.disk_usage(SCRATCH).free,cpu_count=os.cpu_count());save()
        if available<14*2**30 or report['resources']['free_disk_bytes']<24*2**30 or os.cpu_count()<2:
            raise ValueError('CPU task needs14GiB available RAM/24GiB disk/two CPUs before packages/model')
        if shutil.which('nvidia-smi'):
            gpu=subprocess.run(['nvidia-smi','-L'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,timeout=15)
            if gpu.returncode==0 and gpu.stdout.strip():raise ValueError('CPU task must not reserve a GPU session')
        env={k:v for k,v in os.environ.items()if k not in ('PYTHONPATH','PYTHONHOME','PIP_TARGET','PIP_PREFIX')and not k.startswith('CRISPEMBED_')}
        env.update(PYTHONNOUSERSITE='1',PIP_CONFIG_FILE='/dev/null',PIP_USER='0',PIP_DISABLE_PIP_VERSION_CHECK='1',
                   OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2')
        checkout=SCRATCH/'source'
        run('clone',['git','clone','--quiet','--filter=blob:none','--no-checkout','https://github.com/CrispStrobe/CrispEmbed.git',checkout],env=env)
        run('checkout',['git','-C',checkout,'checkout','--quiet',COMMIT],env=env)
        if subprocess.check_output(['git','-C',checkout,'rev-parse','HEAD'],text=True).strip()!=COMMIT:raise ValueError('checkout differs')
        run('fetch-original-source',['git','-C',checkout,'fetch','--quiet','origin','b2edf8c83c9aca1f8a9210198d47798ac13828b8'],env=env)
        run('source-closure',[sys.executable,checkout/'tools/josev2b_q4k_contract.py'],env=env)
        run('submodule',['git','-C',checkout,'submodule','update','--init','--recursive','--depth','1'],env=env)
        prefix=SCRATCH/'venv';venv.EnvBuilder(with_pip=False,system_site_packages=False).create(prefix)
        py=prefix/'bin/python'
        # Proven no-ensurepip route. Only pip and NumPy, with no transitive deps;
        # NumPy's exact wheel is retained from the reviewed original57 lock.
        run('install-locked-tools',[sys.executable,'-m','pip','--python',py,'install','--no-deps',
            '--only-binary=:all:','--require-hashes','-r',checkout/'tools/kaggle/josev-q4k/requirements.lock'],env=env)
        run('pip-check',[py,'-I','-m','pip','check'],env=env)
        run('stdlib-controls',[py,'-I',checkout/'tests/test_josev_q4k_packer.py','-v'],env=env)
        run('child-accounting-controls',[py,'-I',checkout/'tests/test_ci_process.py','-v'],env=env)
        run('actual-model-panel',[py,'-I',checkout/'tools/kaggle/josev-q4k/worker.py','--reviewed-source',COMMIT,
            '--mount','/kaggle/input','--scratch',SCRATCH/'model','--out',OUT/'result'],timeout=4*3600,env=env)
        result=json.loads((OUT/'result/review/hosted.json').read_text())
        if not (result['completed']and result['f32']['strict_original_f32_numeric_and_typed_passed']and
                result['q4k']['structural_passed']and result['publication_qualified']is False):
            raise ValueError('complete strict-control/diagnostic receipt required')
        report.update(completed=True,strict_q4_f32_gates_passed=result['strict_q4_f32_gates_passed'],
                      quant_task_decisions_passed=result['quant_task_decisions_passed']);save()
    except Exception as error:
        report.update(error=type(error).__name__+': '+str(error));save();raise


if __name__=='__main__':main()
