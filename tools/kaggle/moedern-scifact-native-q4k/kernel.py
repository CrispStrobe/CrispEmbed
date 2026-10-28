"""Render @COMMIT@ and @VARIANT@; attach frozen originals privately.

CPU2 model execution only. No submission or GPU allocation. Private mount
metadata is outside Git. Every old/new original is authenticated before models.
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
VARIANT='@VARIANT@'
OUT=Path('/kaggle/working/moedern-scifact-native-q4k')
SCRATCH=Path('/kaggle/temp/crispembed-moedern-scifact-native-q4k')


def main():
    if not (Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()and
            os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in('Batch','Interactive')):
        raise ValueError('genuine hosted Kaggle CPU model task required')
    if VARIANT not in ('1.5M','27M','115M'):raise ValueError('reviewed explicit variant required before network')
    if not re.fullmatch('[0-9a-f]{40}',COMMIT):raise ValueError('reviewed exact worker source required before network')
    if platform.system()!='Linux'or platform.machine()!='x86_64'or sys.version_info[:2]!=(3,13):
        raise ValueError('locked Linux x64 CPython3.13 required')
    OUT.mkdir(exist_ok=False);SCRATCH.mkdir(parents=True,exist_ok=False)
    report=dict(schema='crispembed-moedern-scifact-native-q4k-bootstrap-v1',source=COMMIT,completed=False,commands=[],
                variant=VARIANT,execution='CPU2 all5483 native F32/Q4_K SciFact replay',gpu_execution=False,publication_qualified=False,
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
        run('fetch-original-source',['git','-C',checkout,'fetch','--quiet','origin','2a872f9fe7ac5637286ff7bda2b5c19afee23cd7'],env=env)
        run('source-closure',[sys.executable,checkout/'tools/moedern-scifact/contract_native_v1.py','--variant',VARIANT],env=env)
        run('submodule',['git','-C',checkout,'submodule','update','--init','--recursive','--depth','1'],env=env)
        prefix=SCRATCH/'venv';venv.EnvBuilder(with_pip=False,system_site_packages=False).create(prefix)
        py=prefix/'bin/python'
        # Proven no-ensurepip route; five exact wheels, Linux dependency closure.
        # No framework, Rust tokenizer or dataset package is installed.
        run('install-locked-tools',[sys.executable,'-m','pip','--python',py,'install','--no-deps',
            '--only-binary=:all:','--require-hashes','-r',checkout/'tools/kaggle/moedern-scifact-native-q4k/requirements.lock'],env=env)
        run('pip-check',[py,'-I','-m','pip','check'],env=env)
        run('stdlib-controls',[py,'-I',checkout/'tests/test_moedern_scifact_native_q4k.py','-v'],env=env)
        run('child-accounting-controls',[py,'-I',checkout/'tests/test_ci_process.py','-v'],env=env)
        run('client-bit-controls',[py,'-I',checkout/'tests/test_moedern_all_size_clients.py','-v'],env=env)
        run('actual-model-panel',[py,'-I',checkout/'tools/moedern-scifact/worker_v1.py','--reviewed-source',COMMIT,'--variant',VARIANT,
            '--mount','/kaggle/input','--scratch',SCRATCH/'model','--out',OUT/'result'],timeout=9*3600,env=env)
        result=json.loads((OUT/'result/review/native.json').read_text())
        if not (result['completed']and result['f32']['strict_f32_numeric_passed']and
                result['f32']['full_retrieval_gates_passed']and result['q4k']['capture']['completed']and result['publication_qualified']is False):
            raise ValueError('complete strict-control/diagnostic receipt required')
        report.update(completed=True,quantized_numeric_passed=result['quantized_numeric_passed'],
                      full_retrieval_gates_passed=result['full_retrieval_gates_passed']);save()
    except Exception as error:
        report.update(error=type(error).__name__+': '+str(error));save();raise


if __name__=='__main__':main()
