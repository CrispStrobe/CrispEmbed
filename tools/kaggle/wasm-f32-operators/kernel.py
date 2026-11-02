"""Actual CPU/browser retained-operator replay; render only an immutable @COMMIT@.

No production inference, original forward, GPU allocation or submission here.
An authorized artifact-read input must be attached privately by the operator.
"""
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time

COMMIT='@COMMIT@'
SDK='ca38f487f28b7c3c16f8f70cd0e012099ac4b7e2'
NODE_SHA='f4cb75bb036f0d0eddf6b79d9596df1aaab9ddccd6a20bf489be5abe9467e84e'
OUT=Path('/kaggle/working/wasm-f32-operators')
SCRATCH=Path('/kaggle/temp/wasm-f32-operators')


def main():
    if not (Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()and
            os.environ.get('KAGGLE_KERNEL_RUN_TYPE')in('Batch','Interactive')):
        raise ValueError('Actual Kaggle CPU task required')
    if not re.fullmatch('[0-9a-f]{40}',COMMIT):raise ValueError('Immutable source required')
    if os.environ.get('GITHUB_ACTIONS')=='true':raise ValueError('Do not impersonate GitHub Actions')
    OUT.mkdir(exist_ok=False);SCRATCH.mkdir(parents=True,exist_ok=False)
    script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report=dict(source=COMMIT,provider='Kaggle CPU',completed=False,model_forward=False,
                original_rerun=False,executed_script_sha256=script_sha,commands=[])
    def save():(OUT/'setup.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(args,timeout=1800):
        log=OUT/('command-'+str(len(report['commands']))+'.log');start=time.monotonic()
        with log.open('wb')as stream:
            child=subprocess.Popen(list(map(str,args)),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait();report['timed_out']=True
        report['commands'].append(dict(argv=list(map(str,args)),exit=child.returncode,
            seconds=time.monotonic()-start,log=log.name,sha256=hashlib.sha256(log.read_bytes()).hexdigest()));save()
        if child.returncode:raise ValueError('Hosted command rejected: '+log.name)
    try:
        available=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024
        report['resources']=dict(available_ram_bytes=available,disk_free_bytes=shutil.disk_usage(SCRATCH).free,cpu_count=os.cpu_count());save()
        if available<8*2**30 or report['resources']['disk_free_bytes']<8*2**30 or os.cpu_count()<2:
            raise ValueError('Remote task requires8GiB RAM/disk andtwoCPUs')
        if shutil.which('nvidia-smi'):
            gpu=subprocess.run(['nvidia-smi','-L'],capture_output=True,text=True,timeout=15)
            if gpu.returncode==0 and gpu.stdout.strip():raise ValueError('CPU task must not reserve a GPU')
        tokens={str(Path(p).resolve())for p in glob.glob('/kaggle/input/**/gh_token.txt',recursive=True)}
        if len(tokens)!=1:raise ValueError('One authorized artifact-read credential input required')
        os.environ['GH_TOKEN']=Path(next(iter(tokens))).read_text().strip()
        if not os.environ['GH_TOKEN']:raise ValueError('Empty artifact-read credential')
        os.environ.update(CRISPEMBED_CAPTURE_SOURCE=COMMIT,CRISPEMBED_BOOTSTRAP_SHA=script_sha,
            OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',CMAKE_BUILD_PARALLEL_LEVEL='2')
        repo=SCRATCH/'source'
        run(['git','clone','-q','--filter=blob:none','https://github.com/CrispStrobe/CrispEmbed.git',repo]);os.chdir(repo)
        run(['git','checkout','-q',COMMIT])
        run([sys.executable,'-m','pip','install','-q','uv==0.8.17'])
        run([sys.executable,'-m','uv','venv','-q','--python','3.12','--seed',SCRATCH/'venv']);py=SCRATCH/'venv/bin/python'
        run([py,'tests/test_wasm_f32_operators_v1.py','-v'])
        panel=OUT/'panel';driver=repo/'tools/ci-heavy/wasm_f32_operators_v1.py'
        run([py,driver,'prepare','--output',panel])
        sdk=SCRATCH/'emsdk'
        run(['git','clone','-q','--depth','1','--branch','6.0.2','https://github.com/emscripten-core/emsdk.git',sdk])
        if subprocess.check_output(['git','-C',sdk,'rev-parse','HEAD'],text=True).strip()!=SDK:raise ValueError('SDK tag identity differs')
        if hashlib.sha256((sdk/'emsdk_manifest.json').read_bytes()).hexdigest()!='7a7201711bd5311a17d95a3cf97329ebc5bebe35d36ab4657631a5c74662c188':
            raise ValueError('SDK dependency manifest differs')
        run([sdk/'emsdk','install','6.0.2'],1800);run([sdk/'emsdk','activate','6.0.2'])
        # Positional shell argument; environment values, including credentials,
        # are captured only in memory and never written into logs or receipts.
        raw=subprocess.check_output(['bash','-c','. "$1/emsdk_env.sh" >/dev/null 2>&1; env -0','emsdk-env',str(sdk)])
        os.environ.update(dict(p.split('=',1)for p in raw.decode().split('\0')if '='in p))
        run([py,driver,'build','--output',panel]);run([py,driver,'native','--output',panel])
        archive=SCRATCH/'node.tar.xz'
        run(['curl','--fail','--location','--http1.1','--retry','2','--retry-all-errors','--max-time','600',
             '--max-filesize','100000000','https://nodejs.org/dist/v22.16.0/node-v22.16.0-linux-x64.tar.xz','-o',archive])
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=NODE_SHA:raise ValueError('Pinned Node22 archive differs')
        with tarfile.open(archive)as source:source.extractall(SCRATCH,filter='data')
        node=SCRATCH/'node-v22.16.0-linux-x64/bin';os.environ['PATH']=str(node)+os.pathsep+os.environ['PATH']
        run([node/'node','--version']);browser=SCRATCH/'browser'
        run([node/'npm','install','--prefix',browser,'--no-save','playwright@1.56.0'])
        run([browser/'node_modules/.bin/playwright','install','--with-deps','chromium'],1800)
        os.environ['NODE_PATH']=str(browser/'node_modules')
        run([py,driver,'verify','--output',panel])
        run([node/'node',repo/'tests/wasm-f32-operators/browser.cjs',panel],1200)
        run([py,'-m','pip','install','-q','numpy==2.2.6'])
        run([py,driver,'compare','--output',panel])
        result=json.loads((panel/'review/operators.json').read_text())
        if not result['completed']or result['qualification']or result['own_anchors']['exact_own_input_output_anchors']!=64:
            raise ValueError('Complete attribution/own-anchor receipt required')
        report.update(completed=True,qualification=False);save()
    except Exception as error:
        report['error']=type(error).__name__+': '+str(error);save();raise


if __name__=='__main__':main()
