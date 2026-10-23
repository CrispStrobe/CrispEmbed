#!/usr/bin/env python3
"""Paired actual CPU/CUDA F16 dense API timings with frozen numerical controls.

No full-stage repeat, quantization, default promotion or task-quality claim.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

SCRIPT_VERSION = 'lfm2-encoder-cuda-performance-v2'
EMBED_SOURCE = '898af2c2a76369ceecd700a2b021ca001483a9c1'
HARNESS_SOURCE = '1395603321611cfc9c89f1474a0c0093ac3ba571'
MODEL_SOURCE = 'bb26e78c7164b386028928ebfcecebcc42467689'
FIXTURE_SOURCE = 'e3599a94f861d85ab735d68cb57b9f5aedc395a6'
MODEL_SHA = 'f097dbdb27c565b6f08cdbbfeb5d674ad7c205cfd1d8362f5415e243b2ec2e93'
OUT = Path('/kaggle/working'); LOGS = OUT / 'logs'
SCRATCH = Path('/tmp/lfm2-encoder-cuda-performance')
REPO = SCRATCH / 'CrispEmbed'; BUILD = SCRATCH / 'build'
for directory in (OUT, LOGS, SCRATCH): directory.mkdir(parents=True, exist_ok=True)
report = dict(script_version=SCRIPT_VERSION, embed_source=EMBED_SOURCE, harness_source=HARNESS_SOURCE,
              model_source=MODEL_SOURCE, fixture_source=FIXTURE_SOURCE, status='started',
              scope='Paired ordinary dense F16 API latency and process RSS; fixed15-case numerical panel, no independent task quality or default promotion',
              commands=[], arms={}, frozen_inputs_unchanged=True)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def save():
    (OUT / 'lfm2-cuda-performance.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

def run(name, command, env=None, timeout=1800):
    print(name, flush=True)
    path = LOGS / (name + '.log')
    started = time.monotonic();timed_out=False
    with path.open('w') as log:
        proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait()
    entry=dict(name=name,exit_code=proc.returncode,log=path.name,log_sha256=digest(path),
               wall_seconds=time.monotonic()-started,timed_out=timed_out,timeout_seconds=timeout)
    report['commands'].append(entry);save()
    if timed_out:raise subprocess.TimeoutExpired(command,timeout)
    return entry,path

def checked(name,command,env=None,timeout=1800):
    entry,path=run(name,command,env,timeout)
    if entry['exit_code']:raise RuntimeError(name+' failed; inspect retained log')
    return path

def clone(url,commit,directory):
    checked(directory.name+'-clone',['git','clone','--no-checkout','--filter=blob:none',url,str(directory)])
    checked(directory.name+'-checkout',['git','-C',str(directory),'checkout',commit])
    actual=subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
    if actual!=commit: raise ValueError('Source checkout identity changed')

EXPECTED_REFS = {'005.gguf': {'bytes': 730208, 'sha256': 'ec8e6824e123f556f70797ce977bdc5a8064a5361663cb6fd69b73c55f6c675b'}, '004.gguf': {'bytes': 402528, 'sha256': 'd25133090c540486fe67825b94909accc39c213d29f5b61b6a3442a59bebf933'}, '009.gguf': {'bytes': 1451200, 'sha256': '8e587a6bc26e593545b5c740610d0eae61c5f5bf11fea7dc369f18e62a565744'}, '003.gguf': {'bytes': 1647840, 'sha256': '42aabe3a90d2f9056fd4a983de86c31c4aa398257cbfcd9376a695e5d0b8abac'}, '006.gguf': {'bytes': 74688, 'sha256': '3b0b8cfaff49ad3b2180ad6980d4dc877d74994dae72a3b499ab5c9377121998'}, '007.gguf': {'bytes': 205824, 'sha256': '724f00a7c5540ae0051d2454cfabe8086ff73853d6ae676160cb1f17e1302a60'}, '008.gguf': {'bytes': 926784, 'sha256': 'a1d9d35c734098efefdf6308a9e036f5d3b94cc646051574b13265e23f85e79d'}, '012.gguf': {'bytes': 730240, 'sha256': 'f6129c72fb26e0e4976aa30d4603dc7a0f98f895a5bd1bb7c756fe7c7407fdc7'}, '011.gguf': {'bytes': 861344, 'sha256': '2629a306cc8a35f1fea6344d1f0ea3a1a55e89400dea9da50d3973568fe7a7d9'}, 'decoded_vocab.json': {'bytes': 658507, 'sha256': 'db20e09e100885c8f319e937dbae831538972592e0fae84adc547563c4ec2581'}, '013.gguf': {'bytes': 1582304, 'sha256': '8ab88c1e6629d380ce200532d6f38419589cd2beeda5139dc3ee504b924bab10'}, '010.gguf': {'bytes': 992416, 'sha256': '1af170602341f94a77e155275d7e162e8a6b98ef66ef2c512a70d038eb3b49ae'}, '014.gguf': {'bytes': 42809792, 'sha256': '21bebf1af934251a7557b8d5267a254b85bfbdbe781bd0cd0f28be205ea44c13'}, '002.gguf': {'bytes': 926880, 'sha256': '8b4896bbb72bf1b76580a0dd765a9dd9ae71277613e85de7ab351ae6e87d7205'}, 'index.json': {'bytes': 3530, 'sha256': 'd7f837bc60e6fe0256614790f74f9475a851ad4f8f7b4e71064fe60faeb2b907'}, 'versions.json': {'bytes': 168, 'sha256': '3f61651a76af5f0c9a2c0fdc43627153e78dec51ea4909f1c866449d545e9ec6'}, '000.gguf': {'bytes': 861312, 'sha256': 'c13c73a3bfe1ad59f4aea5a8832b37c009570e74f107bfc9ca74d279bce2ebdf'}, '001.gguf': {'bytes': 861344, 'sha256': '32c3b368d3e5188926783daac6651e6cbc80bc855f179fdf85ce11871580776a'}}

WORKER = r'''
import sys,os,json,time,resource,hashlib,statistics
from pathlib import Path
import numpy as np
from gguf import GGUFReader
repo,library,model_path,refs,output,backend,mode=sys.argv[1:]
sys.path.insert(0,str(Path(repo)/'python'))
from crispembed import CrispEmbed
started=time.perf_counter();model=CrispEmbed(model_path,n_threads=2,lib_path=library,offline=True)
initialization=time.perf_counter()-started
if Path(model._lib._name).resolve()!=Path(library).resolve():raise ValueError('Library fallback')
texts=json.loads((Path(refs)/'index.json').read_text())
if mode=='placement':
    model.encode(texts[0]);print('PLACEMENT_COMPLETE',flush=True);raise SystemExit(0)
result=dict(backend=backend,initialization_seconds=initialization,threads=2,warmups=2,repeats=7,
    scope='Dense encode latency includes Python FFI/output copy and completed native computation; excludes init/token/reference checks and warmups',cases=[])
def save():Path(output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
save()
for index,text in enumerate(texts):
    ref=GGUFReader(Path(refs)/f'{index:03}.gguf')
    ids,hidden=model.encode_tokens(text,normalize=False)
    np.testing.assert_array_equal(ids,np.asarray(ref.fields['ref.input_ids'].contents(),dtype=np.int32))
    expected=np.array(next(t.data for t in ref.tensors if t.name=='cls_raw'),dtype=np.float64).reshape(-1)
    actual=hidden[0].astype(np.float64).reshape(-1)
    cosine=float(np.dot(actual,expected)/(np.linalg.norm(actual)*np.linalg.norm(expected)))
    relative=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    if not np.isfinite(actual).all() or cosine<.999 or relative>.02:raise ValueError('Original CLS numerical failure')
    first=model.encode(text)
    np.testing.assert_allclose(first,hidden[0]/np.linalg.norm(hidden[0]),atol=2e-6,rtol=2e-5)
    for _ in range(2):model.encode(text)
    elapsed=[]
    for _ in range(7):
        start=time.perf_counter_ns();vector=model.encode(text);elapsed.append((time.perf_counter_ns()-start)/1e6)
        np.testing.assert_array_equal(vector,first)
    result['cases'].append(dict(case=index,tokens=len(ids),milliseconds=elapsed,
        median_ms=statistics.median(elapsed),min_ms=min(elapsed),max_ms=max(elapsed),
        vector_sha256=hashlib.sha256(first.tobytes()).hexdigest(),cls_cosine=cosine,
        cls_relative_error=relative,cls_max_abs=float(np.max(np.abs(actual-expected))),
        native_norm=float(np.linalg.norm(actual)),reference_norm=float(np.linalg.norm(expected))))
    save()
result['peak_process_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
result['completed']=len(result['cases'])==15;save()
'''

try:
    save()
    report['script_sha256']=digest(__file__)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap','--format=csv,noheader'],text=True)
    nvcc=subprocess.check_output(['nvcc','--version'],text=True)
    report.update(gpu=gpu,nvcc=nvcc)
    capability=re.search(r',\s*(\d+)\.(\d+)',gpu);toolkit=re.search(r'release (\d+)\.',nvcc)
    if not capability or not toolkit:raise ValueError('Unknown CUDA hardware/toolkit')
    arch=int(capability[1])*10+int(capability[2])
    if int(toolkit[1])>=13 and arch<75:
        report.update(status='inconclusive unsupported legacy GPU/toolkit; no redraw',conclusive=False)
        save();raise SystemExit(0)
    support=SCRATCH/'audit-support';clone('https://github.com/CrispStrobe/CrispASR.git',HARNESS_SOURCE,support)
    sys.path.insert(0,str(support/'tools/kaggle'))
    import kaggle_harness as kh
    kh.init_progress();kh.resolve_hf_token(require=False)
    checked('python-dependencies',[sys.executable,'-m','pip','install','gguf==0.19.0','numpy==2.2.6','huggingface-hub==1.33.0'])
    from huggingface_hub import hf_hub_download,snapshot_download
    model=Path(hf_hub_download('LiquidAI/LFM2.5-Encoder-230M-GGUF','LFM2.5-Encoder-230M-F16.gguf',revision=MODEL_SOURCE))
    if digest(model)!=MODEL_SHA:raise ValueError('Official F16 model changed')
    base=Path(snapshot_download('cstr/crispembed-regression-fixtures',repo_type='dataset',revision=FIXTURE_SOURCE,
        allow_patterns=['lfm2-encoder-230m/original-fp32/*']))
    refs=base/'lfm2-encoder-230m/original-fp32'
    for name,entry in EXPECTED_REFS.items():
        if (refs/name).stat().st_size!=entry['bytes'] or digest(refs/name)!=entry['sha256']:
            raise ValueError('Previously audited reference bytes changed')
    report['reference_files']=EXPECTED_REFS;report['model_sha256']=MODEL_SHA
    clone('https://github.com/CrispStrobe/CrispEmbed.git',EMBED_SOURCE,REPO)
    checked('submodules',['git','-C',str(REPO),'submodule','update','--init','--recursive','--depth','1'])
    report['ggml_source']=subprocess.check_output(['git','-C',str(REPO/'ggml'),'rev-parse','HEAD'],text=True).strip()
    kh.install_build_toolchain()
    flags=kh.cuda_build_flags(kh.detect_cuda_arch())+kh.cache_and_link_flags()
    checked('configure',['cmake','-S',str(REPO),'-B',str(BUILD),'-DCMAKE_BUILD_TYPE=Release',
        '-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_METAL=OFF','-DGGML_BLAS=OFF',
        '-DCRISPEMBED_BUILD_SHARED=ON',*flags])
    with kh.build_heartbeat('lfm2-cuda-performance-build'):
        checked('build',['cmake','--build',str(BUILD),'--target','crispembed-shared','--parallel',kh.safe_build_jobs(gpu=True)],timeout=3600)
    library=BUILD/'libcrispembed.so';report['library_sha256']=digest(library)
    worker=SCRATCH/'performance-worker.py';worker.write_text(WORKER);report['worker_sha256']=digest(worker)
    env=dict(os.environ,CRISPEMBED_LFM2_ENCODER='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    env.pop('GGML_SCHED_DEBUG',None)
    for backend in ('cpu','cuda'):
        selected=dict(env,CRISPEMBED_FORCE_CPU='1' if backend=='cpu' else '0',GGML_SCHED_DEBUG='2')
        log=checked(backend+'-placement',[sys.executable,str(worker),str(REPO),str(library),str(model),str(refs),'unused',backend,'placement'],selected).read_text()
        count=len(re.findall(r'node #[^\n]*\(\s*MUL_MAT\)[^\n]*\[\s*CUDA',log))
        if ('PLACEMENT_COMPLETE' not in log or (backend=='cuda' and not count) or (backend=='cpu' and count)):
            raise ValueError('Actual ordinary scheduler placement not established')
        report['arms'][backend]=dict(cuda_matmul_assignments=count,samples=[]);save()
    # ABBA balances first/last positions. Fresh process/context per block.
    for block,backend in enumerate(('cpu','cuda','cuda','cpu')):
        selected=dict(env,CRISPEMBED_FORCE_CPU='1' if backend=='cpu' else '0')
        path=OUT/(f'block-{block}-{backend}.json')
        checked(f'block-{block}-{backend}',[sys.executable,str(worker),str(REPO),str(library),str(model),str(refs),str(path),backend,'timed'],selected)
        data=json.loads(path.read_text())
        if not data.get('completed') or len(data['cases'])!=15 or data['repeats']!=7:raise ValueError('Incomplete timing block')
        report['arms'][backend]['samples'].append(dict(block=block,report_sha256=digest(path),**data));save()
    report.update(status='completed bounded paired dense API performance; no default promotion',conclusive=True,
        limits='One detected GPU/host, fixed15-case F16 panel, two threads, two warmups/seven repeats, ABBA; RSS is per-process high-water including initialization/quality probes, not incremental VRAM or throughput')
    save();print('Completed paired performance',flush=True)
except Exception as error:
    report.update(status='failed structurally; all partial timings/logs retained',error_type=type(error).__name__,conclusive=False)
    save();raise
