#!/usr/bin/env python3
"""One genuine CUDA F16 audit, replaying original frozen arrays without Torch.

P100 is useful here: ggml is compiled for the detected GPU; no Torch CUDA/MMQ
claim is made. Never retry draws. No runtime gate/default/precision changes.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SCRIPT_VERSION = 'lfm2-encoder-cuda-v1'
EMBED_SOURCE = '898af2c2a76369ceecd700a2b021ca001483a9c1'
HARNESS_SOURCE = '1395603321611cfc9c89f1474a0c0093ac3ba571'
MODEL_SOURCE = 'bb26e78c7164b386028928ebfcecebcc42467689'
FIXTURE_SOURCE = 'e3599a94f861d85ab735d68cb57b9f5aedc395a6'
MODEL_SHA = 'f097dbdb27c565b6f08cdbbfeb5d674ad7c205cfd1d8362f5415e243b2ec2e93'
OUT = Path('/kaggle/working'); LOGS = OUT / 'logs'
SCRATCH = Path('/tmp/lfm2-encoder-cuda-audit')
REPO = SCRATCH / 'CrispEmbed'; BUILD = SCRATCH / 'build'
for directory in (OUT, LOGS, SCRATCH): directory.mkdir(parents=True, exist_ok=True)
report = dict(script_version=SCRIPT_VERSION, embed_source=EMBED_SOURCE, harness_source=HARNESS_SOURCE,
              model_source=MODEL_SOURCE, fixture_source=FIXTURE_SOURCE, status='started',
              scope='F16 CPU/CUDA reference agreement; no quantized MMQ, latency or independent task-quality claim',
              commands=[], arms={}, frozen_inputs_unchanged=True)

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def save():
    (OUT / 'lfm2-cuda-audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

def run(name, command, env=None, timeout=1800):
    print(name, flush=True)
    path = LOGS / (name + '.log')
    started = time.monotonic()
    with path.open('w') as log:
        proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=timeout)
    entry = dict(name=name, exit_code=proc.returncode, log=path.name, log_sha256=digest(path),
                 wall_seconds=time.monotonic()-started)
    report['commands'].append(entry);save()
    return entry,path

def checked(name,command,env=None):
    entry,path=run(name,command,env)
    if entry['exit_code']: raise RuntimeError(name+' failed; inspect retained log')
    return path

def clone(url,commit,directory):
    checked(directory.name+'-clone',['git','clone','--no-checkout','--filter=blob:none',url,str(directory)])
    checked(directory.name+'-checkout',['git','-C',str(directory),'checkout',commit])
    actual=subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
    if actual!=commit: raise ValueError('Source checkout identity changed')

try:
    save()
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap','--format=csv,noheader'],text=True)
    nvcc=subprocess.check_output(['nvcc','--version'],text=True)
    report.update(gpu=gpu,nvcc=nvcc)
    capability=re.search(r',\s*(\d+)\.(\d+)',gpu)
    toolkit=re.search(r'release (\d+)\.',nvcc)
    if not capability or not toolkit: raise ValueError('Cannot identify CUDA hardware/toolkit')
    arch=int(capability[1])*10+int(capability[2])
    if int(toolkit[1])>=13 and arch<75:
        report.update(status='inconclusive: installed toolkit cannot compile detected legacy GPU',conclusive=False)
        save();raise SystemExit(0)
    # Published inputs need no private upload. Resolve optional auth through the
    # standard harness; never print token contents or attach cache datasets.
    support=SCRATCH/'audit-support'
    clone('https://github.com/CrispStrobe/CrispASR.git',HARNESS_SOURCE,support)
    sys.path.insert(0,str(support/'tools/kaggle'))
    import kaggle_harness as kh
    kh.resolve_hf_token(require=False)
    checked('python-dependencies',[sys.executable,'-m','pip','install','gguf==0.19.0','numpy==2.2.6','huggingface-hub==1.33.0'])
    from huggingface_hub import hf_hub_download,snapshot_download
    model=Path(hf_hub_download('LiquidAI/LFM2.5-Encoder-230M-GGUF','LFM2.5-Encoder-230M-F16.gguf',revision=MODEL_SOURCE))
    if digest(model)!=MODEL_SHA: raise ValueError('Official F16 weight hash changed')
    base=Path(snapshot_download('cstr/crispembed-regression-fixtures',repo_type='dataset',revision=FIXTURE_SOURCE,
                               allow_patterns=['lfm2-encoder-230m/original-fp32/*']))
    refs=base/'lfm2-encoder-230m/original-fp32'
    texts=json.loads((refs/'index.json').read_text())
    if len(texts)!=15: raise ValueError('Incomplete original input panel')
    report['model_sha256']=MODEL_SHA
    report['reference_files']={f.name:dict(bytes=f.stat().st_size,sha256=digest(f)) for f in refs.iterdir() if f.is_file()}
    report['reference_versions']=json.loads((refs/'versions.json').read_text())
    clone('https://github.com/CrispStrobe/CrispEmbed.git',EMBED_SOURCE,REPO)
    checked('submodules',['git','-C',str(REPO),'submodule','update','--init','--recursive','--depth','1'])
    checked('configure',['cmake','-S',str(REPO),'-B',str(BUILD),'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF',
                         '-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=ON','-DGGML_METAL=OFF','-DGGML_BLAS=OFF',
                         '-DCRISPEMBED_BUILD_SHARED=ON','-DCMAKE_CUDA_ARCHITECTURES='+str(arch)])
    checked('build',['cmake','--build',str(BUILD),'--target','test-lfm2-diff','crispembed-shared','--parallel','2'])
    binary=BUILD/'test-lfm2-diff'
    env=dict(os.environ,CRISPEMBED_LFM2_ENCODER='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    # Reject unknown requests; model-free probe must not hide CUDA absence.
    entry,_=run('unknown-backend',[str(binary),'--backend-probe'],dict(env,CRISPEMBED_TEST_BACKEND='invalid'))
    if entry['exit_code']!=2: raise ValueError('Unknown backend was not rejected')
    for backend in ('cpu','cuda'):
        selected=dict(env,CRISPEMBED_TEST_BACKEND=backend,CRISPEMBED_FORCE_CPU='1' if backend=='cpu' else '0')
        probe=checked(backend+'-probe',[str(binary),'--backend-probe'],selected).read_text()
        expected='CPU' if backend=='cpu' else 'CUDA'
        if not re.search(r'AUDIT_BACKEND\t'+expected,probe): raise ValueError('Backend request fell back')
        arm=dict(direct_cases=[],api=None);report['arms'][backend]=arm
        for i,text in enumerate(texts):
            entry,path=run(f'{backend}-stage-{i:03}',[str(binary),str(model),str(refs/f'{i:03}.gguf'),text],selected)
            log=path.read_text(errors='replace')
            totals=re.search(r'PASS:\s*(\d+)\s+FAIL:\s*(\d+)',log)
            arm['direct_cases'].append(dict(case=i,exit_code=entry['exit_code'],pass_count=int(totals[1]) if totals else None,
                                            fail_count=int(totals[2]) if totals else None,backend_verified=bool(re.search(r'AUDIT_BACKEND\t'+expected,log))))
            save()
        api_path=OUT/(backend+'-api.json')
        entry,path=run(backend+'-api',[sys.executable,str(REPO/'tests/lfm2_encoder_parity.py'),'--model',str(model),
                    '--refs',str(refs),'--lib',str(BUILD/'libcrispembed.so'),'--measure-only','--output',str(api_path)],
                    dict(selected,GGML_SCHED_DEBUG='2'),timeout=1200)
        arm['api_exit']=entry['exit_code']
        if api_path.exists(): arm['api']=json.loads(api_path.read_text())
        log=path.read_text(errors='replace')
        arm['cuda_matmul_assignments']=len(re.findall(r'node #[^\n]*\(\s*MUL_MAT\)[^\n]*\[\s*CUDA',log))
        if backend=='cpu' and arm['cuda_matmul_assignments']: raise ValueError('CPU control assigned CUDA matmul')
        save()
    failures=[]
    for backend,arm in report['arms'].items():
        if len(arm['direct_cases'])!=15 or sum(x['pass_count'] or 0 for x in arm['direct_cases'])!=297:
            failures.append(backend+': incomplete per-layer panel')
        if any(x['exit_code'] or x['fail_count']!=0 or not x['backend_verified'] for x in arm['direct_cases']):
            failures.append(backend+': direct graph failure')
        api=arm['api']
        if arm['api_exit'] or not api or len(api['cases'])!=15 or api['masks']!=15 or api['top1_matches']!=15:
            failures.append(backend+': API/mask failure')
        if api:
            for case in api['cases']:
                for key in ('hidden','cls','logits'):
                    if key not in case:continue
                    record=case[key]
                    if (not all(math.isfinite(v) for v in record.values())
                            or record['cos_min']<0.999 or record['relative_error']>0.02):
                        failures.append(backend+': numerical failure '+key)
        if backend=='cuda' and not arm['cuda_matmul_assignments']:
            failures.append('CUDA: no ordinary scheduler matmul execution evidence')
    report.update(status='completed; all failures retained',failures=failures,conclusive=True,
                  numerical_gate=dict(min_cos=0.999,max_relative_error=0.02,mask_top1=15),
                  qualification='pass' if not failures else 'failed; no default promotion')
    save()
    print('Qualification:',report['qualification'],flush=True)
    if failures: raise SystemExit(1)
except Exception as error:
    report.update(status='failed structurally; retained logs',error_type=type(error).__name__,error=str(error)[:300],conclusive=False)
    save();print('Audit failed:',type(error).__name__,flush=True)
    raise
