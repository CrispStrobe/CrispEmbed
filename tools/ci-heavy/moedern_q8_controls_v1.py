#!/usr/bin/env python3
"""Hosted model-free execution of frozen1f MoE Q8 controls; never model/corpus work."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
PROFILE=ROOT/'docs/moedern-q8-controls-v1.json'
FROZEN='1f35731372cc5c9a60cd2431d58353f4742e25fb'
REQUIRED={'src','CMakeLists.txt','ggml','cmake','tests/moedern-q8','tests/test_moedern_q8_profile.py',
          'tools/moedern-colbert','docs/moedern-q8-f32-critical-v1.json','tools/quantize.cpp',
          'tools/gguf_file_type_canonicalize.py','tools/gguf_file_type_values.py','tools/ci_process.py'}
EXPECTED_CACHE={'CMAKE_BUILD_TYPE':'Release','BUILD_SHARED_LIBS':'OFF','GGML_NATIVE':'OFF','GGML_BACKEND_DL':'OFF',
                'GGML_BLAS':'OFF','GGML_CUDA':'OFF','GGML_METAL':'OFF','GGML_VULKAN':'OFF','GGML_OPENCL':'OFF',
                'GGML_LLAMAFILE':'ON'}


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def identity(path):return dict(bytes=Path(path).stat().st_size,sha256=sha(path))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial')
    with temp.open('w',encoding='utf-8')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temp.replace(path)


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def validate(p):
    require(p['schema']=='crispembed-moedern-q8-model-free-controls-v1'and p['runtime_source']==FROZEN
            and p['model_payloads']is False and p['corpus_access']is False and p['model_qualified']is False
            and p['dispatch_authorized']is False,'model-free immutable scope differs')
    require(REQUIRED<=set(p['protected_git_objects']),'complete runtime/build/fixture closure required')
    require(p['authored_fixture']==dict(actual_q8_tensors=2,critical_f32_tensors=7,models=0)
            and p['typed_profile_counts']=={'1.5M':20,'27M':20,'115M':52}
            and p['authored_graphs']==6,'authored versus actual-model scope differs')
    require(not any(x.startswith('src/')or x=='CMakeLists.txt'for x in p['owned_sha256']),'caller cannot own runtime/build changes')
    return p


def verify(reviewed):
    p=validate(read(PROFILE));require(re.fullmatch('[0-9a-f]{40}',reviewed or '')is not None and git('rev-parse','HEAD')==reviewed,'exact reviewed source required')
    paths=[*p['protected_git_objects'],*p['owned_sha256'],str(PROFILE.relative_to(ROOT))]
    require(not git('status','--porcelain','--',*paths),'dirty source')
    for name,obj in p['protected_git_objects'].items():
        require(git('rev-parse',FROZEN+':'+name)==obj and git('rev-parse','HEAD:'+name)==obj,'frozen1f object differs: '+name)
        if (ROOT/name).is_file():require(git('hash-object','--',name)==obj,'frozen worktree differs: '+name)
    for name,h in p['owned_sha256'].items():require(sha(ROOT/name)==h,'owned caller source differs: '+name)
    return p


def service(reviewed):
    require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('GITHUB_SHA')==reviewed
            and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','genuine hosted source required')
    require(os.environ.get('GH_TOKEN')and os.environ.get('GITHUB_RUN_ID'),'existing repository service credential required')
    req=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/runs/'+os.environ['GITHUB_RUN_ID'],
                              headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
    try:
        with urllib.request.urlopen(req,timeout=60)as r:data=r.read(2*2**20+1)
    except Exception:raise ValueError('hosted service admission rejected')from None
    require(len(data)<=2*2**20,'service metadata bound exceeded');run=json.loads(data)
    require(run['id']==int(os.environ['GITHUB_RUN_ID'])and run['head_sha']==reviewed and run['status']=='in_progress'
            and run['event']=='workflow_dispatch'and run['path']=='.github/workflows/moedern-q8-controls-v1.yml','actual service/workflow/source differs')
    return dict(run=run['id'],source=run['head_sha'],attempt=run['run_attempt'],workflow=run['path'])


def resources(path):
    require(platform.system()=='Linux'and platform.machine()=='x86_64'and hasattr(os,'wait4'),'Linux x64 wait4 controls required')
    ram=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
    disk=shutil.disk_usage(path).free
    require(ram>=4*2**30 and disk>=8*2**30 and os.cpu_count()>=2,'model-free4GiB RAM/8GiB disk/two CPU required')
    return dict(available_ram_bytes=ram,free_disk_bytes=disk,threads=2,models_accessed=False,
                old_model_14gib_24gib_guard_unchanged=True)


def build_profile(cache,commands):
    for name,value in EXPECTED_CACHE.items():require(re.search(r'^'+name+r':[^=]+='+value+r'$',cache,re.M),'actual CMake profile differs: '+name)
    names={'gguf_loader.cpp','moedern_text_engine.cpp','moedern_family_engine.cpp','moedern_text_frontend.cpp','moedern_text.cpp','moedern_text_api.cpp'}
    rows=[r for r in commands if Path(r['file']).name in names]
    require(len(rows)==6 and {Path(r['file']).name for r in rows}==names
            and all('-ffp-contract=off'in r['command']and not any(x in r['command']for x in ['-ffast-math','-Ofast','-ffp-contract=fast'])for r in rows),
            'frozen MoE arithmetic build/contraction differs')
    require(any(Path(r['file']).name=='quantize.cpp'for r in commands)and any(Path(r['file']).name=='controls.cpp'for r in commands),'actual controls/quantizer compilation absent')
    return dict(cache=EXPECTED_CACHE,library_commands=rows,fixture_sources_only=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reviewed-source',required=True)
    for name in ['out','build']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();review=a.out/'review';raw=a.out/'raw';review.mkdir(parents=True,exist_ok=True);raw.mkdir(exist_ok=True)
    report=dict(schema='crispembed-moedern-q8-model-free-controls-v1',source=a.reviewed_source,completed=False,
                models_accessed=False,corpus_access=False,model_qualified=False,publication_qualified=False,commands={},phase='source')
    def save():write(review/'controls.json',report)
    save();code=1
    try:
        cfg=verify(a.reviewed_source);report['source_profile_sha256']=sha(PROFILE)
        report['service']=service(a.reviewed_source);report['resources']=resources(a.out)
        require(not a.build.exists(),'fresh task-owned hosted build required')
        require(git('-C',str(ROOT/'ggml'),'rev-parse','HEAD')==cfg['protected_git_objects']['ggml']
                and not git('-C',str(ROOT/'ggml'),'status','--porcelain','--untracked-files=no'),'actual ggml checkout differs/dirty')
        # Existing1f guard remains unmodified and independently executes too.
        sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/moedern-colbert')]
        import q8_source_contract
        from ci_process import run
        report['frozen_q8_source_closure']=q8_source_contract.verify(a.reviewed_source);save()
        env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')}
        env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        def command(name,argv,timeout):
            report['phase']=name;save();log=review/(name+'.log');result=run(argv,log,cwd=ROOT,env=env,timeout=timeout)
            report['commands'][name]=dict(result,argv=list(map(str,argv)),log=log.name,**identity(log));save()
            require(not result['timed_out']and result['exit']==0,'hosted control rejected: '+name)
        command('compiler',['c++','--version'],30)
        command('configure',['cmake','-S',ROOT/'tests/moedern-q8','-B',a.build,'-DCMAKE_BUILD_TYPE=Release'],120)
        command('build',['cmake','--build',a.build,'--target','moedern-q8-controls','moedern-q8-quantize','--parallel','2'],2400)
        cache=(a.build/'CMakeCache.txt').read_text();commands=read(a.build/'compile_commands.json')
        report['build_profile']=build_profile(cache,commands)
        shutil.copyfile(a.build/'CMakeCache.txt',review/'CMakeCache.txt');write(review/'compile-commands.json',commands)
        for name in ['moedern-q8-controls','moedern-q8-quantize','libmoedern-q8.so']:
            source=a.build/name;require(source.is_file()and not source.is_symlink(),'actual compiled binary missing')
            target=raw/'binaries'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            report.setdefault('binaries',{})[name]=identity(target)
        symbols=subprocess.check_output(['nm','-C',str(a.build/'libmoedern-q8.so')],text=True)
        require(not any(x in symbols for x in ['moedern_text_test::','moedern_text_native::test::','encode_observed']), 'production test observer leaked')
        (review/'library-symbols.txt').write_text(symbols)
        report['plain_observer_exclusion']=True;save()
        command('controls',[a.build/'moedern-q8-controls'],180)
        require('PASS3 named Q8 metadata profiles,9 storage mutations' in (review/'controls.log').read_text()
                and '6 actual Q8 direct/expert graphs' in (review/'controls.log').read_text(),'complete authored control coverage absent')
        command('actual-c-quantizer-fixture',[sys.executable,'-I',ROOT/'tests/moedern-q8/quantizer_fixture.py',
                                             '--quantizer',a.build/'moedern-q8-quantize','--out',raw/'fixture'],180)
        fixture=read(raw/'fixture/fixture.json')
        require(fixture['completed']is True and fixture['model_qualified']is False and fixture['actual_c_quantizer']is True
                and fixture['actual_q8_tensors']==2 and fixture['critical_f32_tensors']==7 and fixture['general_file_type']==7,
                'actual C fixture identity/coverage differs')
        require(fixture['quantizer_sha256']==report['binaries']['moedern-q8-quantize']['sha256'],'fixture used different quantizer')
        write(review/'actual-c-quantizer-fixture.json',fixture)
        report.update(completed=True,phase='model-free-controls-complete',typed_profile_counts={'1.5M':20,'27M':20,'115M':52},
                      actual_c_fixture=dict(q8_tensors=2,critical_f32_tensors=7),authored_direct_expert_graphs=6,
                      full_model_quantization_executed=False,strict_f32_corpus_blockers_unchanged=True)
        save();code=0
    except Exception as error:report['error']=type(error).__name__+': '+str(error);save()
    finally:
        files={str(f.relative_to(a.out)):identity(f)for f in sorted(a.out.rglob('*'))if f.is_file()and f.name!='manifest.json'}
        write(review/'manifest.json',files)
    return code


if __name__=='__main__':raise SystemExit(main())
