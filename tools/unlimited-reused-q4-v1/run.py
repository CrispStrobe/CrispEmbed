"""Genuine hosted build/profile, then only the authenticated retained last1 panel."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import lane as c
from process import run as child
import overlay
import build_profile


def run(a):
    cfg=c.guard(a.source,True);a.out.mkdir(parents=True,exist_ok=False);a.build.mkdir(parents=True,exist_ok=False)
    deadline=time.monotonic()+cfg['panel_seconds']
    report=dict(schema='crispembed-unlimited-reused-q4-hosted-v1',source=a.source,completed=False,commands={},
        original_forwards=0,quantizer_runs=0,publication_qualified=False,production_qualified=False)
    save=lambda:c.atomic(a.out/'hosted.json',report);save()
    env={k:v for k,v in os.environ.items()if not k.startswith(('CRISPEMBED_','UOCR'))}
    env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def command(name,argv,timeout=3600,allowed=(0,)):
        remaining=deadline-time.monotonic();c.require(remaining>0,'declared hosted deadline reached')
        log=a.out/(name+'.log');status=child(argv,log,cwd=c.ROOT,env=env,timeout=min(timeout,remaining))
        report['commands'][name]=dict(status,log=log.name,sha256=c.digest(log));save()
        c.require(not status['timed_out']and status['exit']in allowed,'hosted source/build/process rejection: '+name)
    try:
        report['resources']=c.resources(a.build,True);proof=json.loads((a.auth/'inputs.json').read_text())
        c.require(proof['completed']is True and proof['source']==a.source and proof['all_544_authenticated_before_array_image_import']is True,'full544 byte barrier required before builds/models')
        report['inputs_sha256']=c.digest(a.auth/'inputs.json');save()
        plain=a.build/'plain';observed=a.build/'observed';tree=a.build/'observed-source'
        command('configure-plain',['cmake','-S',c.ROOT/'tests/unlimited-q4k','-B',plain,'-DCMAKE_BUILD_TYPE=Release','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF'])
        command('compile-plain',['cmake','--build',plain,'--target','crispembed-cli','unlimited-q4-diff','unlimited-q4-projection','--parallel','2'],5400)
        command('observer-worktree',['git','-c','gc.auto=0','worktree','add','--detach',tree,a.source])
        command('observer-backend',['git','-C',tree,'submodule','update','--init','--depth','1','ggml'])
        patched=overlay.transform((c.ROOT/'src/unlimited_ocr.cpp').read_bytes());(tree/'src/unlimited_ocr.cpp').write_bytes(patched)
        c.require(subprocess.check_output(['git','-C',tree,'diff','--name-only'],text=True).splitlines()==['src/unlimited_ocr.cpp'],'observer overlay delta differs')
        report['overlay']=dict(production_source_sha256=c.digest(c.ROOT/'src/unlimited_ocr.cpp'),test_overlay_sha256=hashlib.sha256(patched).hexdigest(),source=a.source,arithmetic_changes=False);save()
        command('configure-observed',['cmake','-S',tree,'-B',observed,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DCRISPEMBED_NATIVE=OFF','-DGGML_NATIVE=OFF','-DGGML_LLAMAFILE=ON','-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF'])
        command('compile-observed',['cmake','--build',observed,'--target','crispembed-cli','--parallel','2'],5400)
        profile=build_profile.verify(plain,observed,c.ROOT,tree);c.atomic(a.out/'build-profile.json',profile)
        symbols=subprocess.check_output(['nm','-C',plain/'root/crispembed'],text=True)
        instrumented=subprocess.check_output(['nm','-C',observed/'crispembed'],text=True)
        strings=subprocess.check_output(['strings',plain/'root/crispembed'],text=True)
        c.require('uocr_q4_audit::'not in symbols and 'UOCR_Q4_AUDIT_DIR'not in strings and 'uocr_q4_audit::'in instrumented,'ordinary observer exclusion/positive control rejected')
        report.update(build_profile_sha256=c.digest(a.out/'build-profile.json'),production_observer_excluded=True);save()
        command('actual-ref-controls',[sys.executable,'-I',c.ROOT/'tools/unlimited-ocr2-q4k/controls.py','--diff',plain/'unlimited-q4-diff','--output',a.out/'ref-controls'])
        control=a.build/'published-f16.gguf';command('published-f16',c.frozen.model_download_command(control),2400)
        c.require(control.stat().st_size==cfg['control']['bytes']and c.digest(control)==cfg['control']['sha256'],'published F16 exact extent/SHA differs')
        remaining=int(deadline-time.monotonic())-60;c.require(remaining>120,'insufficient declared time before any full forward')
        command('paired-last1',[sys.executable,'-I',c.ROOT/'tools/unlimited-reused-q4-v1/replay.py','--source',a.source,
            '--auth',a.auth,'--inputs',a.inputs,'--control',control,'--out',a.raw,'--cli',plain/'root/crispembed',
            '--observed',observed/'crispembed','--diff',plain/'unlimited-q4-diff','--projection',plain/'unlimited-q4-projection',
            '--budget-seconds',str(remaining)],remaining+30,allowed=(0,1))
        result=json.loads((a.raw/'panel.json').read_text());report['panel_sha256']=c.digest(a.raw/'panel.json');report['panel_completed']=result['completed']
        report['actual_calls']=result['actual_calls'];report['all_numeric_passed']=result.get('all_numeric_passed',False);report['all_task_quality_passed']=result.get('all_task_quality_passed',False)
        c.require(result['completed']is True,'paired panel structurally incomplete; partial evidence retained')
        report['completed']=True;save();return 0 if report['all_numeric_passed']and report['all_task_quality_passed']else 1
    except Exception as error:report['error']=type(error).__name__+': '+str(error);save();raise
    finally:
        # Small evidence copies retain partial state after either structural or numerical failure.
        evidence={}
        for p in a.raw.rglob('*')if a.raw.exists()else():
            if p.is_file()and (p.suffix=='.json'or p.name in ('stdout','stderr')):
                c.require(p.stat().st_size<=4*2**20,'small paired evidence exceeds bound')
                dest=a.out/'evidence'/p.relative_to(a.raw);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
                evidence[str(dest.relative_to(a.out))]=dict(bytes=dest.stat().st_size,sha256=c.digest(dest))
        c.atomic(a.out/'evidence-files.json',evidence);save()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True)
    for n in ('auth','inputs','build','out','raw'):p.add_argument('--'+n,type=Path,required=True)
    raise SystemExit(run(p.parse_args()))
