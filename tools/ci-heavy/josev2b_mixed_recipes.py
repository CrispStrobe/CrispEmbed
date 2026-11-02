#!/usr/bin/env python3
"""Hosted four-case A/B/C development replay, never original/NEW20 forwards.

Native build/panel control body adapted from frozen9c475d0d worker.main. Pure
low-level original fetch/typed/command functions are reused without invoking any
old source guard or changing its globals. A separately closed protocol owns this
new caller. Numerical negatives are retained while later recipes still execute.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import site
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import josev2b_mixed_contract as closure
import quantize_josev2b_mixed as packer
import josev2b_composed_panel as panel
spec=importlib.util.spec_from_file_location('josev_frozen_q4_worker',ROOT/'tools/kaggle/josev-q4k/worker.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
require=closure.require
command=old.command
original=old.original
fetch_f32=old.fetch_f32
typed_decisions=old.typed_decisions
GATES=old.GATES
QUANT=old.QUANT


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.partial')
    with temporary.open('w')as f:
        json.dump(value,f,indent=2,allow_nan=False,ensure_ascii=False);f.write('\n');f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def resources(scratch):
    require(os.environ.get('GITHUB_ACTIONS')=='true','genuine GHA CPU development task required')
    require(platform.system()=='Linux'and platform.machine()=='x86_64'and hasattr(os,'wait4'),'Linux x64/wait4 required')
    available=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024
    free=shutil.disk_usage(scratch).free
    require(available>=14*2**30 and free>=64*2**30 and os.cpu_count()>=2,'need14GiB available RAM/64GiB disk/two CPUs before payload')
    return dict(available_ram_bytes=available,free_disk_bytes=free,cpu_count=os.cpu_count(),threads=2,backend='cpu',gpu_execution=False)


def fetch_reference(destination):
    cfg=closure.verify();resources(destination.parent)
    require(os.environ.get('GITHUB_SHA')==closure.git('rev-parse','HEAD').decode().strip(),'service checkout differs')
    spec=importlib.util.spec_from_file_location('josev_frozen_fetch',ROOT/'tools/ci-heavy/josev2b_composed.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    report=json.loads((ROOT/'tests/results/josev2b/original-reference.json').read_text())
    names={'review/artifact.json','original/reference.json','original/input-freeze.json'}|{'original/'+r['file']for r in report['arrays']}
    destination.mkdir(parents=True,exist_ok=False)
    directory,proof=helper.fetch(cfg['original_artifact'],destination,names)
    receipt=json.loads((directory/'review/artifact.json').read_text())
    require(receipt['completed']is True and receipt['arrays']==251 and receipt['reference_sha256']==closure.REFERENCE_SHA,'original archive receipt differs')
    for name in names-{'review/artifact.json'}:
        path=directory/name;pin=receipt['files'][name]
        require(path.is_file()and not path.is_symlink()and path.stat().st_size==pin['bytes']and closure.sha(path)==pin['sha256'],'original archive member differs: '+name)
    write(destination/'service-proof.json',dict(artifact=proof,selected_files=sorted(names),original_rerun=False))


def main(args):
    packer.hosted();require(args.reviewed_source==closure.git('rev-parse','HEAD').decode().strip(),'explicit reviewed source differs')
    cfg=closure.verify();require(os.environ.get('GITHUB_SHA')==args.reviewed_source,'service source differs');args.scratch.mkdir(parents=True,exist_ok=False);args.out.mkdir(parents=True,exist_ok=False)
    review=args.out/'review';raw=args.out/'raw';review.mkdir();raw.mkdir()
    result=dict(schema='crispembed-josev-mixed-hosted-v1',source=args.reviewed_source,completed=False,
                default_enabled=False,publication_qualified=False,independent_task_quality=False,commands={},phase='preflight')
    def save():write(review/'hosted.json',result)
    result['protocol_sha256']=closure.sha(closure.PATH)
    write(review/'source-closure.json',cfg)
    save()
    try:
        result['resources']=resources(args.scratch)
        require(sys.version_info[:2]==(3,13)and sys.prefix!=sys.base_prefix and site.ENABLE_USER_SITE is False and sys.flags.isolated and
                importlib.metadata.version('numpy')=='2.2.6'and importlib.util.find_spec('torch')is None,'isolated CPython3.13/NumPy2.2.6 and no Torch required')
        from josev2b_environment import owner_metadata_file
        package=next(r for r in json.loads((ROOT/'docs/josev2b-environment-lock.json').read_text())['packages']if r['name']=='numpy')
        owners=[d for d in importlib.metadata.distributions()if d.metadata['Name'].lower().replace('_','-')=='numpy']
        require(len(owners)==1,'unique NumPy distribution owner required')
        owner=Path(owners[0].locate_file(owner_metadata_file(owners[0],'numpy','2.2.6'))).resolve()
        require(owner.is_relative_to(Path(sys.prefix).resolve())and owner.stat().st_size==package['core_metadata_bytes']and
                closure.sha(owner)==package['core_metadata_sha256'],'installed NumPy owner metadata/origin differs')
        require(subprocess.check_output(['git','-C',str(ROOT/'ggml'),'rev-parse','HEAD'],text=True).strip()==cfg['ggml']and
                not subprocess.check_output(['git','-C',str(ROOT/'ggml'),'status','--porcelain','--untracked-files=no']),
                'actual backend checkout dirty or differs')
        import numpy as np
        require(Path(np.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'NumPy imported outside isolated environment')
        result['environment']=dict(python=list(sys.version_info[:3]),numpy=np.__version__,isolated_prefix=True,torch_imported=False,
                                  numpy_owner_metadata_sha256=closure.sha(owner))
        result['phase']='verify-original';save()
        freeze,gold,result['original']=original(args.mount,args.out/'original');save()
        env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')}
        env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        build=args.scratch/'build'
        for name,cmd in [('configure',['cmake','-S',ROOT/'tests/josev-q4k','-B',build,'-DCMAKE_BUILD_TYPE=Release']),
                         ('build',['cmake','--build',build,'--parallel','2']),
                         ('profile-controls',[build/'josev-q4k-profile-controls']),
                         ('metadata-controls',[build/'test-josev-q4k']),
                         ('gates-plain',[build/'josev-q4k-plain-probe','--gates']),
                         ('gates-observed',[build/'josev-q4k-observed-probe','--gates'])]:
            result['commands'][name]=command(cmd,review/(name+'.log'),env);save()
        require('GGML_LLAMAFILE:BOOL=ON'in(build/'CMakeCache.txt').read_text(),'actual root LLAMAFILE ON required')
        commands=json.loads((build/'compile_commands.json').read_text())
        arithmetic=[r for r in commands if Path(r['file']).name in {'josev2b_backbone.cpp','josev2b_composed.cpp','josev2b_head.cpp','qwen35_decoder.cpp','josev2b_answer_serializer.cpp','josev2b_text_frontend.cpp','shortest_double.c'}]
        require(len(arithmetic)==14 and all('-ffp-contract=off'in r['command']and not any(k in r['command']for k in('-ffast-math','-Ofast','-ffp-contract=fast'))for r in arithmetic),'ordinary/observed arithmetic build differs')
        write(review/'compile-commands.json',arithmetic)
        plain,observed=build/'libjosev-q4k-plain.a',build/'libjosev-q4k-observed.a'
        symbols=subprocess.check_output(['nm','-C',str(plain)],text=True);instrumented=subprocess.check_output(['nm','-C',str(observed)],text=True)
        require(not any(s in symbols for s in('composed_test::','josev2b_head::test::','run_observed'))and
                'head/option_context_projection'not in subprocess.check_output(['strings',str(plain)],text=True),'production observer/probe leakage')
        require('josev2b::composed_test::observe'in instrumented and 'josev2b_head::test::run_observed'in instrumented,'observer positive control missing')
        result['archives']=dict(plain_sha256=closure.sha(plain),observed_sha256=closure.sha(observed),ordinary_excludes_observers=True)
        (review/'ordinary-symbols.txt').write_text(symbols)
        result['ref_controls']=panel.ref_controls(build/'josev-q4k-diff',review/'ref-controls');save()
        result['phase']='authenticate-published-F32';save();model=args.scratch/'published-f32.gguf';result['model']=fetch_f32(model);save()
        requests=raw/'requests.json';write(requests,[dict(id=r['id'],request_json=json.dumps(r['request'],ensure_ascii=False,allow_nan=False,separators=(',',':')))for r in freeze['records']])
        enabled=dict(env,**{k:'1'for k in GATES})
        def execute(label,path,active):
            outputs={};producer={}
            for mode in ('plain','observed','repeat'):
                destination=raw/(label+'-'+mode);outputs[mode]=destination
                producer[mode]=command([build/('josev-q4k-'+('plain'if mode=='plain'else'observed')+'-probe'),path,requests,destination],review/(label+'-'+mode+'.log'),active,allowed=(0,1))
            comparison=panel.compare(args.out/'original',freeze,gold,outputs,build/'josev-q4k-diff',raw/(label+'-comparison'))
            write(review/(label+'-panel.json'),comparison)
            structural=(all(r['exit']==0 for r in producer.values())and comparison['compared_stages']==227 and
                        all(c.get('completed')and c.get('ordinary_observer_repeat_raw_equal')and c['actual_ref']['all_metrics_agree']for c in comparison['cases']))
            require(structural,'structural/input/observer/metric consistency rejected for '+label)
            numeric=all(all(m['passed']for m in c['metrics'])and all(v['passed']for v in c['raw_finals'])and
                        all(v['passed']for v in c['ordinal'])for c in comparison['cases'])
            decisions=typed_decisions(freeze,gold,outputs['plain']);write(review/(label+'-typed.json'),decisions)
            result[label]=dict(commands=producer,compared_stages=227,structural_passed=True,
                               strict_original_numeric_passed=numeric,
                               complete_typed_response_exact=all(c['complete_response_exact']for c in comparison['cases']),
                               strict_original_f32_numeric_and_typed_passed=comparison['numeric_and_complete_typed_parity'],
                               quant_task_decisions=decisions);save();return comparison
        result['phase']='strict-original-F32-control';save()
        control=execute('f32',model,enabled);require(control['numeric_and_complete_typed_parity'],'strict227/four-typed F32 control failed; candidate forbidden')
        quant_env=dict(enabled,**{k:'1'for k in QUANT});result['recipes']={}
        for identity in ('A','B','C'):
            label='recipe-'+identity
            result['phase']='convert-'+label;save();candidate=raw/('josev2b-mixed-'+identity+'.gguf')
            conversion=packer.quantize(model,candidate,build/'josev-q4k-quantize',identity)
            write(review/(label+'-conversion.json'),conversion);result['recipes'][identity]=dict(conversion=conversion);save()
            rejected={}
            for gate in QUANT:
                for value in (None,'','0'):
                    off=dict(quant_env)
                    if value is None:off.pop(gate)
                    else:off[gate]=value
                    suffix='unset'if value is None else'empty'if value==''else'zero'
                    key=gate+'='+suffix
                    rejected[key]=command([build/'josev-q4k-plain-probe',candidate,requests,
                        raw/(label+'-off-'+gate+'-'+suffix)],review/(label+'-off-'+gate+'-'+suffix+'.log'),off,allowed=(1,))
            result['recipes'][identity]['quant_gate_rejections']=rejected;save()
            result['phase']='native-'+label;save();execute(label,candidate,quant_env)
            result['commands'][label+'-recovery']=command([build/'josev-q4k-recovery',candidate,requests],review/(label+'-recovery.log'),quant_env)
            result['recipes'][identity]['panel']=result[label];save()
        manifest={}
        for path in sorted(args.out.rglob('*')):
            if path.is_file()and not path.is_symlink()and path!=review/'hosted.json':
                manifest[str(path.relative_to(args.out))]=dict(bytes=path.stat().st_size,sha256=closure.sha(path))
        write(review/'retained-files.json',manifest)
        result['retained_files_manifest_sha256']=closure.sha(review/'retained-files.json')
        result.update(completed=True,phase='development-diagnostic-complete',publication_qualified=False,
                      holdout_native_outputs=False,recipe_selected=False,
                      limits='Four old development controls only. All three recipes retained; no NEW20 access/selection. Strict failures remain negative.')
        save();return result
    except Exception as error:
        result.update(phase='rejected',error=type(error).__name__+': '+str(error));save();raise


if __name__=='__main__':
    def terminate(_signal,_frame):raise RuntimeError('hosted worker termination requested')
    signal.signal(signal.SIGTERM,terminate)
    p=argparse.ArgumentParser();p.add_argument('--fetch-reference',type=Path)
    p.add_argument('--reviewed-source')
    for name in ('mount','scratch','out'):p.add_argument('--'+name,type=Path)
    a=p.parse_args()
    if a.fetch_reference:fetch_reference(a.fetch_reference)
    else:
        require(a.reviewed_source and all(getattr(a,n)for n in ('mount','scratch','out')),'explicit source/mount/scratch/output required')
        main(a)
