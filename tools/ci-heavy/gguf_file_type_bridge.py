#!/usr/bin/env python3
"""Hosted old-four metadata bridge, prepared without model dispatch.

Inputs are pre-staged and authenticated here. Reuses low-level frozen original
and metric controls without changing historical guards or repeating originals.
The caller stages/builds the exact current-source public C ABI library first.
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
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import gguf_file_type_contract as closure
import gguf_file_type_canonicalize as canonicalizer
import gguf_file_type_public_bridge as public_bridge
import quantize_josev2b_q4k as packer
import josev2b_composed_panel as panel
spec=importlib.util.spec_from_file_location('old_q4_functions',ROOT/'tools/kaggle/josev-q4k/worker.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
require=closure.require
write=old.write
command=old.command
original=old.original
typed_decisions=old.typed_decisions
GATES=old.GATES
QUANT=old.QUANT


def resources(scratch):
    require(os.environ.get('GITHUB_ACTIONS')=='true'and platform.system()=='Linux'and platform.machine()=='x86_64','hosted Linux x64 CPU bridge required')
    available=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024
    disk=shutil.disk_usage(scratch).free
    require(available>=14*2**30 and disk>=64*2**30 and os.cpu_count()>=2,'14GiB available RAM/64GiB disk/two CPUs required')
    return dict(available_ram_bytes=available,free_disk_bytes=disk,cpu_count=os.cpu_count(),threads=2)


def main(args):
    packer.hosted();require(args.reviewed_source==closure.git('rev-parse','HEAD').decode().strip(),'explicit reviewed source differs')
    cfg=closure.verify(args.reviewed_source);args.scratch.mkdir(parents=True,exist_ok=False);args.out.mkdir(parents=True,exist_ok=False)
    review=args.out/'review';raw=args.out/'raw';review.mkdir();raw.mkdir()
    result=dict(schema='crispembed-gguf-file-type-same-payload-v1',source=args.reviewed_source,completed=False,
                default_enabled=False,publication_qualified=False,independent_task_quality=False,commands={},phase='preflight')
    def save():write(review/'hosted.json',result)
    save()
    try:
        result['resources']=resources(args.scratch)
        descriptor=canonicalizer.descriptor(json.loads(args.descriptor.read_text()),args.producer_report)
        require(descriptor['family']=='josev'and descriptor['architecture']=='josev2b_systemone','old-four bridge only supports exact Josev profile')
        require(args.source_model.is_file()and not args.source_model.is_symlink()and
                args.source_model.stat().st_size==descriptor['source_model']['bytes']and
                closure.sha(args.source_model)==descriptor['source_model']['sha256'],'selected legacy identity differs before native forwards')
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
        result['phase']='authenticate-published-F32';save();model=args.f32_model;require(model.stat().st_size==packer.SOURCE['bytes']and closure.sha(model)==packer.SOURCE['sha256'],'exact staged F32 model required');result['model']=packer.SOURCE;save()
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
        result['phase']='canonicalize-reviewed-candidate';save()
        candidate=raw/'canonical-model.gguf'
        descriptor=json.loads(args.descriptor.read_text())
        migration=canonicalizer.canonicalize(args.source_model,candidate,descriptor,args.producer_report)
        migration.update(runtime_source=args.reviewed_source,source_profile_sha256=closure.sha(closure.PATH),
                         input_descriptor_file_sha256=closure.sha(args.descriptor))
        write(review/'canonicalization.json',migration);result['canonicalization_report_sha256']=closure.sha(review/'canonicalization.json');save()
        quant_env=dict(enabled,**{k:'1'for k in QUANT})
        legacy=execute('legacy',args.source_model,quant_env);canonical=execute('canonical',candidate,quant_env)
        exact=[]
        for index,record in enumerate(freeze['records']):
            modes=[]
            for mode in ('plain','observed','repeat'):
                left=raw/('legacy-'+mode);right=raw/('canonical-'+mode)
                a=panel.manifest(left/str(index));b=panel.manifest(right/str(index))
                require(set(a)==set(b),'same-payload stage coverage differs')
                for key,row in a.items():
                    other=b[key]
                    require(row['shape']==other['shape']and closure.sha(left/str(index)/row['file'])==closure.sha(right/str(index)/other['file']),
                            'same-payload captured stage bytes differ')
                aj=json.loads((left/'native.jsonl').read_text().splitlines()[index]);bj=json.loads((right/'native.jsonl').read_text().splitlines()[index])
                require(aj['response']==bj['response'],'same-payload typed response bytes differ')
                modes.append(dict(mode=mode,stages=len(a),all_payload_sha_exact=True,response_bytes_exact=True))
            exact.append(dict(id=record['id'],modes=modes))
        require(legacy['numeric_and_complete_typed_parity']==canonical['numeric_and_complete_typed_parity'],'metadata changed strict numeric verdict')
        result['phase']='actual-public-capi-continuity';save()
        public_env={k:v for k,v in os.environ.items()if k.startswith('CRISPEMBED_')}
        try:
            for key in list(os.environ):
                if key.startswith('CRISPEMBED_'):os.environ.pop(key)
            os.environ.update(quant_env)
            result['public_capi']=public_bridge.compare(args.library,json.loads(args.library_proof.read_text()),args.reviewed_source,
                                                       args.source_model,candidate,freeze)
        finally:
            for key in list(os.environ):
                if key.startswith('CRISPEMBED_'):os.environ.pop(key)
            os.environ.update(public_env)
        for label,path in (('legacy',args.source_model),('canonical',candidate)):
            result['commands'][label+'-recovery']=command([build/'josev-q4k-recovery',path,requests],review/(label+'-recovery.log'),quant_env)
        result.update(completed=True,phase='metadata-only-continuity-complete',source=args.reviewed_source,
            runtime_source=args.reviewed_source,source_profile_sha256=closure.sha(closure.PATH),source_model=migration['source_model'],
            canonical_model=migration['canonical_model'],four_cases=exact,all_mapped_stage_bytes_exact=True,
            all_typed_response_bytes_exact=True,all_public_capi_response_bytes_exact=True,
            old_numeric_results_replaced=False,numerical_qualified=False,publication_qualified=False)
        save();return result
    except Exception as error:
        result.update(phase='rejected',error=type(error).__name__+': '+str(error));save();raise


if __name__=='__main__':
    def terminate(_signal,_frame):raise RuntimeError('hosted bridge termination requested')
    signal.signal(signal.SIGTERM,terminate)
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True)
    for name in ('mount','scratch','out','descriptor','producer-report','source-model','f32-model','library','library-proof'):
        p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
