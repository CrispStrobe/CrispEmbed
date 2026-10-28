#!/usr/bin/env python3
"""One real hosted CPU mixed-Q4_K experiment, never an original rerun/publisher."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import site
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import josev2b_q4k_contract as closure
import quantize_josev2b_q4k as packer
import josev2b_composed_panel as panel

GATES=('CRISPEMBED_JOSEV_COMPOSED','CRISPEMBED_JOSEV_BACKBONE','CRISPEMBED_QWEN35_OCR',
       'CRISPEMBED_JOSEV_HEAD','CRISPEMBED_JOSEV_TEXT_FRONTEND','CRISPEMBED_JOSEV_ANSWER_SERIALIZER',
       'CRISPEMBED_JOSEV_PACKED')
QUANT=('CRISPEMBED_JOSEV_QUANT','CRISPEMBED_QWEN35_QUANT')
require=closure.require


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n')


def resources(scratch):
    packer.hosted()
    require(platform.system()=='Linux'and platform.machine()=='x86_64','bounded Linux x64 CPU task required')
    require(Path('/usr/bin/time').is_file(),'GNU time must be present before packages/model access')
    available=int(next(row.split()[1]for row in Path('/proc/meminfo').read_text().splitlines()if row.startswith('MemAvailable:')))*1024
    disk=shutil.disk_usage(scratch).free
    require(available>=14*2**30 and disk>=24*2**30 and os.cpu_count()>=2,'need14GiB available RAM/24GiB disk/two CPUs before model access')
    return dict(available_ram_bytes=available,free_disk_bytes=disk,cpu_count=os.cpu_count(),threads=2,backend='cpu',gpu_execution=False)


def command(args,log,env=None,*,allowed=(0,),timeout=2400):
    start=time.monotonic();timing=log.with_name(log.name+'.time')
    with log.open('w')as stream:
        child=subprocess.run(['/usr/bin/time','-f','maximum_rss_kib=%M','-o',str(timing),*map(str,args)],
                             cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    measured=re.search(r'^maximum_rss_kib=([0-9]+)$',timing.read_text(),re.M)
    require(measured is not None,'per-process memory metric missing')
    result=dict(exit=child.returncode,seconds=time.monotonic()-start,maximum_rss_kib=int(measured[1]),log=log.name,sha256=closure.sha(log))
    require(child.returncode in allowed,'process/structural rejection in '+log.name)
    return result


def original(mount,destination):
    """Read-only v5 payload identity, every251 authenticated arrays before model."""
    hits=[p for p in Path(mount).rglob('reference.json')if p.is_file()and not p.is_symlink()and
          p.stat().st_size==152274 and closure.sha(p)==closure.REFERENCE_SHA]
    require(len(hits)==1,'one exact frozen v5 reference mount required; no latest fallback')
    source=hits[0].parent;report=json.loads(hits[0].read_text());freeze=source/'input-freeze.json'
    require(closure.sha(freeze)==closure.FREEZE_SHA and len(report['arrays'])==251,'frozen input/251-array scope differs')
    require(hits[0].read_bytes()==(ROOT/'tests/results/josev2b/original-reference.json').read_bytes()and
            freeze.read_bytes()==(ROOT/'tests/results/josev2b/original-input-freeze.json').read_bytes(),'public frozen reference differs')
    for name,pin in report['source'].items():
        require(hashlib.sha256(closure.git('show',closure.ORIGINAL_SOURCE+':'+name)).hexdigest()==pin,'original worker code identity differs: '+name)
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(hits[0],destination/'reference.json');shutil.copyfile(freeze,destination/'input-freeze.json')
    names=set();total=0
    for row in report['arrays']:
        rel=PurePosixPath(row['file'])
        require(not rel.is_absolute()and '..'not in rel.parts and '\\'not in row['file']and row['file']not in names,'unsafe/duplicate original array')
        names.add(row['file']);path=source/rel
        require(path.is_file()and not path.is_symlink()and path.resolve().is_relative_to(source.resolve())and
                path.stat().st_size==row['bytes']and closure.sha(path)==row['sha256'],'frozen v5 array bytes differ')
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);total+=row['bytes']
    require(total==279087860,'original array payload bytes differ')
    return json.loads(freeze.read_text()),report,dict(version_label='v5',reference_sha256=closure.REFERENCE_SHA,
               freeze_sha256=closure.FREEZE_SHA,source=closure.ORIGINAL_SOURCE,arrays=251,array_bytes=total,
               original_rerun=False,original_payload_identity_verified=True)


def fetch_f32(path):
    """Immutable anonymous public input, streamed with exact bytes and SHA."""
    pin=packer.SOURCE;url=f"https://huggingface.co/{pin['repo']}/resolve/{pin['revision']}/{pin['filename']}"
    partial=path.with_suffix('.partial');h=hashlib.sha256();count=0
    require(not path.exists()and not partial.exists(),'fresh public F32 download required')
    try:
        with urllib.request.urlopen(url,timeout=180)as response,partial.open('xb')as target:
            require(response.status==200 and int(response.headers['Content-Length'])==pin['bytes'],'public payload declared length differs')
            while block:=response.read(4*2**20):
                count+=len(block);require(count<=pin['bytes'],'public payload exceeds declared length');h.update(block);target.write(block)
    except Exception:raise ValueError('public F32 transfer rejected; task-owned partial retained')from None
    require(count==pin['bytes']and h.hexdigest()==pin['sha256'],'public F32 payload differs')
    partial.replace(path);path.chmod(0o444)
    return dict(**pin,download_verified=True)


def typed_decisions(freeze,report,output):
    """Owned task decisions, separately from unchanged full F32 numerical flags."""
    import numpy as np
    entries=[json.loads(line)for line in (output/'native.jsonl').read_text().splitlines()]
    require(len(entries)==4,'four raw JSON responses required');cases=[]
    for i,(record,target,native)in enumerate(zip(freeze['records'],report['cases'],entries)):
        require(native['completed']is True and native['id']==record['id']==target['id'],'native typed response rejected/out of order')
        response=json.loads(native['response']);expected=target['official_response'];fields=[]
        require(set(response)==set(expected)and response['model']==expected['model']and response['usage']==expected['usage']and
                list(response['answers'])==list(expected['answers']),'typed response structure/model/usage/order differs')
        captured=panel.manifest(output/str(i))
        for f,(question,field)in enumerate(zip(record['encoded']['questions'],target['fields'])):
            require(question['question_id']==field['id'],'field order differs')
            declaration=captured['result/probabilities/'+str(f),0]
            p=np.fromfile(output/str(i)/declaration['file'],dtype='<f4');gold=np.asarray(field['probabilities'],dtype='<f4')
            require(p.shape==gold.shape and np.isfinite(p).all(),'typed probability shape/finiteness differs')
            discrete=int(p.argmax())==int(gold.argmax())
            expectation=None
            if question['question_type']==2:
                a=sum(n*float(v)for n,v in enumerate(gold));b=sum(n*float(v)for n,v in enumerate(p))
                expectation=dict(original=a,native=b,absolute_error=abs(a-b),budget=.001,passed=abs(a-b)<=.001)
            fields.append(dict(id=field['id'],type=question['question_type'],ordered_option_ids=question['option_ids'],
                               original_argmax=int(gold.argmax()),native_argmax=int(p.argmax()),
                               discrete_argmax_exact=discrete if question['question_type']in(0,1)else None,
                               ordinal_expectation=expectation))
        canonical=json.dumps(expected,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        cases.append(dict(id=record['id'],native_response_text=native['response'],original_response=expected,
                          complete_typed_json_exact=response==expected,original_canonical_json_bytes_exact=native['response']==canonical,
                          fields=fields))
    discrete=all(f['discrete_argmax_exact']is not False for c in cases for f in c['fields'])
    ordinal=all(f['ordinal_expectation']is None or f['ordinal_expectation']['passed']for c in cases for f in c['fields'])
    exact=all(c['complete_typed_json_exact']for c in cases)
    return dict(cases=cases,complete_typed_json_exact=exact,ordered_discrete_option_argmax_exact=discrete,
                ordinal_expectation_within_original_0_001=ordinal,passed=exact and discrete and ordinal,
                independent_task_quality=False,publication_qualified=False)


def main(args):
    packer.hosted();require(args.reviewed_source==closure.git('rev-parse','HEAD').decode().strip(),'explicit reviewed source differs')
    cfg=closure.verify();args.scratch.mkdir(parents=True,exist_ok=False);args.out.mkdir(parents=True,exist_ok=False)
    review=args.out/'review';raw=args.out/'raw';review.mkdir();raw.mkdir()
    result=dict(schema='crispembed-josev-q4k-hosted-v1',source=args.reviewed_source,completed=False,
                default_enabled=False,publication_qualified=False,independent_task_quality=False,commands={},phase='preflight')
    def save():write(review/'hosted.json',result)
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
        result['phase']='convert-protected-Q4_K';save();candidate=raw/'josev2b-systemone-q4k.gguf'
        result['conversion']=packer.quantize(model,candidate,build/'josev-q4k-quantize');write(review/'conversion.json',result['conversion']);save()
        quant_env=dict(enabled,**{k:'1'for k in QUANT});result['quant_gate_rejections']={}
        for gate in QUANT:
            for value in ('','0'):
                off=dict(quant_env);off[gate]=value
                result['quant_gate_rejections'][gate+'='+value]=command([build/'josev-q4k-plain-probe',candidate,requests,raw/('off-'+gate+'-'+str(len(value)))],review/('off-'+gate+'-'+str(len(value))+'.log'),off,allowed=(1,))
        result['phase']='candidate-Q4_K';save();execute('q4k',candidate,quant_env)
        result['commands']['recovery']=command([build/'josev-q4k-recovery',candidate,requests],review/'recovery.log',quant_env)
        result.update(completed=True,phase='diagnostic-complete',strict_q4_f32_gates_passed=result['q4k']['strict_original_numeric_passed'],
                      quant_task_decisions_passed=result['q4k']['quant_task_decisions']['passed'],publication_qualified=False)
        save();return result
    except Exception as error:
        result.update(phase='rejected',error=type(error).__name__+': '+str(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True)
    for name in ('mount','scratch','out'):p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
