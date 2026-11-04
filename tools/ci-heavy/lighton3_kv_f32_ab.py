#!/usr/bin/env python3
"""First-only retained-original CPU KV precision experiment; no original calls."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'models'),str(ROOT/'tools'),str(ROOT/'tools/ci-heavy')]
_spec=importlib.util.spec_from_file_location('frozen_eps_worker',Path(__file__).with_name('lighton3_pixtral_eps_ab.py'))
eps=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(eps)
archive=eps.archive
require=eps.require
file_sha,sha,write,trace_control,download=eps.file_sha,eps.sha,eps.write,eps.trace_control,eps.download
from ci_process import run as execute
PROTOCOL=ROOT/'docs/lighton3-kv-f32-ab-v1.json'
GATE='CRISPEMBED_LIGHTONOCR3_KV_F32'
CASE='form-transcription'
BASELINE='0473dd5e813fdf48d2bafbd16811271cf6c42599'
TARGET=dict(run_id=41889499814,artifact_id=13355422898,bytes=8831289731,
            sha256='b57f2b906da12f20fc1ef5462c4800b801a5349ed38eee790b3ecdbf1b057078',
            source=BASELINE,branch='fix/ocr-qualification-nov4',conclusion='failure')


def source_guard(protocol):
    require(protocol['schema']=='crispembed.lighton3.kv-f32-ab.v1' and protocol['baseline_source']==BASELINE,
            'KV protocol differs')
    require(protocol['original_artifact']==TARGET,'retained baseline differs')
    for key,value in dict(GITHUB_ACTIONS='true',GITHUB_REPOSITORY=archive.REPOSITORY,RUNNER_OS='Linux',
                          KV_SELECTED_SCRIPT='tools/ci-heavy/lighton3_kv_f32_ab.py',KV_SELECTED_RUNNER='ubuntu-latest',
                          KV_SELECTED_MODE='portable',KV_SELECTED_PIP='',KV_SELECTED_ARGS='').items():
        require(os.environ.get(key,'')==value,'closed KV dispatch differs: '+key)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(source==os.environ['GITHUB_SHA'] and not subprocess.check_output(
        ['git','status','--porcelain','--untracked-files=no'],cwd=ROOT),'source must be clean/exact')
    for name,digest in protocol['owned_sha256'].items():require(sha(ROOT/name)==digest,'owned source differs: '+name)
    for name,digest in protocol['protected_git_objects'].items():
        require(subprocess.check_output(['git','rev-parse','HEAD:'+name],cwd=ROOT,text=True).strip()==digest,
                'protected source differs: '+name)
    changed=subprocess.check_output(['git','diff','--name-only',BASELINE,'HEAD','--','src'],cwd=ROOT,text=True).splitlines()
    require(set(changed)=={'src/lightonocr.cpp','src/lightonocr3_profile.h'},'unreviewed runtime delta')
    old=archive.read_json((ROOT/'docs/lighton3-pixtral-epsilon-ab-v1.json').read_bytes())
    require(protocol['models']==old['models'],'frozen model hashes differ')
    form=next(row for row in old['fixture_cases'] if row['id']=='form')
    require(protocol['fixture']=={key:form[key] for key in ('image','sha256')},'frozen form differs')
    return source


def validate_metadata(run,item):
    for key,value in dict(id=TARGET['run_id'],head_sha=BASELINE,head_branch=TARGET['branch'],event='workflow_dispatch',
                          path='.github/workflows/heavy-cpu.yml',status='completed',conclusion='failure',run_attempt=1).items():
        require(run.get(key)==value,'retained producer differs: '+key)
    for key in ('repository','head_repository'):
        require(run.get(key,{}).get('full_name')==archive.REPOSITORY and run[key].get('id')==1210669686,
                'retained repository differs')
    for key,value in dict(id=TARGET['artifact_id'],name='lighton3-eps-raw-'+str(TARGET['run_id']),
                          size_in_bytes=TARGET['bytes'],digest='sha256:'+TARGET['sha256'],expired=False,
                          url=archive.API+'/actions/artifacts/'+str(TARGET['artifact_id']),
                          archive_download_url=archive.API+'/actions/artifacts/'+str(TARGET['artifact_id'])+'/zip').items():
        require(item.get(key)==value,'retained artifact differs: '+key)
    for key,value in dict(id=TARGET['run_id'],head_sha=BASELINE,head_branch=TARGET['branch'],
                          repository_id=1210669686,head_repository_id=1210669686).items():
        require(item.get('workflow_run',{}).get(key)==value,'retained artifact producer differs: '+key)


def selected(name):
    parts=Path(name).parts
    return ((len(parts)>=3 and parts[:2]==('references',CASE)) or
            (len(parts)>=5 and parts[0]==CASE and parts[1] in ('f32','f16') and parts[2]=='gate1'
             and parts[3] in ('forced','autonomous','plain'))) and Path(name).suffix in ('.f32','.npy','.json','.txt')


def extract_inputs(path,destination):
    require(not destination.exists(),'retained destination must be fresh');destination.mkdir()
    rows=[];total=0
    with zipfile.ZipFile(path) as z:
        archive.safe_members(z)
        for info in z.infolist():
            if info.is_dir() or not selected(info.filename):continue
            total+=info.file_size
            require(info.file_size<=512*2**20 and total<=3*2**30,'selected member budget exceeded')
            output=destination/info.filename;output.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as stream,output.open('xb') as target:shutil.copyfileobj(stream,target,2**20)
            require(output.stat().st_size==info.file_size,'retained member truncated')
            rows.append(dict(path=info.filename,bytes=info.file_size,sha256=file_sha(output)))
    require(bool(rows),'retained selection empty')
    return rows


def prefill_control(old,new):
    def names_in(directory):
        return sorted(p.name for p in directory.glob('*.f32') if not
                      (re.fullmatch(r'logits_step[0-9]+\.f32',p.name) and int(p.name[11:-4])>0))
    names=names_in(old)
    require(names==names_in(new) and bool(names),'KV prefill tensor inventory differs/empty')
    require('logits_step0.f32' in names,'prefill logits missing')
    rows=[]
    for name in names:
        require((old/name).stat().st_size==(new/name).stat().st_size and file_sha(old/name)==file_sha(new/name),
                'KV changed prefill bytes: '+name)
        rows.append(dict(name=name,bytes=(new/name).stat().st_size,sha256=file_sha(new/name)))
    return dict(all_bytes_exact=True,tensor_files=len(rows),tensors=rows)


def logit_probe(reference,native):
    import numpy as np
    from compare_lightonocr3 import read_reference_array,validate_reference
    meta=archive.read_json((reference/'reference.json').read_bytes());shapes,_=validate_reference(meta)
    r=read_reference_array(reference,'logits_step6',meta['arrays']['logits_step6'],shapes['logits_step6']).reshape(-1).astype(np.float64)
    path=native/'logits_step6.f32';a=np.fromfile(path,dtype='<f4').astype(np.float64)
    require(a.shape==r.shape and np.isfinite(a).all(),'probe vector invalid')
    def stats(values):
        top=np.argsort(-values,kind='stable')[:5]
        return dict(top5=[dict(id=int(i),logit=float(values[i])) for i in top],
                    top1_top2_margin=float(values[top[0]]-values[top[1]]),
                    logit_2918=float(values[2918]),logit_3377=float(values[3377]),
                    margin_2918_minus_3377=float(values[2918]-values[3377]))
    return dict(step=6,scope='full vocabulary on retained original forced prefix',elements=int(r.size),
                original=stats(r),native=stats(a),max_abs=float(np.abs(a-r).max()),
                native_vector_sha256=file_sha(path),reference_vector_sha256=file_sha(reference/'logits_step6.npy'),reference_array_manifest=meta['arrays']['logits_step6'])


def main():
    out=Path(os.environ['KV_OUT']);review=out/'review';raw=out/'raw';review.mkdir(parents=True,exist_ok=False);raw.mkdir()
    report=dict(schema='crispembed.lighton3.kv-f32-ab.v1',phase='first',case=CASE,rows=[],commands=[],
                original_forward_calls=0,native_request_processes=0,completed=False,**{'pass':False},
                default_changed=False,publication_qualified=False)
    def save():write(review/'summary.json',report)
    save()
    try:
        require(len(sys.argv)==1,'first-only worker accepts no arguments')
        protocol=archive.read_json(PROTOCOL.read_bytes());source=source_guard(protocol)
        report.update(source=source,protocol_sha256=sha(PROTOCOL));save()
        scratch=Path(os.environ['KV_SCRATCH']);require(not scratch.exists(),'fresh scratch required');scratch.mkdir(parents=True)
        memory=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        require(memory>=12*2**30 and shutil.disk_usage(scratch).free>=40*2**30,'need 12GiB RAM and40GiB disk')
        env={k:v for k,v in os.environ.items() if not k.startswith(('CRISPEMBED_LIGHTON','LOCR_')) and
             k not in ('CRISPEMBED_MAX_PIXELS','GH_TOKEN','GITHUB_TOKEN','HF_TOKEN','HF_TOKEN_2','KAGGLE_API_TOKEN','KAGGLE_ACCOUNT')}
        env.update(CRISPEMBED_BUILD_MODE='portable',CRISPEMBED_LIGHTONOCR3='1',CRISPEMBED_LIGHTONOCR_FORCE_CPU='1',
                   CRISPEMBED_LIGHTONOCR3_PIXTRAL_EPS='1',CRISPEMBED_LIGHTONOCR3_GROUNDING='0',OMP_NUM_THREADS='4',
                   MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',HF_HOME=str(scratch/'hf'),PIP_NO_CACHE_DIR='1')
        def command(name,argv,extra=None):
            result=execute(argv,review/(name+'.log'),cwd=ROOT,env=env|(extra or {}),timeout=3600)
            report['commands'].append(dict(name=name,**result,log=name+'.log'));save()
            require(not result['timed_out'] and result['exit']==0,'KV command failed: '+name)
            return result
        build=scratch/'build'
        command('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF',
                              '-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF',
                              '-DGGML_BLAS=OFF','-DCRISPEMBED_BUILD_SHARED=OFF'])
        command('build',['cmake','--build',build,'--parallel','4','--target','test-lightonocr3-contract','lightonocr3-trace'])
        command('component-controls',[build/'test-lightonocr3-contract'])
        token=os.environ['GH_TOKEN'];run=archive.fetch_metadata('/actions/runs/'+str(TARGET['run_id']),token)
        item=archive.fetch_metadata('/actions/artifacts/'+str(TARGET['artifact_id']),token);validate_metadata(run,item)
        report['baseline_provenance']=dict(run=run,artifact=item);save()
        path=scratch/'baseline.zip';report['verified_archive']=download(TARGET,path,token);save()
        retained=scratch/'retained';write(review/'retained-members.json',extract_inputs(path,retained))
        command('requirements',[sys.executable,'-m','pip','install','--report',review/'resolver.json','-r',ROOT/'models/requirements-lightonocr3-reference.txt'])
        command('dependency-check',[sys.executable,'-m','pip','check'])
        command('source-check',[sys.executable,'-c',"import sys;sys.path.insert(0,'tools');from dump_lightonocr3_reference import verify_environment;print(verify_environment())"])
        from compare_lightonocr3 import compare,validate_reference
        from lightonocr3_profile import source_directory,MODEL_ID
        ref=retained/'references'/CASE;meta=archive.read_json((ref/'reference.json').read_bytes());validate_reference(meta)
        require(meta['mode']=='transcription' and meta['max_new_tokens']==128,'reference task differs')
        shutil.copytree(ref,raw/'references'/CASE)
        image=ROOT/protocol['fixture']['image'];require(file_sha(image)==protocol['fixture']['sha256']==meta['image_sha256'],'form image differs')
        checkpoint=source_directory(MODEL_ID,cache_dir=scratch/'hf/hub');models={}
        for precision,expected in protocol['models'].items():
            model=scratch/(precision+'.gguf')
            command('convert-'+precision,[sys.executable,'models/convert-lightonocr-to-gguf.py','--model',checkpoint,
                                          '--profile','lightonocr3-1b','--dtype',precision,'--output',model])
            require(model.stat().st_size==expected['bytes'] and file_sha(model)==expected['sha256'],'frozen model differs')
            models[precision]=model
        binary=build/'lightonocr3-trace';binary_sha=file_sha(binary)
        for precision,model in models.items():
            for gate in ('0','1'):
                base=raw/CASE/precision/('kv'+gate);dirs={arm:base/arm for arm in ('forced','autonomous','plain')};controls={}
                for arm,output in dirs.items():
                    cap=len(meta['token_ids']) if arm=='forced' else 128;forced=ref/'forced-ids.txt' if arm=='forced' else '-'
                    report['native_request_processes']+=1
                    result=command(precision+'-kv'+gate+'-'+arm,[binary,model,image,output,str(cap),forced,'plain' if arm=='plain' else 'observe'],{GATE:gate})
                    write(output/'execution.json',dict(schema_version=1,build_mode='portable',git_commit=source,precision=precision,
                        arm=arm,process=result,model_sha256=protocol['models'][precision]['sha256'],model_bytes=model.stat().st_size,
                        binary_sha256=binary_sha,trace_sha256=file_sha(output/'trace.json'),gate=gate))
                    if gate=='0':controls[arm]=trace_control(retained/CASE/precision/'gate1'/arm,output)
                    elif arm!='plain':controls[arm]=prefill_control(raw/CASE/precision/'kv0'/arm,output)
                comparison=compare(ref,dirs['forced'],dirs['autonomous'],dirs['plain'],precision)
                row=dict(precision=precision,gate=gate,comparison=comparison,byte_controls=controls,step6=logit_probe(ref,dirs['forced']))
                if gate=='1':
                    old=next(x for x in report['rows'] if x['precision']==precision and x['gate']=='0')
                    row['step6_max_abs_reduced']=row['step6']['max_abs']<old['step6']['max_abs']
                report['rows'].append(row);write(review/(precision+'-kv'+gate+'.json'),row);save()
        require(file_sha(binary)==binary_sha and report['native_request_processes']==12 and len(report['rows'])==4,'first coverage/binary differs')
        candidates=[r for r in report['rows'] if r['gate']=='1']
        report.update(completed=True,**{'pass':all(r['step6_max_abs_reduced'] and r['comparison']['owner_bar_pass'] and
                      r['comparison']['decoded']['ids_exact'] and r['comparison']['decoded']['text_exact'] for r in candidates)})
        save();return 0 if report['pass'] else 1
    except Exception as error:
        report['failure']=dict(type=type(error).__name__,message=str(error));save();raise


if __name__=='__main__':raise SystemExit(main())
