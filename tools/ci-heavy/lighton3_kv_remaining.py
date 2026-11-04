#!/usr/bin/env python3
"""Complete the five retained-reference KV cases after an authenticated first proof."""
from __future__ import annotations
import copy
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

ROOT=Path(os.environ.get('GITHUB_WORKSPACE',Path(__file__).resolve().parents[2]))
sys.path[:0]=[str(ROOT/'models'),str(ROOT/'tools'),str(ROOT/'tools/ci-heavy')]
NATIVE='3e82a12058cfca8869f4aa8ead250acf2b708934'
SELECTOR='tools/ci-heavy/lighton3_kv_remaining.py'
PROTOCOL_NAME='docs/lighton3-kv-remaining-v1.json'
BINARY_SHA='7d4b027feb14b51261b36efbb3efc6cea73b20ee81aa8bd3bbf85976fea4dfdd'
CASES=['fox-transcription','fox-grounding','receipt-transcription','receipt-grounding','form-grounding']
FIRST=dict(run_id=41901879295,source='fc8ce8d2de1d9bc89790c0e6892a52fd51a2c1cf',branch='audit/lighton3-kv-complete-first-20261104',
           conclusion='success',artifact_id=13359865825,bytes=503653,
           sha256='e194a312cb84c17cf7a702cd585e2d202fbca45eb451ff80198a168140f1d6fb',
           protocol_sha256='b8d3955c0066474630de20111128f6ffa1c01d42f265aa9445a631fa0bec2c5b',binary_sha256=BINARY_SHA)
BASELINES=dict(
    full_eps=dict(run_id=41889499814,source='0473dd5e813fdf48d2bafbd16811271cf6c42599',branch='fix/ocr-qualification-nov4',
                  conclusion='failure',artifact_id=13355422898,bytes=8831289731,
                  sha256='b57f2b906da12f20fc1ef5462c4800b801a5349ed38eee790b3ecdbf1b057078'),
    first_eps=dict(run_id=41885702781,source='f27223fc276674c4621f32b709aa03ef0d035f09',branch='fix/ocr-qualification-nov4',
                   conclusion='success',artifact_id=13353235590,bytes=1459280989,
                   sha256='3f6087476bdc9d18ad9d58bbfad926a1f78d3f2d2fa4f98e41e0af0804adaba0'))
ARMS=('forced','autonomous','plain')


def require(ok,message):
    if not ok:raise ValueError(message)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def load_completion():
    path=Path(__file__).with_name('completion.py')
    if not path.exists():path=ROOT/'tools/ci-heavy/lighton3_kv_complete_first.py'
    spec=importlib.util.spec_from_file_location('kv_completion',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,path


def validate_closure(protocol,native,operator,files,blobs):
    require(protocol['schema']=='crispembed.lighton3.kv-remaining.v1' and protocol['native_source']==NATIVE,'remaining source differs')
    require(protocol['first_proof']==FIRST and protocol['baselines']==BASELINES and protocol['cases']==CASES,'closed panel inputs differ')
    require(native==NATIVE and re.fullmatch(r'[0-9a-f]{40}',operator),'native/operator revision differs')
    require(len(files)==3 and len(blobs)==3 and files==blobs,'operator script/helper/protocol differs')


def source_guard(protocol,protocol_path,helper_path):
    for key,value in dict(GITHUB_ACTIONS='true',GITHUB_REPOSITORY='CrispStrobe/CrispEmbed',RUNNER_OS='Linux',
                          REMAIN_SELECTED_SCRIPT=SELECTOR,REMAIN_SELECTED_RUNNER='ubuntu-latest',REMAIN_SELECTED_MODE='portable',
                          REMAIN_SELECTED_PIP='',REMAIN_SELECTED_ARGS='').items():
        require(os.environ.get(key,'')==value,'closed remaining dispatch differs: '+key)
    native=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();operator=os.environ['GITHUB_SHA']
    names=[SELECTOR,'tools/ci-heavy/lighton3_kv_complete_first.py',PROTOCOL_NAME]
    blobs=[subprocess.check_output(['git','show',operator+':'+name],cwd=ROOT) for name in names]
    validate_closure(protocol,native,operator,[Path(__file__).read_bytes(),helper_path.read_bytes(),protocol_path.read_bytes()],blobs)
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT),'native source dirty')
    for name,digest in protocol['owned_native_sha256'].items():require(file_sha(ROOT/name)==digest,'native owned source differs: '+name)
    for name,digest in protocol['protected_git_objects'].items():
        require(subprocess.check_output(['git','rev-parse','HEAD:'+name],cwd=ROOT,text=True).strip()==digest,'protected native source differs: '+name)
    old=json.loads((ROOT/'docs/lighton3-pixtral-epsilon-ab-v1.json').read_text())
    require(protocol['models']==old['models'] and protocol['fixture_cases']==old['fixture_cases'],'model/fixture protocol differs')
    return native,operator


def validate_metadata(target,run,item,api,name):
    require(target==FIRST or target in BASELINES.values(),'unreviewed artifact target')
    for key,value in dict(id=target['run_id'],head_sha=target['source'],head_branch=target['branch'],event='workflow_dispatch',
                          path='.github/workflows/heavy-cpu.yml',status='completed',conclusion=target['conclusion'],run_attempt=1).items():
        require(run.get(key)==value,'producer differs: '+key)
    for key in ('repository','head_repository'):
        require(run.get(key,{}).get('full_name')=='CrispStrobe/CrispEmbed' and run[key].get('id')==1210669686,'producer repository differs')
    for key,value in dict(id=target['artifact_id'],name=name,size_in_bytes=target['bytes'],digest='sha256:'+target['sha256'],expired=False,
                          url=api+'/actions/artifacts/'+str(target['artifact_id']),
                          archive_download_url=api+'/actions/artifacts/'+str(target['artifact_id'])+'/zip').items():
        require(item.get(key)==value,'artifact differs: '+key)
    for key,value in dict(id=target['run_id'],head_sha=target['source'],head_branch=target['branch'],repository_id=1210669686,
                          head_repository_id=1210669686).items():
        require(item.get('workflow_run',{}).get(key)==value,'artifact producer differs: '+key)


def validate_control(control):
    require(isinstance(control,dict) and control.get('all_bytes_exact') is True,'first prefill control failed')
    count=control.get('tensor_files');rows=control.get('tensors')
    require(type(count) is int and count>0 and isinstance(rows,list) and len(rows)==count,'first prefill inventory differs')
    names=[]
    for row in rows:
        name=row.get('name');size=row.get('bytes');digest=row.get('sha256')
        require(isinstance(name,str) and name.endswith('.f32') and name!='generated_ids.f32' and not
                (re.fullmatch(r'logits_step[0-9]+\.f32',name) and int(name[11:-4])>0),'first prefill name differs')
        require(type(size) is int and size>0 and size%4==0 and isinstance(digest,str) and re.fullmatch(r'[0-9a-f]{64}',digest),
                'first prefill digest/size differs')
        names.append(name)
    require(len(set(names))==len(names) and 'logits_step0.f32' in names,'first prefill inventory incomplete')


def validate_first(summary):
    require(summary.get('schema')=='crispembed.lighton3.kv-complete-first.v1' and summary.get('phase')=='first' and
            summary.get('operator_source')==FIRST['source'] and summary.get('native_source')==NATIVE and
            summary.get('protocol_sha256')==FIRST['protocol_sha256'],'first proof source differs')
    for key in ('completed','pass','rebuilt_binary_byte_exact'):require(summary.get(key) is True,'first flag differs: '+key)
    require(summary.get('binary_sha256')==BINARY_SHA,'first binary differs')
    for key,value in (('reused_native_request_processes',10),('new_native_request_processes',2),('native_request_processes',12),('original_forward_calls',0)):
        require(type(summary.get(key)) is int and summary[key]==value,'first counter differs: '+key)
    rows=summary.get('rows');require(isinstance(rows,list) and len(rows)==4,'first row count differs')
    identities=[(r.get('precision'),r.get('gate')) for r in rows]
    require(all(isinstance(p,str) and isinstance(g,str) for p,g in identities) and
            set(identities)=={(p,g) for p in ('f32','f16') for g in ('0','1')},'first row coverage differs')
    for row in rows:
        if row['gate']=='1':
            comparison=row.get('comparison',{});decoded=comparison.get('decoded',{})
            require(row.get('step6_max_abs_reduced') is True and comparison.get('owner_bar_pass') is True and
                    decoded.get('ids_exact') is True and decoded.get('text_exact') is True,'first candidate failed')
            controls=row.get('prefill_byte_controls');require(isinstance(controls,dict) and set(controls)=={'forced','autonomous'},'first control arms differ')
            for control in controls.values():validate_control(control)
    negative=next(r for r in rows if r['precision']=='f16' and r['gate']=='0')['comparison']['decoded']
    require(negative.get('ids_exact') is False and negative.get('text_exact') is False and
            negative['reference_ids'][6]==2918 and negative['native_ids'][6]==3377,'French baseline negative not preserved')
    return rows


def selected(name,kind):
    parts=Path(name).parts
    reference=kind=='full_eps' and len(parts)>=3 and parts[0]=='references' and parts[1] in CASES
    native=(len(parts)>=5 and parts[0] in CASES and parts[1] in ('f32','f16') and parts[2]=='gate1' and parts[3] in ARMS and
            ((kind=='first_eps' and parts[0]=='form-grounding') or (kind=='full_eps' and parts[0]!='form-grounding')))
    return (reference or native) and Path(name).suffix in ('.npy','.f32','.json','.txt')


def extract_inputs(path,destination,kind,archive):
    require(kind in BASELINES,'unknown baseline extraction');require(not destination.exists(),'fresh retained root required');destination.mkdir()
    rows=[];total=0
    with zipfile.ZipFile(path) as z:
        archive.safe_members(z)
        for info in z.infolist():
            if info.is_dir() or not selected(info.filename,kind):continue
            total+=info.file_size
            require(info.file_size<=512*2**20 and total<=(12 if kind=='full_eps' else 3)*2**30,'baseline selected byte budget exceeded')
            output=destination/info.filename;output.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as stream,output.open('xb') as target:shutil.copyfileobj(stream,target,2**20)
            require(output.stat().st_size==info.file_size,'retained baseline truncated')
            rows.append(dict(path=info.filename,bytes=info.file_size,sha256=file_sha(output)))
    require(bool(rows),'baseline selection empty');return rows


def main():
    out=Path(os.environ['KV_OUT']);review=out/'review';raw=out/'raw';review.mkdir(parents=True,exist_ok=False);raw.mkdir()
    report=dict(schema='crispembed.lighton3.kv-remaining.v1',phase='remaining',completed=False,**{'pass':False},rows=[],commands=[],
                original_forward_calls=0,new_native_request_processes=0,imported_native_request_processes=0,
                default_changed=False,publication_qualified=False)
    def save():
        path=review/'summary.pending.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');path.replace(review/'summary.json')
    save()
    try:
        require(len(sys.argv)==1,'remaining panel accepts no arguments')
        protocol_path=Path(os.environ['REMAIN_PROTOCOL']);protocol=json.loads(protocol_path.read_text())
        completion,helper_path=load_completion();native,operator=source_guard(protocol,protocol_path,helper_path)
        worker=completion.load_worker();archive=worker.archive
        report.update(native_source=native,operator_source=operator,protocol_sha256=file_sha(protocol_path));save()
        scratch=Path(os.environ['KV_SCRATCH']);require(not scratch.exists(),'fresh scratch required');scratch.mkdir(parents=True)
        require(out.stat().st_dev==scratch.stat().st_dev,'shared scratch/output disk required')
        memory=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        report['resource_admission']=dict(available_memory_bytes=memory,free_disk_bytes=shutil.disk_usage(scratch).free,
             required_memory_bytes=12*2**30,required_disk_bytes=48*2**30,
             baseline_zip_bytes=sum(t['bytes'] for t in BASELINES.values()),selected_baseline_cap_bytes=15*2**30,
             scope='48GiB covers retained ZIPs, extracted baselines, copied references, new traces, checkpoint, two models and build')
        save();require(memory>=12*2**30 and shutil.disk_usage(scratch).free>=48*2**30,'need12GiB RAM/48GiB disk')
        token=os.environ['GH_TOKEN']
        def acquire(label,target,name):
            run=archive.fetch_metadata('/actions/runs/'+str(target['run_id']),token)
            item=archive.fetch_metadata('/actions/artifacts/'+str(target['artifact_id']),token)
            validate_metadata(target,run,item,archive.API,name)
            report[label+'_provenance']=dict(run=run,artifact=item);save()
            path=scratch/(label+'.zip');report[label+'_archive']=worker.download(target,path,token);save();return path
        first_zip=acquire('first_proof',FIRST,'lighton3-kv-completion-review-'+str(FIRST['run_id']))
        with zipfile.ZipFile(first_zip) as z:
            archive.safe_members(z);first=archive.read_json(z.read('summary.json'))
        imported=validate_first(first);report['first_proof_summary']=first
        for row in imported:
            item=copy.deepcopy(row);item.update(case='form-transcription',imported_first_proof=True,proof_run_id=FIRST['run_id'])
            report['rows'].append(item)
        report['imported_native_request_processes']=12;save()
        env={k:v for k,v in os.environ.items() if not k.startswith(('CRISPEMBED_LIGHTON','LOCR_')) and k not in
             ('CRISPEMBED_MAX_PIXELS','GH_TOKEN','GITHUB_TOKEN','HF_TOKEN','HF_TOKEN_2','KAGGLE_API_TOKEN','KAGGLE_ACCOUNT')}
        env.update(CRISPEMBED_BUILD_MODE='portable',CRISPEMBED_LIGHTONOCR3='1',CRISPEMBED_LIGHTONOCR_FORCE_CPU='1',
                   CRISPEMBED_LIGHTONOCR3_PIXTRAL_EPS='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                   TOKENIZERS_PARALLELISM='false',HF_HOME=str(scratch/'hf'),PIP_NO_CACHE_DIR='1')
        def command(name,argv,extra=None):
            result=worker.execute(argv,review/(name+'.log'),cwd=ROOT,env=env|(extra or {}),timeout=3600)
            report['commands'].append(dict(name=name,**result,log=name+'.log'));save()
            require(result['exit']==0 and result['timed_out'] is False,'remaining command failed: '+name);return result
        build=scratch/'build'
        command('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF',
                              '-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF','-DGGML_BLAS=OFF','-DCRISPEMBED_BUILD_SHARED=OFF'])
        command('build',['cmake','--build',build,'--parallel','4','--target','test-lightonocr3-contract','lightonocr3-trace'])
        command('component-controls',[build/'test-lightonocr3-contract'])
        binary=build/'lightonocr3-trace';require(file_sha(binary)==BINARY_SHA,'rebuilt binary differs from first proof; no native calls')
        report['binary_sha256']=BINARY_SHA;report['rebuilt_binary_byte_exact']=True
        shutil.copyfile(build/'CMakeCache.txt',review/'CMakeCache.txt');report['cmake_cache_sha256']=file_sha(review/'CMakeCache.txt')
        diagnostic=raw/'diagnostic-binary';diagnostic.mkdir();shutil.copyfile(binary,diagnostic/'lightonocr3-trace');save()
        baseline_roots={};inventories={}
        for kind,target in BASELINES.items():
            path=acquire(kind,target,'lighton3-eps-raw-'+str(target['run_id']));destination=scratch/kind
            inventories[kind]=extract_inputs(path,destination,kind,archive);baseline_roots[kind]=destination
            worker.write(review/(kind+'-members.json'),inventories[kind])
        command('requirements',[sys.executable,'-m','pip','install','--report',review/'resolver.json','-r',ROOT/'models/requirements-lightonocr3-reference.txt'])
        command('dependency-check',[sys.executable,'-m','pip','check'])
        command('source-check',[sys.executable,'-c',"import sys;sys.path.insert(0,'tools');from dump_lightonocr3_reference import verify_environment;print(verify_environment())"])
        from compare_lightonocr3 import compare,validate_reference
        from lightonocr3_profile import source_directory,MODEL_ID
        checkpoint=source_directory(MODEL_ID,cache_dir=scratch/'hf/hub');models={}
        for precision,expected in protocol['models'].items():
            model=scratch/(precision+'.gguf')
            command('convert-'+precision,[sys.executable,'models/convert-lightonocr-to-gguf.py','--model',checkpoint,'--profile','lightonocr3-1b','--dtype',precision,'--output',model])
            require(model.stat().st_size==expected['bytes'] and file_sha(model)==expected['sha256'],'frozen GGUF differs');models[precision]=model
        reference_copies=[]
        for case in CASES:
            ref=baseline_roots['full_eps']/'references'/case;meta=archive.read_json((ref/'reference.json').read_bytes());validate_reference(meta)
            require(meta['max_new_tokens']==128 and case.endswith('-'+meta['mode']),'retained task differs')
            copied=raw/'references'/case;shutil.copytree(ref,copied)
            for path in copied.rglob('*'):
                if path.is_file():reference_copies.append(dict(path=path.relative_to(raw).as_posix(),bytes=path.stat().st_size,sha256=file_sha(path)))
            fixture=next(f for f in protocol['fixture_cases'] if case==f['id']+'-'+meta['mode'])
            image=ROOT/fixture['image'];require(file_sha(image)==fixture['sha256']==meta['image_sha256'],'frozen image differs')
            baseline=baseline_roots['first_eps' if case=='form-grounding' else 'full_eps']/case
            for precision,model in models.items():
                for gate in ('0','1'):
                    base=raw/case/precision/('kv'+gate);dirs={arm:base/arm for arm in ARMS};controls={}
                    for arm,output in dirs.items():
                        cap=len(meta['token_ids']) if arm=='forced' else 128;forced=ref/'forced-ids.txt' if arm=='forced' else '-'
                        report['new_native_request_processes']+=1
                        result=command(case+'-'+precision+'-kv'+gate+'-'+arm,[binary,model,image,output,str(cap),forced,'plain' if arm=='plain' else 'observe'],
                                       {'CRISPEMBED_LIGHTONOCR3_KV_F32':gate,'CRISPEMBED_LIGHTONOCR3_GROUNDING':'1' if meta['mode']=='grounding' else '0'})
                        worker.write(output/'execution.json',dict(schema_version=1,build_mode='portable',git_commit=NATIVE,precision=precision,arm=arm,process=result,
                                     model_sha256=protocol['models'][precision]['sha256'],model_bytes=model.stat().st_size,binary_sha256=BINARY_SHA,
                                     trace_sha256=file_sha(output/'trace.json'),gate=gate))
                        if gate=='0':controls[arm]=worker.trace_control(baseline/precision/'gate1'/arm,output)
                        elif arm!='plain':controls[arm]=completion.prefill_control(raw/case/precision/'kv0'/arm,output)
                    comparison=compare(ref,dirs['forced'],dirs['autonomous'],dirs['plain'],precision)
                    row=dict(case=case,precision=precision,gate=gate,comparison=comparison,byte_controls=controls,imported_first_proof=False)
                    report['rows'].append(row);worker.write(review/(case+'-'+precision+'-kv'+gate+'.json'),row);save()
        for kind,inventory in inventories.items():
            for member in inventory:
                path=baseline_roots[kind]/member['path'];require(path.stat().st_size==member['bytes'] and file_sha(path)==member['sha256'],'retained baseline mutated')
        for member in reference_copies:
            path=raw/member['path'];require(path.stat().st_size==member['bytes'] and file_sha(path)==member['sha256'],'copied reference mutated')
        worker.write(review/'copied-reference-members.json',reference_copies)
        require(file_sha(binary)==BINARY_SHA and file_sha(diagnostic/'lightonocr3-trace')==BINARY_SHA,'native binary mutated')
        require(report['new_native_request_processes']==60 and len(report['rows'])==24,'remaining coverage incomplete')
        report['native_request_processes']=72
        candidates=[r for r in report['rows'] if r['gate']=='1']
        report.update(completed=True,**{'pass':len(candidates)==12 and all(r['comparison']['owner_bar_pass'] and r['comparison']['decoded']['ids_exact'] and
                      r['comparison']['decoded']['text_exact'] for r in candidates)})
        save();return 0 if report['pass'] else 1
    except Exception as error:
        report['failure']=dict(type=type(error).__name__,message=str(error));save();raise


if __name__=='__main__':raise SystemExit(main())
