#!/usr/bin/env python3
"""Authenticate ten retained KV arms and complete only two missing CPU calls."""
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

ROOT=Path(os.environ.get('GITHUB_WORKSPACE',Path(__file__).resolve().parents[2]))
sys.path[:0]=[str(ROOT/'models'),str(ROOT/'tools'),str(ROOT/'tools/ci-heavy')]
NATIVE='3e82a12058cfca8869f4aa8ead250acf2b708934'
OLD_PROTOCOL_SHA='a6fe086d4d2bce01980f455b489edf8b401cce73456d64c31610e44f4760ad8e'
SELECTOR='tools/ci-heavy/lighton3_kv_complete_first.py'
PROTOCOL_NAME='docs/lighton3-kv-complete-first-v1.json'
CASE='form-transcription'
PRESERVED=dict(run_id=41897136488,source=NATIVE,branch='audit/lighton3-kv-f32-first-20261104',conclusion='failure',
               raw=dict(artifact_id=13355853749,bytes=1300045675,sha256='4a5453695794b1a8a920d7e9b8ad6a218a80f52b7407c50d13fd5f1c3820999f'),
               review=dict(artifact_id=13355723878,bytes=398550,sha256='89f1fdf2b705ca57c46bb92ae3d7c64a014d3387f62dc423bafc09f7a8476a5c'))
ARMS=('forced','autonomous','plain')
EXPECTED={(p,g,a) for p in ('f32','f16') for g in ('0','1') for a in ARMS} - {('f16','1','autonomous'),('f16','1','plain')}


def require(ok,message):
    if not ok:raise ValueError(message)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def load_worker():
    spec=importlib.util.spec_from_file_location('retained_kv_worker',ROOT/'tools/ci-heavy/lighton3_kv_f32_ab.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    return worker


def validate_closure(protocol,native,operator,script_bytes,protocol_bytes,script_blob,protocol_blob):
    require(protocol['schema']=='crispembed.lighton3.kv-complete-first.v1' and protocol['native_source']==NATIVE,
            'completion source protocol differs')
    require(protocol['preserved']==PRESERVED,'preserved producer differs')
    require(native==NATIVE and re.fullmatch(r'[0-9a-f]{40}',operator),'native/operator revision invalid')
    require(script_bytes==script_blob and protocol_bytes==protocol_blob,'operator script/protocol blob differs')


def source_guard(protocol,protocol_path):
    env=os.environ
    for key,value in dict(GITHUB_ACTIONS='true',GITHUB_REPOSITORY='CrispStrobe/CrispEmbed',RUNNER_OS='Linux',
                          COMPLETE_SELECTED_SCRIPT=SELECTOR,COMPLETE_SELECTED_RUNNER='ubuntu-latest',
                          COMPLETE_SELECTED_MODE='portable',COMPLETE_SELECTED_PIP='',COMPLETE_SELECTED_ARGS='').items():
        require(env.get(key,'')==value,'closed completion dispatch differs: '+key)
    native=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();operator=env['GITHUB_SHA']
    def blob(name):return subprocess.check_output(['git','show',operator+':'+name],cwd=ROOT)
    validate_closure(protocol,native,operator,Path(__file__).read_bytes(),protocol_path.read_bytes(),blob(SELECTOR),blob(PROTOCOL_NAME))
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT),'native source dirty')
    for name,digest in protocol['owned_native_sha256'].items():require(file_sha(ROOT/name)==digest,'native owned source differs: '+name)
    for name,digest in protocol['protected_git_objects'].items():
        require(subprocess.check_output(['git','rev-parse','HEAD:'+name],cwd=ROOT,text=True).strip()==digest,'protected source differs: '+name)
    require(file_sha(ROOT/'docs/lighton3-kv-f32-ab-v1.json')==OLD_PROTOCOL_SHA,'original KV protocol differs')
    return native,operator


def validate_metadata(run,item,kind,api):
    target=PRESERVED[kind]
    for key,value in dict(id=PRESERVED['run_id'],head_sha=NATIVE,head_branch=PRESERVED['branch'],event='workflow_dispatch',
                          path='.github/workflows/heavy-cpu.yml',status='completed',conclusion='failure',run_attempt=1).items():
        require(run.get(key)==value,'preserved producer differs: '+key)
    for key in ('repository','head_repository'):
        require(run.get(key,{}).get('full_name')=='CrispStrobe/CrispEmbed' and run[key].get('id')==1210669686,'producer repository differs')
    for key,value in dict(id=target['artifact_id'],name='lighton3-kv-'+kind+'-'+str(PRESERVED['run_id']),
                          size_in_bytes=target['bytes'],digest='sha256:'+target['sha256'],expired=False,
                          url=api+'/actions/artifacts/'+str(target['artifact_id']),
                          archive_download_url=api+'/actions/artifacts/'+str(target['artifact_id'])+'/zip').items():
        require(item.get(key)==value,'preserved artifact differs: '+key)
    for key,value in dict(id=PRESERVED['run_id'],head_sha=NATIVE,head_branch=PRESERVED['branch'],
                          repository_id=1210669686,head_repository_id=1210669686).items():
        require(item.get('workflow_run',{}).get(key)==value,'artifact producer differs: '+key)


def validate_summary(summary):
    require(summary.get('source')==NATIVE and summary.get('protocol_sha256')==OLD_PROTOCOL_SHA,'first summary source differs')
    require(summary.get('completed') is False and summary.get('pass') is False and summary.get('phase')=='first' and
            summary.get('case')==CASE,'first summary state differs')
    for key,value in (('native_request_processes',10),('original_forward_calls',0)):
        require(type(summary.get(key)) is int and summary[key]==value,'first summary counter differs')
    require(summary.get('failure')==dict(type='ValueError',message='KV changed prefill bytes: generated_ids.f32'),
            'first failure is not the retained postdecode-inventory defect')
    rows=summary.get('rows');require(isinstance(rows,list) and len(rows)==3,'retained row count differs')
    identities=[(r.get('precision'),r.get('gate')) for r in rows]
    require(identities==[('f32','0'),('f32','1'),('f16','0')],'retained rows differ')
    commands=summary.get('commands');require(isinstance(commands,list) and bool(commands),'retained commands missing')
    for row in commands:require(type(row.get('exit')) is int and row['exit']==0 and row.get('timed_out') is False,'retained command failed')
    native_commands=[r['name'] for r in commands if re.fullmatch(r'(f32|f16)-kv[01]-(forced|autonomous|plain)',r.get('name',''))]
    expected={p+'-kv'+g+'-'+a for p,g,a in EXPECTED}
    require(len(native_commands)==10 and set(native_commands)==expected,'retained native command coverage differs')


def selected(name):
    parts=Path(name).parts
    return ((len(parts)>=3 and parts[:2]==('references',CASE)) or
            (len(parts)>=5 and parts[0]==CASE and parts[1] in ('f32','f16') and parts[2] in ('kv0','kv1') and parts[3] in ARMS)) and Path(name).suffix in ('.npy','.f32','.json','.txt')


def extract_raw(path,destination,archive):
    require(not destination.exists(),'fresh retained destination required');destination.mkdir()
    rows=[];total=0
    with zipfile.ZipFile(path) as z:
        archive.safe_members(z)
        for info in z.infolist():
            if info.is_dir() or not selected(info.filename):continue
            total+=info.file_size;require(info.file_size<=512*2**20 and total<=3*2**30,'retained byte budget exceeded')
            output=destination/info.filename;output.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source,output.open('xb') as target:shutil.copyfileobj(source,target,2**20)
            require(output.stat().st_size==info.file_size,'retained member truncated')
            rows.append(dict(path=info.filename,bytes=info.file_size,sha256=file_sha(output)))
    return rows


def executions(root):
    directories={ (p.parts[-3],p.parts[-2].removeprefix('kv'),p.parts[-1]) for p in (root/CASE).glob('*/kv*/*') if p.is_dir() }
    require(directories==EXPECTED,'retained arm directories differ')
    paths=list((root/CASE).glob('*/kv*/*/execution.json'))
    found={}
    for path in paths:
        relative=path.relative_to(root/CASE).parts
        key=(relative[0],relative[1].removeprefix('kv'),relative[2])
        found[key]=json.loads(path.read_text())
    require(set(found)==EXPECTED and len(paths)==10,'retained execution inventory differs')
    return found


def validate_executions(found,models,binary_sha=None):
    require(set(found)==EXPECTED,'retained execution coverage differs')
    hashes=set()
    for (precision,gate,arm),row in found.items():
        require(row.get('git_commit')==NATIVE and row.get('precision')==precision and row.get('arm')==arm and
                row.get('gate')==gate and row.get('build_mode')=='portable' and row.get('schema_version')==1,'retained execution identity differs')
        require(row.get('model_sha256')==models[precision]['sha256'] and row.get('model_bytes')==models[precision]['bytes'],'retained model differs')
        require(type(row.get('process',{}).get('exit')) is int and row['process']['exit']==0 and row['process'].get('timed_out') is False,
                'retained native execution failed')
        value=row.get('binary_sha256');require(isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value),'retained binary digest missing')
        hashes.add(value)
    require(len(hashes)==1,'retained arms have different binaries')
    if binary_sha is not None:require(hashes=={binary_sha},'rebuilt binary is not byte-identical; stop before native calls')
    return next(iter(hashes))


def stored_byte_controls(summary,root):
    proof=[]
    for row in summary['rows']:
        if row['gate']!='0':continue
        controls=row.get('byte_controls');require(isinstance(controls,dict) and set(controls)==set(ARMS),'gate0 control arms differ')
        for arm,control in controls.items():
            directory=root/CASE/row['precision']/'kv0'/arm
            files=sorted(p.name for p in directory.glob('*.f32'))
            require(control.get('all_bytes_exact') is True and type(control.get('tensor_files')) is int and
                    control['tensor_files']==len(files),'gate0 byte control count differs')
            require((not files if arm=='plain' else bool(files)),'gate0 observation inventory invalid')
            tensors=control.get('tensors');require(isinstance(tensors,list) and len(tensors)==len(files),'gate0 inventory missing')
            require([t.get('name') for t in tensors]==files,'gate0 tensor names differ')
            for tensor in tensors:
                path=directory/tensor['name']
                require(tensor.get('bytes')==path.stat().st_size and tensor.get('sha256')==file_sha(path),'gate0 tensor bytes differ')
            proof.append(dict(precision=row['precision'],arm=arm,verified_control=control))
    require(len(proof)==6,'both KV0 proofs required')
    return proof


def prefill_control(old,new):
    def inventory(directory):
        return sorted(p.name for p in directory.glob('*.f32') if p.name!='generated_ids.f32' and not
                      (re.fullmatch(r'logits_step[0-9]+\.f32',p.name) and int(p.name[11:-4])>0))
    names=inventory(old);require(names==inventory(new) and 'logits_step0.f32' in names,'prefill inventory differs')
    rows=[]
    for name in names:
        require((old/name).stat().st_size==(new/name).stat().st_size and file_sha(old/name)==file_sha(new/name),'prefill bytes differ: '+name)
        rows.append(dict(name=name,bytes=(new/name).stat().st_size,sha256=file_sha(new/name)))
    return dict(all_bytes_exact=True,tensor_files=len(rows),tensors=rows)


def main():
    out=Path(os.environ['KV_OUT']);review=out/'review';review.mkdir(parents=True,exist_ok=False)
    report=dict(schema='crispembed.lighton3.kv-complete-first.v1',phase='first',completed=False,**{'pass':False},
                original_forward_calls=0,reused_native_request_processes=0,new_native_request_processes=0,
                default_changed=False,publication_qualified=False,commands=[],rows=[])
    def save():
        temp=review/'summary.pending.json';temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(review/'summary.json')
    save()
    try:
        require(len(sys.argv)==1,'completion accepts no arguments')
        protocol_path=Path(os.environ['COMPLETE_PROTOCOL']);protocol=json.loads(protocol_path.read_text())
        native,operator=source_guard(protocol,protocol_path);worker=load_worker();archive=worker.archive
        report.update(native_source=native,operator_source=operator,protocol_sha256=file_sha(protocol_path));save()
        old_protocol=archive.read_json((ROOT/'docs/lighton3-kv-f32-ab-v1.json').read_bytes())
        require(protocol['models']==old_protocol['models'],'frozen models differ')
        scratch=Path(os.environ['KV_SCRATCH']);require(not scratch.exists(),'fresh scratch required');scratch.mkdir(parents=True)
        require(out.stat().st_dev==scratch.stat().st_dev,'shared scratch/output storage required')
        memory=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
        require(memory>=12*2**30 and shutil.disk_usage(scratch).free>=40*2**30,'need12GiB RAM/40GiB disk')
        token=os.environ['GH_TOKEN'];run=archive.fetch_metadata('/actions/runs/'+str(PRESERVED['run_id']),token)
        for kind in ('review','raw'):
            target=PRESERVED[kind];item=archive.fetch_metadata('/actions/artifacts/'+str(target['artifact_id']),token)
            validate_metadata(run,item,kind,archive.API)
            report[kind+'_provenance']=dict(run=run,artifact=item);save()
            worker.download(target,scratch/(kind+'.zip'),token)
        with zipfile.ZipFile(scratch/'review.zip') as z:
            archive.safe_members(z);summary=archive.read_json(z.read('summary.json'))
        validate_summary(summary);report['retained_first_failure']=summary;save()
        raw=out/'raw';members=extract_raw(scratch/'raw.zip',raw,archive);worker.write(review/'retained-members.json',members)
        found=executions(raw);validate_executions(found,protocol['models'])
        for key,row in found.items():
            precision,gate,arm=key;directory=raw/CASE/precision/('kv'+gate)/arm
            require(file_sha(directory/'trace.json')==row['trace_sha256'],'retained trace digest differs')
        report['gate0_preserved_proofs']=stored_byte_controls(summary,raw)
        report['reused_native_request_processes']=10;save()
        env={k:v for k,v in os.environ.items() if not k.startswith(('CRISPEMBED_LIGHTON','LOCR_')) and k not in
             ('CRISPEMBED_MAX_PIXELS','GH_TOKEN','GITHUB_TOKEN','HF_TOKEN','HF_TOKEN_2','KAGGLE_API_TOKEN','KAGGLE_ACCOUNT')}
        env.update(CRISPEMBED_BUILD_MODE='portable',CRISPEMBED_LIGHTONOCR3='1',CRISPEMBED_LIGHTONOCR_FORCE_CPU='1',
                   CRISPEMBED_LIGHTONOCR3_PIXTRAL_EPS='1',CRISPEMBED_LIGHTONOCR3_KV_F32='1',CRISPEMBED_LIGHTONOCR3_GROUNDING='0',
                   OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',
                   HF_HOME=str(scratch/'hf'),PIP_NO_CACHE_DIR='1')
        def command(name,argv):
            result=worker.execute(argv,review/(name+'.log'),cwd=ROOT,env=env,timeout=3600)
            report['commands'].append(dict(name=name,**result,log=name+'.log'));save()
            require(result['exit']==0 and result['timed_out'] is False,'completion command failed: '+name)
            return result
        build=scratch/'build'
        command('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF',
                              '-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF','-DGGML_BLAS=OFF','-DCRISPEMBED_BUILD_SHARED=OFF'])
        command('build',['cmake','--build',build,'--parallel','4','--target','test-lightonocr3-contract','lightonocr3-trace'])
        command('component-controls',[build/'test-lightonocr3-contract'])
        binary=build/'lightonocr3-trace';binary_sha=file_sha(binary)
        report['rebuilt_binary_sha256']=binary_sha
        report['expected_binary_sha256']=validate_executions(found,protocol['models'])
        shutil.copyfile(build/'CMakeCache.txt',review/'CMakeCache.txt');save()
        validate_executions(found,protocol['models'],binary_sha)
        report['binary_sha256']=binary_sha;report['rebuilt_binary_byte_exact']=True;save()
        diagnostic=raw/'diagnostic-binary';diagnostic.mkdir();shutil.copyfile(binary,diagnostic/'lightonocr3-trace')
        require(file_sha(diagnostic/'lightonocr3-trace')==binary_sha,'diagnostic binary copy differs')
        command('requirements',[sys.executable,'-m','pip','install','--report',review/'resolver.json','-r',ROOT/'models/requirements-lightonocr3-reference.txt'])
        command('dependency-check',[sys.executable,'-m','pip','check'])
        command('source-check',[sys.executable,'-c',"import sys;sys.path.insert(0,'tools');from dump_lightonocr3_reference import verify_environment;print(verify_environment())"])
        from compare_lightonocr3 import compare,validate_reference
        from lightonocr3_profile import source_directory,MODEL_ID
        ref=raw/'references'/CASE;meta=archive.read_json((ref/'reference.json').read_bytes());validate_reference(meta)
        require(meta['mode']=='transcription' and meta['max_new_tokens']==128,'reference task differs')
        fixture=old_protocol['fixture'];image=ROOT/fixture['image']
        require(file_sha(image)==fixture['sha256']==meta['image_sha256'],'frozen form image differs')
        checkpoint=source_directory(MODEL_ID,cache_dir=scratch/'hf/hub');model=scratch/'f16.gguf'
        command('convert-f16',[sys.executable,'models/convert-lightonocr-to-gguf.py','--model',checkpoint,'--profile','lightonocr3-1b','--dtype','f16','--output',model])
        require(model.stat().st_size==protocol['models']['f16']['bytes'] and file_sha(model)==protocol['models']['f16']['sha256'],'frozen F16 bytes differ')
        for arm in ('autonomous','plain'):
            output=raw/CASE/'f16'/'kv1'/arm;require(not output.exists(),'missing arm unexpectedly exists')
            report['new_native_request_processes']+=1
            result=command('f16-kv1-'+arm,[binary,model,image,output,'128','-','plain' if arm=='plain' else 'observe'])
            worker.write(output/'execution.json',dict(schema_version=1,build_mode='portable',git_commit=NATIVE,precision='f16',arm=arm,
                         process=result,model_sha256=protocol['models']['f16']['sha256'],model_bytes=model.stat().st_size,
                         binary_sha256=binary_sha,trace_sha256=file_sha(output/'trace.json'),gate='1'))
        for precision in ('f32','f16'):
            for gate in ('0','1'):
                base=raw/CASE/precision/('kv'+gate);dirs={arm:base/arm for arm in ARMS}
                comparison=compare(ref,dirs['forced'],dirs['autonomous'],dirs['plain'],precision)
                row=dict(precision=precision,gate=gate,comparison=comparison,step6=worker.logit_probe(ref,dirs['forced']))
                if gate=='1':
                    row['prefill_byte_controls']={arm:prefill_control(raw/CASE/precision/'kv0'/arm,dirs[arm]) for arm in ('forced','autonomous')}
                    previous=next(r for r in report['rows'] if r['precision']==precision and r['gate']=='0')
                    row['step6_max_abs_reduced']=row['step6']['max_abs']<previous['step6']['max_abs']
                report['rows'].append(row);worker.write(review/(precision+'-kv'+gate+'.json'),row);save()
        # Rehash every retained file after computation, including original execution receipts.
        for member in members:
            path=raw/member['path'];require(path.stat().st_size==member['bytes'] and file_sha(path)==member['sha256'],'retained member mutated')
        require(file_sha(binary)==binary_sha and report['new_native_request_processes']==2,'completion binary/counter differs')
        report['native_request_processes']=12
        candidates=[r for r in report['rows'] if r['gate']=='1']
        report.update(completed=True,**{'pass':len(candidates)==2 and all(r['step6_max_abs_reduced'] and r['comparison']['owner_bar_pass'] and
                      r['comparison']['decoded']['ids_exact'] and r['comparison']['decoded']['text_exact'] for r in candidates)})
        save();return 0 if report['pass'] else 1
    except Exception as error:
        report['failure']=dict(type=type(error).__name__,message=str(error));save();raise


if __name__=='__main__':raise SystemExit(main())
