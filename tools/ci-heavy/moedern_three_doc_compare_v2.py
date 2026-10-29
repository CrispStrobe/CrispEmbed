#!/usr/bin/env python3
"""Zero-forward recovery of18 retained MoEdern final-vector comparisons.

No original/native runner, model loader, tokenizer, Torch/ST/HF/Kaggle dependency.
The frozen b86 adapter and comparator remain unmodified.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[2]
PROFILE=ROOT/'docs/moedern-three-doc-compare-v2.json'
SOURCE='b86e8130947b59e09dfe4dde0e2dae01957c443d'
KEYS=['d:24726600','d:24928817','d:5849439']
KINDS=['all','selected']


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for data in iter(lambda:f.read(2**20),b''):h.update(data)
    return h.hexdigest()


def identity(path):return dict(bytes=Path(path).stat().st_size,sha256=sha(path))
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    with tmp.open('w',encoding='utf-8')as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def checked(path,pin):require(identity(path)==pin,'exact retained file differs: '+Path(path).name)


def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def verify(reviewed):
    p=read(PROFILE)
    require(re.fullmatch('[0-9a-f]{40}',reviewed or '')is not None and git('rev-parse','HEAD')==reviewed,'exact reviewed source required')
    paths=[*p['protected_git_objects'],*p['owned_sha256'],str(PROFILE.relative_to(ROOT))]
    require(not git('status','--porcelain','--',*paths),'dirty compare-only source')
    for path,obj in p['protected_git_objects'].items():
        require(git('rev-parse',SOURCE+':'+path)==obj and git('rev-parse','HEAD:'+path)==obj,'frozen b86 source object differs: '+path)
        if (ROOT/path).is_file():require(git('hash-object','--',path)==obj,'frozen worktree file differs')
    for path,digest in p['owned_sha256'].items():require(sha(ROOT/path)==digest,'owned compare-only source differs: '+path)
    require(p['keys']==KEYS and p['gates']==dict(max_absolute=.002,min_row_cosine=.99999)and p['forward_count']==0,'fixed zero-forward scope differs')
    return p


def safe_name(key,kind):
    # Explicit bijection, not lossy arbitrary input sanitization. Filenames,
    # source case IDs, keys and float bits are never rewritten.
    require(key in KEYS and kind in KINDS,'unknown retained case/kind')
    return {'d:24726600':'doc_24726600','d:24928817':'doc_24928817','d:5849439':'doc_5849439'}[key]+'_'+kind


def api(path):
    require(os.environ.get('GH_TOKEN'),'existing repository artifact credential absent')
    req=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/'+path,
        headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
    try:
        with urllib.request.urlopen(req,timeout=60)as r:data=r.read(2*2**20+1)
    except Exception:raise ValueError('bounded GitHub service request rejected')from None
    require(len(data)<=2*2**20,'service metadata bound exceeded');return json.loads(data)


def admission(reviewed,out):
    require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('GITHUB_SHA')==reviewed
            and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted','genuine hosted analysis source required')
    run=api('actions/runs/'+os.environ.get('GITHUB_RUN_ID','0'))
    require(run['head_sha']==reviewed and run['event']=='workflow_dispatch'and run['status']=='in_progress'
            and run['path']=='.github/workflows/moedern-three-doc-compare-v2.yml','actual compare-only workflow identity differs')
    out.mkdir(parents=True,exist_ok=True)
    ram=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
    disk=shutil.disk_usage(out).free
    require(ram>=2*2**30 and disk>=2*2**30,'separate zero-forward2GiB RAM/2GiB disk admission required')
    return dict(run=run['id'],source=reviewed,attempt=run['run_attempt'],workflow=run['path'],available_ram_bytes=ram,
                free_disk_bytes=disk,old_model_guard_14gib_24gib_unchanged=True,models_accessed=False)


def download(pin,path):
    row=api('actions/artifacts/'+str(pin['id']))
    require(not row['expired']and row['workflow_run']['id']==40825166813 and row['workflow_run']['head_sha']==SOURCE
            and row['name']==pin['name']and row['size_in_bytes']==pin['bytes']and row['digest']=='sha256:'+pin['sha256'],
            'exact retained artifact service identity differs')
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*args,**kwargs):return None
    req=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(pin['id'])+'/zip',
        headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
    try:r=urllib.request.build_opener(NoRedirect).open(req,timeout=60)
    except urllib.error.HTTPError as error:
        require(error.code==302 and error.headers['Location'].startswith('https://'),'artifact redirect differs')
        try:r=urllib.request.urlopen(error.headers['Location'],timeout=120)
        except Exception:raise ValueError('signed archive transfer rejected')from None
    try:
        with r,path.open('xb')as f:
            count=0
            while data:=r.read(2**20):
                count+=len(data);require(count<=pin['bytes'],'archive exceeded exact extent');f.write(data)
    except Exception:raise ValueError('bounded retained archive transfer rejected')from None
    checked(path,{k:pin[k]for k in ['bytes','sha256']})
    return {k:row[k]for k in ['id','name','size_in_bytes','digest','expires_at']}


def extract(archive,target,selected):
    target.mkdir(parents=True,exist_ok=True);found={}
    with zipfile.ZipFile(archive)as z:
        names=[r.filename for r in z.infolist()]
        require(len(names)==len(set(names))and len(names)<=256 and sum(r.file_size for r in z.infolist())<=32*2**20,'retained archive member/count bound differs')
        for r in z.infolist():
            p=PurePosixPath(r.filename)
            require(not p.is_absolute()and '..'not in p.parts and '\\'not in r.filename and
                    ((r.external_attr>>16)&0o170000)!=0o120000,'unsafe retained archive member')
            if r.filename not in selected:continue
            require(not r.is_dir()and r.file_size<=2*2**20,'exact bounded selected file required')
            data=z.read(r);out=target/p;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
            found[r.filename]=identity(out)
    require(set(found)==set(selected),'retained required member missing')
    for name,pin in selected.items():require(found[name]==pin,'retained member SHA/extent differs: '+name)
    return found


def validate_inputs(root,p):
    packet=read(root/'three-doc-input/input-packet.json');prepared=read(root/'three-doc-input/prepare.json')
    current=read(root/'three-doc-out/raw/original/original.json')
    require(packet['completed']is True and packet['keys']==KEYS and [c['key']for c in packet['cases']]==KEYS
            and packet['all_raw_inputs_authenticated_before_framework']is True and packet['full_original_recapture']is False
            and packet['source_protocol_sha256']==p['initial_protocol_sha256'],'authenticated initial packet differs')
    require(prepared['completed']is True and prepared['source']==SOURCE and prepared['input_packet_sha256']==p['members']['three-doc-input/input-packet.json']['sha256'],'initial preparation differs')
    old=read(ROOT/'tests/results/moedern-colbert/scifact-original-1p5m.json')
    native=read(ROOT/'tests/results/moedern-colbert/scifact-native-1p5m-v3-native.json')
    require(current['source']==SOURCE and current['completed']is False and current['arrays']==[]
            and current['error']=='ValueError: Unsafe/duplicate adapter tensor name'and current['model']==old['model'],
            'initial original model/runtime/error identity differs')
    loaded=[v for group in current['model']['loaded_tensor_sha256'].values()for v in group.values()]
    require(len(loaded)==32 and all(isinstance(v,str)and re.fullmatch('[0-9a-f]{64}',v)for v in loaded),'exact32 loaded weight identities required')
    require(len(current['cases'])==3 and [c['key']for c in current['cases']]==KEYS,'all three saved ordinary cases required')
    records=[]
    for case,observed in zip(packet['cases'],current['cases']):
        count=len(case['ids']);require(case['id']==observed['id']=='doc-'+case['key'].split(':')[1]and case['role']=='document'and 3<=count<=512
            and case['ids'][:2]==[2,30001]and hashlib.sha256(case['text'].encode()).hexdigest()==case['text_sha256'],
            'stable input ID/text/role identity differs')
        for key in ['supplied_mask','effective_mask','scoring_mask']:
            require(len(case[key])==count and all(type(x)is int and x in(0,1)for x in case[key]),'exact three-mask/expansion shape differs')
        require(case['expansion_positions']==[],'document expansion-position list must stay empty')
        require(all(type(i)is int and 0<=i<30002 for i in case['ids']),'bounded original IDs required')
        for kind in KINDS:
            shape=[count if kind=='all'else sum(case['scoring_mask']),128]
            for side,producer in [('original',old['arrays']),('native',native['f32']['capture']['files'])]:
                pin=case['anchors'][side][kind]
                require(pin['shape']==shape and pin['bytes']==math.prod(shape)*4 and pin['retained_file']==('all.f32'if kind=='all'else'selected.f32')
                    and pin['retained_whole_sha256']==producer[pin['retained_file']]['sha256']
                    and 0<=pin['retained_offset']<=producer[pin['retained_file']]['bytes']-pin['bytes'],'old whole-stream lineage/shape differs')
                path=root/'three-doc-input'/pin['path'];checked(path,{k:pin[k]for k in ['bytes','sha256']})
            now=observed['anchor'][kind]
            require(now['shape']==shape and now['frozen_sha256']==case['anchors']['original'][kind]['sha256']
                    and now['current_file']==case['id']+'/'+kind+'.f32','fresh ordinary descriptor differs')
            path=root/'three-doc-out/raw/original'/now['current_file'];checked(path,dict(bytes=math.prod(shape)*4,sha256=now['current_sha256']))
            for side,path in [('old_original',root/'three-doc-input'/case['anchors']['original'][kind]['path']),
                              ('old_native93',root/'three-doc-input'/case['anchors']['native'][kind]['path']),('fresh_original',path)]:
                require(all(math.isfinite(x[0])for x in struct.iter_unpack('<f',path.read_bytes())),'nonfinite retained final vector')
                records.append(dict(key=case['key'],kind=kind,side=side,path=str(path.relative_to(root)),shape=shape,**identity(path)))
    return packet,current,records


def fetch(args,p,report):
    run=api('actions/runs/40825166813');jobs=api('actions/runs/40825166813/jobs?per_page=100')['jobs']
    require(run['head_sha']==SOURCE and run['run_attempt']==1 and run['status']=='completed'and run['conclusion']=='failure'
            and run['path']=='.github/workflows/moedern-three-doc-v1.yml'and len(jobs)==1
            and jobs[0]['id']==123286439520 and jobs[0]['name']=='diagnostic'and jobs[0]['conclusion']=='failure','initial failed producer identity differs')
    args.out.mkdir(parents=True,exist_ok=True);raw=args.out/'raw';raw.mkdir(exist_ok=True)
    report['producer']=dict(run=run['id'],source=run['head_sha'],attempt=1,conclusion=run['conclusion'],job_id=jobs[0]['id'])
    report['artifacts']={}
    for kind,pin in p['artifacts'].items():report['artifacts'][kind]=download(pin,raw/(kind+'.zip'))
    # Raw extraction includes only the exact eighteen already-retained matrices;
    # no model/checkpoint/tokenizer payload or SDK is accessed.
    extract(raw/'small.zip',raw/'small',p['small_members'])
    extract(raw/'raw.zip',raw/'retained',p['members'])
    packet,current,records=validate_inputs(raw/'retained',p)
    review=args.out/'review'
    for src,name in [('three-doc-out/raw/original/original.json','initial-original.json'),
                     ('three-doc-out/review/hosted.json','initial-hosted.json'),
                     ('three-doc-out/review/original-three-doc.log','initial-original.log')]:
        shutil.copyfile(raw/'small'/src,review/name)
    report.update(completed=True,matrices=records,initial_numpy_rejections=sum(not v['passed']for c in current['cases']for v in c['anchor'].values()),
                  original_observers_reached=False,native_forward_reached=False,framework_imported=False,models_accessed=False,
                  old_whole_blobs_rehashed_now=False,old_authenticated_span_copies_rehashed=True)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def diagnostics(np,a,b,positions,ids):
    delta=np.abs(b.astype('float64')-a.astype('float64'));row_max=delta.max(axis=1)
    x,y=a.astype('float64'),b.astype('float64');left=np.linalg.norm(x,axis=1);right=np.linalg.norm(y,axis=1)
    cos=np.divide((x*y).sum(axis=1),left*right,out=np.zeros_like(left),where=(left*right)!=0);cos[(left==0)&(right==0)]=1
    failed=np.flatnonzero((row_max>.002)|(cos<.99999));changed=np.flatnonzero(np.any(a.view('<u4')!=b.view('<u4'),axis=1))
    row,col=map(int,np.unravel_index(int(delta.argmax()),delta.shape))
    def where(index):
        return None if index is None else dict(matrix_row=int(index),input_position=positions[int(index)],token_id=ids[positions[int(index)]])
    return dict(failing_rows=int(len(failed)),first_failing_row=where(int(failed[0])if len(failed)else None),
                first_bit_different_row=where(int(changed[0])if len(changed)else None),
                worst_absolute=dict(**where(row),component=col,absolute_error=float(delta[row,col]),left=float(a[row,col]),right=float(b[row,col])))


def compare(args,p,report):
    require(read(args.out/'fetch.json')['completed']is True,'completed zero-forward input fetch required')
    root=args.out/'raw/retained'
    for name,pin in p['members'].items():checked(root/name,pin)
    packet,current,records=validate_inputs(root,p)
    # Optional NumPy only, after all artifact/input/model-identity checks. No
    # inference framework is required or installed by this workflow.
    import numpy as np
    require(np.__version__=='2.2.6','frozen independent NumPy2.2.6 required')
    sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'))
    reference=module(ROOT/'tools/moedern-native-v2/replay.py','retained_three_doc_ref')
    report['actual_ref_controls']=reference.comparator_controls(args.diff,args.out/'raw/ref-controls')
    arrays={};paths={}
    for r in records:
        path=root/r['path'];key=(r['key'],r['side'],r['kind']);a=np.fromfile(path,dtype='<f4').reshape(r['shape'])
        arrays[key]=a;paths[key]=dict(path=path,shape=r['shape'],sha256=r['sha256'],label=r['path'])
    for case in packet['cases']:
        for side in ['old_original','old_native93','fresh_original']:
            require(np.array_equal(arrays[(case['key'],side,'all')][np.asarray(case['scoring_mask'],dtype=bool)].view('<u4'),
                arrays[(case['key'],side,'selected')].view('<u4')),'selected vectors are not bit-exact scoring-mask slices')
    report['groups']=[];all_agree=True;numeric_failures=0
    for name,left,right in [('frozen_vs_fresh_original','old_original','fresh_original'),
                            ('frozen_original_vs_retained_native93','old_original','old_native93'),
                            ('fresh_original_vs_retained_native93','fresh_original','old_native93')]:
        folder=args.out/'raw/comparisons'/name;folder.mkdir(parents=True,exist_ok=False)
        pairs=[];metrics={};row_reports={};mapping=[]
        for case in packet['cases']:
            for kind in KINDS:
                safe=safe_name(case['key'],kind);a=arrays[(case['key'],left,kind)];b=arrays[(case['key'],right,kind)]
                metrics[safe]=reference.compare(a,b)
                if name=='frozen_vs_fresh_original':
                    old=current['cases'][KEYS.index(case['key'])]['anchor'][kind]
                    for field in ['max_absolute','min_row_cosine','mean_absolute','mean_error','rms','original_l2','native_l2','original_min','original_max','native_min','native_max']:
                        require(abs(metrics[safe][field]-old[field])<=2e-7*max(1.,abs(old[field])),'retained producer independent metric differs')
                positions=list(range(len(case['ids'])))if kind=='all'else[i for i,v in enumerate(case['scoring_mask'])if v]
                row_reports[safe]=diagnostics(np,a,b,positions,case['ids'])
                pairs.append(dict(name=safe,original=paths[(case['key'],left,kind)],native=paths[(case['key'],right,kind)]))
                mapping.append(dict(adapter_name=safe,key=case['key'],original_case_id=case['id'],kind=kind,shape=list(a.shape),left=left,right=right,
                                    left_sha256=paths[(case['key'],left,kind)]['sha256'],right_sha256=paths[(case['key'],right,kind)]['sha256']))
        adapters=reference.write_pair(folder/'original.gguf',folder/'native.gguf',pairs)
        actual=reference.actual_ref(args.diff,folder/'original.gguf',folder/'native.gguf',folder/'diff.json',folder/'diff.log',metrics)
        all_agree=all_agree and actual['all_metrics_agree'];numeric_failures+=sum(not v['passed']for v in metrics.values())
        group=dict(name=name,actual_ref=actual,metrics=metrics,adapters=adapters,mapping=mapping,rows=row_reports,
                   report=identity(folder/'diff.json'),log=identity(folder/'diff.log'))
        report['groups'].append(group);write(args.out/'review/compare.json',report)
    require(all_agree and len(report['groups'])==3 and sum(len(g['metrics'])for g in report['groups'])==18,'all eighteen measurement contracts must complete')
    report.update(completed=True,structural_complete=True,matrices_compared=18,comparator_invocations=3,numeric_failures=numeric_failures,
                  all_metrics_agree=all_agree,numerical_qualified=numeric_failures==0,causal_attribution_complete=False,
                  new_gold=False,model_forward_count=0,original_forward_count=0,native_forward_count=0)
    return int(numeric_failures>0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['fetch','compare'])
    parser.add_argument('--reviewed-source',required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--diff',type=Path);parser.add_argument('--build-proof',type=Path)
    args=parser.parse_args();review=args.out/'review';review.mkdir(parents=True,exist_ok=True)
    report=dict(schema='crispembed-moedern-three-doc-compare-only-v2',source=args.reviewed_source,phase=args.phase,completed=False,
                publication_qualified=False,new_gold=False,model_forward_count=0,original_forward_count=0,native_forward_count=0,
                models_accessed=False,inference_framework_imported=False,tokenizer_payload_accessed=False)
    code=1
    try:
        p=verify(args.reviewed_source);report['source_profile_sha256']=sha(PROFILE);report['service']=admission(args.reviewed_source,args.out)
        for key in list(os.environ):
            if key.startswith('CRISPEMBED_'):os.environ.pop(key)
        if args.phase=='fetch':fetch(args,p,report);code=0
        else:
            require(args.diff and args.diff.is_file(),'actual compiled Ref comparator required')
            require(args.build_proof and args.build_proof.is_file(),'compiled comparator source/command proof required')
            proof=read(args.build_proof)
            expected=['c++','-std=c++17','-O2','-ffp-contract=off','-I',str(ROOT/'src'),str(ROOT/'tests/moedern-native-v2/diff.cpp'),'-o',str(args.diff)]
            require(proof['argv']==expected and proof['comparator']==identity(args.diff),'actual comparator command/binary identity differs')
            for name,pin in proof['logs'].items():checked(args.out/'review'/name,pin)
            report['comparator_build']=proof;report['comparator']=identity(args.diff);code=compare(args,p,report)
    except Exception as error:report['error']=type(error).__name__+': '+str(error)
    finally:
        write(review/(args.phase+'.json'),report)
        if args.phase=='fetch':write(args.out/'fetch.json',report)
        files={str(f.relative_to(args.out)):identity(f)for f in sorted(args.out.rglob('*'))if f.is_file()and f.name!='manifest.json'}
        write(review/'manifest.json',files)
    return code


if __name__=='__main__':raise SystemExit(main())
