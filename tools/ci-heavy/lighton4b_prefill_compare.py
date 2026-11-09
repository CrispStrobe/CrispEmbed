#!/usr/bin/env python3
"""Hosted read-only original BF16 vs retained Q8/Q4 stage comparison; no models."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import zipfile

ORIGINAL_REPORT_SHA='036a936cdbd360208b5cf54c2604d671b9a919f1c8777bc7cf97b2585b010a78'
SOURCE='8d3b2331d1b57e285f7ba348fc1eacd689b79aa8'
TARGETS=[('q4_k',41999584163,13401573855,'862f4be15fc64e9b14d4b26e5ee7862366435059fb0a828226bc351fae0607a1'),('q8_0',41999580781,13401793576,'a949c71bc59b61d96778f2d324b7da104ea54f98004bf9e1100167dd7d16e3eb')]
def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def download(url,path,headers=None,max_bytes=256*2**20):
    request=urllib.request.Request(url,headers=headers or {})
    with urllib.request.urlopen(request,timeout=90) as response,path.open('wb') as stream:
        length=response.headers.get('Content-Length');require(length is None or int(length)<=max_bytes,'download byte bound')
        count=0
        while block:=response.read(2**20):
            count+=len(block);require(count<=max_bytes,'download byte bound');stream.write(block)
    return count
def api(path):
    request=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/'+path,
        headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=60) as response:return json.load(response)
def compare(original,native):
    import numpy as np
    a=original.reshape(-1).astype(np.float64);b=native.reshape(-1).astype(np.float64)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'composed shape/finite mismatch')
    e=np.abs(a-b);na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));different=np.flatnonzero(e)
    cosine=float(a@b/(na*nb)) if na and nb else float(na==nb)
    row={'elements':int(a.size),'max_abs':float(e.max()),'mean_abs':float(e.mean()),'rms_error':float(np.sqrt(np.mean(e**2))),
        'cosine':cosine,'original_l2':na,'native_l2':nb,'original_abs_max':float(np.abs(a).max()),'native_abs_max':float(np.abs(b).max()),
        'different_values':int(np.count_nonzero(e)),'first_differing_flat_index':int(different[0]) if different.size else None,
        'raw_strict_flag':bool(e.max()>0.002 or cosine<0.99999)}
    if different.size:row.update(first_original=float(a[different[0]]),first_native=float(b[different[0]]))
    return row
def main():
    import numpy as np
    require(os.environ.get('GITHUB_ACTIONS')=='true','hosted-only comparison')
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(tempfile.mkdtemp(prefix='lighton4b-prefill-compare-'))
    report={'schema':'lighton4b-prefill-stage-compare/v1','status':'failed','model_bytes_downloaded':0,
        'original_forward_executed':False,'native_forward_executed':False,'decoded_parity_qualified':False,'recipes':{}}
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        provider=KaggleApi();provider.authenticate()
        ref=os.environ['KAGGLE_ACCOUNT']+'/lighton4b-bf16-targeted-original-cpu-prefill'
        provider.kernels_output(ref,str(scratch/'original'),file_pattern=r'^lighton4b-bf16-prefill/stages/(prefill\.json|[A-Za-z0-9_.]+\.f32)$',quiet=True)
        original=scratch/'original/lighton4b-bf16-prefill/stages';manifest=original/'prefill.json'
        require(sha(manifest)==ORIGINAL_REPORT_SHA,'original stage manifest differs')
        receipt=json.loads(manifest.read_text());require(receipt['source_sha']==SOURCE and receipt['observer_first_score_bit_exact'] and receipt['model_forward_calls']==1,'original capture control/source failed')
        require(len(receipt['stages'])==71 and sum(row['bytes']for row in receipt['stages'].values())==130163104,'original complete stage inventory differs')
        for name,row in receipt['stages'].items():
            require(row['file']==name+'.f32' and sha(original/row['file'])==row['sha256'] and (original/row['file']).stat().st_size==row['bytes'],'original stage digest differs')
        report['original']={'source_sha':SOURCE,'manifest_sha256':ORIGINAL_REPORT_SHA,'stages':71,'bytes':130163104,'observer_first_score_bit_exact':True}
        names=['processor.patches','vision.patch','vision.position','vision.embedded']+[f'vision.layer.{i}'for i in range(24)]+['vision.merged','text.embedded','prefill.input','prefill.positions']+[f'prefill.layer.{i}'for i in range(32)]+['prefill.norm','prefill.logits']
        for recipe,run,artifact,report_sha in TARGETS:
            item=api('actions/artifacts/'+str(artifact));require(item['workflow_run']['id']==run and not item['expired'] and item['name']=='heavy-'+str(run),'native artifact identity differs')
            require(item['workflow_run']['head_sha']=='fd150b98f488a6884a6e8d8127e7dfb263cab237','native source differs')
            archive=scratch/(recipe+'.zip')
            # urllib strips Authorization when following GitHub's signed redirect.
            # Never print requests, signed URLs or credential-bearing exceptions.
            class Redirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self,req,fp,code,msg,headers,newurl):
                    return urllib.request.Request(newurl,headers={'Accept-Encoding':'identity'})
            opener=urllib.request.build_opener(Redirect)
            request=urllib.request.Request(item['archive_download_url'],headers={'Authorization':'Bearer '+os.environ['GH_TOKEN']})
            with opener.open(request,timeout=90)as response,archive.open('wb')as stream:
                count=0
                while block:=response.read(2**20):count+=len(block);require(count<=256*2**20,'native archive bound');stream.write(block)
            require(count==item['size_in_bytes'],'native archive size differs')
            if item.get('digest'):require(item['digest']=='sha256:'+sha(archive),'native archive digest differs')
            with zipfile.ZipFile(archive)as z:
                reports=[n for n in z.namelist()if n=='native-document.json' or n.endswith('/native-document.json')];require(len(reports)==1,'native report inventory differs')
                raw=z.read(reports[0]);require(hashlib.sha256(raw).hexdigest()==report_sha,'native frozen report differs')
                native_report=json.loads(raw);require(native_report['recipe']==recipe and native_report['status']=='completed','native recipe/status differs')
                rows={}
                for name in names:
                    filename=name+'.f32';record=native_report['artifacts'][filename]
                    candidates=[n for n in z.namelist()if n.endswith('/'+filename) or n==filename];require(len(candidates)==1,'native stage inventory differs: '+name)
                    raw=z.read(candidates[0]);require(len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256'],'native stage hash differs: '+name)
                    a=np.fromfile(original/receipt['stages'][name]['file'],dtype='<f4').reshape(receipt['stages'][name]['shape'])
                    mapping='original contiguous tensor flattened; batch=1 axis removed only'
                    if name=='prefill.positions':
                        require(a.shape==(4,1,154) and np.array_equal(a[0,0],np.arange(154)),'original chronology plane differs')
                        a=a[1:,0,:];mapping='verified4-axis original chronology plane0 removed; THW axes retained'
                    b=np.frombuffer(raw,dtype='<f4');row=compare(a,b);row['original_shape']=receipt['stages'][name]['shape'];row['comparison_shape']=list(a.shape);row['native_elements']=b.size;row['layout_mapping']=mapping
                    row['original_dtype']=receipt['stages'][name]['original_dtype'];row['native_sha256']=record['sha256'];row['original_sha256']=receipt['stages'][name]['sha256'];rows[name]=row
                order=[name for name in names if name!='text.embedded']
                report['recipes'][recipe]={'run_id':run,'artifact_id':artifact,'source_sha':native_report['source_sha'],'native_report_sha256':report_sha,
                    'stages':rows,'first_differing_stage':next((n for n in order if rows[n]['different_values']),None),
                    'first_raw_strict_stage':next((n for n in order if rows[n]['raw_strict_flag']),None),'native_decoded':native_report['decoded'],
                    'original_first_id':27,'native_first_id':int(np.argmax(np.frombuffer(z.read(next(n for n in z.namelist()if n.endswith('/prefill.logits.f32'))),dtype='<f4')))}
            archive.unlink()
        report['status']='compared'
    except Exception as error:
        report['error_type']=type(error).__name__
        if isinstance(error,ModuleNotFoundError):report['missing_module']=error.name
        if isinstance(error,ValueError):report['error']=str(error)[:200]
        else:report['error']='authenticated retained stage comparison failed'
        raise RuntimeError(report['error']) from None
    finally:(out/'prefill-compare.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
