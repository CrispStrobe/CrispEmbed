"""Same exact-v3 GET primitives, new closed caller; all544 inputs before arrays."""
import argparse
import concurrent.futures
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import urllib.request
sys.path.insert(0,str(Path(__file__).resolve().parent))
import lane as c
import retained_fetch_gha as delivery
import retained_v3_gha as structural

class NoPayloadImports(delivery.NoArrays):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.',1)[0]=='PIL':raise ImportError('Image import blocked before all544 input hashes')
        return super().find_spec(fullname,path,target)

def image(url,path,pin):
    c.require(url.startswith('https://')and not path.exists()and not path.is_symlink(),'fresh HTTPS frozen PNG required')
    path.parent.mkdir(parents=True,exist_ok=True);n=0;h=hashlib.sha256()
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Accept-Encoding':'identity'}),timeout=180)as r,path.open('xb')as f:
        size=int(r.headers.get('Content-Length','-1'))
        c.require(r.status==200 and 0<size<=pin['maximum_bytes']and r.headers.get('Content-Encoding','identity')=='identity','bounded exact unencoded PNG GET required')
        while data:=r.read(min(2**20,size-n+1)):
            n+=len(data);c.require(n<=size,'frozen PNG extent exceeded');h.update(data);f.write(data)
    c.require(n==size and h.hexdigest()==pin['sha256'],'frozen original PNG SHA differs')
    return dict(bytes=n,sha256=h.hexdigest())

def run(source,scratch,out):
    cfg=c.guard(source,True);out.mkdir(parents=True,exist_ok=False);scratch.mkdir(parents=True,exist_ok=False)
    result=dict(schema='crispembed-unlimited-reused-q4-input-v1',source=source,completed=False,original_forwards=0,native_forwards=0,quantizer_runs=0,files={},images={})
    save=lambda:c.atomic(out/'inputs.json',result);save();block=NoPayloadImports()
    c.require(not any(n in sys.modules for n in ('numpy','torch','tensorflow','transformers','PIL')),'array/image/framework imported before input admission')
    sys.meta_path.insert(0,block)
    try:
        result['resources']=c.resources(scratch,True);save()
        c.require(os.environ.get('KAGGLE_API_TOKEN')and os.environ.get('KAGGLE_ACCOUNT'),'existing private exact-version read-service configuration absent')
        import importlib.metadata
        for package,version in [('kaggle','2.2.2'),('kagglesdk','0.1.38')]:c.require(importlib.metadata.version(package)==version,'exact SDK package version required')
        sdk=importlib.import_module('kaggle.api.kaggle_api_extended');api=sdk.KaggleApi();api.authenticate()
        urls,result['service']=delivery.output_urls(api,os.environ['KAGGLE_ACCOUNT'])
        reviewed=json.loads(c.REVIEW.read_text());c.require(result['service']==reviewed['transfer']['service'],'reviewed immutable v3 service index differs')
        root=scratch/'panel';root.mkdir();panel=root/'panel.json'
        result['panel']=delivery.download(urls[delivery.PREFIX+'panel.json'],panel,dict(bytes=structural.PANEL_BYTES,sha256=structural.PANEL_SHA))
        producer=json.loads(panel.read_text());expected=(c.ROOT/'tests/results/unlimited-ocr2/mixed-q4-v3-panel.json').read_bytes()
        c.require(panel.read_bytes()==expected and len(producer['files'])==540 and sum(p['bytes']for p in producer['files'].values())==7901314457,'exact retained540 inventory required')
        def one(item):
            name,pin=item;rel=delivery.safe_name(name);key=delivery.PREFIX+name
            c.require(key in urls,'immutable v3 member absent: '+name)
            try:return name,delivery.download(urls[key],root.joinpath(*rel.parts),pin)
            except Exception:raise ValueError('Retained member rejected: '+name)from None
        with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
            for name,proof in pool.map(one,producer['files'].items()):result['files'][name]=proof;save()
        for case,pin in cfg['images'].items():
            c.require(pin['member']in urls,'frozen original PNG absent; no rendering fallback')
            result['images'][case]=image(urls[pin['member']],scratch/'pages'/(case+'.png'),pin);save()
        c.require(len(result['files'])==540 and len(result['images'])==4 and not any(n in sys.modules for n in ('numpy','PIL')),'all544 payloads must authenticate before array/image imports')
        result.update(all_544_authenticated_before_array_image_import=True,panel_files_authenticated=540,additional_images_authenticated=4);save()
        sys.meta_path.remove(block)
        proof=structural.audit(scratch,out,source);c.atomic(out/'structural.json',proof)
        c.require(proof['completed']and proof['model']==cfg['candidate']and proof['actual_inventory']==reviewed['actual_inventory']and proof['protected']==reviewed['protected'],'reviewed candidate structural evidence differs')
        result.update(completed=True,structural_report_sha256=c.digest(out/'structural.json'));save();return result
    except Exception as error:
        result['error']='Input authentication rejected: '+type(error).__name__
        if str(error).startswith('Retained member rejected: '):result['failed_member']=str(error).split(': ',1)[1]
        save();raise ValueError(result['error'])from None
    finally:
        if block in sys.meta_path:sys.meta_path.remove(block)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True)
    for n in ('scratch','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.source,a.scratch,a.out)
