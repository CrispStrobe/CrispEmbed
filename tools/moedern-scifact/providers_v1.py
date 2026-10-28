"""Hosted immutable inputs; credentials never enter argv, logs or receipts."""
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import urllib.parse
import urllib.request
import zipfile
from contract_native_v1 import ROOT,require,sha


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,request,*args,**kwargs):
        result=super().redirect_request(request,*args,**kwargs)
        if result and urllib.parse.urlsplit(result.full_url).hostname!='api.github.com':
            result.remove_header('Authorization')
        return result


def credential():
    value=os.environ.get('GH_TOKEN')
    if not value:
        explicit=os.environ.get('GH_TOKEN_FILE')
        files=[Path(explicit)]if explicit else [Path(p)for p in glob.glob('/kaggle/input/**/gh_token.txt',recursive=True)]
        files=sorted({p.resolve()for p in files if p.is_file()and not p.is_symlink()and p.stat().st_size<=4096})
        require(len(files)==1,'one authorized artifact-read credential input required')
        value=files[0].read_text().strip()
    require(20<=len(value)<=4096 and '\n'not in value and '\r'not in value,'artifact-read credential format rejected')
    return value


def api(path,token):
    request=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/'+path,
                                 headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json'})
    try:
        with urllib.request.urlopen(request,timeout=90)as response:
            data=response.read(2*2**20+1);require(len(data)<=2*2**20,'API metadata exceeds bound');return json.loads(data)
    except Exception:raise ValueError('authenticated immutable artifact metadata request rejected')from None


def artifact(pin,run,source,target,token,cap):
    service=api('actions/runs/'+str(run),token)
    require((service['head_sha'],service['status'],service['conclusion'],service['run_attempt'])==(source,'completed','success',1),'old producer run differs')
    meta=api('actions/artifacts/'+str(pin['id']),token)
    require(meta['workflow_run']['id']==run and not meta['expired']and meta['size_in_bytes']==pin['bytes']and
            meta['digest']=='sha256:'+pin['sha256']and 0<pin['bytes']<=cap,'old artifact service identity differs')
    target=Path(target);opener=urllib.request.build_opener(Redirect());h=hashlib.sha256();count=0
    request=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(pin['id'])+'/zip',headers={'Authorization':'Bearer '+token})
    try:
        with opener.open(request,timeout=180)as response,target.open('xb')as output:
            while block:=response.read(2**20):
                count+=len(block);require(count<=pin['bytes'],'old artifact exceeds service extent');h.update(block);output.write(block)
    except Exception:raise ValueError('authenticated artifact transfer rejected; task-owned partial retained')from None
    require(count==pin['bytes']and h.hexdigest()==pin['sha256'],'old artifact ZIP bytes differ')
    return zipfile.ZipFile(target)


def old_original(mount,variant,scratch):
    """Exact old ten/20/three plus four captured expansion controls, no rerun."""
    config=json.loads((ROOT/'docs/moedern-q4k-padded.json').read_text());pin=config['originals'][variant]
    paths=[p for p in Path(mount).rglob('original.json')if p.is_file()and not p.is_symlink()and
           p.stat().st_size<2*2**20 and sha(p)==pin['report_sha256']]
    require(len(paths)<=1,'multiple old authored original mounts')
    if paths:root=paths[0].parent.parent
    else:
        require(variant=='1.5M','exact old authored27M/115M original mount required')
        root=Path(scratch)/'old-original';(root/'review').mkdir(parents=True,exist_ok=False)
        token=credential();p=json.loads((ROOT/'docs/moedern-text-composed.json').read_text())['primary']
        z=artifact(p['review_artifact'],p['run'],p['source'],Path(scratch)/'old-review.zip',token,2**20)
        data=z.read('moedern-original/review/original.json');require(hashlib.sha256(data).hexdigest()==pin['report_sha256'],'old authored original report differs')
        (root/'review/original.json').write_bytes(data);report=json.loads(data)
        z=artifact(p['raw_artifact'],p['run'],p['source'],Path(scratch)/'old-raw.zip',token,32*2**20)
        for row in report['stage_arrays']:
            if row['stage']not in('ordinary.features.token_embeddings','ordinary.selected','ordinary.features.query_expansion_positions'):continue
            rel=PurePosixPath(row['path']);require(not rel.is_absolute()and '..'not in rel.parts and '\\'not in row['path'],'unsafe old authored member')
            data=z.read('moedern-original/'+row['path'])
            require(len(data)==row['bytes']and hashlib.sha256(data).hexdigest()==row['sha256'],'old authored array differs')
            target=root/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    driver=module('authenticated_old_authored',ROOT/'tools/ci-heavy/moedern_q4k_padded.py')
    report,arrays=driver.original_inputs(root,variant,config)
    index={(r['case'],r['stage']):r for r in report['stage_arrays']}
    expansion=0
    for case in report['cases']:
        if case['role']!='query':continue
        row=index[case['id'],'ordinary.features.query_expansion_positions'];path=root/row['path']
        require(path.is_file()and path.stat().st_size==row['bytes']and sha(path)==row['sha256']and
                list(path.read_bytes())==[1-int(x)for x in case['supplied_attention_mask']],
                'captured original expansion/padding rule differs')
        expansion+=1
    require(expansion==4,'four actual old query expansion controls required')
    return root,report,arrays


def published_f32(variant,target):
    quant=module('frozen_q4k_converter',ROOT/'tools/moedern-colbert/quantize_q4k_padded.py');pin=quant.publication(variant)
    helper=module('immutable_moedern_download',ROOT/'tools/ci-heavy/moedern_q4k_padded.py')
    helper.stream_file(pin['url'],Path(target),pin);return pin
