"""Authenticate exact retained compare-only92 ZIPs and every member before frameworks."""
import argparse
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import shutil
import sys
import urllib.request
import urllib.error
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,PATH,KEY,CID,atomic,require,sha,verify,hosted


def credential():
    value=os.environ.get('GH_TOKEN')
    if not value:
        name=os.environ.get('CRISPEMBED_ARTIFACT_TOKEN_FILE','');p=Path(name)
        require(name and p.is_file()and not p.is_symlink()and '/kaggle/input/'in str(p.resolve())and p.stat().st_size<=4096,'approved small mounted artifact credential required')
        value=p.read_text().strip()
    require(isinstance(value,str)and 0<len(value)<=4096 and not any(c.isspace()for c in value),'artifact credential invalid')
    return value


def api(path):
    request=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/'+path,
        headers={'Authorization':'Bearer '+credential(),'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*a,**k):return None
    try:
        with urllib.request.build_opener(NoRedirect).open(request,timeout=90)as response:data=response.read(2*2**20+1)
        require(len(data)<=2*2**20,'service metadata exceeds bound');return json.loads(data)
    except Exception:raise ValueError('Bounded authenticated GitHub metadata request failed')from None


def download(pin,path):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self,*a,**k):return None
    request=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(pin['id'])+'/zip',
        headers={'Authorization':'Bearer '+credential(),'Accept':'application/vnd.github+json'})
    try:
        try:response=urllib.request.build_opener(NoRedirect).open(request,timeout=90)
        except urllib.error.HTTPError as error:
            require(error.code==302 and error.headers['Location'].startswith('https://'),'signed HTTPS artifact redirect required')
            # A new request, without API Authorization, to the signed storage URL.
            response=urllib.request.urlopen(error.headers['Location'],timeout=180)
        with response,path.open('xb')as stream:
            count=0
            while data:=response.read(2**20):
                count+=len(data);require(count<=pin['bytes'],'artifact exceeded exact size');stream.write(data)
        require(count==pin['bytes']and sha(path)==pin['sha256'],'whole artifact extent/SHA differs')
    except Exception:raise ValueError('Authenticated retained artifact transfer rejected')from None


def inventory(z,pins):
    rows=z.infolist();names=[r.filename for r in rows]
    require(len(rows)==len(set(names))and set(names)==set(pins),'exact ZIP member set required')
    require(sum(r.file_size for r in rows)<=32*2**20,'uncompressed retained ZIP exceeds bound')
    for r in rows:
        p=PurePosixPath(r.filename)
        require(not p.is_absolute()and '..'not in p.parts and '\\'not in r.filename and not r.is_dir()and
                ((r.external_attr>>16)&0o170000)!=0o120000,'unsafe retained ZIP member')
        pin=pins[r.filename];data=z.read(r)
        require(len(data)==pin['bytes']and hashlib.sha256(data).hexdigest()==pin['sha256'],'retained member differs: '+r.filename)
    return len(rows)


def transfer(url,path,pin):
    require(url.startswith('https://')and 0<pin['bytes']<=1024**3,'bounded immutable model transfer required')
    h=hashlib.sha256();n=0
    with urllib.request.urlopen(url,timeout=180)as response,Path(path).open('xb')as output:
        while data:=response.read(2**20):
            n+=len(data);require(n<=pin['bytes'],'published model exceeded exact extent');h.update(data);output.write(data)
    require(n==pin['bytes']and h.hexdigest()==pin['sha256'],'published model SHA/extent differs')


def run(source,out):
    c=verify(source);provider=hosted(source);out.mkdir(parents=True,exist_ok=False)
    report=dict(schema='crispembed-moedern-first-row-input-auth-v1',source=source,completed=False,provider=provider,
        protocol_sha256=sha(PATH),models_accessed=False,framework_imported=False,original_forward_count=0,native_forward_count=0)
    save=lambda:atomic(out/'prepare.json',report);save()
    try:
        require(not any(n in sys.modules for n in ('numpy','torch','transformers','sentence_transformers')),'framework imported before retained-byte authentication')
        from common import budget
        report['resources']=budget.admit(out);budget.single_file_limit();save()
        if os.environ.get('GITHUB_ACTIONS')=='true':
            service=api('actions/runs/'+os.environ['GITHUB_RUN_ID'])
            require(service['head_sha']==source and service['id']==int(os.environ['GITHUB_RUN_ID'])and service['status']=='in_progress'and
                    service['path']=='.github/workflows/moedern-first-row-stages-v1.yml','genuine caller workflow/source differs')
            report['caller_service']={k:service[k]for k in ('id','head_sha','run_attempt','path')}
        producer=c['retained'];run=api('actions/runs/'+str(producer['run']))
        require(run['head_sha']==producer['source']and run['status']=='completed'and run['conclusion']=='failure'and run['run_attempt']==1 and
                run['path']=='.github/workflows/moedern-three-doc-compare-v2.yml','preserved numeric-negative producer identity differs')
        report['producer']={k:run[k]for k in ('id','head_sha','conclusion','run_attempt','path')};report['artifacts']={};save()
        for name in ('small','raw'):
            pin=producer[name];actual=api('actions/artifacts/'+str(pin['id']))
            require(actual['id']==pin['id']and actual['workflow_run']['id']==producer['run']and actual['workflow_run']['head_sha']==producer['source']and
                actual['name']==pin['name']and actual['size_in_bytes']==pin['bytes']and actual['digest']=='sha256:'+pin['sha256']and not actual['expired'],'retained service artifact identity differs')
            report['artifacts'][name]={k:actual[k]for k in ('id','name','size_in_bytes','digest','expires_at')};save();budget.live(out);download(pin,out/(name+'.zip'));budget.live(out)
        with zipfile.ZipFile(out/'small.zip')as z:
            report['small_members_authenticated']=inventory(z,producer['small_members'])
            manifest=json.loads(z.read('manifest.json'));compare=json.loads(z.read('compare.json'))
            require(compare['completed']is True and compare['structural_complete']is True and compare['matrices_compared']==18 and compare['numeric_failures']==12 and compare['all_metrics_agree']is True,'retained18 negative comparison evidence differs')
            atomic(out/'retained-compare.json',compare)
        rawpins={k:v for k,v in manifest.items()if k.startswith('raw/')or k=='fetch.json'}
        with zipfile.ZipFile(out/'raw.zip')as z:
            report['raw_members_authenticated']=inventory(z,rawpins)
            oldpath='raw/retained/three-doc-input/input-packet.json';old=json.loads(z.read(oldpath))
            require(old['completed']is True and old['keys']==['d:24726600','d:24928817','d:5849439']and len(old['cases'])==3 and
                    old['source_protocol_sha256']==sha(ROOT/'docs/moedern-three-doc-attribution-v1.json'),'unmodified retained three-document packet required')
            atomic(out/'retained-three-doc-input.json',old)
            case=next(r for r in old['cases']if r['key']==KEY)
            require(case['id']==CID and len(case['ids'])==307 and case['ids'][58]==29084 and sum(case['scoring_mask'])==278,'exact first failing document/token differs')
            for side in ('original','native'):
                for kind in ('all','selected'):
                    pin=case['anchors'][side][kind];member='raw/retained/three-doc-input/'+pin['path']
                    require(rawpins[member]['sha256']==pin['sha256']and rawpins[member]['bytes']==pin['bytes'],'frozen packet anchor identity differs')
                    target=out/pin['path'];target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(member))
            # Authenticate both retained fresh-original finals too, without relabeling them as gold.
            report['six_first_doc_anchor_identities']={k:v for k,v in rawpins.items()if CID in k}
            require(len(report['six_first_doc_anchor_identities'])==6,'all six relevant packet anchors must remain authenticated')
        packet=dict(schema='crispembed-moedern-first-row-input-v1',key=KEY,case=case,completed=True,
            source_protocol_sha256=sha(PATH),retained_three_case_packet_sha256=manifest[oldpath]['sha256'],new_gold=False)
        atomic(out/'input-packet.json',packet)
        report.update(completed=True,input_packet_sha256=sha(out/'input-packet.json'),resource_final=budget.live(out));save();return report
    except Exception as error:report['error']=type(error).__name__+': '+str(error);save();raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.reviewed_source,a.out)
