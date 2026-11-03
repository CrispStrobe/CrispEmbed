#!/usr/bin/env python3
"""Closed prerequisite and post-analysis only; frozen19c alone runs seven arms."""
import argparse,hashlib,json,os,sys,urllib.error,urllib.request,zipfile
from pathlib import Path,PurePosixPath
sys.path.insert(0,str(Path(__file__).resolve().parent))
import seven_contract as c

def api(path):
 req=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/'+path,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(req,timeout=90)as r:b=r.read(2*2**20+1)
 c.require(len(b)<=2*2**20,'Small service metadata bound');return json.loads(b)

def prepare(out,reviewed):
 source=c.source_guard(reviewed,True);c.old.resources(out,out.parent/'mini-stage');cfg=c.declaration();pin=cfg['completed_analysis_artifact'];expected=json.loads(c.REVIEW.read_text());service=api('actions/runs/'+str(expected['service']['id']))
 c.require({k:service[k]for k in expected['service']}==expected['service'],'Exact completed cec service differs')
 desc=api('actions/artifacts/'+str(pin['id']));c.require(desc['workflow_run']['id']==service['id']and desc['size_in_bytes']==pin['size_in_bytes']and desc['digest']==pin['digest']and not desc['expired'],'Exact small proof artifact differs')
 destination=out/'approved-analysis';destination.mkdir(parents=True,exist_ok=False)
 class Stop(urllib.request.HTTPRedirectHandler):
  def redirect_request(self,*a,**k):return None
 req=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(pin['id'])+'/zip',headers={'Authorization':'Bearer '+os.environ['GH_TOKEN']})
 try:r=urllib.request.build_opener(Stop).open(req,timeout=90)
 except urllib.error.HTTPError as e:
  c.require(e.code==302 and e.headers['Location'].startswith('https://'),'Signed HTTPS redirect required');r=urllib.request.urlopen(e.headers['Location'],timeout=180)
 else:r.close();raise ValueError('Expected signed storage redirect')
 archive=destination/'review.zip';count=0
 with r,archive.open('xb')as f:
  while b:=r.read(2**20):count+=len(b);c.require(count<=pin['size_in_bytes'],'Small proof ZIP bound');f.write(b)
 c.require(count==pin['size_in_bytes']and 'sha256:'+c.sha(archive)==pin['digest'],'Whole small ZIP identity differs')
 members={}
 with zipfile.ZipFile(archive)as z:
  c.require(len(z.infolist())==134 and sum(i.file_size for i in z.infolist())<=32*2**20,'Bounded complete proof inventory required')
  for i in z.infolist():
   rel=PurePosixPath(i.filename);c.require(not i.is_dir()and not rel.is_absolute()and '..'not in rel.parts and '\\'not in i.filename and i.filename not in members and ((i.external_attr>>16)&0o170000)!=0o120000,'Unsafe proof member');b=z.read(i);members[i.filename]=b
 c.require(set(members)==set(expected['small_members']),'Exact134 small members required')
 for name,p in expected['small_members'].items():c.require(len(members[name])==p['bytes']and hashlib.sha256(members[name]).hexdigest()==p['sha256'],'Proof member identity differs: '+name)
 report=json.loads(members['analysis/analysis.json']);c.verify_analysis(report);(destination/'analysis.json').write_bytes(members['analysis/analysis.json'])
 canonical=expected['canonical_parent']['artifact'];remote=api('actions/artifacts/'+str(canonical['id']));c.require(remote['workflow_run']['id']==40783734766 and remote['size_in_bytes']==canonical['bytes']and remote['digest']=='sha256:'+canonical['sha256']and not remote['expired'],'Retained old23 canonical parent unavailable/changed')
 c.atomic(destination/'authenticated.json',dict(completed=True,source=source,service=expected['service'],zip_sha256=c.sha(archive),analysis_sha256=c.sha(destination/'analysis.json'),all134_members_authenticated=True,old23_canonical_parent=canonical,model_access=False,native_forwards=0))

def verify(out,reviewed):
 c.source_guard(reviewed,True);cfg=c.declaration();root=out/'approved-analysis';proof=json.loads((root/'authenticated.json').read_text());pin=cfg['completed_analysis_artifact']
 c.require(proof['completed']and proof['all134_members_authenticated']is True and proof['source']['source']==reviewed and proof['source']['contract_sha256']==c.sha(c.PATH)and 'sha256:'+c.sha(root/'review.zip')==pin['digest']and c.sha(root/'analysis.json')==cfg['completed_analysis_report']['sha256']==proof['analysis_sha256'],'Approved dependency/source changed')
 return c.verify_analysis(json.loads((root/'analysis.json').read_text()))

def finish(result,path):
 cfg=c.declaration();expected=[tuple(r)for r in cfg['remaining_arms']];rows=result.get('new_vision',[]);keys=[tuple(r.get(k)for k in ['id','dtype','graph'])for r in rows]
 complete=(result.get('old_seven_completed')is True and len(rows)==7 and keys==expected and all(r.get('completed')is True and r.get('pixels_bit_exact')is True and set(r.get('vision',{}).get('stages',{}))=={'vision_'+str(i)for i in range(6)}and r['vision'].get('actual_ref',{}).get('stages')==6 and r['vision']['actual_ref'].get('all_metrics_agree')is True for r in rows))
 result.update(completed=complete,new_vision_comparisons=sum(len(r.get('vision',{}).get('stages',{}))for r in rows),combined_vision_comparisons=138+sum(len(r.get('vision',{}).get('stages',{}))for r in rows),qualified=False,publication_qualified=False)
 c.atomic(path,result);c.require(complete,'Seven-only structural continuation incomplete; atomic receipt retained')

def finalize(out,reference,scratch,diff,reviewed):
 prior=verify(out,reviewed);from graph_panel import verify_reference
 original,identity=verify_reference(reference);c.require(identity==prior['original'],'Frozen original identity differs');refs={r['id']:r for r in original['cases']};root=out/'continuation';old=json.loads((root/'continuation.json').read_text());cfg=c.declaration()
 result=dict(schema='crispembed-minicpm-seven-arm-v2',completed=False,qualified=False,source=c.source_guard(reviewed,True),original=identity,original_rerun=False,runtime_math_changes=False,prior_analysis_sha256=c.sha(out/'approved-analysis/analysis.json'),old_continuation_sha256=c.sha(root/'continuation.json'),old_seven_completed=old['completed'],carried_vision=prior['cases'],new_vision=[],historical_numeric_failures_preserved=True,limits='Only seven missing ordinary CLI forwards plus two old-profile projection probes. Old23 decoded/138 vision comparisons remain frozen; six-stage vision comparison on newly saved traces adds42 comparisons without any second forward. Coverage is not qualification.')
 try:
  c.require(old['source']['source']==reviewed and old['source']['old_source']==c.old.OLD_SOURCE and old['old23_mutated']is False and old['original_rerun']is False,'Frozen seven caller source/history differs')
  built=out/'build-continue.json';build=json.loads(built.read_text());c.require(c.sha(built)==old['build_receipt_sha256']and build['completed']is True and build['mode']=='continue'and build['source']['source']==reviewed and c.sha(diff)==build['binary_sha256']['diff'],'Actual same-source comparator/build receipt changed')
  result['build_receipt_sha256']=c.sha(built)
  catalog=json.loads(c.old.CATALOG.read_text());panel=c.old.checked(root/'retained-v3','panel.json',catalog['panel']);c.require(old['carried_rows']==json.loads(panel.read_text())['cases']and len(old['carried_rows'])==23,'All original23 native rows must remain exact')
  reconstruction=root/'retained-v3/old-reconstruction-manifest.json';c.require(c.sha(reconstruction)==old['carried']['reconstruction_sha256']and json.loads(reconstruction.read_text())['all_omitted_adapter_bytes_reconstructed_and_sha_verified']is True,'Old23 reconstruction proof changed')
  expected=[tuple(v)for v in cfg['remaining_arms']];keys=[tuple(r[k]for k in ['id','dtype','graph'])for r in old['new_cases']];c.require(keys==expected,'Exactly seven absent native attempts required')
  import vision
  for row in old['new_cases']:
   key=tuple(row[k]for k in ['id','dtype','graph']);item=dict(id=key[0],dtype=key[1],graph=key[2],completed=False);result['new_vision'].append(item)
   try:
    result['resources_before_comparison']=c.old.resources(out,scratch)
    folder=root/'new'/f'{key[1]}-graph{key[2]}'/key[0];c.require(c.sha(folder/'files.json')==row['files_manifest_sha256']and c.sha(root/row['case_report'])==row['case_report_sha256'],'New canonical manifest/case changed')
    files=json.loads((folder/'files.json').read_text())
    for name,pin in files.items():c.old.checked(folder,name,pin)
    item.update(vision.compare(folder,refs[key[0]],reference,diff,scratch/f'{key[1]}-graph{key[2]}'/key[0],out/'new-vision'/f'{key[1]}-graph{key[2]}'/key[0]))
   except Exception as e:item['error']=type(e).__name__+': '+str(e)
   c.atomic(out/'seven-v2.json',result)
 finally:c.atomic(out/'seven-v2.json',result)
 finish(result,out/'seven-v2.json')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','verify','finalize']);p.add_argument('--reviewed-source',required=True);p.add_argument('--out',type=Path,required=True)
 for name in ['reference','scratch','diff']:p.add_argument('--'+name,type=Path)
 a=p.parse_args()
 try:
  if a.phase=='finalize':c.require(a.reference and a.scratch and a.diff,'Explicit retained analysis paths required');finalize(a.out,a.reference,a.scratch,a.diff,a.reviewed_source)
  elif a.phase=='prepare':prepare(a.out,a.reviewed_source)
  else:verify(a.out,a.reviewed_source)
 except Exception as e:c.atomic(a.out/('seven-v2-rejection-'+a.phase+'.json'),dict(completed=False,phase=a.phase,error=type(e).__name__+': '+str(e),qualified=False));raise
