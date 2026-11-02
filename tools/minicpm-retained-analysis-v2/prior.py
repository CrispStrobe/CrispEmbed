"""Small retained proof only: carry completed six line arms, never their forwards."""
import hashlib,json,os,urllib.error,urllib.request,zipfile
from pathlib import Path,PurePosixPath
from analysis_contract import INITIAL,atomic,require,sha


def api(path):
 req=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/'+path,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(req,timeout=90)as response:raw=response.read(2*2**20+1)
 require(len(raw)<=2*2**20,'Bounded GitHub metadata required');return json.loads(raw)


def authenticate(cfg,out):
 out.mkdir(parents=True,exist_ok=False);s=cfg['prior_service'];pin=cfg['prior']['review']
 service=api('actions/runs/'+str(s['run']))
 require((service['head_sha'],service['path'],service['status'],service['conclusion'],service['run_attempt'])==(s['source'],s['workflow'],'completed','success',1),'Exact initial success/partial service differs')
 row=api('actions/artifacts/'+str(pin['id']))
 require(row['workflow_run']['id']==s['run']and row['size_in_bytes']==pin['bytes']and row['digest']=='sha256:'+pin['sha256']and not row['expired'],'Exact small artifact identity differs')
 class Stop(urllib.request.HTTPRedirectHandler):
  def redirect_request(self,*a,**k):return None
 req=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(pin['id'])+'/zip',headers={'Authorization':'Bearer '+os.environ['GH_TOKEN']})
 try:response=urllib.request.build_opener(Stop).open(req,timeout=90)
 except urllib.error.HTTPError as error:
  require(error.code==302 and error.headers['Location'].startswith('https://'),'Signed HTTPS artifact required')
  response=urllib.request.urlopen(error.headers['Location'],timeout=180) # no API Authorization forwarded
 else:response.close();raise ValueError('Expected signed artifact redirect')
 archive=out/'prior-review.zip';count=0
 with response,archive.open('xb')as target:
  while block:=response.read(2**20):count+=len(block);require(count<=pin['bytes'],'Small ZIP extent exceeded');target.write(block)
 require(count==pin['bytes']and sha(archive)==pin['sha256'],'Initial small whole ZIP SHA differs')
 initial=json.loads(INITIAL.read_text());members={}
 with zipfile.ZipFile(archive)as z:
  require(len(z.infolist())<=500 and sum(i.file_size for i in z.infolist())<=64*2**20,'Bounded small proof ZIP required')
  for info in z.infolist():
   n=PurePosixPath(info.filename);require(not info.is_dir()and not n.is_absolute()and'..'not in n.parts and'\\'not in info.filename and info.filename not in members and((info.external_attr>>16)&0o170000)!=0o120000,'Unsafe/duplicate initial proof member')
   raw=z.read(info);members[info.filename]=raw
 require(set(members)==set(initial['review_members']),'Exact initial proof inventory required')
 for name,p in initial['review_members'].items():require(len(members[name])==p['bytes']and hashlib.sha256(members[name]).hexdigest()==p['sha256'],'Initial member SHA differs: '+name)
 report=verify(members,cfg)
 require(hashlib.sha256(members['analysis/retained-v3/old-reconstruction-manifest.json']).hexdigest()==report['compaction']['reconstruction_sha256'],'Old full compaction proof differs')
 canonical=cfg['prior']['canonical'];kept=api('actions/artifacts/'+str(canonical['id']))
 require(kept['workflow_run']['id']==s['run']and kept['size_in_bytes']==canonical['bytes']and kept['digest']=='sha256:'+canonical['sha256']and not kept['expired'],'Immutable remote canonical parent unavailable/changed')
 manifest='analysis/retained-v3/old-reconstruction-manifest.json';(out/'old-reconstruction-manifest.json').write_bytes(members[manifest])
 atomic(out/'authenticated.json',dict(completed=True,source_service=s,artifact=pin,old_compaction_proof_sha256=hashlib.sha256(members[manifest]).hexdigest(),model_accessed=False,native_forwards=0))
 return report,members


def verify(members,cfg):
 old=json.loads(members['analysis/analysis.json'])
 require(old['completed']is False and old['comparisons']==36 and len(old['cases'])==23 and old['all_pixels_bit_exact']is True and old['model_accessed']is False and old['native_forwards']==0 and old['original_rerun']is False,'Exact initial structural partial outcome required')
 require(old['source']['source']==cfg['inherited_source']and old['original']['reference_sha256']==cfg['original_reference_sha256'],'Initial producer/original source differs')
 expected={('line',d,g)for d,g in [('f32',0),('f32',1),('f16',0),('f16',1),('q8_0',1),('q4_k',1)]};carried=[]
 for i,row in enumerate(old['cases']):
  key=(row['id'],row['dtype'],row['graph'])
  if not row['completed']:require(row['id']!='line'and 'known batch1'in row['error'],'Initial incomplete classification changed');continue
  require(key in expected and row['pixels_bit_exact']is True,'Only completed line arms may be carried')
  actual=json.loads(members[f'analysis/{i}/diff.json']);comparison=json.loads(members[f'analysis/{i}/comparison.json'])
  require(len(actual['stages'])==6 and len(row['vision']['stages'])==6 and row['vision']['actual_ref']['all_metrics_agree']is True and comparison['actual_ref']==row['vision']['actual_ref'],'Initial actual Ref coverage/metric agreement differs')
  by_name={r['name']:r for r in actual['stages']}
  for name,value in row['vision']['stages'].items():
   require(value['cpp_ref']==by_name[name]and value['passed']==bool(value['independent_passed']and by_name[name]['passed'])and all(v['passed']is True for v in value['metric_agreement'].values()),'Carried numerical/Ref flags changed')
  carried.append(key)
 require(len(carried)==6 and set(carried)==expected,'Exact six completed line arms required')
 return old
