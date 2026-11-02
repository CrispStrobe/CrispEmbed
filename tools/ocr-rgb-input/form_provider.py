#!/usr/bin/env python3
"""Host-only exact-version producer/image admission. No decoder/model import."""
import argparse,hashlib,importlib,importlib.metadata,json,os,sys,urllib.error,urllib.request,zipfile,io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(Path(__file__).parent),str(ROOT/'tools/ocr-real-scans')]
import form_contract as c
from retained_inputs import indexed_outputs

def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def api(path):
 r=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/'+path,headers={'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(r,timeout=60)as response:data=response.read(2**20+1)
 c.require(len(data)<=2**20,'Bounded service metadata required');return json.loads(data)

def initial_review(cfg,out):
 pin=cfg['initial_artifact'];run=api('actions/runs/40776499260')
 c.require((run['head_sha'],run['path'],run['conclusion'],run['run_attempt'])==('813e1ec9b813ebf0dc72c8faf5841659a024fbce','.github/workflows/ocr-rgb-input-audit-v1.yml','failure',1),'Initial19 service differs')
 row=api('actions/artifacts/'+str(pin['id']));c.require(row['size_in_bytes']==pin['bytes']and row['digest']=='sha256:'+pin['sha256']and row['workflow_run']['id']==40776499260 and not row['expired'],'Initial review artifact identity differs')
 class Stop(urllib.request.HTTPRedirectHandler):
  def redirect_request(self,*a,**k):return None
 request=urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(pin['id'])+'/zip',headers={'Authorization':'Bearer '+os.environ['GH_TOKEN']})
 try:response=urllib.request.build_opener(Stop).open(request,timeout=60)
 except urllib.error.HTTPError as e:
  c.require(e.code==302 and e.headers['Location'].startswith('https://'),'Signed artifact redirect required')
  response=urllib.request.urlopen(e.headers['Location'],timeout=60) # Fresh request: no Authorization at storage.
 with response:data=response.read(pin['bytes']+1)
 c.require(len(data)==pin['bytes']and hashlib.sha256(data).hexdigest()==pin['sha256'],'Initial review ZIP SHA differs')
 (out/'initial-review.zip').write_bytes(data)
 with zipfile.ZipFile(io.BytesIO(data))as z:
  c.require(len(z.namelist())==len(set(z.namelist()))and sum(x.file_size for x in z.infolist())<16*2**20,'Initial review ZIP layout differs')
  report=z.read('rgb-input-out/review/rgb-input.json')
 c.require(hashlib.sha256(report).hexdigest()==c.INITIAL_REPORT,'Initial19 report SHA differs')
 d=json.loads(report);c.require(d['source']=='813e1ec9b813ebf0dc72c8faf5841659a024fbce'and d['completed']and d['pages']==20 and d['compared_pages']==19 and len(d['rows'])==20,'Initial19 scope incomplete')
 return c.INITIAL_REPORT

def stream(url,target,pin):
 c.require(url.startswith('https://')and not target.exists(),'Fresh HTTPS retained member required')
 h=hashlib.sha256();count=0
 try:
  with urllib.request.urlopen(url,timeout=90)as response,target.open('xb')as out:
   while data:=response.read(65536):
    count+=len(data);c.require(count<=pin['bytes'],'Retained member exceeds extent');out.write(data);h.update(data)
 except Exception:raise ValueError('Retained member transfer rejected; task-owned partial preserved')from None
 c.require(count==pin['bytes']and h.hexdigest()==pin['sha256'],'Retained member size/SHA differs')

def main(args):
 c.source();cfg=c.declaration()
 c.require(os.environ.get('GITHUB_ACTIONS')=='true'and os.environ.get('RUNNER_ENVIRONMENT')=='github-hosted'and
  os.environ.get('GITHUB_SHA')==__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'Genuine exact-source hosted provider required')
 c.require(os.environ.get('GH_TOKEN')and os.environ.get('KAGGLE_API_TOKEN')and os.environ.get('KAGGLE_ACCOUNT'),'Existing retained-read credentials required beforefetch')
 args.out.mkdir(parents=True,exist_ok=False);proof=dict(schema='crispembed-ocr-rgb-form-provider-v1',completed=False,model_execution=False,original_rerun=False)
 try:
  proof['initial_report_sha256']=initial_review(cfg,args.out)
  c.require(importlib.metadata.version('kaggle')=='2.2.2'and sys.prefix!=sys.base_prefix,'Exact isolated Kaggle SDK required')
  owner=importlib.metadata.distribution('kaggle');c.require(Path(owner._path).resolve().is_relative_to(Path(sys.prefix).resolve()),'Kaggle tooling owner escaped isolated venv')
  mod=importlib.import_module('kaggle.api.kaggle_api_extended');c.require(Path(mod.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()),'Loaded SDK escaped isolated venv');client=mod.KaggleApi()
  try:client.authenticate()
  except Exception:raise ValueError('Existing retained-read credential rejected')from None
  pin=cfg['retained_original']
  with client.build_kaggle_client()as api_client:
   files,proof['original']=indexed_outputs(api_client.kernels.kernels_api_client,mod.ApiGetKernelRequest,mod.ApiListKernelSessionOutputRequest,mod.ApiGetKernelSessionStatusRequest,os.environ['KAGGLE_ACCOUNT'],pin)
  report_member='ocr-real-scans/result.json';c.require(report_member in files and pin['image_member']in files,'Exact original/image members absent')
  stream(files[report_member],args.out/'original-result.json',dict(bytes=pin['report_bytes'],sha256=pin['report_sha256']))
  original=json.loads((args.out/'original-result.json').read_text());c.require(original['commit']==pin['source']and original['mode']=='ref-minicpm'and original['scope']['selected_page_ids']==[c.PAGE_ID]and original['scope']['manifest_sha256']==c.old.MANIFEST['sha256'],'Original form source/scope differs')
  proof['original_report_sha256']=pin['report_sha256']
  stream(files[pin['image_member']],args.out/(c.PAGE_ID+'.jpg'),pin['image'])
  proof['completed']=True;c.verify_provider(args.out,proof)
 except Exception as error:proof['completed']=False;proof['error']=type(error).__name__+': '+str(error);raise
 finally:write(args.out/'provider.json',proof)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
