"""Separate remaining-seven controller; historical guards/outputs stay immutable."""
import hashlib,json,re,subprocess
from pathlib import Path
import ultrabert_q4k_sharded as legacy
import ultrabert_reviewed_q4_contract as previous
ROOT=legacy.ROOT
BASE='fffd7bf635986799685dadef4c05e364193e046b'
PATH=ROOT/'docs/ultrabert-remaining-q4-v1.json'
REVIEW=previous.REVIEW
FIRST_REVIEW=ROOT/'tests/results/ultrabert/first-q4-shard0-review.json'
OWNED={'docs/ultrabert-remaining-q4-v1.json','docs/ultrabert-remaining-q4-v1.md','tools/portfolio/ultrabert_remaining_q4_contract.py','tools/portfolio/ultrabert_remaining_q4.py','tests/test_ultrabert_remaining_q4.py','.github/workflows/ultrabert-remaining-q4-v1.yml'}
require,read,sha=legacy.require,legacy.read,legacy.sha

def approval(kind,cfg=None):
 cfg=cfg or declaration();require(kind in ('freeze','model'),'No other model/resume activation')
 return cfg['reviewed_dependencies'][kind]

def validate(cfg):
 old=previous.declaration();review=read(FIRST_REVIEW)
 require(cfg['schema']=='crispembed-ultrabert-remaining-q4-v1'and cfg['base']==BASE and cfg['runtime_source']==legacy.RUNTIME and cfg['ggml']==old['ggml'],'Exact separately closed source required')
 require(cfg['model']==previous.MODEL and cfg['recipe']==legacy.FIRST_RECIPE and cfg['actual_stored_types']=={'F32':206,'Q4_K':12},'Same immutable candidate required')
 require(cfg['activation']==dict(reviewed=True,recipe=legacy.FIRST_RECIPE,shards=list(range(1,8)))and cfg['resume']is False and cfg['workflow_hours']==6,'Only remaining seven sequential shards permitted')
 require(cfg['counts']==dict(new_rows=4797,new_public_calls=9594,imported_rows=686,imported_public_calls=1372,aggregate_rows=5483,aggregate_public_calls=10966,queries=300,original_forwards=0,f32_forwards=0,quantizer_invocations=0),'Forward/input counts differ')
 require(cfg['gates']==old['gates']and all(cfg[k]is False for k in ['publication_qualified','release_qualified','default_enabled','standard_header_publication_qualified','q8_activation_authorized'])and cfg['legacy_header_from_frozen018_quantizer']is True,'No numerical/default/header qualification waiver')
 require(cfg['reviewed_dependencies']=={k:old['reviewed_dependencies'][k]for k in ['freeze','model']}and cfg['freeze_review_sha256']==old['freeze_review_sha256']and cfg['freeze_manifest_sha256']==old['freeze_manifest_sha256'],'Frozen original/F32/model dependencies changed')
 first=cfg['shard0'];pin=first['artifact']
 require(pin==dict(reviewed=True,artifact_id=12891218847,source='a5feac83e177d7e08cfa23384089f7584f0736a9',sha256='f0be04cc0893eaf1ab814a3aadf2fe25d125ab6ac14321e6c1059084443f92e6',bytes=19663308,name='ultrabert-reviewed-q4-shard0-payload-40855927708',receipt_bytes=9767,receipt_sha256='ff561a165792fe696daccca1ae007694e5a95f131c30e5c92eab5746f5a1e8e1'),'Exact retained shard0 artifact required')
 require(sha(FIRST_REVIEW)==first['independent_review_sha256']and review['source']==pin['source']and review['run']==40855927708 and review['completed']is True and review['rows']==686 and review['raw_and_normalized_public_calls']==1372 and review['strict_f32_vector_flags']is False and review['actual_ref']['exit']==1 and review['model']==cfg['model'],'Preserve actual shard0 negative')
 require(hashlib.sha256((json.dumps(first['manifest'],indent=2,allow_nan=False)+'\n').encode()).hexdigest()==first['manifest_sha256']==review['small_members']['payload/manifest.json']['sha256'],'Exact complete imported manifest digest required')
 require(first['manifest']['shard.json']==dict(bytes=pin['receipt_bytes'],sha256=pin['receipt_sha256'])and all(first['manifest'][n]==p for n,p in review['streams'].items()),'Shard0 report/streams differ')
 require(first['source_binding']['audit_source']==pin['source']and first['source_binding']['runtime_source']==legacy.RUNTIME and first['source_binding']['ggml']==cfg['ggml']and first['source_binding']['runtime_git_objects']==read(legacy.PROTOCOL)['runtime_git_objects']and first['source_binding']['protocol_sha256']==sha(legacy.PROTOCOL)and first['source_binding']['controller_sha256']==sha(ROOT/'tools/portfolio/ultrabert_q4k_sharded.py')and first['source_binding']['new_controller_sha256']==sha(ROOT/'tools/portfolio/ultrabert_reviewed_q4_shard0.py')and first['source_binding']['new_protocol_sha256']==sha(previous.PATH),'Exact old runtime/controller/protocol binding required')
 expected_services={}
 for key,path in [('freeze',previous.REVIEW),('model',previous.MODEL_REVIEW)]:
  service=read(path)['service'];expected_services[key]={k:service[k]for k in ['id','head_sha','path','event','status','conclusion','run_attempt']}
 expected_services['shard0']=dict(id=40855927708,head_sha=pin['source'],path='.github/workflows/ultrabert-reviewed-q4-shard0-v1.yml',event='workflow_dispatch',status='completed',conclusion='success',run_attempt=1)
 require(cfg['service_dependencies']==expected_services,'Exact prior service binding required')
 return cfg

def declaration():return validate(read(PATH))

def verify_source(source):
 cfg=declaration();git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(re.fullmatch('[0-9a-f]{40}',source or '')and git('rev-parse','HEAD').decode().strip()==source and git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Exact reviewed caller source required')
 old=previous.declaration();required=set(old['protected_sha256'])|set(old['owned_sha256'])|{str(previous.PATH.relative_to(ROOT)),str(FIRST_REVIEW.relative_to(ROOT)),'docs/ultrabert-first-q4-shard0-result.md'}
 require(set(cfg['protected_sha256'])==required and set(cfg['owned_sha256'])==OWNED-{str(PATH.relative_to(ROOT))},'Complete inherited/new source closure required')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED|{'PLAN.md'},'Undeclared source changes')
 for group in ['protected_sha256','owned_sha256']:
  for name,pin in cfg[group].items():
   data=(ROOT/name).read_bytes();require(hashlib.sha256(data).hexdigest()==pin and git('show','HEAD:'+name)==data,'Source identity changed: '+name)
   if group=='protected_sha256':require(git('show',BASE+':'+name)==data,'Historical protected source changed: '+name)
 require(cfg['protected_git_objects']==old['protected_git_objects']and not git('status','--porcelain','--untracked-files=no'),'Production closure/clean source required')
 for name,obj in cfg['protected_git_objects'].items():require(git('rev-parse','HEAD:'+name).decode().strip()==obj and git('rev-parse',BASE+':'+name).decode().strip()==obj,'Production object changed: '+name)
 return cfg
