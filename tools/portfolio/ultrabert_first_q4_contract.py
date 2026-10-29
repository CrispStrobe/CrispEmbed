"""New closed activation owner; every old018/37/08 source stays immutable."""
import hashlib,json,subprocess
from pathlib import Path
import ultrabert_q4k_sharded as legacy
ROOT=legacy.ROOT
BASE='e36597379aa5c8a43426309a4ebc7faf429686ea'
PATH=ROOT/'docs/ultrabert-first-q4-shard-v1.json'
REVIEW=ROOT/'tests/results/ultrabert/retained-freeze-review.json'
OWNED={'docs/ultrabert-first-q4-shard-v1.json','docs/ultrabert-first-q4-shard-v1.md','tools/portfolio/ultrabert_first_q4_contract.py','tools/portfolio/ultrabert_first_q4_shard.py','tests/test_ultrabert_first_q4_shard.py','.github/workflows/ultrabert-first-q4-shard-v1.yml'}
require,read,sha=legacy.require,legacy.read,legacy.sha

def approval(kind,cfg=None):
 cfg=cfg or read(PATH);p=cfg['reviewed_dependencies'].get(kind)
 require(type(p)is dict and p.get('reviewed')is True,'pending reviewed '+kind+' dependency; no model access')
 for key in ['artifact_id','source','sha256','bytes','name','receipt_sha256','receipt_bytes']:
  require(key in p,'incomplete reviewed '+kind+' identity')
 if kind=='model':
  require(type(p.get('model'))is dict and set(p['model'])=={'bytes','sha256'}and p['model']['bytes']>0 and len(p['model']['sha256'])==64,'explicit converted model SHA required before shard activation')
 return p

def phase_admission(phase,cfg=None):
 cfg=cfg or declaration();require(phase in ['model','shard0'],'Only model conversion and first shard0 phases exist');approval('freeze',cfg)
 if phase=='shard0':
  approval('model',cfg);a=cfg['activation'];require(a['reviewed']is True and a['shards']==[0]and a['recipe']==legacy.FIRST_RECIPE,'pending explicit shard0 activation; no forward')
 return cfg

def declaration():
 cfg=read(PATH);review=read(REVIEW);previous=read(legacy.PROTOCOL)
 require(cfg['schema']=='crispembed-ultrabert-first-q4-shard-v1'and cfg['base']==BASE and cfg['runtime_source']==legacy.RUNTIME and cfg['ggml']==previous['ggml'],'Exact new caller/old runtime declaration required')
 require(cfg['recipe']==legacy.FIRST_RECIPE and cfg['actual_stored_types']=={'F32':206,'Q4_K':12}and cfg['rows_in_shard0']==686 and cfg['allowed_shards']==[0],'Existing conservative recipe/shard0 only')
 require(cfg['gates']==previous['gates']and cfg['original_forwards']==cfg['f32_forwards']==0 and all(cfg[k]is False for k in ['default_enabled','publication_qualified','release_qualified','q8_activation_authorized']),'Old gates and zero original/F32 forwards required')
 p=cfg['reviewed_dependencies']['freeze'];require(p['reviewed']is True and p['artifact_id']==review['payload_artifact']['id']and p['source']==review['service']['head_sha']and p['sha256']==review['payload_artifact']['digest'].removeprefix('sha256:')and p['bytes']==review['payload_artifact']['size_in_bytes']and p['name']==review['payload_artifact']['name']and p['receipt_sha256']==review['freeze_receipt']['sha256']and p['receipt_bytes']==review['freeze_receipt']['bytes'],'Exact reviewed freeze lineage required')
 require(sha(REVIEW)==cfg['freeze_review_sha256']and cfg['freeze_manifest_sha256']==review['payload_manifest']['sha256'],'Immutable independent freeze proof required')
 return cfg

def verify_source(reviewed):
 cfg=declaration();git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(git('rev-parse','HEAD').decode().strip()==reviewed and git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Exact new caller source required')
 required=set(read(legacy.PROTOCOL)['owned_sha256'])|{str(legacy.PROTOCOL.relative_to(ROOT)),str(REVIEW.relative_to(ROOT)),'docs/ultrabert-retained-freeze-result.md'}
 require(set(cfg['protected_sha256'])==required and set(cfg['owned_sha256'])==OWNED-{str(PATH.relative_to(ROOT))},'Exact immutable/new source closure required')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED,'Undeclared first-shard source change')
 for group in ['protected_sha256','owned_sha256']:
  for name,pin in cfg[group].items():
   raw=(ROOT/name).read_bytes();require(hashlib.sha256(raw).hexdigest()==pin and git('show','HEAD:'+name)==raw,'Changed source: '+name)
   if group=='protected_sha256':require(git('show',BASE+':'+name)==raw,'Historical source changed: '+name)
 require(set(cfg['protected_git_objects'])=={'src','CMakeLists.txt','ggml','models','python/crispembed','crispembed-sys','crispembed','flutter','Cargo.toml','Cargo.lock','cmake'},'Exact production/build object closure required')
 require(not git('status','--porcelain','--untracked-files=no'),'Dirty tracked source')
 for path,obj in cfg['protected_git_objects'].items():
  require(git('rev-parse',BASE+':'+path).decode().strip()==obj and git('rev-parse','HEAD:'+path).decode().strip()==obj,'Production/build objects changed: '+path)
 return cfg
