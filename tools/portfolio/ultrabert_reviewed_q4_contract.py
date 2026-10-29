"""Separate reviewed shard0 activation; c3d/018/first-controller guards unchanged."""
import hashlib,json,subprocess
from pathlib import Path
import ultrabert_q4k_sharded as legacy
ROOT=legacy.ROOT
BASE='64f57398deb64e98062ec332ff11f72b1cbe2bec'
CONVERSION='c3d2fc051f6632e06d5e7ba69ac700a03a703e22'
PATH=ROOT/'docs/ultrabert-reviewed-q4-shard0-v1.json'
REVIEW=ROOT/'tests/results/ultrabert/retained-freeze-review.json'
MODEL_REVIEW=ROOT/'tests/results/ultrabert/first-q4-model-40843704011-review.json'
MODEL_RECEIPT=ROOT/'tests/results/ultrabert/first-q4-model-40843704011.json'
MODEL=dict(bytes=576997440,sha256='087c9be0fa50ba57db74e650889521c985d306863c94aabf731764538eb4533c')
OWNED={'docs/ultrabert-reviewed-q4-shard0-v1.json','docs/ultrabert-reviewed-q4-shard0-v1.md',
 'tools/portfolio/ultrabert_reviewed_q4_contract.py','tools/portfolio/ultrabert_reviewed_q4_shard0.py',
 'tests/test_ultrabert_reviewed_q4_shard0.py','.github/workflows/ultrabert-reviewed-q4-shard0-v1.yml',
 str(MODEL_REVIEW.relative_to(ROOT)),str(MODEL_RECEIPT.relative_to(ROOT))}
require,read,sha=legacy.require,legacy.read,legacy.sha

def approval(kind,cfg=None):
 cfg=cfg or declaration();require(kind in ['freeze','model'],'Resume/other dependency pending; no activation')
 value=cfg['reviewed_dependencies'][kind];require(value['reviewed']is True,'Reviewed dependency required')
 return value

def phase_admission(phase,cfg=None):
 cfg=validate(cfg or declaration());require(phase=='shard0','Only reviewed shard0 permitted; no model conversion/other shards')
 return cfg

def validate(cfg):
 require(cfg['schema']=='crispembed-ultrabert-reviewed-q4-shard0-v1'and cfg['base']==BASE and cfg['runtime_source']==legacy.RUNTIME,
         'Exact new source declaration required')
 require(cfg['activation']==dict(reviewed=True,recipe=legacy.FIRST_RECIPE,shards=[0])and cfg['rows_in_shard0']==686
         and cfg['allowed_shards']==[0]and cfg['actual_stored_types']=={'F32':206,'Q4_K':12},'Only686-row conservative shard0 reviewed')
 require(cfg['gates']==read(legacy.PROTOCOL)['gates']and cfg['original_forwards']==cfg['f32_forwards']==0
         and all(cfg[k]is False for k in ['default_enabled','publication_qualified','release_qualified','q8_activation_authorized']),
         'Original/F32/otherprecision/default/release forbidden')
 model=cfg['reviewed_dependencies']['model']
 require(model==dict(reviewed=True,artifact_id=12885865196,source=CONVERSION,
  sha256='c2919ce6b58db9083ecbbebf8cffe310e55593200a1569663dfa7bbfe7799405',bytes=531993310,
  name='ultrabert-first-q4-model-payload-40843704011',receipt_sha256='02578697aa01516825e46bb1153bb55fa2d9604567007dfcdc86daad00445eec',
  receipt_bytes=49715,model=MODEL),'Immutable reviewed model/archive/receipt differs')
 require(cfg['reviewed_dependencies']['resume']is None and cfg['legacy_header_from_frozen018_quantizer']is True
         and cfg['standard_header_publication_qualified']is False,'No resume/canonicalheader activation')
 return cfg

def model_inventory(producer):
 import re
 expected={'token_embd.weight':[768,49152],'output_norm.weight':[768]}
 for layer in range(24):
  prefix='blk.'+str(layer)+'.'
  for name in ['attn_norm','ffn_norm']:expected[prefix+name+'.weight']=[768]
  for name in ['attn_q','attn_k','attn_v','attn_output']:expected[prefix+name+'.weight']=[768,768]
  for name in ['ffn_gate','ffn_up']:expected[prefix+name+'.weight']=[768,1024]
  expected[prefix+'ffn_down.weight']=[1024,768]
 inventory=producer['inventory'];require(set(inventory)==set(expected),'Exact218 schema names required')
 counts={'F32':0,'Q4_K':0}
 for name,shape in expected.items():
  row=inventory[name];kind='Q4_K'if re.fullmatch(r'blk\.(20|21|22|23)\.ffn_(gate|up|down)\.weight',name)else'F32'
  require(row['shape']==shape and row['type']==kind and re.fullmatch('[a-f0-9]{64}',row['payload_sha256'])is not None,'Model role/shape/hash differs: '+name)
  counts[kind]+=1
 require(counts=={'F32':206,'Q4_K':12},'Converted storage role counts differ')
 return counts

def declaration():
 cfg=validate(read(PATH));previous=read(ROOT/'docs/ultrabert-first-q4-shard-v1.json');review=read(MODEL_REVIEW);producer=read(MODEL_RECEIPT)
 model_inventory(producer)
 require(cfg['ggml']==legacy.read(legacy.PROTOCOL)['ggml']and cfg['recipe']==legacy.FIRST_RECIPE,'Runtime backend/recipe differs')
 require(cfg['reviewed_dependencies']['freeze']==previous['reviewed_dependencies']['freeze']and cfg['freeze_review_sha256']==previous['freeze_review_sha256']
         and cfg['freeze_manifest_sha256']==previous['freeze_manifest_sha256'],'Frozen original/F32 reference lineage differs')
 require(sha(MODEL_REVIEW)==cfg['model_review_sha256']and sha(MODEL_RECEIPT)==cfg['reviewed_dependencies']['model']['receipt_sha256'], 'Immutable independent model proof differs')
 require(review['model']==producer['model']==MODEL and producer['completed']and producer['recipe']==legacy.FIRST_RECIPE
         and producer['actual_stored_types']=={'F32':206,'Q4_K':12}and producer['freeze_receipt_sha256']==cfg['reviewed_dependencies']['freeze']['receipt_sha256']
         and producer['source']['audit_source']==CONVERSION and producer['source']['runtime_source']==legacy.RUNTIME
         and review['qualification']is False and producer['original_forwards']==producer['f32_forwards']==producer['model_or_original_forwards']==0
         and producer['legacy_header_from_frozen018_quantizer']is True and producer['standard_header_publication_qualified']is False,'Reviewed conversion/provenance differs')
 return cfg

def verify_source(reviewed):
 cfg=declaration();git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(git('rev-parse','HEAD').decode().strip()==reviewed and git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Exact new caller source required')
 previous=read(ROOT/'docs/ultrabert-first-q4-shard-v1.json')
 required=set(previous['protected_sha256'])|set(previous['owned_sha256'])|{'docs/ultrabert-first-q4-shard-v1.json'}
 require(set(cfg['protected_sha256'])==required and set(cfg['owned_sha256'])==OWNED-{str(PATH.relative_to(ROOT))},'Complete immutable/new source closure required')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED,'Undeclared caller changes')
 for group in ['protected_sha256','owned_sha256']:
  for name,pin in cfg[group].items():
   raw=(ROOT/name).read_bytes();require(hashlib.sha256(raw).hexdigest()==pin and git('show','HEAD:'+name)==raw,'Source changed: '+name)
   if group=='protected_sha256':require(git('show',BASE+':'+name)==raw and git('show',CONVERSION+':'+name)==raw,'Frozen c3d source changed: '+name)
 require(cfg['protected_git_objects']==previous['protected_git_objects']and not git('status','--porcelain','--untracked-files=no'),'Production closure/cleanliness differs')
 for name,obj in cfg['protected_git_objects'].items():
  require(git('rev-parse',BASE+':'+name).decode().strip()==obj and git('rev-parse','HEAD:'+name).decode().strip()==obj,'Runtime/build/bindings changed: '+name)
 return cfg
