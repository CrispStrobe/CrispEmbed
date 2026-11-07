"""Frozen three-gate panel replay contracts. References remain provisional."""
import hashlib,importlib.util,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/heldout'));import execution as ex
require=ex.require;sha=ex.sha
CHANGED={'CRISPEMBED_LIGHTON4B_PHYSICAL_GDN_SOURCE_OPS','CRISPEMBED_LIGHTON4B_PHYSICAL_Q4_REPACK','CRISPEMBED_LIGHTON4B_PHYSICAL_Q6_TILED'}
CFG_SHA='bb43eee1f6717b21d3625693c9b1934c0c6f16ccc2a6c7b7a3d25426e95649c4'
def admit(cfg):
 require(hashlib.sha256((json.dumps(cfg,indent=2)+'\n').encode()).hexdigest()==CFG_SHA,'immutable corrected panel protocol')
 require(cfg['cases']==ex.prepare.CASE_IDS and cfg['cap']==1536 and cfg['context']==2048 and cfg['threads']==4 and cfg['worker_seconds']==6000 and cfg['arm_seconds']==1200,'frozen selection/capacity/budgets')
 require(set(cfg['changed_gates'])==CHANGED and cfg['qualified']is False and cfg['references_expert_adjudicated']is False and not cfg['reference_forward_allowed']and not cfg['heldout_tuning_allowed'],'closed diagnostic scope')
 selection=ROOT/'docs/lighton4b-heldout-v1.json';selected=json.loads(selection.read_text());ex.prepare.admit(selected);require(sha(selection)==cfg['selection_protocol_sha256'],'original selection unchanged')
 for case in selected['cases']:
  for key in ['source','gold']:require(sha(ROOT/case[key])==case[key+'_sha256'],'unchanged frozen '+key)
 require(sha(ROOT/'tests/regression/images/cc0/ground_truth.json')==selected['reference']['gold_registry_sha256'],'unchanged original reference registry')
 original=ROOT/'docs/lighton4b-heldout-executor-v1.json';require(sha(original)==cfg['frozen_executor_sha256'],'original reference protocol unchanged');old=json.loads(original.read_text())
 require(cfg['model_sha256']==old['candidate_sha256'],'same weights')
 require(sha(ROOT/'tests/lighton4b_native_document.cpp')==cfg['driver_sha256']and sha(ROOT/'CMakeLists.txt')==cfg['cmake_sha256'],'only reviewed CLI/CMake exceptions')
 for path,digest in json.loads(selection.read_text())['reference']['scorer'].items():require(sha(ROOT/path)==digest,'unchanged scorer')
 return old

def gates(old,new):
 before={k:v for k,v in old.items()if k.startswith('CRISPEMBED_')};after={k:v for k,v in new.items()if k.startswith('CRISPEMBED_')}
 for key in ['CRISPEMBED_LIGHTON4B_PROVISIONAL_HELDOUT','CRISPEMBED_LIGHTON4B_FROZEN_CASES']:before[key]=after[key]='1'
 for key in CHANGED:before.setdefault(key,'0')
 # These two diagnostic settings were absent in earlier saved environments.
 for key in set(before)|set(after):before.setdefault(key,'0');after.setdefault(key,'0')
 require({k for k in before if before[k]!=after[k]}==CHANGED,'exact three-gate comparison')
 require(all(before[k]=='0'and after[k]=='1'for k in CHANGED),'fixed three-gate values')
 return before,after

def archive_closed(folder,arm):
 """Authenticate historical raw ZIP without extracting or duplicating it."""
 record=arm['raw_rows'];root=Path(folder);a=record['archive'];p=root/a['file'];require(p.stat().st_size==a['bytes']and sha(p)==a['sha256'],'whole historical raw archive SHA')
 with zipfile.ZipFile(p)as z:
  expected={('source.logits.'+str(i)+'.f32'if a['file'].startswith('source')else'greedy.'+str(i)+'.logits.f32'):row for i,row in enumerate(record['rows'])}
  require(len(z.infolist())==len(expected)and set(z.namelist())==set(expected),'closed raw ZIP census')
  for name,row in expected.items():require(z.getinfo(name).file_size==row['bytes']==248320*4,'full vocabulary archive member')
 return {'sha256':a['sha256'],'bytes':a['bytes'],'rows':len(record['rows']),'scope':'Whole ZIP identity and member census rechecked; prior independent finite/argmax/per-row audit retained, not rerun.'}

def raw_equal(folder,arm):
 root=Path(folder)
 for i,row in enumerate(arm['raw_rows']['rows']):require(sha(root/f'greedy.{i}.logits.f32')==row['sha256'],'all old native rows byte exact')

def native_complete(summary,ids,source_fed):
 require(summary['generated_tokens']==len(ids)and 1<=len(ids)<=1536 and summary['eos']==(ids[-1]in[248044,248046])and(summary['eos']or len(ids)==1536),'closed native EOS/cap')
 require(summary['provisional_heldout_cap']==1536 and summary['source_fed_vision']==source_fed and summary['teacher_positions']==1,'actual heldout role')
 require(summary['converted_f32_weight_entries']==summary['converted_f32_weight_bytes']==0,'no F32 text weight materialization')
