"""Selected retained mounted bytes admitted BEFORE numerical imports/model access."""
import hashlib,math,shutil,struct,sys,urllib.request
from pathlib import Path,PurePosixPath
from common import read,write,require,identity,sha

def safe(root,name):
 p=PurePosixPath(name);require(not p.is_absolute()and '..'not in p.parts and '\\'not in name,'Unsafe selected path')
 result=Path(root)/p;require(result.is_file()and not result.is_symlink()and result.resolve().is_relative_to(Path(root).resolve()),'Missing/unsafe selected retained member: '+name)
 return result

def native_ids(original,native,layer):
 require(len(original)==len(native)==4912 and layer in [0,1],'Canonical307x2 littleendian I64 required')
 o=list(struct.iter_unpack('<2q',original));n=list(struct.iter_unpack('<2q',native))
 require(all(0<=a<8 and 0<=b<8 and a!=b for a,b in o+n),'Expert I64 range/unique schema differs')
 changed=[i for i,(a,b)in enumerate(zip(o,n))if a!=b]
 require(changed==([]if layer==0 else[58]),'Retained native full-row delta differs')
 if layer==1:require(o[58]==(2,4)and n[58]==(2,6),'Frozen actual row58 experts differ')
 return dict(bytes=len(native),sha256=hashlib.sha256(native).hexdigest(),first_observed_sha256=True,pre_reviewed_sha256=False,semantic_rows_validated=307,changed_rows=changed,native_gold_regenerated=False)

def authenticate(root,out,c):
 require(not any(k in sys.modules for k in ['torch','numpy','transformers']),'No numerical import before selected authentication')
 out=Path(out);out.mkdir(parents=True,exist_ok=False);records={}
 for name,pin in c['capture']['pre_pinned_members'].items():
  p=safe(root,name);require(identity(p)==pin,'Retained selected bytes differ: '+name)
  dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);records[name]=pin
 original=read(out/'retained/output/raw/original/original.json');diag=read(out/'retained/output/raw/comparisons/diagnostic.json');tokens=read(out/'retained/output/raw/observed/doc-24726600/tokens.json')
 require(original['source']==diag['source']==c['capture']['source']and original['completed']is True and diag['completed']is True and original['original_forward_total_including_barrier']==5 and original['actual_operator_inputs_outputs_repeat_raw_bits_exact']is True and original['observer_feature_raw_bits_neutral']is True and diag['native93_final_anchor_exact']is True,'Actual completed neutral capture required')
 require(original['model']['versions']['torch']==c['torch']['version']and original['model']['versions']['numpy']==c['torch']['numpy']and original['hardware']['threads']==2 and original['hardware']['interop_threads']==1,'Original component runtime/thread profile differs')
 ids=[v[0]for v in struct.iter_unpack('<q',(out/c['ids']['path']).read_bytes())];mask=[v[0]for v in struct.iter_unpack('<q',(out/c['mask']['path']).read_bytes())]
 require(len(ids)==307 and ids[58]==29084 and tokens['ids']==ids and mask==tokens['effective_mask']==[1]*307,'Actual full token/mask/row58 input differs')
 arrays={x['stage']:x for x in original['arrays']};native=diag['cases'][0]['adapters']['native']['tensors'];integers=diag['cases'][0]['integer'];observed_ids=[]
 for layer in c['layers']:
  i=layer['index'];require(original['model']['loaded_tensor_sha256']['model.safetensors'][f'layers.{i}.ffn.router.weight']==layer['weight_sha256'],'Loaded original weight identity differs')
  for kind,p in layer['original'].items():
   stage=p['stage'];actual=arrays[stage]
   require({k:actual[k]for k in ['shape','dtype','bytes','sha256']}=={k:p[k]for k in ['shape','dtype','bytes','sha256']}and p['path']=='retained/output/raw/original/'+actual['path'],'Actual original operand mapping differs')
  # Bind the replay primitives to ACTUAL captured argument/return fingerprints.
  operators={r['stage']:r for r in original['operators']}
  for op_name,actual_name in [('softmax','aten.softmax.int'),('topk','aten.topk.default'),('div','aten.div.Tensor')]:
   op=operators[f'actual.layers.{i}.ffn.{op_name}'];require(op['operator']==actual_name,'Actual router operator differs')
   items=dict(op['input']['items']);args=items['args']['items'];require(items['kwargs']['items']==[]and len(args)==2,'Actual operator defaults/arguments differ')
   left=layer['original']['logits'if op_name=='softmax'else'probs'if op_name=='topk'else'values']
   require(args[0]['kind']=='tensor'and args[0]['shape']==left['shape']and args[0]['sha256']==left['sha256']and args[0]['dtype']=='torch.float32','Captured component input fingerprint differs')
   if op_name!='div':require(args[1]==dict(kind='int',value=-1 if op_name=='softmax'else 2),'Captured dimension/top-k differs')
   else:require(args[1]['sha256']==layer['original']['denominator']['sha256']and args[1]['shape']==[307,1],'Actual denominator input differs')
   if op_name=='topk':
    outputs=op['output']['items'];require([x['sha256']for x in outputs]==[layer['original'][k]['sha256']for k in ['values','indices']],'Actual top-k return fingerprints differ')
   else:require(op['output']['sha256']==layer['original']['probs'if op_name=='softmax'else'weights']['sha256'],'Actual operator return fingerprint differs')
  for kind,p in layer['native'].items():
   if kind=='indices':continue
   name=f'layer.{i}.ffn.'+dict(input='norm',logits='router',probs='router_probabilities',weights='topk_weights')[kind];actual=native[name]
   require({k:actual[k]for k in ['shape','bytes','sha256']}=={k:p[k]for k in ['shape','bytes','sha256']}and p['path']=='retained/output/raw/observed/'+actual['source'],'Actual native operand mapping differs')
  constraint=next(x for x in integers if x['stage']==f'layer.{i}.ffn.topk_indices')
  require(constraint==dict(stage=f'layer.{i}.ffn.topk_indices',exact=i==0,changed_rows=i,first_changed_rows=[]if i==0 else[58]),'Actual integer difference scope changed')
  n=layer['native']['indices'];p=safe(root,n['path']);require(p.stat().st_size==4912,'Native I64 selected extent differs')
  data=p.read_bytes();proof=native_ids((out/layer['original']['indices']['path']).read_bytes(),data,i)
  dest=out/n['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data);records[n['path']]={k:proof[k]for k in ['bytes','sha256']};observed_ids.append(dict(layer=i,path=n['path'],**proof))
 for layer in c['layers']:
  for side in ['original','native']:
   for kind,p in layer[side].items():
    data=(out/p['path']).read_bytes();require(len(data)==p['bytes']and len(data)==math.prod(p['shape'])*(8 if p['dtype']=='i64'else 4),'Exact selected operand extent differs')
    if p['dtype']=='f32':require(all(math.isfinite(v[0])for v in struct.iter_unpack('<f',data)),'Nonfinite captured operand')
 report=dict(completed=True,capture_source=c['capture']['source'],version_label='v4',script_sha256=c['capture']['script_sha256'],selected_members=records,pre_pinned_member_count=len(c['capture']['pre_pinned_members']),semantic_i64_members=observed_ids,full400_outputs_authenticated=False,whole_zip_authenticated=False,original_full_model_forwards=0,source_offsets_rewritten=False)
 write(out/'prepared.json',report);return report

def verify_prepared(root,c):
 root=Path(root);saved=read(root/'prepared.json');require(saved['completed']is True and saved['capture_source']==c['capture']['source']and saved['version_label']=='v4','Selected authentication receipt absent')
 expected=set(c['capture']['pre_pinned_members'])|{r['native']['indices']['path']for r in c['layers']}
 require(set(saved['selected_members'])==expected,'Selected operand closure differs')
 for name,pin in saved['selected_members'].items():require(identity(safe(root,name))==pin,'Authenticated selected bytes changed')
 for name,pin in c['capture']['pre_pinned_members'].items():require(saved['selected_members'][name]==pin,'Pre-reviewed selected identity changed')
 for layer in c['layers']:
  p=layer['native']['indices']['path'];native_ids((root/layer['original']['indices']['path']).read_bytes(),(root/p).read_bytes(),layer['index'])
 return saved

def model(out,pin):
 out=Path(out);require(not out.exists()and pin['bytes']==7037952 and pin['sha256']=='0e2b8773abd8c78a5825280f9a73cb808b88d56f0675a8e4bb7f0509f5b6cc51','Exact published F32 packed model required')
 with urllib.request.urlopen(pin['url'],timeout=120)as response,out.open('xb')as target:
  count=0
  while chunk:=response.read(2**20):count+=len(chunk);require(count<=pin['bytes'],'Model byte bound');target.write(chunk)
 require(identity(out)=={k:pin[k]for k in ['bytes','sha256']},'Published F32 bytes differ')
 return identity(out)
