"""Retained e3 result reanalysis. No builds/forwards and no historical override."""
import hashlib,json,struct,subprocess,zipfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[2]
BASE='9b0f833ccae0cd129326bbd5cc256541af45c3b3'
SOURCE='e3df7cfd2e38f6f1bd20d68f1eb05dce11f7cd9d'
PATH=ROOT/'docs/wasm-layer3-f32-reanalysis-v1.json'
INITIAL=ROOT/'tests/results/packaging/wasm-layer3-f32-risk-initial.json'
OWNED={'.github/workflows/wasm-layer3-f32-reanalysis-v1.yml','docs/wasm-layer3-f32-reanalysis-v1.json','docs/wasm-layer3-f32-reanalysis-v1.md','docs/wasm-layer3-f32-risk-initial.md','tests/results/packaging/wasm-layer3-f32-risk-initial.json','tools/wasm-layer3-f32-reanalysis/protocol.py','tools/wasm-layer3-f32-reanalysis/replay.py','tests/test_wasm_layer3_f32_reanalysis_v1.py'}
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def source():
 cfg=json.loads(PATH.read_text());git=lambda *a:subprocess.check_output(['git',*a],cwd=ROOT)
 require(cfg['schema']=='crispembed-wasm-layer3-f32-reanalysis-v1'and cfg['base']==BASE and cfg['producer_source']==SOURCE,'Exact reanalysis declaration differs')
 require(cfg['comparisons']==25 and cfg['expected_pass']==7 and cfg['expected_fail']==18 and cfg['gates']=={'max_absolute':.002,'minimum_row_cosine':.99999},'Frozen comparisons/gates/negative counts differ')
 require(cfg['model_forward']is False and cfg['operator_forward']is False and cfg['build']is False and cfg['optional_f64_dot']=='omitted-not-retyped'and cfg['qualification']is False,'Compare-only scope differs')
 initial=json.loads(INITIAL.read_text());require(cfg['producer']=={'run':initial['service']['run'],**{k:{n:initial['artifacts'][k][n]for n in ['id','bytes','sha256']}for k in ['review','raw']}},'Exact initial immutable service/artifacts required')
 old=json.loads((ROOT/'docs/wasm-layer3-f32-v1.json').read_text());required=set(old['protected_sha256'])|set(old['owned_sha256'])|{'docs/wasm-layer3-f32-v1.json'}
 require(set(cfg['protected_sha256'])==required and set(cfg['owned_sha256'])==OWNED-{str(PATH.relative_to(ROOT))},'Exact old/new closure required')
 require(set(git('diff','--name-only',BASE,'HEAD').decode().splitlines())<=OWNED and git('show','HEAD:'+str(PATH.relative_to(ROOT)))==PATH.read_bytes(),'Undeclared/dirty replay source')
 for group in ['protected_sha256','owned_sha256']:
  for name,pin in cfg[group].items():
   raw=(ROOT/name).read_bytes();require(sha(raw)==pin and git('show','HEAD:'+name)==raw,'Dirty/changed source: '+name)
   if group=='protected_sha256':require(git('show',SOURCE+':'+name)==raw,'Frozen e3 source changed: '+name)
 return cfg

def archive(path,maximum):
 """Read only bounded unique regular members; never execute retained binaries."""
 result={}
 with zipfile.ZipFile(path)as z:
  infos=z.infolist();require(len(infos)<=100 and sum(v.file_size for v in infos)<=maximum,'Bounded retained ZIP required')
  for info in infos:
   name=PurePosixPath(info.filename)
   require(not info.is_dir()and not name.is_absolute()and '\\'not in info.filename and all(v not in('','.','..')for v in name.parts)and info.filename not in result and not info.flag_bits&1 and((info.external_attr>>16)&0o170000)!=0o120000,'Unsafe/duplicate retained ZIP member')
   value=z.read(info);require(len(value)==info.file_size,'Retained ZIP extent differs');result[info.filename]=value
 return result

def pairs(outputs):
 """Literal e3 comparison ordering, including unchanged original native targets."""
 rows=[]
 def add(name,a,b):rows.append((name,a,b))
 for axis in ['native','candidate']:
  add('backend.'+axis,outputs['native1']['cast_'+axis]['file'],outputs['wasm']['cast_'+axis]['file'])
  add('threads.'+axis,outputs['native1']['cast_'+axis]['file'],outputs['native2']['cast_'+axis]['file'])
 for backend in ['native1','native2','wasm']:
  for axis in ['native','candidate']:
   add(backend+'.same_input.'+axis,outputs[backend]['quant_'+axis]['file'],outputs[backend]['cast_'+axis]['file'])
   for target in ['native1','native2']:add(backend+'.old_'+target+'.'+axis,'browser/cast-targets/'+target+'-native.f32',outputs[backend]['cast_'+axis]['file'])
  add(backend+'.natural_input_sensitivity',outputs[backend]['cast_native']['file'],outputs[backend]['cast_candidate']['file'])
 require(len(rows)==25 and len({v[0]for v in rows})==25,'Exact25 pair coverage required');return rows


def controls(review):
 native=json.loads(review['review/native-controls.json']);browser=json.loads(review['browser/cast-report.json'])
 keys={'positive','wrong_mode','wrong_extents','nonfinite_input','nonfinite_weight','output_preserved','coefficient_output_preserved','recovery'}
 require(native.pop('threads')==[1,2]and set(native)==keys and all(v is True for v in native.values()),'Actual native controls incomplete')
 require(browser['completed']is True and browser['baseline_completed']is True and browser['isolated']is True and browser['pthreads']is True and not browser['page_errors']and set(browser['controls'])==keys and all(v is True for v in browser['controls'].values()),'Actual browser controls incomplete')
 return browser
