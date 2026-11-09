#!/usr/bin/env python3
"""Read-only actual trace closure against authenticated operator reconstruction."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def require(ok,msg):
 if not ok:raise RuntimeError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 require(Path('/kaggle/input').is_dir()and Path('/kaggle/working').is_dir(),'Kaggle read-only worker only')
 import numpy as np
 require(np.__version__=='2.3.5','numpy pin')
 cfg=json.loads((ROOT/'docs/lighton4b-physical-form-trace-v2.json').read_text());out=Path(os.environ['HEAVY_OUT']);report={'schema':'lighton4b-physical-actual-trace-closure/v1','status':'failed','source_sha':subprocess.check_output(['git','-C',ROOT,'rev-parse','HEAD'],text=True).strip(),'model_forward_executed':False,'build_executed':False,'quality_qualified':False,'upstream_internal_graph_captured':False,'comparison_scope':'actual native first4 prefix versus authenticated4-token primitives; native main uses16-token batching'}
 def locate(name,expected):
  candidates=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4*1024*1024 and sha(p)==expected]
  require(len(candidates)==1,'one authenticated retained report required: '+name);return candidates[0]
 def admit(p,row):require(p.is_file()and p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'artifact identity: '+p.name)
 def diff(a,b):
  x=np.asarray(a,dtype='<f4');y=np.asarray(b,dtype='<f4');require(x.shape==y.shape and np.isfinite(x).all()and np.isfinite(y).all(),'same finite shape');d=x.astype('float64')-y
  return {'bit_exact':x.tobytes()==y.tobytes(),'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(np.abs(d).max()),'reference_max_abs':float(np.abs(y).max())}
 try:
  require(report['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact closure source')
  basepath=locate('physical-decode.json',cfg['report_sha256']);base=json.loads(basepath.read_text());require(base['source_sha']==cfg['source']and base['candidate_sha256']==cfg['candidate_sha256']and base['status']=='measured','complete exact actual baseline');primpath=locate('prefix-primitives.json',cfg['primitive_report_sha256']);prim=json.loads(primpath.read_text());require(prim['source_sha']==cfg['primitive_source_sha']and prim['status']=='measured_operator_only'and prim['ggml_profile']['llamafile'] is True,'exact matched-profile primitive');report['baseline_report_sha256']=sha(basepath);report['primitive_report_sha256']=sha(primpath)
  # All raw per-case files stay in provider-mounted input. Stream their hashes.
  counts={};total=0
  for name,case in base['cases'].items():
   counts[name]=0
   for rel,row in case['artifacts'].items():
    p=basepath.parent/name/rel;admit(p,row);counts[name]+=1;total+=row['bytes']
  report['all_actual_case_artifacts_verified']={'counts':counts,'bytes_streamed':total,'all_exact':True}
  folder=basepath.parent/cfg['case']
  for rel,row in cfg['captures'].items():admit(folder/rel,row)
  for rel,row in cfg['primitive_files'].items():admit(primpath.parent/rel,row)
  load=lambda p:np.fromfile(p,dtype='<f4')
  reconstructed=load(primpath.parent/'reconstructed-prefix.f32').reshape(4,2560)
  actual=load(folder/'upstream_features/prefill.input.f32').reshape(154,2560)
  embeddings=load(folder/'upstream_features/text.embedded.f32').reshape(154,2560)
  require(actual[:4].tobytes()==reconstructed.tobytes()==embeddings[:4].tobytes(),'actual main4-prefix bytes match authenticated reconstruction');report['actual_prefix_input_exact']=True;report['actual_prefix_sha256']=hashlib.sha256(actual[:4].tobytes()).hexdigest()
  stages=folder/'upstream_features';norm=load(stages/'prefill.first_chunk.layer0.input_norm.f32').reshape(16,2560)[:4];scalar=load(primpath.parent/'reconstructed-native-scalar-norm.f32').reshape(4,2560);upnorm=load(primpath.parent/'upstream-norm.f32').reshape(4,2560)
  report['norm']={'actual_vs_reconstructed_scalar':diff(norm,scalar),'actual_vs_upstream_rms':diff(norm,upnorm)}
  residual=load(stages/'prefill.first_chunk.layer0.attention_residual.f32').reshape(16,2560);down=load(stages/'prefill.first_chunk.layer0.down.f32').reshape(16,2560);main=load(stages/'prefill.layer.0.f32').reshape(154,2560)[:16];require((residual+down).astype('<f4').tobytes()==main.tobytes(),'duplicate first16 stages reconstruct actual main output bytes');report['actual_first16_main_output_exact']=True
  heads=np.array([16*(h%2)+h//2 for h in range(32)]);values=(heads[:,None]*128+np.arange(128)[None,:]).reshape(-1);report['projections']={}
  for native,source,columns in [('qkv','attn_qkv',8192),('z','attn_gate',4096),('a','ssm_alpha',32),('b','ssm_beta',32)]:
   got=load(stages/('prefill.first_chunk.layer0.'+native+'.f32')).reshape(16,columns)[:4];order=np.concatenate([np.arange(4096),4096+values])if native=='qkv'else values if native=='z'else heads
   metrics={}
   for inputname in ('reconstructed-native-scalar-norm','upstream-norm'):
    for mode in ('direct','dequant'):
     want=load(primpath.parent/(source+'-'+inputname+'-upstream-'+mode+'.f32')).reshape(4,columns)[:,order]
     metrics[inputname+'-'+mode]=diff(got,want)
   metrics['same_input_scalar_norm_exact']=report['norm']['actual_vs_reconstructed_scalar']['bit_exact'];metrics['native_batch_tokens']=16;metrics['primitive_batch_tokens']=4;report['projections'][native]=metrics
  report['status']='compared_actual_trace';report['limits']=['Actual source internal graph intermediates are not captured.','16-token native projection and4-token source primitive can differ in batching kernels.','No learned model inference, new generation or quality promotion.']
 except Exception as error:report['error_type']=type(error).__name__;report['error']=str(error);raise
 finally:(out/'actual-trace-closure.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
