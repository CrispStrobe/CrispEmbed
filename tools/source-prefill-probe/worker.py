#!/usr/bin/env python3
"""Actual observed source inputs and scheduler controls; no model forward."""
import csv,hashlib,importlib.util,json,os,platform,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('rangebase',ROOT/'tools/repack-probe/worker.py');require=base.require;sha=base.sha
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'));from runtime_guard import Deadline,command
CFG_SHA='65e70e1c18b42d461f4ef49c5108fb0122f67786e192c3eec8b9be13494599c3'
def admit(cfg):
 require(hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':')).encode()).hexdigest()==CFG_SHA,'immutable source-prefill proof protocol')
 require(cfg['input_norm']['type']==0 and cfg['input_norm']['source_shape']==[2560]and cfg['input_norm']['sha256']==cfg['input_norm']['source_packed_sha256'],'direct original gamma storage')
 require(cfg['worker_seconds']==1800 and not cfg['model_forward_allowed']and not cfg['heldout_allowed'],'operator-only bounded scope')
def source_graphs(trace):
 records=trace['analysis']['excluded_nonprompt_graphs'];require(len(records)==1 and records[0]['graph']==0 and records[0]['tokens']==[0,0]and records[0]['positions_4axes']==[[0,1]]*3+[[0,0]],'exact admitted startup probe')
 excluded={g['graph']for g in records}
 return [g for g in trace['analysis']['tensor_metadata']if g['sample_count']==0 and g['index']not in excluded]
def main():
 import numpy as np
 cfg=json.loads((Path(__file__).parent/'protocol.json').read_text());admit(cfg);require(np.__version__=='2.3.5'and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'pinned CPU profile');require(sys.byteorder=='little'and platform.system()=='Linux'and platform.machine().lower()in['x86_64','amd64'],'portable source CPU');out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'source-prefill-proof';work.mkdir();started=time.monotonic();deadline=Deadline(1800);deadline.__enter__();r=dict(schema=cfg['schema'],status='failed',commands={},cases={},model_forward_executed=False,heldout_data_accessed=False,quality_qualified=False)
 def save():(out/'source-prefill-proof.json').write_text(json.dumps(r,indent=2)+'\n')
 def canceled(sig,frame):raise SystemExit(128+sig)
 prior={sig:signal.signal(sig,canceled)for sig in[signal.SIGTERM,signal.SIGINT]}
 def run(label,args,limit=600,env=None):
  r['phase']=label;r['commands'][label]={};save();print('PHASE '+label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,1800-(time.monotonic()-started)),r['commands'][label],cwd=ROOT,env=env)
  finally:save()
 def locate(name,digest):
  paths=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest];require(len(paths)==1,'one exact '+name);return paths[0],json.loads(paths[0].read_text())
 def verify(path,row):require(path.stat().st_size==row['bytes']and sha(path)==row['sha256'],'complete immutable '+path.name)
 try:
  run('source',['git','rev-parse','HEAD']);r['source_sha']=(out/'source.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact controller');physical=load('capacity',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');r['capacity']=dict(memory=physical.memory(),scratch=shutil.disk_usage(work).free);require(r['capacity']['memory']['MemAvailable']>=4<<30 and r['capacity']['scratch']>=6<<30,'bounded operator resources')
  run('cpu',['lscpu']);run('compiler',['c++','--version'])
  flags=next(line.split(':',1)[1].split()for line in Path('/proc/cpuinfo').read_text().splitlines()if line.startswith('flags'));require({'avx2','fma','f16c'}<=set(flags),'actual supported CPU')
  tp,trace=locate('form-text-trace.json',cfg['trace_sha256']);npth,native=locate('q6-tiled-full-ab.json',cfg['native_sha256']);require(trace['status']=='measured_neutral_form_text_trace'and trace['all170_scores_observer_neutral']and trace['analysis']['same_model_input_bits']and trace['analysis']['same_positions_first3_axes'],'neutral actual source');require(native['status']=='measured_q6_tiled_full_ab','retained corrected native')
  for name,digest in cfg['production_file_sha256'].items():require(sha(ROOT/name)==digest,'immutable Q6 math '+name)
  graphs=source_graphs(trace);require([g['tokens']for g in graphs]==trace['analysis']['actual_graph_token_counts']and sum(g['tokens']for g in graphs)==cfg['prompt_tokens'],'actual source prompt geometry');norms=[];outputs=[]
  for g in graphs:
   arrays=[]
   for role,width in [('attn_norm-0',2560),('linear_attn_qkv_mixed-0',8192)]:
    key=f'on/graph-{g["index"]}.{role}.bin';p=tp.parent/key;verify(p,trace['artifacts'][key]);x=np.fromfile(p,dtype='<f4').reshape(g['tokens'],width);require(np.isfinite(x).all(),'finite source operator inputs');arrays.append(x)
   norms.append(arrays[0]);outputs.append(arrays[1])
  r['logical_source_batches']=[g['tokens']for g in graphs]
  x=np.concatenate(norms);permutation=base.row_map('qkv',8192);expected=np.concatenate(outputs)[:,permutation];tokens=len(x);prefix='simple_form/on/';inputs={}
  for name in ['modalities.ids','prefill.positions.f32','prefill.input.f32','prefill.first_chunk.layer0.input_norm.f32','prefill.first_chunk.layer0.qkv.f32']:
   p=npth.parent/prefix/name;verify(p,native['artifacts'][prefix+name]);inputs[name]=p
  require(x[:16].tobytes()==inputs['prefill.first_chunk.layer0.input_norm.f32'].read_bytes(),'same actual native first16 normalized input');r['actual_native_input_first16_exact']=True;mods=list(map(int,inputs['modalities.ids'].read_text().split()));require(len(mods)==tokens and sum(mods)==cfg['image_tokens']and set(mods)=={0,1},'actual modality census');positions=np.fromfile(inputs['prefill.positions.f32'],dtype='<f4');require(positions.size==3*tokens and np.isfinite(positions).all(),'actual section-major positions');inp=work/'norm.f32';x.astype('<f4').tofile(inp)
  run('ggml',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==cfg['native_ggml_source'],'unchanged GGML');build=work/'build';flags=cfg['build_flags'];run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in flags.items()]]);cache=(build/'CMakeCache.txt').read_text().splitlines();require(all(f'{k}:BOOL={v}'in cache for k,v in flags.items()),'actual admitted build');shutil.copy2(build/'CMakeCache.txt',out/'CMakeCache.txt');targets=['lighton4b-source-prefill-probe','lighton4b-native-document','test-lighton4b-source-prefill-chunks','test-lighton4b-physical-q6-tiled'];run('build',['cmake','--build',build,'--target',*targets,'--parallel','2'],1200)
  for target,marker in [('test-lighton4b-source-prefill-chunks','SOURCE_PREFILL_CHUNKS_CONTROLS PASS'),('test-lighton4b-physical-q6-tiled','PHYSICAL_Q6_TILED_CONTROLS PASS checks=38')]:run(target,[build/target]);require(marker in(out/(target+'.log')).read_text(),'supported model-free controls')
  op_env=dict(os.environ);op_env.update({k:'0'for k in op_env if k.startswith('CRISPEMBED_')});op_env.update(native['cases']['simple_form']['arms']['on']['gates']);op_env['CRISPEMBED_LIGHTON4B_PHYSICAL_SOURCE_PREFILL']='0';op_env['LD_LIBRARY_PATH']=':'.join(sorted({str(p.parent)for p in build.rglob('libggml*.so*')if p.is_file()}));r['operator_gates']={k:v for k,v in op_env.items()if k.startswith('CRISPEMBED_')}
  program=build/'lighton4b-source-prefill-probe';zero=work/'zero.bin';zero.write_bytes(bytes(cfg['matrix']['bytes']));zx=work/'zero.f32';np.zeros((20,2560),dtype='<f4').tofile(zx);zm=work/'zero.mods';zm.write_text('1\n'*20);zp=work/'zero.positions';np.zeros(60,dtype='<f4').tofile(zp)
  zg=work/'zero.gamma';np.ones(2560,dtype='<f4').tofile(zg)
  for mode in ['source-q6','source-plan']:
   prefixout=work/'zero';run('synthetic-'+mode,[program,zero,zx,14,2560,8192,20,mode,prefixout,4,16,zm,zp,zg,zx],30,op_env);require(not np.any(np.fromfile(str(prefixout)+'.f32',dtype='<f4')),'synthetic balanced-tail zero Q6')
  gamma=work/'gamma.f32';base.download_range(cfg,cfg['input_norm'],gamma);r['source_gamma_bytes_read']=cfg['input_norm']['bytes'];r['source_gamma_sha256']=sha(gamma)
  weight=work/'qkv.bin';base.download_range(cfg,cfg['matrix'],weight);r['weight_sha256']=sha(weight);r['checkpoint_quantized_tensor_bytes_read']=cfg['matrix']['bytes']
  for threads in[1,4]:
   for mode in ['source-q6','source-plan']:
    tag=mode+'-t'+str(threads);prefixout=out/tag;run(tag,[program,weight,inp,14,2560,8192,tokens,mode,prefixout,threads,16,inputs['modalities.ids'],inputs['prefill.positions.f32'],gamma,inputs['prefill.input.f32']],120,op_env);require(Path(str(prefixout)+'.positions.f32').read_bytes()==inputs['prefill.positions.f32'].read_bytes(),'all original position bits survive actual schedule');require(Path(str(prefixout)+'.normalized.f32').read_bytes()==inp.read_bytes(),'all source normalized inputs exact');value=np.fromfile(str(prefixout)+'.f32',dtype='<f4').reshape(tokens,8192)[:,permutation];r['cases'][tag]=dict(vs_actual_source=base.metric(value,expected),first16_vs_actual_native=base.metric(value[:16],np.fromfile(inputs['prefill.first_chunk.layer0.qkv.f32'],dtype='<f4').reshape(16,8192)),hf_output_sha256=hashlib.sha256(value.tobytes()).hexdigest());plan=list(csv.DictReader(Path(str(prefixout)+'.plan.tsv').open(),delimiter='\t'));r['cases'][tag]['plan']=plan
    if mode=='source-plan':
     logical=[];previous=-1
     for row in plan:
      if int(row['logical_offset'])!=previous:logical.append(int(row['logical_tokens']));previous=int(row['logical_offset'])
     require(logical==[g['tokens']for g in graphs],'generic logical plan equals actual source graphs');require(r['cases'][tag]['vs_actual_source']['bit_exact'],'all actual source QKV outputs byte exact')
    else:require(r['cases'][tag]['first16_vs_actual_native']['bit_exact'],'actual existing mixed16 output closes')
    save()
  for mode in ['source-q6','source-plan']:require(r['cases'][mode+'-t1']['hf_output_sha256']==r['cases'][mode+'-t4']['hf_output_sha256'],'thread repeat exact')
  image_rows=np.flatnonzero(np.asarray(mods)==1)[:cfg['counterfactual_image_tokens']];cx=work/'image20.f32';x[image_rows].astype('<f4').tofile(cx);cm=work/'image20.mods';cm.write_text('1\n'*len(image_rows));cp=work/'image20.positions';positions.reshape(3,tokens)[:,image_rows].astype('<f4').tofile(cp);ce=work/'image20.embedding';np.fromfile(inputs['prefill.input.f32'],dtype='<f4').reshape(tokens,2560)[image_rows].astype('<f4').tofile(ce)
  for mode in ['source-q6','source-plan']:
   prefixout=out/('counterfactual-image20-'+mode);run('counterfactual-'+mode,[program,weight,cx,14,2560,8192,len(image_rows),mode,prefixout,4,16,cm,cp,gamma,ce],120,op_env);value=np.fromfile(str(prefixout)+'.f32',dtype='<f4').reshape(len(image_rows),8192)[:,permutation];r.setdefault('counterfactual_image20',{})[mode]=base.metric(value,expected[image_rows])
  require(r['counterfactual_image20']['source-plan']['bit_exact'],'nontrivial balanced image20 prefix closure')
  r['all_source154_projection_exact']=all(r['cases'][mode]['vs_actual_source']['bit_exact']for mode in ['source-plan-t1','source-plan-t4'])
  bundle=out/'runtime';(bundle/'bin').mkdir(parents=True);(bundle/'lib').mkdir();shutil.copy2(build/'lighton4b-native-document',bundle/'bin/lighton4b-native-document')
  for p in build.rglob('libggml*.so*'):
   if p.is_file():shutil.copy2(p,bundle/'lib'/p.name)
  r['runtime_bundle']={str(p.relative_to(bundle)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in bundle.rglob('*')if p.is_file()};r['status']='measured_exact_source_prefill_projection';r['limits']=['Observed form source-fed operator only; no decoder/model forward.','Actual native normalization is directly captured first16; full154 production normalization is replayed from actual native embeddings and original source gamma.','No heldout data or recipe/tensor-specific routing. Full observed decoded A/B remains required.']
 except BaseException as e:r.update(error_type=type(e).__name__,error=str(e));raise
 finally:
  deadline.close()
  for sig,handler in prior.items():signal.signal(sig,handler)
  r['seconds']=time.monotonic()-started;owned={label+'.log'for label in r['commands']}|{'CMakeCache.txt'};r['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=sha(p))for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned or p.name.startswith(('source-q6-t','source-plan-t','counterfactual-image20-')))};save()
if __name__=='__main__':main()
