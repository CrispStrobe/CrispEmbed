#!/usr/bin/env python3
"""Prepared projector-only actual-input comparison; blocked until controls frozen."""
import hashlib,importlib.util,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'))
from runtime_guard import Deadline,command
sys.path.insert(0,str(ROOT/'tools/projector-probe'))
import canonical
spec=importlib.util.spec_from_file_location('projector_reference',ROOT/'tools/projector-probe/worker.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
spec=importlib.util.spec_from_file_location('vision_controls',Path(__file__).with_name('controls.py'));controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)
CONTROL_SOURCE='0eca8d13ff55169ad47efcb446fa32786bbf9d3e'
CONTROL_RECEIPT='31e6d6e54ec5a26c464e0e0f7a151e2408f04c4ec517b59648997a76a3f5ee09'
FORM='1458d07e6098b4b1b8fa078c18fe0b636f832a0d1219f789b035e1a033ef4ba0'
LEGACY_GATES=tuple('CRISPEMBED_LIGHTON4B_VISION_'+s for s in ('BF16','BF16_NORM','BF16_FMA_NORM','BF16_LINEAR','BF16_CONV','BF16_ATTENTION','BF16_SOFTMAX_AVX2','BF16_GELU_AVX2','BF16_MERGER_ERF_AVX2','BF16_ROTARY_TABLE'))
def require(ok,why):
 if not ok:raise ValueError(why)
def admit_control(passed):
 require(passed['status']=='passed_model_free_vision_controls'and passed['source_sha']==CONTROL_SOURCE and passed['model_acquisition']is False and passed['checkpoint_vision_forward_executed']is False and passed['actual_form_preprocessing_exact']is True and passed['synthetic_vision_component_executed']is True,'completed supported synthetic/resize controls')
def locate(name,digest):
 matches=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and canonical.sha(p)==digest]
 require(len(matches)==1,'one immutable '+name);return matches[0]
def verify(path,meta):require(path.stat().st_size==meta['bytes']and canonical.sha(path)==meta['sha256'],'authenticated '+path.name)
def metric(a,b):
 import numpy as np
 x=np.fromfile(a,dtype='<f4');y=np.fromfile(b,dtype='<f4');require(x.size and x.shape==y.shape and np.isfinite(x).all()and np.isfinite(y).all(),'same finite stage')
 d=x.astype('f8')-y.astype('f8');return dict(bit_exact=x.tobytes()==y.tobytes(),elements=int(x.size),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),reference_max_abs=float(abs(y).max()),candidate_sha256=canonical.sha(a),reference_sha256=canonical.sha(b))
def main():
 require(len(CONTROL_RECEIPT)==64 and len(CONTROL_SOURCE)==40,'actual completed controls source/receipt pinned')
 import numpy as np
 require(np.__version__=='2.3.5'and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*'))and sys.byteorder=='little','pinned CPU/endian/numpy')
 out=Path(os.environ['HEAVY_OUT']);out.mkdir(parents=True,exist_ok=True);work=Path(os.environ['HEAVY_SCRATCH'])/'vision-component';work.mkdir();started=time.monotonic()
 report=dict(schema='lighton4b-physical-vision-component/v1',status='failed',commands={},legacy_anchor_exact=False,source_metrics={},text_model_instantiated=False,original_bf16_forward_executed=False,decoded_quality_qualified=False,downloaded_weight_bytes=0)
 def save():(out/'vision-component.json').write_text(json.dumps(report,indent=2)+'\n')
 def run(label,args,env=None,limit=900):
  report['phase']=label;report['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(limit,2400-(time.monotonic()-started)),report['commands'][label],cwd=ROOT,env=env)
  finally:save()
 deadline=Deadline(2400);deadline.__enter__()
 def canceled(sig,_frame):raise SystemExit('bounded component canceled by signal '+str(sig))
 prior={sig:signal.signal(sig,canceled)for sig in(signal.SIGTERM,signal.SIGINT)}
 try:
  run('source',['git','rev-parse','HEAD']);report['source_sha']=(out/'source.log').read_text().strip();require(report['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source')
  cp=locate('vision-controls.json',CONTROL_RECEIPT);passed=json.loads(cp.read_text());admit_control(passed)
  for name,meta in passed['artifacts'].items():verify(cp.parent/name,meta)
  controls.profile.validate_profile((cp.parent/'native-CMakeCache.txt').read_text().splitlines())
  require('PHYSICAL_F16_VISION_SYNTHETIC PASS 'in(cp.parent/'run-test-lighton4b-physical-f16-vision.log').read_text(),'actual supported synthetic control marker')
  run('fetch-tested',['git','fetch','--depth','1','origin',CONTROL_SOURCE]);run('runtime-byte-guard',['git','diff','--exit-code',CONTROL_SOURCE,'HEAD','--','src','ggml','CMakeLists.txt','tests/test_lighton4b_physical_f16_vision.cpp','tests/test_lighton4b_pillow_resize.cpp','tools/vision-profile-probe/native.cpp'])
  pp=locate('projector-audit.json',controls.PROJECTOR);source=json.loads(pp.read_text());require(source['source_observer_byte_neutral']and source['live_arithmetic_closures_exact'],'valid actual source observer')
  fp=locate('composed-vision.json',FORM);form=json.loads(fp.read_text());require(form['status']=='measured_observed_projector_pair'and form['case']=='simple_form','frozen observed form only')
  cfg=json.loads((ROOT/'docs/lighton4b-projector-canonical-v1.json').read_text());canonical.admit(cfg)
  stages={'vision.patch.f32':'on/patch_bias.f32','vision.position.f32':'on/learned_pos.f32','vision.embedded.f32':'on/inp_pos_emb.f32',**{f'vision.layer.{i}.f32':f'on/layer_out-{i}.f32'for i in range(24)},'vision.merged.f32':'on/source.merged.f32'}
  for name,target in stages.items():verify(pp.parent/target,source['artifacts'][target]);verify(fp.parent/'composed'/name,form['artifacts']['composed/'+name])
  for name in['on/source.input.rgb','on/resized.rgb','source.patches.f32']:verify(pp.parent/name,source['artifacts'][name])
  base=importlib.util.spec_from_file_location('capacity',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');capacity=importlib.util.module_from_spec(base);base.loader.exec_module(capacity)
  report['capacity']=dict(memory=capacity.memory(),scratch_free=shutil.disk_usage(work).free,load=os.getloadavg());require(report['capacity']['memory']['MemAvailable']>=4<<30 and report['capacity']['scratch_free']>=6<<30,'bounded vision-only capacity')
  run('ggml-init',['git','submodule','update','--init','ggml']);require(subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()==controls.NATIVE,'exact native ggml')
  build=work/'build';run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in controls.profile.BUILD_FLAGS.items()],'-DGGML_CPU_REPACK=ON']);controls.profile.validate_profile((build/'CMakeCache.txt').read_text().splitlines());shutil.copyfile(build/'CMakeCache.txt',out/'CMakeCache.txt')
  run('build',['cmake','--build',build,'--target','test-lighton4b-physical-f16-vision','lighton4b-vision-profile-probe','--parallel','2']);run('synthetic',[build/'test-lighton4b-physical-f16-vision']);require('PHYSICAL_F16_VISION_SYNTHETIC PASS 'in(out/'synthetic.log').read_text(),'valid graph closes before parameter transfers')
  start,end=cfg['native_vision_range'];suffix=work/'vision-only.f32';reference.range_download(cfg['native_base_url'],suffix,start,end,cfg['native_base_bytes']);report['downloaded_weight_bytes']=suffix.stat().st_size;require(report['downloaded_weight_bytes']==1334056960,'only complete vision suffix acquired')
  schema=work/'vision.schema';cursor=0
  with schema.open('w')as table:
   for row in sorted(cfg['tensors'],key=lambda x:x['native_offset']):
    relative=row['native_offset']-start;require(relative==cursor,'no omitted protected tensor bytes');h=hashlib.sha256()
    with suffix.open('rb')as f:
     f.seek(relative);remaining=row['native_bytes']
     while remaining:
      chunk=f.read(min(1<<20,remaining));require(bool(chunk),'complete tensor span');h.update(chunk);remaining-=len(chunk)
    require(h.hexdigest()==row['native_sha256'],'actual complete tensor SHA '+row['native_name']);shape=row['native_shape']+[1]*(4-len(row['native_shape']));table.write(' '.join(map(str,[row['native_name'],*shape,relative,row['native_bytes']]))+'\n');cursor+=row['native_bytes']
  require(cursor==suffix.stat().st_size,'all297 protected vision tensors authenticated');report['vision_tensor_count']=297;report['full_native_container_downloaded']=False;report['whole_native_container_sha_recomputed']=False;report['vision_schema_sha256']=canonical.sha(schema);save()
  env=dict(os.environ);env.update({k:'0'for k in env if k.startswith('CRISPEMBED_')});env.update({k:'1'for k in LEGACY_GATES});env['CRISPEMBED_LIGHTON4B_VISION_OWNER']='1';report['legacy_gate_tuple']={k:v for k,v in env.items()if k.startswith('CRISPEMBED_')}
  for mode in['legacy','f16']:
   dest=out/mode;dest.mkdir();run(mode,[build/'lighton4b-vision-profile-probe',suffix,schema,pp.parent/'on/source.input.rgb',317,452,mode,dest],env=env)
   if mode=='legacy':
    for name in stages:require(canonical.sha(dest/name)==form['artifacts']['composed/'+name]['sha256'],'retained legacy stage exact '+name)
    require(canonical.sha(dest/'processor.patches.f32')==form['artifacts']['composed/processor.patches.f32']['sha256'],'legacy processor exact');report['legacy_anchor_exact']=True;save()
   else:
    require((dest/'resized.rgb').read_bytes()==(pp.parent/'on/resized.rgb').read_bytes(),'actual source resize byte closure');require((dest/'processor.patches.f32').read_bytes()==(pp.parent/'source.patches.f32').read_bytes(),'actual source packed pixels byte closure')
    for name,target in stages.items():report['source_metrics'][name]=metric(dest/name,pp.parent/target);save()
  report['all_source_stages_byte_exact']=all(m['bit_exact']for m in report['source_metrics'].values());report['first_differing_stage']=next((n for n,m in report['source_metrics'].items()if not m['bit_exact']),None);report['status']='measured_same_input_projector_component';report['scope']='Observed form vision-only; canonical source precision authenticated; full checkpoint/container identity and decoded quality not claimed.'
 except BaseException as e:report['error_type']=type(e).__name__;report['error']=str(e);raise
 finally:
  deadline.close()
  for sig,handler in prior.items():signal.signal(sig,handler)
  report['elapsed_seconds']=time.monotonic()-started;owned={name+'.log'for name in report['commands']}|{'CMakeCache.txt'};report['artifacts']={str(p.relative_to(out)):dict(bytes=p.stat().st_size,sha256=canonical.sha(p))for p in out.rglob('*')if p.is_file()and(p.parent!=out or p.name in owned)};save()
if __name__=='__main__':main()
