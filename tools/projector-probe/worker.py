#!/usr/bin/env python3
"""Projector-only canonical audit and observed-form source stages; no text model."""
import importlib.util,json,os,shutil,subprocess,sys,time,urllib.request,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent));import canonical,source_patch
sys.path.insert(0,str(ROOT/'tools/physical-f16-attention-probe'));from runtime_guard import Deadline,command
require,sha=canonical.require,canonical.sha


def locate(name,digest):
 rows=[p for p in Path('/kaggle/input').rglob(name)if p.is_file()and p.stat().st_size<4<<20 and sha(p)==digest]
 require(len(rows)==1,'unique authenticated receipt '+name);return rows[0]


def verify(p,row):require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'retained artifact '+p.name)


def pixels_to_patches(pixels,h,w):
 import numpy as np
 require(h%32==w%32==0 and pixels.size==h*w*3,'aligned source pixels')
 gh,gw=h//16,w//16
 patches=pixels.reshape(gh,16,gw,16,3).transpose(0,2,4,1,3)
 patches=np.repeat(patches[:,:,:,None,:,:],2,axis=3)
 return patches.reshape(gh//2,2,gw//2,2,3,2,16,16).transpose(0,2,1,3,4,5,6,7).reshape(gh*gw,1536)


def patches_to_pixels(patches,h,w):
 import numpy as np
 gh,gw=h//16,w//16
 require(h%32==w%32==0 and patches.size==gh*gw*1536,'native patch shape')
 shaped=patches.reshape(gh//2,gw//2,2,2,3,2,16,16).transpose(0,2,1,3,4,5,6,7).reshape(gh,gw,3,2,16,16)
 require(np.array_equal(shaped[:,:,:,0],shaped[:,:,:,1]),'identical still-image temporal frames')
 return shaped[:,:,:,0].transpose(0,3,1,4,2).reshape(h,w,3)


def preprocessing_audit(native_patches,source_pixels,resized,h,w,mean,std):
 import numpy as np
 native=patches_to_pixels(native_patches,h,w);source=source_pixels.reshape(h,w,3)
 recovered=np.rint(native.astype('f8')*127.5+127.5)
 require(recovered.min()>=0 and recovered.max()<=255,'native resized pixel inversion bound')
 native_rgb=recovered.astype('u1');source_rgb=resized.reshape(h,w,3)
 def native_formula(rgb):return (rgb.astype('<f4')-np.float32(127.5))/np.float32(127.5)
 def source_formula(rgb):return (rgb.astype('<f4')/np.float32(255)-np.asarray(mean,dtype='<f4'))/np.asarray(std,dtype='<f4')
 require(native_formula(native_rgb).tobytes()==native.tobytes(),'native normalization inversion exact')
 require(source_formula(source_rgb).tobytes()==source.tobytes(),'source resized RGB normalization exact')
 delta=native_rgb.astype('i2')-source_rgb.astype('i2')
 same_source_native_norm=native_formula(source_rgb)
 return dict(native_resize_recovered_exact=True,source_normalization_exact=True,resize_different_channels=int(np.count_nonzero(delta)),resize_total_channels=int(delta.size),resize_max_abs_u8=int(abs(delta).max()),resize_mean_abs_u8=float(np.mean(abs(delta))),normalization_only_same_source_rgb=metric(same_source_native_norm.reshape(-1),source.reshape(-1)),resize_only_same_native_normalization=metric(native.reshape(-1),same_source_native_norm.reshape(-1)),source_mean=mean,source_std=std),native_rgb


def metric(a,b):
 import numpy as np
 require(a.shape==b.shape and a.size>0 and np.isfinite(a).all()and np.isfinite(b).all(),'equal finite stage')
 d=a.astype('f8')-b.astype('f8');return dict(bit_exact=a.tobytes()==b.tobytes(),elements=int(a.size),rms=float(np.sqrt(np.mean(d*d))),max_abs=float(abs(d).max()),reference_max_abs=float(abs(b).max()))


def range_download(url,path,start,end,total):
 req=urllib.request.Request(url+f'?projector_range={start}-{end}',headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
 with urllib.request.urlopen(req,timeout=120)as response,path.open('xb')as f:
  require(response.status==206 and response.headers.get('Content-Range')==f'bytes {start}-{end}/{total}','exact bounded HTTP range');n=0
  while b:=response.read(1<<20):
   n+=len(b);require(n<=end-start+1,'range maximum');f.write(b)
 require(n==end-start+1,'complete range')


def main():
 import numpy as np
 require(np.__version__=='2.3.5','pinned parameter serializer')
 require(Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*'))and sys.byteorder=='little','Kaggle CPU/little-endian')
 cfgpath=ROOT/'docs/lighton4b-projector-canonical-v1.json';cfg=json.loads(cfgpath.read_text());canonical.admit(cfg)
 out=Path(os.environ['HEAVY_OUT']);work=Path(os.environ['HEAVY_SCRATCH'])/'projector';work.mkdir();started=time.monotonic();deadline=Deadline(3600);deadline.__enter__()
 r=dict(schema=cfg['schema'],status='failed',quality_qualified=False,text_model_instantiated=False,original_bf16_forward_executed=False,projector_forward_executed=False,commands={},stage_metrics={},canonical_weight_equivalence_proven=False)
 def save():(out/'projector-audit.json').write_text(json.dumps(r,indent=2)+'\n')
 def run(label,args,env=None,timeout=900):
  r['phase']=label;r['commands'][label]={};save();print('phase',label,flush=True)
  try:command(args,out/(label+'.log'),min(timeout,3600-(time.monotonic()-started)),r['commands'][label],env=env,cwd=ROOT)
  finally:save()
 try:
  run('source-pin',['git','rev-parse','HEAD']);r['source_sha']=(out/'source-pin.log').read_text().strip();require(r['source_sha']==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'exact source')
  spec=importlib.util.spec_from_file_location('base',ROOT/'tools/kaggle/lighton4b_upstream_physical_decode.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
  r['preflight']=dict(memory=base.memory(),scratch_free=shutil.disk_usage(work).free,output_free=shutil.disk_usage(out).free,load=os.getloadavg())
  require(r['preflight']['memory']['MemAvailable']>=4<<30 and r['preflight']['scratch_free']>=8<<30 and r['preflight']['output_free']>=2<<30,'projector-only resource guard')
  fp=locate('composed-vision.json',cfg['form_receipt_sha256']);form=json.loads(fp.read_text());require(form['status']=='measured_observed_projector_pair'and form['case']=='simple_form','complete observed form')
  ap=fp.parent/'physical-pack.json';require(ap.is_file()and sha(ap)==cfg['native_pack_receipt_sha256'],'exact pack proof co-located with authenticated form');audit=json.loads(ap.read_text());require(audit['status']=='converted'and audit['candidate_sha256']==form['candidate_sha256'],'whole-file authenticated protected-source audit')
  native_rows={x['name']:x for x in audit['tensors']if x['protected_vision']};require(len(native_rows)==297,'all protected projector tensors')
  for row in cfg['tensors']:
   actual=native_rows[row['native_name']];require(row['native_offset']==actual['source_offset']and row['native_sha256']==actual['source_packed_sha256']and row['native_bytes']==actual['bytes'],'canonical manifest provenance')
  pp=locate('physical-decode.json','712d4c86be0053a68dc82872b7323e3c8232ac6624d27463773ec71ddb7a61e2');physical=json.loads(pp.read_text());folder=pp.parent/'simple_form';art=physical['cases']['simple_form']['artifacts'];rgb=folder/'input.rgb';verify(rgb,art['input.rgb']);require(art['input.rgb']==form['actual_input_hashes']['input.rgb'],'same actual form RGB')
  # Authenticate all native comparison stages before transfers or any source forward.
  native_stages=['processor.patches.f32','vision.patch.f32','vision.position.f32','vision.embedded.f32','vision.merged.f32']+[f'vision.layer.{i}.f32'for i in range(24)]
  for name in native_stages:verify(fp.parent/'composed'/name,form['artifacts']['composed/'+name])
  source_feature=folder/'observer-on/source.vision.0.f32';verify(source_feature,art['observer-on/source.vision.0.f32']);r['inputs_authenticated']=True;save()
  upstream=work/'upstream'
  for label,args in [('up-init',['git','init',upstream]),('up-remote',['git','-C',upstream,'remote','add','origin','https://github.com/ggml-org/llama.cpp.git']),('up-fetch',['git','-C',upstream,'fetch','--depth','1','origin',cfg['upstream_commit']]),('up-checkout',['git','-C',upstream,'checkout','--detach','FETCH_HEAD'])]:run(label,args)
  require(subprocess.check_output(['git','-C',upstream,'rev-parse','HEAD'],text=True).strip()==cfg['upstream_commit'],'pinned source')
  r['reference_patch']=source_patch.patch(upstream);save()
  shutil.copy(ROOT/'tools/projector-probe/source.cpp',upstream/'crisp-projector-probe.cpp')
  with(upstream/'CMakeLists.txt').open('a')as f:f.write('\nadd_executable(crisp-projector-probe crisp-projector-probe.cpp)\ntarget_include_directories(crisp-projector-probe PRIVATE tools/mtmd)\ntarget_link_libraries(crisp-projector-probe PRIVATE mtmd)\n')
  build=work/'build';flags={'BUILD_SHARED_LIBS':'ON','GGML_CPU_REPACK':'ON','GGML_OPENMP_FETCH':'OFF','LLAMA_BUILD_TESTS':'OFF','LLAMA_BUILD_EXAMPLES':'OFF','LLAMA_BUILD_TOOLS':'ON','LLAMA_BUILD_SERVER':'OFF','MTMD_VIDEO':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_AVX512':'OFF','GGML_LLAMAFILE':'ON','GGML_OPENMP':'ON','GGML_BACKEND_DL':'OFF','GGML_BLAS':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}
  run('configure',['cmake','-S',upstream,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in flags.items()]])
  cache=(build/'CMakeCache.txt').read_text().splitlines()
  for k,v in flags.items():require(f'{k}:BOOL={v}'in cache,'actual reference profile '+k)
  shutil.copy(build/'CMakeCache.txt',out/'CMakeCache.txt');run('build',['cmake','--build',build,'--target','crisp-projector-probe','--parallel','2']);program=build/'bin/crisp-projector-probe'
  if not program.exists():program=build/'crisp-projector-probe'
  require(program.is_file(),'actual standalone source program');r['source_binary_sha256']=sha(program)
  env=dict(os.environ);env.pop('CRISP_PROJECTOR_CAPTURE',None);env['LD_LIBRARY_PATH']=':'.join(sorted({str(p.parent)for p in build.rglob('*.so*')if p.is_file()}))
  try:run('source-cli-noargs',[program],env,30)
  except subprocess.CalledProcessError as e:require(e.returncode==1,'standalone CLI expected argument rejection')
  else:raise ValueError('standalone CLI must reject missing arguments')
  require('projector RGB height width output-prefix required'in(out/'source-cli-noargs.log').read_text(),'standalone loader/CLI closure')
  projector=work/'mmproj.gguf';suffix=work/'native-vision.f32';r['phase']='projector-only-transfer';save()
  base.download(cfg['projector_url'],projector,cfg['projector_bytes'],cfg['projector_sha256']);start,end=cfg['native_vision_range'];range_download(cfg['native_base_url'],suffix,start,end,cfg['native_base_bytes'])
  require(projector.stat().st_size+suffix.stat().st_size<=cfg['maximum_download_bytes'],'total transfer quota');r['downloaded_weight_bytes']=projector.stat().st_size+suffix.stat().st_size
  r['phase']='canonical-all-parameters';save();r['canonical']=canonical.audit(cfg,suffix,projector);r['canonical_weight_equivalence_proven']=r['canonical']['all_canonical_f32_bit_exact'];save()
  for arm in ['off','on']:
   dest=out/arm;dest.mkdir();child=dict(env)
   if arm=='on':child['CRISP_PROJECTOR_CAPTURE']=str(dest)
   r['projector_forward_executed']=True;run(arm,[program,projector,rgb,317,452,dest/'source'],child,900)
   require(sha(dest/'source.merged.f32')==art['observer-on/source.vision.0.f32']['sha256'],'standalone source actual merged closure '+arm)
  require(sha(out/'off/source.pixels.f32')==sha(out/'on/source.pixels.f32'),'preprocessing observer neutral');r['source_observer_byte_neutral']=True;r['actual_source_merged_sha256']=sha(source_feature)
  require(sha(out/'off/source.input.rgb')==sha(out/'on/source.input.rgb')==art['input.rgb']['sha256'],'actual C++ RGB bytes read')
  counts=dict(line.split()for line in(out/'on/live-counts.tsv').read_text().splitlines());require(set(counts)==set(source_patch.STAGES)and all(v=='1'for v in counts.values()),'each live selected stage captured once');r['live_callback_counts']=counts
  meta=json.loads((out/'on/source.json').read_text());pixels=np.fromfile(out/'on/source.pixels.f32',dtype='<f4');patches=pixels_to_patches(pixels,meta['height'],meta['width']).reshape(-1);patches.tofile(out/'source.patches.f32')
  native=np.fromfile(fp.parent/'composed/processor.patches.f32',dtype='<f4');r['stage_metrics']['processor.patches']=metric(native,patches)
  inp=np.fromfile(out/'on/inp_raw.f32',dtype='<f4');require(inp.tobytes()==pixels.reshape(meta['height'],meta['width'],3).transpose(2,0,1).copy().tobytes(),'actual source normalized HWC to graph CHW exact')
  resized=np.fromfile(out/'on/resized.rgb',dtype='u1');r['preprocessing'],native_rgb=preprocessing_audit(native,pixels,resized,meta['height'],meta['width'],meta['mean'],meta['std']);native_rgb.tofile(out/'native.recovered-resized.rgb')
  bias=np.fromfile(out/'on/patch_bias.f32',dtype='<f4');position=np.fromfile(out/'on/learned_pos.f32',dtype='<f4');embedded=np.fromfile(out/'on/inp_pos_emb.f32',dtype='<f4');require(np.any(position)and (bias+position).tobytes()==embedded.tobytes(),'live patch plus position arithmetic closure')
  ff_in=np.fromfile(out/'on/ffn_inp-0.f32',dtype='<f4');ff_out=np.fromfile(out/'on/ffn_out-0.f32',dtype='<f4');layer=np.fromfile(out/'on/layer_out-0.f32',dtype='<f4');require((ff_in+ff_out).tobytes()==layer.tobytes(),'live first-layer residual arithmetic closure');r['live_arithmetic_closures_exact']=True
  pairs=[('patch_bias','vision.patch.f32'),('learned_pos','vision.position.f32'),('inp_pos_emb','vision.embedded.f32')]+[(f'layer_out-{i}',f'vision.layer.{i}.f32')for i in range(24)]
  for source,native in pairs:
   x=np.fromfile(fp.parent/'composed'/native,dtype='<f4');y=np.fromfile(out/'on'/(source+'.f32'),dtype='<f4');require(y.size>0 and np.any(y),'materialized nonzero actual source stage');r['stage_metrics'][source]=metric(x,y)
  r['stage_metrics']['merged']=metric(np.fromfile(fp.parent/'composed/vision.merged.f32',dtype='<f4'),np.fromfile(source_feature,dtype='<f4'));r['earliest_comparable_difference']=next((name for name,row in r['stage_metrics'].items()if not row['bit_exact']),None);r['status']='measured_projector_only_audit'
 except BaseException as e:r['error_type']=type(e).__name__;r['error']=str(e);raise
 finally:
  deadline.close();r['elapsed_seconds']=time.monotonic()-started;r['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in out.rglob('*')if p.is_file()and p.name!='projector-audit.json'};save()
if __name__=='__main__':main()
