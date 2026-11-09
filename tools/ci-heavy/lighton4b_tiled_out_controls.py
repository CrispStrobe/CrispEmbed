#!/usr/bin/env python3
"""Synthetic operator-only F16/Q4 chunk, coordinate, weighting and source guards."""
import argparse,hashlib,importlib.util,json,os,struct,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tools'))
import lighton4b_hf_tiled_out_reproduce as reproducer

def require(value,message):
 if not value:raise ValueError(message)

def controls(helper,out):
 import numpy as np
 require(np.__version__=='2.3.5','pinned fixture serializer required');helper=Path(helper).absolute();out=Path(out);require(not out.exists(),'fresh fixture output required');out.mkdir();r={'schema':'lighton4b-tiled-out-synthetic-controls/v1','status':'failed','model_inference_executed':False,'real_checkpoint_payload_bytes':0,'full_f32_model_instantiated':False,'checks':[]};started=time.monotonic()
 try:
  n=17;columns=4096;row=np.arange(n,dtype=np.int64)[:,None];col=np.arange(columns,dtype=np.int64)[None,:];values=((((row*17+col*37)%257)-128)/256.0*(1+(col//128)%7)).astype('<f2');raw=values.tobytes();prefix=b'SOURCE-F16-PREFIX-CANARY'.ljust(64,b'!');source=out/'synthetic.f16';source.write_bytes(prefix+raw+b'SOURCE-SUFFIX');source_sha=hashlib.sha256(raw).hexdigest();means=(((np.arange(columns)*7)%23+1)/16.0+(np.arange(columns)//128)/8.0).astype('<f4');means_path=out/'means.f32';means_path.write_bytes(means.tobytes());perm=np.array([(16*(h%2)+h//2)*128+c for h in range(32)for c in range(128)],dtype=np.int64)
  def run(label,src=source,stats=means_path,digest=source_sha,maximum=65536,offset=64,rows=n,expected=0):
   folder=out/label;folder.mkdir();tiled=folder/'tiled.q4';native=folder/'hf.q4';report=folder/'rows.json';args=[str(helper),str(src),str(offset),str(rows),str(stats),digest,str(tiled),str(native),str(report),str(maximum)]
   with(folder/'helper.log').open('w')as log:child=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=60)
   require(child.returncode==expected,'fixture helper return code differs: '+label)
   if expected:
    require(not tiled.exists()and not native.exists()and not report.exists()and not list(folder.glob('*.partial')),'negative published a partial/final output: '+label);r['checks'].append(label);return None
   result=json.loads(report.read_text());require(result['source_f16_sha256']==digest and result['coordinate_map_sha256']==reproducer.MAP_SHA and result['importance_supplied'],'source/map/importance fixture differs');result['independent']=reproducer.independent_decode(src,offset,native,tiled,stats.read_bytes(),result);r['checks'].append(label);return folder,result
  full,result=run('chunk16');small,_=run('chunk3',maximum=12288);single,_=run('chunk1',maximum=4096)
  for name in ('tiled.q4','hf.q4'):require((full/name).read_bytes()==(small/name).read_bytes()==(single/name).read_bytes(),'whole-row chunk invariance differs')
  r['checks'].append('both-payloads-chunk-invariant');require(result['canonical_different_bits']>0 and result['canonical_rms_difference']>0,'fixture must detect nontrivial canonical quantization difference');r['checks'].append('nontrivial-canonical-weight-control')
  # Independent Python matrix construction; the reference uses the helper's
  # unpermuted native-control branch, not its tiled quantization branch.
  physical=np.empty_like(values);physical[:,perm]=values;ref_source=out/'already-physical.f16';ref_raw=physical.tobytes();ref_source.write_bytes(ref_raw);forward_means=np.empty_like(means);forward_means[perm]=means;ref_means=out/'reference-means.f32';ref_means.write_bytes(forward_means.tobytes());reference,_=run('python-coordinate-reference',src=ref_source,stats=ref_means,digest=hashlib.sha256(ref_raw).hexdigest(),offset=0)
  require((full/'tiled.q4').read_bytes()==(reference/'hf.q4').read_bytes(),'independent physical-coordinate reference differs');r['checks'].append('python-constructed-physical-reference-exact')
  wrongly_inverted=out/'wrongly-inverted.f32';wrongly_inverted.write_bytes(means[perm].tobytes());wrong,_=run('inverse-statistics-negative',stats=wrongly_inverted);require((wrong/'tiled.q4').read_bytes()!=(full/'tiled.q4').read_bytes(),'incorrect inverse statistics did not change fixture');r['checks'].append('wrong-statistics-not-silently-ignored')
  zero=out/'zero.f32';zero.write_bytes(bytes(columns*4));run('empty-energy-rejected',stats=zero,expected=2)
  narrow=out/'narrow.f32';narrow.write_bytes(bytes(4));run('wrong-width-rejected',stats=narrow,expected=2)
  run('wrong-source-span-rejected',digest='0'*64,expected=2);run('split-row-chunk-rejected',maximum=4095,expected=2);run('truncated-source-rejected',offset=source.stat().st_size-2,expected=2)
  nonfinite=out/'nonfinite.f16';bad=bytearray(raw);bad[:2]=struct.pack('<H',0x7c00);nonfinite.write_bytes(bad);run('nonfinite-source-rejected',src=nonfinite,digest=hashlib.sha256(bad).hexdigest(),offset=0,expected=2)
  invalid_means=out/'nonfinite-means.f32';invalid=means.copy();invalid[0]=np.nan;invalid_means.write_bytes(invalid.tobytes());run('nonfinite-importance-rejected',stats=invalid_means,expected=2)
  invalid[0]=-1.;invalid_means=out/'negative-means.f32';invalid_means.write_bytes(invalid.tobytes());run('negative-importance-rejected',stats=invalid_means,expected=2)
  protected=out/'overwrite';protected.mkdir();target=protected/'tiled.q4';target.write_bytes(b'never-overwrite');args=[str(helper),str(source),'64',str(n),str(means_path),source_sha,str(target),str(protected/'hf.q4'),str(protected/'rows.json'),'65536']
  with(protected/'helper.log').open('w')as log:child=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,timeout=60)
  require(child.returncode==2 and target.read_bytes()==b'never-overwrite'and not(protected/'hf.q4').exists(),'existing output was not protected');r['checks'].append('existing-output-protected')
  require(source.read_bytes()==prefix+raw+b'SOURCE-SUFFIX','source canaries or bytes changed');r['checks'].append('source-file-immutable');r['status']='passed';r['helper_sha256']=hashlib.sha256(helper.read_bytes()).hexdigest();r['fixture_elements']=values.size;r['max_chunk_elements']=65536;return r
 finally:r['seconds']=time.monotonic()-started;(out/'controls.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--helper',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();r=controls(a.helper,a.output);print(json.dumps({'status':r['status'],'checks':len(r['checks']),'real_checkpoint_payload_bytes':0}),flush=True)
