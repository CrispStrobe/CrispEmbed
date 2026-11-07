#!/usr/bin/env python3
"""Closed original-F16 row reproduction, before any HF-tiled-out model conversion."""
import argparse,hashlib,importlib.metadata,json,math,os,re,struct,subprocess,time
from pathlib import Path
import lighton4b_imatrix_adapter as adapter
from lighton4b_imatrix_q4km_recipe import plan,SOURCE_F16_SHA
from lighton4b_q6_recipe import validate_source_inventory
from gguf_file_type_canonicalize import sha,span
ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/'docs/lighton4b-hf-tiled-out-reproducer-protocol-v1.json'
MAP_SHA='d85c5067c02735fb9f5701d64a1f04ff43d45d20d57612f0a015a57b7ce72b5c'
def require(value,message):
 if not value:raise ValueError(message)
def physical_mean(sums,count):
 require(len(sums)==4096 and len(count)==1 and count[0]==66048,'exact dense4096-column count profile required')
 require(all(math.isfinite(x)and x>=0 for x in sums)and any(x>0 for x in sums),'finite nonnegative nonempty physical energy required')
 means=[adapter.f32(float(x)/66048)for x in sums];require(all(math.isfinite(x)for x in means),'physical means overflow');return struct.pack('<4096f',*means)
def protocol():
 p=json.loads(PROTOCOL.read_text());names=[r['name']for r in p['selected']];expected={r['target']for r in adapter.inventory().values()if r['inverse_v_head_columns']};require(len(names)==24 and len(set(names))==24 and set(names)==expected,'exact24 original recurrent-output names required');require(p['coordinate_map_uint32_le_sha256']==MAP_SHA,'coordinate map differs');require(p['source_original_f16']['sha256']==SOURCE_F16_SHA and p['source_original_f16']['bytes']==9757493600 and p['statistics']['sha256']==adapter.SOURCE_SHA,'closed original/statistics identities differ')
 for row in p['selected']:
  require(row['source_type']==1 and row['source_stored_shape']==[4096,2560]and row['rows']==2560 and row['columns']==4096 and row['expected_q4_bytes']==5898240 and row['source_count']==66048,'frozen source/output geometry differs')
  require(all(re.fullmatch('[0-9a-f]{64}',row[key])for key in ('physical_mean_sha256','hf_mean_sha256','expected_hf_control_q4_sha256','expected_publisher_tiled_q4_sha256')),'typed complete payload/mean hash required')
 return p

def independent_decode(source,offset,native,tiled,means,result):
 import numpy as np
 from gguf import GGMLQuantizationType
 from gguf.quants import dequantize
 permutation=np.array([(16*(h%2)+h//2)*128+c for h in range(32)for c in range(128)],dtype=np.int64)
 mean_hf=np.frombuffer(means,dtype='<f4')[permutation].astype(np.float64)
 hashes=[hashlib.sha256()for _ in range(3)];difference_bits=0;maximum=0.;squared=[];old_squared=[];new_squared=[];old_weighted=[];new_weighted=[]
 remaining=result['elements'];chunk=result['max_chunk_elements']
 require(4096<=chunk<=65536 and chunk%4096==0,'independent chunk bound differs')
 with Path(source).open('rb')as original,Path(native).open('rb')as a,Path(tiled).open('rb')as b:
  original.seek(offset)
  while remaining:
   count=min(remaining,chunk);size=count//256*144;left=a.read(size);right=b.read(size);raw=original.read(count*2)
   require(len(left)==len(right)==size and len(raw)==count*2,'independent row read truncated')
   old=dequantize(np.frombuffer(left,dtype=np.uint8),GGMLQuantizationType.Q4_K).astype('<f4',copy=False).reshape(-1,4096)
   physical=dequantize(np.frombuffer(right,dtype=np.uint8),GGMLQuantizationType.Q4_K).astype('<f4',copy=False).reshape(-1,4096)
   canonical=physical[:,permutation];require(np.isfinite(old).all()and np.isfinite(canonical).all(),'nonfinite independent decode')
   for h,v in zip(hashes,(old,physical,canonical)):h.update(v.tobytes(order='C'))
   difference_bits+=int(np.count_nonzero(old.view('<u4')!=canonical.view('<u4')))
   source_values=np.frombuffer(raw,dtype='<f2').astype(np.float64).reshape(-1,4096);x=old.astype(np.float64)-source_values;y=canonical.astype(np.float64)-source_values;d=canonical.astype(np.float64)-old.astype(np.float64)
   maximum=max(maximum,float(np.max(np.abs(d))));squared.append(float(np.sum(d*d,dtype=np.float64)));old_squared.append(float(np.sum(x*x,dtype=np.float64)));new_squared.append(float(np.sum(y*y,dtype=np.float64)));old_weighted.append(float(np.sum(x*x*mean_hf,dtype=np.float64)));new_weighted.append(float(np.sum(y*y*mean_hf,dtype=np.float64)));remaining-=count
 require([h.hexdigest()for h in hashes]==[result['hf_control_dequant_f32_sha256'],result['physical_dequant_f32_sha256'],result['tiled_dequant_as_hf_f32_sha256']],'independent dequantized value hashes differ')
 require(difference_bits==result['canonical_different_bits']and maximum==result['canonical_max_abs_difference'],'independent canonical difference differs')
 n=result['elements'];denominator=math.fsum(mean_hf.tolist())*result['rows'];metrics={'canonical_rms_difference':math.sqrt(math.fsum(squared)/n),'hf_rmse_vs_original_f16':math.sqrt(math.fsum(old_squared)/n),'tiled_rmse_vs_original_f16':math.sqrt(math.fsum(new_squared)/n),'hf_importance_normalized_weight_rmse':math.sqrt(math.fsum(old_weighted)/denominator),'tiled_importance_normalized_weight_rmse':math.sqrt(math.fsum(new_weighted)/denominator)}
 for key,value in metrics.items():require(math.isfinite(value)and abs(value-result[key])<=1e-10,'independent error metric differs: '+key)
 return {'all_three_dequant_hashes_exact':True,'canonical_difference_bits':difference_bits,'maximum_chunk_elements':chunk,'error_metrics_reproduced_absolute_tolerance':1e-10,**metrics}

def reproduce(source,statistics,helper,output,receipt):
 source=Path(source);statistics=Path(statistics);helper=Path(helper).absolute();output=Path(output);receipt=Path(receipt);p=protocol()
 require(not output.exists()and not receipt.exists()and not output.is_symlink()and not receipt.is_symlink()and output.absolute()!=receipt.absolute()and output.parent.is_dir()and receipt.parent.is_dir(),'fresh distinct outputs under existing parents required')
 r={'schema':'lighton4b-hf-tiled-out-reproducer/v1','status':'blocked','protocol_sha256':sha(PROTOCOL),'quantization_attempted':False,'model_inference_executed':False,'original_forward_executed':False,'full_f32_model_instantiated':False,'quantized_donor_used':False,'all24_native_controls_exact':False,'all24_publisher_tiled_payloads_exact':False,'candidate_conversion_allowed':False,'tensors':{}};started=time.monotonic()
 def save():receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
 try:
  output.mkdir();require(source.stat().st_size==p['source_original_f16']['bytes'],'original unquantized F16 byte count differs')
  require(helper.is_file()and os.access(helper,os.X_OK),'compiled explicit row helper required');r['helper_sha256']=sha(helper);require(importlib.metadata.version('gguf')=='0.19.0'and importlib.metadata.version('numpy')=='2.3.5','pinned metadata/array reader versions required')
  from gguf import GGUFReader
  from gguf.quants import dequantize
  probe=subprocess.run([str(helper)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30);require(probe.returncode==2,'row helper loader/argument guard failed')
  require(sha(source)==p['source_original_f16']['sha256'],'original unquantized F16 full-file identity differs');r['source_original_f16']=p['source_original_f16'];save()
  # Validate the complete published domain first; keep the raw vectors unchanged.
  stats=adapter.source_stats(statistics);_,adapter_rows=adapter.validate_statistics(stats);hf_means={row['target']:row['normalized_f32_sha256']for row in adapter_rows};r['statistics_sha256']=sha(statistics);r['all248_statistics_validated']=True
  from gguf import GGUFReader
  reader=GGUFReader(str(source),'r');inventory={t.name:{'stored_shape':list(map(int,t.shape)),'type':int(t.tensor_type)}for t in reader.tensors};validate_source_inventory(inventory,plan());tensors={t.name:t for t in reader.tensors};r['all723_source_geometry_validated']=True;save()
  for row in p['selected']:
   name=row['name'];tensor=tensors[name];require(int(tensor.tensor_type)==1 and list(map(int,tensor.shape))==row['source_stored_shape']and int(tensor.n_bytes)==row['rows']*row['columns']*2,'original F16 matrix geometry differs');require(hf_means[name]==row['hf_mean_sha256'],'predeclared native mean control differs');mean=physical_mean(*stats[row['statistics_source_name']]);require(hashlib.sha256(mean).hexdigest()==row['physical_mean_sha256'],'predeclared physical mean differs');folder=output/name;folder.mkdir();means=folder/'physical-means.f32';means.write_bytes(mean)
   with source.open('rb')as stream:original_sha=span(stream,int(tensor.data_offset),int(tensor.n_bytes))
   native=folder/'hf-control.q4';tiled=folder/'tiled.q4';report=folder/'rows.json';command=[str(helper),str(source),str(int(tensor.data_offset)),str(row['rows']),str(means),original_sha,str(tiled),str(native),str(report),str(p['limits']['maximum_chunk_elements'])];r['quantization_attempted']=True;r['stage']=name;save();print('TENSOR '+name,flush=True)
   before=time.monotonic()
   with(folder/'helper.log').open('w')as log:child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=p['limits']['helper_timeout_seconds_per_tensor'])
   require(child.returncode==0,'row helper failed: '+name);result=json.loads(report.read_text());require(result['source_f16_sha256']==original_sha and result['source_offset']==int(tensor.data_offset)and result['rows']==row['rows']and result['columns']==4096 and result['bytes']==row['expected_q4_bytes']and result['coordinate_map_sha256']==MAP_SHA,'row helper source/layout identity differs');require(result['hf_means_sha256']==row['hf_mean_sha256']and result['physical_means_sha256']==row['physical_mean_sha256']and result['importance_supplied']is True,'actual normalized means differ');require(native.stat().st_size==tiled.stat().st_size==row['expected_q4_bytes']and sha(native)==result['hf_control_q4_sha256']and sha(tiled)==result['tiled_q4_sha256'],'actual output payload readback differs')
   result['independent']=independent_decode(source,int(tensor.data_offset),native,tiled,mean,result)
   result.update(expected_hf_control_sha256=row['expected_hf_control_q4_sha256'],expected_publisher_tiled_sha256=row['expected_publisher_tiled_q4_sha256'],hf_control_exact=result['hf_control_q4_sha256']==row['expected_hf_control_q4_sha256'],publisher_tiled_exact=result['tiled_q4_sha256']==row['expected_publisher_tiled_q4_sha256'],seconds=time.monotonic()-before);r['tensors'][name]=result;save()
  require(len(r['tensors'])==24,'complete24-tensor result required');r['all24_native_controls_exact']=all(x['hf_control_exact']for x in r['tensors'].values());r['all24_publisher_tiled_payloads_exact']=all(x['publisher_tiled_exact']for x in r['tensors'].values());r['candidate_conversion_allowed']=r['all24_native_controls_exact']and r['all24_publisher_tiled_payloads_exact'];r['status']='reproduced'if r['candidate_conversion_allowed']else'valid-quantization-byte-control-failed';return r
 except Exception as error:r['error_type']=type(error).__name__;r['error']=str(error)if isinstance(error,ValueError)else'bounded tensor reproduction failed';raise
 finally:r['seconds']=time.monotonic()-started;save()
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--statistics',type=Path,required=True);parser.add_argument('--helper',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--receipt',type=Path,required=True);args=parser.parse_args();result=reproduce(args.source,args.statistics,args.helper,args.output,args.receipt);raise SystemExit(0 if result['candidate_conversion_allowed']else 2)
