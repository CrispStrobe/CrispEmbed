#!/usr/bin/env python3
"""Adapt one authenticated dense llama.cpp input-statistics file; never quantize weights."""
import argparse,hashlib,json,math,os,struct,tempfile
from pathlib import Path
from gguf_file_type_canonicalize import parse
from lighton4b_q6_recipe import plan
SOURCE_SHA='10e8194af99deb61c457b7f84b39c57224c72f0b9ed262b27c626dc92887afd7'
SOURCE_REPO='richardyoung/LightOnOCR-3-4B-GGUF'
SOURCE_REV='8406c78c9f35b0a4d6c9cc5f69dff0e14e93ce01'
UPSTREAM='6c01da6278ab6c69ed67d9124806b50d2b299ba7'
PROFILE='lighton4b-text-inputs-hf-grouped-v1'
SUFFIX={'.mlp.gate_proj.weight':'ffn_gate.weight','.mlp.up_proj.weight':'ffn_up.weight','.mlp.down_proj.weight':'ffn_down.weight','.self_attn.q_proj.weight':'attn_q.weight','.self_attn.k_proj.weight':'attn_k.weight','.self_attn.v_proj.weight':'attn_v.weight','.self_attn.o_proj.weight':'attn_output.weight','.linear_attn.in_proj_qkv.weight':'attn_qkv.weight','.linear_attn.in_proj_z.weight':'attn_gate.weight','.linear_attn.in_proj_a.weight':'ssm_alpha.weight','.linear_attn.in_proj_b.weight':'ssm_beta.weight','.linear_attn.out_proj.weight':'ssm_out.weight'}
def require(ok,message):
 if not ok:raise ValueError(message)
def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def inventory():
 result={}
 for row in plan()['all_tensors']:
  name=row['name']
  if not name.startswith('l.layers.')or row['type']==0:continue
  suffix='.'+'.'.join(name.split('.')[3:]);require(suffix in SUFFIX,'unknown frozen projection name');up='blk.'+name.split('.')[2]+'.'+SUFFIX[suffix];require(up not in result,'duplicate projection map');result[up]={'target':name,'width':row['source_shape'][1],'inverse_v_head_columns':suffix=='.linear_attn.out_proj.weight'}
 require(len(result)==248,'exact248 nonembedding projection coverage required');return result

def native_order(values):
 require(len(values)==4096,'GDN V statistics must be4096 columns');out=[0.0]*4096
 # llama[new=(v_in_group*16+key)*128+d] = HF[old=(key*2+v_in_group)*128+d].
 for key in range(16):
  for sub in range(2):
   for d in range(128):out[(key*2+sub)*128+d]=values[(sub*16+key)*128+d]
 return out

def validate_statistics(stats,chunk_count=129,chunk_size=512):
 expected=inventory();require(set(stats)==set(expected),'missing/extra/wrong-domain matrix statistics');require(chunk_count==129 and chunk_size==512,'unreviewed calibration chunk profile');converted={};rows=[]
 for name,entry in sorted(expected.items()):
  sums,counts=stats[name];require(len(counts)==1,'dense single-count statistics only; no experts/padding');count=counts[0];require(math.isfinite(count)and count>0 and count<=2**24 and count==int(count)and count==chunk_count*chunk_size,'positive exact complete-row count required');require(len(sums)==entry['width'],'importance input-column width/padding mismatch');require(all(math.isfinite(x)and x>=0 for x in sums)and any(x>0 for x in sums),'finite nonnegative nonempty input energy required');values=native_order(sums)if entry['inverse_v_head_columns']else list(sums);means=[f32(float(x)/int(count))for x in values];require(all(math.isfinite(x)and x>=0 for x in means),'normalized importance overflow');converted[entry['target']]=(values,int(count));rows.append({'source':name,'target':entry['target'],'width':len(values),'count':int(count),'inverse_v_head_columns':entry['inverse_v_head_columns'],'normalized_f32_sha256':hashlib.sha256(struct.pack('<'+'f'*len(means),*means)).hexdigest()})
 return converted,rows

def source_stats(path):
 path=Path(path);require(path.stat().st_size==3626496 and hashlib.sha256(path.read_bytes()).hexdigest()==SOURCE_SHA,'unapproved source file identity');g=parse(path,{0:(1,4)});m=g['metadata'];require(m.get('general.type',{}).get('value')=='imatrix','source is not upstream imatrix');require(m.get('imatrix.chunk_count',{}).get('value')==129 and m.get('imatrix.chunk_size',{}).get('value')==512,'source chunk metadata differs');require(m.get('imatrix.datasets',{}).get('value')=={'element':8,'count':1},'single declared dataset required');expected=inventory();names={name+suffix for name in expected for suffix in('.in_sum2','.counts')};require(set(g['tensors'])==names,'exact496 upstream sum/count tensors required');stats={}
 with path.open('rb')as f:
  for name,entry in expected.items():
   vectors=[]
   for suffix,width in[('.in_sum2',entry['width']),('.counts',1)]:
    t=g['tensors'][name+suffix];require(t['type']==0 and t['stored_shape']in([width],[width,1]),'dense F32 shape domain mismatch');f.seek(t['data_offset']);raw=f.read(width*4);require(len(raw)==width*4,'truncated statistic values');vectors.append(list(struct.unpack('<'+'f'*width,raw)))
   stats[name]=vectors
 return stats

def text(s):
 b=s.encode();return struct.pack('<Q',len(b))+b
def encode(converted):
 metadata={'general.architecture':'crispembed-imatrix','crispembed.imatrix.profile':PROFILE,'crispembed.imatrix.source_sha256':SOURCE_SHA,'crispembed.imatrix.source_repository':SOURCE_REPO,'crispembed.imatrix.source_revision':SOURCE_REV,'crispembed.imatrix.upstream_semantics_commit':UPSTREAM,'crispembed.imatrix.calibration_domain':'external-text; publisher calibration_datav3; corpus content/source weights not independently authenticated','crispembed.imatrix.coordinates':'HF grouped V heads; inverse128-channel tiles only for24 GDN out inputs'}
 records=[text(k)+struct.pack('<I',8)+text(v)for k,v in metadata.items()];records += [text('imatrix.version')+struct.pack('<II',4,1),text('crispembed.imatrix.expected_entries')+struct.pack('<II',4,248)];descriptors=[];payload=bytearray()
 for name,(values,count)in sorted(converted.items()):
  records.append(text('count.'+name)+struct.pack('<IQ',10,count));payload.extend(b'\0'*((-len(payload))%32));descriptors.append(text(name)+struct.pack('<IQIQ',1,len(values),0,len(payload)));payload.extend(struct.pack('<'+'f'*len(values),*values))
 header=b'GGUF'+struct.pack('<IQQ',3,len(descriptors),len(records))+b''.join(records)+b''.join(descriptors);header+=b'\0'*((-len(header))%32);return header+payload

def validate_output(path,rows):
 g=parse(path,{0:(1,4)});require(g['metadata']['general.architecture']['value']=='crispembed-imatrix'and g['metadata']['crispembed.imatrix.profile']['value']==PROFILE and g['metadata']['crispembed.imatrix.source_sha256']['value']==SOURCE_SHA,'output domain/provenance differs');require(set(g['tensors'])=={r['target']for r in rows},'output coverage differs')
 with Path(path).open('rb')as f:
  for row in rows:
   name=row['target'];count=g['metadata'].get('count.'+name,{});require(count.get('kind')==10 and count.get('value')==row['count'],'output count type/value differs');t=g['tensors'][name];require(t['stored_shape']==[row['width']],'output input width differs');f.seek(t['data_offset']);values=struct.unpack('<'+'f'*row['width'],f.read(row['width']*4));means=[f32(float(x)/row['count'])for x in values];require(hashlib.sha256(struct.pack('<'+'f'*len(means),*means)).hexdigest()==row['normalized_f32_sha256'],'consumer mean-squared values differ')
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def adapt(source,output,receipt):
 output=Path(output);receipt=Path(receipt);require(output.absolute()!=receipt.absolute()and not output.exists()and not receipt.exists()and not output.is_symlink()and not receipt.is_symlink(),'distinct new output/receipt paths required');report={'schema':'lighton4b-imatrix-adapter/v1','status':'failed','source_sha256':SOURCE_SHA,'source_repository':SOURCE_REPO,'source_revision':SOURCE_REV,'upstream_semantics_commit':UPSTREAM,'quantization_executed':False,'model_forward_executed':False,'quality_qualified':False,'external_text_calibration':True,'calibration_corpus_content_authenticated':False,'calibration_source_weights_authenticated':False};temporary=None
 try:
  converted,rows=validate_statistics(source_stats(source));data=encode(converted)
  with tempfile.NamedTemporaryFile(prefix=output.name+'.',suffix='.partial',dir=output.parent,delete=False)as f:temporary=Path(f.name);f.write(data)
  digest=validate_output(temporary,rows);os.replace(temporary,output);temporary=None;report.update(status='adapted',output_sha256=digest,output_bytes=len(data),entries=248,inverse_v_head_vectors=24,normalization='F32 raw sum2 / exact positive integer row count; same dense upstream convention',tied_embedding_head_statistics='intentionally absent; upstream process-output disabled',tensors=rows)
 except Exception as e:
  report['error_type']=type(e).__name__;report['error']=str(e)if isinstance(e,ValueError)else'bounded adaptation failed';raise
 finally:
  if temporary is not None:temporary.unlink(missing_ok=True)
  receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 return report
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--receipt',required=True,type=Path);a=p.parse_args();adapt(a.input,a.output,a.receipt)
