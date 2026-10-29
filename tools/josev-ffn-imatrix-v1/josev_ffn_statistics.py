"""Native F32 development activation sums/counts, exact existing imatrix format."""
import hashlib,json,math,struct,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from gguf_file_type_canonicalize import parse,sha,require
NAMES=('l.layers.23.mlp.gate_proj.weight','l.layers.23.mlp.up_proj.weight')
IDS=('mixed-de','null-and-options','json-unicode','state-only-clip')
ROWS=(319,197,165,512)
WIDTH=2048

def capture(path,rows):
 path=Path(path);body=json.loads((path/'pre_ffn23.json').read_text())
 require(body==dict(schema='crispembed-josev-native-f32-pre-ffn23-v1',calls=1,layer=23,batch=1,past=0,rows=rows,columns=WIDTH,
  epsilon_bits=struct.unpack('<I',struct.pack('<f',1e-6))[0],gate=NAMES[0],up=NAMES[1],post_norm='l.layers.23.post_attention_layernorm.weight',
  stored_types=[0,0,0],actual_native_f32_operand=True,python_reference_activation=False),'Exact actual normalized-operand metadata required')
 data=(path/'pre_ffn23.f32').read_bytes();require(len(data)==rows*WIDTH*4,'Exact live normalized axes/extent')
 require(all(math.isfinite(v)for(v,)in struct.iter_unpack('<f',data)),'Nonfinite live operand')
 return data

def statistics(packets,width=WIDTH):
 require(type(width)is int and 1<=width<=WIDTH and 1<=len(packets)<=4,'Bounded source-derived statistics')
 sums=[0.]*width;count=0
 for data in packets:
  require(data and len(data)%(width*4)==0 and len(data)<=512*WIDTH*4,'Whole finite F32 rows required')
  count+=len(data)//(width*4)
  for i,(value,)in enumerate(struct.iter_unpack('<f',data)):
   require(math.isfinite(value),'Nonfinite activation')
   # Predeclared case,row,column F64 multiply then add; no fused operation.
   sums[i%width]+=value*value
 require(count>0 and all(math.isfinite(v)and v>=0 for v in sums)and any(v>0 for v in sums),'Finite nonzero F64 sums required')
 stored=struct.pack('<'+'f'*width,*sums)
 require(all(math.isfinite(v)and v>=0 for(v,)in struct.iter_unpack('<f',stored)),'F32 sumsq overflow/nonfinite')
 return stored,count

def write(path,sums,count,names=NAMES):
 require(tuple(names)==NAMES and len(set(names))==2 and len(sums)==WIDTH*4 and type(count)is int and 0<count<=2048,'Exact two-name imatrix/count/width')
 require(all(math.isfinite(v)and v>=0 for(v,)in struct.iter_unpack('<f',sums)),'Finite nonnegative stored sums')
 def text(v):b=v.encode();return struct.pack('<Q',len(b))+b
 metadata=[text('general.architecture')+struct.pack('<I',8)+text('crispembed-imatrix'),text('imatrix.version')+struct.pack('<II',4,1)]
 for name in names:metadata.append(text('count.'+name)+struct.pack('<IQ',10,count))
 header=b'GGUF'+struct.pack('<IQQ',3,2,len(metadata))+b''.join(metadata)
 for i,name in enumerate(names):header+=text(name)+struct.pack('<I Q I Q',1,WIDTH,0,i*len(sums))
 start=(len(header)+31)//32*32;path=Path(path)
 with path.open('xb')as f:f.write(header);f.write(b'\0'*(start-len(header)));f.write(sums);f.write(sums)
 return verify(path,count)

def verify(path,count=None):
 p=parse(path);m=p['metadata'];rows=p['tensors']
 require(set(m)=={'general.architecture','imatrix.version',*('count.'+n for n in NAMES)}and m['general.architecture']['kind']==8
  and m['general.architecture']['value']=='crispembed-imatrix'and m['imatrix.version']['kind']==4 and m['imatrix.version']['value']==1,'Actual CrispEmbed imatrix metadata')
 require(set(rows)==set(NAMES),'Exact two target weight names required')
 data=[];counts=[]
 with Path(path).open('rb')as f:
  for name in NAMES:
   row=rows[name];require(row['type']==0 and row['stored_shape']==[WIDTH]and row['bytes']==WIDTH*4,'Exact F32 per-column sums vector')
   c=m['count.'+name];require(c['kind']==10 and type(c['value'])is int and 0<c['value']<=2048,'Positive U64 row count required')
   counts.append(c['value']);f.seek(row['data_offset']);b=f.read(row['bytes']);require(len(b)==WIDTH*4,'Truncated sums')
   values=[v for(v,)in struct.iter_unpack('<f',b)];require(all(math.isfinite(v)and v>=0 for v in values)and any(v>0 for v in values),'Finite nonzero imatrix payload')
   data.append(b)
 require(counts[0]==counts[1]and(count is None or counts[0]==count)and data[0]==data[1],'Shared operand implies identical vectors/counts')
 # Exactly mirror the C loader: F64 reciprocal once, multiply, then F32 cast.
 inverse=1.0/counts[0]
 means=struct.pack('<2048f',*(v*inverse for(v,)in struct.iter_unpack('<f',data[0])))
 return dict(sha256=sha(path),bytes=Path(path).stat().st_size,names=list(NAMES),columns=WIDTH,row_count=counts[0],
  positive_columns=sum(v>0 for(v,)in struct.iter_unpack('<f',data[0])),zero_columns=sum(v==0 for(v,)in struct.iter_unpack('<f',data[0])),raw_sums_sha256=hashlib.sha256(data[0]).hexdigest(),loader_mean_squared_f32_sha256=hashlib.sha256(means).hexdigest(),vectors_identical=True)

def collect(root,freeze,model):
 root=Path(root);require([r['id']for r in freeze['records']]==list(IDS),'All four frozen development records, no NEW20')
 packets=[];records=[]
 for i,(record,rows)in enumerate(zip(freeze['records'],ROWS)):
  require(len(record['encoded']['input_ids'])==rows and len(record['attention_mask'])==rows,'Exact source IDs/masks/length')
  a=capture(root/'f32-observed'/str(i),rows);b=capture(root/'f32-repeat'/str(i),rows)
  require(a==b,'Observed/repeat live normalized operands differ')
  packets.append(a);records.append(dict(id=record['id'],rows=rows,columns=WIDTH,bytes=len(a),sha256=hashlib.sha256(a).hexdigest(),
   original_ids_sha256=hashlib.sha256(json.dumps(record['encoded']['input_ids'],separators=(',',':')).encode()).hexdigest(),repeat_bit_exact=True))
 sums,count=statistics(packets);require(count==1193,'Each frozen prefill row counted once, no repeat doubling')
 path=root/'final-pair-native-f32.imatrix.gguf';proof=write(path,sums,count)
 return dict(schema='crispembed-josev-final-pair-imatrix-v1',source_model=model,native_capture=True,python_reference_activation=False,
  accumulation='case,row,column order; F64 unfused multiplication/addition; one F32 cast before storing raw sums; existing loader divides by U64 row count',
  records=records,imatrix=proof,holdout_access=False,development_only=True,publication_qualified=False)

def usage(log):
 require('imatrix: loaded importance vectors for 2 tensors'in log,'Imatrix loader did not load exactly two vectors')
 require(log.count('(imatrix)')==2 and ', 2 with imatrix'in log and 'imatrix shape'not in log and '(fallback 'not in log,'Actual C use count/skipped/fallback differs')
 matched=[]
 for line in log.splitlines():
  if '(imatrix)'in line:
   names=[n for n in NAMES if n in line];require(len(names)==1,'Unexpected calibrated weight name');matched+=names
 require(sorted(matched)==sorted(NAMES),'Both final gate/up must actually consume calibration')
 return dict(loaded=2,used=2,matched_names=matched,no_skipped_or_fallback=True)
