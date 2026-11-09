import hashlib,json,math,zipfile
from pathlib import Path
import numpy as np
if not __debug__:raise RuntimeError("audit assertions must be enabled")
assert np.__version__=='2.3.5'
expected='d5b7f992249a3f6b553bd3e41f73cc5e590eeaae2d1380d9d34d81aaf3d316e8'
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
matches=[p for p in Path('/kaggle/input').rglob('heldout-result.json')if p.is_file()and p.stat().st_size<4<<20 and sha(p)==expected]
assert len(matches)==1
p=matches[0];r=json.loads(p.read_text());base=p.parent.resolve()
assert r['source_sha']=='d378c6cc055e3a31b0d4cc9dad54a5b41f10d792'and r['status']=='measured_provisional_heldout'and r['case']=='inventory-table'and r['qualified']is False and r['source_observer_full_neutral']is True
assert len(r['artifacts'])<=1000 and sum(x['bytes']for x in r['artifacts'].values())<=8<<30
verified={}
for name,row in r['artifacts'].items():
 f=(base/name).resolve();assert base in f.parents and f.is_file()and f.stat().st_size==row['bytes']and sha(f)==row['sha256'];verified[name]=row
roles=[('observed-control',r['observed_control_rows'],'greedy'),('upstream-on',r['source_raw_rows'],'source')]+[(arm,r['arms'][arm]['raw_rows'],'greedy')for arm in['native-composed','native-source-features']]
rows_verified=0
for folder,meta,stem in roles:
 archive=(base/folder/meta['archive']['file']).resolve();assert base in archive.parents and sha(archive)==meta['archive']['sha256']
 with zipfile.ZipFile(archive)as z:
  names=[f'{stem}.{row["position"]}.logits.f32'if stem=='greedy'else f'source.logits.{row["position"]}.f32'for row in meta['rows']]
  assert z.namelist()==names and len(names)<=1536
  for name,row in zip(names,meta['rows']):
   info=z.getinfo(name);assert info.file_size==row['bytes']==248320*4 and info.compress_type==zipfile.ZIP_STORED
   raw=z.read(name);assert hashlib.sha256(raw).hexdigest()==row['sha256'];values=np.frombuffer(raw,dtype='<f4');assert np.isfinite(values).all()and int(values.argmax())==row['argmax'];rows_verified+=1
out=Path('/kaggle/working/inventory-raw-audit.json');out.write_text(json.dumps(dict(schema='lighton4b-inventory-raw-audit/v1',receipt_sha256=expected,owned_files_verified=len(verified),owned_bytes_verified=sum(v['bytes']for v in verified.values()),raw_zip_archives_verified=4,raw_rows_finite_argmax_and_hash_verified=rows_verified,artifacts=verified,model_forward_executed=False),indent=2)+'\n')
print('INVENTORY_RAW_AUDIT PASS files='+str(len(verified))+' rows='+str(rows_verified),flush=True)
