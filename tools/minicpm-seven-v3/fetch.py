#!/usr/bin/env python3
"""Versioned checkpoint identity adapter; frozen retained-array fetch is unchanged."""
import argparse,hashlib,importlib.util,json,os,re,sys,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import seven_contract as c
spec=importlib.util.spec_from_file_location('frozen_minicpm_fetch',c.ROOT/'tools/minicpm-retained-v3/fetch.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
BLOB=dict(bytes=2213,git_oid='6e5b931515842cfaa2c774e9117266bb19a23372')

def identities(files):
 c.require(files and '.gitattributes'in files,'Frozen checkpoint inventory required')
 for name,pin in files.items():
  c.require(type(pin.get('bytes'))is int and 0<pin['bytes']<=8*2**30,'Bounded checkpoint extent required')
  if 'sha256'in pin:c.require(isinstance(pin['sha256'],str)and re.fullmatch('[0-9a-f]{64}',pin['sha256']),'Invalid checkpoint SHA256')
  else:c.require(name=='.gitattributes'and pin==BLOB,'Only frozen attributes blob may lack SHA256')
 return files

def blob_identity(data,pin):
 c.require(len(data)==pin['bytes']and hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==pin['git_oid'],'Checkpoint Git blob identity differs')
 return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),git_blob_oid=pin['git_oid'])

def attributes(url,path,pin):
 c.require(url.startswith('https://')and pin==BLOB and not path.exists()and not path.is_symlink(),'Fresh frozen attributes destination required')
 try:
  with urllib.request.urlopen(url,timeout=180)as response:
   c.require(int(response.headers['Content-Length'])==pin['bytes'],'Attributes advertised extent differs')
   data=response.read(pin['bytes']+1)
 except Exception:raise ValueError('Checkpoint attributes transfer rejected')from None
 result=blob_identity(data,pin)
 with path.open('xb')as output:output.write(data)
 return result

def fetch(stage,out,reviewed):
 result=dict(completed=False,mode='continue',original_rerun=False,model_forwards=0,phase='preflight')
 try:
  result['source']=c.source_guard(reviewed,True);result['resources']=c.old.resources(out,stage);result['tooling']=c.old.numerical_environment()
  source=json.loads((c.ROOT/'docs/minicpm-ocr-source.json').read_text());files=identities(source['files'])
  result['phase']='retained-array-authentication';c.atomic(out/'fetch.json',result)
  # This low-level call only authenticates retained arrays. It has no checkpoint
  # accesses, source-guard overrides, original forwards or framework imports.
  proof=old.fetch('analysis',stage,out);c.atomic(out/'retained-array-fetch.json',proof)
  c.require(proof['completed']and proof['model_files_accessed']is False,'Retained-array barrier rejected')
  result.update(phase='checkpoint-authentication',retained_array_fetch_sha256=c.sha(out/'retained-array-fetch.json'),checkpoint={});c.atomic(out/'fetch.json',result)
  import importlib
  module=importlib.import_module('kaggle.api.kaggle_api_extended');api=module.KaggleApi()
  try:api.authenticate()
  except Exception:raise ValueError('Existing hosted Kaggle credentials rejected')from None
  catalog=json.loads(c.old.CATALOG.read_text());urls,service=old.client_index(api,catalog['original_service']);result['checkpoint_service']=service
  checkpoint=stage/'checkpoint';checkpoint.mkdir(exist_ok=False)
  for name,pin in files.items():
   member='checkpoint/'+name;c.require(member in urls,'Exact checkpoint member absent: '+name);path=c.old.safe(checkpoint,name)
   if 'sha256'in pin:identity=old.transfer(urls[member],path,pin,8*2**30)
   else:identity=attributes(urls[member],path,pin)
   result['checkpoint'][name]=identity;c.atomic(out/'fetch.json',result)
  result.update(completed=True,phase='authenticated',model_files_accessed=True,retained_service=proof['retained_service'],original_service=proof['original_service'],catalog_sha256=proof['catalog_sha256'],historical_missing_sha256_failure_preserved=True)
  return result
 except Exception as error:result['error']=type(error).__name__+': '+str(error);raise
 finally:c.atomic(out/'fetch.json',result)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['continue']);p.add_argument('--reviewed-source',required=True)
 for name in ['stage','out']:p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();fetch(a.stage,a.out,a.reviewed_source)
