#!/usr/bin/env python3
"""Stream authenticated retained activations only; private capability URLs are never emitted."""
import concurrent.futures,hashlib,json,os,signal,sys,time,urllib.parse,urllib.request
from pathlib import Path
import read_trace
MAX_BYTES=2<<30
HOSTS={'www.kaggleusercontent.com','storage.googleapis.com'}
def require(x,m):
 if not x:raise ValueError(m)
def closed_name(name):
 p=Path(name);require(not p.is_absolute()and'..'not in p.parts and name==p.as_posix()and name not in['','.'],'closed artifact name');return p
def safe_url(url):
 u=urllib.parse.urlparse(url);require(u.scheme=='https'and u.hostname in HOSTS and u.username is None and u.password is None,'allowlisted artifact transport');return url
class ClosedRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  safe_url(newurl);return super().redirect_request(req,fp,code,msg,headers,newurl)
def fetch(name,row,url,root,end):
 target=root/closed_name(name);target.parent.mkdir(parents=True,exist_ok=True);partial=target.with_name(target.name+'.partial');require(not target.exists()and not partial.exists(),'fresh task-owned artifact destination');expected=row['bytes'];require(0<=expected<=8<<20,'bounded retained single artifact');h=hashlib.sha256();count=0
 try:
  safe_url(url)
  with urllib.request.build_opener(ClosedRedirect()).open(url,timeout=min(60,max(1,end-time.monotonic())))as source,partial.open('xb')as out:
   safe_url(source.geturl())
   while True:
    require(time.monotonic()<end,'artifact relay whole deadline');data=source.read(min(1<<20,expected-count+1))
    if not data:break
    count+=len(data);require(count<=expected,'bounded artifact transport bytes');h.update(data);out.write(data)
  require(count==expected and h.hexdigest()==row['sha256'],'exact retained artifact bytes/hash '+name);partial.rename(target)
 except BaseException:
  if partial.exists():partial.unlink()
  raise RuntimeError('retained artifact transfer rejected: '+name)from None
 return count
def main():
 started=time.monotonic();end=started+540;out=Path(os.environ['HEAVY_OUT']);root=out/'retained-source-capture';root.mkdir();report={'schema':'lighton4b-private-artifact-relay/v1','status':'failed','inference_executed':False,'model_downloaded':False,'credential_values_logged':False};old=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in()).throw(TimeoutError('bounded900s relay/reader')));signal.alarm(900)
 try:
  private=Path(os.environ['LIGHTON4B_PRIVATE_TRACE_URLS']);require(private.is_relative_to(Path(os.environ['HEAVY_SCRATCH']))and not private.is_relative_to(out),'private capability manifest outside artifacts');m=json.loads(private.read_text());require(m['schema']=='lighton4b-private-artifact-urls/v1'and m['receipt_sha256']==read_trace.PRODUCER and len(m['urls'])==916,'immutable private artifact URL manifest');receipt_row={'bytes':int(os.environ['LIGHTON4B_TRACE_RECEIPT_BYTES']),'sha256':read_trace.PRODUCER};fetch('text-trace.json',receipt_row,m['urls']['text-trace.json'],root,end);p=json.loads((root/'text-trace.json').read_text());rows=p['artifacts'];require(set(m['urls'])==set(rows)|{'text-trace.json'}and len(rows)==915 and sum(r['bytes']for r in rows.values())<MAX_BYTES,'complete bounded producer artifact inventory');require(p['status']=='failed'and p['error']=='actual table prompt token census'and p['all282_scores_observer_neutral'],'complete neutral source capture with reader-only negative')
  for name,row in rows.items():closed_name(name);require(0<=row['bytes']<=8<<20 and len(row['sha256'])==64,'closed producer artifact size/hash');safe_url(m['urls'][name])
  with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
   tasks=[pool.submit(fetch,name,row,m['urls'][name],root,end)for name,row in rows.items()]
   try:report['copied_bytes']=sum(task.result()for task in concurrent.futures.as_completed(tasks))
   except BaseException:
    for task in tasks:task.cancel()
    raise
  report['copied_files']=len(rows);report['seconds_copy']=time.monotonic()-started;private.unlink();m.clear();os.environ['LIGHTON4B_RETAINED_TRACE_ROOT']=str(root);read_trace.main();report['status']='completed_authenticated_artifact_analysis'
 except BaseException as e:report['error_type']=type(e).__name__;report['error']='Artifact relay or strict reader failed; capability URLs redacted';raise RuntimeError(report['error'])from None
 finally:
  signal.alarm(0);signal.signal(signal.SIGALRM,old);report['seconds']=time.monotonic()-started;(out/'text-trace-relay.json').write_text(json.dumps(report,indent=2)+'\n');private_path=os.environ.get('LIGHTON4B_PRIVATE_TRACE_URLS')
  if private_path and Path(private_path).exists():Path(private_path).unlink()
if __name__=='__main__':main()
