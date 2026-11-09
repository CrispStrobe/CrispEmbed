"""Private capability transport only. URLs are never printed or credentials loaded."""
import argparse,concurrent.futures,hashlib,json,os,signal,threading,time,urllib.error,urllib.parse,urllib.request
from pathlib import Path
HOST='www.kaggleusercontent.com';FILES=1457;BYTES=2735343103
class ClosedRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,request,fp,code,msg,headers,newurl):
  p=urllib.parse.urlsplit(newurl)
  if p.scheme!='https'or p.hostname!=HOST:raise ValueError('closed redirect host')
  return super().redirect_request(request,fp,code,msg,headers,newurl)
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  while b:=f.read(8<<20):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();manifest=json.loads(a.manifest.read_text());require(len(manifest)==FILES and sum(v['bytes']for v in manifest.values())==BYTES,'exact closed source census')
 for name,v in manifest.items():
  path=Path(name);u=urllib.parse.urlsplit(v['url']);require(not path.is_absolute()and '..'not in path.parts and path.suffix not in['.gguf','.safetensors']and u.scheme=='https'and u.hostname==HOST and not u.username and not u.password and not u.fragment and u.port in[None,443]and len(v['url'])<=8192,'closed source path/HTTPS host');require(0<=v['bytes']<=1<<30 and len(v['sha256'])==64,'bounded actual artifact')
 require(not a.out.exists(),'fresh owned mirror');a.out.mkdir();started=time.monotonic();r=dict(status='failed',files=0,bytes=0,urls_published=False,credentials_loaded=False,model_files=0,model_forward_executed=False);lock=threading.Lock();stop=threading.Event()
 def canceled(sig,frame):stop.set();raise TimeoutError('bounded relay interrupted')
 for s in[signal.SIGALRM,signal.SIGTERM,signal.SIGINT]:signal.signal(s,canceled)
 signal.alarm(900)
 def fetch(item):
  name,v=item;target=a.out/name;target.parent.mkdir(parents=True,exist_ok=True);part=target.with_name(target.name+'.relay-part')
  for attempt in range(2):
   require(not stop.is_set(),'relay canceled');h=hashlib.sha256();size=0
   try:
    opener=urllib.request.build_opener(ClosedRedirect())
    with opener.open(urllib.request.Request(v['url'],headers={'User-Agent':'CrispEmbed-task-artifact-relay/1'}),timeout=45)as response,part.open('xb')as output:
     require(urllib.parse.urlsplit(response.geturl()).hostname==HOST,'actual host preserved');length=response.headers.get('Content-Length');require(length is None or int(length)==v['bytes'],'exact response length')
     while True:
      require(not stop.is_set(),'relay canceled');b=response.read(1<<20)
      if not b:break
      size+=len(b);require(size<=v['bytes'],'bounded exact response bytes');h.update(b);output.write(b)
    require(size==v['bytes']and h.hexdigest()==v['sha256'],'exact downloaded SHA');part.replace(target)
    with lock:r['files']+=1;r['bytes']+=size
    return
   except (TimeoutError,ConnectionError,urllib.error.URLError)as error:
    if part.exists():part.unlink()
    if isinstance(error,urllib.error.HTTPError)and error.code not in[429,500,502,503,504]:raise RuntimeError('closed transport HTTP status')from None
    if attempt:raise RuntimeError('bounded artifact transport exhausted')from None
   except BaseException:
    if part.exists():part.unlink()
    raise
  raise RuntimeError('transport exhausted')
 try:
  pool=concurrent.futures.ThreadPoolExecutor(max_workers=8);futures=[]
  try:
   futures=[pool.submit(fetch,item)for item in manifest.items()]
   for future in concurrent.futures.as_completed(futures):future.result()
  except BaseException:
   stop.set()
   for future in futures:future.cancel()
   pool.shutdown(wait=False,cancel_futures=True);raise
  else:pool.shutdown(wait=True)
  require(r['files']==FILES and r['bytes']==BYTES,'all owned bytes mirrored');require(sha(a.out/'planner-heldout.json')=='f231ab48c9bff411a1f9c96cfb79eda084f2f0b7cee46f9ea747ccc734d5f8e6','exact failed producer');r['status']='mirrored-all-authenticated'
 except BaseException as error:stop.set();r['error_type']=type(error).__name__;raise RuntimeError('private relay failed; original artifacts untouched')from None
 finally:stop.set();signal.alarm(0);r['elapsed_seconds']=time.monotonic()-started;a.report.write_text(json.dumps(r,indent=2)+'\n')
if __name__=='__main__':main()
