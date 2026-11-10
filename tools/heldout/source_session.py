"""Bounded localhost source session; no external messages or floating source."""
import json,os,resource,signal,socket,subprocess,sys,threading,time,urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"physical-f16-attention-probe"))
from runtime_guard import WorkerDeadline

class ArmDeadline(TimeoutError):pass

def stop(proc):
 if proc is not None and proc.poll()is None:
  os.killpg(proc.pid,signal.SIGTERM)
  try:proc.wait(timeout=5)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=5)

def api(port,path,payload,deadline):
 remaining=deadline-time.monotonic()-12
 if remaining<=0:raise ArmDeadline('source arm deadline')
 raw=None if payload is None else json.dumps(payload).encode()
 request=urllib.request.Request(f'http://127.0.0.1:{port}'+path,data=raw,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(request,timeout=min(remaining,1200))as response:
  data=response.read((16<<20)+1)
  if len(data)>16<<20:raise ValueError('bounded source JSON response')
  return json.loads(data)

class Session:
 def __init__(self,args,log,env,seconds=1200):
  if not 0<seconds<=1200:raise ValueError('closed source arm budget')
  self.args=args;self.log=Path(log);self.env=env;self.seconds=seconds;self.proc=None;self.done=threading.Event();self.stats={}
 def __enter__(self):
  self.started=time.monotonic();self.deadline=self.started+self.seconds;self.before=resource.getrusage(resource.RUSAGE_CHILDREN);self.file=self.log.open('w');self.proc=subprocess.Popen(list(map(str,self.args)),stdout=self.file,stderr=subprocess.STDOUT,env=self.env,start_new_session=True);self.stats.update(pid=self.proc.pid,status='running',peak_rss_bytes_observed=0)
  def watch():
   while not self.done.wait(.5):
    try:
     peak=int(next(x.split()[1]for x in Path(f'/proc/{self.proc.pid}/status').read_text().splitlines()if x.startswith('VmHWM:')))*1024;self.stats['peak_rss_bytes_observed']=max(peak,self.stats['peak_rss_bytes_observed'])
    except(OSError,StopIteration):pass
    if time.monotonic()>=self.deadline-12:
     self.stats['deadline_reached']=True
     if self.proc.poll()is None:
      try:os.killpg(self.proc.pid,signal.SIGTERM)
      except ProcessLookupError:pass
     return
  self.thread=threading.Thread(target=watch,daemon=True)
  try:self.thread.start()
  except BaseException:
   try:stop(self.proc)
   finally:self.done.set();self.file.close();self.stats.update(status='failed_during_session_setup',returncode=self.proc.returncode,elapsed_seconds=time.monotonic()-self.started)
   raise
  return self
 def ready(self,port):
  startup_deadline=min(self.deadline,time.monotonic()+180)
  for _ in range(360):
   if time.monotonic()>=startup_deadline-12:raise ArmDeadline('source startup arm deadline')
   if self.proc.poll()is not None:raise RuntimeError('source server stopped during startup')
   try:api(port,'/health',None,min(startup_deadline,time.monotonic()+14));return
   except (ArmDeadline,WorkerDeadline):raise
   except Exception:
    remaining=startup_deadline-time.monotonic()-12
    if remaining<=0:raise TimeoutError('source startup deadline')
    time.sleep(min(.5,remaining))
  raise TimeoutError('source startup bounded attempts')
 def __exit__(self,typ,value,tb):
  try:stop(self.proc)
  finally:
   self.done.set();self.thread.join(timeout=2);self.file.close();after=resource.getrusage(resource.RUSAGE_CHILDREN);self.stats.update(status='completed'if typ is None else'failed',elapsed_seconds=time.monotonic()-self.started,returncode=self.proc.returncode,child_user_seconds=after.ru_utime-self.before.ru_utime,child_system_seconds=after.ru_stime-self.before.ru_stime,rss_scope='Server VmHWM sampled every0.5s; may miss short-lived peaks.')
  return False

def port():
 with socket.socket()as sock:sock.bind(('127.0.0.1',0));return sock.getsockname()[1]
