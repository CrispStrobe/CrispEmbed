"""Finite retries for task-owned authenticated downloads; never retry deadlines."""
import time,urllib.error
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'physical-f16-attention-probe'))
from runtime_guard import WorkerDeadline

def download(fetch,url,path,size,digest,records):
 path=Path(path)
 if path.exists():raise ValueError('task download destination already exists')
 for attempt in range(1,4):
  row={'attempt':attempt,'expected_bytes':size};records.append(row);started=time.monotonic()
  try:
   fetch(url,path,size,digest);row.update(status='authenticated',elapsed_seconds=time.monotonic()-started);return
  except WorkerDeadline:
   row.update(status='worker_deadline',elapsed_seconds=time.monotonic()-started);raise
  except (TimeoutError,ConnectionError,urllib.error.URLError)as error:
   row.update(status='transport_failure',error_type=type(error).__name__,partial_bytes=path.stat().st_size if path.exists()else 0,elapsed_seconds=time.monotonic()-started)
   # Only this initially-absent task destination may be removed for a retry.
   if path.exists():path.unlink()
   if attempt==3:raise
