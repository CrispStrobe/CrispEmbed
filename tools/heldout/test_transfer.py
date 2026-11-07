import importlib.util,tempfile,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('transfer',Path(__file__).with_name('transfer.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Transfers(unittest.TestCase):
 def test_retry_owned_partial_and_authenticate(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'payload';rows=[]
   def fetch(url,path,size,digest):
    self.assertFalse(path.exists());path.write_bytes(b'x')
    if len(rows)==1:raise TimeoutError('transport')
   m.download(fetch,'unused',p,1,'frozen',rows);self.assertEqual([r['status']for r in rows],['transport_failure','authenticated']);self.assertEqual(rows[0]['partial_bytes'],1)
 def test_deadline_and_integrity_fail_closed(self):
  for error in [m.WorkerDeadline('whole deadline'),ValueError('hash mismatch')]:
   with tempfile.TemporaryDirectory()as d:
    p=Path(d)/'payload';rows=[]
    def fetch(*args):p.write_bytes(b'x');raise error
    with self.assertRaises(type(error)):m.download(fetch,'unused',p,1,'frozen',rows)
    self.assertEqual(len(rows),1);self.assertTrue(p.exists())
 def test_finite_and_preexisting_guard(self):
  with tempfile.TemporaryDirectory()as d:
   p=Path(d)/'payload';rows=[]
   def fetch(*args):raise TimeoutError('transport')
   with self.assertRaises(TimeoutError):m.download(fetch,'unused',p,1,'frozen',rows)
   self.assertEqual(len(rows),3);p.write_bytes(b'owned-by-other')
   with self.assertRaises(ValueError):m.download(fetch,'unused',p,1,'frozen',[])
   self.assertEqual(p.read_bytes(),b'owned-by-other')
if __name__=='__main__':unittest.main()
