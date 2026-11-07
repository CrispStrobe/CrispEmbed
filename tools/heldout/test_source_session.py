import importlib.util,os,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch,Mock
s=importlib.util.spec_from_file_location('session',Path(__file__).with_name('source_session.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class Lifecycle(unittest.TestCase):
 def test_normal_and_exception_cleanup(self):
  with tempfile.TemporaryDirectory()as d:
   with m.Session([sys.executable,'-S','-c','pass'],Path(d)/'ok.log',dict(os.environ),15)as session:session.proc.wait(timeout=5)
   self.assertEqual(session.stats['returncode'],0);self.assertEqual(session.stats['status'],'completed')
   with self.assertRaises(ValueError):
    with m.Session([sys.executable,'-S','-c','import time; time.sleep(60)'],Path(d)/'stop.log',dict(os.environ),15)as session:raise ValueError('intentional control')
   self.assertIsNotNone(session.proc.poll());self.assertEqual(session.stats['status'],'failed')
 def test_closed_deadline_not_swallowed(self):
  session=m.Session([],Path('unused'),{},15);session.deadline=time.monotonic()-1;session.proc=Mock();session.proc.poll.return_value=None
  with patch.object(m,'api')as api:
   with self.assertRaises(m.ArmDeadline):session.ready(1)
   api.assert_not_called()
 def test_transport_timeout_retries_but_arm_deadline_propagates(self):
  session=m.Session([],Path('unused'),{},1200);session.deadline=time.monotonic()+1200;session.proc=Mock();session.proc.poll.return_value=None
  with patch.object(m,'api',side_effect=[TimeoutError('transport'),{}])as api,patch.object(m.time,'sleep')as sleep:session.ready(1);self.assertEqual(api.call_count,2);sleep.assert_called_once()
  with patch.object(m,'api',side_effect=m.ArmDeadline('closed')):
   with self.assertRaises(m.ArmDeadline):session.ready(1)
 def test_whole_worker_alarm_inside_health_not_swallowed(self):
  session=m.Session([],Path('unused'),{},1200);session.deadline=time.monotonic()+1200;session.proc=Mock();session.proc.poll.return_value=None
  with patch.object(m,'api',side_effect=m.WorkerDeadline('whole worker expired')),patch.object(m.time,'sleep')as sleep:
   with self.assertRaises(m.WorkerDeadline):session.ready(1)
   sleep.assert_not_called()
 def test_budget_rejects_unbounded(self):
  for n in[0,-1,1201]:
   with self.assertRaises(ValueError):m.Session([],Path('unused'),{},n)
if __name__=='__main__':unittest.main()
