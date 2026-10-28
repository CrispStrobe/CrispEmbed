#!/usr/bin/env python3
"""Tiny stdlib children only; no compiler, package, model or inference."""
import importlib.util
from pathlib import Path
import signal
import sys
import tempfile
import unittest
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ci_process',ROOT/'tools/ci_process.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@unittest.skipUnless(sys.platform.startswith('linux'),'wait4 units are Linux KiB')
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def child(self,code,*,timeout=3,name='child'):
        return m.run([sys.executable,'-I','-c',code],self.root/(name+'.log'),timeout=timeout)

    def test_success_and_actual_per_child_rss_units(self):
        high=self.child('x=bytearray(24*1024*1024); print("ok",flush=True)',name='high')
        low=self.child('print("small",flush=True)',name='low')
        self.assertEqual((high['exit'],low['exit']),(0,0));self.assertFalse(high['timed_out'])
        self.assertGreater(high['maximum_rss_bytes'],low['maximum_rss_bytes']+8*2**20)
        for row in (high,low):
            self.assertEqual(row['maximum_rss_bytes'],row['maximum_rss_kib']*1024)
            self.assertGreater(row['maximum_rss_bytes'],0);self.assertTrue(row['reaped']);self.assertGreater(row['seconds'],0)
            with self.assertRaises(ChildProcessError):m.os.wait4(row['direct_child_pid'],m.os.WNOHANG)

    def test_nonzero_and_signal_exit(self):
        normal=self.child('raise SystemExit(7)',name='exit7')
        killed=self.child('import os,signal; os.kill(os.getpid(),signal.SIGTERM)',name='signal')
        self.assertEqual(normal['exit'],7);self.assertIsNone(normal['termination_signal']);self.assertFalse(normal['timed_out'])
        self.assertEqual(killed['exit'],-signal.SIGTERM);self.assertEqual(killed['termination_signal'],signal.SIGTERM)
        self.assertFalse(killed['timed_out'])

    def test_timeout_kills_process_group_and_reaps_child(self):
        descendant_code='import sys,time;print("startup diagnostic",file=sys.stderr,flush=True);time.sleep(20)'
        parent_code=('import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-I","-c",'+repr(descendant_code)+']);'
                     'print("DESCENDANT_PID="+str(p.pid),flush=True);time.sleep(20)')
        with patch.object(m.subprocess.Popen,'poll',side_effect=AssertionError('poll reaps resources')),patch.object(m.subprocess.Popen,'wait',side_effect=AssertionError('wait reaps resources')):
            row=self.child(parent_code,timeout=.7,name='timeout')
        self.assertTrue(row['timed_out']);self.assertEqual(row['exit'],-signal.SIGKILL)
        self.assertEqual(row['termination_signal'],signal.SIGKILL);self.assertTrue(row['reaped']);self.assertLess(row['seconds'],2)
        # Provider sitecustomize diagnostics can arrive on the merged stream
        # after the parent PID record. They are not a process identity.
        lines=(self.root/'timeout.log').read_text().splitlines()
        records=[line.removeprefix('DESCENDANT_PID=')for line in lines if line.startswith('DESCENDANT_PID=')]
        self.assertEqual(len(records),1);descendant=int(records[0])
        self.assertIn('startup diagnostic',lines)
        stat=Path('/proc')/str(descendant)/'stat'
        deadline=time.monotonic()+.3
        while stat.exists()and stat.read_text().split()[2]not in('Z','X')and time.monotonic()<deadline:time.sleep(.01)
        if stat.exists():self.assertIn(stat.read_text().split()[2],('Z','X'))
        with self.assertRaises(ChildProcessError):m.os.wait4(row['direct_child_pid'],m.os.WNOHANG)

    def test_timeout_validation_and_separate_stdout(self):
        for value in (0,-1,float('inf'),float('nan'),True,90000):
            with self.assertRaises(ValueError):self.child('pass',timeout=value)
        out=self.root/'stdout';err=self.root/'stderr'
        row=m.run([sys.executable,'-I','-c','import sys;print("out");print("err",file=sys.stderr)'],err,stdout_path=out)
        self.assertEqual(row['exit'],0);self.assertEqual(out.read_text(),'out\n');self.assertIn('err\n',err.read_text())

    def test_interrupted_parent_wait_still_kills_and_reaps(self):
        started=[];original=m.subprocess.Popen
        def launch(*args,**kwargs):
            child=original(*args,**kwargs);started.append(child.pid);return child
        with patch.object(m.subprocess,'Popen',side_effect=launch),patch.object(m.time,'sleep',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):self.child('import time;time.sleep(20)',name='interrupt')
        self.assertEqual(len(started),1)
        with self.assertRaises(ChildProcessError):m.os.wait4(started[0],m.os.WNOHANG)


if __name__=='__main__':unittest.main()
