"""Small process lifecycle controls; never import/run the Kaggle kernel itself."""
import ast
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

root=Path(__file__).resolve().parents[1]
source=root/'tools/kaggle/lfm2-encoder-cuda-performance/lfm2_encoder_cuda_performance.py'

class ProcessControls(unittest.TestCase):
    def runtime(self, folder):
        tree=ast.parse(source.read_text())
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('run','checked','digest')]
        env=dict(Path=Path,os=os,signal=signal,subprocess=subprocess,time=time,hashlib=hashlib,LOGS=folder,report={'commands':[]},save=lambda:None)
        exec(compile(ast.Module(body=functions,type_ignores=[]),'isolated-process-control','exec'),env)
        return env
    def test_success_and_nonzero_journals(self):
        with tempfile.TemporaryDirectory() as d:
            e=self.runtime(Path(d));entry,path=e['run']('ok',[sys.executable,'-c','print("ready")'])
            self.assertEqual(entry['exit_code'],0)
            self.assertFalse(entry['timed_out'])
            self.assertEqual(entry['log_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(RuntimeError):e['checked']('bad',[sys.executable,'-c','raise SystemExit(3)'])
            self.assertEqual(e['report']['commands'][-1]['exit_code'],3)
    def test_timeout_is_killed_and_journaled_before_raise(self):
        with tempfile.TemporaryDirectory() as d:
            e=self.runtime(Path(d))
            with self.assertRaises(subprocess.TimeoutExpired):
                e['run']('deadline',[sys.executable,'-c','import time; time.sleep(30)'],timeout=.05)
            record=e['report']['commands'][-1]
            self.assertTrue(record['timed_out'])
            self.assertEqual(record['exit_code'],-signal.SIGKILL)
            self.assertEqual(record['timeout_seconds'],.05)

if __name__=='__main__':unittest.main()
