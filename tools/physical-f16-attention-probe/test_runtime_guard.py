import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
sys.path.insert(0, str(Path(__file__).parent))
from runtime_guard import Deadline, WorkerDeadline, command


class RuntimeGuardTests(unittest.TestCase):
    def test_outer_deadline_interrupts_noncommand_work(self):
        with self.assertRaises(WorkerDeadline):
            with Deadline(.05):
                time.sleep(5)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL), (0., 0.))

    def test_child_tree_cleanup_and_receipts(self):
        with tempfile.TemporaryDirectory() as folder:
            pidfile = Path(folder) / 'child.pid'
            code = 'import subprocess,time,pathlib; p=subprocess.Popen(["sleep","30"]); pathlib.Path(' + repr(str(pidfile)) + ').write_text(str(p.pid)); time.sleep(30)'
            receipt = {}
            with self.assertRaises(subprocess.TimeoutExpired):
                command([sys.executable, '-c', code], Path(folder) / 'log', 2, receipt)
            child = int(pidfile.read_text())
            status = Path(f'/proc/{child}/status')
            if status.exists():
                self.assertIn('State:\tZ', status.read_text())
            self.assertEqual(receipt['error_type'], 'TimeoutExpired')
            self.assertIsInstance(receipt['returncode'], int)
            self.assertGreater(receipt['elapsed_seconds'], 0)
            self.assertIn('cumulative_child_peak_rss_bytes', receipt)

    def test_alarm_also_cleans_command_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = {}
            with self.assertRaises(WorkerDeadline):
                with Deadline(.1):
                    command(['sleep', '30'], Path(folder) / 'log', 20, receipt)
            self.assertEqual(receipt['error_type'], 'WorkerDeadline')
            self.assertLess(receipt['returncode'], 0)

    def test_environment_and_callback_failure_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = {}
            command([sys.executable, '-c', 'import os; print(os.environ["TASK_TEST_VALUE"])'],
                    Path(folder) / 'env', 2, receipt, env={**os.environ, 'TASK_TEST_VALUE': 'bounded'})
            self.assertEqual((Path(folder) / 'env').read_text().strip(), 'bounded')
            def broken():
                raise ValueError('callback original cause')
            receipt = {}
            with self.assertRaisesRegex(ValueError, 'callback original cause'):
                command(['sleep', '30'], Path(folder) / 'callback', 10, receipt, on_tick=broken)
            self.assertEqual(receipt['error_type'], 'ValueError')
            self.assertLess(receipt['returncode'], 0)

    def test_nonzero_and_success(self):
        with tempfile.TemporaryDirectory() as folder:
            receipt = {}
            command([sys.executable, '-c', 'pass'], Path(folder) / 'ok', 2, receipt)
            self.assertEqual(receipt['status'], 'completed')
            receipt = {}
            with self.assertRaises(subprocess.CalledProcessError):
                command([sys.executable, '-c', 'raise SystemExit(7)'], Path(folder) / 'bad', 2, receipt)
            self.assertEqual(receipt['returncode'], 7)


if __name__ == '__main__':
    unittest.main()
