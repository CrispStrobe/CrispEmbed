import copy
import importlib.util
import json
from pathlib import Path
import unittest
ROOT = Path(__file__).resolve().parents[2]
s = importlib.util.spec_from_file_location('composed', ROOT/'tools/kaggle/lighton4b_composed_vision.py')
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)


class Controls(unittest.TestCase):
    def test_protocol_and_mutations(self):
        cfg = json.loads((ROOT/'docs/lighton4b-composed-vision-v1.json').read_text());m.admit(cfg)
        for key,value in [('cap',256),('threads',8),('worker_seconds',0),('arm_seconds',0),('quality_qualified',True),('cases',list(reversed(m.CASES)))]:
            c=copy.deepcopy(cfg);c[key]=value
            with self.assertRaises(RuntimeError):m.admit(c)
        c=copy.deepcopy(cfg);c['receipts']['form']='0'*64
        with self.assertRaises(RuntimeError):m.admit(c)

    def test_real_cli_arity(self):
        from pathlib import Path
        rgb=Path('/case/input.rgb')
        for arm,argc in [('composed',11),('source-features',13)]:
            args=m.native_args('binary','candidate',rgb,10,10,'teacher','out',arm)
            self.assertEqual(len(args),argc)
            self.assertEqual(args[-1],'--observed-composed-cap512')
            self.assertEqual(args[7:10],[512,4,'greedy-full'])
            self.assertEqual('--upstream-vision'in args,arm=='source-features')
        with self.assertRaises(RuntimeError):m.native_args('', '', rgb,10,10,'','','unknown')

    def test_test_only_driver_extension(self):
        source=(ROOT/'tests/lighton4b_native_document.cpp').read_text()
        self.assertIn('ids.size() + size_t(cap) > 2048',source)
        self.assertIn('--extended-table-cap512',source)
        self.assertIn('extended_cap && (cap != 512 || threads != 4 || mode != "greedy-full")',source)


if __name__=='__main__':unittest.main()
