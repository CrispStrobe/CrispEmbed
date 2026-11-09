import copy
import importlib.util
import json
import unittest
from pathlib import Path

s = importlib.util.spec_from_file_location('prepare', Path(__file__).with_name('lighton4b_prepare.py'))
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


class Admission(unittest.TestCase):
    def test_frozen_protocol(self):
        m.admit(json.loads(m.PROTOCOL.read_text()))

    def test_mutations_fail(self):
        cfg = json.loads(m.PROTOCOL.read_text())
        for key, value in [('max_workers', 2), ('max_new_tokens', 2048),
                           ('replacement_cases', 1), ('arm_seconds', 0)]:
            c = copy.deepcopy(cfg)
            c['quota'][key] = value
            with self.assertRaises(ValueError):
                m.admit(c)
        c = copy.deepcopy(cfg)
        c['cases'].reverse()
        with self.assertRaises(ValueError):
            m.admit(c)

    def test_capacity_worst_case(self):
        m.budget(512, 256)
        for prompt, image in [(513, 256), (256, 257), (128, 256), (True, 1), (0, 0)]:
            with self.assertRaises(ValueError):
                m.budget(prompt, image)

    def test_archive_rejected_before_parse(self):
        import tempfile
        cfg = json.loads(m.PROTOCOL.read_text())
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'prepared.zip'
            p.write_bytes(b'not an authenticated archive')
            with self.assertRaisesRegex(ValueError, 'immutable prepared archive'):
                m.prepare(cfg, p)


if __name__ == '__main__':
    unittest.main()
