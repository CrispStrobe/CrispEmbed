"""Small orchestration guards only; no GPU/model/framework access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('head_gpu',ROOT/'tools/kaggle/josev2b-head/worker.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class Controls(unittest.TestCase):
    def test_hardware_rejects_missing_old_and_insufficient_gpu(self):
        for text in ('','P100, 6.0, 16384, 15000, 580.1','T4, 7.5, 16384, 1000, 580.1'):
            with patch.object(m.subprocess,'check_output',return_value=text):
                with self.assertRaises(ValueError):m.hardware()
        with patch.object(m.subprocess,'check_output',return_value='T4, 7.5, 15360, 14000, 580.65.6'):
            self.assertEqual(m.hardware()[0][0],'T4')

    def frozen(self,base):
        source=base/'mount'/'nested';source.mkdir(parents=True)
        (source/'reference.json').write_bytes(b'original')
        (source/'input-freeze.json').write_bytes(b'frozen')
        (source/'arrays').mkdir();(source/'arrays/0.npy').write_bytes(b'bits')
        def row(path):return dict(file=str(path.relative_to(source)),bytes=path.stat().st_size,sha256=m.sha256(path))
        report=dict(input_freeze_sha256=m.sha256(source/'input-freeze.json'),arrays=[row(source/'arrays/0.npy')])
        artifact=SimpleNamespace(RECEIPT=m.sha256(source/'reference.json'),SOURCE='worker',safe_member=lambda p:None)
        return source,report,artifact

    def test_exact_mount_preserved_and_read_back(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);source,report,artifact=self.frozen(base)
            with patch.object(m,'reference',return_value=(report,{},{})):
                self.assertEqual(m.copy_original(base/'mount',base/'out',artifact),3)
            self.assertEqual((base/'out/arrays/0.npy').read_bytes(),b'bits')

    def test_drift_or_symlink_rejected_before_copying_payload(self):
        for symlink in (False,True):
            with tempfile.TemporaryDirectory() as d:
                base=Path(d);source,report,artifact=self.frozen(base)
                target=source/'arrays/0.npy'
                if symlink:
                    target.unlink();(base/'other').write_bytes(b'bits');target.symlink_to(base/'other')
                else:target.write_bytes(b'BITS')
                with patch.object(m,'reference',return_value=(report,{},{})):
                    with self.assertRaises(ValueError):m.copy_original(base/'mount',base/'out',artifact)
                self.assertFalse((base/'out/arrays/0.npy').exists())

    def test_ambiguous_or_missing_original_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);source,report,artifact=self.frozen(base)
            extra=source.parent/'other';extra.mkdir();(extra/'reference.json').write_bytes(b'original')
            with self.assertRaises(ValueError):m.copy_original(base/'mount',base/'out',artifact)
            (source/'reference.json').write_bytes(b'changed');(extra/'reference.json').unlink()
            with self.assertRaises(ValueError):m.copy_original(base/'mount',base/'out',artifact)

    def test_import_does_not_load_frameworks_or_network(self):
        for name in ('numpy','torch','gguf','kaggle'):self.assertNotIn(name,sys.modules)


if __name__=='__main__':unittest.main()
