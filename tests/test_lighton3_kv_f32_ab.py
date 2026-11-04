#!/usr/bin/env python3
"""Synthetic retained ZIP/prefill byte barriers; no network or model work."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('kv_worker',ROOT/'tools/ci-heavy/lighton3_kv_f32_ab.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)


class Controls(unittest.TestCase):
    def test_owned_source_closure(self):
        protocol=json.loads(worker.PROTOCOL.read_text())
        self.assertEqual(protocol['schema'],'crispembed.lighton3.kv-f32-ab.v1')
        self.assertEqual(protocol['original_artifact'],worker.TARGET)
        for name,digest in protocol['owned_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)

    def test_closed_prefix_selection(self):
        for name in ('references/form-transcription/reference.json',
                     'form-transcription/f32/gate1/forced/logits_step6.f32',
                     'form-transcription/f16/gate1/plain/decoded.txt'):
            self.assertTrue(worker.selected(name),name)
        for name in ('references/form-grounding/reference.json','form-transcription/f32/gate0/forced/a.f32',
                     'form-transcription/q8/gate1/forced/a.f32','form-transcription/f32/gate1/unknown/a.f32',
                     'references/form-transcription/model.gguf','extra/references/form-transcription/a.npy'):
            self.assertFalse(worker.selected(name),name)

    def test_safe_extraction_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'input.zip'
            name='references/form-transcription/reference.json'
            with zipfile.ZipFile(path,'w') as z:
                z.writestr(name,b'{}');z.writestr('references/fox-transcription/reference.json',b'ignored')
            rows=worker.extract_inputs(path,root/'selected')
            self.assertEqual(rows,[dict(path=name,bytes=2,sha256=hashlib.sha256(b'{}').hexdigest())])
            self.assertEqual((root/'selected'/name).read_bytes(),b'{}')
            with self.assertRaises(ValueError):worker.extract_inputs(path,root/'selected')

    def test_unsafe_zip_rejected_before_selected_extraction(self):
        for name in ('../escape.json','/absolute.json','references/form-transcription/../escape.json'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);path=root/'bad.zip'
                with zipfile.ZipFile(path,'w') as z:z.writestr(name,b'bad')
                with self.assertRaises(ValueError):worker.extract_inputs(path,root/'selected')

    def dirs(self,root):
        a,b=root/'a',root/'b';a.mkdir();b.mkdir()
        for directory in (a,b):
            for name in ('pixel_values.f32','vis_ln_pre.f32','logits_step0.f32','logits_step6.f32'):
                (directory/name).write_bytes(b'1234')
        return a,b

    def test_prefill_exact_decode_changes_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=self.dirs(Path(directory));(b/'logits_step6.f32').write_bytes(b'5678')
            (b/'logits_step7.f32').write_bytes(b'5678')
            result=worker.prefill_control(a,b)
            self.assertTrue(result['all_bytes_exact']);self.assertEqual(result['tensor_files'],3)

    def test_prefill_changes_rejected(self):
        for mutation in ('bytes','missing','extra','missing-step0'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as directory:
                a,b=self.dirs(Path(directory))
                if mutation=='bytes':(b/'vis_ln_pre.f32').write_bytes(b'5678')
                elif mutation=='missing':(b/'vis_ln_pre.f32').unlink()
                elif mutation=='extra':(b/'new_stage.f32').write_bytes(b'1234')
                else:
                    (a/'logits_step0.f32').unlink();(b/'logits_step0.f32').unlink()
                with self.assertRaises(ValueError):worker.prefill_control(a,b)


if __name__=='__main__':unittest.main()
