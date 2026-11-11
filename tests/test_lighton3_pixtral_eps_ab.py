#!/usr/bin/env python3
"""Small retained-input and gate0 byte-replay controls, without model work."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/ci-heavy'))
spec=importlib.util.spec_from_file_location('eps_ab',ROOT/'tools/ci-heavy/lighton3_pixtral_eps_ab.py')
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
sys.path.insert(0,str(ROOT/'tools'))
import frozen_guard_amendments as amendments


class Controls(unittest.TestCase):
    def test_owned_sources_are_exact_and_only_native_norm_files_differ(self):
        p=json.loads(worker.PROTOCOL.read_text())
        for name,digest in p['owned_sha256'].items():
            # Frozen digest, or the newest reviewed behaviour-neutral re-pin
            # (tests/results/frozen-guards/misc.json); still an exact match.
            digest=amendments.expected('lighton3-pixtral-eps-ab',name,digest)
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)
        self.assertEqual(p['first_phase_native_requests'],12)
        self.assertEqual(p['remaining_phase_new_native_requests'],60)
        self.assertEqual(p['original_forward_calls'],0)

    def traces(self,root):
        a,b=root/'old',root/'new';a.mkdir();b.mkdir()
        for p in (a,b):
            (p/'trace.json').write_text(json.dumps({'seconds':1 if p==a else 2,'generated_ids':[3,4],'observed':True}))
            (p/'decoded.txt').write_text('text')
            (p/'stage.f32').write_bytes(b'1234')
        return a,b

    def test_only_elapsed_time_can_differ_in_gate0_trace(self):
        with tempfile.TemporaryDirectory() as t:
            a,b=self.traces(Path(t));self.assertTrue(worker.trace_control(a,b)['all_bytes_exact'])
            d=json.loads((b/'trace.json').read_text());d['generated_ids']=[3,5];(b/'trace.json').write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'trace/IDs'):
                worker.trace_control(a,b)

    def test_tensor_byte_inventory_and_decoded_changes_rejected(self):
        for kind in ('bytes','inventory','text'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as t:
                a,b=self.traces(Path(t))
                if kind=='bytes':(b/'stage.f32').write_bytes(b'1235')
                elif kind=='inventory':(b/'new.f32').write_bytes(b'1234')
                else:(b/'decoded.txt').write_text('other')
                with self.assertRaises(ValueError):worker.trace_control(a,b)


if __name__=='__main__': unittest.main()
