"""Stdlib source/negative controls; no model, NumPy or native library on this host."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('q4worker',ROOT/'tools/portfolio/ultrabert_q4k.py');w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
class Controls(unittest.TestCase):
    def test_failed_producer_blocks_download(self):
        pin=dict(run=3,head='a'*40)
        with mock.patch.object(w,'api',return_value=dict(head_sha='a'*40,status='completed',conclusion='failure',run_attempt=1)),mock.patch.object(w.subprocess,'run')as process:
            with self.assertRaisesRegex(ValueError,'producer'):w.artifact(pin,Path('unused'),{})
            process.assert_not_called()
    def test_frozen_original_jobs_survive_later_native_cancellation(self):
        pin=dict(run=w.ORIGINAL_RUN,head=w.ORIGINAL_SOURCE)
        run=dict(head_sha=w.ORIGINAL_SOURCE,status='completed',conclusion='cancelled',run_attempt=1)
        jobs=[dict(id=i,name=n,status='completed',conclusion='success')for i,n in w.ORIGINAL_JOBS.items()]
        with mock.patch.object(w,'api',return_value=dict(jobs=jobs)):
            w.producer_identity(pin,run)
            for field,value in [('conclusion','cancelled'),('status','in_progress'),('name','unrelated')]:
                bad=[dict(j)for j in jobs];bad[0][field]=value
                with mock.patch.object(w,'api',return_value=dict(jobs=bad)):
                    with self.assertRaisesRegex(ValueError,'job'):w.producer_identity(pin,run)
            with mock.patch.object(w,'api',return_value=dict(jobs=jobs[:-1])):
                with self.assertRaisesRegex(ValueError,'job'):w.producer_identity(pin,run)
        wrong=dict(run);wrong['run_attempt']=2
        with self.assertRaisesRegex(ValueError,'producer'):w.producer_identity(pin,wrong)
    def test_original_profile_and_recipe_protection(self):
        self.assertEqual(w.MODEL,'dmis-lab/UltraBERT-Embed-150M')
        recipes=w.recipes();self.assertEqual(len(recipes),3)
        for flags in recipes.values():self.assertEqual(flags[:2],['--tensor-type','token_embd.weight=f32'])
        for key in ['q4k-ffn','q4k-last-four-ffn']:
            for kind in ['q','k','v','output']:self.assertIn('blk.*.attn_'+kind+'.weight=f32',recipes[key])
        self.assertNotIn('blk.20.ffn_gate.weight=f32',recipes['q4k-last-four-ffn'])
    def test_actual_abi_signatures_and_id_count(self):
        s=(ROOT/'tools/portfolio/ultrabert_q4k.py').read_text()
        self.assertIn('lib.crispembed_last_token_ids.argtypes=[ctypes.c_void_p];',s)
        self.assertIn('ctx,text,ctypes.byref(count),ctypes.byref(dim)',s)
        self.assertIn('ctx,text,ctypes.byref(dim)',s)
        self.assertIn("count.value==len(expected[row['id']]['ids'])",s)
    def test_f32_control_before_quantized_and_unchanged_ranking(self):
        s=(ROOT/'tools/portfolio/ultrabert_q4k.py').read_text()
        self.assertIn("[('f32',None),*recipes().items()]",s)
        self.assertIn("require(record['qualified'],'F32 control fails",s)
        self.assertIn('quality(before,rr)',s)
        self.assertIn('Ref/independent metric disagreement',s)
        self.assertNotIn('from transformers',s)
if __name__=='__main__':unittest.main()
