"""Stdlib source/negative controls; no model, NumPy or native library on this host."""
import importlib.util
import io
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
    def test_hosted_source_does_not_spoof_github_on_kaggle(self):
        source='b'*40
        with mock.patch.object(w.subprocess,'check_output',return_value=source+'\n'),mock.patch.object(w.Path,'is_dir',return_value=True):
            with mock.patch.dict(w.os.environ,dict(KAGGLE_KERNEL_RUN_TYPE='Batch',CRISPEMBED_CAPTURE_SOURCE=source),clear=True):
                self.assertEqual(w.hosted_source(),(source,'Kaggle CPU'))
                w.os.environ['CRISPEMBED_CAPTURE_SOURCE']='c'*40
                with self.assertRaisesRegex(ValueError,'submitted source'):w.hosted_source()
            with mock.patch.dict(w.os.environ,dict(GITHUB_ACTIONS='true',GITHUB_SHA=source),clear=True):
                self.assertEqual(w.hosted_source(),(source,'GitHub Actions'))
    def test_signed_storage_request_never_receives_api_credential(self):
        redirect=w.urllib.error.HTTPError('https://api.github.com/artifact',302,'redirect',{'Location':'https://artifact.example/signed'},None)
        opener=mock.Mock();opener.open.side_effect=redirect
        with tempfile.TemporaryDirectory()as folder,mock.patch.dict(w.os.environ,dict(GH_TOKEN='dummy'),clear=True),mock.patch.object(w.urllib.request,'build_opener',return_value=opener),mock.patch.object(w.urllib.request,'urlopen',return_value=io.BytesIO(b'zip'))as fetch:
            path=Path(folder)/'archive.zip';w.download_archive(1,path,3)
            self.assertEqual(path.read_bytes(),b'zip')
            self.assertEqual(opener.open.call_args[0][0].get_header('Authorization'),'Bearer dummy')
            self.assertEqual(fetch.call_args[0],('https://artifact.example/signed',))
            fetch.return_value=io.BytesIO(b'oversized')
            with self.assertRaisesRegex(ValueError,'immutable size'):w.download_archive(1,Path(folder)/'large.zip',3)
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
