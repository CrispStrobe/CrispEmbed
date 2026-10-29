"""Tiny authored receipt mutations; no images, models, frameworks or batches."""
import copy,hashlib,importlib.util,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('scan_receipt',ROOT/'tools/kaggle/ocr-real-scans/receipt.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Controls(unittest.TestCase):
    def test_targeted_scope_preserves_exact_frozen_rows(self):
        pages=[dict(id='a',max_tokens=768,ground_truth='fixed'),dict(id='b',max_tokens=1024,ground_truth='other')]
        manifest=dict(pages=pages)
        self.assertEqual(m.select_pages(manifest),pages)
        chosen=m.select_pages(manifest,('b',));self.assertEqual(chosen,[pages[1]]);self.assertIs(chosen[0],pages[1])
        self.assertEqual(m.select_pages(manifest,('b','a')),pages)
        for bad in [('missing',),('a','a'),'a',(1,)]:
            with self.assertRaises(ValueError):m.select_pages(manifest,bad)
    def setup_receipts(self):
        checksum=hashlib.sha256(b'image').hexdigest();manifest=dict(pages=[dict(id='page',image=dict(sha256=checksum),max_tokens=10)])
        image=dict(id='page',expected_sha256=checksum,sha256=checksum,sha256_ok=True)
        row=dict(page='page',exit=0,text='<tr>hello</tr>',ids=[1,2,3],max_tokens=10,cap_hit=False)
        native=dict(commit='native',images=[image],runs=[row]);original=copy.deepcopy(native);original.update(commit='original',reference_meta=dict(completed=True,exit=0))
        return manifest,native,original
    def test_hash_or_size_rejection_before_admission(self):
        pin=dict(sha256=hashlib.sha256(b'image').hexdigest(),bytes=5)
        self.assertEqual(m.admit_image(b'image',pin),pin['sha256'])
        for data in [b'Image',b'image!']:
            with self.assertRaisesRegex(ValueError,'identity'):m.admit_image(data,pin)
    def test_false_hash_flag_or_changed_expected_hash_excluded(self):
        manifest,native,original=self.setup_receipts()
        self.assertEqual(m.paired(native,original,manifest)['exact_ids'],1)
        native['images'][0]['sha256_ok']=False
        result=m.paired(native,original,manifest);self.assertEqual(result['eligible_pairs'],0);self.assertFalse(result['complete_input_proof']);self.assertEqual(len(result['rows']),1)
        native['images'][0]['sha256_ok']=True;native['images'][0]['expected_sha256']='changed'
        self.assertEqual(m.paired(native,original,manifest)['eligible_pairs'],0)
    def test_duplicate_missing_and_decoded_vs_id_distinction(self):
        manifest,native,original=self.setup_receipts();native['runs'][0]['text']='hello'
        result=m.paired(native,original,manifest);self.assertEqual(result['exact_ids'],1);self.assertEqual(result['exact_raw_text'],0)
        native['runs'][0]['ids']=[1,2];self.assertEqual(m.paired(native,original,manifest)['rows'][0]['first_token_difference'],2)
        native['images'].append(native['images'][0])
        with self.assertRaisesRegex(ValueError,'duplicate'):m.paired(native,original,manifest)
if __name__=='__main__':unittest.main()
