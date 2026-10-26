#!/usr/bin/env python3
"""Model-free negative controls for pending original/native source preparation."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/moedern-colbert'))
from native_contract import dependencies,protocol,skiplist_ids


class Controls(unittest.TestCase):
    def test_pending_original_blocks_without_model_or_framework(self):
        with self.assertRaisesRegex(ValueError,'Original review pending'):
            dependencies()

    def test_reviewed_boolean_cannot_replace_real_pins(self):
        cfg=copy.deepcopy(protocol());cfg['reviewed_original']['reviewed']=True
        with self.assertRaisesRegex(ValueError,'run/source'):
            dependencies(cfg)
        cfg['reviewed_original']['run']=1
        with self.assertRaisesRegex(ValueError,'digest missing'):
            dependencies(cfg)

    def test_complete_synthetic_pins_and_each_tamper(self):
        cfg=copy.deepcopy(protocol());p=cfg['reviewed_original'];p.update(reviewed=True,run=1,
            receipt_sha256='1'*64,gguf_sha256='2'*64,conversion_receipt_sha256='3'*64)
        for name in ('review_artifact','raw_artifact'):
            p[name]={'id':2,'bytes':1024,'sha256':'4'*64}
        self.assertEqual(dependencies(cfg)['run'],1)
        for change in [lambda x:x.update(source='0'*40),lambda x:x.update(receipt_sha256=None),
                       lambda x:x['raw_artifact'].update(bytes=1024**4),lambda x:x['review_artifact'].update(id=True),
                       lambda x:x['raw_artifact'].update(sha256='not a digest')]:
            altered=copy.deepcopy(cfg);change(altered['reviewed_original'])
            with self.assertRaises(ValueError):dependencies(altered)

    def test_skiplist_resolves_source_vocabulary_not_output_masks(self):
        tokenizer={'model':{'vocab':{'<unk>':0,'!':1,'word':2}},'added_tokens':[{'id':9,'content':'[Q]'}]}
        self.assertEqual(skiplist_ids(tokenizer,{'skiplist_words':['!','missing','[Q]']},{'unk_token':'<unk>'}),[1,9])
        self.assertEqual(skiplist_ids({'model':{'vocab':[['<unk>',0],['!',-1]]}},
            {'skiplist_words':['!','missing','<unk>']},{'unk_token':{'content':'<unk>'}}),[0,1])
        with self.assertRaisesRegex(ValueError,'unknown token'):
            skiplist_ids(tokenizer,{'skiplist_words':['!']},{'unk_token':'not present'})

    def test_preflight_rejection_retains_structured_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'attempt'
            result=subprocess.run([sys.executable,str(ROOT/'tools/moedern-colbert/native.py'),
                '--phase','preflight','--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            receipt=json.loads((out/'native.json').read_text())
            self.assertFalse(receipt['completed']);self.assertIn('Original review pending',receipt['error'])

    def test_scopes_and_budgets_are_not_promoted(self):
        c=protocol();self.assertEqual(c['scope']['tensor_count'],32)
        self.assertEqual(c['gates']['max_absolute'],.002);self.assertEqual(c['gates']['min_row_cosine'],.99999)
        self.assertEqual(c['gates']['maxsim_max_absolute'],.002)
        self.assertFalse(any(c['qualification'].values()))
        self.assertFalse(c['scope']['tokenizer_implemented']);self.assertFalse(c['scope']['public_ABI_registered'])


if __name__=='__main__':unittest.main()
