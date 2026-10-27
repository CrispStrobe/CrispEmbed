#!/usr/bin/env python3
"""Source/stdlib controls only; synthetic IDs, no tokenizer/compiler/network."""
import bisect
import copy
import json
from pathlib import Path
import re
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import ultrabert_unicode_pretoken as c
from scifact_tokenizer_diagnostic import BASE,canonical,digest
class Controls(unittest.TestCase):
    def test_old_gpt2_and_clip_bodies_unchanged(self):
        old=subprocess.check_output(['git','show',BASE+':src/tokenizer_bpe.cpp'],cwd=ROOT,text=True)
        new=(ROOT/'src/tokenizer_bpe.cpp').read_text()
        name='std::vector<std::string> BPETokenizer::gpt2_pretokenize'
        end='embed_tokens BPETokenizer::encode'
        body=new[new.index(name):new.index(end)]
        body=body.replace('    if (ultrabert_unicode_gpt2_) return ultrabert_unicode_gpt2_pretokenize(text);\n','')
        self.assertEqual(body,old[old.index(name):old.index(end)])
        start='std::vector<std::string> BPETokenizer::clip_pretokenize'
        end='// Byte-level encode one CLIP pre-token'
        self.assertEqual(new[new.index(start):new.index(end)],old[old.index(start):old.index(end)])
    def test_only_model_profile_reads_value_parsed_gate(self):
        header=(ROOT/'src/ultrabert_tokenizer_mode.h').read_text();source=(ROOT/'src/tokenizer_bpe.cpp').read_text()
        self.assertIn('core_env::on("CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN")',header)
        candidate=source.split('BPETokenizer::ultrabert_unicode_gpt2_pretokenize',1)[1].split('// GPT-2 ByteLevel regex pre-tokenizer',1)[0]
        self.assertNotIn('getenv',candidate);self.assertNotIn('CRISPEMBED_ULTRABERT_UNICODE_PRETOKEN',source)
        profile=(ROOT/'src/ultrabert_profile.h').read_text();self.assertIn('configure_tokenizer_mode(raw)',profile)
        self.assertIn('ultrabert_unicode_gpt2_ = false;',source)
    def test_existing_unicode_table_selected_exact_classes(self):
        text=(ROOT/'src/core/unicode_categ.h').read_text();table=text.split('static const range k_ranges[] = {',1)[1].split('};',1)[0]
        rows=[(int(a,16),int(b,16),cat) for a,b,cat in re.findall(r'\{ 0x([0-9A-F]+), 0x([0-9A-F]+), (CAT_\w+) \}',table)]
        self.assertGreater(len(rows),1000);starts=[a for a,_,_ in rows]
        def category(cp):
            i=bisect.bisect_right(starts,cp)-1
            return rows[i][2] if i>=0 and cp<=rows[i][1] else 'CAT_LO'
        for cp,cat in [(0x2018,'CAT_P'),(0x201C,'CAT_P'),(0x301,'CAT_M'),(0x661,'CAT_N'),(0x2163,'CAT_N'),(0x2003,'CAT_WS'),(0xA0,'CAT_WS'),(0x200D,'CAT_C'),(0x378,'CAT_C')]:self.assertEqual(category(cp),cat)
        self.assertIn('cp >= 0x09 && cp <= 0x0D',text)
    def test_registered_caller_preserves_every_old_phase_and_job(self):
        name='.github/workflows/ovisocr2-metadata.yml'
        previous=subprocess.check_output(['git','show','33c50803c537b679add2f68248752900a2205561:'+name],cwd=ROOT,text=True)
        current=(ROOT/name).read_text()
        addition="\n  ultrabert-unicode-pretoken:\n    if: github.event_name == 'workflow_dispatch' && inputs.phase == 'ultrabert-unicode-pretoken'\n    needs: controls\n    uses: ./.github/workflows/ultrabert-unicode-pretoken.yml\n"
        self.assertEqual(current.count(addition),1)
        self.assertEqual(current.replace(addition,'').replace('options: [ultrabert-unicode-pretoken, ','options: [',1),previous)
        driver=(ROOT/'tools/ci-heavy/ultrabert_unicode_pretoken.py').read_text()
        self.assertIn('str(contract.CONTRACT_PATH.relative_to(ROOT))',driver)
        self.assertIn("*measured],cwd=ROOT)",driver)

    def test_owned_cases_preserve_literal_bytes_and_coverage(self):
        cases=json.loads(c.CASES_PATH.read_text());self.assertEqual(len(cases),14)
        self.assertEqual(len({x['id'] for x in cases}),14)
        for case in cases:self.assertEqual(''.join(case['pretokens']),case['text'])
        self.assertIn('ascii-not-white-space',{x['id'] for x in cases})
        probe=(ROOT/'tests/test_ultrabert_unicode_pretoken.cpp').read_text()
        self.assertIn('context A0/B1/A0 isolation failed',probe);self.assertIn('generic reload inherited UltraBERT mode',probe)
        self.assertIn('core_util::clean_exit',probe)
    def synthetic(self):
        tokens=[];legacy=[];actual=[]
        for i in range(5483):
            gold=[0]+[10]*(512 if i==24 else 1);old=gold.copy()
            if i<24:old[1]=11
            if i==24:old[-1]=11
            token=dict(id=f'owned:{i}',ids=gold[:512],attention_mask=[1]*min(512,len(gold)),untruncated_tokens=len(gold),untruncated_ids_sha256=digest(canonical(gold)))
            row=dict(row=i,id=token['id'],observer_bit_exact=True,ids=old[:512],untruncated_ids=old)
            legacy.append(row);actual.append(dict(row,attention_mask=[1]*len(row['ids'])));tokens.append(token)
        baseline=dict(clipped_ids=[f'owned:{i}' for i in range(24)],full_ids=[f'owned:{i}' for i in range(25)])
        return tokens,legacy,actual,baseline
    def test_gate0_keeps_exact24_25_negative_sets_and_all_legacy_ids(self):
        tokens,legacy,actual,baseline=self.synthetic()
        result=c.compare_rows(actual,tokens,legacy,baseline,'0');self.assertEqual(len(result['clipped_mismatches']),24);self.assertEqual(len(result['full_mismatches']),25)
        actual[0]['untruncated_ids']=[0,12];actual[0]['ids']=[0,12]
        with self.assertRaisesRegex(ValueError,'actual frozen native'):c.compare_rows(actual,tokens,legacy,baseline,'0')
    def test_gate1_requires_every_clipped_mask_and_full_hash(self):
        tokens,legacy,actual,baseline=self.synthetic()
        for row,t in zip(actual,tokens):
            row['ids']=t['ids'];row['untruncated_ids']=[0]+[10]*(t['untruncated_tokens']-1);row['attention_mask']=t['attention_mask']
        self.assertTrue(c.compare_rows(actual,tokens,legacy,baseline,'1')['original_full_lengths_hashes_exact'])
        actual[24]['untruncated_ids'][-1]=11
        with self.assertRaises(c.ParityError) as context:c.compare_rows(actual,tokens,legacy,baseline,'1')
        self.assertEqual(context.exception.details['full_mismatches'],['owned:24'])
        self.assertEqual(context.exception.details['clipped_mismatches'],[])
        actual[0]['attention_mask']=[0,1]
        with self.assertRaises(ValueError):c.compare_rows(actual,tokens,legacy,baseline,'1')
    def test_baseline_and_historical_source_guards_remain_frozen(self):
        baseline=json.loads(c.BASELINE_PATH.read_text());self.assertEqual(len(baseline['clipped_ids']),24);self.assertEqual(len(baseline['full_ids']),25)
        cfg=json.loads(c.CONTRACT_PATH.read_text());self.assertEqual(set(cfg['candidate_sources']),set(c.SOURCES))
        for name,sha in {**cfg['candidate_sources'],**cfg['frozen_sources']}.items():self.assertEqual(digest((ROOT/name).read_bytes()),sha,name)
        for name in ('tools/ci-heavy/scifact_tokenizer_diagnostic.py','docs/scifact-tokenizer-diagnostic.json','tests/test_scifact_tokenizer_diagnostic.cpp'):
            self.assertEqual((ROOT/name).read_bytes(),subprocess.check_output(['git','show',baseline['source']+':'+name],cwd=ROOT))
if __name__=='__main__':unittest.main()
