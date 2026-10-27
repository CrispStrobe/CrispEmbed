#!/usr/bin/env python3
"""Small stdlib controls; no tokenizer payload/compiler/model imports."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import scifact_tokenizer_diagnostic as d

class Controls(unittest.TestCase):
    def states(self):
        run=dict(id=d.RUN,head_sha=d.BASE,path='.github/workflows/ovisocr2-metadata.yml',status='completed',conclusion='cancelled',event='workflow_dispatch')
        jobs=[]
        for name,conclusion in [(f'original-shard-{i}','success') for i in range(8)]+[('freeze-original','success')]+[(f'native-shard-{i}','failure' if i<2 else 'cancelled') for i in range(8)]:
            index=int(name[-1]) if name.startswith('native-shard-') else -1
            jobs.append(dict(id=d.NEGATIVES[index]['job'] if index in (0,1) else len(jobs)+1,name='diagnostic / '+name,status='completed',conclusion=conclusion,steps=[] if conclusion=='cancelled' else [{}]))
        return run,dict(total_count=len(jobs),jobs=jobs)

    def test_preserved_execution_and_six_unexecuted_cancellations(self):
        proof=d.check_jobs(*self.states());self.assertEqual(len(proof['native_cancelled_before_steps']),6)
        self.assertEqual([r['id'] for r in proof['native_failures']],[121891185945,121891185906])

    def test_cancelled_after_step_rejected(self):
        run,jobs=self.states();jobs['jobs'][-1]['steps']=[{}]
        with self.assertRaises(ValueError):d.check_jobs(run,jobs)

    def test_failed_original_or_changed_source_rejected(self):
        run,jobs=self.states();jobs['jobs'][0]['conclusion']='failure'
        with self.assertRaises(ValueError):d.check_jobs(run,jobs)
        run,jobs=self.states();run['head_sha']='0'*40
        with self.assertRaises(ValueError):d.check_jobs(run,jobs)

    def test_missing_duplicate_or_replaced_negative_rejected(self):
        for mutate in ('missing','duplicate','replaced'):
            run,jobs=self.states()
            if mutate=='missing':jobs['jobs'].pop(0)
            elif mutate=='duplicate':jobs['jobs'].append(copy.deepcopy(jobs['jobs'][0]))
            else:jobs['jobs'][9]['id']=99
            with self.assertRaises(ValueError):d.check_jobs(run,jobs)

    def test_bytelevel_whitespace_unicode_roundtrip_and_piece_spans(self):
        decoder=d.byte_decoder();self.assertEqual(set(decoder.values()),set(range(256)))
        encoder={v:k for k,v in decoder.items()}
        text=' café\n';pieces=['<|endoftext|>']+[encoder[b] for b in text.encode()]
        detail=d.describe_ids(list(range(len(pieces))),pieces,text)
        self.assertEqual(detail[-1]['derived_byte_span'],[len(text.encode())-1,len(text.encode())])
        self.assertEqual(b''.join(bytes.fromhex(p['raw_hex']) for p in detail[1:]),text.encode())
        self.assertEqual(d.codepoints('é')[0]['byte_span'],[0,2])

    def test_wrong_piece_and_incomplete_input_rejected(self):
        for text in ('b','aa'):
            with self.assertRaises(ValueError):d.describe_ids([0,1],['<|endoftext|>','a'],text)
        with self.assertRaises(ValueError):d.piece_bytes('👋')

    def test_boundary_observation_preserves_unicode_bytes(self):
        text='a—b';original=d.pretoken_spans([b'a','—'.encode(),b'b'],text)
        native=d.pretoken_spans([text.encode()],text)
        self.assertNotEqual(original,native);self.assertEqual(original[1]['derived_byte_span'],[1,4])
        with self.assertRaises(ValueError):d.pretoken_spans([b'a'],text)
        with self.assertRaises(ValueError):d.pretoken_spans([b'bad'],text)

    def test_header_overlay_only_accessor(self):
        text=subprocess.check_output(['git','show',d.BASE+':src/tokenizer.h'],cwd=ROOT,text=True)
        observed=d.expose_pretokens(text)
        inserted='    // Test executable only: expose the unchanged const split for observation.\n    std::vector<std::string> diagnostic_gpt2_pretokens(const std::string & text) const { return gpt2_pretokenize(text); }\n'
        self.assertEqual(observed.replace(inserted,''),text)
        with self.assertRaises(ValueError):d.expose_pretokens('wrong source')

    def test_packet_literal_utf8_and_detail_scope(self):
        data=d.packet(['a'],['a b'],[{'id':'d:21676556','input':'é\n'}])
        self.assertTrue(data.startswith(b'SCITOK1\0'));self.assertIn('é\n'.encode(),data)
        self.assertEqual(data[-4:],b'\1\0\0\0')

    def test_exact_manifest_and_frozen_runtime_sources(self):
        spec=json.loads((ROOT/'docs/scifact-tokenizer-diagnostic.json').read_text());d.validate_manifest(spec)
        for name,sha in spec['diagnostic_sources'].items():self.assertEqual(d.digest((ROOT/name).read_bytes()),sha,name)
        for name,sha in spec['runtime_sources'].items():self.assertEqual(d.digest(subprocess.check_output(['git','show',d.BASE+':'+name],cwd=ROOT)),sha,name)
        for section in ('runtime_sources','diagnostic_sources'):
            bad=copy.deepcopy(spec);bad[section].pop(next(iter(bad[section])))
            with self.assertRaises(ValueError):d.validate_manifest(bad)

    def test_model_free_workflow_and_gate_controls(self):
        workflow=(ROOT/'.github/workflows/scifact-tokenizer-diagnostic.yml').read_text()
        self.assertIn('fetch-depth: 0',workflow);self.assertIn('--no-deps',workflow);self.assertIn('--require-hashes',workflow)
        self.assertNotIn('cmake',workflow);self.assertNotIn('torch==',workflow)
        cpp=(ROOT/'tests/test_scifact_tokenizer_diagnostic.cpp').read_text()
        self.assertLess(cpp.index('core_env::on('),cpp.index('std::ifstream'))
        self.assertIn('core_util::clean_exit',cpp)
        driver=(ROOT/'tools/ci-heavy/scifact_tokenizer_diagnostic.py').read_text()
        self.assertLess(driver.index('all_frozen_ids_masks_full_hashes_exact'),driver.index("subprocess.run(['c++'"))
        self.assertIn("native['ids']!=old['actual']",driver)

if __name__=='__main__':unittest.main()
