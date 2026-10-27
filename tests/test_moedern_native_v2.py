"""Stdlib source/lineage controls. No model, build, arrays or framework access."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'))
import contract_v2 as c
import diff_adapter as adapter


class Controls(unittest.TestCase):
    def fixture(self):
        cfg=c.protocol();rows=[]
        for i in range(250):rows.append(dict(case='q-de',stage=f'owned.retained.{i}',path=f'owned/{i}.f32',dtype='f32',shape=[1],bytes=4,sha256='1'*64))
        primary=dict(completed=True,github_sha=cfg['primary']['source'],schema='crispembed.moedern-colbert.original.v1',variant='1.5M',
            revision=c.read_contract()['variants']['1.5M']['original']['revision'],source_contract_sha256=cfg['historical_sources']['docs/moedern-colbert-source-contract.json'],
            panel_sha256=cfg['historical_sources']['tests/fixtures/moedern-colbert/panel-v1.json'],versions={'torch':'2.8.0+cpu','transformers':'5.3.0','sentence-transformers':'6.1.0','numpy':'2.2.6'},pipeline=['Transformer','Dense','MultiVectorMask','Normalize'],
            tokenizer_descriptor={'owned':'metadata'},checkpoint_files={'owned':'metadata'},inventory={'owned':'metadata'},loaded_tensor_sha256={'owned':'metadata'},
            driver_sources=cfg['original_driver_sources']['primary'],cases=copy.deepcopy(cfg['case_contracts']),stage_arrays=rows,
            scores=[dict(query=q,document=d,original_maxsim=float(i),explicit_maxsim=float(i)) for i,(q,d) in enumerate(c.panel()['score_pairs'])])
        supplemental=copy.deepcopy(primary);supplemental.update(completed=False,github_sha=cfg['supplemental']['source'])
        supplemental['driver_sources']=cfg['original_driver_sources']['supplemental']
        supplemental['stage_arrays']+=copy.deepcopy(cfg['supplemental_extra_descriptors'])
        supplemental['stage_arrays'][0]['sha256']='2'*64;supplemental['scores'][0]['original_maxsim']+=1e-6
        return primary,supplemental

    def test_rejected_observer_never_promoted_or_substituted(self):
        primary,supplemental=self.fixture();before=copy.deepcopy(primary)
        result=c.check_reports(primary,supplemental)
        self.assertEqual(primary,before);self.assertFalse(result['accepted_original_continuity']);self.assertFalse(result['primary_replaced'])
        self.assertEqual((result['primary_arrays'],result['supplemental_only_arrays']),(250,21))
        for change in ('completed','missing','overlap','shape','weights','ids','neutrality','scores'):
            a,b=copy.deepcopy(primary),copy.deepcopy(supplemental)
            if change=='completed':b['completed']=True
            if change=='missing':b['stage_arrays'].pop()
            if change=='overlap':b['stage_arrays'][-1]=copy.deepcopy(a['stage_arrays'][0])
            if change=='shape':b['stage_arrays'][0]['shape']=[1,1]
            if change=='weights':b['loaded_tensor_sha256']={'changed':'weights'}
            if change=='ids':b['cases'][0]['token_ids'][2]+=1
            if change=='neutrality':b['cases'][0]['ordinary_observer_bit_exact']=False
            if change=='scores':b['scores']=copy.deepcopy(a['scores']);b['stage_arrays'][:250]=copy.deepcopy(a['stage_arrays'])
            with self.assertRaises(ValueError,msg=change):c.check_reports(a,b)

    def test_supplied_effective_scoring_masks_and_expansion_are_distinct(self):
        cfg=c.protocol();q=next(x for x in cfg['case_contracts'] if x['id']=='q-de');d=next(x for x in cfg['case_contracts'] if x['id']=='d-de')
        self.assertNotEqual(q['supplied_attention_mask'],q['effective_attention_mask'])
        self.assertNotEqual(d['effective_attention_mask'],d['scoring_mask'])
        self.assertEqual(len(cfg['supplemental_extra_descriptors']),21)
        self.assertEqual(sum(r['dtype']=='f32' for r in cfg['supplemental_extra_descriptors']),6)
        a,b=self.fixture();c.check_reports(a,b)
        bad=copy.deepcopy(b);row=next(x for x in bad['stage_arrays'] if x['stage']=='pipeline.1.output#0.attention_mask')
        row['sha256']=hashlib.sha256(struct.pack('<'+'q'*32,*q['effective_attention_mask'])).hexdigest()
        with self.assertRaises(ValueError):c.check_reports(a,bad)

    def test_native_math_and_probe_control_bytes_exact79(self):
        cfg=c.protocol()
        for path,pin in cfg['copied_native_sources'].items():
            target=ROOT/path.replace('tests/moedern-native/','tests/moedern-native-v2/')
            self.assertEqual(c.digest(target),pin)
            self.assertEqual(target.read_bytes(),c.git_blob(cfg['native_blueprint'],path))
        self.assertEqual(cfg['ggml'],'a06fc0bb46b73cc10011fc90ae4054aee8438aeb')
        old=c.git_blob(cfg['native_blueprint'],'docs/moedern-colbert-native-contract-v1.json')
        self.assertFalse(json.loads(old)['reviewed_original']['reviewed'])
        self.assertFalse((ROOT/'docs/moedern-colbert-native-contract-v1.json').exists())

    def test_coverage_is_explicit_and_no_missing_native_required_stage(self):
        cfg=c.protocol();coverage=cfg['coverage']
        self.assertEqual(coverage['mapped_primary_arrays'],112);self.assertEqual(len(coverage['retained_not_mapped']),138)
        self.assertEqual(112+138,250)
        self.assertEqual(len({(x['case'],x['stage']) for x in coverage['retained_not_mapped']}),138)
        self.assertTrue(all(x['reason'] for x in coverage['retained_not_mapped']))
        self.assertEqual(set(coverage['required_native_stages']),c.native_names());self.assertEqual(len(c.native_names()),55)
        self.assertNotIn('dense',c.primary_map());self.assertNotIn('normalized',c.primary_map())
        self.assertEqual(set(coverage['native_without_reference']),{'layer.0.attn.probabilities','layer.1.attn.probabilities','layer.0.ffn.routed','layer.1.ffn.routed'})
        self.assertEqual(coverage['supplemental_native_map']['pipeline.1.output#0.token_embeddings'],'dense')
        driver=(ROOT/'tools/ci-heavy/moedern_native_v2.py').read_text()
        self.assertIn('set(observed)!=c.native_names()',driver)

    def test_exact_old_schema_materializes_without_inference_worker(self):
        with tempfile.TemporaryDirectory() as d:
            module=c.schema_module(Path(d));self.assertEqual(module.read_contract(),c.read_contract())
            self.assertEqual(len(module.tensor_schema(module.read_contract()['variants']['1.5M']['config'])),32)
            actual={str(p.relative_to(d)) for p in Path(d).rglob('*') if p.is_file()}
            self.assertEqual(actual,set(c.protocol()['historical_sources']))
            self.assertFalse(any('reference.py' in p or 'native_contract.py' in p for p in actual))
        self.assertTrue(all(n not in sys.modules for n in ('numpy','torch','transformers','gguf')))

    def test_lossless_adapter_rejects_changed_shape_hash_and_structural_exit(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'owned.f32';raw=struct.pack('<6f',1,0,0,1,1,0);p.write_bytes(raw)
            pin=dict(path=p,shape=[1,3,2],sha256=hashlib.sha256(raw).hexdigest())
            result=adapter.write_pair(root/'old.gguf',root/'new.gguf',[dict(name='primary.owned',original=pin,native=pin)])
            self.assertEqual(result['original']['tensors']['primary.owned']['shape'],[1,3,2])
            with self.assertRaises(ValueError):adapter.write_pair(root/'x.gguf',root/'y.gguf',[dict(name='wrong',original=pin,native={**pin,'shape':[3,2]})])
            with self.assertRaises(ValueError):adapter.collect(2,root/'absent',{})
            p.write_bytes(raw[:-1]+b'\xff')
            with self.assertRaises(ValueError):adapter.source_bytes(pin)

    def test_dispatcher_union_and_original_first_barrier(self):
        cfg=c.protocol();old=c.git_blob(cfg['base_source'],'.github/workflows/ovisocr2-metadata.yml').decode();new=(ROOT/'.github/workflows/ovisocr2-metadata.yml').read_text()
        def options(s):return s.split('options: [',1)[1].split(']',1)[0].split(', ')
        self.assertEqual(set(options(new))-set(options(old)),{'moedern-native-v2'});self.assertTrue(set(options(old))<=set(options(new)))
        driver=(ROOT/'tools/ci-heavy/moedern_native_v2.py').read_text()
        self.assertLess(driver.index('load_retained(scratch)',driver.index('def execute(')),driver.index('import numpy as np',driver.index('def execute(')))
        self.assertIn("'ordinary.features.token_embeddings' if label=='all_vectors'",driver)
        self.assertIn("for target in reports['primary']['scores']",driver)
        self.assertIn('shared_rows=sum',driver);self.assertIn("value['shared_rows']!=250",driver)
        self.assertIn('for case in reports[\'primary\'][\'cases\']',driver)
        self.assertEqual(cfg['gates']['max_absolute'],.002);self.assertEqual(cfg['gates']['min_row_cosine'],.99999)

if __name__=='__main__':unittest.main()
