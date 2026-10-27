"""Small source/lineage/identity controls; no model, corpus or optional imports."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import ultrabert_scifact_native_v3 as contract
from ultrabert_scifact import rank_scores, quality
spec=importlib.util.spec_from_file_location('native_v3_driver',ROOT/'tools/ci-heavy/ultrabert_scifact_native_v3.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class Controls(unittest.TestCase):
    def test_protocol_keeps_old_budgets_and_separate_authorization(self):
        p=contract.protocol();old=json.loads((ROOT/'tests/regression/ultrabert/scifact-protocol-v2.json').read_text())
        self.assertEqual(p['acceptance'],old['acceptance']);self.assertEqual(p['licenses'],old['licenses'])
        self.assertFalse(p['native_authorization']['old_producer_embeddings_authorized'])
        self.assertEqual(p['execution']['original_jobs'],0)
        self.assertEqual(p['execution']['max_parallel'],2)
        self.assertTrue(p['mask_contract']['BOS0_attended'])
        self.assertIn('DERIVED',p['mask_contract']['provenance'])

    def token_fixture(self):
        return dict(source=contract.protocol()['token_proof']['source'],completed=True,tokenizer_parity=True,errors=[],
                    embeddings_authorized=False,new_embeddings_executed=False,model_weights_accessed=False,
                    all_original_ids_masks_fullhashes_exact=True,authored_cases_original_exact=14,
                    modes=[dict(mode=k,passed=True,rows=5483,clipped_mismatches=[],full_mismatches=[]) for k in ('unset','empty','0','1')])

    def test_token_proof_rejects_partial_and_relabelled_evidence(self):
        good=self.token_fixture();contract.check_token_proof(good)
        for change in (lambda x:x.update(embeddings_authorized=True),lambda x:x.update(source='0'*40),
                       lambda x:x['modes'][-1].update(rows=5482),lambda x:x['modes'][-1]['clipped_mismatches'].append('d:1'),
                       lambda x:x.update(authored_cases_original_exact=13)):
            bad=copy.deepcopy(good);change(bad)
            with self.assertRaises(ValueError):contract.check_token_proof(bad)

    def test_exact_ids_and_derived_mask_are_not_observed_mask(self):
        frozen=dict(ids=[0,1,9],attention_mask=[1,1,1])
        self.assertEqual(contract.check_actual_ids('q:1',[0,1,9],frozen)['mask_provenance'],'DERIVED')
        for actual,expected in (([0,1],frozen),([0,9,1],frozen),([0,1,9],dict(ids=[0,1,9],attention_mask=[0,1,1]))):
            with self.assertRaises(ValueError):contract.check_actual_ids('q:1',actual,expected)

    def dataset_fixture(self):
        rows=[dict(id=('q:' if i<300 else 'd:')+str(i),text='duplicate text allowed') for i in range(5483)]
        return dict(queries=rows[:300],documents=rows[300:]),[dict(id=r['id'],ids=[0,1],attention_mask=[1,1]) for r in rows]

    def test_complete_ids_preserves_duplicate_text_rejects_drop_reorder_or_mask(self):
        data,tokens=self.dataset_fixture();self.assertEqual(len(contract.check_rows(data,tokens)),5483)
        for change in (lambda d,t:t.pop(),lambda d,t:t.reverse(),lambda d,t:t[0].update(attention_mask=[0,1]),
                       lambda d,t:d['documents'][0].update(id=d['documents'][1]['id'])):
            d,t=copy.deepcopy((data,tokens));change(d,t)
            with self.assertRaises(ValueError):contract.check_rows(d,t)

    def test_zip_rejects_traversal_duplicates_and_symlink(self):
        for names in (['../outside'],['/outside'],['a','a'],['a\\b'],['a//b'],['a/./b']):
            f=io.BytesIO()
            with zipfile.ZipFile(f,'w') as z:
                for n in names:z.writestr(n,b'x')
            with zipfile.ZipFile(f) as z, self.assertRaises(ValueError):contract.safe_zip(z)
        f=io.BytesIO()
        with zipfile.ZipFile(f,'w') as z:
            n=zipfile.ZipInfo('link');n.create_system=3;n.external_attr=(0o120777<<16);z.writestr(n,'target')
        with zipfile.ZipFile(f) as z,self.assertRaises(ValueError):contract.safe_zip(z)

    def test_native_receipt_requires_whole_preflight_and_gate_scope(self):
        p=contract.protocol();r=dict(schema='crispembed-scifact-native-shard-v3',status='completed native shard',shard=0,
             rows=['q:1'],data_freeze_sha256=p['prepared']['data_freeze_sha256'],all_actual_ids_preflight_exact=True,
             gates=p['gates'],model_sha256=p['native_model_sha256'],mask_contract=p['mask_contract'])
        contract.check_shard(r,'native',0,['q:1'])
        for key,value in [('all_actual_ids_preflight_exact',False),('data_freeze_sha256','0'*64),('gates',{}),('rows',[])]:
            bad=copy.deepcopy(r);bad[key]=value
            with self.assertRaises(ValueError):contract.check_shard(bad,'native',0,['q:1'])

    def test_source_guard_rejects_changed_runtime_before_any_payload(self):
        p=contract.protocol()
        def fake(*args):
            if args[0]=='rev-parse' and args[1].startswith(p['runtime_base']+':'):return 'old'
            if args[0]=='rev-parse' and args[1].startswith('HEAD:'):return 'changed'
            return ''
        with patch.object(driver,'git',side_effect=fake),self.assertRaisesRegex(ValueError,'closure'):
            driver.source()

    def test_same_run_payload_requires_successful_actual_job(self):
        meta=dict(head_sha='a'*40,path='.github/workflows/ovisocr2-metadata.yml',event='workflow_dispatch',run_attempt=1)
        with patch.object(driver,'api',side_effect=[meta,dict(total_count=1,jobs=[dict(name='native-shard-0',conclusion='failure',status='completed')])]),self.assertRaisesRegex(ValueError,'job'):
            driver.fetch_artifact(1,'a'*40,dict(name='unused'),Path('/never-created'),complete=False,job='native-shard-0')

    def test_parent_cancelled_state_is_not_upgraded_by_successful_originals(self):
        import scifact_tokenizer_diagnostic as history
        jobs=[]
        for i in range(8):jobs.append(dict(id=i,name='original-shard-'+str(i),status='completed',conclusion='success',steps=[{}]))
        jobs.append(dict(id=8,name='freeze-original',status='completed',conclusion='success',steps=[{}]))
        for i in range(8):jobs.append(dict(id=history.NEGATIVES[i]['job'] if i<2 else 100+i,name='native-shard-'+str(i),status='completed',conclusion='failure' if i<2 else 'cancelled',steps=[{}] if i<2 else []))
        run=dict(id=history.RUN,head_sha=history.BASE,path='.github/workflows/ovisocr2-metadata.yml',status='completed',conclusion='cancelled',event='workflow_dispatch')
        result=history.check_jobs(run,dict(total_count=len(jobs),jobs=jobs));self.assertEqual(len(result['native_failures']),2)
        jobs[-1]['steps']=[{}]
        with self.assertRaises(ValueError):history.check_jobs(run,dict(total_count=len(jobs),jobs=jobs))

    def test_runtime_composition_retains_truncation_and_exact_token_source(self):
        p=contract.protocol()
        for file,h in p['token_proof']['copied_runtime_sha256'].items():self.assertEqual(contract.sha(ROOT/file),h)
        old=subprocess.check_output(['git','show',p['assembly_base']+':src/ultrabert_profile.h'],cwd=ROOT).decode()
        new=(ROOT/'src/ultrabert_profile.h').read_text()
        self.assertEqual(old[old.index('inline bool tokenize('):],new[new.index('inline bool tokenize('):])
        self.assertIn('configure_tokenizer_mode(raw)',new)

    def test_ranking_lexical_tie_and_quality_rejection(self):
        before=rank_scores(['d:2','d:10'],[.5,.5],['d:10']);self.assertEqual(before['top1'],'d:10')
        changed=rank_scores(['d:2','d:10'],[.6,.5],['d:10'])
        self.assertFalse(quality([before],[changed])['passed'])

    def test_workflow_barrier_and_no_historical_guard_injection(self):
        text=(ROOT/'.github/workflows/ultrabert-scifact-native-v3.yml').read_text()
        self.assertIn('needs: build',text);self.assertIn('max-parallel: 2',text);self.assertIn('fail-fast: false',text)
        self.assertNotIn('SentenceTransformer',text);self.assertNotIn('--phase original',text)
        tree=ast.parse((ROOT/'tools/ci-heavy/ultrabert_scifact_native_v3.py').read_text())
        forbidden={'source','reuse','prepared','native_shard','aggregate','original_freeze'}
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='old' and n.func.attr in forbidden for n in ast.walk(tree)))


if __name__=='__main__':unittest.main()
