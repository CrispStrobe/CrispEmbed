"""Small source/lineage/identity controls; no model, corpus or optional imports."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import os
import tempfile
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

    def test_fresh_python_launch_has_only_existing_exact_bundle_loader_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);scratch=root/'scratch';lib=scratch/'bundle/lib'
            with self.assertRaisesRegex(ValueError,'already materialized'):
                driver.launch_plan(2,scratch,'a'*64)
            lib.mkdir(parents=True)
            script=root/'tools/ci-heavy/ultrabert_scifact_native_v3.py';script.parent.mkdir(parents=True)
            script.write_text("import json,os,sys\nprint(json.dumps(dict(pid=os.getpid(),ppid=os.getppid(),ld=os.environ.get('LD_LIBRARY_PATH'),exists=os.path.isdir(os.environ['LD_LIBRARY_PATH']),preload=os.environ.get('LD_PRELOAD'),audit=os.environ.get('LD_AUDIT'),gates={k:v for k,v in os.environ.items() if k.startswith('CRISPEMBED_')},argv=sys.argv[1:])))\n")
            with patch.object(driver,'ROOT',root),patch.dict(os.environ,{'LD_LIBRARY_PATH':'/untrusted','LD_PRELOAD':'untrusted.so','LD_AUDIT':'untrusted.so','CRISPEMBED_UNRELATED':'1'}):
                argv,env=driver.launch_plan(2,scratch,'a'*64)
                result=json.loads(subprocess.check_output(argv,env=env,text=True))
            self.assertNotEqual(result['pid'],os.getpid());self.assertEqual(result['ppid'],os.getpid())
            self.assertEqual(result['ld'],str(lib.resolve()));self.assertTrue(result['exists'])
            self.assertIsNone(result['preload']);self.assertIsNone(result['audit']);self.assertEqual(result['gates'],contract.protocol()['gates'])
            self.assertEqual(result['argv'],['--phase','native-child','--shard','2','--stage',str(scratch/'native-stage.json'),'--stage-sha','a'*64])

    def test_stage_hash_shard_source_and_materialized_tampering_reject_before_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            scratch=Path(tmp);stage=scratch/'native-stage.json';closure={'head':'a'*40}
            dirs={}
            for name in ('token-proof','freeze','prepared','bundle'):
                path=scratch/name;path.mkdir();(path/'small.txt').write_text(name);dirs[name]=driver.inventory(path)
            value=dict(schema='crispembed-scifact-native-child-stage-v1',shard=0,source=closure,directories=dirs)
            driver.write(stage,value);good_sha=contract.sha(stage)
            with patch.object(driver,'api') as api,patch.object(driver,'fetch_artifact') as download:
                with self.assertRaisesRegex(ValueError,'hash'):
                    driver.validate_stage(stage,'0'*64,0,scratch,closure)
                with self.assertRaisesRegex(ValueError,'source/shard'):
                    driver.validate_stage(stage,good_sha,1,scratch,closure)
                with self.assertRaisesRegex(ValueError,'source/shard'):
                    driver.validate_stage(stage,good_sha,0,scratch,{'head':'b'*40})
                (scratch/'prepared/small.txt').write_text('tampered')
                with self.assertRaisesRegex(ValueError,'materialized'):
                    driver.validate_stage(stage,good_sha,0,scratch,closure)
                api.assert_not_called();download.assert_not_called()

    def test_cached_archive_receipt_binds_materialized_payload_without_redownload(self):
        with tempfile.TemporaryDirectory() as tmp:
            scratch=Path(tmp);target=scratch/'bundle';target.mkdir();driver.write(target/'receipt.json',{'source':'small fixture'})
            archive=scratch/'bundle.zip'
            with zipfile.ZipFile(archive,'w') as z:z.writestr('receipt.json',(target/'receipt.json').read_bytes())
            proof=dict(run=1,source='a'*40,artifact_id=2,name='bundle',bytes=archive.stat().st_size,sha256=contract.sha(archive))
            meta=dict(head_sha='a'*40,path='.github/workflows/ovisocr2-metadata.yml',event='workflow_dispatch',run_attempt=1,status='completed',conclusion='success')
            artifact=dict(workflow_run=dict(id=1,head_sha='a'*40),name='bundle',size_in_bytes=proof['bytes'],expired=False,digest='sha256:'+proof['sha256'])
            with patch.object(driver,'api',side_effect=[meta,artifact]),patch.object(driver.subprocess,'run')as download:
                driver.check_cached_artifact(proof,target);download.assert_not_called()
            driver.write(target/'receipt.json',{'source':'substituted fixture'})
            with patch.object(driver,'api',side_effect=[meta,artifact]),patch.object(driver.subprocess,'run')as download:
                with self.assertRaisesRegex(ValueError,'archive member'):driver.check_cached_artifact(proof,target)
                download.assert_not_called()

    def test_initial_all8_loader_failures_remain_structural_and_before_vectors(self):
        report=json.loads((ROOT/'tests/results/ultrabert/initial-scifact-native-v3-loader-failure.json').read_text())
        self.assertEqual(len(report['rows']),8);self.assertEqual([r['shard']for r in report['rows']],list(range(8)))
        for row in report['rows']:
            self.assertEqual(row['producer']['source']['head'],'ee51e10e94a923a56f01b336ae2159a652c2c0da')
            self.assertEqual(row['producer']['error'],'libggml.so.0: cannot open shared object file: No such file or directory')
        self.assertFalse(report['numerical_qualified']);self.assertFalse(report['native_embeddings_executed'])

    def test_loader_manifest_exact_files_and_bytes_without_any_library_load(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);directory=root/'lib';directory.mkdir();(directory/'libcrispembed.so').write_bytes(b'owned fake library')
            manifest=root/'libraries.json';files={p.name:dict(bytes=p.stat().st_size,sha256=contract.sha(p))for p in directory.iterdir()}
            driver.write(manifest,files);digest=contract.sha(manifest)
            self.assertEqual(driver.loader.checked_manifest(directory,manifest,digest),files)
            (directory/'libcrispembed.so').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'bytes'):driver.loader.checked_manifest(directory,manifest,digest)
            (directory/'extra.so').write_bytes(b'x')
            with self.assertRaisesRegex(ValueError,'coverage'):driver.loader.checked_manifest(directory,manifest,digest)

    def test_python_library_path_is_derived_verified_and_never_inherited(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);directory=root/'bundle/lib';directory.mkdir(parents=True)
            runtime=dict(directory=str(root/'python/lib'),library='libpython3.12.so.1.0',bytes=123,sha256='a'*64,version='owned fixture')
            policy=dict(mode='bundle-and-python-lib',python_library=driver.loader.fingerprint(runtime))
            with patch.object(driver.loader,'python_library',return_value=runtime),patch.dict(os.environ,{'LD_LIBRARY_PATH':'/untrusted','LD_PRELOAD':'bad.so'}):
                env=driver.smoke_environment(directory,policy)
                self.assertEqual(env['LD_LIBRARY_PATH'],str(directory.resolve())+':'+runtime['directory']);self.assertNotIn('LD_PRELOAD',env)
                changed=dict(policy,python_library=dict(policy['python_library'],sha256='b'*64))
                with self.assertRaisesRegex(ValueError,'differs'):driver.smoke_environment(directory,changed)
            self.assertEqual(driver.loader.loader_path(directory,dict(mode='bundle-only')),str(directory.resolve()))

    def test_actual_smoke_retry_only_for_interpreter_startup_not_ggml(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);payload=root/'payload';directory=payload/'lib';directory.mkdir(parents=True);(directory/'libcrispembed.so').write_bytes(b'owned fake')
            out=root/'out';out.mkdir()
            runtime=dict(directory=str(root/'python/lib'),library='libpython3.12.so.1.0',bytes=123,sha256='a'*64,version='owned fixture')
            calls=[]
            def run(argv,**kwargs):
                calls.append((argv,kwargs['env']))
                if len(calls)==1:
                    kwargs['stdout'].write('python: error while loading shared libraries: libpython3.12.so.1.0: missing\n')
                    return type('Result',(),{'returncode':127})()
                manifest=payload/'loader-libraries.json'
                result=dict(schema='crispembed-scifact-native-loader-smoke-v1',completed=True,mode='bundle-and-python-lib',
                    python_library=driver.loader.fingerprint(runtime),library_manifest_sha256=contract.sha(manifest),
                    library_sha256=contract.sha(directory/'libcrispembed.so'),symbols=list(driver.loader.SYMBOLS),
                    all_ggml_mappings_in_bundle=True,model_files_accessed=False,framework_imported=False,forward_called=False)
                driver.write(payload/'loader-smoke.json',result)
                return type('Result',(),{'returncode':0})()
            with patch.object(driver.subprocess,'run',side_effect=run),patch.object(driver.loader,'python_library',return_value=runtime):
                result=driver.library_smoke(payload,out)
            self.assertEqual(result['mode'],'bundle-and-python-lib');self.assertEqual(len(calls),2)
            self.assertEqual(calls[0][1]['LD_LIBRARY_PATH'],str(directory.resolve()))
            self.assertEqual(calls[1][1]['LD_LIBRARY_PATH'],str(directory.resolve())+':'+runtime['directory'])
            (payload/'loader-smoke.json').unlink();calls.clear()
            def ggml_failure(argv,**kwargs):
                calls.append(argv);kwargs['stdout'].write('python: error while loading shared libraries: libggml.so.0: missing\n')
                return type('Result',(),{'returncode':127})()
            with patch.object(driver.subprocess,'run',side_effect=ggml_failure),patch.object(driver.loader,'python_library',return_value=runtime):
                with self.assertRaisesRegex(ValueError,'not Python'):driver.library_smoke(payload,out)
            self.assertEqual(len(calls),1)

    def test_build_smoke_is_mandatory_before_bundle_release_and_native_matrix(self):
        text=(ROOT/'tools/ci-heavy/ultrabert_scifact_native_v3.py').read_text();tree=ast.parse(text)
        build=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='build')
        body=ast.get_source_segment(text,build)
        self.assertLess(body.index('smoke=library_smoke(payload,out)'),body.index("write(payload / 'receipt.json'"))
        self.assertIn('loader_smoke=smoke',body)
        workflow=(ROOT/'.github/workflows/ultrabert-scifact-native-v3.yml').read_text()
        self.assertIn('Build bundle and pass actual fresh-Python C API loader smoke',workflow)
        self.assertIn('needs: build',workflow)
        helper=(ROOT/'tools/scifact_native_loader_smoke.py').read_text()
        self.assertIn('ctypes.CDLL',helper);self.assertIn('os.RTLD_NOW|os.RTLD_LOCAL',helper)
        self.assertNotIn('.gguf',helper);self.assertNotIn('CrispEmbed(',helper)

    def test_workflow_barrier_and_no_historical_guard_injection(self):
        text=(ROOT/'.github/workflows/ultrabert-scifact-native-v3.yml').read_text()
        self.assertIn('needs: build',text);self.assertIn('max-parallel: 2',text);self.assertIn('fail-fast: false',text)
        self.assertNotIn('SentenceTransformer',text);self.assertNotIn('--phase original',text)
        tree=ast.parse((ROOT/'tools/ci-heavy/ultrabert_scifact_native_v3.py').read_text())
        forbidden={'source','reuse','prepared','native_shard','aggregate','original_freeze'}
        self.assertFalse(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='old' and n.func.attr in forbidden for n in ast.walk(tree)))


if __name__=='__main__':unittest.main()
