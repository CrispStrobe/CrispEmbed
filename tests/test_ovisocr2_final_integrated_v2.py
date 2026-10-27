"""Source/stdlib controls only. Final runtime declaration stays pending."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import ovisocr2_final_integrated_v2 as c

class FinalV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value=json.loads((ROOT/'docs/ovisocr2-final-integrated-v2.json').read_text())
        # A local source-only test fixture, never the declared production base.
        cls.fixture_base=c.BASE if cls.value['runtime_closure_reviewed'] else subprocess.check_output(['git','rev-parse','e3c2718b'],cwd=ROOT,text=True).strip()
        cls.objects={p:subprocess.check_output(['git','rev-parse',cls.fixture_base+':'+p],cwd=ROOT,text=True).strip() for p in c.PROTECTED_PATHS}
        cls.declared={p:c.sha(ROOT/p) for p in c.DECLARED}
    def protocol(self):return copy.deepcopy(self.value)
    def ready(self):
        value=self.protocol();value.update(runtime_base=self.fixture_base,ggml=self.objects['ggml'],runtime_closure_reviewed=True,declared_source_sha256=self.declared,protected_git_objects=self.objects.copy())
        return value
    def ready_constants(self):return mock.patch.multiple(c,BASE=self.fixture_base,GGML=self.objects['ggml'],DECLARED=self.declared)
    def driver(self):
        spec=importlib.util.spec_from_file_location('final_v2_driver',ROOT/'tools/ci-heavy/ovisocr2_final_integrated_v2.py')
        result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
    def test_pending_source_blocks_before_git_network_or_model(self):
        driver=self.driver()
        with tempfile.TemporaryDirectory() as tmp,mock.patch.dict('os.environ',GITHUB_ACTIONS='true'),mock.patch.object(driver,'api',side_effect=AssertionError('no API')) as api,mock.patch.object(c.subprocess,'check_output',side_effect=AssertionError('no source traversal before pending')):
            with self.assertRaisesRegex(ValueError,'pending'):driver.preflight(Path(tmp)/'out',require_submodule=False)
            with self.assertRaisesRegex(ValueError,'pending'):c.source_guard(ROOT,self.protocol(),require_submodule=False)
            api.assert_not_called()
    def test_reviewed_dependencies_do_not_choose_runtime(self):
        self.assertTrue(all(d['reviewed'] and d['panel_completed'] for d in self.value['dependencies'].values()))
        self.assertIs(c.checked_protocol(self.value,require_closure=False),self.value)
        with self.assertRaisesRegex(ValueError,'pending'):c.checked_protocol(self.value)
    def test_exact_eight_grid_gate_zip_member_and_binary_pins(self):
        d=self.value['dependencies']['ovis_composed'];self.assertEqual(len(d['artifacts']),8)
        self.assertEqual({(p['case'],p['gate']) for p in d['artifacts']},{(case,g) for case in ('frozen-control','control-square','large-landscape','large-boundary') for g in (0,1)})
        self.assertEqual(set(d['reviewed_binary_sha256']),set(c.BINARIES))
        for pin in d['artifacts']:
            self.assertTrue(pin['members']);self.assertIn('ovisocr2-large-inputs.json',pin['reports'])
            self.assertTrue(set(pin['reports'])<=set(pin['members']))
        for kind in ('missing','gate','zip','member','default','strict','2048','binary','service'):
            value=self.protocol();row=value['dependencies']['ovis_composed']
            if kind=='missing':row['artifacts'].pop()
            elif kind=='gate':row['artifacts'][0]['gate']=99
            elif kind=='zip':row['artifacts'][0]['sha256']='bad'
            elif kind=='member':row['artifacts'][0]['members']={}
            elif kind=='default':row['final_default']='on'
            elif kind=='strict':row['strict_numerical_qualified']=True
            elif kind=='2048':row['input_2048_qualified']=True
            elif kind=='binary':row['reviewed_binary_sha256'].pop(c.BINARIES[0])
            else:row['run']=123
            with self.subTest(kind=kind),self.assertRaises(ValueError):c.checked_protocol(value,require_closure=False)
    def test_wasm_negative_and_v1_files_are_unchanged(self):
        old=json.loads((ROOT/'docs/ovisocr2-final-integrated-v1.json').read_text())
        self.assertEqual(self.value['dependencies']['wasm_rounding_ab'],old['dependencies']['wasm_rounding_ab'])
        for path in ('tools/ovisocr2_final_integrated_v1.py','tools/ci-heavy/ovisocr2_final_integrated_v1.py','tests/test_ovisocr2_final_integrated_v1.py','docs/ovisocr2-final-integrated-v1.json','docs/ovisocr2-final-integrated-v1.md','.github/workflows/ovisocr2-final-integrated-v1.yml'):
            self.assertEqual((ROOT/path).read_bytes(),subprocess.check_output(['git','show','e3c2718b:'+path],cwd=ROOT),path)
    def test_explicit_closed_source_and_only_one_fixture_addition(self):
        with self.ready_constants():proof=c.source_guard(ROOT,self.ready(),require_submodule=False)
        self.assertEqual(proof['fixture_inheritance']['new_fixture'],self.value['fixture_inheritance']['additions'])
        old=subprocess.check_output(['git','ls-tree','-r',c.FIXTURE_BASE,'--','tests/regression'],cwd=ROOT,text=True).splitlines()
        current=subprocess.check_output(['git','ls-tree','-r','HEAD','--','tests/regression'],cwd=ROOT,text=True).splitlines()
        self.assertEqual(set(current)-set(old),{'100644 blob '+self.value['fixture_inheritance']['additions'][c.FIXTURE_ADDITION]['git_object']+'\t'+c.FIXTURE_ADDITION})
        self.assertEqual(set(old)-set(current),set())
    def test_missing_source_or_undeclared_fixture_change_rejects(self):
        with self.ready_constants():
            bad=self.ready();bad['protected_git_objects'].pop('src')
            with self.assertRaisesRegex(ValueError,'Incomplete'):c.source_guard(ROOT,bad,require_submodule=False)
        original=c.subprocess.check_output
        for kind in ('changed','deleted','extra'):
            def git(args,**kwargs):
                value=original(args,**kwargs)
                if args[1:4]==['ls-tree','-r','HEAD']:
                    lines=value.splitlines()
                    if kind=='changed':lines[0]=lines[0].replace(' blob ',' blob 0',1)
                    elif kind=='deleted':lines.pop(0)
                    else:lines.append('100644 blob '+'a'*40+'\ttests/regression/undeclared.json')
                    return '\n'.join(lines)+'\n'
                return value
            with mock.patch.object(c.subprocess,'check_output',side_effect=git),self.assertRaisesRegex(ValueError,'Historical fixtures'):
                c.fixture_guard(ROOT,self.fixture_base,self.value['fixture_inheritance'])
    def test_untracked_production_source_rejects(self):
        path=ROOT/'src/.final-v2-owned-negative';self.assertFalse(path.exists())
        try:
            path.write_text('owned source control\n')
            with self.ready_constants(),self.assertRaisesRegex(ValueError,'Dirty'):c.source_guard(ROOT,self.ready(),require_submodule=False)
        finally:path.unlink(missing_ok=True)
    def test_auditors_and_low_level_import_closure_are_pinned(self):
        self.assertEqual(set(self.value['auditor_source_sha256']),set(c.AUDITORS))
        for path,sha in self.value['auditor_source_sha256'].items():self.assertEqual(c.sha(ROOT/path),sha,path)
        required={'tools/ci-heavy/successor_assets.py','tools/ci-heavy/ovisocr2_published_consumer.py','.gitmodules','tools/ci-heavy/ocr_quality_prepare.py','tools/ci-heavy/qwen35_quality.py','tests/test_ovisocr2_image_bounds.cpp','tests/results/ocr-quality-prepared.json'}
        self.assertTrue(required<=set(c.PROTECTED_PATHS))
    def test_exact_six_binaries_source_and_protocol_required(self):
        good=dict(source_commit='a'*40,protocol_sha256='b'*64,portable_cpu_baseline_checked=True,binary_sha256={p:'c'*64 for p in c.BINARIES})
        c.checked_binaries(good,'a'*40,'b'*64)
        for kind in ('missing','extra','sha','source','protocol'):
            bad=copy.deepcopy(good)
            if kind=='missing':bad['binary_sha256'].pop(c.BINARIES[0])
            elif kind=='extra':bad['binary_sha256']['extra']='c'*64
            elif kind=='sha':bad['binary_sha256'][c.BINARIES[0]]=False
            elif kind=='source':bad['source_commit']='d'*40
            else:bad['protocol_sha256']='d'*64
            with self.assertRaises(ValueError):c.checked_binaries(bad,'a'*40,'b'*64)
    def fixture(self):
        rows={str(i):dict(input_rgb_sha256='rgb'+str(i),input_png_sha256='png'+str(i),raw_text='text',processed_text='text',token_ids=[17,248046],stop_id=248046,stop_kind=1,finish_reason='stop',cap=32,confidences=[.5,.75],mean_confidence=.625) for i in range(11)}
        return dict(native_cases={'f16':rows},scores={'f16':{'exact':11}},f16_paired_synthetic={'screen_passed':True},f16_paired_documents={'screen_passed':True})
    def test_same_eleven_case_quality_and_confidence_policies(self):
        before=self.fixture();self.assertTrue(c.compare_quality(copy.deepcopy(before),before,0)['scope_passed'])
        changed=copy.deepcopy(before);changed['native_cases']['f16']['0']['confidences'][0]+=.00001;changed['native_cases']['f16']['0']['mean_confidence']+=.000005
        with self.assertRaisesRegex(ValueError,'Gate0'):c.compare_quality(changed,before,0)
        self.assertGreater(c.compare_quality(changed,before,1)['cases']['0']['max_selected_confidence_abs_drift'],0)
        for kind in ('id','mean','nan','score','missing'):
            changed=copy.deepcopy(before);row=changed['native_cases']['f16']['0']
            if kind=='id':row['token_ids'][0]=0
            elif kind=='mean':row['mean_confidence']=.9
            elif kind=='nan':row['confidences'][0]=float('nan')
            elif kind=='score':changed['scores']['f16']['exact']=10
            else:changed['native_cases']['f16'].pop('0')
            with self.assertRaises(ValueError):c.compare_quality(changed,before,1)
        old=ast.parse((ROOT/'tools/ovisocr2_final_integrated_v1.py').read_text());new=ast.parse((ROOT/'tools/ovisocr2_final_integrated_v2.py').read_text())
        body=lambda t:ast.dump(next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='compare_quality'),include_attributes=False)
        self.assertEqual(body(old),body(new))
    def test_failure_receipts_keep_each_phase_and_gate(self):
        driver=self.driver()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);a=driver.save_rejection(path,'preflight',None,ValueError('first'));before=a.read_bytes()
            b=driver.save_rejection(path,'quality-check',1,ValueError('later'));self.assertNotEqual(a,b);self.assertEqual(a.read_bytes(),before)
    def test_cli_requires_gate_and_submodule_bypass_is_preflight_only(self):
        for args,text in [(['--phase','frozen'],'--gate is required'),(['--phase','frozen','--gate','0','--no-submodule-check'],'preflight-only')]:
            r=subprocess.run([sys.executable,ROOT/'tools/ci-heavy/ovisocr2_final_integrated_v2.py',*args],capture_output=True,text=True)
            self.assertEqual(r.returncode,2);self.assertIn(text,r.stderr)
    def test_workflow_caps_two_models_and_declares_before_source_dependency_access(self):
        workflow=(ROOT/'.github/workflows/ovisocr2-final-integrated-v2.yml').read_text()
        self.assertEqual(workflow.count('max-parallel: 2'),2);self.assertIn('needs: [build, frozen]',workflow)
        self.assertNotIn('submodules: recursive',workflow)
        self.assertLess(workflow.index('--no-submodule-check'),workflow.index('git submodule update'))
        self.assertIn('if: always()',workflow);self.assertIn("CRISPEMBED_WASM_Q8_0_ROUND: '0'",workflow)
if __name__=='__main__':unittest.main()
