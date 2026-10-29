"""Tiny source/packet controls only; no Torch, NumPy, compilation or payloads."""
import copy
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/moedern-router-topk-v1'))
import common
spec = importlib.util.spec_from_file_location('router_driver', ROOT / 'tools/ci-heavy/moedern_router_topk_v1.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

class Controls(unittest.TestCase):
    def test_exact_authored_inputs(self):
        p = common.read(common.PROFILE)
        rows = common.rows(p)
        self.assertEqual({k: len(v) for k,v in rows.items()}, {'probabilities': 11, 'logits': 11})
        for values in rows.values():
            names = {r['name'] for r in values}
            self.assertTrue({'all_equal','selected_pair_tie','permuted_tie','plus_ulp','minus_ulp','two_ulp'} <= names)
        ulp = next(r['bits'] for r in rows['probabilities'] if r['name'] == 'plus_ulp')
        self.assertEqual(ulp[1] - ulp[2], 1)

    def test_reject_nonfinite_duplicate_and_axes(self):
        p = common.read(common.PROFILE)
        for mutate in [lambda x: x['inputs']['logits'][0]['bits'].__setitem__(0,0x7fc00000),
                       lambda x: x['inputs']['logits'][0]['bits'].pop(),
                       lambda x: x['inputs']['logits'][1].__setitem__('name',x['inputs']['logits'][0]['name'])]:
            bad = copy.deepcopy(p)
            mutate(bad)
            with self.assertRaises(ValueError):
                common.rows(bad)

    def test_contract_rejects_missing_source_and_changed_budgets(self):
        cfg=common.read(common.PROFILE)
        common.validate(cfg)
        for mutate in [lambda x:x['protected_git_objects'].pop('src'),
                       lambda x:x.__setitem__('top_k',3),
                       lambda x:x.__setitem__('absolute_budget',.003),
                       lambda x:x.__setitem__('model_access',True)]:
            bad=copy.deepcopy(cfg)
            mutate(bad)
            with self.assertRaises(ValueError):
                common.validate(bad)

    def test_packet_preserves_signed_zero_and_extent(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'input'
            values = [[.5,.25,.125,.125,0.,-0.,0.,-0.]]
            common.packet(p, values)
            b = p.read_bytes()
            self.assertEqual(len(b),44)
            self.assertEqual(b[:12], b'MRTOPK1\0' + struct.pack('<I',1))
            self.assertEqual(struct.unpack_from('<I',b,12+5*4)[0],0x80000000)
            for values in [[], [[0.]*7], [[float('inf')]+[0.]*7]]:
                with self.assertRaises(ValueError):
                    common.packet(p,values)

    def test_build_profile_rejects_changed_flags(self):
        cache='\n'.join(k+':BOOL='+v for k,v in driver.CACHE.items())
        cmds=[dict(file=f, command='c++ -ffp-contract=off') for f in ['probe.cpp','diff.cpp']]
        driver.build_profile(cache,cmds)
        for c,rows in [(cache.replace('GGML_LLAMAFILE:BOOL=ON','GGML_LLAMAFILE:BOOL=OFF'),cmds),
                       (cache,[dict(file='probe.cpp',command='c++ -ffp-contract=off -ffast-math'),cmds[1]])]:
            with self.assertRaises(ValueError):
                driver.build_profile(c,rows)

    def test_wheel_lock_is_explicit_and_cpu(self):
        pins=common.read(ROOT/'tools/moedern-router-topk-v1/wheels.json')
        lock=(ROOT/'tools/moedern-router-topk-v1/requirements.txt').read_text()
        self.assertEqual(pins['torch']['version'],'2.8.0+cpu')
        self.assertIn('cp312-cp312-manylinux_2_28_x86_64',pins['torch']['url'])
        for name,pin in pins.items():
            if name!='uv':
                self.assertIn(name+' @ '+pin['url']+' --hash=sha256:'+pin['sha256'],lock)

    def test_kaggle_is_not_fake_github(self):
        path=ROOT/'tools/kaggle/moedern-router-topk-v1/kernel.py'
        code=path.read_text()
        self.assertIn("SOURCE = '@SOURCE@'",code)
        self.assertNotIn("os.environ['GITHUB_ACTIONS'] =",code)
        self.assertIn("'--provider', 'kaggle-cpu'",code)
        meta=json.loads(path.with_name('kernel-metadata.json').read_text())
        self.assertFalse(meta['enable_gpu'])
        self.assertFalse(meta['enable_tpu'])

    def test_operators_are_test_only(self):
        cmake=(ROOT/'tests/moedern-router-topk-v1/CMakeLists.txt').read_text()
        self.assertNotIn('moedern_family_engine.cpp',cmake)
        cpp=(ROOT/'tests/moedern-router-topk-v1/probe.cpp').read_text()
        self.assertIn('std::nth_element',cpp)
        self.assertIn('queue.begin() + 1',cpp)
        self.assertNotIn('stable_sort',cpp)
        self.assertIn('ggml_argsort',cpp)
        self.assertIn('ggml_backend_alloc_ctx_tensors',cpp)
        self.assertIn('core_util::clean_exit',cpp)

    def test_protocol_closure_does_not_own_production(self):
        p=common.read(common.PROFILE)
        self.assertFalse(p['model_access'])
        self.assertFalse(p['production_qualified'])
        self.assertIn('src',p['protected_git_objects'])
        self.assertEqual(p['protected_git_objects']['ggml'],common.GGML)
        self.assertFalse(any(n.startswith('src/') for n in p['owned_sha256']))
        self.assertTrue(p['owned_sha256'])
        for name,sha in p['owned_sha256'].items():
            self.assertEqual(common.sha(ROOT/name),sha,name)

if __name__=='__main__':
    unittest.main()
