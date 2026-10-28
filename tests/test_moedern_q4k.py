#!/usr/bin/env python3
"""Bounded stdlib-only plan and source controls; no model/framework imports."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import copy
import tempfile
import types
import contextlib
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools/moedern-colbert'))
import contract
import quantize_q4k_padded as q
import q4k_contract as guard


class Controls(unittest.TestCase):
    def test_exact_inventory_real_q4k_all_three_sizes(self):
        for variant, p in q.PROFILES.items():
            rows = []
            for row in contract.tensor_schema(contract.read_contract()['variants'][variant]['config']):
                shape = row['shape']
                if row['transform'] == 'expert_transpose': shape = [shape[0], shape[2], shape[1]]
                if row['transform'] == 'depthwise_transpose': shape = [shape[2], shape[0]]
                rows.append(q.plan(row['target'], list(reversed(shape))))
            self.assertEqual(len(rows), p['tensors'])
            self.assertEqual(sum(__import__('math').prod(r['logical_shape']) for r in rows), p['parameters'])
            self.assertEqual(sum(r['type'] == 'Q4_K' for r in rows), p['q4k'])
            self.assertLess(sum(r['bytes'] for r in rows), sum(r['original_bytes'] for r in rows))
            for row in rows:
                self.assertEqual(row['logical_shape'][1:], row['storage_shape'][1:])
                if row['type'] == 'Q4_K':
                    self.assertEqual(row['storage_shape'][0] % 256, 0)
                    self.assertGreater(row['bytes_saved'], 0)
                else:
                    self.assertEqual(row['logical_shape'], row['storage_shape'])

    def test_padding_is_columns_only_and_narrow_has_no_false_compression(self):
        for logical, stored in [(128,256), (640,768), (1024,1024)]:
            r = q.plan('blk.0.ffn_up_exps.weight', [logical, 128, 4])
            self.assertEqual(r['storage_shape'], [stored,128,4])
            self.assertEqual(r['bytes'], stored//256*144*128*4)
        self.assertEqual(q.plan('blk.0.attn_q.weight', [16,128])['type'], 'F32')

    def test_critical_names_remain_bit_preserved_f32(self):
        for name in ['token_embd.hash.weight','token_embd.proj.weight','dense_head.weight',
                     'blk.0.ffn_gate_inp.weight','blk.0.conv_dw.weight','blk.0.attn_norm.weight']:
            r = q.plan(name, [640,1024])
            self.assertEqual(r['type'], 'F32')
            self.assertEqual(r['bytes_saved'], 0)

    def test_malformed_shape_rejected(self):
        for shape in [[], [0,128], [-1,128], [128]*5, [30003,128], [128.5,128], [True,128]]:
            with self.assertRaises(ValueError): q.plan('blk.0.attn_q.weight', shape)

    def test_host_barrier_precedes_payload_and_frameworks(self):
        from unittest.mock import patch
        with patch.dict(q.os.environ, {}, clear=True), patch.object(q.Path,'is_dir',return_value=False):
            with self.assertRaisesRegex(ValueError, 'hosted'): q.quantize('/missing','/missing','1.5M','/missing')
        self.assertNotIn('numpy', sys.modules)
        self.assertNotIn('gguf', sys.modules)

    def test_real_type_and_no_fallback_readback(self):
        text = (ROOT/'tools/moedern-colbert/quantize_q4k_padded.py').read_text()
        for required in ['GGMLQuantizationType.Q4_K', "'(fallback '", "t.tensor_type==expected", "critical F32 payload changed", "quantized_publication_qualified"]:
            self.assertIn(required, text)

    def test_old_loader_remains_strict_and_new_flags_value_parsed(self):
        old = (ROOT/'src/moedern_packed_profile.h').read_text()
        self.assertNotIn('GGML_TYPE_Q4_K', old)
        engine = (ROOT/'src/moedern_family_engine.cpp').read_text()
        self.assertIn('return init_impl(path, threads, failure, false)', engine)
        self.assertIn('return init_impl(path, threads, failure, true)', engine)
        self.assertIn('moedern_q4k::pad_input(g, input, columns.logical, columns.stored)', engine)
        self.assertIn('mm_id(b + "ffn_down_exps.weight", act, chosen)', engine)
        self.assertNotIn('getenv("CRISPEMBED_MOEDERN_Q4_K_PADDED', engine)
        self.assertIn('core_env::on("CRISPEMBED_MOEDERN_Q4_K_PADDED")', engine)

    def test_hosted_profile_and_authored_graph_controls(self):
        cmake = (ROOT/'tests/moedern-q4k/CMakeLists.txt').read_text()
        self.assertIn('set(GGML_LLAMAFILE ON', cmake)
        controls = (ROOT/'tests/moedern-q4k/controls.cpp').read_text()
        for token in ['ggml_quantize_chunk', 'ggml_mul_mat_id', 'moedern_q4k::pad_input', 'core_util::clean_exit']:
            self.assertIn(token, controls)

    def test_complete_source_closure_required(self):
        cfg=json.loads(guard.PATH.read_text())
        guard.validate(cfg)
        for name in ['src/core/env_gate.h','src/moedern_packed_profile.h',*guard.UNCHANGED_REQUIRED]:
            bad=copy.deepcopy(cfg);bad['unchanged_sha256'].pop(name)
            with self.assertRaisesRegex(ValueError,'closure'): guard.validate(bad)
        bad=copy.deepcopy(cfg);bad['runtime_owned'].remove('src/moedern_q4k_profile.h')
        with self.assertRaisesRegex(ValueError,'closure'): guard.validate(bad)

    def test_runtime_pins_match_publications(self):
        header=(ROOT/'src/moedern_q4k_profile.h').read_text()
        cfg=json.loads(guard.PATH.read_text())
        for variant in q.PROFILES:
            p=q.publication(variant)
            self.assertIn(p['sha256'],header)
            self.assertIn(cfg['originals'][variant]['report_sha256'],header)
        self.assertFalse(cfg['retrieval_quality']['reviewed'])

    def driver(self):
        spec=importlib.util.spec_from_file_location('moedern_q4k_driver',ROOT/'tools/ci-heavy/moedern_q4k_padded.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

    def test_driver_rejects_local_before_transfer_or_build(self):
        m=self.driver()
        with patch.dict(m.os.environ,{},clear=True),patch.object(m.Path,'is_dir',return_value=False),patch.object(m.subprocess,'run')as run:
            with self.assertRaisesRegex(ValueError,'hosted'):m.build_only('/missing','/missing')
            with self.assertRaisesRegex(ValueError,'hosted'):m.run_panel('/missing','/missing','1.5M','/missing','/missing')
            with self.assertRaisesRegex(ValueError,'hosted'):m.stream_file('https://example.invalid','/missing',{})
            run.assert_not_called()

    def test_original_wrong_hash_rejected_before_arrays(self):
        m=self.driver()
        with tempfile.TemporaryDirectory()as directory:
            root=Path(directory);(root/'review').mkdir();(root/'review/original.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'original report differs'):
                m.original_inputs(root,'27M',{'originals':{'27M':{'report_sha256':'0'*64}}})

    def test_explicit_kaggle_version_on_metadata_and_every_output_page(self):
        m=self.driver();calls=[];events=[];code='COMMIT="'+'f'*40+'"'
        class Request: pass
        def get(request):
            calls.append(('get',request));return json.dumps({'blob':{'source':code}})
        def listing(request):
            calls.append(('list',request))
            if hasattr(request,'page_token'):
                self.assertEqual(request.page_token,'next')
                return types.SimpleNamespace(files=[],next_page_token=None)
            return types.SimpleNamespace(files=[types.SimpleNamespace(file_name='moedern-original/review/original.json',url='test-report')],next_page_token='next')
        client=types.SimpleNamespace(kernels=types.SimpleNamespace(kernels_api_client=types.SimpleNamespace(get_kernel=get,list_kernel_session_output=listing)))
        class Api:
            kernels_pull=types.SimpleNamespace(__globals__={'ApiGetKernelRequest':Request})
            kernels_output=types.SimpleNamespace(__globals__={'ApiListKernelSessionOutputRequest':Request})
            def authenticate(self): pass
            def build_kaggle_client(self): return contextlib.nullcontext(client)
        module=types.ModuleType('kaggle.api.kaggle_api_extended');module.KaggleApi=Api
        modules={name:types.ModuleType(name) for name in ['kaggle','kaggle.api']};modules[module.__name__]=module
        def stream(url,path,pin):
            events.append(url);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({'stage_arrays':[]}) if url=='test-report' else 'test-model')
        cfg={'originals':{'27M':dict(kernel_slug='crispembed-moedern-27m-original',version_label='v3',
             source='f'*40,script_sha256=guard.sha(code.encode()),report_bytes=20,report_sha256='a'*64)}}
        with tempfile.TemporaryDirectory()as directory,patch.dict(sys.modules,modules),patch.dict(m.os.environ,{'KAGGLE_ACCOUNT':'owned-test'},clear=True),patch.object(m,'hosted'),patch.object(m,'stream_file',side_effect=stream),patch.object(m,'original_inputs',side_effect=lambda *a:events.append('verified-all20')),patch.object(m.quantizer,'publication',return_value={'url':'test-model'}):
            m.prepare('27M',Path(directory)/'inputs',cfg)
        self.assertEqual([name for name,_ in calls],['get','list','list'])
        self.assertTrue(all(request.version_label=='v3' and request.kernel_slug==cfg['originals']['27M']['kernel_slug'] for _,request in calls))
        self.assertEqual(events,['test-report','verified-all20','test-model'])

    def test_replay_cannot_publish_or_weaken_numeric_gates(self):
        text=(ROOT/'tools/ci-heavy/moedern_q4k_padded.py').read_text()
        self.assertIn('reference.actual_ref',text)
        self.assertIn("require(control['passed']",text)
        self.assertIn('FAST family F32 control violates original gates',text)
        self.assertIn('publication_qualified=False',text)
        self.assertIn("'q4k-repeat'",text)
        self.assertIn("source=closure.git('rev-parse','HEAD')",text)


if __name__ == '__main__': unittest.main()
