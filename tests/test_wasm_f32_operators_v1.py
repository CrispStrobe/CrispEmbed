"""Small source/stdlib fixtures; no vectors, model, framework or compiler."""
import copy
import io
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile
import urllib.error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/wasm-f32-operators'))
import contract
import adapter
import hosted as provider
spec = importlib.util.spec_from_file_location('f32_driver', ROOT / 'tools/ci-heavy/wasm_f32_operators_v1.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


def fixtures():
    packets = {}
    for axis in contract.AXES:
        for name in contract.OPERATORS:
            packets[axis + '/' + name] = dict(width=384, expected_sha256='a' * 64,
                inputs={'a': {'sha256': 'b' * 64}})
    outputs = {backend: {key: dict(sha256='a' * 64, shape=[12, 384], bytes=12 * 384 * 4,
                input_hashes={'a': 'b' * 64}, repeat_bit_exact=True) for key in packets}
               for backend in ['native1', 'native2', 'wasm']}
    return packets, outputs


class OperatorContract(unittest.TestCase):
    def test_preparation_rechecks_provider_source_script_and_attempt(self):
        current=dict(provider='Kaggle CPU',source='a'*40,kernel_run_type='Batch',executed_script_sha256='b'*64)
        provider.verify_service(dict(current),current)
        for key in current:
            changed=dict(current);changed[key]='changed'
            with self.assertRaises(ValueError):provider.verify_service(changed,current)
        gha=dict(provider='GitHub Actions',source='a'*40)
        record=dict(provider='GitHub Actions',head_sha='a'*40,id=123,run_attempt=1)
        with patch.dict(os.environ,dict(GITHUB_RUN_ID='123',GITHUB_RUN_ATTEMPT='1')):
            provider.verify_service(record,gha)
            for key,value in [('provider','Kaggle CPU'),('head_sha','c'*40),('id',124),('run_attempt',2)]:
                changed=dict(record);changed[key]=value
                with self.assertRaises(ValueError):provider.verify_service(changed,gha)

    def test_genuine_provider_identity_without_github_impersonation(self):
        source='a'*40
        with patch.dict(os.environ,{},clear=True),patch.object(Path,'is_dir',return_value=True):
            with self.assertRaises(ValueError):provider.identity(source)
            with patch.dict(os.environ,dict(KAGGLE_KERNEL_RUN_TYPE='Batch',CRISPEMBED_CAPTURE_SOURCE=source,
                                           CRISPEMBED_BOOTSTRAP_SHA='b'*64)):
                self.assertEqual(provider.identity(source)['provider'],'Kaggle CPU')
                with self.assertRaises(ValueError):provider.identity('c'*40)
                with patch.dict(os.environ,{'GITHUB_ACTIONS':'true','GITHUB_SHA':'c'*40}):
                    with self.assertRaises(ValueError):provider.identity(source)
            with patch.dict(os.environ,{'GITHUB_ACTIONS':'true','GITHUB_SHA':source}):
                self.assertEqual(provider.identity(source)['provider'],'GitHub Actions')

    def test_storage_redirect_does_not_receive_api_authorization(self):
        data=b'zip bytes';row=dict(id=5,name='frozen',expired=False,size_in_bytes=len(data),
            digest='sha256:'+contract.sha(data),workflow_run=dict(id=7))
        redirect=urllib.error.HTTPError('https://api.github.com/artifact',302,'redirect',
            {'Location':'https://signed.example/artifact'},None)
        with tempfile.TemporaryDirectory()as tmp,patch.dict(os.environ,{'GH_TOKEN':'test-only'}),\
             patch.object(provider,'api',return_value=dict(total_count=1,artifacts=[row])),\
             patch.object(provider.urllib.request,'build_opener')as opener,\
             patch.object(provider.urllib.request,'urlopen',return_value=io.BytesIO(data))as storage:
            opener.return_value.open.side_effect=redirect
            result=provider.artifact(7,'frozen',Path(tmp)/'artifact.zip',1024)
            self.assertEqual(result['id'],5)
            storage.assert_called_once_with('https://signed.example/artifact',timeout=180)
            request=opener.return_value.open.call_args.args[0]
            self.assertEqual(request.get_header('Authorization'),'Bearer test-only')
            self.assertIsNone(opener.call_args.args[0].redirect_request(None,None,None,None,None,None))

    def test_exact64_own_anchors(self):
        packets, outputs = fixtures()
        proof = contract.own_anchors(packets, outputs)
        self.assertEqual(proof['exact_own_input_output_anchors'], 64)
        self.assertFalse(proof['full_model_qualified'])

    def test_own_output_input_repeat_and_missing_coverage_reject(self):
        for label, field, value in [('native1', 'sha256', 'c' * 64),
                                     ('wasm', 'input_hashes', {'a': 'c' * 64}),
                                     ('native2', 'repeat_bit_exact', False),
                                     ('wasm', 'shape', [384, 12])]:
            packets, outputs = fixtures()
            key = 'native/attention_norm' if label != 'wasm' else 'wasm1/attention_norm'
            outputs[label][key][field] = value
            with self.assertRaises(ValueError):
                contract.own_anchors(packets, outputs)
        packets, outputs = fixtures()
        outputs['native1'].pop('wasm0/flash')
        with self.assertRaises(ValueError):
            contract.own_anchors(packets, outputs)

    def test_exact_raw_member_and_soname_authentication(self):
        payload = b'old bytes'
        row = dict(bytes=len(payload), sha256=contract.sha(payload))
        binary = {'native-build/libcrispembed.so': row}
        files = {'old/input.f32': payload, 'native-build/libcrispembed.so': payload,
                 'native-build/libcrispembed.so.0': payload}
        self.assertTrue(contract.authenticate_files(files, {'old/input.f32': row}, binary)['all_sha256_exact'])
        for name, value in [('unexpected.so', payload), ('native-build/libcrispembed.so.0', b'changed')]:
            changed = dict(files)
            changed[name] = value
            with self.assertRaises(ValueError):
                contract.authenticate_files(changed, {'old/input.f32': row}, binary)
        with self.assertRaises(ValueError):
            contract.authenticate_files({}, {'old/input.f32': row}, binary)

    def test_epsilon_has_source_order_and_exact_f32_value(self):
        self.assertEqual(contract.epsilon({})['source'], 'source580 opt_f32 default1e-12f')
        self.assertEqual(contract.epsilon({'bert.layer_norm_eps': 1e-12, 'bert.attention.layer_norm_epsilon': 1e-5})['key'],
                         'bert.layer_norm_eps')
        for value in [1e-5, float('nan'), True, '1e-12']:
            with self.assertRaises((ValueError, OverflowError)):
                contract.epsilon({'bert.layer_norm_eps': value})

    def test_safe_zip_rejects_traversal_duplicate_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for index, name in enumerate(['../escape', '/absolute', 'bad\\member']):
                path = out / str(index)
                with zipfile.ZipFile(path, 'w') as archive:
                    archive.writestr(name, b'x')
                with self.assertRaises(ValueError):
                    contract.safe_zip(path, 1024)
            path = out / 'symlink'
            with zipfile.ZipFile(path, 'w') as archive:
                row = zipfile.ZipInfo('link')
                row.external_attr = 0o120777 << 16
                archive.writestr(row, b'target')
            with self.assertRaises(ValueError):
                contract.safe_zip(path, 1024)
            path = out / 'valid'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('folder/', b'')
                archive.writestr('folder/file', b'bytes')
            self.assertEqual(contract.safe_zip(path, 1024), {'folder/file': b'bytes'})
            with self.assertRaises(ValueError):
                contract.safe_zip(path, 4)

    def test_f32_adapter_preserves_bits_and_rejects_reshaping(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source = out / 'source.f32'
            source.write_bytes(struct.pack('<4f', -0., 1, 2, 3))
            row = dict(path=source, shape=[2, 2], sha256=contract.digest(source))
            report = adapter.write_gguf(out / 'input.gguf', [dict(name='stage', **row)])
            self.assertEqual(report['tensors']['stage']['sha256'], contract.digest(source))
            with self.assertRaises(ValueError):
                adapter.write_pair(out / 'a.gguf', out / 'b.gguf',
                    [dict(name='stage', original=row, native=dict(row, shape=[1, 4]))])

    def test_actual_old_new_compile_profile_is_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for label in ['native', 'wasm']:
                rows = [dict(file='/frozen/ggml-cpu/' + name, arguments=['cc', '-O3', '-fPIC', '-DGGML', '-msimd128', '-pthread'])
                        for name in ['ggml-cpu.c', 'ggml-cpu.cpp', 'vec.cpp', 'ops.cpp']]
                if label == 'wasm':
                    rows.append(dict(file='/frozen/ggml-cpu/arch/wasm/quants.c',
                                     arguments=['cc', '-O3', '-fPIC', '-DGGML', '-msimd128', '-pthread']))
                (out / ('old-' + label + '-compile.json')).write_text(json.dumps(rows))
                (out / (label + '-build')).mkdir()
                (out / (label + '-build/compile_commands.json')).write_text(json.dumps(rows))
            self.assertEqual(set(driver.compile_profile(out)), {'native', 'wasm'})
            path = out / 'wasm-build/compile_commands.json'
            rows = json.loads(path.read_text())
            rows[-1]['arguments'].append('-ffast-math')
            path.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                driver.compile_profile(out)
            evidence = json.loads((out / 'review/compile-profile-inputs.json').read_text())
            self.assertEqual(len(evidence['wasm']['original']), 5)
            self.assertIn('-ffast-math', evidence['wasm']['actual'][-1]['arithmetic_flags'])
            self.assertNotIn('-ffast-math', evidence['wasm']['original'][-1]['arithmetic_flags'])
            # PIC remains part of the strict comparison, never filtered out.
            rows = json.loads((out / 'old-wasm-compile.json').read_text())
            rows[0]['arguments'].remove('-fPIC')
            path.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):
                driver.compile_profile(out)

    def test_frozen_wasm_selector_and_actual_architecture_controls(self):
        source, out = Path('/frozen/source'), Path('/output')
        wasm = driver.configuration(out, source, 'wasm')
        native = driver.configuration(out, source, 'native')
        self.assertEqual(wasm[0], 'emcmake')
        self.assertIn('-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm', wasm)
        self.assertNotIn('-DEMSCRIPTEN_SYSTEM_PROCESSOR=wasm', native)
        with self.assertRaises(ValueError):
            driver.configuration(out, source, 'generic')
        rows = [dict(file='/frozen/ggml-cpu/' + name, arguments=['cc', '-O3', '-pthread', '-msimd128'])
                for name in ['ggml-cpu.c', 'ggml-cpu.cpp', 'vec.cpp', 'ops.cpp', 'arch/wasm/quants.c']]
        self.assertEqual(len(driver.wasm_sources(rows)), 5)
        with self.assertRaises(ValueError):
            driver.wasm_sources(rows[:-1])  # Actual initial build omission.
        for flag in ['-msimd128', '-pthread']:
            changed = copy.deepcopy(rows)
            changed[-1]['arguments'].remove(flag)
            with self.assertRaises(ValueError):
                driver.wasm_sources(changed)
        changed = copy.deepcopy(rows)
        changed[0]['arguments'].append('-DGGML_CPU_GENERIC')
        with self.assertRaises(ValueError):
            driver.wasm_sources(changed)
        with self.assertRaises(ValueError):
            driver.wasm_sources(rows + [rows[-1]])

    def test_frozen_pic_and_compiler_evidence_precede_rejection(self):
        cmake = (ROOT / 'tests/wasm-f32-operators/CMakeLists.txt').read_text()
        self.assertIn('set(CMAKE_POSITION_INDEPENDENT_CODE ON)', cmake)
        self.assertLess(cmake.index('set(CMAKE_POSITION_INDEPENDENT_CODE ON)'), cmake.index('add_subdirectory('))
        source = (ROOT / 'tools/ci-heavy/wasm_f32_operators_v1.py').read_text()
        build = source[source.index('def build(out):'):source.index('def verified_build(out):')]
        self.assertLess(build.index('shutil.copy2('), build.index("label + '-build', 1800"))
        self.assertLess(build.index('shutil.copy2('), build.index('compile_profile(out)'))

    def test_prepare_fails_before_network_outside_matching_hosted_source(self):
        with patch.dict(os.environ, {'GITHUB_ACTIONS': 'false'}), patch.object(driver, 'api') as api:
            with self.assertRaises(ValueError):
                driver.hosted()
            api.assert_not_called()

    def test_dirty_protocol_is_part_of_source_guard(self):
        with patch.dict(os.environ, {'GITHUB_ACTIONS': 'true', 'GITHUB_SHA': 'a' * 40}), \
             patch.object(driver, 'git', side_effect=lambda root, *args:
                          'a' * 40 if args == ('rev-parse', 'HEAD') else ' M docs/wasm-f32-operators-v1.json'):
            with self.assertRaisesRegex(ValueError, 'Dirty'):
                driver.hosted()

    def test_sixteen_boundaries_exclude_all_old_quantized_projections(self):
        self.assertEqual(len(contract.OPERATORS), 16)
        self.assertEqual({row[0] for row in contract.OPERATORS.values()}, {0, 1, 2, 3, 4})
        self.assertTrue(all(not name.endswith('.weight') for name in contract.OPERATORS))
        source = (ROOT / 'tests/wasm-f32-operators/operator.cpp').read_text()
        self.assertNotIn('ggml_mul_mat(', source)
        self.assertIn('ggml_flash_attn_ext', source)
        self.assertIn('ggml_norm', source)
        self.assertIn('ggml_gelu_erf', source)
        self.assertIn('ggml_permute', source)
        self.assertIn('sqrtf(32.0f)', source)

    def test_anchor_barrier_precedes_any_cross_backend_metrics(self):
        source = (ROOT / 'tools/ci-heavy/wasm_f32_operators_v1.py').read_text()
        compare = source[source.index('def compare(out):'):]
        self.assertLess(compare.index('own_anchors(packets, outputs)'), compare.index('metrics(np.memmap'))
        workflow = (ROOT / '.github/workflows/wasm-f32-operators-v1.yml').read_text()
        self.assertIn("version: '6.0.2'", workflow)
        self.assertIn('playwright@1.56.0', workflow)
        self.assertIn('if: always()', workflow)
        self.assertNotIn('model forward', workflow)
        browser = (ROOT / 'tests/wasm-f32-operators/browser.cjs').read_text()
        self.assertIn("Cross-Origin-Embedder-Policy','require-corp'", browser)
        self.assertIn('SharedArrayBuffer', browser)
        self.assertIn('retainPacket', browser)


if __name__ == '__main__':
    unittest.main()
