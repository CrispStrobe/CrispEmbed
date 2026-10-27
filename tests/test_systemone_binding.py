#!/usr/bin/env python3
"""Stdlib-only wrapper contract; native and model proof are separate hosted gates."""
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def load(lib):
    package = types.ModuleType('systemone_test_package')
    package.__path__ = []
    binding = types.ModuleType('systemone_test_package._binding')
    binding._load_library = Mock(return_value=lib)
    binding.CrispEmbed = types.SimpleNamespace(resolve_model=Mock(return_value='模型.gguf'))
    binding.set_offline = Mock()
    spec = importlib.util.spec_from_file_location('systemone_test_package._systemone', ROOT/'python/crispembed/_systemone.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {package.__name__: package, binding.__name__: binding}):
        spec.loader.exec_module(module)
    return module.CrispSystemOne, binding


def library():
    lib = types.SimpleNamespace()
    for name, result in [('init', 19), ('free', None), ('run_json', b'{"model":"test","answers":{},"usage":{}}'),
                         ('error', b'actual native rejection'), ('profile', b'official-pytorch-text-batch1-512-no-extra-temperature')]:
        setattr(lib, 'crispembed_systemone_'+name, Mock(return_value=result))
    return lib


class BindingControls(unittest.TestCase):
    def test_old_library_import_and_explicit_capability_error(self):
        cls, binding = load(types.SimpleNamespace())
        binding._load_library.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, 'does not provide SystemOne'):
            cls('model.gguf')
        binding.CrispEmbed.resolve_model.assert_not_called()

    def test_utf8_resolution_offline_library_and_deterministic_close(self):
        lib = library(); cls, binding = load(lib)
        with cls('model.gguf', lib_path='chosen.so', auto_download=False, offline=True, n_threads=0) as ctx:
            binding._load_library.assert_called_once_with('chosen.so')
            binding.CrispEmbed.resolve_model.assert_called_once_with('model.gguf', auto_download=False, lib_path='chosen.so')
            binding.set_offline.assert_called_once_with(True, lib_path='chosen.so')
            lib.crispembed_systemone_init.assert_called_once_with('模型.gguf'.encode(), 0)
            self.assertEqual(ctx.profile, 'official-pytorch-text-batch1-512-no-extra-temperature')
        ctx.close(); lib.crispembed_systemone_free.assert_called_once_with(19)
        with self.assertRaisesRegex(RuntimeError, 'closed'): ctx.run_json('{}')
        with self.assertRaisesRegex(RuntimeError, 'closed'): _ = ctx.profile

    def test_raw_order_and_complete_response(self):
        lib = library(); cls, _ = load(lib)
        with cls('model.gguf') as ctx:
            raw = '{"questions":{"z":{},"a":{}},"state":"Ä", "model":"x"}'
            self.assertIn('usage', json.loads(ctx.run_json(raw)))
            lib.crispembed_systemone_run_json.assert_called_with(19, raw.encode())
            request = {'questions': {'z': {}, 'a': {}}, 'state': 'Ä', 'model': 'x'}
            self.assertEqual(ctx.run(request), {'model':'test', 'answers':{}, 'usage':{}})
            sent = lib.crispembed_systemone_run_json.call_args.args[1]
            self.assertIn('Ä'.encode(), sent); self.assertLess(sent.index(b'"z"'), sent.index(b'"a"'))

    def test_native_failure_copies_error_and_next_call_recovers(self):
        lib = library(); cls, _ = load(lib)
        with cls('model.gguf') as ctx:
            lib.crispembed_systemone_run_json.return_value = None
            with self.assertRaisesRegex(ValueError, 'actual native rejection'): ctx.run_json('{}')
            lib.crispembed_systemone_error.assert_called_with(19)
            lib.crispembed_systemone_run_json.return_value = b'{"answers":{}}'
            self.assertEqual(ctx.run_json('{}'), '{"answers":{}}')

    def test_utf8_byte_boundary_nul_and_nonfinite_input_before_native(self):
        lib = library(); cls, _ = load(lib)
        with cls('model.gguf') as ctx:
            ctx.run_json('Ä'*32768)
            lib.crispembed_systemone_run_json.reset_mock()
            for text in ['Ä'*32769, 'x\0y', {}, 5]:
                with self.assertRaises(ValueError): ctx.run_json(text)
            with self.assertRaises(ValueError): ctx.run({'state':float('nan')})
            with self.assertRaises(TypeError): ctx.run([])
            lib.crispembed_systemone_run_json.assert_not_called()

    def test_invalid_constructor_and_native_init_error(self):
        for threads in [-1, 65, True, 1.5]:
            lib = library(); cls, binding = load(lib)
            with self.assertRaises(ValueError): cls('model.gguf', n_threads=threads)
            binding.CrispEmbed.resolve_model.assert_not_called()
        lib = library(); cls, binding = load(lib)
        with self.assertRaises(ValueError): cls('bad\0.gguf')
        binding._load_library.assert_not_called()
        lib.crispembed_systemone_init.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'actual native rejection'): cls('model.gguf')
        lib.crispembed_systemone_free.assert_not_called()


if __name__ == '__main__': unittest.main()
