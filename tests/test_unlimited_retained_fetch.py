"""Small mocked delivery corruption checks; no network, arrays or compiler."""
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace, ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/unlimited-ocr2-q4k'))
import retained_fetch_gha as f


class Response(io.BytesIO):
    def __init__(self, body, *, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = {'Content-Length': str(len(body)), **(headers or {})}


class Controls(unittest.TestCase):
    def run_transfer(self, body, pin=None, response=None):
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        path = Path(td.name) / 'retained.bin'
        if pin is None:
            pin = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        def open_response(request, timeout):
            self.assertEqual(request.get_method(), 'GET')
            self.assertEqual(timeout, 180)
            return response or Response(body)
        return path, pin, open_response

    def test_exact_small_get_admitted_only_after_digest(self):
        body = b'bounded retained payload'
        path, pin, opener = self.run_transfer(body)
        proof = f.download('https://example.invalid/signed', path, pin, opener=opener)
        self.assertEqual(path.read_bytes(), body)
        self.assertEqual(proof['sha256'], pin['sha256'])
        self.assertFalse(path.with_name(path.name + '.partial').exists())
        with self.assertRaisesRegex(ValueError, 'Fresh'):
            f.download('https://example.invalid/signed', path, pin, opener=opener)

    def test_corrupt_truncated_excess_encoded_or_error_response_never_admitted(self):
        body = b'unchanged producer bytes'
        for response in [Response(body, status=404), Response(body[:-1], headers={'Content-Length': str(len(body))}),
                         Response(body + b'x', headers={'Content-Length': str(len(body))}),
                         Response(body, headers={'Content-Encoding': 'gzip'}),
                         Response(b'X' + body[1:])]:
            path, pin, opener = self.run_transfer(body, response=response)
            with self.assertRaisesRegex(ValueError, 'Retained GET rejected'):
                f.download('https://example.invalid/never-log-this', path, pin, opener=opener)
            self.assertFalse(path.exists())

    def test_large_get_requires_exact_range_before_payload(self):
        # Reject at headers: never materialize or hash a large local fixture.
        pin = dict(bytes=2**30, sha256='0' * 64)
        for response in [Response(b'', status=200), Response(b'', status=206,
                         headers={'Content-Range': 'bytes 1-67108864/1073741824', 'Content-Length': '67108864'})]:
            path, _, opener = self.run_transfer(b'x', pin=pin, response=response)
            with self.assertRaisesRegex(ValueError, 'Retained GET rejected'):
                f.download('https://example.invalid/signed', path, pin, opener=opener)
            self.assertFalse(path.exists())

    def test_unsafe_paths_reject(self):
        for name in ('/outside', '../outside', 'a/../b', 'a//b', 'a\\b', './a', ''):
            with self.assertRaises(ValueError):
                f.safe_name(name)
        self.assertEqual(str(f.safe_name('baseline/table/stages.jsonl')), 'baseline/table/stages.jsonl')

    def test_sdk_cannot_import_array_or_framework_packages_before_byte_barrier(self):
        finder = f.NoArrays()
        sys.meta_path.insert(0, finder)
        try:
            for name in ('numpy', 'torch', 'tensorflow', 'transformers'):
                self.assertNotIn(name, sys.modules)
                with self.assertRaisesRegex(ImportError, 'before retained byte authentication'):
                    importlib.import_module(name)
                self.assertNotIn(name, sys.modules)
        finally:
            sys.meta_path.remove(finder)

    def test_shadowed_sdk_package_and_every_paginated_request_use_v3(self):
        class Request:
            def __str__(self):
                return json.dumps({'versionLabel': self.version_label})
        module = ModuleType('kaggle.api.kaggle_api_extended')
        for name in ('ApiGetKernelRequest', 'ApiGetKernelSessionStatusRequest', 'ApiListKernelSessionOutputRequest'):
            setattr(module, name, Request)
        calls = []
        script = f.audit.PRODUCER_SOURCE
        def check(q):
            self.assertEqual((q.user_name, q.kernel_slug, q.version_label), ('fixture', f.SLUG, 'v3'))
            calls.append(q)
        class Kernel:
            def get_kernel(inner, q):
                check(q)
                return json.dumps({'blob': {'source': script}})
            def get_kernel_session_status(inner, q):
                check(q)
                return SimpleNamespace(status='ERROR')
            def list_kernel_session_output(inner, q):
                check(q)
                second = hasattr(q, 'page_token')
                return SimpleNamespace(files=[SimpleNamespace(file_name='second' if second else 'first',
                                           url='https://example.invalid/not-serialized')],
                                       next_page_token=None if second else 'next')
        class Client:
            def __enter__(inner):
                return SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=Kernel()))
            def __exit__(inner, *args):
                pass
        shadow = ModuleType('kaggle')
        shadow.api = object()  # Exact top-level attribute shadow that broke dotted import.
        with patch.dict(sys.modules, {'kaggle': shadow, module.__name__: module}), \
                patch.object(f, 'SCRIPT_BYTES', len(script)), \
                patch.object(f, 'SCRIPT_SHA', hashlib.sha256(script.encode()).hexdigest()):
            urls, proof = f.output_urls(SimpleNamespace(build_kaggle_client=Client), 'fixture')
        self.assertEqual(set(urls), {'first', 'second'})
        self.assertEqual(len(calls), 4)
        self.assertEqual(proof['pages'], 2)
        self.assertNotIn('https://', json.dumps(proof))

    def test_old_audit_source_preserved_except_host_admission_in_new_copy(self):
        previous = (ROOT / 'tools/unlimited-ocr2-q4k/retained_v3.py').read_text()
        new = (ROOT / 'tools/unlimited-ocr2-q4k/retained_v3_gha.py').read_text()
        a = "c.require(Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir() and\n              os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Batch', 'Interactive'), 'Genuine hosted CPU audit required')"
        b = "c.require(os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_SHA') == source,\n              'Genuine source-bound GHA CPU audit required')"
        self.assertEqual(previous.replace('Read-only hosted audit of the stopped v3 conversion; no model forwards.',
                                         'Read-only GHA audit of authenticated stopped-v3 bytes; no model forwards.').replace(a, b), new)
        self.assertNotIn('numpy', sys.modules)


if __name__ == '__main__':
    unittest.main()
