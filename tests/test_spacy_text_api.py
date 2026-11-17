#!/usr/bin/env python3
"""Explicit real-model controls for the counted-text spaCy C ABI.

Run with --library, --bundle and a new --report path. Missing assets or text
capabilities are failures, never skips. These authored API/ownership cases do
not use the frozen acceptance inputs or establish linguistic model quality.
Only the Python standard library is required by this developer harness.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAX_TEXT_BYTES = 1024 * 1024
MAX_TOKENS = 2048
OK, INVALID_ARGUMENT, CANCELLED, TRACE_ERROR = 0, 2, 6, 7
CANCEL = C.CFUNCTYPE(C.c_int, C.c_void_p)
TRACE = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_char_p,
                    C.POINTER(C.c_float), C.POINTER(C.c_size_t), C.c_size_t)
TEXT_FIELDS = ("text", "norm", "prefix", "suffix", "shape", "lemma",
               "tag", "pos", "morph", "dependency")
SPAN_FIELDS = ("byte_start", "byte_end", "scalar_start", "scalar_end",
               "utf16_start", "utf16_end")


class TextView(C.Structure):
    # c_char_p would eagerly truncate at NUL when reading the field.
    _fields_ = [("data", C.c_void_p), ("length", C.c_size_t)]


class TokenView(C.Structure):
    _fields_ = ([(name, TextView) for name in TEXT_FIELDS]
                + [(name, C.c_size_t) for name in SPAN_FIELDS]
                + [("orth_id", C.c_uint64), ("features", C.c_uint64 * 6), ("head", C.c_int32),
                   ("flags", C.c_uint32)])


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=nonfinite)


def validate_bundle(bundle):
    manifest = read_json(bundle / "manifest.json")
    pinned = read_json(ROOT / "docs/validation/native-spacy-20261117/bundle-manifest.json")
    require(manifest == pinned, "Bundle differs from the externally retained W1 manifest")
    require(set(manifest["files"]) == {"model.gguf", "resources.json", "LICENSE", "LICENSES_SOURCES"},
            "Incomplete bundle inventory")
    for name, entry in manifest["files"].items():
        path = bundle / name
        require(path.is_file() and not path.is_symlink(), f"Missing/linked asset: {name}")
        require(path.stat().st_size == entry["size_bytes"] and sha256(path) == entry["sha256"],
                f"Changed bundle asset: {name}")
    return manifest


class NativeAPI:
    def __init__(self, library, bundle, manifest):
        self.bundle = bundle
        self.manifest = manifest
        self.lib = lib = C.CDLL(str(library.resolve()))
        for name in ("abi_version", "text_abi_version", "capabilities"):
            fn = getattr(lib, "crispembed_spacy_" + name)
            fn.argtypes, fn.restype = [], C.c_uint32
        lib.crispembed_spacy_context_capabilities.argtypes = [C.c_void_p]
        lib.crispembed_spacy_context_capabilities.restype = C.c_uint32
        lib.crispembed_spacy_create.argtypes = [C.c_char_p, C.c_char_p, C.c_char_p, C.c_size_t]
        lib.crispembed_spacy_create.restype = C.c_void_p
        lib.crispembed_spacy_create_with_resources.argtypes = [C.c_char_p] * 5 + [C.c_size_t]
        lib.crispembed_spacy_create_with_resources.restype = C.c_void_p
        lib.crispembed_spacy_free.argtypes, lib.crispembed_spacy_free.restype = [C.c_void_p], None
        lib.crispembed_spacy_tokenize_utf8.argtypes = [
            C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p),
            CANCEL, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.crispembed_spacy_tokenize_utf8.restype = C.c_int
        lib.crispembed_spacy_analyze_utf8.argtypes = [
            C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_void_p),
            TRACE, C.c_void_p, CANCEL, C.c_void_p, C.c_char_p, C.c_size_t]
        lib.crispembed_spacy_analyze_utf8.restype = C.c_int
        lib.crispembed_spacy_document_token_count.argtypes = [C.c_void_p]
        lib.crispembed_spacy_document_token_count.restype = C.c_size_t
        lib.crispembed_spacy_document_is_analyzed.argtypes = [C.c_void_p]
        lib.crispembed_spacy_document_is_analyzed.restype = C.c_int
        lib.crispembed_spacy_document_token.argtypes = [C.c_void_p, C.c_size_t, C.POINTER(TokenView)]
        lib.crispembed_spacy_document_token.restype = C.c_int
        lib.crispembed_spacy_document_free.argtypes = [C.c_void_p]
        lib.crispembed_spacy_document_free.restype = None

    def open(self, *, resources=True, model_hash=None, resource_hash=None, resource_path=None):
        error = C.create_string_buffer(1024)
        model_hash = model_hash or self.manifest["files"]["model.gguf"]["sha256"]
        if resources:
            resource_hash = resource_hash or self.manifest["files"]["resources.json"]["sha256"]
            context = self.lib.crispembed_spacy_create_with_resources(
                os.fsencode(self.bundle / "model.gguf"), model_hash.encode("ascii"),
                os.fsencode(resource_path or self.bundle / "resources.json"), resource_hash.encode("ascii"),
                error, len(error))
        else:
            context = self.lib.crispembed_spacy_create(
                os.fsencode(self.bundle / "model.gguf"), model_hash.encode("ascii"), error, len(error))
        return context, error.value.decode("utf-8", "replace")

    def call(self, context, data, *, analyze=False, cancel=None, trace=None):
        buffer = C.create_string_buffer(data) if data is not None else None
        document = C.c_void_p(1)  # Every failure must replace this sentinel.
        error = C.create_string_buffer(1024)
        args = [context, buffer, len(data) if data is not None else 0, C.byref(document)]
        if analyze:
            args.extend([trace or TRACE(), None])
        args.extend([cancel or CANCEL(), None, error, len(error)])
        fn = self.lib.crispembed_spacy_analyze_utf8 if analyze else self.lib.crispembed_spacy_tokenize_utf8
        status = fn(*args)
        return status, document.value, error.value.decode("utf-8", "replace")

    def tokens(self, document):
        result = []
        for index in range(self.lib.crispembed_spacy_document_token_count(document)):
            view = TokenView()
            require(self.lib.crispembed_spacy_document_token(document, index, C.byref(view)) == OK,
                    f"Token view failed at index {index}")
            copied = {}
            for name in TEXT_FIELDS:
                field = getattr(view, name)
                require(field.data or field.length == 0, f"Null counted payload: {name}")
                copied[name] = C.string_at(field.data, field.length) if field.length else b""
            copied.update({name: int(getattr(view, name)) for name in SPAN_FIELDS})
            copied.update(orth_id=int(view.orth_id), features=list(view.features), head=view.head, flags=view.flags)
            result.append(copied)
        return result


class TextAPITests(unittest.TestCase):
    api = None
    scratch = None

    @classmethod
    def setUpClass(cls):
        cls.context, error = cls.api.open()
        require(cls.context, "Required authenticated text context failed: " + error)

    @classmethod
    def tearDownClass(cls):
        cls.api.lib.crispembed_spacy_free(cls.context)

    def successful(self, data, *, context=None, analyze=False):
        status, document, error = self.api.call(context or self.context, data, analyze=analyze)
        self.assertEqual(status, OK, error)
        self.assertTrue(document and document != 1, "Success requires an owned document")
        return document

    def rejected(self, status, document, error, expected=None):
        self.assertNotEqual(status, OK, error)
        if expected is not None:
            self.assertEqual(status, expected, error)
        self.assertIsNone(document, "Failure did not clear the output pointer")
        self.assertTrue(error, "Failure needs an actionable diagnostic")

    def assert_source(self, data, tokens):
        rebuilt = bytearray()
        previous = 0
        for token in tokens:
            start, end = token["byte_start"], token["byte_end"]
            self.assertEqual(start, previous)
            self.assertGreater(end, start)
            self.assertLessEqual(end, len(data))
            self.assertEqual(token["text"], data[start:end])
            for byte_key, scalar_key, utf16_key in (("byte_start", "scalar_start", "utf16_start"),
                                                   ("byte_end", "scalar_end", "utf16_end")):
                prefix = data[:token[byte_key]].decode("utf-8", "strict")
                self.assertEqual(token[scalar_key], len(prefix))
                self.assertEqual(token[utf16_key], len(prefix.encode("utf-16-le")) // 2)
            rebuilt.extend(token["text"])
            if token["flags"] & 1:
                rebuilt.extend(b" ")
            previous = len(rebuilt)
            self.assertEqual(token["features"][4], bool(token["flags"] & 1))
            self.assertEqual(token["features"][5], bool(token["flags"] & 2))
            for name in TEXT_FIELDS:
                token[name].decode("utf-8", "strict")
        self.assertEqual(bytes(rebuilt), data)

    def test_versions_and_capabilities(self):
        lib = self.api.lib
        self.assertEqual(lib.crispembed_spacy_abi_version(), 1)
        self.assertEqual(lib.crispembed_spacy_text_abi_version(), 1)
        self.assertEqual(lib.crispembed_spacy_capabilities(), 3)
        self.assertEqual(lib.crispembed_spacy_context_capabilities(None), 0)
        self.assertEqual(lib.crispembed_spacy_context_capabilities(self.context), 15)

    def test_disabled_gate_on_contexts_and_calls(self):
        old_gate = os.environ.get("CRISPEMBED_SPACY")
        try:
            os.environ["CRISPEMBED_SPACY"] = "off"
            context, error = self.api.open()
            try:
                self.assertFalse(context)
                self.assertTrue(error)
            finally:
                self.api.lib.crispembed_spacy_free(context)
            for analyze in (False, True):
                with self.subTest(analyze=analyze):
                    self.rejected(*self.api.call(self.context, b"Ein Satz.", analyze=analyze), expected=1)
        finally:
            if old_gate is None:
                os.environ.pop("CRISPEMBED_SPACY", None)
            else:
                os.environ["CRISPEMBED_SPACY"] = old_gate

    def test_legacy_context_rejects_text(self):
        context, error = self.api.open(resources=False)
        self.assertTrue(context, error)
        try:
            self.assertEqual(self.api.lib.crispembed_spacy_context_capabilities(context), 3)
            for analyze in (False, True):
                with self.subTest(analyze=analyze):
                    self.rejected(*self.api.call(context, b"Ein Satz.", analyze=analyze))
        finally:
            self.api.lib.crispembed_spacy_free(context)

    def test_admission_hashes_and_missing_resources(self):
        checks = [dict(model_hash="0" * 64), dict(resource_hash="0" * 64),
                  dict(resource_hash=self.api.manifest["files"]["resources.json"]["sha256"].upper()),
                  dict(resource_path=self.scratch / "does-not-exist.json")]
        for index, options in enumerate(checks):
            with self.subTest(case=index):
                context, error = self.api.open(**options)
                try:
                    self.assertFalse(context, "Untrusted/missing resources were admitted")
                    self.assertTrue(error)
                finally:
                    self.api.lib.crispembed_spacy_free(context)

    def test_corrupt_and_oversized_resources_rejected(self):
        original = (self.api.bundle / "resources.json").read_bytes()
        path = self.scratch / "corrupt-resources.json"
        for label, payload in (("changed_same_size", b"!" + original[1:]),
                               ("truncated", original[:-1]), ("oversized", b" " * (32 * 1024 * 1024 + 1))):
            with self.subTest(case=label):
                path.write_bytes(payload)
                context, error = self.api.open(resource_path=path)
                try:
                    self.assertFalse(context)
                    self.assertTrue(error)
                finally:
                    self.api.lib.crispembed_spacy_free(context)
        path.unlink()

    def test_resources_owned_after_open(self):
        path = self.scratch / "owned-resources.json"
        path.write_bytes((self.api.bundle / "resources.json").read_bytes())
        context, error = self.api.open(resource_path=path)
        self.assertTrue(context, error)
        path.unlink()
        try:
            document = self.successful(b"Der Hund laeuft.", context=context, analyze=True)
            try:
                self.assert_source(b"Der Hund laeuft.", self.api.tokens(document))
            finally:
                self.api.lib.crispembed_spacy_document_free(document)
        finally:
            self.api.lib.crispembed_spacy_free(context)

    def test_empty_counted_and_null_empty(self):
        for data in (b"", None):
            for analyze in (False, True):
                with self.subTest(null=data is None, analyze=analyze):
                    document = self.successful(data, analyze=analyze)
                    try:
                        self.assertEqual(self.api.lib.crispembed_spacy_document_token_count(document), 0)
                        self.assertEqual(self.api.lib.crispembed_spacy_document_is_analyzed(document), int(analyze))
                    finally:
                        self.api.lib.crispembed_spacy_document_free(document)

    def test_malformed_utf8_and_output_reset(self):
        malformed = (b"\x80", b"\xc0\xaf", b"\xe0\x80\x80", b"\xed\xa0\x80",
                     b"\xf4\x90\x80\x80", b"\xf5\x80\x80\x80", b"\xe2\x82", b"\xff", b"A\xf0\x9f")
        for index, data in enumerate(malformed):
            for analyze in (False, True):
                with self.subTest(case=index, analyze=analyze):
                    self.rejected(*self.api.call(self.context, data, analyze=analyze))

    def test_counted_nul_and_original_offsets(self):
        samples = (b"vor\x00nach", b"\x00", " Ä\u0308 😀\u200d💻\r\nStraße\u00a0  Ende. ".encode("utf-8"))
        for index, data in enumerate(samples):
            for analyze in (False, True):
                with self.subTest(case=index, analyze=analyze):
                    document = self.successful(data, analyze=analyze)
                    try:
                        tokens = self.api.tokens(document)
                        self.assert_source(data, tokens)
                        self.assertEqual(self.api.lib.crispembed_spacy_document_is_analyzed(document), int(analyze))
                        for token in tokens:
                            if analyze:
                                self.assertTrue(token["tag"] and token["pos"] and token["dependency"])
                                self.assertGreaterEqual(token["head"], 0)
                                self.assertLess(token["head"], len(tokens))
                            else:
                                self.assertEqual(token["head"], -1)
                                self.assertTrue(all(not token[name] for name in TEXT_FIELDS[5:]))
                    finally:
                        self.api.lib.crispembed_spacy_document_free(document)

    def test_token_count_boundary(self):
        data = b"a " * MAX_TOKENS
        document = self.successful(data)
        try:
            self.assertEqual(self.api.lib.crispembed_spacy_document_token_count(document), MAX_TOKENS)
            self.assert_source(data, self.api.tokens(document))
        finally:
            self.api.lib.crispembed_spacy_document_free(document)
        self.rejected(*self.api.call(self.context, data + b"a"))

    def test_text_byte_boundary(self):
        data = b" " * MAX_TEXT_BYTES
        document = self.successful(data)
        try:
            self.assert_source(data, self.api.tokens(document))
        finally:
            self.api.lib.crispembed_spacy_document_free(document)
        for analyze in (False, True):
            with self.subTest(analyze=analyze):
                self.rejected(*self.api.call(self.context, data + b" ", analyze=analyze))

    def test_null_argument_combinations(self):
        lib = self.api.lib
        for analyze in (False, True):
            with self.subTest(analyze=analyze):
                self.rejected(*self.api.call(None, b"Text", analyze=analyze), expected=INVALID_ARGUMENT)
                output = C.c_void_p(1)
                error = C.create_string_buffer(128)
                args = [self.context, None, 1, C.byref(output)]
                if analyze:
                    args.extend([TRACE(), None])
                args.extend([CANCEL(), None, error, len(error)])
                fn = lib.crispembed_spacy_analyze_utf8 if analyze else lib.crispembed_spacy_tokenize_utf8
                status = fn(*args)
                self.rejected(status, output.value, error.value, INVALID_ARGUMENT)
                args[1], args[2], args[3] = C.create_string_buffer(b"X"), 1, None
                self.assertEqual(fn(*args), INVALID_ARGUMENT)

    def test_error_buffer_capacity(self):
        lib = self.api.lib
        for capacity in (0, 1, 4):
            with self.subTest(capacity=capacity):
                buffer = C.create_string_buffer(b"XXXXXXXX", 8)
                document = C.c_void_p(1)
                status = lib.crispembed_spacy_tokenize_utf8(
                    None, None, 1, C.byref(document), CANCEL(), None, buffer, capacity)
                self.assertEqual(status, INVALID_ARGUMENT)
                self.assertIsNone(document.value)
                self.assertEqual(buffer.raw[capacity:], b"X" * (8 - capacity))
                if capacity:
                    self.assertIn(0, buffer.raw[:capacity])
                else:
                    self.assertEqual(buffer.raw, b"XXXXXXXX")
        document = C.c_void_p(1)
        self.assertEqual(lib.crispembed_spacy_tokenize_utf8(
            self.context, None, 0, C.byref(document), CANCEL(), None, None, 1), INVALID_ARGUMENT)
        self.assertIsNone(document.value)

    def test_cancellation_and_recovery(self):
        @CANCEL
        def cancel(_user):
            return 1

        for analyze in (False, True):
            with self.subTest(analyze=analyze):
                self.rejected(*self.api.call(self.context, b"Ein Satz.", analyze=analyze, cancel=cancel),
                              expected=CANCELLED)
                document = self.successful(b"Ein Satz.", analyze=analyze)
                self.api.lib.crispembed_spacy_document_free(document)

    def test_cancellation_after_work_begins(self):
        calls = [0]

        @CANCEL
        def cancel(_user):
            calls[0] += 1
            return int(calls[0] >= 4)

        self.rejected(*self.api.call(self.context, b"Der Hund sieht den Mann.", analyze=True, cancel=cancel),
                      expected=CANCELLED)
        self.assertGreaterEqual(calls[0], 4)
        document = self.successful(b"Ein Satz.", analyze=True)
        self.api.lib.crispembed_spacy_document_free(document)

    def test_trace_abort_and_recovery(self):
        stages = []

        @TRACE
        def abort(_user, stage, _values, _shape, _rank):
            stages.append(stage.decode("utf-8"))
            return 1

        self.rejected(*self.api.call(self.context, b"Ein Satz.", analyze=True, trace=abort),
                      expected=TRACE_ERROR)
        self.assertTrue(stages)
        document = self.successful(b"Ein Satz.", analyze=True)
        self.api.lib.crispembed_spacy_document_free(document)

    def test_input_buffer_is_not_borrowed(self):
        data = "Straße 😀 bleibt.".encode("utf-8")
        buffer = C.create_string_buffer(data)
        output = C.c_void_p()
        error = C.create_string_buffer(256)
        status = self.api.lib.crispembed_spacy_tokenize_utf8(
            self.context, buffer, len(data), C.byref(output), CANCEL(), None, error, len(error))
        self.assertEqual(status, OK, error.value)
        try:
            C.memset(buffer, ord("X"), len(data))
            self.assert_source(data, self.api.tokens(output))
        finally:
            self.api.lib.crispembed_spacy_document_free(output)

    def test_document_lifetime_after_context(self):
        context, error = self.api.open()
        self.assertTrue(context, error)
        document = None
        try:
            document = self.successful(b"Der Hund sieht den Mann.", context=context, analyze=True)
            before = self.api.tokens(document)
        finally:
            self.api.lib.crispembed_spacy_free(context)
        try:
            self.assertEqual(self.api.tokens(document), before)
            self.assertEqual(self.api.lib.crispembed_spacy_document_is_analyzed(document), 1)
        finally:
            self.api.lib.crispembed_spacy_document_free(document)

    def test_invalid_document_views_are_cleared(self):
        lib = self.api.lib
        self.assertEqual(lib.crispembed_spacy_document_token_count(None), 0)
        self.assertEqual(lib.crispembed_spacy_document_is_analyzed(None), 0)
        document = self.successful(b"Text")
        try:
            for pointer, index in ((None, 0), (document, 1), (document, C.c_size_t(-1).value)):
                with self.subTest(null=pointer is None, index=index):
                    token = TokenView()
                    C.memset(C.byref(token), 255, C.sizeof(token))
                    self.assertEqual(lib.crispembed_spacy_document_token(pointer, index, C.byref(token)),
                                     INVALID_ARGUMENT)
                    for name in TEXT_FIELDS:
                        self.assertIsNone(getattr(token, name).data)
                        self.assertEqual(getattr(token, name).length, 0)
                    self.assertTrue(all(getattr(token, name) == 0 for name in SPAN_FIELDS))
                    self.assertEqual(token.orth_id, 0)
                    self.assertEqual(list(token.features), [0] * 6)
                    self.assertEqual((token.head, token.flags), (0, 0))
            self.assertEqual(lib.crispembed_spacy_document_token(document, 0, None), INVALID_ARGUMENT)
        finally:
            lib.crispembed_spacy_document_free(document)
        lib.crispembed_spacy_document_free(None)
        lib.crispembed_spacy_free(None)

    def test_same_context_concurrent_calls(self):
        samples = (b"Der Hund sieht den Mann.", "Straße 😀 Straße.".encode("utf-8"), b"A\x00B", b"  X\r\n Y")

        def run(item):
            index, data = item
            status, document, error = self.api.call(self.context, data, analyze=bool(index % 2))
            require(status == OK and document and document != 1, f"Concurrent call failed: {status}: {error}")
            try:
                return data, self.api.tokens(document)
            finally:
                self.api.lib.crispembed_spacy_document_free(document)

        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(run, enumerate(samples)))
        self.assertEqual(len(outputs), len(samples))
        for data, tokens in outputs:
            self.assert_source(data, tokens)

    def test_immutable_document_concurrent_readers(self):
        document = self.successful(b"Der Hund sieht den Mann.", analyze=True)
        try:
            expected = self.api.tokens(document)
            with ThreadPoolExecutor(max_workers=4) as pool:
                snapshots = list(pool.map(lambda _index: self.api.tokens(document), range(16)))
            self.assertEqual(snapshots, [expected] * 16)
        finally:
            self.api.lib.crispembed_spacy_document_free(document)


class RecordingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.events.append({"test": test.id(), "status": "pass"})

    def addFailure(self, test, error):
        super().addFailure(test, error)
        self.events.append({"test": test.id(), "status": "fail", "diagnostic": self._exc_info_to_string(error, test)})

    def addError(self, test, error):
        super().addError(test, error)
        self.events.append({"test": test.id(), "status": "error", "diagnostic": self._exc_info_to_string(error, test)})

    def addSubTest(self, test, subtest, error):
        super().addSubTest(test, subtest, error)
        event = {"test": subtest.id(), "status": "pass" if error is None else "fail", "subtest": True}
        if error is not None:
            event["diagnostic"] = self._exc_info_to_string(error, subtest)
        self.events.append(event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require(args.library.is_file(), "The compiled text API library is required")
    require(not args.report.exists(), "Refusing to replace a retained report")
    manifest = validate_bundle(args.bundle)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    old_gate = os.environ.get("CRISPEMBED_SPACY")
    os.environ["CRISPEMBED_SPACY"] = "1"
    started = time.perf_counter()
    try:
        TextAPITests.api = NativeAPI(args.library, args.bundle, manifest)
        with tempfile.TemporaryDirectory(prefix="spacy-text-api-", dir=args.report.parent) as scratch:
            TextAPITests.scratch = Path(scratch)
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(TextAPITests)
            result = unittest.TextTestRunner(verbosity=2, resultclass=RecordingResult).run(suite)
        report = {
            "schema_version": 1,
            "scope": "real-model authored counted-text C ABI and ownership controls; no acceptance inputs or grammar quality claims",
            "status": "pass" if result.wasSuccessful() and result.testsRun > 0 and not result.skipped else "fail",
            "tests_run": result.testsRun,
            "failures": len(result.failures), "errors": len(result.errors), "skips": len(result.skipped),
            "elapsed_seconds": time.perf_counter() - started,
            "python_version": platform.python_version(), "platform": platform.system(), "machine": platform.machine(),
            "library_sha256": sha256(args.library),
            "bundle_manifest_sha256": sha256(args.bundle / "manifest.json"),
            "model_sha256": manifest["files"]["model.gguf"]["sha256"],
            "resources_sha256": manifest["files"]["resources.json"]["sha256"],
            "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                [Path(__file__), *sorted((ROOT / "src").glob("spacy_*.h")), *sorted((ROOT / "src").glob("spacy_*.cpp"))]},
            "results": result.events,
        }
        encoded = json.dumps(report, ensure_ascii=False, indent=2)
        for actual, symbolic in sorted(((str(ROOT), "$CRISPEMBED_CHECKOUT"),
                                         (str(args.bundle.resolve()), "$SPACY_BUNDLE"),
                                         (str(args.report.parent.resolve()), "$SPACY_REPORT_DIR")),
                                        key=lambda pair: len(pair[0]), reverse=True):
            encoded = encoded.replace(actual, symbolic)
        with args.report.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
        return 0 if report["status"] == "pass" else 1
    finally:
        if old_gate is None:
            os.environ.pop("CRISPEMBED_SPACY", None)
        else:
            os.environ["CRISPEMBED_SPACY"] = old_gate


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, AttributeError) as error:
        print(f"Required text API qualification failed: {error}", file=sys.stderr)
        sys.exit(2)
