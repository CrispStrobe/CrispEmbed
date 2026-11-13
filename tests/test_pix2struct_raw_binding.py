#!/usr/bin/env python3
"""Model-free RGB buffer contract controls for the production Pix2Struct binding."""

import ctypes
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_preprocess_bindings import BINDING


class NativePix2Struct:
    """Typed callbacks matching the public ABI; legacy text is borrowed/no-op free."""

    def __init__(self, output="日本語 €"):
        self.lib = SimpleNamespace()
        self.calls = []
        self.freed = []
        self.output = None if output is None else ctypes.create_string_buffer(output.encode())

        def export(name, result, args, callback):
            setattr(self.lib, "crispembed_pix2struct_" + name,
                    ctypes.CFUNCTYPE(result, *args)(callback))

        def generate(ctx, pixels, width, height, max_tokens):
            self.calls.append((ctx, width, height, max_tokens,
                               list(pixels[:width * height * 3])))
            return None if self.output is None else ctypes.addressof(self.output)

        ptr = ctypes.POINTER
        export("init", ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_int], lambda path, threads: 1)
        export("free", None, [ctypes.c_void_p], self.freed.append)
        export("generate", ctypes.c_void_p,
               [ctypes.c_void_p, ptr(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int], generate)
        export("free_text", None, [ctypes.c_char_p], lambda text: None)
        export("confidences", ctypes.c_void_p, [ctypes.c_void_p, ptr(ctypes.c_int)], lambda *args: None)
        export("mean_confidence", ctypes.c_float, [ctypes.c_void_p], lambda ctx: 0.75)
        export("encode_patches", ctypes.c_void_p,
               [ctypes.c_void_p, ptr(ctypes.c_float), ctypes.c_int, ptr(ctypes.c_int)], lambda *args: None)


class Pix2StructRawBindingTests(unittest.TestCase):
    def create(self, **kwargs):
        native = NativePix2Struct(**kwargs)
        with patch.object(BINDING, "_load_library", return_value=native.lib):
            model = BINDING.CrispPix2Struct("model.gguf")
        self.addCleanup(model.__del__)
        return model, native

    def test_exact_and_oversized_rgb_bytes_preserve_prefix_and_output(self):
        pixels = bytes(range(18))
        for raw in (pixels, pixels + b"extra", bytearray(pixels), memoryview(pixels)):
            with self.subTest(type=type(raw).__name__, size=len(raw)):
                model, native = self.create()
                self.assertEqual(model.generate_raw(raw, 3, 2, max_tokens=17), "日本語 €")
                self.assertEqual(native.calls, [(1, 3, 2, 17, list(pixels))])

    def test_noncontiguous_buffer_views_follow_logical_pixel_order(self):
        base = np.arange(36, dtype=np.uint8).reshape(2, 6, 3)
        for array in (base[:, ::2], base[::-1, ::-2]):
            with self.subTest(strides=array.strides):
                model, native = self.create()
                before = base.copy()
                model.generate_raw(memoryview(array), 3, 2)
                self.assertEqual(native.calls[-1][-1], array.flatten().tolist())
                np.testing.assert_array_equal(base, before)

    def test_typed_buffers_use_byte_count_instead_of_element_count(self):
        # Nine uint16 elements hold eighteen RGB bytes; len(view)==9 is not its byte size.
        array = np.arange(9, dtype=np.uint16)
        model, native = self.create()
        model.generate_raw(memoryview(array), 3, 2)
        self.assertEqual(native.calls[-1][-1], list(array.tobytes()))

    def test_invalid_input_rejects_before_any_native_pixel_read(self):
        for width, height, size in ((0, 1, 0), (1, 0, 0), (-1, 1, 0), (1, -1, 0),
                                    (2**31, 1, 0), (1, 2**31, 0), (2**30, 1, 0),
                                    (65536, 65536, 0), (1.5, 1, 3), (1, 1.5, 3),
                                    (1, 1, 2), (3, 2, 17)):
            with self.subTest(width=width, height=height, size=size):
                model, native = self.create()
                reject = Mock(side_effect=AssertionError("native read before validation"))
                native.lib.crispembed_pix2struct_generate = reject
                with self.assertRaises(ValueError):
                    model.generate_raw(bytes(size), width, height)
                reject.assert_not_called()
                self.assertEqual(native.calls, [])

    def test_null_and_empty_output_behavior_and_token_forwarding_are_unchanged(self):
        for output in (None, ""):
            for tokens in (0, -1, 256):
                with self.subTest(output=output, max_tokens=tokens):
                    model, native = self.create(output=output)
                    self.assertEqual(model.generate_raw(b"\x01\x02\x03", 1, 1, tokens), "")
                    self.assertEqual(native.calls[-1][3], tokens)


if __name__ == "__main__":
    unittest.main()
