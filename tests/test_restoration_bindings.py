#!/usr/bin/env python3
"""Model-free restoration API calls through typed ctypes native stand-ins."""

import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "restoration_binding_under_test", ROOT / "python/crispembed/_binding.py")
BINDING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BINDING)
U8P = ctypes.POINTER(ctypes.c_uint8)
INTP = ctypes.POINTER(ctypes.c_int)
ENGINES = {
    "swinir_sr": (BINDING.CrispSwinirSr, "SwinIR SR"),
    "scunet": (BINDING.CrispScunet, "SCUNet"),
    "safmn_sr": (BINDING.CrispSafmnSr, "SAFMN SR"),
    "tbsrn_sr": (BINDING.CrispTbsrnSr, "TBSRN SR"),
}


class NativeRestoration:
    """Only exposes the selected engine; ctypes checks the actual call arity."""

    def __init__(self, engine, rc=0, null_output=False, init_ok=True,
                 allocate_on_error=False, invalid_output_dimensions=False):
        self.engine = engine
        self.rc = rc
        self.null_output = null_output
        self.allocate_on_error = allocate_on_error
        self.calls = []
        self.freed = []
        self.images_freed = 0
        self.storage = None
        prefix = "crispembed_" + engine
        self.lib = SimpleNamespace()

        def export(name, result, args, callback):
            setattr(self.lib, prefix + name, ctypes.CFUNCTYPE(result, *args)(callback))

        def init(path, threads):
            self.calls.append(("init", path, threads))
            return 123 if init_ok else None

        def free(ctx):
            self.freed.append(ctx)

        def free_image(ptr):
            self.images_freed += 1
            # Invalidate the stand-in native storage: returned arrays must own copies.
            ctypes.memset(ptr, 255, len(self.storage))

        export("_init", ctypes.c_void_p, [ctypes.c_char_p, ctypes.c_int], init)
        export("_free", None, [ctypes.c_void_p], free)
        export("_free_image", None, [U8P], free_image)
        if engine in ("swinir_sr", "safmn_sr"):
            export("_scale", ctypes.c_int, [ctypes.c_void_p], lambda ctx: 2)

            def process(ctx, pixels, w, h, tile, overlap, out, ow, oh):
                self.calls.append(("process", ctx, w, h, tile, overlap,
                                   list(pixels[:w * h * 3])))
                ow[0], oh[0] = invalid_output_dimensions or (w * 2, h * 2)
                return self.output(out, w * h * 12)

            args = [ctypes.c_void_p, U8P, ctypes.c_int, ctypes.c_int,
                    ctypes.c_int, ctypes.c_int, ctypes.POINTER(U8P), INTP, INTP]
        elif engine == "tbsrn_sr":
            def process(ctx, pixels, w, h, out, ow, oh):
                self.calls.append(("process", ctx, w, h, list(pixels[:w * h * 3])))
                ow[0], oh[0] = invalid_output_dimensions or (128, 32)
                return self.output(out, 128 * 32 * 3)

            args = [ctypes.c_void_p, U8P, ctypes.c_int, ctypes.c_int,
                    ctypes.POINTER(U8P), INTP, INTP]
        else:
            def process(ctx, pixels, w, h, out):
                self.calls.append(("process", ctx, w, h, list(pixels[:w * h * 3])))
                return self.output(out, w * h * 3)

            args = [ctypes.c_void_p, U8P, ctypes.c_int, ctypes.c_int, ctypes.POINTER(U8P)]
        export("_process", ctypes.c_int, args, process)

    def output(self, out, count):
        if (self.rc and not self.allocate_on_error) or self.null_output:
            return self.rc
        self.storage = (ctypes.c_uint8 * count)(*range(count))
        out[0] = ctypes.cast(self.storage, U8P)
        return self.rc


class RestorationBindingTests(unittest.TestCase):
    def create(self, engine, **kwargs):
        native = NativeRestoration(engine, **kwargs)
        cls = ENGINES[engine][0]
        with patch.object(BINDING, "_load_library", return_value=native.lib):
            model = cls("模型.gguf", n_threads=3, lib_path="chosen-library")
        self.addCleanup(model.__del__)
        return model, native

    def test_swinir_upscales_and_forwards_tiling(self):
        model, native = self.create("swinir_sr")
        self.assertEqual(native.calls[0], ("init", "模型.gguf".encode(), 3))
        self.assertEqual(model.scale, 2)
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)[:, ::-1]
        out, w, h = model.process(pixels, 3, 2, tile_size=64, tile_overlap=8)
        self.assertEqual((w, h, out.shape, out.dtype), (6, 4, (4, 6, 3), np.uint8))
        self.assertEqual(native.calls[-1],
                         ("process", 123, 3, 2, 64, 8, pixels.flatten().tolist()))
        np.testing.assert_array_equal(out.flatten(), np.arange(72, dtype=np.uint8))
        self.assertEqual(native.images_freed, 1)

    def test_swinir_auto_tiling(self):
        model, native = self.create("swinir_sr")
        model.process(np.zeros(3, dtype=np.uint8), 1, 1)
        self.assertEqual(native.calls[-1][4:6], (0, 0))

    def test_scunet_denoises_without_swinir_symbols(self):
        model, native = self.create("scunet")
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)[:, ::-1]
        out = model.process(pixels, 3, 2)
        self.assertEqual(out.shape, (2, 3, 3))
        self.assertEqual(native.calls[-1], ("process", 123, 3, 2, pixels.flatten().tolist()))
        np.testing.assert_array_equal(out.flatten(), np.arange(18, dtype=np.uint8))
        self.assertEqual(native.images_freed, 1)

    def test_matching_context_free_is_idempotent(self):
        for engine in ENGINES:
            with self.subTest(engine=engine):
                model, native = self.create(engine)
                model.__del__()
                model.__del__()
                self.assertEqual(native.freed, [123])

    def test_load_errors_name_correct_engine(self):
        for engine, (_, label) in ENGINES.items():
            with self.subTest(engine=engine):
                with self.assertRaisesRegex(RuntimeError, "Failed to load " + label):
                    self.create(engine, init_ok=False)

    def test_native_errors_and_null_outputs(self):
        for engine, (_, label) in ENGINES.items():
            for kwargs in ({"rc": -1}, {"null_output": True}):
                with self.subTest(engine=engine, kwargs=kwargs):
                    model, native = self.create(engine, **kwargs)
                    with self.assertRaisesRegex(RuntimeError, label + ".*failed"):
                        model.process(np.zeros(3, dtype=np.uint8), 1, 1)
                    self.assertEqual(native.images_freed, 0)

    def test_invalid_input_rejected_before_native_read(self):
        for engine in ENGINES:
            model, native = self.create(engine)
            for w, h, count in ((0, 1, 0), (-1, 1, 0), (1, 0, 0),
                                (1, -1, 0), (2**31, 1, 3), (1, 2**31, 3),
                                (1.5, 1, 3), (1, 1.5, 3), (1, 1, 2),
                                (1, 1, 4), (2, 1, 4)):
                with self.subTest(engine=engine, width=w, height=h, count=count):
                    with self.assertRaises(ValueError):
                        model.process(np.zeros(count, dtype=np.uint8), w, h)
                    self.assertEqual(len(native.calls), 1)

    def test_native_error_with_allocation_is_freed(self):
        for engine in ENGINES:
            with self.subTest(engine=engine):
                model, native = self.create(engine, rc=-1, allocate_on_error=True)
                with self.assertRaises(RuntimeError):
                    model.process(np.zeros(3, dtype=np.uint8), 1, 1)
                self.assertEqual(native.images_freed, 1)

    def test_invalid_upscaler_output_dimensions_are_freed(self):
        for engine in ("swinir_sr", "safmn_sr", "tbsrn_sr"):
            for dimensions in ((0, 2), (-1, -1)):
                with self.subTest(engine=engine, dimensions=dimensions):
                    model, native = self.create(engine, invalid_output_dimensions=dimensions)
                    with patch.object(np.ctypeslib, "as_array") as read:
                        with self.assertRaisesRegex(RuntimeError, "invalid output dimensions"):
                            model.process(np.zeros(3, dtype=np.uint8), 1, 1)
                        read.assert_not_called()
                    self.assertEqual(native.images_freed, 1)

    def test_safmn_upscales_using_only_matching_symbols(self):
        model, native = self.create("safmn_sr")
        self.assertEqual(native.calls[0], ("init", "模型.gguf".encode(), 3))
        self.assertEqual(model.scale, 2)
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)[:, ::-1]
        out, w, h = model.process(pixels, 3, 2, tile_size=64, tile_overlap=8)
        self.assertEqual((w, h, out.shape, out.dtype), (6, 4, (4, 6, 3), np.uint8))
        self.assertEqual(native.calls[-1],
                         ("process", 123, 3, 2, 64, 8, pixels.flatten().tolist()))
        np.testing.assert_array_equal(out.flatten(), np.arange(72, dtype=np.uint8))
        self.assertEqual(native.images_freed, 1)

    def test_safmn_auto_tiling(self):
        model, native = self.create("safmn_sr")
        model.process(np.zeros(3, dtype=np.uint8), 1, 1)
        self.assertEqual(native.calls[-1][4:6], (0, 0))

    def test_tbsrn_preserves_fixed_line_output_and_matching_symbols(self):
        model, native = self.create("tbsrn_sr")
        self.assertEqual(native.calls[0], ("init", "模型.gguf".encode(), 3))
        pixels = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)[:, ::-1]
        out, w, h = model.process(pixels, 3, 2)
        self.assertEqual((w, h, out.shape, out.dtype), (128, 32, (32, 128, 3), np.uint8))
        self.assertEqual(native.calls[-1], ("process", 123, 3, 2, pixels.flatten().tolist()))
        np.testing.assert_array_equal(out.flatten(), np.arange(128 * 32 * 3).astype(np.uint8))
        self.assertEqual(native.images_freed, 1)

    def test_output_is_freed_if_array_copy_fails(self):
        for engine in ENGINES:
            with self.subTest(engine=engine):
                model, native = self.create(engine)
                with patch.object(np.ctypeslib, "as_array", side_effect=MemoryError):
                    with self.assertRaises(MemoryError):
                        model.process(np.zeros(3, dtype=np.uint8), 1, 1)
                self.assertEqual(native.images_freed, 1)


if __name__ == "__main__":
    unittest.main()
