"""Weight-free Python binding controls and actual upstream-helper composition.

Run with PYTHONPATH=python python3 -m unittest discover -s tests -p test_mineru_binding.py.
Install tools/mineru25-blueprint-requirements.txt for the helper composition
tests. Native inference is deliberately stubbed; these are not model parity.
"""
import ctypes as C
import importlib.util
import json
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from crispembed import CrispMinerU
from crispembed import _mineru


class _Fn:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class _Native:
    def __init__(self, texts=()):
        self.texts = list(texts)
        self.calls = []
        self.freed = 0
        self.eos = True
        self.error = b"injected native failure"
        self.crispembed_mineru_init = _Fn(lambda path, threads: 123)
        self.crispembed_mineru_free = _Fn(self._free)
        self.crispembed_mineru_predict = _Fn(self._predict)
        self.crispembed_mineru_extract = _Fn(lambda *args: b'{"profile":"native-basic","capped":false}')
        self.crispembed_mineru_last_markdown = _Fn(lambda *args: b"")
        self.crispembed_mineru_extract_document = _Fn(lambda *args: b'{"profile":"native-full","blocks":[]}')
        self.crispembed_mineru_last_document = _Fn(lambda *args: b"")
        self.crispembed_mineru_last_error = _Fn(lambda *args: self.error)

    def _free(self, ctx):
        self.freed += 1

    def _predict(self, ctx, pointer, width, height, channels, prompt, budget):
        pixels = None
        if pointer:
            pixels = np.ctypeslib.as_array(pointer, shape=(width * height * channels,)).copy().reshape(height, width, channels)
        self.calls.append((pixels, prompt.decode("utf-8"), budget))
        text = self.texts.pop(0) if self.texts else "ok"
        if text is None:
            return None
        return json.dumps({"text": text, "raw_text": text, "eos": self.eos, "capped": not self.eos,
                           "token_ids": [1, 151645] if self.eos else [1], "prompt_ids": [151644]}).encode()


class BindingControls(unittest.TestCase):
    def model(self, native, **kwargs):
        with patch.object(_mineru, "_load_library", return_value=native):
            return CrispMinerU("model.gguf", **kwargs)

    def test_lifetime_and_error(self):
        native = _Native([None])
        model = self.model(native)
        with self.assertRaisesRegex(RuntimeError, "injected native failure"):
            model.predict(None, "text-only")
        model.close()
        model.close()
        self.assertEqual(native.freed, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            model.predict(None, "text-only")

    def test_rgba_drops_alpha_like_pillow(self):
        native = _Native()
        with self.model(native) as model:
            rgba = np.array([[[9, 24, 60, 0], [200, 80, 40, 128]]], dtype=np.uint8)
            model.predict(rgba, "\nText Recognition:", 7)
            np.testing.assert_array_equal(native.calls[0][0], rgba[:, :, :3])
            self.assertEqual(native.calls[0][2], 7)
            self.assertEqual(model.last_trace[0]["width"], 2)

    def test_low_level_preserves_caps_and_text_only(self):
        native = _Native()
        native.eos = False
        with self.model(native) as model:
            result = model.predict(None, "Please merge the next two tables.")
            self.assertTrue(result["capped"])
            self.assertIsNone(native.calls[0][0])
            self.assertEqual(model.native_extract(Image.new("RGB", (28, 28)))["profile"], "native-basic")

    def test_invalid_inputs_do_not_call_native(self):
        native = _Native()
        with self.model(native) as model:
            for budget in (-1, 8193, True, 1.5):
                with self.assertRaises(ValueError):
                    model.predict(None, "prompt", budget)
            for prompt in ("", "bad\0prompt", None):
                with self.assertRaises(ValueError):
                    model.predict(None, prompt)
            with self.assertRaises(ValueError):
                model.predict(np.ones((2, 2, 3), dtype=np.float32), "prompt")
            self.assertEqual(native.calls, [])


@unittest.skipUnless(importlib.util.find_spec("mineru_vl_utils"), "pinned upstream helper is optional")
class OfficialHelperComposition(unittest.TestCase):
    model = BindingControls.model
    def test_actual_helper_rotated_crop_and_markdown(self):
        layout = (
            "<|box_start|>100 200 900 800<|box_end|><|ref_start|>text<|ref_end|><|rotate_right|>"
            "<|box_start|>0 900 1000 1000<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>txt_contd_tgt"
        )
        native = _Native([layout, "Hello", "world"])
        image = Image.fromarray(np.arange(100 * 80 * 3, dtype=np.uint8).reshape(80, 100, 3))
        with self.model(native) as model:
            result = model.two_step_extract(image)
            self.assertEqual(model.to_markdown(result), "Hello world")
            self.assertEqual(len(native.calls), 3)
            np.testing.assert_array_equal(native.calls[0][0], np.asarray(image.resize((1036, 1036), Image.Resampling.BICUBIC)))
            expected_crop = image.crop((10, 16, 90, 64)).rotate(90, expand=True)
            np.testing.assert_array_equal(native.calls[1][0], np.asarray(expected_crop))
            self.assertEqual(native.calls[0][1], "\nLayout Detection:")
            self.assertEqual(native.calls[1][1], "\nText Recognition:")
            self.assertEqual(result[0]["angle"], 90)
            self.assertTrue(result[1]["merge_prev"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_actual_helper_table_and_formula_repairs(self):
        layout = (
            "<|box_start|>0 0 1000 400<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>"
            "<|box_start|>0 500 1000 700<|box_end|><|ref_start|>equation<|ref_end|><|rotate_up|>"
        )
        native = _Native([layout, "<fcel>A<lcel><nl><fcel>B<fcel>C<nl>", "x+y"])
        with self.model(native) as model:
            result = model.two_step_extract(Image.new("RGB", (200, 300), "white"))
            self.assertEqual(result[0]["content"], "<table><tr><td colspan=\"2\">A</td></tr><tr><td>B</td><td>C</td></tr></table>")
            self.assertEqual(result[1]["content"], "\\[\nx+y\n\\]")
            self.assertEqual([c[1] for c in native.calls], ["\nLayout Detection:", "\nTable Recognition:", "\nFormula Recognition:"])

    def test_actual_helper_rejects_cap(self):
        native = _Native(["partial layout"])
        native.eos = False
        with self.model(native) as model:
            with self.assertRaisesRegex(RuntimeError, "did not reach EOS"):
                model.two_step_extract(Image.new("RGB", (30, 30)))

    def test_actual_table_image_mask_and_restore(self):
        from mineru_vl_utils.post_process import table_image_processor

        layout = (
            "<|box_start|>0 0 1000 1000<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>"
            "<|box_start|>100 100 500 500<|box_end|><|ref_start|>image<|ref_end|><|rotate_up|>"
        )
        native = _Native([layout, "<fcel>[AC23]<nl>"])
        image = Image.new("RGB", (200, 300), "white")
        with self.model(native) as model, patch.object(table_image_processor, "_generate_uid", return_value="AC23"):
            result = model.two_step_extract(image)
            self.assertEqual(len(result), 1)
            self.assertIn('<img src="data:image/jpeg;base64,', result[0]["content"])
            self.assertNotIn("_table_image_token_map", result[0])
            self.assertEqual(len(native.calls), 2)
            self.assertFalse(np.array_equal(native.calls[1][0], np.asarray(image)))

    def test_actual_image_analysis(self):
        layout = "<|box_start|>0 0 1000 1000<|box_end|><|ref_start|>image<|ref_end|><|rotate_up|>"
        image_text = "<|class_start|>natural_image<|class_end|><|caption_start|>A diagram<|caption_end|>"
        native = _Native([layout, image_text])
        with self.model(native, image_analysis=True) as model:
            result = model.two_step_extract(Image.new("RGB", (200, 300)))
            self.assertEqual(result[0]["content"], "A diagram")
            self.assertEqual(result[0]["sub_type"], "natural_image")
            self.assertEqual(native.calls[1][1], "\nImage Analysis:")

    def test_cross_page_dependency_missing_fails_before_inference(self):
        from mineru_vl_utils.post_process import cross_page_table

        native = _Native()
        with self.model(native, enable_cross_page_table_merge=True) as model, patch.object(cross_page_table, "_HAS_TABLE_MERGE", False):
            with self.assertRaisesRegex(RuntimeError, "requires the official MinerU table_merge"):
                model.batch_two_step_extract([Image.new("RGB", (200, 300))])
            self.assertEqual(native.calls, [])

    def test_source_drift_fails_before_inference(self):
        native = _Native()
        with self.model(native) as model, patch.object(_mineru, "BLUEPRINT_SOURCE_SHA256", "wrong"):
            with self.assertRaisesRegex(RuntimeError, "differs from the pinned blueprint"):
                model.two_step_extract(Image.new("RGB", (30, 30)))
            self.assertEqual(native.calls, [])


if __name__ == "__main__":
    unittest.main()
