"""Weight-free native binding and actual pinned OXR document-helper controls.

Set OXR_BLUEPRINT_PATH to the pinned OXR source checkout, or install the pinned
helper requirements. Native inference is mocked; these tests are not numerical
or OCR quality evidence. Tests still run the original crop/order/postprocessing.
"""

import asyncio
import ctypes as C
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from crispembed import _oxr


class Function:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class Native:
    def __init__(self, answer=None):
        self.requests = []
        self.freed = []
        self.answer = answer or (lambda request: "recognized")
        self.crispembed_oxr_init = Function(lambda path, threads: 19)
        self.crispembed_oxr_free = Function(self.freed.append)
        self.crispembed_oxr_last_error = Function(lambda _: b"native test error")
        self.crispembed_oxr_predict = Function(self.predict)

    def predict(self, ctx, pixels, width, height, channels, prompt, budget, penalty):
        array = np.ctypeslib.as_array(pixels, shape=(height * width * channels,)).reshape(height, width, channels).copy()
        request = dict(image=array, width=width, height=height, prompt=prompt.decode(),
                       budget=budget, penalty=float(penalty), index=len(self.requests))
        self.requests.append(request)
        answer = self.answer(request)
        if answer is None:
            return None
        if isinstance(answer, str):
            answer = {"text": answer}
        result = dict(text="", raw_text="", input_ids=[1, 2], token_ids=[3, 151645],
                      eos=True, capped=False, finish_reason="stop", inference_ms=1.0)
        result.update(answer)
        return json.dumps(result).encode()


class BindingTests(unittest.TestCase):
    def context(self, native=None, **kwargs):
        native = native or Native()
        with mock.patch.object(_oxr, "_load_library", return_value=native):
            model = _oxr.CrispOXR("model.gguf", **kwargs)
        self.addCleanup(model.close)
        return model, native

    def test_lifetime_and_signature(self):
        model, native = self.context()
        self.assertIs(native.crispembed_oxr_predict.argtypes[-1], C.c_float)
        model.close()
        model.close()
        self.assertEqual(native.freed, [19])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            model.predict(np.zeros((2, 3, 3), dtype=np.uint8))

    def test_rgb_alpha_prompt_and_owned_trace(self):
        model, native = self.context()
        rgba = np.zeros((2, 3, 4), dtype=np.uint8)
        rgba[:, :, :3] = [17, 29, 41]
        result = model.predict(rgba, task="formula", max_new_tokens=43)
        self.assertEqual(native.requests[0]["prompt"], _oxr.PROMPTS["formula"])
        self.assertEqual(native.requests[0]["budget"], 43)
        self.assertEqual(native.requests[0]["image"][0, 0].tolist(), [17, 29, 41])
        result["token_ids"].append(77)
        self.assertNotIn(77, model.last_trace[0]["result"]["token_ids"])

    def test_native_error_and_low_level_caps(self):
        native = Native(lambda _: None)
        model, _ = self.context(native)
        with self.assertRaisesRegex(RuntimeError, "native test error"):
            model.predict(np.zeros((2, 2, 3), dtype=np.uint8))
        native.answer = lambda _: {"text": "partial", "eos": False, "capped": True, "finish_reason": "length"}
        self.assertTrue(model.predict(np.zeros((2, 2, 3), dtype=np.uint8))["capped"])

    def test_reject_invalid_inputs(self):
        model, native = self.context()
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for kwargs in ({"prompt": "\0"}, {"max_new_tokens": True},
                       {"max_new_tokens": -1}, {"repetition_penalty": float("nan")},
                       {"repetition_penalty": 0}, {"task": "imaginary"}):
            with self.assertRaises(ValueError):
                model.predict(image, **kwargs)
        for invalid in (np.zeros((0, 2, 3), dtype=np.uint8),
                        np.zeros((2, 2, 2), dtype=np.uint8), image.astype(np.float32)):
            with self.assertRaises(ValueError):
                model.predict(invalid)
        self.assertEqual(native.requests, [])

    def test_layout_payload_class_order(self):
        payload = _oxr.CrispOXR.layout_response_from_regions([
            [{"type": "Code Block", "bbox": [1, 2, 3, 4], "score": 0.9},
             {"type": "Chemical Structure", "bbox": [2, 3, 4, 5]}], []
        ])
        self.assertEqual(payload["data"]["labels"], [[9, 11], []])
        self.assertEqual(payload["data"]["scores"], [[0.9, 1.0], []])
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _oxr.CrispOXR.layout_response_from_regions([[{"type": "wrong", "bbox": []}]])


def blueprint_path():
    explicit = os.environ.get("OXR_BLUEPRINT_PATH")
    if explicit:
        return explicit
    spec = importlib.util.find_spec("oxr")
    return str(Path(spec.origin).parent) if spec and spec.origin else None


@unittest.skipUnless(blueprint_path(), "pinned OXR document helper not installed")
class BlueprintTests(unittest.TestCase):
    def context(self, native=None, **kwargs):
        kwargs.setdefault("blueprint_path", blueprint_path())
        return BindingTests.context(self, native, **kwargs)

    @staticmethod
    def region(kind="Text", bbox=None):
        return {"type": kind, "bbox": bbox or [0, 0, 12, 12], "score": 0.9}

    def test_private_original_modules_and_source_drift(self):
        sentinel = types.ModuleType("oxr.pipeline.recognition")
        with mock.patch.dict(sys.modules, {"oxr.pipeline.recognition": sentinel}):
            first, _ = self.context()
            second, _ = self.context()
            first._official_helper()
            second._official_helper()
            self.assertIs(sys.modules["oxr.pipeline.recognition"], sentinel)
            self.assertIsNot(first._orchestrator, second._orchestrator)
            self.assertIsNot(first._orchestrator.settings, second._orchestrator.settings)
            prefix = first._blueprint.prefix
            first.close()
            self.assertFalse(any(k.startswith(prefix) for k in sys.modules))
            self.assertIsNotNone(second._orchestrator)
        root = _oxr._source_root(blueprint_path())
        with tempfile.TemporaryDirectory() as directory:
            copy = Path(directory) / "oxr"
            shutil.copytree(root, copy, ignore=shutil.ignore_patterns("__pycache__", "demo"))
            # The skipped demo assets contain no Python in the pinned source.
            (copy / "model" / "prompts.py").write_text("# changed source\n")
            with self.assertRaisesRegex(RuntimeError, "source differs"):
                _oxr._source_root(copy)

    def test_prompts_are_original(self):
        model, _ = self.context()
        model._official_helper()
        prompts = model._blueprint.load("oxr.model.prompts")
        names = dict(text="TEXT", formula="FORMULA", table="TABLE", code="CODE",
                     page="WHOLE_PAGE", chemical_structure="CHEMICAL_STRUCTURE")
        for key, part in names.items():
            self.assertEqual(_oxr.PROMPTS[key], getattr(prompts, "PROMPT_" + part + "_RECOGNITION"))
        self.assertEqual(_oxr.PROMPTS["layout"], prompts.PROMPT_LAYOUT_ANALYSIS)

    def test_crop_rounding_rgb_and_full_serialization(self):
        model, native = self.context()
        image = np.arange(12 * 12 * 3, dtype=np.uint8).reshape(12, 12, 3)
        output = model.process_pages([image], layouts=[[self.region(bbox=[0.5, 1.5, 8.5, 9.5])]])
        self.assertEqual(native.requests[0]["image"].shape, (8, 8, 3))
        np.testing.assert_array_equal(native.requests[0]["image"], image[2:10, 0:8])
        self.assertEqual(output["outputs"]["markdown"], "recognized\n")
        block = output["outputs"]["json"]["pages"][0]["blocks"][0]
        self.assertEqual(block["content"], "recognized")
        self.assertEqual(block["bbox"], [0.5, 1.5, 8.5, 9.5])

    def test_reading_order_and_header_filtering(self):
        def answer(request):
            if "Boxes:\n" in request["prompt"]:
                return "1 0"
            return "left" if request["image"][0, 0, 0] == 10 else "right"
        model, native = self.context(Native(answer))
        image = np.full((24, 24, 3), 10, dtype=np.uint8)
        image[:, 12:] = 30
        regions = [self.region(bbox=[0, 6, 12, 24]),
                   self.region(bbox=[12, 6, 24, 24]),
                   self.region("Page-header", [0, 0, 24, 6])]
        output = model.process_pages([image], layouts=[regions])
        blocks = output["outputs"]["json"]["pages"][0]["blocks"]
        self.assertEqual([b["content"] for b in blocks], ["right", "left"])
        order = next(r for r in native.requests if "Boxes:\n" in r["prompt"])
        self.assertIn("<|box_start|>(0,250),(500,1000)<|box_end|>", order["prompt"])
        self.assertEqual(order["budget"], 16)
        self.assertEqual(len(native.requests), 3)

    def test_length_recovery_matches_original_penalty(self):
        native = Native(lambda r: {"text": "partial", "eos": False, "capped": True,
                                  "finish_reason": "length"} if r["index"] == 0 else "complete")
        model, _ = self.context(native)
        output = model.process_pages([np.zeros((12, 12, 3), dtype=np.uint8)], layouts=[[self.region()]])
        self.assertEqual([r["penalty"] for r in native.requests], [1.0, 1.1])
        self.assertEqual(output["outputs"]["markdown"], "complete\n")
        self.assertEqual(output["metadata"]["crispembed"]["unresolved_capped_requests"], 0)

    def test_final_caps_rejected_or_explicitly_retained(self):
        def answer(_):
            return {"text": "partial", "eos": False, "capped": True, "finish_reason": "length"}
        model, _ = self.context(Native(answer))
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "after upstream recovery"):
            model.process_pages([image], layouts=[[self.region()]])
        model.allow_truncated_content = True
        output = model.process_pages([image], layouts=[[self.region()]])
        self.assertEqual(output["outputs"]["markdown"], "partial\n")
        self.assertEqual(output["metadata"]["crispembed"]["unresolved_capped_requests"], 1)

    def test_table_length_recovery_resizes(self):
        native = Native(lambda r: {"text": "<table>", "eos": False, "capped": True,
                                  "finish_reason": "length"} if r["index"] == 0
                        else "<table><tr><td>done</td></tr></table>")
        model, _ = self.context(native)
        output = model.process_pages([np.zeros((12, 12, 3), dtype=np.uint8)],
                                     layouts=[[self.region("Table", [0, 0, 10, 8])]])
        self.assertEqual([(r["width"], r["height"]) for r in native.requests], [(10, 8), (8, 6)])
        self.assertIn("done", output["outputs"]["markdown"])

    def test_formula_numbering_and_chemical_equation(self):
        native = Native(lambda _: r"$$\ce{H2 + O2 -> H2O}\tag{3}$$")
        model, _ = self.context(native)
        output = model.process_pages([np.zeros((12, 12, 3), dtype=np.uint8)],
                                     layouts=[[self.region("Formula")]])
        block = output["outputs"]["json"]["pages"][0]["blocks"][0]
        self.assertEqual(block["type"], "Chemical Equation")
        self.assertEqual(block["content"], r"\ce{H2 + O2 -> H2O}\tag{3}")
        self.assertIn("(3)", output["outputs"]["markdown"])

    def test_table_image_restoration_uses_original_helper(self):
        def answer(request):
            if "Boxes:\n" in request["prompt"]:
                return "0 1 2"
            return "<table><tr><td><tit>2</td><td><tit>1</td></tr></table>"
        model, native = self.context(Native(answer), table_image_placeholders=True)
        image = np.full((180, 260, 3), 255, dtype=np.uint8)
        image[40:85, 135:235] = [10, 70, 130]
        image[100:145, 135:235] = [40, 100, 160]
        regions = [self.region("Table", [10.4, 10.4, 250.2, 170.2]),
                   self.region("Picture", [135, 40, 235, 85]),
                   self.region("Picture", [135, 100, 235, 145])]
        output = model.process_pages([image], layouts=[regions])
        blocks = output["outputs"]["json"]["pages"][0]["blocks"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["content"].count("data:image/png;base64,"), 2)
        table_request = next(r for r in native.requests if r["prompt"] == _oxr.PROMPTS["table"])
        # Original placeholders are blue in BGR; the model sees correct RGB.
        self.assertTrue(np.any(np.all(table_request["image"] == [0, 0, 255], axis=2)))

    def test_document_decode_and_async_wrapper(self):
        model, native = self.context()
        image = Image.new("RGB", (12, 12), (7, 21, 35))
        data = io.BytesIO()
        image.save(data, format="PNG")
        output = asyncio.run(model.aprocess_document(data.getvalue(), file_name="page.png",
                                                     layouts=[[self.region()]]))
        self.assertEqual(output["metadata"]["file_type"], ".png")
        self.assertEqual(native.requests[0]["image"][0, 0].tolist(), [7, 21, 35])
        async def bad_sync_call():
            with self.assertRaisesRegex(RuntimeError, "active asyncio"):
                model.process_pages([np.zeros((12, 12, 3), dtype=np.uint8)], layouts=[[self.region()]])
        asyncio.run(bad_sync_call())

    def test_missing_layout_and_invalid_layout_fail(self):
        model, native = self.context()
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "supplied layouts"):
            model.process_pages([image])
        with self.assertRaisesRegex(ValueError, "nondegenerate"):
            model.process_pages([image], layouts=[[self.region(bbox=[3, 3, 2, 2])]])
        self.assertEqual(native.requests, [])


if __name__ == "__main__":
    unittest.main()
