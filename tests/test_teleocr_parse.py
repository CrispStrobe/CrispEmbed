#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Weight-free parser/geometry tests; these do not establish neural-model parity."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("teleocr_parse", ROOT / "tools/teleocr_parse.py")
parser = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = parser
SPEC.loader.exec_module(parser)


class LayoutTests(unittest.TestCase):
    def test_reading_order_and_orientation(self):
        blocks, warnings = parser.parse_layout(
            "<box:900 700 200 100><label:Title><left>\n"
            "<box:010 020 400 020 350 600 010 500><label:text><up>")
        self.assertEqual(warnings, [])
        self.assertEqual([b.type for b in blocks], ["title", "text"])
        self.assertEqual(blocks[0].bbox, [.2, .1, .9, .7])
        self.assertEqual(blocks[0].angle, 270)
        self.assertEqual(blocks[1].bbox, [.01, .02, .4, .6])
        self.assertEqual(len(blocks[1].polygon), 8)

    def test_bad_boxes_are_visible(self):
        blocks, warnings = parser.parse_layout(
            "junk\n<box:1 2 3><label:text><up>\n"
            "<box:0 0 1001 200><label:text><up>\n"
            "<box:0 0 100 200><label:alien><up>\n"
            "<box:2 2 2 3><label:text><up>")
        self.assertFalse(blocks)
        self.assertEqual(len(warnings), 5)

    def test_block_limit(self):
        line = "<box:0 0 1000 1000><label:text><up>\n"
        with self.assertRaisesRegex(ValueError, "block limit"):
            parser.parse_layout(line * 2, max_blocks=1)

    def test_polygon_mask(self):
        block = parser.Block("text", [0, 0, 1, 1], [0, 0, 1, 0, 0, 1], 0)
        crop = np.asarray(parser.crop_block(Image.new("RGB", (56, 56), "white"), block))
        self.assertTrue((crop[4, 4] == 255).all())
        self.assertTrue((crop[50, 50] == 0).all())

    def test_rotation_is_counterclockwise(self):
        image = Image.new("RGB", (28, 56), "white")
        image.paste((255, 0, 0), (0, 0, 28, 14))
        crop = parser.crop_block(image, parser.Block("text", [0, 0, 1, 1], angle=90))
        self.assertEqual(crop.size, (56, 28))
        self.assertEqual(crop.getpixel((1, 1)), (255, 0, 0))
        self.assertEqual(crop.getpixel((54, 1)), (255, 255, 255))

    def test_extreme_aspect_and_small_crop(self):
        image = parser.resize_by_need(Image.new("RGB", (1000, 1), "black"))
        self.assertEqual(image.size, (1400, 28))
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(parser.resize_by_need(Image.new("RGB", (5, 10))).size, (28, 56))


class TableTests(unittest.TestCase):
    def test_horizontal_vertical_spans(self):
        source = "<fcel>Header<lcel><nl><fcel>left<fcel>right<nl><ucel><fcel>bottom<nl>"
        self.assertEqual(parser.otsl_to_html(source),
                         '<table><tr><td colspan="2">Header</td></tr>'
                         '<tr><td rowspan="2">left</td><td>right</td></tr>'
                         '<tr><td>bottom</td></tr></table>')

    def test_rectangle_span(self):
        self.assertEqual(parser.otsl_to_html("<fcel>A<lcel><nl><ucel><xcel><nl>"),
                         '<table><tr><td rowspan="2" colspan="2">A</td></tr><tr></tr></table>')

    def test_escaping_and_ragged_rows(self):
        self.assertEqual(parser.otsl_to_html("<fcel>x<y & z<ecel><nl><fcel>next<nl>"),
                         '<table><tr><td>x&lt;y &amp; z</td><td></td></tr>'
                         '<tr><td>next</td><td></td></tr></table>')

    def test_orphan_and_overlap_rejected(self):
        for source in ("<lcel><nl>", "<fcel>A<lcel><nl><ucel><fcel>B<nl>", "no table here"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                parser.otsl_to_html(source)

    def test_raw_html_has_no_active_attributes(self):
        self.assertEqual(parser.otsl_to_html('<table><tr><td onclick="alert(1)">x &lt; y</td></tr></table>'),
                         '<table><tr><td>x &lt; y</td></tr></table>')
        with self.assertRaises(ValueError):
            parser.otsl_to_html('<table><script>bad()</script></table>')

    def test_bad_table_retains_original(self):
        warnings = []
        block = parser.Block("table", [0, 0, 1, 1], raw="<ucel>")
        parser.post_process(block, warnings)
        self.assertEqual(block.raw, "<ucel>")
        self.assertIn("<ucel>", block.content)
        self.assertTrue(warnings)


class OutputTests(unittest.TestCase):
    def test_equations(self):
        self.assertEqual(parser.post_equation(r"\[ E=mc^2 \]"), "$$E=mc^2$$")
        self.assertEqual(parser.post_equation("$$E=mc^2(2.1)$$"), "$E=mc^2$ (2.1)")
        self.assertEqual(parser.post_equation(r"E=mc^2\eqno(2.1)"), r"$$E=mc^2\eqno(2.1)$$")
        self.assertEqual(parser.post_equation("x tag{3}"), r"$$x \tag{3}$$")

    def test_inline_math_and_currency(self):
        self.assertEqual(parser.post_text("a$$ x $$b"), "a $x$ b")
        self.assertEqual(parser.post_text(r"cost \$12"), r"cost \$12")
        self.assertEqual(parser.post_text("unbalanced $ formula"), "unbalanced $ formula")

    def test_code_language_and_fence(self):
        self.assertEqual(parser.split_code("```\n<_Python_>print(1)\n```"), ("python", "print(1)"))
        text = parser.code_fence("contains ``` marker", "txt")
        self.assertTrue(text.startswith("````txt\n"))
        self.assertTrue(text.endswith("\n````"))

    def test_reading_order_and_page_furniture(self):
        blocks = [parser.Block("header", [0, 0, 1, .1], content="Header"),
                  parser.Block("title", [0, .1, 1, .2], content="Title"),
                  parser.Block("text", [0, .2, 1, .9], content="Body")]
        self.assertEqual(parser.to_markdown(blocks), "# Title\n\nBody")
        self.assertEqual(parser.to_markdown(blocks, True), "Header\n\n# Title\n\nBody")

    def test_generation_presets(self):
        self.assertEqual(parser.generation_params("layout"),
                         {"repetition_penalty": 1.0, "presence_penalty": 0.0,
                          "frequency_penalty": 0.0, "no_repeat_ngram": 100})
        self.assertEqual(parser.generation_params("table")["frequency_penalty"], .005)
        self.assertEqual(parser.generation_params("text")["frequency_penalty"], .05)

    def test_orchestration_task_contract(self):
        engine = object.__new__(parser.TeleocrParser)
        engine.max_tokens, engine.max_page_pixels, engine.max_blocks = 16, 4_500_000, 512
        calls = []
        outputs = iter([
            ("<box:0 0 1000 500><label:title><up>\n<box:0 500 1000 1000><label:table><up>", 15, .9),
            ("A  Title", 3, .9),
            ("<fcel>A<fcel>B<nl>", 16, .8),
        ])

        def generate(image, prompt, kind):
            calls.append((image.size, prompt, kind))
            return next(outputs)

        engine._generate = generate
        page = engine.parse(Image.new("RGB", (112, 112), "white"))
        self.assertEqual(calls[0], ((1036, 1036), "\nAnalyze the image layout.", "layout"))
        self.assertEqual(calls[1], ((112, 56), parser.TEXT_PROMPT, "title"))
        self.assertEqual(calls[2][1], parser.PROMPTS["table"])
        self.assertIn("# A Title", page["markdown"])
        self.assertIn('<table border="1">', page["markdown"])
        self.assertTrue(page["blocks"][1]["token_cap_reached"])
        self.assertEqual(page["stats"]["generated_tokens"], 34)
        self.assertTrue(page["warnings"])

    def test_empty_layout_is_error(self):
        engine = object.__new__(parser.TeleocrParser)
        engine.max_tokens, engine.max_page_pixels, engine.max_blocks = 16, 4_500_000, 512
        engine._generate = lambda *args: ("invalid layout", 3, .9)
        with self.assertRaisesRegex(ValueError, "no valid layout blocks"):
            engine.parse(Image.new("RGB", (56, 56), "white"))


if __name__ == "__main__":
    unittest.main()
