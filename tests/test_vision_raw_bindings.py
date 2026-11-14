#!/usr/bin/env python3
"""Model-free raw buffers through actual Granite/LightOn public C wrappers."""

import ctypes as C
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from test_cc_detect_ownership import extract_function
from test_preprocess_bindings import BINDING

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (("granite_vision", BINDING.CrispGraniteVision),
            ("lightonocr", BINDING.CrispLightOnOCR))


class VisionRawBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which("c++") or shutil.which("g++")
        if not compiler:
            raise unittest.SkipTest("C++ compiler unavailable")
        cls.temp = tempfile.TemporaryDirectory(prefix="vision-raw-interface-")
        cls.addClassCleanup(cls.temp.cleanup)
        source = (ROOT / "src/crispembed.cpp").read_text()
        wrappers = "\n".join(extract_function(source, prefix) for prefix in (
            'extern "C" crispembed_granite_vision_context * crispembed_granite_vision_init(',
            'extern "C" void crispembed_granite_vision_free(',
            'extern "C" const char * crispembed_granite_vision_recognize(',
            'extern "C" crispembed_lightonocr_context * crispembed_lightonocr_init(',
            'extern "C" void crispembed_lightonocr_free(',
            'extern "C" const char * crispembed_lightonocr_recognize('))
        harness = r'''
#include "crispembed.h"
#include <string>
#include <vector>
struct granite_vision_context {};
struct lightonocr_context {};
static granite_vision_context granite;
static lightonocr_context lighton;
static int calls=0, channels_seen=0, width_seen=0, height_seen=0, mode=0;
static std::vector<uint8_t> seen;
static std::string prompt_seen;
static int ce_resolve_threads(int n) { return n; }
static granite_vision_context * granite_vision_init(const char *,int) { return &granite; }
static lightonocr_context * lightonocr_init(const char *,int) { return &lighton; }
static void granite_vision_free(granite_vision_context *) {}
static void lightonocr_free(lightonocr_context *) {}
static const char * recognize(const uint8_t * p,int w,int h,int ch,const char * prompt,int * out_len) {
    ++calls; width_seen=w; height_seen=h; channels_seen=ch;
    seen.assign(p,p+w*h*ch); prompt_seen=prompt?prompt:"";
    const char * text=mode==1?nullptr:mode==2?"":mode==3?"\xff":"\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e \xe2\x82\xac";
    if (out_len) *out_len=text?int(std::char_traits<char>::length(text)):0;
    return text;
}
static const char * granite_vision_recognize(granite_vision_context *,const uint8_t * p,int w,int h,int ch,
                                            const char * prompt,int * out_len) {
    return recognize(p,w,h,ch,prompt,out_len);
}
static const char * lightonocr_recognize_raw(lightonocr_context *,const uint8_t * p,int w,int h,int ch,int * out_len) {
    return recognize(p,w,h,ch,nullptr,out_len);
}
extern "C" void test_reset(int value) { calls=0; mode=value; seen.clear(); prompt_seen.clear(); }
extern "C" int test_calls() { return calls; }
extern "C" int test_width() { return width_seen; }
extern "C" int test_height() { return height_seen; }
extern "C" int test_channels() { return channels_seen; }
extern "C" int test_count() { return int(seen.size()); }
extern "C" int test_byte(int at) { return seen.at(at); }
extern "C" const char * test_prompt() { return prompt_seen.c_str(); }
'''
        cpp = Path(cls.temp.name) / "native.cpp"
        library = Path(cls.temp.name) / "native.so"
        cpp.write_text(harness + wrappers)
        subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-I", str(ROOT / "src"),
                        str(cpp), "-o", str(library)], check=True, timeout=30)
        cls.lib = C.CDLL(str(library))
        cls.lib.test_reset.argtypes = [C.c_int]
        cls.lib.test_byte.argtypes = [C.c_int]
        cls.lib.test_prompt.restype = C.c_char_p

    def create(self, constructor, mode=0):
        self.lib.test_reset(mode)
        with patch.object(BINDING, "_load_library", return_value=self.lib):
            model = constructor("fixture.gguf")
        self.addCleanup(model.__del__)
        return model

    def assert_pixels(self, expected, width, height, channels):
        self.assertEqual(self.lib.test_calls(), 1)
        self.assertEqual((self.lib.test_width(), self.lib.test_height(), self.lib.test_channels()),
                         (width, height, channels))
        self.assertEqual(self.lib.test_count(), len(expected))
        self.assertEqual(bytes(self.lib.test_byte(i) for i in range(len(expected))), expected)

    def test_supported_channels_exact_and_oversized_prefixes(self):
        for _, constructor in FAMILIES:
            for channels in (1, 3, 4):
                pixels = bytes(range(6*channels))
                for raw in (pixels, pixels+b"extra", bytearray(pixels), memoryview(pixels)):
                    with self.subTest(model=constructor.__name__, channels=channels, size=len(raw)):
                        model = self.create(constructor)
                        self.assertEqual(model.recognize_raw(raw, np.int64(3), np.int64(2), channels), "日本語 €")
                        self.assert_pixels(pixels, 3, 2, channels)

    def test_logical_strides_and_typed_views_preserve_bytes_and_input(self):
        for _, constructor in FAMILIES:
            for channels in (1, 3, 4):
                base = np.arange(12*channels, dtype=np.uint8).reshape(2, 6, channels)
                for array in (base[:, ::2], base[::-1, ::-2]):
                    model = self.create(constructor)
                    before = base.copy()
                    model.recognize_raw(memoryview(array), 3, 2, channels)
                    self.assert_pixels(array.tobytes(), 3, 2, channels)
                    np.testing.assert_array_equal(base, before)
                typed = np.arange(3*channels, dtype=np.uint16)
                model = self.create(constructor)
                model.recognize_raw(memoryview(typed), 3, 2, channels)
                self.assert_pixels(typed.tobytes(), 3, 2, channels)

    def test_invalid_metadata_and_short_buffers_never_reach_native(self):
        cases = ((0, 1, 3, b""), (1, 0, 3, b""), (-1, 1, 3, b""), (1, -1, 3, b""),
                 (-1, -1, 3, b""), (2**31, 1, 3, b""), (1, 2**31, 3, b""),
                 (2**30, 1, 3, b""), (2**29, 1, 4, b""), (65536, 65536, 1, b""),
                 (1.5, 1, 3, b"123"), (1, 1.5, 3, b"123"),
                 (1, 1, 0, b""), (1, 1, 2, b"12"), (1, 1, 5, b"12345"), (1, 1, 3.0, b"123"),
                 (3, 2, 1, b"12345"), (3, 2, 3, bytes(17)), (3, 2, 4, bytes(23)))
        for family, constructor in FAMILIES:
            for width, height, channels, raw in cases:
                with self.subTest(model=family, width=width, height=height, channels=channels):
                    model = self.create(constructor)
                    reject = Mock(side_effect=AssertionError("native call before preflight"))
                    with patch.object(self.lib, "crispembed_"+family+"_recognize", reject):
                        with self.assertRaises(ValueError):
                            model.recognize_raw(raw, width, height, channels)
                    reject.assert_not_called()
                    self.assertEqual(self.lib.test_calls(), 0)

    def test_lighton_grayscale_rgb_intermediate_overflow_precedes_buffer_access(self):
        model = self.create(BINDING.CrispLightOnOCR)
        with patch.object(BINDING, "memoryview", create=True,
                          side_effect=AssertionError("buffer touched before geometry guard")):
            with self.assertRaises(ValueError):
                model.recognize_raw(b"", (2**31-1)//3+1, 1, 1)
        self.assertEqual(self.lib.test_calls(), 0)

    def test_output_and_prompt_semantics_are_unchanged(self):
        for _, constructor in FAMILIES:
            for mode, output in ((0, "日本語 €"), (1, ""), (2, "")):
                model = self.create(constructor, mode)
                self.assertEqual(model.recognize_raw(b"123", 1, 1), output)
            model = self.create(constructor, 3)
            with self.assertRaises(UnicodeDecodeError):
                model.recognize_raw(b"123", 1, 1)
            self.assertEqual(self.lib.test_calls(), 1)
        for prompt in (None, "", "日本語 OCR"):
            model = self.create(BINDING.CrispGraniteVision)
            model.recognize_raw(b"123", 1, 1, prompt=prompt)
            self.assertEqual(self.lib.test_prompt(), (prompt or "").encode())


if __name__ == "__main__":
    unittest.main()
