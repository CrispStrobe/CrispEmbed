#!/usr/bin/env python3
"""Run production Rust grayscale methods against tiny guarded C API fixtures."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_cc_detect_ownership import extract_function

ROOT = Path(__file__).resolve().parents[1]


def build_and_run(directory, rust_source):
    cxx = shutil.which("c++") or shutil.which("g++")
    rustc = shutil.which("rustc")
    if not cxx or not rustc:
        raise unittest.SkipTest("C++ and Rust compilers required for focused interface test")
    native = (ROOT / "src/crispembed.cpp").read_text()
    wrappers = "\n".join(extract_function(native, prefix) for prefix in (
        'extern "C" int crispembed_dewarp(',
        'extern "C" int crispembed_tps_auto_dewarp(',
        'extern "C" int crispembed_find_skew(',
        'extern "C" crispembed_ocr_result * crispembed_cc_detect(',
        'extern "C" void crispembed_cc_detect_free('))
    harness = r'''
#include "crispembed.h"
#include "cc_detect.h"
#include <cstdlib>
#include <cstring>
#include <climits>
static int calls = 0, mode = 0, input_length = 0;
extern "C" void test_reset(int length, int value) { calls=0; input_length=length; mode=value; }
extern "C" int test_calls() { return calls; }
static bool input_ok(int w, int h) {
    ++calls;
    return w>0 && h>0 && w<=INT_MAX/h && w*h<=input_length;
}
int dewarp_page(const uint8_t * gray, int w, int h, uint8_t * out, int * ow, int * oh) {
    if (!input_ok(w,h) || mode==1) return -1;
    std::memcpy(out, gray, w*h);
    *ow=w; *oh=h;
    if (mode==2) { *ow=0; *oh=h; }
    if (mode==3) { *ow=-2; *oh=-2; }
    if (mode==4) { *ow=INT_MAX; *oh=2; }
    if (mode==5) { *ow=w+1; *oh=h; }
    if (mode==6) { *ow=2; *oh=2; }
    return 0;
}
int tps_auto_dewarp(const uint8_t * gray, int w, int h, const char *, uint8_t * out) {
    if (!input_ok(w,h) || mode==1) return -1;
    std::memcpy(out, gray, w*h);
    return 0;
}
int find_skew_angle(const uint8_t *, int w, int h, float * angle, float * confidence) {
    if (!input_ok(w,h) || mode==1) return -1;
    *angle=1.5f; *confidence=0.75f;
    return 0;
}
extern "C" cc_text_region * cc_detect_lines(const uint8_t *, int w, int h, int * count) {
    static cc_text_region region = {1,2,3,4};
    *count=0;
    if (!input_ok(w,h) || mode==1) return nullptr;
    *count=1;
    return &region;
}
extern "C" void cc_detect_free(cc_text_region *) {}
'''
    directory = Path(directory)
    cpp, library = directory / "native.cpp", directory / "native.so"
    cpp.write_text(harness + wrappers)
    subprocess.run([cxx, "-std=c++17", "-fPIC", "-shared", "-I", str(ROOT / "src"),
                    str(cpp), "-o", str(library)], check=True, text=True, timeout=30)
    rs, binary = directory / "consumer.rs", directory / "consumer"
    rs.write_text(rust_source)
    subprocess.run([rustc, "--edition=2021", str(rs), "-o", str(binary),
                    "-C", "link-arg=" + str(library),
                    "-C", "link-arg=-Wl,-rpath," + str(directory)],
                   check=True, text=True, timeout=30)
    subprocess.run([str(binary)], check=True, text=True, timeout=10)


def rust_methods():
    raw = (ROOT / "crispembed-sys/src/lib.rs").read_text()
    safe = (ROOT / "crispembed/src/lib.rs").read_text()
    result = extract_function(raw, "pub struct CrispembedOcrResult {")
    region = extract_function(safe, "pub struct OcrResult {")
    region += "\n" + next(line for line in safe.splitlines() if line.startswith("pub type OcrRegion ="))
    helper = extract_function(safe, "fn grayscale_image_len(")
    methods = "\n".join(extract_function(safe, prefix) for prefix in (
        "pub fn dewarp(", "pub fn tps_auto_dewarp(", "pub fn find_skew(", "pub fn cc_detect("))
    declarations = []
    for name in ("crispembed_dewarp", "crispembed_tps_auto_dewarp", "crispembed_find_skew",
                 "crispembed_cc_detect", "crispembed_cc_detect_free"):
        start = raw.index("    pub fn " + name + "(")
        declarations.append(raw[start:raw.index(";", start) + 1])
    return ("mod crispembed_sys { use std::ffi::{c_char,c_int,c_float};\n#[repr(C)]\n" + result +
            '\nextern "C" {\n' + "\n".join(declarations) + "\n}\n}\n" + region + "\n" +
            helper + "\n" + methods)


class RustGrayscaleSafetyTests(unittest.TestCase):
    def test_actual_methods_validate_slices_and_preserve_valid_native_behavior(self):
        main = r'''
extern "C" { fn test_reset(length:i32, mode:i32); fn test_calls()->i32; }
fn main() {
    let inputs = [(0,1,0), (1,0,0), (-1,1,0), (1,-1,0), (-1,-1,1),
                  (i32::MAX,2,0), (65536,65536,0), (3,2,5)];
    for (w,h,length) in inputs {
        let gray = vec![0u8; length];
        unsafe { test_reset(length as i32,0); }
        assert!(dewarp(&gray,w,h).is_err(), "dewarp accepted invalid input");
        assert!(tps_auto_dewarp(&gray,w,h,"model.gguf").is_err(), "tps accepted invalid input");
        assert!(find_skew(&gray,w,h).is_err(), "skew accepted invalid input");
        assert!(cc_detect(&gray,w,h).is_empty(), "CC accepted invalid input");
        unsafe { assert_eq!(test_calls(),0, "invalid buffer reached native code"); }
    }
    for gray in [&[1u8,2,3,4,5,6][..], &[1u8,2,3,4,5,6,255,254][..]] {
        unsafe { test_reset(gray.len() as i32,0); }
        let before = gray.to_vec();
        let (image,w,h) = dewarp(gray,3,2).unwrap();
        assert_eq!((w,h),(3,2));
        assert_eq!(image,&gray[..6]);
        assert_eq!(tps_auto_dewarp(gray,3,2,"model.gguf").unwrap(),&gray[..6]);
        assert_eq!(find_skew(gray,3,2).unwrap(),(1.5,0.75));
        let regions = cc_detect(gray,3,2);
        assert_eq!(regions.len(),1);
        assert_eq!((regions[0].x,regions[0].y,regions[0].w,regions[0].h),(1.0,2.0,3.0,4.0));
        assert_eq!(gray,&before);
        unsafe { assert_eq!(test_calls(),4); }
    }
    let gray = [1u8,2,3,4,5,6];
    unsafe { test_reset(6,1); }
    assert!(dewarp(&gray,3,2).is_err());
    assert!(tps_auto_dewarp(&gray,3,2,"model.gguf").is_err());
    assert!(find_skew(&gray,3,2).is_err());
    assert!(cc_detect(&gray,3,2).is_empty());
    unsafe { assert_eq!(test_calls(),4); }
    for mode in [2,3,4,5] {
        unsafe { test_reset(6,mode); }
        assert!(dewarp(&gray,3,2).is_err(), "invalid native output dimensions accepted");
    }
    unsafe { test_reset(6,6); }
    let (small,w,h) = dewarp(&gray,3,2).unwrap();
    assert_eq!((w,h),(2,2));
    assert_eq!(small,gray[..4]);
}
'''
        with tempfile.TemporaryDirectory(prefix="rust-gray-interface-") as directory:
            build_and_run(directory, rust_methods() + main)


if __name__ == "__main__":
    unittest.main()
