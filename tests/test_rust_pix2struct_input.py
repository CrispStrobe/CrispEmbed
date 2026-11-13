#!/usr/bin/env python3
"""Production Rust Pix2Struct input methods against a guarded native RGB sink."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_cc_detect_ownership import extract_function

ROOT = Path(__file__).resolve().parents[1]


def run_consumer(directory, rust_source):
    cxx, rustc = shutil.which("c++") or shutil.which("g++"), shutil.which("rustc")
    if not cxx or not rustc:
        raise unittest.SkipTest("C++ and Rust compilers required for focused native interface test")
    native = (ROOT / "src/crispembed.cpp").read_text()
    wrappers = "\n".join(extract_function(native, prefix) for prefix in (
        'extern "C" const char * crispembed_pix2struct_generate(',
        'extern "C" void crispembed_pix2struct_free_text('))
    harness = r'''
#include "crispembed.h"
#include "pix2struct.h"
#include <climits>
#include <string>
static int calls=0, input_length=0, mode=0, tokens=0;
extern "C" void test_reset(int length, int value) { calls=0; input_length=length; mode=value; }
extern "C" int test_calls() { return calls; }
extern "C" int test_tokens() { return tokens; }
extern "C" const char * pix2struct_generate(pix2struct_context *, const uint8_t * pixels,
                                            int w, int h, int max_tokens) {
    ++calls; tokens=max_tokens;
    // Guard the test process even if a future Rust regression bypasses validation.
    if (w<=0 || h<=0 || w>INT_MAX/h || w*h>INT_MAX/3 || input_length<w*h*3 || mode==1)
        return nullptr;
    static std::string text;
    text="日本語:";
    for (int i=0; i<w*h*3; ++i) {
        if (i) text+=",";
        text+=std::to_string(pixels[i]);
    }
    return text.c_str();
}
extern "C" void pix2struct_free_text(const char *) {} // Preserve the actual legacy borrowed/no-op contract.
'''
    directory = Path(directory)
    cpp, library = directory / "native.cpp", directory / "native.so"
    cpp.write_text(harness + wrappers)
    subprocess.run([cxx, "-std=c++17", "-fPIC", "-shared", "-I", str(ROOT / "src"),
                    str(cpp), "-o", str(library)], check=True, text=True, timeout=30)
    source, binary = directory / "consumer.rs", directory / "consumer"
    source.write_text(rust_source)
    subprocess.run([rustc, "--edition=2021", str(source), "-o", str(binary),
                    "-C", "link-arg=" + str(library),
                    "-C", "link-arg=-Wl,-rpath," + str(directory)],
                   check=True, text=True, timeout=30)
    subprocess.run([str(binary)], check=True, text=True, timeout=10)


def production_methods(source=None, include_rgba=True):
    raw = (ROOT / "crispembed-sys/src/lib.rs").read_text()
    safe = source if source is not None else (ROOT / "crispembed/src/lib.rs").read_text()
    impl = safe[safe.index("impl CrispPix2Struct {"):]
    methods = extract_function(impl, "    pub fn generate(")
    if include_rgba:
        methods += "\n" + extract_function(impl, "    pub fn generate_rgba(")
    context = next(line for line in raw.splitlines() if line.startswith("pub struct Pix2StructContext("))
    declarations = []
    for name in ("crispembed_pix2struct_generate", "crispembed_pix2struct_free_text"):
        start = raw.index("    pub fn " + name + "(")
        declarations.append(raw[start:raw.index(";", start)+1])
    return ("mod crispembed_sys { use std::ffi::{c_char,c_int,c_void};\n#[repr(C)]\n" +
            context + '\nextern "C" {\n' + "\n".join(declarations) + "\n}\n}\n" +
            extract_function(safe, "pub struct CrispPix2Struct {") + "\nimpl CrispPix2Struct {\n" +
            methods + "\n}\n")


class RustPix2StructInputTests(unittest.TestCase):
    def test_actual_methods_rgb_prefix_explicit_rgba_and_invalid_input(self):
        main = r'''
extern "C" { fn test_reset(length:i32,mode:i32); fn test_calls()->i32; fn test_tokens()->i32; }
fn main() {
    let mut model=CrispPix2Struct { ctx:std::ptr::NonNull::dangling().as_ptr() };
    for (w,h,length) in [(0,1,0),(1,0,0),(-1,1,0),(1,-1,0),(-1,-1,1),
                         (i32::MAX,2,0),(65536,65536,0),(i32::MAX/2,1,0),(2,1,5)] {
        let input=vec![0u8;length];
        unsafe { test_reset(length as i32,0); }
        assert!(model.generate(&input,w,h,7).is_none());
        assert!(model.generate_rgba(&input,w,h,7).is_none());
        unsafe { assert_eq!(test_calls(),0,"invalid input reached native code"); }
    }
    let rgb=[10u8,20,30,40,50,60];
    // Every RGB prefix remains RGB, including the ambiguous exact RGBA-sized buffer.
    for extra in [0,1,2,3,6] {
        let mut input=rgb.to_vec(); input.extend(vec![255;extra]);
        unsafe { test_reset(input.len() as i32,0); }
        assert_eq!(model.generate(&input,2,1,7).unwrap(),"日本語:10,20,30,40,50,60");
        unsafe { assert_eq!(test_calls(),1); assert_eq!(test_tokens(),7); }
    }
    // Identical bytes have an explicit channel interpretation at each API.
    let rgba=[10u8,20,30,0,40,50,60,255];
    unsafe { test_reset(8,0); }
    assert_eq!(model.generate(&rgba,2,1,0).unwrap(),"日本語:10,20,30,0,40,50");
    assert_eq!(model.generate_rgba(&rgba,2,1,0).unwrap(),"日本語:10,20,30,40,50,60");
    unsafe { assert_eq!(test_tokens(),0); }
    assert_eq!(rgba,[10,20,30,0,40,50,60,255]);
    for extra in [0,1,4] {
        let mut input=rgba.to_vec(); input.extend(vec![99;extra]);
        unsafe { test_reset(input.len() as i32,0); }
        assert_eq!(model.generate_rgba(&input,2,1,-1).unwrap(),"日本語:10,20,30,40,50,60");
        unsafe { assert_eq!(test_calls(),1); assert_eq!(test_tokens(),-1); }
    }
    // Valid RGB but incomplete RGBA must be rejected by the explicit RGBA path.
    for length in [6,7] {
        unsafe { test_reset(length,0); }
        assert!(model.generate_rgba(&[0u8;7][..length as usize],2,1,7).is_none());
        unsafe { assert_eq!(test_calls(),0); }
    }
    unsafe { test_reset(4,0); }
    assert_eq!(model.generate(&[1,2,3],1,1,256).unwrap(),"日本語:1,2,3");
    assert_eq!(model.generate_rgba(&[1,2,3,128],1,1,256).unwrap(),"日本語:1,2,3");
    unsafe { test_reset(8,1); }
    assert!(model.generate(&rgb,2,1,7).is_none());
    assert!(model.generate_rgba(&rgba,2,1,7).is_none());
    unsafe { assert_eq!(test_calls(),2); }
}
'''
        with tempfile.TemporaryDirectory(prefix="rust-pix2struct-input-") as directory:
            run_consumer(directory, production_methods() + main)


if __name__ == "__main__":
    unittest.main()
