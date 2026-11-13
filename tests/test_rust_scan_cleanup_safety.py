#!/usr/bin/env python3
"""Production Rust scan-cleanup methods with actual C wrappers and guarded stubs."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_cc_detect_ownership import extract_function

ROOT = Path(__file__).resolve().parents[1]


def native_fixture(directory):
    cxx = shutil.which("c++") or shutil.which("g++")
    if not cxx:
        raise unittest.SkipTest("C++ compiler required for focused interface test")
    native = (ROOT / "src/crispembed.cpp").read_text()
    wrappers = "\n".join(extract_function(native, prefix) for prefix in (
        'static scan_cleanup_params to_cleanup(',
        'extern "C" crispembed_scan_cleanup_params crispembed_scan_cleanup_defaults(',
        'extern "C" void * crispembed_scan_cleanup_init(',
        'extern "C" void crispembed_scan_cleanup_free(',
        'extern "C" int crispembed_scan_cleanup_process(',
        'extern "C" void crispembed_scan_cleanup_free_image(',
        'extern "C" int crispembed_scan_cleanup_detect_page_split(',
        'extern "C" int crispembed_scan_cleanup_content_bbox('))
    harness = r'''
#include "crispembed.h"
#include "scan_cleanup.h"
#include <climits>
#include <cstdlib>
#include <cstring>
static int calls=0, defaults_calls=0, init_calls=0, image_frees=0, length=0, mode=0;
static int model_null=0, captured_param=0;
static float captured_float=0;
static uint8_t * outstanding=nullptr;
static size_t allocation_size=0;
extern "C" void test_reset(int input_length,int value) {
    if (outstanding) std::abort();
    calls=defaults_calls=init_calls=image_frees=0; length=input_length; mode=value;
}
extern "C" int test_calls() { return calls; }
extern "C" int test_defaults_calls() { return defaults_calls; }
extern "C" int test_init_calls() { return init_calls; }
extern "C" int test_image_frees() { return image_frees; }
extern "C" int test_live() { return outstanding!=nullptr; }
extern "C" int test_model_null() { return model_null; }
extern "C" int test_param() { return captured_param; }
extern "C" float test_param_float() { return captured_float; }
extern "C" size_t test_params_size() { return sizeof(crispembed_scan_cleanup_params); }
static bool input_ok(int w,int h,int ch) {
    ++calls;
    return w>0 && h>0 && (ch==1 || ch==3 || ch==4) && w<=INT_MAX/h &&
           w*h<=INT_MAX/ch && w*h<=INT_MAX/3 && length>=w*h*ch;
}
int ce_resolve_threads(int threads) { return threads; }
extern "C" scan_cleanup_params scan_cleanup_defaults() {
    ++defaults_calls;
    scan_cleanup_params params={}; params.deskew=7; params.sauvola_k=0.5f; return params;
}
extern "C" scan_cleanup_ctx * scan_cleanup_init(const char * path,int) {
    ++init_calls; model_null=path==nullptr;
    return reinterpret_cast<scan_cleanup_ctx *>(1);
}
extern "C" void scan_cleanup_free(scan_cleanup_ctx *) {}
extern "C" int scan_cleanup_process(scan_cleanup_ctx *,const uint8_t *,int w,int h,int ch,
                                     scan_cleanup_params params,uint8_t ** out,int * ow,int * oh) {
    if (!input_ok(w,h,ch) || mode==1) return -1;
    captured_param=params.deskew; captured_float=params.sauvola_k;
    *ow=w; *oh=h;
    if (mode==3) *ow=0;
    if (mode==4) { *ow=-2; *oh=-2; }
    if (mode==5) { *ow=INT_MAX; *oh=2; }
    if (mode==6) { *ow=5; *oh=4; }
    if (mode==7) return 0; // success code with null image
    if (mode==8) { *ow=INT_MAX/3+1; *oh=1; }
    allocation_size=mode==6 ? 60 : size_t(w*h*3);
    outstanding=static_cast<uint8_t *>(std::malloc(allocation_size));
    for (size_t i=0;i<allocation_size;++i) outstanding[i]=uint8_t(i);
    *out=outstanding;
    return mode==2 ? -1 : 0; // reported failure with an allocated image
}
extern "C" void scan_cleanup_free_image(uint8_t * ptr) {
    if (!ptr) return;
    if (ptr!=outstanding) std::abort();
    ++image_frees; std::memset(ptr,255,allocation_size); std::free(ptr); outstanding=nullptr;
}
extern "C" int scan_cleanup_detect_page_split(const uint8_t *,int w,int h,int ch) {
    if (!input_ok(w,h,ch) || mode==1) return -1;
    return w/2;
}
extern "C" int scan_cleanup_content_bbox(const uint8_t *,int w,int h,int ch,int * x0,int * y0,int * x1,int * y1) {
    if (!input_ok(w,h,ch) || mode==1) return -1;
    *x0=*y0=0; *x1=w; *y1=h; return 0;
}
'''
    directory = Path(directory)
    cpp, library = directory / "native.cpp", directory / "native.so"
    cpp.write_text(harness + wrappers)
    subprocess.run([cxx, "-std=c++17", "-fPIC", "-shared", "-I", str(ROOT / "src"),
                    str(cpp), "-o", str(library)], check=True, text=True, timeout=30)
    return library


def run_consumer(directory, rust_source):
    rustc = shutil.which("rustc")
    if not rustc:
        raise unittest.SkipTest("Rust compiler required for focused interface test")
    library = native_fixture(directory)
    directory = Path(directory)
    source, binary = directory / "consumer.rs", directory / "consumer"
    source.write_text(rust_source)
    subprocess.run([rustc, "--edition=2021", str(source), "-o", str(binary),
                    "-C", "link-arg=" + str(library),
                    "-C", "link-arg=-Wl,-rpath," + str(directory)],
                   check=True, text=True, timeout=30)
    subprocess.run([str(binary)], check=True, text=True, timeout=10)


def production_methods(source=None, include_helper=True):
    raw = (ROOT / "crispembed-sys/src/lib.rs").read_text()
    safe = source if source is not None else (ROOT / "crispembed/src/lib.rs").read_text()
    declarations = []
    for name in ("defaults", "init", "free", "process", "free_image", "detect_page_split", "content_bbox"):
        start = raw.index("    pub fn crispembed_scan_cleanup_" + name + "(")
        declarations.append(raw[start:raw.index(";", start)+1])
    helper = extract_function(safe, "fn scan_cleanup_input_count(") if include_helper else ""
    return ("mod crispembed_sys {use std::ffi::{c_void,c_char,c_int,c_float};\n#[repr(C)]\n" +
            extract_function(raw, "pub struct ScanCleanupParams {") + '\nextern "C" {\n' +
            "\n".join(declarations) + "\n}\n}\n" + helper + "\n" +
            extract_function(safe, "pub struct CrispScanCleanup {") + "\n" +
            extract_function(safe, "impl CrispScanCleanup {") + "\n" +
            extract_function(safe, "impl Drop for CrispScanCleanup {"))


class RustScanCleanupSafetyTests(unittest.TestCase):
    def test_actual_class_input_sizes_params_outputs_and_path_errors(self):
        main = r'''
extern "C" {
 fn test_reset(length:i32,mode:i32); fn test_calls()->i32; fn test_defaults_calls()->i32;
 fn test_init_calls()->i32; fn test_image_frees()->i32; fn test_live()->i32;
 fn test_model_null()->i32; fn test_param()->i32; fn test_param_float()->f32; fn test_params_size()->usize;
}
fn params()->crispembed_sys::ScanCleanupParams { unsafe {std::mem::zeroed()} }
fn main() {
 unsafe {test_reset(0,0);}
 assert!(CrispScanCleanup::new(Some("bad\0path"),4).is_err());
 unsafe {assert_eq!(test_init_calls(),0);}
 let cleanup=CrispScanCleanup::new(None,4).unwrap();
 unsafe {
  assert_eq!(test_model_null(),1);
  assert_eq!(std::mem::size_of::<crispembed_sys::ScanCleanupParams>(),test_params_size());
 }
 assert!(CrispScanCleanup::new(Some("model.gguf"),4).is_ok());
 unsafe {assert_eq!(test_model_null(),0);}
 for (w,h,ch,len) in [(0,1,3,0),(1,0,3,0),(-1,1,3,0),(-1,-1,3,3),
                      (1,1,0,0),(1,1,2,2),(1,1,5,5),(3,2,3,17),
                      (i32::MAX,2,1,0),(i32::MAX/2,1,1,0),(i32::MAX/4+1,1,4,0)] {
  let bytes=vec![0u8;len]; unsafe {test_reset(len as i32,0);}
  assert!(cleanup.process(&bytes,w,h,ch).is_err());
  assert!(cleanup.process_with_params(&bytes,w,h,ch,params()).is_err());
  assert!(cleanup.detect_page_split(&bytes,w,h,ch).is_none());
  assert!(cleanup.content_bbox(&bytes,w,h,ch).is_none());
  unsafe {assert_eq!(test_calls(),0); assert_eq!(test_defaults_calls(),0);}
 }
 for ch in [1,3,4] {
  for extra in [0,5] {
   let bytes=vec![42u8;6*ch as usize+extra]; unsafe {test_reset(bytes.len() as i32,0);}
   let (pixels,w,h)=cleanup.process(&bytes,3,2,ch).unwrap();
   assert_eq!((w,h),(3,2)); assert_eq!(pixels,(0u8..18).collect::<Vec<_>>());
   unsafe {assert_eq!(test_param(),7); assert_eq!(test_param_float(),0.5); assert_eq!(test_image_frees(),1);}
   let mut custom=params(); custom.deskew=19; custom.sauvola_k=0.375;
   assert!(cleanup.process_with_params(&bytes,3,2,ch,custom).is_ok());
   unsafe {assert_eq!(test_param(),19); assert_eq!(test_param_float(),0.375); assert_eq!(test_image_frees(),2);}
   assert_eq!(cleanup.detect_page_split(&bytes,3,2,ch),Some(1));
   assert_eq!(cleanup.content_bbox(&bytes,3,2,ch),Some((0,0,3,2)));
   assert!(bytes.iter().all(|v|*v==42)); unsafe {assert_eq!(test_calls(),4); assert_eq!(test_live(),0);}
  }
 }
 let bytes=[0u8;18];
 for mode in [1,2,3,4,5,7,8] {
  unsafe {test_reset(18,mode);}
  assert!(cleanup.process_with_params(&bytes,3,2,3,params()).is_err());
  unsafe {assert_eq!(test_image_frees(),if mode==1 || mode==7 {0} else {1}); assert_eq!(test_live(),0);}
 }
 unsafe {test_reset(18,6);}
 let (larger,w,h)=cleanup.process_with_params(&bytes,3,2,3,params()).unwrap();
 assert_eq!((w,h),(5,4)); assert_eq!(larger,(0u8..60).collect::<Vec<_>>());
 unsafe {assert_eq!(test_image_frees(),1); assert_eq!(test_live(),0); test_reset(18,1);}
 assert_eq!(cleanup.detect_page_split(&bytes,3,2,3),None);
 assert_eq!(cleanup.content_bbox(&bytes,3,2,3),None);
}
'''
        with tempfile.TemporaryDirectory(prefix="rust-scan-cleanup-") as directory:
            run_consumer(directory, production_methods() + main)


if __name__ == "__main__":
    unittest.main()
