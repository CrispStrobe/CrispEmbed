#!/usr/bin/env python3
"""Actual safe Rust vision OCR methods against the existing public C fixture."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_cc_detect_ownership import extract_function
import test_vision_raw_bindings as vision_fixture

ROOT = Path(__file__).resolve().parents[1]


def production_methods():
    safe = (ROOT / "crispembed/src/lib.rs").read_text()
    raw = (ROOT / "crispembed-sys/src/lib.rs").read_text()
    contexts, declarations, classes = [], [], []
    for context, name, family in (("GraniteVisionContext", "CrispGraniteVision", "granite_vision"),
                                  ("LightOnOcrContext", "CrispLightOnOcr", "lightonocr")):
        start = raw.index("pub struct " + context + "(")
        contexts.append("#[repr(C)]\n" + raw[start:raw.index(";", start)+1])
        for suffix in ("init", "free", "recognize"):
            start = raw.index("    pub fn crispembed_" + family + "_" + suffix + "(")
            declarations.append(raw[start:raw.index(";", start)+1])
        classes.extend(extract_function(safe, prefix) for prefix in (
            "pub struct " + name + " {", "impl " + name + " {", "impl Drop for " + name + " {"))
    return ("use std::ffi::{CStr,CString};\nmod crispembed_sys { use std::ffi::{c_char,c_int,c_void};\n" +
            "\n".join(contexts) + '\nextern "C" {\n' + "\n".join(declarations) + "\n}\n}\n" +
            extract_function(safe, "fn vision_image_len(") + "\n" + "\n".join(classes))


class RustVisionRawSafetyTests(unittest.TestCase):
    def test_actual_methods_validate_before_ffi_and_preserve_valid_results(self):
        rustc = shutil.which("rustc")
        if not rustc:
            self.skipTest("Rust compiler unavailable")
        main = r'''
extern "C" {
 fn test_reset(mode:i32); fn test_input_length(length:i32); fn test_calls()->i32;
 fn test_count()->i32; fn test_byte(index:i32)->i32; fn test_channels()->i32;
 fn test_width()->i32; fn test_height()->i32; fn test_prompt()->*const std::ffi::c_char;
}
fn reset(length:usize,mode:i32) { unsafe { test_reset(mode); test_input_length(length as i32); } }
fn check_pixels(input:&[u8],count:usize,channels:i32) {
 unsafe {
  assert_eq!(test_calls(),1); assert_eq!(test_count(),count as i32);
  assert_eq!((test_width(),test_height(),test_channels()),(3,2,channels));
  for i in 0..count { assert_eq!(test_byte(i as i32),input[i] as i32); }
 }
}
fn main() {
 let mut granite=CrispGraniteVision::new("fixture.gguf",0).unwrap();
 let mut lighton=CrispLightOnOcr::new("fixture.gguf",0).unwrap();
 // Both safe methods must reject without even entering the guarded lower sink.
 for (w,h,ch,len) in [(0,1,3,0),(1,0,3,0),(-1,1,3,0),(1,-1,3,0),(-1,-1,3,3),
                      (i32::MAX,2,1,0),(65536,65536,1,0),(i32::MAX/2,1,3,0),
                      (i32::MAX/3+1,1,3,0),(i32::MAX/4+1,1,4,0),
                      (1,1,0,0),(1,1,2,2),(1,1,5,5),(1,1,-1,0),
                      (3,2,1,5),(3,2,3,17),(3,2,4,23)] {
  let input=vec![0u8;len];
  reset(len,0); assert!(granite.recognize(&input,w,h,ch,None).is_none());
  unsafe { assert_eq!(test_calls(),0); }
  reset(len,0); assert!(lighton.recognize(&input,w,h,ch).is_none());
  unsafe { assert_eq!(test_calls(),0); }
 }
 // Pure count boundaries avoid allocating giant slices. This checks index bounds,
 // not every floating-point resize/intermediate operation in native inference.
 assert_eq!(vision_image_len(i32::MAX,1,1,false),Some(i32::MAX as usize));
 assert_eq!(vision_image_len(i32::MAX/3,1,1,true),Some((i32::MAX/3) as usize));
 assert_eq!(vision_image_len(i32::MAX/3+1,1,1,true),None);
 assert_eq!(vision_image_len(i32::MAX/3,1,3,false),Some((i32::MAX/3*3) as usize));
 assert_eq!(vision_image_len(i32::MAX/4,1,4,true),Some((i32::MAX/4*4) as usize));
 for ch in [1,3,4] {
  let count=6*ch as usize;
  let data:Vec<u8>=(0..count+5).map(|i|i as u8).collect(); let before=data.clone();
  for len in [count,count+1,count+5] {
   let input=&data[..len];
   reset(len,0);
   let text=granite.recognize(input,3,2,ch,Some("日本語 OCR")).unwrap();
   assert_eq!(text,"日本語 €"); check_pixels(input,count,ch);
   unsafe { assert_eq!(CStr::from_ptr(test_prompt()).to_str().unwrap(),"日本語 OCR"); }
   reset(len,0); assert_eq!(lighton.recognize(input,3,2,ch).as_deref(),Some("日本語 €"));
   check_pixels(input,count,ch);
   assert_eq!(text,"日本語 €"); // Owned copy remains usable after subsequent native calls.
  }
  assert_eq!(data,before);
 }
 for prompt in [None,Some(""),Some("embedded\0NUL")] {
  reset(3,0); assert!(granite.recognize(b"123",1,1,3,prompt).is_some());
  unsafe { assert_eq!(CStr::from_ptr(test_prompt()).to_bytes(),b""); }
 }
 for (mode,expected) in [(0,Some("日本語 €")),(1,None),(2,Some("")),(3,None)] {
  reset(3,mode); assert_eq!(granite.recognize(b"123",1,1,3,None).as_deref(),expected);
  reset(3,mode); assert_eq!(lighton.recognize(b"123",1,1,3).as_deref(),expected);
 }
 assert!(CrispGraniteVision::new("bad\0path",0).is_err());
 assert!(CrispLightOnOcr::new("bad\0path",0).is_err());
 println!("Rust Granite/LightOn raw image contract controls passed");
}
'''
        with tempfile.TemporaryDirectory(prefix="rust-vision-raw-") as directory:
            library = vision_fixture.VisionRawBindingTests.build_fixture(directory)
            source, binary = Path(directory)/"consumer.rs", Path(directory)/"consumer"
            source.write_text(production_methods()+main)
            subprocess.run([rustc, "--edition=2021", str(source), "-o", str(binary),
                            "-C", "link-arg="+str(library),
                            "-C", "link-arg=-Wl,-rpath,"+str(directory)],
                           check=True, timeout=30)
            subprocess.run([str(binary)], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
