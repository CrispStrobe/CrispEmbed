#!/usr/bin/env python3
"""Actual Rust DAT process using the existing native DAT/NN ownership fixture."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import test_dat_tile_blending as dat_fixture
from test_cc_detect_ownership import extract_function

ROOT = Path(__file__).resolve().parents[1]


def production_methods():
    safe = (ROOT / "crispembed/src/lib.rs").read_text()
    raw = (ROOT / "crispembed-sys/src/lib.rs").read_text()
    declarations = []
    for name in ("init", "free", "process", "free_image"):
        start = raw.index("    pub fn crispembed_dat_sr_"+name+"(")
        declarations.append(raw[start:raw.index(";",start)+1])
    return ('use std::ffi::CString;\nmod crispembed_sys {use std::ffi::{c_void,c_char,c_int};\n'
            'extern "C" {\n'+"\n".join(declarations)+"\n}}\n" +
            "\n".join(extract_function(safe,prefix) for prefix in (
                "fn vision_image_len(", "pub struct CrispDatSr {", "impl CrispDatSr {", "impl Drop for CrispDatSr {")))


class RustDatBufferSafetyTests(unittest.TestCase):
    def test_actual_safe_method_input_geometry_output_owner_and_gate(self):
        rustc = shutil.which("rustc")
        if not rustc:
            self.skipTest("Rust compiler unavailable")
        main = r'''
extern "C" {
 fn test_context(scale:i32,split:i32)->*mut std::ffi::c_void;
 fn test_input_length(length:i32); fn test_entries()->i32; fn test_calls()->i32;
 fn test_result(mode:i32,width:i32,height:i32); fn test_fail_alloc(value:i32);
 fn test_live()->i32; fn test_allocations()->i32; fn test_frees()->i32;
 fn test_image_frees()->i32; fn test_context_frees()->i32;
 fn test_tile_width()->i32; fn test_tile_height()->i32;
}
fn reset(scale:i32,length:usize) { unsafe {test_context(scale,32);test_input_length(length as i32);} }
fn released(api_frees:i32) {
 unsafe {
  assert_eq!(test_live(),0); assert_eq!(test_allocations(),test_frees());
  assert_eq!(test_image_frees(),api_frees);
 }
}
fn main() {
 std::env::remove_var("CRISPEMBED_DAT_POSITIVE_TILE_BLEND");
 let model=CrispDatSr::new("fixture.gguf",0).unwrap();
 for (w,h,len) in [(0,1,0),(1,0,0),(-1,1,0),(1,-1,0),(-1,-1,3),
                   (i32::MAX,2,0),(65536,65536,0),(i32::MAX/3+1,1,0),(3,2,17)] {
  let input=vec![0u8;len]; reset(2,len);
  assert!(model.process(&input,w,h,0,0).is_err());
  unsafe {assert_eq!(test_entries(),0,"invalid slice entered native");assert_eq!(test_calls(),0);}
  released(0);
 }
 let pixels:Vec<u8>=(0..24).collect();let before=pixels.clone();
 for scale in [1,2,3,4] {
  for len in [18,19,24] {
   for tile in [(0,0),(-1,-2),(32,64),(1,1)] {
    reset(scale,len);
    let (output,w,h)=model.process(&pixels[..len],3,2,tile.0,tile.1).unwrap();
    assert_eq!((w,h),(3*scale,2*scale));
    let mut expected=Vec::new();
    for y in 0..h {for x in 0..w {for c in 0..3 {
     expected.push(pixels[(((y/scale)*3+x/scale)*3+c) as usize]);
    }}}
    assert_eq!(output,expected); released(1);
    unsafe {assert_eq!((test_tile_width(),test_tile_height()),tile);assert_eq!(test_entries(),1);}
   }
  }
 }
 assert_eq!(pixels,before);
 // The shared count guard is signed i32, including RGB multiplication.
 assert_eq!(vision_image_len(i32::MAX/3,1,3,false),Some((i32::MAX/3*3) as usize));
 assert_eq!(vision_image_len(i32::MAX/3+1,1,3,false),None);
 for (ow,oh) in [(0,4),(-1,-1),(6,0),(i32::MAX,2),(1,i32::MAX)] {
  reset(2,18);unsafe{test_result(3,ow,oh);}
  assert!(model.process(&pixels[..18],3,2,0,0).unwrap_err().contains("invalid output dimensions"));
  released(1);
 }
 reset(2,18);unsafe{test_result(1,0,0);}
 assert_eq!(model.process(&pixels[..18],3,2,0,0).unwrap_err(),"DAT SR processing failed");
 released(1); // Nonzero status with an allocated output must still release it.
 reset(2,18);unsafe{test_result(2,0,0);}
 assert_eq!(model.process(&pixels[..18],3,2,0,0).unwrap_err(),"DAT SR processing failed");
 released(0); // Fixture freed its allocation before returning NULL success.
 reset(2,18);unsafe{test_fail_alloc(1);}
 assert!(model.process(&pixels[..18],3,2,0,0).is_err()); released(0);
 reset(2,18);
 assert!(model.process(&pixels[..18],3,2,i32::MAX,64).is_err());
 unsafe {assert_eq!(test_entries(),1);assert_eq!(test_tile_width(),i32::MAX);}
 released(0);
 let constant=vec![200u8;96*64*3];
 for (gate,errors) in [(None,1920),(Some("0"),1920),(Some("1"),0)] {
  match gate {Some(value)=>std::env::set_var("CRISPEMBED_DAT_POSITIVE_TILE_BLEND",value),
              None=>std::env::remove_var("CRISPEMBED_DAT_POSITIVE_TILE_BLEND")}
  reset(2,constant.len());
  let (output,w,h)=model.process(&constant,96,64,0,0).unwrap();
  assert_eq!((w,h),(192,128));assert_eq!(output.iter().filter(|&&v|v!=200).count(),errors);
  released(1);
 }
 drop(model);unsafe {assert_eq!(test_context_frees(),1);}
 assert!(CrispDatSr::new("bad\0path",0).is_err());
 println!("Rust DAT checked RGB, returned geometry, ownership and gate controls passed");
}
'''
        with tempfile.TemporaryDirectory(prefix="rust-dat-buffer-") as directory:
            library = dat_fixture.DatTileBlendTests.build_fixture(directory)
            source, binary = Path(directory)/"consumer.rs", Path(directory)/"consumer"
            source.write_text(production_methods()+main)
            subprocess.run([rustc,"--edition=2021",str(source),"-o",str(binary),
                            "-C","link-arg="+str(library),"-C","link-arg=-Wl,-rpath,"+directory],
                           check=True,timeout=30)
            subprocess.run([str(binary)],check=True,timeout=30)


if __name__ == "__main__":
    unittest.main()
