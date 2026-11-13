#!/usr/bin/env python3
"""Actual Rust TableParse methods reuse the small owned-HTML/grid C fixture."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import test_table_parse_binding as table_fixture
from test_cc_detect_ownership import extract_function

ROOT = Path(__file__).resolve().parents[1]


def production_methods(source=None, include_helpers=True):
    raw = (ROOT / "crispembed-sys/src/lib.rs").read_text()
    safe = source if source is not None else (ROOT / "crispembed/src/lib.rs").read_text()
    declarations = []
    for name in ("init", "free", "to_html", "free_string", "detect_grid"):
        start = raw.index("    pub fn crispembed_table_parse_" + name + "(")
        declarations.append(raw[start:raw.index(";", start)+1])
    helpers = "\n".join(extract_function(safe, prefix) for prefix in (
        "fn grayscale_image_len(", "fn table_parse_image_len(")) if include_helpers else ""
    return ('use std::ffi::CString;\nmod crispembed_sys {use std::ffi::{c_char,c_int,c_void};\n'
            'extern "C" {\n' + "\n".join(declarations) + "\n}\n}\n" + helpers + "\n" +
            extract_function(safe, "pub struct CrispTableParse {") + "\n" +
            extract_function(safe, "impl CrispTableParse {") + "\n" +
            extract_function(safe, "impl Drop for CrispTableParse {"))


def run_consumer(directory, rust_source):
    rustc = shutil.which("rustc")
    if not rustc:
        raise unittest.SkipTest("Rust compiler required for focused interface test")
    library = table_fixture.TableParseBindingTests.build_fixture(directory)
    directory = Path(directory)
    source, binary = directory / "consumer.rs", directory / "consumer"
    source.write_text(rust_source)
    subprocess.run([rustc, "--edition=2021", str(source), "-o", str(binary),
                    "-C", "link-arg=" + str(library),
                    "-C", "link-arg=-Wl,-rpath," + str(directory)],
                   check=True, text=True, timeout=30)
    subprocess.run([str(binary)], check=True, text=True, timeout=10)


class RustTableParseSafetyTests(unittest.TestCase):
    def test_actual_methods_input_prefixes_html_owner_and_native_grid_status(self):
        main = r'''
extern "C" {
 fn test_reset(length:i32,mode:i32); fn test_calls()->i32; fn test_frees()->i32;
 fn test_live()->i32; fn test_byte(index:i32)->i32; fn test_context_frees()->i32;
}
fn main() {
 let parser=CrispTableParse::new("",4).unwrap();
 assert!(CrispTableParse::new("bad\0path",4).is_err());
 for (w,h,len) in [(0,1,0),(1,0,0),(-1,1,0),(-1,-1,1),
                    (i32::MAX,1,0),(i32::MAX-30,1,0),(65536,65536,0),(4,3,11)] {
  let gray=vec![0u8;len]; unsafe{test_reset(len as i32,0);}
  assert!(parser.to_html(&gray,w,h).is_none());
  assert!(CrispTableParse::detect_grid(&gray,w,h).is_none());
  unsafe{assert_eq!(test_calls(),0,"invalid input reached native parser");assert_eq!(test_live(),0);}
 }
 for extra in [0,1,5] {
  let gray:Vec<u8>=(0..12+extra).collect(); unsafe{test_reset(gray.len() as i32,0);}
  let html=parser.to_html(&gray,4,3).unwrap();
  assert_eq!(html,"<table><tr><td>日本語 €</td></tr></table>");
  unsafe{assert_eq!(test_frees(),1);assert_eq!(test_live(),0);}
  assert_eq!(CrispTableParse::detect_grid(&gray,4,3),Some((2,3)));
  unsafe{
   assert_eq!(test_calls(),2);
   for i in 0..12 {assert_eq!(test_byte(i),i);}
  }
 }
 let gray=[0u8;12];
 for (mode,expected) in [(1,None),(2,Some("")),(3,Some("\u{FFFD}"))] {
  unsafe{test_reset(12,mode);}
  assert_eq!(parser.to_html(&gray,4,3).as_deref(),expected);
  unsafe{assert_eq!(test_frees(),if mode==1 {0}else{1});assert_eq!(test_live(),0);}
 }
 // Test the exact internal return-code implementation and public wrapper,
 // not the formerly incorrect header claim that success returns cell count.
 let (mut rows,mut cols)=(0,0);
 unsafe{
  test_reset(12,0);
  assert_eq!(crispembed_sys::crispembed_table_parse_detect_grid(gray.as_ptr(),4,3,&mut rows,&mut cols),0);
  assert_eq!((rows,cols),(2,3));
  test_reset(12,1);
  assert_eq!(crispembed_sys::crispembed_table_parse_detect_grid(gray.as_ptr(),4,3,&mut rows,&mut cols),-1);
 }
 assert_eq!(CrispTableParse::detect_grid(&gray,4,3),None);
 drop(parser);
 unsafe{assert_eq!(test_context_frees(),1);assert_eq!(test_live(),0);}
}
'''
        with tempfile.TemporaryDirectory(prefix="rust-table-parse-") as directory:
            run_consumer(directory, production_methods() + main)


if __name__ == "__main__":
    unittest.main()
