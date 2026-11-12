"""Hosted-only production Python/Dart FFI control; no models or image inference.

Usage: python tests/ocr_binding_native_memory.py --library /abs/libcrispembed.so
       [--dart /abs/dart]
Compiles a tiny header-owned fixture and links the supplied real native library.
Dart execution requires a preinstalled SDK and fetches only pinned ffi 2.1.4.
"""
import argparse
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
C_FIXTURE = r'''
#include "crispembed.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
static crispembed_ocr_result rows[3];
static const char *texts[] = {"Gr\303\274\303\237e", "second line", "third & line"};
static int failed, packing;
static const crispembed_ocr_result *make(int *n) {
  for(int i=0;i<3;i++) rows[i]=(crispembed_ocr_result){
    .x=i+1,.y=10+i,.w=100,.h=20,.confidence=.75f,.text=texts[i],
    .text_len=(int)strlen(texts[i]),.orientation_corrected=i>0,
    .orientation_angle=i*90,.orientation_confidence=.25f*(i+1)};
  *n=3; return rows;
}
void *crispembed_ocr_init(const char*a,const char*b,int t){(void)a;(void)b;(void)t;return (void*)1;}
void crispembed_ocr_free(void*c){(void)c;}
const crispembed_ocr_result *crispembed_ocr(void*c,const char*p,int*n){(void)c;(void)p;return make(n);}
void *crispembed_ocr_pipeline_init(const crispembed_ocr_pipeline_params*p,int t){(void)p;(void)t;return (void*)1;}
void crispembed_ocr_pipeline_free(void*c){(void)c;}
const crispembed_ocr_result *crispembed_ocr_pipeline_run(void*c,const char*p,int*n,const char**text,float*conf){
  (void)c;failed=!strcmp(p,"fail");
  if(failed){*n=0;*text=NULL;*conf=0;return NULL;}
  *text="Gr\303\274\303\237e\nsecond line\nthird & line";*conf=.75f;return make(n);
}
const char *crispembed_ocr_pipeline_last_error(void*c){(void)c;return failed?"fixture reported failure":"";}
size_t fixture_size(void){return sizeof(crispembed_ocr_result);}
size_t fixture_text_offset(void){return offsetof(crispembed_ocr_result,text);}
int fixture_packing(void){return packing;}
char *crispembed_ocr_render(const crispembed_ocr_result*r,int n,int w,int h,const char*format){
  packing=n==3;
  for(int i=0;i<n && packing;i++) packing= r[i].x==i+1 && r[i].y==10+i &&
    r[i].w==100 && r[i].h==20 && r[i].confidence==.75f &&
    !strcmp(r[i].text,texts[i]) && r[i].text_len==(int)strlen(texts[i]) &&
    r[i].orientation_corrected==(i>0) && r[i].orientation_angle==0 && r[i].orientation_confidence==0;
  typedef char *(*render_fn)(const crispembed_ocr_result*,int,int,int,const char*);
  render_fn actual=(render_fn)dlsym(RTLD_NEXT,"crispembed_ocr_render");
  if(!packing || !actual) return NULL;
  return actual(r,n,w,h,format);
}
'''
DART_CONTROL = r'''
import 'dart:ffi';
import 'crispembed.dart';
void require(bool value) { if (!value) throw StateError('native-memory control failed'); }
void main(List<String> args) {
  final pipeline = CrispOcrPipeline('fixture','fixture',libPath:args[0]);
  final rows = pipeline.run('fixture');
  require(rows.length==3);
  final texts=['Grüße','second line','third & line'];
  for(var i=0;i<3;i++) {
    require(rows[i].text==texts[i] && rows[i].x==i+1 && rows[i].y==10+i &&
      rows[i].w==100 && rows[i].h==20 && rows[i].confidence==.75 &&
      rows[i].orientationCorrected==(i>0));
  }
  final lib=DynamicLibrary.open(args[0]);
  final rendered=ocrRender(lib,rows,640,480,'hocr');
  final text=rendered ?? '';
  require(text.contains('Grüße') && text.contains('second line') && text.contains('third &amp; line'));
  final packed=lib.lookupFunction<Int32 Function(),int Function()>('fixture_packing');
  require(packed()==1);
  pipeline.dispose();
  print('production Dart three-region read and native render packing passed');
}
'''


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--library', required=True)
    parser.add_argument('--dart')
    args = parser.parse_args()
    library = Path(args.library).resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='ocr-binding-control-') as temp:
        out = Path(temp)
        (out / 'fixture.c').write_text(C_FIXTURE)
        fixture = out / 'fixture.so'
        subprocess.run(['cc', '-shared', '-fPIC', '-I', str(ROOT / 'src'),
                        str(out / 'fixture.c'), '-Wl,--no-as-needed', str(library),
                        '-Wl,-rpath,' + str(library.parent), '-ldl', '-o', str(fixture)], check=True, timeout=60)
        spec = importlib.util.spec_from_file_location('production_binding', ROOT / 'python/crispembed/_binding.py')
        binding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(binding)
        native = C.CDLL(str(fixture))
        for name in ['fixture_size', 'fixture_text_offset']:
            getattr(native, name).restype = C.c_size_t
        assert C.sizeof(binding._CrispOcrResult) == native.fixture_size()
        assert binding._CrispOcrResult.text.offset == native.fixture_text_offset()
        basic = binding.CrispOcrPipeline('fixture', 'fixture', lib_path=str(fixture))
        orchestrator = binding.CrispOcrOrchestrator(lib_path=str(fixture))
        expected = ['Grüße', 'second line', 'third & line']
        for rows in [basic.run('fixture'), orchestrator.run('fixture')['regions']]:
            assert [r['text'] for r in rows] == expected
            assert [r['x'] for r in rows] == [1, 2, 3]
            assert [r['y'] for r in rows] == [10, 11, 12]
            assert all(r['w'] == 100 and r['h'] == 20 and r['confidence'] == .75 for r in rows)
        saved = orchestrator.run('fixture')
        try:
            orchestrator.run('fail')
        except RuntimeError as error:
            assert 'fixture reported failure' in str(error)
        else:
            raise AssertionError('reported failure was accepted')
        assert [r['text'] for r in saved['regions']] == expected
        assert orchestrator.run('fixture')['n_regions'] == 3
        result = {'production_python_three_region_read': True, 'failure_recovery_owned_output': True,
                  'native_result_bytes': native.fixture_size(), 'dart_executed': False,
                  'other_platforms_qualified': False}
        if args.dart:
            (out / 'pubspec.yaml').write_text('name: ocr_native_memory_control\nenvironment:\n  sdk: ">=3.0.0 <4.0.0"\ndependencies:\n  ffi: 2.1.4\n')
            for name in ['crispembed.dart', 'crispembed_bindings.dart']:
                shutil.copyfile(ROOT / 'flutter/crispembed/lib/src' / name, out / name)
            (out / 'control.dart').write_text(DART_CONTROL)
            subprocess.run([args.dart, 'pub', 'get'], cwd=out, check=True, timeout=120)
            subprocess.run([args.dart, 'run', 'control.dart', str(fixture)], cwd=out, check=True, timeout=120)
            result['dart_executed'] = True
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
