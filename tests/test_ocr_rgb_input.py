"""Source/stdlib-only controls; no decoder, image, array package or compiler."""
import hashlib
import copy
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools/ocr-rgb-input'),str(ROOT/'tools')]
import contract as c
import metrics
spec=importlib.util.spec_from_file_location('rgb_input_driver',ROOT/'tools/ocr-rgb-input/audit.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class Controls(unittest.TestCase):
    def test_isolated_help_needs_no_decoder_or_framework_package(self):
        code='''import builtins,runpy,sys
original=builtins.__import__
def guard(name,*a,**k):
 if name.split('.')[0]in{'PIL','numpy','pyarrow','torch','transformers'}:raise RuntimeError('decoder import before hosted admission')
 return original(name,*a,**k)
builtins.__import__=guard
target=sys.argv[1];sys.argv=[target,'--help'];runpy.run_path(target,run_name='__main__')
'''
        result=subprocess.run([sys.executable,'-I','-c',code,str(ROOT/'tools/ocr-rgb-input/audit.py')],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr);self.assertIn('--images-dir',result.stdout)

    def test_scope_source_omission_and_model_claims_fail_closed(self):
        cfg=json.loads(c.PATH.read_text());c.validate(cfg)
        for field in('model_execution','model_qualified','normalization_changed','cause_established'):
            bad=copy.deepcopy(cfg);bad[field]=True
            with self.assertRaises(ValueError):c.validate(bad)
        for name in('src/image_preprocess.cpp','tools/ocr-rgb-input/metrics.py','tests/ocr-rgb-input/probe.cpp'):
            bad=copy.deepcopy(cfg);bad['source_sha256'].pop(name)
            with self.assertRaises(ValueError):c.validate(bad)
        bad=copy.deepcopy(cfg);bad['manifest']['sha256']='0'*64
        with self.assertRaises(ValueError):c.validate(bad)

    def test_full_manifest_identity_and_credits(self):
        cfg=json.loads(c.PATH.read_text());p=ROOT/'docs/ocr-real-scans-v1.json'
        self.assertEqual(c.sha(p),cfg['manifest']['sha256']);self.assertEqual(p.stat().st_size,cfg['manifest']['bytes'])
        rows=json.loads(p.read_text())['pages'];self.assertEqual(len(rows),20);self.assertEqual(len({r['id']for r in rows}),20)
        self.assertEqual(sum('hf_parquet'in r['image']for r in rows),5)
        for row in rows:
            self.assertTrue(row['image_license']);self.assertTrue(row['ground_truth_license']);self.assertTrue(row['ground_truth_provenance'])

    def test_wrong_bytes_never_reach_decoder(self):
        data=b'authored input bytes';pin=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());called=[]
        decoder=lambda value:called.append(value)or'decoded'
        for value,bad in[(data[:-1],pin),(data,dict(pin,sha256='0'*64)),(data,dict(pin,bytes=len(data)+1))]:
            with self.assertRaisesRegex(ValueError,'decoder forbidden'):c.admitted_decode(value,bad,decoder)
        self.assertEqual(called,[]);self.assertEqual(c.admitted_decode(data,pin,decoder),'decoded');self.assertEqual(called,[data])

    def test_exact_production_prefix_and_macro_overrides_reject(self):
        source=(ROOT/'src/image_preprocess.cpp').read_bytes();prefix=c.prefix(source)
        self.assertEqual(prefix,source.split(b'#include "../ggml/examples/stb_image.h"\n')[0]+b'#include "../ggml/examples/stb_image.h"\n')
        for value in[source.replace(b'#define STB_IMAGE_IMPLEMENTATION',b'#define STB_IMAGE_STATIC\n#define STB_IMAGE_IMPLEMENTATION'),
                      source.replace(b'#define STB_IMAGE_IMPLEMENTATION',b'#define STBI_NO_SIMD\n#define STB_IMAGE_IMPLEMENTATION'),b'']:
            with self.assertRaises(ValueError):c.prefix(value)

    def test_actual_compiler_profile_stays_production_baseline(self):
        row=dict(arguments=['c++','-I/source/src',*c.PROFILE,'-o','object','-c','prefix.cpp'])
        self.assertEqual(driver.flags(row),row['arguments'])
        for flag in('-DSTBI_NO_SIMD','-DSTBI_ONLY_JPEG','-ffast-math','-march=native','@unchecked.rsp'):
            with self.assertRaises(ValueError):driver.flags(dict(arguments=[*row['arguments'],flag]))
        for flag in c.PROFILE:
            with self.assertRaises(ValueError):driver.flags(dict(arguments=[a for a in row['arguments']if a!=flag]))

    def test_decoder_macro_subset_and_duplicate_rejection(self):
        self.assertEqual(driver.macros('#define STBI_SSE2\n#define STB_IMAGE_IMPLEMENTATION\n#define __SSE2__ 1\n#define OTHER 2\n'),
                         {'STBI_SSE2':'','STB_IMAGE_IMPLEMENTATION':'','__SSE2__':'1'})
        with self.assertRaises(ValueError):driver.macros('#define STBI_X 1\n#define STBI_X 2\n')

    def test_raw_byte_metrics_first_pixel_and_channels(self):
        a=bytes([0,1,255,5,6,7]);b=bytes([1,1,254,5,8,7]);r=metrics.small(a,b,2)
        self.assertEqual(r['max_absolute'],2);self.assertEqual(r['mean_absolute'],2/3);self.assertEqual(r['rms'],1)
        self.assertEqual(r['changed_values'],3);self.assertEqual(r['changed_pixels'],2);self.assertFalse(r['rgb_exact'])
        self.assertEqual(r['first_difference'],dict(byte_offset=0,pixel_index=0,x=0,y=0,channel='R',pillow_pixel=[0,1,255],stb_pixel=[1,1,254],signed_difference=1))
        self.assertEqual(r['per_channel']['G']['max_absolute'],2);self.assertEqual(r['per_channel']['B']['mean_absolute'],.5)
        zero=metrics.small(a,a,2);self.assertTrue(zero['rgb_exact']);self.assertIsNone(zero['first_difference'])
        for left,right,width in[(b'',b'',1),(a,b[:-1],2),(a,b,4)]:
            with self.assertRaises(ValueError):metrics.small(left,right,width)

    def test_cpp_has_no_model_build_and_real_rgb_request(self):
        cmake=(ROOT/'tests/ocr-rgb-input/CMakeLists.txt').read_text();probe=(ROOT/'tests/ocr-rgb-input/probe.cpp').read_text()
        self.assertNotIn('add_subdirectory(',cmake);self.assertNotIn('ggml-cpu',cmake);self.assertNotIn('scan_cleanup.cpp',cmake)
        self.assertIn('stbi_load(argv[1], &width, &height, &channels, 3)',probe);self.assertIn('core_util::clean_exit(body(',probe)
        self.assertNotIn('preprocess_rgb(',probe);self.assertNotIn('recognize(',probe)

    def test_workflow_resource_and_source_checks_before_packages(self):
        text=(ROOT/'.github/workflows/ocr-rgb-input-audit-v1.yml').read_text()
        self.assertIn('workflow_dispatch:',text);self.assertNotIn('workflow_call:',text)
        self.assertLess(text.index('Resource admission'),text.index('Pinned codec/input packages'))
        self.assertLess(text.index('Source and stdlib controls'),text.index('Pinned codec/input packages'))
        self.assertIn('if: always()',text);self.assertIn("CMAKE_BUILD_PARALLEL_LEVEL: '2'",text)


if __name__=='__main__':unittest.main()
