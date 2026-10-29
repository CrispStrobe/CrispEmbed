#!/usr/bin/env python3
"""Stdlib admission/build-profile guards; no compilation or real payloads."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('q8controls',ROOT/'tools/ci-heavy/moedern_q8_controls_v1.py')
w=importlib.util.module_from_spec(s);s.loader.exec_module(w)


class Controls(unittest.TestCase):
    def test_complete_runtime_fixture_closure(self):
        p=w.validate(w.read(w.PROFILE))
        for path,h in p['owned_sha256'].items():self.assertEqual(w.sha(ROOT/path),h,path)
        for path,obj in p['protected_git_objects'].items():
            self.assertEqual(subprocess.check_output(['git','rev-parse',w.FROZEN+':'+path],cwd=ROOT,text=True).strip(),obj,path)
            self.assertEqual(subprocess.check_output(['git','rev-parse','HEAD:'+path],cwd=ROOT,text=True).strip(),obj,path)
        for name in w.REQUIRED:
            bad=copy.deepcopy(p);bad['protected_git_objects'].pop(name)
            with self.assertRaisesRegex(ValueError,'complete runtime'):w.validate(bad)

    def test_no_model_or_corpus_waiver(self):
        p=w.read(w.PROFILE)
        for key in ['model_payloads','corpus_access','model_qualified','dispatch_authorized']:
            bad=copy.deepcopy(p);bad[key]=True
            with self.assertRaisesRegex(ValueError,'model-free'):w.validate(bad)
        for key in ['authored_graphs','typed_profile_counts','authored_fixture']:
            bad=copy.deepcopy(p);bad[key]=None
            with self.assertRaisesRegex(ValueError,'scope differs'):w.validate(bad)
        bad=copy.deepcopy(p);bad['owned_sha256']['src/pretend.cpp']='0'*64
        with self.assertRaisesRegex(ValueError,'cannot own runtime'):w.validate(bad)

    def test_real_build_flags_and_source_counts(self):
        cache='\n'.join(k+':BOOL='+v for k,v in w.EXPECTED_CACHE.items())
        names=['gguf_loader.cpp','moedern_text_engine.cpp','moedern_family_engine.cpp','moedern_text_frontend.cpp','moedern_text.cpp','moedern_text_api.cpp']
        commands=[dict(file='/fixture/'+n,command='c++ -O3 -ffp-contract=off -c '+n)for n in names]
        commands += [dict(file='/fixture/'+n,command='c++ -O3 -c '+n)for n in ['controls.cpp','quantize.cpp']]
        w.build_profile(cache,commands)
        with self.assertRaisesRegex(ValueError,'CMake profile'):w.build_profile(cache.replace('GGML_LLAMAFILE:BOOL=ON','GGML_LLAMAFILE:BOOL=OFF'),commands)
        with self.assertRaisesRegex(ValueError,'contraction'):w.build_profile(cache,commands+commands[:1])
        bad=copy.deepcopy(commands);bad[0]['command']+=' -ffast-math'
        with self.assertRaisesRegex(ValueError,'contraction'):w.build_profile(cache,bad)
        with self.assertRaisesRegex(ValueError,'compilation absent'):w.build_profile(cache,commands[:-1])

    def test_workflow_has_only_hosted_existing_targets(self):
        text=(ROOT/'.github/workflows/moedern-q8-controls-v1.yml').read_text()
        self.assertNotIn('pip install',text);self.assertNotIn('KAGGLE',text);self.assertNotIn('matrix:',text)
        self.assertIn('if: always()',text);self.assertIn('retention-days: 90',text)
        source=(ROOT/'tools/ci-heavy/moedern_q8_controls_v1.py').read_text()
        self.assertIn("'moedern-q8-controls','moedern-q8-quantize','--parallel','2'",source)
        for forbidden in ['snapshot_download','import torch','import numpy','encode_document(','http://','huggingface.co']:
            self.assertNotIn(forbidden,source)
        self.assertIn('q8_source_contract.verify(a.reviewed_source)',source)
        self.assertIn('fixture[\'actual_q8_tensors\']==2',source)
        self.assertIn('full_model_quantization_executed=False',source)


if __name__=='__main__':unittest.main()
