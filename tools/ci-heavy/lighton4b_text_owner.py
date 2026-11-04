#!/usr/bin/env python3
"""Hosted, model-free exact text-owner checks against the original header."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute():
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':
        raise RuntimeError('hosted portable execution required')
    out=Path(os.environ['HEAVY_OUT']); scratch=Path(os.environ['HEAVY_SCRATCH'])/'lighton4b-text-owner'
    scratch.mkdir(parents=True,exist_ok=False)
    report={'schema':'lighton4b-text-owner/v1','status':'failed','model_payload_bytes':0,
            'native_inference_observed':False,'format5_admitted':False,'quality_qualified':False,
            'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_sha256':{p:sha(ROOT/p) for p in ('src/qwen35_lighton4b_text.h',
                'tests/test_lighton4b_text_owner.cpp','tools/ci-heavy/lighton4b_text_owner.py','CMakeLists.txt',
                'tests/test_lighton4b_gdn_reference.py','tests/test_ovisocr2_native_replay.py',
                'tests/test_ovisocr2_large_inputs.py',
                'tools/ci-heavy/ovisocr2_native_replay.py','tools/ci-heavy/lighton4b_gdn_reference.py')}}
    def command(label,argv):
        with (out/(label+'.log')).open('w') as stream:
            subprocess.run([str(x) for x in argv],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        command('gdn-protocol-controls',[sys.executable,ROOT/'tests/test_lighton4b_gdn_reference.py','-v'])
        command('ovis-frozen-closure-controls',[sys.executable,ROOT/'tests/test_ovisocr2_native_replay.py','-v'])
        command('ovis-frozen-large-input-controls',[sys.executable,ROOT/'tests/test_ovisocr2_large_inputs.py','-v'])
        sys.path.insert(0,str(ROOT/'models'))
        spec=importlib.util.spec_from_file_location('original_exporter',ROOT/'models/convert-qwen35-ocr-to-gguf.py')
        converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
        policy=converter._lighton4b_profile();policy.load_profile()
        header_path=policy.FIXTURE/'model.safetensors.header.json'
        header=json.loads(header_path.read_text())
        rows=[]
        for source,tensor in sorted(header.items()):
            if source=='__metadata__':continue
            name=converter.tensor_name(source)
            if not name.startswith('l.'):continue
            shape=tensor['shape'];ne=list(reversed(shape))+[1]*(4-len(shape))
            matrix=len(shape)==2 and 'norm' not in name
            rows.append(' '.join([name]+[str(n) for n in ne]+[str(int(matrix))]))
        if len(rows)!=426:raise ValueError('original text inventory must contain exactly426 tensors')
        fixture=out/'original-text-schema.tsv';fixture.write_text('\n'.join(rows)+'\n')
        report['original']={'model':policy.MODEL,'revision':policy.REVISION,
            'header_sha256':sha(header_path),'schema_sha256':sha(fixture),'text_tensors':426}
        command('configure',['cmake','-S',ROOT,'-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release',
            '-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        command('build',['cmake','--build',scratch/'build','--target','test-lighton4b-text-owner','--parallel','2'])
        binary=scratch/'build/test-lighton4b-text-owner'
        report['executable_sha256']=sha(binary)
        command('controls',[binary,fixture])
        text=(out/'controls.log').read_text()
        if 'passed=1 payload_bytes=0 model_inference=0' not in text:raise ValueError('component controls receipt missing')
        report['controls_stdout']=text
        report['status']='passed'
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error).replace(str(ROOT),'[repository]').replace(str(scratch),'[scratch]')
        raise
    finally:
        (out/'text-owner-receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':execute()
