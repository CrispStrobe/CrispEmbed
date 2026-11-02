#!/usr/bin/env python3
"""Hosted source/profile build and exact model-file reconstruction, no originals."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,CATALOG,atomic,require,sha,source_guard,resources,numerical_environment
sys.path.insert(0,str(ROOT/'tools'))
from ci_process import run


def command(args,log,timeout=3600):
    result=run(args,log,cwd=ROOT,timeout=timeout)
    result.update(log=str(log.name),log_sha256=sha(log));atomic(Path(str(log)+'.process.json'),result)
    require(result['exit']==0 and not result['timed_out'],'Hosted build/reconstruction command rejected: '+log.name)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['analysis','continue','models']);parser.add_argument('--reviewed-source',required=True)
    for name in ['build','out','stage']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True);args.build.mkdir(parents=True,exist_ok=True)
    mode='continue'if args.mode=='models'else args.mode
    receipt=dict(completed=False,mode=args.mode,model_forwards=0,original_rerun=False,commands=[])
    try:
        receipt['source']=source_guard(mode,args.reviewed_source);receipt['resources']=resources(args.out,args.build)
        receipt['tooling']=numerical_environment()
        catalog=json.loads(CATALOG.read_text());root=args.build/'root';probe=args.build/'probe'
        def invoke(name,values,timeout=3600):
            result=command(values,args.out/(name+'.log'),timeout);receipt['commands'].append(result);atomic(args.out/('build-'+args.mode+'.json'),receipt)
        if args.mode=='models':
            require(json.loads((args.out/'build-continue.json').read_text())['completed'],'Completed source/profile build required before checkpoint access')
            require(json.loads((args.out/'fetch.json').read_text())['completed'],'Complete authenticated original/checkpoint fetch required')
            models=args.build/'models';models.mkdir(exist_ok=False)
            for dtype in ['f32','f16']:
                invoke('convert-'+dtype,[sys.executable,ROOT/'models/convert-minicpm-ocr-to-gguf.py',args.stage/'checkpoint',models/(dtype+'.gguf'),'--dtype',dtype])
            quantizer=root/'crispembed-quantize'
            invoke('quantize-q8',[quantizer,models/'f32.gguf',models/'q8_0.gguf','q8_0','--tensor-type','v.emb.patch_embedding.weight=f16'])
            invoke('quantize-q4',[quantizer,models/'f16.gguf',models/'q4_k.gguf','q4_k','--tensor-type','v.*=f16','--tensor-type','m.*=f16','--tensor-type','l.embed_tokens.weight=f16'])
            for dtype,pin in catalog['models'].items():
                path=models/(dtype+'.gguf');require(path.stat().st_size==pin['bytes']and sha(path)==pin['sha256'],'Reconstructed exact-v3 model bytes differ')
            receipt['models_exact_whole_sha']=True
        else:
            invoke('configure-probe',['cmake','-S',ROOT/'tests/minicpm-graph','-B',probe,'-DCMAKE_BUILD_TYPE=Release','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF'])
            targets=['minicpm-graph-diff']if args.mode=='analysis'else ['minicpm-graph-diff','minicpm-graph-probe','minicpm-graph-controls']
            invoke('compile-probe',['cmake','--build',probe,'--parallel','2','--target',*targets])
            if args.mode=='continue':
                invoke('graph-controls',[probe/'minicpm-graph-controls'])
                invoke('configure-root',['cmake','-S',ROOT,'-B',root,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                    '-DCRISPEMBED_NATIVE=OFF','-DGGML_NATIVE=OFF','-DGGML_LLAMAFILE=ON','-DGGML_CUDA=OFF'])
                invoke('compile-root',['cmake','--build',root,'--target','crispembed-cli','crispembed-quantize','test-ultrabert-profile','--parallel','2'],5400)
                invoke('ultra-controls',[root/'test-ultrabert-profile'])
                invoke('patch-controls',[sys.executable,ROOT/'tests/test_quantize_patch_float.py','--quantizer',root/'crispembed-quantize','--output',args.out/'patch-controls.json'])
                invoke('override-controls',[sys.executable,ROOT/'tests/test_quantize_tensor_overrides.py','--quantizer',root/'crispembed-quantize'])
                sys.path.insert(0,str(ROOT/'tools/minicpm-ocr'));from build_profile import verify
                profile=verify(root,probe);require(profile['cache']==catalog['build_profile']['cache'],'Compiled CPU/cache profile differs from actualv3')
                atomic(args.out/'build-profile.json',profile)
                symbols=subprocess.check_output(['nm','-C',root/'crispembed'],text=True)
                require('qwen35_ocr::graph_decoder::observe('not in symbols and 'qwen35_ocr::graph_decoder::observe('in subprocess.check_output(['nm','-C',probe/'minicpm-graph-probe'],text=True),'Production observer exclusion/positive probe failed')
                receipt.update(profile_sha256=sha(args.out/'build-profile.json'),production_observer_excluded=True,
                    binary_sha256={key:sha(path)for key,path in {'cli':root/'crispembed','probe':probe/'minicpm-graph-probe','diff':probe/'minicpm-graph-diff','quantizer':root/'crispembed-quantize'}.items()})
        if args.mode=='analysis':receipt['binary_sha256']={'diff':sha(probe/'minicpm-graph-diff')}
        receipt['completed']=True
    except Exception as error:receipt['error']=type(error).__name__+': '+str(error);raise
    finally:atomic(args.out/('build-'+args.mode+'.json'),receipt)


if __name__=='__main__':main()
