#!/usr/bin/env python3
"""Hosted-only NEW20 public SystemOne F32/Q4 replay; explicit pending pins.

Inputs are staged by the reviewed hosted owner. This worker has no downloader,
latest-source lookup, original recapture, hidden-state injection or CPU original.
All originals are authenticated before NumPy, model loading or native forwards.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import copy
import ctypes
import importlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from josev2b_task_quality_v1 import (source_guard, native_prerequisites, declaration, panel, protocol,
    expected_stages, require, sha256, compare_public)

GATES=('JOSEV_PACKED','JOSEV_COMPOSED','JOSEV_BACKBONE','QWEN35_OCR','JOSEV_HEAD',
       'JOSEV_TEXT_FRONTEND','JOSEV_ANSWER_SERIALIZER')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def runtime_guard(runtime, pins):
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime,text=True).strip()
    require(head==pins['runtime_source'],'Explicit reviewed runtime source differs')
    objects=pins['runtime_protected_git_objects']
    require(all(name in objects for name in ['src','ggml','CMakeLists.txt','python/crispembed','tests','tools/cmake','examples']),
            'Complete runtime/library/public-binding/probe source closure required')
    for name,expected in objects.items():
        actual=subprocess.check_output(['git','rev-parse','HEAD:'+name],cwd=runtime,text=True).strip()
        require(actual==expected,'Runtime Git object differs: '+name)
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=all','--',*objects],
                                      cwd=runtime,text=True).strip(),'Dirty reviewed runtime closure')
    return dict(source=head,git_objects=objects)


def checked_npy(path,row):
    """NPY v1 exact C axes/extent, streaming finite-policy validation; no NumPy."""
    require(not path.is_symlink() and path.is_file() and path.stat().st_size==row['bytes'] and
            sha256(path)==row['sha256'],'Original NPY bytes/SHA differ')
    with path.open('rb') as stream:
        require(stream.read(8)==b'\x93NUMPY\x01\x00','Exact NPY1 format required')
        extent=stream.read(2);require(len(extent)==2,'Truncated NPY extent')
        size=struct.unpack('<H',extent)[0];require(0<size<=4096,'Bounded NPY header required')
        header=ast.literal_eval(stream.read(size).decode('latin1'))
        require(set(header)=={'descr','fortran_order','shape'} and header['fortran_order'] is False,
                'Exact C-order NPY required')
        dims=header['shape']
        require(isinstance(dims,tuple) and 1<=len(dims)<=4 and all(type(n)is int and 0<n<=2048 for n in dims)
                and list(dims)==row['shape'],'Original NPY shape differs')
        formats={'float32':('<f4','<f',4),'int64':('<i8','<q',8)}
        require(row['dtype'] in formats,'Original NPY dtype outside stage contract')
        descr,fmt,width=formats[row['dtype']]
        require(header['descr']==descr and stream.tell()+math.prod(dims)*width==row['bytes'],
                'Exact NPY dtype/extent differs')
        negative=0
        while True:
            block=stream.read(1024*1024)
            if not block:break
            require(len(block)%width==0,'NPY chunk misaligned')
            if row['dtype']=='float32':
                for (value,) in struct.iter_unpack(fmt,block):
                    require(not math.isnan(value) and value!=math.inf,'NaN/positive infinity in original')
                    if value==-math.inf:negative+=1
        permitted=row['stage']=='backbone/layer/3/input/attention_mask'
        require(negative==row['negative_infinity'] and (permitted or negative==0),
                'Original negative infinity differs from declared causal-mask policy')


def original_guard(directory,pins):
    for filename,key in [('reference.json','original_reference_sha256'),('input-freeze.json','original_freeze_sha256'),
                         ('raw-manifest.json','original_raw_manifest_sha256')]:
        require(sha256(directory/filename)==pins[key],'Reviewed original receipt differs: '+filename)
    report=json.loads((directory/'reference.json').read_text())
    freeze=json.loads((directory/'input-freeze.json').read_text())
    raw=json.loads((directory/'raw-manifest.json').read_text())
    cases=panel()['cases'];ids=[c['id'] for c in cases]
    require(report['status']=='completed' and report['conclusive'] is True and report['native_outputs'] is False and
            report['old_four_recaptured'] is False and report['git_commit']==pins['original_source'] and
            report['contract_sha256']==pins['original_protocol_sha256'] and report['panel_sha256']==declaration()['panel_sha256'] and
            report['model_revision']==protocol()['revision'] and report['input_freeze_sha256']==pins['original_freeze_sha256'],
            'Completed NEW20 original source/model/protocol identity differs')
    require(report['arrays']==raw and len(raw)==1480 and [r['id'] for r in report['cases']]==ids and
            [r['id'] for r in freeze['records']]==ids,'Exact ordered20/1480 original scope required')
    require(freeze['panel_sha256']==declaration()['panel_sha256'] and freeze['max_length']==512,
            'Frozen panel/length differs')
    require(report['snapshot']=={name:{key:row[key] for key in ['bytes','sha256']} for name,row in protocol()['files'].items()},'Pinned complete official snapshot proof differs')
    names=set();payload=0
    for case,record,observed in zip(cases,freeze['records'],report['cases']):
        require(record['request']==case['request'] and record['attention_mask']==[1]*len(record['encoded']['input_ids']) and
                0<len(record['attention_mask'])<=512 and len(record['attention_mask'])==record['unclipped_tokens_under_16384'] and
                observed['observer_bit_exact'] is True and observed['repeat_bit_exact'] is True,
                'Original input/no-clipping/neutrality proof differs')
        require(Counter(r['stage'] for r in raw if r['case']==case['id'])==expected_stages(case),
                'Missing/extra mandatory original stages')
    for row in raw:
        name=row['file'];relative=Path(name)
        require(relative.parts and relative.parts[0]=='arrays' and len(relative.parts)==2 and
                relative.name.endswith('.npy') and '..' not in relative.parts and name not in names,
                'Unsafe/duplicate original array member')
        names.add(name);payload+=row['bytes'];checked_npy(directory/relative,row)
    require(payload<=declaration()['original_scope']['maximum_array_bytes']+1480*4096,
            'Original array budget exceeded')
    from josev2b_head_schema import tensor_schema
    expected=tensor_schema();weights=report['weights']
    require(weights['head_identity_exact'] is True and weights['head_tensor_count']==122 and
            weights['head_checkpoint_sha256']==protocol()['files']['joint_head.safetensors']['sha256'] and
            set(weights['head_tensors'])==set(expected),'Complete122-tensor original loaded-head identity proof required')
    for name,row in weights['head_tensors'].items():
        require(row['shape']==list(expected[name]) and row['loaded_shape']==row['shape'] and
                row['loaded_dtype']=='float32' and row['checkpoint_promoted_f32_sha256']==row['loaded_f32_sha256'],
                'Original head F32 promotion/shape differs: '+name)
    return report,freeze


def run(command,log,*,env=None,timeout=3600,cwd=None):
    started=time.monotonic()
    with log.open('w') as stream:
        result=subprocess.run([str(x) for x in command],stdout=stream,stderr=subprocess.STDOUT,
                              env=env,timeout=timeout,cwd=cwd)
    receipt=dict(command=[str(x) for x in command],exit_code=result.returncode,seconds=time.monotonic()-started,
                 log=log.name,sha256=sha256(log))
    require(result.returncode==0,'Command rejected; retained log '+log.name)
    return receipt


def public_calls(runtime, library, model, profile, data, out):
    """Actual production C and byte-identical Python wrapper, no model aliases."""
    sys.path.insert(0,str(runtime/'python'))
    import crispembed
    require(Path(crispembed.__file__).resolve().is_relative_to(runtime/'python'),'Production Python import drift')
    from crispembed import CrispSystemOne
    api=ctypes.CDLL(str(library))
    api.crispembed_systemone_init.argtypes=[ctypes.c_char_p,ctypes.c_int];api.crispembed_systemone_init.restype=ctypes.c_void_p
    api.crispembed_systemone_free.argtypes=[ctypes.c_void_p];api.crispembed_systemone_free.restype=None
    api.crispembed_systemone_run_json.argtypes=[ctypes.c_void_p,ctypes.c_char_p];api.crispembed_systemone_run_json.restype=ctypes.c_char_p
    api.crispembed_systemone_error.argtypes=[ctypes.c_void_p];api.crispembed_systemone_error.restype=ctypes.c_char_p
    api.crispembed_systemone_profile.argtypes=[ctypes.c_void_p];api.crispembed_systemone_profile.restype=ctypes.c_char_p
    context=api.crispembed_systemone_init(str(model).encode(),2)
    require(context,'C API failed to load authenticated model')
    rows={};records=[]
    try:
        require(api.crispembed_systemone_profile(context).decode()==profile,'Actual C profile differs')
        for case in data['cases']:
            raw=json.dumps(case['request'],ensure_ascii=False,separators=(',',':'),allow_nan=False)
            response=api.crispembed_systemone_run_json(context,raw.encode());require(response,'C inference failed')
            first=response.decode();repeat=api.crispembed_systemone_run_json(context,raw.encode())
            require(repeat and repeat.decode()==first,'C repeat bytes differ')
            require(api.crispembed_systemone_run_json(context,b'{') is None and api.crispembed_systemone_error(context),
                    'C malformed-input rejection failed')
            recovered=api.crispembed_systemone_run_json(context,raw.encode())
            require(recovered and recovered.decode()==first,'C recovery bytes differ')
            rows[case['id']]=json.loads(first)
            records.append(dict(id=case['id'],response=first,ordinary_repeat_recovery_equal=True,c_python_equal=False))
            write(out/'public.json',records)
    finally:api.crispembed_systemone_free(context)
    # Never hold two2B weight contexts at once just to compare language bindings.
    with CrispSystemOne(str(model),lib_path=str(library),auto_download=False,offline=True,n_threads=2) as python:
        require(python.profile==profile,'Actual Python profile differs')
        for case,record in zip(data['cases'],records):
            raw=json.dumps(case['request'],ensure_ascii=False,separators=(',',':'),allow_nan=False)
            first=record['response']
            require(python.run_json(raw)==first and python.run_json(raw)==first,'C/Python/repeat bytes differ')
            try:python.run_json('{')
            except ValueError:pass
            else:raise ValueError('Python malformed-input rejection failed')
            require(python.run_json(raw)==first,'Python recovery bytes differ')
            record['c_python_equal']=True;write(out/'public.json',records)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['runtime','reference','candidate-report','f32-model','q4-model','build','out']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-child',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=args.native_child)
    receipt=dict(completed=False,publication_authorized=False,full_model_qualified=False,commands=[])
    try:
        receipt['source']=source_guard();pins=native_prerequisites()
        runtime=args.runtime.resolve();receipt['runtime']=runtime_guard(runtime,pins)
        require(sha256(args.candidate_report)==pins['candidate_report_sha256'],'Reviewed old-four candidate evidence differs')
        original,freeze=original_guard(args.reference,pins)
        require(not args.f32_model.is_symlink() and not args.q4_model.is_symlink() and
                sha256(args.f32_model)==pins['f32_model_sha256'] and sha256(args.q4_model)==pins['q4_model_sha256'],
                'Exact reviewed F32/Q4 model differs')
        if not args.native_child:
            require(shutil_free(args.build.parent)>=25*2**30 and memory_available()>=10*2**30,
                    'Hosted build needs25GiB disk/10GiB available RAM')
            receipt['commands'].append(run(['cmake','-S',ROOT/'tests/josev-task-quality','-B',args.build,
                '-DCMAKE_BUILD_TYPE=Release','-DREVIEWED_RUNTIME='+str(runtime)],args.out/'configure.log'))
            receipt['commands'].append(run(['cmake','--build',args.build,'--target','crispembed-shared',
                'josev-quality-plain-probe','josev-quality-observed-probe','josev-quality-diff','-j2'],args.out/'build.log'))
            receipt['build_manifest']=binary_inventory(args.build)
            require(receipt['build_manifest'],'Empty build manifest')
            write(args.out/'build-manifest.json',receipt['build_manifest'])
            # Fresh loader search path must exist at interpreter startup. No inherited path fallback.
            libraries=list(args.build.rglob('libcrispembed.so'))
            require(len(libraries)==1,'Exact single public shared library required')
            dirs=sorted({str(p.parent.resolve()) for p in args.build.rglob('libggml*.so*')}|{str(libraries[0].parent.resolve())})
            env=os.environ.copy();env['LD_LIBRARY_PATH']=':'.join(dirs+[str(Path(sys.base_prefix)/'lib')])
            receipt['commands'].append(run([sys.executable,Path(__file__),*sys.argv[1:],'--native-child'],args.out/'native-child.log',env=env,timeout=7200))
            child=json.loads((args.out/'native.json').read_text());receipt.update(completed=child['completed'],owned_panel_qualified=child['owned_panel_qualified'],child_sha256=sha256(args.out/'native.json'))
            return
        build_manifest=json.loads((args.out/'build-manifest.json').read_text())
        require(binary_inventory(args.build)==build_manifest,'Built libraries/probes/comparator changed before fresh child')
        receipt['build_manifest_sha256']=sha256(args.out/'build-manifest.json')
        from josev2b_composed_panel import ref_controls
        from josev2b_task_quality_panel_v1 import compare
        ref_controls(args.build/'josev-quality-diff',args.out/'ref-controls')
        off=os.environ.copy()
        for key in list(off):
            if key.startswith('CRISPEMBED_'):off.pop(key)
        for variant in ['plain','observed']:
            receipt['commands'].append(run([args.build/('josev-quality-'+variant+'-probe'),'--gates'],
                args.out/('gates-'+variant+'.log'),env=off,timeout=30))
        libraries=list(args.build.rglob('libcrispembed.so'));require(len(libraries)==1,'Single public library required')
        library=libraries[0].resolve();receipt['library_sha256']=sha256(library)
        symbols=subprocess.check_output(['nm','-C',str(library)],text=True)
        require('josev2b::composed_test::' not in symbols and 'josev2b_head::test::' not in symbols,
                'Test observers leaked into public production library')
        (args.out/'plain-symbols.log').write_text(symbols)
        receipt['plain_observer_excluded']=True
        packet=[dict(id=c['id'],request_json=json.dumps(c['request'],ensure_ascii=False,separators=(',',':'),allow_nan=False)) for c in panel()['cases']]
        write(args.out/'requests.json',packet)
        results={}
        for precision,model in [('f32',args.f32_model),('q4',args.q4_model)]:
            if precision=='q4':require(results['f32']['qualified_for_owned_panel'],'F32 NEW20 control rejected; Q4 not accessed')
            env=os.environ.copy()
            for key in list(env):
                if key.startswith('CRISPEMBED_'):env.pop(key)
            env.update({('CRISPEMBED_'+key):'1' for key in GATES})
            env.update(CRISPEMBED_JOSEV_QUANT='1' if precision=='q4' else '0',
                       CRISPEMBED_QWEN35_QUANT='1' if precision=='q4' else '0',
                       CRISPEMBED_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            folder=args.out/precision;folder.mkdir()
            outputs={}
            for mode in ['plain','observed','repeat']:
                binary=args.build/('josev-quality-'+('plain' if mode=='plain' else 'observed')+'-probe')
                outputs[mode]=folder/mode
                receipt['commands'].append(run([binary,model,args.out/'requests.json',outputs[mode]],folder/(mode+'.log'),env=env,timeout=3600))
            result=compare(args.reference,freeze,original,outputs,args.build/'josev-quality-diff',folder/'comparison',precision=precision)
            # Explicit process environment matches the spawned probes for actual public calls.
            os.environ.clear();os.environ.update(env)
            public=public_calls(runtime,library,model,pins[precision+'_profile'],panel(),folder)
            probe={r['id']:json.loads(r['response']) for r in [json.loads(line)for line in (outputs['plain']/'native.jsonl').read_text().splitlines()]}
            require(public==probe,'Public C/Python results differ from actual detailed plain calls')
            result['public_c_python_exact']=True
            if precision=='f32':f32public=public
            else:
                result['q4_vs_f32_task_quality']=compare_public(f32public,public,precision='q4')
                result['qualified_for_owned_panel'] &= result['q4_vs_f32_task_quality']['independent_task_quality_passed']
            results[precision]=result;write(folder/'result.json',result)
        receipt.update(completed=True,results=results,owned_panel_qualified=all(r['qualified_for_owned_panel'] for r in results.values()))
    except Exception as error:
        receipt['error']=type(error).__name__+': '+str(error)
        raise
    finally:write(args.out/('native.json'if args.native_child else 'hosted.json'),receipt)


def binary_inventory(build):
    build=build.resolve();result={}
    for path in build.rglob('*'):
        if path.is_file() and (path.name.startswith(('libggml','libcrispembed.so','josev-quality-'))):
            actual=path.resolve()
            require(actual.is_relative_to(build),'Library/probe symlink escapes authenticated build')
            result[str(path.relative_to(build))]=dict(bytes=path.stat().st_size,sha256=sha256(path),
                target=str(actual.relative_to(build)) if path.is_symlink() else None)
    return result


def memory_available():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()if line.startswith('MemAvailable:')))*1024


def shutil_free(path):
    import shutil
    path.mkdir(parents=True,exist_ok=True)
    return shutil.disk_usage(path).free


if __name__=='__main__':main()
