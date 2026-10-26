#!/usr/bin/env python3
"""Hosted original-sentence layer3 capture and same-input projection replay v1."""
import argparse
import ctypes as C
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(Path(__file__).parent)]
from wasm_layer3_attribution_v1 import *
from successor_wasm_trace_contract import (FIXTURE_ID,FIXTURE_BYTES,FIXTURE_SHA,INITIAL_RUN,INITIAL_SOURCE,NATIVE_REPORT_SHA)
from successor_assets import api,artifact
from successor_asset_manifest import sha
from successor_asset_manifest import safe_name
import zipfile
SPEC=ROOT/'docs/wasm-layer3-attribution-v1.json'

def git(root,*args):
    return subprocess.check_output(['git',*args],cwd=root,text=True,encoding='utf-8',timeout=180).strip()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def command(args,out,name,cwd=ROOT,timeout=2400):
    with (out/(name+'.log')).open('w') as stream:
        subprocess.run([str(x) for x in args],cwd=cwd,stdout=stream,stderr=subprocess.STDOUT,
                       timeout=timeout,check=True)


def hosted():
    if os.environ.get('GITHUB_ACTIONS')!='true' or git(ROOT,'rev-parse','HEAD')!=os.environ.get('GITHUB_SHA'):
        raise ValueError('Hosted matching service/source required')
    spec=json.loads(SPEC.read_text())
    required={'tools/wasm_layer3_attribution_v1.py','tools/ci-heavy/wasm_layer3_attribution_v1.py',
        'tests/wasm-layer3/kernel_capture.inc','tests/wasm-layer3/replay.inc','tests/test_wasm_layer3_attribution_v1.cjs',
        'tests/test_wasm_layer3_diff.cpp','tools/successor_wasm_trace_contract.py',
        'tools/ci-heavy/successor_assets.py','tools/successor_asset_manifest.py'}
    if (set(spec['protected_sources'])!=required or spec.get('runtime_base')!=BASE or
            spec.get('ggml')!=GGML or spec.get('ab',{}).get('report_sha256')!=AB_REPORT or
            spec.get('gates')!={'max_abs':.002,'min_row_cosine':.99999}):
        raise ValueError('Exact source/protocol declaration differs')
    for name,value in spec['protected_sources'].items():
        if sha(ROOT/name)!=value:raise ValueError('Protected diagnostic source differs: '+name)
    return spec


def zip_files(path,maximum=64*1024**2):
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();names=[safe_name(x.filename) for x in infos]
        if len(infos)>300 or sum(x.file_size for x in infos)>maximum or len(set(names))!=len(names) or any(x.is_dir() or x.flag_bits&1 for x in infos):
            raise ValueError('ZIP member/budget differs')
        return {name:z.read(info) for name,info in zip(names,infos)}


def service(run,source,conclusion):
    row=api('actions/runs/'+str(run))
    if (row['head_sha'],row['path'],row['status'],row['conclusion'],row['event'])!=(source,'.github/workflows/ovisocr2-metadata.yml','completed',conclusion,'workflow_dispatch'):
        raise ValueError('Frozen service identity differs')
    return {k:row[k] for k in ('id','head_sha','path','status','conclusion','event')}


def pinned(run,name,out,identity):
    got=artifact(run,name,out,32*1024**2)
    if (got['id'],got['size_in_bytes'],got['digest'])!=(identity['id'],identity['bytes'],'sha256:'+identity['sha256']):
        raise ValueError('Frozen artifact identity differs')
    return got


def prepare(out):
    spec=hosted();out.mkdir(exist_ok=False)
    proof={'source':os.environ['GITHUB_SHA'],'protocol_sha256':sha(SPEC),'base':BASE,'ggml':GGML,'services':{},'artifacts':{},'qualification':False}
    proof['services']['ab']=service(AB_RUN,BASE,'failure')
    proof['artifacts']['review']=pinned(AB_RUN,'wasm-q8-rounding-ab-v1-review-'+str(AB_RUN),out/'ab-review.zip',AB_REVIEW)
    review=zip_files(out/'ab-review.zip')
    if digest(review['ab.json'])!=AB_REPORT:raise ValueError('Negative A/B report differs')
    ab=trusted_ab(json.loads(review['ab.json']))
    (out/'ab.json').write_bytes(review['ab.json'])
    (out/'old-panel-freeze.json').write_bytes(review['old-panel-freeze.json'])
    proof['artifacts']['raw']=pinned(AB_RUN,'wasm-q8-rounding-ab-v1-arrays-'+str(AB_RUN),out/'ab-raw.zip',AB_RAW)
    files=zip_files(out/'ab-raw.zip')
    for name,identity in ab['artifact_files'].items():
        if name not in files or identity!={'bytes':len(files[name]),'sha256':digest(files[name])}:
            raise ValueError('A/B raw manifest differs: '+name)
    for name,data in files.items():
        target=out/'frozen'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    proof['raw_members']={n:{'bytes':len(b),'sha256':digest(b)} for n,b in files.items()}
    # Exact original model bytes only from the original immutable fixture.
    proof['services']['fixture']=service(INITIAL_RUN,INITIAL_SOURCE,'failure')
    proof['artifacts']['fixture']=pinned(INITIAL_RUN,'successor-minilm-fixture-'+str(INITIAL_RUN),out/'fixture.zip',
        {'id':FIXTURE_ID,'bytes':FIXTURE_BYTES,'sha256':FIXTURE_SHA})
    fixture=zip_files(out/'fixture.zip');models=[n for n in fixture if n.endswith('/all-MiniLM-L6-v2-iq4_xs.gguf')]
    if len(fixture)!=2 or len(models)!=1 or digest(fixture['consumer.json'])!=NATIVE_REPORT_SHA or digest(fixture[models[0]])!=MODEL_SHA:
        raise ValueError('Frozen model/native fixture differs')
    model=fixture[models[0]];(out/'model.gguf').write_bytes(model)
    _,inventory=gguf(model);weights={r['name']:r for r in inventory if r['name'] in WEIGHTS}
    if set(weights)!=set(WEIGHTS):raise ValueError('Actual layer3 weight inventory differs')
    # Determine each actual stored type before choosing an activation converter.
    sizes={20:(32,18),23:(256,136),8:(32,34)}
    for name,row in weights.items():
        if row['type'] not in sizes or len(row['shape'])!=2:raise ValueError('Unsupported actual layer3 weight type')
        block,size=sizes[row['type']];k,m=row['shape']
        if k%block:raise ValueError('Actual weight block width differs')
        n=k//block*size*m;data=model[row['data_offset']:row['data_offset']+n]
        if len(data)!=n:raise ValueError('Truncated actual weight')
        row['bytes']=n;row['sha256']=digest(data)
    proof['weights']=weights
    save(out/'prepared.json',proof)
    (out/'browser/arrays').mkdir(parents=True)
    shutil.copy2(out/'model.gguf',out/'browser/model.gguf')
    save(out/'browser/weights.json',weights)
    save(out/'review/prepared.json',proof)


def build(out):
    hosted()
    source=out/'source'
    subprocess.run(['git','worktree','add','--detach',str(source),BASE],cwd=ROOT,check=True,timeout=180)
    subprocess.run(['git','submodule','update','--init','--depth','1','ggml'],cwd=source,check=True,timeout=180)
    if git(source,'rev-parse','HEAD')!=BASE or git(source/'ggml','rev-parse','HEAD')!=GGML or sha(source/'src/crispembed.cpp')!=ENCODER:
        raise ValueError('Frozen build source differs')
    if git(source,'status','--porcelain') or git(source/'ggml','status','--porcelain'):
        raise ValueError('Disposable frozen source is dirty before overlay')
    changes=overlay(source,ROOT)
    save(out/'review/overlay.json',{'base':BASE,'ggml':GGML,'changes':changes,
        'encoder_diff':git(source,'diff','--','src/crispembed.cpp','wasm/embed_wrapper.c','build-embed-wasm.sh'),
        'ggml_diff':git(source/'ggml','diff','--','src/ggml-cpu/ggml-cpu.c'),'qualification':False})
    # Portable original release ISA/settings. Exact frozen anchors below are
    # mandatory: a toolchain-dependent numerical change blocks attribution.
    command(['cmake','-S',source,'-B',out/'native-build','-DCMAKE_BUILD_TYPE=Release',
             '-DGGML_NATIVE=OFF','-DGGML_BLAS=OFF','-DGGML_LLAMAFILE=ON',
             '-DGGML_METAL=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF',
             '-DCRISPEMBED_BUILD_SHARED=ON','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],out,'native-configure')
    command(['cmake','--build',out/'native-build','-j2','--target','crispembed-shared'],out,'native-build')
    command(['bash','build-embed-wasm.sh','--clean','--no-simd','--','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],out,'wasm-build',source)
    for name,path in [('native',out/'native-build/compile_commands.json'),('wasm',source/'build-embed-wasm/compile_commands.json')]:
        shutil.copy2(path,out/'review'/('compile-'+name+'.json'))
    compile_guard=check_wasm_commands(json.loads((out/'review/compile-wasm.json').read_text()))
    native_libraries=list((out/'native-build').rglob('libcrispembed.so'))
    if len(native_libraries)!=1:raise ValueError('Expected one diagnostic native shared library')
    symbols=subprocess.check_output(['nm','-D','--defined-only',str(native_libraries[0])],text=True,encoding='utf-8',timeout=60)
    (out/'review/native-symbols.txt').write_text(symbols)
    save(out/'review/build-controls.json',{'wasm_compile_commands':compile_guard,'native_export':check_native_export(symbols)})
    destination=out/'browser/diagnostic';destination.mkdir()
    for suffix in ('js','wasm'):shutil.copy2(source/('build-embed-wasm/crispembed_embed.'+suffix),destination)
    binaries=list((out/'native-build').rglob('libcrispembed.so'))+[destination/'crispembed_embed.js',destination/'crispembed_embed.wasm']
    save(out/'review/build-artifacts.json',{str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in binaries})
    command(['c++','-std=c++17','-O2','-I'+str(ROOT/'src'),ROOT/'tests/test_wasm_layer3_diff.cpp','-o',out/'stage-diff'],out,'diff-build',timeout=120)


def library(out,threads):
    paths=list((out/'native-build').rglob('libcrispembed.so'))
    if len(paths)!=1:raise ValueError('Expected one diagnostic shared library')
    lib=C.CDLL(str(paths[0]));lib.crispembed_init.argtypes=[C.c_char_p,C.c_int];lib.crispembed_init.restype=C.c_void_p
    lib.crispembed_encode.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_int)];lib.crispembed_encode.restype=C.POINTER(C.c_float)
    lib.crispembed_free.argtypes=[C.c_void_p];lib.crispembed_free.restype=None
    lib.crispembed_layer3_replay.argtypes=[C.c_void_p,C.c_char_p,C.POINTER(C.c_float),C.c_int,C.POINTER(C.c_float),C.c_int]
    lib.crispembed_layer3_replay.restype=C.c_int
    ctx=lib.crispembed_init(os.fsencode(out/'model.gguf'),threads)
    if not ctx:raise ValueError('Diagnostic init failed')
    return lib,ctx


def native_worker(out,threads,mode):
    hosted()
    os.environ['CRISPEMBED_DEBUG_TOKENS']='1';os.environ['CRISPEMBED_EMBED_RAW_NORM']='1'
    os.environ.pop('CRISPEMBED_WASM_Q8_0_ROUND',None)
    prefix=out/'native'/str(threads)/mode;prefix.mkdir(parents=True)
    for key in ('CRISPEMBED_DUMP_LAYERS','CRISPEMBED_DUMP_LAYERS_GGUF','CRISPEMBED_LAYER3_TRACE','CRISPEMBED_LAYER3_KERNEL'):
        os.environ.pop(key,None)
    if mode=='capture':
        os.environ['CRISPEMBED_DUMP_LAYERS']='1';os.environ['CRISPEMBED_DUMP_LAYERS_GGUF']=str(prefix/'stages.gguf')
        os.environ['CRISPEMBED_LAYER3_TRACE']='1';os.environ['CRISPEMBED_LAYER3_KERNEL']=str(prefix)
    lib,ctx=library(out,threads)
    try:
        n=C.c_int();pointer=lib.crispembed_encode(ctx,TEXT.encode(),C.byref(n))
        if not pointer or n.value!=384:raise ValueError('Diagnostic encode shape')
        raw=C.string_at(pointer,1536);(prefix/'final.f32').write_bytes(raw)
        save(out/'review'/('native-'+str(threads)+'-'+mode+'.json'),{'sha256':digest(raw),'vector':floats(raw)})
    finally:lib.crispembed_free(ctx)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)


def native(out):
    hosted()
    for thread in (1,2):
        for mode in ('ordinary','capture'):
            command([sys.executable,__file__,'worker','--output',out,'--threads',thread,'--mode',mode],out,
                    'native-'+str(thread)+'-'+mode,timeout=180)
    for file in (out/'native/1/capture').glob('*.input.f32'):
        shutil.copy2(file,out/'browser/arrays'/('native-'+file.name))


def replay_native(out):
    hosted()
    proof=json.loads((out/'review/anchors.json').read_text())
    if proof.get('passed') is not True or proof.get('source')!=os.environ['GITHUB_SHA'] or proof.get('ab_report_sha256')!=AB_REPORT:
        raise ValueError('Frozen anchors must pass before local replay')
    lib,ctx=library(out,1)
    try:
        os.environ.pop('CRISPEMBED_LAYER3_TRACE',None)
        for axis in ('native','candidate'):
            for weight in WEIGHTS:
                input_path=out/'browser/arrays'/(axis+'-'+weight+'.input.f32')
                values=floats(input_path.read_bytes());result=(C.c_float*(1536*12))();array=(C.c_float*len(values))(*values)
                destination=out/'replay/native'/axis;destination.mkdir(parents=True,exist_ok=True)
                os.environ['CRISPEMBED_LAYER3_KERNEL']=str(destination)
                n=lib.crispembed_layer3_replay(ctx,weight.encode(),array,len(values),result,len(result))
                if n<=0:raise ValueError('Native local replay failed: '+str(n))
                (destination/(weight+'.output.f32')).write_bytes(C.string_at(result,n*4))
    finally:lib.crispembed_free(ctx)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)


def kernel(out,prefix,weights,rounding):
    rows={}
    for name,expected in weights.items():
        p=prefix/name;meta=json.loads(p.with_suffix(p.suffix+'.json').read_text())
        # Prefix names already end in .weight; never strip that suffix.
        get=lambda suffix:Path(str(p)+'.'+suffix).read_bytes()
        weight,converted,input_bytes,dots=get('weight.bin'),get('converted.bin'),get('input.f32'),get('dots.f32')
        if (meta['wasm_round_plan']!=int(rounding) or meta['weight']!=name or meta['weight_type']!=expected['type'] or meta['vec_dot_type']!=(15 if expected['type']==23 else 8) or meta['input_shape']!=[expected['shape'][0],12] or
                meta['output_shape']!=[expected['shape'][1],12] or len(weight)!=expected['bytes'] or digest(weight)!=expected['sha256']):
            raise ValueError('Captured actual weight differs from original GGUF: '+name)
        if len(input_bytes)!=4*math.prod(meta['input_shape']) or len(dots)!=48:raise ValueError('Actual kernel array shape differs')
        rows[name]={'metadata':meta,'input_sha256':digest(input_bytes),'converted_sha256':digest(converted),
                    'weight_sha256':digest(weight),'dots':floats(dots),
                    'conversion_blocks':conversion_blocks(converted,meta['vec_dot_type'],expected['shape'][0])}
    return rows


def capture_controls(out,prepared,ab,browser):
    old=json.loads((out/'old-panel-freeze.json').read_text())['cases']['original']
    expected_ids=old['token_ids']
    report={'anchors':{},'kernels':{}}
    captures={}
    for label in ('native1','native2','wasm0','wasm1'):
        if label.startswith('native'):
            thread=label[-1];path=out/'native'/thread/'capture/stages.gguf'
            ordinary=(out/'native'/thread/'ordinary/final.f32').read_bytes();retained=(out/'native'/thread/'capture/final.f32').read_bytes()
            for mode in ('ordinary','capture'):
                if tokens((out/('native-'+thread+'-'+mode+'.log')).read_text())!=expected_ids:
                    raise ValueError('Native original IDs differ')
            # Native final is bound directly to original retained native bytes.
            frozen=out/'frozen/native-vectors'/('original-'+thread+'.f32')
            if ordinary!=retained or ordinary!=frozen.read_bytes() or digest(ordinary)!=old['native'][thread]['sha256']:raise ValueError('Native ordinary/observer/frozen final differs')
            stage_hashes={r['stage']:r['native_sha256'] for r in ab['first_sentence_stages']['1'][thread]}
            prefix=out/'native'/thread/'capture'
        else:
            gate=label[-1];path=out/'browser/arrays'/('wasm'+gate+'-stages.gguf')
            row=browser['captures'][gate]
            if any(tokens('\n'.join(value['logs']))!=expected_ids for value in row.values()):
                raise ValueError('WASM original IDs differ')
            ordinary=f32_bytes(row['ordinary']['vector']);retained=f32_bytes(row['capture']['vector'])
            if ordinary!=retained or digest(ordinary)!=ab['cases']['original'][gate]['sha256']:
                raise ValueError('WASM ordinary/observer/frozen final differs')
            stage_hashes={r['stage']:r['candidate_sha256'] for r in ab['first_sentence_stages'][gate]['1']}
            prefix=out/'browser/arrays'/('wasm'+gate)
        rows=layer_rows(path.read_bytes());captures[label]=rows
        if any(rows[name]['sha256']!=value for name,value in stage_hashes.items()):raise ValueError('Old eight stage anchors differ: '+label)
        report['anchors'][label]={'final_sha256':digest(ordinary),'old_eight_stages_exact':True,'ordinary_observer_exact':True}
        kernels=kernel(out,prefix,prepared['weights'],label=='wasm1');report['kernels'][label]=kernels
        for weight,(source,target) in WEIGHTS.items():
            if kernels[weight]['input_sha256']!=rows['layer3_'+source]['sha256']:
                raise ValueError('Captured kernel input is not the observed graph input')
    return captures,report


def anchors(out):
    hosted();prepared=json.loads((out/'prepared.json').read_text())
    ab=trusted_ab(json.loads((out/'ab.json').read_text()))
    browser=json.loads((out/'browser/report.json').read_text())
    if browser['state']!='captured' or browser['page_errors']:
        raise ValueError('Complete neutral captures required before replay')
    _,proof=capture_controls(out,prepared,ab,browser)
    proof.update(source=os.environ['GITHUB_SHA'],passed=True,ab_report_sha256=AB_REPORT,qualification=False)
    save(out/'review/anchors.json',proof)
    save(out/'browser/anchors.json',proof)


def compare(out):
    hosted();prepared=json.loads((out/'prepared.json').read_text());ab=trusted_ab(json.loads((out/'ab.json').read_text()))
    browser=json.loads((out/'browser/report.json').read_text())
    if browser['state']!='complete' or browser['page_errors']:raise ValueError('Browser capture/replay incomplete')
    old=json.loads((out/'old-panel-freeze.json').read_text())['cases']['original']
    expected_ids=old['token_ids']
    if f32_bytes(browser['replay']['vector_after_replay'])!=f32_bytes(ab['cases']['original']['1']['vector']):
        raise ValueError('WASM ordinary output changed after local operator replay')
    if tokens('\n'.join(browser['replay']['logs']))!=expected_ids:
        raise ValueError('WASM token IDs changed after replay')
    report={'schema':'crispembed-wasm-layer3-attribution-v1','source':os.environ['GITHUB_SHA'],'base':BASE,'ggml':GGML,
            'model_sha256':MODEL_SHA,'qualification':False,'anchors':{},'subnodes':{},'kernels':{},'replays':{},'weights':prepared['weights'],
            'limits':['Original sentence only. Attribution never qualifies the rejected full panel.',
                      'Same-input projection replay does not replace whole-network or nonlinear boundary evidence.']}
    captures,controls=capture_controls(out,prepared,ab,browser)
    report.update(controls)
    for name in NAMES:
        n,w=captures['native1'][name],captures['wasm1'][name]
        m=metrics(n['values'],w['values']);report['subnodes'][name]=m
        m['native1_native2_bit_exact']=n['sha256']==captures['native2'][name]['sha256']
    command([out/'stage-diff',out/'native/1/capture/stages.gguf',out/'browser/arrays/wasm1-stages.gguf',*NAMES],out,'subnode-diff',timeout=120)
    cpp=[json.loads(line) for line in (out/'subnode-diff.log').read_text().splitlines() if line.startswith('{')]
    if [r['stage'] for r in cpp]!=list(NAMES):raise ValueError('crispembed-diff subnode coverage differs')
    for row in cpp:
        m=report['subnodes'][row['stage']]
        if abs(m['max_abs']-row['max_abs'])>max(1e-8,m['max_abs']*1e-5) or abs(m['cosine']-row['cos_global'])>2e-6:
            raise ValueError('Independent subnode comparator differs')
        m['crispembed_diff']=row
    for axis in ('native','candidate'):
        native_prefix=out/'replay/native'/axis;wasm_prefix=out/'browser/arrays/replay'/axis
        nk,wk=kernel(out,native_prefix,prepared['weights'],False),kernel(out,wasm_prefix,prepared['weights'],True)
        for weight,(source,target) in WEIGHTS.items():
            nv=floats((native_prefix/(weight+'.output.f32')).read_bytes());wv=floats((wasm_prefix/(weight+'.output.f32')).read_bytes())
            source_key='native1' if axis=='native' else 'wasm1'
            if nk[weight]['input_sha256']!=wk[weight]['input_sha256'] or nk[weight]['input_sha256']!=captures[source_key]['layer3_'+source]['sha256']:
                raise ValueError('Same-input replay axes differ')
            # Own-input local replay must reproduce the original full graph raw
            # projection before any cross-backend result can be interpreted.
            own=nv if axis=='native' else wv
            if f32_bytes(own)!=f32_bytes(captures[source_key]['layer3_'+target]['values']):
                raise ValueError('Own-input local operator does not reproduce original full graph')
            report['replays'][axis+'/'+weight]={'comparison':row_metrics(nv,wv,prepared['weights'][weight]['shape'][1]),
                'native_kernel':nk[weight],'wasm_kernel':wk[weight],
                'activation_bytes_exact':nk[weight]['converted_sha256']==wk[weight]['converted_sha256'],
                'activation_byte_mismatches':sum(a!=b for a,b in zip((native_prefix/(weight+'.converted.bin')).read_bytes(),(wasm_prefix/(weight+'.converted.bin')).read_bytes())),
                'scale_mismatches':sum(a['scale']!=b['scale'] for ar,br in zip(nk[weight]['conversion_blocks'],wk[weight]['conversion_blocks']) for a,b in zip(ar,br)),
                'selected_first_channel_dots_comparison':metrics(nk[weight]['dots'],wk[weight]['dots'])}
    report['state']='attribution-complete'
    report['earliest_named_gate_failure']=next((name for name,m in report['subnodes'].items() if m['max_abs']>.002 or m['crispembed_diff']['cos_min']<.99999),None)
    report['raw_members']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)} for folder in ('native','replay','browser/arrays') for p in (out/folder).rglob('*') if p.is_file()}
    save(out/'review/layer3.json',report)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','build','native','worker','anchors','replay-native','compare'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,choices=[1,2]);p.add_argument('--mode',choices=['ordinary','capture']);a=p.parse_args()
    if a.phase=='worker' and (a.threads is None or a.mode is None):
        p.error('worker requires explicit --threads and --mode')
    try:
        if a.phase=='worker':native_worker(a.output,a.threads,a.mode)
        elif a.phase=='replay-native':replay_native(a.output)
        else:globals()[a.phase](a.output)
    except Exception as error:
        save(a.output/'review'/('rejection-'+a.phase+'.json'),{'phase':a.phase,'error':str(error),'qualification':False})
        raise

if __name__=='__main__':main()
