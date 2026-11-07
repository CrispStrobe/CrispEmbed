"""Bounded Kaggle physical-only F16 attention controls, no model forward."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from runtime_guard import Deadline, command
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def require(ok,message):
    if not ok:raise RuntimeError(message)

def text_schema(plan):
    rows=[];counts={0:0,12:0,14:0};names=set()
    for row in plan['all_tensors']:
        if not row['name'].startswith('l.'):continue
        name=row['name'];kind=14 if name=='l.embed_tokens.weight'else row['type']
        require(name not in names and kind in counts,'closed text schema');names.add(name);counts[kind]+=1
        dims=list(reversed(row['source_shape']));dims += [1]*(4-len(dims))
        rows.append(name+'\t'+str(kind)+'\t'+'\t'.join(map(str,dims))+'\n')
    require(len(rows)==426 and counts=={0:177,12:216,14:33},'exact426 schema inventory')
    return ''.join(rows)


def control_args(target,build,schema):
    schema_targets={'test-lighton4b-upstream-physical','test-lighton4b-hf-tiled-out-recipe'}
    fixture_targets={'test-lighton4b-release-fence'}
    noarg_targets={'test-lighton4b-physical-f16-attention','test-lighton4b-physical-f16-unsupported','test-lighton4b-physical-f16-wasm-guard','test-lighton4b-physical-native-ops','test-lighton4b-hf-tiled-out-operator','test-lighton4b-q6-tied-head'}
    require(target in schema_targets|fixture_targets|noarg_targets,'unknown control CLI')
    if target in schema_targets:return [build/target,schema]
    if target in fixture_targets:return [build/target,schema.parent/'release-fence-fixture.gguf']
    return [build/target]


BUILD_FLAGS={'CRISPEMBED_NATIVE':'OFF','GGML_NATIVE':'OFF','GGML_AVX2':'ON','GGML_FMA':'ON','GGML_F16C':'ON','GGML_LLAMAFILE':'ON','GGML_AVX512':'OFF','GGML_BLAS':'OFF','GGML_BACKEND_DL':'OFF','GGML_OPENMP':'OFF','GGML_CUDA':'OFF','GGML_VULKAN':'OFF'}

def validate_profile(lines,reference=False):
    for key,value in BUILD_FLAGS.items():
        if reference and key=='CRISPEMBED_NATIVE':continue
        if reference and key=='GGML_OPENMP':value='ON' # Actual pinned upstream build; native app uses OFF.
        require(f'{key}:BOOL={value}'in lines,('reference'if reference else'native')+' profile '+key)


def main():
    require(Path('/kaggle/working').is_dir()and Path('/kaggle/input').is_dir()and not list(Path('/dev').glob('nvidia[0-9]*')),'Kaggle CPU only')
    import numpy as np
    require(np.__version__=='2.3.5','numpy pin')
    cfg=json.loads((HERE/'fixtures-v1.json').read_text())
    out=Path(os.environ['HEAVY_OUT']);out.mkdir(parents=True,exist_ok=True)
    work=Path(os.environ['HEAVY_SCRATCH'])/'physical-f16-integration';work.mkdir()
    started=time.monotonic()
    report={'schema':cfg['schema'],'status':'failed','commands':{},'sourcefed':{},'model_forward_executed':False,'quality_qualified':False}
    def save():(out/'physical-f16-controls.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(label,args):
        report['phase']=label;report['commands'][label]={};save()
        try:command(args,out/(label+'.log'),min(600,1200-(time.monotonic()-started)),report['commands'][label])
        finally:save()
    deadline=Deadline(1200);deadline.__enter__()
    try:
        run('source-pin',['git','-C',ROOT,'rev-parse','HEAD']);source=(out/'source-pin.log').read_text().strip();require(source==os.environ['LIGHTON4B_EXPECTED_SOURCE'],'source pin');report['source_sha']=source
        available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
        limit=Path('/sys/fs/cgroup/memory.max').read_text().strip()
        if limit!='max':available=min(available,int(limit)-int(Path('/sys/fs/cgroup/memory.current').read_text()))
        report['preflight']={'available_bytes':available,'disk_free_bytes':shutil.disk_usage(work).free,'load':os.getloadavg()}
        require(available>=2<<30 and shutil.disk_usage(work).free>=4<<30,'resource admission')
        matches=[p for p in Path('/kaggle/input').rglob('attention-primitives.json')if p.is_file()and p.stat().st_size<1<<20 and sha(p)==cfg['frozen_report_sha256']]
        require(len(matches)==1,'one frozen actual attention V2 receipt');frozen=matches[0];old=json.loads(frozen.read_text())
        require(old['status']=='measured_operator_only'and old['source_sha']==cfg['frozen_source_sha']and old['actual_native_hidden_closed'],'frozen trace admission')
        report['frozen_receipt_sha256']=sha(frozen)
        for rel,row in cfg['files'].items():
            p=frozen.parent/rel;require(p.stat().st_size==row['bytes']and sha(p)==row['sha256'],'fixture '+rel)
        up=work/'llama'
        for label,args in [('up-init',['git','init',up]),('up-remote',['git','-C',up,'remote','add','origin','https://github.com/ggml-org/llama.cpp']),('up-fetch',['git','-C',up,'fetch','--depth','1','origin',cfg['upstream_commit']]),('up-checkout',['git','-C',up,'checkout','--detach','FETCH_HEAD']),('native-submodule',['git','-C',ROOT,'submodule','update','--init','ggml']),('native-pin',['git','-C',ROOT/'ggml','rev-parse','HEAD'])]:run(label,args)
        require((out/'native-pin.log').read_text().strip()==cfg['native_ggml_commit'],'native backend pin')
        sys.path.insert(0,str(ROOT/'tools'));import qwen35_source_pins as source_pins
        commits=sorted({row[-1]for row in source_pins.pins(ROOT)})
        require(commits and all(len(c)==40 and all(ch in '0123456789abcdef'for ch in c)for c in commits),'closed historical receipt commits')
        run('fetch-frozen-receipt-commits',['git','-C',ROOT,'fetch','--depth','1','origin',*commits])
        run('source-pins',['python3',ROOT/'tools/qwen35_source_pins.py','--check','--verify-git'])
        run('environment',['python3',ROOT/'tests/test_env_vars_documented.py'])
        build=work/'runtime-build'
        flags=dict(BUILD_FLAGS)
        run('runtime-configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release',*[f'-D{k}={v}'for k,v in flags.items()]])
        cache=(build/'CMakeCache.txt').read_text().splitlines()
        shutil.copy(build/'CMakeCache.txt',out/'runtime-CMakeCache.txt')
        validate_profile(cache)
        targets=['test-lighton4b-physical-f16-attention','test-lighton4b-physical-f16-unsupported','test-lighton4b-physical-f16-wasm-guard','lighton4b-physical-f16-attention-probe','test-lighton4b-physical-native-ops','test-lighton4b-upstream-physical','test-lighton4b-release-fence','test-lighton4b-hf-tiled-out-recipe','test-lighton4b-hf-tiled-out-operator','test-lighton4b-q6-tied-head']
        run('runtime-build',['cmake','--build',build,'--target',*targets,'--parallel','2'])
        sys.path.insert(0,str(ROOT/'tools'));from lighton4b_q6_recipe import plan
        schema=out/'closed426.tsv';schema.write_text(text_schema(plan()))
        primary={'test-lighton4b-physical-f16-attention','test-lighton4b-physical-f16-unsupported','test-lighton4b-physical-f16-wasm-guard'}
        for target in targets:
            if target in primary:
                run(target,control_args(target,build,schema))
                if target=='test-lighton4b-physical-f16-attention':require('PHYSICAL_F16_ATTENTION PASS 'in(out/(target+'.log')).read_text(),'supported runtime controls required, not skipped')
        run('runtime-sourcefed',[build/'lighton4b-physical-f16-attention-probe',frozen.parent/'native',out/'runtime'])
        ref=work/'reference-build'
        run('reference-configure',['cmake','-S',HERE,'-B',ref,'-DCMAKE_BUILD_TYPE=Release','-DGGML_SOURCE='+str(up/'ggml')])
        rc=(ref/'CMakeCache.txt').read_text().splitlines()
        shutil.copy(ref/'CMakeCache.txt',out/'reference-CMakeCache.txt')
        validate_profile(rc,reference=True)
        report['thread_engines']={'native_openmp':False,'pinned_upstream_reference_openmp':True}
        run('reference-build',['cmake','--build',ref,'--target','physical-f16-attention-reference','--parallel','2'])
        cases=[('prefill4',4,0),('prefill16',16,0)]+[(f'step{t:02}',1,t)for t in range(16)]
        for label,n,past in cases:
            for mode in ('composed','observed'):run(f'reference-{label}-{mode}',[ref/'physical-f16-attention-reference',frozen.parent/'native',out/'reference'/label/mode,n,'f16',mode,past])
            gold=out/'reference'/label/'observed/gated.f32'
            require(gold.read_bytes()==(out/'reference'/label/'composed/gated.f32').read_bytes(),'reference observer '+label)
            actual=out/'runtime'/label/'gated.f32'
            require(actual.stat().st_size==n*4096*4,'runtime output extent')
            x=np.fromfile(actual,dtype='<f4');y=np.fromfile(gold,dtype='<f4');d=x.astype('f8')-y
            report['sourcefed'][label]={'bit_exact':actual.read_bytes()==gold.read_bytes(),'rms':float(np.sqrt(np.mean(d*d))),'max_abs':float(abs(d).max()),'past':past,'new_tokens':n}
            require(report['sourcefed'][label]['bit_exact'],'runtime/source byte closure '+label)
            if past==0 and n in(4,16):require(gold.read_bytes()==(frozen.parent/f'upstream/{n}/f16/observed/gated.f32').read_bytes(),'fresh original source profile '+label)
        require((out/'runtime/prefill16/keys.f32').read_bytes()==(out/'runtime/step15/keys.f32').read_bytes()and(out/'runtime/prefill16/values.f32').read_bytes()==(out/'runtime/step15/values.f32').read_bytes(),'prefill/tokenwise canonical cache identity')
        for target in targets:
            if target not in primary and target!='lighton4b-physical-f16-attention-probe':run(target,control_args(target,build,schema))
        report['status']='model_free_sourcefed_pass';report['legacy_gates_default_unchanged']=True;report['public_release_fence_preserved']=True
    except BaseException as error:report['error_type'],report['error']=type(error).__name__,str(error);raise
    finally:
        deadline.close();report['elapsed_seconds']=time.monotonic()-started
        report['artifacts']={str(p.relative_to(out)):{'bytes':p.stat().st_size,'sha256':sha(p)}for p in sorted(out.rglob('*'))if p.is_file()and p.name!='physical-f16-controls.json'};save()

if __name__=='__main__':main()
