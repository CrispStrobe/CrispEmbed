"""Hosted actual Rust/Dart wrappers vs unchanged preserved original CLS arrays."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ultrabert_contract import QUERY, sha256
from ultrabert_native import prepare_frozen_artifacts
# The hosted producer and model-free scorer intentionally share a basename.
# Load the producer explicitly; a plain import can resolve to the scorer.
_spec = importlib.util.spec_from_file_location('qualified_ultrabert_task', Path(__file__).with_name('ultrabert_retrieval.py'))
_qualification = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_qualification)
qualification = _qualification.qualification


def compare(expected, actual):
    if (set(actual.get('contracts', {})) != {'repeat','batch','context_prefix','rejection_recovery'}
            or not all(v is True for v in actual['contracts'].values())
            or [x.get('id') for x in actual.get('rows', [])] != [x['id'] for x in expected]):
        raise ValueError('Missing/reordered binding cases or contracts')
    result=[]
    for reference, row in zip(expected, actual['rows']):
        if row.get('ids') != reference['ids']:raise ValueError('Binding token IDs changed')
        for mode, target in [('raw_cls','raw_cls'),('dense','normalized_cls'),('token_normalized_cls','normalized_cls')]:
            a=reference[target];b=row.get(mode,[])
            if len(a)!=768 or len(b)!=768 or not all(type(x) in (int,float) and math.isfinite(x) for x in a+b):
                raise ValueError('Missing/nonfinite binding vectors')
            an=math.sqrt(math.fsum(x*x for x in a));bn=math.sqrt(math.fsum(x*x for x in b))
            if min(an,bn)<=0:raise ValueError('Zero binding vector')
            cosine=math.fsum(x*y for x,y in zip(a,b))/(an*bn)
            error=max(abs(x-y) for x,y in zip(a,b))
            result.append(dict(id=row['id'],mode=mode,cosine=cosine,max_abs=error,
                mean_abs=math.fsum(abs(x-y) for x,y in zip(a,b))/768,
                rms=math.sqrt(math.fsum((x-y)**2 for x,y in zip(a,b))/768),
                native_norm=bn,reference_norm=an,passed=cosine>=.99999 and error<=.002))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('frozen',type=Path);args=parser.parse_args()
    scratch=Path(os.environ['HEAVY_SCRATCH'])/'ultrabert-bindings';scratch.mkdir(parents=True,exist_ok=True)
    out=Path(os.environ['HEAVY_OUT']);out.mkdir(parents=True,exist_ok=True)
    report=dict(source_commit=os.environ['GITHUB_SHA'],driver_sha256=sha256(__file__),status='started',surfaces={})
    def save():(out/'ultrabert-bindings.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    def run(command,name,env=None,cwd=ROOT):
        proc=subprocess.run([str(x) for x in command],env=env,cwd=cwd,capture_output=True,text=True,timeout=3600)
        (out/(name+'.log')).write_text(proc.stdout+'\n'+proc.stderr)
        if proc.returncode:raise ValueError(name+' failed')
        return proc.stdout
    save()
    try:
        receipt,proof=qualification('39943940607','4be9f864e0e7bc154325e1edc1bdf29a12210a69',scratch,out)
        for name,digest in receipt['native_source_sha256'].items():
            if sha256(ROOT/name)!=digest:raise ValueError('Qualified native source changed')
        report['api_qualification']=proof;report['native_source_sha256']=receipt['native_source_sha256'];save()
        original,refs=prepare_frozen_artifacts(json.loads((ROOT/'tests/results/ultrabert/reference-audit.json').read_text()),args.frozen,scratch)
        report['model_sha256']=sha256(scratch/'f32.gguf')
        if report['model_sha256']!=receipt['f32_model_sha256']:raise ValueError('Qualified model changed')
        import gguf
        expected=[];requests=[]
        for record in original['reference']['cases']:
            case=record['case']
            if 'text_pair' in case:continue
            texts=case['text'] if isinstance(case['text'],list) else [case['text']]
            prefix=QUERY if case.get('prompt')=='query' else ''
            reader=gguf.GGUFReader(str(refs/record['archive']));arrays={t.name:t.data for t in reader.tensors if t.name in ('cls_raw','cls_normalized')}
            for i,text in enumerate(texts):
                name=case['id']+'-'+str(i)
                requests.append(dict(id=name,text=text,prefix=prefix))
                expected.append(dict(id=name,ids=[v for v,keep in zip(record['ids'][i],record['attention_mask'][i]) if keep],
                    raw_cls=arrays['cls_raw'].reshape(len(texts),768)[i].tolist(),
                    normalized_cls=arrays['cls_normalized'].reshape(len(texts),768)[i].tolist()))
            del arrays,reader
        if len(expected)!=7:raise ValueError('Original binding coverage changed')
        request=scratch/'request.json';request.write_text(json.dumps(requests,ensure_ascii=False))
        (out/'expected-pooled.json').write_text(json.dumps(expected,indent=2)+'\n')
        report['original_source']=original['source_commit'];report['expected_sha256']=sha256(out/'expected-pooled.json')
        build=scratch/'build'
        run(['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCRISPEMBED_BUILD_SHARED=ON',
             '-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_BLAS=OFF','-DGGML_METAL=OFF'],'configure')
        run([sys.executable,ROOT/'scripts/check-cpu-baseline.py',build],'cpu-baseline')
        run(['cmake','--build',build,'--target','crispembed-shared','--parallel','2'],'build')
        lib=next(build.rglob('libcrispembed.so')).resolve();report['library_sha256']=sha256(lib)
        env={**os.environ,'CRISPEMBED_SYS_LIB_DIR':str(lib.parent),'CARGO_BUILD_JOBS':'2',
             'CRISPEMBED_ULTRABERT':'1','CRISPEMBED_FORCE_CPU':'1','OMP_NUM_THREADS':'2',
             'LD_LIBRARY_PATH':str(lib.parent)}
        rust=scratch/'rust';(rust/'src').mkdir(parents=True)
        (rust/'Cargo.toml').write_text('[package]\nname="ultrabert-probe"\nversion="0.0.0"\nedition="2021"\n[dependencies]\nserde_json="1.0"\ncrispembed={path='+json.dumps(str(ROOT/'crispembed'))+'}\n')
        shutil.copy(ROOT/'tests/ultrabert_bindings.rs',rust/'src/main.rs')
        run(['cargo','build','--manifest-path',rust/'Cargo.toml'],'rust-build',env)
        rust_binary=rust/'target/debug/ultrabert-probe'
        ldd=run(['ldd',rust_binary],'rust-loaded-library',env)
        resolved=[line.split('=>',1)[1].strip().split()[0] for line in ldd.splitlines() if 'libcrispembed.so =>' in line]
        if len(resolved)!=1 or Path(resolved[0]).resolve()!=lib:raise ValueError('Rust library fallback')
        dart=scratch/'dart';(dart/'lib').mkdir(parents=True)
        for name in ('crispembed.dart','crispembed_bindings.dart'):
            shutil.copy(ROOT/'flutter/crispembed/lib/src'/name,dart/'lib'/name)
            if sha256(dart/'lib'/name)!=sha256(ROOT/'flutter/crispembed/lib/src'/name):raise ValueError('Dart source changed')
        shutil.copy(ROOT/'tests/ultrabert_bindings.dart',dart/'main.dart')
        (dart/'pubspec.yaml').write_text("name: ultrabert_probe\nenvironment:\n  sdk: '>=3.0.0 <4.0.0'\ndependencies:\n  ffi: ^2.0.0\n")
        run(['dart','--disable-analytics'],'dart-analytics')
        run(['dart','pub','get'],'dart-dependencies',cwd=dart)
        run(['dart','analyze','--no-fatal-warnings'],'dart-analyze',cwd=dart)
        commands={'rust':[rust_binary],'dart':['dart','run',dart/'main.dart']}
        for language,base in commands.items():
            suffix=[lib] if language=='dart' else []
            gate_results={}
            for gate in ('unset','0','f16'):
                gated=env.copy();gated.pop('CRISPEMBED_ULTRABERT',None)
                if gate!='unset':gated['CRISPEMBED_ULTRABERT']='0' if gate=='0' else '1'
                model=scratch/('f16.gguf' if gate=='f16' else 'f32.gguf')
                rejection=json.loads(run(base+[model,'--reject']+suffix,language+'-gate-'+gate,gated))
                if rejection!={'rejected':True}:raise ValueError('Binding gate did not reject')
                gate_results[gate]=True
            actual=json.loads(run(base+[scratch/'f32.gguf',request]+suffix,language+'-actual',env))
            (out/(language+'-actual.json')).write_text(json.dumps(actual,indent=2)+'\n')
            metrics=compare(expected,actual)
            report['surfaces'][language]=dict(metrics=metrics,contracts=actual['contracts'],gates=gate_results,
                passed=all(m['passed'] for m in metrics));save()
        report['source_sha256']={name:sha256(ROOT/name) for name in ['crispembed/src/lib.rs','crispembed-sys/build.rs',
            'flutter/crispembed/lib/src/crispembed.dart','flutter/crispembed/lib/src/crispembed_bindings.dart',
            'tests/ultrabert_bindings.rs','tests/ultrabert_bindings.dart']}
        report['rust_version']=run(['rustc','--version'],'rust-version').strip()
        report['dart_version']=run(['dart','--version'],'dart-version').strip()
        shutil.copy(rust/'Cargo.lock',out/'consumer-Cargo.lock');shutil.copy(dart/'pubspec.lock',out/'consumer-pubspec.lock')
        report['passed']=all(v['passed'] for v in report['surfaces'].values())
        report['status']='completed bounded Rust/Dart F32 wrapper audit'
        save();return 0 if report['passed'] else 1
    except BaseException as error:
        report['status']='failed producer';report['error_type']=type(error).__name__;save();raise


if __name__=='__main__':raise SystemExit(main())
