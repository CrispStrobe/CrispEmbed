"""Hosted composed A/B using immutable original references; never recapture gold."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import ovisocr2_large_inputs as old
import ovisocr2_2048_attribution as frozen

ROOT=old.ROOT
BASE='9d107710b0fc59f1e0d3fd3ac491db28fe74e2a3'
VISION='bc479ff5438138f55815769c59776bd761e2d6189182c137ed2d8fa82b01caa3'
CASES=('frozen-control','control-square','large-landscape','large-boundary')

def preflight():
    if os.environ.get('GITHUB_ACTIONS')!='true':
        raise ValueError('Hosted execution only')
    panel=old.checked_panel()
    # New independently scoped contract; historical helpers keep their old guards.
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'src','tests/qwen35_parity.h',
        'tests/test_qwen35_decoder.cpp','tests/test_qwen35_delta.cpp','tests/test_qwen35_vision.cpp',
        'tests/test_ovisocr2_context.cpp','tests/ovisocr2_quality_run.cpp'],cwd=ROOT,text=True).splitlines()
    changed=[]
    for name in paths:
        before=subprocess.check_output(['git','show',BASE+':'+name],cwd=ROOT)
        if (ROOT/name).read_bytes()!=before:changed.append(name)
    if changed!=['src/qwen35_vision.cpp'] or old.audit.sha(ROOT/changed[0])!=VISION:
        raise ValueError('Unexpected source/comparator changes: '+repr(changed))
    expected=subprocess.check_output(['git','ls-tree',BASE,'ggml'],cwd=ROOT,text=True).split()[2]
    actual=subprocess.check_output(['git','rev-parse','HEAD:ggml'],cwd=ROOT,text=True).strip()
    if actual!=expected:raise ValueError('ggml gitlink changed')
    return {'baseline':BASE,'changed_files':changed,'vision_sha256':VISION,'ggml_gitlink':actual,
            'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}


def run(case,gate):
    closure=preflight();out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/case
    out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    wrapper=json.loads((ROOT/'tests/results/qwen35-ocr/ovisocr2-larger-inputs.json').read_text())
    item=wrapper['cases'][case];producer=ROOT/item['producer_report_git_path']
    if old.audit.sha(producer)!=item['producer_report_sha256']:raise ValueError('Original producer changed')
    p=json.loads(producer.read_text())
    r={'schema':'crispembed-ovisocr2-position-composed-replay-v1','status':'started','case':case,'gate':gate,
       'closure':closure,'original_producer_sha256':item['producer_report_sha256'],'commands':{},'arms':{},
       'limits':['Original references, budgets, caps and gold unchanged. No default/release promotion.',
                 'Historical six rejected boundary arms remain recorded separately.']}
    def save():old.save_report(out,r)
    def command(label,args,env=None):
        result=old.timed_command(label,args,out,r,env,timeout=7200);save();return result
    save()
    try:
        build=Path(os.environ['AUDIT_BIN_DIR']);manifest=json.loads((build/'binaries.json').read_text())
        if manifest['source_commit']!=os.environ['GITHUB_SHA']:raise ValueError('Binary source differs')
        for name,digest in manifest['binary_sha256'].items():
            if old.audit.sha(build/name)!=digest:raise ValueError('Binary SHA differs')
            (build/name).chmod(0o700)
        r['build']=manifest
        if not command('bounds-controls',[build/'test-ovisocr2-image-bounds']):raise ValueError('Bounds/recovery controls failed')
        if shutil.disk_usage(scratch).free<24*1024**3:raise ValueError('At least24GiB remote scratch required')
        source=old.source_and_exports(out,scratch,r)
        if case=='frozen-control':
            reference=old.replay.source_reference(old.ab,scratch,out,r,source)
            image=scratch/'owned-readable.png'
        else:
            rec=item['large_reference_and_raw_artifact'];pin=(rec['id'],rec['name'],rec['bytes'],rec['zip_sha256'])
            r['retained_service']=old.ab.service_source(frozen.RUN,frozen.HEAD,pin,conclusion='failure')
            archive=scratch/'retained.zip';old.ab.download_artifact(pin,archive,timeout=3600)
            with zipfile.ZipFile(archive) as z:
                names=frozen.archive_members(z);matches=[n for n in names if n.endswith('official-delta.gguf')]
                if len(matches)!=1:raise ValueError('Ambiguous retained reference')
                prefix=matches[0].removesuffix('official-delta.gguf')
                frozen.extract(z,matches[0],scratch/'official-delta.gguf',p['original_files']['official-delta.gguf'])
                frozen.extract(z,prefix+'original.json',scratch/'original.json',p['original_files']['original.json'])
                image=scratch/'input.png'
                frozen.extract(z,prefix+'input.png',image,{'bytes':z.getinfo(prefix+'input.png').file_size,'sha256':p['input']['png_sha256']})
            archive.unlink()
            reference=json.loads((scratch/'original.json').read_text())
            if reference!=p['original_capture']:raise ValueError('Frozen original JSON/producer disagree')
        ref=scratch/'official-delta.gguf';r['reference_sha256']=old.audit.sha(ref)
        env={k:v for k,v in os.environ.items() if not k.startswith('CRISPEMBED_')}
        known=set(old.audit.NATIVE_ENV)|{'CRISPEMBED_QWEN35_VISION_PRECISE_NORM','CRISPEMBED_QWEN35_VISION_PRECISE_ATTN',
            'CRISPEMBED_QWEN35_VISION_CASCADE_NORM','CRISPEMBED_OVISOCR2_LARGE_IMAGES'}
        env.update({k:v for k,v in old.native_environment(case).items() if k in known})
        # Explicitly reset every arithmetic experiment inherited from a caller.
        for name in ('WEIGHT_CACHE','PARALLEL_F16','PRECISE_RMS'):
            env['CRISPEMBED_QWEN35_'+name]='0'
        env['CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION']=str(gate)
        r['native_gates']={k:v for k,v in env.items() if k.startswith('CRISPEMBED_')}
        failures=[]
        for precision,(filename,_,_) in old.ab.MODELS.items():
            for label,args in old.audit.native_arms(precision,build,scratch/filename,image,ref).items():
                folder=scratch/'native-diff'/label;folder.mkdir(parents=True,exist_ok=True)
                passed=command(label,args,dict(env,CRISPEMBED_DIFF_DUMP_DIR=str(folder)))
                log=(out/(label+'.log')).read_text()
                try:raw=old.raw_manifest(folder,reference,log,label)
                except (ValueError,KeyError) as e:raw={'coverage_complete':False,'error':str(e)}
                r['arms'][label]={'passed':passed and raw['coverage_complete'],'raw':raw}
                if not r['arms'][label]['passed']:failures.append(label)
                save()
            label=precision+'-owning-ocr'
            passed=command(label,[build/'test-ovisocr2-quality-run',scratch/filename,image,'32','2'],env)
            lines=[x.removeprefix('OVIS_QUALITY_RESULT ') for x in (out/(label+'.log')).read_text().splitlines() if x.startswith('OVIS_QUALITY_RESULT ')]
            if len(lines)!=1:failures.append(label);save();continue
            actual=json.loads(lines[0]);text=bytes.fromhex(actual['raw_text_utf8_hex']).decode()
            exact=(actual['token_ids']==reference['generated_ids'][0] and text==reference['decoded'][0]
                   and actual['stop_kind']==1 and actual['stop_id']==reference['generated_stop_id'])
            r.setdefault('owning_cases',{})[precision]={**actual,'raw_text':text,'original_ids_text_eos_exact':exact}
            if not passed or not exact:failures.append(label)
            save()
        if len(r['arms'])!=14:raise ValueError('Incomplete14-arm coverage')
        r['failures']=failures;r['status']='completed composed replay' if not failures else 'rejected composed numerical/decoded replay'
        save()
        if failures:raise ValueError('Composed gates failed; negative evidence retained')
    except Exception as e:
        r['error']=str(e);save();raise

def verify_quality(path):
    current=json.loads(Path(path).read_text())
    first=json.loads((ROOT/'tests/results/qwen35-ocr/ovisocr2-f16-quality.json').read_text())['producer']
    actual=current.get('native_cases',{}).get('f16',{});old_rows=first['native_cases']['f16']
    if len(actual)!=11 or set(actual)!=set(old_rows):raise ValueError('Full eleven-case coverage absent')
    keys=('input_rgb_sha256','input_png_sha256','raw_text','processed_text','token_ids','stop_id','finish_reason','stop_kind','cap')
    report={'schema':'crispembed-ovisocr2-position-quality-comparison-v1','cases':{},'source_head':os.environ.get('GITHUB_SHA'),
        'interpolation_gate':os.environ.get('CRISPEMBED_QWEN35_VISION_SOURCE_INTERPOLATION'),
        'confidence_policy':'Measure drift; no new confidence tolerance or exactness claim.'}
    failures=[]
    for name,reference in old_rows.items():
        row=actual[name];exact=all(row.get(k)==reference.get(k) for k in keys)
        values=row.get('confidences',[]);expected=reference['confidences']
        if len(values)!=len(expected):exact=False
        import math
        if not all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=1 for v in values):exact=False
        drift=max((abs(a-b) for a,b in zip(values,expected)),default=0)
        report['cases'][name]={'decoded_ids_text_eos_inputs_exact':exact,'token_count':len(row['token_ids']),
            'selected_confidences_exact':values==expected,'max_confidence_abs_drift':drift}
        if not exact:failures.append(name)
    report['failures']=failures;report['passed']=not failures
    destination=Path(path).parent/'position-quality-comparison.json'
    destination.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    if failures:raise ValueError('First-F16 decoded/input regression: '+repr(failures))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--verify-quality',type=Path)
    parser.add_argument('--case',choices=CASES);parser.add_argument('--gate',type=int,choices=(0,1));args=parser.parse_args()
    if args.verify_quality:print(json.dumps(verify_quality(args.verify_quality),indent=2))
    elif args.preflight:print(json.dumps(preflight(),indent=2))
    else:run(args.case,args.gate)
