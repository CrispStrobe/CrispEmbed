"""Hosted current-runtime qualification on frozen and new larger Ovis inputs.

Original module capture completes before native output for each new case. The
old943-field reference is reused byte-identically for the default-off control.
"""
import argparse
import hashlib
import json
import os
import re
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import ovisocr2_model_audit as audit
import ovisocr2_native_replay as replay
import ovisocr2_vision_audit as ab
from ovisocr2_published_consumer import cpu_record

ROOT=Path(__file__).resolve().parents[2]
PANEL=ROOT/'tests/results/qwen35-ocr/ovisocr2-large-input-panel.json'
PANEL_CANONICAL_SHA256='1bc48ad2ee2be1f3c791a79837f7869eb9c5a9a1a89749cc8184a236b81cb4f2'
PUBLIC_ENGINE22_WRAPPER_SHA256='9ee4e21e007d1ca3375008d3f5c45db283d8aa1ef5434992fdc895115de4b24a'
SOURCES=tuple(sorted(str(p.relative_to(ROOT)) for p in (ROOT/'src').glob('qwen35*') if p.suffix in ('.cpp','.h')))


def source_contract():
    old=json.loads((ROOT/'tests/results/qwen35-ocr/ovisocr2-f16-quality.json').read_text())['producer']['native_source_sha256']
    allowed={'src/qwen35_image.cpp','src/qwen35_vision.cpp','src/qwen35_frontend.h',
             'src/qwen35_tokenizer.cpp','src/qwen35_ocr.cpp','src/qwen35_ocr.h'}
    changed={name:{'old_sha256':value,'current_sha256':audit.sha(ROOT/name)}
             for name,value in old.items() if audit.sha(ROOT/name)!=value}
    if 'src/crispembed.cpp' in changed:
        # The bounded append-only engine22 map and fixed-task prompt rejection
        # are independently pinned, not a general allowance for C API changes.
        if changed['src/crispembed.cpp']['current_sha256']!=PUBLIC_ENGINE22_WRAPPER_SHA256:
            raise ValueError('Unreviewed public wrapper source changed')
        allowed.add('src/crispembed.cpp')
    if set(changed)-allowed:raise ValueError('Unreviewed unrelated arithmetic source changed')
    if set(SOURCES)-set(old)!={'src/qwen35_image_bounds.h'}:
        raise ValueError('Unexpected new native source file')
    # Shared comparisons, reductions, decoder, attention, recurrence, norms and
    # frozen original observer must retain their bytes. New branches are explicit.
    return {'source_sha256':{name:audit.sha(ROOT/name) for name in SOURCES},
            'intentional_changed_files':changed,'allowed_new_files':['src/qwen35_image_bounds.h'],
            'scope':'Intentional profile/bounds source changes; not an unchanged-source replay'}


def checked_panel():
    p=json.loads(PANEL.read_text())
    if hashlib.sha256(json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()!=PANEL_CANONICAL_SHA256:
        raise ValueError('Frozen authored larger-input panel bytes/contract changed')
    if (p['license']!='CC0-1.0' or p['default_visual_patch_limit']!=1024
        or p['optin_visual_patch_limit']!=2048 or p['max_new_tokens']!=32
        or [(c['id'],c['patches']) for c in p['cases']]!=[
            ('control-square',1024),('large-landscape',1200),('large-boundary',2048)]):
        raise ValueError('Frozen larger-input panel differs')
    return p


def preflight():
    p=checked_panel();s=source_contract()
    for path,digest in [(ab.__file__,replay.AB_HELPER_SHA),(audit.__file__,replay.MODEL_HELPER_SHA)]:
        if audit.sha(path)!=digest:raise ValueError('Frozen observer/comparator helper changed')
    return p,s


def save_report(out,report):
    # Command provenance is useful without embedding a runner's locations.
    text=json.dumps(report,indent=2,allow_nan=False)
    for name,value in [('repo',str(ROOT)),('scratch',os.environ.get('HEAVY_SCRATCH','')),('out',str(out))]:
        if value:text=text.replace(value,'<'+name+'>')
    (out/'ovisocr2-large-inputs.json').write_text(text+'\n')


def source_and_exports(out,scratch,report):
    ab.service_source(ab.SOURCE_RUN,ab.SOURCE_HEAD,ab.SOURCE_REPORT_ARTIFACT,conclusion='failure')
    archive=scratch/'source-report.zip';ab.download_artifact(ab.SOURCE_REPORT_ARTIFACT,archive)
    with zipfile.ZipFile(archive) as z:raw=z.read('ovisocr2-model-audit.json')
    if hashlib.sha256(raw).hexdigest()!=ab.SOURCE_REPORT_SHA:raise ValueError('Frozen producer report differs')
    source=json.loads(raw);archive.unlink()
    if source['prior_reference_comparison']['fresh_stage_count']!=943 or source['prior_reference_comparison']['unchanged_prior_members']!=871:
        raise ValueError('Frozen original lineage differs')
    ab.service_source(ab.EXPORT_RUN,ab.EXPORT_HEAD,ab.EXPORT_ARTIFACT,conclusion='success')
    replay.source_exports(ab,scratch,source,report)
    return source


def raw_manifest(folder,reference,log,label):
    if label.endswith('-context'):
        return {'scope':'C API result checks; no Ref.compare dump', 'coverage_complete':True}
    if label.endswith('-delta') or label.endswith('-vision'):
        result=replay.raw_vector_manifest(folder,reference['stages'],log,label,ab)
        result['coverage_complete']=True
        return result
    names,observed=replay.parse_metric_names(log)
    required={f'{phase}.decoder.layer.{layer}' for phase in ('prefill','cached','full_prefix') for layer in range(24)}
    required|={name for name in reference['stages'] if re.fullmatch(r'(prefill|cached)\.cache\.\d+\.(keys|values|conv_states|recurrent_states)',name)}
    required|={f'generated.scores.{i}' for i in range(len(reference['generated_ids'][0]))}
    if not required.issubset(names):raise ValueError('Missing all-layer/cache/generated vocabulary comparison')
    files={p.name.removesuffix('.f32'):p for p in Path(folder).glob('*.f32')}
    if set(files)!=names:raise ValueError('Raw stage set differs from actual comparisons/TRACE')
    rows=[]
    for name,path in files.items():
        shape=reference['stages'].get(name)
        if not shape or path.stat().st_size!=math.prod(shape)*4:raise ValueError('Raw vector original shape/bytes differ')
        rows.append({'stage':name,'shape':shape,'bytes':path.stat().st_size,'sha256':audit.sha(path)})
    return {'coverage_complete':True,'metric_stage_count':len(names),'file_count':len(rows),'files':rows}


def timed_command(label,args,out,report,env=None,timeout=3600):
    resources=out/(label+'-resources.json');started=time.monotonic()
    wrapped=['/usr/bin/time','-f','{"max_rss_kib":%M,"wall_seconds":%e}',
             '-o',str(resources),*map(str,args)]
    result=audit.command(label,wrapped,out,report,env=env,timeout=timeout)
    report['commands'][label]['seconds']=time.monotonic()-started
    if resources.is_file():
        try:
            measured=json.loads(resources.read_text())
            if measured['max_rss_kib']<=0 or measured['wall_seconds']<0:return False
            report['commands'][label]['resources']=measured
        except (ValueError,KeyError):return False
    else:return False
    return result


def native_environment(case):
    env=dict(os.environ)
    env.update(audit.NATIVE_ENV)
    env.update(CRISPEMBED_QWEN35_VISION_PRECISE_NORM='1',
               CRISPEMBED_QWEN35_VISION_PRECISE_ATTN='1',
               CRISPEMBED_QWEN35_VISION_CASCADE_NORM='0',
               CRISPEMBED_OVISOCR2_LARGE_IMAGES='0' if case=='frozen-control' else '1')
    env.pop('CRISPEMBED_DIFF_COS_THRESHOLD',None)
    return env


def run(case):
    panel,closure=preflight()
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/case
    out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    build=Path(os.environ['AUDIT_BIN_DIR'])
    report={'schema':'crispembed-ovisocr2-large-inputs-v1','status':'started','case':case,
            'source_commit':os.environ['GITHUB_SHA'],'source_contract':closure,'cpu':cpu_record(),
            'panel_sha256':audit.sha(PANEL),'commands':{},'limits':[
                'Bounded experimental input extension; default remains1024 and opt-in max2048.',
                'No 2880-squared maximum, multi-image, video, GPU, public benchmark or release claim.',
                'New source must pass frozen-control14 and larger original-module/decoded checks; raw strict flags retained.']}
    def save():save_report(out,report)
    def command(label,args,env=None,timeout=3600):
        result=timed_command(label,args,out,report,env,timeout);save();return result
    save()
    try:
        manifest=json.loads((build/'binaries.json').read_text())
        if manifest['source_commit']!=report['source_commit']:raise ValueError('Build source differs')
        for name,digest in manifest['binary_sha256'].items():
            if audit.sha(build/name)!=digest:raise ValueError('Hosted binary hash differs')
            (build/name).chmod(0o700)
        report['build']=manifest;save()
        if not command('bounds-controls',[build/'test-ovisocr2-image-bounds']):raise ValueError('Downloaded binary controls failed before model transfer')
        if shutil.disk_usage(scratch).free<18*1024**3:raise ValueError('At least18GiB free scratch required before model transfer')
        source=source_and_exports(out,scratch,report);save()
        env=native_environment(case)
        if case=='frozen-control':
            reference=replay.source_reference(ab,scratch,out,report,source)
            image=scratch/'owned-readable.png';ref=scratch/'official-delta.gguf'
            report['original_reference_reused']=True
        else:
            from PIL import Image,ImageDraw
            from huggingface_hub import snapshot_download,hf_hub_download
            import qwen35_ovis_profile as policy
            spec=next(c for c in panel['cases'] if c['id']==case)
            image=scratch/'input.png';rgb=Image.new('RGB',tuple(spec['size']),panel['renderer']['background'])
            ImageDraw.Draw(rgb).text(tuple(panel['renderer']['position']),panel['gold'],
                fill=panel['renderer']['fill'],font_size=panel['renderer']['font_size']);rgb.save(image)
            report['input']={'png_sha256':audit.sha(image),'rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest(),
                             'size':spec['size'],'gold':panel['gold'],'cap':panel['max_new_tokens']};save()
            source_path=Path(snapshot_download(policy.MODEL,revision=policy.REVISION,allow_patterns=list(policy.metadata_files())))
            checkpoint=Path(hf_hub_download(policy.MODEL,'model.safetensors',revision=policy.REVISION))
            profile=policy.load_profile(source_path,policy.MODEL,policy.REVISION);policy.verify_checkpoint(checkpoint,profile)
            # Separate original process releases all framework/model memory before native.
            if not command('original-reference',[sys.executable,ROOT/'tools/dump_ovisocr2_reference.py',
                '--model-dir',source_path,'--image',image,'--output',scratch/'original.npz',
                '--max-new-tokens',str(panel['max_new_tokens']),'--threads','4'],env,timeout=7200):
                raise ValueError('Original full reference failed before native')
            reference=json.loads((scratch/'original.json').read_text());ref=scratch/'official-delta.gguf'
            if reference['missing_stages'] or not reference['generation_result']['reached_eos']:
                raise ValueError('Original incomplete capture or missing EOS')
            import numpy as np
            with np.load(scratch/'original.npz') as arrays:
                if arrays['input.image_grid_thw'].tolist()!=[spec['grid']]:raise ValueError('Original grid differs from frozen geometry')
            report['original_capture']=reference
            report['original_files']={name:{'sha256':audit.sha(scratch/name),'bytes':(scratch/name).stat().st_size}
                for name in ['original.npz','original.json','official-delta.gguf']};save()
        failures=[];report['arms']={}
        for precision,(filename,_,_) in ab.MODELS.items():
            for label,args in audit.native_arms(precision,build,scratch/filename,image,ref).items():
                folder=scratch/'native-diff'/label;folder.mkdir(parents=True,exist_ok=True)
                actual_env=dict(env,CRISPEMBED_DIFF_DUMP_DIR=str(folder))
                passed=command(label,args,actual_env,timeout=7200)
                log=(out/(label+'.log')).read_text()
                try:
                    raw=raw_manifest(folder,reference,log,label)
                except (ValueError,KeyError) as error:
                    raw={'coverage_complete':False,'error':str(error)}
                report['arms'][label]={'passed':passed,'raw':raw}
                # Context is an API comparison and has no Ref.compare raw dumps.
                if label.endswith('-context'):
                    if 'OVIS_CONTEXT' not in log:passed=False
                elif not raw['coverage_complete']:passed=False
                report['arms'][label]['passed']=passed
                if not passed:failures.append(label)
                save()
        if case!='frozen-control':
            report['owning_cases']={}
            from qwen35_quant_calibration import errors
            for precision,(filename,_,_) in ab.MODELS.items():
                label=precision+'-owning-ocr'
                passed=command(label,[build/'test-ovisocr2-quality-run',scratch/filename,image,panel['max_new_tokens'],'2'],env)
                lines=[line.removeprefix('OVIS_QUALITY_RESULT ') for line in (out/(label+'.log')).read_text().splitlines() if line.startswith('OVIS_QUALITY_RESULT ')]
                if len(lines)!=1:
                    failures.append(label);continue
                actual=json.loads(lines[0]);text=bytes.fromhex(actual['raw_text_utf8_hex']).decode()
                actual['raw_text']=text
                actual['original_ids_text_eos_exact']=(actual['token_ids']==reference['generated_ids'][0] and text==reference['decoded'][0] and actual['stop_kind']==1 and actual['stop_id']==reference['generated_stop_id'])
                actual['authored_raw_errors']=errors(panel['gold'],text)
                actual['authored_whitespace_normalized_errors']=errors(' '.join(panel['gold'].split()),' '.join(text.split()))
                report['owning_cases'][precision]=actual
                if not passed or not actual['original_ids_text_eos_exact']:failures.append(label)
                save()
        report['failures']=failures
        report['status']='completed larger-input audit' if not failures else 'rejected larger-input numerical/decoded audit'
        save()
        if failures:raise ValueError('One or more native arms failed; all retained')
    except Exception as error:
        detail=re.sub(r'https?://\S+','<url>',str(error))
        detail=re.sub(r'(?:hf_|gh[pousr]_)[A-Za-z0-9]+','<credential>',detail)
        report['error']=detail
        if report['status']=='started':report['status']='failed before complete observations'
        save();raise


if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'models'))
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['frozen-control','control-square','large-landscape','large-boundary']);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:print(json.dumps(preflight(),sort_keys=True))
    else:run(args.case)
