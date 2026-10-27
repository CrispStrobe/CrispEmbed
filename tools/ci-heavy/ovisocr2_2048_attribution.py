"""Hosted analysis of frozen 2048 vectors; no model forward/conversion/native rerun."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'models'))
from dump_ovisocr2_ffn_reference import sha
import ovisocr2_vision_audit as service

RUN=40177774574
HEAD='ac1bb9cb33f58fb0e7082cc4ba2ddb38793c48cb'
RAW=(12598011436,'ovisocr2-larger-reference-and-raw-large-boundary-40177774574',7863146668,
     '2075144bf45b8cc1827488c089fee4d828e9c3a4d844975a7169bc007eea5020')
NORM_SHA='9f0a8dc560e75f7d8aa86866026d189473788913c27a23e9fe76c29969a87b54'
PRODUCER='tests/results/qwen35-ocr/ovisocr2-larger-input-producers/large-boundary.json'
PHASES=('prefill','cached','full_prefix')


def preflight():
    wrapper=json.loads((ROOT/'tests/results/qwen35-ocr/ovisocr2-larger-inputs.json').read_text())
    case=wrapper['cases']['large-boundary'];p=json.loads((ROOT/PRODUCER).read_text())
    if (wrapper['run']!=RUN or wrapper['source_commit']!=HEAD or p['source_commit']!=HEAD
        or sha(ROOT/PRODUCER)!=case['producer_report_sha256'] or p['case']!='large-boundary'
        or case['passed_arm_count']!=8 or len(case['failed_arms'])!=6):
        raise ValueError('Frozen source/producer rejection differs')
    if p['original_capture']['loaded_source_sha256']['model']!='788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263':
        raise ValueError('Original model blueprint identity differs')
    return p,{'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip(),
        'source_run':RUN,'source_head':HEAD,'producer_sha256':sha(ROOT/PRODUCER),
        'files':{f:sha(ROOT/f) for f in ('tools/ci-heavy/ovisocr2_2048_attribution.py',
        'tools/dump_ovisocr2_ffn_reference.py','.github/workflows/ovisocr2-2048-attribution.yml')}}


def archive_members(z):
    names=z.namelist()
    if (len(names)!=len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names)
        or sum(i.file_size for i in z.infolist())>48*1024**3):
        raise ValueError('Unsafe, duplicate or oversized preserved archive')
    return names


def extract(z,name,path,record):
    if z.getinfo(name).file_size!=record['bytes']:raise ValueError('Preserved member size differs: '+name)
    with z.open(name) as src,Path(path).open('wb') as dst:shutil.copyfileobj(src,dst)
    if sha(path)!=record['sha256']:raise ValueError('Preserved member SHA differs: '+name)


def indexed_rows(p,arm):
    raw=p['arms'][arm]['raw']
    if raw.get('coverage_complete') is not True:raise ValueError('Incomplete retained native coverage')
    result={}
    for row in raw['files']:
        if row.get('variant','primary')!='primary':continue
        name=row.get('original_stage',row['stage'])
        shape=row.get('source_shape',row.get('shape'))
        if name in result or not shape or any(type(n) is not int or n<=0 for n in shape):
            raise ValueError('Duplicate/invalid preserved native row')
        member='native-diff/'+arm+'/'+row['stage']+'.f32'
        if row.get('file',member)!=member or row['bytes']!=4*math.prod(shape):
            raise ValueError('Preserved native path/shape/bytes differ')
        result[name]={**row,'shape':shape,'member':member}
    return result


def vision_stage_order(stages):
    # Captured dictionary insertion is not the DAG: the interpolated position
    # probe was appended after the layer observations. Inspect both input
    # branches before their sum, rather than misattributing it to embedding.
    early=['prefill.vision.'+n for n in ('patch_embed','position','embedded','rotary_cos','rotary_sin')]
    rest=[s for s in stages if s.startswith('prefill.vision.') and s not in early]
    return [s for s in early if s in stages]+rest


def calibrated(actual,reference,control=None,vision=False):
    import numpy as np
    from dump_ovisocr2_ffn_reference import metrics
    row=metrics(actual,reference)
    variation=np.zeros_like(reference,dtype=np.float64) if control is None else abs(control.astype('f8')-reference.astype('f8'))
    if vision and control is not None:variation=np.full_like(variation,float(variation.max()))
    row.update(strict_absolute_pass=row['max_abs']<=.002,calibrated=control is not None,
        self_max_abs=float(variation.max()),max_budget_ratio=float((abs(actual.astype('f8')-reference.astype('f8'))/(.002+2*variation)).max()))
    row['calibrated_passed']=row['cos_min']>=.99999 and row['max_budget_ratio']<=1
    return row


def operator(norm,values):
    import torch
    with torch.inference_mode():return norm(torch.from_numpy(values.copy())).detach().numpy().copy()


def run():
    p,source=preflight()
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'2048-attribution'
    out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    r={'schema':'crispembed-ovisocr2-2048-attribution-v1','status':'started','source':source,
       'scope':'retained native vectors and actual original RMSNorm on saved inputs; no full model/native rerun',
       'limits':['All old eight-pass/six-fail arms and budgets remain unchanged.',
                 'Local operator targets are distinct from immutable full-model references.',
                 'No increased bound, numerical waiver, default, quantization or release change.']}
    def save():(out/'ovisocr2-2048-attribution.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    save()
    try:
        if shutil.disk_usage(scratch).free<20*1024**3:raise ValueError('At least20GiB hosted free scratch required')
        from dump_ovisocr2_ffn_reference import require_environment,metrics
        r['environment']=require_environment();save()
        r['service']=service.service_source(RUN,HEAD,RAW,conclusion='failure')
        archive=scratch/'retained.zip';service.download_artifact(RAW,archive,timeout=3600)
        import gguf
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        import qwen35_ovis_profile as profile
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNorm
        model=Path(snapshot_download(profile.MODEL,revision=profile.REVISION,
                     allow_patterns=list(profile.metadata_files())+['model.safetensors']))
        profile.load_profile(model,profile.MODEL,profile.REVISION)
        if sha(model/'model.safetensors')!=profile.CHECKPOINT_SHA256:raise ValueError('Original checkpoint SHA differs')
        with safe_open(str(model/'model.safetensors'),framework='pt',device='cpu') as sf:
            weight=sf.get_tensor('model.language_model.norm.weight').float().contiguous()
        if hashlib.sha256(weight.numpy().tobytes()).hexdigest()!=NORM_SHA:raise ValueError('Qualified F32 norm weight differs')
        norm=Qwen3_5RMSNorm(1024,eps=1e-6);norm.load_state_dict({'weight':weight});norm.eval()
        torch.set_num_threads(4) # Original larger-input producer uses four threads.
        r['operator']={'class':'original Qwen3_5RMSNorm','norm_weight_sha256':NORM_SHA,
                       'checkpoint_sha256':profile.CHECKPOINT_SHA256,'threads':4,'eps':1e-6}
        with zipfile.ZipFile(archive) as z:
            names=archive_members(z);r['archive_member_count']=len(names)
            # upload-artifact retains the case root directory's relative contents.
            matches=[n for n in names if n.endswith('official-delta.gguf')]
            if len(matches)!=1:raise ValueError('Original reference member is ambiguous')
            base=matches[0].removesuffix('official-delta.gguf')
            refpath=scratch/'official-delta.gguf';extract(z,matches[0],refpath,p['original_files']['official-delta.gguf'])
            r['original_reference']={**p['original_files']['official-delta.gguf'],'member':matches[0]}
            targets={t.name:t for t in gguf.GGUFReader(str(refpath)).tensors}
            def original(name):
                t=targets[name]
                if t.tensor_type!=gguf.GGMLQuantizationType.F32:raise ValueError('Original target not F32')
                return np.asarray(t.data,dtype=np.float32)
            rawpath=scratch/'selected.f32';payloads={}
            def actual(arm,stage):
                row=rows[arm][stage];member=base+row['member']
                extract(z,member,rawpath,row)
                a=np.fromfile(rawpath,dtype='<f4').reshape(row['shape']);rawpath.unlink()
                payloads[arm+'/'+stage]={'sha256':row['sha256'],'bytes':row['bytes'],'shape':row['shape'],'member':member}
                return a
            arms=[precision+'-'+suffix for precision in ('f32','f16') for suffix in
                   ('vision','decoder-prepared','decoder-native-vision','decoder-native-positions','decoder-native-frontend')]
            rows={arm:indexed_rows(p,arm) for arm in arms}
            # Trace all available original stage boundaries in their captured order.
            r['vision_trace']={};r['decoder_trace']={};r['norm_isolation']={};r['composed_manifest_controls']={}
            for precision in ('f32','f16'):
                arm=precision+'-vision';trace=[]
                for stage in vision_stage_order(p['original_capture']['stages']):
                    if stage not in rows[arm]:continue
                    a=actual(arm,stage);control='backend_off.'+stage.removeprefix('prefill.')
                    trace.append({'stage':stage,**calibrated(a,original(stage),original(control) if control in targets else None,vision=True)})
                if len(trace)!=37:raise ValueError('Missing frozen vision boundaries')
                r['vision_trace'][precision]={'stages':trace,'first_non_bit_exact':next((x['stage'] for x in trace if not x['bit_exact']),None),
                    'first_strict_failure':next((x['stage'] for x in trace if not x['passed']),None)}
                for suffix in ('decoder-prepared','decoder-native-vision'):
                    arm=precision+'-'+suffix;trace=[]
                    wanted=['prefill.decoder.inputs_embeds']+[f'prefill.decoder.layer.{i}' for i in range(24)]+['prefill.decoder.norm']
                    for stage in wanted:
                        if stage not in rows[arm]:
                            if suffix=='decoder-prepared' and stage.endswith('inputs_embeds'):continue
                            raise ValueError('Missing retained decoder stage')
                        control='backend_off.'+stage.removeprefix('prefill.') if stage.endswith('.norm') else None
                        trace.append({'stage':stage,**calibrated(actual(arm,stage),original(stage),original(control) if control in targets else None)})
                    r['decoder_trace'][arm]=trace
                    # Existing native-positions/frontend outputs must agree with natural vision by SHA.
                    if suffix=='decoder-native-vision':
                        for other in ('decoder-native-positions','decoder-native-frontend'):
                            otherarm=precision+'-'+other
                            shared={s for s in rows[arm] if s in rows[otherarm]}
                            differences=[s for s in shared if rows[arm][s]['sha256']!=rows[otherarm][s]['sha256']]
                            r['composed_manifest_controls'][otherarm]={'shared_stages':len(shared),'differing_stage_hashes':sorted(differences)}
                    isolations=[]
                    for phase in PHASES:
                        hidden=phase+'.decoder.layer.23';final=phase+'.decoder.norm'
                        h=actual(arm,hidden);n=actual(arm,final)
                        original_h=original(hidden);original_n=original(final)
                        observed=operator(norm,original_h);local=operator(norm,h)
                        control=('backend_off.' if phase=='prefill' else 'backend_off_full_prefix.')+'decoder.norm'
                        ctrl=original(control) if phase!='cached' and control in targets else None
                        local_metrics=metrics(n,local)
                        dest=scratch/'operator';dest.mkdir(exist_ok=True)
                        file=dest/(arm+'-'+phase+'-norm.f32');local.astype('<f4').tofile(file)
                        isolations.append({'phase':phase,'original_anchor':metrics(observed,original_n),
                            'native_input_error':metrics(h,original_h),'same_input_operator_error':local_metrics,
                            'observed_composed_error':calibrated(n,original_n,ctrl),
                            'propagated_input_error':calibrated(local,original_n,ctrl),
                            'operator_output_sha256':sha(file),'operator_output_bytes':file.stat().st_size,
                            'error_decomposition_max_residual':float(abs((n.astype('f8')-original_n.astype('f8'))-
                                ((n.astype('f8')-local.astype('f8'))+(local.astype('f8')-original_n.astype('f8')))).max())})
                    r['norm_isolation'][arm]=isolations;save()
            r['selected_native_payloads']=payloads
            r['decoded_evidence']={'producer':PRODUCER,'cases':p['owning_cases'],
                 'original_generated_ids':p['original_capture']['generated_ids'],
                 'original_stop_id':p['original_capture']['generated_stop_id'],
                 'raw_decoded':p['original_capture']['raw_decoded'],'decoded':p['original_capture']['decoded']}
            r['all_original_norm_anchors_bit_exact']=all(x['original_anchor']['bit_exact'] for values in r['norm_isolation'].values() for x in values)
            r['all_same_input_norm_gates_pass']=all(x['same_input_operator_error']['passed'] for values in r['norm_isolation'].values() for x in values)
            if not r['all_original_norm_anchors_bit_exact']:raise ValueError('Original RMSNorm anchor drift; preserve distinct local targets')
            if not r['all_same_input_norm_gates_pass']:raise ValueError('Same-input original/native norm mismatch requires further attribution')
            r['status']='completed retained-vector 2048 attribution'
    except Exception as e:
        r['status']='failed retained-vector 2048 attribution';r['error']=str(e);save();raise
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:print(json.dumps(preflight()[1],indent=2))
    else:run()
