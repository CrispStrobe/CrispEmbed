"""Hosted first-FFN attribution of the retained rejected OvisOCR2 Q4 candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'models'))
from dump_ovisocr2_ffn_reference import sha
import ovisocr2_vision_audit as artifacts

RUN=40172527229
HEAD='72698a7640ad10e3d3fb0b090b2d318af85f6da1'
NATIVE_TREE='e5cbd629818f0424e272bc84469e9375eee804d2'
GGML='890278a8342c620197c90e702e1188bcab94f510'
REFERENCE=(12544166139,'ovisocr2-model-diff-reference-40049244654',1303553123,
           '615c0255b6cc9afc757e607e1e154ba2fda7b5d432dd129d4e69032111987fb2')
# No model/reference conversion or quantization occurs in this lane.
RAW_STAGES={
 'prefill.decoder.inputs_embeds': 'f6acfa492d897cbabebcad8f6851d1275be82777497adc18bf6aadcceb4fdd1f',
 'prefill.decoder.layer.0':'ffc114c62273c72d179872530841ecfa394c7b8cc29b40fd5398963cb651f873'}


def pin(row):
    return (row['id'],row['name'],row['bytes'],row['zip_sha256'])


def member(z, name, target, digest, size):
    names=z.namelist()
    if len(names)!=len(set(names)) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):
        raise ValueError('Unsafe/duplicate artifact member')
    if z.getinfo(name).file_size != size: raise ValueError('Member size differs: '+name)
    with z.open(name) as src, Path(target).open('wb') as dst: shutil.copyfileobj(src,dst)
    if sha(target)!=digest: raise ValueError('Member hash differs: '+name)
    return {'bytes':size,'sha256':digest,'member':name}


def require_rows(manifest):
    if (manifest.get('coverage_complete') is not True or manifest.get('probe')!='decoder'
            or manifest.get('phase')!='q4-attention-f16'):
        raise ValueError('Retained candidate raw coverage differs')
    selected={}
    for stage,digest in RAW_STAGES.items():
        rows=[r for r in manifest['files'] if r.get('stage')==stage]
        if len(rows)!=1: raise ValueError('Missing/duplicate retained stage')
        row=rows[0]
        if row['sha256']!=digest or row['original_shape']!=[1,323,1024] or row['bytes']!=1323008:
            raise ValueError('Retained native input/layer identity differs')
        expected='native-diff/q4-attention-f16/decoder/'+stage+'.f32'
        if row['file']!=expected:raise ValueError('Retained native member path differs')
        selected[stage]=row
    return selected


def preflight():
    git=lambda *args:subprocess.check_output(['git',*args],cwd=ROOT).decode().strip()
    if git('rev-parse','HEAD:src')!=NATIVE_TREE or git('rev-parse','HEAD:ggml')!=GGML:
        raise ValueError('Native arithmetic changed; declare a new attribution protocol')
    if git('diff','--name-only','HEAD','--','src','ggml'):
        raise ValueError('Dirty native source')
    receipt=json.loads((ROOT/'tests/results/qwen35-ocr/ovisocr2-q4-attention-f16-quality.json').read_text())
    if receipt['run']!=RUN or receipt['head_sha']!=HEAD or receipt['producer']['quant_numerical']['passed'] is not False:
        raise ValueError('Retained rejected precision receipt differs')
    files=['tests/test_ovisocr2_ffn.cpp','tools/dump_ovisocr2_ffn_reference.py',
           'tools/ci-heavy/ovisocr2_ffn.py','CMakeLists.txt','.github/workflows/ovisocr2-ffn.yml']
    return receipt,{'head':git('rev-parse','HEAD'),'native_tree':NATIVE_TREE,'ggml':GGML,
                    'files':{f:sha(ROOT/f) for f in files}}


def run():
    receipt,source=preflight()
    out=Path(os.environ['HEAVY_OUT']);out.mkdir(parents=True,exist_ok=True)
    scratch=Path(os.environ['HEAVY_SCRATCH'])/'ffn';scratch.mkdir(parents=True,exist_ok=True)
    report={'schema':'crispembed-ovisocr2-ffn-attribution-v1','status':'started','source':source,
            'source_run':RUN,'source_head':HEAD,'acceptance':{'same_weight_max_abs':.002,'same_weight_cos_min':.99999,
                'dequantization':'bit-exact all values','observer_neutrality':'bit-exact'},
            'limits':['Prefill first-layer original input only, no full model recapture or new quantization.',
                      'All four candidate quality rejections and published F16 remain unchanged.',
                      'No decoded-quality qualification, publication, release or default flip.',
                      'Production dequantizes weight storage to F32; activations/matmul use F32, not quantized vector dots.']}
    def save(): (out/'ovisocr2-ffn.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        if shutil.disk_usage(scratch).free<18*1024**3:raise ValueError('At least18GiB hosted scratch required')
        from dump_ovisocr2_ffn_reference import require_environment
        report['environment']=require_environment()
        save()
        report['artifacts']={}
        report_pin=(receipt['artifact_id'],receipt['artifact_name'],receipt['artifact_bytes'],receipt['artifact_zip_sha256'])
        for key,p in [('reports',report_pin),('candidate',pin(receipt['candidate_artifact'])),('raw',pin(receipt['raw_diff_artifact']))]:
            report['artifacts'][key]=artifacts.service_source(RUN,HEAD,p,conclusion='failure')
            artifacts.download_artifact(p,scratch/(key+'.zip'))
        producer=receipt['producer']
        with zipfile.ZipFile(scratch/'reports.zip') as z:
            name=producer['quant_numerical']['raw_diff_manifest']
            raw=z.read(name)
            if hashlib.sha256(raw).hexdigest()!=producer['quant_numerical']['raw_diff_manifest_sha256']:raise ValueError('Frozen raw manifest changed')
            manifest=json.loads(raw)
        rows=require_rows(manifest)
        report['native_payloads']={}
        with zipfile.ZipFile(scratch/'raw.zip') as z:
            for stage,row in rows.items():
                # Artifact uploader strips the native-diff prefix.
                name=row['file'].removeprefix('native-diff/')
                report['native_payloads'][stage]=member(z,name,scratch/(stage+'.f32'),row['sha256'],row['bytes'])
        with zipfile.ZipFile(scratch/'candidate.zip') as z:
            model=scratch/'candidate.gguf';rec=producer['models']['q4-attention-f16']
            report['candidate']=member(z,'ovisocr2-q4-attention-f16.gguf',model,rec['sha256'],rec['bytes'])
        # Only explicit historical metadata is accepted, never a "latest" artifact.
        ref_pin=REFERENCE
        report['artifacts']['reference']=artifacts.service_source(40049244654,'a2a17735d05f172c3ec7ce9675ad51a6adc313f6',ref_pin,conclusion='success')
        artifacts.download_artifact(ref_pin,scratch/'reference.zip')
        with zipfile.ZipFile(scratch/'reference.zip') as z:
            original=scratch/'official-delta.gguf'
            report['original']=member(z,'official-delta.gguf',original,
                'e8e20c654645092ef11e8632db6d84393761df3f8b1d34bc48ae1ddac668e588',1454224192)
        save()
        import qwen35_ovis_profile as profile
        from huggingface_hub import snapshot_download
        model_dir=Path(snapshot_download(profile.MODEL,revision=profile.REVISION,
                    allow_patterns=list(profile.metadata_files())+['model.safetensors']))
        profile.load_profile(model_dir,profile.MODEL,profile.REVISION)
        if sha(model_dir/'model.safetensors')!=profile.CHECKPOINT_SHA256:raise ValueError('Original checkpoint hash differs')
        report['original_checkpoint_sha256']=profile.CHECKPOINT_SHA256
        from dump_ovisocr2_ffn_reference import build
        reference=scratch/'ffn.gguf'
        try:
            report['python']=build(original,model,model_dir,scratch/'prefill.decoder.inputs_embeds.f32',
                scratch/'prefill.decoder.layer.0.f32',reference,producer['tensor_records']['before'])
        finally:
            if reference.with_suffix('.json').exists():shutil.copy2(reference.with_suffix('.json'),out/'python-reference.json')
        save()
        rawdir=scratch/'native';rawdir.mkdir(exist_ok=True)
        env={k:v for k,v in os.environ.items() if not k.startswith('CRISPEMBED_')}
        gates={'CRISPEMBED_QWEN35_OCR':'1','CRISPEMBED_QWEN35_QUANT':'1',
               'CRISPEMBED_QWEN35_WEIGHT_CACHE':'0','CRISPEMBED_QWEN35_PRECISE_RMS':'0',
               'CRISPEMBED_QWEN35_PARALLEL_F16':'0','CRISPEMBED_DIFF_DUMP_DIR':str(rawdir)}
        env.update(gates);report['native_gates']={k:v for k,v in gates.items() if k!='CRISPEMBED_DIFF_DUMP_DIR'}
        executable=Path(os.environ['AUDIT_BIN_DIR'])/'test-ovisocr2-ffn'
        with (out/'native.log').open('w') as log:
            result=subprocess.run([str(executable),str(model),str(reference)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        report['native_exit_code']=result.returncode;report['native_log_sha256']=sha(out/'native.log')
        # Independently recompute every dumped native comparison, retaining all raw arrays remotely.
        import gguf
        import numpy as np
        from dump_ovisocr2_ffn_reference import metrics
        expected={t.name:t.data for t in gguf.GGUFReader(str(reference)).tensors}
        report['native_metrics']={}
        for path in sorted(rawdir.rglob('*.f32')):
            name=path.stem;a=np.fromfile(path,dtype='<f4').reshape(expected[name].shape)
            report['native_metrics'][path.relative_to(rawdir).as_posix()]={**metrics(a,expected[name]),'sha256':sha(path),'bytes':path.stat().st_size}
        required={f'source_input/quant.{s}.f32' for s in ('input','gate','up','product','down')}|{
            'source_gate_up/quant.product.f32','source_gate_up/quant.down.f32','source_silu/quant.silu.f32','source_product/quant.down.f32','natural/native.layer0.f32','source_input/quant.layer0.f32'}|{
            f'weight.{name}.f32' for name in ('gate','up','down')}
        if not required.issubset(report['native_metrics']):raise ValueError('Missing native stage coverage')
        report['same_weight_passed']=(all(report['native_metrics'][n]['passed'] for n in required)
            and report['native_metrics']['source_input/quant.input.f32']['bit_exact']
            and all(report['native_metrics'][f'weight.{name}.f32']['bit_exact'] for name in ('gate','up','down')))
        report['old_decoded_quality']={
            'retained_receipt':'tests/results/qwen35-ocr/ovisocr2-q4-attention-f16-quality.json',
            'receipt_sha256':sha(ROOT/'tests/results/qwen35-ocr/ovisocr2-q4-attention-f16-quality.json'),
            'candidate_qualified':False,'all_eleven_eos':True,
            'regressions':['historical-receipt content CER/WER','two-column raw/content CER/WER']}
        if result.returncode or not report['same_weight_passed']:raise ValueError('Same-weight native FFN audit failed; original rejection stays intact')
        report['status']='completed same-weight first-FFN attribution'
    except Exception as error:
        report['status']='failed first-FFN attribution';report['error']=str(error);save();raise
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:print(json.dumps(preflight()[1],indent=2))
    else:run()
