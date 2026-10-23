"""Hosted experimental Ovis full-payload conversion/roundtrip; no forward pass."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def policy():
    spec=importlib.util.spec_from_file_location('ovis_export_policy',ROOT/'models/qwen35_ovis_profile.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def preflight():
    p=policy();p._verified_evidence()
    run=json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/actions/runs/39983101540']))
    if (run['status']!='completed' or run['conclusion']!='success'
            or run['head_sha']!='ca3d096fe673f3554e95ef8e5f534b597564233e'
            or run['path']!='.github/workflows/ovisocr2-metadata.yml'):
        raise ValueError('Original metadata service run not qualified')
    return p


def validate_manifest(manifest,dtype,schema):
    """Check the requested export policy independently of exporter records."""
    expected=set(schema)-{'lm_head.weight'}
    records=manifest['records']
    if (dtype not in ('f32','f16') or manifest['dtype']!=dtype or manifest['keep_f32']!=[]
            or manifest['tie_word_embeddings'] is not True
            or manifest['checkpoint_tensor_count']!=473 or manifest['export_tensor_count']!=473
            or len(records)!=473 or len(expected)!=473
            or {r['source'] for r in records}!=expected
            or len({r['name'] for r in records})!=473):
        raise ValueError('Export manifest coverage or requested precision differs')
    for record in records:
        original=record['source']
        if original.startswith('model.language_model.'):
            name='l.'+original[len('model.language_model.'):]
        elif original.startswith('model.visual.'):
            name='v.'+original[len('model.visual.'):]
        else:
            raise ValueError('Unexpected original tensor prefix')
        shape=list(schema[original])
        if name=='v.patch_embed.proj.weight':
            import math
            shape=[shape[0],math.prod(shape[1:])]
        expected_dtype='float16' if dtype=='f16' and len(shape)==2 and 'norm' not in name else 'float32'
        if (record['name']!=name or record['source_shape']!=schema[original]
                or record['shape']!=shape or record['dtype']!=expected_dtype):
            raise ValueError('Export record differs from original schema/precision policy: '+original)


def roundtrip(model,manifest,source,dtype):
    schema=policy()._verified_evidence()['schema']['official_state_dict']
    validate_manifest(manifest,dtype,schema)
    import numpy as np
    from gguf import GGUFReader,GGMLQuantizationType
    from safetensors import safe_open
    reader=GGUFReader(model);tensors={t.name:t for t in reader.tensors}
    records=manifest['records']
    if len(reader.tensors)!=473 or len(tensors)!=473 or set(tensors)!={r['name'] for r in records}:raise ValueError('Export tensor coverage changed')
    if (reader.fields['qwen35_ocr.format_version'].contents()!=2
            or reader.fields['qwen35_ocr.profile'].contents()!='ovisocr2'
            or reader.fields['qwen35_ocr.stop_token_ids'].contents()!=[248044,248046]
            or reader.fields['qwen35_ocr.primary_eos_id'].contents()!=248046
            or reader.fields['qwen35_ocr.generation_eos_id'].contents()!=248044
            or reader.fields['tokenizer.ggml.eos_token_id'].contents()!=248046
            or reader.fields['qwen35_ocr.native_alias_qualified'].contents()!=False
            or any('qwen35_ocr.'+key in reader.fields for key in ['processor_config','generation_config'])):
        raise ValueError('Experimental profile metadata differs')
    checked=[]
    with safe_open(source/'model.safetensors',framework='pt',device='cpu') as checkpoint:
        if set(checkpoint.keys())!={r['source'] for r in records}:raise ValueError('Checkpoint tensor was dropped')
        for record in records:
            tensor=tensors[record['name']]
            original=checkpoint.get_tensor(record['source']).float().numpy()
            if list(original.shape)!=record['source_shape']:raise ValueError('Original shape changed')
            expected=np.ascontiguousarray(original.reshape(record['shape']),dtype=np.dtype(record['dtype']))
            digest=hashlib.sha256(expected.tobytes()).hexdigest()
            actual=hashlib.sha256(tensor.data.tobytes()).hexdigest()
            kind=GGMLQuantizationType.F32 if record['dtype']=='float32' else GGMLQuantizationType.F16
            if (digest!=record['sha256'] or actual!=digest or tensor.tensor_type!=kind
                    or list(reversed(tensor.shape.tolist()))!=record['shape']):
                raise ValueError('Exact original cast/export payload mismatch: '+record['source'])
            checked.append(dict(name=record['name'],sha256=actual))
    return dict(tensor_count=len(checked),all_original_cast_payloads_exact=True,
                records=checked,model_sha256=sha(model),bytes=Path(model).stat().st_size)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'ovis-export'
    out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    report=dict(source_commit=os.environ['GITHUB_SHA'],driver_sha256=sha(__file__),status='started',exports={},
                scope='Full checkpoint/export bytes and experimental metadata only; no native/vLLM forward, decoded quality, alias or publication')
    def save():(out/'ovisocr2-export.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        p=preflight();report['metadata_run']=39983101540
        if args.preflight:report['status']='preflight passed; no models or frameworks loaded';save();return
        from huggingface_hub import snapshot_download
        metadata=Path(snapshot_download(p.MODEL,revision=p.REVISION,allow_patterns=list(p.metadata_files())))
        profile=p.load_profile(metadata,p.MODEL,p.REVISION)
        report['metadata_qualification']=profile['qualification'];save()
        shutil.copy(metadata/'LICENSE',scratch/'LICENSE')
        (scratch/'README.md').write_text('Experimental OvisOCR2 audit exports. Source ATH-MaaS/OvisOCR2@'+p.REVISION+
            '. Apache-2.0 upstream license included. Format2 is not a supported native alias. No model-forward/decoded-quality qualification or public model release.\n')
        for dtype in ('f32','f16'):
            model=scratch/('ovisocr2-'+dtype+'.gguf')
            command=[sys.executable,str(ROOT/'models/convert-qwen35-ocr-to-gguf.py'),'--model',p.MODEL,'--revision',p.REVISION,
                     '--profile','ovisocr2','--dtype',dtype,'--output',str(model)]
            with (out/(dtype+'-convert.log')).open('w') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
            manifest_path=model.with_suffix('.gguf.manifest.json');manifest=json.loads(manifest_path.read_text())
            shutil.copy(manifest_path,out/manifest_path.name)
            result=roundtrip(model,manifest,metadata,dtype)
            report['exports'][dtype]=dict(manifest_sha256=sha(manifest_path),**result);save()
        if (metadata/'generation_config.json').exists() or (metadata/'processor_config.json').exists():
            raise ValueError('Conversion fabricated source metadata')
        report['checkpoint_sha256']=p.verify_checkpoint(metadata/'model.safetensors',profile)
        report['source_sha256']={n:sha(ROOT/n) for n in ['models/convert-qwen35-ocr-to-gguf.py','models/qwen35_ovis_profile.py']}
        report['status']='completed experimental F32/F16 full-payload and tensor roundtrip; native/decoded qualification pending';save()
    except BaseException as error:
        report['status']='failed export probe';report['error_type']=type(error).__name__;save();raise


if __name__=='__main__':main()
