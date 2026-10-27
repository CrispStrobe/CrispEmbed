"""Hosted original-input interpolation contract; no full-model/reference recapture."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import ovisocr2_2048_attribution as frozen

ROOT=frozen.ROOT
sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'models'))
from dump_ovisocr2_ffn_reference import sha
from dump_ovisocr2_position_reference import UTIL_SHA


def preflight():
    p,source=frozen.preflight()
    source['probe_files']={f:sha(ROOT/f) for f in ('src/qwen35_vision.cpp','tests/test_qwen35_position.cpp',
            'tools/dump_ovisocr2_position_reference.py','tools/ci-heavy/ovisocr2_position_probe.py',
            '.github/workflows/ovisocr2-position-probe.yml','CMakeLists.txt')}
    source['vision_utils_sha256']=UTIL_SHA
    return p,source


def run():
    p,source=preflight();out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'position-probe'
    out.mkdir(parents=True,exist_ok=True);scratch.mkdir(parents=True,exist_ok=True)
    r={'schema':'crispembed-ovisocr2-position-probe-v1','status':'started','source':source,
       'source_run':frozen.RUN,'source_head':frozen.HEAD,'acceptance':{'source_coordinate_and_weight_bits':'exact',
           'legacy_position_and_patch_bits':'exact retained ordinary output','position_max_abs':.002,'position_cos_min':.99999},
       'limits':['Local position/patch operators only; no2048/default/decoded-quality promotion.',
                 'Original full-model arrays are immutable and stay separate from additional local targets.']}
    def save():(out/'ovisocr2-position-probe.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    save()
    try:
        if shutil.disk_usage(scratch).free<20*1024**3:raise ValueError('At least20GiB hosted scratch required')
        from dump_ovisocr2_ffn_reference import require_environment,metrics
        r['environment']=require_environment()
        r['service']=frozen.service.service_source(frozen.RUN,frozen.HEAD,frozen.RAW,conclusion='failure')
        archive=scratch/'retained.zip';frozen.service.download_artifact(frozen.RAW,archive,timeout=3600)
        import gguf
        import numpy as np
        from huggingface_hub import snapshot_download
        import qwen35_ovis_profile as policy
        with zipfile.ZipFile(archive) as z:
            names=frozen.archive_members(z);matches=[n for n in names if n.endswith('official-delta.gguf')]
            if len(matches)!=1:raise ValueError('Ambiguous original reference')
            prefix=matches[0].removesuffix('official-delta.gguf');ref=scratch/'official-delta.gguf'
            frozen.extract(z,matches[0],ref,p['original_files']['official-delta.gguf'])
            tensors={t.name:t for t in gguf.GGUFReader(str(ref)).tensors}
            def array(name):
                t=tensors[name]
                if t.tensor_type!=gguf.GGMLQuantizationType.F32:raise ValueError('Original operator input not F32')
                return np.array(t.data,dtype=np.float32,copy=True)
            pixels=array('input.pixel_values');position=array('prefill.vision.position');patch=array('prefill.vision.patch_embed')
            grid=array('input.image_grid_thw')
            if list(grid.reshape(-1))!=[1,32,64]:raise ValueError('Original boundary grid differs')
            rows=frozen.indexed_rows(p,'f32-vision');legacy={}
            for name in ('position','patch_embed'):
                row=rows['prefill.vision.'+name];file=scratch/(name+'.f32')
                frozen.extract(z,prefix+row['member'],file,row)
                legacy[name]=np.fromfile(file,dtype='<f4').reshape(row['shape'])
        del tensors;ref.unlink();archive.unlink() # Known task-owned remote transients only.
        directory=Path(snapshot_download(policy.MODEL,revision=policy.REVISION,
                          allow_patterns=list(policy.metadata_files())+['model.safetensors']))
        policy.load_profile(directory,policy.MODEL,policy.REVISION)
        if sha(directory/'model.safetensors')!=policy.CHECKPOINT_SHA256:raise ValueError('Source checkpoint hash differs')
        from dump_ovisocr2_position_reference import build
        target=scratch/'position-reference.gguf'
        try:
            r['python']=build(directory,position,legacy['position'],pixels,patch,legacy['patch_embed'],
                           p['original_capture']['source_config']['vision_config'],target)
        finally:
            if target.with_suffix('.json').exists():
                r['python']=json.loads(target.with_suffix('.json').read_text())
                shutil.copy2(target.with_suffix('.json'),out/'python-position-reference.json')
        save()
        env={k:v for k,v in os.environ.items() if not k.startswith('CRISPEMBED_')}
        gates={n:'1' for n in ('CRISPEMBED_QWEN35_OCR','CRISPEMBED_OVISOCR2_FRONTEND',
                               'CRISPEMBED_OVISOCR2_INFERENCE','CRISPEMBED_OVISOCR2_LARGE_IMAGES')}
        gates.update(CRISPEMBED_QWEN35_WEIGHT_CACHE='0',CRISPEMBED_QWEN35_PARALLEL_F16='0',CRISPEMBED_QWEN35_PRECISE_RMS='0')
        env.update(gates);r['native_gates']=gates
        raw=scratch/'native';raw.mkdir(exist_ok=True);env['CRISPEMBED_DIFF_DUMP_DIR']=str(raw)
        binary=Path(os.environ['AUDIT_BIN_DIR'])/'test-qwen35-position';r['binary_sha256']=sha(binary)
        with (out/'native.log').open('w') as log:
            result=subprocess.run([str(binary),str(target)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200)
        r['native_exit_code']=result.returncode;r['native_log_sha256']=sha(out/'native.log')
        expected={t.name:t.data for t in gguf.GGUFReader(str(target)).tensors}
        r['native_metrics']={}
        for file in sorted(raw.glob('*.f32')):
            name=file.stem;reference=expected[name];a=np.fromfile(file,dtype='<f4').reshape(reference.shape)
            r['native_metrics'][name]={**metrics(a,reference),'sha256':sha(file),'bytes':file.stat().st_size}
        wanted={f'case.{i}.{name}' for i in range(5) for name in ('indices','weights','row_src','col_src','position')}|{'frozen.patch','local.patch'}
        if set(r['native_metrics'])!=wanted:raise ValueError('Incomplete operator dump coverage')
        r['all_source_bits_exact']=all(v['bit_exact'] for k,v in r['native_metrics'].items() if k.endswith(('indices','weights','row_src','col_src')))
        if result.returncode or not r['all_source_bits_exact'] or not all(v['passed'] for v in r['native_metrics'].values()):
            raise ValueError('Native interpolation contract failed; all diagnostics retained')
        r['status']='completed original-input position contract'
    except Exception as e:r.update(status='failed original-input position contract',error=str(e));save();raise
    save()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.preflight:print(json.dumps(preflight()[1],indent=2))
    else:run()
