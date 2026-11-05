#!/usr/bin/env python3
"""Authenticate retained original input, execute native processor, report errors.

No checkpoint transfer, original forward or native model inference occurs.
Existing configured provider credentials are read only inside the hosted job.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[2]
CAPTURE_SHA='170bc06c8b5e929e45d334a5eadc284024f4265b2549f2feddb46607cb49251a'
ORIGINAL_SOURCE='07543dcc80ebcecad644815d9397be2570a51dd4'
PIXELS_SHA='4475c40a09fc35f429c104c6899a1d7a29e124d1327f893411c7d1699c09512b'
IMAGE='tests/regression/images/cc0/simple_form.png'
IMAGE_SHA='3d986ec0d131e4e429127627929e12fd4845313f43849c26bb3d1231c3b038dc'
RGB_SHA='f29f48e94fd623b16bbe203f6c5918cdd5b08c9bc8926a1cc9659e5e67ff0a22'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def main():
    require(os.environ.get('GITHUB_ACTIONS')=='true','hosted-only execution')
    import numpy as np
    from PIL import Image
    out=Path(os.environ['HEAVY_OUT']);out.mkdir(exist_ok=True)
    scratch=Path(tempfile.mkdtemp(prefix='lighton4b-input-'))
    report={'schema':'lighton4b-retained-preprocess/v1','status':'failed',
        'native_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'original_source':ORIGINAL_SOURCE,'original_capture_sha256':CAPTURE_SHA,
        'model_bytes_downloaded':0,'original_forward_executed':False,'native_model_forward_executed':False,
        'decoded_parity_qualified':False}
    def run(label,argv):
        with (out/(label+'.log')).open('w') as log:
            subprocess.run([str(x) for x in argv],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate()
        # Exact saved output is authenticated by its frozen report and member hashes;
        # a later mutable provider output cannot silently become a new reference.
        kernel=os.environ['KAGGLE_ACCOUNT']+'/lightonocr-4b-bf16-cpu-capture-v1'
        api.kernels_output(kernel,str(scratch/'original'),file_pattern=r'^lighton4b-bf16-freegen/(capture\.json|input-(pixel_values|image_grid_thw|input_ids|attention_mask|mm_token_type_ids)\.npy)$',quiet=True)
        retained=scratch/'original/lighton4b-bf16-freegen';capture=retained/'capture.json'
        require(sha(capture)==CAPTURE_SHA,'retained original capture hash differs')
        original=json.loads(capture.read_text());require(original['source_sha']==ORIGINAL_SOURCE and original['status']=='captured','retained source/status differs')
        report['original_processor']={key:original[key]for key in ('image_processor_class','image_processor_backend','image_processor_source_sha256')}
        arrays={}
        for key,row in original['prepared_inputs'].items():
            require(Path(row['file']).name==row['file'],'invalid retained input basename')
            path=retained/row['file'];require(sha(path)==row['sha256'],'retained input hash differs: '+key)
            value=np.load(path,allow_pickle=False);require(list(value.shape)==row['shape'],'retained input shape differs')
            arrays[key]=value
        require(sha(retained/'input-pixel_values.npy')==PIXELS_SHA,'frozen pixel identity differs')
        expected=arrays['pixel_values'];require(expected.shape==(560,1536) and np.isfinite(expected).all(),'original pixel geometry/finite check')
        require(arrays['image_grid_thw'].tolist()==[[1,20,28]],'original grid differs')
        require(sha(ROOT/IMAGE)==IMAGE_SHA,'frozen encoded image differs')
        rgb=np.asarray(Image.open(ROOT/IMAGE).convert('RGB'),dtype=np.uint8)
        require(rgb.shape==(317,452,3) and hashlib.sha256(rgb.tobytes()).hexdigest()==RGB_SHA,'actual decoded image differs')
        rgb.tofile(scratch/'image.rgb')
        run('configure',['cmake','-S',ROOT,'-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        run('build',['cmake','--build',scratch/'build','--target','test-lighton4b-image-compose','--parallel','2'])
        binary=scratch/'build/test-lighton4b-image-compose'
        run('component-controls',[binary])
        run('native-preprocess',[binary,scratch/'image.rgb','317','452',scratch/'native','dump'])
        actual=np.fromfile(scratch/'native.f32',dtype='<f4').reshape(560,1536)
        require((out/'native-preprocess.log').read_text().strip()=='320 448 1 20 28','native grid differs')
        error=np.abs(actual.astype(np.float64)-expected.astype(np.float64));first=np.flatnonzero(error)
        flat_actual=actual.reshape(-1).astype(np.float64);flat_expected=expected.reshape(-1).astype(np.float64)
        report['pixels']={'shape':[560,1536],'elements':int(actual.size),'different_values':int(np.count_nonzero(error)),
            'max_abs':float(error.max()),'mean_abs':float(error.mean()),'rms_error':float(np.sqrt(np.mean(error**2))),
            'native_abs_max':float(np.abs(actual).max()),'original_abs_max':float(np.abs(expected).max()),
            'native_l2':float(np.linalg.norm(flat_actual)),'original_l2':float(np.linalg.norm(flat_expected)),
            'cosine':float(flat_actual@flat_expected/(np.linalg.norm(flat_actual)*np.linalg.norm(flat_expected))),
            'first_differing_flat_index':int(first[0]) if first.size else None,
            'original_npy_sha256':PIXELS_SHA,'native_raw_sha256':sha(scratch/'native.f32')}
        if first.size:
            at=int(first[0]);report['pixels']['first_native']=float(actual.flat[at]);report['pixels']['first_original']=float(expected.flat[at])
        report['input_sha256']={k:v['sha256'] for k,v in original['prepared_inputs'].items()}
        report['source_image']={'file':IMAGE,'sha256':IMAGE_SHA,'rgb_sha256':RGB_SHA}
        # This is a measured processor gate only; downstream decoded quality stays open.
        report['processor_gate_passed']=bool(error.max()<=1.2e-7)
        report['status']='passed' if report['processor_gate_passed'] else 'numerical-difference'
    except Exception as error:
        report['error_type']=type(error).__name__
        if isinstance(error,ModuleNotFoundError):report['missing_module']=error.name
        # Provider exception strings may include private locators or signed URLs.
        report['error']='retained input replay failed; inspect authenticated hosted logs privately'
        raise RuntimeError(report['error']) from None
    finally:
        (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
