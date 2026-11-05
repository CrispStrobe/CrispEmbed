#!/usr/bin/env python3
"""Hosted exact original BF16 prompt/decode controls; metadata only, no weights."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
if os.environ.get('GITHUB_ACTIONS')!='true':raise SystemExit('hosted controls required')
out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);build=scratch/'tokenizer-build'
r={'schema':'lighton4b-tokenizer-controls/v1','status':'failed','source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'model_weights_downloaded':False,'model_forward_executed':False,'native_model_qualified':False}
def run(label,args):
    with(out/(label+'.log')).open('w')as stream:subprocess.run([str(x)for x in args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
try:
    run('dependencies',[sys.executable,'-m','pip','install','numpy==2.3.4','gguf==0.17.1','huggingface_hub==0.36.0'])
    import numpy as np
    import gguf
    from huggingface_hub import snapshot_download
    spec=importlib.util.spec_from_file_location('converter',ROOT/'models/convert-qwen35-ocr-to-gguf.py');c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    model=Path(snapshot_download(c.LIGHTON4B_MODEL,revision=c.LIGHTON4B_REVISION,allow_patterns=list(c.lighton4b_metadata_files()),cache_dir=scratch/'hf',token=False))
    pinned,config=c.load_lighton4b_export_source(model,c.LIGHTON4B_MODEL,c.LIGHTON4B_REVISION)
    tokenizer=c.read_tokenizer_metadata(model,248320)
    metadata=out/'tokenizer-metadata.gguf';writer=gguf.GGUFWriter(str(metadata),'qwen35_ocr')
    writer.add_uint32('qwen35_ocr.format_version',5)
    for key,value in {'general.source.huggingface.repository':c.LIGHTON4B_MODEL,'general.source.huggingface.revision':c.LIGHTON4B_REVISION,'qwen35_ocr.profile':c.LIGHTON4B_PROFILE,'qwen35_ocr.checkpoint_sha256':c._lighton4b_profile().CHECKPOINT_SHA256,'qwen35_ocr.tokenizer_sha256':c._lighton4b_profile().TOKENIZER_SHA256}.items():writer.add_string(key,value)
    c.write_tokenizer_metadata(writer,tokenizer);writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
    reference=json.loads((ROOT/'docs/lighton4b-tokenizer-reference-v1.json').read_text())
    (out/'generated.ids').write_text('\n'.join(map(str,reference['generated_ids']))+'\n')
    run('configure',['cmake','-S',ROOT,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF'])
    run('build',['cmake','--build',build,'--target','crispembed-test-lighton4b-tokenizer','--parallel','2'])
    run('native',[build/'crispembed-test-lighton4b-tokenizer',metadata,out/'generated.ids',out])
    for field,file in [('input_ids','prompt.ids'),('mm_token_type_ids','modalities.ids')]:
        values=[int(x)for x in (out/file).read_text().split()];array=np.asarray([values],dtype=np.int64);path=out/(field+'.npy');np.save(path,array,allow_pickle=False)
        expected=reference['prepared_inputs'][field]
        if list(array.shape)!=expected['shape']or digest(path)!=expected['sha256']:raise ValueError('actual original BF16 '+field+' differs')
    if (out/'decoded.txt').read_text()!=reference['decoded']:raise ValueError('all177 original generated IDs decoded text differs')
    protocol=json.loads((ROOT/'docs/lighton4b-bf16-freegen-v1.json').read_text())
    if (out/'rendered.txt').read_text()!=protocol['case']['rendered_prompt']:raise ValueError('frozen original rendered prompt differs')
    r.update(status='passed',original_prompt_ids_exact=True,original_modalities_exact=True,original_decoded_text_exact=True,original_generated_ids=177,original_prompt_ids=154,negative_controls=7,tokenizer_payload_sha256=digest(model/'tokenizer.json'))
except Exception as error:
    r.update(error_type=type(error).__name__,error=str(error).replace(str(ROOT),'[repository]').replace(str(scratch),'[scratch]').replace(str(out),'[report]'));raise
finally:
    r['source_sha256']={name:digest(ROOT/name)for name in ('src/lighton4b_frontend.h','src/lighton4b_tokenizer.cpp','src/qwen35_tokenizer.cpp','tests/lighton4b_tokenizer.cpp','tools/ci-heavy/lighton4b_tokenizer_controls.py','docs/lighton4b-tokenizer-reference-v1.json')}
    (out/'tokenizer-controls.json').write_text(json.dumps(r,indent=2)+'\n')
