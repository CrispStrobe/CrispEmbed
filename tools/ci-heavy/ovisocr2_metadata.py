"""Hosted weight-free OvisOCR2 metadata/schema probe; no model forward or alias."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import typing
import types
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
MODEL='ATH-MaaS/OvisOCR2'
REVISION='cbdede82e048659b83310476ff807da825f8fe78'
VLLM='0decac0d96c42b49572498019f0a0e3600f50398'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_schema(expected, actual, tied):
    missing=set(expected)-set(actual)
    if tied and missing=={'lm_head.weight'}:missing.clear()
    extra=set(actual)-set(expected)
    wrong=[n for n in actual if n in expected and actual[n]['shape']!=expected[n]]
    if missing or extra or wrong:
        raise ValueError('Official/checkpoint schema mismatch: '+str((sorted(missing),sorted(extra),wrong)))
    return dict(expected_tensors=len(expected),checkpoint_tensors=len(actual),tied_head_absent='lm_head.weight' not in actual)


def literal_text(node):
    if isinstance(node,ast.Constant) and isinstance(node.value,str):return node.value
    if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return literal_text(node.left)+literal_text(node.right)
    raise ValueError('Original prompt is not a literal string expression')


def original_prompt(readme):
    blocks=re.findall(r'```python\s*\n(.*?)```',readme,re.S)
    candidates=[]
    for block in blocks:
        for node in ast.walk(ast.parse(block)):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompt' for t in node.targets):
                candidates.append(literal_text(node.value))
    if len(candidates)!=1:raise ValueError('Ambiguous original example prompt')
    return candidates[0]


def stop_probe(source, generation, tokenizer_eos):
    tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SamplingParams')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='update_from_generation_config')
    env={'Any':typing.Any};exec(compile(ast.Module(body=[method],type_ignores=[]),'pinned-vllm-stop-method','exec'),env)
    obj=types.SimpleNamespace(ignore_eos=False,_eos_token_id=None,_all_stop_token_ids=set(),stop_token_ids=[])
    env['update_from_generation_config'](obj,generation,tokenizer_eos)
    return dict(primary_eos=obj._eos_token_id,additional_stop_ids=sorted(obj.stop_token_ids),all_stop_ids=sorted(obj._all_stop_token_ids))


def main():
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);out.mkdir(parents=True,exist_ok=True)
    report=dict(source_commit=os.environ['GITHUB_SHA'],driver_sha256=sha(__file__),status='started',
        scope='Weight-free original HF metadata/processor and meta-schema; isolated pinned vLLM stop method, no installed vLLM/GPU/model forward or decoded-quality claim')
    def save():(out/'ovisocr2-metadata.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        import numpy as np
        import torch,torchvision,transformers
        from PIL import Image
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig,AutoProcessor,GenerationConfig,Qwen3_5ForConditionalGeneration
        if (transformers.__version__!='5.16.1' or torch.__version__.split('+')[0]!='2.14.0'
                or torchvision.__version__.split('+')[0]!='0.29.0'):
            raise ValueError('Pinned metadata environment required')
        torch.set_num_threads(2)
        audit=json.loads((ROOT/'tests/results/qwen35-ocr/ovisocr2-source-contract.json').read_text())
        inspected=audit['models'][MODEL]
        if inspected['revision']!=REVISION:raise ValueError('Checkpoint source changed')
        names=['LICENSE','README.md','config.json','preprocessor_config.json','tokenizer_config.json',
               'chat_template.jinja','tokenizer.json','merges.txt','vocab.json','video_preprocessor_config.json']
        directory=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=names))
        if any(directory.glob('*.safetensors')):raise ValueError('Unexpected weight payload')
        for name,digest in inspected['downloaded_file_sha256'].items():
            if name=='safetensors-header.json':continue
            if sha(directory/name)!=digest:raise ValueError('Audited source bytes changed: '+name)
        if (directory/'generation_config.json').exists() or (directory/'processor_config.json').exists():
            raise ValueError('Missing-source contract changed')
        report['checkpoint_files_sha256']={n:sha(directory/n) for n in names if (directory/n).is_file()}
        config=AutoConfig.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
        generation=GenerationConfig.from_model_config(config)
        report['resolved_generation']=generation.to_diff_dict()
        processor=AutoProcessor.from_pretrained(directory,local_files_only=True,trust_remote_code=False)
        report['processor']=dict(class_name=type(processor).__module__+'.'+type(processor).__name__,
            image_class=type(processor.image_processor).__module__+'.'+type(processor.image_processor).__name__,
            image_backend=getattr(processor.image_processor,'backend',None),image_config=processor.image_processor.to_dict())
        report['loaded_source_sha256']={}
        for label,cls in [('generation',GenerationConfig),('model',Qwen3_5ForConditionalGeneration),
                          ('processor',type(processor)),('image_processor',type(processor.image_processor)),('tokenizer',type(processor.tokenizer))]:
            file=Path(inspect.getfile(cls));report['loaded_source_sha256'][label]=sha(file)
            if label in ('generation','model'):
                expected='427f7e57922fb58c3b321f5e5837ee782bd640cd5b22833029b16f9d4b9c4cdb' if label=='generation' else '788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263'
                if hashlib.sha256(file.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:
                    raise ValueError('Loaded original HF implementation changed')
        report['tokenizer_eos']=processor.tokenizer.eos_token_id;save()
        manifest=json.loads((ROOT/'tests/results/qwen35-ocr/ovisocr2-vllm-blueprint.json').read_text())
        pin=next(x for x in manifest['vllm_sources'] if x['path']=='vllm/sampling_params.py')
        payload=urllib.request.urlopen('https://raw.githubusercontent.com/vllm-project/vllm/'+VLLM+'/vllm/sampling_params.py',timeout=30).read()
        if hashlib.sha256(payload).hexdigest()!=pin['sha256']:raise ValueError('vLLM method source changed')
        report['effective_stops_source_probe']=stop_probe(payload.decode(),report['resolved_generation'],processor.tokenizer.eos_token_id)
        if (report['effective_stops_source_probe']['all_stop_ids']!=[248044,248046]
                or report['resolved_generation']['eos_token_id']!=248044 or processor.tokenizer.eos_token_id!=248046):
            raise ValueError('Derived stop contract differs; preserve before profile changes')
        save()
        config._attn_implementation='eager'
        with torch.device('meta'):model=Qwen3_5ForConditionalGeneration(config)
        expected={n:list(t.shape) for n,t in model.state_dict().items()}
        if not all(p.device.type=='meta' for p in model.parameters()):raise ValueError('Schema model allocated payloads')
        report['schema']=check_schema(expected,audit['weight_header']['tensors'],bool(config.tie_word_embeddings))
        report['schema']['official_state_dict']=expected;del model
        prompt=original_prompt((directory/'README.md').read_text())
        messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=prompt)])]
        text=processor.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        (out/'rendered-prompt.txt').write_text(text);report['rendered_prompt_sha256']=sha(out/'rendered-prompt.txt');report['inputs']=[];save()
        for label,width,height in [('square',448,448),('odd',449,417)]:
            y,x=np.indices((height,width));pixels=np.stack([(x*3+y)%256,(y*5+x)%256,(x+y*7)%256],axis=-1).astype(np.uint8)
            image=Image.fromarray(pixels)
            inputs=processor(text=[text],images=[image],return_tensors='pt',images_kwargs={'min_pixels':448*448,'max_pixels':2880*2880})
            repeated=processor(text=[text],images=[image],return_tensors='pt',images_kwargs={'min_pixels':448*448,'max_pixels':2880*2880})
            if set(inputs)!=set(repeated) or not all(torch.equal(v,repeated[k]) for k,v in inputs.items()):
                raise ValueError('Original processor repeat changed')
            arrays={k:v.detach().cpu().numpy() for k,v in inputs.items()}
            if not np.isfinite(arrays['pixel_values']).all():raise ValueError('Nonfinite processor output')
            np.savez_compressed(out/(label+'-processor.npz'),**arrays)
            report['inputs'].append(dict(case=label,rgb_shape=list(pixels.shape),rgb_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                fields={k:dict(shape=list(v.shape),dtype=str(v.dtype),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()},
                grid=arrays['image_grid_thw'].tolist(),ids=arrays['input_ids'].tolist(),attention_mask=arrays['attention_mask'].tolist(),
                archive_sha256=sha(out/(label+'-processor.npz')),repeat_exact=True));save()
        report['status']='completed weight-free HF metadata/schema/processor probe; actual vLLM/inference remains unqualified';save()
    except BaseException as error:
        report['status']='failed metadata probe';report['error_type']=type(error).__name__;save();raise


if __name__=='__main__':main()
