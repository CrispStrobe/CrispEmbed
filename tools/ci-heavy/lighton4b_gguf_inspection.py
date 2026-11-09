#!/usr/bin/env python3
"""Hosted, model-free atomic model-owner checks against the original header."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execute():
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':
        raise RuntimeError('hosted portable execution required')
    out=Path(os.environ['HEAVY_OUT']); scratch=Path(os.environ['HEAVY_SCRATCH'])/'lighton4b-gguf-inspection'
    scratch.mkdir(parents=True,exist_ok=False)
    report={'schema':'lighton4b-gguf-inspection/v1','status':'failed','model_payload_bytes':0,
            'native_inference_observed':False,'format5_admitted':False,'quality_qualified':False,
            'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_sha256':{p:sha(ROOT/p) for p in ('tools/ci-heavy/lighton4b_published_header.py', 'src/lighton4b_gguf_inspection.h', 'src/lighton4b_model_owner.h', 'src/qwen35_lighton4b_text.h', 'src/lighton4b_vision_owner.cpp', 'src/qwen35_vision.h',
                'tests/test_lighton4b_gguf_inspection.cpp','tools/ci-heavy/lighton4b_gguf_inspection.py','CMakeLists.txt',
                'tests/test_lighton4b_gdn_reference.py','tests/test_ovisocr2_native_replay.py',
                'tests/test_ovisocr2_large_inputs.py',
                'tools/ci-heavy/ovisocr2_native_replay.py','tools/ci-heavy/lighton4b_gdn_reference.py')}}
    def command(label,argv):
        with (out/(label+'.log')).open('w') as stream:
            subprocess.run([str(x) for x in argv],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        command('gdn-protocol-controls',[sys.executable,ROOT/'tests/test_lighton4b_gdn_reference.py','-v'])
        command('ovis-frozen-closure-controls',[sys.executable,ROOT/'tests/test_ovisocr2_native_replay.py','-v'])
        command('ovis-frozen-large-input-controls',[sys.executable,ROOT/'tests/test_ovisocr2_large_inputs.py','-v'])
        sys.path.insert(0,str(ROOT/'models'))
        spec=importlib.util.spec_from_file_location('original_exporter',ROOT/'models/convert-qwen35-ocr-to-gguf.py')
        converter=importlib.util.module_from_spec(spec);spec.loader.exec_module(converter)
        policy=converter._lighton4b_profile();policy.load_profile()
        header_path=policy.FIXTURE/'model.safetensors.header.json'
        header=json.loads(header_path.read_text())
        rows=[]
        for source,tensor in sorted(header.items()):
            if source=='__metadata__':continue
            name=converter.tensor_name(source)
            shape=tensor['shape']
            if name=='v.patch_embed.proj.weight':shape=[1024,1536]
            ne=list(reversed(shape))+[1]*(4-len(shape))
            matrix=len(shape)==2 and 'norm' not in name
            rows.append(' '.join([name]+[str(n) for n in ne]+[str(int(matrix))]))
        if len(rows)!=723:raise ValueError('original model inventory must contain exactly723 tensors')
        fixture=out/'original-model-schema.tsv';fixture.write_text('\n'.join(rows)+'\n')
        report['original']={'model':policy.MODEL,'revision':policy.REVISION,
            'header_sha256':sha(header_path),'schema_sha256':sha(fixture),'text_tensors':426,'vision_tensors':297,'total_tensors':723}
        config=json.loads((policy.FIXTURE/'config.json').read_text())
        t=config['text_config'];v=config['vision_config'];p='qwen35_ocr.'
        metadata=[('s','general.architecture','qwen35_ocr'),
                  ('s','general.source.huggingface.repository',policy.MODEL),
                  ('s','general.source.huggingface.revision',policy.REVISION),
                  ('s',p+'profile','lightonocr3-4b'),('u',p+'format_version',5),
                  ('b',p+'tie_word_embeddings',int(t['tie_word_embeddings'])),
                  ('b',p+'zero_centered_text_norm',1)]
        text_fields={'embedding_length':'hidden_size','feed_forward_length':'intermediate_size',
            'block_count':'num_hidden_layers','vocab_size':'vocab_size','attention.head_count':'num_attention_heads',
            'attention.head_count_kv':'num_key_value_heads','attention.key_length':'head_dim','attention.value_length':'head_dim',
            'attention.full_attention_interval':'full_attention_interval','linear_attention.key_head_count':'linear_num_key_heads',
            'linear_attention.value_head_count':'linear_num_value_heads','linear_attention.key_length':'linear_key_head_dim',
            'linear_attention.value_length':'linear_value_head_dim','linear_attention.conv_kernel':'linear_conv_kernel_dim'}
        vision_fields={'vision.block_count':'depth','vision.embedding_length':'hidden_size',
            'vision.feed_forward_length':'intermediate_size','vision.attention.head_count':'num_heads',
            'vision.patch_size':'patch_size','vision.temporal_patch_size':'temporal_patch_size',
            'vision.spatial_merge_size':'spatial_merge_size','vision.position_embedding_count':'num_position_embeddings',
            'vision.projection_dim':'out_hidden_size'}
        metadata += [('u',p+k,t[value]) for k,value in text_fields.items()]
        metadata += [('u',p+k,v[value]) for k,value in vision_fields.items()]
        metadata += [('u',p+'rope.dimension_count',int(t['head_dim']*t['partial_rotary_factor'])),
            ('u',p+'primary_eos_id',json.loads((policy.FIXTURE/'generation_config.json').read_text())['eos_token_id']),
            ('u',p+'source_text_eos_id',t['eos_token_id']),('u',p+'source_top_level_eos_id',config['eos_token_id'])]
        meta_fixture=out/'original-metadata.tsv'
        meta_fixture.write_text(''.join('%s %s %s\n'%item for item in metadata))
        report['metadata_fields']=len(metadata)
        command('configure',['cmake','-S',ROOT,'-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release',
            '-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        command('build',['cmake','--build',scratch/'build','--target','test-lighton4b-gguf-inspection','--parallel','2'])
        binary=scratch/'build/test-lighton4b-gguf-inspection'
        report['executable_sha256']=sha(binary)
        command('controls',[binary,fixture,meta_fixture,scratch/'inspection.gguf'])
        text=(out/'controls.log').read_text()
        if 'passed=1 payload_bytes=0 model_inference=0' not in text:raise ValueError('component controls receipt missing')
        report['controls_stdout']=text
        if sys.argv[1:]==['--published-header']:
            from lighton4b_published_header import inspect_published
            inspect_published(binary,out)
        elif sys.argv[1:]:raise ValueError('unknown inspection arguments')
        report['status']='passed'
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error).replace(str(ROOT),'[repository]').replace(str(scratch),'[scratch]')
        raise
    finally:
        (out/'gguf-inspection-receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')


if __name__=='__main__':execute()
