#!/usr/bin/env python3
"""Hosted full-width synthetic original/native linear decoder block, no checkpoint."""
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def execute():
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':
        raise RuntimeError('hosted portable execution required')
    import lighton4b_gdn_reference as reference
    torch,np,gguf,original,Config,Cache,runtime=reference.load_original()
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'lighton4b-linear-block'
    scratch.mkdir(parents=True,exist_ok=False);(out/'original').mkdir();(out/'native').mkdir()
    report={'schema':'lighton4b-linear-block/v1','status':'failed','full_model_inference_observed':False,
        'checkpoint_payload_bytes':0,'format5_admitted':False,'logits_observed':False,'quality_qualified':False,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{p:sha(ROOT/p) for p in ('src/lighton4b_linear_block.h','src/qwen35_linear.cpp',
            'src/qwen35_delta.cpp','tests/test_lighton4b_linear_block.cpp','tools/ci-heavy/lighton4b_linear_block.py','CMakeLists.txt')},
        'original_runtime':runtime,'stages':[]}
    def command(label,argv):
        with (out/(label+'.log')).open('w') as stream:
            subprocess.run([str(v) for v in argv],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        config=Config(**json.loads((ROOT/'tests/fixtures/lightonocr3-4b-source/config.json').read_text())['text_config'])
        torch.manual_seed(195032)
        block=original.Qwen3_5DecoderLayer(config,0).eval().float()
        report['original_decoder_source_sha256']=hashlib.sha256(inspect.getsource(type(block)).encode()).hexdigest()
        report['original_decoder_forward_signature']=str(inspect.signature(block.forward))
        with torch.no_grad():
            for name,param in block.named_parameters():
                if param.ndim>=2:param.normal_(0,0.015)
                elif name.endswith('A_log'):param.copy_(torch.linspace(-2,0,param.numel()))
                elif name.endswith('dt_bias'):param.fill_(-0.5)
                elif name=='linear_attn.norm.weight':param.fill_(0.9)
                else:param.zero_()
        # Exact checkpoint geometry; all fourteen block tensors are retained by hash.
        weights={name:np.ascontiguousarray(p.detach().numpy()) for name,p in block.named_parameters()}
        if len(weights)!=14:raise ValueError('expected fourteen original block parameters')
        report['synthetic_weight_records']={name:{'shape':list(value.shape),'sha256':hashlib.sha256(value.tobytes()).hexdigest()} for name,value in weights.items()}
        reference.write_gguf(scratch/'weights.gguf',weights,gguf)
        report['weights_gguf_sha256']=sha(scratch/'weights.gguf')
        hidden=(torch.sin(torch.arange(3*2560,dtype=torch.float32)*0.013)*0.3).reshape(1,3,2560)
        inputs={'prefill':hidden[:,:2].contiguous(),'cached0':hidden[:,2:].contiguous(),'full_prefix':hidden.contiguous()}
        phase=['prefill'];refs={};handles=[]
        def put(name,value):
            a=np.ascontiguousarray(value.detach().float().numpy()).copy()
            if not np.isfinite(a).all():raise ValueError('nonfinite original '+name)
            refs[phase[0]+'.'+name]=a
        def hook(name):
            def capture(module,args,result):put(name,result)
            return capture
        handles.append(block.input_layernorm.register_forward_hook(hook('input_norm')))
        handles.append(block.linear_attn.register_forward_hook(hook('attention')))
        def post(module,args,result):put('attention_residual',args[0]);put('post_norm',result)
        handles.append(block.post_attention_layernorm.register_forward_hook(post))
        handles.append(block.mlp.gate_proj.register_forward_hook(hook('gate')))
        handles.append(block.mlp.up_proj.register_forward_hook(hook('up')))
        def down(module,args,result):put('activation',args[0]);put('down',result)
        handles.append(block.mlp.down_proj.register_forward_hook(down))
        cache=Cache(config=config)
        def forward(x,cache):
            result=block(hidden_states=x,past_key_values=cache,position_embeddings=None,
                         cache_position=torch.arange(x.shape[1],dtype=torch.long))
            return result[0] if isinstance(result,tuple) else result
        with torch.inference_mode():
            for label,x in inputs.items():
                phase[0]=label
                if label=='full_prefix':cache=Cache(config=config)
                put('input',x);put('hidden',forward(x,cache))
                put('conv',cache.layers[0].conv_states);put('recurrent',cache.layers[0].recurrent_states)
        for handle in handles:handle.remove()
        neutrality=[];cache=Cache(config=config)
        with torch.inference_mode():
            for label,x in inputs.items():
                if label=='full_prefix':cache=Cache(config=config)
                result=forward(x,cache)
                for name,value in [('hidden',result),('conv',cache.layers[0].conv_states),('recurrent',cache.layers[0].recurrent_states)]:
                    exact=np.ascontiguousarray(value.numpy()).tobytes()==refs[label+'.'+name].tobytes()
                    neutrality.append({'stage':label+'.'+name,'byte_exact':exact})
        report['observer_neutrality']=neutrality
        if not all(x['byte_exact'] for x in neutrality):raise ValueError('original hooks changed ordinary output')
        reference.write_gguf(scratch/'reference.gguf',refs,gguf)
        for name,value in refs.items():(out/'original'/(name+'.f32')).write_bytes(value.astype('<f4').tobytes())
        report['original_arrays']={name:{'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for name,v in refs.items()}
        del block,weights,cache;gc.collect()
        command('configure',['cmake','-S',ROOT,'-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF',
            '-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        command('build',['cmake','--build',scratch/'build','--target','test-lighton4b-linear-block','--parallel','2'])
        binary=scratch/'build/test-lighton4b-linear-block';report['executable_sha256']=sha(binary)
        command('native',[binary,scratch/'weights.gguf',scratch/'reference.gguf',out/'native'])
        report['native_stdout']=(out/'native.log').read_text()
        for phase in inputs:
            for stage in ('input_norm','attention','attention_residual','post_norm','gate','up','activation','down','hidden','conv','recurrent'):
                name=phase+'.'+stage;actual=np.fromfile(out/'native'/(name+'.f32'),dtype='<f4').reshape(refs[name].shape)
                result=reference.metrics(actual,refs[name],np)
                report['stages'].append({'stage':name,**result})
        report['earliest_failure']=next((r['stage'] for r in report['stages'] if not r['passed']),None)
        if report['earliest_failure']:raise ValueError('original/native stage mismatch: '+report['earliest_failure'])
        report['status']='passed'
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error).replace(str(ROOT),'[repository]').replace(str(scratch),'[scratch]')
        raise
    finally:(out/'linear-block-receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':execute()
