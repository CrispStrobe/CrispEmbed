#!/usr/bin/env python3
"""Hosted original32-layer hidden forward with explicitly shared synthetic weights."""
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def execute():
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':
        raise RuntimeError('hosted portable execution required')
    import lighton4b_gdn_reference as reference
    torch,np,gguf,original,Config,Cache,runtime=reference.load_original()
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'lighton4b-decoder-compose'
    scratch.mkdir(parents=True,exist_ok=False);(out/'original').mkdir();(out/'native').mkdir()
    paths=('src/lighton4b_decoder_compose.h','src/lighton4b_linear_block.h','src/lighton4b_attention_block.h',
        'src/qwen35_attention.cpp','src/qwen35_linear.cpp','src/qwen35_delta.cpp','src/qwen35_math.h',
        'tests/test_lighton4b_decoder_compose.cpp','tools/ci-heavy/lighton4b_decoder_compose.py',
        'tools/ci-heavy/lighton4b_gdn_reference.py','CMakeLists.txt')
    report={'schema':'lighton4b-decoder-compose/v1','status':'failed','checkpoint_payload_bytes':0,
        'format5_admitted':False,'logits_observed':False,'quality_qualified':False,'full_model_inference_observed':False,
        'shared_synthetic_weight_sets':2,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{p:sha(ROOT/p) for p in paths},'original_runtime':runtime,'stages':[]}
    def command(label,argv):
        with (out/(label+'.log')).open('w') as stream:
            subprocess.run([str(v) for v in argv],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        config=Config(**json.loads((ROOT/'tests/fixtures/lightonocr3-4b-source/config.json').read_text())['text_config'])
        config._attn_implementation='eager';torch.manual_seed(195034)
        prototypes={False:original.Qwen3_5DecoderLayer(config,0).eval().float(),
                    True:original.Qwen3_5DecoderLayer(config,3).eval().float()}
        with torch.no_grad():
            for prototype in prototypes.values():
                for name,param in prototype.named_parameters():
                    if param.ndim>=2:param.normal_(0,0.015)
                    elif name.endswith('A_log'):param.copy_(torch.linspace(-2,0,param.numel()))
                    elif name.endswith('dt_bias'):param.fill_(-0.5)
                    elif name=='linear_attn.norm.weight':param.fill_(0.9)
                    elif name in ('self_attn.q_norm.weight','self_attn.k_norm.weight'):param.normal_(0,0.15)
                    else:param.zero_()
        # Construct the genuine original TextModel without allocating32 distinct
        # dense parameter sets or its unused2.5GB embedding. Only parameters used
        # by inputs_embeds forward are assigned CPU storage from two prototypes.
        with torch.device('meta'):model=original.Qwen3_5TextModel(config)
        shared=[]
        for i,layer in enumerate(model.layers):
            prototype=prototypes[i%4==3]
            layer.load_state_dict(prototype.state_dict(),assign=True)
            exact=all(p.untyped_storage().data_ptr()==dict(prototype.named_parameters())[name].untyped_storage().data_ptr()
                for name,p in layer.named_parameters())
            if not exact:raise ValueError('original shared-weight storage differs')
            shared.append({'layer':i,'kind':'full_attention' if i%4==3 else 'linear_attention','storage_shared':True})
        model.norm.weight=torch.nn.Parameter(torch.linspace(-0.02,0.02,2560))
        model.rotary_emb=original.Qwen3_5TextRotaryEmbedding(config)
        model.eval()
        if not model.embed_tokens.weight.is_meta:raise ValueError('unused embedding unexpectedly allocated')
        if [name for name,p in model.named_parameters() if p.is_meta]!=['embed_tokens.weight']:
            raise ValueError('required original CPU parameter remains meta')
        report['original_text_model_source_sha256']=hashlib.sha256(inspect.getsource(type(model)).encode()).hexdigest()
        report['original_text_model_forward_signature']=str(inspect.signature(model.forward))
        report['shared_weight_controls']=shared;report['unused_embedding_meta']=True
        weights={('full.' if full else 'linear.')+name:np.ascontiguousarray(p.detach().numpy())
            for full,prototype in prototypes.items() for name,p in prototype.named_parameters()}
        weights['norm.weight']=np.ascontiguousarray(model.norm.weight.detach().numpy())
        if len(weights)!=26:raise ValueError('expected25 shared block parameters plus final norm')
        report['synthetic_weight_records']={name:{'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for name,v in weights.items()}
        reference.write_gguf(scratch/'weights.gguf',weights,gguf);report['weights_gguf_sha256']=sha(scratch/'weights.gguf')
        hidden=(torch.sin(torch.arange(3*2560,dtype=torch.float32)*0.013)*0.3).reshape(1,3,2560)
        inputs={'prefill':hidden[:,:2].contiguous(),'cached0':hidden[:,2:].contiguous(),'full_prefix':hidden.contiguous()}
        coords=torch.stack([torch.arange(3),torch.arange(3)*3+7,torch.arange(3)*5+11]).unsqueeze(1)
        positions={'prefill':coords[:,:,:2].contiguous(),'cached0':coords[:,:,2:].contiguous(),'full_prefix':coords.contiguous()}
        refs={};phase=['prefill'];handles=[]
        def put(name,value):
            a=np.ascontiguousarray(value.detach().float().numpy()).copy()
            if not np.isfinite(a).all():raise ValueError('nonfinite original '+name)
            if phase[0]+'.'+name in refs:raise ValueError('duplicate original stage')
            refs[phase[0]+'.'+name]=a
        for i,layer in enumerate(model.layers):
            def capture(module,args,result,i=i):put('layer.'+str(i),result[0] if isinstance(result,tuple) else result)
            handles.append(layer.register_forward_hook(capture))
        cache=Cache(config=config)
        def forward(x,cache,label):
            result=model(inputs_embeds=x,position_ids=positions[label],past_key_values=cache,use_cache=True,return_dict=True)
            return result.last_hidden_state
        with torch.inference_mode():
            for label,x in inputs.items():
                phase[0]=label
                if label=='full_prefix':cache=Cache(config=config)
                put('input',x);put('positions',positions[label].float());put('norm',forward(x,cache,label))
                for i,c in enumerate(cache.layers):
                    for field in (('keys','values') if i%4==3 else ('conv_states','recurrent_states')):
                        put('cache.'+str(i)+'.'+{'conv_states':'conv','recurrent_states':'recurrent'}.get(field,field),getattr(c,field))
        for handle in handles:handle.remove()
        # Original observation neutrality covers norm and every independent cache.
        neutral=[];cache=Cache(config=config)
        with torch.inference_mode():
            for label,x in inputs.items():
                if label=='full_prefix':cache=Cache(config=config)
                result=forward(x,cache,label)
                pairs=[('norm',result)]
                for i,c in enumerate(cache.layers):
                    pairs += [('cache.'+str(i)+'.'+{'conv_states':'conv','recurrent_states':'recurrent'}.get(field,field),getattr(c,field))
                        for field in (('keys','values') if i%4==3 else ('conv_states','recurrent_states'))]
                for name,v in pairs:
                    neutral.append({'stage':label+'.'+name,'byte_exact':np.ascontiguousarray(v.numpy()).tobytes()==refs[label+'.'+name].tobytes()})
        report['observer_neutrality']=neutral
        if not all(x['byte_exact'] for x in neutral):raise ValueError('original hooks changed ordinary forward/cache')
        report['original_text_model_forward_observed']=True
        reference.write_gguf(scratch/'reference.gguf',refs,gguf)
        for name,v in refs.items():(out/'original'/(name+'.f32')).write_bytes(v.astype('<f4').tobytes())
        report['original_arrays']={name:{'shape':list(v.shape),'sha256':hashlib.sha256(v.tobytes()).hexdigest()} for name,v in refs.items()}
        del model,prototypes,weights,cache;gc.collect()
        command('configure',['cmake','-S',ROOT,'-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF',
            '-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        command('build',['cmake','--build',scratch/'build','--target','test-lighton4b-decoder-compose','--parallel','2'])
        binary=scratch/'build/test-lighton4b-decoder-compose';report['executable_sha256']=sha(binary)
        command('native',[binary,scratch/'weights.gguf',scratch/'reference.gguf',out/'native'])
        report['native_stdout']=(out/'native.log').read_text().replace(str(ROOT),'[repository]').replace(str(scratch),'[scratch]')
        for label in inputs:
            names=['layer.'+str(i) for i in range(32)]+['norm']
            for i in range(32):names += ['cache.'+str(i)+'.'+field for field in (('keys','values') if i%4==3 else ('conv','recurrent'))]
            for suffix in names:
                name=label+'.'+suffix;path=out/'native'/(name+'.f32');actual=np.fromfile(path,dtype='<f4').reshape(refs[name].shape)
                report['stages'].append({'stage':name,'native_sha256':sha(path),**reference.metrics(actual,refs[name],np)})
        report['earliest_failure']=next((x['stage'] for x in report['stages'] if not x['passed']),None)
        if report['earliest_failure']:raise ValueError('original/native decoder mismatch: '+report['earliest_failure'])
        report['status']='passed'
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error).replace(str(ROOT),'[repository]').replace(str(scratch),'[scratch]')
        raise
    finally:(out/'decoder-compose-receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':execute()
