#!/usr/bin/env python3
"""Retained upstream replay only; never repeat either32-layer decoder forward."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
FROZEN='bf6a7f930df72b79e80189356abe68f3cb29c2ec'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def execute():
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':
        raise RuntimeError('hosted portable replay required')
    import lighton4b_gdn_reference as reference
    torch,np,gguf,original,Config,Cache,runtime=reference.load_original()
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])
    retained=Path(os.environ['RETAINED_DECODER']);(out/'original').mkdir();(out/'native').mkdir()
    frozen=json.loads((retained/'decoder-compose-receipt.json').read_text())
    report={'schema':'lighton4b-retained-upstream/v1','status':'failed','frozen_run':41979168300,
        'full_original_decoder_repeated':False,'full_native_decoder_repeated':False,
        'checkpoint_payload_bytes':0,'quality_qualified':False,'original_runtime':runtime,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{p:sha(ROOT/p) for p in ('src/lighton4b_linear_block.h','src/qwen35_linear.cpp',
            'src/qwen35_delta.cpp','src/qwen35_math.h','tests/test_lighton4b_decoder_upstream.cpp',
            'tools/ci-heavy/lighton4b_decoder_upstream.py','CMakeLists.txt')}}
    def command(label,argv):
        with (out/(label+'.log')).open('w') as stream:
            subprocess.run([str(v) for v in argv],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        if frozen['source_commit']!=FROZEN:raise ValueError('frozen decoder source differs')
        for path,digest in frozen['source_sha256'].items():
            blob=subprocess.check_output(['git','show',FROZEN+':'+path],cwd=ROOT)
            if hashlib.sha256(blob).hexdigest()!=digest:raise ValueError('frozen source record differs')
            if path!='CMakeLists.txt' and sha(ROOT/path)!=digest:raise ValueError('retained runtime changed')
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
        weights={('full.' if full else 'linear.')+name:np.ascontiguousarray(p.detach().numpy())
            for full,prototype in prototypes.items() for name,p in prototype.named_parameters()}
        weights['norm.weight']=torch.linspace(-0.02,0.02,2560).numpy()
        for name,value in weights.items():
            record=frozen['synthetic_weight_records'][name]
            if list(value.shape)!=record['shape'] or hashlib.sha256(value.tobytes()).hexdigest()!=record['sha256']:
                raise ValueError('regenerated synthetic weight differs: '+name)
        report['exact_regenerated_weight_records']=len(weights)
        linear_weights={name:np.ascontiguousarray(p.detach().numpy()) for name,p in prototypes[False].named_parameters()}
        reference.write_gguf(scratch/'weights.gguf',linear_weights,gguf)
        # Genuine layer1, but no full TextModel is constructed or executed.
        with torch.device('meta'):block=original.Qwen3_5DecoderLayer(config,1)
        block.load_state_dict(prototypes[False].state_dict(),assign=True);block.eval()
        def load(lane,name):
            path=retained/lane/(name+'.f32')
            record=frozen['original_arrays'][name]
            digest=record['sha256'] if lane=='original' else next(x['native_sha256'] for x in frozen['stages'] if x['stage']==name)
            if sha(path)!=digest:raise ValueError('retained array changed')
            return np.fromfile(path,dtype='<f4').reshape(record['shape'])
        refs={};phase=['source_input'];handles=[]
        def put(name,value):refs[phase[0]+'.'+name]=np.ascontiguousarray(value.detach().float().numpy()).copy()
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
        with torch.inference_mode():
            for label,lane in [('source_input','original'),('native_input','native')]:
                phase[0]=label;x=torch.from_numpy(load(lane,'prefill.layer.0').copy());cache=Cache(config=config)
                put('input',x)
                result=block(hidden_states=x,position_embeddings=None,past_key_values=cache)
                put('hidden',result[0] if isinstance(result,tuple) else result)
                put('conv',cache.layers[1].conv_states);put('recurrent',cache.layers[1].recurrent_states)
        for handle in handles:handle.remove()
        reference.write_gguf(scratch/'reference.gguf',refs,gguf)
        for name,value in refs.items():(out/'original'/(name+'.f32')).write_bytes(value.astype('<f4').tobytes())
        del block,prototypes,weights,linear_weights,cache;gc.collect()
        command('configure',['cmake','-S',ROOT,'-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF',
            '-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        command('build',['cmake','--build',scratch/'build','--target','test-lighton4b-decoder-upstream','--parallel','2'])
        command('native',[scratch/'build/test-lighton4b-decoder-upstream',scratch/'weights.gguf',scratch/'reference.gguf',out/'native'])
        comparisons=[]
        for label in ('source_input','native_input'):
            for stage in ('input_norm','attention','attention_residual','post_norm','gate','up','activation','down','hidden','conv','recurrent'):
                name=label+'.'+stage;value=np.fromfile(out/'native'/(name+'.f32'),dtype='<f4').reshape(refs[name].shape)
                comparisons.append({'stage':name,**reference.metrics(value,refs[name],np)})
        report['same_input_stages']=comparisons
        report['source_fed_ffn_stages']=[]
        for label in ('source_input','native_input'):
            for boundary,stage in [('source_post_norm','gate'),('source_post_norm','up'),('source_gate_up','activation'),('source_activation','down')]:
                name=label+'.'+boundary+'.'+stage;gold=refs[label+'.'+stage]
                value=np.fromfile(out/'native'/(name+'.f32'),dtype='<f4').reshape(gold.shape)
                report['source_fed_ffn_stages'].append({'stage':name,**reference.metrics(value,gold,np)})
        import math
        progress=[]
        for i in range(32):
            record=next(x for x in frozen['stages'] if x['stage']=='prefill.layer.'+str(i))
            relative_l2=record['rms']*math.sqrt(record['elements'])/record['reference_l2']
            progress.append({'layer':i,'kind':'full_attention' if i%4==3 else 'linear_attention',
                'max_abs':record['max_abs'],'reference_max_abs':record['reference_max_abs'],
                'rms_error':record['rms'],'relative_l2_error':relative_l2,
                'relative_l2_growth':relative_l2/progress[-1]['relative_l2_error'] if progress else None})
        report['prefill_layer_progression']=progress
        peak=max(progress[1:],key=lambda x:x['relative_l2_growth'])
        if peak['layer']!=1:raise ValueError('declared first-amplification target differs from frozen panel')
        report['target_reason']='Layer0-to1 is the earliest and largest adjacent multiplicative growth in normalized L2 error; no new quality threshold is declared.'
        report['target_layer']=1
        report['source_norm_stages']=[]
        for label in ('source_input','native_input'):
            for stage in ('conv','recurrent'):
                name=label+'.source_norm.'+stage;gold=refs[label+'.'+stage]
                value=np.fromfile(out/'native'/(name+'.f32'),dtype='<f4').reshape(gold.shape)
                report['source_norm_stages'].append({'stage':name,**reference.metrics(value,gold,np)})
        anchors=[]
        for stage,retained_name in [('hidden','prefill.layer.1'),('conv','prefill.cache.1.conv'),('recurrent','prefill.cache.1.recurrent')]:
            gold=load('original',retained_name);value=refs['source_input.'+stage]
            anchors.append({'stage':'original.'+stage,'byte_exact':value.tobytes()==gold.tobytes(),**reference.metrics(value,gold,np)})
            gold=load('native',retained_name);value=np.fromfile(out/'native'/('native_input.'+stage+'.f32'),dtype='<f4').reshape(gold.shape)
            anchors.append({'stage':'native.'+stage,'byte_exact':value.tobytes()==gold.tobytes(),**reference.metrics(value,gold,np)})
        report['frozen_anchors']=anchors
        report['original_input_propagation']=[{'stage':stage,**reference.metrics(refs['native_input.'+stage],refs['source_input.'+stage],np)}
            for stage in ('input_norm','hidden','conv','recurrent')]
        report['initial_failure_retained']=next(x for x in frozen['stages'] if x['stage']=='prefill.cache.21.recurrent')
        if not all(x['passed'] for x in comparisons+report['source_norm_stages']+report['source_fed_ffn_stages']+anchors):
            raise ValueError('same-input/retained anchor gate failed; no attribution promotion')
        report['status']='diagnostic_complete'
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error).replace(str(ROOT),'[repository]').replace(str(scratch),'[scratch]')
        raise
    finally:(out/'upstream-receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':execute()
