#!/usr/bin/env python3
"""Checkpoint layer0 attribution only: identical retained input and independent weights.

No full model, image forward, vocabulary projection, generation or quality claim.
The F32 arms are local same-weight operator diagnostics, never new reference gold.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request
ROOT=Path(__file__).resolve().parents[2]
STAGES=('input_norm','qkv','z','a','b','gdn_norm','attention','attention_residual','post_norm','gate','up','activation','down','hidden','conv','recurrent')
def require(value,message):
    if not value:raise ValueError(message)
def digest(data):return hashlib.sha256(data).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',choices=('v1','v2-unanchored-diagnostic'),default='v1')
    parser.add_argument('--bf16-boundaries',action='store_true',help='Also compare gated quant-preserving BF16 activation oracle')
    parser.add_argument('--bf16-only',action='store_true',help='Official retained-input layer0 only; no native build or candidate')
    parser.add_argument('--retained-root',type=Path,help='Mounted original stage directory; exact manifest and payload hashes required')
    args=parser.parse_args()
    require((os.environ.get('GITHUB_ACTIONS')=='true' or (args.bf16_only and Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir())) and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable only')
    import lighton4b_gdn_reference as ref
    import lighton4b_native_document as native
    from lighton4b_prefill_compare import ORIGINAL_REPORT_SHA
    torch,np,gguf,original,Config,Cache,runtime=ref.load_original()
    sys.path.insert(0,str(ROOT/'models'));import lightonocr3_4b_profile as policy
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'layer0';scratch.mkdir()
    report={'schema':'lighton4b-layer0-source-fed/v1','status':'failed','quality_qualified':False,
        'full_model_forward_executed':False,'generation_executed':False,'runtime':runtime,
        'protocol':args.protocol,'bf16_only':args.bf16_only,'hosted_provider':'github-actions' if os.environ.get('GITHUB_ACTIONS')=='true' else 'kaggle',
        'frozen_bf16_parity_qualified':False,
        'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'arms':{}}
    def save(): (out/'layer0-source-fed.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    def run(label,args,env=None):
        with(out/(label+'.log')).open('w')as log:subprocess.run(list(map(str,args)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        import shutil
        report['preflight']={'memory':native.memory(),'free_bytes':shutil.disk_usage(scratch).free}
        require(report['preflight']['memory']['available_bytes']>=5*2**30 and report['preflight']['free_bytes']>=9*2**30,'resource headroom')
        retained=args.retained_root
        if retained is None:
            from kaggle.api.kaggle_api_extended import KaggleApi
            provider=KaggleApi();provider.authenticate()
            provider.kernels_output(os.environ['KAGGLE_ACCOUNT']+'/lighton4b-bf16-targeted-original-cpu-prefill',str(scratch/'retained'),
                file_pattern=r'^lighton4b-bf16-prefill/stages/(prefill\.json|prefill\.(input|layer\.0)\.f32)$',quiet=True)
            retained=scratch/'retained/lighton4b-bf16-prefill/stages'
        manifest=retained/'prefill.json'
        require(native.sha(manifest)==ORIGINAL_REPORT_SHA,'original manifest differs')
        records=json.loads(manifest.read_text())['stages']
        for name in ('prefill.input','prefill.layer.0'):
            require(native.sha(retained/(name+'.f32'))==records[name]['sha256'],'original payload differs')
        x=np.fromfile(retained/'prefill.input.f32',dtype='<f4').reshape(1,154,2560)
        frozen=np.fromfile(retained/'prefill.layer.0.f32',dtype='<f4').reshape(1,154,2560)
        report['input']={'manifest_sha256':ORIGINAL_REPORT_SHA,'sha256':digest(x.tobytes()),'shape':list(x.shape)}
        config=Config(**json.loads((policy.FIXTURE/'config.json').read_text())['text_config'])
        if not args.bf16_only:run('configure',['cmake','-S',ROOT,'-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        if not args.bf16_only:run('build',['cmake','--build',scratch/'build','--target','lighton4b-layer0-source-fed','--parallel','2'])
        binary=scratch/'build/lighton4b-layer0-source-fed'
        env=os.environ.copy();env.update({k:'1' for k in ('CRISPEMBED_LIGHTON4B_LINEAR_BLOCK','CRISPEMBED_QWEN35_OCR','CRISPEMBED_QWEN35_GDN_GROUPED_HEADS','CRISPEMBED_LIGHTON4B_QUANT','CRISPEMBED_QWEN35_QUANT')})
        env.update({k:'0' for k in ('CRISPEMBED_QWEN35_PRECISE_RMS','CRISPEMBED_QWEN35_IMATRIX','CRISPEMBED_QWEN35_WEIGHT_CACHE')})
        report['gates']={k:v for k,v in env.items() if k.startswith('CRISPEMBED_')}
        if not args.bf16_only:report['binary_sha256']=native.sha(binary)
        def original_arm(label,weights,dtype,boundary=False):
            # Frozen BF16 capture used the default CPU backend; component F32
            # diagnostics retain the audited fallback setup. Record both.
            torch.backends.mkldnn.enabled = dtype == torch.bfloat16 and not boundary
            torch.use_deterministic_algorithms(dtype != torch.bfloat16 or boundary)
            with torch.device('meta'):block=original.Qwen3_5DecoderLayer(config,0)
            block.load_state_dict({k:torch.from_numpy(v.copy()).to(torch.float32 if boundary else dtype)for k,v in weights.items()},assign=True,strict=True);block.eval()
            if boundary:
                import types
                import torch.nn.functional as F
                def linear(module,value):return F.linear(value.float(),module.weight.float(),module.bias).to(torch.bfloat16)
                for module in block.modules():
                    if isinstance(module,torch.nn.Linear):module.forward=types.MethodType(linear,module)
                def convolution(module,value):return module._conv_forward(value.float(),module.weight.float(),module.bias).to(torch.bfloat16)
                block.linear_attn.conv1d.forward=types.MethodType(convolution,block.linear_attn.conv1d)
                def gated_norm(module,value,gate=None):
                    value=value.float();normalized=value*torch.rsqrt(value.square().mean(-1,keepdim=True)+module.variance_epsilon)
                    weighted=(normalized.bfloat16().float()*module.weight.float()).bfloat16()
                    return (weighted.float()*F.silu(gate.float())).bfloat16()
                block.linear_attn.norm.forward=types.MethodType(gated_norm,block.linear_attn.norm)
            stages={};handles=[]
            def put(name,t):stages[name]=np.ascontiguousarray(t.detach().float().numpy()).copy()
            def hook(name):
                def capture(module,args,result):put(name,result)
                return capture
            for module,name in ((block.input_layernorm,'input_norm'),(block.linear_attn,'attention'),(block.mlp.gate_proj,'gate'),(block.mlp.up_proj,'up')):
                handles.append(module.register_forward_hook(hook(name)))
            for attr,name in (('in_proj_qkv','qkv'),('in_proj_z','z'),('in_proj_a','a'),('in_proj_b','b')):
                handles.append(getattr(block.linear_attn,attr).register_forward_hook(hook(name)))
            def norm(module,args,result):put('gdn_norm',result.reshape(1,154,32,128))
            handles.append(block.linear_attn.norm.register_forward_hook(norm))
            def post(module,args,result):put('attention_residual',args[0]);put('post_norm',result)
            def down(module,args,result):put('activation',args[0]);put('down',result)
            handles.append(block.post_attention_layernorm.register_forward_hook(post));handles.append(block.mlp.down_proj.register_forward_hook(down))
            cache=Cache(config=config);input_tensor=torch.from_numpy(x.copy()).to(dtype)
            with torch.inference_mode():
                result=block(hidden_states=input_tensor,past_key_values=cache,position_embeddings=None,cache_position=torch.arange(154))
            put('hidden',result[0]if isinstance(result,tuple)else result)
            put('conv',cache.layers[0].conv_states);put('recurrent',cache.layers[0].recurrent_states)
            for handle in handles:handle.remove()
            # Ordinary replay must agree before any attribution is accepted.
            cache2=Cache(config=config)
            with torch.inference_mode():ordinary=block(hidden_states=input_tensor,past_key_values=cache2,position_embeddings=None,cache_position=torch.arange(154))
            ordinary=ordinary[0]if isinstance(ordinary,tuple)else ordinary
            require(digest(ordinary.float().numpy().tobytes())==digest(stages['hidden'].tobytes()),'observer changed layer hidden')
            folder=out/label;folder.mkdir()
            for name,value in stages.items():(folder/(name+'.f32')).write_bytes(value.astype('<f4').tobytes())
            report['arms'][label]={'dtype':str(dtype),'quant_preserving_bf16_activation_oracle':boundary,'mkldnn_enabled':torch.backends.mkldnn.enabled,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'observer_neutral':True,'hidden_vs_frozen_bf16':ref.metrics(stages['hidden'],frozen,np),
                'stages':{k:{'shape':list(v.shape),'sha256':digest(v.tobytes()),'abs_max':float(np.abs(v).max())}for k,v in stages.items()}}
            baseline=out/'official-same-weight-f32-diagnostic'
            if baseline.is_dir() and label!='official-same-weight-f32-diagnostic':
                report['arms'][label]['vs_official_same_input_f32']=[{'stage':name,**ref.metrics(value,np.fromfile(baseline/(name+'.f32'),dtype='<f4').reshape(value.shape),np)}for name,value in stages.items()]
            del block,cache,cache2;gc.collect();save();return stages
        def native_arm(label,path,expected,boundary=False):
            local_env=env.copy();local_env['CRISPEMBED_LIGHTON4B_BF16_LINEAR']='1' if boundary else '0'
            folder=out/label;folder.mkdir();run(label,[binary,path,retained/'prefill.input.f32',folder],local_env)
            metrics=[]
            for name in STAGES:
                value=np.fromfile(folder/(name+'.f32'),dtype='<f4').reshape(expected[name].shape)
                metrics.append({'stage':name,**ref.metrics(value,expected[name],np),
                    'original_abs_max':float(np.abs(expected[name]).max()),'native_abs_max':float(np.abs(value).max())})
            report['arms'][label]={'bf16_boundary_gate':boundary,'same_input_same_weights':True,'stages':metrics,'earliest_failure':next((r['stage']for r in metrics if not r['passed']),None)};save()
        # Read only the immutable official layer0 interval, not the9GB model.
        header_path=policy.FIXTURE/policy.HEADER_FILE;raw_header=header_path.read_bytes()
        require(digest(raw_header)==policy.HEADER_SHA256,'checkpoint header differs')
        header=json.loads(raw_header);prefix='model.language_model.layers.0.'
        selected={k[len(prefix):]:v for k,v in header.items()if k.startswith(prefix)}
        require(len(selected)==14,'official layer tensor count differs')
        begin=min(v['data_offsets'][0]for v in selected.values());end=max(v['data_offsets'][1]for v in selected.values())
        start=8+policy.HEADER_BYTES+begin;stop=8+policy.HEADER_BYTES+end
        request=urllib.request.Request(f'https://huggingface.co/{policy.MODEL}/resolve/{policy.REVISION}/model.safetensors',headers={'Range':f'bytes={start}-{stop-1}'})
        with urllib.request.urlopen(request,timeout=120)as response:
            require(response.status==206 and response.headers['Content-Range']==f'bytes {start}-{stop-1}/{policy.CHECKPOINT_BYTES}','official bounded range not honored')
            raw=response.read(stop-start+1)
        require(len(raw)==stop-start,'official range size differs')
        weights={}
        for name,row in selected.items():
            require(row['dtype']=='BF16','official source dtype differs');a,b=row['data_offsets']
            values=np.frombuffer(raw[a-begin:b-begin],dtype='<u2').astype(np.uint32)<<16
            weights[name]=values.view('<f4').reshape(row['shape']).copy()
        report['official_layer0']={'repository':policy.MODEL,'revision':policy.REVISION,'range':[start,stop],
            'range_sha256':digest(raw),'whole_checkpoint_rehashed':False,
            'tensor_sha256':{k:digest(v.tobytes())for k,v in weights.items()}}
        del raw
        bf16=original_arm('official-bf16',weights,torch.bfloat16)
        report['frozen_bf16_hidden_exact']=digest(bf16['hidden'].tobytes())==digest(frozen.tobytes())
        if not report['frozen_bf16_hidden_exact']:
            report['frozen_bf16_control_failure']='official BF16 layer0 anchor not exact'
            require(args.protocol=='v2-unanchored-diagnostic','official BF16 layer0 anchor not exact')
        # V2 preserves failed frozen BF16 replay as a blocking parity result,
        # but permits same-input/same-weight local operator diagnostics.
        save()
        if args.bf16_only:
            report['status']='bf16-layer-anchored' if report['frozen_bf16_hidden_exact'] else 'bf16-layer-unanchored'
            return
        source_f32=original_arm('official-same-weight-f32-diagnostic',weights,torch.float32)
        ref.write_gguf(scratch/'official-layer0.gguf',weights,gguf)
        native_arm('native-official-f32',scratch/'official-layer0.gguf',source_f32)
        if args.bf16_boundaries:
            oracle=original_arm('official-bf16-boundary-oracle',weights,torch.bfloat16,True)
            native_arm('native-official-bf16-boundary',scratch/'official-layer0.gguf',oracle,True)
            del oracle
        del weights,bf16,source_f32;gc.collect()
        for recipe,(filename,expected,size)in native.CANDIDATES.items():
            model=scratch/filename;native.download(f'https://huggingface.co/{native.MODEL_REPO}/resolve/{native.MODEL_REVISION}/{filename}',model)
            require(model.stat().st_size==size and native.sha(model)==expected,'candidate bytes differ')
            reader=gguf.GGUFReader(str(model));tensors=[t for t in reader.tensors if t.name.startswith('l.layers.0.')]
            require(len(tensors)==14,'candidate layer0 count differs')
            quant_path=scratch/(recipe+'.gguf');writer=gguf.GGUFWriter(str(quant_path),'qwen35_gdn_component');weights={}
            for tensor in tensors:
                name=tensor.name[len('l.layers.0.'):];writer.add_tensor(name,np.array(tensor.data),raw_dtype=tensor.tensor_type)
                weights[name]=np.ascontiguousarray(gguf.quants.dequantize(tensor.data,tensor.tensor_type)).reshape(tuple(reversed(tensor.shape.tolist())))
            writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
            report['arms'][recipe+'-weights']={'candidate_sha256':expected,'independent_dequantizer':'gguf.quants.dequantize',
                'dequantized_tensor_sha256':{k:digest(v.tobytes())for k,v in weights.items()}}
            expected_stages=original_arm(recipe+'-original-dequant-f32',weights,torch.float32)
            native_arm(recipe+'-native-quant',quant_path,expected_stages)
            ref.write_gguf(scratch/'dequant.gguf',weights,gguf)
            native_arm(recipe+'-native-dequant-f32',scratch/'dequant.gguf',expected_stages)
            if args.bf16_boundaries:
                oracle=original_arm(recipe+'-bf16-boundary-oracle',weights,torch.bfloat16,True)
                native_arm(recipe+'-native-bf16-boundary',quant_path,oracle,True)
                # Gate-on calls must not alter a later legacy result or cache.
                native_arm(recipe+'-native-legacy-recovery',quant_path,expected_stages)
                for stage in STAGES:
                    require(native.sha(out/(recipe+'-native-legacy-recovery')/(stage+'.f32'))==native.sha(out/(recipe+'-native-quant')/(stage+'.f32')),'gated call changed legacy output')
                del oracle
            del tensors,reader,weights,expected_stages;gc.collect();model.unlink()
        report['status']='compared' if report['frozen_bf16_hidden_exact'] else 'diagnostic-compared-unanchored'
    except Exception as error:
        report['error_type']=type(error).__name__
        # Avoid signed URLs, credentials and provider exception bodies.
        report['error']=str(error)[:240]if isinstance(error,ValueError)else 'source-fed hosted replay failed'
        raise RuntimeError(report['error'])from None
    finally:save()
if __name__=='__main__':main()
