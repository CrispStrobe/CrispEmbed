#!/usr/bin/env python3
"""Real Q8 layer0 source-fed QKV dot/rounding attribution; no decoder forward."""
import csv
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parents[2]
def require(v,why):
    if not v:raise ValueError(why)
def digest(data):return hashlib.sha256(data).hexdigest()
def main():
    require(os.environ.get('GITHUB_ACTIONS')=='true' and os.environ.get('CRISPEMBED_BUILD_MODE')=='portable','hosted portable only')
    import lighton4b_gdn_reference as ref
    import lighton4b_native_document as native
    torch,np,gguf,original,Config,Cache,runtime=ref.load_original()
    runtime['verified_auxiliary_function_sha256']=runtime.pop('executed_function_sha256')
    runtime['auxiliary_scope']='Inherited module/GDN source verification; no decoder or GDN forward is executed.'
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH'])/'qkv-dot';scratch.mkdir()
    protocol=json.loads((ROOT/'docs/lighton4b-qkv-dot-v1.json').read_text())
    report={'schema':'lighton4b-qkv-dot/v1','status':'failed','recipe':'q8_0','quality_qualified':False,
        'decoder_forward_executed':False,'full_model_forward_executed':False,'generation_executed':False,
        'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'runtime':runtime,'torch_git_version':torch.version.git_version,'torch_cpu_capability':torch.backends.cpu.get_cpu_capability(),
        'protocol':protocol,'upstream_native_payload_reproduced_claimed':False}
    def save():(out/'qkv-dot.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    def command(label,args):
        with(out/(label+'.log')).open('w')as log:subprocess.run(list(map(str,args)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    env=os.environ.copy();env.update({k:'1'for k in ('CRISPEMBED_QWEN35_OCR','CRISPEMBED_QWEN35_QUANT','CRISPEMBED_LIGHTON4B_QUANT')})
    env.update({k:'0'for k in ('CRISPEMBED_QWEN35_WEIGHT_CACHE','CRISPEMBED_QWEN35_IMATRIX')})
    try:
        report['source_sha256']={name:native.sha(ROOT/name)for name in ('src/qwen35_linear.cpp','src/qwen35_math.h','src/lighton4b_bf16_linear.h','tests/lighton4b_qkv_dot.cpp','tools/ci-heavy/lighton4b_qkv_dot.py','docs/lighton4b-qkv-dot-v1.json','CMakeLists.txt')}
        import shutil
        report['preflight']={'memory':native.memory(),'free_bytes':shutil.disk_usage(scratch).free}
        require(report['preflight']['memory']['available_bytes']>=4*2**30 and report['preflight']['free_bytes']>=9*2**30,'bounded QKV resource headroom')
        from kaggle.api.kaggle_api_extended import KaggleApi
        provider=KaggleApi();provider.authenticate()
        provider.kernels_output(os.environ['KAGGLE_ACCOUNT']+'/lighton4b-bf16-targeted-original-cpu-prefill',str(scratch/'retained'),
            file_pattern=r'^lighton4b-bf16-prefill/stages/(prefill\.json|prefill\.input\.f32)$',quiet=True)
        retained=scratch/'retained/lighton4b-bf16-prefill/stages'
        require(native.sha(retained/'prefill.json')==protocol['original_manifest_sha256'],'original manifest differs')
        require(native.sha(retained/'prefill.input.f32')==protocol['original_input_sha256'],'original decoder input differs')
        filename,expected,size=native.CANDIDATES['q8_0'];model=scratch/filename
        native.download(f'https://huggingface.co/{native.MODEL_REPO}/resolve/{native.MODEL_REVISION}/{filename}',model)
        require(model.stat().st_size==size and native.sha(model)==expected,'immutable Q8 candidate differs')
        report['model']={'repository':native.MODEL_REPO,'revision':native.MODEL_REVISION,'file':filename,'bytes':size,'sha256':expected}
        reader=gguf.GGUFReader(str(model));prefix='l.layers.0.'
        tensors={t.name[len(prefix):]:t for t in reader.tensors if t.name.startswith(prefix)}
        require(len(tensors)==14,'candidate layer0 tensor inventory differs')
        def dequant(name):
            t=tensors[name]
            return np.ascontiguousarray(gguf.quants.dequantize(t.data,t.tensor_type)).reshape(tuple(reversed(t.shape.tolist())))
        weight=dequant('linear_attn.in_proj_qkv.weight');norm_weight=dequant('input_layernorm.weight')
        require(weight.shape==(8192,2560) and norm_weight.shape==(2560,),'actual source QKV/norm geometry differs')
        norm=original.Qwen3_5RMSNorm(2560,eps=1e-6);norm.weight=torch.nn.Parameter(torch.from_numpy(norm_weight.copy()));norm.eval()
        report['original_norm_forward_sha256']=digest(inspect.getsource(type(norm).forward).encode())
        x=torch.from_numpy(np.fromfile(retained/'prefill.input.f32',dtype='<f4').reshape(1,154,2560).copy()).bfloat16()
        with torch.inference_mode():normalized=norm(x).float().contiguous()
        require(digest(normalized.numpy().tobytes())==protocol['original_bf16_norm_sha256'],'frozen original BF16-normalized input differs')
        normalized.numpy().astype('<f4').tofile(scratch/'normalized.f32');weight.astype('<f4').tofile(scratch/'independent-weight.f32')
        with torch.inference_mode():gold=torch.nn.functional.linear(normalized,torch.from_numpy(weight)).contiguous();ordinary=torch.nn.functional.linear(normalized,torch.from_numpy(weight)).contiguous()
        require(gold.numpy().tobytes()==ordinary.numpy().tobytes(),'ordinary original QKV repeat differs')
        rounded=gold.bfloat16().float().numpy().copy();gold=gold.numpy().copy()
        require(digest(rounded.tobytes())==protocol['original_q8_oracle_qkv_bf16_sha256'],'retained original Q8 BF16 projection anchor differs')
        report['original']={'normalized_sha256':native.sha(scratch/'normalized.f32'),'independent_weight_sha256':native.sha(scratch/'independent-weight.f32'),
            'raw_sha256':digest(gold.tobytes()),'rounded_sha256':digest(rounded.tobytes()),'shape':list(gold.shape),'ordinary_repeat_exact':True}
        gold.astype('<f4').tofile(out/'original.raw.f32');rounded.astype('<f4').tofile(out/'original.bf16.f32')
        writer=gguf.GGUFWriter(str(scratch/'weights.gguf'),'qwen35_gdn_component')
        for name in ('linear_attn.in_proj_qkv.weight','linear_attn.in_proj_z.weight','linear_attn.in_proj_a.weight','linear_attn.in_proj_b.weight'):
            t=tensors[name];writer.add_tensor(name,np.array(t.data),raw_dtype=t.tensor_type)
        writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()
        del tensors,reader,norm,x,ordinary;gc.collect();model.unlink()
        command('configure',['cmake','-S',ROOT,'-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF','-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        command('build',['cmake','--build',scratch/'build','--target','lighton4b-qkv-dot','--parallel','2'])
        binary=scratch/'build/lighton4b-qkv-dot';report['binary_sha256']=native.sha(binary)
        libraries=list((scratch/'build').rglob('libggml-cpu.so'))
        require(len(libraries)==1,'one actual shared ggml CPU arithmetic object required')
        report['native_arithmetic_object_sha256']=native.sha(libraries[0])
        base=[binary,scratch/'weights.gguf',scratch/'normalized.f32',scratch/'independent-weight.f32']
        for mode in ('plain','observe'):
            folder=out/mode;folder.mkdir();command(mode,[*base,folder,mode,'-'])
        require((out/'plain/qkv.raw.f32').read_bytes()==(out/'observe/qkv.raw.f32').read_bytes(),'route observation changed native pre-round projection')
        require((out/'plain/qkv.bf16.f32').read_bytes()==(out/'observe/qkv.bf16.f32').read_bytes(),'route observation changed native rounded projection')
        report['route_observer_neutral']=True
        raw=np.fromfile(out/'observe/qkv.raw.f32',dtype='<f4').reshape(gold.shape);actual=np.fromfile(out/'observe/qkv.bf16.f32',dtype='<f4').reshape(gold.shape)
        report['pre_round']=ref.metrics(raw,gold,np);report['post_round']=ref.metrics(actual,rounded,np)
        bad=np.flatnonzero(actual.reshape(-1).view(np.uint32)!=rounded.reshape(-1).view(np.uint32));require(bad.size>0,'retained QKV rounding difference unexpectedly absent')
        selected=list(dict.fromkeys([int(bad[0]),int(np.argmax(np.abs(actual.astype(np.float64)-rounded.astype(np.float64))))]))
        (scratch/'selected.txt').write_text(''.join(str(i)+'\n'for i in selected));folder=out/'selected';folder.mkdir()
        command('selected',[*base,folder,'observe',scratch/'selected.txt'])
        require((folder/'qkv.raw.f32').read_bytes()==(out/'observe/qkv.raw.f32').read_bytes(),'selected trace changed native full projection')
        report['native_repeat_exact']=True;report['selected']=[]
        with(folder/'selected.csv').open()as f:rows=list(csv.DictReader(f))
        for which,index in enumerate(selected):
            token,row=divmod(index,8192);v=normalized[0,token];w=torch.from_numpy(weight[row].copy())
            products=(v*w).numpy().copy()
            require((folder/f'weight-{which}.f32').read_bytes()==weight[row].tobytes(),'native selected weight operand differs')
            require((folder/f'input-{which}.f32').read_bytes()==v.numpy().tobytes(),'native selected input operand differs')
            require((folder/f'products-{which}.f32').read_bytes()==products.tobytes(),'standalone native/original products differ')
            original=float(gold.reshape(-1)[index]);native_raw=float(raw.reshape(-1)[index])
            with(folder/f'trace-{which}.csv').open()as f:updates=list(csv.DictReader(f))
            first=next((r for r in updates if float.fromhex(r['fused_after'])!=float.fromhex(r['separate_from_same_before'])),None)
            report['selected'].append({'flat_index':index,'token':token,'output_row':row,'original_pre_round':original,'native_pre_round':native_raw,
                'pre_round_abs_error':abs(original-native_raw),'original_bf16':float(rounded.reshape(-1)[index]),'native_bf16':float(actual.reshape(-1)[index]),
                'torch_dot':float(torch.dot(v,w)),'torch_sum_products':float(torch.sum(v*w)),
                'f64_exact_products_fsum':math.fsum(float(a)*float(b)for a,b in zip(v.numpy(),weight[row])),
                'native_hypotheses':{k:float.fromhex(val)for k,val in rows[which].items()if k!='index'},
                'standalone_products_exact':True,'first_fused_vs_separate_update':first,
                'trace_scope':'Native eight-lane hypothesis recurrence, not observed Torch GEMM internal partial sums.'})
        report['hypotheses']=[]
        for name in ('lane8_fma','lane64_fma','lane64_separate','f64_products_sum'):
            values=np.fromfile(folder/(name+'.f32'),dtype='<f4').reshape(gold.shape)
            rounded_values=torch.from_numpy(values.copy()).bfloat16().float().numpy()
            report['hypotheses'].append({'name':name,'exact_native_raw_values':int(np.count_nonzero(values.view(np.uint32)==raw.view(np.uint32))),
                'exact_original_raw_values':int(np.count_nonzero(values.view(np.uint32)==gold.view(np.uint32))),
                'original_rounded_differences':int(np.count_nonzero(rounded_values.view(np.uint32)!=rounded.view(np.uint32))),
                'versus_original_raw':ref.metrics(values,gold,np),'versus_original_rounded':ref.metrics(rounded_values,rounded,np)})
        text=(out/'observe.log').read_text();match=re.search(r'qkv_calls=(\d+) qkv_used=(\d+) exact_operand_calls=(\d+) vec_calls=(\d+)',text);require(match is not None,'actual route receipt missing')
        calls,used,matched,vec=map(int,match.groups());require(calls>0 and used==matched==calls,'actual native QKV route/operand observation failed')
        report['route']={'qkv_calls':calls,'qkv_used':used,'exact_full_matrix_input_matches':matched,'other_or_fallback_vec_calls':vec,'all_qkv_multiplicands':'F32'}
        report['ggml_commit']=subprocess.check_output(['git','-C',ROOT/'ggml','rev-parse','HEAD'],text=True).strip()
        report['status']='compared';report['comparison_completed']=True;report['native_composed_qkv_payload_anchor_available']=False;save()
        require(report['post_round']['passed'],'QKV BF16 rounded projection retains numerical rejection')
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error)[:260]if isinstance(error,ValueError)else 'bounded QKV attribution failed';raise RuntimeError(report['error'])from None
    finally:save()
if __name__=='__main__':main()
