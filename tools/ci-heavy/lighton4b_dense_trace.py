#!/usr/bin/env python3
"""Exact retained gate projection only, with test-only native route/order traces."""
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess

ROOT=Path(__file__).resolve().parents[2]
UPSTREAM='132c800b63a888f9639dcdc3094b6b38ae0c538b'
GGML='3a27b6c665b191cb230be367db5c52d81dd3e2ac'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def execute():
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('CRISPEMBED_BUILD_MODE')!='portable':
        raise RuntimeError('hosted portable operator trace required')
    import lighton4b_gdn_reference as reference
    torch,np,gguf,original,Config,Cache,runtime=reference.load_original()
    out=Path(os.environ['HEAVY_OUT']);scratch=Path(os.environ['HEAVY_SCRATCH']);retained=Path(os.environ['RETAINED_UPSTREAM'])
    old=json.loads((retained/'upstream-receipt.json').read_text())
    frozen=json.loads((ROOT/'tests/results/qwen35-ocr/lightonocr3-4b-decoder-compose-attribution.json').read_text())['initial_receipt']
    report={'schema':'lighton4b-dense-trace/v1','status':'failed','upstream_run':41981848201,
        'decoder_repeated':False,'original_decoder_repeated':False,'checkpoint_payload_bytes':0,'quality_qualified':False,
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{p:sha(ROOT/p) for p in ('src/qwen35_linear.cpp','src/qwen35_math.h','tests/test_lighton4b_dense_trace.cpp',
            'tools/ci-heavy/lighton4b_dense_trace.py','tools/ci-heavy/lighton4b_gdn_reference.py','CMakeLists.txt')},
        'original_runtime':runtime,'torch_cpu_capability':torch.backends.cpu.get_cpu_capability(),
        'torch_build_config':torch.__config__.show(),'torch_parallel_info':torch.__config__.parallel_info()}
    def command(label,argv):
        with (out/(label+'.log')).open('w') as stream:
            subprocess.run([str(v) for v in argv],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    try:
        if old['source_commit']!=UPSTREAM:raise ValueError('retained upstream source differs')
        for path,digest in old['source_sha256'].items():
            if hashlib.sha256(subprocess.check_output(['git','show',UPSTREAM+':'+path],cwd=ROOT)).hexdigest()!=digest:
                raise ValueError('retained source digest differs')
        if subprocess.check_output(['git','-C',str(ROOT/'ggml'),'rev-parse','HEAD'],text=True).strip()!=GGML:
            raise ValueError('ggml source differs')
        report['ggml_commit']=GGML
        report['ggml_sources']={p:sha(ROOT/'ggml'/p) for p in ('src/ggml-cpu/llamafile/sgemm.cpp','src/ggml-cpu/vec.cpp',
            'src/ggml-cpu/simd-mappings.h','src/ggml-cpu/ggml-cpu.c')}
        config=Config(**json.loads((ROOT/'tests/fixtures/lightonocr3-4b-source/config.json').read_text())['text_config'])
        config._attn_implementation='eager';torch.manual_seed(195034)
        prototypes={False:original.Qwen3_5DecoderLayer(config,0).eval().float(),True:original.Qwen3_5DecoderLayer(config,3).eval().float()}
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
                raise ValueError('regenerated weight differs: '+name)
        report['exact_regenerated_weight_records']=len(weights)
        weight=prototypes[False].mlp.gate_proj.weight.detach()
        xraw=retained/'original/source_input.post_norm.f32';goldraw=retained/'original/source_input.gate.f32'
        nativeraw=retained/'native/source_input.source_post_norm.gate.f32'
        x=torch.from_numpy(np.fromfile(xraw,dtype='<f4').reshape(1,2,2560).copy())
        gold=np.fromfile(goldraw,dtype='<f4').reshape(1,2,9216);native_gold=np.fromfile(nativeraw,dtype='<f4').reshape(gold.shape)
        report['retained_records']={name:{'bytes':path.stat().st_size,'sha256':sha(path)} for name,path in
            [('post_norm',xraw),('original_gate',goldraw),('native_source_postnorm_gate',nativeraw)]}
        with torch.inference_mode():fresh=torch.nn.functional.linear(x,weight).numpy().copy()
        report['original_projection_reproduced_bit_exact']=fresh.tobytes()==gold.tobytes()
        if not report['original_projection_reproduced_bit_exact']:raise ValueError('original retained projection anchor changed')
        delta=np.abs(native_gold.astype(np.float64)-gold.astype(np.float64)).reshape(-1)
        unequal=np.flatnonzero(native_gold.reshape(-1).view(np.uint32)!=gold.reshape(-1).view(np.uint32))
        if not unequal.size:raise ValueError('retained projection unexpectedly exact')
        selected=[int(unequal[0]),int(delta.argmax())]
        report['selected_indices']={'first_row_major_difference':selected[0],'worst_absolute_difference':selected[1]}
        reference.write_gguf(scratch/'weights.gguf',{'gate.weight':weight.numpy()},gguf)
        reference.write_gguf(scratch/'inputs.gguf',{'input':x.numpy(),'selected_indices':np.array(selected,dtype=np.float32)},gguf)
        selected_data=[]
        for index in selected:
            row=index%9216;token=index//9216;w=weight[row];v=x[0,token]
            with torch.inference_mode():
                products=(w*v).numpy().copy();dot=float(torch.dot(w,v));separate=float(torch.sum(w*v))
            rounded_fsum=math.fsum(float(a) for a in products)
            exact_products_fsum=math.fsum(float(a)*float(b) for a,b in zip(w.numpy(),v.numpy()))
            selected_data.append({'index':index,'token':token,'row':row,'original_gemm':float(gold.reshape(-1)[index]),
                'native_project':float(native_gold.reshape(-1)[index]),'torch_dot':dot,'torch_sum_rounded_products':separate,
                'f64_products_fsum':exact_products_fsum,'rounded_f32_products_fsum':rounded_fsum})
            (out/('torch-products-'+str(len(selected_data)-1)+'.f32')).write_bytes(products.tobytes())
        report['selected_outputs']=selected_data
        del prototypes,weights;gc.collect()
        command('configure',['cmake','-S',ROOT,'-B',scratch/'build','-DCMAKE_BUILD_TYPE=Release','-DGGML_NATIVE=OFF',
            '-DCRISPEMBED_NATIVE=OFF','-DGGML_CUDA=OFF','-DGGML_VULKAN=OFF'])
        command('build',['cmake','--build',scratch/'build','--target','test-lighton4b-dense-trace','--parallel','2'])
        binary=scratch/'build/test-lighton4b-dense-trace';report['executable_sha256']=sha(binary)
        cache=(scratch/'build/CMakeCache.txt').read_text();report['build_options']=[line for line in cache.splitlines()
            if re.match(r'(GGML_(NATIVE|AVX|AVX2|AVX512|FMA|LLAMAFILE|BLAS)|CRISPEMBED_NATIVE):',line)]
        command('native',[binary,scratch/'weights.gguf',scratch/'inputs.gguf',out])
        report['native_stdout']=(out/'native.log').read_text().replace(str(ROOT),'[repository]').replace(str(scratch),'[scratch]')
        route=re.search(r'project_vec_calls=(\d+) project_llama_calls=(\d+) project_llama_used=(\d+) exact_operand_calls=(\d+)',report['native_stdout'])
        if not route:raise ValueError('native route receipt missing')
        vec,llama,used,matched=map(int,route.groups());report['observed_route']={'vec_calls':vec,'llama_calls':llama,'llama_used':used,'exact_operand_calls':matched}
        if not (used>0 and matched==llama and vec==0):raise ValueError('exact native llamafile route not observed; trace remains unproven')
        actual=np.fromfile(out/'project.f32',dtype='<f4').reshape(gold.shape)
        report['native_projection_reproduced_bit_exact']=actual.tobytes()==native_gold.tobytes()
        if not report['native_projection_reproduced_bit_exact']:raise ValueError('native retained projection anchor changed')
        report['native_vs_original']=reference.metrics(actual,gold,np)
        report['hypotheses']=[]
        for name in ('lane8_fma','lane8_separate','lane32_fma','sequential_fma','sequential_separate','f64_products_sum','rounded_f32_products_f64_sum','direct_vecdot'):
            values=np.fromfile(out/(name+'.f32'),dtype='<f4').reshape(gold.shape)
            report['hypotheses'].append({'name':name,'bit_exact_native_outputs':int(np.sum(values.view(np.uint32)==actual.view(np.uint32))),
                'bit_exact_original_outputs':int(np.sum(values.view(np.uint32)==gold.view(np.uint32))),
                'versus_native':reference.metrics(values,actual,np),'versus_original':reference.metrics(values,gold,np)})
        trace=[]
        for which,index in enumerate(selected):
            if (out/('torch-products-'+str(which)+'.f32')).read_bytes()!=(out/('products-'+str(which)+'.f32')).read_bytes():
                raise ValueError('standalone rounded products differ')
            with (out/('trace-'+str(which)+'.csv')).open() as stream:
                rows=list(csv.DictReader(stream))
            first=next((row for row in rows if struct.pack('<f',float.fromhex(row['fused_after']))!=struct.pack('<f',float.fromhex(row['separate_from_same_before']))),None)
            trace.append({'output_index':index,'standalone_products_bit_exact':True,'updates':len(rows),
                'first_fma_vs_separate_from_same_before':first,
                'scope':'Fixed2560-width native eight-lane recurrence reconstruction, not internal Torch GEMM partial sums.'})
        report['partial_traces']=trace
        dump=scratch/'disassembly.txt'
        libraries=list((scratch/'build').rglob('libggml-cpu.so'))
        native_object=libraries[0] if len(libraries)==1 else binary
        report['native_arithmetic_object_sha256']=sha(native_object)
        with dump.open('w') as stream:subprocess.run(['objdump','-d','--demangle',str(native_object)],stdout=stream,check=True)
        lines=dump.read_text().splitlines();snippets=[];active=False
        for line in lines:
            if line.endswith('>:'):active=('tinyBLAS<8' in line or 'ggml_vec_dot_f32' in line)
            if active and ('vfmadd' in line or 'vhadd' in line or 'vaddss' in line):snippets.append(line)
        (out/'native-arithmetic-disassembly.txt').write_text('\n'.join(snippets[:256])+'\n')
        report['arithmetic_instruction_records']=len(snippets)
        report['status']='operator_trace_complete'
    except Exception as error:
        report['error_type']=type(error).__name__;report['error']=str(error).replace(str(ROOT),'[repository]').replace(str(scratch),'[scratch]')
        raise
    finally:(out/'dense-trace-receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':execute()
