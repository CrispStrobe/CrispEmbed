#!/usr/bin/env python3
"""Hosted first-attention operator diagnosis; historical gold/guards stay frozen."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'tools/moedern-sdpa'), str(ROOT/'tools/moedern-colbert')]
from sdpa_contract import (PROTOCOL, exact_control, historical_contract, indexed, checked_arrays,
                           kernel_evidence, metrics, protocol, replace_call, sha, source_guard, tensor_layout)
from diff_adapter import collect, write_pair
from observer_continuity import prepare as prepare_old, validate_prior, capture_leaves
from contract import read_contract, verify_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def api(path):
    return json.loads(subprocess.check_output(['gh','api','repos/CrispStrobe/CrispEmbed/'+path], timeout=120))


def authenticated_observer(cfg, scratch):
    run = api('actions/runs/'+str(cfg['observer_run']))
    if (run['head_sha'] != cfg['observer_source'] or run['run_attempt'] != 1 or
            run['status'] != 'completed' or run['conclusion'] != 'failure'):
        raise ValueError('Retained observer failure service lineage differs')
    service = {'run':cfg['observer_run'], 'source':run['head_sha'], 'conclusion':run['conclusion'], 'artifacts':{}}
    for kind in ('review','raw'):
        pin = cfg['observer_'+kind+'_artifact']
        actual = api('actions/artifacts/'+str(pin['id']))
        if (actual['workflow_run']['id'] != cfg['observer_run'] or
                actual['workflow_run']['head_sha'] != cfg['observer_source'] or actual['expired'] or
                actual['name'] != pin['name'] or actual['size_in_bytes'] != pin['bytes'] or
                actual['digest'] != 'sha256:'+pin['sha256']):
            raise ValueError('Retained observer artifact identity differs')
        archive = scratch/(kind+'.zip')
        with archive.open('xb') as out:
            subprocess.run(['gh','api','repos/CrispStrobe/CrispEmbed/actions/artifacts/'+str(pin['id'])+'/zip'],
                           stdout=out, check=True, timeout=300)
        if archive.stat().st_size != pin['bytes'] or sha(archive.read_bytes()) != pin['sha256']:
            raise ValueError('Retained observer ZIP SHA/extent differs')
        with zipfile.ZipFile(archive) as z:
            names=z.namelist()
            if (len(names)>1024 or len(names)!=len(set(names)) or sum(i.file_size for i in z.infolist())>128*1024**2 or
                    any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names) or
                    any((i.external_attr >> 16)&0o170000 == 0o120000 for i in z.infolist())):
                raise ValueError('Unsafe retained observer ZIP')
            z.extractall(scratch/kind)
        service['artifacts'][kind]={k:actual[k] for k in ('id','name','size_in_bytes','digest','expires_at')}
    return service


def retained(scratch):
    cfg=protocol()
    oldroot,old=validate_prior(scratch/'old')
    root=scratch/'observer/raw/moedern-original'
    data=(root/'review/original.json').read_bytes()
    if (sha(data)!=cfg['observer_receipt_sha256'] or
            data!=(scratch/'observer/review/moedern-original/review/original.json').read_bytes()):
        raise ValueError('Retained observer receipt SHA differs')
    rejected=json.loads((root/'review/rejection.json').read_text())
    if rejected!={'completed':False,'error':cfg['observer_rejection']}:
        raise ValueError('Initial observer rejection must remain unchanged')
    observed=json.loads(data);proof=historical_contract(old,observed,cfg)
    checked_arrays(root,observed)
    declaration=read_contract()['variants']['1.5M']['original']
    for name,identity in declaration['files'].items():
        verify_file(oldroot/'checkpoint'/name,identity)
        verify_file(root/'checkpoint'/name,identity)
    for name,expected in observed['driver_sources'].items():
        actual=subprocess.check_output(['git','show',cfg['observer_source']+':'+name],cwd=ROOT)
        if sha(actual)!=expected:
            raise ValueError('Retained observer source hash differs')
    return oldroot,root,old,observed,proof


def prepare(out,scratch):
    cfg=protocol();measured=source_guard()
    if measured['head']!=os.environ.get('GITHUB_SHA'):
        raise ValueError('Current service/diagnostic source differs')
    if scratch.exists():raise ValueError('Fresh task scratch required')
    scratch.mkdir(parents=True)
    service={'old':prepare_old(scratch/'old')}
    (scratch/'observer').mkdir()
    service['observer']=authenticated_observer(cfg,scratch/'observer')
    _,_,_,_,proof=retained(scratch)
    record={'schema':cfg['schema'],'phase':'prepared-authenticated-inputs','qualification':False,
            'source':measured,'protocol_sha256':sha(PROTOCOL.read_bytes()),'service':service,'historical':proof,
            'framework_imported':False,'native_model_accessed':False}
    write(out/'review/prepared.json',record)
    return record


def run_command(args,log,timeout=180):
    with log.open('w') as stream:
        subprocess.run([str(a) for a in args],stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)


def compare_pair(binary,folder,name,left,right,dims):
    folder.mkdir(parents=True)
    records=[]
    for key,data in (('original',left),('replay',right)):
        path=folder/(key+'.f32');path.write_bytes(data)
        records.append({'path':path,'shape':dims,'sha256':sha(data),'label':name+'/'+key+'.f32'})
    adapters=write_pair(folder/'original.gguf',folder/'replay.gguf',
        [{'name':name,'original':records[0],'native':records[1]}])
    independent={name:metrics(left,right,dims)}
    env={k:v for k,v in os.environ.items() if k not in ('CRISPEMBED_DIFF_COS_THRESHOLD','CRISPEMBED_DIFF_DUMP_DIR')}
    with (folder/'diff.log').open('w') as stream:
        result=subprocess.run([str(binary),str(folder/'original.gguf'),str(folder/'replay.gguf'),str(folder/'diff.json')],
            stdout=stream,stderr=subprocess.STDOUT,env=env,timeout=30,check=False)
    actual=collect(result.returncode,folder/'diff.json',independent)
    if not actual['all_metrics_agree']:
        raise ValueError('Independent metrics differ from actual crispembed-diff')
    return {'bit_exact':left==right,'metrics':independent[name],'actual_ref':actual,'adapters':adapters}


def execute(out,scratch):
    cfg=protocol();measured=source_guard()
    prepared=json.loads((out/'review/prepared.json').read_text())
    if prepared['source']!=measured or prepared['protocol_sha256']!=sha(PROTOCOL.read_bytes()):
        raise ValueError('Prepared source/protocol changed before framework/model access')
    oldroot,newroot,old,new,proof=retained(scratch)
    checkpoint=oldroot/'checkpoint';declaration=read_contract()['variants']['1.5M']['original']
    for name,identity in declaration['files'].items():verify_file(checkpoint/name,identity)
    # Only now may original packages or authenticated checkpoint weights load.
    expected={'torch':'2.8.0','transformers':'5.3.0','sentence-transformers':'6.1.0','numpy':'2.2.6'}
    if any(importlib.metadata.version(k).split('+')[0]!=v for k,v in expected.items()):
        raise ValueError('Original package profile changed')
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend,sdpa_kernel
    from safetensors.torch import load_file
    if torch.version.cuda is not None:
        raise ValueError('Exact CPU-only Torch wheel required')
    st_root=Path(importlib.util.find_spec('sentence_transformers').origin).parent.parent
    for name,identity in read_contract()['reference_runtime']['sentence_transformers']['files'].items():
        verify_file(st_root/name,identity)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.set_float32_matmul_precision('highest')
    path=checkpoint/'modeling_modernbert_hash_moe_v10.py'
    module_spec=importlib.util.spec_from_file_location('moedern_original_first_attention',path)
    original_source=importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name]=original_source;module_spec.loader.exec_module(original_source)
    model_config=read_contract()['variants']['1.5M']['config']
    module=original_source.HybridAttention(hidden=model_config['hidden_size'],num_heads=model_config['num_attention_heads'],
        num_kv_heads=model_config['num_key_value_heads'],max_position_embeddings=model_config['max_position_embeddings'],
        rope_theta=model_config['rope_theta'],allow_mask_attention=model_config['allow_mask_attention'],
        attention_dropout=model_config['attention_dropout'],norm_eps=model_config['norm_eps'],
        inference_window_half=model_config.get('inference_window_half',256)).eval()
    weights=load_file(str(checkpoint/'model.safetensors'),device='cpu')
    prefix='layers.0.attention.';state={k.removeprefix(prefix):v for k,v in weights.items() if k.startswith(prefix)}
    module.load_state_dict(state,strict=True);del weights
    def tensor_bytes(value):
        if value.device.type!='cpu':raise ValueError('CPU-only operator replay required')
        array=value.detach().contiguous().numpy()
        dtype={torch.float32:'<f4',torch.int64:'<i8',torch.bool:'u1'}.get(value.dtype)
        if dtype is None:raise ValueError('Unexpected original tensor dtype')
        if not np.isfinite(array).all():raise ValueError('Nonfinite original operator tensor')
        return array.astype(dtype,copy=False).tobytes()
    for name,value in module.state_dict().items():
        if value.dtype!=torch.float32 or sha(tensor_bytes(value))!=old['loaded_tensor_sha256']['model.safetensors'][prefix+name]:
            raise ValueError('Actual loaded first-attention parameter changed')
    build=scratch/'diff-build'
    run_command(['cmake','-S',ROOT/'tests/moedern-sdpa','-B',build,'-DCMAKE_BUILD_TYPE=Release',
                 '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON'],out/'review/configure.log')
    run_command(['cmake','--build',build,'--parallel','2'],out/'review/build.log')
    binary=build/'moedern-sdpa-diff'
    shutil_copy=__import__('shutil').copyfile
    shutil_copy(build/'compile_commands.json',out/'review/compile_commands.json')
    (out/'raw/tools').mkdir(parents=True)
    shutil_copy(binary,out/'raw/tools/moedern-sdpa-diff')
    # Actual compiled equal/non-equal row controls precede retained operators.
    a=np.asarray([[1.,0.],[0.,1.],[1.,0.]],dtype='<f4').tobytes()
    z=np.asarray([[1.,0.],[0.,1.],[0.,1.]],dtype='<f4').tobytes()
    controls={}
    for name,data,code in [('same',a,0),('changed_row',z,1)]:
        row=compare_pair(binary,out/'raw/comparator-controls'/name,'control',a,data,[3,2])
        if row['actual_ref']['exit_code']!=code:raise ValueError('Actual Ref numeric exit control failed')
        controls[name]=row
    hardware=[]
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if ':' in line:
            k,v=line.split(':',1)
            if k.strip() in ('vendor_id','model name','cpu family','model','stepping','flags'):
                item=(k.strip(),v.strip())
                if item not in hardware:hardware.append(item)
    report={'schema':cfg['schema'],'completed':False,'qualification':False,'new_gold':False,'native_outputs':False,
        'old_continuity_accepted':False,'source':measured,'historical':proof,'cases':[],
        'environment':{'packages':{k:importlib.metadata.version(k) for k in expected},
            'safetensors':importlib.metadata.version('safetensors'),'python':sys.version.split()[0],'hardware':hardware,
            'architecture':platform.machine(),'torch_git':torch.version.git_version,
            'torch_config':torch.__config__.show(),'parallel_config':torch.__config__.parallel_info(),
            'cpu_capability':torch.backends.cpu.get_cpu_capability(),
            'mkldnn_enabled':torch.backends.mkldnn.enabled,'intra_op_threads':torch.get_num_threads(),
            'inter_op_threads':torch.get_num_interop_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'matmul_precision':torch.get_float32_matmul_precision(),
            'thread_environment':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',
                'ATEN_CPU_CAPABILITY','MKL_CBWR','MKL_DEBUG_CPU_TYPE','DNNL_MAX_CPU_ISA','ONEDNN_MAX_CPU_ISA')}},
        'comparator_sha256':sha(binary.read_bytes()),'comparator_controls':controls,'limits':cfg['limits']}
    def save():write(out/'review/diagnostic.json',report)
    save()
    old_rows,new_rows=indexed(old),indexed(new)
    def array(case,stage):
        row=old_rows[case,stage];dtype={'f32':'<f4','i64':'<i8','u8':'u1'}[row['dtype']]
        value=torch.from_numpy(np.fromfile(oldroot/row['path'],dtype=dtype).reshape(row['shape']).copy())
        return value.bool() if row['dtype']=='u8' else value
    def sdp_flags():
        return {k:getattr(torch.backends.cuda,k+'_sdp_enabled')() for k in ('flash','math','mem_efficient','cudnn')}
    with torch.inference_mode():
        for case,pin in cfg['cases'].items():
            values=array(case,'backbone.layers.0.attention.input#0.args.0')
            mask=array(case,'backbone.layers.0.attention.input#0.kwargs.attention_mask')
            op=F.scaled_dot_product_attention;captured={};seen={};handles=[]
            plain=module(values,attention_mask=mask)
            for name,child in [('norm',module.norm),('Q_proj',module.Q_proj),('K_proj',module.K_proj),('V_proj',module.V_proj)]:
                def capture(_m,_a,value,name=name):
                    stage='backbone.layers.0.attention.'+name+'.output#0'
                    if stage in seen:raise ValueError('Repeated first-attention module observation')
                    seen[stage]=sha(tensor_bytes(value))
                handles.append(child.register_forward_hook(capture))
            def observed(*args,**kwargs):
                if captured:raise ValueError('First-attention module must invoke SDPA once')
                if (len(args)!=3 or set(kwargs)!={'attn_mask','dropout_p','is_causal'} or
                        kwargs['dropout_p']!=0. or kwargs['is_causal'] is not False):
                    raise ValueError('Actual original SDPA scalar/argument contract changed')
                for label,tensor in capture_leaves('actual.sdpa.0.input',{'args':args,'kwargs':kwargs},
                                                  lambda x:isinstance(x,torch.Tensor)):
                    row=pin['common'][label]
                    if list(tensor.shape)!=row['shape'] or sha(tensor_bytes(tensor))!=row['sha256']:
                        raise ValueError('Source-reconstructed Q/K/V/mask differs from both historical inputs: '+label)
                result=op(*args,**kwargs)
                captured.update(args=args,kwargs=kwargs,result=result)
                return result
            try:
                with replace_call(F,'scaled_dot_product_attention',observed):
                    traced=module(values,attention_mask=mask)
            finally:
                for handle in handles:handle.remove()
            repeated=module(values,attention_mask=mask)
            exact_control(tensor_bytes(plain),tensor_bytes(traced),tensor_bytes(repeated))
            for label,digest in seen.items():
                if digest!=pin['common'][label]['sha256']:raise ValueError('Original pre-SDPA norm/projection differs')
            if len(seen)!=4 or not captured:raise ValueError('Actual original source observation incomplete')
            exact_control(tensor_bytes(values),(oldroot/old_rows[case,'backbone.layers.0.attention.input#0.args.0']['path']).read_bytes())
            args,kwargs=captured['args'],captured['kwargs']
            tensor_objects=(*args,kwargs['attn_mask'])
            original_inputs=[(id(t),tensor_layout(t),sha(tensor_bytes(t))) for t in tensor_objects]
            row={'id':case,'tokens':values.shape[1],'module_observer_repeat_bit_exact':True,
                'shared_inputs_bit_exact':True,'layout':{str(i):tensor_layout(t) for i,t in enumerate(args)},
                'mask_layout':tensor_layout(kwargs['attn_mask']),'scalar_kwargs':cfg['scalar_kwargs'],'arms':{},'comparisons':{}}
            report['cases'].append(row);save()
            for backend in cfg['backends']:
                flags=sdp_flags()
                manager=contextlib.nullcontext() if backend=='default' else sdpa_kernel(
                    backends=[SDPBackend.MATH if backend=='math' else SDPBackend.FLASH_ATTENTION])
                arm={'status':'started','same_actual_tensor_objects':True,'backend_flags_before':flags}
                row['arms'][backend]=arm;save()
                try:
                    with manager:
                        arm['backend_flags_selected']=sdp_flags()
                        first=op(*args,**kwargs);repeat=op(*args,**kwargs)
                        # Observe a feature Mapping without modifying the actual arguments.
                        observations=[]
                        for label,t in capture_leaves('input',{'args':args,'kwargs':kwargs},lambda x:isinstance(x,torch.Tensor)):
                            observations.append((label,sha(tensor_bytes(t))))
                        observed_result=op(*args,**kwargs)
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
                            profiled=op(*args,**kwargs)
                    raw=tensor_bytes(first)
                    exact_control(raw,tensor_bytes(repeat),tensor_bytes(observed_result),tensor_bytes(profiled))
                    names=kernel_evidence(backend,[x.key for x in profile.key_averages()])
                    if backend=='default':exact_control(raw,tensor_bytes(captured['result']))
                    for generation,root,rows in [('old',oldroot,old_rows),('observer',newroot,new_rows)]:
                        frozen=rows[case,'actual.sdpa.0.output']
                        measured=compare_pair(binary,out/'raw'/case/backend/generation,case.replace('-','_')+'.'+backend.replace('-','_')+'.'+generation,
                            (root/frozen['path']).read_bytes(),raw,frozen['shape'])
                        arm[generation]=measured
                    arm.update(status='completed-operator-control',profiler_kernels=names,
                        ordinary_repeat_observer_profiled_bit_exact=True,output_sha256=sha(raw),
                        observation_sha256=sha(json.dumps(observations,sort_keys=True).encode()))
                except RuntimeError as exc:
                    if backend=='default':raise
                    arm.update(status='rejected-no-fallback',error=type(exc).__name__+': '+str(exc))
                finally:
                    if sdp_flags()!=flags:raise ValueError('Backend context state leaked after operator arm')
                    arm['backend_state_restored']=True
                    if [(id(t),tensor_layout(t),sha(tensor_bytes(t))) for t in tensor_objects]!=original_inputs:
                        raise ValueError('Actual SDPA tensor objects/layout/values changed across backend controls')
                    arm['actual_input_objects_layout_and_bits_unchanged']=True
                    save()
            for generation,root,rows in [('old',oldroot,old_rows),('observer',newroot,new_rows)]:
                frozen=rows[case,'backbone.layers.0.attention.output#0']
                row['comparisons'][generation+'_module_output']=compare_pair(binary,out/'raw'/case/'module'/generation,
                    case.replace('-','_')+'.module.'+generation,(root/frozen['path']).read_bytes(),tensor_bytes(plain),frozen['shape'])
            left=old_rows[case,'actual.sdpa.0.output'];right=new_rows[case,'actual.sdpa.0.output']
            row['comparisons']['historical_sdpa_difference']=compare_pair(binary,out/'raw'/case/'historical','historical',
                (oldroot/left['path']).read_bytes(),(newroot/right['path']).read_bytes(),left['shape'])
            save()
    report.update(completed=True,status='completed bounded operator diagnosis; original continuity remains rejected')
    save()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','run'),required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--scratch',type=Path,required=True)
    args=parser.parse_args()
    if os.environ.get('GITHUB_ACTIONS')!='true':parser.error('All artifacts/frameworks/builds/operators belong on hosted CPU CI')
    if args.phase=='prepare' and args.output.exists():parser.error('Fresh task-owned output required')
    if args.phase=='run' and (args.output/'review/diagnostic.json').exists():parser.error('Preserve prior diagnostic; fresh execution required')
    args.output.mkdir(parents=True,exist_ok=True);(args.output/'review').mkdir(exist_ok=True)
    try:
        if args.phase=='prepare':prepare(args.output,args.scratch)
        else:execute(args.output,args.scratch)
    except Exception as exc:
        write(args.output/'review'/('rejection-'+args.phase+'.json'),{'qualification':False,'new_gold':False,
            'native_outputs':False,'phase':args.phase,'error':type(exc).__name__+': '+str(exc)})
        raise
    finally:
        write(args.output/'review'/('files-'+args.phase+'.json'),{
            str(p.relative_to(args.output)):{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}
            for p in args.output.rglob('*') if p.is_file() and p.name!=('files-'+args.phase+'.json')})


if __name__=='__main__':main()
