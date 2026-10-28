# Provider-only copy of worker_v1.py at3c3e4c8e. Inference functions unchanged.
# Only closure import and resources() admission differ; historical worker remains intact.
"""One variant per hosted CPU session, frozen original -> strict F32 -> Q4_K.

No original execution or original-ID/vector injection. All large files stay on
the hosted worker. Every numeric failure is retained, with quality separate.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(Path(__file__).parent)]
import contract_gha_v1 as closure
from contract_native_v1 import require,sha
from ci_process import run as run_child
from original_inputs_v1 import authenticate,expected_tokens
from providers_v1 import module,old_original,published_f32
import authored_v1 as authored
import compare_v1 as comparison

GATES=('CRISPEMBED_MOEDERN_PUBLIC_TEXT','CRISPEMBED_MOEDERN_PACKED','CRISPEMBED_MOEDERN_TEXT_COMPOSED',
       'CRISPEMBED_MOEDERN_TEXT_FRONTEND','CRISPEMBED_MOEDERN_COLBERT','CRISPEMBED_MOEDERN_FAMILY')
QUANT=('CRISPEMBED_MOEDERN_FAST','CRISPEMBED_MOEDERN_QUANT','CRISPEMBED_MOEDERN_Q4_K_PADDED')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n')


def command(args,log,env=None,*,allowed=(0,),timeout=2400,stdout_path=None):
    result=run_child(args,log,cwd=ROOT,env=env,timeout=timeout,stdout_path=stdout_path)
    result.update(log=Path(log).name,sha256=sha(log));write(Path(str(log)+'.process.json'),result)
    require(not result['timed_out']and result['exit']in allowed,'process/structural rejection in '+Path(log).name)
    return result


def environment(variant,quantized=False):
    env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')}
    env.update({g:'1'for g in GATES},OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2',
               CARGO_BUILD_JOBS='2',CRISPEMBED_OFFLINE='1')
    if variant!='1.5M'or quantized:env['CRISPEMBED_MOEDERN_FAST']='1'
    if quantized:env.update({g:'1'for g in QUANT})
    return env


def resources(scratch):
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('gha_provider_admission',ROOT/'tools/ci-heavy/moedern_scifact_gha_v1.py')
    provider=module_from_spec(spec);spec.loader.exec_module(provider)
    service=provider.admission(scratch,closure.git('rev-parse','HEAD').decode().strip())
    require(platform.system()=='Linux'and platform.machine()=='x86_64'and hasattr(os,'wait4'),'Linux x64 wait4 required')
    ram=int(next(r.split()[1]for r in Path('/proc/meminfo').read_text().splitlines()if r.startswith('MemAvailable:')))*1024
    disk=shutil.disk_usage(scratch).free
    require(ram>=14*2**30 and disk>=24*2**30 and os.cpu_count()>=2,'need14GiB available RAM/24GiB disk/two CPU before inputs/model')
    require(sys.version_info[:2]==(3,13)and sys.prefix!=sys.base_prefix,'isolated CPython3.13 required')
    expected={'pip':'26.2.2','numpy':'2.2.6','gguf':'0.17.1','PyYAML':'6.0.2','tqdm':'4.67.1'}
    owners={}
    for name,version in expected.items():
        d=importlib.metadata.distribution(name);p=Path(d._path)/'METADATA'
        require(d.version==version and p.resolve().is_relative_to(Path(sys.prefix).resolve()),'isolated package owner/version differs: '+name)
        owners[name]=dict(version=version,metadata_sha256=sha(p),inside_venv=True)
    return dict(available_ram_bytes=ram,free_disk_bytes=disk,threads=2,cpu_count=os.cpu_count(),packages=owners,gpu_execution=False,provider=service)


def build(rootbuild,tests,review,env):
    review.mkdir(parents=True,exist_ok=False)
    flags=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DBUILD_SHARED_LIBS=OFF',
           '-DCRISPEMBED_BUILD_SHARED=ON','-DCRISPEMBED_NATIVE=OFF','-DGGML_NATIVE=OFF','-DGGML_BLAS=OFF',
           '-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF','-DGGML_OPENCL=OFF','-DGGML_LLAMAFILE=ON']
    result={}
    result['root_configure']=command(['cmake','-S',ROOT,'-B',rootbuild,*flags],review/'root-configure.log',env)
    result['root_build']=command(['cmake','--build',rootbuild,'--parallel','2','--target','crispembed-shared','crispembed-cli','crispembed-server','crispembed-quantize'],review/'root-build.log',env)
    result['tests_configure']=command(['cmake','-S',ROOT/'tests/moedern-scifact','-B',tests,'-DCMAKE_BUILD_TYPE=Release','-DROOT_BUILD='+str(rootbuild)],review/'tests-configure.log',env)
    result['tests_build']=command(['cmake','--build',tests,'--parallel','2'],review/'tests-build.log',env)
    result['q4k_controls']=command([tests/'replay-controls'],review/'q4k-controls.log',env)
    cache=(rootbuild/'CMakeCache.txt').read_text()
    require('GGML_LLAMAFILE:BOOL=ON'in cache and'GGML_NATIVE:BOOL=OFF'in cache,'actual root arithmetic profile differs')
    cc=json.loads((rootbuild/'compile_commands.json').read_text())
    records=[r for r in cc if Path(r['file']).name in('moedern_text_engine.cpp','moedern_family_engine.cpp','moedern_text.cpp')]
    require(len(records)>=6 and all('-ffp-contract=off'in r['command']for r in records),'actual production arithmetic compilation differs')
    symbols=review/'root-symbols.txt';command(['nm','-C',rootbuild/'libcrispembed.so'],review/'symbols.log',env,stdout_path=symbols)
    strings=review/'root-strings.txt';command(['strings',rootbuild/'libcrispembed.so'],review/'strings.log',env,stdout_path=strings)
    require(all(s not in symbols.read_text()for s in('moedern_text_test::','moedern_text_native::test::'))and
            all(s not in strings.read_text()for s in('ffn.router_probabilities','Observer rejected:')),'MoE reference probes leaked into ordinary root library')
    result.update(root_library_sha256=sha(rootbuild/'libcrispembed.so'),root_profile='LLAMAFILE ON; CPU2; native/BLAS/CUDA/Metal OFF',
                  observer_exclusion=True,actual_compile_commands_sha256=sha(rootbuild/'compile_commands.json'))
    return result


def ref_controls(binary,out,env):
    import struct
    import diff_adapter_v1 as adapter
    out.mkdir(parents=True,exist_ok=False);records=[]
    for name,bad in(('same',False),('badrow',True)):
        folder=out/name;folder.mkdir();specs={}
        for side in('original','native'):
            row=[1.]+[0.]*127;values=row*3
            if bad and side=='native':values[-128:]=[0.,1.]+[0.]*126
            path=folder/(side+'.f32');path.write_bytes(struct.pack('<384f',*values))
            streams={k:dict(path=path,offset=0,bytes=1536,shape=[3,128])for k in('normalized_all','normalized_selected')}
            specs[side]=adapter.write(folder/(side+'.gguf'),streams)
        run=command([binary,folder/'original.gguf',folder/'native.gguf',folder/'ref.json'],folder/'ref.log',env,allowed=(1 if bad else 0,))
        measured=json.loads((folder/'ref.json').read_text())
        require(len(measured['stages'])==2 and all(r['elements']==384 and r['passed']==(not bad)and
                r['max_absolute']==(1 if bad else 0)and r['min_row_cosine']==(0 if bad else 1)for r in measured['stages']),
                'actual full-stream Ref positive/bad-row control differs')
        records.append(dict(control=name,process=run,adapters=specs))
    return records


def token_preflight(model,original,binary,out,env,quantized=False):
    out.mkdir(parents=True,exist_ok=False);raw=out/'texts.tsv'
    raw.write_text(''.join(r['key']+'\t'+r['role']+'\t'+(r['text'].encode().hex()or'-')+'\n'for r in original['rows']))
    result=command([binary,model,raw,out/'tokens.jsonl','q4k'if quantized else'f32'],out/'tokens.log',env,allowed=(0,1),timeout=1200)
    rows=[json.loads(line)for line in(out/'tokens.jsonl').read_text().splitlines()]
    require(len(rows)==5483,'complete token-only corpus journal required')
    bad=[]
    for target,row in zip(original['journal'],rows):
        if not(row.get('completed')is True and row['key']==target['key']and row['role']==target['role']and
               all(row.get(k)==v for k,v in expected_tokens(target).items())):bad.append(target['key'])
    return dict(completed=True,rows=5483,mismatches=len(bad),first_mismatches=bad[:16],passed=not bad and result['exit']==0,
                process=result,original_expansion_provenance=original['expansion_truth'],journal_sha256=sha(out/'tokens.jsonl'))


def main(args):
    args.out.mkdir(parents=True,exist_ok=False);args.scratch.mkdir(parents=True,exist_ok=False)
    review=args.out/'review';raw=args.out/'raw';review.mkdir();raw.mkdir()
    result=dict(schema='crispembed-moedern-scifact-native-q4k-v1',variant=args.variant,completed=False,
                source=args.reviewed_source,phase='source',default_enabled=False,publication_qualified=False,original_rerun=False)
    save=lambda:write(review/'native.json',result);save()
    try:
        cfg=closure.source(args.variant);require(closure.git('rev-parse','HEAD').decode().strip()==args.reviewed_source,'reviewed worker source differs')
        result['source_closure']=dict(base=cfg['base'],ggml=cfg['ggml'],contract_sha256=sha(closure.PATH),
                                      protected_files=len(cfg['protected_sha256']),owned_files=len(cfg['owned_sha256']),
                                      linked_src_tree=closure.git('rev-parse','HEAD:src').decode().strip())
        result['resources']=resources(args.scratch);result['phase']='authenticate-all-originals';save()
        original=authenticate(args.mount,args.variant,cfg)
        oldroot,oldreport,oldarrays=old_original(args.mount,args.variant,args.scratch)
        result['original']=dict(**original['pin'],rows=5483,all_raw_sha256_verified=True,data_sha256=original['report']['data']['all_rows_sha256'],
                                old_authored_report_sha256=sha(oldroot/'review/original.json'),old_query_expansion_tensors=4)
        save();env=environment(args.variant);buildroot=args.scratch/'root';tests=args.scratch/'tests'
        result['phase']='build-root-profile';save();result['build']=build(buildroot,tests,review/'build',env)
        result['ref_controls']=ref_controls(tests/'replay-diff',raw/'ref-controls',env)
        result['phase']='published-f32';save();model=args.scratch/'published-f32.gguf';result['model']=published_f32(args.variant,model);save()
        data=original['raw']/'data.json';rows=original['raw']/'rows.json'
        def preflight(label,path,active,quantized):
            result['phase']=label+'-token-only';save();tokens=token_preflight(path,original,tests/'replay-tokens',raw/(label+'-tokens'),active,quantized)
            result[label]=dict(tokens=tokens);save();require(tokens['passed'],'all5483 token-only preflight failed; embeddings forbidden')
        def full(label,path,active,quantized):
            require(result[label]['tokens']['passed'],'complete token-only preflight required before embeddings')
            result['phase']=label+'-full-native-capture';save();capture=raw/(label+'-capture')
            process=command([sys.executable,'-I',ROOT/'tools/moedern-scifact/capture_v1.py','--library',buildroot/'libcrispembed.so',
                             '--model',path,'--data',data,'--rows',rows,'--out',capture],review/(label+'-capture.log'),active,allowed=(0,1),timeout=3*3600)
            state=json.loads((capture/'capture.json').read_text());result[label].update(process=process,capture=state);save()
            require(state['completed']and state['rows']==5483 and state['failure_count']==0,'full actual raw-text C-ABI structural/typed rejection')
            result['phase']=label+'-all-vectors-eight-shards';save()
            measured=comparison.vectors(original,capture,tests/'replay-diff',raw/(label+'-vectors'),command,active)
            result[label]['vectors']=measured;save();result['phase']=label+'-full-rankings';save()
            rank=comparison.ranking(original,capture,raw/(label+'-ranking'),quantized=quantized)
            result[label].update(ranking=rank,strict_f32_numeric_passed=measured['strict_numeric_passed']and rank['strict_score_absolute_passed'],
                                 full_retrieval_gates_passed=rank['quality_passed']);save()
        preflight('f32',model,env,False)
        result['phase']='authored-f32';save()
        control,native_rows=authored.evaluate(model,oldreport,oldarrays,tests,raw/'authored-f32',command,env)
        result['authored_f32']=control;save();require(control['strict_numeric_passed'],'authored F32 rejected; full corpus/candidate forbidden')
        familyenv=dict(env,CRISPEMBED_MOEDERN_FAST='1')
        result['family_f32']=authored.family_control(model,oldreport,oldarrays,native_rows,tests,raw/'family-f32',command,familyenv)
        save();require(result['family_f32']['strict_numeric_passed'],'family FAST F32 rejected; Q4 forbidden')
        full('f32',model,env,False)
        require(result['f32']['strict_f32_numeric_passed']and result['f32']['full_retrieval_gates_passed'],'full strict F32 original control rejected; Q4 forbidden')
        result['phase']='convert-real-q4k';save();candidate=raw/'model-q4k.gguf'
        result['conversion_process']=command([sys.executable,'-I',ROOT/'tools/moedern-colbert/quantize_q4k_padded.py','--input',model,
                '--output',candidate,'--variant',args.variant,'--quantizer',buildroot/'crispembed-quantize','--report',review/'conversion.json'],review/'conversion.log',env)
        result['conversion']=json.loads((review/'conversion.json').read_text());save();qenv=environment(args.variant,True)
        preflight('q4k',candidate,qenv,True)
        result['phase']='authored-q4k-and-repeat';save()
        quant,repeat_rows=authored.evaluate(candidate,oldreport,oldarrays,tests,raw/'authored-q4k',command,qenv)
        repeat,repeated=authored.evaluate(candidate,oldreport,oldarrays,tests,raw/'authored-q4k-repeat',command,qenv)
        bit_identity=module('pinned_bits_control',ROOT/'tools/moedern-colbert/public_clients.py')
        require(repeat_rows==repeated and bit_identity.canonical_bits(repeat_rows)==bit_identity.canonical_bits(repeated),
                'authored Q4 repeat differs after F32 roundtrip')
        result['authored_q4k']=quant;result['authored_q4k_repeat_exact']=True
        result['recovery']=command([tests/'replay-recovery',candidate],review/'recovery.log',qenv);save()
        result['phase']='all-size-public-consumers';save()
        for label,path,active in(('f32',model,env),('q4k',candidate,qenv)):
            destination=raw/(label+'-public-clients')
            result[label+'_clients_process']=command([sys.executable,'-I',Path(__file__),'--clients','--variant',args.variant,
                        '--model',path,'--original',oldroot,'--build',buildroot,'--out',destination,
                        *(['--quantized']if label=='q4k'else[])],review/(label+'-clients.log'),active,timeout=3600)
            result[label+'_clients']=json.loads((destination/'review.json').read_text());save()
        full('q4k',candidate,qenv,True)
        result.update(completed=True,phase='diagnostic-complete',quantized_numeric_passed=result['q4k']['strict_f32_numeric_passed'],
                      full_retrieval_gates_passed=result['q4k']['full_retrieval_gates_passed'],publication_qualified=False)
        save();return result
    except Exception as error:
        result.update(failed_phase=result['phase'],phase='rejected',error=type(error).__name__+': '+str(error));save();raise


if __name__=='__main__':
    def terminate(_signal,_frame):raise RuntimeError('hosted worker termination requested')
    signal.signal(signal.SIGTERM,terminate)
    p=argparse.ArgumentParser();p.add_argument('--clients',action='store_true');p.add_argument('--quantized',action='store_true')
    p.add_argument('--variant',choices=('1.5M','27M','115M'),required=True);p.add_argument('--reviewed-source')
    for name in('mount','scratch','out','model','original','build'):p.add_argument('--'+name,type=Path)
    args=p.parse_args()
    if args.clients:
        require(all((args.model,args.original,args.build,args.out)),'client child arguments missing')
        client=module('pinned_all_size_clients',ROOT/'tools/moedern-colbert/public_clients.py')
        client.execute(args.variant,args.model,args.original,args.build,args.out,quantized=args.quantized)
    else:
        require(all((args.reviewed_source,args.mount,args.scratch,args.out)),'worker arguments missing')
        main(args)
