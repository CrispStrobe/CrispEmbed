#!/usr/bin/env python3
"""Hosted-only one-document actual-operation source-bound diagnostic controller; no submission."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools/moedern-first-row'),str(ROOT/'tools')]
from common import PATH,CID,atomic,module,require,sha,verify,hosted,authenticated_input
from ci_process import run as child


def main(args):
    cfg=verify(args.reviewed_source);provider=hosted(args.reviewed_source)
    import subprocess
    require(subprocess.check_output(['git','-C',str(ROOT/'ggml'),'rev-parse','HEAD'],text=True).strip()==cfg['ggml']and not subprocess.check_output(['git','-C',str(ROOT/'ggml'),'status','--porcelain','--untracked-files=no']),'actual clean backend checkout required')
    args.scratch.parent.mkdir(parents=True,exist_ok=True)
    ram=int(next(l.split()[1]for l in Path('/proc/meminfo').read_text().splitlines()if l.startswith('MemAvailable:')))*1024
    require(ram>=14*2**30 and shutil.disk_usage(args.scratch.parent).free>=24*2**30 and os.cpu_count()>=2,'14GiB available RAM/24GiB disk/two CPU before build/models')
    args.out.mkdir(parents=True,exist_ok=False);args.scratch.mkdir(parents=True,exist_ok=False)
    review=args.out/'review';review.mkdir();raw=args.out/'raw';raw.mkdir()
    report=dict(source=args.reviewed_source,completed=False,phase='input-authentication',commands={},publication_qualified=False,original_full_corpus_recaptured=False,original_rerun=False,provider=provider)
    def save():atomic(review/'hosted.json',report)
    env={k:v for k,v in os.environ.items()if not k.startswith('CRISPEMBED_')}
    env['CRISPEMBED_CAPTURE_SOURCE']=args.reviewed_source
    env.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',HF_HUB_DISABLE_TELEMETRY='1')
    def command(name,argv,active=env,timeout=1200,allowed=(0,)):
        log=review/(name+'.log');result=child(argv,log,cwd=ROOT,env=active,timeout=timeout)
        report['commands'][name]=dict(result,log=log.name,sha256=sha(log));save()
        require(not result['timed_out']and result['exit']in allowed,'structural/process rejection: '+name)
    save()
    try:
        case=authenticated_input(args.input,cfg);packet={'cases':[case]}
        report['input_packet_sha256']=sha(args.input/'input-packet.json');save()
        build=args.scratch/'root';tests=args.scratch/'tests'
        flags=['-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DBUILD_SHARED_LIBS=OFF','-DCRISPEMBED_NATIVE=OFF',
            '-DGGML_NATIVE=OFF','-DGGML_BACKEND_DL=OFF','-DGGML_BLAS=OFF','-DGGML_CUDA=OFF','-DGGML_METAL=OFF','-DGGML_VULKAN=OFF','-DGGML_OPENCL=OFF','-DGGML_LLAMAFILE=ON']
        command('root-configure',['cmake','-S',ROOT,'-B',build,*flags]);command('root-ggml-build',['cmake','--build',build,'--target','ggml','ggml-cpu','--parallel','2'])
        command('probe-configure',['cmake','-S',ROOT/'tests/moedern-first-row','-B',tests,'-DCMAKE_BUILD_TYPE=Release','-DROOT_BUILD='+str(build)])
        command('probe-build',['cmake','--build',tests,'--parallel','2'])
        cache=(build/'CMakeCache.txt').read_text();require('GGML_LLAMAFILE:BOOL=ON'in cache and'GGML_NATIVE:BOOL=OFF'in cache,'root backend profile differs')
        root_commands=json.loads((build/'compile_commands.json').read_text());probe_commands=json.loads((tests/'compile_commands.json').read_text())
        expected={flag for r in root_commands if Path(r['file']).name=='moedern_text_engine.cpp'for flag in r['command'].split()if flag.startswith(('-mavx','-mfma','-mf16c'))}
        actual=[r for r in probe_commands if Path(r['file']).name=='moedern_text_engine.cpp'];require(len(actual)==2 and all('-ffp-contract=off'in r['command']and
            {f for f in r['command'].split()if f.startswith(('-mavx','-mfma','-mf16c'))}==expected for r in actual),'scalar observer/plain ISA or contraction flags differ from configured root')
        atomic(review/'build-profile.json',dict(root_cache_sha256=sha(build/'CMakeCache.txt'),root_configured_engine_commands=[r for r in root_commands if Path(r['file']).name=='moedern_text_engine.cpp'],
            actual_probe_engine_commands=actual,ggml_archives={n:sha(build/'ggml/src'/n)for n in ('libggml.a','libggml-cpu.a','libggml-base.a')},
            root_production_library_compiled=False,root_backend_compiled=True))
        import subprocess
        symbols=subprocess.check_output(['nm','-C',str(tests/'libfirst-row-plain.a')],text=True)
        strings=subprocess.check_output(['strings',str(tests/'libfirst-row-plain.a')],text=True)
        instrumented=subprocess.check_output(['nm','-C',str(tests/'libfirst-row-observed.a')],text=True)
        require(all(n not in symbols for n in('moedern_text_test::','moedern_text_native::test::'))and
                'ffn.router_probabilities'not in strings and 'moedern_text_test::encode_observed'in instrumented,'ordinary archive exclusion/observed positive control failed')
        report['ordinary_excludes_observers']=True;save()
        sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'))
        reference=module(ROOT/'tools/moedern-native-v2/replay.py','three_doc_actual_ref_controls')
        report['actual_ref_controls']=reference.comparator_controls(tests/'first-row-diff',raw/'ref-controls');save()
        report['phase']='ordinary-original-barrier';save()
        command('original-ordinary',[sys.executable,'-I',ROOT/'tools/moedern-first-row/original.py','--reviewed-source',args.reviewed_source,
            '--phase','ordinary','--input',args.input,'--out',raw/'ordinary','--checkpoint',args.scratch/'checkpoint','--diff',tests/'first-row-diff',
            '--barrier-root',raw],timeout=1800)
        original=json.loads((raw/'ordinary/original.json').read_text());require(original['completed']is True and original['strict_anchor_passed']is True,'ordinary original barrier rejected')
        report.update(original_rerun=True,original_forward_completed=1);save()
        model=args.scratch/'model.gguf'
        sys.path[:0]=[str(ROOT/'tools/moedern-scifact')]
        converter=module(ROOT/'tools/moedern-colbert/quantize_q4k_padded.py','frozen_published_f32_metadata')
        provider=module(ROOT/'tools/moedern-first-row/prepare.py','one_row_span_transfer')
        pin=converter.publication('1.5M');require(pin['sha256']==cfg['native_model']['sha256']and pin['bytes']==cfg['native_model']['bytes'],'native93 model identity differs');provider.transfer(pin['url'],model,pin);report['model']=pin;save()
        text_packet=raw/'three-doc.tsv';text_packet.write_text(''.join(c['id']+'\tdocument\t'+c['text'].encode().hex()+'\n'for c in packet['cases']))
        Path(str(text_packet)+'.pairs').write_text('')
        active=dict(env,CRISPEMBED_MOEDERN_PACKED='1',CRISPEMBED_MOEDERN_TEXT_FRONTEND='1',CRISPEMBED_MOEDERN_COLBERT='1',
                    CRISPEMBED_MOEDERN_PUBLIC_TEXT='1',CRISPEMBED_MOEDERN_FAMILY='1',CRISPEMBED_OFFLINE='1')
        command('native-plain',[tests/'first-row-plain-probe',model,'',text_packet,raw/'plain'],active,timeout=1200)
        counts=json.loads((raw/'plain/counts.json').read_text())
        require(counts==dict(encode_attempts=9,full_computations=6,frontend_rejections=3,pre_math_observer_abort=0,init_rejections=3),'plain computation budget differs')
        controls=json.loads((raw/'plain'/CID/'controls.json').read_text());tokens=json.loads((raw/'plain'/CID/'tokens.json').read_text())
        require(all(controls[k]is True for k in ('repeat','gate_recovery','role_bounds_recovery','context_ownership')),'ordinary native controls absent')
        require(all(tokens[k]==case[k]for k in ('ids','supplied_mask','effective_mask','scoring_mask','expansion_positions')),'ordinary native IDs/masks differ')
        for kind in ('all','selected'):
            pin=case['anchors']['native'][kind];path=raw/'plain'/CID/(kind+'.f32')
            require(path.stat().st_size==pin['bytes']and sha(path)==pin['sha256'],'ordinary native93 exact final anchor rejected; no observer permitted')
        atomic(raw/'barriers.json',dict(completed=True,original_strict_anchor_passed=True,native93_plain_anchor_exact=True,native_plain_full_computations=6,
            original_report_sha256=sha(raw/'ordinary/original.json'),plain_report_sha256=sha(raw/'plain'/CID/'controls.json')))
        report.update(phase='both-ordinary-barriers-closed-before-any-observer',barriers_sha256=sha(raw/'barriers.json'));save()
        command('original-observe',[sys.executable,'-I',ROOT/'tools/moedern-first-row/original.py','--reviewed-source',args.reviewed_source,
            '--phase','observe','--input',args.input,'--out',raw/'original','--checkpoint',args.scratch/'checkpoint','--diff',tests/'first-row-diff',
            '--barrier-root',raw],timeout=1800)
        observed=json.loads((raw/'original/original.json').read_text())
        require(observed['completed']is True and observed['original_forward_completed']==4,'original observed/repeated coverage incomplete')
        report['original_forward_completed']=5;save()
        command('native-observed',[tests/'first-row-observed-probe',model,'',text_packet,raw/'observed'],active,timeout=1200)
        observed_counts=json.loads((raw/'observed/counts.json').read_text())
        require(observed_counts==dict(encode_attempts=12,full_computations=8,frontend_rejections=3,pre_math_observer_abort=1,init_rejections=3),'observed computation budget differs')
        report['native_counts']={k:counts[k]+observed_counts[k]for k in counts};save()
        report['phase']='exact-native93-anchor-before-attribution';save()
        command('compare',[sys.executable,'-I',ROOT/'tools/moedern-first-row/compare.py','--reviewed-source',args.reviewed_source,'--input',args.input,
            '--original',raw/'original','--plain',raw/'plain','--observed',raw/'observed','--diff',tests/'first-row-diff','--out',raw/'comparisons'],timeout=1200,allowed=(0,1))
        report['diagnostic']=json.loads((raw/'comparisons/diagnostic.json').read_text());require(report['diagnostic']['completed']is True,'bounded diagnostics incomplete')
        report.update(completed=True,phase='diagnostic-complete',diagnostic_report_sha256=sha(raw/'comparisons/diagnostic.json'),numerical_qualified=False);save()
        return 1 if any(r['status']=='numeric_rejection'for r in report['diagnostic']['cases'])else 0
    except Exception as error:report['error']=type(error).__name__+': '+str(error);save();raise
    finally:
        original_counts=[]
        for phase in ('ordinary','original'):
            path=raw/phase/'original.json'
            if path.is_file():
                row=json.loads(path.read_text());original_counts.append(dict(phase=phase,attempts=row['original_forward_attempts'],completed=row['original_forward_completed']))
        report['actual_original_calls']=original_counts
        report['original_rerun']=any(r['attempts']for r in original_counts)
        report['original_forward_attempts']=sum(r['attempts']for r in original_counts)
        report['original_forward_completed']=sum(r['completed']for r in original_counts)
        files={}
        for path in raw.rglob('*'):
            if path.is_file()and path.name in('ref.json','ref.log','anchor-ref.json','anchor-ref.log','diff.json','diff.log','controls.json','tokens.json','counts.json','original.json','barriers.json','diagnostic.json','stages.tsv'):
                require(path.stat().st_size<=2*2**20,'small evidence exceeds declared bound')
                target=review/'evidence'/path.relative_to(raw);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
                files[str(target.relative_to(review))]=dict(bytes=target.stat().st_size,sha256=sha(target))
        atomic(review/'evidence-files.json',files);report['small_evidence_manifest_sha256']=sha(review/'evidence-files.json');save()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reviewed-source',required=True)
    for n in ('input','out','scratch'):p.add_argument('--'+n,type=Path,required=True)
    raise SystemExit(main(p.parse_args()))
