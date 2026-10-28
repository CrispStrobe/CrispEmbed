#!/usr/bin/env python3
"""Hosted CPU-only diagnostic: immutable original -> F32 control -> real Q4_K.

The caller supplies a read-only frozen original capture directory and the exact
published self-contained F32 package. This function performs no original rerun.
Neither component parity nor the owned panel qualifies retrieval publication.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import struct
import subprocess
import sys
import time
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/moedern-colbert'))
import q4k_contract as closure
import quantize_q4k_padded as quantizer
sys.path.insert(0,str(ROOT/'tools/moedern-native-v2'))
import replay as reference
from diff_adapter import write_pair, write_gguf

GATES = ('CRISPEMBED_MOEDERN_PUBLIC_TEXT','CRISPEMBED_MOEDERN_PACKED',
         'CRISPEMBED_MOEDERN_TEXT_COMPOSED','CRISPEMBED_MOEDERN_TEXT_FRONTEND',
         'CRISPEMBED_MOEDERN_COLBERT','CRISPEMBED_MOEDERN_FAMILY')
EXTRA = ('CRISPEMBED_MOEDERN_FAST','CRISPEMBED_MOEDERN_QUANT','CRISPEMBED_MOEDERN_Q4_K_PADDED')
require = closure.require


def hosted():
    kaggle = (Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir() and
              os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Batch','Interactive'))
    require(os.environ.get('GITHUB_ACTIONS') == 'true' or kaggle, 'hosted execution required')


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def command(args, log, env=None, allowed=(0,), timeout=1800):
    start = time.monotonic()
    timing=log.with_name(log.name+'.time')
    require(Path('/usr/bin/time').is_file(),'GNU time required for per-process peak RSS')
    with log.open('w') as err:
        run = subprocess.run(['/usr/bin/time','-f','maximum_rss_kib=%M','-o',str(timing),*map(str,args)],cwd=ROOT,env=env,text=True,
                             stdout=subprocess.PIPE,stderr=err,timeout=timeout)
    require(run.returncode in allowed, log.name+' rejected with exit '+str(run.returncode))
    rss=re.search(r'^maximum_rss_kib=([0-9]+)$',timing.read_text(),re.M)
    require(rss is not None,'per-process peak RSS missing')
    return run.stdout, {'exit':run.returncode,'seconds':time.monotonic()-start,'maximum_rss_kib':int(rss[1])}


def original_inputs(root, variant, cfg):
    """Authenticate report and every selected array before packages/model access."""
    root = Path(root)
    report_path = root/'review/original.json'
    pin = cfg['originals'][variant]
    require(report_path.is_file() and not report_path.is_symlink() and
            quantizer.sha_file(report_path) == pin['report_sha256'], 'frozen original report differs')
    report = json.loads(report_path.read_text())
    require(report['completed'] is True and report['variant'] == variant and
            report['revision'] == quantizer.PROFILES[variant]['revision'], 'original profile differs')
    panel = json.loads((ROOT/'tests/fixtures/moedern-colbert/panel-v1.json').read_text())
    require([{k:c[k] for k in ('id','role','text')} for c in report['cases']] == panel['cases'] and
            len(report['scores']) == 3, 'frozen raw-text panel differs')
    originals = {}
    for row in report['stage_arrays']:
        if row['stage'] not in ('ordinary.features.token_embeddings','ordinary.selected'): continue
        rel = PurePosixPath(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in row['path'], 'unsafe original array path')
        path = root/rel
        require(path.is_file() and not any(p.is_symlink() for p in [path,*path.parents]) and
                path.stat().st_size == row['bytes'] and quantizer.sha_file(path) == row['sha256'] and
                len(row['shape']) == 2 and row['shape'][1] == 128 and 1 <= row['shape'][0] <= 512 and
                row['bytes'] == math.prod(row['shape'])*4, 'frozen array identity/shape differs')
        key = (row['case'],row['stage']); require(key not in originals, 'duplicate original vector')
        originals[key] = dict(path=path,shape=row['shape'],sha256=row['sha256'])
    require(set(originals) == {(c['id'],s) for c in report['cases'] for s in
            ('ordinary.features.token_embeddings','ordinary.selected')}, 'twenty original vectors required')
    return report, originals


def stream_file(url, destination, pin):
    """Only hosted callers; exact length and SHA, no signed URLs in receipts."""
    hosted(); destination=Path(destination); destination.parent.mkdir(parents=True,exist_ok=True)
    require(not destination.exists() and 0 < pin['bytes'] <= 512*2**20, 'fresh bounded transfer required')
    h=hashlib.sha256(); received=0
    try:
        with urllib.request.urlopen(url,timeout=90) as response, destination.open('xb') as output:
            require(response.status==200 and int(response.headers['Content-Length'])==pin['bytes'], 'declared payload length differs')
            while block:=response.read(1<<20):
                received+=len(block);require(received<=pin['bytes'],'payload exceeds declared length');h.update(block);output.write(block)
    except Exception:
        raise ValueError('authenticated transfer failed; partial task-owned file retained') from None
    require(received==pin['bytes'] and h.hexdigest()==pin['sha256'],'download bytes/SHA differ')


def prepare(variant, inputs, cfg):
    """Exact historical GHA or explicit Kaggle version; never latest fallback."""
    hosted(); inputs=Path(inputs); inputs.mkdir(parents=True,exist_ok=False)
    original=inputs/'original'; original.mkdir(); pin=cfg['originals'][variant]
    write(inputs/'source-service.json',{'variant':variant,'state':'authenticating_frozen_original',
          'completed':False,'pins':pin,'all20_arrays_verified':False})
    if variant=='1.5M':
        spec=importlib.util.spec_from_file_location('old_public_download',ROOT/'tools/ci-heavy/moedern_public_consumers.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        z,review_service=old.download(pin['run'],pin['source'],pin['review_artifact'],inputs,2**20)
        data=z.read('moedern-original/review/original.json')
        require(closure.sha(data)==pin['report_sha256'],'frozen GHA report differs')
        (original/'review').mkdir();(original/'review/original.json').write_bytes(data)
        report=json.loads(data)
        z,raw_service=old.download(pin['run'],pin['source'],pin['raw_artifact'],inputs,32*2**20)
        for row in report['stage_arrays']:
            if row['stage'] not in ('ordinary.features.token_embeddings','ordinary.selected'):continue
            rel=PurePosixPath(row['path']);require(not rel.is_absolute() and '..' not in rel.parts,'unsafe original member')
            path=original/rel;path.parent.mkdir(parents=True,exist_ok=True);data=z.read('moedern-original/'+row['path'])
            require(len(data)==row['bytes'] and closure.sha(data)==row['sha256'],'original raw member differs');path.write_bytes(data)
        proof={'provider':'GitHub Actions','run':pin['run'],'source':pin['source'],'review':review_service,'raw':raw_service}
    else:
        # Private account reference comes only from hosted credentials. Every
        # page explicitly uses version_label; kernels_output(ref+'/N') is unsafe.
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi();api.authenticate();owner=os.environ['KAGGLE_ACCOUNT']
        Get=KaggleApi.kernels_pull.__globals__['ApiGetKernelRequest']
        List=KaggleApi.kernels_output.__globals__['ApiListKernelSessionOutputRequest']
        files={};token=None
        with api.build_kaggle_client() as client:
            request=Get();request.user_name=owner;request.kernel_slug=pin['kernel_slug'];request.version_label=pin['version_label']
            meta=json.loads(str(client.kernels.kernels_api_client.get_kernel(request)))
            source=meta['blob']['source'];require(closure.sha(source.encode())==pin['script_sha256'] and pin['source'] in source,'historical submitted source differs')
            while True:
                request=List();request.user_name=owner;request.kernel_slug=pin['kernel_slug'];request.version_label=pin['version_label'];request.page_size=100
                if token:request.page_token=token
                page=client.kernels.kernels_api_client.list_kernel_session_output(request)
                for item in page.files or []:
                    require(item.file_name not in files,'duplicate historical output member');files[item.file_name]=item.url
                token=page.next_page_token
                if not token:break
        member='moedern-original/review/original.json'
        require(member in files,'historical report member missing')
        stream_file(files[member],original/'review/original.json',{'bytes':pin['report_bytes'],'sha256':pin['report_sha256']})
        report=json.loads((original/'review/original.json').read_text())
        for row in report['stage_arrays']:
            if row['stage'] not in ('ordinary.features.token_embeddings','ordinary.selected'):continue
            member='moedern-original/'+row['path'];require(member in files and row['bytes']<=512*128*4,'bounded original array missing')
            rel=PurePosixPath(row['path']);require(not rel.is_absolute() and '..' not in rel.parts,'unsafe original array')
            stream_file(files[member],original/rel,row)
        proof={k:pin[k] for k in ('kernel_slug','version_label','source','script_sha256','report_sha256')};proof['provider']='Kaggle explicit version'
    original_inputs(original,variant,cfg)  # all20 identities before model access
    proof.update(completed=False,all20_arrays_verified=True,state='original_verified_before_model')
    write(inputs/'source-service.json',proof)
    model=inputs/'published-f32.gguf';publication=quantizer.publication(variant)
    stream_file(publication['url'],model,publication)
    proof.update(completed=True,state='published_f32_verified',f32_model_sha256=publication['sha256'])
    write(inputs/'source-service.json',proof)
    return original, model, proof


def build_only(build, out):
    hosted(); cfg = closure.source(); out=Path(out); out.mkdir(parents=True,exist_ok=True); build=Path(build)
    proof = {'source':closure.git('rev-parse','HEAD').decode().strip(),'base':cfg['base'],'ggml':cfg['ggml'],
             'compiled':False,'inference':False,'publication_qualified':False}
    for name,args in [('configure',['cmake','-S',ROOT/'tests/moedern-q4k','-B',build,'-DCMAKE_BUILD_TYPE=Release']),
                      ('build',['cmake','--build',build,'--parallel','2'])]:
        _,proof[name] = command(args,out/(name+'.log'),timeout=1800)
    _,proof['authored_controls'] = command([build/'moedern-q4k-controls'],out/'authored-controls.log')
    symbols = subprocess.check_output(['nm','-C',str(build/'libmoedern-q4k.so')],text=True)
    require('moedern_text_test::' not in symbols and 'moedern_text_native::test::' not in symbols and
            'std::function<' not in symbols, 'test observer callback leaked into production library')
    strings=subprocess.check_output(['strings',str(build/'libmoedern-q4k.so')],text=True)
    require(all(s not in strings for s in ('embedding.lookup','embedding.projection','attn.probabilities',
                                         'ffn.router_probabilities','Observer rejected:')),'observer stage labels leaked into production library')
    (out/'ordinary-symbols.txt').write_text(symbols)
    cache = (build/'CMakeCache.txt').read_text()
    require('GGML_LLAMAFILE:BOOL=ON' in cache and 'GGML_NATIVE:BOOL=OFF' in cache, 'actual build profile differs')
    fixture=out/'ref-controls';fixture.mkdir()
    for name,values in [('positive',[1.,0.,1.,0.]),('badrow',[1.,0.,0.,1.])]:
        path=fixture/(name+'.f32');path.write_bytes(struct.pack('<4f',*values))
        write_gguf(fixture/(name+'.gguf'),[dict(name='probe',path=path,shape=[2,2],sha256=quantizer.sha_file(path))])
    proof['ref_controls']={}
    for name,code in [('positive',0),('badrow',1)]:
        _,control=command([build/'moedern-q4k-diff',fixture/'positive.gguf',fixture/(name+'.gguf'),fixture/(name+'.json')],fixture/(name+'.log'),allowed=(code,))
        measured=json.loads((fixture/(name+'.json')).read_text());row=measured['stages'][0]
        require(len(measured['stages'])==1 and row['elements']==4 and row['passed']==(code==0) and
                row['max_absolute']==(0 if code==0 else 1) and row['min_row_cosine']==(1 if code==0 else 0),
                'actual Ref positive/bad-row control differs')
        proof['ref_controls'][name]=control
    proof['compiled'] = True; write(out/'build.json',proof); return proof


def run_panel(original_dir, f32_model, variant, build, out):
    """Return complete diagnostic dict; numerical Q4 failures retained, not waived."""
    hosted(); cfg = closure.source(); out=Path(out); build=Path(build); out.mkdir(parents=True,exist_ok=True)
    result = dict(schema='crispembed-moedern-q4k-padded-panel-v1',variant=variant,completed=False,
                  source=closure.git('rev-parse','HEAD').decode().strip(),default_enabled=False,
                  publication_qualified=False,retrieval_quality=cfg['retrieval_quality'],failures=[])
    raw = out/'raw'; raw.mkdir(); review = out/'review'; review.mkdir()
    def save(): write(review/'q4k.json',result)
    save()
    try:
        original, arrays = original_inputs(original_dir,variant,cfg)
        result['original_report_sha256'] = cfg['originals'][variant]['report_sha256']
        pin = quantizer.publication(variant)
        require(Path(f32_model).stat().st_size == pin['bytes'] and quantizer.sha_file(f32_model) == pin['sha256'],
                'published self-contained F32 model differs')
        for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='2'
        import numpy as np
        result['build'] = build_only(build,review)
        text = {c['id']:c['text'] for c in original['cases']}; hx=lambda s:s.encode().hex() or '-'
        cases,pairs = raw/'cases.tsv',raw/'pairs.tsv'
        cases.write_text(''.join(c['id']+'\t'+c['role']+'\t'+hx(c['text'])+'\n' for c in original['cases']))
        pairs.write_text(''.join(s['query']+'|'+s['document']+'\t'+hx(text[s['query']])+'\t'+hx(text[s['document']])+'\n' for s in original['scores']))
        env={k:v for k,v in os.environ.items() if not k.startswith('CRISPEMBED_')}
        env.update({g:'1' for g in GATES},OMP_NUM_THREADS='2')
        all_outputs={}

        def evaluate(mode, model, active):
            folder=raw/mode; folder.mkdir(); stdout,timing=command([build/'moedern-q4k-consumer',model,cases,pairs],review/(mode+'.log'),active)
            (folder/'consumer.jsonl').write_text(stdout)
            outputs=[json.loads(line) for line in stdout.splitlines() if line.strip()]
            cs=[r for r in outputs if r['kind']=='case']; ss=[r for r in outputs if r['kind']=='score']
            require([r['id'] for r in cs]==[r['id'] for r in original['cases']] and len(ss)==3 and len(outputs)==13,'exact ten-case/three-score output required')
            measurements={}; specs=[]; typed={}
            for target,native in zip(original['cases'],cs):
                cid=target['id']; supplied=list(map(int,target['supplied_attention_mask']))
                expect=dict(ids=target['token_ids'],supplied_mask=supplied,
                            effective_mask=list(map(int,target['effective_attention_mask'])),
                            scoring_mask=list(map(int,target['scoring_mask'])),
                            expansion_positions=[1-x for x in supplied] if target['role']=='query' else [])
                require(all(native[k]==v for k,v in expect.items()) and native['role']==target['role'],'IDs/masks/role differ: '+cid)
                typed[cid]=True
                for kind,stage in [('all','ordinary.features.token_embeddings'),('selected','ordinary.selected')]:
                    name=cid.replace('-','_')+'.'+kind; left=arrays[cid,stage]
                    a=np.fromfile(left['path'],dtype='<f4').reshape(left['shape']); b=np.asarray(native[kind+'_vectors'],dtype='<f4')
                    require(b.size==a.size,'vector extent differs: '+name); b=b.reshape(a.shape)
                    path=folder/(name+'.f32'); path.write_bytes(b.tobytes())
                    measurements[name]=reference.compare(a,b)
                    specs.append({'name':name,'original':left,'native':dict(path=path,shape=left['shape'],sha256=quantizer.sha_file(path))})
            adapters=write_pair(folder/'original.gguf',folder/'native.gguf',specs)
            ref=reference.actual_ref(build/'moedern-q4k-diff',folder/'original.gguf',folder/'native.gguf',review/(mode+'-ref.json'),review/(mode+'-ref.log'),measurements)
            require(ref['all_metrics_agree'],'actual Ref/independent NumPy metrics disagree')
            byid={r['id']:r for r in cs}; scores=[]
            for target,native in zip(original['scores'],ss):
                require(native['id']==target['query']+'|'+target['document'],'score order/identity differs')
                q=np.asarray(byid[target['query']]['selected_vectors'],dtype=np.float64).reshape(-1,128)
                d=np.asarray(byid[target['document']]['selected_vectors'],dtype=np.float64).reshape(-1,128)
                value=float(native['score']); independent=float((q@d.T).max(axis=1).sum())
                require(math.isfinite(value) and abs(value-independent)<=.002,'native MaxSim/independent selected vectors differ')
                scores.append(dict(id=native['id'],native=value,original=target['original_maxsim'],
                                   independent=independent,max_absolute=abs(value-target['original_maxsim']),
                                   passed=abs(value-target['original_maxsim'])<=.002))
            all_outputs[mode]=outputs
            record=dict(timing=timing,typed=typed,vectors=measurements,actual_ref=ref,adapters=adapters,scores=scores,
                        passed=all(r['passed'] for r in measurements.values()) and all(r['passed'] for r in scores))
            result[mode]=record; save(); return record

        # Strict F32 host control precedes candidate conversion and inference.
        control_env=dict(env)
        if variant!='1.5M': control_env['CRISPEMBED_MOEDERN_FAST']='1'
        control=evaluate('f32',f32_model,control_env); require(control['passed'],'F32 control violates original gates')
        # The 1.5M quantized route uses the family graph; test its F32 arithmetic
        # from IDs produced by the real raw-text frontend, never original ID injection.
        token_cases=raw/'native-derived-tokens.tsv'
        token_cases.write_text(''.join(r['id']+'\t'+r['role']+'\t'+','.join(map(str,r['ids']))+'\t'+','.join(map(str,r['supplied_mask']))+'\n' for r in all_outputs['f32'] if r['kind']=='case'))
        fast_env=dict(env,CRISPEMBED_MOEDERN_FAST='1'); fast_dir=raw/'family-f32';fast_dir.mkdir()
        _,result['family_f32_control']=command([build/'moedern-q4k-family-control',f32_model,token_cases,fast_dir,2],review/'family-f32.log',fast_env)
        checks={}; specs=[]
        for target in original['cases']:
            cid=target['id']; left=arrays[cid,'ordinary.features.token_embeddings']; right=fast_dir/(cid+'.f32')
            a=np.fromfile(left['path'],dtype='<f4').reshape(left['shape']);b=np.fromfile(right,dtype='<f4').reshape(a.shape)
            name=cid.replace('-','_');checks[name]=reference.compare(a,b)
            specs.append(dict(name=name,original=left,native=dict(path=right,shape=left['shape'],sha256=quantizer.sha_file(right))))
        write_pair(fast_dir/'original.gguf',fast_dir/'native.gguf',specs)
        proof=reference.actual_ref(build/'moedern-q4k-diff',fast_dir/'original.gguf',fast_dir/'native.gguf',review/'family-f32-ref.json',review/'family-f32-ref.log',checks)
        result['family_f32_control'].update(checks=checks,actual_ref=proof)
        require(proof['all_metrics_agree'] and all(v['passed'] for v in checks.values()),'FAST family F32 control violates original gates')
        candidate=raw/'model-q4-k.gguf'
        result['conversion']=quantizer.quantize(f32_model,candidate,variant,build/'moedern-q4k-quantize');write(review/'conversion.json',result['conversion']);save()
        quant_env=dict(env,**{g:'1' for g in EXTRA})
        result['gate_off_rejections']={}
        for gate in EXTRA:
            for value in ('0',''):
                off=dict(quant_env);off[gate]=value
                _,run=command([build/'moedern-q4k-consumer',candidate,cases,pairs],review/('off-'+gate+'-'+str(len(value))+'.log'),off,allowed=(1,))
                result['gate_off_rejections'][gate+'='+value]=run
        candidate_result=evaluate('q4k',candidate,quant_env)
        evaluate('q4k-repeat',candidate,quant_env)
        require(all_outputs['q4k']==all_outputs['q4k-repeat'],'candidate repeat not bit-exact after float32 roundtrip')
        _,result['context_recovery']=command([build/'moedern-q4k-recovery',candidate],review/'context-recovery.log',quant_env)
        result.update(completed=True,structural_passed=True,quantized_numeric_passed=candidate_result['passed'],
                      status='diagnostic_only' if candidate_result['passed'] else 'numeric_rejection',
                      retrieval_quality_qualified=False,publication_qualified=False)
        save();return result
    except Exception as error:
        result.update(status='structural_rejection',error=type(error).__name__+': '+str(error));save();raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--build-only',action='store_true');p.add_argument('--fetch-and-run',action='store_true');p.add_argument('--variant',choices=list(quantizer.PROFILES))
    p.add_argument('--original-dir',type=Path);p.add_argument('--f32-model',type=Path);a=p.parse_args()
    if a.build_only: build_only(a.build,a.out)
    elif a.fetch_and_run:
        p.error('variant required') if not a.variant else None
        hosted();cfg=closure.source()
        try: original,model,proof=prepare(a.variant,a.out/'authenticated-inputs',cfg)
        except Exception as error:
            # SDK exceptions may contain private access URLs. Retain the phase,
            # public immutable pins and error class without echoing those URLs.
            write(a.out/'prepare-rejection.json',{'schema':'crispembed-moedern-q4k-input-rejection-v1',
                  'variant':a.variant,'error_type':type(error).__name__,'completed':False,
                  'source':closure.git('rev-parse','HEAD').decode().strip()})
            raise ValueError('authenticated input preparation rejected; partial public-pin receipt retained') from None
        result=run_panel(original,model,a.variant,a.build,a.out/'panel');write(a.out/'source-service.json',proof)
    else:
        p.error('variant, original-dir and f32-model required') if not all((a.variant,a.original_dir,a.f32_model)) else None
        run_panel(a.original_dir,a.f32_model,a.variant,a.build,a.out)
