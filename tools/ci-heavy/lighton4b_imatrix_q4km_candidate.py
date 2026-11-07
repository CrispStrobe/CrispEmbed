#!/usr/bin/env python3
"""Kaggle-only external-text weighted Q4 calibration; never instantiate a reference model."""
import hashlib, importlib.util, importlib.metadata, json, os, re, shutil, signal, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import lighton4b_imatrix_adapter as adapter
import lighton4b_imatrix_q4km_recipe as profile
from gguf_file_type_canonicalize import parse, span, sha
from lighton4b_q6_recipe import plan, validate_source_inventory

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def unique_capture(mount, case, digest):
    # Read only the two predeclared capture files; never enumerate another case's data.
    choices = [p for p in mount.rglob(case + '/capture.json') if p.is_file()]
    adapter.require(len(choices) == 1 and sha(choices[0]) == digest, 'unique authenticated calibration mount required')
    return choices[0]

def audit_baseline(candidate, baseline_rows):
    g = parse(candidate); expected = {x['name']: x for x in plan()['all_tensors']}
    adapter.require(set(g['tensors']) == set(expected), 'candidate723 inventory differs')
    counts = {}; rows = {}; changed = []
    with candidate.open('rb') as stream:
        for name, tensor in g['tensors'].items():
            wanted = expected[name]; kind = 0 if wanted['type'] == 0 else 8 if name == 'l.embed_tokens.weight' else 12
            shape = wanted['source_shape']; shape = [shape[0], 1536] if name == 'v.patch_embed.proj.weight' else shape
            adapter.require(tensor['type'] == kind and tensor['stored_shape'] == list(reversed(shape)), 'candidate type/shape differs: ' + name)
            counts[kind] = counts.get(kind, 0) + 1
            rows[name] = span(stream, tensor['data_offset'], tensor['bytes'])
            if baseline_rows is not None:
                if kind in (0, 8): adapter.require(rows[name] == baseline_rows[name], 'protected tensor changed: ' + name)
                elif rows[name] != baseline_rows[name]: changed.append(name)
    adapter.require(counts == {0: 474, 8: 1, 12: 248}, 'exact plain Q4 types required')
    if baseline_rows is not None:
        m = g['metadata']
        adapter.require(m['crispembed.quantization.imatrix_expected_entries']['value'] == 248, 'strict248 imatrix metadata absent')
        adapter.require(len(changed) > 0, 'weighted payloads unexpectedly unchanged')
    return {'tensor_payload_sha256': rows, 'types': counts, 'changed_q4_names': changed, 'sha256': sha(candidate), 'bytes': candidate.stat().st_size}

def audit_q4km(candidate, baseline_rows):
    p=profile.plan();g=parse(candidate,{0:(1,4),12:(256,144),14:(256,210)});expected={x['name']:x for x in p['all_tensors']};adapter.require(set(g['tensors'])==set(expected),'candidate723 inventory differs');counts={};rows={};changed=[]
    with candidate.open('rb')as stream:
        for name,t in g['tensors'].items():
            row=expected[name];shape=row['source_shape'];shape=[shape[0],1536]if name=='v.patch_embed.proj.weight'else shape
            adapter.require(t['type']==row['type']and t['stored_shape']==list(reversed(shape)),'closed type/shape differs: '+name);counts[t['type']]=counts.get(t['type'],0)+1;rows[name]=span(stream,t['data_offset'],t['bytes'])
            if t['type']==0:adapter.require(rows[name]==baseline_rows[name],'protected F32 payload changed: '+name)
            elif rows[name]!=baseline_rows[name]:changed.append(name)
    adapter.require(counts=={0:474,12:216,14:33},'exact474/216/33 type counts required')
    m=g['metadata'];strings={'qwen35_ocr.mixed_quant_recipe':profile.RECIPE,'qwen35_ocr.q6_names_sha256':p['q6_names_sha256'],'qwen35_ocr.quant_source_f16_sha256':profile.SOURCE_F16_SHA,'qwen35_ocr.quant_coordinate_profile':'hf-grouped-v1','crispembed.quantization.imatrix_source_sha256':profile.IMATRIX_SHA,'crispembed.quantization.imatrix_profile':adapter.PROFILE}
    for key,value in strings.items():adapter.require(m[key]['kind']==8 and m[key]['value']==value,'typed profile metadata differs: '+key)
    for key,value in [('general.file_type',15),('qwen35_ocr.q6_tensor_count',33),('crispembed.quantization.imatrix_expected_entries',248),('crispembed.quantization.imatrix_applied_entries',248)]:adapter.require(m[key]['kind']==4 and m[key]['value']==value,'actual typed count differs: '+key)
    return {'tensor_payload_sha256':rows,'types':counts,'changed_quantized_names':changed,'sha256':sha(candidate),'bytes':candidate.stat().st_size,'q6_names_sha256':p['q6_names_sha256'],'all474_f32_exact':True,'actual_imatrix_applied_entries':248}

def bundle_runtime(build,out):
    import stat
    bundle=out/'runtime';binaries=bundle/'bin';libs=bundle/'lib';binaries.mkdir(parents=True);libs.mkdir();manifest={}
    for source in build.rglob('libggml*.so*'):
        if source.is_file():
            target=libs/source.name
            if target.exists():adapter.require(sha(target)==sha(source),'ambiguous runtime library');continue
            shutil.copyfile(source,target);manifest[str(target.relative_to(bundle))]={'sha256':sha(target),'bytes':target.stat().st_size}
    for name in ('lighton4b-native-document','lighton4b-native-case-preflight','test-lighton4b-stream-head-benchmark'):
        target=binaries/name;shutil.copyfile(build/name,target);target.chmod(0o755);manifest['bin/'+name]={'sha256':sha(target),'bytes':target.stat().st_size}
        env=dict(os.environ,LD_LIBRARY_PATH=str(libs));ldd=subprocess.check_output(['ldd',str(target)],env=env,text=True);(out/(name+'-ldd.log')).write_text(ldd)
        adapter.require('not found'not in ldd,'runtime library unresolved')
        for line in ldd.splitlines():
            if 'libggml'in line:adapter.require(str(libs)+'/'in line,'runtime uses unbundled ggml')
        test=subprocess.run([str(target)],env=env,capture_output=True,text=True,timeout=30);adapter.require(test.returncode==2,'bundled native argument guard differs')
    (out/'runtime-bundle.json').write_text(json.dumps({'schema':'lighton4b-native-runtime-bundle/v1','files':manifest,'source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()},indent=2,sort_keys=True)+'\n')
    return bundle

def main():
    require = adapter.require
    require(Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir() and not list(Path('/dev').glob('nvidia[0-9]*')), 'actual Kaggle CPU environment required')
    require(os.environ.get('CRISPEMBED_BUILD_MODE') == 'portable', 'portable build required')
    out = Path(os.environ['HEAVY_OUT']); scratch = Path(os.environ['HEAVY_SCRATCH']) / 'weighted-q4km'; scratch.mkdir()
    protocol_path = ROOT / 'docs/lighton4b-imatrix-q4km-candidate-protocol-v1.json'; p = json.loads(protocol_path.read_text())
    common = load('weighted_common', ROOT / 'tools/ci-heavy/lighton4b_native_document.py')
    cases = load('weighted_cases', ROOT / 'tools/ci-heavy/lighton4b_frozen_case.py')
    vision = load('weighted_vision', ROOT / 'tools/ci-heavy/lighton4b_rotary_q8_smoke.py')
    r = {'schema': p['schema'], 'status': 'failed', 'source_sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'protocol_sha256': sha(protocol_path), 'execution_host': 'kaggle-cpu', 'f32_model_instantiated': False, 'weight_quantization_executed': False, 'quantization_attempted': False, 'reference_model_forward_executed': False, 'heldout_read': False, 'quality_qualified': False, 'cases': {}, 'limitations': p['limitations']}
    active = None; started = time.monotonic()
    def save(): (out / 'q4km-candidate-first-score.json').write_text(json.dumps(r, indent=2, sort_keys=True) + '\n')
    def terminate(*args):
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try: active.wait(timeout=15)
            except subprocess.TimeoutExpired: os.killpg(active.pid, signal.SIGKILL); active.wait()
        raise TimeoutError('bounded weighted-Q4 deadline')
    def run(label, args, env=None, timeout=2700):
        nonlocal active
        r['stage'] = label; print('PHASE '+label,flush=True); save()
        with (out / (label + '.log')).open('w') as log:
            active = subprocess.Popen(list(map(str, args)), cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try: code = active.wait(timeout=timeout)
            except subprocess.TimeoutExpired: terminate()
            active = None
        require(code == 0, 'bounded child failed: ' + label)
    def capacity(initial=False):
        m = common.memory(); disk = shutil.disk_usage(scratch).free
        require(m['available_bytes'] >= 8*2**30 and (m['cgroup_limit_bytes'] is None or m['cgroup_limit_bytes']-m['cgroup_current_bytes'] >= 2*2**30), 'RAM/cgroup headroom unavailable')
        if initial: require(disk >= 18*2**30, '18GiB free disk required before staging')
        return {'memory': m, 'free_disk_bytes': disk, 'load_average': os.getloadavg()}
    def download(pin, path):
        r['stage']='download-'+path.name;print('PHASE '+r['stage'],flush=True);save()
        require(not path.exists(), 'fresh authenticated output required'); h = hashlib.sha256(); n = 0
        url = 'https://huggingface.co/' + pin['repository'] + '/resolve/' + pin['revision'] + '/' + pin['file']
        with urllib.request.urlopen(url, timeout=120) as src, path.open('xb') as dst:
            while block := src.read(4*2**20):
                n += len(block); require(n <= pin['bytes'], 'transfer exceeds declared bytes'); h.update(block); dst.write(block)
        r.setdefault('downloads',{})[path.name]={'bytes':n,'sha256':h.hexdigest(),'expected_bytes':pin['bytes'],'expected_sha256':pin['sha256']};save()
        require(n == pin['bytes'] and h.hexdigest() == pin['sha256'], 'immutable transfer differs')
    try:
        import numpy as np
        import PIL
        import gguf
        from PIL import Image
        require(np.__version__ == '2.3.5' and PIL.__version__ == '11.3.0' and importlib.metadata.version('gguf') == '0.19.0', 'frozen input serializer versions differ')
        first_cpu=Path('/proc/cpuinfo').read_text().split('\n\n',1)[0];cpu=dict(line.split(':',1)for line in first_cpu.splitlines()if ':'in line);r['cpu_profile']={key:cpu.get(key,'').strip()for key in('vendor_id','model name','flags')};r['cpu_profile'].update(logical_count=os.cpu_count(),affinity_cpus=len(os.sched_getaffinity(0)),cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip()if Path('/sys/fs/cgroup/cpu.max').exists()else None)
        r['preflight'] = capacity(True); signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate); signal.signal(signal.SIGALRM, terminate); signal.alarm(75*60)
        require(sha(ROOT/'docs/lighton4b-mixed-calibration-v1.json') == p['calibration_only']['frozen_protocol_sha256'] and sha(ROOT/'tests/regression/lighton4b-mixed-v1/frozen-inputs.json') == p['calibration_only']['input_manifest_sha256'], 'calibration protocol/input pins differ')
        # Repeat the full already-passed model-free union on this post-merge source before staging any model.
        phase_out=out/'controls';phase_out.mkdir();run('post-merge-model-free',[sys.executable,ROOT/'tools/ci-heavy/lighton4b_imatrix_q4km_controls.py'],dict(os.environ,HEAVY_OUT=str(phase_out)),timeout=20*60)
        control_receipt=json.loads((phase_out/'imatrix-q4km-controls.json').read_text());require(control_receipt['status']=='passed','post-merge model-free controls failed');r['post_merge_model_free_controls_passed']=True
        build=scratch.parent/'lighton-imatrix/build';targets=['lighton4b-native-document','lighton4b-native-case-preflight','test-lighton4b-stream-head-benchmark'];run('build-native-runners',['cmake','--build',build,'--target',*targets,'--parallel','2']);r['binary_sha256']={x:sha(build/x)for x in targets};bundle=bundle_runtime(build,out);r['runtime_bundle_manifest_sha256']=sha(out/'runtime-bundle.json')
        adapted=phase_out/'adapter/adapted-imatrix.gguf';stats=json.loads((phase_out/'adapter/adapter.json').read_text());require(sha(adapted)==p['statistics']['adapted_sha256'],'approved adapted file differs')
        baseline=out/'unweighted-null-control.gguf';download(p['null_control'],baseline);r['null_control_measured']={'bytes':baseline.stat().st_size,'sha256':sha(baseline)};save();require(r['null_control_measured']=={'bytes':p['null_control']['bytes'],'sha256':p['null_control']['sha256']},'exact f873 null file failed')
        baseline_audit=audit_baseline(baseline,None);(out/'null-payload-audit.json').write_text(json.dumps(baseline_audit,indent=2,sort_keys=True)+'\n');r['null_file_exact']=True
        env = dict(os.environ); env.update({x:'1' for x in common.GATES}); env.update({x:'1' for x in vision.VISION_GATES})
        for name in ('BF16_LINEAR','BF16_QUANT_WEIGHTS','VISION_BF16_ROTARY_AVX2','DIRECT_QUANT','GREEDY_TRACE','MIXED_ATTN_Q8_FFN_Q4','MIXED_GDN_OUT_Q8','MIXED_GDN_OUT_Q6','Q6_K','IMATRIX_Q4KM_Q6HEAD','STREAM_GRAPH_REUSE'): env['CRISPEMBED_LIGHTON4B_'+name] = '0'
        for name in ('WEIGHT_CACHE','PRECISE_RMS','IMATRIX'): env['CRISPEMBED_QWEN35_'+name] = '0'
        env.update(CRISPEMBED_LIGHTON4B_HEAD_BF16='1', CRISPEMBED_LIGHTON4B_PARALLEL_DEQUANT='1', CRISPEMBED_LIGHTON4B_FROZEN_CASES='1')
        env['LD_LIBRARY_PATH']=str(bundle/'lib')
        r['fixed_gates'] = {k:v for k,v in env.items() if k.startswith('CRISPEMBED_')}
        full_helpers = load('weighted_full_helpers', ROOT/'tools/ci-heavy/lighton4b_gdn_out_q6_full.py')
        baseline_report_path = scratch/'frozen-decoded-baselines.json'
        baseline_url = 'https://raw.githubusercontent.com/CrispStrobe/CrispEmbed/8790b0f345c0c2445c18f176ed3b38f97ba4d64c/tests/results/qwen35-ocr/lightonocr3-4b-gdn-out-full/report.json'
        with urllib.request.urlopen(baseline_url,timeout=90) as remote: baseline_bytes=remote.read(8*2**20+1)
        require(len(baseline_bytes)<=8*2**20 and hashlib.sha256(baseline_bytes).hexdigest()==p['calibration_only']['baseline_calibration_report_sha256'],'decoded baseline receipt differs'); baseline_report_path.write_bytes(baseline_bytes); prior=json.loads(baseline_bytes)
        originals = {}; captures = {'simple_table':'6be5428f247b616e76d79513b62873cf0238842999be2fb0848997f8ced25763','arabic_printed_line':'ff1a390370b702fabf8b5a25f0feec76888bb1ad8197cf08c3efd44adec1de84'}
        for case, digest in captures.items():
            path = unique_capture(Path('/kaggle/input'),case,digest); capture = json.loads(path.read_text()); _,contract,frozen = cases.load_contract(case); cases.validate_original(capture,contract); require(contract['split']=='calibration','calibration partition required')
            for record in list(capture['prepared_inputs'].values())+[capture['scores'][0]]:
                require(Path(record['file']).name == record['file'] and sha(path.parent/record['file']) == record['sha256'], 'source array immutable identity differs')
            png = ROOT/'tests/regression/lighton4b-mixed-v1'/frozen['png_file']; require(sha(png)==frozen['png_sha256'],'image hash differs'); image=Image.open(png).convert('RGB'); rgb=image.tobytes(); require(image.size==(frozen['width'],frozen['height']) and hashlib.sha256(rgb).hexdigest()==frozen['rgb_sha256'],'image geometry/RGB differs'); rgb_path=scratch/(case+'.rgb');rgb_path.write_bytes(rgb);teacher=scratch/(case+'.teacher.ids');teacher.write_text(str(capture['generated_ids'][0])+'\n'); originals[case]=(path,capture,image,rgb_path,teacher); r['cases'][case]={}
        def score(case, model, arm, cap=1):
            path,capture,image,rgb,teacher=originals[case]; capacity(); folder=out/case/arm; folder.mkdir(parents=True); pre=folder/'preflight';pre.mkdir();native=folder/'native';native.mkdir()
            child=dict(env);candidate_arm=arm=='weighted_q4km';child['CRISPEMBED_LIGHTON4B_Q6_K']='1'if candidate_arm else'0';child['CRISPEMBED_LIGHTON4B_IMATRIX_Q4KM_Q6HEAD']='1'if candidate_arm else'0'
            run(case+'-'+arm+'-preflight',[bundle/'bin/lighton4b-native-case-preflight',model,rgb,image.height,image.width,pre,profile.RECIPE if candidate_arm else'q4_k'],child)
            arrays={k:np.load(path.parent/v['file'],allow_pickle=False) for k,v in capture['prepared_inputs'].items()};require(set(arrays)=={'pixel_values','image_grid_thw','input_ids','mm_token_type_ids','attention_mask'},'five source inputs required');require((pre/'processor.patches.f32').read_bytes()==arrays['pixel_values'].tobytes(),'native processor differs')
            for field,name in [('image_grid_thw','grid'),('input_ids','prompt'),('mm_token_type_ids','modalities')]:require([int(x) for x in (pre/(name+'.ids')).read_text().split()]==arrays[field].reshape(-1).tolist(),'native input differs: '+field)
            require(np.all(arrays['attention_mask']==1),'source mask differs');run(case+'-'+arm+'-native',[bundle/'bin/lighton4b-native-document',model,rgb,image.height,image.width,teacher,native,cap,4,'smoke'],child)
            row=native/'prefill.logits.f32';require(row.stat().st_size==993280 and sha(row)==sha(native/'teacher.0.logits.f32')==sha(native/'greedy.0.logits.f32'),'native row0 paths differ');v=np.fromfile(row,dtype='<f4').astype(np.float64);gold=np.fromfile(path.parent/capture['scores'][0]['file'],dtype='<f4').astype(np.float64);require(v.size==gold.size==248320 and np.isfinite(v).all() and np.isfinite(gold).all(),'complete finite vocabulary required');ids=[int(x)for x in(native/'generated.ids').read_text().split()];require(ids and ids[0]==int(v.argmax()),'independent generation differs from argmax');e=v-gold;prob=np.exp(v-v.max());prob/=prob.sum()
            if arm=='plain_q4':require(sha(row)==p['calibration_only']['plain_q4_first_score_sha256'][case],'frozen null score failed; candidate prohibited')
            result={'first_score_sha256':sha(row),'generated_ids':ids,'actual_eos':ids[-1]==248046,'cap_hit':ids[-1]!=248046 and len(ids)==cap,'cap':cap,'max_abs':float(np.abs(e).max()),'rms':float(np.sqrt(np.mean(e*e))),'cosine':float(v@gold/(np.linalg.norm(v)*np.linalg.norm(gold))),'probability_of_original_id':float(prob[capture['generated_ids'][0]]),'source_first_score_sha256':capture['scores'][0]['sha256'],'source_capture_sha256':sha(path),'native_summary':json.loads((native/'native-summary.json').read_text()),'artifacts':{x.name:{'bytes':x.stat().st_size,'sha256':sha(x)}for x in native.iterdir()if x.is_file()}}
            summary=result['native_summary']; require(summary['eos']==result['actual_eos'] and summary['generated_tokens']==len(ids) and summary['cap']==cap,'native independent EOS/cap summary differs')
            if cap>1:
                require(result['actual_eos'] or len(ids)==cap,'generation stopped before EOS/cap')
                text=(native/'generated.txt').read_text(); result['decoded']=text
                for i,token in enumerate(ids):
                    row_values=np.fromfile(native/f'greedy.{i}.logits.f32',dtype='<f4');require(row_values.size==248320 and np.isfinite(row_values).all() and int(row_values.argmax())==token,'independent score argmax differs')
                require(len(list(native.glob('greedy.*.logits.f32')))==len(ids),'independent greedy score rows differ')
                require(result['first_score_sha256']==r['cases'][case]['weighted_q4km']['first_score_sha256'],'full generation prefill differs from smoke')
                result['against_genuine_bf16']=full_helpers.agreement(ids,text,capture)
                result['against_retained_plain_q4']=full_helpers.agreement(ids,text,prior['cases'][case]['q4'])
                result['against_retained_q8_out']=full_helpers.agreement(ids,text,prior['cases'][case]['out'])
            r['cases'][case][arm]=result;save()
        for case in originals:score(case,baseline,'plain_q4')
        r['both_frozen_null_scores_exact']=True;save();baseline.unlink()
        source=scratch/'original-f16.gguf';download(p['source'],source);r['source_f16_authenticated']=p['source'];save()
        reader=gguf.GGUFReader(str(source),'r');inventory={t.name:{'stored_shape':list(map(int,t.shape)),'type':int(t.tensor_type)}for t in reader.tensors};validate_source_inventory(inventory,plan())
        protected={t.name:(int(t.data_offset),int(t.n_bytes))for t in reader.tensors if int(t.tensor_type)==0};require(len(protected)==474,'source protection inventory differs')
        with source.open('rb')as stream:
            source_protected={name:span(stream,offset,size)for name,(offset,size)in protected.items()}
        require(all(source_protected[name]==baseline_audit['tensor_payload_sha256'][name]for name in source_protected),'source/null protected F32 data differ');del reader
        actual=out/'actual248-source-header.log';run('actual248-source-header',[build/'crispembed-quantize','--check-lighton4b-imatrix',adapted,source]);means=json.loads(actual.read_text());require(means['normalized_f32_sha256']=={x['target']:x['normalized_f32_sha256']for x in stats['tensors']},'all248 actual means differ')
        candidate=out/(profile.RECIPE+'.gguf');r['quantization_attempted']=True;save();run('one-pass-original-q4km-quantization',[build/'crispembed-quantize',*profile.quantizer_arguments(source,candidate,adapted)])
        log=(out/'one-pass-original-q4km-quantization.log').read_text();require(log.count('(imatrix)')==248 and re.search(r'248 with imatrix',log),'actual248 weighted applications absent');audit=audit_q4km(candidate,baseline_audit['tensor_payload_sha256']);(out/'candidate-payload-audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n');r['candidate']={k:v for k,v in audit.items()if k!='tensor_payload_sha256'};r['actual248_applied']=True;r['weight_quantization_executed']=True;save();source.unlink()
        for case in originals:score(case,candidate,'weighted_q4km')
        # Retain a bounded real-head fixture for exact OFF/ON replay, not another model forward.
        q=parse(candidate,{0:(1,4),12:(256,144),14:(256,210)})['tensors']['l.embed_tokens.weight'];head=out/'head';head.mkdir();h=hashlib.sha256();remaining=q['bytes']
        with candidate.open('rb')as src,(head/'weights.q6').open('xb')as dst:
            src.seek(q['data_offset'])
            while remaining:
                block=src.read(min(4*2**20,remaining));require(block,'truncated retained head');dst.write(block);h.update(block);remaining-=len(block)
        require(h.hexdigest()==audit['tensor_payload_sha256']['l.embed_tokens.weight'],'retained raw Q6 head differs');norm=out/'simple_table/weighted_q4km/native/prefill.norm.f32';require(norm.stat().st_size%10240==0,'real head norm geometry differs')
        with norm.open('rb')as stream:stream.seek(-10240,2);hidden=stream.read(10240)
        (head/'input.f32').write_bytes(hidden);head_manifest={'type':'q6_k','width':2560,'rows':248320,'weights':{'file':'weights.q6','sha256':h.hexdigest()},'hidden':{'file':'input.f32','sha256':hashlib.sha256(hidden).hexdigest()},'source_prefill_logits_sha256':r['cases']['simple_table']['weighted_q4km']['first_score_sha256']};(head/'manifest.json').write_text(json.dumps(head_manifest,indent=2,sort_keys=True)+'\n');r['head_manifest_sha256']=sha(head/'manifest.json')
        r['full_generation_executed']=False;r['full_generation_followup_required']=True
        r['candidate_native_source_sha']=r['source_sha'];r['source_blob_sha256']={name:sha(ROOT/name)for name in('tools/quantize.cpp','src/lighton4b_native_context.h','src/lighton4b_q6_policy.h','src/lighton4b_q4km_types.h','src/qwen35_decoder.cpp','src/qwen35_linear.cpp','src/core/lighton4b_imatrix_q4km.h')};r['status']='measured'
    except Exception as error:
        r['error_type']=type(error).__name__;r['error']=str(error) if isinstance(error,ValueError) else 'bounded native-HF Q4KM-style conversion/first-score worker failed';raise
    finally:signal.alarm(0);r['seconds']=time.monotonic()-started;save()
if __name__=='__main__':main()
