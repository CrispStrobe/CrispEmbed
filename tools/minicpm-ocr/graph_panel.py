#!/usr/bin/env python3
"""Hosted-only frozen MiniCPM graph/Q8/protected-Q4 experiment; no new gold."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import types
import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools'))
from josev2b_composed_panel import metrics
from graph_adapter import write_gguf, write_pair, collect

CASES = ['line', 'para-en', 'table', 'tall', 'tiny']
MAXIMUM = 1024
PROTOCOL = ROOT/'docs/minicpm-ocr-graph-v1.json'
SELECTED = re.compile(r'l\.layers\.\d+\.(mlp\.(gate_proj|up_proj|down_proj)|self_attn\.(q_proj|k_proj|v_proj|o_proj)|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))\.weight')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=1, allow_nan=False, ensure_ascii=False)+'\n')


def spec(path, shape):
    return dict(path=str(path), shape=list(shape), sha256=digest(path))


def edits(a, b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def first_difference(a, b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None if len(a)==len(b) else min(len(a),len(b)))


def captured(command, folder, name, env, timeout):
    """Keep logs on timeout/termination instead of discarding capture buffers."""
    stdout=folder/(name+'.stdout');stderr=folder/(name+'.stderr')
    timed_out=False
    with stdout.open('wb') as a,stderr.open('wb') as b:
        process=subprocess.Popen(list(map(str,command)),stdout=a,stderr=b,env=env,start_new_session=(os.name!='nt'))
        try:process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            if os.name=='nt':process.kill()
            else:os.killpg(process.pid,signal.SIGKILL)
            process.wait()
    return types.SimpleNamespace(returncode=process.returncode,stdout=stdout.read_bytes(),stderr=stderr.read_bytes(),timed_out=timed_out)

TIME_RE = re.compile(r'^\s*\[time\] minicpm_ocr\.(\w+) (.*)$')


def parse_timing(stderr):
    t = dict(vision=[])
    for line in stderr.splitlines():
        m = TIME_RE.match(line)
        if not m:
            continue
        name, rest = m.groups()
        if name == 'vision':
            kv = dict(re.findall(r'(\w+)=([\d.]+)(?:ms)?', rest))
            size = re.search(r' (\d+)x(\d+) ', ' ' + rest + ' ')
            t['vision'].append(dict(part=int(kv['part']), size=[int(size.group(1)), int(size.group(2))] if size else None,
                                    patches=int(kv['patches']), tokens=int(kv['tokens']),
                                    **{k + '_ms': float(kv[k]) for k in ('total', 'setup', 'encoder', 'projector', 'readback')}))
        elif name == 'decode':
            kv = dict(re.findall(r'(\w+)=([\d.]+)ms', rest))
            steps = re.search(r'steps=(\d+)', rest)
            lst = re.search(r'list=([\d.,]*)', rest)
            t['decode'] = dict(steps=int(steps.group(1)), **{k + '_ms': float(v) for k, v in kv.items()},
                               step_ms=[float(x) for x in lst.group(1).split(',') if x] if lst else [])
        elif name == 'ids':
            lst = re.search(r'list=([\d,]*)', rest)
            t['ids'] = [int(x) for x in lst.group(1).split(',') if x] if lst else []
        else:
            ms = re.match(r'([\d.]+)ms', rest)
            t[name + '_ms'] = float(ms.group(1)) if ms else None
            extra = dict(re.findall(r'(\w+)=(\S+)', rest))
            if extra:
                t[name + '_info'] = extra
    if t['vision']:
        t['vision_encoder_ms'] = round(sum(v['encoder_ms'] for v in t['vision']), 3)
        t['vision_projector_ms'] = round(sum(v['projector_ms'] for v in t['vision']), 3)
        t['vision_total_ms'] = round(sum(v['total_ms'] for v in t['vision']), 3)
    d = t.get('decode')
    if d and d['step_ms']:
        s = d['step_ms']
        d['first_step_ms'] = s[0]
        d['steady_mean_ms'] = round(sum(s[1:]) / max(1, len(s) - 1), 3)
        d['step_ms_sample'] = s[:8] + (['...'] + s[-4:] if len(s) > 12 else s[8:])
        del d['step_ms']
    return t


def recover_case(directory, cid, dtype, graph, error):
    partial=Path(directory)/'partial-case.json'
    row=json.loads(partial.read_text()) if partial.is_file() else dict(id=cid,dtype=dtype,graph=graph)
    require((row.get('id'),row.get('dtype'),row.get('graph'))==(cid,dtype,graph),'partial case identity differs')
    row.update(passed=False,error=str(error))
    return row


def source_guard(source):
    require(re.fullmatch('[0-9a-f]{40}', source), 'immutable source required')
    actual = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(actual == source, 'caller source differs')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True), 'tracked source dirty')
    link = subprocess.check_output(['git','ls-tree','HEAD','ggml'],cwd=ROOT,text=True).split()[2]
    checkout = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'ggml',text=True).strip()
    require(link == checkout, 'ggml gitlink/checkout mismatch')
    paths = subprocess.check_output(['git','ls-files','src','CMakeLists.txt','tools/minicpm-ocr','tests/minicpm-graph','models/convert-minicpm-ocr-to-gguf.py','docs/minicpm-ocr-graph-v1.json','tools/josev2b_composed_panel.py','tools/josev2b_composed_diff.py','tests/results/minicpm-ocr/native-parity.json','docs/minicpm-ocr-source.json','models/convert-qwen35-ocr-to-gguf.py','tools/quantize.cpp','tests/test_quantize_patch_float.py','tests/test_quantize_tensor_overrides.py','tools/kaggle/minicpm-ocr-graph','tests/test_ultrabert_profile.cpp','tests/josev-quant','tests/josev-packed'],cwd=ROOT,text=True).splitlines()
    return dict(source=source, ggml=link, files={p:digest(ROOT/p) for p in paths})


def verify_reference(base):
    source = json.loads((ROOT/'docs/minicpm-ocr-source.json').read_text())
    parity = json.loads((ROOT/'tests/results/minicpm-ocr/native-parity.json').read_text())
    ref = json.loads((base/'reference.json').read_text())
    require(digest(base/'reference.json') == 'aeb1756dcbcfb72d72c13cf5a8d8131bb90e8d8884cf07c70723cc1fe4a64c32', 'immutable original report SHA differs')
    require(ref.get('completed') is True and ref['revision'] == source['revision'] == parity['reference_revision'], 'original reference identity')
    require(ref['driver_sha256'] == digest(ROOT/'tools/minicpm-ocr/reference.py') and ref['contract_sha256'] == digest(ROOT/'docs/minicpm-ocr-source.json'), 'frozen original source differs')
    require([c['id'] for c in ref['cases']] == CASES == [c['id'] for c in parity['cases']], 'all five stable cases required')
    for row, gold in zip(ref['cases'], parity['cases']):
        require(row['task'] == gold['task'] and row['size'] == gold['size'] and row['ids'] == gold['reference_ids'] and row['text'] == gold['reference_text'] and row['eos'] == gold['reference_eos'] and row['cap_hit'] == gold['reference_cap_hit'], 'original decoded panel differs')
        require(digest(base/(row['id']+'.png')) == row['image_sha256'], 'frozen image differs')
        for name, desc in row['arrays'].items():
            path = base/desc['path']
            require(path.resolve().is_relative_to(base.resolve()) and digest(path) == desc['sha256'], 'original array SHA differs: '+name)
        for label in [f'model.language_model.layers.{i}' for i in range(24)] + ['model.language_model.norm','lm_head']:
            require(label in row['arrays'], 'missing original prefill stage: '+label)
    return ref, dict(reference_sha256=digest(base/'reference.json'), parity_sha256=digest(ROOT/'tests/results/minicpm-ocr/native-parity.json'), driver_sha256=ref['driver_sha256'], contract_sha256=ref['contract_sha256'], revision=ref['revision'])


def tensor_inventory(path):
    from gguf import GGUFReader
    result = {}
    for t in GGUFReader(str(path)).tensors:
        h = hashlib.sha256(); view = memoryview(t.data).cast('B')
        for at in range(0,len(view),8 << 20): h.update(view[at:at+(8 << 20)])
        result[t.name] = dict(type=int(t.tensor_type),shape=[int(x) for x in t.shape],bytes=len(view),sha256=h.hexdigest())
    require(len(result) == 779, 'exact779 stored tensors required')
    return result


def protected_q4(control, candidate):
    require(set(control) == set(candidate), 'Q4 names differ')
    selected = []
    for name, row in control.items():
        other = candidate[name]
        require(row['shape'] == other['shape'], 'Q4 shape differs: '+name)
        if SELECTED.fullmatch(name):
            require(row['type'] in (0,1) and other['type'] == 12, 'selected decoder matrix is not actual Q4_K: '+name)
            selected.append(name)
        else:
            require(row == other, 'protected tensor bytes/type differ: '+name)
    require(len(selected) == 186, 'exact186 decoder projection matrices required')
    return dict(selected=sorted(selected), protected=593, all_protected_byte_exact=True)


def protected_q8_patch(control, candidate):
    name='v.emb.patch_embedding.weight'
    expected=control[name]
    require(expected['type']==1 and len(expected['shape'])==4,'F16 source patch profile differs')
    require(candidate[name]==expected,'Q8 patch must retain exact F16 type, four axes and payload SHA')
    return dict(name=name,**expected,exact_f16_source=True)


def load_npy(path, shape=None):
    import numpy as np
    a = np.load(path,allow_pickle=False)
    require(a.dtype == np.dtype('<f4') and a.size and np.isfinite(a).all(), 'finite F32 stage required')
    if shape is not None:
        require(a.shape == shape, 'explicit native shape differs')
    return a


def raw_spec(path, a):
    import numpy as np
    require(a.dtype == np.dtype('<f4') and np.isfinite(a).all(), 'finite exact F32 adapter required')
    a.tofile(path)
    return spec(path,a.shape)


def compare_pairs(pairs, binary, folder):
    import numpy as np
    folder.mkdir(parents=True,exist_ok=True)
    write_pair(folder/'original.gguf',folder/'native.gguf',pairs)
    run = subprocess.run([str(binary),str(folder/'original.gguf'),str(folder/'native.gguf'),str(folder/'diff.json')],capture_output=True,text=True)
    (folder/'diff.log').write_text(run.stdout+run.stderr)
    independent = {}
    for p in pairs:
        a=np.fromfile(p['original']['path'],dtype='<f4').reshape(p['original']['shape'])
        b=np.fromfile(p['native']['path'],dtype='<f4').reshape(p['native']['shape'])
        independent[p['name']]=metrics(a,b)
    proof=collect(run.returncode,folder/'diff.json',independent)
    write(folder/'comparison.json',dict(actual_ref=proof,stages=independent))
    return dict(actual_ref=proof,stages=independent,passed=all(x['passed'] for x in independent.values()),diff_log_sha256=digest(folder/'diff.log'))


def run_case(args, base, case, dtype, graph, model, output):
    import numpy as np
    folder=output/f'{dtype}-graph{graph}'/case['id']; dump=folder/'dump';dump.mkdir(parents=True)
    env=dict(os.environ,CRISPEMBED_MINICPM_OCR='1',CRISPEMBED_QWEN35_OCR='1',CRISPEMBED_QWEN35_QUANT='1' if dtype in ('q8_0','q4_k') else '0',CRISPEMBED_MINICPM_OCR_GRAPH=str(graph),CRISPEMBED_QWEN35_WEIGHT_CACHE='0',CRISPEMBED_MINICPM_OCR_TIMING='1',CRISPEMBED_MINICPM_OCR_DUMP=str(dump),OMP_NUM_THREADS=str(args.threads))
    for name in ['CRISPEMBED_QWEN35_IMATRIX','CRISPEMBED_QWEN35_PARALLEL_F16','CRISPEMBED_MINICPM_OCR_VISION_FLASH']:
        env.pop(name,None)
    start=time.monotonic()
    process=captured([str(args.cli),'-m',str(model),'--ocr',str(base/(case['id']+'.png')),'--ocr-prompt',case['task'],'--ocr-max-tokens',str(MAXIMUM),'-t',str(args.threads)],folder,'native',env,5400)
    row=dict(id=case['id'],dtype=dtype,graph=graph,exit=process.returncode,wall_seconds=time.monotonic()-start,stdout_sha256=digest(folder/'native.stdout'),stderr_sha256=digest(folder/'native.stderr'),timed_out=process.timed_out,timings=parse_timing(process.stderr.decode('utf-8','replace')))
    if process.returncode:
        return dict(row,passed=False,error='native command failed before complete output')
    ids=np.load(dump/'ids.npy',allow_pickle=False).tolist()
    native_input=np.load(dump/'input_ids.npy',allow_pickle=False).tolist()
    text=process.stdout.decode('utf-8')
    if text.endswith('\n'):text=text[:-1]
    row.update(ids=ids,ids_exact=ids==case['ids'],input_ids_exact=native_input==case['input_ids'],text=text,text_exact=text==case['text'],eos=bool(ids) and ids[-1] in (248044,248046),cap_hit=len(ids)>=MAXIMUM)
    row.update(first_input_id_difference=first_difference(native_input,case['input_ids']),first_generated_id_difference=first_difference(ids,case['ids']),characters=len(text),reference_characters=len(case['text']),cer=edits(text,case['text'])/max(1,len(case['text'])),wer=edits(text.split(),case['text'].split())/max(1,len(case['text'].split())))
    write(folder/'partial-case.json',row)
    confidence=load_npy(dump/'confidences.npy').reshape(-1)
    require(confidence.size==len(ids) and ((confidence>=0)&(confidence<=1)).all(), 'every native confidence finite/bounded')
    row['confidences']=confidence.tolist()
    row['mean_confidence']=float(confidence.astype('float64').mean())
    write(folder/'partial-case.json',row)
    pairs=[]
    for i,label in enumerate([f'model.language_model.layers.{i}' for i in range(24)]+['model.language_model.norm','lm_head']):
        original=load_npy(base/case['arrays'][label]['path'])
        native=load_npy(dump/(label.replace('.','_')+'.npy'))
        # Publisher stores batch1 features [1,T,C]; native stores explicit
        # token rows [T,C]. Removing only known singleton batch is declared.
        if original.ndim==3:
            require(original.shape[0]==1,'only known batch1 alignment');original=original[0]
        if label=='lm_head':
            require(original.size==248100 and native.shape==(1,248100),'first vocabulary row shape')
            original=original.reshape(1,248100)
        require(original.shape==native.shape,'prefill stage shape differs: '+label)
        pairs.append(dict(name=f'stage_{i}',original=raw_spec(folder/f'original_{i}.f32',original),native=raw_spec(folder/f'native_{i}.f32',native)))
    row['prefill']=compare_pairs(pairs,args.diff,folder/'prefill')
    row['first_divergence']=next((dict(stage=i,**v) for i,v in row['prefill']['stages'].items() if not v['passed']),None)
    row['passed']=row['input_ids_exact'] and row['ids_exact'] and row['text_exact'] and row['eos']==case['eos'] and row['cap_hit']==case['cap_hit'] and row['prefill']['passed']
    write(folder/'partial-case.json',row)
    if graph and dtype in ('f32','f16'):
        audit=folder/'projected';audit.mkdir()
        idfile=raw_spec(folder/'ids-as-exact-f32.f32',np.asarray(case['input_ids'],dtype='<f4'))
        merged=raw_spec(folder/'merged.f32',load_npy(dump/'model_merger.npy'))
        write_gguf(folder/'probe-input.gguf',[dict(name='input_ids',**idfile),dict(name='merged',**merged)])
        probe=captured([str(args.probe),str(model),str(folder/'probe-input.gguf'),str(audit),str(args.threads)],audit,'probe',env,3600)
        (audit/'probe.log').write_bytes(probe.stdout+probe.stderr)
        require(probe.returncode==0,'projected probe failed')
        journal=json.loads((audit/'projections.json').read_text());require(journal['completed'] and journal['calls']==187 and journal['ordinary_observer_repeat_exact'] and journal['cache_exact'],'projection coverage/neutrality')
        pp=[]
        for entry in journal['stages']:
            key=entry['key'];shape=[len(entry['selected_rows']),entry['output_width']]
            pp.append(dict(name=key,original=spec(audit/(key+'.control.f32'),shape),native=spec(audit/(key+'.native.f32'),shape)))
        # Actual root CLI and the isolated observed build must compute the
        # identical full prefill, not merely similar sampled projections.
        for i in range(24):
            require(digest(audit/f'layer_{i}.f32')==digest(folder/f'native_{i}.f32'),'root/isolated full layer byte mismatch')
        require(digest(audit/'norm.f32')==digest(folder/'native_24.f32') and digest(audit/'logits.f32')==digest(folder/'native_25.f32'),'root/isolated final byte mismatch')
        row['projected']=compare_pairs(pp,args.diff,audit/'comparison')
        row['projected']['scope']=journal['sampling']
        row['projected']['stored_types']=journal['stages']
        row['first_projected_divergence']=next((dict(weight=item['weight'],stored_type=item['stored_type'],**row['projected']['stages'][item['key']]) for item in journal['stages'] if not row['projected']['stages'][item['key']]['passed']),None)
        row['passed'] &= row['projected']['passed']
    return row


def reference_controls(binary, out):
    """Real Ref positive, row-sensitive numeric failure, wrong-shape failure."""
    import struct
    out.mkdir(parents=True,exist_ok=False)
    values=[1.,.5,.25,.0001,.0002,.0003]
    good=out/'good.f32';bad=out/'bad.f32'
    good.write_bytes(struct.pack('<6f',*values));bad.write_bytes(struct.pack('<6f',*values[:3],*[-x for x in values[3:]]))
    left=spec(good,[2,3]);negative=spec(bad,[2,3])
    proofs=[]
    for name,right,expected in [('positive',left,0),('bad-row',negative,1),('wrong-shape',dict(left,shape=[6]),2)]:
        folder=out/name;folder.mkdir()
        write_gguf(folder/'original.gguf',[dict(name='control',**left)])
        write_gguf(folder/'native.gguf',[dict(name='control',**right)])
        process=subprocess.run([str(binary),str(folder/'original.gguf'),str(folder/'native.gguf'),str(folder/'diff.json')],capture_output=True,text=True)
        (folder/'diff.log').write_text(process.stdout+process.stderr)
        require(process.returncode==expected,'actual Ref control exit mismatch: '+name)
        receipt=json.loads((folder/'diff.json').read_text())
        if name=='bad-row':
            row=receipt['stages'][0]
            require(row['max_absolute']<.002 and row['min_row_cosine']<.99999 and not row['passed'],'row-sensitive Ref control failed')
        proofs.append(dict(name=name,exit=process.returncode,report_sha256=digest(folder/'diff.json'),log_sha256=digest(folder/'diff.log')))
    write(out/'controls.json',dict(passed=True,proofs=proofs))
    return proofs


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--ref-controls':
        reference_controls(Path(sys.argv[2]),Path(sys.argv[3]));return 0
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--reference',type=Path,required=True)
    for key in ('cli','probe','diff','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--model',action='append',required=True);parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    require(os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or os.environ.get('GITHUB_ACTIONS')=='true','hosted execution only')
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(PROTOCOL.read_text())
    require(config['cases']==CASES and config['arms']==[['f32',0],['f32',1],['f16',0],['f16',1],['q8_0',1],['q4_k',1]],'fixed protocol cases/arms differ')
    require(config['execution_order']=='case-major-with-declared-six-arm-order','declared case-major execution order required')
    require(config['comparison']['max_absolute']==.002 and config['comparison']['minimum_row_cosine']==.99999 and config['max_new_tokens']==MAXIMUM,'fixed protocol gates differ')
    report=dict(schema='crispembed-minicpm-stored-graph-v1',completed=False,qualified=False,cases=[],source=source_guard(args.source),protocol_sha256=digest(PROTOCOL))
    save=lambda:write(args.output/'panel.json',report)
    save()
    try:
        report['ref_controls']=reference_controls(args.diff,args.output/'ref-controls');save()
        ref,report['original']=verify_reference(args.reference);save()
        require(len(args.model)==4,'exact four distinct model arguments required')
        models=dict(x.split('=',1) for x in args.model)
        require(set(models)=={'f32','f16','q8_0','q4_k'},'all four fixed model arms required')
        report['models']={key:dict(bytes=Path(path).stat().st_size,sha256=digest(path),tensors=tensor_inventory(path)) for key,path in models.items()}
        report['q4_protection']=protected_q4(report['models']['f16']['tensors'],report['models']['q4_k']['tensors'])
        report['q8_patch_protection']=protected_q8_patch(report['models']['f16']['tensors'],report['models']['q8_0']['tensors'])
        report['binary_sha256']={key:digest(getattr(args,key)) for key in ('cli','probe','diff')};save()
        for case in ref['cases']:
            for dtype,graph in [('f32',0),('f32',1),('f16',0),('f16',1),('q8_0',1),('q4_k',1)]:
                try:row=run_case(args,args.reference,case,dtype,graph,models[dtype],args.output)
                except Exception as error:
                    row=recover_case(args.output/f'{dtype}-graph{graph}'/case['id'],case['id'],dtype,graph,error)
                directory=args.output/f'{dtype}-graph{graph}'/case['id']
                directory.mkdir(parents=True,exist_ok=True)
                inventory={str(path.relative_to(directory)):dict(bytes=path.stat().st_size,sha256=digest(path)) for path in sorted(directory.rglob('*')) if path.is_file()}
                write(directory/'files.json',inventory)
                row['files_manifest_sha256']=digest(directory/'files.json')
                case_report=directory/'case-report.json'
                case_report.parent.mkdir(parents=True,exist_ok=True);write(case_report,row)
                summary={key:value for key,value in row.items() if key not in ('prefill','projected','first_divergence')}
                summary.update(case_report=str(case_report.relative_to(args.output)),case_report_sha256=digest(case_report))
                for scope in ('prefill','projected'):
                    if scope in row:
                        stages=row[scope]['stages']
                        summary[scope]=dict(passed=row[scope]['passed'],stages=len(stages),worst=max((dict(name=k,**v) for k,v in stages.items()),key=lambda v:v['max_absolute']))
                report['cases'].append(summary);save()
        report['paired_confidences']=[]
        for dtype in ('f32','f16'):
            for case in CASES:
                pair=[next(x for x in report['cases'] if x['dtype']==dtype and x['graph']==gate and x['id']==case) for gate in (0,1)]
                valid=all(x.get('passed') and isinstance(x.get('confidences'),list) for x in pair)
                if valid:
                    error=max(abs(a-b) for a,b in zip(pair[0]['confidences'],pair[1]['confidences']))
                    valid=len(pair[0]['confidences'])==len(pair[1]['confidences']) and error<=.002
                else:error=None
                report['paired_confidences'].append(dict(dtype=dtype,id=case,passed=valid,max_absolute=error,budget=.002))
        report['completed']=True
        report['qualified']=len(report['cases'])==30 and all(x.get('passed') is True for x in report['cases']) and all(x['passed'] for x in report['paired_confidences'])
        report['per_precision']={dtype:dict(rows=sum(x['dtype']==dtype for x in report['cases']),passed=all(x.get('passed') is True for x in report['cases'] if x['dtype']==dtype) and all(x['passed'] for x in report['paired_confidences'] if x['dtype']==dtype)) for dtype in models}
        save()
        return 0 if report['qualified'] else 1
    except Exception as error:
        report['error']=str(error);save();return 2


if __name__=='__main__':
    raise SystemExit(main())
