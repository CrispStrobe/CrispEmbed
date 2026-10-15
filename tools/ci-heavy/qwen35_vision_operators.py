"""Original-blueprint vision localization using the shared crispembed-diff harness."""
import gc
import json
import os
import re
import subprocess
from collections import Counter
from qwen35_quant_calibration import file_sha


METRIC = r'n=(\d+) cos_min=(\S+) cos_global=(\S+) max_abs=(\S+) mean_abs=(\S+) rms=(\S+) norms=(\S+)/(\S+)'
BUDGET = r'^BUDGET (\S+): strict_absolute_pass=(\d+) calibrated=(\d+) self_max_abs=(\S+) max_ratio=(\S+)$'


def metric(match):
    return dict(elements=int(match[0]), **dict(zip(
        ('cos_min', 'cos_global', 'max_abs', 'mean_abs', 'rms', 'native_norm', 'reference_norm'), map(float, match[1:]))))


def vision_records(log):
    lines = log.splitlines()
    records = []
    for i, line in enumerate(lines):
        m = re.match(r'^VISION_METRIC (\S+) (\S+) ' + METRIC + '$', line)
        if not m:
            continue
        mode, stage = m[1], m[2]
        if lines[i+1] != 'MODE ' + mode or not lines[i+2].startswith(('PASS ' + stage + ':', 'FAIL ' + stage + ':')):
            raise ValueError('Missing shared comparator verdict')
        budget = re.match(BUDGET, lines[i+3])
        if not budget or budget[1] != stage:
            raise ValueError('Missing unchanged qualified budget')
        records.append(dict(mode=mode, stage=stage, **metric(m.groups()[2:]),
            qualified_pass=lines[i+2].startswith('PASS '), strict_absolute_pass=bool(int(budget[2])),
            calibrated=bool(int(budget[3])), self_max_abs=float(budget[4]), max_ratio=float(budget[5])))
    status = re.search(r'^VISION_OPS finite=1 metrics=218 qualified_failures=(\d+) controls=1$', log, re.M)
    if len(records) != 218 or not status or sum(not m['qualified_pass'] for m in records) != int(status[1]):
        raise ValueError('Incomplete finite vision operator diagnostics')
    if Counter(m['mode'] for m in records) != {'composed': 117, 'same-input': 101}:
        raise ValueError('Missing composed or same-input boundaries')
    return records


def prefill_records(log):
    records = [dict(stage=m[0], **metric(m[1:])) for m in re.findall(r'^METRIC (\S+) ' + METRIC + '$', log, re.M)]
    budgets = [dict(stage=m[0], strict_absolute_pass=bool(int(m[1])), calibrated=bool(int(m[2])),
                    self_max_abs=float(m[3]), max_ratio=float(m[4])) for m in re.findall(BUDGET, log, re.M)]
    failed = re.findall(r'^FAIL (\S+): cos_min=', log, re.M)
    status = re.search(r'^PREFILL_ONLY finite=1 metrics=77 selected=(\d+) reference=(\d+) strict_failures=(\d+)$', log, re.M)
    if len(records) != 77 or len(budgets) != 77 or not status or len(failed) != int(status[3]):
        raise ValueError('Incomplete composed decoder diagnostics')
    return dict(metrics=records, budgets=budgets, qualified_failed_stages=failed,
                selected=int(status[1]), reference=int(status[2]))


def run(repo, build, baseline, local, out, module, anchor_image, anchor_npz):
    import gguf
    import numpy as np
    from PIL import Image
    capture = module(repo / 'tools/dump_qwen35_ocr_page_reference.py', 'qwen35_operator_reference')
    frozen = json.loads((repo / 'tests/results/qwen35-ocr/page-parity.json').read_text())
    previous = json.loads((repo / 'tests/results/qwen35-ocr/prefill-isolation.json').read_text())
    root = out / 'vision-operators'
    root.mkdir(exist_ok=True)
    report = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
        model=frozen['model'], revision=frozen['revision'], reference='Original Transformers5.5.4 F32 eager modules and hooks',
        scope='Vision stage traces, test-only same-input operators and native composed prefill; no full-output/latency/corpus claim',
        comparator='crispembed_diff::Ref + unchanged qwen35_compare; all raw failures retained', cases=[], arms=[])
    try:
        commands = json.loads((build / 'compile_commands.json').read_text())
        commands = [c['command'] for c in commands if c['file'].endswith('/qwen35_vision.cpp')]
        if not any('CRISPEMBED_QWEN35_VISION_TEST_OPS=1' in c for c in commands) or not any('CRISPEMBED_QWEN35_VISION_TEST_OPS' not in c for c in commands):
            raise ValueError('Test operator compile isolation missing')
        report['test_operator_compile_isolation'] = 'macro present only in validation target; production source compile command excludes it'
        report['compact_anchor'] = capture.dump_reference(local, anchor_image, root / 'compact-anchor', max_new_tokens=12, anchor=anchor_npz)
        gc.collect()
        report['vision_anchor'] = capture.dump_reference(local, anchor_image, root / 'vision-anchor', max_new_tokens=12, vision_probes=True)
        reader = gguf.GGUFReader(str(root / 'vision-anchor/official-delta.gguf'))
        new_names = {'input.decoded_length', 'prefill.vision.merger.input'}
        fields = ('norm1','qkv','pooled','attention','norm2','up','activation','down')
        for i in range(12):
            new_names.update(f'prefill.vision.probe.{i}.{f}' for f in ('input','norm2_input'))
            if i not in (5,11): new_names.update(f'prefill.vision.probe.{i}.{f}' for f in fields)
        shared, extra = [], []
        with np.load(anchor_npz) as full:
            for t in reader.tensors:
                if t.name in full:
                    if not np.array_equal(t.data, full[t.name]): raise ValueError('Expanded/full original anchor differs: '+t.name)
                    shared.append(t.name)
                else:
                    extra.append(t.name)
        if len(shared) != 126 or set(extra) != new_names: raise ValueError('Unexpected expanded anchor schema')
        report['vision_anchor_audit'] = dict(shared_arrays_exact=shared, additional_original_hook_arrays=sorted(extra))
        del reader
        gc.collect()
        for old in frozen['cases']:
            if old['name'] not in ('form','receipt','german-header'): continue
            source = repo / old['source']['local_file']
            if file_sha(source) != old['source']['sha256']: raise ValueError('Changed source image')
            folder = root / old['name']; folder.mkdir(exist_ok=True)
            image = Image.open(source).convert('RGB')
            if old['crop']: image=image.crop(old['crop'])
            if old['explicit_resize']: image=image.resize(old['explicit_resize'],resample=Image.Resampling.BICUBIC)
            path=folder/'input.png';image.save(path)
            if file_sha(path) != old['input_sha256']: raise ValueError('Changed evaluation derivative')
            case=dict(name=old['name'],source=old['source'],crop=old['crop'],explicit_resize=old['explicit_resize'],input_sha256=file_sha(path))
            report['cases'].append(case)
            case['reference']=capture.dump_reference(local,path,folder,max_new_tokens=1,prefill_controls=True,vision_probes=True)
            if any(len(rows)!=217 for rows in case['reference']['cpu_controls'].values()):raise ValueError('Incomplete original operator controls')
            gc.collect()
        manifest=json.loads((out/'export-manifest.json').read_text())
        report['weights']=dict(bytes=baseline.stat().st_size,sha256=file_sha(baseline),tensor_audit=manifest['records'])
        if report['weights']['sha256']!=previous['arms'][0]['gguf_sha256'] or len(manifest['records'])!=473:raise ValueError('Changed baseline weights')
        for name,norm,attention in [('baseline',False,False),('precise-norm',True,False),('precise-attention',False,True),('precise-both',True,True)]:
            arm=dict(name=name,precise_norm=norm,precise_attention=attention,cases=[]);report['arms'].append(arm)
            env=dict(os.environ,CRISPEMBED_QWEN35_VISION_PRECISE_NORM=str(int(norm)),CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=str(int(attention)),CRISPEMBED_QWEN35_PRECISE_RMS='0',CRISPEMBED_QWEN35_VISION_PROBES='0')
            for case in report['cases']:
                folder=root/case['name'];ref=folder/'official-delta.gguf'
                cmd=[str(build/'test-qwen35-vision-ops'),str(baseline),str(ref)]
                if name=='baseline':cmd.append('--gate-controls')
                result=subprocess.run(cmd,capture_output=True,text=True,env=env)
                log=result.stdout+result.stderr;(folder/(name+'-vision.log')).write_text(log);result.check_returncode()
                entry=dict(name=case['name'],vision=vision_records(log));arm['cases'].append(entry)
                result=subprocess.run([str(build/'test-qwen35-quant'),str(baseline),str(folder/'input.png'),str(ref),'','1','--prefill-only'],capture_output=True,text=True,env=env)
                log=result.stdout+result.stderr;(folder/(name+'-prefill.log')).write_text(log);result.check_returncode()
                entry['prefill']=prefill_records(log)
                if name=='baseline':
                    old=next(c for c in previous['arms'][0]['cases'] if c['name']==case['name'])
                    entry['prior_scalar_prefill_metrics_exact']=entry['prefill']['metrics']==old['metrics']
                entry['vision_merger_matches_composed_prefill']=next(m['max_abs'] for m in entry['vision'] if m['mode']=='composed' and m['stage']=='prefill.vision.merger')==next(m['max_abs'] for m in entry['prefill']['metrics'] if m['stage']=='prefill.vision.merger')
                if not entry['vision_merger_matches_composed_prefill']:raise ValueError('Vision diagnostic differs from native composed inference')
                print(name,case['name'],'qualified prefill failures',entry['prefill']['qualified_failed_stages'],flush=True)
        report['result']='completed unchanged-gate diagnostic sweep; see every measured failure before promotion'
    finally:
        (out/'vision-operators.json').write_text(json.dumps(report,indent=2)+'\n')
