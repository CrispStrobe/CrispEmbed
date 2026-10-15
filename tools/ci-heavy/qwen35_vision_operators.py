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


def stage_records(log, prefix="VISION_METRIC"):
    lines = log.splitlines()
    records = []
    for i, line in enumerate(lines):
        m = re.match(r'^' + prefix + r' (\S+) (\S+) ' + METRIC + '$', line)
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
    return records


def vision_records(log, blocks=False):
    records = stage_records(log)
    label, count = ('VISION_BLOCKS', 266) if blocks else ('VISION_OPS', 178)
    status = re.search(r'^' + label + r' finite=1 metrics=' + str(count) + r' qualified_failures=(\d+) controls=1 observer=51$', log, re.M)
    if len(records) != count or not status or sum(not m['qualified_pass'] for m in records) != int(status[1]):
        raise ValueError('Incomplete finite vision operator diagnostics')
    expected = {'composed': 111, 'block-input': 108, 'suffix-input-0': 16, 'suffix-input-4': 12, 'suffix-input-8': 8, 'suffix-input-10': 6, 'suffix-input-11': 5} if blocks else {'composed': 77, 'same-input': 101}
    if blocks and 'BLOCK_CONTROL_COMPLETE pass=1 blocks=12' not in log: raise ValueError('Native block replay does not reproduce production outputs')
    if Counter(m['mode'] for m in records) != expected:
        raise ValueError('Missing composed or same-input boundaries')
    return records


def block_controls(log):
    records = [dict(name=m[0], passed=bool(int(m[1])), max_abs=float(m[2]), native_elements=int(m[3]), expected_elements=int(m[4]))
               for m in re.findall(r'^BLOCK_CONTROL (\S+) pass=(\d+) max_abs=(\S+) n=(\d+)/(\d+)$', log, re.M)]
    if len(records)!=113 or any(not m['passed'] or m['max_abs']!=0 or m['native_elements']<1 or m['native_elements']!=m['expected_elements'] for m in records):
        raise ValueError('Incomplete or unequal native block/identity/residual controls')
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


def run(repo, build, baseline, local, out, module, anchor_image, anchor_npz, blocks=False, stats=False, cascade=False):
    import gguf
    import numpy as np
    from PIL import Image
    capture = module(repo / 'tools/dump_qwen35_ocr_page_reference.py', 'qwen35_operator_reference')
    frozen = json.loads((repo / 'tests/results/qwen35-ocr/page-parity.json').read_text())
    previous = json.loads((repo / 'tests/results/qwen35-ocr/prefill-isolation.json').read_text())
    if blocks and stats: raise ValueError('Choose one bounded audit')
    report_name = 'cascade-norm' if cascade else 'norm-statistics' if stats else 'vision-blocks' if blocks else 'vision-operators'
    root = out / report_name
    root.mkdir(exist_ok=True)
    report = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip(),
        block_replay=blocks, normalization_statistics=stats, model=frozen['model'], revision=frozen['revision'], reference='Original Transformers5.5.4 F32 eager modules and hooks',
        scope='Vision stage traces, test-only same-input operators and native composed prefill; no full-output/latency/corpus claim',
        comparator='crispembed_diff::Ref + unchanged qwen35_compare; all raw failures retained', cases=[], arms=[])
    try:
        import hashlib
        source_hash = hashlib.sha256((repo / 'src/qwen35_vision.cpp').read_bytes()).hexdigest()
        if source_hash not in ('c0ec7611af79293e8f5dc6bcc4b695b285cdff87b61d9016bfe3773f0af5a257', 'e41acf49f15d05e55dc2faf158e82b95e3d73c72a968a646a875b569991357db'): raise ValueError('Production vision source changed during audit')
        commands = json.loads((build / 'compile_commands.json').read_text())
        production = [c['command'] for c in commands if c['file'].endswith('/qwen35_vision.cpp')]
        audit = [c['command'] for c in commands if c['file'].endswith('/test_qwen35_vision_ops.cpp')]
        if not production or any('CRISPEMBED_QWEN35_VISION_OPERATOR_AUDIT' in c for c in production) or not any('CRISPEMBED_QWEN35_VISION_OPERATOR_AUDIT=1' in c for c in audit):
            raise ValueError('Test translation-unit isolation missing')
        report['production_source_sha256'] = source_hash
        report['cascade_runtime_experiment'] = cascade
        report['production_source_scope'] = 'original' if source_hash.startswith('c0ec7611') else 'gated four-lane cascade; baseline requires unchanged measurements'
        if blocks or stats:
            block_commands = [c['command'] for c in commands if c['file'].endswith('/test_qwen35_norm_statistics.cpp' if stats else '/test_qwen35_vision_blocks.cpp')]
            if not block_commands or any('CRISPEMBED_QWEN35_VISION_OPERATOR_AUDIT' in c for c in block_commands):
                raise ValueError('Block replay must link the ordinary production implementation')
        report['test_operator_compile_isolation'] = ('linked production implementation without test macros; test-only identity patch fixtures' if blocks or stats else 'hash-guarded production source; replay compiled only in separate validation translation unit')
        if blocks:
            report['block_replay_limits'] = 'Original block inputs enter through an exactly checked test-only identity frontend. Rotary geometry stays native and is measured against original geometry. Residuals use public native projections and float additions, checked against native block outputs. No production source injection or private implementation copy.'
        report['compact_anchor'] = capture.dump_reference(local, anchor_image, root / 'compact-anchor', max_new_tokens=12, anchor=anchor_npz)
        gc.collect()
        report['vision_anchor'] = capture.dump_reference(local, anchor_image, root / 'vision-anchor', max_new_tokens=12, vision_probes=True, norm_statistics=stats)
        reader = gguf.GGUFReader(str(root / 'vision-anchor/official-delta.gguf'))
        new_names = {'input.decoded_length', 'prefill.vision.merger.input'}
        fields = ('norm1','qkv','pooled','attention','norm2','up','activation','down')
        for i in range(12):
            new_names.update(f'prefill.vision.probe.{i}.{f}' for f in ('input','norm2_input'))
            if i not in (5,11): new_names.update(f'prefill.vision.probe.{i}.{f}' for f in fields)
        if stats:
            for i in range(12):
                new_names.update(f'prefill.vision.probe.{i}.{n}.{s}' for n in ('norm1','norm2') for s in ('mean','rstd'))
            new_names.update('prefill.vision.merger.norm.'+s for s in ('mean','rstd'))
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
            case['reference']=capture.dump_reference(local,path,folder,max_new_tokens=1,prefill_controls=True,vision_probes=True,norm_statistics=stats,
                native_geometry=(build/'test-qwen35-norm-statistics',baseline) if stats else None)
            if any(len(rows)!=(267 if stats else 217) for rows in case['reference']['cpu_controls'].values()):raise ValueError('Incomplete original operator controls')
            gc.collect()
        manifest=json.loads((out/'export-manifest.json').read_text())
        report['weights']=dict(bytes=baseline.stat().st_size,sha256=file_sha(baseline),tensor_audit=manifest['records'])
        if report['weights']['sha256']!=previous['arms'][0]['gguf_sha256'] or len(manifest['records'])!=473:raise ValueError('Changed baseline weights')
        if stats:
            module(repo/'tools/ci-heavy/qwen35_norm_statistics.py','qwen35_norm_statistics').run_cases(repo,build,baseline,root,out,report,cascade=cascade)
            return
        recipes = [('baseline',False,False),('precise-both',True,True)] if blocks else [('baseline',False,False),('precise-norm',True,False),('precise-attention',False,True),('precise-both',True,True)]
        for name,norm,attention in recipes:
            arm=dict(name=name,precise_norm=norm,precise_attention=attention,cases=[]);report['arms'].append(arm)
            env=dict(os.environ,CRISPEMBED_QWEN35_VISION_PRECISE_NORM=str(int(norm)),CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=str(int(attention)),CRISPEMBED_QWEN35_PRECISE_RMS='0',CRISPEMBED_QWEN35_VISION_PROBES='0')
            for case in report['cases']:
                folder=root/case['name'];ref=folder/'official-delta.gguf'
                cmd=[str(build/('test-qwen35-vision-blocks' if blocks else 'test-qwen35-vision-ops')),str(baseline),str(ref)]
                if name=='baseline' and not blocks:cmd.append('--gate-controls')
                result=subprocess.run(cmd,capture_output=True,text=True,env=env)
                log=result.stdout+result.stderr;(folder/(name+'-vision.log')).write_text(log)
                entry=dict(name=case['name'],vision=vision_records(log, blocks=blocks),vision_exit_code=result.returncode);arm['cases'].append(entry)
                if blocks: entry['block_controls'] = block_controls(log)
                expected_exit=int(any(not m['qualified_pass'] for m in entry['vision']))
                if result.returncode != expected_exit:raise ValueError('Vision exit status disagrees with shared comparator verdicts')
                result=subprocess.run([str(build/'test-qwen35-quant'),str(baseline),str(folder/'input.png'),str(ref),'','1','--prefill-only'],capture_output=True,text=True,env=env)
                log=result.stdout+result.stderr;(folder/(name+'-prefill.log')).write_text(log);result.check_returncode()
                entry['prefill']=prefill_records(log)
                if name=='baseline':
                    old=next(c for c in previous['arms'][0]['cases'] if c['name']==case['name'])
                    entry['prior_scalar_prefill_metrics_exact']=entry['prefill']['metrics']==old['metrics']
                entry['vision_merger_matches_composed_prefill']=next(m['max_abs'] for m in entry['vision'] if m['mode']=='composed' and m['stage']=='prefill.vision.merger')==next(m['max_abs'] for m in entry['prefill']['metrics'] if m['stage']=='prefill.vision.merger')
                if not entry['vision_merger_matches_composed_prefill']:raise ValueError('Vision diagnostic differs from native composed inference')
                if blocks:
                    prior_ops = json.loads((repo/'tests/results/qwen35-ocr/vision-operators.json').read_text())
                    prior_arm = next(a for a in prior_ops['arms'] if a['name']==name)
                    prior_case = next(c for c in prior_arm['cases'] if c['name']==case['name'])
                    entry['prior_operator_prefill_metrics_exact'] = entry['prefill']['metrics']==prior_case['prefill']['metrics']
                (out/(report_name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
                print(name,case['name'],'qualified prefill failures',entry['prefill']['qualified_failed_stages'],flush=True)
        report['qualified_failure_count']=sum(
            sum(not m['qualified_pass'] for m in case['vision']) + len(case['prefill']['qualified_failed_stages'])
            for arm in report['arms'] for case in arm['cases'])
        report['result']='completed unchanged-gate diagnostic sweep; see every measured failure before promotion'
        if report['qualified_failure_count']:
            raise AssertionError(f"Completed sweep retains {report['qualified_failure_count']} failed stage comparisons")
    finally:
        (out/(report_name+'.json')).write_text(json.dumps(report,indent=2)+'\n')
