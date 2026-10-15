"""Original LayerNorm mean/rstd and native-geometry controls; shared diff verdicts."""
import json
import os
import re
import subprocess
from collections import Counter
from qwen35_vision_operators import stage_records, prefill_records


def records(log,cascade=False):
    control_count=75 if cascade else 50
    rows=stage_records(log,'NORM_METRIC')
    status=re.search(r'^NORM_AUDIT finite=1 metrics=837 qualified_failures=(\d+) controls=(\d+) exact_controls=(\d+)$',log,re.M)
    expected={name:75 for name in ('two-pass-f32','two-pass-f64','welford-4','welford-8','welford-16','original-moments-fused','original-moments-split','welford-4-split','welford-8-split','welford-16-split','two-pass-f32-fused-variance')}
    expected['python-native-geometry']=12
    if len(rows)!=837 or not status or Counter(m['mode'] for m in rows)!=expected or sum(not m['qualified_pass'] for m in rows)!=int(status[1]):
        raise ValueError('Incomplete original-statistics diagnostics')
    controls=[dict(mode=m[0],stage=m[1],passed=bool(int(m[2])),max_abs=float(m[3]),native_elements=int(m[4]),expected_elements=int(m[5]))
        for m in re.findall(r'^NORM_CONTROL (\S+) (\S+) pass=(\d+) max_abs=(\S+) n=(\d+)/(\d+)$',log,re.M)]
    delivered=[dict(mode=m[0],stage=m[1],passed=bool(int(m[2]))) for m in re.findall(r'^NORM_INPUT (\S+) (\S+) pass=(\d+)$',log,re.M)]
    if len(controls)!=control_count or len(delivered)!=control_count or sum(m['passed'] for m in controls)!=int(status[3]):
        raise ValueError('Missing actual native LayerNorm controls')
    for c,d in zip(controls,delivered):
        if (c['mode'],c['stage'])!=(d['mode'],d['stage']) or c['native_elements']!=c['expected_elements'] or c['native_elements']<1:
            raise ValueError('Misaligned original/native statistics controls')
        if c['passed']!=(d['passed'] and c['max_abs']==0):raise ValueError('Native equality verdict mismatch')
    if bool(int(status[2]))!=all(c['passed'] and d['passed'] for c,d in zip(controls,delivered)):
        raise ValueError('Gate/native equality controls failed')
    contrast=[dict(stage=m[0],split_max_abs=float(m[1]),fused_max_abs=float(m[2]),elements=int(m[3])) for m in re.findall(r'^VARIANCE_CONTRAST (\S+) split_max_abs=(\S+) fused_max_abs=(\S+) n=(\d+)$',log,re.M)]
    if len(contrast)!=25:raise ValueError('Missing fused/split variance contrasts')
    return dict(metrics=rows,native_controls=controls,input_controls=delivered,variance_contrasts=contrast,controls_passed=bool(int(status[2])))


def run_cases(repo,build,baseline,root,out,report,cascade=False):
    report_file=out/("cascade-norm.json" if cascade else "norm-statistics.json")
    previous=json.loads((repo/'tests/results/qwen35-ocr/vision-operators.json').read_text())
    report['scope']='Original native_layer_norm mean/rstd, native reconstruction controls, scalar-lane reduction candidates, original-Python/native-geometry controls, and ordinary composed prefill; no runtime promotion'
    report['tested_lane_widths']=[4,8,16]
    report['reference_kernel_sources']={
        'revision':'449b1768410104d3ed79d3bcfe4ba1d65c7f22c0',
        'moments':'https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/moments_utils.h',
        'layer_norm':'https://github.com/pytorch/pytorch/blob/449b1768410104d3ed79d3bcfe4ba1d65c7f22c0/aten/src/ATen/native/cpu/layer_norm_kernel.cpp'}
    arm=dict(name='original-input-statistics',cases=[]);report['arms'].append(arm)
    for case in report['cases']:
        folder=root/case['name'];ref=folder/'official-delta.gguf'
        env=dict(os.environ,CRISPEMBED_QWEN35_VISION_PRECISE_NORM='0',CRISPEMBED_QWEN35_VISION_PRECISE_ATTN='0',CRISPEMBED_QWEN35_PRECISE_RMS='0',CRISPEMBED_QWEN35_VISION_CASCADE_NORM='0')
        result=subprocess.run([str(build/'test-qwen35-norm-statistics'),str(baseline),str(ref),*(["--cascade"] if cascade else [])],capture_output=True,text=True,env=env)
        log=result.stdout+result.stderr;(folder/'statistics.log').write_text(log)
        entry=dict(name=case['name'],statistics=records(log,cascade),statistics_exit_code=result.returncode);arm['cases'].append(entry)
        expected=int(not entry['statistics']['controls_passed'] or any(not m['qualified_pass'] for m in entry['statistics']['metrics']))
        if result.returncode!=expected:raise ValueError('Statistics exit disagrees with unchanged comparisons/controls')
        result=subprocess.run([str(build/'test-qwen35-quant'),str(baseline),str(folder/'input.png'),str(ref),'','1','--prefill-only'],capture_output=True,text=True,env=env)
        log=result.stdout+result.stderr;(folder/'baseline-prefill.log').write_text(log);result.check_returncode()
        entry['prefill']=prefill_records(log)
        old=next(c for c in previous['arms'][0]['cases'] if c['name']==case['name'])
        entry['prior_baseline_prefill_metrics_exact']=entry['prefill']['metrics']==old['prefill']['metrics']
        if cascade:
            entry['runtime_prefill']=[]
            for name,attention in (('cascade',False),('cascade-attention',True)):
                candidate_env=dict(env,CRISPEMBED_QWEN35_VISION_CASCADE_NORM='1',CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=str(int(attention)))
                result=subprocess.run([str(build/'test-qwen35-quant'),str(baseline),str(folder/'input.png'),str(ref),'','1','--prefill-only'],capture_output=True,text=True,env=candidate_env)
                log=result.stdout+result.stderr;(folder/(name+'-prefill.log')).write_text(log);result.check_returncode()
                entry['runtime_prefill'].append(dict(name=name,prefill=prefill_records(log)))
        report_file.write_text(json.dumps(report,indent=2)+'\n')
        print(case['name'],'native controls',entry['statistics']['controls_passed'],'qualified prefill failures',entry['prefill']['qualified_failed_stages'],flush=True)
    report['qualified_failure_count']=sum(sum(not m['qualified_pass'] for m in c['statistics']['metrics'])+len(c['prefill']['qualified_failed_stages']) for c in arm['cases'])
    report['failed_native_controls']=sum(not c['passed'] for case in arm['cases'] for c in case['statistics']['native_controls'])
    report['baseline_reproduction_failures']=sum(not c['prior_baseline_prefill_metrics_exact'] for c in arm['cases'])
    if cascade:
        report['small_decode_controls']=[]
        for name,enabled,attention in (('literal0',False,False),('cascade',True,False),('cascade-attention',True,True)):
            env=dict(os.environ,CRISPEMBED_QWEN35_VISION_CASCADE_NORM=str(int(enabled)),CRISPEMBED_QWEN35_VISION_PRECISE_ATTN=str(int(attention)),CRISPEMBED_QWEN35_VISION_PRECISE_NORM='0')
            result=subprocess.run([str(build/'test-qwen35-quant'),str(baseline),str(out/'reference-input.png'),str(root/'compact-anchor/official-delta.gguf'),'','12'],capture_output=True,text=True,env=env)
            log=result.stdout+result.stderr;(root/(name+'-small-decode.log')).write_text(log)
            matched=re.search(r'^FREE ids_exact=(\d+) text_exact=(\d+) stop=(\d+) mean_confidence=(\S+) bytes=(\d+)$',log,re.M)
            ids=re.search(r'^FREE ids:(.*)$',log,re.M);text=re.search(r'^FREE utf8_hex:(.*)$',log,re.M)
            if not matched or not ids or not text:raise ValueError('Incomplete actual small decode')
            generated=list(map(int,ids[1].split()));decoded=bytes.fromhex(text[1]).decode('utf-8')
            passed=result.returncode==0 and tuple(matched.group(i) for i in (1,2,3))==('1','1','1') and generated==report['compact_anchor']['generated_ids'] and decoded==report['compact_anchor']['decoded']
            report['small_decode_controls'].append(dict(name=name,passed=passed,exit_code=result.returncode,generated_ids=generated,decoded=decoded,stop=int(matched[3]),mean_confidence=float(matched[4]),bytes=int(matched[5])))
        report['qualified_failure_count']+=sum(len(p['prefill']['qualified_failed_stages']) for c in arm['cases'] for p in c['runtime_prefill'])
    all_prefills=[c['prefill'] for c in arm['cases']]+[p['prefill'] for c in arm['cases'] for p in c.get('runtime_prefill',[])]
    report['argmax_failures']=sum(p['selected']!=p['reference'] for p in all_prefills)
    report['result']='completed unchanged-gate original-statistics/geometry diagnostic; native reconstruction controls govern interpretation'
    report_file.write_text(json.dumps(report,indent=2)+'\n')
    if report['qualified_failure_count'] or report['failed_native_controls'] or report['baseline_reproduction_failures'] or report['argmax_failures'] or any(not c['passed'] for c in report.get('small_decode_controls',[])):
        raise AssertionError(f"Completed statistics audit retains {report['qualified_failure_count']} qualified failures and {report['failed_native_controls']} failed native controls")
