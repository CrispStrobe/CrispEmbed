"""Original LayerNorm mean/rstd and native-geometry controls; shared diff verdicts."""
import json
import os
import re
import subprocess
from collections import Counter
from qwen35_vision_operators import stage_records, prefill_records


def records(log):
    rows=stage_records(log,'NORM_METRIC')
    status=re.search(r'^NORM_AUDIT finite=1 metrics=762 qualified_failures=(\d+) controls=(\d+) exact_controls=(\d+)$',log,re.M)
    expected={name:75 for name in ('two-pass-f32','two-pass-f64','welford-4','welford-8','welford-16','original-moments-fused','original-moments-split','welford-4-split','welford-8-split','welford-16-split')}
    expected['python-native-geometry']=12
    if len(rows)!=762 or not status or Counter(m['mode'] for m in rows)!=expected or sum(not m['qualified_pass'] for m in rows)!=int(status[1]):
        raise ValueError('Incomplete original-statistics diagnostics')
    controls=[dict(mode=m[0],stage=m[1],passed=bool(int(m[2])),max_abs=float(m[3]),native_elements=int(m[4]),expected_elements=int(m[5]))
        for m in re.findall(r'^NORM_CONTROL (\S+) (\S+) pass=(\d+) max_abs=(\S+) n=(\d+)/(\d+)$',log,re.M)]
    delivered=[dict(mode=m[0],stage=m[1],passed=bool(int(m[2]))) for m in re.findall(r'^NORM_INPUT (\S+) (\S+) pass=(\d+)$',log,re.M)]
    if len(controls)!=50 or len(delivered)!=50 or sum(m['passed'] for m in controls)!=int(status[3]):
        raise ValueError('Missing actual native LayerNorm controls')
    for c,d in zip(controls,delivered):
        if (c['mode'],c['stage'])!=(d['mode'],d['stage']) or c['native_elements']!=c['expected_elements'] or c['native_elements']<1:
            raise ValueError('Misaligned original/native statistics controls')
        if c['passed']!=(d['passed'] and c['max_abs']==0):raise ValueError('Native equality verdict mismatch')
    if bool(int(status[2]))!=all(c['passed'] and d['passed'] for c,d in zip(controls,delivered)):
        raise ValueError('Gate/native equality controls failed')
    return dict(metrics=rows,native_controls=controls,input_controls=delivered,controls_passed=bool(int(status[2])))


def run_cases(repo,build,baseline,root,out,report):
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
        env=dict(os.environ,CRISPEMBED_QWEN35_VISION_PRECISE_NORM='0',CRISPEMBED_QWEN35_VISION_PRECISE_ATTN='0',CRISPEMBED_QWEN35_PRECISE_RMS='0')
        result=subprocess.run([str(build/'test-qwen35-norm-statistics'),str(baseline),str(ref)],capture_output=True,text=True,env=env)
        log=result.stdout+result.stderr;(folder/'statistics.log').write_text(log)
        entry=dict(name=case['name'],statistics=records(log),statistics_exit_code=result.returncode);arm['cases'].append(entry)
        expected=int(not entry['statistics']['controls_passed'] or any(not m['qualified_pass'] for m in entry['statistics']['metrics']))
        if result.returncode!=expected:raise ValueError('Statistics exit disagrees with unchanged comparisons/controls')
        result=subprocess.run([str(build/'test-qwen35-quant'),str(baseline),str(folder/'input.png'),str(ref),'','1','--prefill-only'],capture_output=True,text=True,env=env)
        log=result.stdout+result.stderr;(folder/'baseline-prefill.log').write_text(log);result.check_returncode()
        entry['prefill']=prefill_records(log)
        old=next(c for c in previous['arms'][0]['cases'] if c['name']==case['name'])
        entry['prior_baseline_prefill_metrics_exact']=entry['prefill']['metrics']==old['prefill']['metrics']
        (out/'norm-statistics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(case['name'],'native controls',entry['statistics']['controls_passed'],'qualified prefill failures',entry['prefill']['qualified_failed_stages'],flush=True)
    report['qualified_failure_count']=sum(sum(not m['qualified_pass'] for m in c['statistics']['metrics'])+len(c['prefill']['qualified_failed_stages']) for c in arm['cases'])
    report['failed_native_controls']=sum(not c['passed'] for case in arm['cases'] for c in case['statistics']['native_controls'])
    report['result']='completed unchanged-gate original-statistics/geometry diagnostic; native reconstruction controls govern interpretation'
    if report['qualified_failure_count'] or report['failed_native_controls']:
        raise AssertionError(f"Completed statistics audit retains {report['qualified_failure_count']} qualified failures and {report['failed_native_controls']} failed native controls")
